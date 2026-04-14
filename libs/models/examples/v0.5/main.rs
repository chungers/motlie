//! v0.5 — ASR vertical slice: `.wav` file transcription via the streaming PCM contract.
//!
//! Usage:
//!   cargo run -p motlie-models --example models_v0_5 \
//!     --no-default-features --features model-whisper-base-en \
//!     -- --wav path/to/audio.wav
//!
//! Preconditions:
//!   - The `ggml-base.en.bin` model must be pre-downloaded under the default
//!     artifact root or the path specified by `--artifact-root`.
//!   - The `.wav` file must be a valid PCM audio file (16-bit int or 32-bit float).
//!     The backend normalizes any sample rate and channel count to mono 16 kHz.
//!
//! The example reads the `.wav` file, decodes it to raw PCM, chunks it into
//! the streaming `TranscriptionStream` contract, and prints partial and final
//! transcript segments as they are emitted by the backend.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use motlie_model::{
    ArtifactPolicy, AudioSpec, PcmChunk, PcmEncoding, StartOptions, TranscriptionParams,
};
use motlie_models::asr::AsrModels;

fn main() -> Result<()> {
    let args = parse_args()?;

    let runtime = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;
    runtime.block_on(run(args))
}

struct Args {
    wav_path: PathBuf,
    artifact_root: Option<PathBuf>,
    language: Option<String>,
}

fn parse_args() -> Result<Args> {
    let mut wav_path: Option<PathBuf> = None;
    let mut artifact_root: Option<PathBuf> = None;
    let mut language: Option<String> = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--wav" => {
                wav_path = Some(PathBuf::from(
                    args.next().context("--wav requires a path argument")?,
                ));
            }
            "--artifact-root" => {
                artifact_root = Some(PathBuf::from(
                    args.next()
                        .context("--artifact-root requires a path argument")?,
                ));
            }
            "--language" => {
                language = Some(
                    args.next()
                        .context("--language requires a language code")?,
                );
            }
            other => bail!("unknown argument: {other}"),
        }
    }

    let wav_path = wav_path.context("--wav <path> is required")?;
    Ok(Args {
        wav_path,
        artifact_root,
        language,
    })
}

async fn run(args: Args) -> Result<()> {
    println!("=== motlie v0.5 — ASR vertical slice ===");
    println!("wav:   {}", args.wav_path.display());

    // 1. Read and decode the .wav file
    let reader = hound::WavReader::open(&args.wav_path)
        .with_context(|| format!("failed to open wav file: {}", args.wav_path.display()))?;

    let wav_spec = reader.spec();
    println!(
        "format: {} Hz, {} ch, {:?}, {} bits",
        wav_spec.sample_rate, wav_spec.channels, wav_spec.sample_format, wav_spec.bits_per_sample,
    );

    let (pcm_bytes, encoding) = decode_wav(reader)?;
    let audio_spec = AudioSpec {
        sample_rate_hz: wav_spec.sample_rate,
        channels: wav_spec.channels,
        encoding,
    };

    // 2. Start the curated ASR bundle
    let artifact_root = args
        .artifact_root
        .unwrap_or_else(motlie_models::default_artifact_root);

    println!("artifacts: {}", artifact_root.display());

    let bundle = AsrModels::WhisperBaseEn.bundle();
    let handle = bundle
        .start(StartOptions {
            artifact_policy: Some(ArtifactPolicy::LocalOnly {
                root: artifact_root,
            }),
            ..Default::default()
        })
        .await
        .context("failed to start whisper bundle")?;

    // 3. Open a transcription stream
    let asr = handle
        .transcription()
        .context("whisper bundle should expose transcription capability")?;

    let mut stream = asr
        .open_stream(
            audio_spec,
            TranscriptionParams {
                language: args.language,
                emit_partials: true,
            },
        )
        .await
        .context("failed to open transcription stream")?;

    // 4. Feed PCM chunks (16KB per chunk ≈ 0.5s at 16kHz mono S16)
    let chunk_size = 16_000;
    let total_bytes = pcm_bytes.len();
    let mut offset = 0;
    let mut sequence = 0u64;

    println!("\n--- transcribing {} bytes of audio ---\n", total_bytes);

    while offset < total_bytes {
        let end = (offset + chunk_size).min(total_bytes);
        let is_last = end >= total_bytes;

        let chunk = PcmChunk {
            data: pcm_bytes[offset..end].to_vec(),
            sequence,
            end_of_stream: is_last,
        };

        if let Some(update) = stream
            .push_chunk(chunk)
            .await
            .context("push_chunk failed")?
        {
            for segment in &update.segments {
                let marker = if segment.final_segment {
                    "[final]"
                } else {
                    "[partial]"
                };
                println!(
                    "  {marker} [{:.1}s - {:.1}s] {}",
                    segment.start_ms as f64 / 1000.0,
                    segment.end_ms as f64 / 1000.0,
                    segment.text.trim()
                );
            }
        }

        offset = end;
        sequence += 1;
    }

    // 5. Flush remaining audio
    let final_update = stream
        .finish()
        .await
        .context("finish() failed")?;

    if !final_update.segments.is_empty() {
        println!("\n--- final flush ---\n");
        for segment in &final_update.segments {
            println!(
                "  [final] [{:.1}s - {:.1}s] {}",
                segment.start_ms as f64 / 1000.0,
                segment.end_ms as f64 / 1000.0,
                segment.text.trim()
            );
        }
    }

    println!("\n--- done ---");

    handle
        .shutdown()
        .await
        .context("shutdown failed")?;

    Ok(())
}

fn decode_wav(reader: hound::WavReader<std::io::BufReader<std::fs::File>>) -> Result<(Vec<u8>, PcmEncoding)> {
    let spec = reader.spec();
    match spec.sample_format {
        hound::SampleFormat::Int => {
            let samples: Vec<i16> = reader
                .into_samples::<i16>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("failed to decode wav samples")?;

            let bytes: Vec<u8> = samples
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect();

            Ok((bytes, PcmEncoding::S16Le))
        }
        hound::SampleFormat::Float => {
            let samples: Vec<f32> = reader
                .into_samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("failed to decode wav samples")?;

            let bytes: Vec<u8> = samples
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect();

            Ok((bytes, PcmEncoding::F32Le))
        }
    }
}
