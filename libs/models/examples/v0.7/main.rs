//! v0.7 — Moonshine secondary ASR slice for chunked `.wav` transcription.
//!
//! Usage:
//!   cargo run -p motlie-models --example models_v0_7 \
//!     --no-default-features --features model-moonshine-streaming \
//!     -- --wav path/to/audio.wav

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
}

fn parse_args() -> Result<Args> {
    let mut wav_path = None;
    let mut artifact_root = None;

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
            other => bail!("unknown argument: {other}"),
        }
    }

    Ok(Args {
        wav_path: wav_path.context("--wav <path> is required")?,
        artifact_root,
    })
}

async fn run(args: Args) -> Result<()> {
    println!("=== motlie v0.7 — Moonshine secondary ASR ===");
    println!("wav: {}", args.wav_path.display());

    let reader = hound::WavReader::open(&args.wav_path)
        .with_context(|| format!("failed to open wav file: {}", args.wav_path.display()))?;
    let wav_spec = reader.spec();
    let (pcm_bytes, encoding) = decode_wav(reader)?;

    let bundle = AsrModels::MoonshineStreamingEn.bundle();
    let handle = bundle
        .start(StartOptions {
            artifact_policy: Some(ArtifactPolicy::LocalOnly {
                root: args
                    .artifact_root
                    .unwrap_or_else(motlie_models::default_artifact_root),
            }),
            ..Default::default()
        })
        .await
        .context("failed to start Moonshine bundle")?;

    let model = handle
        .transcription()
        .context("bundle should expose transcription capability")?;
    let mut stream = model
        .open_stream(
            AudioSpec {
                sample_rate_hz: wav_spec.sample_rate,
                channels: wav_spec.channels,
                encoding,
            },
            TranscriptionParams {
                language: Some("en".into()),
                emit_partials: false,
            },
        )
        .await
        .context("failed to open transcription stream")?;

    let chunk_size = 6_400;
    let mut offset = 0;
    let mut sequence = 0_u64;

    while offset < pcm_bytes.len() {
        let end = (offset + chunk_size).min(pcm_bytes.len());
        stream
            .push_chunk(PcmChunk {
                data: pcm_bytes[offset..end].to_vec(),
                sequence,
                end_of_stream: end == pcm_bytes.len(),
            })
            .await
            .context("push_chunk failed")?;
        offset = end;
        sequence += 1;
    }

    let final_update = stream.finish().await.context("finish failed")?;
    if final_update.segments.is_empty() {
        println!("transcript: <empty>");
    } else {
        for segment in final_update.segments {
            println!(
                "[final] [{:.2}s - {:.2}s] {}",
                segment.start_ms as f64 / 1000.0,
                segment.end_ms as f64 / 1000.0,
                segment.text
            );
        }
    }

    handle.shutdown().await.context("shutdown failed")?;
    Ok(())
}

fn decode_wav(
    reader: hound::WavReader<std::io::BufReader<std::fs::File>>,
) -> Result<(Vec<u8>, PcmEncoding)> {
    let spec = reader.spec();
    match spec.sample_format {
        hound::SampleFormat::Int => {
            let samples: Vec<i16> = reader
                .into_samples::<i16>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("failed to decode integer wav samples")?;
            Ok((
                samples
                    .iter()
                    .flat_map(|sample| sample.to_le_bytes())
                    .collect(),
                PcmEncoding::S16Le,
            ))
        }
        hound::SampleFormat::Float => {
            let samples: Vec<f32> = reader
                .into_samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("failed to decode float wav samples")?;
            Ok((
                samples
                    .iter()
                    .flat_map(|sample| sample.to_le_bytes())
                    .collect(),
                PcmEncoding::F32Le,
            ))
        }
    }
}
