//! tts_qwen3_onnx — Qwen3-TTS synthesis with optional voice cloning.
//!
//! Usage:
//!   cargo run -p motlie-models --example tts_qwen3_onnx \
//!     --no-default-features --features model-qwen3-tts-0_6b \
//!     -- --text "Hello from Motlie." --wav /tmp/out.wav

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use motlie_model::typed::{SpeechStream, SpeechSynthesizer};
use motlie_model::{
    ArtifactPolicy, AudioSpec, PcmEncoding, SpeechParams, SpeechRequest, StartOptions,
    VoiceConditioning,
};
use motlie_models::tts::qwen3_tts_12hz_0_6b;

const TARGET_SAMPLE_RATE_HZ: u32 = 24_000;

fn main() -> Result<()> {
    let args = parse_args()?;
    let runtime = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;
    runtime.block_on(run(args))
}

struct Args {
    text: String,
    wav_path: PathBuf,
    artifact_root: Option<PathBuf>,
    reference_audio: Option<PathBuf>,
    reference_text: Option<String>,
}

fn parse_args() -> Result<Args> {
    let mut text = None;
    let mut wav_path = None;
    let mut artifact_root = None;
    let mut reference_audio = None;
    let mut reference_text = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--text" => {
                text = Some(args.next().context("--text requires a value")?);
            }
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
            "--reference-audio" => {
                reference_audio = Some(PathBuf::from(
                    args.next()
                        .context("--reference-audio requires a path argument")?,
                ));
            }
            "--reference-text" => {
                reference_text = Some(args.next().context("--reference-text requires a value")?);
            }
            other => bail!("unknown argument: {other}"),
        }
    }

    Ok(Args {
        text: text.context("--text <value> is required")?,
        wav_path: wav_path.context("--wav <path> is required")?,
        artifact_root,
        reference_audio,
        reference_text,
    })
}

async fn run(args: Args) -> Result<()> {
    println!("=== motlie tts_qwen3_onnx — typed Qwen3-TTS speech synthesis ===");
    println!("wav:  {}", args.wav_path.display());

    let handle = qwen3_tts_12hz_0_6b::start_typed(StartOptions {
        artifact_policy: Some(ArtifactPolicy::LocalOnly {
            root: args
                .artifact_root
                .unwrap_or_else(motlie_models::default_artifact_root),
        }),
        ..Default::default()
    })
    .await
    .context("failed to start typed qwen3-tts bundle")?;

    let conditioning = if let Some(ref_path) = &args.reference_audio {
        println!("reference: {}", ref_path.display());
        let reader = hound::WavReader::open(ref_path)
            .with_context(|| format!("failed to open reference audio: {}", ref_path.display()))?;
        let ref_spec = reader.spec();
        let (pcm, encoding) = decode_wav(reader)?;
        Some(VoiceConditioning::ReferenceAudio {
            audio_spec: AudioSpec {
                sample_rate_hz: ref_spec.sample_rate,
                channels: ref_spec.channels,
                encoding,
                preferred_chunk_bytes: 0,
            },
            pcm,
            reference_text: args.reference_text.clone(),
        })
    } else {
        None
    };

    let mut stream = handle
        .synthesize(SpeechRequest {
            text: args.text,
            params: SpeechParams::default(),
            conditioning,
        })
        .await
        .context("failed to open typed speech stream")?;

    let mut writer = hound::WavWriter::create(
        &args.wav_path,
        hound::WavSpec {
            channels: 1,
            sample_rate: TARGET_SAMPLE_RATE_HZ,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )
    .with_context(|| format!("failed to create wav file `{}`", args.wav_path.display()))?;

    let mut total_samples = 0usize;
    while let Some(chunk) = stream.next_chunk().await.context("next_chunk failed")? {
        total_samples += chunk.samples().len();
        for sample in chunk.into_samples() {
            writer
                .write_sample(sample)
                .context("failed to write wav sample")?;
        }
    }

    writer.finalize().context("failed to finalize wav file")?;
    stream.finish().await.context("finish failed")?;
    handle.shutdown().await.context("shutdown failed")?;

    println!(
        "wrote {} mono f32 samples at {} Hz to {}",
        total_samples,
        TARGET_SAMPLE_RATE_HZ,
        args.wav_path.display()
    );
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
                .context("failed to decode wav samples")?;
            let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
            Ok((bytes, PcmEncoding::S16Le))
        }
        hound::SampleFormat::Float => {
            let samples: Vec<f32> = reader
                .into_samples::<f32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("failed to decode wav samples")?;
            let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
            Ok((bytes, PcmEncoding::F32Le))
        }
    }
}
