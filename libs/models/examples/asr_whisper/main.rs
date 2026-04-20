//! asr_whisper — Whisper `.wav` transcription through the typed batch API.
//!
//! Usage:
//!   cargo run -p motlie-models --example asr_whisper \
//!     --no-default-features --features model-whisper-base-en \
//!     -- --wav path/to/audio.wav

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use motlie_model::typed::{AudioBuf, BatchTranscriber, Mono};
use motlie_model::{ArtifactPolicy, StartOptions, TranscriptionParams};
use motlie_models::asr::whisper_base_en;

#[path = "../asr_support.rs"]
mod asr_support;
#[path = "../audio_support.rs"]
mod audio_support;

const TARGET_SAMPLE_RATE_HZ: u32 = 16_000;

fn main() -> Result<()> {
    let args = parse_args()?;

    let runtime = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;
    runtime.block_on(run(args))
}

struct Args {
    wav_path: Option<PathBuf>,
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
                language = Some(args.next().context("--language requires a language code")?);
            }
            other => bail!("unknown argument: {other}"),
        }
    }

    Ok(Args {
        wav_path,
        artifact_root,
        language,
    })
}

async fn run(args: Args) -> Result<()> {
    asr_support::log_status("=== motlie asr_whisper — typed Whisper batch transcription ===");
    let input = asr_support::open_asr_input(args.wav_path)?;
    asr_support::log_status(&format!(
        "wav:   {}",
        asr_support::describe_input(&input.source)
    ));

    let wav_spec = input.reader.spec();
    asr_support::log_status(&format!(
        "format: {} Hz, {} ch, {:?}, {} bits",
        wav_spec.sample_rate, wav_spec.channels, wav_spec.sample_format, wav_spec.bits_per_sample,
    ));

    let audio = decode_wav_to_whisper_input(input.reader)?;

    let artifact_root = args
        .artifact_root
        .unwrap_or_else(motlie_models::default_artifact_root);
    asr_support::log_status(&format!("artifacts: {}", artifact_root.display()));

    let handle = whisper_base_en::start_typed(StartOptions {
        artifact_policy: Some(ArtifactPolicy::LocalOnly {
            root: artifact_root,
        }),
        ..Default::default()
    })
    .await
    .context("failed to start typed whisper bundle")?;

    let update = handle
        .transcribe(
            audio,
            TranscriptionParams {
                language: args.language,
                emit_partials: false,
            },
        )
        .await
        .context("typed whisper transcription failed")?;

    for segment in &update.segments {
        println!(
            "[final] [{:.1}s - {:.1}s] {}",
            segment.start_ms as f64 / 1000.0,
            segment.end_ms as f64 / 1000.0,
            segment.text.trim()
        );
    }

    handle.shutdown().await.context("shutdown failed")?;

    Ok(())
}

fn decode_wav_to_whisper_input<R: std::io::Read>(
    reader: hound::WavReader<R>,
) -> Result<AudioBuf<f32, TARGET_SAMPLE_RATE_HZ, Mono>> {
    let (spec, samples) = audio_support::decode_wav_to_f32(reader)?;
    let mono = audio_support::downmix_to_mono(&samples, spec.channels);
    let resampled =
        audio_support::resample_linear_f32(&mono, spec.sample_rate, TARGET_SAMPLE_RATE_HZ);
    Ok(AudioBuf::new(resampled))
}
