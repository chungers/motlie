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
#[path = "../quiet_support.rs"]
mod quiet_support;

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
    quiet: bool,
}

fn parse_args() -> Result<Args> {
    let mut wav_path: Option<PathBuf> = None;
    let mut artifact_root: Option<PathBuf> = None;
    let mut language: Option<String> = None;
    let mut quiet = false;

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
            "--quiet" => quiet = true,
            other => bail!("unknown argument: {other}"),
        }
    }

    Ok(Args {
        wav_path,
        artifact_root,
        language,
        quiet,
    })
}

async fn run(args: Args) -> Result<()> {
    asr_support::log_status(
        args.quiet,
        "=== motlie asr_whisper — typed Whisper batch transcription ===",
    );
    let input = asr_support::open_asr_input(args.wav_path)?;
    asr_support::log_status(
        args.quiet,
        &format!("wav:   {}", asr_support::describe_input(&input.source)),
    );

    let wav_spec = input.reader.spec();
    asr_support::log_status(
        args.quiet,
        &format!(
            "format: {} Hz, {} ch, {:?}, {} bits",
            wav_spec.sample_rate,
            wav_spec.channels,
            wav_spec.sample_format,
            wav_spec.bits_per_sample,
        ),
    );

    let audio = decode_wav_to_whisper_input(input.reader)?;

    let artifact_root = args
        .artifact_root
        .unwrap_or_else(motlie_models::default_artifact_root);
    asr_support::log_status(
        args.quiet,
        &format!("artifacts: {}", artifact_root.display()),
    );
    let _quiet_stderr = quiet_support::QuietStderrGuard::maybe_enable(args.quiet)
        .context("failed to enable quiet stderr mode")?;

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

    asr_support::print_plain_transcript(&update.segments);

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
