//! tts_piper — synthesize text and write a `.wav` file.
//!
//! Usage:
//!   cargo run -p motlie-models --example tts_piper \
//!     --no-default-features --features model-piper-en-us-ljspeech-medium \
//!     -- --text "Hello from Motlie." --wav /tmp/out.wav

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use motlie_model::typed::{SpeechStream, SpeechSynthesizer, SynthesisRequest};
use motlie_model::{ArtifactPolicy, SpeechParams, StartOptions};
use motlie_models::tts::piper_en_us_ljspeech_medium;

#[path = "../bundle_support.rs"]
mod bundle_support;
#[path = "../quiet_support.rs"]
mod quiet_support;
#[path = "../tts_support.rs"]
mod tts_support;

fn main() -> Result<()> {
    let args = parse_args()?;
    let runtime = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;
    runtime.block_on(run(args))
}

struct Args {
    text: Option<String>,
    wav_path: Option<PathBuf>,
    artifact_root: Option<PathBuf>,
    quiet: bool,
}

fn parse_args() -> Result<Args> {
    let mut text = None;
    let mut wav_path = None;
    let mut artifact_root = None;
    let mut quiet = false;

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
            "--quiet" => quiet = true,
            other => bail!("unknown argument: {other}"),
        }
    }

    Ok(Args {
        text,
        wav_path,
        artifact_root,
        quiet,
    })
}

async fn run(args: Args) -> Result<()> {
    let io = tts_support::resolve_text_and_output(args.text, args.wav_path)?;
    tts_support::log_status(
        args.quiet,
        "=== motlie tts_piper — typed Piper speech synthesis ===",
    );
    match &io.output {
        tts_support::TtsOutput::WavFile(path) => {
            tts_support::log_status(args.quiet, &format!("wav:  {}", path.display()));
        }
        tts_support::TtsOutput::Stdout => {
            tts_support::log_status(args.quiet, "wav:  <stdout>");
        }
    }
    let _quiet_stderr = quiet_support::QuietStderrGuard::maybe_enable(args.quiet)
        .context("failed to enable quiet stderr mode")?;

    let handle = piper_en_us_ljspeech_medium::start_typed(StartOptions {
        artifact_policy: Some(ArtifactPolicy::LocalOnly {
            root: args
                .artifact_root
                .unwrap_or_else(motlie_models::default_artifact_root),
        }),
        ..Default::default()
    })
    .await
    .context("failed to start typed Piper bundle")?;

    let output_label = match &io.output {
        tts_support::TtsOutput::WavFile(path) => path.display().to_string(),
        tts_support::TtsOutput::Stdout => "<stdout>".into(),
    };
    let (sample_count, sample_rate_hz) = bundle_support::run_with_shutdown(handle, |handle| {
        Box::pin(async move {
            let mut stream = handle
                .synthesize(SynthesisRequest {
                    text: io.text,
                    params: SpeechParams::default(),
                })
                .await
                .context("failed to open typed speech stream")?;

            let first_chunk = stream
                .next_chunk()
                .await
                .context("next_chunk failed")?
                .context("typed Piper stream produced no audio chunks")?;
            let sample_rate_hz = first_chunk.sample_rate_hz();
            let mut sample_count = 0usize;
            let mut sink = tts_support::WavSink::<i16>::new(&io.output, sample_rate_hz)?;

            let first_samples = first_chunk.into_samples();
            sample_count += first_samples.len();
            sink.write_chunk(&first_samples)?;

            while let Some(chunk) = stream.next_chunk().await.context("next_chunk failed")? {
                let samples = chunk.into_samples();
                sample_count += samples.len();
                sink.write_chunk(&samples)?;
            }

            stream.finish().await.context("finish failed")?;
            sink.finalize()?;
            Ok((sample_count, sample_rate_hz))
        })
    })
    .await?;

    tts_support::log_status(
        args.quiet,
        &format!(
            "wrote {} mono i16 samples at {} Hz to {}",
            sample_count, sample_rate_hz, output_label
        ),
    );
    Ok(())
}
