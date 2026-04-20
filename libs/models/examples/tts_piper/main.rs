//! tts_piper — synthesize text and write a `.wav` file.
//!
//! Usage:
//!   cargo run -p motlie-models --example tts_piper \
//!     --no-default-features --features model-piper-en-us-ljspeech-medium \
//!     -- --text "Hello from Motlie." --wav /tmp/out.wav

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use motlie_model::typed::{SpeechStream, SpeechSynthesizer, SynthesisRequest};
use motlie_model::{ArtifactPolicy, SpeechParams, StartOptions};
use motlie_models::tts::piper_en_us_ljspeech_medium;

#[path = "../tts_support.rs"]
mod tts_support;

const TARGET_SAMPLE_RATE_HZ: u32 = 22_050;

fn main() -> Result<()> {
    let args = parse_args()?;
    let runtime = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;
    runtime.block_on(run(args))
}

struct Args {
    text: Option<String>,
    wav_path: Option<PathBuf>,
    artifact_root: Option<PathBuf>,
}

fn parse_args() -> Result<Args> {
    let mut text = None;
    let mut wav_path = None;
    let mut artifact_root = None;

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
            other => bail!("unknown argument: {other}"),
        }
    }

    Ok(Args {
        text,
        wav_path,
        artifact_root,
    })
}

async fn run(args: Args) -> Result<()> {
    let io = tts_support::resolve_text_and_output(args.text, args.wav_path)?;
    tts_support::log_status("=== motlie tts_piper — typed Piper speech synthesis ===");
    match &io.output {
        tts_support::TtsOutput::WavFile(path) => {
            tts_support::log_status(&format!("wav:  {}", path.display()));
        }
        tts_support::TtsOutput::Stdout => {
            tts_support::log_status("wav:  <stdout>");
        }
    }

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

    let mut stream = handle
        .synthesize(SynthesisRequest {
            text: io.text,
            params: SpeechParams::default(),
        })
        .await
        .context("failed to open typed speech stream")?;

    let mut samples = Vec::new();
    while let Some(chunk) = stream.next_chunk().await.context("next_chunk failed")? {
        samples.extend(chunk.into_samples());
    }

    tts_support::write_wav(&io.output, TARGET_SAMPLE_RATE_HZ, &samples)?;
    stream.finish().await.context("finish failed")?;
    handle.shutdown().await.context("shutdown failed")?;

    tts_support::log_status(&format!(
        "wrote {} mono i16 samples at {} Hz to {}",
        samples.len(),
        TARGET_SAMPLE_RATE_HZ,
        match &io.output {
            tts_support::TtsOutput::WavFile(path) => path.display().to_string(),
            tts_support::TtsOutput::Stdout => "<stdout>".into(),
        }
    ));
    Ok(())
}
