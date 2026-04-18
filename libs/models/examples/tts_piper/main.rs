//! tts_piper — synthesize text and write a `.wav` file.
//!
//! Usage:
//!   cargo run -p motlie-models --example tts_piper \
//!     --no-default-features --features model-piper-en-us-ljspeech-medium \
//!     -- --text "Hello from Motlie." --wav /tmp/out.wav

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use motlie_model::typed::{SpeechStream, SpeechSynthesizer};
use motlie_model::{ArtifactPolicy, SpeechParams, SpeechRequest, StartOptions};
use motlie_models::tts::piper_en_us_ljspeech_medium;

const TARGET_SAMPLE_RATE_HZ: u32 = 22_050;

fn main() -> Result<()> {
    let args = parse_args()?;
    let runtime = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;
    runtime.block_on(run(args))
}

struct Args {
    text: String,
    wav_path: PathBuf,
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
        text: text.context("--text <value> is required")?,
        wav_path: wav_path.context("--wav <path> is required")?,
        artifact_root,
    })
}

async fn run(args: Args) -> Result<()> {
    println!("=== motlie tts_piper — typed Piper speech synthesis ===");
    println!("wav:  {}", args.wav_path.display());

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
        .synthesize(SpeechRequest {
            text: args.text,
            params: SpeechParams::default(),
            conditioning: None,
        })
        .await
        .context("failed to open typed speech stream")?;

    let mut writer = hound::WavWriter::create(
        &args.wav_path,
        hound::WavSpec {
            channels: 1,
            sample_rate: TARGET_SAMPLE_RATE_HZ,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
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
        "wrote {} mono i16 samples at {} Hz to {}",
        total_samples,
        TARGET_SAMPLE_RATE_HZ,
        args.wav_path.display()
    );
    Ok(())
}
