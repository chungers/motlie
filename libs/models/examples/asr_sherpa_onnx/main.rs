//! asr_sherpa_onnx — streaming Zipformer ASR via the typed streaming API.
//!
//! Usage:
//!   cargo run -p motlie-models --example asr_sherpa_onnx \
//!     --no-default-features --features model-sherpa-onnx-streaming \
//!     -- --wav path/to/audio.wav

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use motlie_models::asr::sherpa_onnx_streaming_en;

#[path = "../support/asr.rs"]
mod asr_support;
#[path = "../support/audio.rs"]
mod audio_support;
#[path = "../support/bundle.rs"]
mod bundle_support;
#[path = "../support/quiet.rs"]
mod quiet_support;
#[path = "../support/streaming_asr.rs"]
mod streaming_asr_support;

fn main() -> Result<()> {
    let args = parse_args()?;
    let runtime = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;
    runtime.block_on(run(args))
}

struct Args {
    wav_path: Option<PathBuf>,
    artifact_root: Option<PathBuf>,
    quiet: bool,
    partials: bool,
}

fn parse_args() -> Result<Args> {
    let mut wav_path = None;
    let mut artifact_root = None;
    let mut quiet = false;
    let mut partials = false;

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
            "--quiet" => quiet = true,
            "--partials" => partials = true,
            other => bail!("unknown argument: {other}"),
        }
    }

    Ok(Args {
        wav_path,
        artifact_root,
        quiet,
        partials,
    })
}

async fn run(args: Args) -> Result<()> {
    streaming_asr_support::run_streaming_asr(
        streaming_asr_support::StreamingAsrArgs {
            wav_path: args.wav_path,
            artifact_root: args.artifact_root,
            quiet: args.quiet,
            partials: args.partials,
        },
        "=== motlie asr_sherpa_onnx — typed sherpa-onnx streaming ASR ===",
        "failed to start typed sherpa-onnx bundle",
        "failed to open typed sherpa-onnx session",
        "typed sherpa ingest failed",
        sherpa_onnx_streaming_en::start_typed,
    )
    .await
}
