//! asr_moonshine — Moonshine ASR slice through the typed streaming API.
//!
//! Usage:
//!   cargo run -p motlie-models --example asr_moonshine \
//!     --no-default-features --features model-moonshine-streaming \
//!     -- --wav path/to/audio.wav

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use motlie_models::asr::moonshine_streaming_en;

#[path = "../asr_support.rs"]
mod asr_support;
#[path = "../audio_support.rs"]
mod audio_support;
#[path = "../bundle_support.rs"]
mod bundle_support;
#[path = "../quiet_support.rs"]
mod quiet_support;
#[path = "../streaming_asr_support.rs"]
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
        "=== motlie asr_moonshine — typed Moonshine streaming ASR ===",
        "failed to start typed Moonshine bundle",
        "failed to open typed Moonshine session",
        "typed Moonshine ingest failed",
        moonshine_streaming_en::start_typed,
    )
    .await
}
