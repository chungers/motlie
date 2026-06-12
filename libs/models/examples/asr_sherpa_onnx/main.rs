//! asr_sherpa_onnx — streaming Zipformer ASR via the typed streaming API.
//!
//! Usage:
//!   cargo run -p motlie-models --example asr_sherpa_onnx \
//!     --no-default-features --features model-sherpa-onnx-streaming \
//!     -- --wav path/to/audio.wav [--model=zipformer-en|kroko-2025]

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use motlie_models::asr::{sherpa_onnx_streaming_en, sherpa_onnx_streaming_en_kroko_2025};

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SherpaExampleModel {
    ZipformerEn,
    Kroko2025,
}

impl SherpaExampleModel {
    fn parse(raw: &str) -> Result<Self> {
        match raw {
            "zipformer-en" | "sherpa-2023" | "sherpa-onnx/streaming_zipformer_en" => {
                Ok(Self::ZipformerEn)
            }
            "kroko-2025"
            | "sherpa-zipformer-kroko-2025"
            | "sherpa-onnx/streaming_zipformer_en_kroko_2025" => Ok(Self::Kroko2025),
            other => bail!("unknown --model `{other}`; expected zipformer-en or kroko-2025"),
        }
    }

    fn banner(self) -> &'static str {
        match self {
            Self::ZipformerEn => "=== motlie asr_sherpa_onnx — typed sherpa-onnx streaming ASR ===",
            Self::Kroko2025 => {
                "=== motlie asr_sherpa_onnx — typed sherpa-onnx Kroko 2025 streaming ASR ==="
            }
        }
    }
}

struct Args {
    wav_path: Option<PathBuf>,
    artifact_root: Option<PathBuf>,
    quiet: bool,
    partials: bool,
    model: SherpaExampleModel,
}

fn parse_args() -> Result<Args> {
    let mut wav_path = None;
    let mut artifact_root = None;
    let mut quiet = false;
    let mut partials = false;
    let mut model = SherpaExampleModel::ZipformerEn;

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
            "--model" => {
                model = SherpaExampleModel::parse(
                    &args.next().context("--model requires a model name")?,
                )?;
            }
            "--quiet" => quiet = true,
            "--partials" => partials = true,
            other if other.starts_with("--model=") => {
                model = SherpaExampleModel::parse(
                    other
                        .split_once('=')
                        .map(|(_, value)| value)
                        .unwrap_or_default(),
                )?;
            }
            other => bail!("unknown argument: {other}"),
        }
    }

    Ok(Args {
        wav_path,
        artifact_root,
        quiet,
        partials,
        model,
    })
}

async fn run(args: Args) -> Result<()> {
    let model = args.model;
    let streaming_args = streaming_asr_support::StreamingAsrArgs {
        wav_path: args.wav_path,
        artifact_root: args.artifact_root,
        quiet: args.quiet,
        partials: args.partials,
    };

    match model {
        SherpaExampleModel::ZipformerEn => {
            streaming_asr_support::run_streaming_asr(
                streaming_args,
                model.banner(),
                "failed to start typed sherpa-onnx bundle",
                "failed to open typed sherpa-onnx session",
                "typed sherpa ingest failed",
                sherpa_onnx_streaming_en::start_typed,
            )
            .await
        }
        SherpaExampleModel::Kroko2025 => {
            streaming_asr_support::run_streaming_asr(
                streaming_args,
                model.banner(),
                "failed to start typed sherpa-onnx Kroko 2025 bundle",
                "failed to open typed sherpa-onnx Kroko 2025 session",
                "typed sherpa Kroko 2025 ingest failed",
                sherpa_onnx_streaming_en_kroko_2025::start_typed,
            )
            .await
        }
    }
}
