//! TTS↔ASR pipeline: qwen3-tts.cpp TTS → sherpa-onnx ASR
//!
//! Required features: model-qwen3-tts-cpp, model-sherpa-onnx-streaming

#[path = "../common.rs"]
mod common;

use std::path::PathBuf;

use anyhow::{Context, Result};
use motlie_model::{ArtifactPolicy, StartOptions};

const PIPELINE_CHUNK_BYTES: usize = 16_000;

fn main() -> Result<()> {
    let args = parse_args()?;
    let runtime = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;
    runtime.block_on(run(args))
}

struct Args {
    data_dir: PathBuf,
}

fn parse_args() -> Result<Args> {
    let mut data_dir = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data-dir" => {
                data_dir = Some(PathBuf::from(
                    args.next().context("--data-dir requires a path")?,
                ));
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }

    Ok(Args {
        data_dir: data_dir.context("--data-dir <path> is required")?,
    })
}

async fn run(args: Args) -> Result<()> {
    let dataset_path = args.data_dir.join("samples.json");
    let samples = common::load_dataset(&dataset_path)?;

    eprintln!(
        "=== qwen3cpp_sherpa pipeline: {} samples ===",
        samples.len()
    );

    let artifact_root = motlie_models::default_artifact_root();

    let tts_bundle = motlie_models::tts::TtsModels::Qwen3TtsCpp0_6B.bundle();
    let tts_handle = tts_bundle
        .start(StartOptions {
            artifact_policy: Some(ArtifactPolicy::LocalOnly {
                root: artifact_root.clone(),
            }),
            ..Default::default()
        })
        .await
        .context("failed to start qwen3-tts.cpp TTS bundle")?;
    let speech = tts_handle
        .speech()
        .context("qwen3-tts.cpp bundle should expose speech")?;

    let asr_bundle = motlie_models::asr::AsrModels::SherpaOnnxStreamingEn.bundle();
    let asr_handle = asr_bundle
        .start(StartOptions {
            artifact_policy: Some(ArtifactPolicy::LocalOnly {
                root: artifact_root,
            }),
            ..Default::default()
        })
        .await
        .context("failed to start sherpa-onnx ASR bundle")?;
    let transcription = asr_handle
        .transcription()
        .context("sherpa bundle should expose transcription")?;

    for sample in &samples {
        match common::run_pipeline_with_asr_chunk_bytes(
            "qwen3cpp_sherpa",
            sample,
            speech,
            transcription,
            PIPELINE_CHUNK_BYTES,
        )
        .await
        {
            Ok(result) => common::emit_jsonl(&result),
            Err(err) => {
                eprintln!("WARN: sample {} failed: {err:#}", sample.sample_id);
            }
        }
    }

    tts_handle.shutdown().await.context("TTS shutdown failed")?;
    asr_handle.shutdown().await.context("ASR shutdown failed")?;

    eprintln!("=== qwen3cpp_sherpa pipeline complete ===");
    Ok(())
}
