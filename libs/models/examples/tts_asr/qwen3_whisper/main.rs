//! TTS↔ASR pipeline: Qwen3-TTS → whisper.cpp ASR
//!
//! Required features: model-qwen3-tts-0_6b, model-whisper-base-en

#[path = "../common.rs"]
mod common;

use anyhow::{Context, Result};
use motlie_model::{ArtifactPolicy, StartOptions};
use std::path::PathBuf;

fn main() -> Result<()> {
    let args = parse_args()?;
    let runtime = tokio::runtime::Runtime::new().context("failed to create tokio runtime")?;
    runtime.block_on(run(args))
}

struct Args {
    data_dir: PathBuf,
    cuda: bool,
}

fn parse_args() -> Result<Args> {
    let mut data_dir = None;
    let mut cuda = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data-dir" => {
                data_dir = Some(PathBuf::from(
                    args.next().context("--data-dir requires a path")?,
                ));
            }
            "--cuda" => cuda = true,
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }

    Ok(Args {
        data_dir: data_dir.context("--data-dir <path> is required")?,
        cuda,
    })
}

async fn run(args: Args) -> Result<()> {
    let dataset_path = args.data_dir.join("samples.json");
    let samples = common::load_dataset(&dataset_path)?;

    eprintln!(
        "=== qwen3_whisper pipeline: {} samples ===",
        samples.len()
    );

    let artifact_root = motlie_models::default_artifact_root();

    let tts_bundle = motlie_models::tts::TtsModels::Qwen3Tts12Hz0_6B.bundle();
    let tts_handle = tts_bundle
        .start(StartOptions {
            artifact_policy: Some(ArtifactPolicy::LocalOnly {
                root: artifact_root.clone(),
            }),
            ..Default::default()
        })
        .await
        .context("failed to start Qwen3-TTS bundle")?;
    let speech = tts_handle
        .speech()
        .context("Qwen3-TTS bundle should expose speech")?;

    let asr_bundle = motlie_models::asr::AsrModels::WhisperBaseEn.bundle();
    let asr_handle = asr_bundle
        .start(StartOptions {
            artifact_policy: Some(ArtifactPolicy::LocalOnly {
                root: artifact_root,
            }),
            ..Default::default()
        })
        .await
        .context("failed to start whisper ASR bundle")?;
    let transcription = asr_handle
        .transcription()
        .context("whisper bundle should expose transcription")?;

    for sample in &samples {
        match common::run_pipeline("qwen3_whisper", sample, speech, transcription).await {
            Ok(result) => common::emit_jsonl(&result),
            Err(err) => {
                eprintln!("WARN: sample {} failed: {err:#}", sample.sample_id);
            }
        }
    }

    tts_handle.shutdown().await.context("TTS shutdown failed")?;
    asr_handle.shutdown().await.context("ASR shutdown failed")?;

    eprintln!("=== qwen3_whisper pipeline complete ===");
    Ok(())
}
