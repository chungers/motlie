use std::path::{Path, PathBuf};

use motlie_model::{ArtifactPolicy, AudioSpec, CheckpointFormat, ModelError, PcmEncoding, ResolvedCheckpoint};
use serde::Deserialize;

pub(crate) use motlie_model::metrics_runtime::{
    lock_metrics, observe_latency, observe_memory, RuntimeMetricState,
};

/// Paths to the three ONNX model components and the config file.
#[derive(Clone, Debug)]
pub(crate) struct Qwen3TtsArtifactPaths {
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub vocoder: PathBuf,
    pub config: PathBuf,
}

/// Resolve ONNX artifacts from a checkpoint root directory.
pub(crate) fn resolve_onnx_artifacts(
    checkpoint: &ResolvedCheckpoint,
) -> Result<Qwen3TtsArtifactPaths, ModelError> {
    if checkpoint.checkpoint.format != CheckpointFormat::Onnx {
        return Err(ModelError::InvalidConfiguration(format!(
            "qwen3-tts expected Onnx checkpoint, got {:?}",
            checkpoint.checkpoint.format
        )));
    }

    let root = if checkpoint.path.is_dir() {
        checkpoint.path.clone()
    } else {
        checkpoint
            .path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."))
    };

    let paths = build_artifact_paths(&root);
    validate_artifacts(&paths)?;
    Ok(paths)
}

/// Resolve artifacts from an artifact policy.
pub(crate) fn configure_artifact_policy(
    policy: ArtifactPolicy,
) -> Result<Qwen3TtsArtifactPaths, ModelError> {
    let root = match policy {
        ArtifactPolicy::AllowFetch { root } => root.unwrap_or_else(|| PathBuf::from(".")),
        ArtifactPolicy::LocalOnly { root } => root,
    };

    let paths = build_artifact_paths(&root);
    validate_artifacts(&paths)?;
    Ok(paths)
}

fn build_artifact_paths(root: &Path) -> Qwen3TtsArtifactPaths {
    Qwen3TtsArtifactPaths {
        encoder: root.join("encoder.onnx"),
        decoder: root.join("decoder.onnx"),
        vocoder: root.join("vocoder.onnx"),
        config: root.join("config.json"),
    }
}

fn validate_artifacts(paths: &Qwen3TtsArtifactPaths) -> Result<(), ModelError> {
    for (label, path) in [
        ("encoder model", &paths.encoder),
        ("decoder model", &paths.decoder),
        ("vocoder model", &paths.vocoder),
        ("model config", &paths.config),
    ] {
        if !path.is_file() {
            return Err(ModelError::InvalidConfiguration(format!(
                "qwen3-tts {label} `{}` does not exist",
                path.display()
            )));
        }
    }
    Ok(())
}

/// Model configuration parsed from config.json.
#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Qwen3TtsConfig {
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_hop_length")]
    pub hop_length: u32,
    #[serde(default = "default_mel_channels")]
    pub mel_channels: u32,
}

fn default_sample_rate() -> u32 {
    22_050
}

fn default_hop_length() -> u32 {
    256
}

fn default_mel_channels() -> u32 {
    80
}

impl Qwen3TtsConfig {
    pub(crate) fn from_path(path: &Path) -> Result<Self, ModelError> {
        let file = std::fs::File::open(path).map_err(|err| {
            ModelError::InvalidConfiguration(format!(
                "failed to open qwen3-tts config `{}`: {err}",
                path.display()
            ))
        })?;
        let config: Self = serde_json::from_reader(file).map_err(|err| {
            ModelError::InvalidConfiguration(format!(
                "failed to parse qwen3-tts config `{}`: {err}",
                path.display()
            ))
        })?;

        if config.sample_rate == 0 {
            return Err(ModelError::InvalidConfiguration(format!(
                "qwen3-tts config `{}` declares sample_rate = 0",
                path.display()
            )));
        }

        Ok(config)
    }

    pub(crate) fn audio_spec(&self) -> AudioSpec {
        AudioSpec {
            sample_rate_hz: self.sample_rate,
            channels: 1,
            encoding: PcmEncoding::F32Le,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_rejects_missing_components() {
        let root = std::env::temp_dir().join("motlie-qwen3-tts-missing");
        std::fs::create_dir_all(&root).ok();

        let paths = build_artifact_paths(&root);
        let err = validate_artifacts(&paths).expect_err("missing components should fail");

        assert!(matches!(err, ModelError::InvalidConfiguration(msg) if msg.contains("does not exist")));
    }

    #[test]
    fn resolve_rejects_wrong_checkpoint_format() {
        let checkpoint = ResolvedCheckpoint {
            checkpoint: motlie_model::ModelCheckpoint {
                format: CheckpointFormat::Gguf,
                source: motlie_model::ArtifactSource::HuggingFace {
                    repo: "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                },
                include: vec![],
                quantization: None,
            },
            path: PathBuf::from("/tmp/fake"),
        };

        let err = resolve_onnx_artifacts(&checkpoint).expect_err("wrong format should fail");
        assert!(matches!(err, ModelError::InvalidConfiguration(msg) if msg.contains("Onnx")));
    }
}
