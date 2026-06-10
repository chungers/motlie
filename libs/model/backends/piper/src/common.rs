use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use motlie_model::{ArtifactPolicy, CheckpointFormat, ModelError, ResolvedCheckpoint};
use serde::Deserialize;

pub(crate) use motlie_model::metrics_runtime::{
    lock_metrics, observe_latency, observe_memory, RuntimeMetricState,
};

#[derive(Clone, Debug)]
pub(crate) struct PiperArtifactPaths {
    pub model: PathBuf,
    pub config: PathBuf,
}

pub(crate) fn resolve_onnx_artifacts(
    checkpoint: &ResolvedCheckpoint,
    filename: &str,
) -> Result<PiperArtifactPaths, ModelError> {
    if checkpoint.checkpoint.format != CheckpointFormat::Onnx {
        return Err(ModelError::InvalidConfiguration(format!(
            "piper expected Onnx checkpoint, got {:?}",
            checkpoint.checkpoint.format
        )));
    }

    let model = if checkpoint.path.is_dir() {
        checkpoint.path.join(filename)
    } else {
        checkpoint.path.clone()
    };

    validate_model_path(&model)?;
    let config = sidecar_path(&model);
    require_file(&config, "piper sidecar config")?;

    Ok(PiperArtifactPaths { model, config })
}

pub(crate) fn configure_artifact_policy(
    filename: &str,
    policy: ArtifactPolicy,
) -> Result<PiperArtifactPaths, ModelError> {
    let root = match policy {
        ArtifactPolicy::AllowFetch { root } => root.unwrap_or_else(|| PathBuf::from(".")),
        ArtifactPolicy::LocalOnly { root } => root,
    };

    let model = root.join(filename);
    validate_model_path(&model)?;
    let config = sidecar_path(&model);
    require_file(&config, "piper sidecar config")?;

    Ok(PiperArtifactPaths { model, config })
}

#[derive(Clone, Debug)]
pub(crate) struct PiperConfig {
    pub sample_rate_hz: u32,
    pub espeak_voice: String,
    pub phoneme_id_map: HashMap<String, Vec<i64>>,
    pub default_noise_scale: f32,
    pub default_length_scale: f32,
    pub default_noise_w: f32,
    pub bos_id: i64,
    pub eos_id: i64,
    pub pad_id: i64,
}

impl PiperConfig {
    pub(crate) fn from_path(path: &Path) -> Result<Self, ModelError> {
        let file = File::open(path).map_err(|err| {
            ModelError::InvalidConfiguration(format!(
                "failed to open piper sidecar config `{}`: {err}",
                path.display()
            ))
        })?;
        let parsed: PiperSidecar = serde_json::from_reader(file).map_err(|err| {
            ModelError::InvalidConfiguration(format!(
                "failed to parse piper sidecar config `{}`: {err}",
                path.display()
            ))
        })?;
        parsed.into_config(path)
    }
}

#[derive(Clone, Debug, Deserialize)]
struct PiperSidecar {
    audio: PiperAudioConfig,
    espeak: PiperEspeakConfig,
    inference: PiperInferenceConfig,
    phoneme_id_map: HashMap<String, Vec<i64>>,
}

impl PiperSidecar {
    fn into_config(self, path: &Path) -> Result<PiperConfig, ModelError> {
        if self.audio.sample_rate == 0 {
            return Err(ModelError::InvalidConfiguration(format!(
                "piper sidecar `{}` declares sample_rate = 0",
                path.display()
            )));
        }

        let bos_id = first_symbol_id(&self.phoneme_id_map, "^", path)?;
        let eos_id = first_symbol_id(&self.phoneme_id_map, "$", path)?;
        let pad_id = first_symbol_id(&self.phoneme_id_map, "_", path)?;

        Ok(PiperConfig {
            sample_rate_hz: self.audio.sample_rate,
            espeak_voice: self.espeak.voice,
            phoneme_id_map: self.phoneme_id_map,
            default_noise_scale: self.inference.noise_scale,
            default_length_scale: self.inference.length_scale,
            default_noise_w: self.inference.noise_w,
            bos_id,
            eos_id,
            pad_id,
        })
    }
}

#[derive(Clone, Debug, Deserialize)]
struct PiperAudioConfig {
    sample_rate: u32,
}

#[derive(Clone, Debug, Deserialize)]
struct PiperEspeakConfig {
    voice: String,
}

#[derive(Clone, Debug, Deserialize)]
struct PiperInferenceConfig {
    noise_scale: f32,
    length_scale: f32,
    noise_w: f32,
}

fn first_symbol_id(
    phoneme_id_map: &HashMap<String, Vec<i64>>,
    symbol: &str,
    path: &Path,
) -> Result<i64, ModelError> {
    phoneme_id_map
        .get(symbol)
        .and_then(|ids| ids.first())
        .copied()
        .ok_or_else(|| {
            ModelError::InvalidConfiguration(format!(
                "piper sidecar `{}` is missing phoneme_id_map entry `{symbol}`",
                path.display()
            ))
        })
}

fn validate_model_path(path: &Path) -> Result<(), ModelError> {
    require_file(path, "piper model")?;
    if path.extension().and_then(|ext| ext.to_str()) != Some("onnx") {
        return Err(ModelError::InvalidConfiguration(format!(
            "piper model path `{}` must end with `.onnx`",
            path.display()
        )));
    }
    Ok(())
}

fn sidecar_path(model: &Path) -> PathBuf {
    model.with_extension("onnx.json")
}

fn require_file(path: &Path, label: &str) -> Result<(), ModelError> {
    if !path.is_file() {
        return Err(ModelError::InvalidConfiguration(format!(
            "{label} `{}` does not exist",
            path.display()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_policy_requires_sidecar() {
        let root = std::env::temp_dir().join("motlie-piper-artifacts-missing");
        std::fs::create_dir_all(&root).ok();
        let model_path = root.join("voice.onnx");
        std::fs::write(&model_path, b"stub").expect("stub model should be writable");

        let error = configure_artifact_policy(
            "voice.onnx",
            ArtifactPolicy::LocalOnly { root: root.clone() },
        )
        .expect_err("missing sidecar should fail");

        assert!(matches!(
            error,
            ModelError::InvalidConfiguration(message) if message.contains("onnx.json")
        ));
    }

    #[test]
    fn sidecar_parser_requires_special_symbols() {
        let root = std::env::temp_dir().join("motlie-piper-sidecar-parse");
        std::fs::create_dir_all(&root).expect("temp root should exist");
        let config_path = root.join("voice.onnx.json");
        std::fs::write(
            &config_path,
            r#"{
                "audio": { "sample_rate": 22050 },
                "espeak": { "voice": "en-us" },
                "inference": { "noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8 },
                "num_speakers": 1,
                "speaker_id_map": {},
                "phoneme_id_map": { "a": [1] }
            }"#,
        )
        .expect("stub config should be writable");

        let error = PiperConfig::from_path(&config_path)
            .expect_err("missing BOS/EOS/PAD symbols should fail");

        assert!(matches!(
            error,
            ModelError::InvalidConfiguration(message) if message.contains("phoneme_id_map entry")
        ));
    }
}
