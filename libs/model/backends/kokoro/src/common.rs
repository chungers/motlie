use std::path::{Path, PathBuf};

use motlie_model::{ArtifactPolicy, CheckpointFormat, ModelError, ResolvedCheckpoint};

pub(crate) use motlie_model::metrics_runtime::{
    lock_metrics, observe_latency, observe_memory, RuntimeMetricState,
};

#[derive(Clone, Debug)]
pub(crate) struct KokoroArtifactPaths {
    pub model: PathBuf,
    pub tokenizer_json: PathBuf,
    pub voice: PathBuf,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct KokoroArtifactSpec<'a> {
    pub model: &'a str,
    pub tokenizer_json: &'a str,
    pub voice: &'a str,
}

pub(crate) fn resolve_onnx_artifacts(
    checkpoint: &ResolvedCheckpoint,
    spec: KokoroArtifactSpec<'_>,
) -> Result<KokoroArtifactPaths, ModelError> {
    if checkpoint.checkpoint.format != CheckpointFormat::Onnx {
        return Err(ModelError::InvalidConfiguration(format!(
            "kokoro expected Onnx checkpoint, got {:?}",
            checkpoint.checkpoint.format
        )));
    }

    let root = if checkpoint.path.is_dir() {
        checkpoint.path.clone()
    } else {
        checkpoint
            .path
            .parent()
            .ok_or_else(|| {
                ModelError::InvalidConfiguration(format!(
                    "onnx checkpoint path `{}` has no parent directory",
                    checkpoint.path.display()
                ))
            })?
            .to_path_buf()
    };

    build_artifacts(&root, spec)
}

pub(crate) fn configure_artifact_policy(
    spec: KokoroArtifactSpec<'_>,
    policy: ArtifactPolicy,
) -> Result<KokoroArtifactPaths, ModelError> {
    let root = match policy {
        ArtifactPolicy::AllowFetch { root } => root.unwrap_or_else(|| PathBuf::from(".")),
        ArtifactPolicy::LocalOnly { root } => root,
    };

    build_artifacts(&root, spec)
}

fn build_artifacts(
    root: &Path,
    spec: KokoroArtifactSpec<'_>,
) -> Result<KokoroArtifactPaths, ModelError> {
    Ok(KokoroArtifactPaths {
        model: require_file(root, spec.model, "kokoro model")?,
        tokenizer_json: require_file(root, spec.tokenizer_json, "kokoro tokenizer")?,
        voice: require_file(root, spec.voice, "kokoro voice")?,
    })
}

fn require_file(root: &Path, relative: &str, label: &str) -> Result<PathBuf, ModelError> {
    let nested = root.join(relative);
    if nested.is_file() {
        return Ok(nested);
    }

    let basename = Path::new(relative).file_name().ok_or_else(|| {
        ModelError::InvalidConfiguration(format!("invalid {label} artifact path `{relative}`"))
    })?;
    let direct = root.join(basename);
    if direct.is_file() {
        return Ok(direct);
    }

    Err(ModelError::InvalidConfiguration(format!(
        "required {label} artifact `{relative}` not found under `{}`",
        root.display()
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_basename_artifacts_are_accepted() {
        let root = unique_temp_dir();
        std::fs::create_dir_all(&root).expect("temp root should exist");
        std::fs::write(root.join("model_quantized.onnx"), b"model").expect("model");
        std::fs::write(root.join("tokenizer.json"), b"{}").expect("tokenizer");
        std::fs::write(root.join("af_bella.bin"), b"voice").expect("voice");

        let paths = build_artifacts(
            &root,
            KokoroArtifactSpec {
                model: "onnx/model_quantized.onnx",
                tokenizer_json: "tokenizer.json",
                voice: "voices/af_bella.bin",
            },
        )
        .expect("direct artifacts should resolve");

        assert_eq!(paths.model, root.join("model_quantized.onnx"));
        assert_eq!(paths.tokenizer_json, root.join("tokenizer.json"));
        assert_eq!(paths.voice, root.join("af_bella.bin"));
        std::fs::remove_dir_all(root).ok();
    }

    fn unique_temp_dir() -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("motlie-kokoro-artifacts-{unique}"))
    }
}
