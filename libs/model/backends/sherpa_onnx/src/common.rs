use std::path::{Path, PathBuf};

use motlie_model::{ArtifactPolicy, CheckpointFormat, ModelError, ResolvedCheckpoint};

pub(crate) use motlie_model::metrics_runtime::{
    lock_metrics, observe_latency, observe_memory, RuntimeMetricState,
};

#[derive(Clone, Debug)]
pub(crate) struct SherpaArtifactPaths {
    pub encoder: PathBuf,
    pub decoder: PathBuf,
    pub joiner: PathBuf,
    pub tokens: PathBuf,
}

pub(crate) fn resolve_onnx_artifacts(
    checkpoint: &ResolvedCheckpoint,
    spec: SherpaArtifactSpec<'_>,
) -> Result<SherpaArtifactPaths, ModelError> {
    if checkpoint.checkpoint.format != CheckpointFormat::Onnx {
        return Err(ModelError::InvalidConfiguration(format!(
            "sherpa-onnx expected Onnx checkpoint, got {:?}",
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

    let encoder = require_file(&root, spec.encoder)?;
    let decoder = require_file(&root, spec.decoder)?;
    let joiner = require_file(&root, spec.joiner)?;
    let tokens = require_file(&root, spec.tokens)?;

    Ok(SherpaArtifactPaths {
        encoder,
        decoder,
        joiner,
        tokens,
    })
}

pub(crate) fn configure_artifact_policy(
    spec: SherpaArtifactSpec<'_>,
    policy: ArtifactPolicy,
) -> Result<SherpaArtifactPaths, ModelError> {
    let root = match policy {
        ArtifactPolicy::AllowFetch { root } => root.unwrap_or_else(|| PathBuf::from(".")),
        ArtifactPolicy::LocalOnly { root } => root,
    };

    Ok(SherpaArtifactPaths {
        encoder: require_file(&root, spec.encoder)?,
        decoder: require_file(&root, spec.decoder)?,
        joiner: require_file(&root, spec.joiner)?,
        tokens: require_file(&root, spec.tokens)?,
    })
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SherpaArtifactSpec<'a> {
    pub encoder: &'a str,
    pub decoder: &'a str,
    pub joiner: &'a str,
    pub tokens: &'a str,
}

fn require_file(root: &Path, filename: &str) -> Result<PathBuf, ModelError> {
    let path = root.join(filename);
    if !path.is_file() {
        return Err(ModelError::InvalidConfiguration(format!(
            "required sherpa-onnx artifact `{filename}` not found under `{}`",
            root.display()
        )));
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_policy_requires_all_artifacts() {
        let root = std::env::temp_dir().join("motlie-sherpa-onnx-artifacts-missing");
        std::fs::create_dir_all(&root).ok();

        let error = configure_artifact_policy(
            SherpaArtifactSpec {
                encoder: "encoder.onnx",
                decoder: "decoder.onnx",
                joiner: "joiner.onnx",
                tokens: "tokens.txt",
            },
            ArtifactPolicy::LocalOnly { root: root.clone() },
        )
        .expect_err("missing artifacts should fail");

        assert!(matches!(
            error,
            ModelError::InvalidConfiguration(message) if message.contains("encoder.onnx")
        ));
    }
}
