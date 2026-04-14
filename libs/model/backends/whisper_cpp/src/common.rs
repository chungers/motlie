use std::path::PathBuf;

use motlie_model::{ArtifactPolicy, CheckpointFormat, ModelError, ResolvedCheckpoint};

// Re-export shared metric types and helpers so backend code imports from one place.
pub(crate) use motlie_model::metrics_runtime::{
    lock_metrics, observe_latency, observe_memory, RuntimeMetricState,
};

/// Validate a resolved checkpoint has the expected ggml format and return
/// the model file path.
pub(crate) fn resolve_ggml_model_path(
    checkpoint: &ResolvedCheckpoint,
) -> Result<PathBuf, ModelError> {
    if checkpoint.checkpoint.format != CheckpointFormat::Ggml {
        return Err(ModelError::InvalidConfiguration(format!(
            "whisper.cpp expected Ggml checkpoint, got {:?}",
            checkpoint.checkpoint.format
        )));
    }

    let path = &checkpoint.path;

    // If the resolved path points directly at a .bin file, use it.
    if path.is_file() {
        return Ok(path.clone());
    }

    // Otherwise scan the directory for a single .bin model file.
    let mut matches: Vec<PathBuf> = std::fs::read_dir(path)
        .map_err(|e| {
            ModelError::InvalidConfiguration(format!(
                "failed to inspect ggml checkpoint root `{}`: {e}",
                path.display()
            ))
        })?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|candidate| {
            candidate.is_file()
                && candidate
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext == "bin")
                    .unwrap_or(false)
        })
        .collect();
    matches.sort();

    match matches.len() {
        1 => Ok(matches.remove(0)),
        0 => Err(ModelError::InvalidConfiguration(format!(
            "ggml checkpoint root `{}` does not contain a .bin model file",
            path.display()
        ))),
        count => Err(ModelError::InvalidConfiguration(format!(
            "ggml checkpoint root `{}` has {count} .bin files; expected exactly one",
            path.display()
        ))),
    }
}

pub(crate) fn configure_artifact_policy(
    ggml_filename: &str,
    policy: ArtifactPolicy,
) -> Result<PathBuf, ModelError> {
    match policy {
        ArtifactPolicy::AllowFetch { root } => {
            let root = root.unwrap_or_else(|| PathBuf::from("."));
            let model_path = root.join(ggml_filename);
            if !model_path.exists() {
                return Err(ModelError::InvalidConfiguration(format!(
                    "ggml artifact `{}` not found under `{}` (auto-download not yet supported for whisper ggml)",
                    ggml_filename,
                    root.display()
                )));
            }
            Ok(model_path)
        }
        ArtifactPolicy::LocalOnly { root } => {
            let model_path = root.join(ggml_filename);
            if !model_path.exists() {
                return Err(ModelError::InvalidConfiguration(format!(
                    "ggml artifact `{}` not found under `{}`",
                    ggml_filename,
                    root.display()
                )));
            }
            Ok(model_path)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_only_policy_rejects_missing_ggml() {
        let root = std::env::temp_dir().join("motlie-whisper-cpp-test-missing");
        std::fs::create_dir_all(&root).ok();

        let err = configure_artifact_policy(
            "ggml-base.en.bin",
            ArtifactPolicy::LocalOnly { root },
        )
        .expect_err("missing ggml should fail");

        assert!(matches!(err, ModelError::InvalidConfiguration(msg) if msg.contains("not found")));
    }

    #[test]
    fn local_only_policy_accepts_existing_ggml() {
        let root = std::env::temp_dir().join("motlie-whisper-cpp-test-exists");
        std::fs::create_dir_all(&root).ok();
        let bin = root.join("ggml-base.en.bin");
        std::fs::write(&bin, b"stub").ok();

        let resolved = configure_artifact_policy(
            "ggml-base.en.bin",
            ArtifactPolicy::LocalOnly { root: root.clone() },
        )
        .expect("existing ggml should succeed");

        assert_eq!(resolved, bin);
        std::fs::remove_file(bin).ok();
    }

    #[test]
    fn resolve_ggml_rejects_wrong_format() {
        let checkpoint = ResolvedCheckpoint {
            checkpoint: motlie_model::ModelCheckpoint {
                format: CheckpointFormat::Gguf,
                source: motlie_model::ArtifactSource::HuggingFace {
                    repo: "ggerganov/whisper.cpp",
                },
                include: vec![],
                quantization: None,
            },
            path: PathBuf::from("/tmp/fake"),
        };

        let err = resolve_ggml_model_path(&checkpoint)
            .expect_err("wrong format should fail");
        assert!(matches!(err, ModelError::InvalidConfiguration(msg) if msg.contains("Ggml")));
    }
}
