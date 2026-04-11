use std::path::PathBuf;

use motlie_model::{ArtifactPolicy, ModelError};

// Re-export shared metric types and helpers so backend code imports from one place.
pub(crate) use motlie_model::metrics_runtime::{
    lock_metrics, observe_latency, observe_memory, observe_text_generation,
    snapshot_text_metrics, RuntimeMetricState, TextMetricState,
    should_force_cpu,
};

#[derive(Debug)]
pub(crate) struct ConfiguredGguf {
    pub(crate) model_path: PathBuf,
}

pub(crate) fn configure_artifact_policy(
    gguf_filename: &str,
    policy: ArtifactPolicy,
) -> Result<ConfiguredGguf, ModelError> {
    match policy {
        ArtifactPolicy::AllowFetch { root } => {
            let root = root.unwrap_or_else(|| PathBuf::from("."));
            let model_path = root.join(gguf_filename);
            if !model_path.exists() {
                return Err(ModelError::InvalidConfiguration(format!(
                    "GGUF artifact `{}` not found under `{}` (auto-download not yet supported for GGUF)",
                    gguf_filename,
                    root.display()
                )));
            }
            Ok(ConfiguredGguf { model_path })
        }
        ArtifactPolicy::LocalOnly { root } => {
            let model_path = root.join(gguf_filename);
            if !model_path.exists() {
                return Err(ModelError::InvalidConfiguration(format!(
                    "GGUF artifact `{}` not found under `{}`",
                    gguf_filename,
                    root.display()
                )));
            }
            Ok(ConfiguredGguf { model_path })
        }
    }
}

/// Resolve the number of model layers to offload to GPU.
///
/// Priority:
/// 1. `MOTLIE_MODEL_FORCE_CPU=1` → 0 layers (all CPU)
/// 2. `MOTLIE_MODEL_GPU_LAYERS=<n>` → explicit layer count
/// 3. Default → 9999 (offload everything available to GPU; no-op on CPU-only builds)
pub(crate) fn resolve_gpu_layers() -> u32 {
    if should_force_cpu() {
        return 0;
    }
    if let Ok(val) = std::env::var("MOTLIE_MODEL_GPU_LAYERS") {
        match val.parse::<u32>() {
            Ok(n) => return n,
            Err(_) => {
                tracing::warn!(
                    env_var = "MOTLIE_MODEL_GPU_LAYERS",
                    value = %val,
                    "ignoring unparseable GPU layer count; defaulting to full offload (9999)"
                );
            }
        }
    }
    9999
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_only_policy_rejects_missing_gguf() {
        let root = std::env::temp_dir().join("motlie-llama-cpp-test-missing");
        std::fs::create_dir_all(&root).ok();

        let err = configure_artifact_policy(
            "model-Q4_K_M.gguf",
            ArtifactPolicy::LocalOnly { root },
        )
        .expect_err("missing GGUF should fail");

        assert!(matches!(err, ModelError::InvalidConfiguration(msg) if msg.contains("not found")));
    }

    #[test]
    fn local_only_policy_accepts_existing_gguf() {
        let root = std::env::temp_dir().join("motlie-llama-cpp-test-exists");
        std::fs::create_dir_all(&root).ok();
        let gguf = root.join("model-Q4_K_M.gguf");
        std::fs::write(&gguf, b"stub").ok();

        let configured = configure_artifact_policy(
            "model-Q4_K_M.gguf",
            ArtifactPolicy::LocalOnly { root: root.clone() },
        )
        .expect("existing GGUF should succeed");

        assert_eq!(configured.model_path, gguf);
        std::fs::remove_file(gguf).ok();
    }
}
