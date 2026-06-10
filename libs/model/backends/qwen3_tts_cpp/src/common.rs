use std::path::{Path, PathBuf};

use motlie_model::{ArtifactPolicy, CheckpointFormat, ModelError, ResolvedCheckpoint};

pub(crate) use motlie_model::metrics_runtime::{
    lock_metrics, observe_latency, observe_memory, RuntimeMetricState,
};

pub(crate) const MODEL_FILE_Q8_0: &str = "qwen3-tts-0.6b-q8_0.gguf";
pub(crate) const MODEL_FILE_F16: &str = "qwen3-tts-0.6b-f16.gguf";
pub(crate) const TOKENIZER_FILE_F16: &str = "qwen3-tts-tokenizer-f16.gguf";
pub(crate) const DEFAULT_SAMPLE_RATE_HZ: u32 = 24_000;

#[derive(Clone, Debug)]
pub(crate) struct Qwen3TtsCppArtifactPaths {
    pub model_dir: PathBuf,
    pub model: PathBuf,
    pub tokenizer: PathBuf,
}

pub(crate) fn resolve_gguf_artifacts(
    checkpoint: &ResolvedCheckpoint,
) -> Result<Qwen3TtsCppArtifactPaths, ModelError> {
    if checkpoint.checkpoint.format != CheckpointFormat::Gguf {
        return Err(ModelError::InvalidConfiguration(format!(
            "qwen3-tts.cpp expected Gguf checkpoint, got {:?}",
            checkpoint.checkpoint.format
        )));
    }

    let root = if checkpoint.path.is_dir() {
        checkpoint.path.clone()
    } else {
        checkpoint
            .path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."))
    };

    resolve_from_model_root(&root)
}

pub(crate) fn configure_artifact_policy(
    repo: &str,
    policy: ArtifactPolicy,
) -> Result<Qwen3TtsCppArtifactPaths, ModelError> {
    let root = match policy {
        ArtifactPolicy::AllowFetch { root } => root.unwrap_or_else(|| PathBuf::from(".")),
        ArtifactPolicy::LocalOnly { root } => root,
    };

    let model_root = if looks_like_model_dir(&root) {
        root
    } else {
        resolve_hf_snapshot_root(repo, &root)?
    };

    resolve_from_model_root(&model_root)
}

fn resolve_from_model_root(root: &Path) -> Result<Qwen3TtsCppArtifactPaths, ModelError> {
    let tokenizer = root.join(TOKENIZER_FILE_F16);
    require_file(&tokenizer, "qwen3-tts.cpp tokenizer GGUF")?;

    let model_q8 = root.join(MODEL_FILE_Q8_0);
    let model_f16 = root.join(MODEL_FILE_F16);

    let model = if model_q8.is_file() {
        model_q8
    } else if model_f16.is_file() {
        model_f16
    } else {
        return Err(ModelError::InvalidConfiguration(format!(
            "qwen3-tts.cpp model root `{}` must contain `{MODEL_FILE_Q8_0}` or `{MODEL_FILE_F16}`",
            root.display()
        )));
    };

    Ok(Qwen3TtsCppArtifactPaths {
        model_dir: root.to_path_buf(),
        model,
        tokenizer,
    })
}

fn looks_like_model_dir(root: &Path) -> bool {
    root.join(TOKENIZER_FILE_F16).is_file()
        && (root.join(MODEL_FILE_Q8_0).is_file() || root.join(MODEL_FILE_F16).is_file())
}

fn resolve_hf_snapshot_root(repo: &str, cache_root: &Path) -> Result<PathBuf, ModelError> {
    let repo_folder = format!("models--{}", repo.replace('/', "--"));
    let repo_root = cache_root.join(&repo_folder);
    let refs_dir = repo_root.join("refs");
    let main_ref = refs_dir.join("main");

    if !main_ref.exists() {
        return Err(ModelError::InvalidConfiguration(format!(
            "artifact root `{}` is neither a qwen3-tts.cpp model directory nor an HF cache containing `{repo}`",
            cache_root.display()
        )));
    }

    let commit = std::fs::read_to_string(&main_ref).map_err(|err| {
        ModelError::InvalidConfiguration(format!("failed to read HF cache ref for `{repo}`: {err}"))
    })?;
    let snapshot_dir = repo_root.join("snapshots").join(commit.trim());

    if !snapshot_dir.is_dir() {
        return Err(ModelError::InvalidConfiguration(format!(
            "HF cache snapshot for `{repo}` not found under `{}`",
            cache_root.display()
        )));
    }

    Ok(snapshot_dir)
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

pub(crate) fn resample_mono(samples: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    if src_rate == dst_rate || samples.is_empty() {
        return samples.to_vec();
    }

    let ratio = src_rate as f64 / dst_rate as f64;
    let output_len = ((samples.len() as f64) / ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos as usize;
        let frac = src_pos - idx as f64;
        let sample = if idx + 1 < samples.len() {
            samples[idx] as f64 * (1.0 - frac) + samples[idx + 1] as f64 * frac
        } else if idx < samples.len() {
            samples[idx] as f64
        } else {
            0.0
        };
        output.push(sample as f32);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direct_root_resolution_prefers_q8_transformer() {
        let root = std::env::temp_dir().join("motlie-qwen3-tts-cpp-direct-root");
        std::fs::create_dir_all(&root).expect("temp root should be creatable");
        std::fs::write(root.join(MODEL_FILE_Q8_0), b"stub").expect("q8 should be writable");
        std::fs::write(root.join(MODEL_FILE_F16), b"stub").expect("f16 should be writable");
        std::fs::write(root.join(TOKENIZER_FILE_F16), b"stub")
            .expect("tokenizer should be writable");

        let resolved = configure_artifact_policy(
            "koboldcpp/tts",
            ArtifactPolicy::LocalOnly { root: root.clone() },
        )
        .expect("direct model dir should resolve");

        assert_eq!(resolved.model, root.join(MODEL_FILE_Q8_0));
        assert_eq!(resolved.tokenizer, root.join(TOKENIZER_FILE_F16));

        std::fs::remove_dir_all(root).ok();
    }
}
