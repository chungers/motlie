use std::path::{Path, PathBuf};
use std::sync::Arc;

use motlie_model::eval::EvalTrack;
use motlie_model::{
    BundleId, CheckpointFormat, ModelBundle, ModelCheckpoint, ModelError, ModelIdentity,
};
use motlie_model_llama_cpp::LlamaCppTextAdapter;

use crate::{
    ArtifactRule, ArtifactSource, BackendKind, BuildConstraint, BundleArtifacts, BundleDescriptor,
    BundleFamily, BundleRequirements, PlatformConstraint,
};

pub const SELECTOR: &str = "google/gemma4_e2b_gguf";

pub(crate) fn register(catalog: &mut crate::Catalog) {
    catalog.register(descriptor(), bundle);
    catalog.register_model_variant(
        identity(),
        checkpoint(),
        Arc::new(resolve_local_gguf_root),
        Arc::new(LlamaCppTextAdapter::gemma4()),
    );
}

pub(crate) fn identity() -> ModelIdentity {
    super::gemma4_e2b::identity()
}

pub(crate) fn checkpoint() -> ModelCheckpoint {
    ModelCheckpoint {
        format: CheckpointFormat::Gguf,
        source: ArtifactSource::HuggingFace {
            repo: "unsloth/gemma-4-E2B-it-GGUF",
        },
        include: vec![
            ArtifactRule::Suffix("-Q4_K_M.gguf"),
            ArtifactRule::Suffix("-Q8_0.gguf"),
            ArtifactRule::Suffix("-f16.gguf"),
        ],
        quantization: None,
    }
}

/// Curated bundle descriptor for Gemma 4 E2B-it running on the llama.cpp
/// backend with GGUF-quantized weights.
///
/// ## Weight compatibility with mistral.rs
///
/// The mistral.rs `gemma4_e2b` bundle uses **safetensors** weights from
/// `google/gemma-4-E2B-it`. This bundle uses **GGUF** weights from
/// `unsloth/gemma-4-E2B-it-GGUF`. The two artifact sets are **not
/// interchangeable** — each backend requires its own format — but they target
/// the identical upstream Gemma 4 E2B-it architecture. Both backends produce
/// equivalent inference results at the same quantization level.
///
/// Note: unlike the Qwen3-4B GGUF which is published by the model vendor,
/// the Gemma 4 GGUF is a community quantization. The curated artifact rules
/// target the unsloth quantization repository on Hugging Face.
pub fn descriptor() -> BundleDescriptor {
    let identity = identity();
    let checkpoint = checkpoint();
    BundleDescriptor {
        id: BundleId::new("gemma4_e2b_gguf"),
        model_id: identity.id,
        display_name: "Gemma 4 E2B-it (GGUF/llama.cpp)".into(),
        family: identity.family,
        capabilities: motlie_model::Capabilities::chat_and_completion(),
        backend: BackendKind::LlamaCpp,
        requirements: BundleRequirements {
            platform: identity.requirements.platform,
            build: vec![BuildConstraint::Feature("backend-llama-cpp".into())],
        },
        eval_tracks: identity.eval_tracks,
        artifacts: Some(crate::bundle_artifacts_from_checkpoint(
            "gemma4_e2b_gguf",
            &checkpoint,
        )),
    }
}

pub fn bundle() -> Box<dyn ModelBundle> {
    let descriptor = descriptor();
    crate::adapter_backed_bundle(
        descriptor.id,
        descriptor.display_name,
        identity(),
        checkpoint(),
        Arc::new(LlamaCppTextAdapter::gemma4()),
        Arc::new(resolve_local_gguf_root),
    )
}

fn resolve_local_gguf_root(root: &Path) -> Result<PathBuf, ModelError> {
    crate::resolve_hf_gguf_snapshot("unsloth/gemma-4-E2B-it-GGUF", root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use motlie_model::CapabilityKind;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn descriptor_is_reviewable_as_data() {
        let descriptor = descriptor();

        assert_eq!(descriptor.id.as_str(), "gemma4_e2b_gguf");
        assert_eq!(descriptor.display_name, "Gemma 4 E2B-it (GGUF/llama.cpp)");
        assert_eq!(descriptor.family, BundleFamily::Gemma);
        assert_eq!(descriptor.backend, BackendKind::LlamaCpp);
        assert!(descriptor.capabilities.supports(CapabilityKind::Chat));
        assert!(descriptor.capabilities.supports(CapabilityKind::Completion));
        let artifacts = descriptor
            .artifacts
            .expect("descriptor should expose curated artifact control");
        assert_eq!(artifacts.control_name, "gemma4_e2b_gguf");
        assert!(artifacts.includes("gemma-4-e2b-it-Q4_K_M.gguf"));
        assert!(artifacts.includes("gemma-4-e2b-it-Q8_0.gguf"));
        assert!(!artifacts.includes("README.md"));
    }

    #[test]
    fn identity_matches_logical_gemma4_model() {
        assert_eq!(identity(), super::gemma4_e2b::identity());
    }

    #[test]
    fn local_gguf_resolution_rejects_missing_cache() {
        let root = unique_temp_dir();

        let error = resolve_local_gguf_root(&root).expect_err("missing cache should fail closed");

        assert!(matches!(
            error,
            ModelError::InvalidConfiguration(message) if message.contains("refs/main")
        ));
    }

    #[test]
    fn local_gguf_resolution_rejects_cache_without_gguf_files() {
        let root = unique_temp_dir();
        let _snapshot = create_fake_hf_gguf_cache(&root, "unsloth/gemma-4-E2B-it-GGUF");

        let error = resolve_local_gguf_root(&root).expect_err("cache without .gguf should fail");

        assert!(matches!(
            error,
            ModelError::InvalidConfiguration(message) if message.contains(".gguf")
        ));

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn local_gguf_resolution_accepts_cache_with_gguf_file() {
        let root = unique_temp_dir();
        let snapshot = create_fake_hf_gguf_cache(&root, "unsloth/gemma-4-E2B-it-GGUF");
        std::fs::write(snapshot.join("gemma-4-E2B-it-Q4_K_M.gguf"), "stub")
            .expect("gguf stub should be writable");

        let resolved = resolve_local_gguf_root(&root).expect("cache with .gguf should resolve");

        assert_eq!(resolved, snapshot);
        std::fs::remove_dir_all(&root).ok();
    }

    fn unique_temp_dir() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic enough")
            .as_nanos();
        std::env::temp_dir().join(format!("motlie-models-gemma4-gguf-test-{unique}"))
    }

    fn create_fake_hf_gguf_cache(root: &Path, model_id: &str) -> PathBuf {
        let repo_folder = format!("models--{}", model_id.replace('/', "--"));
        let repo_root = root.join(repo_folder);
        let refs_dir = repo_root.join("refs");
        let commit = "test-commit";
        let snapshot = repo_root.join("snapshots").join(commit);

        std::fs::create_dir_all(&snapshot).expect("snapshot dir should be creatable");
        std::fs::create_dir_all(&refs_dir).expect("refs dir should be creatable");
        std::fs::write(refs_dir.join("main"), commit).expect("ref file should be writable");

        snapshot
    }
}
