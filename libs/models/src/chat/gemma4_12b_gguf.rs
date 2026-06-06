use std::path::{Path, PathBuf};

use motlie_model::{
    BundleId, CheckpointFormat, ModelBundle, ModelCheckpoint, ModelError, ModelIdentity,
    StartOptions,
};
use motlie_model_llama_cpp::{LlamaCppTextBundle, LlamaCppTextHandle, LlamaCppTextSpec};

use crate::{
    ArtifactRule, ArtifactSource, BackendKind, BuildConstraint, BundleDescriptor,
    BundleRequirements,
};

pub const SELECTOR: &str = "google/gemma4_12b_gguf";
const REPO: &str = "unsloth/gemma-4-12b-it-GGUF";

pub(crate) fn register(catalog: &mut crate::Catalog) {
    catalog.register_descriptor(descriptor());
    catalog.register_model_variant(identity(), variant_descriptor());
}

pub(crate) fn identity() -> ModelIdentity {
    super::gemma4_12b_identity()
}

pub(crate) fn checkpoint() -> ModelCheckpoint {
    ModelCheckpoint {
        format: CheckpointFormat::Gguf,
        source: ArtifactSource::HuggingFace { repo: REPO },
        include: vec![
            ArtifactRule::Exact("gemma-4-12b-it-Q4_K_M.gguf"),
            ArtifactRule::Exact("gemma-4-12b-it-Q8_0.gguf"),
        ],
        quantization: None,
    }
}

/// Curated bundle descriptor for Gemma 4 12B-it running on the llama.cpp
/// backend with GGUF-quantized weights from the Unsloth repository.
///
/// ## Weight compatibility with mistral.rs
///
/// The `gemma4_12b` bundle is the official full-model curated variant using
/// safetensors from `google/gemma-4-12B-it` through `mistral.rs`. This bundle
/// uses **GGUF** weights from `unsloth/gemma-4-12b-it-GGUF` through
/// `llama.cpp`. The two artifact sets are **not interchangeable**; each
/// backend requires its own format.
///
/// ## Capability boundary
///
/// The upstream GGUF repository includes multimodal projector files, but
/// Motlie's current `llama.cpp` wrapper is text-only. This bundle therefore
/// follows the existing Gemma GGUF pattern and advertises chat, completion, and
/// tool use only.
pub fn descriptor() -> BundleDescriptor {
    let identity = identity();
    let checkpoint = checkpoint();
    let spec = LlamaCppTextSpec::gemma4_12b();
    BundleDescriptor {
        id: BundleId::new("gemma4_12b_gguf"),
        model_id: identity.id,
        display_name: "Gemma 4 12B-it (GGUF/llama.cpp)".into(),
        family: identity.family,
        capabilities: spec.capabilities,
        backend: BackendKind::LlamaCpp,
        requirements: BundleRequirements {
            platform: identity.requirements.platform,
            build: vec![BuildConstraint::Feature("backend-llama-cpp".into())],
        },
        eval_tracks: identity.eval_tracks,
        artifacts: Some(crate::bundle_artifacts_from_checkpoint(
            "gemma4_12b_gguf",
            &checkpoint,
        )),
    }
}

pub(crate) fn variant_descriptor() -> crate::ModelVariantDescriptor {
    let spec = LlamaCppTextSpec::gemma4_12b();
    crate::ModelVariantDescriptor {
        backend: BackendKind::LlamaCpp,
        capabilities: spec.capabilities,
        quantization: spec.quantization,
        checkpoint: checkpoint(),
    }
}

pub fn bundle() -> crate::CuratedBundle {
    crate::CuratedBundle::Gemma4_12B_Gguf
}

pub async fn start(options: StartOptions) -> Result<LlamaCppTextHandle, ModelError> {
    LlamaCppTextBundle::new(LlamaCppTextSpec::gemma4_12b())
        .start(crate::resolve_typed_artifact_policy(
            options,
            resolve_local_gguf_root,
        )?)
        .await
}

fn resolve_local_gguf_root(root: &Path) -> Result<PathBuf, ModelError> {
    crate::resolve_hf_gguf_snapshot(REPO, root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BundleFamily;
    use motlie_model::CapabilityKind;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn descriptor_is_reviewable_as_data() {
        let descriptor = descriptor();

        assert_eq!(descriptor.id.as_str(), "gemma4_12b_gguf");
        assert_eq!(descriptor.display_name, "Gemma 4 12B-it (GGUF/llama.cpp)");
        assert_eq!(descriptor.family, BundleFamily::Gemma);
        assert_eq!(descriptor.backend, BackendKind::LlamaCpp);
        assert!(descriptor.capabilities.supports(CapabilityKind::Chat));
        assert!(descriptor.capabilities.supports(CapabilityKind::Completion));
        assert!(descriptor.capabilities.supports(CapabilityKind::ToolUse));
        let artifacts = descriptor
            .artifacts
            .expect("descriptor should expose curated artifact control");
        assert_eq!(artifacts.control_name, "gemma4_12b_gguf");
        assert!(artifacts.includes("gemma-4-12b-it-Q4_K_M.gguf"));
        assert!(artifacts.includes("gemma-4-12b-it-Q8_0.gguf"));
        assert!(!artifacts.includes("gemma-4-12b-it-BF16.gguf"));
        assert!(!artifacts.includes("MTP/gemma-4-12B-it-MTP-Q8_0.gguf"));
        assert!(!artifacts.includes("mmproj-F16.gguf"));
        assert!(!artifacts.includes("README.md"));
    }

    #[cfg(feature = "model-gemma4-12b")]
    #[test]
    fn identity_matches_logical_gemma4_model() {
        assert_eq!(identity(), crate::chat::gemma4_12b::identity());
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
        let _snapshot = create_fake_hf_gguf_cache(&root, REPO);

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
        let snapshot = create_fake_hf_gguf_cache(&root, REPO);
        std::fs::write(snapshot.join("gemma-4-12b-it-Q4_K_M.gguf"), "stub")
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
        std::env::temp_dir().join(format!("motlie-models-gemma4-12b-gguf-test-{unique}"))
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
