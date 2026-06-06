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

pub const SELECTOR: &str = "google/gemma4_12b_qat_q4_0_gguf";
const REPO: &str = "google/gemma-4-12B-it-qat-q4_0-gguf";
const QAT_Q4_0_GGUF_FILE: &str = "gemma-4-12b-it-qat-q4_0.gguf";

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
        include: vec![ArtifactRule::Exact(QAT_Q4_0_GGUF_FILE)],
        quantization: None,
    }
}

/// @gemma4-cdx 2026-06-05 17:45 PDT: Curated bundle descriptor for
/// Gemma 4 12B-it QAT Q4_0 running on the llama.cpp backend with the official
/// Google GGUF artifact.
///
/// ## Weight compatibility with mistral.rs
///
/// The official full-safetensors `google/gemma-4-12B-it` path is deferred
/// because the current `mistral.rs` Gemma4 generation path failed DGX live
/// validation. This bundle uses **GGUF Q4_0** weights from
/// `google/gemma-4-12B-it-qat-q4_0-gguf` through `llama.cpp`. The artifact sets
/// are **not interchangeable**; each backend requires its own format.
///
/// ## Capability boundary
///
/// The upstream QAT GGUF repository also includes `mmproj-gemma-4-12b-it-qat-q4_0.gguf`,
/// but Motlie's current `llama.cpp` wrapper is text-only. This bundle therefore
/// follows the existing Gemma GGUF pattern and advertises chat, completion, and
/// tool use only.
pub fn descriptor() -> BundleDescriptor {
    let identity = identity();
    let checkpoint = checkpoint();
    let spec = LlamaCppTextSpec::gemma4_12b_qat_q4_0();
    BundleDescriptor {
        id: BundleId::new("gemma4_12b_qat_q4_0_gguf"),
        model_id: identity.id,
        display_name: "Gemma 4 12B-it QAT Q4_0 (GGUF/llama.cpp)".into(),
        family: identity.family,
        capabilities: spec.capabilities,
        backend: BackendKind::LlamaCpp,
        requirements: BundleRequirements {
            platform: identity.requirements.platform,
            build: vec![BuildConstraint::Feature("backend-llama-cpp".into())],
        },
        eval_tracks: identity.eval_tracks,
        artifacts: Some(crate::bundle_artifacts_from_checkpoint(
            "gemma4_12b_qat_q4_0_gguf",
            &checkpoint,
        )),
    }
}

pub(crate) fn variant_descriptor() -> crate::ModelVariantDescriptor {
    let spec = LlamaCppTextSpec::gemma4_12b_qat_q4_0();
    crate::ModelVariantDescriptor {
        backend: BackendKind::LlamaCpp,
        capabilities: spec.capabilities,
        quantization: spec.quantization,
        checkpoint: checkpoint(),
    }
}

pub fn bundle() -> crate::CuratedBundle {
    crate::CuratedBundle::Gemma4_12B_QatQ4_0_Gguf
}

pub async fn start(options: StartOptions) -> Result<LlamaCppTextHandle, ModelError> {
    LlamaCppTextBundle::new(LlamaCppTextSpec::gemma4_12b_qat_q4_0())
        .start(crate::resolve_typed_artifact_policy(
            options,
            resolve_local_gguf_root,
        )?)
        .await
}

fn resolve_local_gguf_root(root: &Path) -> Result<PathBuf, ModelError> {
    crate::resolve_hf_gguf_snapshot_with_any_file(REPO, root, &[QAT_Q4_0_GGUF_FILE])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BundleFamily;
    use motlie_model::CapabilityKind;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[test]
    fn descriptor_is_reviewable_as_data() {
        let descriptor = descriptor();

        assert_eq!(descriptor.id.as_str(), "gemma4_12b_qat_q4_0_gguf");
        assert_eq!(
            descriptor.display_name,
            "Gemma 4 12B-it QAT Q4_0 (GGUF/llama.cpp)"
        );
        assert_eq!(descriptor.family, BundleFamily::Gemma);
        assert_eq!(descriptor.backend, BackendKind::LlamaCpp);
        assert!(descriptor.capabilities.supports(CapabilityKind::Chat));
        assert!(descriptor.capabilities.supports(CapabilityKind::Completion));
        assert!(descriptor.capabilities.supports(CapabilityKind::ToolUse));
        let artifacts = descriptor
            .artifacts
            .expect("descriptor should expose curated artifact control");
        assert_eq!(artifacts.control_name, "gemma4_12b_qat_q4_0_gguf");
        assert!(artifacts.includes("gemma-4-12b-it-qat-q4_0.gguf"));
        assert!(!artifacts.includes("gemma-4-12b-it-Q4_K_M.gguf"));
        assert!(!artifacts.includes("gemma-4-12b-it-Q8_0.gguf"));
        assert!(!artifacts.includes("mmproj-gemma-4-12b-it-qat-q4_0.gguf"));
        assert!(!artifacts.includes("README.md"));
    }

    #[test]
    fn local_gguf_resolution_rejects_missing_cache() {
        let temp = TestCacheDir::new("missing-cache");

        let error =
            resolve_local_gguf_root(temp.path()).expect_err("missing cache should fail closed");

        assert!(matches!(
            error,
            ModelError::InvalidConfiguration(message) if message.contains("refs/main")
        ));
    }

    #[test]
    fn local_gguf_resolution_rejects_cache_without_gguf_files() {
        let temp = TestCacheDir::new("cache-without-gguf");
        let _snapshot = create_fake_hf_gguf_cache(temp.path(), REPO);

        let error =
            resolve_local_gguf_root(temp.path()).expect_err("cache without .gguf should fail");

        assert!(matches!(
            error,
            ModelError::InvalidConfiguration(message) if message.contains(".gguf")
        ));
    }

    #[test]
    fn local_gguf_resolution_accepts_cache_with_gguf_file() {
        let temp = TestCacheDir::new("qat-accepts-cache-with-gguf");
        let snapshot = create_fake_hf_gguf_cache(temp.path(), REPO);
        std::fs::write(snapshot.join(QAT_Q4_0_GGUF_FILE), "stub")
            .expect("gguf stub should be writable");

        let resolved =
            resolve_local_gguf_root(temp.path()).expect("cache with .gguf should resolve");

        assert_eq!(resolved, snapshot);
    }

    #[test]
    fn local_gguf_resolution_rejects_standard_only_cache_for_qat_variant() {
        let temp = TestCacheDir::new("qat-rejects-standard-cache");
        let snapshot = create_fake_hf_gguf_cache(temp.path(), REPO);
        std::fs::write(snapshot.join("gemma-4-12b-it-Q4_K_M.gguf"), "stub")
            .expect("wrong gguf stub should be writable");

        let error = resolve_local_gguf_root(temp.path())
            .expect_err("QAT GGUF resolver should reject standard-only cache");

        assert!(matches!(
            error,
            ModelError::InvalidConfiguration(message) if message.contains(QAT_Q4_0_GGUF_FILE)
        ));
    }

    struct TestCacheDir {
        path: PathBuf,
    }

    impl TestCacheDir {
        fn new(label: &str) -> Self {
            static NEXT_ID: AtomicU64 = AtomicU64::new(0);
            let unique = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "motlie-models-gemma4-12b-qat-q4-0-gguf-test-{}-{unique}-{label}",
                std::process::id()
            ));
            std::fs::remove_dir_all(&path).ok();
            Self { path }
        }

        fn path(&self) -> &Path {
            &self.path
        }
    }

    impl Drop for TestCacheDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.path).ok();
        }
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
