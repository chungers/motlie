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

pub const SELECTOR: &str = "qwen/qwen3_6_27b_gguf";
const HF_REPO: &str = "unsloth/Qwen3.6-27B-GGUF";

pub(crate) fn register(catalog: &mut crate::Catalog) {
    catalog.register_descriptor(descriptor());
    catalog.register_model_variant(identity(), variant_descriptor());
}

pub(crate) fn identity() -> ModelIdentity {
    super::qwen3_6_27b_identity()
}

pub(crate) fn checkpoint() -> ModelCheckpoint {
    ModelCheckpoint {
        format: CheckpointFormat::Gguf,
        source: ArtifactSource::HuggingFace { repo: HF_REPO },
        include: vec![
            ArtifactRule::Exact("Qwen3.6-27B-Q4_K_M.gguf"),
            ArtifactRule::Exact("Qwen3.6-27B-Q5_K_M.gguf"),
            ArtifactRule::Exact("Qwen3.6-27B-Q8_0.gguf"),
        ],
        quantization: None,
    }
}

/// Curated bundle descriptor for Qwen3.6 27B running on the llama.cpp backend
/// with GGUF-quantized weights.
///
/// This first implementation is text-only even though the upstream model is
/// multimodal. The core `ChatModel` API already accepts image content parts, but
/// the current llama.cpp backend path rejects images until mmproj execution is
/// wired through the Rust binding.
pub fn descriptor() -> BundleDescriptor {
    let identity = identity();
    let checkpoint = checkpoint();
    BundleDescriptor {
        id: BundleId::new("qwen3_6_27b_gguf"),
        model_id: identity.id,
        display_name: "Qwen3.6 27B (GGUF/llama.cpp)".into(),
        family: identity.family,
        capabilities: motlie_model::Capabilities::chat_and_completion(),
        backend: BackendKind::LlamaCpp,
        requirements: BundleRequirements {
            platform: identity.requirements.platform,
            build: vec![BuildConstraint::Feature("backend-llama-cpp".into())],
        },
        eval_tracks: identity.eval_tracks,
        artifacts: Some(crate::bundle_artifacts_from_checkpoint(
            "qwen3_6_27b_gguf",
            &checkpoint,
        )),
    }
}

pub(crate) fn variant_descriptor() -> crate::ModelVariantDescriptor {
    let spec = LlamaCppTextSpec::qwen3_6_27b();
    crate::ModelVariantDescriptor {
        backend: BackendKind::LlamaCpp,
        capabilities: spec.capabilities,
        quantization: spec.quantization,
        checkpoint: checkpoint(),
    }
}

pub fn bundle() -> crate::CuratedBundle {
    crate::CuratedBundle::Qwen3_6_27B_Gguf
}

pub async fn start(options: StartOptions) -> Result<LlamaCppTextHandle, ModelError> {
    LlamaCppTextBundle::new(LlamaCppTextSpec::qwen3_6_27b())
        .start(crate::resolve_typed_artifact_policy(
            options,
            resolve_local_gguf_root,
        )?)
        .await
}

fn resolve_local_gguf_root(root: &Path) -> Result<PathBuf, ModelError> {
    crate::resolve_hf_gguf_snapshot(HF_REPO, root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BundleFamily;
    use motlie_model::{CapabilityDescriptor, CapabilityKind};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn descriptor_is_reviewable_as_data() {
        let descriptor = descriptor();

        assert_eq!(descriptor.id.as_str(), "qwen3_6_27b_gguf");
        assert_eq!(descriptor.model_id.as_str(), "qwen3_6_27b");
        assert_eq!(descriptor.display_name, "Qwen3.6 27B (GGUF/llama.cpp)");
        assert_eq!(descriptor.family, BundleFamily::Qwen);
        assert_eq!(descriptor.backend, BackendKind::LlamaCpp);
        assert!(descriptor.capabilities.supports(CapabilityKind::Chat));
        assert!(descriptor.capabilities.supports(CapabilityKind::Completion));
        assert!(!descriptor.capabilities.supports(CapabilityKind::Vision));
        assert_eq!(
            descriptor.capability_descriptors(),
            &[
                CapabilityDescriptor::chat(),
                CapabilityDescriptor::completion(),
            ]
        );

        let artifacts = descriptor
            .artifacts
            .expect("descriptor should expose curated artifact control");
        assert_eq!(artifacts.control_name, "qwen3_6_27b_gguf");
        assert!(artifacts.includes("Qwen3.6-27B-Q4_K_M.gguf"));
        assert!(artifacts.includes("Qwen3.6-27B-Q5_K_M.gguf"));
        assert!(artifacts.includes("Qwen3.6-27B-Q8_0.gguf"));
        assert!(!artifacts.includes("Qwen3.6-27B-FP8.gguf"));
        assert!(!artifacts.includes("mmproj-F16.gguf"));
    }

    #[test]
    fn variant_advertises_q5_default_and_no_fp8_until_gguf_exists() {
        let variant = variant_descriptor();

        assert_eq!(variant.backend, BackendKind::LlamaCpp);
        assert_eq!(variant.checkpoint.format, CheckpointFormat::Gguf);
        assert_eq!(
            variant.quantization.recommended(),
            Some(motlie_model::QuantizationBits::Five)
        );
        assert!(variant
            .quantization
            .supports(motlie_model::QuantizationBits::Four));
        assert!(variant
            .quantization
            .supports(motlie_model::QuantizationBits::Five));
        assert!(variant
            .quantization
            .supports(motlie_model::QuantizationBits::Eight));
        assert!(!variant
            .quantization
            .supports(motlie_model::QuantizationBits::FloatEight));
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
    fn local_gguf_resolution_accepts_cache_with_gguf_file() {
        let root = unique_temp_dir();
        let snapshot = create_fake_hf_gguf_cache(&root, HF_REPO);
        std::fs::write(snapshot.join("Qwen3.6-27B-Q5_K_M.gguf"), "stub")
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
        std::env::temp_dir().join(format!("motlie-models-qwen36-gguf-test-{unique}"))
    }

    fn create_fake_hf_gguf_cache(root: &Path, repo: &str) -> PathBuf {
        let repo_dir = root.join(format!("models--{}", repo.replace('/', "--")));
        let refs_dir = repo_dir.join("refs");
        let snapshots_dir = repo_dir.join("snapshots");
        let commit = "1234567890abcdef";
        let snapshot = snapshots_dir.join(commit);
        std::fs::create_dir_all(&snapshot).expect("snapshot should be creatable");
        std::fs::create_dir_all(&refs_dir).expect("refs should be creatable");
        std::fs::write(refs_dir.join("main"), commit).expect("ref should be writable");
        snapshot
    }
}
