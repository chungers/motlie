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

pub const SELECTOR: &str = "deepreinforce-ai/ornith_1_0_35b_gguf";
const HF_REPO: &str = "deepreinforce-ai/Ornith-1.0-35B-GGUF";

pub(crate) fn register(catalog: &mut crate::Catalog) {
    catalog.register_descriptor(descriptor());
    catalog.register_model_variant(identity(), variant_descriptor());
}

pub(crate) fn identity() -> ModelIdentity {
    super::ornith_1_0_35b_identity()
}

pub(crate) fn checkpoint() -> ModelCheckpoint {
    ModelCheckpoint {
        format: CheckpointFormat::Gguf,
        source: ArtifactSource::HuggingFace { repo: HF_REPO },
        include: vec![
            ArtifactRule::Exact("ornith-1.0-35b-Q4_K_M.gguf"),
            ArtifactRule::Exact("ornith-1.0-35b-Q8_0.gguf"),
        ],
        quantization: None,
    }
}

/// Curated bundle descriptor for Ornith 1.0 35B running on the llama.cpp
/// backend with GGUF-quantized weights. The bundle follows the Qwen-style
/// reasoning and tool-call path documented by the upstream model card.
pub fn descriptor() -> BundleDescriptor {
    let identity = identity();
    let checkpoint = checkpoint();
    BundleDescriptor {
        id: BundleId::new("ornith_1_0_35b_gguf"),
        model_id: identity.id,
        display_name: "Ornith 1.0 35B (GGUF/llama.cpp)".into(),
        family: identity.family,
        capabilities: motlie_model::Capabilities::chat_completion_and_tool_use(),
        backend: BackendKind::LlamaCpp,
        requirements: BundleRequirements {
            platform: identity.requirements.platform,
            build: vec![BuildConstraint::Feature("backend-llama-cpp".into())],
        },
        eval_tracks: identity.eval_tracks,
        artifacts: Some(crate::bundle_artifacts_from_checkpoint(
            "ornith_1_0_35b_gguf",
            &checkpoint,
            crate::ArtifactProvenance::new("mit", crate::ArtifactGating::Public),
        )),
    }
}

pub(crate) fn variant_descriptor() -> crate::ModelVariantDescriptor {
    let spec = LlamaCppTextSpec::ornith_1_0_35b();
    crate::ModelVariantDescriptor {
        backend: BackendKind::LlamaCpp,
        capabilities: spec.capabilities,
        quantization: spec.quantization,
        checkpoint: checkpoint(),
    }
}

pub fn bundle() -> crate::CuratedBundle {
    crate::CuratedBundle::Ornith_1_0_35B_Gguf
}

pub async fn start(options: StartOptions) -> Result<LlamaCppTextHandle, ModelError> {
    LlamaCppTextBundle::new(LlamaCppTextSpec::ornith_1_0_35b())
        .start(crate::resolve_typed_artifact_policy(
            options,
            resolve_local_gguf_root,
        )?)
        .await
}

fn resolve_local_gguf_root(root: &Path) -> Result<PathBuf, ModelError> {
    crate::resolve_hf_gguf_snapshot_with_any_file(
        HF_REPO,
        root,
        &["ornith-1.0-35b-Q4_K_M.gguf", "ornith-1.0-35b-Q8_0.gguf"],
    )
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

        assert_eq!(descriptor.id.as_str(), "ornith_1_0_35b_gguf");
        assert_eq!(descriptor.model_id.as_str(), "ornith_1_0_35b");
        assert_eq!(descriptor.display_name, "Ornith 1.0 35B (GGUF/llama.cpp)");
        assert_eq!(descriptor.family, BundleFamily::Other("Ornith".into()));
        assert_eq!(descriptor.backend, BackendKind::LlamaCpp);
        assert!(descriptor.capabilities.supports(CapabilityKind::Chat));
        assert!(descriptor.capabilities.supports(CapabilityKind::Completion));
        assert!(descriptor.capabilities.supports(CapabilityKind::ToolUse));
        assert!(!descriptor.capabilities.supports(CapabilityKind::Vision));
        assert_eq!(
            descriptor.capability_descriptors(),
            &[
                CapabilityDescriptor::chat(),
                CapabilityDescriptor::completion(),
                CapabilityDescriptor::tool_use(),
            ]
        );

        let artifacts = descriptor
            .artifacts
            .expect("descriptor should expose curated artifact control");
        assert_eq!(artifacts.control_name, "ornith_1_0_35b_gguf");
        assert!(artifacts.includes("ornith-1.0-35b-Q4_K_M.gguf"));
        assert!(artifacts.includes("ornith-1.0-35b-Q8_0.gguf"));
        assert!(!artifacts.includes("ornith-1.0-35b-Q5_K_M.gguf"));
        assert!(!artifacts.includes("ornith-1.0-35b-bf16.gguf"));
        assert!(!artifacts.includes("mmproj-F16.gguf"));
    }

    #[test]
    fn variant_advertises_q4_q8_and_no_fp16() {
        let variant = variant_descriptor();

        assert_eq!(variant.backend, BackendKind::LlamaCpp);
        assert_eq!(variant.checkpoint.format, CheckpointFormat::Gguf);
        assert_eq!(
            variant.quantization.recommended(),
            Some(motlie_model::QuantizationScheme::GgufQ4_K_M)
        );
        assert!(variant
            .quantization
            .supports(motlie_model::QuantizationScheme::GgufQ4_K_M));
        assert!(!variant
            .quantization
            .supports(motlie_model::QuantizationScheme::GgufQ5_K_M));
        assert!(variant
            .quantization
            .supports(motlie_model::QuantizationScheme::GgufQ8_0));
        assert!(!variant
            .quantization
            .supports(motlie_model::QuantizationScheme::Fp16));
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
        std::fs::write(snapshot.join("ornith-1.0-35b-Q4_K_M.gguf"), "stub")
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
        std::env::temp_dir().join(format!("motlie-models-ornith-gguf-test-{unique}"))
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
