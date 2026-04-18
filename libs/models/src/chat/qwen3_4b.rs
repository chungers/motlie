use std::path::{Path, PathBuf};

use motlie_model::{
    BundleId, CheckpointFormat, ModelBundle, ModelCheckpoint, ModelError, ModelIdentity,
    StartOptions,
};
use motlie_model_mistral::{MistralTextBundle, MistralTextHandle, MistralTextSpec};

use crate::{
    ArtifactRule, ArtifactSource, BackendKind, BuildConstraint, BundleDescriptor,
    BundleRequirements,
};

pub const SELECTOR: &str = "qwen/qwen3_4b";

pub(crate) fn register(catalog: &mut crate::Catalog) {
    catalog.register_descriptor(descriptor());
    catalog.register_model_variant(identity(), variant_descriptor());
}

pub(crate) fn identity() -> ModelIdentity {
    super::qwen3_4b_identity()
}

pub(crate) fn checkpoint() -> ModelCheckpoint {
    ModelCheckpoint {
        format: CheckpointFormat::Safetensors,
        source: ArtifactSource::HuggingFace {
            repo: "Qwen/Qwen3-4B",
        },
        include: vec![
            ArtifactRule::Exact("config.json"),
            ArtifactRule::Exact("tokenizer.json"),
            ArtifactRule::Exact("tokenizer_config.json"),
            ArtifactRule::Exact("generation_config.json"),
            ArtifactRule::Exact("special_tokens_map.json"),
            ArtifactRule::Suffix(".safetensors"),
            ArtifactRule::Suffix(".safetensors.index.json"),
        ],
        quantization: None,
    }
}

pub fn descriptor() -> BundleDescriptor {
    let identity = identity();
    let checkpoint = checkpoint();
    BundleDescriptor {
        id: BundleId::new("qwen3_4b"),
        model_id: identity.id.clone(),
        display_name: identity.display_name.clone(),
        family: identity.family,
        capabilities: identity.capabilities,
        backend: BackendKind::MistralRs,
        requirements: BundleRequirements {
            platform: identity.requirements.platform,
            build: vec![BuildConstraint::Feature("backend-mistral".into())],
        },
        eval_tracks: identity.eval_tracks,
        artifacts: Some(crate::bundle_artifacts_from_checkpoint(
            "qwen3_4b",
            &checkpoint,
        )),
    }
}

pub(crate) fn variant_descriptor() -> crate::ModelVariantDescriptor {
    let spec = MistralTextSpec::qwen3_4b();
    crate::ModelVariantDescriptor {
        backend: BackendKind::MistralRs,
        capabilities: spec.capabilities,
        quantization: spec.quantization,
        checkpoint: checkpoint(),
    }
}

pub fn bundle() -> crate::CuratedBundle {
    crate::CuratedBundle::Qwen3_4B
}

pub async fn start(options: StartOptions) -> Result<MistralTextHandle, ModelError> {
    MistralTextBundle::new(MistralTextSpec::qwen3_4b())
        .start(crate::resolve_typed_artifact_policy(
            options,
            resolve_local_snapshot_root,
        )?)
        .await
}

fn resolve_local_snapshot_root(root: &Path) -> Result<PathBuf, ModelError> {
    crate::resolve_hf_snapshot("Qwen/Qwen3-4B", root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BundleFamily, Catalog};
    use motlie_model::CapabilityDescriptor;
    use motlie_model::eval::EvalTrack;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn descriptor_is_reviewable_as_data() {
        let descriptor = descriptor();

        assert_eq!(descriptor.id.as_str(), "qwen3_4b");
        assert_eq!(descriptor.display_name, "Qwen3 4B");
        assert_eq!(descriptor.family, BundleFamily::Qwen);
        assert_eq!(descriptor.backend, BackendKind::MistralRs);
        assert!(descriptor.eval_tracks.contains(&EvalTrack::Chat));
        assert!(descriptor.eval_tracks.contains(&EvalTrack::Reasoning));
        assert!(
            descriptor
                .capabilities
                .supports(motlie_model::CapabilityKind::Chat)
        );
        assert!(
            descriptor
                .capabilities
                .supports(motlie_model::CapabilityKind::Completion)
        );
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
        assert_eq!(artifacts.control_name, "qwen3_4b");
        assert!(artifacts.includes("config.json"));
        assert!(artifacts.includes("model-00001-of-00002.safetensors"));
        assert!(!artifacts.includes("README.md"));
    }

    #[test]
    fn default_catalog_includes_qwen3_bundle() {
        let catalog = Catalog::with_defaults();
        let bundle_id = BundleId::new("qwen3_4b");

        #[cfg(feature = "model-qwen3-4b")]
        {
            assert!(catalog.instantiate(&bundle_id).is_some());
            assert!(
                catalog
                    .bundles_for_track(EvalTrack::Chat)
                    .any(|b| b.id == bundle_id)
            );
        }
    }

    #[test]
    fn local_snapshot_resolution_rejects_missing_cache() {
        let root = unique_temp_dir();
        std::fs::create_dir_all(&root).expect("temp root should be creatable");

        let error =
            resolve_local_snapshot_root(&root).expect_err("missing local cache should fail closed");

        assert!(matches!(
            error,
            ModelError::InvalidConfiguration(message) if message.contains("config.json")
        ));
    }

    #[test]
    fn local_snapshot_resolution_accepts_complete_hf_cache_layout() {
        let root = unique_temp_dir();
        let snapshot = create_fake_hf_cache(&root, "Qwen/Qwen3-4B", "main");

        std::fs::write(snapshot.join("config.json"), "{}").expect("config should be writable");
        std::fs::write(snapshot.join("tokenizer.json"), "{}")
            .expect("tokenizer should be writable");
        std::fs::write(snapshot.join("model-00001-of-00002.safetensors"), "stub")
            .expect("weights should be writable");

        let resolved = resolve_local_snapshot_root(&root)
            .expect("complete cache layout should resolve to snapshot path");

        assert_eq!(resolved, snapshot);
    }

    fn unique_temp_dir() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic enough")
            .as_nanos();
        std::env::temp_dir().join(format!("motlie-models-qwen3-test-{unique}"))
    }

    fn create_fake_hf_cache(root: &Path, model_id: &str, revision: &str) -> PathBuf {
        let repo_folder = format!("models--{}", model_id.replace('/', "--"));
        let repo_root = root.join(repo_folder);
        let refs_dir = repo_root.join("refs");
        let snapshots_dir = repo_root.join("snapshots");
        let commit = "test-commit";
        let snapshot = snapshots_dir.join(commit);

        std::fs::create_dir_all(&snapshot).expect("snapshot dir should be creatable");
        std::fs::create_dir_all(&refs_dir).expect("refs dir should be creatable");
        std::fs::write(refs_dir.join(revision), commit).expect("ref file should be writable");

        snapshot
    }
}
