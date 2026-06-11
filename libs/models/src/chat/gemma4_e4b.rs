use std::path::{Path, PathBuf};

use motlie_model::{
    BundleId, CheckpointFormat, ModelBundle, ModelCheckpoint, ModelError, ModelIdentity,
    StartOptions,
};
use motlie_model_mistral::{
    MistralMultimodalBundle, MistralMultimodalHandle, MistralMultimodalSpec,
};

use crate::{
    ArtifactRule, ArtifactSource, BackendKind, BuildConstraint, BundleDescriptor,
    BundleRequirements,
};

pub const SELECTOR: &str = "google/gemma4_e4b";

pub(crate) fn register(catalog: &mut crate::Catalog) {
    catalog.register_descriptor(descriptor());
    catalog.register_model_variant(identity(), variant_descriptor());
}

pub(crate) fn identity() -> ModelIdentity {
    super::gemma4_e4b_identity()
}

pub(crate) fn checkpoint() -> ModelCheckpoint {
    ModelCheckpoint {
        format: CheckpointFormat::Safetensors,
        source: ArtifactSource::HuggingFace {
            repo: "google/gemma-4-E4B-it",
        },
        include: vec![
            ArtifactRule::Exact("chat_template.jinja"),
            ArtifactRule::Exact("config.json"),
            ArtifactRule::Exact("generation_config.json"),
            ArtifactRule::Exact("tokenizer.json"),
            ArtifactRule::Exact("tokenizer.model"),
            ArtifactRule::Exact("tokenizer_config.json"),
            ArtifactRule::Exact("special_tokens_map.json"),
            ArtifactRule::Exact("preprocessor_config.json"),
            ArtifactRule::Exact("processor_config.json"),
            ArtifactRule::Suffix(".safetensors"),
            ArtifactRule::Suffix(".safetensors.index.json"),
        ],
        quantization: None,
    }
}

pub fn descriptor() -> BundleDescriptor {
    let identity = identity();
    let checkpoint = checkpoint();
    let spec = MistralMultimodalSpec::gemma4_e4b();
    BundleDescriptor {
        id: BundleId::new("gemma4_e4b"),
        model_id: identity.id.clone(),
        display_name: identity.display_name.clone(),
        family: identity.family,
        capabilities: spec.capabilities,
        backend: BackendKind::MistralRs,
        requirements: BundleRequirements {
            platform: identity.requirements.platform,
            build: vec![BuildConstraint::Feature("backend-mistral".into())],
        },
        eval_tracks: identity.eval_tracks,
        artifacts: Some(crate::bundle_artifacts_from_checkpoint(
            "gemma4_e4b",
            &checkpoint,
        )),
    }
}

pub(crate) fn variant_descriptor() -> crate::ModelVariantDescriptor {
    let spec = MistralMultimodalSpec::gemma4_e4b();
    crate::ModelVariantDescriptor {
        backend: BackendKind::MistralRs,
        capabilities: spec.capabilities,
        quantization: spec.quantization,
        checkpoint: checkpoint(),
    }
}

pub fn bundle() -> crate::CuratedBundle {
    crate::CuratedBundle::Gemma4E4B
}

pub async fn start(options: StartOptions) -> Result<MistralMultimodalHandle, ModelError> {
    MistralMultimodalBundle::new(MistralMultimodalSpec::gemma4_e4b())
        .start(crate::resolve_typed_artifact_policy(
            options,
            resolve_local_snapshot_root,
        )?)
        .await
}

fn resolve_local_snapshot_root(root: &Path) -> Result<PathBuf, ModelError> {
    let snapshot = crate::resolve_hf_snapshot("google/gemma-4-E4B-it", root)?;
    let has_processor = snapshot.join("preprocessor_config.json").exists()
        || snapshot.join("processor_config.json").exists();
    if !has_processor {
        return Err(ModelError::InvalidConfiguration(format!(
            "artifact policy `LocalOnly` requires cached multimodal processor config for `google/gemma-4-E4B-it` under `{}`",
            root.display()
        )));
    }
    if !snapshot.join("chat_template.jinja").exists() {
        return Err(ModelError::InvalidConfiguration(format!(
            "artifact policy `LocalOnly` requires `chat_template.jinja` for `google/gemma-4-E4B-it` under `{}`",
            root.display()
        )));
    }
    Ok(snapshot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BundleFamily, Catalog};
    use motlie_model::eval::EvalTrack;
    use motlie_model::{CapabilityDescriptor, CapabilityKind};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn descriptor_is_reviewable_as_data() {
        let descriptor = descriptor();

        assert_eq!(descriptor.id.as_str(), "gemma4_e4b");
        assert_eq!(descriptor.display_name, "Gemma 4 E4B-it");
        assert_eq!(descriptor.family, BundleFamily::Gemma);
        assert_eq!(descriptor.backend, BackendKind::MistralRs);
        assert!(descriptor.capabilities.supports(CapabilityKind::Chat));
        assert!(descriptor.capabilities.supports(CapabilityKind::Vision));
        assert!(descriptor.capabilities.supports(CapabilityKind::ToolUse));
        assert_eq!(
            descriptor.capability_descriptors(),
            &[
                CapabilityDescriptor::multimodal_chat(),
                CapabilityDescriptor::vision(),
                CapabilityDescriptor::tool_use(),
            ]
        );
        let artifacts = descriptor
            .artifacts
            .expect("descriptor should expose curated artifact control");
        assert_eq!(artifacts.control_name, "gemma4_e4b");
        assert!(artifacts.includes("chat_template.jinja"));
        assert!(artifacts.includes("preprocessor_config.json"));
        assert!(artifacts.includes("model-00001-of-00004.safetensors"));
    }

    #[test]
    fn default_catalog_includes_gemma4_e4b_bundle_when_feature_enabled() {
        let catalog = Catalog::with_defaults();
        let bundle_id = BundleId::new("gemma4_e4b");

        #[cfg(feature = "model-gemma4-e4b")]
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
        let snapshot = create_fake_hf_cache(&root, "google/gemma-4-E4B-it", "main");

        std::fs::write(snapshot.join("config.json"), "{}").expect("config should be writable");
        std::fs::write(snapshot.join("tokenizer.json"), "{}")
            .expect("tokenizer should be writable");
        std::fs::write(snapshot.join("preprocessor_config.json"), "{}")
            .expect("preprocessor config should be writable");
        std::fs::write(snapshot.join("chat_template.jinja"), "{{ messages }}")
            .expect("chat template should be writable");
        std::fs::write(snapshot.join("model-00001-of-00004.safetensors"), "stub")
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
        std::env::temp_dir().join(format!("motlie-models-gemma4-e4b-test-{unique}"))
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
