use std::path::{Path, PathBuf};
use std::sync::Arc;

use motlie_model::eval::EvalTrack;
use motlie_model::{
    BundleId, CheckpointFormat, ModelBundle, ModelCheckpoint, ModelError, ModelIdentity,
};
use motlie_model_mistral::MistralMultimodalAdapter;

use crate::{
    ArtifactRule, ArtifactSource, BackendKind, BuildConstraint, BundleDescriptor, BundleFamily,
    BundleRequirements, PlatformConstraint,
};

pub const SELECTOR: &str = "google/gemma4_e2b";

pub(crate) fn register(catalog: &mut crate::Catalog) {
    catalog.register(descriptor(), bundle);
    catalog.register_model_variant(
        identity(),
        checkpoint(),
        Arc::new(resolve_local_snapshot_root),
        Arc::new(MistralMultimodalAdapter::gemma4()),
    );
}

pub(crate) fn identity() -> ModelIdentity {
    ModelIdentity {
        id: BundleId::new("gemma4_e2b"),
        display_name: "Gemma 4 E2B-it".into(),
        family: BundleFamily::Gemma,
        capabilities: motlie_model::Capabilities::multimodal_chat_and_vision(),
        eval_tracks: vec![
            EvalTrack::Chat,
            EvalTrack::Reasoning,
            EvalTrack::Summarization,
            EvalTrack::Classification,
        ],
        requirements: BundleRequirements {
            platform: vec![PlatformConstraint::Linux, PlatformConstraint::Macos],
            build: vec![],
        },
    }
}

pub(crate) fn checkpoint() -> ModelCheckpoint {
    ModelCheckpoint {
        format: CheckpointFormat::Safetensors,
        source: ArtifactSource::HuggingFace {
            repo: "google/gemma-4-E2B-it",
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
    BundleDescriptor {
        id: BundleId::new("gemma4_e2b"),
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
            "gemma4_e2b",
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
        Arc::new(MistralMultimodalAdapter::gemma4()),
        Arc::new(resolve_local_snapshot_root),
    )
}

fn resolve_local_snapshot_root(root: &Path) -> Result<PathBuf, ModelError> {
    let snapshot = crate::resolve_hf_snapshot("google/gemma-4-E2B-it", root)?;
    let has_processor = snapshot.join("preprocessor_config.json").exists()
        || snapshot.join("processor_config.json").exists();
    if !has_processor {
        return Err(ModelError::InvalidConfiguration(format!(
            "artifact policy `LocalOnly` requires cached multimodal processor config for `google/gemma-4-E2B-it` under `{}`",
            root.display()
        )));
    }
    if !snapshot.join("chat_template.jinja").exists() {
        return Err(ModelError::InvalidConfiguration(format!(
            "artifact policy `LocalOnly` requires `chat_template.jinja` for `google/gemma-4-E2B-it` under `{}`",
            root.display()
        )));
    }
    Ok(snapshot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Catalog;
    use motlie_model::CapabilityKind;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn descriptor_is_reviewable_as_data() {
        let descriptor = descriptor();

        assert_eq!(descriptor.id.as_str(), "gemma4_e2b");
        assert_eq!(descriptor.display_name, "Gemma 4 E2B-it");
        assert_eq!(descriptor.family, BundleFamily::Gemma);
        assert_eq!(descriptor.backend, BackendKind::MistralRs);
        assert!(descriptor.capabilities.supports(CapabilityKind::Chat));
        assert!(descriptor.capabilities.supports(CapabilityKind::Vision));
        let artifacts = descriptor
            .artifacts
            .expect("descriptor should expose curated artifact control");
        assert_eq!(artifacts.control_name, "gemma4_e2b");
        assert!(artifacts.includes("chat_template.jinja"));
        assert!(artifacts.includes("preprocessor_config.json"));
        assert!(artifacts.includes("model-00001-of-00002.safetensors"));
    }

    #[test]
    fn identity_exposes_logical_multimodal_capabilities_only() {
        let identity = identity();

        assert!(identity.capabilities.supports(CapabilityKind::Chat));
        assert!(identity.capabilities.supports(CapabilityKind::Vision));
        assert!(!identity.capabilities.supports(CapabilityKind::Completion));
    }

    #[test]
    fn default_catalog_includes_gemma4_bundle() {
        let catalog = Catalog::with_defaults();
        let bundle_id = BundleId::new("gemma4_e2b");

        #[cfg(feature = "model-gemma4-e2b")]
        {
            assert!(catalog.instantiate(&bundle_id).is_some());
            assert!(catalog
                .bundles_for_track(EvalTrack::Chat)
                .any(|b| b.id == bundle_id));
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
        let snapshot = create_fake_hf_cache(&root, "google/gemma-4-E2B-it", "main");

        std::fs::write(snapshot.join("config.json"), "{}").expect("config should be writable");
        std::fs::write(snapshot.join("tokenizer.json"), "{}")
            .expect("tokenizer should be writable");
        std::fs::write(snapshot.join("preprocessor_config.json"), "{}")
            .expect("preprocessor config should be writable");
        std::fs::write(snapshot.join("chat_template.jinja"), "{{ messages }}")
            .expect("chat template should be writable");
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
        std::env::temp_dir().join(format!("motlie-models-gemma4-test-{unique}"))
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
