use std::path::{Path, PathBuf};

use motlie_model::eval::EvalTrack;
use motlie_model::{
    BundleId, CheckpointFormat, ModelCheckpoint, ModelError, ModelIdentity, StartOptions,
};
use motlie_model_whisper_cpp::{
    WhisperCppHandle, WhisperCppTranscriptionBundle, WhisperCppTranscriptionSpec,
};

use crate::{
    ArtifactRule, ArtifactSource, BackendKind, BuildConstraint, BundleDescriptor, BundleFamily,
    BundleRequirements, PlatformConstraint,
};

pub const SELECTOR: &str = "openai/whisper_base_en";

pub(crate) fn register(catalog: &mut crate::Catalog) {
    catalog.register_descriptor(descriptor());
    catalog.register_model_variant(identity(), variant_descriptor());
}

pub(crate) fn identity() -> ModelIdentity {
    ModelIdentity {
        id: BundleId::new("whisper_base_en"),
        display_name: "Whisper Base.en".into(),
        family: BundleFamily::Whisper,
        capabilities: motlie_model::Capabilities::transcription_stream_only(),
        eval_tracks: vec![EvalTrack::Transcription],
        requirements: BundleRequirements {
            platform: vec![PlatformConstraint::Linux, PlatformConstraint::Macos],
            build: vec![],
        },
    }
}

pub(crate) fn checkpoint() -> ModelCheckpoint {
    ModelCheckpoint {
        format: CheckpointFormat::Ggml,
        source: ArtifactSource::HuggingFace {
            repo: "ggerganov/whisper.cpp",
        },
        include: vec![ArtifactRule::Exact("ggml-base.en.bin")],
        quantization: None,
    }
}

pub fn descriptor() -> BundleDescriptor {
    let identity = identity();
    let checkpoint = checkpoint();
    BundleDescriptor {
        id: identity.id.clone(),
        model_id: identity.id,
        display_name: identity.display_name.clone(),
        family: identity.family,
        capabilities: identity.capabilities,
        backend: BackendKind::WhisperCpp,
        requirements: BundleRequirements {
            platform: identity.requirements.platform,
            build: vec![BuildConstraint::Feature("backend-whisper-cpp".into())],
        },
        eval_tracks: identity.eval_tracks,
        artifacts: Some(crate::bundle_artifacts_from_checkpoint(
            "whisper_base_en",
            &checkpoint,
        )),
    }
}

pub(crate) fn variant_descriptor() -> crate::ModelVariantDescriptor {
    let spec = WhisperCppTranscriptionSpec::whisper_base_en();
    crate::ModelVariantDescriptor {
        backend: BackendKind::WhisperCpp,
        capabilities: spec.capabilities,
        quantization: spec.quantization,
        checkpoint: checkpoint(),
    }
}

pub fn typed_bundle() -> WhisperCppTranscriptionBundle {
    WhisperCppTranscriptionBundle::new(WhisperCppTranscriptionSpec::whisper_base_en())
}

pub async fn start_typed(options: StartOptions) -> Result<WhisperCppHandle, ModelError> {
    typed_bundle()
        .start_typed(crate::resolve_typed_artifact_policy(
            options,
            resolve_local_ggml_root,
        )?)
        .await
}

fn resolve_local_ggml_root(root: &Path) -> Result<PathBuf, ModelError> {
    if root.join("ggml-base.en.bin").is_file() {
        return Ok(root.to_path_buf());
    }

    // For ggml single-file artifacts, the HF cache layout puts files in a
    // snapshot directory. We navigate refs/main → snapshots/{commit}/.
    let repo_folder = "models--ggerganov--whisper.cpp";
    let repo_root = root.join(repo_folder);
    let refs_dir = repo_root.join("refs");
    let main_ref = refs_dir.join("main");

    if !main_ref.exists() {
        return Err(ModelError::InvalidConfiguration(format!(
            "artifact policy `LocalOnly` requires cached ggml artifacts for `ggerganov/whisper.cpp` under `{}`; \
             no refs/main found — run the download step first",
            root.display()
        )));
    }

    let commit = std::fs::read_to_string(&main_ref).map_err(|err| {
        ModelError::InvalidConfiguration(format!(
            "failed to read HF cache ref for `ggerganov/whisper.cpp`: {err}"
        ))
    })?;
    let commit = commit.trim();

    let snapshot_dir = repo_root.join("snapshots").join(commit);
    if !snapshot_dir.exists() {
        return Err(ModelError::InvalidConfiguration(format!(
            "HF cache snapshot `{commit}` for `ggerganov/whisper.cpp` not found under `{}`",
            root.display()
        )));
    }

    let model_file = snapshot_dir.join("ggml-base.en.bin");
    if !model_file.exists() {
        return Err(ModelError::InvalidConfiguration(format!(
            "artifact policy `LocalOnly` requires `ggml-base.en.bin` in cached snapshot for `ggerganov/whisper.cpp` under `{}`",
            root.display()
        )));
    }

    Ok(snapshot_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Catalog;
    use motlie_model::CapabilityKind;
    use motlie_model::ModelBundle;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn descriptor_is_reviewable_as_data() {
        let descriptor = descriptor();

        assert_eq!(descriptor.id.as_str(), "whisper_base_en");
        assert_eq!(descriptor.display_name, "Whisper Base.en");
        assert_eq!(descriptor.family, BundleFamily::Whisper);
        assert_eq!(descriptor.backend, BackendKind::WhisperCpp);
        assert!(
            descriptor
                .capabilities
                .supports(CapabilityKind::Transcription)
        );
        assert!(!descriptor.capabilities.supports(CapabilityKind::Chat));
        assert_eq!(descriptor.eval_tracks, vec![EvalTrack::Transcription]);

        let artifacts = descriptor
            .artifacts
            .expect("descriptor should expose curated artifact control");
        assert_eq!(artifacts.control_name, "whisper_base_en");
        assert!(artifacts.includes("ggml-base.en.bin"));
        assert!(!artifacts.includes("README.md"));
    }

    #[test]
    fn identity_matches_descriptor_core_fields() {
        let identity = identity();
        let descriptor = descriptor();

        assert_eq!(identity.id, descriptor.model_id);
        assert_eq!(identity.display_name, descriptor.display_name);
        assert_eq!(identity.family, descriptor.family);
    }

    #[test]
    fn quantization_is_explicitly_none() {
        let bundle = typed_bundle();

        assert_eq!(
            bundle.metadata().quantization,
            motlie_model::QuantizationSupport::none()
        );
    }

    #[test]
    fn default_catalog_includes_whisper_bundle_when_feature_enabled() {
        let catalog = Catalog::with_defaults();
        let bundle_id = BundleId::new("whisper_base_en");

        #[cfg(feature = "model-whisper-base-en")]
        {
            assert!(catalog.instantiate(&bundle_id).is_some());
            assert!(
                catalog
                    .bundles_for_track(EvalTrack::Transcription)
                    .any(|b| b.id == bundle_id)
            );
        }
    }

    #[test]
    fn local_ggml_resolution_rejects_missing_cache() {
        let root = unique_temp_dir();
        std::fs::create_dir_all(&root).expect("temp root should be creatable");

        let error = resolve_local_ggml_root(&root).expect_err("missing cache should fail closed");

        assert!(matches!(
            error,
            ModelError::InvalidConfiguration(message) if message.contains("refs/main")
        ));
    }

    #[test]
    fn local_ggml_resolution_accepts_cache_with_model_file() {
        let root = unique_temp_dir();
        let snapshot = create_fake_hf_cache(&root);
        std::fs::write(snapshot.join("ggml-base.en.bin"), "stub")
            .expect("model stub should be writable");

        let resolved =
            resolve_local_ggml_root(&root).expect("cache with model file should resolve");

        assert_eq!(resolved, snapshot);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn local_ggml_resolution_accepts_direct_artifact_dir() {
        let root = unique_temp_dir();
        std::fs::create_dir_all(&root).expect("temp root should be creatable");
        std::fs::write(root.join("ggml-base.en.bin"), "stub").expect("model should be writable");

        let resolved = resolve_local_ggml_root(&root).expect("direct artifact dir should resolve");

        assert_eq!(resolved, root);
        std::fs::remove_dir_all(&resolved).ok();
    }

    fn unique_temp_dir() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic enough")
            .as_nanos();
        std::env::temp_dir().join(format!("motlie-models-whisper-test-{unique}"))
    }

    fn create_fake_hf_cache(root: &Path) -> PathBuf {
        let repo_folder = "models--ggerganov--whisper.cpp";
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
