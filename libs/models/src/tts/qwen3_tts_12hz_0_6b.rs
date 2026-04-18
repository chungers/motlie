use std::path::{Path, PathBuf};

use motlie_model::eval::EvalTrack;
use motlie_model::{
    BundleId, CheckpointFormat, ModelCheckpoint, ModelError, ModelIdentity, StartOptions,
};
use motlie_model_qwen3_tts::{Qwen3TtsHandle, Qwen3TtsSpeechBundle, Qwen3TtsSpeechSpec};

use crate::{
    ArtifactRule, ArtifactSource, BackendKind, BuildConstraint, BundleDescriptor, BundleFamily,
    BundleRequirements, PlatformConstraint,
};

pub const SELECTOR: &str = "qwen/qwen3_tts_12hz_0_6b";

const HF_REPO: &str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base";

/// ONNX-exported model components expected under the checkpoint root.
///
/// The upstream HuggingFace model uses safetensors format. These ONNX files
/// must be produced by an offline export step before the curated bundle can
/// start. See `libs/models/docs/DESIGN_TTS.md` Phase 2 for the export guide.
const ENCODER_FILE: &str = "encoder.onnx";
const DECODER_FILE: &str = "decoder.onnx";
const VOCODER_FILE: &str = "vocoder.onnx";
const CONFIG_FILE: &str = "config.json";
/// Custom tokenizer vocabulary produced by the ONNX export step. This is NOT
/// the upstream `vocab.json` from the HuggingFace model tree — it is a
/// flattened token→ID mapping derived from the full BPE tokenizer
/// (`vocab.json` + `merges.txt` + `tokenizer_config.json`) during export.
/// The greedy longest-match tokenizer in `motlie-model-qwen3-tts` consumes
/// this flattened format directly without needing the BPE merge rules.
const VOCAB_FILE: &str = "vocab.json";

pub(crate) fn register(catalog: &mut crate::Catalog) {
    catalog.register_descriptor(descriptor());
    catalog.register_model_variant(identity(), variant_descriptor());
}

pub(crate) fn identity() -> ModelIdentity {
    ModelIdentity {
        id: BundleId::new("qwen3_tts_12hz_0_6b"),
        display_name: "Qwen3-TTS 12Hz 0.6B".into(),
        family: BundleFamily::Qwen,
        capabilities: motlie_model::Capabilities::speech_stream_only(),
        eval_tracks: vec![EvalTrack::Speech],
        requirements: BundleRequirements {
            platform: vec![PlatformConstraint::Linux, PlatformConstraint::Macos],
            build: vec![],
        },
    }
}

pub(crate) fn checkpoint() -> ModelCheckpoint {
    ModelCheckpoint {
        format: CheckpointFormat::Onnx,
        source: ArtifactSource::HuggingFace { repo: HF_REPO },
        include: vec![
            ArtifactRule::Exact(ENCODER_FILE),
            ArtifactRule::Exact(DECODER_FILE),
            ArtifactRule::Exact(VOCODER_FILE),
            ArtifactRule::Exact(CONFIG_FILE),
            ArtifactRule::Exact(VOCAB_FILE),
        ],
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
        backend: BackendKind::Ort,
        requirements: BundleRequirements {
            platform: identity.requirements.platform,
            build: vec![BuildConstraint::Feature("backend-qwen3-tts".into())],
        },
        eval_tracks: identity.eval_tracks,
        artifacts: Some(crate::bundle_artifacts_from_checkpoint(
            "qwen3_tts_12hz_0_6b",
            &checkpoint,
        )),
    }
}

pub(crate) fn variant_descriptor() -> crate::ModelVariantDescriptor {
    let spec = Qwen3TtsSpeechSpec::qwen3_tts_12hz_0_6b();
    crate::ModelVariantDescriptor {
        backend: BackendKind::Ort,
        capabilities: spec.capabilities,
        quantization: spec.quantization,
        checkpoint: checkpoint(),
    }
}

pub fn typed_bundle() -> Qwen3TtsSpeechBundle {
    Qwen3TtsSpeechBundle::new(Qwen3TtsSpeechSpec::qwen3_tts_12hz_0_6b())
}

pub async fn start_typed(options: StartOptions) -> Result<Qwen3TtsHandle, ModelError> {
    typed_bundle()
        .start_typed(crate::resolve_typed_artifact_policy(
            options,
            resolve_local_model_path,
        )?)
        .await
}

fn resolve_local_model_path(root: &Path) -> Result<PathBuf, ModelError> {
    if [
        ENCODER_FILE,
        DECODER_FILE,
        VOCODER_FILE,
        CONFIG_FILE,
        VOCAB_FILE,
    ]
    .into_iter()
    .all(|filename| root.join(filename).is_file())
    {
        return Ok(root.to_path_buf());
    }

    let repo_folder = format!("models--{}", HF_REPO.replace('/', "--"));
    let repo_root = root.join(&repo_folder);
    let refs_dir = repo_root.join("refs");
    let main_ref = refs_dir.join("main");

    if !main_ref.exists() {
        return Err(ModelError::InvalidConfiguration(format!(
            "artifact policy `LocalOnly` requires cached ONNX artifacts for `{HF_REPO}` under `{}`; \
             no refs/main found — run the download and ONNX export steps first",
            root.display()
        )));
    }

    let commit = std::fs::read_to_string(&main_ref).map_err(|err| {
        ModelError::InvalidConfiguration(format!(
            "failed to read HF cache ref for `{HF_REPO}`: {err}"
        ))
    })?;
    let commit = commit.trim();

    let snapshot_dir = repo_root.join("snapshots").join(commit);
    if !snapshot_dir.exists() {
        return Err(ModelError::InvalidConfiguration(format!(
            "HF cache snapshot `{commit}` for `{HF_REPO}` not found under `{}`",
            root.display()
        )));
    }

    for filename in [
        ENCODER_FILE,
        DECODER_FILE,
        VOCODER_FILE,
        CONFIG_FILE,
        VOCAB_FILE,
    ] {
        let path = snapshot_dir.join(filename);
        if !path.exists() {
            return Err(ModelError::InvalidConfiguration(format!(
                "artifact policy `LocalOnly` requires `{filename}` in cached snapshot for \
                 `{HF_REPO}` under `{}`; ensure ONNX export has been performed",
                root.display()
            )));
        }
    }

    // Return the snapshot directory — the backend resolves individual files within it.
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

        assert_eq!(descriptor.id.as_str(), "qwen3_tts_12hz_0_6b");
        assert_eq!(descriptor.display_name, "Qwen3-TTS 12Hz 0.6B");
        assert_eq!(descriptor.family, BundleFamily::Qwen);
        assert_eq!(descriptor.backend, BackendKind::Ort);
        assert!(descriptor.capabilities.supports(CapabilityKind::Speech));
        assert!(!descriptor.capabilities.supports(CapabilityKind::Chat));
        assert_eq!(descriptor.eval_tracks, vec![EvalTrack::Speech]);

        let artifacts = descriptor
            .artifacts
            .expect("descriptor should expose curated artifact control");
        assert_eq!(artifacts.control_name, "qwen3_tts_12hz_0_6b");
        assert!(artifacts.includes("encoder.onnx"));
        assert!(artifacts.includes("decoder.onnx"));
        assert!(artifacts.includes("vocoder.onnx"));
        assert!(artifacts.includes("config.json"));
        assert!(artifacts.includes("vocab.json"));
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
    fn default_catalog_includes_bundle_when_feature_enabled() {
        let catalog = Catalog::with_defaults();
        let bundle_id = BundleId::new("qwen3_tts_12hz_0_6b");

        #[cfg(feature = "model-qwen3-tts-0_6b")]
        {
            assert!(catalog.instantiate(&bundle_id).is_none());
            assert!(
                catalog
                    .bundles_for_track(EvalTrack::Speech)
                    .any(|b| b.id == bundle_id)
            );
        }
    }

    #[test]
    fn local_resolution_rejects_missing_cache() {
        let root = unique_temp_dir();
        std::fs::create_dir_all(&root).expect("temp root should be creatable");

        let error = resolve_local_model_path(&root).expect_err("missing cache should fail closed");

        assert!(matches!(
            error,
            ModelError::InvalidConfiguration(message) if message.contains("refs/main")
        ));
    }

    #[test]
    fn local_resolution_rejects_missing_onnx_components() {
        let root = unique_temp_dir();
        let _snapshot = create_fake_hf_cache(&root);
        // Snapshot exists but has no ONNX files.

        let error =
            resolve_local_model_path(&root).expect_err("missing ONNX components should fail");

        assert!(matches!(
            error,
            ModelError::InvalidConfiguration(message) if message.contains("ONNX export")
        ));

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn local_resolution_accepts_complete_onnx_artifacts() {
        let root = unique_temp_dir();
        let snapshot = create_fake_hf_cache(&root);

        for filename in [
            ENCODER_FILE,
            DECODER_FILE,
            VOCODER_FILE,
            CONFIG_FILE,
            VOCAB_FILE,
        ] {
            std::fs::write(snapshot.join(filename), "stub").expect("stub file should be writable");
        }

        let resolved =
            resolve_local_model_path(&root).expect("complete ONNX artifacts should resolve");

        assert_eq!(resolved, snapshot);
        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn local_resolution_accepts_direct_artifact_dir() {
        let root = unique_temp_dir();
        std::fs::create_dir_all(&root).expect("temp root should exist");

        for filename in [
            ENCODER_FILE,
            DECODER_FILE,
            VOCODER_FILE,
            CONFIG_FILE,
            VOCAB_FILE,
        ] {
            std::fs::write(root.join(filename), "stub").expect("stub file should be writable");
        }

        let resolved = resolve_local_model_path(&root).expect("direct dir should resolve");

        assert_eq!(resolved, root);
        std::fs::remove_dir_all(&resolved).ok();
    }

    fn unique_temp_dir() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic enough")
            .as_nanos();
        std::env::temp_dir().join(format!("motlie-models-qwen3-tts-test-{unique}"))
    }

    fn create_fake_hf_cache(root: &Path) -> PathBuf {
        let repo_folder = format!("models--{}", HF_REPO.replace('/', "--"));
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
