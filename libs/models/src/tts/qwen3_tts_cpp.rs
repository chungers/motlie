use std::path::{Path, PathBuf};

use motlie_model::eval::EvalTrack;
use motlie_model::{
    BundleId, CheckpointFormat, CheckpointQuantization, ModelCheckpoint, ModelError, ModelIdentity,
    StartOptions,
};
use motlie_model_qwen3_tts_cpp::{
    Qwen3TtsCppHandle, Qwen3TtsCppSpeechBundle, Qwen3TtsCppSpeechSpec,
};

use crate::LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX;
use crate::{
    ArtifactRule, ArtifactSource, BackendKind, BuildConstraint, BundleDescriptor, BundleFamily,
    BundleRequirements, PlatformConstraint,
};

pub const SELECTOR: &str = "qwen/qwen3_tts_cpp_0_6b";

const HF_REPO: &str = "koboldcpp/tts";
const MODEL_FILE_Q8_0: &str = "qwen3-tts-0.6b-q8_0.gguf";
const MODEL_FILE_F16: &str = "qwen3-tts-0.6b-f16.gguf";
const TOKENIZER_FILE_F16: &str = "qwen3-tts-tokenizer-f16.gguf";

pub(crate) fn register(catalog: &mut crate::Catalog) {
    catalog.register_descriptor(descriptor());
    catalog.register_model_variant(identity(), variant_descriptor());
}

pub(crate) fn identity() -> ModelIdentity {
    ModelIdentity {
        id: BundleId::new("qwen3_tts_cpp_0_6b"),
        display_name: "Qwen3-TTS CPP 0.6B".into(),
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
        format: CheckpointFormat::Gguf,
        source: ArtifactSource::HuggingFace { repo: HF_REPO },
        include: vec![
            ArtifactRule::Exact(MODEL_FILE_Q8_0),
            ArtifactRule::Exact(TOKENIZER_FILE_F16),
        ],
        quantization: Some(CheckpointQuantization::Gguf {
            label: "Q8_0".into(),
        }),
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
        backend: BackendKind::Qwen3TtsCpp,
        requirements: BundleRequirements {
            platform: identity.requirements.platform,
            build: vec![BuildConstraint::Feature("backend-qwen3-tts-cpp".into())],
        },
        eval_tracks: identity.eval_tracks,
        artifacts: Some(crate::bundle_artifacts_from_checkpoint(
            "qwen3_tts_cpp_0_6b",
            &checkpoint,
        )),
    }
}

pub(crate) fn variant_descriptor() -> crate::ModelVariantDescriptor {
    let spec = Qwen3TtsCppSpeechSpec::qwen3_tts_cpp_0_6b();
    crate::ModelVariantDescriptor {
        backend: BackendKind::Qwen3TtsCpp,
        capabilities: spec.capabilities,
        quantization: spec.quantization,
        checkpoint: checkpoint(),
    }
}

pub fn typed_bundle() -> Qwen3TtsCppSpeechBundle {
    Qwen3TtsCppSpeechBundle::new(Qwen3TtsCppSpeechSpec::qwen3_tts_cpp_0_6b())
}

pub async fn start_typed(options: StartOptions) -> Result<Qwen3TtsCppHandle, ModelError> {
    typed_bundle()
        .start_typed(crate::resolve_typed_artifact_policy(
            options,
            resolve_local_model_path,
        )?)
        .await
}

fn resolve_local_model_path(root: &Path) -> Result<PathBuf, ModelError> {
    if root.join(TOKENIZER_FILE_F16).is_file()
        && (root.join(MODEL_FILE_Q8_0).is_file() || root.join(MODEL_FILE_F16).is_file())
    {
        return Ok(root.to_path_buf());
    }

    let repo_folder = format!("models--{}", HF_REPO.replace('/', "--"));
    let repo_root = root.join(&repo_folder);
    let refs_dir = repo_root.join("refs");
    let main_ref = refs_dir.join("main");

    if !main_ref.exists() {
        return Err(ModelError::InvalidConfiguration(format!(
            "{LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX} requires cached GGUF artifacts for `{HF_REPO}` under `{}` or a direct model directory; no refs/main found",
            root.display()
        )));
    }

    let commit = std::fs::read_to_string(&main_ref).map_err(|err| {
        ModelError::InvalidConfiguration(format!(
            "failed to read HF cache ref for `{HF_REPO}`: {err}"
        ))
    })?;
    let snapshot_dir = repo_root.join("snapshots").join(commit.trim());

    if !snapshot_dir.exists() {
        return Err(ModelError::InvalidConfiguration(format!(
            "HF cache snapshot for `{HF_REPO}` not found under `{}`",
            root.display()
        )));
    }

    if !snapshot_dir.join(TOKENIZER_FILE_F16).is_file() {
        return Err(ModelError::InvalidConfiguration(format!(
            "{LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX} requires `{TOKENIZER_FILE_F16}` in cached snapshot for `{HF_REPO}` under `{}`",
            root.display()
        )));
    }

    if !snapshot_dir.join(MODEL_FILE_Q8_0).is_file() && !snapshot_dir.join(MODEL_FILE_F16).is_file()
    {
        return Err(ModelError::InvalidConfiguration(format!(
            "{LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX} requires `{MODEL_FILE_Q8_0}` or `{MODEL_FILE_F16}` in cached snapshot for `{HF_REPO}` under `{}`",
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
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn descriptor_is_reviewable_as_data() {
        let descriptor = descriptor();

        assert_eq!(descriptor.id.as_str(), "qwen3_tts_cpp_0_6b");
        assert_eq!(descriptor.display_name, "Qwen3-TTS CPP 0.6B");
        assert_eq!(descriptor.family, BundleFamily::Qwen);
        assert_eq!(descriptor.backend, BackendKind::Qwen3TtsCpp);
        assert!(descriptor.capabilities.supports(CapabilityKind::Speech));
        assert_eq!(descriptor.eval_tracks, vec![EvalTrack::Speech]);

        let artifacts = descriptor
            .artifacts
            .expect("descriptor should expose curated artifact control");
        assert_eq!(artifacts.control_name, "qwen3_tts_cpp_0_6b");
        assert!(artifacts.includes(MODEL_FILE_Q8_0));
        assert!(artifacts.includes(TOKENIZER_FILE_F16));
    }

    #[test]
    fn default_catalog_includes_bundle_when_feature_enabled() {
        let catalog = Catalog::with_defaults();
        let bundle_id = BundleId::new("qwen3_tts_cpp_0_6b");

        #[cfg(feature = "model-qwen3-tts-cpp")]
        {
            assert!(catalog.instantiate(&bundle_id).is_some());
            assert!(
                catalog
                    .bundles_for_track(EvalTrack::Speech)
                    .any(|bundle| bundle.id == bundle_id)
            );
        }
    }

    #[test]
    fn local_resolution_accepts_direct_model_dir() {
        let root = unique_temp_dir();
        std::fs::create_dir_all(&root).expect("temp root should exist");
        std::fs::write(root.join(MODEL_FILE_F16), "stub").expect("model should be writable");
        std::fs::write(root.join(TOKENIZER_FILE_F16), "stub")
            .expect("tokenizer should be writable");

        let resolved = resolve_local_model_path(&root).expect("direct model dir should resolve");

        assert_eq!(resolved, root);
        std::fs::remove_dir_all(resolved).ok();
    }

    fn unique_temp_dir() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("motlie-qwen3-tts-cpp-{unique}"))
    }
}
