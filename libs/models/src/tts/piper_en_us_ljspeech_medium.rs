use std::path::{Path, PathBuf};

use motlie_model::eval::EvalTrack;
use motlie_model::{
    BundleId, CheckpointFormat, ModelCheckpoint, ModelError, ModelIdentity, StartOptions,
};
use motlie_model_piper::{PiperHandle, PiperSpeechBundle, PiperSpeechSpec};

use crate::{
    ArtifactRule, ArtifactSource, BackendKind, BuildConstraint, BundleDescriptor, BundleFamily,
    BundleRequirements, PlatformConstraint,
};

pub const SELECTOR: &str = "piper/en_us_ljspeech_medium";

const HF_REPO: &str = "rhasspy/piper-voices";
const MODEL_FILE: &str = "en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx";
const CONFIG_FILE: &str = "en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx.json";

pub(crate) fn register(catalog: &mut crate::Catalog) {
    catalog.register_descriptor(descriptor());
    catalog.register_model_variant(identity(), variant_descriptor());
}

pub(crate) fn identity() -> ModelIdentity {
    ModelIdentity {
        id: BundleId::new("piper_en_us_ljspeech_medium"),
        display_name: "Piper en_US ljspeech medium".into(),
        family: BundleFamily::Piper,
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
            ArtifactRule::Exact(MODEL_FILE),
            ArtifactRule::Exact(CONFIG_FILE),
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
            build: vec![BuildConstraint::Feature("backend-piper".into())],
        },
        eval_tracks: identity.eval_tracks,
        artifacts: Some(crate::bundle_artifacts_from_checkpoint(
            "piper_en_us_ljspeech_medium",
            &checkpoint,
        )),
    }
}

pub(crate) fn variant_descriptor() -> crate::ModelVariantDescriptor {
    let spec = PiperSpeechSpec::en_us_ljspeech_medium();
    crate::ModelVariantDescriptor {
        backend: BackendKind::Ort,
        capabilities: spec.capabilities,
        quantization: spec.quantization,
        checkpoint: checkpoint(),
    }
}

pub fn typed_bundle() -> PiperSpeechBundle {
    PiperSpeechBundle::new(PiperSpeechSpec::en_us_ljspeech_medium())
}

pub async fn start_typed(options: StartOptions) -> Result<PiperHandle, ModelError> {
    typed_bundle()
        .start_typed(crate::resolve_typed_artifact_policy(
            options,
            resolve_local_model_path,
        )?)
        .await
}

fn resolve_local_model_path(root: &Path) -> Result<PathBuf, ModelError> {
    let model_parent = Path::new(MODEL_FILE).parent().ok_or_else(|| {
        ModelError::InvalidConfiguration(format!("invalid curated Piper model path `{MODEL_FILE}`"))
    })?;
    let config_parent = Path::new(CONFIG_FILE).parent().ok_or_else(|| {
        ModelError::InvalidConfiguration(format!(
            "invalid curated Piper config path `{CONFIG_FILE}`"
        ))
    })?;
    if model_parent != config_parent {
        return Err(ModelError::InvalidConfiguration(format!(
            "curated Piper model/config paths diverge: `{MODEL_FILE}` vs `{CONFIG_FILE}`"
        )));
    }

    if [MODEL_FILE, CONFIG_FILE]
        .into_iter()
        .all(|filename| root.join(filename).is_file())
    {
        return Ok(root.join(model_parent));
    }

    let direct_model = root.join(Path::new(MODEL_FILE).file_name().ok_or_else(|| {
        ModelError::InvalidConfiguration(format!("invalid curated Piper model path `{MODEL_FILE}`"))
    })?);
    let direct_config = root.join(Path::new(CONFIG_FILE).file_name().ok_or_else(|| {
        ModelError::InvalidConfiguration(format!(
            "invalid curated Piper config path `{CONFIG_FILE}`"
        ))
    })?);
    if direct_model.is_file() && direct_config.is_file() {
        return Ok(root.to_path_buf());
    }

    let repo_folder = format!("models--{}", HF_REPO.replace('/', "--"));
    let repo_root = root.join(&repo_folder);
    let refs_dir = repo_root.join("refs");
    let main_ref = refs_dir.join("main");

    if !main_ref.exists() {
        return Err(ModelError::InvalidConfiguration(format!(
            "artifact policy `LocalOnly` requires cached ONNX artifacts for `{HF_REPO}` under `{}`; no refs/main found",
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

    for filename in [MODEL_FILE, CONFIG_FILE] {
        let path = snapshot_dir.join(filename);
        if !path.exists() {
            return Err(ModelError::InvalidConfiguration(format!(
                "artifact policy `LocalOnly` requires `{filename}` in cached snapshot for `{HF_REPO}` under `{}`",
                root.display()
            )));
        }
    }

    Ok(snapshot_dir.join(model_parent))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Catalog;
    use motlie_model::typed::SynthesisRequest;
    use motlie_model::{ArtifactPolicy, StartOptions};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn descriptor_is_reviewable_as_data() {
        let descriptor = descriptor();

        assert_eq!(descriptor.id.as_str(), "piper_en_us_ljspeech_medium");
        assert_eq!(descriptor.display_name, "Piper en_US ljspeech medium");
        assert_eq!(descriptor.backend, BackendKind::Ort);
        assert!(
            descriptor
                .capabilities
                .supports(motlie_model::CapabilityKind::Speech)
        );
    }

    #[test]
    fn default_catalog_includes_bundle_when_feature_enabled() {
        let catalog = Catalog::with_defaults();
        let bundle_id = BundleId::new("piper_en_us_ljspeech_medium");

        #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
        {
            assert!(catalog.instantiate(&bundle_id).is_some());
            assert!(
                catalog
                    .bundles_for_track(EvalTrack::Speech)
                    .any(|bundle| bundle.id == bundle_id)
            );
        }
    }

    #[tokio::test]
    async fn env_gated_bundle_synthesizes_sentence() {
        let Some(root) =
            std::env::var_os("MOTLIE_TEST_PIPER_ARTIFACT_ROOT").map(std::path::PathBuf::from)
        else {
            return;
        };

        let handle = start_typed(StartOptions {
            artifact_policy: Some(ArtifactPolicy::LocalOnly { root }),
            ..Default::default()
        })
        .await
        .expect("typed bundle should start when test artifacts are present");

        let mut stream = motlie_model::typed::SpeechSynthesizer::synthesize(
            &handle,
            SynthesisRequest {
                text: "Hello from Motlie.".into(),
                params: Default::default(),
            },
        )
        .await
        .expect("typed speech stream should open");

        let mut total_samples = 0usize;
        let mut saw_final = false;
        while let Some(chunk) = motlie_model::typed::SpeechStream::next_chunk(&mut stream)
            .await
            .expect("typed speech stream should yield chunks")
        {
            total_samples += chunk.samples().len();
            saw_final = true;
        }

        assert!(total_samples > 0);
        assert!(saw_final);

        motlie_model::typed::SpeechStream::finish(stream)
            .await
            .expect("finish should succeed");
        handle.shutdown().await.expect("shutdown should succeed");
    }

    #[test]
    fn local_model_resolution_accepts_direct_artifact_dir() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("piper-direct-root-{unique}"));
        let model_path = root.join(MODEL_FILE);
        let config_path = root.join(CONFIG_FILE);
        std::fs::create_dir_all(
            model_path
                .parent()
                .expect("model file should have a parent directory"),
        )
        .expect("direct root should be creatable");
        std::fs::write(&model_path, b"test").expect("model should be writable");
        std::fs::write(&config_path, b"test").expect("config should be writable");

        let resolved = resolve_local_model_path(&root).expect("direct artifact dir should resolve");

        assert_eq!(resolved, root.join("en/en_US/ljspeech/medium"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn local_model_resolution_accepts_flat_direct_artifact_dir() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("piper-flat-root-{unique}"));
        std::fs::create_dir_all(&root).expect("direct root should be creatable");
        std::fs::write(root.join("en_US-ljspeech-medium.onnx"), b"test")
            .expect("flat model should be writable");
        std::fs::write(root.join("en_US-ljspeech-medium.onnx.json"), b"test")
            .expect("flat config should be writable");

        let resolved =
            resolve_local_model_path(&root).expect("flat direct artifact dir should resolve");

        assert_eq!(resolved, root);
        let _ = std::fs::remove_dir_all(resolved);
    }
}
