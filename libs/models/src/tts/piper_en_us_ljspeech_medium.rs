use std::path::{Path, PathBuf};
use std::sync::Arc;

use motlie_model::eval::EvalTrack;
use motlie_model::{
    BundleId, CheckpointFormat, ModelBundle, ModelCheckpoint, ModelError, ModelIdentity,
};
use motlie_model_piper::PiperSpeechAdapter;

use crate::{
    ArtifactRule, ArtifactSource, BackendKind, BuildConstraint, BundleDescriptor, BundleFamily,
    BundleRequirements, PlatformConstraint,
};

pub const SELECTOR: &str = "piper/en_us_ljspeech_medium";

const HF_REPO: &str = "rhasspy/piper-voices";
const MODEL_FILE: &str = "en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx";
const CONFIG_FILE: &str = "en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx.json";

pub(crate) fn register(catalog: &mut crate::Catalog) {
    catalog.register(descriptor(), bundle);
    catalog.register_model_variant(
        identity(),
        checkpoint(),
        Arc::new(resolve_local_model_path),
        Arc::new(PiperSpeechAdapter::en_us_ljspeech_medium()),
    );
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

pub fn bundle() -> Box<dyn ModelBundle> {
    let descriptor = descriptor();
    crate::adapter_backed_bundle(
        descriptor.id,
        descriptor.display_name,
        identity(),
        checkpoint(),
        Arc::new(PiperSpeechAdapter::en_us_ljspeech_medium()),
        Arc::new(resolve_local_model_path),
    )
}

fn resolve_local_model_path(root: &Path) -> Result<PathBuf, ModelError> {
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

    Ok(snapshot_dir.join(MODEL_FILE))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Catalog;
    use motlie_model::{ArtifactPolicy, PcmEncoding, SpeechRequest, StartOptions};

    #[test]
    fn descriptor_is_reviewable_as_data() {
        let descriptor = descriptor();

        assert_eq!(descriptor.id.as_str(), "piper_en_us_ljspeech_medium");
        assert_eq!(descriptor.display_name, "Piper en_US ljspeech medium");
        assert_eq!(descriptor.backend, BackendKind::Ort);
        assert!(descriptor
            .capabilities
            .supports(motlie_model::CapabilityKind::Speech));
    }

    #[test]
    fn default_catalog_includes_bundle_when_feature_enabled() {
        let catalog = Catalog::with_defaults();
        let bundle_id = BundleId::new("piper_en_us_ljspeech_medium");

        #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
        {
            assert!(catalog.instantiate(&bundle_id).is_some());
            assert!(catalog
                .bundles_for_track(EvalTrack::Speech)
                .any(|bundle| bundle.id == bundle_id));
        }
    }

    #[tokio::test]
    async fn env_gated_bundle_synthesizes_sentence() {
        let Some(root) =
            std::env::var_os("MOTLIE_TEST_PIPER_ARTIFACT_ROOT").map(std::path::PathBuf::from)
        else {
            return;
        };

        let handle = bundle()
            .start(StartOptions {
                artifact_policy: Some(ArtifactPolicy::LocalOnly { root }),
                ..Default::default()
            })
            .await
            .expect("bundle should start when test artifacts are present");

        let speech = handle
            .speech()
            .expect("bundle should expose speech capability");
        let mut stream = speech
            .open_stream(SpeechRequest {
                text: "Hello from Motlie.".into(),
                params: Default::default(),
                conditioning: None,
            })
            .await
            .expect("speech stream should open");

        let spec = stream.audio_spec().clone();
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.encoding, PcmEncoding::S16Le);

        let mut total_bytes = 0usize;
        let mut saw_final = false;
        while let Some(chunk) = stream
            .next_chunk()
            .await
            .expect("speech stream should yield chunks")
        {
            total_bytes += chunk.data.len();
            saw_final = chunk.end_of_stream;
            if saw_final {
                break;
            }
        }

        assert!(total_bytes > 0);
        assert!(saw_final);

        stream.finish().await.expect("finish should succeed");
        handle.shutdown().await.expect("shutdown should succeed");
    }
}
