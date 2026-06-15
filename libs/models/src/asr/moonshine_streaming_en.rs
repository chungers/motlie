use std::path::{Path, PathBuf};

use motlie_model::eval::EvalTrack;
use motlie_model::{
    BundleId, CheckpointFormat, ModelCheckpoint, ModelError, ModelIdentity, QuantizationScheme,
    StartOptions,
};
use motlie_model_moonshine::{MoonshineHandle, MoonshineStreamingBundle, MoonshineStreamingSpec};

use crate::LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX;
use crate::{
    ArtifactRule, ArtifactSource, BackendKind, BuildConstraint, BundleDescriptor, BundleFamily,
    BundleRequirements, PlatformConstraint,
};

pub const SELECTOR: &str = "moonshine/streaming_en";

const HF_REPO: &str = "UsefulSensors/moonshine-streaming";

const FRONTEND_FILE: &str = "onnx/small/frontend.ort";
const ENCODER_FILE: &str = "onnx/small/encoder.ort";
const ADAPTER_FILE: &str = "onnx/small/adapter.ort";
const CROSS_KV_FILE: &str = "onnx/small/cross_kv.ort";
const DECODER_KV_FILE: &str = "onnx/small/decoder_kv.ort";
const STREAMING_CONFIG_FILE: &str = "onnx/small/streaming_config.json";
const TOKENIZER_JSON_FILE: &str = "onnx/small/tokenizer.json";

pub(crate) fn register(catalog: &mut crate::Catalog) {
    catalog.register_descriptor(descriptor());
    catalog.register_model_variant(identity(), variant_descriptor());
}

pub(crate) fn identity() -> ModelIdentity {
    ModelIdentity {
        id: BundleId::new("moonshine_streaming_en"),
        display_name: "Moonshine Streaming EN".into(),
        family: BundleFamily::Other("Moonshine".into()),
        capabilities: motlie_model::Capabilities::transcription_stream_partial_only(),
        eval_tracks: vec![EvalTrack::Transcription],
        requirements: BundleRequirements {
            platform: vec![PlatformConstraint::Linux, PlatformConstraint::Macos],
            build: Vec::new(),
        },
    }
}

pub(crate) fn checkpoint() -> ModelCheckpoint {
    ModelCheckpoint {
        format: CheckpointFormat::Onnx,
        source: ArtifactSource::HuggingFace { repo: HF_REPO },
        include: vec![
            ArtifactRule::Exact(FRONTEND_FILE),
            ArtifactRule::Exact(ENCODER_FILE),
            ArtifactRule::Exact(ADAPTER_FILE),
            ArtifactRule::Exact(CROSS_KV_FILE),
            ArtifactRule::Exact(DECODER_KV_FILE),
            ArtifactRule::Exact(STREAMING_CONFIG_FILE),
            ArtifactRule::Exact(TOKENIZER_JSON_FILE),
        ],
        quantization: Some(QuantizationScheme::Fp32),
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
            build: vec![BuildConstraint::Feature("backend-moonshine".into())],
        },
        eval_tracks: identity.eval_tracks,
        artifacts: Some(crate::bundle_artifacts_from_checkpoint(
            "moonshine_streaming_en",
            &checkpoint,
        )),
    }
}

pub(crate) fn variant_descriptor() -> crate::ModelVariantDescriptor {
    let spec = MoonshineStreamingSpec::small_en();
    crate::ModelVariantDescriptor {
        backend: BackendKind::Ort,
        capabilities: spec.capabilities,
        quantization: spec.quantization,
        checkpoint: checkpoint(),
    }
}

pub fn typed_bundle() -> MoonshineStreamingBundle {
    MoonshineStreamingBundle::new(MoonshineStreamingSpec::small_en())
}

pub async fn start_typed(options: StartOptions) -> Result<MoonshineHandle, ModelError> {
    typed_bundle()
        .start_typed(crate::resolve_typed_artifact_policy(
            options,
            resolve_local_model_root,
        )?)
        .await
}

fn resolve_local_model_root(root: &Path) -> Result<PathBuf, ModelError> {
    if [
        FRONTEND_FILE,
        ENCODER_FILE,
        ADAPTER_FILE,
        CROSS_KV_FILE,
        DECODER_KV_FILE,
        STREAMING_CONFIG_FILE,
        TOKENIZER_JSON_FILE,
    ]
    .into_iter()
    .all(|filename| root.join(filename).is_file())
    {
        return Ok(root.to_path_buf());
    }

    let repo_folder = format!("models--{}", HF_REPO.replace('/', "--"));
    let repo_root = root.join(repo_folder);
    let refs_dir = repo_root.join("refs");
    let main_ref = refs_dir.join("main");

    if !main_ref.exists() {
        return Err(ModelError::InvalidConfiguration(format!(
            "{LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX} requires cached ONNX artifacts for `{HF_REPO}` under `{}`; no refs/main found",
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
        FRONTEND_FILE,
        ENCODER_FILE,
        ADAPTER_FILE,
        CROSS_KV_FILE,
        DECODER_KV_FILE,
        STREAMING_CONFIG_FILE,
        TOKENIZER_JSON_FILE,
    ] {
        let path = snapshot_dir.join(filename);
        if !path.exists() {
            return Err(ModelError::InvalidConfiguration(format!(
                "{LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX} requires `{filename}` in cached snapshot for `{HF_REPO}` under `{}`",
                root.display()
            )));
        }
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

        assert_eq!(descriptor.id.as_str(), "moonshine_streaming_en");
        assert_eq!(descriptor.display_name, "Moonshine Streaming EN");
        assert_eq!(descriptor.backend, BackendKind::Ort);
        assert!(!descriptor
            .requirements
            .build
            .contains(&BuildConstraint::CpuOnly));
        assert!(descriptor
            .capabilities
            .supports(CapabilityKind::Transcription));
    }

    #[test]
    fn default_catalog_includes_bundle_when_feature_enabled() {
        let catalog = Catalog::with_defaults();
        let bundle_id = BundleId::new("moonshine_streaming_en");

        #[cfg(feature = "model-moonshine-streaming")]
        {
            assert!(catalog.instantiate(&bundle_id).is_some());
            assert!(catalog
                .bundles_for_track(EvalTrack::Transcription)
                .any(|bundle| bundle.id == bundle_id));
        }
    }

    #[test]
    fn local_root_resolver_returns_snapshot_dir_without_double_nesting() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("moonshine-root-resolver-{unique}"));
        let repo_root = root.join(format!("models--{}", HF_REPO.replace('/', "--")));
        let snapshot_dir = repo_root.join("snapshots").join("test-commit");
        std::fs::create_dir_all(snapshot_dir.join("onnx/small"))
            .expect("should create test snapshot layout");
        std::fs::create_dir_all(repo_root.join("refs")).expect("should create refs dir");
        std::fs::write(repo_root.join("refs/main"), "test-commit\n").expect("should write ref");

        for filename in [
            FRONTEND_FILE,
            ENCODER_FILE,
            ADAPTER_FILE,
            CROSS_KV_FILE,
            DECODER_KV_FILE,
            STREAMING_CONFIG_FILE,
            TOKENIZER_JSON_FILE,
        ] {
            std::fs::write(snapshot_dir.join(filename), b"test").expect("should write artifact");
        }

        let resolved = resolve_local_model_root(&root).expect("resolver should succeed");
        assert_eq!(resolved, snapshot_dir);

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn local_root_resolver_accepts_direct_artifact_dir() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("moonshine-direct-root-{unique}"));
        std::fs::create_dir_all(root.join("onnx/small")).expect("should create model dir");
        for filename in [
            FRONTEND_FILE,
            ENCODER_FILE,
            ADAPTER_FILE,
            CROSS_KV_FILE,
            DECODER_KV_FILE,
            STREAMING_CONFIG_FILE,
            TOKENIZER_JSON_FILE,
        ] {
            std::fs::write(root.join(filename), b"test").expect("should write artifact");
        }

        let resolved = resolve_local_model_root(&root).expect("direct artifact dir should resolve");
        assert_eq!(resolved, root);

        let _ = std::fs::remove_dir_all(resolved);
    }
}
