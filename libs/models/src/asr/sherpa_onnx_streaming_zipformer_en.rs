use std::path::{Path, PathBuf};

use motlie_model::eval::EvalTrack;
use motlie_model::{
    BundleId, CheckpointFormat, ModelCheckpoint, ModelError, ModelIdentity, QuantizationScheme,
    StartOptions,
};
use motlie_model_sherpa_onnx::{
    SherpaOnnxHandle, SherpaOnnxStreamingBundle, SherpaOnnxStreamingSpec,
};

use crate::LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX;
use crate::{
    ArtifactRule, ArtifactSource, BackendKind, BuildConstraint, BundleDescriptor, BundleFamily,
    BundleRequirements, PlatformConstraint,
};

pub const SELECTOR: &str = "sherpa-onnx/streaming_zipformer_en";

const HF_REPO: &str = "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26";
const ENCODER_FILE: &str = "encoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx";
const DECODER_FILE: &str = "decoder-epoch-99-avg-1-chunk-16-left-64.onnx";
const JOINER_FILE: &str = "joiner-epoch-99-avg-1-chunk-16-left-64.int8.onnx";
const TOKENS_FILE: &str = "tokens.txt";

pub(crate) fn register(catalog: &mut crate::Catalog) {
    catalog.register_descriptor(descriptor());
    catalog.register_model_variant(identity(), variant_descriptor());
}

pub(crate) fn identity() -> ModelIdentity {
    ModelIdentity {
        id: BundleId::new("sherpa_onnx_streaming_zipformer_en"),
        display_name: "Sherpa ONNX Streaming Zipformer EN".into(),
        family: BundleFamily::Other("SherpaOnnx".into()),
        capabilities: motlie_model::Capabilities::transcription_stream_partial_only(),
        eval_tracks: vec![EvalTrack::Transcription],
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
            ArtifactRule::Exact(JOINER_FILE),
            ArtifactRule::Exact(TOKENS_FILE),
        ],
        quantization: Some(QuantizationScheme::OnnxInt8),
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
        backend: BackendKind::SherpaOnnx,
        requirements: BundleRequirements {
            platform: identity.requirements.platform,
            build: vec![BuildConstraint::Feature("backend-sherpa-onnx".into())],
        },
        eval_tracks: identity.eval_tracks,
        artifacts: Some(crate::bundle_artifacts_from_checkpoint(
            "sherpa_onnx_streaming_zipformer_en",
            &checkpoint,
        )),
    }
}

pub(crate) fn variant_descriptor() -> crate::ModelVariantDescriptor {
    let spec = SherpaOnnxStreamingSpec::zipformer_en_streaming();
    crate::ModelVariantDescriptor {
        backend: BackendKind::SherpaOnnx,
        capabilities: spec.capabilities,
        quantization: spec.quantization,
        checkpoint: checkpoint(),
    }
}

pub fn typed_bundle() -> SherpaOnnxStreamingBundle {
    SherpaOnnxStreamingBundle::new(SherpaOnnxStreamingSpec::zipformer_en_streaming())
}

pub async fn start_typed(options: StartOptions) -> Result<SherpaOnnxHandle, ModelError> {
    typed_bundle()
        .start_typed(crate::resolve_typed_artifact_policy(
            options,
            resolve_local_onnx_root,
        )?)
        .await
}

fn resolve_local_onnx_root(root: &Path) -> Result<PathBuf, ModelError> {
    if [ENCODER_FILE, DECODER_FILE, JOINER_FILE, TOKENS_FILE]
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

    for filename in [ENCODER_FILE, DECODER_FILE, JOINER_FILE, TOKENS_FILE] {
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
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn descriptor_is_reviewable_as_data() {
        let descriptor = descriptor();

        assert_eq!(descriptor.id.as_str(), "sherpa_onnx_streaming_zipformer_en");
        assert_eq!(
            descriptor.display_name,
            "Sherpa ONNX Streaming Zipformer EN"
        );
        assert_eq!(descriptor.backend, BackendKind::SherpaOnnx);
        assert!(descriptor
            .capabilities
            .supports(motlie_model::CapabilityKind::Transcription));
    }

    #[test]
    fn default_catalog_includes_bundle_when_feature_enabled() {
        let catalog = Catalog::with_defaults();
        let bundle_id = BundleId::new("sherpa_onnx_streaming_zipformer_en");

        #[cfg(feature = "model-sherpa-onnx-streaming")]
        {
            assert!(catalog.instantiate(&bundle_id).is_some());
            assert!(catalog
                .bundles_for_track(EvalTrack::Transcription)
                .any(|bundle| bundle.id == bundle_id));
        }
    }

    #[test]
    fn local_root_resolution_accepts_direct_artifact_dir() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("sherpa-direct-root-{unique}"));
        std::fs::create_dir_all(&root).expect("direct root should be creatable");
        for filename in [ENCODER_FILE, DECODER_FILE, JOINER_FILE, TOKENS_FILE] {
            std::fs::write(root.join(filename), b"test").expect("artifact should be writable");
        }

        let resolved = resolve_local_onnx_root(&root).expect("direct artifact dir should resolve");

        assert_eq!(resolved, root);
        let _ = std::fs::remove_dir_all(resolved);
    }
}
