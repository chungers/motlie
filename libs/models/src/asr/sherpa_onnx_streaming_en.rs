use std::path::{Path, PathBuf};
use std::sync::Arc;

use motlie_model::eval::EvalTrack;
use motlie_model::{
    BundleId, CheckpointFormat, ModelBundle, ModelCheckpoint, ModelError, ModelIdentity,
};
use motlie_model_sherpa_onnx::SherpaOnnxStreamingAdapter;

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
    catalog.register(descriptor(), bundle);
    catalog.register_model_variant(
        identity(),
        checkpoint(),
        Arc::new(resolve_local_onnx_root),
        Arc::new(SherpaOnnxStreamingAdapter::zipformer_en_streaming()),
    );
}

pub(crate) fn identity() -> ModelIdentity {
    ModelIdentity {
        id: BundleId::new("sherpa_onnx_streaming_zipformer_en"),
        display_name: "Sherpa ONNX Streaming Zipformer EN".into(),
        family: BundleFamily::Other("SherpaOnnx".into()),
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
        format: CheckpointFormat::Onnx,
        source: ArtifactSource::HuggingFace { repo: HF_REPO },
        include: vec![
            ArtifactRule::Exact(ENCODER_FILE),
            ArtifactRule::Exact(DECODER_FILE),
            ArtifactRule::Exact(JOINER_FILE),
            ArtifactRule::Exact(TOKENS_FILE),
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

pub fn bundle() -> Box<dyn ModelBundle> {
    let descriptor = descriptor();
    crate::adapter_backed_bundle(
        descriptor.id,
        descriptor.display_name,
        identity(),
        checkpoint(),
        Arc::new(SherpaOnnxStreamingAdapter::zipformer_en_streaming()),
        Arc::new(resolve_local_onnx_root),
    )
}

fn resolve_local_onnx_root(root: &Path) -> Result<PathBuf, ModelError> {
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

    for filename in [ENCODER_FILE, DECODER_FILE, JOINER_FILE, TOKENS_FILE] {
        let path = snapshot_dir.join(filename);
        if !path.exists() {
            return Err(ModelError::InvalidConfiguration(format!(
                "artifact policy `LocalOnly` requires `{filename}` in cached snapshot for `{HF_REPO}` under `{}`",
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
}
