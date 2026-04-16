use std::path::{Path, PathBuf};
use std::sync::Arc;

use motlie_model::eval::EvalTrack;
use motlie_model::{
    BundleId, CheckpointFormat, ModelBundle, ModelCheckpoint, ModelError, ModelIdentity,
};
use motlie_model_moonshine::MoonshineStreamingAdapter;

use crate::{
    ArtifactRule, ArtifactSource, BackendKind, BuildConstraint, BundleDescriptor, BundleFamily,
    BundleRequirements, PlatformConstraint,
};

pub const SELECTOR: &str = "moonshine/streaming_en";

const HF_REPO: &str = "UsefulSensors/moonshine-streaming";
const MODEL_ROOT: &str = "onnx/small";

const FRONTEND_FILE: &str = "onnx/small/frontend.ort";
const ENCODER_FILE: &str = "onnx/small/encoder.ort";
const ADAPTER_FILE: &str = "onnx/small/adapter.ort";
const CROSS_KV_FILE: &str = "onnx/small/cross_kv.ort";
const DECODER_KV_FILE: &str = "onnx/small/decoder_kv.ort";
const STREAMING_CONFIG_FILE: &str = "onnx/small/streaming_config.json";
const TOKENIZER_JSON_FILE: &str = "onnx/small/tokenizer.json";

pub(crate) fn register(catalog: &mut crate::Catalog) {
    catalog.register(descriptor(), bundle);
    catalog.register_model_variant(
        identity(),
        checkpoint(),
        Arc::new(resolve_local_model_root),
        Arc::new(MoonshineStreamingAdapter::small_en()),
    );
}

pub(crate) fn identity() -> ModelIdentity {
    ModelIdentity {
        id: BundleId::new("moonshine_streaming_en"),
        display_name: "Moonshine Streaming EN".into(),
        family: BundleFamily::Other("Moonshine".into()),
        capabilities: motlie_model::Capabilities::transcription_stream_only(),
        eval_tracks: vec![EvalTrack::Transcription],
        requirements: BundleRequirements {
            platform: vec![PlatformConstraint::Linux, PlatformConstraint::Macos],
            build: vec![BuildConstraint::CpuOnly],
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
            build: vec![
                BuildConstraint::CpuOnly,
                BuildConstraint::Feature("backend-moonshine".into()),
            ],
        },
        eval_tracks: identity.eval_tracks,
        artifacts: Some(crate::bundle_artifacts_from_checkpoint(
            "moonshine_streaming_en",
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
        Arc::new(MoonshineStreamingAdapter::small_en()),
        Arc::new(resolve_local_model_root),
    )
}

fn resolve_local_model_root(root: &Path) -> Result<PathBuf, ModelError> {
    let repo_folder = format!("models--{}", HF_REPO.replace('/', "--"));
    let repo_root = root.join(repo_folder);
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
                "artifact policy `LocalOnly` requires `{filename}` in cached snapshot for `{HF_REPO}` under `{}`",
                root.display()
            )));
        }
    }

    Ok(snapshot_dir.join(MODEL_ROOT))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Catalog;
    use motlie_model::CapabilityKind;

    #[test]
    fn descriptor_is_reviewable_as_data() {
        let descriptor = descriptor();

        assert_eq!(descriptor.id.as_str(), "moonshine_streaming_en");
        assert_eq!(descriptor.display_name, "Moonshine Streaming EN");
        assert_eq!(descriptor.backend, BackendKind::Ort);
        assert!(descriptor
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
}
