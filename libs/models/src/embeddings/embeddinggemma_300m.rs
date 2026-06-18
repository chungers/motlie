use std::path::{Path, PathBuf};

use motlie_model::eval::EvalTrack;
use motlie_model::{
    BundleId, CapabilityDescriptor, CheckpointFormat, ContentKind, EmbeddingDistance,
    EmbeddingNormalization, EmbeddingSpec, ModelBundle, ModelCheckpoint, ModelError, ModelIdentity,
    QuantizationScheme, StartOptions,
};
use motlie_model_mistral::{MistralEmbeddingBundle, MistralEmbeddingHandle, MistralEmbeddingSpec};

use crate::{
    ArtifactRule, ArtifactSource, BackendKind, BuildConstraint, BundleDescriptor, BundleFamily,
    BundleRequirements, PlatformConstraint,
};

pub const SELECTOR: &str = "google/embeddinggemma_300m";
const REQUIRED_LOCAL_ARTIFACTS: &[&str] = &[
    "modules.json",
    "1_Pooling/config.json",
    "2_Dense/config.json",
    "2_Dense/model.safetensors",
    "3_Dense/config.json",
    "3_Dense/model.safetensors",
];

const EMBEDDING_SPEC: EmbeddingSpec = EmbeddingSpec {
    dimensions: Some(768),
    distance: EmbeddingDistance::Cosine,
    normalization: EmbeddingNormalization::L2,
    input: ContentKind::Text,
    output: ContentKind::EmbeddingVector,
    summary: "Normalized text embeddings for semantic similarity and retrieval.",
};

pub(crate) fn register(catalog: &mut crate::Catalog) {
    catalog.register_descriptor(descriptor());
    catalog.register_model_variant(identity(), variant_descriptor());
}

pub(crate) fn identity() -> ModelIdentity {
    ModelIdentity {
        id: BundleId::new("embeddinggemma_300m"),
        display_name: "EmbeddingGemma 300M".into(),
        family: BundleFamily::Embeddings,
        capabilities: motlie_model::Capabilities::new(vec![CapabilityDescriptor::embeddings()]),
        eval_tracks: vec![EvalTrack::Embeddings],
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
            repo: "google/embeddinggemma-300m",
        },
        include: vec![
            ArtifactRule::Exact("config.json"),
            ArtifactRule::Exact("modules.json"),
            ArtifactRule::Exact("tokenizer.json"),
            ArtifactRule::Exact("tokenizer.model"),
            ArtifactRule::Exact("tokenizer_config.json"),
            ArtifactRule::Exact("special_tokens_map.json"),
            ArtifactRule::Exact("1_Pooling/config.json"),
            ArtifactRule::Exact("2_Dense/config.json"),
            ArtifactRule::Exact("3_Dense/config.json"),
            ArtifactRule::Suffix(".safetensors"),
        ],
        quantization: Some(QuantizationScheme::Fp32),
    }
}

pub fn embedding_spec() -> &'static EmbeddingSpec {
    &EMBEDDING_SPEC
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
        backend: BackendKind::MistralRs,
        requirements: BundleRequirements {
            platform: identity.requirements.platform,
            build: vec![BuildConstraint::Feature("backend-mistral".into())],
        },
        eval_tracks: identity.eval_tracks,
        artifacts: Some(crate::bundle_artifacts_from_checkpoint(
            "embeddinggemma_300m",
            &checkpoint,
            crate::ArtifactProvenance::new("gemma", crate::ArtifactGating::Manual),
        )),
    }
}

pub(crate) fn variant_descriptor() -> crate::ModelVariantDescriptor {
    let spec = MistralEmbeddingSpec::embeddinggemma_300m();
    crate::ModelVariantDescriptor {
        backend: BackendKind::MistralRs,
        capabilities: spec.capabilities,
        quantization: spec.quantization,
        checkpoint: checkpoint(),
    }
}

pub fn bundle() -> crate::CuratedBundle {
    crate::CuratedBundle::EmbeddingGemma300m
}

pub async fn start(options: StartOptions) -> Result<MistralEmbeddingHandle, ModelError> {
    MistralEmbeddingBundle::new(MistralEmbeddingSpec::embeddinggemma_300m())
        .start(crate::resolve_typed_artifact_policy(
            options,
            resolve_local_snapshot_root,
        )?)
        .await
}

fn resolve_local_snapshot_root(root: &Path) -> Result<PathBuf, ModelError> {
    let snapshot_dir = crate::resolve_hf_snapshot("google/embeddinggemma-300m", root)?;

    // EmbeddingGemma requires sentence-transformer module files beyond the
    // standard transformer layout validated by the shared resolver.
    for required in REQUIRED_LOCAL_ARTIFACTS {
        if !snapshot_dir.join(required).exists() {
            return Err(ModelError::InvalidConfiguration(format!(
                "artifact policy `LocalOnly` requires cached `{required}` for `google/embeddinggemma-300m` under `{}`",
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
    use motlie_model::{
        ArtifactPolicy, BundleHandle, EmbeddingModel, EmbeddingRequest, StartOptions,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn descriptor_is_reviewable_as_data() {
        let descriptor = descriptor();

        assert_eq!(descriptor.id.as_str(), "embeddinggemma_300m");
        assert_eq!(descriptor.family, BundleFamily::Embeddings);
        assert_eq!(descriptor.backend, BackendKind::MistralRs);
        assert_eq!(descriptor.eval_tracks, vec![EvalTrack::Embeddings]);
        assert!(descriptor
            .capabilities
            .supports(motlie_model::CapabilityKind::Embeddings));
        assert_eq!(
            descriptor.capability_descriptors(),
            &[CapabilityDescriptor::embeddings()]
        );

        let artifacts = descriptor
            .artifacts
            .expect("descriptor should expose curated artifact control");
        assert_eq!(artifacts.control_name, "embeddinggemma_300m");
        assert!(artifacts.includes("model.safetensors"));
        assert!(artifacts.includes("config.json"));
        assert!(!artifacts.includes("README.md"));
    }

    #[test]
    fn embedding_spec_matches_expected_semantics() {
        let spec = embedding_spec();

        assert_eq!(spec.dimensions, Some(768));
        assert_eq!(spec.distance, EmbeddingDistance::Cosine);
        assert_eq!(spec.normalization, EmbeddingNormalization::L2);
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
        let snapshot = create_fake_hf_cache(&root, "google/embeddinggemma-300m", "main");

        std::fs::write(snapshot.join("config.json"), "{}").expect("config should be writable");
        std::fs::write(snapshot.join("tokenizer.json"), "{}")
            .expect("tokenizer should be writable");
        std::fs::write(snapshot.join("model-00001-of-00001.safetensors"), "stub")
            .expect("weights should be writable");
        std::fs::create_dir_all(snapshot.join("1_Pooling"))
            .expect("pooling dir should be creatable");
        std::fs::create_dir_all(snapshot.join("2_Dense")).expect("dense dir should be creatable");
        std::fs::create_dir_all(snapshot.join("3_Dense")).expect("dense dir should be creatable");
        std::fs::write(snapshot.join("modules.json"), "[]").expect("modules should be writable");
        std::fs::write(snapshot.join("1_Pooling/config.json"), "{}")
            .expect("pooling config should be writable");
        std::fs::write(snapshot.join("2_Dense/config.json"), "{}")
            .expect("dense config should be writable");
        std::fs::write(snapshot.join("2_Dense/model.safetensors"), "stub")
            .expect("dense weights should be writable");
        std::fs::write(snapshot.join("3_Dense/config.json"), "{}")
            .expect("dense config should be writable");
        std::fs::write(snapshot.join("3_Dense/model.safetensors"), "stub")
            .expect("dense weights should be writable");

        let resolved = resolve_local_snapshot_root(&root)
            .expect("complete cache layout should resolve to snapshot path");

        assert_eq!(resolved, snapshot);
    }

    #[tokio::test]
    #[ignore = "requires pre-downloaded embeddinggemma artifacts under MOTLIE_EMBEDDINGGEMMA_ROOT"]
    async fn catalog_can_start_embeddinggemma_and_generate_finite_vectors() {
        let root = std::env::var("MOTLIE_EMBEDDINGGEMMA_ROOT")
            .expect("MOTLIE_EMBEDDINGGEMMA_ROOT must point at the curated HF cache root");
        let catalog = Catalog::with_defaults();
        let bundle = catalog
            .instantiate(&BundleId::new("embeddinggemma_300m"))
            .expect("default catalog should instantiate embeddinggemma");
        let handle = bundle
            .start(StartOptions {
                artifact_policy: Some(ArtifactPolicy::LocalOnly { root: root.into() }),
                ..Default::default()
            })
            .await
            .expect("bundle should start from local artifacts");
        let response = handle
            .embeddings()
            .expect("embeddings capability should exist")
            .embed(EmbeddingRequest {
                inputs: vec!["motlie curated model bundle".into()],
            })
            .await
            .expect("embedding request should succeed");
        let vector = response
            .vectors
            .into_iter()
            .next()
            .expect("embedding output should contain one vector");

        assert!(!vector.is_empty(), "embedding vector should not be empty");
        assert!(
            vector.iter().all(|value| value.is_finite()),
            "embedding vector should not contain NaN or Inf values: {vector:?}"
        );

        handle.shutdown().await.expect("shutdown should succeed");
    }

    fn unique_temp_dir() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic enough")
            .as_nanos();
        std::env::temp_dir().join(format!("motlie-models-google-gemma-test-{unique}"))
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
