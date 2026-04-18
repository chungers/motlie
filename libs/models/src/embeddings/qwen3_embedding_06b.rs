use std::path::{Path, PathBuf};

use motlie_model::eval::EvalTrack;
use motlie_model::{
    BundleId, CapabilityDescriptor, CheckpointFormat, ContentKind, EmbeddingDistance,
    EmbeddingNormalization, EmbeddingSpec, ModelBundle, ModelCheckpoint, ModelError, ModelIdentity,
    StartOptions,
};
use motlie_model_mistral::{MistralEmbeddingBundle, MistralEmbeddingHandle, MistralEmbeddingSpec};

use crate::{
    ArtifactRule, ArtifactSource, BackendKind, BuildConstraint, BundleDescriptor, BundleFamily,
    BundleRequirements, PlatformConstraint,
};

pub const SELECTOR: &str = "qwen/qwen3_embedding_06b";
const REQUIRED_LOCAL_ARTIFACTS: &[&str] = &["modules.json", "1_Pooling/config.json"];

const EMBEDDING_SPEC: EmbeddingSpec = EmbeddingSpec {
    dimensions: Some(1024),
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
        id: BundleId::new("qwen3_embedding_06b"),
        display_name: "Qwen3 Embedding 0.6B".into(),
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
            repo: "Qwen/Qwen3-Embedding-0.6B",
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
            ArtifactRule::Suffix(".safetensors"),
            ArtifactRule::Suffix(".safetensors.index.json"),
        ],
        quantization: None,
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
            "qwen3_embedding_06b",
            &checkpoint,
        )),
    }
}

pub(crate) fn variant_descriptor() -> crate::ModelVariantDescriptor {
    let spec = MistralEmbeddingSpec::qwen3_embedding_06b();
    crate::ModelVariantDescriptor {
        backend: BackendKind::MistralRs,
        capabilities: spec.capabilities,
        quantization: spec.quantization,
        checkpoint: checkpoint(),
    }
}

pub fn bundle() -> crate::CuratedBundle {
    crate::CuratedBundle::Qwen3Embedding06B
}

pub async fn start(options: StartOptions) -> Result<MistralEmbeddingHandle, ModelError> {
    MistralEmbeddingBundle::new(MistralEmbeddingSpec::qwen3_embedding_06b())
        .start(crate::resolve_typed_artifact_policy(
            options,
            resolve_local_snapshot_root,
        )?)
        .await
}

fn resolve_local_snapshot_root(root: &Path) -> Result<PathBuf, ModelError> {
    let snapshot_dir = crate::resolve_hf_snapshot("Qwen/Qwen3-Embedding-0.6B", root)?;

    for required in REQUIRED_LOCAL_ARTIFACTS {
        if !snapshot_dir.join(required).exists() {
            return Err(ModelError::InvalidConfiguration(format!(
                "artifact policy `LocalOnly` requires cached `{required}` for `Qwen/Qwen3-Embedding-0.6B` under `{}`",
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
    use motlie_model::{ArtifactPolicy, EmbeddingRequest, QuantizationBits, StartOptions};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn descriptor_is_reviewable_as_data() {
        let descriptor = descriptor();

        assert_eq!(descriptor.id.as_str(), "qwen3_embedding_06b");
        assert_eq!(descriptor.family, BundleFamily::Embeddings);
        assert_eq!(descriptor.backend, BackendKind::MistralRs);
        assert_eq!(descriptor.eval_tracks, vec![EvalTrack::Embeddings]);
        assert!(
            descriptor
                .capabilities
                .supports(motlie_model::CapabilityKind::Embeddings)
        );
        assert_eq!(
            descriptor.capability_descriptors(),
            &[CapabilityDescriptor::embeddings()]
        );

        let artifacts = descriptor
            .artifacts
            .expect("descriptor should expose curated artifact control");
        assert_eq!(artifacts.control_name, "qwen3_embedding_06b");
        assert!(artifacts.includes("model.safetensors"));
        assert!(artifacts.includes("model.safetensors.index.json"));
        assert!(artifacts.includes("config.json"));
        assert!(artifacts.includes("modules.json"));
        assert!(artifacts.includes("tokenizer.model"));
        assert!(artifacts.includes("1_Pooling/config.json"));
        assert!(artifacts.includes("2_Dense/config.json"));
        assert!(!artifacts.includes("README.md"));
    }

    #[test]
    fn embedding_spec_matches_expected_semantics() {
        let spec = embedding_spec();

        assert_eq!(spec.dimensions, Some(1024));
        assert_eq!(spec.distance, EmbeddingDistance::Cosine);
        assert_eq!(spec.normalization, EmbeddingNormalization::L2);
    }

    #[test]
    fn q8_is_supported_but_q4_is_rejected() {
        let quantization = bundle().metadata().quantization.clone();

        assert_eq!(quantization.recommended(), None);
        assert!(quantization.supports(motlie_model::QuantizationBits::Eight));
        assert!(!quantization.supports(motlie_model::QuantizationBits::Four));
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
        let snapshot = create_fake_hf_cache(&root, "Qwen/Qwen3-Embedding-0.6B", "main");

        std::fs::write(snapshot.join("config.json"), "{}").expect("config should be writable");
        std::fs::write(snapshot.join("tokenizer.json"), "{}")
            .expect("tokenizer should be writable");
        std::fs::create_dir_all(snapshot.join("1_Pooling"))
            .expect("pooling dir should be creatable");
        std::fs::write(snapshot.join("modules.json"), "[]").expect("modules should be writable");
        std::fs::write(snapshot.join("1_Pooling/config.json"), "{}")
            .expect("pooling config should be writable");
        std::fs::write(snapshot.join("model.safetensors.index.json"), "{}")
            .expect("weights index should be writable");

        let resolved = resolve_local_snapshot_root(&root)
            .expect("complete cache layout should resolve to snapshot path");

        assert_eq!(resolved, snapshot);
    }

    #[tokio::test]
    #[ignore = "requires pre-downloaded Qwen3 embedding artifacts under MOTLIE_QWEN3_EMBEDDING_06B_ROOT"]
    async fn catalog_can_start_qwen3_embedding_and_generate_vectors() {
        let root = std::env::var("MOTLIE_QWEN3_EMBEDDING_06B_ROOT")
            .expect("MOTLIE_QWEN3_EMBEDDING_06B_ROOT must point at the curated HF cache root");
        let catalog = Catalog::with_defaults();
        let bundle = catalog
            .instantiate(&BundleId::new("qwen3_embedding_06b"))
            .expect("default catalog should instantiate qwen3 embedding");
        let handle = bundle
            .start(StartOptions {
                artifact_policy: Some(ArtifactPolicy::LocalOnly { root: root.into() }),
                quantization: Some(QuantizationBits::Eight),
                ..Default::default()
            })
            .await
            .expect("bundle should start from local artifacts");

        let response = handle
            .embeddings()
            .expect("embeddings capability should exist")
            .embed(EmbeddingRequest {
                inputs: vec![
                    "motlie curated model bundle".into(),
                    "qwen embedding quantized startup".into(),
                ],
            })
            .await
            .expect("embedding request should succeed");

        assert_eq!(response.vectors.len(), 2);
        for vector in response.vectors {
            assert_eq!(vector.len(), 1024);
            assert!(vector.iter().all(|value| value.is_finite()));
        }

        let unsupported = handle
            .descriptor()
            .quantization
            .resolve(Some(QuantizationBits::Four), &handle.descriptor().id)
            .expect_err("unsupported Q4 should fail");
        assert!(matches!(unsupported, ModelError::InvalidConfiguration(_)));

        handle.shutdown().await.expect("shutdown should succeed");
    }

    fn unique_temp_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be monotonic enough for temp names")
            .as_nanos();
        std::env::temp_dir().join(format!("motlie-models-qwen3-embedding-test-{nanos}"))
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
