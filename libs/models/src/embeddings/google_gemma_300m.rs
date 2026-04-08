use motlie_model::eval::EvalTrack;
use motlie_model::{
    BundleId, CapabilityDescriptor, ContentKind, Embedding as EmbeddingBundle, EmbeddingDistance,
    EmbeddingNormalization, EmbeddingSpec, ModelBundle,
};
use motlie_model_mistral::{MistralEmbeddingBundle, MistralEmbeddingSpec};

use crate::{
    ArtifactRule, ArtifactSource, BackendKind, BuildConstraint, BundleArtifacts, BundleDescriptor,
    BundleFamily, BundleRequirements, PlatformConstraint,
};

pub const SELECTOR: &str = "google/embeddinggemma_300m";

const EMBEDDING_SPEC: EmbeddingSpec = EmbeddingSpec {
    dimensions: Some(768),
    distance: EmbeddingDistance::Cosine,
    normalization: EmbeddingNormalization::L2,
    input: ContentKind::Text,
    output: ContentKind::EmbeddingVector,
    summary: "Normalized text embeddings for semantic similarity and retrieval.",
};

#[derive(Clone, Debug)]
pub struct GoogleGemma300m {
    inner: MistralEmbeddingBundle,
}

impl GoogleGemma300m {
    pub fn new() -> Self {
        Self {
            inner: MistralEmbeddingBundle::new(MistralEmbeddingSpec::embeddinggemma_300m()),
        }
    }
}

impl Default for GoogleGemma300m {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl ModelBundle for GoogleGemma300m {
    fn id(&self) -> &BundleId {
        self.inner.id()
    }

    fn metadata(&self) -> &motlie_model::BundleMetadata {
        self.inner.metadata()
    }

    fn capabilities(&self) -> &motlie_model::Capabilities {
        self.inner.capabilities()
    }

    async fn start(
        &self,
        options: motlie_model::StartOptions,
    ) -> Result<Box<dyn motlie_model::BundleHandle>, motlie_model::ModelError> {
        self.inner.start(options).await
    }
}

impl EmbeddingBundle for GoogleGemma300m {
    fn embedding_spec(&self) -> &EmbeddingSpec {
        &EMBEDDING_SPEC
    }
}

pub fn embedding_spec() -> &'static EmbeddingSpec {
    &EMBEDDING_SPEC
}

pub fn descriptor() -> BundleDescriptor {
    BundleDescriptor {
        id: BundleId::new("embeddinggemma_300m"),
        display_name: "EmbeddingGemma 300M".into(),
        family: BundleFamily::Embeddings,
        capabilities: motlie_model::Capabilities::new(vec![CapabilityDescriptor::embeddings()]),
        backend: BackendKind::MistralRs,
        requirements: BundleRequirements {
            platform: vec![PlatformConstraint::Linux, PlatformConstraint::Macos],
            build: vec![BuildConstraint::Feature("backend-mistral".into())],
        },
        eval_tracks: vec![EvalTrack::Embeddings],
        artifacts: Some(BundleArtifacts {
            control_name: "embeddinggemma_300m",
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
                ArtifactRule::Suffix(".safetensors.index.json"),
            ],
        }),
    }
}

pub fn bundle() -> Box<dyn ModelBundle> {
    Box::new(GoogleGemma300m::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Catalog;
    use motlie_model::{ArtifactPolicy, EmbeddingRequest, StartOptions};

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
}
