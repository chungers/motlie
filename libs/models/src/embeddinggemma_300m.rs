use motlie_model::eval::EvalTrack;
use motlie_model::{BundleId, CapabilityDescriptor, ModelBundle};
use motlie_model_mistral::{MistralEmbeddingBundle, MistralEmbeddingSpec};

use crate::{
    ArtifactRule, ArtifactSource, BackendKind, BuildConstraint, BundleArtifacts, BundleDescriptor,
    BundleFamily, BundleRequirements, PackagingMode, PlatformConstraint, SupportTier,
};

pub fn descriptor() -> BundleDescriptor {
    BundleDescriptor {
        id: BundleId::new("embeddinggemma_300m"),
        display_name: "EmbeddingGemma 300M".into(),
        family: BundleFamily::Embeddings,
        support_tier: SupportTier::Experimental,
        capabilities: motlie_model::Capabilities::new(vec![CapabilityDescriptor::embeddings()]),
        packaging: PackagingMode::Sidecar,
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
    Box::new(MistralEmbeddingBundle::new(
        MistralEmbeddingSpec::embeddinggemma_300m(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use motlie_model::{ArtifactPolicy, EmbeddingRequest, StartOptions};
    use crate::Catalog;

    #[test]
    fn descriptor_is_reviewable_as_data() {
        let descriptor = descriptor();

        assert_eq!(descriptor.id.as_str(), "embeddinggemma_300m");
        assert_eq!(descriptor.family, BundleFamily::Embeddings);
        assert_eq!(descriptor.backend, BackendKind::MistralRs);
        assert_eq!(descriptor.packaging, PackagingMode::Sidecar);
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
