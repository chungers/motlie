use motlie_model::eval::EvalTrack;
use motlie_model::{BundleId, CapabilityDescriptor, ModelBundle};
use motlie_model_mistral::{MistralEmbeddingBundle, MistralEmbeddingSpec};

use crate::artifacts::{ArtifactRule, ArtifactSource, BundleArtifacts};
use crate::catalog::{
    BackendKind, BuildConstraint, BundleDescriptor, BundleFamily, BundleRequirements,
    PackagingMode, PlatformConstraint, SupportTier,
};

pub fn embeddinggemma_300m_descriptor() -> BundleDescriptor {
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
                ArtifactRule::Exact("tokenizer.json"),
                ArtifactRule::Exact("tokenizer.model"),
                ArtifactRule::Exact("tokenizer_config.json"),
                ArtifactRule::Exact("special_tokens_map.json"),
                ArtifactRule::Suffix(".safetensors"),
                ArtifactRule::Suffix(".safetensors.index.json"),
            ],
        }),
    }
}

pub fn embeddinggemma_300m_bundle() -> Box<dyn ModelBundle> {
    Box::new(MistralEmbeddingBundle::new(
        MistralEmbeddingSpec::embeddinggemma_300m(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeddinggemma_descriptor_is_reviewable_as_data() {
        let descriptor = embeddinggemma_300m_descriptor();

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
}
