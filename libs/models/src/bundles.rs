use motlie_model::eval::EvalTrack;
use motlie_model::{BundleId, CapabilityDescriptor, ModelBundle};
use motlie_model_mistral::{MistralEmbeddingBundle, MistralEmbeddingSpec};

use crate::catalog::{
    BackendKind, BuildConstraint, BundleDescriptor, BundleFamily, BundleRequirements, PackagingMode,
    PlatformConstraint, SupportTier,
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
    }
}

pub fn embeddinggemma_300m_bundle() -> Box<dyn ModelBundle> {
    Box::new(MistralEmbeddingBundle::new(
        MistralEmbeddingSpec::embeddinggemma_300m(),
    ))
}
