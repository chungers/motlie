//! Curated model bundle catalog for the Motlie ecosystem.
//!
//! This crate owns the bundle/catalog layer above `motlie-model`.

mod bundles;
mod catalog;

pub use motlie_model::eval::EvalTrack;
pub use motlie_model::{
    BundleId, Capabilities, CapabilityDescriptor, CapabilityKind, ContentKind, InteractionStyle,
};

pub use bundles::embeddinggemma_300m_bundle;
pub use catalog::{
    BackendKind, BuildConstraint, BundleDescriptor, BundleFamily, BundleRequirements, Catalog,
    PackagingMode, PlatformConstraint, SupportTier,
};
