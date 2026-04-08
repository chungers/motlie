//! Curated model bundle catalog for the Motlie ecosystem.
//!
//! This crate owns the bundle/catalog layer above `motlie-model`.

mod catalog;

pub use motlie_model::eval::EvalTrack;
pub use motlie_model::{
    BundleId, Capabilities, CapabilityDescriptor, CapabilityKind, ContentKind, InteractionStyle,
};

pub use catalog::{
    BackendKind, BuildConstraint, BundleDescriptor, BundleFamily, BundleRequirements, Catalog,
    PackagingMode, PlatformConstraint, SupportTier,
};
