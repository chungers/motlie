use std::collections::BTreeMap;

use motlie_model::eval::EvalTrack;
use motlie_model::{BundleId, Capabilities, CapabilityDescriptor};

/// Organizational family for related curated bundles.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BundleFamily {
    Embeddings,
    Gpt,
    Hermes,
    Other(String),
    Qwen,
}

/// Product support level for a curated bundle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SupportTier {
    Experimental,
    Stable,
    Supported,
}

/// Public packaging mode for a bundle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackagingMode {
    Embedded,
    Remote,
    Sidecar,
}

/// Internal execution substrate chosen for a bundle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendKind {
    Http,
    LlamaCpp,
    MistralRs,
    Ort,
}

/// Platform scoping visible to operators and release tooling.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlatformConstraint {
    Linux,
    Macos,
    Distribution(String),
    Architecture(String),
}

/// Build-time constraints kept close to the bundle definition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BuildConstraint {
    CpuOnly,
    CudaRequired,
    Feature(String),
    Profile(String),
}

/// Requirements that affect whether and how a bundle may be loaded or built.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BundleRequirements {
    pub platform: Vec<PlatformConstraint>,
    pub build: Vec<BuildConstraint>,
}

/// Product-facing descriptor for a curated model bundle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BundleDescriptor {
    pub id: BundleId,
    pub display_name: String,
    pub family: BundleFamily,
    pub support_tier: SupportTier,
    pub capabilities: Capabilities,
    pub packaging: PackagingMode,
    pub backend: BackendKind,
    pub requirements: BundleRequirements,
    pub eval_tracks: Vec<EvalTrack>,
}

impl BundleDescriptor {
    pub fn supports_track(&self, track: EvalTrack) -> bool {
        self.eval_tracks.contains(&track)
    }

    pub fn capability_descriptors(&self) -> &[CapabilityDescriptor] {
        self.capabilities.descriptors()
    }
}

/// In-memory registry of curated bundle descriptors.
#[derive(Clone, Debug, Default)]
pub struct Catalog {
    bundles: BTreeMap<BundleId, BundleDescriptor>,
}

impl Catalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, descriptor: BundleDescriptor) -> Option<BundleDescriptor> {
        self.bundles.insert(descriptor.id.clone(), descriptor)
    }

    pub fn bundle(&self, id: &BundleId) -> Option<&BundleDescriptor> {
        self.bundles.get(id)
    }

    pub fn bundles(&self) -> impl Iterator<Item = &BundleDescriptor> {
        self.bundles.values()
    }

    pub fn bundles_for_track(&self, track: EvalTrack) -> impl Iterator<Item = &BundleDescriptor> {
        self.bundles
            .values()
            .filter(move |descriptor| descriptor.supports_track(track))
    }

    pub fn len(&self) -> usize {
        self.bundles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bundles.is_empty()
    }
}
