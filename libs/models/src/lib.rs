//! Curated model bundle catalog for the Motlie ecosystem.
//!
//! This crate owns the bundle/catalog layer above `motlie-model`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

mod embeddinggemma_300m;

use anyhow::{Context, Result};
use hf_hub::api::sync::ApiBuilder;

pub use motlie_model::eval::EvalTrack;
pub use motlie_model::{
    BundleId, Capabilities, CapabilityDescriptor, CapabilityKind, ContentKind, InteractionStyle,
    ModelBundle,
};

pub use embeddinggemma_300m::{
    bundle as embeddinggemma_300m_bundle, descriptor as embeddinggemma_300m_descriptor,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArtifactSource {
    HuggingFace { repo: &'static str },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArtifactRule {
    Exact(&'static str),
    Suffix(&'static str),
}

impl ArtifactRule {
    fn matches(&self, filename: &str) -> bool {
        match self {
            Self::Exact(expected) => filename == *expected,
            Self::Suffix(suffix) => filename.ends_with(suffix),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BundleArtifacts {
    pub control_name: &'static str,
    pub source: ArtifactSource,
    pub include: Vec<ArtifactRule>,
}

impl BundleArtifacts {
    pub fn includes(&self, filename: &str) -> bool {
        self.include.iter().any(|rule| rule.matches(filename))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArtifactDownloadSummary {
    pub bundle_id: BundleId,
    pub downloaded: Vec<PathBuf>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ArtifactDownloadOptions {
    pub hf_token: Option<String>,
}

pub fn default_artifact_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/models/hf-cache")
}

pub fn download_bundle_artifacts(
    catalog: &Catalog,
    bundle_id: &BundleId,
    artifact_root: &Path,
) -> Result<ArtifactDownloadSummary> {
    download_bundle_artifacts_with_options(
        catalog,
        bundle_id,
        artifact_root,
        &ArtifactDownloadOptions::default(),
    )
}

pub fn download_bundle_artifacts_with_options(
    catalog: &Catalog,
    bundle_id: &BundleId,
    artifact_root: &Path,
    options: &ArtifactDownloadOptions,
) -> Result<ArtifactDownloadSummary> {
    let descriptor = catalog
        .bundle(bundle_id)
        .with_context(|| format!("unknown bundle `{bundle_id}`"))?;
    let artifacts = descriptor
        .artifacts
        .as_ref()
        .with_context(|| format!("bundle `{bundle_id}` does not define artifacts"))?;

    match &artifacts.source {
        ArtifactSource::HuggingFace { repo } => {
            std::fs::create_dir_all(artifact_root).with_context(|| {
                format!(
                    "failed to create artifact root `{}`",
                    artifact_root.display()
                )
            })?;

            let api = ApiBuilder::new()
                .with_cache_dir(artifact_root.to_path_buf())
                .with_token(options.hf_token.clone())
                .build()
                .context("failed to create Hugging Face API client")?;
            let repo_api = api.model((*repo).to_string());
            let info = repo_api
                .info()
                .with_context(|| format!("failed to inspect model repo `{repo}`"))?;

            let mut downloaded = Vec::new();
            for sibling in info.siblings {
                if artifacts.includes(&sibling.rfilename) {
                    let path = repo_api.get(&sibling.rfilename).with_context(|| {
                        format!(
                            "failed to download `{}` from repo `{repo}`",
                            sibling.rfilename
                        )
                    })?;
                    downloaded.push(path);
                }
            }

            downloaded.sort();

            Ok(ArtifactDownloadSummary {
                bundle_id: bundle_id.clone(),
                downloaded,
            })
        }
    }
}

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
    pub artifacts: Option<BundleArtifacts>,
}

impl BundleDescriptor {
    pub fn supports_track(&self, track: EvalTrack) -> bool {
        self.eval_tracks.contains(&track)
    }

    pub fn capability_descriptors(&self) -> &[CapabilityDescriptor] {
        self.capabilities.descriptors()
    }
}

trait BundleFactory: Send + Sync {
    fn instantiate(&self) -> Box<dyn ModelBundle>;
}

impl<F> BundleFactory for F
where
    F: Fn() -> Box<dyn ModelBundle> + Send + Sync + 'static,
{
    fn instantiate(&self) -> Box<dyn ModelBundle> {
        (self)()
    }
}

struct CatalogEntry {
    descriptor: BundleDescriptor,
    factory: Arc<dyn BundleFactory>,
}

/// In-memory registry of curated bundle descriptors and constructors.
#[derive(Default)]
pub struct Catalog {
    bundles: BTreeMap<BundleId, CatalogEntry>,
}

impl Catalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_defaults() -> Self {
        let mut catalog = Self::new();
        catalog.register(embeddinggemma_300m::descriptor(), || {
            embeddinggemma_300m::bundle()
        });
        catalog
    }

    pub fn register<F>(
        &mut self,
        descriptor: BundleDescriptor,
        factory: F,
    ) -> Option<BundleDescriptor>
    where
        F: Fn() -> Box<dyn ModelBundle> + Send + Sync + 'static,
    {
        self.bundles
            .insert(
                descriptor.id.clone(),
                CatalogEntry {
                    descriptor: descriptor.clone(),
                    factory: Arc::new(factory),
                },
            )
            .map(|entry| entry.descriptor)
    }

    pub fn bundle(&self, id: &BundleId) -> Option<&BundleDescriptor> {
        self.bundles.get(id).map(|entry| &entry.descriptor)
    }

    pub fn artifacts(&self, id: &BundleId) -> Option<&BundleArtifacts> {
        self.bundle(id)
            .and_then(|descriptor| descriptor.artifacts.as_ref())
    }

    pub fn instantiate(&self, id: &BundleId) -> Option<Box<dyn ModelBundle>> {
        self.bundles
            .get(id)
            .map(|entry| entry.factory.instantiate())
    }

    pub fn bundles(&self) -> impl Iterator<Item = &BundleDescriptor> {
        self.bundles.values().map(|entry| &entry.descriptor)
    }

    pub fn bundles_for_track(&self, track: EvalTrack) -> impl Iterator<Item = &BundleDescriptor> {
        self.bundles
            .values()
            .map(|entry| &entry.descriptor)
            .filter(move |descriptor| descriptor.supports_track(track))
    }

    pub fn len(&self) -> usize {
        self.bundles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bundles.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use motlie_model::BundleMetadata;

    use super::*;

    #[derive(Clone)]
    struct StubBundle {
        metadata: BundleMetadata,
    }

    #[async_trait::async_trait]
    impl ModelBundle for StubBundle {
        fn id(&self) -> &BundleId {
            &self.metadata.id
        }

        fn metadata(&self) -> &BundleMetadata {
            &self.metadata
        }

        fn capabilities(&self) -> &Capabilities {
            &self.metadata.capabilities
        }

        async fn start(
            &self,
            _options: motlie_model::StartOptions,
        ) -> Result<Box<dyn motlie_model::BundleHandle>, motlie_model::ModelError> {
            Err(motlie_model::ModelError::InvalidConfiguration(
                "stub bundle is not startable".into(),
            ))
        }
    }

    fn stub_descriptor(id: &str) -> BundleDescriptor {
        BundleDescriptor {
            id: BundleId::new(id),
            display_name: format!("Bundle {id}"),
            family: BundleFamily::Embeddings,
            support_tier: SupportTier::Experimental,
            capabilities: Capabilities::embeddings_only(),
            packaging: PackagingMode::Sidecar,
            backend: BackendKind::MistralRs,
            requirements: BundleRequirements::default(),
            eval_tracks: vec![EvalTrack::Embeddings],
            artifacts: None,
        }
    }

    #[test]
    fn register_overwrites_prior_descriptor() {
        let mut catalog = Catalog::new();

        let first = stub_descriptor("bundle");
        let second = BundleDescriptor {
            display_name: "Bundle v2".into(),
            ..stub_descriptor("bundle")
        };
        let first_for_factory = first.clone();
        let second_for_factory = second.clone();

        assert!(catalog
            .register(first.clone(), move || {
                Box::new(StubBundle {
                    metadata: BundleMetadata {
                        id: first_for_factory.id.clone(),
                        display_name: first_for_factory.display_name.clone(),
                        capabilities: first_for_factory.capabilities.clone(),
                    },
                })
            })
            .is_none());

        let replaced = catalog.register(second.clone(), move || {
            Box::new(StubBundle {
                metadata: BundleMetadata {
                    id: second_for_factory.id.clone(),
                    display_name: second_for_factory.display_name.clone(),
                    capabilities: second_for_factory.capabilities.clone(),
                },
            })
        });

        assert_eq!(replaced, Some(first));
        assert_eq!(
            catalog
                .bundle(&BundleId::new("bundle"))
                .map(|bundle| &bundle.display_name),
            Some(&"Bundle v2".to_string())
        );
    }

    #[test]
    fn defaults_include_embeddinggemma_bundle_and_artifact_control() {
        let catalog = Catalog::with_defaults();
        let bundle_id = BundleId::new("embeddinggemma_300m");

        assert_eq!(catalog.len(), 1);
        assert!(catalog.instantiate(&bundle_id).is_some());
        assert!(catalog
            .bundles_for_track(EvalTrack::Embeddings)
            .any(|bundle| bundle.id == bundle_id));

        let artifacts = catalog
            .artifacts(&bundle_id)
            .expect("default embedder should expose artifact control");
        assert_eq!(artifacts.control_name, "embeddinggemma_300m");
    }

    #[test]
    fn artifact_rules_match_expected_files() {
        let artifacts = BundleArtifacts {
            control_name: "embeddinggemma_300m",
            source: ArtifactSource::HuggingFace {
                repo: "google/embeddinggemma-300m",
            },
            include: vec![
                ArtifactRule::Exact("config.json"),
                ArtifactRule::Suffix(".safetensors"),
            ],
        };

        assert!(artifacts.includes("config.json"));
        assert!(artifacts.includes("weights-00001.safetensors"));
        assert!(!artifacts.includes("README.md"));
    }
}
