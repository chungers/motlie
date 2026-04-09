//! Curated model bundle catalog for the Motlie ecosystem.
//!
//! This crate owns the bundle/catalog layer above `motlie-model`.

use std::collections::BTreeMap;
use std::error::Error as StdError;
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

pub mod chat;
pub mod embeddings;

use hf_hub::api::sync::ApiBuilder;
use thiserror::Error;

pub use chat::ChatModels;
pub use embeddings::EmbeddingModels;
pub use motlie_model::eval::EvalTrack;
pub use motlie_model::{
    BundleId, Capabilities, CapabilityDescriptor, CapabilityKind, ContentKind, InteractionStyle,
    ModelBundle,
};

type BoxError = Box<dyn StdError + Send + Sync + 'static>;

#[derive(Debug, Error)]
pub enum ModelsError {
    #[error("unknown bundle `{bundle_id}`")]
    UnknownBundle { bundle_id: BundleId },
    #[error("bundle `{bundle_id}` does not define curated artifacts")]
    MissingArtifacts { bundle_id: BundleId },
    #[error("failed to create artifact root `{path}`")]
    CreateArtifactRoot {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to create Hugging Face API client")]
    HuggingFaceClient {
        #[source]
        source: BoxError,
    },
    #[error("failed to inspect model repo `{repo}`")]
    InspectModelRepo {
        repo: &'static str,
        #[source]
        source: BoxError,
    },
    #[error("failed to download `{filename}` from repo `{repo}`")]
    DownloadArtifact {
        repo: &'static str,
        filename: String,
        #[source]
        source: BoxError,
    },
    #[error("unknown embedding model selector `{selector}`")]
    UnknownEmbeddingModel { selector: String },
    #[error("unknown chat model selector `{selector}`")]
    UnknownChatModel { selector: String },
    #[error("unknown model selector `{selector}`")]
    UnknownModelSelector { selector: String },
    #[error("model selector `{selector}` is unavailable in this build")]
    ModelUnavailable { selector: String },
}

pub type Result<T> = std::result::Result<T, ModelsError>;

/// Resolve a Hugging Face cache root to the concrete snapshot directory for a model.
///
/// Validates that `config.json`, a tokenizer file, and at least one weight file
/// are present. Returns the snapshot directory path suitable for passing to a
/// backend as `ArtifactPolicy::LocalOnly { root }`.
pub fn resolve_hf_snapshot(
    model_id: &str,
    cache_root: &Path,
) -> std::result::Result<PathBuf, motlie_model::ModelError> {
    use hf_hub::{Cache, Repo, RepoType};

    let repo = Cache::new(cache_root.to_path_buf())
        .repo(Repo::new(model_id.to_owned(), RepoType::Model));

    let config = repo.get("config.json").ok_or_else(|| {
        motlie_model::ModelError::InvalidConfiguration(format!(
            "artifact policy `LocalOnly` requires cached `config.json` for `{model_id}` under `{}`",
            cache_root.display()
        ))
    })?;

    if repo.get("tokenizer.json").is_none() && repo.get("tokenizer.model").is_none() {
        return Err(motlie_model::ModelError::InvalidConfiguration(format!(
            "artifact policy `LocalOnly` requires cached tokenizer files for `{model_id}` under `{}`",
            cache_root.display()
        )));
    }

    let snapshot_dir = config.parent().ok_or_else(|| {
        motlie_model::ModelError::InvalidConfiguration(format!(
            "artifact policy `LocalOnly` found invalid cache layout for `{model_id}` under `{}`",
            cache_root.display()
        ))
    })?;

    let has_weights = std::fs::read_dir(snapshot_dir)
        .map_err(|err| {
            motlie_model::ModelError::InvalidConfiguration(format!(
                "failed to inspect cached artifacts for `{model_id}` in `{}`: {err}",
                snapshot_dir.display()
            ))
        })?
        .filter_map(std::result::Result::ok)
        .any(|entry| {
            entry
                .file_name()
                .to_str()
                .map(|name| {
                    name.ends_with(".safetensors") || name.ends_with(".safetensors.index.json")
                })
                .unwrap_or(false)
        });

    if !has_weights {
        return Err(motlie_model::ModelError::InvalidConfiguration(format!(
            "artifact policy `LocalOnly` requires cached weight files for `{model_id}` under `{}`",
            cache_root.display()
        )));
    }

    Ok(snapshot_dir.to_path_buf())
}

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
        .ok_or_else(|| ModelsError::UnknownBundle {
            bundle_id: bundle_id.clone(),
        })?;
    let artifacts = descriptor
        .artifacts
        .as_ref()
        .ok_or_else(|| ModelsError::MissingArtifacts {
            bundle_id: bundle_id.clone(),
        })?;

    match &artifacts.source {
        ArtifactSource::HuggingFace { repo } => {
            std::fs::create_dir_all(artifact_root).map_err(|source| {
                ModelsError::CreateArtifactRoot {
                    path: artifact_root.to_path_buf(),
                    source,
                }
            })?;

            let api = ApiBuilder::new()
                .with_cache_dir(artifact_root.to_path_buf())
                .with_token(options.hf_token.clone())
                .build()
                .map_err(|source| ModelsError::HuggingFaceClient {
                    source: Box::new(source),
                })?;
            let repo_api = api.model((*repo).to_string());
            let info = repo_api
                .info()
                .map_err(|source| ModelsError::InspectModelRepo {
                    repo,
                    source: Box::new(source),
                })?;

            let mut downloaded = Vec::new();
            for sibling in info.siblings {
                if artifacts.includes(&sibling.rfilename) {
                    let path = repo_api.get(&sibling.rfilename).map_err(|source| {
                        ModelsError::DownloadArtifact {
                            repo,
                            filename: sibling.rfilename.clone(),
                            source: Box::new(source),
                        }
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
    pub capabilities: Capabilities,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelSelector {
    Chat(ChatModels),
    Embedding(EmbeddingModels),
}

impl ModelSelector {
    pub fn as_str(&self) -> String {
        match self {
            Self::Chat(model) => format!("chat:{}", model.as_str()),
            Self::Embedding(model) => format!("embedding:{}", model.as_str()),
        }
    }

    pub fn bundle_id(&self) -> BundleId {
        match self {
            Self::Chat(model) => model.bundle_id(),
            Self::Embedding(model) => model.bundle_id(),
        }
    }

    pub fn descriptor(&self) -> BundleDescriptor {
        match self {
            Self::Chat(model) => model.descriptor(),
            Self::Embedding(model) => model.descriptor(),
        }
    }

    pub fn bundle(&self) -> Box<dyn ModelBundle> {
        match self {
            Self::Chat(model) => model.bundle(),
            Self::Embedding(model) => model.bundle(),
        }
    }
}

impl fmt::Display for ModelSelector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.as_str())
    }
}

impl FromStr for ModelSelector {
    type Err = ModelsError;

    fn from_str(value: &str) -> Result<Self> {
        if let Some(raw) = value.strip_prefix("chat:") {
            #[cfg(not(feature = "model-qwen3-4b"))]
            if raw == chat::QWEN3_4B_SELECTOR {
                return Err(ModelsError::ModelUnavailable {
                    selector: value.to_owned(),
                });
            }
            return Ok(Self::Chat(raw.parse()?));
        }

        if let Some(raw) = value.strip_prefix("embedding:") {
            #[cfg(not(feature = "model-google-gemma-300m"))]
            if raw == embeddings::GOOGLE_GEMMA_300M_SELECTOR {
                return Err(ModelsError::ModelUnavailable {
                    selector: value.to_owned(),
                });
            }
            return Ok(Self::Embedding(raw.parse()?));
        }

        Err(ModelsError::UnknownModelSelector {
            selector: value.to_owned(),
        })
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
        #[allow(unused_mut)]
        let mut catalog = Self::new();
        #[cfg(feature = "model-google-gemma-300m")]
        catalog.register(embeddings::google_gemma_300m::descriptor(), || {
            embeddings::google_gemma_300m::bundle()
        });
        #[cfg(feature = "model-qwen3-4b")]
        catalog.register(chat::qwen3_4b::descriptor(), || {
            chat::qwen3_4b::bundle()
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
        ) -> std::result::Result<Box<dyn motlie_model::BundleHandle>, motlie_model::ModelError>
        {
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
            capabilities: Capabilities::embeddings_only(),
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

        assert!(
            catalog
                .register(first.clone(), move || {
                    Box::new(StubBundle {
                        metadata: BundleMetadata {
                            id: first_for_factory.id.clone(),
                            display_name: first_for_factory.display_name.clone(),
                            capabilities: first_for_factory.capabilities.clone(),
                        },
                    })
                })
                .is_none()
        );

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

        #[cfg(feature = "model-google-gemma-300m")]
        {
            assert!(catalog.len() >= 1);
            assert!(catalog.instantiate(&bundle_id).is_some());
            assert!(
                catalog
                    .bundles_for_track(EvalTrack::Embeddings)
                    .any(|bundle| bundle.id == bundle_id)
            );

            let artifacts = catalog
                .artifacts(&bundle_id)
                .expect("default embedder should expose artifact control");
            assert_eq!(artifacts.control_name, "embeddinggemma_300m");
        }

        #[cfg(not(feature = "model-google-gemma-300m"))]
        {
            assert_eq!(catalog.len(), 0);
            assert!(catalog.instantiate(&bundle_id).is_none());
            assert!(catalog.artifacts(&bundle_id).is_none());
        }
    }

    #[test]
    fn embedding_models_round_trip_string_selectors() {
        #[cfg(feature = "model-google-gemma-300m")]
        {
            let model: EmbeddingModels = "google/embeddinggemma_300m"
                .parse()
                .expect("known embedding selector should parse");

            assert_eq!(model, EmbeddingModels::GoogleGemma300m);
            assert_eq!(model.to_string(), "google/embeddinggemma_300m");
        }
    }

    #[test]
    fn model_selector_parses_embedding_prefix() {
        #[cfg(feature = "model-google-gemma-300m")]
        {
            let selector: ModelSelector = "embedding:google/embeddinggemma_300m"
                .parse()
                .expect("known embedding model selector should parse");

            assert_eq!(
                selector,
                ModelSelector::Embedding(EmbeddingModels::GoogleGemma300m)
            );
            assert_eq!(selector.to_string(), "embedding:google/embeddinggemma_300m");
        }
    }

    #[cfg(not(feature = "model-google-gemma-300m"))]
    #[test]
    fn selector_reports_unavailable_for_disabled_bundles() {
        let err = "embedding:google/embeddinggemma_300m"
            .parse::<ModelSelector>()
            .expect_err("disabled known selector should be unavailable");

        assert!(matches!(
            err,
            ModelsError::ModelUnavailable { selector }
            if selector == "embedding:google/embeddinggemma_300m"
        ));
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
