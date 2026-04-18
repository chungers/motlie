//! Curated model bundle catalog for the Motlie ecosystem.
//!
//! This crate owns the bundle/catalog layer above `motlie-model`.

use std::collections::BTreeMap;
use std::error::Error as StdError;
#[cfg(any(
    feature = "model-qwen3-4b",
    feature = "model-qwen3-4b-gguf",
    feature = "model-google-gemma-300m",
    feature = "model-qwen3-embedding-06b",
    feature = "model-gemma4-e2b",
    feature = "model-gemma4-e2b-gguf",
    feature = "model-piper-en-us-ljspeech-medium",
    feature = "model-qwen3-tts-cpp",
    feature = "model-qwen3-tts-0_6b",
    feature = "model-moonshine-streaming",
    feature = "model-sherpa-onnx-streaming",
    feature = "model-whisper-base-en",
))]
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

pub mod asr;
pub mod chat;
pub mod embeddings;
pub mod tts;

use hf_hub::api::sync::ApiBuilder;
use thiserror::Error;

pub use asr::AsrModels;
pub use chat::ChatModels;
pub use embeddings::EmbeddingModels;
use motlie_model::{
    ArtifactPolicy, BackendAdapter, BundleMetadata, ModelError, ResolvedCheckpoint, StartOptions,
};
pub use motlie_model::{
    ArtifactRule, ArtifactSource, BackendKind, BuildConstraint, BundleFamily, BundleId,
    BundleRequirements, Capabilities, CapabilityDescriptor, CapabilityKind, CheckpointFormat,
    ContentKind, EvalTrack, InteractionStyle, ModelBundle, ModelCheckpoint, ModelIdentity,
    PlatformConstraint, QuantizationSupport,
};
pub use tts::TtsModels;

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
    #[error("unknown ASR model selector `{selector}`")]
    UnknownAsrModel { selector: String },
    #[error("unknown TTS model selector `{selector}`")]
    UnknownTtsModel { selector: String },
    #[error("unknown chat model selector `{selector}`")]
    UnknownChatModel { selector: String },
    #[error("unknown model selector `{selector}`")]
    UnknownModelSelector { selector: String },
    #[error("model selector `{selector}` is unavailable in this build")]
    ModelUnavailable { selector: String },
    #[error("no curated embedding models are enabled in this build")]
    NoEmbeddingModelsEnabled,
    #[error("expected exactly one curated embedding model in this build, found {count}")]
    AmbiguousEmbeddingModelSelection { count: usize },
}

pub type Result<T> = std::result::Result<T, ModelsError>;

fn models_error_to_model_error(error: ModelsError) -> ModelError {
    ModelError::InvalidConfiguration(error.to_string())
}

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

    let repo =
        Cache::new(cache_root.to_path_buf()).repo(Repo::new(model_id.to_owned(), RepoType::Model));

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

/// Resolve a Hugging Face cache root to the snapshot directory for a GGUF model.
///
/// GGUF repos contain only `.gguf` weight files (no config.json or tokenizer).
/// This function navigates the HF cache layout and validates that at least one
/// `.gguf` file is present. Returns the snapshot directory path.
pub fn resolve_hf_gguf_snapshot(
    model_id: &str,
    cache_root: &Path,
) -> std::result::Result<PathBuf, motlie_model::ModelError> {
    // GGUF repos don't have config.json. Probe for any .gguf file via the refs/snapshots layout.
    let repo_folder = format!("models--{}", model_id.replace('/', "--"));
    let repo_root = cache_root.join(&repo_folder);
    let refs_dir = repo_root.join("refs");

    let main_ref = refs_dir.join("main");
    if !main_ref.exists() {
        return Err(motlie_model::ModelError::InvalidConfiguration(format!(
            "artifact policy `LocalOnly` requires cached GGUF artifacts for `{model_id}` under `{}`; \
             no refs/main found — run the download step first",
            cache_root.display()
        )));
    }

    let commit = std::fs::read_to_string(&main_ref).map_err(|err| {
        motlie_model::ModelError::InvalidConfiguration(format!(
            "failed to read HF cache ref for `{model_id}`: {err}"
        ))
    })?;
    let commit = commit.trim();

    let snapshot_dir = repo_root.join("snapshots").join(commit);
    if !snapshot_dir.exists() {
        return Err(motlie_model::ModelError::InvalidConfiguration(format!(
            "HF cache snapshot `{commit}` for `{model_id}` not found under `{}`",
            cache_root.display()
        )));
    }

    let has_gguf = std::fs::read_dir(&snapshot_dir)
        .map_err(|err| {
            motlie_model::ModelError::InvalidConfiguration(format!(
                "failed to inspect cached GGUF artifacts for `{model_id}` in `{}`: {err}",
                snapshot_dir.display()
            ))
        })?
        .filter_map(std::result::Result::ok)
        .any(|entry| {
            entry
                .file_name()
                .to_str()
                .map(|name| name.ends_with(".gguf"))
                .unwrap_or(false)
        });

    if !has_gguf {
        return Err(motlie_model::ModelError::InvalidConfiguration(format!(
            "artifact policy `LocalOnly` requires cached .gguf files for `{model_id}` under `{}`",
            cache_root.display()
        )));
    }

    Ok(snapshot_dir)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BundleArtifacts {
    pub control_name: &'static str,
    pub format: CheckpointFormat,
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

    let downloaded = download_checkpoint_artifacts_with_options(
        &ModelCheckpoint {
            format: artifacts.format,
            source: artifacts.source.clone(),
            include: artifacts.include.clone(),
            quantization: None,
        },
        artifact_root,
        options,
    )?;

    Ok(ArtifactDownloadSummary {
        bundle_id: bundle_id.clone(),
        downloaded,
    })
}

fn download_checkpoint_artifacts_with_options(
    checkpoint: &ModelCheckpoint,
    artifact_root: &Path,
    options: &ArtifactDownloadOptions,
) -> Result<Vec<PathBuf>> {
    match &checkpoint.source {
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
                if checkpoint.includes(&sibling.rfilename) {
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
            Ok(downloaded)
        }
    }
}

/// Product-facing descriptor for a curated model bundle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BundleDescriptor {
    pub id: BundleId,
    pub model_id: BundleId,
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

    pub fn identity(&self) -> ModelIdentity {
        ModelIdentity {
            id: self.model_id.clone(),
            display_name: self.display_name.clone(),
            family: self.family.clone(),
            capabilities: self.capabilities.clone(),
            eval_tracks: self.eval_tracks.clone(),
            requirements: self.requirements.clone(),
        }
    }

    pub fn checkpoint(&self) -> Option<ModelCheckpoint> {
        self.artifacts.as_ref().map(|artifacts| ModelCheckpoint {
            format: artifacts.format,
            source: artifacts.source.clone(),
            include: artifacts.include.clone(),
            quantization: None,
        })
    }
}

pub(crate) fn bundle_artifacts_from_checkpoint(
    control_name: &'static str,
    checkpoint: &ModelCheckpoint,
) -> BundleArtifacts {
    BundleArtifacts {
        control_name,
        format: checkpoint.format,
        source: checkpoint.source.clone(),
        include: checkpoint.include.clone(),
    }
}

impl ArtifactDownloadOptions {
    fn from_env() -> Self {
        Self {
            hf_token: std::env::var("HF_TOKEN")
                .ok()
                .or_else(|| std::env::var("HUGGING_FACE_HUB_TOKEN").ok()),
        }
    }
}

trait LocalCheckpointResolver: Send + Sync {
    fn resolve(&self, root: &Path) -> std::result::Result<PathBuf, ModelError>;
}

impl<F> LocalCheckpointResolver for F
where
    F: Fn(&Path) -> std::result::Result<PathBuf, ModelError> + Send + Sync + 'static,
{
    fn resolve(&self, root: &Path) -> std::result::Result<PathBuf, ModelError> {
        (self)(root)
    }
}

#[derive(Clone)]
struct AdapterBackedBundle {
    metadata: BundleMetadata,
    identity: ModelIdentity,
    checkpoint: ModelCheckpoint,
    adapter: Arc<dyn BackendAdapter>,
    local_resolver: Arc<dyn LocalCheckpointResolver>,
}

#[async_trait::async_trait]
impl ModelBundle for AdapterBackedBundle {
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
        options: StartOptions,
    ) -> std::result::Result<Box<dyn motlie_model::BundleHandle>, ModelError> {
        let StartOptions {
            artifact_policy,
            quantization,
            unpack_root,
            max_concurrency,
        } = options;

        let artifact_root = match artifact_policy {
            Some(ArtifactPolicy::LocalOnly { root }) => root,
            Some(ArtifactPolicy::AllowFetch { root }) => {
                let root = root.unwrap_or_else(default_artifact_root);
                download_checkpoint_artifacts_with_options(
                    &self.checkpoint,
                    &root,
                    &ArtifactDownloadOptions::from_env(),
                )
                .map_err(models_error_to_model_error)?;
                root
            }
            None => default_artifact_root(),
        };

        let resolved = ResolvedCheckpoint {
            checkpoint: self.checkpoint.clone(),
            path: self.local_resolver.resolve(&artifact_root)?,
        };

        self.adapter
            .start(
                &self.identity,
                &resolved,
                StartOptions {
                    artifact_policy: None,
                    quantization,
                    unpack_root,
                    max_concurrency,
                },
            )
            .await
    }
}

pub(crate) fn adapter_backed_bundle(
    bundle_id: BundleId,
    display_name: String,
    identity: ModelIdentity,
    checkpoint: ModelCheckpoint,
    adapter: Arc<dyn BackendAdapter>,
    local_resolver: Arc<dyn LocalCheckpointResolver>,
) -> Box<dyn ModelBundle> {
    Box::new(AdapterBackedBundle {
        metadata: BundleMetadata {
            id: bundle_id,
            display_name,
            capabilities: adapter.capabilities().clone(),
            quantization: adapter.quantization().clone(),
        },
        identity,
        checkpoint,
        adapter,
        local_resolver,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelVariantDescriptor {
    pub backend: BackendKind,
    pub capabilities: Capabilities,
    pub quantization: QuantizationSupport,
    pub checkpoint: ModelCheckpoint,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ResolveModelOptions {
    pub backend_preference: Option<BackendKind>,
    pub format_preference: Option<CheckpointFormat>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedModelDescriptor {
    pub identity: ModelIdentity,
    pub variant: ModelVariantDescriptor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ModelSelector {
    #[cfg(any(
        feature = "model-piper-en-us-ljspeech-medium",
        feature = "model-qwen3-tts-cpp",
        feature = "model-qwen3-tts-0_6b",
    ))]
    Tts(TtsModels),
    #[cfg(any(
        feature = "model-moonshine-streaming",
        feature = "model-sherpa-onnx-streaming",
        feature = "model-whisper-base-en"
    ))]
    Asr(AsrModels),
    #[cfg(any(
        feature = "model-qwen3-4b",
        feature = "model-qwen3-4b-gguf",
        feature = "model-gemma4-e2b",
        feature = "model-gemma4-e2b-gguf",
    ))]
    Chat(ChatModels),
    #[cfg(any(
        feature = "model-google-gemma-300m",
        feature = "model-qwen3-embedding-06b"
    ))]
    Embedding(EmbeddingModels),
}

#[cfg(any(
    feature = "model-qwen3-4b",
    feature = "model-qwen3-4b-gguf",
    feature = "model-google-gemma-300m",
    feature = "model-qwen3-embedding-06b",
    feature = "model-gemma4-e2b",
    feature = "model-gemma4-e2b-gguf",
    feature = "model-piper-en-us-ljspeech-medium",
    feature = "model-qwen3-tts-cpp",
    feature = "model-qwen3-tts-0_6b",
    feature = "model-moonshine-streaming",
    feature = "model-sherpa-onnx-streaming",
    feature = "model-whisper-base-en",
))]
impl ModelSelector {
    pub fn as_str(&self) -> String {
        match self {
            #[cfg(any(
                feature = "model-piper-en-us-ljspeech-medium",
                feature = "model-qwen3-tts-cpp",
                feature = "model-qwen3-tts-0_6b"
            ))]
            Self::Tts(model) => format!("tts:{}", model.as_str()),
            #[cfg(any(
                feature = "model-moonshine-streaming",
                feature = "model-sherpa-onnx-streaming",
                feature = "model-whisper-base-en"
            ))]
            Self::Asr(model) => format!("asr:{}", model.as_str()),
            #[cfg(any(
                feature = "model-qwen3-4b",
                feature = "model-qwen3-4b-gguf",
                feature = "model-gemma4-e2b",
                feature = "model-gemma4-e2b-gguf",
            ))]
            Self::Chat(model) => format!("chat:{}", model.as_str()),
            #[cfg(any(
                feature = "model-google-gemma-300m",
                feature = "model-qwen3-embedding-06b"
            ))]
            Self::Embedding(model) => format!("embedding:{}", model.as_str()),
        }
    }

    pub fn bundle_id(&self) -> BundleId {
        match self {
            #[cfg(any(
                feature = "model-piper-en-us-ljspeech-medium",
                feature = "model-qwen3-tts-cpp",
                feature = "model-qwen3-tts-0_6b"
            ))]
            Self::Tts(model) => model.bundle_id(),
            #[cfg(any(
                feature = "model-moonshine-streaming",
                feature = "model-sherpa-onnx-streaming",
                feature = "model-whisper-base-en"
            ))]
            Self::Asr(model) => model.bundle_id(),
            #[cfg(any(
                feature = "model-qwen3-4b",
                feature = "model-qwen3-4b-gguf",
                feature = "model-gemma4-e2b",
                feature = "model-gemma4-e2b-gguf",
            ))]
            Self::Chat(model) => model.bundle_id(),
            #[cfg(any(
                feature = "model-google-gemma-300m",
                feature = "model-qwen3-embedding-06b"
            ))]
            Self::Embedding(model) => model.bundle_id(),
        }
    }

    pub fn descriptor(&self) -> BundleDescriptor {
        match self {
            #[cfg(any(
                feature = "model-piper-en-us-ljspeech-medium",
                feature = "model-qwen3-tts-cpp",
                feature = "model-qwen3-tts-0_6b"
            ))]
            Self::Tts(model) => model.descriptor(),
            #[cfg(any(
                feature = "model-moonshine-streaming",
                feature = "model-sherpa-onnx-streaming",
                feature = "model-whisper-base-en"
            ))]
            Self::Asr(model) => model.descriptor(),
            #[cfg(any(
                feature = "model-qwen3-4b",
                feature = "model-qwen3-4b-gguf",
                feature = "model-gemma4-e2b",
                feature = "model-gemma4-e2b-gguf",
            ))]
            Self::Chat(model) => model.descriptor(),
            #[cfg(any(
                feature = "model-google-gemma-300m",
                feature = "model-qwen3-embedding-06b"
            ))]
            Self::Embedding(model) => model.descriptor(),
        }
    }

    pub fn bundle(&self) -> Result<Box<dyn ModelBundle>> {
        match self {
            #[cfg(any(
                feature = "model-piper-en-us-ljspeech-medium",
                feature = "model-qwen3-tts-cpp",
                feature = "model-qwen3-tts-0_6b"
            ))]
            Self::Tts(model) => Err(ModelsError::ModelUnavailable {
                selector: format!(
                    "typed-only selector `{}` does not support erased bundle startup",
                    model.as_str()
                ),
            }),
            #[cfg(any(
                feature = "model-moonshine-streaming",
                feature = "model-sherpa-onnx-streaming",
                feature = "model-whisper-base-en"
            ))]
            Self::Asr(model) => Err(ModelsError::ModelUnavailable {
                selector: format!(
                    "typed-only selector `{}` does not support erased bundle startup",
                    model.as_str()
                ),
            }),
            #[cfg(any(
                feature = "model-qwen3-4b",
                feature = "model-qwen3-4b-gguf",
                feature = "model-gemma4-e2b",
                feature = "model-gemma4-e2b-gguf",
            ))]
            Self::Chat(model) => Ok(model.bundle()),
            #[cfg(any(
                feature = "model-google-gemma-300m",
                feature = "model-qwen3-embedding-06b"
            ))]
            Self::Embedding(model) => Ok(model.bundle()),
        }
    }
}

#[cfg(any(
    feature = "model-qwen3-4b",
    feature = "model-qwen3-4b-gguf",
    feature = "model-google-gemma-300m",
    feature = "model-qwen3-embedding-06b",
    feature = "model-gemma4-e2b",
    feature = "model-gemma4-e2b-gguf",
    feature = "model-piper-en-us-ljspeech-medium",
    feature = "model-qwen3-tts-cpp",
    feature = "model-qwen3-tts-0_6b",
    feature = "model-moonshine-streaming",
    feature = "model-sherpa-onnx-streaming",
    feature = "model-whisper-base-en",
))]
impl fmt::Display for ModelSelector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.as_str())
    }
}

impl FromStr for ModelSelector {
    type Err = ModelsError;

    fn from_str(value: &str) -> Result<Self> {
        if let Some(raw) = value.strip_prefix("tts:") {
            #[cfg(not(feature = "model-piper-en-us-ljspeech-medium"))]
            if raw == tts::PIPER_EN_US_LJSPEECH_MEDIUM_SELECTOR {
                return Err(ModelsError::ModelUnavailable {
                    selector: value.to_owned(),
                });
            }
            #[cfg(not(feature = "model-qwen3-tts-cpp"))]
            if raw == tts::QWEN3_TTS_CPP_0_6B_SELECTOR {
                return Err(ModelsError::ModelUnavailable {
                    selector: value.to_owned(),
                });
            }
            #[cfg(not(feature = "model-qwen3-tts-0_6b"))]
            if raw == tts::QWEN3_TTS_12HZ_0_6B_SELECTOR {
                return Err(ModelsError::ModelUnavailable {
                    selector: value.to_owned(),
                });
            }
            #[cfg(any(
                feature = "model-piper-en-us-ljspeech-medium",
                feature = "model-qwen3-tts-cpp",
                feature = "model-qwen3-tts-0_6b"
            ))]
            return Ok(Self::Tts(raw.parse()?));
            #[cfg(not(any(
                feature = "model-piper-en-us-ljspeech-medium",
                feature = "model-qwen3-tts-cpp",
                feature = "model-qwen3-tts-0_6b"
            )))]
            return Err(ModelsError::UnknownModelSelector {
                selector: value.to_owned(),
            });
        }

        if let Some(raw) = value.strip_prefix("chat:") {
            #[cfg(not(feature = "model-gemma4-e2b"))]
            if raw == chat::GEMMA4_E2B_SELECTOR {
                return Err(ModelsError::ModelUnavailable {
                    selector: value.to_owned(),
                });
            }
            #[cfg(not(feature = "model-gemma4-e2b-gguf"))]
            if raw == chat::GEMMA4_E2B_GGUF_SELECTOR {
                return Err(ModelsError::ModelUnavailable {
                    selector: value.to_owned(),
                });
            }
            #[cfg(not(feature = "model-qwen3-4b"))]
            if raw == chat::QWEN3_4B_SELECTOR {
                return Err(ModelsError::ModelUnavailable {
                    selector: value.to_owned(),
                });
            }
            #[cfg(not(feature = "model-qwen3-4b-gguf"))]
            if raw == chat::QWEN3_4B_GGUF_SELECTOR {
                return Err(ModelsError::ModelUnavailable {
                    selector: value.to_owned(),
                });
            }
            #[cfg(any(
                feature = "model-qwen3-4b",
                feature = "model-qwen3-4b-gguf",
                feature = "model-gemma4-e2b",
                feature = "model-gemma4-e2b-gguf",
            ))]
            return Ok(Self::Chat(raw.parse()?));
            #[cfg(not(any(
                feature = "model-qwen3-4b",
                feature = "model-qwen3-4b-gguf",
                feature = "model-gemma4-e2b",
                feature = "model-gemma4-e2b-gguf",
            )))]
            return Err(ModelsError::UnknownModelSelector {
                selector: value.to_owned(),
            });
        }

        if let Some(raw) = value.strip_prefix("embedding:") {
            #[cfg(not(feature = "model-google-gemma-300m"))]
            if raw == embeddings::GOOGLE_GEMMA_300M_SELECTOR {
                return Err(ModelsError::ModelUnavailable {
                    selector: value.to_owned(),
                });
            }
            #[cfg(not(feature = "model-qwen3-embedding-06b"))]
            if raw == embeddings::QWEN3_EMBEDDING_06B_SELECTOR {
                return Err(ModelsError::ModelUnavailable {
                    selector: value.to_owned(),
                });
            }
            #[cfg(any(
                feature = "model-google-gemma-300m",
                feature = "model-qwen3-embedding-06b"
            ))]
            return Ok(Self::Embedding(raw.parse()?));
            #[cfg(not(any(
                feature = "model-google-gemma-300m",
                feature = "model-qwen3-embedding-06b"
            )))]
            return Err(ModelsError::UnknownModelSelector {
                selector: value.to_owned(),
            });
        }

        if let Some(raw) = value.strip_prefix("asr:") {
            #[cfg(not(feature = "model-moonshine-streaming"))]
            if raw == asr::MOONSHINE_STREAMING_SELECTOR {
                return Err(ModelsError::ModelUnavailable {
                    selector: value.to_owned(),
                });
            }
            #[cfg(not(feature = "model-sherpa-onnx-streaming"))]
            if raw == asr::SHERPA_ONNX_STREAMING_SELECTOR {
                return Err(ModelsError::ModelUnavailable {
                    selector: value.to_owned(),
                });
            }
            #[cfg(not(feature = "model-whisper-base-en"))]
            if raw == asr::WHISPER_BASE_EN_SELECTOR {
                return Err(ModelsError::ModelUnavailable {
                    selector: value.to_owned(),
                });
            }
            #[cfg(any(
                feature = "model-moonshine-streaming",
                feature = "model-sherpa-onnx-streaming",
                feature = "model-whisper-base-en"
            ))]
            return Ok(Self::Asr(raw.parse()?));
            #[cfg(not(any(
                feature = "model-moonshine-streaming",
                feature = "model-sherpa-onnx-streaming",
                feature = "model-whisper-base-en"
            )))]
            return Err(ModelsError::UnknownModelSelector {
                selector: value.to_owned(),
            });
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
    factory: Option<Arc<dyn BundleFactory>>,
}

#[derive(Clone)]
struct CheckpointCatalogEntry {
    checkpoint: ModelCheckpoint,
    local_resolver: Arc<dyn LocalCheckpointResolver>,
}

struct ModelCatalogEntry {
    identity: ModelIdentity,
    checkpoints: Vec<CheckpointCatalogEntry>,
    adapters: Vec<Arc<dyn BackendAdapter>>,
}

/// In-memory registry of curated bundle descriptors and constructors.
#[derive(Default)]
pub struct Catalog {
    bundles: BTreeMap<BundleId, CatalogEntry>,
    models: BTreeMap<BundleId, ModelCatalogEntry>,
}

impl Catalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_defaults() -> Self {
        #[allow(unused_mut)]
        let mut catalog = Self::new();
        #[cfg(feature = "model-google-gemma-300m")]
        embeddings::google_gemma_300m::register(&mut catalog);
        #[cfg(feature = "model-qwen3-embedding-06b")]
        embeddings::qwen3_embedding_06b::register(&mut catalog);
        #[cfg(feature = "model-qwen3-4b")]
        chat::qwen3_4b::register(&mut catalog);
        #[cfg(feature = "model-gemma4-e2b")]
        chat::gemma4_e2b::register(&mut catalog);
        #[cfg(feature = "model-qwen3-4b-gguf")]
        chat::qwen3_4b_gguf::register(&mut catalog);
        #[cfg(feature = "model-gemma4-e2b-gguf")]
        chat::gemma4_e2b_gguf::register(&mut catalog);
        #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
        tts::piper_en_us_ljspeech_medium::register(&mut catalog);
        #[cfg(feature = "model-qwen3-tts-cpp")]
        tts::qwen3_tts_cpp::register(&mut catalog);
        #[cfg(feature = "model-qwen3-tts-0_6b")]
        tts::qwen3_tts_12hz_0_6b::register(&mut catalog);
        #[cfg(feature = "model-moonshine-streaming")]
        asr::moonshine_streaming_en::register(&mut catalog);
        #[cfg(feature = "model-sherpa-onnx-streaming")]
        asr::sherpa_onnx_streaming_en::register(&mut catalog);
        #[cfg(feature = "model-whisper-base-en")]
        asr::whisper_base_en::register(&mut catalog);
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
                    factory: Some(Arc::new(factory)),
                },
            )
            .map(|entry| entry.descriptor)
    }

    pub fn register_descriptor(
        &mut self,
        descriptor: BundleDescriptor,
    ) -> Option<BundleDescriptor> {
        self.bundles
            .insert(
                descriptor.id.clone(),
                CatalogEntry {
                    descriptor: descriptor.clone(),
                    factory: None,
                },
            )
            .map(|entry| entry.descriptor)
    }

    pub(crate) fn register_model_variant(
        &mut self,
        identity: ModelIdentity,
        checkpoint: ModelCheckpoint,
        local_resolver: Arc<dyn LocalCheckpointResolver>,
        adapter: Arc<dyn BackendAdapter>,
    ) {
        let entry = self
            .models
            .entry(identity.id.clone())
            .or_insert_with(|| ModelCatalogEntry {
                identity: identity.clone(),
                checkpoints: Vec::new(),
                adapters: Vec::new(),
            });
        entry.identity = identity;

        if !entry
            .checkpoints
            .iter()
            .any(|existing| existing.checkpoint == checkpoint)
        {
            entry.checkpoints.push(CheckpointCatalogEntry {
                checkpoint,
                local_resolver,
            });
        }

        if !entry.adapters.iter().any(|existing| {
            existing.backend_kind() == adapter.backend_kind()
                && existing.supported_formats() == adapter.supported_formats()
                && existing.capabilities() == adapter.capabilities()
                && existing.quantization() == adapter.quantization()
        }) {
            entry.adapters.push(adapter);
        }
    }

    pub fn bundle(&self, id: &BundleId) -> Option<&BundleDescriptor> {
        self.bundles.get(id).map(|entry| &entry.descriptor)
    }

    pub fn model(&self, id: &BundleId) -> Option<&ModelIdentity> {
        self.models.get(id).map(|entry| &entry.identity)
    }

    pub fn models(&self) -> impl Iterator<Item = &ModelIdentity> {
        self.models.values().map(|entry| &entry.identity)
    }

    pub fn variants_for_model(
        &self,
        id: &BundleId,
    ) -> Option<impl Iterator<Item = ModelVariantDescriptor>> {
        let entry = self.models.get(id)?;
        let variants = entry
            .adapters
            .iter()
            .flat_map(|adapter| {
                entry
                    .checkpoints
                    .iter()
                    .filter(move |checkpoint| {
                        adapter
                            .supported_formats()
                            .contains(&checkpoint.checkpoint.format)
                    })
                    .map(move |checkpoint| ModelVariantDescriptor {
                        backend: adapter.backend_kind(),
                        capabilities: adapter.capabilities().clone(),
                        quantization: adapter.quantization().clone(),
                        checkpoint: checkpoint.checkpoint.clone(),
                    })
            })
            .collect::<Vec<_>>();
        Some(variants.into_iter())
    }

    pub fn artifacts(&self, id: &BundleId) -> Option<&BundleArtifacts> {
        self.bundle(id)
            .and_then(|descriptor| descriptor.artifacts.as_ref())
    }

    pub fn instantiate(&self, id: &BundleId) -> Option<Box<dyn ModelBundle>> {
        self.bundles
            .get(id)
            .and_then(|entry| entry.factory.as_ref().map(|factory| factory.instantiate()))
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

    pub fn resolve_model(
        &self,
        id: &BundleId,
        options: &ResolveModelOptions,
    ) -> Option<ResolvedModelDescriptor> {
        let entry = self.models.get(id)?;

        let exact = entry.adapters.iter().find_map(|adapter| {
            let backend_ok = options
                .backend_preference
                .is_none_or(|backend| adapter.backend_kind() == backend);
            if !backend_ok {
                return None;
            }

            entry.checkpoints.iter().find_map(|checkpoint| {
                let format_ok = options
                    .format_preference
                    .is_none_or(|format| checkpoint.checkpoint.format == format);
                let supported = adapter
                    .supported_formats()
                    .contains(&checkpoint.checkpoint.format);
                if !format_ok || !supported {
                    return None;
                }

                Some(ModelVariantDescriptor {
                    backend: adapter.backend_kind(),
                    capabilities: adapter.capabilities().clone(),
                    quantization: adapter.quantization().clone(),
                    checkpoint: checkpoint.checkpoint.clone(),
                })
            })
        })?;

        Some(ResolvedModelDescriptor {
            identity: entry.identity.clone(),
            variant: exact,
        })
    }

    pub fn instantiate_resolved(
        &self,
        resolved: &ResolvedModelDescriptor,
    ) -> Option<Box<dyn ModelBundle>> {
        if resolved
            .identity
            .capabilities
            .supports(CapabilityKind::Speech)
            || resolved
                .identity
                .capabilities
                .supports(CapabilityKind::Transcription)
        {
            return None;
        }

        let entry = self.models.get(&resolved.identity.id)?;
        let checkpoint = entry
            .checkpoints
            .iter()
            .find(|existing| existing.checkpoint == resolved.variant.checkpoint)?;
        let adapter = entry.adapters.iter().find(|existing| {
            existing.backend_kind() == resolved.variant.backend
                && existing.capabilities() == &resolved.variant.capabilities
                && existing.quantization() == &resolved.variant.quantization
                && existing
                    .supported_formats()
                    .contains(&resolved.variant.checkpoint.format)
        })?;

        Some(adapter_backed_bundle(
            resolved.identity.id.clone(),
            resolved.identity.display_name.clone(),
            resolved.identity.clone(),
            checkpoint.checkpoint.clone(),
            Arc::clone(adapter),
            Arc::clone(&checkpoint.local_resolver),
        ))
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

    struct StubAdapter {
        backend: BackendKind,
        capabilities: Capabilities,
        quantization: QuantizationSupport,
        supported_formats: Vec<CheckpointFormat>,
    }

    #[async_trait::async_trait]
    impl BackendAdapter for StubAdapter {
        fn supported_formats(&self) -> &[CheckpointFormat] {
            &self.supported_formats
        }

        fn backend_kind(&self) -> BackendKind {
            self.backend
        }

        fn capabilities(&self) -> &Capabilities {
            &self.capabilities
        }

        fn quantization(&self) -> &QuantizationSupport {
            &self.quantization
        }

        async fn start(
            &self,
            _identity: &ModelIdentity,
            _checkpoint: &ResolvedCheckpoint,
            _options: StartOptions,
        ) -> std::result::Result<Box<dyn motlie_model::BundleHandle>, ModelError> {
            Err(ModelError::InvalidConfiguration(
                "stub adapter is not startable".into(),
            ))
        }
    }

    fn stub_descriptor(id: &str) -> BundleDescriptor {
        BundleDescriptor {
            id: BundleId::new(id),
            model_id: BundleId::new(id),
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
                            quantization: motlie_model::QuantizationSupport::none(),
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
                    quantization: motlie_model::QuantizationSupport::none(),
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
        assert!(catalog.model(&BundleId::new("bundle")).is_none());
    }

    #[test]
    fn resolve_model_prefers_requested_backend_and_format() {
        let mut catalog = Catalog::new();

        let mistral = BundleDescriptor {
            artifacts: Some(BundleArtifacts {
                control_name: "qwen3_4b",
                format: CheckpointFormat::Safetensors,
                source: ArtifactSource::HuggingFace {
                    repo: "Qwen/Qwen3-4B",
                },
                include: vec![ArtifactRule::Exact("config.json")],
            }),
            model_id: BundleId::new("qwen3_4b"),
            display_name: "Qwen3 4B".into(),
            family: BundleFamily::Qwen,
            capabilities: Capabilities::chat_and_completion(),
            backend: BackendKind::MistralRs,
            requirements: BundleRequirements::default(),
            eval_tracks: vec![EvalTrack::Chat],
            id: BundleId::new("qwen3_4b"),
        };
        let llama = BundleDescriptor {
            id: BundleId::new("qwen3_4b_gguf"),
            model_id: BundleId::new("qwen3_4b"),
            backend: BackendKind::LlamaCpp,
            artifacts: Some(BundleArtifacts {
                control_name: "qwen3_4b_gguf",
                format: CheckpointFormat::Gguf,
                source: ArtifactSource::HuggingFace {
                    repo: "Qwen/Qwen3-4B-GGUF",
                },
                include: vec![ArtifactRule::Suffix(".gguf")],
            }),
            ..mistral.clone()
        };
        let mistral_capabilities = mistral.capabilities.clone();
        let llama_capabilities = llama.capabilities.clone();
        let mistral_for_factory = mistral.clone();
        let llama_for_factory = llama.clone();

        catalog.register(mistral.clone(), move || {
            Box::new(StubBundle {
                metadata: BundleMetadata {
                    id: mistral_for_factory.id.clone(),
                    display_name: mistral_for_factory.display_name.clone(),
                    capabilities: mistral_capabilities.clone(),
                    quantization: motlie_model::QuantizationSupport::none(),
                },
            })
        });
        catalog.register(llama.clone(), move || {
            Box::new(StubBundle {
                metadata: BundleMetadata {
                    id: llama_for_factory.id.clone(),
                    display_name: llama_for_factory.display_name.clone(),
                    capabilities: llama_capabilities.clone(),
                    quantization: motlie_model::QuantizationSupport::none(),
                },
            })
        });
        catalog.register_model_variant(
            ModelIdentity {
                id: BundleId::new("qwen3_4b"),
                display_name: "Qwen3 4B".into(),
                family: BundleFamily::Qwen,
                capabilities: Capabilities::chat_and_completion(),
                eval_tracks: vec![EvalTrack::Chat],
                requirements: BundleRequirements::default(),
            },
            mistral
                .checkpoint()
                .expect("mistral descriptor should expose checkpoint"),
            Arc::new(|root: &Path| Ok(root.to_path_buf())),
            Arc::new(StubAdapter {
                backend: BackendKind::MistralRs,
                capabilities: Capabilities::chat_and_completion(),
                quantization: QuantizationSupport::none(),
                supported_formats: vec![CheckpointFormat::Safetensors],
            }),
        );
        catalog.register_model_variant(
            ModelIdentity {
                id: BundleId::new("qwen3_4b"),
                display_name: "Qwen3 4B".into(),
                family: BundleFamily::Qwen,
                capabilities: Capabilities::chat_and_completion(),
                eval_tracks: vec![EvalTrack::Chat],
                requirements: BundleRequirements::default(),
            },
            llama
                .checkpoint()
                .expect("llama descriptor should expose checkpoint"),
            Arc::new(|root: &Path| Ok(root.to_path_buf())),
            Arc::new(StubAdapter {
                backend: BackendKind::LlamaCpp,
                capabilities: Capabilities::chat_and_completion(),
                quantization: QuantizationSupport::with_recommended(
                    [motlie_model::QuantizationBits::Four],
                    motlie_model::QuantizationBits::Four,
                )
                .expect("test quantization support should be valid"),
                supported_formats: vec![CheckpointFormat::Gguf],
            }),
        );

        let resolved = catalog
            .resolve_model(
                &BundleId::new("qwen3_4b"),
                &ResolveModelOptions {
                    backend_preference: Some(BackendKind::LlamaCpp),
                    format_preference: Some(CheckpointFormat::Gguf),
                },
            )
            .expect("requested llama.cpp gguf variant should resolve");

        assert_eq!(resolved.identity.id.as_str(), "qwen3_4b");
        assert_eq!(resolved.variant.backend, BackendKind::LlamaCpp);
        assert_eq!(resolved.variant.checkpoint.format, CheckpointFormat::Gguf);
        assert!(catalog.instantiate_resolved(&resolved).is_some());
    }

    #[test]
    fn defaults_include_curated_embedding_bundles_and_artifact_control() {
        let catalog = Catalog::with_defaults();

        #[cfg(feature = "model-google-gemma-300m")]
        {
            let bundle_id = BundleId::new("embeddinggemma_300m");
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
            let bundle_id = BundleId::new("embeddinggemma_300m");
            assert!(catalog.instantiate(&bundle_id).is_none());
            assert!(catalog.artifacts(&bundle_id).is_none());
        }

        #[cfg(feature = "model-qwen3-embedding-06b")]
        {
            let bundle_id = BundleId::new("qwen3_embedding_06b");
            assert!(catalog.instantiate(&bundle_id).is_some());
            assert!(
                catalog
                    .bundles_for_track(EvalTrack::Embeddings)
                    .any(|bundle| bundle.id == bundle_id)
            );

            let artifacts = catalog
                .artifacts(&bundle_id)
                .expect("default qwen embedder should expose artifact control");
            assert_eq!(artifacts.control_name, "qwen3_embedding_06b");
        }

        #[cfg(not(feature = "model-qwen3-embedding-06b"))]
        {
            let bundle_id = BundleId::new("qwen3_embedding_06b");
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

        #[cfg(feature = "model-qwen3-embedding-06b")]
        {
            let model: EmbeddingModels = "qwen/qwen3_embedding_06b"
                .parse()
                .expect("known qwen embedding selector should parse");

            assert_eq!(model, EmbeddingModels::Qwen3Embedding06B);
            assert_eq!(model.to_string(), "qwen/qwen3_embedding_06b");
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

        #[cfg(feature = "model-qwen3-embedding-06b")]
        {
            let selector: ModelSelector = "embedding:qwen/qwen3_embedding_06b"
                .parse()
                .expect("known qwen embedding model selector should parse");

            assert_eq!(
                selector,
                ModelSelector::Embedding(EmbeddingModels::Qwen3Embedding06B)
            );
            assert_eq!(selector.to_string(), "embedding:qwen/qwen3_embedding_06b");
        }
    }

    #[test]
    fn selector_reports_unavailable_for_disabled_embedding_bundles() {
        #[cfg(not(feature = "model-google-gemma-300m"))]
        {
            let err = "embedding:google/embeddinggemma_300m"
                .parse::<ModelSelector>()
                .expect_err("disabled known gemma selector should be unavailable");

            assert!(matches!(
                err,
                ModelsError::ModelUnavailable { selector }
                if selector == "embedding:google/embeddinggemma_300m"
            ));
        }

        #[cfg(not(feature = "model-qwen3-embedding-06b"))]
        {
            let err = "embedding:qwen/qwen3_embedding_06b"
                .parse::<ModelSelector>()
                .expect_err("disabled known qwen selector should be unavailable");

            assert!(matches!(
                err,
                ModelsError::ModelUnavailable { selector }
                if selector == "embedding:qwen/qwen3_embedding_06b"
            ));
        }
    }

    #[test]
    fn only_enabled_embedding_model_matches_build_shape() {
        #[cfg(all(
            feature = "model-google-gemma-300m",
            not(feature = "model-qwen3-embedding-06b")
        ))]
        {
            assert_eq!(
                EmbeddingModels::only_enabled().expect("single gemma build should succeed"),
                EmbeddingModels::GoogleGemma300m
            );
        }

        #[cfg(all(
            feature = "model-qwen3-embedding-06b",
            not(feature = "model-google-gemma-300m")
        ))]
        {
            assert_eq!(
                EmbeddingModels::only_enabled().expect("single qwen build should succeed"),
                EmbeddingModels::Qwen3Embedding06B
            );
        }

        #[cfg(all(
            feature = "model-google-gemma-300m",
            feature = "model-qwen3-embedding-06b"
        ))]
        {
            let err = EmbeddingModels::only_enabled()
                .expect_err("multi-embedding build should be ambiguous");
            assert!(matches!(
                err,
                ModelsError::AmbiguousEmbeddingModelSelection { count } if count == 2
            ));
        }
    }

    #[test]
    fn chat_models_round_trip_string_selectors() {
        #[cfg(feature = "model-gemma4-e2b")]
        {
            let model: ChatModels = "google/gemma4_e2b"
                .parse()
                .expect("known multimodal chat selector should parse");

            assert_eq!(model, ChatModels::Gemma4E2B);
            assert_eq!(model.to_string(), "google/gemma4_e2b");
        }

        #[cfg(feature = "model-qwen3-4b")]
        {
            let model: ChatModels = "qwen/qwen3_4b"
                .parse()
                .expect("known chat selector should parse");

            assert_eq!(model, ChatModels::Qwen3_4B);
            assert_eq!(model.to_string(), "qwen/qwen3_4b");
        }
    }

    #[test]
    fn model_selector_parses_chat_prefix() {
        #[cfg(feature = "model-gemma4-e2b")]
        {
            let selector: ModelSelector = "chat:google/gemma4_e2b"
                .parse()
                .expect("known multimodal chat model selector should parse");

            assert_eq!(selector, ModelSelector::Chat(ChatModels::Gemma4E2B));
            assert_eq!(selector.to_string(), "chat:google/gemma4_e2b");
        }

        #[cfg(feature = "model-qwen3-4b")]
        {
            let selector: ModelSelector = "chat:qwen/qwen3_4b"
                .parse()
                .expect("known chat model selector should parse");

            assert_eq!(selector, ModelSelector::Chat(ChatModels::Qwen3_4B));
            assert_eq!(selector.to_string(), "chat:qwen/qwen3_4b");
        }
    }

    #[cfg(not(any(feature = "model-gemma4-e2b", feature = "model-qwen3-4b")))]
    #[test]
    fn chat_selector_reports_unavailable_for_disabled_bundles() {
        let gemma_err = "chat:google/gemma4_e2b"
            .parse::<ModelSelector>()
            .expect_err("disabled known gemma chat selector should be unavailable");

        assert!(matches!(
            gemma_err,
            ModelsError::ModelUnavailable { selector }
            if selector == "chat:google/gemma4_e2b"
        ));

        let err = "chat:qwen/qwen3_4b"
            .parse::<ModelSelector>()
            .expect_err("disabled known chat selector should be unavailable");

        assert!(matches!(
            err,
            ModelsError::ModelUnavailable { selector }
            if selector == "chat:qwen/qwen3_4b"
        ));
    }

    #[cfg(all(feature = "model-gemma4-e2b", not(feature = "model-qwen3-4b")))]
    #[test]
    fn qwen_chat_selector_reports_unavailable_when_only_gemma_is_enabled() {
        let err = "chat:qwen/qwen3_4b"
            .parse::<ModelSelector>()
            .expect_err("disabled qwen selector should be unavailable");

        assert!(matches!(
            err,
            ModelsError::ModelUnavailable { selector }
            if selector == "chat:qwen/qwen3_4b"
        ));
    }

    #[cfg(all(not(feature = "model-gemma4-e2b"), feature = "model-qwen3-4b"))]
    #[test]
    fn gemma_chat_selector_reports_unavailable_when_only_qwen_is_enabled() {
        let err = "chat:google/gemma4_e2b"
            .parse::<ModelSelector>()
            .expect_err("disabled gemma selector should be unavailable");

        assert!(matches!(
            err,
            ModelsError::ModelUnavailable { selector }
            if selector == "chat:google/gemma4_e2b"
        ));
    }

    #[test]
    fn tts_models_round_trip_string_selectors() {
        #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
        {
            let model: TtsModels = "piper/en_us_ljspeech_medium"
                .parse()
                .expect("known TTS selector should parse");

            assert_eq!(model, TtsModels::PiperEnUsLjspeechMedium);
            assert_eq!(model.to_string(), "piper/en_us_ljspeech_medium");
        }
    }

    #[test]
    fn model_selector_parses_tts_prefix() {
        #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
        {
            let selector: ModelSelector = "tts:piper/en_us_ljspeech_medium"
                .parse()
                .expect("known TTS selector should parse");

            assert_eq!(
                selector,
                ModelSelector::Tts(TtsModels::PiperEnUsLjspeechMedium)
            );
            assert_eq!(selector.to_string(), "tts:piper/en_us_ljspeech_medium");
        }
    }

    #[cfg(not(feature = "model-piper-en-us-ljspeech-medium"))]
    #[test]
    fn tts_selector_reports_unavailable_for_disabled_bundles() {
        let err = "tts:piper/en_us_ljspeech_medium"
            .parse::<ModelSelector>()
            .expect_err("disabled known TTS selector should be unavailable");

        assert!(matches!(
            err,
            ModelsError::ModelUnavailable { selector }
            if selector == "tts:piper/en_us_ljspeech_medium"
        ));
    }

    #[test]
    fn artifact_rules_match_expected_files() {
        let artifacts = BundleArtifacts {
            control_name: "embeddinggemma_300m",
            format: CheckpointFormat::Safetensors,
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
