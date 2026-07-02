//! Curated model bundle catalog for the Motlie ecosystem.
//!
//! This crate owns the bundle/catalog layer above `motlie-model`.

use std::collections::BTreeMap;
#[cfg(any(
    feature = "model-qwen3-4b",
    feature = "model-qwen3-4b-gguf",
    feature = "model-qwen3-6-27b-gguf",
    feature = "model-ornith-1-0-35b-gguf",
    feature = "model-google-gemma-300m",
    feature = "model-qwen3-embedding-06b",
    feature = "model-gemma4-e2b",
    feature = "model-gemma4-e2b-gguf",
    feature = "model-gemma4-e4b",
    feature = "model-gemma4-12b-gguf",
    feature = "model-gemma4-12b-qat-gguf",
    feature = "model-gemma4-e4b-gguf",
    feature = "model-piper-en-us-ljspeech-medium",
    feature = "model-kokoro-82m",
    feature = "model-qwen3-tts-cpp",
    feature = "model-moonshine-streaming",
    feature = "model-sherpa-onnx-streaming",
    feature = "model-whisper-base-en",
))]
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

pub mod artifact_manifest;
pub mod asr;
pub mod chat;
pub mod embeddings;
pub mod tool_registry;
pub mod tts;

use hf_hub::api::sync::ApiBuilder;
use thiserror::Error;

#[cfg(feature = "model-kokoro-82m")]
use motlie_model_kokoro::KokoroHandle;
#[cfg(any(
    feature = "model-qwen3-4b-gguf",
    feature = "model-qwen3-6-27b-gguf",
    feature = "model-ornith-1-0-35b-gguf",
    feature = "model-gemma4-e2b-gguf",
    feature = "model-gemma4-e4b-gguf",
    feature = "model-gemma4-12b-gguf",
    feature = "model-gemma4-12b-qat-gguf",
))]
use motlie_model_llama_cpp::LlamaCppTextHandle;
#[cfg(any(
    feature = "model-google-gemma-300m",
    feature = "model-qwen3-embedding-06b"
))]
use motlie_model_mistral::MistralEmbeddingHandle;
#[cfg(any(feature = "model-gemma4-e2b", feature = "model-gemma4-e4b",))]
use motlie_model_mistral::MistralMultimodalHandle;
#[cfg(feature = "model-qwen3-4b")]
use motlie_model_mistral::MistralTextHandle;
#[cfg(feature = "model-moonshine-streaming")]
use motlie_model_moonshine::MoonshineHandle;
#[cfg(feature = "model-piper-en-us-ljspeech-medium")]
use motlie_model_piper::PiperHandle;
#[cfg(feature = "model-qwen3-tts-cpp")]
use motlie_model_qwen3_tts_cpp::Qwen3TtsCppHandle;
#[cfg(feature = "model-sherpa-onnx-streaming")]
use motlie_model_sherpa_onnx::SherpaOnnxHandle;
#[cfg(feature = "model-whisper-base-en")]
use motlie_model_whisper_cpp::WhisperCppHandle;

pub use asr::AsrModels;
pub use chat::ChatModels;
pub use embeddings::EmbeddingModels;
pub use motlie_model::{
    ArtifactRule, ArtifactSource, BackendKind, BuildConstraint, BundleFamily, BundleId,
    BundleRequirements, Capabilities, CapabilityDescriptor, CapabilityKind, CheckpointFormat,
    ContentKind, EvalTrack, InteractionStyle, ModelBundle, ModelCheckpoint, ModelIdentity,
    PlatformConstraint, QuantizationScheme, QuantizationSupport,
};
use motlie_model::{
    BundleHandle, ChatModel, CompletionModel, EmbeddingModel, LoadedBundleDescriptor, ModelError,
    ModelMetricSnapshot, RuntimeAcceleratorObservation, StartOptions, UnsupportedChat,
    UnsupportedCompletion, UnsupportedEmbeddings,
};
pub use tool_registry::{Mcp, McpError, McpTransport, ToolDispatch, ToolList, ToolListError};
pub use tts::TtsModels;

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
    #[error("failed to create Hugging Face API client: {message}")]
    HuggingFaceClient { message: String },
    #[error("failed to inspect model repo `{repo}`: {message}")]
    InspectModelRepo { repo: &'static str, message: String },
    #[error("failed to download `{filename}` from repo `{repo}`: {message}")]
    DownloadArtifact {
        repo: &'static str,
        filename: String,
        message: String,
    },
    #[error("failed to prepare artifacts for bundle `{bundle_id}`: {message}")]
    ArtifactPreparation {
        bundle_id: BundleId,
        message: String,
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
    #[error("invalid quantization scheme: {message}")]
    InvalidQuantizationScheme { message: String },
}

pub type Result<T> = std::result::Result<T, ModelsError>;

pub const LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX: &str = "artifact policy `LocalOnly`";

pub fn quantization_label_isq(quantization: Option<QuantizationScheme>) -> &'static str {
    match quantization {
        Some(QuantizationScheme::IsqQ4) => "ISQ Q4",
        Some(QuantizationScheme::IsqQ8) => "ISQ Q8",
        Some(QuantizationScheme::Bf16) => "BF16",
        Some(QuantizationScheme::Fp32) => "F32",
        Some(QuantizationScheme::Fp16) => "F16",
        Some(_) => "unsupported ISQ precision",
        None => "default",
    }
}

pub fn quantization_label_gguf(quantization: Option<QuantizationScheme>) -> &'static str {
    match quantization {
        Some(QuantizationScheme::GgufQ4_K_M) => "GGUF Q4_K_M",
        Some(QuantizationScheme::GgufQ4_0) => "GGUF Q4_0",
        Some(QuantizationScheme::GgufQ5_K_M) => "GGUF Q5_K_M",
        Some(QuantizationScheme::GgufQ8_0) => "GGUF Q8_0",
        Some(QuantizationScheme::Fp16) => "GGUF F16",
        Some(_) => "unsupported GGUF precision",
        None => "GGUF F16 (no quantization)",
    }
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
            "{LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX} requires cached `config.json` for `{model_id}` under `{}`",
            cache_root.display()
        ))
    })?;

    if repo.get("tokenizer.json").is_none() && repo.get("tokenizer.model").is_none() {
        return Err(motlie_model::ModelError::InvalidConfiguration(format!(
            "{LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX} requires cached tokenizer files for `{model_id}` under `{}`",
            cache_root.display()
        )));
    }

    let snapshot_dir = config.parent().ok_or_else(|| {
        motlie_model::ModelError::InvalidConfiguration(format!(
            "{LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX} found invalid cache layout for `{model_id}` under `{}`",
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
            "{LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX} requires cached weight files for `{model_id}` under `{}`",
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
            "{LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX} requires cached GGUF artifacts for `{model_id}` under `{}`; \
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
            "{LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX} requires cached .gguf files for `{model_id}` under `{}`",
            cache_root.display()
        )));
    }

    Ok(snapshot_dir)
}

/// Resolve a Hugging Face GGUF cache snapshot and require at least one exact
/// root-level GGUF filename for the selected curated variant.
#[cfg(any(
    feature = "model-gemma4-12b-gguf",
    feature = "model-gemma4-12b-qat-gguf",
    feature = "model-ornith-1-0-35b-gguf",
))]
pub(crate) fn resolve_hf_gguf_snapshot_with_any_file(
    model_id: &str,
    cache_root: &Path,
    accepted_filenames: &[&str],
) -> std::result::Result<PathBuf, motlie_model::ModelError> {
    let snapshot_dir = resolve_hf_gguf_snapshot(model_id, cache_root)?;

    if accepted_filenames.is_empty()
        || accepted_filenames
            .iter()
            .any(|filename| snapshot_dir.join(filename).is_file())
    {
        return Ok(snapshot_dir);
    }

    Err(motlie_model::ModelError::InvalidConfiguration(format!(
        "{LOCAL_ONLY_ARTIFACT_POLICY_ERROR_PREFIX} requires one of [{}] for `{model_id}` under `{}`",
        accepted_filenames.join(", "),
        snapshot_dir.display()
    )))
}

#[cfg(any(
    feature = "model-qwen3-4b",
    feature = "model-qwen3-4b-gguf",
    feature = "model-qwen3-6-27b-gguf",
    feature = "model-ornith-1-0-35b-gguf",
    feature = "model-gemma4-e2b",
    feature = "model-gemma4-e2b-gguf",
    feature = "model-gemma4-e4b",
    feature = "model-gemma4-12b-gguf",
    feature = "model-gemma4-12b-qat-gguf",
    feature = "model-gemma4-e4b-gguf",
    feature = "model-google-gemma-300m",
    feature = "model-qwen3-embedding-06b",
    feature = "model-whisper-base-en",
    feature = "model-sherpa-onnx-streaming",
    feature = "model-moonshine-streaming",
    feature = "model-piper-en-us-ljspeech-medium",
    feature = "model-kokoro-82m",
    feature = "model-qwen3-tts-cpp",
))]
pub(crate) fn resolve_typed_artifact_policy(
    options: StartOptions,
    resolver: impl FnOnce(&Path) -> std::result::Result<PathBuf, motlie_model::ModelError>,
) -> std::result::Result<StartOptions, motlie_model::ModelError> {
    let StartOptions {
        artifact_policy,
        quantization_scheme,
        unpack_root,
        max_concurrency,
    } = options;

    let artifact_policy = match artifact_policy {
        Some(motlie_model::ArtifactPolicy::LocalOnly { root }) => {
            Some(motlie_model::ArtifactPolicy::LocalOnly {
                root: resolver(&root)?,
            })
        }
        other => other,
    };

    Ok(StartOptions {
        artifact_policy,
        quantization_scheme,
        unpack_root,
        max_concurrency,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
#[allow(non_camel_case_types)]
pub enum CuratedBundle {
    #[cfg(feature = "model-qwen3-4b")]
    Qwen3_4B,
    #[cfg(feature = "model-gemma4-e2b")]
    Gemma4E2B,
    #[cfg(feature = "model-gemma4-e4b")]
    Gemma4E4B,
    #[cfg(feature = "model-qwen3-4b-gguf")]
    Qwen3_4B_Gguf,
    #[cfg(feature = "model-qwen3-6-27b-gguf")]
    Qwen3_6_27B_Gguf,
    #[cfg(feature = "model-ornith-1-0-35b-gguf")]
    Ornith_1_0_35B_Gguf,
    #[cfg(feature = "model-gemma4-e2b-gguf")]
    Gemma4E2B_Gguf,
    #[cfg(feature = "model-gemma4-e4b-gguf")]
    Gemma4E4B_Gguf,
    #[cfg(feature = "model-gemma4-12b-gguf")]
    Gemma4_12B_Gguf,
    #[cfg(feature = "model-gemma4-12b-qat-gguf")]
    Gemma4_12B_Qat_Gguf,
    #[cfg(feature = "model-google-gemma-300m")]
    EmbeddingGemma300m,
    #[cfg(feature = "model-qwen3-embedding-06b")]
    Qwen3Embedding06B,
    #[cfg(feature = "model-whisper-base-en")]
    WhisperBaseEn,
    #[cfg(feature = "model-sherpa-onnx-streaming")]
    SherpaOnnxStreamingZipformerEn,
    #[cfg(feature = "model-sherpa-onnx-streaming")]
    SherpaOnnxStreamingZipformerEnKroko2025,
    #[cfg(feature = "model-moonshine-streaming")]
    MoonshineStreamingEn,
    #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
    PiperEnUsLjspeechMedium,
    #[cfg(feature = "model-kokoro-82m")]
    Kokoro82m,
    #[cfg(feature = "model-qwen3-tts-cpp")]
    Qwen3TtsCpp0_6B,
}

impl CuratedBundle {
    /// The single canonical id string for every curated bundle, used identically
    /// across the eval snapshot, the driver artifact map, and result records.
    /// This list is always present (not feature-gated) so snapshot/data lints can
    /// enumerate the full set without compiling every model backend. The
    /// `canonical_ids_match_bundle_ids` test (run with all model features) keeps
    /// this list exactly in sync with the enum's `bundle_id()`/`canonical_id()`.
    pub const CANONICAL_IDS: &'static [&'static str] = &[
        "qwen3_4b",
        "gemma4_e2b",
        "gemma4_e4b",
        "qwen3_4b_gguf",
        "qwen3_6_27b_gguf",
        "ornith_1_0_35b_gguf",
        "gemma4_e2b_gguf",
        "gemma4_e4b_gguf",
        "gemma4_12b_gguf",
        "gemma4_12b_qat_gguf",
        "embeddinggemma_300m",
        "qwen3_embedding_06b",
        "whisper_base_en",
        "sherpa_onnx_streaming_zipformer_en",
        "sherpa_onnx_streaming_zipformer_en_kroko_2025",
        "moonshine_streaming_en",
        "piper_en_us_ljspeech_medium",
        "kokoro_82m",
        "qwen3_tts_cpp_0_6b",
    ];

    pub fn bundle_id(&self) -> BundleId {
        self.descriptor().id
    }

    /// The canonical id string for this variant. Equals `bundle_id().as_str()`
    /// (enforced by the `canonical_ids_match_bundle_ids` test) and is always a
    /// member of [`CuratedBundle::CANONICAL_IDS`].
    pub fn canonical_id(&self) -> &'static str {
        #[allow(unreachable_patterns)]
        match self {
            #[cfg(feature = "model-qwen3-4b")]
            Self::Qwen3_4B => "qwen3_4b",
            #[cfg(feature = "model-gemma4-e2b")]
            Self::Gemma4E2B => "gemma4_e2b",
            #[cfg(feature = "model-gemma4-e4b")]
            Self::Gemma4E4B => "gemma4_e4b",
            #[cfg(feature = "model-qwen3-4b-gguf")]
            Self::Qwen3_4B_Gguf => "qwen3_4b_gguf",
            #[cfg(feature = "model-qwen3-6-27b-gguf")]
            Self::Qwen3_6_27B_Gguf => "qwen3_6_27b_gguf",
            #[cfg(feature = "model-ornith-1-0-35b-gguf")]
            Self::Ornith_1_0_35B_Gguf => "ornith_1_0_35b_gguf",
            #[cfg(feature = "model-gemma4-e2b-gguf")]
            Self::Gemma4E2B_Gguf => "gemma4_e2b_gguf",
            #[cfg(feature = "model-gemma4-e4b-gguf")]
            Self::Gemma4E4B_Gguf => "gemma4_e4b_gguf",
            #[cfg(feature = "model-gemma4-12b-gguf")]
            Self::Gemma4_12B_Gguf => "gemma4_12b_gguf",
            #[cfg(feature = "model-gemma4-12b-qat-gguf")]
            Self::Gemma4_12B_Qat_Gguf => "gemma4_12b_qat_gguf",
            #[cfg(feature = "model-google-gemma-300m")]
            Self::EmbeddingGemma300m => "embeddinggemma_300m",
            #[cfg(feature = "model-qwen3-embedding-06b")]
            Self::Qwen3Embedding06B => "qwen3_embedding_06b",
            #[cfg(feature = "model-whisper-base-en")]
            Self::WhisperBaseEn => "whisper_base_en",
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            Self::SherpaOnnxStreamingZipformerEn => "sherpa_onnx_streaming_zipformer_en",
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            Self::SherpaOnnxStreamingZipformerEnKroko2025 => {
                "sherpa_onnx_streaming_zipformer_en_kroko_2025"
            }
            #[cfg(feature = "model-moonshine-streaming")]
            Self::MoonshineStreamingEn => "moonshine_streaming_en",
            #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
            Self::PiperEnUsLjspeechMedium => "piper_en_us_ljspeech_medium",
            #[cfg(feature = "model-kokoro-82m")]
            Self::Kokoro82m => "kokoro_82m",
            #[cfg(feature = "model-qwen3-tts-cpp")]
            Self::Qwen3TtsCpp0_6B => "qwen3_tts_cpp_0_6b",
            _ => unreachable!("no curated bundle variants are enabled"),
        }
    }

    pub fn descriptor(&self) -> BundleDescriptor {
        #[allow(unreachable_patterns)]
        match self {
            #[cfg(feature = "model-qwen3-4b")]
            Self::Qwen3_4B => chat::qwen3_4b::descriptor(),
            #[cfg(feature = "model-gemma4-e2b")]
            Self::Gemma4E2B => chat::gemma4_e2b::descriptor(),
            #[cfg(feature = "model-gemma4-e4b")]
            Self::Gemma4E4B => chat::gemma4_e4b::descriptor(),
            #[cfg(feature = "model-qwen3-4b-gguf")]
            Self::Qwen3_4B_Gguf => chat::qwen3_4b_gguf::descriptor(),
            #[cfg(feature = "model-qwen3-6-27b-gguf")]
            Self::Qwen3_6_27B_Gguf => chat::qwen3_6_27b_gguf::descriptor(),
            #[cfg(feature = "model-ornith-1-0-35b-gguf")]
            Self::Ornith_1_0_35B_Gguf => chat::ornith_1_0_35b_gguf::descriptor(),
            #[cfg(feature = "model-gemma4-e2b-gguf")]
            Self::Gemma4E2B_Gguf => chat::gemma4_e2b_gguf::descriptor(),
            #[cfg(feature = "model-gemma4-e4b-gguf")]
            Self::Gemma4E4B_Gguf => chat::gemma4_e4b_gguf::descriptor(),
            #[cfg(feature = "model-gemma4-12b-gguf")]
            Self::Gemma4_12B_Gguf => chat::gemma4_12b_gguf::descriptor(),
            #[cfg(feature = "model-gemma4-12b-qat-gguf")]
            Self::Gemma4_12B_Qat_Gguf => chat::gemma4_12b_qat_gguf::descriptor(),
            #[cfg(feature = "model-google-gemma-300m")]
            Self::EmbeddingGemma300m => embeddings::embeddinggemma_300m::descriptor(),
            #[cfg(feature = "model-qwen3-embedding-06b")]
            Self::Qwen3Embedding06B => embeddings::qwen3_embedding_06b::descriptor(),
            #[cfg(feature = "model-whisper-base-en")]
            Self::WhisperBaseEn => asr::whisper_base_en::descriptor(),
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            Self::SherpaOnnxStreamingZipformerEn => {
                asr::sherpa_onnx_streaming_zipformer_en::descriptor()
            }
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            Self::SherpaOnnxStreamingZipformerEnKroko2025 => {
                asr::sherpa_onnx_streaming_zipformer_en_kroko_2025::descriptor()
            }
            #[cfg(feature = "model-moonshine-streaming")]
            Self::MoonshineStreamingEn => asr::moonshine_streaming_en::descriptor(),
            #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
            Self::PiperEnUsLjspeechMedium => tts::piper_en_us_ljspeech_medium::descriptor(),
            #[cfg(feature = "model-kokoro-82m")]
            Self::Kokoro82m => tts::kokoro_82m::descriptor(),
            #[cfg(feature = "model-qwen3-tts-cpp")]
            Self::Qwen3TtsCpp0_6B => tts::qwen3_tts_cpp_0_6b::descriptor(),
            _ => unreachable!("no curated bundle variants are enabled"),
        }
    }

    pub async fn start(
        &self,
        options: StartOptions,
    ) -> std::result::Result<CuratedHandle, ModelError> {
        let _ = &options;

        #[allow(unreachable_patterns)]
        match self {
            #[cfg(feature = "model-qwen3-4b")]
            Self::Qwen3_4B => chat::qwen3_4b::start(options)
                .await
                .map(CuratedHandle::Qwen3_4B),
            #[cfg(feature = "model-gemma4-e2b")]
            Self::Gemma4E2B => chat::gemma4_e2b::start(options)
                .await
                .map(CuratedHandle::Gemma4E2B),
            #[cfg(feature = "model-gemma4-e4b")]
            Self::Gemma4E4B => chat::gemma4_e4b::start(options)
                .await
                .map(CuratedHandle::Gemma4E4B),
            #[cfg(feature = "model-qwen3-4b-gguf")]
            Self::Qwen3_4B_Gguf => chat::qwen3_4b_gguf::start(options)
                .await
                .map(CuratedHandle::Qwen3_4B_Gguf),
            #[cfg(feature = "model-qwen3-6-27b-gguf")]
            Self::Qwen3_6_27B_Gguf => chat::qwen3_6_27b_gguf::start(options)
                .await
                .map(CuratedHandle::Qwen3_6_27B_Gguf),
            #[cfg(feature = "model-ornith-1-0-35b-gguf")]
            Self::Ornith_1_0_35B_Gguf => chat::ornith_1_0_35b_gguf::start(options)
                .await
                .map(CuratedHandle::Ornith_1_0_35B_Gguf),
            #[cfg(feature = "model-gemma4-e2b-gguf")]
            Self::Gemma4E2B_Gguf => chat::gemma4_e2b_gguf::start(options)
                .await
                .map(CuratedHandle::Gemma4E2B_Gguf),
            #[cfg(feature = "model-gemma4-e4b-gguf")]
            Self::Gemma4E4B_Gguf => chat::gemma4_e4b_gguf::start(options)
                .await
                .map(CuratedHandle::Gemma4E4B_Gguf),
            #[cfg(feature = "model-gemma4-12b-gguf")]
            Self::Gemma4_12B_Gguf => chat::gemma4_12b_gguf::start(options)
                .await
                .map(CuratedHandle::Gemma4_12B_Gguf),
            #[cfg(feature = "model-gemma4-12b-qat-gguf")]
            Self::Gemma4_12B_Qat_Gguf => chat::gemma4_12b_qat_gguf::start(options)
                .await
                .map(CuratedHandle::Gemma4_12B_Qat_Gguf),
            #[cfg(feature = "model-google-gemma-300m")]
            Self::EmbeddingGemma300m => embeddings::embeddinggemma_300m::start(options)
                .await
                .map(CuratedHandle::EmbeddingGemma300m),
            #[cfg(feature = "model-qwen3-embedding-06b")]
            Self::Qwen3Embedding06B => embeddings::qwen3_embedding_06b::start(options)
                .await
                .map(CuratedHandle::Qwen3Embedding06B),
            #[cfg(feature = "model-whisper-base-en")]
            Self::WhisperBaseEn => asr::whisper_base_en::start_typed(options)
                .await
                .map(CuratedHandle::WhisperBaseEn),
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            Self::SherpaOnnxStreamingZipformerEn => {
                asr::sherpa_onnx_streaming_zipformer_en::start_typed(options)
                    .await
                    .map(CuratedHandle::SherpaOnnxStreamingZipformerEn)
            }
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            Self::SherpaOnnxStreamingZipformerEnKroko2025 => {
                asr::sherpa_onnx_streaming_zipformer_en_kroko_2025::start_typed(options)
                    .await
                    .map(CuratedHandle::SherpaOnnxStreamingZipformerEnKroko2025)
            }
            #[cfg(feature = "model-moonshine-streaming")]
            Self::MoonshineStreamingEn => asr::moonshine_streaming_en::start_typed(options)
                .await
                .map(CuratedHandle::MoonshineStreamingEn),
            #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
            Self::PiperEnUsLjspeechMedium => tts::piper_en_us_ljspeech_medium::start_typed(options)
                .await
                .map(CuratedHandle::PiperEnUsLjspeechMedium),
            #[cfg(feature = "model-kokoro-82m")]
            Self::Kokoro82m => tts::kokoro_82m::start_typed(options)
                .await
                .map(CuratedHandle::Kokoro82m),
            #[cfg(feature = "model-qwen3-tts-cpp")]
            Self::Qwen3TtsCpp0_6B => tts::qwen3_tts_cpp_0_6b::start_typed(options)
                .await
                .map(CuratedHandle::Qwen3TtsCpp0_6B),
            _ => Err(ModelError::InvalidConfiguration(
                "no curated bundle variants are enabled".into(),
            )),
        }
    }
}

#[allow(non_camel_case_types)]
pub enum CuratedHandle {
    #[cfg(feature = "model-qwen3-4b")]
    Qwen3_4B(MistralTextHandle),
    #[cfg(feature = "model-gemma4-e2b")]
    Gemma4E2B(MistralMultimodalHandle),
    #[cfg(feature = "model-gemma4-e4b")]
    Gemma4E4B(MistralMultimodalHandle),
    #[cfg(feature = "model-qwen3-4b-gguf")]
    Qwen3_4B_Gguf(LlamaCppTextHandle),
    #[cfg(feature = "model-qwen3-6-27b-gguf")]
    Qwen3_6_27B_Gguf(LlamaCppTextHandle),
    #[cfg(feature = "model-ornith-1-0-35b-gguf")]
    Ornith_1_0_35B_Gguf(LlamaCppTextHandle),
    #[cfg(feature = "model-gemma4-e2b-gguf")]
    Gemma4E2B_Gguf(LlamaCppTextHandle),
    #[cfg(feature = "model-gemma4-e4b-gguf")]
    Gemma4E4B_Gguf(LlamaCppTextHandle),
    #[cfg(feature = "model-gemma4-12b-gguf")]
    Gemma4_12B_Gguf(LlamaCppTextHandle),
    #[cfg(feature = "model-gemma4-12b-qat-gguf")]
    Gemma4_12B_Qat_Gguf(LlamaCppTextHandle),
    #[cfg(feature = "model-google-gemma-300m")]
    EmbeddingGemma300m(MistralEmbeddingHandle),
    #[cfg(feature = "model-qwen3-embedding-06b")]
    Qwen3Embedding06B(MistralEmbeddingHandle),
    #[cfg(feature = "model-whisper-base-en")]
    WhisperBaseEn(WhisperCppHandle),
    #[cfg(feature = "model-sherpa-onnx-streaming")]
    SherpaOnnxStreamingZipformerEn(SherpaOnnxHandle),
    #[cfg(feature = "model-sherpa-onnx-streaming")]
    SherpaOnnxStreamingZipformerEnKroko2025(SherpaOnnxHandle),
    #[cfg(feature = "model-moonshine-streaming")]
    MoonshineStreamingEn(MoonshineHandle),
    #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
    PiperEnUsLjspeechMedium(PiperHandle),
    #[cfg(feature = "model-kokoro-82m")]
    Kokoro82m(KokoroHandle),
    #[cfg(feature = "model-qwen3-tts-cpp")]
    Qwen3TtsCpp0_6B(Qwen3TtsCppHandle),
}

#[async_trait::async_trait]
impl BundleHandle for CuratedHandle {
    type Chat = Self;
    type Completion = Self;
    type Embeddings = Self;

    fn descriptor(&self) -> &LoadedBundleDescriptor {
        #[allow(unreachable_patterns)]
        match self {
            #[cfg(feature = "model-qwen3-4b")]
            Self::Qwen3_4B(handle) => handle.descriptor(),
            #[cfg(feature = "model-gemma4-e2b")]
            Self::Gemma4E2B(handle) => handle.descriptor(),
            #[cfg(feature = "model-gemma4-e4b")]
            Self::Gemma4E4B(handle) => handle.descriptor(),
            #[cfg(feature = "model-qwen3-4b-gguf")]
            Self::Qwen3_4B_Gguf(handle) => handle.descriptor(),
            #[cfg(feature = "model-qwen3-6-27b-gguf")]
            Self::Qwen3_6_27B_Gguf(handle) => handle.descriptor(),
            #[cfg(feature = "model-ornith-1-0-35b-gguf")]
            Self::Ornith_1_0_35B_Gguf(handle) => handle.descriptor(),
            #[cfg(feature = "model-gemma4-e2b-gguf")]
            Self::Gemma4E2B_Gguf(handle) => handle.descriptor(),
            #[cfg(feature = "model-gemma4-e4b-gguf")]
            Self::Gemma4E4B_Gguf(handle) => handle.descriptor(),
            #[cfg(feature = "model-gemma4-12b-gguf")]
            Self::Gemma4_12B_Gguf(handle) => handle.descriptor(),
            #[cfg(feature = "model-gemma4-12b-qat-gguf")]
            Self::Gemma4_12B_Qat_Gguf(handle) => handle.descriptor(),
            #[cfg(feature = "model-google-gemma-300m")]
            Self::EmbeddingGemma300m(handle) => handle.descriptor(),
            #[cfg(feature = "model-qwen3-embedding-06b")]
            Self::Qwen3Embedding06B(handle) => handle.descriptor(),
            #[cfg(feature = "model-whisper-base-en")]
            Self::WhisperBaseEn(handle) => handle.descriptor(),
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            Self::SherpaOnnxStreamingZipformerEn(handle) => handle.descriptor(),
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            Self::SherpaOnnxStreamingZipformerEnKroko2025(handle) => handle.descriptor(),
            #[cfg(feature = "model-moonshine-streaming")]
            Self::MoonshineStreamingEn(handle) => handle.descriptor(),
            #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
            Self::PiperEnUsLjspeechMedium(handle) => handle.descriptor(),
            #[cfg(feature = "model-kokoro-82m")]
            Self::Kokoro82m(handle) => handle.descriptor(),
            #[cfg(feature = "model-qwen3-tts-cpp")]
            Self::Qwen3TtsCpp0_6B(handle) => handle.descriptor(),
            _ => unreachable!("no curated handle variants are enabled"),
        }
    }

    fn capabilities(&self) -> &Capabilities {
        #[allow(unreachable_patterns)]
        match self {
            #[cfg(feature = "model-qwen3-4b")]
            Self::Qwen3_4B(handle) => handle.capabilities(),
            #[cfg(feature = "model-gemma4-e2b")]
            Self::Gemma4E2B(handle) => handle.capabilities(),
            #[cfg(feature = "model-gemma4-e4b")]
            Self::Gemma4E4B(handle) => handle.capabilities(),
            #[cfg(feature = "model-qwen3-4b-gguf")]
            Self::Qwen3_4B_Gguf(handle) => handle.capabilities(),
            #[cfg(feature = "model-qwen3-6-27b-gguf")]
            Self::Qwen3_6_27B_Gguf(handle) => handle.capabilities(),
            #[cfg(feature = "model-ornith-1-0-35b-gguf")]
            Self::Ornith_1_0_35B_Gguf(handle) => handle.capabilities(),
            #[cfg(feature = "model-gemma4-e2b-gguf")]
            Self::Gemma4E2B_Gguf(handle) => handle.capabilities(),
            #[cfg(feature = "model-gemma4-e4b-gguf")]
            Self::Gemma4E4B_Gguf(handle) => handle.capabilities(),
            #[cfg(feature = "model-gemma4-12b-gguf")]
            Self::Gemma4_12B_Gguf(handle) => handle.capabilities(),
            #[cfg(feature = "model-gemma4-12b-qat-gguf")]
            Self::Gemma4_12B_Qat_Gguf(handle) => handle.capabilities(),
            #[cfg(feature = "model-google-gemma-300m")]
            Self::EmbeddingGemma300m(handle) => handle.capabilities(),
            #[cfg(feature = "model-qwen3-embedding-06b")]
            Self::Qwen3Embedding06B(handle) => handle.capabilities(),
            #[cfg(feature = "model-whisper-base-en")]
            Self::WhisperBaseEn(handle) => handle.capabilities(),
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            Self::SherpaOnnxStreamingZipformerEn(handle) => handle.capabilities(),
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            Self::SherpaOnnxStreamingZipformerEnKroko2025(handle) => handle.capabilities(),
            #[cfg(feature = "model-moonshine-streaming")]
            Self::MoonshineStreamingEn(handle) => handle.capabilities(),
            #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
            Self::PiperEnUsLjspeechMedium(handle) => handle.capabilities(),
            #[cfg(feature = "model-kokoro-82m")]
            Self::Kokoro82m(handle) => handle.capabilities(),
            #[cfg(feature = "model-qwen3-tts-cpp")]
            Self::Qwen3TtsCpp0_6B(handle) => handle.capabilities(),
            _ => unreachable!("no curated handle variants are enabled"),
        }
    }

    fn metric_snapshot(&self) -> Option<ModelMetricSnapshot> {
        #[allow(unreachable_patterns)]
        match self {
            #[cfg(feature = "model-qwen3-4b")]
            Self::Qwen3_4B(handle) => handle.metric_snapshot(),
            #[cfg(feature = "model-gemma4-e2b")]
            Self::Gemma4E2B(handle) => handle.metric_snapshot(),
            #[cfg(feature = "model-gemma4-e4b")]
            Self::Gemma4E4B(handle) => handle.metric_snapshot(),
            #[cfg(feature = "model-qwen3-4b-gguf")]
            Self::Qwen3_4B_Gguf(handle) => handle.metric_snapshot(),
            #[cfg(feature = "model-qwen3-6-27b-gguf")]
            Self::Qwen3_6_27B_Gguf(handle) => handle.metric_snapshot(),
            #[cfg(feature = "model-ornith-1-0-35b-gguf")]
            Self::Ornith_1_0_35B_Gguf(handle) => handle.metric_snapshot(),
            #[cfg(feature = "model-gemma4-e2b-gguf")]
            Self::Gemma4E2B_Gguf(handle) => handle.metric_snapshot(),
            #[cfg(feature = "model-gemma4-e4b-gguf")]
            Self::Gemma4E4B_Gguf(handle) => handle.metric_snapshot(),
            #[cfg(feature = "model-gemma4-12b-gguf")]
            Self::Gemma4_12B_Gguf(handle) => handle.metric_snapshot(),
            #[cfg(feature = "model-gemma4-12b-qat-gguf")]
            Self::Gemma4_12B_Qat_Gguf(handle) => handle.metric_snapshot(),
            #[cfg(feature = "model-google-gemma-300m")]
            Self::EmbeddingGemma300m(handle) => handle.metric_snapshot(),
            #[cfg(feature = "model-qwen3-embedding-06b")]
            Self::Qwen3Embedding06B(handle) => handle.metric_snapshot(),
            #[cfg(feature = "model-whisper-base-en")]
            Self::WhisperBaseEn(handle) => handle.metric_snapshot(),
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            Self::SherpaOnnxStreamingZipformerEn(handle) => handle.metric_snapshot(),
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            Self::SherpaOnnxStreamingZipformerEnKroko2025(handle) => handle.metric_snapshot(),
            #[cfg(feature = "model-moonshine-streaming")]
            Self::MoonshineStreamingEn(handle) => handle.metric_snapshot(),
            #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
            Self::PiperEnUsLjspeechMedium(handle) => handle.metric_snapshot(),
            #[cfg(feature = "model-kokoro-82m")]
            Self::Kokoro82m(handle) => handle.metric_snapshot(),
            #[cfg(feature = "model-qwen3-tts-cpp")]
            Self::Qwen3TtsCpp0_6B(handle) => handle.metric_snapshot(),
            _ => unreachable!("no curated handle variants are enabled"),
        }
    }

    fn accelerator_observation(&self) -> Option<RuntimeAcceleratorObservation> {
        #[allow(unreachable_patterns)]
        match self {
            #[cfg(feature = "model-qwen3-4b")]
            Self::Qwen3_4B(handle) => handle.accelerator_observation(),
            #[cfg(feature = "model-gemma4-e2b")]
            Self::Gemma4E2B(handle) => handle.accelerator_observation(),
            #[cfg(feature = "model-gemma4-e4b")]
            Self::Gemma4E4B(handle) => handle.accelerator_observation(),
            #[cfg(feature = "model-qwen3-4b-gguf")]
            Self::Qwen3_4B_Gguf(handle) => handle.accelerator_observation(),
            #[cfg(feature = "model-qwen3-6-27b-gguf")]
            Self::Qwen3_6_27B_Gguf(handle) => handle.accelerator_observation(),
            #[cfg(feature = "model-ornith-1-0-35b-gguf")]
            Self::Ornith_1_0_35B_Gguf(handle) => handle.accelerator_observation(),
            #[cfg(feature = "model-gemma4-e2b-gguf")]
            Self::Gemma4E2B_Gguf(handle) => handle.accelerator_observation(),
            #[cfg(feature = "model-gemma4-e4b-gguf")]
            Self::Gemma4E4B_Gguf(handle) => handle.accelerator_observation(),
            #[cfg(feature = "model-gemma4-12b-gguf")]
            Self::Gemma4_12B_Gguf(handle) => handle.accelerator_observation(),
            #[cfg(feature = "model-gemma4-12b-qat-gguf")]
            Self::Gemma4_12B_Qat_Gguf(handle) => handle.accelerator_observation(),
            #[cfg(feature = "model-google-gemma-300m")]
            Self::EmbeddingGemma300m(handle) => handle.accelerator_observation(),
            #[cfg(feature = "model-qwen3-embedding-06b")]
            Self::Qwen3Embedding06B(handle) => handle.accelerator_observation(),
            #[cfg(feature = "model-whisper-base-en")]
            Self::WhisperBaseEn(handle) => handle.accelerator_observation(),
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            Self::SherpaOnnxStreamingZipformerEn(handle) => handle.accelerator_observation(),
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            Self::SherpaOnnxStreamingZipformerEnKroko2025(handle) => {
                handle.accelerator_observation()
            }
            #[cfg(feature = "model-moonshine-streaming")]
            Self::MoonshineStreamingEn(handle) => handle.accelerator_observation(),
            #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
            Self::PiperEnUsLjspeechMedium(handle) => handle.accelerator_observation(),
            #[cfg(feature = "model-kokoro-82m")]
            Self::Kokoro82m(handle) => handle.accelerator_observation(),
            #[cfg(feature = "model-qwen3-tts-cpp")]
            Self::Qwen3TtsCpp0_6B(handle) => handle.accelerator_observation(),
            _ => unreachable!("no curated handle variants are enabled"),
        }
    }

    fn chat(&self) -> std::result::Result<&Self::Chat, ModelError> {
        #[allow(unreachable_patterns)]
        match self {
            #[cfg(feature = "model-qwen3-4b")]
            Self::Qwen3_4B(_) => Ok(self),
            #[cfg(feature = "model-gemma4-e2b")]
            Self::Gemma4E2B(_) => Ok(self),
            #[cfg(feature = "model-gemma4-e4b")]
            Self::Gemma4E4B(_) => Ok(self),
            #[cfg(feature = "model-qwen3-4b-gguf")]
            Self::Qwen3_4B_Gguf(_) => Ok(self),
            #[cfg(feature = "model-qwen3-6-27b-gguf")]
            Self::Qwen3_6_27B_Gguf(_) => Ok(self),
            #[cfg(feature = "model-ornith-1-0-35b-gguf")]
            Self::Ornith_1_0_35B_Gguf(_) => Ok(self),
            #[cfg(feature = "model-gemma4-e2b-gguf")]
            Self::Gemma4E2B_Gguf(_) => Ok(self),
            #[cfg(feature = "model-gemma4-e4b-gguf")]
            Self::Gemma4E4B_Gguf(_) => Ok(self),
            #[cfg(feature = "model-gemma4-12b-gguf")]
            Self::Gemma4_12B_Gguf(_) => Ok(self),
            #[cfg(feature = "model-gemma4-12b-qat-gguf")]
            Self::Gemma4_12B_Qat_Gguf(_) => Ok(self),
            _ => Err(ModelError::UnsupportedCapability(CapabilityKind::Chat)),
        }
    }

    fn completion(&self) -> std::result::Result<&Self::Completion, ModelError> {
        #[allow(unreachable_patterns)]
        match self {
            #[cfg(feature = "model-qwen3-4b")]
            Self::Qwen3_4B(_) => Ok(self),
            #[cfg(feature = "model-qwen3-4b-gguf")]
            Self::Qwen3_4B_Gguf(_) => Ok(self),
            #[cfg(feature = "model-qwen3-6-27b-gguf")]
            Self::Qwen3_6_27B_Gguf(_) => Ok(self),
            #[cfg(feature = "model-ornith-1-0-35b-gguf")]
            Self::Ornith_1_0_35B_Gguf(_) => Ok(self),
            #[cfg(feature = "model-gemma4-e2b-gguf")]
            Self::Gemma4E2B_Gguf(_) => Ok(self),
            #[cfg(feature = "model-gemma4-e4b-gguf")]
            Self::Gemma4E4B_Gguf(_) => Ok(self),
            #[cfg(feature = "model-gemma4-12b-gguf")]
            Self::Gemma4_12B_Gguf(_) => Ok(self),
            #[cfg(feature = "model-gemma4-12b-qat-gguf")]
            Self::Gemma4_12B_Qat_Gguf(_) => Ok(self),
            _ => Err(ModelError::UnsupportedCapability(
                CapabilityKind::Completion,
            )),
        }
    }

    fn embeddings(&self) -> std::result::Result<&Self::Embeddings, ModelError> {
        #[allow(unreachable_patterns)]
        match self {
            #[cfg(feature = "model-google-gemma-300m")]
            Self::EmbeddingGemma300m(_) => Ok(self),
            #[cfg(feature = "model-qwen3-embedding-06b")]
            Self::Qwen3Embedding06B(_) => Ok(self),
            _ => Err(ModelError::UnsupportedCapability(
                CapabilityKind::Embeddings,
            )),
        }
    }

    async fn shutdown(self) -> std::result::Result<(), ModelError> {
        match self {
            #[cfg(feature = "model-qwen3-4b")]
            Self::Qwen3_4B(handle) => handle.shutdown().await,
            #[cfg(feature = "model-gemma4-e2b")]
            Self::Gemma4E2B(handle) => handle.shutdown().await,
            #[cfg(feature = "model-gemma4-e4b")]
            Self::Gemma4E4B(handle) => handle.shutdown().await,
            #[cfg(feature = "model-qwen3-4b-gguf")]
            Self::Qwen3_4B_Gguf(handle) => handle.shutdown().await,
            #[cfg(feature = "model-qwen3-6-27b-gguf")]
            Self::Qwen3_6_27B_Gguf(handle) => handle.shutdown().await,
            #[cfg(feature = "model-ornith-1-0-35b-gguf")]
            Self::Ornith_1_0_35B_Gguf(handle) => handle.shutdown().await,
            #[cfg(feature = "model-gemma4-e2b-gguf")]
            Self::Gemma4E2B_Gguf(handle) => handle.shutdown().await,
            #[cfg(feature = "model-gemma4-e4b-gguf")]
            Self::Gemma4E4B_Gguf(handle) => handle.shutdown().await,
            #[cfg(feature = "model-gemma4-12b-gguf")]
            Self::Gemma4_12B_Gguf(handle) => handle.shutdown().await,
            #[cfg(feature = "model-gemma4-12b-qat-gguf")]
            Self::Gemma4_12B_Qat_Gguf(handle) => handle.shutdown().await,
            #[cfg(feature = "model-google-gemma-300m")]
            Self::EmbeddingGemma300m(handle) => handle.shutdown().await,
            #[cfg(feature = "model-qwen3-embedding-06b")]
            Self::Qwen3Embedding06B(handle) => handle.shutdown().await,
            #[cfg(feature = "model-whisper-base-en")]
            Self::WhisperBaseEn(handle) => handle.shutdown().await,
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            Self::SherpaOnnxStreamingZipformerEn(handle) => handle.shutdown().await,
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            Self::SherpaOnnxStreamingZipformerEnKroko2025(handle) => handle.shutdown().await,
            #[cfg(feature = "model-moonshine-streaming")]
            Self::MoonshineStreamingEn(handle) => handle.shutdown().await,
            #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
            Self::PiperEnUsLjspeechMedium(handle) => handle.shutdown().await,
            #[cfg(feature = "model-kokoro-82m")]
            Self::Kokoro82m(handle) => handle.shutdown().await,
            #[cfg(feature = "model-qwen3-tts-cpp")]
            Self::Qwen3TtsCpp0_6B(handle) => handle.shutdown().await,
        }
    }
}

#[async_trait::async_trait]
impl ChatModel for CuratedHandle {
    async fn generate(
        &self,
        request: motlie_model::ChatRequest,
    ) -> std::result::Result<motlie_model::ChatResponse, ModelError> {
        #[allow(unreachable_patterns)]
        match self {
            #[cfg(feature = "model-qwen3-4b")]
            Self::Qwen3_4B(handle) => handle.generate(request).await,
            #[cfg(feature = "model-gemma4-e2b")]
            Self::Gemma4E2B(handle) => handle.generate(request).await,
            #[cfg(feature = "model-gemma4-e4b")]
            Self::Gemma4E4B(handle) => handle.generate(request).await,
            #[cfg(feature = "model-qwen3-4b-gguf")]
            Self::Qwen3_4B_Gguf(handle) => handle.generate(request).await,
            #[cfg(feature = "model-qwen3-6-27b-gguf")]
            Self::Qwen3_6_27B_Gguf(handle) => handle.generate(request).await,
            #[cfg(feature = "model-ornith-1-0-35b-gguf")]
            Self::Ornith_1_0_35B_Gguf(handle) => handle.generate(request).await,
            #[cfg(feature = "model-gemma4-e2b-gguf")]
            Self::Gemma4E2B_Gguf(handle) => handle.generate(request).await,
            #[cfg(feature = "model-gemma4-e4b-gguf")]
            Self::Gemma4E4B_Gguf(handle) => handle.generate(request).await,
            #[cfg(feature = "model-gemma4-12b-gguf")]
            Self::Gemma4_12B_Gguf(handle) => handle.generate(request).await,
            #[cfg(feature = "model-gemma4-12b-qat-gguf")]
            Self::Gemma4_12B_Qat_Gguf(handle) => handle.generate(request).await,
            _ => UnsupportedChat.generate(request).await,
        }
    }
}

#[async_trait::async_trait]
impl CompletionModel for CuratedHandle {
    async fn complete(
        &self,
        request: motlie_model::CompletionRequest,
    ) -> std::result::Result<motlie_model::CompletionResponse, ModelError> {
        #[allow(unreachable_patterns)]
        match self {
            #[cfg(feature = "model-qwen3-4b")]
            Self::Qwen3_4B(handle) => handle.complete(request).await,
            #[cfg(feature = "model-qwen3-4b-gguf")]
            Self::Qwen3_4B_Gguf(handle) => handle.complete(request).await,
            #[cfg(feature = "model-qwen3-6-27b-gguf")]
            Self::Qwen3_6_27B_Gguf(handle) => handle.complete(request).await,
            #[cfg(feature = "model-ornith-1-0-35b-gguf")]
            Self::Ornith_1_0_35B_Gguf(handle) => handle.complete(request).await,
            #[cfg(feature = "model-gemma4-e2b-gguf")]
            Self::Gemma4E2B_Gguf(handle) => handle.complete(request).await,
            #[cfg(feature = "model-gemma4-e4b-gguf")]
            Self::Gemma4E4B_Gguf(handle) => handle.complete(request).await,
            #[cfg(feature = "model-gemma4-12b-gguf")]
            Self::Gemma4_12B_Gguf(handle) => handle.complete(request).await,
            #[cfg(feature = "model-gemma4-12b-qat-gguf")]
            Self::Gemma4_12B_Qat_Gguf(handle) => handle.complete(request).await,
            _ => UnsupportedCompletion.complete(request).await,
        }
    }
}

#[async_trait::async_trait]
impl EmbeddingModel for CuratedHandle {
    async fn embed(
        &self,
        request: motlie_model::EmbeddingRequest,
    ) -> std::result::Result<motlie_model::EmbeddingResponse, ModelError> {
        #[allow(unreachable_patterns)]
        match self {
            #[cfg(feature = "model-google-gemma-300m")]
            Self::EmbeddingGemma300m(handle) => handle.embed(request).await,
            #[cfg(feature = "model-qwen3-embedding-06b")]
            Self::Qwen3Embedding06B(handle) => handle.embed(request).await,
            _ => UnsupportedEmbeddings.embed(request).await,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArtifactGating {
    Public,
    Manual,
    Unknown,
}

impl ArtifactGating {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Public => "public",
            Self::Manual => "manual",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArtifactProvenance {
    pub license: &'static str,
    pub gating: ArtifactGating,
}

impl ArtifactProvenance {
    pub const fn new(license: &'static str, gating: ArtifactGating) -> Self {
        Self { license, gating }
    }

    pub const fn unknown() -> Self {
        Self {
            license: "unknown",
            gating: ArtifactGating::Unknown,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BundleArtifactSource {
    pub label: &'static str,
    pub source: ArtifactSource,
    pub include: Vec<ArtifactRule>,
    pub provenance: ArtifactProvenance,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DerivedArtifactRecipe {
    CopyFromDownloaded { source: &'static str },
    KokoroTokensFromTokenizerJson { source: &'static str },
}

impl DerivedArtifactRecipe {
    pub const fn source(&self) -> &'static str {
        match self {
            Self::CopyFromDownloaded { source } => source,
            Self::KokoroTokensFromTokenizerJson { source } => source,
        }
    }

    pub const fn label(&self) -> &'static str {
        match self {
            Self::CopyFromDownloaded { .. } => "copy from downloaded source artifact",
            Self::KokoroTokensFromTokenizerJson { .. } => {
                "generate from tokenizer.json model.vocab via kokoro_82m::tokens_txt_from_tokenizer_json (introduced by 91cc0f32)"
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DerivedBundleArtifact {
    pub label: &'static str,
    pub output: &'static str,
    pub recipe: DerivedArtifactRecipe,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BundleArtifacts {
    pub source_label: &'static str,
    pub control_name: &'static str,
    pub format: CheckpointFormat,
    pub source: ArtifactSource,
    pub include: Vec<ArtifactRule>,
    pub quantization: Option<QuantizationScheme>,
    pub provenance: ArtifactProvenance,
    pub extra_sources: Vec<BundleArtifactSource>,
    pub derived: Vec<DerivedBundleArtifact>,
}

impl BundleArtifacts {
    pub fn includes(&self, filename: &str) -> bool {
        self.include.iter().any(|rule| rule.matches(filename))
            || self
                .extra_sources
                .iter()
                .any(|source| source.include.iter().any(|rule| rule.matches(filename)))
            || self.derived.iter().any(|artifact| {
                artifact.output == filename
                    || (artifact.output.ends_with('/') && filename.starts_with(artifact.output))
            })
    }

    pub fn include_for_quantization(
        &self,
        quantization: Option<QuantizationScheme>,
    ) -> Vec<ArtifactRule> {
        artifact_rules_for_quantization(self.format, &self.include, quantization)
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
    pub quantization_scheme: Option<QuantizationScheme>,
}

pub fn default_artifact_root() -> PathBuf {
    artifact_manifest::canonical_artifact_root()
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

    let mut downloaded = download_checkpoint_artifacts_with_options(
        &ModelCheckpoint {
            format: artifacts.format,
            source: artifacts.source.clone(),
            include: artifacts.include_for_quantization(options.quantization_scheme),
            quantization: options.quantization_scheme.or(artifacts.quantization),
        },
        artifact_root,
        options,
    )?;
    for source in &artifacts.extra_sources {
        downloaded.extend(download_artifact_source_with_options(
            &source.source,
            &source.include,
            artifact_root,
            options,
        )?);
    }
    prepare_downloaded_bundle_artifacts(descriptor, &mut downloaded)?;
    downloaded.sort();
    downloaded.dedup();

    Ok(ArtifactDownloadSummary {
        bundle_id: bundle_id.clone(),
        downloaded,
    })
}

#[cfg(feature = "model-kokoro-82m")]
fn prepare_downloaded_bundle_artifacts(
    descriptor: &BundleDescriptor,
    downloaded: &mut Vec<PathBuf>,
) -> Result<()> {
    if descriptor.id.as_str() == "kokoro_82m" {
        tts::kokoro_82m::prepare_downloaded_artifacts(downloaded).map_err(|message| {
            ModelsError::ArtifactPreparation {
                bundle_id: descriptor.id.clone(),
                message,
            }
        })?;
    }

    Ok(())
}

#[cfg(not(feature = "model-kokoro-82m"))]
fn prepare_downloaded_bundle_artifacts(
    _descriptor: &BundleDescriptor,
    _downloaded: &mut [PathBuf],
) -> Result<()> {
    Ok(())
}

fn download_checkpoint_artifacts_with_options(
    checkpoint: &ModelCheckpoint,
    artifact_root: &Path,
    options: &ArtifactDownloadOptions,
) -> Result<Vec<PathBuf>> {
    checkpoint.validate_quantization().map_err(|source| {
        ModelsError::InvalidQuantizationScheme {
            message: source.to_string(),
        }
    })?;

    download_artifact_source_with_options(
        &checkpoint.source,
        &checkpoint.include,
        artifact_root,
        options,
    )
}

fn download_artifact_source_with_options(
    source: &ArtifactSource,
    include: &[ArtifactRule],
    artifact_root: &Path,
    options: &ArtifactDownloadOptions,
) -> Result<Vec<PathBuf>> {
    match source {
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
                .with_progress(false)
                .build()
                .map_err(|source| ModelsError::HuggingFaceClient {
                    message: source.to_string(),
                })?;
            let repo_api = api.model((*repo).to_string());
            let info = repo_api
                .info()
                .map_err(|source| ModelsError::InspectModelRepo {
                    repo,
                    message: source.to_string(),
                })?;

            let mut downloaded = Vec::new();
            for sibling in info.siblings {
                if include.iter().any(|rule| rule.matches(&sibling.rfilename)) {
                    let path = repo_api.get(&sibling.rfilename).map_err(|source| {
                        ModelsError::DownloadArtifact {
                            repo,
                            filename: sibling.rfilename.clone(),
                            message: source.to_string(),
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

pub fn artifact_rules_for_quantization(
    format: CheckpointFormat,
    include: &[ArtifactRule],
    quantization: Option<QuantizationScheme>,
) -> Vec<ArtifactRule> {
    if format != CheckpointFormat::Gguf {
        return include.to_vec();
    }
    let Some(quantization) = quantization else {
        return include.to_vec();
    };

    include
        .iter()
        .filter(|rule| artifact_rule_matches_gguf_quant(rule, quantization))
        .cloned()
        .collect()
}

fn artifact_rule_matches_gguf_quant(rule: &ArtifactRule, quantization: QuantizationScheme) -> bool {
    let raw = match rule {
        ArtifactRule::Exact(value) | ArtifactRule::Prefix(value) | ArtifactRule::Suffix(value) => {
            value
        }
    };
    let normalized = raw.to_ascii_lowercase();
    if normalized.contains("tokenizer") {
        return true;
    }
    gguf_quant_markers(quantization)
        .iter()
        .any(|marker| normalized.contains(marker))
}

fn gguf_quant_markers(quantization: QuantizationScheme) -> &'static [&'static str] {
    match quantization {
        QuantizationScheme::GgufQ4_K_M => &["q4_k_m"],
        QuantizationScheme::GgufQ4_0 => &["q4_0"],
        QuantizationScheme::GgufQ5_K_M => &["q5_k_m"],
        QuantizationScheme::GgufQ8_0 => &["q8_0"],
        QuantizationScheme::Fp16 => &["f16"],
        _ => &[],
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
            quantization: artifacts.quantization,
        })
    }
}

#[allow(dead_code)]
pub(crate) fn bundle_artifacts_from_checkpoint(
    control_name: &'static str,
    checkpoint: &ModelCheckpoint,
    provenance: ArtifactProvenance,
) -> BundleArtifacts {
    checkpoint
        .validate_quantization()
        .expect("curated checkpoint quantization must use a checkpoint-legal scheme");
    BundleArtifacts {
        source_label: "primary",
        control_name,
        format: checkpoint.format,
        source: checkpoint.source.clone(),
        include: checkpoint.include.clone(),
        quantization: checkpoint.quantization,
        provenance,
        extra_sources: Vec::new(),
        derived: Vec::new(),
    }
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
        feature = "model-kokoro-82m",
        feature = "model-qwen3-tts-cpp",
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
        feature = "model-qwen3-6-27b-gguf",
        feature = "model-ornith-1-0-35b-gguf",
        feature = "model-gemma4-e2b",
        feature = "model-gemma4-e2b-gguf",
        feature = "model-gemma4-e4b",
        feature = "model-gemma4-12b-gguf",
        feature = "model-gemma4-12b-qat-gguf",
        feature = "model-gemma4-e4b-gguf",
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
    feature = "model-qwen3-6-27b-gguf",
    feature = "model-ornith-1-0-35b-gguf",
    feature = "model-google-gemma-300m",
    feature = "model-qwen3-embedding-06b",
    feature = "model-gemma4-e2b",
    feature = "model-gemma4-e2b-gguf",
    feature = "model-gemma4-e4b",
    feature = "model-gemma4-12b-gguf",
    feature = "model-gemma4-12b-qat-gguf",
    feature = "model-gemma4-e4b-gguf",
    feature = "model-piper-en-us-ljspeech-medium",
    feature = "model-kokoro-82m",
    feature = "model-qwen3-tts-cpp",
    feature = "model-moonshine-streaming",
    feature = "model-sherpa-onnx-streaming",
    feature = "model-whisper-base-en",
))]
impl ModelSelector {
    pub fn as_str(&self) -> String {
        match self {
            #[cfg(any(
                feature = "model-piper-en-us-ljspeech-medium",
                feature = "model-kokoro-82m",
                feature = "model-qwen3-tts-cpp",
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
                feature = "model-qwen3-6-27b-gguf",
                feature = "model-ornith-1-0-35b-gguf",
                feature = "model-gemma4-e2b",
                feature = "model-gemma4-e2b-gguf",
                feature = "model-gemma4-e4b",
                feature = "model-gemma4-12b-gguf",
                feature = "model-gemma4-12b-qat-gguf",
                feature = "model-gemma4-e4b-gguf",
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
                feature = "model-kokoro-82m",
                feature = "model-qwen3-tts-cpp",
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
                feature = "model-qwen3-6-27b-gguf",
                feature = "model-ornith-1-0-35b-gguf",
                feature = "model-gemma4-e2b",
                feature = "model-gemma4-e2b-gguf",
                feature = "model-gemma4-e4b",
                feature = "model-gemma4-12b-gguf",
                feature = "model-gemma4-12b-qat-gguf",
                feature = "model-gemma4-e4b-gguf",
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
                feature = "model-kokoro-82m",
                feature = "model-qwen3-tts-cpp",
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
                feature = "model-qwen3-6-27b-gguf",
                feature = "model-ornith-1-0-35b-gguf",
                feature = "model-gemma4-e2b",
                feature = "model-gemma4-e2b-gguf",
                feature = "model-gemma4-e4b",
                feature = "model-gemma4-12b-gguf",
                feature = "model-gemma4-12b-qat-gguf",
                feature = "model-gemma4-e4b-gguf",
            ))]
            Self::Chat(model) => model.descriptor(),
            #[cfg(any(
                feature = "model-google-gemma-300m",
                feature = "model-qwen3-embedding-06b"
            ))]
            Self::Embedding(model) => model.descriptor(),
        }
    }

    pub fn bundle(&self) -> Result<CuratedBundle> {
        match self {
            #[cfg(any(
                feature = "model-piper-en-us-ljspeech-medium",
                feature = "model-kokoro-82m",
                feature = "model-qwen3-tts-cpp",
            ))]
            Self::Tts(model) => Ok(model.bundle()),
            #[cfg(any(
                feature = "model-moonshine-streaming",
                feature = "model-sherpa-onnx-streaming",
                feature = "model-whisper-base-en"
            ))]
            Self::Asr(model) => Ok(model.bundle()),
            #[cfg(any(
                feature = "model-qwen3-4b",
                feature = "model-qwen3-4b-gguf",
                feature = "model-qwen3-6-27b-gguf",
                feature = "model-ornith-1-0-35b-gguf",
                feature = "model-gemma4-e2b",
                feature = "model-gemma4-e2b-gguf",
                feature = "model-gemma4-e4b",
                feature = "model-gemma4-12b-gguf",
                feature = "model-gemma4-12b-qat-gguf",
                feature = "model-gemma4-e4b-gguf",
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
    feature = "model-qwen3-6-27b-gguf",
    feature = "model-ornith-1-0-35b-gguf",
    feature = "model-google-gemma-300m",
    feature = "model-qwen3-embedding-06b",
    feature = "model-gemma4-e2b",
    feature = "model-gemma4-e2b-gguf",
    feature = "model-gemma4-e4b",
    feature = "model-gemma4-12b-gguf",
    feature = "model-gemma4-12b-qat-gguf",
    feature = "model-gemma4-e4b-gguf",
    feature = "model-piper-en-us-ljspeech-medium",
    feature = "model-kokoro-82m",
    feature = "model-qwen3-tts-cpp",
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
            #[cfg(not(feature = "model-kokoro-82m"))]
            if raw == tts::KOKORO_82M_SELECTOR {
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
            #[cfg(any(
                feature = "model-piper-en-us-ljspeech-medium",
                feature = "model-kokoro-82m",
                feature = "model-qwen3-tts-cpp",
            ))]
            return Ok(Self::Tts(raw.parse()?));
            #[cfg(not(any(
                feature = "model-piper-en-us-ljspeech-medium",
                feature = "model-kokoro-82m",
                feature = "model-qwen3-tts-cpp",
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
            #[cfg(not(feature = "model-gemma4-e4b"))]
            if raw == chat::GEMMA4_E4B_SELECTOR {
                return Err(ModelsError::ModelUnavailable {
                    selector: value.to_owned(),
                });
            }
            #[cfg(not(feature = "model-gemma4-e4b-gguf"))]
            if raw == chat::GEMMA4_E4B_GGUF_SELECTOR {
                return Err(ModelsError::ModelUnavailable {
                    selector: value.to_owned(),
                });
            }
            #[cfg(not(feature = "model-gemma4-12b-gguf"))]
            if raw == chat::GEMMA4_12B_GGUF_SELECTOR {
                return Err(ModelsError::ModelUnavailable {
                    selector: value.to_owned(),
                });
            }
            #[cfg(not(feature = "model-gemma4-12b-qat-gguf"))]
            if raw == chat::GEMMA4_12B_QAT_GGUF_SELECTOR {
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
            #[cfg(not(feature = "model-qwen3-6-27b-gguf"))]
            if raw == chat::QWEN3_6_27B_GGUF_SELECTOR {
                return Err(ModelsError::ModelUnavailable {
                    selector: value.to_owned(),
                });
            }
            #[cfg(not(feature = "model-ornith-1-0-35b-gguf"))]
            if raw == chat::ORNITH_1_0_35B_GGUF_SELECTOR {
                return Err(ModelsError::ModelUnavailable {
                    selector: value.to_owned(),
                });
            }
            #[cfg(any(
                feature = "model-qwen3-4b",
                feature = "model-qwen3-4b-gguf",
                feature = "model-qwen3-6-27b-gguf",
                feature = "model-ornith-1-0-35b-gguf",
                feature = "model-gemma4-e2b",
                feature = "model-gemma4-e2b-gguf",
                feature = "model-gemma4-e4b",
                feature = "model-gemma4-12b-gguf",
                feature = "model-gemma4-12b-qat-gguf",
                feature = "model-gemma4-e4b-gguf",
            ))]
            return Ok(Self::Chat(raw.parse()?));
            #[cfg(not(any(
                feature = "model-qwen3-4b",
                feature = "model-qwen3-4b-gguf",
                feature = "model-qwen3-6-27b-gguf",
                feature = "model-ornith-1-0-35b-gguf",
                feature = "model-gemma4-e2b",
                feature = "model-gemma4-e2b-gguf",
                feature = "model-gemma4-e4b",
                feature = "model-gemma4-12b-gguf",
                feature = "model-gemma4-12b-qat-gguf",
                feature = "model-gemma4-e4b-gguf",
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
            #[cfg(not(feature = "model-sherpa-onnx-streaming"))]
            if raw == asr::SHERPA_ONNX_STREAMING_KROKO_2025_SELECTOR {
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

fn bundle_from_id(id: &BundleId) -> Option<CuratedBundle> {
    match id.as_str() {
        #[cfg(feature = "model-qwen3-4b")]
        "qwen3_4b" => Some(CuratedBundle::Qwen3_4B),
        #[cfg(feature = "model-gemma4-e2b")]
        "gemma4_e2b" => Some(CuratedBundle::Gemma4E2B),
        #[cfg(feature = "model-gemma4-e4b")]
        "gemma4_e4b" => Some(CuratedBundle::Gemma4E4B),
        #[cfg(feature = "model-qwen3-4b-gguf")]
        "qwen3_4b_gguf" => Some(CuratedBundle::Qwen3_4B_Gguf),
        #[cfg(feature = "model-qwen3-6-27b-gguf")]
        "qwen3_6_27b_gguf" => Some(CuratedBundle::Qwen3_6_27B_Gguf),
        #[cfg(feature = "model-ornith-1-0-35b-gguf")]
        "ornith_1_0_35b_gguf" => Some(CuratedBundle::Ornith_1_0_35B_Gguf),
        #[cfg(feature = "model-gemma4-e2b-gguf")]
        "gemma4_e2b_gguf" => Some(CuratedBundle::Gemma4E2B_Gguf),
        #[cfg(feature = "model-gemma4-e4b-gguf")]
        "gemma4_e4b_gguf" => Some(CuratedBundle::Gemma4E4B_Gguf),
        #[cfg(feature = "model-gemma4-12b-gguf")]
        "gemma4_12b_gguf" => Some(CuratedBundle::Gemma4_12B_Gguf),
        #[cfg(feature = "model-gemma4-12b-qat-gguf")]
        "gemma4_12b_qat_gguf" => Some(CuratedBundle::Gemma4_12B_Qat_Gguf),
        #[cfg(feature = "model-google-gemma-300m")]
        "embeddinggemma_300m" => Some(CuratedBundle::EmbeddingGemma300m),
        #[cfg(feature = "model-qwen3-embedding-06b")]
        "qwen3_embedding_06b" => Some(CuratedBundle::Qwen3Embedding06B),
        #[cfg(feature = "model-whisper-base-en")]
        "whisper_base_en" => Some(CuratedBundle::WhisperBaseEn),
        #[cfg(feature = "model-sherpa-onnx-streaming")]
        "sherpa_onnx_streaming_zipformer_en" => Some(CuratedBundle::SherpaOnnxStreamingZipformerEn),
        #[cfg(feature = "model-sherpa-onnx-streaming")]
        "sherpa_onnx_streaming_zipformer_en_kroko_2025" => {
            Some(CuratedBundle::SherpaOnnxStreamingZipformerEnKroko2025)
        }
        #[cfg(feature = "model-moonshine-streaming")]
        "moonshine_streaming_en" => Some(CuratedBundle::MoonshineStreamingEn),
        #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
        "piper_en_us_ljspeech_medium" => Some(CuratedBundle::PiperEnUsLjspeechMedium),
        #[cfg(feature = "model-kokoro-82m")]
        "kokoro_82m" => Some(CuratedBundle::Kokoro82m),
        #[cfg(feature = "model-qwen3-tts-cpp")]
        "qwen3_tts_cpp_0_6b" => Some(CuratedBundle::Qwen3TtsCpp0_6B),
        _ => None,
    }
}

fn bundle_from_resolved(resolved: &ResolvedModelDescriptor) -> Option<CuratedBundle> {
    match (
        resolved.identity.id.as_str(),
        resolved.variant.backend,
        resolved.variant.checkpoint.format,
    ) {
        #[cfg(feature = "model-qwen3-4b")]
        ("qwen3_4b", BackendKind::MistralRs, CheckpointFormat::Safetensors) => {
            Some(CuratedBundle::Qwen3_4B)
        }
        #[cfg(feature = "model-gemma4-e2b")]
        ("gemma4_e2b", BackendKind::MistralRs, CheckpointFormat::Safetensors) => {
            Some(CuratedBundle::Gemma4E2B)
        }
        #[cfg(feature = "model-gemma4-e4b")]
        ("gemma4_e4b", BackendKind::MistralRs, CheckpointFormat::Safetensors) => {
            Some(CuratedBundle::Gemma4E4B)
        }
        #[cfg(feature = "model-qwen3-4b-gguf")]
        ("qwen3_4b", BackendKind::LlamaCpp, CheckpointFormat::Gguf) => {
            Some(CuratedBundle::Qwen3_4B_Gguf)
        }
        #[cfg(feature = "model-qwen3-6-27b-gguf")]
        ("qwen3_6_27b", BackendKind::LlamaCpp, CheckpointFormat::Gguf) => {
            Some(CuratedBundle::Qwen3_6_27B_Gguf)
        }
        #[cfg(feature = "model-ornith-1-0-35b-gguf")]
        ("ornith_1_0_35b", BackendKind::LlamaCpp, CheckpointFormat::Gguf) => {
            Some(CuratedBundle::Ornith_1_0_35B_Gguf)
        }
        #[cfg(feature = "model-gemma4-e2b-gguf")]
        ("gemma4_e2b", BackendKind::LlamaCpp, CheckpointFormat::Gguf) => {
            Some(CuratedBundle::Gemma4E2B_Gguf)
        }
        #[cfg(feature = "model-gemma4-e4b-gguf")]
        ("gemma4_e4b", BackendKind::LlamaCpp, CheckpointFormat::Gguf) => {
            Some(CuratedBundle::Gemma4E4B_Gguf)
        }
        #[cfg(feature = "model-gemma4-12b-gguf")]
        ("gemma4_12b", BackendKind::LlamaCpp, CheckpointFormat::Gguf) => {
            Some(CuratedBundle::Gemma4_12B_Gguf)
        }
        #[cfg(feature = "model-google-gemma-300m")]
        ("embeddinggemma_300m", BackendKind::MistralRs, CheckpointFormat::Safetensors) => {
            Some(CuratedBundle::EmbeddingGemma300m)
        }
        #[cfg(feature = "model-qwen3-embedding-06b")]
        ("qwen3_embedding_06b", BackendKind::MistralRs, CheckpointFormat::Safetensors) => {
            Some(CuratedBundle::Qwen3Embedding06B)
        }
        #[cfg(feature = "model-whisper-base-en")]
        ("whisper_base_en", BackendKind::WhisperCpp, CheckpointFormat::Ggml) => {
            Some(CuratedBundle::WhisperBaseEn)
        }
        #[cfg(feature = "model-sherpa-onnx-streaming")]
        ("sherpa_onnx_streaming_zipformer_en", BackendKind::SherpaOnnx, CheckpointFormat::Onnx) => {
            Some(CuratedBundle::SherpaOnnxStreamingZipformerEn)
        }
        #[cfg(feature = "model-sherpa-onnx-streaming")]
        (
            "sherpa_onnx_streaming_zipformer_en_kroko_2025",
            BackendKind::SherpaOnnx,
            CheckpointFormat::Onnx,
        ) => Some(CuratedBundle::SherpaOnnxStreamingZipformerEnKroko2025),
        #[cfg(feature = "model-moonshine-streaming")]
        ("moonshine_streaming_en", BackendKind::Ort, CheckpointFormat::Onnx) => {
            Some(CuratedBundle::MoonshineStreamingEn)
        }
        #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
        ("piper_en_us_ljspeech_medium", BackendKind::Ort, CheckpointFormat::Onnx) => {
            Some(CuratedBundle::PiperEnUsLjspeechMedium)
        }
        #[cfg(feature = "model-kokoro-82m")]
        ("kokoro_82m", BackendKind::Ort, CheckpointFormat::Onnx) => Some(CuratedBundle::Kokoro82m),
        #[cfg(feature = "model-qwen3-tts-cpp")]
        ("qwen3_tts_cpp_0_6b", BackendKind::Qwen3TtsCpp, CheckpointFormat::Gguf) => {
            Some(CuratedBundle::Qwen3TtsCpp0_6B)
        }
        _ => None,
    }
}

struct ModelCatalogEntry {
    identity: ModelIdentity,
    variants: Vec<ModelVariantDescriptor>,
}

/// In-memory registry of curated bundle descriptors and constructors.
#[derive(Default)]
pub struct Catalog {
    bundles: BTreeMap<BundleId, BundleDescriptor>,
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
        embeddings::embeddinggemma_300m::register(&mut catalog);
        #[cfg(feature = "model-qwen3-embedding-06b")]
        embeddings::qwen3_embedding_06b::register(&mut catalog);
        #[cfg(feature = "model-qwen3-4b")]
        chat::qwen3_4b::register(&mut catalog);
        #[cfg(feature = "model-gemma4-e2b")]
        chat::gemma4_e2b::register(&mut catalog);
        #[cfg(feature = "model-gemma4-e4b")]
        chat::gemma4_e4b::register(&mut catalog);
        #[cfg(feature = "model-qwen3-4b-gguf")]
        chat::qwen3_4b_gguf::register(&mut catalog);
        #[cfg(feature = "model-qwen3-6-27b-gguf")]
        chat::qwen3_6_27b_gguf::register(&mut catalog);
        #[cfg(feature = "model-ornith-1-0-35b-gguf")]
        chat::ornith_1_0_35b_gguf::register(&mut catalog);
        #[cfg(feature = "model-gemma4-e2b-gguf")]
        chat::gemma4_e2b_gguf::register(&mut catalog);
        #[cfg(feature = "model-gemma4-e4b-gguf")]
        chat::gemma4_e4b_gguf::register(&mut catalog);
        #[cfg(feature = "model-gemma4-12b-gguf")]
        chat::gemma4_12b_gguf::register(&mut catalog);
        #[cfg(feature = "model-gemma4-12b-qat-gguf")]
        chat::gemma4_12b_qat_gguf::register(&mut catalog);
        #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
        tts::piper_en_us_ljspeech_medium::register(&mut catalog);
        #[cfg(feature = "model-kokoro-82m")]
        tts::kokoro_82m::register(&mut catalog);
        #[cfg(feature = "model-qwen3-tts-cpp")]
        tts::qwen3_tts_cpp_0_6b::register(&mut catalog);
        #[cfg(feature = "model-moonshine-streaming")]
        asr::moonshine_streaming_en::register(&mut catalog);
        #[cfg(feature = "model-sherpa-onnx-streaming")]
        asr::sherpa_onnx_streaming_zipformer_en::register(&mut catalog);
        #[cfg(feature = "model-sherpa-onnx-streaming")]
        asr::sherpa_onnx_streaming_zipformer_en_kroko_2025::register(&mut catalog);
        #[cfg(feature = "model-whisper-base-en")]
        asr::whisper_base_en::register(&mut catalog);
        catalog
    }

    pub fn register_descriptor(
        &mut self,
        descriptor: BundleDescriptor,
    ) -> Option<BundleDescriptor> {
        self.bundles
            .insert(descriptor.id.clone(), descriptor.clone())
    }

    #[allow(dead_code)]
    pub(crate) fn register_model_variant(
        &mut self,
        identity: ModelIdentity,
        variant: ModelVariantDescriptor,
    ) {
        let entry = self
            .models
            .entry(identity.id.clone())
            .or_insert_with(|| ModelCatalogEntry {
                identity: identity.clone(),
                variants: Vec::new(),
            });
        entry.identity = identity;
        if !entry.variants.iter().any(|existing| existing == &variant) {
            entry.variants.push(variant);
        }
    }

    pub fn bundle(&self, id: &BundleId) -> Option<&BundleDescriptor> {
        self.bundles.get(id)
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
        let variants = entry.variants.clone();
        Some(variants.into_iter())
    }

    pub fn artifacts(&self, id: &BundleId) -> Option<&BundleArtifacts> {
        self.bundle(id)
            .and_then(|descriptor| descriptor.artifacts.as_ref())
    }

    pub fn instantiate(&self, id: &BundleId) -> Option<CuratedBundle> {
        bundle_from_id(id)
    }

    pub fn bundles(&self) -> impl Iterator<Item = &BundleDescriptor> {
        self.bundles.values()
    }

    pub fn bundles_for_track(&self, track: EvalTrack) -> impl Iterator<Item = &BundleDescriptor> {
        self.bundles
            .values()
            .filter(move |descriptor| descriptor.supports_track(track))
    }

    pub fn resolve_model(
        &self,
        id: &BundleId,
        options: &ResolveModelOptions,
    ) -> Option<ResolvedModelDescriptor> {
        let entry = self.models.get(id)?;

        let exact = entry.variants.iter().find_map(|variant| {
            let backend_ok = options
                .backend_preference
                .is_none_or(|backend| variant.backend == backend);
            if !backend_ok {
                return None;
            }

            let format_ok = options
                .format_preference
                .is_none_or(|format| variant.checkpoint.format == format);
            if !format_ok {
                return None;
            }

            Some(variant.clone())
        })?;

        Some(ResolvedModelDescriptor {
            identity: entry.identity.clone(),
            variant: exact,
        })
    }

    pub fn instantiate_resolved(
        &self,
        resolved: &ResolvedModelDescriptor,
    ) -> Option<CuratedBundle> {
        bundle_from_resolved(resolved)
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
    use super::*;

    #[test]
    fn canonical_ids_are_unique_and_complete() {
        let mut seen = std::collections::BTreeSet::new();
        for id in CuratedBundle::CANONICAL_IDS {
            assert!(
                seen.insert(*id),
                "duplicate canonical id `{id}` in CANONICAL_IDS"
            );
        }
        assert_eq!(
            CuratedBundle::CANONICAL_IDS.len(),
            19,
            "expected 19 curated bundles; update CANONICAL_IDS and the enum together"
        );
    }

    #[test]
    fn canonical_ids_match_bundle_ids() {
        // Each compiled variant's canonical_id() must equal its descriptor
        // bundle_id() and be a member of CANONICAL_IDS, and must round-trip
        // through bundle_from_id(). Run with all model features (CI) to cover
        // all 19 variants; with fewer features this covers the compiled subset.
        let variants: Vec<CuratedBundle> = vec![
            #[cfg(feature = "model-qwen3-4b")]
            CuratedBundle::Qwen3_4B,
            #[cfg(feature = "model-gemma4-e2b")]
            CuratedBundle::Gemma4E2B,
            #[cfg(feature = "model-gemma4-e4b")]
            CuratedBundle::Gemma4E4B,
            #[cfg(feature = "model-qwen3-4b-gguf")]
            CuratedBundle::Qwen3_4B_Gguf,
            #[cfg(feature = "model-qwen3-6-27b-gguf")]
            CuratedBundle::Qwen3_6_27B_Gguf,
            #[cfg(feature = "model-ornith-1-0-35b-gguf")]
            CuratedBundle::Ornith_1_0_35B_Gguf,
            #[cfg(feature = "model-gemma4-e2b-gguf")]
            CuratedBundle::Gemma4E2B_Gguf,
            #[cfg(feature = "model-gemma4-e4b-gguf")]
            CuratedBundle::Gemma4E4B_Gguf,
            #[cfg(feature = "model-gemma4-12b-gguf")]
            CuratedBundle::Gemma4_12B_Gguf,
            #[cfg(feature = "model-gemma4-12b-qat-gguf")]
            CuratedBundle::Gemma4_12B_Qat_Gguf,
            #[cfg(feature = "model-google-gemma-300m")]
            CuratedBundle::EmbeddingGemma300m,
            #[cfg(feature = "model-qwen3-embedding-06b")]
            CuratedBundle::Qwen3Embedding06B,
            #[cfg(feature = "model-whisper-base-en")]
            CuratedBundle::WhisperBaseEn,
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            CuratedBundle::SherpaOnnxStreamingZipformerEn,
            #[cfg(feature = "model-sherpa-onnx-streaming")]
            CuratedBundle::SherpaOnnxStreamingZipformerEnKroko2025,
            #[cfg(feature = "model-moonshine-streaming")]
            CuratedBundle::MoonshineStreamingEn,
            #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
            CuratedBundle::PiperEnUsLjspeechMedium,
            #[cfg(feature = "model-kokoro-82m")]
            CuratedBundle::Kokoro82m,
            #[cfg(feature = "model-qwen3-tts-cpp")]
            CuratedBundle::Qwen3TtsCpp0_6B,
        ];

        for variant in variants {
            let canonical = variant.canonical_id();
            assert_eq!(
                canonical,
                variant.bundle_id().as_str(),
                "canonical_id() must equal bundle_id() for {variant:?}"
            );
            assert!(
                CuratedBundle::CANONICAL_IDS.contains(&canonical),
                "canonical id `{canonical}` is missing from CANONICAL_IDS"
            );
            assert_eq!(
                bundle_from_id(&BundleId::new(canonical)),
                Some(variant),
                "canonical id `{canonical}` must round-trip through bundle_from_id()"
            );
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

    #[cfg(any(
        feature = "model-whisper-base-en",
        feature = "model-sherpa-onnx-streaming",
        feature = "model-moonshine-streaming",
        feature = "model-piper-en-us-ljspeech-medium",
        feature = "model-qwen3-tts-cpp",
    ))]
    #[test]
    fn resolve_typed_artifact_policy_rewrites_local_only_root() {
        let options = StartOptions {
            artifact_policy: Some(motlie_model::ArtifactPolicy::LocalOnly {
                root: PathBuf::from("/tmp/cache"),
            }),
            quantization_scheme: Some(motlie_model::QuantizationScheme::GgufQ4_K_M),
            unpack_root: Some(PathBuf::from("/tmp/unpack")),
            max_concurrency: Some(2),
        };

        let resolved =
            resolve_typed_artifact_policy(options, |root| Ok(root.join("snapshots/commit")))
                .expect("local-only policy should resolve");

        assert_eq!(
            resolved.artifact_policy,
            Some(motlie_model::ArtifactPolicy::LocalOnly {
                root: PathBuf::from("/tmp/cache/snapshots/commit"),
            })
        );
        assert_eq!(
            resolved.quantization_scheme,
            Some(motlie_model::QuantizationScheme::GgufQ4_K_M)
        );
        assert_eq!(resolved.unpack_root, Some(PathBuf::from("/tmp/unpack")));
        assert_eq!(resolved.max_concurrency, Some(2));
    }

    #[cfg(any(
        feature = "model-whisper-base-en",
        feature = "model-sherpa-onnx-streaming",
        feature = "model-moonshine-streaming",
        feature = "model-piper-en-us-ljspeech-medium",
        feature = "model-qwen3-tts-cpp",
    ))]
    #[test]
    fn resolve_typed_artifact_policy_leaves_allow_fetch_unchanged() {
        let options = StartOptions {
            artifact_policy: Some(motlie_model::ArtifactPolicy::AllowFetch {
                root: Some(PathBuf::from("/tmp/cache")),
            }),
            ..Default::default()
        };

        let resolved =
            resolve_typed_artifact_policy(options.clone(), |_| unreachable!("resolver unused"))
                .expect("allow-fetch policy should remain unchanged");

        assert_eq!(resolved, options);
    }

    #[test]
    fn register_overwrites_prior_descriptor() {
        let mut catalog = Catalog::new();

        let first = stub_descriptor("bundle");
        let second = BundleDescriptor {
            display_name: "Bundle v2".into(),
            ..stub_descriptor("bundle")
        };

        assert!(catalog.register_descriptor(first.clone()).is_none());

        let replaced = catalog.register_descriptor(second.clone());

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
                source_label: "primary",
                control_name: "qwen3_4b",
                format: CheckpointFormat::Safetensors,
                source: ArtifactSource::HuggingFace {
                    repo: "Qwen/Qwen3-4B",
                },
                include: vec![ArtifactRule::Exact("config.json")],
                quantization: None,
                provenance: ArtifactProvenance::unknown(),
                extra_sources: Vec::new(),
                derived: Vec::new(),
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
                source_label: "primary",
                control_name: "qwen3_4b_gguf",
                format: CheckpointFormat::Gguf,
                source: ArtifactSource::HuggingFace {
                    repo: "Qwen/Qwen3-4B-GGUF",
                },
                include: vec![ArtifactRule::Suffix(".gguf")],
                quantization: None,
                provenance: ArtifactProvenance::unknown(),
                extra_sources: Vec::new(),
                derived: Vec::new(),
            }),
            ..mistral.clone()
        };
        catalog.register_descriptor(mistral.clone());
        catalog.register_descriptor(llama.clone());
        catalog.register_model_variant(
            ModelIdentity {
                id: BundleId::new("qwen3_4b"),
                display_name: "Qwen3 4B".into(),
                family: BundleFamily::Qwen,
                capabilities: Capabilities::chat_and_completion(),
                eval_tracks: vec![EvalTrack::Chat],
                requirements: BundleRequirements::default(),
            },
            ModelVariantDescriptor {
                backend: BackendKind::MistralRs,
                capabilities: Capabilities::chat_and_completion(),
                quantization: QuantizationSupport::none(),
                checkpoint: mistral
                    .checkpoint()
                    .expect("mistral descriptor should expose checkpoint"),
            },
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
            ModelVariantDescriptor {
                backend: BackendKind::LlamaCpp,
                capabilities: Capabilities::chat_and_completion(),
                quantization: QuantizationSupport::with_recommended(
                    [motlie_model::QuantizationScheme::GgufQ4_K_M],
                    motlie_model::QuantizationScheme::GgufQ4_K_M,
                )
                .expect("test quantization support should be valid"),
                checkpoint: llama
                    .checkpoint()
                    .expect("llama descriptor should expose checkpoint"),
            },
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
        #[cfg(feature = "model-qwen3-4b-gguf")]
        assert!(catalog.instantiate_resolved(&resolved).is_some());
        #[cfg(not(feature = "model-qwen3-4b-gguf"))]
        assert!(catalog.instantiate_resolved(&resolved).is_none());
    }

    #[test]
    fn defaults_include_curated_embedding_bundles_and_artifact_control() {
        let catalog = Catalog::with_defaults();

        #[cfg(feature = "model-google-gemma-300m")]
        {
            let bundle_id = BundleId::new("embeddinggemma_300m");
            assert!(!catalog.is_empty());
            assert!(catalog.instantiate(&bundle_id).is_some());
            assert!(catalog
                .bundles_for_track(EvalTrack::Embeddings)
                .any(|bundle| bundle.id == bundle_id));

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
            assert!(catalog
                .bundles_for_track(EvalTrack::Embeddings)
                .any(|bundle| bundle.id == bundle_id));

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

        #[cfg(feature = "model-qwen3-6-27b-gguf")]
        {
            let bundle_id = BundleId::new("qwen3_6_27b_gguf");
            assert!(catalog.instantiate(&bundle_id).is_some());
            assert!(catalog
                .bundles_for_track(EvalTrack::Chat)
                .any(|bundle| bundle.id == bundle_id));

            let artifacts = catalog
                .artifacts(&bundle_id)
                .expect("Qwen3.6 GGUF bundle should expose artifact control");
            assert_eq!(artifacts.control_name, "qwen3_6_27b_gguf");
            assert!(artifacts.includes("Qwen3.6-27B-Q5_K_M.gguf"));
            assert!(!artifacts.includes("Qwen3.6-27B-FP8.gguf"));
        }

        #[cfg(feature = "model-ornith-1-0-35b-gguf")]
        {
            let bundle_id = BundleId::new("ornith_1_0_35b_gguf");
            assert!(catalog.instantiate(&bundle_id).is_some());
            assert!(catalog
                .bundles_for_track(EvalTrack::Chat)
                .any(|bundle| bundle.id == bundle_id));

            let artifacts = catalog
                .artifacts(&bundle_id)
                .expect("Ornith GGUF bundle should expose artifact control");
            assert_eq!(artifacts.control_name, "ornith_1_0_35b_gguf");
            assert!(artifacts.includes("ornith-1.0-35b-Q4_K_M.gguf"));
            assert!(artifacts.includes("ornith-1.0-35b-Q8_0.gguf"));
        }
    }

    #[test]
    fn embedding_models_round_trip_string_selectors() {
        #[cfg(feature = "model-google-gemma-300m")]
        {
            let model: EmbeddingModels = "google/embeddinggemma_300m"
                .parse()
                .expect("known embedding selector should parse");

            assert_eq!(model, EmbeddingModels::EmbeddingGemma300m);
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
                ModelSelector::Embedding(EmbeddingModels::EmbeddingGemma300m)
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
                EmbeddingModels::EmbeddingGemma300m
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

        #[cfg(feature = "model-qwen3-6-27b-gguf")]
        {
            let model: ChatModels = "qwen/qwen3_6_27b_gguf"
                .parse()
                .expect("known Qwen3.6 GGUF chat selector should parse");

            assert_eq!(model, ChatModels::Qwen3_6_27B_Gguf);
            assert_eq!(model.to_string(), "qwen/qwen3_6_27b_gguf");
        }

        #[cfg(feature = "model-ornith-1-0-35b-gguf")]
        {
            let model: ChatModels = "deepreinforce-ai/ornith_1_0_35b_gguf"
                .parse()
                .expect("known Ornith GGUF chat selector should parse");

            assert_eq!(model, ChatModels::Ornith_1_0_35B_Gguf);
            assert_eq!(model.to_string(), "deepreinforce-ai/ornith_1_0_35b_gguf");
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

        #[cfg(feature = "model-qwen3-6-27b-gguf")]
        {
            let selector: ModelSelector = "chat:qwen/qwen3_6_27b_gguf"
                .parse()
                .expect("known Qwen3.6 GGUF chat model selector should parse");

            assert_eq!(selector, ModelSelector::Chat(ChatModels::Qwen3_6_27B_Gguf));
            assert_eq!(selector.to_string(), "chat:qwen/qwen3_6_27b_gguf");
        }

        #[cfg(feature = "model-ornith-1-0-35b-gguf")]
        {
            let selector: ModelSelector = "chat:deepreinforce-ai/ornith_1_0_35b_gguf"
                .parse()
                .expect("known Ornith GGUF chat model selector should parse");

            assert_eq!(
                selector,
                ModelSelector::Chat(ChatModels::Ornith_1_0_35B_Gguf)
            );
            assert_eq!(
                selector.to_string(),
                "chat:deepreinforce-ai/ornith_1_0_35b_gguf"
            );
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

    #[cfg(not(feature = "model-qwen3-6-27b-gguf"))]
    #[test]
    fn qwen36_gguf_chat_selector_reports_unavailable_when_disabled() {
        let err = "chat:qwen/qwen3_6_27b_gguf"
            .parse::<ModelSelector>()
            .expect_err("disabled Qwen3.6 GGUF selector should be unavailable");

        assert!(matches!(
            err,
            ModelsError::ModelUnavailable { selector }
            if selector == "chat:qwen/qwen3_6_27b_gguf"
        ));
    }

    #[cfg(not(feature = "model-ornith-1-0-35b-gguf"))]
    #[test]
    fn ornith_gguf_chat_selector_reports_unavailable_when_disabled() {
        let err = "chat:deepreinforce-ai/ornith_1_0_35b_gguf"
            .parse::<ModelSelector>()
            .expect_err("disabled Ornith GGUF selector should be unavailable");

        assert!(matches!(
            err,
            ModelsError::ModelUnavailable { selector }
            if selector == "chat:deepreinforce-ai/ornith_1_0_35b_gguf"
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

        #[cfg(feature = "model-kokoro-82m")]
        {
            let model: TtsModels = "kokoro/kokoro_82m"
                .parse()
                .expect("known TTS selector should parse");

            assert_eq!(model, TtsModels::Kokoro82m);
            assert_eq!(model.to_string(), "kokoro/kokoro_82m");
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

        #[cfg(feature = "model-kokoro-82m")]
        {
            let selector: ModelSelector = "tts:kokoro/kokoro_82m"
                .parse()
                .expect("known TTS selector should parse");

            assert_eq!(selector, ModelSelector::Tts(TtsModels::Kokoro82m));
            assert_eq!(selector.to_string(), "tts:kokoro/kokoro_82m");
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

    #[cfg(any(
        feature = "model-whisper-base-en",
        feature = "model-moonshine-streaming",
        feature = "model-sherpa-onnx-streaming",
        feature = "model-piper-en-us-ljspeech-medium",
        feature = "model-kokoro-82m",
        feature = "model-qwen3-tts-cpp",
    ))]
    #[test]
    fn curated_speech_bundle_metadata_matches_execution_contracts() {
        #[cfg(feature = "model-whisper-base-en")]
        {
            let descriptor = crate::asr::whisper_base_en::descriptor();
            assert!(descriptor
                .capabilities
                .supports(CapabilityKind::Transcription));
            assert!(!descriptor.capabilities.supports(CapabilityKind::VoiceClone));
            assert_eq!(
                descriptor.capabilities.descriptors(),
                &[CapabilityDescriptor::transcription_batch()]
            );
        }

        #[cfg(feature = "model-moonshine-streaming")]
        {
            let descriptor = crate::asr::moonshine_streaming_en::descriptor();
            assert_eq!(
                descriptor.capabilities.descriptors(),
                &[CapabilityDescriptor::transcription_stream_partial()]
            );
        }

        #[cfg(feature = "model-sherpa-onnx-streaming")]
        {
            let descriptor = crate::asr::sherpa_onnx_streaming_zipformer_en::descriptor();
            assert_eq!(
                descriptor.capabilities.descriptors(),
                &[CapabilityDescriptor::transcription_stream_partial()]
            );
        }

        #[cfg(feature = "model-piper-en-us-ljspeech-medium")]
        {
            let descriptor = crate::tts::piper_en_us_ljspeech_medium::descriptor();
            assert_eq!(
                descriptor.capabilities.descriptors(),
                &[CapabilityDescriptor::speech_buffered()]
            );
        }

        #[cfg(feature = "model-kokoro-82m")]
        {
            let descriptor = crate::tts::kokoro_82m::descriptor();
            assert_eq!(
                descriptor.capabilities.descriptors(),
                &[CapabilityDescriptor::speech_buffered_and_streaming()]
            );
        }

        #[cfg(feature = "model-qwen3-tts-cpp")]
        {
            let descriptor = crate::tts::qwen3_tts_cpp_0_6b::descriptor();
            assert_eq!(
                descriptor.capabilities.descriptors(),
                &[
                    CapabilityDescriptor::speech_buffered(),
                    CapabilityDescriptor::voice_clone(),
                ]
            );
        }
    }

    #[test]
    fn artifact_rules_match_expected_files() {
        let artifacts = BundleArtifacts {
            source_label: "primary",
            control_name: "embeddinggemma_300m",
            format: CheckpointFormat::Safetensors,
            source: ArtifactSource::HuggingFace {
                repo: "google/embeddinggemma-300m",
            },
            include: vec![
                ArtifactRule::Exact("config.json"),
                ArtifactRule::Suffix(".safetensors"),
            ],
            quantization: None,
            provenance: ArtifactProvenance::unknown(),
            extra_sources: Vec::new(),
            derived: Vec::new(),
        };

        assert!(artifacts.includes("config.json"));
        assert!(artifacts.includes("weights-00001.safetensors"));
        assert!(!artifacts.includes("README.md"));
    }

    #[test]
    fn gguf_artifact_rules_filter_to_requested_quantization() {
        let artifacts = BundleArtifacts {
            source_label: "primary",
            control_name: "qwen3_6_27b_gguf",
            format: CheckpointFormat::Gguf,
            source: ArtifactSource::HuggingFace {
                repo: "unsloth/Qwen3.6-27B-GGUF",
            },
            include: vec![
                ArtifactRule::Exact("Qwen3.6-27B-Q4_K_M.gguf"),
                ArtifactRule::Exact("Qwen3.6-27B-Q5_K_M.gguf"),
                ArtifactRule::Exact("Qwen3.6-27B-Q8_0.gguf"),
            ],
            quantization: None,
            provenance: ArtifactProvenance::unknown(),
            extra_sources: Vec::new(),
            derived: Vec::new(),
        };

        assert_eq!(
            artifacts.include_for_quantization(Some(QuantizationScheme::GgufQ4_K_M)),
            vec![ArtifactRule::Exact("Qwen3.6-27B-Q4_K_M.gguf")]
        );
        assert_eq!(
            artifacts.include_for_quantization(Some(QuantizationScheme::GgufQ5_K_M)),
            vec![ArtifactRule::Exact("Qwen3.6-27B-Q5_K_M.gguf")]
        );
        assert_eq!(artifacts.include_for_quantization(None), artifacts.include);
    }
}
