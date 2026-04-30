//! Core contracts for curated model bundles in the Motlie ecosystem.

use std::collections::BTreeSet;
use std::fmt;
use std::path::PathBuf;

use async_trait::async_trait;
use thiserror::Error;

pub mod chat;
pub mod embedding;
pub mod eval;
pub mod generation;
pub mod metrics;
#[cfg(feature = "metrics-runtime")]
pub mod metrics_runtime;
pub mod speech;
pub mod transcription;
pub mod typed;
pub mod units;

pub use chat::{ChatMessage, ChatRole, ContentPart};
pub use embedding::{Embedding, EmbeddingDistance, EmbeddingNormalization, EmbeddingSpec};
pub use eval::EvalTrack;
pub use generation::{
    ChatRequest, ChatResponse, CompletionRequest, CompletionResponse, GenerationParams,
};
pub use metrics::{EmbeddingMetrics, ModelMetricSnapshot, RuntimeMetrics, TextGenerationMetrics};
pub use speech::SpeechParams;
pub use transcription::{TranscriptSegment, TranscriptionParams, TranscriptionUpdate};
pub use typed::{
    AudioBuf, AudioTransform, BatchTranscriber, BufferedSpeechChunkStream,
    BufferedSpeechSynthesizer, BufferedVoiceCloneSynthesizer, CloneReference, Compose,
    I16MonoResampler, I16ToF32, IdentityTransform, Mono, SpeechStream as TypedSpeechStream,
    SpeechSynthesizer as TypedSpeechSynthesizer, Stereo, StreamingTranscriber, SynthesisRequest,
    TranscriptionSession, VoiceCloneSynthesizer, stream_speech_into_asr,
};
pub use units::{Bytes, Milliseconds, Tokens, TokensPerSecond};

/// Stable product-facing identifier for a curated bundle.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BundleId(String);

impl BundleId {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for BundleId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

impl From<String> for BundleId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl fmt::Display for BundleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// Organizational family for related curated bundles.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BundleFamily {
    Embeddings,
    Gemma,
    Gpt,
    Hermes,
    Other(String),
    Piper,
    Qwen,
    Whisper,
}

/// Internal execution substrate chosen for a bundle or adapter.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackendKind {
    Http,
    LlamaCpp,
    MistralRs,
    Ort,
    Qwen3TtsCpp,
    SherpaOnnx,
    WhisperCpp,
}

/// Platform scoping visible to operators and release tooling.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PlatformConstraint {
    Linux,
    Macos,
    Distribution(String),
    Architecture(String),
}

/// Build-time constraints kept close to the model declaration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BuildConstraint {
    CpuOnly,
    CudaRequired,
    Feature(String),
    Profile(String),
}

/// Requirements that affect whether and how a model may be loaded or built.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BundleRequirements {
    pub platform: Vec<PlatformConstraint>,
    pub build: Vec<BuildConstraint>,
}

/// Artifact source backing a curated checkpoint declaration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArtifactSource {
    HuggingFace { repo: &'static str },
}

/// Rule used to include concrete artifacts from a source.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArtifactRule {
    Exact(&'static str),
    Suffix(&'static str),
}

impl ArtifactRule {
    pub fn matches(&self, filename: &str) -> bool {
        match self {
            Self::Exact(expected) => filename == *expected,
            Self::Suffix(suffix) => filename.ends_with(suffix),
        }
    }
}

/// High-level capabilities that a loaded bundle may expose.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CapabilityKind {
    Chat,
    Completion,
    Embeddings,
    Ocr,
    Speech,
    Transcription,
    Vision,
    VoiceClone,
}

/// Normalized content kinds used in capability introspection.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ContentKind {
    Audio,
    EmbeddingVector,
    Image,
    StructuredJson,
    Text,
}

/// Interaction pattern expected by a capability.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum InteractionStyle {
    Batch,
    MultiTurn,
    RequestResponse,
    Streaming,
}

/// Introspective description of a supported capability.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CapabilityDescriptor {
    pub kind: CapabilityKind,
    pub summary: &'static str,
    pub inputs: Vec<ContentKind>,
    pub outputs: Vec<ContentKind>,
    pub interaction: InteractionStyle,
}

impl CapabilityDescriptor {
    pub fn new(
        kind: CapabilityKind,
        summary: &'static str,
        inputs: Vec<ContentKind>,
        outputs: Vec<ContentKind>,
        interaction: InteractionStyle,
    ) -> Self {
        Self {
            kind,
            summary,
            inputs,
            outputs,
            interaction,
        }
    }

    pub fn chat() -> Self {
        Self::new(
            CapabilityKind::Chat,
            "Multi-turn text interaction with text output.",
            vec![ContentKind::Text],
            vec![ContentKind::Text],
            InteractionStyle::MultiTurn,
        )
    }

    pub fn completion() -> Self {
        Self::new(
            CapabilityKind::Completion,
            "Single-prompt text completion.",
            vec![ContentKind::Text],
            vec![ContentKind::Text],
            InteractionStyle::RequestResponse,
        )
    }

    pub fn embeddings() -> Self {
        Self::new(
            CapabilityKind::Embeddings,
            "Text input mapped to embedding vectors.",
            vec![ContentKind::Text],
            vec![ContentKind::EmbeddingVector],
            InteractionStyle::Batch,
        )
    }

    pub fn multimodal_chat() -> Self {
        Self::new(
            CapabilityKind::Chat,
            "Multi-turn interaction with text and image input, text output.",
            vec![ContentKind::Text, ContentKind::Image],
            vec![ContentKind::Text],
            InteractionStyle::MultiTurn,
        )
    }

    pub fn vision() -> Self {
        Self::new(
            CapabilityKind::Vision,
            "Image content parts accepted on the chat surface.",
            vec![ContentKind::Image, ContentKind::Text],
            vec![ContentKind::Text],
            InteractionStyle::MultiTurn,
        )
    }

    pub fn transcription_batch() -> Self {
        Self::new(
            CapabilityKind::Transcription,
            "Batch voice-to-text transcription from complete audio input.",
            vec![ContentKind::Audio],
            vec![ContentKind::Text],
            InteractionStyle::Batch,
        )
    }

    pub fn transcription_stream_final_only() -> Self {
        Self::new(
            CapabilityKind::Transcription,
            "Streaming voice-to-text transcription from PCM audio chunks with final transcript delivery on session completion.",
            vec![ContentKind::Audio],
            vec![ContentKind::Text],
            InteractionStyle::Streaming,
        )
    }

    pub fn transcription_stream_partial() -> Self {
        Self::new(
            CapabilityKind::Transcription,
            "Streaming voice-to-text transcription from PCM audio chunks with partial transcript updates.",
            vec![ContentKind::Audio],
            vec![ContentKind::Text],
            InteractionStyle::Streaming,
        )
    }

    pub fn transcription_stream() -> Self {
        Self::transcription_stream_partial()
    }

    pub fn speech_buffered() -> Self {
        Self::new(
            CapabilityKind::Speech,
            "Buffered text-to-speech synthesis that returns full audio before chunked consumption.",
            vec![ContentKind::Text],
            vec![ContentKind::Audio],
            InteractionStyle::RequestResponse,
        )
    }

    pub fn speech_stream() -> Self {
        Self::new(
            CapabilityKind::Speech,
            "Streaming text-to-speech synthesis with incremental PCM audio output.",
            vec![ContentKind::Text],
            vec![ContentKind::Audio],
            InteractionStyle::Streaming,
        )
    }

    pub fn voice_clone() -> Self {
        Self::new(
            CapabilityKind::VoiceClone,
            "Reference-conditioned voice cloning on the speech synthesis surface.",
            vec![ContentKind::Text, ContentKind::Audio],
            vec![ContentKind::Audio],
            InteractionStyle::RequestResponse,
        )
    }
}

/// Supported capability set plus introspective metadata.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Capabilities {
    descriptors: Vec<CapabilityDescriptor>,
    kinds: BTreeSet<CapabilityKind>,
}

impl Capabilities {
    pub fn new(descriptors: Vec<CapabilityDescriptor>) -> Self {
        let mut descriptors_by_kind = BTreeSet::new();
        let mut unique_descriptors = Vec::new();

        for descriptor in descriptors {
            if descriptors_by_kind.insert(descriptor.kind) {
                unique_descriptors.push(descriptor);
            }
        }

        Self {
            descriptors: unique_descriptors,
            kinds: descriptors_by_kind,
        }
    }

    pub fn descriptors(&self) -> &[CapabilityDescriptor] {
        &self.descriptors
    }

    pub fn supports(&self, capability: CapabilityKind) -> bool {
        self.kinds.contains(&capability)
    }

    pub fn chat_only() -> Self {
        Self::new(vec![CapabilityDescriptor::chat()])
    }

    pub fn completion_only() -> Self {
        Self::new(vec![CapabilityDescriptor::completion()])
    }

    pub fn embeddings_only() -> Self {
        Self::new(vec![CapabilityDescriptor::embeddings()])
    }

    pub fn chat_and_completion() -> Self {
        Self::new(vec![
            CapabilityDescriptor::chat(),
            CapabilityDescriptor::completion(),
        ])
    }

    pub fn multimodal_chat_and_vision() -> Self {
        Self::new(vec![
            CapabilityDescriptor::multimodal_chat(),
            CapabilityDescriptor::vision(),
        ])
    }

    pub fn transcription_batch_only() -> Self {
        Self::new(vec![CapabilityDescriptor::transcription_batch()])
    }

    pub fn transcription_stream_final_only() -> Self {
        Self::new(vec![CapabilityDescriptor::transcription_stream_final_only()])
    }

    pub fn transcription_stream_partial_only() -> Self {
        Self::new(vec![CapabilityDescriptor::transcription_stream_partial()])
    }

    pub fn transcription_stream_only() -> Self {
        Self::transcription_stream_partial_only()
    }

    pub fn speech_buffered_only() -> Self {
        Self::new(vec![CapabilityDescriptor::speech_buffered()])
    }

    pub fn speech_buffered_with_voice_clone() -> Self {
        Self::new(vec![
            CapabilityDescriptor::speech_buffered(),
            CapabilityDescriptor::voice_clone(),
        ])
    }

    pub fn speech_stream_only() -> Self {
        Self::new(vec![CapabilityDescriptor::speech_stream()])
    }
}

/// Stable metadata for a curated bundle definition or a loaded bundle instance.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BundleMetadata {
    pub id: BundleId,
    pub display_name: String,
    pub capabilities: Capabilities,
    pub quantization: QuantizationSupport,
}

/// Stable metadata for a logical model independent of checkpoint format or backend.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelIdentity {
    pub id: BundleId,
    pub display_name: String,
    pub family: BundleFamily,
    pub capabilities: Capabilities,
    pub eval_tracks: Vec<EvalTrack>,
    pub requirements: BundleRequirements,
}

/// Physical checkpoint format used to package model weights.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CheckpointFormat {
    Safetensors,
    Gguf,
    /// Legacy ggml format used by whisper.cpp curated artifacts.
    /// The canonical first-slice artifact is `ggml-base.en.bin`.
    Ggml,
    Onnx,
}

/// Checkpoint-native quantization metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CheckpointQuantization {
    Gguf { label: String },
    Onnx { bits: u8 },
}

/// Artifact declaration for a concrete checkpoint variant of a model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ModelCheckpoint {
    pub format: CheckpointFormat,
    pub source: ArtifactSource,
    pub include: Vec<ArtifactRule>,
    pub quantization: Option<CheckpointQuantization>,
}

impl ModelCheckpoint {
    pub fn includes(&self, filename: &str) -> bool {
        self.include.iter().any(|rule| rule.matches(filename))
    }
}

/// Checkpoint resolved to a local path that a backend can open.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResolvedCheckpoint {
    pub checkpoint: ModelCheckpoint,
    pub path: PathBuf,
}

/// Artifact acquisition policy that bundle startup must honor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArtifactPolicy {
    AllowFetch { root: Option<PathBuf> },
    LocalOnly { root: PathBuf },
}

/// Backend-agnostic quantization precision for model weights.
///
/// Backends map this to their native quantization mechanism — ISQ for
/// mistral.rs, GGUF bit width for llama.cpp, etc. `None` (the default)
/// means the backend chooses its own default precision.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum QuantizationBits {
    Four,
    Five,
    Eight,
    FloatEight,
}

/// Bundle-level declaration of which quantization precisions are supported
/// and which precision is recommended for default deployments.
///
/// Invariant: `recommended` is either `None` or present in `supported`.
/// Use the constructors (`none`, `with_recommended`, `without_recommended`)
/// to enforce this.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct QuantizationSupport {
    supported: BTreeSet<QuantizationBits>,
    recommended: Option<QuantizationBits>,
}

impl QuantizationSupport {
    /// Bundle does not support any quantization.
    pub fn none() -> Self {
        Self {
            supported: BTreeSet::new(),
            recommended: None,
        }
    }

    /// Bundle supports the given precisions with a curated default.
    ///
    /// Returns `InvalidConfiguration` if `recommended` is not in `supported`.
    pub fn with_recommended(
        supported: impl IntoIterator<Item = QuantizationBits>,
        recommended: QuantizationBits,
    ) -> Result<Self, ModelError> {
        let supported: BTreeSet<_> = supported.into_iter().collect();
        if !supported.contains(&recommended) {
            return Err(ModelError::InvalidConfiguration(format!(
                "recommended quantization {recommended:?} must be in supported set {supported:?}"
            )));
        }
        Ok(Self {
            supported,
            recommended: Some(recommended),
        })
    }

    /// Bundle supports the given precisions with no curated default (F32 by default).
    pub fn without_recommended(supported: impl IntoIterator<Item = QuantizationBits>) -> Self {
        Self {
            supported: supported.into_iter().collect(),
            recommended: None,
        }
    }

    pub fn supported(&self) -> &BTreeSet<QuantizationBits> {
        &self.supported
    }

    pub fn recommended(&self) -> Option<QuantizationBits> {
        self.recommended
    }

    pub fn supports(&self, bits: QuantizationBits) -> bool {
        self.supported.contains(&bits)
    }

    /// Resolve the caller's quantization request against this bundle's support.
    ///
    /// - `Some(bits)` where bits is supported → `Ok(Some(bits))`
    /// - `Some(bits)` where bits is NOT supported → `Err(InvalidConfiguration)`
    /// - `None` → `Ok(self.recommended)` (curated default or None for F32)
    pub fn resolve(
        &self,
        requested: Option<QuantizationBits>,
        bundle_id: &BundleId,
    ) -> Result<Option<QuantizationBits>, ModelError> {
        match requested {
            Some(bits) if !self.supports(bits) => Err(ModelError::InvalidConfiguration(format!(
                "bundle `{bundle_id}` does not support {bits:?} quantization; supported: {:?}",
                self.supported
            ))),
            Some(bits) => Ok(Some(bits)),
            None => Ok(self.recommended),
        }
    }
}

/// Deployment-oriented knobs used when starting a bundle.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StartOptions {
    pub artifact_policy: Option<ArtifactPolicy>,
    pub quantization: Option<QuantizationBits>,
    pub unpack_root: Option<PathBuf>,
    pub max_concurrency: Option<usize>,
}

/// Runtime-resolved descriptor for a loaded bundle instance.
///
/// Carries the curated metadata plus the quantization precision that was
/// actually applied at startup (which may differ from the bundle's
/// `recommended` default).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoadedBundleDescriptor {
    pub id: BundleId,
    pub display_name: String,
    pub capabilities: Capabilities,
    pub quantization: QuantizationSupport,
    /// The quantization precision actually applied at startup, or `None` for F32.
    pub resolved_quantization: Option<QuantizationBits>,
}

/// Structured error surface for core bundle operations.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ModelError {
    #[error("internal model error: {0}")]
    Internal(String),
    #[error("invalid model configuration: {0}")]
    InvalidConfiguration(String),
    #[error("failed to initialize `{backend}` backend: {message}")]
    BackendInitialization {
        backend: &'static str,
        message: String,
    },
    #[error("`{backend}` backend failed during `{operation}`: {message}")]
    BackendExecution {
        backend: &'static str,
        operation: &'static str,
        message: String,
    },
    #[error("unsupported capability: {0:?}")]
    UnsupportedCapability(CapabilityKind),
}

/// Request for embedding generation.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EmbeddingRequest {
    pub inputs: Vec<String>,
}

/// Response from embedding generation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EmbeddingResponse {
    pub vectors: Vec<Vec<f32>>,
}

/// Bundle definition that can be started into a loaded handle.
#[async_trait]
pub trait ModelBundle: Send + Sync {
    type Handle: BundleHandle;

    fn id(&self) -> &BundleId;
    fn metadata(&self) -> &BundleMetadata;
    fn capabilities(&self) -> &Capabilities;
    async fn start(&self, options: StartOptions) -> Result<Self::Handle, ModelError>;
}

/// Loaded bundle state that exposes capability adapters.
#[async_trait]
pub trait BundleHandle: Send + Sync + Sized {
    type Chat: ChatModel;
    type Completion: CompletionModel;
    type Embeddings: EmbeddingModel;

    fn descriptor(&self) -> &LoadedBundleDescriptor;
    fn capabilities(&self) -> &Capabilities;
    fn supports(&self, capability: CapabilityKind) -> bool {
        self.capabilities().supports(capability)
    }
    fn metric_snapshot(&self) -> Option<ModelMetricSnapshot> {
        None
    }

    fn chat(&self) -> Result<&Self::Chat, ModelError>;
    fn completion(&self) -> Result<&Self::Completion, ModelError>;
    fn embeddings(&self) -> Result<&Self::Embeddings, ModelError>;
    async fn shutdown(self) -> Result<(), ModelError>;
}

/// Backend-specific loader for one or more checkpoint formats.
#[async_trait]
pub trait BackendAdapter: Send + Sync {
    type Handle: BundleHandle;

    fn supported_formats(&self) -> &[CheckpointFormat];
    fn backend_kind(&self) -> BackendKind;
    fn capabilities(&self) -> &Capabilities;
    fn quantization(&self) -> &QuantizationSupport;
    async fn start(
        &self,
        identity: &ModelIdentity,
        checkpoint: &ResolvedCheckpoint,
        options: StartOptions,
    ) -> Result<Self::Handle, ModelError>;
}

/// Chat generation capability.
#[async_trait]
pub trait ChatModel: Send + Sync {
    async fn generate(&self, request: ChatRequest) -> Result<ChatResponse, ModelError>;
}

/// Text completion capability.
#[async_trait]
pub trait CompletionModel: Send + Sync {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ModelError>;
}

/// Embedding generation capability.
#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, ModelError>;
}

/// Marker model used when a bundle does not support chat generation.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnsupportedChat;

#[async_trait]
impl ChatModel for UnsupportedChat {
    async fn generate(&self, _request: ChatRequest) -> Result<ChatResponse, ModelError> {
        Err(ModelError::UnsupportedCapability(CapabilityKind::Chat))
    }
}

/// Marker model used when a bundle does not support text completion.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnsupportedCompletion;

#[async_trait]
impl CompletionModel for UnsupportedCompletion {
    async fn complete(
        &self,
        _request: CompletionRequest,
    ) -> Result<CompletionResponse, ModelError> {
        Err(ModelError::UnsupportedCapability(
            CapabilityKind::Completion,
        ))
    }
}

/// Marker model used when a bundle does not support embeddings.
#[derive(Clone, Copy, Debug, Default)]
pub struct UnsupportedEmbeddings;

#[async_trait]
impl EmbeddingModel for UnsupportedEmbeddings {
    async fn embed(&self, _request: EmbeddingRequest) -> Result<EmbeddingResponse, ModelError> {
        Err(ModelError::UnsupportedCapability(
            CapabilityKind::Embeddings,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeHandle {
        descriptor: LoadedBundleDescriptor,
    }

    #[async_trait]
    impl BundleHandle for FakeHandle {
        type Chat = UnsupportedChat;
        type Completion = UnsupportedCompletion;
        type Embeddings = Self;

        fn descriptor(&self) -> &LoadedBundleDescriptor {
            &self.descriptor
        }

        fn capabilities(&self) -> &Capabilities {
            &self.descriptor.capabilities
        }

        fn chat(&self) -> Result<&Self::Chat, ModelError> {
            Err(ModelError::UnsupportedCapability(CapabilityKind::Chat))
        }

        fn completion(&self) -> Result<&Self::Completion, ModelError> {
            Err(ModelError::UnsupportedCapability(
                CapabilityKind::Completion,
            ))
        }

        fn embeddings(&self) -> Result<&Self::Embeddings, ModelError> {
            Ok(self)
        }

        async fn shutdown(self) -> Result<(), ModelError> {
            Ok(())
        }
    }

    #[async_trait]
    impl EmbeddingModel for FakeHandle {
        async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse, ModelError> {
            Ok(EmbeddingResponse {
                vectors: request
                    .inputs
                    .into_iter()
                    .map(|input| vec![input.len() as f32])
                    .collect(),
            })
        }
    }

    #[test]
    fn bundle_id_display_and_ordering_are_stable() {
        let alpha = BundleId::new("alpha");
        let beta = BundleId::from("beta");

        assert_eq!(alpha.to_string(), "alpha");
        assert_eq!(beta.as_str(), "beta");
        assert!(alpha < beta);
    }

    #[test]
    fn capabilities_supports_descriptor_kinds() {
        let capabilities = Capabilities::new(vec![
            CapabilityDescriptor::chat(),
            CapabilityDescriptor::embeddings(),
        ]);

        assert!(capabilities.supports(CapabilityKind::Chat));
        assert!(capabilities.supports(CapabilityKind::Embeddings));
        assert!(!capabilities.supports(CapabilityKind::Completion));
        assert_eq!(capabilities.descriptors().len(), 2);
    }

    #[test]
    fn capabilities_preserve_first_descriptor_order_and_drop_duplicates() {
        let capabilities = Capabilities::new(vec![
            CapabilityDescriptor::embeddings(),
            CapabilityDescriptor::chat(),
            CapabilityDescriptor::embeddings(),
        ]);

        let kinds: Vec<_> = capabilities
            .descriptors()
            .iter()
            .map(|descriptor| descriptor.kind)
            .collect();

        assert_eq!(
            kinds,
            vec![CapabilityKind::Embeddings, CapabilityKind::Chat]
        );
        assert!(capabilities.supports(CapabilityKind::Embeddings));
        assert!(capabilities.supports(CapabilityKind::Chat));
        assert!(!capabilities.supports(CapabilityKind::Completion));
    }

    #[test]
    fn embedding_builtins_have_expected_shapes() {
        let descriptor = CapabilityDescriptor::embeddings();

        assert_eq!(descriptor.kind, CapabilityKind::Embeddings);
        assert_eq!(descriptor.inputs, vec![ContentKind::Text]);
        assert_eq!(descriptor.outputs, vec![ContentKind::EmbeddingVector]);
        assert_eq!(descriptor.interaction, InteractionStyle::Batch);
    }

    #[test]
    fn metadata_round_trips_clone_and_equality() {
        let metadata = BundleMetadata {
            id: BundleId::new("embeddinggemma_300m"),
            display_name: "EmbeddingGemma 300M".into(),
            capabilities: Capabilities::embeddings_only(),
            quantization: QuantizationSupport::none(),
        };

        assert_eq!(metadata, metadata.clone());

        let loaded = LoadedBundleDescriptor {
            id: metadata.id.clone(),
            display_name: metadata.display_name.clone(),
            capabilities: metadata.capabilities.clone(),
            quantization: metadata.quantization.clone(),
            resolved_quantization: None,
        };
        assert_eq!(loaded, loaded.clone());
    }

    #[test]
    fn artifact_rule_and_checkpoint_match_expected_filenames() {
        let checkpoint = ModelCheckpoint {
            format: CheckpointFormat::Gguf,
            source: ArtifactSource::HuggingFace {
                repo: "Qwen/Qwen3-4B-GGUF",
            },
            include: vec![
                ArtifactRule::Suffix("-Q4_K_M.gguf"),
                ArtifactRule::Suffix("-Q8_0.gguf"),
            ],
            quantization: Some(CheckpointQuantization::Gguf {
                label: "Q4_K_M".into(),
            }),
        };

        assert!(checkpoint.includes("Qwen3-4B-Q4_K_M.gguf"));
        assert!(!checkpoint.includes("config.json"));
    }

    #[test]
    fn model_identity_is_decoupled_from_checkpoint_and_backend_details() {
        let identity = ModelIdentity {
            id: BundleId::new("qwen3_4b"),
            display_name: "Qwen3 4B".into(),
            family: BundleFamily::Qwen,
            capabilities: Capabilities::chat_and_completion(),
            eval_tracks: vec![EvalTrack::Chat, EvalTrack::Reasoning],
            requirements: BundleRequirements {
                platform: vec![PlatformConstraint::Linux, PlatformConstraint::Macos],
                build: vec![BuildConstraint::Feature("backend-mistral".into())],
            },
        };

        assert_eq!(identity.id.as_str(), "qwen3_4b");
        assert_eq!(identity.family, BundleFamily::Qwen);
        assert!(identity.capabilities.supports(CapabilityKind::Chat));
        assert!(identity.eval_tracks.contains(&EvalTrack::Reasoning));
    }

    #[test]
    fn artifact_policy_is_part_of_start_options() {
        let options = StartOptions {
            artifact_policy: Some(ArtifactPolicy::LocalOnly {
                root: PathBuf::from("/tmp/models"),
            }),
            quantization: None,
            unpack_root: None,
            max_concurrency: Some(4),
        };

        assert_eq!(
            options.artifact_policy,
            Some(ArtifactPolicy::LocalOnly {
                root: PathBuf::from("/tmp/models"),
            })
        );
    }

    #[test]
    fn start_options_carry_quantization_policy() {
        let q4 = StartOptions {
            quantization: Some(QuantizationBits::Four),
            ..Default::default()
        };
        let q8 = StartOptions {
            quantization: Some(QuantizationBits::Eight),
            ..Default::default()
        };
        let none = StartOptions::default();

        assert_eq!(q4.quantization, Some(QuantizationBits::Four));
        assert_eq!(q8.quantization, Some(QuantizationBits::Eight));
        assert_eq!(none.quantization, None);
    }

    #[test]
    fn quantization_support_resolves_against_bundle_metadata() {
        let bundle_id = BundleId::new("test_bundle");

        let no_support = QuantizationSupport::none();
        assert!(
            no_support
                .resolve(Some(QuantizationBits::Four), &bundle_id)
                .is_err()
        );
        assert_eq!(no_support.resolve(None, &bundle_id).unwrap(), None);

        let q4_q8 = QuantizationSupport::with_recommended(
            [QuantizationBits::Four, QuantizationBits::Eight],
            QuantizationBits::Four,
        )
        .expect("test support is valid");
        assert_eq!(
            q4_q8
                .resolve(Some(QuantizationBits::Four), &bundle_id)
                .unwrap(),
            Some(QuantizationBits::Four)
        );
        assert_eq!(
            q4_q8
                .resolve(Some(QuantizationBits::Eight), &bundle_id)
                .unwrap(),
            Some(QuantizationBits::Eight)
        );
        assert_eq!(
            q4_q8.resolve(None, &bundle_id).unwrap(),
            Some(QuantizationBits::Four)
        );

        let q8_only = QuantizationSupport::without_recommended([QuantizationBits::Eight]);
        assert!(
            q8_only
                .resolve(Some(QuantizationBits::Four), &bundle_id)
                .is_err()
        );
        assert_eq!(
            q8_only
                .resolve(Some(QuantizationBits::Eight), &bundle_id)
                .unwrap(),
            Some(QuantizationBits::Eight)
        );
        assert_eq!(q8_only.resolve(None, &bundle_id).unwrap(), None);
    }

    #[test]
    fn quantization_support_rejects_contradictory_recommended() {
        let err = QuantizationSupport::with_recommended([], QuantizationBits::Four)
            .expect_err("contradictory recommended should fail");

        assert!(
            matches!(err, ModelError::InvalidConfiguration(msg) if msg.contains("must be in supported set"))
        );
    }

    #[tokio::test]
    async fn embedding_only_handle_reports_supported_surfaces() {
        let handle = FakeHandle {
            descriptor: LoadedBundleDescriptor {
                id: BundleId::new("embedder"),
                display_name: "Embedder".into(),
                capabilities: Capabilities::embeddings_only(),
                quantization: QuantizationSupport::none(),
                resolved_quantization: None,
            },
        };

        assert!(handle.supports(CapabilityKind::Embeddings));
        assert!(!handle.supports(CapabilityKind::Chat));
        assert!(matches!(
            handle.chat(),
            Err(ModelError::UnsupportedCapability(CapabilityKind::Chat))
        ));
        assert!(matches!(
            handle.completion(),
            Err(ModelError::UnsupportedCapability(
                CapabilityKind::Completion
            ))
        ));

        let response = handle
            .embeddings()
            .expect("embedding handle should expose embeddings")
            .embed(EmbeddingRequest {
                inputs: vec!["a".into(), "abcd".into()],
            })
            .await
            .expect("fake embed should succeed");

        assert_eq!(response.vectors, vec![vec![1.0], vec![4.0]]);
    }

    #[test]
    fn request_types_preserve_expected_defaults() {
        let chat = ChatRequest::default();
        let completion = CompletionRequest::default();
        let embeddings = EmbeddingRequest::default();
        let speech = SynthesisRequest::default();
        let multi = EmbeddingRequest {
            inputs: vec!["one".into(), "two".into()],
        };
        let response = EmbeddingResponse {
            vectors: vec![vec![], vec![1.0, 2.0]],
        };

        assert!(chat.messages.is_empty());
        assert!(completion.prompt.is_empty());
        assert!(embeddings.inputs.is_empty());
        assert!(speech.text.is_empty());
        assert_eq!(speech.params, SpeechParams::default());
        assert!(chat.params.stop_sequences.is_empty());
        assert_eq!(multi.inputs.len(), 2);
        assert_eq!(response.vectors.len(), 2);
    }

    #[test]
    fn speech_buffered_descriptor_uses_text_input_and_audio_output() {
        let descriptor = CapabilityDescriptor::speech_buffered();

        assert_eq!(descriptor.kind, CapabilityKind::Speech);
        assert_eq!(descriptor.inputs, vec![ContentKind::Text]);
        assert_eq!(descriptor.outputs, vec![ContentKind::Audio]);
        assert_eq!(descriptor.interaction, InteractionStyle::RequestResponse);
    }

    #[test]
    fn speech_stream_descriptor_uses_streaming_interaction() {
        let descriptor = CapabilityDescriptor::speech_stream();

        assert_eq!(descriptor.kind, CapabilityKind::Speech);
        assert_eq!(descriptor.inputs, vec![ContentKind::Text]);
        assert_eq!(descriptor.outputs, vec![ContentKind::Audio]);
        assert_eq!(descriptor.interaction, InteractionStyle::Streaming);
    }

    #[test]
    fn transcription_batch_descriptor_uses_audio_input_and_batch_interaction() {
        let descriptor = CapabilityDescriptor::transcription_batch();

        assert_eq!(descriptor.kind, CapabilityKind::Transcription);
        assert_eq!(descriptor.inputs, vec![ContentKind::Audio]);
        assert_eq!(descriptor.outputs, vec![ContentKind::Text]);
        assert_eq!(descriptor.interaction, InteractionStyle::Batch);
    }

    #[test]
    fn transcription_stream_descriptor_uses_audio_input_and_streaming_interaction() {
        let descriptor = CapabilityDescriptor::transcription_stream_partial();

        assert_eq!(descriptor.kind, CapabilityKind::Transcription);
        assert_eq!(descriptor.inputs, vec![ContentKind::Audio]);
        assert_eq!(descriptor.outputs, vec![ContentKind::Text]);
        assert_eq!(descriptor.interaction, InteractionStyle::Streaming);
    }

    #[test]
    fn voice_clone_descriptor_is_separate_from_speech_transport() {
        let descriptor = CapabilityDescriptor::voice_clone();

        assert_eq!(descriptor.kind, CapabilityKind::VoiceClone);
        assert_eq!(
            descriptor.inputs,
            vec![ContentKind::Text, ContentKind::Audio]
        );
        assert_eq!(descriptor.outputs, vec![ContentKind::Audio]);
        assert_eq!(descriptor.interaction, InteractionStyle::RequestResponse);
    }

    #[test]
    fn transcription_batch_only_capabilities_supports_transcription_but_not_chat() {
        let capabilities = Capabilities::transcription_batch_only();

        assert!(capabilities.supports(CapabilityKind::Transcription));
        assert!(!capabilities.supports(CapabilityKind::Chat));
        assert!(!capabilities.supports(CapabilityKind::Embeddings));
    }

    #[test]
    fn transcription_stream_partial_only_capabilities_supports_transcription_but_not_chat() {
        let capabilities = Capabilities::transcription_stream_partial_only();

        assert!(capabilities.supports(CapabilityKind::Transcription));
        assert!(!capabilities.supports(CapabilityKind::Chat));
        assert!(!capabilities.supports(CapabilityKind::Embeddings));
    }

    #[test]
    fn speech_buffered_with_voice_clone_supports_both_kinds() {
        let capabilities = Capabilities::speech_buffered_with_voice_clone();

        assert!(capabilities.supports(CapabilityKind::Speech));
        assert!(capabilities.supports(CapabilityKind::VoiceClone));
        assert!(!capabilities.supports(CapabilityKind::Chat));
    }

    #[test]
    fn speech_stream_only_capabilities_supports_speech_but_not_chat() {
        let capabilities = Capabilities::speech_stream_only();

        assert!(capabilities.supports(CapabilityKind::Speech));
        assert!(!capabilities.supports(CapabilityKind::VoiceClone));
        assert!(!capabilities.supports(CapabilityKind::Chat));
        assert!(!capabilities.supports(CapabilityKind::Embeddings));
    }

    #[test]
    fn embedding_spec_can_describe_vector_semantics() {
        let spec = EmbeddingSpec {
            dimensions: Some(768),
            distance: EmbeddingDistance::Cosine,
            normalization: EmbeddingNormalization::L2,
            input: ContentKind::Text,
            output: ContentKind::EmbeddingVector,
            summary: "Normalized text embeddings for semantic similarity and retrieval.",
        };

        assert_eq!(spec.dimensions, Some(768));
        assert_eq!(spec.distance, EmbeddingDistance::Cosine);
        assert_eq!(spec.normalization, EmbeddingNormalization::L2);
    }
}
