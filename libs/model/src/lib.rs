//! Core contracts for curated model bundles in the Motlie ecosystem.

use std::collections::BTreeSet;
use std::fmt;
use std::path::PathBuf;

use async_trait::async_trait;
use thiserror::Error;

pub mod embedding;
pub mod eval;

pub use embedding::{Embedding, EmbeddingDistance, EmbeddingNormalization, EmbeddingSpec};

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

/// High-level capabilities that a loaded bundle may expose.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum CapabilityKind {
    Chat,
    Completion,
    Embeddings,
    Ocr,
    Vision,
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
}

/// Supported capability set plus introspective metadata.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Capabilities {
    descriptors: Vec<CapabilityDescriptor>,
    kinds: BTreeSet<CapabilityKind>,
}

impl Capabilities {
    pub fn new(descriptors: Vec<CapabilityDescriptor>) -> Self {
        let kinds = descriptors
            .iter()
            .map(|descriptor| descriptor.kind)
            .collect();
        Self { descriptors, kinds }
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
}

/// Stable metadata for a curated bundle definition.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BundleMetadata {
    pub id: BundleId,
    pub display_name: String,
    pub capabilities: Capabilities,
}

/// Deployment-oriented knobs used when starting a bundle.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArtifactPolicy {
    AllowFetch { root: Option<PathBuf> },
    LocalOnly { root: PathBuf },
}

/// Deployment-oriented knobs used when starting a bundle.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct StartOptions {
    pub artifact_policy: Option<ArtifactPolicy>,
    pub unpack_root: Option<PathBuf>,
    pub max_concurrency: Option<usize>,
}

/// Lightweight descriptor for a loaded bundle instance.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoadedBundleDescriptor {
    pub id: BundleId,
    pub display_name: String,
    pub capabilities: Capabilities,
}

/// Structured error surface for core bundle operations.
#[derive(Clone, Debug, Eq, Error, PartialEq)]
pub enum ModelError {
    #[error("internal model error: {0}")]
    Internal(String),
    #[error("invalid model configuration: {0}")]
    InvalidConfiguration(String),
    #[error("unsupported capability: {0:?}")]
    UnsupportedCapability(CapabilityKind),
}

/// Role labels used in chat-style requests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChatRole {
    Assistant,
    System,
    User,
}

/// Single message in a chat request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: ChatRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

/// Shared generation parameters used by text-producing capabilities.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GenerationParams {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop_sequences: Vec<String>,
}

/// Request for chat-style generation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub params: GenerationParams,
}

/// Response from a chat-style generation call.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ChatResponse {
    pub content: String,
}

/// Request for text completion.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CompletionRequest {
    pub prompt: String,
    pub params: GenerationParams,
}

/// Response from text completion.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CompletionResponse {
    pub content: String,
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
    fn id(&self) -> &BundleId;
    fn metadata(&self) -> &BundleMetadata;
    fn capabilities(&self) -> &Capabilities;
    async fn start(&self, options: StartOptions) -> Result<Box<dyn BundleHandle>, ModelError>;
}

/// Loaded bundle state that exposes capability adapters.
#[async_trait]
pub trait BundleHandle: Send + Sync {
    fn descriptor(&self) -> &LoadedBundleDescriptor;
    fn capabilities(&self) -> &Capabilities;
    fn supports(&self, capability: CapabilityKind) -> bool {
        self.capabilities().supports(capability)
    }

    fn chat(&self) -> Result<&dyn ChatModel, ModelError>;
    fn completion(&self) -> Result<&dyn CompletionModel, ModelError>;
    fn embeddings(&self) -> Result<&dyn EmbeddingModel, ModelError>;

    async fn shutdown(self: Box<Self>) -> Result<(), ModelError>;
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

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeHandle {
        descriptor: LoadedBundleDescriptor,
    }

    #[async_trait]
    impl BundleHandle for FakeHandle {
        fn descriptor(&self) -> &LoadedBundleDescriptor {
            &self.descriptor
        }

        fn capabilities(&self) -> &Capabilities {
            &self.descriptor.capabilities
        }

        fn chat(&self) -> Result<&dyn ChatModel, ModelError> {
            Err(ModelError::UnsupportedCapability(CapabilityKind::Chat))
        }

        fn completion(&self) -> Result<&dyn CompletionModel, ModelError> {
            Err(ModelError::UnsupportedCapability(
                CapabilityKind::Completion,
            ))
        }

        fn embeddings(&self) -> Result<&dyn EmbeddingModel, ModelError> {
            Ok(self)
        }

        async fn shutdown(self: Box<Self>) -> Result<(), ModelError> {
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
        };

        assert_eq!(metadata, metadata.clone());

        let loaded = LoadedBundleDescriptor {
            id: metadata.id.clone(),
            display_name: metadata.display_name.clone(),
            capabilities: metadata.capabilities.clone(),
        };
        assert_eq!(loaded, loaded.clone());
    }

    #[test]
    fn artifact_policy_is_part_of_start_options() {
        let options = StartOptions {
            artifact_policy: Some(ArtifactPolicy::LocalOnly {
                root: PathBuf::from("/tmp/models"),
            }),
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

    #[tokio::test]
    async fn embedding_only_handle_reports_supported_surfaces() {
        let handle = FakeHandle {
            descriptor: LoadedBundleDescriptor {
                id: BundleId::new("embedder"),
                display_name: "Embedder".into(),
                capabilities: Capabilities::embeddings_only(),
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
        let multi = EmbeddingRequest {
            inputs: vec!["one".into(), "two".into()],
        };
        let response = EmbeddingResponse {
            vectors: vec![vec![], vec![1.0, 2.0]],
        };

        assert!(chat.messages.is_empty());
        assert!(completion.prompt.is_empty());
        assert!(embeddings.inputs.is_empty());
        assert!(chat.params.stop_sequences.is_empty());
        assert_eq!(multi.inputs.len(), 2);
        assert_eq!(response.vectors.len(), 2);
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
