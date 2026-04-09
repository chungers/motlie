use std::path::PathBuf;

use async_trait::async_trait;
use mistralrs::core::{NormalLoaderType, StopTokens};
use mistralrs::{IsqBits, RequestBuilder, SamplingParams, TextModelBuilder};
use motlie_model::{
    ArtifactPolicy, BundleHandle, BundleId, BundleMetadata, Capabilities, CapabilityKind,
    ChatModel, ChatRequest, ChatResponse, ChatRole, CompletionModel, CompletionRequest,
    CompletionResponse, EmbeddingModel, GenerationParams, LoadedBundleDescriptor, ModelBundle,
    ModelError, QuantizationBits, StartOptions,
};

/// Text model architecture discriminant that selects the correct `mistralrs` loader path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MistralTextArch {
    Qwen3,
}

impl MistralTextArch {
    fn loader_type(self) -> NormalLoaderType {
        match self {
            Self::Qwen3 => NormalLoaderType::Qwen3,
        }
    }
}

/// Static bundle specification for a curated Mistral-backed text generation stack.
#[derive(Clone, Debug)]
pub struct MistralTextSpec {
    pub id: BundleId,
    pub display_name: &'static str,
    pub model_id: &'static str,
    pub arch: MistralTextArch,
    pub capabilities: Capabilities,
}

impl MistralTextSpec {
    pub fn qwen3_4b() -> Self {
        Self {
            id: BundleId::new("qwen3_4b"),
            display_name: "Qwen3 4B",
            model_id: "Qwen/Qwen3-4B",
            arch: MistralTextArch::Qwen3,
            capabilities: Capabilities::chat_and_completion(),
        }
    }
}

/// Generic `ModelBundle` implementation backed by `mistralrs` text generation.
#[derive(Clone, Debug)]
pub struct MistralTextBundle {
    metadata: BundleMetadata,
    arch: MistralTextArch,
    model_id: &'static str,
}

impl MistralTextBundle {
    pub fn new(spec: MistralTextSpec) -> Self {
        Self {
            metadata: BundleMetadata {
                id: spec.id,
                display_name: spec.display_name.into(),
                capabilities: spec.capabilities,
            },
            arch: spec.arch,
            model_id: spec.model_id,
        }
    }
}

#[async_trait]
impl ModelBundle for MistralTextBundle {
    fn id(&self) -> &BundleId {
        &self.metadata.id
    }

    fn metadata(&self) -> &BundleMetadata {
        &self.metadata
    }

    fn capabilities(&self) -> &Capabilities {
        &self.metadata.capabilities
    }

    async fn start(&self, options: StartOptions) -> Result<Box<dyn BundleHandle>, ModelError> {
        let model = build_text_model(self.model_id, self.arch, options).await?;

        Ok(Box::new(MistralTextHandle {
            descriptor: LoadedBundleDescriptor {
                id: self.metadata.id.clone(),
                display_name: self.metadata.display_name.clone(),
                capabilities: self.metadata.capabilities.clone(),
            },
            runtime: Box::new(MistralTextRuntime { model }),
        }))
    }
}

// ---------------------------------------------------------------------------
// Internal runtime abstraction (enables stub testing without real mistralrs)
// ---------------------------------------------------------------------------

#[async_trait]
trait TextRuntime: Send + Sync {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ModelError>;
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ModelError>;
}

struct MistralTextRuntime {
    model: mistralrs::Model,
}

#[async_trait]
impl TextRuntime for MistralTextRuntime {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ModelError> {
        let builder = to_request_builder(&request);

        let response = self
            .model
            .send_chat_request(builder)
            .await
            .map_err(|err| ModelError::BackendExecution {
                backend: "mistralrs",
                operation: "send_chat_request",
                message: err.to_string(),
            })?;

        let content = response
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .ok_or_else(|| ModelError::BackendExecution {
                backend: "mistralrs",
                operation: "send_chat_request",
                message: "response contained no text content".into(),
            })?;

        Ok(ChatResponse { content })
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ModelError> {
        let chat_request = ChatRequest {
            messages: vec![motlie_model::ChatMessage::new(
                ChatRole::User,
                request.prompt,
            )],
            params: request.params,
        };
        let chat_response = self.chat(chat_request).await?;
        Ok(CompletionResponse {
            content: chat_response.content,
        })
    }
}

fn to_request_builder(request: &ChatRequest) -> RequestBuilder {
    let mut builder = RequestBuilder::new();
    for msg in &request.messages {
        let role = match msg.role {
            ChatRole::System => mistralrs::TextMessageRole::System,
            ChatRole::User => mistralrs::TextMessageRole::User,
            ChatRole::Assistant => mistralrs::TextMessageRole::Assistant,
        };
        builder = builder.add_message(role, &msg.content);
    }
    builder = apply_generation_params(builder, &request.params);
    builder
}

fn apply_generation_params(builder: RequestBuilder, params: &GenerationParams) -> RequestBuilder {
    let mut sampling = SamplingParams::deterministic();
    if let Some(temperature) = params.temperature {
        sampling.temperature = Some(temperature as f64);
        sampling.top_k = None; // disable deterministic top_k=1 when temperature is set
    }
    if let Some(top_p) = params.top_p {
        sampling.top_p = Some(top_p as f64);
    }
    if let Some(max_tokens) = params.max_tokens {
        sampling.max_len = Some(max_tokens as usize);
    }
    if !params.stop_sequences.is_empty() {
        sampling.stop_toks = Some(StopTokens::Seqs(params.stop_sequences.clone()));
    }
    builder.set_sampling(sampling)
}

// ---------------------------------------------------------------------------
// Handle
// ---------------------------------------------------------------------------

struct MistralTextHandle {
    descriptor: LoadedBundleDescriptor,
    runtime: Box<dyn TextRuntime>,
}

#[async_trait]
impl BundleHandle for MistralTextHandle {
    fn descriptor(&self) -> &LoadedBundleDescriptor {
        &self.descriptor
    }

    fn capabilities(&self) -> &Capabilities {
        &self.descriptor.capabilities
    }

    fn chat(&self) -> Result<&dyn ChatModel, ModelError> {
        Ok(self)
    }

    fn completion(&self) -> Result<&dyn CompletionModel, ModelError> {
        Ok(self)
    }

    fn embeddings(&self) -> Result<&dyn EmbeddingModel, ModelError> {
        Err(ModelError::UnsupportedCapability(
            CapabilityKind::Embeddings,
        ))
    }

    async fn shutdown(self: Box<Self>) -> Result<(), ModelError> {
        Ok(())
    }
}

#[async_trait]
impl ChatModel for MistralTextHandle {
    async fn generate(&self, request: ChatRequest) -> Result<ChatResponse, ModelError> {
        self.runtime.chat(request).await
    }
}

#[async_trait]
impl CompletionModel for MistralTextHandle {
    async fn complete(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse, ModelError> {
        self.runtime.complete(request).await
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

async fn build_text_model(
    model_id: &str,
    arch: MistralTextArch,
    options: StartOptions,
) -> Result<mistralrs::Model, ModelError> {
    let StartOptions {
        artifact_policy,
        quantization,
        unpack_root,
        max_concurrency,
    } = options;

    if let Some(unpack_root) = unpack_root {
        return Err(ModelError::InvalidConfiguration(format!(
            "`mistralrs` text startup does not support `unpack_root` yet (got `{}`)",
            unpack_root.display()
        )));
    }

    let mut model_target = model_id.to_owned();
    let mut hf_cache_root = None;

    if let Some(artifact_policy) = artifact_policy {
        let configured = configure_artifact_policy(model_id, artifact_policy)?;
        model_target = configured.model_target;
        hf_cache_root = configured.hf_cache_root;
    }

    let mut builder = TextModelBuilder::new(model_target)
        .with_loader_type(arch.loader_type());

    if let Some(bits) = quantization {
        builder = builder.with_auto_isq(map_quantization_bits(bits));
    }
    if should_force_cpu() {
        builder = builder.with_force_cpu();
    }
    if let Some(hf_cache_root) = hf_cache_root {
        builder = builder.from_hf_cache_path(hf_cache_root);
    }
    if let Some(max_num_seqs) = max_concurrency {
        builder = builder.with_max_num_seqs(max_num_seqs);
    }

    builder
        .build()
        .await
        .map_err(|err| ModelError::BackendInitialization {
            backend: "mistralrs",
            message: err.to_string(),
        })
}

fn map_quantization_bits(bits: QuantizationBits) -> IsqBits {
    match bits {
        QuantizationBits::Four => IsqBits::Four,
        QuantizationBits::Eight => IsqBits::Eight,
    }
}

struct ConfiguredBuilder {
    model_target: String,
    hf_cache_root: Option<PathBuf>,
}

fn configure_artifact_policy(
    model_id: &str,
    policy: ArtifactPolicy,
) -> Result<ConfiguredBuilder, ModelError> {
    match policy {
        ArtifactPolicy::AllowFetch { root } => Ok(ConfiguredBuilder {
            model_target: model_id.to_owned(),
            hf_cache_root: root,
        }),
        ArtifactPolicy::LocalOnly { root } => Ok(ConfiguredBuilder {
            model_target: root.display().to_string(),
            hf_cache_root: None,
        }),
    }
}

fn should_force_cpu() -> bool {
    matches!(
        std::env::var("MOTLIE_MODEL_FORCE_CPU"),
        Ok(value) if matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use motlie_model::StartOptions;

    struct StubTextRuntime;

    #[async_trait]
    impl TextRuntime for StubTextRuntime {
        async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ModelError> {
            let prompt = request
                .messages
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default();
            Ok(ChatResponse {
                content: format!("stub response to: {prompt}"),
            })
        }

        async fn complete(
            &self,
            request: CompletionRequest,
        ) -> Result<CompletionResponse, ModelError> {
            Ok(CompletionResponse {
                content: format!("stub completion of: {}", request.prompt),
            })
        }
    }

    #[test]
    fn qwen3_spec_has_expected_identity() {
        let spec = MistralTextSpec::qwen3_4b();

        assert_eq!(spec.id.as_str(), "qwen3_4b");
        assert_eq!(spec.display_name, "Qwen3 4B");
        assert_eq!(spec.model_id, "Qwen/Qwen3-4B");
        assert_eq!(spec.arch, MistralTextArch::Qwen3);
        assert!(spec.capabilities.supports(CapabilityKind::Chat));
        assert!(spec.capabilities.supports(CapabilityKind::Completion));
        assert!(!spec.capabilities.supports(CapabilityKind::Embeddings));
    }

    #[tokio::test]
    async fn text_handle_exposes_chat_and_completion_but_not_embeddings() {
        let handle = MistralTextHandle {
            descriptor: LoadedBundleDescriptor {
                id: BundleId::new("qwen3_4b"),
                display_name: "Qwen3 4B".into(),
                capabilities: Capabilities::chat_and_completion(),
            },
            runtime: Box::new(StubTextRuntime),
        };

        assert!(handle.supports(CapabilityKind::Chat));
        assert!(handle.supports(CapabilityKind::Completion));
        assert!(!handle.supports(CapabilityKind::Embeddings));
        assert!(matches!(
            handle.embeddings(),
            Err(ModelError::UnsupportedCapability(
                CapabilityKind::Embeddings
            ))
        ));

        let chat_response = handle
            .chat()
            .expect("chat should be available")
            .generate(ChatRequest {
                messages: vec![motlie_model::ChatMessage::new(
                    ChatRole::User,
                    "hello",
                )],
                ..Default::default()
            })
            .await
            .expect("stub chat should succeed");
        assert_eq!(chat_response.content, "stub response to: hello");

        let completion_response = handle
            .completion()
            .expect("completion should be available")
            .complete(CompletionRequest {
                prompt: "explain".into(),
                ..Default::default()
            })
            .await
            .expect("stub completion should succeed");
        assert_eq!(
            completion_response.content,
            "stub completion of: explain"
        );
    }

    #[test]
    fn quantization_bits_map_to_isq_bits() {
        assert_eq!(
            map_quantization_bits(QuantizationBits::Four),
            IsqBits::Four
        );
        assert_eq!(
            map_quantization_bits(QuantizationBits::Eight),
            IsqBits::Eight
        );
    }

    #[test]
    fn local_only_policy_uses_resolved_path() {
        let configured = configure_artifact_policy(
            "Qwen/Qwen3-4B",
            ArtifactPolicy::LocalOnly {
                root: PathBuf::from("/models/qwen3-4b"),
            },
        )
        .expect("local policy should succeed");

        assert_eq!(configured.model_target, "/models/qwen3-4b");
        assert_eq!(configured.hf_cache_root, None);
    }

    #[test]
    fn unpack_root_is_rejected_explicitly() {
        let result = tokio::runtime::Runtime::new()
            .expect("test runtime")
            .block_on(build_text_model(
                "Qwen/Qwen3-4B",
                MistralTextArch::Qwen3,
                StartOptions {
                    unpack_root: Some(PathBuf::from("/tmp/motlie-model-unpack")),
                    ..Default::default()
                },
            ));
        let error = result
            .err()
            .expect("unpack_root should be rejected for mistral text");

        assert!(matches!(
            error,
            ModelError::InvalidConfiguration(message)
                if message.contains("unpack_root")
        ));
    }
}
