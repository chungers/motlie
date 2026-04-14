use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use mistralrs::core::NormalLoaderType;
use mistralrs::{RequestBuilder, TextModelBuilder};
use motlie_model::{
    BackendAdapter, BackendKind, BundleHandle, BundleId, BundleMetadata, Capabilities,
    CapabilityKind, ChatModel, ChatRequest, ChatResponse, ChatRole, CheckpointFormat,
    CompletionModel, CompletionRequest, CompletionResponse, EmbeddingModel, LoadedBundleDescriptor,
    ModelBundle, ModelError, ModelIdentity, ModelMetricSnapshot, QuantizationBits,
    QuantizationSupport, ResolvedCheckpoint, StartOptions, TranscriptionModel,
};

use crate::common::{
    apply_generation_params, configure_artifact_policy, lock_metrics, map_chat_role,
    map_quantization_bits, observe_latency, observe_memory, observe_text_usage,
    paged_attn_context_size, resolve_local_checkpoint, should_force_cpu, snapshot_text_metrics,
    RuntimeMetricState, TextMetricState,
};

const MISTRAL_TEXT_FORMATS: [CheckpointFormat; 1] = [CheckpointFormat::Safetensors];

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
    pub quantization: QuantizationSupport,
}

impl MistralTextSpec {
    pub fn qwen3_4b() -> Self {
        Self {
            id: BundleId::new("qwen3_4b"),
            display_name: "Qwen3 4B",
            model_id: "Qwen/Qwen3-4B",
            arch: MistralTextArch::Qwen3,
            capabilities: Capabilities::chat_and_completion(),
            quantization: QuantizationSupport::with_recommended(
                [QuantizationBits::Four, QuantizationBits::Eight],
                QuantizationBits::Four,
            )
            .unwrap_or_else(|e| {
                tracing::error!("curated quantization construction failed (this is a bug): {e}");
                QuantizationSupport::without_recommended([
                    QuantizationBits::Four,
                    QuantizationBits::Eight,
                ])
            }),
        }
    }
}

/// Backend adapter for `mistralrs` text-generation architectures over safetensors checkpoints.
#[derive(Clone, Debug)]
pub struct MistralTextAdapter {
    arch: MistralTextArch,
    capabilities: Capabilities,
    quantization: QuantizationSupport,
}

impl MistralTextAdapter {
    pub fn qwen3() -> Self {
        let spec = MistralTextSpec::qwen3_4b();
        Self {
            arch: spec.arch,
            capabilities: spec.capabilities,
            quantization: spec.quantization,
        }
    }
}

#[async_trait]
impl BackendAdapter for MistralTextAdapter {
    fn supported_formats(&self) -> &[CheckpointFormat] {
        &MISTRAL_TEXT_FORMATS
    }

    fn backend_kind(&self) -> BackendKind {
        BackendKind::MistralRs
    }

    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    fn quantization(&self) -> &QuantizationSupport {
        &self.quantization
    }

    async fn start(
        &self,
        identity: &ModelIdentity,
        checkpoint: &ResolvedCheckpoint,
        options: StartOptions,
    ) -> Result<Box<dyn BundleHandle>, ModelError> {
        let resolved_quantization = self
            .quantization
            .resolve(options.quantization, &identity.id)?;
        let (model_id, options) =
            resolve_local_checkpoint(checkpoint, CheckpointFormat::Safetensors, options)?;
        let model = build_text_model(model_id, self.arch, resolved_quantization, options).await?;

        Ok(new_text_handle(
            identity.id.clone(),
            identity.display_name.clone(),
            self.capabilities.clone(),
            self.quantization.clone(),
            resolved_quantization,
            model,
        ))
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
                quantization: spec.quantization,
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
        let resolved_quantization = self
            .metadata
            .quantization
            .resolve(options.quantization, &self.metadata.id)?;
        let model =
            build_text_model(self.model_id, self.arch, resolved_quantization, options).await?;

        Ok(new_text_handle(
            self.metadata.id.clone(),
            self.metadata.display_name.clone(),
            self.metadata.capabilities.clone(),
            self.metadata.quantization.clone(),
            resolved_quantization,
            model,
        ))
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
    metrics: Arc<Mutex<TextMetrics>>,
}

#[async_trait]
impl TextRuntime for MistralTextRuntime {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ModelError> {
        let builder = to_request_builder(&request)?;
        let started_at = Instant::now();

        let response = self.model.send_chat_request(builder).await.map_err(|err| {
            ModelError::BackendExecution {
                backend: "mistralrs",
                operation: "send_chat_request",
                message: err.to_string(),
            }
        })?;
        let elapsed = started_at.elapsed();

        let usage = response.usage.clone();
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

        {
            let mut metrics = lock_metrics(&self.metrics, "mistral-text-chat");
            observe_latency(&mut metrics.runtime, elapsed);
            observe_text_usage(&mut metrics.text, &usage);
        }

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

fn to_request_builder(request: &ChatRequest) -> Result<RequestBuilder, ModelError> {
    let mut builder = RequestBuilder::new();
    for msg in &request.messages {
        builder = builder.add_message(map_chat_role(msg.role), collect_text_only_message(msg)?);
    }
    Ok(apply_generation_params(builder, &request.params))
}

fn collect_text_only_message(message: &motlie_model::ChatMessage) -> Result<String, ModelError> {
    let mut text = String::new();
    for part in &message.content {
        match part {
            motlie_model::ContentPart::Text(part) => text.push_str(part),
            motlie_model::ContentPart::Image { .. }
            | motlie_model::ContentPart::ImageUrl { .. } => {
                return Err(ModelError::UnsupportedCapability(CapabilityKind::Vision));
            }
        }
    }
    Ok(text)
}

// ---------------------------------------------------------------------------
// Handle
// ---------------------------------------------------------------------------

struct MistralTextHandle {
    descriptor: LoadedBundleDescriptor,
    runtime: Box<dyn TextRuntime>,
    metrics: Arc<Mutex<TextMetrics>>,
}

#[async_trait]
impl BundleHandle for MistralTextHandle {
    fn descriptor(&self) -> &LoadedBundleDescriptor {
        &self.descriptor
    }

    fn capabilities(&self) -> &Capabilities {
        &self.descriptor.capabilities
    }

    fn metric_snapshot(&self) -> Option<ModelMetricSnapshot> {
        let metrics = lock_metrics(&self.metrics, "mistral-text-metric-snapshot").clone();
        Some(snapshot_text_metrics(&metrics.runtime, &metrics.text))
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

    fn transcription(&self) -> Result<&dyn TranscriptionModel, ModelError> {
        Err(ModelError::UnsupportedCapability(
            CapabilityKind::Transcription,
        ))
    }

    async fn shutdown(self: Box<Self>) -> Result<(), ModelError> {
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
struct TextMetrics {
    runtime: RuntimeMetricState,
    text: TextMetricState,
}

#[async_trait]
impl ChatModel for MistralTextHandle {
    async fn generate(&self, request: ChatRequest) -> Result<ChatResponse, ModelError> {
        self.runtime.chat(request).await
    }
}

#[async_trait]
impl CompletionModel for MistralTextHandle {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ModelError> {
        self.runtime.complete(request).await
    }
}

fn new_text_handle(
    id: BundleId,
    display_name: String,
    capabilities: Capabilities,
    quantization: QuantizationSupport,
    resolved_quantization: Option<QuantizationBits>,
    model: mistralrs::Model,
) -> Box<dyn BundleHandle> {
    let metrics = Arc::new(Mutex::new(TextMetrics::default()));
    {
        let mut metrics = lock_metrics(&metrics, "mistral-text-start");
        observe_memory(&mut metrics.runtime);
    }

    Box::new(MistralTextHandle {
        descriptor: LoadedBundleDescriptor {
            id,
            display_name,
            capabilities,
            quantization,
            resolved_quantization,
        },
        runtime: Box::new(MistralTextRuntime {
            model,
            metrics: Arc::clone(&metrics),
        }),
        metrics,
    })
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

async fn build_text_model(
    model_id: &str,
    arch: MistralTextArch,
    resolved_quantization: Option<QuantizationBits>,
    options: StartOptions,
) -> Result<mistralrs::Model, ModelError> {
    let StartOptions {
        artifact_policy,
        quantization: _, // already resolved by caller
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

    let mut builder = TextModelBuilder::new(model_target).with_loader_type(arch.loader_type());

    if let Some(bits) = resolved_quantization {
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
    if let Some(context_size) = paged_attn_context_size() {
        match mistralrs::PagedAttentionMetaBuilder::default()
            .with_gpu_memory(mistralrs::MemoryGpuConfig::ContextSize(context_size))
            .build()
        {
            Ok(pa_config) => {
                builder = builder.with_paged_attn(pa_config);
            }
            Err(err) => {
                tracing::warn!("failed to configure PagedAttention with context size {context_size}, continuing without it: {err}");
            }
        }
    }

    builder
        .build()
        .await
        .map_err(|err| ModelError::BackendInitialization {
            backend: "mistralrs",
            message: err.to_string(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use mistralrs::IsqBits;
    use motlie_model::{
        ArtifactPolicy, BackendAdapter, BackendKind, QuantizationBits, StartOptions,
    };

    struct StubTextRuntime;

    #[async_trait]
    impl TextRuntime for StubTextRuntime {
        async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ModelError> {
            let prompt = request
                .messages
                .last()
                .and_then(|m| m.content.first())
                .and_then(|part| match part {
                    motlie_model::ContentPart::Text(text) => Some(text.clone()),
                    _ => None,
                })
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

    #[test]
    fn qwen3_adapter_reports_backend_metadata() {
        let adapter = MistralTextAdapter::qwen3();

        assert_eq!(
            adapter.supported_formats(),
            &[CheckpointFormat::Safetensors]
        );
        assert_eq!(adapter.backend_kind(), BackendKind::MistralRs);
        assert_eq!(adapter.capabilities(), &Capabilities::chat_and_completion());
        assert_eq!(
            adapter.quantization().recommended(),
            Some(QuantizationBits::Four)
        );
    }

    #[tokio::test]
    async fn text_handle_exposes_chat_and_completion_but_not_embeddings() {
        let handle = MistralTextHandle {
            descriptor: LoadedBundleDescriptor {
                id: BundleId::new("qwen3_4b"),
                display_name: "Qwen3 4B".into(),
                capabilities: Capabilities::chat_and_completion(),
                quantization: QuantizationSupport::with_recommended(
                    [QuantizationBits::Four, QuantizationBits::Eight],
                    QuantizationBits::Four,
                )
                .expect("test quantization support is valid"),
                resolved_quantization: Some(QuantizationBits::Four),
            },
            runtime: Box::new(StubTextRuntime),
            metrics: Arc::new(Mutex::new(TextMetrics::default())),
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
                messages: vec![motlie_model::ChatMessage::new(ChatRole::User, "hello")],
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
        assert_eq!(completion_response.content, "stub completion of: explain");
    }

    #[test]
    fn quantization_bits_map_to_isq_bits() {
        assert_eq!(map_quantization_bits(QuantizationBits::Four), IsqBits::Four);
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
                None, // resolved quantization not relevant for unpack_root test
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
