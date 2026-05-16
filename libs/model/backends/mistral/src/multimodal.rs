use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use mistralrs::ModelBuilder;
use motlie_model::{
    BackendAdapter, BackendKind, BundleHandle, BundleId, BundleMetadata, Capabilities,
    CapabilityKind, ChatModel, ChatRequest, ChatResponse, CheckpointFormat, LoadedBundleDescriptor,
    ModelBundle, ModelError, ModelIdentity, ModelMetricSnapshot, QuantizationBits,
    QuantizationSupport, ResolvedCheckpoint, StartOptions, UnsupportedCompletion,
    UnsupportedEmbeddings,
};

use crate::common::{
    configure_artifact_policy, lock_metrics, map_quantization_bits, multimodal_message_parts,
    observe_memory, paged_attn_context_size, resolve_local_checkpoint, should_force_cpu,
    snapshot_text_metrics, MistralChatMetrics, MistralChatRuntime,
};

const MISTRAL_MULTIMODAL_FORMATS: [CheckpointFormat; 1] = [CheckpointFormat::Safetensors];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MistralMultimodalArch {
    Gemma4,
}

#[derive(Clone, Debug)]
pub struct MistralMultimodalSpec {
    pub id: BundleId,
    pub display_name: &'static str,
    pub model_id: &'static str,
    pub arch: MistralMultimodalArch,
    pub capabilities: Capabilities,
    pub quantization: QuantizationSupport,
}

impl MistralMultimodalSpec {
    pub fn gemma4_e2b() -> Self {
        Self {
            id: BundleId::new("gemma4_e2b"),
            display_name: "Gemma 4 E2B-it",
            model_id: "google/gemma-4-E2B-it",
            arch: MistralMultimodalArch::Gemma4,
            capabilities: Capabilities::multimodal_chat_vision_and_tool_use(),
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

/// Backend adapter for `mistralrs` multimodal chat over safetensors checkpoints.
#[derive(Clone, Debug)]
pub struct MistralMultimodalAdapter {
    arch: MistralMultimodalArch,
    capabilities: Capabilities,
    quantization: QuantizationSupport,
}

impl MistralMultimodalAdapter {
    pub fn gemma4() -> Self {
        let spec = MistralMultimodalSpec::gemma4_e2b();
        Self {
            arch: spec.arch,
            capabilities: spec.capabilities,
            quantization: spec.quantization,
        }
    }
}

#[async_trait]
impl BackendAdapter for MistralMultimodalAdapter {
    type Handle = MistralMultimodalHandle;

    fn supported_formats(&self) -> &[CheckpointFormat] {
        &MISTRAL_MULTIMODAL_FORMATS
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
    ) -> Result<Self::Handle, ModelError> {
        let resolved_quantization = self
            .quantization
            .resolve(options.quantization, &identity.id)?;
        let (model_id, options) =
            resolve_local_checkpoint(checkpoint, CheckpointFormat::Safetensors, options)?;
        let model =
            build_multimodal_model(model_id, self.arch, resolved_quantization, options).await?;

        Ok(new_multimodal_handle(
            identity.id.clone(),
            identity.display_name.clone(),
            self.capabilities.clone(),
            self.quantization.clone(),
            resolved_quantization,
            model,
        ))
    }
}

#[derive(Clone, Debug)]
pub struct MistralMultimodalBundle {
    metadata: BundleMetadata,
    arch: MistralMultimodalArch,
    model_id: &'static str,
}

impl MistralMultimodalBundle {
    pub fn new(spec: MistralMultimodalSpec) -> Self {
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
impl ModelBundle for MistralMultimodalBundle {
    type Handle = MistralMultimodalHandle;

    fn id(&self) -> &BundleId {
        &self.metadata.id
    }

    fn metadata(&self) -> &BundleMetadata {
        &self.metadata
    }

    fn capabilities(&self) -> &Capabilities {
        &self.metadata.capabilities
    }

    async fn start(&self, options: StartOptions) -> Result<Self::Handle, ModelError> {
        let resolved_quantization = self
            .metadata
            .quantization
            .resolve(options.quantization, &self.metadata.id)?;
        let model =
            build_multimodal_model(self.model_id, self.arch, resolved_quantization, options)
                .await?;

        Ok(new_multimodal_handle(
            self.metadata.id.clone(),
            self.metadata.display_name.clone(),
            self.metadata.capabilities.clone(),
            self.metadata.quantization.clone(),
            resolved_quantization,
            model,
        ))
    }
}

enum MultimodalRuntime {
    Real(MistralChatRuntime),
    #[cfg(test)]
    Stub(StubMultimodalRuntime),
}

#[cfg(test)]
struct StubMultimodalRuntime;

#[cfg(test)]
impl StubMultimodalRuntime {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ModelError> {
        let last_text = request
            .messages
            .last()
            .map(|m| {
                m.content
                    .iter()
                    .filter_map(|part| match part {
                        motlie_model::ContentPart::Text(text) => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<String>()
            })
            .unwrap_or_default();
        Ok(ChatResponse::text(format!(
            "multimodal stub response to: {last_text}"
        )))
    }
}

impl MultimodalRuntime {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ModelError> {
        match self {
            Self::Real(runtime) => runtime.chat(request).await,
            #[cfg(test)]
            Self::Stub(runtime) => runtime.chat(request).await,
        }
    }
}

pub struct MistralMultimodalHandle {
    descriptor: LoadedBundleDescriptor,
    runtime: MultimodalRuntime,
    metrics: Arc<Mutex<MistralChatMetrics>>,
}

#[async_trait]
impl BundleHandle for MistralMultimodalHandle {
    type Chat = Self;
    type Completion = UnsupportedCompletion;
    type Embeddings = UnsupportedEmbeddings;

    fn descriptor(&self) -> &LoadedBundleDescriptor {
        &self.descriptor
    }

    fn capabilities(&self) -> &Capabilities {
        &self.descriptor.capabilities
    }

    fn metric_snapshot(&self) -> Option<ModelMetricSnapshot> {
        let metrics = lock_metrics(&self.metrics, "mistral-multimodal-metric-snapshot").clone();
        Some(snapshot_text_metrics(&metrics.runtime, &metrics.text))
    }

    fn chat(&self) -> Result<&Self::Chat, ModelError> {
        Ok(self)
    }

    fn completion(&self) -> Result<&Self::Completion, ModelError> {
        Err(ModelError::UnsupportedCapability(
            CapabilityKind::Completion,
        ))
    }

    fn embeddings(&self) -> Result<&Self::Embeddings, ModelError> {
        Err(ModelError::UnsupportedCapability(
            CapabilityKind::Embeddings,
        ))
    }

    async fn shutdown(self) -> Result<(), ModelError> {
        Ok(())
    }
}

#[async_trait]
impl ChatModel for MistralMultimodalHandle {
    async fn generate(&self, request: ChatRequest) -> Result<ChatResponse, ModelError> {
        self.runtime.chat(request).await
    }
}

fn new_multimodal_handle(
    id: BundleId,
    display_name: String,
    capabilities: Capabilities,
    quantization: QuantizationSupport,
    resolved_quantization: Option<QuantizationBits>,
    model: mistralrs::Model,
) -> MistralMultimodalHandle {
    let metrics = Arc::new(Mutex::new(MistralChatMetrics::default()));
    {
        let mut metrics = lock_metrics(&metrics, "mistral-multimodal-start");
        observe_memory(&mut metrics.runtime);
    }

    MistralMultimodalHandle {
        descriptor: LoadedBundleDescriptor {
            id,
            display_name,
            capabilities,
            quantization,
            resolved_quantization,
        },
        runtime: MultimodalRuntime::Real(MistralChatRuntime::new(
            model,
            Arc::clone(&metrics),
            "mistral-multimodal-chat",
            multimodal_message_parts,
        )),
        metrics,
    }
}

async fn build_multimodal_model(
    model_id: &str,
    _arch: MistralMultimodalArch,
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
            "`mistralrs` multimodal startup does not support `unpack_root` yet (got `{}`)",
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

    // Use the auto-detecting builder here. Upstream's Gemma 4 multimodal examples use
    // `ModelBuilder`, and that path preserves multimodal chat-template discovery.
    let mut builder = ModelBuilder::new(model_target);
    if let Some(bits) = resolved_quantization {
        builder = builder.with_auto_isq(map_quantization_bits(bits)?);
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
                tracing::warn!(
                    "failed to configure PagedAttention with context size {context_size}, continuing without it: {err}"
                );
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
    use motlie_model::{BackendAdapter, BackendKind, ChatMessage, ChatRole, ContentPart};

    #[test]
    fn gemma4_spec_has_expected_identity() {
        let spec = MistralMultimodalSpec::gemma4_e2b();

        assert_eq!(spec.id.as_str(), "gemma4_e2b");
        assert_eq!(spec.display_name, "Gemma 4 E2B-it");
        assert_eq!(spec.model_id, "google/gemma-4-E2B-it");
        assert_eq!(spec.arch, MistralMultimodalArch::Gemma4);
        assert!(spec.capabilities.supports(CapabilityKind::Chat));
        assert!(spec.capabilities.supports(CapabilityKind::Vision));
        assert!(spec.capabilities.supports(CapabilityKind::ToolUse));
        assert!(!spec.capabilities.supports(CapabilityKind::Completion));
    }

    #[test]
    fn gemma4_adapter_reports_backend_metadata() {
        let adapter = MistralMultimodalAdapter::gemma4();

        assert_eq!(
            adapter.supported_formats(),
            &[CheckpointFormat::Safetensors]
        );
        assert_eq!(adapter.backend_kind(), BackendKind::MistralRs);
        assert_eq!(
            adapter.capabilities(),
            &Capabilities::multimodal_chat_vision_and_tool_use()
        );
        assert_eq!(
            adapter.quantization().recommended(),
            Some(QuantizationBits::Four)
        );
    }

    #[tokio::test]
    async fn multimodal_handle_exposes_chat_only() {
        let handle = MistralMultimodalHandle {
            descriptor: LoadedBundleDescriptor {
                id: BundleId::new("gemma4_e2b"),
                display_name: "Gemma 4 E2B-it".into(),
                capabilities: Capabilities::multimodal_chat_vision_and_tool_use(),
                quantization: QuantizationSupport::with_recommended(
                    [QuantizationBits::Four, QuantizationBits::Eight],
                    QuantizationBits::Four,
                )
                .expect("test quantization support is valid"),
                resolved_quantization: Some(QuantizationBits::Four),
            },
            runtime: MultimodalRuntime::Stub(StubMultimodalRuntime),
            metrics: Arc::new(Mutex::new(MistralChatMetrics::default())),
        };

        assert!(handle.supports(CapabilityKind::Chat));
        assert!(handle.supports(CapabilityKind::Vision));
        assert!(handle.supports(CapabilityKind::ToolUse));
        assert!(!handle.supports(CapabilityKind::Completion));
        assert!(matches!(
            handle.completion(),
            Err(ModelError::UnsupportedCapability(
                CapabilityKind::Completion
            ))
        ));
        assert!(matches!(
            handle.embeddings(),
            Err(ModelError::UnsupportedCapability(
                CapabilityKind::Embeddings
            ))
        ));

        let response = handle
            .chat()
            .expect("chat should be available")
            .generate(ChatRequest {
                messages: vec![ChatMessage::with_parts(
                    ChatRole::User,
                    vec![ContentPart::text("describe this image")],
                )],
                ..Default::default()
            })
            .await
            .expect("stub chat should succeed");

        assert_eq!(
            response.content,
            "multimodal stub response to: describe this image"
        );
    }

    #[test]
    fn image_urls_are_rejected_for_local_runtime() {
        let message = ChatMessage::with_parts(
            ChatRole::User,
            vec![ContentPart::image_url("https://example.com/cat.jpg")],
        );

        let error = multimodal_message_parts(&message)
            .expect_err("image urls should be rejected for local mistral runtime");
        assert!(matches!(error, ModelError::InvalidConfiguration(_)));
    }
}
