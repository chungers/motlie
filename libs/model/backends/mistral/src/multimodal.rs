use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use image::DynamicImage;
use mistralrs::{ModelBuilder, RequestBuilder};
use motlie_model::{
    BundleHandle, BundleId, BundleMetadata, Capabilities, CapabilityKind, ChatModel, ChatRequest,
    ChatResponse, CompletionModel, ContentPart, EmbeddingModel, LoadedBundleDescriptor,
    ModelBundle, ModelError, ModelMetricSnapshot, QuantizationBits, QuantizationSupport,
    StartOptions,
};

use crate::common::{
    apply_generation_params, configure_artifact_policy, map_chat_role, map_quantization_bits,
    lock_metrics, observe_latency, observe_memory, observe_text_usage, should_enable_paged_attn,
    should_force_cpu, snapshot_text_metrics, RuntimeMetricState, TextMetricState,
};

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
            capabilities: Capabilities::multimodal_chat_and_vision(),
            quantization: QuantizationSupport::with_recommended(
                [QuantizationBits::Four, QuantizationBits::Eight],
                QuantizationBits::Four,
            )
            .expect("curated quantization support is valid"),
        }
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
            build_multimodal_model(self.model_id, self.arch, resolved_quantization, options)
                .await?;
        let metrics = Arc::new(Mutex::new(MultimodalMetrics::default()));
        {
            let mut metrics = lock_metrics(&metrics, "mistral-multimodal-start");
            observe_memory(&mut metrics.runtime);
        }

        Ok(Box::new(MistralMultimodalHandle {
            descriptor: LoadedBundleDescriptor {
                id: self.metadata.id.clone(),
                display_name: self.metadata.display_name.clone(),
                capabilities: self.metadata.capabilities.clone(),
                quantization: self.metadata.quantization.clone(),
                resolved_quantization,
            },
            runtime: Box::new(MistralMultimodalRuntime {
                model,
                metrics: Arc::clone(&metrics),
            }),
            metrics,
        }))
    }
}

#[async_trait]
trait MultimodalRuntime: Send + Sync {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ModelError>;
}

struct MistralMultimodalRuntime {
    model: mistralrs::Model,
    metrics: Arc<Mutex<MultimodalMetrics>>,
}

#[async_trait]
impl MultimodalRuntime for MistralMultimodalRuntime {
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
            let mut metrics = lock_metrics(&self.metrics, "mistral-multimodal-chat");
            observe_latency(&mut metrics.runtime, elapsed);
            observe_text_usage(&mut metrics.text, &usage);
        }

        Ok(ChatResponse { content })
    }
}

fn to_request_builder(request: &ChatRequest) -> Result<RequestBuilder, ModelError> {
    let mut builder = RequestBuilder::new();
    for msg in &request.messages {
        let (text, images) = collect_multimodal_parts(msg)?;
        if images.is_empty() {
            builder = builder.add_message(map_chat_role(msg.role), text);
        } else {
            builder = builder.add_image_message(map_chat_role(msg.role), text, images);
        }
    }
    Ok(apply_generation_params(builder, &request.params))
}

fn collect_multimodal_parts(
    message: &motlie_model::ChatMessage,
) -> Result<(String, Vec<DynamicImage>), ModelError> {
    let mut text = String::new();
    let mut images = Vec::new();

    for part in &message.content {
        match part {
            ContentPart::Text(part) => text.push_str(part),
            ContentPart::Image { data, media_type } => {
                if !media_type.starts_with("image/") {
                    return Err(ModelError::InvalidConfiguration(format!(
                        "mistralrs multimodal chat requires image/* media types, got `{media_type}`"
                    )));
                }
                let image = image::load_from_memory(data).map_err(|err| {
                    ModelError::InvalidConfiguration(format!(
                        "failed to decode image content part: {err}"
                    ))
                })?;
                images.push(image);
            }
            ContentPart::ImageUrl { url } => {
                return Err(ModelError::InvalidConfiguration(format!(
                    "mistralrs multimodal chat does not support `ContentPart::ImageUrl` yet (`{url}`); provide inline image bytes instead"
                )));
            }
        }
    }

    Ok((text, images))
}

struct MistralMultimodalHandle {
    descriptor: LoadedBundleDescriptor,
    runtime: Box<dyn MultimodalRuntime>,
    metrics: Arc<Mutex<MultimodalMetrics>>,
}

#[async_trait]
impl BundleHandle for MistralMultimodalHandle {
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

    fn chat(&self) -> Result<&dyn ChatModel, ModelError> {
        Ok(self)
    }

    fn completion(&self) -> Result<&dyn CompletionModel, ModelError> {
        Err(ModelError::UnsupportedCapability(
            CapabilityKind::Completion,
        ))
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
impl ChatModel for MistralMultimodalHandle {
    async fn generate(&self, request: ChatRequest) -> Result<ChatResponse, ModelError> {
        self.runtime.chat(request).await
    }
}

#[derive(Clone, Debug, Default)]
struct MultimodalMetrics {
    runtime: RuntimeMetricState,
    text: TextMetricState,
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
    if should_enable_paged_attn() {
        match mistralrs::PagedAttentionMetaBuilder::default().build() {
            Ok(pa_config) => {
                builder = builder.with_paged_attn(pa_config);
            }
            Err(err) => {
                eprintln!("warning: failed to configure PagedAttention, continuing without it: {err}");
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
    use motlie_model::{ChatMessage, ChatRole, ContentPart};

    struct StubMultimodalRuntime;

    #[async_trait]
    impl MultimodalRuntime for StubMultimodalRuntime {
        async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ModelError> {
            let last_text = request
                .messages
                .last()
                .map(|m| {
                    m.content
                        .iter()
                        .filter_map(|part| match part {
                            ContentPart::Text(text) => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<String>()
                })
                .unwrap_or_default();
            Ok(ChatResponse {
                content: format!("multimodal stub response to: {last_text}"),
            })
        }
    }

    #[test]
    fn gemma4_spec_has_expected_identity() {
        let spec = MistralMultimodalSpec::gemma4_e2b();

        assert_eq!(spec.id.as_str(), "gemma4_e2b");
        assert_eq!(spec.display_name, "Gemma 4 E2B-it");
        assert_eq!(spec.model_id, "google/gemma-4-E2B-it");
        assert_eq!(spec.arch, MistralMultimodalArch::Gemma4);
        assert!(spec.capabilities.supports(CapabilityKind::Chat));
        assert!(spec.capabilities.supports(CapabilityKind::Vision));
        assert!(!spec.capabilities.supports(CapabilityKind::Completion));
    }

    #[tokio::test]
    async fn multimodal_handle_exposes_chat_only() {
        let handle = MistralMultimodalHandle {
            descriptor: LoadedBundleDescriptor {
                id: BundleId::new("gemma4_e2b"),
                display_name: "Gemma 4 E2B-it".into(),
                capabilities: Capabilities::multimodal_chat_and_vision(),
                quantization: QuantizationSupport::with_recommended(
                    [QuantizationBits::Four, QuantizationBits::Eight],
                    QuantizationBits::Four,
                )
                .expect("test quantization support is valid"),
                resolved_quantization: Some(QuantizationBits::Four),
            },
            runtime: Box::new(StubMultimodalRuntime),
            metrics: Arc::new(Mutex::new(MultimodalMetrics::default())),
        };

        assert!(handle.supports(CapabilityKind::Chat));
        assert!(handle.supports(CapabilityKind::Vision));
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

        let error = collect_multimodal_parts(&message)
            .expect_err("image urls should be rejected for local mistral runtime");
        assert!(matches!(error, ModelError::InvalidConfiguration(_)));
    }
}
