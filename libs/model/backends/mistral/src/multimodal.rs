use std::future::Future;

use mistralrs::{ModelBuilder, ModelDType};
use motlie_model::{
    BundleId, Capabilities, CapabilityKind, ChatMessage, CheckpointFormat, GenerationParams,
    ModelError, QuantizationScheme, QuantizationSupport, StartOptions, UnsupportedCompletion,
    UnsupportedEmbeddings,
};

use crate::common::{
    configure_artifact_policy, map_quantization_scheme, multimodal_message_parts,
    paged_attn_context_size, should_force_cpu, MistralMessageParts,
};
use crate::runtime::{MistralAdapter, MistralBundle, MistralHandle, MistralProfile};

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
    pub recommended_generation_params: GenerationParams,
    pub recommended_system_prompt: Option<&'static str>,
}

impl MistralMultimodalSpec {
    pub fn gemma4_e2b() -> Self {
        Self {
            id: BundleId::new("gemma4_e2b"),
            display_name: "Gemma 4 E2B-it",
            model_id: "google/gemma-4-E2B-it",
            arch: MistralMultimodalArch::Gemma4,
            // ToolUse relies on the shared template-compatible transcript
            // adapter in common.rs, not mistralrs::RequestBuilder's tool replay.
            capabilities: Capabilities::multimodal_chat_vision_and_tool_use(),
            quantization: QuantizationSupport::with_recommended(
                [
                    QuantizationScheme::Bf16,
                    QuantizationScheme::IsqQ4,
                    QuantizationScheme::IsqQ8,
                ],
                QuantizationScheme::Bf16,
            )
            .unwrap_or_else(|e| {
                tracing::error!("curated quantization construction failed (this is a bug): {e}");
                QuantizationSupport::without_recommended([
                    QuantizationScheme::Bf16,
                    QuantizationScheme::IsqQ4,
                    QuantizationScheme::IsqQ8,
                ])
            }),
            recommended_generation_params: GenerationParams::default(),
            recommended_system_prompt: None,
        }
    }

    pub fn gemma4_e4b() -> Self {
        Self {
            id: BundleId::new("gemma4_e4b"),
            display_name: "Gemma 4 E4B-it",
            model_id: "google/gemma-4-E4B-it",
            arch: MistralMultimodalArch::Gemma4,
            // Same safetensors Gemma 4 template family and adapter path as E2B.
            capabilities: Capabilities::multimodal_chat_vision_and_tool_use(),
            quantization: QuantizationSupport::with_recommended(
                [
                    QuantizationScheme::Bf16,
                    QuantizationScheme::IsqQ4,
                    QuantizationScheme::IsqQ8,
                ],
                QuantizationScheme::Bf16,
            )
            .unwrap_or_else(|e| {
                tracing::error!("curated quantization construction failed (this is a bug): {e}");
                QuantizationSupport::without_recommended([
                    QuantizationScheme::Bf16,
                    QuantizationScheme::IsqQ4,
                    QuantizationScheme::IsqQ8,
                ])
            }),
            recommended_generation_params: GenerationParams {
                temperature: Some(1.0),
                top_p: Some(0.95),
                ..Default::default()
            },
            recommended_system_prompt: Some("You are Gemma, a helpful assistant."),
        }
    }
}

pub struct MultimodalProfile;

pub type MistralMultimodalAdapter = MistralAdapter<MultimodalProfile>;
pub type MistralMultimodalBundle = MistralBundle<MultimodalProfile>;
pub type MistralMultimodalHandle = MistralHandle<MultimodalProfile>;

impl MistralAdapter<MultimodalProfile> {
    pub fn gemma4() -> Self {
        let spec = MistralMultimodalSpec::gemma4_e2b();
        Self::from_parts(spec.arch, spec.capabilities, spec.quantization)
    }
}

impl MistralBundle<MultimodalProfile> {
    pub fn new(spec: MistralMultimodalSpec) -> Self {
        Self::from_parts(
            spec.id,
            spec.display_name,
            spec.model_id,
            spec.arch,
            spec.capabilities,
            spec.quantization,
        )
    }
}

impl MistralProfile for MultimodalProfile {
    type Arch = MistralMultimodalArch;
    type Completion = UnsupportedCompletion;
    type Embeddings = UnsupportedEmbeddings;

    const FORMATS: &'static [CheckpointFormat] = &MISTRAL_MULTIMODAL_FORMATS;
    const START_METRIC_CONTEXT: &'static str = "mistral-multimodal-start";
    const CHAT_METRIC_CONTEXT: &'static str = "mistral-multimodal-chat";
    const SNAPSHOT_METRIC_CONTEXT: &'static str = "mistral-multimodal-metric-snapshot";

    fn build_model(
        model_id: &str,
        arch: Self::Arch,
        resolved_quantization: Option<QuantizationScheme>,
        options: StartOptions,
    ) -> impl Future<Output = Result<mistralrs::Model, ModelError>> + Send {
        build_multimodal_model(model_id, arch, resolved_quantization, options)
    }

    fn collect_message(message: &ChatMessage) -> Result<MistralMessageParts, ModelError> {
        multimodal_message_parts(message)
    }

    fn completion(handle: &MistralMultimodalHandle) -> Result<&Self::Completion, ModelError> {
        let _ = handle.unsupported_completion();
        Err(ModelError::UnsupportedCapability(
            CapabilityKind::Completion,
        ))
    }

    fn embeddings(handle: &MistralMultimodalHandle) -> Result<&Self::Embeddings, ModelError> {
        let _ = handle.unsupported_embeddings();
        Err(ModelError::UnsupportedCapability(
            CapabilityKind::Embeddings,
        ))
    }
}

async fn build_multimodal_model(
    model_id: &str,
    _arch: MistralMultimodalArch,
    resolved_quantization: Option<QuantizationScheme>,
    options: StartOptions,
) -> Result<mistralrs::Model, ModelError> {
    let StartOptions {
        artifact_policy,
        quantization_scheme: _, // already resolved by caller
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
    if let Some(scheme) = resolved_quantization {
        builder = match scheme {
            QuantizationScheme::Bf16 => builder.with_dtype(ModelDType::BF16),
            QuantizationScheme::Fp16 => builder.with_dtype(ModelDType::F16),
            QuantizationScheme::Fp32 => builder.with_dtype(ModelDType::F32),
            QuantizationScheme::IsqQ4 | QuantizationScheme::IsqQ8 => {
                builder.with_auto_isq(map_quantization_scheme(scheme)?)
            }
            other => {
                return Err(ModelError::InvalidConfiguration(format!(
                    "mistral.rs multimodal backend does not support {other:?} quantization"
                )))
            }
        };
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
    use crate::runtime::MistralStubKind;
    use motlie_model::{
        BackendAdapter, BackendKind, BundleHandle, ChatMessage, ChatModel, ChatRequest, ChatRole,
        ContentPart,
    };

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
    fn gemma4_e4b_spec_uses_model_card_defaults() {
        let spec = MistralMultimodalSpec::gemma4_e4b();

        assert_eq!(spec.id.as_str(), "gemma4_e4b");
        assert_eq!(spec.display_name, "Gemma 4 E4B-it");
        assert_eq!(spec.model_id, "google/gemma-4-E4B-it");
        assert_eq!(
            spec.quantization.recommended(),
            Some(QuantizationScheme::Bf16)
        );
        assert_eq!(spec.recommended_generation_params.temperature, Some(1.0));
        assert_eq!(spec.recommended_generation_params.top_p, Some(0.95));
        assert_eq!(
            spec.recommended_system_prompt,
            Some("You are Gemma, a helpful assistant.")
        );
        assert!(spec.capabilities.supports(CapabilityKind::Chat));
        assert!(spec.capabilities.supports(CapabilityKind::Vision));
        assert!(spec.capabilities.supports(CapabilityKind::ToolUse));
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
            Some(QuantizationScheme::Bf16)
        );
    }

    #[tokio::test]
    async fn multimodal_handle_exposes_chat_only() {
        let handle = MistralMultimodalHandle::stub(
            BundleId::new("gemma4_e2b"),
            "Gemma 4 E2B-it".into(),
            Capabilities::multimodal_chat_vision_and_tool_use(),
            QuantizationSupport::with_recommended(
                [QuantizationScheme::IsqQ4, QuantizationScheme::IsqQ8],
                QuantizationScheme::IsqQ4,
            )
            .expect("test quantization support is valid"),
            Some(QuantizationScheme::IsqQ4),
            MistralStubKind::Multimodal,
        );

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
