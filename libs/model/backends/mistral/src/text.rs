use std::future::Future;

use mistralrs::core::NormalLoaderType;
use mistralrs::TextModelBuilder;
use motlie_model::{
    BundleId, Capabilities, CapabilityKind, ChatMessage, CheckpointFormat, ModelError,
    QuantizationBits, QuantizationSupport, StartOptions, UnsupportedEmbeddings,
};

use crate::common::{
    configure_artifact_policy, map_quantization_bits, paged_attn_context_size, should_force_cpu,
    text_only_message_parts, MistralMessageParts,
};
use crate::runtime::{MistralAdapter, MistralBundle, MistralHandle, MistralProfile};

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
            capabilities: Capabilities::chat_completion_and_tool_use(),
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

pub struct TextProfile;

pub type MistralTextAdapter = MistralAdapter<TextProfile>;
pub type MistralTextBundle = MistralBundle<TextProfile>;
pub type MistralTextHandle = MistralHandle<TextProfile>;

impl MistralAdapter<TextProfile> {
    pub fn qwen3() -> Self {
        let spec = MistralTextSpec::qwen3_4b();
        Self::from_parts(spec.arch, spec.capabilities, spec.quantization)
    }
}

impl MistralBundle<TextProfile> {
    pub fn new(spec: MistralTextSpec) -> Self {
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

impl MistralProfile for TextProfile {
    type Arch = MistralTextArch;
    type Completion = MistralTextHandle;
    type Embeddings = UnsupportedEmbeddings;

    const FORMATS: &'static [CheckpointFormat] = &MISTRAL_TEXT_FORMATS;
    const START_METRIC_CONTEXT: &'static str = "mistral-text-start";
    const CHAT_METRIC_CONTEXT: &'static str = "mistral-text-chat";
    const SNAPSHOT_METRIC_CONTEXT: &'static str = "mistral-text-metric-snapshot";

    fn build_model(
        model_id: &str,
        arch: Self::Arch,
        resolved_quantization: Option<QuantizationBits>,
        options: StartOptions,
    ) -> impl Future<Output = Result<mistralrs::Model, ModelError>> + Send {
        build_text_model(model_id, arch, resolved_quantization, options)
    }

    fn collect_message(message: &ChatMessage) -> Result<MistralMessageParts, ModelError> {
        text_only_message_parts(message)
    }

    fn completion(handle: &MistralTextHandle) -> Result<&Self::Completion, ModelError> {
        Ok(handle)
    }

    fn embeddings(handle: &MistralTextHandle) -> Result<&Self::Embeddings, ModelError> {
        let _ = handle.unsupported_embeddings();
        Err(ModelError::UnsupportedCapability(
            CapabilityKind::Embeddings,
        ))
    }
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
    use std::path::PathBuf;

    use crate::runtime::MistralStubKind;
    use mistralrs::IsqBits;
    use motlie_model::{
        ArtifactPolicy, BackendAdapter, BackendKind, BundleHandle, ChatModel, ChatRequest,
        ChatRole, CompletionModel, CompletionRequest, QuantizationBits, StartOptions,
    };

    #[test]
    fn qwen3_spec_has_expected_identity() {
        let spec = MistralTextSpec::qwen3_4b();

        assert_eq!(spec.id.as_str(), "qwen3_4b");
        assert_eq!(spec.display_name, "Qwen3 4B");
        assert_eq!(spec.model_id, "Qwen/Qwen3-4B");
        assert_eq!(spec.arch, MistralTextArch::Qwen3);
        assert!(spec.capabilities.supports(CapabilityKind::Chat));
        assert!(spec.capabilities.supports(CapabilityKind::Completion));
        assert!(spec.capabilities.supports(CapabilityKind::ToolUse));
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
        assert_eq!(
            adapter.capabilities(),
            &Capabilities::chat_completion_and_tool_use()
        );
        assert_eq!(
            adapter.quantization().recommended(),
            Some(QuantizationBits::Four)
        );
    }

    #[tokio::test]
    async fn text_handle_exposes_chat_and_completion_but_not_embeddings() {
        let handle = MistralTextHandle::stub(
            BundleId::new("qwen3_4b"),
            "Qwen3 4B".into(),
            Capabilities::chat_completion_and_tool_use(),
            QuantizationSupport::with_recommended(
                [QuantizationBits::Four, QuantizationBits::Eight],
                QuantizationBits::Four,
            )
            .expect("test quantization support is valid"),
            Some(QuantizationBits::Four),
            MistralStubKind::Text,
        );

        assert!(handle.supports(CapabilityKind::Chat));
        assert!(handle.supports(CapabilityKind::Completion));
        assert!(handle.supports(CapabilityKind::ToolUse));
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
        assert_eq!(
            map_quantization_bits(QuantizationBits::Four).expect("q4 should map"),
            IsqBits::Four
        );
        assert_eq!(
            map_quantization_bits(QuantizationBits::Eight).expect("q8 should map"),
            IsqBits::Eight
        );
        assert!(map_quantization_bits(QuantizationBits::Five).is_err());
        assert!(map_quantization_bits(QuantizationBits::FloatEight).is_err());
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
