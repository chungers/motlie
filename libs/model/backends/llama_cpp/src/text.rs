use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use motlie_model::{
    BundleHandle, BundleId, BundleMetadata, Capabilities, CapabilityKind, ChatModel, ChatRequest,
    ChatResponse, ChatRole, CompletionModel, CompletionRequest, CompletionResponse, EmbeddingModel,
    GenerationParams, LoadedBundleDescriptor, ModelBundle, ModelError, ModelMetricSnapshot,
    QuantizationBits, QuantizationSupport, StartOptions,
};

use crate::common::{
    configure_artifact_policy, lock_metrics, observe_latency, observe_memory,
    observe_text_generation, should_force_cpu, snapshot_text_metrics, RuntimeMetricState,
    TextMetricState,
};

/// Architecture discriminant selecting the correct chat template and model behavior.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LlamaCppTextArch {
    Qwen3,
    Gemma4,
}

/// Maps `QuantizationBits` to the curated GGUF filename for a given model.
///
/// llama.cpp uses pre-quantized GGUF files (unlike mistral.rs which applies
/// ISQ at load time from safetensors). Each precision level maps to a specific
/// GGUF file that must be downloaded separately.
fn gguf_filename(model_prefix: &str, bits: Option<QuantizationBits>) -> String {
    match bits {
        Some(QuantizationBits::Four) => format!("{model_prefix}-Q4_K_M.gguf"),
        Some(QuantizationBits::Eight) => format!("{model_prefix}-Q8_0.gguf"),
        None => format!("{model_prefix}-f16.gguf"),
    }
}

/// Static bundle specification for a curated llama.cpp-backed text generation stack.
#[derive(Clone, Debug)]
pub struct LlamaCppTextSpec {
    pub id: BundleId,
    pub display_name: &'static str,
    pub model_prefix: &'static str,
    pub arch: LlamaCppTextArch,
    pub capabilities: Capabilities,
    pub quantization: QuantizationSupport,
    pub default_context_length: u32,
}

impl LlamaCppTextSpec {
    pub fn qwen3_4b() -> Self {
        Self {
            id: BundleId::new("qwen3_4b_gguf"),
            display_name: "Qwen3 4B (GGUF)",
            model_prefix: "qwen3-4b",
            arch: LlamaCppTextArch::Qwen3,
            capabilities: Capabilities::chat_and_completion(),
            quantization: QuantizationSupport::with_recommended(
                [QuantizationBits::Four, QuantizationBits::Eight],
                QuantizationBits::Four,
            )
            .expect("curated quantization support is valid"),
            default_context_length: 4096,
        }
    }

    pub fn gemma4_e2b() -> Self {
        Self {
            id: BundleId::new("gemma4_e2b_gguf"),
            display_name: "Gemma 4 E2B-it (GGUF)",
            model_prefix: "gemma-4-e2b-it",
            arch: LlamaCppTextArch::Gemma4,
            capabilities: Capabilities::chat_and_completion(),
            quantization: QuantizationSupport::with_recommended(
                [QuantizationBits::Four, QuantizationBits::Eight],
                QuantizationBits::Four,
            )
            .expect("curated quantization support is valid"),
            default_context_length: 4096,
        }
    }
}

/// Generic `ModelBundle` implementation backed by `llama-cpp-2`.
#[derive(Clone, Debug)]
pub struct LlamaCppTextBundle {
    metadata: BundleMetadata,
    spec: LlamaCppTextSpec,
}

impl LlamaCppTextBundle {
    pub fn new(spec: LlamaCppTextSpec) -> Self {
        Self {
            metadata: BundleMetadata {
                id: spec.id.clone(),
                display_name: spec.display_name.into(),
                capabilities: spec.capabilities.clone(),
                quantization: spec.quantization.clone(),
            },
            spec,
        }
    }
}

#[async_trait]
impl ModelBundle for LlamaCppTextBundle {
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

        let model = build_llama_model(&self.spec, resolved_quantization, options)?;
        let metrics = Arc::new(Mutex::new(TextMetrics::default()));
        {
            let mut metrics = lock_metrics(&metrics, "llama-cpp-text-start");
            observe_memory(&mut metrics.runtime);
        }

        Ok(Box::new(LlamaCppTextHandle {
            descriptor: LoadedBundleDescriptor {
                id: self.metadata.id.clone(),
                display_name: self.metadata.display_name.clone(),
                capabilities: self.metadata.capabilities.clone(),
                quantization: self.metadata.quantization.clone(),
                resolved_quantization,
            },
            runtime: Box::new(LlamaCppRuntime {
                model,
                arch: self.spec.arch,
                context_length: self.spec.default_context_length,
                metrics: Arc::clone(&metrics),
            }),
            metrics,
        }))
    }
}

// ---------------------------------------------------------------------------
// Internal runtime abstraction (enables stub testing without real llama.cpp)
// ---------------------------------------------------------------------------

#[async_trait]
trait TextRuntime: Send + Sync {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ModelError>;
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ModelError>;
}

struct LlamaCppRuntime {
    model: Arc<LlamaModel>,
    arch: LlamaCppTextArch,
    context_length: u32,
    metrics: Arc<Mutex<TextMetrics>>,
}

// SAFETY: LlamaModel is thread-safe for read operations after construction.
// The llama_cpp_2 crate documents that model weights are immutable once loaded.
unsafe impl Send for LlamaCppRuntime {}
unsafe impl Sync for LlamaCppRuntime {}

#[async_trait]
impl TextRuntime for LlamaCppRuntime {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ModelError> {
        let prompt = format_chat_prompt(self.arch, &request);
        self.generate_text(&prompt, &request.params).await
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ModelError> {
        let chat_response = self
            .generate_text(&request.prompt, &request.params)
            .await?;
        Ok(CompletionResponse {
            content: chat_response.content,
        })
    }
}

impl LlamaCppRuntime {
    async fn generate_text(
        &self,
        prompt: &str,
        params: &GenerationParams,
    ) -> Result<ChatResponse, ModelError> {
        let model = Arc::clone(&self.model);
        let prompt = prompt.to_owned();
        let max_tokens: u32 = params.max_tokens.unwrap_or(512);
        let temperature: f32 = params.temperature.unwrap_or(0.7_f32);
        let top_p: Option<f32> = params.top_p;
        let stop_sequences = params.stop_sequences.clone();
        let context_length = self.context_length;
        let metrics = Arc::clone(&self.metrics);

        // Run inference on a blocking thread — llama.cpp is synchronous.
        tokio::task::spawn_blocking(move || {
            let started_at = Instant::now();

            let ctx_params = LlamaContextParams::default()
                .with_n_ctx(std::num::NonZeroU32::new(context_length));

            let mut ctx = model
                .new_context(&LlamaBackend::init().map_err(|e| {
                    ModelError::BackendInitialization {
                        backend: "llama-cpp",
                        message: e.to_string(),
                    }
                })?, ctx_params)
                .map_err(|e| ModelError::BackendExecution {
                    backend: "llama-cpp",
                    operation: "new_context",
                    message: e.to_string(),
                })?;

            let tokens = model
                .str_to_token(&prompt, AddBos::Always)
                .map_err(|e| ModelError::BackendExecution {
                    backend: "llama-cpp",
                    operation: "tokenize",
                    message: e.to_string(),
                })?;

            let prompt_token_count = tokens.len() as u32;

            let mut batch = LlamaBatch::new(context_length as usize, 1);
            for (i, token) in tokens.iter().enumerate() {
                let is_last = i == tokens.len() - 1;
                batch.add(*token, i as i32, &[0], is_last).map_err(|e| {
                    ModelError::BackendExecution {
                        backend: "llama-cpp",
                        operation: "batch_add",
                        message: e.to_string(),
                    }
                })?;
            }

            ctx.decode(&mut batch).map_err(|e| ModelError::BackendExecution {
                backend: "llama-cpp",
                operation: "decode_prompt",
                message: e.to_string(),
            })?;

            let mut sampler = if let Some(top_p) = top_p {
                LlamaSampler::chain_simple([
                    LlamaSampler::top_p(top_p, 1),
                    LlamaSampler::temp(temperature),
                    LlamaSampler::dist(42),
                ])
            } else {
                LlamaSampler::chain_simple([
                    LlamaSampler::temp(temperature),
                    LlamaSampler::dist(42),
                ])
            };

            let mut output_tokens: Vec<LlamaToken> = Vec::new();
            let mut generated_text = String::new();
            let mut n_cur = tokens.len() as i32;
            let mut decoder = encoding_rs::UTF_8.new_decoder();

            for _ in 0..max_tokens {
                let token = sampler.sample(&ctx, batch.n_tokens() - 1);
                sampler.accept(token);

                if model.is_eog_token(token) {
                    break;
                }

                let piece = model.token_to_piece(token, &mut decoder, false, None).map_err(|e| {
                    ModelError::BackendExecution {
                        backend: "llama-cpp",
                        operation: "token_to_piece",
                        message: e.to_string(),
                    }
                })?;

                // Check stop sequences.
                generated_text.push_str(&piece);
                let should_stop = stop_sequences.iter().any(|seq| generated_text.ends_with(seq));
                if should_stop {
                    for seq in &stop_sequences {
                        if generated_text.ends_with(seq) {
                            generated_text.truncate(generated_text.len() - seq.len());
                            break;
                        }
                    }
                    break;
                }

                output_tokens.push(token);
                batch.clear();
                batch.add(token, n_cur, &[0], true).map_err(|e| {
                    ModelError::BackendExecution {
                        backend: "llama-cpp",
                        operation: "batch_add_token",
                        message: e.to_string(),
                    }
                })?;
                n_cur += 1;

                ctx.decode(&mut batch).map_err(|e| ModelError::BackendExecution {
                    backend: "llama-cpp",
                    operation: "decode_token",
                    message: e.to_string(),
                })?;
            }

            let elapsed = started_at.elapsed();
            let generated_token_count = output_tokens.len() as u32;

            {
                let mut m = lock_metrics(&metrics, "llama-cpp-text-generate");
                observe_latency(&mut m.runtime, elapsed);
                observe_text_generation(
                    &mut m.text,
                    prompt_token_count,
                    generated_token_count,
                    elapsed,
                );
            }

            Ok(ChatResponse {
                content: generated_text,
            })
        })
        .await
        .map_err(|e| ModelError::BackendExecution {
            backend: "llama-cpp",
            operation: "spawn_blocking",
            message: e.to_string(),
        })?
    }
}

/// Format a chat request into the model's expected prompt template.
fn format_chat_prompt(arch: LlamaCppTextArch, request: &ChatRequest) -> String {
    match arch {
        LlamaCppTextArch::Qwen3 => format_qwen3_prompt(&request.messages),
        LlamaCppTextArch::Gemma4 => format_gemma4_prompt(&request.messages),
    }
}

fn format_qwen3_prompt(messages: &[motlie_model::ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let role = match msg.role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        };
        prompt.push_str(&format!("<|im_start|>{role}\n"));
        for part in &msg.content {
            if let motlie_model::ContentPart::Text(text) = part {
                prompt.push_str(text);
            }
        }
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

fn format_gemma4_prompt(messages: &[motlie_model::ChatMessage]) -> String {
    let mut prompt = String::new();
    for msg in messages {
        let role = match msg.role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "model",
        };
        prompt.push_str(&format!("<start_of_turn>{role}\n"));
        for part in &msg.content {
            if let motlie_model::ContentPart::Text(text) = part {
                prompt.push_str(text);
            }
        }
        prompt.push_str("<end_of_turn>\n");
    }
    prompt.push_str("<start_of_turn>model\n");
    prompt
}

// ---------------------------------------------------------------------------
// Handle
// ---------------------------------------------------------------------------

struct LlamaCppTextHandle {
    descriptor: LoadedBundleDescriptor,
    runtime: Box<dyn TextRuntime>,
    metrics: Arc<Mutex<TextMetrics>>,
}

#[async_trait]
impl BundleHandle for LlamaCppTextHandle {
    fn descriptor(&self) -> &LoadedBundleDescriptor {
        &self.descriptor
    }

    fn capabilities(&self) -> &Capabilities {
        &self.descriptor.capabilities
    }

    fn metric_snapshot(&self) -> Option<ModelMetricSnapshot> {
        let metrics = lock_metrics(&self.metrics, "llama-cpp-text-metric-snapshot").clone();
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
impl ChatModel for LlamaCppTextHandle {
    async fn generate(&self, request: ChatRequest) -> Result<ChatResponse, ModelError> {
        self.runtime.chat(request).await
    }
}

#[async_trait]
impl CompletionModel for LlamaCppTextHandle {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ModelError> {
        self.runtime.complete(request).await
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

fn build_llama_model(
    spec: &LlamaCppTextSpec,
    resolved_quantization: Option<QuantizationBits>,
    options: StartOptions,
) -> Result<Arc<LlamaModel>, ModelError> {
    let StartOptions {
        artifact_policy,
        quantization: _, // already resolved by caller
        unpack_root,
        max_concurrency: _,
    } = options;

    if let Some(unpack_root) = unpack_root {
        return Err(ModelError::InvalidConfiguration(format!(
            "`llama-cpp` startup does not support `unpack_root` (got `{}`)",
            unpack_root.display()
        )));
    }

    let filename = gguf_filename(spec.model_prefix, resolved_quantization);

    let model_path = if let Some(artifact_policy) = artifact_policy {
        configure_artifact_policy(&filename, artifact_policy)?.model_path
    } else {
        PathBuf::from(&filename)
    };

    let backend = LlamaBackend::init().map_err(|e| ModelError::BackendInitialization {
        backend: "llama-cpp",
        message: e.to_string(),
    })?;

    let mut model_params = LlamaModelParams::default();
    if should_force_cpu() {
        model_params = model_params.with_n_gpu_layers(0);
    }

    let model = LlamaModel::load_from_file(&backend, model_path, &model_params).map_err(|e| {
        ModelError::BackendInitialization {
            backend: "llama-cpp",
            message: e.to_string(),
        }
    })?;

    Ok(Arc::new(model))
}

#[cfg(test)]
mod tests {
    use super::*;
    use motlie_model::{ChatMessage, QuantizationBits, StartOptions};

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
                content: format!("llama-cpp stub response to: {prompt}"),
            })
        }

        async fn complete(
            &self,
            request: CompletionRequest,
        ) -> Result<CompletionResponse, ModelError> {
            Ok(CompletionResponse {
                content: format!("llama-cpp stub completion of: {}", request.prompt),
            })
        }
    }

    #[test]
    fn qwen3_spec_has_expected_identity() {
        let spec = LlamaCppTextSpec::qwen3_4b();

        assert_eq!(spec.id.as_str(), "qwen3_4b_gguf");
        assert_eq!(spec.display_name, "Qwen3 4B (GGUF)");
        assert_eq!(spec.arch, LlamaCppTextArch::Qwen3);
        assert!(spec.capabilities.supports(CapabilityKind::Chat));
        assert!(spec.capabilities.supports(CapabilityKind::Completion));
        assert!(!spec.capabilities.supports(CapabilityKind::Embeddings));
    }

    #[test]
    fn gemma4_spec_has_expected_identity() {
        let spec = LlamaCppTextSpec::gemma4_e2b();

        assert_eq!(spec.id.as_str(), "gemma4_e2b_gguf");
        assert_eq!(spec.display_name, "Gemma 4 E2B-it (GGUF)");
        assert_eq!(spec.arch, LlamaCppTextArch::Gemma4);
        assert!(spec.capabilities.supports(CapabilityKind::Chat));
        assert!(spec.capabilities.supports(CapabilityKind::Completion));
    }

    #[test]
    fn gguf_filename_maps_quantization_to_curated_filenames() {
        assert_eq!(
            gguf_filename("qwen3-4b", Some(QuantizationBits::Four)),
            "qwen3-4b-Q4_K_M.gguf"
        );
        assert_eq!(
            gguf_filename("qwen3-4b", Some(QuantizationBits::Eight)),
            "qwen3-4b-Q8_0.gguf"
        );
        assert_eq!(gguf_filename("qwen3-4b", None), "qwen3-4b-f16.gguf");
    }

    #[test]
    fn qwen3_chat_template_formats_correctly() {
        let messages = vec![
            ChatMessage::new(ChatRole::System, "Be concise."),
            ChatMessage::new(ChatRole::User, "Hello"),
        ];
        let prompt = format_qwen3_prompt(&messages);

        assert!(prompt.contains("<|im_start|>system\nBe concise.<|im_end|>"));
        assert!(prompt.contains("<|im_start|>user\nHello<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn gemma4_chat_template_formats_correctly() {
        let messages = vec![
            ChatMessage::new(ChatRole::System, "Be concise."),
            ChatMessage::new(ChatRole::User, "Hello"),
        ];
        let prompt = format_gemma4_prompt(&messages);

        assert!(prompt.contains("<start_of_turn>system\nBe concise.<end_of_turn>"));
        assert!(prompt.contains("<start_of_turn>user\nHello<end_of_turn>"));
        assert!(prompt.ends_with("<start_of_turn>model\n"));
    }

    #[tokio::test]
    async fn text_handle_exposes_chat_and_completion_but_not_embeddings() {
        let handle = LlamaCppTextHandle {
            descriptor: LoadedBundleDescriptor {
                id: BundleId::new("qwen3_4b_gguf"),
                display_name: "Qwen3 4B (GGUF)".into(),
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
                messages: vec![ChatMessage::new(ChatRole::User, "hello")],
                ..Default::default()
            })
            .await
            .expect("stub chat should succeed");
        assert_eq!(chat_response.content, "llama-cpp stub response to: hello");

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
            "llama-cpp stub completion of: explain"
        );
    }

    #[test]
    fn unpack_root_is_rejected_explicitly() {
        let spec = LlamaCppTextSpec::qwen3_4b();
        let result = build_llama_model(
            &spec,
            None,
            StartOptions {
                unpack_root: Some(PathBuf::from("/tmp/motlie-model-unpack")),
                ..Default::default()
            },
        );
        let error = result.err().expect("unpack_root should be rejected");

        assert!(matches!(
            error,
            ModelError::InvalidConfiguration(message)
                if message.contains("unpack_root")
        ));
    }

    fn collect_text_only_message(
        message: &motlie_model::ChatMessage,
    ) -> Result<String, ModelError> {
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

    #[test]
    fn image_content_parts_are_rejected() {
        let message = motlie_model::ChatMessage::with_parts(
            ChatRole::User,
            vec![motlie_model::ContentPart::image(vec![1, 2, 3], "image/png")],
        );

        let error = collect_text_only_message(&message)
            .expect_err("images should be rejected for text-only runtime");
        assert!(matches!(
            error,
            ModelError::UnsupportedCapability(CapabilityKind::Vision)
        ));
    }
}
