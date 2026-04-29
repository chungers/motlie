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
use motlie_model::{
    BackendAdapter, BackendKind, BundleHandle, BundleId, BundleMetadata, Capabilities,
    CapabilityKind, ChatModel, ChatRequest, ChatResponse, ChatRole, CheckpointFormat,
    CompletionModel, CompletionRequest, CompletionResponse, GenerationParams,
    LoadedBundleDescriptor, ModelBundle, ModelError, ModelIdentity, ModelMetricSnapshot,
    QuantizationBits, QuantizationSupport, ResolvedCheckpoint, StartOptions, UnsupportedEmbeddings,
};

use crate::common::{
    RuntimeMetricState, TextMetricState, configure_artifact_policy, lock_metrics, observe_latency,
    observe_memory, observe_text_generation, resolve_gpu_layers, snapshot_text_metrics,
};

const LLAMA_CPP_TEXT_FORMATS: [CheckpointFormat; 1] = [CheckpointFormat::Gguf];

/// Architecture discriminant selecting the correct chat template and model behavior.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LlamaCppTextArch {
    Qwen3,
    Qwen35,
    Gemma4,
}

/// Maps `QuantizationBits` to the curated GGUF filename for a given model.
///
/// llama.cpp uses pre-quantized GGUF files (unlike mistral.rs which applies
/// ISQ at load time from safetensors). Each precision level maps to a specific
/// GGUF file that must be downloaded separately.
fn gguf_filename(model_prefix: &str, bits: Option<QuantizationBits>) -> String {
    // INVARIANT: `bits` must already be resolved through the spec's
    // `QuantizationSupport`; this mapper only names the curated GGUF artifact.
    format!("{model_prefix}{}", gguf_quant_suffix(bits))
}

fn gguf_quant_suffix(bits: Option<QuantizationBits>) -> &'static str {
    match bits {
        Some(QuantizationBits::Four) => "-Q4_K_M.gguf",
        Some(QuantizationBits::Five) => "-Q5_K_M.gguf",
        Some(QuantizationBits::Eight) => "-Q8_0.gguf",
        Some(QuantizationBits::FloatEight) => "-FP8.gguf",
        None => "-f16.gguf",
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
            model_prefix: "Qwen3-4B",
            arch: LlamaCppTextArch::Qwen3,
            capabilities: Capabilities::chat_and_completion(),
            quantization: curated_q4_q8_support(),
            default_context_length: 4096,
        }
    }

    pub fn gemma4_e2b() -> Self {
        Self {
            id: BundleId::new("gemma4_e2b_gguf"),
            display_name: "Gemma 4 E2B-it (GGUF)",
            model_prefix: "gemma-4-E2B-it",
            arch: LlamaCppTextArch::Gemma4,
            capabilities: Capabilities::chat_and_completion(),
            quantization: curated_q4_q8_support(),
            default_context_length: 4096,
        }
    }

    pub fn qwen3_6_27b() -> Self {
        Self {
            id: BundleId::new("qwen3_6_27b_gguf"),
            display_name: "Qwen3.6 27B (GGUF)",
            model_prefix: "Qwen3.6-27B",
            arch: LlamaCppTextArch::Qwen35,
            capabilities: Capabilities::chat_and_completion(),
            quantization: curated_qwen36_gguf_support(),
            default_context_length: 32768,
        }
    }
}

/// Backend adapter for `llama.cpp` text-generation over GGUF checkpoints.
#[derive(Clone, Debug)]
pub struct LlamaCppTextAdapter {
    arch: LlamaCppTextArch,
    capabilities: Capabilities,
    quantization: QuantizationSupport,
    default_context_length: u32,
}

impl LlamaCppTextAdapter {
    pub fn qwen3() -> Self {
        let spec = LlamaCppTextSpec::qwen3_4b();
        Self {
            arch: spec.arch,
            capabilities: spec.capabilities,
            quantization: spec.quantization,
            default_context_length: spec.default_context_length,
        }
    }

    pub fn gemma4() -> Self {
        let spec = LlamaCppTextSpec::gemma4_e2b();
        Self {
            arch: spec.arch,
            capabilities: spec.capabilities,
            quantization: spec.quantization,
            default_context_length: spec.default_context_length,
        }
    }

    pub fn qwen35() -> Self {
        let spec = LlamaCppTextSpec::qwen3_6_27b();
        Self {
            arch: spec.arch,
            capabilities: spec.capabilities,
            quantization: spec.quantization,
            default_context_length: spec.default_context_length,
        }
    }
}

#[async_trait]
impl BackendAdapter for LlamaCppTextAdapter {
    type Handle = LlamaCppTextHandle;

    fn supported_formats(&self) -> &[CheckpointFormat] {
        &LLAMA_CPP_TEXT_FORMATS
    }

    fn backend_kind(&self) -> BackendKind {
        BackendKind::LlamaCpp
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
        let model_path = resolve_checkpoint_model_path(checkpoint, resolved_quantization)?;
        let built = load_llama_model(model_path)?;

        Ok(new_text_handle(
            TextHandleConfig {
                id: identity.id.clone(),
                display_name: identity.display_name.clone(),
                capabilities: self.capabilities.clone(),
                quantization: self.quantization.clone(),
                resolved_quantization,
                arch: self.arch,
                context_length: self.default_context_length,
            },
            built,
        ))
    }
}

/// Q4 recommended, Q8 supported. Inputs are compile-time constants (Four ∈ [Four, Eight]).
/// On the unreachable error path, degrades to no-recommended rather than panicking.
fn curated_q4_q8_support() -> QuantizationSupport {
    QuantizationSupport::with_recommended(
        [QuantizationBits::Four, QuantizationBits::Eight],
        QuantizationBits::Four,
    )
    .unwrap_or_else(|e| {
        tracing::error!("curated quantization construction failed (this is a bug): {e}");
        QuantizationSupport::without_recommended([QuantizationBits::Four, QuantizationBits::Eight])
    })
}

/// Qwen3.6 currently has validated GGUF Q4/Q5/Q8 artifacts in the curated repo.
/// FP8 is intentionally not advertised until a real FP8 GGUF artifact is available.
fn curated_qwen36_gguf_support() -> QuantizationSupport {
    QuantizationSupport::with_recommended(
        [
            QuantizationBits::Four,
            QuantizationBits::Five,
            QuantizationBits::Eight,
        ],
        QuantizationBits::Five,
    )
    .unwrap_or_else(|e| {
        tracing::error!("curated quantization construction failed (this is a bug): {e}");
        QuantizationSupport::without_recommended([
            QuantizationBits::Four,
            QuantizationBits::Five,
            QuantizationBits::Eight,
        ])
    })
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
    type Handle = LlamaCppTextHandle;

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

        let built = build_llama_model(&self.spec, resolved_quantization, options)?;

        Ok(new_text_handle(
            TextHandleConfig {
                id: self.metadata.id.clone(),
                display_name: self.metadata.display_name.clone(),
                capabilities: self.metadata.capabilities.clone(),
                quantization: self.metadata.quantization.clone(),
                resolved_quantization,
                arch: self.spec.arch,
                context_length: self.spec.default_context_length,
            },
            built,
        ))
    }
}

// ---------------------------------------------------------------------------
// Internal runtime abstraction (enables stub testing without real llama.cpp)
// ---------------------------------------------------------------------------

#[cfg(test)]
struct StubTextRuntime;

#[cfg(test)]
impl StubTextRuntime {
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

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ModelError> {
        Ok(CompletionResponse {
            content: format!("llama-cpp stub completion of: {}", request.prompt),
        })
    }
}

enum TextRuntime {
    Real(LlamaCppRuntime),
    #[cfg(test)]
    Stub(StubTextRuntime),
}

impl TextRuntime {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ModelError> {
        match self {
            Self::Real(runtime) => runtime.chat(request).await,
            #[cfg(test)]
            Self::Stub(runtime) => runtime.chat(request).await,
        }
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ModelError> {
        match self {
            Self::Real(runtime) => runtime.complete(request).await,
            #[cfg(test)]
            Self::Stub(runtime) => runtime.complete(request).await,
        }
    }
}

struct LlamaCppRuntime {
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
    arch: LlamaCppTextArch,
    context_length: u32,
    metrics: Arc<Mutex<TextMetrics>>,
}

// SAFETY: The only non-Send/Sync field is `model: Arc<LlamaModel>` and
// `backend: Arc<LlamaBackend>`. LlamaModel weights are immutable after
// load_from_file(). All mutable state (LlamaContext, LlamaBatch,
// LlamaSampler) is created per-request inside spawn_blocking and never
// escapes the closure.
//
// INVARIANT: `LlamaCppRuntime` must not gain `LlamaContext`, `LlamaBatch`, or
// `LlamaSampler` fields. If per-request mutable llama.cpp state is ever stored
// on the struct, these unsafe impls must be revisited.
unsafe impl Send for LlamaCppRuntime {}
unsafe impl Sync for LlamaCppRuntime {}

impl LlamaCppRuntime {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse, ModelError> {
        let prompt = format_chat_prompt(self.arch, &request)?;
        self.generate_text(&prompt, &request.params).await
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, ModelError> {
        let chat_response = self.generate_text(&request.prompt, &request.params).await?;
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
        let backend = Arc::clone(&self.backend);
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

            let ctx_params =
                LlamaContextParams::default().with_n_ctx(std::num::NonZeroU32::new(context_length));

            let mut ctx = model.new_context(&backend, ctx_params).map_err(|e| {
                ModelError::BackendExecution {
                    backend: "llama-cpp",
                    operation: "new_context",
                    message: e.to_string(),
                }
            })?;

            let tokens = model.str_to_token(&prompt, AddBos::Always).map_err(|e| {
                ModelError::BackendExecution {
                    backend: "llama-cpp",
                    operation: "tokenize",
                    message: e.to_string(),
                }
            })?;

            if tokens.is_empty() {
                return Err(ModelError::BackendExecution {
                    backend: "llama-cpp",
                    operation: "tokenize",
                    message: "prompt tokenized to zero tokens".into(),
                });
            }

            let prompt_token_count = tokens.len() as u32;

            let mut batch = LlamaBatch::new(context_length as usize, 1);
            let last_idx = tokens.len() - 1;
            for (i, token) in tokens.iter().enumerate() {
                batch
                    .add(*token, i as i32, &[0], i == last_idx)
                    .map_err(|e| ModelError::BackendExecution {
                        backend: "llama-cpp",
                        operation: "batch_add",
                        message: e.to_string(),
                    })?;
            }

            ctx.decode(&mut batch)
                .map_err(|e| ModelError::BackendExecution {
                    backend: "llama-cpp",
                    operation: "decode_prompt",
                    message: e.to_string(),
                })?;

            // Use process-id-derived seed so temperature produces varied output
            // across requests while remaining reproducible within a process lifetime.
            let seed = std::process::id();
            let mut sampler = if let Some(top_p) = top_p {
                LlamaSampler::chain_simple([
                    LlamaSampler::top_p(top_p, 1),
                    LlamaSampler::temp(temperature),
                    LlamaSampler::dist(seed),
                ])
            } else {
                LlamaSampler::chain_simple([
                    LlamaSampler::temp(temperature),
                    LlamaSampler::dist(seed),
                ])
            };

            let mut generated_token_count: u32 = 0;
            let mut generated_text = String::new();
            let mut n_cur = tokens.len() as i32;
            let mut decoder = encoding_rs::UTF_8.new_decoder();

            for _ in 0..max_tokens {
                let token = sampler.sample(&ctx, batch.n_tokens() - 1);
                sampler.accept(token);

                if model.is_eog_token(token) {
                    break;
                }

                let piece = model
                    .token_to_piece(token, &mut decoder, false, None)
                    .map_err(|e| ModelError::BackendExecution {
                        backend: "llama-cpp",
                        operation: "token_to_piece",
                        message: e.to_string(),
                    })?;

                generated_text.push_str(&piece);
                generated_token_count += 1;

                // Check stop sequences on the tail of generated text.
                let should_stop = stop_sequences
                    .iter()
                    .any(|seq| generated_text.ends_with(seq));
                if should_stop {
                    for seq in &stop_sequences {
                        if generated_text.ends_with(seq) {
                            generated_text.truncate(generated_text.len() - seq.len());
                            break;
                        }
                    }
                    break;
                }

                batch.clear();
                batch
                    .add(token, n_cur, &[0], true)
                    .map_err(|e| ModelError::BackendExecution {
                        backend: "llama-cpp",
                        operation: "batch_add_token",
                        message: e.to_string(),
                    })?;
                n_cur += 1;

                ctx.decode(&mut batch)
                    .map_err(|e| ModelError::BackendExecution {
                        backend: "llama-cpp",
                        operation: "decode_token",
                        message: e.to_string(),
                    })?;
            }

            let elapsed = started_at.elapsed();

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
///
/// Returns an error if any message contains non-text content parts (images).
/// llama.cpp text-only backends do not support multimodal input.
fn format_chat_prompt(arch: LlamaCppTextArch, request: &ChatRequest) -> Result<String, ModelError> {
    match arch {
        LlamaCppTextArch::Qwen3 | LlamaCppTextArch::Qwen35 => {
            // Qwen35 currently shares the Qwen3 ChatML template; split this arm
            // if Qwen3.5 chat tokens or think-block handling diverge.
            format_qwen3_prompt(&request.messages)
        }
        LlamaCppTextArch::Gemma4 => format_gemma4_prompt(&request.messages),
    }
}

fn collect_text(message: &motlie_model::ChatMessage) -> Result<String, ModelError> {
    let mut text = String::new();
    for part in &message.content {
        match part {
            motlie_model::ContentPart::Text(t) => text.push_str(t),
            motlie_model::ContentPart::Image { .. }
            | motlie_model::ContentPart::ImageUrl { .. } => {
                return Err(ModelError::UnsupportedCapability(CapabilityKind::Vision));
            }
        }
    }
    Ok(text)
}

fn format_qwen3_prompt(messages: &[motlie_model::ChatMessage]) -> Result<String, ModelError> {
    let mut prompt = String::new();
    for msg in messages {
        let role = match msg.role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        };
        prompt.push_str(&format!("<|im_start|>{role}\n"));
        prompt.push_str(&collect_text(msg)?);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>assistant\n");
    Ok(prompt)
}

fn format_gemma4_prompt(messages: &[motlie_model::ChatMessage]) -> Result<String, ModelError> {
    let mut prompt = String::new();
    for msg in messages {
        let role = match msg.role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "model",
        };
        prompt.push_str(&format!("<start_of_turn>{role}\n"));
        prompt.push_str(&collect_text(msg)?);
        prompt.push_str("<end_of_turn>\n");
    }
    prompt.push_str("<start_of_turn>model\n");
    Ok(prompt)
}

// ---------------------------------------------------------------------------
// Handle
// ---------------------------------------------------------------------------

pub struct LlamaCppTextHandle {
    descriptor: LoadedBundleDescriptor,
    runtime: TextRuntime,
    metrics: Arc<Mutex<TextMetrics>>,
}

#[async_trait]
impl BundleHandle for LlamaCppTextHandle {
    type Chat = Self;
    type Completion = Self;
    type Embeddings = UnsupportedEmbeddings;

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

    fn chat(&self) -> Result<&Self::Chat, ModelError> {
        Ok(self)
    }

    fn completion(&self) -> Result<&Self::Completion, ModelError> {
        Ok(self)
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

struct TextHandleConfig {
    id: BundleId,
    display_name: String,
    capabilities: Capabilities,
    quantization: QuantizationSupport,
    resolved_quantization: Option<QuantizationBits>,
    arch: LlamaCppTextArch,
    context_length: u32,
}

fn new_text_handle(config: TextHandleConfig, built: BuiltModel) -> LlamaCppTextHandle {
    let metrics = Arc::new(Mutex::new(TextMetrics::default()));
    {
        let mut metrics = lock_metrics(&metrics, "llama-cpp-text-start");
        observe_memory(&mut metrics.runtime);
    }

    let TextHandleConfig {
        id,
        display_name,
        capabilities,
        quantization,
        resolved_quantization,
        arch,
        context_length,
    } = config;

    LlamaCppTextHandle {
        descriptor: LoadedBundleDescriptor {
            id,
            display_name,
            capabilities,
            quantization,
            resolved_quantization,
        },
        runtime: TextRuntime::Real(LlamaCppRuntime {
            backend: built.backend,
            model: built.model,
            arch,
            context_length,
            metrics: Arc::clone(&metrics),
        }),
        metrics,
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

struct BuiltModel {
    backend: Arc<LlamaBackend>,
    model: Arc<LlamaModel>,
}

fn build_llama_model(
    spec: &LlamaCppTextSpec,
    resolved_quantization: Option<QuantizationBits>,
    options: StartOptions,
) -> Result<BuiltModel, ModelError> {
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

    load_llama_model(model_path)
}

fn load_llama_model(model_path: PathBuf) -> Result<BuiltModel, ModelError> {
    let backend = LlamaBackend::init().map_err(|e| ModelError::BackendInitialization {
        backend: "llama-cpp",
        message: e.to_string(),
    })?;

    let mut model_params = LlamaModelParams::default();
    let n_gpu_layers = resolve_gpu_layers();
    model_params = model_params.with_n_gpu_layers(n_gpu_layers);

    let model = LlamaModel::load_from_file(&backend, model_path, &model_params).map_err(|e| {
        ModelError::BackendInitialization {
            backend: "llama-cpp",
            message: e.to_string(),
        }
    })?;

    Ok(BuiltModel {
        backend: Arc::new(backend),
        model: Arc::new(model),
    })
}

fn resolve_checkpoint_model_path(
    checkpoint: &ResolvedCheckpoint,
    resolved_quantization: Option<QuantizationBits>,
) -> Result<PathBuf, ModelError> {
    if checkpoint.checkpoint.format != CheckpointFormat::Gguf {
        return Err(ModelError::InvalidConfiguration(format!(
            "llama.cpp expected GGUF checkpoint, got {:?}",
            checkpoint.checkpoint.format
        )));
    }

    let expected_suffix = gguf_quant_suffix(resolved_quantization);
    let path = &checkpoint.path;

    if path.is_file() {
        let filename = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                ModelError::InvalidConfiguration(format!(
                    "resolved GGUF checkpoint path `{}` has no filename",
                    path.display()
                ))
            })?;
        if filename.ends_with(expected_suffix) {
            return Ok(path.clone());
        }
        return Err(ModelError::InvalidConfiguration(format!(
            "resolved GGUF checkpoint `{}` does not match requested quantization suffix `{expected_suffix}`",
            path.display()
        )));
    }

    let mut matches = std::fs::read_dir(path)
        .map_err(|e| {
            ModelError::InvalidConfiguration(format!(
                "failed to inspect GGUF checkpoint root `{}`: {e}",
                path.display()
            ))
        })?
        .filter_map(std::result::Result::ok)
        .map(|entry| entry.path())
        .filter(|candidate| candidate.is_file())
        .filter(|candidate| {
            candidate
                .file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.ends_with(expected_suffix))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();
    matches.sort();

    match matches.len() {
        1 => Ok(matches.remove(0)),
        0 => Err(ModelError::InvalidConfiguration(format!(
            "resolved GGUF checkpoint root `{}` does not contain a file matching `{expected_suffix}`",
            path.display()
        ))),
        count => Err(ModelError::InvalidConfiguration(format!(
            "resolved GGUF checkpoint root `{}` has {count} files matching `{expected_suffix}`; expected exactly one",
            path.display()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use motlie_model::{
        BackendAdapter, BackendKind, ChatMessage, ModelCheckpoint, QuantizationBits, StartOptions,
    };
    use std::time::{SystemTime, UNIX_EPOCH};

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
    fn qwen35_spec_has_expected_identity_and_quantization() {
        let spec = LlamaCppTextSpec::qwen3_6_27b();

        assert_eq!(spec.id.as_str(), "qwen3_6_27b_gguf");
        assert_eq!(spec.display_name, "Qwen3.6 27B (GGUF)");
        assert_eq!(spec.arch, LlamaCppTextArch::Qwen35);
        assert_eq!(
            spec.quantization.recommended(),
            Some(QuantizationBits::Five)
        );
        assert!(spec.quantization.supports(QuantizationBits::Four));
        assert!(spec.quantization.supports(QuantizationBits::Five));
        assert!(spec.quantization.supports(QuantizationBits::Eight));
        assert!(!spec.quantization.supports(QuantizationBits::FloatEight));
        assert!(spec.capabilities.supports(CapabilityKind::Chat));
        assert!(spec.capabilities.supports(CapabilityKind::Completion));
        assert!(!spec.capabilities.supports(CapabilityKind::Vision));
    }

    #[test]
    fn qwen3_adapter_reports_backend_metadata() {
        let adapter = LlamaCppTextAdapter::qwen3();

        assert_eq!(adapter.supported_formats(), &[CheckpointFormat::Gguf]);
        assert_eq!(adapter.backend_kind(), BackendKind::LlamaCpp);
        assert_eq!(adapter.capabilities(), &Capabilities::chat_and_completion());
        assert_eq!(
            adapter.quantization().recommended(),
            Some(QuantizationBits::Four)
        );
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
        assert_eq!(
            gguf_filename("qwen3-4b", Some(QuantizationBits::Five)),
            "qwen3-4b-Q5_K_M.gguf"
        );
        assert_eq!(
            gguf_filename("qwen3-4b", Some(QuantizationBits::FloatEight)),
            "qwen3-4b-FP8.gguf"
        );
        assert_eq!(gguf_filename("qwen3-4b", None), "qwen3-4b-f16.gguf");
    }

    #[test]
    fn qwen3_chat_template_formats_correctly() {
        let messages = vec![
            ChatMessage::new(ChatRole::System, "Be concise."),
            ChatMessage::new(ChatRole::User, "Hello"),
        ];
        let prompt = format_qwen3_prompt(&messages).expect("text-only messages should format");

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
        let prompt = format_gemma4_prompt(&messages).expect("text-only messages should format");

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
            runtime: TextRuntime::Stub(StubTextRuntime),
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

    #[test]
    fn image_content_parts_are_rejected_by_chat_template() {
        let request = ChatRequest {
            messages: vec![motlie_model::ChatMessage::with_parts(
                ChatRole::User,
                vec![motlie_model::ContentPart::image(vec![1, 2, 3], "image/png")],
            )],
            ..Default::default()
        };

        let error = format_chat_prompt(LlamaCppTextArch::Qwen3, &request)
            .expect_err("images should be rejected for text-only runtime");
        assert!(matches!(
            error,
            ModelError::UnsupportedCapability(CapabilityKind::Vision)
        ));
    }

    #[test]
    fn resolved_checkpoint_selects_matching_quantized_file() {
        let root = unique_temp_dir();
        let gguf = root.join("Qwen3-4B-Q4_K_M.gguf");
        std::fs::write(&gguf, b"stub").expect("test gguf should be writable");

        let checkpoint = ResolvedCheckpoint {
            checkpoint: ModelCheckpoint {
                format: CheckpointFormat::Gguf,
                source: motlie_model::ArtifactSource::HuggingFace {
                    repo: "Qwen/Qwen3-4B-GGUF",
                },
                include: vec![motlie_model::ArtifactRule::Suffix(".gguf")],
                quantization: None,
            },
            path: root.clone(),
        };

        let resolved = resolve_checkpoint_model_path(&checkpoint, Some(QuantizationBits::Four))
            .expect("Q4 GGUF should be selected");
        assert_eq!(resolved, gguf);
    }

    #[test]
    fn resolved_checkpoint_rejects_missing_quantized_file() {
        let root = unique_temp_dir();
        std::fs::write(root.join("Qwen3-4B-f16.gguf"), b"stub")
            .expect("test gguf should be writable");

        let checkpoint = ResolvedCheckpoint {
            checkpoint: ModelCheckpoint {
                format: CheckpointFormat::Gguf,
                source: motlie_model::ArtifactSource::HuggingFace {
                    repo: "Qwen/Qwen3-4B-GGUF",
                },
                include: vec![motlie_model::ArtifactRule::Suffix(".gguf")],
                quantization: None,
            },
            path: root.clone(),
        };

        let error = resolve_checkpoint_model_path(&checkpoint, Some(QuantizationBits::Four))
            .expect_err("missing Q4 GGUF should fail");
        assert!(matches!(
            error,
            ModelError::InvalidConfiguration(message)
                if message.contains("-Q4_K_M.gguf")
        ));
    }

    fn unique_temp_dir() -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be monotonic enough for tests")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("motlie-llama-cpp-checkpoint-{suffix}"));
        std::fs::create_dir_all(&path).expect("unique temp dir should be creatable");
        path
    }
}
