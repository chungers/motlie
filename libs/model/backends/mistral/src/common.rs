use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use either::Either;
use image::DynamicImage;
use indexmap::IndexMap;
use mistralrs::core::{StopTokens, Usage};
use mistralrs::{
    AudioInput, Constraint, CustomLogitsProcessor, Function, IsqBits, MessageContent,
    ModelCategory, RequestLike, RequestMessage, ResponseMessage, SamplingParams, Tool,
    ToolCallResponse, ToolChoice as MistralToolChoice, ToolType, VideoInput, WebSearchOptions,
};
#[cfg(test)]
use mistralrs::{CalledFunction, ToolCallType};
use motlie_model::{
    ArtifactPolicy, ArtifactSource, Bytes, CapabilityKind, ChatFinishReason, ChatMessage,
    ChatRequest, ChatResponse, ChatRole, CheckpointFormat, ContentPart, EmbeddingMetrics,
    GenerationParams, GenerationUsage, Milliseconds, ModelError, ModelMetricSnapshot,
    QuantizationBits, ResolvedCheckpoint, RuntimeMetrics, StartOptions, TextGenerationMetrics,
    Tokens, TokensPerSecond,
};
use serde_json::Value;

pub(crate) struct ConfiguredBuilder {
    pub(crate) model_target: String,
    pub(crate) hf_cache_root: Option<PathBuf>,
}

pub(crate) fn configure_artifact_policy(
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

pub(crate) fn resolve_local_checkpoint(
    checkpoint: &ResolvedCheckpoint,
    expected_format: CheckpointFormat,
    options: StartOptions,
) -> Result<(&'static str, StartOptions), ModelError> {
    if checkpoint.checkpoint.format != expected_format {
        return Err(ModelError::InvalidConfiguration(format!(
            "mistralrs expected {expected_format:?} checkpoint, got {:?}",
            checkpoint.checkpoint.format
        )));
    }

    let repo = match checkpoint.checkpoint.source {
        ArtifactSource::HuggingFace { repo } => repo,
    };

    Ok((
        repo,
        StartOptions {
            artifact_policy: Some(ArtifactPolicy::LocalOnly {
                root: checkpoint.path.clone(),
            }),
            ..options
        },
    ))
}

pub(crate) fn map_quantization_bits(bits: QuantizationBits) -> Result<IsqBits, ModelError> {
    match bits {
        QuantizationBits::Four => Ok(IsqBits::Four),
        QuantizationBits::Eight => Ok(IsqBits::Eight),
        other => Err(ModelError::InvalidConfiguration(format!(
            "mistral.rs backend does not support {other:?} quantization"
        ))),
    }
}

pub(crate) fn should_force_cpu() -> bool {
    matches!(
        std::env::var("MOTLIE_MODEL_FORCE_CPU"),
        Ok(value) if matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES")
    )
}

/// Returns the requested PagedAttention context size, if configured.
///
/// Set `MOTLIE_PAGED_ATTN_CONTEXT=N` where N is the max context length in tokens.
/// Omit the variable entirely to disable PagedAttention.
///
/// The old boolean `MOTLIE_PAGED_ATTN=1` is no longer supported because the
/// `PagedAttentionMetaBuilder` default `ContextSize(4096)` is too small for
/// non-trivial inputs and causes silent channel-closed crashes.
pub(crate) fn paged_attn_context_size() -> Option<usize> {
    std::env::var("MOTLIE_PAGED_ATTN_CONTEXT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&size| size > 0)
}

pub(crate) fn map_chat_role(role: ChatRole) -> mistralrs::TextMessageRole {
    match role {
        ChatRole::System => mistralrs::TextMessageRole::System,
        ChatRole::User => mistralrs::TextMessageRole::User,
        ChatRole::Assistant => mistralrs::TextMessageRole::Assistant,
        ChatRole::Tool => mistralrs::TextMessageRole::Tool,
    }
}

pub(crate) fn sampling_params_from_generation_params(params: &GenerationParams) -> SamplingParams {
    let mut sampling = SamplingParams::deterministic();
    if let Some(temperature) = params.temperature {
        sampling.temperature = Some(temperature as f64);
        sampling.top_k = None;
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
    sampling
}

pub(crate) fn collect_tools(
    request: &motlie_model::ChatRequest,
) -> Result<(Vec<Tool>, MistralToolChoice), ModelError> {
    if request.tools.is_empty() {
        return match &request.tool_choice {
            None | Some(motlie_model::ToolChoice::None) => {
                Ok((Vec::new(), MistralToolChoice::Auto))
            }
            Some(_) => Err(ModelError::InvalidConfiguration(
                "`tool_choice` requires at least one tool".into(),
            )),
        };
    }

    let tools = request
        .tools
        .iter()
        .map(motlie_tool_spec_to_mistral)
        .collect::<Result<Vec<_>, _>>()?;
    let tool_choice = motlie_tool_choice_to_mistral(request.tool_choice.as_ref(), &request.tools)?;

    Ok((tools, tool_choice))
}

#[derive(Clone, Debug)]
struct PendingImagePrefix {
    message_index: usize,
    image_indices: Vec<usize>,
}

#[derive(Clone)]
pub(crate) struct MotlieMistralRequest {
    messages: Vec<IndexMap<String, MessageContent>>,
    images: Vec<DynamicImage>,
    sampling_params: SamplingParams,
    tools: Vec<Tool>,
    tool_choice: MistralToolChoice,
    enable_thinking: Option<bool>,
    pending_image_prefixes: Vec<PendingImagePrefix>,
}

impl MotlieMistralRequest {
    fn new(
        sampling_params: SamplingParams,
        tools: Vec<Tool>,
        tool_choice: MistralToolChoice,
        enable_thinking: Option<bool>,
    ) -> Self {
        Self {
            messages: Vec::new(),
            images: Vec::new(),
            sampling_params,
            tools,
            tool_choice,
            enable_thinking,
            pending_image_prefixes: Vec::new(),
        }
    }

    fn add_text_message(&mut self, role: ChatRole, text: String) {
        self.messages.push(text_message(map_chat_role(role), text));
    }

    fn add_image_message(&mut self, role: ChatRole, text: String, images: Vec<DynamicImage>) {
        if images.is_empty() {
            self.add_text_message(role, text);
            return;
        }

        let image_start = self.images.len();
        let image_count = images.len();
        self.images.extend(images);
        let image_indices = (image_start..image_start + image_count).collect::<Vec<_>>();

        let mut content = Vec::with_capacity(image_count + 1);
        for _ in 0..image_count {
            content.push(IndexMap::from([(
                "type".to_string(),
                Value::String("image".to_string()),
            )]));
        }
        content.push(IndexMap::from([
            ("type".to_string(), Value::String("text".to_string())),
            ("text".to_string(), Value::String(text)),
        ]));

        let message_index = self.messages.len();
        self.messages.push(IndexMap::from([
            (
                "role".to_string(),
                Either::Left(map_chat_role(role).to_string()),
            ),
            ("content".to_string(), Either::Right(content)),
        ]));
        self.pending_image_prefixes.push(PendingImagePrefix {
            message_index,
            image_indices,
        });
    }

    fn add_assistant_tool_calls(
        &mut self,
        text: String,
        tool_calls: &[motlie_model::ToolCall],
    ) -> Result<(), ModelError> {
        let tool_calls = tool_calls
            .iter()
            .map(tool_call_message_value)
            .collect::<Result<Vec<_>, _>>()?;
        self.messages.push(IndexMap::from([
            (
                "role".to_string(),
                Either::Left(mistralrs::TextMessageRole::Assistant.to_string()),
            ),
            ("content".to_string(), Either::Left(text)),
            ("tool_calls".to_string(), Either::Right(tool_calls)),
        ]));
        Ok(())
    }

    fn add_tool_result(
        &mut self,
        text: String,
        tool_call_id: &motlie_model::ToolCallId,
        name: Option<&motlie_model::ToolName>,
    ) {
        let mut message = IndexMap::from([
            (
                "role".to_string(),
                Either::Left(mistralrs::TextMessageRole::Tool.to_string()),
            ),
            ("content".to_string(), Either::Left(text)),
            (
                "tool_call_id".to_string(),
                Either::Left(tool_call_id.as_str().to_string()),
            ),
        ]);
        if let Some(name) = name {
            message.insert("name".to_string(), Either::Left(name.as_str().to_string()));
        }
        self.messages.push(message);
    }
}

impl RequestLike for MotlieMistralRequest {
    fn messages_ref(&self) -> &[IndexMap<String, MessageContent>] {
        &self.messages
    }

    fn images_ref(&self) -> &[DynamicImage] {
        &self.images
    }

    fn take_messages(&mut self) -> RequestMessage {
        let mut messages = Vec::new();
        std::mem::swap(&mut messages, &mut self.messages);

        if self.images.is_empty() {
            RequestMessage::Chat {
                messages,
                enable_thinking: self.enable_thinking,
                reasoning_effort: None,
            }
        } else {
            let mut images = Vec::new();
            std::mem::swap(&mut images, &mut self.images);
            RequestMessage::MultimodalChat {
                images,
                audios: Vec::<AudioInput>::new(),
                videos: Vec::<VideoInput>::new(),
                messages,
                enable_thinking: self.enable_thinking,
                reasoning_effort: None,
            }
        }
    }

    fn take_logits_processors(&mut self) -> Option<Vec<Arc<dyn CustomLogitsProcessor>>> {
        None
    }

    fn take_adapters(&mut self) -> Option<Vec<String>> {
        None
    }

    fn return_logprobs(&self) -> bool {
        false
    }

    fn enable_search(&self) -> Option<bool> {
        None
    }

    fn take_constraint(&mut self) -> Constraint {
        Constraint::None
    }

    fn take_tools(&mut self) -> Option<(Vec<Tool>, MistralToolChoice)> {
        if self.tools.is_empty() {
            None
        } else {
            let mut tools = Vec::new();
            std::mem::swap(&mut tools, &mut self.tools);
            let mut tool_choice = MistralToolChoice::Auto;
            std::mem::swap(&mut tool_choice, &mut self.tool_choice);
            Some((tools, tool_choice))
        }
    }

    fn take_sampling_params(&mut self) -> SamplingParams {
        let mut sampling = SamplingParams::deterministic();
        std::mem::swap(&mut sampling, &mut self.sampling_params);
        sampling
    }

    fn take_web_search_options(&mut self) -> Option<WebSearchOptions> {
        None
    }

    fn resolve_pending_prefixes(&mut self, category: &ModelCategory) {
        let prefixer = match category {
            ModelCategory::Multimodal { prefixer } => prefixer,
            _ => {
                self.pending_image_prefixes.clear();
                return;
            }
        };

        for pending in self.pending_image_prefixes.drain(..) {
            let Some(message) = self.messages.get_mut(pending.message_index) else {
                continue;
            };
            let Some(Either::Right(content)) = message.get_mut("content") else {
                continue;
            };
            for part in content {
                let is_text = part
                    .get("type")
                    .is_some_and(|kind| kind == &Value::String("text".to_string()));
                if !is_text {
                    continue;
                }
                if let Some(Value::String(text)) = part.get_mut("text") {
                    *text = prefixer.prefix_image(pending.image_indices.clone(), text);
                }
                break;
            }
        }
    }
}

fn text_message(
    role: mistralrs::TextMessageRole,
    text: String,
) -> IndexMap<String, MessageContent> {
    IndexMap::from([
        ("role".to_string(), Either::Left(role.to_string())),
        ("content".to_string(), Either::Left(text)),
    ])
}

fn tool_call_message_value(
    call: &motlie_model::ToolCall,
) -> Result<IndexMap<String, Value>, ModelError> {
    let arguments = call
        .arguments
        .parse::<Value>()
        .map_err(|err| ModelError::InvalidConfiguration(err.to_string()))?;
    Ok(IndexMap::from([
        (
            "id".to_string(),
            Value::String(call.id.as_str().to_string()),
        ),
        ("type".to_string(), Value::String("function".to_string())),
        (
            "function".to_string(),
            serde_json::json!({
                "name": call.name.as_str(),
                "arguments": arguments,
            }),
        ),
    ]))
}

fn enable_thinking_for_request(request: &ChatRequest) -> Option<bool> {
    match request.thinking {
        Some(motlie_model::ThinkingMode::Disabled) => Some(false),
        Some(motlie_model::ThinkingMode::Auto) => Some(true),
        // Qwen3/Gemma safetensors templates default to thinking when the
        // variable is omitted. Tool loops need structured calls inside the
        // request token budget, so default tool-bearing requests to no thinking
        // unless the caller explicitly opts in.
        None if request.requires_tool_use() => Some(false),
        None => None,
    }
}

#[derive(Debug)]
pub struct MistralMessageParts {
    pub(crate) text: String,
    pub(crate) images: Vec<DynamicImage>,
}

pub(crate) fn text_only_message_parts(
    message: &ChatMessage,
) -> Result<MistralMessageParts, ModelError> {
    let mut text = String::new();
    for part in &message.content {
        match part {
            ContentPart::Text(part) => text.push_str(part),
            ContentPart::Image { .. } | ContentPart::ImageUrl { .. } => {
                return Err(ModelError::UnsupportedCapability(CapabilityKind::Vision));
            }
        }
    }

    Ok(MistralMessageParts {
        text,
        images: Vec::new(),
    })
}

pub(crate) fn multimodal_message_parts(
    message: &ChatMessage,
) -> Result<MistralMessageParts, ModelError> {
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

    Ok(MistralMessageParts { text, images })
}

pub(crate) fn chat_request_to_builder<F>(
    request: &ChatRequest,
    collect_parts: F,
) -> Result<MotlieMistralRequest, ModelError>
where
    F: Fn(&ChatMessage) -> Result<MistralMessageParts, ModelError>,
{
    let (tools, tool_choice) = collect_tools(request)?;
    let mut builder = MotlieMistralRequest::new(
        sampling_params_from_generation_params(&request.params),
        tools,
        tool_choice,
        enable_thinking_for_request(request),
    );
    for message in &request.messages {
        message
            .validate_tool_metadata()
            .map_err(|err| ModelError::InvalidConfiguration(err.to_string()))?;
        let MistralMessageParts { text, images } = collect_parts(message)?;
        match message.role {
            ChatRole::Tool => {
                if !images.is_empty() {
                    return Err(ModelError::InvalidConfiguration(
                        "tool result messages cannot carry image content".into(),
                    ));
                }
                let tool_call_id = message
                    .tool_call_id
                    .as_ref()
                    .expect("validated tool messages carry a tool call id");
                builder.add_tool_result(text, tool_call_id, message.name.as_ref());
            }
            ChatRole::Assistant if !message.tool_calls.is_empty() => {
                if !images.is_empty() {
                    return Err(ModelError::InvalidConfiguration(
                        "assistant tool-call replay messages cannot carry image content".into(),
                    ));
                }
                builder.add_assistant_tool_calls(text, &message.tool_calls)?;
            }
            _ if images.is_empty() => builder.add_text_message(message.role, text),
            _ => builder.add_image_message(message.role, text, images),
        };
    }

    Ok(builder)
}

pub(crate) fn motlie_tool_spec_to_mistral(
    spec: &motlie_model::ToolSpec,
) -> Result<Tool, ModelError> {
    let parameters = spec
        .input_schema
        .to_json_value()
        .map_err(|err| ModelError::InvalidConfiguration(err.to_string()))?;
    let parameters = serde_json::from_value::<std::collections::HashMap<String, Value>>(parameters)
        .map_err(|err| {
            ModelError::InvalidConfiguration(format!(
                "failed to convert tool schema for `{}` to mistral.rs parameters: {err}",
                spec.name
            ))
        })?;

    Ok(Tool {
        tp: ToolType::Function,
        function: Function {
            description: Some(spec.description.clone()),
            name: spec.name.as_str().to_string(),
            parameters: Some(parameters),
            strict: None,
        },
    })
}

#[cfg(test)]
pub(crate) fn motlie_tool_call_to_mistral(
    index: usize,
    call: &motlie_model::ToolCall,
) -> ToolCallResponse {
    ToolCallResponse {
        index,
        id: call.id.as_str().to_string(),
        tp: ToolCallType::Function,
        function: CalledFunction {
            name: call.name.as_str().to_string(),
            arguments: call.arguments.raw_json_str().to_string(),
        },
    }
}

pub(crate) fn mistral_response_to_chat_response(
    message: ResponseMessage,
    finish_reason: String,
    usage: &Usage,
) -> Result<ChatResponse, ModelError> {
    let tool_calls = message
        .tool_calls
        .unwrap_or_default()
        .into_iter()
        .map(mistral_tool_call_to_motlie)
        .collect::<Result<Vec<_>, _>>()?;

    let content = message.content.unwrap_or_default();
    if content.is_empty() && tool_calls.is_empty() {
        return Err(ModelError::BackendExecution {
            backend: "mistralrs",
            operation: "send_chat_request",
            message: "response contained neither text content nor tool calls".into(),
        });
    }

    let finish_reason = if !tool_calls.is_empty() {
        Some(ChatFinishReason::ToolCalls)
    } else {
        map_finish_reason(&finish_reason)
    };

    Ok(ChatResponse {
        content,
        tool_calls,
        finish_reason,
        reasoning: message.reasoning_content,
        usage: Some(GenerationUsage {
            prompt_tokens: Some(usage_count_to_u32(usage.prompt_tokens)),
            completion_tokens: Some(usage_count_to_u32(usage.completion_tokens)),
            total_tokens: Some(usage_count_to_u32(usage.total_tokens)),
        }),
    })
}

fn motlie_tool_choice_to_mistral(
    choice: Option<&motlie_model::ToolChoice>,
    specs: &[motlie_model::ToolSpec],
) -> Result<MistralToolChoice, ModelError> {
    match choice {
        None | Some(motlie_model::ToolChoice::Auto) => Ok(MistralToolChoice::Auto),
        Some(motlie_model::ToolChoice::None) => Ok(MistralToolChoice::None),
        Some(motlie_model::ToolChoice::Required) => Err(ModelError::InvalidConfiguration(
            "mistral.rs backend does not expose required tool-choice enforcement".into(),
        )),
        Some(motlie_model::ToolChoice::Named(name)) => {
            let spec = specs
                .iter()
                .find(|spec| spec.name.as_str() == name.as_str())
                .ok_or_else(|| {
                    ModelError::InvalidConfiguration(format!(
                        "named tool choice `{name}` does not match a request tool"
                    ))
                })?;
            Ok(MistralToolChoice::Tool(motlie_tool_spec_to_mistral(spec)?))
        }
    }
}

fn mistral_tool_call_to_motlie(
    call: ToolCallResponse,
) -> Result<motlie_model::ToolCall, ModelError> {
    motlie_model::ToolCall::from_json_args(call.id, call.function.name, call.function.arguments)
        .map_err(|err| ModelError::BackendExecution {
            backend: "mistralrs",
            operation: "parse_tool_call",
            message: err.to_string(),
        })
}

fn map_finish_reason(reason: &str) -> Option<ChatFinishReason> {
    match reason {
        "" => None,
        "stop" => Some(ChatFinishReason::Stop),
        "length" => Some(ChatFinishReason::Length),
        "tool_calls" => Some(ChatFinishReason::ToolCalls),
        "content_filter" => Some(ChatFinishReason::ContentFilter),
        other => Some(ChatFinishReason::Other(other.to_string())),
    }
}

fn usage_count_to_u32(count: usize) -> u32 {
    count.min(u32::MAX as usize) as u32
}

pub(crate) fn lock_metrics<'a, T>(mutex: &'a Mutex<T>, context: &'static str) -> MutexGuard<'a, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::warn!(
                "recovering poisoned metrics lock in `{context}`; continuing with potentially incomplete metric state"
            );
            poisoned.into_inner()
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct RuntimeMetricState {
    pub(crate) request_count: u64,
    pub(crate) total_latency_msec: u128,
    pub(crate) last_latency_msec: Option<u64>,
    pub(crate) max_latency_msec: Option<u64>,
    pub(crate) peak_resident_memory_bytes: Option<u64>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct TextMetricState {
    pub(crate) total_prompt_tokens: u64,
    pub(crate) total_generated_tokens: u64,
    pub(crate) total_tokens: u64,
    pub(crate) total_prompt_time_msec: u128,
    pub(crate) total_generated_time_msec: u128,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct EmbeddingMetricState {
    pub(crate) request_count: u64,
    pub(crate) input_count: u64,
}

pub(crate) fn observe_latency(runtime: &mut RuntimeMetricState, elapsed: Duration) {
    let elapsed_msec = duration_to_milliseconds(elapsed);
    runtime.request_count = runtime.request_count.saturating_add(1);
    runtime.total_latency_msec = runtime
        .total_latency_msec
        .saturating_add(elapsed_msec as u128);
    runtime.last_latency_msec = Some(elapsed_msec);
    runtime.max_latency_msec = Some(
        runtime
            .max_latency_msec
            .map(|current| current.max(elapsed_msec))
            .unwrap_or(elapsed_msec),
    );
    observe_memory(runtime);
}

pub(crate) fn observe_memory(runtime: &mut RuntimeMetricState) {
    let resident_memory_bytes = current_resident_memory_bytes();
    runtime.peak_resident_memory_bytes =
        max_opt_u64(runtime.peak_resident_memory_bytes, resident_memory_bytes);
}

pub(crate) fn observe_text_usage(state: &mut TextMetricState, usage: &Usage) {
    state.total_prompt_tokens = state
        .total_prompt_tokens
        .saturating_add(usage.prompt_tokens as u64);
    state.total_generated_tokens = state
        .total_generated_tokens
        .saturating_add(usage.completion_tokens as u64);
    state.total_tokens = state.total_tokens.saturating_add(usage.total_tokens as u64);
    state.total_prompt_time_msec = state
        .total_prompt_time_msec
        .saturating_add(seconds_to_milliseconds(usage.total_prompt_time_sec));
    state.total_generated_time_msec = state
        .total_generated_time_msec
        .saturating_add(seconds_to_milliseconds(usage.total_completion_time_sec));
}

pub(crate) fn observe_embedding_request(
    runtime: &mut RuntimeMetricState,
    embeddings: &mut EmbeddingMetricState,
    elapsed: Duration,
    input_count: usize,
) {
    observe_latency(runtime, elapsed);
    embeddings.request_count = embeddings.request_count.saturating_add(1);
    embeddings.input_count = embeddings.input_count.saturating_add(input_count as u64);
}

pub(crate) fn snapshot_text_metrics(
    runtime: &RuntimeMetricState,
    text: &TextMetricState,
) -> ModelMetricSnapshot {
    ModelMetricSnapshot {
        runtime: Some(RuntimeMetrics {
            resident_memory: current_resident_memory_bytes().map(Bytes),
            peak_resident_memory: runtime.peak_resident_memory_bytes.map(Bytes),
            request_count: Some(runtime.request_count),
            last_latency: runtime.last_latency_msec.map(Milliseconds),
            max_latency: runtime.max_latency_msec.map(Milliseconds),
            avg_latency: average_latency(runtime),
        }),
        text_generation: Some(TextGenerationMetrics {
            total_prompt_tokens: Some(Tokens(text.total_prompt_tokens)),
            total_generated_tokens: Some(Tokens(text.total_generated_tokens)),
            total_tokens: Some(Tokens(text.total_tokens)),
            avg_prompt_tokens_per_sec: aggregate_tokens_per_second(
                text.total_prompt_tokens,
                text.total_prompt_time_msec,
            )
            .map(TokensPerSecond),
            avg_generated_tokens_per_sec: aggregate_tokens_per_second(
                text.total_generated_tokens,
                text.total_generated_time_msec,
            )
            .map(TokensPerSecond),
        }),
        embeddings: None,
    }
}

pub(crate) fn snapshot_embedding_metrics(
    runtime: &RuntimeMetricState,
    embeddings: &EmbeddingMetricState,
) -> ModelMetricSnapshot {
    ModelMetricSnapshot {
        runtime: Some(RuntimeMetrics {
            resident_memory: current_resident_memory_bytes().map(Bytes),
            peak_resident_memory: runtime.peak_resident_memory_bytes.map(Bytes),
            request_count: Some(runtime.request_count),
            last_latency: runtime.last_latency_msec.map(Milliseconds),
            max_latency: runtime.max_latency_msec.map(Milliseconds),
            avg_latency: average_latency(runtime),
        }),
        text_generation: None,
        embeddings: Some(EmbeddingMetrics {
            request_count: Some(embeddings.request_count),
            input_count: Some(embeddings.input_count),
        }),
    }
}

fn duration_to_milliseconds(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

fn seconds_to_milliseconds(seconds: f32) -> u128 {
    if !seconds.is_finite() || seconds <= 0.0 {
        return 0;
    }
    (seconds as f64 * 1000.0)
        .round()
        .clamp(0.0, u64::MAX as f64) as u128
}

fn average_latency(runtime: &RuntimeMetricState) -> Option<Milliseconds> {
    if runtime.request_count == 0 {
        return None;
    }
    Some(Milliseconds(
        (runtime.total_latency_msec / runtime.request_count as u128).min(u128::from(u64::MAX))
            as u64,
    ))
}

fn aggregate_tokens_per_second(tokens: u64, total_time_msec: u128) -> Option<u64> {
    if tokens == 0 || total_time_msec == 0 {
        return None;
    }
    Some(((tokens as u128 * 1000) / total_time_msec).min(u128::from(u64::MAX)) as u64)
}

fn current_resident_memory_bytes() -> Option<u64> {
    use sysinfo::{ProcessesToUpdate, System, get_current_pid};

    let pid = get_current_pid().ok()?;
    let mut system = System::new();
    system.refresh_processes(ProcessesToUpdate::Some(&[pid]), false);
    let process = system.process(pid)?;
    Some(process.memory())
}

fn max_opt_u64(lhs: Option<u64>, rhs: Option<u64>) -> Option<u64> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => Some(lhs.max(rhs)),
        (Some(lhs), None) => Some(lhs),
        (None, Some(rhs)) => Some(rhs),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mistralrs::core::Usage;

    fn usage() -> Usage {
        Usage {
            completion_tokens: 7,
            prompt_tokens: 11,
            total_tokens: 18,
            avg_tok_per_sec: 0.0,
            avg_prompt_tok_per_sec: 0.0,
            avg_compl_tok_per_sec: 0.0,
            total_time_sec: 0.0,
            total_prompt_time_sec: 0.0,
            total_completion_time_sec: 0.0,
        }
    }

    #[test]
    fn tool_spec_maps_to_mistral_tool_schema() {
        let spec = motlie_model::ToolSpec::from_json_schema(
            "get_weather",
            "Get weather.",
            r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#,
        )
        .expect("schema should be valid");

        let tool = motlie_tool_spec_to_mistral(&spec).expect("tool should map");

        assert_eq!(tool.function.name, "get_weather");
        assert_eq!(tool.function.description.as_deref(), Some("Get weather."));
        assert_eq!(
            tool.function
                .parameters
                .as_ref()
                .and_then(|params| params.get("type"))
                .and_then(Value::as_str),
            Some("object")
        );
    }

    #[test]
    fn tool_call_round_trips_between_contract_and_mistral_shape() {
        let call = motlie_model::ToolCall::from_json_args(
            "call-1",
            "get_weather",
            r#"{"city":"Seattle"}"#,
        )
        .expect("tool call should be valid");

        let mistral_call = motlie_tool_call_to_mistral(0, &call);
        let mapped = mistral_tool_call_to_motlie(mistral_call).expect("call should map back");

        assert_eq!(mapped.id.as_str(), "call-1");
        assert_eq!(mapped.name.as_str(), "get_weather");
        assert_eq!(mapped.arguments.raw_json_str(), r#"{"city":"Seattle"}"#);
    }

    #[test]
    fn chat_request_builder_uses_template_tool_call_transcript_shape() {
        let call = motlie_model::ToolCall::from_json_args(
            "call-1",
            "get_weather",
            r#"{"city":"Seattle"}"#,
        )
        .expect("tool call should be valid");
        let request = ChatRequest {
            messages: vec![
                ChatMessage::text(ChatRole::User, "weather?"),
                ChatMessage::assistant_tool_calls(vec![call]),
                ChatMessage::tool_result("call-1", "get_weather", r#"{"temp":72}"#)
                    .expect("tool result should be valid"),
            ],
            tools: vec![motlie_model::ToolSpec::from_json_schema(
                "get_weather",
                "Get weather.",
                r#"{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}"#,
            )
            .expect("schema should be valid")],
            tool_choice: Some(motlie_model::ToolChoice::Auto),
            ..Default::default()
        };

        let builder =
            chat_request_to_builder(&request, text_only_message_parts).expect("request should map");
        let mut request_like = builder.clone();
        let messages = builder.messages_ref();

        assert_eq!(messages.len(), 3);
        assert!(messages[1].contains_key("tool_calls"));
        assert!(!messages[1].contains_key("function"));

        let Some(Either::Right(tool_calls)) = messages[1].get("tool_calls") else {
            panic!("assistant replay should carry structured tool_calls");
        };
        let function = tool_calls[0]
            .get("function")
            .and_then(Value::as_object)
            .expect("tool call function should be an object");
        assert_eq!(
            function.get("name").and_then(Value::as_str),
            Some("get_weather")
        );
        assert_eq!(
            function
                .get("arguments")
                .and_then(|arguments| arguments.get("city"))
                .and_then(Value::as_str),
            Some("Seattle")
        );

        assert_eq!(
            messages[2].get("tool_call_id"),
            Some(&Either::Left("call-1".to_string()))
        );
        assert_eq!(
            messages[2].get("name"),
            Some(&Either::Left("get_weather".to_string()))
        );
        assert_eq!(enable_thinking_for_request(&request), Some(false));
        let RequestMessage::Chat {
            enable_thinking, ..
        } = request_like.take_messages()
        else {
            panic!("text-only tool request should remain a chat request");
        };
        assert_eq!(enable_thinking, Some(false));

        let explicit_thinking_request = ChatRequest {
            thinking: Some(motlie_model::ThinkingMode::Auto),
            ..request.clone()
        };
        assert_eq!(
            enable_thinking_for_request(&explicit_thinking_request),
            Some(true)
        );
    }

    #[test]
    fn response_with_tool_calls_maps_to_chat_response() {
        let message = ResponseMessage {
            content: None,
            role: "assistant".to_string(),
            tool_calls: Some(vec![ToolCallResponse {
                index: 0,
                id: "call-1".to_string(),
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: "get_weather".to_string(),
                    arguments: r#"{"city":"Seattle"}"#.to_string(),
                },
            }]),
            reasoning_content: Some("thinking".to_string()),
        };

        let response =
            mistral_response_to_chat_response(message, "tool_calls".to_string(), &usage())
                .expect("response should map");

        assert_eq!(response.finish_reason, Some(ChatFinishReason::ToolCalls));
        assert_eq!(response.reasoning.as_deref(), Some("thinking"));
        assert_eq!(response.tool_calls.len(), 1);
        assert_eq!(
            response.usage.as_ref().and_then(|usage| usage.total_tokens),
            Some(18)
        );
    }

    #[test]
    fn text_metrics_aggregate_tokens_per_second_across_requests() {
        let mut runtime = RuntimeMetricState::default();
        let mut text = TextMetricState::default();

        observe_latency(&mut runtime, Duration::from_millis(100));
        observe_text_usage(
            &mut text,
            &Usage {
                completion_tokens: 20,
                prompt_tokens: 10,
                total_tokens: 30,
                avg_tok_per_sec: 75.0,
                avg_prompt_tok_per_sec: 50.0,
                avg_compl_tok_per_sec: 100.0,
                total_time_sec: 0.3,
                total_prompt_time_sec: 0.2,
                total_completion_time_sec: 0.2,
            },
        );

        observe_latency(&mut runtime, Duration::from_millis(200));
        observe_text_usage(
            &mut text,
            &Usage {
                completion_tokens: 30,
                prompt_tokens: 30,
                total_tokens: 60,
                avg_tok_per_sec: 300.0,
                avg_prompt_tok_per_sec: 300.0,
                avg_compl_tok_per_sec: 300.0,
                total_time_sec: 0.2,
                total_prompt_time_sec: 0.1,
                total_completion_time_sec: 0.1,
            },
        );

        let snapshot = snapshot_text_metrics(&runtime, &text);
        let text_metrics = snapshot
            .text_generation
            .expect("text metrics should be present");

        assert_eq!(text_metrics.total_prompt_tokens, Some(Tokens(40)));
        assert_eq!(text_metrics.total_generated_tokens, Some(Tokens(50)));
        assert_eq!(text_metrics.total_tokens, Some(Tokens(90)));
        assert_eq!(
            text_metrics.avg_prompt_tokens_per_sec,
            Some(TokensPerSecond(133))
        );
        assert_eq!(
            text_metrics.avg_generated_tokens_per_sec,
            Some(TokensPerSecond(166))
        );
    }
}
