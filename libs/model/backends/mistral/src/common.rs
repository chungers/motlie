use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use mistralrs::core::StopTokens;
use mistralrs::core::Usage;
use mistralrs::{IsqBits, RequestBuilder, SamplingParams};
use motlie_model::{
    ArtifactPolicy, ArtifactSource, Bytes, ChatRole, CheckpointFormat, EmbeddingMetrics,
    GenerationParams, Milliseconds, ModelError, ModelMetricSnapshot, QuantizationBits,
    ResolvedCheckpoint, RuntimeMetrics, StartOptions, TextGenerationMetrics, Tokens,
    TokensPerSecond,
};

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

pub(crate) fn map_quantization_bits(bits: QuantizationBits) -> IsqBits {
    match bits {
        QuantizationBits::Four => IsqBits::Four,
        QuantizationBits::Eight => IsqBits::Eight,
        QuantizationBits::Five | QuantizationBits::FloatEight => {
            unreachable!("mistral.rs specs must reject unsupported quantization before mapping")
        }
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
    }
}

pub(crate) fn apply_generation_params(
    builder: RequestBuilder,
    params: &GenerationParams,
) -> RequestBuilder {
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
    builder.set_sampling(sampling)
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
    use sysinfo::{get_current_pid, ProcessesToUpdate, System};

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
