//! Shared runtime metric state and observation helpers for backend crates.
//!
//! Gated behind `metrics-runtime` feature to avoid pulling `sysinfo` into
//! builds that only need the contract types from `metrics`.

use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use crate::metrics::{
    EmbeddingMetrics, ModelMetricSnapshot, RuntimeMetrics, TextGenerationMetrics,
};
use crate::units::{Bytes, Milliseconds, Tokens, TokensPerSecond};

// ---------------------------------------------------------------------------
// Metric state types (mutable, backend-internal)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct RuntimeMetricState {
    pub request_count: u64,
    pub total_latency_msec: u128,
    pub last_latency_msec: Option<u64>,
    pub max_latency_msec: Option<u64>,
    pub peak_resident_memory_bytes: Option<u64>,
}

#[derive(Clone, Debug, Default)]
pub struct TextMetricState {
    pub total_prompt_tokens: u64,
    pub total_generated_tokens: u64,
    pub total_tokens: u64,
    pub total_prompt_time_msec: u128,
    pub total_generation_time_msec: u128,
}

#[derive(Clone, Debug, Default)]
pub struct EmbeddingMetricState {
    pub request_count: u64,
    pub input_count: u64,
}

// ---------------------------------------------------------------------------
// Observation helpers
// ---------------------------------------------------------------------------

pub fn observe_latency(runtime: &mut RuntimeMetricState, elapsed: Duration) {
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

pub fn observe_memory(runtime: &mut RuntimeMetricState) {
    let resident_memory_bytes = current_resident_memory_bytes();
    runtime.peak_resident_memory_bytes =
        max_opt_u64(runtime.peak_resident_memory_bytes, resident_memory_bytes);
}

pub fn observe_text_generation(
    state: &mut TextMetricState,
    prompt_tokens: u32,
    generated_tokens: u32,
    generation_time: Duration,
) {
    state.total_prompt_tokens = state
        .total_prompt_tokens
        .saturating_add(prompt_tokens as u64);
    state.total_generated_tokens = state
        .total_generated_tokens
        .saturating_add(generated_tokens as u64);
    state.total_tokens = state
        .total_tokens
        .saturating_add(prompt_tokens as u64)
        .saturating_add(generated_tokens as u64);
    state.total_generation_time_msec = state
        .total_generation_time_msec
        .saturating_add(duration_to_milliseconds(generation_time) as u128);
}

pub fn observe_embedding_request(
    runtime: &mut RuntimeMetricState,
    embeddings: &mut EmbeddingMetricState,
    elapsed: Duration,
    input_count: usize,
) {
    observe_latency(runtime, elapsed);
    embeddings.request_count = embeddings.request_count.saturating_add(1);
    embeddings.input_count = embeddings.input_count.saturating_add(input_count as u64);
}

// ---------------------------------------------------------------------------
// Snapshot builders
// ---------------------------------------------------------------------------

pub fn snapshot_text_metrics(
    runtime: &RuntimeMetricState,
    text: &TextMetricState,
) -> ModelMetricSnapshot {
    ModelMetricSnapshot {
        runtime: Some(build_runtime_metrics(runtime)),
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
                text.total_generation_time_msec,
            )
            .map(TokensPerSecond),
        }),
        embeddings: None,
    }
}

pub fn snapshot_embedding_metrics(
    runtime: &RuntimeMetricState,
    embeddings: &EmbeddingMetricState,
) -> ModelMetricSnapshot {
    ModelMetricSnapshot {
        runtime: Some(build_runtime_metrics(runtime)),
        text_generation: None,
        embeddings: Some(EmbeddingMetrics {
            request_count: Some(embeddings.request_count),
            input_count: Some(embeddings.input_count),
        }),
    }
}

fn build_runtime_metrics(runtime: &RuntimeMetricState) -> RuntimeMetrics {
    RuntimeMetrics {
        resident_memory: current_resident_memory_bytes().map(Bytes),
        peak_resident_memory: runtime.peak_resident_memory_bytes.map(Bytes),
        request_count: Some(runtime.request_count),
        last_latency: runtime.last_latency_msec.map(Milliseconds),
        max_latency: runtime.max_latency_msec.map(Milliseconds),
        avg_latency: average_latency(runtime),
    }
}

// ---------------------------------------------------------------------------
// Lock helper
// ---------------------------------------------------------------------------

pub fn lock_metrics<'a, T>(mutex: &'a Mutex<T>, context: &'static str) -> MutexGuard<'a, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::warn!(
                context,
                "recovering poisoned metrics lock; continuing with potentially incomplete metric state"
            );
            poisoned.into_inner()
        }
    }
}

// ---------------------------------------------------------------------------
// Env helpers
// ---------------------------------------------------------------------------

pub fn should_force_cpu() -> bool {
    matches!(
        std::env::var("MOTLIE_MODEL_FORCE_CPU"),
        Ok(value) if matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES")
    )
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

pub fn current_resident_memory_bytes() -> Option<u64> {
    use sysinfo::{get_current_pid, ProcessesToUpdate, System};

    let pid = get_current_pid().ok()?;
    let mut system = System::new();
    system.refresh_processes(ProcessesToUpdate::Some(&[pid]), false);
    let process = system.process(pid)?;
    Some(process.memory())
}

pub fn duration_to_milliseconds(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
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

fn max_opt_u64(lhs: Option<u64>, rhs: Option<u64>) -> Option<u64> {
    match (lhs, rhs) {
        (Some(lhs), Some(rhs)) => Some(lhs.max(rhs)),
        (Some(lhs), None) => Some(lhs),
        (None, Some(rhs)) => Some(rhs),
        (None, None) => None,
    }
}
