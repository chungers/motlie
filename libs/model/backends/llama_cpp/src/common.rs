use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use motlie_model::{
    ArtifactPolicy, Bytes, Milliseconds, ModelError, ModelMetricSnapshot, RuntimeMetrics,
    TextGenerationMetrics, Tokens, TokensPerSecond,
};

#[derive(Debug)]
pub(crate) struct ConfiguredGguf {
    pub(crate) model_path: PathBuf,
}

pub(crate) fn configure_artifact_policy(
    gguf_filename: &str,
    policy: ArtifactPolicy,
) -> Result<ConfiguredGguf, ModelError> {
    match policy {
        ArtifactPolicy::AllowFetch { root } => {
            let root = root.unwrap_or_else(|| PathBuf::from("."));
            Ok(ConfiguredGguf {
                model_path: root.join(gguf_filename),
            })
        }
        ArtifactPolicy::LocalOnly { root } => {
            let model_path = root.join(gguf_filename);
            if !model_path.exists() {
                return Err(ModelError::InvalidConfiguration(format!(
                    "GGUF artifact `{}` not found under `{}`",
                    gguf_filename,
                    root.display()
                )));
            }
            Ok(ConfiguredGguf { model_path })
        }
    }
}

pub(crate) fn should_force_cpu() -> bool {
    matches!(
        std::env::var("MOTLIE_MODEL_FORCE_CPU"),
        Ok(value) if matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES")
    )
}

pub(crate) fn lock_metrics<'a, T>(
    mutex: &'a Mutex<T>,
    context: &'static str,
) -> MutexGuard<'a, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            eprintln!(
                "warning: recovering poisoned metrics lock in `{context}`; continuing with potentially incomplete metric state"
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
    pub(crate) total_generation_time_msec: u128,
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
    runtime.peak_resident_memory_bytes = max_opt_u64(
        runtime.peak_resident_memory_bytes,
        resident_memory_bytes,
    );
}

pub(crate) fn observe_text_generation(
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
        .saturating_add((prompt_tokens + generated_tokens) as u64);
    state.total_generation_time_msec = state
        .total_generation_time_msec
        .saturating_add(duration_to_milliseconds(generation_time) as u128);
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
            avg_prompt_tokens_per_sec: None, // llama.cpp doesn't separate prompt/gen timing
            avg_generated_tokens_per_sec: aggregate_tokens_per_second(
                text.total_generated_tokens,
                text.total_generation_time_msec,
            )
            .map(TokensPerSecond),
        }),
        embeddings: None,
    }
}

fn duration_to_milliseconds(duration: Duration) -> u64 {
    duration.as_millis().min(u128::from(u64::MAX)) as u64
}

fn average_latency(runtime: &RuntimeMetricState) -> Option<Milliseconds> {
    if runtime.request_count == 0 {
        return None;
    }
    Some(Milliseconds(
        (runtime.total_latency_msec / runtime.request_count as u128)
            .min(u128::from(u64::MAX)) as u64,
    ))
}

fn aggregate_tokens_per_second(tokens: u64, total_time_msec: u128) -> Option<u64> {
    if tokens == 0 || total_time_msec == 0 {
        return None;
    }
    Some(
        ((tokens as u128 * 1000) / total_time_msec)
            .min(u128::from(u64::MAX)) as u64,
    )
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

    #[test]
    fn local_only_policy_rejects_missing_gguf() {
        let root = std::env::temp_dir().join("motlie-llama-cpp-test-missing");
        std::fs::create_dir_all(&root).ok();

        let err = configure_artifact_policy(
            "model-Q4_K_M.gguf",
            ArtifactPolicy::LocalOnly { root },
        )
        .expect_err("missing GGUF should fail");

        assert!(matches!(err, ModelError::InvalidConfiguration(msg) if msg.contains("not found")));
    }

    #[test]
    fn local_only_policy_accepts_existing_gguf() {
        let root = std::env::temp_dir().join("motlie-llama-cpp-test-exists");
        std::fs::create_dir_all(&root).ok();
        let gguf = root.join("model-Q4_K_M.gguf");
        std::fs::write(&gguf, b"stub").ok();

        let configured = configure_artifact_policy(
            "model-Q4_K_M.gguf",
            ArtifactPolicy::LocalOnly { root: root.clone() },
        )
        .expect("existing GGUF should succeed");

        assert_eq!(configured.model_path, gguf);
        std::fs::remove_file(gguf).ok();
    }

    #[test]
    fn text_metrics_aggregate_across_requests() {
        let mut runtime = RuntimeMetricState::default();
        let mut text = TextMetricState::default();

        observe_latency(&mut runtime, Duration::from_millis(100));
        observe_text_generation(&mut text, 10, 20, Duration::from_millis(80));

        observe_latency(&mut runtime, Duration::from_millis(200));
        observe_text_generation(&mut text, 30, 40, Duration::from_millis(150));

        let snapshot = snapshot_text_metrics(&runtime, &text);
        let rt = snapshot.runtime.expect("runtime metrics should be present");
        let tg = snapshot
            .text_generation
            .expect("text metrics should be present");

        assert_eq!(rt.request_count, Some(2));
        assert_eq!(tg.total_prompt_tokens, Some(Tokens(40)));
        assert_eq!(tg.total_generated_tokens, Some(Tokens(60)));
        assert_eq!(tg.total_tokens, Some(Tokens(100)));
        // 60 tokens in 230ms = ~260 tok/s
        assert_eq!(tg.avg_generated_tokens_per_sec, Some(TokensPerSecond(260)));
    }
}
