#[cfg(target_os = "linux")]
use std::fs;

use serde::{Deserialize, Serialize};
use sysinfo::{get_current_pid, ProcessesToUpdate, System};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub startup_ms: Option<u64>,
    #[serde(default)]
    pub warmup_ms: Option<u64>,
    pub request_latencies_ms: Vec<u64>,
    #[serde(default)]
    pub unavailable: Vec<MetricUnavailable>,
    pub capability_metrics: CapabilityPerformanceMetrics,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
#[serde(tag = "capability", rename_all = "snake_case")]
pub enum CapabilityPerformanceMetrics {
    #[default]
    NotMeasured,
    Embeddings(EmbeddingPerformanceMetrics),
    Chat(ChatPerformanceMetrics),
    ToolUse(ToolUsePerformanceMetrics),
    Asr(AsrPerformanceMetrics),
    Tts(TtsPerformanceMetrics),
    Perf(PerfPerformanceMetrics),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingPerformanceMetrics {
    pub custom_embedding_latency_ms: Option<u64>,
    pub similar_pair_latency_ms: Option<u64>,
    pub dissimilar_pair_latency_ms: Option<u64>,
    pub embedding_dimensions: Option<usize>,
    pub vectors_per_second: Option<f64>,
    pub similar_cosine: Option<f64>,
    pub dissimilar_cosine: Option<f64>,
    pub similarity_gap: Option<f64>,
    pub model_embedding_request_count: Option<u64>,
    pub model_embedding_input_count: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatPerformanceMetrics {
    pub prompt_tokens: Option<u64>,
    pub completion_tokens: Option<u64>,
    #[serde(default)]
    pub time_to_first_token_ms: Option<u64>,
    #[serde(default)]
    pub ttft_first_token_ms: Option<u64>,
    #[serde(default)]
    pub ttft_first_answer_token_ms: Option<u64>,
    /// Reasoning/`<think>` token count emitted before the first answer token —
    /// the token-count companion to `ttft_first_answer_token_ms`, for empirical
    /// cross-model comparison of reasoning overhead (#492). Passthrough from the
    /// backend; `None` when the backend does not report the boundary count.
    #[serde(default)]
    pub thinking_tokens_to_answer: Option<u64>,
    #[serde(default)]
    pub decode_ms: Option<u64>,
    pub tokens_per_second: Option<f64>,
    #[serde(default)]
    pub decode_tokens_per_second: Option<f64>,
    pub response_chars: Option<u64>,
    pub followup_response_chars: Option<u64>,
    pub completion_chars: Option<u64>,
    pub tool_call_count: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolUsePerformanceMetrics {
    pub tool_call_count: Option<u64>,
    pub expected_tool_count: Option<u64>,
    pub tool_selection_precision: Option<f64>,
    pub tool_selection_recall: Option<f64>,
    pub argument_precision: Option<f64>,
    pub argument_recall: Option<f64>,
    pub repair_turns: Option<u64>,
    pub round_trip_latency_ms: Option<u64>,
    pub tool_execution_latency_ms: Option<u64>,
    pub final_response_chars: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AsrPerformanceMetrics {
    #[serde(default)]
    pub iterations: Option<u64>,
    #[serde(default)]
    pub successful_iterations: Option<u64>,
    #[serde(default)]
    pub failed_iterations: Option<u64>,
    #[serde(default)]
    pub last_iteration_error: Option<String>,
    pub audio_duration_ms: Option<u64>,
    pub transcription_latency_ms: Option<u64>,
    #[serde(default)]
    pub mean_transcription_latency_ms: Option<f64>,
    /// `support::percentile` p95 over measured iterations, using a nearest
    /// sorted-sample rank rather than interpolation. With small iteration
    /// counts such as the curated smoke n=3, this is the sample maximum;
    /// interpret it together with `iterations` and
    /// `PerformanceMetrics::request_latencies_ms`.
    #[serde(default)]
    pub p95_transcription_latency_ms: Option<f64>,
    /// Driver-measured wall time from submitting the first file-fed audio chunk
    /// to receiving the first non-empty-after-trim, non-final partial
    /// transcript segment.
    /// The value is observed only at `ingest()` returns, so its resolution is
    /// quantized by the scenario chunk size and cross-run comparisons require
    /// constant chunking. Eval audio is pushed as fast as the backend accepts
    /// chunks, so this is not comparable to realtime telephony first-partial
    /// latency targets. Batch engines and streaming engines that emit no such
    /// partial report `None` with a `MetricUnavailable` gap entry.
    /// Schema v5 runners leave this single-shot field null; schema v4 cold-start
    /// baseline records may populate it.
    #[serde(default)]
    pub ttfp_first_partial_ms: Option<u64>,
    #[serde(default)]
    pub ttfp_first_partial_samples_ms: Vec<u64>,
    #[serde(default)]
    pub mean_ttfp_first_partial_ms: Option<f64>,
    /// `support::percentile` p95 over measured first-partial samples, using a
    /// nearest sorted-sample rank rather than interpolation. With small
    /// iteration counts such as the curated smoke n=3, this is the sample
    /// maximum; interpret it together with `iterations` and
    /// `ttfp_first_partial_samples_ms`.
    #[serde(default)]
    pub p95_ttfp_first_partial_ms: Option<f64>,
    pub real_time_factor: Option<f64>,
    pub transcript_chars: Option<u64>,
    pub segment_count: Option<u64>,
    pub word_error_rate: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TtsPerformanceMetrics {
    #[serde(default)]
    pub iterations: Option<u64>,
    #[serde(default)]
    pub successful_iterations: Option<u64>,
    #[serde(default)]
    pub failed_iterations: Option<u64>,
    #[serde(default)]
    pub last_iteration_error: Option<String>,
    pub text_chars: Option<u64>,
    pub synthesis_latency_ms: Option<u64>,
    #[serde(default)]
    pub mean_synthesis_latency_ms: Option<f64>,
    /// `support::percentile` p95 over measured iterations, using a nearest
    /// sorted-sample rank rather than interpolation. With small iteration
    /// counts such as the curated smoke n=3, this is the sample maximum;
    /// interpret it together with `iterations` and
    /// `PerformanceMetrics::request_latencies_ms`.
    #[serde(default)]
    pub p95_synthesis_latency_ms: Option<f64>,
    /// Driver-measured wall time from `synthesize(request)` to the first audio
    /// chunk returned by `next_chunk()`.
    /// Schema v5 runners leave this single-shot field null; schema v4 cold-start
    /// baseline records may populate it.
    #[serde(default)]
    pub ttfa_first_chunk_ms: Option<u64>,
    #[serde(default)]
    pub ttfa_first_chunk_samples_ms: Vec<u64>,
    #[serde(default)]
    pub mean_ttfa_first_chunk_ms: Option<f64>,
    /// `support::percentile` p95 over measured first-audio samples, using a
    /// nearest sorted-sample rank rather than interpolation. With small
    /// iteration counts such as the curated smoke n=3, this is the sample
    /// maximum; interpret it together with `iterations` and
    /// `ttfa_first_chunk_samples_ms`.
    #[serde(default)]
    pub p95_ttfa_first_chunk_ms: Option<f64>,
    /// True only for incremental TTS runs where playable PCM arrived before the
    /// independent synthesis-complete signal returned by `finish()`. Buffered
    /// TTS records leave this null.
    #[serde(default)]
    pub streaming_proof_first_pcm_before_synth_complete: Option<bool>,
    #[serde(default)]
    pub synth_complete_samples_ms: Vec<u64>,
    #[serde(default)]
    pub mean_synth_complete_ms: Option<f64>,
    #[serde(default)]
    pub p95_synth_complete_ms: Option<f64>,
    #[serde(default)]
    pub inter_chunk_gap_samples_ms: Vec<u64>,
    #[serde(default)]
    pub mean_inter_chunk_gap_ms: Option<f64>,
    #[serde(default)]
    pub p95_inter_chunk_gap_ms: Option<f64>,
    #[serde(default)]
    pub max_inter_chunk_gap_ms: Option<u64>,
    #[serde(default)]
    pub underrun_count: Option<u64>,
    #[serde(default)]
    pub streaming_frame_ms: Option<u64>,
    #[serde(default)]
    pub packetized_frame_count: Option<u64>,
    #[serde(default)]
    pub max_buffered_audio_ms: Option<u64>,
    pub audio_duration_ms: Option<u64>,
    pub real_time_factor: Option<f64>,
    pub sample_count: Option<u64>,
    pub sample_rate_hz: Option<u64>,
    pub chunk_count: Option<u64>,
    #[serde(default)]
    pub per_length: Vec<TtsLengthPerformanceMetrics>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TtsLengthPerformanceMetrics {
    pub id: String,
    pub label: Option<String>,
    pub text_chars: u64,
    pub iterations: u64,
    pub successful_iterations: u64,
    pub failed_iterations: u64,
    pub ttfa_first_chunk_samples_ms: Vec<u64>,
    pub mean_ttfa_first_chunk_ms: Option<f64>,
    pub p95_ttfa_first_chunk_ms: Option<f64>,
    pub streaming_proof_first_pcm_before_synth_complete: Option<bool>,
    pub synth_complete_samples_ms: Vec<u64>,
    pub mean_synth_complete_ms: Option<f64>,
    pub p95_synth_complete_ms: Option<f64>,
    pub inter_chunk_gap_samples_ms: Vec<u64>,
    pub mean_inter_chunk_gap_ms: Option<f64>,
    pub p95_inter_chunk_gap_ms: Option<f64>,
    pub max_inter_chunk_gap_ms: Option<u64>,
    pub underrun_count: Option<u64>,
    pub streaming_frame_ms: Option<u64>,
    pub packetized_frame_count: Option<u64>,
    pub max_buffered_audio_ms: Option<u64>,
    pub audio_duration_ms: Option<u64>,
    pub real_time_factor: Option<f64>,
    pub sample_count: Option<u64>,
    pub sample_rate_hz: Option<u64>,
    pub chunk_count: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PerfPerformanceMetrics {
    pub iterations: Option<u64>,
    pub successful_iterations: Option<u64>,
    pub failed_iterations: Option<u64>,
    pub mean_latency_ms: Option<f64>,
    pub p95_latency_ms: Option<f64>,
    #[serde(default)]
    pub mean_ttft_first_token_ms: Option<f64>,
    #[serde(default)]
    pub p95_ttft_first_token_ms: Option<f64>,
    #[serde(default)]
    pub mean_ttft_first_answer_token_ms: Option<f64>,
    #[serde(default)]
    pub p95_ttft_first_answer_token_ms: Option<f64>,
    #[serde(default)]
    pub mean_decode_ms: Option<f64>,
    #[serde(default)]
    pub p95_decode_ms: Option<f64>,
    #[serde(default)]
    pub mean_decode_tokens_per_second: Option<f64>,
    #[serde(default)]
    pub total_output_tokens: Option<u64>,
    pub total_output_words: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub rss_start_bytes: Option<u64>,
    pub rss_peak_bytes: Option<u64>,
    pub rss_final_bytes: Option<u64>,
    pub process_swap_start_bytes: Option<u64>,
    pub process_swap_peak_bytes: Option<u64>,
    pub process_swap_final_bytes: Option<u64>,
    pub process_swap_delta_peak_bytes: Option<u64>,
    pub gpu_memory_peak_bytes: Option<u64>,
    #[serde(default)]
    pub gpu_utilization_peak_percent: Option<f64>,
    #[serde(default)]
    pub memory_peaks: Vec<MemoryPeak>,
    pub unavailable: Vec<String>,
    #[serde(default)]
    pub unavailable_metrics: Vec<MetricUnavailable>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MetricUnavailable {
    pub metric: String,
    pub reason: String,
    pub source: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MemoryPeak {
    pub kind: MemoryPeakKind,
    pub bytes: Option<u64>,
    pub device_id: Option<String>,
    pub source: String,
    pub unavailable_reason: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryPeakKind {
    ProcessRss,
    CudaVram,
    MetalCurrentAllocated,
    AppleFootprint,
    SystemUnifiedMemory,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ResourceSample {
    pub rss_bytes: Option<u64>,
    pub process_swap_bytes: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct MetricsSampler {
    start: ResourceSample,
    peak_rss_bytes: Option<u64>,
    peak_process_swap_bytes: Option<u64>,
    last: ResourceSample,
}

impl MetricsSampler {
    pub fn new() -> Self {
        let start = current_resource_sample();
        Self {
            peak_rss_bytes: start.rss_bytes,
            peak_process_swap_bytes: start.process_swap_bytes,
            last: start.clone(),
            start,
        }
    }

    pub fn sample(&mut self) -> ResourceSample {
        let sample = current_resource_sample();
        self.peak_rss_bytes = max_optional(self.peak_rss_bytes, sample.rss_bytes);
        self.peak_process_swap_bytes =
            max_optional(self.peak_process_swap_bytes, sample.process_swap_bytes);
        self.last = sample.clone();
        sample
    }

    pub fn finish(&mut self) -> ResourceMetrics {
        self.sample();
        let mut unavailable = Vec::new();
        let mut unavailable_metrics = Vec::new();
        if self.peak_rss_bytes.is_none() {
            unavailable.push("rss_peak_bytes".to_owned());
            unavailable_metrics.push(metric_unavailable(
                "rss_peak_bytes",
                "metric_collection_failed",
                "sysinfo",
            ));
        }
        if self.peak_process_swap_bytes.is_none() {
            unavailable.push("process_swap_peak_bytes".to_owned());
            unavailable.push("process_swap_delta_peak_bytes".to_owned());
            unavailable_metrics.push(metric_unavailable(
                "process_swap_delta_peak_bytes",
                if process_swap_metric_supported() {
                    "metric_collection_failed"
                } else {
                    "metric_unavailable_on_platform"
                },
                process_swap_source(),
            ));
        }
        unavailable.push("gpu_memory_peak_bytes".to_owned());
        unavailable_metrics.push(metric_unavailable(
            "gpu_memory_peak_bytes",
            "metric_not_instrumented",
            "accelerator_sampler",
        ));

        let process_swap_delta_peak_bytes =
            match (self.start.process_swap_bytes, self.peak_process_swap_bytes) {
                (Some(start), Some(peak)) => Some(peak.saturating_sub(start)),
                _ => None,
            };

        let mut memory_peaks = Vec::new();
        memory_peaks.push(MemoryPeak {
            kind: MemoryPeakKind::ProcessRss,
            bytes: self.peak_rss_bytes,
            device_id: None,
            source: "sysinfo".to_owned(),
            unavailable_reason: self
                .peak_rss_bytes
                .is_none()
                .then(|| "metric_collection_failed".to_owned()),
        });
        memory_peaks.push(MemoryPeak {
            kind: MemoryPeakKind::CudaVram,
            bytes: None,
            device_id: None,
            source: "unavailable".to_owned(),
            unavailable_reason: Some("metric_not_instrumented".to_owned()),
        });
        if cfg!(target_os = "macos") {
            memory_peaks.push(MemoryPeak {
                kind: MemoryPeakKind::AppleFootprint,
                bytes: self.peak_rss_bytes,
                device_id: None,
                source: "sysinfo_process_rss_proxy".to_owned(),
                unavailable_reason: None,
            });
        }

        ResourceMetrics {
            rss_start_bytes: self.start.rss_bytes,
            rss_peak_bytes: self.peak_rss_bytes,
            rss_final_bytes: self.last.rss_bytes,
            process_swap_start_bytes: self.start.process_swap_bytes,
            process_swap_peak_bytes: self.peak_process_swap_bytes,
            process_swap_final_bytes: self.last.process_swap_bytes,
            process_swap_delta_peak_bytes,
            gpu_memory_peak_bytes: None,
            gpu_utilization_peak_percent: None,
            memory_peaks,
            unavailable,
            unavailable_metrics,
        }
    }
}

impl MetricUnavailable {
    pub fn new(metric: &str, reason: &str, source: &str) -> Self {
        Self {
            metric: metric.to_owned(),
            reason: reason.to_owned(),
            source: Some(source.to_owned()),
        }
    }
}

impl Default for MetricsSampler {
    fn default() -> Self {
        Self::new()
    }
}

fn current_resource_sample() -> ResourceSample {
    let mut system = System::new();
    system.refresh_memory();

    let rss_bytes = get_current_pid().ok().and_then(|pid| {
        system.refresh_processes(ProcessesToUpdate::Some(&[pid]), false);
        system.process(pid).map(|process| process.memory())
    });

    ResourceSample {
        rss_bytes,
        process_swap_bytes: current_process_swap_bytes(),
    }
}

/// Whether the current platform implements per-process swap sampling.
/// Only Linux does (procfs `VmSwap`); macOS is an acknowledged TODO
/// (`mach_task_info_unimplemented`). Where this is false, a missing swap
/// sample is a structural platform gap, not a collection failure.
pub fn process_swap_metric_supported() -> bool {
    cfg!(target_os = "linux")
}

fn current_process_swap_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let status = fs::read_to_string("/proc/self/status").ok()?;
        parse_linux_status_vm_swap_bytes(&status)
    }

    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(target_os = "linux")]
fn parse_linux_status_vm_swap_bytes(status: &str) -> Option<u64> {
    for line in status.lines() {
        let Some(value) = line.strip_prefix("VmSwap:") else {
            continue;
        };
        return parse_kib_value(value.trim());
    }
    None
}

#[cfg(target_os = "linux")]
fn parse_kib_value(raw: &str) -> Option<u64> {
    let mut parts = raw.split_whitespace();
    let value = parts.next()?.parse::<u64>().ok()?;
    let unit = parts.next().unwrap_or("kB");
    match unit {
        "kB" => Some(value.saturating_mul(1024)),
        "mB" | "MB" => Some(value.saturating_mul(1024 * 1024)),
        "gB" | "GB" => Some(value.saturating_mul(1024 * 1024 * 1024)),
        _ => None,
    }
}

fn metric_unavailable(metric: &str, reason: &str, source: &str) -> MetricUnavailable {
    MetricUnavailable::new(metric, reason, source)
}

fn process_swap_source() -> &'static str {
    if cfg!(target_os = "linux") {
        "procfs"
    } else if cfg!(target_os = "macos") {
        "mach_task_info_unimplemented"
    } else {
        "platform_unavailable"
    }
}

fn max_optional(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

#[cfg(test)]
mod serde_tests {
    use super::*;

    #[test]
    fn audio_first_latency_fields_default_when_absent() {
        let asr: AsrPerformanceMetrics = serde_json::from_str("{}").unwrap();
        let tts: TtsPerformanceMetrics = serde_json::from_str("{}").unwrap();

        assert_eq!(asr.ttfp_first_partial_ms, None);
        assert_eq!(asr.last_iteration_error, None);
        assert!(asr.ttfp_first_partial_samples_ms.is_empty());
        assert_eq!(asr.mean_ttfp_first_partial_ms, None);
        assert_eq!(asr.p95_ttfp_first_partial_ms, None);
        assert_eq!(tts.ttfa_first_chunk_ms, None);
        assert_eq!(tts.last_iteration_error, None);
        assert!(tts.ttfa_first_chunk_samples_ms.is_empty());
        assert_eq!(tts.mean_ttfa_first_chunk_ms, None);
        assert_eq!(tts.p95_ttfa_first_chunk_ms, None);
        assert_eq!(tts.streaming_proof_first_pcm_before_synth_complete, None);
        assert!(tts.synth_complete_samples_ms.is_empty());
        assert!(tts.inter_chunk_gap_samples_ms.is_empty());
        assert_eq!(tts.underrun_count, None);
        assert_eq!(tts.streaming_frame_ms, None);
        assert_eq!(tts.packetized_frame_count, None);
        assert_eq!(tts.max_buffered_audio_ms, None);
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    #[test]
    fn parses_linux_process_swap_from_proc_status() {
        let status = "Name:\tevals\nVmRSS:\t  1234 kB\nVmSwap:\t  2048 kB\n";

        assert_eq!(parse_linux_status_vm_swap_bytes(status), Some(2048 * 1024));
    }

    #[test]
    fn missing_process_swap_is_unavailable() {
        let status = "Name:\tevals\nVmRSS:\t  1234 kB\n";

        assert_eq!(parse_linux_status_vm_swap_bytes(status), None);
    }

    #[test]
    fn gpu_memory_gap_is_recorded_as_not_instrumented() {
        let mut sampler = MetricsSampler::new();

        let resources = sampler.finish();

        assert_eq!(resources.gpu_memory_peak_bytes, None);
        assert!(resources.unavailable_metrics.iter().any(|metric| {
            metric.metric == "gpu_memory_peak_bytes"
                && metric.reason == "metric_not_instrumented"
                && metric.source.as_deref() == Some("accelerator_sampler")
        }));
    }
}
