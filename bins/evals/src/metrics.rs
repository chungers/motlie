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
    pub decode_ms: Option<u64>,
    pub tokens_per_second: Option<f64>,
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
    pub audio_duration_ms: Option<u64>,
    pub transcription_latency_ms: Option<u64>,
    pub real_time_factor: Option<f64>,
    pub transcript_chars: Option<u64>,
    pub segment_count: Option<u64>,
    pub word_error_rate: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TtsPerformanceMetrics {
    pub text_chars: Option<u64>,
    pub synthesis_latency_ms: Option<u64>,
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
                "metric_unavailable_on_platform",
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
    MetricUnavailable {
        metric: metric.to_owned(),
        reason: reason.to_owned(),
        source: Some(source.to_owned()),
    }
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
}
