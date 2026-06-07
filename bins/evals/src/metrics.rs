use serde::{Deserialize, Serialize};
use sysinfo::{get_current_pid, ProcessesToUpdate, System};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub startup_ms: Option<u64>,
    pub request_latencies_ms: Vec<u64>,
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
pub struct ResourceMetrics {
    pub rss_start_bytes: Option<u64>,
    pub rss_peak_bytes: Option<u64>,
    pub rss_final_bytes: Option<u64>,
    pub swap_start_bytes: Option<u64>,
    pub swap_peak_bytes: Option<u64>,
    pub swap_final_bytes: Option<u64>,
    pub gpu_memory_peak_bytes: Option<u64>,
    pub unavailable: Vec<String>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ResourceSample {
    pub rss_bytes: Option<u64>,
    pub swap_used_bytes: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct MetricsSampler {
    start: ResourceSample,
    peak_rss_bytes: Option<u64>,
    peak_swap_bytes: Option<u64>,
    last: ResourceSample,
}

impl MetricsSampler {
    pub fn new() -> Self {
        let start = current_resource_sample();
        Self {
            peak_rss_bytes: start.rss_bytes,
            peak_swap_bytes: start.swap_used_bytes,
            last: start.clone(),
            start,
        }
    }

    pub fn sample(&mut self) -> ResourceSample {
        let sample = current_resource_sample();
        self.peak_rss_bytes = max_optional(self.peak_rss_bytes, sample.rss_bytes);
        self.peak_swap_bytes = max_optional(self.peak_swap_bytes, sample.swap_used_bytes);
        self.last = sample.clone();
        sample
    }

    pub fn finish(&mut self) -> ResourceMetrics {
        self.sample();
        let mut unavailable = Vec::new();
        if self.peak_rss_bytes.is_none() {
            unavailable.push("rss_peak_bytes".to_owned());
        }
        if self.peak_swap_bytes.is_none() {
            unavailable.push("swap_peak_bytes".to_owned());
        }
        unavailable.push("gpu_memory_peak_bytes".to_owned());

        ResourceMetrics {
            rss_start_bytes: self.start.rss_bytes,
            rss_peak_bytes: self.peak_rss_bytes,
            rss_final_bytes: self.last.rss_bytes,
            swap_start_bytes: self.start.swap_used_bytes,
            swap_peak_bytes: self.peak_swap_bytes,
            swap_final_bytes: self.last.swap_used_bytes,
            gpu_memory_peak_bytes: None,
            unavailable,
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
        swap_used_bytes: Some(system.used_swap()),
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
