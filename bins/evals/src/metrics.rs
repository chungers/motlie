use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub startup_ms: Option<u64>,
    pub request_latencies_ms: Vec<u64>,
    pub vectors_per_second: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub rss_peak_bytes: Option<u64>,
    pub swap_peak_bytes: Option<u64>,
    pub gpu_memory_peak_bytes: Option<u64>,
}
