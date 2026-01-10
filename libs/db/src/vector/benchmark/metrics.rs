//! Metrics computation for vector search benchmarks.
//!
//! Provides recall, latency, and throughput metrics.

use std::collections::HashSet;

/// Recall metrics for a set of queries.
#[derive(Debug, Clone, Default)]
pub struct RecallMetrics {
    /// Recall at each k value: (k, recall)
    pub recall_at_k: Vec<(usize, f64)>,
    /// Number of queries evaluated
    pub num_queries: usize,
}

impl RecallMetrics {
    /// Get recall at a specific k.
    pub fn get(&self, k: usize) -> Option<f64> {
        self.recall_at_k
            .iter()
            .find(|(key, _)| *key == k)
            .map(|(_, v)| *v)
    }

    /// Format as percentage string.
    pub fn format_percent(&self, k: usize) -> String {
        match self.get(k) {
            Some(v) => format!("{:.1}%", v * 100.0),
            None => "N/A".to_string(),
        }
    }
}

/// Latency statistics from a benchmark run.
#[derive(Debug, Clone, Default)]
pub struct LatencyStats {
    /// Average latency in milliseconds
    pub avg_ms: f64,
    /// Median (p50) latency in milliseconds
    pub p50_ms: f64,
    /// 95th percentile latency in milliseconds
    pub p95_ms: f64,
    /// 99th percentile latency in milliseconds
    pub p99_ms: f64,
    /// Minimum latency in milliseconds
    pub min_ms: f64,
    /// Maximum latency in milliseconds
    pub max_ms: f64,
    /// Queries per second
    pub qps: f64,
    /// Number of measurements
    pub count: usize,
}

impl LatencyStats {
    /// Compute latency statistics from a slice of latency measurements (in milliseconds).
    pub fn from_latencies(latencies: &[f64]) -> Self {
        if latencies.is_empty() {
            return Self::default();
        }

        let mut sorted = latencies.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let sum: f64 = sorted.iter().sum();
        let avg_ms = sum / sorted.len() as f64;

        Self {
            avg_ms,
            p50_ms: percentile(&sorted, 50.0),
            p95_ms: percentile(&sorted, 95.0),
            p99_ms: percentile(&sorted, 99.0),
            min_ms: sorted[0],
            max_ms: sorted[sorted.len() - 1],
            qps: 1000.0 / avg_ms,
            count: sorted.len(),
        }
    }

    /// Format as a summary string.
    pub fn summary(&self) -> String {
        format!(
            "avg={:.2}ms, p50={:.2}ms, p95={:.2}ms, p99={:.2}ms, QPS={:.1}",
            self.avg_ms, self.p50_ms, self.p95_ms, self.p99_ms, self.qps
        )
    }
}

/// Compute Recall@k for search results vs ground truth.
///
/// Recall = |retrieved ∩ relevant| / |relevant|
///
/// # Arguments
///
/// * `search_results` - Retrieved indices for each query
/// * `ground_truth` - True top-k indices for each query
/// * `k` - Number of results to consider
///
/// # Returns
///
/// Average recall across all queries.
pub fn compute_recall(
    search_results: &[Vec<usize>],
    ground_truth: &[Vec<usize>],
    k: usize,
) -> f64 {
    if search_results.len() != ground_truth.len() {
        return 0.0;
    }

    let mut total_recall = 0.0;

    for (results, truth) in search_results.iter().zip(ground_truth.iter()) {
        let retrieved: HashSet<_> = results.iter().take(k).collect();
        let relevant: HashSet<_> = truth.iter().take(k).collect();

        let intersection = retrieved.intersection(&relevant).count();
        let recall = intersection as f64 / relevant.len().min(k) as f64;
        total_recall += recall;
    }

    total_recall / search_results.len() as f64
}

/// Compute recall at multiple k values.
pub fn compute_recall_at_ks(
    search_results: &[Vec<usize>],
    ground_truth: &[Vec<usize>],
    k_values: &[usize],
) -> RecallMetrics {
    let recall_at_k: Vec<(usize, f64)> = k_values
        .iter()
        .map(|&k| (k, compute_recall(search_results, ground_truth, k)))
        .collect();

    RecallMetrics {
        recall_at_k,
        num_queries: search_results.len(),
    }
}

/// Compute percentile from sorted values.
///
/// # Arguments
///
/// * `sorted_values` - Values sorted in ascending order
/// * `p` - Percentile (0-100)
pub fn percentile(sorted_values: &[f64], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted_values.len() - 1) as f64).round() as usize;
    sorted_values[idx.min(sorted_values.len() - 1)]
}

/// Compute throughput in queries per second.
pub fn compute_qps(total_queries: usize, total_time_ms: f64) -> f64 {
    if total_time_ms <= 0.0 {
        return 0.0;
    }
    1000.0 * total_queries as f64 / total_time_ms
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_recall_perfect() {
        let search_results = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let ground_truth = vec![vec![0, 1, 2], vec![3, 4, 5]];

        let recall = compute_recall(&search_results, &ground_truth, 3);
        assert!((recall - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_recall_partial() {
        let search_results = vec![vec![0, 1, 9]]; // 2 out of 3 correct
        let ground_truth = vec![vec![0, 1, 2]];

        let recall = compute_recall(&search_results, &ground_truth, 3);
        assert!((recall - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_recall_zero() {
        let search_results = vec![vec![7, 8, 9]];
        let ground_truth = vec![vec![0, 1, 2]];

        let recall = compute_recall(&search_results, &ground_truth, 3);
        assert!((recall).abs() < 1e-6);
    }

    #[test]
    fn test_percentile() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert!((percentile(&values, 0.0) - 1.0).abs() < 1e-6);
        assert!((percentile(&values, 50.0) - 3.0).abs() < 1e-6);
        assert!((percentile(&values, 100.0) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_latency_stats() {
        let latencies = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let stats = LatencyStats::from_latencies(&latencies);

        assert!((stats.avg_ms - 5.5).abs() < 1e-6);
        assert!((stats.min_ms - 1.0).abs() < 1e-6);
        assert!((stats.max_ms - 10.0).abs() < 1e-6);
        assert_eq!(stats.count, 10);
    }

    #[test]
    fn test_recall_metrics_get() {
        let metrics = RecallMetrics {
            recall_at_k: vec![(1, 0.8), (5, 0.9), (10, 0.95)],
            num_queries: 100,
        };

        assert!((metrics.get(1).unwrap() - 0.8).abs() < 1e-6);
        assert!((metrics.get(10).unwrap() - 0.95).abs() < 1e-6);
        assert!(metrics.get(20).is_none());
    }

    #[test]
    fn test_compute_qps() {
        let qps = compute_qps(100, 1000.0); // 100 queries in 1 second
        assert!((qps - 100.0).abs() < 1e-6);
    }
}
