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

// ============================================================================
// Pareto Frontier Analysis
// ============================================================================

/// A point on the Recall vs QPS Pareto frontier.
///
/// Captures the parameters that achieved this trade-off point.
#[derive(Debug, Clone)]
pub struct ParetoPoint {
    /// Bits per dimension (for RaBitQ)
    pub bits_per_dim: u8,
    /// ef_search parameter
    pub ef_search: usize,
    /// Rerank factor (for RaBitQ)
    pub rerank_factor: usize,
    /// k value for Recall@k
    pub k: usize,
    /// Recall achieved (0.0 to 1.0)
    pub recall: f64,
    /// Queries per second achieved
    pub qps: f64,
}

impl ParetoPoint {
    /// Format as CSV row.
    pub fn to_csv_row(&self) -> String {
        format!(
            "{},{},{},{},{:.4},{:.1}",
            self.bits_per_dim, self.ef_search, self.rerank_factor, self.k, self.recall, self.qps
        )
    }

    /// CSV header for Pareto points.
    pub fn csv_header() -> &'static str {
        "bits,ef_search,rerank_factor,k,recall,qps"
    }
}

/// Input data for Pareto frontier computation.
///
/// Generic representation of benchmark results for Pareto analysis.
#[derive(Debug, Clone)]
pub struct ParetoInput {
    /// Bits per dimension (for RaBitQ, use 0 for non-RaBitQ)
    pub bits_per_dim: u8,
    /// ef_search parameter
    pub ef_search: usize,
    /// Rerank factor (for RaBitQ, use 1 for non-RaBitQ)
    pub rerank_factor: usize,
    /// k value for Recall@k
    pub k: usize,
    /// Recall achieved (0.0 to 1.0)
    pub recall: f64,
    /// Queries per second achieved
    pub qps: f64,
}

/// Compute Pareto-optimal points from benchmark results.
///
/// A point is Pareto-optimal if no other point has BOTH:
/// - Higher recall AND higher QPS
///
/// These represent the best recall-throughput trade-offs. Points on the
/// frontier cannot be improved in one dimension without sacrificing the other.
///
/// # Arguments
///
/// * `results` - Benchmark results to analyze
///
/// # Returns
///
/// Vector of Pareto-optimal points, sorted by recall (ascending).
pub fn compute_pareto_frontier(results: &[ParetoInput]) -> Vec<ParetoPoint> {
    let mut frontier = Vec::new();

    for r in results {
        // Check if this point is dominated by any other point
        let is_dominated = results.iter().any(|other| {
            // other dominates r if other is better in both dimensions
            // (strictly better in at least one, not worse in the other)
            (other.recall > r.recall && other.qps >= r.qps)
                || (other.recall >= r.recall && other.qps > r.qps)
        });

        if !is_dominated {
            frontier.push(ParetoPoint {
                bits_per_dim: r.bits_per_dim,
                ef_search: r.ef_search,
                rerank_factor: r.rerank_factor,
                k: r.k,
                recall: r.recall,
                qps: r.qps,
            });
        }
    }

    // Sort by recall (ascending) for visualization
    frontier.sort_by(|a, b| a.recall.partial_cmp(&b.recall).unwrap());

    // Remove duplicates (same recall/qps but different params)
    frontier.dedup_by(|a, b| {
        (a.recall - b.recall).abs() < 1e-6 && (a.qps - b.qps).abs() < 1e-6
    });

    frontier
}

/// Compute Pareto frontier for a specific k value.
///
/// Filters results to only include entries with the specified k.
pub fn compute_pareto_frontier_for_k(results: &[ParetoInput], k: usize) -> Vec<ParetoPoint> {
    let filtered: Vec<ParetoInput> = results
        .iter()
        .filter(|r| r.k == k)
        .cloned()
        .collect();
    compute_pareto_frontier(&filtered)
}

// ============================================================================
// Rotation Quality Analysis (RaBitQ)
// ============================================================================

/// Statistics about RaBitQ rotation quality.
///
/// Used to validate that the rotation matrix has correct √D scaling.
/// With proper scaling, unit vectors should have component variance ≈ 1.0.
#[derive(Debug, Clone)]
pub struct RotationStats {
    /// Mean of all rotated vector components.
    pub component_mean: f32,
    /// Variance of all rotated vector components.
    pub component_variance: f32,
    /// Expected variance (1.0 for properly scaled rotation).
    pub expected_variance: f32,
    /// Whether the scaling is valid (variance in [0.8, 1.2]).
    pub scaling_valid: bool,
    /// Number of vectors sampled.
    pub sample_size: usize,
}

impl RotationStats {
    /// Check if rotation quality is acceptable.
    pub fn is_valid(&self) -> bool {
        self.scaling_valid
    }

    /// Format as summary string.
    pub fn summary(&self) -> String {
        format!(
            "mean={:.4}, variance={:.4} (expected {:.1}), valid={}",
            self.component_mean, self.component_variance, self.expected_variance, self.scaling_valid
        )
    }
}

/// Compute variance of rotated vector components.
///
/// For correctly scaled rotation matrix (√D scaling), unit vectors
/// should have component variance ≈ 1.0 after rotation.
/// Without scaling, variance would be ≈ 1/D.
///
/// # Arguments
///
/// * `rotate_fn` - Function that rotates a vector (e.g., `encoder.rotate_query`)
/// * `vectors` - Vectors to analyze
/// * `sample_size` - Maximum number of vectors to sample (for efficiency)
///
/// # Returns
///
/// Statistics about the rotation quality.
pub fn compute_rotated_variance<F>(rotate_fn: F, vectors: &[Vec<f32>], sample_size: usize) -> RotationStats
where
    F: Fn(&[f32]) -> Vec<f32>,
{
    let sample_count = sample_size.min(vectors.len());
    let mut all_components = Vec::new();

    for vector in vectors.iter().take(sample_count) {
        let rotated = rotate_fn(vector);
        all_components.extend(rotated);
    }

    if all_components.is_empty() {
        return RotationStats {
            component_mean: 0.0,
            component_variance: 0.0,
            expected_variance: 1.0,
            scaling_valid: false,
            sample_size: 0,
        };
    }

    let n = all_components.len() as f32;
    let mean: f32 = all_components.iter().sum::<f32>() / n;
    let variance: f32 = all_components
        .iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f32>()
        / n;

    RotationStats {
        component_mean: mean,
        component_variance: variance,
        expected_variance: 1.0,
        scaling_valid: (0.8..=1.2).contains(&variance),
        sample_size: sample_count,
    }
}

/// Print rotation quality analysis to stdout.
pub fn print_rotation_stats(stats: &RotationStats, title: &str) {
    println!("\n{}", title);
    println!("{}", "-".repeat(50));
    println!("  Sample size: {} vectors", stats.sample_size);
    println!("  Component mean: {:.6}", stats.component_mean);
    println!(
        "  Component variance: {:.6} (expected: {:.1})",
        stats.component_variance, stats.expected_variance
    );
    println!(
        "  Scaling valid: {} (variance in [0.8, 1.2])",
        if stats.scaling_valid { "YES ✓" } else { "NO ✗" }
    );
    println!("{}", "-".repeat(50));
}

/// Print Pareto frontier summary to stdout.
pub fn print_pareto_frontier(frontier: &[ParetoPoint], title: &str) {
    println!("\n{}", title);
    println!("{}", "-".repeat(70));
    println!(
        "{:>6} {:>8} {:>8} {:>4} {:>10} {:>10}",
        "bits", "ef", "rerank", "k", "recall", "QPS"
    );
    println!("{}", "-".repeat(70));

    for p in frontier {
        println!(
            "{:>6} {:>8} {:>8} {:>4} {:>9.1}% {:>10.0}",
            p.bits_per_dim,
            p.ef_search,
            p.rerank_factor,
            p.k,
            p.recall * 100.0,
            p.qps
        );
    }
    println!("{}", "-".repeat(70));
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

    #[test]
    fn test_pareto_frontier_basic() {
        // Create test data with clear Pareto-optimal points:
        // - Point A: recall=0.9, qps=1000 (Pareto-optimal: high recall, low qps)
        // - Point B: recall=0.8, qps=2000 (Pareto-optimal: balanced)
        // - Point C: recall=0.7, qps=3000 (Pareto-optimal: low recall, high qps)
        // - Point D: recall=0.75, qps=1500 (dominated by B)
        let results = vec![
            ParetoInput {
                bits_per_dim: 1,
                ef_search: 200,
                rerank_factor: 10,
                k: 10,
                recall: 0.9,
                qps: 1000.0,
            },
            ParetoInput {
                bits_per_dim: 1,
                ef_search: 100,
                rerank_factor: 4,
                k: 10,
                recall: 0.8,
                qps: 2000.0,
            },
            ParetoInput {
                bits_per_dim: 1,
                ef_search: 50,
                rerank_factor: 1,
                k: 10,
                recall: 0.7,
                qps: 3000.0,
            },
            ParetoInput {
                bits_per_dim: 1,
                ef_search: 75,
                rerank_factor: 2,
                k: 10,
                recall: 0.75,
                qps: 1500.0, // dominated by B (0.8 recall, 2000 qps)
            },
        ];

        let frontier = compute_pareto_frontier(&results);

        // Should have 3 Pareto-optimal points (A, B, C)
        assert_eq!(frontier.len(), 3);

        // Should be sorted by recall ascending
        assert!(frontier[0].recall < frontier[1].recall);
        assert!(frontier[1].recall < frontier[2].recall);

        // Verify the dominated point is not in the frontier
        assert!(frontier.iter().all(|p| (p.recall - 0.75).abs() > 1e-6));
    }

    #[test]
    fn test_pareto_frontier_single_point() {
        let results = vec![ParetoInput {
            bits_per_dim: 2,
            ef_search: 100,
            rerank_factor: 4,
            k: 10,
            recall: 0.85,
            qps: 1500.0,
        }];

        let frontier = compute_pareto_frontier(&results);
        assert_eq!(frontier.len(), 1);
        assert!((frontier[0].recall - 0.85).abs() < 1e-6);
    }

    #[test]
    fn test_pareto_frontier_empty() {
        let results: Vec<ParetoInput> = vec![];
        let frontier = compute_pareto_frontier(&results);
        assert!(frontier.is_empty());
    }

    #[test]
    fn test_pareto_frontier_for_k() {
        let results = vec![
            ParetoInput {
                bits_per_dim: 1,
                ef_search: 100,
                rerank_factor: 4,
                k: 10,
                recall: 0.9,
                qps: 1000.0,
            },
            ParetoInput {
                bits_per_dim: 1,
                ef_search: 100,
                rerank_factor: 4,
                k: 5,
                recall: 0.95,
                qps: 1200.0,
            },
            ParetoInput {
                bits_per_dim: 1,
                ef_search: 50,
                rerank_factor: 1,
                k: 10,
                recall: 0.7,
                qps: 2000.0,
            },
        ];

        // Filter for k=10 only
        let frontier_k10 = compute_pareto_frontier_for_k(&results, 10);
        assert_eq!(frontier_k10.len(), 2);
        assert!(frontier_k10.iter().all(|p| p.k == 10));

        // Filter for k=5 only
        let frontier_k5 = compute_pareto_frontier_for_k(&results, 5);
        assert_eq!(frontier_k5.len(), 1);
        assert_eq!(frontier_k5[0].k, 5);
    }

    #[test]
    fn test_pareto_point_csv() {
        let point = ParetoPoint {
            bits_per_dim: 2,
            ef_search: 100,
            rerank_factor: 4,
            k: 10,
            recall: 0.923,
            qps: 1456.7,
        };

        let csv = point.to_csv_row();
        assert_eq!(csv, "2,100,4,10,0.9230,1456.7");

        let header = ParetoPoint::csv_header();
        assert_eq!(header, "bits,ef_search,rerank_factor,k,recall,qps");
    }

    #[test]
    fn test_rotation_stats_valid() {
        // Identity rotation should preserve variance
        let identity_rotate = |v: &[f32]| v.to_vec();

        // Create unit vectors (variance should be preserved)
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                let mut v = vec![0.0f32; 128];
                let angle = i as f32 * 0.1;
                v[0] = angle.cos();
                v[1] = angle.sin();
                // Normalize
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter_mut().for_each(|x| *x /= norm);
                v
            })
            .collect();

        let stats = compute_rotated_variance(identity_rotate, &vectors, 100);

        assert_eq!(stats.sample_size, 100);
        // For sparse unit vectors, variance will be very low (most components are 0)
        // This is expected since we're not using a real rotation
    }

    #[test]
    fn test_rotation_stats_empty() {
        let identity_rotate = |v: &[f32]| v.to_vec();
        let vectors: Vec<Vec<f32>> = vec![];

        let stats = compute_rotated_variance(identity_rotate, &vectors, 100);

        assert_eq!(stats.sample_size, 0);
        assert!(!stats.scaling_valid);
    }

    #[test]
    fn test_rotation_stats_summary() {
        let stats = RotationStats {
            component_mean: 0.01,
            component_variance: 0.95,
            expected_variance: 1.0,
            scaling_valid: true,
            sample_size: 1000,
        };

        let summary = stats.summary();
        assert!(summary.contains("0.0100"));
        assert!(summary.contains("0.9500"));
        assert!(summary.contains("true"));
        assert!(stats.is_valid());
    }

    #[test]
    fn test_rotation_stats_invalid_variance() {
        let stats = RotationStats {
            component_mean: 0.0,
            component_variance: 0.5, // Too low, not in [0.8, 1.2]
            expected_variance: 1.0,
            scaling_valid: false,
            sample_size: 100,
        };

        assert!(!stats.is_valid());

        let stats_high = RotationStats {
            component_mean: 0.0,
            component_variance: 1.5, // Too high
            expected_variance: 1.0,
            scaling_valid: false,
            sample_size: 100,
        };

        assert!(!stats_high.is_valid());
    }
}
