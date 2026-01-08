//! Benchmark dataset support for ANN algorithms
//!
//! This module provides support for industry-standard benchmark datasets:
//! - SIFT1M: 128-dim, 1M base vectors, 10K queries, Euclidean distance
//! - GIST1M: 960-dim, 1M base vectors, 1K queries, Euclidean distance
//! - GloVe: Various dimensions, angular distance
//!
//! Datasets are downloaded from HuggingFace and cached locally.
//!
//! ## File Formats
//!
//! ### fvecs (float vectors)
//! Each vector: 4 bytes (dim as i32) + dim * 4 bytes (f32 values)
//!
//! ### ivecs (integer vectors)
//! Each vector: 4 bytes (dim as i32) + dim * 4 bytes (i32 values)
//! Used for ground truth neighbor indices.
//!
//! ## Sources
//! - Original: http://corpus-texmex.irisa.fr/
//! - HuggingFace: https://huggingface.co/datasets/qbo-odp/sift1m
//! - ANN-Benchmarks: https://ann-benchmarks.com/

#![allow(dead_code)]

use anyhow::{Context, Result};
use std::fs::{self, File};
use std::io::{BufReader, Read, Write as IoWrite};
use std::path::{Path, PathBuf};

// ============================================================================
// Constants
// ============================================================================

/// Default cache directory for downloaded datasets
pub const CACHE_DIR: &str = "/tmp/ann_benchmarks";

/// HuggingFace base URL for dataset downloads
pub const HF_BASE_URL: &str = "https://huggingface.co/datasets";

// ============================================================================
// Dataset Definitions
// ============================================================================

/// Supported benchmark datasets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetName {
    /// SIFT1M: 128-dim, 1M vectors, Euclidean
    Sift1M,
    /// SIFT10K: 128-dim, 10K vectors, Euclidean (smaller for testing)
    Sift10K,
    /// Random synthetic data (for comparison)
    Random,
}

impl DatasetName {
    /// Parse dataset name from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "sift1m" | "sift-1m" | "sift" => Some(Self::Sift1M),
            "sift10k" | "sift-10k" => Some(Self::Sift10K),
            "random" | "synthetic" => Some(Self::Random),
            _ => None,
        }
    }

    /// Get dataset dimensions
    pub fn dimensions(&self) -> usize {
        match self {
            Self::Sift1M | Self::Sift10K => 128,
            Self::Random => 1024, // Default for synthetic
        }
    }

    /// Get distance metric name
    pub fn distance_metric(&self) -> &'static str {
        match self {
            Self::Sift1M | Self::Sift10K => "euclidean",
            Self::Random => "euclidean",
        }
    }
}

impl std::fmt::Display for DatasetName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sift1M => write!(f, "sift1m"),
            Self::Sift10K => write!(f, "sift10k"),
            Self::Random => write!(f, "random"),
        }
    }
}

/// Dataset configuration
#[derive(Debug, Clone)]
pub struct DatasetConfig {
    /// Dataset name
    pub name: DatasetName,
    /// Number of base vectors to use (None = all)
    pub num_base: Option<usize>,
    /// Number of query vectors to use (None = all)
    pub num_queries: Option<usize>,
    /// Number of ground truth neighbors (k)
    pub ground_truth_k: usize,
    /// Cache directory
    pub cache_dir: PathBuf,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            name: DatasetName::Sift1M,
            num_base: None,
            num_queries: None,
            ground_truth_k: 100,
            cache_dir: PathBuf::from(CACHE_DIR),
        }
    }
}

// ============================================================================
// Dataset Structure
// ============================================================================

/// Loaded benchmark dataset with vectors and ground truth
#[derive(Debug)]
pub struct BenchmarkDataset {
    /// Dataset name
    pub name: DatasetName,
    /// Vector dimensions
    pub dimensions: usize,
    /// Base vectors (to be indexed)
    pub base_vectors: Vec<Vec<f32>>,
    /// Query vectors (for search evaluation)
    pub query_vectors: Vec<Vec<f32>>,
    /// Ground truth: for each query, indices of k nearest neighbors in base_vectors
    /// Outer vec: one per query, Inner vec: neighbor indices sorted by distance
    pub ground_truth: Vec<Vec<usize>>,
    /// Ground truth distances (optional, for validation)
    pub ground_truth_distances: Option<Vec<Vec<f32>>>,
}

impl BenchmarkDataset {
    /// Get number of base vectors
    pub fn num_base(&self) -> usize {
        self.base_vectors.len()
    }

    /// Get number of queries
    pub fn num_queries(&self) -> usize {
        self.query_vectors.len()
    }

    /// Calculate recall@k for search results
    ///
    /// # Arguments
    /// * `query_idx` - Index of the query
    /// * `results` - Search results as (distance, index) pairs
    /// * `k` - Number of neighbors to consider
    ///
    /// # Returns
    /// Recall value between 0.0 and 1.0
    pub fn calculate_recall(&self, query_idx: usize, results: &[(f32, usize)], k: usize) -> f32 {
        if query_idx >= self.ground_truth.len() {
            return 0.0;
        }

        let gt = &self.ground_truth[query_idx];
        let gt_k: std::collections::HashSet<_> = gt.iter().take(k).cloned().collect();

        let result_k: std::collections::HashSet<_> = results.iter()
            .take(k)
            .map(|(_, idx)| *idx)
            .collect();

        let intersection = gt_k.intersection(&result_k).count();
        intersection as f32 / k as f32
    }

    /// Calculate average recall@k over all queries
    pub fn calculate_average_recall<F>(&self, k: usize, search_fn: F) -> f32
    where
        F: Fn(&[f32]) -> Vec<(f32, usize)>,
    {
        let mut total_recall = 0.0;
        for (i, query) in self.query_vectors.iter().enumerate() {
            let results = search_fn(query);
            total_recall += self.calculate_recall(i, &results, k);
        }
        total_recall / self.num_queries() as f32
    }

    /// Validate search results against ground truth
    /// Returns detailed validation results
    pub fn validate_results(
        &self,
        query_idx: usize,
        results: &[(f32, usize)],
        k: usize,
    ) -> ValidationResult {
        let gt = &self.ground_truth[query_idx];
        let gt_k: Vec<usize> = gt.iter().take(k).cloned().collect();

        let result_indices: Vec<usize> = results.iter()
            .take(k)
            .map(|(_, idx)| *idx)
            .collect();

        let gt_set: std::collections::HashSet<_> = gt_k.iter().cloned().collect();
        let result_set: std::collections::HashSet<_> = result_indices.iter().cloned().collect();

        let true_positives: Vec<usize> = result_indices.iter()
            .filter(|idx| gt_set.contains(idx))
            .cloned()
            .collect();

        let false_positives: Vec<usize> = result_indices.iter()
            .filter(|idx| !gt_set.contains(idx))
            .cloned()
            .collect();

        let false_negatives: Vec<usize> = gt_k.iter()
            .filter(|idx| !result_set.contains(idx))
            .cloned()
            .collect();

        ValidationResult {
            query_idx,
            k,
            recall: true_positives.len() as f32 / k as f32,
            true_positives,
            false_positives,
            false_negatives,
            ground_truth: gt_k,
            results: result_indices,
        }
    }
}

/// Validation result for a single query
#[derive(Debug)]
pub struct ValidationResult {
    pub query_idx: usize,
    pub k: usize,
    pub recall: f32,
    pub true_positives: Vec<usize>,
    pub false_positives: Vec<usize>,
    pub false_negatives: Vec<usize>,
    pub ground_truth: Vec<usize>,
    pub results: Vec<usize>,
}

impl std::fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Query {}: Recall@{}={:.3} (TP={}, FP={}, FN={})",
            self.query_idx,
            self.k,
            self.recall,
            self.true_positives.len(),
            self.false_positives.len(),
            self.false_negatives.len()
        )
    }
}

// ============================================================================
// File Format Readers
// ============================================================================

/// Read vectors from fvecs format file
///
/// Format: Each vector is stored as:
/// - 4 bytes: dimension (i32, little-endian)
/// - dim * 4 bytes: vector components (f32, little-endian)
pub fn read_fvecs(path: &Path) -> Result<Vec<Vec<f32>>> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open fvecs file: {}", path.display()))?;
    let file_size = file.metadata()?.len();
    let mut reader = BufReader::new(file);

    let mut vectors = Vec::new();
    let mut bytes_read = 0u64;

    while bytes_read < file_size {
        // Read dimension
        let mut dim_buf = [0u8; 4];
        reader.read_exact(&mut dim_buf)
            .with_context(|| "Failed to read dimension")?;
        let dim = i32::from_le_bytes(dim_buf) as usize;
        bytes_read += 4;

        // Read vector components
        let mut vector = vec![0.0f32; dim];
        for i in 0..dim {
            let mut val_buf = [0u8; 4];
            reader.read_exact(&mut val_buf)
                .with_context(|| format!("Failed to read component {}", i))?;
            vector[i] = f32::from_le_bytes(val_buf);
        }
        bytes_read += (dim * 4) as u64;

        vectors.push(vector);
    }

    Ok(vectors)
}

/// Read vectors from fvecs format file with limit
pub fn read_fvecs_limited(path: &Path, limit: usize) -> Result<Vec<Vec<f32>>> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open fvecs file: {}", path.display()))?;
    let mut reader = BufReader::new(file);

    let mut vectors = Vec::new();

    while vectors.len() < limit {
        // Read dimension
        let mut dim_buf = [0u8; 4];
        if reader.read_exact(&mut dim_buf).is_err() {
            break; // EOF
        }
        let dim = i32::from_le_bytes(dim_buf) as usize;

        // Read vector components
        let mut vector = vec![0.0f32; dim];
        for i in 0..dim {
            let mut val_buf = [0u8; 4];
            reader.read_exact(&mut val_buf)
                .with_context(|| format!("Failed to read component {}", i))?;
            vector[i] = f32::from_le_bytes(val_buf);
        }

        vectors.push(vector);
    }

    Ok(vectors)
}

/// Read integer vectors from ivecs format file (for ground truth)
///
/// Format: Each vector is stored as:
/// - 4 bytes: dimension (i32, little-endian)
/// - dim * 4 bytes: vector components (i32, little-endian)
pub fn read_ivecs(path: &Path) -> Result<Vec<Vec<i32>>> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open ivecs file: {}", path.display()))?;
    let file_size = file.metadata()?.len();
    let mut reader = BufReader::new(file);

    let mut vectors = Vec::new();
    let mut bytes_read = 0u64;

    while bytes_read < file_size {
        // Read dimension
        let mut dim_buf = [0u8; 4];
        reader.read_exact(&mut dim_buf)
            .with_context(|| "Failed to read dimension")?;
        let dim = i32::from_le_bytes(dim_buf) as usize;
        bytes_read += 4;

        // Read vector components
        let mut vector = vec![0i32; dim];
        for i in 0..dim {
            let mut val_buf = [0u8; 4];
            reader.read_exact(&mut val_buf)
                .with_context(|| format!("Failed to read component {}", i))?;
            vector[i] = i32::from_le_bytes(val_buf);
        }
        bytes_read += (dim * 4) as u64;

        vectors.push(vector);
    }

    Ok(vectors)
}

/// Read integer vectors from ivecs format file with limit
pub fn read_ivecs_limited(path: &Path, limit: usize) -> Result<Vec<Vec<i32>>> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open ivecs file: {}", path.display()))?;
    let mut reader = BufReader::new(file);

    let mut vectors = Vec::new();

    while vectors.len() < limit {
        // Read dimension
        let mut dim_buf = [0u8; 4];
        if reader.read_exact(&mut dim_buf).is_err() {
            break; // EOF
        }
        let dim = i32::from_le_bytes(dim_buf) as usize;

        // Read vector components
        let mut vector = vec![0i32; dim];
        for i in 0..dim {
            let mut val_buf = [0u8; 4];
            reader.read_exact(&mut val_buf)
                .with_context(|| format!("Failed to read component {}", i))?;
            vector[i] = i32::from_le_bytes(val_buf);
        }

        vectors.push(vector);
    }

    Ok(vectors)
}

// ============================================================================
// Dataset Download
// ============================================================================

/// Download a file from URL to destination path
pub async fn download_file(url: &str, dest: &Path) -> Result<()> {
    println!("Downloading {} to {}", url, dest.display());

    // Create parent directory if needed
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }

    // Download using reqwest
    let response = reqwest::get(url).await
        .with_context(|| format!("Failed to download: {}", url))?;

    if !response.status().is_success() {
        anyhow::bail!("Download failed with status: {}", response.status());
    }

    let bytes = response.bytes().await?;

    let mut file = File::create(dest)?;
    file.write_all(&bytes)?;

    println!("Downloaded {} bytes to {}", bytes.len(), dest.display());
    Ok(())
}

/// Get the cache path for a dataset file
fn get_cache_path(cache_dir: &Path, dataset: DatasetName, filename: &str) -> PathBuf {
    cache_dir.join(dataset.to_string()).join(filename)
}

/// Check if dataset files exist in cache
fn dataset_cached(cache_dir: &Path, dataset: DatasetName) -> bool {
    match dataset {
        DatasetName::Sift1M | DatasetName::Sift10K => {
            let base = get_cache_path(cache_dir, dataset, "sift_base.fvecs");
            let query = get_cache_path(cache_dir, dataset, "sift_query.fvecs");
            let gt = get_cache_path(cache_dir, dataset, "sift_groundtruth.ivecs");
            base.exists() && query.exists() && gt.exists()
        }
        DatasetName::Random => true, // No download needed
    }
}

/// Download SIFT1M dataset from HuggingFace
async fn download_sift1m(cache_dir: &Path) -> Result<()> {
    let dataset_dir = cache_dir.join("sift1m");
    fs::create_dir_all(&dataset_dir)?;

    let files = [
        ("sift_base.fvecs", "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_base.fvecs"),
        ("sift_query.fvecs", "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_query.fvecs"),
        ("sift_groundtruth.ivecs", "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_groundtruth.ivecs"),
    ];

    for (filename, url) in &files {
        let dest = dataset_dir.join(filename);
        if !dest.exists() {
            download_file(url, &dest).await?;
        } else {
            println!("Using cached: {}", dest.display());
        }
    }

    Ok(())
}

// ============================================================================
// Dataset Loading
// ============================================================================

/// Load a benchmark dataset
pub async fn load_dataset(config: &DatasetConfig) -> Result<BenchmarkDataset> {
    match config.name {
        DatasetName::Sift1M => load_sift_dataset(config, false).await,
        DatasetName::Sift10K => load_sift_dataset(config, true).await,
        DatasetName::Random => generate_random_dataset(config),
    }
}

/// Compute Euclidean distance between two vectors
fn euclidean_distance_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// Compute ground truth via brute force for a subset of vectors
fn compute_ground_truth_brute_force(
    base_vectors: &[Vec<f32>],
    query_vectors: &[Vec<f32>],
    k: usize,
) -> Vec<Vec<usize>> {
    println!("Computing ground truth via brute force ({} queries)...", query_vectors.len());

    query_vectors.iter()
        .enumerate()
        .map(|(q_idx, query)| {
            let mut distances: Vec<(f32, usize)> = base_vectors.iter()
                .enumerate()
                .map(|(idx, base)| (euclidean_distance_f32(query, base), idx))
                .collect();
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            if (q_idx + 1) % 100 == 0 {
                println!("  Computed ground truth for {}/{} queries", q_idx + 1, query_vectors.len());
            }

            distances.into_iter()
                .take(k)
                .map(|(_, idx)| idx)
                .collect()
        })
        .collect()
}

/// Load SIFT dataset (1M or 10K subset)
async fn load_sift_dataset(config: &DatasetConfig, is_10k: bool) -> Result<BenchmarkDataset> {
    // Ensure dataset is downloaded
    if !dataset_cached(&config.cache_dir, DatasetName::Sift1M) {
        println!("Downloading SIFT1M dataset...");
        download_sift1m(&config.cache_dir).await?;
    }

    let dataset_dir = config.cache_dir.join("sift1m");

    // Determine limits
    let base_limit = if is_10k {
        config.num_base.unwrap_or(10_000)
    } else {
        config.num_base.unwrap_or(1_000_000)
    };

    let query_limit = config.num_queries.unwrap_or(10_000);

    println!("Loading SIFT dataset: {} base vectors, {} queries...", base_limit, query_limit);

    // Load vectors
    let base_path = dataset_dir.join("sift_base.fvecs");
    let query_path = dataset_dir.join("sift_query.fvecs");
    let gt_path = dataset_dir.join("sift_groundtruth.ivecs");

    let base_vectors = read_fvecs_limited(&base_path, base_limit)?;
    let query_vectors = read_fvecs_limited(&query_path, query_limit)?;

    // For subsets, we need to compute ground truth ourselves since the pre-computed
    // ground truth references indices in the full 1M dataset
    let ground_truth = if base_limit < 100_000 {
        // For small subsets, compute ground truth via brute force
        compute_ground_truth_brute_force(&base_vectors, &query_vectors, config.ground_truth_k)
    } else {
        // For larger datasets, use pre-computed ground truth and filter
        let gt_raw = read_ivecs_limited(&gt_path, query_limit)?;
        gt_raw.into_iter()
            .map(|gt| {
                gt.into_iter()
                    .take(config.ground_truth_k)
                    .filter(|&idx| idx >= 0 && (idx as usize) < base_limit)
                    .map(|idx| idx as usize)
                    .collect()
            })
            .collect()
    };

    let dimensions = base_vectors.first().map(|v| v.len()).unwrap_or(128);

    // Compute average ground truth entries per query for validation
    let avg_gt_entries: f32 = ground_truth.iter()
        .map(|gt| gt.len() as f32)
        .sum::<f32>() / ground_truth.len() as f32;

    println!(
        "Loaded: {} base vectors ({}D), {} queries, avg {:.1} ground truth per query",
        base_vectors.len(),
        dimensions,
        query_vectors.len(),
        avg_gt_entries
    );

    Ok(BenchmarkDataset {
        name: if is_10k { DatasetName::Sift10K } else { DatasetName::Sift1M },
        dimensions,
        base_vectors,
        query_vectors,
        ground_truth,
        ground_truth_distances: None,
    })
}

/// Generate random synthetic dataset for comparison
fn generate_random_dataset(config: &DatasetConfig) -> Result<BenchmarkDataset> {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    let num_base = config.num_base.unwrap_or(10_000);
    let num_queries = config.num_queries.unwrap_or(100);
    let dimensions = 1024; // Default for synthetic

    println!("Generating random dataset: {} base vectors, {} queries...", num_base, num_queries);

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Generate base vectors
    let base_vectors: Vec<Vec<f32>> = (0..num_base)
        .map(|_| (0..dimensions).map(|_| rng.gen::<f32>()).collect())
        .collect();

    // Generate query vectors
    let query_vectors: Vec<Vec<f32>> = (0..num_queries)
        .map(|_| (0..dimensions).map(|_| rng.gen::<f32>()).collect())
        .collect();

    // Compute ground truth via brute force
    println!("Computing ground truth via brute force...");
    let ground_truth: Vec<Vec<usize>> = query_vectors.iter()
        .map(|query| {
            let mut distances: Vec<(f32, usize)> = base_vectors.iter()
                .enumerate()
                .map(|(idx, base)| {
                    let dist: f32 = query.iter()
                        .zip(base.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f32>()
                        .sqrt();
                    (dist, idx)
                })
                .collect();
            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            distances.into_iter()
                .take(config.ground_truth_k)
                .map(|(_, idx)| idx)
                .collect()
        })
        .collect();

    Ok(BenchmarkDataset {
        name: DatasetName::Random,
        dimensions,
        base_vectors,
        query_vectors,
        ground_truth,
        ground_truth_distances: None,
    })
}

// ============================================================================
// Benchmark Metrics
// ============================================================================

/// Metrics collected during benchmark run
#[derive(Debug, Clone, Default)]
pub struct BenchmarkMetrics {
    /// Dataset name
    pub dataset: String,
    /// Algorithm name
    pub algorithm: String,
    /// Search mode (standard, RaBitQ, etc.)
    pub search_mode: String,
    /// Number of indexed vectors
    pub num_vectors: usize,
    /// Number of queries executed
    pub num_queries: usize,
    /// K value for recall
    pub k: usize,
    /// Block cache size in MB
    pub cache_size_mb: usize,
    /// Index build time in seconds
    pub build_time_secs: f64,
    /// Index throughput (vectors/sec)
    pub build_throughput: f64,
    /// Average search latency in milliseconds
    pub avg_search_latency_ms: f64,
    /// P50 search latency
    pub p50_latency_ms: f64,
    /// P99 search latency
    pub p99_latency_ms: f64,
    /// Queries per second
    pub qps: f64,
    /// Average recall@k
    pub recall_at_k: f64,
    /// Disk usage in bytes
    pub disk_usage_bytes: u64,
}

impl BenchmarkMetrics {
    /// Create new metrics instance
    pub fn new(dataset: &str, algorithm: &str) -> Self {
        Self {
            dataset: dataset.to_string(),
            algorithm: algorithm.to_string(),
            ..Default::default()
        }
    }

    /// Calculate latency percentiles from sorted latencies
    pub fn set_latency_percentiles(&mut self, mut latencies: Vec<f64>) {
        if latencies.is_empty() {
            return;
        }
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = latencies.len();
        self.avg_search_latency_ms = latencies.iter().sum::<f64>() / n as f64;
        self.p50_latency_ms = latencies[n / 2];
        self.p99_latency_ms = latencies[(n * 99) / 100];
        self.qps = 1000.0 / self.avg_search_latency_ms;
    }

    /// Print metrics in table format
    pub fn print(&self) {
        println!("\n=== Benchmark Results: {} on {} ===", self.algorithm, self.dataset);
        println!("| Metric | Value |");
        println!("|--------|-------|");
        println!("| Vectors | {} |", self.num_vectors);
        println!("| Queries | {} |", self.num_queries);
        println!("| K | {} |", self.k);
        println!("| Cache Size | {} MB |", self.cache_size_mb);
        if !self.search_mode.is_empty() {
            println!("| Search Mode | {} |", self.search_mode);
        }
        println!("| Build Time | {:.2}s |", self.build_time_secs);
        println!("| Build Throughput | {:.1} vec/s |", self.build_throughput);
        println!("| Avg Latency | {:.2}ms |", self.avg_search_latency_ms);
        println!("| P50 Latency | {:.2}ms |", self.p50_latency_ms);
        println!("| P99 Latency | {:.2}ms |", self.p99_latency_ms);
        println!("| QPS | {:.1} |", self.qps);
        println!("| Recall@{} | {:.4} |", self.k, self.recall_at_k);
        println!("| Disk Usage | {} MB |", self.disk_usage_bytes / 1_000_000);
    }

    /// Format as markdown table row
    pub fn to_markdown_row(&self) -> String {
        format!(
            "| {} | {} | {} | {:.2}s | {:.1}/s | {:.2}ms | {:.1} | {:.4} |",
            self.algorithm,
            self.dataset,
            self.num_vectors,
            self.build_time_secs,
            self.build_throughput,
            self.avg_search_latency_ms,
            self.qps,
            self.recall_at_k
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_name_parsing() {
        assert_eq!(DatasetName::from_str("sift1m"), Some(DatasetName::Sift1M));
        assert_eq!(DatasetName::from_str("SIFT-1M"), Some(DatasetName::Sift1M));
        assert_eq!(DatasetName::from_str("sift10k"), Some(DatasetName::Sift10K));
        assert_eq!(DatasetName::from_str("random"), Some(DatasetName::Random));
        assert_eq!(DatasetName::from_str("unknown"), None);
    }

    #[test]
    fn test_validation_result() {
        let dataset = BenchmarkDataset {
            name: DatasetName::Random,
            dimensions: 128,
            base_vectors: vec![],
            query_vectors: vec![],
            ground_truth: vec![vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]],
            ground_truth_distances: None,
        };

        // Perfect recall
        let results = vec![
            (0.1, 0), (0.2, 1), (0.3, 2), (0.4, 3), (0.5, 4),
            (0.6, 5), (0.7, 6), (0.8, 7), (0.9, 8), (1.0, 9),
        ];
        let validation = dataset.validate_results(0, &results, 10);
        assert_eq!(validation.recall, 1.0);
        assert_eq!(validation.true_positives.len(), 10);
        assert_eq!(validation.false_positives.len(), 0);
        assert_eq!(validation.false_negatives.len(), 0);

        // 50% recall
        let results = vec![
            (0.1, 0), (0.2, 1), (0.3, 2), (0.4, 3), (0.5, 4),
            (0.6, 100), (0.7, 101), (0.8, 102), (0.9, 103), (1.0, 104),
        ];
        let validation = dataset.validate_results(0, &results, 10);
        assert_eq!(validation.recall, 0.5);
        assert_eq!(validation.true_positives.len(), 5);
        assert_eq!(validation.false_positives.len(), 5);
        assert_eq!(validation.false_negatives.len(), 5);
    }
}
