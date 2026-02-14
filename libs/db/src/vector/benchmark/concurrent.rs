//! Concurrent benchmark infrastructure for vector search.
//!
//! This module provides tools for measuring performance under concurrent load
//! using the proper MPSC (writes) and MPMC (reads) channel architecture.
//!
//! ## Architecture
//!
//! - **Writes**: Multiple producer tasks send to a single MPSC Writer channel.
//!   A single consumer processes all mutations sequentially.
//! - **Reads**: Multiple producer tasks send queries through an MPMC channel.
//!   A configurable pool of query workers processes searches in parallel.
//!
//! ## Components
//!
//! - [`ConcurrentMetrics`] - Lock-free metrics collection with atomic counters
//! - [`BenchConfig`] - Benchmark configuration (producers, workers, duration)
//! - [`ConcurrentBenchmark`] - Async benchmark runner
//! - [`DatasetSource`] - LAION or random vector generation
//! - [`SearchMode`] - RaBitQ or exact search strategy
//!
//! ## Throughput vs Quality Measurement
//!
//! Concurrent benchmarks measure **throughput only** (ops/sec, latency percentiles).
//! Recall measurement is intentionally omitted because:
//! - Tracking query/result pairs under concurrent load adds overhead
//! - Ground truth computation interferes with throughput measurement
//! - Quality baselines should isolate recall from concurrency effects
//!
//! For recall measurement, use the dedicated CLI path:
//! ```bash
//! cargo run --release --bin bench_vector -- sweep --dataset laion --assert-recall 0.80
//! ```
//!
//! # Example
//!
//! ```ignore
//! use motlie_db::vector::benchmark::concurrent::{BenchConfig, ConcurrentBenchmark, DatasetSource};
//!
//! let config = BenchConfig::balanced()
//!     .with_dataset(DatasetSource::Random { seed: 42 });
//! let bench = ConcurrentBenchmark::new(config);
//! let result = bench.run(storage, embedding_code).await?;
//! println!("Insert: {:.1}/s, Search: {:.1}/s", result.insert_throughput, result.search_throughput);
//! ```

use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use tokio::task::JoinHandle;

use crate::reader::Runnable as QueryRunnable;
use crate::vector::benchmark::dataset::LAION_EMBEDDING_DIM;
use crate::vector::schema::{EmbeddingCode, ExternalKey};
use crate::vector::writer::{create_writer, spawn_mutation_consumer_with_storage_autoreg, WriterConfig};
use crate::vector::reader::{create_reader_with_storage, spawn_query_consumers_with_storage_autoreg, ReaderConfig};
use crate::vector::{
    Distance, EmbeddingBuilder, InsertVector, InsertVectorBatch, MutationRunnable, SearchKNN,
    Storage,
};
use crate::Id;

// ============================================================================
// DatasetSource - Vector data source (LAION or random)
// ============================================================================

/// Source of vectors for benchmarking.
///
/// LAION provides real embeddings with ground truth for recall measurement.
/// Random vectors are useful for stress testing but cannot measure recall.
#[derive(Debug, Clone)]
pub enum DatasetSource {
    /// Generate random vectors (no recall measurement possible).
    Random {
        /// Random seed for reproducibility.
        seed: u64,
    },
    /// Use LAION-400M CLIP embeddings with ground truth.
    ///
    /// Requires downloading LAION embeddings first:
    /// `LaionDataset::download(&data_dir)?`
    Laion {
        /// Directory containing img_emb_0.npy and text_emb_0.npy.
        data_dir: PathBuf,
    },
}

impl Default for DatasetSource {
    fn default() -> Self {
        Self::Random { seed: 42 }
    }
}

// ============================================================================
// SearchMode - Search strategy configuration
// ============================================================================

/// Search strategy for benchmark queries.
///
/// Controls whether RaBitQ approximation or exact distance is used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchMode {
    /// Exact distance computation (no approximation).
    /// Works with all distance metrics.
    Exact,
    /// RaBitQ binary quantization with reranking.
    /// Only valid for Cosine distance.
    RaBitQ {
        /// Bits per dimension (1, 2, or 4).
        /// Higher bits = better recall, more memory.
        bits: u8,
    },
}

impl Default for SearchMode {
    fn default() -> Self {
        Self::Exact
    }
}

impl fmt::Display for SearchMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchMode::Exact => write!(f, "Exact"),
            SearchMode::RaBitQ { bits } => write!(f, "RaBitQ-{}bit", bits),
        }
    }
}

// ============================================================================
// ConcurrentMetrics - Lock-free metrics collection
// ============================================================================

/// Number of histogram buckets (log2 scale from 1us to ~1s)
const NUM_BUCKETS: usize = 20;

/// Lock-free metrics collector for concurrent benchmarking.
///
/// Uses atomic operations for all updates, enabling accurate metrics
/// collection without synchronization overhead.
///
/// # Histogram Buckets
///
/// Latency is tracked in log2 buckets:
/// - Bucket 0: 0-1us
/// - Bucket 1: 1-2us
/// - Bucket 2: 2-4us
/// - ...
/// - Bucket 19: 512ms+
#[derive(Debug)]
pub struct ConcurrentMetrics {
    // Operation counts
    insert_count: AtomicU64,
    search_count: AtomicU64,
    delete_count: AtomicU64,
    error_count: AtomicU64,

    // Latency totals (nanoseconds)
    insert_total_ns: AtomicU64,
    search_total_ns: AtomicU64,
    delete_total_ns: AtomicU64,

    // Latency histograms (log2 buckets)
    insert_latency_buckets: [AtomicU64; NUM_BUCKETS],
    search_latency_buckets: [AtomicU64; NUM_BUCKETS],
    delete_latency_buckets: [AtomicU64; NUM_BUCKETS],
}

impl ConcurrentMetrics {
    /// Create a new metrics collector with all counters at zero.
    pub fn new() -> Self {
        Self {
            insert_count: AtomicU64::new(0),
            search_count: AtomicU64::new(0),
            delete_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            insert_total_ns: AtomicU64::new(0),
            search_total_ns: AtomicU64::new(0),
            delete_total_ns: AtomicU64::new(0),
            insert_latency_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            search_latency_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            delete_latency_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }

    /// Record an insert operation with its latency.
    pub fn record_insert(&self, latency: Duration) {
        self.insert_count.fetch_add(1, Ordering::Relaxed);
        self.insert_total_ns
            .fetch_add(latency.as_nanos() as u64, Ordering::Relaxed);
        let bucket = latency_to_bucket(latency);
        self.insert_latency_buckets[bucket].fetch_add(1, Ordering::Relaxed);
    }

    /// Record a search operation with its latency.
    pub fn record_search(&self, latency: Duration) {
        self.search_count.fetch_add(1, Ordering::Relaxed);
        self.search_total_ns
            .fetch_add(latency.as_nanos() as u64, Ordering::Relaxed);
        let bucket = latency_to_bucket(latency);
        self.search_latency_buckets[bucket].fetch_add(1, Ordering::Relaxed);
    }

    /// Record a delete operation with its latency.
    pub fn record_delete(&self, latency: Duration) {
        self.delete_count.fetch_add(1, Ordering::Relaxed);
        self.delete_total_ns
            .fetch_add(latency.as_nanos() as u64, Ordering::Relaxed);
        let bucket = latency_to_bucket(latency);
        self.delete_latency_buckets[bucket].fetch_add(1, Ordering::Relaxed);
    }

    /// Record an error (any operation type).
    pub fn record_error(&self) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get total insert count.
    pub fn insert_count(&self) -> u64 {
        self.insert_count.load(Ordering::Relaxed)
    }

    /// Get total search count.
    pub fn search_count(&self) -> u64 {
        self.search_count.load(Ordering::Relaxed)
    }

    /// Get total delete count.
    pub fn delete_count(&self) -> u64 {
        self.delete_count.load(Ordering::Relaxed)
    }

    /// Get total error count.
    pub fn error_count(&self) -> u64 {
        self.error_count.load(Ordering::Relaxed)
    }

    /// Calculate percentile latency for insert operations.
    pub fn insert_percentile(&self, p: f64) -> Duration {
        percentile_from_histogram(&self.insert_latency_buckets, p)
    }

    /// Calculate percentile latency for search operations.
    pub fn search_percentile(&self, p: f64) -> Duration {
        percentile_from_histogram(&self.search_latency_buckets, p)
    }

    /// Calculate percentile latency for delete operations.
    pub fn delete_percentile(&self, p: f64) -> Duration {
        percentile_from_histogram(&self.delete_latency_buckets, p)
    }

    /// Calculate average insert latency.
    pub fn insert_avg(&self) -> Duration {
        let count = self.insert_count.load(Ordering::Relaxed);
        if count == 0 {
            return Duration::ZERO;
        }
        let total_ns = self.insert_total_ns.load(Ordering::Relaxed);
        Duration::from_nanos(total_ns / count)
    }

    /// Calculate average search latency.
    pub fn search_avg(&self) -> Duration {
        let count = self.search_count.load(Ordering::Relaxed);
        if count == 0 {
            return Duration::ZERO;
        }
        let total_ns = self.search_total_ns.load(Ordering::Relaxed);
        Duration::from_nanos(total_ns / count)
    }

    /// Generate a summary of all metrics.
    pub fn summary(&self) -> MetricsSummary {
        MetricsSummary {
            insert_count: self.insert_count(),
            search_count: self.search_count(),
            delete_count: self.delete_count(),
            error_count: self.error_count(),
            insert_avg: self.insert_avg(),
            insert_p50: self.insert_percentile(0.50),
            insert_p95: self.insert_percentile(0.95),
            insert_p99: self.insert_percentile(0.99),
            search_avg: self.search_avg(),
            search_p50: self.search_percentile(0.50),
            search_p95: self.search_percentile(0.95),
            search_p99: self.search_percentile(0.99),
        }
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.insert_count.store(0, Ordering::Relaxed);
        self.search_count.store(0, Ordering::Relaxed);
        self.delete_count.store(0, Ordering::Relaxed);
        self.error_count.store(0, Ordering::Relaxed);
        self.insert_total_ns.store(0, Ordering::Relaxed);
        self.search_total_ns.store(0, Ordering::Relaxed);
        self.delete_total_ns.store(0, Ordering::Relaxed);
        for bucket in &self.insert_latency_buckets {
            bucket.store(0, Ordering::Relaxed);
        }
        for bucket in &self.search_latency_buckets {
            bucket.store(0, Ordering::Relaxed);
        }
        for bucket in &self.delete_latency_buckets {
            bucket.store(0, Ordering::Relaxed);
        }
    }
}

impl Default for ConcurrentMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert latency to histogram bucket index (log2 scale).
fn latency_to_bucket(latency: Duration) -> usize {
    let micros = latency.as_micros() as u64;
    if micros == 0 {
        return 0;
    }
    // log2(micros) gives us the bucket, capped at NUM_BUCKETS-1
    let bucket = (64 - micros.leading_zeros()) as usize;
    bucket.min(NUM_BUCKETS - 1)
}

/// Convert bucket index back to approximate latency (lower bound).
fn bucket_to_latency(bucket: usize) -> Duration {
    if bucket == 0 {
        Duration::from_micros(0)
    } else {
        Duration::from_micros(1 << (bucket - 1))
    }
}

/// Calculate percentile from histogram buckets.
fn percentile_from_histogram(buckets: &[AtomicU64; NUM_BUCKETS], p: f64) -> Duration {
    let total: u64 = buckets.iter().map(|b| b.load(Ordering::Relaxed)).sum();
    if total == 0 {
        return Duration::ZERO;
    }

    let target = (total as f64 * p) as u64;
    let mut cumulative = 0u64;

    for (i, bucket) in buckets.iter().enumerate() {
        cumulative += bucket.load(Ordering::Relaxed);
        if cumulative >= target {
            return bucket_to_latency(i + 1); // Upper bound of bucket
        }
    }

    bucket_to_latency(NUM_BUCKETS)
}

// ============================================================================
// MetricsSummary - Snapshot of metrics at a point in time
// ============================================================================

/// Summary of concurrent benchmark metrics.
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    pub insert_count: u64,
    pub search_count: u64,
    pub delete_count: u64,
    pub error_count: u64,
    pub insert_avg: Duration,
    pub insert_p50: Duration,
    pub insert_p95: Duration,
    pub insert_p99: Duration,
    pub search_avg: Duration,
    pub search_p50: Duration,
    pub search_p95: Duration,
    pub search_p99: Duration,
}

impl fmt::Display for MetricsSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Concurrent Metrics Summary ===")?;
        writeln!(f, "Operations:")?;
        writeln!(f, "  Inserts: {}", self.insert_count)?;
        writeln!(f, "  Searches: {}", self.search_count)?;
        writeln!(f, "  Deletes: {}", self.delete_count)?;
        writeln!(f, "  Errors: {}", self.error_count)?;
        writeln!(f, "Insert Latency:")?;
        writeln!(f, "  Avg: {:?}", self.insert_avg)?;
        writeln!(f, "  P50: {:?}", self.insert_p50)?;
        writeln!(f, "  P95: {:?}", self.insert_p95)?;
        writeln!(f, "  P99: {:?}", self.insert_p99)?;
        writeln!(f, "Search Latency:")?;
        writeln!(f, "  Avg: {:?}", self.search_avg)?;
        writeln!(f, "  P50: {:?}", self.search_p50)?;
        writeln!(f, "  P95: {:?}", self.search_p95)?;
        writeln!(f, "  P99: {:?}", self.search_p99)?;
        Ok(())
    }
}

// ============================================================================
// BenchConfig - Configuration for concurrent benchmarks
// ============================================================================

/// Configuration for concurrent benchmarks.
///
/// ## Channel Architecture
///
/// - `insert_producers`: Number of async tasks sending inserts to the MPSC Writer.
///   All producers share a single channel with ONE consumer processing mutations.
/// - `query_workers`: Number of query consumer workers in the MPMC pool.
///   Searches are distributed across workers for parallel processing.
/// - `search_producers`: Number of async tasks sending search queries.
///
/// ## Dataset and Search Mode
///
/// - `dataset`: LAION (with ground truth for recall) or Random vectors.
/// - `search_mode`: Exact distance or RaBitQ approximation (2-bit/4-bit).
/// - `distance`: Distance metric (Cosine required for RaBitQ).
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Number of insert producer tasks (all send to single MPSC consumer).
    pub insert_producers: usize,
    /// Number of search producer tasks.
    pub search_producers: usize,
    /// Number of query worker threads (MPMC consumers).
    pub query_workers: usize,
    /// Total benchmark duration.
    pub duration: Duration,
    /// Vectors to insert per producer task.
    pub vectors_per_producer: usize,
    /// Vector dimension.
    pub vector_dim: usize,
    /// K for search (top-k results).
    pub k: usize,
    /// ef_search parameter for HNSW.
    pub ef_search: usize,
    /// HNSW M parameter.
    pub hnsw_m: usize,
    /// HNSW ef_construction parameter.
    pub hnsw_ef_construction: usize,
    /// Dataset source (LAION or Random).
    pub dataset: DatasetSource,
    /// Search mode (Exact or RaBitQ).
    pub search_mode: SearchMode,
    /// Distance metric (Cosine required for RaBitQ).
    pub distance: Distance,
    /// Number of queries for recall measurement (LAION only).
    pub num_queries: usize,
    /// Rerank factor for RaBitQ (candidates = k * rerank_factor).
    pub rerank_factor: usize,
    /// Batch size for insert commits (Task 8.2.4).
    ///
    /// When > 1, inserts are batched and committed together for higher throughput.
    /// Default is 1 (per-vector commits) for correctness validation.
    /// Use higher values (e.g., 100) for pure throughput measurement.
    pub batch_size: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            insert_producers: 4,
            search_producers: 4,
            query_workers: 4,
            duration: Duration::from_secs(30),
            vectors_per_producer: 1000,
            vector_dim: 128,
            k: 10,
            ef_search: 100,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            dataset: DatasetSource::default(),
            search_mode: SearchMode::default(),
            distance: Distance::L2,
            num_queries: 100,
            rerank_factor: 10,
            batch_size: 1, // Per-vector commits by default for correctness
        }
    }
}

impl BenchConfig {
    /// Read-heavy: 1 insert producer, 8 search producers, 4 query workers.
    pub fn read_heavy() -> Self {
        Self {
            insert_producers: 1,
            search_producers: 8,
            query_workers: 4,
            ..Default::default()
        }
    }

    /// Write-heavy: 4 insert producers, 1 search producer, 1 query worker.
    pub fn write_heavy() -> Self {
        Self {
            insert_producers: 4,
            search_producers: 1,
            query_workers: 1,
            ..Default::default()
        }
    }

    /// Balanced: 4 insert producers, 4 search producers, 4 query workers.
    pub fn balanced() -> Self {
        Self {
            insert_producers: 4,
            search_producers: 4,
            query_workers: 4,
            ..Default::default()
        }
    }

    /// Stress: 8 insert producers, 8 search producers, 8 query workers.
    pub fn stress() -> Self {
        Self {
            insert_producers: 8,
            search_producers: 8,
            query_workers: 8,
            duration: Duration::from_secs(60),
            ..Default::default()
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Builder methods for dataset and search configuration
    // ─────────────────────────────────────────────────────────────

    /// Set dataset source.
    pub fn with_dataset(mut self, dataset: DatasetSource) -> Self {
        self.dataset = dataset;
        // Auto-adjust dimension for LAION
        if matches!(self.dataset, DatasetSource::Laion { .. }) {
            self.vector_dim = LAION_EMBEDDING_DIM;
            // LAION uses Cosine distance
            self.distance = Distance::Cosine;
        }
        self
    }

    /// Set search mode (Exact or RaBitQ).
    pub fn with_search_mode(mut self, mode: SearchMode) -> Self {
        self.search_mode = mode;
        // RaBitQ requires Cosine distance
        if matches!(mode, SearchMode::RaBitQ { .. }) {
            self.distance = Distance::Cosine;
        }
        self
    }

    /// Set distance metric.
    pub fn with_distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }

    /// Set number of queries for recall measurement.
    pub fn with_num_queries(mut self, num_queries: usize) -> Self {
        self.num_queries = num_queries;
        self
    }

    /// Set rerank factor for RaBitQ.
    pub fn with_rerank_factor(mut self, factor: usize) -> Self {
        self.rerank_factor = factor;
        self
    }

    /// Set ef_search parameter.
    pub fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }

    /// Set k (number of results).
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set vectors per producer.
    pub fn with_vectors_per_producer(mut self, count: usize) -> Self {
        self.vectors_per_producer = count;
        self
    }

    /// Set batch size for insert commits (Task 8.2.4).
    ///
    /// Batching inserts reduces transaction overhead and increases throughput.
    /// Use batch_size=1 (default) for correctness validation where each insert
    /// is committed independently. Use larger values (e.g., 100) for pure
    /// throughput measurement.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // High throughput mode: batch 100 inserts per commit
    /// let config = BenchConfig::balanced()
    ///     .with_batch_size(100);
    /// ```
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1); // Minimum of 1
        self
    }

    // Legacy compatibility aliases
    #[doc(hidden)]
    pub fn writer_threads(&self) -> usize {
        self.insert_producers
    }

    #[doc(hidden)]
    pub fn reader_threads(&self) -> usize {
        self.search_producers
    }

    /// Check if this config uses LAION dataset (recall can be measured).
    pub fn has_ground_truth(&self) -> bool {
        matches!(self.dataset, DatasetSource::Laion { .. })
    }

    /// Validate configuration for consistency.
    pub fn validate(&self) -> Result<()> {
        // RaBitQ requires Cosine distance
        if let SearchMode::RaBitQ { bits } = self.search_mode {
            if self.distance != Distance::Cosine {
                anyhow::bail!(
                    "RaBitQ requires Cosine distance, got {:?}",
                    self.distance
                );
            }
            if ![1, 2, 4].contains(&bits) {
                anyhow::bail!(
                    "RaBitQ bits must be 1, 2, or 4, got {}",
                    bits
                );
            }
        }
        Ok(())
    }
}

// ============================================================================
// BenchResult - Results from a concurrent benchmark run
// ============================================================================

/// Results from a concurrent benchmark run.
#[derive(Debug, Clone)]
pub struct BenchResult {
    /// Benchmark configuration used.
    pub config: BenchConfig,
    /// Actual benchmark duration.
    pub duration: Duration,
    /// Metrics summary.
    pub metrics: MetricsSummary,
    /// Insert throughput (ops/sec).
    pub insert_throughput: f64,
    /// Search throughput (ops/sec).
    pub search_throughput: f64,
    /// Recall@k (None if no ground truth available).
    pub recall_at_k: Option<f64>,
    /// Number of queries used for recall measurement.
    pub recall_queries: Option<usize>,
}

impl BenchResult {
    /// Format recall as percentage string.
    pub fn recall_percent(&self) -> String {
        match self.recall_at_k {
            Some(r) => format!("{:.1}%", r * 100.0),
            None => "N/A".to_string(),
        }
    }

    /// CSV header for benchmark results.
    pub fn csv_header() -> &'static str {
        "scenario,insert_producers,search_producers,query_workers,duration_s,insert_count,search_count,error_count,insert_throughput,search_throughput,insert_p50_us,insert_p99_us,search_p50_us,search_p99_us"
    }

    /// Convert result to CSV row.
    pub fn to_csv_row(&self, scenario: &str) -> String {
        format!(
            "{},{},{},{},{:.2},{},{},{},{:.1},{:.1},{},{},{},{}",
            scenario,
            self.config.insert_producers,
            self.config.search_producers,
            self.config.query_workers,
            self.duration.as_secs_f64(),
            self.metrics.insert_count,
            self.metrics.search_count,
            self.metrics.error_count,
            self.insert_throughput,
            self.search_throughput,
            self.metrics.insert_p50.as_micros(),
            self.metrics.insert_p99.as_micros(),
            self.metrics.search_p50.as_micros(),
            self.metrics.search_p99.as_micros(),
        )
    }
}

/// Save multiple benchmark results to a CSV file.
pub fn save_benchmark_results_csv(
    results: &[(String, BenchResult)],
    csv_path: &std::path::Path,
) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(csv_path)?;
    writeln!(file, "{}", BenchResult::csv_header())?;
    for (scenario, result) in results {
        writeln!(file, "{}", result.to_csv_row(scenario))?;
    }
    Ok(())
}

impl fmt::Display for BenchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Concurrent Benchmark Results ===")?;
        writeln!(f, "Configuration:")?;
        writeln!(f, "  Insert Producers: {} (MPSC → 1 consumer)", self.config.insert_producers)?;
        writeln!(f, "  Search Producers: {}", self.config.search_producers)?;
        writeln!(f, "  Query Workers: {} (MPMC pool)", self.config.query_workers)?;
        writeln!(f, "  Search Mode: {}", self.config.search_mode)?;
        writeln!(f, "  Distance: {:?}", self.config.distance)?;
        writeln!(f, "  Duration: {:?}", self.duration)?;
        writeln!(f, "Throughput:")?;
        writeln!(f, "  Inserts: {:.1} ops/sec", self.insert_throughput)?;
        writeln!(f, "  Searches: {:.1} ops/sec", self.search_throughput)?;
        if let Some(recall) = self.recall_at_k {
            writeln!(f, "Recall:")?;
            writeln!(f, "  Recall@{}: {:.1}%", self.config.k, recall * 100.0)?;
            if let Some(n) = self.recall_queries {
                writeln!(f, "  Queries: {}", n)?;
            }
        }
        writeln!(f, "{}", self.metrics)?;
        Ok(())
    }
}

// ============================================================================
// ConcurrentBenchmark - Async benchmark runner using channel API
// ============================================================================

/// Concurrent benchmark runner using MPSC/MPMC channel architecture.
///
/// This benchmark correctly uses the production channel infrastructure:
/// - Inserts go through Writer (MPSC) → single mutation consumer
/// - Searches go through Reader (MPMC) → query worker pool
pub struct ConcurrentBenchmark {
    config: BenchConfig,
}

impl ConcurrentBenchmark {
    /// Create a new benchmark with the given configuration.
    pub fn new(config: BenchConfig, _metrics: Arc<ConcurrentMetrics>) -> Self {
        // Note: metrics parameter kept for API compatibility but we create fresh metrics internally
        Self { config }
    }

    /// Run the benchmark against the given storage.
    ///
    /// This method:
    /// 1. Registers an embedding for the benchmark
    /// 2. Creates Writer (MPSC) and Reader (MPMC) channels
    /// 3. Spawns insert producer tasks (all send to same Writer)
    /// 4. Spawns search producer tasks
    /// 5. Waits for duration or completion
    /// 6. Returns aggregated metrics
    pub async fn run(
        &self,
        storage: Arc<Storage>,
        embedding_code: EmbeddingCode,
    ) -> Result<BenchResult> {
        let start = Instant::now();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let metrics = Arc::new(ConcurrentMetrics::new());

        // Validate configuration before running
        self.config.validate()?;

        // Register embedding for this benchmark
        let registry = storage.cache().clone();

        let embedding_name = format!("bench-{}", embedding_code);
        let embedding = registry
            .register(
                EmbeddingBuilder::new(&embedding_name, self.config.vector_dim as u32, self.config.distance)
                    .with_hnsw_m(self.config.hnsw_m as u16)
                    .with_hnsw_ef_construction(self.config.hnsw_ef_construction as u16),
            )?;

        // Create Writer (MPSC) - all insert producers share this
        let (writer, writer_rx) = create_writer(WriterConfig {
            channel_buffer_size: 10000,
        });
        let mutation_handle = spawn_mutation_consumer_with_storage_autoreg(
            writer_rx,
            WriterConfig::default(),
            storage.clone(),
        );

        // Create Reader (MPMC) - query workers share this
        let (search_reader, reader_rx) = create_reader_with_storage(ReaderConfig::default());
        let query_handles = spawn_query_consumers_with_storage_autoreg(
            reader_rx,
            ReaderConfig::default(),
            storage.clone(),
            self.config.query_workers,
        );

        let mut producer_handles: Vec<JoinHandle<()>> = Vec::new();

        // Spawn insert producer tasks
        for producer_id in 0..self.config.insert_producers {
            let writer = writer.clone();
            let embedding = embedding.clone();
            let metrics = Arc::clone(&metrics);
            let stop = Arc::clone(&stop_flag);
            let vectors_per_producer = self.config.vectors_per_producer;
            let dim = self.config.vector_dim;
            let duration = self.config.duration;
            let deadline = start + duration;
            let batch_size = self.config.batch_size;

            producer_handles.push(tokio::spawn(async move {
                insert_producer_workload(
                    writer,
                    embedding,
                    metrics,
                    stop,
                    vectors_per_producer,
                    dim,
                    producer_id,
                    batch_size,
                    deadline,
                )
                .await;
            }));
        }

        // Spawn search producer tasks
        for producer_id in 0..self.config.search_producers {
            let search_reader = search_reader.clone();
            let embedding = embedding.clone();
            let metrics = Arc::clone(&metrics);
            let stop = Arc::clone(&stop_flag);
            let dim = self.config.vector_dim;
            let k = self.config.k;
            let ef = self.config.ef_search;
            let duration = self.config.duration;
            let deadline = start + duration;

            producer_handles.push(tokio::spawn(async move {
                search_producer_workload(
                    search_reader,
                    embedding,
                    metrics,
                    stop,
                    dim,
                    k,
                    ef,
                    deadline,
                    producer_id,
                )
                .await;
            }));
        }

        // Wait for benchmark duration
        tokio::time::sleep(self.config.duration).await;
        stop_flag.store(true, Ordering::Relaxed);

        // Wait for all producer tasks to finish
        for handle in producer_handles {
            let _ = handle.await;
        }

        // Flush writer to ensure all mutations are processed
        let _ = writer.flush().await;

        // Drop writer and search_reader to close channels
        drop(writer);
        drop(search_reader);

        // Wait for consumers to finish (they exit when channel closes)
        let _ = mutation_handle.await;
        for handle in query_handles {
            let _ = handle.await;
        }

        let duration = start.elapsed();
        let summary = metrics.summary();

        let insert_throughput = if duration.as_secs_f64() > 0.0 {
            summary.insert_count as f64 / duration.as_secs_f64()
        } else {
            0.0
        };

        let search_throughput = if duration.as_secs_f64() > 0.0 {
            summary.search_count as f64 / duration.as_secs_f64()
        } else {
            0.0
        };

        // Concurrent benchmarks intentionally omit recall measurement.
        // Recall tracking adds overhead and interferes with throughput measurement.
        // Use `bench_vector sweep --dataset laion` for quality baselines.
        let recall_at_k = None;
        let recall_queries = None;

        Ok(BenchResult {
            config: self.config.clone(),
            duration,
            metrics: summary,
            insert_throughput,
            search_throughput,
            recall_at_k,
            recall_queries,
        })
    }
}

// ============================================================================
// Producer workload functions (async)
// ============================================================================

use crate::vector::writer::Writer;
use crate::vector::reader::Reader;
use crate::vector::Embedding;

/// Insert producer workload: sends inserts to Writer channel until deadline.
async fn insert_producer_workload(
    writer: Writer,
    embedding: Embedding,
    metrics: Arc<ConcurrentMetrics>,
    stop: Arc<AtomicBool>,
    max_vectors: usize,
    dim: usize,
    producer_id: usize,
    batch_size: usize,
    deadline: Instant,
) {
    // Use StdRng with seed based on producer_id for reproducibility and Send-safety
    let mut rng = StdRng::seed_from_u64(producer_id as u64 + 1000);

    let batch_size = batch_size.max(1);
    // T7.1: Use ExternalKey for polymorphic ID mapping
    let mut batch: Vec<(ExternalKey, Vec<f32>)> = Vec::with_capacity(batch_size);

    for _i in 0..max_vectors {
        if stop.load(Ordering::Relaxed) || Instant::now() >= deadline {
            break;
        }

        // Generate random vector
        let vector: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
        let id = Id::new();

        if batch_size == 1 {
            let op_start = Instant::now();
            // T7.1: Use ExternalKey::NodeId for polymorphic ID mapping
            let result = InsertVector::new(&embedding, ExternalKey::NodeId(id), vector)
                .immediate()
                .run(&writer)
                .await;
            let latency = op_start.elapsed();
            match result {
                Ok(()) => metrics.record_insert(latency),
                Err(_) => metrics.record_error(),
            }
            continue;
        }

        // T7.1: Use ExternalKey::NodeId for polymorphic ID mapping
        batch.push((ExternalKey::NodeId(id), vector));
        if batch.len() >= batch_size {
            let current = std::mem::take(&mut batch);
            let batch_len = current.len();
            let op_start = Instant::now();
            let result = InsertVectorBatch::new(&embedding, current)
                .immediate()
                .run(&writer)
                .await;
            let latency = op_start.elapsed();

            match result {
                Ok(()) => {
                    let per_ns = latency.as_nanos() / batch_len as u128;
                    let per_ns = per_ns.min(u128::from(u64::MAX));
                    let per_vec = Duration::from_nanos(per_ns as u64);
                    for _ in 0..batch_len {
                        metrics.record_insert(per_vec);
                    }
                }
                Err(_) => {
                    for _ in 0..batch_len {
                        metrics.record_error();
                    }
                }
            }
        }
    }

    if batch_size > 1 && !batch.is_empty() {
        let current = std::mem::take(&mut batch);
        let batch_len = current.len();
        let op_start = Instant::now();
        let result = InsertVectorBatch::new(&embedding, current)
            .immediate()
            .run(&writer)
            .await;
        let latency = op_start.elapsed();

        match result {
            Ok(()) => {
                let per_ns = latency.as_nanos() / batch_len as u128;
                let per_ns = per_ns.min(u128::from(u64::MAX));
                let per_vec = Duration::from_nanos(per_ns as u64);
                for _ in 0..batch_len {
                    metrics.record_insert(per_vec);
                }
            }
            Err(_) => {
                for _ in 0..batch_len {
                    metrics.record_error();
                }
            }
        }
    }
}

/// Search producer workload: sends searches to Reader until deadline.
async fn search_producer_workload(
    search_reader: Reader,
    embedding: Embedding,
    metrics: Arc<ConcurrentMetrics>,
    stop: Arc<AtomicBool>,
    dim: usize,
    k: usize,
    ef: usize,
    deadline: Instant,
    producer_id: usize,
) {
    // Use StdRng with seed based on producer_id for reproducibility and Send-safety
    let mut rng = StdRng::seed_from_u64(producer_id as u64 + 2000);
    let timeout = Duration::from_secs(5);

    while !stop.load(Ordering::Relaxed) && Instant::now() < deadline {
        // Generate random query
        let query: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();

        let op_start = Instant::now();

        // Send search through Reader channel (MPMC)
        let result = SearchKNN::new(&embedding, query, k)
            .with_ef(ef)
            .run(&search_reader, timeout)
            .await;

        let latency = op_start.elapsed();

        match result {
            Ok(_) => metrics.record_search(latency),
            Err(_) => metrics.record_error(),
        }
    }
}

// ============================================================================
// Sync vs Async Insert Latency Comparison
// ============================================================================

/// Result of sync vs async insert latency comparison.
#[derive(Debug, Clone)]
pub struct SyncAsyncLatencyResult {
    /// Number of vectors inserted for each mode
    pub num_vectors: usize,
    /// Sync insert latency P50 (immediate graph build)
    pub sync_p50: Duration,
    /// Sync insert latency P99
    pub sync_p99: Duration,
    /// Async insert latency P50 (deferred graph build)
    pub async_p50: Duration,
    /// Async insert latency P99
    pub async_p99: Duration,
    /// Speedup factor (sync P50 / async P50)
    pub speedup_p50: f64,
    /// Speedup factor (sync P99 / async P99)
    pub speedup_p99: f64,
}

impl fmt::Display for SyncAsyncLatencyResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Sync vs Async Insert Latency Comparison ===")?;
        writeln!(f, "Vectors per mode: {}", self.num_vectors)?;
        writeln!(f)?;
        writeln!(f, "Sync Insert (immediate graph build):")?;
        writeln!(f, "  P50: {:?}", self.sync_p50)?;
        writeln!(f, "  P99: {:?}", self.sync_p99)?;
        writeln!(f)?;
        writeln!(f, "Async Insert (deferred graph build):")?;
        writeln!(f, "  P50: {:?}", self.async_p50)?;
        writeln!(f, "  P99: {:?}", self.async_p99)?;
        writeln!(f)?;
        writeln!(f, "Speedup:")?;
        writeln!(f, "  P50: {:.1}x faster", self.speedup_p50)?;
        writeln!(f, "  P99: {:.1}x faster", self.speedup_p99)?;
        Ok(())
    }
}

/// Compare sync vs async insert latency.
///
/// This benchmark measures the latency difference between:
/// - **Sync insert** (build_index=true): Builds HNSW graph edges immediately
/// - **Async insert** (build_index=false): Stores vector, adds to pending queue, returns immediately
///
/// # Arguments
/// * `storage` - Vector storage
/// * `embedding_code` - Embedding space code
/// * `num_vectors` - Number of vectors to insert per mode
/// * `dim` - Vector dimension
///
/// # Returns
/// `SyncAsyncLatencyResult` with P50/P99 latency for both modes and speedup factors.
///
/// # Example
///
/// ```ignore
/// let result = compare_sync_async_latency(storage.clone(), embedding_code, 1000, 128).await?;
/// println!("{}", result);
/// // Output:
/// // Sync Insert (immediate graph build):
/// //   P50: 5.2ms
/// //   P99: 48.3ms
/// // Async Insert (deferred graph build):
/// //   P50: 0.4ms
/// //   P99: 2.1ms
/// // Speedup:
/// //   P50: 13.0x faster
/// //   P99: 23.0x faster
/// ```
pub async fn compare_sync_async_latency(
    storage: Arc<Storage>,
    embedding_code: EmbeddingCode,
    num_vectors: usize,
    dim: usize,
) -> Result<SyncAsyncLatencyResult> {
    // Create writer and spawn consumer
    let (writer, receiver) = create_writer(WriterConfig::default());
    let _consumer_handle = spawn_mutation_consumer_with_storage_autoreg(
        receiver,
        WriterConfig::default(),
        storage.clone(),
    );

    // Get embedding from registry
    let embedding = storage
        .cache()
        .get_by_code(embedding_code)
        .ok_or_else(|| anyhow::anyhow!("Embedding not found for code {}", embedding_code))?;

    // Metrics for sync inserts
    let sync_metrics = ConcurrentMetrics::new();
    let mut rng = StdRng::seed_from_u64(42);

    // Phase 1: Sync inserts (immediate graph build)
    tracing::info!(num_vectors, "Starting sync insert phase (immediate graph build)");
    for _ in 0..num_vectors {
        let vector: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
        let id = Id::new();

        let op_start = Instant::now();
        // T7.1: Use ExternalKey::NodeId for polymorphic ID mapping
        InsertVector::new(&embedding, ExternalKey::NodeId(id), vector)
            .immediate() // build_index = true (sync)
            .run(&writer)
            .await?;
        sync_metrics.record_insert(op_start.elapsed());
    }
    writer.flush().await?;

    // Metrics for async inserts
    let async_metrics = ConcurrentMetrics::new();
    let mut rng = StdRng::seed_from_u64(43); // Different seed

    // Phase 2: Async inserts (deferred graph build)
    tracing::info!(num_vectors, "Starting async insert phase (deferred graph build)");
    for _ in 0..num_vectors {
        let vector: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
        let id = Id::new();

        let op_start = Instant::now();
        // Default is immediate_index=false (async - defers graph build)
        // T7.1: Use ExternalKey::NodeId for polymorphic ID mapping
        InsertVector::new(&embedding, ExternalKey::NodeId(id), vector)
            .run(&writer)
            .await?;
        async_metrics.record_insert(op_start.elapsed());
    }
    writer.flush().await?;

    // Calculate results
    let sync_p50 = sync_metrics.insert_percentile(0.50);
    let sync_p99 = sync_metrics.insert_percentile(0.99);
    let async_p50 = async_metrics.insert_percentile(0.50);
    let async_p99 = async_metrics.insert_percentile(0.99);

    let speedup_p50 = if async_p50.as_nanos() > 0 {
        sync_p50.as_nanos() as f64 / async_p50.as_nanos() as f64
    } else {
        f64::INFINITY
    };
    let speedup_p99 = if async_p99.as_nanos() > 0 {
        sync_p99.as_nanos() as f64 / async_p99.as_nanos() as f64
    } else {
        f64::INFINITY
    };

    Ok(SyncAsyncLatencyResult {
        num_vectors,
        sync_p50,
        sync_p99,
        async_p50,
        async_p99,
        speedup_p50,
        speedup_p99,
    })
}

// ============================================================================
// Backpressure Throughput Impact Benchmark (Task 8.2.9)
// ============================================================================

/// Result of backpressure throughput impact measurement.
///
/// Compares throughput across different channel buffer sizes to show
/// how backpressure affects performance when producers outpace consumers.
#[derive(Debug, Clone)]
pub struct BackpressureResult {
    /// Number of vectors inserted per test
    pub num_vectors: usize,
    /// Number of concurrent producers
    pub num_producers: usize,
    /// Results for each buffer size tested
    pub results: Vec<BackpressureSample>,
    /// Baseline throughput (largest buffer, minimal backpressure)
    pub baseline_throughput: f64,
}

/// Single sample from backpressure test at a specific buffer size.
#[derive(Debug, Clone)]
pub struct BackpressureSample {
    /// Channel buffer size for this sample
    pub buffer_size: usize,
    /// Total duration for all inserts
    pub duration: Duration,
    /// Insert throughput (ops/sec)
    pub throughput: f64,
    /// Average latency
    pub avg_latency: Duration,
    /// P50 latency
    pub p50_latency: Duration,
    /// P99 latency
    pub p99_latency: Duration,
    /// Number of errors (channel full rejections)
    pub errors: u64,
    /// Throughput as percentage of baseline
    pub throughput_pct: f64,
}

impl fmt::Display for BackpressureResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Backpressure Throughput Impact ===")?;
        writeln!(f, "Configuration:")?;
        writeln!(f, "  Vectors per test: {}", self.num_vectors)?;
        writeln!(f, "  Concurrent producers: {}", self.num_producers)?;
        writeln!(f, "  Baseline throughput: {:.1} ops/sec", self.baseline_throughput)?;
        writeln!(f)?;
        writeln!(f, "Buffer Size | Throughput | % Baseline | P50 Latency | P99 Latency | Errors")?;
        writeln!(f, "------------|------------|------------|-------------|-------------|-------")?;
        for sample in &self.results {
            writeln!(
                f,
                "{:>11} | {:>10.1} | {:>10.1}% | {:>11.2?} | {:>11.2?} | {:>6}",
                sample.buffer_size,
                sample.throughput,
                sample.throughput_pct,
                sample.p50_latency,
                sample.p99_latency,
                sample.errors,
            )?;
        }
        Ok(())
    }
}

/// Measure the impact of backpressure on insert throughput.
///
/// This benchmark tests how throughput degrades when channel buffers are small
/// and producers must wait for the consumer. It helps determine optimal buffer
/// sizes for production deployments.
///
/// # Arguments
/// * `storage` - Vector storage
/// * `embedding_code` - Embedding space code
/// * `num_vectors` - Total vectors to insert per buffer size test
/// * `dim` - Vector dimension
/// * `num_producers` - Number of concurrent producer tasks
/// * `buffer_sizes` - Buffer sizes to test (smallest to largest)
///
/// # Returns
/// `BackpressureResult` showing throughput at each buffer size.
///
/// # Example
///
/// ```ignore
/// let result = measure_backpressure_impact(
///     storage.clone(),
///     embedding_code,
///     1000,  // vectors per test
///     128,   // dimension
///     4,     // producers
///     &[10, 100, 1000, 10000],  // buffer sizes
/// ).await?;
/// println!("{}", result);
/// // Output:
/// // Buffer Size | Throughput | % Baseline | P50 Latency | P99 Latency | Errors
/// // ------------|------------|------------|-------------|-------------|-------
/// //          10 |      120.5 |      24.1% |       8.2ms |      42.1ms |      0
/// //         100 |      380.2 |      76.0% |       2.6ms |      12.3ms |      0
/// //        1000 |      485.1 |      97.0% |       2.1ms |       8.4ms |      0
/// //       10000 |      500.0 |     100.0% |       2.0ms |       8.1ms |      0
/// ```
pub async fn measure_backpressure_impact(
    storage: Arc<Storage>,
    embedding_code: EmbeddingCode,
    num_vectors: usize,
    dim: usize,
    num_producers: usize,
    buffer_sizes: &[usize],
) -> Result<BackpressureResult> {
    if buffer_sizes.is_empty() {
        anyhow::bail!("buffer_sizes must not be empty");
    }

    // Get embedding from registry
    let embedding = storage
        .cache()
        .get_by_code(embedding_code)
        .ok_or_else(|| anyhow::anyhow!("Embedding not found for code {}", embedding_code))?;

    let mut results = Vec::with_capacity(buffer_sizes.len());

    // Test each buffer size
    for &buffer_size in buffer_sizes {
        tracing::info!(buffer_size, num_vectors, num_producers, "Testing buffer size");

        // Create writer with specific buffer size
        let (writer, receiver) = create_writer(WriterConfig {
            channel_buffer_size: buffer_size,
        });
        let _consumer_handle = spawn_mutation_consumer_with_storage_autoreg(
            receiver,
            WriterConfig::default(),
            storage.clone(),
        );

        let metrics = Arc::new(ConcurrentMetrics::new());
        let vectors_per_producer = num_vectors / num_producers;
        let start = Instant::now();
        let stop = Arc::new(AtomicBool::new(false));

        // Spawn producer tasks
        let mut handles = Vec::with_capacity(num_producers);
        for producer_id in 0..num_producers {
            let writer = writer.clone();
            let embedding = embedding.clone();
            let metrics = Arc::clone(&metrics);
            let stop = Arc::clone(&stop);

            handles.push(tokio::spawn(async move {
                let mut rng = StdRng::seed_from_u64(producer_id as u64 + 3000);

                for _ in 0..vectors_per_producer {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }

                    let vector: Vec<f32> = (0..dim).map(|_| rng.random::<f32>()).collect();
                    let id = Id::new();

                    let op_start = Instant::now();
                    // T7.1: Use ExternalKey::NodeId for polymorphic ID mapping
                    let result = InsertVector::new(&embedding, ExternalKey::NodeId(id), vector)
                        .immediate()
                        .run(&writer)
                        .await;
                    let latency = op_start.elapsed();

                    match result {
                        Ok(()) => metrics.record_insert(latency),
                        Err(_) => metrics.record_error(),
                    }
                }
            }));
        }

        // Wait for all producers to complete
        for handle in handles {
            let _ = handle.await;
        }

        // Flush and get final metrics
        let _ = writer.flush().await;
        let duration = start.elapsed();

        let insert_count = metrics.insert_count();
        let throughput = if duration.as_secs_f64() > 0.0 {
            insert_count as f64 / duration.as_secs_f64()
        } else {
            0.0
        };

        results.push(BackpressureSample {
            buffer_size,
            duration,
            throughput,
            avg_latency: metrics.insert_avg(),
            p50_latency: metrics.insert_percentile(0.50),
            p99_latency: metrics.insert_percentile(0.99),
            errors: metrics.error_count(),
            throughput_pct: 0.0, // Will be calculated after we know baseline
        });

        // Small delay between tests to let resources settle
        tokio::time::sleep(Duration::from_millis(100)).await;
    }

    // Calculate baseline (largest buffer = minimal backpressure)
    let baseline_throughput = results.last().map(|r| r.throughput).unwrap_or(1.0);

    // Calculate throughput percentages
    for sample in &mut results {
        sample.throughput_pct = if baseline_throughput > 0.0 {
            (sample.throughput / baseline_throughput) * 100.0
        } else {
            0.0
        };
    }

    Ok(BackpressureResult {
        num_vectors,
        num_producers,
        results,
        baseline_throughput,
    })
}

/// Quick backpressure test with default buffer sizes.
///
/// Tests buffer sizes: [10, 50, 100, 500, 1000, 5000]
pub async fn measure_backpressure_impact_quick(
    storage: Arc<Storage>,
    embedding_code: EmbeddingCode,
    num_vectors: usize,
    dim: usize,
) -> Result<BackpressureResult> {
    measure_backpressure_impact(
        storage,
        embedding_code,
        num_vectors,
        dim,
        4, // 4 producers
        &[10, 50, 100, 500, 1000, 5000],
    )
    .await
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_to_bucket() {
        assert_eq!(latency_to_bucket(Duration::from_nanos(500)), 0); // <1us
        assert_eq!(latency_to_bucket(Duration::from_micros(1)), 1); // 1us
        assert_eq!(latency_to_bucket(Duration::from_micros(2)), 2); // 2us
        assert_eq!(latency_to_bucket(Duration::from_micros(4)), 3); // 4us
        // 1ms = 1000us, log2(1000) ≈ 9.97, leading_zeros gives bucket 10
        assert_eq!(latency_to_bucket(Duration::from_millis(1)), 10); // 1ms = 1000us
        assert_eq!(latency_to_bucket(Duration::from_secs(1)), NUM_BUCKETS - 1); // capped
    }

    #[test]
    fn test_metrics_record_and_read() {
        let metrics = ConcurrentMetrics::new();

        metrics.record_insert(Duration::from_millis(5));
        metrics.record_insert(Duration::from_millis(10));
        metrics.record_search(Duration::from_millis(2));
        metrics.record_error();

        assert_eq!(metrics.insert_count(), 2);
        assert_eq!(metrics.search_count(), 1);
        assert_eq!(metrics.error_count(), 1);
    }

    #[test]
    fn test_metrics_percentile() {
        let metrics = ConcurrentMetrics::new();

        // Add 100 samples with increasing latency
        for i in 1..=100 {
            metrics.record_search(Duration::from_millis(i));
        }

        let p50 = metrics.search_percentile(0.50);
        let p99 = metrics.search_percentile(0.99);

        // P50 should be around 50ms, P99 around 99ms
        // Due to bucket granularity, exact values may vary
        assert!(p50 >= Duration::from_millis(32)); // bucket lower bound
        assert!(p99 >= Duration::from_millis(64)); // bucket lower bound
    }

    #[test]
    fn test_metrics_reset() {
        let metrics = ConcurrentMetrics::new();

        metrics.record_insert(Duration::from_millis(5));
        metrics.record_search(Duration::from_millis(2));

        assert_eq!(metrics.insert_count(), 1);
        assert_eq!(metrics.search_count(), 1);

        metrics.reset();

        assert_eq!(metrics.insert_count(), 0);
        assert_eq!(metrics.search_count(), 0);
    }

    #[test]
    fn test_bench_config_presets() {
        let read_heavy = BenchConfig::read_heavy();
        assert_eq!(read_heavy.insert_producers, 1);
        assert_eq!(read_heavy.search_producers, 8);

        let write_heavy = BenchConfig::write_heavy();
        assert_eq!(write_heavy.insert_producers, 4);
        assert_eq!(write_heavy.search_producers, 1);

        let balanced = BenchConfig::balanced();
        assert_eq!(balanced.insert_producers, 4);
        assert_eq!(balanced.search_producers, 4);

        let stress = BenchConfig::stress();
        assert_eq!(stress.insert_producers, 8);
        assert_eq!(stress.search_producers, 8);
    }
}
