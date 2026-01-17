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
use rand::{Rng, SeedableRng};
use tokio::task::JoinHandle;

use crate::reader::Runnable as QueryRunnable;
use crate::vector::benchmark::dataset::LAION_EMBEDDING_DIM;
use crate::vector::schema::EmbeddingCode;
use crate::vector::writer::{create_writer, spawn_mutation_consumer_with_storage_autoreg, WriterConfig};
use crate::vector::reader::{create_search_reader_with_storage, spawn_query_consumers_with_storage_autoreg, ReaderConfig};
use crate::vector::{Distance, EmbeddingBuilder, InsertVector, MutationRunnable, SearchKNN, Storage};
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
/// - Searches go through SearchReader (MPMC) → query worker pool
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
    /// 2. Creates Writer (MPSC) and SearchReader (MPMC) channels
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
        let txn_db = storage.transaction_db()?;

        let embedding_name = format!("bench-{}", embedding_code);
        let embedding = registry
            .register(
                EmbeddingBuilder::new(&embedding_name, self.config.vector_dim as u32, self.config.distance)
                    .with_hnsw_m(self.config.hnsw_m as u16)
                    .with_hnsw_ef_construction(self.config.hnsw_ef_construction as u16),
                &txn_db,
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

        // Create SearchReader (MPMC) - query workers share this
        let (search_reader, reader_rx) =
            create_search_reader_with_storage(ReaderConfig::default(), storage.clone());
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

            producer_handles.push(tokio::spawn(async move {
                insert_producer_workload(
                    writer,
                    embedding,
                    metrics,
                    stop,
                    vectors_per_producer,
                    dim,
                    producer_id,
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
use crate::vector::reader::SearchReader;
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
    deadline: Instant,
) {
    // Use StdRng with seed based on producer_id for reproducibility and Send-safety
    let mut rng = StdRng::seed_from_u64(producer_id as u64 + 1000);

    for _i in 0..max_vectors {
        if stop.load(Ordering::Relaxed) || Instant::now() >= deadline {
            break;
        }

        // Generate random vector
        let vector: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let id = Id::new();

        let op_start = Instant::now();

        // Send insert through Writer channel (MPSC)
        let result = InsertVector::new(&embedding, id, vector)
            .immediate()
            .run(&writer)
            .await;

        let latency = op_start.elapsed();

        match result {
            Ok(()) => metrics.record_insert(latency),
            Err(_) => metrics.record_error(),
        }
    }
}

/// Search producer workload: sends searches to SearchReader until deadline.
async fn search_producer_workload(
    search_reader: SearchReader,
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
        let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

        let op_start = Instant::now();

        // Send search through SearchReader channel (MPMC)
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
