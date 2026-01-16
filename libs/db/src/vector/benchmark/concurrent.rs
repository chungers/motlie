//! Concurrent benchmark infrastructure for vector search.
//!
//! This module provides tools for measuring performance under concurrent load:
//! - Lock-free metrics collection with atomic counters
//! - Latency histograms with log2 buckets
//! - Concurrent benchmark runner with configurable thread counts
//!
//! # Example
//!
//! ```ignore
//! use motlie_db::vector::benchmark::concurrent::{ConcurrentMetrics, ConcurrentBenchmark, BenchConfig};
//!
//! let metrics = Arc::new(ConcurrentMetrics::new());
//! let config = BenchConfig {
//!     writer_threads: 4,
//!     reader_threads: 4,
//!     duration: Duration::from_secs(30),
//!     ..Default::default()
//! };
//!
//! let bench = ConcurrentBenchmark::new(config, metrics.clone());
//! let result = bench.run(storage, registry)?;
//! println!("{}", result);
//! ```

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use rand::Rng;

use crate::rocksdb::ColumnFamily;
use crate::vector::cache::NavigationCache;
use crate::vector::hnsw;
use crate::vector::schema::{EmbeddingCode, VectorCfKey, VectorCfValue, Vectors};
use crate::vector::{Distance, Storage};

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
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Number of writer threads (inserting vectors).
    pub writer_threads: usize,
    /// Number of reader threads (searching).
    pub reader_threads: usize,
    /// Total benchmark duration.
    pub duration: Duration,
    /// Vectors to insert per writer thread.
    pub vectors_per_writer: usize,
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
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            writer_threads: 4,
            reader_threads: 4,
            duration: Duration::from_secs(30),
            vectors_per_writer: 1000,
            vector_dim: 128,
            k: 10,
            ef_search: 100,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
        }
    }
}

impl BenchConfig {
    /// Create a read-heavy configuration (1 writer, 8 readers).
    pub fn read_heavy() -> Self {
        Self {
            writer_threads: 1,
            reader_threads: 8,
            ..Default::default()
        }
    }

    /// Create a write-heavy configuration (8 writers, 1 reader).
    pub fn write_heavy() -> Self {
        Self {
            writer_threads: 8,
            reader_threads: 1,
            ..Default::default()
        }
    }

    /// Create a balanced configuration (4 writers, 4 readers).
    pub fn balanced() -> Self {
        Self {
            writer_threads: 4,
            reader_threads: 4,
            ..Default::default()
        }
    }

    /// Create a stress configuration (16 writers, 16 readers).
    pub fn stress() -> Self {
        Self {
            writer_threads: 16,
            reader_threads: 16,
            duration: Duration::from_secs(60),
            ..Default::default()
        }
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
}

impl fmt::Display for BenchResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Concurrent Benchmark Results ===")?;
        writeln!(f, "Configuration:")?;
        writeln!(f, "  Writers: {}", self.config.writer_threads)?;
        writeln!(f, "  Readers: {}", self.config.reader_threads)?;
        writeln!(f, "  Duration: {:?}", self.duration)?;
        writeln!(f, "Throughput:")?;
        writeln!(f, "  Inserts: {:.1} ops/sec", self.insert_throughput)?;
        writeln!(f, "  Searches: {:.1} ops/sec", self.search_throughput)?;
        writeln!(f, "{}", self.metrics)?;
        Ok(())
    }
}

// ============================================================================
// ConcurrentBenchmark - Benchmark runner
// ============================================================================

/// Concurrent benchmark runner.
///
/// Spawns writer and reader threads to measure performance under concurrent load.
pub struct ConcurrentBenchmark {
    config: BenchConfig,
    metrics: Arc<ConcurrentMetrics>,
}

impl ConcurrentBenchmark {
    /// Create a new benchmark with the given configuration.
    pub fn new(config: BenchConfig, metrics: Arc<ConcurrentMetrics>) -> Self {
        Self { config, metrics }
    }

    /// Run the benchmark against the given storage.
    ///
    /// This method:
    /// 1. Creates an HNSW index with the configured parameters
    /// 2. Spawns writer threads that insert random vectors
    /// 3. Spawns reader threads that search with random queries
    /// 4. Waits for all threads to complete or timeout
    /// 5. Returns aggregated metrics
    pub fn run(
        &self,
        storage: Arc<Storage>,
        embedding: EmbeddingCode,
    ) -> Result<BenchResult> {
        let start = Instant::now();
        let deadline = start + self.config.duration;

        // Reset metrics before run
        self.metrics.reset();

        // Create HNSW index
        let nav_cache = Arc::new(NavigationCache::new());
        let hnsw_config = hnsw::Config {
            dim: self.config.vector_dim,
            m: self.config.hnsw_m,
            m_max: self.config.hnsw_m * 2,
            m_max_0: self.config.hnsw_m * 2,
            ef_construction: self.config.hnsw_ef_construction,
            ..Default::default()
        };
        let index = Arc::new(hnsw::Index::new(
            embedding,
            Distance::L2,
            hnsw_config,
            nav_cache,
        ));

        let mut handles = Vec::new();

        // Spawn writer threads
        for thread_id in 0..self.config.writer_threads {
            let storage = Arc::clone(&storage);
            let index = Arc::clone(&index);
            let metrics = Arc::clone(&self.metrics);
            let vectors_per_writer = self.config.vectors_per_writer;
            let dim = self.config.vector_dim;

            handles.push(thread::spawn(move || {
                writer_workload(
                    storage,
                    index,
                    embedding,
                    metrics,
                    vectors_per_writer,
                    dim,
                    thread_id,
                    deadline,
                );
            }));
        }

        // Spawn reader threads
        for _thread_id in 0..self.config.reader_threads {
            let storage = Arc::clone(&storage);
            let index = Arc::clone(&index);
            let metrics = Arc::clone(&self.metrics);
            let dim = self.config.vector_dim;
            let k = self.config.k;
            let ef = self.config.ef_search;

            handles.push(thread::spawn(move || {
                reader_workload(storage, index, metrics, dim, k, ef, deadline);
            }));
        }

        // Wait for all threads
        for handle in handles {
            handle.join().expect("Thread panicked");
        }

        let duration = start.elapsed();
        let summary = self.metrics.summary();

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

        Ok(BenchResult {
            config: self.config.clone(),
            duration,
            metrics: summary,
            insert_throughput,
            search_throughput,
        })
    }
}

// ============================================================================
// Workload functions
// ============================================================================

/// Writer workload: insert random vectors until deadline or count reached.
fn writer_workload(
    storage: Arc<Storage>,
    index: Arc<hnsw::Index>,
    embedding: EmbeddingCode,
    metrics: Arc<ConcurrentMetrics>,
    max_vectors: usize,
    dim: usize,
    thread_id: usize,
    deadline: Instant,
) {
    let mut rng = rand::thread_rng();
    let txn_db = match storage.transaction_db() {
        Ok(db) => db,
        Err(_) => {
            metrics.record_error();
            return;
        }
    };

    for i in 0..max_vectors {
        if Instant::now() >= deadline {
            break;
        }

        // Generate random vector
        let vector: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

        // Use thread_id and index to generate unique vec_id
        let vec_id = ((thread_id as u32) << 24) | (i as u32);

        let op_start = Instant::now();

        // Store vector and insert into HNSW
        let result = (|| -> Result<()> {
            let txn = txn_db.transaction();

            // Store vector in Vectors CF
            let vectors_cf = txn_db
                .cf_handle(Vectors::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;

            let key = VectorCfKey(embedding, vec_id);
            let value = VectorCfValue(vector.clone());
            txn.put_cf(
                &vectors_cf,
                Vectors::key_to_bytes(&key),
                Vectors::value_to_bytes(&value),
            )?;

            // Insert into HNSW
            let cache_update = hnsw::insert(&index, &txn, &txn_db, &storage, vec_id, &vector)?;
            txn.commit()?;
            cache_update.apply(index.nav_cache());

            Ok(())
        })();

        let latency = op_start.elapsed();

        match result {
            Ok(()) => metrics.record_insert(latency),
            Err(_) => metrics.record_error(),
        }
    }
}

/// Reader workload: search with random queries until deadline.
fn reader_workload(
    storage: Arc<Storage>,
    index: Arc<hnsw::Index>,
    metrics: Arc<ConcurrentMetrics>,
    dim: usize,
    k: usize,
    ef: usize,
    deadline: Instant,
) {
    let mut rng = rand::thread_rng();

    while Instant::now() < deadline {
        // Generate random query
        let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();

        let op_start = Instant::now();

        // Perform search
        let result = index.search(&storage, &query, k, ef);

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
        assert_eq!(read_heavy.writer_threads, 1);
        assert_eq!(read_heavy.reader_threads, 8);

        let write_heavy = BenchConfig::write_heavy();
        assert_eq!(write_heavy.writer_threads, 8);
        assert_eq!(write_heavy.reader_threads, 1);

        let balanced = BenchConfig::balanced();
        assert_eq!(balanced.writer_threads, 4);
        assert_eq!(balanced.reader_threads, 4);

        let stress = BenchConfig::stress();
        assert_eq!(stress.writer_threads, 16);
        assert_eq!(stress.reader_threads, 16);
    }
}
