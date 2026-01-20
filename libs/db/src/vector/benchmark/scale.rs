//! Scale benchmark infrastructure for 10M-1B vector validation.
//!
//! This module provides streaming benchmarks that can handle datasets too large
//! to fit in memory. Key features:
//!
//! - **Streaming generation**: Vectors generated on-the-fly, not pre-loaded
//! - **Progress reporting**: Real-time throughput and ETA updates
//! - **Memory profiling**: Track peak RSS and cache sizes
//! - **Checkpoint support**: Resume interrupted benchmarks
//!
//! ## Usage
//!
//! ```ignore
//! use motlie_db::vector::benchmark::scale::{ScaleConfig, ScaleBenchmark};
//!
//! let config = ScaleConfig::new(10_000_000, 512)
//!     .with_batch_size(10_000)
//!     .with_progress_interval(Duration::from_secs(10));
//!
//! let result = ScaleBenchmark::run(&storage, &embedding, config)?;
//! println!("{}", result);
//! ```

use std::fmt;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;

use crate::vector::{Distance, Embedding, Processor, Storage};
use crate::Id;

// ============================================================================
// ScaleConfig
// ============================================================================

/// Configuration for scale benchmarks.
#[derive(Debug, Clone)]
pub struct ScaleConfig {
    /// Total number of vectors to insert.
    pub num_vectors: usize,
    /// Vector dimension.
    pub dim: usize,
    /// Batch size for inserts (vectors per transaction).
    pub batch_size: usize,
    /// RNG seed for reproducible vector generation.
    pub seed: u64,
    /// Progress reporting interval.
    pub progress_interval: Duration,
    /// Number of search queries to run after insert.
    pub num_queries: usize,
    /// ef_search parameter for queries.
    pub ef_search: usize,
    /// k for top-k search.
    pub k: usize,
    /// Whether to build HNSW index during insert (vs async).
    pub immediate_index: bool,
    /// Distance metric.
    pub distance: Distance,
    /// HNSW M parameter.
    pub hnsw_m: usize,
    /// HNSW ef_construction parameter.
    pub hnsw_ef_construction: usize,
}

impl ScaleConfig {
    /// Create a new scale config with required parameters.
    pub fn new(num_vectors: usize, dim: usize) -> Self {
        Self {
            num_vectors,
            dim,
            batch_size: 1000,
            seed: 42,
            progress_interval: Duration::from_secs(10),
            num_queries: 1000,
            ef_search: 100,
            k: 10,
            immediate_index: true,
            distance: Distance::Cosine,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
        }
    }

    /// Set batch size for inserts.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size.max(1);
        self
    }

    /// Set RNG seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set progress reporting interval.
    pub fn with_progress_interval(mut self, interval: Duration) -> Self {
        self.progress_interval = interval;
        self
    }

    /// Set number of search queries.
    pub fn with_num_queries(mut self, num_queries: usize) -> Self {
        self.num_queries = num_queries;
        self
    }

    /// Set ef_search parameter.
    pub fn with_ef_search(mut self, ef_search: usize) -> Self {
        self.ef_search = ef_search;
        self
    }

    /// Set k for top-k search.
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set immediate vs async indexing.
    pub fn with_immediate_index(mut self, immediate: bool) -> Self {
        self.immediate_index = immediate;
        self
    }

    /// Set HNSW M parameter.
    pub fn with_hnsw_m(mut self, m: usize) -> Self {
        self.hnsw_m = m;
        self
    }

    /// Set HNSW ef_construction parameter.
    pub fn with_hnsw_ef_construction(mut self, ef: usize) -> Self {
        self.hnsw_ef_construction = ef;
        self
    }

    /// Estimated memory usage in bytes.
    pub fn estimated_memory_bytes(&self) -> u64 {
        // Per-vector estimate (from PHASE8.md):
        // - Vector data (f16): dim * 2 bytes
        // - Binary code: dim / 8 bytes
        // - HNSW edges: ~128 bytes avg
        // - VecMeta: 16 bytes
        // - IdForward: 24 bytes
        // - IdReverse: 24 bytes
        let per_vector = (self.dim * 2) + (self.dim / 8) + 128 + 16 + 24 + 24;
        (self.num_vectors as u64) * (per_vector as u64)
    }

    /// Human-readable estimated memory.
    pub fn estimated_memory_human(&self) -> String {
        let bytes = self.estimated_memory_bytes();
        if bytes >= 1_000_000_000_000 {
            format!("{:.1} TB", bytes as f64 / 1_000_000_000_000.0)
        } else if bytes >= 1_000_000_000 {
            format!("{:.1} GB", bytes as f64 / 1_000_000_000.0)
        } else if bytes >= 1_000_000 {
            format!("{:.1} MB", bytes as f64 / 1_000_000.0)
        } else {
            format!("{:.1} KB", bytes as f64 / 1_000.0)
        }
    }
}

impl Default for ScaleConfig {
    fn default() -> Self {
        Self::new(1_000_000, 512)
    }
}

// ============================================================================
// ScaleProgress
// ============================================================================

/// Real-time progress tracking for scale benchmarks.
#[derive(Debug)]
pub struct ScaleProgress {
    /// Total vectors to insert.
    pub total: u64,
    /// Vectors inserted so far.
    pub inserted: AtomicU64,
    /// Errors encountered.
    pub errors: AtomicU64,
    /// Start time.
    pub start: Instant,
    /// Whether benchmark is complete.
    pub done: AtomicBool,
}

impl ScaleProgress {
    /// Create new progress tracker.
    pub fn new(total: usize) -> Self {
        Self {
            total: total as u64,
            inserted: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            start: Instant::now(),
            done: AtomicBool::new(false),
        }
    }

    /// Record inserted vectors.
    pub fn record_insert(&self, count: usize) {
        self.inserted.fetch_add(count as u64, Ordering::Relaxed);
    }

    /// Record an error.
    pub fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    /// Mark as done.
    pub fn mark_done(&self) {
        self.done.store(true, Ordering::Relaxed);
    }

    /// Get current progress percentage.
    pub fn percent(&self) -> f64 {
        let inserted = self.inserted.load(Ordering::Relaxed);
        if self.total == 0 {
            return 100.0;
        }
        (inserted as f64 / self.total as f64) * 100.0
    }

    /// Get current throughput (vectors/second).
    pub fn throughput(&self) -> f64 {
        let inserted = self.inserted.load(Ordering::Relaxed);
        let elapsed = self.start.elapsed().as_secs_f64();
        if elapsed < 0.001 {
            return 0.0;
        }
        inserted as f64 / elapsed
    }

    /// Get estimated time remaining.
    pub fn eta(&self) -> Duration {
        let inserted = self.inserted.load(Ordering::Relaxed);
        let remaining = self.total.saturating_sub(inserted);
        let throughput = self.throughput();
        if throughput < 1.0 {
            return Duration::from_secs(u64::MAX);
        }
        Duration::from_secs_f64(remaining as f64 / throughput)
    }

    /// Format progress as human-readable string.
    pub fn format(&self) -> String {
        let inserted = self.inserted.load(Ordering::Relaxed);
        let errors = self.errors.load(Ordering::Relaxed);
        let elapsed = self.start.elapsed();
        let eta = self.eta();

        let eta_str = if eta.as_secs() > 86400 * 365 {
            "N/A".to_string()
        } else if eta.as_secs() > 3600 {
            format!("{}h{}m", eta.as_secs() / 3600, (eta.as_secs() % 3600) / 60)
        } else if eta.as_secs() > 60 {
            format!("{}m{}s", eta.as_secs() / 60, eta.as_secs() % 60)
        } else {
            format!("{}s", eta.as_secs())
        };

        format!(
            "[{:>6.2}%] {}/{} vectors | {:.1} vec/s | Errors: {} | Elapsed: {}s | ETA: {}",
            self.percent(),
            format_number(inserted),
            format_number(self.total),
            self.throughput(),
            errors,
            elapsed.as_secs(),
            eta_str
        )
    }
}

/// Format large numbers with commas.
fn format_number(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

// ============================================================================
// ScaleResult
// ============================================================================

/// Results from a scale benchmark run.
#[derive(Debug, Clone)]
pub struct ScaleResult {
    /// Configuration used.
    pub config: ScaleConfig,
    /// Total vectors inserted.
    pub vectors_inserted: u64,
    /// Insert errors.
    pub insert_errors: u64,
    /// Total insert duration.
    pub insert_duration: Duration,
    /// Insert throughput (vectors/second).
    pub insert_throughput: f64,
    /// Search queries executed.
    pub queries_executed: u64,
    /// Total search duration.
    pub search_duration: Duration,
    /// Search throughput (queries/second).
    pub search_qps: f64,
    /// Search latency P50.
    pub search_p50: Duration,
    /// Search latency P99.
    pub search_p99: Duration,
    /// Peak memory usage (RSS) in bytes.
    pub peak_memory_bytes: u64,
    /// Navigation cache size in bytes.
    pub nav_cache_bytes: u64,
}

impl ScaleResult {
    /// Format peak memory as human-readable string.
    pub fn peak_memory_human(&self) -> String {
        format_bytes(self.peak_memory_bytes)
    }

    /// Format nav cache size as human-readable string.
    pub fn nav_cache_human(&self) -> String {
        format_bytes(self.nav_cache_bytes)
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes >= 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.2} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} B", bytes)
    }
}

impl fmt::Display for ScaleResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Scale Benchmark Results ===")?;
        writeln!(f, "Configuration:")?;
        writeln!(f, "  Vectors: {}", format_number(self.config.num_vectors as u64))?;
        writeln!(f, "  Dimension: {}D", self.config.dim)?;
        writeln!(f, "  HNSW M={}, ef_construction={}", self.config.hnsw_m, self.config.hnsw_ef_construction)?;
        writeln!(f, "  Batch size: {}", self.config.batch_size)?;
        writeln!(f)?;
        writeln!(f, "Insert Performance:")?;
        writeln!(f, "  Vectors inserted: {}", format_number(self.vectors_inserted))?;
        writeln!(f, "  Errors: {}", self.insert_errors)?;
        writeln!(f, "  Duration: {:.1}s", self.insert_duration.as_secs_f64())?;
        writeln!(f, "  Throughput: {:.1} vec/s", self.insert_throughput)?;
        writeln!(f)?;
        writeln!(f, "Search Performance:")?;
        writeln!(f, "  Queries: {}", self.queries_executed)?;
        writeln!(f, "  Duration: {:.1}s", self.search_duration.as_secs_f64())?;
        writeln!(f, "  QPS: {:.1}", self.search_qps)?;
        writeln!(f, "  Latency P50: {:?}", self.search_p50)?;
        writeln!(f, "  Latency P99: {:?}", self.search_p99)?;
        writeln!(f)?;
        writeln!(f, "Memory:")?;
        writeln!(f, "  Peak RSS: {}", self.peak_memory_human())?;
        writeln!(f, "  Nav cache: {}", self.nav_cache_human())?;
        Ok(())
    }
}

// ============================================================================
// StreamingVectorGenerator
// ============================================================================

/// Streaming generator for reproducible random vectors.
///
/// Generates vectors on-the-fly without holding them all in memory.
pub struct StreamingVectorGenerator {
    rng: ChaCha8Rng,
    dim: usize,
    generated: usize,
    total: usize,
}

impl StreamingVectorGenerator {
    /// Create a new generator.
    pub fn new(dim: usize, total: usize, seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
            dim,
            generated: 0,
            total,
        }
    }

    /// Generate next batch of normalized vectors.
    ///
    /// Returns None when all vectors have been generated.
    pub fn next_batch(&mut self, batch_size: usize) -> Option<Vec<Vec<f32>>> {
        if self.generated >= self.total {
            return None;
        }

        let remaining = self.total - self.generated;
        let count = batch_size.min(remaining);
        let mut batch = Vec::with_capacity(count);

        for _ in 0..count {
            let mut vector: Vec<f32> = (0..self.dim)
                .map(|_| {
                    let r: f32 = self.rng.gen();
                    r * 2.0 - 1.0
                })
                .collect();

            // Normalize to unit length (for Cosine distance)
            let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                for v in &mut vector {
                    *v /= norm;
                }
            }

            batch.push(vector);
            self.generated += 1;
        }

        Some(batch)
    }

    /// Generate a single query vector (uses separate seed offset).
    pub fn generate_query(&mut self) -> Vec<f32> {
        let mut vector: Vec<f32> = (0..self.dim)
            .map(|_| {
                let r: f32 = self.rng.gen();
                r * 2.0 - 1.0
            })
            .collect();
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for v in &mut vector {
                *v /= norm;
            }
        }
        vector
    }

    /// Vectors generated so far.
    pub fn generated(&self) -> usize {
        self.generated
    }

    /// Total vectors to generate.
    pub fn total(&self) -> usize {
        self.total
    }
}

// ============================================================================
// Memory Profiling
// ============================================================================

/// Get current process RSS (Resident Set Size) in bytes.
///
/// Returns 0 if unable to read memory info.
pub fn get_rss_bytes() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/self/statm") {
            let parts: Vec<&str> = contents.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(pages) = parts[1].parse::<u64>() {
                    // Page size is typically 4KB
                    return pages * 4096;
                }
            }
        }
        0
    }

    #[cfg(target_os = "macos")]
    {
        // macOS: use mach task_info (simplified - returns 0 for now)
        // Full implementation would use mach_task_self() and task_info()
        0
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        0
    }
}

// ============================================================================
// ScaleBenchmark
// ============================================================================

/// Async metrics callback for tracking pending queue and worker progress.
pub struct AsyncMetrics {
    /// Get current pending queue size
    pub pending_queue_size: Box<dyn Fn() -> usize + Send + Sync>,
    /// Get items processed by async workers
    pub items_processed: Box<dyn Fn() -> u64 + Send + Sync>,
}

/// Scale benchmark runner.
pub struct ScaleBenchmark;

impl ScaleBenchmark {
    /// Run a scale benchmark with the given configuration.
    ///
    /// This method:
    /// 1. Streams vectors into the index in batches
    /// 2. Reports progress at configured intervals
    /// 3. Runs search queries after insert completes
    /// 4. Collects memory and latency metrics
    pub fn run(
        storage: &Arc<Storage>,
        embedding: &Embedding,
        config: ScaleConfig,
    ) -> Result<ScaleResult> {
        Self::run_with_async_metrics(storage, embedding, config, None)
    }

    /// Run a scale benchmark with optional async metrics tracking.
    ///
    /// When `async_metrics` is provided, the progress output includes:
    /// - Pending queue size (backpressure indicator)
    /// - Async worker processing rate
    pub fn run_with_async_metrics(
        storage: &Arc<Storage>,
        embedding: &Embedding,
        config: ScaleConfig,
        async_metrics: Option<AsyncMetrics>,
    ) -> Result<ScaleResult> {
        let progress = Arc::new(ScaleProgress::new(config.num_vectors));

        println!("=== Scale Benchmark: {} vectors ===", format_number(config.num_vectors as u64));
        println!("Estimated storage: {}", config.estimated_memory_human());
        if !config.immediate_index {
            println!("Mode: ASYNC (deferred HNSW construction)");
        }
        println!();

        // Create processor (handles all storage setup, caches, etc.)
        let registry = storage.cache().clone();
        let processor = Processor::new(storage.clone(), registry);

        // Phase 1: Insert using Processor batch API
        let insert_start = Instant::now();
        let mut generator = StreamingVectorGenerator::new(config.dim, config.num_vectors, config.seed);
        let mut last_progress = Instant::now();
        let mut peak_rss: u64 = 0;
        let mut peak_pending: usize = 0;
        let mut last_items_processed: u64 = 0;

        while let Some(batch) = generator.next_batch(config.batch_size) {
            // Convert to (Id, Vec<f32>) format for processor
            let vectors: Vec<(Id, Vec<f32>)> = batch
                .into_iter()
                .map(|v| (Id::new(), v))
                .collect();

            let batch_len = vectors.len();

            match processor.insert_batch(embedding, &vectors, config.immediate_index) {
                Ok(_vec_ids) => {
                    progress.record_insert(batch_len);
                }
                Err(e) => {
                    tracing::warn!("Batch insert failed: {}", e);
                    for _ in 0..batch_len {
                        progress.record_error();
                    }
                }
            }

            // Progress reporting
            if last_progress.elapsed() >= config.progress_interval {
                let rss = get_rss_bytes();
                peak_rss = peak_rss.max(rss);

                // Format progress with async metrics if available
                if let Some(ref metrics) = async_metrics {
                    let pending = (metrics.pending_queue_size)();
                    let items = (metrics.items_processed)();
                    let worker_rate = (items - last_items_processed) as f64
                        / config.progress_interval.as_secs_f64();
                    last_items_processed = items;
                    peak_pending = peak_pending.max(pending);

                    print!(
                        "\r{} | Pending: {} | Workers: {:.1}/s",
                        progress.format(),
                        format_number(pending as u64),
                        worker_rate
                    );
                } else {
                    print!("\r{}", progress.format());
                }
                std::io::stdout().flush().ok();
                last_progress = Instant::now();
            }
        }

        let insert_duration = insert_start.elapsed();
        progress.mark_done();

        // Final progress line
        if let Some(ref metrics) = async_metrics {
            let pending = (metrics.pending_queue_size)();
            let items = (metrics.items_processed)();
            println!(
                "\r{} | Pending: {} | Workers processed: {}",
                progress.format(),
                format_number(pending as u64),
                format_number(items)
            );
            println!("Peak pending queue: {}", format_number(peak_pending as u64));
        } else {
            println!("\r{}", progress.format());
        }
        println!();

        let vectors_inserted = progress.inserted.load(Ordering::Relaxed);
        let insert_errors = progress.errors.load(Ordering::Relaxed);
        let insert_throughput = if insert_duration.as_secs_f64() > 0.0 {
            vectors_inserted as f64 / insert_duration.as_secs_f64()
        } else {
            0.0
        };

        // Phase 2: Search using Processor's index
        println!("Running {} search queries...", config.num_queries);
        let search_start = Instant::now();
        let mut latencies: Vec<Duration> = Vec::with_capacity(config.num_queries);

        // Use different seed for queries
        let mut query_gen = StreamingVectorGenerator::new(config.dim, config.num_queries, config.seed + 1_000_000);

        // Get the HNSW index from processor
        let index = processor
            .get_or_create_index(embedding.code())
            .ok_or_else(|| anyhow::anyhow!("Failed to get HNSW index"))?;

        for _ in 0..config.num_queries {
            let query = query_gen.generate_query();
            let query_start = Instant::now();

            match index.search(storage.as_ref(), &query, config.ef_search, config.k) {
                Ok(_results) => {
                    latencies.push(query_start.elapsed());
                }
                Err(_) => {
                    // Record as max latency on error
                    latencies.push(Duration::from_secs(10));
                }
            }
        }

        let search_duration = search_start.elapsed();
        let queries_executed = latencies.len() as u64;
        let search_qps = if search_duration.as_secs_f64() > 0.0 {
            queries_executed as f64 / search_duration.as_secs_f64()
        } else {
            0.0
        };

        // Calculate percentiles
        latencies.sort();
        let search_p50 = latencies.get(latencies.len() / 2).copied().unwrap_or_default();
        let search_p99 = latencies.get(latencies.len() * 99 / 100).copied().unwrap_or_default();

        // Final memory snapshot
        let final_rss = get_rss_bytes();
        peak_rss = peak_rss.max(final_rss);

        let (nav_entries, nav_bytes) = processor.nav_cache().edge_cache_stats();
        println!("Nav cache: {} entries, {} bytes", nav_entries, nav_bytes);

        Ok(ScaleResult {
            config,
            vectors_inserted,
            insert_errors,
            insert_duration,
            insert_throughput,
            queries_executed,
            search_duration,
            search_qps,
            search_p50,
            search_p99,
            peak_memory_bytes: peak_rss,
            nav_cache_bytes: nav_bytes as u64,
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_generator() {
        let mut gen = StreamingVectorGenerator::new(128, 100, 42);

        let mut total = 0;
        while let Some(batch) = gen.next_batch(30) {
            total += batch.len();
            for v in &batch {
                assert_eq!(v.len(), 128);
                // Check normalized
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!((norm - 1.0).abs() < 1e-5);
            }
        }

        assert_eq!(total, 100);
    }

    #[test]
    fn test_streaming_generator_deterministic() {
        let mut gen1 = StreamingVectorGenerator::new(64, 50, 42);
        let mut gen2 = StreamingVectorGenerator::new(64, 50, 42);

        let batch1 = gen1.next_batch(50).unwrap();
        let batch2 = gen2.next_batch(50).unwrap();

        assert_eq!(batch1, batch2);
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(100), "100");
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(1000000), "1,000,000");
        assert_eq!(format_number(10000000), "10,000,000");
    }

    #[test]
    fn test_config_memory_estimate() {
        let config = ScaleConfig::new(1_000_000, 512);
        let bytes = config.estimated_memory_bytes();
        // Should be roughly 1.8GB for 1M 512D vectors
        assert!(bytes > 1_000_000_000); // > 1GB
        assert!(bytes < 3_000_000_000); // < 3GB
    }

    #[test]
    fn test_progress_format() {
        let progress = ScaleProgress::new(1000);
        progress.record_insert(500);
        let fmt = progress.format();
        assert!(fmt.contains("50.00%"));
        assert!(fmt.contains("500"));
    }
}
