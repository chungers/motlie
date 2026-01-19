//! Garbage Collection worker for deleted vector cleanup.
//!
//! This module provides a background worker that cleans up soft-deleted vectors
//! by removing edges pointing to them and reclaiming storage.
//!
//! ## Delete Cleanup Pipeline
//!
//! ```text
//! Delete(id) [sync]          GarbageCollector [async background]
//!   │                             │
//!   ├─ Remove IdForward          ├─ Scan VecMeta for is_deleted() == true
//!   ├─ Remove IdReverse          ├─ For each deleted vec_id:
//!   ├─ Set VecMeta → Deleted     │   ├─ Scan Edges CF for references
//!   └─ Keep edges/data           │   ├─ Remove vec_id from neighbor bitmaps
//!                                │   ├─ Delete vector data + binary codes
//!                                │   ├─ Return VecId to free list
//!                                │   └─ Delete VecMeta entry
//!                                └─ Repeat on configured interval
//! ```
//!
//! ## Design Notes
//!
//! **VecMeta Scan vs Dedicated Tracking CF:**
//!
//! We use O(n) VecMeta scan rather than a dedicated DeletedVectors CF because:
//! 1. VecMeta is the source of truth for lifecycle state
//! 2. Avoids schema complexity and migration
//! 3. Background GC can tolerate O(n) scan cost
//! 4. Prefix iteration by embedding makes scans efficient
//!
//! See PHASE8.md Task 8.1.1 for full discussion.
//!
//! ## Usage
//!
//! ```ignore
//! let config = GcConfig::default()
//!     .with_interval(Duration::from_secs(60))
//!     .with_batch_size(100);
//!
//! let gc = GarbageCollector::start(storage, registry, config);
//!
//! // ... application runs ...
//!
//! gc.shutdown(); // Graceful shutdown
//! ```

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use roaring::RoaringBitmap;
use tracing::{debug, error, info, warn};

use crate::rocksdb::{ColumnFamily, HotColumnFamilyRecord};
use crate::vector::merge::EdgeOp;
use crate::vector::registry::EmbeddingRegistry;
use crate::vector::schema::{
    BinaryCodeCfKey, BinaryCodes, EmbeddingCode, Edges, EdgeCfKey, IdAlloc, IdAllocCfKey,
    IdAllocCfValue, IdAllocField, VecId, VecMeta, VecMetaCfKey, VectorCfKey, Vectors,
};
use crate::vector::Storage;

// ============================================================================
// GcConfig
// ============================================================================

/// Configuration for the garbage collection worker.
///
/// ## Tuning Guidance
///
/// - **`interval`**: How often the GC worker runs a cleanup cycle. Lower values
///   provide faster cleanup but increase background I/O. Default: 60s
///
/// - **`batch_size`**: Maximum deleted vectors to process per cycle. Limits
///   transaction size and memory usage. Default: 100
///
/// - **`process_on_startup`**: Run a cleanup cycle immediately on startup to
///   clean up any vectors deleted during previous run. Default: true
///
/// - **`edge_scan_limit`**: Maximum edges to scan when looking for references
///   to a deleted vector. Prevents runaway scans. Default: 10000
#[derive(Debug, Clone)]
pub struct GcConfig {
    /// Interval between GC cycles.
    ///
    /// The worker sleeps this long between cleanup cycles.
    /// Default: 60 seconds
    pub interval: Duration,

    /// Maximum deleted vectors to process per cycle.
    ///
    /// Limits memory usage and transaction size.
    /// Default: 100
    pub batch_size: usize,

    /// Whether to run a cleanup cycle immediately on startup.
    ///
    /// Helps recover from crashes where deletes were not fully cleaned up.
    /// Default: true
    pub process_on_startup: bool,

    /// Maximum edges to scan when searching for references.
    ///
    /// For each deleted vector, we scan the Edges CF to find neighbors that
    /// reference it. This limit prevents runaway scans.
    /// Default: 10000
    pub edge_scan_limit: usize,

    /// Whether to also free VecIds after edge cleanup.
    ///
    /// When true, VecIds are returned to the free list after all edge
    /// references are removed. When false, IDs are never reused (safer
    /// but wastes ID space).
    /// Default: false (conservative - defer ID recycling)
    pub enable_id_recycling: bool,
}

impl Default for GcConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(60),
            batch_size: 100,
            process_on_startup: true,
            edge_scan_limit: 10000,
            enable_id_recycling: false, // Conservative default
        }
    }
}

impl GcConfig {
    /// Create a new config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set interval between GC cycles.
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.interval = interval;
        self
    }

    /// Set maximum deleted vectors per cycle.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set whether to process on startup.
    pub fn with_process_on_startup(mut self, process: bool) -> Self {
        self.process_on_startup = process;
        self
    }

    /// Set edge scan limit.
    pub fn with_edge_scan_limit(mut self, limit: usize) -> Self {
        self.edge_scan_limit = limit;
        self
    }

    /// Enable VecId recycling after edge cleanup.
    pub fn with_id_recycling(mut self, enable: bool) -> Self {
        self.enable_id_recycling = enable;
        self
    }
}

// ============================================================================
// GcMetrics
// ============================================================================

/// Metrics for the garbage collector.
#[derive(Debug, Default)]
pub struct GcMetrics {
    /// Total vectors cleaned up
    pub vectors_cleaned: AtomicU64,
    /// Total edges pruned
    pub edges_pruned: AtomicU64,
    /// Total GC cycles completed
    pub cycles_completed: AtomicU64,
    /// Total VecIds recycled (if enabled)
    pub ids_recycled: AtomicU64,
    /// Total vectors skipped due to incomplete edge scan (ID recycling deferred)
    pub recycling_skipped_incomplete: AtomicU64,
}
// ADDRESSED (Claude, 2026-01-19): Removed `bytes_reclaimed` - tracking byte sizes would
// require reading values before delete which adds I/O overhead. Added `recycling_skipped_incomplete`
// to track when ID recycling is deferred due to edge_scan_limit being hit.

// ============================================================================
// Internal Result Types
// ============================================================================

/// Result of edge pruning operation.
struct PruneResult {
    /// Number of edges pruned
    edges_pruned: u64,
    /// Whether the scan completed fully (did not hit edge_scan_limit)
    fully_scanned: bool,
}

/// Result of cleaning up a single deleted vector.
struct CleanupResult {
    /// Number of edges pruned
    edges_pruned: u64,
    /// Whether the VecId was recycled
    id_recycled: bool,
    /// Whether ID recycling was skipped due to incomplete edge scan
    recycling_skipped: bool,
}

// ============================================================================
// GarbageCollector
// ============================================================================

/// Background garbage collector for deleted vector cleanup.
///
/// The GC worker periodically scans for soft-deleted vectors and:
/// 1. Removes edges pointing to deleted vectors
/// 2. Deletes vector data and binary codes
/// 3. Optionally returns VecIds to free list
/// 4. Deletes VecMeta entries
pub struct GarbageCollector {
    storage: Arc<Storage>,
    #[allow(dead_code)]
    registry: Arc<EmbeddingRegistry>,
    config: GcConfig,
    shutdown: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    metrics: Arc<GcMetrics>,
}

impl GarbageCollector {
    /// Start the garbage collector with a background worker thread.
    ///
    /// # Arguments
    /// * `storage` - Storage for RocksDB access
    /// * `registry` - Embedding registry for looking up specs
    /// * `config` - GC configuration
    ///
    /// If `config.process_on_startup` is true, runs an immediate cleanup cycle
    /// before starting the periodic worker.
    pub fn start(
        storage: Arc<Storage>,
        registry: Arc<EmbeddingRegistry>,
        config: GcConfig,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let metrics = Arc::new(GcMetrics::default());

        // Run startup cleanup if configured
        if config.process_on_startup {
            info!("GarbageCollector: running startup cleanup");
            if let Err(e) = Self::run_cleanup_cycle_static(&storage, &config, &metrics) {
                error!(error = %e, "Failed startup cleanup cycle");
            }
        }

        // Spawn worker thread
        let worker = {
            let storage = Arc::clone(&storage);
            let config = config.clone();
            let shutdown = Arc::clone(&shutdown);
            let metrics = Arc::clone(&metrics);

            thread::spawn(move || {
                Self::worker_loop(storage, config, shutdown, metrics);
            })
        };

        info!(
            interval_secs = config.interval.as_secs(),
            batch_size = config.batch_size,
            id_recycling = config.enable_id_recycling,
            "GarbageCollector started"
        );

        Self {
            storage,
            registry,
            config,
            shutdown,
            worker: Some(worker),
            metrics,
        }
    }

    /// Gracefully shutdown the garbage collector.
    ///
    /// Signals the worker to stop and waits for the current cycle to complete.
    pub fn shutdown(mut self) {
        info!("GarbageCollector: initiating shutdown");
        self.shutdown.store(true, Ordering::SeqCst);

        if let Some(worker) = self.worker.take() {
            if let Err(e) = worker.join() {
                error!("GC worker thread panicked: {:?}", e);
            }
        }

        info!(
            vectors_cleaned = self.metrics.vectors_cleaned.load(Ordering::Relaxed),
            edges_pruned = self.metrics.edges_pruned.load(Ordering::Relaxed),
            cycles_completed = self.metrics.cycles_completed.load(Ordering::Relaxed),
            "GarbageCollector shutdown complete"
        );
    }

    /// Check if shutdown has been requested.
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }

    /// Get the GC metrics.
    pub fn metrics(&self) -> &GcMetrics {
        &self.metrics
    }

    /// Get the configuration.
    pub fn config(&self) -> &GcConfig {
        &self.config
    }

    // ========================================================================
    // Worker Implementation
    // ========================================================================

    fn worker_loop(
        storage: Arc<Storage>,
        config: GcConfig,
        shutdown: Arc<AtomicBool>,
        metrics: Arc<GcMetrics>,
    ) {
        info!("GC worker started");

        let mut last_cycle = Instant::now();

        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            // Sleep until next cycle
            let elapsed = last_cycle.elapsed();
            if elapsed < config.interval {
                let remaining = config.interval - elapsed;
                // Sleep in small increments to check shutdown flag
                let sleep_chunk = Duration::from_millis(100);
                let chunks = remaining.as_millis() / sleep_chunk.as_millis();
                for _ in 0..chunks {
                    if shutdown.load(Ordering::Relaxed) {
                        break;
                    }
                    thread::sleep(sleep_chunk);
                }
                if shutdown.load(Ordering::Relaxed) {
                    break;
                }
            }

            last_cycle = Instant::now();

            // Run cleanup cycle
            match Self::run_cleanup_cycle_static(&storage, &config, &metrics) {
                Ok(cleaned) => {
                    if cleaned > 0 {
                        info!(
                            vectors_cleaned = cleaned,
                            total = metrics.vectors_cleaned.load(Ordering::Relaxed),
                            "GC cycle completed"
                        );
                    } else {
                        debug!("GC cycle completed (no deleted vectors found)");
                    }
                }
                Err(e) => {
                    error!(error = %e, "GC cycle failed");
                }
            }

            metrics.cycles_completed.fetch_add(1, Ordering::Relaxed);
        }

        info!("GC worker stopped");
    }

    /// Run a single cleanup cycle (static version for internal use).
    fn run_cleanup_cycle_static(
        storage: &Storage,
        config: &GcConfig,
        metrics: &GcMetrics,
    ) -> anyhow::Result<usize> {
        let txn_db = storage.transaction_db()?;

        // Find deleted vectors by scanning VecMeta
        let deleted = Self::find_deleted_vectors(&txn_db, config.batch_size)?;

        if deleted.is_empty() {
            return Ok(0);
        }

        debug!(count = deleted.len(), "Found deleted vectors to clean");

        let mut total_cleaned = 0u64;
        let mut total_edges_pruned = 0u64;
        let mut total_ids_recycled = 0u64;
        let mut total_recycling_skipped = 0u64;

        for (embedding, vec_id) in &deleted {
            match Self::cleanup_vector(&txn_db, config, *embedding, *vec_id) {
                Ok(result) => {
                    total_cleaned += 1;
                    total_edges_pruned += result.edges_pruned;
                    if result.id_recycled {
                        total_ids_recycled += 1;
                    }
                    if result.recycling_skipped {
                        total_recycling_skipped += 1;
                    }
                }
                Err(e) => {
                    warn!(embedding, vec_id, error = %e, "Failed to cleanup vector");
                }
            }
        }

        metrics
            .vectors_cleaned
            .fetch_add(total_cleaned, Ordering::Relaxed);
        metrics
            .edges_pruned
            .fetch_add(total_edges_pruned, Ordering::Relaxed);
        metrics
            .ids_recycled
            .fetch_add(total_ids_recycled, Ordering::Relaxed);
        metrics
            .recycling_skipped_incomplete
            .fetch_add(total_recycling_skipped, Ordering::Relaxed);

        Ok(total_cleaned as usize)
    }

    /// Find vectors marked as deleted by scanning VecMeta.
    ///
    /// Returns up to `limit` (embedding_code, vec_id) pairs.
    fn find_deleted_vectors(
        txn_db: &rocksdb::TransactionDB,
        limit: usize,
    ) -> anyhow::Result<Vec<(EmbeddingCode, VecId)>> {
        let meta_cf = txn_db
            .cf_handle(VecMeta::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("VecMeta CF not found"))?;

        let mut deleted = Vec::new();
        let iter = txn_db.iterator_cf(&meta_cf, rocksdb::IteratorMode::Start);

        for item in iter {
            if deleted.len() >= limit {
                break;
            }

            let (key, value) = item?;

            // Parse key to get embedding_code and vec_id
            let meta_key = VecMeta::key_from_bytes(&key)?;

            // Check if deleted using zero-copy archived access
            // VecMetaCfValue wraps VecMetadata, so we access .0 for the inner type
            if let Ok(archived) = VecMeta::value_archived(&value) {
                if archived.0.is_deleted() {
                    deleted.push((meta_key.0, meta_key.1));
                }
            }
        }

        Ok(deleted)
    }

    /// Clean up a single deleted vector.
    ///
    /// 1. Scans Edges CF to find neighbors that reference this vec_id
    /// 2. Removes vec_id from those neighbor bitmaps
    /// 3. Deletes vector data and binary codes
    /// 4. Deletes VecMeta entry
    /// 5. Optionally recycles VecId (if enabled AND edge scan was complete)
    ///
    /// Returns CleanupResult with edges_pruned, id_recycled, and recycling_skipped.
    fn cleanup_vector(
        txn_db: &rocksdb::TransactionDB,
        config: &GcConfig,
        embedding: EmbeddingCode,
        vec_id: VecId,
    ) -> anyhow::Result<CleanupResult> {
        let txn = txn_db.transaction();

        // 1. Find and prune edges pointing to this vec_id
        let prune_result = Self::prune_edges(&txn, txn_db, config, embedding, vec_id)?;
        // COMMENT (CODEX, 2026-01-19): If prune_result.fully_scanned is false, we still delete
        // vector data and VecMeta below. This can leave dangling edge references to missing
        // vectors and cause search errors. Consider deferring data/VecMeta deletion or
        // marking a GC-pending state for retry.

        // 2. Delete vector data
        let vectors_cf = txn_db
            .cf_handle(Vectors::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;
        let vec_key = VectorCfKey(embedding, vec_id);
        txn.delete_cf(&vectors_cf, Vectors::key_to_bytes(&vec_key))?;

        // 3. Delete binary codes
        let codes_cf = txn_db
            .cf_handle(BinaryCodes::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("BinaryCodes CF not found"))?;
        let code_key = BinaryCodeCfKey(embedding, vec_id);
        txn.delete_cf(&codes_cf, BinaryCodes::key_to_bytes(&code_key))?;

        // 4. Delete this vector's own edges (all layers)
        let edges_cf = txn_db
            .cf_handle(Edges::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Edges CF not found"))?;
        // Scan all layers (0-255)
        for layer in 0..=255u8 {
            let edge_key = EdgeCfKey(embedding, vec_id, layer);
            let key_bytes = Edges::key_to_bytes(&edge_key);
            // Check if key exists before deleting (to avoid unnecessary operations)
            if txn.get_cf(&edges_cf, &key_bytes)?.is_some() {
                txn.delete_cf(&edges_cf, &key_bytes)?;
            } else {
                // Once we hit a layer that doesn't exist, higher layers won't either
                break;
            }
        }

        // 5. Delete VecMeta entry (marks cleanup complete)
        let meta_cf = txn_db
            .cf_handle(VecMeta::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("VecMeta CF not found"))?;
        let meta_key = VecMetaCfKey(embedding, vec_id);
        txn.delete_cf(&meta_cf, VecMeta::key_to_bytes(&meta_key))?;

        // 6. Recycle VecId if enabled AND edge scan was complete (Task 8.1.3 + 8.1.11)
        // ADDRESSED (Claude, 2026-01-19): Skip ID recycling when edge_scan_limit is hit
        // to prevent reusing an ID that may still have edge references.
        let (id_recycled, recycling_skipped) = if config.enable_id_recycling {
            if prune_result.fully_scanned {
                // Safe: we confirmed all edges were scanned and pruned
                Self::recycle_vec_id(&txn, txn_db, embedding, vec_id)?;
                (true, false)
            } else {
                // Unsafe: edge scan was incomplete, defer ID recycling
                warn!(
                    embedding,
                    vec_id,
                    "Skipping ID recycling - edge scan incomplete"
                );
                (false, true)
            }
        } else {
            (false, false)
        };

        // Commit the cleanup transaction
        txn.commit()?;

        debug!(
            embedding,
            vec_id,
            edges_pruned = prune_result.edges_pruned,
            id_recycled,
            recycling_skipped,
            "Cleaned up deleted vector"
        );

        Ok(CleanupResult {
            edges_pruned: prune_result.edges_pruned,
            id_recycled,
            recycling_skipped,
        })
    }

    /// Prune edges pointing to a deleted vector.
    ///
    /// Scans the Edges CF for this embedding and removes `vec_id` from any
    /// neighbor bitmaps that contain it.
    ///
    /// This is O(E) where E is the number of edge entries for the embedding,
    /// limited by `config.edge_scan_limit`.
    ///
    /// Returns a `PruneResult` with the number of edges pruned and whether
    /// the scan completed fully. If `fully_scanned` is false, ID recycling
    /// should be skipped to avoid reusing an ID that may still have references.
    fn prune_edges(
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        config: &GcConfig,
        embedding: EmbeddingCode,
        deleted_vec_id: VecId,
    ) -> anyhow::Result<PruneResult> {
        let edges_cf = txn_db
            .cf_handle(Edges::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Edges CF not found"))?;

        // Prefix for this embedding (8 bytes)
        let prefix = embedding.to_be_bytes();

        let iter = txn_db.iterator_cf(
            &edges_cf,
            rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
        );

        let mut edges_pruned = 0u64;
        let mut scanned = 0usize;
        let mut fully_scanned = true;

        for item in iter {
            if scanned >= config.edge_scan_limit {
                warn!(
                    embedding,
                    vec_id = deleted_vec_id,
                    scanned,
                    "Edge scan limit reached, some edges may not be pruned"
                );
                fully_scanned = false;
                break;
            }

            let (key, value) = item?;

            // Check prefix match
            if key.len() < 8 || key[0..8] != prefix {
                break;
            }

            scanned += 1;

            // Parse the edge entry
            let edge_key = Edges::key_from_bytes(&key)?;

            // Skip if this is the deleted vector's own edge list
            if edge_key.1 == deleted_vec_id {
                continue;
            }

            // Deserialize the neighbor bitmap
            let bitmap = match RoaringBitmap::deserialize_from(&value[..]) {
                Ok(bm) => bm,
                Err(e) => {
                    warn!(error = %e, "Failed to deserialize edge bitmap, skipping");
                    continue;
                }
            };

            // Check if this bitmap contains the deleted vec_id
            if bitmap.contains(deleted_vec_id) {
                // Use merge operator to remove the deleted vec_id
                let remove_op = EdgeOp::Remove(deleted_vec_id);
                txn.merge_cf(&edges_cf, &key, remove_op.to_bytes())?;
                edges_pruned += 1;
            }
        }

        Ok(PruneResult {
            edges_pruned,
            fully_scanned,
        })
    }

    /// Recycle a VecId by adding it to the free bitmap (Task 8.1.3).
    ///
    /// This directly updates the IdAlloc CF's free bitmap within the transaction.
    /// Safe to call after edge pruning confirms no references to this vec_id.
    ///
    /// # Note
    ///
    /// This bypasses the in-memory IdAllocator state. If a Processor is using
    /// the allocator concurrently, there may be temporary inconsistency until
    /// the allocator is reloaded from RocksDB. This is acceptable for background
    /// GC because:
    /// 1. Recycled IDs become available on next allocator reload
    /// 2. The allocator won't accidentally reuse an ID before it's freed here
    /// 3. GC is the only code path that frees IDs for HNSW-enabled embeddings
    fn recycle_vec_id(
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        embedding: EmbeddingCode,
        vec_id: VecId,
    ) -> anyhow::Result<()> {
        let alloc_cf = txn_db
            .cf_handle(IdAlloc::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("IdAlloc CF not found"))?;

        // Load current free bitmap
        let bitmap_key = IdAllocCfKey::free_bitmap(embedding);
        let key_bytes = IdAlloc::key_to_bytes(&bitmap_key);

        let mut bitmap = match txn.get_cf(&alloc_cf, &key_bytes)? {
            Some(bytes) => {
                let value = IdAlloc::value_from_bytes(&bitmap_key, &bytes)?;
                match value.0 {
                    IdAllocField::FreeBitmap(bitmap_bytes) => {
                        RoaringBitmap::deserialize_from(&bitmap_bytes[..])?
                    }
                    _ => anyhow::bail!("Unexpected IdAlloc value type"),
                }
            }
            None => RoaringBitmap::new(), // No free bitmap yet
        };

        // Add the recycled vec_id
        bitmap.insert(vec_id);

        // Write back
        let mut bitmap_bytes = Vec::new();
        bitmap.serialize_into(&mut bitmap_bytes)?;
        let value = IdAllocCfValue(IdAllocField::FreeBitmap(bitmap_bytes));
        txn.put_cf(&alloc_cf, &key_bytes, IdAlloc::value_to_bytes(&value))?;

        debug!(embedding, vec_id, "Recycled VecId");
        Ok(())
    }

    /// Run a single cleanup cycle manually (for testing).
    ///
    /// This is a convenience method that runs one cleanup cycle synchronously.
    pub fn run_cycle(&self) -> anyhow::Result<usize> {
        Self::run_cleanup_cycle_static(&self.storage, &self.config, &self.metrics)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::embedding::EmbeddingBuilder;
    use crate::vector::processor::Processor;
    use crate::vector::Distance;
    use tempfile::TempDir;

    fn setup_test_env(
        temp_dir: &TempDir,
    ) -> (Arc<Storage>, Arc<EmbeddingRegistry>) {
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().expect("Failed to initialize storage");
        let storage = Arc::new(storage);
        let registry = Arc::new(EmbeddingRegistry::new());
        (storage, registry)
    }

    fn register_embedding(
        storage: &Storage,
        registry: &EmbeddingRegistry,
        dim: u32,
    ) -> crate::vector::embedding::Embedding {
        let txn_db = storage.transaction_db().expect("Failed to get txn_db");
        let builder = EmbeddingBuilder::new("test-model", dim, Distance::L2);
        registry
            .register(builder, &txn_db)
            .expect("Failed to register embedding")
    }

    fn test_vector(dim: usize, seed: u64) -> Vec<f32> {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha20Rng;
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..dim).map(|_| rng.gen::<f32>()).collect()
    }

    #[test]
    fn test_gc_config_defaults() {
        let config = GcConfig::default();
        assert_eq!(config.interval, Duration::from_secs(60));
        assert_eq!(config.batch_size, 100);
        assert!(config.process_on_startup);
        assert_eq!(config.edge_scan_limit, 10000);
        assert!(!config.enable_id_recycling);
    }

    #[test]
    fn test_gc_config_builder() {
        let config = GcConfig::new()
            .with_interval(Duration::from_secs(30))
            .with_batch_size(50)
            .with_process_on_startup(false)
            .with_edge_scan_limit(5000)
            .with_id_recycling(true);

        assert_eq!(config.interval, Duration::from_secs(30));
        assert_eq!(config.batch_size, 50);
        assert!(!config.process_on_startup);
        assert_eq!(config.edge_scan_limit, 5000);
        assert!(config.enable_id_recycling);
    }

    #[test]
    fn test_find_deleted_vectors_empty() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let (storage, registry) = setup_test_env(&temp_dir);
        let _embedding = register_embedding(&storage, &registry, 64);

        let txn_db = storage.transaction_db().expect("Failed to get txn_db");
        let deleted = GarbageCollector::find_deleted_vectors(&txn_db, 100)
            .expect("Should succeed");

        assert!(deleted.is_empty(), "No deleted vectors should exist");
    }

    #[test]
    fn test_gc_finds_deleted_vectors() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let (storage, registry) = setup_test_env(&temp_dir);
        let embedding = register_embedding(&storage, &registry, 64);

        let processor = Processor::new(storage.clone(), registry.clone());

        // Insert vectors
        let mut ids = Vec::new();
        for i in 0..5 {
            let id = crate::Id::new();
            let vector = test_vector(64, i);
            processor
                .insert_vector(&embedding, id, &vector, true)
                .expect("Insert should succeed");
            ids.push(id);
        }

        // Delete some vectors
        for id in &ids[0..3] {
            processor
                .delete_vector(&embedding, *id)
                .expect("Delete should succeed");
        }

        // Find deleted vectors
        let txn_db = storage.transaction_db().expect("Failed to get txn_db");
        let deleted = GarbageCollector::find_deleted_vectors(&txn_db, 100)
            .expect("Should succeed");

        assert_eq!(deleted.len(), 3, "Should find 3 deleted vectors");
    }

    #[test]
    fn test_gc_cleanup_removes_vector_data() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let (storage, registry) = setup_test_env(&temp_dir);
        let embedding = register_embedding(&storage, &registry, 64);

        let processor = Processor::new(storage.clone(), registry.clone());

        // Insert a vector with HNSW indexing
        let id = crate::Id::new();
        let vector = test_vector(64, 42);
        let vec_id = processor
            .insert_vector(&embedding, id, &vector, true)
            .expect("Insert should succeed");

        // Delete it (soft delete)
        processor
            .delete_vector(&embedding, id)
            .expect("Delete should succeed");

        // Verify vector data still exists (soft delete keeps it)
        let txn_db = storage.transaction_db().expect("Failed to get txn_db");
        let vectors_cf = txn_db.cf_handle(Vectors::CF_NAME).unwrap();
        let vec_key = VectorCfKey(embedding.code(), vec_id);
        assert!(
            txn_db.get_cf(&vectors_cf, Vectors::key_to_bytes(&vec_key)).unwrap().is_some(),
            "Vector data should still exist after soft delete"
        );

        // Run GC
        let config = GcConfig::new().with_process_on_startup(false);
        let metrics = Arc::new(GcMetrics::default());
        GarbageCollector::run_cleanup_cycle_static(&storage, &config, &metrics)
            .expect("GC should succeed");

        // Verify vector data is now gone
        assert!(
            txn_db.get_cf(&vectors_cf, Vectors::key_to_bytes(&vec_key)).unwrap().is_none(),
            "Vector data should be deleted after GC"
        );

        // Verify VecMeta is gone
        let meta_cf = txn_db.cf_handle(VecMeta::CF_NAME).unwrap();
        let meta_key = VecMetaCfKey(embedding.code(), vec_id);
        assert!(
            txn_db.get_cf(&meta_cf, VecMeta::key_to_bytes(&meta_key)).unwrap().is_none(),
            "VecMeta should be deleted after GC"
        );

        // Verify metrics
        assert_eq!(
            metrics.vectors_cleaned.load(Ordering::Relaxed),
            1,
            "Should report 1 vector cleaned"
        );
    }

    #[test]
    fn test_gc_edge_pruning() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let (storage, registry) = setup_test_env(&temp_dir);
        let embedding = register_embedding(&storage, &registry, 64);

        let processor = Processor::new(storage.clone(), registry.clone());

        // Insert multiple vectors to create a connected graph
        let mut ids = Vec::new();
        for i in 0..10 {
            let id = crate::Id::new();
            let vector = test_vector(64, i);
            processor
                .insert_vector(&embedding, id, &vector, true)
                .expect("Insert should succeed");
            ids.push(id);
        }

        // Delete one vector
        let deleted_result = processor
            .delete_vector(&embedding, ids[5])
            .expect("Delete should succeed");
        let deleted_vec_id = deleted_result.unwrap();

        // Run GC
        let config = GcConfig::new().with_process_on_startup(false);
        let metrics = Arc::new(GcMetrics::default());
        GarbageCollector::run_cleanup_cycle_static(&storage, &config, &metrics)
            .expect("GC should succeed");

        // Verify edges no longer contain the deleted vec_id
        let txn_db = storage.transaction_db().expect("Failed to get txn_db");
        let edges_cf = txn_db.cf_handle(Edges::CF_NAME).unwrap();

        let prefix = embedding.code().to_be_bytes();
        let iter = txn_db.iterator_cf(
            &edges_cf,
            rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
        );

        for item in iter {
            let (key, value) = item.expect("Iterator error");
            if key.len() < 8 || key[0..8] != prefix {
                break;
            }

            let bitmap = RoaringBitmap::deserialize_from(&value[..])
                .expect("Should deserialize bitmap");

            assert!(
                !bitmap.contains(deleted_vec_id),
                "No edge should contain the deleted vec_id after GC"
            );
        }
    }

    #[test]
    fn test_gc_startup_and_shutdown() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let (storage, registry) = setup_test_env(&temp_dir);

        // Start GC with short interval
        let config = GcConfig::new()
            .with_interval(Duration::from_millis(100))
            .with_process_on_startup(false);

        let gc = GarbageCollector::start(storage.clone(), registry.clone(), config);

        // Let it run briefly
        std::thread::sleep(Duration::from_millis(50));

        assert!(!gc.is_shutdown(), "Should not be shutdown yet");

        // Shutdown
        gc.shutdown();
    }

    #[test]
    fn test_gc_metrics() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let (storage, registry) = setup_test_env(&temp_dir);
        let embedding = register_embedding(&storage, &registry, 64);

        let processor = Processor::new(storage.clone(), registry.clone());

        // Insert and delete vectors
        for i in 0..5 {
            let id = crate::Id::new();
            let vector = test_vector(64, i);
            processor
                .insert_vector(&embedding, id, &vector, true)
                .expect("Insert should succeed");
            processor
                .delete_vector(&embedding, id)
                .expect("Delete should succeed");
        }

        // Run GC
        let config = GcConfig::new().with_process_on_startup(false);
        let gc = GarbageCollector::start(storage.clone(), registry.clone(), config);

        // Run a manual cycle
        gc.run_cycle().expect("Cycle should succeed");

        let metrics = gc.metrics();
        assert_eq!(
            metrics.vectors_cleaned.load(Ordering::Relaxed),
            5,
            "Should clean 5 vectors"
        );

        gc.shutdown();
    }

    #[test]
    fn test_gc_id_recycling() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let (storage, registry) = setup_test_env(&temp_dir);
        let embedding = register_embedding(&storage, &registry, 64);

        let processor = Processor::new(storage.clone(), registry.clone());

        // Insert a vector with HNSW indexing and note the vec_id
        let id = crate::Id::new();
        let vector = test_vector(64, 42);
        let vec_id = processor
            .insert_vector(&embedding, id, &vector, true)
            .expect("Insert should succeed");

        // Delete it (soft delete)
        processor
            .delete_vector(&embedding, id)
            .expect("Delete should succeed");

        // Run GC with ID recycling enabled
        let config = GcConfig::new()
            .with_process_on_startup(false)
            .with_id_recycling(true);
        let metrics = Arc::new(GcMetrics::default());
        GarbageCollector::run_cleanup_cycle_static(&storage, &config, &metrics)
            .expect("GC should succeed");

        // Verify ID was recycled - check free bitmap in IdAlloc CF
        let txn_db = storage.transaction_db().expect("Failed to get txn_db");
        let alloc_cf = txn_db.cf_handle(IdAlloc::CF_NAME).unwrap();
        let bitmap_key = IdAllocCfKey::free_bitmap(embedding.code());
        let key_bytes = IdAlloc::key_to_bytes(&bitmap_key);

        let bitmap_bytes = txn_db
            .get_cf(&alloc_cf, &key_bytes)
            .expect("Should be able to read IdAlloc")
            .expect("Free bitmap should exist");

        let value = IdAlloc::value_from_bytes(&bitmap_key, &bitmap_bytes)
            .expect("Should deserialize value");
        let bitmap = match value.0 {
            IdAllocField::FreeBitmap(bytes) => {
                RoaringBitmap::deserialize_from(&bytes[..]).expect("Should deserialize bitmap")
            }
            _ => panic!("Expected FreeBitmap"),
        };

        assert!(
            bitmap.contains(vec_id),
            "Recycled vec_id {} should be in free bitmap",
            vec_id
        );

        // Verify metrics
        assert_eq!(
            metrics.ids_recycled.load(Ordering::Relaxed),
            1,
            "Should report 1 ID recycled"
        );
    }

    #[test]
    fn test_gc_id_recycling_disabled_by_default() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let (storage, registry) = setup_test_env(&temp_dir);
        let embedding = register_embedding(&storage, &registry, 64);

        let processor = Processor::new(storage.clone(), registry.clone());

        // Insert and delete a vector
        let id = crate::Id::new();
        let vector = test_vector(64, 42);
        let vec_id = processor
            .insert_vector(&embedding, id, &vector, true)
            .expect("Insert should succeed");
        processor
            .delete_vector(&embedding, id)
            .expect("Delete should succeed");

        // Run GC with default config (ID recycling disabled)
        let config = GcConfig::new().with_process_on_startup(false);
        let metrics = Arc::new(GcMetrics::default());
        GarbageCollector::run_cleanup_cycle_static(&storage, &config, &metrics)
            .expect("GC should succeed");

        // Verify ID was NOT recycled - check free bitmap is empty or doesn't contain vec_id
        let txn_db = storage.transaction_db().expect("Failed to get txn_db");
        let alloc_cf = txn_db.cf_handle(IdAlloc::CF_NAME).unwrap();
        let bitmap_key = IdAllocCfKey::free_bitmap(embedding.code());
        let key_bytes = IdAlloc::key_to_bytes(&bitmap_key);

        let bitmap_empty = match txn_db.get_cf(&alloc_cf, &key_bytes).expect("Read should succeed") {
            Some(bitmap_bytes) => {
                let value = IdAlloc::value_from_bytes(&bitmap_key, &bitmap_bytes)
                    .expect("Should deserialize value");
                match value.0 {
                    IdAllocField::FreeBitmap(bytes) => {
                        let bitmap = RoaringBitmap::deserialize_from(&bytes[..])
                            .expect("Should deserialize bitmap");
                        !bitmap.contains(vec_id)
                    }
                    _ => true,
                }
            }
            None => true, // No bitmap means no recycling happened
        };

        assert!(
            bitmap_empty,
            "Vec_id {} should NOT be in free bitmap when recycling disabled",
            vec_id
        );

        // Verify metrics
        assert_eq!(
            metrics.ids_recycled.load(Ordering::Relaxed),
            0,
            "Should report 0 IDs recycled when disabled"
        );
    }
}
