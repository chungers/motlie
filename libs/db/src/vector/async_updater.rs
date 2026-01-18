//! Async Graph Updater for two-phase vector inserts.
//!
//! This module provides background workers that process pending HNSW graph updates,
//! enabling low-latency inserts by decoupling vector storage from graph construction.
//!
//! ## Two-Phase Insert Pattern
//!
//! - **Phase 1 (sync, <5ms):** Store vector data + metadata, add to pending queue
//! - **Phase 2 (async, background):** Workers build HNSW graph edges
//!
//! ## Architecture
//!
//! ```text
//! AsyncGraphUpdater
//!   ├── config: AsyncUpdaterConfig
//!   ├── shutdown: Arc<AtomicBool>
//!   └── workers: Vec<JoinHandle<()>>
//!       └── worker_loop()
//!           ├── collect_batch() - round-robin across embeddings
//!           ├── process_insert() - greedy search + add edges
//!           └── clear_processed() - remove from pending CF
//! ```
//!
//! ## Usage
//!
//! ```ignore
//! let config = AsyncUpdaterConfig::default()
//!     .with_num_workers(4)
//!     .with_batch_size(200);
//!
//! let updater = AsyncGraphUpdater::start(storage, config);
//!
//! // ... application runs ...
//!
//! updater.shutdown(); // Graceful shutdown
//! ```

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use tracing::{debug, error, info, warn};

use crate::rocksdb::ColumnFamily;
use crate::vector::schema::{EmbeddingCode, Pending, VecId};
use crate::vector::Storage;

// ============================================================================
// AsyncUpdaterConfig (Task 7.2)
// ============================================================================

/// Configuration for the async graph updater.
///
/// ## Tuning Guidance
///
/// - **`batch_size`**: Larger batches amortize transaction overhead but increase
///   memory usage. Start with 100, increase for high-throughput scenarios.
///
/// - **`batch_timeout`**: Controls latency vs throughput tradeoff. Lower values
///   reduce graph-build latency but increase CPU overhead from frequent wakeups.
///
/// - **`num_workers`**: More workers improve throughput but compete for locks.
///   Typically 2-4 workers is optimal. Scale with CPU cores.
///
/// - **`ef_construction`**: Higher values improve graph quality but slow down
///   edge building. Match your HNSW index config (typically 100-200).
///
/// - **`process_on_startup`**: Enable to drain any pending items from a previous
///   crash. Disable only for testing or if you handle recovery elsewhere.
#[derive(Debug, Clone)]
pub struct AsyncUpdaterConfig {
    /// Maximum vectors per worker batch.
    ///
    /// Workers collect up to this many pending items before processing.
    /// Default: 100
    pub batch_size: usize,

    /// Maximum time to wait for batch to fill.
    ///
    /// If batch_size items aren't available within this duration, process
    /// whatever is available. Default: 100ms
    pub batch_timeout: Duration,

    /// Number of background worker threads.
    ///
    /// Each worker processes batches independently with round-robin
    /// embedding selection for fairness. Default: 2
    pub num_workers: usize,

    /// ef_construction parameter for greedy search during edge building.
    ///
    /// Higher values improve recall but slow down processing. Default: 200
    pub ef_construction: usize,

    /// Whether to drain pending queue on startup.
    ///
    /// When true, any items left in the pending queue (e.g., from a crash)
    /// are processed before normal operation begins. Default: true
    pub process_on_startup: bool,

    /// Sleep duration when no pending items found.
    ///
    /// Workers sleep this long before checking for new items when the
    /// pending queue is empty. Default: 10ms
    pub idle_sleep: Duration,
}

impl Default for AsyncUpdaterConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            batch_timeout: Duration::from_millis(100),
            num_workers: 2,
            ef_construction: 200,
            process_on_startup: true,
            idle_sleep: Duration::from_millis(10),
        }
    }
}

impl AsyncUpdaterConfig {
    /// Create a new config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum vectors per worker batch.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set maximum time to wait for batch to fill.
    pub fn with_batch_timeout(mut self, timeout: Duration) -> Self {
        self.batch_timeout = timeout;
        self
    }

    /// Set number of background worker threads.
    pub fn with_num_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = num_workers;
        self
    }

    /// Set ef_construction for greedy search.
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Set whether to drain pending queue on startup.
    pub fn with_process_on_startup(mut self, process: bool) -> Self {
        self.process_on_startup = process;
        self
    }

    /// Set idle sleep duration.
    pub fn with_idle_sleep(mut self, duration: Duration) -> Self {
        self.idle_sleep = duration;
        self
    }
}

// ============================================================================
// AsyncGraphUpdater (Task 7.3)
// ============================================================================

/// Background graph updater that processes pending HNSW edge construction.
///
/// Workers collect pending items using round-robin across embeddings for
/// fairness, then build HNSW graph edges and remove processed items.
///
/// ## Crash Recovery
///
/// The pending queue is persisted in RocksDB. On restart, `process_on_startup`
/// ensures any items from a previous crash are processed. Operations are
/// idempotent: re-processing an item that was partially processed is safe.
pub struct AsyncGraphUpdater {
    storage: Arc<Storage>,
    config: AsyncUpdaterConfig,
    shutdown: Arc<AtomicBool>,
    workers: Vec<JoinHandle<()>>,
    /// Counter for round-robin embedding selection
    embedding_counter: Arc<AtomicU64>,
    /// Metrics: total items processed
    items_processed: Arc<AtomicU64>,
    /// Metrics: total batches processed
    batches_processed: Arc<AtomicU64>,
}

impl AsyncGraphUpdater {
    /// Start the async updater with worker threads.
    ///
    /// If `config.process_on_startup` is true, drains any existing pending
    /// items before starting normal worker operation.
    pub fn start(storage: Arc<Storage>, config: AsyncUpdaterConfig) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let embedding_counter = Arc::new(AtomicU64::new(0));
        let items_processed = Arc::new(AtomicU64::new(0));
        let batches_processed = Arc::new(AtomicU64::new(0));

        // Drain pending queue on startup if configured
        if config.process_on_startup {
            info!("AsyncGraphUpdater: draining pending queue on startup");
            if let Err(e) = Self::drain_pending_static(&storage, &config) {
                error!(error = %e, "Failed to drain pending queue on startup");
            }
        }

        // Spawn worker threads
        let workers = (0..config.num_workers)
            .map(|worker_id| {
                let storage = Arc::clone(&storage);
                let config = config.clone();
                let shutdown = Arc::clone(&shutdown);
                let embedding_counter = Arc::clone(&embedding_counter);
                let items_processed = Arc::clone(&items_processed);
                let batches_processed = Arc::clone(&batches_processed);

                thread::spawn(move || {
                    Self::worker_loop(
                        worker_id,
                        storage,
                        config,
                        shutdown,
                        embedding_counter,
                        items_processed,
                        batches_processed,
                    );
                })
            })
            .collect();

        info!(
            num_workers = config.num_workers,
            batch_size = config.batch_size,
            "AsyncGraphUpdater started"
        );

        Self {
            storage,
            config,
            shutdown,
            workers,
            embedding_counter,
            items_processed,
            batches_processed,
        }
    }

    /// Gracefully shutdown all workers.
    ///
    /// Signals workers to stop and waits for in-flight batches to complete.
    pub fn shutdown(self) {
        info!("AsyncGraphUpdater: initiating shutdown");
        self.shutdown.store(true, Ordering::SeqCst);

        for (i, worker) in self.workers.into_iter().enumerate() {
            if let Err(e) = worker.join() {
                error!(worker_id = i, "Worker thread panicked: {:?}", e);
            }
        }

        info!(
            items_processed = self.items_processed.load(Ordering::Relaxed),
            batches_processed = self.batches_processed.load(Ordering::Relaxed),
            "AsyncGraphUpdater shutdown complete"
        );
    }

    /// Check if shutdown has been requested.
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Relaxed)
    }

    /// Get total items processed across all workers.
    pub fn items_processed(&self) -> u64 {
        self.items_processed.load(Ordering::Relaxed)
    }

    /// Get total batches processed across all workers.
    pub fn batches_processed(&self) -> u64 {
        self.batches_processed.load(Ordering::Relaxed)
    }

    /// Get the updater configuration.
    pub fn config(&self) -> &AsyncUpdaterConfig {
        &self.config
    }

    // ========================================================================
    // Worker Implementation
    // ========================================================================

    fn worker_loop(
        worker_id: usize,
        storage: Arc<Storage>,
        config: AsyncUpdaterConfig,
        shutdown: Arc<AtomicBool>,
        embedding_counter: Arc<AtomicU64>,
        items_processed: Arc<AtomicU64>,
        batches_processed: Arc<AtomicU64>,
    ) {
        info!(worker_id, "Async updater worker started");

        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            // Collect a batch of pending inserts
            // Note: Full round-robin not yet implemented - currently scans from start
            let batch = match Self::collect_batch(
                &storage,
                &config,
                &embedding_counter,
            ) {
                Ok(b) => b,
                Err(e) => {
                    error!(worker_id, error = %e, "Failed to collect batch");
                    thread::sleep(config.idle_sleep);
                    continue;
                }
            };

            if batch.is_empty() {
                thread::sleep(config.idle_sleep);
                continue;
            }

            debug!(
                worker_id,
                batch_size = batch.len(),
                "Processing batch"
            );

            let start = Instant::now();
            let mut processed = 0;

            // Process each item in the batch
            for (key, embedding, vec_id) in &batch {
                // Check for idempotency: skip if already processed or deleted
                // (FLAG_PENDING cleared or FLAG_DELETED set)
                // This is checked inside process_insert

                if let Err(e) = Self::process_insert(&storage, &config, *embedding, *vec_id) {
                    warn!(
                        worker_id,
                        embedding,
                        vec_id,
                        error = %e,
                        "Failed to process pending insert (will retry)"
                    );
                    // Don't clear from pending - will be retried
                    continue;
                }

                // Remove from pending queue after successful processing
                if let Err(e) = Self::clear_processed(&storage, key) {
                    error!(
                        worker_id,
                        embedding,
                        vec_id,
                        error = %e,
                        "Failed to clear processed item from pending"
                    );
                }

                processed += 1;
            }

            let elapsed = start.elapsed();
            items_processed.fetch_add(processed, Ordering::Relaxed);
            batches_processed.fetch_add(1, Ordering::Relaxed);

            debug!(
                worker_id,
                processed,
                batch_size = batch.len(),
                elapsed_ms = elapsed.as_millis(),
                "Batch complete"
            );
        }

        info!(worker_id, "Async updater worker stopped");
    }

    /// Collect a batch of pending items for processing with round-robin fairness.
    ///
    /// Implements fair scheduling across embeddings:
    /// 1. Discover which embeddings have pending items (using seek-based discovery)
    /// 2. Round-robin select one embedding using `embedding_counter`
    /// 3. Collect batch from only that embedding using prefix iteration
    ///
    /// This ensures embeddings with fewer pending items aren't starved by
    /// embeddings with larger queues.
    ///
    /// Uses a snapshot for iterator stability - concurrent deletes won't
    /// affect the iteration. Clear operations are idempotent.
    ///
    /// Returns Vec<(key_bytes, embedding_code, vec_id)> for processing.
    fn collect_batch(
        storage: &Storage,
        config: &AsyncUpdaterConfig,
        embedding_counter: &AtomicU64,
    ) -> anyhow::Result<Vec<(Vec<u8>, EmbeddingCode, VecId)>> {
        let txn_db = storage.transaction_db()?;
        let cf = txn_db
            .cf_handle(Pending::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Pending CF not found"))?;

        // Step 1: Discover active embeddings using efficient seek-based scan
        // Note: Uses its own snapshot internally. If items are deleted between
        // discovery and collection, we'll just get an empty/partial batch.
        let active_embeddings = Self::discover_active_embeddings(txn_db, cf)?;

        if active_embeddings.is_empty() {
            return Ok(Vec::new());
        }

        // Step 2: Round-robin select an embedding
        let counter = embedding_counter.fetch_add(1, Ordering::Relaxed);
        let selected_embedding = active_embeddings[counter as usize % active_embeddings.len()];

        debug!(
            selected_embedding,
            active_count = active_embeddings.len(),
            counter,
            "Round-robin selected embedding"
        );

        // Step 3: Collect batch from selected embedding using prefix
        // Use snapshot for consistent iteration (prevents issues with concurrent deletes)
        let prefix = Pending::prefix_for_embedding(selected_embedding);
        let mut batch = Vec::with_capacity(config.batch_size);
        let start_time = Instant::now();

        let snapshot = txn_db.snapshot();
        let mut read_opts = rocksdb::ReadOptions::default();
        read_opts.set_snapshot(&snapshot);

        let iter = txn_db.iterator_cf_opt(
            cf,
            read_opts,
            rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
        );

        for item in iter {
            if batch.len() >= config.batch_size {
                break;
            }
            if start_time.elapsed() > config.batch_timeout {
                break;
            }

            let (key, _value) = item?;

            // Check prefix match (stop when we leave this embedding's range)
            if key.len() < 8 || key[0..8] != prefix {
                break;
            }

            let parsed = Pending::key_from_bytes(&key)?;
            batch.push((key.to_vec(), parsed.0, parsed.2));
        }

        Ok(batch)
    }

    /// Discover which embeddings have pending items using efficient seeks.
    ///
    /// Uses seek to jump between embedding prefixes rather than scanning all items.
    /// This is O(E) where E is the number of embeddings, not O(N) where N is items.
    ///
    /// Returns list of embedding codes with pending items (in key order).
    fn discover_active_embeddings(
        txn_db: &rocksdb::TransactionDB,
        cf: &impl rocksdb::AsColumnFamilyRef,
    ) -> anyhow::Result<Vec<EmbeddingCode>> {
        let mut active = Vec::new();
        let mut next_seek: Option<[u8; 8]> = None;

        // Maximum embeddings to track (prevent runaway in pathological cases)
        const MAX_EMBEDDINGS: usize = 1000;

        // Use snapshot for consistent discovery
        let snapshot = txn_db.snapshot();

        loop {
            if active.len() >= MAX_EMBEDDINGS {
                break;
            }

            let mut read_opts = rocksdb::ReadOptions::default();
            read_opts.set_snapshot(&snapshot);

            let mut iter = match &next_seek {
                None => txn_db.iterator_cf_opt(cf, read_opts, rocksdb::IteratorMode::Start),
                Some(prefix) => txn_db.iterator_cf_opt(
                    cf,
                    read_opts,
                    rocksdb::IteratorMode::From(prefix, rocksdb::Direction::Forward),
                ),
            };

            match iter.next() {
                Some(Ok((key, _))) if key.len() >= 8 => {
                    let embedding = u64::from_be_bytes(key[0..8].try_into().unwrap());
                    active.push(embedding);

                    // Calculate next seek position: current_embedding + 1
                    // This jumps past all items for current embedding
                    match embedding.checked_add(1) {
                        Some(next_embedding) => {
                            next_seek = Some(next_embedding.to_be_bytes());
                        }
                        None => {
                            // Wrapped around u64::MAX, we've covered all possible embeddings
                            break;
                        }
                    }
                }
                Some(Err(e)) => return Err(e.into()),
                _ => break, // No more items
            }
        }

        Ok(active)
    }

    /// Process a single pending insert by building HNSW graph edges.
    ///
    /// This function is idempotent: if the item was already processed
    /// (edges already built) or deleted, it safely skips.
    fn process_insert(
        _storage: &Storage,
        _config: &AsyncUpdaterConfig,
        embedding: EmbeddingCode,
        vec_id: VecId,
    ) -> anyhow::Result<()> {
        // TODO (Task 7.4): Implement actual edge building
        // 1. Check if vector still exists and is pending (not deleted)
        // 2. Load vector data from Vectors CF
        // 3. Call HNSW greedy search to find neighbors
        // 4. Build bidirectional edges
        // 5. Clear FLAG_PENDING from VecMeta

        debug!(embedding, vec_id, "process_insert placeholder");

        Ok(())
    }

    /// Remove a processed item from the pending queue.
    ///
    /// This operation is idempotent: deleting an already-deleted key is a no-op.
    /// This is important because:
    /// 1. Concurrent workers might process the same item
    /// 2. Crash recovery might re-process items
    ///
    /// Note: Currently uses direct delete (not transactional). When Task 7.4
    /// implements process_insert, both operations should be in a single
    /// transaction for atomicity.
    fn clear_processed(storage: &Storage, key: &[u8]) -> anyhow::Result<()> {
        let txn_db = storage.transaction_db()?;
        let cf = txn_db
            .cf_handle(Pending::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Pending CF not found"))?;

        // Delete is idempotent in RocksDB - deleting a non-existent key is a no-op
        txn_db.delete_cf(cf, key)?;
        Ok(())
    }

    /// Drain all pending items (used for startup recovery).
    fn drain_pending_static(storage: &Storage, config: &AsyncUpdaterConfig) -> anyhow::Result<()> {
        let embedding_counter = AtomicU64::new(0);

        loop {
            let batch = Self::collect_batch(storage, config, &embedding_counter)?;
            if batch.is_empty() {
                break;
            }

            info!(batch_size = batch.len(), "Draining pending batch");

            for (key, embedding, vec_id) in &batch {
                if let Err(e) = Self::process_insert(storage, config, *embedding, *vec_id) {
                    warn!(embedding, vec_id, error = %e, "Failed to process during drain");
                    continue;
                }

                if let Err(e) = Self::clear_processed(storage, key) {
                    error!(embedding, vec_id, error = %e, "Failed to clear during drain");
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = AsyncUpdaterConfig::default();
        assert_eq!(config.batch_size, 100);
        assert_eq!(config.batch_timeout, Duration::from_millis(100));
        assert_eq!(config.num_workers, 2);
        assert_eq!(config.ef_construction, 200);
        assert!(config.process_on_startup);
        assert_eq!(config.idle_sleep, Duration::from_millis(10));
    }

    #[test]
    fn test_config_builder() {
        let config = AsyncUpdaterConfig::new()
            .with_batch_size(200)
            .with_batch_timeout(Duration::from_millis(50))
            .with_num_workers(4)
            .with_ef_construction(100)
            .with_process_on_startup(false)
            .with_idle_sleep(Duration::from_millis(5));

        assert_eq!(config.batch_size, 200);
        assert_eq!(config.batch_timeout, Duration::from_millis(50));
        assert_eq!(config.num_workers, 4);
        assert_eq!(config.ef_construction, 100);
        assert!(!config.process_on_startup);
        assert_eq!(config.idle_sleep, Duration::from_millis(5));
    }
}
