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
//!           └── process_insert() - greedy search + add edges + remove from pending (atomic)
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

use crate::rocksdb::{ColumnFamily, ColumnFamilySerde, HotColumnFamilyRecord};
use crate::vector::cache::NavigationCache;
use crate::vector::hnsw;
use crate::vector::registry::EmbeddingRegistry;
use crate::vector::schema::{
    EmbeddingCode, EmbeddingSpecCfKey, EmbeddingSpecs, LifecycleCounts, LifecycleCountsCfKey,
    LifecycleCountsDelta, Pending, VecId, VecMeta, VecMetaCfKey, VectorCfKey, Vectors,
};
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
/// - **`process_on_startup`**: Enable to drain any pending items from a previous
///   crash. Disable only for testing or if you handle recovery elsewhere.
///
/// - **`backpressure_threshold`**: Maximum pending items before inserts block.
///   Set to 0 to disable backpressure. Default: 10000
///
/// - **`metrics_interval`**: How often to log backlog depth and drain rate.
///   Set to None to disable periodic logging. Default: 10s
#[derive(Clone)]
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

    /// Maximum pending items before backpressure is applied.
    ///
    /// When the pending queue exceeds this threshold, new async inserts
    /// will be rejected with an error until the queue drains below threshold.
    /// Set to 0 to disable backpressure (unbounded queue). Default: 10000
    pub backpressure_threshold: usize,

    /// Interval for logging metrics (backlog depth, drain rate).
    ///
    /// Workers log pending queue size and processing rate at this interval.
    /// Set to None to disable periodic logging. Default: Some(10s)
    pub metrics_interval: Option<Duration>,

    /// Test-only shutdown hook for verifying shutdown ordering.
    #[cfg(any(test, feature = "test-hooks"))]
    pub shutdown_hook: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl std::fmt::Debug for AsyncUpdaterConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncUpdaterConfig")
            .field("batch_size", &self.batch_size)
            .field("batch_timeout", &self.batch_timeout)
            .field("num_workers", &self.num_workers)
            .field("process_on_startup", &self.process_on_startup)
            .field("idle_sleep", &self.idle_sleep)
            .field("backpressure_threshold", &self.backpressure_threshold)
            .field("metrics_interval", &self.metrics_interval)
            .finish()
    }
}

#[cfg(not(any(test, feature = "test-hooks")))]
impl Default for AsyncUpdaterConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            batch_timeout: Duration::from_millis(100),
            num_workers: 2,
            process_on_startup: true,
            idle_sleep: Duration::from_millis(10),
            backpressure_threshold: 10000,
            metrics_interval: Some(Duration::from_secs(10)),
        }
    }
}

#[cfg(any(test, feature = "test-hooks"))]
impl Default for AsyncUpdaterConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            batch_timeout: Duration::from_millis(100),
            num_workers: 2,
            process_on_startup: true,
            idle_sleep: Duration::from_millis(10),
            backpressure_threshold: 10000,
            metrics_interval: Some(Duration::from_secs(10)),
            shutdown_hook: None,
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

    /// Set backpressure threshold.
    ///
    /// When pending queue exceeds this, async inserts are rejected.
    /// Set to 0 to disable backpressure.
    pub fn with_backpressure_threshold(mut self, threshold: usize) -> Self {
        self.backpressure_threshold = threshold;
        self
    }

    /// Set metrics logging interval.
    ///
    /// Set to None to disable periodic metrics logging.
    pub fn with_metrics_interval(mut self, interval: Option<Duration>) -> Self {
        self.metrics_interval = interval;
        self
    }

    /// Disable backpressure (allow unbounded queue growth).
    pub fn no_backpressure(mut self) -> Self {
        self.backpressure_threshold = 0;
        self
    }

    /// Disable periodic metrics logging.
    pub fn no_metrics_logging(mut self) -> Self {
        self.metrics_interval = None;
        self
    }

    /// Set shutdown hook for testing shutdown ordering.
    #[cfg(any(test, feature = "test-hooks"))]
    pub fn with_shutdown_hook<F>(mut self, hook: F) -> Self
    where
        F: Fn() + Send + Sync + 'static,
    {
        self.shutdown_hook = Some(Arc::new(hook));
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
    registry: Arc<EmbeddingRegistry>,
    nav_cache: Arc<NavigationCache>,
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
    /// # Arguments
    /// * `storage` - Storage for RocksDB access
    /// * `registry` - Embedding registry for looking up specs
    /// * `nav_cache` - Navigation cache for HNSW operations
    /// * `config` - Updater configuration
    ///
    /// If `config.process_on_startup` is true, drains any existing pending
    /// items before starting normal worker operation.
    pub fn start(
        storage: Arc<Storage>,
        registry: Arc<EmbeddingRegistry>,
        nav_cache: Arc<NavigationCache>,
        config: AsyncUpdaterConfig,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let embedding_counter = Arc::new(AtomicU64::new(0));
        let items_processed = Arc::new(AtomicU64::new(0));
        let batches_processed = Arc::new(AtomicU64::new(0));

        // Drain pending queue on startup if configured
        if config.process_on_startup {
            info!("AsyncGraphUpdater: draining pending queue on startup");
            if let Err(e) =
                Self::drain_pending_static(&storage, &registry, &nav_cache, &config)
            {
                error!(error = %e, "Failed to drain pending queue on startup");
            }
        }

        // Spawn worker threads
        let workers = (0..config.num_workers)
            .map(|worker_id| {
                let storage = Arc::clone(&storage);
                let registry = Arc::clone(&registry);
                let nav_cache = Arc::clone(&nav_cache);
                let config = config.clone();
                let shutdown = Arc::clone(&shutdown);
                let embedding_counter = Arc::clone(&embedding_counter);
                let items_processed = Arc::clone(&items_processed);
                let batches_processed = Arc::clone(&batches_processed);

                thread::spawn(move || {
                    Self::worker_loop(
                        worker_id,
                        storage,
                        registry,
                        nav_cache,
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
            registry,
            nav_cache,
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

        // Call test shutdown hook if configured
        #[cfg(any(test, feature = "test-hooks"))]
        if let Some(hook) = &self.config.shutdown_hook {
            hook();
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

    /// Get the current pending queue size across all embeddings.
    ///
    /// This scans the Pending CF to count items. For high-frequency monitoring,
    /// consider using `items_processed()` and `batches_processed()` instead.
    pub fn pending_queue_size(&self) -> usize {
        Self::count_pending_items_static(&self.storage)
    }

    /// Check if backpressure should be applied.
    ///
    /// Returns true if pending queue exceeds `backpressure_threshold` and
    /// backpressure is enabled (threshold > 0).
    pub fn should_apply_backpressure(&self) -> bool {
        if self.config.backpressure_threshold == 0 {
            return false;
        }
        self.pending_queue_size() >= self.config.backpressure_threshold
    }

    /// Count pending items across all embeddings (static version).
    fn count_pending_items_static(storage: &Storage) -> usize {
        let txn_db = match storage.transaction_db() {
            Ok(db) => db,
            Err(_) => return 0,
        };

        let pending_cf = match txn_db.cf_handle(Pending::CF_NAME) {
            Some(cf) => cf,
            None => return 0,
        };

        let iter = txn_db.iterator_cf(&pending_cf, rocksdb::IteratorMode::Start);
        iter.count()
    }

    // ========================================================================
    // Worker Implementation
    // ========================================================================

    fn worker_loop(
        worker_id: usize,
        storage: Arc<Storage>,
        registry: Arc<EmbeddingRegistry>,
        nav_cache: Arc<NavigationCache>,
        config: AsyncUpdaterConfig,
        shutdown: Arc<AtomicBool>,
        embedding_counter: Arc<AtomicU64>,
        items_processed: Arc<AtomicU64>,
        batches_processed: Arc<AtomicU64>,
    ) {
        info!(worker_id, "Async updater worker started");

        // Metrics logging state (only worker 0 logs to avoid duplicates)
        let mut last_metrics_log = Instant::now();
        let mut last_items_processed = 0u64;

        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            // Periodic metrics logging (worker 0 only)
            if worker_id == 0 {
                if let Some(interval) = config.metrics_interval {
                    if last_metrics_log.elapsed() >= interval {
                        let current_items = items_processed.load(Ordering::Relaxed);
                        let current_batches = batches_processed.load(Ordering::Relaxed);
                        let pending_size = Self::count_pending_items_static(&storage);
                        let drain_rate = (current_items - last_items_processed) as f64
                            / last_metrics_log.elapsed().as_secs_f64();

                        info!(
                            pending_queue_size = pending_size,
                            items_processed = current_items,
                            batches_processed = current_batches,
                            drain_rate_per_sec = format!("{:.1}", drain_rate),
                            backpressure_threshold = config.backpressure_threshold,
                            "Async updater metrics"
                        );

                        last_metrics_log = Instant::now();
                        last_items_processed = current_items;
                    }
                }
            }

            // Collect a batch of pending inserts using round-robin across embeddings
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
                // process_insert handles idempotency (checks FLAG_PENDING/FLAG_DELETED)
                // and deletes from pending CF within the same transaction
                if let Err(e) = Self::process_insert(
                    &storage,
                    &registry,
                    &nav_cache,
                    &config,
                    key,
                    *embedding,
                    *vec_id,
                ) {
                    warn!(
                        worker_id,
                        embedding,
                        vec_id,
                        error = %e,
                        "Failed to process pending insert (will retry)"
                    );
                    // Transaction rolled back - item stays in pending for retry
                    continue;
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
    ///
    /// The processing is done in a single transaction for atomicity:
    /// 1. Check if vector still exists and is pending (not deleted)
    /// 2. Load vector data from Vectors CF
    /// 3. Build HNSW graph edges
    /// 4. Update VecMeta to clear FLAG_PENDING
    /// 5. Delete from Pending CF
    /// 6. Commit (all operations atomic)
    ///
    /// This ensures no crash window between graph update and pending deletion.
    fn process_insert(
        storage: &Storage,
        registry: &EmbeddingRegistry,
        nav_cache: &Arc<NavigationCache>,
        _config: &AsyncUpdaterConfig,
        pending_key: &[u8],
        embedding: EmbeddingCode,
        vec_id: VecId,
    ) -> anyhow::Result<()> {
        let txn_db = storage.transaction_db()?;
        let txn = txn_db.transaction();

        // 1. Check if vector still exists and is pending
        let vec_meta_cf = txn_db
            .cf_handle(VecMeta::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("VecMeta CF not found"))?;
        let meta_key = VecMetaCfKey(embedding, vec_id);
        let meta_bytes = match txn.get_cf(&vec_meta_cf, VecMeta::key_to_bytes(&meta_key))? {
            Some(bytes) => bytes,
            None => {
                // Vector metadata doesn't exist - skip (may have been deleted)
                // Still delete from pending to clean up
                let pending_cf = txn_db
                    .cf_handle(Pending::CF_NAME)
                    .ok_or_else(|| anyhow::anyhow!("Pending CF not found"))?;
                txn.delete_cf(&pending_cf, pending_key)?;
                txn.commit()?;
                debug!(embedding, vec_id, "Vector metadata not found, skipping");
                return Ok(());
            }
        };

        let meta = VecMeta::value_from_bytes(&meta_bytes)?;
        if !meta.0.is_pending() {
            // Already processed - idempotent skip
            // Still delete from pending to clean up
            let pending_cf = txn_db
                .cf_handle(Pending::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("Pending CF not found"))?;
            txn.delete_cf(&pending_cf, pending_key)?;
            txn.commit()?;
            debug!(embedding, vec_id, "Vector already processed, skipping");
            return Ok(());
        }
        if meta.0.is_deleted() {
            // Deleted - skip
            // Still delete from pending to clean up
            let pending_cf = txn_db
                .cf_handle(Pending::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("Pending CF not found"))?;
            txn.delete_cf(&pending_cf, pending_key)?;
            txn.commit()?;
            debug!(embedding, vec_id, "Vector deleted, skipping");
            return Ok(());
        }

        // 2. Load vector data
        let vectors_cf = txn_db
            .cf_handle(Vectors::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;
        let vec_key = VectorCfKey(embedding, vec_id);
        let vec_bytes = txn
            .get_cf(&vectors_cf, Vectors::key_to_bytes(&vec_key))?
            .ok_or_else(|| anyhow::anyhow!("Vector data not found for {:?}", vec_id))?;

        // Get embedding spec from registry for runtime info (storage_type, distance)
        let emb = registry
            .get_by_code(embedding)
            .ok_or_else(|| anyhow::anyhow!("Unknown embedding code: {}", embedding))?;
        let storage_type = emb.storage_type();
        let vector = Vectors::value_from_bytes_typed(&vec_bytes, storage_type)?;

        // 3. Build HNSW index
        // Get EmbeddingSpec from storage (single source of truth for HNSW params)
        let specs_cf = txn_db
            .cf_handle(EmbeddingSpecs::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("EmbeddingSpecs CF not found"))?;
        let spec_key = EmbeddingSpecCfKey(embedding);
        let spec_bytes = txn
            .get_cf(&specs_cf, EmbeddingSpecs::key_to_bytes(&spec_key))?
            .ok_or_else(|| anyhow::anyhow!("EmbeddingSpec not found for {}", embedding))?;
        let embedding_spec = EmbeddingSpecs::value_from_bytes(&spec_bytes)?.0;

        // Create HNSW index using EmbeddingSpec as single source of truth
        // All structural params (m, m_max, ef_construction) derived from spec
        let index = hnsw::Index::from_spec(
            embedding,
            &embedding_spec,
            64, // batch_threshold: runtime knob, not persisted
            Arc::clone(nav_cache),
        );

        // Call hnsw::insert to build edges
        let cache_update = hnsw::insert(&index, &txn, &txn_db, storage, vec_id, &vector)?;

        // Note: hnsw::insert already updates VecMeta with max_layer and flags=0
        // But we need to make sure FLAG_PENDING is cleared. The hnsw::insert
        // creates new VecMeta with flags=0, so FLAG_PENDING will be cleared.

        // 4. Delete from pending CF (inside transaction for atomicity)
        let pending_cf = txn_db
            .cf_handle(Pending::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Pending CF not found"))?;
        txn.delete_cf(&pending_cf, pending_key)?;

        // 5. Update lifecycle counters: pending -> indexed
        let lifecycle_cf = txn_db
            .cf_handle(LifecycleCounts::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("LifecycleCounts CF not found"))?;
        let lifecycle_key = LifecycleCountsCfKey(embedding);
        let delta = LifecycleCountsDelta::pending_to_indexed();
        txn.merge_cf(
            &lifecycle_cf,
            LifecycleCounts::key_to_bytes(&lifecycle_key),
            delta.to_bytes(),
        )?;

        // 6. Commit transaction (all operations atomic)
        txn.commit()?;

        // 6. Apply cache update
        cache_update.apply(nav_cache);

        debug!(embedding, vec_id, "Processed pending insert");
        Ok(())
    }

    /// Drain all pending items (used for startup recovery).
    fn drain_pending_static(
        storage: &Storage,
        registry: &EmbeddingRegistry,
        nav_cache: &Arc<NavigationCache>,
        config: &AsyncUpdaterConfig,
    ) -> anyhow::Result<()> {
        let embedding_counter = AtomicU64::new(0);

        loop {
            let batch = Self::collect_batch(storage, config, &embedding_counter)?;
            if batch.is_empty() {
                break;
            }

            info!(batch_size = batch.len(), "Draining pending batch");

            for (key, embedding, vec_id) in &batch {
                // process_insert handles both graph building and pending deletion atomically
                if let Err(e) =
                    Self::process_insert(storage, registry, nav_cache, config, key, *embedding, *vec_id)
                {
                    warn!(embedding, vec_id, error = %e, "Failed to process during drain");
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
    use crate::vector::embedding::EmbeddingBuilder;
    use crate::vector::processor::Processor;
    use crate::vector::schema::ExternalKey;
    use crate::vector::search::SearchConfig;
    use crate::vector::Distance;
    use tempfile::TempDir;

    // =========================================================================
    // Config Tests
    // =========================================================================

    #[test]
    fn test_config_defaults() {
        let config = AsyncUpdaterConfig::default();
        assert_eq!(config.batch_size, 100);
        assert_eq!(config.batch_timeout, Duration::from_millis(100));
        assert_eq!(config.num_workers, 2);
        assert!(config.process_on_startup);
        assert_eq!(config.idle_sleep, Duration::from_millis(10));
        // Task 7.8: Backpressure and metrics
        assert_eq!(config.backpressure_threshold, 10000);
        assert_eq!(config.metrics_interval, Some(Duration::from_secs(10)));
    }

    #[test]
    fn test_config_builder() {
        let config = AsyncUpdaterConfig::new()
            .with_batch_size(200)
            .with_batch_timeout(Duration::from_millis(50))
            .with_num_workers(4)
            .with_process_on_startup(false)
            .with_idle_sleep(Duration::from_millis(5))
            .with_backpressure_threshold(5000)
            .with_metrics_interval(Some(Duration::from_secs(30)));

        assert_eq!(config.batch_size, 200);
        assert_eq!(config.batch_timeout, Duration::from_millis(50));
        assert_eq!(config.num_workers, 4);
        assert!(!config.process_on_startup);
        assert_eq!(config.idle_sleep, Duration::from_millis(5));
        // Task 7.8: Backpressure and metrics builder
        assert_eq!(config.backpressure_threshold, 5000);
        assert_eq!(config.metrics_interval, Some(Duration::from_secs(30)));
    }

    #[test]
    fn test_config_disable_backpressure_and_metrics() {
        let config = AsyncUpdaterConfig::new()
            .no_backpressure()
            .no_metrics_logging();

        assert_eq!(config.backpressure_threshold, 0);
        assert_eq!(config.metrics_interval, None);
    }

    // =========================================================================
    // Helper Functions
    // =========================================================================

    /// Create test storage with registry and processor
    fn setup_test_env(
        temp_dir: &TempDir,
    ) -> (Arc<Storage>, Arc<EmbeddingRegistry>, Arc<NavigationCache>) {
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().expect("Failed to initialize storage");
        let storage = Arc::new(storage);

        let registry = Arc::new(EmbeddingRegistry::new(storage.clone()));
        let nav_cache = Arc::new(NavigationCache::new());

        (storage, registry, nav_cache)
    }

    /// Register a test embedding
    fn register_embedding(
        registry: &EmbeddingRegistry,
        dim: u32,
    ) -> crate::vector::embedding::Embedding {
        let builder = EmbeddingBuilder::new("test-model", dim, Distance::L2);
        registry
            .register(builder)
            .expect("Failed to register embedding")
    }

    /// Generate a deterministic test vector
    fn test_vector(dim: usize, seed: u64) -> Vec<f32> {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha20Rng;
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..dim).map(|_| rng.gen::<f32>()).collect()
    }

    // =========================================================================
    // Phase 7.6 Tests: Async Graph Updater
    // =========================================================================

    /// 7.6.1: Vectors are searchable immediately via pending fallback
    #[test]
    fn test_insert_async_immediate_searchability() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let (storage, registry, nav_cache) = setup_test_env(&temp_dir);

        // Register embedding
        let embedding = register_embedding(&registry, 64);

        // Create processor (default config)
        let processor = Processor::new(storage.clone(), registry.clone());

        // Insert vector with build_index=false (async path)
        let id = crate::Id::new();
        let vector = test_vector(64, 42);
        let _vec_id = processor
            .insert_vector(&embedding, ExternalKey::NodeId(id), &vector, false)
            .expect("Insert should succeed");

        // Search with pending fallback enabled (default)
        let config = SearchConfig::new(embedding.clone(), 10);
        let results = processor
            .search_with_config(&config, &vector)
            .expect("Search should succeed");

        // Vector should be found via pending fallback
        assert_eq!(results.len(), 1, "Should find the pending vector");
        assert_eq!(
            results[0].node_id().expect("expected NodeId"),
            id,
            "Should find the correct vector"
        );
    }

    /// 7.6.2: Workers drain pending queue
    #[test]
    fn test_pending_queue_drains() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let (storage, registry, nav_cache) = setup_test_env(&temp_dir);

        // Register embedding
        let embedding = register_embedding(&registry, 64);

        // Create processor
        let processor = Processor::new(storage.clone(), registry.clone());

        // Insert multiple vectors with build_index=false
        let mut ids = Vec::new();
        let mut first = None;
        for i in 0..5 {
            let id = crate::Id::new();
            let vector = test_vector(64, i);
            processor
                .insert_vector(&embedding, ExternalKey::NodeId(id), &vector, false)
                .expect("Insert should succeed");
            if first.is_none() {
                first = Some((id, vector));
            }
            ids.push(id);
        }

        // Verify pending queue has items
        let pending_count = count_pending_items(&storage, embedding.code());
        assert_eq!(pending_count, 5, "Should have 5 pending items");

        // Use drain_pending_static to process all
        let config = AsyncUpdaterConfig::new().with_process_on_startup(true);
        AsyncGraphUpdater::drain_pending_static(&storage, &registry, &nav_cache, &config)
            .expect("Drain should succeed");

        // Verify pending queue is empty
        let pending_count = count_pending_items(&storage, embedding.code());
        assert_eq!(pending_count, 0, "Pending queue should be drained");

        // Verify vectors are indexed/searchable after draining
        let (first_id, first_vector) = first.expect("Expected at least one vector");
        let search_config = SearchConfig::new(embedding.clone(), 5);
        let results = processor
            .search_with_config(&search_config, &first_vector)
            .expect("Search should succeed");
        assert!(
            results
                .iter()
                .any(|result| result.node_id() == Some(first_id)),
            "Drained vector should be searchable"
        );
    }

    /// 7.6.3: Pending items survive restart (crash recovery)
    #[test]
    fn test_pending_queue_crash_recovery() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let embedding_code: EmbeddingCode;
        let inserted_id: crate::Id;

        // Phase 1: Insert vector, then "crash" (drop storage without draining)
        {
            let (storage, registry, _nav_cache) = setup_test_env(&temp_dir);
            let embedding = register_embedding(&registry, 64);
            embedding_code = embedding.code();

            let processor = Processor::new(storage.clone(), registry.clone());
            inserted_id = crate::Id::new();
            let vector = test_vector(64, 42);
            processor
                .insert_vector(&embedding, ExternalKey::NodeId(inserted_id), &vector, false)
                .expect("Insert should succeed");

            // Verify pending item exists
            let pending_count = count_pending_items(&storage, embedding_code);
            assert_eq!(pending_count, 1, "Should have 1 pending item before crash");

            // Storage dropped here (simulating crash)
        }

        // Phase 2: Reopen storage and verify pending item persists
        {
            let (storage, registry, nav_cache) = setup_test_env(&temp_dir);

            // Re-load embeddings from storage (needed after restart)
            registry.prewarm().expect("Failed to prewarm registry");
            let embedding = registry
                .get_by_code(embedding_code)
                .expect("Embedding should exist after restart");
            let processor = Processor::new(storage.clone(), registry.clone());

            // Verify pending item survived restart
            let pending_count = count_pending_items(&storage, embedding_code);
            assert_eq!(pending_count, 1, "Pending item should survive restart");

            // Drain and verify processing works
            let config = AsyncUpdaterConfig::new();
            AsyncGraphUpdater::drain_pending_static(&storage, &registry, &nav_cache, &config)
                .expect("Drain after restart should succeed");

            let pending_count = count_pending_items(&storage, embedding_code);
            assert_eq!(pending_count, 0, "Pending queue should be drained after recovery");

            // Verify vector is indexed/searchable after recovery
            let query = test_vector(64, 42);
            let search_config = SearchConfig::new(embedding, 5);
            let results = processor
                .search_with_config(&search_config, &query)
                .expect("Search should succeed");
            assert!(
                results
                    .iter()
                    .any(|result| result.node_id() == Some(inserted_id)),
                "Recovered vector should be searchable"
            );
        }
    }

    /// 7.6.4: Delete removes vector from pending queue
    #[test]
    fn test_delete_removes_from_pending() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let (storage, registry, _nav_cache) = setup_test_env(&temp_dir);

        // Register embedding
        let embedding = register_embedding(&registry, 64);

        // Create processor
        let processor = Processor::new(storage.clone(), registry.clone());

        // Insert vector with build_index=false
        let id = crate::Id::new();
        let vector = test_vector(64, 42);
        processor
            .insert_vector(&embedding, ExternalKey::NodeId(id), &vector, false)
            .expect("Insert should succeed");

        // Verify pending item exists
        let pending_count = count_pending_items(&storage, embedding.code());
        assert_eq!(pending_count, 1, "Should have 1 pending item");

        // Delete the vector
        processor
            .delete_vector(&embedding, ExternalKey::NodeId(id))
            .expect("Delete should succeed");

        // Verify pending queue is cleared
        let pending_count = count_pending_items(&storage, embedding.code());
        assert_eq!(pending_count, 0, "Pending item should be removed after delete");

        // Search should not find the deleted vector
        let config = SearchConfig::new(embedding.clone(), 10);
        let results = processor
            .search_with_config(&config, &vector)
            .expect("Search should succeed");
        assert!(results.is_empty(), "Deleted vector should not be found");
    }

    /// 7.6.5: Concurrent insert and worker don't race
    #[test]
    fn test_concurrent_insert_and_worker() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let (storage, registry, nav_cache) = setup_test_env(&temp_dir);

        // Register embedding
        let embedding = register_embedding(&registry, 64);

        // Create processor
        let processor = Arc::new(Processor::new(storage.clone(), registry.clone()));

        // Start async updater with fast processing
        let config = AsyncUpdaterConfig::new()
            .with_num_workers(2)
            .with_batch_size(10)
            .with_idle_sleep(Duration::from_millis(1));
        let updater =
            AsyncGraphUpdater::start(storage.clone(), registry.clone(), nav_cache.clone(), config);

        // Insert vectors concurrently while workers are running
        let processor_clone = processor.clone();
        let embedding_clone = embedding.clone();
        let insert_handle = std::thread::spawn(move || {
            for i in 0..50 {
                let id = crate::Id::new();
                let vector = test_vector(64, i);
                processor_clone
                    .insert_vector(&embedding_clone, ExternalKey::NodeId(id), &vector, false)
                    .expect("Insert should succeed");
            }
        });

        // Wait for inserts to complete
        insert_handle.join().expect("Insert thread should complete");

        // Give workers time to process
        std::thread::sleep(Duration::from_millis(200));

        // Get metrics before shutdown (shutdown consumes self)
        let items_processed = updater.items_processed();

        // Shutdown
        updater.shutdown();

        // Verify no panic occurred and items were processed
        // (exact count depends on timing, but should be > 0)
        let remaining = count_pending_items(&storage, embedding.code());
        assert!(
            items_processed > 0 || remaining < 50,
            "Workers should have processed some items (processed={}, remaining={})",
            items_processed,
            remaining
        );
    }

    /// 7.6.6: Graceful shutdown completes in-flight batch
    #[test]
    fn test_shutdown_completes_gracefully() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let (storage, registry, nav_cache) = setup_test_env(&temp_dir);

        // Register embedding
        let embedding = register_embedding(&registry, 64);

        // Create processor and insert some vectors
        let processor = Processor::new(storage.clone(), registry.clone());
        for i in 0..10 {
            let id = crate::Id::new();
            let vector = test_vector(64, i);
            processor
                .insert_vector(&embedding, ExternalKey::NodeId(id), &vector, false)
                .expect("Insert should succeed");
        }
        let pending_initial = count_pending_items(&storage, embedding.code());
        assert_eq!(pending_initial, 10, "Expected pending backlog before start");

        // Start async updater
        let config = AsyncUpdaterConfig::new()
            .with_num_workers(1)
            .with_batch_size(5)
            .with_process_on_startup(false);
        let updater =
            AsyncGraphUpdater::start(storage.clone(), registry.clone(), nav_cache.clone(), config);

        // Give workers time to pick up work
        std::thread::sleep(Duration::from_millis(100));

        let mut processed = 0u64;
        for _ in 0..200 {
            processed = updater.items_processed();
            if processed >= 5 {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            processed >= 5,
            "Expected at least one full batch to process before shutdown"
        );

        // Shutdown should complete gracefully
        updater.shutdown();

        // Verify shutdown waited for in-flight work (pending should not exceed pre-shutdown backlog)
        let pending_after = count_pending_items(&storage, embedding.code());
        assert!(
            pending_after <= pending_initial.saturating_sub(processed as usize),
            "Pending queue should not grow after shutdown (initial={}, processed={}, after={})",
            pending_initial,
            processed,
            pending_after
        );
        assert!(
            pending_after < pending_initial,
            "Shutdown should allow in-flight batch completion"
        );
    }

    /// 7.6.7: Partial batch processing is idempotent
    #[test]
    fn test_partial_batch_idempotent() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let (storage, registry, nav_cache) = setup_test_env(&temp_dir);

        // Register embedding
        let embedding = register_embedding(&registry, 64);

        // Create processor and insert vectors
        let processor = Processor::new(storage.clone(), registry.clone());
        let mut ids = Vec::new();
        for i in 0..5 {
            let id = crate::Id::new();
            let vector = test_vector(64, i);
            processor
                .insert_vector(&embedding, ExternalKey::NodeId(id), &vector, false)
                .expect("Insert should succeed");
            ids.push(id);
        }

        // Process first time
        let config = AsyncUpdaterConfig::new();
        AsyncGraphUpdater::drain_pending_static(&storage, &registry, &nav_cache, &config)
            .expect("First drain should succeed");

        // Process again (idempotent - should be no-op)
        AsyncGraphUpdater::drain_pending_static(&storage, &registry, &nav_cache, &config)
            .expect("Second drain should succeed (idempotent)");

        // Search should still work correctly
        let search_config = SearchConfig::new(embedding.clone(), 10);
        let query = test_vector(64, 0);
        let results = processor
            .search_with_config(&search_config, &query)
            .expect("Search should succeed");

        // Should find vectors (now indexed, not pending)
        assert!(!results.is_empty(), "Should find indexed vectors");
    }

    /// 7.6.8: Pending scan respects limit (backlog bound)
    #[test]
    fn test_pending_scan_respects_limit() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let (storage, registry, _nav_cache) = setup_test_env(&temp_dir);

        // Register embedding
        let embedding = register_embedding(&registry, 64);

        // Create processor and insert many vectors (more than default limit)
        let processor = Processor::new(storage.clone(), registry.clone());
        for i in 0..1500 {
            let id = crate::Id::new();
            let vector = test_vector(64, i);
            processor
                .insert_vector(&embedding, ExternalKey::NodeId(id), &vector, false)
                .expect("Insert should succeed");
        }

        // Search with tiny pending scan limit
        let config = SearchConfig::new(embedding.clone(), 10).with_pending_scan_limit(1);
        let query = test_vector(64, 0);
        let results = processor
            .search_with_config(&config, &query)
            .expect("Search should succeed");

        // Should only return from the single scanned pending vector
        assert_eq!(results.len(), 1, "Should respect pending scan limit");

        // Search with no pending fallback
        let config_no_pending = SearchConfig::new(embedding.clone(), 10).no_pending_fallback();
        let results_no_pending = processor
            .search_with_config(&config_no_pending, &query)
            .expect("Search should succeed");

        // Should find nothing (no indexed vectors, no pending fallback)
        assert!(
            results_no_pending.is_empty(),
            "Should find nothing with pending fallback disabled"
        );
    }

    /// 7.8.2: Async insert backpressure blocks when threshold exceeded
    #[test]
    fn test_async_insert_backpressure_blocks() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let (storage, registry, _nav_cache) = setup_test_env(&temp_dir);

        // Register embedding
        let embedding = register_embedding(&registry, 64);

        // Create processor with backpressure threshold of 1
        let processor = Processor::new(storage.clone(), registry.clone());
        processor.set_async_backpressure_threshold(1);

        // First async insert should succeed
        let id1 = crate::Id::new();
        let vector1 = test_vector(64, 1);
        processor
            .insert_vector(&embedding, ExternalKey::NodeId(id1), &vector1, false)
            .expect("First async insert should succeed");

        // Second async insert should be blocked by backpressure
        let id2 = crate::Id::new();
        let vector2 = test_vector(64, 2);
        let err = processor
            .insert_vector(&embedding, ExternalKey::NodeId(id2), &vector2, false)
            .expect_err("Second async insert should fail due to backpressure");
        assert!(
            err.to_string().to_lowercase().contains("backpressure"),
            "Expected backpressure error, got: {}",
            err
        );
    }

    // =========================================================================
    // Helper: Count pending items for an embedding
    // =========================================================================

    fn count_pending_items(storage: &Storage, embedding: EmbeddingCode) -> usize {
        let txn_db = storage.transaction_db().expect("Failed to get txn_db");
        let pending_cf = txn_db
            .cf_handle(Pending::CF_NAME)
            .expect("Pending CF not found");

        let prefix = Pending::prefix_for_embedding(embedding);
        let iter = txn_db.iterator_cf(
            &pending_cf,
            rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
        );

        let mut count = 0;
        for item in iter {
            let (key, _) = item.expect("Iterator error");
            if key.len() < 8 || key[0..8] != prefix {
                break;
            }
            count += 1;
        }
        count
    }

    /// Latency comparison: sync vs async insert
    ///
    /// This test measures the speedup from using async inserts (build_index=false)
    /// vs sync inserts (build_index=true). Run with --nocapture to see results:
    ///
    /// ```sh
    /// cargo test -p motlie-db --lib test_sync_vs_async_latency -- --nocapture
    /// ```
    #[test]
    fn test_sync_vs_async_latency() {
        use std::time::Instant;

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let (storage, registry, _nav_cache) = setup_test_env(&temp_dir);

        // Register embedding (64 dimensions for faster test)
        let embedding = register_embedding(&registry, 64);
        let processor = Processor::new(storage.clone(), registry.clone());

        const NUM_VECTORS: usize = 100;

        // Measure SYNC insert latency (build_index=true)
        let mut sync_latencies = Vec::with_capacity(NUM_VECTORS);
        for i in 0..NUM_VECTORS {
            let id = crate::Id::new();
            let vector = test_vector(64, i as u64 + 1000);

            let start = Instant::now();
            processor
                .insert_vector(&embedding, ExternalKey::NodeId(id), &vector, true) // SYNC
                .expect("Sync insert should succeed");
            sync_latencies.push(start.elapsed());
        }

        // Measure ASYNC insert latency (build_index=false)
        let mut async_latencies = Vec::with_capacity(NUM_VECTORS);
        for i in 0..NUM_VECTORS {
            let id = crate::Id::new();
            let vector = test_vector(64, i as u64 + 2000);

            let start = Instant::now();
            processor
                .insert_vector(&embedding, ExternalKey::NodeId(id), &vector, false) // ASYNC
                .expect("Async insert should succeed");
            async_latencies.push(start.elapsed());
        }

        // Calculate percentiles
        sync_latencies.sort();
        async_latencies.sort();

        let sync_p50 = sync_latencies[NUM_VECTORS / 2];
        let sync_p99 = sync_latencies[NUM_VECTORS * 99 / 100];
        let async_p50 = async_latencies[NUM_VECTORS / 2];
        let async_p99 = async_latencies[NUM_VECTORS * 99 / 100];

        let speedup_p50 = sync_p50.as_nanos() as f64 / async_p50.as_nanos().max(1) as f64;
        let speedup_p99 = sync_p99.as_nanos() as f64 / async_p99.as_nanos().max(1) as f64;

        println!("\n=== Sync vs Async Insert Latency ===");
        println!("Vectors: {}", NUM_VECTORS);
        println!("\nSync Insert (build_index=true):");
        println!("  P50: {:?}", sync_p50);
        println!("  P99: {:?}", sync_p99);
        println!("\nAsync Insert (build_index=false):");
        println!("  P50: {:?}", async_p50);
        println!("  P99: {:?}", async_p99);
        println!("\nSpeedup:");
        println!("  P50: {:.1}x faster", speedup_p50);
        println!("  P99: {:.1}x faster", speedup_p99);

        // Assert async is meaningfully faster (at least 2x at P50)
        assert!(
            speedup_p50 > 2.0,
            "Async should be at least 2x faster at P50 (got {:.1}x)",
            speedup_p50
        );
    }
}
