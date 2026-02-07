//! Garbage Collection for CONTENT-ADDRESS reverse index system.
//!
//! This module provides incremental GC for:
//! - Stale index entries (NodeSummaryIndex, EdgeSummaryIndex)
//! - Tombstoned entities after retention period (future)
//! - Orphan summaries via OrphanSummaries CF (VERSIONING)
//!
//! # VERSIONING: OrphanSummaries-Based GC (Updated 2026-02-07)
//!
//! The VERSIONING design uses **deferred deletion** for summaries to enable rollback:
//! - Summary rows are append-only (no RefCount decrement deletes inline)
//! - When a summary is no longer referenced, it's added to OrphanSummaries CF
//! - GC scans OrphanSummaries CF and deletes entries older than retention period
//! - Rollback operations can resurrect orphaned summaries within the retention window
//!
//! See `libs/db/src/graph/docs/VERSIONING.md` for full design details.
// (claude, 2026-02-07, FIXED: Updated header to reflect VERSIONING OrphanSummaries GC plan - Codex Item 14)
// (codex, 2026-02-07, decision: header reflects intended design, but OrphanSummaries tracking/worker is not implemented in this module yet.)
//!
//! # What GC Still Handles
//!
//! GC cleans up STALE index entries. When a node/edge is updated:
//! 1. The old index entry is marked STALE (not deleted)
//! 2. A new CURRENT index entry is created
//! 3. GC later deletes old STALE entries based on version retention policy
//!
//! # Design
//!
//! GC uses cursor-based incremental processing:
//! 1. Each cycle processes `batch_size` entries from cursor position
//! 2. Cursor is persisted to GraphMeta CF for crash recovery
//! 3. When cursor reaches end, it resets to start fresh
//!
//! # Usage
//!
//! ```rust,ignore
//! use motlie_db::graph::{Storage, GraphGcConfig, GraphGarbageCollector};
//!
//! let config = GraphGcConfig::default();
//! let gc = GraphGarbageCollector::new(storage, config);
//!
//! // Run a single GC cycle
//! let metrics = gc.run_cycle()?;
//! println!("Deleted {} stale index entries", metrics.node_index_entries_deleted);
//!
//! // Or start background worker
//! let handle = gc.spawn_worker();
//! // ... later ...
//! gc.shutdown();
//! handle.await?;
//! ```

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;

use super::name_hash::NameHash;
use super::schema::{
    GraphMeta, GraphMetaCfKey, GraphMetaCfValue, GraphMetaField,
    NodeSummaryIndex, EdgeSummaryIndex,
    Nodes, NodeCfValue,
    ForwardEdges,
    Version,
};
use super::{ColumnFamily, ColumnFamilySerde, HotColumnFamilyRecord, Storage};
use crate::Id;

// ============================================================================
// VERSIONING Helpers for GC
// ============================================================================
/// (claude, 2026-02-06, in-progress: VERSIONING prefix scan helper for GC)

/// Find the current version of a node via prefix scan (within transaction).
/// Returns the current version number and deleted flag.
fn find_current_node_version_for_gc(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    nodes_cf: &impl rocksdb::AsColumnFamilyRef,
    node_id: Id,
) -> Result<Option<(Version, bool)>> {
    // Prefix scan on node_id (first 16 bytes)
    let prefix = node_id.into_bytes().to_vec();
    let iter = txn.prefix_iterator_cf(nodes_cf, &prefix);

    for item in iter {
        let (key_bytes, value_bytes) = item?;
        // Stop if we've gone past our prefix
        if !key_bytes.starts_with(&prefix) {
            break;
        }
        let value: NodeCfValue = Nodes::value_from_bytes(&value_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize node value: {}", e))?;
        // Current version has ValidUntil = None (field 0)
        // Field indices (VERSIONING): 0=ValidUntil, 1=ActivePeriod, 2=NameHash, 3=SummaryHash, 4=Version, 5=Deleted
        if value.0.is_none() {
            return Ok(Some((value.4, value.5)));
        }
    }
    Ok(None)
}

/// Find the current version of an edge via prefix scan (within transaction).
/// (claude, 2026-02-06, in-progress: VERSIONING prefix scan helper for GC)
/// Returns the current version number and deleted flag.
fn find_current_edge_version_for_gc(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    edges_cf: &impl rocksdb::AsColumnFamilyRef,
    src_id: Id,
    dst_id: Id,
    name_hash: NameHash,
) -> Result<Option<(Version, bool)>> {
    use super::schema::{ForwardEdgeCfValue, ForwardEdges};

    // Prefix scan on (src_id, dst_id, name_hash) = 40 bytes
    let mut prefix = Vec::with_capacity(40);
    prefix.extend_from_slice(&src_id.into_bytes());
    prefix.extend_from_slice(&dst_id.into_bytes());
    prefix.extend_from_slice(name_hash.as_bytes());
    let iter = txn.prefix_iterator_cf(edges_cf, &prefix);

    for item in iter {
        let (key_bytes, value_bytes) = item?;
        // Stop if we've gone past our prefix
        if !key_bytes.starts_with(&prefix) {
            break;
        }
        let value: ForwardEdgeCfValue = ForwardEdges::value_from_bytes(&value_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize edge value: {}", e))?;
        // Current version has ValidUntil = None (field 0)
        // Field indices (VERSIONING): 0=ValidUntil, 1=ActivePeriod, 2=Weight, 3=SummaryHash, 4=Version, 5=Deleted
        if value.0.is_none() {
            return Ok(Some((value.4, value.5)));
        }
    }
    Ok(None)
}

/// Check if any CURRENT entry in NodeSummaryIndex references this summary hash.
/// (claude, 2026-02-07, FIXED: Added reference check for shared summaries per Codex review)
///
/// NodeSummaryIndex key is (SummaryHash, NodeId, Version), so we prefix-scan on hash.
fn has_current_node_summary_reference(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    index_cf: &impl rocksdb::AsColumnFamilyRef,
    hash: super::summary_hash::SummaryHash,
) -> Result<bool> {
    use super::schema::{NodeSummaryIndex, NodeSummaryIndexCfValue};

    // Prefix scan on SummaryHash (first 32 bytes)
    let prefix = hash.as_bytes().to_vec();
    let iter = txn.prefix_iterator_cf(index_cf, &prefix);

    for item in iter {
        let (key_bytes, value_bytes) = item?;
        if !key_bytes.starts_with(&prefix) {
            break;
        }
        let value: NodeSummaryIndexCfValue = NodeSummaryIndex::value_from_bytes(&value_bytes)?;
        if value.is_current() {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Check if any CURRENT entry in EdgeSummaryIndex references this summary hash.
/// (claude, 2026-02-07, FIXED: Added reference check for shared summaries per Codex review)
///
/// EdgeSummaryIndex key is (SummaryHash, SrcId, DstId, NameHash, Version), so we prefix-scan on hash.
fn has_current_edge_summary_reference(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    index_cf: &impl rocksdb::AsColumnFamilyRef,
    hash: super::summary_hash::SummaryHash,
) -> Result<bool> {
    use super::schema::{EdgeSummaryIndex, EdgeSummaryIndexCfValue};

    // Prefix scan on SummaryHash (first 32 bytes)
    let prefix = hash.as_bytes().to_vec();
    let iter = txn.prefix_iterator_cf(index_cf, &prefix);

    for item in iter {
        let (key_bytes, value_bytes) = item?;
        if !key_bytes.starts_with(&prefix) {
            break;
        }
        let value: EdgeSummaryIndexCfValue = EdgeSummaryIndex::value_from_bytes(&value_bytes)?;
        if value.is_current() {
            return Ok(true);
        }
    }
    Ok(false)
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the graph garbage collector.
#[derive(Debug, Clone)]
pub struct GraphGcConfig {
    /// Interval between GC cycles.
    /// Default: 60 seconds
    pub interval: Duration,

    /// Maximum entries to process per cycle.
    /// Bounds latency by limiting work per cycle.
    /// Default: 1000
    pub batch_size: usize,

    /// Number of summary versions to retain per entity.
    /// Older versions are eligible for deletion.
    /// Default: 2 (keep current and one previous)
    pub versions_to_keep: usize,

    /// Tombstone retention period before hard delete.
    /// Entities marked deleted are kept for audit/time-travel until this period.
    /// Default: 7 days
    pub tombstone_retention: Duration,

    /// Orphan summary retention period (VERSIONING).
    /// Summaries in OrphanSummaries CF older than this are eligible for deletion.
    /// This enables time-travel/rollback within the retention window.
    /// Default: 7 days
    /// (claude, 2026-02-07, FIXED: Added orphan_retention config per VERSIONING)
    pub orphan_retention: Duration,

    /// Run a GC cycle immediately on startup.
    /// Default: true
    pub process_on_startup: bool,
}

impl Default for GraphGcConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(60),
            batch_size: 1000,
            versions_to_keep: 2,
            tombstone_retention: Duration::from_secs(7 * 24 * 60 * 60), // 7 days
            orphan_retention: Duration::from_secs(7 * 24 * 60 * 60), // 7 days (VERSIONING)
            process_on_startup: true,
        }
    }
}

impl GraphGcConfig {
    /// Create a new config with custom interval.
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.interval = interval;
        self
    }

    /// Create a new config with custom batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Create a new config with custom versions to keep.
    pub fn with_versions_to_keep(mut self, versions: usize) -> Self {
        self.versions_to_keep = versions.max(1); // At least keep current version
        self
    }

    /// Create a new config with custom tombstone retention.
    pub fn with_tombstone_retention(mut self, retention: Duration) -> Self {
        self.tombstone_retention = retention;
        self
    }

    /// Create a new config with custom orphan summary retention (VERSIONING).
    /// (claude, 2026-02-07, FIXED: Added orphan_retention builder per VERSIONING)
    pub fn with_orphan_retention(mut self, retention: Duration) -> Self {
        self.orphan_retention = retention;
        self
    }
}

// ============================================================================
// Metrics
// ============================================================================

/// Metrics collected during GC operations.
#[derive(Debug, Default)]
pub struct GcMetrics {
    /// Number of node index entries deleted
    pub node_index_entries_deleted: AtomicU64,
    /// Number of edge index entries deleted
    pub edge_index_entries_deleted: AtomicU64,
    /// Number of tombstoned nodes hard-deleted
    pub node_tombstones_deleted: AtomicU64,
    /// Number of tombstoned edges hard-deleted
    pub edge_tombstones_deleted: AtomicU64,
    /// Number of orphan node summaries deleted (VERSIONING)
    /// (claude, 2026-02-07, FIXED: Added orphan summary GC metrics)
    pub orphan_node_summaries_deleted: AtomicU64,
    /// Number of orphan edge summaries deleted (VERSIONING)
    pub orphan_edge_summaries_deleted: AtomicU64,
    /// Number of GC cycles completed
    pub cycles_completed: AtomicU64,
}

impl GcMetrics {
    /// Create a new metrics instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get snapshot of current metrics.
    pub fn snapshot(&self) -> GcMetricsSnapshot {
        GcMetricsSnapshot {
            node_index_entries_deleted: self.node_index_entries_deleted.load(Ordering::Relaxed),
            edge_index_entries_deleted: self.edge_index_entries_deleted.load(Ordering::Relaxed),
            node_tombstones_deleted: self.node_tombstones_deleted.load(Ordering::Relaxed),
            edge_tombstones_deleted: self.edge_tombstones_deleted.load(Ordering::Relaxed),
            orphan_node_summaries_deleted: self.orphan_node_summaries_deleted.load(Ordering::Relaxed),
            orphan_edge_summaries_deleted: self.orphan_edge_summaries_deleted.load(Ordering::Relaxed),
            cycles_completed: self.cycles_completed.load(Ordering::Relaxed),
        }
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.node_index_entries_deleted.store(0, Ordering::Relaxed);
        self.edge_index_entries_deleted.store(0, Ordering::Relaxed);
        self.node_tombstones_deleted.store(0, Ordering::Relaxed);
        self.edge_tombstones_deleted.store(0, Ordering::Relaxed);
        self.orphan_node_summaries_deleted.store(0, Ordering::Relaxed);
        self.orphan_edge_summaries_deleted.store(0, Ordering::Relaxed);
        self.cycles_completed.store(0, Ordering::Relaxed);
    }
}

/// Point-in-time snapshot of GC metrics.
#[derive(Debug, Clone, Default)]
pub struct GcMetricsSnapshot {
    pub node_index_entries_deleted: u64,
    pub edge_index_entries_deleted: u64,
    pub node_tombstones_deleted: u64,
    pub edge_tombstones_deleted: u64,
    /// (claude, 2026-02-07, FIXED: Added orphan summary GC metrics per VERSIONING)
    pub orphan_node_summaries_deleted: u64,
    pub orphan_edge_summaries_deleted: u64,
    pub cycles_completed: u64,
}

impl GcMetricsSnapshot {
    /// Total entries deleted across all categories.
    pub fn total_deleted(&self) -> u64 {
        self.node_index_entries_deleted
            + self.edge_index_entries_deleted
            + self.node_tombstones_deleted
            + self.edge_tombstones_deleted
            + self.orphan_node_summaries_deleted
            + self.orphan_edge_summaries_deleted
    }
}

// ============================================================================
// Garbage Collector
// ============================================================================

/// Graph garbage collector for CONTENT-ADDRESS reverse index.
///
/// Provides incremental GC with cursor-based progress tracking.
///
/// # What GC Cleans Up
///
/// 1. **Stale index entries** - When entities are updated, old index entries
///    are marked STALE. GC deletes these based on `versions_to_keep` policy.
///
/// 2. **Tombstone cleanup** (future) - Hard-delete tombstoned entities after
///    `tombstone_retention` period.
///
/// 3. **Orphan summaries** (VERSIONING) - Summaries no longer referenced by
///    any current entity are tracked in OrphanSummaries CF. GC deletes these
///    after `orphan_retention` period, enabling rollback within that window.
///
/// # VERSIONING: Deferred Deletion
/// (claude, 2026-02-07, FIXED: Updated doc to reflect OrphanSummaries GC)
/// (claude, 2026-02-07, FIXED: P1.1-P1.3 - Store worker handle, use std::thread, shutdown joins)
///
/// Summary rows are NOT deleted inline when their entity is updated/deleted.
/// Instead, orphan candidates are written to OrphanSummaries CF. GC scans
/// this CF and deletes summaries older than `orphan_retention` that haven't
/// been restored. This enables time-travel and rollback operations.
pub struct GraphGarbageCollector {
    storage: Arc<Storage>,
    config: GraphGcConfig,
    shutdown: Arc<AtomicBool>,
    metrics: Arc<GcMetrics>,
    /// Worker thread handle for shutdown join.
    /// (claude, 2026-02-07, FIXED: P1.1 - Store handle instead of discarding)
    worker_handle: Option<std::thread::JoinHandle<()>>,
}

impl GraphGarbageCollector {
    /// Create a new garbage collector (without starting worker).
    ///
    /// Use `start()` to create and start a GC with embedded worker thread.
    pub fn new(storage: Arc<Storage>, config: GraphGcConfig) -> Self {
        Self {
            storage,
            config,
            shutdown: Arc::new(AtomicBool::new(false)),
            metrics: Arc::new(GcMetrics::new()),
            worker_handle: None,
        }
    }

    /// Create and start a garbage collector with embedded worker thread.
    ///
    /// This is the recommended way to create a GC for production use.
    /// The worker thread runs GC cycles at the configured interval.
    ///
    /// (claude, 2026-02-07, FIXED: P1.1-P1.3 - Use std::thread, store handle)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let gc = GraphGarbageCollector::start(storage, config);
    /// // ... later ...
    /// gc.shutdown(); // Blocks until worker completes
    /// ```
    pub fn start(storage: Arc<Storage>, config: GraphGcConfig) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let metrics = Arc::new(GcMetrics::new());

        // Clone for the worker thread
        let worker_storage = storage.clone();
        let worker_config = config.clone();
        let worker_shutdown = shutdown.clone();
        let worker_metrics = metrics.clone();

        // Spawn worker thread (not tokio task - blocking RocksDB work)
        // (claude, 2026-02-07, FIXED: P1.3 - Use std::thread instead of tokio::spawn)
        let worker_handle = std::thread::spawn(move || {
            tracing::info!("GC worker thread started");

            // Run startup cycle if configured
            if worker_config.process_on_startup {
                if let Err(e) = Self::run_cycle_inner(
                    &worker_storage,
                    &worker_config,
                    &worker_metrics,
                ) {
                    tracing::error!(error = %e, "GC startup cycle failed");
                }
            }

            // Main loop
            loop {
                // Sleep for interval
                std::thread::sleep(worker_config.interval);

                // Check shutdown flag
                if worker_shutdown.load(Ordering::SeqCst) {
                    tracing::info!("GC worker thread shutting down");
                    break;
                }

                // Run GC cycle
                if let Err(e) = Self::run_cycle_inner(
                    &worker_storage,
                    &worker_config,
                    &worker_metrics,
                ) {
                    tracing::error!(error = %e, "GC cycle failed");
                }
            }

            tracing::info!("GC worker thread exited");
        });

        Self {
            storage,
            config,
            shutdown,
            metrics,
            worker_handle: Some(worker_handle),
        }
    }

    /// Get reference to the metrics.
    pub fn metrics(&self) -> &Arc<GcMetrics> {
        &self.metrics
    }

    /// Shutdown GC and wait for worker thread to complete.
    ///
    /// Takes ownership to prevent use-after-shutdown.
    ///
    /// (claude, 2026-02-07, FIXED: P1.2 - Shutdown joins worker thread)
    pub fn shutdown(mut self) {
        tracing::info!("GC: initiating shutdown");
        self.shutdown.store(true, Ordering::SeqCst);

        if let Some(handle) = self.worker_handle.take() {
            if let Err(e) = handle.join() {
                tracing::error!("GC worker thread panicked: {:?}", e);
            }
        }

        tracing::info!("GC: shutdown complete");
    }

    /// Signal shutdown without waiting (for backwards compatibility).
    ///
    /// Prefer `shutdown(self)` or `shutdown_arc()` which wait for the worker.
    pub fn signal_shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    /// Shutdown GC from an Arc reference.
    ///
    /// Attempts to unwrap the Arc and perform a clean shutdown with thread join.
    /// If other references exist, signals shutdown and waits briefly for the worker.
    ///
    /// (claude, 2026-02-07, FIXED: P1.4 compatibility - handle Arc-wrapped GC)
    pub fn shutdown_arc(gc: Arc<Self>) {
        tracing::info!("GC: initiating shutdown (from Arc)");

        // Signal shutdown first (visible to worker immediately)
        gc.shutdown.store(true, Ordering::SeqCst);

        // Try to get sole ownership for clean thread join
        match Arc::try_unwrap(gc) {
            Ok(mut gc) => {
                // We have sole ownership - join the thread
                if let Some(handle) = gc.worker_handle.take() {
                    if let Err(e) = handle.join() {
                        tracing::error!("GC worker thread panicked: {:?}", e);
                    }
                }
                tracing::info!("GC: shutdown complete (owned)");
            }
            Err(arc) => {
                // Other references exist - give worker time to notice shutdown
                // The thread will exit on next iteration when it sees shutdown flag
                tracing::warn!(
                    "GC shutdown called but Arc has {} other references - worker will drain",
                    Arc::strong_count(&arc) - 1
                );
                // Brief wait to let worker notice shutdown
                std::thread::sleep(std::time::Duration::from_millis(100));
                tracing::info!("GC: shutdown signaled (shared)");
            }
        }
    }

    /// Check if shutdown has been signaled.
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::SeqCst)
    }

    /// Run a single GC cycle.
    ///
    /// Processes up to `batch_size` entries from each index CF,
    /// deleting stale entries based on version retention policy.
    pub fn run_cycle(&self) -> Result<GcMetricsSnapshot> {
        Self::run_cycle_inner(&self.storage, &self.config, &self.metrics)
    }

    /// Internal GC cycle implementation (can be called from worker thread).
    ///
    /// (claude, 2026-02-07, FIXED: P1.3 - Extracted for use by std::thread worker)
    fn run_cycle_inner(
        storage: &Arc<Storage>,
        config: &GraphGcConfig,
        metrics: &Arc<GcMetrics>,
    ) -> Result<GcMetricsSnapshot> {
        let before = metrics.snapshot();

        // GC stale node summary index entries
        Self::gc_node_summary_index_inner(storage, config, metrics)?;

        // GC stale edge summary index entries
        Self::gc_edge_summary_index_inner(storage, config, metrics)?;

        // GC orphan summaries (VERSIONING: deferred deletion with retention)
        // (claude, 2026-02-07, FIXED: Added orphan summary GC per VERSIONING)
        Self::gc_orphan_summaries_inner(storage, config, metrics)?;

        metrics.cycles_completed.fetch_add(1, Ordering::Relaxed);

        let after = metrics.snapshot();

        tracing::info!(
            node_index_deleted = after.node_index_entries_deleted - before.node_index_entries_deleted,
            edge_index_deleted = after.edge_index_entries_deleted - before.edge_index_entries_deleted,
            orphan_node_summaries_deleted = after.orphan_node_summaries_deleted - before.orphan_node_summaries_deleted,
            orphan_edge_summaries_deleted = after.orphan_edge_summaries_deleted - before.orphan_edge_summaries_deleted,
            cycle = after.cycles_completed,
            "GC cycle completed"
        );

        Ok(after)
    }

    /// GC stale entries from NodeSummaryIndex CF.
    ///
    /// Scans from cursor position, deletes STALE entries for old versions.
    fn gc_node_summary_index(&self) -> Result<u64> {
        Self::gc_node_summary_index_inner(&self.storage, &self.config, &self.metrics)
    }

    /// Internal implementation of node summary index GC.
    /// (claude, 2026-02-07, FIXED: P1.3 - Extracted for use by std::thread worker)
    fn gc_node_summary_index_inner(
        storage: &Arc<Storage>,
        config: &GraphGcConfig,
        metrics: &Arc<GcMetrics>,
    ) -> Result<u64> {
        let txn_db = storage.transaction_db()?;
        let txn = txn_db.transaction();
        let mut deleted = 0u64;
        let mut processed = 0usize;

        // Get CFs
        let index_cf = txn_db
            .cf_handle(NodeSummaryIndex::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("NodeSummaryIndex CF not found"))?;
        let nodes_cf = txn_db
            .cf_handle(Nodes::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Nodes CF not found"))?;
        let meta_cf = txn_db
            .cf_handle(GraphMeta::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("GraphMeta CF not found"))?;

        // Load cursor
        let cursor_key = GraphMetaCfKey::gc_cursor_node_summary_index();
        let cursor_key_bytes = GraphMeta::key_to_bytes(&cursor_key);
        let start_key = txn
            .get_cf(meta_cf, &cursor_key_bytes)?
            .unwrap_or_default();

        let n = config.versions_to_keep as Version;

        // Create iterator from cursor position
        let iter = if start_key.is_empty() {
            txn.iterator_cf(index_cf, rocksdb::IteratorMode::Start)
        } else {
            txn.iterator_cf(
                index_cf,
                rocksdb::IteratorMode::From(&start_key, rocksdb::Direction::Forward),
            )
        };

        let mut last_key: Option<Vec<u8>> = None;

        for item in iter {
            if processed >= config.batch_size {
                break;
            }

            let (key_bytes, value_bytes) = item?;
            last_key = Some(key_bytes.to_vec());
            processed += 1;

            // Parse index entry
            let index_key = NodeSummaryIndex::key_from_bytes(&key_bytes)?;
            let index_value = NodeSummaryIndex::value_from_bytes(&value_bytes)?;

            // Only GC stale entries
            if index_value.is_current() {
                continue;
            }

            let node_id = index_key.1;
            let version = index_key.2;

            // Look up current node version via prefix scan (VERSIONING)
            match find_current_node_version_for_gc(&txn, nodes_cf, node_id)? {
                Some((current_version, is_deleted)) => {
                    let min_keep = current_version.saturating_sub(n - 1);

                    // Delete if version is too old OR node is tombstoned
                    if version < min_keep || is_deleted {
                        txn.delete_cf(index_cf, &key_bytes)?;
                        deleted += 1;
                    }
                }
                None => {
                    // Node hard-deleted, remove orphan index entry
                    txn.delete_cf(index_cf, &key_bytes)?;
                    deleted += 1;
                }
            }
        }

        // Persist cursor
        if let Some(key) = last_key {
            let cursor_value = GraphMetaCfValue(GraphMetaField::GcCursorNodeSummaryIndex(key));
            txn.put_cf(meta_cf, &cursor_key_bytes, GraphMeta::value_to_bytes(&cursor_value))?;
        } else {
            // Iterator exhausted - delete cursor to start fresh
            txn.delete_cf(meta_cf, &cursor_key_bytes)?;
        }

        txn.commit()?;
        metrics.node_index_entries_deleted.fetch_add(deleted, Ordering::Relaxed);

        Ok(deleted)
    }

    /// GC stale entries from EdgeSummaryIndex CF.
    fn gc_edge_summary_index(&self) -> Result<u64> {
        Self::gc_edge_summary_index_inner(&self.storage, &self.config, &self.metrics)
    }

    /// Internal implementation of edge summary index GC.
    /// (claude, 2026-02-07, FIXED: P1.3 - Extracted for use by std::thread worker)
    fn gc_edge_summary_index_inner(
        storage: &Arc<Storage>,
        config: &GraphGcConfig,
        metrics: &Arc<GcMetrics>,
    ) -> Result<u64> {
        let txn_db = storage.transaction_db()?;
        let txn = txn_db.transaction();
        let mut deleted = 0u64;
        let mut processed = 0usize;

        // Get CFs
        let index_cf = txn_db
            .cf_handle(EdgeSummaryIndex::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("EdgeSummaryIndex CF not found"))?;
        let edges_cf = txn_db
            .cf_handle(ForwardEdges::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;
        let meta_cf = txn_db
            .cf_handle(GraphMeta::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("GraphMeta CF not found"))?;

        // Load cursor
        let cursor_key = GraphMetaCfKey::gc_cursor_edge_summary_index();
        let cursor_key_bytes = GraphMeta::key_to_bytes(&cursor_key);
        let start_key = txn
            .get_cf(meta_cf, &cursor_key_bytes)?
            .unwrap_or_default();

        let n = config.versions_to_keep as Version;

        // Create iterator from cursor position
        let iter = if start_key.is_empty() {
            txn.iterator_cf(index_cf, rocksdb::IteratorMode::Start)
        } else {
            txn.iterator_cf(
                index_cf,
                rocksdb::IteratorMode::From(&start_key, rocksdb::Direction::Forward),
            )
        };

        let mut last_key: Option<Vec<u8>> = None;

        for item in iter {
            if processed >= config.batch_size {
                break;
            }

            let (key_bytes, value_bytes) = item?;
            last_key = Some(key_bytes.to_vec());
            processed += 1;

            // Parse index entry
            let index_key = EdgeSummaryIndex::key_from_bytes(&key_bytes)?;
            let index_value = EdgeSummaryIndex::value_from_bytes(&value_bytes)?;

            // Only GC stale entries
            if index_value.is_current() {
                continue;
            }

            let src_id = index_key.1;
            let dst_id = index_key.2;
            let name_hash = index_key.3;
            let version = index_key.4;

            // Look up current edge version via prefix scan
            match find_current_edge_version_for_gc(&txn, edges_cf, src_id, dst_id, name_hash)? {
                Some((current_version, is_deleted)) => {
                    let min_keep = current_version.saturating_sub(n - 1);

                    // Delete if version is too old OR edge is tombstoned
                    if version < min_keep || is_deleted {
                        txn.delete_cf(index_cf, &key_bytes)?;
                        deleted += 1;
                    }
                }
                None => {
                    // Edge hard-deleted, remove orphan index entry
                    txn.delete_cf(index_cf, &key_bytes)?;
                    deleted += 1;
                }
            }
        }

        // Persist cursor
        if let Some(key) = last_key {
            let cursor_value = GraphMetaCfValue(GraphMetaField::GcCursorEdgeSummaryIndex(key));
            txn.put_cf(meta_cf, &cursor_key_bytes, GraphMeta::value_to_bytes(&cursor_value))?;
        } else {
            // Iterator exhausted - delete cursor to start fresh
            txn.delete_cf(meta_cf, &cursor_key_bytes)?;
        }

        txn.commit()?;
        metrics.edge_index_entries_deleted.fetch_add(deleted, Ordering::Relaxed);

        Ok(deleted)
    }

    /// GC orphan summaries from OrphanSummaries CF (VERSIONING).
    /// (claude, 2026-02-07, FIXED: Implemented orphan summary GC per VERSIONING)
    /// (claude, 2026-02-07, FIXED: Added reference check before deletion per Codex review)
    ///
    /// Scans OrphanSummaries CF and deletes summaries older than `orphan_retention`.
    /// Before deleting, verifies no CURRENT entries in the summary index reference this hash.
    /// Also deletes the summary data from NodeSummaries/EdgeSummaries CFs.
    fn gc_orphan_summaries(&self) -> Result<(u64, u64)> {
        Self::gc_orphan_summaries_inner(&self.storage, &self.config, &self.metrics)
    }

    /// Internal implementation of orphan summary GC.
    /// (claude, 2026-02-07, FIXED: P1.3 - Extracted for use by std::thread worker)
    fn gc_orphan_summaries_inner(
        storage: &Arc<Storage>,
        config: &GraphGcConfig,
        metrics: &Arc<GcMetrics>,
    ) -> Result<(u64, u64)> {
        use super::schema::{
            OrphanSummaries, SummaryKind,
            NodeSummaries, NodeSummaryCfKey,
            EdgeSummaries, EdgeSummaryCfKey,
            NodeSummaryIndex,
            EdgeSummaryIndex,
        };

        let txn_db = storage.transaction_db()?;
        let txn = txn_db.transaction();
        let mut node_deleted = 0u64;
        let mut edge_deleted = 0u64;
        let mut processed = 0usize;

        // Get CFs
        let orphan_cf = txn_db
            .cf_handle(OrphanSummaries::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("OrphanSummaries CF not found"))?;
        let node_summaries_cf = txn_db
            .cf_handle(NodeSummaries::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("NodeSummaries CF not found"))?;
        let edge_summaries_cf = txn_db
            .cf_handle(EdgeSummaries::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("EdgeSummaries CF not found"))?;
        let node_index_cf = txn_db
            .cf_handle(NodeSummaryIndex::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("NodeSummaryIndex CF not found"))?;
        let edge_index_cf = txn_db
            .cf_handle(EdgeSummaryIndex::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("EdgeSummaryIndex CF not found"))?;

        // Calculate cutoff time (entries older than this are eligible)
        let now = crate::TimestampMilli::now();
        let retention_millis = config.orphan_retention.as_millis() as u64;
        let cutoff = crate::TimestampMilli(now.0.saturating_sub(retention_millis));

        // Scan from start - OrphanSummaries is time-ordered
        let iter = txn.iterator_cf(orphan_cf, rocksdb::IteratorMode::Start);

        let mut orphan_keys_to_delete: Vec<Vec<u8>> = Vec::new();

        for item in iter {
            if processed >= config.batch_size {
                break;
            }

            let (key_bytes, value_bytes) = item?;
            processed += 1;

            // Parse orphan entry
            let orphan_key = OrphanSummaries::key_from_bytes(&key_bytes)?;
            let orphan_value = OrphanSummaries::value_from_bytes(&value_bytes)?;

            // Key format: (TimestampMilli, SummaryHash)
            let orphan_time = orphan_key.0;
            let summary_hash = orphan_key.1;

            // Skip entries that are still within retention window
            if orphan_time.0 > cutoff.0 {
                // OrphanSummaries is time-ordered, so all subsequent entries are newer
                break;
            }

            // Check if summary is still referenced by any CURRENT entity before deleting
            match orphan_value.0 {
                SummaryKind::Node => {
                    // Check if any CURRENT entries in NodeSummaryIndex reference this hash
                    if has_current_node_summary_reference(&txn, node_index_cf, summary_hash)? {
                        // Still referenced - just delete the orphan entry, not the summary
                        orphan_keys_to_delete.push(key_bytes.to_vec());
                        continue;
                    }
                    let summary_key = NodeSummaryCfKey(summary_hash);
                    let summary_key_bytes = NodeSummaries::key_to_bytes(&summary_key);
                    txn.delete_cf(node_summaries_cf, &summary_key_bytes)?;
                    node_deleted += 1;
                }
                SummaryKind::Edge => {
                    // Check if any CURRENT entries in EdgeSummaryIndex reference this hash
                    if has_current_edge_summary_reference(&txn, edge_index_cf, summary_hash)? {
                        // Still referenced - just delete the orphan entry, not the summary
                        orphan_keys_to_delete.push(key_bytes.to_vec());
                        continue;
                    }
                    let summary_key = EdgeSummaryCfKey(summary_hash);
                    let summary_key_bytes = EdgeSummaries::key_to_bytes(&summary_key);
                    txn.delete_cf(edge_summaries_cf, &summary_key_bytes)?;
                    edge_deleted += 1;
                }
            }

            // Mark orphan entry for deletion
            orphan_keys_to_delete.push(key_bytes.to_vec());
        }

        // Delete all processed orphan entries
        for key_bytes in orphan_keys_to_delete {
            txn.delete_cf(orphan_cf, &key_bytes)?;
        }

        txn.commit()?;

        metrics.orphan_node_summaries_deleted.fetch_add(node_deleted, Ordering::Relaxed);
        metrics.orphan_edge_summaries_deleted.fetch_add(edge_deleted, Ordering::Relaxed);

        Ok((node_deleted, edge_deleted))
    }

    /// Spawn background GC worker.
    ///
    /// Returns a JoinHandle that completes when shutdown is signaled.
    pub fn spawn_worker(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        let gc = self;
        tokio::spawn(async move {
            if gc.config.process_on_startup {
                if let Err(e) = gc.run_cycle() {
                    tracing::error!(error = %e, "GC startup cycle failed");
                }
            }

            let mut interval = tokio::time::interval(gc.config.interval);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                interval.tick().await;

                if gc.is_shutdown() {
                    tracing::info!("GC worker shutting down");
                    break;
                }

                if let Err(e) = gc.run_cycle() {
                    tracing::error!(error = %e, "GC cycle failed");
                }
            }
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
    fn test_gc_config_defaults() {
        let config = GraphGcConfig::default();
        assert_eq!(config.interval, Duration::from_secs(60));
        assert_eq!(config.batch_size, 1000);
        assert_eq!(config.versions_to_keep, 2);
        assert_eq!(config.tombstone_retention, Duration::from_secs(7 * 24 * 60 * 60));
        assert_eq!(config.orphan_retention, Duration::from_secs(7 * 24 * 60 * 60));
        assert!(config.process_on_startup);
    }

    #[test]
    fn test_gc_config_builder() {
        let config = GraphGcConfig::default()
            .with_interval(Duration::from_secs(30))
            .with_batch_size(500)
            .with_versions_to_keep(3)
            .with_tombstone_retention(Duration::from_secs(3600))
            .with_orphan_retention(Duration::from_secs(1800));

        assert_eq!(config.interval, Duration::from_secs(30));
        assert_eq!(config.batch_size, 500);
        assert_eq!(config.versions_to_keep, 3);
        assert_eq!(config.tombstone_retention, Duration::from_secs(3600));
        assert_eq!(config.orphan_retention, Duration::from_secs(1800));
    }

    #[test]
    fn test_gc_config_versions_minimum() {
        // versions_to_keep should be at least 1
        let config = GraphGcConfig::default().with_versions_to_keep(0);
        assert_eq!(config.versions_to_keep, 1);
    }

    #[test]
    fn test_gc_metrics_snapshot() {
        let metrics = GcMetrics::new();
        metrics.node_index_entries_deleted.fetch_add(10, Ordering::Relaxed);
        metrics.edge_index_entries_deleted.fetch_add(5, Ordering::Relaxed);
        metrics.cycles_completed.fetch_add(1, Ordering::Relaxed);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.node_index_entries_deleted, 10);
        assert_eq!(snapshot.edge_index_entries_deleted, 5);
        assert_eq!(snapshot.cycles_completed, 1);
        assert_eq!(snapshot.total_deleted(), 15);
    }

    #[test]
    fn test_gc_metrics_reset() {
        let metrics = GcMetrics::new();
        metrics.node_index_entries_deleted.fetch_add(10, Ordering::Relaxed);
        metrics.reset();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.node_index_entries_deleted, 0);
    }

    // ========================================================================
    // Lifecycle Tests (P1.5)
    // (claude, 2026-02-07, FIXED: Added lifecycle tests per ARCH2 P1.5)
    // ========================================================================

    #[test]
    fn test_gc_start_shutdown_lifecycle() {
        use tempfile::TempDir;
        use crate::graph::Storage;

        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("gc_lifecycle_test");

        // Create and ready storage, then wrap in Arc
        let mut storage = Storage::readwrite(&db_path);
        storage.ready().unwrap();
        let storage_arc = Arc::new(storage);

        // Start GC with short interval for testing
        let config = GraphGcConfig::default()
            .with_interval(Duration::from_millis(50))
            .with_batch_size(10);

        let gc = GraphGarbageCollector::start(storage_arc.clone(), config);

        // Let GC run a couple cycles
        std::thread::sleep(Duration::from_millis(150));

        // Verify at least one cycle ran
        let metrics = gc.metrics().snapshot();
        assert!(metrics.cycles_completed >= 1, "GC should have completed at least 1 cycle");

        // Shutdown - should block until worker completes
        gc.shutdown();

        // Storage Arc remains valid for cleanup
        drop(storage_arc);
    }

    #[test]
    fn test_gc_shutdown_arc() {
        use tempfile::TempDir;
        use crate::graph::Storage;

        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("gc_arc_test");

        // Create and ready storage, then wrap in Arc
        let mut storage = Storage::readwrite(&db_path);
        storage.ready().unwrap();
        let storage_arc = Arc::new(storage);

        // Start GC
        let config = GraphGcConfig::default()
            .with_interval(Duration::from_millis(100));

        let gc = GraphGarbageCollector::start(storage_arc.clone(), config);

        // Wrap in Arc (simulating subsystem ownership)
        let gc_arc = Arc::new(gc);

        // Get metrics reference before shutdown
        let _metrics = gc_arc.metrics().clone();

        // Extract inner GC for shutdown
        // Note: shutdown_arc handles Arc unwrap or signal+wait
        match Arc::try_unwrap(gc_arc) {
            Ok(gc) => {
                gc.shutdown();
            }
            Err(arc) => {
                // Would happen if there are other references
                GraphGarbageCollector::shutdown_arc(arc);
            }
        }

        // Storage Arc remains valid
        drop(storage_arc);
    }

    #[test]
    fn test_gc_signal_shutdown_non_blocking() {
        use tempfile::TempDir;
        use crate::graph::Storage;

        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("gc_signal_test");

        // Create and ready storage, then wrap in Arc
        let mut storage = Storage::readwrite(&db_path);
        storage.ready().unwrap();
        let storage_arc = Arc::new(storage);

        // Start GC with short interval so shutdown is quick
        let config = GraphGcConfig::default()
            .with_interval(Duration::from_millis(50));

        let gc = GraphGarbageCollector::start(storage_arc.clone(), config);

        // Signal shutdown (non-blocking)
        gc.signal_shutdown();

        // Verify shutdown flag is set
        assert!(gc.is_shutdown(), "Shutdown flag should be set after signal_shutdown");

        // Worker will exit on next iteration - shutdown to join
        gc.shutdown();

        // Storage Arc remains valid
        drop(storage_arc);
    }

    #[test]
    fn test_gc_new_without_start() {
        use tempfile::TempDir;
        use crate::graph::Storage;

        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("gc_new_test");

        // Create and ready storage, then wrap in Arc
        let mut storage = Storage::readwrite(&db_path);
        storage.ready().unwrap();
        let storage_arc = Arc::new(storage);

        // Create GC without starting worker
        let config = GraphGcConfig::default();
        let gc = GraphGarbageCollector::new(storage_arc.clone(), config);

        // Should have no worker handle
        assert!(!gc.is_shutdown(), "Shutdown flag should not be set initially");

        // Can run a single cycle manually
        let result = gc.run_cycle();
        assert!(result.is_ok(), "run_cycle should succeed");

        // Shutdown (no-op since no worker)
        gc.shutdown();

        // Storage Arc remains valid
        drop(storage_arc);
    }
}
