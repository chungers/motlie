//! Garbage Collection for CONTENT-ADDRESS reverse index system.
//!
//! This module provides incremental GC for:
//! - Stale index entries (NodeSummaryIndex, EdgeSummaryIndex)
//! - Tombstoned entities after retention period (future)
//!
//! # RefCount Eliminates Orphan Summary Scans
//!
//! Summary rows include a RefCount field that tracks how many entities reference them.
//! When RefCount reaches 0, the summary row is deleted inline by the mutation executor.
//! This eliminates the need for expensive background scans to find orphan summaries.
// (codex, 2026-02-07, eval: VERSIONING now adds OrphanSummaries + deferred deletion; this header is outdated relative to the new GC plan and should be updated to avoid conflicting guidance.)
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
            cycles_completed: self.cycles_completed.load(Ordering::Relaxed),
        }
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.node_index_entries_deleted.store(0, Ordering::Relaxed);
        self.edge_index_entries_deleted.store(0, Ordering::Relaxed);
        self.node_tombstones_deleted.store(0, Ordering::Relaxed);
        self.edge_tombstones_deleted.store(0, Ordering::Relaxed);
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
    pub cycles_completed: u64,
}

impl GcMetricsSnapshot {
    /// Total entries deleted across all categories.
    pub fn total_deleted(&self) -> u64 {
        self.node_index_entries_deleted
            + self.edge_index_entries_deleted
            + self.node_tombstones_deleted
            + self.edge_tombstones_deleted
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
/// # What RefCount Handles (No GC Needed)
///
/// Summary rows (NodeSummaries, EdgeSummaries) are automatically deleted
/// when their RefCount reaches 0. This happens inline during mutation
/// execution, so GC doesn't need to scan for orphan summaries.
pub struct GraphGarbageCollector {
    storage: Arc<Storage>,
    config: GraphGcConfig,
    shutdown: Arc<AtomicBool>,
    metrics: Arc<GcMetrics>,
}

impl GraphGarbageCollector {
    /// Create a new garbage collector.
    pub fn new(storage: Arc<Storage>, config: GraphGcConfig) -> Self {
        Self {
            storage,
            config,
            shutdown: Arc::new(AtomicBool::new(false)),
            metrics: Arc::new(GcMetrics::new()),
        }
    }

    /// Get reference to the metrics.
    pub fn metrics(&self) -> &Arc<GcMetrics> {
        &self.metrics
    }

    /// Signal shutdown to the GC worker.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
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
        let before = self.metrics.snapshot();

        // GC stale node summary index entries
        self.gc_node_summary_index()?;

        // GC stale edge summary index entries
        self.gc_edge_summary_index()?;

        // Note: Summary rows are cleaned up by RefCount (inline during mutations)
        // No orphan scan needed!

        self.metrics.cycles_completed.fetch_add(1, Ordering::Relaxed);

        let after = self.metrics.snapshot();

        tracing::info!(
            node_index_deleted = after.node_index_entries_deleted - before.node_index_entries_deleted,
            edge_index_deleted = after.edge_index_entries_deleted - before.edge_index_entries_deleted,
            cycle = after.cycles_completed,
            "GC cycle completed"
        );

        Ok(after)
    }

    /// GC stale entries from NodeSummaryIndex CF.
    ///
    /// Scans from cursor position, deletes STALE entries for old versions.
    fn gc_node_summary_index(&self) -> Result<u64> {
        let txn_db = self.storage.transaction_db()?;
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

        let n = self.config.versions_to_keep as Version;

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
            if processed >= self.config.batch_size {
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
        self.metrics.node_index_entries_deleted.fetch_add(deleted, Ordering::Relaxed);

        Ok(deleted)
    }

    /// GC stale entries from EdgeSummaryIndex CF.
    fn gc_edge_summary_index(&self) -> Result<u64> {
        let txn_db = self.storage.transaction_db()?;
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

        let n = self.config.versions_to_keep as Version;

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
            if processed >= self.config.batch_size {
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
            // (claude, 2026-02-06, in-progress: VERSIONING uses prefix scan)
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
        self.metrics.edge_index_entries_deleted.fetch_add(deleted, Ordering::Relaxed);

        Ok(deleted)
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
        assert!(config.process_on_startup);
    }

    #[test]
    fn test_gc_config_builder() {
        let config = GraphGcConfig::default()
            .with_interval(Duration::from_secs(30))
            .with_batch_size(500)
            .with_versions_to_keep(3)
            .with_tombstone_retention(Duration::from_secs(3600));

        assert_eq!(config.interval, Duration::from_secs(30));
        assert_eq!(config.batch_size, 500);
        assert_eq!(config.versions_to_keep, 3);
        assert_eq!(config.tombstone_retention, Duration::from_secs(3600));
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
}
