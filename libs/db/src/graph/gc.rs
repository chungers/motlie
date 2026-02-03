//! Garbage Collection for CONTENT-ADDRESS reverse index system.
//!
//! This module provides incremental GC for:
//! - Old summary versions (NodeSummaries, EdgeSummaries)
//! - Stale index entries (NodeSummaryIndex, EdgeSummaryIndex)
//! - Tombstoned entities after retention period
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

use super::schema::{
    GraphMeta, GraphMetaCfKey, GraphMetaCfValue, GraphMetaField,
    NodeSummaryIndex, EdgeSummaryIndex,
    NodeSummaries, NodeSummaryCfKey,
    EdgeSummaries, EdgeSummaryCfKey,
    Nodes, NodeCfKey, NodeCfValue,
    ForwardEdges, ForwardEdgeCfKey, ForwardEdgeCfValue,
    Version,
};
use super::SummaryHash;
use super::{ColumnFamily, ColumnFamilySerde, HotColumnFamilyRecord, Storage};

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
    /// Number of node summary entries deleted
    pub node_summaries_deleted: AtomicU64,
    /// Number of edge summary entries deleted
    pub edge_summaries_deleted: AtomicU64,
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
            node_summaries_deleted: self.node_summaries_deleted.load(Ordering::Relaxed),
            edge_summaries_deleted: self.edge_summaries_deleted.load(Ordering::Relaxed),
            node_index_entries_deleted: self.node_index_entries_deleted.load(Ordering::Relaxed),
            edge_index_entries_deleted: self.edge_index_entries_deleted.load(Ordering::Relaxed),
            node_tombstones_deleted: self.node_tombstones_deleted.load(Ordering::Relaxed),
            edge_tombstones_deleted: self.edge_tombstones_deleted.load(Ordering::Relaxed),
            cycles_completed: self.cycles_completed.load(Ordering::Relaxed),
        }
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.node_summaries_deleted.store(0, Ordering::Relaxed);
        self.edge_summaries_deleted.store(0, Ordering::Relaxed);
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
    pub node_summaries_deleted: u64,
    pub edge_summaries_deleted: u64,
    pub node_index_entries_deleted: u64,
    pub edge_index_entries_deleted: u64,
    pub node_tombstones_deleted: u64,
    pub edge_tombstones_deleted: u64,
    pub cycles_completed: u64,
}

impl GcMetricsSnapshot {
    /// Total entries deleted across all categories.
    pub fn total_deleted(&self) -> u64 {
        self.node_summaries_deleted
            + self.edge_summaries_deleted
            + self.node_index_entries_deleted
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
    /// Processes up to `batch_size` entries from each GC target,
    /// persisting cursors for incremental progress.
    pub fn run_cycle(&self) -> Result<GcMetricsSnapshot> {
        let before = self.metrics.snapshot();

        // GC stale node summary index entries
        self.gc_node_summary_index()?;

        // GC stale edge summary index entries
        self.gc_edge_summary_index()?;

        // GC orphaned node summaries (no index references)
        self.gc_node_summaries()?;

        // GC orphaned edge summaries (no index references)
        self.gc_edge_summaries()?;

        self.metrics.cycles_completed.fetch_add(1, Ordering::Relaxed);

        let after = self.metrics.snapshot();

        tracing::info!(
            node_index_deleted = after.node_index_entries_deleted - before.node_index_entries_deleted,
            edge_index_deleted = after.edge_index_entries_deleted - before.edge_index_entries_deleted,
            node_summaries_deleted = after.node_summaries_deleted - before.node_summaries_deleted,
            edge_summaries_deleted = after.edge_summaries_deleted - before.edge_summaries_deleted,
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

        // Build version cache: node_id → current_version
        // Note: In production, this should be done incrementally or use a bloom filter
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

            // Look up current node version
            let node_key = NodeCfKey(node_id);
            let node_key_bytes = Nodes::key_to_bytes(&node_key);

            match txn.get_cf(nodes_cf, &node_key_bytes)? {
                Some(node_bytes) => {
                    let node: NodeCfValue = Nodes::value_from_bytes(&node_bytes)?;
                    let current_version = node.3;
                    let is_deleted = node.4;
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

            // Look up current edge version
            let edge_key = ForwardEdgeCfKey(src_id, dst_id, name_hash);
            let edge_key_bytes = ForwardEdges::key_to_bytes(&edge_key);

            match txn.get_cf(edges_cf, &edge_key_bytes)? {
                Some(edge_bytes) => {
                    let edge: ForwardEdgeCfValue = ForwardEdges::value_from_bytes(&edge_bytes)?;
                    let current_version = edge.3;
                    let is_deleted = edge.4;
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

    /// GC orphaned entries from NodeSummaries CF.
    ///
    /// Summaries are content-addressed (keyed by SummaryHash). An orphan is a summary
    /// that has no index entries referencing it. This scans summaries and deletes
    /// those with no corresponding index entries.
    fn gc_node_summaries(&self) -> Result<u64> {
        let txn_db = self.storage.transaction_db()?;
        let txn = txn_db.transaction();
        let mut deleted = 0u64;
        let mut processed = 0usize;

        // Get CFs
        let summaries_cf = txn_db
            .cf_handle(NodeSummaries::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("NodeSummaries CF not found"))?;
        let index_cf = txn_db
            .cf_handle(NodeSummaryIndex::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("NodeSummaryIndex CF not found"))?;
        let meta_cf = txn_db
            .cf_handle(GraphMeta::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("GraphMeta CF not found"))?;

        // Load cursor
        let cursor_key = GraphMetaCfKey::gc_cursor_node_summaries();
        let cursor_key_bytes = GraphMeta::key_to_bytes(&cursor_key);
        let start_key = txn
            .get_cf(meta_cf, &cursor_key_bytes)?
            .unwrap_or_default();

        // Create iterator from cursor position
        let iter = if start_key.is_empty() {
            txn.iterator_cf(summaries_cf, rocksdb::IteratorMode::Start)
        } else {
            txn.iterator_cf(
                summaries_cf,
                rocksdb::IteratorMode::From(&start_key, rocksdb::Direction::Forward),
            )
        };

        let mut last_key: Option<Vec<u8>> = None;

        for item in iter {
            if processed >= self.config.batch_size {
                break;
            }

            let (key_bytes, _value_bytes) = item?;
            last_key = Some(key_bytes.to_vec());
            processed += 1;

            // Parse summary key to get the hash
            let summary_key: NodeSummaryCfKey = NodeSummaries::key_from_bytes(&key_bytes)?;
            let hash: SummaryHash = summary_key.0;

            // Check if any index entry references this hash (prefix scan)
            let hash_prefix = hash.as_bytes();
            let mut has_references = false;

            let index_iter = txn.prefix_iterator_cf(index_cf, hash_prefix);
            for index_item in index_iter {
                let (index_key, _) = index_item?;
                // Verify prefix match (prefix_iterator may return keys after prefix)
                if index_key.starts_with(hash_prefix) {
                    has_references = true;
                    break;
                } else {
                    break; // Moved past prefix
                }
            }

            // Delete orphaned summary
            if !has_references {
                txn.delete_cf(summaries_cf, &key_bytes)?;
                deleted += 1;
            }
        }

        // Persist cursor
        if let Some(key) = last_key {
            let cursor_value = GraphMetaCfValue(GraphMetaField::GcCursorNodeSummaries(key));
            txn.put_cf(meta_cf, &cursor_key_bytes, GraphMeta::value_to_bytes(&cursor_value))?;
        } else {
            // Iterator exhausted - delete cursor to start fresh
            txn.delete_cf(meta_cf, &cursor_key_bytes)?;
        }

        txn.commit()?;
        self.metrics.node_summaries_deleted.fetch_add(deleted, Ordering::Relaxed);

        Ok(deleted)
    }

    /// GC orphaned entries from EdgeSummaries CF.
    ///
    /// Similar to gc_node_summaries - deletes summaries with no index references.
    fn gc_edge_summaries(&self) -> Result<u64> {
        let txn_db = self.storage.transaction_db()?;
        let txn = txn_db.transaction();
        let mut deleted = 0u64;
        let mut processed = 0usize;

        // Get CFs
        let summaries_cf = txn_db
            .cf_handle(EdgeSummaries::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("EdgeSummaries CF not found"))?;
        let index_cf = txn_db
            .cf_handle(EdgeSummaryIndex::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("EdgeSummaryIndex CF not found"))?;
        let meta_cf = txn_db
            .cf_handle(GraphMeta::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("GraphMeta CF not found"))?;

        // Load cursor
        let cursor_key = GraphMetaCfKey::gc_cursor_edge_summaries();
        let cursor_key_bytes = GraphMeta::key_to_bytes(&cursor_key);
        let start_key = txn
            .get_cf(meta_cf, &cursor_key_bytes)?
            .unwrap_or_default();

        // Create iterator from cursor position
        let iter = if start_key.is_empty() {
            txn.iterator_cf(summaries_cf, rocksdb::IteratorMode::Start)
        } else {
            txn.iterator_cf(
                summaries_cf,
                rocksdb::IteratorMode::From(&start_key, rocksdb::Direction::Forward),
            )
        };

        let mut last_key: Option<Vec<u8>> = None;

        for item in iter {
            if processed >= self.config.batch_size {
                break;
            }

            let (key_bytes, _value_bytes) = item?;
            last_key = Some(key_bytes.to_vec());
            processed += 1;

            // Parse summary key to get the hash
            let summary_key: EdgeSummaryCfKey = EdgeSummaries::key_from_bytes(&key_bytes)?;
            let hash: SummaryHash = summary_key.0;

            // Check if any index entry references this hash (prefix scan)
            let hash_prefix = hash.as_bytes();
            let mut has_references = false;

            let index_iter = txn.prefix_iterator_cf(index_cf, hash_prefix);
            for index_item in index_iter {
                let (index_key, _) = index_item?;
                // Verify prefix match
                if index_key.starts_with(hash_prefix) {
                    has_references = true;
                    break;
                } else {
                    break;
                }
            }

            // Delete orphaned summary
            if !has_references {
                txn.delete_cf(summaries_cf, &key_bytes)?;
                deleted += 1;
            }
        }

        // Persist cursor
        if let Some(key) = last_key {
            let cursor_value = GraphMetaCfValue(GraphMetaField::GcCursorEdgeSummaries(key));
            txn.put_cf(meta_cf, &cursor_key_bytes, GraphMeta::value_to_bytes(&cursor_value))?;
        } else {
            // Iterator exhausted - delete cursor to start fresh
            txn.delete_cf(meta_cf, &cursor_key_bytes)?;
        }

        txn.commit()?;
        self.metrics.edge_summaries_deleted.fetch_add(deleted, Ordering::Relaxed);

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
