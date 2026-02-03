//! Repair module for graph index consistency checking.
//!
//! This module provides integrity verification and repair for:
//! - Forward/Reverse edge consistency
//! - Summary index consistency
//!
//! # Design
//!
//! Repair uses incremental batch processing similar to GC:
//! 1. Each cycle processes `batch_size` entries from cursor position
//! 2. Cursor is persisted for crash recovery
//! 3. Reports inconsistencies and optionally auto-fixes them
//!
//! # Usage
//!
//! ```rust,ignore
//! use motlie_db::graph::{Storage, RepairConfig, GraphRepairer};
//!
//! let config = RepairConfig::default();
//! let repairer = GraphRepairer::new(storage, config);
//!
//! // Run a single repair cycle
//! let metrics = repairer.run_cycle()?;
//! println!("Found {} missing reverse edges", metrics.missing_reverse);
//! ```

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;

use super::schema::{
    ForwardEdges, ForwardEdgeCfKey, ForwardEdgeCfValue,
    ReverseEdges, ReverseEdgeCfKey, ReverseEdgeCfValue,
    GraphMeta, GraphMetaCfKey, GraphMetaCfValue, GraphMetaField,
};
use super::{ColumnFamily, ColumnFamilySerde, HotColumnFamilyRecord, Storage};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the graph repairer.
#[derive(Debug, Clone)]
pub struct RepairConfig {
    /// Interval between repair cycles.
    /// Default: 1 hour
    pub interval: Duration,

    /// Maximum entries to check per cycle.
    /// Default: 10000
    pub batch_size: usize,

    /// Automatically fix inconsistencies (vs. report only).
    /// Default: false (report only)
    pub auto_fix: bool,

    /// Run a repair cycle immediately on startup.
    /// Default: false
    pub process_on_startup: bool,
}

impl Default for RepairConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(3600), // 1 hour
            batch_size: 10000,
            auto_fix: false,
            process_on_startup: false,
        }
    }
}

impl RepairConfig {
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

    /// Enable auto-fix mode.
    pub fn with_auto_fix(mut self, auto_fix: bool) -> Self {
        self.auto_fix = auto_fix;
        self
    }

    /// Process on startup.
    pub fn with_process_on_startup(mut self, process: bool) -> Self {
        self.process_on_startup = process;
        self
    }
}

// ============================================================================
// Metrics
// ============================================================================

/// Metrics collected during repair operations.
#[derive(Debug, Default)]
pub struct RepairMetrics {
    /// Forward edges missing their reverse entry
    pub missing_reverse: AtomicU64,
    /// Reverse edges pointing to non-existent forward edges (orphans)
    pub orphan_reverse: AtomicU64,
    /// Forward edges checked
    pub forward_checked: AtomicU64,
    /// Reverse edges checked
    pub reverse_checked: AtomicU64,
    /// Entries fixed (if auto_fix enabled)
    pub entries_fixed: AtomicU64,
    /// Number of repair cycles completed
    pub cycles_completed: AtomicU64,
}

impl RepairMetrics {
    /// Create a new metrics instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get snapshot of current metrics.
    pub fn snapshot(&self) -> RepairMetricsSnapshot {
        RepairMetricsSnapshot {
            missing_reverse: self.missing_reverse.load(Ordering::Relaxed),
            orphan_reverse: self.orphan_reverse.load(Ordering::Relaxed),
            forward_checked: self.forward_checked.load(Ordering::Relaxed),
            reverse_checked: self.reverse_checked.load(Ordering::Relaxed),
            entries_fixed: self.entries_fixed.load(Ordering::Relaxed),
            cycles_completed: self.cycles_completed.load(Ordering::Relaxed),
        }
    }

    /// Reset all counters to zero.
    pub fn reset(&self) {
        self.missing_reverse.store(0, Ordering::Relaxed);
        self.orphan_reverse.store(0, Ordering::Relaxed);
        self.forward_checked.store(0, Ordering::Relaxed);
        self.reverse_checked.store(0, Ordering::Relaxed);
        self.entries_fixed.store(0, Ordering::Relaxed);
        self.cycles_completed.store(0, Ordering::Relaxed);
    }
}

/// Point-in-time snapshot of repair metrics.
#[derive(Debug, Clone, Default)]
pub struct RepairMetricsSnapshot {
    pub missing_reverse: u64,
    pub orphan_reverse: u64,
    pub forward_checked: u64,
    pub reverse_checked: u64,
    pub entries_fixed: u64,
    pub cycles_completed: u64,
}

impl RepairMetricsSnapshot {
    /// Total inconsistencies found.
    pub fn total_inconsistencies(&self) -> u64 {
        self.missing_reverse + self.orphan_reverse
    }

    /// Returns true if no inconsistencies were found.
    pub fn is_consistent(&self) -> bool {
        self.total_inconsistencies() == 0
    }
}

// ============================================================================
// Graph Repairer
// ============================================================================

/// Graph repairer for consistency checking and repair.
pub struct GraphRepairer {
    storage: Arc<Storage>,
    config: RepairConfig,
    shutdown: Arc<AtomicBool>,
    metrics: Arc<RepairMetrics>,
}

impl GraphRepairer {
    /// Create a new repairer.
    pub fn new(storage: Arc<Storage>, config: RepairConfig) -> Self {
        Self {
            storage,
            config,
            shutdown: Arc::new(AtomicBool::new(false)),
            metrics: Arc::new(RepairMetrics::new()),
        }
    }

    /// Get reference to the metrics.
    pub fn metrics(&self) -> &Arc<RepairMetrics> {
        &self.metrics
    }

    /// Signal shutdown to the repair worker.
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    /// Check if shutdown has been signaled.
    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::SeqCst)
    }

    /// Run a single repair cycle.
    ///
    /// Checks forward→reverse and reverse→forward consistency.
    pub fn run_cycle(&self) -> Result<RepairMetricsSnapshot> {
        let before = self.metrics.snapshot();

        // Check forward edges have corresponding reverse entries
        self.check_forward_to_reverse()?;

        // Check reverse edges point to existing forward edges
        self.check_reverse_to_forward()?;

        self.metrics.cycles_completed.fetch_add(1, Ordering::Relaxed);

        let after = self.metrics.snapshot();

        tracing::info!(
            missing_reverse = after.missing_reverse - before.missing_reverse,
            orphan_reverse = after.orphan_reverse - before.orphan_reverse,
            entries_fixed = after.entries_fixed - before.entries_fixed,
            cycle = after.cycles_completed,
            "Repair cycle completed"
        );

        Ok(after)
    }

    /// Check that each forward edge has a corresponding reverse edge.
    fn check_forward_to_reverse(&self) -> Result<u64> {
        let txn_db = self.storage.transaction_db()?;
        let txn = txn_db.transaction();
        let mut missing = 0u64;
        let mut fixed = 0u64;
        let mut checked = 0usize;

        // Get CFs
        let forward_cf = txn_db
            .cf_handle(ForwardEdges::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;
        let reverse_cf = txn_db
            .cf_handle(ReverseEdges::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("ReverseEdges CF not found"))?;
        let meta_cf = txn_db
            .cf_handle(GraphMeta::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("GraphMeta CF not found"))?;

        // Load cursor (reuse NodeTombstones cursor slot for forward scan)
        let cursor_key = GraphMetaCfKey::gc_cursor_node_tombstones();
        let cursor_key_bytes = GraphMeta::key_to_bytes(&cursor_key);
        let start_key = txn
            .get_cf(meta_cf, &cursor_key_bytes)?
            .unwrap_or_default();

        // Create iterator from cursor position
        let iter = if start_key.is_empty() {
            txn.iterator_cf(forward_cf, rocksdb::IteratorMode::Start)
        } else {
            txn.iterator_cf(
                forward_cf,
                rocksdb::IteratorMode::From(&start_key, rocksdb::Direction::Forward),
            )
        };

        let mut last_key: Option<Vec<u8>> = None;

        for item in iter {
            if checked >= self.config.batch_size {
                break;
            }

            let (key_bytes, value_bytes) = item?;
            last_key = Some(key_bytes.to_vec());
            checked += 1;

            // Parse forward edge key
            let fwd_key: ForwardEdgeCfKey = ForwardEdges::key_from_bytes(&key_bytes)?;
            let fwd_value: ForwardEdgeCfValue = ForwardEdges::value_from_bytes(&value_bytes)?;

            // Skip deleted edges
            if fwd_value.4 {
                continue;
            }

            // Construct corresponding reverse key
            let rev_key = ReverseEdgeCfKey(fwd_key.1, fwd_key.0, fwd_key.2);
            let rev_key_bytes = ReverseEdges::key_to_bytes(&rev_key);

            // Check if reverse entry exists
            if txn.get_cf(reverse_cf, &rev_key_bytes)?.is_none() {
                missing += 1;

                if self.config.auto_fix {
                    // Create missing reverse entry
                    let rev_value = ReverseEdgeCfValue(fwd_value.0.clone());
                    let rev_value_bytes = ReverseEdges::value_to_bytes(&rev_value)?;
                    txn.put_cf(reverse_cf, rev_key_bytes, rev_value_bytes)?;
                    fixed += 1;
                }
            }
        }

        // Persist cursor
        if let Some(key) = last_key {
            let cursor_value = GraphMetaCfValue(GraphMetaField::GcCursorNodeTombstones(key));
            txn.put_cf(meta_cf, &cursor_key_bytes, GraphMeta::value_to_bytes(&cursor_value))?;
        } else {
            // Iterator exhausted - delete cursor to start fresh
            txn.delete_cf(meta_cf, &cursor_key_bytes)?;
        }

        txn.commit()?;
        self.metrics.forward_checked.fetch_add(checked as u64, Ordering::Relaxed);
        self.metrics.missing_reverse.fetch_add(missing, Ordering::Relaxed);
        self.metrics.entries_fixed.fetch_add(fixed, Ordering::Relaxed);

        Ok(missing)
    }

    /// Check that each reverse edge points to an existing forward edge.
    fn check_reverse_to_forward(&self) -> Result<u64> {
        let txn_db = self.storage.transaction_db()?;
        let txn = txn_db.transaction();
        let mut orphans = 0u64;
        let mut fixed = 0u64;
        let mut checked = 0usize;

        // Get CFs
        let forward_cf = txn_db
            .cf_handle(ForwardEdges::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;
        let reverse_cf = txn_db
            .cf_handle(ReverseEdges::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("ReverseEdges CF not found"))?;
        let meta_cf = txn_db
            .cf_handle(GraphMeta::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("GraphMeta CF not found"))?;

        // Load cursor (reuse EdgeTombstones cursor slot for reverse scan)
        let cursor_key = GraphMetaCfKey::gc_cursor_edge_tombstones();
        let cursor_key_bytes = GraphMeta::key_to_bytes(&cursor_key);
        let start_key = txn
            .get_cf(meta_cf, &cursor_key_bytes)?
            .unwrap_or_default();

        // Create iterator from cursor position
        let iter = if start_key.is_empty() {
            txn.iterator_cf(reverse_cf, rocksdb::IteratorMode::Start)
        } else {
            txn.iterator_cf(
                reverse_cf,
                rocksdb::IteratorMode::From(&start_key, rocksdb::Direction::Forward),
            )
        };

        let mut last_key: Option<Vec<u8>> = None;

        for item in iter {
            if checked >= self.config.batch_size {
                break;
            }

            let (key_bytes, _value_bytes) = item?;
            last_key = Some(key_bytes.to_vec());
            checked += 1;

            // Parse reverse edge key
            let rev_key: ReverseEdgeCfKey = ReverseEdges::key_from_bytes(&key_bytes)?;

            // Construct corresponding forward key
            let fwd_key = ForwardEdgeCfKey(rev_key.1, rev_key.0, rev_key.2);
            let fwd_key_bytes = ForwardEdges::key_to_bytes(&fwd_key);

            // Check if forward entry exists
            match txn.get_cf(forward_cf, &fwd_key_bytes)? {
                Some(fwd_value_bytes) => {
                    // Forward exists - check if it's deleted
                    let fwd_value: ForwardEdgeCfValue = ForwardEdges::value_from_bytes(&fwd_value_bytes)?;
                    if fwd_value.4 {
                        // Forward edge is tombstoned, reverse is orphan
                        orphans += 1;

                        if self.config.auto_fix {
                            txn.delete_cf(reverse_cf, &key_bytes)?;
                            fixed += 1;
                        }
                    }
                }
                None => {
                    // Forward edge doesn't exist - reverse is orphan
                    orphans += 1;

                    if self.config.auto_fix {
                        txn.delete_cf(reverse_cf, &key_bytes)?;
                        fixed += 1;
                    }
                }
            }
        }

        // Persist cursor
        if let Some(key) = last_key {
            let cursor_value = GraphMetaCfValue(GraphMetaField::GcCursorEdgeTombstones(key));
            txn.put_cf(meta_cf, &cursor_key_bytes, GraphMeta::value_to_bytes(&cursor_value))?;
        } else {
            // Iterator exhausted - delete cursor to start fresh
            txn.delete_cf(meta_cf, &cursor_key_bytes)?;
        }

        txn.commit()?;
        self.metrics.reverse_checked.fetch_add(checked as u64, Ordering::Relaxed);
        self.metrics.orphan_reverse.fetch_add(orphans, Ordering::Relaxed);
        self.metrics.entries_fixed.fetch_add(fixed, Ordering::Relaxed);

        Ok(orphans)
    }

    /// Spawn background repair worker.
    ///
    /// Returns a JoinHandle that completes when shutdown is signaled.
    pub fn spawn_worker(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        let repairer = self;
        tokio::spawn(async move {
            if repairer.config.process_on_startup {
                if let Err(e) = repairer.run_cycle() {
                    tracing::error!(error = %e, "Repair startup cycle failed");
                }
            }

            let mut interval = tokio::time::interval(repairer.config.interval);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                interval.tick().await;

                if repairer.is_shutdown() {
                    tracing::info!("Repair worker shutting down");
                    break;
                }

                if let Err(e) = repairer.run_cycle() {
                    tracing::error!(error = %e, "Repair cycle failed");
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
    fn test_repair_config_defaults() {
        let config = RepairConfig::default();
        assert_eq!(config.interval, Duration::from_secs(3600));
        assert_eq!(config.batch_size, 10000);
        assert!(!config.auto_fix);
        assert!(!config.process_on_startup);
    }

    #[test]
    fn test_repair_config_builder() {
        let config = RepairConfig::default()
            .with_interval(Duration::from_secs(1800))
            .with_batch_size(5000)
            .with_auto_fix(true)
            .with_process_on_startup(true);

        assert_eq!(config.interval, Duration::from_secs(1800));
        assert_eq!(config.batch_size, 5000);
        assert!(config.auto_fix);
        assert!(config.process_on_startup);
    }

    #[test]
    fn test_repair_metrics_snapshot() {
        let metrics = RepairMetrics::new();
        metrics.missing_reverse.fetch_add(5, Ordering::Relaxed);
        metrics.orphan_reverse.fetch_add(3, Ordering::Relaxed);
        metrics.cycles_completed.fetch_add(1, Ordering::Relaxed);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.missing_reverse, 5);
        assert_eq!(snapshot.orphan_reverse, 3);
        assert_eq!(snapshot.total_inconsistencies(), 8);
        assert!(!snapshot.is_consistent());
    }

    #[test]
    fn test_repair_metrics_consistent() {
        let metrics = RepairMetrics::new();
        metrics.forward_checked.fetch_add(100, Ordering::Relaxed);
        metrics.cycles_completed.fetch_add(1, Ordering::Relaxed);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_inconsistencies(), 0);
        assert!(snapshot.is_consistent());
    }

    #[test]
    fn test_repair_metrics_reset() {
        let metrics = RepairMetrics::new();
        metrics.missing_reverse.fetch_add(5, Ordering::Relaxed);
        metrics.reset();

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.missing_reverse, 0);
    }
}
