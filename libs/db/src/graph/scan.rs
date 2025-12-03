//! Scan API for iterating over column families with pagination support.
//!
//! This module provides a visitor-based API for scanning column families without
//! exposing internal schema types. Users define visitors that receive user-friendly
//! record types.
//!
//! # Example
//!
//! ```ignore
//! use motlie_db::{Storage, graph::scan::{AllNodes, Visitable}};
//!
//! let mut storage = Storage::readonly(db_path);
//! storage.ready()?;
//!
//! let scan = AllNodes { last: None, limit: 100 };
//! scan.accept(&storage, &mut |record| {
//!     println!("{}\t{}", record.id, record.name);
//!     true // continue scanning
//! })?;
//! ```

use anyhow::Result;
use rocksdb::{Direction, IteratorMode};

use super::ColumnFamilyRecord;
use super::schema::{
    self, is_valid_at_time, DstId, EdgeName, EdgeSummary, FragmentContent, NodeName, NodeSummary,
    SrcId, ValidTemporalRange,
};
use super::Storage;
use crate::{Id, TimestampMilli};

// Re-export ValidTemporalRange for use in record types
pub use crate::ValidTemporalRange as TemporalRange;

// ============================================================================
// Visitor Trait
// ============================================================================

/// Visitor trait for processing scanned records.
///
/// The visitor is called for each record in the scan.
/// Return `true` to continue scanning, `false` to stop early.
pub trait Visitor<R> {
    /// Visit a single record.
    /// Returns `true` to continue, `false` to stop scanning.
    fn visit(&mut self, record: &R) -> bool;
}

/// Blanket implementation for closures that take a reference and return bool.
impl<R, F> Visitor<R> for F
where
    F: FnMut(&R) -> bool,
{
    fn visit(&mut self, record: &R) -> bool {
        self(record)
    }
}

// ============================================================================
// Visitable Trait
// ============================================================================

/// Trait for scan types that can accept visitors.
pub trait Visitable {
    /// The record type this scan produces.
    type Record;

    /// Execute the scan, calling the visitor for each record.
    /// Returns the number of records visited.
    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize>;
}

// ============================================================================
// Public Record Types
// ============================================================================

/// A node record as seen by scan visitors.
#[derive(Debug, Clone)]
pub struct NodeRecord {
    pub id: Id,
    pub name: NodeName,
    pub summary: NodeSummary,
    pub valid_range: Option<ValidTemporalRange>,
}

/// A forward edge record as seen by scan visitors.
#[derive(Debug, Clone)]
pub struct EdgeRecord {
    pub src_id: SrcId,
    pub dst_id: DstId,
    pub name: EdgeName,
    pub summary: EdgeSummary,
    pub weight: Option<f64>,
    pub valid_range: Option<ValidTemporalRange>,
}

/// A reverse edge record as seen by scan visitors (index only, no summary/weight).
#[derive(Debug, Clone)]
pub struct ReverseEdgeRecord {
    pub dst_id: DstId,
    pub src_id: SrcId,
    pub name: EdgeName,
    pub valid_range: Option<ValidTemporalRange>,
}

/// A node fragment record as seen by scan visitors.
#[derive(Debug, Clone)]
pub struct NodeFragmentRecord {
    pub node_id: Id,
    pub timestamp: TimestampMilli,
    pub content: FragmentContent,
    pub valid_range: Option<ValidTemporalRange>,
}

/// An edge fragment record as seen by scan visitors.
#[derive(Debug, Clone)]
pub struct EdgeFragmentRecord {
    pub src_id: SrcId,
    pub dst_id: DstId,
    pub edge_name: EdgeName,
    pub timestamp: TimestampMilli,
    pub content: FragmentContent,
    pub valid_range: Option<ValidTemporalRange>,
}

/// A node name index record as seen by scan visitors.
#[derive(Debug, Clone)]
pub struct NodeNameRecord {
    pub name: NodeName,
    pub node_id: Id,
    pub valid_range: Option<ValidTemporalRange>,
}

/// An edge name index record as seen by scan visitors.
#[derive(Debug, Clone)]
pub struct EdgeNameRecord {
    pub name: EdgeName,
    pub src_id: SrcId,
    pub dst_id: DstId,
    pub valid_range: Option<ValidTemporalRange>,
}

// ============================================================================
// Scan Types
// ============================================================================

/// Scan all nodes with pagination.
#[derive(Debug, Clone, Default)]
pub struct AllNodes {
    /// Last node ID from previous page (exclusive start for pagination).
    pub last: Option<Id>,
    /// Maximum number of records to return.
    pub limit: usize,
    /// Scan in reverse direction (from end to start).
    pub reverse: bool,
    /// Reference timestamp for temporal validity check.
    /// If Some, only records valid at this time are returned.
    /// If None, all records are returned regardless of temporal validity.
    pub reference_ts_millis: Option<TimestampMilli>,
}

/// Scan all forward edges with pagination.
#[derive(Debug, Clone, Default)]
pub struct AllEdges {
    /// Last edge cursor from previous page (src_id, dst_id, edge_name).
    pub last: Option<(SrcId, DstId, EdgeName)>,
    /// Maximum number of records to return.
    pub limit: usize,
    /// Scan in reverse direction (from end to start).
    pub reverse: bool,
    /// Reference timestamp for temporal validity check.
    /// If Some, only records valid at this time are returned.
    /// If None, all records are returned regardless of temporal validity.
    pub reference_ts_millis: Option<TimestampMilli>,
}

/// Scan all reverse edges with pagination.
#[derive(Debug, Clone, Default)]
pub struct AllReverseEdges {
    /// Last edge cursor from previous page (dst_id, src_id, edge_name).
    pub last: Option<(DstId, SrcId, EdgeName)>,
    /// Maximum number of records to return.
    pub limit: usize,
    /// Scan in reverse direction (from end to start).
    pub reverse: bool,
    /// Reference timestamp for temporal validity check.
    /// If Some, only records valid at this time are returned.
    /// If None, all records are returned regardless of temporal validity.
    pub reference_ts_millis: Option<TimestampMilli>,
}

/// Scan all node fragments with pagination.
#[derive(Debug, Clone, Default)]
pub struct AllNodeFragments {
    /// Last fragment cursor from previous page (node_id, timestamp).
    pub last: Option<(Id, TimestampMilli)>,
    /// Maximum number of records to return.
    pub limit: usize,
    /// Scan in reverse direction (from end to start).
    pub reverse: bool,
    /// Reference timestamp for temporal validity check.
    /// If Some, only records valid at this time are returned.
    /// If None, all records are returned regardless of temporal validity.
    pub reference_ts_millis: Option<TimestampMilli>,
}

/// Scan all edge fragments with pagination.
#[derive(Debug, Clone, Default)]
pub struct AllEdgeFragments {
    /// Last fragment cursor from previous page (src_id, dst_id, edge_name, timestamp).
    pub last: Option<(SrcId, DstId, EdgeName, TimestampMilli)>,
    /// Maximum number of records to return.
    pub limit: usize,
    /// Scan in reverse direction (from end to start).
    pub reverse: bool,
    /// Reference timestamp for temporal validity check.
    /// If Some, only records valid at this time are returned.
    /// If None, all records are returned regardless of temporal validity.
    pub reference_ts_millis: Option<TimestampMilli>,
}

/// Scan all node names with pagination.
#[derive(Debug, Clone, Default)]
pub struct AllNodeNames {
    /// Last cursor from previous page (name, node_id).
    pub last: Option<(NodeName, Id)>,
    /// Maximum number of records to return.
    pub limit: usize,
    /// Scan in reverse direction (from end to start).
    pub reverse: bool,
    /// Reference timestamp for temporal validity check.
    /// If Some, only records valid at this time are returned.
    /// If None, all records are returned regardless of temporal validity.
    pub reference_ts_millis: Option<TimestampMilli>,
}

/// Scan all edge names with pagination.
#[derive(Debug, Clone, Default)]
pub struct AllEdgeNames {
    /// Last cursor from previous page (name, src_id, dst_id).
    pub last: Option<(EdgeName, SrcId, DstId)>,
    /// Maximum number of records to return.
    pub limit: usize,
    /// Scan in reverse direction (from end to start).
    pub reverse: bool,
    /// Reference timestamp for temporal validity check.
    /// If Some, only records valid at this time are returned.
    /// If None, all records are returned regardless of temporal validity.
    pub reference_ts_millis: Option<TimestampMilli>,
}

// ============================================================================
// Internal Helpers
// ============================================================================

/// Internal helper to run iteration over a column family.
/// Handles both readonly and readwrite storage modes.
fn iterate_and_visit<CF, R, V, F, G>(
    storage: &Storage,
    seek_key: Vec<u8>,
    limit: usize,
    skip_cursor: bool,
    reverse: bool,
    reference_ts_millis: Option<TimestampMilli>,
    cursor_matches: impl Fn(&[u8]) -> bool,
    transform: F,
    get_valid_range: G,
    visitor: &mut V,
) -> Result<usize>
where
    CF: ColumnFamilyRecord,
    V: Visitor<R>,
    F: Fn(&[u8], &[u8]) -> Result<R>,
    G: Fn(&R) -> &Option<ValidTemporalRange>,
{
    let direction = if reverse {
        Direction::Reverse
    } else {
        Direction::Forward
    };

    let mode = if seek_key.is_empty() {
        if reverse {
            IteratorMode::End
        } else {
            IteratorMode::Start
        }
    } else {
        IteratorMode::From(&seek_key, direction)
    };

    let mut count = 0;
    let mut skipped_cursor = !skip_cursor;

    // Handle readonly/secondary mode
    if let Ok(db) = storage.db() {
        let cf = db.cf_handle(CF::CF_NAME).ok_or_else(|| {
            anyhow::anyhow!("Column family '{}' not found", CF::CF_NAME)
        })?;

        for item in db.iterator_cf(cf, mode) {
            if count >= limit {
                break;
            }

            let (key_bytes, value_bytes) = item?;

            // Skip the cursor record if needed
            if !skipped_cursor && cursor_matches(&key_bytes) {
                skipped_cursor = true;
                continue;
            }
            skipped_cursor = true;

            let record = transform(&key_bytes, &value_bytes)?;

            // Check temporal validity if reference time is specified
            if let Some(ref_ts) = reference_ts_millis {
                if !is_valid_at_time(get_valid_range(&record), ref_ts) {
                    continue; // Skip invalid records but don't count them
                }
            }

            count += 1;
            if !visitor.visit(&record) {
                break;
            }
        }
    } else {
        // Handle readwrite mode (TransactionDB)
        let txn_db = storage.transaction_db()?;
        let cf = txn_db.cf_handle(CF::CF_NAME).ok_or_else(|| {
            anyhow::anyhow!("Column family '{}' not found", CF::CF_NAME)
        })?;

        for item in txn_db.iterator_cf(cf, mode) {
            if count >= limit {
                break;
            }

            let (key_bytes, value_bytes) = item?;

            // Skip the cursor record if needed
            if !skipped_cursor && cursor_matches(&key_bytes) {
                skipped_cursor = true;
                continue;
            }
            skipped_cursor = true;

            let record = transform(&key_bytes, &value_bytes)?;

            // Check temporal validity if reference time is specified
            if let Some(ref_ts) = reference_ts_millis {
                if !is_valid_at_time(get_valid_range(&record), ref_ts) {
                    continue; // Skip invalid records but don't count them
                }
            }

            count += 1;
            if !visitor.visit(&record) {
                break;
            }
        }
    }

    Ok(count)
}

// ============================================================================
// Visitable Implementations
// ============================================================================

impl Visitable for AllNodes {
    type Record = NodeRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        let seek_key = self
            .last
            .map(|id| schema::Nodes::key_to_bytes(&schema::NodeCfKey(id)))
            .unwrap_or_default();

        let last_id = self.last;

        iterate_and_visit::<schema::Nodes, _, _, _, _>(
            storage,
            seek_key,
            self.limit,
            self.last.is_some(),
            self.reverse,
            self.reference_ts_millis,
            |key_bytes| {
                if let Some(last) = last_id {
                    key_bytes == last.into_bytes()
                } else {
                    false
                }
            },
            |key_bytes, value_bytes| {
                let key = schema::Nodes::key_from_bytes(key_bytes)?;
                let value = schema::Nodes::value_from_bytes(value_bytes)?;
                Ok(NodeRecord {
                    id: key.0,
                    name: value.1,
                    summary: value.2,
                    valid_range: value.0,
                })
            },
            |record| &record.valid_range,
            visitor,
        )
    }
}

impl Visitable for AllEdges {
    type Record = EdgeRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        let seek_key = self
            .last
            .as_ref()
            .map(|(src, dst, name)| {
                schema::ForwardEdges::key_to_bytes(&schema::ForwardEdgeCfKey(
                    *src,
                    *dst,
                    name.clone(),
                ))
            })
            .unwrap_or_default();

        let last_cursor = self.last.clone();

        iterate_and_visit::<schema::ForwardEdges, _, _, _, _>(
            storage,
            seek_key,
            self.limit,
            self.last.is_some(),
            self.reverse,
            self.reference_ts_millis,
            |key_bytes| {
                if let Some((src, dst, name)) = &last_cursor {
                    let cursor_key = schema::ForwardEdges::key_to_bytes(
                        &schema::ForwardEdgeCfKey(*src, *dst, name.clone()),
                    );
                    key_bytes == cursor_key.as_slice()
                } else {
                    false
                }
            },
            |key_bytes, value_bytes| {
                let key = schema::ForwardEdges::key_from_bytes(key_bytes)?;
                let value = schema::ForwardEdges::value_from_bytes(value_bytes)?;
                Ok(EdgeRecord {
                    src_id: key.0,
                    dst_id: key.1,
                    name: key.2,
                    summary: value.2,
                    weight: value.1,
                    valid_range: value.0,
                })
            },
            |record| &record.valid_range,
            visitor,
        )
    }
}

impl Visitable for AllReverseEdges {
    type Record = ReverseEdgeRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        let seek_key = self
            .last
            .as_ref()
            .map(|(dst, src, name)| {
                schema::ReverseEdges::key_to_bytes(&schema::ReverseEdgeCfKey(
                    *dst,
                    *src,
                    name.clone(),
                ))
            })
            .unwrap_or_default();

        let last_cursor = self.last.clone();

        iterate_and_visit::<schema::ReverseEdges, _, _, _, _>(
            storage,
            seek_key,
            self.limit,
            self.last.is_some(),
            self.reverse,
            self.reference_ts_millis,
            |key_bytes| {
                if let Some((dst, src, name)) = &last_cursor {
                    let cursor_key = schema::ReverseEdges::key_to_bytes(
                        &schema::ReverseEdgeCfKey(*dst, *src, name.clone()),
                    );
                    key_bytes == cursor_key.as_slice()
                } else {
                    false
                }
            },
            |key_bytes, value_bytes| {
                let key = schema::ReverseEdges::key_from_bytes(key_bytes)?;
                let value = schema::ReverseEdges::value_from_bytes(value_bytes)?;
                Ok(ReverseEdgeRecord {
                    dst_id: key.0,
                    src_id: key.1,
                    name: key.2,
                    valid_range: value.0,
                })
            },
            |record| &record.valid_range,
            visitor,
        )
    }
}

impl Visitable for AllNodeFragments {
    type Record = NodeFragmentRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        let seek_key = self
            .last
            .map(|(id, ts)| {
                schema::NodeFragments::key_to_bytes(&schema::NodeFragmentCfKey(id, ts))
            })
            .unwrap_or_default();

        let last_cursor = self.last;

        iterate_and_visit::<schema::NodeFragments, _, _, _, _>(
            storage,
            seek_key,
            self.limit,
            self.last.is_some(),
            self.reverse,
            self.reference_ts_millis,
            |key_bytes| {
                if let Some((id, ts)) = last_cursor {
                    let cursor_key = schema::NodeFragments::key_to_bytes(
                        &schema::NodeFragmentCfKey(id, ts),
                    );
                    key_bytes == cursor_key.as_slice()
                } else {
                    false
                }
            },
            |key_bytes, value_bytes| {
                let key = schema::NodeFragments::key_from_bytes(key_bytes)?;
                let value = schema::NodeFragments::value_from_bytes(value_bytes)?;
                Ok(NodeFragmentRecord {
                    node_id: key.0,
                    timestamp: key.1,
                    content: value.1,
                    valid_range: value.0,
                })
            },
            |record| &record.valid_range,
            visitor,
        )
    }
}

impl Visitable for AllEdgeFragments {
    type Record = EdgeFragmentRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        let seek_key = self
            .last
            .as_ref()
            .map(|(src, dst, name, ts)| {
                schema::EdgeFragments::key_to_bytes(&schema::EdgeFragmentCfKey(
                    *src,
                    *dst,
                    name.clone(),
                    *ts,
                ))
            })
            .unwrap_or_default();

        let last_cursor = self.last.clone();

        iterate_and_visit::<schema::EdgeFragments, _, _, _, _>(
            storage,
            seek_key,
            self.limit,
            self.last.is_some(),
            self.reverse,
            self.reference_ts_millis,
            |key_bytes| {
                if let Some((src, dst, name, ts)) = &last_cursor {
                    let cursor_key = schema::EdgeFragments::key_to_bytes(
                        &schema::EdgeFragmentCfKey(*src, *dst, name.clone(), *ts),
                    );
                    key_bytes == cursor_key.as_slice()
                } else {
                    false
                }
            },
            |key_bytes, value_bytes| {
                let key = schema::EdgeFragments::key_from_bytes(key_bytes)?;
                let value = schema::EdgeFragments::value_from_bytes(value_bytes)?;
                Ok(EdgeFragmentRecord {
                    src_id: key.0,
                    dst_id: key.1,
                    edge_name: key.2,
                    timestamp: key.3,
                    content: value.1,
                    valid_range: value.0,
                })
            },
            |record| &record.valid_range,
            visitor,
        )
    }
}

impl Visitable for AllNodeNames {
    type Record = NodeNameRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        let seek_key = self
            .last
            .as_ref()
            .map(|(name, id)| {
                schema::NodeNames::key_to_bytes(&schema::NodeNameCfKey(name.clone(), *id))
            })
            .unwrap_or_default();

        let last_cursor = self.last.clone();

        iterate_and_visit::<schema::NodeNames, _, _, _, _>(
            storage,
            seek_key,
            self.limit,
            self.last.is_some(),
            self.reverse,
            self.reference_ts_millis,
            |key_bytes| {
                if let Some((name, id)) = &last_cursor {
                    let cursor_key = schema::NodeNames::key_to_bytes(
                        &schema::NodeNameCfKey(name.clone(), *id),
                    );
                    key_bytes == cursor_key.as_slice()
                } else {
                    false
                }
            },
            |key_bytes, value_bytes| {
                let key = schema::NodeNames::key_from_bytes(key_bytes)?;
                let value = schema::NodeNames::value_from_bytes(value_bytes)?;
                Ok(NodeNameRecord {
                    name: key.0,
                    node_id: key.1,
                    valid_range: value.0,
                })
            },
            |record| &record.valid_range,
            visitor,
        )
    }
}

impl Visitable for AllEdgeNames {
    type Record = EdgeNameRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        let seek_key = self
            .last
            .as_ref()
            .map(|(name, src, dst)| {
                schema::EdgeNames::key_to_bytes(&schema::EdgeNameCfKey(
                    name.clone(),
                    *src,
                    *dst,
                ))
            })
            .unwrap_or_default();

        let last_cursor = self.last.clone();

        iterate_and_visit::<schema::EdgeNames, _, _, _, _>(
            storage,
            seek_key,
            self.limit,
            self.last.is_some(),
            self.reverse,
            self.reference_ts_millis,
            |key_bytes| {
                if let Some((name, src, dst)) = &last_cursor {
                    let cursor_key = schema::EdgeNames::key_to_bytes(
                        &schema::EdgeNameCfKey(name.clone(), *src, *dst),
                    );
                    key_bytes == cursor_key.as_slice()
                } else {
                    false
                }
            },
            |key_bytes, value_bytes| {
                let key = schema::EdgeNames::key_from_bytes(key_bytes)?;
                let value = schema::EdgeNames::value_from_bytes(value_bytes)?;
                Ok(EdgeNameRecord {
                    name: key.0,
                    src_id: key.1,
                    dst_id: key.2,
                    valid_range: value.0,
                })
            },
            |record| &record.valid_range,
            visitor,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::mutation::Runnable;
    use super::super::writer::{create_mutation_writer, spawn_graph_consumer, WriterConfig};
    use super::super::mutation::{AddEdge, AddNode};
    use crate::DataUrl;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_scan_all_nodes() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path();

        // Create and populate database
        let config = WriterConfig {
            channel_buffer_size: 100,
        };
        let (writer, receiver) = create_mutation_writer(config.clone());
        let handle = spawn_graph_consumer(receiver, config, db_path);

        // Add some test nodes
        for i in 0..5 {
            let node = AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                name: format!("test_node_{}", i),
                temporal_range: None,
                summary: super::super::schema::NodeSummary::from_text(&format!("test summary {}", i)),
            };
            node.run(&writer).await.unwrap();
        }

        // Shutdown writer and wait for consumer
        drop(writer);
        handle.await.unwrap().unwrap();

        // Now scan the nodes
        let mut storage = Storage::readonly(db_path);
        storage.ready().unwrap();

        let scan = AllNodes {
            last: None,
            limit: 100,
            ..Default::default()
        };

        let count = AtomicUsize::new(0);
        let visited = scan
            .accept(&storage, &mut |record: &NodeRecord| {
                count.fetch_add(1, Ordering::SeqCst);
                assert!(record.name.starts_with("test_node_"));
                true
            })
            .unwrap();

        assert_eq!(visited, 5);
        assert_eq!(count.load(Ordering::SeqCst), 5);
    }

    #[tokio::test]
    async fn test_scan_with_pagination() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path();

        // Create and populate database
        let config = WriterConfig {
            channel_buffer_size: 100,
        };
        let (writer, receiver) = create_mutation_writer(config.clone());
        let handle = spawn_graph_consumer(receiver, config, db_path);

        // Add some test nodes
        let mut node_ids = Vec::new();
        for i in 0..10 {
            let id = Id::new();
            node_ids.push(id);
            let node = AddNode {
                id,
                ts_millis: TimestampMilli::now(),
                name: format!("test_node_{}", i),
                temporal_range: None,
                summary: super::super::schema::NodeSummary::from_text(&format!("test summary {}", i)),
            };
            node.run(&writer).await.unwrap();
            tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;
        }

        // Shutdown writer and wait for consumer
        drop(writer);
        handle.await.unwrap().unwrap();

        // Now scan with pagination
        let mut storage = Storage::readonly(db_path);
        storage.ready().unwrap();

        // First page: 3 records
        let scan = AllNodes {
            last: None,
            limit: 3,
            ..Default::default()
        };

        let mut first_page_ids = Vec::new();
        scan.accept(&storage, &mut |record: &NodeRecord| {
            first_page_ids.push(record.id);
            true
        })
        .unwrap();

        assert_eq!(first_page_ids.len(), 3);

        // Second page: next 3 records, starting after the last one from first page
        let scan = AllNodes {
            last: Some(*first_page_ids.last().unwrap()),
            limit: 3,
            ..Default::default()
        };

        let mut second_page_ids = Vec::new();
        scan.accept(&storage, &mut |record: &NodeRecord| {
            second_page_ids.push(record.id);
            true
        })
        .unwrap();

        assert_eq!(second_page_ids.len(), 3);

        // Ensure no overlap
        for id in &second_page_ids {
            assert!(!first_page_ids.contains(id));
        }
    }

    #[tokio::test]
    async fn test_visitor_early_stop() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path();

        // Create and populate database
        let config = WriterConfig {
            channel_buffer_size: 100,
        };
        let (writer, receiver) = create_mutation_writer(config.clone());
        let handle = spawn_graph_consumer(receiver, config, db_path);

        // Add some test nodes
        for i in 0..10 {
            let node = AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                name: format!("test_node_{}", i),
                temporal_range: None,
                summary: super::super::schema::NodeSummary::from_text(&format!("test summary {}", i)),
            };
            node.run(&writer).await.unwrap();
        }

        // Shutdown writer and wait for consumer
        drop(writer);
        handle.await.unwrap().unwrap();

        // Scan but stop after 3 records
        let mut storage = Storage::readonly(db_path);
        storage.ready().unwrap();

        let scan = AllNodes {
            last: None,
            limit: 100,
            ..Default::default()
        };

        let mut count = 0;
        let visited = scan
            .accept(&storage, &mut |_record: &NodeRecord| {
                count += 1;
                count < 3 // Stop after 3
            })
            .unwrap();

        assert_eq!(visited, 3);
        assert_eq!(count, 3);
    }

    #[tokio::test]
    async fn test_scan_all_edges() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path();

        // Create and populate database
        let config = WriterConfig {
            channel_buffer_size: 100,
        };
        let (writer, receiver) = create_mutation_writer(config.clone());
        let handle = spawn_graph_consumer(receiver, config, db_path);

        // Add some test nodes first
        let node1 = Id::new();
        let node2 = Id::new();
        let node3 = Id::new();

        for (id, name) in [(node1, "node1"), (node2, "node2"), (node3, "node3")] {
            let node = AddNode {
                id,
                ts_millis: TimestampMilli::now(),
                name: name.to_string(),
                temporal_range: None,
                summary: super::super::schema::NodeSummary::from_text(&format!("{} summary", name)),
            };
            node.run(&writer).await.unwrap();
        }

        // Add edges between nodes
        for (src, dst, name) in [
            (node1, node2, "edge_1_2"),
            (node1, node3, "edge_1_3"),
            (node2, node3, "edge_2_3"),
        ] {
            let edge = AddEdge {
                source_node_id: src,
                target_node_id: dst,
                ts_millis: TimestampMilli::now(),
                name: name.to_string(),
                weight: Some(1.0),
                summary: DataUrl::from_text("test edge"),
                temporal_range: None,
            };
            edge.run(&writer).await.unwrap();
        }

        // Shutdown writer and wait for consumer
        drop(writer);
        handle.await.unwrap().unwrap();

        // Scan forward edges
        let mut storage = Storage::readonly(db_path);
        storage.ready().unwrap();

        let scan = AllEdges {
            last: None,
            limit: 100,
            ..Default::default()
        };

        let mut edge_count = 0;
        scan.accept(&storage, &mut |record: &EdgeRecord| {
            edge_count += 1;
            assert!(record.name.starts_with("edge_"));
            assert!(record.weight.is_some());
            true
        })
        .unwrap();

        assert_eq!(edge_count, 3);
    }

    #[tokio::test]
    async fn test_scan_reverse_edges() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path();

        // Create and populate database
        let config = WriterConfig {
            channel_buffer_size: 100,
        };
        let (writer, receiver) = create_mutation_writer(config.clone());
        let handle = spawn_graph_consumer(receiver, config, db_path);

        // Add some test nodes first
        let node1 = Id::new();
        let node2 = Id::new();
        let node3 = Id::new();

        for (id, name) in [(node1, "node1"), (node2, "node2"), (node3, "node3")] {
            let node = AddNode {
                id,
                ts_millis: TimestampMilli::now(),
                name: name.to_string(),
                temporal_range: None,
                summary: super::super::schema::NodeSummary::from_text(&format!("{} summary", name)),
            };
            node.run(&writer).await.unwrap();
        }

        // Add edges: node1 -> node2, node1 -> node3, node2 -> node3
        // So node3 has 2 incoming edges, node2 has 1 incoming edge
        for (src, dst, name) in [
            (node1, node2, "edge_1_2"),
            (node1, node3, "edge_1_3"),
            (node2, node3, "edge_2_3"),
        ] {
            let edge = AddEdge {
                source_node_id: src,
                target_node_id: dst,
                ts_millis: TimestampMilli::now(),
                name: name.to_string(),
                weight: Some(1.0),
                summary: DataUrl::from_text("test edge"),
                temporal_range: None,
            };
            edge.run(&writer).await.unwrap();
        }

        // Shutdown writer and wait for consumer
        drop(writer);
        handle.await.unwrap().unwrap();

        // Scan reverse edges (incoming edges index)
        let mut storage = Storage::readonly(db_path);
        storage.ready().unwrap();

        let scan = AllReverseEdges {
            last: None,
            limit: 100,
            ..Default::default()
        };

        let mut reverse_edge_count = 0;
        let mut edges_to_node3 = 0;
        scan.accept(&storage, &mut |record: &ReverseEdgeRecord| {
            reverse_edge_count += 1;
            if record.dst_id == node3 {
                edges_to_node3 += 1;
            }
            true
        })
        .unwrap();

        // Should have 3 reverse edges total
        assert_eq!(reverse_edge_count, 3);
        // node3 should have 2 incoming edges
        assert_eq!(edges_to_node3, 2);
    }

    #[tokio::test]
    async fn test_scan_edges_pagination() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path();

        // Create and populate database
        let config = WriterConfig {
            channel_buffer_size: 100,
        };
        let (writer, receiver) = create_mutation_writer(config.clone());
        let handle = spawn_graph_consumer(receiver, config, db_path);

        // Add test nodes
        let mut nodes = Vec::new();
        for i in 0..5 {
            let id = Id::new();
            nodes.push(id);
            let node = AddNode {
                id,
                ts_millis: TimestampMilli::now(),
                name: format!("node_{}", i),
                temporal_range: None,
                summary: super::super::schema::NodeSummary::from_text(&format!("node {} summary", i)),
            };
            node.run(&writer).await.unwrap();
            tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;
        }

        // Add edges between consecutive nodes (4 edges total)
        for i in 0..4 {
            let edge = AddEdge {
                source_node_id: nodes[i],
                target_node_id: nodes[i + 1],
                ts_millis: TimestampMilli::now(),
                name: format!("edge_{}_{}", i, i + 1),
                weight: Some(i as f64),
                summary: DataUrl::from_text("test"),
                temporal_range: None,
            };
            edge.run(&writer).await.unwrap();
            tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;
        }

        // Shutdown writer and wait for consumer
        drop(writer);
        handle.await.unwrap().unwrap();

        // Scan with pagination
        let mut storage = Storage::readonly(db_path);
        storage.ready().unwrap();

        // First page: 2 edges
        let scan = AllEdges {
            last: None,
            limit: 2,
            ..Default::default()
        };

        let mut first_page: Vec<(Id, Id, String)> = Vec::new();
        scan.accept(&storage, &mut |record: &EdgeRecord| {
            first_page.push((record.src_id, record.dst_id, record.name.clone()));
            true
        })
        .unwrap();

        assert_eq!(first_page.len(), 2);

        // Second page: next 2 edges
        let last = first_page.last().unwrap();
        let scan = AllEdges {
            last: Some((last.0, last.1, last.2.clone())),
            limit: 2,
            ..Default::default()
        };

        let mut second_page: Vec<(Id, Id, String)> = Vec::new();
        scan.accept(&storage, &mut |record: &EdgeRecord| {
            second_page.push((record.src_id, record.dst_id, record.name.clone()));
            true
        })
        .unwrap();

        assert_eq!(second_page.len(), 2);

        // Ensure no overlap
        for edge in &second_page {
            assert!(!first_page.contains(edge));
        }
    }

    #[tokio::test]
    async fn test_scan_reverse_direction() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path();

        // Create and populate database
        let config = WriterConfig {
            channel_buffer_size: 100,
        };
        let (writer, receiver) = create_mutation_writer(config.clone());
        let handle = spawn_graph_consumer(receiver, config, db_path);

        // Add some test nodes with delays to ensure different IDs
        let mut node_ids = Vec::new();
        for i in 0..5 {
            let id = Id::new();
            node_ids.push(id);
            let node = AddNode {
                id,
                ts_millis: TimestampMilli::now(),
                name: format!("test_node_{}", i),
                temporal_range: None,
                summary: super::super::schema::NodeSummary::from_text(&format!("test summary {}", i)),
            };
            node.run(&writer).await.unwrap();
            tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;
        }

        // Shutdown writer and wait for consumer
        drop(writer);
        handle.await.unwrap().unwrap();

        // Scan in forward direction
        let mut storage = Storage::readonly(db_path);
        storage.ready().unwrap();

        let scan = AllNodes {
            last: None,
            limit: 100,
            reverse: false,
            reference_ts_millis: None,
        };

        let mut forward_ids = Vec::new();
        scan.accept(&storage, &mut |record: &NodeRecord| {
            forward_ids.push(record.id);
            true
        })
        .unwrap();

        assert_eq!(forward_ids.len(), 5);

        // Scan in reverse direction
        let scan = AllNodes {
            last: None,
            limit: 100,
            reverse: true,
            reference_ts_millis: None,
        };

        let mut reverse_ids = Vec::new();
        scan.accept(&storage, &mut |record: &NodeRecord| {
            reverse_ids.push(record.id);
            true
        })
        .unwrap();

        assert_eq!(reverse_ids.len(), 5);

        // Reverse scan should return IDs in opposite order
        let mut forward_reversed = forward_ids.clone();
        forward_reversed.reverse();
        assert_eq!(reverse_ids, forward_reversed);
    }

    #[tokio::test]
    async fn test_scan_reverse_with_pagination() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path();

        // Create and populate database
        let config = WriterConfig {
            channel_buffer_size: 100,
        };
        let (writer, receiver) = create_mutation_writer(config.clone());
        let handle = spawn_graph_consumer(receiver, config, db_path);

        // Add test nodes
        for i in 0..10 {
            let id = Id::new();
            let node = AddNode {
                id,
                ts_millis: TimestampMilli::now(),
                name: format!("test_node_{}", i),
                temporal_range: None,
                summary: super::super::schema::NodeSummary::from_text(&format!("test summary {}", i)),
            };
            node.run(&writer).await.unwrap();
            tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;
        }

        // Shutdown writer and wait for consumer
        drop(writer);
        handle.await.unwrap().unwrap();

        let mut storage = Storage::readonly(db_path);
        storage.ready().unwrap();

        // First page in reverse: 3 records
        let scan = AllNodes {
            last: None,
            limit: 3,
            reverse: true,
            reference_ts_millis: None,
        };

        let mut first_page_ids = Vec::new();
        scan.accept(&storage, &mut |record: &NodeRecord| {
            first_page_ids.push(record.id);
            true
        })
        .unwrap();

        assert_eq!(first_page_ids.len(), 3);

        // Second page in reverse: next 3 records
        let scan = AllNodes {
            last: Some(*first_page_ids.last().unwrap()),
            limit: 3,
            reverse: true,
            reference_ts_millis: None,
        };

        let mut second_page_ids = Vec::new();
        scan.accept(&storage, &mut |record: &NodeRecord| {
            second_page_ids.push(record.id);
            true
        })
        .unwrap();

        assert_eq!(second_page_ids.len(), 3);

        // Ensure no overlap
        for id in &second_page_ids {
            assert!(!first_page_ids.contains(id));
        }
    }
}
