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

use super::name_hash::NameHash;
use super::summary_hash::SummaryHash;
use super::{ColumnFamily, ColumnFamilySerde, HotColumnFamilyRecord};
use super::schema::{
    self, is_active_at_time, DstId, EdgeName, EdgeSummary, EdgeSummaries, EdgeSummaryCfKey,
    EdgeWeight, FragmentContent, Names, NameCfKey, NodeName, NodeSummary, NodeSummaries,
    NodeSummaryCfKey, SrcId, ActivePeriod, Version,
    // Scan: summary index CFs
    NodeSummaryIndex, NodeSummaryIndexCfKey, NodeSummaryIndexCfValue,
    EdgeSummaryIndex, EdgeSummaryIndexCfKey, EdgeSummaryIndexCfValue,
    // Scan: version history CFs
    NodeVersionHistory, NodeVersionHistoryCfKey,
    EdgeVersionHistory, EdgeVersionHistoryCfKey,
    // Scan: orphan + meta CFs
    OrphanSummaries, OrphanSummaryCfKey, SummaryKind,
    GraphMeta,
};
use super::Storage;
use crate::{ActiveTimeMillis, Id, TimestampMilli};

// ============================================================================
// Name Resolution Helpers
// ============================================================================

/// Resolve a NameHash to its full String name.
///
/// Uses the in-memory NameCache first for O(1) lookup, falling back to
/// Names CF lookup only for cache misses. On cache miss, the name is
/// added to the cache for future lookups.
fn resolve_name(storage: &Storage, name_hash: NameHash) -> Result<String> {
    // Check cache first (O(1) DashMap lookup)
    let cache = storage.cache();
    if let Some(name) = cache.get(&name_hash) {
        return Ok((*name).clone());
    }

    // Cache miss: fetch from Names CF
    let key_bytes = Names::key_to_bytes(&NameCfKey(name_hash));

    let value_bytes = if let Ok(db) = storage.db() {
        let names_cf = db
            .cf_handle(Names::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Names CF not found"))?;
        db.get_cf(names_cf, &key_bytes)?
    } else {
        let txn_db = storage.transaction_db()?;
        let names_cf = txn_db
            .cf_handle(Names::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Names CF not found"))?;
        txn_db.get_cf(names_cf, &key_bytes)?
    };

    let value_bytes = value_bytes
        .ok_or_else(|| anyhow::anyhow!("Name not found for hash: {}", name_hash))?;

    let value = Names::value_from_bytes(&value_bytes)?;
    let name = value.0;

    // Populate cache for future lookups
    cache.insert(name_hash, name.clone());

    Ok(name)
}

// ============================================================================
// Summary Resolution Helpers
// ============================================================================

/// Resolve a SummaryHash to its full NodeSummary from the cold CF.
///
/// Returns an empty summary if the hash is None or the summary is not found.
fn resolve_node_summary(storage: &Storage, summary_hash: Option<SummaryHash>) -> Result<NodeSummary> {
    let Some(hash) = summary_hash else {
        return Ok(NodeSummary::from_text(""));
    };

    let key_bytes = NodeSummaries::key_to_bytes(&NodeSummaryCfKey(hash));

    let value_bytes = if let Ok(db) = storage.db() {
        let summaries_cf = db
            .cf_handle(NodeSummaries::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("NodeSummaries CF not found"))?;
        db.get_cf(summaries_cf, &key_bytes)?
    } else {
        let txn_db = storage.transaction_db()?;
        let summaries_cf = txn_db
            .cf_handle(NodeSummaries::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("NodeSummaries CF not found"))?;
        txn_db.get_cf(summaries_cf, &key_bytes)?
    };

    match value_bytes {
        Some(bytes) => {
            let value = NodeSummaries::value_from_bytes(&bytes)?;
            Ok(value.0) // .0 is the summary (no refcount in VERSIONING)
        }
        None => Ok(NodeSummary::from_text("")),
    }
}

/// Resolve a SummaryHash to its full EdgeSummary from the cold CF.
///
/// Returns an empty summary if the hash is None or the summary is not found.
fn resolve_edge_summary(storage: &Storage, summary_hash: Option<SummaryHash>) -> Result<EdgeSummary> {
    let Some(hash) = summary_hash else {
        return Ok(EdgeSummary::from_text(""));
    };

    let key_bytes = EdgeSummaries::key_to_bytes(&EdgeSummaryCfKey(hash));

    let value_bytes = if let Ok(db) = storage.db() {
        let summaries_cf = db
            .cf_handle(EdgeSummaries::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("EdgeSummaries CF not found"))?;
        db.get_cf(summaries_cf, &key_bytes)?
    } else {
        let txn_db = storage.transaction_db()?;
        let summaries_cf = txn_db
            .cf_handle(EdgeSummaries::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("EdgeSummaries CF not found"))?;
        txn_db.get_cf(summaries_cf, &key_bytes)?
    };

    match value_bytes {
        Some(bytes) => {
            let value = EdgeSummaries::value_from_bytes(&bytes)?;
            Ok(value.0) // .0 is the summary (no refcount in VERSIONING)
        }
        None => Ok(EdgeSummary::from_text("")),
    }
}

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
    pub valid_range: Option<ActivePeriod>,
    pub version: Version,
}

/// A forward edge record as seen by scan visitors.
#[derive(Debug, Clone)]
pub struct EdgeRecord {
    pub src_id: SrcId,
    pub dst_id: DstId,
    pub name: EdgeName,
    pub summary: EdgeSummary,
    pub weight: Option<schema::EdgeWeight>,
    pub valid_range: Option<ActivePeriod>,
    pub version: Version,
}

/// A reverse edge record as seen by scan visitors (index only, no summary/weight).
#[derive(Debug, Clone)]
pub struct ReverseEdgeRecord {
    pub dst_id: DstId,
    pub src_id: SrcId,
    pub name: EdgeName,
    pub valid_range: Option<ActivePeriod>,
}

/// A node fragment record as seen by scan visitors.
#[derive(Debug, Clone)]
pub struct NodeFragmentRecord {
    pub node_id: Id,
    pub timestamp: TimestampMilli,
    pub content: FragmentContent,
    pub valid_range: Option<ActivePeriod>,
}

/// An edge fragment record as seen by scan visitors.
#[derive(Debug, Clone)]
pub struct EdgeFragmentRecord {
    pub src_id: SrcId,
    pub dst_id: DstId,
    pub edge_name: EdgeName,
    pub timestamp: TimestampMilli,
    pub content: FragmentContent,
    pub valid_range: Option<ActivePeriod>,
}

/// A name mapping record (NameHash → String).
#[derive(Debug, Clone)]
pub struct NameRecord {
    pub hash: String,
    pub name: String,
}

/// A node summary record from the cold content store.
#[derive(Debug, Clone)]
pub struct NodeSummaryRecord {
    pub hash: String,
    pub content: NodeSummary,
}

/// An edge summary record from the cold content store.
#[derive(Debug, Clone)]
pub struct EdgeSummaryRecord {
    pub hash: String,
    pub content: EdgeSummary,
}

/// A node summary index record (reverse lookup: hash → node).
#[derive(Debug, Clone)]
pub struct NodeSummaryIndexRecord {
    pub hash: String,
    pub node_id: Id,
    pub version: Version,
    pub status: String,
}

/// An edge summary index record (reverse lookup: hash → edge).
#[derive(Debug, Clone)]
pub struct EdgeSummaryIndexRecord {
    pub hash: String,
    pub src_id: SrcId,
    pub dst_id: DstId,
    pub edge_name: String,
    pub version: Version,
    pub status: String,
}

/// A node version history record (version snapshot for rollback).
#[derive(Debug, Clone)]
pub struct NodeVersionHistoryRecord {
    pub node_id: Id,
    pub valid_since: TimestampMilli,
    pub version: Version,
    pub updated_at: TimestampMilli,
    pub summary_hash: Option<String>,
    pub name: String,
    pub active_period: Option<ActivePeriod>,
}

/// An edge version history record (version snapshot for rollback).
#[derive(Debug, Clone)]
pub struct EdgeVersionHistoryRecord {
    pub src_id: SrcId,
    pub dst_id: DstId,
    pub edge_name: String,
    pub valid_since: TimestampMilli,
    pub version: Version,
    pub updated_at: TimestampMilli,
    pub summary_hash: Option<String>,
    pub weight: Option<EdgeWeight>,
    pub active_period: Option<ActivePeriod>,
}

/// An orphan summary record (GC tracking).
#[derive(Debug, Clone)]
pub struct OrphanSummaryRecord {
    pub orphaned_at: TimestampMilli,
    pub hash: String,
    pub kind: String,
}

/// A graph metadata record (GC cursors).
#[derive(Debug, Clone)]
pub struct GraphMetaRecord {
    pub field: String,
    pub cursor_bytes_hex: String,
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
    pub reference_ts_millis: Option<ActiveTimeMillis>,
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
    pub reference_ts_millis: Option<ActiveTimeMillis>,
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
    pub reference_ts_millis: Option<ActiveTimeMillis>,
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
    pub reference_ts_millis: Option<ActiveTimeMillis>,
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
    pub reference_ts_millis: Option<ActiveTimeMillis>,
}

/// Scan all name mappings.
#[derive(Debug, Clone, Default)]
pub struct AllNames {
    /// Last NameHash hex from previous page (exclusive start for pagination).
    pub last: Option<NameHash>,
    pub limit: usize,
    pub reverse: bool,
}

/// Scan all node summaries (cold content store).
#[derive(Debug, Clone, Default)]
pub struct AllNodeSummaries {
    /// Last SummaryHash hex from previous page.
    pub last: Option<SummaryHash>,
    pub limit: usize,
    pub reverse: bool,
}

/// Scan all edge summaries (cold content store).
#[derive(Debug, Clone, Default)]
pub struct AllEdgeSummaries {
    /// Last SummaryHash hex from previous page.
    pub last: Option<SummaryHash>,
    pub limit: usize,
    pub reverse: bool,
}

/// Scan all node summary index entries (reverse lookup).
#[derive(Debug, Clone, Default)]
pub struct AllNodeSummaryIndex {
    pub last: Option<(SummaryHash, Id, Version)>,
    pub limit: usize,
    pub reverse: bool,
}

/// Scan all edge summary index entries (reverse lookup).
#[derive(Debug, Clone, Default)]
pub struct AllEdgeSummaryIndex {
    pub last: Option<(SummaryHash, SrcId, DstId, NameHash, Version)>,
    pub limit: usize,
    pub reverse: bool,
}

/// Scan all node version history entries.
#[derive(Debug, Clone, Default)]
pub struct AllNodeVersionHistory {
    pub last: Option<(Id, TimestampMilli, Version)>,
    pub limit: usize,
    pub reverse: bool,
}

/// Scan all edge version history entries.
#[derive(Debug, Clone, Default)]
pub struct AllEdgeVersionHistory {
    pub last: Option<(SrcId, DstId, NameHash, TimestampMilli, Version)>,
    pub limit: usize,
    pub reverse: bool,
}

/// Scan all orphan summary entries (GC tracking).
#[derive(Debug, Clone, Default)]
pub struct AllOrphanSummaries {
    pub last: Option<(TimestampMilli, SummaryHash)>,
    pub limit: usize,
    pub reverse: bool,
}

/// Scan all graph metadata entries.
#[derive(Debug, Clone, Default)]
pub struct AllGraphMeta {
    pub limit: usize,
    pub reverse: bool,
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
    reference_ts_millis: Option<ActiveTimeMillis>,
    cursor_matches: impl Fn(&[u8]) -> bool,
    transform: F,
    get_valid_range: G,
    visitor: &mut V,
) -> Result<usize>
where
    CF: ColumnFamily,
    V: Visitor<R>,
    F: Fn(&[u8], &[u8]) -> Result<R>,
    G: Fn(&R) -> &Option<ActivePeriod>,
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
    // Note: skip_cursor parameter is used by cursor_matches closure, not by this variable
    let _ = skip_cursor;

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

            // VERSIONING: Always check cursor_matches to skip all versions of cursor record
            if cursor_matches(&key_bytes) {
                continue;
            }

            // VERSIONING: transform may return skip errors for non-current versions
            let record = match transform(&key_bytes, &value_bytes) {
                Ok(r) => r,
                Err(e) => {
                    let msg = e.to_string();
                    if msg.starts_with("skip_") {
                        continue;
                    }
                    return Err(e);
                }
            };

            // Check temporal validity if reference time is specified
            if let Some(ref_ts) = reference_ts_millis {
                if !is_active_at_time(get_valid_range(&record), ref_ts) {
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

            // VERSIONING: Always check cursor_matches to skip all versions of cursor record
            if cursor_matches(&key_bytes) {
                continue;
            }

            // VERSIONING: transform may return skip errors for non-current versions
            let record = match transform(&key_bytes, &value_bytes) {
                Ok(r) => r,
                Err(e) => {
                    let msg = e.to_string();
                    if msg.starts_with("skip_") {
                        continue;
                    }
                    return Err(e);
                }
            };

            // Check temporal validity if reference time is specified
            if let Some(ref_ts) = reference_ts_millis {
                if !is_active_at_time(get_valid_range(&record), ref_ts) {
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

/// Internal helper to run iteration over a hot column family (rkyv serialized).
/// Same as `iterate_and_visit` but for HotColumnFamilyRecord types.
fn iterate_and_visit_hot<CF, R, V, F, G>(
    storage: &Storage,
    seek_key: Vec<u8>,
    limit: usize,
    skip_cursor: bool,
    reverse: bool,
    reference_ts_millis: Option<ActiveTimeMillis>,
    cursor_matches: impl Fn(&[u8]) -> bool,
    transform: F,
    get_valid_range: G,
    visitor: &mut V,
) -> Result<usize>
where
    CF: HotColumnFamilyRecord,
    V: Visitor<R>,
    F: Fn(&[u8], &[u8]) -> Result<R>,
    G: Fn(&R) -> &Option<ActivePeriod>,
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
    // Note: skip_cursor parameter is used by cursor_matches closure, not by this variable
    let _ = skip_cursor;

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

            // VERSIONING: Always check cursor_matches to skip all versions of cursor record
            if cursor_matches(&key_bytes) {
                continue;
            }

            // VERSIONING: transform may return skip errors for non-current versions
            let record = match transform(&key_bytes, &value_bytes) {
                Ok(r) => r,
                Err(e) => {
                    let msg = e.to_string();
                    if msg.starts_with("skip_") {
                        continue;
                    }
                    return Err(e);
                }
            };

            // Check temporal validity if reference time is specified
            if let Some(ref_ts) = reference_ts_millis {
                if !is_active_at_time(get_valid_range(&record), ref_ts) {
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

            // VERSIONING: Always check cursor_matches to skip all versions of cursor record
            if cursor_matches(&key_bytes) {
                continue;
            }

            // VERSIONING: transform may return skip errors for non-current versions
            let record = match transform(&key_bytes, &value_bytes) {
                Ok(r) => r,
                Err(e) => {
                    let msg = e.to_string();
                    if msg.starts_with("skip_") {
                        continue;
                    }
                    return Err(e);
                }
            };

            // Check temporal validity if reference time is specified
            if let Some(ref_ts) = reference_ts_millis {
                if !is_active_at_time(get_valid_range(&record), ref_ts) {
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

    /// (claude, 2026-02-06, in-progress: VERSIONING uses node_id prefix for seek)
    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        tracing::debug!(limit = self.limit, reverse = self.reverse, has_cursor = self.last.is_some(), "Executing AllNodes scan");

        // With VERSIONING, key is (Id, ValidSince). For cursor positioning,
        // we seek past all versions of the cursor ID by using max timestamp.
        let seek_key = self
            .last
            .map(|id| {
                let mut bytes = id.into_bytes().to_vec();
                bytes.extend_from_slice(&u64::MAX.to_be_bytes());
                bytes
            })
            .unwrap_or_default();

        let cursor_id = self.last;

        iterate_and_visit_hot::<schema::Nodes, _, _, _, _>(
            storage,
            seek_key,
            self.limit,
            false, // Don't skip first - we handle cursor skipping explicitly below
            self.reverse,
            self.reference_ts_millis,
            |key_bytes| {
                // VERSIONING: Skip all versions of the cursor node_id.
                // For forward iteration, seek_key = (cursor_id, MAX) puts us past the cursor.
                // For reverse iteration, we might land on cursor versions and need to skip them.
                if let Some(cursor) = cursor_id {
                    if key_bytes.len() >= 16 {
                        let mut id_bytes = [0u8; 16];
                        id_bytes.copy_from_slice(&key_bytes[0..16]);
                        let node_id = crate::Id::from_bytes(id_bytes);
                        if node_id == cursor {
                            return true; // Skip cursor node
                        }
                    }
                }
                false
            },
            |key_bytes, value_bytes| {
                let key = schema::Nodes::key_from_bytes(key_bytes)?;
                let value = schema::Nodes::value_from_bytes(value_bytes)?;

                // Field indices (VERSIONING): 0=ValidUntil, 1=ActivePeriod, 2=NameHash, 3=SummaryHash, 4=Version, 5=Deleted
                // Only process current versions (ValidUntil = None)
                if value.0.is_some() {
                    return Err(anyhow::anyhow!("skip_non_current"));
                }
                // Skip deleted nodes
                if value.5 {
                    return Err(anyhow::anyhow!("skip_deleted"));
                }

                // Resolve NameHash to String (index 2)
                let name = resolve_name(storage, value.2)?;
                // Resolve SummaryHash to NodeSummary (index 3)
                let summary = resolve_node_summary(storage, value.3)?;
                Ok(NodeRecord {
                    id: key.0,
                    name,
                    summary,
                    valid_range: value.1, // ActivePeriod at index 1
                    version: value.4,
                })
            },
            |record| &record.valid_range,
            visitor,
        )
    }
}

impl Visitable for AllEdges {
    type Record = EdgeRecord;

    /// (claude, 2026-02-06, in-progress: VERSIONING uses edge topology prefix for seek)
    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        tracing::debug!(limit = self.limit, reverse = self.reverse, has_cursor = self.last.is_some(), "Executing AllEdges scan");

        // With VERSIONING, key is (SrcId, DstId, NameHash, ValidSince). For cursor positioning,
        // we seek past all versions of the cursor edge by using max timestamp.
        let seek_key = self
            .last
            .as_ref()
            .map(|(src, dst, name)| {
                let name_hash = NameHash::from_name(name);
                let mut bytes = Vec::with_capacity(48);
                bytes.extend_from_slice(&src.into_bytes());
                bytes.extend_from_slice(&dst.into_bytes());
                bytes.extend_from_slice(name_hash.as_bytes());
                bytes.extend_from_slice(&u64::MAX.to_be_bytes());
                bytes
            })
            .unwrap_or_default();

        // Track last seen edge topology for deduplication
        let last_seen_edge: Option<(crate::Id, crate::Id, NameHash)> = None;

        iterate_and_visit_hot::<schema::ForwardEdges, _, _, _, _>(
            storage,
            seek_key,
            self.limit,
            false, // Don't skip first - our seek key is already past cursor
            self.reverse,
            self.reference_ts_millis,
            |key_bytes| {
                // With VERSIONING, check if this is a duplicate edge topology
                // by comparing first 40 bytes (src_id + dst_id + name_hash)
                if key_bytes.len() >= 40 {
                    let mut src_bytes = [0u8; 16];
                    src_bytes.copy_from_slice(&key_bytes[0..16]);
                    let mut dst_bytes = [0u8; 16];
                    dst_bytes.copy_from_slice(&key_bytes[16..32]);
                    let mut name_bytes = [0u8; 8];
                    name_bytes.copy_from_slice(&key_bytes[32..40]);

                    let src_id = crate::Id::from_bytes(src_bytes);
                    let dst_id = crate::Id::from_bytes(dst_bytes);
                    let name_hash = NameHash::from_bytes(name_bytes);

                    if last_seen_edge == Some((src_id, dst_id, name_hash)) {
                        return true; // Skip duplicates
                    }
                }
                false
            },
            |key_bytes, value_bytes| {
                let key = schema::ForwardEdges::key_from_bytes(key_bytes)?;
                let value = schema::ForwardEdges::value_from_bytes(value_bytes)?;

                // Field indices (VERSIONING): 0=ValidUntil, 1=ActivePeriod, 2=Weight, 3=SummaryHash, 4=Version, 5=Deleted
                // Only process current versions (ValidUntil = None)
                if value.0.is_some() {
                    return Err(anyhow::anyhow!("skip_non_current"));
                }
                // Skip deleted edges
                if value.5 {
                    return Err(anyhow::anyhow!("skip_deleted"));
                }

                // Resolve NameHash to String
                let name = resolve_name(storage, key.2)?;
                // Resolve SummaryHash to EdgeSummary (index 3)
                let summary = resolve_edge_summary(storage, value.3)?;
                Ok(EdgeRecord {
                    src_id: key.0,
                    dst_id: key.1,
                    name,
                    summary,
                    weight: value.2,           // Weight at index 2
                    valid_range: value.1,      // ActivePeriod at index 1
                    version: value.4,
                })
            },
            |record| &record.valid_range,
            visitor,
        )
    }
}

impl Visitable for AllReverseEdges {
    type Record = ReverseEdgeRecord;

    /// (claude, 2026-02-06, in-progress: VERSIONING uses edge topology prefix for seek)
    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        tracing::debug!(limit = self.limit, reverse = self.reverse, has_cursor = self.last.is_some(), "Executing AllReverseEdges scan");

        // With VERSIONING, key is (DstId, SrcId, NameHash, ValidSince). For cursor positioning,
        // we seek past all versions of the cursor edge by using max timestamp.
        let seek_key = self
            .last
            .as_ref()
            .map(|(dst, src, name)| {
                let name_hash = NameHash::from_name(name);
                let mut bytes = Vec::with_capacity(48);
                bytes.extend_from_slice(&dst.into_bytes());
                bytes.extend_from_slice(&src.into_bytes());
                bytes.extend_from_slice(name_hash.as_bytes());
                bytes.extend_from_slice(&u64::MAX.to_be_bytes());
                bytes
            })
            .unwrap_or_default();

        // Track last seen edge topology for deduplication
        let last_seen_edge: Option<(crate::Id, crate::Id, NameHash)> = None;

        iterate_and_visit_hot::<schema::ReverseEdges, _, _, _, _>(
            storage,
            seek_key,
            self.limit,
            false, // Don't skip first - our seek key is already past cursor
            self.reverse,
            self.reference_ts_millis,
            |key_bytes| {
                // With VERSIONING, check if this is a duplicate edge topology
                // by comparing first 40 bytes (dst_id + src_id + name_hash)
                if key_bytes.len() >= 40 {
                    let mut dst_bytes = [0u8; 16];
                    dst_bytes.copy_from_slice(&key_bytes[0..16]);
                    let mut src_bytes = [0u8; 16];
                    src_bytes.copy_from_slice(&key_bytes[16..32]);
                    let mut name_bytes = [0u8; 8];
                    name_bytes.copy_from_slice(&key_bytes[32..40]);

                    let dst_id = crate::Id::from_bytes(dst_bytes);
                    let src_id = crate::Id::from_bytes(src_bytes);
                    let name_hash = NameHash::from_bytes(name_bytes);

                    if last_seen_edge == Some((dst_id, src_id, name_hash)) {
                        return true; // Skip duplicates
                    }
                }
                false
            },
            |key_bytes, value_bytes| {
                let key = schema::ReverseEdges::key_from_bytes(key_bytes)?;
                let value = schema::ReverseEdges::value_from_bytes(value_bytes)?;

                // Field indices (VERSIONING): 0=ValidUntil, 1=ActivePeriod
                // Only process current versions (ValidUntil = None)
                if value.0.is_some() {
                    return Err(anyhow::anyhow!("skip_non_current"));
                }

                // Resolve NameHash to String
                let name = resolve_name(storage, key.2)?;
                Ok(ReverseEdgeRecord {
                    dst_id: key.0,
                    src_id: key.1,
                    name,
                    valid_range: value.1, // ActivePeriod at index 1
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
        tracing::debug!(limit = self.limit, reverse = self.reverse, has_cursor = self.last.is_some(), "Executing AllNodeFragments scan");

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
        tracing::debug!(limit = self.limit, reverse = self.reverse, has_cursor = self.last.is_some(), "Executing AllEdgeFragments scan");

        // Convert String cursor to NameHash for key construction
        let seek_key = self
            .last
            .as_ref()
            .map(|(src, dst, name, ts)| {
                let name_hash = NameHash::from_name(name);
                schema::EdgeFragments::key_to_bytes(&schema::EdgeFragmentCfKey(
                    *src,
                    *dst,
                    name_hash,
                    *ts,
                ))
            })
            .unwrap_or_default();

        // Pre-compute cursor key hash for comparison
        let last_cursor_hash = self.last.as_ref().map(|(src, dst, name, ts)| {
            let name_hash = NameHash::from_name(name);
            (*src, *dst, name_hash, *ts)
        });

        iterate_and_visit::<schema::EdgeFragments, _, _, _, _>(
            storage,
            seek_key,
            self.limit,
            self.last.is_some(),
            self.reverse,
            self.reference_ts_millis,
            |key_bytes| {
                if let Some((src, dst, name_hash, ts)) = &last_cursor_hash {
                    let cursor_key = schema::EdgeFragments::key_to_bytes(
                        &schema::EdgeFragmentCfKey(*src, *dst, *name_hash, *ts),
                    );
                    key_bytes == cursor_key.as_slice()
                } else {
                    false
                }
            },
            |key_bytes, value_bytes| {
                let key = schema::EdgeFragments::key_from_bytes(key_bytes)?;
                let value = schema::EdgeFragments::value_from_bytes(value_bytes)?;
                // Resolve NameHash to String
                let edge_name = resolve_name(storage, key.2)?;
                Ok(EdgeFragmentRecord {
                    src_id: key.0,
                    dst_id: key.1,
                    edge_name,
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

// ============================================================================
// Hex helpers
// ============================================================================

fn hash_to_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

// ============================================================================
// Visitable Implementations — Cold CFs
// ============================================================================

impl Visitable for AllNames {
    type Record = NameRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        let seek_key = self
            .last
            .map(|h| Names::key_to_bytes(&NameCfKey(h)))
            .unwrap_or_default();

        let cursor_key = self.last.map(|h| Names::key_to_bytes(&NameCfKey(h)));

        iterate_and_visit::<Names, _, _, _, _>(
            storage,
            seek_key,
            self.limit,
            self.last.is_some(),
            self.reverse,
            None,
            |key_bytes| {
                cursor_key.as_ref().map_or(false, |ck| key_bytes == ck.as_slice())
            },
            |key_bytes, value_bytes| {
                let key = Names::key_from_bytes(key_bytes)?;
                let value = Names::value_from_bytes(value_bytes)?;
                Ok(NameRecord {
                    hash: hash_to_hex(key.0.as_bytes()),
                    name: value.0,
                })
            },
            |_record| &None,
            visitor,
        )
    }
}

impl Visitable for AllNodeSummaries {
    type Record = NodeSummaryRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        let seek_key = self
            .last
            .map(|h| NodeSummaries::key_to_bytes(&NodeSummaryCfKey(h)))
            .unwrap_or_default();

        let cursor_key = self.last.map(|h| NodeSummaries::key_to_bytes(&NodeSummaryCfKey(h)));

        iterate_and_visit::<NodeSummaries, _, _, _, _>(
            storage,
            seek_key,
            self.limit,
            self.last.is_some(),
            self.reverse,
            None,
            |key_bytes| {
                cursor_key.as_ref().map_or(false, |ck| key_bytes == ck.as_slice())
            },
            |key_bytes, value_bytes| {
                let key = NodeSummaries::key_from_bytes(key_bytes)?;
                let value = NodeSummaries::value_from_bytes(value_bytes)?;
                Ok(NodeSummaryRecord {
                    hash: hash_to_hex(key.0.as_bytes()),
                    content: value.0,
                })
            },
            |_record| &None,
            visitor,
        )
    }
}

impl Visitable for AllEdgeSummaries {
    type Record = EdgeSummaryRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        let seek_key = self
            .last
            .map(|h| EdgeSummaries::key_to_bytes(&EdgeSummaryCfKey(h)))
            .unwrap_or_default();

        let cursor_key = self.last.map(|h| EdgeSummaries::key_to_bytes(&EdgeSummaryCfKey(h)));

        iterate_and_visit::<EdgeSummaries, _, _, _, _>(
            storage,
            seek_key,
            self.limit,
            self.last.is_some(),
            self.reverse,
            None,
            |key_bytes| {
                cursor_key.as_ref().map_or(false, |ck| key_bytes == ck.as_slice())
            },
            |key_bytes, value_bytes| {
                let key = EdgeSummaries::key_from_bytes(key_bytes)?;
                let value = EdgeSummaries::value_from_bytes(value_bytes)?;
                Ok(EdgeSummaryRecord {
                    hash: hash_to_hex(key.0.as_bytes()),
                    content: value.0,
                })
            },
            |_record| &None,
            visitor,
        )
    }
}

impl Visitable for AllNodeSummaryIndex {
    type Record = NodeSummaryIndexRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        let seek_key = self
            .last
            .map(|(h, id, v)| {
                NodeSummaryIndex::key_to_bytes(&NodeSummaryIndexCfKey(h, id, v))
            })
            .unwrap_or_default();

        let cursor_key = self.last.map(|(h, id, v)| {
            NodeSummaryIndex::key_to_bytes(&NodeSummaryIndexCfKey(h, id, v))
        });

        iterate_and_visit::<NodeSummaryIndex, _, _, _, _>(
            storage,
            seek_key,
            self.limit,
            self.last.is_some(),
            self.reverse,
            None,
            |key_bytes| {
                cursor_key.as_ref().map_or(false, |ck| key_bytes == ck.as_slice())
            },
            |key_bytes, value_bytes| {
                let key = NodeSummaryIndex::key_from_bytes(key_bytes)?;
                let value = NodeSummaryIndex::value_from_bytes(value_bytes)?;
                let status = if value.0 == NodeSummaryIndexCfValue::CURRENT {
                    "current".to_string()
                } else {
                    "stale".to_string()
                };
                Ok(NodeSummaryIndexRecord {
                    hash: hash_to_hex(key.0.as_bytes()),
                    node_id: key.1,
                    version: key.2,
                    status,
                })
            },
            |_record| &None,
            visitor,
        )
    }
}

impl Visitable for AllEdgeSummaryIndex {
    type Record = EdgeSummaryIndexRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        let seek_key = self
            .last
            .map(|(h, src, dst, name_hash, v)| {
                EdgeSummaryIndex::key_to_bytes(&EdgeSummaryIndexCfKey(h, src, dst, name_hash, v))
            })
            .unwrap_or_default();

        let cursor_key = self.last.map(|(h, src, dst, name_hash, v)| {
            EdgeSummaryIndex::key_to_bytes(&EdgeSummaryIndexCfKey(h, src, dst, name_hash, v))
        });

        iterate_and_visit::<EdgeSummaryIndex, _, _, _, _>(
            storage,
            seek_key,
            self.limit,
            self.last.is_some(),
            self.reverse,
            None,
            |key_bytes| {
                cursor_key.as_ref().map_or(false, |ck| key_bytes == ck.as_slice())
            },
            |key_bytes, value_bytes| {
                let key = EdgeSummaryIndex::key_from_bytes(key_bytes)?;
                let value = EdgeSummaryIndex::value_from_bytes(value_bytes)?;
                let status = if value.0 == EdgeSummaryIndexCfValue::CURRENT {
                    "current".to_string()
                } else {
                    "stale".to_string()
                };
                let edge_name = resolve_name(storage, key.3)
                    .unwrap_or_else(|_| hash_to_hex(key.3.as_bytes()));
                Ok(EdgeSummaryIndexRecord {
                    hash: hash_to_hex(key.0.as_bytes()),
                    src_id: key.1,
                    dst_id: key.2,
                    edge_name,
                    version: key.4,
                    status,
                })
            },
            |_record| &None,
            visitor,
        )
    }
}

impl Visitable for AllNodeVersionHistory {
    type Record = NodeVersionHistoryRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        let seek_key = self
            .last
            .map(|(id, ts, v)| {
                NodeVersionHistory::key_to_bytes(&NodeVersionHistoryCfKey(id, ts, v))
            })
            .unwrap_or_default();

        let cursor_key = self.last.map(|(id, ts, v)| {
            NodeVersionHistory::key_to_bytes(&NodeVersionHistoryCfKey(id, ts, v))
        });

        iterate_and_visit::<NodeVersionHistory, _, _, _, _>(
            storage,
            seek_key,
            self.limit,
            self.last.is_some(),
            self.reverse,
            None,
            |key_bytes| {
                cursor_key.as_ref().map_or(false, |ck| key_bytes == ck.as_slice())
            },
            |key_bytes, value_bytes| {
                let key = NodeVersionHistory::key_from_bytes(key_bytes)?;
                let value = NodeVersionHistory::value_from_bytes(value_bytes)?;
                let name = resolve_name(storage, value.2)
                    .unwrap_or_else(|_| hash_to_hex(value.2.as_bytes()));
                Ok(NodeVersionHistoryRecord {
                    node_id: key.0,
                    valid_since: key.1,
                    version: key.2,
                    updated_at: value.0,
                    summary_hash: value.1.map(|h| hash_to_hex(h.as_bytes())),
                    name,
                    active_period: value.3,
                })
            },
            |_record| &None,
            visitor,
        )
    }
}

impl Visitable for AllEdgeVersionHistory {
    type Record = EdgeVersionHistoryRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        let seek_key = self
            .last
            .map(|(src, dst, name_hash, ts, v)| {
                EdgeVersionHistory::key_to_bytes(&EdgeVersionHistoryCfKey(src, dst, name_hash, ts, v))
            })
            .unwrap_or_default();

        let cursor_key = self.last.map(|(src, dst, name_hash, ts, v)| {
            EdgeVersionHistory::key_to_bytes(&EdgeVersionHistoryCfKey(src, dst, name_hash, ts, v))
        });

        iterate_and_visit::<EdgeVersionHistory, _, _, _, _>(
            storage,
            seek_key,
            self.limit,
            self.last.is_some(),
            self.reverse,
            None,
            |key_bytes| {
                cursor_key.as_ref().map_or(false, |ck| key_bytes == ck.as_slice())
            },
            |key_bytes, value_bytes| {
                let key = EdgeVersionHistory::key_from_bytes(key_bytes)?;
                let value = EdgeVersionHistory::value_from_bytes(value_bytes)?;
                let edge_name = resolve_name(storage, key.2)
                    .unwrap_or_else(|_| hash_to_hex(key.2.as_bytes()));
                Ok(EdgeVersionHistoryRecord {
                    src_id: key.0,
                    dst_id: key.1,
                    edge_name,
                    valid_since: key.3,
                    version: key.4,
                    updated_at: value.0,
                    summary_hash: value.1.map(|h| hash_to_hex(h.as_bytes())),
                    weight: value.2,
                    active_period: value.3,
                })
            },
            |_record| &None,
            visitor,
        )
    }
}

impl Visitable for AllOrphanSummaries {
    type Record = OrphanSummaryRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        let seek_key = self
            .last
            .map(|(ts, h)| {
                OrphanSummaries::key_to_bytes(&OrphanSummaryCfKey(ts, h))
            })
            .unwrap_or_default();

        let cursor_key = self.last.map(|(ts, h)| {
            OrphanSummaries::key_to_bytes(&OrphanSummaryCfKey(ts, h))
        });

        iterate_and_visit::<OrphanSummaries, _, _, _, _>(
            storage,
            seek_key,
            self.limit,
            self.last.is_some(),
            self.reverse,
            None,
            |key_bytes| {
                cursor_key.as_ref().map_or(false, |ck| key_bytes == ck.as_slice())
            },
            |key_bytes, value_bytes| {
                let key = OrphanSummaries::key_from_bytes(key_bytes)?;
                let value = OrphanSummaries::value_from_bytes(value_bytes)?;
                let kind = match value.0 {
                    SummaryKind::Node => "node".to_string(),
                    SummaryKind::Edge => "edge".to_string(),
                };
                Ok(OrphanSummaryRecord {
                    orphaned_at: key.0,
                    hash: hash_to_hex(key.1.as_bytes()),
                    kind,
                })
            },
            |_record| &None,
            visitor,
        )
    }
}

impl Visitable for AllGraphMeta {
    type Record = GraphMetaRecord;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize> {
        // GraphMeta keys are 1 byte each (discriminant), max 4 entries.
        // No pagination cursor needed; iterate from start.

        iterate_and_visit::<GraphMeta, _, _, _, _>(
            storage,
            vec![],
            self.limit,
            false,
            self.reverse,
            None,
            |_key_bytes| false,
            |key_bytes, value_bytes| {
                let key = GraphMeta::key_from_bytes(key_bytes)?;
                let value = GraphMeta::value_from_bytes(&key, value_bytes)?;
                let field = match key.0.discriminant() {
                    0x00 => "gc_cursor_node_summary_index".to_string(),
                    0x01 => "gc_cursor_edge_summary_index".to_string(),
                    0x02 => "gc_cursor_node_tombstones".to_string(),
                    0x03 => "gc_cursor_edge_tombstones".to_string(),
                    d => format!("unknown(0x{:02x})", d),
                };
                Ok(GraphMetaRecord {
                    field,
                    cursor_bytes_hex: hash_to_hex(value.0.cursor_bytes()),
                })
            },
            |_record| &None,
            visitor,
        )
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::Runnable;
    use super::super::writer::{create_mutation_writer, spawn_mutation_consumer, WriterConfig};
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
        let handle = spawn_mutation_consumer(receiver, config, db_path);

        // Add some test nodes
        for i in 0..5 {
            let node = AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                name: format!("test_node_{}", i),
                valid_range: None,
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
        let handle = spawn_mutation_consumer(receiver, config, db_path);

        // Add some test nodes
        let mut node_ids = Vec::new();
        for i in 0..10 {
            let id = Id::new();
            node_ids.push(id);
            let node = AddNode {
                id,
                ts_millis: TimestampMilli::now(),
                name: format!("test_node_{}", i),
                valid_range: None,
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
        let handle = spawn_mutation_consumer(receiver, config, db_path);

        // Add some test nodes
        for i in 0..10 {
            let node = AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                name: format!("test_node_{}", i),
                valid_range: None,
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
        let handle = spawn_mutation_consumer(receiver, config, db_path);

        // Add some test nodes first
        let node1 = Id::new();
        let node2 = Id::new();
        let node3 = Id::new();

        for (id, name) in [(node1, "node1"), (node2, "node2"), (node3, "node3")] {
            let node = AddNode {
                id,
                ts_millis: TimestampMilli::now(),
                name: name.to_string(),
                valid_range: None,
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
                valid_range: None,
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
        let handle = spawn_mutation_consumer(receiver, config, db_path);

        // Add some test nodes first
        let node1 = Id::new();
        let node2 = Id::new();
        let node3 = Id::new();

        for (id, name) in [(node1, "node1"), (node2, "node2"), (node3, "node3")] {
            let node = AddNode {
                id,
                ts_millis: TimestampMilli::now(),
                name: name.to_string(),
                valid_range: None,
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
                valid_range: None,
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
        let handle = spawn_mutation_consumer(receiver, config, db_path);

        // Add test nodes
        let mut nodes = Vec::new();
        for i in 0..5 {
            let id = Id::new();
            nodes.push(id);
            let node = AddNode {
                id,
                ts_millis: TimestampMilli::now(),
                name: format!("node_{}", i),
                valid_range: None,
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
                valid_range: None,
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
        let handle = spawn_mutation_consumer(receiver, config, db_path);

        // Add some test nodes with delays to ensure different IDs
        let mut node_ids = Vec::new();
        for i in 0..5 {
            let id = Id::new();
            node_ids.push(id);
            let node = AddNode {
                id,
                ts_millis: TimestampMilli::now(),
                name: format!("test_node_{}", i),
                valid_range: None,
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
        let handle = spawn_mutation_consumer(receiver, config, db_path);

        // Add test nodes
        for i in 0..10 {
            let id = Id::new();
            let node = AddNode {
                id,
                ts_millis: TimestampMilli::now(),
                name: format!("test_node_{}", i),
                valid_range: None,
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
