//! Query module providing query types and their business logic implementations.
//!
//! This module contains only business logic - query type definitions and their
//! QueryExecutor implementations. Infrastructure (traits, Reader, Consumer, spawn
//! functions) is in the `reader` module.
//!
//! # Transaction Support
//!
//! Query types implement [`TransactionQueryExecutor`] to support read-your-writes
//! semantics within a transaction scope. This allows queries to see uncommitted
//! mutations in the same transaction.

use anyhow::Result;
use std::ops::Bound;
use std::time::Duration;

use super::name_hash::NameHash;
use super::reader::{QueryExecutor, QueryRequest};
use super::scan::{self, Visitable};
use super::HotColumnFamilyRecord;
use super::summary_hash::SummaryHash;
use crate::reader::Runnable;
use super::schema::{
    self, DstId, EdgeName, EdgeSummary, EdgeSummaries, EdgeSummaryCfKey, FragmentContent,
    Names, NameCfKey, NodeName, NodeSummary, NodeSummaries, NodeSummaryCfKey, SrcId,
};
use super::{ColumnFamily, ColumnFamilySerde};
use super::Storage;
use crate::{Id, TimestampMilli};
use crate::request::{new_request_id, RequestMeta};
use tokio::sync::oneshot;

// ============================================================================
// Name Resolution Helpers
// ============================================================================

/// Resolve a NameHash to its full String name.
///
/// Uses the in-memory NameCache first for O(1) lookup, falling back to
/// Names CF lookup only for cache misses. On cache miss, the name is
/// added to the cache for future lookups.
///
/// This is the primary function for QueryExecutor implementations that
/// have access to Storage.
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

/// Resolve a NameHash from a Transaction (sees uncommitted writes).
///
/// For TransactionQueryExecutor implementations that need read-your-writes
/// semantics. Uses cache first, then falls back to transaction lookup.
fn resolve_name_from_txn(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    name_hash: NameHash,
    cache: &super::name_hash::NameCache,
) -> Result<String> {
    // Check cache first (O(1) DashMap lookup)
    if let Some(name) = cache.get(&name_hash) {
        return Ok((*name).clone());
    }

    // Cache miss: fetch from transaction (sees uncommitted writes)
    let names_cf = txn_db
        .cf_handle(Names::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Names CF not found"))?;

    let key_bytes = Names::key_to_bytes(&NameCfKey(name_hash));

    let value_bytes = txn
        .get_cf(names_cf, &key_bytes)?
        .ok_or_else(|| anyhow::anyhow!("Name not found for hash: {}", name_hash))?;

    let value = Names::value_from_bytes(&value_bytes)?;
    let name = value.0;

    // Populate cache for future lookups
    cache.insert(name_hash, name.clone());

    Ok(name)
}

/// Resolve a node summary from the NodeSummaries cold CF.
///
/// If the summary_hash is None or the summary is not found, returns an empty DataUrl.
fn resolve_node_summary(
    storage: &Storage,
    summary_hash: Option<SummaryHash>,
) -> Result<NodeSummary> {
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

// ============================================================================
// VERSIONING Helpers for Current Version Lookup
// ============================================================================
// (claude, 2026-02-07, FIXED: Items 15-16 - Unified storage access with reverse seek)

/// Storage access abstraction for unified query logic.
/// (claude, 2026-02-07, FIXED: Item 16 - Eliminates readonly/readwrite/txn duplication)
#[derive(Clone)]
enum StorageAccess<'a> {
    Readonly(&'a rocksdb::DB),
    Readwrite(&'a rocksdb::TransactionDB),
    Transaction(&'a rocksdb::Transaction<'a, rocksdb::TransactionDB>, &'a rocksdb::TransactionDB),
}

impl<'a> StorageAccess<'a> {
    /// Get a column family handle.
    fn cf_handle(&self, cf_name: &str) -> Option<&rocksdb::ColumnFamily> {
        match self {
            StorageAccess::Readonly(db) => db.cf_handle(cf_name),
            StorageAccess::Readwrite(db) => db.cf_handle(cf_name),
            StorageAccess::Transaction(_, db) => db.cf_handle(cf_name),
        }
    }
}

/// Unified node version finder using forward scan with early termination.
/// Works with any storage access type.
///
/// (claude, 2026-02-07, FIXED: Item 16 - Single implementation for all storage types)
fn find_node_version_unified(
    storage: StorageAccess<'_>,
    node_id: Id,
    as_of: Option<TimestampMilli>,
) -> Result<Option<(Vec<u8>, schema::NodeCfValue)>> {
    let prefix = node_id.into_bytes().to_vec();

    // For current version queries, we scan forward looking for ValidUntil = None.
    // For as_of queries, we use reverse iteration from the seek point for O(1) lookup.
    // (claude, 2026-02-07, FIXED: Item 15 - Use reverse seek for as_of queries)

    match as_of {
        None => {
            // Current version query: scan forward, find ValidUntil = None
            find_node_current_version_scan(&storage, &prefix)
        }
        Some(as_of_ts) => {
            // Point-in-time query: use reverse seek from (prefix, as_of_ts)
            find_node_version_at_time_reverse(&storage, &prefix, as_of_ts)
        }
    }
}

/// Find current node version using forward scan (ValidUntil = None).
fn find_node_current_version_scan(
    storage: &StorageAccess<'_>,
    prefix: &[u8],
) -> Result<Option<(Vec<u8>, schema::NodeCfValue)>> {
    let nodes_cf = storage.cf_handle(schema::Nodes::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Nodes CF not found"))?;

    // Create iterator based on storage type
    match storage {
        StorageAccess::Readonly(db) => {
            for item in db.prefix_iterator_cf(nodes_cf, prefix) {
                let (key_bytes, value_bytes) = item?;
                if !key_bytes.starts_with(prefix) {
                    break;
                }
                let value: schema::NodeCfValue = schema::Nodes::value_from_bytes(&value_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize node value: {}", e))?;
                if value.0.is_none() {
                    return Ok(Some((key_bytes.to_vec(), value)));
                }
            }
        }
        StorageAccess::Readwrite(db) => {
            for item in db.prefix_iterator_cf(nodes_cf, prefix) {
                let (key_bytes, value_bytes) = item?;
                if !key_bytes.starts_with(prefix) {
                    break;
                }
                let value: schema::NodeCfValue = schema::Nodes::value_from_bytes(&value_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize node value: {}", e))?;
                if value.0.is_none() {
                    return Ok(Some((key_bytes.to_vec(), value)));
                }
            }
        }
        StorageAccess::Transaction(txn, _) => {
            for item in txn.prefix_iterator_cf(nodes_cf, prefix) {
                let (key_bytes, value_bytes) = item?;
                if !key_bytes.starts_with(prefix) {
                    break;
                }
                let value: schema::NodeCfValue = schema::Nodes::value_from_bytes(&value_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize node value: {}", e))?;
                if value.0.is_none() {
                    return Ok(Some((key_bytes.to_vec(), value)));
                }
            }
        }
    }
    Ok(None)
}

/// Find node version at specific time using reverse seek.
/// (claude, 2026-02-07, FIXED: Item 15 - O(1) lookup instead of O(k) scan)
///
/// Strategy: Seek to (prefix, as_of_ts) and scan backwards to find the version
/// where ValidSince <= as_of_ts AND (ValidUntil is None OR ValidUntil > as_of_ts).
fn find_node_version_at_time_reverse(
    storage: &StorageAccess<'_>,
    prefix: &[u8],
    as_of_ts: TimestampMilli,
) -> Result<Option<(Vec<u8>, schema::NodeCfValue)>> {
    let nodes_cf = storage.cf_handle(schema::Nodes::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Nodes CF not found"))?;
    // Build seek key: prefix + as_of_ts (24 bytes for nodes)
    let mut seek_key = Vec::with_capacity(24);
    seek_key.extend_from_slice(prefix);
    seek_key.extend_from_slice(&as_of_ts.0.to_be_bytes());

    // Use iterator with SeekForPrev to find the largest key <= seek_key
    // Then scan forward from there to find valid versions
    match storage {
        StorageAccess::Readonly(db) => {
            let mut iter = db.raw_iterator_cf(nodes_cf);
            iter.seek_for_prev(&seek_key);

            // Check entries starting from seek position
            while iter.valid() {
                let Some(key_bytes) = iter.key() else { break };
                if !key_bytes.starts_with(prefix) {
                    // Went past prefix boundary, try previous
                    iter.prev();
                    continue;
                }

                let Some(value_bytes) = iter.value() else { break };
                // Copy to aligned buffer for rkyv deserialization
                let value_bytes_aligned = value_bytes.to_vec();

                // Extract ValidSince from key
                let valid_since = if key_bytes.len() >= 24 {
                    let ts_bytes: [u8; 8] = key_bytes[16..24].try_into().unwrap();
                    TimestampMilli(u64::from_be_bytes(ts_bytes))
                } else {
                    iter.prev();
                    continue;
                };

                // Skip if ValidSince > as_of (shouldn't happen after seek_for_prev, but safety check)
                if valid_since.0 > as_of_ts.0 {
                    iter.prev();
                    continue;
                }

                let value: schema::NodeCfValue = schema::Nodes::value_from_bytes(&value_bytes_aligned)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize node value: {}", e))?;

                // Check ValidUntil
                let is_valid = match value.0 {
                    None => true,
                    Some(valid_until) => valid_until.0 > as_of_ts.0,
                };

                if is_valid {
                    return Ok(Some((key_bytes.to_vec(), value)));
                }

                // This version ended before as_of, try earlier version
                iter.prev();
            }
        }
        StorageAccess::Readwrite(db) => {
            let mut iter = db.raw_iterator_cf(nodes_cf);
            iter.seek_for_prev(&seek_key);

            while iter.valid() {
                let Some(key_bytes) = iter.key() else { break };
                if !key_bytes.starts_with(prefix) {
                    iter.prev();
                    continue;
                }

                let Some(value_bytes) = iter.value() else { break };
                // Copy to aligned buffer for rkyv deserialization
                let value_bytes_aligned = value_bytes.to_vec();

                let valid_since = if key_bytes.len() >= 24 {
                    let ts_bytes: [u8; 8] = key_bytes[16..24].try_into().unwrap();
                    TimestampMilli(u64::from_be_bytes(ts_bytes))
                } else {
                    iter.prev();
                    continue;
                };

                if valid_since.0 > as_of_ts.0 {
                    iter.prev();
                    continue;
                }

                let value: schema::NodeCfValue = schema::Nodes::value_from_bytes(&value_bytes_aligned)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize node value: {}", e))?;

                let is_valid = match value.0 {
                    None => true,
                    Some(valid_until) => valid_until.0 > as_of_ts.0,
                };

                if is_valid {
                    return Ok(Some((key_bytes.to_vec(), value)));
                }

                iter.prev();
            }
        }
        StorageAccess::Transaction(txn, _) => {
            // Transaction doesn't have raw_iterator_cf with seek_for_prev,
            // fall back to forward scan for transaction context
            // (This is acceptable since transaction queries are less common)
            let iter = txn.prefix_iterator_cf(nodes_cf, prefix);
            let mut best_match: Option<(Vec<u8>, schema::NodeCfValue)> = None;

            for item in iter {
                let (key_bytes, value_bytes) = item?;
                if !key_bytes.starts_with(prefix) {
                    break;
                }

                let valid_since = if key_bytes.len() >= 24 {
                    let ts_bytes: [u8; 8] = key_bytes[16..24].try_into().unwrap();
                    TimestampMilli(u64::from_be_bytes(ts_bytes))
                } else {
                    continue;
                };

                if valid_since.0 > as_of_ts.0 {
                    continue;
                }

                let value: schema::NodeCfValue = schema::Nodes::value_from_bytes(&value_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize node value: {}", e))?;

                let is_valid = match value.0 {
                    None => true,
                    Some(valid_until) => valid_until.0 > as_of_ts.0,
                };

                if is_valid {
                    best_match = Some((key_bytes.to_vec(), value));
                }
            }
            return Ok(best_match);
        }
    }

    Ok(None)
}

// Legacy wrapper functions for backwards compatibility
// These delegate to the unified implementation

/// Find the current version of a node via prefix scan (readonly mode).
fn find_current_node_version_readonly(
    db: &rocksdb::DB,
    node_id: Id,
) -> Result<Option<(Vec<u8>, schema::NodeCfValue)>> {
    find_node_version_unified(StorageAccess::Readonly(db), node_id, None)
}

/// Find the current version of a node via prefix scan (readwrite/transaction mode).
fn find_current_node_version_readwrite(
    txn_db: &rocksdb::TransactionDB,
    node_id: Id,
) -> Result<Option<(Vec<u8>, schema::NodeCfValue)>> {
    find_node_version_unified(StorageAccess::Readwrite(txn_db), node_id, None)
}

/// Find the current version of a node via prefix scan (within transaction).
fn find_current_node_version_txn(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    node_id: Id,
) -> Result<Option<(Vec<u8>, schema::NodeCfValue)>> {
    find_node_version_unified(StorageAccess::Transaction(txn, txn_db), node_id, None)
}

// ============================================================================
// VERSIONING Helpers for Point-in-Time Queries
// ============================================================================
// (claude, 2026-02-07, FIXED: Items 15-16 - Now delegates to unified implementation)

/// Find the version of a node valid at a specific system time (readonly mode).
/// (claude, 2026-02-07, FIXED: Item 15 - Uses reverse seek for O(1) lookup)
/// (claude, 2026-02-07, FIXED: Item 16 - Delegates to unified implementation)
fn find_node_version_at_time_readonly(
    db: &rocksdb::DB,
    node_id: Id,
    as_of: Option<TimestampMilli>,
) -> Result<Option<(Vec<u8>, schema::NodeCfValue)>> {
    find_node_version_unified(StorageAccess::Readonly(db), node_id, as_of)
}

/// Find the version of a node valid at a specific system time (readwrite mode).
/// (claude, 2026-02-07, FIXED: Items 15-16 - Delegates to unified implementation)
fn find_node_version_at_time_readwrite(
    txn_db: &rocksdb::TransactionDB,
    node_id: Id,
    as_of: Option<TimestampMilli>,
) -> Result<Option<(Vec<u8>, schema::NodeCfValue)>> {
    find_node_version_unified(StorageAccess::Readwrite(txn_db), node_id, as_of)
}

/// Find the version of a node valid at a specific system time (within transaction).
/// (claude, 2026-02-07, FIXED: Items 15-16 - Delegates to unified implementation)
fn find_node_version_at_time_txn(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    node_id: Id,
    as_of: Option<TimestampMilli>,
) -> Result<Option<(Vec<u8>, schema::NodeCfValue)>> {
    find_node_version_unified(StorageAccess::Transaction(txn, txn_db), node_id, as_of)
}

// ============================================================================
// Unified Edge Version Finder
// ============================================================================
// (claude, 2026-02-07, FIXED: Items 15-16 - Unified edge version lookup)

/// Unified edge version finder using reverse seek for efficiency.
/// Works with any storage access type.
fn find_edge_version_unified(
    storage: StorageAccess<'_>,
    src_id: Id,
    dst_id: Id,
    name_hash: NameHash,
    as_of: Option<TimestampMilli>,
) -> Result<Option<(Vec<u8>, schema::ForwardEdgeCfValue)>> {
    // Build 40-byte prefix: src_id (16) + dst_id (16) + name_hash (8)
    let mut prefix = Vec::with_capacity(40);
    prefix.extend_from_slice(src_id.as_bytes());
    prefix.extend_from_slice(dst_id.as_bytes());
    prefix.extend_from_slice(name_hash.as_bytes());

    match as_of {
        None => find_edge_current_version_scan(&storage, &prefix),
        Some(as_of_ts) => find_edge_version_at_time_reverse(&storage, &prefix, as_of_ts),
    }
}

/// Find current edge version using forward scan (ValidUntil = None).
fn find_edge_current_version_scan(
    storage: &StorageAccess<'_>,
    prefix: &[u8],
) -> Result<Option<(Vec<u8>, schema::ForwardEdgeCfValue)>> {
    let edges_cf = storage.cf_handle(schema::ForwardEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;

    match storage {
        StorageAccess::Readonly(db) => {
            for item in db.prefix_iterator_cf(edges_cf, prefix) {
                let (key_bytes, value_bytes) = item?;
                if !key_bytes.starts_with(prefix) {
                    break;
                }
                let value: schema::ForwardEdgeCfValue = schema::ForwardEdges::value_from_bytes(&value_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize edge value: {}", e))?;
                if value.0.is_none() {
                    return Ok(Some((key_bytes.to_vec(), value)));
                }
            }
        }
        StorageAccess::Readwrite(db) => {
            for item in db.prefix_iterator_cf(edges_cf, prefix) {
                let (key_bytes, value_bytes) = item?;
                if !key_bytes.starts_with(prefix) {
                    break;
                }
                let value: schema::ForwardEdgeCfValue = schema::ForwardEdges::value_from_bytes(&value_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize edge value: {}", e))?;
                if value.0.is_none() {
                    return Ok(Some((key_bytes.to_vec(), value)));
                }
            }
        }
        StorageAccess::Transaction(txn, _) => {
            for item in txn.prefix_iterator_cf(edges_cf, prefix) {
                let (key_bytes, value_bytes) = item?;
                if !key_bytes.starts_with(prefix) {
                    break;
                }
                let value: schema::ForwardEdgeCfValue = schema::ForwardEdges::value_from_bytes(&value_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize edge value: {}", e))?;
                if value.0.is_none() {
                    return Ok(Some((key_bytes.to_vec(), value)));
                }
            }
        }
    }
    Ok(None)
}

/// Find edge version at specific time using reverse seek.
/// (claude, 2026-02-07, FIXED: Item 15 - O(1) lookup instead of O(k) scan)
fn find_edge_version_at_time_reverse(
    storage: &StorageAccess<'_>,
    prefix: &[u8],
    as_of_ts: TimestampMilli,
) -> Result<Option<(Vec<u8>, schema::ForwardEdgeCfValue)>> {
    let edges_cf = storage.cf_handle(schema::ForwardEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;

    // Build seek key: prefix + as_of_ts (48 bytes for edges)
    let mut seek_key = Vec::with_capacity(48);
    seek_key.extend_from_slice(prefix);
    seek_key.extend_from_slice(&as_of_ts.0.to_be_bytes());

    match storage {
        StorageAccess::Readonly(db) => {
            let mut iter = db.raw_iterator_cf(edges_cf);
            iter.seek_for_prev(&seek_key);

            while iter.valid() {
                let Some(key_bytes) = iter.key() else { break };
                if !key_bytes.starts_with(prefix) {
                    iter.prev();
                    continue;
                }

                let Some(value_bytes) = iter.value() else { break };
                // Copy to aligned buffer for rkyv deserialization
                let value_bytes_aligned = value_bytes.to_vec();

                let valid_since = if key_bytes.len() >= 48 {
                    let ts_bytes: [u8; 8] = key_bytes[40..48].try_into().unwrap();
                    TimestampMilli(u64::from_be_bytes(ts_bytes))
                } else {
                    iter.prev();
                    continue;
                };

                if valid_since.0 > as_of_ts.0 {
                    iter.prev();
                    continue;
                }

                let value: schema::ForwardEdgeCfValue = schema::ForwardEdges::value_from_bytes(&value_bytes_aligned)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize edge value: {}", e))?;

                let is_valid = match value.0 {
                    None => true,
                    Some(valid_until) => valid_until.0 > as_of_ts.0,
                };

                if is_valid {
                    return Ok(Some((key_bytes.to_vec(), value)));
                }

                iter.prev();
            }
        }
        StorageAccess::Readwrite(db) => {
            let mut iter = db.raw_iterator_cf(edges_cf);
            iter.seek_for_prev(&seek_key);

            while iter.valid() {
                let Some(key_bytes) = iter.key() else { break };
                if !key_bytes.starts_with(prefix) {
                    iter.prev();
                    continue;
                }

                let Some(value_bytes) = iter.value() else { break };
                // Copy to aligned buffer for rkyv deserialization
                let value_bytes_aligned = value_bytes.to_vec();

                let valid_since = if key_bytes.len() >= 48 {
                    let ts_bytes: [u8; 8] = key_bytes[40..48].try_into().unwrap();
                    TimestampMilli(u64::from_be_bytes(ts_bytes))
                } else {
                    iter.prev();
                    continue;
                };

                if valid_since.0 > as_of_ts.0 {
                    iter.prev();
                    continue;
                }

                let value: schema::ForwardEdgeCfValue = schema::ForwardEdges::value_from_bytes(&value_bytes_aligned)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize edge value: {}", e))?;

                let is_valid = match value.0 {
                    None => true,
                    Some(valid_until) => valid_until.0 > as_of_ts.0,
                };

                if is_valid {
                    return Ok(Some((key_bytes.to_vec(), value)));
                }

                iter.prev();
            }
        }
        StorageAccess::Transaction(txn, _) => {
            // Fall back to forward scan for transaction context
            let iter = txn.prefix_iterator_cf(edges_cf, prefix);
            let mut best_match: Option<(Vec<u8>, schema::ForwardEdgeCfValue)> = None;

            for item in iter {
                let (key_bytes, value_bytes) = item?;
                if !key_bytes.starts_with(prefix) {
                    break;
                }

                let valid_since = if key_bytes.len() >= 48 {
                    let ts_bytes: [u8; 8] = key_bytes[40..48].try_into().unwrap();
                    TimestampMilli(u64::from_be_bytes(ts_bytes))
                } else {
                    continue;
                };

                if valid_since.0 > as_of_ts.0 {
                    continue;
                }

                let value: schema::ForwardEdgeCfValue = schema::ForwardEdges::value_from_bytes(&value_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize edge value: {}", e))?;

                let is_valid = match value.0 {
                    None => true,
                    Some(valid_until) => valid_until.0 > as_of_ts.0,
                };

                if is_valid {
                    best_match = Some((key_bytes.to_vec(), value));
                }
            }
            return Ok(best_match);
        }
    }

    Ok(None)
}

// Legacy wrapper functions for edge queries

/// Find the version of a forward edge valid at a specific system time (readonly mode).
/// (claude, 2026-02-07, FIXED: Items 15-16 - Delegates to unified implementation)
fn find_forward_edge_version_at_time_readonly(
    db: &rocksdb::DB,
    src_id: Id,
    dst_id: Id,
    name_hash: NameHash,
    as_of: Option<TimestampMilli>,
) -> Result<Option<(Vec<u8>, schema::ForwardEdgeCfValue)>> {
    find_edge_version_unified(StorageAccess::Readonly(db), src_id, dst_id, name_hash, as_of)
}

/// Find the version of a forward edge valid at a specific system time (readwrite mode).
/// (claude, 2026-02-07, FIXED: Items 15-16 - Delegates to unified implementation)
fn find_forward_edge_version_at_time_readwrite(
    txn_db: &rocksdb::TransactionDB,
    src_id: Id,
    dst_id: Id,
    name_hash: NameHash,
    as_of: Option<TimestampMilli>,
) -> Result<Option<(Vec<u8>, schema::ForwardEdgeCfValue)>> {
    find_edge_version_unified(StorageAccess::Readwrite(txn_db), src_id, dst_id, name_hash, as_of)
}

/// Find the current version of a forward edge via prefix scan (readonly mode).
/// (claude, 2026-02-07, FIXED: Item 16 - Delegates to unified implementation)
fn find_current_forward_edge_readonly(
    db: &rocksdb::DB,
    src_id: Id,
    dst_id: Id,
    name_hash: NameHash,
) -> Result<Option<(Vec<u8>, schema::ForwardEdgeCfValue)>> {
    find_edge_version_unified(StorageAccess::Readonly(db), src_id, dst_id, name_hash, None)
}

/// Find the current version of a forward edge via prefix scan (readwrite mode).
/// (claude, 2026-02-07, FIXED: Item 16 - Delegates to unified implementation)
fn find_current_forward_edge_readwrite(
    txn_db: &rocksdb::TransactionDB,
    src_id: Id,
    dst_id: Id,
    name_hash: NameHash,
) -> Result<Option<(Vec<u8>, schema::ForwardEdgeCfValue)>> {
    find_edge_version_unified(StorageAccess::Readwrite(txn_db), src_id, dst_id, name_hash, None)
}

/// Find the current version of a forward edge via prefix scan (within transaction).
/// (claude, 2026-02-07, FIXED: Item 16 - Delegates to unified implementation)
fn find_current_forward_edge_txn(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    src_id: Id,
    dst_id: Id,
    name_hash: NameHash,
) -> Result<Option<(Vec<u8>, schema::ForwardEdgeCfValue)>> {
    find_edge_version_unified(StorageAccess::Transaction(txn, txn_db), src_id, dst_id, name_hash, None)
}

/// Resolve an edge summary from the EdgeSummaries cold CF.
///
/// If the summary_hash is None or the summary is not found, returns an empty DataUrl.
fn resolve_edge_summary(
    storage: &Storage,
    summary_hash: Option<SummaryHash>,
) -> Result<EdgeSummary> {
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

/// Resolve an edge summary from a transaction context (for read-your-writes).
fn resolve_edge_summary_from_txn(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    summary_hash: Option<SummaryHash>,
) -> Result<EdgeSummary> {
    let Some(hash) = summary_hash else {
        return Ok(EdgeSummary::from_text(""));
    };

    let summaries_cf = txn_db
        .cf_handle(EdgeSummaries::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("EdgeSummaries CF not found"))?;

    let key_bytes = EdgeSummaries::key_to_bytes(&EdgeSummaryCfKey(hash));

    match txn.get_cf(summaries_cf, &key_bytes)? {
        Some(bytes) => {
            let value = EdgeSummaries::value_from_bytes(&bytes)?;
            Ok(value.0) // .0 is the summary (no refcount in VERSIONING)
        }
        None => Ok(EdgeSummary::from_text("")),
    }
}

/// Resolve a node summary from a transaction context (for read-your-writes).
fn resolve_node_summary_from_txn(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    summary_hash: Option<SummaryHash>,
) -> Result<NodeSummary> {
    let Some(hash) = summary_hash else {
        return Ok(NodeSummary::from_text(""));
    };

    let summaries_cf = txn_db
        .cf_handle(NodeSummaries::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("NodeSummaries CF not found"))?;

    let key_bytes = NodeSummaries::key_to_bytes(&NodeSummaryCfKey(hash));

    match txn.get_cf(summaries_cf, &key_bytes)? {
        Some(bytes) => {
            let value = NodeSummaries::value_from_bytes(&bytes)?;
            Ok(value.0) // .0 is the summary (no refcount in VERSIONING)
        }
        None => Ok(NodeSummary::from_text("")),
    }
}

// ============================================================================
// TransactionQueryExecutor Trait
// ============================================================================

/// Trait for queries that can execute within a transaction scope.
///
/// Unlike [`QueryExecutor`] (which takes `&Storage` and routes through
/// MPSC channels), this trait executes directly against an active
/// RocksDB transaction, enabling read-your-writes semantics.
///
/// # When to Use
///
/// Use this trait when you need to:
/// - Read data that was just written in the same transaction
/// - Execute queries within an atomic transaction scope
/// - Implement algorithms that interleave reads and writes (e.g., HNSW insert)
///
/// # Implementation Pattern
///
/// Each query type implements this trait alongside `QueryExecutor`:
///
/// ```rust,ignore
/// impl TransactionQueryExecutor for NodeById {
///     type Output = (NodeName, NodeSummary);
///
///     fn execute_in_transaction(
///         &self,
///         txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
///         txn_db: &rocksdb::TransactionDB,
///     ) -> Result<Self::Output> {
///         let cf = txn_db.cf_handle(Nodes::CF_NAME)?;
///         // Use txn.get_cf() to see uncommitted writes
///         let value = txn.get_cf(cf, &key)?;
///         // ... deserialize and return
///     }
/// }
/// ```
///
/// # Example Usage
///
/// ```rust,ignore
/// let mut txn = writer.transaction()?;
///
/// // Write a node
/// txn.write(AddNode { id, ... })?;
///
/// // Read sees the uncommitted node!
/// let (name, summary) = txn.read(NodeById::new(id, None))?;
///
/// txn.commit()?;
/// ```
pub trait TransactionQueryExecutor: Send + Sync {
    /// The output type of this query.
    type Output: Send;

    /// Execute query within a transaction (sees uncommitted writes).
    ///
    /// # Arguments
    ///
    /// * `txn` - Active RocksDB transaction
    /// * `txn_db` - TransactionDB for column family handles
    /// * `cache` - Name cache for efficient hash-to-name resolution
    ///
    /// # Returns
    ///
    /// Query result, which may include data from uncommitted writes
    /// within the same transaction.
    fn execute_in_transaction(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        cache: &super::name_hash::NameCache,
    ) -> Result<Self::Output>;
}

/// Helper function to check if a timestamp falls within a time range
fn timestamp_in_range(
    ts: TimestampMilli,
    time_range: &(Bound<TimestampMilli>, Bound<TimestampMilli>),
) -> bool {
    let (start_bound, end_bound) = time_range;

    let start_ok = match start_bound {
        Bound::Unbounded => true,
        Bound::Included(start) => ts.0 >= start.0,
        Bound::Excluded(start) => ts.0 > start.0,
    };

    let end_ok = match end_bound {
        Bound::Unbounded => true,
        Bound::Included(end) => ts.0 <= end.0,
        Bound::Excluded(end) => ts.0 < end.0,
    };

    start_ok && end_ok
}

/// Macro to iterate over a column family in both readonly and readwrite storage modes
/// This reduces boilerplate for the common pattern of handling both DB and TransactionDB
/// The process_body should be a block of code that can access variables from the enclosing scope
macro_rules! iterate_cf {
    ($storage:expr, $cf_type:ty, $start_key:expr, |$item:ident| $process_body:block) => {{
        if let Ok(db) = $storage.db() {
            let cf = db
                .cf_handle(<$cf_type>::CF_NAME)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Column family '{}' not found",
                        <$cf_type>::CF_NAME
                    )
                })?;

            let iter = db.iterator_cf(
                cf,
                rocksdb::IteratorMode::From(&$start_key, rocksdb::Direction::Forward),
            );

            for $item in iter $process_body
        } else {
            let txn_db = $storage.transaction_db()?;
            let cf = txn_db
                .cf_handle(<$cf_type>::CF_NAME)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Column family '{}' not found",
                        <$cf_type>::CF_NAME
                    )
                })?;

            let iter = txn_db.iterator_cf(
                cf,
                rocksdb::IteratorMode::From(&$start_key, rocksdb::Direction::Forward),
            );

            for $item in iter $process_body
        }
    }};
}

// ============================================================================
// Query Enum
// ============================================================================

/// Query enum representing all possible query types.
///
/// **Note**: This is internal infrastructure for the query dispatch pipeline.
/// Users interact with query parameter structs directly via the `Runnable` trait.
/// This enum is public because it appears in reader function signatures, but users
/// should not construct variants directly.
#[derive(Debug)]
#[doc(hidden)]
#[allow(private_interfaces)]
pub enum Query {
    NodeById(NodeById),
    NodesByIdsMulti(NodesByIdsMulti),
    EdgeSummaryBySrcDstName(EdgeSummaryBySrcDstName),
    NodeFragmentsByIdTimeRange(NodeFragmentsByIdTimeRange),
    EdgeFragmentsByIdTimeRange(EdgeFragmentsByIdTimeRange),
    OutgoingEdges(OutgoingEdges),
    IncomingEdges(IncomingEdges),
    AllNodes(AllNodes),
    AllEdges(AllEdges),
    // CONTENT-ADDRESS reverse lookup queries
    NodesBySummaryHash(NodesBySummaryHash),
    EdgesBySummaryHash(EdgesBySummaryHash),
}

impl std::fmt::Display for Query {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Query::NodeById(q) => write!(f, "NodeById: id={}", q.id),
            Query::NodesByIdsMulti(q) => {
                write!(f, "NodesByIdsMulti: count={}", q.ids.len())
            }
            Query::EdgeSummaryBySrcDstName(q) => write!(
                f,
                "EdgeBySrcDstName: source={}, dest={}, name={}",
                q.source_id, q.dest_id, q.name
            ),
            Query::NodeFragmentsByIdTimeRange(q) => {
                write!(
                    f,
                    "NodeFragmentsByIdTimeRange: id={}, range={:?}",
                    q.id, q.time_range
                )
            }
            Query::EdgeFragmentsByIdTimeRange(q) => {
                write!(
                    f,
                    "EdgeFragmentsByIdTimeRange: source={}, dest={}, name={}, range={:?}",
                    q.source_id, q.dest_id, q.edge_name, q.time_range
                )
            }
            Query::OutgoingEdges(q) => write!(f, "OutgoingEdges: id={}", q.id),
            Query::IncomingEdges(q) => write!(f, "IncomingEdges: id={}", q.id),
            Query::AllNodes(q) => write!(f, "AllNodes: limit={}", q.limit),
            Query::AllEdges(q) => write!(f, "AllEdges: limit={}", q.limit),
            Query::NodesBySummaryHash(q) => write!(
                f,
                "NodesBySummaryHash: hash={:?}, current_only={}",
                q.hash, q.current_only
            ),
            Query::EdgesBySummaryHash(q) => write!(
                f,
                "EdgesBySummaryHash: hash={:?}, current_only={}",
                q.hash, q.current_only
            ),
        }
    }
}

#[derive(Debug)]
pub enum QueryResult {
    NodeById((NodeName, NodeSummary)),
    NodesByIdsMulti(Vec<(Id, NodeName, NodeSummary)>),
    EdgeSummaryBySrcDstName((EdgeSummary, Option<f64>)),
    NodeFragmentsByIdTimeRange(Vec<(TimestampMilli, FragmentContent)>),
    EdgeFragmentsByIdTimeRange(Vec<(TimestampMilli, FragmentContent)>),
    OutgoingEdges(Vec<(Option<f64>, SrcId, DstId, EdgeName)>),
    IncomingEdges(Vec<(Option<f64>, DstId, SrcId, EdgeName)>),
    AllNodes(Vec<(Id, NodeName, NodeSummary)>),
    AllEdges(Vec<(Option<f64>, SrcId, DstId, EdgeName)>),
    NodesBySummaryHash(Vec<NodeSummaryLookupResult>),
    EdgesBySummaryHash(Vec<EdgeSummaryLookupResult>),
}

impl Query {
    pub async fn execute(&self, storage: &Storage) -> Result<QueryResult> {
        match self {
            Query::NodeById(q) => q.execute(storage).await.map(QueryResult::NodeById),
            Query::NodesByIdsMulti(q) => q.execute(storage).await.map(QueryResult::NodesByIdsMulti),
            Query::EdgeSummaryBySrcDstName(q) => {
                q.execute(storage).await.map(QueryResult::EdgeSummaryBySrcDstName)
            }
            Query::NodeFragmentsByIdTimeRange(q) => {
                q.execute(storage).await.map(QueryResult::NodeFragmentsByIdTimeRange)
            }
            Query::EdgeFragmentsByIdTimeRange(q) => {
                q.execute(storage).await.map(QueryResult::EdgeFragmentsByIdTimeRange)
            }
            Query::OutgoingEdges(q) => q.execute(storage).await.map(QueryResult::OutgoingEdges),
            Query::IncomingEdges(q) => q.execute(storage).await.map(QueryResult::IncomingEdges),
            Query::AllNodes(q) => q.execute(storage).await.map(QueryResult::AllNodes),
            Query::AllEdges(q) => q.execute(storage).await.map(QueryResult::AllEdges),
            Query::NodesBySummaryHash(q) => {
                q.execute(storage).await.map(QueryResult::NodesBySummaryHash)
            }
            Query::EdgesBySummaryHash(q) => {
                q.execute(storage).await.map(QueryResult::EdgesBySummaryHash)
            }
        }
    }
}

impl RequestMeta for Query {
    type Reply = QueryResult;
    type Options = ();

    fn request_kind(&self) -> &'static str {
        match self {
            Query::NodeById(q) => q.request_kind(),
            Query::NodesByIdsMulti(q) => q.request_kind(),
            Query::EdgeSummaryBySrcDstName(q) => q.request_kind(),
            Query::NodeFragmentsByIdTimeRange(q) => q.request_kind(),
            Query::EdgeFragmentsByIdTimeRange(q) => q.request_kind(),
            Query::OutgoingEdges(q) => q.request_kind(),
            Query::IncomingEdges(q) => q.request_kind(),
            Query::AllNodes(q) => q.request_kind(),
            Query::AllEdges(q) => q.request_kind(),
            Query::NodesBySummaryHash(q) => q.request_kind(),
            Query::EdgesBySummaryHash(q) => q.request_kind(),
        }
    }
}

macro_rules! impl_request_meta {
    ($ty:ty, $reply:ty, $kind:expr) => {
        impl RequestMeta for $ty {
            type Reply = $reply;
            type Options = ();

            fn request_kind(&self) -> &'static str {
                $kind
            }
        }
    };
}

impl_request_meta!(NodeById, (NodeName, NodeSummary), "node_by_id");
impl_request_meta!(NodesByIdsMulti, Vec<(Id, NodeName, NodeSummary)>, "nodes_by_ids_multi");
impl_request_meta!(EdgeSummaryBySrcDstName, (EdgeSummary, Option<f64>), "edge_summary_by_src_dst_name");
impl_request_meta!(NodeFragmentsByIdTimeRange, Vec<(TimestampMilli, FragmentContent)>, "node_fragments_by_id_time_range");
impl_request_meta!(EdgeFragmentsByIdTimeRange, Vec<(TimestampMilli, FragmentContent)>, "edge_fragments_by_id_time_range");
impl_request_meta!(OutgoingEdges, Vec<(Option<f64>, SrcId, DstId, EdgeName)>, "outgoing_edges");
impl_request_meta!(IncomingEdges, Vec<(Option<f64>, DstId, SrcId, EdgeName)>, "incoming_edges");
impl_request_meta!(AllNodes, Vec<(Id, NodeName, NodeSummary)>, "all_nodes");
impl_request_meta!(AllEdges, Vec<(Option<f64>, SrcId, DstId, EdgeName)>, "all_edges");
impl_request_meta!(NodesBySummaryHash, Vec<NodeSummaryLookupResult>, "nodes_by_summary_hash");
impl_request_meta!(EdgesBySummaryHash, Vec<EdgeSummaryLookupResult>, "edges_by_summary_hash");

/// Query parameters for finding a node by its ID.
///
/// This struct contains only user-facing parameters and can be:
/// - Constructed via struct initialization
/// - Cloned and reused
///
/// # Time Dimensions
///
/// The graph supports two orthogonal time dimensions:
/// - **System time** (`as_of_system_time`): When versions were created/superseded (ValidSince/ValidUntil)
/// - **Application time** (`reference_ts_millis`): Business validity (ActivePeriod)
#[derive(Debug, Clone, PartialEq)]
pub struct NodeById {
    /// The entity ID to search for
    pub id: Id,

    /// Reference timestamp for ActivePeriod (application/business time) validity checks.
    /// If None, defaults to current time in the query executor.
    /// Records without an ActivePeriod (None) are considered always valid.
    pub reference_ts_millis: Option<TimestampMilli>,

    /// Point-in-time query for system time (ValidSince/ValidUntil).
    /// If None, returns the current version (ValidUntil = None).
    /// If Some(ts), returns the version that was valid at that system time.
    pub as_of_system_time: Option<TimestampMilli>,
}


/// Query parameters for batch lookup of multiple nodes by ID.
///
/// More efficient than multiple `NodeById` calls as it uses RocksDB's
/// `multi_get_cf()` for batched reads. Missing IDs are silently omitted
/// from results.
///
/// # Example
///
/// ```ignore
/// let nodes = NodesByIdsMulti::new(vec![id1, id2, id3], None)
///     .run(&reader, timeout)
///     .await?;
/// // Returns Vec<(Id, NodeName, NodeSummary)> for found nodes
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NodesByIdsMulti {
    /// Node IDs to look up
    pub ids: Vec<Id>,

    /// Reference timestamp for ActivePeriod (application time) validity checks.
    /// If None, defaults to current time in the query executor.
    pub reference_ts_millis: Option<TimestampMilli>,

    /// Point-in-time query for system time (ValidSince/ValidUntil).
    /// If None, returns the current version. If Some(ts), returns versions valid at that time.
    pub as_of_system_time: Option<TimestampMilli>,
}


/// Query parameters for scanning node fragments by ID with time range filtering.
///
/// This struct contains only user-facing parameters and can be:
/// - Constructed via struct initialization
/// - Cloned and reused
#[derive(Debug, Clone, PartialEq)]
pub struct NodeFragmentsByIdTimeRange {
    /// The entity ID to search for
    pub id: Id,

    /// Time range bounds (start_bound, end_bound)
    /// Use std::ops::Bound for idiomatic Rust range specification
    pub time_range: (Bound<TimestampMilli>, Bound<TimestampMilli>),

    /// Reference timestamp for temporal validity checks
    /// If None, defaults to current time in the query executor
    pub reference_ts_millis: Option<TimestampMilli>,
}


/// Query parameters for scanning edge fragments by source ID, destination ID, edge name, and time range.
///
/// This struct contains only user-facing parameters and can be:
/// - Constructed via struct initialization
/// - Cloned and reused
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeFragmentsByIdTimeRange {
    /// Source node ID
    pub source_id: SrcId,

    /// Destination node ID
    pub dest_id: DstId,

    /// Edge name
    pub edge_name: EdgeName,

    /// Time range bounds (start_bound, end_bound)
    /// Use std::ops::Bound for idiomatic Rust range specification
    pub time_range: (Bound<TimestampMilli>, Bound<TimestampMilli>),

    /// Reference timestamp for temporal validity checks
    /// If None, defaults to current time in the query executor
    pub reference_ts_millis: Option<TimestampMilli>,
}


/// Query parameters for finding an edge by source ID, destination ID, and name.
///
/// This struct contains only user-facing parameters and can be:
/// - Constructed via struct initialization
/// - Cloned and reused
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeSummaryBySrcDstName {
    /// Source node ID
    pub source_id: Id,

    /// Destination node ID
    pub dest_id: Id,

    /// Edge name
    pub name: String,

    /// Reference timestamp for ActivePeriod (application time) validity checks.
    /// If None, defaults to current time in the query executor.
    pub reference_ts_millis: Option<TimestampMilli>,

    /// Point-in-time query for system time (ValidSince/ValidUntil).
    /// If None, returns current version. If Some(ts), returns version valid at that time.
    pub as_of_system_time: Option<TimestampMilli>,
}


/// Query parameters for finding all outgoing edges from a node.
///
/// This struct contains only user-facing parameters and can be:
/// - Constructed via struct initialization with `..Default::default()`
/// - Deserialized from JSON or other formats
/// - Cloned and reused
///
/// # Examples
///
/// ```ignore
/// // Struct initialization
/// let query = OutgoingEdges {
///     id: node_id,
///     reference_ts_millis: Some(TimestampMilli::now()),
/// };
///
/// // Constructor
/// let query = OutgoingEdges::new(node_id, None);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct OutgoingEdges {
    /// The node ID to search for
    pub id: Id,

    /// Reference timestamp for ActivePeriod (application time) validity checks.
    /// If None, defaults to current time in the query executor.
    /// Records without an ActivePeriod (None) are considered always valid.
    pub reference_ts_millis: Option<TimestampMilli>,

    /// Point-in-time query for system time (ValidSince/ValidUntil).
    /// If None, returns current versions. If Some(ts), returns versions valid at that time.
    pub as_of_system_time: Option<TimestampMilli>,
}


/// Query parameters for finding all incoming edges to a node.
///
/// This struct contains only user-facing parameters and can be:
/// - Constructed via struct initialization with `..Default::default()`
/// - Deserialized from JSON or other formats
/// - Cloned and reused
#[derive(Debug, Clone, PartialEq)]
pub struct IncomingEdges {
    /// The node ID to search for
    pub id: DstId,

    /// Reference timestamp for ActivePeriod (application time) validity checks.
    /// If None, defaults to current time in the query executor.
    /// Records without an ActivePeriod (None) are considered always valid.
    pub reference_ts_millis: Option<TimestampMilli>,

    /// Point-in-time query for system time (ValidSince/ValidUntil).
    /// If None, returns current versions. If Some(ts), returns versions valid at that time.
    pub as_of_system_time: Option<TimestampMilli>,
}


/// Query parameters for enumerating all nodes with pagination.
///
/// This is the query-based interface for graph enumeration, useful for
/// graph algorithms that need to iterate over all nodes (e.g., PageRank).
///
/// # Examples
///
/// ```ignore
/// // Get first page of nodes
/// let nodes = AllNodes::new(1000)
///     .run(&reader, timeout)
///     .await?;
///
/// // Get next page using cursor
/// if let Some((last_id, _, _)) = nodes.last() {
///     let next_page = AllNodes::new(1000)
///         .with_cursor(*last_id)
///         .run(&reader, timeout)
///         .await?;
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AllNodes {
    /// Cursor for pagination - last ID from previous page (exclusive)
    pub last: Option<Id>,

    /// Maximum number of records to return
    pub limit: usize,

    /// Reference timestamp for temporal validity checks
    /// If None, defaults to current time in the query executor
    pub reference_ts_millis: Option<TimestampMilli>,
}


/// Query parameters for enumerating all edges with pagination.
///
/// This is the query-based interface for graph enumeration, useful for
/// graph algorithms that need to iterate over all edges (e.g., Kruskal's MST).
///
/// # Examples
///
/// ```ignore
/// // Get first page of edges
/// let edges = AllEdges::new(1000)
///     .run(&reader, timeout)
///     .await?;
///
/// // Get next page using cursor
/// if let Some((_, src, dst, name)) = edges.last() {
///     let next_page = AllEdges::new(1000)
///         .with_cursor((*src, *dst, name.clone()))
///         .run(&reader, timeout)
///         .await?;
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AllEdges {
    /// Cursor for pagination - (src_id, dst_id, edge_name) from previous page (exclusive)
    pub last: Option<(SrcId, DstId, EdgeName)>,

    /// Maximum number of records to return
    pub limit: usize,

    /// Reference timestamp for temporal validity checks
    /// If None, defaults to current time in the query executor
    pub reference_ts_millis: Option<TimestampMilli>,
}


impl NodeById {
    /// Create a new query request for the current version.
    pub fn new(id: Id, reference_ts_millis: Option<TimestampMilli>) -> Self {
        Self {
            id,
            reference_ts_millis,
            as_of_system_time: None,
        }
    }

    /// Create a point-in-time query at a specific system time.
    pub fn as_of(id: Id, system_time: TimestampMilli, reference_ts_millis: Option<TimestampMilli>) -> Self {
        Self {
            id,
            reference_ts_millis,
            as_of_system_time: Some(system_time),
        }
    }
}

impl NodeById {
    /// Execute this query directly on storage (for testing and simple use cases).
    pub async fn execute_on(&self, storage: &Storage) -> Result<(NodeName, NodeSummary)> {
        self.execute(storage).await
    }
}

impl NodesByIdsMulti {
    /// Create a new batch query request for current versions.
    pub fn new(ids: Vec<Id>, reference_ts_millis: Option<TimestampMilli>) -> Self {
        Self {
            ids,
            reference_ts_millis,
            as_of_system_time: None,
        }
    }

    /// Create a point-in-time batch query at a specific system time.
    pub fn as_of(ids: Vec<Id>, system_time: TimestampMilli, reference_ts_millis: Option<TimestampMilli>) -> Self {
        Self {
            ids,
            reference_ts_millis,
            as_of_system_time: Some(system_time),
        }
    }
}

impl NodeFragmentsByIdTimeRange {
    /// Create a new query request.
    pub fn new(
        id: Id,
        time_range: (Bound<TimestampMilli>, Bound<TimestampMilli>),
        reference_ts_millis: Option<TimestampMilli>,
    ) -> Self {
        Self {
            id,
            time_range,
            reference_ts_millis,
        }
    }

    /// Check if a timestamp falls within this range.
    pub fn contains(&self, ts: TimestampMilli) -> bool {
        timestamp_in_range(ts, &self.time_range)
    }

    /// Execute this query directly on storage (for testing and simple use cases).
    pub async fn execute_on(&self, storage: &Storage) -> Result<Vec<(TimestampMilli, FragmentContent)>> {
        self.execute(storage).await
    }
}

impl EdgeFragmentsByIdTimeRange {
    /// Create a new query request.
    pub fn new(
        source_id: SrcId,
        dest_id: DstId,
        edge_name: EdgeName,
        time_range: (Bound<TimestampMilli>, Bound<TimestampMilli>),
        reference_ts_millis: Option<TimestampMilli>,
    ) -> Self {
        Self {
            source_id,
            dest_id,
            edge_name,
            time_range,
            reference_ts_millis,
        }
    }

    /// Check if a timestamp falls within this range.
    pub fn contains(&self, ts: TimestampMilli) -> bool {
        timestamp_in_range(ts, &self.time_range)
    }
}

impl EdgeSummaryBySrcDstName {
    /// Create a new query request for the current version.
    pub fn new(
        source_id: SrcId,
        dest_id: DstId,
        name: String,
        reference_ts_millis: Option<TimestampMilli>,
    ) -> Self {
        Self {
            source_id,
            dest_id,
            name,
            reference_ts_millis,
            as_of_system_time: None,
        }
    }

    /// Create a point-in-time query at a specific system time.
    pub fn as_of(
        source_id: SrcId,
        dest_id: DstId,
        name: String,
        system_time: TimestampMilli,
        reference_ts_millis: Option<TimestampMilli>,
    ) -> Self {
        Self {
            source_id,
            dest_id,
            name,
            reference_ts_millis,
            as_of_system_time: Some(system_time),
        }
    }
}

impl EdgeSummaryBySrcDstName {
    /// Execute this query directly on storage (for testing and simple use cases).
    pub async fn execute_on(&self, storage: &Storage) -> Result<(EdgeSummary, Option<f64>)> {
        self.execute(storage).await
    }
}

impl OutgoingEdges {
    /// Create a new query request for current versions.
    pub fn new(id: Id, reference_ts_millis: Option<TimestampMilli>) -> Self {
        Self {
            id,
            reference_ts_millis,
            as_of_system_time: None,
        }
    }

    /// Create a point-in-time query at a specific system time.
    pub fn as_of(id: Id, system_time: TimestampMilli, reference_ts_millis: Option<TimestampMilli>) -> Self {
        Self {
            id,
            reference_ts_millis,
            as_of_system_time: Some(system_time),
        }
    }

    /// Execute this query directly on storage (for testing and simple use cases).
    pub async fn execute_on(&self, storage: &Storage) -> Result<Vec<(Option<f64>, SrcId, DstId, EdgeName)>> {
        self.execute(storage).await
    }
}

impl IncomingEdges {
    /// Create a new query request for current versions.
    pub fn new(id: DstId, reference_ts_millis: Option<TimestampMilli>) -> Self {
        Self {
            id,
            reference_ts_millis,
            as_of_system_time: None,
        }
    }

    /// Create a point-in-time query at a specific system time.
    pub fn as_of(id: DstId, system_time: TimestampMilli, reference_ts_millis: Option<TimestampMilli>) -> Self {
        Self {
            id,
            reference_ts_millis,
            as_of_system_time: Some(system_time),
        }
    }

    /// Execute this query directly on storage (for testing and simple use cases).
    pub async fn execute_on(&self, storage: &Storage) -> Result<Vec<(Option<f64>, DstId, SrcId, EdgeName)>> {
        self.execute(storage).await
    }
}

impl AllNodes {
    /// Create a new AllNodes query with the specified limit.
    pub fn new(limit: usize) -> Self {
        Self {
            last: None,
            limit,
            reference_ts_millis: None,
        }
    }

    /// Set the cursor for pagination (exclusive start).
    pub fn with_cursor(mut self, last: Id) -> Self {
        self.last = Some(last);
        self
    }

    /// Set the reference timestamp for temporal validity checks.
    pub fn with_reference_time(mut self, ts: TimestampMilli) -> Self {
        self.reference_ts_millis = Some(ts);
        self
    }
}

impl AllEdges {
    /// Create a new AllEdges query with the specified limit.
    pub fn new(limit: usize) -> Self {
        Self {
            last: None,
            limit,
            reference_ts_millis: None,
        }
    }

    /// Set the cursor for pagination (exclusive start).
    pub fn with_cursor(mut self, last: (SrcId, DstId, EdgeName)) -> Self {
        self.last = Some(last);
        self
    }

    /// Set the reference timestamp for temporal validity checks.
    pub fn with_reference_time(mut self, ts: TimestampMilli) -> Self {
        self.reference_ts_millis = Some(ts);
        self
    }
}

// ============================================================================
// QueryReply - typed replies for graph queries
// ============================================================================

pub trait QueryReply: Send {
    type Reply: Send + 'static;
    fn into_query(self) -> Query;
    fn from_result(result: QueryResult) -> Result<Self::Reply>;
}

#[async_trait::async_trait]
impl<Q> Runnable<super::Reader> for Q
where
    Q: QueryReply,
{
    type Output = Q::Reply;

    async fn run(self, reader: &super::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = oneshot::channel();
        let request = QueryRequest {
            payload: self.into_query(),
            options: (),
            reply: Some(result_tx),
            timeout: Some(timeout),
            request_id: new_request_id(),
            created_at: std::time::Instant::now(),
        };
        reader.send_query(request).await?;
        let result = result_rx.await??;
        Q::from_result(result)
    }
}

impl QueryReply for NodeById {
    type Reply = (NodeName, NodeSummary);

    fn into_query(self) -> Query {
        Query::NodeById(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::NodeById(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for NodesByIdsMulti {
    type Reply = Vec<(Id, NodeName, NodeSummary)>;

    fn into_query(self) -> Query {
        Query::NodesByIdsMulti(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::NodesByIdsMulti(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for NodeFragmentsByIdTimeRange {
    type Reply = Vec<(TimestampMilli, FragmentContent)>;

    fn into_query(self) -> Query {
        Query::NodeFragmentsByIdTimeRange(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::NodeFragmentsByIdTimeRange(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for EdgeFragmentsByIdTimeRange {
    type Reply = Vec<(TimestampMilli, FragmentContent)>;

    fn into_query(self) -> Query {
        Query::EdgeFragmentsByIdTimeRange(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::EdgeFragmentsByIdTimeRange(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for EdgeSummaryBySrcDstName {
    type Reply = (EdgeSummary, Option<f64>);

    fn into_query(self) -> Query {
        Query::EdgeSummaryBySrcDstName(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::EdgeSummaryBySrcDstName(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for OutgoingEdges {
    type Reply = Vec<(Option<f64>, SrcId, DstId, EdgeName)>;

    fn into_query(self) -> Query {
        Query::OutgoingEdges(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::OutgoingEdges(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for IncomingEdges {
    type Reply = Vec<(Option<f64>, DstId, SrcId, EdgeName)>;

    fn into_query(self) -> Query {
        Query::IncomingEdges(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::IncomingEdges(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for AllNodes {
    type Reply = Vec<(Id, NodeName, NodeSummary)>;

    fn into_query(self) -> Query {
        Query::AllNodes(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::AllNodes(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for AllEdges {
    type Reply = Vec<(Option<f64>, SrcId, DstId, EdgeName)>;

    fn into_query(self) -> Query {
        Query::AllEdges(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::AllEdges(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

/// Implement QueryExecutor for NodeById
/// (claude, 2026-02-06: VERSIONING with point-in-time query support)
#[async_trait::async_trait]
impl QueryExecutor for NodeById {
    type Output = (NodeName, NodeSummary);

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let params = self;
        tracing::debug!(
            id = %params.id,
            as_of = ?params.as_of_system_time,
            "Executing NodeById query"
        );

        // Default None to current time for ActivePeriod validity checks
        let ref_time = params
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let id = params.id;

        // Find version via prefix scan (VERSIONING)
        // If as_of_system_time is Some, find version valid at that time
        // Otherwise find current version (ValidUntil = None)
        let (_key_bytes, value) = if let Ok(db) = storage.db() {
            find_node_version_at_time_readonly(db, id, params.as_of_system_time)?
        } else {
            let txn_db = storage.transaction_db()?;
            find_node_version_at_time_readwrite(txn_db, id, params.as_of_system_time)?
        }
        .ok_or_else(|| {
            if params.as_of_system_time.is_some() {
                anyhow::anyhow!("Node {} not found at system time {:?}", id, params.as_of_system_time)
            } else {
                anyhow::anyhow!("Node not found: {}", id)
            }
        })?;

        // Field indices (VERSIONING): 0=ValidUntil, 1=ActivePeriod, 2=NameHash, 3=SummaryHash, 4=Version, 5=Deleted
        // Check if deleted (only for current time queries, not time-travel)
        // For time-travel queries, we want to see the node's state at that past time
        if params.as_of_system_time.is_none() && value.5 {
            return Err(anyhow::anyhow!("Node {} has been deleted", id));
        }

        // Check temporal validity (ActivePeriod at index 1)
        if !schema::is_active_at_time(&value.1, ref_time) {
            return Err(anyhow::anyhow!(
                "Node {} not valid at time {}",
                id,
                ref_time.0
            ));
        }

        // Resolve NameHash to String (uses cache) - NameHash is at index 2
        let node_name = resolve_name(storage, value.2)?;

        // Resolve summary from cold CF - SummaryHash is at index 3
        let summary = resolve_node_summary(storage, value.3)?;

        Ok((node_name, summary))
    }
}

/// Implement QueryExecutor for NodesByIdsMulti
///
/// (claude, 2026-02-06: VERSIONING with point-in-time query support)
/// Missing nodes and temporally invalid nodes are silently omitted from results.
#[async_trait::async_trait]
impl QueryExecutor for NodesByIdsMulti {
    type Output = Vec<(Id, NodeName, NodeSummary)>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let params = self;
        tracing::debug!(
            count = params.ids.len(),
            as_of = ?params.as_of_system_time,
            "Executing NodesByIdsMulti query"
        );

        if params.ids.is_empty() {
            return Ok(Vec::new());
        }

        // Default None to current time for temporal validity checks
        let ref_time = params
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        // With VERSIONING, we need prefix scans for each ID to find version at time
        // Collect valid entries with their NameHash and SummaryHash first, then resolve
        // Field indices (VERSIONING): 0=ValidUntil, 1=ActivePeriod, 2=NameHash, 3=SummaryHash, 4=Version, 5=Deleted
        let mut valid_entries: Vec<(Id, NameHash, Option<SummaryHash>)> = Vec::with_capacity(params.ids.len());

        for id in &params.ids {
            let result = if let Ok(db) = storage.db() {
                find_node_version_at_time_readonly(db, *id, params.as_of_system_time)
            } else {
                let txn_db = storage.transaction_db()?;
                find_node_version_at_time_readwrite(txn_db, *id, params.as_of_system_time)
            };

            match result {
                Ok(Some((_key_bytes, value))) => {
                    // Skip deleted nodes
                    if value.5 {
                        tracing::trace!(id = %id, "Skipping node: deleted");
                        continue;
                    }
                    // Check temporal validity (ActivePeriod is at index 1)
                    if schema::is_active_at_time(&value.1, ref_time) {
                        // NameHash at index 2, SummaryHash at index 3
                        valid_entries.push((*id, value.2, value.3));
                    } else {
                        tracing::trace!(
                            id = %id,
                            "Skipping node: not valid at time {}",
                            ref_time.0
                        );
                    }
                }
                Ok(None) => {
                    // Node not found - silently skip
                    tracing::trace!(id = %id, "Node not found");
                }
                Err(e) => {
                    tracing::warn!(id = %id, error = %e, "Error fetching node");
                }
            }
        }

        // Resolve NameHashes to Strings and SummaryHashes to Summaries
        let mut output = Vec::with_capacity(valid_entries.len());
        for (id, name_hash, summary_hash) in valid_entries {
            let node_name = resolve_name(storage, name_hash)?;
            let summary = resolve_node_summary(storage, summary_hash)?;
            output.push((id, node_name, summary));
        }

        Ok(output)
    }
}

/// Implement QueryExecutor for NodeFragmentsByIdTimeRange
#[async_trait::async_trait]
impl QueryExecutor for NodeFragmentsByIdTimeRange {
    type Output = Vec<(TimestampMilli, FragmentContent)>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let params = self;
        tracing::debug!(id = %params.id, time_range = ?params.time_range, "Executing NodeFragmentsByIdTimeRange query");

        use std::ops::Bound;

        // Default None to current time for temporal validity checks
        let ref_time = params
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let id = params.id;
        let mut fragments: Vec<(TimestampMilli, FragmentContent)> = Vec::new();

        // Construct optimal starting key based on start bound
        let start_key = match &params.time_range.0 {
            Bound::Unbounded => {
                let mut key = Vec::with_capacity(24);
                key.extend_from_slice(&id.into_bytes());
                key.extend_from_slice(&0u64.to_be_bytes());
                key
            }
            Bound::Included(start_ts) => {
                schema::NodeFragments::key_to_bytes(&schema::NodeFragmentCfKey(id, *start_ts))
            }
            Bound::Excluded(start_ts) => schema::NodeFragments::key_to_bytes(
                &schema::NodeFragmentCfKey(id, TimestampMilli(start_ts.0 + 1)),
            ),
        };

        iterate_cf!(storage, schema::NodeFragments, start_key, |item| {
            let (key_bytes, value_bytes) = item?;
            let key: schema::NodeFragmentCfKey = schema::NodeFragments::key_from_bytes(&key_bytes)
                .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

            if key.0 != id {
                break;
            }

            let timestamp = key.1;

            match &params.time_range.1 {
                Bound::Unbounded => { /* continue scanning */ }
                Bound::Included(end_ts) => {
                    if timestamp.0 > end_ts.0 {
                        break;
                    }
                }
                Bound::Excluded(end_ts) => {
                    if timestamp.0 >= end_ts.0 {
                        break;
                    }
                }
            }

            let value: schema::NodeFragmentCfValue =
                schema::NodeFragments::value_from_bytes(&value_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

            // Always check temporal validity - skip invalid fragments
            if !schema::is_active_at_time(&value.0, ref_time) {
                continue;
            }

            fragments.push((timestamp, value.1));
        });

        Ok(fragments)
    }
}

/// Implement QueryExecutor for EdgeFragmentsByIdTimeRange
#[async_trait::async_trait]
impl QueryExecutor for EdgeFragmentsByIdTimeRange {
    type Output = Vec<(TimestampMilli, FragmentContent)>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let params = self;
        tracing::debug!(
            src_id = %params.source_id,
            dst_id = %params.dest_id,
            edge_name = %params.edge_name,
            time_range = ?params.time_range,
            "Executing EdgeFragmentsByIdTimeRange query"
        );

        use std::ops::Bound;

        // Default None to current time for temporal validity checks
        let ref_time = params
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let source_id = params.source_id;
        let dest_id = params.dest_id;
        // Convert edge_name String to NameHash for key construction
        let edge_name_hash = NameHash::from_name(&params.edge_name);
        let mut fragments: Vec<(TimestampMilli, FragmentContent)> = Vec::new();

        // Construct optimal starting key based on start bound
        // EdgeFragmentCfKey: (SrcId, DstId, NameHash, TimestampMilli) - now fixed 40 bytes
        let start_key = match &params.time_range.0 {
            Bound::Unbounded => {
                schema::EdgeFragments::key_to_bytes(
                    &schema::EdgeFragmentCfKey(source_id, dest_id, edge_name_hash, TimestampMilli(0)),
                )
            }
            Bound::Included(start_ts) => schema::EdgeFragments::key_to_bytes(
                &schema::EdgeFragmentCfKey(source_id, dest_id, edge_name_hash, *start_ts),
            ),
            Bound::Excluded(start_ts) => {
                schema::EdgeFragments::key_to_bytes(&schema::EdgeFragmentCfKey(
                    source_id,
                    dest_id,
                    edge_name_hash,
                    TimestampMilli(start_ts.0 + 1),
                ))
            }
        };

        iterate_cf!(storage, schema::EdgeFragments, start_key, |item| {
            let (key_bytes, value_bytes) = item?;
            let key: schema::EdgeFragmentCfKey = schema::EdgeFragments::key_from_bytes(&key_bytes)
                .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

            // Check if we're still in the same edge (source_id, dest_id, edge_name_hash)
            if key.0 != source_id || key.1 != dest_id || key.2 != edge_name_hash {
                break;
            }

            let timestamp = key.3;

            match &params.time_range.1 {
                Bound::Unbounded => { /* continue scanning */ }
                Bound::Included(end_ts) => {
                    if timestamp.0 > end_ts.0 {
                        break;
                    }
                }
                Bound::Excluded(end_ts) => {
                    if timestamp.0 >= end_ts.0 {
                        break;
                    }
                }
            }

            let value: schema::EdgeFragmentCfValue =
                schema::EdgeFragments::value_from_bytes(&value_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

            // Always check temporal validity - skip invalid fragments
            if !schema::is_active_at_time(&value.0, ref_time) {
                continue;
            }

            fragments.push((timestamp, value.1));
        });

        Ok(fragments)
    }
}

/// Implement QueryExecutor for EdgeSummaryBySrcDstName
/// (claude, 2026-02-06, in-progress: VERSIONING prefix scan and new field indices)
#[async_trait::async_trait]
impl QueryExecutor for EdgeSummaryBySrcDstName {
    type Output = (EdgeSummary, Option<f64>);

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let params = self;
        tracing::debug!(
            src_id = %params.source_id,
            dst_id = %params.dest_id,
            name = %params.name,
            "Executing EdgeSummaryBySrcDstName query"
        );

        // Default None to current time for temporal validity checks
        let ref_time = params
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let source_id = params.source_id;
        let dest_id = params.dest_id;
        let name = &params.name;
        // Convert edge name to NameHash for key construction
        let name_hash = NameHash::from_name(name);

        // Find edge version via prefix scan (point-in-time if as_of_system_time is set)
        let value = if let Ok(db) = storage.db() {
            find_forward_edge_version_at_time_readonly(db, source_id, dest_id, name_hash, params.as_of_system_time)?
                .map(|(_, v)| v)
        } else {
            let txn_db = storage.transaction_db()?;
            find_forward_edge_version_at_time_readwrite(txn_db, source_id, dest_id, name_hash, params.as_of_system_time)?
                .map(|(_, v)| v)
        };

        let value = value.ok_or_else(|| {
            if params.as_of_system_time.is_some() {
                anyhow::anyhow!(
                    "Edge not found at system time {:?}: source={}, dest={}, name={}",
                    params.as_of_system_time,
                    source_id,
                    dest_id,
                    name
                )
            } else {
                anyhow::anyhow!(
                    "Edge not found: source={}, dest={}, name={}",
                    source_id,
                    dest_id,
                    name
                )
            }
        })?;

        // Field indices (VERSIONING): 0=ValidUntil, 1=ActivePeriod, 2=Weight, 3=SummaryHash, 4=Version, 5=Deleted
        // Check if deleted (only for current time queries, not time-travel)
        if params.as_of_system_time.is_none() && value.5 {
            return Err(anyhow::anyhow!(
                "Edge deleted: source={}, dest={}, name={}",
                source_id,
                dest_id,
                name
            ));
        }

        // Check temporal validity (ActivePeriod at index 1)
        if !schema::is_active_at_time(&value.1, ref_time) {
            return Err(anyhow::anyhow!(
                "Edge not valid at time {}: source={}, dest={}, name={}",
                ref_time.0,
                source_id,
                dest_id,
                name
            ));
        }

        // Field order: ValidUntil, ActivePeriod, Weight, SummaryHash, Version, Deleted
        // Return (summary, weight) - resolve summary from cold CF
        let summary = resolve_edge_summary(storage, value.3)?;
        Ok((summary, value.2))
    }
}

/// Implement QueryExecutor for OutgoingEdges
#[async_trait::async_trait]
/// (claude, 2026-02-06, in-progress: VERSIONING prefix scan and new field indices)
impl QueryExecutor for OutgoingEdges {
    type Output = Vec<(Option<f64>, SrcId, DstId, EdgeName)>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let params = self;
        tracing::debug!(node_id = %params.id, "Executing OutgoingEdges query");

        // Default None to current time for temporal validity checks
        let ref_time = params
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let id = params.id;
        // Use HashSet for deduplication with versioned keys
        let mut seen_edges: std::collections::HashSet<(Id, Id, NameHash)> = std::collections::HashSet::new();
        let mut edges_with_hash: Vec<(Option<f64>, SrcId, DstId, NameHash)> = Vec::new();
        let prefix = id.into_bytes();

        if let Ok(db) = storage.db() {
            let cf = db.cf_handle(schema::ForwardEdges::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!(
                    "Column family '{}' not found",
                    schema::ForwardEdges::CF_NAME
                )
            })?;

            let iter = db.iterator_cf(
                cf,
                rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
            );

            for item in iter {
                let (key_bytes, value_bytes) = item?;
                // Stop if we've gone past our prefix (src_id = 16 bytes)
                if !key_bytes.starts_with(&prefix) {
                    break;
                }

                let key: schema::ForwardEdgeCfKey =
                    schema::ForwardEdges::key_from_bytes(&key_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

                let value: schema::ForwardEdgeCfValue =
                    schema::ForwardEdges::value_from_bytes(&value_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

                // Field indices (VERSIONING): 0=ValidUntil, 1=ActivePeriod, 2=Weight, 3=SummaryHash, 4=Version, 5=Deleted
                // Skip non-current versions
                if value.0.is_some() {
                    continue;
                }
                // Skip deleted edges
                if value.5 {
                    continue;
                }
                // Check temporal validity (ActivePeriod at index 1)
                if !schema::is_active_at_time(&value.1, ref_time) {
                    continue;
                }

                let source_id = key.0;
                let dest_id = key.1;
                let edge_name_hash = key.2;
                let edge_topology = (source_id, dest_id, edge_name_hash);
                if seen_edges.insert(edge_topology) {
                    let weight = value.2;
                    edges_with_hash.push((weight, source_id, dest_id, edge_name_hash));
                }
            }
        } else {
            let txn_db = storage.transaction_db()?;
            let cf = txn_db
                .cf_handle(schema::ForwardEdges::CF_NAME)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Column family '{}' not found",
                        schema::ForwardEdges::CF_NAME
                    )
                })?;

            let iter = txn_db.iterator_cf(
                cf,
                rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
            );

            for item in iter {
                let (key_bytes, value_bytes) = item?;
                // Stop if we've gone past our prefix (src_id = 16 bytes)
                if !key_bytes.starts_with(&prefix) {
                    break;
                }

                let key: schema::ForwardEdgeCfKey =
                    schema::ForwardEdges::key_from_bytes(&key_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

                let value: schema::ForwardEdgeCfValue =
                    schema::ForwardEdges::value_from_bytes(&value_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

                // Field indices (VERSIONING): 0=ValidUntil, 1=ActivePeriod, 2=Weight, 3=SummaryHash, 4=Version, 5=Deleted
                // Skip non-current versions
                if value.0.is_some() {
                    continue;
                }
                // Skip deleted edges
                if value.5 {
                    continue;
                }
                // Check temporal validity (ActivePeriod at index 1)
                if !schema::is_active_at_time(&value.1, ref_time) {
                    continue;
                }

                let source_id = key.0;
                let dest_id = key.1;
                let edge_name_hash = key.2;
                let edge_topology = (source_id, dest_id, edge_name_hash);
                if seen_edges.insert(edge_topology) {
                    let weight = value.2;
                    edges_with_hash.push((weight, source_id, dest_id, edge_name_hash));
                }
            }
        }

        // Resolve NameHashes to Strings (uses cache)
        let mut edges = Vec::with_capacity(edges_with_hash.len());
        for (weight, src_id, dst_id, name_hash) in edges_with_hash {
            let edge_name = resolve_name(storage, name_hash)?;
            edges.push((weight, src_id, dst_id, edge_name));
        }
        Ok(edges)
    }
}

/// Implement QueryExecutor for IncomingEdges
/// (claude, 2026-02-06, in-progress: VERSIONING prefix scan and new field indices)
#[async_trait::async_trait]
impl QueryExecutor for IncomingEdges {
    type Output = Vec<(Option<f64>, DstId, SrcId, EdgeName)>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let params = self;
        tracing::debug!(node_id = %params.id, "Executing IncomingEdges query");

        // Default None to current time for temporal validity checks
        let ref_time = params
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let id = params.id;
        // Use HashSet for deduplication with versioned keys
        let mut seen_edges: std::collections::HashSet<(Id, Id, NameHash)> = std::collections::HashSet::new();
        let mut edges_with_hash: Vec<(Option<f64>, DstId, SrcId, NameHash)> = Vec::new();
        let prefix = id.into_bytes();

        if let Ok(db) = storage.db() {
            let reverse_cf = db.cf_handle(schema::ReverseEdges::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!(
                    "Column family '{}' not found",
                    schema::ReverseEdges::CF_NAME
                )
            })?;

            let iter = db.iterator_cf(
                reverse_cf,
                rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
            );

            for item in iter {
                let (key_bytes, value_bytes) = item?;
                // Stop if we've gone past our prefix (dst_id = 16 bytes)
                if !key_bytes.starts_with(&prefix) {
                    break;
                }

                let key: schema::ReverseEdgeCfKey =
                    schema::ReverseEdges::key_from_bytes(&key_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

                let value: schema::ReverseEdgeCfValue =
                    schema::ReverseEdges::value_from_bytes(&value_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

                // Field indices (VERSIONING): 0=ValidUntil, 1=ActivePeriod
                // Skip non-current versions
                if value.0.is_some() {
                    continue;
                }
                // Check temporal validity (ActivePeriod at index 1)
                if !schema::is_active_at_time(&value.1, ref_time) {
                    continue;
                }

                let dest_id = key.0;
                let source_id = key.1;
                let edge_name_hash = key.2;

                // Deduplication check
                let edge_topology = (source_id, dest_id, edge_name_hash);
                if !seen_edges.insert(edge_topology) {
                    continue;
                }

                // Lookup weight from ForwardEdges CF via prefix scan
                let weight = if let Some((_, forward_value)) =
                    find_current_forward_edge_readonly(db, source_id, dest_id, edge_name_hash)?
                {
                    // Skip if deleted
                    if forward_value.5 {
                        continue;
                    }
                    forward_value.2 // Extract weight from field 2
                } else {
                    None
                };

                edges_with_hash.push((weight, dest_id, source_id, edge_name_hash));
            }
        } else {
            let txn_db = storage.transaction_db()?;
            let reverse_cf = txn_db
                .cf_handle(schema::ReverseEdges::CF_NAME)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Column family '{}' not found",
                        schema::ReverseEdges::CF_NAME
                    )
                })?;

            let iter = txn_db.iterator_cf(
                reverse_cf,
                rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
            );

            for item in iter {
                let (key_bytes, value_bytes) = item?;
                // Stop if we've gone past our prefix (dst_id = 16 bytes)
                if !key_bytes.starts_with(&prefix) {
                    break;
                }

                let key: schema::ReverseEdgeCfKey =
                    schema::ReverseEdges::key_from_bytes(&key_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

                let value: schema::ReverseEdgeCfValue =
                    schema::ReverseEdges::value_from_bytes(&value_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

                // Field indices (VERSIONING): 0=ValidUntil, 1=ActivePeriod
                // Skip non-current versions
                if value.0.is_some() {
                    continue;
                }
                // Check temporal validity (ActivePeriod at index 1)
                if !schema::is_active_at_time(&value.1, ref_time) {
                    continue;
                }

                let dest_id = key.0;
                let source_id = key.1;
                let edge_name_hash = key.2;

                // Deduplication check
                let edge_topology = (source_id, dest_id, edge_name_hash);
                if !seen_edges.insert(edge_topology) {
                    continue;
                }

                // Lookup weight from ForwardEdges CF via prefix scan
                let weight = if let Some((_, forward_value)) =
                    find_current_forward_edge_readwrite(txn_db, source_id, dest_id, edge_name_hash)?
                {
                    // Skip if deleted
                    if forward_value.5 {
                        continue;
                    }
                    forward_value.2 // Extract weight from field 2
                } else {
                    None
                };

                edges_with_hash.push((weight, dest_id, source_id, edge_name_hash));
            }
        }

        // Resolve NameHashes to Strings (uses cache)
        let mut edges = Vec::with_capacity(edges_with_hash.len());
        for (weight, dst_id, src_id, name_hash) in edges_with_hash {
            let edge_name = resolve_name(storage, name_hash)?;
            edges.push((weight, dst_id, src_id, edge_name));
        }
        Ok(edges)
    }

}

/// Implement QueryExecutor for AllNodes
/// Delegates to the scan module's Visitable implementation.
#[async_trait::async_trait]
impl QueryExecutor for AllNodes {
    type Output = Vec<(Id, NodeName, NodeSummary)>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let params = self;
        tracing::debug!(limit = params.limit, has_cursor = params.last.is_some(), "Executing AllNodes query");

        // Create scan request matching scan::AllNodes
        let scan_request = scan::AllNodes {
            last: params.last,
            limit: params.limit,
            reverse: false,
            reference_ts_millis: params.reference_ts_millis,
        };

        // Collect results via visitor pattern
        let mut results = Vec::with_capacity(params.limit);
        scan_request.accept(storage, &mut |record: &scan::NodeRecord| {
            results.push((record.id, record.name.clone(), record.summary.clone()));
            true // continue
        })?;

        Ok(results)
    }
}

/// Implement QueryExecutor for AllEdges
/// Delegates to the scan module's Visitable implementation.
#[async_trait::async_trait]
impl QueryExecutor for AllEdges {
    type Output = Vec<(Option<f64>, SrcId, DstId, EdgeName)>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let params = self;
        tracing::debug!(limit = params.limit, has_cursor = params.last.is_some(), "Executing AllEdges query");

        // Create scan request matching scan::AllEdges
        let scan_request = scan::AllEdges {
            last: params.last.clone(),
            limit: params.limit,
            reverse: false,
            reference_ts_millis: params.reference_ts_millis,
        };

        // Collect results via visitor pattern
        let mut results = Vec::with_capacity(params.limit);
        scan_request.accept(storage, &mut |record: &scan::EdgeRecord| {
            results.push((record.weight, record.src_id, record.dst_id, record.name.clone()));
            true // continue
        })?;

        Ok(results)
    }
}

// ============================================================================
// TransactionQueryExecutor Implementations
// ============================================================================

/// Implement TransactionQueryExecutor for NodeById
/// (claude, 2026-02-06, in-progress: VERSIONING prefix scan for current version)
impl TransactionQueryExecutor for NodeById {
    type Output = (NodeName, NodeSummary);

    fn execute_in_transaction(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        cache: &super::name_hash::NameCache,
    ) -> Result<Self::Output> {
        tracing::debug!(id = %self.id, "Executing NodeById query in transaction");

        // Default None to current time for temporal validity checks
        let ref_time = self
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        // Find current version via prefix scan (VERSIONING)
        let (_key_bytes, value) = find_current_node_version_txn(txn, txn_db, self.id)?
            .ok_or_else(|| anyhow::anyhow!("Node not found: {}", self.id))?;

        // Field indices (VERSIONING): 0=ValidUntil, 1=ActivePeriod, 2=NameHash, 3=SummaryHash, 4=Version, 5=Deleted
        // Check if deleted
        if value.5 {
            return Err(anyhow::anyhow!("Node {} has been deleted", self.id));
        }

        // Check temporal validity (ActivePeriod is at index 1)
        if !schema::is_active_at_time(&value.1, ref_time) {
            return Err(anyhow::anyhow!(
                "Node {} not valid at time {}",
                self.id,
                ref_time.0
            ));
        }

        // Resolve NameHash to String (uses cache) - NameHash is at index 2
        let node_name = resolve_name_from_txn(txn, txn_db, value.2, cache)?;
        // Resolve SummaryHash to NodeSummary (from cold CF) - SummaryHash is at index 3
        let summary = resolve_node_summary_from_txn(txn, txn_db, value.3)?;
        Ok((node_name, summary))
    }
}

/// Implement TransactionQueryExecutor for NodesByIdsMulti
/// (claude, 2026-02-06, in-progress: VERSIONING prefix scan for current versions)
impl TransactionQueryExecutor for NodesByIdsMulti {
    type Output = Vec<(Id, NodeName, NodeSummary)>;

    fn execute_in_transaction(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        cache: &super::name_hash::NameCache,
    ) -> Result<Self::Output> {
        tracing::debug!(count = self.ids.len(), "Executing NodesByIdsMulti query in transaction");

        if self.ids.is_empty() {
            return Ok(Vec::new());
        }

        let ref_time = self
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        // Collect entries with NameHash and SummaryHash first
        // Field indices (VERSIONING): 0=ValidUntil, 1=ActivePeriod, 2=NameHash, 3=SummaryHash, 4=Version, 5=Deleted
        let mut entries_with_hash: Vec<(Id, NameHash, Option<SummaryHash>)> = Vec::with_capacity(self.ids.len());

        for id in &self.ids {
            // Find current version via prefix scan (VERSIONING)
            match find_current_node_version_txn(txn, txn_db, *id) {
                Ok(Some((_key_bytes, value))) => {
                    // Skip deleted nodes
                    if value.5 {
                        continue;
                    }
                    // Check temporal validity (ActivePeriod is at index 1)
                    if schema::is_active_at_time(&value.1, ref_time) {
                        // NameHash at index 2, SummaryHash at index 3
                        entries_with_hash.push((*id, value.2, value.3));
                    }
                }
                Ok(None) => {
                    // Node not found - silently skip
                }
                Err(e) => {
                    tracing::warn!(id = %id, error = %e, "Failed to find node version");
                }
            }
        }

        // Resolve NameHashes to Strings and SummaryHashes to Summaries
        let mut output = Vec::with_capacity(entries_with_hash.len());
        for (id, name_hash, summary_hash) in entries_with_hash {
            let node_name = resolve_name_from_txn(txn, txn_db, name_hash, cache)?;
            let summary = resolve_node_summary_from_txn(txn, txn_db, summary_hash)?;
            output.push((id, node_name, summary));
        }

        Ok(output)
    }
}

/// Implement TransactionQueryExecutor for NodeFragmentsByIdTimeRange
impl TransactionQueryExecutor for NodeFragmentsByIdTimeRange {
    type Output = Vec<(TimestampMilli, FragmentContent)>;

    fn execute_in_transaction(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        _cache: &super::name_hash::NameCache,
    ) -> Result<Self::Output> {
        tracing::debug!(id = %self.id, time_range = ?self.time_range, "Executing NodeFragmentsByIdTimeRange query in transaction");

        let ref_time = self
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let cf = txn_db
            .cf_handle(schema::NodeFragments::CF_NAME)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Column family '{}' not found",
                    schema::NodeFragments::CF_NAME
                )
            })?;

        // Create prefix for this node's fragments
        let prefix_key = schema::NodeFragmentCfKey(self.id, TimestampMilli(0));
        let prefix_bytes = schema::NodeFragments::key_to_bytes(&prefix_key);
        let prefix_len = std::mem::size_of::<Id>();

        let mut results = Vec::new();

        // Use transaction iterator
        let iter = txn.iterator_cf(
            cf,
            rocksdb::IteratorMode::From(&prefix_bytes, rocksdb::Direction::Forward),
        );

        for item in iter {
            let (key_bytes, value_bytes) = item?;

            // Check prefix match
            if key_bytes.len() < prefix_len || &key_bytes[..prefix_len] != &prefix_bytes[..prefix_len]
            {
                break;
            }

            let key: schema::NodeFragmentCfKey =
                schema::NodeFragments::key_from_bytes(&key_bytes)?;
            let ts = key.1;

            // Check time range
            if !timestamp_in_range(ts, &self.time_range) {
                continue;
            }

            let value: schema::NodeFragmentCfValue =
                schema::NodeFragments::value_from_bytes(&value_bytes)?;

            // Check temporal validity
            if schema::is_active_at_time(&value.0, ref_time) {
                results.push((ts, value.1));
            }
        }

        Ok(results)
    }
}

/// Implement TransactionQueryExecutor for EdgeFragmentsByIdTimeRange
impl TransactionQueryExecutor for EdgeFragmentsByIdTimeRange {
    type Output = Vec<(TimestampMilli, FragmentContent)>;

    fn execute_in_transaction(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        _cache: &super::name_hash::NameCache,
    ) -> Result<Self::Output> {
        tracing::debug!(
            source_id = %self.source_id,
            dest_id = %self.dest_id,
            edge_name = %self.edge_name,
            time_range = ?self.time_range,
            "Executing EdgeFragmentsByIdTimeRange query in transaction"
        );

        let ref_time = self
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let cf = txn_db
            .cf_handle(schema::EdgeFragments::CF_NAME)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Column family '{}' not found",
                    schema::EdgeFragments::CF_NAME
                )
            })?;

        // Convert edge_name String to NameHash for key construction
        let edge_name_hash = NameHash::from_name(&self.edge_name);

        // Create prefix for this edge's fragments
        let prefix_key = schema::EdgeFragmentCfKey(
            self.source_id,
            self.dest_id,
            edge_name_hash,
            TimestampMilli(0),
        );
        let prefix_bytes = schema::EdgeFragments::key_to_bytes(&prefix_key);
        // Prefix is src_id + dst_id + name_hash (now fixed 40 bytes)

        let mut results = Vec::new();

        let iter = txn.iterator_cf(
            cf,
            rocksdb::IteratorMode::From(&prefix_bytes, rocksdb::Direction::Forward),
        );

        for item in iter {
            let (key_bytes, value_bytes) = item?;

            // Parse key to check if it matches our edge
            let key: schema::EdgeFragmentCfKey = match schema::EdgeFragments::key_from_bytes(&key_bytes)
            {
                Ok(k) => k,
                Err(_) => break,
            };

            // Check if this is still our edge
            if key.0 != self.source_id || key.1 != self.dest_id || key.2 != edge_name_hash {
                break;
            }

            let ts = key.3;

            // Check time range
            if !timestamp_in_range(ts, &self.time_range) {
                continue;
            }

            let value: schema::EdgeFragmentCfValue =
                schema::EdgeFragments::value_from_bytes(&value_bytes)?;

            // Check temporal validity
            if schema::is_active_at_time(&value.0, ref_time) {
                results.push((ts, value.1));
            }
        }

        Ok(results)
    }
}

/// Implement TransactionQueryExecutor for EdgeSummaryBySrcDstName
/// (claude, 2026-02-06, in-progress: VERSIONING prefix scan and new field indices)
impl TransactionQueryExecutor for EdgeSummaryBySrcDstName {
    type Output = (EdgeSummary, Option<f64>);

    fn execute_in_transaction(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        _cache: &super::name_hash::NameCache,
    ) -> Result<Self::Output> {
        tracing::debug!(
            source_id = %self.source_id,
            dest_id = %self.dest_id,
            name = %self.name,
            "Executing EdgeSummaryBySrcDstName query in transaction"
        );

        let ref_time = self
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        // Convert edge name to NameHash for key construction
        let name_hash = NameHash::from_name(&self.name);

        // Find current edge version via prefix scan
        let (_, value) = find_current_forward_edge_txn(txn, txn_db, self.source_id, self.dest_id, name_hash)?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Edge not found: {} -> {} ({})",
                    self.source_id,
                    self.dest_id,
                    self.name
                )
            })?;

        // Field indices (VERSIONING): 0=ValidUntil, 1=ActivePeriod, 2=Weight, 3=SummaryHash, 4=Version, 5=Deleted
        // Check if deleted
        if value.5 {
            return Err(anyhow::anyhow!(
                "Edge deleted: {} -> {} ({})",
                self.source_id,
                self.dest_id,
                self.name
            ));
        }

        // Check temporal validity (ActivePeriod at index 1)
        if !schema::is_active_at_time(&value.1, ref_time) {
            return Err(anyhow::anyhow!(
                "Edge {} -> {} ({}) not valid at time {}",
                self.source_id,
                self.dest_id,
                self.name,
                ref_time.0
            ));
        }

        // Resolve SummaryHash to EdgeSummary (from cold CF)
        let summary = resolve_edge_summary_from_txn(txn, txn_db, value.3)?;
        Ok((summary, value.2))
    }
}

/// Implement TransactionQueryExecutor for OutgoingEdges
/// (claude, 2026-02-06, in-progress: VERSIONING prefix scan and new field indices)
impl TransactionQueryExecutor for OutgoingEdges {
    type Output = Vec<(Option<f64>, SrcId, DstId, EdgeName)>;

    fn execute_in_transaction(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        cache: &super::name_hash::NameCache,
    ) -> Result<Self::Output> {
        tracing::debug!(id = %self.id, "Executing OutgoingEdges query in transaction");

        let ref_time = self
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let cf = txn_db
            .cf_handle(schema::ForwardEdges::CF_NAME)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Column family '{}' not found",
                    schema::ForwardEdges::CF_NAME
                )
            })?;

        // Use source ID as prefix for iteration
        let prefix = self.id.into_bytes();
        // Use HashSet for deduplication with versioned keys
        let mut seen_edges: std::collections::HashSet<(Id, Id, NameHash)> = std::collections::HashSet::new();
        let mut edges_with_hash: Vec<(Option<f64>, SrcId, DstId, NameHash)> = Vec::new();

        // NOTE: We use iterator_cf instead of prefix_iterator_cf because the CF has a
        // 40-byte prefix extractor but we're searching with a 16-byte prefix (src_id only).
        // Using prefix_iterator with mismatched prefix length causes bloom filter issues.
        let iter = txn.iterator_cf(cf, rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward));

        for item in iter {
            let (key_bytes, value_bytes) = item?;
            // Stop if we've gone past our prefix (src_id = 16 bytes)
            if !key_bytes.starts_with(&prefix) {
                break;
            }

            let key: schema::ForwardEdgeCfKey = schema::ForwardEdges::key_from_bytes(&key_bytes)?;

            let value: schema::ForwardEdgeCfValue =
                schema::ForwardEdges::value_from_bytes(&value_bytes)?;

            // Field indices (VERSIONING): 0=ValidUntil, 1=ActivePeriod, 2=Weight, 3=SummaryHash, 4=Version, 5=Deleted
            // Skip non-current versions
            if value.0.is_some() {
                continue;
            }
            // Skip deleted edges
            if value.5 {
                continue;
            }
            // Check temporal validity (ActivePeriod at index 1)
            if !schema::is_active_at_time(&value.1, ref_time) {
                continue;
            }

            let source_id = key.0;
            let dest_id = key.1;
            let edge_name_hash = key.2;
            let edge_topology = (source_id, dest_id, edge_name_hash);
            if seen_edges.insert(edge_topology) {
                let weight = value.2;
                edges_with_hash.push((weight, source_id, dest_id, edge_name_hash));
            }
        }

        // Resolve NameHashes to Strings (uses cache)
        let mut results = Vec::with_capacity(edges_with_hash.len());
        for (weight, src_id, dst_id, name_hash) in edges_with_hash {
            let edge_name = resolve_name_from_txn(txn, txn_db, name_hash, cache)?;
            results.push((weight, src_id, dst_id, edge_name));
        }

        Ok(results)
    }
}

/// Implement TransactionQueryExecutor for IncomingEdges
/// (claude, 2026-02-06, in-progress: VERSIONING prefix scan and new field indices)
impl TransactionQueryExecutor for IncomingEdges {
    type Output = Vec<(Option<f64>, DstId, SrcId, EdgeName)>;

    fn execute_in_transaction(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        cache: &super::name_hash::NameCache,
    ) -> Result<Self::Output> {
        tracing::debug!(id = %self.id, "Executing IncomingEdges query in transaction");

        let ref_time = self
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let reverse_cf = txn_db
            .cf_handle(schema::ReverseEdges::CF_NAME)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Column family '{}' not found",
                    schema::ReverseEdges::CF_NAME
                )
            })?;

        // Use destination ID as prefix for iteration
        let prefix = self.id.into_bytes();
        // Use HashSet for deduplication with versioned keys
        let mut seen_edges: std::collections::HashSet<(Id, Id, NameHash)> = std::collections::HashSet::new();
        let mut edges_with_hash: Vec<(Option<f64>, DstId, SrcId, NameHash)> = Vec::new();

        // NOTE: We use iterator_cf instead of prefix_iterator_cf because the CF has a
        // 40-byte prefix extractor but we're searching with a 16-byte prefix (dst_id only).
        // Using prefix_iterator with mismatched prefix length causes bloom filter issues.
        let iter = txn.iterator_cf(reverse_cf, rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward));

        for item in iter {
            let (key_bytes, value_bytes) = item?;
            // Stop if we've gone past our prefix (dst_id = 16 bytes)
            if !key_bytes.starts_with(&prefix) {
                break;
            }

            let key: schema::ReverseEdgeCfKey = schema::ReverseEdges::key_from_bytes(&key_bytes)?;

            let value: schema::ReverseEdgeCfValue =
                schema::ReverseEdges::value_from_bytes(&value_bytes)?;

            // Field indices (VERSIONING): 0=ValidUntil, 1=ActivePeriod
            // Skip non-current versions
            if value.0.is_some() {
                continue;
            }
            // Check temporal validity (ActivePeriod at index 1)
            if !schema::is_active_at_time(&value.1, ref_time) {
                continue;
            }

            let dest_id = key.0;
            let source_id = key.1;
            let edge_name_hash = key.2;

            // Deduplication check
            let edge_topology = (source_id, dest_id, edge_name_hash);
            if !seen_edges.insert(edge_topology) {
                continue;
            }

            // Lookup weight from ForwardEdges CF via prefix scan
            let weight = if let Some((_, forward_value)) =
                find_current_forward_edge_txn(txn, txn_db, source_id, dest_id, edge_name_hash)?
            {
                // Skip if deleted
                if forward_value.5 {
                    continue;
                }
                forward_value.2 // Extract weight from field 2
            } else {
                None
            };

            edges_with_hash.push((weight, dest_id, source_id, edge_name_hash));
        }

        // Resolve NameHashes to Strings (uses cache)
        let mut results = Vec::with_capacity(edges_with_hash.len());
        for (weight, dst_id, src_id, name_hash) in edges_with_hash {
            let edge_name = resolve_name_from_txn(txn, txn_db, name_hash, cache)?;
            results.push((weight, dst_id, src_id, edge_name));
        }

        Ok(results)
    }
}

/// Implement TransactionQueryExecutor for AllNodes
/// (claude, 2026-02-06, in-progress: VERSIONING iteration with deduplication)
impl TransactionQueryExecutor for AllNodes {
    type Output = Vec<(Id, NodeName, NodeSummary)>;

    fn execute_in_transaction(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        cache: &super::name_hash::NameCache,
    ) -> Result<Self::Output> {
        tracing::debug!(limit = self.limit, cursor = ?self.last, "Executing AllNodes query in transaction");

        let ref_time = self
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let cf = txn_db.cf_handle(schema::Nodes::CF_NAME).ok_or_else(|| {
            anyhow::anyhow!("Column family '{}' not found", schema::Nodes::CF_NAME)
        })?;

        let mut results = Vec::with_capacity(self.limit);

        // With VERSIONING, key is (Id, ValidSince). Cursor is just the Id.
        // We use Id prefix (16 bytes) for positioning, skipping past all versions
        // Field indices: 0=ValidUntil, 1=ActivePeriod, 2=NameHash, 3=SummaryHash, 4=Version, 5=Deleted
        let start_bytes: Option<Vec<u8>> = self.last.as_ref().map(|cursor| {
            // Create a prefix that starts after all versions of the cursor ID
            // by appending max timestamp
            let mut bytes = cursor.into_bytes().to_vec();
            bytes.extend_from_slice(&u64::MAX.to_be_bytes());
            bytes
        });

        let iter = if let Some(ref bytes) = start_bytes {
            txn.iterator_cf(cf, rocksdb::IteratorMode::From(bytes, rocksdb::Direction::Forward))
        } else {
            txn.iterator_cf(cf, rocksdb::IteratorMode::Start)
        };

        let mut last_node_id: Option<Id> = None;

        for item in iter {
            let (key_bytes, value_bytes) = item?;

            if results.len() >= self.limit {
                break;
            }

            let key: schema::NodeCfKey = schema::Nodes::key_from_bytes(&key_bytes)?;
            let node_id = key.0;

            // Skip duplicate node_ids (we already saw this node's current version)
            if last_node_id == Some(node_id) {
                continue;
            }

            let value: schema::NodeCfValue = schema::Nodes::value_from_bytes(&value_bytes)?;

            // Only process current versions (ValidUntil = None)
            if value.0.is_some() {
                continue;
            }

            // Skip deleted nodes
            if value.5 {
                last_node_id = Some(node_id);
                continue;
            }

            // Check temporal validity (ActivePeriod at index 1)
            if schema::is_active_at_time(&value.1, ref_time) {
                // Resolve NameHash to String (uses cache) - index 2
                let node_name = resolve_name_from_txn(txn, txn_db, value.2, cache)?;
                // Resolve SummaryHash to NodeSummary (from cold CF) - index 3
                let summary = resolve_node_summary_from_txn(txn, txn_db, value.3)?;
                results.push((node_id, node_name, summary));
            }

            last_node_id = Some(node_id);
        }

        Ok(results)
    }
}

/// Implement TransactionQueryExecutor for AllEdges
/// (claude, 2026-02-06, in-progress: VERSIONING iteration with deduplication)
impl TransactionQueryExecutor for AllEdges {
    type Output = Vec<(Option<f64>, SrcId, DstId, EdgeName)>;

    fn execute_in_transaction(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        cache: &super::name_hash::NameCache,
    ) -> Result<Self::Output> {
        tracing::debug!(limit = self.limit, cursor = ?self.last, "Executing AllEdges query in transaction");

        let ref_time = self
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let cf = txn_db
            .cf_handle(schema::ForwardEdges::CF_NAME)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Column family '{}' not found",
                    schema::ForwardEdges::CF_NAME
                )
            })?;

        // With VERSIONING, key is (SrcId, DstId, NameHash, ValidSince). Cursor is (SrcId, DstId, Name).
        // Create a prefix that starts after all versions of the cursor edge.
        let start_bytes: Option<Vec<u8>> = self.last.as_ref().map(|(src, dst, name)| {
            let name_hash = NameHash::from_name(name);
            // Append max timestamp to skip all versions of cursor edge
            let mut bytes = Vec::with_capacity(48);
            bytes.extend_from_slice(&src.into_bytes());
            bytes.extend_from_slice(&dst.into_bytes());
            bytes.extend_from_slice(name_hash.as_bytes());
            bytes.extend_from_slice(&u64::MAX.to_be_bytes());
            bytes
        });

        let iter = if let Some(ref bytes) = start_bytes {
            txn.iterator_cf(cf, rocksdb::IteratorMode::From(bytes, rocksdb::Direction::Forward))
        } else {
            txn.iterator_cf(cf, rocksdb::IteratorMode::Start)
        };

        // Use HashSet for deduplication with versioned keys
        let mut seen_edges: std::collections::HashSet<(Id, Id, NameHash)> = std::collections::HashSet::new();
        let mut edges_with_hash: Vec<(Option<f64>, SrcId, DstId, NameHash)> = Vec::new();

        for item in iter {
            let (key_bytes, value_bytes) = item?;

            if edges_with_hash.len() >= self.limit {
                break;
            }

            let key: schema::ForwardEdgeCfKey = schema::ForwardEdges::key_from_bytes(&key_bytes)?;
            let value: schema::ForwardEdgeCfValue =
                schema::ForwardEdges::value_from_bytes(&value_bytes)?;

            // Field indices (VERSIONING): 0=ValidUntil, 1=ActivePeriod, 2=Weight, 3=SummaryHash, 4=Version, 5=Deleted
            // Skip non-current versions
            if value.0.is_some() {
                continue;
            }
            // Skip deleted edges
            if value.5 {
                continue;
            }
            // Check temporal validity (ActivePeriod at index 1)
            if !schema::is_active_at_time(&value.1, ref_time) {
                continue;
            }

            let source_id = key.0;
            let dest_id = key.1;
            let edge_name_hash = key.2;
            let edge_topology = (source_id, dest_id, edge_name_hash);
            if seen_edges.insert(edge_topology) {
                let weight = value.2;
                edges_with_hash.push((weight, source_id, dest_id, edge_name_hash));
            }
        }

        // Resolve NameHashes to Strings (uses cache)
        let mut results = Vec::with_capacity(edges_with_hash.len());
        for (weight, src_id, dst_id, name_hash) in edges_with_hash {
            let edge_name = resolve_name_from_txn(txn, txn_db, name_hash, cache)?;
            results.push((weight, src_id, dst_id, edge_name));
        }

        Ok(results)
    }
}

// ============================================================================
// Reverse Lookup Query Types (CONTENT-ADDRESS)
// ============================================================================
// (claude, 2026-02-02, in-progress) CONTENT-ADDRESS reverse lookup APIs

/// Query parameters for finding nodes by summary hash.
///
/// Returns all (node_id, version, is_current) tuples that have the given summary hash.
/// Use `current_only: true` to filter to only current versions.
#[derive(Debug, Clone, PartialEq)]
pub struct NodesBySummaryHash {
    /// The summary hash to search for
    pub hash: SummaryHash,

    /// If true, only return current versions (marker = CURRENT)
    pub current_only: bool,
}

/// Result type for reverse node lookup
#[derive(Debug, Clone, PartialEq)]
pub struct NodeSummaryLookupResult {
    /// The node ID
    pub node_id: Id,
    /// The version at which this hash was set
    pub version: schema::Version,
    /// Whether this is the current version (marker = CURRENT)
    pub is_current: bool,
}

impl NodesBySummaryHash {
    /// Create a new query for all nodes (any version) with the given hash.
    pub fn all(hash: SummaryHash) -> Self {
        Self {
            hash,
            current_only: false,
        }
    }

    /// Create a new query for only current nodes with the given hash.
    pub fn current(hash: SummaryHash) -> Self {
        Self {
            hash,
            current_only: true,
        }
    }

    /// Execute this query directly on storage (for testing and simple use cases).
    pub async fn execute_on(&self, storage: &Storage) -> Result<Vec<NodeSummaryLookupResult>> {
        self.execute(storage).await
    }
}

/// Query parameters for finding edges by summary hash.
///
/// Returns all (edge_key, version, is_current) tuples that have the given summary hash.
/// Use `current_only: true` to filter to only current versions.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgesBySummaryHash {
    /// The summary hash to search for
    pub hash: SummaryHash,

    /// If true, only return current versions (marker = CURRENT)
    pub current_only: bool,
}

/// Result type for reverse edge lookup
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeSummaryLookupResult {
    /// Source node ID
    pub src_id: SrcId,
    /// Destination node ID
    pub dst_id: DstId,
    /// Edge name hash (use resolve_name to get full name)
    pub name_hash: NameHash,
    /// The version at which this hash was set
    pub version: schema::Version,
    /// Whether this is the current version (marker = CURRENT)
    pub is_current: bool,
}

impl EdgesBySummaryHash {
    /// Create a new query for all edges (any version) with the given hash.
    pub fn all(hash: SummaryHash) -> Self {
        Self {
            hash,
            current_only: false,
        }
    }

    /// Create a new query for only current edges with the given hash.
    pub fn current(hash: SummaryHash) -> Self {
        Self {
            hash,
            current_only: true,
        }
    }
}

// ============================================================================
// QueryReply Implementations for Reverse Lookup Queries
// ============================================================================

impl QueryReply for NodesBySummaryHash {
    type Reply = Vec<NodeSummaryLookupResult>;

    fn into_query(self) -> Query {
        Query::NodesBySummaryHash(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::NodesBySummaryHash(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for EdgesBySummaryHash {
    type Reply = Vec<EdgeSummaryLookupResult>;

    fn into_query(self) -> Query {
        Query::EdgesBySummaryHash(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::EdgesBySummaryHash(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

// ============================================================================
// QueryExecutor Implementations for Reverse Lookup Queries
// ============================================================================

#[async_trait::async_trait]
impl QueryExecutor for NodesBySummaryHash {
    type Output = Vec<NodeSummaryLookupResult>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        use schema::NodeSummaryIndex;

        let params = self;
        tracing::debug!(hash = ?params.hash, current_only = params.current_only, "Executing NodesBySummaryHash query");

        let txn_db = storage.transaction_db()?;
        let cf = txn_db
            .cf_handle(NodeSummaryIndex::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("NodeSummaryIndex CF not found"))?;

        // Use prefix scan with just the hash (8 bytes)
        let prefix = params.hash.as_bytes();
        let iter = txn_db.prefix_iterator_cf(cf, prefix);

        let mut results = Vec::new();
        for item in iter {
            let (key_bytes, value_bytes) = item?;

            // Check if key still starts with our prefix (prefix iterator may overrun)
            if key_bytes.len() < 8 || &key_bytes[0..8] != prefix {
                break;
            }

            let index_key = NodeSummaryIndex::key_from_bytes(&key_bytes)?;
            let index_value = NodeSummaryIndex::value_from_bytes(&value_bytes)?;

            let is_current = index_value.is_current();

            // Filter by current_only if requested
            if params.current_only && !is_current {
                continue;
            }

            results.push(NodeSummaryLookupResult {
                node_id: index_key.1,
                version: index_key.2,
                is_current,
            });
        }

        Ok(results)
    }
}

#[async_trait::async_trait]
impl QueryExecutor for EdgesBySummaryHash {
    type Output = Vec<EdgeSummaryLookupResult>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        use schema::EdgeSummaryIndex;

        let params = self;
        tracing::debug!(hash = ?params.hash, current_only = params.current_only, "Executing EdgesBySummaryHash query");

        let txn_db = storage.transaction_db()?;
        let cf = txn_db
            .cf_handle(EdgeSummaryIndex::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("EdgeSummaryIndex CF not found"))?;

        // Use prefix scan with just the hash (8 bytes)
        let prefix = params.hash.as_bytes();
        let iter = txn_db.prefix_iterator_cf(cf, prefix);

        let mut results = Vec::new();
        for item in iter {
            let (key_bytes, value_bytes) = item?;

            // Check if key still starts with our prefix (prefix iterator may overrun)
            if key_bytes.len() < 8 || &key_bytes[0..8] != prefix {
                break;
            }

            let index_key = EdgeSummaryIndex::key_from_bytes(&key_bytes)?;
            let index_value = EdgeSummaryIndex::value_from_bytes(&value_bytes)?;

            let is_current = index_value.is_current();

            // Filter by current_only if requested
            if params.current_only && !is_current {
                continue;
            }

            results.push(EdgeSummaryLookupResult {
                src_id: index_key.1,
                dst_id: index_key.2,
                name_hash: index_key.3,
                version: index_key.4,
                is_current,
            });
        }

        Ok(results)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::super::mutation::AddNode;
    use crate::writer::Runnable as MutRunnable;
    use super::super::reader::{
        create_query_reader, spawn_consumer, Consumer, Reader, ReaderConfig,
    };
    use super::super::writer::{create_mutation_writer, spawn_mutation_consumer, WriterConfig};
    use super::super::{Processor, Storage};
    use super::*;
    use crate::{DataUrl, Id, TimestampMilli};
    use std::sync::Arc;
    use tempfile::TempDir;
    use tokio::time::Duration;

    #[tokio::test]
    async fn test_consumer_basic() {
        // Create a temporary database with real data
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Create writer and mutation consumer
        let writer_config = WriterConfig {
            channel_buffer_size: 10,
        };
        let (writer, mutation_receiver) = create_mutation_writer(writer_config.clone());
        let mutation_consumer_handle =
            spawn_mutation_consumer(mutation_receiver, writer_config, &db_path);

        // Insert a test node
        let node_id = Id::new();
        let node_name = "test_node".to_string();
        let node_args = AddNode {
            id: node_id,
            ts_millis: TimestampMilli::now(),
            name: node_name.clone(),
            valid_range: None,
            summary: NodeSummary::from_text("test summary"),
        };
        node_args.run(&writer).await.unwrap();

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Drop writer to close mutation channel
        drop(writer);

        // Wait for mutation consumer to finish
        mutation_consumer_handle.await.unwrap().unwrap();

        // Now open storage for reading
        let mut storage = Storage::readonly(&db_path);
        storage.ready().unwrap();
        let processor = Processor::new(Arc::new(storage));

        // Create reader and query consumer
        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };

        let (reader, receiver) = create_query_reader(reader_config.clone());

        let consumer = Consumer::new(receiver, reader_config, processor);

        // Spawn query consumer
        let consumer_handle = spawn_consumer(consumer);

        // Query the node we just created
        let (returned_name, returned_summary) = NodeById::new(node_id, None)
            .run(&reader, Duration::from_secs(5))
            .await
            .unwrap();

        // Verify the data matches what we inserted
        assert_eq!(returned_name, node_name);
        assert!(returned_summary.decode_string().is_ok());

        // Drop reader to close channel
        drop(reader);

        // Wait for consumer to finish
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_edge_summary_by_src_dst_name_query() {
        use super::super::mutation::{AddEdge, AddNode};
        use super::super::schema::EdgeSummary;
        use crate::writer::Runnable as MutationRunnable;
        use super::super::writer::{create_mutation_writer, spawn_mutation_consumer, WriterConfig};
        use crate::{Id, TimestampMilli};
        use tempfile::TempDir;

        // Create temporary database
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let config = WriterConfig {
            channel_buffer_size: 10,
        };

        // Create writer and spawn graph consumer for mutations
        let (writer, mutation_receiver) = create_mutation_writer(config.clone());
        let mutation_consumer_handle = spawn_mutation_consumer(mutation_receiver, config, &db_path);

        // Create source and destination nodes
        let source_id = Id::new();
        let dest_id = Id::new();

        AddNode {
            id: source_id,
            ts_millis: TimestampMilli::now(),
            name: "source_node".to_string(),
            valid_range: None,
            summary: NodeSummary::from_text("source summary"),
        }
        .run(&writer)
        .await
        .unwrap();

        AddNode {
            id: dest_id,
            ts_millis: TimestampMilli::now(),
            name: "dest_node".to_string(),
            valid_range: None,
            summary: NodeSummary::from_text("dest summary"),
        }
        .run(&writer)
        .await
        .unwrap();

        // Create an edge
        let edge_name = "test_edge";
        let edge_summary = EdgeSummary::from_text("This is a test edge");
        let edge_weight = 2.5;

        AddEdge {
            source_node_id: source_id,
            target_node_id: dest_id,
            ts_millis: TimestampMilli::now(),
            name: edge_name.to_string(),
            summary: edge_summary.clone(),
            weight: Some(edge_weight),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        // Give time for mutations to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Close writer to ensure all mutations are flushed
        drop(writer);

        // Wait for mutation consumer to finish
        mutation_consumer_handle.await.unwrap().unwrap();

        // Now open storage for reading
        let mut storage = Storage::readonly(&db_path);
        storage.ready().unwrap();
        let processor = Processor::new(Arc::new(storage));

        // Create reader and query consumer
        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };

        let (reader, receiver) = {
            let (sender, receiver) = flume::bounded(reader_config.channel_buffer_size);
            let reader = Reader::new(sender);
            (reader, receiver)
        };

        let consumer = Consumer::new(receiver, reader_config, processor);

        // Spawn query consumer
        let consumer_handle = spawn_consumer(consumer);

        // Query the edge using EdgeSummaryBySrcDstName with Runnable pattern
        let (returned_summary, returned_weight) =
            EdgeSummaryBySrcDstName::new(source_id, dest_id, edge_name.to_string(), None)
                .run(&reader, Duration::from_secs(5))
                .await
                .unwrap();

        // Verify edge summary content matches
        assert_eq!(returned_summary.0, edge_summary.0);

        // Verify weight matches
        assert_eq!(returned_weight, Some(edge_weight));

        // Cleanup
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_edge_fragments_by_id_time_range_basic() {
        use super::super::mutation::{
            AddEdge, AddEdgeFragment, AddNode,
        };
        use crate::writer::Runnable as MutationRunnable;
        use super::super::schema::EdgeSummary;
        use super::super::writer::{create_mutation_writer, spawn_mutation_consumer, WriterConfig};
        use crate::{Id, TimestampMilli};
        use std::ops::Bound;
        use tempfile::TempDir;

        // Create temporary database
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let config = WriterConfig {
            channel_buffer_size: 10,
        };

        // Create writer and spawn graph consumer for mutations
        let (writer, mutation_receiver) = create_mutation_writer(config.clone());
        let mutation_consumer_handle = spawn_mutation_consumer(mutation_receiver, config, &db_path);

        // Create source and destination nodes
        let source_id = Id::new();
        let dest_id = Id::new();
        let edge_name = "test_edge";

        AddNode {
            id: source_id,
            ts_millis: TimestampMilli::now(),
            name: "source_node".to_string(),
            valid_range: None,
            summary: NodeSummary::from_text("source summary"),
        }
        .run(&writer)
        .await
        .unwrap();

        AddNode {
            id: dest_id,
            ts_millis: TimestampMilli::now(),
            name: "dest_node".to_string(),
            valid_range: None,
            summary: NodeSummary::from_text("dest summary"),
        }
        .run(&writer)
        .await
        .unwrap();

        // Create an edge
        let edge_summary = EdgeSummary::from_text("Test edge for fragments");
        AddEdge {
            source_node_id: source_id,
            target_node_id: dest_id,
            ts_millis: TimestampMilli::now(),
            name: edge_name.to_string(),
            summary: edge_summary.clone(),
            weight: Some(1.0),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        // Add edge fragments at different timestamps
        let base_time = TimestampMilli::now();
        let fragment1_time = TimestampMilli(base_time.0 + 1000);
        let fragment2_time = TimestampMilli(base_time.0 + 2000);
        let fragment3_time = TimestampMilli(base_time.0 + 3000);

        AddEdgeFragment {
            src_id: source_id,
            dst_id: dest_id,
            edge_name: edge_name.to_string(),
            ts_millis: fragment1_time,
            content: DataUrl::from_markdown("Fragment 1 content"),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        AddEdgeFragment {
            src_id: source_id,
            dst_id: dest_id,
            edge_name: edge_name.to_string(),
            ts_millis: fragment2_time,
            content: DataUrl::from_markdown("Fragment 2 content"),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        AddEdgeFragment {
            src_id: source_id,
            dst_id: dest_id,
            edge_name: edge_name.to_string(),
            ts_millis: fragment3_time,
            content: DataUrl::from_markdown("Fragment 3 content"),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        // Give time for mutations to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Close writer to ensure all mutations are flushed
        drop(writer);

        // Wait for mutation consumer to finish
        mutation_consumer_handle.await.unwrap().unwrap();

        // Now open storage for reading
        let mut storage = Storage::readonly(&db_path);
        storage.ready().unwrap();
        let processor = Processor::new(Arc::new(storage));

        // Create reader and query consumer
        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };

        let (reader, receiver) = {
            let (sender, receiver) = flume::bounded(reader_config.channel_buffer_size);
            let reader = Reader::new(sender);
            (reader, receiver)
        };

        let consumer = Consumer::new(receiver, reader_config, processor);

        // Spawn query consumer
        let consumer_handle = spawn_consumer(consumer);

        // Query all fragments (unbounded range)
        let fragments = EdgeFragmentsByIdTimeRange::new(
            source_id,
            dest_id,
            edge_name.to_string(),
            (Bound::Unbounded, Bound::Unbounded),
            None,
        )
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

        // Verify we got all 3 fragments
        assert_eq!(fragments.len(), 3);
        assert_eq!(fragments[0].0, fragment1_time);
        assert_eq!(fragments[1].0, fragment2_time);
        assert_eq!(fragments[2].0, fragment3_time);

        // Verify content
        assert!(fragments[0]
            .1
            .decode_string()
            .unwrap()
            .contains("Fragment 1 content"));
        assert!(fragments[1]
            .1
            .decode_string()
            .unwrap()
            .contains("Fragment 2 content"));
        assert!(fragments[2]
            .1
            .decode_string()
            .unwrap()
            .contains("Fragment 3 content"));

        // Cleanup
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_edge_fragments_by_id_time_range_with_bounds() {
        use super::super::mutation::{
            AddEdge, AddEdgeFragment, AddNode,
        };
        use crate::writer::Runnable as MutationRunnable;
        use super::super::schema::EdgeSummary;
        use super::super::writer::{create_mutation_writer, spawn_mutation_consumer, WriterConfig};
        use crate::{Id, TimestampMilli};
        use std::ops::Bound;
        use tempfile::TempDir;

        // Create temporary database
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let config = WriterConfig {
            channel_buffer_size: 10,
        };

        // Create writer and spawn graph consumer for mutations
        let (writer, mutation_receiver) = create_mutation_writer(config.clone());
        let mutation_consumer_handle = spawn_mutation_consumer(mutation_receiver, config, &db_path);

        // Create source and destination nodes
        let source_id = Id::new();
        let dest_id = Id::new();
        let edge_name = "bounded_edge";

        AddNode {
            id: source_id,
            ts_millis: TimestampMilli::now(),
            name: "source_node".to_string(),
            valid_range: None,
            summary: NodeSummary::from_text("source summary"),
        }
        .run(&writer)
        .await
        .unwrap();

        AddNode {
            id: dest_id,
            ts_millis: TimestampMilli::now(),
            name: "dest_node".to_string(),
            valid_range: None,
            summary: NodeSummary::from_text("dest summary"),
        }
        .run(&writer)
        .await
        .unwrap();

        // Create an edge
        AddEdge {
            source_node_id: source_id,
            target_node_id: dest_id,
            ts_millis: TimestampMilli::now(),
            name: edge_name.to_string(),
            summary: EdgeSummary::from_text("Bounded test edge"),
            weight: Some(1.0),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        // Add edge fragments at specific timestamps
        let t1 = TimestampMilli(1000);
        let t2 = TimestampMilli(2000);
        let t3 = TimestampMilli(3000);
        let t4 = TimestampMilli(4000);
        let t5 = TimestampMilli(5000);

        for (ts, content) in [
            (t1, "Fragment at t1"),
            (t2, "Fragment at t2"),
            (t3, "Fragment at t3"),
            (t4, "Fragment at t4"),
            (t5, "Fragment at t5"),
        ] {
            AddEdgeFragment {
                src_id: source_id,
                dst_id: dest_id,
                edge_name: edge_name.to_string(),
                ts_millis: ts,
                content: DataUrl::from_markdown(content),
                valid_range: None,
            }
            .run(&writer)
            .await
            .unwrap();
        }

        // Give time for mutations to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Close writer
        drop(writer);
        mutation_consumer_handle.await.unwrap().unwrap();

        // Now open storage for reading
        let mut storage = Storage::readonly(&db_path);
        storage.ready().unwrap();
        let processor = Processor::new(Arc::new(storage));

        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };

        let (reader, receiver) = {
            let (sender, receiver) = flume::bounded(reader_config.channel_buffer_size);
            let reader = Reader::new(sender);
            (reader, receiver)
        };

        let consumer = Consumer::new(receiver, reader_config, processor);
        let consumer_handle = spawn_consumer(consumer);

        // Test 1: Query with inclusive bounds [t2, t4]
        let fragments = EdgeFragmentsByIdTimeRange::new(
            source_id,
            dest_id,
            edge_name.to_string(),
            (Bound::Included(t2), Bound::Included(t4)),
            None,
        )
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

        assert_eq!(fragments.len(), 3); // t2, t3, t4
        assert_eq!(fragments[0].0, t2);
        assert_eq!(fragments[1].0, t3);
        assert_eq!(fragments[2].0, t4);

        // Test 2: Query with exclusive bounds (t2, t4)
        let fragments = EdgeFragmentsByIdTimeRange::new(
            source_id,
            dest_id,
            edge_name.to_string(),
            (Bound::Excluded(t2), Bound::Excluded(t4)),
            None,
        )
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

        assert_eq!(fragments.len(), 1); // Only t3
        assert_eq!(fragments[0].0, t3);

        // Test 3: Query with unbounded start
        let fragments = EdgeFragmentsByIdTimeRange::new(
            source_id,
            dest_id,
            edge_name.to_string(),
            (Bound::Unbounded, Bound::Included(t2)),
            None,
        )
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

        assert_eq!(fragments.len(), 2); // t1, t2
        assert_eq!(fragments[0].0, t1);
        assert_eq!(fragments[1].0, t2);

        // Test 4: Query with unbounded end
        let fragments = EdgeFragmentsByIdTimeRange::new(
            source_id,
            dest_id,
            edge_name.to_string(),
            (Bound::Included(t4), Bound::Unbounded),
            None,
        )
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

        assert_eq!(fragments.len(), 2); // t4, t5
        assert_eq!(fragments[0].0, t4);
        assert_eq!(fragments[1].0, t5);

        // Cleanup
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_edge_fragments_by_id_time_range_with_temporal_validity() {
        use super::super::mutation::{
            AddEdge, AddEdgeFragment, AddNode,
        };
        use crate::writer::Runnable as MutationRunnable;
        use super::super::schema::EdgeSummary;
        use super::super::writer::{create_mutation_writer, spawn_mutation_consumer, WriterConfig};
        use crate::{Id, ActivePeriod, TimestampMilli};
        use std::ops::Bound;
        use tempfile::TempDir;

        // Create temporary database
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let config = WriterConfig {
            channel_buffer_size: 10,
        };

        let (writer, mutation_receiver) = create_mutation_writer(config.clone());
        let mutation_consumer_handle = spawn_mutation_consumer(mutation_receiver, config, &db_path);

        // Create source and destination nodes
        let source_id = Id::new();
        let dest_id = Id::new();
        let edge_name = "temporal_edge";

        AddNode {
            id: source_id,
            ts_millis: TimestampMilli::now(),
            name: "source_node".to_string(),
            valid_range: None,
            summary: NodeSummary::from_text("source summary"),
        }
        .run(&writer)
        .await
        .unwrap();

        AddNode {
            id: dest_id,
            ts_millis: TimestampMilli::now(),
            name: "dest_node".to_string(),
            valid_range: None,
            summary: NodeSummary::from_text("dest summary"),
        }
        .run(&writer)
        .await
        .unwrap();

        // Create an edge
        AddEdge {
            source_node_id: source_id,
            target_node_id: dest_id,
            ts_millis: TimestampMilli::now(),
            name: edge_name.to_string(),
            summary: EdgeSummary::from_text("Temporal validity test edge"),
            weight: Some(1.0),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        // Add edge fragments with different temporal ranges
        // Fragment 1: Valid from 1000 to 3000
        AddEdgeFragment {
            src_id: source_id,
            dst_id: dest_id,
            edge_name: edge_name.to_string(),
            ts_millis: TimestampMilli(1000),
            content: DataUrl::from_markdown("Valid from 1000 to 3000"),
            valid_range: ActivePeriod::active_between(TimestampMilli(1000), TimestampMilli(3000)),
        }
        .run(&writer)
        .await
        .unwrap();

        // Fragment 2: Valid from 2000 to 5000
        AddEdgeFragment {
            src_id: source_id,
            dst_id: dest_id,
            edge_name: edge_name.to_string(),
            ts_millis: TimestampMilli(2000),
            content: DataUrl::from_markdown("Valid from 2000 to 5000"),
            valid_range: ActivePeriod::active_between(TimestampMilli(2000), TimestampMilli(5000)),
        }
        .run(&writer)
        .await
        .unwrap();

        // Fragment 3: No temporal range (always valid)
        AddEdgeFragment {
            src_id: source_id,
            dst_id: dest_id,
            edge_name: edge_name.to_string(),
            ts_millis: TimestampMilli(3000),
            content: DataUrl::from_markdown("Always valid"),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;
        drop(writer);
        mutation_consumer_handle.await.unwrap().unwrap();

        // Now open storage for reading
        let mut storage = Storage::readonly(&db_path);
        storage.ready().unwrap();
        let processor = Processor::new(Arc::new(storage));

        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };

        let (reader, receiver) = {
            let (sender, receiver) = flume::bounded(reader_config.channel_buffer_size);
            let reader = Reader::new(sender);
            (reader, receiver)
        };

        let consumer = Consumer::new(receiver, reader_config, processor);
        let consumer_handle = spawn_consumer(consumer);

        // Query at reference time 2500 - should see fragments 1, 2, and 3
        let fragments = EdgeFragmentsByIdTimeRange::new(
            source_id,
            dest_id,
            edge_name.to_string(),
            (Bound::Unbounded, Bound::Unbounded),
            Some(TimestampMilli(2500)),
        )
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

        assert_eq!(fragments.len(), 3);

        // Query at reference time 500 - should see only fragment 3 (always valid)
        let fragments = EdgeFragmentsByIdTimeRange::new(
            source_id,
            dest_id,
            edge_name.to_string(),
            (Bound::Unbounded, Bound::Unbounded),
            Some(TimestampMilli(500)),
        )
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0].0, TimestampMilli(3000));
        assert!(fragments[0]
            .1
            .decode_string()
            .unwrap()
            .contains("Always valid"));

        // Query at reference time 3500 - should see fragments 2 and 3
        let fragments = EdgeFragmentsByIdTimeRange::new(
            source_id,
            dest_id,
            edge_name.to_string(),
            (Bound::Unbounded, Bound::Unbounded),
            Some(TimestampMilli(3500)),
        )
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

        assert_eq!(fragments.len(), 2);
        assert_eq!(fragments[0].0, TimestampMilli(2000));
        assert_eq!(fragments[1].0, TimestampMilli(3000));

        // Cleanup
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_edge_fragments_empty_result() {
        use super::super::mutation::{AddEdge, AddNode};
        use super::super::schema::{EdgeSummary, NodeSummary};
        use crate::writer::Runnable as MutationRunnable;
        use super::super::writer::{create_mutation_writer, spawn_mutation_consumer, WriterConfig};
        use crate::{Id, TimestampMilli};
        use std::ops::Bound;
        use tempfile::TempDir;

        // Create temporary database
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let config = WriterConfig {
            channel_buffer_size: 10,
        };

        let (writer, mutation_receiver) = create_mutation_writer(config.clone());
        let mutation_consumer_handle = spawn_mutation_consumer(mutation_receiver, config, &db_path);

        // Create source and destination nodes and edge but NO fragments
        let source_id = Id::new();
        let dest_id = Id::new();
        let edge_name = "empty_edge";

        AddNode {
            id: source_id,
            ts_millis: TimestampMilli::now(),
            name: "source_node".to_string(),
            valid_range: None,
            summary: NodeSummary::from_text("source summary"),
        }
        .run(&writer)
        .await
        .unwrap();

        AddNode {
            id: dest_id,
            ts_millis: TimestampMilli::now(),
            name: "dest_node".to_string(),
            valid_range: None,
            summary: NodeSummary::from_text("dest summary"),
        }
        .run(&writer)
        .await
        .unwrap();

        AddEdge {
            source_node_id: source_id,
            target_node_id: dest_id,
            ts_millis: TimestampMilli::now(),
            name: edge_name.to_string(),
            summary: EdgeSummary::from_text("Edge with no fragments"),
            weight: Some(1.0),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;
        drop(writer);
        mutation_consumer_handle.await.unwrap().unwrap();

        // Now open storage for reading
        let mut storage = Storage::readonly(&db_path);
        storage.ready().unwrap();
        let processor = Processor::new(Arc::new(storage));

        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };

        let (reader, receiver) = {
            let (sender, receiver) = flume::bounded(reader_config.channel_buffer_size);
            let reader = Reader::new(sender);
            (reader, receiver)
        };

        let consumer = Consumer::new(receiver, reader_config, processor);
        let consumer_handle = spawn_consumer(consumer);

        // Query should return empty vector
        let fragments = EdgeFragmentsByIdTimeRange::new(
            source_id,
            dest_id,
            edge_name.to_string(),
            (Bound::Unbounded, Bound::Unbounded),
            None,
        )
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

        assert_eq!(fragments.len(), 0);

        // Cleanup
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_nodes_by_ids_multi_basic() {
        // Create a temporary database with real data
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Create writer and mutation consumer
        let writer_config = WriterConfig {
            channel_buffer_size: 10,
        };
        let (writer, mutation_receiver) = create_mutation_writer(writer_config.clone());
        let mutation_consumer_handle =
            spawn_mutation_consumer(mutation_receiver, writer_config, &db_path);

        // Insert multiple test nodes
        let node_ids: Vec<Id> = (0..5).map(|_| Id::new()).collect();
        for (i, &id) in node_ids.iter().enumerate() {
            let node_args = AddNode {
                id,
                ts_millis: TimestampMilli::now(),
                name: format!("node_{}", i),
                valid_range: None,
                summary: NodeSummary::from_text(&format!("summary for node {}", i)),
            };
            node_args.run(&writer).await.unwrap();
        }

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Drop writer to close mutation channel
        drop(writer);

        // Wait for mutation consumer to finish
        mutation_consumer_handle.await.unwrap().unwrap();

        // Now open storage for reading
        let mut storage = Storage::readonly(&db_path);
        storage.ready().unwrap();
        let processor = Processor::new(Arc::new(storage));

        // Create reader and query consumer
        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };

        let (reader, receiver) = create_query_reader(reader_config.clone());
        let consumer = Consumer::new(receiver, reader_config, processor);
        let consumer_handle = spawn_consumer(consumer);

        // Query all nodes with batch lookup
        let results = NodesByIdsMulti::new(node_ids.clone(), None)
            .run(&reader, Duration::from_secs(5))
            .await
            .unwrap();

        // Verify all nodes were found
        assert_eq!(results.len(), 5);

        // Verify each node is present
        for (id, name, _summary) in &results {
            assert!(node_ids.contains(id));
            assert!(name.starts_with("node_"));
        }

        // Cleanup
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_nodes_by_ids_multi_missing_nodes() {
        // Create a temporary database with real data
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Create writer and mutation consumer
        let writer_config = WriterConfig {
            channel_buffer_size: 10,
        };
        let (writer, mutation_receiver) = create_mutation_writer(writer_config.clone());
        let mutation_consumer_handle =
            spawn_mutation_consumer(mutation_receiver, writer_config, &db_path);

        // Insert only 2 test nodes
        let existing_ids: Vec<Id> = (0..2).map(|_| Id::new()).collect();
        for (i, &id) in existing_ids.iter().enumerate() {
            let node_args = AddNode {
                id,
                ts_millis: TimestampMilli::now(),
                name: format!("node_{}", i),
                valid_range: None,
                summary: NodeSummary::from_text(&format!("summary for node {}", i)),
            };
            node_args.run(&writer).await.unwrap();
        }

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Drop writer to close mutation channel
        drop(writer);

        // Wait for mutation consumer to finish
        mutation_consumer_handle.await.unwrap().unwrap();

        // Now open storage for reading
        let mut storage = Storage::readonly(&db_path);
        storage.ready().unwrap();
        let processor = Processor::new(Arc::new(storage));

        // Create reader and query consumer
        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };

        let (reader, receiver) = create_query_reader(reader_config.clone());
        let consumer = Consumer::new(receiver, reader_config, processor);
        let consumer_handle = spawn_consumer(consumer);

        // Query with a mix of existing and non-existing IDs
        let non_existing_ids: Vec<Id> = (0..3).map(|_| Id::new()).collect();
        let all_ids: Vec<Id> = existing_ids
            .iter()
            .chain(non_existing_ids.iter())
            .copied()
            .collect();

        let results = NodesByIdsMulti::new(all_ids, None)
            .run(&reader, Duration::from_secs(5))
            .await
            .unwrap();

        // Should only return the 2 existing nodes
        assert_eq!(results.len(), 2);

        // Verify only existing IDs are in results
        for (id, _name, _summary) in &results {
            assert!(existing_ids.contains(id));
        }

        // Cleanup
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_nodes_by_ids_multi_empty_input() {
        // Create a temporary database
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Create writer and mutation consumer
        let writer_config = WriterConfig {
            channel_buffer_size: 10,
        };
        let (writer, mutation_receiver) = create_mutation_writer(writer_config.clone());
        let mutation_consumer_handle =
            spawn_mutation_consumer(mutation_receiver, writer_config, &db_path);

        // Insert one node so DB is properly initialized
        let node_args = AddNode {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "dummy".to_string(),
            valid_range: None,
            summary: NodeSummary::from_text("dummy"),
        };
        node_args.run(&writer).await.unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;
        drop(writer);
        mutation_consumer_handle.await.unwrap().unwrap();

        // Now open storage for reading
        let mut storage = Storage::readonly(&db_path);
        storage.ready().unwrap();
        let processor = Processor::new(Arc::new(storage));

        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };

        let (reader, receiver) = create_query_reader(reader_config.clone());
        let consumer = Consumer::new(receiver, reader_config, processor);
        let consumer_handle = spawn_consumer(consumer);

        // Query with empty input
        let results = NodesByIdsMulti::new(vec![], None)
            .run(&reader, Duration::from_secs(5))
            .await
            .unwrap();

        // Should return empty vector
        assert_eq!(results.len(), 0);

        // Cleanup
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_all_nodes_query() {
        // Create a temporary database with real data
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Create writer and mutation consumer
        let writer_config = WriterConfig {
            channel_buffer_size: 10,
        };
        let (writer, mutation_receiver) = create_mutation_writer(writer_config.clone());
        let mutation_consumer_handle =
            spawn_mutation_consumer(mutation_receiver, writer_config, &db_path);

        // Insert multiple test nodes
        let node_ids: Vec<Id> = (0..5).map(|_| Id::new()).collect();
        for (i, &node_id) in node_ids.iter().enumerate() {
            let node_args = AddNode {
                id: node_id,
                ts_millis: TimestampMilli::now(),
                name: format!("node_{}", i),
                valid_range: None,
                summary: NodeSummary::from_text(&format!("summary for node {}", i)),
            };
            node_args.run(&writer).await.unwrap();
        }

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;
        drop(writer);
        mutation_consumer_handle.await.unwrap().unwrap();

        // Now open storage for reading
        let mut storage = Storage::readonly(&db_path);
        storage.ready().unwrap();
        let processor = Processor::new(Arc::new(storage));

        // Create reader and query consumer
        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };
        let (reader, receiver) = create_query_reader(reader_config.clone());
        let consumer = Consumer::new(receiver, reader_config, processor);
        let consumer_handle = spawn_consumer(consumer);

        // Query all nodes
        let results = AllNodes::new(100)
            .run(&reader, Duration::from_secs(5))
            .await
            .unwrap();

        // Should return all 5 nodes
        assert_eq!(results.len(), 5, "Should return all 5 nodes");

        // Verify each node has expected data
        for (id, name, summary) in &results {
            assert!(!id.is_nil(), "Node ID should not be nil");
            assert!(name.starts_with("node_"), "Node name should start with 'node_'");
            assert!(
                summary.decode_string().unwrap().contains("summary for node"),
                "Summary should contain expected text"
            );
        }

        // Cleanup
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_all_nodes_pagination() {
        // Create a temporary database with real data
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Create writer and mutation consumer
        let writer_config = WriterConfig {
            channel_buffer_size: 10,
        };
        let (writer, mutation_receiver) = create_mutation_writer(writer_config.clone());
        let mutation_consumer_handle =
            spawn_mutation_consumer(mutation_receiver, writer_config, &db_path);

        // Insert 10 test nodes
        for i in 0..10 {
            let node_args = AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                name: format!("node_{}", i),
                valid_range: None,
                summary: NodeSummary::from_text(&format!("summary {}", i)),
            };
            node_args.run(&writer).await.unwrap();
        }

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;
        drop(writer);
        mutation_consumer_handle.await.unwrap().unwrap();

        // Now open storage for reading
        let mut storage = Storage::readonly(&db_path);
        storage.ready().unwrap();
        let processor = Processor::new(Arc::new(storage));

        // Create reader and query consumer
        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };
        let (reader, receiver) = create_query_reader(reader_config.clone());
        let consumer = Consumer::new(receiver, reader_config, processor);
        let consumer_handle = spawn_consumer(consumer);

        // Get first page of 3 nodes
        let page1 = AllNodes::new(3)
            .run(&reader, Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(page1.len(), 3, "First page should have 3 nodes");

        // Get second page using cursor
        let last_id = page1.last().unwrap().0;
        let page2 = AllNodes::new(3)
            .with_cursor(last_id)
            .run(&reader, Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(page2.len(), 3, "Second page should have 3 nodes");

        // Ensure no overlap between pages
        let page1_ids: std::collections::HashSet<_> = page1.iter().map(|(id, _, _)| *id).collect();
        let page2_ids: std::collections::HashSet<_> = page2.iter().map(|(id, _, _)| *id).collect();
        assert!(
            page1_ids.is_disjoint(&page2_ids),
            "Pages should not overlap"
        );

        // Cleanup
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_all_edges_query() {
        use super::super::mutation::AddEdge;

        // Create a temporary database with real data
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Create writer and mutation consumer
        let writer_config = WriterConfig {
            channel_buffer_size: 10,
        };
        let (writer, mutation_receiver) = create_mutation_writer(writer_config.clone());
        let mutation_consumer_handle =
            spawn_mutation_consumer(mutation_receiver, writer_config, &db_path);

        // Insert nodes first
        let node1 = Id::new();
        let node2 = Id::new();
        let node3 = Id::new();

        for (id, name) in [(node1, "A"), (node2, "B"), (node3, "C")] {
            AddNode {
                id,
                ts_millis: TimestampMilli::now(),
                name: name.to_string(),
                valid_range: None,
                summary: NodeSummary::from_text(&format!("Node {}", name)),
            }
            .run(&writer)
            .await
            .unwrap();
        }

        // Insert edges
        AddEdge {
            source_node_id: node1,
            target_node_id: node2,
            ts_millis: TimestampMilli::now(),
            name: "connects".to_string(),
            summary: schema::EdgeSummary::from_text("A connects to B"),
            weight: Some(1.0),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        AddEdge {
            source_node_id: node2,
            target_node_id: node3,
            ts_millis: TimestampMilli::now(),
            name: "links".to_string(),
            summary: schema::EdgeSummary::from_text("B links to C"),
            weight: Some(0.5),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;
        drop(writer);
        mutation_consumer_handle.await.unwrap().unwrap();

        // Now open storage for reading
        let mut storage = Storage::readonly(&db_path);
        storage.ready().unwrap();
        let processor = Processor::new(Arc::new(storage));

        // Create reader and query consumer
        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };
        let (reader, receiver) = create_query_reader(reader_config.clone());
        let consumer = Consumer::new(receiver, reader_config, processor);
        let consumer_handle = spawn_consumer(consumer);

        // Query all edges
        let results = AllEdges::new(100)
            .run(&reader, Duration::from_secs(5))
            .await
            .unwrap();

        // Should return 2 edges
        assert_eq!(results.len(), 2, "Should return 2 edges");

        // Verify edge data
        let edge_names: Vec<_> = results.iter().map(|(_, _, _, name)| name.as_str()).collect();
        assert!(edge_names.contains(&"connects"), "Should contain 'connects' edge");
        assert!(edge_names.contains(&"links"), "Should contain 'links' edge");

        // Verify weights are present
        for (weight, _, _, _) in &results {
            assert!(weight.is_some(), "Edge weight should be present");
        }

        // Cleanup
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_all_nodes_empty_database() {
        // Create an empty database
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Create writer but don't insert any data
        let writer_config = WriterConfig {
            channel_buffer_size: 10,
        };
        let (writer, mutation_receiver) = create_mutation_writer(writer_config.clone());
        let mutation_consumer_handle =
            spawn_mutation_consumer(mutation_receiver, writer_config, &db_path);

        // Need to insert at least one item to initialize the DB properly
        // then we'll test a different scenario
        let node_args = AddNode {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "temp".to_string(),
            valid_range: None,
            summary: NodeSummary::from_text("temp"),
        };
        node_args.run(&writer).await.unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;
        drop(writer);
        mutation_consumer_handle.await.unwrap().unwrap();

        // Now open storage for reading
        let mut storage = Storage::readonly(&db_path);
        storage.ready().unwrap();
        let processor = Processor::new(Arc::new(storage));

        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };
        let (reader, receiver) = create_query_reader(reader_config.clone());
        let consumer = Consumer::new(receiver, reader_config, processor);
        let consumer_handle = spawn_consumer(consumer);

        // Query with limit 0 should return empty
        let results = AllNodes::new(0)
            .run(&reader, Duration::from_secs(5))
            .await
            .unwrap();

        assert_eq!(results.len(), 0, "Limit 0 should return empty results");

        // Cleanup
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }
}
