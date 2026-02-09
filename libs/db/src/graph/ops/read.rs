//! Read operations for graph entities.
//!
//! This module provides unified read operations that work with any storage access type
//! (readonly, readwrite, or transaction). These are the core building blocks used by
//! QueryExecutor and TransactionQueryExecutor implementations.

use anyhow::Result;

use crate::rocksdb::{ColumnFamily, HotColumnFamilyRecord};

use super::super::name_hash::NameHash;
use super::super::schema::{self, ForwardEdges, Nodes};
use crate::{Id, TimestampMilli};

// ============================================================================
// Storage Access Abstraction
// ============================================================================

/// Storage access abstraction for unified query logic.
///
/// This enum allows the same query implementation to work with:
/// - `Readonly`: Read-only RocksDB `DB` access
/// - `Readwrite`: Read-write `TransactionDB` access (outside transactions)
/// - `Transaction`: Within an active transaction (sees uncommitted writes)
#[derive(Clone)]
pub(crate) enum StorageAccess<'a> {
    Readonly(&'a rocksdb::DB),
    Readwrite(&'a rocksdb::TransactionDB),
    Transaction(
        &'a rocksdb::Transaction<'a, rocksdb::TransactionDB>,
        &'a rocksdb::TransactionDB,
    ),
}

impl<'a> StorageAccess<'a> {
    /// Get a column family handle.
    pub(crate) fn cf_handle(&self, cf_name: &str) -> Option<&rocksdb::ColumnFamily> {
        match self {
            StorageAccess::Readonly(db) => db.cf_handle(cf_name),
            StorageAccess::Readwrite(db) => db.cf_handle(cf_name),
            StorageAccess::Transaction(_, db) => db.cf_handle(cf_name),
        }
    }
}

// ============================================================================
// Node Version Lookup
// ============================================================================

/// Find a node version, optionally at a specific point in time.
///
/// This is the primary entry point for node lookups. It handles:
/// - Current version queries (`as_of = None`): Forward scan for `ValidUntil = None`
/// - Point-in-time queries (`as_of = Some(ts)`): Reverse seek for O(1) lookup
///
/// # Arguments
///
/// * `storage` - Storage access (readonly, readwrite, or transaction)
/// * `node_id` - The node to look up
/// * `as_of` - Optional timestamp for point-in-time query
///
/// # Returns
///
/// The key bytes and value of the matching node version, or None if not found.
pub(crate) fn find_node_version(
    storage: StorageAccess<'_>,
    node_id: Id,
    as_of: Option<TimestampMilli>,
) -> Result<Option<(Vec<u8>, schema::NodeCfValue)>> {
    let prefix = node_id.into_bytes().to_vec();

    match as_of {
        None => find_node_current_version_scan(&storage, &prefix),
        Some(as_of_ts) => find_node_version_at_time_reverse(&storage, &prefix, as_of_ts),
    }
}

/// Find current node version using forward scan (ValidUntil = None).
fn find_node_current_version_scan(
    storage: &StorageAccess<'_>,
    prefix: &[u8],
) -> Result<Option<(Vec<u8>, schema::NodeCfValue)>> {
    let nodes_cf = storage
        .cf_handle(Nodes::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Nodes CF not found"))?;

    match storage {
        StorageAccess::Readonly(db) => {
            for item in db.prefix_iterator_cf(nodes_cf, prefix) {
                let (key_bytes, value_bytes) = item?;
                if !key_bytes.starts_with(prefix) {
                    break;
                }
                let value: schema::NodeCfValue = Nodes::value_from_bytes(&value_bytes)
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
                let value: schema::NodeCfValue = Nodes::value_from_bytes(&value_bytes)
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
                let value: schema::NodeCfValue = Nodes::value_from_bytes(&value_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize node value: {}", e))?;
                if value.0.is_none() {
                    return Ok(Some((key_bytes.to_vec(), value)));
                }
            }
        }
    }
    Ok(None)
}

/// Find node version at specific time using reverse seek (O(1) lookup).
fn find_node_version_at_time_reverse(
    storage: &StorageAccess<'_>,
    prefix: &[u8],
    as_of_ts: TimestampMilli,
) -> Result<Option<(Vec<u8>, schema::NodeCfValue)>> {
    let nodes_cf = storage
        .cf_handle(Nodes::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Nodes CF not found"))?;

    // Build seek key: prefix + as_of_ts (24 bytes for nodes)
    let mut seek_key = Vec::with_capacity(24);
    seek_key.extend_from_slice(prefix);
    seek_key.extend_from_slice(&as_of_ts.0.to_be_bytes());

    match storage {
        StorageAccess::Readonly(db) => {
            let mut iter = db.raw_iterator_cf(nodes_cf);
            iter.seek_for_prev(&seek_key);

            while iter.valid() {
                let Some(key_bytes) = iter.key() else {
                    break;
                };
                if !key_bytes.starts_with(prefix) {
                    iter.prev();
                    continue;
                }

                let Some(value_bytes) = iter.value() else {
                    break;
                };
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

                let value: schema::NodeCfValue = Nodes::value_from_bytes(&value_bytes_aligned)
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
        StorageAccess::Readwrite(db) => {
            let mut iter = db.raw_iterator_cf(nodes_cf);
            iter.seek_for_prev(&seek_key);

            while iter.valid() {
                let Some(key_bytes) = iter.key() else {
                    break;
                };
                if !key_bytes.starts_with(prefix) {
                    iter.prev();
                    continue;
                }

                let Some(value_bytes) = iter.value() else {
                    break;
                };
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

                let value: schema::NodeCfValue = Nodes::value_from_bytes(&value_bytes_aligned)
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

                let value: schema::NodeCfValue = Nodes::value_from_bytes(&value_bytes)
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

// ============================================================================
// Edge Version Lookup
// ============================================================================

/// Find an edge version, optionally at a specific point in time.
///
/// This is the primary entry point for edge lookups. It handles:
/// - Current version queries (`as_of = None`): Forward scan for `ValidUntil = None`
/// - Point-in-time queries (`as_of = Some(ts)`): Reverse seek for O(1) lookup
///
/// # Arguments
///
/// * `storage` - Storage access (readonly, readwrite, or transaction)
/// * `src_id` - Source node ID
/// * `dst_id` - Destination node ID
/// * `name_hash` - Edge name hash
/// * `as_of` - Optional timestamp for point-in-time query
///
/// # Returns
///
/// The key bytes and value of the matching edge version, or None if not found.
pub(crate) fn find_edge_version(
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
    let edges_cf = storage
        .cf_handle(ForwardEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;

    match storage {
        StorageAccess::Readonly(db) => {
            for item in db.prefix_iterator_cf(edges_cf, prefix) {
                let (key_bytes, value_bytes) = item?;
                if !key_bytes.starts_with(prefix) {
                    break;
                }
                let value: schema::ForwardEdgeCfValue = ForwardEdges::value_from_bytes(&value_bytes)
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
                let value: schema::ForwardEdgeCfValue = ForwardEdges::value_from_bytes(&value_bytes)
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
                let value: schema::ForwardEdgeCfValue = ForwardEdges::value_from_bytes(&value_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize edge value: {}", e))?;
                if value.0.is_none() {
                    return Ok(Some((key_bytes.to_vec(), value)));
                }
            }
        }
    }
    Ok(None)
}

/// Find edge version at specific time using reverse seek (O(1) lookup).
fn find_edge_version_at_time_reverse(
    storage: &StorageAccess<'_>,
    prefix: &[u8],
    as_of_ts: TimestampMilli,
) -> Result<Option<(Vec<u8>, schema::ForwardEdgeCfValue)>> {
    let edges_cf = storage
        .cf_handle(ForwardEdges::CF_NAME)
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
                let Some(key_bytes) = iter.key() else {
                    break;
                };
                if !key_bytes.starts_with(prefix) {
                    iter.prev();
                    continue;
                }

                let Some(value_bytes) = iter.value() else {
                    break;
                };
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

                let value: schema::ForwardEdgeCfValue =
                    ForwardEdges::value_from_bytes(&value_bytes_aligned)
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
                let Some(key_bytes) = iter.key() else {
                    break;
                };
                if !key_bytes.starts_with(prefix) {
                    iter.prev();
                    continue;
                }

                let Some(value_bytes) = iter.value() else {
                    break;
                };
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

                let value: schema::ForwardEdgeCfValue =
                    ForwardEdges::value_from_bytes(&value_bytes_aligned)
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

                let value: schema::ForwardEdgeCfValue = ForwardEdges::value_from_bytes(&value_bytes)
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
