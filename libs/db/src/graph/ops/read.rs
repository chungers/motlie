//! Read operations for graph entities.
//!
//! This module provides unified read operations that work with any storage access type
//! (readonly, readwrite, or transaction). These are the core building blocks used by
//! QueryExecutor and TransactionQueryExecutor implementations.

use anyhow::Result;

use crate::graph::ColumnFamilySerde;
use crate::rocksdb::{ColumnFamily, HotColumnFamilyRecord};

use super::name::resolve_name;
use super::summary::{resolve_edge_summary, resolve_node_summary};
use super::super::name_hash::NameHash;
use super::super::query::{
    AllEdges, AllNodes, EdgeFragmentsByIdTimeRange, EdgeSummaryBySrcDstName, EdgesBySummaryHash,
    IncomingEdges, NodeById, NodeFragmentsByIdTimeRange, NodesByIdsMulti, NodesBySummaryHash,
    OutgoingEdges,
};
use super::super::scan;
use super::super::scan::Visitable;
use super::super::schema::{
    self, EdgeName, EdgeSummary, EdgeWeight, FragmentContent, ForwardEdges, Nodes, NodeName,
    NodeSummary, Version,
};
use super::super::summary_hash::SummaryHash;
use crate::{Id, SystemTimeMillis, TimestampMilli};

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

/// Macro to iterate over a column family in both readonly and readwrite storage modes.
macro_rules! iterate_cf {
    ($storage:expr, $cf_type:ty, $start_key:expr, |$item:ident| $process_body:block) => {{
        let start_key = $start_key.as_ref();
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
                rocksdb::IteratorMode::From(start_key, rocksdb::Direction::Forward),
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
                rocksdb::IteratorMode::From(start_key, rocksdb::Direction::Forward),
            );

            for $item in iter $process_body
        }
    }};
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
    as_of: Option<SystemTimeMillis>,
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
    as_of: Option<SystemTimeMillis>,
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

// ============================================================================
// Query Ops (Storage)
// ============================================================================

pub(crate) fn node_by_id(storage: &super::super::Storage, query: &NodeById) -> Result<(NodeName, NodeSummary, Version)> {
    let params = query;
    tracing::debug!(
        id = %params.id,
        as_of = ?params.as_of,
        "Executing NodeById query"
    );

    let ref_time = params.reference_ts_millis.unwrap_or_else(|| TimestampMilli::now());
    let id = params.id;

    let (_key_bytes, value) = if let Ok(db) = storage.db() {
        find_node_version(StorageAccess::Readonly(db), id, params.as_of)?
    } else {
        let txn_db = storage.transaction_db()?;
        find_node_version(StorageAccess::Readwrite(txn_db), id, params.as_of)?
    }
    .ok_or_else(|| {
        if params.as_of.is_some() {
            anyhow::anyhow!("Node {} not found at system time {:?}", id, params.as_of)
        } else {
            anyhow::anyhow!("Node not found: {}", id)
        }
    })?;

    if params.as_of.is_none() && value.5 {
        return Err(anyhow::anyhow!("Node {} has been deleted", id));
    }

    if !schema::is_active_at_time(&value.1, ref_time) {
        return Err(anyhow::anyhow!("Node {} not valid at time {}", id, ref_time.0));
    }

    let node_name = resolve_name(storage, value.2)?;
    let summary = resolve_node_summary(storage, value.3)?;

    Ok((node_name, summary, value.4))
}

pub(crate) fn nodes_by_ids_multi(
    storage: &super::super::Storage,
    query: &NodesByIdsMulti,
) -> Result<Vec<(Id, NodeName, NodeSummary, Version)>> {
    let params = query;
    tracing::debug!(
        count = params.ids.len(),
        as_of = ?params.as_of,
        "Executing NodesByIdsMulti query"
    );

    if params.ids.is_empty() {
        return Ok(Vec::new());
    }

    let ref_time = params.reference_ts_millis.unwrap_or_else(|| TimestampMilli::now());

    let mut valid_entries: Vec<(Id, NameHash, Option<SummaryHash>, Version)> =
        Vec::with_capacity(params.ids.len());

    for id in &params.ids {
        let result = if let Ok(db) = storage.db() {
            find_node_version(StorageAccess::Readonly(db), *id, params.as_of)
        } else {
            let txn_db = storage.transaction_db()?;
            find_node_version(StorageAccess::Readwrite(txn_db), *id, params.as_of)
        };

        match result {
            Ok(Some((_key_bytes, value))) => {
                if value.5 {
                    tracing::trace!(id = %id, "Skipping node: deleted");
                    continue;
                }
                if schema::is_active_at_time(&value.1, ref_time) {
                    valid_entries.push((*id, value.2, value.3, value.4));
                } else {
                    tracing::trace!(id = %id, "Skipping node: not valid at time {}", ref_time.0);
                }
            }
            Ok(None) => {
                tracing::trace!(id = %id, "Node not found");
            }
            Err(e) => {
                tracing::warn!(id = %id, error = %e, "Error fetching node");
            }
        }
    }

    let mut output = Vec::with_capacity(valid_entries.len());
    for (id, name_hash, summary_hash, version) in valid_entries {
        let node_name = resolve_name(storage, name_hash)?;
        let summary = resolve_node_summary(storage, summary_hash)?;
        output.push((id, node_name, summary, version));
    }

    Ok(output)
}

pub(crate) fn node_fragments_by_id_time_range(
    storage: &super::super::Storage,
    query: &NodeFragmentsByIdTimeRange,
) -> Result<Vec<(TimestampMilli, FragmentContent)>> {
    let params = query;
    tracing::debug!(
        id = %params.id,
        time_range = ?params.time_range,
        "Executing NodeFragmentsByIdTimeRange query"
    );

    use std::ops::Bound;

    let ref_time = params.reference_ts_millis.unwrap_or_else(|| TimestampMilli::now());

    let id = params.id;
    let mut fragments: Vec<(TimestampMilli, FragmentContent)> = Vec::new();

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
            Bound::Unbounded => {}
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

        if !schema::is_active_at_time(&value.0, ref_time) {
            continue;
        }

        fragments.push((timestamp, value.1));
    });

    Ok(fragments)
}

pub(crate) fn edge_fragments_by_id_time_range(
    storage: &super::super::Storage,
    query: &EdgeFragmentsByIdTimeRange,
) -> Result<Vec<(TimestampMilli, FragmentContent)>> {
    let params = query;
    tracing::debug!(
        src_id = %params.source_id,
        dst_id = %params.dest_id,
        edge_name = %params.edge_name,
        time_range = ?params.time_range,
        "Executing EdgeFragmentsByIdTimeRange query"
    );

    use std::ops::Bound;

    let ref_time = params.reference_ts_millis.unwrap_or_else(|| TimestampMilli::now());

    let source_id = params.source_id;
    let dest_id = params.dest_id;
    let edge_name_hash = NameHash::from_name(&params.edge_name);
    let mut fragments: Vec<(TimestampMilli, FragmentContent)> = Vec::new();

    let start_key = match &params.time_range.0 {
        Bound::Unbounded => schema::EdgeFragments::key_to_bytes(
            &schema::EdgeFragmentCfKey(source_id, dest_id, edge_name_hash, TimestampMilli(0)),
        ),
        Bound::Included(start_ts) => schema::EdgeFragments::key_to_bytes(
            &schema::EdgeFragmentCfKey(source_id, dest_id, edge_name_hash, *start_ts),
        ),
        Bound::Excluded(start_ts) => schema::EdgeFragments::key_to_bytes(
            &schema::EdgeFragmentCfKey(
                source_id,
                dest_id,
                edge_name_hash,
                TimestampMilli(start_ts.0 + 1),
            ),
        ),
    };

    iterate_cf!(storage, schema::EdgeFragments, start_key, |item| {
        let (key_bytes, value_bytes) = item?;
        let key: schema::EdgeFragmentCfKey = schema::EdgeFragments::key_from_bytes(&key_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

        if key.0 != source_id || key.1 != dest_id || key.2 != edge_name_hash {
            break;
        }

        let timestamp = key.3;

        match &params.time_range.1 {
            Bound::Unbounded => {}
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

        if !schema::is_active_at_time(&value.0, ref_time) {
            continue;
        }

        fragments.push((timestamp, value.1));
    });

    Ok(fragments)
}

pub(crate) fn edge_summary_by_src_dst_name(
    storage: &super::super::Storage,
    query: &EdgeSummaryBySrcDstName,
) -> Result<(EdgeSummary, Option<EdgeWeight>, Version)> {
    let params = query;
    tracing::debug!(
        source_id = %params.source_id,
        dest_id = %params.dest_id,
        edge_name = %params.name,
        as_of = ?params.as_of,
        "Executing EdgeSummaryBySrcDstName query"
    );

    let ref_time = params.reference_ts_millis.unwrap_or_else(|| TimestampMilli::now());

    let source_id = params.source_id;
    let dest_id = params.dest_id;
    let name = &params.name;
    let name_hash = NameHash::from_name(name);

    let value = if let Ok(db) = storage.db() {
        find_edge_version(
            StorageAccess::Readonly(db),
            source_id,
            dest_id,
            name_hash,
            params.as_of,
        )?
        .map(|(_, v)| v)
    } else {
        let txn_db = storage.transaction_db()?;
        find_edge_version(
            StorageAccess::Readwrite(txn_db),
            source_id,
            dest_id,
            name_hash,
            params.as_of,
        )?
        .map(|(_, v)| v)
    };

    let value = value.ok_or_else(|| {
        if params.as_of.is_some() {
            anyhow::anyhow!(
                "Edge not found at system time {:?}: source={}, dest={}, name={}",
                params.as_of,
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

    if params.as_of.is_none() && value.5 {
        return Err(anyhow::anyhow!(
            "Edge deleted: source={}, dest={}, name={}",
            source_id,
            dest_id,
            name
        ));
    }

    if !schema::is_active_at_time(&value.1, ref_time) {
        return Err(anyhow::anyhow!(
            "Edge not valid at time {}: source={}, dest={}, name={}",
            ref_time.0,
            source_id,
            dest_id,
            name
        ));
    }

    let summary = resolve_edge_summary(storage, value.3)?;
    Ok((summary, value.2, value.4))
}

pub(crate) fn outgoing_edges(
    storage: &super::super::Storage,
    query: &OutgoingEdges,
) -> Result<Vec<(Option<EdgeWeight>, Id, Id, EdgeName, Version)>> {
    let params = query;
    tracing::debug!(node_id = %params.id, "Executing OutgoingEdges query");

    let ref_time = params.reference_ts_millis.unwrap_or_else(|| TimestampMilli::now());

    let id = params.id;
    let mut seen_edges: std::collections::HashSet<(Id, Id, NameHash)> = std::collections::HashSet::new();
    let mut edges_with_hash: Vec<(Option<EdgeWeight>, Id, Id, NameHash, Version)> = Vec::new();
    let prefix = id.into_bytes();

    if let Ok(db) = storage.db() {
        let cf = db.cf_handle(schema::ForwardEdges::CF_NAME).ok_or_else(|| {
            anyhow::anyhow!("Column family '{}' not found", schema::ForwardEdges::CF_NAME)
        })?;

        let iter = db.iterator_cf(
            cf,
            rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
        );

        for item in iter {
            let (key_bytes, value_bytes) = item?;
            if !key_bytes.starts_with(&prefix) {
                break;
            }

            let key: schema::ForwardEdgeCfKey =
                schema::ForwardEdges::key_from_bytes(&key_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

            let value: schema::ForwardEdgeCfValue =
                schema::ForwardEdges::value_from_bytes(&value_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

            if value.0.is_some() {
                continue;
            }
            if value.5 {
                continue;
            }
            if !schema::is_active_at_time(&value.1, ref_time) {
                continue;
            }

            let source_id = key.0;
            let dest_id = key.1;
            let edge_name_hash = key.2;
            let edge_topology = (source_id, dest_id, edge_name_hash);
            if seen_edges.insert(edge_topology) {
                let weight = value.2;
                edges_with_hash.push((weight, source_id, dest_id, edge_name_hash, value.4));
            }
        }
    } else {
        let txn_db = storage.transaction_db()?;
        let cf = txn_db
            .cf_handle(schema::ForwardEdges::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Column family '{}' not found", schema::ForwardEdges::CF_NAME))?;

        let iter = txn_db.iterator_cf(
            cf,
            rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
        );

        for item in iter {
            let (key_bytes, value_bytes) = item?;
            if !key_bytes.starts_with(&prefix) {
                break;
            }

            let key: schema::ForwardEdgeCfKey =
                schema::ForwardEdges::key_from_bytes(&key_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

            let value: schema::ForwardEdgeCfValue =
                schema::ForwardEdges::value_from_bytes(&value_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

            if value.0.is_some() {
                continue;
            }
            if value.5 {
                continue;
            }
            if !schema::is_active_at_time(&value.1, ref_time) {
                continue;
            }

            let source_id = key.0;
            let dest_id = key.1;
            let edge_name_hash = key.2;
            let edge_topology = (source_id, dest_id, edge_name_hash);
            if seen_edges.insert(edge_topology) {
                let weight = value.2;
                edges_with_hash.push((weight, source_id, dest_id, edge_name_hash, value.4));
            }
        }
    }

    let mut edges = Vec::with_capacity(edges_with_hash.len());
    for (weight, src_id, dst_id, name_hash, version) in edges_with_hash {
        let edge_name = resolve_name(storage, name_hash)?;
        edges.push((weight, src_id, dst_id, edge_name, version));
    }
    Ok(edges)
}

pub(crate) fn incoming_edges(
    storage: &super::super::Storage,
    query: &IncomingEdges,
) -> Result<Vec<(Option<EdgeWeight>, Id, Id, EdgeName, Version)>> {
    let params = query;
    tracing::debug!(node_id = %params.id, "Executing IncomingEdges query");

    let ref_time = params.reference_ts_millis.unwrap_or_else(|| TimestampMilli::now());

    let id = params.id;
    let mut seen_edges: std::collections::HashSet<(Id, Id, NameHash)> = std::collections::HashSet::new();
    let mut edges_with_hash: Vec<(Option<EdgeWeight>, Id, Id, NameHash, Version)> = Vec::new();
    let prefix = id.into_bytes();

    if let Ok(db) = storage.db() {
        let reverse_cf = db.cf_handle(schema::ReverseEdges::CF_NAME).ok_or_else(|| {
            anyhow::anyhow!("Column family '{}' not found", schema::ReverseEdges::CF_NAME)
        })?;

        let iter = db.iterator_cf(
            reverse_cf,
            rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
        );

        for item in iter {
            let (key_bytes, value_bytes) = item?;
            if !key_bytes.starts_with(&prefix) {
                break;
            }

            let key: schema::ReverseEdgeCfKey =
                schema::ReverseEdges::key_from_bytes(&key_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

            let value: schema::ReverseEdgeCfValue =
                schema::ReverseEdges::value_from_bytes(&value_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

            if value.0.is_some() {
                continue;
            }
            if !schema::is_active_at_time(&value.1, ref_time) {
                continue;
            }

            let dest_id = key.0;
            let source_id = key.1;
            let edge_name_hash = key.2;

            let edge_topology = (source_id, dest_id, edge_name_hash);
            if !seen_edges.insert(edge_topology) {
                continue;
            }

            let (weight, version) = match find_edge_version(
                StorageAccess::Readonly(db),
                source_id,
                dest_id,
                edge_name_hash,
                None,
            )? {
                Some((_, forward_value)) => {
                    if forward_value.5 {
                        continue;
                    }
                    (forward_value.2, forward_value.4)
                }
                None => continue,
            };

            edges_with_hash.push((weight, dest_id, source_id, edge_name_hash, version));
        }
    } else {
        let txn_db = storage.transaction_db()?;
        let reverse_cf = txn_db
            .cf_handle(schema::ReverseEdges::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Column family '{}' not found", schema::ReverseEdges::CF_NAME))?;

        let iter = txn_db.iterator_cf(
            reverse_cf,
            rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
        );

        for item in iter {
            let (key_bytes, value_bytes) = item?;
            if !key_bytes.starts_with(&prefix) {
                break;
            }

            let key: schema::ReverseEdgeCfKey =
                schema::ReverseEdges::key_from_bytes(&key_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

            let value: schema::ReverseEdgeCfValue =
                schema::ReverseEdges::value_from_bytes(&value_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

            if value.0.is_some() {
                continue;
            }
            if !schema::is_active_at_time(&value.1, ref_time) {
                continue;
            }

            let dest_id = key.0;
            let source_id = key.1;
            let edge_name_hash = key.2;

            let edge_topology = (source_id, dest_id, edge_name_hash);
            if !seen_edges.insert(edge_topology) {
                continue;
            }

            let (weight, version) = match find_edge_version(
                StorageAccess::Readwrite(txn_db),
                source_id,
                dest_id,
                edge_name_hash,
                None,
            )? {
                Some((_, forward_value)) => {
                    if forward_value.5 {
                        continue;
                    }
                    (forward_value.2, forward_value.4)
                }
                None => continue,
            };

            edges_with_hash.push((weight, dest_id, source_id, edge_name_hash, version));
        }
    }

    let mut edges = Vec::with_capacity(edges_with_hash.len());
    for (weight, dst_id, src_id, name_hash, version) in edges_with_hash {
        let edge_name = resolve_name(storage, name_hash)?;
        edges.push((weight, dst_id, src_id, edge_name, version));
    }
    Ok(edges)
}

pub(crate) fn all_nodes(
    storage: &super::super::Storage,
    query: &AllNodes,
) -> Result<Vec<(Id, NodeName, NodeSummary, Version)>> {
    let params = query;
    tracing::debug!(limit = params.limit, has_cursor = params.last.is_some(), "Executing AllNodes query");

    let scan_request = scan::AllNodes {
        last: params.last,
        limit: params.limit,
        reverse: false,
        reference_ts_millis: params.reference_ts_millis,
    };

    let mut results = Vec::with_capacity(params.limit);
    scan_request.accept(storage, &mut |record: &scan::NodeRecord| {
        results.push((record.id, record.name.clone(), record.summary.clone(), record.version));
        true
    })?;

    Ok(results)
}

pub(crate) fn all_edges(
    storage: &super::super::Storage,
    query: &AllEdges,
) -> Result<Vec<(Option<EdgeWeight>, Id, Id, EdgeName, Version)>> {
    let params = query;
    tracing::debug!(limit = params.limit, has_cursor = params.last.is_some(), "Executing AllEdges query");

    let scan_request = scan::AllEdges {
        last: params.last.clone(),
        limit: params.limit,
        reverse: false,
        reference_ts_millis: params.reference_ts_millis,
    };

    let mut results = Vec::with_capacity(params.limit);
    scan_request.accept(storage, &mut |record: &scan::EdgeRecord| {
        results.push((
            record.weight,
            record.src_id,
            record.dst_id,
            record.name.clone(),
            record.version,
        ));
        true
    })?;

    Ok(results)
}

pub(crate) fn nodes_by_summary_hash(
    storage: &super::super::Storage,
    query: &NodesBySummaryHash,
) -> Result<Vec<super::super::query::NodeSummaryLookupResult>> {
    let params = query;
    tracing::debug!(hash = ?params.hash, "Executing NodesBySummaryHash query");

    let mut results: Vec<super::super::query::NodeSummaryLookupResult> = Vec::new();
    let prefix = params.hash.as_bytes();

    iterate_cf!(storage, schema::NodeSummaryIndex, prefix, |item| {
        let (key_bytes, value_bytes) = item?;
        let key: schema::NodeSummaryIndexCfKey = schema::NodeSummaryIndex::key_from_bytes(&key_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;
        if key.0 != params.hash {
            break;
        }

        let value: schema::NodeSummaryIndexCfValue = schema::NodeSummaryIndex::value_from_bytes(&value_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

        let is_current = value.is_current();
        if params.current_only && !is_current {
            continue;
        }

        results.push(super::super::query::NodeSummaryLookupResult {
            node_id: key.1,
            version: key.2,
            is_current,
        });
    });

    Ok(results)
}

pub(crate) fn edges_by_summary_hash(
    storage: &super::super::Storage,
    query: &EdgesBySummaryHash,
) -> Result<Vec<super::super::query::EdgeSummaryLookupResult>> {
    let params = query;
    tracing::debug!(hash = ?params.hash, "Executing EdgesBySummaryHash query");

    let mut results: Vec<super::super::query::EdgeSummaryLookupResult> = Vec::new();
    let prefix = params.hash.as_bytes();

    iterate_cf!(storage, schema::EdgeSummaryIndex, prefix, |item| {
        let (key_bytes, value_bytes) = item?;
        let key: schema::EdgeSummaryIndexCfKey = schema::EdgeSummaryIndex::key_from_bytes(&key_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;
        if key.0 != params.hash {
            break;
        }

        let value: schema::EdgeSummaryIndexCfValue = schema::EdgeSummaryIndex::value_from_bytes(&value_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

        let is_current = value.is_current();
        if params.current_only && !is_current {
            continue;
        }

        results.push(super::super::query::EdgeSummaryLookupResult {
            src_id: key.1,
            dst_id: key.2,
            name_hash: key.3,
            version: key.4,
            is_current,
        });
    });

    Ok(results)
}
