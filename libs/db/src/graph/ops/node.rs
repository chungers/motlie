use anyhow::Result;

use crate::rocksdb::{ColumnFamily, ColumnFamilySerde, HotColumnFamilyRecord};

use super::name::{write_name_to_cf, write_name_to_cf_cached};
use super::summary::{
    ensure_node_summary, mark_node_summary_orphan_candidate, remove_summary_from_orphans,
    verify_node_summary_exists,
};
use super::super::mutation::{
    AddNode, DeleteNode, RestoreNode, UpdateNode,
};
use super::super::name_hash::NameCache;
use super::super::schema::{
    self, NodeCfKey, NodeCfValue, NodeSummaryIndex, NodeSummaryIndexCfKey,
    NodeSummaryIndexCfValue, NodeVersionHistory, NodeVersionHistoryCfKey,
    NodeVersionHistoryCfValue, Nodes, VERSION_MAX,
};
use super::super::summary_hash::SummaryHash;
use crate::Id;

pub(crate) fn add_node(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    mutation: &AddNode,
    cache: Option<&NameCache>,
) -> Result<(Id, schema::Version)> {
    tracing::debug!(id = %mutation.id, name = %mutation.name, "Executing AddNode op");

    let name_hash = match cache {
        Some(cache) => write_name_to_cf_cached(txn, txn_db, &mutation.name, cache)?,
        None => write_name_to_cf(txn, txn_db, &mutation.name)?,
    };

    if !mutation.summary.is_empty() {
        if let Ok(summary_hash) = SummaryHash::from_summary(&mutation.summary) {
            ensure_node_summary(txn, txn_db, summary_hash, &mutation.summary)?;

            let index_cf = txn_db
                .cf_handle(NodeSummaryIndex::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("Column family '{}' not found", NodeSummaryIndex::CF_NAME))?;
            let index_key = NodeSummaryIndexCfKey(summary_hash, mutation.id, 1);
            let index_key_bytes = NodeSummaryIndex::key_to_bytes(&index_key);
            let index_value_bytes = NodeSummaryIndex::value_to_bytes(&NodeSummaryIndexCfValue::current())?;
            txn.put_cf(index_cf, index_key_bytes, index_value_bytes)?;
        }
    }

    let nodes_cf = txn_db
        .cf_handle(Nodes::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Column family '{}' not found", Nodes::CF_NAME))?;
    let (node_key, node_value) = Nodes::create_bytes(mutation)?;
    txn.put_cf(nodes_cf, node_key, node_value)?;

    let history_cf = txn_db
        .cf_handle(NodeVersionHistory::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("NodeVersionHistory CF not found"))?;
    let summary_hash = if !mutation.summary.is_empty() {
        SummaryHash::from_summary(&mutation.summary).ok()
    } else {
        None
    };
    let history_key = NodeVersionHistoryCfKey(mutation.id, mutation.ts_millis, 1);
    let history_key_bytes = NodeVersionHistory::key_to_bytes(&history_key);
    let history_value = NodeVersionHistoryCfValue(
        mutation.ts_millis,
        summary_hash,
        name_hash,
        mutation.valid_range.clone(),
    );
    let history_value_bytes = NodeVersionHistory::value_to_bytes(&history_value)?;
    txn.put_cf(history_cf, history_key_bytes, history_value_bytes)?;

    Ok((mutation.id, 1))
}

/// Consolidated node update with optimistic locking.
/// Handles any combination of active_period and summary updates.
///
/// ## Option<Option<T>> pattern:
/// - `None` = no change
/// - `Some(None)` = reset/clear the field
/// - `Some(Some(value))` = set to specific value
pub(crate) fn update_node(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    mutation: &UpdateNode,
) -> Result<(Id, schema::Version)> {
    tracing::debug!(
        id = %mutation.id,
        expected_version = mutation.expected_version,
        "Executing UpdateNode op"
    );

    // Check if any field is being updated
    if mutation.new_active_period.is_none() && mutation.new_summary.is_none() {
        tracing::warn!(id = %mutation.id, "UpdateNode called with no changes");
        return Ok((mutation.id, mutation.expected_version));
    }

    let nodes_cf = txn_db
        .cf_handle(Nodes::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Nodes CF not found"))?;

    let (node_key_bytes, current) = find_current_node_version(txn, txn_db, mutation.id)?
        .ok_or_else(|| anyhow::anyhow!("Node not found: {}", mutation.id))?;

    let current_version = current.4;
    let old_hash = current.3;
    let is_deleted = current.5;

    if is_deleted {
        return Err(anyhow::anyhow!("Cannot update deleted node: {}", mutation.id));
    }

    if current_version != mutation.expected_version {
        return Err(anyhow::anyhow!(
            "Version mismatch for node {}: expected {}, actual {}",
            mutation.id,
            mutation.expected_version,
            current_version
        ));
    }

    if current_version == VERSION_MAX {
        return Err(anyhow::anyhow!("Version overflow for node: {}", mutation.id));
    }

    let new_version = current_version + 1;
    let now = crate::TimestampMilli::now();

    // Compute new active_period
    let new_active_period = match &mutation.new_active_period {
        None => current.1.clone(),           // No change
        Some(None) => None,                  // Reset/clear
        Some(Some(p)) => Some(p.clone()),    // Set to specific period
    };

    // Compute new summary_hash
    let new_hash = if let Some(ref new_summary) = mutation.new_summary {
        let hash = SummaryHash::from_summary(new_summary)?;
        ensure_node_summary(txn, txn_db, hash, new_summary)?;
        Some(hash)
    } else {
        old_hash
    };

    // Mark old summary as orphan candidate if summary changed
    if mutation.new_summary.is_some() {
        if let Some(old_h) = old_hash {
            mark_node_summary_orphan_candidate(txn, txn_db, old_h)?;
        }
    }

    // Update summary index (mark old as stale, add new as current)
    if mutation.new_summary.is_some() {
        if let Some(old_h) = old_hash {
            let index_cf = txn_db
                .cf_handle(NodeSummaryIndex::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("NodeSummaryIndex CF not found"))?;
            let old_index_key = NodeSummaryIndexCfKey(old_h, mutation.id, current_version);
            let old_index_key_bytes = NodeSummaryIndex::key_to_bytes(&old_index_key);
            let stale_value_bytes = NodeSummaryIndex::value_to_bytes(&NodeSummaryIndexCfValue::stale())?;
            txn.put_cf(index_cf, old_index_key_bytes, stale_value_bytes)?;
        }

        if let Some(new_h) = new_hash {
            let index_cf = txn_db
                .cf_handle(NodeSummaryIndex::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("NodeSummaryIndex CF not found"))?;
            let new_index_key = NodeSummaryIndexCfKey(new_h, mutation.id, new_version);
            let new_index_key_bytes = NodeSummaryIndex::key_to_bytes(&new_index_key);
            let current_value_bytes = NodeSummaryIndex::value_to_bytes(&NodeSummaryIndexCfValue::current())?;
            txn.put_cf(index_cf, new_index_key_bytes, current_value_bytes)?;
        }
    }

    // Mark old node version with superseded_at
    let old_node_value = NodeCfValue(
        Some(now),
        current.1.clone(),
        current.2,
        current.3,
        current.4,
        current.5,
    );
    let old_node_bytes = Nodes::value_to_bytes(&old_node_value)
        .map_err(|e| anyhow::anyhow!("Failed to serialize old node version: {}", e))?;
    txn.put_cf(nodes_cf, &node_key_bytes, old_node_bytes)?;

    // Write new node version
    let new_node_key = NodeCfKey(mutation.id, now);
    let new_node_key_bytes = Nodes::key_to_bytes(&new_node_key);
    let new_node_value = NodeCfValue(
        None,
        new_active_period.clone(),
        current.2,
        new_hash,
        new_version,
        false,
    );
    let new_node_bytes = Nodes::value_to_bytes(&new_node_value)
        .map_err(|e| anyhow::anyhow!("Failed to serialize new node version: {}", e))?;
    txn.put_cf(nodes_cf, new_node_key_bytes, new_node_bytes)?;

    // Write to NodeVersionHistory
    let history_cf = txn_db
        .cf_handle(NodeVersionHistory::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("NodeVersionHistory CF not found"))?;
    let history_key = NodeVersionHistoryCfKey(mutation.id, now, new_version);
    let history_key_bytes = NodeVersionHistory::key_to_bytes(&history_key);
    let history_value = NodeVersionHistoryCfValue(
        now,
        new_hash,
        current.2,
        new_active_period,
    );
    let history_value_bytes = NodeVersionHistory::value_to_bytes(&history_value)?;
    txn.put_cf(history_cf, history_key_bytes, history_value_bytes)?;

    tracing::info!(
        id = %mutation.id,
        old_version = current_version,
        new_version = new_version,
        "UpdateNode completed"
    );

    Ok((mutation.id, new_version))
}

pub(crate) fn delete_node(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    mutation: &DeleteNode,
) -> Result<()> {
    tracing::debug!(
        id = %mutation.id,
        expected_version = mutation.expected_version,
        "Executing DeleteNode op"
    );

    let nodes_cf = txn_db
        .cf_handle(Nodes::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Nodes CF not found"))?;

    let (node_key_bytes, current) = find_current_node_version(txn, txn_db, mutation.id)?
        .ok_or_else(|| anyhow::anyhow!("Node not found: {}", mutation.id))?;

    let current_version = current.4;
    let current_hash = current.3;
    let is_deleted = current.5;

    if is_deleted {
        return Err(anyhow::anyhow!("Node already deleted: {}", mutation.id));
    }

    if current_version != mutation.expected_version {
        return Err(anyhow::anyhow!(
            "Version mismatch for node {}: expected {}, actual {}",
            mutation.id,
            mutation.expected_version,
            current_version
        ));
    }

    if current_version == VERSION_MAX {
        return Err(anyhow::anyhow!("Version overflow for node: {}", mutation.id));
    }

    let now = crate::TimestampMilli::now();
    let new_version = current_version + 1;

    let old_node_value = NodeCfValue(
        Some(now),
        current.1.clone(),
        current.2,
        current.3,
        current.4,
        current.5,
    );
    let old_node_bytes = Nodes::value_to_bytes(&old_node_value)
        .map_err(|e| anyhow::anyhow!("Failed to serialize old node version: {}", e))?;
    txn.put_cf(nodes_cf, &node_key_bytes, old_node_bytes)?;

    let new_node_key = NodeCfKey(mutation.id, now);
    let new_node_key_bytes = Nodes::key_to_bytes(&new_node_key);
    let new_node_value = NodeCfValue(
        None,
        current.1,
        current.2,
        current.3,
        new_version,
        true,
    );
    let new_node_bytes = Nodes::value_to_bytes(&new_node_value)
        .map_err(|e| anyhow::anyhow!("Failed to serialize new node version: {}", e))?;
    txn.put_cf(nodes_cf, new_node_key_bytes, new_node_bytes)?;

    if let Some(hash) = current_hash {
        mark_node_summary_orphan_candidate(txn, txn_db, hash)?;
    }

    if let Some(hash) = current_hash {
        let index_cf = txn_db
            .cf_handle(NodeSummaryIndex::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("NodeSummaryIndex CF not found"))?;
        let index_key = NodeSummaryIndexCfKey(hash, mutation.id, current_version);
        let index_key_bytes = NodeSummaryIndex::key_to_bytes(&index_key);
        let stale_value_bytes = NodeSummaryIndex::value_to_bytes(&NodeSummaryIndexCfValue::stale())?;
        txn.put_cf(index_cf, index_key_bytes, stale_value_bytes)?;
    }

    tracing::info!(
        id = %mutation.id,
        old_version = current_version,
        new_version = new_version,
        "DeleteNode completed (tombstoned)"
    );

    Ok(())
}

pub(crate) fn restore_node(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    mutation: &RestoreNode,
) -> Result<(Id, schema::Version)> {
    tracing::debug!(
        id = %mutation.id,
        as_of = ?mutation.as_of,
        "Executing RestoreNode op"
    );

    let history_cf = txn_db
        .cf_handle(NodeVersionHistory::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("NodeVersionHistory CF not found"))?;

    let prefix = mutation.id.into_bytes().to_vec();
    let iter = txn.prefix_iterator_cf(history_cf, &prefix);

    let mut target_history: Option<(NodeVersionHistoryCfKey, NodeVersionHistoryCfValue)> = None;
    for item in iter {
        let (key_bytes, value_bytes) = item?;
        if !key_bytes.starts_with(&prefix) {
            break;
        }
        let key: NodeVersionHistoryCfKey = NodeVersionHistory::key_from_bytes(&key_bytes)?;
        let value: NodeVersionHistoryCfValue = NodeVersionHistory::value_from_bytes(&value_bytes)?;

        if value.0 <= mutation.as_of {
            if target_history.is_none() || value.0 > target_history.as_ref().unwrap().1.0 {
                target_history = Some((key, value));
            }
        }
    }

    let (_history_key, history_value) = target_history
        .ok_or_else(|| anyhow::anyhow!(
            "No version found in NodeVersionHistory at or before {} for node {}",
            mutation.as_of.0,
            mutation.id
        ))?;

    if let Some(hash) = history_value.1 {
        let summary_exists = verify_node_summary_exists(txn, txn_db, hash)?;
        if !summary_exists {
            return Err(anyhow::anyhow!(
                "Cannot restore node {} - summary hash {:?} not found in NodeSummaries",
                mutation.id,
                hash
            ));
        }
    }

    let nodes_cf = txn_db
        .cf_handle(Nodes::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Nodes CF not found"))?;

    if let Some((current_key_bytes, current)) = find_current_node_version(txn, txn_db, mutation.id)? {
        let current_version = current.4;
        let current_hash = current.3;

        if let Some(expected) = mutation.expected_version {
            if current_version != expected {
                return Err(anyhow::anyhow!(
                    "Version mismatch for node {}: expected {}, actual {}",
                    mutation.id,
                    expected,
                    current_version
                ));
            }
        }

        if !current.5 {
            return Err(anyhow::anyhow!(
                "Cannot restore node {} because it is not deleted",
                mutation.id
            ));
        }

        let now = crate::TimestampMilli::now();

        let old_node_value = NodeCfValue(
            Some(now),
            current.1.clone(),
            current.2,
            current.3,
            current.4,
            current.5,
        );
        let old_node_bytes = Nodes::value_to_bytes(&old_node_value)
            .map_err(|e| anyhow::anyhow!("Failed to serialize old node version: {}", e))?;
        txn.put_cf(nodes_cf, &current_key_bytes, old_node_bytes)?;

        let new_version = current_version + 1;
        let new_node_key = NodeCfKey(mutation.id, now);
        let new_node_key_bytes = Nodes::key_to_bytes(&new_node_key);
        let new_node_value = NodeCfValue(
            None,
            history_value.3.clone(),
            history_value.2,
            history_value.1,
            new_version,
            false,
        );
        let new_node_bytes = Nodes::value_to_bytes(&new_node_value)
            .map_err(|e| anyhow::anyhow!("Failed to serialize new node version: {}", e))?;
        txn.put_cf(nodes_cf, new_node_key_bytes, new_node_bytes)?;

        if let Some(hash) = history_value.1 {
            remove_summary_from_orphans(txn, txn_db, hash)?;
        }

        if let Some(old_hash) = current_hash {
            mark_node_summary_orphan_candidate(txn, txn_db, old_hash)?;
        }

        if let Some(hash) = history_value.1 {
            let index_cf = txn_db
                .cf_handle(NodeSummaryIndex::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("NodeSummaryIndex CF not found"))?;
            let index_key = NodeSummaryIndexCfKey(hash, mutation.id, new_version);
            let index_key_bytes = NodeSummaryIndex::key_to_bytes(&index_key);
            let current_value_bytes = NodeSummaryIndex::value_to_bytes(&NodeSummaryIndexCfValue::current())?;
            txn.put_cf(index_cf, index_key_bytes, current_value_bytes)?;
        }

        if let Some(old_hash) = current_hash {
            let index_cf = txn_db
                .cf_handle(NodeSummaryIndex::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("NodeSummaryIndex CF not found"))?;
            let index_key = NodeSummaryIndexCfKey(old_hash, mutation.id, current_version);
            let index_key_bytes = NodeSummaryIndex::key_to_bytes(&index_key);
            let stale_value_bytes = NodeSummaryIndex::value_to_bytes(&NodeSummaryIndexCfValue::stale())?;
            txn.put_cf(index_cf, index_key_bytes, stale_value_bytes)?;
        }

        let history_key = NodeVersionHistoryCfKey(mutation.id, now, new_version);
        let history_key_bytes = NodeVersionHistory::key_to_bytes(&history_key);
        let history_value = NodeVersionHistoryCfValue(
            now,
            history_value.1,
            history_value.2,
            history_value.3,
        );
        let history_value_bytes = NodeVersionHistory::value_to_bytes(&history_value)?;
        txn.put_cf(history_cf, history_key_bytes, history_value_bytes)?;

        tracing::info!(
            id = %mutation.id,
            old_version = current_version,
            new_version = new_version,
            "RestoreNode completed"
        );

        Ok((mutation.id, new_version))
    } else {
        if let Some(expected) = mutation.expected_version {
            return Err(anyhow::anyhow!(
                "Version mismatch for node {}: expected {}, no current version",
                mutation.id,
                expected
            ));
        }

        let now = crate::TimestampMilli::now();
        let new_version = 1;

        let new_node_key = NodeCfKey(mutation.id, now);
        let new_node_key_bytes = Nodes::key_to_bytes(&new_node_key);
        let new_node_value = NodeCfValue(
            None,
            history_value.3.clone(),
            history_value.2,
            history_value.1,
            new_version,
            false,
        );
        let new_node_bytes = Nodes::value_to_bytes(&new_node_value)
            .map_err(|e| anyhow::anyhow!("Failed to serialize new node version: {}", e))?;
        txn.put_cf(nodes_cf, new_node_key_bytes, new_node_bytes)?;

        if let Some(hash) = history_value.1 {
            remove_summary_from_orphans(txn, txn_db, hash)?;
        }

        if let Some(hash) = history_value.1 {
            let index_cf = txn_db
                .cf_handle(NodeSummaryIndex::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("NodeSummaryIndex CF not found"))?;
            let index_key = NodeSummaryIndexCfKey(hash, mutation.id, new_version);
            let index_key_bytes = NodeSummaryIndex::key_to_bytes(&index_key);
            let current_value_bytes = NodeSummaryIndex::value_to_bytes(&NodeSummaryIndexCfValue::current())?;
            txn.put_cf(index_cf, index_key_bytes, current_value_bytes)?;
        }

        let history_key = NodeVersionHistoryCfKey(mutation.id, now, new_version);
        let history_key_bytes = NodeVersionHistory::key_to_bytes(&history_key);
        let history_value = NodeVersionHistoryCfValue(
            now,
            history_value.1,
            history_value.2,
            history_value.3,
        );
        let history_value_bytes = NodeVersionHistory::value_to_bytes(&history_value)?;
        txn.put_cf(history_cf, history_key_bytes, history_value_bytes)?;

        tracing::info!(
            id = %mutation.id,
            new_version = new_version,
            "RestoreNode completed (created new node)"
        );

        Ok((mutation.id, new_version))
    }
}

fn find_current_node_version(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    node_id: Id,
) -> Result<Option<(Vec<u8>, schema::NodeCfValue)>> {
    let nodes_cf = txn_db
        .cf_handle(Nodes::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Nodes CF not found"))?;

    let prefix = node_id.into_bytes().to_vec();
    let iter = txn.prefix_iterator_cf(nodes_cf, &prefix);

    for item in iter {
        let (key_bytes, value_bytes) = item?;
        if !key_bytes.starts_with(&prefix) {
            break;
        }
        let value: schema::NodeCfValue = Nodes::value_from_bytes(&value_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize node value: {}", e))?;
        if value.0.is_none() {
            return Ok(Some((key_bytes.to_vec(), value)));
        }
    }
    Ok(None)
}
