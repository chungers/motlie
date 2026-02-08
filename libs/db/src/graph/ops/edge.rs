use anyhow::Result;

use crate::rocksdb::{ColumnFamily, ColumnFamilySerde, HotColumnFamilyRecord};

use super::name::{write_name_to_cf, write_name_to_cf_cached};
use super::summary::{
    ensure_edge_summary, mark_edge_summary_orphan_candidate, remove_summary_from_orphans,
    verify_edge_summary_exists,
};
use super::super::mutation::{
    AddEdge, DeleteEdge, RestoreEdge, RestoreEdges, RestoreEdgesReport, UpdateEdgeSummary,
    UpdateEdgeActivePeriod, UpdateEdgeWeight,
};
use super::super::name_hash::{NameCache, NameHash};
use super::super::schema::{
    self, EdgeSummaryIndex, EdgeSummaryIndexCfKey, EdgeSummaryIndexCfValue, EdgeVersionHistory,
    EdgeVersionHistoryCfKey, EdgeVersionHistoryCfValue, ForwardEdgeCfKey, ForwardEdgeCfValue,
    ForwardEdges, ReverseEdgeCfKey, ReverseEdgeCfValue, ReverseEdges, VERSION_MAX,
};
use super::super::summary_hash::SummaryHash;
use crate::Id;

pub(crate) fn add_edge(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    mutation: &AddEdge,
    cache: Option<&NameCache>,
) -> Result<()> {
    tracing::debug!(
        src = %mutation.source_node_id,
        dst = %mutation.target_node_id,
        name = %mutation.name,
        "Executing AddEdge op"
    );

    let name_hash = match cache {
        Some(cache) => write_name_to_cf_cached(txn, txn_db, &mutation.name, cache)?,
        None => write_name_to_cf(txn, txn_db, &mutation.name)?,
    };

    if !mutation.summary.is_empty() {
        if let Ok(summary_hash) = SummaryHash::from_summary(&mutation.summary) {
            ensure_edge_summary(txn, txn_db, summary_hash, &mutation.summary)?;

            let index_cf = txn_db
                .cf_handle(EdgeSummaryIndex::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("Column family '{}' not found", EdgeSummaryIndex::CF_NAME))?;
            let index_key = EdgeSummaryIndexCfKey(
                summary_hash,
                mutation.source_node_id,
                mutation.target_node_id,
                name_hash,
                1,
            );
            let index_key_bytes = EdgeSummaryIndex::key_to_bytes(&index_key);
            let index_value_bytes = EdgeSummaryIndex::value_to_bytes(&EdgeSummaryIndexCfValue::current())?;
            txn.put_cf(index_cf, index_key_bytes, index_value_bytes)?;
        }
    }

    let forward_cf = txn_db
        .cf_handle(ForwardEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Column family '{}' not found", ForwardEdges::CF_NAME))?;
    let (forward_key, forward_value) = ForwardEdges::create_bytes(mutation)?;
    txn.put_cf(forward_cf, forward_key, forward_value)?;

    let reverse_cf = txn_db
        .cf_handle(ReverseEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Column family '{}' not found", ReverseEdges::CF_NAME))?;
    let (reverse_key, reverse_value) = ReverseEdges::create_bytes(mutation)?;
    txn.put_cf(reverse_cf, reverse_key, reverse_value)?;

    let history_cf = txn_db
        .cf_handle(EdgeVersionHistory::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("EdgeVersionHistory CF not found"))?;
    let summary_hash = if !mutation.summary.is_empty() {
        SummaryHash::from_summary(&mutation.summary).ok()
    } else {
        None
    };
    let history_key = EdgeVersionHistoryCfKey(
        mutation.source_node_id,
        mutation.target_node_id,
        name_hash,
        mutation.ts_millis,
        1,
    );
    let history_key_bytes = EdgeVersionHistory::key_to_bytes(&history_key);
    let history_value = EdgeVersionHistoryCfValue(
        mutation.ts_millis,
        summary_hash,
        mutation.weight,
        mutation.valid_range.clone(),
    );
    let history_value_bytes = EdgeVersionHistory::value_to_bytes(&history_value)?;
    txn.put_cf(history_cf, history_key_bytes, history_value_bytes)?;

    Ok(())
}

pub(crate) fn update_edge_active_period(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    mutation: &UpdateEdgeActivePeriod,
) -> Result<()> {
    let name_hash = NameHash::from_name(&mutation.name);
    update_edge_valid_range(
        txn,
        txn_db,
        mutation.src_id,
        mutation.dst_id,
        name_hash,
        mutation.temporal_range,
    )
}

pub(crate) fn update_edge_weight(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    mutation: &UpdateEdgeWeight,
) -> Result<()> {
    tracing::debug!(
        src = %mutation.src_id,
        dst = %mutation.dst_id,
        name = %mutation.name,
        weight = %mutation.weight,
        "Executing UpdateEdgeWeight op"
    );

    let name_hash = NameHash::from_name(&mutation.name);

    let forward_cf = txn_db
        .cf_handle(ForwardEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;
    let reverse_cf = txn_db
        .cf_handle(ReverseEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ReverseEdges CF not found"))?;
    let history_cf = txn_db
        .cf_handle(EdgeVersionHistory::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("EdgeVersionHistory CF not found"))?;

    let (current_forward_key_bytes, current_forward) =
        find_current_forward_edge_version(txn, txn_db, mutation.src_id, mutation.dst_id, name_hash)?
            .ok_or_else(|| anyhow::anyhow!(
                "ForwardEdge not found: src={}, dst={}, name_hash={}",
                mutation.src_id,
                mutation.dst_id,
                name_hash
            ))?;

    let (current_reverse_key_bytes, current_reverse) =
        find_current_reverse_edge_version(txn, txn_db, mutation.dst_id, mutation.src_id, name_hash)?
            .ok_or_else(|| anyhow::anyhow!(
                "ReverseEdge not found: src={}, dst={}, name_hash={}",
                mutation.src_id,
                mutation.dst_id,
                name_hash
            ))?;

    let now = crate::TimestampMilli::now();
    let new_version = current_forward.4 + 1;

    let old_forward_value = ForwardEdgeCfValue(
        Some(now),
        current_forward.1.clone(),
        current_forward.2,
        current_forward.3,
        current_forward.4,
        current_forward.5,
    );
    let old_forward_bytes = ForwardEdges::value_to_bytes(&old_forward_value)?;
    txn.put_cf(forward_cf, &current_forward_key_bytes, old_forward_bytes)?;

    let old_reverse_value = ReverseEdgeCfValue(
        Some(now),
        current_reverse.1.clone(),
    );
    let old_reverse_bytes = ReverseEdges::value_to_bytes(&old_reverse_value)?;
    txn.put_cf(reverse_cf, &current_reverse_key_bytes, old_reverse_bytes)?;

    let new_forward_key = ForwardEdgeCfKey(mutation.src_id, mutation.dst_id, name_hash, now);
    let new_forward_key_bytes = ForwardEdges::key_to_bytes(&new_forward_key);
    let new_forward_value = ForwardEdgeCfValue(
        None,
        current_forward.1.clone(),
        Some(mutation.weight),
        current_forward.3,
        new_version,
        current_forward.5,
    );
    let new_forward_bytes = ForwardEdges::value_to_bytes(&new_forward_value)?;
    txn.put_cf(forward_cf, new_forward_key_bytes, new_forward_bytes)?;

    let new_reverse_key = ReverseEdgeCfKey(mutation.dst_id, mutation.src_id, name_hash, now);
    let new_reverse_key_bytes = ReverseEdges::key_to_bytes(&new_reverse_key);
    let new_reverse_value = ReverseEdgeCfValue(
        None,
        current_forward.1.clone(),
    );
    let new_reverse_bytes = ReverseEdges::value_to_bytes(&new_reverse_value)?;
    txn.put_cf(reverse_cf, new_reverse_key_bytes, new_reverse_bytes)?;

    let history_key = EdgeVersionHistoryCfKey(mutation.src_id, mutation.dst_id, name_hash, now, new_version);
    let history_key_bytes = EdgeVersionHistory::key_to_bytes(&history_key);
    let history_value = EdgeVersionHistoryCfValue(
        now,
        current_forward.3,
        Some(mutation.weight),
        current_forward.1,
    );
    let history_value_bytes = EdgeVersionHistory::value_to_bytes(&history_value)?;
    txn.put_cf(history_cf, history_key_bytes, history_value_bytes)?;

    tracing::info!(
        src = %mutation.src_id,
        dst = %mutation.dst_id,
        old_version = current_forward.4,
        new_version = new_version,
        "UpdateEdgeWeight completed"
    );

    Ok(())
}

pub(crate) fn update_edge_summary(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    mutation: &UpdateEdgeSummary,
) -> Result<()> {
    tracing::debug!(
        src = %mutation.src_id,
        dst = %mutation.dst_id,
        name = %mutation.name,
        expected_version = mutation.expected_version,
        "Executing UpdateEdgeSummary op"
    );

    let name_hash = NameHash::from_name(&mutation.name);

    let forward_cf = txn_db
        .cf_handle(ForwardEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;

    let (edge_key_bytes, current) =
        find_current_forward_edge_version(txn, txn_db, mutation.src_id, mutation.dst_id, name_hash)?
            .ok_or_else(|| anyhow::anyhow!("Edge not found: {}→{}", mutation.src_id, mutation.dst_id))?;

    let current_version = current.4;
    let old_hash = current.3;
    let is_deleted = current.5;

    if is_deleted {
        return Err(anyhow::anyhow!(
            "Cannot update deleted edge: {}→{}",
            mutation.src_id,
            mutation.dst_id
        ));
    }

    if current_version != mutation.expected_version {
        return Err(anyhow::anyhow!(
            "Version mismatch for edge {}→{}: expected {}, actual {}",
            mutation.src_id,
            mutation.dst_id,
            mutation.expected_version,
            current_version
        ));
    }

    if current_version == VERSION_MAX {
        return Err(anyhow::anyhow!(
            "Version overflow for edge: {}→{}",
            mutation.src_id,
            mutation.dst_id
        ));
    }

    let new_version = current_version + 1;
    let new_hash = SummaryHash::from_summary(&mutation.new_summary)?;

    ensure_edge_summary(txn, txn_db, new_hash, &mutation.new_summary)?;

    if let Some(old_h) = old_hash {
        mark_edge_summary_orphan_candidate(txn, txn_db, old_h)?;
    }

    if let Some(old_h) = old_hash {
        let index_cf = txn_db
            .cf_handle(EdgeSummaryIndex::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("EdgeSummaryIndex CF not found"))?;
        let old_index_key = EdgeSummaryIndexCfKey(old_h, mutation.src_id, mutation.dst_id, name_hash, current_version);
        let old_index_key_bytes = EdgeSummaryIndex::key_to_bytes(&old_index_key);
        let stale_value_bytes = EdgeSummaryIndex::value_to_bytes(&EdgeSummaryIndexCfValue::stale())?;
        txn.put_cf(index_cf, old_index_key_bytes, stale_value_bytes)?;
    }

    let index_cf = txn_db
        .cf_handle(EdgeSummaryIndex::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("EdgeSummaryIndex CF not found"))?;
    let new_index_key = EdgeSummaryIndexCfKey(new_hash, mutation.src_id, mutation.dst_id, name_hash, new_version);
    let new_index_key_bytes = EdgeSummaryIndex::key_to_bytes(&new_index_key);
    let current_value_bytes = EdgeSummaryIndex::value_to_bytes(&EdgeSummaryIndexCfValue::current())?;
    txn.put_cf(index_cf, new_index_key_bytes, current_value_bytes)?;

    let now = crate::TimestampMilli::now();

    let old_edge_value = ForwardEdgeCfValue(
        Some(now),
        current.1.clone(),
        current.2,
        current.3,
        current.4,
        current.5,
    );
    let old_edge_bytes = ForwardEdges::value_to_bytes(&old_edge_value)
        .map_err(|e| anyhow::anyhow!("Failed to serialize old edge version: {}", e))?;
    txn.put_cf(forward_cf, &edge_key_bytes, old_edge_bytes)?;

    let new_edge_key = ForwardEdgeCfKey(mutation.src_id, mutation.dst_id, name_hash, now);
    let new_edge_key_bytes = ForwardEdges::key_to_bytes(&new_edge_key);
    let new_edge_value = ForwardEdgeCfValue(
        None,
        current.1,
        current.2,
        Some(new_hash),
        new_version,
        false,
    );
    let new_edge_bytes = ForwardEdges::value_to_bytes(&new_edge_value)
        .map_err(|e| anyhow::anyhow!("Failed to serialize new edge version: {}", e))?;
    txn.put_cf(forward_cf, new_edge_key_bytes, new_edge_bytes)?;

    let history_cf = txn_db
        .cf_handle(EdgeVersionHistory::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("EdgeVersionHistory CF not found"))?;
    let history_key = EdgeVersionHistoryCfKey(mutation.src_id, mutation.dst_id, name_hash, now, new_version);
    let history_key_bytes = EdgeVersionHistory::key_to_bytes(&history_key);
    let history_value = EdgeVersionHistoryCfValue(
        now,
        Some(new_hash),
        current.2,
        current.1.clone(),
    );
    let history_value_bytes = EdgeVersionHistory::value_to_bytes(&history_value)?;
    txn.put_cf(history_cf, history_key_bytes, history_value_bytes)?;

    tracing::info!(
        src = %mutation.src_id,
        dst = %mutation.dst_id,
        old_version = current_version,
        new_version = new_version,
        "UpdateEdgeSummary completed"
    );

    Ok(())
}

pub(crate) fn delete_edge(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    mutation: &DeleteEdge,
) -> Result<()> {
    tracing::debug!(
        src = %mutation.src_id,
        dst = %mutation.dst_id,
        name = %mutation.name,
        expected_version = mutation.expected_version,
        "Executing DeleteEdge op"
    );

    let name_hash = NameHash::from_name(&mutation.name);

    let forward_cf = txn_db
        .cf_handle(ForwardEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;

    let (edge_key_bytes, current) =
        find_current_forward_edge_version(txn, txn_db, mutation.src_id, mutation.dst_id, name_hash)?
            .ok_or_else(|| anyhow::anyhow!("Edge not found: {}→{}", mutation.src_id, mutation.dst_id))?;

    let current_version = current.4;
    let current_hash = current.3;
    let is_deleted = current.5;

    if is_deleted {
        return Err(anyhow::anyhow!(
            "Edge already deleted: {}→{}",
            mutation.src_id,
            mutation.dst_id
        ));
    }

    if current_version != mutation.expected_version {
        return Err(anyhow::anyhow!(
            "Version mismatch for edge {}→{}: expected {}, actual {}",
            mutation.src_id,
            mutation.dst_id,
            mutation.expected_version,
            current_version
        ));
    }

    if current_version == VERSION_MAX {
        return Err(anyhow::anyhow!(
            "Version overflow for edge: {}→{}",
            mutation.src_id,
            mutation.dst_id
        ));
    }

    let now = crate::TimestampMilli::now();
    let new_version = current_version + 1;

    let old_edge_value = ForwardEdgeCfValue(
        Some(now),
        current.1.clone(),
        current.2,
        current.3,
        current.4,
        current.5,
    );
    let old_edge_bytes = ForwardEdges::value_to_bytes(&old_edge_value)
        .map_err(|e| anyhow::anyhow!("Failed to serialize old edge version: {}", e))?;
    txn.put_cf(forward_cf, &edge_key_bytes, old_edge_bytes)?;

    let new_edge_key = ForwardEdgeCfKey(mutation.src_id, mutation.dst_id, name_hash, now);
    let new_edge_key_bytes = ForwardEdges::key_to_bytes(&new_edge_key);
    let new_edge_value = ForwardEdgeCfValue(
        None,
        current.1,
        current.2,
        current.3,
        new_version,
        true,
    );
    let new_edge_bytes = ForwardEdges::value_to_bytes(&new_edge_value)
        .map_err(|e| anyhow::anyhow!("Failed to serialize new edge version: {}", e))?;
    txn.put_cf(forward_cf, new_edge_key_bytes, new_edge_bytes)?;

    if let Some(hash) = current_hash {
        mark_edge_summary_orphan_candidate(txn, txn_db, hash)?;
    }

    if let Some(hash) = current_hash {
        let index_cf = txn_db
            .cf_handle(EdgeSummaryIndex::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("EdgeSummaryIndex CF not found"))?;
        let index_key = EdgeSummaryIndexCfKey(hash, mutation.src_id, mutation.dst_id, name_hash, current_version);
        let index_key_bytes = EdgeSummaryIndex::key_to_bytes(&index_key);
        let stale_value_bytes = EdgeSummaryIndex::value_to_bytes(&EdgeSummaryIndexCfValue::stale())?;
        txn.put_cf(index_cf, index_key_bytes, stale_value_bytes)?;
    }

    tracing::info!(
        src = %mutation.src_id,
        dst = %mutation.dst_id,
        old_version = current_version,
        new_version = new_version,
        "DeleteEdge completed (tombstoned)"
    );

    Ok(())
}

pub(crate) fn restore_edge(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    mutation: &RestoreEdge,
) -> Result<()> {
    tracing::debug!(
        src = %mutation.src_id,
        dst = %mutation.dst_id,
        name = %mutation.name,
        as_of = ?mutation.as_of,
        "Executing RestoreEdge op"
    );

    let name_hash = NameHash::from_name(&mutation.name);

    let history_cf = txn_db
        .cf_handle(EdgeVersionHistory::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("EdgeVersionHistory CF not found"))?;

    let mut prefix = Vec::with_capacity(40);
    prefix.extend_from_slice(&mutation.src_id.into_bytes());
    prefix.extend_from_slice(&mutation.dst_id.into_bytes());
    prefix.extend_from_slice(name_hash.as_bytes());
    let iter = txn.prefix_iterator_cf(history_cf, &prefix);

    let mut target_history: Option<(EdgeVersionHistoryCfKey, EdgeVersionHistoryCfValue)> = None;
    for item in iter {
        let (key_bytes, value_bytes) = item?;
        if !key_bytes.starts_with(&prefix) {
            break;
        }
        let key: EdgeVersionHistoryCfKey = EdgeVersionHistory::key_from_bytes(&key_bytes)?;
        let value: EdgeVersionHistoryCfValue = EdgeVersionHistory::value_from_bytes(&value_bytes)?;

        if value.0 <= mutation.as_of {
            if target_history.is_none() || value.0 > target_history.as_ref().unwrap().1.0 {
                target_history = Some((key, value));
            }
        }
    }

    let (_history_key, history_value) = target_history
        .ok_or_else(|| anyhow::anyhow!(
            "No version found in EdgeVersionHistory at or before {} for edge {}→{}",
            mutation.as_of.0,
            mutation.src_id,
            mutation.dst_id
        ))?;

    if let Some(hash) = history_value.1 {
        let summary_exists = verify_edge_summary_exists(txn, txn_db, hash)?;
        if !summary_exists {
            return Err(anyhow::anyhow!(
                "Cannot restore edge {}→{} - summary hash {:?} not found in EdgeSummaries",
                mutation.src_id,
                mutation.dst_id,
                hash
            ));
        }
    }

    let forward_cf = txn_db
        .cf_handle(ForwardEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;
    let reverse_cf = txn_db
        .cf_handle(ReverseEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ReverseEdges CF not found"))?;

    if let Some((current_key_bytes, current)) =
        find_current_forward_edge_version(txn, txn_db, mutation.src_id, mutation.dst_id, name_hash)?
    {
        let current_version = current.4;
        let current_hash = current.3;

        if !current.5 {
            return Err(anyhow::anyhow!(
                "Cannot restore edge {}→{} because it is not deleted",
                mutation.src_id,
                mutation.dst_id
            ));
        }

        let now = crate::TimestampMilli::now();

        let old_edge_value = ForwardEdgeCfValue(
            Some(now),
            current.1.clone(),
            current.2,
            current.3,
            current.4,
            current.5,
        );
        let old_edge_bytes = ForwardEdges::value_to_bytes(&old_edge_value)
            .map_err(|e| anyhow::anyhow!("Failed to serialize old edge version: {}", e))?;
        txn.put_cf(forward_cf, &current_key_bytes, old_edge_bytes)?;

        if let Some((reverse_key_bytes, reverse)) =
            find_current_reverse_edge_version(txn, txn_db, mutation.dst_id, mutation.src_id, name_hash)?
        {
            let old_reverse_value = ReverseEdgeCfValue(
                Some(now),
                reverse.1.clone(),
            );
            let old_reverse_bytes = ReverseEdges::value_to_bytes(&old_reverse_value)?;
            txn.put_cf(reverse_cf, &reverse_key_bytes, old_reverse_bytes)?;
        }

        let new_version = current_version + 1;
        let new_forward_key = ForwardEdgeCfKey(mutation.src_id, mutation.dst_id, name_hash, now);
        let new_forward_key_bytes = ForwardEdges::key_to_bytes(&new_forward_key);
        let new_forward_value = ForwardEdgeCfValue(
            None,
            history_value.3.clone(),
            history_value.2,
            history_value.1,
            new_version,
            false,
        );
        let new_forward_bytes = ForwardEdges::value_to_bytes(&new_forward_value)?;
        txn.put_cf(forward_cf, new_forward_key_bytes, new_forward_bytes)?;

        let new_reverse_key = ReverseEdgeCfKey(mutation.dst_id, mutation.src_id, name_hash, now);
        let new_reverse_key_bytes = ReverseEdges::key_to_bytes(&new_reverse_key);
        let new_reverse_value = ReverseEdgeCfValue(
            None,
            history_value.3.clone(),
        );
        let new_reverse_bytes = ReverseEdges::value_to_bytes(&new_reverse_value)?;
        txn.put_cf(reverse_cf, new_reverse_key_bytes, new_reverse_bytes)?;

        if let Some(hash) = history_value.1 {
            remove_summary_from_orphans(txn, txn_db, hash)?;
        }

        if let Some(old_hash) = current_hash {
            mark_edge_summary_orphan_candidate(txn, txn_db, old_hash)?;
        }

        if let Some(hash) = history_value.1 {
            let index_cf = txn_db
                .cf_handle(EdgeSummaryIndex::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("EdgeSummaryIndex CF not found"))?;
            let index_key = EdgeSummaryIndexCfKey(hash, mutation.src_id, mutation.dst_id, name_hash, new_version);
            let index_key_bytes = EdgeSummaryIndex::key_to_bytes(&index_key);
            let current_value_bytes = EdgeSummaryIndex::value_to_bytes(&EdgeSummaryIndexCfValue::current())?;
            txn.put_cf(index_cf, index_key_bytes, current_value_bytes)?;
        }

        if let Some(old_hash) = current_hash {
            let index_cf = txn_db
                .cf_handle(EdgeSummaryIndex::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("EdgeSummaryIndex CF not found"))?;
            let index_key = EdgeSummaryIndexCfKey(old_hash, mutation.src_id, mutation.dst_id, name_hash, current_version);
            let index_key_bytes = EdgeSummaryIndex::key_to_bytes(&index_key);
            let stale_value_bytes = EdgeSummaryIndex::value_to_bytes(&EdgeSummaryIndexCfValue::stale())?;
            txn.put_cf(index_cf, index_key_bytes, stale_value_bytes)?;
        }

        let history_key = EdgeVersionHistoryCfKey(mutation.src_id, mutation.dst_id, name_hash, now, new_version);
        let history_key_bytes = EdgeVersionHistory::key_to_bytes(&history_key);
        let history_value = EdgeVersionHistoryCfValue(
            now,
            history_value.1,
            history_value.2,
            history_value.3,
        );
        let history_value_bytes = EdgeVersionHistory::value_to_bytes(&history_value)?;
        txn.put_cf(history_cf, history_key_bytes, history_value_bytes)?;

        tracing::info!(
            src = %mutation.src_id,
            dst = %mutation.dst_id,
            old_version = current_version,
            new_version = new_version,
            "RestoreEdge completed"
        );

        Ok(())
    } else {
        let now = crate::TimestampMilli::now();
        let new_version = 1;

        let new_forward_key = ForwardEdgeCfKey(mutation.src_id, mutation.dst_id, name_hash, now);
        let new_forward_key_bytes = ForwardEdges::key_to_bytes(&new_forward_key);
        let new_forward_value = ForwardEdgeCfValue(
            None,
            history_value.3.clone(),
            history_value.2,
            history_value.1,
            new_version,
            false,
        );
        let new_forward_bytes = ForwardEdges::value_to_bytes(&new_forward_value)?;
        txn.put_cf(forward_cf, new_forward_key_bytes, new_forward_bytes)?;

        let new_reverse_key = ReverseEdgeCfKey(mutation.dst_id, mutation.src_id, name_hash, now);
        let new_reverse_key_bytes = ReverseEdges::key_to_bytes(&new_reverse_key);
        let new_reverse_value = ReverseEdgeCfValue(
            None,
            history_value.3.clone(),
        );
        let new_reverse_bytes = ReverseEdges::value_to_bytes(&new_reverse_value)?;
        txn.put_cf(reverse_cf, new_reverse_key_bytes, new_reverse_bytes)?;

        if let Some(hash) = history_value.1 {
            remove_summary_from_orphans(txn, txn_db, hash)?;
        }

        if let Some(hash) = history_value.1 {
            let index_cf = txn_db
                .cf_handle(EdgeSummaryIndex::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("EdgeSummaryIndex CF not found"))?;
            let index_key = EdgeSummaryIndexCfKey(hash, mutation.src_id, mutation.dst_id, name_hash, new_version);
            let index_key_bytes = EdgeSummaryIndex::key_to_bytes(&index_key);
            let current_value_bytes = EdgeSummaryIndex::value_to_bytes(&EdgeSummaryIndexCfValue::current())?;
            txn.put_cf(index_cf, index_key_bytes, current_value_bytes)?;
        }

        let history_key = EdgeVersionHistoryCfKey(mutation.src_id, mutation.dst_id, name_hash, now, new_version);
        let history_key_bytes = EdgeVersionHistory::key_to_bytes(&history_key);
        let history_value = EdgeVersionHistoryCfValue(
            now,
            history_value.1,
            history_value.2,
            history_value.3,
        );
        let history_value_bytes = EdgeVersionHistory::value_to_bytes(&history_value)?;
        txn.put_cf(history_cf, history_key_bytes, history_value_bytes)?;

        tracing::info!(
            src = %mutation.src_id,
            dst = %mutation.dst_id,
            new_version = new_version,
            "RestoreEdge completed (created new edge)"
        );

        Ok(())
    }
}

pub(crate) fn restore_edges(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    mutation: &RestoreEdges,
) -> Result<()> {
    restore_edges_with_report(txn, txn_db, mutation).map(|_| ())
}

pub(crate) fn restore_edges_with_report(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    mutation: &RestoreEdges,
) -> Result<RestoreEdgesReport> {
    tracing::debug!(
        src = %mutation.src_id,
        name = ?mutation.name,
        as_of = ?mutation.as_of,
        "Executing RestoreEdges op"
    );

    let history_cf = txn_db
        .cf_handle(EdgeVersionHistory::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("EdgeVersionHistory CF not found"))?;

    let forward_cf = txn_db
        .cf_handle(ForwardEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;
    let reverse_cf = txn_db
        .cf_handle(ReverseEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ReverseEdges CF not found"))?;

    let filter_hash = mutation.name.as_ref().map(|name| NameHash::from_name(name));

    let prefix = mutation.src_id.into_bytes().to_vec();
    let iter = txn.iterator_cf(
        forward_cf,
        rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    );

    let mut edges_to_restore: Vec<(Id, NameHash)> = Vec::new();
    let mut seen: std::collections::HashSet<(Id, NameHash)> = std::collections::HashSet::new();

    for item in iter {
        let (key_bytes, value_bytes) = item?;
        if !key_bytes.starts_with(&prefix) {
            break;
        }
        let key: ForwardEdgeCfKey = ForwardEdges::key_from_bytes(&key_bytes)?;
        let value: ForwardEdgeCfValue = ForwardEdges::value_from_bytes(&value_bytes)?;

        if value.0.is_some() {
            continue;
        }

        let dst_id = key.1;
        let name_hash = key.2;

        if let Some(filter) = filter_hash {
            if name_hash != filter {
                continue;
            }
        }

        if seen.insert((dst_id, name_hash)) {
            edges_to_restore.push((dst_id, name_hash));
        }
    }

    let now = crate::TimestampMilli::now();
    let mut restored_count = 0u32;
    let mut report = RestoreEdgesReport {
        candidates: edges_to_restore.len() as u32,
        restorable: 0,
        skipped_no_version: Vec::new(),
    };

    for (dst_id, name_hash) in edges_to_restore {
        let mut history_prefix = Vec::with_capacity(40);
        history_prefix.extend_from_slice(&mutation.src_id.into_bytes());
        history_prefix.extend_from_slice(&dst_id.into_bytes());
        history_prefix.extend_from_slice(name_hash.as_bytes());

        let history_iter = txn.prefix_iterator_cf(history_cf, &history_prefix);

        let mut target_history: Option<(EdgeVersionHistoryCfKey, EdgeVersionHistoryCfValue)> = None;
        for item in history_iter {
            let (key_bytes, value_bytes) = item?;
            if !key_bytes.starts_with(&history_prefix) {
                break;
            }
            let key: EdgeVersionHistoryCfKey = EdgeVersionHistory::key_from_bytes(&key_bytes)?;
            let value: EdgeVersionHistoryCfValue = EdgeVersionHistory::value_from_bytes(&value_bytes)?;

            if value.0 <= mutation.as_of {
                if target_history.is_none() || value.0 > target_history.as_ref().unwrap().1.0 {
                    target_history = Some((key, value));
                }
            }
        }

        let Some((_history_key, history_value)) = target_history else {
            tracing::debug!(
                src = %mutation.src_id,
                dst = %dst_id,
                "No version found at as_of for edge, skipping"
            );
            report.skipped_no_version.push((dst_id, name_hash));
            continue;
        };

        let current_edge = find_current_forward_edge_version(
            txn,
            txn_db,
            mutation.src_id,
            dst_id,
            name_hash,
        )?;
        let current_hash = current_edge.as_ref().and_then(|(_, current)| current.3);

        if let Some(hash) = history_value.1 {
            let summary_exists = verify_edge_summary_exists(txn, txn_db, hash)?;
            if !summary_exists {
                return Err(anyhow::anyhow!(
                    "Cannot restore edge {}→{} - summary hash {:?} not found in EdgeSummaries",
                    mutation.src_id,
                    dst_id,
                    hash
                ));
            }
        }

        if let Some(hash) = current_hash {
            if history_value.1.map_or(true, |history| history != hash) {
                let summary_exists = verify_edge_summary_exists(txn, txn_db, hash)?;
                if !summary_exists {
                    return Err(anyhow::anyhow!(
                        "Cannot restore edge {}→{} - current summary hash {:?} not found in EdgeSummaries",
                        mutation.src_id,
                        dst_id,
                        hash
                    ));
                }
            }
        }

        if let Some((current_key_bytes, current)) = current_edge {
            let current_version = current.4;

            if !current.5 {
                return Err(anyhow::anyhow!(
                    "Cannot restore edge {}→{} because it is not deleted",
                    mutation.src_id,
                    dst_id
                ));
            }

            if mutation.dry_run {
                report.restorable += 1;
                restored_count += 1;
                continue;
            }

            let old_edge_value = ForwardEdgeCfValue(
                Some(now),
                current.1.clone(),
                current.2,
                current.3,
                current.4,
                current.5,
            );
            let old_edge_bytes = ForwardEdges::value_to_bytes(&old_edge_value)
                .map_err(|e| anyhow::anyhow!("Failed to serialize old edge version: {}", e))?;
            txn.put_cf(forward_cf, &current_key_bytes, old_edge_bytes)?;

            if let Some((reverse_key_bytes, reverse)) =
                find_current_reverse_edge_version(txn, txn_db, dst_id, mutation.src_id, name_hash)?
            {
                let old_reverse_value = ReverseEdgeCfValue(
                    Some(now),
                    reverse.1.clone(),
                );
                let old_reverse_bytes = ReverseEdges::value_to_bytes(&old_reverse_value)?;
                txn.put_cf(reverse_cf, &reverse_key_bytes, old_reverse_bytes)?;
            }

            let new_version = current_version + 1;
            let new_forward_key = ForwardEdgeCfKey(mutation.src_id, dst_id, name_hash, now);
            let new_forward_key_bytes = ForwardEdges::key_to_bytes(&new_forward_key);
            let new_forward_value = ForwardEdgeCfValue(
                None,
                history_value.3.clone(),
                history_value.2,
                history_value.1,
                new_version,
                false,
            );
            let new_forward_bytes = ForwardEdges::value_to_bytes(&new_forward_value)?;
            txn.put_cf(forward_cf, new_forward_key_bytes, new_forward_bytes)?;

            let new_reverse_key = ReverseEdgeCfKey(dst_id, mutation.src_id, name_hash, now);
            let new_reverse_key_bytes = ReverseEdges::key_to_bytes(&new_reverse_key);
            let new_reverse_value = ReverseEdgeCfValue(
                None,
                history_value.3.clone(),
            );
            let new_reverse_bytes = ReverseEdges::value_to_bytes(&new_reverse_value)?;
            txn.put_cf(reverse_cf, new_reverse_key_bytes, new_reverse_bytes)?;

            if let Some(hash) = history_value.1 {
                remove_summary_from_orphans(txn, txn_db, hash)?;
            }

            if let Some(old_hash) = current_hash {
                mark_edge_summary_orphan_candidate(txn, txn_db, old_hash)?;
            }

            if let Some(hash) = history_value.1 {
                let index_cf = txn_db
                    .cf_handle(EdgeSummaryIndex::CF_NAME)
                    .ok_or_else(|| anyhow::anyhow!("EdgeSummaryIndex CF not found"))?;
                let index_key = EdgeSummaryIndexCfKey(hash, mutation.src_id, dst_id, name_hash, new_version);
                let index_key_bytes = EdgeSummaryIndex::key_to_bytes(&index_key);
                let current_value_bytes = EdgeSummaryIndex::value_to_bytes(&EdgeSummaryIndexCfValue::current())?;
                txn.put_cf(index_cf, index_key_bytes, current_value_bytes)?;
            }

            if let Some(old_hash) = current_hash {
                let index_cf = txn_db
                    .cf_handle(EdgeSummaryIndex::CF_NAME)
                    .ok_or_else(|| anyhow::anyhow!("EdgeSummaryIndex CF not found"))?;
                let index_key = EdgeSummaryIndexCfKey(old_hash, mutation.src_id, dst_id, name_hash, current_version);
                let index_key_bytes = EdgeSummaryIndex::key_to_bytes(&index_key);
                let stale_value_bytes = EdgeSummaryIndex::value_to_bytes(&EdgeSummaryIndexCfValue::stale())?;
                txn.put_cf(index_cf, index_key_bytes, stale_value_bytes)?;
            }

            let history_key = EdgeVersionHistoryCfKey(mutation.src_id, dst_id, name_hash, now, new_version);
            let history_key_bytes = EdgeVersionHistory::key_to_bytes(&history_key);
            let history_value = EdgeVersionHistoryCfValue(
                now,
                history_value.1,
                history_value.2,
                history_value.3,
            );
            let history_value_bytes = EdgeVersionHistory::value_to_bytes(&history_value)?;
            txn.put_cf(history_cf, history_key_bytes, history_value_bytes)?;

            tracing::info!(
                src = %mutation.src_id,
                dst = %dst_id,
                old_version = current_version,
                new_version = new_version,
                "RestoreEdge completed"
            );
        } else {
            if mutation.dry_run {
                report.restorable += 1;
                restored_count += 1;
                continue;
            }

            let new_version = 1;

            let new_forward_key = ForwardEdgeCfKey(mutation.src_id, dst_id, name_hash, now);
            let new_forward_key_bytes = ForwardEdges::key_to_bytes(&new_forward_key);
            let new_forward_value = ForwardEdgeCfValue(
                None,
                history_value.3.clone(),
                history_value.2,
                history_value.1,
                new_version,
                false,
            );
            let new_forward_bytes = ForwardEdges::value_to_bytes(&new_forward_value)?;
            txn.put_cf(forward_cf, new_forward_key_bytes, new_forward_bytes)?;

            let new_reverse_key = ReverseEdgeCfKey(dst_id, mutation.src_id, name_hash, now);
            let new_reverse_key_bytes = ReverseEdges::key_to_bytes(&new_reverse_key);
            let new_reverse_value = ReverseEdgeCfValue(
                None,
                history_value.3.clone(),
            );
            let new_reverse_bytes = ReverseEdges::value_to_bytes(&new_reverse_value)?;
            txn.put_cf(reverse_cf, new_reverse_key_bytes, new_reverse_bytes)?;

            if let Some(hash) = history_value.1 {
                remove_summary_from_orphans(txn, txn_db, hash)?;
            }

            if let Some(hash) = history_value.1 {
                let index_cf = txn_db
                    .cf_handle(EdgeSummaryIndex::CF_NAME)
                    .ok_or_else(|| anyhow::anyhow!("EdgeSummaryIndex CF not found"))?;
                let index_key = EdgeSummaryIndexCfKey(hash, mutation.src_id, dst_id, name_hash, new_version);
                let index_key_bytes = EdgeSummaryIndex::key_to_bytes(&index_key);
                let current_value_bytes = EdgeSummaryIndex::value_to_bytes(&EdgeSummaryIndexCfValue::current())?;
                txn.put_cf(index_cf, index_key_bytes, current_value_bytes)?;
            }

            let history_key = EdgeVersionHistoryCfKey(mutation.src_id, dst_id, name_hash, now, new_version);
            let history_key_bytes = EdgeVersionHistory::key_to_bytes(&history_key);
            let history_value = EdgeVersionHistoryCfValue(
                now,
                history_value.1,
                history_value.2,
                history_value.3,
            );
            let history_value_bytes = EdgeVersionHistory::value_to_bytes(&history_value)?;
            txn.put_cf(history_cf, history_key_bytes, history_value_bytes)?;

            tracing::info!(
                src = %mutation.src_id,
                dst = %dst_id,
                new_version = new_version,
                "RestoreEdge completed (created new edge)"
            );
        }

        restored_count += 1;
        report.restorable += 1;
    }

    tracing::info!(
        src = %mutation.src_id,
        name = ?mutation.name,
        as_of = ?mutation.as_of,
        restored_count = restored_count,
        dry_run = mutation.dry_run,
        "RestoreEdges batch completed"
    );
    if mutation.dry_run {
        tracing::info!(
            src = %mutation.src_id,
            name = ?mutation.name,
            as_of = ?mutation.as_of,
            candidates = report.candidates,
            restorable = report.restorable,
            skipped_no_version = ?report.skipped_no_version,
            "RestoreEdges dry_run report"
        );
    }

    Ok(report)
}

fn find_current_forward_edge_version(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    src_id: Id,
    dst_id: Id,
    name_hash: NameHash,
) -> Result<Option<(Vec<u8>, schema::ForwardEdgeCfValue)>> {
    let forward_cf = txn_db
        .cf_handle(ForwardEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;

    let mut prefix = Vec::with_capacity(40);
    prefix.extend_from_slice(&src_id.into_bytes());
    prefix.extend_from_slice(&dst_id.into_bytes());
    prefix.extend_from_slice(name_hash.as_bytes());
    let iter = txn.prefix_iterator_cf(forward_cf, &prefix);

    for item in iter {
        let (key_bytes, value_bytes) = item?;
        if !key_bytes.starts_with(&prefix) {
            break;
        }
        let value: schema::ForwardEdgeCfValue = ForwardEdges::value_from_bytes(&value_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize edge value: {}", e))?;
        if value.0.is_none() {
            return Ok(Some((key_bytes.to_vec(), value)));
        }
    }
    Ok(None)
}

fn find_current_reverse_edge_version(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    dst_id: Id,
    src_id: Id,
    name_hash: NameHash,
) -> Result<Option<(Vec<u8>, schema::ReverseEdgeCfValue)>> {
    let reverse_cf = txn_db
        .cf_handle(ReverseEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ReverseEdges CF not found"))?;

    let mut prefix = Vec::with_capacity(40);
    prefix.extend_from_slice(&dst_id.into_bytes());
    prefix.extend_from_slice(&src_id.into_bytes());
    prefix.extend_from_slice(name_hash.as_bytes());
    let iter = txn.prefix_iterator_cf(reverse_cf, &prefix);

    for item in iter {
        let (key_bytes, value_bytes) = item?;
        if !key_bytes.starts_with(&prefix) {
            break;
        }
        let value: schema::ReverseEdgeCfValue = ReverseEdges::value_from_bytes(&value_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize reverse edge value: {}", e))?;
        if value.0.is_none() {
            return Ok(Some((key_bytes.to_vec(), value)));
        }
    }
    Ok(None)
}

pub(crate) fn update_edge_valid_range(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    src_id: Id,
    dst_id: Id,
    name_hash: NameHash,
    new_range: schema::ActivePeriod,
) -> Result<()> {
    let forward_cf = txn_db
        .cf_handle(ForwardEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ForwardEdges CF not found"))?;
    let reverse_cf = txn_db
        .cf_handle(ReverseEdges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("ReverseEdges CF not found"))?;
    let history_cf = txn_db
        .cf_handle(EdgeVersionHistory::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("EdgeVersionHistory CF not found"))?;

    let (current_forward_key_bytes, current_forward) =
        find_current_forward_edge_version(txn, txn_db, src_id, dst_id, name_hash)?
            .ok_or_else(|| anyhow::anyhow!(
                "ForwardEdge not found: src={}, dst={}, name_hash={}",
                src_id, dst_id, name_hash
            ))?;

    let (current_reverse_key_bytes, current_reverse) =
        find_current_reverse_edge_version(txn, txn_db, dst_id, src_id, name_hash)?
            .ok_or_else(|| anyhow::anyhow!(
                "ReverseEdge not found: src={}, dst={}, name_hash={}",
                src_id, dst_id, name_hash
            ))?;

    let now = crate::TimestampMilli::now();
    let new_version = current_forward.4 + 1;

    let old_forward_value = ForwardEdgeCfValue(
        Some(now),
        current_forward.1.clone(),
        current_forward.2,
        current_forward.3,
        current_forward.4,
        current_forward.5,
    );
    let old_forward_bytes = ForwardEdges::value_to_bytes(&old_forward_value)?;
    txn.put_cf(forward_cf, &current_forward_key_bytes, old_forward_bytes)?;

    let old_reverse_value = ReverseEdgeCfValue(
        Some(now),
        current_reverse.1.clone(),
    );
    let old_reverse_bytes = ReverseEdges::value_to_bytes(&old_reverse_value)?;
    txn.put_cf(reverse_cf, &current_reverse_key_bytes, old_reverse_bytes)?;

    let new_forward_key = ForwardEdgeCfKey(src_id, dst_id, name_hash, now);
    let new_forward_key_bytes = ForwardEdges::key_to_bytes(&new_forward_key);
    let new_forward_value = ForwardEdgeCfValue(
        None,
        Some(new_range.clone()),
        current_forward.2,
        current_forward.3,
        new_version,
        current_forward.5,
    );
    let new_forward_bytes = ForwardEdges::value_to_bytes(&new_forward_value)?;
    txn.put_cf(forward_cf, new_forward_key_bytes, new_forward_bytes)?;

    let new_reverse_key = ReverseEdgeCfKey(dst_id, src_id, name_hash, now);
    let new_reverse_key_bytes = ReverseEdges::key_to_bytes(&new_reverse_key);
    let new_reverse_value = ReverseEdgeCfValue(
        None,
        Some(new_range.clone()),
    );
    let new_reverse_bytes = ReverseEdges::value_to_bytes(&new_reverse_value)?;
    txn.put_cf(reverse_cf, new_reverse_key_bytes, new_reverse_bytes)?;

    let history_key = EdgeVersionHistoryCfKey(src_id, dst_id, name_hash, now, new_version);
    let history_key_bytes = EdgeVersionHistory::key_to_bytes(&history_key);
    let history_value = EdgeVersionHistoryCfValue(
        now,
        current_forward.3,
        current_forward.2,
        Some(new_range),
    );
    let history_value_bytes = EdgeVersionHistory::value_to_bytes(&history_value)?;
    txn.put_cf(history_cf, history_key_bytes, history_value_bytes)?;

    Ok(())
}
