use anyhow::Result;

use crate::rocksdb::{ColumnFamily, ColumnFamilySerde};

use super::super::schema::{
    EdgeSummary, EdgeSummaries, EdgeSummaryCfKey, EdgeSummaryCfValue, NodeSummary,
    NodeSummaries, NodeSummaryCfKey, NodeSummaryCfValue, OrphanSummaries, OrphanSummaryCfKey,
    OrphanSummaryCfValue, SummaryKind,
};
use super::super::summary_hash::SummaryHash;

/// Ensure a node summary exists in the summaries CF.
pub(crate) fn ensure_node_summary(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    hash: SummaryHash,
    summary: &NodeSummary,
) -> Result<()> {
    let cf = txn_db
        .cf_handle(NodeSummaries::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("NodeSummaries CF not found"))?;
    let key_bytes = NodeSummaries::key_to_bytes(&NodeSummaryCfKey(hash));

    if txn.get_cf(cf, &key_bytes)?.is_some() {
        tracing::trace!(hash = ?hash, "Node summary already exists");
        return Ok(());
    }

    let value = NodeSummaryCfValue(summary.clone());
    let value_bytes = NodeSummaries::value_to_bytes(&value)?;
    txn.put_cf(cf, key_bytes, value_bytes)?;

    tracing::trace!(hash = ?hash, "Created node summary");
    Ok(())
}

/// Mark a node summary as potentially orphaned (deferred deletion).
pub(crate) fn mark_node_summary_orphan_candidate(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    hash: SummaryHash,
) -> Result<()> {
    let cf = txn_db
        .cf_handle(OrphanSummaries::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("OrphanSummaries CF not found"))?;

    let now = crate::TimestampMilli::now();
    let key = OrphanSummaryCfKey(now, hash);
    let key_bytes = OrphanSummaries::key_to_bytes(&key);
    let value = OrphanSummaryCfValue(SummaryKind::Node);
    let value_bytes = OrphanSummaries::value_to_bytes(&value)?;

    txn.put_cf(cf, key_bytes, value_bytes)?;

    tracing::trace!(hash = ?hash, "Wrote node summary orphan candidate to OrphanSummaries CF");
    Ok(())
}

/// Ensure an edge summary exists in the summaries CF.
pub(crate) fn ensure_edge_summary(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    hash: SummaryHash,
    summary: &EdgeSummary,
) -> Result<()> {
    let cf = txn_db
        .cf_handle(EdgeSummaries::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("EdgeSummaries CF not found"))?;
    let key_bytes = EdgeSummaries::key_to_bytes(&EdgeSummaryCfKey(hash));

    if txn.get_cf(cf, &key_bytes)?.is_some() {
        tracing::trace!(hash = ?hash, "Edge summary already exists");
        return Ok(());
    }

    let value = EdgeSummaryCfValue(summary.clone());
    let value_bytes = EdgeSummaries::value_to_bytes(&value)?;
    txn.put_cf(cf, key_bytes, value_bytes)?;

    tracing::trace!(hash = ?hash, "Created edge summary");
    Ok(())
}

/// Mark an edge summary as potentially orphaned (deferred deletion).
pub(crate) fn mark_edge_summary_orphan_candidate(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    hash: SummaryHash,
) -> Result<()> {
    let cf = txn_db
        .cf_handle(OrphanSummaries::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("OrphanSummaries CF not found"))?;

    let now = crate::TimestampMilli::now();
    let key = OrphanSummaryCfKey(now, hash);
    let key_bytes = OrphanSummaries::key_to_bytes(&key);
    let value = OrphanSummaryCfValue(SummaryKind::Edge);
    let value_bytes = OrphanSummaries::value_to_bytes(&value)?;

    txn.put_cf(cf, key_bytes, value_bytes)?;

    tracing::trace!(hash = ?hash, "Wrote edge summary orphan candidate to OrphanSummaries CF");
    Ok(())
}

/// Remove a summary from OrphanSummaries when it's being reused.
pub(crate) fn remove_summary_from_orphans(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    hash: SummaryHash,
) -> Result<()> {
    let cf = txn_db
        .cf_handle(OrphanSummaries::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("OrphanSummaries CF not found"))?;

    let iter = txn.iterator_cf(cf, rocksdb::IteratorMode::Start);
    let mut keys_to_delete = Vec::new();

    for item in iter {
        let (key_bytes, _value_bytes) = item?;
        let key: OrphanSummaryCfKey = OrphanSummaries::key_from_bytes(&key_bytes)?;
        if key.1 == hash {
            keys_to_delete.push(key_bytes.to_vec());
        }
    }

    for key_bytes in keys_to_delete {
        txn.delete_cf(cf, &key_bytes)?;
        tracing::trace!(hash = ?hash, "Removed summary from OrphanSummaries CF (reused by restore)");
    }

    Ok(())
}

/// Verify that a node summary exists.
pub(crate) fn verify_node_summary_exists(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    hash: SummaryHash,
) -> Result<bool> {
    let cf = txn_db
        .cf_handle(NodeSummaries::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("NodeSummaries CF not found"))?;

    let key = NodeSummaryCfKey(hash);
    let key_bytes = NodeSummaries::key_to_bytes(&key);
    let in_txn = txn.get_cf(cf, &key_bytes)?.is_some();
    let in_db = txn_db.get_cf(cf, &key_bytes)?.is_some();
    Ok(in_txn || in_db)
}

/// Verify that an edge summary exists.
pub(crate) fn verify_edge_summary_exists(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    hash: SummaryHash,
) -> Result<bool> {
    let cf = txn_db
        .cf_handle(EdgeSummaries::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("EdgeSummaries CF not found"))?;

    let key = EdgeSummaryCfKey(hash);
    let key_bytes = EdgeSummaries::key_to_bytes(&key);
    let in_txn = txn.get_cf(cf, &key_bytes)?.is_some();
    let in_db = txn_db.get_cf(cf, &key_bytes)?.is_some();
    Ok(in_txn || in_db)
}
