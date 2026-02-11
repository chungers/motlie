use anyhow::Result;

use crate::rocksdb::{ColumnFamily, ColumnFamilySerde};

use super::super::schema::{
    EdgeSummary, EdgeSummaries, EdgeSummaryCfKey, EdgeSummaryCfValue, NodeSummary,
    NodeSummaries, NodeSummaryCfKey, NodeSummaryCfValue, OrphanSummaries, OrphanSummaryCfKey,
    OrphanSummaryCfValue, SummaryKind,
};
use super::super::summary_hash::SummaryHash;
use super::super::Storage;

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

// ============================================================================
// Summary Resolution (Read Operations)
// ============================================================================

/// Resolve a node summary from the NodeSummaries cold CF.
///
/// If the summary_hash is None or the summary is not found, returns an empty DataUrl.
pub(crate) fn resolve_node_summary(
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
            Ok(value.0)
        }
        None => Ok(NodeSummary::from_text("")),
    }
}

/// Resolve a node summary from the NodeSummaries cold CF (strict).
///
/// Returns an error if the summary hash is missing or not found.
pub(crate) fn resolve_node_summary_strict(
    storage: &Storage,
    summary_hash: Option<SummaryHash>,
) -> Result<NodeSummary> {
    let hash = summary_hash.ok_or_else(|| anyhow::anyhow!("Missing node summary hash"))?;

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
            Ok(value.0)
        }
        None => Err(anyhow::anyhow!("Missing node summary for hash {:?}", hash)),
    }
}

/// Resolve a node summary from a transaction context (for read-your-writes).
pub(crate) fn resolve_node_summary_from_txn(
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
            Ok(value.0)
        }
        None => Ok(NodeSummary::from_text("")),
    }
}

/// Resolve a node summary from a transaction context (strict).
///
/// Returns an error if the summary hash is missing or not found.
pub(crate) fn resolve_node_summary_from_txn_strict(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    summary_hash: Option<SummaryHash>,
) -> Result<NodeSummary> {
    let hash = summary_hash.ok_or_else(|| anyhow::anyhow!("Missing node summary hash"))?;

    let summaries_cf = txn_db
        .cf_handle(NodeSummaries::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("NodeSummaries CF not found"))?;

    let key_bytes = NodeSummaries::key_to_bytes(&NodeSummaryCfKey(hash));

    match txn.get_cf(summaries_cf, &key_bytes)? {
        Some(bytes) => {
            let value = NodeSummaries::value_from_bytes(&bytes)?;
            Ok(value.0)
        }
        None => Err(anyhow::anyhow!("Missing node summary for hash {:?}", hash)),
    }
}

/// Resolve an edge summary from the EdgeSummaries cold CF.
///
/// If the summary_hash is None or the summary is not found, returns an empty DataUrl.
pub(crate) fn resolve_edge_summary(
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
            Ok(value.0)
        }
        None => Ok(EdgeSummary::from_text("")),
    }
}

/// Resolve an edge summary from the EdgeSummaries cold CF (strict).
///
/// Returns an error if the summary hash is missing or not found.
pub(crate) fn resolve_edge_summary_strict(
    storage: &Storage,
    summary_hash: Option<SummaryHash>,
) -> Result<EdgeSummary> {
    let hash = summary_hash.ok_or_else(|| anyhow::anyhow!("Missing edge summary hash"))?;

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
            Ok(value.0)
        }
        None => Err(anyhow::anyhow!("Missing edge summary for hash {:?}", hash)),
    }
}

/// Resolve an edge summary from a transaction context (for read-your-writes).
pub(crate) fn resolve_edge_summary_from_txn(
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
            Ok(value.0)
        }
        None => Ok(EdgeSummary::from_text("")),
    }
}

/// Resolve an edge summary from a transaction context (strict).
///
/// Returns an error if the summary hash is missing or not found.
pub(crate) fn resolve_edge_summary_from_txn_strict(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    summary_hash: Option<SummaryHash>,
) -> Result<EdgeSummary> {
    let hash = summary_hash.ok_or_else(|| anyhow::anyhow!("Missing edge summary hash"))?;

    let summaries_cf = txn_db
        .cf_handle(EdgeSummaries::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("EdgeSummaries CF not found"))?;

    let key_bytes = EdgeSummaries::key_to_bytes(&EdgeSummaryCfKey(hash));

    match txn.get_cf(summaries_cf, &key_bytes)? {
        Some(bytes) => {
            let value = EdgeSummaries::value_from_bytes(&bytes)?;
            Ok(value.0)
        }
        None => Err(anyhow::anyhow!("Missing edge summary for hash {:?}", hash)),
    }
}

/// Check whether a node summary exists in storage.
pub(crate) fn node_summary_exists(storage: &Storage, hash: SummaryHash) -> Result<bool> {
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
    Ok(value_bytes.is_some())
}

/// Check whether an edge summary exists in storage.
pub(crate) fn edge_summary_exists(storage: &Storage, hash: SummaryHash) -> Result<bool> {
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
    Ok(value_bytes.is_some())
}
