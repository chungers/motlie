use anyhow::Result;

use super::name::{write_name_to_cf, write_name_to_cf_cached};
use super::super::mutation::{AddEdgeFragment, AddNodeFragment};
use super::super::name_hash::NameCache;
use super::super::schema::{EdgeFragments, NodeFragments};
use crate::rocksdb::{ColumnFamily, MutationCodec};

pub(crate) fn add_node_fragment(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    mutation: &AddNodeFragment,
) -> Result<()> {
    tracing::debug!(
        id = %mutation.id,
        ts = %mutation.ts_millis.0,
        content_len = mutation.content.as_ref().len(),
        "Executing AddNodeFragment op"
    );

    let cf = txn_db.cf_handle(NodeFragments::CF_NAME).ok_or_else(|| {
        anyhow::anyhow!("Column family '{}' not found", NodeFragments::CF_NAME)
    })?;
    let (key, value) = mutation.to_cf_bytes()?;
    txn.put_cf(cf, key, value)?;

    Ok(())
}

pub(crate) fn add_edge_fragment(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    mutation: &AddEdgeFragment,
    cache: Option<&NameCache>,
) -> Result<()> {
    tracing::debug!(
        src = %mutation.src_id,
        dst = %mutation.dst_id,
        edge_name = %mutation.edge_name,
        ts = %mutation.ts_millis.0,
        content_len = mutation.content.as_ref().len(),
        "Executing AddEdgeFragment op"
    );

    match cache {
        Some(cache) => {
            write_name_to_cf_cached(txn, txn_db, &mutation.edge_name, cache)?;
        }
        None => {
            write_name_to_cf(txn, txn_db, &mutation.edge_name)?;
        }
    }

    let cf = txn_db.cf_handle(EdgeFragments::CF_NAME).ok_or_else(|| {
        anyhow::anyhow!("Column family '{}' not found", EdgeFragments::CF_NAME)
    })?;
    let (key, value) = mutation.to_cf_bytes()?;
    txn.put_cf(cf, key, value)?;

    Ok(())
}
