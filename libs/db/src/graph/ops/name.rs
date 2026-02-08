use anyhow::Result;

use crate::rocksdb::{ColumnFamily, ColumnFamilySerde};

use super::super::name_hash::{NameCache, NameHash};
use super::super::schema::{NameCfKey, NameCfValue, Names};

/// Write a name to the Names CF (idempotent).
pub(crate) fn write_name_to_cf(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    name: &str,
) -> Result<NameHash> {
    let name_hash = NameHash::from_name(name);

    let names_cf = txn_db
        .cf_handle(Names::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Column family '{}' not found", Names::CF_NAME))?;

    let key_bytes = Names::key_to_bytes(&NameCfKey(name_hash));
    let value_bytes = Names::value_to_bytes(&NameCfValue(name.to_string()))?;

    txn.put_cf(names_cf, key_bytes, value_bytes)?;

    Ok(name_hash)
}

/// Write a name to the Names CF with cache optimization.
pub(crate) fn write_name_to_cf_cached(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    name: &str,
    cache: &NameCache,
) -> Result<NameHash> {
    // Check cache first - if already interned, skip DB write
    let (name_hash, is_new) = cache.intern_if_new(name);

    if is_new {
        let names_cf = txn_db
            .cf_handle(Names::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Column family '{}' not found", Names::CF_NAME))?;

        let key_bytes = Names::key_to_bytes(&NameCfKey(name_hash));
        let value_bytes = Names::value_to_bytes(&NameCfValue(name.to_string()))?;

        txn.put_cf(names_cf, key_bytes, value_bytes)?;
        tracing::trace!(name = %name, "Wrote new name to Names CF");
    } else {
        tracing::trace!(name = %name, "Name already cached, skipping DB write");
    }

    Ok(name_hash)
}
