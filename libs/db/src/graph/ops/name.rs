use anyhow::Result;

use crate::rocksdb::{ColumnFamily, ColumnFamilySerde};

use super::super::name_hash::{NameCache, NameHash};
use super::super::schema::{NameCfKey, NameCfValue, Names};
use super::super::Storage;

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

// ============================================================================
// Name Resolution (Read Operations)
// ============================================================================

/// Resolve a NameHash to its full String name.
///
/// Uses the in-memory NameCache first for O(1) lookup, falling back to
/// Names CF lookup only for cache misses. On cache miss, the name is
/// added to the cache for future lookups.
///
/// This is the primary function for QueryExecutor implementations that
/// have access to Storage.
pub(crate) fn resolve_name(storage: &Storage, name_hash: NameHash) -> Result<String> {
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
pub(crate) fn resolve_name_from_txn(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    name_hash: NameHash,
    cache: &NameCache,
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
