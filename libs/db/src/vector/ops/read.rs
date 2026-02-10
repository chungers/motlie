//! Read operations for vector queries.
//!
//! This module provides the core read logic used by `QueryExecutor`
//! implementations. Each function takes storage/registry references
//! directly rather than going through `Processor`.

use anyhow::Result;

use super::super::embedding::Embedding;
use super::super::registry::EmbeddingRegistry;
use super::super::schema::{
    EmbeddingCode, ExternalKey, IdForward, IdForwardCfKey, IdReverse, IdReverseCfKey, VecId,
    VectorCfKey, Vectors,
};
use super::super::Storage;
use crate::rocksdb::ColumnFamily;
use crate::Id;

/// Look up a vector's data by external ID.
///
/// Returns `None` if the external ID is not found or the vector data is missing.
pub(crate) fn get_vector(
    storage: &Storage,
    embedding: EmbeddingCode,
    id: Id,
) -> Result<Option<Vec<f32>>> {
    let forward_key = IdForwardCfKey(embedding, ExternalKey::NodeId(id));
    let forward_key_bytes = IdForward::key_to_bytes(&forward_key);

    let txn_db = storage.transaction_db()?;
    let forward_cf = txn_db
        .cf_handle(IdForward::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("IdForward CF not found"))?;

    let vec_id = match txn_db.get_cf(&forward_cf, &forward_key_bytes)? {
        Some(bytes) => IdForward::value_from_bytes(&bytes)?.0,
        None => return Ok(None),
    };

    let vec_key = VectorCfKey(embedding, vec_id);
    let vec_key_bytes = Vectors::key_to_bytes(&vec_key);

    let vectors_cf = txn_db
        .cf_handle(Vectors::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;

    match txn_db.get_cf(&vectors_cf, &vec_key_bytes)? {
        Some(bytes) => {
            let value = Vectors::value_from_bytes(&bytes)?;
            Ok(Some(value.0))
        }
        None => Ok(None),
    }
}

/// Resolve an external key to its internal vec_id.
pub(crate) fn get_internal_id(
    storage: &Storage,
    embedding: EmbeddingCode,
    external_key: &ExternalKey,
) -> Result<Option<VecId>> {
    let forward_key = IdForwardCfKey(embedding, external_key.clone());
    let forward_key_bytes = IdForward::key_to_bytes(&forward_key);

    let txn_db = storage.transaction_db()?;
    let forward_cf = txn_db
        .cf_handle(IdForward::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("IdForward CF not found"))?;

    match txn_db.get_cf(&forward_cf, &forward_key_bytes)? {
        Some(bytes) => Ok(Some(IdForward::value_from_bytes(&bytes)?.0)),
        None => Ok(None),
    }
}

/// Resolve an internal vec_id to its external key.
pub(crate) fn get_external_id(
    storage: &Storage,
    embedding: EmbeddingCode,
    vec_id: VecId,
) -> Result<Option<ExternalKey>> {
    let reverse_key = IdReverseCfKey(embedding, vec_id);
    let reverse_key_bytes = IdReverse::key_to_bytes(&reverse_key);

    let txn_db = storage.transaction_db()?;
    let reverse_cf = txn_db
        .cf_handle(IdReverse::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("IdReverse CF not found"))?;

    match txn_db.get_cf(&reverse_cf, &reverse_key_bytes)? {
        Some(bytes) => {
            let external_key = IdReverse::value_from_bytes(&bytes)?.0;
            Ok(Some(external_key))
        }
        None => Ok(None),
    }
}

/// Batch resolve vec_ids to external keys using multi_get_cf.
pub(crate) fn resolve_ids(
    storage: &Storage,
    embedding: EmbeddingCode,
    vec_ids: &[VecId],
) -> Result<Vec<Option<ExternalKey>>> {
    if vec_ids.is_empty() {
        return Ok(Vec::new());
    }

    let txn_db = storage.transaction_db()?;
    let reverse_cf = txn_db
        .cf_handle(IdReverse::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("IdReverse CF not found"))?;

    let keys: Vec<Vec<u8>> = vec_ids
        .iter()
        .map(|&vec_id| {
            let key = IdReverseCfKey(embedding, vec_id);
            IdReverse::key_to_bytes(&key)
        })
        .collect();

    let results: Vec<std::result::Result<Option<Vec<u8>>, rocksdb::Error>> =
        txn_db.multi_get_cf(keys.iter().map(|k| (&reverse_cf, k.as_slice())));

    let resolved: Vec<Option<ExternalKey>> = results
        .into_iter()
        .map(|result| {
            result
                .ok()
                .flatten()
                .and_then(|bytes| IdReverse::value_from_bytes(&bytes).ok().map(|v| v.0))
        })
        .collect();

    Ok(resolved)
}

/// List all registered embedding spaces.
pub(crate) fn list_embeddings(registry: &EmbeddingRegistry) -> Vec<Embedding> {
    registry.list_all()
}

/// Find embedding spaces matching filter criteria.
pub(crate) fn find_embeddings(
    registry: &EmbeddingRegistry,
    filter: &super::super::registry::EmbeddingFilter,
) -> Vec<Embedding> {
    registry.find(filter)
}
