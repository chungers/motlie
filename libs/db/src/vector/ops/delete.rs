//! Vector deletion operations.
//!
//! Shared helpers for deleting vectors, used by both `Processor` and
//! `MutationExecutor` implementations.

use anyhow::Result;

use crate::rocksdb::ColumnFamily;
use crate::vector::processor::Processor;
use crate::vector::schema::{
    BinaryCodeCfKey, BinaryCodes, EmbeddingCode, IdForward, IdForwardCfKey, IdReverse,
    IdReverseCfKey, VecId, VectorCfKey, Vectors,
};
use crate::Id;

// ============================================================================
// DeleteResult
// ============================================================================

/// Result of a vector deletion.
pub enum DeleteResult {
    /// Vector was deleted successfully
    Deleted {
        /// The internal VecId that was deleted
        vec_id: VecId,
        /// Whether this was a soft delete (HNSW enabled)
        soft_delete: bool,
    },
    /// Vector did not exist (idempotent)
    NotFound,
}

impl DeleteResult {
    /// Returns the VecId if the vector was deleted.
    pub fn vec_id(&self) -> Option<VecId> {
        match self {
            DeleteResult::Deleted { vec_id, .. } => Some(*vec_id),
            DeleteResult::NotFound => None,
        }
    }

    /// Returns true if this was a soft delete.
    pub fn is_soft_delete(&self) -> bool {
        matches!(
            self,
            DeleteResult::Deleted {
                soft_delete: true,
                ..
            }
        )
    }
}

// ============================================================================
// delete_vector_in_txn
// ============================================================================

/// Delete a vector within an existing transaction.
///
/// This is the shared implementation used by both `Processor::delete_vector()`
/// and `MutationExecutor for DeleteVector`.
///
/// # Behavior
///
/// **When HNSW is disabled:**
/// - Deletes ID mappings (forward and reverse)
/// - Deletes vector data and binary codes
/// - Returns VecId to free list for reuse
///
/// **When HNSW is enabled (soft delete):**
/// - Deletes ID mappings only (vector unreachable by external ID)
/// - Keeps vector data (needed for HNSW distance calculations)
/// - Does NOT free VecId (prevents graph corruption from reuse)
/// - HNSW edges remain intact (standard tombstone approach)
///
/// # Arguments
/// * `txn` - Active RocksDB transaction
/// * `txn_db` - Transaction DB for CF handles
/// * `processor` - Processor for allocators and config
/// * `embedding` - Embedding space code
/// * `id` - External ID (ULID) to delete
///
/// # Returns
/// `DeleteResult` indicating whether deletion occurred and if it was soft.
pub fn delete_vector_in_txn(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    processor: &Processor,
    embedding: EmbeddingCode,
    id: Id,
) -> Result<DeleteResult> {
    // 1. Look up VecId from external Id (with lock to prevent races)
    let forward_cf = txn_db
        .cf_handle(IdForward::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("IdForward CF not found"))?;
    let forward_key = IdForwardCfKey(embedding, id);

    let vec_id = match txn.get_for_update_cf(
        &forward_cf,
        IdForward::key_to_bytes(&forward_key),
        true,
    )? {
        Some(bytes) => IdForward::value_from_bytes(&bytes)?.0,
        None => return Ok(DeleteResult::NotFound),
    };

    // 2. Delete IdForward mapping
    txn.delete_cf(&forward_cf, IdForward::key_to_bytes(&forward_key))?;

    // 3. Delete IdReverse mapping
    let reverse_key = IdReverseCfKey(embedding, vec_id);
    let reverse_cf = txn_db
        .cf_handle(IdReverse::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("IdReverse CF not found"))?;
    txn.delete_cf(&reverse_cf, IdReverse::key_to_bytes(&reverse_key))?;

    // Check if HNSW indexing is enabled - affects cleanup strategy
    let hnsw_enabled = processor.hnsw_config().enabled;

    // 4. Delete vector data (only if HNSW disabled)
    // If HNSW is enabled, keep vector data for distance calculations
    // during search traversal (entry point or neighbor edges may reference it)
    let vec_key = VectorCfKey(embedding, vec_id);
    let vectors_cf = txn_db
        .cf_handle(Vectors::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;
    if !hnsw_enabled {
        txn.delete_cf(&vectors_cf, Vectors::key_to_bytes(&vec_key))?;
    }

    // 5. Delete binary code (only if RaBitQ enabled AND HNSW disabled)
    if processor.rabitq_enabled() && !hnsw_enabled {
        let code_key = BinaryCodeCfKey(embedding, vec_id);
        let codes_cf = txn_db
            .cf_handle(BinaryCodes::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("BinaryCodes CF not found"))?;
        txn.delete_cf(&codes_cf, BinaryCodes::key_to_bytes(&code_key))?;
    }

    // 6. Return VecId to free list (only if HNSW disabled)
    // If HNSW is enabled, we cannot reuse the VecId because existing
    // HNSW edges still reference it. Reuse would corrupt graph semantics.
    if !hnsw_enabled {
        let allocator = processor.get_or_create_allocator(embedding);
        allocator.free_in_txn(txn, txn_db, embedding, vec_id)?;
    }

    Ok(DeleteResult::Deleted {
        vec_id,
        soft_delete: hnsw_enabled,
    })
}
