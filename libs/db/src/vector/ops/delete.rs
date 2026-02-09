//! Vector deletion operations.
//!
//! Shared helpers for deleting vectors, used by both `Processor` and
//! `MutationExecutor` implementations.
//!
//! ## Lifecycle Integration
//!
//! Delete operations integrate with the VecLifecycle state machine:
//! - `Indexed` → `Deleted`: Normal delete of indexed vector
//! - `Pending` → `PendingDeleted`: Delete before async graph construction
//!
//! The delete also removes the vector from the Pending queue if present,
//! preventing the async updater from processing a deleted vector.

use anyhow::Result;

use crate::rocksdb::{ColumnFamily, HotColumnFamilyRecord};
use crate::vector::processor::Processor;
use crate::vector::schema::{
    BinaryCodeCfKey, BinaryCodes, EmbeddingCode, ExternalKey, IdForward, IdForwardCfKey, IdReverse,
    IdReverseCfKey, LifecycleCounts, LifecycleCountsCfKey, LifecycleCountsDelta,
    Pending, VecId, VecMeta, VecMetaCfKey, VectorCfKey, Vectors,
};

// ============================================================================
// DeleteResult
// ============================================================================

/// Result of a vector deletion.
#[derive(Debug, Clone)]
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
// ops::delete::vector
// ============================================================================

/// Delete a vector within an existing transaction.
///
/// This is the shared implementation used by both `Processor::delete_vector()`
/// and `MutationExecutor for DeleteVector`.
///
/// # Usage
///
/// ```rust,ignore
/// use crate::vector::ops;
///
/// let result = ops::delete::vector(&txn, &txn_db, processor, embedding, id)?;
/// txn.commit()?;
/// ```
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
/// - Sets VecMetadata to `VecLifecycle::Deleted` (or `PendingDeleted`)
/// - Removes from Pending queue if present
/// - Does NOT free VecId (prevents graph corruption from reuse)
/// - HNSW edges remain intact (standard tombstone approach)
///
/// # Lifecycle Transitions
///
/// The delete operation transitions the vector's lifecycle state:
/// - `Indexed` → `Deleted`
/// - `Pending` → `PendingDeleted` (async updater will skip but clean up)
///
/// # Arguments
/// * `txn` - Active RocksDB transaction
/// * `txn_db` - Transaction DB for CF handles
/// * `processor` - Processor for allocators and config
/// * `embedding` - Embedding space code
/// * `external_key` - External key to delete
///
/// # Returns
/// `DeleteResult` indicating whether deletion occurred and if it was soft.
pub(crate) fn vector(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    _processor: &Processor,
    embedding: EmbeddingCode,
    external_key: ExternalKey,
) -> Result<DeleteResult> {
    // 1. Look up VecId from external key (with lock to prevent races)
    let forward_cf = txn_db
        .cf_handle(IdForward::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("IdForward CF not found"))?;
    let forward_key = IdForwardCfKey(embedding, external_key);

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

    // 4. Update VecMetadata lifecycle state (soft delete marker)
    // This allows search to skip deleted vectors even if they're still
    // referenced by HNSW edges.
    // Note: HNSW is always enabled, so we always soft-delete rather than hard-delete.
    let was_pending = mark_deleted(txn, txn_db, embedding, vec_id)?;

    // 5. Remove from Pending queue if vector was pending
    // This prevents async updater from processing a deleted vector.
    // Note: If mark_deleted returned true, the vector was pending.
    if was_pending {
        remove_from_pending(txn, txn_db, embedding, vec_id)?;
    }

    // 6. Keep vector data for HNSW graph traversal
    // Vector data is needed for distance calculations during search.
    // Actual cleanup happens during compaction or background GC.
    // Note: HNSW is always enabled, so we never delete vector data immediately.
    let _ = VectorCfKey(embedding, vec_id); // Used by background GC
    let _ = txn_db.cf_handle(Vectors::CF_NAME); // Used by background GC

    // 7. Keep binary codes for RaBitQ search
    // Note: HNSW is always enabled, so we never delete binary codes immediately.
    let _ = BinaryCodeCfKey(embedding, vec_id); // Used by background GC
    let _ = txn_db.cf_handle(BinaryCodes::CF_NAME); // Used by background GC

    // 8. VecId reuse disabled when HNSW is enabled
    // HNSW edges still reference deleted VecIds, so we cannot reuse them.
    // The free list is only used when HNSW is disabled (which never happens now).
    // Note: processor.get_or_create_allocator(embedding).free() is NOT called.

    // 9. Update lifecycle counters via merge operator
    // Always soft delete: transition indexed->deleted or pending->pending_deleted
    let lifecycle_cf = txn_db
        .cf_handle(LifecycleCounts::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("LifecycleCounts CF not found"))?;
    let lifecycle_key = LifecycleCountsCfKey(embedding);
    let delta = if was_pending {
        LifecycleCountsDelta::pending_to_pending_deleted()
    } else {
        LifecycleCountsDelta::indexed_to_deleted()
    };
    txn.merge_cf(
        &lifecycle_cf,
        LifecycleCounts::key_to_bytes(&lifecycle_key),
        delta.to_bytes(),
    )?;

    Ok(DeleteResult::Deleted {
        vec_id,
        soft_delete: true, // Always soft delete (HNSW always enabled)
    })
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Mark a vector as deleted by updating its VecMetadata lifecycle state.
///
/// This function reads the current VecMetadata, calls `set_deleted()` to
/// transition the lifecycle state, and writes it back.
///
/// # Lifecycle Transitions
/// - `Indexed` → `Deleted`
/// - `Pending` → `PendingDeleted`
/// - `Deleted` → `Deleted` (idempotent)
/// - `PendingDeleted` → `PendingDeleted` (idempotent)
///
/// # Returns
/// `true` if the vector was in a pending state (needs removal from Pending queue)
fn mark_deleted(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    embedding: EmbeddingCode,
    vec_id: VecId,
) -> Result<bool> {
    let meta_cf = txn_db
        .cf_handle(VecMeta::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("VecMeta CF not found"))?;
    let meta_key = VecMetaCfKey(embedding, vec_id);
    let key_bytes = VecMeta::key_to_bytes(&meta_key);

    // Read current metadata (with lock for update)
    let meta_bytes = match txn.get_for_update_cf(&meta_cf, &key_bytes, true)? {
        Some(bytes) => bytes,
        None => {
            // VecMeta doesn't exist - vector may have been inserted without
            // async path (build_index=true) and never had metadata created.
            // Nothing to mark as deleted.
            return Ok(false);
        }
    };

    let mut meta_value = VecMeta::value_from_bytes(&meta_bytes)?;
    let was_pending = meta_value.0.is_pending();

    // Transition lifecycle state
    meta_value.0.set_deleted();

    // Write updated metadata
    txn.put_cf(&meta_cf, &key_bytes, &VecMeta::value_to_bytes(&meta_value)?)?;

    Ok(was_pending)
}

/// Remove a vector from the Pending queue.
///
/// This scans the Pending CF for entries matching the (embedding, vec_id)
/// and deletes them. The scan is necessary because the Pending key includes
/// a timestamp that we don't know.
///
/// Key format: `[embedding: u64][timestamp: u64][vec_id: u32]`
///
/// This is idempotent - if the vector isn't in the pending queue, this is a no-op.
fn remove_from_pending(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    embedding: EmbeddingCode,
    vec_id: VecId,
) -> Result<()> {
    let pending_cf = txn_db
        .cf_handle(Pending::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Pending CF not found"))?;

    // Scan all entries for this embedding to find the one with matching vec_id.
    // This is O(P) where P is pending items for this embedding, but deletions
    // of pending items should be rare.
    let prefix = Pending::prefix_for_embedding(embedding);

    let iter = txn_db.iterator_cf(
        &pending_cf,
        rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    );

    for item in iter {
        let (key, _value) = item?;

        // Check prefix match (stop when we leave this embedding's range)
        if key.len() < 8 || key[0..8] != prefix {
            break;
        }

        // Parse the key to check vec_id (last 4 bytes)
        let parsed = Pending::key_from_bytes(&key)?;
        if parsed.2 == vec_id {
            // Found it - delete and we're done (should only be one entry per vec_id)
            txn.delete_cf(&pending_cf, &key)?;
            break;
        }
    }

    Ok(())
}
