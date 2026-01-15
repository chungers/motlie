//! Embedding spec operations.
//!
//! Shared helpers for managing embedding specs, used by both `Processor` and
//! `MutationExecutor` implementations.

use anyhow::Result;

use crate::rocksdb::{ColumnFamily, MutationCodec};
use crate::vector::mutation::AddEmbeddingSpec;
use crate::vector::processor::Processor;
use crate::vector::schema::{EmbeddingSpecs, VectorElementType};

// ============================================================================
// add_embedding_spec_in_txn
// ============================================================================

/// Add an embedding spec within an existing transaction.
///
/// This is the shared implementation used by both direct registration and
/// `MutationExecutor for AddEmbeddingSpec`.
///
/// # What it does
/// 1. Writes the EmbeddingSpec to RocksDB (transactional)
/// 2. Updates the in-memory EmbeddingRegistry
///
/// # Arguments
/// * `txn` - Active RocksDB transaction
/// * `txn_db` - Transaction DB for CF handles
/// * `processor` - Processor for registry access
/// * `spec` - The AddEmbeddingSpec to persist
///
/// # Note on atomicity
///
/// The registry update happens immediately (before transaction commit).
/// If the transaction is rolled back, the registry will be inconsistent.
/// However:
/// - In practice, AddEmbeddingSpec rarely fails
/// - Registry is eventually consistent on restart (prewarm from DB)
/// - This matches the pattern used elsewhere in the codebase
///
/// For stricter atomicity, consider adding a rollback callback, but this
/// is overkill for the current use case.
pub fn add_embedding_spec_in_txn(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    processor: &Processor,
    spec: &AddEmbeddingSpec,
) -> Result<()> {
    // 1. Serialize and write to RocksDB
    let (key_bytes, value_bytes) = spec.to_cf_bytes()?;
    let cf = txn_db
        .cf_handle(EmbeddingSpecs::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("EmbeddingSpecs CF not found"))?;
    txn.put_cf(&cf, key_bytes, value_bytes)?;

    // 2. Update in-memory registry
    // This ensures subsequent mutations in the same process can resolve the code
    processor.registry().register_from_db(
        spec.code,
        &spec.model,
        spec.dim,
        spec.distance,
        VectorElementType::default(), // F32 for backward compat
    );

    Ok(())
}
