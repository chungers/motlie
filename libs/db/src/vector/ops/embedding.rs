//! Embedding spec operations.
//!
//! Shared helpers for managing embedding specs, used by both `Processor` and
//! `MutationExecutor` implementations.

use anyhow::Result;

use crate::rocksdb::{ColumnFamily, MutationCodec};
use crate::vector::embedding::EmbeddingBuilder;
use crate::vector::mutation::AddEmbeddingSpec;
use crate::vector::processor::Processor;
use crate::vector::schema::EmbeddingSpecs;

// ============================================================================
// ops::embedding::spec
// ============================================================================

/// Add an embedding spec within an existing transaction.
///
/// This is the shared implementation used by both direct registration and
/// `MutationExecutor for AddEmbeddingSpec`.
///
/// # Usage
///
/// ```rust,ignore
/// use crate::vector::ops;
///
/// ops::embedding::spec(&txn, &txn_db, processor, &add_spec)?;
/// txn.commit()?;
/// ```
///
/// # What it does
/// 1. Validates build parameters via EmbeddingBuilder (single source of truth)
/// 2. Writes the EmbeddingSpec to RocksDB (transactional)
/// 3. Updates the in-memory EmbeddingRegistry
///
/// # Arguments
/// * `txn` - Active RocksDB transaction
/// * `txn_db` - Transaction DB for CF handles
/// * `processor` - Processor for registry access
/// * `add_spec` - The AddEmbeddingSpec to persist
///
/// # Errors
/// - Invalid build parameters (hnsw_m < 2, ef_construction < 1, invalid rabitq_bits)
/// - RocksDB write failure
///
/// # Note on atomicity
///
/// The registry update happens immediately (before transaction commit).
/// If the transaction is rolled back, the registry will be inconsistent.
/// However:
/// - In practice, AddEmbeddingSpec rarely fails after validation
/// - Registry is eventually consistent on restart (prewarm from DB)
/// - This matches the pattern used elsewhere in the codebase
///
/// For stricter atomicity, consider adding a rollback callback, but this
/// is overkill for the current use case.
pub fn spec(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    processor: &Processor,
    spec: &AddEmbeddingSpec,
) -> Result<()> {
    // 1. Validate build parameters using EmbeddingBuilder (single source of truth)
    //    This avoids duplicating validation logic in AddEmbeddingSpec.
    EmbeddingBuilder::new(&spec.model, spec.dim, spec.distance)
        .with_hnsw_m(spec.hnsw_m)
        .with_hnsw_ef_construction(spec.hnsw_ef_construction)
        .with_rabitq_bits(spec.rabitq_bits)
        .with_rabitq_seed(spec.rabitq_seed)
        .validate()?;

    // 2. Serialize and write to RocksDB
    let (key_bytes, value_bytes) = spec.to_cf_bytes()?;
    let cf = txn_db
        .cf_handle(EmbeddingSpecs::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("EmbeddingSpecs CF not found"))?;
    txn.put_cf(&cf, key_bytes, value_bytes)?;

    // 3. Update in-memory registry
    // This ensures subsequent mutations in the same process can resolve the code
    processor.registry().register_from_db(
        spec.code,
        &spec.model,
        spec.dim,
        spec.distance,
        spec.storage_type,
    );

    Ok(())
}
