//! Vector mutation types.
//!
//! This module contains mutation type definitions for vector storage operations.
//! Following the pattern from `graph::mutation`, mutations are grouped in an enum
//! for type-safe dispatch.
//!
//! Each mutation type implements `MutationExecutor` to define how it executes
//! against storage. The `Mutation::execute()` method dispatches to the appropriate
//! executor.
//!
//! Mutations implement `MutationCodec` to marshal themselves to CF key-value pairs.
//! This keeps marshaling logic with the mutation definitions rather than in schema.

use anyhow::Result;
use tokio::sync::oneshot;

use crate::rocksdb::MutationCodec;
use crate::Id;

use super::distance::Distance;
use super::processor::Processor;
use super::schema::{
    EmbeddingCode, EmbeddingSpec, EmbeddingSpecCfKey, EmbeddingSpecCfValue, EmbeddingSpecs,
};
use super::writer::{MutationCacheUpdate, MutationExecutor};

// ============================================================================
// Mutation Enum
// ============================================================================

/// Mutation enum for vector storage operations.
///
/// All vector mutations are variants of this enum, enabling type-safe
/// dispatch in mutation consumers.
///
/// **Note on graph repair:** HNSW graph operations (edge updates, metadata)
/// are handled internally during InsertVector. There is no public API for
/// partial graph repair - repair requires full rebuild.
#[derive(Debug)]
pub enum Mutation {
    // ─────────────────────────────────────────────────────────────
    // Embedding Space Management
    // ─────────────────────────────────────────────────────────────
    /// Register a new embedding space
    AddEmbeddingSpec(AddEmbeddingSpec),

    // ─────────────────────────────────────────────────────────────
    // Vector Operations
    // ─────────────────────────────────────────────────────────────
    /// Insert a new vector into an embedding space
    InsertVector(InsertVector),
    /// Delete a vector from an embedding space
    DeleteVector(DeleteVector),
    /// Batch insert multiple vectors (optimization)
    InsertVectorBatch(InsertVectorBatch),

    // ─────────────────────────────────────────────────────────────
    // Synchronization
    // ─────────────────────────────────────────────────────────────
    /// Synchronization marker with oneshot channel for flush semantics
    Flush(FlushMarker),
}

// ============================================================================
// AddEmbeddingSpec
// ============================================================================

/// Mutation for registering an embedding space.
///
/// Used by `EmbeddingRegistry::register()` to persist new embedding specs
/// to the `EmbeddingSpecs` column family.
///
/// # Example
///
/// ```rust,ignore
/// let add_op = AddEmbeddingSpec {
///     code: allocated_code,
///     model: "gemma".to_string(),
///     dim: 768,
///     distance: Distance::Cosine,
/// };
/// let (key_bytes, value_bytes) = EmbeddingSpecs::create_bytes(&add_op)?;
/// ```
#[derive(Debug, Clone)]
pub struct AddEmbeddingSpec {
    /// Allocated embedding code (primary key)
    pub code: EmbeddingCode,
    /// Model name (e.g., "gemma", "qwen3", "ada-002")
    pub model: String,
    /// Vector dimensionality (e.g., 128, 768, 1536)
    pub dim: u32,
    /// Distance metric for similarity computation
    pub distance: Distance,
    /// Storage type for vectors (default: F32)
    pub storage_type: super::schema::VectorElementType,
    /// HNSW M parameter (default: 16)
    pub hnsw_m: u16,
    /// HNSW ef_construction parameter (default: 200)
    pub hnsw_ef_construction: u16,
    /// RaBitQ bits per dimension (default: 1)
    pub rabitq_bits: u8,
    /// RaBitQ rotation seed (default: 42)
    pub rabitq_seed: u64,
}

impl From<AddEmbeddingSpec> for Mutation {
    fn from(op: AddEmbeddingSpec) -> Self {
        Mutation::AddEmbeddingSpec(op)
    }
}

// ============================================================================
// MutationCodec Implementations
// ============================================================================

impl MutationCodec for AddEmbeddingSpec {
    type Cf = EmbeddingSpecs;

    fn to_record(&self) -> (EmbeddingSpecCfKey, EmbeddingSpecCfValue) {
        let key = EmbeddingSpecCfKey(self.code);
        let value = EmbeddingSpecCfValue(EmbeddingSpec {
            model: self.model.clone(),
            dim: self.dim,
            distance: self.distance,
            storage_type: self.storage_type,
            hnsw_m: self.hnsw_m,
            hnsw_ef_construction: self.hnsw_ef_construction,
            rabitq_bits: self.rabitq_bits,
            rabitq_seed: self.rabitq_seed,
        });
        (key, value)
    }
}

// ============================================================================
// InsertVector
// ============================================================================

/// Insert a vector into an embedding space.
///
/// This mutation:
/// 1. Allocates a u32 vec_id via IdAllocator
/// 2. Stores ULID ↔ vec_id mappings (IdForward, IdReverse)
/// 3. Stores the raw vector (Vectors CF)
/// 4. Stores vector metadata (VecMeta CF)
/// 5. Optionally: queues for async HNSW graph update (Pending CF)
#[derive(Debug, Clone)]
pub struct InsertVector {
    /// Embedding space to insert into
    pub embedding: EmbeddingCode,
    /// External document ID
    pub id: Id,
    /// Vector data (must match embedding dimension)
    pub vector: Vec<f32>,
    /// Whether to index immediately or queue for async processing
    pub immediate_index: bool,
}

impl InsertVector {
    /// Create a new InsertVector mutation.
    pub fn new(embedding: EmbeddingCode, id: Id, vector: Vec<f32>) -> Self {
        Self {
            embedding,
            id,
            vector,
            immediate_index: false,
        }
    }

    /// Set immediate indexing (synchronous HNSW update).
    pub fn immediate(mut self) -> Self {
        self.immediate_index = true;
        self
    }
}

impl From<InsertVector> for Mutation {
    fn from(op: InsertVector) -> Self {
        Mutation::InsertVector(op)
    }
}

// ============================================================================
// DeleteVector
// ============================================================================

/// Delete a vector from an embedding space.
///
/// This mutation:
/// 1. Looks up the vec_id from the ULID
/// 2. Removes the ULID ↔ vec_id mappings
/// 3. Removes the vector data
/// 4. Removes vector metadata
/// 5. Returns the vec_id to the free list
/// 6. Marks edges for cleanup (lazy deletion)
#[derive(Debug, Clone)]
pub struct DeleteVector {
    /// Embedding space
    pub embedding: EmbeddingCode,
    /// External document ID
    pub id: Id,
}

impl DeleteVector {
    /// Create a new DeleteVector mutation.
    pub fn new(embedding: EmbeddingCode, id: Id) -> Self {
        Self { embedding, id }
    }
}

impl From<DeleteVector> for Mutation {
    fn from(op: DeleteVector) -> Self {
        Mutation::DeleteVector(op)
    }
}

// ============================================================================
// InsertVectorBatch
// ============================================================================

/// Batch insert multiple vectors for efficiency.
///
/// More efficient than individual InsertVector mutations for bulk loading.
/// All vectors in the batch must belong to the same embedding space.
#[derive(Debug, Clone)]
pub struct InsertVectorBatch {
    /// Embedding space
    pub embedding: EmbeddingCode,
    /// Batch of (id, vector) pairs
    pub vectors: Vec<(Id, Vec<f32>)>,
    /// Whether to index immediately
    pub immediate_index: bool,
}

impl InsertVectorBatch {
    /// Create a new batch insert mutation.
    pub fn new(embedding: EmbeddingCode, vectors: Vec<(Id, Vec<f32>)>) -> Self {
        Self {
            embedding,
            vectors,
            immediate_index: false,
        }
    }

    /// Set immediate indexing.
    pub fn immediate(mut self) -> Self {
        self.immediate_index = true;
        self
    }
}

impl From<InsertVectorBatch> for Mutation {
    fn from(op: InsertVectorBatch) -> Self {
        Mutation::InsertVectorBatch(op)
    }
}

// ============================================================================
// FlushMarker
// ============================================================================

/// Synchronization marker for flush semantics.
///
/// When a FlushMarker is processed by the consumer, it signals the
/// oneshot channel to indicate that all prior mutations have been committed.
/// This enables the `Writer::flush()` and `Writer::send_sync()` methods.
#[derive(Debug)]
pub struct FlushMarker(pub(crate) oneshot::Sender<()>);

impl FlushMarker {
    /// Create a new flush marker with a oneshot channel.
    pub fn new() -> (Self, oneshot::Receiver<()>) {
        let (tx, rx) = oneshot::channel();
        (Self(tx), rx)
    }

    /// Signal that flush is complete.
    pub fn complete(self) {
        let _ = self.0.send(());
    }
}

impl From<FlushMarker> for Mutation {
    fn from(op: FlushMarker) -> Self {
        Mutation::Flush(op)
    }
}

// ============================================================================
// Mutation Dispatch
// ============================================================================

impl Mutation {
    /// Execute this mutation against storage.
    ///
    /// This is the main dispatch method that routes to the appropriate
    /// `MutationExecutor` implementation based on the mutation variant.
    ///
    /// Returns an optional `MutationCacheUpdate` for cache updates.
    /// The cache update should be applied AFTER transaction commit.
    pub fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        processor: &Processor,
    ) -> Result<Option<MutationCacheUpdate>> {
        match self {
            Mutation::AddEmbeddingSpec(op) => op.execute(txn, txn_db, processor),
            Mutation::InsertVector(op) => op.execute(txn, txn_db, processor),
            Mutation::DeleteVector(op) => op.execute(txn, txn_db, processor),
            Mutation::InsertVectorBatch(op) => op.execute(txn, txn_db, processor),
            Mutation::Flush(_) => Ok(None), // No-op for storage
        }
    }
}

// ============================================================================
// MutationExecutor Implementations
// ============================================================================

impl MutationExecutor for AddEmbeddingSpec {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        processor: &Processor,
    ) -> Result<Option<MutationCacheUpdate>> {
        // Delegate to shared ops helper (also updates in-memory registry)
        super::ops::add_embedding_spec_in_txn(txn, txn_db, processor, self)?;
        Ok(None)
    }
}

impl MutationExecutor for InsertVector {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        processor: &Processor,
    ) -> Result<Option<MutationCacheUpdate>> {
        // Delegate to shared ops helper (includes all validation)
        let result = super::ops::insert_vector_in_txn(
            txn,
            txn_db,
            processor,
            self.embedding,
            self.id,
            &self.vector,
            self.immediate_index,
        )?;

        // Return combined cache update for both nav and code caches
        Ok(MutationCacheUpdate::from_insert(
            self.embedding,
            result.vec_id,
            result.nav_cache_update,
            result.code_cache_update,
        ))
    }
}

impl MutationExecutor for DeleteVector {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        processor: &Processor,
    ) -> Result<Option<MutationCacheUpdate>> {
        // Delegate to shared ops helper (handles soft-delete when HNSW enabled)
        let result =
            super::ops::delete_vector_in_txn(txn, txn_db, processor, self.embedding, self.id)?;

        // Log soft-delete if it occurred
        if result.is_soft_delete() {
            tracing::debug!(
                vec_id = ?result.vec_id(),
                "DeleteVector: soft-delete (HNSW enabled, vector data retained)"
            );
        }

        Ok(None)
    }
}

impl MutationExecutor for InsertVectorBatch {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        processor: &Processor,
    ) -> Result<Option<MutationCacheUpdate>> {
        // Delegate to shared ops helper (includes batch validation)
        let result = super::ops::insert_batch_in_txn(
            txn,
            txn_db,
            processor,
            self.embedding,
            &self.vectors,
            self.immediate_index,
        )?;

        tracing::debug!(
            count = result.vec_ids.len(),
            "InsertVectorBatch: inserted vectors"
        );

        // Return combined batch cache update
        Ok(MutationCacheUpdate::from_batch(
            self.embedding,
            &result.vec_ids,
            result.nav_cache_updates,
            result.code_cache_updates,
        ))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_vector_builder() {
        let mutation = InsertVector::new(1, Id::new(), vec![1.0, 2.0, 3.0]);
        assert_eq!(mutation.embedding, 1);
        assert!(!mutation.immediate_index);

        let mutation = mutation.immediate();
        assert!(mutation.immediate_index);
    }

    #[test]
    fn test_delete_vector() {
        let id = Id::new();
        let mutation = DeleteVector::new(42, id);
        assert_eq!(mutation.embedding, 42);
        assert_eq!(mutation.id, id);
    }

    #[test]
    fn test_insert_batch() {
        let vectors = vec![
            (Id::new(), vec![1.0, 2.0]),
            (Id::new(), vec![3.0, 4.0]),
        ];
        let mutation = InsertVectorBatch::new(1, vectors.clone());
        assert_eq!(mutation.embedding, 1);
        assert_eq!(mutation.vectors.len(), 2);
        assert!(!mutation.immediate_index);
    }

    #[test]
    fn test_mutation_from_conversions() {
        let insert = InsertVector::new(1, Id::new(), vec![1.0]);
        let _: Mutation = insert.into();

        let delete = DeleteVector::new(1, Id::new());
        let _: Mutation = delete.into();

        let batch = InsertVectorBatch::new(1, vec![(Id::new(), vec![1.0])]);
        let _: Mutation = batch.into();
    }

    #[test]
    fn test_flush_marker() {
        let (marker, mut rx) = FlushMarker::new();
        marker.complete();
        // Should not panic - receiver gets the signal
        assert!(rx.try_recv().is_ok());
    }
}
