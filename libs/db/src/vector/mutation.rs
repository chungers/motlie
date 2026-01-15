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

use crate::rocksdb::{ColumnFamily, MutationCodec};
use crate::Id;

use super::distance::Distance;
use super::hnsw::{insert_in_txn, CacheUpdate};
use super::processor::Processor;
use super::schema::{
    BinaryCodeCfKey, BinaryCodeCfValue, BinaryCodes, EmbeddingCode, EmbeddingSpec,
    EmbeddingSpecCfKey, EmbeddingSpecCfValue, EmbeddingSpecs, HnswLayer, IdForward, IdForwardCfKey,
    IdForwardCfValue, IdReverse, IdReverseCfKey, IdReverseCfValue, VecId, VectorCfKey, Vectors,
};
use super::writer::MutationExecutor;

// ============================================================================
// Mutation Enum
// ============================================================================

/// Mutation enum for vector storage operations.
///
/// All vector mutations are variants of this enum, enabling type-safe
/// dispatch in mutation consumers.
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
    // HNSW Graph Operations (internal, used by InsertVector)
    // ─────────────────────────────────────────────────────────────
    /// Add/update edges for a node in the HNSW graph
    UpdateEdges(UpdateEdges),
    /// Update graph metadata (entry_point, max_level, count)
    UpdateGraphMeta(UpdateGraphMeta),

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
// UpdateEdges
// ============================================================================

/// Update HNSW graph edges for a node.
///
/// Internal mutation used during HNSW graph construction and maintenance.
#[derive(Debug, Clone)]
pub struct UpdateEdges {
    /// Embedding space
    pub embedding: EmbeddingCode,
    /// Vector ID
    pub vec_id: VecId,
    /// Layer to update
    pub layer: HnswLayer,
    /// Operation: add or replace neighbors
    pub operation: EdgeOperation,
}

/// Edge update operation type.
#[derive(Debug, Clone)]
pub enum EdgeOperation {
    /// Add neighbors (merge with existing)
    Add(Vec<VecId>),
    /// Replace all neighbors at this layer
    Replace(Vec<VecId>),
}

impl From<UpdateEdges> for Mutation {
    fn from(op: UpdateEdges) -> Self {
        Mutation::UpdateEdges(op)
    }
}

// ============================================================================
// UpdateGraphMeta
// ============================================================================

/// Update graph-level metadata.
///
/// Internal mutation for updating HNSW graph state.
#[derive(Debug, Clone)]
pub struct UpdateGraphMeta {
    /// Embedding space
    pub embedding: EmbeddingCode,
    /// Field to update
    pub update: GraphMetaUpdate,
}

/// Graph metadata update type.
#[derive(Debug, Clone)]
pub enum GraphMetaUpdate {
    /// Set the entry point for the HNSW graph
    EntryPoint(VecId),
    /// Set the maximum level in the graph
    MaxLevel(HnswLayer),
    /// Increment the vector count
    IncrementCount,
    /// Decrement the vector count
    DecrementCount,
}

impl From<UpdateGraphMeta> for Mutation {
    fn from(op: UpdateGraphMeta) -> Self {
        Mutation::UpdateGraphMeta(op)
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
    /// Returns an optional `CacheUpdate` if HNSW indexing was performed.
    /// The cache update should be applied AFTER transaction commit.
    pub fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        processor: &Processor,
    ) -> Result<Option<CacheUpdate>> {
        match self {
            Mutation::AddEmbeddingSpec(op) => op.execute(txn, txn_db, processor),
            Mutation::InsertVector(op) => op.execute(txn, txn_db, processor),
            Mutation::DeleteVector(op) => op.execute(txn, txn_db, processor),
            Mutation::InsertVectorBatch(op) => op.execute(txn, txn_db, processor),
            Mutation::UpdateEdges(op) => op.execute(txn, txn_db, processor),
            Mutation::UpdateGraphMeta(op) => op.execute(txn, txn_db, processor),
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
        _processor: &Processor,
    ) -> Result<Option<CacheUpdate>> {
        let (key_bytes, value_bytes) = self.to_cf_bytes()?;
        let cf = txn_db
            .cf_handle(EmbeddingSpecs::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("EmbeddingSpecs CF not found"))?;
        txn.put_cf(&cf, key_bytes, value_bytes)?;
        Ok(None)
    }
}

impl MutationExecutor for InsertVector {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        processor: &Processor,
    ) -> Result<Option<CacheUpdate>> {
        // 0. Look up embedding spec to get storage type
        let storage_type = processor
            .registry()
            .get_by_code(self.embedding)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Embedding {} not registered; cannot determine storage type",
                    self.embedding
                )
            })?
            .storage_type();

        // 1. Allocate vec_id
        let vec_id = {
            let allocator = processor.get_or_create_allocator(self.embedding);
            allocator.allocate_in_txn(txn, txn_db, self.embedding)?
        };

        // 2. Store Id -> VecId mapping (IdForward)
        let forward_key = IdForwardCfKey(self.embedding, self.id);
        let forward_value = IdForwardCfValue(vec_id);
        let forward_cf = txn_db
            .cf_handle(IdForward::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("IdForward CF not found"))?;
        txn.put_cf(
            &forward_cf,
            IdForward::key_to_bytes(&forward_key),
            IdForward::value_to_bytes(&forward_value),
        )?;

        // 3. Store VecId -> Id mapping (IdReverse)
        let reverse_key = IdReverseCfKey(self.embedding, vec_id);
        let reverse_value = IdReverseCfValue(self.id);
        let reverse_cf = txn_db
            .cf_handle(IdReverse::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("IdReverse CF not found"))?;
        txn.put_cf(
            &reverse_cf,
            IdReverse::key_to_bytes(&reverse_key),
            IdReverse::value_to_bytes(&reverse_value),
        )?;

        // 4. Store vector data
        let vec_key = VectorCfKey(self.embedding, vec_id);
        let vectors_cf = txn_db
            .cf_handle(Vectors::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;
        txn.put_cf(
            &vectors_cf,
            Vectors::key_to_bytes(&vec_key),
            Vectors::value_to_bytes_typed(&self.vector, storage_type),
        )?;

        // 5. Store binary code (if RaBitQ enabled)
        if let Some(encoder) = processor.get_or_create_encoder(self.embedding) {
            let (code, correction) = encoder.encode_with_correction(&self.vector);
            let code_key = BinaryCodeCfKey(self.embedding, vec_id);
            let code_value = BinaryCodeCfValue { code, correction };
            let codes_cf = txn_db
                .cf_handle(BinaryCodes::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("BinaryCodes CF not found"))?;
            txn.put_cf(
                &codes_cf,
                BinaryCodes::key_to_bytes(&code_key),
                BinaryCodes::value_to_bytes(&code_value),
            )?;
        }

        // 6. HNSW indexing (if immediate_index)
        if self.immediate_index {
            if let Some(index) = processor.get_or_create_index(self.embedding) {
                let cache_update =
                    insert_in_txn(&index, txn, txn_db, processor.storage(), vec_id, &self.vector)?;
                return Ok(Some(cache_update));
            }
        }

        Ok(None)
    }
}

impl MutationExecutor for DeleteVector {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        processor: &Processor,
    ) -> Result<Option<CacheUpdate>> {
        // 1. Look up vec_id from external Id
        let forward_key = IdForwardCfKey(self.embedding, self.id);
        let forward_cf = txn_db
            .cf_handle(IdForward::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("IdForward CF not found"))?;

        let vec_id = match txn.get_cf(&forward_cf, IdForward::key_to_bytes(&forward_key))? {
            Some(bytes) => IdForward::value_from_bytes(&bytes)?.0,
            None => return Ok(None), // Already deleted or never existed
        };

        // 2. Delete IdForward mapping
        txn.delete_cf(&forward_cf, IdForward::key_to_bytes(&forward_key))?;

        // 3. Delete IdReverse mapping
        let reverse_key = IdReverseCfKey(self.embedding, vec_id);
        let reverse_cf = txn_db
            .cf_handle(IdReverse::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("IdReverse CF not found"))?;
        txn.delete_cf(&reverse_cf, IdReverse::key_to_bytes(&reverse_key))?;

        // 4. Delete vector data
        let vec_key = VectorCfKey(self.embedding, vec_id);
        let vectors_cf = txn_db
            .cf_handle(Vectors::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;
        txn.delete_cf(&vectors_cf, Vectors::key_to_bytes(&vec_key))?;

        // 5. Delete binary code (if RaBitQ enabled)
        if processor.rabitq_enabled() {
            let code_key = BinaryCodeCfKey(self.embedding, vec_id);
            let codes_cf = txn_db
                .cf_handle(BinaryCodes::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("BinaryCodes CF not found"))?;
            txn.delete_cf(&codes_cf, BinaryCodes::key_to_bytes(&code_key))?;
        }

        // 6. Return vec_id to free list
        {
            let allocator = processor.get_or_create_allocator(self.embedding);
            allocator.free_in_txn(txn, txn_db, self.embedding, vec_id)?;
        }

        // TODO: HNSW edge cleanup (mark node as deleted)
        Ok(None)
    }
}

impl MutationExecutor for InsertVectorBatch {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        processor: &Processor,
    ) -> Result<Option<CacheUpdate>> {
        // Batch mutations are expanded in Consumer::execute_mutations()
        // This implementation handles single-transaction batch execution
        // when called directly (not through Consumer)
        for (id, vector) in &self.vectors {
            let single = InsertVector {
                embedding: self.embedding,
                id: *id,
                vector: vector.clone(),
                immediate_index: self.immediate_index,
            };
            single.execute(txn, txn_db, processor)?;
        }
        Ok(None)
    }
}

impl MutationExecutor for UpdateEdges {
    fn execute(
        &self,
        _txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        _txn_db: &rocksdb::TransactionDB,
        _processor: &Processor,
    ) -> Result<Option<CacheUpdate>> {
        // Direct edge updates are handled internally by HNSW insert
        tracing::debug!("UpdateEdges handled internally by HNSW insert");
        Ok(None)
    }
}

impl MutationExecutor for UpdateGraphMeta {
    fn execute(
        &self,
        _txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        _txn_db: &rocksdb::TransactionDB,
        _processor: &Processor,
    ) -> Result<Option<CacheUpdate>> {
        // Graph metadata updates are handled internally by HNSW insert
        tracing::debug!("UpdateGraphMeta handled internally by HNSW insert");
        Ok(None)
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

        let edges = UpdateEdges {
            embedding: 1,
            vec_id: 0,
            layer: 0,
            operation: EdgeOperation::Add(vec![1, 2, 3]),
        };
        let _: Mutation = edges.into();

        let meta = UpdateGraphMeta {
            embedding: 1,
            update: GraphMetaUpdate::IncrementCount,
        };
        let _: Mutation = meta.into();
    }

    #[test]
    fn test_flush_marker() {
        let (marker, mut rx) = FlushMarker::new();
        marker.complete();
        // Should not panic - receiver gets the signal
        assert!(rx.try_recv().is_ok());
    }
}
