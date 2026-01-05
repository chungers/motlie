//! Vector mutation types.
//!
//! This module contains mutation type definitions for vector storage operations.
//! Following the pattern from `graph::mutation`, mutations are grouped in an enum
//! for type-safe dispatch.
//!
//! Mutations implement `MutationCodec` to marshal themselves to CF key-value pairs.
//! This keeps marshaling logic with the mutation definitions rather than in schema.

use tokio::sync::oneshot;

use crate::rocksdb::MutationCodec;
use crate::Id;

use super::distance::Distance;
use super::schema::{
    EmbeddingCode, EmbeddingSpec, EmbeddingSpecCfKey, EmbeddingSpecCfValue, EmbeddingSpecs,
    HnswLayer, VecId,
};

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
