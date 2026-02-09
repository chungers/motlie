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

use std::sync::Mutex;

use anyhow::Result;
use tokio::sync::oneshot;

use crate::request::{ReplyEnvelope, RequestMeta};
use crate::rocksdb::MutationCodec;

use super::distance::Distance;
use super::embedding::Embedding;
use super::processor::Processor;
use super::schema::{
    EmbeddingCode, EmbeddingSpec, EmbeddingSpecCfKey, EmbeddingSpecCfValue, EmbeddingSpecs,
    ExternalKey,
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
#[derive(Debug, Clone)]
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
// MutationResult / MutationOutcome
// ============================================================================

#[derive(Debug)]
pub enum MutationResult {
    AddEmbeddingSpec,
    InsertVector(super::ops::InsertResult),
    DeleteVector(super::ops::DeleteResult),
    InsertVectorBatch(super::ops::InsertBatchResult),
    Flush,
}

#[derive(Debug)]
pub struct MutationOutcome {
    pub result: MutationResult,
    pub cache_update: Option<MutationCacheUpdate>,
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
// RequestMeta for mutation types
// ============================================================================

macro_rules! impl_request_meta {
    ($ty:ty, $reply:ty, $kind:expr) => {
        impl RequestMeta for $ty {
            type Reply = ReplyEnvelope<$reply>;
            type Options = ();

            fn request_kind(&self) -> &'static str {
                $kind
            }
        }
    };
}

impl_request_meta!(AddEmbeddingSpec, (), "add_embedding_spec");
impl_request_meta!(InsertVector, super::ops::InsertResult, "insert_vector");
impl_request_meta!(DeleteVector, super::ops::DeleteResult, "delete_vector");
impl_request_meta!(InsertVectorBatch, super::ops::InsertBatchResult, "insert_vector_batch");

// ============================================================================
// MutationCodec Implementations
// ============================================================================

impl MutationCodec for AddEmbeddingSpec {
    type Cf = EmbeddingSpecs;

    fn to_record(&self) -> (EmbeddingSpecCfKey, EmbeddingSpecCfValue) {
        let key = EmbeddingSpecCfKey(self.code);
        let value = EmbeddingSpecCfValue(EmbeddingSpec {
            code: self.code,
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
    /// External key identifying the source entity
    pub external_key: ExternalKey,
    /// Vector data (must match embedding dimension)
    pub vector: Vec<f32>,
    /// Whether to index immediately or queue for async processing
    pub immediate_index: bool,
}

impl InsertVector {
    /// Create a new InsertVector mutation with an ExternalKey.
    ///
    /// # Arguments
    /// * `embedding` - Reference to embedding space (code extracted internally)
    /// * `external_key` - External key identifying the source entity
    /// * `vector` - Vector data (must match embedding dimension)
    pub fn new(embedding: &Embedding, external_key: ExternalKey, vector: Vec<f32>) -> Self {
        Self {
            embedding: embedding.code(),
            external_key,
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
/// 1. Looks up the vec_id from the ExternalKey
/// 2. Removes the ExternalKey ↔ vec_id mappings
/// 3. Removes the vector data
/// 4. Removes vector metadata
/// 5. Returns the vec_id to the free list
/// 6. Marks edges for cleanup (lazy deletion)
#[derive(Debug, Clone)]
pub struct DeleteVector {
    /// Embedding space
    pub embedding: EmbeddingCode,
    /// External key identifying the source entity
    pub external_key: ExternalKey,
}

impl DeleteVector {
    /// Create a new DeleteVector mutation with an ExternalKey.
    ///
    /// # Arguments
    /// * `embedding` - Reference to embedding space (code extracted internally)
    /// * `external_key` - External key identifying the entity to delete
    pub fn new(embedding: &Embedding, external_key: ExternalKey) -> Self {
        Self {
            embedding: embedding.code(),
            external_key,
        }
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
    /// Batch of (external_key, vector) pairs
    pub vectors: Vec<(ExternalKey, Vec<f32>)>,
    /// Whether to index immediately
    pub immediate_index: bool,
}

impl InsertVectorBatch {
    /// Create a new batch insert mutation with ExternalKeys.
    ///
    /// # Arguments
    /// * `embedding` - Reference to embedding space (code extracted internally)
    /// * `vectors` - Batch of (external_key, vector) pairs
    pub fn new(embedding: &Embedding, vectors: Vec<(ExternalKey, Vec<f32>)>) -> Self {
        Self {
            embedding: embedding.code(),
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
/// Marker for flush synchronization.
///
/// Contains a oneshot sender that signals when the flush completes.
/// Uses `Mutex<Option<...>>` to allow taking ownership from a shared reference,
/// since `process_batch` may receive `&[Mutation]`.
///
/// When a FlushMarker is processed by the consumer, it signals the
/// oneshot channel to indicate that all prior mutations have been committed.
/// This enables the `Writer::flush()` and `Writer::send_sync()` methods.
pub struct FlushMarker {
    completion: Mutex<Option<oneshot::Sender<()>>>,
}

impl FlushMarker {
    /// Create a new flush marker with completion channel.
    pub fn new(completion: oneshot::Sender<()>) -> Self {
        Self {
            completion: Mutex::new(Some(completion)),
        }
    }

    /// Take the completion sender (can only be called once).
    ///
    /// Returns `None` if already taken or if the mutex is poisoned.
    pub fn take_completion(&self) -> Option<oneshot::Sender<()>> {
        self.completion.lock().ok()?.take()
    }
}

impl std::fmt::Debug for FlushMarker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let has_completion = self
            .completion
            .lock()
            .map(|guard| guard.is_some())
            .unwrap_or(false);
        f.debug_struct("FlushMarker")
            .field("has_completion", &has_completion)
            .finish()
    }
}

// FlushMarker cannot be Clone since oneshot::Sender is not Clone
// We implement a manual Clone that creates an "empty" marker
impl Clone for FlushMarker {
    fn clone(&self) -> Self {
        // Cloning a FlushMarker creates an empty one (no completion channel)
        // This is intentional - only the original can signal completion
        Self {
            completion: Mutex::new(None),
        }
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
    /// Returns the mutation result and any cache update (to be applied after commit).
    pub(crate) fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        processor: &Processor,
    ) -> Result<MutationOutcome> {
        match self {
            Mutation::AddEmbeddingSpec(op) => op.execute(txn, txn_db, processor),
            Mutation::InsertVector(op) => op.execute(txn, txn_db, processor),
            Mutation::DeleteVector(op) => op.execute(txn, txn_db, processor),
            Mutation::InsertVectorBatch(op) => op.execute(txn, txn_db, processor),
            Mutation::Flush(_) => Ok(MutationOutcome {
                result: MutationResult::Flush,
                cache_update: None,
            }),
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
    ) -> Result<MutationOutcome> {
        // Delegate to shared ops helper (also updates in-memory registry)
        super::ops::embedding::spec(txn, txn_db, processor, self)?;
        Ok(MutationOutcome {
            result: MutationResult::AddEmbeddingSpec,
            cache_update: None,
        })
    }
}

impl MutationExecutor for InsertVector {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        processor: &Processor,
    ) -> Result<MutationOutcome> {
        // Delegate to shared ops helper (includes all validation)
        let result = super::ops::insert::vector(
            txn,
            txn_db,
            processor,
            self.embedding,
            self.external_key.clone(),
            &self.vector,
            self.immediate_index,
        )?;

        // Return combined cache update for both nav and code caches
        Ok(MutationOutcome {
            result: MutationResult::InsertVector(result.clone()),
            cache_update: MutationCacheUpdate::from_insert(
                self.embedding,
                result.vec_id,
                result.nav_cache_update,
                result.code_cache_update,
            ),
        })
    }
}

impl MutationExecutor for DeleteVector {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        processor: &Processor,
    ) -> Result<MutationOutcome> {
        // Delegate to shared ops helper (handles soft-delete when HNSW enabled)
        let result =
            super::ops::delete::vector(txn, txn_db, processor, self.embedding, self.external_key.clone())?;

        // Log soft-delete if it occurred
        if result.is_soft_delete() {
            tracing::debug!(
                vec_id = ?result.vec_id(),
                "DeleteVector: soft-delete (HNSW enabled, vector data retained)"
            );
        }

        Ok(MutationOutcome {
            result: MutationResult::DeleteVector(result),
            cache_update: None,
        })
    }
}

impl MutationExecutor for InsertVectorBatch {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        processor: &Processor,
    ) -> Result<MutationOutcome> {
        // Delegate to shared ops helper (includes batch validation)
        let result = super::ops::insert::batch(
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
        Ok(MutationOutcome {
            result: MutationResult::InsertVectorBatch(result.clone()),
            cache_update: MutationCacheUpdate::from_batch(
                self.embedding,
                &result.vec_ids,
                result.nav_cache_updates,
                result.code_cache_updates,
            ),
        })
    }
}

// ============================================================================
// Runnable Implementations
// ============================================================================

/// Trait for mutations to execute themselves against a Writer.
///
/// This enables the ergonomic pattern:
/// ```rust,ignore
/// InsertVector::new(embedding, id, vec).run(&writer).await?;
/// ```
///
/// Note: This sends the mutation asynchronously without waiting for commit.
/// Use `Writer::flush()` after `run()` if you need confirmation, or use
/// `Writer::send_sync()` directly for send-and-wait semantics.
#[async_trait::async_trait]
pub trait Runnable {
    /// Execute this mutation against the writer.
    async fn run(self, writer: &super::writer::Writer) -> Result<()>;
}

#[async_trait::async_trait]
pub trait RunnableWithResult<R> {
    async fn run_with_result(
        self,
        writer: &super::writer::Writer,
    ) -> Result<ReplyEnvelope<R>>;
}

pub trait MutationReply: Into<Mutation> + Send {
    type Reply;
    fn from_result(result: MutationResult) -> Result<Self::Reply>;
}

#[async_trait::async_trait]
impl Runnable for InsertVector {
    async fn run(self, writer: &super::writer::Writer) -> Result<()> {
        writer.send(vec![Mutation::InsertVector(self)]).await
    }
}

#[async_trait::async_trait]
impl Runnable for InsertVectorBatch {
    async fn run(self, writer: &super::writer::Writer) -> Result<()> {
        writer.send(vec![Mutation::InsertVectorBatch(self)]).await
    }
}

#[async_trait::async_trait]
impl Runnable for DeleteVector {
    async fn run(self, writer: &super::writer::Writer) -> Result<()> {
        writer.send(vec![Mutation::DeleteVector(self)]).await
    }
}

#[async_trait::async_trait]
impl Runnable for AddEmbeddingSpec {
    async fn run(self, writer: &super::writer::Writer) -> Result<()> {
        writer.send(vec![Mutation::AddEmbeddingSpec(self)]).await
    }
}

// ============================================================================
// Typed replies for mutations
// ============================================================================

#[async_trait::async_trait]
impl<M> RunnableWithResult<M::Reply> for M
where
    M: MutationReply,
{
    async fn run_with_result(
        self,
        writer: &super::writer::Writer,
    ) -> Result<ReplyEnvelope<M::Reply>> {
        let reply = writer.send_with_result(vec![self.into()]).await?;
        if reply.payload.len() != 1 {
            return Err(anyhow::anyhow!(
                "Expected exactly one mutation result, got {}",
                reply.payload.len()
            ));
        }
        let mut results = reply.payload;
        let result = results.remove(0);
        let payload = M::from_result(result)?;
        Ok(ReplyEnvelope::new(
            reply.request_id,
            reply.elapsed_time,
            payload,
        ))
    }
}

impl MutationReply for AddEmbeddingSpec {
    type Reply = ();

    fn from_result(result: MutationResult) -> Result<Self::Reply> {
        match result {
            MutationResult::AddEmbeddingSpec => Ok(()),
            other => Err(anyhow::anyhow!("Unexpected mutation result: {:?}", other)),
        }
    }
}

impl MutationReply for InsertVector {
    type Reply = super::ops::InsertResult;

    fn from_result(result: MutationResult) -> Result<Self::Reply> {
        match result {
            MutationResult::InsertVector(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected mutation result: {:?}", other)),
        }
    }
}

impl MutationReply for DeleteVector {
    type Reply = super::ops::DeleteResult;

    fn from_result(result: MutationResult) -> Result<Self::Reply> {
        match result {
            MutationResult::DeleteVector(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected mutation result: {:?}", other)),
        }
    }
}

impl MutationReply for InsertVectorBatch {
    type Reply = super::ops::InsertBatchResult;

    fn from_result(result: MutationResult) -> Result<Self::Reply> {
        match result {
            MutationResult::InsertVectorBatch(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected mutation result: {:?}", other)),
        }
    }
}

impl RequestMeta for Vec<Mutation> {
    type Reply = ReplyEnvelope<Vec<MutationResult>>;
    type Options = ();

    fn request_kind(&self) -> &'static str {
        "vector_mutation_batch"
    }
}

#[async_trait::async_trait]
impl Runnable for Vec<Mutation> {
    async fn run(self, writer: &super::writer::Writer) -> Result<()> {
        if self.is_empty() {
            return Ok(());
        }
        writer.send(self).await
    }
}

/// Extension trait for batch-only helpers on Vec<Mutation>.
#[async_trait::async_trait]
pub trait MutationBatchExt {
    async fn run_with_result(
        self,
        writer: &super::writer::Writer,
    ) -> Result<ReplyEnvelope<Vec<MutationResult>>>;
}

#[async_trait::async_trait]
impl MutationBatchExt for Vec<Mutation> {
    async fn run_with_result(
        self,
        writer: &super::writer::Writer,
    ) -> Result<ReplyEnvelope<Vec<MutationResult>>> {
        writer.send_with_result(self).await
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::schema::{EmbeddingSpec, VectorElementType};
    use crate::Id;
    use std::sync::Arc;

    /// Create a test embedding for unit tests.
    fn test_embedding(code: EmbeddingCode) -> Embedding {
        let spec = Arc::new(EmbeddingSpec {
            code,
            model: "test".to_string(),
            dim: 128,
            distance: Distance::Cosine,
            storage_type: VectorElementType::F32,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            rabitq_bits: 1,
            rabitq_seed: 42,
        });
        Embedding::new(spec, None)
    }

    #[test]
    fn test_insert_vector_builder() {
        let embedding = test_embedding(1);
        let mutation = InsertVector::new(&embedding, ExternalKey::NodeId(Id::new()), vec![1.0, 2.0, 3.0]);
        assert_eq!(mutation.embedding, 1);
        assert!(!mutation.immediate_index);

        let mutation = mutation.immediate();
        assert!(mutation.immediate_index);
    }

    #[test]
    fn test_delete_vector() {
        let embedding = test_embedding(42);
        let id = Id::new();
        let mutation = DeleteVector::new(&embedding, ExternalKey::NodeId(id));
        assert_eq!(mutation.embedding, 42);
        assert_eq!(mutation.external_key, ExternalKey::NodeId(id));
    }

    #[test]
    fn test_insert_batch() {
        let embedding = test_embedding(1);
        let vectors: Vec<(ExternalKey, Vec<f32>)> = vec![
            (ExternalKey::NodeId(Id::new()), vec![1.0, 2.0]),
            (ExternalKey::NodeId(Id::new()), vec![3.0, 4.0]),
        ];
        let mutation = InsertVectorBatch::new(&embedding, vectors.clone());
        assert_eq!(mutation.embedding, 1);
        assert_eq!(mutation.vectors.len(), 2);
        assert!(!mutation.immediate_index);
    }

    #[test]
    fn test_mutation_from_conversions() {
        let embedding = test_embedding(1);
        let insert = InsertVector::new(&embedding, ExternalKey::NodeId(Id::new()), vec![1.0]);
        let _: Mutation = insert.into();

        let delete = DeleteVector::new(&embedding, ExternalKey::NodeId(Id::new()));
        let _: Mutation = delete.into();

        let batch = InsertVectorBatch::new(&embedding, vec![(ExternalKey::NodeId(Id::new()), vec![1.0])]);
        let _: Mutation = batch.into();
    }

    #[test]
    fn test_flush_marker() {
        let (tx, mut rx) = tokio::sync::oneshot::channel();
        let marker = FlushMarker::new(tx);

        // Take completion and signal
        let completion = marker.take_completion().expect("should have completion");
        completion.send(()).expect("should send");

        // Receiver gets the signal
        assert!(rx.try_recv().is_ok());

        // Second take returns None
        assert!(marker.take_completion().is_none());
    }

    #[test]
    fn test_flush_marker_clone() {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        let marker = FlushMarker::new(tx);

        // Clone creates empty marker
        let cloned = marker.clone();
        assert!(cloned.take_completion().is_none());

        // Original still has completion
        assert!(marker.take_completion().is_some());
    }
}
