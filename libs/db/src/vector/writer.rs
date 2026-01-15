//! Vector mutation writer module providing mutation infrastructure.
//!
//! This module follows the same pattern as graph::writer:
//! - Writer - handle for sending mutations
//! - WriterConfig - configuration
//! - Consumer - processes mutations from channel
//! - Spawn functions for creating consumers
//!
//! # Architecture
//!
//! ```text
//! ┌─────────┐     MPSC      ┌──────────┐
//! │ Writer  │───────────────│ Consumer │
//! └─────────┘  Vec<Mutation> └──────────┘
//!                                 │
//!                                 ▼
//!                           ┌──────────┐
//!                           │Processor │
//!                           └──────────┘
//!                                 │
//!                                 ▼
//!                           ┌──────────┐
//!                           │ RocksDB  │
//!                           └──────────┘
//! ```

use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::mpsc;

use super::mutation::{FlushMarker, Mutation};
use super::processor::Processor;

// ============================================================================
// WriterConfig
// ============================================================================

/// Configuration for the mutation writer.
#[derive(Debug, Clone)]
pub struct WriterConfig {
    /// Size of the MPSC channel buffer
    pub channel_buffer_size: usize,
}

impl Default for WriterConfig {
    fn default() -> Self {
        Self {
            channel_buffer_size: 1000,
        }
    }
}

// ============================================================================
// Writer
// ============================================================================

/// Handle for sending vector mutations to be processed asynchronously.
///
/// The Writer sends mutations through an MPSC channel as `Vec<Mutation>` to enable
/// efficient transaction batching in downstream processors.
///
/// # Example
///
/// ```rust,ignore
/// use motlie_db::vector::{Writer, InsertVector, Mutation};
///
/// // Create writer and consumer
/// let (writer, receiver) = create_writer(WriterConfig::default());
///
/// // Send a vector mutation
/// let mutation = InsertVector::new(embedding_code, id, vec![1.0, 2.0, 3.0]);
/// writer.send(vec![mutation.into()]).await?;
///
/// // Flush to ensure commit
/// writer.flush().await?;
/// ```
#[derive(Clone)]
pub struct Writer {
    sender: mpsc::Sender<Vec<Mutation>>,
}

impl std::fmt::Debug for Writer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Writer")
            .field("sender", &"<mpsc::Sender>")
            .finish()
    }
}

impl Writer {
    /// Create a new Writer with the given sender.
    pub fn new(sender: mpsc::Sender<Vec<Mutation>>) -> Self {
        Self { sender }
    }

    /// Send a batch of mutations to be processed asynchronously.
    ///
    /// This method returns immediately after enqueueing the mutations.
    /// Use `flush()` to wait for mutations to be committed, or use
    /// `send_sync()` to send and wait in one call.
    pub async fn send(&self, mutations: Vec<Mutation>) -> Result<()> {
        self.sender
            .send(mutations)
            .await
            .context("Failed to send mutations to writer queue")
    }

    /// Flush all pending mutations and wait for commit.
    ///
    /// Returns when all mutations sent before this call are committed
    /// to RocksDB and visible to readers.
    pub async fn flush(&self) -> Result<()> {
        let (marker, rx) = FlushMarker::new();

        // Send flush marker through the same channel as mutations
        self.sender
            .send(vec![Mutation::Flush(marker)])
            .await
            .context("Failed to send flush marker - channel closed")?;

        // Wait for consumer to process it
        rx.await
            .context("Flush failed - consumer dropped completion channel")?;

        Ok(())
    }

    /// Send mutations and wait for commit.
    ///
    /// This is a convenience method equivalent to `send()` followed by `flush()`.
    /// Returns when all mutations are visible to readers.
    pub async fn send_sync(&self, mutations: Vec<Mutation>) -> Result<()> {
        if mutations.is_empty() {
            return Ok(());
        }
        self.send(mutations).await?;
        self.flush().await
    }

    /// Check if the writer is still active (receiver hasn't been dropped)
    pub fn is_closed(&self) -> bool {
        self.sender.is_closed()
    }
}

/// Create a new mutation writer and receiver pair.
pub fn create_writer(config: WriterConfig) -> (Writer, mpsc::Receiver<Vec<Mutation>>) {
    let (sender, receiver) = mpsc::channel(config.channel_buffer_size);
    let writer = Writer::new(sender);
    (writer, receiver)
}

// ============================================================================
// Consumer
// ============================================================================

/// Consumer that processes vector mutations from a channel.
///
/// The consumer receives mutation batches and delegates to a Processor
/// for database operations.
pub struct Consumer {
    receiver: mpsc::Receiver<Vec<Mutation>>,
    config: WriterConfig,
    processor: Arc<Processor>,
}

impl Consumer {
    /// Create a new Consumer.
    pub fn new(
        receiver: mpsc::Receiver<Vec<Mutation>>,
        config: WriterConfig,
        processor: Arc<Processor>,
    ) -> Self {
        Self {
            receiver,
            config,
            processor,
        }
    }

    /// Process mutations continuously until the channel is closed.
    #[tracing::instrument(skip(self), name = "vector_mutation_consumer")]
    pub async fn run(mut self) -> Result<()> {
        tracing::info!(config = ?self.config, "Starting vector mutation consumer");

        loop {
            match self.receiver.recv().await {
                Some(mutations) => {
                    self.process_batch(mutations)
                        .await
                        .with_context(|| "Failed to process vector mutations")?;
                }
                None => {
                    // Channel closed
                    tracing::info!("Vector mutation consumer shutting down - channel closed");
                    return Ok(());
                }
            }
        }
    }

    /// Process a batch of mutations.
    #[tracing::instrument(skip(self, mutations), fields(batch_size = mutations.len()))]
    async fn process_batch(&self, mutations: Vec<Mutation>) -> Result<()> {
        // Log what we're processing
        for mutation in &mutations {
            match mutation {
                Mutation::AddEmbeddingSpec(args) => {
                    tracing::debug!(
                        code = args.code,
                        model = %args.model,
                        dim = args.dim,
                        "Processing AddEmbeddingSpec"
                    );
                }
                Mutation::InsertVector(args) => {
                    tracing::debug!(
                        embedding = args.embedding,
                        id = %args.id,
                        dim = args.vector.len(),
                        "Processing InsertVector"
                    );
                }
                Mutation::DeleteVector(args) => {
                    tracing::debug!(
                        embedding = args.embedding,
                        id = %args.id,
                        "Processing DeleteVector"
                    );
                }
                Mutation::InsertVectorBatch(args) => {
                    tracing::debug!(
                        embedding = args.embedding,
                        count = args.vectors.len(),
                        "Processing InsertVectorBatch"
                    );
                }
                Mutation::UpdateEdges(args) => {
                    tracing::debug!(
                        embedding = args.embedding,
                        vec_id = args.vec_id,
                        layer = args.layer,
                        "Processing UpdateEdges"
                    );
                }
                Mutation::UpdateGraphMeta(args) => {
                    tracing::debug!(
                        embedding = args.embedding,
                        "Processing UpdateGraphMeta"
                    );
                }
                Mutation::Flush(_) => {
                    tracing::debug!("Processing Flush marker");
                }
            }
        }

        // Process all mutations in the batch
        self.execute_mutations(&mutations).await?;

        // Signal completion for any flush markers
        for mutation in mutations {
            if let Mutation::Flush(marker) = mutation {
                marker.complete();
            }
        }

        Ok(())
    }

    /// Execute mutations against storage.
    ///
    /// Phase 1: Basic implementation - execute mutations synchronously
    /// Phase 2+: Will add proper batching and transaction support
    async fn execute_mutations(&self, mutations: &[Mutation]) -> Result<()> {
        let txn_db = self.processor.storage().transaction_db()?;
        let txn = txn_db.transaction();

        for mutation in mutations {
            self.execute_single(&txn, &txn_db, mutation)?;
        }

        txn.commit()?;
        Ok(())
    }

    /// Execute a single mutation.
    fn execute_single(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        mutation: &Mutation,
    ) -> Result<()> {
        use super::schema::{
            BinaryCodeCfKey, BinaryCodeCfValue, BinaryCodes, EmbeddingSpecs, IdForward,
            IdForwardCfKey, IdForwardCfValue, IdReverse, IdReverseCfKey, IdReverseCfValue,
            VectorCfKey, VectorCfValue, Vectors,
        };
        use crate::rocksdb::{ColumnFamily, MutationCodec};

        match mutation {
            Mutation::AddEmbeddingSpec(op) => {
                // Use MutationCodec to serialize
                let (key_bytes, value_bytes) = op.to_cf_bytes()?;
                let cf = txn_db
                    .cf_handle(EmbeddingSpecs::CF_NAME)
                    .ok_or_else(|| anyhow::anyhow!("EmbeddingSpecs CF not found"))?;
                txn.put_cf(&cf, key_bytes, value_bytes)?;
            }

            Mutation::InsertVector(op) => {
                // 0. Look up embedding spec to get storage type (hard-fail if not registered)
                // This prevents silent data corruption from storage type mismatch
                let storage_type = self
                    .processor
                    .registry()
                    .get_by_code(op.embedding)
                    .ok_or_else(|| anyhow::anyhow!(
                        "Embedding {} not registered; cannot determine storage type",
                        op.embedding
                    ))?
                    .storage_type();

                // 1. Allocate vec_id
                let vec_id = {
                    let allocator = self.processor.get_or_create_allocator(op.embedding);
                    allocator.allocate()
                };

                // 2. Store Id -> VecId mapping (IdForward)
                let forward_key = IdForwardCfKey(op.embedding, op.id);
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
                let reverse_key = IdReverseCfKey(op.embedding, vec_id);
                let reverse_value = IdReverseCfValue(op.id);
                let reverse_cf = txn_db
                    .cf_handle(IdReverse::CF_NAME)
                    .ok_or_else(|| anyhow::anyhow!("IdReverse CF not found"))?;
                txn.put_cf(
                    &reverse_cf,
                    IdReverse::key_to_bytes(&reverse_key),
                    IdReverse::value_to_bytes(&reverse_value),
                )?;

                // 4. Store vector data using correct storage type (F32 or F16)
                let vec_key = VectorCfKey(op.embedding, vec_id);
                let vectors_cf = txn_db
                    .cf_handle(Vectors::CF_NAME)
                    .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;
                txn.put_cf(
                    &vectors_cf,
                    Vectors::key_to_bytes(&vec_key),
                    Vectors::value_to_bytes_typed(&op.vector, storage_type),
                )?;

                // 5. Store binary code with ADC correction (if RaBitQ enabled)
                if let Some(encoder) = self.processor.get_or_create_encoder(op.embedding) {
                    let (code, correction) = encoder.encode_with_correction(&op.vector);
                    let code_key = BinaryCodeCfKey(op.embedding, vec_id);
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

                // Note: HNSW indexing will be added in Phase 2
            }

            Mutation::DeleteVector(op) => {
                // 1. Look up vec_id from external Id
                let forward_key = IdForwardCfKey(op.embedding, op.id);
                let forward_cf = txn_db
                    .cf_handle(IdForward::CF_NAME)
                    .ok_or_else(|| anyhow::anyhow!("IdForward CF not found"))?;

                let vec_id = match txn.get_cf(&forward_cf, IdForward::key_to_bytes(&forward_key))? {
                    Some(bytes) => IdForward::value_from_bytes(&bytes)?.0,
                    None => return Ok(()), // Already deleted or never existed
                };

                // 2. Delete IdForward mapping
                txn.delete_cf(&forward_cf, IdForward::key_to_bytes(&forward_key))?;

                // 3. Delete IdReverse mapping
                let reverse_key = IdReverseCfKey(op.embedding, vec_id);
                let reverse_cf = txn_db
                    .cf_handle(IdReverse::CF_NAME)
                    .ok_or_else(|| anyhow::anyhow!("IdReverse CF not found"))?;
                txn.delete_cf(&reverse_cf, IdReverse::key_to_bytes(&reverse_key))?;

                // 4. Delete vector data
                let vec_key = VectorCfKey(op.embedding, vec_id);
                let vectors_cf = txn_db
                    .cf_handle(Vectors::CF_NAME)
                    .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;
                txn.delete_cf(&vectors_cf, Vectors::key_to_bytes(&vec_key))?;

                // 5. Delete binary code (if RaBitQ enabled)
                if self.processor.rabitq_enabled() {
                    let code_key = BinaryCodeCfKey(op.embedding, vec_id);
                    let codes_cf = txn_db
                        .cf_handle(BinaryCodes::CF_NAME)
                        .ok_or_else(|| anyhow::anyhow!("BinaryCodes CF not found"))?;
                    txn.delete_cf(&codes_cf, BinaryCodes::key_to_bytes(&code_key))?;
                }

                // 6. Return vec_id to free list
                {
                    let allocator = self.processor.get_or_create_allocator(op.embedding);
                    allocator.free(vec_id);
                }

                // Note: HNSW edge cleanup will be added in Phase 2
            }

            Mutation::InsertVectorBatch(op) => {
                // Process each vector in the batch
                for (id, vector) in &op.vectors {
                    let single = super::mutation::InsertVector {
                        embedding: op.embedding,
                        id: *id,
                        vector: vector.clone(),
                        immediate_index: op.immediate_index,
                    };
                    self.execute_single(txn, txn_db, &Mutation::InsertVector(single))?;
                }
            }

            Mutation::UpdateEdges(_) => {
                // Phase 2: HNSW graph updates
                tracing::warn!("UpdateEdges not implemented in Phase 1");
            }

            Mutation::UpdateGraphMeta(_) => {
                // Phase 2: HNSW graph metadata
                tracing::warn!("UpdateGraphMeta not implemented in Phase 1");
            }

            Mutation::Flush(_) => {
                // No-op for storage - flush marker is handled after commit
            }
        }

        Ok(())
    }
}

/// Spawn a mutation consumer as a tokio task.
///
/// Returns a JoinHandle that resolves when the consumer completes.
pub fn spawn_consumer(consumer: Consumer) -> tokio::task::JoinHandle<Result<()>> {
    tokio::spawn(async move { consumer.run().await })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_writer_creation() {
        let (writer, _receiver) = create_writer(WriterConfig::default());
        assert!(!writer.is_closed());
    }

    #[tokio::test]
    async fn test_writer_channel_closed() {
        let (writer, receiver) = create_writer(WriterConfig::default());
        drop(receiver);
        assert!(writer.is_closed());
    }

    #[tokio::test]
    async fn test_send_mutations() {
        let (writer, mut receiver) = create_writer(WriterConfig::default());

        // Send a mutation
        let id = crate::Id::new();
        let mutation = super::super::mutation::InsertVector::new(1, id, vec![1.0, 2.0, 3.0]);
        writer.send(vec![mutation.into()]).await.unwrap();

        // Should receive it
        let received = receiver.recv().await.unwrap();
        assert_eq!(received.len(), 1);
    }

    #[tokio::test]
    async fn test_writer_config() {
        let config = WriterConfig::default();
        assert_eq!(config.channel_buffer_size, 1000);

        let custom = WriterConfig {
            channel_buffer_size: 500,
        };
        assert_eq!(custom.channel_buffer_size, 500);
    }
}
