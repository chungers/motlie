//! Vector mutation writer module providing mutation infrastructure.
//!
//! This module follows the same pattern as graph::writer:
//! - Writer - handle for sending mutations
//! - WriterConfig - configuration
//! - Consumer - processes mutations from channel
//! - Spawn functions for creating consumers
//! - MutationExecutor trait for mutation dispatch
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
use tokio::sync::{mpsc, oneshot};

use super::hnsw::CacheUpdate;
use super::mutation::{FlushMarker, Mutation};
use super::processor::Processor;
use super::schema::{AdcCorrection, EmbeddingCode, VecId};

// ============================================================================
// MutationCacheUpdate
// ============================================================================

/// Combined cache update for mutation execution.
///
/// Mutations can produce updates for both the navigation cache (HNSW) and
/// the binary code cache (RaBitQ). This enum allows returning both types
/// from a single mutation execution.
#[derive(Debug)]
pub enum MutationCacheUpdate {
    /// Navigation cache update only (HNSW layer info)
    Nav(CacheUpdate),
    /// Binary code cache update only
    Code {
        embedding: EmbeddingCode,
        vec_id: VecId,
        code: Vec<u8>,
        correction: AdcCorrection,
    },
    /// Both navigation and code cache updates
    Both {
        nav: CacheUpdate,
        embedding: EmbeddingCode,
        vec_id: VecId,
        code: Vec<u8>,
        correction: AdcCorrection,
    },
    /// Batch of cache updates (for InsertVectorBatch)
    Batch(Vec<MutationCacheUpdate>),
}

impl MutationCacheUpdate {
    /// Create from an InsertResult's cache updates.
    pub fn from_insert(
        embedding: EmbeddingCode,
        vec_id: VecId,
        nav: Option<CacheUpdate>,
        code: Option<(Vec<u8>, AdcCorrection)>,
    ) -> Option<Self> {
        match (nav, code) {
            (Some(nav), Some((code, correction))) => Some(Self::Both {
                nav,
                embedding,
                vec_id,
                code,
                correction,
            }),
            (Some(nav), None) => Some(Self::Nav(nav)),
            (None, Some((code, correction))) => Some(Self::Code {
                embedding,
                vec_id,
                code,
                correction,
            }),
            (None, None) => None,
        }
    }

    /// Create from an InsertBatchResult's cache updates.
    ///
    /// Collects all nav and code cache updates from the batch result into
    /// a single `Batch` variant for efficient application.
    pub fn from_batch(
        embedding: EmbeddingCode,
        vec_ids: &[VecId],
        nav_updates: Vec<CacheUpdate>,
        code_updates: Vec<(VecId, Vec<u8>, AdcCorrection)>,
    ) -> Option<Self> {
        let mut updates = Vec::new();

        // Add nav-only updates for vectors that have nav but no code
        for nav in nav_updates {
            // Find matching code update by vec_id
            let vec_id = nav.vec_id;
            if let Some(pos) = code_updates.iter().position(|(id, _, _)| *id == vec_id) {
                let (_, code, correction) = code_updates[pos].clone();
                updates.push(Self::Both {
                    nav,
                    embedding,
                    vec_id,
                    code,
                    correction,
                });
            } else {
                updates.push(Self::Nav(nav));
            }
        }

        // Add code-only updates for vectors without nav updates
        let nav_vec_ids: std::collections::HashSet<_> =
            updates.iter().filter_map(|u| match u {
                Self::Nav(n) => Some(n.vec_id),
                Self::Both { vec_id, .. } => Some(*vec_id),
                _ => None,
            }).collect();

        for (vec_id, code, correction) in code_updates {
            if !nav_vec_ids.contains(&vec_id) {
                updates.push(Self::Code {
                    embedding,
                    vec_id,
                    code,
                    correction,
                });
            }
        }

        if updates.is_empty() {
            None
        } else if updates.len() == 1 {
            Some(updates.remove(0))
        } else {
            Some(Self::Batch(updates))
        }
    }

    /// Apply the cache update to the processor caches.
    pub fn apply(
        self,
        nav_cache: &super::cache::NavigationCache,
        code_cache: &super::cache::BinaryCodeCache,
    ) {
        match self {
            Self::Nav(update) => update.apply(nav_cache),
            Self::Code {
                embedding,
                vec_id,
                code,
                correction,
            } => {
                code_cache.put(embedding, vec_id, code, correction);
            }
            Self::Both {
                nav,
                embedding,
                vec_id,
                code,
                correction,
            } => {
                nav.apply(nav_cache);
                code_cache.put(embedding, vec_id, code, correction);
            }
            Self::Batch(updates) => {
                for update in updates {
                    update.apply(nav_cache, code_cache);
                }
            }
        }
    }
}

// ============================================================================
// MutationExecutor Trait
// ============================================================================

/// Trait for mutations to execute themselves directly against storage.
///
/// This trait defines HOW to write the mutation to the database.
/// Each mutation type knows how to execute its own database write operations.
///
/// Following the same pattern as `graph::writer::MutationExecutor`.
///
/// Note: This is synchronous because RocksDB operations are blocking.
pub trait MutationExecutor: Send + Sync {
    /// Execute this mutation directly against a RocksDB transaction.
    ///
    /// Returns an optional MutationCacheUpdate if caching is needed.
    /// The cache update should be applied AFTER transaction commit.
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        processor: &Processor,
    ) -> Result<Option<MutationCacheUpdate>>;
}

// ============================================================================
// Processor Trait
// ============================================================================

/// Trait for processing batches of mutations atomically.
///
/// Consumers delegate to a MutationProcessor to handle the actual database operations.
/// This separation allows:
/// - Different storage backends
/// - Multiple consumers to process the same mutations
/// - Testing with mock processors
#[async_trait::async_trait]
pub trait MutationProcessor: Send + Sync {
    /// Process a batch of mutations atomically.
    async fn process_mutations(&self, mutations: &[Mutation]) -> Result<()>;
}

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
        let (tx, rx) = oneshot::channel();
        let marker = FlushMarker::new(tx);

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
                Mutation::Flush(_) => {
                    tracing::debug!("Processing Flush marker");
                }
            }
        }

        // Process all mutations in the batch
        self.execute_mutations(&mutations).await?;

        // Signal completion for any flush markers
        for mutation in &mutations {
            if let Mutation::Flush(marker) = mutation {
                if let Some(completion) = marker.take_completion() {
                    // Signal that flush is complete - ignore send errors
                    // (receiver may have been dropped if caller timed out)
                    let _ = completion.send(());
                }
            }
        }

        Ok(())
    }

    /// Execute mutations against storage.
    ///
    /// All mutations are executed within a single transaction for atomicity.
    /// Cache updates (navigation + binary code) are deferred until after commit.
    async fn execute_mutations(&self, mutations: &[Mutation]) -> Result<()> {
        let txn_db = self.processor.storage().transaction_db()?;
        let txn = txn_db.transaction();

        // Collect cache updates to apply after commit
        let mut cache_updates: Vec<MutationCacheUpdate> = Vec::new();

        for mutation in mutations {
            // Expand batch mutations into individual operations
            // This ensures we collect cache updates from each vector
            if let Mutation::InsertVectorBatch(batch) = mutation {
                for (id, vector) in &batch.vectors {
                    let single = super::mutation::InsertVector {
                        embedding: batch.embedding,
                        id: *id,
                        vector: vector.clone(),
                        immediate_index: batch.immediate_index,
                    };
                    if let Some(update) =
                        self.execute_single(&txn, &txn_db, &Mutation::InsertVector(single))?
                    {
                        cache_updates.push(update);
                    }
                }
            } else if let Some(update) = self.execute_single(&txn, &txn_db, mutation)? {
                cache_updates.push(update);
            }
        }

        // Commit transaction - all mutations are atomic
        txn.commit()?;

        // Apply cache updates ONLY after successful commit
        // This ensures cache never contains uncommitted state
        for update in cache_updates {
            update.apply(self.processor.nav_cache(), self.processor.code_cache());
        }

        Ok(())
    }

    /// Execute a single mutation.
    ///
    /// Delegates to `Mutation::execute()` which dispatches to the appropriate
    /// `MutationExecutor` implementation.
    ///
    /// Returns an optional MutationCacheUpdate for caching.
    /// The cache update should be applied AFTER transaction commit.
    fn execute_single(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        mutation: &Mutation,
    ) -> Result<Option<MutationCacheUpdate>> {
        mutation.execute(txn, txn_db, &self.processor)
    }
}

/// Spawn a mutation consumer as a tokio task.
///
/// Returns a JoinHandle that resolves when the consumer completes.
pub fn spawn_consumer(consumer: Consumer) -> tokio::task::JoinHandle<Result<()>> {
    tokio::spawn(async move { consumer.run().await })
}

/// Spawn a mutation consumer with storage and registry.
///
/// This is the recommended way to create a mutation consumer - it constructs
/// the internal Processor automatically, hiding the implementation detail.
///
/// # Arguments
///
/// * `receiver` - The mpsc receiver from `create_writer()`
/// * `config` - Writer configuration
/// * `storage` - Vector storage instance
/// * `registry` - Embedding registry (typically from `storage.cache().clone()`)
///
/// # Returns
///
/// A JoinHandle for the spawned consumer task.
///
/// # Example
///
/// ```rust,ignore
/// let storage = Arc::new(Storage::readwrite(path));
/// let registry = storage.cache().clone();
/// let (writer, receiver) = create_writer(WriterConfig::default());
///
/// let handle = spawn_mutation_consumer_with_storage(
///     receiver,
///     WriterConfig::default(),
///     storage,
///     registry,
/// );
///
/// // Send mutations via writer
/// writer.send(vec![mutation.into()]).await?;
/// ```
pub fn spawn_mutation_consumer_with_storage(
    receiver: mpsc::Receiver<Vec<Mutation>>,
    config: WriterConfig,
    storage: Arc<super::Storage>,
    registry: Arc<super::registry::EmbeddingRegistry>,
) -> tokio::task::JoinHandle<Result<()>> {
    let processor = Arc::new(Processor::new(storage, registry));
    let consumer = Consumer::new(receiver, config, processor);
    spawn_consumer(consumer)
}

/// Spawn a mutation consumer with storage and auto-created registry.
///
/// Convenience function for quick setup - creates the embedding registry
/// from storage automatically.
///
/// # Arguments
///
/// * `receiver` - The mpsc receiver from `create_writer()`
/// * `config` - Writer configuration
/// * `storage` - Vector storage instance
///
/// # Returns
///
/// A JoinHandle for the spawned consumer task.
pub fn spawn_mutation_consumer_with_storage_autoreg(
    receiver: mpsc::Receiver<Vec<Mutation>>,
    config: WriterConfig,
    storage: Arc<super::Storage>,
) -> tokio::task::JoinHandle<Result<()>> {
    let registry = storage.cache().clone();
    spawn_mutation_consumer_with_storage(receiver, config, storage, registry)
}

/// Spawn a mutation consumer with an explicit Processor.
///
/// This is a lower-level API for cases where you need direct Processor access
/// (e.g., for search operations). For simple mutation handling, prefer
/// `spawn_mutation_consumer_with_storage()`.
///
/// # Note
///
/// This function is `pub(crate)` to encourage use of the storage-based API
/// which hides the Processor abstraction.
#[allow(dead_code)] // Available for advanced use cases requiring custom processors
pub(crate) fn spawn_mutation_consumer_with_processor(
    receiver: mpsc::Receiver<Vec<Mutation>>,
    config: WriterConfig,
    processor: Arc<Processor>,
) -> tokio::task::JoinHandle<Result<()>> {
    let consumer = Consumer::new(receiver, config, processor);
    spawn_consumer(consumer)
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
        use super::super::embedding::Embedding;
        use super::super::schema::VectorElementType;
        use super::super::Distance;

        let (writer, mut receiver) = create_writer(WriterConfig::default());

        // Send a mutation
        let id = crate::Id::new();
        let embedding = Embedding::new(1, "test", 128, Distance::Cosine, VectorElementType::F32, None);
        let mutation = super::super::mutation::InsertVector::new(&embedding, id, vec![1.0, 2.0, 3.0]);
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
