//! Mutation writer module providing mutation infrastructure.
//!
//! This module follows the same pattern as fulltext::writer:
//! - Writer - handle for sending mutations
//! - WriterConfig - configuration
//! - Consumer - processes mutations from channel
//! - Spawn functions for creating consumers
//!
//! Also contains the mutation executor traits (MutationExecutor, Processor)
//! which define how mutations execute against the storage layer.

use anyhow::{Context, Result};
use tokio::sync::{mpsc, oneshot};

use super::mutation::{FlushMarker, Mutation};

// ============================================================================
// MutationExecutor Trait
// ============================================================================

/// Trait for mutations to execute themselves directly against storage.
///
/// This trait defines HOW to write the mutation to the database.
/// Each mutation type knows how to execute its own database write operations.
///
/// Following the same pattern as QueryExecutor for queries.
/// Note: This is synchronous because RocksDB operations are blocking.
pub trait MutationExecutor: Send + Sync {
    /// Execute this mutation directly against a RocksDB transaction.
    /// Each mutation type knows how to write itself to storage.
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()>;
}

// ============================================================================
// Processor Trait
// ============================================================================

/// Trait for processing batches of mutations atomically.
///
/// Consumers delegate to a Processor to handle the actual database operations.
/// This separation allows:
/// - Different storage backends (RocksDB, Tantivy, etc.)
/// - Multiple consumers to process the same mutations
/// - Testing with mock processors
///
/// # Example Implementation
///
/// ```rust,ignore
/// #[async_trait::async_trait]
/// impl Processor for Graph {
///     async fn process_mutations(&self, mutations: &[Mutation]) -> Result<()> {
///         // Each mutation generates its own storage operations
///         let mut operations = Vec::new();
///         for mutation in mutations {
///             operations.extend(mutation.plan()?);
///         }
///
///         // Execute all operations in a single RocksDB transaction
///         let txn = self.storage.transaction();
///         for op in operations {
///             txn.put_cf(cf, key, value)?;
///         }
///         txn.commit()?;  // Single commit for entire batch
///         Ok(())
///     }
/// }
/// ```
#[async_trait::async_trait]
pub trait Processor: Send + Sync {
    /// Process a batch of mutations atomically.
    ///
    /// # Arguments
    /// * `mutations` - Slice of mutations to process. Can be a single mutation or many.
    ///
    /// # Returns
    /// * `Ok(())` if all mutations were processed successfully
    /// * `Err(_)` if processing failed (implementations should rollback on error)
    async fn process_mutations(&self, mutations: &[Mutation]) -> Result<()>;
}

// ============================================================================
// Writer
// ============================================================================

/// Configuration for the mutation writer
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

/// Handle for sending mutations to the writer with batching support.
///
/// The `Writer` sends mutations through an MPSC channel as `Vec<Mutation>` to enable
/// efficient transaction batching in downstream processors.
///
/// # Usage
///
/// Use the new mutation API for sending mutations:
///
/// ## Single Mutations
///
/// ```rust,ignore
/// use motlie_db::{Writer, AddNode, MutationRunnable, Id, TimestampMilli};
///
/// let (writer, receiver) = create_mutation_writer(Default::default());
///
/// // Send a single mutation using .run() pattern
/// AddNode {
///     id: Id::new(),
///     name: "Alice".to_string(),
///     ts_millis: TimestampMilli::now(),
///     valid_range: None,
/// }
/// .run(&writer)
/// .await?;
/// ```
///
/// ## Batch Mutations
///
/// ```rust,ignore
/// use motlie_db::{mutations, AddNode, AddEdge, MutationRunnable};
///
/// // Send multiple mutations in a batch
/// mutations![
///     AddNode { /* ... */ },
///     AddEdge { /* ... */ },
/// ]
/// .run(&writer)
/// .await?;
/// ```
///
/// See [Mutation API Guide](../docs/mutation-api-guide.md) for complete documentation.
#[derive(Debug, Clone)]
pub struct Writer {
    sender: mpsc::Sender<Vec<Mutation>>,
}

impl Writer {
    /// Create a new MutationWriter with the given sender
    pub fn new(sender: mpsc::Sender<Vec<Mutation>>) -> Self {
        Writer { sender }
    }

    /// Send a batch of mutations to be processed asynchronously.
    ///
    /// This method returns immediately after enqueueing the mutations.
    /// Use `flush()` to wait for mutations to be committed, or use
    /// `send_sync()` to send and wait in one call.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Fire-and-forget (high throughput)
    /// writer.send(vec![AddNode { ... }.into()]).await?;
    /// writer.send(vec![AddEdge { ... }.into()]).await?;
    ///
    /// // Wait for all to be committed
    /// writer.flush().await?;
    /// ```
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
    ///
    /// # Fulltext Consistency
    ///
    /// This method only guarantees **graph (RocksDB) consistency**.
    /// Fulltext indexing continues asynchronously and may not be complete
    /// when this method returns.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Send mutations
    /// writer.send(vec![AddNode { ... }.into()]).await?;
    /// writer.send(vec![AddEdge { ... }.into()]).await?;
    ///
    /// // Wait for all to be committed
    /// writer.flush().await?;
    ///
    /// // Now safe to read
    /// let node = NodeById::new(id).run(&reader, timeout).await?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The writer channel is closed
    /// - The consumer task has panicked or been dropped
    pub async fn flush(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();

        // Send flush marker through the same channel as mutations
        self.sender
            .send(vec![Mutation::Flush(FlushMarker::new(tx))])
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
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Send and wait in one call
    /// writer.send_sync(vec![
    ///     AddNode { ... }.into(),
    ///     AddEdge { ... }.into(),
    /// ]).await?;
    ///
    /// // Immediately visible
    /// let node = NodeById::new(id).run(&reader, timeout).await?;
    /// ```
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

/// Create a new mutation writer and receiver pair
pub fn create_mutation_writer(config: WriterConfig) -> (Writer, mpsc::Receiver<Vec<Mutation>>) {
    let (sender, receiver) = mpsc::channel(config.channel_buffer_size);
    let writer = Writer::new(sender);
    (writer, receiver)
}

// ============================================================================
// Consumer
// ============================================================================

/// Generic consumer that processes mutations using a Processor
pub struct Consumer<P: Processor> {
    receiver: mpsc::Receiver<Vec<Mutation>>,
    config: WriterConfig,
    processor: P,
    /// Optional sender to forward mutations to the next consumer in the chain
    next: Option<mpsc::Sender<Vec<Mutation>>>,
}

impl<P: Processor> Consumer<P> {
    /// Create a new Consumer
    pub fn new(
        receiver: mpsc::Receiver<Vec<Mutation>>,
        config: WriterConfig,
        processor: P,
    ) -> Self {
        Self {
            receiver,
            config,
            processor,
            next: None,
        }
    }

    /// Create a new Consumer that forwards mutations to the next consumer in the chain
    pub fn with_next(
        receiver: mpsc::Receiver<Vec<Mutation>>,
        config: WriterConfig,
        processor: P,
        next: mpsc::Sender<Vec<Mutation>>,
    ) -> Self {
        Self {
            receiver,
            config,
            processor,
            next: Some(next),
        }
    }

    /// Process mutations continuously until the channel is closed
    #[tracing::instrument(skip(self), name = "mutation_consumer")]
    pub async fn run(mut self) -> Result<()> {
        tracing::info!(config = ?self.config, "Starting mutation consumer");

        loop {
            // Wait for the next batch of mutations
            match self.receiver.recv().await {
                Some(mutations) => {
                    // Process the batch immediately
                    self.process_batch(&mutations)
                        .await
                        .with_context(|| format!("Failed to process mutations: {:?}", mutations))?;
                }
                None => {
                    // Channel closed
                    tracing::info!("Mutation consumer shutting down - channel closed");
                    return Ok(());
                }
            }
        }
    }

    /// Process a batch of mutations
    #[tracing::instrument(skip(self, mutations), fields(batch_size = mutations.len()))]
    async fn process_batch(&self, mutations: &[Mutation]) -> Result<()> {
        // Log what we're processing
        for mutation in mutations {
            match mutation {
                Mutation::AddNode(args) => {
                    tracing::debug!(id = %args.id, name = %args.name, "Processing AddNode");
                }
                Mutation::AddEdge(args) => {
                    tracing::debug!(
                        source = %args.source_node_id,
                        target = %args.target_node_id,
                        name = %args.name,
                        "Processing AddEdge"
                    );
                }
                Mutation::AddNodeFragment(args) => {
                    tracing::debug!(
                        id = %args.id,
                        body_len = args.content.0.len(),
                        "Processing AddNodeFragment"
                    );
                }
                Mutation::AddEdgeFragment(args) => {
                    tracing::debug!(
                        src = %args.src_id,
                        dst = %args.dst_id,
                        name = %args.edge_name,
                        body_len = args.content.0.len(),
                        "Processing AddEdgeFragment"
                    );
                }
                Mutation::UpdateNodeValidSinceUntil(args) => {
                    tracing::debug!(
                        id = %args.id,
                        reason = %args.reason,
                        "Processing UpdateNodeValidSinceUntil"
                    );
                }
                Mutation::UpdateEdgeValidSinceUntil(args) => {
                    tracing::debug!(
                        src = %args.src_id,
                        dst = %args.dst_id,
                        name = %args.name,
                        reason = %args.reason,
                        "Processing UpdateEdgeValidSinceUntil"
                    );
                }
                Mutation::UpdateEdgeWeight(args) => {
                    tracing::debug!(
                        src = %args.src_id,
                        dst = %args.dst_id,
                        name = %args.name,
                        weight = args.weight,
                        "Processing UpdateEdgeWeight"
                    );
                }
                Mutation::Flush(_) => {
                    tracing::debug!("Processing Flush marker");
                }
            }
        }

        // Process all mutations in a single call
        // (Flush markers are no-ops in storage but are included for ordering)
        self.processor.process_mutations(mutations).await?;

        // After successful commit, signal completion for any flush markers
        // This guarantees that all mutations before the flush are now visible to readers
        for mutation in mutations {
            if let Mutation::Flush(marker) = mutation {
                if let Some(completion) = marker.take_completion() {
                    // Signal that flush is complete - ignore send errors
                    // (receiver may have been dropped if caller timed out)
                    let _ = completion.send(());
                    tracing::debug!("Flush completion signaled");
                }
            }
        }

        // Forward the batch to the next consumer in the chain if configured
        // This is a best-effort send - if the buffer is full, we log and continue
        // Note: We filter out Flush markers when forwarding - they are local to this consumer
        if let Some(sender) = &self.next {
            let non_flush_mutations: Vec<_> = mutations
                .iter()
                .filter(|m| !m.is_flush())
                .cloned()
                .collect();

            if !non_flush_mutations.is_empty() {
                if let Err(e) = sender.try_send(non_flush_mutations) {
                    tracing::warn!(
                        err = %e,
                        count = mutations.len(),
                        "[BUFFER FULL] Next consumer busy, dropping mutations"
                    );
                }
            }
        }

        Ok(())
    }
}

/// Spawn a mutation consumer as a background task.
///
/// This is the generic helper used internally and by the fulltext module.
pub fn spawn_consumer<P: Processor + 'static>(
    consumer: Consumer<P>,
) -> tokio::task::JoinHandle<Result<()>> {
    tokio::spawn(async move { consumer.run().await })
}

// ============================================================================
// Graph-specific Consumer Functions
// ============================================================================

use std::path::Path;
use std::sync::Arc;
use tokio::task::JoinHandle;

use super::{Graph, Storage};

/// Create a new graph mutation consumer
pub fn create_mutation_consumer(
    receiver: mpsc::Receiver<Vec<Mutation>>,
    config: WriterConfig,
    db_path: &Path,
) -> Consumer<Graph> {
    let mut storage = Storage::readwrite(db_path);
    storage.ready().expect("Failed to ready storage");
    let storage = Arc::new(storage);
    let processor = Graph::new(storage);
    Consumer::new(receiver, config, processor)
}

/// Create a new graph mutation consumer that chains to another processor
pub fn create_mutation_consumer_with_next(
    receiver: mpsc::Receiver<Vec<Mutation>>,
    config: WriterConfig,
    db_path: &Path,
    next: mpsc::Sender<Vec<Mutation>>,
) -> Consumer<Graph> {
    let mut storage = Storage::readwrite(db_path);
    storage.ready().expect("Failed to ready storage");
    let storage = Arc::new(storage);
    let processor = Graph::new(storage);
    Consumer::with_next(receiver, config, processor, next)
}

/// Spawn the graph mutation consumer as a background task
pub fn spawn_mutation_consumer(
    receiver: mpsc::Receiver<Vec<Mutation>>,
    config: WriterConfig,
    db_path: &Path,
) -> JoinHandle<Result<()>> {
    let consumer = create_mutation_consumer(receiver, config, db_path);
    spawn_consumer(consumer)
}

/// Spawn the graph mutation consumer as a background task with chaining to next processor
pub fn spawn_mutation_consumer_with_next(
    receiver: mpsc::Receiver<Vec<Mutation>>,
    config: WriterConfig,
    db_path: &Path,
    next: mpsc::Sender<Vec<Mutation>>,
) -> JoinHandle<Result<()>> {
    let consumer = create_mutation_consumer_with_next(receiver, config, db_path, next);
    spawn_consumer(consumer)
}

/// Spawn a graph mutation consumer using an existing Graph instance
///
/// This allows using a shared Storage/TransactionDB instance for writes.
/// Use this when you want the writer to share the same TransactionDB as readers.
///
/// # Arguments
/// * `receiver` - Channel to receive mutation batches from
/// * `config` - Writer configuration
/// * `graph` - Shared Graph instance (wrapping the Storage/TransactionDB)
///
/// # Returns
/// A JoinHandle for the spawned consumer task
pub fn spawn_mutation_consumer_with_graph(
    receiver: mpsc::Receiver<Vec<Mutation>>,
    config: WriterConfig,
    graph: Arc<Graph>,
) -> JoinHandle<Result<()>> {
    let consumer = Consumer::new(receiver, config, (*graph).clone());
    spawn_consumer(consumer)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::mutation::{AddEdge, AddNode, AddNodeFragment, UpdateEdgeValidSinceUntil};
    use crate::writer::Runnable as MutRunnable;
    use super::super::schema::{EdgeSummary, NodeSummary};
    use crate::{DataUrl, Id, TimestampMilli};
    use tokio::time::Duration;

    #[tokio::test]
    async fn test_writer_closed_detection() {
        let config = WriterConfig::default();
        let (writer, receiver) = create_mutation_writer(config);

        assert!(!writer.is_closed());

        // Drop receiver to close channel
        drop(receiver);

        // Writer should detect channel is closed
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(writer.is_closed());
    }

    #[tokio::test]
    async fn test_writer_send_operations() {
        let config = WriterConfig::default();
        let (writer, _receiver) = create_mutation_writer(config);

        // Test that all send operations work with new API
        let node_args = AddNode {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "test_node".to_string(),
            valid_range: None,
            summary: NodeSummary::from_text("test node summary"),
        };

        let edge_args = AddEdge {
            source_node_id: Id::new(),
            target_node_id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "test_edge".to_string(),
            summary: EdgeSummary::from_text("edge summary"),
            weight: Some(1.0),
            valid_range: None,
        };

        let fragment_args = AddNodeFragment {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            content: DataUrl::from_text("test fragment"),
            valid_range: None,
        };

        let src_id = Id::new();
        let dst_id = Id::new();
        let invalidate_args = UpdateEdgeValidSinceUntil {
            src_id,
            dst_id,
            name: "test_edge".to_string(),
            temporal_range: crate::TemporalRange(None, None),
            reason: "test reason".to_string(),
        };

        // Test new mutation API
        node_args.run(&writer).await.unwrap();
        edge_args.run(&writer).await.unwrap();
        fragment_args.run(&writer).await.unwrap();
        invalidate_args.run(&writer).await.unwrap();
    }
}
