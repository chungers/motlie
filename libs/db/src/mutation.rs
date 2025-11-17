use anyhow::{Context, Result};
use tokio::sync::mpsc;

use crate::{graph::StorageOperation, schema, Id, TimestampMilli, WriterConfig};

#[derive(Debug, Clone)]
pub enum Mutation {
    AddNode(AddNode),
    AddEdge(AddEdge),
    AddFragment(AddFragment),
    Invalidate(InvalidateArgs),
}

#[derive(Debug, Clone)]
pub struct AddNode {
    /// The UUID of the Node
    pub id: Id,

    /// The timestamp as number of milliseconds since the Unix epoch
    pub ts_millis: TimestampMilli,

    /// The name of the Node
    pub name: schema::NodeName,
}

#[derive(Debug, Clone)]
pub struct AddEdge {
    /// The UUID of the Node, Edge, or Fragment
    pub id: Id,

    /// The UUID of the source Node
    pub source_node_id: Id,

    /// The UUID of the target Node
    pub target_node_id: Id,

    /// The timestamp as number of milliseconds since the Unix epoch
    pub ts_millis: TimestampMilli,

    /// The name of the Edge
    pub name: schema::EdgeName,
}

#[derive(Debug, Clone)]
pub struct AddFragment {
    /// The UUID of the Node, Edge, or Fragment
    pub id: Id,

    /// The timestamp as number of milliseconds since the Unix epoch
    pub ts_millis: TimestampMilli,

    /// The body of the Fragment
    pub content: crate::DataUrl,
}

#[derive(Debug, Clone)]
pub struct InvalidateArgs {
    /// The UUID of the Node, Edge, or Fragment
    pub id: Id,

    /// The timestamp as number of milliseconds since the Unix epoch
    pub ts_millis: TimestampMilli,

    /// The reason for invalidation
    pub reason: String,
}

/// Trait for mutations to generate their own storage operations.
///
/// This trait allows each mutation type to encapsulate the logic for converting
/// itself into the storage operations needed to persist it to the database.
///
/// Following the same pattern as QueryExecutor for queries, this moves the
/// planning logic from a centralized dispatcher into the mutation types themselves.
pub trait MutationPlanner {
    /// Generate the storage operations needed to persist this mutation
    fn plan(&self) -> Result<Vec<StorageOperation>, rmp_serde::encode::Error>;
}

impl MutationPlanner for AddNode {
    fn plan(&self) -> Result<Vec<StorageOperation>, rmp_serde::encode::Error> {
        use crate::graph::{ColumnFamilyRecord, PutCf};
        use crate::schema::{NodeNames, Nodes};

        Ok(vec![
            StorageOperation::PutCf(PutCf(
                Nodes::CF_NAME,
                Nodes::create_bytes(self)?,
            )),
            StorageOperation::PutCf(PutCf(
                NodeNames::CF_NAME,
                NodeNames::create_bytes(self)?,
            )),
        ])
    }
}

impl MutationPlanner for AddEdge {
    fn plan(&self) -> Result<Vec<StorageOperation>, rmp_serde::encode::Error> {
        use crate::graph::{ColumnFamilyRecord, PutCf};
        use crate::schema::{EdgeNames, Edges, ForwardEdges, ReverseEdges};

        Ok(vec![
            StorageOperation::PutCf(PutCf(
                Edges::CF_NAME,
                Edges::create_bytes(self)?,
            )),
            StorageOperation::PutCf(PutCf(
                ForwardEdges::CF_NAME,
                ForwardEdges::create_bytes(self)?,
            )),
            StorageOperation::PutCf(PutCf(
                ReverseEdges::CF_NAME,
                ReverseEdges::create_bytes(self)?,
            )),
            StorageOperation::PutCf(PutCf(
                EdgeNames::CF_NAME,
                EdgeNames::create_bytes(self)?,
            )),
        ])
    }
}

impl MutationPlanner for AddFragment {
    fn plan(&self) -> Result<Vec<StorageOperation>, rmp_serde::encode::Error> {
        use crate::graph::{ColumnFamilyRecord, PutCf};
        use crate::schema::Fragments;

        Ok(vec![
            StorageOperation::PutCf(PutCf(
                Fragments::CF_NAME,
                Fragments::create_bytes(self)?,
            )),
        ])
    }
}

impl MutationPlanner for InvalidateArgs {
    fn plan(&self) -> Result<Vec<StorageOperation>, rmp_serde::encode::Error> {
        // TODO: Implement actual invalidation operations when invalidation is ready
        Ok(vec![])
    }
}

impl Mutation {
    /// Generate the storage operations for this mutation
    ///
    /// Each mutation type knows how to convert itself into the storage operations
    /// needed to persist it. This follows the same pattern as QueryExecutor::execute()
    /// for queries.
    pub fn plan(&self) -> Result<Vec<StorageOperation>, rmp_serde::encode::Error> {
        match self {
            Mutation::AddNode(m) => m.plan(),
            Mutation::AddEdge(m) => m.plan(),
            Mutation::AddFragment(m) => m.plan(),
            Mutation::Invalidate(m) => m.plan(),
        }
    }
}

/// Trait for processing batches of mutations.
///
/// This trait defines a single method that processes mutations in batches,
/// enabling efficient transaction batching in RocksDB and other storage backends.
///
/// # Batching Strategy
///
/// The `process_mutations` method receives a slice of mutations, which can be:
/// - A single mutation (slice of length 1) - wrapped automatically by the Writer
/// - Multiple mutations (slice of length N) - sent explicitly as a batch
///
/// Implementations should process all mutations atomically when possible.
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
    pub async fn run(mut self) -> Result<()> {
        log::info!("Starting mutation consumer with config: {:?}", self.config);

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
                    log::info!("Mutation consumer shutting down - channel closed");
                    return Ok(());
                }
            }
        }
    }

    /// Process a batch of mutations
    async fn process_batch(&self, mutations: &[Mutation]) -> Result<()> {
        // Log what we're processing
        for mutation in mutations {
            match mutation {
                Mutation::AddNode(args) => {
                    log::debug!("Processing AddNode: id={}, name={}", args.id, args.name);
                }
                Mutation::AddEdge(args) => {
                    log::debug!(
                        "Processing AddEdge: source={}, target={}, name={}",
                        args.source_node_id,
                        args.target_node_id,
                        args.name
                    );
                }
                Mutation::AddFragment(args) => {
                    log::debug!(
                        "Processing AddFragment: id={}, body_len={}",
                        args.id,
                        args.content.0.len()
                    );
                }
                Mutation::Invalidate(args) => {
                    log::debug!(
                        "Processing Invalidate: id={}, reason={}",
                        args.id,
                        args.reason
                    );
                }
            }
        }

        // Process all mutations in a single call
        self.processor.process_mutations(mutations).await?;

        // Forward the batch to the next consumer in the chain if configured
        // This is a best-effort send - if the buffer is full, we log and continue
        if let Some(sender) = &self.next {
            if let Err(e) = sender.try_send(mutations.to_vec()) {
                log::warn!(
                    "[BUFFER FULL] Next consumer busy, dropping mutations: err={} count={}",
                    e,
                    mutations.len()
                );
            }
        }

        Ok(())
    }
}

/// Spawn a mutation consumer as a background task
pub fn spawn_consumer<P: Processor + 'static>(
    consumer: Consumer<P>,
) -> tokio::task::JoinHandle<Result<()>> {
    tokio::spawn(async move { consumer.run().await })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Id;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::time::Duration;

    // Mock processor for testing
    struct TestProcessor {
        delay_ms: u64,
        processed_count: Arc<AtomicUsize>,
    }

    impl TestProcessor {
        fn new(delay_ms: u64) -> Self {
            Self {
                delay_ms,
                processed_count: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn with_counter(delay_ms: u64, counter: Arc<AtomicUsize>) -> Self {
            Self {
                delay_ms,
                processed_count: counter,
            }
        }

        fn count(&self) -> usize {
            self.processed_count.load(Ordering::SeqCst)
        }
    }

    #[async_trait::async_trait]
    impl Processor for TestProcessor {
        async fn process_mutations(&self, mutations: &[Mutation]) -> Result<()> {
            if self.delay_ms > 0 {
                tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            }
            self.processed_count
                .fetch_add(mutations.len(), Ordering::SeqCst);
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_generic_consumer_basic() {
        let config = crate::WriterConfig {
            channel_buffer_size: 10,
        };

        let (writer, receiver) = {
            let (sender, receiver) = mpsc::channel(config.channel_buffer_size);
            let writer = crate::Writer::new(sender);
            (writer, receiver)
        };

        let processor = TestProcessor::new(1);
        let consumer = Consumer::new(receiver, config, processor);

        // Spawn consumer
        let consumer_handle = spawn_consumer(consumer);

        // Send a mutation
        let node_args = AddNode {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "test_node".to_string(),
        };
        writer.add_node(node_args).await.unwrap();

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Drop writer to close channel
        drop(writer);

        // Wait for consumer to finish
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_graph_processor_continues_when_fulltext_slow() {
        // This test verifies that the graph processor can continue processing mutations
        // even when the fulltext consumer isn't keeping up and the MPSC buffer is full.

        let graph_config = crate::WriterConfig {
            channel_buffer_size: 100,
        };

        let fulltext_config = crate::WriterConfig {
            channel_buffer_size: 2, // Very small buffer to force overflow
        };

        // Create channels
        let (graph_sender, graph_receiver) = mpsc::channel(graph_config.channel_buffer_size);
        let (fulltext_sender, fulltext_receiver) =
            mpsc::channel(fulltext_config.channel_buffer_size);

        // Create writer
        let writer = crate::Writer::new(graph_sender);

        // Create graph processor (fast, no delay)
        let graph_counter = Arc::new(AtomicUsize::new(0));
        let graph_processor = TestProcessor::with_counter(0, graph_counter.clone());

        // Create fulltext processor (slow, 50ms delay per mutation)
        let fulltext_counter = Arc::new(AtomicUsize::new(0));
        let fulltext_processor = TestProcessor::with_counter(50, fulltext_counter.clone());

        // Create consumers
        let graph_consumer = Consumer::with_next(
            graph_receiver,
            graph_config,
            graph_processor,
            fulltext_sender,
        );
        let fulltext_consumer =
            Consumer::new(fulltext_receiver, fulltext_config, fulltext_processor);

        // Spawn consumers
        let graph_handle = spawn_consumer(graph_consumer);
        let fulltext_handle = spawn_consumer(fulltext_consumer);

        // Send many mutations quickly (more than fulltext buffer can handle)
        let num_mutations = 20;
        for i in 0..num_mutations {
            let node_args = AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                name: format!("test_node_{}", i),
            };
            writer.add_node(node_args).await.unwrap();
        }

        // Give graph processor time to process all mutations
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Verify that graph processor processed all mutations
        let graph_processed = graph_counter.load(Ordering::SeqCst);
        assert_eq!(
            graph_processed, num_mutations,
            "Graph processor should have processed all {} mutations, but processed {}",
            num_mutations, graph_processed
        );

        // Verify that fulltext processor received fewer mutations (due to buffer overflow)
        let fulltext_processed = fulltext_counter.load(Ordering::SeqCst);
        assert!(
            fulltext_processed < num_mutations,
            "Fulltext processor should have processed fewer than {} mutations due to buffer overflow, but processed {}",
            num_mutations, fulltext_processed
        );

        println!(
            "Graph processed: {}, Fulltext processed: {}, Dropped: {}",
            graph_processed,
            fulltext_processed,
            graph_processed - fulltext_processed
        );

        // Drop writer to close channels
        drop(writer);

        // Wait for graph consumer to finish (should finish quickly)
        let graph_result = tokio::time::timeout(Duration::from_secs(5), graph_handle).await;
        assert!(
            graph_result.is_ok(),
            "Graph consumer should finish promptly"
        );
        graph_result.unwrap().unwrap().unwrap();

        // Give fulltext consumer some time to finish processing remaining items
        tokio::time::sleep(Duration::from_millis(500)).await;

        // The fulltext consumer will continue running until its channel is closed
        // Since we dropped the graph consumer's sender to fulltext, the fulltext receiver
        // channel should now be closed, allowing fulltext consumer to exit
        let fulltext_result = tokio::time::timeout(Duration::from_secs(5), fulltext_handle).await;
        assert!(
            fulltext_result.is_ok(),
            "Fulltext consumer should finish after channel closed"
        );
        fulltext_result.unwrap().unwrap().unwrap();
    }
}
