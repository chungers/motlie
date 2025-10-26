//! Provides the RocksDB-specific implementation for processing mutations from the MPSC queue
//! and writing them to RocksDB.

use anyhow::Result;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::{
    mutation::{Consumer, Processor},
    AddEdgeArgs, AddFragmentArgs, AddVertexArgs, InvalidateArgs, WriterConfig,
};

/// RocksDB-specific mutation processor
pub struct RocksDbProcessor;

impl RocksDbProcessor {
    /// Create a new RocksDbProcessor
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl Processor for RocksDbProcessor {
    /// Process an AddVertex mutation
    async fn process_add_vertex(&self, args: &AddVertexArgs) -> Result<()> {
        // TODO: Implement actual vertex insertion into RocksDB
        log::info!("[Rocks] Would insert vertex: {:?}", args);
        // Simulate some async work
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        Ok(())
    }

    /// Process an AddEdge mutation
    async fn process_add_edge(&self, args: &AddEdgeArgs) -> Result<()> {
        // TODO: Implement actual edge insertion into RocksDB
        log::info!("[Rocks] Would insert edge: {:?}", args);
        // Simulate some async work
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        Ok(())
    }

    /// Process an AddFragment mutation
    async fn process_add_fragment(&self, args: &AddFragmentArgs) -> Result<()> {
        // TODO: Implement actual fragment insertion into RocksDB
        log::info!("[Rocks] Would insert fragment: {:?}", args);
        // Simulate some async work
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        Ok(())
    }

    /// Process an Invalidate mutation
    async fn process_invalidate(&self, args: &InvalidateArgs) -> Result<()> {
        // TODO: Implement actual invalidation in RocksDB
        log::info!("[Rocks] Would invalidate: {:?}", args);
        // Simulate some async work
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        Ok(())
    }
}

/// Create a new RocksDB mutation consumer
pub fn create_rocks_consumer(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
) -> Consumer<RocksDbProcessor> {
    let processor = RocksDbProcessor::new();
    Consumer::new(receiver, config, processor)
}

/// Create a new RocksDB mutation consumer that chains to another processor
pub fn create_rocks_consumer_with_next(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
    next: mpsc::Sender<crate::Mutation>,
) -> Consumer<RocksDbProcessor> {
    let processor = RocksDbProcessor::new();
    Consumer::with_next(receiver, config, processor, next)
}

/// Spawn the RocksDB mutation consumer as a background task
pub fn spawn_rocks_consumer(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
) -> JoinHandle<Result<()>> {
    let consumer = create_rocks_consumer(receiver, config);
    crate::mutation::spawn_consumer(consumer)
}

/// Spawn the RocksDB mutation consumer as a background task with chaining to next processor
pub fn spawn_rocks_consumer_with_next(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
    next: mpsc::Sender<crate::Mutation>,
) -> JoinHandle<Result<()>> {
    let consumer = create_rocks_consumer_with_next(receiver, config, next);
    crate::mutation::spawn_consumer(consumer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{create_mutation_writer, Id};
    use tokio::time::Duration;

    #[tokio::test]
    async fn test_rocks_consumer_basic_processing() {
        let config = WriterConfig {
            channel_buffer_size: 10,
            max_batch_size: 2,
            batch_timeout_ms: 50,
        };

        let (writer, receiver) = create_mutation_writer(config.clone());

        // Spawn consumer
        let consumer_handle = spawn_rocks_consumer(receiver, config);

        // Send some mutations
        let vertex_args = AddVertexArgs {
            id: Id::new(),
            ts_millis: 1234567890,
            name: "test_vertex".to_string(),
        };
        writer.add_vertex(vertex_args).await.unwrap();

        let edge_args = AddEdgeArgs {
            source_vertex_id: Id::new(),
            target_vertex_id: Id::new(),
            ts_millis: 1234567890,
            name: "test_edge".to_string(),
        };
        writer.add_edge(edge_args).await.unwrap();

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Drop writer to close channel
        drop(writer);

        // Wait for consumer to finish
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_rocks_consumer_batch_processing() {
        let config = WriterConfig {
            channel_buffer_size: 100,
            max_batch_size: 3,
            batch_timeout_ms: 1000, // High timeout to test batch size trigger
        };

        let (writer, receiver) = create_mutation_writer(config.clone());
        let consumer_handle = spawn_rocks_consumer(receiver, config);

        // Send 5 mutations rapidly
        for i in 0..5 {
            let vertex_args = AddVertexArgs {
                id: Id::new(),
                ts_millis: 1234567890 + i,
                name: format!("test_vertex_{}", i),
            };
            writer.add_vertex(vertex_args).await.unwrap();
        }

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Close and wait
        drop(writer);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_rocks_consumer_all_mutation_types() {
        let config = WriterConfig::default();
        let (writer, receiver) = create_mutation_writer(config.clone());
        let consumer_handle = spawn_rocks_consumer(receiver, config);

        // Test all mutation types
        writer
            .add_vertex(AddVertexArgs {
                id: Id::new(),
                ts_millis: 1234567890,
                name: "vertex".to_string(),
            })
            .await
            .unwrap();

        writer
            .add_edge(AddEdgeArgs {
                source_vertex_id: Id::new(),
                target_vertex_id: Id::new(),
                ts_millis: 1234567890,
                name: "edge".to_string(),
            })
            .await
            .unwrap();

        writer
            .add_fragment(AddFragmentArgs {
                id: Id::new(),
                ts_millis: 1234567890,
                body: "fragment body".to_string(),
            })
            .await
            .unwrap();

        writer
            .invalidate(crate::InvalidateArgs {
                id: Id::new(),
                ts_millis: 1234567890,
                reason: "test invalidation".to_string(),
            })
            .await
            .unwrap();

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Close and wait
        drop(writer);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_rocks_to_bm25_chaining() {
        let config = WriterConfig {
            channel_buffer_size: 100,
            max_batch_size: 5,
            batch_timeout_ms: 100,
        };

        // Create the BM25 consumer (end of chain)
        let (bm25_sender, bm25_receiver) = mpsc::channel(config.channel_buffer_size);
        let bm25_handle = crate::spawn_bm25_consumer(bm25_receiver, config.clone());

        // Create the RocksDB consumer that forwards to BM25
        let (writer, rocks_receiver) = create_mutation_writer(config.clone());
        let rocks_handle =
            spawn_rocks_consumer_with_next(rocks_receiver, config.clone(), bm25_sender);

        // Send mutations - they should flow through RocksDB -> BM25
        for i in 0..3 {
            let vertex_args = AddVertexArgs {
                id: Id::new(),
                ts_millis: 1234567890 + i,
                name: format!("chained_vertex_{}", i),
            };
            let fragment_args = AddFragmentArgs {
                id: Id::new(),
                ts_millis: 1234567890 + i,
                body: format!("Chained fragment {} processed by both RocksDB and BM25", i),
            };

            writer.add_vertex(vertex_args).await.unwrap();
            writer.add_fragment(fragment_args).await.unwrap();
        }

        // Give both consumers time to process
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Shutdown the chain from the beginning
        drop(writer);

        // Wait for RocksDB consumer to complete (which will close BM25's channel)
        rocks_handle.await.unwrap().unwrap();

        // Wait for BM25 consumer to complete
        bm25_handle.await.unwrap().unwrap();
    }
}
