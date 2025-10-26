//! # BM25 Search Index Module
//!
//! Provides the BM25-specific implementation for processing mutations from the MPSC queue
//! and updating the BM25 search index.

use anyhow::Result;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::{
    mutation::{Consumer, Processor},
    AddEdgeArgs, AddFragmentArgs, AddVertexArgs, InvalidateArgs, WriterConfig,
};

/// BM25-specific mutation processor for search indexing
pub struct Bm25Processor {
    /// Configuration for BM25 scoring
    pub k1: f32,
    pub b: f32,
}

impl Bm25Processor {
    /// Create a new BM25 processor with default parameters
    pub fn new() -> Self {
        Self {
            k1: 1.2, // Default BM25 k1 parameter
            b: 0.75, // Default BM25 b parameter
        }
    }

    /// Create a new BM25 processor with custom parameters
    pub fn with_params(k1: f32, b: f32) -> Self {
        Self { k1, b }
    }
}

impl Default for Bm25Processor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Processor for Bm25Processor {
    /// Process an AddVertex mutation - index vertex for search
    async fn process_add_vertex(&self, args: &AddVertexArgs) -> Result<()> {
        // TODO: Implement actual vertex indexing in BM25 search index
        log::info!(
            "[BM25] Would index vertex for search: id={}, name='{}', k1={}, b={}",
            args.id,
            args.name,
            self.k1,
            self.b
        );

        // TODO: Extract terms from vertex name and content
        // TODO: Update document frequencies and term frequencies
        // TODO: Update BM25 index structures

        // Simulate some async indexing work
        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;

        Ok(())
    }

    /// Process an AddEdge mutation - index edge relationships for search
    async fn process_add_edge(&self, args: &AddEdgeArgs) -> Result<()> {
        // TODO: Implement actual edge relationship indexing in BM25 search index
        log::info!(
            "[BM25] Would index edge relationship: source={}, target={}, name='{}', k1={}, b={}",
            args.source_vertex_id,
            args.target_vertex_id,
            args.name,
            self.k1,
            self.b
        );

        // TODO: Index edge name and relationship context
        // TODO: Update graph-aware search features
        // TODO: Update BM25 scores considering edge relationships

        // Simulate some async indexing work
        tokio::time::sleep(tokio::time::Duration::from_millis(2)).await;

        Ok(())
    }

    /// Process an AddFragment mutation - index fragment content for full-text search
    async fn process_add_fragment(&self, args: &AddFragmentArgs) -> Result<()> {
        // TODO: Implement actual fragment content indexing in BM25 search index
        log::info!(
            "[BM25] Would index fragment content: id={}, body_len={}, k1={}, b={}",
            args.id,
            args.body.len(),
            self.k1,
            self.b
        );

        // TODO: Tokenize fragment body
        // TODO: Extract and stem terms
        // TODO: Update term frequencies and document frequencies
        // TODO: Calculate and store BM25 scores

        // Simulate some async indexing work (fragments are more text-heavy)
        tokio::time::sleep(tokio::time::Duration::from_millis(3)).await;

        Ok(())
    }

    /// Process an Invalidate mutation - remove from search index
    async fn process_invalidate(&self, args: &InvalidateArgs) -> Result<()> {
        // TODO: Implement actual invalidation in BM25 search index
        log::info!(
            "[BM25] Would remove from search index: id={}, reason='{}', k1={}, b={}",
            args.id,
            args.reason,
            self.k1,
            self.b
        );

        // TODO: Remove document from index
        // TODO: Update document frequencies
        // TODO: Recalculate BM25 scores for affected terms
        // TODO: Cleanup orphaned index entries

        // Simulate some async cleanup work
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;

        Ok(())
    }
}

/// Create a new BM25 mutation consumer with default parameters
pub fn create_bm25_consumer(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
) -> Consumer<Bm25Processor> {
    let processor = Bm25Processor::new();
    Consumer::new(receiver, config, processor)
}

/// Create a new BM25 mutation consumer with default parameters that chains to another processor
pub fn create_bm25_consumer_with_next(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
    next: mpsc::Sender<crate::Mutation>,
) -> Consumer<Bm25Processor> {
    let processor = Bm25Processor::new();
    Consumer::with_next(receiver, config, processor, next)
}

/// Create a new BM25 mutation consumer with custom BM25 parameters
pub fn create_bm25_consumer_with_params(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
    k1: f32,
    b: f32,
) -> Consumer<Bm25Processor> {
    let processor = Bm25Processor::with_params(k1, b);
    Consumer::new(receiver, config, processor)
}

/// Create a new BM25 mutation consumer with custom BM25 parameters that chains to another processor
pub fn create_bm25_consumer_with_params_and_next(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
    k1: f32,
    b: f32,
    next: mpsc::Sender<crate::Mutation>,
) -> Consumer<Bm25Processor> {
    let processor = Bm25Processor::with_params(k1, b);
    Consumer::with_next(receiver, config, processor, next)
}

/// Spawn the BM25 mutation consumer as a background task with default parameters
pub fn spawn_bm25_consumer(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
) -> JoinHandle<Result<()>> {
    let consumer = create_bm25_consumer(receiver, config);
    crate::mutation::spawn_consumer(consumer)
}

/// Spawn the BM25 mutation consumer as a background task with default parameters and chaining
pub fn spawn_bm25_consumer_with_next(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
    next: mpsc::Sender<crate::Mutation>,
) -> JoinHandle<Result<()>> {
    let consumer = create_bm25_consumer_with_next(receiver, config, next);
    crate::mutation::spawn_consumer(consumer)
}

/// Spawn the BM25 mutation consumer as a background task with custom BM25 parameters
pub fn spawn_bm25_consumer_with_params(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
    k1: f32,
    b: f32,
) -> JoinHandle<Result<()>> {
    let consumer = create_bm25_consumer_with_params(receiver, config, k1, b);
    crate::mutation::spawn_consumer(consumer)
}

/// Spawn the BM25 mutation consumer as a background task with custom BM25 parameters and chaining
pub fn spawn_bm25_consumer_with_params_and_next(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
    k1: f32,
    b: f32,
    next: mpsc::Sender<crate::Mutation>,
) -> JoinHandle<Result<()>> {
    let consumer = create_bm25_consumer_with_params_and_next(receiver, config, k1, b, next);
    crate::mutation::spawn_consumer(consumer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{create_mutation_writer, Id};
    use tokio::time::Duration;

    #[tokio::test]
    async fn test_bm25_consumer_basic_processing() {
        let config = WriterConfig {
            channel_buffer_size: 10,
            max_batch_size: 2,
            batch_timeout_ms: 50,
        };

        let (writer, receiver) = create_mutation_writer(config.clone());

        // Spawn consumer
        let consumer_handle = spawn_bm25_consumer(receiver, config);

        // Send some mutations
        let vertex_args = AddVertexArgs {
            id: Id::new(),
            ts_millis: 1234567890,
            name: "test_vertex".to_string(),
        };

        writer.add_vertex(vertex_args).await.unwrap();

        let fragment_args = AddFragmentArgs {
            id: Id::new(),
            ts_millis: 1234567890,
            body: "This is a test fragment with some searchable content".to_string(),
        };

        writer.add_fragment(fragment_args).await.unwrap();

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Drop writer to close channel
        drop(writer);

        // Wait for consumer to finish
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_bm25_consumer_with_custom_params() {
        let config = WriterConfig {
            channel_buffer_size: 10,
            max_batch_size: 5,
            batch_timeout_ms: 100,
        };

        let k1 = 1.5;
        let b = 0.8;

        let (writer, receiver) = create_mutation_writer(config.clone());
        let consumer_handle = spawn_bm25_consumer_with_params(receiver, config, k1, b);

        // Send a fragment with substantial content
        let fragment_args = AddFragmentArgs {
            id: Id::new(),
            ts_millis: 1234567890,
            body: "The quick brown fox jumps over the lazy dog. This is a longer text fragment that would benefit from BM25 scoring with custom parameters.".to_string(),
        };

        writer.add_fragment(fragment_args).await.unwrap();

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(150)).await;

        // Close and wait
        drop(writer);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_bm25_consumer_all_mutation_types() {
        let config = WriterConfig::default();
        let (writer, receiver) = create_mutation_writer(config.clone());
        let consumer_handle = spawn_bm25_consumer(receiver, config);

        // Test all mutation types with search-relevant content
        writer
            .add_vertex(AddVertexArgs {
                id: Id::new(),
                ts_millis: 1234567890,
                name: "search vertex".to_string(),
            })
            .await
            .unwrap();

        writer
            .add_edge(AddEdgeArgs {
                source_vertex_id: Id::new(),
                target_vertex_id: Id::new(),
                ts_millis: 1234567890,
                name: "connects to".to_string(),
            })
            .await
            .unwrap();

        writer
            .add_fragment(AddFragmentArgs {
                id: Id::new(),
                ts_millis: 1234567890,
                body: "This fragment contains searchable text that should be indexed using BM25 algorithm for effective information retrieval.".to_string(),
            })
            .await
            .unwrap();

        writer
            .invalidate(crate::InvalidateArgs {
                id: Id::new(),
                ts_millis: 1234567890,
                reason: "content removed from search index".to_string(),
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
    async fn test_bm25_processor_creation() {
        // Test default processor
        let processor = Bm25Processor::new();
        assert_eq!(processor.k1, 1.2);
        assert_eq!(processor.b, 0.75);

        // Test processor with custom params
        let processor = Bm25Processor::with_params(2.0, 0.5);
        assert_eq!(processor.k1, 2.0);
        assert_eq!(processor.b, 0.5);

        // Test default trait
        let processor: Bm25Processor = Default::default();
        assert_eq!(processor.k1, 1.2);
        assert_eq!(processor.b, 0.75);
    }

    #[tokio::test]
    async fn test_bm25_batch_processing() {
        let config = WriterConfig {
            channel_buffer_size: 100,
            max_batch_size: 3,
            batch_timeout_ms: 1000, // High timeout to test batch size trigger
        };

        let (writer, receiver) = create_mutation_writer(config.clone());
        let consumer_handle = spawn_bm25_consumer(receiver, config);

        // Send 5 fragments rapidly to test batching
        for i in 0..5 {
            let fragment_args = AddFragmentArgs {
                id: Id::new(),
                ts_millis: 1234567890 + i,
                body: format!("Fragment {} with searchable content for BM25 indexing", i),
            };
            writer.add_fragment(fragment_args).await.unwrap();
        }

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Close and wait
        drop(writer);
        consumer_handle.await.unwrap().unwrap();
    }
}
