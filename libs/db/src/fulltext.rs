//! Provides the full-text search implementation for processing mutations from the MPSC queue
//! and updating the full-text search index.

use anyhow::Result;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::{
    mutation::{Consumer, Processor},
    AddEdge, AddFragment, AddNode, InvalidateArgs, WriterConfig,
};

/// Full-text search mutation processor for search indexing
pub struct FullTextProcessor {
    /// Configuration for BM25 scoring
    pub k1: f32,
    pub b: f32,
}

impl FullTextProcessor {
    /// Create a new full-text processor with default parameters
    pub fn new() -> Self {
        Self {
            k1: 1.2, // Default BM25 k1 parameter
            b: 0.75, // Default BM25 b parameter
        }
    }

    /// Create a new full-text processor with custom parameters
    pub fn with_params(k1: f32, b: f32) -> Self {
        Self { k1, b }
    }
}

impl Default for FullTextProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Processor for FullTextProcessor {
    /// Process an AddNode mutation - index node for search
    async fn process_add_node(&self, args: &AddNode) -> Result<()> {
        // TODO: Implement actual node indexing in full-text search index
        log::info!(
            "[FullText] Would index node for search: id={}, name='{}', k1={}, b={}",
            args.id,
            args.name,
            self.k1,
            self.b
        );
        // TODO: Extract terms from node name and content
        // TODO: Update document frequencies and term frequencies
        // TODO: Update BM25 index structures

        Ok(())
    }

    /// Process an AddEdge mutation - index edge relationships for search
    async fn process_add_edge(&self, args: &AddEdge) -> Result<()> {
        // TODO: Implement actual edge relationship indexing in full-text search index
        log::info!(
            "[FullText] Would index edge relationship: source={}, target={}, name='{}', k1={}, b={}",
            args.source_node_id,
            args.target_node_id,
            args.name,
            self.k1,
            self.b
        );
        // TODO: Index edge name and relationship context
        // TODO: Update graph-aware search features
        // TODO: Update BM25 scores considering edge relationships

        Ok(())
    }

    /// Process an AddFragment mutation - index fragment content for full-text search
    async fn process_add_fragment(&self, args: &AddFragment) -> Result<()> {
        // TODO: Implement actual fragment content indexing in full-text search index
        log::info!(
            "[FullText] Would index fragment content: id={}, body_len={}, k1={}, b={}",
            args.id,
            args.content.len(),
            self.k1,
            self.b
        );
        // TODO: Tokenize fragment body
        // TODO: Extract and stem terms
        // TODO: Update term frequencies and document frequencies
        // TODO: Calculate and store BM25 scores

        Ok(())
    }

    /// Process an Invalidate mutation - remove from search index
    async fn process_invalidate(&self, args: &InvalidateArgs) -> Result<()> {
        // TODO: Implement actual invalidation in full-text search index
        log::info!(
            "[FullText] Would remove from search index: id={}, reason='{}', k1={}, b={}",
            args.id,
            args.reason,
            self.k1,
            self.b
        );
        // TODO: Remove document from index
        // TODO: Update document frequencies
        // TODO: Recalculate BM25 scores for affected terms
        // TODO: Cleanup orphaned index entries

        Ok(())
    }
}

/// Create a new full-text mutation consumer with default parameters
pub fn create_fulltext_consumer(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
) -> Consumer<FullTextProcessor> {
    let processor = FullTextProcessor::new();
    Consumer::new(receiver, config, processor)
}

/// Create a new full-text mutation consumer with default parameters that chains to another processor
pub fn create_fulltext_consumer_with_next(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
    next: mpsc::Sender<crate::Mutation>,
) -> Consumer<FullTextProcessor> {
    let processor = FullTextProcessor::new();
    Consumer::with_next(receiver, config, processor, next)
}

/// Create a new full-text mutation consumer with custom BM25 parameters
pub fn create_fulltext_consumer_with_params(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
    k1: f32,
    b: f32,
) -> Consumer<FullTextProcessor> {
    let processor = FullTextProcessor::with_params(k1, b);
    Consumer::new(receiver, config, processor)
}

/// Create a new full-text mutation consumer with custom BM25 parameters that chains to another processor
pub fn create_fulltext_consumer_with_params_and_next(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
    k1: f32,
    b: f32,
    next: mpsc::Sender<crate::Mutation>,
) -> Consumer<FullTextProcessor> {
    let processor = FullTextProcessor::with_params(k1, b);
    Consumer::with_next(receiver, config, processor, next)
}

/// Spawn the full-text mutation consumer as a background task with default parameters
pub fn spawn_fulltext_consumer(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
) -> JoinHandle<Result<()>> {
    let consumer = create_fulltext_consumer(receiver, config);
    crate::mutation::spawn_consumer(consumer)
}

/// Spawn the full-text mutation consumer as a background task with default parameters and chaining
pub fn spawn_fulltext_consumer_with_next(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
    next: mpsc::Sender<crate::Mutation>,
) -> JoinHandle<Result<()>> {
    let consumer = create_fulltext_consumer_with_next(receiver, config, next);
    crate::mutation::spawn_consumer(consumer)
}

/// Spawn the full-text mutation consumer as a background task with custom BM25 parameters
pub fn spawn_fulltext_consumer_with_params(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
    k1: f32,
    b: f32,
) -> JoinHandle<Result<()>> {
    let consumer = create_fulltext_consumer_with_params(receiver, config, k1, b);
    crate::mutation::spawn_consumer(consumer)
}

/// Spawn the full-text mutation consumer as a background task with custom BM25 parameters and chaining
pub fn spawn_fulltext_consumer_with_params_and_next(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
    k1: f32,
    b: f32,
    next: mpsc::Sender<crate::Mutation>,
) -> JoinHandle<Result<()>> {
    let consumer = create_fulltext_consumer_with_params_and_next(receiver, config, k1, b, next);
    crate::mutation::spawn_consumer(consumer)
}

#[cfg(test)]
#[path = "fulltext_tests.rs"]
mod fulltext_tests;
