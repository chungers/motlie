//! Provides the graph-specific implementation for processing mutations from the MPSC queue
//! and writing them to the graph store.

use anyhow::Result;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use crate::{
    mutation::{Consumer, Processor},
    AddEdgeArgs, AddFragmentArgs, AddVertexArgs, InvalidateArgs, WriterConfig,
};

/// Graph-specific mutation processor
pub struct GraphProcessor;

impl GraphProcessor {
    /// Create a new GraphProcessor
    pub fn new() -> Self {
        Self
    }
}

#[async_trait::async_trait]
impl Processor for GraphProcessor {
    /// Process an AddVertex mutation
    async fn process_add_vertex(&self, args: &AddVertexArgs) -> Result<()> {
        // TODO: Implement actual vertex insertion into graph store
        log::info!("[Graph] Would insert vertex: {:?}", args);
        // Simulate some async work
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        Ok(())
    }

    /// Process an AddEdge mutation
    async fn process_add_edge(&self, args: &AddEdgeArgs) -> Result<()> {
        // TODO: Implement actual edge insertion into graph store
        log::info!("[Graph] Would insert edge: {:?}", args);
        // Simulate some async work
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        Ok(())
    }

    /// Process an AddFragment mutation
    async fn process_add_fragment(&self, args: &AddFragmentArgs) -> Result<()> {
        // TODO: Implement actual fragment insertion into graph store
        log::info!("[Graph] Would insert fragment: {:?}", args);
        // Simulate some async work
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        Ok(())
    }

    /// Process an Invalidate mutation
    async fn process_invalidate(&self, args: &InvalidateArgs) -> Result<()> {
        // TODO: Implement actual invalidation in graph store
        log::info!("[Graph] Would invalidate: {:?}", args);
        // Simulate some async work
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        Ok(())
    }
}

/// Create a new graph mutation consumer
pub fn create_graph_consumer(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
) -> Consumer<GraphProcessor> {
    let processor = GraphProcessor::new();
    Consumer::new(receiver, config, processor)
}

/// Create a new graph mutation consumer that chains to another processor
pub fn create_graph_consumer_with_next(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
    next: mpsc::Sender<crate::Mutation>,
) -> Consumer<GraphProcessor> {
    let processor = GraphProcessor::new();
    Consumer::with_next(receiver, config, processor, next)
}

/// Spawn the graph mutation consumer as a background task
pub fn spawn_graph_consumer(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
) -> JoinHandle<Result<()>> {
    let consumer = create_graph_consumer(receiver, config);
    crate::mutation::spawn_consumer(consumer)
}

/// Spawn the graph mutation consumer as a background task with chaining to next processor
pub fn spawn_graph_consumer_with_next(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
    next: mpsc::Sender<crate::Mutation>,
) -> JoinHandle<Result<()>> {
    let consumer = create_graph_consumer_with_next(receiver, config, next);
    crate::mutation::spawn_consumer(consumer)
}

#[cfg(test)]
#[path = "graph_tests.rs"]
mod graph_tests;
