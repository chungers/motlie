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
#[path = "rocks_tests.rs"]
mod rocks_tests;
