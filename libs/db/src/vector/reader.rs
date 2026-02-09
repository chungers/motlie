//! Vector query reader module providing query infrastructure.
//!
//! This module follows the same pattern as graph::reader:
//! - Reader - handle for sending queries (with Processor for SearchKNN)
//! - ReaderConfig - configuration
//! - Consumer - processes queries from channel
//! - Spawn functions for creating consumers
//!
//! Uses flume for MPMC (multi-producer, multi-consumer) channels,
//! enabling multiple query consumers to process queries in parallel.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────┐     MPMC (flume)     ┌──────────┐
//! │ Reader  │──────────────────────│ Consumer │ (N workers)
//! └─────────┘       Query          └──────────┘
//!      │                                │
//!      ▼                                ▼
//! ┌───────────┐                    ┌──────────┐
//! │ Processor │                    │ Storage  │
//! └───────────┘                    └──────────┘
//! ```
//!
//! The Reader holds a Processor reference enabling SearchKNN queries to
//! perform HNSW index operations directly through the reader.

use std::sync::Arc;

use anyhow::{Context, Result};

use super::processor::Processor;
use super::query::QueryRequest;
use super::Storage;

// ============================================================================
// ReaderConfig
// ============================================================================

/// Configuration for the query reader.
#[derive(Debug, Clone)]
pub struct ReaderConfig {
    /// Size of the MPMC channel buffer
    pub channel_buffer_size: usize,
}

impl Default for ReaderConfig {
    fn default() -> Self {
        Self {
            channel_buffer_size: 1000,
        }
    }
}

// ============================================================================
// Reader
// ============================================================================

/// Handle for sending vector queries to be processed.
///
/// The Reader sends queries through an MPMC (flume) channel, allowing
/// multiple consumer workers to process queries in parallel. It also holds
/// a Processor reference for SearchKNN operations.
///
/// # Example
///
/// ```rust,ignore
/// use motlie_db::vector::{Reader, GetVector, SearchKNN, Runnable};
/// use std::time::Duration;
///
/// // Create reader with storage
/// let (reader, receiver) = create_reader_with_storage(
///     ReaderConfig::default(),
///     storage.clone(),
/// );
///
/// // Point lookup query
/// let query = GetVector::new(embedding_code, id);
/// let result = query.run(&reader, Duration::from_secs(5)).await?;
///
/// // SearchKNN query
/// let results = SearchKNN::new(&embedding, query_vec, 10)
///     .with_ef(100)
///     .run(&reader, Duration::from_secs(5))
///     .await?;
/// ```
#[derive(Clone)]
pub struct Reader {
    sender: flume::Sender<QueryRequest>,
    processor: Arc<Processor>,
}

impl Reader {
    /// Create a new Reader with the given sender and processor.
    pub(crate) fn new(sender: flume::Sender<QueryRequest>, processor: Arc<Processor>) -> Self {
        Self { sender, processor }
    }

    /// Send a query to the reader queue.
    ///
    /// This is typically called by the Runnable trait implementation.
    pub async fn send_query(&self, query: QueryRequest) -> Result<()> {
        self.sender
            .send_async(query)
            .await
            .context("Failed to send query to reader queue")
    }

    /// Check if the reader is still active (receiver hasn't been dropped)
    pub fn is_closed(&self) -> bool {
        self.sender.is_disconnected()
    }

    /// Get the processor reference.
    ///
    /// Used internally by `Runnable` implementations (e.g., `SearchKNN`) to access
    /// the processor for query dispatch.
    pub(crate) fn processor(&self) -> &Arc<Processor> {
        &self.processor
    }
}

impl std::fmt::Debug for Reader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reader")
            .field("sender", &self.sender)
            .field("processor", &"<Arc<Processor>>")
            .finish()
    }
}

/// Create a Reader from Storage (auto-creates Processor internally).
///
/// This is the recommended way to create a Reader - it constructs
/// the internal Processor automatically.
///
/// # Returns
///
/// Tuple of (Reader, flume::Receiver<Query>) - spawn consumers with the receiver.
///
/// # Example
///
/// ```rust,ignore
/// let storage = Arc::new(Storage::readwrite(path));
/// let (reader, receiver) = create_reader_with_storage(
///     ReaderConfig::default(),
///     storage.clone(),
/// );
/// spawn_query_consumers_with_storage_autoreg(receiver, config, storage, 2);
///
/// // Use SearchKNN::run() for searches
/// let results = SearchKNN::new(&embedding, query, 10)
///     .run(&reader, timeout)
///     .await?;
/// ```
pub fn create_reader_with_storage(
    config: ReaderConfig,
    storage: Arc<super::Storage>,
) -> (Reader, flume::Receiver<QueryRequest>) {
    let registry = storage.cache().clone();
    let processor = Arc::new(Processor::new(storage, registry));
    let (sender, receiver) = flume::bounded(config.channel_buffer_size);
    let reader = Reader::new(sender, processor);
    (reader, receiver)
}

/// Create a Reader with an explicit Processor.
///
/// Use this when you need to share a Processor across multiple Readers
/// or have custom Processor configuration.
pub(crate) fn create_reader(
    config: ReaderConfig,
    processor: Arc<Processor>,
) -> (Reader, flume::Receiver<QueryRequest>) {
    let (sender, receiver) = flume::bounded(config.channel_buffer_size);
    let reader = Reader::new(sender, processor);
    (reader, receiver)
}

// ============================================================================
// Consumer
// ============================================================================

/// Consumer that processes vector queries from a channel.
///
/// Multiple consumers can be spawned to process queries in parallel
/// thanks to flume's MPMC semantics.
pub struct Consumer {
    receiver: flume::Receiver<QueryRequest>,
    config: ReaderConfig,
    storage: Arc<Storage>,
}

impl Consumer {
    /// Create a new Consumer.
    pub fn new(
        receiver: flume::Receiver<QueryRequest>,
        config: ReaderConfig,
        storage: Arc<Storage>,
    ) -> Self {
        Self {
            receiver,
            config,
            storage,
        }
    }

    /// Process queries continuously until the channel is closed.
    #[tracing::instrument(skip(self), name = "vector_query_consumer")]
    pub async fn run(self) -> Result<()> {
        tracing::info!(config = ?self.config, "Starting vector query consumer");

        loop {
            match self.receiver.recv_async().await {
                Ok(mut request) => {
                    self.process_query(&mut request).await;
                }
                Err(_) => {
                    // Channel closed
                    tracing::info!("Vector query consumer shutting down - channel closed");
                    return Ok(());
                }
            }
        }
    }

    /// Process a single query.
    #[tracing::instrument(skip(self, request), fields(query_type = %request.payload))]
    async fn process_query(&self, request: &mut QueryRequest) {
        tracing::debug!(query = %request.payload, "Processing vector query");
        let result = if let Some(timeout) = request.timeout {
            match tokio::time::timeout(timeout, request.payload.execute_with_storage(&self.storage)).await
            {
                Ok(result) => result,
                Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
            }
        } else {
            request.payload.execute_with_storage(&self.storage).await
        };
        request.respond(result);
    }
}

/// Spawn a query consumer as a tokio task.
///
/// Returns a JoinHandle that resolves when the consumer completes.
pub fn spawn_consumer(consumer: Consumer) -> tokio::task::JoinHandle<Result<()>> {
    tokio::spawn(async move { consumer.run().await })
}

/// Spawn multiple query consumers for parallel processing.
///
/// This is the recommended way to set up vector query handling,
/// as it leverages flume's MPMC semantics for work-stealing.
///
/// # Arguments
///
/// * `receiver` - The flume receiver to clone for each consumer
/// * `config` - Configuration for each consumer
/// * `storage` - Shared storage reference
/// * `count` - Number of consumers to spawn
///
/// # Returns
///
/// Vector of JoinHandles for all spawned consumers
pub fn spawn_consumers(
    receiver: flume::Receiver<QueryRequest>,
    config: ReaderConfig,
    storage: Arc<Storage>,
    count: usize,
) -> Vec<tokio::task::JoinHandle<Result<()>>> {
    (0..count)
        .map(|_| {
            let consumer = Consumer::new(receiver.clone(), config.clone(), storage.clone());
            spawn_consumer(consumer)
        })
        .collect()
}

// ============================================================================
// ProcessorConsumer - Consumer with Processor for SearchKNN support
// ============================================================================

/// Consumer that processes vector queries with Processor access.
///
/// Unlike the basic `Consumer`, this variant holds a `Processor` reference
/// enabling SearchKNN queries to perform HNSW index operations.
///
/// Use this when your query workload includes SearchKNN.
pub(crate) struct ProcessorConsumer {
    receiver: flume::Receiver<QueryRequest>,
    config: ReaderConfig,
    processor: Arc<Processor>,
}

impl ProcessorConsumer {
    /// Create a new ProcessorConsumer.
    pub(crate) fn new(
        receiver: flume::Receiver<QueryRequest>,
        config: ReaderConfig,
        processor: Arc<Processor>,
    ) -> Self {
        Self {
            receiver,
            config,
            processor,
        }
    }

    /// Process queries continuously until the channel is closed.
    #[tracing::instrument(skip(self), name = "vector_query_consumer_with_processor")]
    pub async fn run(self) -> Result<()> {
        tracing::info!(config = ?self.config, "Starting vector query consumer (with Processor)");

        loop {
            match self.receiver.recv_async().await {
                Ok(mut request) => {
                    self.process_query(&mut request).await;
                }
                Err(_) => {
                    tracing::info!("Vector query consumer shutting down - channel closed");
                    return Ok(());
                }
            }
        }
    }

    /// Process a single query.
    #[tracing::instrument(skip(self, request), fields(query_type = %request.payload))]
    async fn process_query(&self, request: &mut QueryRequest) {
        tracing::debug!(query = %request.payload, "Processing vector query");
        let result = if let Some(timeout) = request.timeout {
            match tokio::time::timeout(
                timeout,
                request.payload.execute_with_processor(&self.processor),
            )
            .await
            {
                Ok(result) => result,
                Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
            }
        } else {
            request.payload.execute_with_processor(&self.processor).await
        };
        request.respond(result);
    }

    /// Get the processor reference (for SearchKNN dispatch setup).
    pub(crate) fn processor(&self) -> &Arc<Processor> {
        &self.processor
    }
}

/// Spawn a Processor-backed query consumer as a tokio task.
pub(crate) fn spawn_consumer_with_processor(
    consumer: ProcessorConsumer,
) -> tokio::task::JoinHandle<Result<()>> {
    tokio::spawn(async move { consumer.run().await })
}

/// Spawn multiple Processor-backed query consumers for parallel processing.
///
/// This is the recommended way to set up vector query handling when
/// your workload includes SearchKNN queries.
///
/// # Arguments
///
/// * `receiver` - The flume receiver to clone for each consumer
/// * `config` - Configuration for each consumer
/// * `processor` - Shared processor reference (includes Storage access)
/// * `count` - Number of consumers to spawn
///
/// # Returns
///
/// Vector of JoinHandles for all spawned consumers
pub(crate) fn spawn_consumers_with_processor(
    receiver: flume::Receiver<QueryRequest>,
    config: ReaderConfig,
    processor: Arc<Processor>,
    count: usize,
) -> Vec<tokio::task::JoinHandle<Result<()>>> {
    (0..count)
        .map(|_| {
            let consumer =
                ProcessorConsumer::new(receiver.clone(), config.clone(), processor.clone());
            spawn_consumer_with_processor(consumer)
        })
        .collect()
}

/// Spawn multiple Processor-backed query consumers with storage and registry.
///
/// This is the recommended way to set up query handling - it constructs
/// the internal Processor automatically, hiding the implementation detail.
///
/// # Arguments
///
/// * `receiver` - The flume receiver from `create_reader()`
/// * `config` - Reader configuration
/// * `storage` - Vector storage instance
/// * `registry` - Embedding registry (typically from `storage.cache().clone()`)
/// * `count` - Number of consumers to spawn
///
/// # Returns
///
/// Vector of JoinHandles for all spawned consumers.
pub fn spawn_query_consumers_with_storage(
    receiver: flume::Receiver<QueryRequest>,
    config: ReaderConfig,
    storage: Arc<super::Storage>,
    registry: Arc<super::registry::EmbeddingRegistry>,
    count: usize,
) -> Vec<tokio::task::JoinHandle<Result<()>>> {
    let processor = Arc::new(Processor::new(storage, registry));
    spawn_consumers_with_processor(receiver, config, processor, count)
}

/// Spawn multiple query consumers with storage and auto-created registry.
///
/// Convenience function for quick setup - creates the embedding registry
/// from storage automatically.
pub fn spawn_query_consumers_with_storage_autoreg(
    receiver: flume::Receiver<QueryRequest>,
    config: ReaderConfig,
    storage: Arc<super::Storage>,
    count: usize,
) -> Vec<tokio::task::JoinHandle<Result<()>>> {
    let registry = storage.cache().clone();
    spawn_query_consumers_with_storage(receiver, config, storage, registry, count)
}

// ============================================================================
// Deprecated Aliases (for backwards compatibility)
// ============================================================================

// SearchReader alias removed (breaking change): use Reader directly.

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn create_test_storage() -> Arc<Storage> {
        let dir = tempdir().unwrap();
        let mut storage = Storage::readwrite(dir.path());
        storage.ready().unwrap();
        Arc::new(storage)
    }

    #[tokio::test]
    async fn test_reader_creation() {
        let storage = create_test_storage();
        let (reader, _receiver) = create_reader_with_storage(ReaderConfig::default(), storage);
        assert!(!reader.is_closed());
    }

    #[tokio::test]
    async fn test_reader_channel_closed() {
        let storage = create_test_storage();
        let (reader, receiver) = create_reader_with_storage(ReaderConfig::default(), storage);
        drop(receiver);
        assert!(reader.is_closed());
    }

    #[tokio::test]
    async fn test_reader_config() {
        let config = ReaderConfig::default();
        assert_eq!(config.channel_buffer_size, 1000);

        let custom = ReaderConfig {
            channel_buffer_size: 500,
        };
        assert_eq!(custom.channel_buffer_size, 500);
    }

    #[test]
    fn test_reader_is_clone() {
        let storage = create_test_storage();
        let (reader, _receiver) = create_reader_with_storage(ReaderConfig::default(), storage);
        let _reader2 = reader.clone();
    }
}
