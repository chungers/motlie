//! Vector query reader module providing query infrastructure.
//!
//! This module follows the same pattern as graph::reader:
//! - Reader - handle for sending queries
//! - ReaderConfig - configuration
//! - Consumer - processes queries from channel (Processor-backed)
//! - Spawn functions for creating consumers
//!
//! Uses flume for MPMC (multi-producer, multi-consumer) channels,
//! enabling multiple query consumers to process queries in parallel.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────┐     MPMC (flume)     ┌──────────────────────┐
//! │ Reader  │──────────────────────│ Consumer (N workers) │
//! └─────────┘       Query          └──────────────────────┘
//!                                           │
//!                                           ▼
//!                                  ┌────────────────────┐
//!                                  │     Processor      │
//!                                  │  (Storage + HNSW)  │
//!                                  └────────────────────┘
//! ```
//!
//! The Reader sends queries through an MPMC channel. Consumer workers
//! process queries using Processor, which provides both Storage access
//! for simple lookups and HNSW index operations for SearchKNN.

use std::sync::Arc;

use anyhow::{Context, Result};

use super::processor::Processor;
use super::query::QueryRequest;

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
/// multiple consumer workers to process queries in parallel.
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
}

impl Reader {
    /// Create a new Reader with the given sender.
    pub(crate) fn new(sender: flume::Sender<QueryRequest>) -> Self {
        Self { sender }
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
}

impl std::fmt::Debug for Reader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reader")
            .field("sender", &self.sender)
            .finish()
    }
}

/// Create a Reader.
///
/// This is the recommended way to create a Reader.
///
/// # Returns
///
/// Tuple of (Reader, flume::Receiver<QueryRequest>) - spawn consumers with the receiver.
///
/// # Example
///
/// ```rust,ignore
/// let storage = Arc::new(Storage::readwrite(path));
/// let (reader, receiver) = create_reader(ReaderConfig::default());
/// spawn_query_consumers_with_storage_autoreg(receiver, config, storage, 2);
///
/// // Use SearchKNN::run() for searches
/// let results = SearchKNN::new(&embedding, query, 10)
///     .run(&reader, timeout)
///     .await?;
/// ```
pub fn create_reader_with_storage(config: ReaderConfig) -> (Reader, flume::Receiver<QueryRequest>) {
    let (sender, receiver) = flume::bounded(config.channel_buffer_size);
    let reader = Reader::new(sender);
    (reader, receiver)
}

/// Create a Reader (internal alias for create_reader_with_storage).
pub(crate) fn create_reader(config: ReaderConfig) -> (Reader, flume::Receiver<QueryRequest>) {
    create_reader_with_storage(config)
}

// ============================================================================
// Consumer (Processor-backed)
// ============================================================================

/// Consumer that processes vector queries from a channel.
///
/// Multiple consumers can be spawned to process queries in parallel
/// thanks to flume's MPMC semantics.
///
/// The Consumer holds a Processor reference, enabling both simple lookups
/// (via Storage) and SearchKNN queries (via HNSW index operations).
pub(crate) struct Consumer {
    receiver: flume::Receiver<QueryRequest>,
    config: ReaderConfig,
    processor: Arc<Processor>,
}

impl Consumer {
    /// Create a new Consumer with Processor.
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
    #[tracing::instrument(skip(self), name = "vector_query_consumer")]
    pub async fn run(self) -> Result<()> {
        tracing::info!(config = ?self.config, "Starting vector query consumer");

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
}

/// Spawn a query consumer as a tokio task.
///
/// Returns a JoinHandle that resolves when the consumer completes.
pub(crate) fn spawn_consumer(consumer: Consumer) -> tokio::task::JoinHandle<Result<()>> {
    tokio::spawn(async move { consumer.run().await })
}

/// Spawn multiple query consumers for parallel processing.
///
/// This is the internal way to set up vector query handling.
/// For public API, use `spawn_query_consumers_with_storage_autoreg`.
///
/// # Arguments
///
/// * `receiver` - The flume receiver to clone for each consumer
/// * `config` - Configuration for each consumer
/// * `processor` - Shared processor reference (includes Storage + HNSW)
/// * `count` - Number of consumers to spawn
///
/// # Returns
///
/// Vector of JoinHandles for all spawned consumers
pub(crate) fn spawn_consumers(
    receiver: flume::Receiver<QueryRequest>,
    config: ReaderConfig,
    processor: Arc<Processor>,
    count: usize,
) -> Vec<tokio::task::JoinHandle<Result<()>>> {
    (0..count)
        .map(|_| {
            let consumer = Consumer::new(receiver.clone(), config.clone(), processor.clone());
            spawn_consumer(consumer)
        })
        .collect()
}

/// Spawn multiple query consumers with storage and registry.
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
    spawn_consumers(receiver, config, processor, count)
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
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::Storage;
    use tempfile::tempdir;

    fn create_test_storage() -> Arc<Storage> {
        let dir = tempdir().unwrap();
        let mut storage = Storage::readwrite(dir.path());
        storage.ready().unwrap();
        Arc::new(storage)
    }

    #[tokio::test]
    async fn test_reader_creation() {
        let _storage = create_test_storage();
        let (reader, _receiver) = create_reader_with_storage(ReaderConfig::default());
        assert!(!reader.is_closed());
    }

    #[tokio::test]
    async fn test_reader_channel_closed() {
        let _storage = create_test_storage();
        let (reader, receiver) = create_reader_with_storage(ReaderConfig::default());
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
        let _storage = create_test_storage();
        let (reader, _receiver) = create_reader_with_storage(ReaderConfig::default());
        let _reader2 = reader.clone();
    }
}
