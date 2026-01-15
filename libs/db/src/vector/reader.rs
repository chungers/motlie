//! Vector query reader module providing query infrastructure.
//!
//! This module follows the same pattern as graph::reader:
//! - Reader - handle for sending queries
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
//!                                       │
//!                                       ▼
//!                                  ┌──────────┐
//!                                  │ Storage  │
//!                                  └──────────┘
//! ```
//!
//! # SearchKNN Support
//!
//! For queries that need HNSW search (SearchKNN), use `spawn_consumers_with_processor`
//! which provides access to the Processor for index operations. This bundles
//! Storage + Processor together, eliminating ad-hoc plumbing at call sites.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};

use super::embedding::Embedding;
use super::processor::{Processor, SearchResult};
use super::query::{Query, SearchKNN, SearchKNNDispatch};
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
/// multiple consumer workers to process queries in parallel.
///
/// # Example
///
/// ```rust,ignore
/// use motlie_db::vector::{Reader, GetVector};
/// use std::time::Duration;
///
/// // Create reader and consumers
/// let (reader, receiver) = create_reader(ReaderConfig::default());
///
/// // Execute a query
/// let query = GetVector::new(embedding_code, id);
/// let result = query.run(&reader, Duration::from_secs(5)).await?;
/// ```
#[derive(Debug, Clone)]
pub struct Reader {
    sender: flume::Sender<Query>,
}

impl Reader {
    /// Create a new Reader with the given sender.
    pub fn new(sender: flume::Sender<Query>) -> Self {
        Self { sender }
    }

    /// Send a query to the reader queue.
    ///
    /// This is typically called by the Runnable trait implementation.
    pub async fn send_query(&self, query: Query) -> Result<()> {
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

/// Create a new query reader and receiver pair.
pub fn create_reader(config: ReaderConfig) -> (Reader, flume::Receiver<Query>) {
    let (sender, receiver) = flume::bounded(config.channel_buffer_size);
    let reader = Reader::new(sender);
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
    receiver: flume::Receiver<Query>,
    config: ReaderConfig,
    storage: Arc<Storage>,
}

impl Consumer {
    /// Create a new Consumer.
    pub fn new(
        receiver: flume::Receiver<Query>,
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
                Ok(query) => {
                    self.process_query(query).await;
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
    #[tracing::instrument(skip(self), fields(query_type = %query))]
    async fn process_query(&self, query: Query) {
        tracing::debug!(query = %query, "Processing vector query");
        query.process(&self.storage).await;
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
    receiver: flume::Receiver<Query>,
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
pub struct ProcessorConsumer {
    receiver: flume::Receiver<Query>,
    config: ReaderConfig,
    processor: Arc<Processor>,
}

impl ProcessorConsumer {
    /// Create a new ProcessorConsumer.
    pub fn new(
        receiver: flume::Receiver<Query>,
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
                Ok(query) => {
                    self.process_query(query).await;
                }
                Err(_) => {
                    tracing::info!("Vector query consumer shutting down - channel closed");
                    return Ok(());
                }
            }
        }
    }

    /// Process a single query.
    #[tracing::instrument(skip(self), fields(query_type = %query))]
    async fn process_query(&self, query: Query) {
        tracing::debug!(query = %query, "Processing vector query");
        query.process(self.processor.storage()).await;
    }

    /// Get the processor reference (for SearchKNN dispatch setup).
    pub fn processor(&self) -> &Arc<Processor> {
        &self.processor
    }
}

/// Spawn a Processor-backed query consumer as a tokio task.
pub fn spawn_consumer_with_processor(
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
pub fn spawn_consumers_with_processor(
    receiver: flume::Receiver<Query>,
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

// ============================================================================
// SearchReader - High-level API for SearchKNN
// ============================================================================

/// High-level reader for SearchKNN queries.
///
/// `SearchReader` bundles a `Reader` with an `Arc<Processor>`, providing
/// a simple API for HNSW nearest neighbor search without ad-hoc processor
/// plumbing at call sites.
///
/// # Example
///
/// ```rust,ignore
/// // Setup
/// let (reader, receiver) = create_reader(ReaderConfig::default());
/// let search_reader = SearchReader::new(reader, processor);
///
/// // Spawn consumers
/// spawn_consumers_with_processor(receiver, config, processor, 4);
///
/// // Simple search API - no dispatch construction needed
/// let results = search_reader
///     .search_knn(embedding, query_vec, 10, 100, Duration::from_secs(5))
///     .await?;
/// ```
#[derive(Clone)]
pub struct SearchReader {
    reader: Reader,
    processor: Arc<Processor>,
}

impl SearchReader {
    /// Create a new SearchReader.
    pub fn new(reader: Reader, processor: Arc<Processor>) -> Self {
        Self { reader, processor }
    }

    /// Perform K-nearest neighbor search.
    ///
    /// This is the ergonomic API for SearchKNN - no dispatch construction needed.
    ///
    /// # Arguments
    ///
    /// * `embedding` - Embedding space reference
    /// * `query` - Query vector
    /// * `k` - Number of results to return
    /// * `ef` - Search expansion factor (higher = more accurate, slower)
    /// * `timeout` - Query timeout
    ///
    /// # Returns
    ///
    /// Vector of `SearchResult` structs, sorted by distance (ascending).
    pub async fn search_knn(
        &self,
        embedding: &Embedding,
        query: Vec<f32>,
        k: usize,
        ef: usize,
        timeout: Duration,
    ) -> Result<Vec<SearchResult>> {
        let params = SearchKNN::new(embedding.code(), query, k).with_ef(ef);
        let (dispatch, rx) = SearchKNNDispatch::new(params, timeout, self.processor.clone());

        self.reader
            .send_query(Query::SearchKNN(dispatch))
            .await
            .context("Failed to send SearchKNN query")?;

        rx.await
            .context("SearchKNN query was cancelled")?
            .context("SearchKNN query failed")
    }

    /// Get the underlying reader for other query types.
    pub fn reader(&self) -> &Reader {
        &self.reader
    }

    /// Get the processor for direct operations.
    pub fn processor(&self) -> &Arc<Processor> {
        &self.processor
    }

    /// Check if the reader is still active.
    pub fn is_closed(&self) -> bool {
        self.reader.is_closed()
    }
}

impl std::fmt::Debug for SearchReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchReader")
            .field("reader", &self.reader)
            .field("processor", &"<Arc<Processor>>")
            .finish()
    }
}

/// Create a SearchReader along with a reader and receiver.
///
/// Convenience function that creates all components needed for SearchKNN.
///
/// # Returns
///
/// Tuple of (SearchReader, flume::Receiver<Query>) - spawn consumers with the receiver.
pub fn create_search_reader(
    config: ReaderConfig,
    processor: Arc<Processor>,
) -> (SearchReader, flume::Receiver<Query>) {
    let (reader, receiver) = create_reader(config);
    let search_reader = SearchReader::new(reader, processor);
    (search_reader, receiver)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_reader_creation() {
        let (reader, _receiver) = create_reader(ReaderConfig::default());
        assert!(!reader.is_closed());
    }

    #[tokio::test]
    async fn test_reader_channel_closed() {
        let (reader, receiver) = create_reader(ReaderConfig::default());
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
        let (reader, _receiver) = create_reader(ReaderConfig::default());
        let _reader2 = reader.clone();
    }
}
