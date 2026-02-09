//! Fulltext reader module providing query consumers and reader infrastructure.
//!
//! This module follows the same pattern as the top-level reader.rs:
//! - Reader - handle for sending queries
//! - ReaderConfig - configuration
//! - Consumer - processes queries from channel
//! - Spawn functions for creating consumers
//!
//! Also contains the query executor traits (QueryExecutor, Processor)
//! which define how queries execute against the storage layer.

use anyhow::Result;
use std::path::Path;
use std::sync::Arc;

use super::{Index, Storage};
use crate::request::RequestEnvelope;

// ============================================================================
// QueryExecutor Trait
// ============================================================================

/// Trait that fulltext query types implement to execute themselves.
///
/// This follows the same pattern as graph queries (db::query::QueryExecutor):
/// - Each query type knows how to fetch its own data from Storage
/// - Logic lives with types, not in central implementation
/// - Easier to extend (add query = impl QueryExecutor)
#[async_trait::async_trait]
pub trait QueryExecutor: Send + Sync {
    /// The type of result this query produces
    type Output: Send;

    /// Execute this query against the fulltext storage layer.
    /// Each query type knows how to fetch its own data.
    async fn execute(&self, storage: &Storage) -> Result<Self::Output>;
}

// ============================================================================
// Processor Trait - bridges Index to QueryExecutor
// ============================================================================

/// Trait for fulltext query processors (mirrors graph::query::Processor).
/// Allows queries to access the underlying Storage.
pub trait Processor {
    /// Get a reference to the underlying Storage
    fn storage(&self) -> &Storage;
}

/// Implement Processor for Index (mirrors graph::Processor implementing reader::Processor)
impl Processor for Index {
    fn storage(&self) -> &Storage {
        Index::storage(self)
    }
}

// ============================================================================
// QueryRequest
// ============================================================================

pub type QueryRequest = RequestEnvelope<super::query::Search>;

async fn execute_request(processor: &Index, mut request: QueryRequest) {
    tracing::debug!(query = %request.payload, "Processing fulltext query");

    let exec = request.payload.execute(processor.storage());
    let result = match request.timeout {
        Some(timeout) => match tokio::time::timeout(timeout, exec).await {
            Ok(r) => r,
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
        },
        None => exec.await,
    };

    request.respond(result);
}

// ============================================================================
// Reader
// ============================================================================

/// Handle for sending fulltext queries to the reader queue.
/// This is the fulltext equivalent of the graph Reader.
#[derive(Debug, Clone)]
pub struct Reader {
    sender: flume::Sender<QueryRequest>,
}

impl Reader {
    /// Create a new Reader with the given sender.
    ///
    /// Note: Most users should use `create_query_reader()` instead.
    #[doc(hidden)]
    pub fn new(sender: flume::Sender<QueryRequest>) -> Self {
        Reader { sender }
    }

    /// Send a query to the reader queue.
    ///
    /// Note: Most users should use `Runnable::run()` instead.
    #[doc(hidden)]
    pub async fn send_query(&self, request: QueryRequest) -> Result<()> {
        self.sender
            .send_async(request)
            .await
            .map_err(|_| anyhow::anyhow!("Failed to send query to fulltext reader queue"))
    }

    /// Check if the reader is still active
    pub fn is_closed(&self) -> bool {
        self.sender.is_disconnected()
    }
}

/// Configuration for the fulltext query reader
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

/// Create a new fulltext query reader and receiver pair
#[doc(hidden)]
pub fn create_query_reader(
    config: ReaderConfig,
) -> (Reader, flume::Receiver<QueryRequest>) {
    let (sender, receiver) = flume::bounded(config.channel_buffer_size);
    let reader = Reader::new(sender);
    (reader, receiver)
}

// ============================================================================
// Consumer
// ============================================================================

/// Consumer that processes fulltext queries using an Index.
/// This is the fulltext equivalent of the graph query Consumer.
#[doc(hidden)]
pub struct Consumer {
    receiver: flume::Receiver<QueryRequest>,
    config: ReaderConfig,
    processor: Index,
}

impl Consumer {
    /// Create a new Consumer
    #[doc(hidden)]
    pub fn new(
        receiver: flume::Receiver<QueryRequest>,
        config: ReaderConfig,
        processor: Index,
    ) -> Self {
        Self {
            receiver,
            config,
            processor,
        }
    }

    /// Process queries continuously until the channel is closed
    #[tracing::instrument(skip(self), name = "fulltext_query_consumer")]
    #[doc(hidden)]
    pub async fn run(self) -> Result<()> {
        tracing::info!(
            config = ?self.config,
            "Starting fulltext query consumer"
        );

        loop {
            match self.receiver.recv_async().await {
                Ok(request) => {
                    execute_request(&self.processor, request).await;
                }
                Err(_) => {
                    tracing::info!("Fulltext query consumer shutting down - channel closed");
                    return Ok(());
                }
            }
        }
    }
}

/// Spawn a fulltext query consumer as a background task
#[doc(hidden)]
pub fn spawn_consumer(consumer: Consumer) -> tokio::task::JoinHandle<Result<()>> {
    tokio::spawn(async move { consumer.run().await })
}

/// Create a fulltext query consumer
#[doc(hidden)]
pub fn create_query_consumer(
    receiver: flume::Receiver<QueryRequest>,
    config: ReaderConfig,
    processor: Index,
) -> Consumer {
    Consumer::new(receiver, config, processor)
}

/// Spawn a fulltext query consumer with a new readonly Index
///
/// Uses readonly Storage since query consumers only need read access.
/// Multiple consumers can share the same index path.
#[doc(hidden)]
pub fn spawn_query_consumer(
    receiver: flume::Receiver<QueryRequest>,
    config: ReaderConfig,
    index_path: &Path,
) -> tokio::task::JoinHandle<Result<()>> {
    // Use readonly mode for query consumers - they don't need write access
    let mut storage = Storage::readonly(index_path);
    storage.ready().expect("Failed to ready readonly storage");
    let processor = Index::new(Arc::new(storage));
    let consumer = create_query_consumer(receiver, config, processor);
    spawn_consumer(consumer)
}

/// Spawn a pool of fulltext query consumers sharing an Arc<Index>.
///
/// This is the recommended approach for multiple query consumers because
/// all workers share the same underlying Tantivy Index via `Arc<Index>`.
///
/// # Example
/// ```no_run
/// use motlie_db::fulltext::{Storage, Index, ReaderConfig, create_query_reader, spawn_query_consumer_pool_shared};
/// use std::sync::Arc;
/// use std::path::Path;
///
/// # async fn example() -> anyhow::Result<()> {
/// let index_path = Path::new("/path/to/fulltext_index");
///
/// // Create a shared readonly Index (follows graph::Graph pattern)
/// let mut storage = Storage::readonly(index_path);
/// storage.ready()?;
/// let index = Index::new(Arc::new(storage));
///
/// // Create reader channel
/// let config = ReaderConfig { channel_buffer_size: 100 };
/// let (reader, receiver) = create_query_reader(config);
///
/// // Spawn multiple consumers sharing the same Index
/// let num_workers = 4;
/// let handles = spawn_query_consumer_pool_shared(
///     receiver,
///     Arc::new(index),
///     num_workers,
/// );
///
/// // All workers share the same Tantivy Index via Arc<Index>
/// # Ok(())
/// # }
/// ```
#[doc(hidden)]
pub fn spawn_query_consumer_pool_shared(
    receiver: flume::Receiver<QueryRequest>,
    index: Arc<Index>,
    num_workers: usize,
) -> Vec<tokio::task::JoinHandle<()>> {
    let mut handles = Vec::with_capacity(num_workers);

    for worker_id in 0..num_workers {
        let receiver = receiver.clone();
        let index = index.clone(); // Cheap Arc clone - shares Tantivy Index

        let handle = tokio::spawn(async move {
            tracing::info!(
                worker_id,
                "Fulltext query worker starting (shared Index mode)"
            );

            // Process queries from shared channel
            // All workers share the same Tantivy Index via Arc<Index>
            while let Ok(request) = receiver.recv_async().await {
                execute_request(&*index, request).await;
            }

            tracing::info!(worker_id, "Fulltext query worker shutting down");
        });

        handles.push(handle);
    }

    handles
}

/// Spawn a pool of fulltext query consumers, each with their own readonly Index.
///
/// Each worker creates its own readonly Storage and Index.
/// This allows independent instances but they all read from the same index.
///
/// Note: For most use cases, `spawn_query_consumer_pool_shared` is preferred
/// since it shares memory more efficiently via Arc.
#[doc(hidden)]
pub fn spawn_query_consumer_pool_readonly(
    receiver: flume::Receiver<QueryRequest>,
    config: ReaderConfig,
    index_path: &Path,
    num_workers: usize,
) -> Vec<tokio::task::JoinHandle<()>> {
    let mut handles = Vec::with_capacity(num_workers);
    let index_path = index_path.to_path_buf();

    for worker_id in 0..num_workers {
        let receiver = receiver.clone();
        let _config = config.clone(); // Reserved for future use
        let index_path = index_path.clone();

        let handle = tokio::spawn(async move {
            tracing::info!(
                worker_id,
                "Fulltext query worker starting (individual readonly mode)"
            );

            // Each worker creates its own readonly Storage and Index
            let mut storage = Storage::readonly(&index_path);
            if let Err(e) = storage.ready() {
                tracing::error!(worker_id, err = %e, "Worker failed to ready storage");
                return;
            }
            let index = Index::new(Arc::new(storage));

            // Process queries from shared channel
            while let Ok(request) = receiver.recv_async().await {
                execute_request(&index, request).await;
            }

            tracing::info!(worker_id, "Fulltext query worker shutting down");
        });

        handles.push(handle);
    }

    handles
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_reader_closed_detection() {
        let config = ReaderConfig::default();
        let (reader, receiver) = create_query_reader(config);

        assert!(!reader.is_closed());

        // Drop receiver to close channel
        drop(receiver);

        // Give tokio time to process the close
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Reader should detect channel is closed
        assert!(reader.is_closed());
    }

    #[tokio::test]
    async fn test_reader_config_default() {
        let config = ReaderConfig::default();
        assert_eq!(config.channel_buffer_size, 1000);
    }

    #[tokio::test]
    async fn test_reader_creation() {
        let config = ReaderConfig {
            channel_buffer_size: 50,
        };
        let (reader, _receiver) = create_query_reader(config);

        // Reader should not be closed when receiver exists
        assert!(!reader.is_closed());
    }
}
