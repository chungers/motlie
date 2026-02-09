//! Query reader module providing query infrastructure.
//!
//! This module follows the same pattern as fulltext::reader:
//! - Reader - handle for sending queries
//! - ReaderConfig - configuration
//! - Consumer - processes queries from channel
//! - Spawn functions for creating consumers
//!
//! Also contains the query executor traits (QueryExecutor)
//! which define how queries execute against the storage layer.

use anyhow::{Context, Result};
use std::sync::Arc;

use super::processor::Processor as GraphProcessor;
use super::query::Query;
use super::Storage;
use crate::request::RequestEnvelope;

// ============================================================================
// QueryExecutor Trait
// ============================================================================

/// Trait that all query types implement to execute themselves.
///
/// This trait defines HOW to fetch the result from storage.
/// Each query type knows how to execute its own data fetching logic.
///
/// # Design Philosophy
///
/// This follows the same pattern as mutations:
/// - Mutations: Each mutation type implements `MutationExecutor::execute()` to write to storage
/// - Queries: Each query type implements `QueryExecutor::execute()` to fetch results
///
/// Benefits:
/// - Logic lives with types, not in central implementation
/// - Easier to extend (add query = impl QueryExecutor, not modify Processor trait)
/// - Better testability (can test query.execute(storage) in isolation)
/// - Consistent with mutation pattern
#[async_trait::async_trait]
pub trait QueryExecutor: Send + Sync {
    /// The type of result this query produces
    type Output: Send;

    /// Execute this query against the storage layer
    /// Each query type knows how to fetch its own data
    async fn execute(&self, storage: &Storage) -> Result<Self::Output>;
}

// ============================================================================
// ============================================================================
// QueryRequest
// ============================================================================

pub type QueryRequest = RequestEnvelope<Query>;

async fn execute_request(processor: &GraphProcessor, mut request: QueryRequest) {
    tracing::debug!(query = %request.payload, "Processing graph query");

    let exec = processor.execute_query(&request.payload);
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

/// Configuration for the query reader
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

/// Handle for sending queries to the reader
///
/// (claude, 2026-02-07, FIXED: P2.3 - Reader holds Arc<Processor> like vector::Reader)
#[derive(Clone)]
pub struct Reader {
    sender: flume::Sender<QueryRequest>,
    /// Processor for cache-aware reads and direct query access
    processor: Arc<GraphProcessor>,
}

impl std::fmt::Debug for Reader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reader")
            .field("sender", &"<flume::Sender>")
            .field("processor", &"<Arc<Processor>>")
            .finish()
    }
}

impl Reader {
    /// Create a new Reader with the given sender.
    ///
    /// Note: Most users should use `create_reader_with_storage()` instead.
    #[doc(hidden)]
    pub(crate) fn new(sender: flume::Sender<QueryRequest>, processor: Arc<GraphProcessor>) -> Self {
        Reader { sender, processor }
    }


    /// Send a query to the reader queue.
    ///
    /// Note: Most users should use `Runnable::run()` instead.
    #[doc(hidden)]
    pub async fn send_query(&self, request: QueryRequest) -> Result<()> {
        // Keep the Processor alive for the Reader's lifetime (aligns with vector::Reader pattern).
        let _ = &self.processor;
        self.sender
            .send_async(request)
            .await
            .context("Failed to send query to reader queue")
    }

    /// Check if the reader is still active (receiver hasn't been dropped)
    pub fn is_closed(&self) -> bool {
        self.sender.is_disconnected()
    }
}

// ============================================================================
// Consumer
// ============================================================================

/// Generic consumer that processes queries using a Processor
#[doc(hidden)]
pub struct Consumer {
    receiver: flume::Receiver<QueryRequest>,
    config: ReaderConfig,
    processor: Arc<GraphProcessor>,
}

impl Consumer {
    /// Create a new Consumer
    #[doc(hidden)]
    pub fn new(
        receiver: flume::Receiver<QueryRequest>,
        config: ReaderConfig,
        processor: Arc<GraphProcessor>,
    ) -> Self {
        Self {
            receiver,
            config,
            processor,
        }
    }

    /// Process queries continuously until the channel is closed
    #[tracing::instrument(skip(self), name = "query_consumer")]
    #[doc(hidden)]
    pub async fn run(self) -> Result<()> {
        tracing::info!(config = ?self.config, "Starting query consumer");

        loop {
            // Wait for the next query (MPMC semantics via flume)
            match self.receiver.recv_async().await {
                Ok(request) => {
                    // Process the query immediately
                    self.process_query(request).await;
                }
                Err(_) => {
                    // Channel closed
                    tracing::info!("Query consumer shutting down - channel closed");
                    return Ok(());
                }
            }
        }
    }

    /// Process a single query
    #[tracing::instrument(skip(self, request), fields(query_type = %request.payload))]
    async fn process_query(&self, mut request: QueryRequest) {
        tracing::debug!(query = %request.payload, "Processing query");

        let exec = self.processor.execute_query(&request.payload);
        let result = match request.timeout {
            Some(timeout) => {
                match tokio::time::timeout(timeout, exec).await {
                    Ok(r) => r,
                    Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
                }
            }
            None => exec.await,
        };

        request.respond(result);
    }
}

/// Spawn a query consumer as a background task
#[doc(hidden)]
pub fn spawn_consumer(consumer: Consumer) -> tokio::task::JoinHandle<Result<()>> {
    tokio::spawn(async move { consumer.run().await })
}

// ============================================================================
// Path-based Consumer Functions (Test convenience)
// (claude, 2026-02-07) - Convenience wrappers for tests
// ============================================================================

use std::path::Path;
use tokio::task::JoinHandle;

/// Spawn a query consumer with path-based storage creation.
///
/// Convenience function for tests that creates storage at the given path.
/// Uses the new Processor-based infrastructure internally.
///
/// # Arguments
/// * `receiver` - Query receiver from create_reader_with_storage
/// * `config` - Reader configuration
/// * `db_path` - Path to create/open the database
///
/// # Returns
/// JoinHandle for the spawned consumer task
pub fn spawn_query_consumer(
    receiver: flume::Receiver<QueryRequest>,
    config: ReaderConfig,
    db_path: &Path,
) -> JoinHandle<Result<()>> {
    // Create storage at path
    let mut storage = Storage::readonly(db_path);
    storage.ready().expect("Failed to initialize storage");
    let storage = Arc::new(storage);

    // Create processor and consumer
    let processor = Arc::new(GraphProcessor::new(storage));
    let consumer = Consumer::new(receiver, config, processor);
    spawn_consumer(consumer)
}

/// Spawn a query consumer pool with shared Processor.
///
/// Convenience function that matches the old API signature.
/// Creates multiple consumers sharing a Processor.
///
/// # Arguments
/// * `receiver` - Query receiver from create_reader_with_storage
/// * `processor` - Shared GraphProcessor instance
/// * `num_workers` - Number of query workers to spawn
///
/// # Returns
/// Vec of JoinHandles for spawned workers
pub fn spawn_query_consumer_pool_shared(
    receiver: flume::Receiver<QueryRequest>,
    processor: Arc<GraphProcessor>,
    num_workers: usize,
) -> Vec<JoinHandle<()>> {
    spawn_consumer_pool_with_processor(receiver, processor, num_workers)
}

/// Spawn a pool of query consumers, each with its own readonly storage.
///
/// Each worker creates its own readonly Storage and Processor instance.
/// This provides isolation between workers at the cost of memory.
///
/// Note: For most use cases, `spawn_query_consumer_pool_shared` is preferred
/// since it shares memory more efficiently via Arc.
// (claude, 2026-02-07, ADDED: Fill gap - README documented but function was missing)
#[doc(hidden)]
pub fn spawn_query_consumer_pool_readonly(
    receiver: flume::Receiver<QueryRequest>,
    config: ReaderConfig,
    db_path: &Path,
    num_workers: usize,
) -> Vec<JoinHandle<()>> {
    let mut handles = Vec::with_capacity(num_workers);
    let db_path = db_path.to_path_buf();

    for worker_id in 0..num_workers {
        let receiver = receiver.clone();
        let _config = config.clone(); // Reserved for future use
        let db_path = db_path.clone();

        let handle = tokio::spawn(async move {
            tracing::info!(
                worker_id,
                "Graph query worker starting (individual readonly mode)"
            );

            // Each worker creates its own readonly Storage and Processor
            let mut storage = Storage::readonly(&db_path);
            if let Err(e) = storage.ready() {
                tracing::error!(worker_id, err = %e, "Worker failed to ready storage");
                return;
            }
            let processor = GraphProcessor::new(Arc::new(storage));

            // Process queries from shared channel
            while let Ok(request) = receiver.recv_async().await {
                tracing::debug!(worker_id, query = %request.payload, "Processing graph query");
                execute_request(&processor, request).await;
            }

            tracing::info!(worker_id, "Graph query worker shutting down");
        });

        handles.push(handle);
    }

    tracing::info!(
        num_workers,
        "Spawned graph query consumer pool (individual readonly mode)"
    );

    handles
}

/// Spawn a single query consumer with shared Processor.
///
/// Convenience function that matches the old `spawn_query_consumer_with_graph` signature.
/// Uses the Processor to process queries.
///
/// # Arguments
/// * `receiver` - Query receiver from create_reader_with_storage
/// * `config` - Reader configuration
/// * `processor` - Shared GraphProcessor instance
///
/// # Returns
/// JoinHandle for the spawned consumer task
pub fn spawn_query_consumer_with_processor(
    receiver: flume::Receiver<QueryRequest>,
    config: ReaderConfig,
    processor: Arc<GraphProcessor>,
) -> JoinHandle<Result<()>> {
    let consumer = Consumer::new(receiver, config, processor);
    spawn_consumer(consumer)
}

// ============================================================================
// Processor-based Consumer Functions (ARCH2 Pattern)
// (claude, 2026-02-07, FIXED: P2.4 - Construction helpers)
// ============================================================================

/// Create a reader with storage.
///
/// Creates a GraphProcessor internally for cache-aware reads.
///
/// # Arguments
/// * `storage` - Shared storage instance
/// * `config` - Reader configuration
///
/// # Returns
/// Reader with embedded Processor
pub fn create_reader_with_storage(storage: Arc<Storage>, config: ReaderConfig) -> (Reader, flume::Receiver<QueryRequest>) {
    let processor = Arc::new(GraphProcessor::new(storage));
    create_reader_with_processor(processor, config)
}

/// Create a reader with existing processor.
///
/// Used when GraphProcessor is shared (e.g., with Writer).
///
/// # Arguments
/// * `processor` - Shared GraphProcessor instance
/// * `config` - Reader configuration
///
/// # Returns
/// Reader with Processor reference
pub(crate) fn create_reader_with_processor(processor: Arc<GraphProcessor>, config: ReaderConfig) -> (Reader, flume::Receiver<QueryRequest>) {
    let (sender, receiver) = flume::bounded(config.channel_buffer_size);
    let reader = Reader::new(sender, processor);
    (reader, receiver)
}

/// Spawn query consumers with storage.
///
/// This is the primary public construction helper.
/// Creates GraphProcessor internally and spawns consumer workers.
///
/// # Arguments
/// * `storage` - Shared storage instance
/// * `config` - Reader configuration
/// * `num_workers` - Number of query workers to spawn
///
/// # Returns
/// Tuple of (Reader, Vec<JoinHandle>) for the spawned consumers
pub fn spawn_query_consumers_with_storage(
    storage: Arc<Storage>,
    config: ReaderConfig,
    num_workers: usize,
) -> (Reader, Vec<JoinHandle<()>>) {
    let processor = Arc::new(GraphProcessor::new(storage));
    spawn_query_consumers_with_processor(processor, config, num_workers)
}

/// Spawn query consumers with existing processor.
///
/// Used when GraphProcessor is shared (e.g., with Writer).
/// This is pub(crate) - use spawn_query_consumers_with_storage for public API.
///
/// # Arguments
/// * `processor` - Shared GraphProcessor instance
/// * `config` - Reader configuration
/// * `num_workers` - Number of query workers to spawn
///
/// # Returns
/// Tuple of (Reader, Vec<JoinHandle>) for the spawned consumers
pub(crate) fn spawn_query_consumers_with_processor(
    processor: Arc<GraphProcessor>,
    config: ReaderConfig,
    num_workers: usize,
) -> (Reader, Vec<JoinHandle<()>>) {
    let (sender, receiver) = flume::bounded(config.channel_buffer_size);
    let reader = Reader::new(sender, processor.clone());

    let mut handles = Vec::with_capacity(num_workers);
    for worker_id in 0..num_workers {
        let receiver = receiver.clone();
        let processor = processor.clone();

        let handle = tokio::spawn(async move {
            tracing::info!(worker_id, "Query worker starting (processor mode)");

            // Process queries using shared Processor
            while let Ok(request) = receiver.recv_async().await {
                tracing::debug!(worker_id, query = %request.payload, "Processing query (processor mode)");
                execute_request(&processor, request).await;
            }

            tracing::info!(worker_id, "Query worker shutting down");
        });

        handles.push(handle);
    }

    tracing::info!(
        num_workers,
        "Spawned query consumer workers (processor mode)"
    );

    (reader, handles)
}

/// Spawn consumer pool for an existing receiver with shared processor.
///
/// This is used by the unified reader to spawn graph consumers with
/// an existing channel setup.
///
/// # Arguments
/// * `receiver` - Existing query receiver
/// * `processor` - Shared GraphProcessor instance
/// * `num_workers` - Number of query workers to spawn
///
/// # Returns
/// Vec of JoinHandles for spawned workers
pub fn spawn_consumer_pool_with_processor(
    receiver: flume::Receiver<QueryRequest>,
    processor: Arc<GraphProcessor>,
    num_workers: usize,
) -> Vec<JoinHandle<()>> {
    let mut handles = Vec::with_capacity(num_workers);
    for worker_id in 0..num_workers {
        let receiver = receiver.clone();
        let processor = processor.clone();

        let handle = tokio::spawn(async move {
            tracing::info!(worker_id, "Query worker starting (processor mode)");

            while let Ok(request) = receiver.recv_async().await {
                tracing::debug!(worker_id, query = %request.payload, "Processing query (processor mode)");
                execute_request(&processor, request).await;
            }

            tracing::info!(worker_id, "Query worker shutting down");
        });

        handles.push(handle);
    }

    tracing::info!(
        num_workers,
        "Spawned query consumer pool (processor mode)"
    );

    handles
}

/// Create reader and spawn consumers with processor, also returning the processor.
///
/// Convenience helper that returns all components for subsystem integration.
///
/// # Arguments
/// * `storage` - Shared storage instance
/// * `config` - Reader configuration
/// * `num_workers` - Number of query workers to spawn
///
/// # Returns
/// Tuple of (Reader, Arc<GraphProcessor>, Vec<JoinHandle>)
pub(crate) fn create_reader_with_processor_and_spawn(
    storage: Arc<Storage>,
    config: ReaderConfig,
    num_workers: usize,
) -> (Reader, Arc<GraphProcessor>, Vec<JoinHandle<()>>) {
    let processor = Arc::new(GraphProcessor::new(storage));
    let (reader, handles) = spawn_query_consumers_with_processor(processor.clone(), config, num_workers);
    (reader, processor, handles)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_reader_closed_detection() {
        let config = ReaderConfig::default();
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("reader_test_closed");
        let mut storage = Storage::readwrite(&db_path);
        storage.ready().unwrap();
        let storage = Arc::new(storage);
        let (reader, receiver) = create_reader_with_storage(storage, config);

        assert!(!reader.is_closed());

        // Drop receiver to close channel
        drop(receiver);

        // Give tokio time to process the close
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Reader should detect channel is closed
        assert!(reader.is_closed());
    }

    #[tokio::test]
    async fn test_reader_send_operations() {
        let config = ReaderConfig::default();
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("reader_test_send");
        let mut storage = Storage::readwrite(&db_path);
        storage.ready().unwrap();
        let storage = Arc::new(storage);
        let (reader, _receiver) = create_reader_with_storage(storage, config);

        // Test that reader is not closed initially
        assert!(!reader.is_closed());

        // We can't test actual sending without a consumer running,
        // but we've verified the reader is constructed properly
    }
}
