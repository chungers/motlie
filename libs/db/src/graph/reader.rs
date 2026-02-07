//! Query reader module providing query infrastructure.
//!
//! This module follows the same pattern as fulltext::reader:
//! - Reader - handle for sending queries
//! - ReaderConfig - configuration
//! - Consumer - processes queries from channel
//! - Spawn functions for creating consumers
//!
//! Also contains the query executor traits (QueryExecutor, QueryWithTimeout, Processor, QueryProcessor)
//! which define how queries execute against the storage layer.

use anyhow::{Context, Result};
use std::sync::Arc;
use std::time::Duration;

use super::processor::Processor as GraphProcessor;
use super::query::Query;
use super::Storage;

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

    /// Get the timeout for this query
    fn timeout(&self) -> Duration;
}

// ============================================================================
// QueryWithTimeout Trait - Blanket implementation for QueryExecutor
// ============================================================================

/// Trait for queries that produce results with timeout handling.
/// Note: This trait is automatically implemented for all QueryExecutor types.
#[async_trait::async_trait]
pub trait QueryWithTimeout: Send + Sync {
    /// The type of result this query produces
    type ResultType: Send;

    /// Execute the query with timeout and return the result
    async fn result<P: Processor>(&self, processor: &P) -> Result<Self::ResultType>;

    /// Get the timeout for this query
    fn timeout(&self) -> Duration;
}

/// Blanket implementation: any QueryExecutor automatically gets QueryWithTimeout
#[async_trait::async_trait]
impl<T: QueryExecutor> QueryWithTimeout for T {
    type ResultType = T::Output;

    async fn result<P: Processor>(&self, processor: &P) -> Result<Self::ResultType> {
        let result = tokio::time::timeout(self.timeout(), self.execute(processor.storage())).await;

        match result {
            Ok(r) => r,
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", self.timeout())),
        }
    }

    fn timeout(&self) -> Duration {
        QueryExecutor::timeout(self)
    }
}

// ============================================================================
// Processor Trait - bridges Graph to QueryExecutor
// ============================================================================

/// Trait for processing different types of queries.
///
/// This trait provides access to storage. Query types implement QueryExecutor
/// to execute themselves against storage, following the same pattern as mutations.
pub trait Processor: Send + Sync {
    /// Get access to the underlying storage
    /// Query types use this to execute themselves via QueryExecutor::execute()
    fn storage(&self) -> &Storage;
}

// ============================================================================
// QueryProcessor Trait
// ============================================================================

/// Trait for processing queries without needing to know the result type.
/// Allows the Consumer to process queries polymorphically.
#[async_trait::async_trait]
pub trait QueryProcessor: Send {
    /// Process the query and send the result (consumes self)
    async fn process_and_send<P: Processor>(self, processor: &P);
}

/// Macro to implement QueryProcessor for query types.
/// Reduces boilerplate by using the QueryWithTimeout::result method.
#[macro_export]
macro_rules! impl_query_processor {
    ($($query_type:ty),+ $(,)?) => {
        $(
            #[async_trait::async_trait]
            impl $crate::graph::reader::QueryProcessor for $query_type {
                async fn process_and_send<P: $crate::graph::reader::Processor>(self, processor: &P) {
                    use $crate::graph::reader::QueryWithTimeout;
                    let result = self.result(processor).await;
                    self.send_result(result);
                }
            }
        )+
    };
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
    sender: flume::Sender<Query>,
    /// Processor for cache-aware reads and direct query access
    processor: Option<Arc<GraphProcessor>>,
}

impl std::fmt::Debug for Reader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reader")
            .field("sender", &"<flume::Sender>")
            .field("processor", &self.processor.as_ref().map(|_| "<Arc<Processor>>"))
            .finish()
    }
}

impl Reader {
    /// Create a new Reader with the given sender.
    ///
    /// Note: Most users should use `create_query_reader()` instead.
    #[doc(hidden)]
    pub fn new(sender: flume::Sender<Query>) -> Self {
        Reader { sender, processor: None }
    }

    /// Create a new Reader with processor for cache-aware reads.
    /// (claude, 2026-02-07, FIXED: P2.3 - Primary construction with Processor)
    pub(crate) fn with_processor(sender: flume::Sender<Query>, processor: Arc<GraphProcessor>) -> Self {
        Reader { sender, processor: Some(processor) }
    }

    /// Get the processor if configured.
    /// (claude, 2026-02-07, FIXED: P2.3 - Processor accessor like vector::Reader:118)
    pub(crate) fn processor(&self) -> Option<&Arc<GraphProcessor>> {
        self.processor.as_ref()
    }

    /// Send a query to the reader queue.
    ///
    /// Note: Most users should use `Runnable::run()` instead.
    #[doc(hidden)]
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

/// Create a new query reader and receiver pair
#[doc(hidden)]
pub fn create_query_reader(config: ReaderConfig) -> (Reader, flume::Receiver<Query>) {
    let (sender, receiver) = flume::bounded(config.channel_buffer_size);
    let reader = Reader::new(sender);
    (reader, receiver)
}

// ============================================================================
// Consumer
// ============================================================================

/// Generic consumer that processes queries using a Processor
#[doc(hidden)]
pub struct Consumer<P: Processor> {
    receiver: flume::Receiver<Query>,
    config: ReaderConfig,
    processor: P,
}

impl<P: Processor> Consumer<P> {
    /// Create a new Consumer
    #[doc(hidden)]
    pub fn new(receiver: flume::Receiver<Query>, config: ReaderConfig, processor: P) -> Self {
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
                Ok(query) => {
                    // Process the query immediately
                    self.process_query(query).await;
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
    #[tracing::instrument(skip(self), fields(query_type = %query))]
    async fn process_query(&self, query: Query) {
        tracing::debug!(query = %query, "Processing query");
        query.process_and_send(&self.processor).await;
    }
}

/// Spawn a query consumer as a background task
#[doc(hidden)]
pub fn spawn_consumer<P: Processor + 'static>(
    consumer: Consumer<P>,
) -> tokio::task::JoinHandle<Result<()>> {
    tokio::spawn(async move { consumer.run().await })
}

// ============================================================================
// Graph-specific Consumer Functions (Legacy)
// ============================================================================
// These functions use the deprecated Graph struct for backward compatibility.
// New code should use spawn_query_consumers_with_storage() or
// spawn_query_consumers_with_processor() instead.

use std::path::Path;
use tokio::task::JoinHandle;

#[allow(deprecated)]
use super::Graph;

/// Create a new query consumer for the graph
#[doc(hidden)]
#[allow(deprecated)]
pub fn create_query_consumer(
    receiver: flume::Receiver<Query>,
    config: ReaderConfig,
    db_path: &Path,
) -> Consumer<Graph> {
    let mut storage = Storage::readonly(db_path);
    storage.ready().expect("Failed to ready storage");
    let storage = Arc::new(storage);
    let processor = Graph::new(storage);
    Consumer::new(receiver, config, processor)
}

/// Spawn a query consumer as a background task
#[doc(hidden)]
#[allow(deprecated)]
pub fn spawn_query_consumer(
    receiver: flume::Receiver<Query>,
    config: ReaderConfig,
    db_path: &Path,
) -> JoinHandle<Result<()>> {
    let consumer = create_query_consumer(receiver, config, db_path);
    spawn_consumer(consumer)
}

/// Create a query consumer with readwrite storage (for testing TransactionDB concurrency)
///
/// Unlike `create_query_consumer` which opens readonly storage, this opens readwrite storage.
/// This allows testing whether RocksDB TransactionDB supports multiple instances accessing
/// the same database path concurrently.
///
/// # Arguments
/// * `receiver` - Channel to receive queries from
/// * `config` - Reader configuration
/// * `db_path` - Path to the database
///
/// # Returns
/// A Consumer configured with readwrite storage
#[doc(hidden)]
#[allow(deprecated)]
pub fn create_query_consumer_readwrite(
    receiver: flume::Receiver<Query>,
    config: ReaderConfig,
    db_path: &Path,
) -> Consumer<Graph> {
    let mut storage = Storage::readwrite(db_path);
    storage.ready().expect("Failed to ready readwrite storage");
    let storage = Arc::new(storage);
    let processor = Graph::new(storage);
    Consumer::new(receiver, config, processor)
}

/// Spawn a query consumer with readwrite storage as a background task
///
/// This is the readwrite variant of `spawn_query_consumer`.
/// Use this to test concurrent access patterns with TransactionDB.
///
/// # Arguments
/// * `receiver` - Channel to receive queries from
/// * `config` - Reader configuration
/// * `db_path` - Path to the database
///
/// # Returns
/// A JoinHandle for the spawned consumer task
#[doc(hidden)]
#[allow(deprecated)]
pub fn spawn_query_consumer_readwrite(
    receiver: flume::Receiver<Query>,
    config: ReaderConfig,
    db_path: &Path,
) -> JoinHandle<Result<()>> {
    let consumer = create_query_consumer_readwrite(receiver, config, db_path);
    spawn_consumer(consumer)
}

/// Spawn a query consumer using an existing Graph instance
///
/// This allows multiple query consumers to share a single Storage/TransactionDB instance.
/// Use this when you want multiple readers to access the same readwrite TransactionDB
/// without opening multiple database instances (which RocksDB doesn't support).
///
/// This is the correct way to have concurrent readers on readwrite storage, since
/// RocksDB TransactionDB:
/// - Does NOT support multiple instances on the same path (lock file prevents this)
/// - DOES support thread-safe access from multiple threads to a single instance
///
/// # Arguments
/// * `receiver` - Channel to receive queries from
/// * `config` - Reader configuration
/// * `graph` - Shared Graph instance (wrapping the Storage/TransactionDB)
///
/// # Example
/// ```no_run
/// use motlie_db::graph::{Graph, Storage};
/// use motlie_db::graph::reader::{Reader, ReaderConfig, spawn_query_consumer_with_graph};
/// use std::sync::Arc;
/// use std::time::Duration;
///
/// # async fn example() -> anyhow::Result<()> {
/// let mut storage = Storage::readwrite(&std::path::PathBuf::from("/tmp/db"));
/// storage.ready()?;
/// let graph = Arc::new(Graph::new(Arc::new(storage)));
///
/// // Spawn multiple readers sharing the same graph/TransactionDB
/// for _ in 0..4 {
///     let (reader, rx) = {
///         let config = ReaderConfig { channel_buffer_size: 10 };
///         let (sender, receiver) = flume::bounded(config.channel_buffer_size);
///         let reader = Reader::new(sender);
///         (reader, receiver)
///     };
///     spawn_query_consumer_with_graph(rx, ReaderConfig { channel_buffer_size: 10 }, graph.clone());
/// }
/// # Ok(())
/// # }
/// ```
#[doc(hidden)]
#[allow(deprecated)]
pub fn spawn_query_consumer_with_graph(
    receiver: flume::Receiver<Query>,
    config: ReaderConfig,
    graph: Arc<Graph>,
) -> JoinHandle<Result<()>> {
    #[allow(deprecated)]
    let consumer = Consumer::new(receiver, config, (*graph).clone());
    spawn_consumer(consumer)
}

/// Spawn N query consumer workers sharing a single readwrite TransactionDB.
///
/// **This is the RECOMMENDED approach for multi-threaded query processing** in
/// single-process applications requiring high consistency.
///
/// All workers share the same TransactionDB instance via `Arc<Graph>`, providing:
/// - ✅ **99%+ read-after-write consistency** (vs 25-30% for readonly mode)
/// - ✅ **Immediate visibility** - All threads see same memtable
/// - ✅ **Thread-safe** - TransactionDB has internal MVCC locking
/// - ✅ **No reopen/catch-up overhead** - Direct memtable access
/// - ✅ **Memory efficient** - Single DB instance shared across workers
///
/// # Storage Mode
///
/// **CONFIRMED**: This uses **readwrite Storage** (TransactionDB mode).
/// The Graph must be created from `Storage::readwrite()` and wrapped in `Arc`.
///
/// # Arguments
/// * `receiver` - Shared flume receiver (MPMC channel supports multiple receivers)
/// * `graph` - Shared Graph wrapping readwrite TransactionDB (via Arc)
/// * `num_workers` - Number of worker threads to spawn
///
/// # Returns
/// Vector of JoinHandles for all worker threads
///
/// # Thread Safety
///
/// RocksDB TransactionDB is fully thread-safe for concurrent access. The constraint
/// is that you CANNOT open multiple TransactionDB instances on the same path (lock
/// file prevents this). Instead, you MUST share one instance via Arc.
///
/// # Performance
///
/// From `libs/db/docs/concurrency-and-storage-modes.md`:
/// - **Success Rate**: 99%+ (vs 25-30% for readonly, 40-45% for secondary)
/// - **Query Latency**: Same as other modes
/// - **Memory**: Single instance footprint (vs N× instances for readonly)
/// - **Consistency**: Immediate (vs eventually consistent for readonly/secondary)
///
/// # Example
/// ```no_run
/// use motlie_db::graph::{Storage, Graph};
/// use motlie_db::graph::reader::{create_query_reader, spawn_query_consumer_pool_shared, ReaderConfig};
/// use std::sync::Arc;
/// use std::path::Path;
///
/// # async fn example() -> anyhow::Result<()> {
/// let db_path = Path::new("/path/to/db");
///
/// // Create ONE shared readwrite Storage (TransactionDB)
/// let mut storage = Storage::readwrite(db_path);
/// storage.ready()?;
/// let storage = Arc::new(storage);
/// let graph = Arc::new(Graph::new(storage));
///
/// // Create reader channel
/// let config = ReaderConfig { channel_buffer_size: 100 };
/// let (reader, receiver) = create_query_reader(config.clone());
///
/// // Spawn worker pool sharing the graph
/// let num_workers = 4; // Or use num_cpus::get() if you have that dependency
/// let handles = spawn_query_consumer_pool_shared(
///     receiver,
///     graph.clone(),
///     num_workers,
/// );
///
/// // All workers share the same TransactionDB via Arc<Graph>
/// # Ok(())
/// # }
/// ```
#[doc(hidden)]
#[allow(deprecated)]
pub fn spawn_query_consumer_pool_shared(
    receiver: flume::Receiver<Query>,
    graph: Arc<Graph>,
    num_workers: usize,
) -> Vec<JoinHandle<()>> {
    let mut handles = Vec::with_capacity(num_workers);

    for worker_id in 0..num_workers {
        let receiver = receiver.clone();
        let graph = graph.clone();  // Cheap Arc clone - shares TransactionDB

        let handle = tokio::spawn(async move {
            tracing::info!(worker_id, "Query worker starting (shared TransactionDB mode)");

            // Process queries from shared channel
            // All workers share the same TransactionDB via Arc<Graph>
            while let Ok(query) = receiver.recv_async().await {
                tracing::debug!(worker_id, query = %query, "Processing query (shared mode)");
                query.process_and_send(&*graph).await;
            }

            tracing::info!(worker_id, "Query worker shutting down");
        });

        handles.push(handle);
    }

    tracing::info!(
        num_workers,
        "Spawned query consumer workers (shared TransactionDB mode)"
    );
    handles
}

/// Spawn N query consumer workers with READONLY Storage instances.
///
/// ⚠️ **WARNING**: This creates separate readonly DB instances per worker.
///
/// Readonly instances have poor consistency (25-30% success rate) because they:
/// - Only see SST files on disk (not active memtable)
/// - Never update after opening (static snapshot)
/// - Require reopening to see new writes (60-1600ms overhead)
///
/// # Use Cases
/// - Historical/archival data analysis
/// - Point-in-time snapshots
/// - Static data where immediate consistency is not needed
///
/// # NOT Recommended For
/// - MCP servers or APIs needing read-after-write consistency
/// - Real-time applications
/// - Use `spawn_query_consumer_pool_shared` instead for 99%+ consistency
///
/// # Arguments
/// * `receiver` - Shared flume receiver (MPMC channel)
/// * `config` - Reader configuration
/// * `db_path` - Path to the database
/// * `num_workers` - Number of worker threads to spawn
///
/// # Returns
/// Vector of JoinHandles for all worker threads
#[doc(hidden)]
#[allow(deprecated)]
pub fn spawn_query_consumer_pool_readonly(
    receiver: flume::Receiver<Query>,
    config: ReaderConfig,
    db_path: &Path,
    num_workers: usize,
) -> Vec<JoinHandle<()>> {
    let mut handles = Vec::with_capacity(num_workers);

    for worker_id in 0..num_workers {
        let receiver = receiver.clone();
        let _config = config.clone();
        let db_path = db_path.to_path_buf();

        let handle = tokio::spawn(async move {
            tracing::info!(worker_id, "Query worker starting (readonly mode)");

            // Each worker opens its own readonly Storage
            let mut storage = Storage::readonly(&db_path);
            if let Err(e) = storage.ready() {
                tracing::error!(worker_id, err = %e, "Query worker failed to ready storage");
                return;
            }

            let storage = Arc::new(storage);
            #[allow(deprecated)]
            let graph = Graph::new(storage);

            // Process queries from shared channel
            while let Ok(query) = receiver.recv_async().await {
                tracing::debug!(worker_id, query = %query, "Processing query (readonly mode)");
                query.process_and_send(&graph).await;
            }

            tracing::info!(worker_id, "Query worker shutting down");
        });

        handles.push(handle);
    }

    tracing::info!(
        num_workers,
        "Spawned query consumer workers (readonly mode - 25-30% consistency)"
    );
    handles
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
pub fn create_reader_with_storage(storage: Arc<Storage>, config: ReaderConfig) -> (Reader, flume::Receiver<Query>) {
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
pub(crate) fn create_reader_with_processor(processor: Arc<GraphProcessor>, config: ReaderConfig) -> (Reader, flume::Receiver<Query>) {
    let (sender, receiver) = flume::bounded(config.channel_buffer_size);
    let reader = Reader::with_processor(sender, processor);
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
    let reader = Reader::with_processor(sender, processor.clone());

    let mut handles = Vec::with_capacity(num_workers);
    for worker_id in 0..num_workers {
        let receiver = receiver.clone();
        let processor = processor.clone();

        let handle = tokio::spawn(async move {
            tracing::info!(worker_id, "Query worker starting (processor mode)");

            // Process queries using shared Processor
            while let Ok(query) = receiver.recv_async().await {
                tracing::debug!(worker_id, query = %query, "Processing query (processor mode)");
                query.process_and_send(&*processor).await;
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

    #[tokio::test]
    async fn test_reader_closed_detection() {
        let config = ReaderConfig::default();
        let (reader, receiver) = create_query_reader(config);

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
        let (reader, _receiver) = create_query_reader(config);

        // Test that reader is not closed initially
        assert!(!reader.is_closed());

        // We can't test actual sending without a consumer running,
        // but we've verified the reader is constructed properly
    }
}
