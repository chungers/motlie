//! Fulltext query module providing search queries against Tantivy index.
//!
//! This module follows the same pattern as graph queries:
//! - Query types implement `QueryExecutor` to execute themselves against Storage
//! - `Runnable` trait enables client-side `query.run(reader, timeout)` API
//! - Oneshot channels for async result delivery
//! - Consumer processes queries from MPMC channel

use std::time::Duration;

use anyhow::Result;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::Value;
use tokio::sync::oneshot;

use super::{Index, Storage};
use crate::Id;

// ============================================================================
// Query Enum
// ============================================================================

/// Query enum representing all possible fulltext query types
#[derive(Debug)]
pub enum Query {
    Nodes(Nodes),
}

impl std::fmt::Display for Query {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Query::Nodes(q) => write!(f, "FulltextNodes: query={}, k={}", q.query, q.k),
        }
    }
}

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

    /// Get the timeout for this query
    fn timeout(&self) -> Duration;
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

/// Implement Processor for Index (mirrors graph::Graph implementing query::Processor)
impl Processor for Index {
    fn storage(&self) -> &Storage {
        Index::storage(self)
    }
}

// ============================================================================
// QueryProcessor Trait
// ============================================================================

/// Trait for processing queries without needing to know the result type.
/// Allows the Consumer to process queries polymorphically.
#[async_trait::async_trait]
pub trait QueryProcessor: Send {
    /// Process the query and send the result (consumes self)
    async fn process_and_send<P: Processor + Sync>(self, processor: &P);
}

// ============================================================================
// Runnable Trait
// ============================================================================

/// Trait for query builders that can be executed via a Reader.
/// This matches the graph query pattern for client-side API.
#[async_trait::async_trait]
pub trait Runnable {
    /// The output type this query produces
    type Output: Send + 'static;

    /// Execute this query against a FulltextReader with the specified timeout
    async fn run(self, reader: &Reader, timeout: Duration) -> Result<Self::Output>;
}

// ============================================================================
// Reader
// ============================================================================

/// Handle for sending fulltext queries to the reader queue.
/// This is the fulltext equivalent of the graph Reader.
#[derive(Debug, Clone)]
pub struct Reader {
    sender: flume::Sender<Query>,
}

impl Reader {
    /// Create a new Reader with the given sender
    pub fn new(sender: flume::Sender<Query>) -> Self {
        Reader { sender }
    }

    /// Send a query to the reader queue
    pub async fn send_query(&self, query: Query) -> Result<()> {
        self.sender
            .send_async(query)
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
pub fn create_query_reader(config: ReaderConfig) -> (Reader, flume::Receiver<Query>) {
    let (sender, receiver) = flume::bounded(config.channel_buffer_size);
    let reader = Reader::new(sender);
    (reader, receiver)
}

// ============================================================================
// Nodes Query - Search for nodes by text
// ============================================================================

/// Search result containing node ID and relevance score
#[derive(Debug, Clone)]
pub struct NodeSearchResult {
    /// The node ID
    pub id: Id,
    /// The node name
    pub name: String,
    /// BM25 relevance score
    pub score: f32,
}

/// Query to search for nodes by fulltext query, returning top K results.
/// This is the fulltext equivalent of searching nodes.
#[derive(Debug)]
pub struct Nodes {
    /// The search query string
    pub query: String,

    /// The top-K to retrieve
    pub k: usize,

    /// Timeout for this query execution (only set when query has channel)
    pub(crate) timeout: Option<Duration>,

    /// Channel to send the result back to the client (only set when ready to execute)
    result_tx: Option<oneshot::Sender<Result<Vec<NodeSearchResult>>>>,
}

impl Nodes {
    /// Create a new query request (public API - no channel, no timeout yet)
    /// Use `.run(reader, timeout)` to execute this query
    pub fn new(query: String, k: usize) -> Self {
        Self {
            query,
            k,
            timeout: None,
            result_tx: None,
        }
    }

    /// Internal constructor used by the query execution machinery (has the channel)
    pub(crate) fn with_channel(
        query: String,
        k: usize,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<Vec<NodeSearchResult>>>,
    ) -> Self {
        Self {
            query,
            k,
            timeout: Some(timeout),
            result_tx: Some(result_tx),
        }
    }

    /// Send the result back to the client (consumes self)
    pub(crate) fn send_result(self, result: Result<Vec<NodeSearchResult>>) {
        // Ignore error if receiver was dropped (client timeout/cancellation)
        if let Some(tx) = self.result_tx {
            let _ = tx.send(result);
        }
    }
}

#[async_trait::async_trait]
impl Runnable for Nodes {
    type Output = Vec<NodeSearchResult>;

    async fn run(self, reader: &Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = oneshot::channel();

        let query = Nodes::with_channel(self.query, self.k, timeout, result_tx);

        reader.send_query(Query::Nodes(query)).await?;

        // Wait for result with timeout
        match tokio::time::timeout(timeout, result_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("Query channel closed")),
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
        }
    }
}

#[async_trait::async_trait]
impl QueryExecutor for Nodes {
    type Output = Vec<NodeSearchResult>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        use std::collections::HashMap;

        let index = storage.index()?;
        let fields = storage.fields()?;

        // Create a reader for searching
        let reader = index
            .reader()
            .map_err(|e| anyhow::anyhow!("Failed to create index reader: {}", e))?;
        let searcher = reader.searcher();

        // Parse and execute the query
        let query_parser =
            QueryParser::for_index(index, vec![fields.content_field, fields.node_name_field]);

        let parsed_query = query_parser
            .parse_query(&self.query)
            .map_err(|e| anyhow::anyhow!("Failed to parse query '{}': {}", self.query, e))?;

        // Search with TopDocs collector - get extra results since we'll dedupe by node ID
        let top_docs = searcher
            .search(&parsed_query, &TopDocs::with_limit(self.k * 3))
            .map_err(|e| anyhow::anyhow!("Search failed: {}", e))?;

        // Collect results, deduplicating by node ID and keeping best score
        // This handles both node documents and node_fragment documents
        let mut node_scores: HashMap<Id, (f32, String)> = HashMap::new();

        for (score, doc_address) in top_docs {
            let doc = searcher
                .doc::<tantivy::TantivyDocument>(doc_address)
                .map_err(|e| anyhow::anyhow!("Failed to retrieve document: {}", e))?;

            // Check doc_type - only process nodes and node_fragments
            let doc_type = if let Some(doc_type_value) = doc.get_first(fields.doc_type_field) {
                doc_type_value.as_str().unwrap_or("")
            } else {
                ""
            };

            if doc_type != "node" && doc_type != "node_fragment" {
                continue; // Skip edges and edge_fragments
            }

            // Extract node ID
            let id = if let Some(id_value) = doc.get_first(fields.id_field) {
                if let Some(bytes) = id_value.as_bytes() {
                    if bytes.len() == 16 {
                        let mut id_bytes = [0u8; 16];
                        id_bytes.copy_from_slice(bytes);
                        Id::from_bytes(id_bytes)
                    } else {
                        continue; // Invalid ID format
                    }
                } else {
                    continue; // No ID bytes
                }
            } else {
                continue; // No ID field
            };

            // Extract node name (only present on node documents, not fragments)
            let name = if let Some(name_value) = doc.get_first(fields.node_name_field) {
                name_value.as_str().unwrap_or("").to_string()
            } else {
                String::new()
            };

            // Update with best score, preferring entries with a name
            node_scores
                .entry(id)
                .and_modify(|(existing_score, existing_name)| {
                    if score > *existing_score {
                        *existing_score = score;
                    }
                    // Prefer non-empty names
                    if existing_name.is_empty() && !name.is_empty() {
                        *existing_name = name.clone();
                    }
                })
                .or_insert((score, name));
        }

        // Convert to results and sort by score
        let mut results: Vec<NodeSearchResult> = node_scores
            .into_iter()
            .map(|(id, (score, name))| NodeSearchResult { id, name, score })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(self.k);

        Ok(results)
    }

    fn timeout(&self) -> Duration {
        self.timeout
            .expect("Query must have timeout set when executing")
    }
}

#[async_trait::async_trait]
impl QueryProcessor for Nodes {
    async fn process_and_send<P: Processor + Sync>(self, processor: &P) {
        let result = self.execute(processor.storage()).await;
        self.send_result(result);
    }
}

#[async_trait::async_trait]
impl QueryProcessor for Query {
    async fn process_and_send<P: Processor + Sync>(self, processor: &P) {
        match self {
            Query::Nodes(q) => q.process_and_send(processor).await,
        }
    }
}

// ============================================================================
// Consumer
// ============================================================================

/// Consumer that processes fulltext queries using an Index.
/// This is the fulltext equivalent of the graph query Consumer.
pub struct Consumer {
    receiver: flume::Receiver<Query>,
    config: ReaderConfig,
    processor: Index,
}

impl Consumer {
    /// Create a new Consumer
    pub fn new(receiver: flume::Receiver<Query>, config: ReaderConfig, processor: Index) -> Self {
        Self {
            receiver,
            config,
            processor,
        }
    }

    /// Process queries continuously until the channel is closed
    pub async fn run(self) -> Result<()> {
        log::info!(
            "Starting fulltext query consumer with config: {:?}",
            self.config
        );

        loop {
            match self.receiver.recv_async().await {
                Ok(query) => {
                    log::debug!("Processing fulltext query: {}", query);
                    query.process_and_send(&self.processor).await;
                }
                Err(_) => {
                    log::info!("Fulltext query consumer shutting down - channel closed");
                    return Ok(());
                }
            }
        }
    }
}

/// Spawn a fulltext query consumer as a background task
pub fn spawn_consumer(consumer: Consumer) -> tokio::task::JoinHandle<Result<()>> {
    tokio::spawn(async move { consumer.run().await })
}

/// Create a fulltext query consumer
pub fn create_query_consumer(
    receiver: flume::Receiver<Query>,
    config: ReaderConfig,
    processor: Index,
) -> Consumer {
    Consumer::new(receiver, config, processor)
}

/// Spawn a fulltext query consumer with a new readonly Index
///
/// Uses readonly Storage since query consumers only need read access.
/// Multiple consumers can share the same index path.
pub fn spawn_query_consumer(
    receiver: flume::Receiver<Query>,
    config: ReaderConfig,
    index_path: &std::path::Path,
) -> tokio::task::JoinHandle<Result<()>> {
    // Use readonly mode for query consumers - they don't need write access
    let mut storage = super::Storage::readonly(index_path);
    storage.ready().expect("Failed to ready readonly storage");
    let processor = Index::new(std::sync::Arc::new(storage));
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
/// use motlie_db::{
///     FulltextStorage, FulltextIndex, FulltextReaderConfig,
///     create_fulltext_query_reader, spawn_fulltext_query_consumer_pool_shared,
/// };
/// use std::sync::Arc;
/// use std::path::Path;
///
/// # async fn example() -> anyhow::Result<()> {
/// let index_path = Path::new("/path/to/fulltext_index");
///
/// // Create a shared readonly Index (follows graph::Graph pattern)
/// let mut storage = FulltextStorage::readonly(index_path);
/// storage.ready()?;
/// let index = FulltextIndex::new(Arc::new(storage));
///
/// // Create reader channel
/// let config = FulltextReaderConfig { channel_buffer_size: 100 };
/// let (reader, receiver) = create_fulltext_query_reader(config);
///
/// // Spawn multiple consumers sharing the same Index
/// let num_workers = 4;
/// let handles = spawn_fulltext_query_consumer_pool_shared(
///     receiver,
///     Arc::new(index),
///     num_workers,
/// );
///
/// // All workers share the same Tantivy Index via Arc<Index>
/// # Ok(())
/// # }
/// ```
pub fn spawn_query_consumer_pool_shared(
    receiver: flume::Receiver<Query>,
    index: std::sync::Arc<Index>,
    num_workers: usize,
) -> Vec<tokio::task::JoinHandle<()>> {
    let mut handles = Vec::with_capacity(num_workers);

    for worker_id in 0..num_workers {
        let receiver = receiver.clone();
        let index = index.clone(); // Cheap Arc clone - shares Tantivy Index

        let handle = tokio::spawn(async move {
            log::info!(
                "Fulltext query worker {} starting (shared Index mode)",
                worker_id
            );

            // Process queries from shared channel
            // All workers share the same Tantivy Index via Arc<Index>
            while let Ok(query) = receiver.recv_async().await {
                log::debug!("Worker {} processing query: {}", worker_id, query);
                query.process_and_send(&*index).await;
            }

            log::info!("Fulltext query worker {} shutting down", worker_id);
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
pub fn spawn_query_consumer_pool_readonly(
    receiver: flume::Receiver<Query>,
    config: ReaderConfig,
    index_path: &std::path::Path,
    num_workers: usize,
) -> Vec<tokio::task::JoinHandle<()>> {
    let mut handles = Vec::with_capacity(num_workers);
    let index_path = index_path.to_path_buf();

    for worker_id in 0..num_workers {
        let receiver = receiver.clone();
        let _config = config.clone(); // Reserved for future use
        let index_path = index_path.clone();

        let handle = tokio::spawn(async move {
            log::info!(
                "Fulltext query worker {} starting (individual readonly mode)",
                worker_id
            );

            // Each worker creates its own readonly Storage and Index
            let mut storage = super::Storage::readonly(&index_path);
            if let Err(e) = storage.ready() {
                log::error!("Worker {} failed to ready storage: {}", worker_id, e);
                return;
            }
            let index = Index::new(std::sync::Arc::new(storage));

            // Process queries from shared channel
            while let Ok(query) = receiver.recv_async().await {
                log::debug!("Worker {} processing query: {}", worker_id, query);
                query.process_and_send(&index).await;
            }

            log::info!("Fulltext query worker {} shutting down", worker_id);
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
    use crate::fulltext::Storage;
    use crate::mutation::{AddNode, AddNodeFragment, Runnable as MutRunnable};
    use crate::writer::{create_mutation_writer, WriterConfig};
    use crate::{spawn_fulltext_consumer, DataUrl, TimestampMilli};
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_nodes_query_basic() {
        // Create temporary directory for fulltext index
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("fulltext_index");

        // Setup mutation writer and fulltext consumer
        let writer_config = WriterConfig {
            channel_buffer_size: 10,
        };
        let (writer, receiver) = create_mutation_writer(writer_config.clone());
        let fulltext_handle = spawn_fulltext_consumer(receiver, writer_config, &index_path);

        // Create test nodes with searchable content
        let node1_id = Id::new();
        let node1 = AddNode {
            id: node1_id,
            ts_millis: TimestampMilli::now(),
            name: "RustLang".to_string(),
            temporal_range: None,
            summary: crate::NodeSummary::from_text("Rust language summary"),
        };
        node1.run(&writer).await.unwrap();

        let fragment1 = AddNodeFragment {
            id: node1_id,
            ts_millis: TimestampMilli::now(),
            content: DataUrl::from_text("Rust is a systems programming language"),
            temporal_range: None,
        };
        fragment1.run(&writer).await.unwrap();

        let node2_id = Id::new();
        let node2 = AddNode {
            id: node2_id,
            ts_millis: TimestampMilli::now(),
            name: "Python".to_string(),
            temporal_range: None,
            summary: crate::NodeSummary::from_text("Python language summary"),
        };
        node2.run(&writer).await.unwrap();

        // Wait for indexing
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Shutdown mutation consumer
        drop(writer);
        fulltext_handle.await.unwrap().unwrap();

        // Now test the query consumer
        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };
        let (reader, query_receiver) = create_query_reader(reader_config.clone());

        // Create processor from existing index (readonly for query consumer)
        let mut storage = Storage::readonly(&index_path);
        storage.ready().unwrap();
        let processor = Index::new(std::sync::Arc::new(storage));
        let consumer = Consumer::new(query_receiver, reader_config, processor);
        let consumer_handle = spawn_consumer(consumer);

        // Search for "Rust"
        let results = Nodes::new("Rust".to_string(), 10)
            .run(&reader, Duration::from_secs(5))
            .await
            .unwrap();

        assert!(!results.is_empty(), "Should find at least one result");

        // Cleanup
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }
}
