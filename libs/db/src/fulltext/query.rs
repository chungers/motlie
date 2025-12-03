//! Fulltext query module providing search queries against Tantivy index.
//!
//! This module contains only business logic - query type definitions and their
//! QueryExecutor implementations. Infrastructure (traits, Reader, Consumer, spawn
//! functions) is in the `reader` module.

use std::time::Duration;

use anyhow::Result;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::Value;
use tokio::sync::oneshot;

use super::reader::{Processor, QueryExecutor, QueryProcessor, Reader};
use super::Storage;
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
        use tantivy::query::{BooleanQuery, Occur, TermQuery};
        use tantivy::schema::IndexRecordOption;
        use tantivy::Term;

        let index = storage.index()?;
        let fields = storage.fields()?;

        // Create a reader for searching
        let reader = index
            .reader()
            .map_err(|e| anyhow::anyhow!("Failed to create index reader: {}", e))?;
        let searcher = reader.searcher();

        // Parse the user's text query
        let query_parser =
            QueryParser::for_index(index, vec![fields.content_field, fields.node_name_field]);

        let text_query = query_parser
            .parse_query(&self.query)
            .map_err(|e| anyhow::anyhow!("Failed to parse query '{}': {}", self.query, e))?;

        // Build doc_type filter: (doc_type = "nodes" OR doc_type = "node_fragments")
        let nodes_term = Term::from_field_text(fields.doc_type_field, "nodes");
        let fragments_term = Term::from_field_text(fields.doc_type_field, "node_fragments");

        let doc_type_filter = BooleanQuery::new(vec![
            (
                Occur::Should,
                Box::new(TermQuery::new(nodes_term, IndexRecordOption::Basic)),
            ),
            (
                Occur::Should,
                Box::new(TermQuery::new(fragments_term, IndexRecordOption::Basic)),
            ),
        ]);

        // Combine: text_query AND doc_type_filter
        let combined_query = BooleanQuery::new(vec![
            (Occur::Must, text_query),
            (Occur::Must, Box::new(doc_type_filter)),
        ]);

        // Search with TopDocs collector - get extra results since we'll dedupe by node ID
        let top_docs = searcher
            .search(&combined_query, &TopDocs::with_limit(self.k * 3))
            .map_err(|e| anyhow::anyhow!("Search failed: {}", e))?;

        // Collect results, deduplicating by node ID and keeping best score
        // This handles both node documents and node_fragment documents
        let mut node_scores: HashMap<Id, (f32, String)> = HashMap::new();

        for (score, doc_address) in top_docs {
            let doc = searcher
                .doc::<tantivy::TantivyDocument>(doc_address)
                .map_err(|e| anyhow::anyhow!("Failed to retrieve document: {}", e))?;

            // Extract node ID
            let id = match doc.get_first(fields.id_field).and_then(|v| v.as_bytes()) {
                Some(bytes) => match Id::from_slice(bytes) {
                    Ok(id) => id,
                    Err(_) => continue, // Invalid ID format
                },
                None => continue, // No ID field
            };

            // Extract node name (only present on node documents, not fragments)
            let name = doc
                .get_first(fields.node_name_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

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

// Use macro to implement QueryProcessor for query types
crate::impl_fulltext_query_processor!(Nodes);

#[async_trait::async_trait]
impl QueryProcessor for Query {
    async fn process_and_send<P: Processor + Sync>(self, processor: &P) {
        match self {
            Query::Nodes(q) => q.process_and_send(processor).await,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fulltext::reader::{
        create_query_reader, spawn_consumer, Consumer, ReaderConfig,
    };
    use crate::fulltext::{Index, Storage};
    use crate::graph::mutation::{AddNode, AddNodeFragment, Runnable as MutRunnable};
    use crate::graph::schema::NodeSummary;
    use crate::graph::writer::{create_mutation_writer, WriterConfig};
    use crate::{spawn_fulltext_consumer, DataUrl, TimestampMilli};
    use std::sync::Arc;
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
        let processor = Index::new(Arc::new(storage));
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
