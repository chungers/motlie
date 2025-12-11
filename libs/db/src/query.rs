//! Unified search query module composing fulltext search with graph lookups.
//!
//! This module provides a unified query interface that:
//! 1. Uses fulltext (Tantivy) for search/ranking
//! 2. Hydrates results from graph (RocksDB) as source of truth
//!
//! # Design
//!
//! Wrapper types delegate builder methods to `fulltext::query::{Nodes, Edges}`
//! and add mix-in behavior for graph hydration in `QueryExecutor::execute()`.
//!
//! # Example
//!
//! ```ignore
//! use motlie_db::{Nodes, Runnable, Reader};
//! use std::time::Duration;
//!
//! let results = Nodes::new("rust programming".to_string(), 10)
//!     .with_fuzzy(FuzzyLevel::Low)
//!     .with_offset(0)
//!     .run(&reader, Duration::from_secs(5))
//!     .await?;
//!
//! // Results are Vec<(Id, NodeName, NodeSummary)> - hydrated from graph
//! for (id, name, summary) in results {
//!     println!("{}: {}", id, name);
//! }
//! ```

use std::time::Duration;

use anyhow::Result;
use tokio::sync::oneshot;

use crate::fulltext;
use crate::graph;
use crate::graph::reader::Processor as GraphProcessor;
use crate::graph::schema::{DstId, EdgeName, EdgeSummary, NodeName, NodeSummary, SrcId};
use crate::Id;

// Re-export shared types from fulltext for API consistency
pub use crate::fulltext::query::FuzzyLevel;
pub use crate::fulltext::search::{EdgeHit, MatchSource, NodeHit};

// ============================================================================
// Query Enum
// ============================================================================

/// Unified Search enum - wraps fulltext queries with graph hydration
#[derive(Debug)]
pub enum Search {
    Nodes(Nodes),
    Edges(Edges),
}

impl std::fmt::Display for Search {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Search::Nodes(q) => write!(
                f,
                "UnifiedNodes: query={}, limit={}, offset={}",
                q.inner.query, q.inner.limit, q.offset
            ),
            Search::Edges(q) => write!(
                f,
                "UnifiedEdges: query={}, limit={}, offset={}",
                q.inner.query, q.inner.limit, q.offset
            ),
        }
    }
}

// ============================================================================
// Result Types
// ============================================================================

/// Result type for node searches: (Id, NodeName, NodeSummary)
pub type NodeResult = (Id, NodeName, NodeSummary);

/// Result type for edge searches: (SrcId, DstId, EdgeName, EdgeSummary)
pub type EdgeResult = (SrcId, DstId, EdgeName, EdgeSummary);

// ============================================================================
// Runnable Trait
// ============================================================================

/// Trait for query builders that can be executed via a unified Reader.
#[async_trait::async_trait]
pub trait Runnable {
    /// The output type this query produces
    type Output: Send + 'static;

    /// Execute this query against a unified Reader with the specified timeout
    async fn run(self, reader: &super::reader::Reader, timeout: Duration) -> Result<Self::Output>;
}

// ============================================================================
// QueryExecutor Trait
// ============================================================================

/// Trait for executing queries against reader::Storage.
///
/// This is distinct from `fulltext::reader::QueryExecutor` which takes `&fulltext::Storage`.
/// Our version takes `&reader::Storage` to access both fulltext and graph.
#[async_trait::async_trait]
pub trait QueryExecutor: Send + Sync {
    /// The type of result this query produces
    type Output: Send;

    /// Execute this query against composite storage (fulltext + graph)
    async fn execute(&self, storage: &super::reader::Storage) -> Result<Self::Output>;

    /// Get the timeout for this query
    fn timeout(&self) -> Duration;
}

// ============================================================================
// QueryProcessor Trait
// ============================================================================

/// Trait for processing queries without needing to know the result type.
/// Allows the Consumer to process queries polymorphically.
#[async_trait::async_trait]
pub trait QueryProcessor: Send {
    /// Process the query and send the result (consumes self)
    async fn process_and_send(self, processor: &super::reader::Storage);
}

// ============================================================================
// Nodes Query - Unified search for nodes
// ============================================================================

/// Unified query to search for nodes by fulltext, returning hydrated graph data.
///
/// Wraps `fulltext::query::Nodes` and adds:
/// - Graph hydration (looks up each hit in RocksDB)
/// - Pagination via `with_offset()`
///
/// Returns `Vec<(Id, NodeName, NodeSummary)>` - full node data from graph.
#[derive(Debug)]
pub struct Nodes {
    /// Inner fulltext query - delegates builder methods here
    pub(crate) inner: fulltext::query::Nodes,

    /// Offset for pagination (skip first N results from fulltext)
    pub offset: usize,

    /// Timeout for this query execution
    pub(crate) timeout: Option<Duration>,

    /// Channel to send the result back to the client
    result_tx: Option<oneshot::Sender<Result<Vec<NodeResult>>>>,
}

impl Nodes {
    /// Create a new unified nodes query.
    ///
    /// # Arguments
    /// * `query` - The search query string
    /// * `limit` - Maximum number of results to return
    pub fn new(query: String, limit: usize) -> Self {
        Self {
            inner: fulltext::query::Nodes::new(query, limit),
            offset: 0,
            timeout: None,
            result_tx: None,
        }
    }

    /// Enable fuzzy matching with the specified level.
    /// Delegates to inner fulltext query.
    pub fn with_fuzzy(mut self, level: FuzzyLevel) -> Self {
        self.inner = self.inner.with_fuzzy(level);
        self
    }

    /// Filter results by tags (documents must have ANY of the specified tags).
    /// Delegates to inner fulltext query.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.inner = self.inner.with_tags(tags);
        self
    }

    /// Set pagination offset (skip first N results from fulltext).
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Internal constructor with channel for query execution machinery
    pub(crate) fn with_channel(
        inner: fulltext::query::Nodes,
        offset: usize,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<Vec<NodeResult>>>,
    ) -> Self {
        Self {
            inner,
            offset,
            timeout: Some(timeout),
            result_tx: Some(result_tx),
        }
    }

    /// Send the result back to the client (consumes self)
    pub(crate) fn send_result(self, result: Result<Vec<NodeResult>>) {
        if let Some(tx) = self.result_tx {
            let _ = tx.send(result);
        }
    }
}

#[async_trait::async_trait]
impl Runnable for Nodes {
    type Output = Vec<NodeResult>;

    async fn run(self, reader: &super::reader::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = oneshot::channel();

        let query = Nodes::with_channel(self.inner, self.offset, timeout, result_tx);

        reader.send_query(Search::Nodes(query)).await?;

        match tokio::time::timeout(timeout, result_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("Query channel closed")),
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
        }
    }
}

#[async_trait::async_trait]
impl QueryExecutor for Nodes {
    type Output = Vec<NodeResult>;

    async fn execute(&self, storage: &super::reader::Storage) -> Result<Self::Output> {
        // Step 1: REUSE existing fulltext query execution to get hits
        use crate::fulltext::reader::QueryExecutor as FulltextQueryExecutor;

        let fulltext_storage = storage.fulltext.storage();
        let hits: Vec<NodeHit> = FulltextQueryExecutor::execute(&self.inner, fulltext_storage).await?;

        // Step 2: NEW hydration behavior - look up each hit in graph
        use crate::graph::reader::QueryExecutor as GraphQueryExecutor;

        let graph_storage = storage.graph.storage();
        let mut results = Vec::with_capacity(hits.len());

        for hit in hits.into_iter().skip(self.offset) {
            // Reuse existing graph query execution
            let query = graph::query::NodeById::new(hit.id, None);
            match GraphQueryExecutor::execute(&query, graph_storage).await {
                Ok((name, summary)) => {
                    results.push((hit.id, name, summary));
                }
                Err(_) => {
                    // Stale index entry - skip silently (log at debug level)
                    tracing::debug!(id = %hit.id, "Node in fulltext but not in graph, skipping");
                    continue;
                }
            }
        }

        Ok(results)
    }

    fn timeout(&self) -> Duration {
        self.timeout.unwrap_or(Duration::from_secs(30))
    }
}

#[async_trait::async_trait]
impl QueryProcessor for Nodes {
    async fn process_and_send(self, processor: &super::reader::Storage) {
        let result = QueryExecutor::execute(&self, processor).await;
        self.send_result(result);
    }
}

// ============================================================================
// Edges Query - Unified search for edges
// ============================================================================

/// Unified query to search for edges by fulltext, returning hydrated graph data.
///
/// Wraps `fulltext::query::Edges` and adds:
/// - Graph hydration (looks up each hit in RocksDB)
/// - Pagination via `with_offset()`
///
/// Returns `Vec<(SrcId, DstId, EdgeName, EdgeSummary)>` - full edge data from graph.
#[derive(Debug)]
pub struct Edges {
    /// Inner fulltext query - delegates builder methods here
    pub(crate) inner: fulltext::query::Edges,

    /// Offset for pagination (skip first N results from fulltext)
    pub offset: usize,

    /// Timeout for this query execution
    pub(crate) timeout: Option<Duration>,

    /// Channel to send the result back to the client
    result_tx: Option<oneshot::Sender<Result<Vec<EdgeResult>>>>,
}

impl Edges {
    /// Create a new unified edges query.
    ///
    /// # Arguments
    /// * `query` - The search query string
    /// * `limit` - Maximum number of results to return
    pub fn new(query: String, limit: usize) -> Self {
        Self {
            inner: fulltext::query::Edges::new(query, limit),
            offset: 0,
            timeout: None,
            result_tx: None,
        }
    }

    /// Enable fuzzy matching with the specified level.
    /// Delegates to inner fulltext query.
    pub fn with_fuzzy(mut self, level: FuzzyLevel) -> Self {
        self.inner = self.inner.with_fuzzy(level);
        self
    }

    /// Filter results by tags (documents must have ANY of the specified tags).
    /// Delegates to inner fulltext query.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.inner = self.inner.with_tags(tags);
        self
    }

    /// Set pagination offset (skip first N results from fulltext).
    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = offset;
        self
    }

    /// Internal constructor with channel for query execution machinery
    pub(crate) fn with_channel(
        inner: fulltext::query::Edges,
        offset: usize,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<Vec<EdgeResult>>>,
    ) -> Self {
        Self {
            inner,
            offset,
            timeout: Some(timeout),
            result_tx: Some(result_tx),
        }
    }

    /// Send the result back to the client (consumes self)
    pub(crate) fn send_result(self, result: Result<Vec<EdgeResult>>) {
        if let Some(tx) = self.result_tx {
            let _ = tx.send(result);
        }
    }
}

#[async_trait::async_trait]
impl Runnable for Edges {
    type Output = Vec<EdgeResult>;

    async fn run(self, reader: &super::reader::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = oneshot::channel();

        let query = Edges::with_channel(self.inner, self.offset, timeout, result_tx);

        reader.send_query(Search::Edges(query)).await?;

        match tokio::time::timeout(timeout, result_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("Query channel closed")),
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
        }
    }
}

#[async_trait::async_trait]
impl QueryExecutor for Edges {
    type Output = Vec<EdgeResult>;

    async fn execute(&self, storage: &super::reader::Storage) -> Result<Self::Output> {
        // Step 1: REUSE existing fulltext query execution to get hits
        use crate::fulltext::reader::QueryExecutor as FulltextQueryExecutor;

        let fulltext_storage = storage.fulltext.storage();
        let hits: Vec<EdgeHit> = FulltextQueryExecutor::execute(&self.inner, fulltext_storage).await?;

        // Step 2: NEW hydration behavior - look up each hit in graph
        use crate::graph::reader::QueryExecutor as GraphQueryExecutor;

        let graph_storage = storage.graph.storage();
        let mut results = Vec::with_capacity(hits.len());

        for hit in hits.into_iter().skip(self.offset) {
            let query = graph::query::EdgeSummaryBySrcDstName::new(
                hit.src_id,
                hit.dst_id,
                hit.edge_name.clone(),
                None,
            );
            match GraphQueryExecutor::execute(&query, graph_storage).await {
                Ok((summary, _weight)) => {
                    results.push((hit.src_id, hit.dst_id, hit.edge_name, summary));
                }
                Err(_) => {
                    // Stale index entry - skip silently (log at debug level)
                    tracing::debug!(
                        src_id = %hit.src_id,
                        dst_id = %hit.dst_id,
                        edge_name = %hit.edge_name,
                        "Edge in fulltext but not in graph, skipping"
                    );
                    continue;
                }
            }
        }

        Ok(results)
    }

    fn timeout(&self) -> Duration {
        self.timeout.unwrap_or(Duration::from_secs(30))
    }
}

#[async_trait::async_trait]
impl QueryProcessor for Edges {
    async fn process_and_send(self, processor: &super::reader::Storage) {
        let result = QueryExecutor::execute(&self, processor).await;
        self.send_result(result);
    }
}

// ============================================================================
// Search QueryProcessor implementation
// ============================================================================

#[async_trait::async_trait]
impl QueryProcessor for Search {
    async fn process_and_send(self, processor: &super::reader::Storage) {
        match self {
            Search::Nodes(q) => q.process_and_send(processor).await,
            Search::Edges(q) => q.process_and_send(processor).await,
        }
    }
}
