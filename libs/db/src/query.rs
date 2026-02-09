//! Unified query module exposing both fulltext search and graph lookups.
//!
//! This module provides the **unified query interface** for `motlie-db`, allowing
//! all query types to be executed through a single [`reader::Reader`](crate::reader::Reader).
//!
//! # Overview
//!
//! The unified query API composes two storage backends:
//! - **Fulltext (Tantivy)**: For text search with BM25 ranking
//! - **Graph (RocksDB)**: For node/edge data and graph traversal
//!
//! Fulltext search results are automatically **hydrated** with full graph data,
//! returning complete node/edge information instead of just search hits.
//!
//! # Query Types
//!
//! ## Fulltext Queries (search + hydration)
//!
//! | Query | Output | Description |
//! |-------|--------|-------------|
//! | [`Nodes`] | `Vec<(Id, NodeName, NodeSummary)>` | Search nodes by text |
//! | [`Edges`] | `Vec<(SrcId, DstId, EdgeName, EdgeSummary)>` | Search edges by text |
//!
//! ## Graph Queries (direct lookups)
//!
//! | Query | Output | Description |
//! |-------|--------|-------------|
//! | [`NodeById`] | `(NodeName, NodeSummary, Version)` | Lookup node by ID |
//! | [`NodesByIdsMulti`] | `Vec<(Id, NodeName, NodeSummary, Version)>` | Batch lookup nodes by IDs |
//! | [`OutgoingEdges`] | `Vec<(Option<EdgeWeight>, SrcId, DstId, EdgeName, Version)>` | Get edges from a node |
//! | [`IncomingEdges`] | `Vec<(Option<EdgeWeight>, DstId, SrcId, EdgeName, Version)>` | Get edges to a node |
//! | [`EdgeDetails`] | `(Option<EdgeWeight>, SrcId, DstId, EdgeName, EdgeSummary, Version)` | Lookup edge by topology |
//! | [`NodeFragments`] | `Vec<(TimestampMilli, FragmentContent)>` | Get node fragment history |
//! | [`EdgeFragments`] | `Vec<(TimestampMilli, FragmentContent)>` | Get edge fragment history |
//!
//! ## Graph Enumeration Queries (pagination)
//!
//! | Query | Output | Description |
//! |-------|--------|-------------|
//! | [`AllNodes`] | `Vec<(Id, NodeName, NodeSummary, Version)>` | Enumerate all nodes |
//! | [`AllEdges`] | `Vec<(Option<EdgeWeight>, SrcId, DstId, EdgeName, Version)>` | Enumerate all edges |
//!
//! # Usage
//!
//! All queries implement [`Runnable<reader::Reader>`](Runnable), allowing execution
//! with `query.run(&reader, timeout)`:
//!
//! ```ignore
//! use motlie_db::{Storage, StorageConfig};
//! use motlie_db::query::{
//!     Nodes, Edges, NodeById, OutgoingEdges, IncomingEdges,
//!     EdgeDetails, NodeFragments, FuzzyLevel, Runnable,
//! };
//! use std::time::Duration;
//!
//! // Initialize unified storage (read-only or read-write)
//! // Storage takes a single path and derives <path>/graph and <path>/fulltext
//! let storage = Storage::readonly(db_path);
//! let handles = storage.ready(StorageConfig::default())?;  // ReadOnlyHandles
//!
//! let timeout = Duration::from_secs(5);
//!
//! // Fulltext search with graph hydration
//! let results = Nodes::new("rust programming".to_string(), 10)
//!     .with_fuzzy(FuzzyLevel::Low)
//!     .with_tags(vec!["systems".to_string()])
//!     .run(handles.reader(), timeout)
//!     .await?;
//!
//! // Direct graph lookup by ID
//! let (name, summary, version) = NodeById::new(node_id, None)
//!     .run(handles.reader(), timeout)
//!     .await?;
//!
//! // Get outgoing edges from a node
//! let outgoing = OutgoingEdges::new(node_id, None)
//!     .run(handles.reader(), timeout)
//!     .await?;
//!
//! // Get incoming edges to a node
//! let incoming = IncomingEdges::new(node_id, None)
//!     .run(handles.reader(), timeout)
//!     .await?;
//!
//! // Lookup edge by topology
//! let edge = EdgeDetails::new(src_id, dst_id, "relationship".to_string(), None)
//!     .run(handles.reader(), timeout)
//!     .await?;
//!
//! // Concurrent queries with tokio::try_join!
//! let reader = handles.reader_clone();
//! let (nodes, outgoing, incoming) = tokio::try_join!(
//!     Nodes::new("search".to_string(), 10).run(&reader, timeout),
//!     OutgoingEdges::new(node_id, None).run(&reader, timeout),
//!     IncomingEdges::new(node_id, None).run(&reader, timeout)
//! )?;
//!
//! // Clean shutdown
//! handles.shutdown().await?;
//! ```
//!
//! # Architecture
//!
//! The unified query system uses an MPMC (multi-producer multi-consumer) pattern
//! with flume channels for efficient concurrent query dispatch:
//!
//! ```text
//! ┌─────────────┐    ┌─────────────┐    ┌──────────────────────┐
//! │ Query.run() │ -> │ Query enum  │ -> │ Consumer Pool        │
//! │             │    │ (dispatch)  │    │ (N async workers)    │
//! └─────────────┘    └─────────────┘    └──────────────────────┘
//!        │                                        │
//!        │                                        ▼
//!        │                              ┌──────────────────────┐
//!        └──────────────────────────────│ oneshot result       │
//!                                       └──────────────────────┘
//! ```
//!
//! # See Also
//!
//! - [`Storage::ready()`](crate::Storage::ready) - Initialize unified storage
//! - [`fulltext::query`](crate::fulltext::query) - Raw fulltext queries (returns hits with scores)
//! - [`graph::query`](crate::graph::query) - Graph-only queries

use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::sync::oneshot;

use crate::fulltext;
use crate::graph;
use crate::graph::reader::QueryExecutor;
use crate::request::{new_request_id, RequestEnvelope, RequestMeta};

// Re-export Runnable trait from reader module
pub use crate::reader::Runnable;
use crate::graph::schema::{
    DstId, EdgeName, EdgeSummary, EdgeWeight, FragmentContent, NodeName, NodeSummary, SrcId,
    Version,
};
use crate::{Id, TimestampMilli};

// ============================================================================
// Re-exports from fulltext::query
// ============================================================================

/// Type alias to fulltext::query::Nodes.
///
/// This is the same type used for both fulltext-only queries and unified queries.
/// The behavior differs based on which reader you use:
///
/// - With `fulltext::Reader`: returns `Vec<NodeHit>` (raw fulltext hits)
/// - With `reader::Reader`: returns `Vec<NodeResult>` (hydrated graph data)
pub type Nodes = fulltext::query::Nodes;

/// Type alias to fulltext::query::Edges.
///
/// This is the same type used for both fulltext-only queries and unified queries.
/// The behavior differs based on which reader you use:
///
/// - With `fulltext::Reader`: returns `Vec<EdgeHit>` (raw fulltext hits)
/// - With `reader::Reader`: returns `Vec<EdgeResult>` (hydrated graph data)
pub type Edges = fulltext::query::Edges;

// Re-export other query-related types
pub use crate::fulltext::query::FuzzyLevel;
pub use crate::fulltext::search::{EdgeHit, MatchSource, NodeHit};

// ============================================================================
// Re-exports from graph::query
// ============================================================================

/// Type alias to graph::query::NodeById.
///
/// Lookup a node by its ID.
///
/// - With `graph::Reader`: returns `(NodeName, NodeSummary)`
/// - With `reader::Reader`: same result, forwarded to graph storage
pub type NodeById = graph::query::NodeById;

/// Type alias to graph::query::NodesByIdsMulti.
///
/// Batch lookup of multiple nodes by ID. Uses RocksDB's `multi_get_cf()`
/// for efficient batch reads. Missing nodes and temporally invalid nodes
/// are silently omitted from results.
///
/// - With `graph::Reader`: returns `Vec<(Id, NodeName, NodeSummary, Version)>`
/// - With `reader::Reader`: same result, forwarded to graph storage
pub type NodesByIdsMulti = graph::query::NodesByIdsMulti;

/// Type alias to graph::query::OutgoingEdges.
///
/// Get all outgoing edges from a node.
///
/// - With `graph::Reader`: returns `Vec<(EdgeName, DstId)>`
/// - With `reader::Reader`: same result, forwarded to graph storage
pub type OutgoingEdges = graph::query::OutgoingEdges;

/// Type alias to graph::query::IncomingEdges.
///
/// Get all incoming edges to a node.
///
/// - With `graph::Reader`: returns `Vec<(EdgeName, SrcId)>`
/// - With `reader::Reader`: same result, forwarded to graph storage
pub type IncomingEdges = graph::query::IncomingEdges;

/// Type alias to graph::query::EdgeSummaryBySrcDstName.
///
/// Look up an edge's details by source ID, destination ID, and edge name.
/// Returns `EdgeDetailsResult` (Option<EdgeWeight>, SrcId, DstId, EdgeName, EdgeSummary, Version).
///
/// - With `graph::Reader`: use `.run()` method (returns `(EdgeSummary, Option<EdgeWeight>, Version)`)
/// - With `reader::Reader`: use `.run()` method (returns `EdgeDetailsResult`)
pub type EdgeDetails = graph::query::EdgeSummaryBySrcDstName;

/// Type alias to graph::query::NodeFragmentsByIdTimeRange.
///
/// Query node fragments within a time range.
///
/// - With `graph::Reader`: use `.run()` method
/// - With `reader::Reader`: use `.run()` method
pub type NodeFragments = graph::query::NodeFragmentsByIdTimeRange;

/// Type alias to graph::query::EdgeFragmentsByIdTimeRange.
///
/// Query edge fragments within a time range.
///
/// - With `graph::Reader`: use `.run()` method
/// - With `reader::Reader`: use `.run()` method
pub type EdgeFragments = graph::query::EdgeFragmentsByIdTimeRange;

/// Type alias to graph::query::AllNodes.
///
/// Enumerate all nodes with pagination support for graph algorithms.
///
/// # Example
///
/// ```ignore
/// // Get first page of nodes
/// let nodes = AllNodes::new(1000)
///     .run(handles.reader(), timeout)
///     .await?;
///
/// // Get next page using cursor
/// if let Some((last_id, _, _, _)) = nodes.last() {
///     let next_page = AllNodes::new(1000)
///         .with_cursor(*last_id)
///         .run(handles.reader(), timeout)
///         .await?;
/// }
/// ```
pub type AllNodes = graph::query::AllNodes;

/// Type alias to graph::query::AllEdges.
///
/// Enumerate all edges with pagination support for graph algorithms.
///
/// # Example
///
/// ```ignore
/// // Get first page of edges
/// let edges = AllEdges::new(1000)
///     .run(handles.reader(), timeout)
///     .await?;
///
/// // Get next page using cursor
/// if let Some((_, src, dst, name)) = edges.last() {
///     let next_page = AllEdges::new(1000)
///         .with_cursor((*src, *dst, name.clone()))
///         .run(handles.reader(), timeout)
///         .await?;
/// }
/// ```
pub type AllEdges = graph::query::AllEdges;

// ============================================================================
// Result Types for Hydrated Queries
// ============================================================================

/// Result type for hydrated node searches: (Id, NodeName, NodeSummary)
pub type NodeResult = (Id, NodeName, NodeSummary);

/// Result type for hydrated edge searches: (SrcId, DstId, EdgeName, EdgeSummary)
pub type EdgeResult = (SrcId, DstId, EdgeName, EdgeSummary);

/// Result type for EdgeDetails query: (weight, src_id, dst_id, edge_name, edge_summary, version)
pub type EdgeDetailsResult = (Option<EdgeWeight>, SrcId, DstId, EdgeName, EdgeSummary, Version);

// ============================================================================
// Unified Query Types and Execution
// ============================================================================

#[derive(Debug, Clone)]
pub struct NodesQuery {
    pub params: fulltext::query::Nodes,
    pub offset: usize,
}

#[derive(Debug, Clone)]
pub struct EdgesQuery {
    pub params: fulltext::query::Edges,
    pub offset: usize,
}

/// Wrapper enum for dispatching queries through the unified reader.
///
/// Users should not construct this directly - use `Runnable::run()` instead.
#[derive(Debug)]
#[doc(hidden)]
#[allow(private_interfaces)]
pub enum Query {
    // Fulltext search queries (with hydration)
    Nodes(NodesQuery),
    Edges(EdgesQuery),
    // Graph queries (forwarded to graph storage)
    NodeById(graph::query::NodeById),
    NodesByIdsMulti(graph::query::NodesByIdsMulti),
    OutgoingEdges(graph::query::OutgoingEdges),
    IncomingEdges(graph::query::IncomingEdges),
    EdgeDetails(graph::query::EdgeSummaryBySrcDstName),
    NodeFragments(graph::query::NodeFragmentsByIdTimeRange),
    EdgeFragments(graph::query::EdgeFragmentsByIdTimeRange),
    // Graph enumeration queries
    AllNodes(graph::query::AllNodes),
    AllEdges(graph::query::AllEdges),
}

impl std::fmt::Display for Query {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Query::Nodes(q) => write!(
                f,
                "UnifiedNodes: query={}, limit={}, offset={}",
                q.params.query, q.params.limit, q.offset
            ),
            Query::Edges(q) => write!(
                f,
                "UnifiedEdges: query={}, limit={}, offset={}",
                q.params.query, q.params.limit, q.offset
            ),
            Query::NodeById(q) => write!(f, "NodeById: id={}", q.id),
            Query::NodesByIdsMulti(q) => {
                write!(f, "NodesByIdsMulti: count={}", q.ids.len())
            }
            Query::OutgoingEdges(q) => write!(f, "OutgoingEdges: id={}", q.id),
            Query::IncomingEdges(q) => write!(f, "IncomingEdges: id={}", q.id),
            Query::EdgeDetails(q) => write!(
                f,
                "EdgeDetails: src={}, dst={}, edge={}",
                q.source_id, q.dest_id, q.name
            ),
            Query::NodeFragments(q) => write!(f, "NodeFragments: id={}", q.id),
            Query::EdgeFragments(q) => write!(
                f,
                "EdgeFragments: src={}, dst={}, edge={}",
                q.source_id, q.dest_id, q.edge_name
            ),
            Query::AllNodes(q) => write!(f, "AllNodes: limit={}", q.limit),
            Query::AllEdges(q) => write!(f, "AllEdges: limit={}", q.limit),
        }
    }
}

#[derive(Debug)]
pub enum QueryResult {
    Nodes(Vec<NodeResult>),
    Edges(Vec<EdgeResult>),
    NodeById((NodeName, NodeSummary, Version)),
    NodesByIdsMulti(Vec<(Id, NodeName, NodeSummary, Version)>),
    OutgoingEdges(Vec<(Option<EdgeWeight>, SrcId, DstId, EdgeName, Version)>),
    IncomingEdges(Vec<(Option<EdgeWeight>, DstId, SrcId, EdgeName, Version)>),
    EdgeDetails(EdgeDetailsResult),
    NodeFragments(Vec<(TimestampMilli, FragmentContent)>),
    EdgeFragments(Vec<(TimestampMilli, FragmentContent)>),
    AllNodes(Vec<(Id, NodeName, NodeSummary, Version)>),
    AllEdges(Vec<(Option<EdgeWeight>, SrcId, DstId, EdgeName, Version)>),
}

impl RequestMeta for Query {
    type Reply = QueryResult;
    type Options = ();

    fn request_kind(&self) -> &'static str {
        match self {
            Query::Nodes(_) => "unified_nodes",
            Query::Edges(_) => "unified_edges",
            Query::NodeById(_) => "node_by_id",
            Query::NodesByIdsMulti(_) => "nodes_by_ids_multi",
            Query::OutgoingEdges(_) => "outgoing_edges",
            Query::IncomingEdges(_) => "incoming_edges",
            Query::EdgeDetails(_) => "edge_details",
            Query::NodeFragments(_) => "node_fragments",
            Query::EdgeFragments(_) => "edge_fragments",
            Query::AllNodes(_) => "all_nodes",
            Query::AllEdges(_) => "all_edges",
        }
    }
}

pub type QueryRequest = RequestEnvelope<Query>;

impl Query {
    pub async fn execute(&self, storage: &super::reader::CompositeStorage) -> Result<QueryResult> {
        match self {
            Query::Nodes(q) => execute_nodes_query(q, storage).await.map(QueryResult::Nodes),
            Query::Edges(q) => execute_edges_query(q, storage).await.map(QueryResult::Edges),
            Query::NodeById(q) => q.execute(storage.graph.storage().as_ref()).await.map(QueryResult::NodeById),
            Query::NodesByIdsMulti(q) => {
                q.execute(storage.graph.storage().as_ref()).await.map(QueryResult::NodesByIdsMulti)
            }
            Query::OutgoingEdges(q) => {
                q.execute(storage.graph.storage().as_ref()).await.map(QueryResult::OutgoingEdges)
            }
            Query::IncomingEdges(q) => {
                q.execute(storage.graph.storage().as_ref()).await.map(QueryResult::IncomingEdges)
            }
            Query::EdgeDetails(q) => {
                let (summary, weight, version) =
                    q.execute(storage.graph.storage().as_ref()).await?;
                Ok(QueryResult::EdgeDetails((
                    weight,
                    q.source_id,
                    q.dest_id,
                    q.name.clone(),
                    summary,
                    version,
                )))
            }
            Query::NodeFragments(q) => {
                q.execute(storage.graph.storage().as_ref()).await.map(QueryResult::NodeFragments)
            }
            Query::EdgeFragments(q) => {
                q.execute(storage.graph.storage().as_ref()).await.map(QueryResult::EdgeFragments)
            }
            Query::AllNodes(q) => q.execute(storage.graph.storage().as_ref()).await.map(QueryResult::AllNodes),
            Query::AllEdges(q) => q.execute(storage.graph.storage().as_ref()).await.map(QueryResult::AllEdges),
        }
    }
}

async fn execute_nodes_query(
    query: &NodesQuery,
    storage: &super::reader::CompositeStorage,
) -> Result<Vec<NodeResult>> {
    use crate::fulltext::reader::Processor as FulltextProcessor;
    use crate::graph::reader::Processor as GraphProcessor;

    let fulltext_storage = FulltextProcessor::storage(storage.fulltext.as_ref());
    let hits: Vec<NodeHit> = fulltext::query::Nodes::execute_params(&query.params, fulltext_storage).await?;

    let graph_storage = GraphProcessor::storage(storage.graph.as_ref());
    let mut results = Vec::with_capacity(hits.len());

    for hit in hits.into_iter().skip(query.offset) {
        let query = graph::query::NodeById::new(hit.id, None);
        match query.execute(graph_storage).await {
            Ok((name, summary, _version)) => results.push((hit.id, name, summary)),
            Err(_) => {
                tracing::debug!(id = %hit.id, "Node in fulltext but not in graph, skipping");
            }
        }
    }

    Ok(results)
}

async fn execute_edges_query(
    query: &EdgesQuery,
    storage: &super::reader::CompositeStorage,
) -> Result<Vec<EdgeResult>> {
    use crate::fulltext::reader::Processor as FulltextProcessor;
    use crate::graph::reader::Processor as GraphProcessor;

    let fulltext_storage = FulltextProcessor::storage(storage.fulltext.as_ref());
    let hits: Vec<EdgeHit> = fulltext::query::Edges::execute_params(&query.params, fulltext_storage).await?;

    let graph_storage = GraphProcessor::storage(storage.graph.as_ref());
    let mut results = Vec::with_capacity(hits.len());

    for hit in hits.into_iter().skip(query.offset) {
        let query = graph::query::EdgeSummaryBySrcDstName::new(
            hit.src_id,
            hit.dst_id,
            hit.edge_name.clone(),
            None,
        );
        match query.execute(graph_storage).await {
            Ok((summary, _weight, _version)) => {
                results.push((hit.src_id, hit.dst_id, hit.edge_name, summary))
            }
            Err(_) => {
                tracing::debug!(
                    src_id = %hit.src_id,
                    dst_id = %hit.dst_id,
                    edge_name = %hit.edge_name,
                    "Edge in fulltext but not in graph, skipping"
                );
            }
        }
    }

    Ok(results)
}

// ============================================================================
// Runnable<reader::Reader> Implementation for Nodes
// ============================================================================

/// Extension trait for adding offset to queries.
pub trait WithOffset: Sized {
    /// Set pagination offset (skip first N results from fulltext).
    fn with_offset(self, offset: usize) -> QueryWithOffset<Self>;
}

/// Wrapper that adds offset to any query type.
#[derive(Debug)]
pub struct QueryWithOffset<T> {
    pub query: T,
    pub offset: usize,
}

impl<T> WithOffset for T {
    fn with_offset(self, offset: usize) -> QueryWithOffset<Self> {
        QueryWithOffset {
            query: self,
            offset,
        }
    }
}

// ============================================================================
// QueryReply - typed replies for unified queries
// ============================================================================

pub trait QueryReply: Send {
    type Reply: Send + 'static;
    fn into_query(self) -> Query;
    fn from_result(result: QueryResult) -> Result<Self::Reply>;
}

#[async_trait::async_trait]
impl<Q> Runnable<super::reader::Reader> for Q
where
    Q: QueryReply,
{
    type Output = Q::Reply;

    async fn run(self, reader: &super::reader::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = oneshot::channel();
        let request = QueryRequest {
            payload: self.into_query(),
            options: (),
            reply: Some(result_tx),
            timeout: Some(timeout),
            request_id: new_request_id(),
            created_at: Instant::now(),
        };

        reader.send_query(request).await?;
        let result = result_rx.await??;
        Q::from_result(result)
    }
}

/// Result type for NodesByIdsMulti query: (id, node_name, node_summary, version)
pub type NodesByIdsMultiResult = Vec<(Id, NodeName, NodeSummary, Version)>;

/// Result type for OutgoingEdges query: (weight, src_id, dst_id, edge_name, version)
pub type OutgoingEdgesResult = Vec<(Option<EdgeWeight>, SrcId, DstId, EdgeName, Version)>;

/// Result type for IncomingEdges query: (weight, dst_id, src_id, edge_name, version)
pub type IncomingEdgesResult = Vec<(Option<EdgeWeight>, DstId, SrcId, EdgeName, Version)>;

/// Result type for AllNodes query: (id, node_name, node_summary, version)
pub type AllNodesResult = Vec<(Id, NodeName, NodeSummary, Version)>;

/// Result type for AllEdges query: (weight, src_id, dst_id, edge_name, version)
pub type AllEdgesResult = Vec<(Option<EdgeWeight>, SrcId, DstId, EdgeName, Version)>;

impl QueryReply for Nodes {
    type Reply = Vec<NodeResult>;

    fn into_query(self) -> Query {
        Query::Nodes(NodesQuery {
            params: self,
            offset: 0,
        })
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::Nodes(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for QueryWithOffset<Nodes> {
    type Reply = Vec<NodeResult>;

    fn into_query(self) -> Query {
        Query::Nodes(NodesQuery {
            params: self.query,
            offset: self.offset,
        })
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::Nodes(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for Edges {
    type Reply = Vec<EdgeResult>;

    fn into_query(self) -> Query {
        Query::Edges(EdgesQuery {
            params: self,
            offset: 0,
        })
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::Edges(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for QueryWithOffset<Edges> {
    type Reply = Vec<EdgeResult>;

    fn into_query(self) -> Query {
        Query::Edges(EdgesQuery {
            params: self.query,
            offset: self.offset,
        })
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::Edges(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for EdgeDetails {
    type Reply = EdgeDetailsResult;

    fn into_query(self) -> Query {
        Query::EdgeDetails(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::EdgeDetails(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for NodeFragments {
    type Reply = Vec<(TimestampMilli, FragmentContent)>;

    fn into_query(self) -> Query {
        Query::NodeFragments(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::NodeFragments(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for EdgeFragments {
    type Reply = Vec<(TimestampMilli, FragmentContent)>;

    fn into_query(self) -> Query {
        Query::EdgeFragments(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::EdgeFragments(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for NodeById {
    type Reply = (NodeName, NodeSummary, Version);

    fn into_query(self) -> Query {
        Query::NodeById(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::NodeById(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for NodesByIdsMulti {
    type Reply = NodesByIdsMultiResult;

    fn into_query(self) -> Query {
        Query::NodesByIdsMulti(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::NodesByIdsMulti(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for OutgoingEdges {
    type Reply = OutgoingEdgesResult;

    fn into_query(self) -> Query {
        Query::OutgoingEdges(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::OutgoingEdges(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for IncomingEdges {
    type Reply = IncomingEdgesResult;

    fn into_query(self) -> Query {
        Query::IncomingEdges(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::IncomingEdges(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for AllNodes {
    type Reply = AllNodesResult;

    fn into_query(self) -> Query {
        Query::AllNodes(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::AllNodes(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}

impl QueryReply for AllEdges {
    type Reply = AllEdgesResult;

    fn into_query(self) -> Query {
        Query::AllEdges(self)
    }

    fn from_result(result: QueryResult) -> Result<Self::Reply> {
        match result {
            QueryResult::AllEdges(result) => Ok(result),
            other => Err(anyhow::anyhow!("Unexpected query result: {:?}", other)),
        }
    }
}
