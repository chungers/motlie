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
//! | [`NodeById`] | `(NodeName, NodeSummary)` | Lookup node by ID |
//! | [`NodesByIdsMulti`] | `Vec<(Id, NodeName, NodeSummary)>` | Batch lookup nodes by IDs |
//! | [`OutgoingEdges`] | `Vec<(Option<f64>, SrcId, DstId, EdgeName)>` | Get edges from a node |
//! | [`IncomingEdges`] | `Vec<(Option<f64>, DstId, SrcId, EdgeName)>` | Get edges to a node |
//! | [`EdgeDetails`] | `(Option<f64>, SrcId, DstId, EdgeName, EdgeSummary)` | Lookup edge by topology |
//! | [`NodeFragments`] | `Vec<(TimestampMilli, FragmentContent)>` | Get node fragment history |
//! | [`EdgeFragments`] | `Vec<(TimestampMilli, FragmentContent)>` | Get edge fragment history |
//!
//! ## Graph Enumeration Queries (pagination)
//!
//! | Query | Output | Description |
//! |-------|--------|-------------|
//! | [`AllNodes`] | `Vec<(Id, NodeName, NodeSummary)>` | Enumerate all nodes |
//! | [`AllEdges`] | `Vec<(Option<f64>, SrcId, DstId, EdgeName)>` | Enumerate all edges |
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
//! let (name, summary) = NodeById::new(node_id, None)
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

use std::time::Duration;

use anyhow::Result;
use tokio::sync::oneshot;

use crate::fulltext;
use crate::graph;

// Re-export Runnable trait from reader module
pub use crate::reader::Runnable;
use crate::graph::schema::{
    DstId, EdgeName, EdgeSummary, FragmentContent, NodeName, NodeSummary, SrcId,
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
/// - With `graph::Reader`: returns `Vec<(Id, NodeName, NodeSummary)>`
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
/// Returns `EdgeDetailsResult` (Option<f64>, SrcId, DstId, EdgeName, EdgeSummary).
///
/// - With `graph::Reader`: use `.run()` method (returns `(EdgeSummary, Option<f64>)`)
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
/// if let Some((last_id, _, _)) = nodes.last() {
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

/// Result type for EdgeDetails query: (weight, src_id, dst_id, edge_name, edge_summary)
pub type EdgeDetailsResult = (Option<f64>, SrcId, DstId, EdgeName, EdgeSummary);

// ============================================================================
// Query Enum for Dispatch
// ============================================================================

/// Wrapper enum for dispatching queries through the unified reader.
///
/// This wraps query types and adds oneshot channels for result delivery.
/// Used internally by the reader infrastructure. Users should not construct
/// this directly - use `Runnable::run()` instead.
#[derive(Debug)]
#[doc(hidden)]
#[allow(private_interfaces)]
pub enum Query {
    // Fulltext search queries (with hydration)
    Nodes(NodesDispatch),
    Edges(EdgesDispatch),
    // Graph queries (forwarded to graph storage)
    NodeById(NodeByIdDispatch),
    NodesByIdsMulti(NodesByIdsMultiDispatch),
    OutgoingEdges(OutgoingEdgesDispatch),
    IncomingEdges(IncomingEdgesDispatch),
    EdgeDetails(EdgeDetailsDispatch),
    NodeFragments(NodeFragmentsDispatch),
    EdgeFragments(EdgeFragmentsDispatch),
    // Graph enumeration queries
    AllNodes(AllNodesDispatch),
    AllEdges(AllEdgesDispatch),
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
            Query::NodeById(q) => write!(f, "NodeById: id={}", q.params.id),
            Query::NodesByIdsMulti(q) => {
                write!(f, "NodesByIdsMulti: count={}", q.params.ids.len())
            }
            Query::OutgoingEdges(q) => write!(f, "OutgoingEdges: id={}", q.params.id),
            Query::IncomingEdges(q) => write!(f, "IncomingEdges: id={}", q.params.id),
            Query::EdgeDetails(q) => write!(
                f,
                "EdgeDetails: src={}, dst={}, edge={}",
                q.params.source_id, q.params.dest_id, q.params.name
            ),
            Query::NodeFragments(q) => write!(f, "NodeFragments: id={}", q.params.id),
            Query::EdgeFragments(q) => write!(
                f,
                "EdgeFragments: src={}, dst={}, edge={}",
                q.params.source_id, q.params.dest_id, q.params.edge_name
            ),
            Query::AllNodes(q) => write!(f, "AllNodes: limit={}", q.params.limit),
            Query::AllEdges(q) => write!(f, "AllEdges: limit={}", q.params.limit),
        }
    }
}

// ============================================================================
// NodesDispatch - Internal wrapper for query dispatch
// ============================================================================

/// Internal dispatch wrapper for unified Nodes query execution.
/// Users interact with `fulltext::Nodes` directly via the `Runnable` trait.
#[derive(Debug)]
pub(crate) struct NodesDispatch {
    /// The underlying fulltext Nodes query params
    pub(crate) params: fulltext::query::Nodes,

    /// Offset for pagination (skip first N results from fulltext)
    pub(crate) offset: usize,

    /// Channel to send the result back to the client
    result_tx: oneshot::Sender<Result<Vec<NodeResult>>>,
}

impl NodesDispatch {
    /// Send the result back to the client (consumes self)
    fn send_result(self, result: Result<Vec<NodeResult>>) {
        let _ = self.result_tx.send(result);
    }

    /// Execute against composite storage
    pub(crate) async fn execute(
        &self,
        storage: &super::reader::CompositeStorage,
    ) -> Result<Vec<NodeResult>> {
        // Step 1: REUSE existing fulltext query execution to get hits
        use crate::fulltext::query::NodesDispatch as FulltextNodesDispatch;
        use crate::fulltext::reader::Processor as FulltextProcessor;
        use crate::graph::reader::Processor as GraphProcessor;

        let fulltext_storage = FulltextProcessor::storage(storage.fulltext.as_ref());
        let hits: Vec<NodeHit> =
            FulltextNodesDispatch::execute_params(&self.params, fulltext_storage).await?;

        // Step 2: Hydrate each hit from graph storage
        use crate::graph::query::NodeByIdDispatch;

        let graph_storage = GraphProcessor::storage(storage.graph.as_ref());
        let mut results = Vec::with_capacity(hits.len());

        for hit in hits.into_iter().skip(self.offset) {
            let query = graph::query::NodeById::new(hit.id, None);
            match NodeByIdDispatch::execute_params(&query, graph_storage).await {
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

    /// Process and send result
    pub(crate) async fn process_and_send(self, storage: &super::reader::CompositeStorage) {
        let result = self.execute(storage).await;
        self.send_result(result);
    }
}

// ============================================================================
// EdgesDispatch - Internal wrapper for query dispatch
// ============================================================================

/// Internal dispatch wrapper for unified Edges query execution.
/// Users interact with `fulltext::Edges` directly via the `Runnable` trait.
#[derive(Debug)]
pub(crate) struct EdgesDispatch {
    /// The underlying fulltext Edges query params
    pub(crate) params: fulltext::query::Edges,

    /// Offset for pagination (skip first N results from fulltext)
    pub(crate) offset: usize,

    /// Channel to send the result back to the client
    result_tx: oneshot::Sender<Result<Vec<EdgeResult>>>,
}

impl EdgesDispatch {
    /// Send the result back to the client (consumes self)
    fn send_result(self, result: Result<Vec<EdgeResult>>) {
        let _ = self.result_tx.send(result);
    }

    /// Execute against composite storage
    pub(crate) async fn execute(
        &self,
        storage: &super::reader::CompositeStorage,
    ) -> Result<Vec<EdgeResult>> {
        // Step 1: REUSE existing fulltext query execution to get hits
        use crate::fulltext::query::EdgesDispatch as FulltextEdgesDispatch;
        use crate::fulltext::reader::Processor as FulltextProcessor;
        use crate::graph::reader::Processor as GraphProcessor;

        let fulltext_storage = FulltextProcessor::storage(storage.fulltext.as_ref());
        let hits: Vec<EdgeHit> =
            FulltextEdgesDispatch::execute_params(&self.params, fulltext_storage).await?;

        // Step 2: Hydrate each hit from graph storage
        use crate::graph::query::EdgeSummaryBySrcDstNameDispatch;

        let graph_storage = GraphProcessor::storage(storage.graph.as_ref());
        let mut results = Vec::with_capacity(hits.len());

        for hit in hits.into_iter().skip(self.offset) {
            let query = graph::query::EdgeSummaryBySrcDstName::new(
                hit.src_id,
                hit.dst_id,
                hit.edge_name.clone(),
                None,
            );
            match EdgeSummaryBySrcDstNameDispatch::execute_params(&query, graph_storage).await {
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

    /// Process and send result
    pub(crate) async fn process_and_send(self, storage: &super::reader::CompositeStorage) {
        let result = self.execute(storage).await;
        self.send_result(result);
    }
}

// ============================================================================
// NodeByIdDispatch - Internal wrapper for NodeById dispatch
// ============================================================================

/// Internal dispatch wrapper for NodeById query execution.
#[derive(Debug)]
pub(crate) struct NodeByIdDispatch {
    /// The underlying graph query params
    pub(crate) params: graph::query::NodeById,

    /// Channel to send the result back to the client
    result_tx: oneshot::Sender<Result<(NodeName, NodeSummary)>>,
}

impl NodeByIdDispatch {
    fn send_result(self, result: Result<(NodeName, NodeSummary)>) {
        let _ = self.result_tx.send(result);
    }

    pub(crate) async fn execute(
        &self,
        storage: &super::reader::CompositeStorage,
    ) -> Result<(NodeName, NodeSummary)> {
        use crate::graph::query::NodeByIdDispatch as GraphNodeByIdDispatch;
        use crate::graph::reader::Processor as GraphProcessor;

        let graph_storage = GraphProcessor::storage(storage.graph.as_ref());
        GraphNodeByIdDispatch::execute_params(&self.params, graph_storage).await
    }

    pub(crate) async fn process_and_send(self, storage: &super::reader::CompositeStorage) {
        let result = self.execute(storage).await;
        self.send_result(result);
    }
}

// ============================================================================
// NodesByIdsMultiDispatch - Internal wrapper for NodesByIdsMulti dispatch
// ============================================================================

/// Internal dispatch wrapper for NodesByIdsMulti query execution.
#[derive(Debug)]
pub(crate) struct NodesByIdsMultiDispatch {
    /// The underlying graph query params
    pub(crate) params: graph::query::NodesByIdsMulti,

    /// Channel to send the result back to the client
    /// Returns (id, node_name, node_summary) tuples for found nodes
    result_tx: oneshot::Sender<Result<Vec<(Id, NodeName, NodeSummary)>>>,
}

impl NodesByIdsMultiDispatch {
    fn send_result(self, result: Result<Vec<(Id, NodeName, NodeSummary)>>) {
        let _ = self.result_tx.send(result);
    }

    pub(crate) async fn execute(
        &self,
        storage: &super::reader::CompositeStorage,
    ) -> Result<Vec<(Id, NodeName, NodeSummary)>> {
        use crate::graph::query::NodesByIdsMultiDispatch as GraphNodesByIdsMultiDispatch;
        use crate::graph::reader::Processor as GraphProcessor;

        let graph_storage = GraphProcessor::storage(storage.graph.as_ref());
        GraphNodesByIdsMultiDispatch::execute_params(&self.params, graph_storage).await
    }

    pub(crate) async fn process_and_send(self, storage: &super::reader::CompositeStorage) {
        let result = self.execute(storage).await;
        self.send_result(result);
    }
}

// ============================================================================
// OutgoingEdgesDispatch - Internal wrapper for OutgoingEdges dispatch
// ============================================================================

/// Internal dispatch wrapper for OutgoingEdges query execution.
#[derive(Debug)]
pub(crate) struct OutgoingEdgesDispatch {
    /// The underlying graph query params
    pub(crate) params: graph::query::OutgoingEdges,

    /// Channel to send the result back to the client
    /// Returns (weight, src_id, dst_id, edge_name) tuples
    result_tx: oneshot::Sender<Result<Vec<(Option<f64>, SrcId, DstId, EdgeName)>>>,
}

impl OutgoingEdgesDispatch {
    fn send_result(self, result: Result<Vec<(Option<f64>, SrcId, DstId, EdgeName)>>) {
        let _ = self.result_tx.send(result);
    }

    pub(crate) async fn execute(
        &self,
        storage: &super::reader::CompositeStorage,
    ) -> Result<Vec<(Option<f64>, SrcId, DstId, EdgeName)>> {
        use crate::graph::query::OutgoingEdgesDispatch as GraphOutgoingEdgesDispatch;
        use crate::graph::reader::Processor as GraphProcessor;

        let graph_storage = GraphProcessor::storage(storage.graph.as_ref());
        GraphOutgoingEdgesDispatch::execute_params(&self.params, graph_storage).await
    }

    pub(crate) async fn process_and_send(self, storage: &super::reader::CompositeStorage) {
        let result = self.execute(storage).await;
        self.send_result(result);
    }
}

// ============================================================================
// IncomingEdgesDispatch - Internal wrapper for IncomingEdges dispatch
// ============================================================================

/// Internal dispatch wrapper for IncomingEdges query execution.
#[derive(Debug)]
pub(crate) struct IncomingEdgesDispatch {
    /// The underlying graph query params
    pub(crate) params: graph::query::IncomingEdges,

    /// Channel to send the result back to the client
    /// Returns (weight, dst_id, src_id, edge_name) tuples
    result_tx: oneshot::Sender<Result<Vec<(Option<f64>, DstId, SrcId, EdgeName)>>>,
}

impl IncomingEdgesDispatch {
    fn send_result(self, result: Result<Vec<(Option<f64>, DstId, SrcId, EdgeName)>>) {
        let _ = self.result_tx.send(result);
    }

    pub(crate) async fn execute(
        &self,
        storage: &super::reader::CompositeStorage,
    ) -> Result<Vec<(Option<f64>, DstId, SrcId, EdgeName)>> {
        use crate::graph::query::IncomingEdgesDispatch as GraphIncomingEdgesDispatch;
        use crate::graph::reader::Processor as GraphProcessor;

        let graph_storage = GraphProcessor::storage(storage.graph.as_ref());
        GraphIncomingEdgesDispatch::execute_params(&self.params, graph_storage).await
    }

    pub(crate) async fn process_and_send(self, storage: &super::reader::CompositeStorage) {
        let result = self.execute(storage).await;
        self.send_result(result);
    }
}

// ============================================================================
// EdgeDetailsDispatch - Internal wrapper for EdgeSummaryBySrcDstName dispatch
// ============================================================================

/// Internal dispatch wrapper for EdgeSummaryBySrcDstName query execution.
#[derive(Debug)]
pub(crate) struct EdgeDetailsDispatch {
    /// The underlying graph query params
    pub(crate) params: graph::query::EdgeSummaryBySrcDstName,

    /// Channel to send the result back to the client
    /// Returns (weight, src_id, dst_id, edge_name, edge_summary) tuples
    result_tx: oneshot::Sender<Result<EdgeDetailsResult>>,
}

impl EdgeDetailsDispatch {
    fn send_result(self, result: Result<EdgeDetailsResult>) {
        let _ = self.result_tx.send(result);
    }

    pub(crate) async fn execute(
        &self,
        storage: &super::reader::CompositeStorage,
    ) -> Result<EdgeDetailsResult> {
        use crate::graph::query::EdgeSummaryBySrcDstNameDispatch;
        use crate::graph::reader::Processor as GraphProcessor;

        let graph_storage = GraphProcessor::storage(storage.graph.as_ref());
        let (summary, weight) =
            EdgeSummaryBySrcDstNameDispatch::execute_params(&self.params, graph_storage).await?;

        // Return EdgeDetailsResult with weight included
        Ok((
            weight,
            self.params.source_id,
            self.params.dest_id,
            self.params.name.clone(),
            summary,
        ))
    }

    pub(crate) async fn process_and_send(self, storage: &super::reader::CompositeStorage) {
        let result = self.execute(storage).await;
        self.send_result(result);
    }
}

// ============================================================================
// NodeFragmentsDispatch - Internal wrapper for NodeFragmentsByIdTimeRange dispatch
// ============================================================================

/// Internal dispatch wrapper for NodeFragmentsByIdTimeRange query execution.
#[derive(Debug)]
pub(crate) struct NodeFragmentsDispatch {
    /// The underlying graph query params
    pub(crate) params: graph::query::NodeFragmentsByIdTimeRange,

    /// Channel to send the result back to the client
    result_tx: oneshot::Sender<Result<Vec<(TimestampMilli, FragmentContent)>>>,
}

impl NodeFragmentsDispatch {
    fn send_result(self, result: Result<Vec<(TimestampMilli, FragmentContent)>>) {
        let _ = self.result_tx.send(result);
    }

    pub(crate) async fn execute(
        &self,
        storage: &super::reader::CompositeStorage,
    ) -> Result<Vec<(TimestampMilli, FragmentContent)>> {
        use crate::graph::query::NodeFragmentsByIdTimeRangeDispatch;
        use crate::graph::reader::Processor as GraphProcessor;

        let graph_storage = GraphProcessor::storage(storage.graph.as_ref());
        NodeFragmentsByIdTimeRangeDispatch::execute_params(&self.params, graph_storage).await
    }

    pub(crate) async fn process_and_send(self, storage: &super::reader::CompositeStorage) {
        let result = self.execute(storage).await;
        self.send_result(result);
    }
}

// ============================================================================
// EdgeFragmentsDispatch - Internal wrapper for EdgeFragmentsByIdTimeRange dispatch
// ============================================================================

/// Internal dispatch wrapper for EdgeFragmentsByIdTimeRange query execution.
#[derive(Debug)]
pub(crate) struct EdgeFragmentsDispatch {
    /// The underlying graph query params
    pub(crate) params: graph::query::EdgeFragmentsByIdTimeRange,

    /// Channel to send the result back to the client
    result_tx: oneshot::Sender<Result<Vec<(TimestampMilli, FragmentContent)>>>,
}

impl EdgeFragmentsDispatch {
    fn send_result(self, result: Result<Vec<(TimestampMilli, FragmentContent)>>) {
        let _ = self.result_tx.send(result);
    }

    pub(crate) async fn execute(
        &self,
        storage: &super::reader::CompositeStorage,
    ) -> Result<Vec<(TimestampMilli, FragmentContent)>> {
        use crate::graph::query::EdgeFragmentsByIdTimeRangeDispatch;
        use crate::graph::reader::Processor as GraphProcessor;

        let graph_storage = GraphProcessor::storage(storage.graph.as_ref());
        EdgeFragmentsByIdTimeRangeDispatch::execute_params(&self.params, graph_storage).await
    }

    pub(crate) async fn process_and_send(self, storage: &super::reader::CompositeStorage) {
        let result = self.execute(storage).await;
        self.send_result(result);
    }
}

// ============================================================================
// AllNodesDispatch - Internal wrapper for AllNodes dispatch
// ============================================================================

/// Internal dispatch wrapper for AllNodes query execution.
#[derive(Debug)]
pub(crate) struct AllNodesDispatch {
    /// The underlying graph query params
    pub(crate) params: graph::query::AllNodes,

    /// Channel to send the result back to the client
    result_tx: oneshot::Sender<Result<Vec<(Id, NodeName, NodeSummary)>>>,
}

impl AllNodesDispatch {
    fn send_result(self, result: Result<Vec<(Id, NodeName, NodeSummary)>>) {
        let _ = self.result_tx.send(result);
    }

    pub(crate) async fn execute(
        &self,
        storage: &super::reader::CompositeStorage,
    ) -> Result<Vec<(Id, NodeName, NodeSummary)>> {
        use crate::graph::query::AllNodesDispatch as GraphAllNodesDispatch;
        use crate::graph::reader::Processor as GraphProcessor;

        let graph_storage = GraphProcessor::storage(storage.graph.as_ref());
        GraphAllNodesDispatch::execute_params(&self.params, graph_storage).await
    }

    pub(crate) async fn process_and_send(self, storage: &super::reader::CompositeStorage) {
        let result = self.execute(storage).await;
        self.send_result(result);
    }
}

// ============================================================================
// AllEdgesDispatch - Internal wrapper for AllEdges dispatch
// ============================================================================

/// Internal dispatch wrapper for AllEdges query execution.
#[derive(Debug)]
pub(crate) struct AllEdgesDispatch {
    /// The underlying graph query params
    pub(crate) params: graph::query::AllEdges,

    /// Channel to send the result back to the client
    result_tx: oneshot::Sender<Result<Vec<(Option<f64>, SrcId, DstId, EdgeName)>>>,
}

impl AllEdgesDispatch {
    fn send_result(self, result: Result<Vec<(Option<f64>, SrcId, DstId, EdgeName)>>) {
        let _ = self.result_tx.send(result);
    }

    pub(crate) async fn execute(
        &self,
        storage: &super::reader::CompositeStorage,
    ) -> Result<Vec<(Option<f64>, SrcId, DstId, EdgeName)>> {
        use crate::graph::query::AllEdgesDispatch as GraphAllEdgesDispatch;
        use crate::graph::reader::Processor as GraphProcessor;

        let graph_storage = GraphProcessor::storage(storage.graph.as_ref());
        GraphAllEdgesDispatch::execute_params(&self.params, graph_storage).await
    }

    pub(crate) async fn process_and_send(self, storage: &super::reader::CompositeStorage) {
        let result = self.execute(storage).await;
        self.send_result(result);
    }
}

// ============================================================================
// QueryProcessor Trait for Search enum
// ============================================================================

/// Trait for processing queries polymorphically.
#[async_trait::async_trait]
pub trait QueryProcessor: Send {
    /// Process the query and send the result (consumes self)
    async fn process_and_send(self, processor: &super::reader::CompositeStorage);
}

#[async_trait::async_trait]
impl QueryProcessor for Query {
    async fn process_and_send(self, processor: &super::reader::CompositeStorage) {
        match self {
            Query::Nodes(q) => q.process_and_send(processor).await,
            Query::Edges(q) => q.process_and_send(processor).await,
            Query::NodeById(q) => q.process_and_send(processor).await,
            Query::NodesByIdsMulti(q) => q.process_and_send(processor).await,
            Query::OutgoingEdges(q) => q.process_and_send(processor).await,
            Query::IncomingEdges(q) => q.process_and_send(processor).await,
            Query::EdgeDetails(q) => q.process_and_send(processor).await,
            Query::NodeFragments(q) => q.process_and_send(processor).await,
            Query::EdgeFragments(q) => q.process_and_send(processor).await,
            Query::AllNodes(q) => q.process_and_send(processor).await,
            Query::AllEdges(q) => q.process_and_send(processor).await,
        }
    }
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

#[async_trait::async_trait]
impl Runnable<super::reader::Reader> for Nodes {
    type Output = Vec<NodeResult>;

    async fn run(self, reader: &super::reader::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = oneshot::channel();

        let dispatch = NodesDispatch {
            params: self,
            offset: 0,
            result_tx,
        };

        reader.send_query(Query::Nodes(dispatch)).await?;

        match tokio::time::timeout(timeout, result_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("Query channel closed")),
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
        }
    }
}

#[async_trait::async_trait]
impl Runnable<super::reader::Reader> for QueryWithOffset<Nodes> {
    type Output = Vec<NodeResult>;

    async fn run(self, reader: &super::reader::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = oneshot::channel();

        let dispatch = NodesDispatch {
            params: self.query,
            offset: self.offset,
            result_tx,
        };

        reader.send_query(Query::Nodes(dispatch)).await?;

        match tokio::time::timeout(timeout, result_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("Query channel closed")),
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
        }
    }
}

// ============================================================================
// Runnable<reader::Reader> Implementation for Edges
// ============================================================================

#[async_trait::async_trait]
impl Runnable<super::reader::Reader> for Edges {
    type Output = Vec<EdgeResult>;

    async fn run(self, reader: &super::reader::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = oneshot::channel();

        let dispatch = EdgesDispatch {
            params: self,
            offset: 0,
            result_tx,
        };

        reader.send_query(Query::Edges(dispatch)).await?;

        match tokio::time::timeout(timeout, result_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("Query channel closed")),
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
        }
    }
}

#[async_trait::async_trait]
impl Runnable<super::reader::Reader> for QueryWithOffset<Edges> {
    type Output = Vec<EdgeResult>;

    async fn run(self, reader: &super::reader::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = oneshot::channel();

        let dispatch = EdgesDispatch {
            params: self.query,
            offset: self.offset,
            result_tx,
        };

        reader.send_query(Query::Edges(dispatch)).await?;

        match tokio::time::timeout(timeout, result_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("Query channel closed")),
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
        }
    }
}

// ============================================================================
// Runnable<reader::Reader> Implementation for EdgeDetails (EdgeSummaryBySrcDstName)
// ============================================================================

#[async_trait::async_trait]
impl Runnable<super::reader::Reader> for EdgeDetails {
    type Output = EdgeDetailsResult;

    async fn run(self, reader: &super::reader::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = oneshot::channel();

        let dispatch = EdgeDetailsDispatch {
            params: self,
            result_tx,
        };

        reader.send_query(Query::EdgeDetails(dispatch)).await?;

        match tokio::time::timeout(timeout, result_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("Query channel closed")),
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
        }
    }
}

// ============================================================================
// Runnable<reader::Reader> Implementation for NodeFragments (NodeFragmentsByIdTimeRange)
// ============================================================================

#[async_trait::async_trait]
impl Runnable<super::reader::Reader> for NodeFragments {
    type Output = Vec<(TimestampMilli, FragmentContent)>;

    async fn run(self, reader: &super::reader::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = oneshot::channel();

        let dispatch = NodeFragmentsDispatch {
            params: self,
            result_tx,
        };

        reader.send_query(Query::NodeFragments(dispatch)).await?;

        match tokio::time::timeout(timeout, result_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("Query channel closed")),
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
        }
    }
}

// ============================================================================
// Runnable<reader::Reader> Implementation for EdgeFragments (EdgeFragmentsByIdTimeRange)
// ============================================================================

#[async_trait::async_trait]
impl Runnable<super::reader::Reader> for EdgeFragments {
    type Output = Vec<(TimestampMilli, FragmentContent)>;

    async fn run(self, reader: &super::reader::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = oneshot::channel();

        let dispatch = EdgeFragmentsDispatch {
            params: self,
            result_tx,
        };

        reader.send_query(Query::EdgeFragments(dispatch)).await?;

        match tokio::time::timeout(timeout, result_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("Query channel closed")),
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
        }
    }
}

// ============================================================================
// Runnable<reader::Reader> Implementation for NodeById
// ============================================================================

#[async_trait::async_trait]
impl Runnable<super::reader::Reader> for NodeById {
    type Output = (NodeName, NodeSummary);

    async fn run(self, reader: &super::reader::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = oneshot::channel();

        let dispatch = NodeByIdDispatch {
            params: self,
            result_tx,
        };

        reader.send_query(Query::NodeById(dispatch)).await?;

        match tokio::time::timeout(timeout, result_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("Query channel closed")),
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
        }
    }
}

// ============================================================================
// Runnable<reader::Reader> Implementation for NodesByIdsMulti
// ============================================================================

/// Result type for NodesByIdsMulti query: (id, node_name, node_summary)
pub type NodesByIdsMultiResult = Vec<(Id, NodeName, NodeSummary)>;

#[async_trait::async_trait]
impl Runnable<super::reader::Reader> for NodesByIdsMulti {
    type Output = NodesByIdsMultiResult;

    async fn run(self, reader: &super::reader::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = oneshot::channel();

        let dispatch = NodesByIdsMultiDispatch {
            params: self,
            result_tx,
        };

        reader.send_query(Query::NodesByIdsMulti(dispatch)).await?;

        match tokio::time::timeout(timeout, result_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("Query channel closed")),
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
        }
    }
}

// ============================================================================
// Runnable<reader::Reader> Implementation for OutgoingEdges
// ============================================================================

/// Result type for OutgoingEdges query: (weight, src_id, dst_id, edge_name)
pub type OutgoingEdgesResult = Vec<(Option<f64>, SrcId, DstId, EdgeName)>;

#[async_trait::async_trait]
impl Runnable<super::reader::Reader> for OutgoingEdges {
    type Output = OutgoingEdgesResult;

    async fn run(self, reader: &super::reader::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = oneshot::channel();

        let dispatch = OutgoingEdgesDispatch {
            params: self,
            result_tx,
        };

        reader.send_query(Query::OutgoingEdges(dispatch)).await?;

        match tokio::time::timeout(timeout, result_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("Query channel closed")),
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
        }
    }
}

// ============================================================================
// Runnable<reader::Reader> Implementation for IncomingEdges
// ============================================================================

/// Result type for IncomingEdges query: (weight, dst_id, src_id, edge_name)
pub type IncomingEdgesResult = Vec<(Option<f64>, DstId, SrcId, EdgeName)>;

#[async_trait::async_trait]
impl Runnable<super::reader::Reader> for IncomingEdges {
    type Output = IncomingEdgesResult;

    async fn run(self, reader: &super::reader::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = oneshot::channel();

        let dispatch = IncomingEdgesDispatch {
            params: self,
            result_tx,
        };

        reader.send_query(Query::IncomingEdges(dispatch)).await?;

        match tokio::time::timeout(timeout, result_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("Query channel closed")),
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
        }
    }
}

// ============================================================================
// Runnable<reader::Reader> Implementation for AllNodes
// ============================================================================

/// Result type for AllNodes query: (id, node_name, node_summary)
pub type AllNodesResult = Vec<(Id, NodeName, NodeSummary)>;

#[async_trait::async_trait]
impl Runnable<super::reader::Reader> for AllNodes {
    type Output = AllNodesResult;

    async fn run(self, reader: &super::reader::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = oneshot::channel();

        let dispatch = AllNodesDispatch {
            params: self,
            result_tx,
        };

        reader.send_query(Query::AllNodes(dispatch)).await?;

        match tokio::time::timeout(timeout, result_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("Query channel closed")),
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
        }
    }
}

// ============================================================================
// Runnable<reader::Reader> Implementation for AllEdges
// ============================================================================

/// Result type for AllEdges query: (weight, src_id, dst_id, edge_name)
pub type AllEdgesResult = Vec<(Option<f64>, SrcId, DstId, EdgeName)>;

#[async_trait::async_trait]
impl Runnable<super::reader::Reader> for AllEdges {
    type Output = AllEdgesResult;

    async fn run(self, reader: &super::reader::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = oneshot::channel();

        let dispatch = AllEdgesDispatch {
            params: self,
            result_tx,
        };

        reader.send_query(Query::AllEdges(dispatch)).await?;

        match tokio::time::timeout(timeout, result_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("Query channel closed")),
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
        }
    }
}
