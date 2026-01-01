//! Query module providing query types and their business logic implementations.
//!
//! This module contains only business logic - query type definitions and their
//! QueryExecutor implementations. Infrastructure (traits, Reader, Consumer, spawn
//! functions) is in the `reader` module.
//!
//! # Transaction Support
//!
//! Query types implement [`TransactionQueryExecutor`] to support read-your-writes
//! semantics within a transaction scope. This allows queries to see uncommitted
//! mutations in the same transaction.

use anyhow::Result;
use std::ops::Bound;
use std::time::Duration;
use tokio::sync::oneshot;

use super::name_hash::NameHash;
use super::reader::{Processor, QueryExecutor, QueryProcessor};
use super::scan::{self, Visitable};
use crate::reader::Runnable;
use super::schema::{
    self, DstId, EdgeName, EdgeSummary, FragmentContent, Names, NamesCfKey,
    NodeName, NodeSummary, SrcId,
};
use super::ColumnFamilyRecord;
use super::Storage;
use crate::{Id, TimestampMilli};

// ============================================================================
// Name Resolution Helpers
// ============================================================================

/// Resolve a NameHash to its full String name.
///
/// Uses the in-memory NameCache first for O(1) lookup, falling back to
/// Names CF lookup only for cache misses. On cache miss, the name is
/// added to the cache for future lookups.
///
/// This is the primary function for QueryExecutor implementations that
/// have access to Storage.
fn resolve_name(storage: &Storage, name_hash: NameHash) -> Result<String> {
    // Check cache first (O(1) DashMap lookup)
    let cache = storage.name_cache();
    if let Some(name) = cache.get(&name_hash) {
        return Ok((*name).clone());
    }

    // Cache miss: fetch from Names CF
    let key_bytes = Names::key_to_bytes(&NamesCfKey(name_hash));

    let value_bytes = if let Ok(db) = storage.db() {
        let names_cf = db
            .cf_handle(Names::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Names CF not found"))?;
        db.get_cf(names_cf, &key_bytes)?
    } else {
        let txn_db = storage.transaction_db()?;
        let names_cf = txn_db
            .cf_handle(Names::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Names CF not found"))?;
        txn_db.get_cf(names_cf, &key_bytes)?
    };

    let value_bytes = value_bytes
        .ok_or_else(|| anyhow::anyhow!("Name not found for hash: {}", name_hash))?;

    let value = Names::value_from_bytes(&value_bytes)?;
    let name = value.0;

    // Populate cache for future lookups
    cache.insert(name_hash, name.clone());

    Ok(name)
}

/// Resolve a NameHash from a Transaction (sees uncommitted writes).
///
/// For TransactionQueryExecutor implementations that need read-your-writes
/// semantics. Uses cache first, then falls back to transaction lookup.
fn resolve_name_from_txn(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    name_hash: NameHash,
    cache: &super::name_hash::NameCache,
) -> Result<String> {
    // Check cache first (O(1) DashMap lookup)
    if let Some(name) = cache.get(&name_hash) {
        return Ok((*name).clone());
    }

    // Cache miss: fetch from transaction (sees uncommitted writes)
    let names_cf = txn_db
        .cf_handle(Names::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Names CF not found"))?;

    let key_bytes = Names::key_to_bytes(&NamesCfKey(name_hash));

    let value_bytes = txn
        .get_cf(names_cf, &key_bytes)?
        .ok_or_else(|| anyhow::anyhow!("Name not found for hash: {}", name_hash))?;

    let value = Names::value_from_bytes(&value_bytes)?;
    let name = value.0;

    // Populate cache for future lookups
    cache.insert(name_hash, name.clone());

    Ok(name)
}

// ============================================================================
// TransactionQueryExecutor Trait
// ============================================================================

/// Trait for queries that can execute within a transaction scope.
///
/// Unlike [`QueryExecutor`] (which takes `&Storage` and routes through
/// MPSC channels), this trait executes directly against an active
/// RocksDB transaction, enabling read-your-writes semantics.
///
/// # When to Use
///
/// Use this trait when you need to:
/// - Read data that was just written in the same transaction
/// - Execute queries within an atomic transaction scope
/// - Implement algorithms that interleave reads and writes (e.g., HNSW insert)
///
/// # Implementation Pattern
///
/// Each query type implements this trait alongside `QueryExecutor`:
///
/// ```rust,ignore
/// impl TransactionQueryExecutor for NodeById {
///     type Output = (NodeName, NodeSummary);
///
///     fn execute_in_transaction(
///         &self,
///         txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
///         txn_db: &rocksdb::TransactionDB,
///     ) -> Result<Self::Output> {
///         let cf = txn_db.cf_handle(Nodes::CF_NAME)?;
///         // Use txn.get_cf() to see uncommitted writes
///         let value = txn.get_cf(cf, &key)?;
///         // ... deserialize and return
///     }
/// }
/// ```
///
/// # Example Usage
///
/// ```rust,ignore
/// let mut txn = writer.transaction()?;
///
/// // Write a node
/// txn.write(AddNode { id, ... })?;
///
/// // Read sees the uncommitted node!
/// let (name, summary) = txn.read(NodeById::new(id, None))?;
///
/// txn.commit()?;
/// ```
pub trait TransactionQueryExecutor: Send + Sync {
    /// The output type of this query.
    type Output: Send;

    /// Execute query within a transaction (sees uncommitted writes).
    ///
    /// # Arguments
    ///
    /// * `txn` - Active RocksDB transaction
    /// * `txn_db` - TransactionDB for column family handles
    /// * `cache` - Name cache for efficient hash-to-name resolution
    ///
    /// # Returns
    ///
    /// Query result, which may include data from uncommitted writes
    /// within the same transaction.
    fn execute_in_transaction(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        cache: &super::name_hash::NameCache,
    ) -> Result<Self::Output>;
}

/// Helper function to check if a timestamp falls within a time range
fn timestamp_in_range(
    ts: TimestampMilli,
    time_range: &(Bound<TimestampMilli>, Bound<TimestampMilli>),
) -> bool {
    let (start_bound, end_bound) = time_range;

    let start_ok = match start_bound {
        Bound::Unbounded => true,
        Bound::Included(start) => ts.0 >= start.0,
        Bound::Excluded(start) => ts.0 > start.0,
    };

    let end_ok = match end_bound {
        Bound::Unbounded => true,
        Bound::Included(end) => ts.0 <= end.0,
        Bound::Excluded(end) => ts.0 < end.0,
    };

    start_ok && end_ok
}

/// Macro to iterate over a column family in both readonly and readwrite storage modes
/// This reduces boilerplate for the common pattern of handling both DB and TransactionDB
/// The process_body should be a block of code that can access variables from the enclosing scope
macro_rules! iterate_cf {
    ($storage:expr, $cf_type:ty, $start_key:expr, |$item:ident| $process_body:block) => {{
        if let Ok(db) = $storage.db() {
            let cf = db
                .cf_handle(<$cf_type>::CF_NAME)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Column family '{}' not found",
                        <$cf_type>::CF_NAME
                    )
                })?;

            let iter = db.iterator_cf(
                cf,
                rocksdb::IteratorMode::From(&$start_key, rocksdb::Direction::Forward),
            );

            for $item in iter $process_body
        } else {
            let txn_db = $storage.transaction_db()?;
            let cf = txn_db
                .cf_handle(<$cf_type>::CF_NAME)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Column family '{}' not found",
                        <$cf_type>::CF_NAME
                    )
                })?;

            let iter = txn_db.iterator_cf(
                cf,
                rocksdb::IteratorMode::From(&$start_key, rocksdb::Direction::Forward),
            );

            for $item in iter $process_body
        }
    }};
}

// ============================================================================
// Query Enum
// ============================================================================

/// Query enum representing all possible query types.
/// Uses dispatch wrappers internally for channel/timeout handling.
///
/// **Note**: This is internal infrastructure for the query dispatch pipeline.
/// Users interact with query parameter structs directly via the `Runnable` trait.
/// This enum is public because it appears in reader function signatures, but users
/// should not construct variants directly.
#[derive(Debug)]
#[doc(hidden)]
#[allow(private_interfaces)]
pub enum Query {
    NodeById(NodeByIdDispatch),
    NodesByIdsMulti(NodesByIdsMultiDispatch),
    EdgeSummaryBySrcDstName(EdgeSummaryBySrcDstNameDispatch),
    NodeFragmentsByIdTimeRange(NodeFragmentsByIdTimeRangeDispatch),
    EdgeFragmentsByIdTimeRange(EdgeFragmentsByIdTimeRangeDispatch),
    OutgoingEdges(OutgoingEdgesDispatch),
    IncomingEdges(IncomingEdgesDispatch),
    AllNodes(AllNodesDispatch),
    AllEdges(AllEdgesDispatch),
}

impl std::fmt::Display for Query {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Query::NodeById(q) => write!(f, "NodeById: id={}", q.params.id),
            Query::NodesByIdsMulti(q) => {
                write!(f, "NodesByIdsMulti: count={}", q.params.ids.len())
            }
            Query::EdgeSummaryBySrcDstName(q) => write!(
                f,
                "EdgeBySrcDstName: source={}, dest={}, name={}",
                q.params.source_id, q.params.dest_id, q.params.name
            ),
            Query::NodeFragmentsByIdTimeRange(q) => {
                write!(
                    f,
                    "NodeFragmentsByIdTimeRange: id={}, range={:?}",
                    q.params.id, q.params.time_range
                )
            }
            Query::EdgeFragmentsByIdTimeRange(q) => {
                write!(
                    f,
                    "EdgeFragmentsByIdTimeRange: source={}, dest={}, name={}, range={:?}",
                    q.params.source_id, q.params.dest_id, q.params.edge_name, q.params.time_range
                )
            }
            Query::OutgoingEdges(q) => write!(f, "OutgoingEdges: id={}", q.params.id),
            Query::IncomingEdges(q) => write!(f, "IncomingEdges: id={}", q.params.id),
            Query::AllNodes(q) => write!(f, "AllNodes: limit={}", q.params.limit),
            Query::AllEdges(q) => write!(f, "AllEdges: limit={}", q.params.limit),
        }
    }
}

// Use macro to implement QueryProcessor for dispatch types
crate::impl_query_processor!(
    NodeByIdDispatch,
    NodesByIdsMultiDispatch,
    EdgeSummaryBySrcDstNameDispatch,
    NodeFragmentsByIdTimeRangeDispatch,
    EdgeFragmentsByIdTimeRangeDispatch,
    OutgoingEdgesDispatch,
    IncomingEdgesDispatch,
    AllNodesDispatch,
    AllEdgesDispatch,
);

#[async_trait::async_trait]
impl QueryProcessor for Query {
    async fn process_and_send<P: Processor>(self, processor: &P) {
        match self {
            Query::NodeById(q) => q.process_and_send(processor).await,
            Query::NodesByIdsMulti(q) => q.process_and_send(processor).await,
            Query::EdgeSummaryBySrcDstName(q) => q.process_and_send(processor).await,
            Query::NodeFragmentsByIdTimeRange(q) => q.process_and_send(processor).await,
            Query::EdgeFragmentsByIdTimeRange(q) => q.process_and_send(processor).await,
            Query::OutgoingEdges(q) => q.process_and_send(processor).await,
            Query::IncomingEdges(q) => q.process_and_send(processor).await,
            Query::AllNodes(q) => q.process_and_send(processor).await,
            Query::AllEdges(q) => q.process_and_send(processor).await,
        }
    }
}

/// Query parameters for finding a node by its ID.
///
/// This struct contains only user-facing parameters and can be:
/// - Constructed via struct initialization
/// - Cloned and reused
#[derive(Debug, Clone, PartialEq)]
pub struct NodeById {
    /// The entity ID to search for
    pub id: Id,

    /// Reference timestamp for temporal validity checks
    /// If None, defaults to current time in the query executor
    pub reference_ts_millis: Option<TimestampMilli>,
}

/// Internal dispatch wrapper for NodeById query execution.
#[derive(Debug)]
pub(crate) struct NodeByIdDispatch {
    pub(crate) params: NodeById,
    pub(crate) timeout: Duration,
    pub(crate) result_tx: oneshot::Sender<Result<(NodeName, NodeSummary)>>,
}

/// Query parameters for batch lookup of multiple nodes by ID.
///
/// More efficient than multiple `NodeById` calls as it uses RocksDB's
/// `multi_get_cf()` for batched reads. Missing IDs are silently omitted
/// from results.
///
/// # Example
///
/// ```ignore
/// let nodes = NodesByIdsMulti::new(vec![id1, id2, id3], None)
///     .run(&reader, timeout)
///     .await?;
/// // Returns Vec<(Id, NodeName, NodeSummary)> for found nodes
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NodesByIdsMulti {
    /// Node IDs to look up
    pub ids: Vec<Id>,

    /// Reference timestamp for temporal validity checks
    /// If None, defaults to current time in the query executor
    pub reference_ts_millis: Option<TimestampMilli>,
}

/// Internal dispatch wrapper for NodesByIdsMulti query execution.
#[derive(Debug)]
pub(crate) struct NodesByIdsMultiDispatch {
    pub(crate) params: NodesByIdsMulti,
    pub(crate) timeout: Duration,
    pub(crate) result_tx: oneshot::Sender<Result<Vec<(Id, NodeName, NodeSummary)>>>,
}

/// Query parameters for scanning node fragments by ID with time range filtering.
///
/// This struct contains only user-facing parameters and can be:
/// - Constructed via struct initialization
/// - Cloned and reused
#[derive(Debug, Clone, PartialEq)]
pub struct NodeFragmentsByIdTimeRange {
    /// The entity ID to search for
    pub id: Id,

    /// Time range bounds (start_bound, end_bound)
    /// Use std::ops::Bound for idiomatic Rust range specification
    pub time_range: (Bound<TimestampMilli>, Bound<TimestampMilli>),

    /// Reference timestamp for temporal validity checks
    /// If None, defaults to current time in the query executor
    pub reference_ts_millis: Option<TimestampMilli>,
}

/// Internal dispatch wrapper for NodeFragmentsByIdTimeRange query execution.
#[derive(Debug)]
pub(crate) struct NodeFragmentsByIdTimeRangeDispatch {
    pub(crate) params: NodeFragmentsByIdTimeRange,
    pub(crate) timeout: Duration,
    pub(crate) result_tx: oneshot::Sender<Result<Vec<(TimestampMilli, FragmentContent)>>>,
}

/// Query parameters for scanning edge fragments by source ID, destination ID, edge name, and time range.
///
/// This struct contains only user-facing parameters and can be:
/// - Constructed via struct initialization
/// - Cloned and reused
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeFragmentsByIdTimeRange {
    /// Source node ID
    pub source_id: SrcId,

    /// Destination node ID
    pub dest_id: DstId,

    /// Edge name
    pub edge_name: EdgeName,

    /// Time range bounds (start_bound, end_bound)
    /// Use std::ops::Bound for idiomatic Rust range specification
    pub time_range: (Bound<TimestampMilli>, Bound<TimestampMilli>),

    /// Reference timestamp for temporal validity checks
    /// If None, defaults to current time in the query executor
    pub reference_ts_millis: Option<TimestampMilli>,
}

/// Internal dispatch wrapper for EdgeFragmentsByIdTimeRange query execution.
#[derive(Debug)]
pub(crate) struct EdgeFragmentsByIdTimeRangeDispatch {
    pub(crate) params: EdgeFragmentsByIdTimeRange,
    pub(crate) timeout: Duration,
    pub(crate) result_tx: oneshot::Sender<Result<Vec<(TimestampMilli, FragmentContent)>>>,
}

/// Query parameters for finding an edge by source ID, destination ID, and name.
///
/// This struct contains only user-facing parameters and can be:
/// - Constructed via struct initialization
/// - Cloned and reused
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeSummaryBySrcDstName {
    /// Source node ID
    pub source_id: Id,

    /// Destination node ID
    pub dest_id: Id,

    /// Edge name
    pub name: String,

    /// Reference timestamp for temporal validity checks
    /// If None, defaults to current time in the query executor
    pub reference_ts_millis: Option<TimestampMilli>,
}

/// Internal dispatch wrapper for EdgeSummaryBySrcDstName query execution.
#[derive(Debug)]
pub(crate) struct EdgeSummaryBySrcDstNameDispatch {
    pub(crate) params: EdgeSummaryBySrcDstName,
    pub(crate) timeout: Duration,
    pub(crate) result_tx: oneshot::Sender<Result<(EdgeSummary, Option<f64>)>>,
}

/// Query parameters for finding all outgoing edges from a node.
///
/// This struct contains only user-facing parameters and can be:
/// - Constructed via struct initialization with `..Default::default()`
/// - Deserialized from JSON or other formats
/// - Cloned and reused
///
/// # Examples
///
/// ```ignore
/// // Struct initialization
/// let query = OutgoingEdges {
///     id: node_id,
///     reference_ts_millis: Some(TimestampMilli::now()),
/// };
///
/// // Constructor
/// let query = OutgoingEdges::new(node_id, None);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct OutgoingEdges {
    /// The node ID to search for
    pub id: Id,

    /// Reference timestamp for temporal validity checks
    /// If None, defaults to current time in the query executor
    /// Temporal validity is always checked against the TemporalRange in the record
    /// Records without a TemporalRange (None) are considered always valid
    pub reference_ts_millis: Option<TimestampMilli>,
}

/// Internal dispatch wrapper for OutgoingEdges query execution.
/// Contains the query parameters plus channel/timeout for async dispatch.
#[derive(Debug)]
pub(crate) struct OutgoingEdgesDispatch {
    /// The query parameters
    pub(crate) params: OutgoingEdges,

    /// Timeout for this query execution
    pub(crate) timeout: Duration,

    /// Channel to send the result back to the client
    pub(crate) result_tx: oneshot::Sender<Result<Vec<(Option<f64>, SrcId, DstId, EdgeName)>>>,
}

/// Query parameters for finding all incoming edges to a node.
///
/// This struct contains only user-facing parameters and can be:
/// - Constructed via struct initialization with `..Default::default()`
/// - Deserialized from JSON or other formats
/// - Cloned and reused
#[derive(Debug, Clone, PartialEq)]
pub struct IncomingEdges {
    /// The node ID to search for
    pub id: DstId,

    /// Reference timestamp for temporal validity checks
    /// If None, defaults to current time in the query executor
    /// Temporal validity is always checked against the TemporalRange in the record
    /// Records without a TemporalRange (None) are considered always valid
    pub reference_ts_millis: Option<TimestampMilli>,
}

/// Internal dispatch wrapper for IncomingEdges query execution.
/// Contains the query parameters plus channel/timeout for async dispatch.
#[derive(Debug)]
pub(crate) struct IncomingEdgesDispatch {
    /// The query parameters
    pub(crate) params: IncomingEdges,

    /// Timeout for this query execution
    pub(crate) timeout: Duration,

    /// Channel to send the result back to the client
    pub(crate) result_tx: oneshot::Sender<Result<Vec<(Option<f64>, DstId, SrcId, EdgeName)>>>,
}

/// Query parameters for enumerating all nodes with pagination.
///
/// This is the query-based interface for graph enumeration, useful for
/// graph algorithms that need to iterate over all nodes (e.g., PageRank).
///
/// # Examples
///
/// ```ignore
/// // Get first page of nodes
/// let nodes = AllNodes::new(1000)
///     .run(&reader, timeout)
///     .await?;
///
/// // Get next page using cursor
/// if let Some((last_id, _, _)) = nodes.last() {
///     let next_page = AllNodes::new(1000)
///         .with_cursor(*last_id)
///         .run(&reader, timeout)
///         .await?;
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AllNodes {
    /// Cursor for pagination - last ID from previous page (exclusive)
    pub last: Option<Id>,

    /// Maximum number of records to return
    pub limit: usize,

    /// Reference timestamp for temporal validity checks
    /// If None, defaults to current time in the query executor
    pub reference_ts_millis: Option<TimestampMilli>,
}

/// Internal dispatch wrapper for AllNodes query execution.
#[derive(Debug)]
pub(crate) struct AllNodesDispatch {
    pub(crate) params: AllNodes,
    pub(crate) timeout: Duration,
    pub(crate) result_tx: oneshot::Sender<Result<Vec<(Id, NodeName, NodeSummary)>>>,
}

/// Query parameters for enumerating all edges with pagination.
///
/// This is the query-based interface for graph enumeration, useful for
/// graph algorithms that need to iterate over all edges (e.g., Kruskal's MST).
///
/// # Examples
///
/// ```ignore
/// // Get first page of edges
/// let edges = AllEdges::new(1000)
///     .run(&reader, timeout)
///     .await?;
///
/// // Get next page using cursor
/// if let Some((_, src, dst, name)) = edges.last() {
///     let next_page = AllEdges::new(1000)
///         .with_cursor((*src, *dst, name.clone()))
///         .run(&reader, timeout)
///         .await?;
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AllEdges {
    /// Cursor for pagination - (src_id, dst_id, edge_name) from previous page (exclusive)
    pub last: Option<(SrcId, DstId, EdgeName)>,

    /// Maximum number of records to return
    pub limit: usize,

    /// Reference timestamp for temporal validity checks
    /// If None, defaults to current time in the query executor
    pub reference_ts_millis: Option<TimestampMilli>,
}

/// Internal dispatch wrapper for AllEdges query execution.
#[derive(Debug)]
pub(crate) struct AllEdgesDispatch {
    pub(crate) params: AllEdges,
    pub(crate) timeout: Duration,
    pub(crate) result_tx: oneshot::Sender<Result<Vec<(Option<f64>, SrcId, DstId, EdgeName)>>>,
}

impl NodeById {
    /// Create a new query request.
    pub fn new(id: Id, reference_ts_millis: Option<TimestampMilli>) -> Self {
        Self {
            id,
            reference_ts_millis,
        }
    }
}

impl NodeByIdDispatch {
    /// Create a new dispatch wrapper.
    pub(crate) fn new(
        params: NodeById,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<(NodeName, NodeSummary)>>,
    ) -> Self {
        Self {
            params,
            timeout,
            result_tx,
        }
    }

    /// Send the result back to the client (consumes self).
    pub(crate) fn send_result(self, result: Result<(NodeName, NodeSummary)>) {
        let _ = self.result_tx.send(result);
    }

    /// Execute a NodeById query directly without dispatch machinery.
    /// This is used by the unified query module for composition.
    pub(crate) async fn execute_params(
        params: &NodeById,
        storage: &Storage,
    ) -> Result<(NodeName, NodeSummary)> {
        let (tx, _rx) = oneshot::channel();
        let dispatch = NodeByIdDispatch {
            params: params.clone(),
            timeout: Duration::from_secs(0),
            result_tx: tx,
        };
        <NodeByIdDispatch as super::reader::QueryExecutor>::execute(&dispatch, storage).await
    }
}

impl NodesByIdsMulti {
    /// Create a new batch query request.
    pub fn new(ids: Vec<Id>, reference_ts_millis: Option<TimestampMilli>) -> Self {
        Self {
            ids,
            reference_ts_millis,
        }
    }
}

impl NodesByIdsMultiDispatch {
    /// Create a new dispatch wrapper.
    pub(crate) fn new(
        params: NodesByIdsMulti,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<Vec<(Id, NodeName, NodeSummary)>>>,
    ) -> Self {
        Self {
            params,
            timeout,
            result_tx,
        }
    }

    /// Send the result back to the client (consumes self).
    pub(crate) fn send_result(self, result: Result<Vec<(Id, NodeName, NodeSummary)>>) {
        let _ = self.result_tx.send(result);
    }

    /// Execute a NodesByIdsMulti query directly without dispatch machinery.
    /// This is used by the unified query module for composition.
    pub(crate) async fn execute_params(
        params: &NodesByIdsMulti,
        storage: &Storage,
    ) -> Result<Vec<(Id, NodeName, NodeSummary)>> {
        let (tx, _rx) = oneshot::channel();
        let dispatch = NodesByIdsMultiDispatch {
            params: params.clone(),
            timeout: Duration::from_secs(0),
            result_tx: tx,
        };
        <NodesByIdsMultiDispatch as super::reader::QueryExecutor>::execute(&dispatch, storage).await
    }
}

impl NodeFragmentsByIdTimeRange {
    /// Create a new query request.
    pub fn new(
        id: Id,
        time_range: (Bound<TimestampMilli>, Bound<TimestampMilli>),
        reference_ts_millis: Option<TimestampMilli>,
    ) -> Self {
        Self {
            id,
            time_range,
            reference_ts_millis,
        }
    }

    /// Check if a timestamp falls within this range.
    pub fn contains(&self, ts: TimestampMilli) -> bool {
        timestamp_in_range(ts, &self.time_range)
    }
}

impl NodeFragmentsByIdTimeRangeDispatch {
    /// Create a new dispatch wrapper.
    pub(crate) fn new(
        params: NodeFragmentsByIdTimeRange,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<Vec<(TimestampMilli, FragmentContent)>>>,
    ) -> Self {
        Self {
            params,
            timeout,
            result_tx,
        }
    }

    /// Send the result back to the client (consumes self).
    pub(crate) fn send_result(self, result: Result<Vec<(TimestampMilli, FragmentContent)>>) {
        let _ = self.result_tx.send(result);
    }

    /// Execute a NodeFragmentsByIdTimeRange query directly without dispatch machinery.
    /// This is used by the unified query module for composition.
    pub(crate) async fn execute_params(
        params: &NodeFragmentsByIdTimeRange,
        storage: &Storage,
    ) -> Result<Vec<(TimestampMilli, FragmentContent)>> {
        let (tx, _rx) = oneshot::channel();
        let dispatch = NodeFragmentsByIdTimeRangeDispatch {
            params: params.clone(),
            timeout: Duration::from_secs(0),
            result_tx: tx,
        };
        <NodeFragmentsByIdTimeRangeDispatch as super::reader::QueryExecutor>::execute(
            &dispatch, storage,
        )
        .await
    }
}

impl EdgeFragmentsByIdTimeRange {
    /// Create a new query request.
    pub fn new(
        source_id: SrcId,
        dest_id: DstId,
        edge_name: EdgeName,
        time_range: (Bound<TimestampMilli>, Bound<TimestampMilli>),
        reference_ts_millis: Option<TimestampMilli>,
    ) -> Self {
        Self {
            source_id,
            dest_id,
            edge_name,
            time_range,
            reference_ts_millis,
        }
    }

    /// Check if a timestamp falls within this range.
    pub fn contains(&self, ts: TimestampMilli) -> bool {
        timestamp_in_range(ts, &self.time_range)
    }
}

impl EdgeFragmentsByIdTimeRangeDispatch {
    /// Create a new dispatch wrapper.
    pub(crate) fn new(
        params: EdgeFragmentsByIdTimeRange,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<Vec<(TimestampMilli, FragmentContent)>>>,
    ) -> Self {
        Self {
            params,
            timeout,
            result_tx,
        }
    }

    /// Send the result back to the client (consumes self).
    pub(crate) fn send_result(self, result: Result<Vec<(TimestampMilli, FragmentContent)>>) {
        let _ = self.result_tx.send(result);
    }

    /// Execute an EdgeFragmentsByIdTimeRange query directly without dispatch machinery.
    /// This is used by the unified query module for composition.
    pub(crate) async fn execute_params(
        params: &EdgeFragmentsByIdTimeRange,
        storage: &Storage,
    ) -> Result<Vec<(TimestampMilli, FragmentContent)>> {
        let (tx, _rx) = oneshot::channel();
        let dispatch = EdgeFragmentsByIdTimeRangeDispatch {
            params: params.clone(),
            timeout: Duration::from_secs(0),
            result_tx: tx,
        };
        <EdgeFragmentsByIdTimeRangeDispatch as super::reader::QueryExecutor>::execute(
            &dispatch, storage,
        )
        .await
    }
}

impl EdgeSummaryBySrcDstName {
    /// Create a new query request.
    pub fn new(
        source_id: SrcId,
        dest_id: DstId,
        name: String,
        reference_ts_millis: Option<TimestampMilli>,
    ) -> Self {
        Self {
            source_id,
            dest_id,
            name,
            reference_ts_millis,
        }
    }
}

impl EdgeSummaryBySrcDstNameDispatch {
    /// Create a new dispatch wrapper.
    pub(crate) fn new(
        params: EdgeSummaryBySrcDstName,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<(EdgeSummary, Option<f64>)>>,
    ) -> Self {
        Self {
            params,
            timeout,
            result_tx,
        }
    }

    /// Send the result back to the client (consumes self).
    pub(crate) fn send_result(self, result: Result<(EdgeSummary, Option<f64>)>) {
        let _ = self.result_tx.send(result);
    }

    /// Execute an EdgeSummaryBySrcDstName query directly without dispatch machinery.
    /// This is used by the unified query module for composition.
    pub(crate) async fn execute_params(
        params: &EdgeSummaryBySrcDstName,
        storage: &Storage,
    ) -> Result<(EdgeSummary, Option<f64>)> {
        let (tx, _rx) = oneshot::channel();
        let dispatch = EdgeSummaryBySrcDstNameDispatch {
            params: params.clone(),
            timeout: Duration::from_secs(0),
            result_tx: tx,
        };
        <EdgeSummaryBySrcDstNameDispatch as super::reader::QueryExecutor>::execute(
            &dispatch, storage,
        )
        .await
    }
}

impl OutgoingEdges {
    /// Create a new query request.
    pub fn new(id: Id, reference_ts_millis: Option<TimestampMilli>) -> Self {
        Self {
            id,
            reference_ts_millis,
        }
    }
}

impl OutgoingEdgesDispatch {
    /// Create a new dispatch wrapper
    pub(crate) fn new(
        params: OutgoingEdges,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<Vec<(Option<f64>, SrcId, DstId, EdgeName)>>>,
    ) -> Self {
        Self {
            params,
            timeout,
            result_tx,
        }
    }

    /// Send the result back to the client (consumes self)
    pub(crate) fn send_result(self, result: Result<Vec<(Option<f64>, SrcId, DstId, EdgeName)>>) {
        let _ = self.result_tx.send(result);
    }

    /// Execute an OutgoingEdges query directly without dispatch machinery.
    /// This is used by the unified query module for composition.
    pub(crate) async fn execute_params(
        params: &OutgoingEdges,
        storage: &Storage,
    ) -> Result<Vec<(Option<f64>, SrcId, DstId, EdgeName)>> {
        let (tx, _rx) = oneshot::channel();
        let dispatch = OutgoingEdgesDispatch {
            params: params.clone(),
            timeout: Duration::from_secs(0),
            result_tx: tx,
        };
        <OutgoingEdgesDispatch as super::reader::QueryExecutor>::execute(&dispatch, storage).await
    }
}

impl IncomingEdges {
    /// Create a new query request.
    pub fn new(id: DstId, reference_ts_millis: Option<TimestampMilli>) -> Self {
        Self {
            id,
            reference_ts_millis,
        }
    }
}

impl IncomingEdgesDispatch {
    /// Create a new dispatch wrapper
    pub(crate) fn new(
        params: IncomingEdges,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<Vec<(Option<f64>, DstId, SrcId, EdgeName)>>>,
    ) -> Self {
        Self {
            params,
            timeout,
            result_tx,
        }
    }

    /// Send the result back to the client (consumes self)
    pub(crate) fn send_result(self, result: Result<Vec<(Option<f64>, DstId, SrcId, EdgeName)>>) {
        let _ = self.result_tx.send(result);
    }

    /// Execute an IncomingEdges query directly without dispatch machinery.
    /// This is used by the unified query module for composition.
    pub(crate) async fn execute_params(
        params: &IncomingEdges,
        storage: &Storage,
    ) -> Result<Vec<(Option<f64>, DstId, SrcId, EdgeName)>> {
        let (tx, _rx) = oneshot::channel();
        let dispatch = IncomingEdgesDispatch {
            params: params.clone(),
            timeout: Duration::from_secs(0),
            result_tx: tx,
        };
        <IncomingEdgesDispatch as super::reader::QueryExecutor>::execute(&dispatch, storage).await
    }
}

impl AllNodes {
    /// Create a new AllNodes query with the specified limit.
    pub fn new(limit: usize) -> Self {
        Self {
            last: None,
            limit,
            reference_ts_millis: None,
        }
    }

    /// Set the cursor for pagination (exclusive start).
    pub fn with_cursor(mut self, last: Id) -> Self {
        self.last = Some(last);
        self
    }

    /// Set the reference timestamp for temporal validity checks.
    pub fn with_reference_time(mut self, ts: TimestampMilli) -> Self {
        self.reference_ts_millis = Some(ts);
        self
    }
}

impl AllNodesDispatch {
    /// Create a new dispatch wrapper.
    pub(crate) fn new(
        params: AllNodes,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<Vec<(Id, NodeName, NodeSummary)>>>,
    ) -> Self {
        Self {
            params,
            timeout,
            result_tx,
        }
    }

    /// Send the result back to the client (consumes self).
    pub(crate) fn send_result(self, result: Result<Vec<(Id, NodeName, NodeSummary)>>) {
        let _ = self.result_tx.send(result);
    }

    /// Execute an AllNodes query directly without dispatch machinery.
    /// This is used by the unified query module for composition.
    pub(crate) async fn execute_params(
        params: &AllNodes,
        storage: &Storage,
    ) -> Result<Vec<(Id, NodeName, NodeSummary)>> {
        let (tx, _rx) = oneshot::channel();
        let dispatch = AllNodesDispatch {
            params: params.clone(),
            timeout: Duration::from_secs(0),
            result_tx: tx,
        };
        <AllNodesDispatch as super::reader::QueryExecutor>::execute(&dispatch, storage).await
    }
}

impl AllEdges {
    /// Create a new AllEdges query with the specified limit.
    pub fn new(limit: usize) -> Self {
        Self {
            last: None,
            limit,
            reference_ts_millis: None,
        }
    }

    /// Set the cursor for pagination (exclusive start).
    pub fn with_cursor(mut self, last: (SrcId, DstId, EdgeName)) -> Self {
        self.last = Some(last);
        self
    }

    /// Set the reference timestamp for temporal validity checks.
    pub fn with_reference_time(mut self, ts: TimestampMilli) -> Self {
        self.reference_ts_millis = Some(ts);
        self
    }
}

impl AllEdgesDispatch {
    /// Create a new dispatch wrapper.
    pub(crate) fn new(
        params: AllEdges,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<Vec<(Option<f64>, SrcId, DstId, EdgeName)>>>,
    ) -> Self {
        Self {
            params,
            timeout,
            result_tx,
        }
    }

    /// Send the result back to the client (consumes self).
    pub(crate) fn send_result(self, result: Result<Vec<(Option<f64>, SrcId, DstId, EdgeName)>>) {
        let _ = self.result_tx.send(result);
    }

    /// Execute an AllEdges query directly without dispatch machinery.
    /// This is used by the unified query module for composition.
    pub(crate) async fn execute_params(
        params: &AllEdges,
        storage: &Storage,
    ) -> Result<Vec<(Option<f64>, SrcId, DstId, EdgeName)>> {
        let (tx, _rx) = oneshot::channel();
        let dispatch = AllEdgesDispatch {
            params: params.clone(),
            timeout: Duration::from_secs(0),
            result_tx: tx,
        };
        <AllEdgesDispatch as super::reader::QueryExecutor>::execute(&dispatch, storage).await
    }
}

/// Implement Runnable<Reader> for NodeById
#[async_trait::async_trait]
impl Runnable<super::Reader> for NodeById {
    type Output = (NodeName, NodeSummary);

    async fn run(self, reader: &super::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let dispatch = NodeByIdDispatch::new(self, timeout, result_tx);
        reader.send_query(Query::NodeById(dispatch)).await?;
        result_rx.await?
    }
}

/// Implement Runnable<Reader> for NodesByIdsMulti
#[async_trait::async_trait]
impl Runnable<super::Reader> for NodesByIdsMulti {
    type Output = Vec<(Id, NodeName, NodeSummary)>;

    async fn run(self, reader: &super::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let dispatch = NodesByIdsMultiDispatch::new(self, timeout, result_tx);
        reader.send_query(Query::NodesByIdsMulti(dispatch)).await?;
        result_rx.await?
    }
}

/// Implement Runnable<Reader> for NodeFragmentsByIdTimeRange
#[async_trait::async_trait]
impl Runnable<super::Reader> for NodeFragmentsByIdTimeRange {
    type Output = Vec<(TimestampMilli, FragmentContent)>;

    async fn run(self, reader: &super::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let dispatch = NodeFragmentsByIdTimeRangeDispatch::new(self, timeout, result_tx);
        reader
            .send_query(Query::NodeFragmentsByIdTimeRange(dispatch))
            .await?;
        result_rx.await?
    }
}

/// Implement Runnable<Reader> for EdgeFragmentsByIdTimeRange
#[async_trait::async_trait]
impl Runnable<super::Reader> for EdgeFragmentsByIdTimeRange {
    type Output = Vec<(TimestampMilli, FragmentContent)>;

    async fn run(self, reader: &super::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let dispatch = EdgeFragmentsByIdTimeRangeDispatch::new(self, timeout, result_tx);
        reader
            .send_query(Query::EdgeFragmentsByIdTimeRange(dispatch))
            .await?;
        result_rx.await?
    }
}

/// Implement Runnable<Reader> for EdgeSummaryBySrcDstName
#[async_trait::async_trait]
impl Runnable<super::Reader> for EdgeSummaryBySrcDstName {
    type Output = (EdgeSummary, Option<f64>);

    async fn run(self, reader: &super::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let dispatch = EdgeSummaryBySrcDstNameDispatch::new(self, timeout, result_tx);
        reader
            .send_query(Query::EdgeSummaryBySrcDstName(dispatch))
            .await?;
        result_rx.await?
    }
}

#[async_trait::async_trait]
impl Runnable<super::Reader> for OutgoingEdges {
    type Output = Vec<(Option<f64>, SrcId, DstId, EdgeName)>;

    async fn run(self, reader: &super::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let dispatch = OutgoingEdgesDispatch::new(self, timeout, result_tx);
        reader.send_query(Query::OutgoingEdges(dispatch)).await?;
        result_rx.await?
    }
}

#[async_trait::async_trait]
impl Runnable<super::Reader> for IncomingEdges {
    type Output = Vec<(Option<f64>, DstId, SrcId, EdgeName)>;

    async fn run(self, reader: &super::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let dispatch = IncomingEdgesDispatch::new(self, timeout, result_tx);
        reader.send_query(Query::IncomingEdges(dispatch)).await?;
        result_rx.await?
    }
}

#[async_trait::async_trait]
impl Runnable<super::Reader> for AllNodes {
    type Output = Vec<(Id, NodeName, NodeSummary)>;

    async fn run(self, reader: &super::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let dispatch = AllNodesDispatch::new(self, timeout, result_tx);
        reader.send_query(Query::AllNodes(dispatch)).await?;
        result_rx.await?
    }
}

#[async_trait::async_trait]
impl Runnable<super::Reader> for AllEdges {
    type Output = Vec<(Option<f64>, SrcId, DstId, EdgeName)>;

    async fn run(self, reader: &super::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let dispatch = AllEdgesDispatch::new(self, timeout, result_tx);
        reader.send_query(Query::AllEdges(dispatch)).await?;
        result_rx.await?
    }
}

/// Implement QueryExecutor for NodeByIdDispatch
#[async_trait::async_trait]
impl QueryExecutor for NodeByIdDispatch {
    type Output = (NodeName, NodeSummary);

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let params = &self.params;
        tracing::debug!(id = %params.id, "Executing NodeById query");

        // Default None to current time for temporal validity checks
        let ref_time = params
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let id = params.id;

        let key = schema::NodeCfKey(id);
        let key_bytes = schema::Nodes::key_to_bytes(&key);

        // Handle both readonly and readwrite modes
        let value_bytes = if let Ok(db) = storage.db() {
            let cf = db.cf_handle(schema::Nodes::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!("Column family '{}' not found", schema::Nodes::CF_NAME)
            })?;
            db.get_cf(cf, key_bytes)?
        } else {
            let txn_db = storage.transaction_db()?;
            let cf = txn_db.cf_handle(schema::Nodes::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!("Column family '{}' not found", schema::Nodes::CF_NAME)
            })?;
            txn_db.get_cf(cf, key_bytes)?
        };

        let value_bytes = value_bytes.ok_or_else(|| anyhow::anyhow!("Node not found: {}", id))?;

        let value: schema::NodeCfValue = schema::Nodes::value_from_bytes(&value_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

        // Always check temporal validity
        if !schema::is_valid_at_time(&value.0, ref_time) {
            return Err(anyhow::anyhow!(
                "Node {} not valid at time {}",
                id,
                ref_time.0
            ));
        }

        // Resolve NameHash to String (uses cache)
        let node_name = resolve_name(storage, value.1)?;

        Ok((node_name, value.2))
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

/// Implement QueryExecutor for NodesByIdsMultiDispatch
///
/// Uses RocksDB's `multi_get_cf()` for efficient batch lookups.
/// Missing nodes and temporally invalid nodes are silently omitted from results.
#[async_trait::async_trait]
impl QueryExecutor for NodesByIdsMultiDispatch {
    type Output = Vec<(Id, NodeName, NodeSummary)>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let params = &self.params;
        tracing::debug!(count = params.ids.len(), "Executing NodesByIdsMulti query");

        if params.ids.is_empty() {
            return Ok(Vec::new());
        }

        // Default None to current time for temporal validity checks
        let ref_time = params
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        // Prepare keys for batch lookup
        let keys: Vec<Vec<u8>> = params
            .ids
            .iter()
            .map(|id| schema::Nodes::key_to_bytes(&schema::NodeCfKey(*id)))
            .collect();

        // Use multi_get_cf for batch lookup - handles both readonly and readwrite modes
        let results: Vec<Result<Option<Vec<u8>>, rocksdb::Error>> = if let Ok(db) = storage.db() {
            let cf = db.cf_handle(schema::Nodes::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!("Column family '{}' not found", schema::Nodes::CF_NAME)
            })?;
            db.multi_get_cf(keys.iter().map(|k| (&cf, k.as_slice())))
        } else {
            let txn_db = storage.transaction_db()?;
            let cf = txn_db.cf_handle(schema::Nodes::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!("Column family '{}' not found", schema::Nodes::CF_NAME)
            })?;
            txn_db.multi_get_cf(keys.iter().map(|k| (&cf, k.as_slice())))
        };

        // Parse results, skipping missing entries and temporally invalid nodes
        // Collect valid entries with their NameHash first, then resolve names
        let mut valid_entries: Vec<(Id, NameHash, NodeSummary)> = Vec::with_capacity(results.len());
        for (id, result) in params.ids.iter().zip(results) {
            match result {
                Ok(Some(value_bytes)) => {
                    match schema::Nodes::value_from_bytes(&value_bytes) {
                        Ok(value) => {
                            // Check temporal validity - skip invalid nodes
                            if schema::is_valid_at_time(&value.0, ref_time) {
                                valid_entries.push((*id, value.1, value.2));
                            } else {
                                tracing::trace!(
                                    id = %id,
                                    "Skipping node: not valid at time {}",
                                    ref_time.0
                                );
                            }
                        }
                        Err(e) => {
                            tracing::warn!(id = %id, error = %e, "Failed to deserialize node value");
                        }
                    }
                }
                Ok(None) => {
                    // Node not found - silently skip
                    tracing::trace!(id = %id, "Node not found");
                }
                Err(e) => {
                    tracing::warn!(id = %id, error = %e, "RocksDB error fetching node");
                }
            }
        }

        // Resolve NameHashes to Strings (uses cache)
        let mut output = Vec::with_capacity(valid_entries.len());
        for (id, name_hash, summary) in valid_entries {
            let node_name = resolve_name(storage, name_hash)?;
            output.push((id, node_name, summary));
        }

        Ok(output)
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

/// Implement QueryExecutor for NodeFragmentsByIdTimeRangeDispatch
#[async_trait::async_trait]
impl QueryExecutor for NodeFragmentsByIdTimeRangeDispatch {
    type Output = Vec<(TimestampMilli, FragmentContent)>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let params = &self.params;
        tracing::debug!(id = %params.id, time_range = ?params.time_range, "Executing NodeFragmentsByIdTimeRange query");

        use std::ops::Bound;

        // Default None to current time for temporal validity checks
        let ref_time = params
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let id = params.id;
        let mut fragments: Vec<(TimestampMilli, FragmentContent)> = Vec::new();

        // Construct optimal starting key based on start bound
        let start_key = match &params.time_range.0 {
            Bound::Unbounded => {
                let mut key = Vec::with_capacity(24);
                key.extend_from_slice(&id.into_bytes());
                key.extend_from_slice(&0u64.to_be_bytes());
                key
            }
            Bound::Included(start_ts) => {
                schema::NodeFragments::key_to_bytes(&schema::NodeFragmentCfKey(id, *start_ts))
            }
            Bound::Excluded(start_ts) => schema::NodeFragments::key_to_bytes(
                &schema::NodeFragmentCfKey(id, TimestampMilli(start_ts.0 + 1)),
            ),
        };

        iterate_cf!(storage, schema::NodeFragments, start_key, |item| {
            let (key_bytes, value_bytes) = item?;
            let key: schema::NodeFragmentCfKey = schema::NodeFragments::key_from_bytes(&key_bytes)
                .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

            if key.0 != id {
                break;
            }

            let timestamp = key.1;

            match &params.time_range.1 {
                Bound::Unbounded => { /* continue scanning */ }
                Bound::Included(end_ts) => {
                    if timestamp.0 > end_ts.0 {
                        break;
                    }
                }
                Bound::Excluded(end_ts) => {
                    if timestamp.0 >= end_ts.0 {
                        break;
                    }
                }
            }

            let value: schema::NodeFragmentCfValue =
                schema::NodeFragments::value_from_bytes(&value_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

            // Always check temporal validity - skip invalid fragments
            if !schema::is_valid_at_time(&value.0, ref_time) {
                continue;
            }

            fragments.push((timestamp, value.1));
        });

        Ok(fragments)
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

/// Implement QueryExecutor for EdgeFragmentsByIdTimeRangeDispatch
#[async_trait::async_trait]
impl QueryExecutor for EdgeFragmentsByIdTimeRangeDispatch {
    type Output = Vec<(TimestampMilli, FragmentContent)>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let params = &self.params;
        tracing::debug!(
            src_id = %params.source_id,
            dst_id = %params.dest_id,
            edge_name = %params.edge_name,
            time_range = ?params.time_range,
            "Executing EdgeFragmentsByIdTimeRange query"
        );

        use std::ops::Bound;

        // Default None to current time for temporal validity checks
        let ref_time = params
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let source_id = params.source_id;
        let dest_id = params.dest_id;
        // Convert edge_name String to NameHash for key construction
        let edge_name_hash = NameHash::from_name(&params.edge_name);
        let mut fragments: Vec<(TimestampMilli, FragmentContent)> = Vec::new();

        // Construct optimal starting key based on start bound
        // EdgeFragmentCfKey: (SrcId, DstId, NameHash, TimestampMilli) - now fixed 40 bytes
        let start_key = match &params.time_range.0 {
            Bound::Unbounded => {
                schema::EdgeFragments::key_to_bytes(
                    &schema::EdgeFragmentCfKey(source_id, dest_id, edge_name_hash, TimestampMilli(0)),
                )
            }
            Bound::Included(start_ts) => schema::EdgeFragments::key_to_bytes(
                &schema::EdgeFragmentCfKey(source_id, dest_id, edge_name_hash, *start_ts),
            ),
            Bound::Excluded(start_ts) => {
                schema::EdgeFragments::key_to_bytes(&schema::EdgeFragmentCfKey(
                    source_id,
                    dest_id,
                    edge_name_hash,
                    TimestampMilli(start_ts.0 + 1),
                ))
            }
        };

        iterate_cf!(storage, schema::EdgeFragments, start_key, |item| {
            let (key_bytes, value_bytes) = item?;
            let key: schema::EdgeFragmentCfKey = schema::EdgeFragments::key_from_bytes(&key_bytes)
                .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

            // Check if we're still in the same edge (source_id, dest_id, edge_name_hash)
            if key.0 != source_id || key.1 != dest_id || key.2 != edge_name_hash {
                break;
            }

            let timestamp = key.3;

            match &params.time_range.1 {
                Bound::Unbounded => { /* continue scanning */ }
                Bound::Included(end_ts) => {
                    if timestamp.0 > end_ts.0 {
                        break;
                    }
                }
                Bound::Excluded(end_ts) => {
                    if timestamp.0 >= end_ts.0 {
                        break;
                    }
                }
            }

            let value: schema::EdgeFragmentCfValue =
                schema::EdgeFragments::value_from_bytes(&value_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

            // Always check temporal validity - skip invalid fragments
            if !schema::is_valid_at_time(&value.0, ref_time) {
                continue;
            }

            fragments.push((timestamp, value.1));
        });

        Ok(fragments)
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

/// Implement QueryExecutor for EdgeSummaryBySrcDstNameDispatch
#[async_trait::async_trait]
impl QueryExecutor for EdgeSummaryBySrcDstNameDispatch {
    type Output = (EdgeSummary, Option<f64>);

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let params = &self.params;
        tracing::debug!(
            src_id = %params.source_id,
            dst_id = %params.dest_id,
            name = %params.name,
            "Executing EdgeSummaryBySrcDstName query"
        );

        // Default None to current time for temporal validity checks
        let ref_time = params
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let source_id = params.source_id;
        let dest_id = params.dest_id;
        let name = &params.name;
        // Convert edge name to NameHash for key construction
        let name_hash = NameHash::from_name(name);

        let key = schema::ForwardEdgeCfKey(source_id, dest_id, name_hash);
        let key_bytes = schema::ForwardEdges::key_to_bytes(&key);

        let value_bytes = if let Ok(db) = storage.db() {
            let cf = db.cf_handle(schema::ForwardEdges::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!(
                    "Column family '{}' not found",
                    schema::ForwardEdges::CF_NAME
                )
            })?;
            db.get_cf(cf, key_bytes)?
        } else {
            let txn_db = storage.transaction_db()?;
            let cf = txn_db
                .cf_handle(schema::ForwardEdges::CF_NAME)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Column family '{}' not found",
                        schema::ForwardEdges::CF_NAME
                    )
                })?;
            txn_db.get_cf(cf, key_bytes)?
        };

        let value_bytes = value_bytes.ok_or_else(|| {
            anyhow::anyhow!(
                "Edge not found: source={}, dest={}, name={}",
                source_id,
                dest_id,
                name
            )
        })?;

        let value: schema::ForwardEdgeCfValue =
            schema::ForwardEdges::value_from_bytes(&value_bytes)
                .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

        // Always check temporal validity
        if !schema::is_valid_at_time(&value.0, ref_time) {
            return Err(anyhow::anyhow!(
                "Edge not valid at time {}: source={}, dest={}, name={}",
                ref_time.0,
                source_id,
                dest_id,
                name
            ));
        }

        // ForwardEdgeCfValue is (temporal_range, weight, summary)
        // Return (summary, weight)
        Ok((value.2.clone(), value.1))
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

/// Implement QueryExecutor for OutgoingEdgesDispatch
#[async_trait::async_trait]
impl QueryExecutor for OutgoingEdgesDispatch {
    type Output = Vec<(Option<f64>, SrcId, DstId, EdgeName)>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let params = &self.params;
        tracing::debug!(node_id = %params.id, "Executing OutgoingEdges query");

        // Default None to current time for temporal validity checks
        let ref_time = params
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let id = params.id;
        // Collect edges with NameHash first, then resolve to String
        let mut edges_with_hash: Vec<(Option<f64>, SrcId, DstId, NameHash)> = Vec::new();
        let prefix = id.into_bytes();

        if let Ok(db) = storage.db() {
            let cf = db.cf_handle(schema::ForwardEdges::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!(
                    "Column family '{}' not found",
                    schema::ForwardEdges::CF_NAME
                )
            })?;

            let iter = db.iterator_cf(
                cf,
                rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
            );

            for item in iter {
                let (key_bytes, value_bytes) = item?;
                let key: schema::ForwardEdgeCfKey =
                    schema::ForwardEdges::key_from_bytes(&key_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

                let source_id = key.0;
                if source_id != id {
                    break;
                }

                // Deserialize and check temporal validity
                let value: schema::ForwardEdgeCfValue =
                    schema::ForwardEdges::value_from_bytes(&value_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

                // Always check temporal validity - skip invalid edges
                if !schema::is_valid_at_time(&value.0, ref_time) {
                    continue;
                }

                let dest_id = key.1;
                let edge_name_hash = key.2;
                let weight = value.1;
                edges_with_hash.push((weight, source_id, dest_id, edge_name_hash));
            }
        } else {
            let txn_db = storage.transaction_db()?;
            let cf = txn_db
                .cf_handle(schema::ForwardEdges::CF_NAME)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Column family '{}' not found",
                        schema::ForwardEdges::CF_NAME
                    )
                })?;

            let iter = txn_db.iterator_cf(
                cf,
                rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
            );

            for item in iter {
                let (key_bytes, value_bytes) = item?;
                let key: schema::ForwardEdgeCfKey =
                    schema::ForwardEdges::key_from_bytes(&key_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

                let source_id = key.0;
                if source_id != id {
                    break;
                }

                // Deserialize and check temporal validity
                let value: schema::ForwardEdgeCfValue =
                    schema::ForwardEdges::value_from_bytes(&value_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

                // Always check temporal validity - skip invalid edges
                if !schema::is_valid_at_time(&value.0, ref_time) {
                    continue;
                }

                let dest_id = key.1;
                let edge_name_hash = key.2;
                let weight = value.1;
                edges_with_hash.push((weight, source_id, dest_id, edge_name_hash));
            }
        }

        // Resolve NameHashes to Strings (uses cache)
        let mut edges = Vec::with_capacity(edges_with_hash.len());
        for (weight, src_id, dst_id, name_hash) in edges_with_hash {
            let edge_name = resolve_name(storage, name_hash)?;
            edges.push((weight, src_id, dst_id, edge_name));
        }
        Ok(edges)
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

/// Implement QueryExecutor for IncomingEdgesDispatch
#[async_trait::async_trait]
impl QueryExecutor for IncomingEdgesDispatch {
    type Output = Vec<(Option<f64>, DstId, SrcId, EdgeName)>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let params = &self.params;
        tracing::debug!(node_id = %params.id, "Executing IncomingEdges query");

        // Default None to current time for temporal validity checks
        let ref_time = params
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let id = params.id;
        // Collect edges with NameHash, then resolve to String
        let mut edges_with_hash: Vec<(Option<f64>, DstId, SrcId, NameHash)> = Vec::new();
        let prefix = id.into_bytes();

        if let Ok(db) = storage.db() {
            let reverse_cf = db.cf_handle(schema::ReverseEdges::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!(
                    "Column family '{}' not found",
                    schema::ReverseEdges::CF_NAME
                )
            })?;

            let forward_cf = db.cf_handle(schema::ForwardEdges::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!(
                    "Column family '{}' not found",
                    schema::ForwardEdges::CF_NAME
                )
            })?;

            let iter = db.iterator_cf(
                reverse_cf,
                rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
            );

            for item in iter {
                let (key_bytes, value_bytes) = item?;
                let key: schema::ReverseEdgeCfKey =
                    schema::ReverseEdges::key_from_bytes(&key_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

                let dest_id = key.0;
                if dest_id != id {
                    break;
                }

                // Deserialize and check temporal validity
                let value: schema::ReverseEdgeCfValue =
                    schema::ReverseEdges::value_from_bytes(&value_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

                // Always check temporal validity - skip invalid edges
                if !schema::is_valid_at_time(&value.0, ref_time) {
                    continue;
                }

                let source_id = key.1;
                let edge_name_hash = key.2;

                // Lookup weight from ForwardEdges CF
                // ReverseEdgeCfKey is (dst_id, src_id, NameHash)
                // ForwardEdgeCfKey is (src_id, dst_id, NameHash)
                let forward_key = schema::ForwardEdgeCfKey(source_id, dest_id, edge_name_hash);
                let forward_key_bytes = schema::ForwardEdges::key_to_bytes(&forward_key);

                let weight = if let Some(forward_value_bytes) =
                    db.get_cf(forward_cf, forward_key_bytes)?
                {
                    let forward_value: schema::ForwardEdgeCfValue =
                        schema::ForwardEdges::value_from_bytes(&forward_value_bytes).map_err(
                            |e| anyhow::anyhow!("Failed to deserialize forward edge value: {}", e),
                        )?;
                    forward_value.1 // Extract weight from field 1
                } else {
                    None
                };

                edges_with_hash.push((weight, dest_id, source_id, edge_name_hash));
            }
        } else {
            let txn_db = storage.transaction_db()?;
            let reverse_cf = txn_db
                .cf_handle(schema::ReverseEdges::CF_NAME)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Column family '{}' not found",
                        schema::ReverseEdges::CF_NAME
                    )
                })?;

            let forward_cf = txn_db
                .cf_handle(schema::ForwardEdges::CF_NAME)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Column family '{}' not found",
                        schema::ForwardEdges::CF_NAME
                    )
                })?;

            let iter = txn_db.iterator_cf(
                reverse_cf,
                rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
            );

            for item in iter {
                let (key_bytes, value_bytes) = item?;
                let key: schema::ReverseEdgeCfKey =
                    schema::ReverseEdges::key_from_bytes(&key_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

                let dest_id = key.0;
                if dest_id != id {
                    break;
                }

                // Deserialize and check temporal validity
                let value: schema::ReverseEdgeCfValue =
                    schema::ReverseEdges::value_from_bytes(&value_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

                // Always check temporal validity - skip invalid edges
                if !schema::is_valid_at_time(&value.0, ref_time) {
                    continue;
                }

                let source_id = key.1;
                let edge_name_hash = key.2;

                // Lookup weight from ForwardEdges CF
                // ReverseEdgeCfKey is (dst_id, src_id, NameHash)
                // ForwardEdgeCfKey is (src_id, dst_id, NameHash)
                let forward_key = schema::ForwardEdgeCfKey(source_id, dest_id, edge_name_hash);
                let forward_key_bytes = schema::ForwardEdges::key_to_bytes(&forward_key);

                let weight = if let Some(forward_value_bytes) =
                    txn_db.get_cf(forward_cf, forward_key_bytes)?
                {
                    let forward_value: schema::ForwardEdgeCfValue =
                        schema::ForwardEdges::value_from_bytes(&forward_value_bytes).map_err(
                            |e| anyhow::anyhow!("Failed to deserialize forward edge value: {}", e),
                        )?;
                    forward_value.1 // Extract weight from field 1
                } else {
                    None
                };

                edges_with_hash.push((weight, dest_id, source_id, edge_name_hash));
            }
        }

        // Resolve NameHashes to Strings (uses cache)
        let mut edges = Vec::with_capacity(edges_with_hash.len());
        for (weight, dst_id, src_id, name_hash) in edges_with_hash {
            let edge_name = resolve_name(storage, name_hash)?;
            edges.push((weight, dst_id, src_id, edge_name));
        }
        Ok(edges)
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

/// Implement QueryExecutor for AllNodesDispatch
/// Delegates to the scan module's Visitable implementation.
#[async_trait::async_trait]
impl QueryExecutor for AllNodesDispatch {
    type Output = Vec<(Id, NodeName, NodeSummary)>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let params = &self.params;
        tracing::debug!(limit = params.limit, has_cursor = params.last.is_some(), "Executing AllNodes query");

        // Create scan request matching scan::AllNodes
        let scan_request = scan::AllNodes {
            last: params.last,
            limit: params.limit,
            reverse: false,
            reference_ts_millis: params.reference_ts_millis,
        };

        // Collect results via visitor pattern
        let mut results = Vec::with_capacity(params.limit);
        scan_request.accept(storage, &mut |record: &scan::NodeRecord| {
            results.push((record.id, record.name.clone(), record.summary.clone()));
            true // continue
        })?;

        Ok(results)
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

/// Implement QueryExecutor for AllEdgesDispatch
/// Delegates to the scan module's Visitable implementation.
#[async_trait::async_trait]
impl QueryExecutor for AllEdgesDispatch {
    type Output = Vec<(Option<f64>, SrcId, DstId, EdgeName)>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let params = &self.params;
        tracing::debug!(limit = params.limit, has_cursor = params.last.is_some(), "Executing AllEdges query");

        // Create scan request matching scan::AllEdges
        let scan_request = scan::AllEdges {
            last: params.last.clone(),
            limit: params.limit,
            reverse: false,
            reference_ts_millis: params.reference_ts_millis,
        };

        // Collect results via visitor pattern
        let mut results = Vec::with_capacity(params.limit);
        scan_request.accept(storage, &mut |record: &scan::EdgeRecord| {
            results.push((record.weight, record.src_id, record.dst_id, record.name.clone()));
            true // continue
        })?;

        Ok(results)
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

// ============================================================================
// TransactionQueryExecutor Implementations
// ============================================================================

/// Implement TransactionQueryExecutor for NodeById
impl TransactionQueryExecutor for NodeById {
    type Output = (NodeName, NodeSummary);

    fn execute_in_transaction(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        cache: &super::name_hash::NameCache,
    ) -> Result<Self::Output> {
        tracing::debug!(id = %self.id, "Executing NodeById query in transaction");

        // Default None to current time for temporal validity checks
        let ref_time = self
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let key = schema::NodeCfKey(self.id);
        let key_bytes = schema::Nodes::key_to_bytes(&key);

        let cf = txn_db.cf_handle(schema::Nodes::CF_NAME).ok_or_else(|| {
            anyhow::anyhow!("Column family '{}' not found", schema::Nodes::CF_NAME)
        })?;

        // Use txn.get_cf to see uncommitted writes
        let value_bytes = txn
            .get_cf(cf, &key_bytes)?
            .ok_or_else(|| anyhow::anyhow!("Node not found: {}", self.id))?;

        let value: schema::NodeCfValue = schema::Nodes::value_from_bytes(&value_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

        // Check temporal validity
        if !schema::is_valid_at_time(&value.0, ref_time) {
            return Err(anyhow::anyhow!(
                "Node {} not valid at time {}",
                self.id,
                ref_time.0
            ));
        }

        // Resolve NameHash to String (uses cache)
        let node_name = resolve_name_from_txn(txn, txn_db, value.1, cache)?;
        Ok((node_name, value.2))
    }
}

/// Implement TransactionQueryExecutor for NodesByIdsMulti
impl TransactionQueryExecutor for NodesByIdsMulti {
    type Output = Vec<(Id, NodeName, NodeSummary)>;

    fn execute_in_transaction(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        cache: &super::name_hash::NameCache,
    ) -> Result<Self::Output> {
        tracing::debug!(count = self.ids.len(), "Executing NodesByIdsMulti query in transaction");

        if self.ids.is_empty() {
            return Ok(Vec::new());
        }

        let ref_time = self
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let cf = txn_db.cf_handle(schema::Nodes::CF_NAME).ok_or_else(|| {
            anyhow::anyhow!("Column family '{}' not found", schema::Nodes::CF_NAME)
        })?;

        // Collect entries with NameHash first
        let mut entries_with_hash: Vec<(Id, NameHash, NodeSummary)> = Vec::with_capacity(self.ids.len());

        for id in &self.ids {
            let key = schema::NodeCfKey(*id);
            let key_bytes = schema::Nodes::key_to_bytes(&key);

            // Use txn.get_cf for each key (transaction doesn't have multi_get)
            if let Some(value_bytes) = txn.get_cf(cf, &key_bytes)? {
                match schema::Nodes::value_from_bytes(&value_bytes) {
                    Ok(value) => {
                        if schema::is_valid_at_time(&value.0, ref_time) {
                            entries_with_hash.push((*id, value.1, value.2));
                        }
                    }
                    Err(e) => {
                        tracing::warn!(id = %id, error = %e, "Failed to deserialize node value");
                    }
                }
            }
        }

        // Resolve NameHashes to Strings (uses cache)
        let mut output = Vec::with_capacity(entries_with_hash.len());
        for (id, name_hash, summary) in entries_with_hash {
            let node_name = resolve_name_from_txn(txn, txn_db, name_hash, cache)?;
            output.push((id, node_name, summary));
        }

        Ok(output)
    }
}

/// Implement TransactionQueryExecutor for NodeFragmentsByIdTimeRange
impl TransactionQueryExecutor for NodeFragmentsByIdTimeRange {
    type Output = Vec<(TimestampMilli, FragmentContent)>;

    fn execute_in_transaction(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        _cache: &super::name_hash::NameCache,
    ) -> Result<Self::Output> {
        tracing::debug!(id = %self.id, time_range = ?self.time_range, "Executing NodeFragmentsByIdTimeRange query in transaction");

        let ref_time = self
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let cf = txn_db
            .cf_handle(schema::NodeFragments::CF_NAME)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Column family '{}' not found",
                    schema::NodeFragments::CF_NAME
                )
            })?;

        // Create prefix for this node's fragments
        let prefix_key = schema::NodeFragmentCfKey(self.id, TimestampMilli(0));
        let prefix_bytes = schema::NodeFragments::key_to_bytes(&prefix_key);
        let prefix_len = std::mem::size_of::<Id>();

        let mut results = Vec::new();

        // Use transaction iterator
        let iter = txn.iterator_cf(
            cf,
            rocksdb::IteratorMode::From(&prefix_bytes, rocksdb::Direction::Forward),
        );

        for item in iter {
            let (key_bytes, value_bytes) = item?;

            // Check prefix match
            if key_bytes.len() < prefix_len || &key_bytes[..prefix_len] != &prefix_bytes[..prefix_len]
            {
                break;
            }

            let key: schema::NodeFragmentCfKey =
                schema::NodeFragments::key_from_bytes(&key_bytes)?;
            let ts = key.1;

            // Check time range
            if !timestamp_in_range(ts, &self.time_range) {
                continue;
            }

            let value: schema::NodeFragmentCfValue =
                schema::NodeFragments::value_from_bytes(&value_bytes)?;

            // Check temporal validity
            if schema::is_valid_at_time(&value.0, ref_time) {
                results.push((ts, value.1));
            }
        }

        Ok(results)
    }
}

/// Implement TransactionQueryExecutor for EdgeFragmentsByIdTimeRange
impl TransactionQueryExecutor for EdgeFragmentsByIdTimeRange {
    type Output = Vec<(TimestampMilli, FragmentContent)>;

    fn execute_in_transaction(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        _cache: &super::name_hash::NameCache,
    ) -> Result<Self::Output> {
        tracing::debug!(
            source_id = %self.source_id,
            dest_id = %self.dest_id,
            edge_name = %self.edge_name,
            time_range = ?self.time_range,
            "Executing EdgeFragmentsByIdTimeRange query in transaction"
        );

        let ref_time = self
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let cf = txn_db
            .cf_handle(schema::EdgeFragments::CF_NAME)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Column family '{}' not found",
                    schema::EdgeFragments::CF_NAME
                )
            })?;

        // Convert edge_name String to NameHash for key construction
        let edge_name_hash = NameHash::from_name(&self.edge_name);

        // Create prefix for this edge's fragments
        let prefix_key = schema::EdgeFragmentCfKey(
            self.source_id,
            self.dest_id,
            edge_name_hash,
            TimestampMilli(0),
        );
        let prefix_bytes = schema::EdgeFragments::key_to_bytes(&prefix_key);
        // Prefix is src_id + dst_id + name_hash (now fixed 40 bytes)

        let mut results = Vec::new();

        let iter = txn.iterator_cf(
            cf,
            rocksdb::IteratorMode::From(&prefix_bytes, rocksdb::Direction::Forward),
        );

        for item in iter {
            let (key_bytes, value_bytes) = item?;

            // Parse key to check if it matches our edge
            let key: schema::EdgeFragmentCfKey = match schema::EdgeFragments::key_from_bytes(&key_bytes)
            {
                Ok(k) => k,
                Err(_) => break,
            };

            // Check if this is still our edge
            if key.0 != self.source_id || key.1 != self.dest_id || key.2 != edge_name_hash {
                break;
            }

            let ts = key.3;

            // Check time range
            if !timestamp_in_range(ts, &self.time_range) {
                continue;
            }

            let value: schema::EdgeFragmentCfValue =
                schema::EdgeFragments::value_from_bytes(&value_bytes)?;

            // Check temporal validity
            if schema::is_valid_at_time(&value.0, ref_time) {
                results.push((ts, value.1));
            }
        }

        Ok(results)
    }
}

/// Implement TransactionQueryExecutor for EdgeSummaryBySrcDstName
impl TransactionQueryExecutor for EdgeSummaryBySrcDstName {
    type Output = (EdgeSummary, Option<f64>);

    fn execute_in_transaction(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        _cache: &super::name_hash::NameCache,
    ) -> Result<Self::Output> {
        tracing::debug!(
            source_id = %self.source_id,
            dest_id = %self.dest_id,
            name = %self.name,
            "Executing EdgeSummaryBySrcDstName query in transaction"
        );

        let ref_time = self
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let cf = txn_db
            .cf_handle(schema::ForwardEdges::CF_NAME)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Column family '{}' not found",
                    schema::ForwardEdges::CF_NAME
                )
            })?;

        // Convert edge name to NameHash for key construction
        let name_hash = NameHash::from_name(&self.name);

        let key = schema::ForwardEdgeCfKey(
            self.source_id,
            self.dest_id,
            name_hash,
        );
        let key_bytes = schema::ForwardEdges::key_to_bytes(&key);

        let value_bytes = txn
            .get_cf(cf, &key_bytes)?
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Edge not found: {} -> {} ({})",
                    self.source_id,
                    self.dest_id,
                    self.name
                )
            })?;

        let value: schema::ForwardEdgeCfValue =
            schema::ForwardEdges::value_from_bytes(&value_bytes)?;

        // Check temporal validity
        if !schema::is_valid_at_time(&value.0, ref_time) {
            return Err(anyhow::anyhow!(
                "Edge {} -> {} ({}) not valid at time {}",
                self.source_id,
                self.dest_id,
                self.name,
                ref_time.0
            ));
        }

        Ok((value.2, value.1))
    }
}

/// Implement TransactionQueryExecutor for OutgoingEdges
impl TransactionQueryExecutor for OutgoingEdges {
    type Output = Vec<(Option<f64>, SrcId, DstId, EdgeName)>;

    fn execute_in_transaction(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        cache: &super::name_hash::NameCache,
    ) -> Result<Self::Output> {
        tracing::debug!(id = %self.id, "Executing OutgoingEdges query in transaction");

        let ref_time = self
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let cf = txn_db
            .cf_handle(schema::ForwardEdges::CF_NAME)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Column family '{}' not found",
                    schema::ForwardEdges::CF_NAME
                )
            })?;

        // Use source ID as prefix for iteration
        let prefix = self.id.into_bytes();
        // Collect edges with NameHash first
        let mut edges_with_hash: Vec<(Option<f64>, SrcId, DstId, NameHash)> = Vec::new();

        let iter = txn.iterator_cf(
            cf,
            rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
        );

        for item in iter {
            let (key_bytes, value_bytes) = item?;

            let key: schema::ForwardEdgeCfKey = schema::ForwardEdges::key_from_bytes(&key_bytes)?;

            // Check if still same source
            if key.0 != self.id {
                break;
            }

            let value: schema::ForwardEdgeCfValue =
                schema::ForwardEdges::value_from_bytes(&value_bytes)?;

            // Check temporal validity
            if schema::is_valid_at_time(&value.0, ref_time) {
                edges_with_hash.push((value.1, key.0, key.1, key.2));
            }
        }

        // Resolve NameHashes to Strings (uses cache)
        let mut results = Vec::with_capacity(edges_with_hash.len());
        for (weight, src_id, dst_id, name_hash) in edges_with_hash {
            let edge_name = resolve_name_from_txn(txn, txn_db, name_hash, cache)?;
            results.push((weight, src_id, dst_id, edge_name));
        }

        Ok(results)
    }
}

/// Implement TransactionQueryExecutor for IncomingEdges
impl TransactionQueryExecutor for IncomingEdges {
    type Output = Vec<(Option<f64>, DstId, SrcId, EdgeName)>;

    fn execute_in_transaction(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        cache: &super::name_hash::NameCache,
    ) -> Result<Self::Output> {
        tracing::debug!(id = %self.id, "Executing IncomingEdges query in transaction");

        let ref_time = self
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let reverse_cf = txn_db
            .cf_handle(schema::ReverseEdges::CF_NAME)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Column family '{}' not found",
                    schema::ReverseEdges::CF_NAME
                )
            })?;

        let forward_cf = txn_db
            .cf_handle(schema::ForwardEdges::CF_NAME)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Column family '{}' not found",
                    schema::ForwardEdges::CF_NAME
                )
            })?;

        // Use destination ID as prefix for iteration
        let prefix = self.id.into_bytes();
        // Collect edges with NameHash first
        let mut edges_with_hash: Vec<(Option<f64>, DstId, SrcId, NameHash)> = Vec::new();

        let iter = txn.iterator_cf(
            reverse_cf,
            rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
        );

        for item in iter {
            let (key_bytes, value_bytes) = item?;

            let key: schema::ReverseEdgeCfKey = schema::ReverseEdges::key_from_bytes(&key_bytes)?;

            // Check if still same destination
            if key.0 != self.id {
                break;
            }

            let value: schema::ReverseEdgeCfValue =
                schema::ReverseEdges::value_from_bytes(&value_bytes)?;

            // Check temporal validity
            if schema::is_valid_at_time(&value.0, ref_time) {
                let dest_id = key.0;
                let source_id = key.1;
                let edge_name_hash = key.2;

                // Lookup weight from ForwardEdges CF
                let forward_key = schema::ForwardEdgeCfKey(source_id, dest_id, edge_name_hash);
                let forward_key_bytes = schema::ForwardEdges::key_to_bytes(&forward_key);

                let weight = if let Some(forward_value_bytes) =
                    txn.get_cf(forward_cf, &forward_key_bytes)?
                {
                    let forward_value: schema::ForwardEdgeCfValue =
                        schema::ForwardEdges::value_from_bytes(&forward_value_bytes)?;
                    forward_value.1
                } else {
                    None
                };

                edges_with_hash.push((weight, dest_id, source_id, edge_name_hash));
            }
        }

        // Resolve NameHashes to Strings (uses cache)
        let mut results = Vec::with_capacity(edges_with_hash.len());
        for (weight, dst_id, src_id, name_hash) in edges_with_hash {
            let edge_name = resolve_name_from_txn(txn, txn_db, name_hash, cache)?;
            results.push((weight, dst_id, src_id, edge_name));
        }

        Ok(results)
    }
}

/// Implement TransactionQueryExecutor for AllNodes
impl TransactionQueryExecutor for AllNodes {
    type Output = Vec<(Id, NodeName, NodeSummary)>;

    fn execute_in_transaction(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        cache: &super::name_hash::NameCache,
    ) -> Result<Self::Output> {
        tracing::debug!(limit = self.limit, cursor = ?self.last, "Executing AllNodes query in transaction");

        let ref_time = self
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let cf = txn_db.cf_handle(schema::Nodes::CF_NAME).ok_or_else(|| {
            anyhow::anyhow!("Column family '{}' not found", schema::Nodes::CF_NAME)
        })?;

        let mut results = Vec::with_capacity(self.limit);

        // Determine start position - need to create bytes outside the if-let
        // to avoid lifetime issues
        let start_bytes: Option<Vec<u8>> = self.last.as_ref().map(|cursor| {
            let start_key = schema::NodeCfKey(*cursor);
            schema::Nodes::key_to_bytes(&start_key)
        });

        let iter = if let Some(ref bytes) = start_bytes {
            txn.iterator_cf(cf, rocksdb::IteratorMode::From(bytes, rocksdb::Direction::Forward))
        } else {
            txn.iterator_cf(cf, rocksdb::IteratorMode::Start)
        };

        let mut skip_first = self.last.is_some();

        for item in iter {
            let (key_bytes, value_bytes) = item?;

            // Skip the cursor position itself
            if skip_first {
                skip_first = false;
                continue;
            }

            if results.len() >= self.limit {
                break;
            }

            let key: schema::NodeCfKey = schema::Nodes::key_from_bytes(&key_bytes)?;
            let value: schema::NodeCfValue = schema::Nodes::value_from_bytes(&value_bytes)?;

            // Check temporal validity
            if schema::is_valid_at_time(&value.0, ref_time) {
                // Resolve NameHash to String (uses cache)
                let node_name = resolve_name_from_txn(txn, txn_db, value.1, cache)?;
                results.push((key.0, node_name, value.2));
            }
        }

        Ok(results)
    }
}

/// Implement TransactionQueryExecutor for AllEdges
impl TransactionQueryExecutor for AllEdges {
    type Output = Vec<(Option<f64>, SrcId, DstId, EdgeName)>;

    fn execute_in_transaction(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
        cache: &super::name_hash::NameCache,
    ) -> Result<Self::Output> {
        tracing::debug!(limit = self.limit, cursor = ?self.last, "Executing AllEdges query in transaction");

        let ref_time = self
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let cf = txn_db
            .cf_handle(schema::ForwardEdges::CF_NAME)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Column family '{}' not found",
                    schema::ForwardEdges::CF_NAME
                )
            })?;

        // Determine start position - need to create bytes outside the if-let
        // to avoid lifetime issues. Convert name to NameHash for cursor.
        let start_bytes: Option<Vec<u8>> = self.last.as_ref().map(|(src, dst, name)| {
            let name_hash = NameHash::from_name(name);
            let start_key = schema::ForwardEdgeCfKey(*src, *dst, name_hash);
            schema::ForwardEdges::key_to_bytes(&start_key)
        });

        let iter = if let Some(ref bytes) = start_bytes {
            txn.iterator_cf(cf, rocksdb::IteratorMode::From(bytes, rocksdb::Direction::Forward))
        } else {
            txn.iterator_cf(cf, rocksdb::IteratorMode::Start)
        };

        let mut skip_first = self.last.is_some();
        // Collect edges with NameHash first
        let mut edges_with_hash: Vec<(Option<f64>, SrcId, DstId, NameHash)> = Vec::new();

        for item in iter {
            let (key_bytes, value_bytes) = item?;

            // Skip the cursor position itself
            if skip_first {
                skip_first = false;
                continue;
            }

            if edges_with_hash.len() >= self.limit {
                break;
            }

            let key: schema::ForwardEdgeCfKey = schema::ForwardEdges::key_from_bytes(&key_bytes)?;
            let value: schema::ForwardEdgeCfValue =
                schema::ForwardEdges::value_from_bytes(&value_bytes)?;

            // Check temporal validity
            if schema::is_valid_at_time(&value.0, ref_time) {
                edges_with_hash.push((value.1, key.0, key.1, key.2));
            }
        }

        // Resolve NameHashes to Strings (uses cache)
        let mut results = Vec::with_capacity(edges_with_hash.len());
        for (weight, src_id, dst_id, name_hash) in edges_with_hash {
            let edge_name = resolve_name_from_txn(txn, txn_db, name_hash, cache)?;
            results.push((weight, src_id, dst_id, edge_name));
        }

        Ok(results)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::super::mutation::AddNode;
    use crate::writer::Runnable as MutRunnable;
    use super::super::reader::{
        create_query_reader, spawn_consumer, Consumer, QueryWithTimeout, Reader, ReaderConfig,
    };
    use super::super::writer::{create_mutation_writer, spawn_mutation_consumer, WriterConfig};
    use super::super::{Graph, Storage};
    use super::*;
    use crate::{DataUrl, Id, TimestampMilli};
    use std::sync::Arc;
    use tempfile::TempDir;
    use tokio::time::Duration;

    #[tokio::test]
    async fn test_consumer_basic() {
        // Create a temporary database with real data
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Create writer and mutation consumer
        let writer_config = WriterConfig {
            channel_buffer_size: 10,
        };
        let (writer, mutation_receiver) = create_mutation_writer(writer_config.clone());
        let mutation_consumer_handle =
            spawn_mutation_consumer(mutation_receiver, writer_config, &db_path);

        // Insert a test node
        let node_id = Id::new();
        let node_name = "test_node".to_string();
        let node_args = AddNode {
            id: node_id,
            ts_millis: TimestampMilli::now(),
            name: node_name.clone(),
            valid_range: None,
            summary: NodeSummary::from_text("test summary"),
        };
        node_args.run(&writer).await.unwrap();

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Drop writer to close mutation channel
        drop(writer);

        // Wait for mutation consumer to finish
        mutation_consumer_handle.await.unwrap().unwrap();

        // Now open storage for reading
        let mut storage = Storage::readonly(&db_path);
        storage.ready().unwrap();
        let graph = Graph::new(Arc::new(storage));

        // Create reader and query consumer
        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };

        let (reader, receiver) = create_query_reader(reader_config.clone());

        let consumer = Consumer::new(receiver, reader_config, graph);

        // Spawn query consumer
        let consumer_handle = spawn_consumer(consumer);

        // Query the node we just created
        let (returned_name, returned_summary) = NodeById::new(node_id, None)
            .run(&reader, Duration::from_secs(5))
            .await
            .unwrap();

        // Verify the data matches what we inserted
        assert_eq!(returned_name, node_name);
        assert!(returned_summary.decode_string().is_ok());

        // Drop reader to close channel
        drop(reader);

        // Wait for consumer to finish
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_query_timeout() {
        // Create a custom QueryExecutor that sleeps longer than the timeout
        struct SlowNodeByIdQuery {
            id: Id,
            timeout: Duration,
        }

        #[async_trait::async_trait]
        impl QueryExecutor for SlowNodeByIdQuery {
            type Output = (NodeName, NodeSummary);

            async fn execute(&self, _storage: &Storage) -> Result<Self::Output> {
                // Sleep longer than the query timeout
                tokio::time::sleep(Duration::from_millis(200)).await;
                Ok((
                    "should_not_get_here".to_string(),
                    DataUrl::from_markdown("Should not get here"),
                ))
            }

            fn timeout(&self) -> Duration {
                self.timeout
            }
        }

        // Create a temporary database
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let mut storage = Storage::readwrite(&db_path);
        storage.ready().unwrap();

        let graph = Graph::new(Arc::new(storage));

        // Test the timeout directly using QueryWithTimeout trait
        let slow_query = SlowNodeByIdQuery {
            id: Id::new(),
            timeout: Duration::from_millis(50), // Short timeout
        };

        let result = slow_query.result(&graph).await;

        // Should timeout because query takes 200ms but timeout is 50ms
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("timeout") || err_msg.contains("Query timeout"));
    }

    #[tokio::test]
    async fn test_edge_summary_by_src_dst_name_query() {
        use super::super::mutation::{AddEdge, AddNode};
        use super::super::schema::EdgeSummary;
        use crate::writer::Runnable as MutationRunnable;
        use super::super::writer::{create_mutation_writer, spawn_mutation_consumer, WriterConfig};
        use crate::{Id, TimestampMilli};
        use tempfile::TempDir;

        // Create temporary database
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let config = WriterConfig {
            channel_buffer_size: 10,
        };

        // Create writer and spawn graph consumer for mutations
        let (writer, mutation_receiver) = create_mutation_writer(config.clone());
        let mutation_consumer_handle = spawn_mutation_consumer(mutation_receiver, config, &db_path);

        // Create source and destination nodes
        let source_id = Id::new();
        let dest_id = Id::new();

        AddNode {
            id: source_id,
            ts_millis: TimestampMilli::now(),
            name: "source_node".to_string(),
            valid_range: None,
            summary: NodeSummary::from_text("source summary"),
        }
        .run(&writer)
        .await
        .unwrap();

        AddNode {
            id: dest_id,
            ts_millis: TimestampMilli::now(),
            name: "dest_node".to_string(),
            valid_range: None,
            summary: NodeSummary::from_text("dest summary"),
        }
        .run(&writer)
        .await
        .unwrap();

        // Create an edge
        let edge_name = "test_edge";
        let edge_summary = EdgeSummary::from_text("This is a test edge");
        let edge_weight = 2.5;

        AddEdge {
            source_node_id: source_id,
            target_node_id: dest_id,
            ts_millis: TimestampMilli::now(),
            name: edge_name.to_string(),
            summary: edge_summary.clone(),
            weight: Some(edge_weight),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        // Give time for mutations to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Close writer to ensure all mutations are flushed
        drop(writer);

        // Wait for mutation consumer to finish
        mutation_consumer_handle.await.unwrap().unwrap();

        // Now open storage for reading
        let mut storage = Storage::readonly(&db_path);
        storage.ready().unwrap();
        let graph = Graph::new(Arc::new(storage));

        // Create reader and query consumer
        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };

        let (reader, receiver) = {
            let (sender, receiver) = flume::bounded(reader_config.channel_buffer_size);
            let reader = Reader::new(sender);
            (reader, receiver)
        };

        let consumer = Consumer::new(receiver, reader_config, graph);

        // Spawn query consumer
        let consumer_handle = spawn_consumer(consumer);

        // Query the edge using EdgeSummaryBySrcDstName with Runnable pattern
        let (returned_summary, returned_weight) =
            EdgeSummaryBySrcDstName::new(source_id, dest_id, edge_name.to_string(), None)
                .run(&reader, Duration::from_secs(5))
                .await
                .unwrap();

        // Verify edge summary content matches
        assert_eq!(returned_summary.0, edge_summary.0);

        // Verify weight matches
        assert_eq!(returned_weight, Some(edge_weight));

        // Cleanup
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_edge_fragments_by_id_time_range_basic() {
        use super::super::mutation::{
            AddEdge, AddEdgeFragment, AddNode,
        };
        use crate::writer::Runnable as MutationRunnable;
        use super::super::schema::EdgeSummary;
        use super::super::writer::{create_mutation_writer, spawn_mutation_consumer, WriterConfig};
        use crate::{Id, TimestampMilli};
        use std::ops::Bound;
        use tempfile::TempDir;

        // Create temporary database
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let config = WriterConfig {
            channel_buffer_size: 10,
        };

        // Create writer and spawn graph consumer for mutations
        let (writer, mutation_receiver) = create_mutation_writer(config.clone());
        let mutation_consumer_handle = spawn_mutation_consumer(mutation_receiver, config, &db_path);

        // Create source and destination nodes
        let source_id = Id::new();
        let dest_id = Id::new();
        let edge_name = "test_edge";

        AddNode {
            id: source_id,
            ts_millis: TimestampMilli::now(),
            name: "source_node".to_string(),
            valid_range: None,
            summary: NodeSummary::from_text("source summary"),
        }
        .run(&writer)
        .await
        .unwrap();

        AddNode {
            id: dest_id,
            ts_millis: TimestampMilli::now(),
            name: "dest_node".to_string(),
            valid_range: None,
            summary: NodeSummary::from_text("dest summary"),
        }
        .run(&writer)
        .await
        .unwrap();

        // Create an edge
        let edge_summary = EdgeSummary::from_text("Test edge for fragments");
        AddEdge {
            source_node_id: source_id,
            target_node_id: dest_id,
            ts_millis: TimestampMilli::now(),
            name: edge_name.to_string(),
            summary: edge_summary.clone(),
            weight: Some(1.0),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        // Add edge fragments at different timestamps
        let base_time = TimestampMilli::now();
        let fragment1_time = TimestampMilli(base_time.0 + 1000);
        let fragment2_time = TimestampMilli(base_time.0 + 2000);
        let fragment3_time = TimestampMilli(base_time.0 + 3000);

        AddEdgeFragment {
            src_id: source_id,
            dst_id: dest_id,
            edge_name: edge_name.to_string(),
            ts_millis: fragment1_time,
            content: DataUrl::from_markdown("Fragment 1 content"),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        AddEdgeFragment {
            src_id: source_id,
            dst_id: dest_id,
            edge_name: edge_name.to_string(),
            ts_millis: fragment2_time,
            content: DataUrl::from_markdown("Fragment 2 content"),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        AddEdgeFragment {
            src_id: source_id,
            dst_id: dest_id,
            edge_name: edge_name.to_string(),
            ts_millis: fragment3_time,
            content: DataUrl::from_markdown("Fragment 3 content"),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        // Give time for mutations to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Close writer to ensure all mutations are flushed
        drop(writer);

        // Wait for mutation consumer to finish
        mutation_consumer_handle.await.unwrap().unwrap();

        // Now open storage for reading
        let mut storage = Storage::readonly(&db_path);
        storage.ready().unwrap();
        let graph = Graph::new(Arc::new(storage));

        // Create reader and query consumer
        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };

        let (reader, receiver) = {
            let (sender, receiver) = flume::bounded(reader_config.channel_buffer_size);
            let reader = Reader::new(sender);
            (reader, receiver)
        };

        let consumer = Consumer::new(receiver, reader_config, graph);

        // Spawn query consumer
        let consumer_handle = spawn_consumer(consumer);

        // Query all fragments (unbounded range)
        let fragments = EdgeFragmentsByIdTimeRange::new(
            source_id,
            dest_id,
            edge_name.to_string(),
            (Bound::Unbounded, Bound::Unbounded),
            None,
        )
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

        // Verify we got all 3 fragments
        assert_eq!(fragments.len(), 3);
        assert_eq!(fragments[0].0, fragment1_time);
        assert_eq!(fragments[1].0, fragment2_time);
        assert_eq!(fragments[2].0, fragment3_time);

        // Verify content
        assert!(fragments[0]
            .1
            .decode_string()
            .unwrap()
            .contains("Fragment 1 content"));
        assert!(fragments[1]
            .1
            .decode_string()
            .unwrap()
            .contains("Fragment 2 content"));
        assert!(fragments[2]
            .1
            .decode_string()
            .unwrap()
            .contains("Fragment 3 content"));

        // Cleanup
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_edge_fragments_by_id_time_range_with_bounds() {
        use super::super::mutation::{
            AddEdge, AddEdgeFragment, AddNode,
        };
        use crate::writer::Runnable as MutationRunnable;
        use super::super::schema::EdgeSummary;
        use super::super::writer::{create_mutation_writer, spawn_mutation_consumer, WriterConfig};
        use crate::{Id, TimestampMilli};
        use std::ops::Bound;
        use tempfile::TempDir;

        // Create temporary database
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let config = WriterConfig {
            channel_buffer_size: 10,
        };

        // Create writer and spawn graph consumer for mutations
        let (writer, mutation_receiver) = create_mutation_writer(config.clone());
        let mutation_consumer_handle = spawn_mutation_consumer(mutation_receiver, config, &db_path);

        // Create source and destination nodes
        let source_id = Id::new();
        let dest_id = Id::new();
        let edge_name = "bounded_edge";

        AddNode {
            id: source_id,
            ts_millis: TimestampMilli::now(),
            name: "source_node".to_string(),
            valid_range: None,
            summary: NodeSummary::from_text("source summary"),
        }
        .run(&writer)
        .await
        .unwrap();

        AddNode {
            id: dest_id,
            ts_millis: TimestampMilli::now(),
            name: "dest_node".to_string(),
            valid_range: None,
            summary: NodeSummary::from_text("dest summary"),
        }
        .run(&writer)
        .await
        .unwrap();

        // Create an edge
        AddEdge {
            source_node_id: source_id,
            target_node_id: dest_id,
            ts_millis: TimestampMilli::now(),
            name: edge_name.to_string(),
            summary: EdgeSummary::from_text("Bounded test edge"),
            weight: Some(1.0),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        // Add edge fragments at specific timestamps
        let t1 = TimestampMilli(1000);
        let t2 = TimestampMilli(2000);
        let t3 = TimestampMilli(3000);
        let t4 = TimestampMilli(4000);
        let t5 = TimestampMilli(5000);

        for (ts, content) in [
            (t1, "Fragment at t1"),
            (t2, "Fragment at t2"),
            (t3, "Fragment at t3"),
            (t4, "Fragment at t4"),
            (t5, "Fragment at t5"),
        ] {
            AddEdgeFragment {
                src_id: source_id,
                dst_id: dest_id,
                edge_name: edge_name.to_string(),
                ts_millis: ts,
                content: DataUrl::from_markdown(content),
                valid_range: None,
            }
            .run(&writer)
            .await
            .unwrap();
        }

        // Give time for mutations to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Close writer
        drop(writer);
        mutation_consumer_handle.await.unwrap().unwrap();

        // Now open storage for reading
        let mut storage = Storage::readonly(&db_path);
        storage.ready().unwrap();
        let graph = Graph::new(Arc::new(storage));

        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };

        let (reader, receiver) = {
            let (sender, receiver) = flume::bounded(reader_config.channel_buffer_size);
            let reader = Reader::new(sender);
            (reader, receiver)
        };

        let consumer = Consumer::new(receiver, reader_config, graph);
        let consumer_handle = spawn_consumer(consumer);

        // Test 1: Query with inclusive bounds [t2, t4]
        let fragments = EdgeFragmentsByIdTimeRange::new(
            source_id,
            dest_id,
            edge_name.to_string(),
            (Bound::Included(t2), Bound::Included(t4)),
            None,
        )
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

        assert_eq!(fragments.len(), 3); // t2, t3, t4
        assert_eq!(fragments[0].0, t2);
        assert_eq!(fragments[1].0, t3);
        assert_eq!(fragments[2].0, t4);

        // Test 2: Query with exclusive bounds (t2, t4)
        let fragments = EdgeFragmentsByIdTimeRange::new(
            source_id,
            dest_id,
            edge_name.to_string(),
            (Bound::Excluded(t2), Bound::Excluded(t4)),
            None,
        )
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

        assert_eq!(fragments.len(), 1); // Only t3
        assert_eq!(fragments[0].0, t3);

        // Test 3: Query with unbounded start
        let fragments = EdgeFragmentsByIdTimeRange::new(
            source_id,
            dest_id,
            edge_name.to_string(),
            (Bound::Unbounded, Bound::Included(t2)),
            None,
        )
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

        assert_eq!(fragments.len(), 2); // t1, t2
        assert_eq!(fragments[0].0, t1);
        assert_eq!(fragments[1].0, t2);

        // Test 4: Query with unbounded end
        let fragments = EdgeFragmentsByIdTimeRange::new(
            source_id,
            dest_id,
            edge_name.to_string(),
            (Bound::Included(t4), Bound::Unbounded),
            None,
        )
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

        assert_eq!(fragments.len(), 2); // t4, t5
        assert_eq!(fragments[0].0, t4);
        assert_eq!(fragments[1].0, t5);

        // Cleanup
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_edge_fragments_by_id_time_range_with_temporal_validity() {
        use super::super::mutation::{
            AddEdge, AddEdgeFragment, AddNode,
        };
        use crate::writer::Runnable as MutationRunnable;
        use super::super::schema::EdgeSummary;
        use super::super::writer::{create_mutation_writer, spawn_mutation_consumer, WriterConfig};
        use crate::{Id, TemporalRange, TimestampMilli};
        use std::ops::Bound;
        use tempfile::TempDir;

        // Create temporary database
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let config = WriterConfig {
            channel_buffer_size: 10,
        };

        let (writer, mutation_receiver) = create_mutation_writer(config.clone());
        let mutation_consumer_handle = spawn_mutation_consumer(mutation_receiver, config, &db_path);

        // Create source and destination nodes
        let source_id = Id::new();
        let dest_id = Id::new();
        let edge_name = "temporal_edge";

        AddNode {
            id: source_id,
            ts_millis: TimestampMilli::now(),
            name: "source_node".to_string(),
            valid_range: None,
            summary: NodeSummary::from_text("source summary"),
        }
        .run(&writer)
        .await
        .unwrap();

        AddNode {
            id: dest_id,
            ts_millis: TimestampMilli::now(),
            name: "dest_node".to_string(),
            valid_range: None,
            summary: NodeSummary::from_text("dest summary"),
        }
        .run(&writer)
        .await
        .unwrap();

        // Create an edge
        AddEdge {
            source_node_id: source_id,
            target_node_id: dest_id,
            ts_millis: TimestampMilli::now(),
            name: edge_name.to_string(),
            summary: EdgeSummary::from_text("Temporal validity test edge"),
            weight: Some(1.0),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        // Add edge fragments with different temporal ranges
        // Fragment 1: Valid from 1000 to 3000
        AddEdgeFragment {
            src_id: source_id,
            dst_id: dest_id,
            edge_name: edge_name.to_string(),
            ts_millis: TimestampMilli(1000),
            content: DataUrl::from_markdown("Valid from 1000 to 3000"),
            valid_range: TemporalRange::valid_between(TimestampMilli(1000), TimestampMilli(3000)),
        }
        .run(&writer)
        .await
        .unwrap();

        // Fragment 2: Valid from 2000 to 5000
        AddEdgeFragment {
            src_id: source_id,
            dst_id: dest_id,
            edge_name: edge_name.to_string(),
            ts_millis: TimestampMilli(2000),
            content: DataUrl::from_markdown("Valid from 2000 to 5000"),
            valid_range: TemporalRange::valid_between(TimestampMilli(2000), TimestampMilli(5000)),
        }
        .run(&writer)
        .await
        .unwrap();

        // Fragment 3: No temporal range (always valid)
        AddEdgeFragment {
            src_id: source_id,
            dst_id: dest_id,
            edge_name: edge_name.to_string(),
            ts_millis: TimestampMilli(3000),
            content: DataUrl::from_markdown("Always valid"),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;
        drop(writer);
        mutation_consumer_handle.await.unwrap().unwrap();

        // Now open storage for reading
        let mut storage = Storage::readonly(&db_path);
        storage.ready().unwrap();
        let graph = Graph::new(Arc::new(storage));

        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };

        let (reader, receiver) = {
            let (sender, receiver) = flume::bounded(reader_config.channel_buffer_size);
            let reader = Reader::new(sender);
            (reader, receiver)
        };

        let consumer = Consumer::new(receiver, reader_config, graph);
        let consumer_handle = spawn_consumer(consumer);

        // Query at reference time 2500 - should see fragments 1, 2, and 3
        let fragments = EdgeFragmentsByIdTimeRange::new(
            source_id,
            dest_id,
            edge_name.to_string(),
            (Bound::Unbounded, Bound::Unbounded),
            Some(TimestampMilli(2500)),
        )
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

        assert_eq!(fragments.len(), 3);

        // Query at reference time 500 - should see only fragment 3 (always valid)
        let fragments = EdgeFragmentsByIdTimeRange::new(
            source_id,
            dest_id,
            edge_name.to_string(),
            (Bound::Unbounded, Bound::Unbounded),
            Some(TimestampMilli(500)),
        )
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0].0, TimestampMilli(3000));
        assert!(fragments[0]
            .1
            .decode_string()
            .unwrap()
            .contains("Always valid"));

        // Query at reference time 3500 - should see fragments 2 and 3
        let fragments = EdgeFragmentsByIdTimeRange::new(
            source_id,
            dest_id,
            edge_name.to_string(),
            (Bound::Unbounded, Bound::Unbounded),
            Some(TimestampMilli(3500)),
        )
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

        assert_eq!(fragments.len(), 2);
        assert_eq!(fragments[0].0, TimestampMilli(2000));
        assert_eq!(fragments[1].0, TimestampMilli(3000));

        // Cleanup
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_edge_fragments_empty_result() {
        use super::super::mutation::{AddEdge, AddNode};
        use super::super::schema::{EdgeSummary, NodeSummary};
        use crate::writer::Runnable as MutationRunnable;
        use super::super::writer::{create_mutation_writer, spawn_mutation_consumer, WriterConfig};
        use crate::{Id, TimestampMilli};
        use std::ops::Bound;
        use tempfile::TempDir;

        // Create temporary database
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let config = WriterConfig {
            channel_buffer_size: 10,
        };

        let (writer, mutation_receiver) = create_mutation_writer(config.clone());
        let mutation_consumer_handle = spawn_mutation_consumer(mutation_receiver, config, &db_path);

        // Create source and destination nodes and edge but NO fragments
        let source_id = Id::new();
        let dest_id = Id::new();
        let edge_name = "empty_edge";

        AddNode {
            id: source_id,
            ts_millis: TimestampMilli::now(),
            name: "source_node".to_string(),
            valid_range: None,
            summary: NodeSummary::from_text("source summary"),
        }
        .run(&writer)
        .await
        .unwrap();

        AddNode {
            id: dest_id,
            ts_millis: TimestampMilli::now(),
            name: "dest_node".to_string(),
            valid_range: None,
            summary: NodeSummary::from_text("dest summary"),
        }
        .run(&writer)
        .await
        .unwrap();

        AddEdge {
            source_node_id: source_id,
            target_node_id: dest_id,
            ts_millis: TimestampMilli::now(),
            name: edge_name.to_string(),
            summary: EdgeSummary::from_text("Edge with no fragments"),
            weight: Some(1.0),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;
        drop(writer);
        mutation_consumer_handle.await.unwrap().unwrap();

        // Now open storage for reading
        let mut storage = Storage::readonly(&db_path);
        storage.ready().unwrap();
        let graph = Graph::new(Arc::new(storage));

        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };

        let (reader, receiver) = {
            let (sender, receiver) = flume::bounded(reader_config.channel_buffer_size);
            let reader = Reader::new(sender);
            (reader, receiver)
        };

        let consumer = Consumer::new(receiver, reader_config, graph);
        let consumer_handle = spawn_consumer(consumer);

        // Query should return empty vector
        let fragments = EdgeFragmentsByIdTimeRange::new(
            source_id,
            dest_id,
            edge_name.to_string(),
            (Bound::Unbounded, Bound::Unbounded),
            None,
        )
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

        assert_eq!(fragments.len(), 0);

        // Cleanup
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_nodes_by_ids_multi_basic() {
        // Create a temporary database with real data
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Create writer and mutation consumer
        let writer_config = WriterConfig {
            channel_buffer_size: 10,
        };
        let (writer, mutation_receiver) = create_mutation_writer(writer_config.clone());
        let mutation_consumer_handle =
            spawn_mutation_consumer(mutation_receiver, writer_config, &db_path);

        // Insert multiple test nodes
        let node_ids: Vec<Id> = (0..5).map(|_| Id::new()).collect();
        for (i, &id) in node_ids.iter().enumerate() {
            let node_args = AddNode {
                id,
                ts_millis: TimestampMilli::now(),
                name: format!("node_{}", i),
                valid_range: None,
                summary: NodeSummary::from_text(&format!("summary for node {}", i)),
            };
            node_args.run(&writer).await.unwrap();
        }

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Drop writer to close mutation channel
        drop(writer);

        // Wait for mutation consumer to finish
        mutation_consumer_handle.await.unwrap().unwrap();

        // Now open storage for reading
        let mut storage = Storage::readonly(&db_path);
        storage.ready().unwrap();
        let graph = Graph::new(Arc::new(storage));

        // Create reader and query consumer
        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };

        let (reader, receiver) = create_query_reader(reader_config.clone());
        let consumer = Consumer::new(receiver, reader_config, graph);
        let consumer_handle = spawn_consumer(consumer);

        // Query all nodes with batch lookup
        let results = NodesByIdsMulti::new(node_ids.clone(), None)
            .run(&reader, Duration::from_secs(5))
            .await
            .unwrap();

        // Verify all nodes were found
        assert_eq!(results.len(), 5);

        // Verify each node is present
        for (id, name, _summary) in &results {
            assert!(node_ids.contains(id));
            assert!(name.starts_with("node_"));
        }

        // Cleanup
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_nodes_by_ids_multi_missing_nodes() {
        // Create a temporary database with real data
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Create writer and mutation consumer
        let writer_config = WriterConfig {
            channel_buffer_size: 10,
        };
        let (writer, mutation_receiver) = create_mutation_writer(writer_config.clone());
        let mutation_consumer_handle =
            spawn_mutation_consumer(mutation_receiver, writer_config, &db_path);

        // Insert only 2 test nodes
        let existing_ids: Vec<Id> = (0..2).map(|_| Id::new()).collect();
        for (i, &id) in existing_ids.iter().enumerate() {
            let node_args = AddNode {
                id,
                ts_millis: TimestampMilli::now(),
                name: format!("node_{}", i),
                valid_range: None,
                summary: NodeSummary::from_text(&format!("summary for node {}", i)),
            };
            node_args.run(&writer).await.unwrap();
        }

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Drop writer to close mutation channel
        drop(writer);

        // Wait for mutation consumer to finish
        mutation_consumer_handle.await.unwrap().unwrap();

        // Now open storage for reading
        let mut storage = Storage::readonly(&db_path);
        storage.ready().unwrap();
        let graph = Graph::new(Arc::new(storage));

        // Create reader and query consumer
        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };

        let (reader, receiver) = create_query_reader(reader_config.clone());
        let consumer = Consumer::new(receiver, reader_config, graph);
        let consumer_handle = spawn_consumer(consumer);

        // Query with a mix of existing and non-existing IDs
        let non_existing_ids: Vec<Id> = (0..3).map(|_| Id::new()).collect();
        let all_ids: Vec<Id> = existing_ids
            .iter()
            .chain(non_existing_ids.iter())
            .copied()
            .collect();

        let results = NodesByIdsMulti::new(all_ids, None)
            .run(&reader, Duration::from_secs(5))
            .await
            .unwrap();

        // Should only return the 2 existing nodes
        assert_eq!(results.len(), 2);

        // Verify only existing IDs are in results
        for (id, _name, _summary) in &results {
            assert!(existing_ids.contains(id));
        }

        // Cleanup
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_nodes_by_ids_multi_empty_input() {
        // Create a temporary database
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Create writer and mutation consumer
        let writer_config = WriterConfig {
            channel_buffer_size: 10,
        };
        let (writer, mutation_receiver) = create_mutation_writer(writer_config.clone());
        let mutation_consumer_handle =
            spawn_mutation_consumer(mutation_receiver, writer_config, &db_path);

        // Insert one node so DB is properly initialized
        let node_args = AddNode {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "dummy".to_string(),
            valid_range: None,
            summary: NodeSummary::from_text("dummy"),
        };
        node_args.run(&writer).await.unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;
        drop(writer);
        mutation_consumer_handle.await.unwrap().unwrap();

        // Now open storage for reading
        let mut storage = Storage::readonly(&db_path);
        storage.ready().unwrap();
        let graph = Graph::new(Arc::new(storage));

        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };

        let (reader, receiver) = create_query_reader(reader_config.clone());
        let consumer = Consumer::new(receiver, reader_config, graph);
        let consumer_handle = spawn_consumer(consumer);

        // Query with empty input
        let results = NodesByIdsMulti::new(vec![], None)
            .run(&reader, Duration::from_secs(5))
            .await
            .unwrap();

        // Should return empty vector
        assert_eq!(results.len(), 0);

        // Cleanup
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_all_nodes_query() {
        // Create a temporary database with real data
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Create writer and mutation consumer
        let writer_config = WriterConfig {
            channel_buffer_size: 10,
        };
        let (writer, mutation_receiver) = create_mutation_writer(writer_config.clone());
        let mutation_consumer_handle =
            spawn_mutation_consumer(mutation_receiver, writer_config, &db_path);

        // Insert multiple test nodes
        let node_ids: Vec<Id> = (0..5).map(|_| Id::new()).collect();
        for (i, &node_id) in node_ids.iter().enumerate() {
            let node_args = AddNode {
                id: node_id,
                ts_millis: TimestampMilli::now(),
                name: format!("node_{}", i),
                valid_range: None,
                summary: NodeSummary::from_text(&format!("summary for node {}", i)),
            };
            node_args.run(&writer).await.unwrap();
        }

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;
        drop(writer);
        mutation_consumer_handle.await.unwrap().unwrap();

        // Now open storage for reading
        let mut storage = Storage::readonly(&db_path);
        storage.ready().unwrap();
        let graph = Graph::new(Arc::new(storage));

        // Create reader and query consumer
        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };
        let (reader, receiver) = create_query_reader(reader_config.clone());
        let consumer = Consumer::new(receiver, reader_config, graph);
        let consumer_handle = spawn_consumer(consumer);

        // Query all nodes
        let results = AllNodes::new(100)
            .run(&reader, Duration::from_secs(5))
            .await
            .unwrap();

        // Should return all 5 nodes
        assert_eq!(results.len(), 5, "Should return all 5 nodes");

        // Verify each node has expected data
        for (id, name, summary) in &results {
            assert!(!id.is_nil(), "Node ID should not be nil");
            assert!(name.starts_with("node_"), "Node name should start with 'node_'");
            assert!(
                summary.decode_string().unwrap().contains("summary for node"),
                "Summary should contain expected text"
            );
        }

        // Cleanup
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_all_nodes_pagination() {
        // Create a temporary database with real data
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Create writer and mutation consumer
        let writer_config = WriterConfig {
            channel_buffer_size: 10,
        };
        let (writer, mutation_receiver) = create_mutation_writer(writer_config.clone());
        let mutation_consumer_handle =
            spawn_mutation_consumer(mutation_receiver, writer_config, &db_path);

        // Insert 10 test nodes
        for i in 0..10 {
            let node_args = AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                name: format!("node_{}", i),
                valid_range: None,
                summary: NodeSummary::from_text(&format!("summary {}", i)),
            };
            node_args.run(&writer).await.unwrap();
        }

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;
        drop(writer);
        mutation_consumer_handle.await.unwrap().unwrap();

        // Now open storage for reading
        let mut storage = Storage::readonly(&db_path);
        storage.ready().unwrap();
        let graph = Graph::new(Arc::new(storage));

        // Create reader and query consumer
        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };
        let (reader, receiver) = create_query_reader(reader_config.clone());
        let consumer = Consumer::new(receiver, reader_config, graph);
        let consumer_handle = spawn_consumer(consumer);

        // Get first page of 3 nodes
        let page1 = AllNodes::new(3)
            .run(&reader, Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(page1.len(), 3, "First page should have 3 nodes");

        // Get second page using cursor
        let last_id = page1.last().unwrap().0;
        let page2 = AllNodes::new(3)
            .with_cursor(last_id)
            .run(&reader, Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(page2.len(), 3, "Second page should have 3 nodes");

        // Ensure no overlap between pages
        let page1_ids: std::collections::HashSet<_> = page1.iter().map(|(id, _, _)| *id).collect();
        let page2_ids: std::collections::HashSet<_> = page2.iter().map(|(id, _, _)| *id).collect();
        assert!(
            page1_ids.is_disjoint(&page2_ids),
            "Pages should not overlap"
        );

        // Cleanup
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_all_edges_query() {
        use super::super::mutation::AddEdge;

        // Create a temporary database with real data
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Create writer and mutation consumer
        let writer_config = WriterConfig {
            channel_buffer_size: 10,
        };
        let (writer, mutation_receiver) = create_mutation_writer(writer_config.clone());
        let mutation_consumer_handle =
            spawn_mutation_consumer(mutation_receiver, writer_config, &db_path);

        // Insert nodes first
        let node1 = Id::new();
        let node2 = Id::new();
        let node3 = Id::new();

        for (id, name) in [(node1, "A"), (node2, "B"), (node3, "C")] {
            AddNode {
                id,
                ts_millis: TimestampMilli::now(),
                name: name.to_string(),
                valid_range: None,
                summary: NodeSummary::from_text(&format!("Node {}", name)),
            }
            .run(&writer)
            .await
            .unwrap();
        }

        // Insert edges
        AddEdge {
            source_node_id: node1,
            target_node_id: node2,
            ts_millis: TimestampMilli::now(),
            name: "connects".to_string(),
            summary: schema::EdgeSummary::from_text("A connects to B"),
            weight: Some(1.0),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        AddEdge {
            source_node_id: node2,
            target_node_id: node3,
            ts_millis: TimestampMilli::now(),
            name: "links".to_string(),
            summary: schema::EdgeSummary::from_text("B links to C"),
            weight: Some(0.5),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;
        drop(writer);
        mutation_consumer_handle.await.unwrap().unwrap();

        // Now open storage for reading
        let mut storage = Storage::readonly(&db_path);
        storage.ready().unwrap();
        let graph = Graph::new(Arc::new(storage));

        // Create reader and query consumer
        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };
        let (reader, receiver) = create_query_reader(reader_config.clone());
        let consumer = Consumer::new(receiver, reader_config, graph);
        let consumer_handle = spawn_consumer(consumer);

        // Query all edges
        let results = AllEdges::new(100)
            .run(&reader, Duration::from_secs(5))
            .await
            .unwrap();

        // Should return 2 edges
        assert_eq!(results.len(), 2, "Should return 2 edges");

        // Verify edge data
        let edge_names: Vec<_> = results.iter().map(|(_, _, _, name)| name.as_str()).collect();
        assert!(edge_names.contains(&"connects"), "Should contain 'connects' edge");
        assert!(edge_names.contains(&"links"), "Should contain 'links' edge");

        // Verify weights are present
        for (weight, _, _, _) in &results {
            assert!(weight.is_some(), "Edge weight should be present");
        }

        // Cleanup
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_all_nodes_empty_database() {
        // Create an empty database
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Create writer but don't insert any data
        let writer_config = WriterConfig {
            channel_buffer_size: 10,
        };
        let (writer, mutation_receiver) = create_mutation_writer(writer_config.clone());
        let mutation_consumer_handle =
            spawn_mutation_consumer(mutation_receiver, writer_config, &db_path);

        // Need to insert at least one item to initialize the DB properly
        // then we'll test a different scenario
        let node_args = AddNode {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "temp".to_string(),
            valid_range: None,
            summary: NodeSummary::from_text("temp"),
        };
        node_args.run(&writer).await.unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;
        drop(writer);
        mutation_consumer_handle.await.unwrap().unwrap();

        // Now open storage for reading
        let mut storage = Storage::readonly(&db_path);
        storage.ready().unwrap();
        let graph = Graph::new(Arc::new(storage));

        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };
        let (reader, receiver) = create_query_reader(reader_config.clone());
        let consumer = Consumer::new(receiver, reader_config, graph);
        let consumer_handle = spawn_consumer(consumer);

        // Query with limit 0 should return empty
        let results = AllNodes::new(0)
            .run(&reader, Duration::from_secs(5))
            .await
            .unwrap();

        assert_eq!(results.len(), 0, "Limit 0 should return empty results");

        // Cleanup
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }
}
