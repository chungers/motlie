use anyhow::Result;
use std::ops::Bound;
use std::time::Duration;
use tokio::sync::oneshot;

use crate::graph::ColumnFamilyRecord;
use crate::schema::{
    self, DstId, EdgeName, EdgeSummary, FragmentContent, NodeName, NodeSummary, SrcId,
};
use crate::{Id, ReaderConfig, Storage, TimestampMilli};

/// Trait that all query types implement to execute themselves.
///
/// This trait defines HOW to fetch the result from storage.
/// Each query type knows how to execute its own data fetching logic.
///
/// # Design Philosophy
///
/// This follows the same pattern as mutations:
/// - Mutations: Each mutation type implements `MutationPlanner::plan()` to generate storage operations
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

/// Blanket implementation of QueryWithTimeout for types that implement QueryExecutor
/// This is the new way - query types execute themselves
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

/// Trait for queries that produce results with timeout handling
/// Note: This trait is now automatically implemented for all QueryExecutor types
#[async_trait::async_trait]
pub trait QueryWithTimeout: Send + Sync {
    /// The type of result this query produces
    type ResultType: Send;

    /// Execute the query with timeout and return the result
    async fn result<P: Processor>(&self, processor: &P) -> Result<Self::ResultType>;

    /// Get the timeout for this query
    fn timeout(&self) -> Duration;
}

/// Trait for processing queries without needing to know the result type
#[async_trait::async_trait]
pub trait QueryProcessor: Send {
    /// Process the query and send the result (consumes self)
    async fn process_and_send<P: Processor>(self, processor: &P);
}

/// Query enum representing all possible query types
#[derive(Debug)]
pub enum Query {
    NodeById(NodeById),
    EdgeSummaryBySrcDstName(EdgeSummaryBySrcDstName),
    FragmentsByIdTimeRange(NodeFragmentsByIdTimeRange),
    OutgoingEdges(OutgoingEdges),
    IncomingEdges(IncomingEdges),
    NodesByName(NodesByName),
    EdgesByName(EdgesByName),
}

impl std::fmt::Display for Query {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Query::NodeById(q) => write!(f, "NodeById: id={}", q.id),
            Query::EdgeSummaryBySrcDstName(q) => write!(
                f,
                "EdgeBySrcDstName: source={}, dest={}, name={}",
                q.source_id, q.dest_id, q.name
            ),
            Query::FragmentsByIdTimeRange(q) => {
                write!(
                    f,
                    "NodeFragmentsByIdTimeRange: id={}, range={:?}",
                    q.id, q.time_range
                )
            }
            Query::OutgoingEdges(q) => write!(f, "OutgoingEdges: id={}", q.id),
            Query::IncomingEdges(q) => write!(f, "IncomingEdges: id={}", q.id),
            Query::NodesByName(q) => write!(f, "NodesByName: name={}", q.name),
            Query::EdgesByName(q) => write!(f, "EdgesByName: name={}", q.name),
        }
    }
}

/// Macro to implement QueryProcessor for query types
macro_rules! impl_query_processor {
    ($($query_type:ty),+ $(,)?) => {
        $(
            #[async_trait::async_trait]
            impl QueryProcessor for $query_type {
                async fn process_and_send<P: Processor>(self, processor: &P) {
                    let result = self.result(processor).await;
                    self.send_result(result);
                }
            }
        )+
    };
}

impl_query_processor!(
    NodeById,
    EdgeSummaryBySrcDstName,
    NodeFragmentsByIdTimeRange,
    OutgoingEdges,
    IncomingEdges,
    NodesByName,
    EdgesByName,
);

#[async_trait::async_trait]
impl QueryProcessor for Query {
    async fn process_and_send<P: Processor>(self, processor: &P) {
        match self {
            Query::NodeById(q) => q.process_and_send(processor).await,
            Query::EdgeSummaryBySrcDstName(q) => q.process_and_send(processor).await,
            Query::FragmentsByIdTimeRange(q) => q.process_and_send(processor).await,
            Query::OutgoingEdges(q) => q.process_and_send(processor).await,
            Query::IncomingEdges(q) => q.process_and_send(processor).await,
            Query::NodesByName(q) => q.process_and_send(processor).await,
            Query::EdgesByName(q) => q.process_and_send(processor).await,
        }
    }
}

/// Trait for query builders that can be executed
#[async_trait::async_trait]
pub trait Runnable {
    /// The output type this query produces
    type Output: Send + 'static;

    /// Execute this query against a Reader with the specified timeout
    async fn run(self, reader: &crate::Reader, timeout: Duration) -> Result<Self::Output>;
}

/// Type alias for querying node by ID (returns name and summary)
pub type NodeById = ByIdQuery<(NodeName, NodeSummary)>;

/// Generic query to find an entity by its ID
#[derive(Debug)]
pub struct ByIdQuery<T: Send + Sync + 'static> {
    /// The entity ID to search for
    pub id: Id,

    /// Reference timestamp for temporal validity checks
    /// If None, defaults to current time in the query executor
    /// Temporal validity is always checked against the ValidTemporalRange in the record
    /// Records without a ValidTemporalRange (None) are considered always valid
    pub reference_ts_millis: Option<TimestampMilli>,

    /// Timeout for this query execution (only set when query has channel)
    pub(crate) timeout: Option<Duration>,

    /// Channel to send the result back to the client (only set when ready to execute)
    result_tx: Option<oneshot::Sender<Result<T>>>,
}

/// Query to scan node fragments by ID with time range filtering
/// This is different from ByIdQuery because it requires time range filtering
/// and returns multiple results (scan operation) rather than a single entity lookup.
#[derive(Debug)]
pub struct NodeFragmentsByIdTimeRange {
    /// The entity ID to search for
    pub id: Id,

    /// Time range bounds (start_bound, end_bound)
    /// Use std::ops::Bound for idiomatic Rust range specification
    pub time_range: (Bound<TimestampMilli>, Bound<TimestampMilli>),

    /// Reference timestamp for temporal validity checks
    /// If None, defaults to current time in the query executor
    /// Temporal validity is always checked against the ValidTemporalRange in the record
    /// Records without a ValidTemporalRange (None) are considered always valid
    pub reference_ts_millis: Option<TimestampMilli>,

    /// Timeout for this query execution (only set when query has channel)
    pub(crate) timeout: Option<Duration>,

    /// Channel to send the result back to the client (only set when ready to execute)
    result_tx: Option<oneshot::Sender<Result<Vec<(TimestampMilli, FragmentContent)>>>>,
}

/// Query to find an edge by source ID, destination ID, and name
#[derive(Debug)]
pub struct EdgeSummaryBySrcDstName {
    /// Source node ID
    pub source_id: Id,

    /// Destination node ID
    pub dest_id: Id,

    /// Edge name
    pub name: String,

    /// Reference timestamp for temporal validity checks
    /// If None, defaults to current time in the query executor
    /// Temporal validity is always checked against the ValidTemporalRange in the record
    /// Records without a ValidTemporalRange (None) are considered always valid
    pub reference_ts_millis: Option<TimestampMilli>,

    /// Timeout for this query execution (only set when query has channel)
    pub(crate) timeout: Option<Duration>,

    /// Channel to send the result back to the client (only set when ready to execute)
    result_tx: Option<oneshot::Sender<Result<(EdgeSummary, Option<f64>)>>>,
}

/// Query to find all outgoing edges from a node by its ID
#[derive(Debug)]
pub struct OutgoingEdges {
    /// The node ID to search for
    pub id: Id,

    /// Reference timestamp for temporal validity checks
    /// If None, defaults to current time in the query executor
    /// Temporal validity is always checked against the ValidTemporalRange in the record
    /// Records without a ValidTemporalRange (None) are considered always valid
    pub reference_ts_millis: Option<TimestampMilli>,

    /// Timeout for this query execution (only set when query has channel)
    pub(crate) timeout: Option<Duration>,

    /// Channel to send the result back to the client (only set when ready to execute)
    result_tx: Option<oneshot::Sender<Result<Vec<(Option<f64>, SrcId, DstId, EdgeName)>>>>,
}

/// Query to find all incoming edges to a node by its ID
#[derive(Debug)]
pub struct IncomingEdges {
    /// The node ID to search for
    pub id: DstId,

    /// Reference timestamp for temporal validity checks
    /// If None, defaults to current time in the query executor
    /// Temporal validity is always checked against the ValidTemporalRange in the record
    /// Records without a ValidTemporalRange (None) are considered always valid
    pub reference_ts_millis: Option<TimestampMilli>,

    /// Timeout for this query execution (only set when query has channel)
    pub(crate) timeout: Option<Duration>,

    /// Channel to send the result back to the client (only set when ready to execute)
    result_tx: Option<oneshot::Sender<Result<Vec<(Option<f64>, DstId, SrcId, EdgeName)>>>>,
}

/// Query to find nodes by name prefix
#[derive(Debug)]
pub struct NodesByName {
    /// The node name or prefix to search for
    pub name: NodeName,

    /// Optional start position for pagination (exclusive): (last_name, last_id)
    /// For prefix queries, this should be the full name and ID of the last result from the previous page
    pub start: Option<(NodeName, Id)>,

    /// Maximum number of results to return
    pub limit: Option<usize>,

    /// Reference timestamp for temporal validity checks
    /// If None, defaults to current time in the query executor
    /// Temporal validity is always checked against the ValidTemporalRange in the record
    /// Records without a ValidTemporalRange (None) are considered always valid
    pub reference_ts_millis: Option<TimestampMilli>,

    /// Timeout for this query execution (only set when query has channel)
    pub(crate) timeout: Option<Duration>,

    /// Channel to send the result back to the client (only set when ready to execute)
    result_tx: Option<oneshot::Sender<Result<Vec<(NodeName, Id)>>>>,
}

/// Query to find edges by name prefix
#[derive(Debug)]
pub struct EdgesByName {
    /// The edge name or prefix to search for
    pub name: String,

    /// Optional start position for pagination (exclusive): (last_name, last_id)
    /// For prefix queries, this should be the full name and ID of the last result from the previous page
    pub start: Option<(EdgeName, Id)>,

    /// The maximum number of results to return
    pub limit: Option<usize>,

    /// Reference timestamp for temporal validity checks
    /// If None, defaults to current time in the query executor
    /// Temporal validity is always checked against the ValidTemporalRange in the record
    /// Records without a ValidTemporalRange (None) are considered always valid
    pub reference_ts_millis: Option<TimestampMilli>,

    /// Timeout for this query execution (only set when query has channel)
    pub(crate) timeout: Option<Duration>,

    /// Channel to send the result back to the client (only set when ready to execute)
    result_tx: Option<oneshot::Sender<Result<Vec<(EdgeName, Id)>>>>,
}

impl<T: Send + Sync + 'static> ByIdQuery<T> {
    /// Create a new query request (public API - no channel, no timeout yet)
    /// Use `.run(reader, timeout)` to execute this query
    pub fn new(id: Id, reference_ts_millis: Option<TimestampMilli>) -> Self {
        Self {
            id,
            reference_ts_millis,
            timeout: None,
            result_tx: None,
        }
    }

    /// Internal constructor used by the query execution machinery (has the channel)
    pub(crate) fn with_channel(
        id: Id,
        reference_ts_millis: Option<TimestampMilli>,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<T>>,
    ) -> Self {
        Self {
            id,
            reference_ts_millis,
            timeout: Some(timeout),
            result_tx: Some(result_tx),
        }
    }

    /// Send the result back to the client (consumes self)
    pub(crate) fn send_result(self, result: Result<T>) {
        // Ignore error if receiver was dropped (client timeout/cancellation)
        if let Some(tx) = self.result_tx {
            let _ = tx.send(result);
        }
    }
}

impl NodeFragmentsByIdTimeRange {
    /// Create a new query request (public API - no channel, no timeout yet)
    /// Use `.run(reader, timeout)` to execute this query
    pub fn new(
        id: Id,
        time_range: (Bound<TimestampMilli>, Bound<TimestampMilli>),
        reference_ts_millis: Option<TimestampMilli>,
    ) -> Self {
        Self {
            id,
            time_range,
            reference_ts_millis,
            timeout: None,
            result_tx: None,
        }
    }

    /// Internal constructor used by the query execution machinery (has the channel)
    pub(crate) fn with_channel(
        id: Id,
        time_range: (Bound<TimestampMilli>, Bound<TimestampMilli>),
        reference_ts_millis: Option<TimestampMilli>,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<Vec<(TimestampMilli, FragmentContent)>>>,
    ) -> Self {
        Self {
            id,
            time_range,
            reference_ts_millis,
            timeout: Some(timeout),
            result_tx: Some(result_tx),
        }
    }

    /// Send the result back to the client (consumes self)
    pub(crate) fn send_result(self, result: Result<Vec<(TimestampMilli, FragmentContent)>>) {
        // Ignore error if receiver was dropped (client timeout/cancellation)
        if let Some(tx) = self.result_tx {
            let _ = tx.send(result);
        }
    }

    /// Check if a timestamp falls within this range
    pub fn contains(&self, ts: TimestampMilli) -> bool {
        let (start_bound, end_bound) = &self.time_range;

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
}

impl EdgeSummaryBySrcDstName {
    /// Create a new query request (public API - no channel, no timeout yet)
    /// Use `.run(reader, timeout)` to execute this query
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
            timeout: None,
            result_tx: None,
        }
    }

    /// Internal constructor used by the query execution machinery (has the channel)
    pub(crate) fn with_channel(
        source_id: SrcId,
        dest_id: DstId,
        name: String,
        reference_ts_millis: Option<TimestampMilli>,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<(EdgeSummary, Option<f64>)>>,
    ) -> Self {
        Self {
            source_id,
            dest_id,
            name,
            reference_ts_millis,
            timeout: Some(timeout),
            result_tx: Some(result_tx),
        }
    }

    pub(crate) fn send_result(self, result: Result<(EdgeSummary, Option<f64>)>) {
        if let Some(tx) = self.result_tx {
            let _ = tx.send(result);
        }
    }
}

impl OutgoingEdges {
    /// Create a new query request (public API - no channel, no timeout yet)
    pub fn new(id: Id, reference_ts_millis: Option<TimestampMilli>) -> Self {
        Self {
            id,
            reference_ts_millis,
            timeout: None,
            result_tx: None,
        }
    }

    /// Internal constructor used by the query execution machinery
    pub(crate) fn with_channel(
        id: Id,
        reference_ts_millis: Option<TimestampMilli>,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<Vec<(Option<f64>, SrcId, DstId, EdgeName)>>>,
    ) -> Self {
        Self {
            id,
            reference_ts_millis,
            timeout: Some(timeout),
            result_tx: Some(result_tx),
        }
    }

    pub(crate) fn send_result(self, result: Result<Vec<(Option<f64>, SrcId, DstId, EdgeName)>>) {
        if let Some(tx) = self.result_tx {
            let _ = tx.send(result);
        }
    }
}

impl IncomingEdges {
    /// Create a new query request (public API - no channel, no timeout yet)
    pub fn new(id: DstId, reference_ts_millis: Option<TimestampMilli>) -> Self {
        Self {
            id,
            reference_ts_millis,
            timeout: None,
            result_tx: None,
        }
    }

    /// Internal constructor used by the query execution machinery
    pub(crate) fn with_channel(
        id: DstId,
        reference_ts_millis: Option<TimestampMilli>,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<Vec<(Option<f64>, DstId, SrcId, EdgeName)>>>,
    ) -> Self {
        Self {
            id,
            reference_ts_millis,
            timeout: Some(timeout),
            result_tx: Some(result_tx),
        }
    }

    pub(crate) fn send_result(self, result: Result<Vec<(Option<f64>, DstId, SrcId, EdgeName)>>) {
        if let Some(tx) = self.result_tx {
            let _ = tx.send(result);
        }
    }
}

impl NodesByName {
    /// Create a new query request (public API - no channel, no timeout yet)
    pub fn new(
        name: NodeName,
        start: Option<(NodeName, Id)>,
        limit: Option<usize>,
        reference_ts_millis: Option<TimestampMilli>,
    ) -> Self {
        Self {
            name,
            start,
            limit,
            reference_ts_millis,
            timeout: None,
            result_tx: None,
        }
    }

    /// Internal constructor used by the query execution machinery
    pub(crate) fn with_channel(
        name: NodeName,
        start: Option<(NodeName, Id)>,
        limit: Option<usize>,
        reference_ts_millis: Option<TimestampMilli>,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<Vec<(NodeName, Id)>>>,
    ) -> Self {
        Self {
            name,
            start,
            limit,
            reference_ts_millis,
            timeout: Some(timeout),
            result_tx: Some(result_tx),
        }
    }

    pub(crate) fn send_result(self, result: Result<Vec<(NodeName, Id)>>) {
        if let Some(tx) = self.result_tx {
            let _ = tx.send(result);
        }
    }
}

impl EdgesByName {
    /// Create a new query request (public API - no channel, no timeout yet)
    pub fn new(
        name: String,
        start: Option<(EdgeName, Id)>,
        limit: Option<usize>,
        reference_ts_millis: Option<TimestampMilli>,
    ) -> Self {
        Self {
            name,
            start,
            limit,
            reference_ts_millis,
            timeout: None,
            result_tx: None,
        }
    }

    /// Internal constructor used by the query execution machinery
    pub(crate) fn with_channel(
        name: String,
        start: Option<(EdgeName, Id)>,
        limit: Option<usize>,
        reference_ts_millis: Option<TimestampMilli>,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<Vec<(EdgeName, Id)>>>,
    ) -> Self {
        Self {
            name,
            start,
            limit,
            reference_ts_millis,
            timeout: Some(timeout),
            result_tx: Some(result_tx),
        }
    }

    pub(crate) fn send_result(self, result: Result<Vec<(EdgeName, Id)>>) {
        if let Some(tx) = self.result_tx {
            let _ = tx.send(result);
        }
    }
}

/// Implement Runnable for NodeByIdQuery specifically
#[async_trait::async_trait]
impl Runnable for NodeById {
    type Output = (NodeName, NodeSummary);

    async fn run(self, reader: &crate::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();

        let query = NodeById::with_channel(self.id, self.reference_ts_millis, timeout, result_tx);

        reader.send_query(Query::NodeById(query)).await?;
        result_rx.await?
    }
}

/// Implement Runnable for NodeFragmentsByIdTimeRangeQuery
#[async_trait::async_trait]
impl Runnable for NodeFragmentsByIdTimeRange {
    type Output = Vec<(TimestampMilli, FragmentContent)>;

    async fn run(self, reader: &crate::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();

        let query = NodeFragmentsByIdTimeRange::with_channel(
            self.id,
            self.time_range,
            self.reference_ts_millis,
            timeout,
            result_tx,
        );

        reader
            .send_query(Query::FragmentsByIdTimeRange(query))
            .await?;
        result_rx.await?
    }
}

/// Implement Runnable for EdgeSummaryBySrcDstNameQuery
#[async_trait::async_trait]
impl Runnable for EdgeSummaryBySrcDstName {
    type Output = (EdgeSummary, Option<f64>);

    async fn run(self, reader: &crate::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();

        let query = EdgeSummaryBySrcDstName::with_channel(
            self.source_id,
            self.dest_id,
            self.name,
            self.reference_ts_millis,
            timeout,
            result_tx,
        );

        reader
            .send_query(Query::EdgeSummaryBySrcDstName(query))
            .await?;
        result_rx.await?
    }
}

#[async_trait::async_trait]
impl Runnable for OutgoingEdges {
    type Output = Vec<(Option<f64>, SrcId, DstId, EdgeName)>;

    async fn run(self, reader: &crate::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let query =
            OutgoingEdges::with_channel(self.id, self.reference_ts_millis, timeout, result_tx);
        reader.send_query(Query::OutgoingEdges(query)).await?;
        result_rx.await?
    }
}

#[async_trait::async_trait]
impl Runnable for IncomingEdges {
    type Output = Vec<(Option<f64>, DstId, SrcId, EdgeName)>;

    async fn run(self, reader: &crate::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let query =
            IncomingEdges::with_channel(self.id, self.reference_ts_millis, timeout, result_tx);
        reader.send_query(Query::IncomingEdges(query)).await?;
        result_rx.await?
    }
}

#[async_trait::async_trait]
impl Runnable for NodesByName {
    type Output = Vec<(NodeName, Id)>;

    async fn run(self, reader: &crate::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let query = NodesByName::with_channel(
            self.name,
            self.start,
            self.limit,
            self.reference_ts_millis,
            timeout,
            result_tx,
        );
        reader.send_query(Query::NodesByName(query)).await?;
        result_rx.await?
    }
}

#[async_trait::async_trait]
impl Runnable for EdgesByName {
    type Output = Vec<(EdgeName, Id)>;

    async fn run(self, reader: &crate::Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let query = EdgesByName::with_channel(
            self.name,
            self.start,
            self.limit,
            self.reference_ts_millis,
            timeout,
            result_tx,
        );
        reader.send_query(Query::EdgesByName(query)).await?;
        result_rx.await?
    }
}

/// Implement QueryExecutor for NodeByIdQuery
#[async_trait::async_trait]
impl QueryExecutor for NodeById {
    type Output = (NodeName, NodeSummary);

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        // Default None to current time for temporal validity checks
        let ref_time = self
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let id = self.id;

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

        Ok((value.1, value.2))
    }

    fn timeout(&self) -> Duration {
        self.timeout
            .expect("Query must have timeout set when executing")
    }
}

/// Implement QueryExecutor for NodeFragmentsByIdTimeRangeQuery
#[async_trait::async_trait]
impl QueryExecutor for NodeFragmentsByIdTimeRange {
    type Output = Vec<(TimestampMilli, FragmentContent)>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        use std::ops::Bound;

        // Default None to current time for temporal validity checks
        let ref_time = self
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let id = self.id;
        let mut fragments: Vec<(TimestampMilli, FragmentContent)> = Vec::new();

        // Construct optimal starting key based on start bound
        let start_key = match &self.time_range.0 {
            Bound::Unbounded => {
                let mut key = Vec::with_capacity(24);
                key.extend_from_slice(&id.into_bytes());
                key.extend_from_slice(&0u64.to_be_bytes());
                key
            }
            Bound::Included(start_ts) => {
                schema::NodeFragments::key_to_bytes(&schema::NodeFragmentCfKey(id, *start_ts))
            }
            Bound::Excluded(start_ts) => schema::NodeFragments::key_to_bytes(&schema::NodeFragmentCfKey(
                id,
                TimestampMilli(start_ts.0 + 1),
            )),
        };

        macro_rules! process_items {
            ($iter:expr) => {{
                for item in $iter {
                    let (key_bytes, value_bytes) = item?;
                    let key: schema::NodeFragmentCfKey = schema::NodeFragments::key_from_bytes(&key_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

                    if key.0 != id {
                        break;
                    }

                    let timestamp = key.1;

                    match &self.time_range.1 {
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
                }
            }};
        }

        if let Ok(db) = storage.db() {
            let cf = db.cf_handle(schema::NodeFragments::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!("Column family '{}' not found", schema::NodeFragments::CF_NAME)
            })?;

            let iter = db.iterator_cf(
                cf,
                rocksdb::IteratorMode::From(&start_key, rocksdb::Direction::Forward),
            );
            process_items!(iter);
        } else {
            let txn_db = storage.transaction_db()?;
            let cf = txn_db
                .cf_handle(schema::NodeFragments::CF_NAME)
                .ok_or_else(|| {
                    anyhow::anyhow!("Column family '{}' not found", schema::NodeFragments::CF_NAME)
                })?;

            let iter = txn_db.iterator_cf(
                cf,
                rocksdb::IteratorMode::From(&start_key, rocksdb::Direction::Forward),
            );
            process_items!(iter);
        }

        Ok(fragments)
    }

    fn timeout(&self) -> Duration {
        self.timeout
            .expect("Query must have timeout set when executing")
    }
}

/// Implement QueryExecutor for EdgeSummaryBySrcDstNameQuery
#[async_trait::async_trait]
impl QueryExecutor for EdgeSummaryBySrcDstName {
    type Output = (EdgeSummary, Option<f64>);

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        // Default None to current time for temporal validity checks
        let ref_time = self
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let source_id = self.source_id;
        let dest_id = self.dest_id;
        let name = &self.name;

        let key = schema::ForwardEdgeCfKey(source_id, dest_id, name.clone());
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
            .expect("Query must have timeout set when executing")
    }
}

/// Implement QueryExecutor for OutgoingEdgesQuery
#[async_trait::async_trait]
impl QueryExecutor for OutgoingEdges {
    type Output = Vec<(Option<f64>, SrcId, DstId, EdgeName)>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        // Default None to current time for temporal validity checks
        let ref_time = self
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let id = self.id;
        let mut edges: Vec<(Option<f64>, SrcId, DstId, EdgeName)> = Vec::new();
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
                let edge_name = key.2;
                let weight = value.1;
                edges.push((weight, source_id, dest_id, edge_name));
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
                let edge_name = key.2;
                let weight = value.1;
                edges.push((weight, source_id, dest_id, edge_name));
            }
        }

        Ok(edges)
    }

    fn timeout(&self) -> Duration {
        self.timeout
            .expect("Query must have timeout set when executing")
    }
}

/// Implement QueryExecutor for IncomingEdgesQuery
#[async_trait::async_trait]
impl QueryExecutor for IncomingEdges {
    type Output = Vec<(Option<f64>, DstId, SrcId, EdgeName)>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        // Default None to current time for temporal validity checks
        let ref_time = self
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let id = self.id;
        let mut edges: Vec<(Option<f64>, DstId, SrcId, EdgeName)> = Vec::new();
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
                let edge_name = key.2.clone();

                // Lookup weight from ForwardEdges CF
                // ReverseEdgeCfKey is (dst_id, src_id, edge_name)
                // ForwardEdgeCfKey is (src_id, dst_id, edge_name)
                let forward_key = schema::ForwardEdgeCfKey(source_id, dest_id, edge_name.clone());
                let forward_key_bytes = schema::ForwardEdges::key_to_bytes(&forward_key);

                let weight = if let Some(forward_value_bytes) = db.get_cf(forward_cf, forward_key_bytes)? {
                    let forward_value: schema::ForwardEdgeCfValue =
                        schema::ForwardEdges::value_from_bytes(&forward_value_bytes)
                            .map_err(|e| anyhow::anyhow!("Failed to deserialize forward edge value: {}", e))?;
                    forward_value.1 // Extract weight from field 1
                } else {
                    None
                };

                edges.push((weight, dest_id, source_id, edge_name));
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
                let edge_name = key.2.clone();

                // Lookup weight from ForwardEdges CF
                // ReverseEdgeCfKey is (dst_id, src_id, edge_name)
                // ForwardEdgeCfKey is (src_id, dst_id, edge_name)
                let forward_key = schema::ForwardEdgeCfKey(source_id, dest_id, edge_name.clone());
                let forward_key_bytes = schema::ForwardEdges::key_to_bytes(&forward_key);

                let weight = if let Some(forward_value_bytes) = txn_db.get_cf(forward_cf, forward_key_bytes)? {
                    let forward_value: schema::ForwardEdgeCfValue =
                        schema::ForwardEdges::value_from_bytes(&forward_value_bytes)
                            .map_err(|e| anyhow::anyhow!("Failed to deserialize forward edge value: {}", e))?;
                    forward_value.1 // Extract weight from field 1
                } else {
                    None
                };

                edges.push((weight, dest_id, source_id, edge_name));
            }
        }

        Ok(edges)
    }

    fn timeout(&self) -> Duration {
        self.timeout
            .expect("Query must have timeout set when executing")
    }
}

/// Implement QueryExecutor for NodesByNameQuery
#[async_trait::async_trait]
impl QueryExecutor for NodesByName {
    type Output = Vec<(NodeName, Id)>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        // Default None to current time for temporal validity checks
        let ref_time = self
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let name = &self.name;
        let mut nodes: Vec<(NodeName, Id)> = Vec::new();

        let seek_key = if let Some((start_name, start_id)) = &self.start {
            let mut bytes = Vec::with_capacity(start_name.len() + 16);
            bytes.extend_from_slice(start_name.as_bytes());
            bytes.extend_from_slice(&start_id.into_bytes());
            bytes
        } else {
            name.as_bytes().to_vec()
        };

        if let Ok(db) = storage.db() {
            let cf = db.cf_handle(schema::NodeNames::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!("Column family '{}' not found", schema::NodeNames::CF_NAME)
            })?;

            let iter = db.iterator_cf(
                cf,
                rocksdb::IteratorMode::From(&seek_key, rocksdb::Direction::Forward),
            );

            for item in iter {
                if let Some(limit) = self.limit {
                    if nodes.len() >= limit {
                        break;
                    }
                }

                let (key_bytes, value_bytes) = item?;
                let key: schema::NodeNameCfKey = schema::NodeNames::key_from_bytes(&key_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

                let node_name = key.0;
                let node_id = key.1;

                if !node_name.starts_with(name) {
                    break;
                }

                if let Some((start_name, start_id)) = &self.start {
                    if node_name == *start_name && node_id == *start_id {
                        continue;
                    }
                }

                // Deserialize and check temporal validity
                let value: schema::NodeNameCfValue =
                    schema::NodeNames::value_from_bytes(&value_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

                // Always check temporal validity - skip invalid node names
                if !schema::is_valid_at_time(&value.0, ref_time) {
                    continue;
                }

                nodes.push((node_name, node_id));
            }
        } else {
            let txn_db = storage.transaction_db()?;
            let cf = txn_db
                .cf_handle(schema::NodeNames::CF_NAME)
                .ok_or_else(|| {
                    anyhow::anyhow!("Column family '{}' not found", schema::NodeNames::CF_NAME)
                })?;

            let iter = txn_db.iterator_cf(
                cf,
                rocksdb::IteratorMode::From(&seek_key, rocksdb::Direction::Forward),
            );

            for item in iter {
                if let Some(limit) = self.limit {
                    if nodes.len() >= limit {
                        break;
                    }
                }

                let (key_bytes, value_bytes) = item?;
                let key: schema::NodeNameCfKey = schema::NodeNames::key_from_bytes(&key_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

                let node_name = key.0;
                let node_id = key.1;

                if !node_name.starts_with(name) {
                    break;
                }

                if let Some((start_name, start_id)) = &self.start {
                    if node_name == *start_name && node_id == *start_id {
                        continue;
                    }
                }

                // Deserialize and check temporal validity
                let value: schema::NodeNameCfValue =
                    schema::NodeNames::value_from_bytes(&value_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

                // Always check temporal validity - skip invalid node names
                if !schema::is_valid_at_time(&value.0, ref_time) {
                    continue;
                }

                nodes.push((node_name, node_id));
            }
        }

        Ok(nodes)
    }

    fn timeout(&self) -> Duration {
        self.timeout
            .expect("Query must have timeout set when executing")
    }
}

/// Implement QueryExecutor for EdgesByNameQuery
#[async_trait::async_trait]
impl QueryExecutor for EdgesByName {
    type Output = Vec<(EdgeName, Id)>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        // Default None to current time for temporal validity checks
        let ref_time = self
            .reference_ts_millis
            .unwrap_or_else(|| TimestampMilli::now());

        let name = &self.name;
        let mut edges: Vec<(EdgeName, Id)> = Vec::new();

        let seek_key = if let Some((start_name, start_id)) = &self.start {
            let mut bytes = Vec::with_capacity(start_name.len() + 16);
            bytes.extend_from_slice(start_name.as_bytes());
            bytes.extend_from_slice(&start_id.into_bytes());
            bytes
        } else {
            name.as_bytes().to_vec()
        };

        if let Ok(db) = storage.db() {
            let cf = db.cf_handle(schema::EdgeNames::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!("Column family '{}' not found", schema::EdgeNames::CF_NAME)
            })?;

            let iter = db.iterator_cf(
                cf,
                rocksdb::IteratorMode::From(&seek_key, rocksdb::Direction::Forward),
            );

            for item in iter {
                if let Some(limit) = self.limit {
                    if edges.len() >= limit {
                        break;
                    }
                }

                let (key_bytes, value_bytes) = item?;
                let key: schema::EdgeNameCfKey = schema::EdgeNames::key_from_bytes(&key_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

                let edge_name = &key.0;
                let edge_id = key.1;

                if !edge_name.starts_with(name) {
                    break;
                }

                if let Some((start_name, start_id)) = &self.start {
                    if edge_name == start_name && edge_id == *start_id {
                        continue;
                    }
                }

                // Deserialize and check temporal validity
                let value: schema::EdgeNameCfValue =
                    schema::EdgeNames::value_from_bytes(&value_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

                // Always check temporal validity - skip invalid edge names
                if !schema::is_valid_at_time(&value.0, ref_time) {
                    continue;
                }

                edges.push((edge_name.clone(), edge_id));
            }
        } else {
            let txn_db = storage.transaction_db()?;
            let cf = txn_db
                .cf_handle(schema::EdgeNames::CF_NAME)
                .ok_or_else(|| {
                    anyhow::anyhow!("Column family '{}' not found", schema::EdgeNames::CF_NAME)
                })?;

            let iter = txn_db.iterator_cf(
                cf,
                rocksdb::IteratorMode::From(&seek_key, rocksdb::Direction::Forward),
            );

            for item in iter {
                if let Some(limit) = self.limit {
                    if edges.len() >= limit {
                        break;
                    }
                }

                let (key_bytes, value_bytes) = item?;
                let key: schema::EdgeNameCfKey = schema::EdgeNames::key_from_bytes(&key_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

                let edge_name = &key.0;
                let edge_id = key.1;

                if !edge_name.starts_with(name) {
                    break;
                }

                if let Some((start_name, start_id)) = &self.start {
                    if edge_name == start_name && edge_id == *start_id {
                        continue;
                    }
                }

                // Deserialize and check temporal validity
                let value: schema::EdgeNameCfValue =
                    schema::EdgeNames::value_from_bytes(&value_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

                // Always check temporal validity - skip invalid edge names
                if !schema::is_valid_at_time(&value.0, ref_time) {
                    continue;
                }

                edges.push((edge_name.clone(), edge_id));
            }
        }

        Ok(edges)
    }

    fn timeout(&self) -> Duration {
        self.timeout
            .expect("Query must have timeout set when executing")
    }
}

/// Trait for processing different types of queries
///
/// This trait provides access to storage. Query types implement QueryExecutor
/// to execute themselves against storage, following the same pattern as mutations.
pub trait Processor: Send + Sync {
    /// Get access to the underlying storage
    /// Query types use this to execute themselves via QueryExecutor::execute()
    fn storage(&self) -> &Storage;
}

/// Generic consumer that processes queries using a Processor
pub struct Consumer<P: Processor> {
    receiver: flume::Receiver<Query>,
    config: ReaderConfig,
    processor: P,
}

impl<P: Processor> Consumer<P> {
    /// Create a new Consumer
    pub fn new(receiver: flume::Receiver<Query>, config: ReaderConfig, processor: P) -> Self {
        Self {
            receiver,
            config,
            processor,
        }
    }

    /// Process queries continuously until the channel is closed
    pub async fn run(self) -> Result<()> {
        log::info!("Starting query consumer with config: {:?}", self.config);

        loop {
            // Wait for the next query (MPMC semantics via flume)
            match self.receiver.recv_async().await {
                Ok(query) => {
                    // Process the query immediately
                    self.process_query(query).await;
                }
                Err(_) => {
                    // Channel closed
                    log::info!("Query consumer shutting down - channel closed");
                    return Ok(());
                }
            }
        }
    }

    /// Process a single query
    async fn process_query(&self, query: Query) {
        log::debug!("Processing {}", query);
        query.process_and_send(&self.processor).await;
    }
}

/// Spawn a query consumer as a background task
pub fn spawn_consumer<P: Processor + 'static>(
    consumer: Consumer<P>,
) -> tokio::task::JoinHandle<Result<()>> {
    tokio::spawn(async move { consumer.run().await })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{spawn_graph_consumer, Graph};
    use crate::mutation::AddNode;
    use crate::mutation::Runnable as MutRunnable;
    use crate::schema::NodeSummary;
    use crate::writer::{create_mutation_writer, WriterConfig};
    use crate::{DataUrl, Id, Storage, TimestampMilli};
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
            spawn_graph_consumer(mutation_receiver, writer_config, &db_path);

        // Insert a test node
        let node_id = Id::new();
        let node_name = "test_node".to_string();
        let node_args = AddNode {
            id: node_id,
            ts_millis: TimestampMilli::now(),
            name: node_name.clone(),
            temporal_range: None,
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
        let reader_config = crate::ReaderConfig {
            channel_buffer_size: 10,
        };

        let (reader, receiver) = {
            let (sender, receiver) = flume::bounded(reader_config.channel_buffer_size);
            let reader = crate::Reader::new(sender);
            (reader, receiver)
        };

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
        use crate::{AddEdge, AddNode, EdgeSummary, Id, MutationRunnable, TimestampMilli, WriterConfig};
        use tempfile::TempDir;

        // Create temporary database
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let config = WriterConfig {
            channel_buffer_size: 10,
        };

        // Create writer and spawn graph consumer for mutations
        let (writer, mutation_receiver) = crate::create_mutation_writer(config.clone());
        let mutation_consumer_handle = crate::spawn_graph_consumer(mutation_receiver, config, &db_path);

        // Create source and destination nodes
        let source_id = Id::new();
        let dest_id = Id::new();

        AddNode {
            id: source_id,
            ts_millis: TimestampMilli::now(),
            name: "source_node".to_string(),
            temporal_range: None,
        }
        .run(&writer)
        .await
        .unwrap();

        AddNode {
            id: dest_id,
            ts_millis: TimestampMilli::now(),
            name: "dest_node".to_string(),
            temporal_range: None,
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
            temporal_range: None,
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
        let reader_config = crate::ReaderConfig {
            channel_buffer_size: 10,
        };

        let (reader, receiver) = {
            let (sender, receiver) = flume::bounded(reader_config.channel_buffer_size);
            let reader = crate::Reader::new(sender);
            (reader, receiver)
        };

        let consumer = Consumer::new(receiver, reader_config, graph);

        // Spawn query consumer
        let consumer_handle = spawn_consumer(consumer);

        // Query the edge using EdgeSummaryBySrcDstName with Runnable pattern
        let (returned_summary, returned_weight) = EdgeSummaryBySrcDstName::new(
            source_id,
            dest_id,
            edge_name.to_string(),
            None,
        )
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
}
