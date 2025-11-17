use anyhow::Result;
use std::ops::Bound;
use std::time::Duration;
use tokio::sync::oneshot;

use crate::graph::ColumnFamilyRecord;
use crate::schema::{self, EdgeName, EdgeSummary, FragmentContent, NodeName, NodeSummary};
use crate::{Id, ReaderConfig, Storage, TimestampMilli};

/// Trait that all query types implement to execute themselves.
///
/// This trait defines HOW to fetch the result from storage.
/// Each query type knows how to execute its own data fetching logic.
///
/// # Design Philosophy
///
/// This follows the same pattern as mutations:
/// - Mutations: `schema::Plan::create_batch()` knows how to convert mutations
/// - Queries: Each query type knows how to execute itself
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
        let result = tokio::time::timeout(
            self.timeout(),
            self.execute(processor.storage())
        ).await;

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
    NodeById(NodeByIdQuery),
    EdgeById(EdgeByIdQuery),
    EdgeSummaryBySrcDstName(EdgeSummaryBySrcDstNameQuery),
    FragmentsByIdTimeRange(FragmentsByIdTimeRangeQuery),
    EdgesFromNode(EdgesFromNodeQuery),
    EdgesToNode(EdgesToNodeQuery),
    NodesByName(NodesByNameQuery),
    EdgesByName(EdgesByNameQuery),
}

impl std::fmt::Display for Query {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Query::NodeById(q) => write!(f, "NodeById: id={}", q.id),
            Query::EdgeById(q) => write!(f, "EdgeById: id={}", q.id),
            Query::EdgeSummaryBySrcDstName(q) => write!(
                f,
                "EdgeBySrcDstName: source={}, dest={}, name={}",
                q.source_id, q.dest_id, q.name
            ),
            Query::FragmentsByIdTimeRange(q) => {
                write!(
                    f,
                    "FragmentsByIdTimeRange: id={}, range={:?}",
                    q.id, q.time_range
                )
            }
            Query::EdgesFromNode(q) => write!(f, "EdgesFromNodeById: id={}", q.id),
            Query::EdgesToNode(q) => write!(f, "EdgesToNodeById: id={}", q.id),
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
    NodeByIdQuery,
    EdgeByIdQuery,
    EdgeSummaryBySrcDstNameQuery,
    FragmentsByIdTimeRangeQuery,
    EdgesFromNodeQuery,
    EdgesToNodeQuery,
    NodesByNameQuery,
    EdgesByNameQuery,
);

#[async_trait::async_trait]
impl QueryProcessor for Query {
    async fn process_and_send<P: Processor>(self, processor: &P) {
        match self {
            Query::NodeById(q) => q.process_and_send(processor).await,
            Query::EdgeById(q) => q.process_and_send(processor).await,
            Query::EdgeSummaryBySrcDstName(q) => q.process_and_send(processor).await,
            Query::FragmentsByIdTimeRange(q) => q.process_and_send(processor).await,
            Query::EdgesFromNode(q) => q.process_and_send(processor).await,
            Query::EdgesToNode(q) => q.process_and_send(processor).await,
            Query::NodesByName(q) => q.process_and_send(processor).await,
            Query::EdgesByName(q) => q.process_and_send(processor).await,
        }
    }
}

/// Type alias for querying node by ID (returns name and summary)
pub type NodeByIdQuery = ByIdQuery<(NodeName, NodeSummary)>;
/// Type alias for querying edge by ID (returns topology and summary)
pub type EdgeByIdQuery = ByIdQuery<(SrcId, DstId, EdgeName, EdgeSummary)>;
// Note: EdgesFromNodeByIdQuery and EdgesToNodeByIdQuery can't use ByIdQuery<T>
// because they have the same result type but different processing logic.
// They are implemented as separate structs below.

/// Generic query to find an entity by its ID
#[derive(Debug)]
pub struct ByIdQuery<T: Send + Sync + 'static> {
    /// The entity ID to search for
    pub id: Id,

    /// Timestamp of when the query was created
    pub ts_millis: TimestampMilli,

    /// Timeout for this query
    pub timeout: Duration,

    /// Channel to send the result back to the client
    result_tx: oneshot::Sender<Result<T>>,
}

impl<T: Send + Sync + 'static> ByIdQuery<T> {
    /// Create a new ByIdQuery
    pub fn new(id: Id, timeout: Duration, result_tx: oneshot::Sender<Result<T>>) -> Self {
        Self {
            id,
            ts_millis: TimestampMilli::now(),
            timeout,
            result_tx,
        }
    }

    /// Send the result back to the client (consumes self)
    pub fn send_result(self, result: Result<T>) {
        // Ignore error if receiver was dropped (client timeout/cancellation)
        let _ = self.result_tx.send(result);
    }
}

/// Implement QueryExecutor for NodeByIdQuery
#[async_trait::async_trait]
impl QueryExecutor for NodeByIdQuery {
    type Output = (NodeName, NodeSummary);

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
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

        Ok((value.0, value.1))
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

/// Implement QueryExecutor for EdgeByIdQuery
#[async_trait::async_trait]
impl QueryExecutor for EdgeByIdQuery {
    type Output = (SrcId, DstId, EdgeName, EdgeSummary);

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let id = self.id;
        let key = schema::EdgeCfKey(id);
        let key_bytes = schema::Edges::key_to_bytes(&key);

        // Handle both readonly and readwrite modes
        let value_bytes = if let Ok(db) = storage.db() {
            let cf = db.cf_handle(schema::Edges::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!("Column family '{}' not found", schema::Edges::CF_NAME)
            })?;
            db.get_cf(cf, key_bytes)?
        } else {
            let txn_db = storage.transaction_db()?;
            let cf = txn_db.cf_handle(schema::Edges::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!("Column family '{}' not found", schema::Edges::CF_NAME)
            })?;
            txn_db.get_cf(cf, key_bytes)?
        };

        let value_bytes = value_bytes.ok_or_else(|| anyhow::anyhow!("Edge not found: {}", id))?;

        let value: schema::EdgeCfValue = schema::Edges::value_from_bytes(&value_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

        // EdgeCfValue is now (SrcId, EdgeName, DstId, EdgeSummary)
        // Return as: (source_id, dest_id, edge_name, summary)
        Ok((value.0, value.2, value.1, value.3))
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

/// Query to scan fragments by ID with time range filtering
/// This is different from ByIdQuery because it requires time range filtering
/// and returns multiple results (scan operation) rather than a single entity lookup.
#[derive(Debug)]
pub struct FragmentsByIdTimeRangeQuery {
    /// The entity ID to search for
    pub id: Id,

    /// Time range bounds (start_bound, end_bound)
    /// Use std::ops::Bound for idiomatic Rust range specification
    pub time_range: (Bound<TimestampMilli>, Bound<TimestampMilli>),

    /// Timestamp of when the query was created
    pub ts_millis: TimestampMilli,

    /// Timeout for this query
    pub timeout: Duration,

    /// Channel to send the result back to the client
    result_tx: oneshot::Sender<Result<Vec<(TimestampMilli, FragmentContent)>>>,
}

impl FragmentsByIdTimeRangeQuery {
    /// Create a new FragmentsByIdTimeRangeQuery
    pub fn new(
        id: Id,
        time_range: (Bound<TimestampMilli>, Bound<TimestampMilli>),
        timeout: Duration,
        result_tx: oneshot::Sender<Result<Vec<(TimestampMilli, FragmentContent)>>>,
    ) -> Self {
        Self {
            id,
            time_range,
            ts_millis: TimestampMilli::now(),
            timeout,
            result_tx,
        }
    }

    /// Send the result back to the client (consumes self)
    pub fn send_result(self, result: Result<Vec<(TimestampMilli, FragmentContent)>>) {
        // Ignore error if receiver was dropped (client timeout/cancellation)
        let _ = self.result_tx.send(result);
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

/// Implement QueryExecutor for FragmentsByIdTimeRangeQuery
#[async_trait::async_trait]
impl QueryExecutor for FragmentsByIdTimeRangeQuery {
    type Output = Vec<(TimestampMilli, FragmentContent)>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        use std::ops::Bound;

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
                schema::Fragments::key_to_bytes(&schema::FragmentCfKey(id, *start_ts))
            }
            Bound::Excluded(start_ts) => {
                schema::Fragments::key_to_bytes(&schema::FragmentCfKey(
                    id,
                    TimestampMilli(start_ts.0 + 1),
                ))
            }
        };

        macro_rules! process_items {
            ($iter:expr) => {{
                for item in $iter {
                    let (key_bytes, value_bytes) = item?;
                    let key: schema::FragmentCfKey = schema::Fragments::key_from_bytes(&key_bytes)
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

                    let value: schema::FragmentCfValue =
                        schema::Fragments::value_from_bytes(&value_bytes)
                            .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;
                    fragments.push((timestamp, value.0));
                }
            }};
        }

        if let Ok(db) = storage.db() {
            let cf = db.cf_handle(schema::Fragments::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!("Column family '{}' not found", schema::Fragments::CF_NAME)
            })?;

            let iter = db.iterator_cf(
                cf,
                rocksdb::IteratorMode::From(&start_key, rocksdb::Direction::Forward),
            );
            process_items!(iter);
        } else {
            let txn_db = storage.transaction_db()?;
            let cf = txn_db
                .cf_handle(schema::Fragments::CF_NAME)
                .ok_or_else(|| {
                    anyhow::anyhow!("Column family '{}' not found", schema::Fragments::CF_NAME)
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
    }
}

/// Query to find an edge by source ID, destination ID, and name
#[derive(Debug)]
pub struct EdgeSummaryBySrcDstNameQuery {
    /// Source node ID
    pub source_id: Id,

    /// Destination node ID
    pub dest_id: Id,

    /// Edge name
    pub name: String,

    /// Timestamp of when the query was created
    pub ts_millis: TimestampMilli,

    /// Timeout for this query
    pub timeout: Duration,

    /// Channel to send the result back to the client
    result_tx: oneshot::Sender<Result<(Id, EdgeSummary)>>,
}

impl EdgeSummaryBySrcDstNameQuery {
    pub fn new(
        source_id: SrcId,
        dest_id: DstId,
        name: String,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<(Id, EdgeSummary)>>,
    ) -> Self {
        Self {
            source_id,
            dest_id,
            name,
            ts_millis: TimestampMilli::now(),
            timeout,
            result_tx,
        }
    }

    pub fn send_result(self, result: Result<(Id, EdgeSummary)>) {
        let _ = self.result_tx.send(result);
    }
}

/// Implement QueryExecutor for EdgeSummaryBySrcDstNameQuery
#[async_trait::async_trait]
impl QueryExecutor for EdgeSummaryBySrcDstNameQuery {
    type Output = (Id, EdgeSummary);

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let source_id = self.source_id;
        let dest_id = self.dest_id;
        let name = &self.name;

        let key = schema::ForwardEdgeCfKey(
            schema::EdgeSourceId(source_id),
            schema::EdgeDestinationId(dest_id),
            schema::EdgeName(name.clone()),
        );
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

        let edge_id = value.0;

        // Look up the edge summary from the edges column family
        let edge_key = schema::EdgeCfKey(edge_id);
        let edge_key_bytes = schema::Edges::key_to_bytes(&edge_key);

        let edge_value_bytes = if let Ok(db) = storage.db() {
            let cf = db.cf_handle(schema::Edges::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!("Column family '{}' not found", schema::Edges::CF_NAME)
            })?;
            db.get_cf(cf, edge_key_bytes)?
        } else {
            let txn_db = storage.transaction_db()?;
            let cf = txn_db.cf_handle(schema::Edges::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!("Column family '{}' not found", schema::Edges::CF_NAME)
            })?;
            txn_db.get_cf(cf, edge_key_bytes)?
        };

        let edge_value_bytes = edge_value_bytes
            .ok_or_else(|| anyhow::anyhow!("Edge summary not found for edge_id: {}", edge_id))?;

        let edge_value: schema::EdgeCfValue = schema::Edges::value_from_bytes(&edge_value_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize edge value: {}", e))?;

        Ok((edge_id, edge_value.3))
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

/// Id of edge source node
pub type SrcId = Id;

/// Id of edge destination node
pub type DstId = Id;

/// Query to find all edges emanating from a node by its ID
#[derive(Debug)]
pub struct EdgesFromNodeQuery {
    /// The node ID to search for
    pub id: Id,

    /// Timestamp of when the query was created
    pub ts_millis: TimestampMilli,

    /// Timeout for this query
    pub timeout: Duration,

    /// Channel to send the result back to the client
    result_tx: oneshot::Sender<Result<Vec<(SrcId, EdgeName, DstId)>>>,
}

impl EdgesFromNodeQuery {
    pub fn new(
        id: Id,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<Vec<(SrcId, EdgeName, DstId)>>>,
    ) -> Self {
        Self {
            id,
            ts_millis: TimestampMilli::now(),
            timeout,
            result_tx,
        }
    }

    pub fn send_result(self, result: Result<Vec<(SrcId, EdgeName, DstId)>>) {
        let _ = self.result_tx.send(result);
    }
}

/// Implement QueryExecutor for EdgesFromNodeQuery
#[async_trait::async_trait]
impl QueryExecutor for EdgesFromNodeQuery {
    type Output = Vec<(SrcId, EdgeName, DstId)>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let id = self.id;
        let mut edges: Vec<(SrcId, EdgeName, DstId)> = Vec::new();
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
                let (key_bytes, _value_bytes) = item?;
                let key: schema::ForwardEdgeCfKey =
                    schema::ForwardEdges::key_from_bytes(&key_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

                let source_id = key.0 .0;
                if source_id != id {
                    break;
                }

                let dest_id = key.1 .0;
                let edge_name = key.2 .0;
                edges.push((source_id, EdgeName(edge_name), dest_id));
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
                let (key_bytes, _value_bytes) = item?;
                let key: schema::ForwardEdgeCfKey =
                    schema::ForwardEdges::key_from_bytes(&key_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

                let source_id = key.0 .0;
                if source_id != id {
                    break;
                }

                let dest_id = key.1 .0;
                let edge_name = key.2 .0;
                edges.push((source_id, EdgeName(edge_name), dest_id));
            }
        }

        Ok(edges)
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

/// Query to find all edges terminating at a node by its ID
#[derive(Debug)]
pub struct EdgesToNodeQuery {
    /// The node ID to search for
    pub id: DstId,

    /// Timestamp of when the query was created
    pub ts_millis: TimestampMilli,

    /// Timeout for this query
    pub timeout: Duration,

    /// Channel to send the result back to the client
    result_tx: oneshot::Sender<Result<Vec<(DstId, EdgeName, SrcId)>>>,
}

impl EdgesToNodeQuery {
    pub fn new(
        id: DstId,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<Vec<(DstId, EdgeName, SrcId)>>>,
    ) -> Self {
        Self {
            id,
            ts_millis: TimestampMilli::now(),
            timeout,
            result_tx,
        }
    }

    pub fn send_result(self, result: Result<Vec<(DstId, EdgeName, SrcId)>>) {
        let _ = self.result_tx.send(result);
    }
}

/// Implement QueryExecutor for EdgesToNodeQuery
#[async_trait::async_trait]
impl QueryExecutor for EdgesToNodeQuery {
    type Output = Vec<(DstId, EdgeName, SrcId)>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let id = self.id;
        let mut edges: Vec<(DstId, EdgeName, SrcId)> = Vec::new();
        let prefix = id.into_bytes();

        if let Ok(db) = storage.db() {
            let cf = db.cf_handle(schema::ReverseEdges::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!(
                    "Column family '{}' not found",
                    schema::ReverseEdges::CF_NAME
                )
            })?;

            let iter = db.iterator_cf(
                cf,
                rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
            );

            for item in iter {
                let (key_bytes, _value_bytes) = item?;
                let key: schema::ReverseEdgeCfKey =
                    schema::ReverseEdges::key_from_bytes(&key_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

                let dest_id = key.0 .0;
                if dest_id != id {
                    break;
                }

                let source_id = key.1 .0;
                let edge_name = key.2 .0;
                edges.push((dest_id, EdgeName(edge_name), source_id));
            }
        } else {
            let txn_db = storage.transaction_db()?;
            let cf = txn_db
                .cf_handle(schema::ReverseEdges::CF_NAME)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Column family '{}' not found",
                        schema::ReverseEdges::CF_NAME
                    )
                })?;

            let iter = txn_db.iterator_cf(
                cf,
                rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
            );

            for item in iter {
                let (key_bytes, _value_bytes) = item?;
                let key: schema::ReverseEdgeCfKey =
                    schema::ReverseEdges::key_from_bytes(&key_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

                let dest_id = key.0 .0;
                if dest_id != id {
                    break;
                }

                let source_id = key.1 .0;
                let edge_name = key.2 .0;
                edges.push((dest_id, EdgeName(edge_name), source_id));
            }
        }

        Ok(edges)
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

/// Query to find nodes by name prefix
#[derive(Debug)]
pub struct NodesByNameQuery {
    /// The node name or prefix to search for
    pub name: NodeName,

    /// Optional start position for pagination (exclusive): (last_name, last_id)
    /// For prefix queries, this should be the full name and ID of the last result from the previous page
    pub start: Option<(NodeName, Id)>,

    /// Timestamp of when the query was created
    pub ts_millis: TimestampMilli,

    /// Maximum number of results to return
    pub limit: Option<usize>,

    /// Timeout for this query
    pub timeout: Duration,

    /// Channel to send the result back to the client
    result_tx: oneshot::Sender<Result<Vec<(NodeName, Id)>>>,
}

impl NodesByNameQuery {
    pub fn new(
        name: NodeName,
        start: Option<(NodeName, Id)>,
        limit: Option<usize>,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<Vec<(NodeName, Id)>>>,
    ) -> Self {
        Self {
            name,
            start,
            ts_millis: TimestampMilli::now(),
            timeout,
            limit,
            result_tx,
        }
    }

    pub fn send_result(self, result: Result<Vec<(NodeName, Id)>>) {
        let _ = self.result_tx.send(result);
    }
}

/// Implement QueryExecutor for NodesByNameQuery
#[async_trait::async_trait]
impl QueryExecutor for NodesByNameQuery {
    type Output = Vec<(NodeName, Id)>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
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

                let (key_bytes, _value_bytes) = item?;
                let key: schema::NodeNamesCfKey = schema::NodeNames::key_from_bytes(&key_bytes)
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

                let (key_bytes, _value_bytes) = item?;
                let key: schema::NodeNamesCfKey = schema::NodeNames::key_from_bytes(&key_bytes)
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

                nodes.push((node_name, node_id));
            }
        }

        Ok(nodes)
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

/// Query to find edges by name prefix
#[derive(Debug)]
pub struct EdgesByNameQuery {
    /// The edge name or prefix to search for
    pub name: String,

    /// Optional start position for pagination (exclusive): (last_name, last_id)
    /// For prefix queries, this should be the full name and ID of the last result from the previous page
    pub start: Option<(EdgeName, Id)>,

    /// The maximum number of results to return
    pub limit: Option<usize>,

    /// Timestamp of when the query was created
    pub ts_millis: TimestampMilli,

    /// Timeout for this query
    pub timeout: Duration,

    /// Channel to send the result back to the client
    result_tx: oneshot::Sender<Result<Vec<(EdgeName, Id)>>>,
}

impl EdgesByNameQuery {
    pub fn new(
        name: String,
        start: Option<(EdgeName, Id)>,
        limit: Option<usize>,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<Vec<(EdgeName, Id)>>>,
    ) -> Self {
        Self {
            name,
            start,
            limit,
            ts_millis: TimestampMilli::now(),
            timeout,
            result_tx,
        }
    }

    pub fn send_result(self, result: Result<Vec<(EdgeName, Id)>>) {
        let _ = self.result_tx.send(result);
    }
}

/// Implement QueryExecutor for EdgesByNameQuery
#[async_trait::async_trait]
impl QueryExecutor for EdgesByNameQuery {
    type Output = Vec<(EdgeName, Id)>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let name = &self.name;
        let mut edges: Vec<(EdgeName, Id)> = Vec::new();

        let seek_key = if let Some((start_name, start_id)) = &self.start {
            let mut bytes = Vec::with_capacity(start_name.0.len() + 16);
            bytes.extend_from_slice(start_name.0.as_bytes());
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

                let (key_bytes, _value_bytes) = item?;
                let key: schema::EdgeNamesCfKey = schema::EdgeNames::key_from_bytes(&key_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

                let edge_name = &key.0;
                let edge_id = key.1;

                if !edge_name.0.starts_with(name) {
                    break;
                }

                if let Some((start_name, start_id)) = &self.start {
                    if edge_name.0 == start_name.0 && edge_id == *start_id {
                        continue;
                    }
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

                let (key_bytes, _value_bytes) = item?;
                let key: schema::EdgeNamesCfKey = schema::EdgeNames::key_from_bytes(&key_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

                let edge_name = &key.0;
                let edge_id = key.1;

                if !edge_name.0.starts_with(name) {
                    break;
                }

                if let Some((start_name, start_id)) = &self.start {
                    if edge_name.0 == start_name.0 && edge_id == *start_id {
                        continue;
                    }
                }

                edges.push((edge_name.clone(), edge_id));
            }
        }

        Ok(edges)
    }

    fn timeout(&self) -> Duration {
        self.timeout
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
    use crate::schema::NodeSummary;
    use crate::writer::{create_mutation_writer, WriterConfig};
    use crate::{Id, Storage, TimestampMilli};
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
        };
        writer.add_node(node_args).await.unwrap();

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
        let (returned_name, returned_summary) = reader
            .node_by_id(node_id, Duration::from_secs(5))
            .await
            .unwrap();

        // Verify the data matches what we inserted
        assert_eq!(returned_name, node_name);
        assert!(returned_summary.content().is_ok());

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
                    NodeSummary::new("Should not get here".to_string()),
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
}
