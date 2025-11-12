use anyhow::Result;
use std::ops::Bound;
use std::time::Duration;
use tokio::sync::oneshot;

use crate::schema::{EdgeName, EdgeSummary, FragmentContent, NodeName, NodeSummary};
use crate::{Id, ReaderConfig, TimestampMilli};

/// Trait for queries that produce results with timeout handling
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
pub enum Queries {
    NodeById(NodeByIdQuery),
    EdgeById(EdgeByIdQuery),
    EdgeSummaryBySrcDstName(EdgeSummaryBySrcDstNameQuery),
    FragmentsByIdTimeRange(FragmentsByIdTimeRangeQuery),
    EdgesFromNode(EdgesFromNodeQuery),
    EdgesToNode(EdgesToNodeQuery),
    NodesByName(NodesByNameQuery),
    EdgesByName(EdgesByNameQuery),
}

impl std::fmt::Display for Queries {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Queries::NodeById(q) => write!(f, "NodeById: id={}", q.id),
            Queries::EdgeById(q) => write!(f, "EdgeById: id={}", q.id),
            Queries::EdgeSummaryBySrcDstName(q) => write!(
                f,
                "EdgeBySrcDstName: source={}, dest={}, name={}",
                q.source_id, q.dest_id, q.name
            ),
            Queries::FragmentsByIdTimeRange(q) => {
                write!(f, "FragmentsByIdTimeRange: id={}, range={:?}", q.id, q.time_range)
            }
            Queries::EdgesFromNode(q) => write!(f, "EdgesFromNodeById: id={}", q.id),
            Queries::EdgesToNode(q) => write!(f, "EdgesToNodeById: id={}", q.id),
            Queries::NodesByName(q) => write!(f, "NodesByName: name={}", q.name),
            Queries::EdgesByName(q) => write!(f, "EdgesByName: name={}", q.name),
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
impl QueryProcessor for Queries {
    async fn process_and_send<P: Processor>(self, processor: &P) {
        match self {
            Queries::NodeById(q) => q.process_and_send(processor).await,
            Queries::EdgeById(q) => q.process_and_send(processor).await,
            Queries::EdgeSummaryBySrcDstName(q) => q.process_and_send(processor).await,
            Queries::FragmentsByIdTimeRange(q) => q.process_and_send(processor).await,
            Queries::EdgesFromNode(q) => q.process_and_send(processor).await,
            Queries::EdgesToNode(q) => q.process_and_send(processor).await,
            Queries::NodesByName(q) => q.process_and_send(processor).await,
            Queries::EdgesByName(q) => q.process_and_send(processor).await,
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

/// Sealed trait module to prevent external implementations
mod sealed {
    use crate::query::{DstId, SrcId};
    use crate::schema::{EdgeName, EdgeSummary, NodeName, NodeSummary};
    use crate::Id;

    pub trait Queryable {}
    impl Queryable for (NodeName, NodeSummary) {}
    impl Queryable for (SrcId, DstId, EdgeName, EdgeSummary) {}
    impl Queryable for Vec<(Id, EdgeName, Id)> {} // Forward and reverse edges use same tuple type
}

/// Trait for types that can be queried by ID
#[async_trait::async_trait]
pub trait ByIdQueryable: sealed::Queryable + Send + Sync + 'static {
    /// Call the appropriate processor method for this type
    async fn fetch_by_id<P: Processor>(
        id: Id,
        processor: &P,
        query: &ByIdQuery<Self>,
    ) -> Result<Self>
    where
        Self: Sized;
}

#[async_trait::async_trait]
impl ByIdQueryable for (NodeName, NodeSummary) {
    async fn fetch_by_id<P: Processor>(
        _id: Id,
        processor: &P,
        query: &ByIdQuery<Self>,
    ) -> Result<Self> {
        processor.get_node_by_id(query).await
    }
}

#[async_trait::async_trait]
impl ByIdQueryable for (SrcId, DstId, EdgeName, EdgeSummary) {
    async fn fetch_by_id<P: Processor>(
        _id: Id,
        processor: &P,
        query: &ByIdQuery<Self>,
    ) -> Result<Self> {
        processor.get_edge_by_id(query).await
    }
}

// Note: EdgesFromNodeByIdQuery and EdgesToNodeByIdQuery share the same result type
// Vec<(Id, EdgeName, Id)>, but the semantics differ:
// - EdgesFromNodeById returns (source_id, edge_name, dest_id)
// - EdgesToNodeById returns (dest_id, edge_name, source_id)
// The ByIdQueryable trait implementation is handled via the Processor trait methods

/// Generic query to find an entity by its ID
#[derive(Debug)]
pub struct ByIdQuery<T: ByIdQueryable> {
    /// The entity ID to search for
    pub id: Id,

    /// Timestamp of when the query was created
    pub ts_millis: TimestampMilli,

    /// Timeout for this query
    pub timeout: Duration,

    /// Channel to send the result back to the client
    result_tx: oneshot::Sender<Result<T>>,
}

impl<T: ByIdQueryable> ByIdQuery<T> {
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

#[async_trait::async_trait]
impl<T: ByIdQueryable> QueryWithTimeout for ByIdQuery<T> {
    type ResultType = T;

    async fn result<P: Processor>(&self, processor: &P) -> Result<T> {
        let result =
            tokio::time::timeout(self.timeout, T::fetch_by_id(self.id, processor, self)).await;

        match result {
            Ok(r) => r,
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", self.timeout)),
        }
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

#[async_trait::async_trait]
impl QueryWithTimeout for FragmentsByIdTimeRangeQuery {
    type ResultType = Vec<(TimestampMilli, FragmentContent)>;

    async fn result<P: Processor>(&self, processor: &P) -> Result<Self::ResultType> {
        let result =
            tokio::time::timeout(self.timeout, processor.get_fragments_by_id_time_range(self))
                .await;

        match result {
            Ok(r) => r,
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", self.timeout)),
        }
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

#[async_trait::async_trait]
impl QueryWithTimeout for EdgeSummaryBySrcDstNameQuery {
    type ResultType = (Id, EdgeSummary);

    async fn result<P: Processor>(&self, processor: &P) -> Result<(Id, EdgeSummary)> {
        let result = tokio::time::timeout(
            self.timeout,
            processor.get_edge_summary_by_src_dst_name(self),
        )
        .await;

        match result {
            Ok(r) => r,
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", self.timeout)),
        }
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

#[async_trait::async_trait]
impl QueryWithTimeout for EdgesFromNodeQuery {
    type ResultType = Vec<(SrcId, EdgeName, DstId)>;

    async fn result<P: Processor>(&self, processor: &P) -> Result<Vec<(SrcId, EdgeName, DstId)>> {
        let result =
            tokio::time::timeout(self.timeout, processor.get_edges_from_node_by_id(self)).await;

        match result {
            Ok(r) => r,
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", self.timeout)),
        }
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

#[async_trait::async_trait]
impl QueryWithTimeout for EdgesToNodeQuery {
    type ResultType = Vec<(SrcId, EdgeName, DstId)>;

    async fn result<P: Processor>(&self, processor: &P) -> Result<Vec<(DstId, EdgeName, SrcId)>> {
        let result =
            tokio::time::timeout(self.timeout, processor.get_edges_to_node_by_id(self)).await;

        match result {
            Ok(r) => r,
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", self.timeout)),
        }
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

    /// Timestamp of when the query was created
    pub ts_millis: TimestampMilli,

    /// Timeout for this query
    pub timeout: Duration,

    /// Channel to send the result back to the client
    result_tx: oneshot::Sender<Result<Vec<(NodeName, Id)>>>,
}

impl NodesByNameQuery {
    pub fn new(
        name: NodeName,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<Vec<(NodeName, Id)>>>,
    ) -> Self {
        Self {
            name,
            ts_millis: TimestampMilli::now(),
            timeout,
            result_tx,
        }
    }

    pub fn send_result(self, result: Result<Vec<(NodeName, Id)>>) {
        let _ = self.result_tx.send(result);
    }
}

#[async_trait::async_trait]
impl QueryWithTimeout for NodesByNameQuery {
    type ResultType = Vec<(NodeName, Id)>;

    async fn result<P: Processor>(&self, processor: &P) -> Result<Vec<(NodeName, Id)>> {
        let result = tokio::time::timeout(self.timeout, processor.get_nodes_by_name(self)).await;

        match result {
            Ok(r) => r,
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", self.timeout)),
        }
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
        timeout: Duration,
        result_tx: oneshot::Sender<Result<Vec<(EdgeName, Id)>>>,
    ) -> Self {
        Self {
            name,
            ts_millis: TimestampMilli::now(),
            timeout,
            result_tx,
        }
    }

    pub fn send_result(self, result: Result<Vec<(EdgeName, Id)>>) {
        let _ = self.result_tx.send(result);
    }
}

#[async_trait::async_trait]
impl QueryWithTimeout for EdgesByNameQuery {
    type ResultType = Vec<(EdgeName, Id)>;

    async fn result<P: Processor>(&self, processor: &P) -> Result<Vec<(EdgeName, Id)>> {
        let result = tokio::time::timeout(self.timeout, processor.get_edges_by_name(self)).await;

        match result {
            Ok(r) => r,
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", self.timeout)),
        }
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

/// Trait for processing different types of queries
#[async_trait::async_trait]
pub trait Processor: Send + Sync {
    /// Get a node by its ID (returns name and summary)
    async fn get_node_by_id(
        &self,
        query: &ByIdQuery<(NodeName, NodeSummary)>,
    ) -> Result<(NodeName, NodeSummary)>;

    /// Get an edge by its ID (returns topology and summary)
    async fn get_edge_by_id(
        &self,
        query: &ByIdQuery<(SrcId, DstId, EdgeName, EdgeSummary)>,
    ) -> Result<(SrcId, DstId, EdgeName, EdgeSummary)>;

    /// Get an edge summary by source ID, destination ID, and name
    /// Returns (edge_id, edge_summary)
    async fn get_edge_summary_by_src_dst_name(
        &self,
        query: &EdgeSummaryBySrcDstNameQuery,
    ) -> Result<(Id, EdgeSummary)>;

    /// Get fragments by ID filtered by time range
    async fn get_fragments_by_id_time_range(
        &self,
        query: &FragmentsByIdTimeRangeQuery,
    ) -> Result<Vec<(TimestampMilli, FragmentContent)>>;

    /// Get all edges emanating from a node by its ID
    /// Returns (source_id, edge_name, dest_id) tuples sorted by RocksDB key order
    async fn get_edges_from_node_by_id(
        &self,
        query: &EdgesFromNodeQuery,
    ) -> Result<Vec<(SrcId, EdgeName, DstId)>>;

    /// Get all edges terminating at a node by its ID
    /// Returns (dest_id, edge_name, source_id) tuples sorted by RocksDB key order
    async fn get_edges_to_node_by_id(
        &self,
        query: &EdgesToNodeQuery,
    ) -> Result<Vec<(DstId, EdgeName, SrcId)>>;

    /// Get nodes by name prefix scan
    /// Returns Vec<(node_name, node_id)> sorted by RocksDB key order
    async fn get_nodes_by_name(&self, query: &NodesByNameQuery) -> Result<Vec<(NodeName, Id)>>;

    /// Get edges by name prefix scan
    /// Returns Vec<(edge_name, edge_id)> sorted by RocksDB key order
    /// Note: EdgeNames CF key is (name, dst_id, src_id, edge_id), so results
    /// will be grouped by name, then ordered by destination, then source
    async fn get_edges_by_name(&self, query: &EdgesByNameQuery) -> Result<Vec<(EdgeName, Id)>>;
}

/// Generic consumer that processes queries using a Processor
pub struct Consumer<P: Processor> {
    receiver: flume::Receiver<Queries>,
    config: ReaderConfig,
    processor: P,
}

impl<P: Processor> Consumer<P> {
    /// Create a new Consumer
    pub fn new(receiver: flume::Receiver<Queries>, config: ReaderConfig, processor: P) -> Self {
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
    async fn process_query(&self, query: Queries) {
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
    use crate::Id;
    use tokio::time::Duration;

    // Mock processor for testing
    struct TestProcessor;

    #[async_trait::async_trait]
    impl Processor for TestProcessor {
        async fn get_node_by_id(&self, query: &NodeByIdQuery) -> Result<(NodeName, NodeSummary)> {
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok((
                format!("node_{}", query.id),
                NodeSummary::new(format!("Node: {:?}", query.id)),
            ))
        }

        async fn get_edge_by_id(
            &self,
            query: &EdgeByIdQuery,
        ) -> Result<(SrcId, DstId, EdgeName, EdgeSummary)> {
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok((
                query.id,
                Id::new(),
                EdgeName("test_edge".to_string()),
                EdgeSummary::new(format!("Edge: {:?}", query.id)),
            ))
        }

        async fn get_edge_summary_by_src_dst_name(
            &self,
            query: &EdgeSummaryBySrcDstNameQuery,
        ) -> Result<(Id, EdgeSummary)> {
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok((
                Id::new(),
                EdgeSummary::new(format!("Edge: {:?}", query.name)),
            ))
        }

        async fn get_fragments_by_id_time_range(
            &self,
            query: &FragmentsByIdTimeRangeQuery,
        ) -> Result<Vec<(TimestampMilli, FragmentContent)>> {
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok(vec![(
                query.ts_millis,
                FragmentContent::new(format!("Fragment: {:?}", query.id)),
            )])
        }

        async fn get_edges_from_node_by_id(
            &self,
            query: &EdgesFromNodeQuery,
        ) -> Result<Vec<(Id, crate::schema::EdgeName, Id)>> {
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok(vec![(
                query.id,
                crate::schema::EdgeName("test_edge".to_string()),
                Id::new(),
            )])
        }

        async fn get_edges_to_node_by_id(
            &self,
            query: &EdgesToNodeQuery,
        ) -> Result<Vec<(Id, crate::schema::EdgeName, Id)>> {
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok(vec![(
                query.id,
                crate::schema::EdgeName("test_edge".to_string()),
                Id::new(),
            )])
        }

        async fn get_nodes_by_name(&self, query: &NodesByNameQuery) -> Result<Vec<(NodeName, Id)>> {
            tokio::time::sleep(Duration::from_millis(10)).await;
            // Mock: return a node matching the prefix
            Ok(vec![(format!("{}_test", query.name), Id::new())])
        }

        async fn get_edges_by_name(
            &self,
            query: &EdgesByNameQuery,
        ) -> Result<Vec<(crate::schema::EdgeName, Id)>> {
            tokio::time::sleep(Duration::from_millis(10)).await;
            // Mock: return an edge matching the prefix
            Ok(vec![(
                crate::schema::EdgeName(format!("{}_test", query.name)),
                Id::new(),
            )])
        }
    }

    #[tokio::test]
    async fn test_consumer_basic() {
        let config = crate::ReaderConfig {
            channel_buffer_size: 10,
        };

        let (reader, receiver) = {
            let (sender, receiver) = flume::bounded(config.channel_buffer_size);
            let reader = crate::Reader::new(sender);
            (reader, receiver)
        };

        let processor = TestProcessor;
        let consumer = Consumer::new(receiver, config, processor);

        // Spawn consumer
        let consumer_handle = spawn_consumer(consumer);

        // Send a query
        let node_id = Id::new();
        let (name, summary) = reader
            .node_by_id(node_id, Duration::from_secs(5))
            .await
            .unwrap();

        // The mock processor returns name and summary
        assert!(name.starts_with("node_"));
        assert!(summary.content().unwrap().contains("Node:"));

        // Drop reader to close channel
        drop(reader);

        // Wait for consumer to finish
        consumer_handle.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_query_timeout() {
        // Test that timeout works correctly with a slow processor
        struct SlowProcessor;

        #[async_trait::async_trait]
        impl Processor for SlowProcessor {
            async fn get_node_by_id(
                &self,
                _query: &NodeByIdQuery,
            ) -> Result<(NodeName, NodeSummary)> {
                // Sleep longer than the query timeout
                tokio::time::sleep(Duration::from_millis(200)).await;
                Ok((
                    "should_not_get_here".to_string(),
                    NodeSummary::new("Should not get here".to_string()),
                ))
            }

            async fn get_edge_by_id(
                &self,
                _query: &EdgeByIdQuery,
            ) -> Result<(SrcId, DstId, EdgeName, EdgeSummary)> {
                Ok((
                    Id::new(),
                    Id::new(),
                    EdgeName("N/A".to_string()),
                    EdgeSummary::new("N/A".to_string()),
                ))
            }

            async fn get_edge_summary_by_src_dst_name(
                &self,
                _query: &EdgeSummaryBySrcDstNameQuery,
            ) -> Result<(Id, EdgeSummary)> {
                Ok((Id::new(), EdgeSummary::new("N/A".to_string())))
            }

            async fn get_fragments_by_id_time_range(
                &self,
                query: &FragmentsByIdTimeRangeQuery,
            ) -> Result<Vec<(TimestampMilli, FragmentContent)>> {
                Ok(vec![(
                    query.ts_millis,
                    FragmentContent::new("N/A".to_string()),
                )])
            }

            async fn get_edges_from_node_by_id(
                &self,
                query: &EdgesFromNodeQuery,
            ) -> Result<Vec<(Id, crate::schema::EdgeName, Id)>> {
                Ok(vec![(
                    query.id,
                    crate::schema::EdgeName("N/A".to_string()),
                    Id::new(),
                )])
            }

            async fn get_edges_to_node_by_id(
                &self,
                query: &EdgesToNodeQuery,
            ) -> Result<Vec<(Id, crate::schema::EdgeName, Id)>> {
                Ok(vec![(
                    query.id,
                    crate::schema::EdgeName("N/A".to_string()),
                    Id::new(),
                )])
            }

            async fn get_nodes_by_name(
                &self,
                _query: &NodesByNameQuery,
            ) -> Result<Vec<(NodeName, Id)>> {
                Ok(vec![("N/A".to_string(), Id::new())])
            }

            async fn get_edges_by_name(
                &self,
                _query: &EdgesByNameQuery,
            ) -> Result<Vec<(crate::schema::EdgeName, Id)>> {
                Ok(vec![(
                    crate::schema::EdgeName("N/A".to_string()),
                    Id::new(),
                )])
            }
        }

        let config = crate::ReaderConfig {
            channel_buffer_size: 10,
        };

        let (sender, receiver) = flume::bounded::<Queries>(config.channel_buffer_size);
        let reader = crate::Reader::new(sender);

        // Spawn consumer with slow processor
        let processor = SlowProcessor;
        let consumer = Consumer::new(receiver, config, processor);
        let consumer_handle = spawn_consumer(consumer);

        // Send query with short timeout
        let node_id = Id::new();
        let result = reader.node_by_id(node_id, Duration::from_millis(50)).await;

        // Should timeout because processor takes 200ms but timeout is 50ms
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("timeout") || err_msg.contains("Query timeout"));

        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }
}
