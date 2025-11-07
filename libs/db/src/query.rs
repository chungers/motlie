use anyhow::Result;
use std::time::Duration;
use tokio::sync::oneshot;

use crate::schema::{EdgeName, EdgeSummary, FragmentContent, NodeName, NodeSummary};
use crate::{Id, ReaderConfig, TimestampMilli};

/// Query enum representing all possible query types
#[derive(Debug)]
pub enum Query {
    NodeById(NodeByIdQuery),
    EdgeSummaryById(EdgeSummaryByIdQuery),
    EdgeSummaryBySrcDstName(EdgeSummaryBySrcDstNameQuery),
    FragmentContentById(FragmentContentByIdQuery),
    EdgesFromNodeById(EdgesFromNodeByIdQuery),
    EdgesToNodeById(EdgesToNodeByIdQuery),
}

/// Type alias for querying node by ID (returns name and summary)
pub type NodeByIdQuery = ByIdQuery<(NodeName, NodeSummary)>;
/// Type alias for querying edge summaries by ID
pub type EdgeSummaryByIdQuery = ByIdQuery<EdgeSummary>;
/// Type alias for querying fragment content by ID of node or edge associated with it
pub type FragmentContentByIdQuery = ByIdQuery<Vec<(TimestampMilli, FragmentContent)>>;
// Note: EdgesFromNodeByIdQuery and EdgesToNodeByIdQuery can't use ByIdQuery<T>
// because they have the same result type but different processing logic.
// They are implemented as separate structs below.

/// Trait for queries that produce results with timeout handling
#[async_trait::async_trait]
pub trait QueryWithResult: Send + Sync {
    /// The type of result this query produces
    type ResultType: Send;

    /// Execute the query with timeout and return the result
    async fn result<P: Processor>(&self, processor: &P) -> Result<Self::ResultType>;

    /// Get the timeout for this query
    fn timeout(&self) -> Duration;
}

/// Sealed trait module to prevent external implementations
mod sealed {
    use crate::schema::{EdgeName, EdgeSummary, FragmentContent, NodeName, NodeSummary};
    use crate::{Id, TimestampMilli};

    pub trait ByIdQueryable {}
    impl ByIdQueryable for (NodeName, NodeSummary) {}
    impl ByIdQueryable for EdgeSummary {}
    impl ByIdQueryable for Vec<(TimestampMilli, FragmentContent)> {}
    impl ByIdQueryable for Vec<(Id, EdgeName, Id)> {} // Forward and reverse edges use same tuple type
}

/// Trait for types that can be queried by ID
#[async_trait::async_trait]
pub trait ByIdQueryable: sealed::ByIdQueryable + Send + Sync + 'static {
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
impl ByIdQueryable for EdgeSummary {
    async fn fetch_by_id<P: Processor>(
        _id: Id,
        processor: &P,
        query: &ByIdQuery<Self>,
    ) -> Result<Self> {
        processor.get_edge_summary_by_id(query).await
    }
}

#[async_trait::async_trait]
impl ByIdQueryable for Vec<(TimestampMilli, FragmentContent)> {
    async fn fetch_by_id<P: Processor>(
        _id: Id,
        processor: &P,
        query: &ByIdQuery<Self>,
    ) -> Result<Self> {
        processor.get_fragment_content_by_id(query).await
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
impl<T: ByIdQueryable> QueryWithResult for ByIdQuery<T> {
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
        source_id: Id,
        dest_id: Id,
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
impl QueryWithResult for EdgeSummaryBySrcDstNameQuery {
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
pub struct EdgesFromNodeByIdQuery {
    /// The node ID to search for
    pub id: Id,

    /// Timestamp of when the query was created
    pub ts_millis: TimestampMilli,

    /// Timeout for this query
    pub timeout: Duration,

    /// Channel to send the result back to the client
    result_tx: oneshot::Sender<Result<Vec<(SrcId, EdgeName, DstId)>>>,
}

impl EdgesFromNodeByIdQuery {
    pub fn new(
        id: Id,
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

    pub fn send_result(self, result: Result<Vec<(Id, EdgeName, Id)>>) {
        let _ = self.result_tx.send(result);
    }
}

#[async_trait::async_trait]
impl QueryWithResult for EdgesFromNodeByIdQuery {
    type ResultType = Vec<(Id, EdgeName, Id)>;

    async fn result<P: Processor>(&self, processor: &P) -> Result<Vec<(Id, EdgeName, Id)>> {
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
pub struct EdgesToNodeByIdQuery {
    /// The node ID to search for
    pub id: Id,

    /// Timestamp of when the query was created
    pub ts_millis: TimestampMilli,

    /// Timeout for this query
    pub timeout: Duration,

    /// Channel to send the result back to the client
    result_tx: oneshot::Sender<Result<Vec<(DstId, EdgeName, SrcId)>>>,
}

impl EdgesToNodeByIdQuery {
    pub fn new(
        id: Id,
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
impl QueryWithResult for EdgesToNodeByIdQuery {
    type ResultType = Vec<(SrcId, EdgeName, DstId)>;

    async fn result<P: Processor>(&self, processor: &P) -> Result<Vec<(Id, EdgeName, Id)>> {
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

/// Trait for processing different types of queries
#[async_trait::async_trait]
pub trait Processor: Send + Sync {
    /// Get a node by its ID (returns name and summary)
    async fn get_node_by_id(&self, query: &ByIdQuery<(NodeName, NodeSummary)>) -> Result<(NodeName, NodeSummary)>;

    /// Get an edge summary by its ID
    async fn get_edge_summary_by_id(&self, query: &ByIdQuery<EdgeSummary>) -> Result<EdgeSummary>;

    /// Get an edge summary by source ID, destination ID, and name
    /// Returns (edge_id, edge_summary)
    async fn get_edge_summary_by_src_dst_name(
        &self,
        query: &EdgeSummaryBySrcDstNameQuery,
    ) -> Result<(Id, EdgeSummary)>;

    /// Get all fragment content with timestamps for a given ID, sorted by timestamp
    async fn get_fragment_content_by_id(
        &self,
        query: &ByIdQuery<Vec<(TimestampMilli, FragmentContent)>>,
    ) -> Result<Vec<(TimestampMilli, FragmentContent)>>;

    /// Get all edges emanating from a node by its ID
    /// Returns (source_id, edge_name, dest_id) tuples sorted by RocksDB key order
    async fn get_edges_from_node_by_id(
        &self,
        query: &EdgesFromNodeByIdQuery,
    ) -> Result<Vec<(Id, EdgeName, Id)>>;

    /// Get all edges terminating at a node by its ID
    /// Returns (dest_id, edge_name, source_id) tuples sorted by RocksDB key order
    async fn get_edges_to_node_by_id(
        &self,
        query: &EdgesToNodeByIdQuery,
    ) -> Result<Vec<(Id, EdgeName, Id)>>;
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
        match query {
            Query::NodeById(q) => {
                log::debug!("Processing NodeById: id={}", q.id);
                let result = q.result(&self.processor).await;
                q.send_result(result);
            }
            Query::EdgeSummaryById(q) => {
                log::debug!("Processing EdgeById: id={}", q.id);
                let result = q.result(&self.processor).await;
                q.send_result(result);
            }
            Query::EdgeSummaryBySrcDstName(q) => {
                log::debug!(
                    "Processing EdgeBySrcDstName: source={}, dest={}, name={}",
                    q.source_id,
                    q.dest_id,
                    q.name
                );
                let result = q.result(&self.processor).await;
                q.send_result(result);
            }
            Query::FragmentContentById(q) => {
                log::debug!("Processing FragmentById: id={}", q.id);
                let result = q.result(&self.processor).await;
                q.send_result(result);
            }
            Query::EdgesFromNodeById(q) => {
                log::debug!("Processing EdgesFromNodeById: id={}", q.id);
                let result = q.result(&self.processor).await;
                q.send_result(result);
            }
            Query::EdgesToNodeById(q) => {
                log::debug!("Processing EdgesToNodeById: id={}", q.id);
                let result = q.result(&self.processor).await;
                q.send_result(result);
            }
        }
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
        async fn get_node_by_id(
            &self,
            query: &NodeByIdQuery,
        ) -> Result<(NodeName, NodeSummary)> {
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok((
                format!("node_{}", query.id),
                NodeSummary::new(format!("Node: {:?}", query.id)),
            ))
        }

        async fn get_edge_summary_by_id(
            &self,
            query: &EdgeSummaryByIdQuery,
        ) -> Result<EdgeSummary> {
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok(EdgeSummary::new(format!("Edge: {:?}", query.id)))
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

        async fn get_fragment_content_by_id(
            &self,
            query: &FragmentContentByIdQuery,
        ) -> Result<Vec<(TimestampMilli, FragmentContent)>> {
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok(vec![(
                query.ts_millis,
                FragmentContent::new(format!("Fragment: {:?}", query.id)),
            )])
        }

        async fn get_edges_from_node_by_id(
            &self,
            query: &EdgesFromNodeByIdQuery,
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
            query: &EdgesToNodeByIdQuery,
        ) -> Result<Vec<(Id, crate::schema::EdgeName, Id)>> {
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok(vec![(
                query.id,
                crate::schema::EdgeName("test_edge".to_string()),
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

            async fn get_edge_summary_by_id(
                &self,
                _query: &EdgeSummaryByIdQuery,
            ) -> Result<EdgeSummary> {
                Ok(EdgeSummary::new("N/A".to_string()))
            }

            async fn get_edge_summary_by_src_dst_name(
                &self,
                _query: &EdgeSummaryBySrcDstNameQuery,
            ) -> Result<(Id, EdgeSummary)> {
                Ok((Id::new(), EdgeSummary::new("N/A".to_string())))
            }

            async fn get_fragment_content_by_id(
                &self,
                query: &FragmentContentByIdQuery,
            ) -> Result<Vec<(TimestampMilli, FragmentContent)>> {
                Ok(vec![(
                    query.ts_millis,
                    FragmentContent::new("N/A".to_string()),
                )])
            }

            async fn get_edges_from_node_by_id(
                &self,
                query: &EdgesFromNodeByIdQuery,
            ) -> Result<Vec<(Id, crate::schema::EdgeName, Id)>> {
                Ok(vec![(
                    query.id,
                    crate::schema::EdgeName("N/A".to_string()),
                    Id::new(),
                )])
            }

            async fn get_edges_to_node_by_id(
                &self,
                query: &EdgesToNodeByIdQuery,
            ) -> Result<Vec<(Id, crate::schema::EdgeName, Id)>> {
                Ok(vec![(
                    query.id,
                    crate::schema::EdgeName("N/A".to_string()),
                    Id::new(),
                )])
            }
        }

        let config = crate::ReaderConfig {
            channel_buffer_size: 10,
        };

        let (sender, receiver) = flume::bounded::<Query>(config.channel_buffer_size);
        let reader = crate::Reader::new(sender);

        // Spawn consumer with slow processor
        let processor = SlowProcessor;
        let consumer = Consumer::new(receiver, config, processor);
        let consumer_handle = spawn_consumer(consumer);

        // Send query with short timeout
        let node_id = Id::new();
        let result = reader
            .node_by_id(node_id, Duration::from_millis(50))
            .await;

        // Should timeout because processor takes 200ms but timeout is 50ms
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("timeout") || err_msg.contains("Query timeout"));

        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }
}
