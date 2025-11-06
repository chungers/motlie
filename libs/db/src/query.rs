use anyhow::Result;
use std::time::Duration;
use tokio::sync::oneshot;

use crate::schema::{EdgeSummary, FragmentContent, NodeSummary};
use crate::{Id, ReaderConfig, TimestampMilli};

/// Query enum representing all possible query types
#[derive(Debug)]
pub enum Query {
    NodeSummaryById(NodeSummaryByIdQuery),
    EdgeSummaryBySrcDstName(EdgeSummaryBySrcDstNameQuery),
    FragmentContentById(FragmentContentByIdQuery),
}

/// Query to find a node by its ID
#[derive(Debug)]
pub struct NodeSummaryByIdQuery {
    /// The node ID to search for
    pub id: Id,

    /// Timestamp of when the query was created
    pub ts_millis: TimestampMilli,

    /// Timeout for this query
    pub timeout: Duration,

    /// Channel to send the result back to the client
    result_tx: oneshot::Sender<Result<NodeSummary>>,
}

impl NodeSummaryByIdQuery {
    /// Create a new NodeByIdQuery
    pub fn new(
        id: Id,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<NodeSummary>>,
    ) -> Self {
        Self {
            id,
            ts_millis: TimestampMilli::now(),
            timeout,
            result_tx,
        }
    }

    /// Send the result back to the client (consumes self)
    pub fn send_result(self, result: Result<NodeSummary>) {
        // Ignore error if receiver was dropped (client timeout/cancellation)
        let _ = self.result_tx.send(result);
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
    result_tx: oneshot::Sender<Result<EdgeSummary>>,
}

impl EdgeSummaryBySrcDstNameQuery {
    pub fn new(
        source_id: Id,
        dest_id: Id,
        name: String,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<EdgeSummary>>,
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

    pub fn send_result(self, result: Result<EdgeSummary>) {
        let _ = self.result_tx.send(result);
    }
}

/// Query to find a fragment by its ID
#[derive(Debug)]
pub struct FragmentContentByIdQuery {
    /// The fragment ID to search for
    pub id: Id,

    /// Timestamp of when the query was created
    pub ts_millis: TimestampMilli,

    /// Timeout for this query
    pub timeout: Duration,

    /// Channel to send the result back to the client
    result_tx: oneshot::Sender<Result<FragmentContent>>,
}

impl FragmentContentByIdQuery {
    pub fn new(
        id: Id,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<FragmentContent>>,
    ) -> Self {
        Self {
            id,
            ts_millis: TimestampMilli::now(),
            timeout,
            result_tx,
        }
    }

    pub fn send_result(self, result: Result<FragmentContent>) {
        let _ = self.result_tx.send(result);
    }
}

/// Trait for processing different types of queries
#[async_trait::async_trait]
pub trait Processor: Send + Sync {
    /// Get a node summary by its ID
    async fn get_node_summary_by_id(&self, query: &NodeSummaryByIdQuery) -> Result<NodeSummary>;

    /// Get an edge summary by source ID, destination ID, and name
    async fn get_edge_summary_by_src_dst_name(
        &self,
        query: &EdgeSummaryBySrcDstNameQuery,
    ) -> Result<EdgeSummary>;

    /// Get fragment content by its ID
    async fn get_fragment_content_by_id(
        &self,
        query: &FragmentContentByIdQuery,
    ) -> Result<FragmentContent>;
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
            Query::NodeSummaryById(q) => {
                log::debug!("Processing NodeById: id={}", q.id);

                // Apply timeout from the query
                let result = tokio::time::timeout(
                    q.timeout,
                    self.processor.get_node_summary_by_id(&q)
                ).await;

                // Convert timeout error to anyhow error
                let result = match result {
                    Ok(r) => r,
                    Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", q.timeout)),
                };

                q.send_result(result);
            }
            Query::EdgeSummaryBySrcDstName(q) => {
                log::debug!(
                    "Processing EdgeBySrcDstName: source={}, dest={}, name={}",
                    q.source_id,
                    q.dest_id,
                    q.name
                );

                let result = tokio::time::timeout(
                    q.timeout,
                    self.processor.get_edge_summary_by_src_dst_name(&q)
                ).await;

                let result = match result {
                    Ok(r) => r,
                    Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", q.timeout)),
                };

                q.send_result(result);
            }
            Query::FragmentContentById(q) => {
                log::debug!("Processing FragmentById: id={}", q.id);

                let result = tokio::time::timeout(
                    q.timeout,
                    self.processor.get_fragment_content_by_id(&q)
                ).await;

                let result = match result {
                    Ok(r) => r,
                    Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", q.timeout)),
                };

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
        async fn get_node_summary_by_id(&self, query: &NodeSummaryByIdQuery) -> Result<NodeSummary> {
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok(NodeSummary::new(format!("Node: {:?}", query.id)))
        }

        async fn get_edge_summary_by_src_dst_name(
            &self,
            query: &EdgeSummaryBySrcDstNameQuery,
        ) -> Result<EdgeSummary> {
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok(EdgeSummary::new(format!("Edge: {:?}", query.name)))
        }

        async fn get_fragment_content_by_id(
            &self,
            query: &FragmentContentByIdQuery,
        ) -> Result<FragmentContent> {
            tokio::time::sleep(Duration::from_millis(10)).await;
            Ok(FragmentContent::new(format!("Fragment: {:?}", query.id)))
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
        let result = reader
            .node_by_id(node_id, Duration::from_secs(5))
            .await
            .unwrap();

        // The mock processor returns "Node: Some(...)" so just check for "Node:"
        assert!(result.content().unwrap().contains("Node:"));

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
            async fn get_node_summary_by_id(&self, _query: &NodeSummaryByIdQuery) -> Result<NodeSummary> {
                // Sleep longer than the query timeout
                tokio::time::sleep(Duration::from_millis(200)).await;
                Ok(NodeSummary::new("Should not get here".to_string()))
            }

            async fn get_edge_summary_by_src_dst_name(&self, _query: &EdgeSummaryBySrcDstNameQuery) -> Result<EdgeSummary> {
                Ok(EdgeSummary::new("N/A".to_string()))
            }

            async fn get_fragment_content_by_id(&self, _query: &FragmentContentByIdQuery) -> Result<FragmentContent> {
                Ok(FragmentContent::new("N/A".to_string()))
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
        let result = reader.node_by_id(node_id, Duration::from_millis(50)).await;

        // Should timeout because processor takes 200ms but timeout is 50ms
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("timeout") || err_msg.contains("Query timeout"));

        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }
}
