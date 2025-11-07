use anyhow::{Context, Result};
use std::time::Duration;

use crate::query::{
    DstId, EdgeByIdQuery, EdgeSummaryBySrcDstNameQuery, EdgesFromNodeByIdQuery,
    EdgesToNodeByIdQuery, FragmentContentByIdQuery, NodeByIdQuery, Query, SrcId,
};
use crate::schema::{EdgeName, EdgeSummary, FragmentContent, NodeName, NodeSummary};
use crate::Id;

/// Configuration for the query reader
#[derive(Debug, Clone)]
pub struct ReaderConfig {
    /// Size of the MPMC channel buffer
    pub channel_buffer_size: usize,
}

impl Default for ReaderConfig {
    fn default() -> Self {
        Self {
            channel_buffer_size: 1000,
        }
    }
}

/// Handle for sending queries to the reader
#[derive(Debug, Clone)]
pub struct Reader {
    sender: flume::Sender<Query>,
}

impl Reader {
    /// Create a new Reader with the given sender
    pub fn new(sender: flume::Sender<Query>) -> Self {
        Reader { sender }
    }

    /// Query a node by its ID (returns name and summary)
    pub async fn node_by_id(&self, id: Id, timeout: Duration) -> Result<(NodeName, NodeSummary)> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();

        let query = NodeByIdQuery::new(id, timeout, result_tx);

        self.sender
            .send_async(Query::NodeById(query))
            .await
            .context("Failed to send query to reader queue")?;

        // Await result - timeout is handled by the consumer
        result_rx.await?
    }

    /// Query an edge by its ID (returns topology and summary)
    /// Returns (source_id, dest_id, edge_name, summary)
    pub async fn edge_by_id(
        &self,
        id: Id,
        timeout: Duration,
    ) -> Result<(SrcId, DstId, EdgeName, EdgeSummary)> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let query = EdgeByIdQuery::new(id, timeout, result_tx);

        self.sender
            .send_async(Query::EdgeById(query))
            .await
            .context("Failed to send query to reader queue")?;

        // Await result - timeout is handled by the consumer
        result_rx.await?
    }

    /// Query an edge by source ID, destination ID, and name
    /// Returns (edge_id, edge_summary)
    pub async fn edge_by_src_dst_name(
        &self,
        source_id: Id,
        dest_id: Id,
        name: String,
        timeout: Duration,
    ) -> Result<(Id, EdgeSummary)> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();

        let query = EdgeSummaryBySrcDstNameQuery::new(source_id, dest_id, name, timeout, result_tx);

        self.sender
            .send_async(Query::EdgeSummaryBySrcDstName(query))
            .await
            .context("Failed to send query to reader queue")?;

        // Await result - timeout is handled by the consumer
        result_rx.await?
    }

    /// Query a fragment by its ID
    pub async fn fragments_by_id(
        &self,
        id: Id,
        timeout: Duration,
    ) -> Result<Vec<(crate::TimestampMilli, FragmentContent)>> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();

        let query = FragmentContentByIdQuery::new(id, timeout, result_tx);

        self.sender
            .send_async(Query::FragmentContentById(query))
            .await
            .context("Failed to send query to reader queue")?;

        // Await result - timeout is handled by the consumer
        result_rx.await?
    }

    /// Query all edges from a node (outgoing edges)
    /// Returns Vec<(source_id, edge_name, dest_id)>
    pub async fn edges_from_node_by_id(
        &self,
        id: Id,
        timeout: Duration,
    ) -> Result<Vec<(SrcId, EdgeName, DstId)>> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();

        let query = EdgesFromNodeByIdQuery::new(id, timeout, result_tx);

        self.sender
            .send_async(Query::EdgesFromNodeById(query))
            .await
            .context("Failed to send query to reader queue")?;

        // Await result - timeout is handled by the consumer
        result_rx.await?
    }

    /// Query all edges to a node (incoming edges)
    /// Returns Vec<(dest_id, edge_name, source_id)>
    pub async fn edges_to_node_by_id(
        &self,
        id: Id,
        timeout: Duration,
    ) -> Result<Vec<(DstId, EdgeName, SrcId)>> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();

        let query = EdgesToNodeByIdQuery::new(id, timeout, result_tx);

        self.sender
            .send_async(Query::EdgesToNodeById(query))
            .await
            .context("Failed to send query to reader queue")?;

        // Await result - timeout is handled by the consumer
        result_rx.await?
    }

    /// Check if the reader is still active (receiver hasn't been dropped)
    pub fn is_closed(&self) -> bool {
        self.sender.is_disconnected()
    }
}

/// Create a new query reader and receiver pair
pub fn create_query_reader(config: ReaderConfig) -> (Reader, flume::Receiver<Query>) {
    let (sender, receiver) = flume::bounded(config.channel_buffer_size);
    let reader = Reader::new(sender);
    (reader, receiver)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::Duration;

    #[tokio::test]
    async fn test_reader_closed_detection() {
        let config = ReaderConfig::default();
        let (reader, receiver) = create_query_reader(config);

        assert!(!reader.is_closed());

        // Drop receiver to close channel
        drop(receiver);

        // Give tokio time to process the close
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Reader should detect channel is closed
        assert!(reader.is_closed());
    }

    #[tokio::test]
    async fn test_reader_send_operations() {
        let config = ReaderConfig::default();
        let (reader, _receiver) = create_query_reader(config);

        // Test that reader is not closed initially
        assert!(!reader.is_closed());

        // We can't test actual sending without a consumer running,
        // but we've verified the reader is constructed properly
    }
}
