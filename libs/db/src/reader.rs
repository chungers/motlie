use anyhow::{Context, Result};
use std::time::Duration;

use crate::query::{DstId, EdgeSummaryBySrcDstName, Query, SrcId};
use crate::schema::{EdgeName, EdgeSummary};
use crate::{Id, TimestampMilli};

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

    /// Send a query to the reader queue (used by query builders)
    pub async fn send_query(&self, query: Query) -> Result<()> {
        self.sender
            .send_async(query)
            .await
            .context("Failed to send query to reader queue")
    }

    /// Query an edge by source ID, destination ID, and name
    /// Returns (edge_id, edge_summary)
    ///
    /// # Arguments
    /// * `source_id` - Source node ID
    /// * `dest_id` - Destination node ID
    /// * `name` - Edge name
    /// * `reference_ts_millis` - Optional reference timestamp for temporal validity checks.
    ///   If None, defaults to current time in the query executor.
    /// * `timeout` - Query timeout duration
    pub async fn edge_by_src_dst_name(
        &self,
        source_id: SrcId,
        dest_id: DstId,
        name: String,
        reference_ts_millis: Option<TimestampMilli>,
        timeout: Duration,
    ) -> Result<(Id, EdgeSummary)> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();

        let query = EdgeSummaryBySrcDstName::with_channel(
            source_id,
            dest_id,
            name,
            reference_ts_millis,
            timeout,
            result_tx,
        );

        self.sender
            .send_async(Query::EdgeSummaryBySrcDstName(query))
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
