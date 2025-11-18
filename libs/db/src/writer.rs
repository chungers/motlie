use anyhow::{Context, Result};
use tokio::sync::mpsc;

use crate::{AddEdge, AddFragment, AddNode, InvalidateArgs, Mutation};

/// Configuration for the mutation writer
#[derive(Debug, Clone)]
pub struct WriterConfig {
    /// Size of the MPSC channel buffer
    pub channel_buffer_size: usize,
}

impl Default for WriterConfig {
    fn default() -> Self {
        Self {
            channel_buffer_size: 1000,
        }
    }
}

/// Handle for sending mutations to the writer with batching support.
///
/// The `Writer` sends mutations through an MPSC channel as `Vec<Mutation>` to enable
/// efficient transaction batching in downstream processors.
///
/// # Batching Strategy
///
/// - Use `send()` to send a batch of mutations at once for maximum efficiency
/// - Use helper methods (`add_node()`, `add_edge()`, etc.) for individual mutations,
///   which are automatically wrapped in `vec![]` before being sent
///
/// # Example
///
/// ```rust,ignore
/// use motlie_db::{Writer, WriterConfig, AddNode, Id, TimestampMilli, Mutation};
///
/// let (writer, receiver) = create_mutation_writer(WriterConfig::default());
///
/// // Send a single mutation using helper method
/// writer.add_node(AddNode {
///     id: Id::new(),
///     name: "Alice".to_string(),
///     ts_millis: TimestampMilli::now(),
///     temporal_range: None,
/// }).await?;
///
/// // Or send multiple mutations in a batch
/// writer.send(vec![
///     Mutation::AddNode(AddNode { /* ... */ }),
///     Mutation::AddEdge(AddEdge { /* ... */ }),
/// ]).await?;
/// ```
#[derive(Debug, Clone)]
pub struct Writer {
    sender: mpsc::Sender<Vec<Mutation>>,
}

impl Writer {
    /// Create a new MutationWriter with the given sender
    pub fn new(sender: mpsc::Sender<Vec<Mutation>>) -> Self {
        Writer { sender }
    }

    /// Send a batch of mutations to be processed.
    ///
    /// Accepts a vector of mutations to enable efficient batching.
    pub async fn send(&self, mutations: Vec<Mutation>) -> Result<()> {
        self.sender
            .send(mutations)
            .await
            .context("Failed to send mutations to writer queue")
    }

    /// Send an AddNode mutation
    pub async fn add_node(&self, args: AddNode) -> Result<()> {
        self.send(vec![Mutation::AddNode(args)]).await
    }

    /// Send an AddEdge mutation
    pub async fn add_edge(&self, args: AddEdge) -> Result<()> {
        self.send(vec![Mutation::AddEdge(args)]).await
    }

    /// Send an AddFragment mutation
    pub async fn add_fragment(&self, args: AddFragment) -> Result<()> {
        self.send(vec![Mutation::AddFragment(args)]).await
    }

    /// Send an Invalidate mutation
    pub async fn invalidate(&self, args: InvalidateArgs) -> Result<()> {
        self.send(vec![Mutation::Invalidate(args)]).await
    }

    /// Check if the writer is still active (receiver hasn't been dropped)
    pub fn is_closed(&self) -> bool {
        self.sender.is_closed()
    }
}

/// Create a new mutation writer and receiver pair
pub fn create_mutation_writer(config: WriterConfig) -> (Writer, mpsc::Receiver<Vec<Mutation>>) {
    let (sender, receiver) = mpsc::channel(config.channel_buffer_size);
    let writer = Writer::new(sender);
    (writer, receiver)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Id, TimestampMilli};
    use tokio::time::Duration;

    #[tokio::test]
    async fn test_writer_closed_detection() {
        let config = WriterConfig::default();
        let (writer, receiver) = create_mutation_writer(config);

        assert!(!writer.is_closed());

        // Drop receiver to close channel
        drop(receiver);

        // Writer should detect channel is closed
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(writer.is_closed());
    }

    #[tokio::test]
    async fn test_writer_send_operations() {
        let config = WriterConfig::default();
        let (writer, _receiver) = create_mutation_writer(config);

        // Test that all send operations work
        let node_args = AddNode {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "test_node".to_string(),
            temporal_range: None,
        };

        let edge_args = AddEdge {
            id: Id::new(),
            source_node_id: Id::new(),
            target_node_id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "test_edge".to_string(),
            temporal_range: None,
        };

        let fragment_args = AddFragment {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            content: crate::DataUrl::from_text("test fragment"),
            temporal_range: None,
        };

        let invalidate_args = InvalidateArgs {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            reason: "test reason".to_string(),
        };

        // These should not panic or error
        writer.add_node(node_args).await.unwrap();
        writer.add_edge(edge_args).await.unwrap();
        writer.add_fragment(fragment_args).await.unwrap();
        writer.invalidate(invalidate_args).await.unwrap();
    }
}
