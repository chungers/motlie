use anyhow::{Context, Result};
use tokio::sync::mpsc;

use crate::{Id, TimestampMilli, WriterConfig};

#[derive(Debug, Clone)]
pub enum Mutation {
    AddNode(AddNode),
    AddEdge(AddEdge),
    AddFragment(AddFragment),
    Invalidate(InvalidateArgs),
}

#[derive(Debug, Clone)]
pub struct AddNode {
    /// The UUID of the Node
    pub id: Id,

    /// The timestamp as number of milliseconds since the Unix epoch
    pub ts_millis: TimestampMilli,

    /// The name of the Node
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct AddEdge {
    /// The UUID of the Node, Edge, or Fragment
    pub id: Id,

    /// The UUID of the source Node
    pub source_node_id: Id,

    /// The UUID of the target Node
    pub target_node_id: Id,

    /// The timestamp as number of milliseconds since the Unix epoch
    pub ts_millis: TimestampMilli,

    /// The name of the Edge
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct AddFragment {
    /// The UUID of the Node, Edge, or Fragment
    pub id: Id,

    /// The timestamp as number of milliseconds since the Unix epoch
    pub ts_millis: u64,

    /// The body of the Fragment
    /// TODO - support image data url (e.g. base64 encoded image -
    /// "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg==")
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct InvalidateArgs {
    /// The UUID of the Node, Edge, or Fragment
    pub id: Id,

    /// The timestamp as number of milliseconds since the Unix epoch
    pub ts_millis: TimestampMilli,

    /// The reason for invalidation
    pub reason: String,
}

/// Trait for processing different types of mutations
#[async_trait::async_trait]
pub trait Processor: Send + Sync {
    /// Process an AddNode mutation
    async fn process_add_node(&self, args: &AddNode) -> Result<()>;

    /// Process an AddEdge mutation
    async fn process_add_edge(&self, args: &AddEdge) -> Result<()>;

    /// Process an AddFragment mutation
    async fn process_add_fragment(&self, args: &AddFragment) -> Result<()>;

    /// Process an Invalidate mutation
    async fn process_invalidate(&self, args: &InvalidateArgs) -> Result<()>;
}

/// Generic consumer that processes mutations using a Processor
pub struct Consumer<P: Processor> {
    receiver: mpsc::Receiver<Mutation>,
    config: WriterConfig,
    processor: P,
    /// Optional sender to forward mutations to the next consumer in the chain
    next: Option<mpsc::Sender<Mutation>>,
}

impl<P: Processor> Consumer<P> {
    /// Create a new Consumer
    pub fn new(receiver: mpsc::Receiver<Mutation>, config: WriterConfig, processor: P) -> Self {
        Self {
            receiver,
            config,
            processor,
            next: None,
        }
    }

    /// Create a new Consumer that forwards mutations to the next consumer in the chain
    pub fn with_next(
        receiver: mpsc::Receiver<Mutation>,
        config: WriterConfig,
        processor: P,
        next: mpsc::Sender<Mutation>,
    ) -> Self {
        Self {
            receiver,
            config,
            processor,
            next: Some(next),
        }
    }

    /// Process mutations continuously until the channel is closed
    pub async fn run(mut self) -> Result<()> {
        log::info!("Starting mutation consumer with config: {:?}", self.config);

        loop {
            // Wait for the next mutation
            match self.receiver.recv().await {
                Some(mutation) => {
                    // Process the mutation immediately
                    self.process_mutation(&mutation)
                        .await
                        .with_context(|| format!("Failed to process mutation: {:?}", mutation))?;
                }
                None => {
                    // Channel closed
                    log::info!("Mutation consumer shutting down - channel closed");
                    return Ok(());
                }
            }
        }
    }

    /// Process a single mutation
    async fn process_mutation(&self, mutation: &Mutation) -> Result<()> {
        // Process the mutation with the processor
        match mutation {
            Mutation::AddNode(args) => {
                log::debug!("Processing AddNode: id={}, name={}", args.id, args.name);
                self.processor.process_add_node(args).await?;
            }
            Mutation::AddEdge(args) => {
                log::debug!(
                    "Processing AddEdge: source={}, target={}, name={}",
                    args.source_node_id,
                    args.target_node_id,
                    args.name
                );
                self.processor.process_add_edge(args).await?;
            }
            Mutation::AddFragment(args) => {
                log::debug!(
                    "Processing AddFragment: id={}, body_len={}",
                    args.id,
                    args.content.len()
                );
                self.processor.process_add_fragment(args).await?;
            }
            Mutation::Invalidate(args) => {
                log::debug!(
                    "Processing Invalidate: id={}, reason={}",
                    args.id,
                    args.reason
                );
                self.processor.process_invalidate(args).await?;
            }
        }

        // Forward the mutation to the next consumer in the chain if configured
        if let Some(sender) = &self.next {
            sender.send(mutation.clone()).await.map_err(|e| {
                anyhow::anyhow!("Failed to forward mutation to next consumer: {}", e)
            })?;
        }

        Ok(())
    }
}

/// Spawn a mutation consumer as a background task
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
        async fn process_add_node(&self, _args: &AddNode) -> Result<()> {
            tokio::time::sleep(Duration::from_millis(1)).await;
            Ok(())
        }

        async fn process_add_edge(&self, _args: &AddEdge) -> Result<()> {
            tokio::time::sleep(Duration::from_millis(1)).await;
            Ok(())
        }

        async fn process_add_fragment(&self, _args: &AddFragment) -> Result<()> {
            tokio::time::sleep(Duration::from_millis(1)).await;
            Ok(())
        }

        async fn process_invalidate(&self, _args: &InvalidateArgs) -> Result<()> {
            tokio::time::sleep(Duration::from_millis(1)).await;
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_generic_consumer_basic() {
        let config = crate::WriterConfig {
            channel_buffer_size: 10,
        };

        let (writer, receiver) = {
            let (sender, receiver) = mpsc::channel(config.channel_buffer_size);
            let writer = crate::Writer::new(sender);
            (writer, receiver)
        };

        let processor = TestProcessor;
        let consumer = Consumer::new(receiver, config, processor);

        // Spawn consumer
        let consumer_handle = spawn_consumer(consumer);

        // Send a mutation
        let node_args = AddNode {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "test_node".to_string(),
        };
        writer.add_node(node_args).await.unwrap();

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Drop writer to close channel
        drop(writer);

        // Wait for consumer to finish
        consumer_handle.await.unwrap().unwrap();
    }
}
