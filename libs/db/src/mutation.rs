use anyhow::{Context, Result};
use tokio::sync::mpsc;

use crate::{Id, WriterConfig};

#[derive(Debug, Clone)]
pub enum Mutation {
    AddVertex(AddVertexArgs),
    AddEdge(AddEdgeArgs),
    AddFragment(AddFragmentArgs),
    Invalidate(InvalidateArgs),
}

#[derive(Debug, Clone)]
pub struct AddVertexArgs {
    /// The UUID of the Vertex
    pub id: Id,

    /// The timestamp as number of milliseconds since the Unix epoch
    pub ts_millis: u64,

    /// The name of the Vertex
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct AddEdgeArgs {
    /// The UUID of the Vertex, Edge, or Fragment
    pub id: Id,

    /// The UUID of the source Vertex
    pub source_vertex_id: Id,

    /// The UUID of the target Vertex
    pub target_vertex_id: Id,

    /// The timestamp as number of milliseconds since the Unix epoch
    pub ts_millis: u64,

    /// The name of the Edge
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct AddFragmentArgs {
    /// The UUID of the Vertex, Edge, or Fragment
    pub id: Id,

    /// The timestamp as number of milliseconds since the Unix epoch
    pub ts_millis: u64,

    /// The body of the Fragment
    pub content: String,
}

#[derive(Debug, Clone)]
pub struct InvalidateArgs {
    /// The UUID of the Vertex, Edge, or Fragment
    pub id: Id,

    /// The timestamp as number of milliseconds since the Unix epoch
    pub ts_millis: u64,

    /// The reason for invalidation
    pub reason: String,
}

/// Trait for processing different types of mutations
#[async_trait::async_trait]
pub trait Processor: Send + Sync {
    /// Process an AddVertex mutation
    async fn process_add_vertex(&self, args: &AddVertexArgs) -> Result<()>;

    /// Process an AddEdge mutation
    async fn process_add_edge(&self, args: &AddEdgeArgs) -> Result<()>;

    /// Process an AddFragment mutation
    async fn process_add_fragment(&self, args: &AddFragmentArgs) -> Result<()>;

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
            Mutation::AddVertex(args) => {
                log::debug!("Processing AddVertex: id={}, name={}", args.id, args.name);
                self.processor.process_add_vertex(args).await?;
            }
            Mutation::AddEdge(args) => {
                log::debug!(
                    "Processing AddEdge: source={}, target={}, name={}",
                    args.source_vertex_id,
                    args.target_vertex_id,
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
        async fn process_add_vertex(&self, _args: &AddVertexArgs) -> Result<()> {
            tokio::time::sleep(Duration::from_millis(1)).await;
            Ok(())
        }

        async fn process_add_edge(&self, _args: &AddEdgeArgs) -> Result<()> {
            tokio::time::sleep(Duration::from_millis(1)).await;
            Ok(())
        }

        async fn process_add_fragment(&self, _args: &AddFragmentArgs) -> Result<()> {
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
        let vertex_args = AddVertexArgs {
            id: Id::new(),
            ts_millis: 1234567890,
            name: "test_vertex".to_string(),
        };
        writer.add_vertex(vertex_args).await.unwrap();

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Drop writer to close channel
        drop(writer);

        // Wait for consumer to finish
        consumer_handle.await.unwrap().unwrap();
    }
}
