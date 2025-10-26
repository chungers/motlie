use anyhow::{Context, Result};
use tokio::sync::mpsc;

use crate::{AddEdgeArgs, AddFragmentArgs, AddVertexArgs, InvalidateArgs, Mutation, WriterConfig};

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

    /// Process mutations in a loop until the channel is closed
    pub async fn run(mut self) -> Result<()> {
        log::info!("Starting mutation consumer with config: {:?}", self.config);

        let mut batch = Vec::with_capacity(self.config.max_batch_size);
        let batch_timeout = tokio::time::Duration::from_millis(self.config.batch_timeout_ms);

        loop {
            // Wait for mutations with timeout for batching
            let timeout_result = tokio::time::timeout(batch_timeout, self.receiver.recv()).await;

            match timeout_result {
                // Received a mutation within timeout
                Ok(Some(mutation)) => {
                    batch.push(mutation);

                    // Continue collecting mutations until batch is full or channel is empty
                    while batch.len() < self.config.max_batch_size {
                        match self.receiver.try_recv() {
                            Ok(mutation) => batch.push(mutation),
                            Err(mpsc::error::TryRecvError::Empty) => break,
                            Err(mpsc::error::TryRecvError::Disconnected) => {
                                // Process final batch and exit
                                if !batch.is_empty() {
                                    self.process_batch(&batch).await?;
                                }
                                log::info!(
                                    "Mutation consumer shutting down - channel disconnected"
                                );
                                return Ok(());
                            }
                        }
                    }

                    // Process the batch
                    if !batch.is_empty() {
                        self.process_batch(&batch).await?;
                        batch.clear();
                    }
                }
                // Channel closed
                Ok(None) => {
                    // Process any remaining mutations
                    if !batch.is_empty() {
                        self.process_batch(&batch).await?;
                    }
                    log::info!("Mutation consumer shutting down - channel closed");
                    return Ok(());
                }
                // Timeout - process accumulated batch if any
                Err(_) => {
                    if !batch.is_empty() {
                        self.process_batch(&batch).await?;
                        batch.clear();
                    }
                }
            }
        }
    }

    /// Process a batch of mutations
    async fn process_batch(&self, batch: &[Mutation]) -> Result<()> {
        log::debug!("Processing batch of {} mutations", batch.len());

        for mutation in batch {
            self.process_mutation(mutation)
                .await
                .with_context(|| format!("Failed to process mutation: {:?}", mutation))?;
        }

        log::debug!("Successfully processed batch of {} mutations", batch.len());
        Ok(())
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
                    args.body.len()
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
            max_batch_size: 2,
            batch_timeout_ms: 50,
        };

        let (writer, receiver) = {
            let (sender, receiver) = mpsc::channel(config.channel_buffer_size);
            let writer = crate::MutationWriter::new(sender);
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
