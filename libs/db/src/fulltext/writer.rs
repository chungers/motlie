//! Fulltext writer module providing mutation consumer infrastructure.
//!
//! This module contains:
//! - `MutationExecutor` trait - defines how mutations index into Tantivy
//! - `Consumer` - concrete consumer that processes mutations from a channel
//! - Consumer creation and spawn functions for mutation processing
//!
//! The actual `MutationExecutor` implementations for each mutation type
//! are in the `mutation` module (business logic).

use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use tantivy::IndexWriter;

use super::schema::DocumentFields;
use super::{Index, Storage};
use crate::graph::mutation::Mutation;
use crate::graph::writer::{MutationRequest, WriterConfig};
use crate::request::new_request_id;

// ============================================================================
// MutationExecutor Trait
// ============================================================================

/// Trait for mutations to index themselves into Tantivy.
///
/// This trait defines HOW to index the mutation into the fulltext search index.
/// Each mutation type knows how to extract and index its own searchable content.
///
/// Following the same pattern as MutationExecutor for graph storage.
pub trait MutationExecutor: Send + Sync {
    /// Index this mutation into the Tantivy index writer.
    /// Each mutation type knows how to extract and index its searchable content.
    fn index(&self, index_writer: &IndexWriter, fields: &DocumentFields) -> Result<()>;
}

// ============================================================================
// Index::process_mutations - inherent async method
// ============================================================================

impl Index {
    /// Process a batch of mutations - index content for full-text search.
    ///
    /// Requires the Index to be in readwrite mode.
    #[tracing::instrument(skip(self, mutations), fields(mutation_count = mutations.len()))]
    pub async fn process_mutations(&self, mutations: &[Mutation]) -> Result<()> {
        if mutations.is_empty() {
            return Ok(());
        }

        tracing::info!(
            count = mutations.len(),
            "[FullText] Processing mutations for indexing"
        );

        // Get the writer (requires readwrite mode)
        let writer_mutex = self
            .storage
            .writer()
            .ok_or_else(|| anyhow::anyhow!("[FullText] Index not in readwrite mode"))?;

        let fields = self.storage.fields()?;
        let mut writer = writer_mutex.lock().await;

        // Index each mutation
        for mutation in mutations {
            match mutation {
                Mutation::AddNode(m) => m.index(&mut writer, fields)?,
                Mutation::AddEdge(m) => m.index(&mut writer, fields)?,
                Mutation::AddNodeFragment(m) => m.index(&mut writer, fields)?,
                Mutation::AddEdgeFragment(m) => m.index(&mut writer, fields)?,
                // CONTENT-ADDRESS: Update/Delete mutations
                Mutation::UpdateNode(m) => m.index(&mut writer, fields)?,
                Mutation::UpdateEdge(m) => m.index(&mut writer, fields)?,
                Mutation::DeleteNode(m) => m.index(&mut writer, fields)?,
                Mutation::DeleteEdge(m) => m.index(&mut writer, fields)?,
                // Restore mutations don't need fulltext indexing - they restore from existing summaries
                // which are already indexed. The restored summary hash points to existing content.
                Mutation::RestoreNode(_) => {}
                Mutation::RestoreEdge(_) => {}
                // Flush is graph-only - no-op for fulltext
                Mutation::Flush(_) => {}
            }
        }

        // Commit the batch atomically
        writer.commit().context("Failed to commit batch to index")?;

        tracing::info!(
            count = mutations.len(),
            "[FullText] Successfully indexed mutations"
        );
        Ok(())
    }
}

// ============================================================================
// Helper Functions for Creating Storage and Index
// ============================================================================

/// Create a readwrite Index from a path (convenience function).
///
/// This handles the full setup: Storage::readwrite -> ready -> Arc -> Index.
/// Use this for mutation consumers.
pub(super) fn create_readwrite_index(index_path: &Path) -> Index {
    let mut storage = Storage::readwrite(index_path);
    storage.ready().expect("Failed to ready storage");
    Index::new(Arc::new(storage))
}

/// Create a readonly Index from a path (convenience function).
///
/// This handles the full setup: Storage::readonly -> ready -> Arc -> Index.
/// Use this for query consumers.
#[allow(dead_code)]
pub(super) fn create_readonly_index(index_path: &Path) -> Index {
    let mut storage = Storage::readonly(index_path);
    storage.ready().expect("Failed to ready storage");
    Index::new(Arc::new(storage))
}

// ============================================================================
// Consumer
// ============================================================================

/// Concrete consumer that processes fulltext mutations from a channel.
///
/// The consumer receives mutation batches and delegates to an Index
/// for Tantivy indexing operations.
pub struct Consumer {
    receiver: mpsc::Receiver<MutationRequest>,
    config: WriterConfig,
    index: Index,
    /// Optional sender to forward mutations to the next consumer in the chain
    next: Option<mpsc::Sender<MutationRequest>>,
}

impl Consumer {
    /// Create a new Consumer.
    pub fn new(
        receiver: mpsc::Receiver<MutationRequest>,
        config: WriterConfig,
        index: Index,
    ) -> Self {
        Self {
            receiver,
            config,
            index,
            next: None,
        }
    }

    /// Create a new Consumer that forwards mutations to the next consumer in the chain.
    pub fn with_next(
        receiver: mpsc::Receiver<MutationRequest>,
        config: WriterConfig,
        index: Index,
        next: mpsc::Sender<MutationRequest>,
    ) -> Self {
        Self {
            receiver,
            config,
            index,
            next: Some(next),
        }
    }

    /// Process mutations continuously until the channel is closed.
    #[tracing::instrument(skip(self), name = "fulltext_mutation_consumer")]
    pub async fn run(mut self) -> Result<()> {
        tracing::info!(config = ?self.config, "Starting fulltext mutation consumer");

        loop {
            match self.receiver.recv().await {
                Some(request) => {
                    self.process_request(request).await?;
                }
                None => {
                    tracing::info!("Fulltext mutation consumer shutting down - channel closed");
                    return Ok(());
                }
            }
        }
    }

    /// Process a batch of mutations.
    #[tracing::instrument(skip(self, request), fields(batch_size = request.payload.len()))]
    async fn process_request(&self, request: MutationRequest) -> Result<()> {
        let MutationRequest {
            payload: mutations,
            options: _,
            reply: _,
            ..
        } = request;

        // Process all mutations
        self.index
            .process_mutations(&mutations)
            .await
            .with_context(|| format!("Failed to index {} mutations", mutations.len()))?;

        // Signal completion for any flush markers
        for mutation in &mutations {
            if let Mutation::Flush(marker) = mutation {
                if let Some(completion) = marker.take_completion() {
                    let _ = completion.send(());
                }
            }
        }

        // Forward to next consumer if configured
        if let Some(sender) = &self.next {
            let non_flush_mutations: Vec<_> = mutations
                .iter()
                .filter(|m: &&Mutation| !m.is_flush())
                .cloned()
                .collect();

            if !non_flush_mutations.is_empty() {
                if let Err(e) = sender.try_send(MutationRequest {
                    payload: non_flush_mutations,
                    options: Default::default(),
                    reply: None,
                    timeout: None,
                    request_id: new_request_id(),
                    created_at: std::time::Instant::now(),
                }) {
                    tracing::warn!(
                        err = %e,
                        count = mutations.len(),
                        "[BUFFER FULL] Next consumer busy, dropping mutations"
                    );
                }
            }
        }

        Ok(())
    }
}

/// Spawn a fulltext mutation consumer as a background task.
pub fn spawn_consumer(consumer: Consumer) -> JoinHandle<Result<()>> {
    tokio::spawn(async move { consumer.run().await })
}

// ============================================================================
// Consumer Creation Functions
// ============================================================================

/// Create a new full-text mutation consumer with default parameters
pub fn create_mutation_consumer(
    receiver: mpsc::Receiver<MutationRequest>,
    config: WriterConfig,
    index_path: &Path,
) -> Consumer {
    let index = create_readwrite_index(index_path);
    Consumer::new(receiver, config, index)
}

/// Create a new full-text mutation consumer with default parameters that chains to another processor
pub fn create_mutation_consumer_with_next(
    receiver: mpsc::Receiver<MutationRequest>,
    config: WriterConfig,
    index_path: &Path,
    next: mpsc::Sender<MutationRequest>,
) -> Consumer {
    let index = create_readwrite_index(index_path);
    Consumer::with_next(receiver, config, index, next)
}

/// Create a new full-text mutation consumer with custom BM25 parameters
pub fn create_mutation_consumer_with_params(
    receiver: mpsc::Receiver<MutationRequest>,
    config: WriterConfig,
    index_path: &Path,
    _k1: f32,
    _b: f32,
) -> Consumer {
    // Note: BM25 params not currently used - Tantivy uses defaults
    let index = create_readwrite_index(index_path);
    Consumer::new(receiver, config, index)
}

/// Create a new full-text mutation consumer with custom BM25 parameters that chains to another processor
pub fn create_mutation_consumer_with_params_and_next(
    receiver: mpsc::Receiver<MutationRequest>,
    config: WriterConfig,
    index_path: &Path,
    _k1: f32,
    _b: f32,
    next: mpsc::Sender<MutationRequest>,
) -> Consumer {
    // Note: BM25 params not currently used - Tantivy uses defaults
    let index = create_readwrite_index(index_path);
    Consumer::with_next(receiver, config, index, next)
}

// ============================================================================
// Spawn Functions
// ============================================================================

/// Spawn the full-text mutation consumer as a background task with default parameters
pub fn spawn_mutation_consumer(
    receiver: mpsc::Receiver<MutationRequest>,
    config: WriterConfig,
    index_path: &Path,
) -> JoinHandle<Result<()>> {
    let consumer = create_mutation_consumer(receiver, config, index_path);
    spawn_consumer(consumer)
}

/// Spawn the full-text mutation consumer as a background task with default parameters and chaining.
///
/// This follows the same pattern as `spawn_mutation_consumer_with_next` - the consumer processes
/// mutations and then forwards them to the next consumer in the chain.
///
/// # Example
/// ```no_run
/// use motlie_db::graph::writer::{create_mutation_writer, WriterConfig};
/// use motlie_db::fulltext::spawn_mutation_consumer_with_next;
/// use std::path::Path;
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = WriterConfig { channel_buffer_size: 100 };
/// let (writer, mutation_rx) = create_mutation_writer(config.clone());
///
/// // Create channel for next consumer in chain
/// let (next_tx, next_rx) = tokio::sync::mpsc::channel(100);
///
/// // Fulltext consumer processes mutations then forwards to next_tx
/// let handle = spawn_mutation_consumer_with_next(
///     mutation_rx,
///     config,
///     Path::new("/data/fulltext"),
///     next_tx,
/// );
/// # Ok(())
/// # }
/// ```
pub fn spawn_mutation_consumer_with_next(
    receiver: mpsc::Receiver<MutationRequest>,
    config: WriterConfig,
    index_path: &Path,
    next: mpsc::Sender<MutationRequest>,
) -> JoinHandle<Result<()>> {
    let consumer = create_mutation_consumer_with_next(receiver, config, index_path, next);
    spawn_consumer(consumer)
}

/// Spawn the full-text mutation consumer as a background task with custom BM25 parameters
pub fn spawn_mutation_consumer_with_params(
    receiver: mpsc::Receiver<MutationRequest>,
    config: WriterConfig,
    index_path: &Path,
    k1: f32,
    b: f32,
) -> JoinHandle<Result<()>> {
    let consumer = create_mutation_consumer_with_params(receiver, config, index_path, k1, b);
    spawn_consumer(consumer)
}

/// Spawn the full-text mutation consumer as a background task with custom BM25 parameters and chaining
pub fn spawn_mutation_consumer_with_params_and_next(
    receiver: mpsc::Receiver<MutationRequest>,
    config: WriterConfig,
    index_path: &Path,
    k1: f32,
    b: f32,
    next: mpsc::Sender<MutationRequest>,
) -> JoinHandle<Result<()>> {
    let consumer =
        create_mutation_consumer_with_params_and_next(receiver, config, index_path, k1, b, next);
    spawn_consumer(consumer)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::mutation::{AddNode, AddNodeFragment};
    use crate::writer::Runnable as MutRunnable;
    use crate::graph::writer::create_mutation_writer;
    use crate::{DataUrl, Id, TimestampMilli};
    use std::time::Duration;
    use tantivy::collector::TopDocs;
    use tantivy::query::QueryParser;

    #[test]
    fn test_fulltext_processor_creation() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");

        let processor = create_readwrite_index(&index_path);

        assert_eq!(processor.index_path(), index_path);
        assert!(index_path.exists());
    }

    #[test]
    fn test_fulltext_processor_with_custom_params() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");

        // BM25 params don't affect creation - just verify setup works
        let processor = create_readwrite_index(&index_path);

        // Just verify it creates successfully
        assert!(processor.index_path().exists());
    }

    #[tokio::test]
    async fn test_fulltext_consumer_integration() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");

        let config = WriterConfig {
            channel_buffer_size: 10,
        };

        let (writer, receiver) = create_mutation_writer(config.clone());

        // Spawn consumer
        let consumer_handle = spawn_mutation_consumer(receiver, config, &index_path);

        // Send some mutations
        let node_args = AddNode {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "test_node".to_string(),
            valid_range: None,
            summary: crate::graph::schema::NodeSummary::from_text("test summary"),
        };
        node_args.run(&writer).await.unwrap();

        let fragment_args = AddNodeFragment {
            id: Id::new(),
            ts_millis: TimestampMilli(1234567890),
            content: DataUrl::from_text("This is a test fragment with some searchable content"),
            valid_range: None,
        };
        fragment_args.run(&writer).await.unwrap();

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Drop writer to close channel
        drop(writer);

        // Wait for consumer to finish
        consumer_handle.await.unwrap().unwrap();

        // Verify the documents are indexed
        let processor = create_readwrite_index(&index_path);
        let reader = processor.tantivy_index().reader().unwrap();
        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(
            processor.tantivy_index(),
            vec![
                processor.fields().node_name_field,
                processor.fields().content_field,
            ],
        );

        // Search for node
        let query = query_parser.parse_query("test_node").unwrap();
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10)).unwrap();
        assert_eq!(top_docs.len(), 1);

        // Search for fragment content
        let query = query_parser.parse_query("searchable").unwrap();
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10)).unwrap();
        assert_eq!(top_docs.len(), 1);
    }

    #[tokio::test]
    async fn test_empty_mutations_batch() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");
        let processor = create_readwrite_index(&index_path);

        // Processing empty batch should not error
        let result = processor.process_mutations(&[]).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_index_persistence() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");

        // Create processor and add a document
        {
            let processor = create_readwrite_index(&index_path);
            let node = AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                name: "persistent_node".to_string(),
                valid_range: None,
                summary: crate::graph::schema::NodeSummary::from_text("persistent summary"),
            };
            processor
                .process_mutations(&[Mutation::AddNode(node)])
                .await
                .unwrap();
        } // Drop processor to close the index

        // Open again and verify the document is still there
        let processor = create_readwrite_index(&index_path);
        let reader = processor.tantivy_index().reader().unwrap();
        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(
            processor.tantivy_index(),
            vec![processor.fields().node_name_field],
        );
        let query = query_parser.parse_query("persistent_node").unwrap();
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10)).unwrap();

        assert_eq!(top_docs.len(), 1);
    }
}
