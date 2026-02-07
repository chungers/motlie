//! Unified writer infrastructure for graph and fulltext mutation subsystems.
//!
//! This module provides the [`Writer`], [`WriterConfig`], and [`WriterBuilder`] types
//! for executing mutations across both graph (RocksDB) and fulltext (Tantivy) backends.
//!
//! # Usage
//!
//! The recommended entry point is [`Storage`](crate::Storage) at the crate root.
//! This module provides the infrastructure types used by `Storage::ready()`.
//!
//! ```ignore
//! use motlie_db::{Storage, StorageConfig};
//! use motlie_db::mutation::{AddNode, Runnable, NodeSummary};
//! use motlie_db::{Id, TimestampMilli};
//!
//! // Create storage in read-write mode
//! // Storage takes a single path and derives <path>/graph and <path>/fulltext
//! let storage = Storage::readwrite(db_path);
//!
//! // ready() returns ReadWriteHandles which guarantees writer() exists
//! let handles = storage.ready(StorageConfig::default())?;
//!
//! // Execute mutations via the writer - no unwrap needed!
//! AddNode {
//!     id: Id::new(),
//!     ts_millis: TimestampMilli::now(),
//!     name: "Alice".to_string(),
//!     summary: NodeSummary::from_text("A person"),
//!     valid_range: None,
//! }
//! .run(handles.writer())  // &Writer, not Option<&Writer>
//! .await?;
//!
//! // Clean shutdown
//! handles.shutdown().await?;
//! ```
//!
//! # Type-Safe Access
//!
//! The unified API uses Rust's type system to ensure type safety:
//!
//! - `Storage::readwrite()` returns `Storage<ReadWrite>`
//! - `Storage<ReadWrite>::ready()` returns [`ReadWriteHandles`](crate::ReadWriteHandles)
//! - `ReadWriteHandles::writer()` returns `&Writer` directly (not `Option`)
//!
//! This eliminates runtime checks and `.unwrap()` calls.
//!
//! # Key Types
//!
//! | Type | Description |
//! |------|-------------|
//! | [`Writer`] | Mutation interface for graph → fulltext pipeline |
//! | [`WriterConfig`] | Configuration for channel buffer sizes |
//! | [`WriterBuilder`] | Builder for creating Writer with infrastructure |
//! | [`Runnable`] | Trait for executing mutations against a writer |
//!
//! # Architecture
//!
//! The unified writer manages a mutation pipeline:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                           Writer (unified)                               │
//! │                                                                          │
//! │  ┌────────────────────────────────────────────────────────────────────┐ │
//! │  │                    Mutation Pipeline                                │ │
//! │  │  User sends: AddNode, AddEdge, AddNodeFragment, AddEdgeFragment,   │ │
//! │  │              UpdateNodeValidSinceUntil, UpdateEdgeValidSinceUntil, │ │
//! │  │              UpdateEdgeWeight, MutationBatch                       │ │
//! │  └────────────────────────────────────────────────────────────────────┘ │
//! │                              │                                           │
//! │                              ▼                                           │
//! │  ┌─────────────────────────────────────────────────────────────────┐    │
//! │  │                    graph::MutationConsumer                       │    │
//! │  │                    (writes to RocksDB)                           │    │
//! │  └───────────────────────────┬─────────────────────────────────────┘    │
//! │                              │ chains to                                 │
//! │                              ▼                                           │
//! │  ┌─────────────────────────────────────────────────────────────────┐    │
//! │  │                  fulltext::MutationConsumer                      │    │
//! │  │                  (indexes in Tantivy)                            │    │
//! │  └─────────────────────────────────────────────────────────────────┘    │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # See Also
//!
//! - [`Storage`](crate::Storage) - Entry point for unified API
//! - [`mutation`](crate::mutation) - Mutation types that work with this writer
//! - [`graph::writer`](crate::graph::writer) - Graph-only writer (advanced use)
//! - [`fulltext::writer`](crate::fulltext::writer) - Fulltext-only consumer (advanced use)

use std::sync::Arc;

use anyhow::Result;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use std::ops::Deref;

use crate::fulltext;
use crate::graph;
use crate::graph::mutation::Mutation;

// ============================================================================
// Runnable Trait - Execute mutations against a Writer
// ============================================================================

/// Trait for mutations that can be executed against a Writer.
///
/// This trait follows the same pattern as the Query API's Runnable trait,
/// enabling mutations to be constructed separately from execution.
///
/// # Examples
///
/// ```rust,ignore
/// use motlie_db::mutation::{AddNode, Runnable};
/// use motlie_db::{Id, TimestampMilli};
///
/// // Construct mutation
/// let mutation = AddNode {
///     id: Id::new(),
///     name: "Alice".to_string(),
///     ts_millis: TimestampMilli::now(),
///     valid_range: None,
/// };
///
/// // Execute it
/// mutation.run(&writer).await?;
/// ```
#[async_trait::async_trait]
pub trait Runnable {
    /// Execute this mutation against the writer
    async fn run(self, writer: &graph::writer::Writer) -> Result<()>;
}

// ============================================================================
// WriterConfig - Composes graph and fulltext configs
// ============================================================================

/// Configuration for the unified writer.
///
/// Composes configurations for both graph and fulltext subsystems.
#[derive(Debug, Clone)]
pub struct WriterConfig {
    /// Configuration for the graph mutation writer
    pub graph: graph::writer::WriterConfig,

    /// Configuration for the fulltext mutation consumer
    pub fulltext: graph::writer::WriterConfig,
}

impl Default for WriterConfig {
    fn default() -> Self {
        Self {
            graph: graph::writer::WriterConfig::default(),
            fulltext: graph::writer::WriterConfig::default(),
        }
    }
}

impl WriterConfig {
    /// Create a config with the same channel buffer size for all subsystems.
    pub fn with_channel_buffer_size(size: usize) -> Self {
        Self {
            graph: graph::writer::WriterConfig {
                channel_buffer_size: size,
            },
            fulltext: graph::writer::WriterConfig {
                channel_buffer_size: size,
            },
        }
    }
}

// ============================================================================
// Writer - Unified mutation interface
// ============================================================================

/// Unified writer for sending mutations to the graph → fulltext pipeline.
///
/// This writer automatically chains mutations through the graph storage
/// (RocksDB) and then to the fulltext indexer (Tantivy). The pipeline
/// setup is an implementation detail - users simply call mutation methods.
///
/// # Usage
///
/// The writer is obtained from [`ReadWriteHandles::writer()`](crate::ReadWriteHandles::writer):
///
/// ```ignore
/// let handles = storage.ready(config)?;
///
/// // Use the Runnable trait for ergonomic mutation execution
/// AddNode { /* ... */ }.run(handles.writer().unwrap()).await?;
/// ```
#[derive(Clone)]
pub struct Writer {
    inner: graph::writer::Writer,
}

impl Writer {
    /// Send a batch of mutations to be processed asynchronously.
    ///
    /// Mutations flow through the graph consumer first (persisted to RocksDB),
    /// then are forwarded to the fulltext consumer (indexed in Tantivy).
    ///
    /// This method returns immediately after enqueueing the mutations.
    /// Use `flush()` to wait for graph mutations to be committed.
    pub async fn send(&self, mutations: Vec<Mutation>) -> Result<()> {
        self.inner.send(mutations).await
    }

    /// Flush all pending mutations and wait for graph commit.
    ///
    /// Returns when all mutations sent before this call are committed to
    /// RocksDB and visible to graph readers.
    ///
    /// # Fulltext Consistency
    ///
    /// This method only guarantees **graph (RocksDB) consistency**.
    /// Fulltext indexing continues asynchronously and may not be complete
    /// when this method returns.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Send mutations
    /// writer.send(vec![AddNode { ... }.into()]).await?;
    /// writer.send(vec![AddEdge { ... }.into()]).await?;
    ///
    /// // Wait for graph commit
    /// writer.flush().await?;
    ///
    /// // Now safe to read from graph
    /// let node = NodeById::new(id).run(&reader, timeout).await?;
    /// ```
    pub async fn flush(&self) -> Result<()> {
        self.inner.flush().await
    }

    /// Send mutations and wait for graph commit.
    ///
    /// This is a convenience method equivalent to `send()` followed by `flush()`.
    /// Returns when all graph mutations are visible to readers.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Send and wait in one call
    /// writer.send_sync(vec![
    ///     AddNode { ... }.into(),
    ///     AddEdge { ... }.into(),
    /// ]).await?;
    ///
    /// // Immediately visible in graph
    /// let node = NodeById::new(id).run(&reader, timeout).await?;
    /// ```
    pub async fn send_sync(&self, mutations: Vec<Mutation>) -> Result<()> {
        self.inner.send_sync(mutations).await
    }

    /// Check if the writer is still active.
    pub fn is_closed(&self) -> bool {
        self.inner.is_closed()
    }

    /// Begin a transaction for read-your-writes semantics.
    ///
    /// The returned Transaction allows interleaved writes and reads
    /// within a single atomic scope.
    ///
    /// See [`graph::writer::Writer::transaction()`](graph::writer::Writer::transaction) for details.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut txn = writer.transaction()?;
    ///
    /// txn.write(AddNode { ... })?;
    /// let result = txn.read(NodeById::new(id, None))?;  // Sees uncommitted AddNode!
    /// txn.write(AddEdge { ... })?;
    ///
    /// txn.commit()?;  // Atomic commit
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if storage is not configured or not in read-write mode.
    pub fn transaction(&self) -> Result<graph::transaction::Transaction<'_>> {
        self.inner.transaction()
    }

    /// Check if transactions are supported by this writer.
    pub fn supports_transactions(&self) -> bool {
        self.inner.supports_transactions()
    }
}

// Deref allows unified Writer to be used where graph::writer::Writer is expected
// This enables: node.run(handle.writer()) via deref coercion
impl Deref for Writer {
    type Target = graph::writer::Writer;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

// ============================================================================
// WriterBuilder - Builder for creating Writer with infrastructure
// ============================================================================

/// Builder for creating a [`Writer`] with all infrastructure.
///
/// This sets up the graph → fulltext consumer chain and returns
/// a Writer along with the JoinHandles for the spawned workers.
///
/// # Usage
///
/// This is typically used internally by [`Storage::ready()`](crate::Storage::ready).
/// For advanced use cases, you can use it directly:
///
/// ```ignore
/// let (writer, handles) = WriterBuilder::new(graph, fulltext)
///     .with_config(config)
///     .build();
/// ```
pub struct WriterBuilder {
    graph: Arc<graph::Processor>,
    fulltext: Arc<fulltext::Index>,
    config: WriterConfig,
}

impl WriterBuilder {
    /// Create a new WriterBuilder.
    ///
    /// # Arguments
    /// * `graph` - The graph processor (RocksDB)
    /// * `fulltext` - The fulltext processor (Tantivy)
    pub fn new(graph: Arc<graph::Processor>, fulltext: Arc<fulltext::Index>) -> Self {
        Self {
            graph,
            fulltext,
            config: WriterConfig::default(),
        }
    }

    /// Set the writer configuration.
    pub fn with_config(mut self, config: WriterConfig) -> Self {
        self.config = config;
        self
    }

    /// Build the Writer and spawn consumer pipeline.
    ///
    /// Creates the graph → fulltext consumer chain:
    /// 1. Writer sends to graph consumer
    /// 2. Graph consumer processes and forwards to fulltext consumer
    /// 3. Fulltext consumer indexes the mutations
    ///
    /// Returns the Writer and JoinHandles for all spawned workers.
    pub fn build(self) -> (Writer, Vec<JoinHandle<Result<()>>>) {
        // Create fulltext consumer channel (end of chain)
        let (fulltext_tx, fulltext_rx) = mpsc::channel(self.config.fulltext.channel_buffer_size);

        // Spawn fulltext consumer
        let fulltext_handle = spawn_fulltext_consumer(fulltext_rx, self.fulltext);

        // Create graph writer and consumer that chains to fulltext
        let (mut graph_writer, graph_rx) =
            graph::writer::create_mutation_writer(self.config.graph.clone());

        // Configure the writer with processor for transaction support
        // and forward transaction mutations to fulltext
        graph_writer.set_processor(self.graph.clone());
        graph_writer.set_transaction_forward_to(fulltext_tx.clone());

        // Spawn graph consumer with chaining to fulltext
        let graph_handle =
            spawn_graph_consumer_with_next(graph_rx, self.config.graph, self.graph, fulltext_tx);

        let writer = Writer { inner: graph_writer };
        let handles = vec![graph_handle, fulltext_handle];

        (writer, handles)
    }
}

// ============================================================================
// Consumer Spawn Functions
// ============================================================================

/// Spawn the graph mutation consumer with chaining to fulltext.
fn spawn_graph_consumer_with_next(
    receiver: mpsc::Receiver<Vec<Mutation>>,
    config: graph::writer::WriterConfig,
    processor: Arc<graph::Processor>,
    next: mpsc::Sender<Vec<Mutation>>,
) -> JoinHandle<Result<()>> {
    // Arc<Processor> implements Processor trait
    let consumer = graph::writer::Consumer::with_next(receiver, config, processor, next);
    graph::writer::spawn_consumer(consumer)
}

/// Spawn the fulltext mutation consumer.
fn spawn_fulltext_consumer(
    receiver: mpsc::Receiver<Vec<Mutation>>,
    index: Arc<fulltext::Index>,
) -> JoinHandle<Result<()>> {
    let config = graph::writer::WriterConfig::default();
    let consumer = graph::writer::Consumer::new(receiver, config, (*index).clone());
    graph::writer::spawn_consumer(consumer)
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Create a [`Writer`] with default configuration.
///
/// This is a convenience function that creates a Writer with the
/// default configuration.
///
/// # Arguments
/// * `graph` - The graph processor (RocksDB)
/// * `fulltext` - The fulltext processor (Tantivy)
///
/// # Returns
/// A tuple of (Writer, handles)
pub fn create_writer(
    graph: Arc<graph::Processor>,
    fulltext: Arc<fulltext::Index>,
) -> (Writer, Vec<JoinHandle<Result<()>>>) {
    WriterBuilder::new(graph, fulltext).build()
}

/// Create a [`Writer`] with custom configuration.
///
/// # Arguments
/// * `graph` - The graph processor (RocksDB)
/// * `fulltext` - The fulltext processor (Tantivy)
/// * `config` - Writer configuration
///
/// # Returns
/// A tuple of (Writer, handles)
pub fn create_writer_with_config(
    graph: Arc<graph::Processor>,
    fulltext: Arc<fulltext::Index>,
    config: WriterConfig,
) -> (Writer, Vec<JoinHandle<Result<()>>>) {
    WriterBuilder::new(graph, fulltext)
        .with_config(config)
        .build()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_writer_config_default() {
        let config = WriterConfig::default();
        assert_eq!(config.graph.channel_buffer_size, 1000);
        assert_eq!(config.fulltext.channel_buffer_size, 1000);
    }

    #[test]
    fn test_writer_config_with_buffer_size() {
        let config = WriterConfig::with_channel_buffer_size(500);
        assert_eq!(config.graph.channel_buffer_size, 500);
        assert_eq!(config.fulltext.channel_buffer_size, 500);
    }
}
