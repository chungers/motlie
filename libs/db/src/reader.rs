//! Unified reader infrastructure for graph and fulltext query subsystems.
//!
//! This module provides the [`Reader`], [`ReaderConfig`], and [`ReaderBuilder`] types
//! for executing queries across both graph (RocksDB) and fulltext (Tantivy) backends.
//!
//! # Usage
//!
//! The recommended entry point is [`Storage`](crate::Storage) at the crate root.
//! This module provides the infrastructure types used by `Storage::ready()`.
//!
//! ```ignore
//! use motlie_db::{Storage, StorageConfig};
//! use motlie_db::query::{Nodes, Runnable};
//! use std::time::Duration;
//!
//! // Read-only mode: Storage::readonly() → ReadOnlyHandles
//! // Storage takes a single path and derives <path>/graph and <path>/fulltext
//! let storage = Storage::readonly(db_path);
//! let handles = storage.ready(StorageConfig::default())?;
//!
//! // Execute queries via the reader
//! let timeout = Duration::from_secs(5);
//! let results = Nodes::new("query".to_string(), 10)
//!     .run(handles.reader(), timeout)
//!     .await?;
//!
//! // Read-write mode: Storage::readwrite() → ReadWriteHandles
//! // Both reader() and writer() are available
//! let storage = Storage::readwrite(db_path);
//! let handles = storage.ready(StorageConfig::default())?;
//! let results = Nodes::new("query".to_string(), 10)
//!     .run(handles.reader(), timeout)
//!     .await?;
//! ```
//!
//! # Key Types
//!
//! | Type | Description |
//! |------|-------------|
//! | [`Reader`] | Query interface for both graph and fulltext |
//! | [`ReaderConfig`] | Configuration for channel buffer sizes |
//! | [`ReaderBuilder`] | Builder for creating Reader with infrastructure |
//! | [`Runnable`] | Trait for executing queries against a reader |
//! | [`CompositeStorage`] | Internal storage reference for query execution |
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                           Reader (unified)                               │
//! │                                                                          │
//! │  ┌────────────────────────────────────────────────────────────────────┐ │
//! │  │                    Unified Query Pipeline                           │ │
//! │  │  Handles: Nodes, Edges (fulltext search + graph hydration)          │ │
//! │  │           NodeById, OutgoingEdges, IncomingEdges (forwarded)        │ │
//! │  └────────────────────────────────────────────────────────────────────┘ │
//! │                              │                                           │
//! │              ┌───────────────┴───────────────┐                          │
//! │              ▼                               ▼                          │
//! │  ┌─────────────────────────┐   ┌─────────────────────────────────┐     │
//! │  │    graph::Reader        │   │       fulltext::Reader          │     │
//! │  │  (graph queries)        │   │   (raw fulltext queries)        │     │
//! │  └───────────┬─────────────┘   └───────────────┬─────────────────┘     │
//! │              │                                 │                        │
//! │              ▼                                 ▼                        │
//! │  ┌─────────────────────────┐   ┌─────────────────────────────────┐     │
//! │  │  Consumer Pool (MPMC)   │   │   Consumer Pool (MPMC)          │     │
//! │  │  (N async workers)      │   │   (N async workers)             │     │
//! │  └───────────┬─────────────┘   └───────────────┬─────────────────┘     │
//! │              │                                 │                        │
//! │              ▼                                 ▼                        │
//! │  ┌─────────────────────────┐   ┌─────────────────────────────────┐     │
//! │  │   graph::Processor      │   │      fulltext::Index            │     │
//! │  │   (RocksDB)             │   │      (Tantivy)                  │     │
//! │  └─────────────────────────┘   └─────────────────────────────────┘     │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # See Also
//!
//! - [`Storage`](crate::Storage) - Entry point for unified API
//! - [`query`](crate::query) - Query types that work with this reader
//! - [`graph::reader`](crate::graph::reader) - Graph-only reader (advanced use)
//! - [`fulltext::reader`](crate::fulltext::reader) - Fulltext-only reader (advanced use)

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use tokio::task::JoinHandle;

use crate::fulltext;
use crate::graph;
use crate::query::QueryRequest;

// ============================================================================
// ReaderConfig - Composes graph and fulltext configs
// ============================================================================

/// Configuration for the unified reader.
///
/// Composes configurations for both graph and fulltext subsystems.
#[derive(Debug, Clone)]
pub struct ReaderConfig {
    /// Configuration for the graph query reader
    pub graph: graph::reader::ReaderConfig,

    /// Configuration for the fulltext query reader
    pub fulltext: fulltext::reader::ReaderConfig,

    /// Channel buffer size for unified search queries
    pub channel_buffer_size: usize,
}

impl Default for ReaderConfig {
    fn default() -> Self {
        Self {
            graph: graph::reader::ReaderConfig::default(),
            fulltext: fulltext::reader::ReaderConfig::default(),
            channel_buffer_size: 1000,
        }
    }
}

impl ReaderConfig {
    /// Create a config with the same channel buffer size for all subsystems.
    pub fn with_channel_buffer_size(size: usize) -> Self {
        Self {
            graph: graph::reader::ReaderConfig {
                channel_buffer_size: size,
            },
            fulltext: fulltext::reader::ReaderConfig {
                channel_buffer_size: size,
            },
            channel_buffer_size: size,
        }
    }
}

// ============================================================================
// Runnable Trait
// ============================================================================

/// Trait for query builders that can be executed via a Reader.
///
/// This trait is generic over the reader type `R`, which allows the same query type
/// (e.g., `Nodes`) to be executed against different readers with different return types:
///
/// - `Runnable<fulltext::Reader>` → returns raw fulltext hits (e.g., `Vec<NodeHit>`)
/// - `Runnable<reader::Reader>` → returns hydrated graph data (e.g., `Vec<NodeResult>`)
///
/// # Example
///
/// ```ignore
/// use motlie_db::fulltext::{Nodes, Runnable, Reader as FulltextReader};
/// use motlie_db::reader::Reader as UnifiedReader;
///
/// // Same Nodes type, different output based on reader
/// let hits: Vec<NodeHit> = Nodes::new("rust", 10).run(&fulltext_reader, timeout).await?;
/// let results: Vec<NodeResult> = Nodes::new("rust", 10).run(&unified_reader, timeout).await?;
/// ```
#[async_trait::async_trait]
pub trait Runnable<R> {
    /// The output type this query produces
    type Output: Send + 'static;

    /// Execute this query against the specified reader with the given timeout
    async fn run(self, reader: &R, timeout: Duration) -> Result<Self::Output>;
}

// ============================================================================
// CompositeStorage - Internal storage reference for query execution
// ============================================================================

/// Internal storage reference combining graph and fulltext for query execution.
///
/// This is used by the consumer pools to execute queries. It holds Arc references
/// to the initialized storage backends.
pub(crate) struct CompositeStorage {
    /// Graph processor (RocksDB) - source of truth for node/edge data
    pub(crate) graph: Arc<graph::Processor>,

    /// Fulltext storage (Tantivy) - for search and ranking
    pub(crate) fulltext: Arc<fulltext::Index>,
}

impl CompositeStorage {
    /// Create a new CompositeStorage from graph and fulltext components.
    pub(crate) fn new(graph: Arc<graph::Processor>, fulltext: Arc<fulltext::Index>) -> Self {
        Self { graph, fulltext }
    }
}

// ============================================================================
// Reader - Unified query interface
// ============================================================================

/// Unified reader managing both graph and fulltext query subsystems.
///
/// Provides a single interface for executing queries, routing them to the
/// appropriate backend based on query type.
///
/// # Usage
///
/// The Reader is obtained from [`ReadOnlyHandles::reader()`](crate::ReadOnlyHandles::reader)
/// or [`ReadWriteHandles::reader()`](crate::ReadWriteHandles::reader):
///
/// ```ignore
/// let handles = storage.ready(config)?;
/// let reader = handles.reader();
///
/// // Execute queries
/// let results = Nodes::new("query", 10).run(reader, timeout).await?;
/// ```
#[derive(Clone)]
pub struct Reader {
    /// Sender for unified search queries (Nodes, Edges with hydration)
    sender: flume::Sender<QueryRequest>,

    /// Graph reader for direct graph queries (NodeById, OutgoingEdges, etc.)
    graph_reader: graph::Reader,

    /// Fulltext reader for search queries (raw fulltext hits)
    fulltext_reader: fulltext::Reader,

    /// Shared storage for hydration (fulltext hits → graph data)
    composite_storage: Arc<CompositeStorage>,
}

impl Reader {
    /// Send a unified search query to the consumer pool.
    pub async fn send_query(&self, query: QueryRequest) -> Result<()> {
        self.sender
            .send_async(query)
            .await
            .map_err(|e| anyhow::anyhow!("Failed to send query: {}", e))
    }

    /// Get the graph reader for direct graph queries.
    pub fn graph(&self) -> &graph::Reader {
        &self.graph_reader
    }

    /// Get the fulltext reader for fulltext search queries.
    pub fn fulltext(&self) -> &fulltext::Reader {
        &self.fulltext_reader
    }

    /// Get the shared composite storage for query execution.
    pub fn storage(&self) -> &CompositeStorage {
        &self.composite_storage
    }

    /// Check if any reader is closed.
    pub fn is_closed(&self) -> bool {
        self.sender.is_disconnected()
            || self.graph_reader.is_closed()
            || self.fulltext_reader.is_closed()
    }
}

// ============================================================================
// ReaderBuilder - Builder for creating unified Reader with infrastructure
// ============================================================================

/// Builder for creating a unified Reader with all infrastructure.
///
/// This sets up both graph and fulltext consumer pools and returns
/// a Reader along with the JoinHandles for the spawned workers.
///
/// # Usage
///
/// This is typically used internally by [`Storage::ready()`](crate::Storage::ready).
/// For advanced use cases, you can use it directly:
///
/// ```ignore
/// let (reader, unified_handles, graph_handles, fulltext_handles) =
///     ReaderBuilder::new(graph, fulltext)
///         .with_config(config)
///         .with_num_workers(4)
///         .build();
/// ```
pub struct ReaderBuilder {
    graph_storage: Arc<graph::Storage>,
    fulltext: Arc<fulltext::Index>,
    config: ReaderConfig,
    num_workers: usize,
}

impl ReaderBuilder {
    /// Create a new ReaderBuilder.
    ///
    /// # Arguments
    /// * `graph_storage` - The graph storage (RocksDB)
    /// * `fulltext` - The fulltext index (Tantivy)
    pub fn new(graph_storage: Arc<graph::Storage>, fulltext: Arc<fulltext::Index>) -> Self {
        Self {
            graph_storage,
            fulltext,
            config: ReaderConfig::default(),
            num_workers: 4,
        }
    }

    /// Set the reader configuration.
    pub fn with_config(mut self, config: ReaderConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the number of worker tasks for each subsystem.
    pub fn with_num_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = num_workers;
        self
    }

    /// Build the Reader and spawn consumer pools.
    ///
    /// Returns the Reader and JoinHandles for all spawned workers.
    /// The handles are grouped as (unified_handles, graph_handles, fulltext_handles).
    pub fn build(
        self,
    ) -> (
        Reader,
        Vec<JoinHandle<()>>,
        Vec<JoinHandle<()>>,
        Vec<JoinHandle<()>>,
    ) {
        // Create graph processor from storage
        let graph_processor = Arc::new(graph::Processor::new(self.graph_storage));

        // Create shared composite storage
        let composite_storage = Arc::new(CompositeStorage::new(
            graph_processor.clone(),
            self.fulltext.clone(),
        ));

        // Create graph reader and consumer pool
        let (graph_reader, graph_receiver) =
            graph::reader::create_reader_with_processor(graph_processor.clone(), self.config.graph.clone());
        let graph_handles = graph::reader::spawn_consumer_pool_with_processor(
            graph_receiver,
            graph_processor,
            self.num_workers,
        );

        // Create fulltext reader and consumer pool
        let (fulltext_reader, fulltext_receiver) =
            fulltext::reader::create_query_reader(self.config.fulltext.clone());
        let fulltext_handles = fulltext::reader::spawn_query_consumer_pool_shared(
            fulltext_receiver,
            self.fulltext.clone(),
            self.num_workers,
        );

        // Create unified search channel and consumer pool
        let (sender, receiver) = flume::bounded(self.config.channel_buffer_size);
        let unified_handles =
            spawn_consumer_pool(receiver, composite_storage.clone(), self.num_workers);

        let reader = Reader {
            sender,
            graph_reader,
            fulltext_reader,
            composite_storage,
        };

        (reader, unified_handles, graph_handles, fulltext_handles)
    }
}

// ============================================================================
// Consumer Pool for Unified Queries
// ============================================================================

/// Spawn a pool of consumers for unified queries.
///
/// Each consumer processes Query instances by executing them against the
/// composite storage (fulltext search + graph hydration).
#[doc(hidden)]
pub fn spawn_consumer_pool(
    receiver: flume::Receiver<QueryRequest>,
    storage: Arc<CompositeStorage>,
    num_workers: usize,
) -> Vec<JoinHandle<()>> {
    (0..num_workers)
        .map(|_| {
            let receiver = receiver.clone();
            let storage = storage.clone();
            tokio::spawn(async move {
                while let Ok(mut request) = receiver.recv_async().await {
                    let exec = request.payload.execute(&storage);
                    let result = match request.timeout {
                        Some(timeout) => match tokio::time::timeout(timeout, exec).await {
                            Ok(r) => r,
                            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
                        },
                        None => exec.await,
                    };
                    request.respond(result);
                }
            })
        })
        .collect()
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Create a unified Reader with default configuration.
///
/// This is a convenience function that creates a Reader with the default
/// configuration and 4 workers per subsystem.
///
/// # Arguments
/// * `graph` - The graph processor (RocksDB)
/// * `fulltext` - The fulltext index (Tantivy)
///
/// # Returns
/// A tuple of (Reader, unified_handles, graph_handles, fulltext_handles)
pub fn create_reader(
    graph_storage: Arc<graph::Storage>,
    fulltext: Arc<fulltext::Index>,
) -> (
    Reader,
    Vec<JoinHandle<()>>,
    Vec<JoinHandle<()>>,
    Vec<JoinHandle<()>>,
) {
    ReaderBuilder::new(graph_storage, fulltext).build()
}

/// Create a unified Reader with custom configuration.
///
/// # Arguments
/// * `graph_storage` - The graph storage (RocksDB)
/// * `fulltext` - The fulltext index (Tantivy)
/// * `config` - Reader configuration
/// * `num_workers` - Number of worker tasks per subsystem
///
/// # Returns
/// A tuple of (Reader, unified_handles, graph_handles, fulltext_handles)
pub fn create_reader_with_config(
    graph_storage: Arc<graph::Storage>,
    fulltext: Arc<fulltext::Index>,
    config: ReaderConfig,
    num_workers: usize,
) -> (
    Reader,
    Vec<JoinHandle<()>>,
    Vec<JoinHandle<()>>,
    Vec<JoinHandle<()>>,
) {
    ReaderBuilder::new(graph_storage, fulltext)
        .with_config(config)
        .with_num_workers(num_workers)
        .build()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reader_config_default() {
        let config = ReaderConfig::default();
        assert_eq!(config.graph.channel_buffer_size, 1000);
        assert_eq!(config.fulltext.channel_buffer_size, 1000);
    }

    #[test]
    fn test_reader_config_with_buffer_size() {
        let config = ReaderConfig::with_channel_buffer_size(500);
        assert_eq!(config.graph.channel_buffer_size, 500);
        assert_eq!(config.fulltext.channel_buffer_size, 500);
    }
}
