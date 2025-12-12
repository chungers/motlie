//! Unified reader module composing graph and fulltext query infrastructure.
//!
//! This module provides the [`Storage`], [`StorageHandle`], and [`Reader`] types
//! that manage both graph (RocksDB) and fulltext (Tantivy) query subsystems,
//! forwarding queries to the appropriate backend.
//!
//! # Quick Start
//!
//! The recommended way to initialize the unified query system is via [`Storage::ready()`]:
//!
//! ```ignore
//! use motlie_db::reader::{Storage, ReaderConfig, StorageHandle};
//! use motlie_db::query::{Nodes, NodeById, OutgoingEdges, Runnable};
//! use std::time::Duration;
//!
//! // 1. Create unified storage pointing to both databases
//! let storage = Storage::readonly(graph_path, fulltext_path);
//!
//! // 2. Initialize storage and get the handle
//! let handle = storage.ready(ReaderConfig::default(), 4)?;
//!
//! let timeout = Duration::from_secs(5);
//!
//! // 3. Execute queries through the reader
//!
//! // Fulltext search with automatic graph hydration
//! let results = Nodes::new("rust programming".to_string(), 10)
//!     .run(handle.reader(), timeout)
//!     .await?;
//!
//! // Direct graph lookups
//! let (name, summary) = NodeById::new(node_id, None)
//!     .run(handle.reader(), timeout)
//!     .await?;
//!
//! let edges = OutgoingEdges::new(node_id, None)
//!     .run(handle.reader(), timeout)
//!     .await?;
//!
//! // 4. Clean shutdown
//! handle.shutdown().await?;
//! ```
//!
//! # Architecture
//!
//! The unified reader manages three query pipelines:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                           Reader (unified)                               │
//! │                                                                          │
//! │  ┌────────────────────────────────────────────────────────────────────┐ │
//! │  │                    Unified Query Pipeline                           │ │
//! │  │  Handles: Nodes, Edges (fulltext search + graph hydration)          │ │
//! │  │           NodeById, OutgoingEdges, IncomingEdges (forwarded)        │ │
//! │  │           EdgeDetails, NodeFragments, EdgeFragments (forwarded)     │ │
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
//! │  │   graph::Graph          │   │      fulltext::Index            │     │
//! │  │   (RocksDB)             │   │      (Tantivy)                  │     │
//! │  └─────────────────────────┘   └─────────────────────────────────┘     │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Key Types
//!
//! | Type | Description |
//! |------|-------------|
//! | [`Storage`] | Unified storage configuration with `readonly()` and `readwrite()` constructors |
//! | [`StorageHandle`] | Lifecycle manager returned by `Storage::ready()` with `shutdown()` method |
//! | [`Reader`] | Query interface obtained via `StorageHandle::reader()` |
//! | [`ReaderConfig`] | Configuration for channel buffer sizes |
//! | [`CompositeStorage`] | Internal storage reference for query execution |
//!
//! # See Also
//!
//! - [`query`](crate::query) - Query types that work with this reader
//! - [`graph::reader`](crate::graph::reader) - Graph-only reader (advanced use)
//! - [`fulltext::reader`](crate::fulltext::reader) - Fulltext-only reader (advanced use)

use std::sync::Arc;

use anyhow::Result;
use tokio::task::JoinHandle;

use crate::fulltext;
use crate::graph;
use crate::query::{Query, QueryProcessor};

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
// Storage - Unified storage with lazy initialization
// ============================================================================

/// Unified storage combining graph (RocksDB) and fulltext (Tantivy).
///
/// This struct manages both storage backends and provides the `ready()` method
/// to initialize them and set up the MPMC query pipelines.
///
/// # Usage
///
/// ```ignore
/// use motlie_db::reader::{Storage, ReaderConfig};
/// use std::path::Path;
///
/// // Create storage configuration (read-write mode)
/// let mut storage = Storage::readwrite(
///     Path::new("/path/to/graph"),
///     Path::new("/path/to/fulltext"),
/// );
///
/// // Initialize storage and get the reader
/// let (reader, handles) = storage.ready(ReaderConfig::default(), 4)?;
///
/// // Use the reader for queries
/// let results = Nodes::new("query", 10).run(&reader, timeout).await?;
/// ```
pub struct Storage {
    /// Graph storage (lazily initialized)
    graph_storage: graph::Storage,

    /// Fulltext storage (lazily initialized)
    fulltext_storage: fulltext::Storage,
}

impl Storage {
    /// Create a new readonly unified Storage.
    ///
    /// # Arguments
    /// * `graph_path` - Path to the graph database (RocksDB)
    /// * `fulltext_path` - Path to the fulltext index (Tantivy)
    pub fn readonly(
        graph_path: &std::path::Path,
        fulltext_path: &std::path::Path,
    ) -> Self {
        Self {
            graph_storage: graph::Storage::readonly(graph_path),
            fulltext_storage: fulltext::Storage::readonly(fulltext_path),
        }
    }

    /// Create a new read-write unified Storage.
    ///
    /// # Arguments
    /// * `graph_path` - Path to the graph database (RocksDB)
    /// * `fulltext_path` - Path to the fulltext index (Tantivy)
    pub fn readwrite(
        graph_path: &std::path::Path,
        fulltext_path: &std::path::Path,
    ) -> Self {
        Self {
            graph_storage: graph::Storage::readwrite(graph_path),
            fulltext_storage: fulltext::Storage::readwrite(fulltext_path),
        }
    }

    /// Initialize both storage backends and create the MPMC query pipelines.
    ///
    /// This method:
    /// 1. Calls `ready()` on both graph and fulltext storage
    /// 2. Creates the MPMC channels and consumer pools for all query types
    /// 3. Returns a [`StorageHandle`] that manages the Reader and worker lifecycle
    ///
    /// Note: This consumes the Storage struct since the underlying storage
    /// objects are moved into Arc wrappers.
    ///
    /// # Arguments
    /// * `config` - Reader configuration for channel buffer sizes
    /// * `num_workers` - Number of worker tasks per subsystem
    ///
    /// # Returns
    /// A [`StorageHandle`] providing access to the Reader and clean shutdown
    ///
    /// # Example
    /// ```ignore
    /// let storage = Storage::readonly(graph_path, fulltext_path);
    /// let handle = storage.ready(ReaderConfig::default(), 4)?;
    ///
    /// // Use the reader
    /// let results = Nodes::new("query", 10).run(handle.reader(), timeout).await?;
    ///
    /// // Clean shutdown
    /// handle.shutdown().await?;
    /// ```
    pub fn ready(
        mut self,
        config: ReaderConfig,
        num_workers: usize,
    ) -> Result<StorageHandle> {
        // Initialize both subsystems
        self.graph_storage.ready()?;
        self.fulltext_storage.ready()?;

        // Wrap storage in Arc and create Graph/Index
        let graph_arc = Arc::new(self.graph_storage);
        let fulltext_arc = Arc::new(self.fulltext_storage);

        let graph = Arc::new(graph::Graph::new(graph_arc));
        let fulltext = Arc::new(fulltext::Index::new(fulltext_arc));

        // Build the reader and get all handles
        let (reader, unified_handles, graph_handles, fulltext_handles) =
            ReaderBuilder::new(graph, fulltext)
                .with_config(config)
                .with_num_workers(num_workers)
                .build();

        // Combine all handles
        let mut all_handles = unified_handles;
        all_handles.extend(graph_handles);
        all_handles.extend(fulltext_handles);

        Ok(StorageHandle {
            reader,
            handles: all_handles,
        })
    }
}

// ============================================================================
// StorageHandle - Lifecycle manager for Reader and worker tasks
// ============================================================================

/// Handle returned by [`Storage::ready()`] that manages the Reader and worker lifecycle.
///
/// This struct provides:
/// - Access to the [`Reader`] for executing queries
/// - A [`shutdown()`](StorageHandle::shutdown) method for clean termination
///
/// # Lifecycle
///
/// ```text
/// Storage::ready()  →  StorageHandle  →  shutdown()
///       │                    │                │
///       │                    │                ├── Drops Reader (closes channels)
///       │                    │                └── Awaits all worker handles
///       │                    │
///       │                    └── Provides Reader for queries
///       │
///       └── Initializes storage, spawns workers
/// ```
///
/// # Example
/// ```ignore
/// use motlie_db::reader::{Storage, ReaderConfig, StorageHandle};
/// use motlie_db::query::{Nodes, Runnable};
/// use std::time::Duration;
///
/// // Initialize storage and get handle
/// let storage = Storage::readonly(graph_path, fulltext_path);
/// let handle = storage.ready(ReaderConfig::default(), 4)?;
///
/// // Execute queries through the reader
/// let timeout = Duration::from_secs(5);
/// let results = Nodes::new("rust programming".to_string(), 10)
///     .run(handle.reader(), timeout)
///     .await?;
///
/// // Clean shutdown - closes channels and waits for workers
/// handle.shutdown().await?;
/// ```
///
/// # Shutdown Semantics
///
/// The [`shutdown()`](StorageHandle::shutdown) method:
/// 1. Drops the Reader, which disconnects all channel senders
/// 2. Workers detect the disconnected channels and exit their loops
/// 3. Awaits all worker JoinHandles to ensure clean termination
///
/// If you drop the StorageHandle without calling shutdown(), the workers
/// will still terminate (channels close on drop), but you won't await
/// their completion.
pub struct StorageHandle {
    reader: Reader,
    handles: Vec<JoinHandle<()>>,
}

impl StorageHandle {
    /// Get a reference to the Reader for executing queries.
    ///
    /// The Reader can be cloned if needed for concurrent access from
    /// multiple tasks.
    pub fn reader(&self) -> &Reader {
        &self.reader
    }

    /// Get a clone of the Reader.
    ///
    /// This is useful when you need to share the reader across multiple
    /// async tasks or threads.
    pub fn reader_clone(&self) -> Reader {
        self.reader.clone()
    }

    /// Check if the storage system is still running.
    ///
    /// Returns `false` if any of the channels are closed, indicating
    /// that the workers are shutting down or have shut down.
    pub fn is_running(&self) -> bool {
        !self.reader.is_closed()
    }

    /// Shut down the storage system cleanly.
    ///
    /// This method:
    /// 1. Drops the Reader, closing all channel senders
    /// 2. Workers detect closed channels and exit their loops
    /// 3. Awaits all worker JoinHandles
    ///
    /// # Errors
    /// Returns an error if any worker task panicked during shutdown.
    ///
    /// # Example
    /// ```ignore
    /// let handle = storage.ready(config, 4)?;
    ///
    /// // ... use the reader ...
    ///
    /// // Clean shutdown
    /// handle.shutdown().await?;
    /// ```
    pub async fn shutdown(self) -> Result<()> {
        // Drop the reader to close all channels
        // This signals workers to stop
        drop(self.reader);

        // Await all worker handles
        for handle in self.handles {
            handle
                .await
                .map_err(|e| anyhow::anyhow!("Worker task panicked during shutdown: {}", e))?;
        }

        tracing::info!("Storage shutdown complete");
        Ok(())
    }

    /// Shut down the storage system, consuming self but ignoring worker errors.
    ///
    /// This is a convenience method when you want to ensure shutdown happens
    /// but don't need to handle individual worker failures.
    ///
    /// Unlike [`shutdown()`](StorageHandle::shutdown), this method:
    /// - Logs warnings for any worker panics instead of returning errors
    /// - Always returns `Ok(())`
    pub async fn shutdown_graceful(self) {
        drop(self.reader);

        for (i, handle) in self.handles.into_iter().enumerate() {
            if let Err(e) = handle.await {
                tracing::warn!(worker_id = i, error = %e, "Worker panicked during shutdown");
            }
        }

        tracing::info!("Storage graceful shutdown complete");
    }
}

// ============================================================================
// CompositeStorage - Internal storage reference for query execution
// ============================================================================

/// Internal storage reference combining graph and fulltext for query execution.
///
/// This is used by the consumer pools to execute queries. It holds Arc references
/// to the initialized storage backends.
pub struct CompositeStorage {
    /// Graph storage (RocksDB) - source of truth for node/edge data
    pub graph: Arc<graph::Graph>,

    /// Fulltext storage (Tantivy) - for search and ranking
    pub fulltext: Arc<fulltext::Index>,
}

impl CompositeStorage {
    /// Create a new CompositeStorage from graph and fulltext components.
    pub fn new(graph: Arc<graph::Graph>, fulltext: Arc<fulltext::Index>) -> Self {
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
#[derive(Clone)]
pub struct Reader {
    /// Sender for unified search queries (Nodes, Edges with hydration)
    sender: flume::Sender<Query>,

    /// Graph reader for direct graph queries (NodeById, OutgoingEdges, etc.)
    graph_reader: graph::Reader,

    /// Fulltext reader for search queries (raw fulltext hits)
    fulltext_reader: fulltext::Reader,

    /// Shared storage for hydration (fulltext hits → graph data)
    composite_storage: Arc<CompositeStorage>,
}

impl Reader {
    /// Send a unified search query to the consumer pool.
    pub async fn send_query(&self, query: Query) -> Result<()> {
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
pub struct ReaderBuilder {
    graph: Arc<graph::Graph>,
    fulltext: Arc<fulltext::Index>,
    config: ReaderConfig,
    num_workers: usize,
}

impl ReaderBuilder {
    /// Create a new ReaderBuilder.
    ///
    /// # Arguments
    /// * `graph` - The graph storage (RocksDB)
    /// * `fulltext` - The fulltext index (Tantivy)
    pub fn new(graph: Arc<graph::Graph>, fulltext: Arc<fulltext::Index>) -> Self {
        Self {
            graph,
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
    pub fn build(self) -> (Reader, Vec<JoinHandle<()>>, Vec<JoinHandle<()>>, Vec<JoinHandle<()>>) {
        // Create shared composite storage
        let composite_storage = Arc::new(CompositeStorage::new(self.graph.clone(), self.fulltext.clone()));

        // Create graph reader and consumer pool
        let (graph_reader, graph_receiver) =
            graph::reader::create_query_reader(self.config.graph.clone());
        let graph_handles = graph::reader::spawn_query_consumer_pool_shared(
            graph_receiver,
            self.graph.clone(),
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
        let unified_handles = spawn_consumer_pool(
            receiver,
            composite_storage.clone(),
            self.num_workers,
        );

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
    receiver: flume::Receiver<Query>,
    storage: Arc<CompositeStorage>,
    num_workers: usize,
) -> Vec<JoinHandle<()>> {
    (0..num_workers)
        .map(|_| {
            let receiver = receiver.clone();
            let storage = storage.clone();
            tokio::spawn(async move {
                while let Ok(query) = receiver.recv_async().await {
                    query.process_and_send(&storage).await;
                }
            })
        })
        .collect()
}

// ============================================================================
// Convenience function
// ============================================================================

/// Create a unified Reader with default configuration.
///
/// This is a convenience function that creates a Reader with the default
/// configuration and 4 workers per subsystem.
///
/// # Arguments
/// * `graph` - The graph storage (RocksDB)
/// * `fulltext` - The fulltext index (Tantivy)
///
/// # Returns
/// A tuple of (Reader, unified_handles, graph_handles, fulltext_handles)
pub fn create_reader(
    graph: Arc<graph::Graph>,
    fulltext: Arc<fulltext::Index>,
) -> (Reader, Vec<JoinHandle<()>>, Vec<JoinHandle<()>>, Vec<JoinHandle<()>>) {
    ReaderBuilder::new(graph, fulltext).build()
}

/// Create a unified Reader with custom configuration.
///
/// # Arguments
/// * `graph` - The graph storage (RocksDB)
/// * `fulltext` - The fulltext index (Tantivy)
/// * `config` - Reader configuration
/// * `num_workers` - Number of worker tasks per subsystem
///
/// # Returns
/// A tuple of (Reader, unified_handles, graph_handles, fulltext_handles)
pub fn create_reader_with_config(
    graph: Arc<graph::Graph>,
    fulltext: Arc<fulltext::Index>,
    config: ReaderConfig,
    num_workers: usize,
) -> (Reader, Vec<JoinHandle<()>>, Vec<JoinHandle<()>>, Vec<JoinHandle<()>>) {
    ReaderBuilder::new(graph, fulltext)
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
