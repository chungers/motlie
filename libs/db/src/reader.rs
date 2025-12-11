//! Unified reader module with MPMC pipeline for search + graph hydration.
//!
//! This module provides the infrastructure for the unified query interface:
//! - `Storage` - combines graph (RocksDB) and fulltext (Tantivy) storage
//! - `Reader` - handle for sending unified queries
//! - `Consumer` - processes unified queries
//! - Spawn functions for creating consumer pools
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────┐
//! │     Reader      │  ◄── Client sends Search queries
//! └────────┬────────┘
//!          │ flume channel (MPMC)
//!          ▼
//! ┌─────────────────┐
//! │    Consumer(s)  │  ◄── Pool of workers processing queries
//! └────────┬────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │    Storage      │
//! │ ├── fulltext    │  ◄── Tantivy (for search/ranking)
//! │ └── graph       │  ◄── RocksDB (source of truth for hydration)
//! └─────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use motlie_db::reader::{Storage, ReaderConfig, create_query_reader, spawn_consumer_pool_shared};
//! use motlie_db::query::{Nodes, Runnable};
//! use std::sync::Arc;
//! use std::time::Duration;
//!
//! // Create storage
//! let storage = Arc::new(Storage::new(graph, fulltext));
//!
//! // Create reader channel
//! let config = ReaderConfig { channel_buffer_size: 100 };
//! let (reader, receiver) = create_query_reader(config);
//!
//! // Spawn consumer pool
//! let handles = spawn_consumer_pool_shared(receiver, storage, 4);
//!
//! // Send queries
//! let results = Nodes::new("rust programming".to_string(), 10)
//!     .run(&reader, Duration::from_secs(5))
//!     .await?;
//! ```

use std::sync::Arc;

use anyhow::Result;

use crate::fulltext;
use crate::graph;
use crate::query::QueryProcessor;

// ============================================================================
// Storage
// ============================================================================

/// Storage combining graph (RocksDB) and fulltext (Tantivy).
///
/// This provides unified access to both storage backends for the query execution
/// layer. Fulltext is used for search/ranking, graph is the source of truth.
pub struct Storage {
    /// Graph storage (RocksDB) - source of truth for node/edge data
    pub graph: Arc<graph::Graph>,

    /// Fulltext storage (Tantivy) - for search and ranking
    pub fulltext: Arc<fulltext::Index>,
}

impl Storage {
    /// Create a new Storage from graph and fulltext components.
    ///
    /// # Arguments
    /// * `graph` - The graph storage (RocksDB), typically in readwrite mode
    /// * `fulltext` - The fulltext index (Tantivy), typically in readonly mode
    pub fn new(graph: Arc<graph::Graph>, fulltext: Arc<fulltext::Index>) -> Self {
        Self { graph, fulltext }
    }

    /// Get a reference to the graph component
    pub fn graph(&self) -> &graph::Graph {
        &self.graph
    }

    /// Get a reference to the fulltext component
    pub fn fulltext(&self) -> &fulltext::Index {
        &self.fulltext
    }
}

// ============================================================================
// Reader
// ============================================================================

/// Handle for sending unified queries to the reader queue.
///
/// This is the entry point for clients to submit `Search` queries that will be
/// processed by the consumer pool. Queries are sent through a bounded MPMC channel.
#[derive(Debug, Clone)]
pub struct Reader {
    sender: flume::Sender<super::query::Search>,
}

impl Reader {
    /// Create a new Reader with the given sender
    pub fn new(sender: flume::Sender<super::query::Search>) -> Self {
        Reader { sender }
    }

    /// Send a query to the reader queue
    pub async fn send_query(&self, query: super::query::Search) -> Result<()> {
        self.sender
            .send_async(query)
            .await
            .map_err(|_| anyhow::anyhow!("Failed to send query to unified reader queue"))
    }

    /// Check if the reader is still active
    pub fn is_closed(&self) -> bool {
        self.sender.is_disconnected()
    }
}

// ============================================================================
// ReaderConfig
// ============================================================================

/// Configuration for the unified query reader
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

// ============================================================================
// Factory Functions
// ============================================================================

/// Create a new unified query reader and receiver pair.
///
/// Returns a tuple of (Reader, Receiver) where:
/// - `Reader` is cloneable and used by clients to send queries
/// - `Receiver` is passed to consumers to process queries
pub fn create_query_reader(
    config: ReaderConfig,
) -> (Reader, flume::Receiver<super::query::Search>) {
    let (sender, receiver) = flume::bounded(config.channel_buffer_size);
    let reader = Reader::new(sender);
    (reader, receiver)
}

// ============================================================================
// Consumer
// ============================================================================

/// Consumer that processes unified queries using Storage.
///
/// Each consumer receives queries from a shared MPMC channel, executes them
/// against the composite storage (fulltext + graph), and sends results back
/// through oneshot channels.
pub struct Consumer {
    receiver: flume::Receiver<super::query::Search>,
    config: ReaderConfig,
    storage: Arc<Storage>,
}

impl Consumer {
    /// Create a new Consumer
    pub fn new(
        receiver: flume::Receiver<super::query::Search>,
        config: ReaderConfig,
        storage: Arc<Storage>,
    ) -> Self {
        Self {
            receiver,
            config,
            storage,
        }
    }

    /// Process queries continuously until the channel is closed
    #[tracing::instrument(skip(self), name = "unified_query_consumer")]
    pub async fn run(self) -> Result<()> {
        tracing::info!(
            config = ?self.config,
            "Starting unified query consumer"
        );

        loop {
            match self.receiver.recv_async().await {
                Ok(query) => {
                    tracing::debug!(query = %query, "Processing unified query");
                    query.process_and_send(&self.storage).await;
                }
                Err(_) => {
                    tracing::info!("Unified query consumer shutting down - channel closed");
                    return Ok(());
                }
            }
        }
    }
}

/// Spawn a unified query consumer as a background task
pub fn spawn_consumer(consumer: Consumer) -> tokio::task::JoinHandle<Result<()>> {
    tokio::spawn(async move { consumer.run().await })
}

/// Create a unified query consumer
pub fn create_consumer(
    receiver: flume::Receiver<super::query::Search>,
    config: ReaderConfig,
    storage: Arc<Storage>,
) -> Consumer {
    Consumer::new(receiver, config, storage)
}

// ============================================================================
// Pool Spawning Functions
// ============================================================================

/// Spawn a pool of unified query consumers sharing the same Storage.
///
/// This is the recommended approach for multiple query consumers because
/// all workers share the same underlying storage via `Arc<Storage>`.
///
/// # Arguments
/// * `receiver` - The receiver end of the query channel (will be cloned for each worker)
/// * `storage` - The shared storage
/// * `num_workers` - Number of worker tasks to spawn
///
/// # Returns
/// A vector of JoinHandles for the spawned worker tasks.
///
/// # Example
/// ```ignore
/// use motlie_db::reader::{Storage, ReaderConfig, create_query_reader, spawn_consumer_pool_shared};
/// use std::sync::Arc;
///
/// let storage = Arc::new(Storage::new(graph, fulltext));
/// let config = ReaderConfig { channel_buffer_size: 100 };
/// let (reader, receiver) = create_query_reader(config);
///
/// // Spawn 4 workers sharing the same storage
/// let handles = spawn_consumer_pool_shared(receiver, storage, 4);
/// ```
pub fn spawn_consumer_pool_shared(
    receiver: flume::Receiver<super::query::Search>,
    storage: Arc<Storage>,
    num_workers: usize,
) -> Vec<tokio::task::JoinHandle<()>> {
    let mut handles = Vec::with_capacity(num_workers);

    for worker_id in 0..num_workers {
        let receiver = receiver.clone();
        let storage = storage.clone(); // Cheap Arc clone

        let handle = tokio::spawn(async move {
            tracing::info!(
                worker_id,
                "Unified query worker starting (shared storage mode)"
            );

            // Process queries from shared channel
            while let Ok(query) = receiver.recv_async().await {
                tracing::debug!(worker_id, query = %query, "Processing unified query");
                query.process_and_send(&storage).await;
            }

            tracing::info!(worker_id, "Unified query worker shutting down");
        });

        handles.push(handle);
    }

    handles
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

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
    async fn test_reader_config_default() {
        let config = ReaderConfig::default();
        assert_eq!(config.channel_buffer_size, 1000);
    }

    #[tokio::test]
    async fn test_reader_creation() {
        let config = ReaderConfig {
            channel_buffer_size: 50,
        };
        let (reader, _receiver) = create_query_reader(config);

        // Reader should not be closed when receiver exists
        assert!(!reader.is_closed());
    }
}
