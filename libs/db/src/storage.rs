//! Unified storage for graph and fulltext backends.
//!
//! This module provides the [`Storage`], [`StorageConfig`], and handle types
//! that manage both graph (RocksDB) and fulltext (Tantivy) backends, providing
//! a single entry point for the motlie_db API.
//!
//! # Directory Structure
//!
//! Storage takes a single base path and automatically creates subdirectories:
//! - `<base_path>/graph` - RocksDB graph database
//! - `<base_path>/fulltext` - Tantivy fulltext index
//!
//! # Type-Safe Access Modes
//!
//! Storage uses Rust's type system to distinguish between read-only and read-write
//! access at compile time:
//!
//! - [`Storage::<ReadOnly>`] / [`Storage::readonly()`] → [`ReadOnlyHandles`] (reader only)
//! - [`Storage::<ReadWrite>`] / [`Storage::readwrite()`] → [`ReadWriteHandles`] (reader + writer)
//!
//! This eliminates runtime checks and `.unwrap()` calls - if you have `ReadWriteHandles`,
//! `writer()` is guaranteed to return a valid writer.
//!
//! # Quick Start
//!
//! ```ignore
//! use motlie_db::{Storage, StorageConfig};
//! use motlie_db::mutation::{AddNode, Runnable};
//! use motlie_db::query::{Nodes, Runnable as QueryRunnable};
//! use std::time::Duration;
//!
//! // Read-write mode - single path, subdirectories created automatically
//! let storage = Storage::readwrite(db_path);
//! let handles = storage.ready(StorageConfig::default())?;
//!
//! // Write mutations - no unwrap needed!
//! AddNode {
//!     id: Id::new(),
//!     ts_millis: TimestampMilli::now(),
//!     name: "Alice".to_string(),
//!     summary: NodeSummary::from_text("A person"),
//!     valid_range: None,
//! }
//! .run(handles.writer())
//! .await?;
//!
//! // Execute queries
//! let timeout = Duration::from_secs(5);
//! let results = Nodes::new("Alice".to_string(), 10)
//!     .run(handles.reader(), timeout)
//!     .await?;
//!
//! // Clean shutdown
//! handles.shutdown().await?;
//! ```
//!
//! # Read-Only Mode
//!
//! ```ignore
//! use motlie_db::{Storage, StorageConfig};
//! use motlie_db::query::{Nodes, Runnable};
//!
//! // Read-only mode - no writer available
//! let storage = Storage::readonly(db_path);
//! let handles = storage.ready(StorageConfig::default())?;
//!
//! // Execute queries
//! let results = Nodes::new("query".to_string(), 10)
//!     .run(handles.reader(), timeout)
//!     .await?;
//!
//! // handles.writer() <- compile error: method doesn't exist on ReadOnlyHandles
//!
//! handles.shutdown().await?;
//! ```

use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use tokio::task::JoinHandle;

use crate::fulltext;
use crate::graph;
use crate::reader::{self, ReaderBuilder};
use crate::writer::{self, WriterBuilder};

// ============================================================================
// Access Mode Markers
// ============================================================================

/// Marker type for read-only storage access.
///
/// When `Storage<ReadOnly>` is used, `ready()` returns [`ReadOnlyHandles`]
/// which only provides a reader - no writer is available.
#[derive(Debug, Clone, Copy)]
pub struct ReadOnly;

/// Marker type for read-write storage access.
///
/// When `Storage<ReadWrite>` is used, `ready()` returns [`ReadWriteHandles`]
/// which provides both reader and writer.
#[derive(Debug, Clone, Copy)]
pub struct ReadWrite;

// ============================================================================
// StorageConfig
// ============================================================================

/// Unified configuration for storage initialization.
///
/// Combines reader and writer configurations into a single struct.
///
/// # Example
///
/// ```ignore
/// use motlie_db::StorageConfig;
///
/// // Default configuration
/// let config = StorageConfig::default();
///
/// // Custom channel buffer size
/// let config = StorageConfig::with_channel_buffer_size(2000);
///
/// // Custom worker count
/// let config = StorageConfig::default().with_num_workers(8);
/// ```
#[derive(Debug, Clone)]
pub struct StorageConfig {
    /// Reader configuration (channel buffer sizes)
    pub reader: reader::ReaderConfig,

    /// Writer configuration (channel buffer sizes)
    pub writer: writer::WriterConfig,

    /// Number of query workers per subsystem
    pub num_query_workers: usize,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            reader: reader::ReaderConfig::default(),
            writer: writer::WriterConfig::default(),
            num_query_workers: 4,
        }
    }
}

impl StorageConfig {
    /// Create a config with uniform channel buffer size for all subsystems.
    pub fn with_channel_buffer_size(size: usize) -> Self {
        Self {
            reader: reader::ReaderConfig::with_channel_buffer_size(size),
            writer: writer::WriterConfig::with_channel_buffer_size(size),
            num_query_workers: 4,
        }
    }

    /// Set the number of query workers per subsystem.
    pub fn with_num_workers(mut self, num_workers: usize) -> Self {
        self.num_query_workers = num_workers;
        self
    }

    /// Set custom reader configuration.
    pub fn with_reader_config(mut self, config: reader::ReaderConfig) -> Self {
        self.reader = config;
        self
    }

    /// Set custom writer configuration.
    pub fn with_writer_config(mut self, config: writer::WriterConfig) -> Self {
        self.writer = config;
        self
    }
}

// ============================================================================
// Storage<Mode> - Type-state pattern for access mode
// ============================================================================

/// Unified storage combining graph (RocksDB) and fulltext (Tantivy) backends.
///
/// This is the primary entry point for the motlie_db API. The type parameter `Mode`
/// determines whether this is read-only or read-write storage:
///
/// - `Storage<ReadOnly>` - Created via [`Storage::readonly()`], returns [`ReadOnlyHandles`]
/// - `Storage<ReadWrite>` - Created via [`Storage::readwrite()`], returns [`ReadWriteHandles`]
///
/// # Directory Structure
///
/// Storage takes a single base path and automatically manages subdirectories:
/// - `<path>/graph` - RocksDB graph database
/// - `<path>/fulltext` - Tantivy fulltext index
///
/// # Type Safety
///
/// The type-state pattern ensures that:
/// - Read-only storage cannot accidentally try to write (compile error)
/// - Read-write storage always has a valid writer (no `Option` or `unwrap()`)
///
/// # Examples
///
/// ## Read-Write Mode
///
/// ```ignore
/// use motlie_db::{Storage, StorageConfig};
/// use motlie_db::mutation::{AddNode, Runnable};
///
/// let storage = Storage::readwrite(db_path);
/// let handles = storage.ready(StorageConfig::default())?;
///
/// // writer() returns &Writer directly - no unwrap needed
/// AddNode { /* ... */ }.run(handles.writer()).await?;
///
/// handles.shutdown().await?;
/// ```
///
/// ## Read-Only Mode
///
/// ```ignore
/// use motlie_db::{Storage, StorageConfig};
/// use motlie_db::query::{Nodes, Runnable};
///
/// let storage = Storage::readonly(db_path);
/// let handles = storage.ready(StorageConfig::default())?;
///
/// // Only reader is available
/// let results = Nodes::new("query", 10).run(handles.reader(), timeout).await?;
///
/// // handles.writer() <- compile error: no such method
///
/// handles.shutdown().await?;
/// ```
pub struct Storage<Mode> {
    path: PathBuf,
    _mode: PhantomData<Mode>,
}

impl Storage<ReadOnly> {
    /// Create a read-only storage.
    ///
    /// Returns `Storage<ReadOnly>` which produces [`ReadOnlyHandles`] when
    /// `ready()` is called. The handles will only have a reader - no writer.
    ///
    /// # Directory Structure
    ///
    /// The path is the base directory. Storage will use:
    /// - `<path>/graph` - RocksDB graph database
    /// - `<path>/fulltext` - Tantivy fulltext index
    ///
    /// # Arguments
    /// * `path` - Base path for the storage directory
    pub fn readonly(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            _mode: PhantomData,
        }
    }

    /// Initialize storage and return read-only handles.
    ///
    /// This method:
    /// 1. Initializes graph storage (RocksDB) in read-only mode at `<path>/graph`
    /// 2. Initializes fulltext storage (Tantivy) in read-only mode at `<path>/fulltext`
    /// 3. Sets up MPMC query consumer pools
    /// 4. Returns [`ReadOnlyHandles`] for reading only
    ///
    /// # Arguments
    /// * `config` - Configuration for channels and worker counts
    ///
    /// # Returns
    /// [`ReadOnlyHandles`] providing access to Reader only
    pub fn ready(self, config: StorageConfig) -> Result<ReadOnlyHandles> {
        let graph_path = self.path.join("graph");
        let fulltext_path = self.path.join("fulltext");

        // Initialize graph storage in read-only mode
        let mut graph_storage = graph::Storage::readonly(&graph_path);
        graph_storage.ready()?;
        let graph_arc = Arc::new(graph_storage);
        let graph = Arc::new(graph::Graph::new(graph_arc));

        // Initialize fulltext storage in read-only mode
        let mut fulltext_storage = fulltext::Storage::readonly(&fulltext_path);
        fulltext_storage.ready()?;
        let fulltext_arc = Arc::new(fulltext_storage);
        let fulltext_index = Arc::new(fulltext::Index::new(fulltext_arc));

        // Build reader infrastructure
        let (reader, unified_handles, graph_handles, fulltext_handles) =
            ReaderBuilder::new(graph, fulltext_index)
                .with_config(config.reader)
                .with_num_workers(config.num_query_workers)
                .build();

        let mut reader_handles: Vec<JoinHandle<()>> = unified_handles;
        reader_handles.extend(graph_handles);
        reader_handles.extend(fulltext_handles);

        Ok(ReadOnlyHandles {
            reader,
            reader_handles,
        })
    }
}

impl Storage<ReadWrite> {
    /// Create a read-write storage.
    ///
    /// Returns `Storage<ReadWrite>` which produces [`ReadWriteHandles`] when
    /// `ready()` is called. The handles will have both reader and writer.
    ///
    /// # Directory Structure
    ///
    /// The path is the base directory. Storage will use:
    /// - `<path>/graph` - RocksDB graph database
    /// - `<path>/fulltext` - Tantivy fulltext index
    ///
    /// # Arguments
    /// * `path` - Base path for the storage directory
    pub fn readwrite(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            _mode: PhantomData,
        }
    }

    /// Initialize storage and return read-write handles.
    ///
    /// This method:
    /// 1. Initializes graph storage (RocksDB) in read-write mode at `<path>/graph`
    /// 2. Initializes fulltext storage (Tantivy) in read-write mode at `<path>/fulltext`
    /// 3. Sets up MPMC query consumer pools
    /// 4. Sets up graph → fulltext mutation pipeline
    /// 5. Returns [`ReadWriteHandles`] for both reading and writing
    ///
    /// # Arguments
    /// * `config` - Configuration for channels and worker counts
    ///
    /// # Returns
    /// [`ReadWriteHandles`] providing access to both Reader and Writer
    pub fn ready(self, config: StorageConfig) -> Result<ReadWriteHandles> {
        let graph_path = self.path.join("graph");
        let fulltext_path = self.path.join("fulltext");

        // Initialize graph storage in read-write mode
        let mut graph_storage = graph::Storage::readwrite(&graph_path);
        graph_storage.ready()?;
        let graph_arc = Arc::new(graph_storage);
        let graph = Arc::new(graph::Graph::new(graph_arc));

        // Initialize fulltext storage in read-write mode
        let mut fulltext_storage = fulltext::Storage::readwrite(&fulltext_path);
        fulltext_storage.ready()?;
        let fulltext_arc = Arc::new(fulltext_storage);
        let fulltext_index = Arc::new(fulltext::Index::new(fulltext_arc));

        // Build reader infrastructure
        let (reader, unified_handles, graph_handles, fulltext_handles) =
            ReaderBuilder::new(graph.clone(), fulltext_index.clone())
                .with_config(config.reader)
                .with_num_workers(config.num_query_workers)
                .build();

        let mut reader_handles: Vec<JoinHandle<()>> = unified_handles;
        reader_handles.extend(graph_handles);
        reader_handles.extend(fulltext_handles);

        // Build writer infrastructure
        let (writer, writer_handles) = WriterBuilder::new(graph, fulltext_index)
            .with_config(config.writer)
            .build();

        Ok(ReadWriteHandles {
            reader,
            writer,
            reader_handles,
            writer_handles,
        })
    }
}

// Common methods for both modes
impl<Mode> Storage<Mode> {
    /// Get the base storage path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the graph subdirectory path (`<base_path>/graph`).
    pub fn graph_path(&self) -> PathBuf {
        self.path.join("graph")
    }

    /// Get the fulltext subdirectory path (`<base_path>/fulltext`).
    pub fn fulltext_path(&self) -> PathBuf {
        self.path.join("fulltext")
    }
}

// ============================================================================
// ReadOnlyHandles
// ============================================================================

/// Handles for read-only storage access.
///
/// Returned by [`Storage::<ReadOnly>::ready()`]. Provides access to a [`Reader`](reader::Reader)
/// for executing queries. No writer is available - attempting to call `writer()` will
/// result in a compile error.
///
/// # Example
///
/// ```ignore
/// let storage = Storage::readonly(db_path);
/// let handles: ReadOnlyHandles = storage.ready(config)?;
///
/// // Execute queries
/// let results = Nodes::new("query", 10).run(handles.reader(), timeout).await?;
///
/// // handles.writer() <- compile error!
///
/// handles.shutdown().await?;
/// ```
pub struct ReadOnlyHandles {
    reader: reader::Reader,
    reader_handles: Vec<JoinHandle<()>>,
}

impl ReadOnlyHandles {
    /// Get the reader for executing queries.
    pub fn reader(&self) -> &reader::Reader {
        &self.reader
    }

    /// Get a clone of the reader.
    ///
    /// Useful for sharing the reader across multiple async tasks.
    pub fn reader_clone(&self) -> reader::Reader {
        self.reader.clone()
    }

    /// Check if the storage system is still running.
    pub fn is_running(&self) -> bool {
        !self.reader.is_closed()
    }

    /// Shut down all workers cleanly.
    ///
    /// This method:
    /// 1. Drops the reader (closes query channels)
    /// 2. Awaits all reader worker handles
    ///
    /// # Errors
    /// Returns an error if any worker task panicked.
    pub async fn shutdown(self) -> Result<()> {
        // Drop reader to close query channels
        drop(self.reader);

        // Wait for reader handles
        for handle in self.reader_handles {
            handle
                .await
                .map_err(|e| anyhow::anyhow!("Reader worker panicked: {}", e))?;
        }

        tracing::info!("ReadOnly storage shutdown complete");
        Ok(())
    }

    /// Shut down gracefully, ignoring worker errors.
    pub async fn shutdown_graceful(self) {
        drop(self.reader);

        for (i, handle) in self.reader_handles.into_iter().enumerate() {
            if let Err(e) = handle.await {
                tracing::warn!(worker_id = i, error = %e, "Reader worker panicked during shutdown");
            }
        }

        tracing::info!("ReadOnly storage graceful shutdown complete");
    }
}

// ============================================================================
// ReadWriteHandles
// ============================================================================

/// Handles for read-write storage access.
///
/// Returned by [`Storage::<ReadWrite>::ready()`]. Provides access to both
/// [`Reader`](reader::Reader) and [`Writer`](writer::Writer).
///
/// Unlike the previous API, `writer()` returns `&Writer` directly - no `Option`
/// or `unwrap()` needed. This is guaranteed by the type system.
///
/// # Example
///
/// ```ignore
/// let storage = Storage::readwrite(db_path);
/// let handles: ReadWriteHandles = storage.ready(config)?;
///
/// // Write mutations - no unwrap needed!
/// AddNode { /* ... */ }.run(handles.writer()).await?;
///
/// // Execute queries
/// let results = Nodes::new("query", 10).run(handles.reader(), timeout).await?;
///
/// handles.shutdown().await?;
/// ```
pub struct ReadWriteHandles {
    reader: reader::Reader,
    writer: writer::Writer,
    reader_handles: Vec<JoinHandle<()>>,
    writer_handles: Vec<JoinHandle<Result<()>>>,
}

impl ReadWriteHandles {
    /// Get the reader for executing queries.
    pub fn reader(&self) -> &reader::Reader {
        &self.reader
    }

    /// Get a clone of the reader.
    ///
    /// Useful for sharing the reader across multiple async tasks.
    pub fn reader_clone(&self) -> reader::Reader {
        self.reader.clone()
    }

    /// Get the writer for executing mutations.
    ///
    /// This always returns a valid writer reference - no `Option` or `unwrap()` needed.
    /// The type system guarantees that `ReadWriteHandles` always has a writer.
    pub fn writer(&self) -> &writer::Writer {
        &self.writer
    }

    /// Get a clone of the writer.
    ///
    /// Useful for sharing the writer across multiple async tasks.
    pub fn writer_clone(&self) -> writer::Writer {
        self.writer.clone()
    }

    /// Check if the storage system is still running.
    pub fn is_running(&self) -> bool {
        !self.reader.is_closed() && !self.writer.is_closed()
    }

    /// Shut down all workers cleanly.
    ///
    /// This method:
    /// 1. Drops the writer (closes mutation channels)
    /// 2. Awaits writer worker handles
    /// 3. Drops the reader (closes query channels)
    /// 4. Awaits reader worker handles
    ///
    /// # Errors
    /// Returns an error if any worker task panicked.
    pub async fn shutdown(self) -> Result<()> {
        // Drop writer first to close mutation channels
        drop(self.writer);

        // Wait for writer handles
        for handle in self.writer_handles {
            handle
                .await
                .map_err(|e| anyhow::anyhow!("Writer worker panicked: {}", e))??;
        }

        // Drop reader to close query channels
        drop(self.reader);

        // Wait for reader handles
        for handle in self.reader_handles {
            handle
                .await
                .map_err(|e| anyhow::anyhow!("Reader worker panicked: {}", e))?;
        }

        tracing::info!("ReadWrite storage shutdown complete");
        Ok(())
    }

    /// Shut down gracefully, ignoring worker errors.
    pub async fn shutdown_graceful(self) {
        drop(self.writer);

        for (i, handle) in self.writer_handles.into_iter().enumerate() {
            match handle.await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    tracing::warn!(worker_id = i, error = %e, "Writer worker error during shutdown");
                }
                Err(e) => {
                    tracing::warn!(worker_id = i, error = %e, "Writer worker panicked during shutdown");
                }
            }
        }

        drop(self.reader);

        for (i, handle) in self.reader_handles.into_iter().enumerate() {
            if let Err(e) = handle.await {
                tracing::warn!(worker_id = i, error = %e, "Reader worker panicked during shutdown");
            }
        }

        tracing::info!("ReadWrite storage graceful shutdown complete");
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_config_default() {
        let config = StorageConfig::default();
        assert_eq!(config.reader.channel_buffer_size, 1000);
        assert_eq!(config.writer.graph.channel_buffer_size, 1000);
        assert_eq!(config.num_query_workers, 4);
    }

    #[test]
    fn test_storage_config_with_buffer_size() {
        let config = StorageConfig::with_channel_buffer_size(500);
        assert_eq!(config.reader.channel_buffer_size, 500);
        assert_eq!(config.writer.graph.channel_buffer_size, 500);
    }

    #[test]
    fn test_storage_config_with_num_workers() {
        let config = StorageConfig::default().with_num_workers(8);
        assert_eq!(config.num_query_workers, 8);
    }

    #[test]
    fn test_storage_readonly_paths() {
        let storage = Storage::readonly(Path::new("/tmp/motlie_db"));
        assert_eq!(storage.path(), Path::new("/tmp/motlie_db"));
        assert_eq!(storage.graph_path(), Path::new("/tmp/motlie_db/graph"));
        assert_eq!(storage.fulltext_path(), Path::new("/tmp/motlie_db/fulltext"));
    }

    #[test]
    fn test_storage_readwrite_paths() {
        let storage = Storage::readwrite(Path::new("/tmp/motlie_db"));
        assert_eq!(storage.path(), Path::new("/tmp/motlie_db"));
        assert_eq!(storage.graph_path(), Path::new("/tmp/motlie_db/graph"));
        assert_eq!(storage.fulltext_path(), Path::new("/tmp/motlie_db/fulltext"));
    }
}
