//! Provides the graph-specific implementation for processing mutations from the MPSC queue
//! and writing them to the graph store.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use rocksdb::{Options, DB};

use crate::{
    index::{Edges, Fragments, Index, Nodes},
    mutation::{Consumer, Processor},
    AddEdgeArgs, AddFragmentArgs, AddVertexArgs, InvalidateArgs, WriterConfig,
};

enum StorageMode {
    ReadOnly,
    ReadWrite,
}

struct StorageOptions {}
impl StorageOptions {
    /// Default options for Storage in Readwrite mode
    /// For RocksDB, the default settings are:
    /// - error_if_exists: false
    /// - create_if_missing: true
    /// - create_missing_column_families: true
    /// This will allow existing databases to be opened, but can
    /// potentially cause corruption if the schema change is incompatible.
    /// The graph processor is the only writer that can modify the graph store.
    /// Readers are expected to be able to open and read the database.
    pub fn default_for_readwrite() -> Options {
        let mut options = Options::default();
        options.set_error_if_exists(false);
        options.create_if_missing(true);
        options.create_missing_column_families(true);
        options
    }

    /// Default options for Storage in Readonly mode
    /// For RocksDB, the default settings are:
    /// - error_if_exists: false
    /// - create_if_missing: false
    /// - create_missing_column_families: false
    pub fn default_for_readonly() -> Options {
        let mut options = Options::default();
        options.set_error_if_exists(false);
        options.create_if_missing(false);
        options.create_missing_column_families(false);
        options
    }
}

/// Graph-specific storage
pub struct Storage {
    db_path: PathBuf,
    db_options: Options,
    db: Option<DB>,
    mode: StorageMode,
}

impl Storage {
    /// Create a new Storage instance in readonly mode
    pub fn readonly(db_path: &Path) -> Self {
        Self {
            db_path: PathBuf::from(db_path),
            db_options: StorageOptions::default_for_readonly(),
            db: None,
            mode: StorageMode::ReadOnly,
        }
    }

    /// Create a new Storage instance in readwrite mode
    pub fn readwrite(db_path: &Path) -> Self {
        Self {
            db_path: PathBuf::from(db_path),
            db_options: StorageOptions::default_for_readwrite(),
            db: None,
            mode: StorageMode::ReadWrite,
        }
    }

    /// Close the database
    pub fn close(&mut self) -> Result<()> {
        if self.db.is_none() {
            return Err(anyhow::anyhow!("[Storage] Storage is not ready. "));
        }
        if let Some(db) = self.db.take() {
            // Only flush if in ReadWrite mode (flush not supported in ReadOnly)
            match self.mode {
                StorageMode::ReadWrite => {
                    db.flush()?;
                }
                StorageMode::ReadOnly => {
                    // Skip flush for ReadOnly mode
                }
            }
            drop(db); // drop the reference for automatic closing the database.
        }
        Ok(())
    }

    fn all_column_families(&self) -> Vec<&str> {
        vec![Nodes::cf_name(), Edges::cf_name(), Fragments::cf_name()]
    }

    /// Readies by opening the database and column families.
    /// Errors:
    /// - If the database path is not a directory or does not exist.
    /// - If the database path is a file or a symlink.
    pub fn ready(&mut self) -> Result<()> {
        if self.db.is_some() {
            return Ok(());
        }
        // The path should be a directory and it should exist.
        match self.db_path.try_exists() {
            Err(e) => return Err(e.into()),
            Ok(true) => {
                if self.db_path.is_file() {
                    return Err(anyhow::anyhow!(
                        "Path is a file: {}",
                        self.db_path.display()
                    ));
                }
                if self.db_path.is_symlink() {
                    return Err(anyhow::anyhow!(
                        "Path is a symlink: {}",
                        self.db_path.display()
                    ));
                }
            }
            Ok(false) => {}
        }

        match self.mode {
            StorageMode::ReadOnly => {
                self.db = Some(DB::open_cf_for_read_only(
                    &self.db_options,
                    &self.db_path,
                    self.all_column_families(),
                    false,
                )?);
            }
            StorageMode::ReadWrite => {
                self.db = Some(DB::open_cf(
                    &self.db_options,
                    &self.db_path,
                    self.all_column_families(),
                )?);
            }
        }

        log::info!("[Storage] Ready");
        Ok(())
    }
}

/// Graph-specific mutation processor
pub struct Graph {
    storage: Arc<Storage>,
}

impl Graph {
    /// Create a new GraphProcessor
    pub fn new(storage: Arc<Storage>) -> Self {
        Self { storage }
    }
}

#[async_trait::async_trait]
impl Processor for Graph {
    /// Process an AddVertex mutation
    async fn process_add_vertex(&self, args: &AddVertexArgs) -> Result<()> {
        // TODO: Implement actual vertex insertion into graph store
        log::info!("[Graph] Would insert vertex: {:?}", args);
        // Simulate some async work
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        Ok(())
    }

    /// Process an AddEdge mutation
    async fn process_add_edge(&self, args: &AddEdgeArgs) -> Result<()> {
        // TODO: Implement actual edge insertion into graph store
        log::info!("[Graph] Would insert edge: {:?}", args);
        // Simulate some async work
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        Ok(())
    }

    /// Process an AddFragment mutation
    async fn process_add_fragment(&self, args: &AddFragmentArgs) -> Result<()> {
        // TODO: Implement actual fragment insertion into graph store
        log::info!("[Graph] Would insert fragment: {:?}", args);
        // Simulate some async work
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        Ok(())
    }

    /// Process an Invalidate mutation
    async fn process_invalidate(&self, args: &InvalidateArgs) -> Result<()> {
        // TODO: Implement actual invalidation in graph store
        log::info!("[Graph] Would invalidate: {:?}", args);
        // Simulate some async work
        tokio::time::sleep(tokio::time::Duration::from_millis(1)).await;
        Ok(())
    }
}

/// Create a new graph mutation consumer
pub fn create_graph_consumer(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
    db_path: &Path,
) -> Consumer<Graph> {
    let storage = Arc::new(Storage::readwrite(db_path));
    let processor = Graph::new(storage);
    Consumer::new(receiver, config, processor)
}

/// Create a new graph mutation consumer that chains to another processor
pub fn create_graph_consumer_with_next(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
    db_path: &Path,
    next: mpsc::Sender<crate::Mutation>,
) -> Consumer<Graph> {
    let storage = Arc::new(Storage::readwrite(db_path));
    let processor = Graph::new(storage);
    Consumer::with_next(receiver, config, processor, next)
}

/// Spawn the graph mutation consumer as a background task
pub fn spawn_graph_consumer(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
    db_path: &Path,
) -> JoinHandle<Result<()>> {
    let consumer = create_graph_consumer(receiver, config, db_path);
    crate::mutation::spawn_consumer(consumer)
}

/// Spawn the graph mutation consumer as a background task with chaining to next processor
pub fn spawn_graph_consumer_with_next(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
    db_path: &Path,
    next: mpsc::Sender<crate::Mutation>,
) -> JoinHandle<Result<()>> {
    let consumer = create_graph_consumer_with_next(receiver, config, db_path, next);
    crate::mutation::spawn_consumer(consumer)
}

#[cfg(test)]
#[path = "graph_tests.rs"]
mod graph_tests;
