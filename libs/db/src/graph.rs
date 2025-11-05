//! Provides the graph-specific implementation for processing mutations from the MPSC queue
//! and writing them to the graph store.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use rocksdb::{Options, TransactionDB, TransactionDBOptions, DB};

use crate::{
    mutation::{Consumer, Processor},
    schema::ALL_COLUMN_FAMILIES,
    AddEdgeArgs, AddFragmentArgs, AddNodeArgs, InvalidateArgs, WriterConfig,
};

/// Handle for either a read-only DB or read-write TransactionDB
enum DatabaseHandle {
    ReadOnly(DB),
    ReadWrite(TransactionDB),
}

impl DatabaseHandle {
    /// Get TransactionDB if in ReadWrite mode, otherwise None
    fn as_transaction_db(&self) -> Option<&TransactionDB> {
        match self {
            DatabaseHandle::ReadWrite(txn_db) => Some(txn_db),
            DatabaseHandle::ReadOnly(_) => None,
        }
    }

    /// Get DB if in ReadOnly mode, otherwise None
    fn as_db(&self) -> Option<&DB> {
        match self {
            DatabaseHandle::ReadOnly(db) => Some(db),
            DatabaseHandle::ReadWrite(_) => None,
        }
    }

    /// Check if this is a read-write handle
    fn is_read_write(&self) -> bool {
        matches!(self, DatabaseHandle::ReadWrite(_))
    }
}

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
    txn_db_options: TransactionDBOptions,
    db: Option<DatabaseHandle>,
    mode: StorageMode,
    column_families: &'static [&'static str],
}

impl Storage {
    /// Create a new Storage instance in readonly mode
    pub fn readonly(db_path: &Path) -> Self {
        Self {
            db_path: PathBuf::from(db_path),
            db_options: StorageOptions::default_for_readonly(),
            txn_db_options: TransactionDBOptions::default(),
            db: None,
            mode: StorageMode::ReadOnly,
            column_families: ALL_COLUMN_FAMILIES,
        }
    }

    /// Create a new Storage instance in readwrite mode
    pub fn readwrite(db_path: &Path) -> Self {
        Self {
            db_path: PathBuf::from(db_path),
            db_options: StorageOptions::default_for_readwrite(),
            txn_db_options: TransactionDBOptions::default(),
            db: None,
            mode: StorageMode::ReadWrite,
            column_families: ALL_COLUMN_FAMILIES,
        }
    }

    /// Create a new Storage instance in readwrite mode with custom options
    pub fn readwrite_with_options(
        db_path: &Path,
        db_options: rocksdb::Options,
        txn_db_options: TransactionDBOptions,
    ) -> Self {
        Self {
            db_path: PathBuf::from(db_path),
            db_options,
            txn_db_options,
            db: None,
            mode: StorageMode::ReadWrite,
            column_families: ALL_COLUMN_FAMILIES,
        }
    }

    /// Close the database
    pub fn close(&mut self) -> Result<()> {
        if self.db.is_none() {
            return Err(anyhow::anyhow!("[Storage] Storage is not ready. "));
        }
        if let Some(db_handle) = self.db.take() {
            // TransactionDB manages persistence automatically through WAL
            // Regular DB in readonly mode doesn't need flushing
            drop(db_handle); // drop the reference for automatic closing the database.
        }
        Ok(())
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
                let db = DB::open_cf_for_read_only(
                    &self.db_options,
                    &self.db_path,
                    self.column_families,
                    false,
                )?;
                self.db = Some(DatabaseHandle::ReadOnly(db));
            }
            StorageMode::ReadWrite => {
                let txn_db = TransactionDB::open_cf(
                    &self.db_options,
                    &self.txn_db_options,
                    &self.db_path,
                    self.column_families,
                )?;
                self.db = Some(DatabaseHandle::ReadWrite(txn_db));
            }
        }

        log::info!("[Storage] Ready");
        Ok(())
    }

    /// Get a reference to the underlying DB (only works in readonly mode)
    pub fn db(&self) -> Result<&DB> {
        self.db
            .as_ref()
            .and_then(|handle| handle.as_db())
            .ok_or_else(|| anyhow::anyhow!("[Storage] Not in readonly mode or not ready"))
    }

    /// Get a reference to the TransactionDB (only works in readwrite mode)
    pub fn transaction_db(&self) -> Result<&TransactionDB> {
        self.db
            .as_ref()
            .and_then(|handle| handle.as_transaction_db())
            .ok_or_else(|| anyhow::anyhow!("[Storage] Not in readwrite mode or not ready"))
    }

    /// Check if storage is in readwrite mode with TransactionDB
    pub fn is_transactional(&self) -> bool {
        self.db
            .as_ref()
            .map(|handle| handle.is_read_write())
            .unwrap_or(false)
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
    async fn process_add_vertex(&self, args: &AddNodeArgs) -> Result<()> {
        let txn_db = self.storage.transaction_db()?;

        // Create a transaction
        let txn = txn_db.transaction();

        // TODO: Implement actual vertex insertion
        // Example: txn.put_cf(cf_handle, key, value)?;
        log::info!("[Graph] Would insert vertex: {:?}", args);

        // Commit the transaction
        txn.commit()?;

        Ok(())
    }

    /// Process an AddEdge mutation
    async fn process_add_edge(&self, args: &AddEdgeArgs) -> Result<()> {
        let txn_db = self.storage.transaction_db()?;

        // Create a transaction
        let txn = txn_db.transaction();

        // TODO: Implement actual edge insertion
        // Example: txn.put_cf(cf_handle, key, value)?;
        log::info!("[Graph] Would insert edge: {:?}", args);

        // Commit the transaction
        txn.commit()?;

        Ok(())
    }

    /// Process an AddFragment mutation
    async fn process_add_fragment(&self, args: &AddFragmentArgs) -> Result<()> {
        let txn_db = self.storage.transaction_db()?;

        // Create a transaction
        let txn = txn_db.transaction();

        // TODO: Implement actual fragment insertion
        // Example: txn.put_cf(cf_handle, key, value)?;
        log::info!("[Graph] Would insert fragment: {:?}", args);

        // Commit the transaction
        txn.commit()?;

        Ok(())
    }

    /// Process an Invalidate mutation
    async fn process_invalidate(&self, args: &InvalidateArgs) -> Result<()> {
        let txn_db = self.storage.transaction_db()?;

        // Create a transaction
        let txn = txn_db.transaction();

        // TODO: Implement actual invalidation
        // Example: txn.delete_cf(cf_handle, key)?;
        log::info!("[Graph] Would invalidate: {:?}", args);

        // Commit the transaction
        txn.commit()?;

        Ok(())
    }
}

/// Create a new graph mutation consumer
pub fn create_graph_consumer(
    receiver: mpsc::Receiver<crate::Mutation>,
    config: WriterConfig,
    db_path: &Path,
) -> Consumer<Graph> {
    let mut storage = Storage::readwrite(db_path);
    storage.ready().expect("Failed to ready storage");
    let storage = Arc::new(storage);
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
    let mut storage = Storage::readwrite(db_path);
    storage.ready().expect("Failed to ready storage");
    let storage = Arc::new(storage);
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
