//! Provides the graph-specific implementation for processing mutations from the MPSC queue
//! and writing them to the graph store.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use rocksdb::{Options, TransactionDB, TransactionDBOptions, DB};

use crate::query::{DstId, SrcId};
use crate::schema::{self, EdgeSummary, FragmentContent, NodeSummary};
use crate::TimestampMilli;
use crate::{
    mutation::{Consumer, Processor},
    schema::ALL_COLUMN_FAMILIES,
    AddEdge, AddFragment, AddNode, Id, InvalidateArgs, WriterConfig,
};

/// Trait for column family record types that can create and serialize key-value pairs.
pub(crate) trait ColumnFamilyRecord {
    const CF_NAME: &'static str;

    /// The key type for this column family
    type Key: Serialize + for<'de> Deserialize<'de>;

    /// The value type for this column family
    type Value: Serialize + for<'de> Deserialize<'de>;

    /// The argument type for creating records
    type CreateOp;

    /// Create a key-value pair from arguments
    fn record_from(args: &Self::CreateOp) -> (Self::Key, Self::Value);

    /// Create and serialize to bytes using MessagePack
    fn create_bytes(args: &Self::CreateOp) -> Result<(Vec<u8>, Vec<u8>), rmp_serde::encode::Error> {
        let (key, value) = Self::record_from(args);
        let key_bytes = rmp_serde::to_vec(&key)?;
        let value_bytes = rmp_serde::to_vec(&value)?;
        Ok((key_bytes, value_bytes))
    }

    /// Serialize the key to bytes using MessagePack
    fn key_to_bytes(key: &Self::Key) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        rmp_serde::to_vec(key)
    }

    /// Serialize the value to bytes using MessagePack
    fn value_to_bytes(value: &Self::Value) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        rmp_serde::to_vec(value)
    }

    /// Deserialize the key from bytes using MessagePack
    fn key_from_bytes(bytes: &[u8]) -> Result<Self::Key, rmp_serde::decode::Error> {
        rmp_serde::from_slice(bytes)
    }

    /// Deserialize the value from bytes using MessagePack
    fn value_from_bytes(bytes: &[u8]) -> Result<Self::Value, rmp_serde::decode::Error> {
        rmp_serde::from_slice(bytes)
    }
}

pub(crate) enum StorageOperation {
    PutCf(PutCf),
}

pub(crate) struct PutCf(pub(crate) &'static str, pub(crate) (Vec<u8>, Vec<u8>));

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
    pub(crate) fn db(&self) -> Result<&DB> {
        self.db
            .as_ref()
            .and_then(|handle| handle.as_db())
            .ok_or_else(|| anyhow::anyhow!("[Storage] Not in readonly mode or not ready"))
    }

    /// Get a reference to the TransactionDB (only works in readwrite mode)
    pub(crate) fn transaction_db(&self) -> Result<&TransactionDB> {
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
    /// Process an AddNode mutation
    async fn process_add_node(&self, args: &AddNode) -> Result<()> {
        log::info!("[Graph] About to insert node: {:?}", args);
        let txn_db = self.storage.transaction_db()?;

        // Create a transaction
        let txn = txn_db.transaction();
        for op in schema::Plan::create_node(args)? {
            match op {
                StorageOperation::PutCf(PutCf(cf_name, (key, value))) => {
                    let cf = txn_db
                        .cf_handle(cf_name)
                        .ok_or_else(|| anyhow::anyhow!("Column family '{}' not found", cf_name))?;
                    txn.put_cf(cf, key, value)?;
                }
            }
        }
        // Commit the transaction
        txn.commit()?;

        Ok(())
    }

    /// Process an AddEdge mutation
    async fn process_add_edge(&self, args: &AddEdge) -> Result<()> {
        log::info!("[Graph] About to insert edge: {:?}", args);
        let txn_db = self.storage.transaction_db()?;

        // Create a transaction
        let txn = txn_db.transaction();
        for op in schema::Plan::create_edge(args)? {
            match op {
                StorageOperation::PutCf(PutCf(cf_name, (key, value))) => {
                    let cf = txn_db
                        .cf_handle(cf_name)
                        .ok_or_else(|| anyhow::anyhow!("Column family '{}' not found", cf_name))?;
                    txn.put_cf(cf, key, value)?;
                }
            }
        }
        // Commit the transaction
        txn.commit()?;

        Ok(())
    }

    /// Process an AddFragment mutation
    async fn process_add_fragment(&self, args: &AddFragment) -> Result<()> {
        log::info!("[Graph] About to insert fragment: {:?}", args);
        let txn_db = self.storage.transaction_db()?;
        // Create a transaction
        let txn = txn_db.transaction();
        for op in schema::Plan::create_fragment(args)? {
            match op {
                StorageOperation::PutCf(PutCf(cf_name, (key, value))) => {
                    let cf = txn_db
                        .cf_handle(cf_name)
                        .ok_or_else(|| anyhow::anyhow!("Column family '{}' not found", cf_name))?;
                    txn.put_cf(cf, key, value)?;
                }
            }
        }
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

/// Implement query processor for Graph
#[async_trait::async_trait]
impl crate::query::Processor for Graph {
    async fn get_node_by_id(
        &self,
        query: &crate::query::NodeByIdQuery,
    ) -> Result<(schema::NodeName, NodeSummary)> {
        let id = query.id;

        let key = schema::NodeCfKey(id);
        let key_bytes = schema::Nodes::key_to_bytes(&key)
            .map_err(|e| anyhow::anyhow!("Failed to serialize key: {}", e))?;

        // Handle both readonly and readwrite modes
        let value_bytes = if let Ok(db) = self.storage.db() {
            let cf = db.cf_handle(schema::Nodes::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!("Column family '{}' not found", schema::Nodes::CF_NAME)
            })?;
            db.get_cf(cf, key_bytes)?
        } else {
            let txn_db = self.storage.transaction_db()?;
            let cf = txn_db.cf_handle(schema::Nodes::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!("Column family '{}' not found", schema::Nodes::CF_NAME)
            })?;
            txn_db.get_cf(cf, key_bytes)?
        };

        let value_bytes = value_bytes.ok_or_else(|| anyhow::anyhow!("Node not found: {}", id))?;

        let value: schema::NodeCfValue = schema::Nodes::value_from_bytes(&value_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

        Ok((value.0, value.1))
    }

    async fn get_edge_summary_by_id(
        &self,
        query: &crate::query::EdgeSummaryByIdQuery,
    ) -> Result<EdgeSummary> {
        let id = query.id;
        let key = schema::EdgeCfKey(id);
        let key_bytes = schema::Edges::key_to_bytes(&key)
            .map_err(|e| anyhow::anyhow!("Failed to serialize key: {}", e))?;

        // Handle both readonly and readwrite modes
        let value_bytes = if let Ok(db) = self.storage.db() {
            let cf = db.cf_handle(schema::Edges::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!("Column family '{}' not found", schema::Edges::CF_NAME)
            })?;
            db.get_cf(cf, key_bytes)?
        } else {
            let txn_db = self.storage.transaction_db()?;
            let cf = txn_db.cf_handle(schema::Edges::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!("Column family '{}' not found", schema::Edges::CF_NAME)
            })?;
            txn_db.get_cf(cf, key_bytes)?
        };

        let value_bytes = value_bytes.ok_or_else(|| anyhow::anyhow!("Edge not found: {}", id))?;

        let value: schema::EdgeCfValue = schema::Edges::value_from_bytes(&value_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

        Ok(value.0)
    }

    async fn get_edge_summary_by_src_dst_name(
        &self,
        query: &crate::query::EdgeSummaryBySrcDstNameQuery,
    ) -> Result<(Id, EdgeSummary)> {
        let source_id = query.source_id;
        let dest_id = query.dest_id;
        let name = &query.name;

        let key = schema::ForwardEdgeCfKey(
            schema::EdgeSourceId(source_id),
            schema::EdgeDestinationId(dest_id),
            schema::EdgeName(name.clone()),
        );
        let key_bytes = schema::ForwardEdges::key_to_bytes(&key)
            .map_err(|e| anyhow::anyhow!("Failed to serialize key: {}", e))?;

        // Handle both readonly and readwrite modes
        let value_bytes = if let Ok(db) = self.storage.db() {
            let cf = db.cf_handle(schema::ForwardEdges::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!(
                    "Column family '{}' not found",
                    schema::ForwardEdges::CF_NAME
                )
            })?;
            db.get_cf(cf, key_bytes)?
        } else {
            let txn_db = self.storage.transaction_db()?;
            let cf = txn_db
                .cf_handle(schema::ForwardEdges::CF_NAME)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Column family '{}' not found",
                        schema::ForwardEdges::CF_NAME
                    )
                })?;
            txn_db.get_cf(cf, key_bytes)?
        };

        let value_bytes = value_bytes.ok_or_else(|| {
            anyhow::anyhow!(
                "Edge not found: source={}, dest={}, name={}",
                source_id,
                dest_id,
                name
            )
        })?;

        let value: schema::ForwardEdgeCfValue =
            schema::ForwardEdges::value_from_bytes(&value_bytes)
                .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;

        // The ForwardEdge value only contains the edge ID, so we need to look up the EdgeSummary
        let edge_id = value.0;

        // Look up the edge summary from the edges column family
        let edge_key = schema::EdgeCfKey(edge_id);
        let edge_key_bytes = schema::Edges::key_to_bytes(&edge_key)
            .map_err(|e| anyhow::anyhow!("Failed to serialize edge key: {}", e))?;

        let edge_value_bytes = if let Ok(db) = self.storage.db() {
            let cf = db.cf_handle(schema::Edges::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!("Column family '{}' not found", schema::Edges::CF_NAME)
            })?;
            db.get_cf(cf, edge_key_bytes)?
        } else {
            let txn_db = self.storage.transaction_db()?;
            let cf = txn_db.cf_handle(schema::Edges::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!("Column family '{}' not found", schema::Edges::CF_NAME)
            })?;
            txn_db.get_cf(cf, edge_key_bytes)?
        };

        let edge_value_bytes = edge_value_bytes.ok_or_else(|| {
            anyhow::anyhow!("Edge summary not found for edge_id: {}", edge_id)
        })?;

        let edge_value: schema::EdgeCfValue = schema::Edges::value_from_bytes(&edge_value_bytes)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize edge value: {}", e))?;

        Ok((edge_id, edge_value.0))
    }

    async fn get_fragment_content_by_id(
        &self,
        query: &crate::query::FragmentContentByIdQuery,
    ) -> Result<Vec<(TimestampMilli, FragmentContent)>> {
        let id = query.id;

        // Scan the fragments column family for all fragments with this ID
        // Keys are (id, timestamp) and RocksDB stores them in sorted order,
        // so fragments will naturally be in chronological order

        let mut fragments: Vec<(TimestampMilli, FragmentContent)> = Vec::new();

        // Handle both readonly and readwrite modes
        if let Ok(db) = self.storage.db() {
            let cf = db.cf_handle(schema::Fragments::CF_NAME).ok_or_else(|| {
                anyhow::anyhow!("Column family '{}' not found", schema::Fragments::CF_NAME)
            })?;

            let iter = db.iterator_cf(cf, rocksdb::IteratorMode::Start);
            for item in iter {
                let (key_bytes, value_bytes) = item?;
                let key: schema::FragmentCfKey = schema::Fragments::key_from_bytes(&key_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

                if key.0 == id {
                    let value: schema::FragmentCfValue =
                        schema::Fragments::value_from_bytes(&value_bytes)
                            .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;
                    fragments.push((key.1, value.0));
                } else if key.0 > id {
                    // Keys are sorted, so once we pass the target ID, we can stop
                    break;
                }
            }
        } else {
            let txn_db = self.storage.transaction_db()?;
            let cf = txn_db
                .cf_handle(schema::Fragments::CF_NAME)
                .ok_or_else(|| {
                    anyhow::anyhow!("Column family '{}' not found", schema::Fragments::CF_NAME)
                })?;

            let iter = txn_db.iterator_cf(cf, rocksdb::IteratorMode::Start);
            for item in iter {
                let (key_bytes, value_bytes) = item?;
                let key: schema::FragmentCfKey = schema::Fragments::key_from_bytes(&key_bytes)
                    .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

                if key.0 == id {
                    let value: schema::FragmentCfValue =
                        schema::Fragments::value_from_bytes(&value_bytes)
                            .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;
                    fragments.push((key.1, value.0));
                } else if key.0 > id {
                    break;
                }
            }
        }

        Ok(fragments)
    }

    async fn get_edges_from_node_by_id(
        &self,
        query: &crate::query::EdgesFromNodeByIdQuery,
    ) -> Result<Vec<(SrcId, crate::schema::EdgeName, DstId)>> {
        let id = query.id;

        // Scan the forward_edges column family for all edges with this source ID
        // Keys are (source_id, dest_id, name) and RocksDB stores them in sorted order

        let mut edges: Vec<(SrcId, crate::schema::EdgeName, DstId)> = Vec::new();

        // Handle both readonly and readwrite modes
        if let Ok(db) = self.storage.db() {
            let cf = db
                .cf_handle(schema::ForwardEdges::CF_NAME)
                .ok_or_else(|| {
                    anyhow::anyhow!("Column family '{}' not found", schema::ForwardEdges::CF_NAME)
                })?;

            let iter = db.iterator_cf(cf, rocksdb::IteratorMode::Start);
            for item in iter {
                let (key_bytes, _value_bytes) = item?;
                let key: schema::ForwardEdgeCfKey =
                    schema::ForwardEdges::key_from_bytes(&key_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

                let source_id = key.0 .0;
                if source_id == id {
                    let dest_id = key.1 .0;
                    let edge_name = key.2 .0;
                    edges.push((source_id, crate::schema::EdgeName(edge_name), dest_id));
                } else if source_id > id {
                    // Keys are sorted by source_id first, so once we pass the target ID, stop
                    break;
                }
            }
        } else {
            let txn_db = self.storage.transaction_db()?;
            let cf = txn_db
                .cf_handle(schema::ForwardEdges::CF_NAME)
                .ok_or_else(|| {
                    anyhow::anyhow!("Column family '{}' not found", schema::ForwardEdges::CF_NAME)
                })?;

            let iter = txn_db.iterator_cf(cf, rocksdb::IteratorMode::Start);
            for item in iter {
                let (key_bytes, _value_bytes) = item?;
                let key: schema::ForwardEdgeCfKey =
                    schema::ForwardEdges::key_from_bytes(&key_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

                let source_id = key.0 .0;
                if source_id == id {
                    let dest_id = key.1 .0;
                    let edge_name = key.2 .0;
                    edges.push((source_id, crate::schema::EdgeName(edge_name), dest_id));
                } else if source_id > id {
                    break;
                }
            }
        }

        Ok(edges)
    }

    async fn get_edges_to_node_by_id(
        &self,
        query: &crate::query::EdgesToNodeByIdQuery,
    ) -> Result<Vec<(DstId, crate::schema::EdgeName, SrcId)>> {
        let id = query.id;

        // Scan the reverse_edges column family for all edges with this destination ID
        // Keys are (dest_id, source_id, name) and RocksDB stores them in sorted order

        let mut edges: Vec<(DstId, crate::schema::EdgeName, SrcId)> = Vec::new();

        // Handle both readonly and readwrite modes
        if let Ok(db) = self.storage.db() {
            let cf = db
                .cf_handle(schema::ReverseEdges::CF_NAME)
                .ok_or_else(|| {
                    anyhow::anyhow!("Column family '{}' not found", schema::ReverseEdges::CF_NAME)
                })?;

            let iter = db.iterator_cf(cf, rocksdb::IteratorMode::Start);
            for item in iter {
                let (key_bytes, _value_bytes) = item?;
                let key: schema::ReverseEdgeCfKey =
                    schema::ReverseEdges::key_from_bytes(&key_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

                let dest_id = key.0 .0;
                if dest_id == id {
                    let source_id = key.1 .0;
                    let edge_name = key.2 .0;
                    edges.push((dest_id, crate::schema::EdgeName(edge_name), source_id));
                } else if dest_id > id {
                    // Keys are sorted by dest_id first, so once we pass the target ID, stop
                    break;
                }
            }
        } else {
            let txn_db = self.storage.transaction_db()?;
            let cf = txn_db
                .cf_handle(schema::ReverseEdges::CF_NAME)
                .ok_or_else(|| {
                    anyhow::anyhow!("Column family '{}' not found", schema::ReverseEdges::CF_NAME)
                })?;

            let iter = txn_db.iterator_cf(cf, rocksdb::IteratorMode::Start);
            for item in iter {
                let (key_bytes, _value_bytes) = item?;
                let key: schema::ReverseEdgeCfKey =
                    schema::ReverseEdges::key_from_bytes(&key_bytes)
                        .map_err(|e| anyhow::anyhow!("Failed to deserialize key: {}", e))?;

                let dest_id = key.0 .0;
                if dest_id == id {
                    let source_id = key.1 .0;
                    let edge_name = key.2 .0;
                    edges.push((dest_id, crate::schema::EdgeName(edge_name), source_id));
                } else if dest_id > id {
                    break;
                }
            }
        }

        Ok(edges)
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

/// Create a new query consumer for the graph
pub fn create_query_consumer(
    receiver: flume::Receiver<crate::query::Query>,
    config: crate::ReaderConfig,
    db_path: &Path,
) -> crate::query::Consumer<Graph> {
    let mut storage = Storage::readonly(db_path);
    storage.ready().expect("Failed to ready storage");
    let storage = Arc::new(storage);
    let processor = Graph::new(storage);
    crate::query::Consumer::new(receiver, config, processor)
}

/// Spawn a query consumer as a background task
pub fn spawn_query_consumer(
    receiver: flume::Receiver<crate::query::Query>,
    config: crate::ReaderConfig,
    db_path: &Path,
) -> JoinHandle<Result<()>> {
    let consumer = create_query_consumer(receiver, config, db_path);
    crate::query::spawn_consumer(consumer)
}

#[cfg(test)]
#[path = "graph_tests.rs"]
mod graph_tests;
