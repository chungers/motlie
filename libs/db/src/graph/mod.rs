//! Graph module - RocksDB-based graph storage.
//!
//! This module provides the graph-specific implementation for processing mutations from
//! the MPSC queue and writing them to the graph store.
//!
//! ## Module Structure
//!
//! - `mod.rs` - Storage, Graph struct, and module exports
//! - `schema.rs` - RocksDB schema definitions (column families)
//! - `mutation.rs` - Mutation types (AddNode, AddEdge, etc.)
//! - `writer.rs` - Writer infrastructure and mutation consumers
//! - `query.rs` - Query types (NodeById, EdgeSummaryBySrcDstName, etc.)
//! - `reader.rs` - Reader infrastructure and query consumers
//! - `scan.rs` - Scan API for pagination

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use rocksdb::{Options, TransactionDB, TransactionDBOptions, DB};

// Submodules
pub mod mutation;
pub mod name_hash;
pub mod query;
pub mod reader;
pub mod schema;
pub mod transaction;
pub mod writer;

/// Scan API for iterating over column families with pagination support.
pub mod scan;

#[cfg(test)]
mod tests;

// Re-export commonly used types from submodules
pub use mutation::{
    AddEdge, AddEdgeFragment, AddNode, AddNodeFragment, Mutation, MutationBatch,
    UpdateEdgeValidSinceUntil, UpdateEdgeWeight, UpdateNodeValidSinceUntil,
};
pub use crate::writer::Runnable;
pub use query::{
    EdgeFragmentsByIdTimeRange, EdgeSummaryBySrcDstName, IncomingEdges, NodeById,
    NodeFragmentsByIdTimeRange, OutgoingEdges, Query, TransactionQueryExecutor,
};
pub use transaction::Transaction;
pub use crate::reader::Runnable as QueryRunnable;
pub use reader::{
    // Query consumer functions
    create_query_consumer,
    create_query_consumer_readwrite,
    create_query_reader,
    spawn_query_consumer,
    spawn_query_consumer_pool_readonly,
    spawn_query_consumer_pool_shared,
    spawn_query_consumer_readwrite,
    spawn_query_consumer_with_graph,
    Consumer as QueryConsumer,
    Processor as ReaderProcessor,
    QueryExecutor,
    QueryProcessor,
    QueryWithTimeout,
    Reader,
    ReaderConfig,
};
pub use name_hash::{NameCache, NameHash};
pub use schema::{DstId, EdgeName, EdgeSummary, FragmentContent, NodeName, NodeSummary, SrcId};
pub use writer::{
    // Mutation consumer functions
    create_mutation_consumer,
    create_mutation_consumer_with_next,
    create_mutation_writer,
    spawn_mutation_consumer,
    spawn_mutation_consumer_with_graph,
    spawn_mutation_consumer_with_next,
    Consumer as MutationConsumer,
    MutationExecutor,
    Processor as MutationProcessor,
    Writer,
    WriterConfig,
};

// Internal imports
use schema::ALL_COLUMN_FAMILIES;
use writer::Processor;

/// Trait for column family record types that can create and serialize key-value pairs.
///
/// This trait uses direct byte concatenation for keys (to enable RocksDB prefix extractors)
/// and MessagePack for values (where self-describing format is beneficial).
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

    /// Serialize the key to bytes using direct concatenation (no MessagePack).
    /// This enables constant-length prefixes for RocksDB prefix extractors.
    fn key_to_bytes(key: &Self::Key) -> Vec<u8>;

    /// Deserialize the key from bytes (direct format, no MessagePack).
    fn key_from_bytes(bytes: &[u8]) -> Result<Self::Key, anyhow::Error>;

    /// Serialize the value to bytes using MessagePack, then compress with LZ4.
    fn value_to_bytes(value: &Self::Value) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        let msgpack_bytes = rmp_serde::to_vec(value)?;
        let compressed = lz4::block::compress(&msgpack_bytes, None, true).map_err(|e| {
            rmp_serde::encode::Error::Syntax(format!("LZ4 compression failed: {}", e))
        })?;
        Ok(compressed)
    }

    /// Decompress with LZ4, then deserialize the value from bytes using MessagePack.
    fn value_from_bytes(bytes: &[u8]) -> Result<Self::Value, rmp_serde::decode::Error> {
        let decompressed = lz4::block::decompress(bytes, None).map_err(|e| {
            rmp_serde::decode::Error::Syntax(format!("LZ4 decompression failed: {}", e))
        })?;
        rmp_serde::from_slice(&decompressed)
    }

    /// Create and serialize to bytes using direct encoding for keys, compressed MessagePack for values.
    fn create_bytes(args: &Self::CreateOp) -> Result<(Vec<u8>, Vec<u8>), rmp_serde::encode::Error> {
        let (key, value) = Self::record_from(args);
        let key_bytes = Self::key_to_bytes(&key);
        let value_bytes = Self::value_to_bytes(&value)?;
        Ok((key_bytes, value_bytes))
    }

    /// Configure RocksDB options for this column family.
    /// Each implementer specifies their own prefix extractor and bloom filter settings.
    fn column_family_options() -> rocksdb::Options {
        rocksdb::Options::default()
    }
}

/// Trait implemented by column families that supports patching of TemporalRange.
pub(crate) trait ValidRangePatchable {
    fn patch_valid_range(
        &self,
        old_value: &[u8],
        new_range: schema::TemporalRange,
    ) -> Result<Vec<u8>, anyhow::Error>;
}

/// Handle for either a read-only DB or read-write TransactionDB
enum DatabaseHandle {
    ReadOnly(DB),
    ReadWrite(TransactionDB),
    Secondary(DB),
}

impl DatabaseHandle {
    /// Get TransactionDB if in ReadWrite mode, otherwise None
    fn as_transaction_db(&self) -> Option<&TransactionDB> {
        match self {
            DatabaseHandle::ReadWrite(txn_db) => Some(txn_db),
            DatabaseHandle::ReadOnly(_) | DatabaseHandle::Secondary(_) => None,
        }
    }

    /// Get DB if in ReadOnly or Secondary mode, otherwise None
    fn as_db(&self) -> Option<&DB> {
        match self {
            DatabaseHandle::ReadOnly(db) | DatabaseHandle::Secondary(db) => Some(db),
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
    Secondary { secondary_path: PathBuf },
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

    /// Default options for Storage in Secondary mode
    ///
    /// Key requirements for secondary instances:
    /// - max_open_files = -1 (REQUIRED: must keep all file descriptors open)
    /// - create_if_missing = false
    /// - create_missing_column_families = false
    pub fn default_for_secondary() -> Options {
        let mut options = Options::default();
        options.set_error_if_exists(false);
        options.create_if_missing(false);
        options.create_missing_column_families(false);
        options.set_max_open_files(-1); // REQUIRED for secondary instances
        options
    }
}

/// Configuration for name cache pre-warming
#[derive(Debug, Clone)]
pub struct NameCacheConfig {
    /// Maximum number of names to pre-load from Names CF on startup.
    /// Set to 0 to disable pre-warming.
    /// Default: 1000
    pub prewarm_limit: usize,
}

impl Default for NameCacheConfig {
    fn default() -> Self {
        Self { prewarm_limit: 1000 }
    }
}

/// Configuration for RocksDB block cache.
///
/// RocksDB's block cache stores uncompressed data blocks (~4KB each).
/// Sharing a single cache across all column families allows RocksDB to
/// dynamically allocate memory based on access patterns.
///
/// See [RocksDB Block Cache Wiki](https://github.com/facebook/rocksdb/wiki/Block-Cache)
/// for detailed documentation.
#[derive(Debug, Clone)]
pub struct BlockCacheConfig {
    /// Total block cache size in bytes.
    /// Default: 256MB. Production deployments should tune to ~1/3 of available memory.
    pub cache_size_bytes: usize,

    /// Block size for graph column families (Nodes, ForwardEdges, ReverseEdges, Names).
    /// Default: 4KB. Optimal for 40-byte fixed keys (~100 keys per block).
    pub graph_block_size: usize,

    /// Block size for fragment column families (NodeFragments, EdgeFragments).
    /// Default: 16KB. Better for larger variable-length content.
    pub fragment_block_size: usize,

    /// Whether to cache index and filter blocks in the block cache.
    /// Default: true. Keeps hot metadata in cache for faster lookups.
    pub cache_index_and_filter_blocks: bool,

    /// Whether to pin L0 filter and index blocks in cache.
    /// Default: true. Prevents eviction of newest (most likely accessed) data.
    pub pin_l0_filter_and_index: bool,
}

impl Default for BlockCacheConfig {
    fn default() -> Self {
        Self {
            cache_size_bytes: 256 * 1024 * 1024,  // 256MB
            graph_block_size: 4 * 1024,            // 4KB
            fragment_block_size: 16 * 1024,        // 16KB
            cache_index_and_filter_blocks: true,
            pin_l0_filter_and_index: true,
        }
    }
}

/// Static configuration info for the graph database subsystem.
///
/// Used by the `motlie info` command to display graph DB settings.
/// Implements [`motlie_core::telemetry::SubsystemInfo`] for consistent formatting.
///
/// # Example
///
/// ```ignore
/// use motlie_db::graph::SystemInfo;
/// use motlie_core::telemetry::{format_subsystem_info, SubsystemInfo};
///
/// let info = SystemInfo::default();
/// println!("{}", format_subsystem_info(&info));
/// ```
#[derive(Debug, Clone)]
pub struct SystemInfo {
    /// Block cache configuration
    pub block_cache_config: BlockCacheConfig,
    /// Name cache configuration
    pub name_cache_config: NameCacheConfig,
    /// List of column families
    pub column_families: Vec<&'static str>,
}

impl Default for SystemInfo {
    fn default() -> Self {
        Self {
            block_cache_config: BlockCacheConfig::default(),
            name_cache_config: NameCacheConfig::default(),
            column_families: ALL_COLUMN_FAMILIES.to_vec(),
        }
    }
}

impl motlie_core::telemetry::SubsystemInfo for SystemInfo {
    fn name(&self) -> &'static str {
        "Graph Database (RocksDB)"
    }

    fn info_lines(&self) -> Vec<(&'static str, String)> {
        vec![
            ("Block Cache Size", format_bytes(self.block_cache_config.cache_size_bytes)),
            ("Graph Block Size", format_bytes(self.block_cache_config.graph_block_size)),
            ("Fragment Block Size", format_bytes(self.block_cache_config.fragment_block_size)),
            ("Cache Index/Filter", self.block_cache_config.cache_index_and_filter_blocks.to_string()),
            ("Pin L0 Blocks", self.block_cache_config.pin_l0_filter_and_index.to_string()),
            ("Name Cache Prewarm", self.name_cache_config.prewarm_limit.to_string()),
            ("Column Families", self.column_families.join(", ")),
        ]
    }
}

/// Format a byte count as a human-readable string.
fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{} GB", bytes / (1024 * 1024 * 1024))
    } else if bytes >= 1024 * 1024 {
        format!("{} MB", bytes / (1024 * 1024))
    } else if bytes >= 1024 {
        format!("{} KB", bytes / 1024)
    } else {
        format!("{} B", bytes)
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
    /// Thread-safe name cache for interning and resolution
    name_cache: std::sync::Arc<NameCache>,
    /// Configuration for name cache behavior
    name_cache_config: NameCacheConfig,
    /// Shared block cache across all column families
    block_cache: Option<rocksdb::Cache>,
    /// Configuration for block cache behavior
    block_cache_config: BlockCacheConfig,
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
            name_cache: std::sync::Arc::new(NameCache::new()),
            name_cache_config: NameCacheConfig::default(),
            block_cache: None,
            block_cache_config: BlockCacheConfig::default(),
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
            name_cache: std::sync::Arc::new(NameCache::new()),
            name_cache_config: NameCacheConfig::default(),
            block_cache: None,
            block_cache_config: BlockCacheConfig::default(),
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
            name_cache: std::sync::Arc::new(NameCache::new()),
            name_cache_config: NameCacheConfig::default(),
            block_cache: None,
            block_cache_config: BlockCacheConfig::default(),
        }
    }

    /// Create a new Storage instance in secondary mode
    ///
    /// Secondary instances can catch up with the primary database via
    /// `try_catch_up_with_primary()`, making them ideal for read replicas
    /// with ongoing writes.
    ///
    /// # Arguments
    /// * `primary_path` - Path to the primary database
    /// * `secondary_path` - Path for secondary's MANIFEST (must be different from primary)
    ///
    /// # Example
    /// ```no_run
    /// use std::path::PathBuf;
    /// use motlie_db::graph::Storage;
    ///
    /// let primary = PathBuf::from("/data/db");
    /// let secondary = primary.join("secondary");
    ///
    /// let mut storage = Storage::secondary(&primary, &secondary);
    /// storage.ready().unwrap();
    ///
    /// // Periodically catch up with primary
    /// storage.try_catch_up_with_primary().unwrap();
    /// ```
    pub fn secondary(primary_path: &Path, secondary_path: &Path) -> Self {
        Self {
            db_path: PathBuf::from(primary_path),
            db_options: StorageOptions::default_for_secondary(),
            txn_db_options: TransactionDBOptions::default(),
            db: None,
            mode: StorageMode::Secondary {
                secondary_path: PathBuf::from(secondary_path),
            },
            column_families: ALL_COLUMN_FAMILIES,
            name_cache: std::sync::Arc::new(NameCache::new()),
            name_cache_config: NameCacheConfig::default(),
            block_cache: None,
            block_cache_config: BlockCacheConfig::default(),
        }
    }

    /// Set the name cache configuration.
    ///
    /// Must be called before `ready()` to take effect.
    pub fn with_name_cache_config(mut self, config: NameCacheConfig) -> Self {
        self.name_cache_config = config;
        self
    }

    /// Set the block cache configuration.
    ///
    /// Must be called before `ready()` to take effect.
    ///
    /// # Example
    /// ```no_run
    /// use motlie_db::graph::{Storage, BlockCacheConfig};
    /// use std::path::Path;
    ///
    /// let config = BlockCacheConfig {
    ///     cache_size_bytes: 512 * 1024 * 1024, // 512MB
    ///     ..Default::default()
    /// };
    ///
    /// let mut storage = Storage::readwrite(Path::new("/data/db"))
    ///     .with_block_cache_config(config);
    /// storage.ready().unwrap();
    /// ```
    pub fn with_block_cache_config(mut self, config: BlockCacheConfig) -> Self {
        self.block_cache_config = config;
        self
    }

    /// Get a reference to the name cache.
    ///
    /// The cache is pre-warmed during `ready()` if `prewarm_limit > 0`.
    pub fn name_cache(&self) -> &std::sync::Arc<NameCache> {
        &self.name_cache
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
    #[tracing::instrument(skip(self), fields(path = ?self.db_path))]
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

        // Create shared block cache for all column families
        let cache = rocksdb::Cache::new_lru_cache(self.block_cache_config.cache_size_bytes);
        self.block_cache = Some(cache);
        let cache_ref = self.block_cache.as_ref().unwrap();

        tracing::info!(
            "[Storage] Created shared block cache: {} MB, graph_block_size: {} KB, fragment_block_size: {} KB",
            self.block_cache_config.cache_size_bytes / (1024 * 1024),
            self.block_cache_config.graph_block_size / 1024,
            self.block_cache_config.fragment_block_size / 1024
        );

        // Create column family descriptors with shared cache and tuned options
        use rocksdb::ColumnFamilyDescriptor;
        let cf_descriptors = vec![
            ColumnFamilyDescriptor::new(
                schema::Names::CF_NAME,
                schema::Names::column_family_options_with_cache(cache_ref, &self.block_cache_config),
            ),
            ColumnFamilyDescriptor::new(
                schema::Nodes::CF_NAME,
                schema::Nodes::column_family_options_with_cache(cache_ref, &self.block_cache_config),
            ),
            ColumnFamilyDescriptor::new(
                schema::NodeFragments::CF_NAME,
                schema::NodeFragments::column_family_options_with_cache(cache_ref, &self.block_cache_config),
            ),
            ColumnFamilyDescriptor::new(
                schema::EdgeFragments::CF_NAME,
                schema::EdgeFragments::column_family_options_with_cache(cache_ref, &self.block_cache_config),
            ),
            ColumnFamilyDescriptor::new(
                schema::ForwardEdges::CF_NAME,
                schema::ForwardEdges::column_family_options_with_cache(cache_ref, &self.block_cache_config),
            ),
            ColumnFamilyDescriptor::new(
                schema::ReverseEdges::CF_NAME,
                schema::ReverseEdges::column_family_options_with_cache(cache_ref, &self.block_cache_config),
            ),
        ];

        match &self.mode {
            StorageMode::ReadOnly => {
                let db = DB::open_cf_descriptors_read_only(
                    &self.db_options,
                    &self.db_path,
                    cf_descriptors,
                    false,
                )?;
                self.db = Some(DatabaseHandle::ReadOnly(db));
            }
            StorageMode::ReadWrite => {
                let txn_db = TransactionDB::open_cf_descriptors(
                    &self.db_options,
                    &self.txn_db_options,
                    &self.db_path,
                    cf_descriptors,
                )?;
                self.db = Some(DatabaseHandle::ReadWrite(txn_db));
            }
            StorageMode::Secondary { secondary_path } => {
                let db = DB::open_cf_descriptors_as_secondary(
                    &self.db_options,
                    &self.db_path,
                    secondary_path,
                    cf_descriptors,
                )?;
                self.db = Some(DatabaseHandle::Secondary(db));
            }
        }

        // Pre-warm the name cache if configured
        if self.name_cache_config.prewarm_limit > 0 {
            let loaded = self.prewarm_name_cache()?;
            tracing::info!(
                "[Storage] Pre-warmed name cache with {} entries (limit: {})",
                loaded,
                self.name_cache_config.prewarm_limit
            );
        }

        tracing::info!("[Storage] Ready");
        Ok(())
    }

    /// Pre-warm the name cache by loading entries from the Names CF.
    ///
    /// Returns the number of entries loaded.
    fn prewarm_name_cache(&self) -> Result<usize> {
        use rocksdb::IteratorMode;

        let limit = self.name_cache_config.prewarm_limit;
        let mut loaded = 0;

        // Get iterator based on storage mode
        if let Ok(db) = self.db() {
            let names_cf = db
                .cf_handle(schema::Names::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("Names CF not found"))?;

            for item in db.iterator_cf(names_cf, IteratorMode::Start) {
                if loaded >= limit {
                    break;
                }

                let (key_bytes, value_bytes) = item?;

                // Parse key (NameHash) and value (name string)
                if key_bytes.len() == 8 {
                    let mut hash_bytes = [0u8; 8];
                    hash_bytes.copy_from_slice(&key_bytes);
                    let hash = NameHash::from_bytes(hash_bytes);

                    if let Ok(value) = schema::Names::value_from_bytes(&value_bytes) {
                        self.name_cache.insert(hash, value.0);
                        loaded += 1;
                    }
                }
            }
        } else if let Ok(txn_db) = self.transaction_db() {
            let names_cf = txn_db
                .cf_handle(schema::Names::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("Names CF not found"))?;

            for item in txn_db.iterator_cf(names_cf, IteratorMode::Start) {
                if loaded >= limit {
                    break;
                }

                let (key_bytes, value_bytes) = item?;

                // Parse key (NameHash) and value (name string)
                if key_bytes.len() == 8 {
                    let mut hash_bytes = [0u8; 8];
                    hash_bytes.copy_from_slice(&key_bytes);
                    let hash = NameHash::from_bytes(hash_bytes);

                    if let Ok(value) = schema::Names::value_from_bytes(&value_bytes) {
                        self.name_cache.insert(hash, value.0);
                        loaded += 1;
                    }
                }
            }
        }

        Ok(loaded)
    }

    /// Get a reference to the underlying DB (only works in readonly or secondary mode)
    pub(crate) fn db(&self) -> Result<&DB> {
        self.db
            .as_ref()
            .and_then(|handle| handle.as_db())
            .ok_or_else(|| anyhow::anyhow!("[Storage] Not in readonly/secondary mode or not ready"))
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

    /// Check if this is a secondary instance
    pub fn is_secondary(&self) -> bool {
        matches!(self.mode, StorageMode::Secondary { .. })
    }

    /// Catch up with primary database (only works for secondary instances)
    ///
    /// This method syncs the secondary instance with the primary by:
    /// 1. Reading MANIFEST changes (SST file additions/deletions)
    /// 2. Replaying WAL entries to reconstruct memtables
    /// 3. Updating file handles
    ///
    /// # Cost
    /// Approximately 7-115ms depending on WAL size since last catch-up.
    /// Safe to call every 1-30 seconds.
    ///
    /// # Example
    /// ```no_run
    /// # use motlie_db::graph::Storage;
    /// # use std::path::PathBuf;
    /// let mut storage = Storage::secondary(
    ///     &PathBuf::from("/data/db"),
    ///     &PathBuf::from("/data/db/secondary")
    /// );
    /// storage.ready().unwrap();
    ///
    /// // Periodically catch up
    /// loop {
    ///     storage.try_catch_up_with_primary().unwrap();
    ///     std::thread::sleep(std::time::Duration::from_secs(5));
    /// }
    /// ```
    ///
    /// # Errors
    /// Returns error if:
    /// - Not in Secondary mode
    /// - Database not ready
    /// - RocksDB catch-up fails
    pub fn try_catch_up_with_primary(&self) -> Result<()> {
        match &self.mode {
            StorageMode::Secondary { .. } => match &self.db {
                Some(DatabaseHandle::Secondary(db)) => {
                    db.try_catch_up_with_primary()?;
                    Ok(())
                }
                _ => Err(anyhow::anyhow!(
                    "[Storage] Database not ready or wrong handle type"
                )),
            },
            _ => Err(anyhow::anyhow!("[Storage] Not a secondary instance")),
        }
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

    /// Get a reference to the storage.
    ///
    /// This is used internally for transaction support.
    pub fn storage(&self) -> &Arc<Storage> {
        &self.storage
    }
}

impl Clone for Graph {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(), // Arc<Storage> is already Clone
        }
    }
}

#[async_trait::async_trait]
impl Processor for Graph {
    /// Process a batch of mutations
    #[tracing::instrument(skip(self, mutations), fields(mutation_count = mutations.len()))]
    async fn process_mutations(&self, mutations: &[Mutation]) -> Result<()> {
        if mutations.is_empty() {
            return Ok(());
        }

        tracing::info!(count = mutations.len(), "[Graph] About to insert mutations");

        // Get transaction and name cache
        let txn_db = self.storage.transaction_db()?;
        let txn = txn_db.transaction();
        let name_cache = self.storage.name_cache();

        // Each mutation executes itself with cache access for name deduplication
        for mutation in mutations {
            mutation.execute_with_cache(&txn, txn_db, name_cache)?;
        }

        // Single commit for all mutations
        txn.commit()?;

        tracing::info!(
            count = mutations.len(),
            "[Graph] Successfully committed mutations"
        );
        Ok(())
    }
}

/// Implement query processor for Graph
impl reader::Processor for Graph {
    fn storage(&self) -> &Storage {
        &self.storage
    }
}
