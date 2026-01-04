//! Generic RocksDB storage parameterized by subsystem.
//!
//! Provides `Storage<S>` that eliminates boilerplate across subsystems while
//! allowing subsystem-specific column families, caches, and initialization.

use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use rocksdb::{Options, TransactionDB, TransactionDBOptions, DB};

use super::config::BlockCacheConfig;
use super::handle::{DatabaseHandle, StorageMode, StorageOptions};
use super::subsystem::{DbAccess, StorageSubsystem};

// ============================================================================
// Storage<S>
// ============================================================================

/// Generic RocksDB storage parameterized by subsystem.
///
/// This type provides the common storage infrastructure (database opening,
/// mode handling, block cache) while delegating subsystem-specific behavior
/// (column families, caches, pre-warming) to the `StorageSubsystem` trait.
///
/// # Type Parameter
///
/// - `S`: A type implementing `StorageSubsystem` that defines the subsystem's
///   column families, cache type, and initialization logic.
///
/// # Example
///
/// ```ignore
/// // Define type alias for ergonomics
/// pub type Storage = rocksdb::Storage<GraphSubsystem>;
///
/// // Use standalone storage
/// let mut storage = graph::Storage::readonly(path);
/// storage.ready()?;
/// let cache = storage.cache();
/// ```
pub struct Storage<S: StorageSubsystem> {
    db_path: PathBuf,
    db_options: Options,
    txn_db_options: TransactionDBOptions,
    db: Option<DatabaseHandle>,
    mode: StorageMode,
    block_cache: Option<rocksdb::Cache>,
    block_cache_config: BlockCacheConfig,
    // Subsystem-specific
    cache: Arc<S::Cache>,
    prewarm_config: S::PrewarmConfig,
    _marker: PhantomData<S>,
}

impl<S: StorageSubsystem> Storage<S> {
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Create a new Storage instance in read-only mode.
    ///
    /// Multiple read-only instances can access the same database simultaneously.
    /// Use this for query consumers.
    pub fn readonly(db_path: &Path) -> Self {
        Self {
            db_path: PathBuf::from(db_path),
            db_options: StorageOptions::default_for_readonly(),
            txn_db_options: TransactionDBOptions::default(),
            db: None,
            mode: StorageMode::ReadOnly,
            block_cache: None,
            block_cache_config: BlockCacheConfig::default(),
            cache: S::create_cache(),
            prewarm_config: S::PrewarmConfig::default(),
            _marker: PhantomData,
        }
    }

    /// Create a new Storage instance in read-write mode.
    ///
    /// Only one read-write instance can access the database at a time.
    /// Use this for mutation consumers.
    pub fn readwrite(db_path: &Path) -> Self {
        Self {
            db_path: PathBuf::from(db_path),
            db_options: StorageOptions::default_for_readwrite(),
            txn_db_options: TransactionDBOptions::default(),
            db: None,
            mode: StorageMode::ReadWrite,
            block_cache: None,
            block_cache_config: BlockCacheConfig::default(),
            cache: S::create_cache(),
            prewarm_config: S::PrewarmConfig::default(),
            _marker: PhantomData,
        }
    }

    /// Create a new Storage instance with custom options.
    pub fn readwrite_with_options(
        db_path: &Path,
        db_options: Options,
        txn_db_options: TransactionDBOptions,
    ) -> Self {
        Self {
            db_path: PathBuf::from(db_path),
            db_options,
            txn_db_options,
            db: None,
            mode: StorageMode::ReadWrite,
            block_cache: None,
            block_cache_config: BlockCacheConfig::default(),
            cache: S::create_cache(),
            prewarm_config: S::PrewarmConfig::default(),
            _marker: PhantomData,
        }
    }

    /// Create a new Storage instance in secondary mode.
    ///
    /// Secondary instances can catch up with the primary database via
    /// `try_catch_up_with_primary()`, ideal for read replicas.
    ///
    /// # Arguments
    /// * `primary_path` - Path to the primary database
    /// * `secondary_path` - Path for secondary's MANIFEST (must differ from primary)
    pub fn secondary(primary_path: &Path, secondary_path: &Path) -> Self {
        Self {
            db_path: PathBuf::from(primary_path),
            db_options: StorageOptions::default_for_secondary(),
            txn_db_options: TransactionDBOptions::default(),
            db: None,
            mode: StorageMode::Secondary {
                secondary_path: PathBuf::from(secondary_path),
            },
            block_cache: None,
            block_cache_config: BlockCacheConfig::default(),
            cache: S::create_cache(),
            prewarm_config: S::PrewarmConfig::default(),
            _marker: PhantomData,
        }
    }

    // =========================================================================
    // Builder Methods
    // =========================================================================

    /// Set the pre-warm configuration.
    ///
    /// Must be called before `ready()` to take effect.
    pub fn with_prewarm_config(mut self, config: S::PrewarmConfig) -> Self {
        self.prewarm_config = config;
        self
    }

    /// Set the block cache configuration.
    ///
    /// Must be called before `ready()` to take effect.
    pub fn with_block_cache_config(mut self, config: BlockCacheConfig) -> Self {
        self.block_cache_config = config;
        self
    }

    // =========================================================================
    // Initialization
    // =========================================================================

    /// Initialize the database and pre-warm the cache.
    ///
    /// This method:
    /// 1. Validates the database path
    /// 2. Creates the shared block cache
    /// 3. Gets column family descriptors from the subsystem
    /// 4. Opens the database in the configured mode
    /// 5. Pre-warms the subsystem cache
    #[tracing::instrument(skip(self), fields(subsystem = S::NAME, path = ?self.db_path))]
    pub fn ready(&mut self) -> Result<()> {
        if self.db.is_some() {
            return Ok(());
        }

        // Validate path
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

        // Create shared block cache
        let cache = rocksdb::Cache::new_lru_cache(self.block_cache_config.cache_size_bytes);
        self.block_cache = Some(cache);
        let cache_ref = self.block_cache.as_ref().unwrap();

        tracing::info!(
            subsystem = S::NAME,
            cache_mb = self.block_cache_config.cache_size_bytes / (1024 * 1024),
            "[{}] Created block cache",
            S::NAME
        );

        // Get CF descriptors from subsystem
        let cf_descriptors = S::cf_descriptors(cache_ref, &self.block_cache_config);

        tracing::debug!(
            subsystem = S::NAME,
            cf_count = cf_descriptors.len(),
            "[{}] Built CF descriptors",
            S::NAME
        );

        // Open database based on mode
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

        // Pre-warm the cache
        let db_access: &dyn DbAccess = match &self.db {
            Some(DatabaseHandle::ReadOnly(db)) => db,
            Some(DatabaseHandle::ReadWrite(txn_db)) => txn_db,
            Some(DatabaseHandle::Secondary(db)) => db,
            None => unreachable!(),
        };

        let loaded = S::prewarm(db_access, &self.cache, &self.prewarm_config)?;
        tracing::info!(
            subsystem = S::NAME,
            loaded,
            "[{}] Pre-warmed cache",
            S::NAME
        );

        tracing::info!(subsystem = S::NAME, "[{}] Ready", S::NAME);
        Ok(())
    }

    // =========================================================================
    // Database Access
    // =========================================================================

    /// Get a reference to the underlying DB (only in readonly/secondary mode).
    pub(crate) fn db(&self) -> Result<&DB> {
        self.db
            .as_ref()
            .and_then(|h| h.as_db())
            .ok_or_else(|| {
                anyhow::anyhow!("[{}] Not in readonly/secondary mode or not ready", S::NAME)
            })
    }

    /// Get a reference to the TransactionDB (only in readwrite mode).
    pub fn transaction_db(&self) -> Result<&TransactionDB> {
        self.db
            .as_ref()
            .and_then(|h| h.as_transaction_db())
            .ok_or_else(|| anyhow::anyhow!("[{}] Not in readwrite mode or not ready", S::NAME))
    }

    /// Check if storage is in readwrite mode with TransactionDB.
    pub fn is_transactional(&self) -> bool {
        self.db
            .as_ref()
            .map(|h| h.is_read_write())
            .unwrap_or(false)
    }

    /// Check if this is a secondary instance.
    pub fn is_secondary(&self) -> bool {
        matches!(self.mode, StorageMode::Secondary { .. })
    }

    /// Catch up with primary database (only for secondary instances).
    ///
    /// Syncs the secondary with the primary by reading MANIFEST and WAL changes.
    pub fn try_catch_up_with_primary(&self) -> Result<()> {
        match &self.db {
            Some(DatabaseHandle::Secondary(db)) => {
                db.try_catch_up_with_primary()?;
                Ok(())
            }
            _ => Err(anyhow::anyhow!(
                "[{}] try_catch_up_with_primary only works for secondary instances",
                S::NAME
            )),
        }
    }

    /// Get the database path.
    pub fn path(&self) -> &Path {
        &self.db_path
    }

    /// Get list of column family names for this subsystem.
    pub fn column_families(&self) -> &'static [&'static str] {
        S::COLUMN_FAMILIES
    }

    /// Close the database.
    pub fn close(&mut self) -> Result<()> {
        if self.db.is_none() {
            return Err(anyhow::anyhow!("[{}] Storage is not ready", S::NAME));
        }
        if let Some(db_handle) = self.db.take() {
            drop(db_handle);
        }
        Ok(())
    }

    // =========================================================================
    // Subsystem Cache Access
    // =========================================================================

    /// Get a reference to the subsystem's in-memory cache.
    pub fn cache(&self) -> &Arc<S::Cache> {
        &self.cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock subsystem for testing
    struct MockSubsystem;

    #[derive(Default, Clone)]
    struct MockPrewarmConfig;

    struct MockCache;

    impl StorageSubsystem for MockSubsystem {
        const NAME: &'static str = "mock";
        const COLUMN_FAMILIES: &'static [&'static str] = &["mock/data"];

        type PrewarmConfig = MockPrewarmConfig;
        type Cache = MockCache;

        fn create_cache() -> Arc<Self::Cache> {
            Arc::new(MockCache)
        }

        fn cf_descriptors(
            _block_cache: &rocksdb::Cache,
            _config: &BlockCacheConfig,
        ) -> Vec<rocksdb::ColumnFamilyDescriptor> {
            vec![rocksdb::ColumnFamilyDescriptor::new(
                "mock/data",
                rocksdb::Options::default(),
            )]
        }

        fn prewarm(
            _db: &dyn DbAccess,
            _cache: &Self::Cache,
            _config: &Self::PrewarmConfig,
        ) -> Result<usize> {
            Ok(0)
        }
    }

    type MockStorage = Storage<MockSubsystem>;

    #[test]
    fn test_storage_readonly_create() {
        let storage = MockStorage::readonly(Path::new("/tmp/test"));
        assert!(!storage.is_transactional());
        assert!(!storage.is_secondary());
    }

    #[test]
    fn test_storage_readwrite_create() {
        let storage = MockStorage::readwrite(Path::new("/tmp/test"));
        assert!(!storage.is_transactional()); // Not ready yet
        assert!(!storage.is_secondary());
    }

    #[test]
    fn test_storage_secondary_create() {
        let storage =
            MockStorage::secondary(Path::new("/tmp/primary"), Path::new("/tmp/secondary"));
        assert!(!storage.is_transactional());
        assert!(storage.is_secondary());
    }

    #[test]
    fn test_storage_column_families() {
        let storage = MockStorage::readonly(Path::new("/tmp/test"));
        assert_eq!(storage.column_families(), &["mock/data"]);
    }

    #[test]
    fn test_storage_readwrite_ready() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("mock_db");

        let mut storage = MockStorage::readwrite(&db_path);
        storage.ready().expect("Failed to initialize storage");

        assert!(storage.is_transactional());
        assert!(storage.transaction_db().is_ok());
    }
}
