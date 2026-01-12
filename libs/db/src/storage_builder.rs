//! Storage builder for composing modular storage subsystems.
//!
//! This module provides [`StorageBuilder`] which enables multiple storage modules
//! to share storage instances without coupling between modules:
//!
//! - **RocksDB subsystems** (`RocksdbSubsystem`): graph, vector modules share TransactionDB
//! - **Fulltext subsystems** (`FulltextSubsystem`): fulltext module for full-text search
//!
//! # Design Rationale (Task 0.4)
//!
//! - Graph module shouldn't know about vector-specific CFs
//! - Vector module needs to share the same TransactionDB
//! - Fulltext module uses Tantivy independently but initialized through same builder
//! - Each module registers its own initialization logic via subsystem traits
//! - Pre-warm isolation: each subsystem's `on_ready()` handles its own caches
//!
//! # Example
//!
//! ```ignore
//! use motlie_db::storage_builder::StorageBuilder;
//! use motlie_db::graph;
//! use motlie_db::vector;
//! use motlie_db::fulltext;
//!
//! // Build shared storage with RocksDB and Tantivy
//! let storage = StorageBuilder::new(&base_path)
//!     .with_rocksdb(Box::new(graph::Subsystem::new()))
//!     .with_rocksdb(Box::new(vector::Subsystem::new()))
//!     .with_fulltext(Box::new(fulltext::Schema::new()))
//!     .with_cache_size(512 * 1024 * 1024)  // 512MB
//!     .build()?;
//!
//! // Access the shared TransactionDB
//! let db = storage.db();
//!
//! // Access the Tantivy index
//! let index = storage.index();
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use rocksdb::{Cache, ColumnFamilyDescriptor, Options, TransactionDB, TransactionDBOptions};

use crate::fulltext::FulltextSubsystem;
use crate::rocksdb::{BlockCacheConfig, RocksdbSubsystem};
use motlie_core::telemetry::SubsystemInfo;

// ============================================================================
// StorageBuilder Configuration
// ============================================================================

/// Configuration for the shared block cache.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Total block cache size in bytes.
    /// Default: 256MB.
    pub size_bytes: usize,

    /// Default block size for column families.
    /// Default: 4KB.
    pub block_size: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            size_bytes: 256 * 1024 * 1024,  // 256MB
            block_size: 4 * 1024,            // 4KB
        }
    }
}

// ============================================================================
// StorageBuilder
// ============================================================================

/// Builder for composing modular storage subsystems into shared storage instances.
///
/// This builder collects:
/// - `RocksdbSubsystem` implementations for RocksDB (graph, vector modules)
/// - `FulltextSubsystem` implementations for Tantivy (fulltext module)
///
/// # Directory Structure
///
/// Storage uses a base path and creates subdirectories:
/// - `<base_path>/rocksdb` - RocksDB TransactionDB
/// - `<base_path>/tantivy` - Tantivy fulltext index
///
/// # Example
///
/// ```ignore
/// use motlie_db::storage_builder::StorageBuilder;
/// use motlie_db::graph;
/// use motlie_db::vector;
/// use motlie_db::fulltext;
///
/// let storage = StorageBuilder::new(&base_path)
///     .with_rocksdb(Box::new(graph::Subsystem::new()))
///     .with_rocksdb(Box::new(vector::Subsystem::new()))
///     .with_fulltext(Box::new(fulltext::Schema::new()))
///     .build()?;
/// ```
pub struct StorageBuilder {
    path: PathBuf,
    /// RocksDB subsystems
    rocksdb_subsystems: Vec<Box<dyn RocksdbSubsystem>>,
    /// Fulltext (Tantivy) subsystems
    fulltext_subsystems: Vec<Box<dyn FulltextSubsystem>>,
    cache_config: CacheConfig,
    db_options: Options,
    txn_db_options: TransactionDBOptions,
}

impl StorageBuilder {
    /// Create a new storage builder for the given base path.
    ///
    /// # Arguments
    /// * `path` - Base path for storage directories
    ///
    /// # Directory Structure
    /// - `<path>/rocksdb` - RocksDB TransactionDB (if subsystems added)
    /// - `<path>/tantivy` - Tantivy fulltext index (if fulltext subsystems added)
    pub fn new(path: &Path) -> Self {
        let mut db_options = Options::default();
        db_options.create_if_missing(true);
        db_options.create_missing_column_families(true);

        Self {
            path: path.to_path_buf(),
            rocksdb_subsystems: Vec::new(),
            fulltext_subsystems: Vec::new(),
            cache_config: CacheConfig::default(),
            db_options,
            txn_db_options: TransactionDBOptions::default(),
        }
    }

    /// Add a RocksDB subsystem.
    ///
    /// Subsystems are registered in order and their column families are collected
    /// during `build()`.
    pub fn with_rocksdb(mut self, subsystem: Box<dyn RocksdbSubsystem>) -> Self {
        self.rocksdb_subsystems.push(subsystem);
        self
    }

    /// Add a fulltext (Tantivy) subsystem.
    ///
    /// Fulltext subsystems define the Tantivy schema and initialization logic.
    /// Currently supports a single fulltext subsystem.
    pub fn with_fulltext(mut self, subsystem: Box<dyn FulltextSubsystem>) -> Self {
        self.fulltext_subsystems.push(subsystem);
        self
    }

    /// Set the block cache size in bytes.
    pub fn with_cache_size(mut self, size_bytes: usize) -> Self {
        self.cache_config.size_bytes = size_bytes;
        self
    }

    /// Set the default block size for column families.
    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.cache_config.block_size = block_size;
        self
    }

    /// Set the full cache configuration.
    pub fn with_cache_config(mut self, config: CacheConfig) -> Self {
        self.cache_config = config;
        self
    }

    /// Set custom RocksDB options.
    pub fn with_db_options(mut self, options: Options) -> Self {
        self.db_options = options;
        self
    }

    /// Set custom TransactionDB options.
    pub fn with_txn_db_options(mut self, options: TransactionDBOptions) -> Self {
        self.txn_db_options = options;
        self
    }

    /// Build the shared storage.
    ///
    /// This method:
    /// 1. Creates a shared block cache for RocksDB
    /// 2. Collects CF descriptors from all RocksDB subsystems
    /// 3. Opens the TransactionDB with all CFs
    /// 4. Calls `on_ready()` on each RocksDB subsystem
    /// 5. Creates/opens Tantivy index if fulltext subsystems are registered
    /// 6. Calls `on_ready()` on each fulltext subsystem
    ///
    /// # Errors
    /// Returns an error if:
    /// - The database cannot be opened
    /// - The Tantivy index cannot be created/opened
    /// - Any subsystem's `on_ready()` fails
    #[tracing::instrument(skip(self), fields(path = ?self.path, rocksdb_subsystems = self.rocksdb_subsystems.len(), fulltext_subsystems = self.fulltext_subsystems.len()))]
    pub fn build(self) -> Result<SharedStorage> {
        let rocksdb_path = self.path.join("rocksdb");
        let tantivy_path = self.path.join("tantivy");

        // ─────────────────────────────────────────────────────────────
        // RocksDB Initialization
        // ─────────────────────────────────────────────────────────────

        let (db_arc, cache, subsystem_map) = if !self.rocksdb_subsystems.is_empty() {
            // Create shared block cache
            let cache = Cache::new_lru_cache(self.cache_config.size_bytes);

            tracing::info!(
                size_mb = self.cache_config.size_bytes / (1024 * 1024),
                block_size_kb = self.cache_config.block_size / 1024,
                "[StorageBuilder] Created shared block cache"
            );

            // Collect CF descriptors from all subsystems
            let mut all_descriptors: Vec<ColumnFamilyDescriptor> = Vec::new();
            let mut subsystem_names: Vec<&'static str> = Vec::new();

            let block_cache_config = BlockCacheConfig {
                cache_size_bytes: self.cache_config.size_bytes,
                default_block_size: self.cache_config.block_size,
                large_block_size: self.cache_config.block_size,
                ..Default::default()
            };

            for subsystem in &self.rocksdb_subsystems {
                let name = SubsystemInfo::name(subsystem.as_ref());
                let descriptors = subsystem.cf_descriptors(&cache, &block_cache_config);

                tracing::debug!(
                    subsystem = name,
                    cf_count = descriptors.len(),
                    "[StorageBuilder] Collected CF descriptors"
                );

                all_descriptors.extend(descriptors);
                subsystem_names.push(name);
            }

            tracing::info!(
                total_cfs = all_descriptors.len(),
                subsystems = ?subsystem_names,
                path = ?rocksdb_path,
                "[StorageBuilder] Opening TransactionDB with all CFs"
            );

            // Open TransactionDB with all column families
            let db = TransactionDB::open_cf_descriptors(
                &self.db_options,
                &self.txn_db_options,
                &rocksdb_path,
                all_descriptors,
            )?;

            let db_arc = Arc::new(db);

            // Call on_ready for each subsystem
            for subsystem in &self.rocksdb_subsystems {
                let name = SubsystemInfo::name(subsystem.as_ref());
                tracing::debug!(subsystem = name, "[StorageBuilder] Calling RocksDB on_ready");
                subsystem.on_ready(&db_arc)?;
            }

            tracing::info!(
                subsystems = ?subsystem_names,
                "[StorageBuilder] All RocksDB subsystems initialized"
            );

            // Build subsystem id -> index map for lookup
            let mut subsystem_map: HashMap<&'static str, usize> = HashMap::new();
            for (i, subsystem) in self.rocksdb_subsystems.iter().enumerate() {
                subsystem_map.insert(subsystem.id(), i);
            }

            (Some(db_arc), Some(cache), subsystem_map)
        } else {
            (None, None, HashMap::new())
        };

        // ─────────────────────────────────────────────────────────────
        // Tantivy Initialization
        // ─────────────────────────────────────────────────────────────

        let (index_arc, fulltext_map) = if !self.fulltext_subsystems.is_empty() {
            // For now, we support a single fulltext subsystem (can be extended later)
            // The first subsystem's schema is used
            let first_subsystem = &self.fulltext_subsystems[0];
            let schema = first_subsystem.schema();
            let writer_heap_size = first_subsystem.writer_heap_size();

            // Create or open index
            let meta_path = tantivy_path.join("meta.json");
            let index = if meta_path.exists() {
                tantivy::Index::open_in_dir(&tantivy_path)
                    .context("Failed to open existing Tantivy index")?
            } else {
                std::fs::create_dir_all(&tantivy_path)
                    .context("Failed to create Tantivy index directory")?;
                tantivy::Index::create_in_dir(&tantivy_path, schema)
                    .context("Failed to create Tantivy index")?
            };

            tracing::info!(
                path = ?tantivy_path,
                writer_heap_size_mb = writer_heap_size / (1024 * 1024),
                "[StorageBuilder] Opened Tantivy index"
            );

            let index_arc = Arc::new(index);

            // Call on_ready for each fulltext subsystem
            for subsystem in &self.fulltext_subsystems {
                let name = SubsystemInfo::name(subsystem.as_ref());
                tracing::debug!(subsystem = name, "[StorageBuilder] Calling Tantivy on_ready");
                subsystem.on_ready(&index_arc)?;
            }

            // Build fulltext subsystem id -> index map
            let mut fulltext_map: HashMap<&'static str, usize> = HashMap::new();
            for (i, subsystem) in self.fulltext_subsystems.iter().enumerate() {
                fulltext_map.insert(subsystem.id(), i);
            }

            tracing::info!(
                subsystems = self.fulltext_subsystems.len(),
                "[StorageBuilder] All fulltext subsystems initialized"
            );

            (Some(index_arc), fulltext_map)
        } else {
            (None, HashMap::new())
        };

        Ok(SharedStorage {
            path: self.path,
            db: db_arc,
            cache,
            rocksdb_subsystems: self.rocksdb_subsystems,
            rocksdb_map: subsystem_map,
            index: index_arc,
            fulltext_subsystems: self.fulltext_subsystems,
            fulltext_map,
        })
    }
}

// ============================================================================
// SharedStorage
// ============================================================================

/// Shared storage handle providing access to RocksDB and Tantivy storage.
///
/// This is the result of [`StorageBuilder::build()`]. It provides:
/// - Access to the shared RocksDB `TransactionDB` (if RocksDB subsystems registered)
/// - Access to the shared Tantivy `Index` (if fulltext subsystems registered)
/// - Access to individual subsystems by name
/// - The shared block cache
pub struct SharedStorage {
    path: PathBuf,
    // RocksDB storage
    db: Option<Arc<TransactionDB>>,
    #[allow(dead_code)]
    cache: Option<Cache>,
    rocksdb_subsystems: Vec<Box<dyn RocksdbSubsystem>>,
    rocksdb_map: HashMap<&'static str, usize>,
    // Tantivy storage
    index: Option<Arc<tantivy::Index>>,
    fulltext_subsystems: Vec<Box<dyn FulltextSubsystem>>,
    fulltext_map: HashMap<&'static str, usize>,
}

impl SharedStorage {
    /// Get the base storage path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the RocksDB subdirectory path.
    pub fn rocksdb_path(&self) -> PathBuf {
        self.path.join("rocksdb")
    }

    /// Get the Tantivy subdirectory path.
    pub fn tantivy_path(&self) -> PathBuf {
        self.path.join("tantivy")
    }

    // ─────────────────────────────────────────────────────────────
    // RocksDB Access
    // ─────────────────────────────────────────────────────────────

    /// Get a reference to the shared TransactionDB.
    ///
    /// Returns `None` if no RocksDB components were registered.
    pub fn db(&self) -> Option<&Arc<TransactionDB>> {
        self.db.as_ref()
    }

    /// Get a clone of the TransactionDB Arc.
    ///
    /// Returns `None` if no RocksDB components were registered.
    pub fn db_clone(&self) -> Option<Arc<TransactionDB>> {
        self.db.clone()
    }

    /// Get a RocksDB component by id (e.g., "graph", "vector").
    ///
    /// Returns `None` if no component with the given id is registered.
    pub fn get_component(&self, id: &str) -> Option<&dyn RocksdbSubsystem> {
        self.rocksdb_map
            .get(id)
            .map(|&i| self.rocksdb_subsystems[i].as_ref())
    }

    /// List all registered RocksDB component ids.
    pub fn component_names(&self) -> Vec<&'static str> {
        self.rocksdb_subsystems
            .iter()
            .map(|p| p.id())
            .collect()
    }

    /// List all column family names across all RocksDB subsystems.
    pub fn all_cf_names(&self) -> Vec<&'static str> {
        let mut names: Vec<&'static str> = Vec::new();
        for subsystem in &self.rocksdb_subsystems {
            names.extend(subsystem.cf_names());
        }
        names
    }

    /// Check if RocksDB storage is available.
    pub fn has_rocksdb(&self) -> bool {
        self.db.is_some()
    }

    // ─────────────────────────────────────────────────────────────
    // Tantivy Access
    // ─────────────────────────────────────────────────────────────

    /// Get a reference to the shared Tantivy Index.
    ///
    /// Returns `None` if no index providers were registered.
    pub fn index(&self) -> Option<&Arc<tantivy::Index>> {
        self.index.as_ref()
    }

    /// Get a clone of the Tantivy Index Arc.
    ///
    /// Returns `None` if no index providers were registered.
    pub fn index_clone(&self) -> Option<Arc<tantivy::Index>> {
        self.index.clone()
    }

    /// Get a fulltext component by id (e.g., "fulltext").
    ///
    /// Returns `None` if no component with the given id is registered.
    pub fn get_fulltext(&self, id: &str) -> Option<&dyn FulltextSubsystem> {
        self.fulltext_map
            .get(id)
            .map(|&i| self.fulltext_subsystems[i].as_ref())
    }

    /// List all registered fulltext component ids.
    pub fn fulltext_names(&self) -> Vec<&'static str> {
        self.fulltext_subsystems
            .iter()
            .map(|p| p.id())
            .collect()
    }

    /// Check if Tantivy storage is available.
    pub fn has_tantivy(&self) -> bool {
        self.index.is_some()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SubsystemProvider;
    use tempfile::TempDir;

    /// Mock RocksDB subsystem for testing
    struct MockRocksdbSubsystem {
        name: &'static str,
        cf_name: &'static str,
        on_ready_called: std::sync::atomic::AtomicBool,
    }

    impl MockRocksdbSubsystem {
        fn new(name: &'static str, cf_name: &'static str) -> Self {
            Self {
                name,
                cf_name,
                on_ready_called: std::sync::atomic::AtomicBool::new(false),
            }
        }

        fn was_ready_called(&self) -> bool {
            self.on_ready_called.load(std::sync::atomic::Ordering::SeqCst)
        }
    }

    impl SubsystemInfo for MockRocksdbSubsystem {
        fn name(&self) -> &'static str {
            self.name
        }

        fn info_lines(&self) -> Vec<(&'static str, String)> {
            vec![]
        }
    }

    impl SubsystemProvider<TransactionDB> for MockRocksdbSubsystem {
        fn on_ready(&self, _db: &TransactionDB) -> Result<()> {
            self.on_ready_called
                .store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
    }

    impl RocksdbSubsystem for MockRocksdbSubsystem {
        fn id(&self) -> &'static str {
            self.name
        }

        fn cf_names(&self) -> &'static [&'static str] {
            // For testing, we return a static slice
            // In real code, this would be a constant
            match self.cf_name {
                "mock_cf" => &["mock_cf"],
                "cf1" => &["cf1"],
                "cf2" => &["cf2"],
                "test_cf" => &["test_cf"],
                "tracked_cf" => &["tracked_cf"],
                _ => &[],
            }
        }

        fn cf_descriptors(
            &self,
            _cache: &Cache,
            _config: &BlockCacheConfig,
        ) -> Vec<ColumnFamilyDescriptor> {
            vec![ColumnFamilyDescriptor::new(
                self.cf_name,
                Options::default(),
            )]
        }
    }

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::default();
        assert_eq!(config.size_bytes, 256 * 1024 * 1024);
        assert_eq!(config.block_size, 4 * 1024);
    }

    #[test]
    fn test_storage_builder_new() {
        let path = Path::new("/tmp/test_db");
        let builder = StorageBuilder::new(path);
        assert_eq!(builder.path, path);
        assert!(builder.rocksdb_subsystems.is_empty());
    }

    #[test]
    fn test_storage_builder_with_rocksdb() {
        let path = Path::new("/tmp/test_db");
        let builder = StorageBuilder::new(path)
            .with_rocksdb(Box::new(MockRocksdbSubsystem::new("test", "test_cf")));
        assert_eq!(builder.rocksdb_subsystems.len(), 1);
    }

    #[test]
    fn test_storage_builder_with_cache_config() {
        let path = Path::new("/tmp/test_db");
        let builder = StorageBuilder::new(path)
            .with_cache_size(512 * 1024 * 1024)
            .with_block_size(8 * 1024);

        assert_eq!(builder.cache_config.size_bytes, 512 * 1024 * 1024);
        assert_eq!(builder.cache_config.block_size, 8 * 1024);
    }

    #[test]
    fn test_storage_builder_build_single_subsystem() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let storage = StorageBuilder::new(&db_path)
            .with_rocksdb(Box::new(MockRocksdbSubsystem::new("mock", "mock_cf")))
            .build()
            .expect("Failed to build storage");

        assert_eq!(storage.path(), db_path);
        assert!(storage.get_component("mock").is_some());
        assert!(storage.get_component("nonexistent").is_none());
        assert_eq!(storage.component_names(), vec!["mock"]);
        assert_eq!(storage.all_cf_names(), vec!["mock_cf"]);
    }

    #[test]
    fn test_storage_builder_build_multiple_subsystems() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let storage = StorageBuilder::new(&db_path)
            .with_rocksdb(Box::new(MockRocksdbSubsystem::new("subsystem1", "cf1")))
            .with_rocksdb(Box::new(MockRocksdbSubsystem::new("subsystem2", "cf2")))
            .build()
            .expect("Failed to build storage");

        assert!(storage.get_component("subsystem1").is_some());
        assert!(storage.get_component("subsystem2").is_some());
        assert_eq!(storage.component_names().len(), 2);
        assert_eq!(storage.all_cf_names().len(), 2);
    }

    #[test]
    fn test_storage_builder_calls_on_ready() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Create subsystem with ready tracking
        let subsystem = std::sync::Arc::new(MockRocksdbSubsystem::new("tracked", "tracked_cf"));

        // We need to use Arc to track the state
        struct TrackedSubsystem {
            inner: std::sync::Arc<MockRocksdbSubsystem>,
        }

        impl SubsystemInfo for TrackedSubsystem {
            fn name(&self) -> &'static str {
                SubsystemInfo::name(self.inner.as_ref())
            }

            fn info_lines(&self) -> Vec<(&'static str, String)> {
                vec![]
            }
        }

        impl SubsystemProvider<TransactionDB> for TrackedSubsystem {
            fn on_ready(&self, db: &TransactionDB) -> Result<()> {
                SubsystemProvider::on_ready(self.inner.as_ref(), db)
            }
        }

        impl RocksdbSubsystem for TrackedSubsystem {
            fn id(&self) -> &'static str {
                self.inner.id()
            }

            fn cf_names(&self) -> &'static [&'static str] {
                self.inner.cf_names()
            }

            fn cf_descriptors(
                &self,
                cache: &Cache,
                config: &BlockCacheConfig,
            ) -> Vec<ColumnFamilyDescriptor> {
                self.inner.cf_descriptors(cache, config)
            }
        }

        let tracked = TrackedSubsystem {
            inner: subsystem.clone(),
        };

        let _storage = StorageBuilder::new(&db_path)
            .with_rocksdb(Box::new(tracked))
            .build()
            .expect("Failed to build storage");

        assert!(subsystem.was_ready_called());
    }

    #[test]
    fn test_shared_storage_db_access() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let storage = StorageBuilder::new(&db_path)
            .with_rocksdb(Box::new(MockRocksdbSubsystem::new("test", "test_cf")))
            .build()
            .expect("Failed to build storage");

        // Should be able to access the DB
        assert!(storage.has_rocksdb());
        let db = storage.db().expect("Should have DB");
        assert!(db.cf_handle("test_cf").is_some());

        // Clone should work
        let db_clone = storage.db_clone().expect("Should have DB");
        assert!(db_clone.cf_handle("test_cf").is_some());
    }

    #[test]
    fn test_storage_builder_with_fulltext() {
        let path = Path::new("/tmp/test_db");
        let builder = StorageBuilder::new(path)
            .with_fulltext(Box::new(MockFulltextSubsystem::new()));
        assert_eq!(builder.fulltext_subsystems.len(), 1);
    }

    #[test]
    fn test_storage_with_fulltext() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let storage = StorageBuilder::new(&db_path)
            .with_fulltext(Box::new(MockFulltextSubsystem::new()))
            .build()
            .expect("Failed to build storage");

        // Should have Tantivy but no RocksDB
        assert!(storage.has_tantivy());
        assert!(!storage.has_rocksdb());
        assert!(storage.index().is_some());
        assert!(storage.db().is_none());
        assert_eq!(storage.fulltext_names(), vec!["mock_fulltext"]);
    }

    #[test]
    fn test_storage_with_both_subsystems() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let storage = StorageBuilder::new(&db_path)
            .with_rocksdb(Box::new(MockRocksdbSubsystem::new("test", "test_cf")))
            .with_fulltext(Box::new(MockFulltextSubsystem::new()))
            .build()
            .expect("Failed to build storage");

        // Should have both
        assert!(storage.has_rocksdb());
        assert!(storage.has_tantivy());
        assert!(storage.db().is_some());
        assert!(storage.index().is_some());
        assert_eq!(storage.component_names(), vec!["test"]);
        assert_eq!(storage.fulltext_names(), vec!["mock_fulltext"]);
    }

    #[test]
    fn test_storage_paths() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let storage = StorageBuilder::new(&db_path)
            .with_rocksdb(Box::new(MockRocksdbSubsystem::new("test", "test_cf")))
            .build()
            .expect("Failed to build storage");

        assert_eq!(storage.path(), db_path);
        assert_eq!(storage.rocksdb_path(), db_path.join("rocksdb"));
        assert_eq!(storage.tantivy_path(), db_path.join("tantivy"));
    }

    /// Mock fulltext subsystem for testing
    struct MockFulltextSubsystem {
        on_ready_called: std::sync::atomic::AtomicBool,
    }

    impl MockFulltextSubsystem {
        fn new() -> Self {
            Self {
                on_ready_called: std::sync::atomic::AtomicBool::new(false),
            }
        }
    }

    impl SubsystemInfo for MockFulltextSubsystem {
        fn name(&self) -> &'static str {
            "mock_fulltext"
        }

        fn info_lines(&self) -> Vec<(&'static str, String)> {
            vec![]
        }
    }

    impl SubsystemProvider<tantivy::Index> for MockFulltextSubsystem {
        fn on_ready(&self, _index: &tantivy::Index) -> Result<()> {
            self.on_ready_called
                .store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
    }

    impl FulltextSubsystem for MockFulltextSubsystem {
        fn id(&self) -> &'static str {
            "mock_fulltext"
        }

        fn schema(&self) -> tantivy::schema::Schema {
            let mut builder = tantivy::schema::Schema::builder();
            builder.add_text_field("content", tantivy::schema::TEXT);
            builder.build()
        }
    }
}
