//! Storage builder for composing modular ColumnFamilyProvider implementations.
//!
//! This module provides [`StorageBuilder`] which enables multiple storage modules
//! (graph, vector, etc.) to share a single RocksDB TransactionDB instance without
//! coupling between modules.
//!
//! # Design Rationale (Task 0.4)
//!
//! - Graph module shouldn't know about vector-specific CFs
//! - Vector module needs to share the same TransactionDB
//! - Each module registers its own CFs and initialization logic via `ColumnFamilyProvider`
//! - Pre-warm isolation: each module's `on_ready()` handles its own caches
//!
//! # Example
//!
//! ```ignore
//! use motlie_db::storage_builder::StorageBuilder;
//! use motlie_db::graph;
//! use motlie_db::vector;
//!
//! // Create providers
//! let graph_schema = graph::Schema::new();
//! let vector_schema = vector::Schema::new();
//!
//! // Build shared storage
//! let storage = StorageBuilder::new(&db_path)
//!     .with_provider(Box::new(graph_schema))
//!     .with_provider(Box::new(vector_schema))
//!     .with_cache_size(512 * 1024 * 1024)  // 512MB
//!     .build()?;
//!
//! // Access the shared TransactionDB
//! let db = storage.db();
//!
//! // Get a specific provider's registry/cache
//! let vector_schema = storage.get_provider::<vector::Schema>("vector");
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use rocksdb::{Cache, ColumnFamilyDescriptor, Options, TransactionDB, TransactionDBOptions};

use crate::provider::ColumnFamilyProvider;

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

/// Builder for composing modular storage providers into a shared RocksDB instance.
///
/// This builder collects `ColumnFamilyProvider` implementations from multiple modules
/// and opens a single `TransactionDB` with all their column families.
///
/// # Example
///
/// ```ignore
/// use motlie_db::storage_builder::StorageBuilder;
/// use motlie_db::graph;
/// use motlie_db::vector;
///
/// let storage = StorageBuilder::new(&db_path)
///     .with_provider(Box::new(graph::Schema::new()))
///     .with_provider(Box::new(vector::Schema::new()))
///     .build()?;
/// ```
pub struct StorageBuilder {
    path: PathBuf,
    providers: Vec<Box<dyn ColumnFamilyProvider>>,
    cache_config: CacheConfig,
    db_options: Options,
    txn_db_options: TransactionDBOptions,
}

impl StorageBuilder {
    /// Create a new storage builder for the given path.
    ///
    /// # Arguments
    /// * `path` - Path to the RocksDB database directory
    pub fn new(path: &Path) -> Self {
        let mut db_options = Options::default();
        db_options.create_if_missing(true);
        db_options.create_missing_column_families(true);

        Self {
            path: path.to_path_buf(),
            providers: Vec::new(),
            cache_config: CacheConfig::default(),
            db_options,
            txn_db_options: TransactionDBOptions::default(),
        }
    }

    /// Add a column family provider.
    ///
    /// Providers are registered in order and their column families are collected
    /// during `build()`.
    pub fn with_provider(mut self, provider: Box<dyn ColumnFamilyProvider>) -> Self {
        self.providers.push(provider);
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
    /// 1. Creates a shared block cache
    /// 2. Collects CF descriptors from all providers
    /// 3. Opens the TransactionDB with all CFs
    /// 4. Calls `on_ready()` on each provider for pre-warming
    ///
    /// # Errors
    /// Returns an error if:
    /// - The database cannot be opened
    /// - Any provider's `on_ready()` fails
    #[tracing::instrument(skip(self), fields(path = ?self.path, providers = self.providers.len()))]
    pub fn build(self) -> Result<SharedStorage> {
        // Create shared block cache
        let cache = Cache::new_lru_cache(self.cache_config.size_bytes);

        tracing::info!(
            size_mb = self.cache_config.size_bytes / (1024 * 1024),
            block_size_kb = self.cache_config.block_size / 1024,
            "[StorageBuilder] Created shared block cache"
        );

        // Collect CF descriptors from all providers
        let mut all_descriptors: Vec<ColumnFamilyDescriptor> = Vec::new();
        let mut provider_names: Vec<&'static str> = Vec::new();

        for provider in &self.providers {
            let name = provider.name();
            let descriptors =
                provider.column_family_descriptors(Some(&cache), self.cache_config.block_size);

            tracing::debug!(
                provider = name,
                cf_count = descriptors.len(),
                "[StorageBuilder] Collected CF descriptors"
            );

            all_descriptors.extend(descriptors);
            provider_names.push(name);
        }

        tracing::info!(
            total_cfs = all_descriptors.len(),
            providers = ?provider_names,
            "[StorageBuilder] Opening TransactionDB with all CFs"
        );

        // Open TransactionDB with all column families
        let db = TransactionDB::open_cf_descriptors(
            &self.db_options,
            &self.txn_db_options,
            &self.path,
            all_descriptors,
        )?;

        let db_arc = Arc::new(db);

        // Call on_ready for each provider
        for provider in &self.providers {
            let name = provider.name();
            tracing::debug!(provider = name, "[StorageBuilder] Calling on_ready");
            provider.on_ready(&db_arc)?;
        }

        tracing::info!(
            providers = ?provider_names,
            "[StorageBuilder] All providers initialized"
        );

        // Build provider name -> index map for lookup
        let mut provider_map: HashMap<&'static str, usize> = HashMap::new();
        for (i, provider) in self.providers.iter().enumerate() {
            provider_map.insert(provider.name(), i);
        }

        Ok(SharedStorage {
            path: self.path,
            db: db_arc,
            cache,
            providers: self.providers,
            provider_map,
        })
    }
}

// ============================================================================
// SharedStorage
// ============================================================================

/// Shared storage handle providing access to the TransactionDB and providers.
///
/// This is the result of [`StorageBuilder::build()`]. It provides:
/// - Access to the shared `TransactionDB`
/// - Access to individual providers by name
/// - The shared block cache
pub struct SharedStorage {
    path: PathBuf,
    db: Arc<TransactionDB>,
    #[allow(dead_code)]
    cache: Cache,
    providers: Vec<Box<dyn ColumnFamilyProvider>>,
    provider_map: HashMap<&'static str, usize>,
}

impl SharedStorage {
    /// Get the database path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get a reference to the shared TransactionDB.
    ///
    /// Use this to perform operations on the database directly.
    pub fn db(&self) -> &Arc<TransactionDB> {
        &self.db
    }

    /// Get a clone of the TransactionDB Arc.
    ///
    /// Useful for sharing the database reference across threads.
    pub fn db_clone(&self) -> Arc<TransactionDB> {
        self.db.clone()
    }

    /// Get a provider by name.
    ///
    /// Returns `None` if no provider with the given name is registered.
    pub fn get_provider(&self, name: &str) -> Option<&dyn ColumnFamilyProvider> {
        self.provider_map
            .get(name)
            .map(|&i| self.providers[i].as_ref())
    }

    /// List all registered provider names.
    pub fn provider_names(&self) -> Vec<&'static str> {
        self.providers.iter().map(|p| p.name()).collect()
    }

    /// List all column family names across all providers.
    pub fn all_cf_names(&self) -> Vec<&'static str> {
        let mut names: Vec<&'static str> = Vec::new();
        for provider in &self.providers {
            names.extend(provider.cf_names());
        }
        names
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Mock provider for testing
    struct MockProvider {
        name: &'static str,
        cf_name: &'static str,
        on_ready_called: std::sync::atomic::AtomicBool,
    }

    impl MockProvider {
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

    impl ColumnFamilyProvider for MockProvider {
        fn name(&self) -> &'static str {
            self.name
        }

        fn column_family_descriptors(
            &self,
            _cache: Option<&Cache>,
            _block_size: usize,
        ) -> Vec<ColumnFamilyDescriptor> {
            vec![ColumnFamilyDescriptor::new(
                self.cf_name,
                Options::default(),
            )]
        }

        fn on_ready(&self, _db: &TransactionDB) -> Result<()> {
            self.on_ready_called
                .store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }

        fn cf_names(&self) -> Vec<&'static str> {
            vec![self.cf_name]
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
        assert!(builder.providers.is_empty());
    }

    #[test]
    fn test_storage_builder_with_provider() {
        let path = Path::new("/tmp/test_db");
        let builder = StorageBuilder::new(path)
            .with_provider(Box::new(MockProvider::new("test", "test_cf")));
        assert_eq!(builder.providers.len(), 1);
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
    fn test_storage_builder_build_single_provider() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let storage = StorageBuilder::new(&db_path)
            .with_provider(Box::new(MockProvider::new("mock", "mock_cf")))
            .build()
            .expect("Failed to build storage");

        assert_eq!(storage.path(), db_path);
        assert!(storage.get_provider("mock").is_some());
        assert!(storage.get_provider("nonexistent").is_none());
        assert_eq!(storage.provider_names(), vec!["mock"]);
        assert_eq!(storage.all_cf_names(), vec!["mock_cf"]);
    }

    #[test]
    fn test_storage_builder_build_multiple_providers() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let storage = StorageBuilder::new(&db_path)
            .with_provider(Box::new(MockProvider::new("provider1", "cf1")))
            .with_provider(Box::new(MockProvider::new("provider2", "cf2")))
            .build()
            .expect("Failed to build storage");

        assert!(storage.get_provider("provider1").is_some());
        assert!(storage.get_provider("provider2").is_some());
        assert_eq!(storage.provider_names().len(), 2);
        assert_eq!(storage.all_cf_names().len(), 2);
    }

    #[test]
    fn test_storage_builder_calls_on_ready() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        // Create provider with ready tracking
        let provider = std::sync::Arc::new(MockProvider::new("tracked", "tracked_cf"));

        // We need to use Arc to track the state
        struct TrackedProvider {
            inner: std::sync::Arc<MockProvider>,
        }

        impl ColumnFamilyProvider for TrackedProvider {
            fn name(&self) -> &'static str {
                self.inner.name()
            }

            fn column_family_descriptors(
                &self,
                cache: Option<&Cache>,
                block_size: usize,
            ) -> Vec<ColumnFamilyDescriptor> {
                self.inner.column_family_descriptors(cache, block_size)
            }

            fn on_ready(&self, db: &TransactionDB) -> Result<()> {
                self.inner.on_ready(db)
            }

            fn cf_names(&self) -> Vec<&'static str> {
                self.inner.cf_names()
            }
        }

        let tracked = TrackedProvider {
            inner: provider.clone(),
        };

        let _storage = StorageBuilder::new(&db_path)
            .with_provider(Box::new(tracked))
            .build()
            .expect("Failed to build storage");

        assert!(provider.was_ready_called());
    }

    #[test]
    fn test_shared_storage_db_access() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test_db");

        let storage = StorageBuilder::new(&db_path)
            .with_provider(Box::new(MockProvider::new("test", "test_cf")))
            .build()
            .expect("Failed to build storage");

        // Should be able to access the DB
        let db = storage.db();
        assert!(db.cf_handle("test_cf").is_some());

        // Clone should work
        let db_clone = storage.db_clone();
        assert!(db_clone.cf_handle("test_cf").is_some());
    }
}
