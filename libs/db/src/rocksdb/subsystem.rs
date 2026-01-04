//! StorageSubsystem trait and ComponentWrapper for modular storage.
//!
//! This module defines the core abstraction for storage subsystems:
//! - `StorageSubsystem`: Trait defining column families, caches, and initialization
//! - `ComponentWrapper`: Adapter that implements `ColumnFamilyProvider` for any subsystem

use std::sync::Arc;

use anyhow::Result;
use rocksdb::{Cache, ColumnFamilyDescriptor, TransactionDB, DB};

use super::config::BlockCacheConfig;
use crate::provider::ColumnFamilyProvider;

// ============================================================================
// DbAccess Trait
// ============================================================================

/// Abstraction over DB and TransactionDB for read operations.
///
/// This trait allows subsystem code to work with both read-only and
/// read-write storage without knowing the concrete type.
pub trait DbAccess: Send + Sync {
    /// Get a value by key from a column family.
    fn get_cf(&self, cf_name: &str, key: &[u8]) -> Result<Option<Vec<u8>>>;

    /// Get a column family handle by name.
    fn cf_handle(&self, name: &str) -> Option<&rocksdb::ColumnFamily>;

    /// Create an iterator over a column family.
    fn iterator_cf(
        &self,
        cf_name: &str,
    ) -> Result<Box<dyn Iterator<Item = Result<(Box<[u8]>, Box<[u8]>)>> + '_>>;
}

impl DbAccess for DB {
    fn get_cf(&self, cf_name: &str, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let cf = DB::cf_handle(self, cf_name)
            .ok_or_else(|| anyhow::anyhow!("Column family not found: {}", cf_name))?;
        Ok(DB::get_cf(self, cf, key)?)
    }

    fn cf_handle(&self, name: &str) -> Option<&rocksdb::ColumnFamily> {
        DB::cf_handle(self, name)
    }

    fn iterator_cf(
        &self,
        cf_name: &str,
    ) -> Result<Box<dyn Iterator<Item = Result<(Box<[u8]>, Box<[u8]>)>> + '_>> {
        let cf = DB::cf_handle(self, cf_name)
            .ok_or_else(|| anyhow::anyhow!("Column family not found: {}", cf_name))?;
        Ok(Box::new(
            DB::iterator_cf(self, cf, rocksdb::IteratorMode::Start)
                .map(|item| item.map_err(|e| anyhow::anyhow!("Iterator error: {}", e))),
        ))
    }
}

impl DbAccess for TransactionDB {
    fn get_cf(&self, cf_name: &str, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let cf = <TransactionDB>::cf_handle(self, cf_name)
            .ok_or_else(|| anyhow::anyhow!("Column family not found: {}", cf_name))?;
        Ok(<TransactionDB>::get_cf(self, cf, key)?)
    }

    fn cf_handle(&self, name: &str) -> Option<&rocksdb::ColumnFamily> {
        <TransactionDB>::cf_handle(self, name)
    }

    fn iterator_cf(
        &self,
        cf_name: &str,
    ) -> Result<Box<dyn Iterator<Item = Result<(Box<[u8]>, Box<[u8]>)>> + '_>> {
        let cf = <TransactionDB>::cf_handle(self, cf_name)
            .ok_or_else(|| anyhow::anyhow!("Column family not found: {}", cf_name))?;
        Ok(Box::new(
            <TransactionDB>::iterator_cf(self, cf, rocksdb::IteratorMode::Start)
                .map(|item| item.map_err(|e| anyhow::anyhow!("Iterator error: {}", e))),
        ))
    }
}

// ============================================================================
// StorageSubsystem Trait
// ============================================================================

/// Trait for RocksDB storage subsystems (graph, vector, etc.).
///
/// Implement this trait to define a storage subsystem with:
/// - Column family definitions and options
/// - In-memory cache type (e.g., NameCache, EmbeddingRegistry)
/// - Pre-warm configuration and logic
///
/// # Example
///
/// ```ignore
/// pub struct GraphSubsystem;
///
/// impl StorageSubsystem for GraphSubsystem {
///     const NAME: &'static str = "graph";
///     const COLUMN_FAMILIES: &'static [&'static str] = &["graph/nodes", "graph/edges"];
///
///     type PrewarmConfig = NameCacheConfig;
///     type Cache = NameCache;
///
///     fn create_cache() -> Arc<Self::Cache> {
///         Arc::new(NameCache::new())
///     }
///
///     fn cf_descriptors(cache: &Cache, config: &BlockCacheConfig) -> Vec<ColumnFamilyDescriptor> {
///         // Return CF descriptors with cache configuration
///     }
///
///     fn prewarm(db: &dyn DbAccess, cache: &Self::Cache, config: &Self::PrewarmConfig) -> Result<usize> {
///         // Pre-warm cache from database
///     }
/// }
/// ```
pub trait StorageSubsystem: Send + Sync + 'static {
    /// Subsystem name for logging and identification (e.g., "graph", "vector").
    const NAME: &'static str;

    /// List of column family names managed by this subsystem.
    const COLUMN_FAMILIES: &'static [&'static str];

    /// Pre-warm configuration type for this subsystem.
    type PrewarmConfig: Default + Clone + Send + Sync;

    /// In-memory cache type (e.g., NameCache, EmbeddingRegistry).
    type Cache: Send + Sync;

    /// Create a new cache instance.
    fn create_cache() -> Arc<Self::Cache>;

    /// Build column family descriptors with shared block cache.
    ///
    /// Called during database initialization to collect all CF descriptors.
    fn cf_descriptors(
        block_cache: &Cache,
        config: &BlockCacheConfig,
    ) -> Vec<ColumnFamilyDescriptor>;

    /// Pre-warm the cache from database contents.
    ///
    /// Called after database is opened to populate the in-memory cache.
    /// Returns the number of entries loaded.
    fn prewarm(
        db: &dyn DbAccess,
        cache: &Self::Cache,
        config: &Self::PrewarmConfig,
    ) -> Result<usize>;
}

// ============================================================================
// ComponentWrapper
// ============================================================================

/// Wrapper that adapts a `StorageSubsystem` to implement `ColumnFamilyProvider`.
///
/// This enables any subsystem to be used with `StorageBuilder`:
///
/// ```ignore
/// let shared = StorageBuilder::new(path)
///     .with_component(Box::new(graph::Component::new()))
///     .with_component(Box::new(vector::Component::new()))
///     .build()?;
/// ```
pub struct ComponentWrapper<S: StorageSubsystem> {
    cache: Arc<S::Cache>,
    prewarm_config: S::PrewarmConfig,
}

impl<S: StorageSubsystem> ComponentWrapper<S> {
    /// Create a new component wrapper with default configuration.
    pub fn new() -> Self {
        Self {
            cache: S::create_cache(),
            prewarm_config: S::PrewarmConfig::default(),
        }
    }

    /// Set the pre-warm configuration.
    pub fn with_prewarm_config(mut self, config: S::PrewarmConfig) -> Self {
        self.prewarm_config = config;
        self
    }

    /// Get a reference to the subsystem's cache.
    pub fn cache(&self) -> &Arc<S::Cache> {
        &self.cache
    }

    /// Get the pre-warm configuration.
    pub fn prewarm_config(&self) -> &S::PrewarmConfig {
        &self.prewarm_config
    }
}

impl<S: StorageSubsystem> Default for ComponentWrapper<S> {
    fn default() -> Self {
        Self::new()
    }
}

// Implement ColumnFamilyProvider for ComponentWrapper<S>
impl<S: StorageSubsystem> ColumnFamilyProvider for ComponentWrapper<S> {
    fn name(&self) -> &'static str {
        S::NAME
    }

    fn column_family_descriptors(
        &self,
        cache: Option<&Cache>,
        block_size: usize,
    ) -> Vec<ColumnFamilyDescriptor> {
        let config = BlockCacheConfig {
            default_block_size: block_size,
            large_block_size: block_size,
            ..Default::default()
        };

        match cache {
            Some(c) => S::cf_descriptors(c, &config),
            None => {
                // Create temporary cache if none provided
                let temp_cache = Cache::new_lru_cache(config.cache_size_bytes);
                S::cf_descriptors(&temp_cache, &config)
            }
        }
    }

    fn on_ready(&self, db: &TransactionDB) -> Result<()> {
        let count = S::prewarm(db, &self.cache, &self.prewarm_config)?;
        tracing::info!(
            subsystem = S::NAME,
            count,
            "[{}] Pre-warmed cache",
            S::NAME
        );
        Ok(())
    }

    fn cf_names(&self) -> Vec<&'static str> {
        S::COLUMN_FAMILIES.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock subsystem for testing
    struct MockSubsystem;

    #[derive(Default, Clone)]
    struct MockPrewarmConfig {
        limit: usize,
    }

    struct MockCache {
        count: std::sync::atomic::AtomicUsize,
    }

    impl MockCache {
        fn new() -> Self {
            Self {
                count: std::sync::atomic::AtomicUsize::new(0),
            }
        }
    }

    impl StorageSubsystem for MockSubsystem {
        const NAME: &'static str = "mock";
        const COLUMN_FAMILIES: &'static [&'static str] = &["mock/data"];

        type PrewarmConfig = MockPrewarmConfig;
        type Cache = MockCache;

        fn create_cache() -> Arc<Self::Cache> {
            Arc::new(MockCache::new())
        }

        fn cf_descriptors(
            _block_cache: &Cache,
            _config: &BlockCacheConfig,
        ) -> Vec<ColumnFamilyDescriptor> {
            vec![ColumnFamilyDescriptor::new(
                "mock/data",
                rocksdb::Options::default(),
            )]
        }

        fn prewarm(
            _db: &dyn DbAccess,
            cache: &Self::Cache,
            config: &Self::PrewarmConfig,
        ) -> Result<usize> {
            cache
                .count
                .store(config.limit, std::sync::atomic::Ordering::SeqCst);
            Ok(config.limit)
        }
    }

    #[test]
    fn test_component_wrapper_new() {
        let wrapper = ComponentWrapper::<MockSubsystem>::new();
        assert_eq!(wrapper.prewarm_config().limit, 0);
    }

    #[test]
    fn test_component_wrapper_with_config() {
        let wrapper = ComponentWrapper::<MockSubsystem>::new()
            .with_prewarm_config(MockPrewarmConfig { limit: 100 });
        assert_eq!(wrapper.prewarm_config().limit, 100);
    }

    #[test]
    fn test_component_wrapper_provider_name() {
        let wrapper = ComponentWrapper::<MockSubsystem>::new();
        assert_eq!(ColumnFamilyProvider::name(&wrapper), "mock");
    }

    #[test]
    fn test_component_wrapper_cf_names() {
        let wrapper = ComponentWrapper::<MockSubsystem>::new();
        assert_eq!(wrapper.cf_names(), vec!["mock/data"]);
    }
}
