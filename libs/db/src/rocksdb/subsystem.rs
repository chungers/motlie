//! RocksDB subsystem traits for modular storage.
//!
//! This module defines the core abstractions for RocksDB storage subsystems:
//! - `DbAccess`: Trait for database read operations
//! - `StorageSubsystem`: Trait for standalone `Storage<S>` usage with associated types
//! - `RocksdbSubsystem`: Extension trait for shared `StorageBuilder` usage

use std::sync::Arc;

use anyhow::Result;
use rocksdb::{Cache, ColumnFamilyDescriptor, TransactionDB, DB};

use super::config::BlockCacheConfig;
use crate::SubsystemProvider;

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
// StorageSubsystem Trait (for standalone Storage<S>)
// ============================================================================

/// Trait for standalone RocksDB storage subsystems.
///
/// This trait is used with `Storage<S>` for subsystems that manage their own
/// RocksDB instance. For shared storage via `StorageBuilder`, use [`RocksdbSubsystem`].
///
/// # Associated Types
///
/// - `PrewarmConfig`: Configuration for cache pre-warming
/// - `Cache`: The in-memory cache type for this subsystem
///
/// # Example
///
/// ```ignore
/// use motlie_db::rocksdb::{StorageSubsystem, Storage, BlockCacheConfig, DbAccess};
/// use std::sync::Arc;
///
/// struct MySubsystem;
///
/// #[derive(Default, Clone)]
/// struct MyPrewarmConfig { limit: usize }
///
/// struct MyCache { /* ... */ }
///
/// impl StorageSubsystem for MySubsystem {
///     const NAME: &'static str = "my-subsystem";
///     const COLUMN_FAMILIES: &'static [&'static str] = &["my/data"];
///
///     type PrewarmConfig = MyPrewarmConfig;
///     type Cache = MyCache;
///
///     fn create_cache() -> Arc<Self::Cache> {
///         Arc::new(MyCache { /* ... */ })
///     }
///
///     fn cf_descriptors(
///         block_cache: &rocksdb::Cache,
///         config: &BlockCacheConfig,
///     ) -> Vec<rocksdb::ColumnFamilyDescriptor> {
///         vec![/* ... */]
///     }
///
///     fn prewarm(
///         db: &dyn DbAccess,
///         cache: &Self::Cache,
///         config: &Self::PrewarmConfig,
///     ) -> anyhow::Result<usize> {
///         // Pre-warm cache from DB
///         Ok(0)
///     }
/// }
///
/// // Use as: Storage<MySubsystem>
/// ```
pub trait StorageSubsystem: Sized {
    /// Subsystem name for logging and identification.
    const NAME: &'static str;

    /// List of column family names managed by this subsystem.
    const COLUMN_FAMILIES: &'static [&'static str];

    /// Configuration type for cache pre-warming.
    type PrewarmConfig: Default + Clone;

    /// In-memory cache type for this subsystem.
    type Cache: Send + Sync;

    /// Create a new instance of the subsystem's cache.
    fn create_cache() -> Arc<Self::Cache>;

    /// Build column family descriptors with shared block cache.
    fn cf_descriptors(
        block_cache: &Cache,
        config: &BlockCacheConfig,
    ) -> Vec<ColumnFamilyDescriptor>;

    /// Pre-warm the cache from the database.
    ///
    /// Returns the number of entries loaded.
    fn prewarm(
        db: &dyn DbAccess,
        cache: &Self::Cache,
        config: &Self::PrewarmConfig,
    ) -> Result<usize>;
}

// ============================================================================
// RocksdbSubsystem Trait (for shared StorageBuilder)
// ============================================================================

/// Extension trait for RocksDB-based subsystems.
///
/// Extends [`SubsystemProvider<TransactionDB>`] with RocksDB-specific methods
/// for column family management.
///
/// # Example
///
/// ```ignore
/// use motlie_db::rocksdb::RocksdbSubsystem;
/// use motlie_db::SubsystemProvider;
/// use motlie_core::telemetry::SubsystemInfo;
///
/// struct MySubsystem { /* ... */ }
///
/// impl SubsystemInfo for MySubsystem {
///     fn name(&self) -> &'static str { "My Subsystem (RocksDB)" }
///     fn info_lines(&self) -> Vec<(&'static str, String)> { vec![] }
/// }
///
/// impl SubsystemProvider<TransactionDB> for MySubsystem {
///     fn on_ready(&self, db: &TransactionDB) -> anyhow::Result<()> {
///         // Pre-warm caches
///         Ok(())
///     }
/// }
///
/// impl RocksdbSubsystem for MySubsystem {
///     fn id(&self) -> &'static str { "my-subsystem" }
///
///     fn cf_names(&self) -> &'static [&'static str] {
///         &["my-subsystem/data"]
///     }
///
///     fn cf_descriptors(&self, cache: &Cache, config: &BlockCacheConfig) -> Vec<ColumnFamilyDescriptor> {
///         // Return CF descriptors
///         vec![]
///     }
/// }
/// ```
pub trait RocksdbSubsystem: SubsystemProvider<TransactionDB> {
    /// Short identifier for this subsystem (e.g., "graph", "vector").
    ///
    /// Used for programmatic lookup via `SharedStorage::get_component()`.
    /// This is distinct from `SubsystemInfo::name()` which is for display.
    fn id(&self) -> &'static str;

    /// List of column family names managed by this subsystem.
    fn cf_names(&self) -> &'static [&'static str];

    /// Build column family descriptors with shared block cache.
    ///
    /// Called during database initialization to collect all CF descriptors.
    fn cf_descriptors(
        &self,
        block_cache: &Cache,
        config: &BlockCacheConfig,
    ) -> Vec<ColumnFamilyDescriptor>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use motlie_core::telemetry::SubsystemInfo;

    // Mock subsystem for testing
    struct MockSubsystem;

    impl SubsystemInfo for MockSubsystem {
        fn name(&self) -> &'static str {
            "mock"
        }

        fn info_lines(&self) -> Vec<(&'static str, String)> {
            vec![("Test", "value".to_string())]
        }
    }

    impl SubsystemProvider<TransactionDB> for MockSubsystem {}

    impl RocksdbSubsystem for MockSubsystem {
        fn id(&self) -> &'static str {
            "mock"
        }

        fn cf_names(&self) -> &'static [&'static str] {
            &["mock/data"]
        }

        fn cf_descriptors(
            &self,
            _block_cache: &Cache,
            _config: &BlockCacheConfig,
        ) -> Vec<ColumnFamilyDescriptor> {
            vec![ColumnFamilyDescriptor::new(
                "mock/data",
                rocksdb::Options::default(),
            )]
        }
    }

    #[test]
    fn test_mock_subsystem_name() {
        let subsystem = MockSubsystem;
        assert_eq!(SubsystemInfo::name(&subsystem), "mock");
    }

    #[test]
    fn test_mock_subsystem_cf_names() {
        let subsystem = MockSubsystem;
        assert_eq!(subsystem.cf_names(), &["mock/data"]);
    }

    #[test]
    fn test_mock_subsystem_cf_descriptors() {
        let subsystem = MockSubsystem;
        let cache = Cache::new_lru_cache(1024 * 1024);
        let config = BlockCacheConfig::default();
        let descriptors = subsystem.cf_descriptors(&cache, &config);
        assert_eq!(descriptors.len(), 1);
    }
}
