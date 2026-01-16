//! Vector storage subsystem implementation.
//!
//! Defines the vector subsystem implementing the unified SubsystemProvider pattern.

use std::sync::{Arc, RwLock};

use anyhow::Result;
use rocksdb::{Cache, ColumnFamilyDescriptor, TransactionDB};

use crate::rocksdb::{prewarm_cf, BlockCacheConfig, ColumnFamily, ColumnFamilyConfig, DbAccess, RocksdbSubsystem, StorageSubsystem};
use crate::SubsystemProvider;
use motlie_core::telemetry::SubsystemInfo;

use super::registry::EmbeddingRegistry;
use super::schema::{self, ALL_COLUMN_FAMILIES, EmbeddingSpecs};
use super::writer::Writer;

// ============================================================================
// Vector Subsystem
// ============================================================================

/// Vector storage subsystem.
///
/// Implements the unified SubsystemProvider pattern:
/// - [`SubsystemInfo`] - Identity and observability
/// - [`SubsystemProvider<TransactionDB>`] - Lifecycle hooks
/// - [`RocksdbSubsystem`] - Column family management
///
/// # Example
///
/// ```ignore
/// use motlie_db::vector::Subsystem;
/// use motlie_db::storage_builder::StorageBuilder;
///
/// let subsystem = Subsystem::new();
/// let registry = subsystem.cache().clone();
///
/// let storage = StorageBuilder::new(path)
///     .with_rocksdb(Box::new(subsystem))
///     .build()?;
/// ```
pub struct Subsystem {
    /// In-memory embedding registry.
    cache: Arc<EmbeddingRegistry>,
    /// Pre-warm configuration.
    prewarm_config: EmbeddingRegistryConfig,
    /// Optional writer for graceful shutdown flush.
    /// Registered via `set_writer()` after storage initialization.
    writer: RwLock<Option<Writer>>,
}

impl Subsystem {
    /// Create a new vector subsystem with default configuration.
    pub fn new() -> Self {
        Self {
            cache: Arc::new(EmbeddingRegistry::new()),
            prewarm_config: EmbeddingRegistryConfig::default(),
            writer: RwLock::new(None),
        }
    }

    /// Set the pre-warm configuration.
    pub fn with_prewarm_config(mut self, config: EmbeddingRegistryConfig) -> Self {
        self.prewarm_config = config;
        self
    }

    /// Get a reference to the embedding registry.
    pub fn cache(&self) -> &Arc<EmbeddingRegistry> {
        &self.cache
    }

    /// Get the pre-warm configuration.
    pub fn prewarm_config(&self) -> &EmbeddingRegistryConfig {
        &self.prewarm_config
    }

    /// Register a writer for graceful shutdown flush.
    ///
    /// When `on_shutdown()` is called, the subsystem will attempt to flush
    /// any pending mutations through this writer before returning.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let subsystem = Subsystem::new();
    /// let storage = StorageBuilder::new(path)
    ///     .with_rocksdb(Box::new(subsystem))
    ///     .build()?;
    ///
    /// let (writer, receiver) = create_writer(WriterConfig::default());
    /// subsystem.set_writer(writer.clone());
    /// // spawn consumers...
    /// ```
    pub fn set_writer(&self, writer: Writer) {
        *self.writer.write().expect("writer lock poisoned") = Some(writer);
    }

    /// Clear the registered writer.
    ///
    /// Call this if the writer/consumers are shut down independently.
    pub fn clear_writer(&self) {
        *self.writer.write().expect("writer lock poisoned") = None;
    }

    /// Internal method to build CF descriptors.
    fn build_cf_descriptors(
        block_cache: &Cache,
        config: &BlockCacheConfig,
    ) -> Vec<ColumnFamilyDescriptor> {
        // Map common BlockCacheConfig to vector-specific config
        let vector_config = VectorBlockCacheConfig {
            cache_size_bytes: config.cache_size_bytes,
            default_block_size: config.default_block_size,
            vector_block_size: config.large_block_size,
        };

        vec![
            ColumnFamilyDescriptor::new(
                <schema::EmbeddingSpecs as ColumnFamily>::CF_NAME,
                <schema::EmbeddingSpecs as ColumnFamilyConfig<VectorBlockCacheConfig>>::cf_options(block_cache, &vector_config),
            ),
            ColumnFamilyDescriptor::new(
                <schema::Vectors as ColumnFamily>::CF_NAME,
                <schema::Vectors as ColumnFamilyConfig<VectorBlockCacheConfig>>::cf_options(block_cache, &vector_config),
            ),
            ColumnFamilyDescriptor::new(
                <schema::Edges as ColumnFamily>::CF_NAME,
                <schema::Edges as ColumnFamilyConfig<VectorBlockCacheConfig>>::cf_options(block_cache, &vector_config),
            ),
            ColumnFamilyDescriptor::new(
                <schema::BinaryCodes as ColumnFamily>::CF_NAME,
                <schema::BinaryCodes as ColumnFamilyConfig<VectorBlockCacheConfig>>::cf_options(block_cache, &vector_config),
            ),
            ColumnFamilyDescriptor::new(
                <schema::VecMeta as ColumnFamily>::CF_NAME,
                <schema::VecMeta as ColumnFamilyConfig<VectorBlockCacheConfig>>::cf_options(block_cache, &vector_config),
            ),
            ColumnFamilyDescriptor::new(
                <schema::GraphMeta as ColumnFamily>::CF_NAME,
                <schema::GraphMeta as ColumnFamilyConfig<VectorBlockCacheConfig>>::cf_options(block_cache, &vector_config),
            ),
            ColumnFamilyDescriptor::new(
                <schema::IdForward as ColumnFamily>::CF_NAME,
                <schema::IdForward as ColumnFamilyConfig<VectorBlockCacheConfig>>::cf_options(block_cache, &vector_config),
            ),
            ColumnFamilyDescriptor::new(
                <schema::IdReverse as ColumnFamily>::CF_NAME,
                <schema::IdReverse as ColumnFamilyConfig<VectorBlockCacheConfig>>::cf_options(block_cache, &vector_config),
            ),
            ColumnFamilyDescriptor::new(
                <schema::IdAlloc as ColumnFamily>::CF_NAME,
                <schema::IdAlloc as ColumnFamilyConfig<VectorBlockCacheConfig>>::cf_options(block_cache, &vector_config),
            ),
            ColumnFamilyDescriptor::new(
                <schema::Pending as ColumnFamily>::CF_NAME,
                <schema::Pending as ColumnFamilyConfig<VectorBlockCacheConfig>>::cf_options(block_cache, &vector_config),
            ),
        ]
    }
}

impl Default for Subsystem {
    fn default() -> Self {
        Self::new()
    }
}

// ----------------------------------------------------------------------------
// SubsystemInfo Implementation (from motlie_core::telemetry)
// ----------------------------------------------------------------------------

impl SubsystemInfo for Subsystem {
    fn name(&self) -> &'static str {
        "Vector Database (RocksDB)"
    }

    fn info_lines(&self) -> Vec<(&'static str, String)> {
        vec![
            ("Prewarm Limit", self.prewarm_config.prewarm_limit.to_string()),
            ("Column Families", ALL_COLUMN_FAMILIES.len().to_string()),
        ]
    }
}

// ----------------------------------------------------------------------------
// SubsystemProvider<TransactionDB> Implementation
// ----------------------------------------------------------------------------

impl SubsystemProvider<TransactionDB> for Subsystem {
    fn on_ready(&self, db: &TransactionDB) -> Result<()> {
        let count = prewarm_cf::<EmbeddingSpecs, _>(db, self.prewarm_config.prewarm_limit, |key, value| {
            let spec = &value.0;
            self.cache.register_from_db(key.0, &spec.model, spec.dim, spec.distance, spec.storage_type);
            Ok(())
        })?;
        tracing::info!(subsystem = "vector", count, "Pre-warmed embedding registry");
        Ok(())
    }

    fn on_shutdown(&self) -> Result<()> {
        tracing::info!(subsystem = "vector", "Shutting down");

        // Best-effort flush of pending mutations
        if let Some(writer) = self.writer.read().expect("writer lock poisoned").as_ref() {
            if !writer.is_closed() {
                tracing::debug!(subsystem = "vector", "Flushing pending mutations");

                // Use block_on since on_shutdown is sync.
                // This is safe because we're just draining an MPSC channel
                // and waiting for the consumer to commit a RocksDB transaction.
                match tokio::runtime::Handle::try_current() {
                    Ok(handle) => {
                        if let Err(e) = handle.block_on(writer.flush()) {
                            tracing::warn!(
                                subsystem = "vector",
                                error = %e,
                                "Failed to flush pending mutations on shutdown (best effort)"
                            );
                        } else {
                            tracing::debug!(subsystem = "vector", "Flushed pending mutations");
                        }
                    }
                    Err(_) => {
                        tracing::warn!(
                            subsystem = "vector",
                            "No tokio runtime available for shutdown flush (best effort)"
                        );
                    }
                }
            }
        }

        Ok(())
    }
}

// ----------------------------------------------------------------------------
// RocksdbSubsystem Implementation
// ----------------------------------------------------------------------------

impl RocksdbSubsystem for Subsystem {
    fn id(&self) -> &'static str {
        "vector"
    }

    fn cf_names(&self) -> &'static [&'static str] {
        ALL_COLUMN_FAMILIES
    }

    fn cf_descriptors(
        &self,
        block_cache: &Cache,
        config: &BlockCacheConfig,
    ) -> Vec<ColumnFamilyDescriptor> {
        Self::build_cf_descriptors(block_cache, config)
    }
}

// ----------------------------------------------------------------------------
// StorageSubsystem Implementation (for standalone Storage<S>)
// ----------------------------------------------------------------------------

impl StorageSubsystem for Subsystem {
    const NAME: &'static str = "vector";
    const COLUMN_FAMILIES: &'static [&'static str] = ALL_COLUMN_FAMILIES;

    type PrewarmConfig = EmbeddingRegistryConfig;
    type Cache = EmbeddingRegistry;

    fn create_cache() -> Arc<Self::Cache> {
        Arc::new(EmbeddingRegistry::new())
    }

    fn cf_descriptors(
        block_cache: &Cache,
        config: &BlockCacheConfig,
    ) -> Vec<ColumnFamilyDescriptor> {
        Self::build_cf_descriptors(block_cache, config)
    }

    fn prewarm(
        db: &dyn DbAccess,
        cache: &Self::Cache,
        config: &Self::PrewarmConfig,
    ) -> Result<usize> {
        prewarm_cf::<EmbeddingSpecs, _>(db, config.prewarm_limit, |key, value| {
            let spec = &value.0;
            cache.register_from_db(key.0, &spec.model, spec.dim, spec.distance, spec.storage_type);
            Ok(())
        })
    }
}

// ============================================================================
// Configuration Types
// ============================================================================

/// Configuration for EmbeddingRegistry pre-warming.
#[derive(Debug, Clone)]
pub struct EmbeddingRegistryConfig {
    /// Maximum number of embedding specs to pre-load from EmbeddingSpecs CF on startup.
    /// Set to 0 to disable pre-warming.
    /// Default: 1000
    pub prewarm_limit: usize,
}

impl Default for EmbeddingRegistryConfig {
    fn default() -> Self {
        Self { prewarm_limit: 1000 }
    }
}

/// Vector-specific block cache configuration.
///
/// This is an internal type that maps from the common `BlockCacheConfig`
/// to the format expected by vector schema methods.
#[derive(Debug, Clone)]
pub struct VectorBlockCacheConfig {
    /// Total block cache size in bytes.
    pub cache_size_bytes: usize,
    /// Default block size for most CFs.
    pub default_block_size: usize,
    /// Block size for vector data (larger for sequential access).
    pub vector_block_size: usize,
}

impl Default for VectorBlockCacheConfig {
    fn default() -> Self {
        Self {
            cache_size_bytes: 256 * 1024 * 1024,
            default_block_size: 4 * 1024,
            vector_block_size: 16 * 1024,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_registry_config_default() {
        let config = EmbeddingRegistryConfig::default();
        assert_eq!(config.prewarm_limit, 1000);
    }

    #[test]
    fn test_vector_block_cache_config_default() {
        let config = VectorBlockCacheConfig::default();
        assert_eq!(config.cache_size_bytes, 256 * 1024 * 1024);
        assert_eq!(config.default_block_size, 4 * 1024);
        assert_eq!(config.vector_block_size, 16 * 1024);
    }

    #[test]
    fn test_subsystem_new() {
        let subsystem = Subsystem::new();
        assert_eq!(subsystem.prewarm_config().prewarm_limit, 1000);
    }

    #[test]
    fn test_subsystem_with_config() {
        let subsystem = Subsystem::new()
            .with_prewarm_config(EmbeddingRegistryConfig { prewarm_limit: 500 });
        assert_eq!(subsystem.prewarm_config().prewarm_limit, 500);
    }

    #[test]
    fn test_subsystem_cf_names() {
        let subsystem = Subsystem::new();
        let cf_names = subsystem.cf_names();
        assert!(!cf_names.is_empty());
        // Verify all CFs have vector/ prefix
        for cf in cf_names {
            assert!(cf.starts_with("vector/"), "CF {} should have vector/ prefix", cf);
        }
    }

    #[test]
    fn test_subsystem_info_name() {
        let subsystem = Subsystem::new();
        assert_eq!(SubsystemInfo::name(&subsystem), "Vector Database (RocksDB)");
    }
}
