//! Vector storage subsystem implementation.
//!
//! Defines the vector subsystem for use with `rocksdb::Storage<S>`.

use std::sync::Arc;

use anyhow::Result;
use rocksdb::{Cache, ColumnFamilyDescriptor, IteratorMode};

use crate::rocksdb::{BlockCacheConfig, DbAccess, StorageSubsystem};

use super::registry::EmbeddingRegistry;
use super::schema::{self, ALL_COLUMN_FAMILIES, EmbeddingSpecs};

// ============================================================================
// Vector Subsystem
// ============================================================================

/// Vector storage subsystem.
///
/// Implements `StorageSubsystem` to define:
/// - Column families for vectors, edges, embeddings, etc.
/// - EmbeddingRegistry for embedding spec lookups
/// - Pre-warming logic for the embedding registry
pub struct Subsystem;

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
        // Map common BlockCacheConfig to vector-specific config
        let vector_config = VectorBlockCacheConfig {
            cache_size_bytes: config.cache_size_bytes,
            default_block_size: config.default_block_size,
            vector_block_size: config.large_block_size,
        };

        vec![
            ColumnFamilyDescriptor::new(
                schema::EmbeddingSpecs::CF_NAME,
                schema::EmbeddingSpecs::column_family_options_with_cache(block_cache, &vector_config),
            ),
            ColumnFamilyDescriptor::new(
                schema::Vectors::CF_NAME,
                schema::Vectors::column_family_options_with_cache(block_cache, &vector_config),
            ),
            ColumnFamilyDescriptor::new(
                schema::Edges::CF_NAME,
                schema::Edges::column_family_options_with_cache(block_cache, &vector_config),
            ),
            ColumnFamilyDescriptor::new(
                schema::BinaryCodes::CF_NAME,
                schema::BinaryCodes::column_family_options_with_cache(block_cache, &vector_config),
            ),
            ColumnFamilyDescriptor::new(
                schema::VecMeta::CF_NAME,
                schema::VecMeta::column_family_options_with_cache(block_cache, &vector_config),
            ),
            ColumnFamilyDescriptor::new(
                schema::GraphMeta::CF_NAME,
                schema::GraphMeta::column_family_options_with_cache(block_cache, &vector_config),
            ),
            ColumnFamilyDescriptor::new(
                schema::IdForward::CF_NAME,
                schema::IdForward::column_family_options_with_cache(block_cache, &vector_config),
            ),
            ColumnFamilyDescriptor::new(
                schema::IdReverse::CF_NAME,
                schema::IdReverse::column_family_options_with_cache(block_cache, &vector_config),
            ),
            ColumnFamilyDescriptor::new(
                schema::IdAlloc::CF_NAME,
                schema::IdAlloc::column_family_options_with_cache(block_cache, &vector_config),
            ),
            ColumnFamilyDescriptor::new(
                schema::Pending::CF_NAME,
                schema::Pending::column_family_options_with_cache(block_cache, &vector_config),
            ),
        ]
    }

    fn prewarm(
        db: &dyn DbAccess,
        cache: &Self::Cache,
        config: &Self::PrewarmConfig,
    ) -> Result<usize> {
        if config.prewarm_limit == 0 {
            return Ok(0);
        }

        let mut loaded = 0;
        let limit = config.prewarm_limit;

        let iter = db.iterator_cf(EmbeddingSpecs::CF_NAME)?;

        for item in iter {
            if loaded >= limit {
                break;
            }

            let (key_bytes, value_bytes) = item?;

            // Parse key and value
            if let (Ok(key), Ok(value)) = (
                EmbeddingSpecs::key_from_bytes(&key_bytes),
                EmbeddingSpecs::value_from_bytes(&value_bytes),
            ) {
                cache.register_from_db(key.0, &value.0, value.1, value.2);
                loaded += 1;
            }
        }

        Ok(loaded)
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
    fn test_subsystem_constants() {
        assert_eq!(Subsystem::NAME, "vector");
        assert!(!Subsystem::COLUMN_FAMILIES.is_empty());
        // Verify all CFs have vector/ prefix
        for cf in Subsystem::COLUMN_FAMILIES {
            assert!(cf.starts_with("vector/"), "CF {} should have vector/ prefix", cf);
        }
    }
}
