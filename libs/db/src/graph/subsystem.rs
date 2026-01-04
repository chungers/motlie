//! Graph storage subsystem implementation.
//!
//! Defines the graph subsystem for use with `rocksdb::Storage<S>`.

use std::sync::Arc;

use anyhow::Result;
use rocksdb::{Cache, ColumnFamilyDescriptor};

use crate::rocksdb::{BlockCacheConfig, DbAccess, StorageSubsystem};

use super::name_hash::{NameCache, NameHash};
use super::schema::{self, ALL_COLUMN_FAMILIES};
use super::ColumnFamilyRecord;

// ============================================================================
// Graph Subsystem
// ============================================================================

/// Graph storage subsystem.
///
/// Implements `StorageSubsystem` to define:
/// - Column families for nodes, edges, fragments, etc.
/// - NameCache for name deduplication
/// - Pre-warming logic for the name cache
pub struct Subsystem;

impl StorageSubsystem for Subsystem {
    const NAME: &'static str = "graph";
    const COLUMN_FAMILIES: &'static [&'static str] = ALL_COLUMN_FAMILIES;

    type PrewarmConfig = NameCacheConfig;
    type Cache = NameCache;

    fn create_cache() -> Arc<Self::Cache> {
        Arc::new(NameCache::new())
    }

    fn cf_descriptors(
        block_cache: &Cache,
        config: &BlockCacheConfig,
    ) -> Vec<ColumnFamilyDescriptor> {
        // Map common BlockCacheConfig to graph-specific config
        let graph_config = GraphBlockCacheConfig {
            cache_size_bytes: config.cache_size_bytes,
            graph_block_size: config.default_block_size,
            fragment_block_size: config.large_block_size,
            cache_index_and_filter_blocks: config.cache_index_and_filter_blocks,
            pin_l0_filter_and_index: config.pin_l0_filter_and_index,
        };

        vec![
            ColumnFamilyDescriptor::new(
                schema::Names::CF_NAME,
                schema::Names::column_family_options_with_cache(block_cache, &graph_config),
            ),
            ColumnFamilyDescriptor::new(
                schema::Nodes::CF_NAME,
                schema::Nodes::column_family_options_with_cache(block_cache, &graph_config),
            ),
            ColumnFamilyDescriptor::new(
                schema::NodeFragments::CF_NAME,
                schema::NodeFragments::column_family_options_with_cache(block_cache, &graph_config),
            ),
            ColumnFamilyDescriptor::new(
                schema::NodeSummaries::CF_NAME,
                schema::NodeSummaries::column_family_options_with_cache(block_cache, &graph_config),
            ),
            ColumnFamilyDescriptor::new(
                schema::EdgeFragments::CF_NAME,
                schema::EdgeFragments::column_family_options_with_cache(block_cache, &graph_config),
            ),
            ColumnFamilyDescriptor::new(
                schema::EdgeSummaries::CF_NAME,
                schema::EdgeSummaries::column_family_options_with_cache(block_cache, &graph_config),
            ),
            ColumnFamilyDescriptor::new(
                schema::ForwardEdges::CF_NAME,
                schema::ForwardEdges::column_family_options_with_cache(block_cache, &graph_config),
            ),
            ColumnFamilyDescriptor::new(
                schema::ReverseEdges::CF_NAME,
                schema::ReverseEdges::column_family_options_with_cache(block_cache, &graph_config),
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

        let iter = db.iterator_cf(schema::Names::CF_NAME)?;

        for item in iter {
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
                    cache.insert(hash, value.0);
                    loaded += 1;
                }
            }
        }

        Ok(loaded)
    }
}

// ============================================================================
// Configuration Types
// ============================================================================

/// Configuration for NameCache pre-warming.
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

/// Graph-specific block cache configuration.
///
/// This is an internal type that maps from the common `BlockCacheConfig`
/// to the format expected by graph schema methods.
#[derive(Debug, Clone)]
pub struct GraphBlockCacheConfig {
    /// Total block cache size in bytes.
    pub cache_size_bytes: usize,
    /// Block size for graph CFs (Nodes, Edges, Names).
    pub graph_block_size: usize,
    /// Block size for fragment CFs (NodeFragments, EdgeFragments).
    pub fragment_block_size: usize,
    /// Whether to cache index and filter blocks.
    pub cache_index_and_filter_blocks: bool,
    /// Whether to pin L0 filter and index blocks.
    pub pin_l0_filter_and_index: bool,
}

impl Default for GraphBlockCacheConfig {
    fn default() -> Self {
        Self {
            cache_size_bytes: 256 * 1024 * 1024,
            graph_block_size: 4 * 1024,
            fragment_block_size: 16 * 1024,
            cache_index_and_filter_blocks: true,
            pin_l0_filter_and_index: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name_cache_config_default() {
        let config = NameCacheConfig::default();
        assert_eq!(config.prewarm_limit, 1000);
    }

    #[test]
    fn test_graph_block_cache_config_default() {
        let config = GraphBlockCacheConfig::default();
        assert_eq!(config.cache_size_bytes, 256 * 1024 * 1024);
        assert_eq!(config.graph_block_size, 4 * 1024);
        assert_eq!(config.fragment_block_size, 16 * 1024);
    }

    #[test]
    fn test_subsystem_constants() {
        assert_eq!(Subsystem::NAME, "graph");
        assert!(!Subsystem::COLUMN_FAMILIES.is_empty());
        // Verify known CFs are present
        let cfs: std::collections::HashSet<_> = Subsystem::COLUMN_FAMILIES.iter().copied().collect();
        assert!(cfs.contains("names"), "Should have names CF");
        assert!(cfs.contains("nodes"), "Should have nodes CF");
        assert!(cfs.contains("forward_edges"), "Should have forward_edges CF");
        assert!(cfs.contains("reverse_edges"), "Should have reverse_edges CF");
    }
}
