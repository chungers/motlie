//! Graph storage subsystem implementation.
//!
//! Defines the graph subsystem implementing the unified SubsystemProvider pattern.

use std::sync::Arc;

use anyhow::Result;
use rocksdb::{Cache, ColumnFamilyDescriptor, TransactionDB};

use crate::rocksdb::{BlockCacheConfig, DbAccess, RocksdbSubsystem, StorageSubsystem};
use crate::SubsystemProvider;
use motlie_core::telemetry::SubsystemInfo;

use super::name_hash::{NameCache, NameHash};
use super::schema::{self, ALL_COLUMN_FAMILIES};
use super::{ColumnFamily, ColumnFamilyConfig, ColumnFamilySerde};

// ============================================================================
// Graph Subsystem
// ============================================================================

/// Graph storage subsystem.
///
/// Implements the unified SubsystemProvider pattern:
/// - [`SubsystemInfo`] - Identity and observability
/// - [`SubsystemProvider<TransactionDB>`] - Lifecycle hooks
/// - [`RocksdbSubsystem`] - Column family management
///
/// # Example
///
/// ```ignore
/// use motlie_db::graph::Subsystem;
/// use motlie_db::storage_builder::StorageBuilder;
///
/// let subsystem = Subsystem::new();
/// let name_cache = subsystem.cache().clone();
///
/// let storage = StorageBuilder::new(path)
///     .with_rocksdb(Box::new(subsystem))
///     .build()?;
/// ```
pub struct Subsystem {
    /// In-memory name cache for deduplication.
    cache: Arc<NameCache>,
    /// Pre-warm configuration.
    prewarm_config: NameCacheConfig,
}

impl Subsystem {
    /// Create a new graph subsystem with default configuration.
    pub fn new() -> Self {
        Self {
            cache: Arc::new(NameCache::new()),
            prewarm_config: NameCacheConfig::default(),
        }
    }

    /// Set the pre-warm configuration.
    pub fn with_prewarm_config(mut self, config: NameCacheConfig) -> Self {
        self.prewarm_config = config;
        self
    }

    /// Get a reference to the name cache.
    pub fn cache(&self) -> &Arc<NameCache> {
        &self.cache
    }

    /// Get the pre-warm configuration.
    pub fn prewarm_config(&self) -> &NameCacheConfig {
        &self.prewarm_config
    }

    /// Internal method to prewarm the name cache.
    fn prewarm_impl(db: &dyn DbAccess, cache: &NameCache, config: &NameCacheConfig) -> Result<usize> {
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

    /// Internal method to build CF descriptors.
    fn build_cf_descriptors(
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
                <schema::Names as ColumnFamily>::CF_NAME,
                <schema::Names as ColumnFamilyConfig<GraphBlockCacheConfig>>::cf_options(block_cache, &graph_config),
            ),
            ColumnFamilyDescriptor::new(
                <schema::Nodes as ColumnFamily>::CF_NAME,
                <schema::Nodes as ColumnFamilyConfig<GraphBlockCacheConfig>>::cf_options(block_cache, &graph_config),
            ),
            ColumnFamilyDescriptor::new(
                <schema::NodeFragments as ColumnFamily>::CF_NAME,
                <schema::NodeFragments as ColumnFamilyConfig<GraphBlockCacheConfig>>::cf_options(block_cache, &graph_config),
            ),
            ColumnFamilyDescriptor::new(
                <schema::NodeSummaries as ColumnFamily>::CF_NAME,
                <schema::NodeSummaries as ColumnFamilyConfig<GraphBlockCacheConfig>>::cf_options(block_cache, &graph_config),
            ),
            ColumnFamilyDescriptor::new(
                <schema::EdgeFragments as ColumnFamily>::CF_NAME,
                <schema::EdgeFragments as ColumnFamilyConfig<GraphBlockCacheConfig>>::cf_options(block_cache, &graph_config),
            ),
            ColumnFamilyDescriptor::new(
                <schema::EdgeSummaries as ColumnFamily>::CF_NAME,
                <schema::EdgeSummaries as ColumnFamilyConfig<GraphBlockCacheConfig>>::cf_options(block_cache, &graph_config),
            ),
            ColumnFamilyDescriptor::new(
                <schema::ForwardEdges as ColumnFamily>::CF_NAME,
                <schema::ForwardEdges as ColumnFamilyConfig<GraphBlockCacheConfig>>::cf_options(block_cache, &graph_config),
            ),
            ColumnFamilyDescriptor::new(
                <schema::ReverseEdges as ColumnFamily>::CF_NAME,
                <schema::ReverseEdges as ColumnFamilyConfig<GraphBlockCacheConfig>>::cf_options(block_cache, &graph_config),
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
        "Graph Database (RocksDB)"
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
        let count = Self::prewarm_impl(db, &self.cache, &self.prewarm_config)?;
        tracing::info!(subsystem = "graph", count, "Pre-warmed name cache");
        Ok(())
    }

    fn on_shutdown(&self) -> Result<()> {
        tracing::info!(subsystem = "graph", "Shutting down");
        Ok(())
    }
}

// ----------------------------------------------------------------------------
// RocksdbSubsystem Implementation
// ----------------------------------------------------------------------------

impl RocksdbSubsystem for Subsystem {
    fn id(&self) -> &'static str {
        "graph"
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
        Self::build_cf_descriptors(block_cache, config)
    }

    fn prewarm(
        db: &dyn DbAccess,
        cache: &Self::Cache,
        config: &Self::PrewarmConfig,
    ) -> Result<usize> {
        Self::prewarm_impl(db, cache, config)
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
    fn test_subsystem_new() {
        let subsystem = Subsystem::new();
        assert_eq!(subsystem.prewarm_config().prewarm_limit, 1000);
    }

    #[test]
    fn test_subsystem_with_config() {
        let subsystem = Subsystem::new()
            .with_prewarm_config(NameCacheConfig { prewarm_limit: 500 });
        assert_eq!(subsystem.prewarm_config().prewarm_limit, 500);
    }

    #[test]
    fn test_subsystem_cf_names() {
        let subsystem = Subsystem::new();
        let cf_names = subsystem.cf_names();
        assert!(!cf_names.is_empty());
        // Verify known CFs are present (all graph CFs use "graph/" prefix)
        let cfs: std::collections::HashSet<_> = cf_names.iter().copied().collect();
        assert!(cfs.contains("graph/names"), "Should have graph/names CF");
        assert!(cfs.contains("graph/nodes"), "Should have graph/nodes CF");
        assert!(cfs.contains("graph/forward_edges"), "Should have graph/forward_edges CF");
        assert!(cfs.contains("graph/reverse_edges"), "Should have graph/reverse_edges CF");
    }

    #[test]
    fn test_subsystem_info_name() {
        let subsystem = Subsystem::new();
        assert_eq!(SubsystemInfo::name(&subsystem), "Graph Database (RocksDB)");
    }
}
