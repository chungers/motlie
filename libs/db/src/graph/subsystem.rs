//! Graph storage subsystem implementation.
//!
//! Defines the graph subsystem implementing the unified SubsystemProvider pattern.

use std::sync::{Arc, RwLock};

use anyhow::Result;
use rocksdb::{Cache, ColumnFamilyDescriptor, TransactionDB};

use crate::rocksdb::{BlockCacheConfig, DbAccess, RocksdbSubsystem, StorageSubsystem};
use crate::SubsystemProvider;
use motlie_core::telemetry::SubsystemInfo;

use super::name_hash::{NameCache, NameHash};
use super::reader::{create_query_reader, spawn_consumer as spawn_query_consumer, Consumer as QueryConsumer, ReaderConfig, Reader};
use super::schema::{self, ALL_COLUMN_FAMILIES};
use super::writer::{create_mutation_writer, spawn_consumer as spawn_mutation_consumer, Consumer as MutationConsumer, WriterConfig, Writer};
use super::{ColumnFamily, ColumnFamilyConfig, ColumnFamilySerde, Graph, Storage};

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
    /// Optional writer for graceful shutdown flush.
    /// Registered via `set_writer()` after storage initialization.
    writer: RwLock<Option<Writer>>,
}

impl Subsystem {
    /// Create a new graph subsystem with default configuration.
    pub fn new() -> Self {
        Self {
            cache: Arc::new(NameCache::new()),
            prewarm_config: NameCacheConfig::default(),
            writer: RwLock::new(None),
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

    /// Start the graph subsystem with managed lifecycle.
    ///
    /// This is the recommended way to use the graph subsystem. It:
    /// 1. Creates Writer + Reader with internal wiring
    /// 2. Spawns mutation consumer (1 worker)
    /// 3. Spawns query consumers (configurable workers)
    /// 4. Registers Writer for automatic shutdown flush
    ///
    /// # Arguments
    ///
    /// * `storage` - Graph storage instance
    /// * `writer_config` - Configuration for mutation writer
    /// * `reader_config` - Configuration for query reader
    /// * `num_query_workers` - Number of parallel query workers
    ///
    /// # Returns
    ///
    /// Tuple of (Writer, Reader) for sending mutations and queries.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use motlie_db::graph::{Subsystem, WriterConfig, ReaderConfig};
    /// use motlie_db::storage_builder::StorageBuilder;
    ///
    /// // Create subsystem and build storage
    /// let subsystem = Arc::new(Subsystem::new());
    /// let name_cache = subsystem.cache().clone();
    ///
    /// let storage = StorageBuilder::new(path)
    ///     .with_rocksdb(subsystem.clone())
    ///     .build()?;
    ///
    /// // Start with managed lifecycle
    /// let (writer, reader) = subsystem.start(
    ///     storage.graph_storage().clone(),
    ///     WriterConfig::default(),
    ///     ReaderConfig::default(),
    ///     4,  // 4 query workers
    /// );
    ///
    /// // Use writer and reader...
    /// AddNode { ... }.run(&writer).await?;
    /// let result = GetNode { ... }.run(&reader, timeout).await?;
    ///
    /// // Shutdown automatically flushes pending mutations
    /// storage.shutdown()?;
    /// ```
    ///
    /// # Lifecycle
    ///
    /// When `on_shutdown()` is called (typically via `storage.shutdown()`),
    /// the subsystem will automatically flush any pending mutations before
    /// returning. This ensures data durability without manual flush calls.
    ///
    /// # Alternative: Manual Lifecycle
    ///
    /// If you need more control, use the raw components directly:
    /// ```rust,ignore
    /// let (writer, receiver) = create_mutation_writer(config);
    /// let graph = Arc::new(Graph::new(storage));
    /// spawn_mutation_consumer(...);
    /// // You are responsible for calling writer.flush() before shutdown
    /// ```
    pub fn start(
        &self,
        storage: Arc<Storage>,
        writer_config: WriterConfig,
        reader_config: ReaderConfig,
        num_query_workers: usize,
    ) -> (Writer, Reader) {
        // Create graph processor - Graph is Clone (holds Arc<Storage>)
        let graph = Graph::new(storage);

        // Create and spawn mutation consumer
        let (writer, mutation_receiver) = create_mutation_writer(writer_config.clone());
        let mutation_consumer = MutationConsumer::new(mutation_receiver, writer_config, graph.clone());
        let _mutation_handle = spawn_mutation_consumer(mutation_consumer);

        // Create and spawn query consumers
        let (reader, query_receiver) = create_query_reader(reader_config.clone());
        for _ in 0..num_query_workers {
            let consumer = QueryConsumer::new(query_receiver.clone(), reader_config.clone(), graph.clone());
            let _query_handle = spawn_query_consumer(consumer);
        }

        // Register writer for shutdown flush
        self.set_writer(writer.clone());

        (writer, reader)
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

        // Best-effort flush of pending mutations
        if let Some(writer) = self.writer.read().expect("writer lock poisoned").as_ref() {
            if !writer.is_closed() {
                tracing::debug!(subsystem = "graph", "Flushing pending mutations");

                // Use block_on since on_shutdown is sync.
                // This is safe because we're just draining an MPSC channel
                // and waiting for the consumer to commit a RocksDB transaction.
                match tokio::runtime::Handle::try_current() {
                    Ok(handle) => {
                        if let Err(e) = handle.block_on(writer.flush()) {
                            tracing::warn!(
                                subsystem = "graph",
                                error = %e,
                                "Failed to flush pending mutations on shutdown (best effort)"
                            );
                        } else {
                            tracing::debug!(subsystem = "graph", "Flushed pending mutations");
                        }
                    }
                    Err(_) => {
                        tracing::warn!(
                            subsystem = "graph",
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
