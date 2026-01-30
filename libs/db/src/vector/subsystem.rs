//! Vector storage subsystem implementation.
//!
//! Defines the vector subsystem implementing the unified SubsystemProvider pattern.

use std::sync::{Arc, RwLock};

use anyhow::Result;
use rocksdb::{Cache, ColumnFamilyDescriptor, TransactionDB};

use crate::rocksdb::{prewarm_cf, BlockCacheConfig, ColumnFamily, ColumnFamilyConfig, DbAccess, RocksdbSubsystem, StorageSubsystem};
use crate::SubsystemProvider;
use motlie_core::telemetry::SubsystemInfo;

use super::async_updater::{AsyncGraphUpdater, AsyncUpdaterConfig};
use super::cache::NavigationCache;
use super::gc::{GarbageCollector, GcConfig};
use super::processor::Processor;
use super::reader::{create_search_reader, ReaderConfig, SearchReader, spawn_consumers_with_processor};
use super::registry::EmbeddingRegistry;
use super::schema::{self, ALL_COLUMN_FAMILIES, EmbeddingSpecs};
use super::writer::{create_writer, spawn_consumer, Consumer, Writer, WriterConfig};

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
    /// Optional async graph updater for two-phase inserts.
    /// Registered via `start_with_async()` when async updates are enabled.
    async_updater: RwLock<Option<AsyncGraphUpdater>>,
    /// Navigation cache for HNSW operations.
    /// Shared between Processor and AsyncGraphUpdater.
    nav_cache: Arc<NavigationCache>,
    /// Optional garbage collector for deleted vector cleanup.
    /// Started via `start_with_async()` when GcConfig provided.
    gc: RwLock<Option<GarbageCollector>>,
    /// Consumer task handles for graceful shutdown.
    /// Joined after writer channel closes to ensure clean exit.
    consumer_handles: RwLock<Vec<tokio::task::JoinHandle<anyhow::Result<()>>>>,
}

impl Subsystem {
    /// Create a new vector subsystem with default configuration.
    pub fn new() -> Self {
        Self {
            cache: Arc::new(EmbeddingRegistry::new()),
            prewarm_config: EmbeddingRegistryConfig::default(),
            writer: RwLock::new(None),
            async_updater: RwLock::new(None),
            nav_cache: Arc::new(NavigationCache::default()),
            gc: RwLock::new(None),
            consumer_handles: RwLock::new(Vec::new()),
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

    /// Start the vector subsystem with managed lifecycle.
    ///
    /// This is the recommended way to use the vector subsystem. It:
    /// 1. Creates Writer + SearchReader with internal wiring
    /// 2. Spawns mutation consumer (1 worker)
    /// 3. Spawns query consumers (configurable workers)
    /// 4. Registers Writer for automatic shutdown flush
    ///
    /// # Arguments
    ///
    /// * `storage` - Vector storage instance
    /// * `writer_config` - Configuration for mutation writer
    /// * `reader_config` - Configuration for query reader
    /// * `num_query_workers` - Number of parallel query workers
    ///
    /// # Returns
    ///
    /// Tuple of (Writer, SearchReader) for sending mutations and queries.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use motlie_db::vector::{Subsystem, WriterConfig, ReaderConfig};
    /// use motlie_db::storage_builder::StorageBuilder;
    ///
    /// // Create subsystem and build storage
    /// let subsystem = Arc::new(Subsystem::new());
    /// let registry = subsystem.cache().clone();
    ///
    /// let storage = StorageBuilder::new(path)
    ///     .with_rocksdb(subsystem.clone())
    ///     .build()?;
    ///
    /// // Start with managed lifecycle
    /// let (writer, search_reader) = subsystem.start(
    ///     storage.vector_storage().clone(),
    ///     WriterConfig::default(),
    ///     ReaderConfig::default(),
    ///     4,  // 4 query workers
    /// );
    ///
    /// // Use writer and search_reader...
    /// InsertVector::new(&embedding, id, vec).run(&writer).await?;
    /// let results = search_reader.search_knn(&embedding, query, 10, 100, timeout).await?;
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
    /// let (writer, receiver) = create_writer(config);
    /// let processor = Arc::new(Processor::new(storage, registry));
    /// spawn_mutation_consumer_with_storage(...);
    /// // You are responsible for calling writer.flush() before shutdown
    /// ```
    ///
    /// # See Also
    ///
    /// - [`start_with_async`] - Start with async graph updater for two-phase inserts
    pub fn start(
        &self,
        storage: Arc<super::Storage>,
        writer_config: WriterConfig,
        reader_config: ReaderConfig,
        num_query_workers: usize,
    ) -> (Writer, SearchReader) {
        self.start_with_async(storage, writer_config, reader_config, num_query_workers, None, None)
    }

    /// Start the vector subsystem with async graph updater for two-phase inserts.
    ///
    /// This extends [`start`] with an optional async graph updater that enables
    /// low-latency inserts by decoupling vector storage from HNSW graph construction,
    /// and an optional garbage collector for deleted vector cleanup.
    ///
    /// ## Two-Phase Insert Pattern
    ///
    /// When async updater is enabled:
    /// - **Phase 1 (sync, <5ms):** Store vector data + metadata, add to pending queue
    /// - **Phase 2 (async, background):** Workers build HNSW graph edges
    ///
    /// Vectors are immediately searchable via brute-force fallback on the pending
    /// queue, then transition to HNSW search once graph edges are built.
    ///
    /// ## Garbage Collection
    ///
    /// When GC is enabled, a background worker periodically scans for soft-deleted
    /// vectors and cleans them up (removes edges, deletes data, optionally recycles IDs).
    ///
    /// # Arguments
    ///
    /// * `storage` - Vector storage instance
    /// * `writer_config` - Configuration for mutation writer
    /// * `reader_config` - Configuration for query reader
    /// * `num_query_workers` - Number of parallel query workers
    /// * `async_config` - Optional async updater config; None disables async updates
    /// * `gc_config` - Optional GC config; None disables garbage collection
    ///
    /// # Returns
    ///
    /// Tuple of (Writer, SearchReader) for sending mutations and queries.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use motlie_db::vector::{Subsystem, WriterConfig, ReaderConfig, AsyncUpdaterConfig, GcConfig};
    ///
    /// // Start with async graph updater and garbage collector
    /// let async_config = AsyncUpdaterConfig::default()
    ///     .with_num_workers(4)
    ///     .with_batch_size(200);
    ///
    /// let gc_config = GcConfig::default()
    ///     .with_interval(Duration::from_secs(60))
    ///     .with_batch_size(100);
    ///
    /// let (writer, search_reader) = subsystem.start_with_async(
    ///     storage.clone(),
    ///     WriterConfig::default(),
    ///     ReaderConfig::default(),
    ///     4,
    ///     Some(async_config),
    ///     Some(gc_config),
    /// );
    ///
    /// // Inserts now use two-phase pattern (fast sync + async graph build)
    /// InsertVector::new(&embedding, id, vec)
    ///     .build_index(false)  // Use async path
    ///     .run(&writer).await?;
    /// ```
    ///
    /// # Lifecycle
    ///
    /// When `on_shutdown()` is called:
    /// 1. GC is shut down (stops background scans)
    /// 2. Writer flushes pending mutations (may add to async pending queue)
    /// 3. Async graph updater is shut down (drains pending queue, builds remaining edges)
    /// 4. Consumer tasks are joined (cooperative shutdown via channel close)
    ///
    /// This ensures all components shut down cleanly before storage closes.
    pub fn start_with_async(
        &self,
        storage: Arc<super::Storage>,
        writer_config: WriterConfig,
        reader_config: ReaderConfig,
        num_query_workers: usize,
        async_config: Option<AsyncUpdaterConfig>,
        gc_config: Option<GcConfig>,
    ) -> (Writer, SearchReader) {
        // Create processor (shared by mutation consumer and query consumers)
        // Processor uses the shared nav_cache for HNSW operations
        let processor = Arc::new(Processor::new_with_nav_cache(
            storage.clone(),
            self.cache.clone(),
            self.nav_cache.clone(),
        ));
        if let Some(ref config) = async_config {
            processor.set_async_backpressure_threshold(config.backpressure_threshold);
        }

        // Create and spawn mutation consumer
        let (writer, mutation_receiver) = create_writer(writer_config.clone());
        let mutation_consumer = Consumer::new(mutation_receiver, writer_config, processor.clone());
        let mutation_handle = spawn_consumer(mutation_consumer);

        // Create and spawn query consumers
        let (search_reader, query_receiver) = create_search_reader(reader_config.clone(), processor.clone());
        let query_handles = spawn_consumers_with_processor(
            query_receiver,
            reader_config,
            processor,
            num_query_workers,
        );

        // Store consumer handles for graceful shutdown
        {
            let mut handles = self.consumer_handles.write().expect("consumer_handles lock poisoned");
            handles.push(mutation_handle);
            handles.extend(query_handles);
        }

        // Register writer for shutdown flush
        self.set_writer(writer.clone());

        // Start async graph updater if configured
        if let Some(config) = async_config {
            let updater = AsyncGraphUpdater::start(
                storage.clone(),
                self.cache.clone(),
                self.nav_cache.clone(),
                config,
            );
            *self.async_updater.write().expect("async_updater lock poisoned") = Some(updater);
        }

        // Start garbage collector if configured
        if let Some(config) = gc_config {
            let gc = GarbageCollector::start(
                storage,
                self.cache.clone(),
                config,
            );
            *self.gc.write().expect("gc lock poisoned") = Some(gc);
        }

        (writer, search_reader)
    }

    /// Get reference to the navigation cache.
    ///
    /// The navigation cache is shared between the Processor and AsyncGraphUpdater
    /// for HNSW operations.
    pub fn nav_cache(&self) -> &Arc<NavigationCache> {
        &self.nav_cache
    }

    /// Check if backpressure should be applied to async inserts.
    ///
    /// Returns true if the async graph updater's pending queue exceeds
    /// the configured backpressure threshold. When true, callers should
    /// either wait before sending more async inserts or switch to sync mode.
    ///
    /// Returns false if no async updater is configured or backpressure is disabled.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Check before sending async insert
    /// if subsystem.should_apply_backpressure() {
    ///     // Option 1: Wait and retry
    ///     tokio::time::sleep(Duration::from_millis(100)).await;
    ///
    ///     // Option 2: Switch to sync mode
    ///     InsertVector::new(&embedding, id, vec)
    ///         .immediate()  // Use sync path instead
    ///         .run(&writer).await?;
    /// }
    /// ```
    pub fn should_apply_backpressure(&self) -> bool {
        self.async_updater
            .read()
            .expect("async_updater lock poisoned")
            .as_ref()
            .map(|u| u.should_apply_backpressure())
            .unwrap_or(false)
    }

    /// Get the current pending queue size for async graph updates.
    ///
    /// Returns the number of vectors waiting for HNSW graph construction.
    /// Returns 0 if no async updater is configured.
    ///
    /// Note: This scans the pending CF and may be slow for large queues.
    /// For high-frequency monitoring, prefer checking `should_apply_backpressure()`.
    pub fn pending_queue_size(&self) -> usize {
        self.async_updater
            .read()
            .expect("async_updater lock poisoned")
            .as_ref()
            .map(|u| u.pending_queue_size())
            .unwrap_or(0)
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
            ColumnFamilyDescriptor::new(
                <schema::LifecycleCounts as ColumnFamily>::CF_NAME,
                <schema::LifecycleCounts as ColumnFamilyConfig<VectorBlockCacheConfig>>::cf_options(block_cache, &vector_config),
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

        // 1. Shutdown GC first (stops background scans immediately)
        if let Some(gc) = self.gc.write().expect("gc lock poisoned").take() {
            tracing::debug!(subsystem = "vector", "Shutting down garbage collector");
            gc.shutdown();
            tracing::debug!(subsystem = "vector", "Garbage collector shut down");
        }

        // 2. Flush pending mutations (closes channel, signals consumers to exit)
        // This must happen BEFORE AsyncUpdater shutdown so that any mutations using
        // the async path (build_index=false) are processed and added to pending queue.
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

        // 3. Shutdown async graph updater (graceful - waits for in-flight batches)
        // Now that all mutations are flushed, AsyncUpdater can drain the pending queue.
        if let Some(updater) = self.async_updater.write().expect("async_updater lock poisoned").take() {
            tracing::debug!(subsystem = "vector", "Shutting down async graph updater");
            updater.shutdown();
            tracing::debug!(subsystem = "vector", "Async graph updater shut down");
        }

        // 4. Join consumer tasks (cooperative shutdown - channels are closed)
        // Consumers exit naturally when recv() returns None, so joins should complete quickly.
        let handles = std::mem::take(&mut *self.consumer_handles.write().expect("consumer_handles lock poisoned"));
        if !handles.is_empty() {
            tracing::debug!(subsystem = "vector", count = handles.len(), "Joining consumer tasks");
            match tokio::runtime::Handle::try_current() {
                Ok(runtime_handle) => {
                    for (i, handle) in handles.into_iter().enumerate() {
                        match runtime_handle.block_on(handle) {
                            Ok(Ok(())) => tracing::debug!(subsystem = "vector", consumer = i, "Consumer joined"),
                            Ok(Err(e)) => tracing::warn!(subsystem = "vector", consumer = i, error = %e, "Consumer returned error"),
                            Err(_) => tracing::warn!(subsystem = "vector", consumer = i, "Consumer task panicked"),
                        }
                    }
                }
                Err(_) => {
                    tracing::warn!(
                        subsystem = "vector",
                        "No tokio runtime available for joining consumer tasks"
                    );
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
