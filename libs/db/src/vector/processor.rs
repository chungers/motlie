//! Vector processor - central state management for vector operations.
//!
//! The Processor is the central component that mutation executors and query
//! executors use to access the database and manage per-embedding state.
//!
//! # Design
//!
//! The Processor follows the same pattern as `graph::mutation::Processor`:
//! - Holds shared storage and registry references
//! - Manages per-embedding ID allocators (lazily created)
//! - Manages per-embedding HNSW indices (lazily created)
//! - Provides helper methods for common operations
//!
//! # Thread Safety
//!
//! The Processor is designed for concurrent access:
//! - Storage is Arc-wrapped for shared ownership
//! - Registry is Arc-wrapped for shared lookups
//! - IdAllocators use DashMap for lock-free concurrent access
//! - HNSW indices use DashMap for lock-free concurrent access

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result};
use dashmap::DashMap;

use super::cache::{BinaryCodeCache, NavigationCache};
use super::config::RaBitQConfig;
use super::embedding::Embedding;
use super::hnsw;
use super::id::IdAllocator;
use super::rabitq::RaBitQ;
use super::registry::EmbeddingRegistry;
use super::schema::{
    EmbeddingCode, EmbeddingSpec, EmbeddingSpecCfKey, EmbeddingSpecs, ExternalKey, GraphMeta,
    GraphMetaCfKey, GraphMetaField, IdForward, IdForwardCfKey, IdReverse, IdReverseCfKey, Pending,
    VecId, VecMeta, VecMetaCfKey, VectorCfKey, Vectors,
};
// These types are only used in tests
#[cfg(test)]
use super::schema::GraphMetaCfValue;
use super::search::{SearchConfig, SearchStrategy};
use super::Storage;
use crate::rocksdb::{ColumnFamily, ColumnFamilySerde, HotColumnFamilyRecord};
use crate::Id;

// ============================================================================
// SearchResult
// ============================================================================

/// Search result containing external key, internal ID, and distance.
///
/// Returned by `Processor::search()` for each matched vector.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Typed external key for the vector (node, edge, fragment, summary, etc.)
    pub external_key: ExternalKey,
    /// Internal vector ID (compact u32)
    pub vec_id: VecId,
    /// Distance from query vector (lower = more similar)
    pub distance: f32,
}

impl SearchResult {
    /// Convenience accessor for NodeId external keys.
    ///
    /// Returns `Some(id)` if the external key is `ExternalKey::NodeId`,
    /// otherwise returns `None`.
    pub fn node_id(&self) -> Option<Id> {
        match &self.external_key {
            ExternalKey::NodeId(id) => Some(*id),
            _ => None,
        }
    }
}

// ============================================================================
// Processor
// ============================================================================

/// Vector processor - provides storage access and state management.
///
/// This is the central component that mutation executors and query executors
/// use to access the database and manage per-embedding state.
///
/// # Example
///
/// ```rust,ignore
/// use std::sync::Arc;
/// use motlie_db::vector::{Storage, Processor, EmbeddingRegistry};
///
/// let storage = Arc::new(Storage::readwrite(path)?);
/// let registry = storage.cache().clone();
/// let processor = Processor::new(storage, registry);
///
/// // Get or create an ID allocator for an embedding space
/// let allocator = processor.get_or_create_allocator(embedding_code);
/// let vec_id = allocator.allocate_local();
/// ```
pub struct Processor {
    /// Vector storage (RocksDB via generic Storage<Subsystem>)
    storage: Arc<Storage>,
    /// Embedding registry (pre-warmed on startup)
    registry: Arc<EmbeddingRegistry>,
    /// Per-embedding ID allocators (lazily created)
    id_allocators: DashMap<EmbeddingCode, IdAllocator>,
    /// Per-embedding RaBitQ encoders (lazily created)
    rabitq_encoders: DashMap<EmbeddingCode, Arc<RaBitQ>>,
    /// RaBitQ configuration (shared across all embeddings)
    rabitq_config: RaBitQConfig,
    /// Per-embedding HNSW indices (lazily created)
    hnsw_indices: DashMap<EmbeddingCode, hnsw::Index>,
    /// Shared navigation cache for all HNSW indices
    nav_cache: Arc<NavigationCache>,
    /// Batch threshold for HNSW neighbor fetching.
    /// Performance knob - can vary per process without affecting index integrity.
    batch_threshold: usize,
    /// Shared binary code cache for RaBitQ search
    code_cache: Arc<BinaryCodeCache>,
    /// Backpressure threshold for async inserts (0 disables).
    async_backpressure_threshold: AtomicUsize,
    /// Skip HNSW operations for testing (test-only flag).
    #[cfg(test)]
    skip_hnsw_for_testing: bool,
}

/// Default batch threshold for HNSW neighbor fetching.
const DEFAULT_BATCH_THRESHOLD: usize = 64;

impl Processor {
    /// Create a new Processor with the given storage and registry.
    ///
    /// HNSW structural parameters (m, m_max, ef_construction) are derived from
    /// `EmbeddingSpec` - the single source of truth. Only runtime knobs like
    /// `batch_threshold` can vary per process.
    pub fn new(storage: Arc<Storage>, registry: Arc<EmbeddingRegistry>) -> Self {
        Self::with_batch_threshold(storage, registry, DEFAULT_BATCH_THRESHOLD)
    }

    /// Create a Processor with custom batch threshold for HNSW neighbor fetching.
    ///
    /// The batch threshold controls when neighbor fetching switches from
    /// individual lookups to batched operations. Higher values improve
    /// throughput but increase memory usage.
    pub fn with_batch_threshold(
        storage: Arc<Storage>,
        registry: Arc<EmbeddingRegistry>,
        batch_threshold: usize,
    ) -> Self {
        Self::with_rabitq_config_and_batch_threshold(
            storage,
            registry,
            RaBitQConfig::default(),
            batch_threshold,
            Arc::new(NavigationCache::new()),
        )
    }

    /// Create a new Processor with a shared navigation cache.
    ///
    /// Use this when the navigation cache needs to be shared with other components
    /// (e.g., AsyncGraphUpdater for two-phase inserts).
    pub fn new_with_nav_cache(
        storage: Arc<Storage>,
        registry: Arc<EmbeddingRegistry>,
        nav_cache: Arc<NavigationCache>,
    ) -> Self {
        Self::with_rabitq_config_and_batch_threshold(
            storage,
            registry,
            RaBitQConfig::default(),
            DEFAULT_BATCH_THRESHOLD,
            nav_cache,
        )
    }

    /// Create a Processor with custom RaBitQ configuration.
    pub fn with_rabitq_config(
        storage: Arc<Storage>,
        registry: Arc<EmbeddingRegistry>,
        rabitq_config: RaBitQConfig,
    ) -> Self {
        Self::with_rabitq_config_and_batch_threshold(
            storage,
            registry,
            rabitq_config,
            DEFAULT_BATCH_THRESHOLD,
            Arc::new(NavigationCache::new()),
        )
    }

    /// Create a Processor with custom RaBitQ and HNSW configurations.
    #[deprecated(
        since = "0.2.0",
        note = "Use with_batch_threshold() - structural params derived from EmbeddingSpec"
    )]
    pub fn with_config(
        storage: Arc<Storage>,
        registry: Arc<EmbeddingRegistry>,
        rabitq_config: RaBitQConfig,
        hnsw_config: hnsw::Config,
    ) -> Self {
        Self::with_rabitq_config_and_batch_threshold(
            storage,
            registry,
            rabitq_config,
            hnsw_config.batch_threshold,
            Arc::new(NavigationCache::new()),
        )
    }

    /// Create a Processor with custom configurations and shared navigation cache.
    #[deprecated(
        since = "0.2.0",
        note = "Use with_rabitq_config_and_batch_threshold() - structural params derived from EmbeddingSpec"
    )]
    pub fn with_config_and_nav_cache(
        storage: Arc<Storage>,
        registry: Arc<EmbeddingRegistry>,
        rabitq_config: RaBitQConfig,
        hnsw_config: hnsw::Config,
        nav_cache: Arc<NavigationCache>,
    ) -> Self {
        Self::with_rabitq_config_and_batch_threshold(
            storage,
            registry,
            rabitq_config,
            hnsw_config.batch_threshold,
            nav_cache,
        )
    }

    /// Create a Processor with full configuration options.
    ///
    /// This is the canonical constructor that all other constructors delegate to.
    pub fn with_rabitq_config_and_batch_threshold(
        storage: Arc<Storage>,
        registry: Arc<EmbeddingRegistry>,
        rabitq_config: RaBitQConfig,
        batch_threshold: usize,
        nav_cache: Arc<NavigationCache>,
    ) -> Self {
        Self {
            storage,
            registry,
            id_allocators: DashMap::new(),
            rabitq_encoders: DashMap::new(),
            rabitq_config,
            hnsw_indices: DashMap::new(),
            nav_cache,
            batch_threshold,
            code_cache: Arc::new(BinaryCodeCache::new()),
            async_backpressure_threshold: AtomicUsize::new(0),
            #[cfg(test)]
            skip_hnsw_for_testing: false,
        }
    }

    /// Create processor that skips HNSW operations (test-only).
    ///
    /// Use this to test non-HNSW code paths in isolation.
    #[cfg(test)]
    pub fn new_without_hnsw_for_testing(
        storage: Arc<Storage>,
        registry: Arc<EmbeddingRegistry>,
    ) -> Self {
        Self {
            storage,
            registry,
            id_allocators: DashMap::new(),
            rabitq_encoders: DashMap::new(),
            rabitq_config: RaBitQConfig::default(),
            hnsw_indices: DashMap::new(),
            nav_cache: Arc::new(NavigationCache::new()),
            batch_threshold: DEFAULT_BATCH_THRESHOLD,
            code_cache: Arc::new(BinaryCodeCache::new()),
            async_backpressure_threshold: AtomicUsize::new(0),
            skip_hnsw_for_testing: true,
        }
    }

    /// Set async insert backpressure threshold.
    ///
    /// When non-zero, async inserts (build_index=false) will error if the
    /// pending queue size is at or above this threshold.
    pub fn set_async_backpressure_threshold(&self, threshold: usize) {
        self.async_backpressure_threshold
            .store(threshold, Ordering::Relaxed);
    }

    /// Get current async insert backpressure threshold.
    pub fn async_backpressure_threshold(&self) -> usize {
        self.async_backpressure_threshold.load(Ordering::Relaxed)
    }

    /// Count pending queue items across all embeddings.
    pub fn pending_queue_size(&self) -> usize {
        let txn_db = match self.storage.transaction_db() {
            Ok(db) => db,
            Err(_) => return 0,
        };

        let pending_cf = match txn_db.cf_handle(Pending::CF_NAME) {
            Some(cf) => cf,
            None => return 0,
        };

        txn_db
            .iterator_cf(&pending_cf, rocksdb::IteratorMode::Start)
            .count()
    }

    /// Get access to the underlying storage.
    ///
    /// Query types use this to execute themselves via QueryExecutor::execute()
    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    /// Get access to the Arc-wrapped storage.
    pub fn storage_arc(&self) -> &Arc<Storage> {
        &self.storage
    }

    /// Get access to the embedding registry.
    pub fn registry(&self) -> &EmbeddingRegistry {
        &self.registry
    }

    /// Get access to the Arc-wrapped registry.
    pub fn registry_arc(&self) -> &Arc<EmbeddingRegistry> {
        &self.registry
    }

    /// Get or create an ID allocator for the given embedding space.
    ///
    /// If an allocator doesn't exist for this embedding, it will:
    /// 1. Try to recover state from storage
    /// 2. Fall back to creating a new allocator
    ///
    /// The returned reference is a DashMap RefMut, allowing allocation.
    pub fn get_or_create_allocator(
        &self,
        embedding: EmbeddingCode,
    ) -> dashmap::mapref::one::RefMut<'_, EmbeddingCode, IdAllocator> {
        self.id_allocators.entry(embedding).or_insert_with(|| {
            // Try to recover from storage, or create new
            match self.storage.transaction_db() {
                Ok(db) => IdAllocator::recover(&db, embedding).unwrap_or_default(),
                Err(_) => IdAllocator::new(),
            }
        })
    }

    /// Check if an allocator exists for the given embedding space.
    pub fn has_allocator(&self, embedding: EmbeddingCode) -> bool {
        self.id_allocators.contains_key(&embedding)
    }

    /// Get the number of registered embedding spaces with allocators.
    pub fn allocator_count(&self) -> usize {
        self.id_allocators.len()
    }

    // ========================================================================
    // EmbeddingSpec Lookup (for build params)
    // ========================================================================

    /// Read EmbeddingSpec from RocksDB storage.
    ///
    /// Used to retrieve persisted build parameters (hnsw_m, hnsw_ef_construction,
    /// rabitq_bits, rabitq_seed) for building HNSW indices and RaBitQ encoders.
    fn get_embedding_spec(&self, embedding: EmbeddingCode) -> Option<EmbeddingSpec> {
        let txn_db = self.storage.transaction_db().ok()?;
        let specs_cf = txn_db.cf_handle(EmbeddingSpecs::CF_NAME)?;
        let spec_key = EmbeddingSpecCfKey(embedding);
        let spec_bytes = txn_db
            .get_cf(&specs_cf, EmbeddingSpecs::key_to_bytes(&spec_key))
            .ok()??;
        let spec = EmbeddingSpecs::value_from_bytes(&spec_bytes).ok()?.0;
        Some(spec)
    }

    // ========================================================================
    // RaBitQ Encoder Management
    // ========================================================================

    /// Get or create a RaBitQ encoder for the given embedding space.
    ///
    /// Returns None if:
    /// - RaBitQ is disabled in config
    /// - Embedding is not found in registry (can't determine dimension)
    /// - EmbeddingSpec not found in storage (can't determine build params)
    ///
    /// **IMPORTANT:** Uses persisted EmbeddingSpec build parameters (rabitq_bits, rabitq_seed)
    /// rather than self.rabitq_config. This ensures consistency with stored SpecHash.
    ///
    /// Encoders are cached per embedding space for efficiency.
    pub fn get_or_create_encoder(&self, embedding: EmbeddingCode) -> Option<Arc<RaBitQ>> {
        if !self.rabitq_config.enabled {
            return None;
        }

        // Check cache first
        if let Some(encoder) = self.rabitq_encoders.get(&embedding) {
            return Some(encoder.clone());
        }

        // Look up embedding to get dimension
        let emb = self.registry.get_by_code(embedding)?;
        let dim = emb.dim() as usize;

        // Read persisted EmbeddingSpec to get build params
        let spec = self.get_embedding_spec(embedding)?;

        // Create encoder using EmbeddingSpec params (not self.rabitq_config)
        // This ensures the encoder matches the stored SpecHash
        let encoder = Arc::new(RaBitQ::new(dim, spec.rabitq_bits, spec.rabitq_seed));
        self.rabitq_encoders.insert(embedding, encoder.clone());
        Some(encoder)
    }

    /// Check if RaBitQ is enabled.
    pub fn rabitq_enabled(&self) -> bool {
        self.rabitq_config.enabled
    }

    /// Get the RaBitQ configuration.
    pub fn rabitq_config(&self) -> &RaBitQConfig {
        &self.rabitq_config
    }

    /// Get the number of cached RaBitQ encoders.
    pub fn encoder_count(&self) -> usize {
        self.rabitq_encoders.len()
    }

    // ========================================================================
    // HNSW Index Management
    // ========================================================================

    /// Get or create an HNSW index for the given embedding space.
    ///
    /// Returns None if:
    /// - Embedding is not found in registry
    /// - EmbeddingSpec not found in storage (can't determine build params)
    ///
    /// **IMPORTANT:** Uses persisted EmbeddingSpec as the single source of truth.
    /// HNSW structural parameters (m, m_max, ef_construction, etc.) are derived
    /// from the spec, ensuring consistency with stored SpecHash.
    ///
    /// Indices are cached per embedding space for efficiency.
    pub fn get_or_create_index(&self, embedding: EmbeddingCode) -> Option<hnsw::Index> {
        // Check cache first
        if let Some(index) = self.hnsw_indices.get(&embedding) {
            return Some(index.clone());
        }

        // Read persisted EmbeddingSpec - single source of truth for HNSW params
        let spec = self.get_embedding_spec(embedding)?;

        // Create index using EmbeddingSpec (derives m, m_max, m_l, ef_construction)
        let index = hnsw::Index::from_spec(
            embedding,
            &spec,
            self.batch_threshold,
            Arc::clone(&self.nav_cache),
        );

        self.hnsw_indices.insert(embedding, index.clone());
        Some(index)
    }

    /// Get the shared navigation cache.
    pub fn nav_cache(&self) -> &Arc<NavigationCache> {
        &self.nav_cache
    }

    /// Get the shared binary code cache.
    pub fn code_cache(&self) -> &Arc<BinaryCodeCache> {
        &self.code_cache
    }

    /// Get the HNSW batch threshold.
    ///
    /// This is a runtime performance knob that doesn't affect index structure.
    pub fn batch_threshold(&self) -> usize {
        self.batch_threshold
    }

    /// Get the number of cached HNSW indices.
    pub fn index_count(&self) -> usize {
        self.hnsw_indices.len()
    }

    // ========================================================================
    // Vector Operations
    // ========================================================================

    /// Insert a single vector into an embedding space.
    ///
    /// This is the primary API for inserting vectors. All operations are
    /// executed within a single transaction for atomicity.
    ///
    /// # Arguments
    /// * `embedding` - The embedding space code
    /// * `external_key` - External key identifying the source entity
    /// * `vector` - The vector data (dimension must match embedding spec)
    /// * `build_index` - Whether to build HNSW graph connections
    ///
    /// # Returns
    /// The internal VecId assigned to this vector.
    ///
    /// # Errors
    /// - Unknown embedding code
    /// - Dimension mismatch
    /// - Storage errors
    ///
    /// # Example
    /// ```rust,ignore
    /// let vec_id = processor.insert_vector(
    ///     embedding_code,
    ///     ExternalKey::NodeId(Id::new()),
    ///     &[0.1, 0.2, 0.3, ...],
    ///     true,  // build HNSW index
    /// )?;
    /// ```
    pub fn insert_vector(
        &self,
        embedding: &Embedding,
        external_key: ExternalKey,
        vector: &[f32],
        build_index: bool,
    ) -> Result<VecId> {
        let embedding_code = embedding.code();

        // Begin transaction
        let txn_db = self
            .storage
            .transaction_db()
            .context("Failed to get transaction DB")?;
        let txn = txn_db.transaction();

        // Delegate to shared ops helper (all validation happens there)
        let result = super::ops::insert::vector(
            &txn,
            &txn_db,
            self,
            embedding_code,
            external_key,
            vector,
            build_index,
        )?;

        // Save vec_id before applying cache updates (which consumes result)
        let vec_id = result.vec_id;

        // Commit transaction
        txn.commit().context("Failed to commit transaction")?;

        // Apply cache updates AFTER successful commit
        result.apply_cache_updates(embedding_code, &self.nav_cache, &self.code_cache);

        Ok(vec_id)
    }

    /// Batch insert multiple vectors into an embedding space.
    ///
    /// More efficient than individual `insert_vector()` calls:
    /// - Single transaction for all vectors (atomic commit)
    /// - Spec hash validated once per batch
    /// - Batched RocksDB writes
    /// - HNSW graph built after all vectors stored (better connectivity)
    ///
    /// # Arguments
    /// * `embedding` - The embedding space code
    /// * `vectors` - Slice of (external_id, vector_data) pairs
    /// * `build_index` - Whether to build HNSW graph connections
    ///
    /// # Returns
    /// A vector of allocated VecIds in the same order as input.
    ///
    /// # Errors
    /// - Unknown embedding code
    /// - Dimension mismatch for any vector
    /// - Duplicate external IDs (within batch or against existing)
    /// - Storage errors
    ///
    /// # Example
    /// ```rust,ignore
    /// let vectors: Vec<(ExternalKey, Vec<f32>)> = data.iter()
    ///     .map(|v| (ExternalKey::NodeId(Id::new()), v.clone()))
    ///     .collect();
    ///
    /// let vec_ids = processor.insert_batch(
    ///     embedding_code,
    ///     &vectors,
    ///     true,  // build HNSW index
    /// )?;
    /// ```
    pub fn insert_batch(
        &self,
        embedding: &Embedding,
        vectors: &[(ExternalKey, Vec<f32>)],
        build_index: bool,
    ) -> Result<Vec<VecId>> {
        // Fast path: empty input
        if vectors.is_empty() {
            return Ok(vec![]);
        }

        let embedding_code = embedding.code();

        // Begin transaction
        let txn_db = self
            .storage
            .transaction_db()
            .context("Failed to get transaction DB")?;
        let txn = txn_db.transaction();

        // Delegate to shared ops helper (all validation happens there)
        let result = super::ops::insert::batch(
            &txn,
            &txn_db,
            self,
            embedding_code,
            vectors,
            build_index,
        )?;

        // Commit transaction
        txn.commit().context("Failed to commit batch transaction")?;

        // Apply cache updates AFTER successful commit
        let vec_ids = result.vec_ids.clone();
        result.apply_cache_updates(embedding_code, &self.nav_cache, &self.code_cache);

        Ok(vec_ids)
    }

    /// Delete a vector from an embedding space.
    ///
    /// This method performs cleanup atomically, with behavior depending on
    /// whether HNSW indexing is enabled:
    ///
    /// **When HNSW is disabled:**
    /// - Deletes ID mappings (forward and reverse)
    /// - Deletes vector data and binary codes
    /// - Returns VecId to free list for reuse
    ///
    /// **When HNSW is enabled (soft delete):**
    /// - Deletes ID mappings only (vector unreachable by external ID)
    /// - Keeps vector data (needed for HNSW distance calculations)
    /// - Does NOT free VecId (prevents graph corruption from reuse)
    /// - HNSW edges remain intact (standard tombstone approach)
    ///
    /// # Arguments
    /// * `embedding` - The embedding space code
    /// * `external_key` - External key identifying the vector to delete
    ///
    /// # Returns
    /// - `Ok(Some(vec_id))` - Vector was deleted, returns the VecId
    /// - `Ok(None)` - Vector did not exist (idempotent)
    /// - `Err(...)` - Storage or transaction error
    ///
    /// # HNSW Considerations
    /// When HNSW is enabled, this performs a "soft delete":
    /// - The vector is unreachable via external ID
    /// - Vector data is kept for HNSW graph traversal
    /// - VecId is NOT reused to prevent edge corruption
    /// - Full cleanup requires index rebuild or compaction (future work)
    ///
    /// # Example
    /// ```rust,ignore
    /// match processor.delete_vector(&embedding, ExternalKey::NodeId(id))? {
    ///     Some(vec_id) => println!("Deleted vector with internal ID {}", vec_id),
    ///     None => println!("Vector did not exist"),
    /// }
    /// ```
    ///
    pub(crate) fn delete_vector(
        &self,
        embedding: &Embedding,
        external_key: ExternalKey,
    ) -> Result<Option<VecId>> {
        let embedding_code = embedding.code();

        // Begin transaction
        let txn_db = self
            .storage
            .transaction_db()
            .context("Failed to get transaction DB")?;
        let txn = txn_db.transaction();

        // Delegate to shared ops helper (soft-delete logic happens there)
        let result = super::ops::delete::vector(&txn, &txn_db, self, embedding_code, external_key)?;

        // Commit transaction
        txn.commit().context("Failed to commit delete transaction")?;

        // Return the vec_id if deletion occurred
        Ok(result.vec_id())
    }

    // =========================================================================
    // ID Mapping Lookup Helpers
    // =========================================================================

    /// Look up internal vec_id for an external key.
    ///
    /// This is a low-level helper that performs a direct IdForward lookup.
    /// Prefer using the query API (`GetInternalId`) for channel-based access.
    ///
    /// # Returns
    /// - `Ok(Some(vec_id))` - External key is mapped to this internal ID
    /// - `Ok(None)` - External key not found (not inserted or deleted)
    /// - `Err(...)` - Storage error
    pub fn vec_id_for_external(
        &self,
        embedding: &Embedding,
        external_key: &ExternalKey,
    ) -> Result<Option<VecId>> {
        let embedding_code = embedding.code();
        let txn_db = self.storage.transaction_db()?;
        let forward_cf = txn_db
            .cf_handle(IdForward::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("IdForward CF not found"))?;

        let forward_key = IdForwardCfKey(embedding_code, external_key.clone());
        let key_bytes = IdForward::key_to_bytes(&forward_key);

        match txn_db.get_cf(&forward_cf, &key_bytes)? {
            Some(bytes) => Ok(Some(IdForward::value_from_bytes(&bytes)?.0)),
            None => Ok(None),
        }
    }

    /// Look up external key for an internal vec_id.
    ///
    /// This is a low-level helper that performs a direct IdReverse lookup.
    /// Prefer using the query API (`GetExternalId`) for channel-based access.
    ///
    /// # Returns
    /// - `Ok(Some(external_key))` - Vec_id maps to this external key
    /// - `Ok(None)` - Vec_id not found (not allocated or recycled)
    /// - `Err(...)` - Storage error
    pub fn external_for_vec_id(
        &self,
        embedding: &Embedding,
        vec_id: VecId,
    ) -> Result<Option<ExternalKey>> {
        let embedding_code = embedding.code();
        let txn_db = self.storage.transaction_db()?;
        let reverse_cf = txn_db
            .cf_handle(IdReverse::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("IdReverse CF not found"))?;

        let reverse_key = IdReverseCfKey(embedding_code, vec_id);
        let key_bytes = IdReverse::key_to_bytes(&reverse_key);

        match txn_db.get_cf(&reverse_cf, &key_bytes)? {
            Some(bytes) => Ok(Some(IdReverse::value_from_bytes(&bytes)?.0)),
            None => Ok(None),
        }
    }

    /// Search for nearest neighbors in an embedding space.
    ///
    /// Performs HNSW approximate nearest neighbor search and returns results
    /// with external IDs. Soft-deleted vectors are automatically filtered out.
    ///
    /// # Arguments
    /// * `embedding` - The embedding space code
    /// * `query` - Query vector (must match embedding dimension)
    /// * `k` - Number of results to return
    /// * `ef_search` - Search beam width (higher = better recall, slower)
    ///
    /// # Returns
    /// Up to k nearest neighbors as `SearchResult` structs, sorted by distance
    /// (ascending). May return fewer than k results if:
    /// - Index has fewer than k vectors
    /// - Some results were soft-deleted
    ///
    /// # Errors
    /// - Unknown embedding code
    /// - Dimension mismatch
    /// - HNSW not enabled
    /// - Storage errors
    ///
    /// # Example
    /// ```rust,ignore
    /// let results = processor.search(embedding_code, &query, 10, 100)?;
    /// for result in results {
    ///     println!("Key: {:?}, distance: {}", result.external_key, result.distance);
    /// }
    /// ```
    pub fn search(
        &self,
        embedding: &Embedding,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<SearchResult>> {
        let embedding_code = embedding.code();

        // 1. Validate dimension (Embedding already validated via registry)
        if query.len() != embedding.dim() as usize {
            return Err(anyhow::anyhow!(
                "Dimension mismatch: expected {}, got {}",
                embedding.dim(),
                query.len()
            ));
        }

        // 2. Check HNSW is enabled (test-only: can be disabled for testing)
        #[cfg(test)]
        if self.skip_hnsw_for_testing {
            return Err(anyhow::anyhow!("HNSW disabled for testing"));
        }

        // 3. Validate SpecHash (drift detection)
        // This ensures the index was built with the current EmbeddingSpec configuration
        {
            let txn_db = self.storage.transaction_db()?;

            // Get current EmbeddingSpec from storage
            let specs_cf = txn_db
                .cf_handle(EmbeddingSpecs::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("EmbeddingSpecs CF not found"))?;
            let spec_key = EmbeddingSpecCfKey(embedding_code);
            let spec_bytes = txn_db
                .get_cf(&specs_cf, EmbeddingSpecs::key_to_bytes(&spec_key))?
                .ok_or_else(|| anyhow::anyhow!("EmbeddingSpec not found for {}", embedding_code))?;
            let embedding_spec = EmbeddingSpecs::value_from_bytes(&spec_bytes)?.0;

            // Compute current hash
            let current_hash = embedding_spec.compute_spec_hash();

            // Check stored hash
            let graph_meta_cf = txn_db
                .cf_handle(GraphMeta::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("GraphMeta CF not found"))?;
            let hash_key = GraphMetaCfKey::spec_hash(embedding_code);
            let hash_key_bytes = GraphMeta::key_to_bytes(&hash_key);

            if let Some(stored_bytes) = txn_db.get_cf(&graph_meta_cf, &hash_key_bytes)? {
                let stored_value = GraphMeta::value_from_bytes(&hash_key, &stored_bytes)?;
                if let GraphMetaField::SpecHash(stored_hash) = stored_value.0 {
                    if stored_hash != current_hash {
                        return Err(anyhow::anyhow!(
                            "EmbeddingSpec changed since index build (hash {} != {}). Rebuild required.",
                            current_hash,
                            stored_hash
                        ));
                    }
                }
            } else {
                // Legacy index without SpecHash - drift check skipped
                // This is expected for indexes built before Phase 5.6
                tracing::warn!(
                    embedding = embedding_code,
                    "Legacy index without SpecHash - drift check skipped"
                );
            }
        }

        // 4. Get or create HNSW index
        let index = self
            .get_or_create_index(embedding_code)
            .ok_or_else(|| anyhow::anyhow!("Failed to get HNSW index for embedding {}", embedding_code))?;

        // 5. Perform HNSW search with overfetch to handle tombstones
        // Overfetch by 2x to ensure we get enough live results after filtering
        // Ensure ef_search >= overfetch_k so HNSW can return enough candidates
        let overfetch_k = k * 2;
        let effective_ef = ef_search.max(overfetch_k);
        let raw_results = index.search(&self.storage, query, overfetch_k, effective_ef)?;

        // 6. Filter deleted vectors using batched IdReverse lookup
        let txn_db = self.storage.transaction_db()?;
        let reverse_cf = txn_db
            .cf_handle(IdReverse::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("IdReverse CF not found"))?;

        // Build batch of keys for multi_get
        let keys: Vec<_> = raw_results
            .iter()
            .map(|(_, vec_id)| {
                let key = IdReverseCfKey(embedding_code, *vec_id);
                IdReverse::key_to_bytes(&key)
            })
            .collect();

        // Batch lookup all IdReverse keys at once
        let key_refs: Vec<_> = keys.iter().map(|k: &Vec<u8>| (&reverse_cf, k.as_slice())).collect();
        let values = txn_db.multi_get_cf(key_refs);

        // Filter and resolve external IDs, truncate to k
        let mut results = Vec::with_capacity(k);
        for (i, value_result) in values.into_iter().enumerate() {
            if results.len() >= k {
                break;
            }
            if let Ok(Some(bytes)) = value_result {
                let external_key = IdReverse::value_from_bytes(&bytes)?.0;
                let (distance, vec_id) = raw_results[i];
                results.push(SearchResult {
                    external_key,
                    vec_id,
                    distance,
                });
            }
            // Skip deleted vectors (IdReverse missing or error)
        }

        Ok(results)
    }

    /// Search using a SearchConfig for strategy-based dispatch.
    ///
    /// This method dispatches to different search strategies based on the
    /// configuration:
    /// - `SearchStrategy::Exact` - Standard HNSW search with exact distance
    /// - `SearchStrategy::RaBitQ` - Two-phase search with Hamming pre-filter
    ///
    /// # Arguments
    /// * `config` - Search configuration with embedding, k, ef, and strategy
    /// * `query` - Query vector (must match embedding dimension)
    ///
    /// # Returns
    /// Up to k nearest neighbors as `SearchResult` structs, sorted by distance.
    ///
    /// # Example
    /// ```rust,ignore
    /// let config = SearchConfig::new(embedding.clone(), 10)
    ///     .with_ef(150)
    ///     .with_rerank_factor(4);
    /// let results = processor.search_with_config(&config, &query)?;
    /// ```
    pub fn search_with_config(
        &self,
        config: &SearchConfig,
        query: &[f32],
    ) -> Result<Vec<SearchResult>> {
        let embedding_code = config.embedding().code();

        // 1. Validate embedding exists
        let spec = self
            .registry
            .get_by_code(embedding_code)
            .ok_or_else(|| anyhow::anyhow!("Unknown embedding code: {}", embedding_code))?;

        // 2. Validate SearchConfig embedding matches registry spec
        // This prevents stale or forged configs from being used
        let config_embedding = config.embedding();
        if config_embedding.dim() != spec.dim() {
            return Err(anyhow::anyhow!(
                "SearchConfig embedding dimension mismatch: config has {}, registry has {}",
                config_embedding.dim(),
                spec.dim()
            ));
        }
        if config_embedding.distance() != spec.distance() {
            return Err(anyhow::anyhow!(
                "SearchConfig embedding distance mismatch: config has {:?}, registry has {:?}",
                config_embedding.distance(),
                spec.distance()
            ));
        }
        // 2a. Validate storage_type (redundant with SpecHash, but explicit for clarity)
        if config_embedding.storage_type() != spec.storage_type() {
            return Err(anyhow::anyhow!(
                "SearchConfig embedding storage_type mismatch: config has {:?}, registry has {:?}",
                config_embedding.storage_type(),
                spec.storage_type()
            ));
        }

        // 2b. Validate registry spec hash vs stored GraphMeta::SpecHash (drift detection)
        // This ensures the index was built with the current EmbeddingSpec configuration
        {
            use super::schema::{EmbeddingSpecCfKey, EmbeddingSpecs};
            use crate::rocksdb::ColumnFamilySerde;

            let txn_db = self.storage.transaction_db()?;

            // Get current EmbeddingSpec from storage
            let specs_cf = txn_db
                .cf_handle(EmbeddingSpecs::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("EmbeddingSpecs CF not found"))?;
            let spec_key = EmbeddingSpecCfKey(embedding_code);
            let spec_bytes = txn_db
                .get_cf(&specs_cf, EmbeddingSpecs::key_to_bytes(&spec_key))?
                .ok_or_else(|| anyhow::anyhow!("EmbeddingSpec not found for {}", embedding_code))?;
            let embedding_spec = EmbeddingSpecs::value_from_bytes(&spec_bytes)?.0;

            // Compute current hash
            let current_hash = embedding_spec.compute_spec_hash();

            // Check stored hash
            let graph_meta_cf = txn_db
                .cf_handle(GraphMeta::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("GraphMeta CF not found"))?;
            let hash_key = GraphMetaCfKey::spec_hash(embedding_code);
            let hash_key_bytes = GraphMeta::key_to_bytes(&hash_key);

            if let Some(stored_bytes) = txn_db.get_cf(&graph_meta_cf, &hash_key_bytes)? {
                let stored_value = GraphMeta::value_from_bytes(&hash_key, &stored_bytes)?;
                if let GraphMetaField::SpecHash(stored_hash) = stored_value.0 {
                    if stored_hash != current_hash {
                        return Err(anyhow::anyhow!(
                            "EmbeddingSpec changed since index build (hash {} != {}). Rebuild required.",
                            current_hash,
                            stored_hash
                        ));
                    }
                }
            } else {
                // Legacy index without SpecHash - drift check skipped
                // This is expected for indexes built before Phase 5.6
                tracing::warn!(
                    embedding = embedding_code,
                    "Legacy index without SpecHash - drift check skipped"
                );
            }
        }

        // 3. Validate query dimension
        if query.len() != spec.dim() as usize {
            return Err(anyhow::anyhow!(
                "Query dimension mismatch: expected {}, got {}",
                spec.dim(),
                query.len()
            ));
        }

        // 4. Check HNSW is enabled (test-only: can be disabled for testing)
        #[cfg(test)]
        if self.skip_hnsw_for_testing {
            return Err(anyhow::anyhow!("HNSW disabled for testing"));
        }

        // 5. Get HNSW index
        let index = self
            .get_or_create_index(embedding_code)
            .ok_or_else(|| anyhow::anyhow!("Failed to get HNSW index"))?;

        // 6. Dispatch based on strategy
        let k = config.k();
        let ef = config.ef();
        let overfetch_k = k * 2;
        let effective_ef = ef.max(overfetch_k);

        let raw_results = match config.strategy() {
            SearchStrategy::Exact => {
                // Standard HNSW search with exact distance
                index.search(&self.storage, query, overfetch_k, effective_ef)?
            }
            SearchStrategy::RaBitQ { use_cache } => {
                if use_cache {
                    // Two-phase search with cached binary codes
                    let encoder = self
                        .get_or_create_encoder(embedding_code)
                        .ok_or_else(|| anyhow::anyhow!("RaBitQ encoder not available"))?;
                    index.search_with_rabitq_cached(
                        &self.storage,
                        query,
                        &encoder,
                        &self.code_cache,
                        overfetch_k,
                        effective_ef,
                        config.rerank_factor(),
                    )?
                } else {
                    // RaBitQ without cache - fall back to exact for now
                    // (uncached RaBitQ would need to read from RocksDB, defeating the purpose)
                    index.search(&self.storage, query, overfetch_k, effective_ef)?
                }
            }
        };

        // 7. Filter deleted vectors using batched IdReverse + VecMeta lookup
        // Primary filter: IdReverse (deleted vectors have no reverse mapping)
        // Defense-in-depth: VecMeta lifecycle check (Task 8.1.5)
        let txn_db = self.storage.transaction_db()?;
        let reverse_cf = txn_db
            .cf_handle(IdReverse::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("IdReverse CF not found"))?;
        let meta_cf = txn_db
            .cf_handle(VecMeta::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("VecMeta CF not found"))?;

        // Batch fetch IdReverse
        let reverse_keys: Vec<_> = raw_results
            .iter()
            .map(|(_, vec_id)| {
                let key = IdReverseCfKey(embedding_code, *vec_id);
                IdReverse::key_to_bytes(&key)
            })
            .collect();

        let reverse_refs: Vec<_> = reverse_keys
            .iter()
            .map(|k: &Vec<u8>| (&reverse_cf, k.as_slice()))
            .collect();
        let reverse_values = txn_db.multi_get_cf(reverse_refs);

        // Batch fetch VecMeta for defense-in-depth deleted check
        let meta_keys: Vec<_> = raw_results
            .iter()
            .map(|(_, vec_id)| {
                let key = VecMetaCfKey(embedding_code, *vec_id);
                VecMeta::key_to_bytes(&key)
            })
            .collect();

        let meta_refs: Vec<_> = meta_keys
            .iter()
            .map(|k: &Vec<u8>| (&meta_cf, k.as_slice()))
            .collect();
        let meta_values = txn_db.multi_get_cf(meta_refs);

        // Collect HNSW results (may contain fewer than k due to deleted vectors)
        let mut hnsw_results = Vec::with_capacity(k * 2);
        for (i, value_result) in reverse_values.into_iter().enumerate() {
            if let Ok(Some(bytes)) = value_result {
                // Primary filter passed: IdReverse exists
                // Defense-in-depth: Check VecMeta lifecycle (Task 8.1.5)
                if let Ok(Some(meta_bytes)) = &meta_values[i] {
                    if let Ok(meta) = VecMeta::value_from_bytes(meta_bytes) {
                        if meta.0.is_deleted() {
                            // VecMeta indicates deleted - skip (defense-in-depth)
                            // This catches edge cases where IdReverse wasn't cleaned up
                            continue;
                        }
                    }
                }

                let external_key = IdReverse::value_from_bytes(&bytes)?.0;
                let (distance, vec_id) = raw_results[i];
                hnsw_results.push(SearchResult {
                    external_key,
                    vec_id,
                    distance,
                });
            }
        }

        // 8. Scan pending vectors if fallback is enabled
        if config.has_pending_fallback() {
            let pending_results = self.scan_pending_vectors(
                config.embedding(),
                query,
                config.pending_scan_limit(),
            )?;

            if !pending_results.is_empty() {
                // Track vec_ids already in HNSW results to avoid duplicates
                // (a vector could theoretically be in both if there's a race condition)
                let hnsw_vec_ids: std::collections::HashSet<VecId> =
                    hnsw_results.iter().map(|r| r.vec_id).collect();

                // Add pending results that aren't duplicates
                for (distance, vec_id, external_key) in pending_results {
                    if !hnsw_vec_ids.contains(&vec_id) {
                        hnsw_results.push(SearchResult {
                            external_key,
                            vec_id,
                            distance,
                        });
                    }
                }

                // Re-sort merged results by distance
                hnsw_results.sort_by(|a, b| {
                    a.distance
                        .partial_cmp(&b.distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        // Truncate to requested k
        hnsw_results.truncate(k);
        Ok(hnsw_results)
    }

    /// Scan pending vectors and compute brute-force distances.
    ///
    /// This function iterates the Pending CF for the given embedding and computes
    /// exact distances for up to `limit` pending vectors. These results can then
    /// be merged with HNSW search results to ensure immediate searchability of
    /// newly inserted vectors.
    ///
    /// # Arguments
    /// * `embedding` - The embedding specification (for distance computation)
    /// * `query` - Query vector
    /// * `limit` - Maximum number of pending vectors to scan
    ///
    /// # Returns
    /// Vector of (distance, vec_id, external_key) tuples, sorted by distance ascending.
    fn scan_pending_vectors(
        &self,
        embedding: &Embedding,
        query: &[f32],
        limit: usize,
    ) -> Result<Vec<(f32, VecId, ExternalKey)>> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let embedding_code = embedding.code();
        let txn_db = self.storage.transaction_db()?;

        // Get CF handles
        let pending_cf = txn_db
            .cf_handle(Pending::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Pending CF not found"))?;
        let vectors_cf = txn_db
            .cf_handle(Vectors::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;
        let meta_cf = txn_db
            .cf_handle(VecMeta::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("VecMeta CF not found"))?;
        let reverse_cf = txn_db
            .cf_handle(IdReverse::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("IdReverse CF not found"))?;

        // Get storage type for proper vector deserialization (f32 vs f16)
        let storage_type = embedding.storage_type();

        // Iterate pending entries for this embedding
        let prefix = Pending::prefix_for_embedding(embedding_code);
        let iter = txn_db.iterator_cf(
            &pending_cf,
            rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
        );

        let mut results = Vec::with_capacity(limit.min(1024));

        for item in iter {
            if results.len() >= limit {
                break;
            }

            let (key, _value) = item?;

            // Check prefix match (stop when we leave this embedding's range)
            if key.len() < 8 || key[0..8] != prefix {
                break;
            }

            // Parse the pending key to get vec_id
            let parsed = Pending::key_from_bytes(&key)?;
            let vec_id = parsed.2;

            // Check VecMeta lifecycle - skip if deleted (PendingDeleted state)
            // This handles the case where a vector was deleted before async indexing completed
            let meta_key = VecMetaCfKey(embedding_code, vec_id);
            if let Some(meta_bytes) = txn_db.get_cf(&meta_cf, VecMeta::key_to_bytes(&meta_key))? {
                let meta = VecMeta::value_from_bytes(&meta_bytes)?.0;
                if meta.is_deleted() {
                    continue; // Skip deleted vectors (PendingDeleted state)
                }
            }

            // Load vector data from Vectors CF using proper storage_type
            let vec_key = VectorCfKey(embedding_code, vec_id);
            let vec_bytes = match txn_db.get_cf(&vectors_cf, Vectors::key_to_bytes(&vec_key))? {
                Some(bytes) => bytes,
                None => continue, // Vector not found, skip
            };
            let vector_data = Vectors::value_from_bytes_typed(&vec_bytes, storage_type)?;

            // Compute exact distance
            let distance = embedding.compute_distance(query, &vector_data);

            // Look up external key via IdReverse
            let reverse_key = IdReverseCfKey(embedding_code, vec_id);
            let external_key = match txn_db.get_cf(&reverse_cf, IdReverse::key_to_bytes(&reverse_key))? {
                Some(bytes) => IdReverse::value_from_bytes(&bytes)?.0,
                None => continue, // No external key mapping (deleted?), skip
            };

            results.push((distance, vec_id, external_key));
        }

        // Sort by distance ascending
        results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full integration tests require a RocksDB instance.
    // These unit tests verify the basic structure and logic.

    #[test]
    fn test_processor_allocator_management() {
        // Create a mock scenario without actual storage
        // This tests the DashMap-based allocator management

        let allocators: DashMap<EmbeddingCode, IdAllocator> = DashMap::new();

        // Simulate get_or_create_allocator
        let embedding: EmbeddingCode = 1;
        allocators.entry(embedding).or_insert_with(IdAllocator::new);

        assert!(allocators.contains_key(&1));
        assert!(!allocators.contains_key(&2));

        // Allocate some IDs
        {
            let mut allocator = allocators.get_mut(&1).unwrap();
            let id1 = allocator.allocate_local();
            let id2 = allocator.allocate_local();
            assert_eq!(id1, 0);
            assert_eq!(id2, 1);
        }

        // Verify state persists
        {
            let allocator = allocators.get(&1).unwrap();
            assert_eq!(allocator.next_id(), 2);
        }
    }

    #[test]
    fn test_allocator_count() {
        let allocators: DashMap<EmbeddingCode, IdAllocator> = DashMap::new();

        assert_eq!(allocators.len(), 0);

        allocators.insert(1, IdAllocator::new());
        assert_eq!(allocators.len(), 1);

        allocators.insert(2, IdAllocator::new());
        assert_eq!(allocators.len(), 2);

        // Inserting same key doesn't increase count
        allocators.insert(1, IdAllocator::new());
        assert_eq!(allocators.len(), 2);
    }

    #[test]
    fn test_insert_vector_integration() {
        use tempfile::TempDir;

        use crate::vector::embedding::EmbeddingBuilder;
        use crate::vector::Distance;

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().expect("Failed to initialize storage");
        let storage = Arc::new(storage);

        // Create registry with an embedding spec
        let registry = Arc::new(EmbeddingRegistry::new());
        let txn_db = storage.transaction_db().expect("Failed to get txn_db");
        let builder = EmbeddingBuilder::new("test-model", 64, Distance::L2);
        let embedding = registry
            .register(builder, &txn_db)
            .expect("Failed to register embedding");
        let embedding_code = embedding.code();

        // Create processor
        let processor = Processor::new(storage.clone(), registry);

        // Generate test vector
        let vector: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let id = crate::Id::new();

        // Insert without building index
        let vec_id = processor
            .insert_vector(&embedding, ExternalKey::NodeId(id), &vector, false)
            .expect("Insert should succeed");

        assert_eq!(vec_id, 0, "First vector should get ID 0");

        // Insert another with building index
        let id2 = crate::Id::new();
        let vector2: Vec<f32> = (0..64).map(|i| (i + 1) as f32 / 64.0).collect();
        let vec_id2 = processor
            .insert_vector(&embedding, ExternalKey::NodeId(id2), &vector2, true)
            .expect("Insert with index should succeed");

        assert_eq!(vec_id2, 1, "Second vector should get ID 1");

        // Verify allocator state
        assert_eq!(processor.allocator_count(), 1);
        {
            let allocator = processor.get_or_create_allocator(embedding_code);
            assert_eq!(allocator.next_id(), 2);
        }
    }

    #[test]
    fn test_insert_vector_dimension_mismatch() {
        use tempfile::TempDir;

        use crate::vector::embedding::EmbeddingBuilder;
        use crate::vector::Distance;

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().expect("Failed to initialize storage");
        let storage = Arc::new(storage);

        // Create registry with 64-dim embedding
        let registry = Arc::new(EmbeddingRegistry::new());
        let txn_db = storage.transaction_db().expect("Failed to get txn_db");
        let builder = EmbeddingBuilder::new("test-model", 64, Distance::L2);
        let embedding = registry
            .register(builder, &txn_db)
            .expect("Failed to register");
        let embedding_code = embedding.code();

        let processor = Processor::new(storage, registry);

        // Try to insert 128-dim vector into 64-dim embedding
        let wrong_dim_vector: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let result =
            processor.insert_vector(&embedding, ExternalKey::NodeId(crate::Id::new()), &wrong_dim_vector, false);

        assert!(result.is_err(), "Should fail on dimension mismatch");
        assert!(
            result.unwrap_err().to_string().contains("Dimension mismatch"),
            "Error should mention dimension mismatch"
        );
    }

    // Note: test_insert_vector_unknown_embedding removed - with &Embedding API,
    // you can't pass an invalid embedding (must register first to get Embedding)

    #[test]
    fn test_insert_vector_duplicate_id() {
        use tempfile::TempDir;

        use crate::vector::embedding::EmbeddingBuilder;
        use crate::vector::Distance;

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().expect("Failed to initialize storage");
        let storage = Arc::new(storage);

        let registry = Arc::new(EmbeddingRegistry::new());
        let txn_db = storage.transaction_db().expect("Failed to get txn_db");
        let builder = EmbeddingBuilder::new("test-model", 64, Distance::L2);
        let embedding = registry
            .register(builder, &txn_db)
            .expect("Failed to register");
        let embedding_code = embedding.code();

        let processor = Processor::new(storage, registry);

        // Insert first vector
        let vector: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let id = crate::Id::new();
        processor
            .insert_vector(&embedding, ExternalKey::NodeId(id), &vector, false)
            .expect("First insert should succeed");

        // Try to insert with same external ID
        let vector2: Vec<f32> = (0..64).map(|i| (i + 1) as f32 / 64.0).collect();
        let result = processor.insert_vector(&embedding, ExternalKey::NodeId(id), &vector2, false);

        assert!(result.is_err(), "Should fail on duplicate ID");
        assert!(
            result.unwrap_err().to_string().contains("already exists"),
            "Error should mention ID already exists"
        );
    }

    #[test]
    fn test_delete_vector_success() {
        use tempfile::TempDir;

        use crate::vector::embedding::EmbeddingBuilder;
        use crate::vector::Distance;

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().expect("Failed to initialize storage");
        let storage = Arc::new(storage);

        let registry = Arc::new(EmbeddingRegistry::new());
        let txn_db = storage.transaction_db().expect("Failed to get txn_db");
        let builder = EmbeddingBuilder::new("test-model", 64, Distance::L2);
        let embedding = registry
            .register(builder, &txn_db)
            .expect("Failed to register");
        let embedding_code = embedding.code();

        let processor = Processor::new(storage.clone(), registry);

        // Insert a vector
        let vector: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let id = crate::Id::new();
        let vec_id = processor
            .insert_vector(&embedding, ExternalKey::NodeId(id), &vector, false)
            .expect("Insert should succeed");

        assert_eq!(vec_id, 0, "First vector should get ID 0");

        // Delete the vector
        let deleted = processor
            .delete_vector(&embedding, ExternalKey::NodeId(id))
            .expect("Delete should succeed");

        assert_eq!(deleted, Some(0), "Should return deleted VecId");

        // Verify IdForward mapping is gone
        let forward_cf = txn_db
            .cf_handle(IdForward::CF_NAME)
            .expect("CF should exist");
        let forward_key = IdForwardCfKey(embedding_code, ExternalKey::NodeId(id));
        let result = txn_db
            .get_cf(&forward_cf, IdForward::key_to_bytes(&forward_key))
            .expect("Read should succeed");
        assert!(result.is_none(), "IdForward mapping should be deleted");
    }

    #[test]
    fn test_delete_vector_not_found() {
        use tempfile::TempDir;

        use crate::vector::embedding::EmbeddingBuilder;
        use crate::vector::Distance;

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().expect("Failed to initialize storage");
        let storage = Arc::new(storage);

        let registry = Arc::new(EmbeddingRegistry::new());
        let txn_db = storage.transaction_db().expect("Failed to get txn_db");
        let builder = EmbeddingBuilder::new("test-model", 64, Distance::L2);
        let embedding = registry
            .register(builder, &txn_db)
            .expect("Failed to register");
        let embedding_code = embedding.code();

        let processor = Processor::new(storage, registry);

        // Try to delete non-existent vector (should be idempotent)
        let id = crate::Id::new();
        let result = processor
            .delete_vector(&embedding, ExternalKey::NodeId(id))
            .expect("Delete should not error");

        assert_eq!(result, None, "Should return None for non-existent vector");
    }

    #[test]
    fn test_delete_vector_soft_delete() {
        use tempfile::TempDir;

        use crate::vector::embedding::EmbeddingBuilder;
        use crate::vector::Distance;

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().expect("Failed to initialize storage");
        let storage = Arc::new(storage);

        let registry = Arc::new(EmbeddingRegistry::new());
        let txn_db = storage.transaction_db().expect("Failed to get txn_db");
        let builder = EmbeddingBuilder::new("test-model", 64, Distance::L2);
        let embedding = registry
            .register(builder, &txn_db)
            .expect("Failed to register");

        // HNSW is always enabled - soft-delete is used, VecIds are NOT reused
        // (to prevent HNSW graph corruption from VecId reuse)
        let processor = Processor::new(storage, registry);

        // Insert two vectors
        let vector: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let id1 = crate::Id::new();
        let id2 = crate::Id::new();

        let vec_id1 = processor
            .insert_vector(&embedding, ExternalKey::NodeId(id1), &vector, false)
            .expect("Insert 1 should succeed");
        let vec_id2 = processor
            .insert_vector(&embedding, ExternalKey::NodeId(id2), &vector, false)
            .expect("Insert 2 should succeed");

        assert_eq!(vec_id1, 0);
        assert_eq!(vec_id2, 1);

        // Delete first vector (soft delete - VecId not freed)
        processor
            .delete_vector(&embedding, ExternalKey::NodeId(id1))
            .expect("Delete should succeed");

        // Insert new vector - should get fresh VecId (no reuse with HNSW)
        let id3 = crate::Id::new();
        let vec_id3 = processor
            .insert_vector(&embedding, ExternalKey::NodeId(id3), &vector, false)
            .expect("Insert 3 should succeed");

        assert_eq!(vec_id3, 2, "Should get fresh VecId (no reuse with HNSW enabled)");

        // Next insert should continue with fresh IDs
        let id4 = crate::Id::new();
        let vec_id4 = processor
            .insert_vector(&embedding, ExternalKey::NodeId(id4), &vector, false)
            .expect("Insert 4 should succeed");

        assert_eq!(vec_id4, 3, "Should get next fresh ID");

        // Verify deleted vector is unreachable by external ID
        // (re-insert with same external key should succeed with new VecId)
        let vec_id1_new = processor
            .insert_vector(&embedding, ExternalKey::NodeId(id1), &vector, false)
            .expect("Re-insert should succeed");

        assert_eq!(vec_id1_new, 4, "Re-insert gets fresh ID, not the deleted one");
    }

    #[test]
    fn test_search_basic() {
        use tempfile::TempDir;

        use crate::vector::embedding::EmbeddingBuilder;
        use crate::vector::Distance;

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().expect("Failed to initialize storage");
        let storage = Arc::new(storage);

        let registry = Arc::new(EmbeddingRegistry::new());
        let txn_db = storage.transaction_db().expect("Failed to get txn_db");
        let builder = EmbeddingBuilder::new("test-model", 64, Distance::L2);
        let embedding = registry
            .register(builder, &txn_db)
            .expect("Failed to register");

        // Create processor (HNSW always enabled, params derived from EmbeddingSpec)
        let processor = Processor::new(storage, registry);

        // Insert several vectors (with index building)
        let mut inserted_ids = Vec::new();
        for i in 0..10 {
            let vector: Vec<f32> = (0..64).map(|j| (i * 64 + j) as f32 / 1000.0).collect();
            let id = crate::Id::new();
            processor
                .insert_vector(&embedding, ExternalKey::NodeId(id), &vector, true)
                .expect("Insert should succeed");
            inserted_ids.push(id);
        }

        // Search for nearest neighbors using first vector as query
        let query: Vec<f32> = (0..64).map(|j| j as f32 / 1000.0).collect();
        let results = processor
            .search(&embedding, &query, 5, 100)
            .expect("Search should succeed");

        // Should return at most k results
        assert!(!results.is_empty(), "Should have results");
        assert!(results.len() <= 5, "Should return at most k results");

        // First result should be closest to query
        let first = &results[0];
        assert_eq!(
            first.node_id().expect("expected NodeId"),
            inserted_ids[0],
            "First inserted vector should be closest"
        );

        // All results should have valid distances
        for (i, result) in results.iter().enumerate() {
            if i > 0 {
                assert!(
                    result.distance >= results[i - 1].distance,
                    "Results should be sorted by distance"
                );
            }
        }
    }

    #[test]
    fn test_search_filters_deleted() {
        use tempfile::TempDir;

        use crate::vector::embedding::EmbeddingBuilder;
        use crate::vector::Distance;

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().expect("Failed to initialize storage");
        let storage = Arc::new(storage);

        let registry = Arc::new(EmbeddingRegistry::new());
        let txn_db = storage.transaction_db().expect("Failed to get txn_db");
        let builder = EmbeddingBuilder::new("test-model", 64, Distance::L2);
        let embedding = registry
            .register(builder, &txn_db)
            .expect("Failed to register");

        // Create processor (HNSW always enabled, params derived from EmbeddingSpec)
        let processor = Processor::new(storage, registry);

        // Insert 5 vectors
        let mut inserted_ids = Vec::new();
        for i in 0..5 {
            let vector: Vec<f32> = (0..64).map(|j| (i * 64 + j) as f32 / 1000.0).collect();
            let id = crate::Id::new();
            processor
                .insert_vector(&embedding, ExternalKey::NodeId(id), &vector, true)
                .expect("Insert should succeed");
            inserted_ids.push(id);
        }

        // Delete first two vectors (soft delete - mappings removed)
        processor
            .delete_vector(&embedding, ExternalKey::NodeId(inserted_ids[0]))
            .expect("Delete should succeed");
        processor
            .delete_vector(&embedding, ExternalKey::NodeId(inserted_ids[1]))
            .expect("Delete should succeed");

        // Search - deleted vectors should be filtered from results
        let query: Vec<f32> = (0..64).map(|j| j as f32 / 1000.0).collect();
        let results = processor
            .search(&embedding, &query, 10, 100)
            .expect("Search should succeed");

        // Should have at most 3 results (5 - 2 deleted)
        assert!(results.len() <= 3, "Should only return non-deleted vectors");

        // Verify deleted IDs are not in results
        let result_ids: Vec<_> = results
            .iter()
            .filter_map(|r| r.node_id())
            .collect();
        assert!(
            !result_ids.contains(&inserted_ids[0]),
            "First deleted ID should not be in results"
        );
        assert!(
            !result_ids.contains(&inserted_ids[1]),
            "Second deleted ID should not be in results"
        );
    }

    #[test]
    fn test_search_dimension_mismatch() {
        use tempfile::TempDir;

        use crate::vector::embedding::EmbeddingBuilder;
        use crate::vector::Distance;

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().expect("Failed to initialize storage");
        let storage = Arc::new(storage);

        let registry = Arc::new(EmbeddingRegistry::new());
        let txn_db = storage.transaction_db().expect("Failed to get txn_db");
        let builder = EmbeddingBuilder::new("test-model", 64, Distance::L2);
        let embedding = registry
            .register(builder, &txn_db)
            .expect("Failed to register");
        let embedding_code = embedding.code();

        let processor = Processor::new(storage, registry);

        // Try to search with wrong dimension
        let wrong_dim_query: Vec<f32> = (0..128).map(|i| i as f32).collect();
        let result = processor.search(&embedding, &wrong_dim_query, 10, 100);

        assert!(result.is_err(), "Should fail on dimension mismatch");
        assert!(
            result.unwrap_err().to_string().contains("Dimension mismatch"),
            "Error should mention dimension mismatch"
        );
    }

    #[test]
    fn test_search_hnsw_disabled() {
        use tempfile::TempDir;

        use crate::vector::embedding::EmbeddingBuilder;
        use crate::vector::Distance;

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().expect("Failed to initialize storage");
        let storage = Arc::new(storage);

        let registry = Arc::new(EmbeddingRegistry::new());
        let txn_db = storage.transaction_db().expect("Failed to get txn_db");
        let builder = EmbeddingBuilder::new("test-model", 64, Distance::L2);
        let embedding = registry
            .register(builder, &txn_db)
            .expect("Failed to register");

        // Create processor with HNSW disabled for testing
        let processor = Processor::new_without_hnsw_for_testing(storage, registry);

        // Try to search when HNSW is disabled
        let query: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let result = processor.search(&embedding, &query, 10, 100);

        assert!(result.is_err(), "Should fail when HNSW disabled");
        assert!(
            result.unwrap_err().to_string().contains("HNSW disabled for testing"),
            "Error should mention HNSW disabled"
        );
    }

    #[test]
    fn test_search_with_config_exact() {
        use tempfile::TempDir;

        use crate::vector::embedding::EmbeddingBuilder;
        use crate::vector::Distance;

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().expect("Failed to initialize storage");
        let storage = Arc::new(storage);

        let registry = Arc::new(EmbeddingRegistry::new());
        let txn_db = storage.transaction_db().expect("Failed to get txn_db");
        // Use L2 distance - will get Exact strategy (not RaBitQ)
        let builder = EmbeddingBuilder::new("test-model", 64, Distance::L2);
        let embedding = registry
            .register(builder, &txn_db)
            .expect("Failed to register");

        // Create processor (HNSW always enabled, params derived from EmbeddingSpec)
        let processor = Processor::new(storage, registry);

        // Insert several vectors
        let mut inserted_ids = Vec::new();
        for i in 0..10 {
            let vector: Vec<f32> = (0..64).map(|j| (i * 64 + j) as f32 / 1000.0).collect();
            let id = crate::Id::new();
            processor
                .insert_vector(&embedding, ExternalKey::NodeId(id), &vector, true)
                .expect("Insert should succeed");
            inserted_ids.push(id);
        }

        // Create SearchConfig with Exact strategy
        let config = SearchConfig::new(embedding.clone(), 5)
            .with_ef(100);

        // Verify strategy is Exact for L2 distance
        assert!(config.strategy().is_exact(), "L2 should use Exact strategy");

        // Search using config
        let query: Vec<f32> = (0..64).map(|j| j as f32 / 1000.0).collect();
        let results = processor
            .search_with_config(&config, &query)
            .expect("Search should succeed");

        // Should return at most k results
        assert!(!results.is_empty(), "Should have results");
        assert!(results.len() <= 5, "Should return at most k results");

        // First result should be closest to query
        let first = &results[0];
        assert_eq!(
            first.node_id().expect("expected NodeId"),
            inserted_ids[0],
            "First inserted vector should be closest"
        );

        // Results should be sorted by distance
        for (i, result) in results.iter().enumerate() {
            if i > 0 {
                assert!(
                    result.distance >= results[i - 1].distance,
                    "Results should be sorted by distance"
                );
            }
        }
    }

    #[test]
    fn test_search_with_config_rabitq() {
        use tempfile::TempDir;

        use crate::vector::embedding::EmbeddingBuilder;
        use crate::vector::Distance;

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().expect("Failed to initialize storage");
        let storage = Arc::new(storage);

        let registry = Arc::new(EmbeddingRegistry::new());
        let txn_db = storage.transaction_db().expect("Failed to get txn_db");
        // Use Cosine distance - will get RaBitQ strategy
        let builder = EmbeddingBuilder::new("test-model", 64, Distance::Cosine);
        let embedding = registry
            .register(builder, &txn_db)
            .expect("Failed to register");

        // Create processor (HNSW always enabled, params derived from EmbeddingSpec)
        let processor = Processor::new(storage, registry);

        // Insert several vectors
        let mut inserted_ids = Vec::new();
        for i in 0..10 {
            let vector: Vec<f32> = (0..64).map(|j| (i * 64 + j) as f32 / 1000.0).collect();
            let id = crate::Id::new();
            processor
                .insert_vector(&embedding, ExternalKey::NodeId(id), &vector, true)
                .expect("Insert should succeed");
            inserted_ids.push(id);
        }

        // Create SearchConfig - will auto-select RaBitQ for Cosine
        let config = SearchConfig::new(embedding.clone(), 5)
            .with_ef(100)
            .with_rerank_factor(4);

        // Verify strategy is RaBitQ for Cosine distance
        assert!(config.strategy().is_rabitq(), "Cosine should use RaBitQ strategy");

        // Search using config
        let query: Vec<f32> = (0..64).map(|j| j as f32 / 1000.0).collect();
        let results = processor
            .search_with_config(&config, &query)
            .expect("Search should succeed");

        // Should return at most k results
        assert!(!results.is_empty(), "Should have results");
        assert!(results.len() <= 5, "Should return at most k results");

        // Results should be sorted by distance
        for (i, result) in results.iter().enumerate() {
            if i > 0 {
                assert!(
                    result.distance >= results[i - 1].distance,
                    "Results should be sorted by distance"
                );
            }
        }
    }

    // =========================================================================
    // Spec Hash (Config Persistence) Tests
    // =========================================================================

    #[test]
    fn test_spec_hash_stored_on_first_insert() {
        use tempfile::TempDir;

        use crate::rocksdb::ColumnFamily;
        use crate::vector::embedding::EmbeddingBuilder;
        use crate::vector::Distance;

        let temp_dir = TempDir::new().expect("temp dir");
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().expect("storage init");
        let storage = Arc::new(storage);
        let registry = Arc::new(EmbeddingRegistry::new());

        // Register embedding
        let txn_db = storage.transaction_db().expect("txn_db");
        let builder = EmbeddingBuilder::new("test", 64, Distance::L2);
        let embedding = registry.register(builder, &txn_db).expect("register");
        let embedding_code = embedding.code();

        // Create processor
        let processor = Processor::new(storage.clone(), registry);

        // Insert a vector (first insert should store spec hash)
        let vector: Vec<f32> = (0..64).map(|i| i as f32 / 100.0).collect();
        let id = crate::Id::new();
        processor
            .insert_vector(&embedding, ExternalKey::NodeId(id), &vector, true)
            .expect("insert");

        // Verify spec hash was stored
        let graph_meta_cf = txn_db
            .cf_handle(GraphMeta::CF_NAME)
            .expect("GraphMeta CF");
        let hash_key = GraphMetaCfKey::spec_hash(embedding_code);
        let hash_bytes = txn_db
            .get_cf(&graph_meta_cf, GraphMeta::key_to_bytes(&hash_key))
            .expect("get")
            .expect("spec hash should exist after first insert");

        let hash_value = GraphMeta::value_from_bytes(&hash_key, &hash_bytes).expect("parse");
        if let GraphMetaField::SpecHash(hash) = hash_value.0 {
            assert!(hash != 0, "Spec hash should be non-zero");
        } else {
            panic!("Expected SpecHash variant");
        }
    }

    #[test]
    fn test_spec_hash_validated_on_subsequent_insert() {
        use tempfile::TempDir;

        use crate::rocksdb::ColumnFamily;
        use crate::vector::embedding::EmbeddingBuilder;
        use crate::vector::Distance;

        let temp_dir = TempDir::new().expect("temp dir");
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().expect("storage init");
        let storage = Arc::new(storage);
        let registry = Arc::new(EmbeddingRegistry::new());

        // Register embedding
        let txn_db = storage.transaction_db().expect("txn_db");
        let builder = EmbeddingBuilder::new("test", 64, Distance::L2);
        let embedding = registry.register(builder, &txn_db).expect("register");
        let embedding_code = embedding.code();

        // Create processor
        let processor = Processor::new(storage.clone(), registry);

        // Insert first vector (stores spec hash)
        let vector: Vec<f32> = (0..64).map(|i| i as f32 / 100.0).collect();
        let id1 = crate::Id::new();
        processor
            .insert_vector(&embedding, ExternalKey::NodeId(id1), &vector, true)
            .expect("first insert");

        // Get the stored hash
        let graph_meta_cf = txn_db
            .cf_handle(GraphMeta::CF_NAME)
            .expect("GraphMeta CF");
        let hash_key = GraphMetaCfKey::spec_hash(embedding_code);
        let original_hash_bytes = txn_db
            .get_cf(&graph_meta_cf, GraphMeta::key_to_bytes(&hash_key))
            .expect("get")
            .expect("hash should exist");

        // Tamper with spec hash (simulate config drift)
        let wrong_hash_value = GraphMetaCfValue(GraphMetaField::SpecHash(12345));
        txn_db
            .put_cf(
                &graph_meta_cf,
                GraphMeta::key_to_bytes(&hash_key),
                GraphMeta::value_to_bytes(&wrong_hash_value),
            )
            .expect("put tampered hash");

        // Try to insert second vector - should fail due to hash mismatch
        let id2 = crate::Id::new();
        let result = processor.insert_vector(&embedding, ExternalKey::NodeId(id2), &vector, true);
        assert!(result.is_err(), "Should fail with hash mismatch");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("EmbeddingSpec changed") || err.contains("Rebuild required"),
            "Error should mention spec change: {}",
            err
        );

        // Restore original hash
        txn_db
            .put_cf(
                &graph_meta_cf,
                GraphMeta::key_to_bytes(&hash_key),
                &original_hash_bytes,
            )
            .expect("restore hash");

        // Now insert should succeed
        processor
            .insert_vector(&embedding, ExternalKey::NodeId(id2), &vector, true)
            .expect("second insert should succeed");
    }

    #[test]
    fn test_search_validates_spec_hash() {
        use tempfile::TempDir;

        use crate::rocksdb::ColumnFamily;
        use crate::vector::embedding::EmbeddingBuilder;
        use crate::vector::search::SearchConfig;
        use crate::vector::Distance;

        let temp_dir = TempDir::new().expect("temp dir");
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().expect("storage init");
        let storage = Arc::new(storage);
        let registry = Arc::new(EmbeddingRegistry::new());

        // Register embedding
        let txn_db = storage.transaction_db().expect("txn_db");
        let builder = EmbeddingBuilder::new("test", 64, Distance::L2);
        let embedding = registry.register(builder, &txn_db).expect("register");
        let embedding_code = embedding.code();

        // Create processor
        let processor = Processor::new(storage.clone(), registry);

        // Insert vectors (stores spec hash)
        for i in 0..5 {
            let vector: Vec<f32> = (0..64).map(|j| (i * 64 + j) as f32 / 1000.0).collect();
            let id = crate::Id::new();
            processor
                .insert_vector(&embedding, ExternalKey::NodeId(id), &vector, true)
                .expect("insert");
        }

        // Search should work initially
        let config = SearchConfig::new(embedding.clone(), 3).with_ef(50);
        let query: Vec<f32> = (0..64).map(|j| j as f32 / 500.0).collect();
        let results = processor.search_with_config(&config, &query);
        assert!(results.is_ok(), "Search should succeed initially");

        // Tamper with spec hash
        let graph_meta_cf = txn_db
            .cf_handle(GraphMeta::CF_NAME)
            .expect("GraphMeta CF");
        let hash_key = GraphMetaCfKey::spec_hash(embedding_code);
        let wrong_hash_value = GraphMetaCfValue(GraphMetaField::SpecHash(99999));
        txn_db
            .put_cf(
                &graph_meta_cf,
                GraphMeta::key_to_bytes(&hash_key),
                GraphMeta::value_to_bytes(&wrong_hash_value),
            )
            .expect("put tampered hash");

        // Search should now fail
        let result = processor.search_with_config(&config, &query);
        assert!(result.is_err(), "Search should fail with tampered hash");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("EmbeddingSpec changed") || err.contains("Rebuild required"),
            "Error should mention spec change: {}",
            err
        );
    }

    // =========================================================================
    // Batch Insert Tests
    // =========================================================================

    #[test]
    fn test_insert_batch_basic() {
        use tempfile::TempDir;

        use crate::vector::embedding::EmbeddingBuilder;
        use crate::vector::Distance;

        let temp_dir = TempDir::new().expect("temp dir");
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().expect("storage init");
        let storage = Arc::new(storage);
        let registry = Arc::new(EmbeddingRegistry::new());

        // Register embedding
        let txn_db = storage.transaction_db().expect("txn_db");
        let builder = EmbeddingBuilder::new("test", 64, Distance::L2);
        let embedding = registry.register(builder, &txn_db).expect("register");
        let embedding_code = embedding.code();

        let processor = Processor::new(storage.clone(), registry);

        // Prepare batch of vectors
        let vectors: Vec<(ExternalKey, Vec<f32>)> = (0..10)
            .map(|i| {
                let id = crate::Id::new();
                let vector: Vec<f32> = (0..64).map(|j| (i * 64 + j) as f32 / 1000.0).collect();
                (ExternalKey::NodeId(id), vector)
            })
            .collect();

        // Insert batch without index
        let vec_ids = processor
            .insert_batch(&embedding, &vectors, false)
            .expect("batch insert should succeed");

        // Verify results
        assert_eq!(vec_ids.len(), 10, "Should return 10 VecIds");
        for (i, vec_id) in vec_ids.iter().enumerate() {
            assert_eq!(*vec_id, i as VecId, "VecIds should be sequential");
        }

        // Verify allocator state
        {
            let allocator = processor.get_or_create_allocator(embedding_code);
            assert_eq!(allocator.next_id(), 10, "Allocator should be at 10");
        }
    }

    #[test]
    fn test_insert_batch_with_index() {
        use tempfile::TempDir;

        use crate::vector::embedding::EmbeddingBuilder;
        use crate::vector::Distance;

        let temp_dir = TempDir::new().expect("temp dir");
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().expect("storage init");
        let storage = Arc::new(storage);
        let registry = Arc::new(EmbeddingRegistry::new());

        // Register embedding
        let txn_db = storage.transaction_db().expect("txn_db");
        let builder = EmbeddingBuilder::new("test", 64, Distance::L2);
        let embedding = registry.register(builder, &txn_db).expect("register");

        // Create processor (HNSW always enabled, params derived from EmbeddingSpec)
        let processor = Processor::new(storage.clone(), registry);

        // Prepare batch of vectors
        let vectors: Vec<(ExternalKey, Vec<f32>)> = (0..20)
            .map(|i| {
                let id = crate::Id::new();
                let vector: Vec<f32> = (0..64).map(|j| (i * 64 + j) as f32 / 1000.0).collect();
                (ExternalKey::NodeId(id), vector)
            })
            .collect();

        // Insert batch with index building
        let vec_ids = processor
            .insert_batch(&embedding, &vectors, true)
            .expect("batch insert with index should succeed");

        assert_eq!(vec_ids.len(), 20, "Should return 20 VecIds");

        // Verify search works
        let query: Vec<f32> = (0..64).map(|j| j as f32 / 1000.0).collect();
        let results = processor
            .search(&embedding, &query, 5, 100)
            .expect("search should succeed");

        assert!(!results.is_empty(), "Should have search results");
        assert!(results.len() <= 5, "Should return at most k results");

        // Verify results are sorted by distance and distances are reasonable
        for (i, result) in results.iter().enumerate() {
            if i > 0 {
                assert!(
                    result.distance >= results[i - 1].distance,
                    "Results should be sorted by distance"
                );
            }
        }

        // The first result should be among the inserted vectors
        // Extract node IDs from ExternalKey::NodeId for comparison
        let inserted_ids: std::collections::HashSet<_> = vectors
            .iter()
            .filter_map(|(key, _)| {
                if let ExternalKey::NodeId(id) = key {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect();
        assert!(
            inserted_ids.contains(&results[0].node_id().expect("expected NodeId")),
            "First result should be one of our inserted vectors"
        );
    }

    #[test]
    fn test_insert_batch_empty() {
        use tempfile::TempDir;

        use crate::vector::embedding::EmbeddingBuilder;
        use crate::vector::Distance;

        let temp_dir = TempDir::new().expect("temp dir");
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().expect("storage init");
        let storage = Arc::new(storage);
        let registry = Arc::new(EmbeddingRegistry::new());

        let txn_db = storage.transaction_db().expect("txn_db");
        let builder = EmbeddingBuilder::new("test", 64, Distance::L2);
        let embedding = registry.register(builder, &txn_db).expect("register");
        let embedding_code = embedding.code();

        let processor = Processor::new(storage, registry);

        // Empty batch should return empty vec
        let empty: Vec<(ExternalKey, Vec<f32>)> = vec![];
        let vec_ids = processor
            .insert_batch(&embedding, &empty, false)
            .expect("empty batch should succeed");

        assert!(vec_ids.is_empty(), "Empty input should return empty output");
    }

    #[test]
    fn test_insert_batch_dimension_mismatch() {
        use tempfile::TempDir;

        use crate::vector::embedding::EmbeddingBuilder;
        use crate::vector::Distance;

        let temp_dir = TempDir::new().expect("temp dir");
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().expect("storage init");
        let storage = Arc::new(storage);
        let registry = Arc::new(EmbeddingRegistry::new());

        let txn_db = storage.transaction_db().expect("txn_db");
        let builder = EmbeddingBuilder::new("test", 64, Distance::L2);
        let embedding = registry.register(builder, &txn_db).expect("register");
        let embedding_code = embedding.code();

        let processor = Processor::new(storage, registry);

        // Create batch with one wrong dimension
        let vectors: Vec<(ExternalKey, Vec<f32>)> = vec![
            (ExternalKey::NodeId(crate::Id::new()), vec![0.0; 64]),  // Correct
            (ExternalKey::NodeId(crate::Id::new()), vec![0.0; 128]), // Wrong dimension
            (ExternalKey::NodeId(crate::Id::new()), vec![0.0; 64]),  // Correct
        ];

        let result = processor.insert_batch(&embedding, &vectors, false);
        assert!(result.is_err(), "Should fail on dimension mismatch");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Dimension mismatch"),
            "Error should mention dimension mismatch: {}",
            err
        );
    }

    #[test]
    fn test_insert_batch_duplicate_in_batch() {
        use tempfile::TempDir;

        use crate::vector::embedding::EmbeddingBuilder;
        use crate::vector::Distance;

        let temp_dir = TempDir::new().expect("temp dir");
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().expect("storage init");
        let storage = Arc::new(storage);
        let registry = Arc::new(EmbeddingRegistry::new());

        let txn_db = storage.transaction_db().expect("txn_db");
        let builder = EmbeddingBuilder::new("test", 64, Distance::L2);
        let embedding = registry.register(builder, &txn_db).expect("register");
        let embedding_code = embedding.code();

        let processor = Processor::new(storage, registry);

        // Create batch with duplicate ID
        let duplicate_id = crate::Id::new();
        let vectors: Vec<(ExternalKey, Vec<f32>)> = vec![
            (ExternalKey::NodeId(crate::Id::new()), vec![0.0; 64]),
            (ExternalKey::NodeId(duplicate_id), vec![0.0; 64]),
            (ExternalKey::NodeId(duplicate_id), vec![0.0; 64]), // Duplicate!
        ];

        let result = processor.insert_batch(&embedding, &vectors, false);
        assert!(result.is_err(), "Should fail on duplicate ID in batch");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Duplicate external key"),
            "Error should mention duplicate: {}",
            err
        );
    }

    #[test]
    fn test_insert_batch_duplicate_existing() {
        use tempfile::TempDir;

        use crate::vector::embedding::EmbeddingBuilder;
        use crate::vector::Distance;

        let temp_dir = TempDir::new().expect("temp dir");
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().expect("storage init");
        let storage = Arc::new(storage);
        let registry = Arc::new(EmbeddingRegistry::new());

        let txn_db = storage.transaction_db().expect("txn_db");
        let builder = EmbeddingBuilder::new("test", 64, Distance::L2);
        let embedding = registry.register(builder, &txn_db).expect("register");
        let embedding_code = embedding.code();

        let processor = Processor::new(storage, registry);

        // Insert a vector first
        let existing_id = crate::Id::new();
        let vector: Vec<f32> = vec![0.0; 64];
        processor
            .insert_vector(&embedding, ExternalKey::NodeId(existing_id), &vector, false)
            .expect("initial insert");

        // Try batch with that existing ID
        let vectors: Vec<(ExternalKey, Vec<f32>)> = vec![
            (ExternalKey::NodeId(crate::Id::new()), vec![0.0; 64]),
            (ExternalKey::NodeId(existing_id), vec![0.0; 64]), // Already exists!
        ];

        let result = processor.insert_batch(&embedding, &vectors, false);
        assert!(result.is_err(), "Should fail on existing ID");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("already exists"),
            "Error should mention already exists: {}",
            err
        );
    }

    // Note: test_insert_batch_unknown_embedding removed - with &Embedding API,
    // you can't pass an invalid embedding (must register first to get Embedding)

    #[test]
    fn test_insert_batch_atomicity() {
        use tempfile::TempDir;

        use crate::vector::embedding::EmbeddingBuilder;
        use crate::vector::Distance;

        let temp_dir = TempDir::new().expect("temp dir");
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().expect("storage init");
        let storage = Arc::new(storage);
        let registry = Arc::new(EmbeddingRegistry::new());

        let txn_db = storage.transaction_db().expect("txn_db");
        let builder = EmbeddingBuilder::new("test", 64, Distance::L2);
        let embedding = registry.register(builder, &txn_db).expect("register");
        let embedding_code = embedding.code();

        let processor = Processor::new(storage.clone(), registry);

        // Insert a vector first
        let existing_id = crate::Id::new();
        processor
            .insert_vector(&embedding, ExternalKey::NodeId(existing_id), &vec![0.0; 64], false)
            .expect("initial insert");

        // Try batch where the LAST vector will fail (duplicate)
        // The first vectors should NOT be committed due to atomicity
        let new_id1 = crate::Id::new();
        let new_id2 = crate::Id::new();
        let vectors: Vec<(ExternalKey, Vec<f32>)> = vec![
            (ExternalKey::NodeId(new_id1), vec![1.0; 64]),
            (ExternalKey::NodeId(new_id2), vec![2.0; 64]),
            (ExternalKey::NodeId(existing_id), vec![3.0; 64]), // Will fail!
        ];

        let result = processor.insert_batch(&embedding, &vectors, false);
        assert!(result.is_err(), "Batch should fail");

        // Verify new_id1 and new_id2 were NOT inserted (atomicity)
        let forward_cf = txn_db
            .cf_handle(IdForward::CF_NAME)
            .expect("CF should exist");

        let key1 = IdForwardCfKey(embedding_code, ExternalKey::NodeId(new_id1));
        let result1 = txn_db
            .get_cf(&forward_cf, IdForward::key_to_bytes(&key1))
            .expect("read");
        assert!(
            result1.is_none(),
            "new_id1 should NOT exist due to atomicity"
        );

        let key2 = IdForwardCfKey(embedding_code, ExternalKey::NodeId(new_id2));
        let result2 = txn_db
            .get_cf(&forward_cf, IdForward::key_to_bytes(&key2))
            .expect("read");
        assert!(
            result2.is_none(),
            "new_id2 should NOT exist due to atomicity"
        );
    }
}
