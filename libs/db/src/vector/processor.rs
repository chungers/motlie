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

use anyhow::{Context, Result};
use dashmap::DashMap;

use super::cache::NavigationCache;
use super::config::RaBitQConfig;
use super::hnsw::{self, insert_in_txn};
use super::id::IdAllocator;
use super::rabitq::RaBitQ;
use super::registry::EmbeddingRegistry;
use super::schema::{
    BinaryCodeCfKey, BinaryCodeCfValue, BinaryCodes, EmbeddingCode, IdForward, IdForwardCfKey,
    IdForwardCfValue, IdReverse, IdReverseCfKey, IdReverseCfValue, VecId, VectorCfKey, Vectors,
};
use super::Storage;
use crate::rocksdb::ColumnFamily;
use crate::Id;

// ============================================================================
// SearchResult
// ============================================================================

/// Search result containing external ID, internal ID, and distance.
///
/// Returned by `Processor::search()` for each matched vector.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// External ID (ULID) for the vector
    pub id: Id,
    /// Internal vector ID (compact u32)
    pub vec_id: VecId,
    /// Distance from query vector (lower = more similar)
    pub distance: f32,
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
/// let vec_id = allocator.allocate();
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
    /// HNSW configuration (shared across all embeddings)
    hnsw_config: hnsw::Config,
}

impl Processor {
    /// Create a new Processor with the given storage and registry.
    pub fn new(storage: Arc<Storage>, registry: Arc<EmbeddingRegistry>) -> Self {
        Self::with_config(
            storage,
            registry,
            RaBitQConfig::default(),
            hnsw::Config::default(),
        )
    }

    /// Create a Processor with custom RaBitQ configuration.
    pub fn with_rabitq_config(
        storage: Arc<Storage>,
        registry: Arc<EmbeddingRegistry>,
        rabitq_config: RaBitQConfig,
    ) -> Self {
        Self::with_config(storage, registry, rabitq_config, hnsw::Config::default())
    }

    /// Create a Processor with custom RaBitQ and HNSW configurations.
    pub fn with_config(
        storage: Arc<Storage>,
        registry: Arc<EmbeddingRegistry>,
        rabitq_config: RaBitQConfig,
        hnsw_config: hnsw::Config,
    ) -> Self {
        Self {
            storage,
            registry,
            id_allocators: DashMap::new(),
            rabitq_encoders: DashMap::new(),
            rabitq_config,
            hnsw_indices: DashMap::new(),
            nav_cache: Arc::new(NavigationCache::new()),
            hnsw_config,
        }
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

    /// Persist all ID allocators to storage.
    ///
    /// Should be called periodically or on shutdown to ensure
    /// crash recovery works correctly.
    pub fn persist_allocators(&self) -> Result<()> {
        let db = self.storage.transaction_db()?;
        for entry in self.id_allocators.iter() {
            let embedding = *entry.key();
            let allocator = entry.value();
            allocator.persist(&db, embedding)?;
        }
        Ok(())
    }

    /// Get the number of registered embedding spaces with allocators.
    pub fn allocator_count(&self) -> usize {
        self.id_allocators.len()
    }

    // ========================================================================
    // RaBitQ Encoder Management
    // ========================================================================

    /// Get or create a RaBitQ encoder for the given embedding space.
    ///
    /// Returns None if:
    /// - RaBitQ is disabled in config
    /// - Embedding is not found in registry (can't determine dimension)
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

        // Create and cache encoder
        let encoder = Arc::new(RaBitQ::from_config(dim, &self.rabitq_config));
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
    /// Returns None if the embedding is not found in the registry.
    /// Indices are cached per embedding space for efficiency.
    pub fn get_or_create_index(&self, embedding: EmbeddingCode) -> Option<hnsw::Index> {
        // Check cache first
        if let Some(index) = self.hnsw_indices.get(&embedding) {
            return Some(index.clone());
        }

        // Look up embedding to get distance metric and storage type
        let emb = self.registry.get_by_code(embedding)?;
        let distance = emb.distance();
        let storage_type = emb.storage_type();

        // Create index with shared navigation cache
        let index = hnsw::Index::with_storage_type(
            embedding,
            distance,
            storage_type,
            self.hnsw_config.clone(),
            Arc::clone(&self.nav_cache),
        );

        self.hnsw_indices.insert(embedding, index.clone());
        Some(index)
    }

    /// Get the shared navigation cache.
    pub fn nav_cache(&self) -> &Arc<NavigationCache> {
        &self.nav_cache
    }

    /// Get the HNSW configuration.
    pub fn hnsw_config(&self) -> &hnsw::Config {
        &self.hnsw_config
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
    /// * `id` - External ID (ULID) for the vector
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
    ///     Id::new(),
    ///     &[0.1, 0.2, 0.3, ...],
    ///     true,  // build HNSW index
    /// )?;
    /// ```
    pub fn insert_vector(
        &self,
        embedding: EmbeddingCode,
        id: Id,
        vector: &[f32],
        build_index: bool,
    ) -> Result<VecId> {
        // 1. Validate embedding exists and get spec
        let spec = self
            .registry
            .get_by_code(embedding)
            .ok_or_else(|| anyhow::anyhow!("Unknown embedding code: {}", embedding))?;

        // 2. Validate dimension
        if vector.len() != spec.dim() as usize {
            return Err(anyhow::anyhow!(
                "Dimension mismatch: expected {}, got {}",
                spec.dim(),
                vector.len()
            ));
        }

        let storage_type = spec.storage_type();

        // 3. Begin transaction
        let txn_db = self
            .storage
            .transaction_db()
            .context("Failed to get transaction DB")?;
        let txn = txn_db.transaction();

        // 4. Check if external ID already exists (avoid duplicates)
        // Use txn.get_for_update_cf() to acquire a lock on the key, preventing
        // concurrent inserts of the same external ID from racing.
        let forward_cf = txn_db
            .cf_handle(IdForward::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("IdForward CF not found"))?;
        let forward_key = IdForwardCfKey(embedding, id);
        if txn
            .get_for_update_cf(&forward_cf, IdForward::key_to_bytes(&forward_key), true)?
            .is_some()
        {
            return Err(anyhow::anyhow!(
                "External ID {} already exists in embedding space {}",
                id,
                embedding
            ));
        }

        // 5. Allocate internal ID (transactionally persisted)
        let vec_id = {
            let allocator = self.get_or_create_allocator(embedding);
            allocator.allocate_in_txn(&txn, &txn_db, embedding)?
        };

        // 6. Store Id -> VecId mapping (IdForward)
        let forward_value = IdForwardCfValue(vec_id);
        txn.put_cf(
            &forward_cf,
            IdForward::key_to_bytes(&forward_key),
            IdForward::value_to_bytes(&forward_value),
        )?;

        // 6. Store VecId -> Id mapping (IdReverse)
        let reverse_key = IdReverseCfKey(embedding, vec_id);
        let reverse_value = IdReverseCfValue(id);
        let reverse_cf = txn_db
            .cf_handle(IdReverse::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("IdReverse CF not found"))?;
        txn.put_cf(
            &reverse_cf,
            IdReverse::key_to_bytes(&reverse_key),
            IdReverse::value_to_bytes(&reverse_value),
        )?;

        // 7. Store vector data
        let vec_key = VectorCfKey(embedding, vec_id);
        let vectors_cf = txn_db
            .cf_handle(Vectors::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;
        txn.put_cf(
            &vectors_cf,
            Vectors::key_to_bytes(&vec_key),
            Vectors::value_to_bytes_typed(vector, storage_type),
        )?;

        // 8. Store binary code with ADC correction (if RaBitQ enabled)
        if let Some(encoder) = self.get_or_create_encoder(embedding) {
            let (code, correction) = encoder.encode_with_correction(vector);
            let code_key = BinaryCodeCfKey(embedding, vec_id);
            let code_value = BinaryCodeCfValue { code, correction };
            let codes_cf = txn_db
                .cf_handle(BinaryCodes::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("BinaryCodes CF not found"))?;
            txn.put_cf(
                &codes_cf,
                BinaryCodes::key_to_bytes(&code_key),
                BinaryCodes::value_to_bytes(&code_value),
            )?;
        }

        // 9. Build HNSW index (if requested)
        let cache_update = if build_index {
            if let Some(index) = self.get_or_create_index(embedding) {
                Some(insert_in_txn(
                    &index,
                    &txn,
                    &txn_db,
                    &self.storage,
                    vec_id,
                    vector,
                )?)
            } else {
                None
            }
        } else {
            None
        };

        // 10. Commit transaction
        txn.commit().context("Failed to commit transaction")?;

        // 11. Apply cache update AFTER successful commit
        if let Some(update) = cache_update {
            update.apply(&self.nav_cache);
        }

        Ok(vec_id)
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
    /// * `id` - External ID (ULID) of the vector to delete
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
    /// match processor.delete_vector(embedding_code, id)? {
    ///     Some(vec_id) => println!("Deleted vector with internal ID {}", vec_id),
    ///     None => println!("Vector did not exist"),
    /// }
    /// ```
    pub fn delete_vector(
        &self,
        embedding: EmbeddingCode,
        id: Id,
    ) -> Result<Option<VecId>> {
        // 1. Begin transaction
        let txn_db = self
            .storage
            .transaction_db()
            .context("Failed to get transaction DB")?;
        let txn = txn_db.transaction();

        // 2. Look up VecId from external Id (with lock to prevent races)
        let forward_cf = txn_db
            .cf_handle(IdForward::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("IdForward CF not found"))?;
        let forward_key = IdForwardCfKey(embedding, id);

        let vec_id = match txn.get_for_update_cf(
            &forward_cf,
            IdForward::key_to_bytes(&forward_key),
            true,
        )? {
            Some(bytes) => IdForward::value_from_bytes(&bytes)?.0,
            None => return Ok(None), // Already deleted or never existed
        };

        // 3. Delete IdForward mapping
        txn.delete_cf(&forward_cf, IdForward::key_to_bytes(&forward_key))?;

        // 4. Delete IdReverse mapping
        let reverse_key = IdReverseCfKey(embedding, vec_id);
        let reverse_cf = txn_db
            .cf_handle(IdReverse::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("IdReverse CF not found"))?;
        txn.delete_cf(&reverse_cf, IdReverse::key_to_bytes(&reverse_key))?;

        // Check if HNSW indexing is enabled - affects cleanup strategy
        let hnsw_enabled = self.hnsw_config.enabled;

        // 5. Delete vector data (only if HNSW disabled)
        // If HNSW is enabled, keep vector data for distance calculations
        // during search traversal (entry point or neighbor edges may reference it)
        let vec_key = VectorCfKey(embedding, vec_id);
        let vectors_cf = txn_db
            .cf_handle(Vectors::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;
        if !hnsw_enabled {
            txn.delete_cf(&vectors_cf, Vectors::key_to_bytes(&vec_key))?;
        }

        // 6. Delete binary code (only if RaBitQ enabled AND HNSW disabled)
        if self.rabitq_enabled() && !hnsw_enabled {
            let code_key = BinaryCodeCfKey(embedding, vec_id);
            let codes_cf = txn_db
                .cf_handle(BinaryCodes::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("BinaryCodes CF not found"))?;
            txn.delete_cf(&codes_cf, BinaryCodes::key_to_bytes(&code_key))?;
        }

        // 7. Return VecId to free list (only if HNSW disabled)
        // If HNSW is enabled, we cannot reuse the VecId because existing
        // HNSW edges still reference it. Reuse would corrupt graph semantics.
        if !hnsw_enabled {
            let allocator = self.get_or_create_allocator(embedding);
            allocator.free_in_txn(&txn, &txn_db, embedding, vec_id)?;
        }

        // 8. Commit transaction
        txn.commit().context("Failed to commit delete transaction")?;

        // Note: When HNSW is enabled, this is a "soft delete" - ID mappings
        // are removed but vector data is kept and VecId is not reused.
        // Full cleanup requires index rebuild or compaction (future work).

        Ok(Some(vec_id))
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
    ///     println!("ID: {}, distance: {}", result.id, result.distance);
    /// }
    /// ```
    pub fn search(
        &self,
        embedding: EmbeddingCode,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<SearchResult>> {
        // 1. Validate embedding exists and get spec
        let spec = self
            .registry
            .get_by_code(embedding)
            .ok_or_else(|| anyhow::anyhow!("Unknown embedding code: {}", embedding))?;

        // 2. Validate dimension
        if query.len() != spec.dim() as usize {
            return Err(anyhow::anyhow!(
                "Dimension mismatch: expected {}, got {}",
                spec.dim(),
                query.len()
            ));
        }

        // 3. Check HNSW is enabled
        if !self.hnsw_config.enabled {
            return Err(anyhow::anyhow!(
                "HNSW indexing is disabled - search not available"
            ));
        }

        // 4. Get or create HNSW index
        let index = self
            .get_or_create_index(embedding)
            .ok_or_else(|| anyhow::anyhow!("Failed to get HNSW index for embedding {}", embedding))?;

        // 5. Perform HNSW search with overfetch to handle tombstones
        // Overfetch by 2x to ensure we get enough live results after filtering
        let overfetch_k = k * 2;
        let raw_results = index.search(&self.storage, query, overfetch_k, ef_search)?;

        // 6. Filter deleted vectors using batched IdReverse lookup
        let txn_db = self.storage.transaction_db()?;
        let reverse_cf = txn_db
            .cf_handle(IdReverse::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("IdReverse CF not found"))?;

        // Build batch of keys for multi_get
        let keys: Vec<_> = raw_results
            .iter()
            .map(|(_, vec_id)| {
                let key = IdReverseCfKey(embedding, *vec_id);
                IdReverse::key_to_bytes(&key)
            })
            .collect();

        // Batch lookup all IdReverse keys at once
        let key_refs: Vec<_> = keys.iter().map(|k| (&reverse_cf, k.as_slice())).collect();
        let values = txn_db.multi_get_cf(key_refs);

        // Filter and resolve external IDs, truncate to k
        let mut results = Vec::with_capacity(k);
        for (i, value_result) in values.into_iter().enumerate() {
            if results.len() >= k {
                break;
            }
            if let Ok(Some(bytes)) = value_result {
                let id = IdReverse::value_from_bytes(&bytes)?.0;
                let (distance, vec_id) = raw_results[i];
                results.push(SearchResult {
                    id,
                    vec_id,
                    distance,
                });
            }
            // Skip deleted vectors (IdReverse missing or error)
        }

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
            let id1 = allocator.allocate();
            let id2 = allocator.allocate();
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
            .insert_vector(embedding_code, id, &vector, false)
            .expect("Insert should succeed");

        assert_eq!(vec_id, 0, "First vector should get ID 0");

        // Insert another with building index
        let id2 = crate::Id::new();
        let vector2: Vec<f32> = (0..64).map(|i| (i + 1) as f32 / 64.0).collect();
        let vec_id2 = processor
            .insert_vector(embedding_code, id2, &vector2, true)
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
            processor.insert_vector(embedding_code, crate::Id::new(), &wrong_dim_vector, false);

        assert!(result.is_err(), "Should fail on dimension mismatch");
        assert!(
            result.unwrap_err().to_string().contains("Dimension mismatch"),
            "Error should mention dimension mismatch"
        );
    }

    #[test]
    fn test_insert_vector_unknown_embedding() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().expect("Failed to initialize storage");
        let storage = Arc::new(storage);

        // Empty registry
        let registry = Arc::new(EmbeddingRegistry::new());
        let processor = Processor::new(storage, registry);

        // Try to insert into non-existent embedding
        let vector: Vec<f32> = vec![0.0; 64];
        let result = processor.insert_vector(999, crate::Id::new(), &vector, false);

        assert!(result.is_err(), "Should fail on unknown embedding");
        assert!(
            result.unwrap_err().to_string().contains("Unknown embedding"),
            "Error should mention unknown embedding"
        );
    }

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
            .insert_vector(embedding_code, id, &vector, false)
            .expect("First insert should succeed");

        // Try to insert with same external ID
        let vector2: Vec<f32> = (0..64).map(|i| (i + 1) as f32 / 64.0).collect();
        let result = processor.insert_vector(embedding_code, id, &vector2, false);

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
            .insert_vector(embedding_code, id, &vector, false)
            .expect("Insert should succeed");

        assert_eq!(vec_id, 0, "First vector should get ID 0");

        // Delete the vector
        let deleted = processor
            .delete_vector(embedding_code, id)
            .expect("Delete should succeed");

        assert_eq!(deleted, Some(0), "Should return deleted VecId");

        // Verify IdForward mapping is gone
        let forward_cf = txn_db
            .cf_handle(IdForward::CF_NAME)
            .expect("CF should exist");
        let forward_key = IdForwardCfKey(embedding_code, id);
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
            .delete_vector(embedding_code, id)
            .expect("Delete should not error");

        assert_eq!(result, None, "Should return None for non-existent vector");
    }

    #[test]
    fn test_delete_vector_id_reuse() {
        use tempfile::TempDir;

        use crate::vector::embedding::EmbeddingBuilder;
        use crate::vector::RaBitQConfig;
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

        // Create processor with HNSW disabled to test full delete with ID reuse
        let mut hnsw_config = hnsw::Config::default();
        hnsw_config.enabled = false;
        let processor = Processor::with_config(
            storage,
            registry,
            RaBitQConfig::default(),
            hnsw_config,
        );

        // Insert two vectors
        let vector: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let id1 = crate::Id::new();
        let id2 = crate::Id::new();

        let vec_id1 = processor
            .insert_vector(embedding_code, id1, &vector, false)
            .expect("Insert 1 should succeed");
        let vec_id2 = processor
            .insert_vector(embedding_code, id2, &vector, false)
            .expect("Insert 2 should succeed");

        assert_eq!(vec_id1, 0);
        assert_eq!(vec_id2, 1);

        // Delete first vector
        processor
            .delete_vector(embedding_code, id1)
            .expect("Delete should succeed");

        // Insert new vector - should reuse VecId 0
        let id3 = crate::Id::new();
        let vec_id3 = processor
            .insert_vector(embedding_code, id3, &vector, false)
            .expect("Insert 3 should succeed");

        assert_eq!(vec_id3, 0, "Should reuse freed VecId 0");

        // Next insert should get fresh ID
        let id4 = crate::Id::new();
        let vec_id4 = processor
            .insert_vector(embedding_code, id4, &vector, false)
            .expect("Insert 4 should succeed");

        assert_eq!(vec_id4, 2, "Should get next fresh ID");
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
        let embedding_code = embedding.code();

        // Create processor with HNSW enabled
        let mut hnsw_config = hnsw::Config::default();
        hnsw_config.enabled = true;
        hnsw_config.dim = 64;
        let processor = Processor::with_config(
            storage,
            registry,
            RaBitQConfig::default(),
            hnsw_config,
        );

        // Insert several vectors (with index building)
        let mut inserted_ids = Vec::new();
        for i in 0..10 {
            let vector: Vec<f32> = (0..64).map(|j| (i * 64 + j) as f32 / 1000.0).collect();
            let id = crate::Id::new();
            processor
                .insert_vector(embedding_code, id, &vector, true)
                .expect("Insert should succeed");
            inserted_ids.push(id);
        }

        // Search for nearest neighbors using first vector as query
        let query: Vec<f32> = (0..64).map(|j| j as f32 / 1000.0).collect();
        let results = processor
            .search(embedding_code, &query, 5, 100)
            .expect("Search should succeed");

        // Should return at most k results
        assert!(!results.is_empty(), "Should have results");
        assert!(results.len() <= 5, "Should return at most k results");

        // First result should be closest to query
        let first = &results[0];
        assert_eq!(first.id, inserted_ids[0], "First inserted vector should be closest");

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
        let embedding_code = embedding.code();

        // Create processor with HNSW enabled
        let mut hnsw_config = hnsw::Config::default();
        hnsw_config.enabled = true;
        hnsw_config.dim = 64;
        let processor = Processor::with_config(
            storage,
            registry,
            RaBitQConfig::default(),
            hnsw_config,
        );

        // Insert 5 vectors
        let mut inserted_ids = Vec::new();
        for i in 0..5 {
            let vector: Vec<f32> = (0..64).map(|j| (i * 64 + j) as f32 / 1000.0).collect();
            let id = crate::Id::new();
            processor
                .insert_vector(embedding_code, id, &vector, true)
                .expect("Insert should succeed");
            inserted_ids.push(id);
        }

        // Delete first two vectors (soft delete - mappings removed)
        processor
            .delete_vector(embedding_code, inserted_ids[0])
            .expect("Delete should succeed");
        processor
            .delete_vector(embedding_code, inserted_ids[1])
            .expect("Delete should succeed");

        // Search - deleted vectors should be filtered from results
        let query: Vec<f32> = (0..64).map(|j| j as f32 / 1000.0).collect();
        let results = processor
            .search(embedding_code, &query, 10, 100)
            .expect("Search should succeed");

        // Should have at most 3 results (5 - 2 deleted)
        assert!(results.len() <= 3, "Should only return non-deleted vectors");

        // Verify deleted IDs are not in results
        let result_ids: Vec<_> = results.iter().map(|r| r.id).collect();
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
        let result = processor.search(embedding_code, &wrong_dim_query, 10, 100);

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
        let embedding_code = embedding.code();

        // Create processor with HNSW disabled
        let mut hnsw_config = hnsw::Config::default();
        hnsw_config.enabled = false;
        let processor = Processor::with_config(
            storage,
            registry,
            RaBitQConfig::default(),
            hnsw_config,
        );

        // Try to search when HNSW is disabled
        let query: Vec<f32> = (0..64).map(|i| i as f32 / 64.0).collect();
        let result = processor.search(embedding_code, &query, 10, 100);

        assert!(result.is_err(), "Should fail when HNSW disabled");
        assert!(
            result.unwrap_err().to_string().contains("HNSW indexing is disabled"),
            "Error should mention HNSW disabled"
        );
    }
}
