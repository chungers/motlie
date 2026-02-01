//! Vector insertion operations.
//!
//! Shared helpers for inserting vectors, used by both `Processor` and
//! `MutationExecutor` implementations.

use std::collections::HashSet;

use anyhow::Result;

use crate::rocksdb::{ColumnFamily, ColumnFamilySerde, HotColumnFamilyRecord};
use crate::vector::cache::BinaryCodeCache;
use crate::vector::hnsw::{self, CacheUpdate};
use crate::vector::processor::Processor;
use crate::vector::schema::{
    AdcCorrection, BinaryCodeCfKey, BinaryCodeCfValue, BinaryCodes, EmbeddingCode, EmbeddingSpecCfKey,
    EmbeddingSpecs, ExternalKey, GraphMeta, GraphMetaCfKey, GraphMetaCfValue, GraphMetaField, IdForward,
    IdForwardCfKey, IdForwardCfValue, IdReverse, IdReverseCfKey, IdReverseCfValue,
    LifecycleCounts, LifecycleCountsCfKey, LifecycleCountsDelta, Pending, VecId,
    VecMeta, VecMetaCfKey, VecMetaCfValue, VecMetadata, VectorCfKey, Vectors,
};

// ============================================================================
// InsertResult
// ============================================================================

/// Result of a single vector insertion.
///
/// Contains the allocated VecId and deferred cache updates to apply
/// after transaction commit.
pub struct InsertResult {
    /// Allocated internal vector ID
    pub vec_id: VecId,
    /// Navigation cache update from HNSW indexing (if any)
    pub nav_cache_update: Option<CacheUpdate>,
    /// Binary code for cache update (if RaBitQ enabled)
    pub code_cache_update: Option<(Vec<u8>, AdcCorrection)>,
}

impl InsertResult {
    /// Apply cache updates to the processor caches.
    ///
    /// MUST be called AFTER transaction commit to ensure consistency.
    pub fn apply_cache_updates(
        self,
        embedding: EmbeddingCode,
        nav_cache: &crate::vector::cache::NavigationCache,
        code_cache: &BinaryCodeCache,
    ) {
        if let Some(update) = self.nav_cache_update {
            update.apply(nav_cache);
        }
        if let Some((code, correction)) = self.code_cache_update {
            code_cache.put(embedding, self.vec_id, code, correction);
        }
    }
}

// ============================================================================
// InsertBatchResult
// ============================================================================

/// Result of a batch vector insertion.
pub struct InsertBatchResult {
    /// Allocated internal vector IDs (same order as input)
    pub vec_ids: Vec<VecId>,
    /// Navigation cache updates from HNSW indexing
    pub nav_cache_updates: Vec<CacheUpdate>,
    /// Binary code cache updates (vec_id, code, correction)
    pub code_cache_updates: Vec<(VecId, Vec<u8>, AdcCorrection)>,
}

impl InsertBatchResult {
    /// Apply all cache updates to the processor caches.
    ///
    /// MUST be called AFTER transaction commit to ensure consistency.
    pub fn apply_cache_updates(
        self,
        embedding: EmbeddingCode,
        nav_cache: &crate::vector::cache::NavigationCache,
        code_cache: &BinaryCodeCache,
    ) {
        for update in self.nav_cache_updates {
            update.apply(nav_cache);
        }
        for (vec_id, code, correction) in self.code_cache_updates {
            code_cache.put(embedding, vec_id, code, correction);
        }
    }
}

// ============================================================================
// ops::insert::vector
// ============================================================================

/// Insert a single vector within an existing transaction.
///
/// This is the shared implementation used by both `Processor::insert_vector()`
/// and `MutationExecutor for InsertVector`.
///
/// # Usage
///
/// ```rust,ignore
/// use crate::vector::ops;
///
/// let result = ops::insert::vector(&txn, &txn_db, processor, embedding, id, &vec, true)?;
/// txn.commit()?;
/// result.apply_cache_updates(embedding, nav_cache, code_cache);
/// ```
///
/// # Validation
/// - Embedding exists in registry
/// - Dimension matches spec
/// - External ID not already used (with lock to prevent races)
/// - Spec hash drift detection
///
/// # Arguments
/// * `txn` - Active RocksDB transaction
/// * `txn_db` - Transaction DB for CF handles
/// * `processor` - Processor for allocators, encoders, indices
/// * `embedding` - Embedding space code
/// * `external_key` - External key identifying the source entity
/// * `vector` - Vector data
/// * `build_index` - Whether to build HNSW graph connections
///
/// # Returns
/// `InsertResult` with vec_id and deferred cache updates.
pub(crate) fn vector(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    processor: &Processor,
    embedding: EmbeddingCode,
    external_key: ExternalKey,
    vector: &[f32],
    build_index: bool,
) -> Result<InsertResult> {
    // 1. Validate embedding exists and get spec
    let spec = processor
        .registry()
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

    // 3. Check if external key already exists (avoid duplicates)
    // Use get_for_update_cf() to acquire a lock on the key
    let forward_cf = txn_db
        .cf_handle(IdForward::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("IdForward CF not found"))?;
    let forward_key = IdForwardCfKey(embedding, external_key.clone());
    if txn
        .get_for_update_cf(&forward_cf, IdForward::key_to_bytes(&forward_key), true)?
        .is_some()
    {
        return Err(anyhow::anyhow!(
            "External key {:?} already exists in embedding space {}",
            external_key,
            embedding
        ));
    }

    // 4. Apply backpressure for async inserts if configured
    if !build_index {
        let threshold = processor.async_backpressure_threshold();
        if threshold > 0 {
            let pending = processor.pending_queue_size();
            if pending >= threshold {
                return Err(anyhow::anyhow!(
                    "Async insert backpressure: pending queue size {} exceeds threshold {}",
                    pending,
                    threshold
                ));
            }
        }
    }

    // 5. Allocate internal ID
    let vec_id = {
        let allocator = processor.get_or_create_allocator(embedding);
        allocator.allocate(txn, txn_db, embedding)?
    };

    // 6. Store Id -> VecId mapping (IdForward)
    let forward_value = IdForwardCfValue(vec_id);
    txn.put_cf(
        &forward_cf,
        IdForward::key_to_bytes(&forward_key),
        IdForward::value_to_bytes(&forward_value),
    )?;

    // 7. Store VecId -> ExternalKey mapping (IdReverse)
    let reverse_key = IdReverseCfKey(embedding, vec_id);
    let reverse_value = IdReverseCfValue(external_key);
    let reverse_cf = txn_db
        .cf_handle(IdReverse::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("IdReverse CF not found"))?;
    txn.put_cf(
        &reverse_cf,
        IdReverse::key_to_bytes(&reverse_key),
        IdReverse::value_to_bytes(&reverse_value),
    )?;

    // 8. Store vector data
    let vec_key = VectorCfKey(embedding, vec_id);
    let vectors_cf = txn_db
        .cf_handle(Vectors::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;
    txn.put_cf(
        &vectors_cf,
        Vectors::key_to_bytes(&vec_key),
        Vectors::value_to_bytes_typed(vector, storage_type),
    )?;

    // 9. Store binary code with ADC correction (if RaBitQ enabled)
    let code_cache_update = if let Some(encoder) = processor.get_or_create_encoder(embedding) {
        let (code, correction) = encoder.encode_with_correction(vector);
        let code_key = BinaryCodeCfKey(embedding, vec_id);
        let code_value = BinaryCodeCfValue {
            code: code.clone(),
            correction,
        };
        let codes_cf = txn_db
            .cf_handle(BinaryCodes::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("BinaryCodes CF not found"))?;
        txn.put_cf(
            &codes_cf,
            BinaryCodes::key_to_bytes(&code_key),
            BinaryCodes::value_to_bytes(&code_value),
        )?;
        Some((code, correction))
    } else {
        None
    };

    // 10. Validate/store spec hash (drift detection)
    validate_or_store_spec_hash(txn, txn_db, embedding)?;

    // 11. Build HNSW index or queue for async processing
    let nav_cache_update = if build_index {
        // Synchronous path: build HNSW graph immediately
        // (VecMeta is written inside hnsw::insert with max_layer and flags=0)
        if let Some(index) = processor.get_or_create_index(embedding) {
            Some(hnsw::insert(
                &index,
                txn,
                txn_db,
                processor.storage(),
                vec_id,
                vector,
            )?)
        } else {
            None
        }
    } else {
        // Async path: store with FLAG_PENDING and add to pending queue
        // Vector data is stored (steps 5-8), graph construction deferred

        // Store VecMeta with FLAG_PENDING
        let vec_meta_cf = txn_db
            .cf_handle(VecMeta::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("VecMeta CF not found"))?;
        let meta_key = VecMetaCfKey(embedding, vec_id);
        let meta_value = VecMetaCfValue(VecMetadata::pending());
        txn.put_cf(
            &vec_meta_cf,
            VecMeta::key_to_bytes(&meta_key),
            VecMeta::value_to_bytes(&meta_value)?,
        )?;

        // Add to pending queue for async graph construction
        let pending_cf = txn_db
            .cf_handle(Pending::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Pending CF not found"))?;
        let pending_key = Pending::key_now(embedding, vec_id);
        txn.put_cf(&pending_cf, Pending::key_to_bytes(&pending_key), &[])?;

        None
    };

    // 12. Update lifecycle counters via merge operator
    let lifecycle_cf = txn_db
        .cf_handle(LifecycleCounts::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("LifecycleCounts CF not found"))?;
    let lifecycle_key = LifecycleCountsCfKey(embedding);
    let delta = if build_index {
        LifecycleCountsDelta::inc_indexed()
    } else {
        LifecycleCountsDelta::inc_pending()
    };
    txn.merge_cf(
        &lifecycle_cf,
        LifecycleCounts::key_to_bytes(&lifecycle_key),
        delta.to_bytes(),
    )?;

    Ok(InsertResult {
        vec_id,
        nav_cache_update,
        code_cache_update,
    })
}

// ============================================================================
// ops::insert::batch
// ============================================================================

/// Insert multiple vectors within an existing transaction.
///
/// This is the shared implementation used by both `Processor::insert_batch()`
/// and `MutationExecutor` for batch operations.
///
/// # Usage
///
/// ```rust,ignore
/// use crate::vector::ops;
///
/// let result = ops::insert::batch(&txn, &txn_db, processor, embedding, &vectors, true)?;
/// txn.commit()?;
/// result.apply_cache_updates(embedding, nav_cache, code_cache);
/// ```
///
/// # Validation
/// - Embedding exists in registry
/// - All dimensions match spec
/// - No duplicate external keys within batch
/// - No duplicate external keys against existing (with locks)
/// - Spec hash drift detection
///
/// # Arguments
/// * `txn` - Active RocksDB transaction
/// * `txn_db` - Transaction DB for CF handles
/// * `processor` - Processor for allocators, encoders, indices
/// * `embedding` - Embedding space code
/// * `vectors` - Slice of (external_key, vector_data) pairs
/// * `build_index` - Whether to build HNSW graph connections
///
/// # Returns
/// `InsertBatchResult` with vec_ids and deferred cache updates.
pub(crate) fn batch(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    processor: &Processor,
    embedding: EmbeddingCode,
    vectors: &[(ExternalKey, Vec<f32>)],
    build_index: bool,
) -> Result<InsertBatchResult> {
    // Fast path: empty input
    if vectors.is_empty() {
        return Ok(InsertBatchResult {
            vec_ids: vec![],
            nav_cache_updates: vec![],
            code_cache_updates: vec![],
        });
    }

    // 1. Validate embedding exists and get spec
    let spec = processor
        .registry()
        .get_by_code(embedding)
        .ok_or_else(|| anyhow::anyhow!("Unknown embedding code: {}", embedding))?;

    let expected_dim = spec.dim() as usize;
    let storage_type = spec.storage_type();

    // 2. Validate all dimensions upfront (fail fast)
    for (i, (external_key, vector)) in vectors.iter().enumerate() {
        if vector.len() != expected_dim {
            return Err(anyhow::anyhow!(
                "Dimension mismatch for vector {} ({:?}): expected {}, got {}",
                i,
                external_key,
                expected_dim,
                vector.len()
            ));
        }
    }

    // 3. Check for duplicate external keys within the batch
    {
        let mut seen: HashSet<Vec<u8>> = HashSet::with_capacity(vectors.len());
        for (external_key, _) in vectors {
            let key_bytes = external_key.to_bytes();
            if !seen.insert(key_bytes) {
                return Err(anyhow::anyhow!("Duplicate external key {:?} in batch", external_key));
            }
        }
    }

    // 4. Apply backpressure for async inserts if configured
    if !build_index {
        let threshold = processor.async_backpressure_threshold();
        if threshold > 0 {
            let pending = processor.pending_queue_size();
            let projected = pending.saturating_add(vectors.len());
            if projected > threshold {
                return Err(anyhow::anyhow!(
                    "Async insert backpressure: pending queue size {} + batch {} exceeds threshold {}",
                    pending,
                    vectors.len(),
                    threshold
                ));
            }
        }
    }

    // 5. Get column family handles
    let forward_cf = txn_db
        .cf_handle(IdForward::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("IdForward CF not found"))?;
    let reverse_cf = txn_db
        .cf_handle(IdReverse::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("IdReverse CF not found"))?;
    let vectors_cf = txn_db
        .cf_handle(Vectors::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;

    // 6. Check for existing external keys (acquire locks to prevent races)
    for (external_key, _) in vectors {
        let forward_key = IdForwardCfKey(embedding, external_key.clone());
        if txn
            .get_for_update_cf(&forward_cf, IdForward::key_to_bytes(&forward_key), true)?
            .is_some()
        {
            return Err(anyhow::anyhow!(
                "External key {:?} already exists in embedding space {}",
                external_key,
                embedding
            ));
        }
    }

    // 7. Validate/store spec hash (drift detection) - once per batch
    validate_or_store_spec_hash(txn, txn_db, embedding)?;

    // 8. Get RaBitQ encoder if enabled
    let encoder = processor.get_or_create_encoder(embedding);

    // 9. Allocate IDs and store all vectors
    let mut vec_ids = Vec::with_capacity(vectors.len());
    let mut code_cache_updates = Vec::with_capacity(vectors.len());

    for (external_key, vector) in vectors {
        // Allocate internal ID
        let vec_id = {
            let allocator = processor.get_or_create_allocator(embedding);
            allocator.allocate(txn, txn_db, embedding)?
        };
        vec_ids.push(vec_id);

        // Store ExternalKey -> VecId mapping (IdForward)
        let forward_key = IdForwardCfKey(embedding, external_key.clone());
        let forward_value = IdForwardCfValue(vec_id);
        txn.put_cf(
            &forward_cf,
            IdForward::key_to_bytes(&forward_key),
            IdForward::value_to_bytes(&forward_value),
        )?;

        // Store VecId -> ExternalKey mapping (IdReverse)
        let reverse_key = IdReverseCfKey(embedding, vec_id);
        let reverse_value = IdReverseCfValue(external_key.clone());
        txn.put_cf(
            &reverse_cf,
            IdReverse::key_to_bytes(&reverse_key),
            IdReverse::value_to_bytes(&reverse_value),
        )?;

        // Store vector data
        let vec_key = VectorCfKey(embedding, vec_id);
        txn.put_cf(
            &vectors_cf,
            Vectors::key_to_bytes(&vec_key),
            Vectors::value_to_bytes_typed(vector, storage_type),
        )?;

        // Store binary code with ADC correction (if RaBitQ enabled)
        if let Some(ref enc) = encoder {
            let codes_cf = txn_db
                .cf_handle(BinaryCodes::CF_NAME)
                .ok_or_else(|| anyhow::anyhow!("BinaryCodes CF not found"))?;
            let (code, correction) = enc.encode_with_correction(vector);
            let code_key = BinaryCodeCfKey(embedding, vec_id);
            let code_value = BinaryCodeCfValue {
                code: code.clone(),
                correction,
            };
            txn.put_cf(
                &codes_cf,
                BinaryCodes::key_to_bytes(&code_key),
                BinaryCodes::value_to_bytes(&code_value),
            )?;
            code_cache_updates.push((vec_id, code, correction));
        }
    }

    // 10. Build HNSW index or queue for async processing - after all vectors are stored
    let mut nav_cache_updates = Vec::new();
    if build_index {
        // Synchronous path: build HNSW graph immediately
        //
        // IMPORTANT: We use insert_for_batch with BatchEdgeCache which:
        // 1. Uses transaction-aware reads to access uncommitted vector data
        // 2. Uses batch edge cache to see edges from earlier inserts in the same batch
        //    (RocksDB transactions don't expose pending merge operands to reads)
        // 3. Applies cache updates incrementally so subsequent inserts see the entry point
        //
        // If the transaction fails after we've applied some cache updates, the nav_cache
        // becomes stale. However, on next access, get_or_init_navigation() will reload
        // from storage and correct itself.
        if let Some(index) = processor.get_or_create_index(embedding) {
            // Create batch edge cache to track edges added during this batch
            let mut batch_edge_cache = hnsw::BatchEdgeCache::new();

            for (i, vec_id) in vec_ids.iter().enumerate() {
                let vector = &vectors[i].1;
                // Use batch-specific insert with edge cache for proper graph connectivity
                let update = hnsw::insert_for_batch(
                    &index,
                    txn,
                    txn_db,
                    processor.storage(),
                    *vec_id,
                    vector,
                    &mut batch_edge_cache,
                )?;
                // Apply cache update immediately so subsequent inserts see the entry point
                update.clone().apply(index.nav_cache());
                nav_cache_updates.push(update);
            }
        }
    } else {
        // Async path: store with FLAG_PENDING and add to pending queue
        // Vector data is stored (steps 5-8), graph construction deferred

        let vec_meta_cf = txn_db
            .cf_handle(VecMeta::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("VecMeta CF not found"))?;
        let pending_cf = txn_db
            .cf_handle(Pending::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Pending CF not found"))?;

        for vec_id in &vec_ids {
            // Store VecMeta with FLAG_PENDING
            let meta_key = VecMetaCfKey(embedding, *vec_id);
            let meta_value = VecMetaCfValue(VecMetadata::pending());
            txn.put_cf(
                &vec_meta_cf,
                VecMeta::key_to_bytes(&meta_key),
                VecMeta::value_to_bytes(&meta_value)?,
            )?;

            // Add to pending queue for async graph construction
            let pending_key = Pending::key_now(embedding, *vec_id);
            txn.put_cf(&pending_cf, Pending::key_to_bytes(&pending_key), &[])?;
        }
    }

    // 11. Update lifecycle counters via merge operator (once for entire batch)
    let lifecycle_cf = txn_db
        .cf_handle(LifecycleCounts::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("LifecycleCounts CF not found"))?;
    let lifecycle_key = LifecycleCountsCfKey(embedding);
    let count = vec_ids.len() as i64;
    let delta = if build_index {
        LifecycleCountsDelta { indexed: count, ..Default::default() }
    } else {
        LifecycleCountsDelta { pending: count, ..Default::default() }
    };
    txn.merge_cf(
        &lifecycle_cf,
        LifecycleCounts::key_to_bytes(&lifecycle_key),
        delta.to_bytes(),
    )?;

    Ok(InsertBatchResult {
        vec_ids,
        nav_cache_updates,
        code_cache_updates,
    })
}

// ============================================================================
// Helper: Spec Hash Validation
// ============================================================================

/// Validate spec hash against stored value, or store it if first insert.
///
/// This ensures that the EmbeddingSpec hasn't changed since index build,
/// which would invalidate the HNSW graph and require a rebuild.
fn validate_or_store_spec_hash(
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    embedding: EmbeddingCode,
) -> Result<()> {
    let specs_cf = txn_db
        .cf_handle(EmbeddingSpecs::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("EmbeddingSpecs CF not found"))?;
    let graph_meta_cf = txn_db
        .cf_handle(GraphMeta::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("GraphMeta CF not found"))?;

    // Get EmbeddingSpec from storage
    let spec_key = EmbeddingSpecCfKey(embedding);
    let spec_bytes = txn
        .get_cf(&specs_cf, EmbeddingSpecs::key_to_bytes(&spec_key))?
        .ok_or_else(|| anyhow::anyhow!("EmbeddingSpec not found for {}", embedding))?;
    let embedding_spec = EmbeddingSpecs::value_from_bytes(&spec_bytes)?.0;

    // Compute current hash
    let current_hash = embedding_spec.compute_spec_hash();

    // Check if spec_hash already exists
    let hash_key = GraphMetaCfKey::spec_hash(embedding);
    let hash_key_bytes = GraphMeta::key_to_bytes(&hash_key);

    if let Some(stored_bytes) = txn.get_cf(&graph_meta_cf, &hash_key_bytes)? {
        // Validate against stored hash
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
        // First insert: store the hash
        let hash_value = GraphMetaCfValue(GraphMetaField::SpecHash(current_hash));
        txn.put_cf(
            &graph_meta_cf,
            hash_key_bytes,
            GraphMeta::value_to_bytes(&hash_value),
        )?;
    }

    Ok(())
}
