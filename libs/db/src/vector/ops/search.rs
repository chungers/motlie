//! Search operations for vector queries.
//!
//! This module provides the core search algorithms used by `Processor`
//! methods. Each function takes resolved dependencies (storage, index,
//! encoder, etc.) rather than going through `Processor`.
//!
//! # Design
//!
//! The Processor acts as an orchestrator:
//! 1. Validates inputs (dimension, registry match)
//! 2. Resolves resources from DashMaps (index, encoder)
//! 3. Delegates to ops::search functions for the actual algorithms
//!
//! This keeps business logic (algorithms) in ops and resource management
//! (cache lifecycle, lazy init) in Processor.

use anyhow::Result;

use super::super::cache::BinaryCodeCache;
use super::super::embedding::Embedding;
use super::super::hnsw;
use super::super::processor::SearchResult;
use super::super::quantization::RaBitQ;
use super::super::schema::{
    EmbeddingCode, EmbeddingSpecCfKey, EmbeddingSpecs, ExternalKey, GraphMeta, GraphMetaCfKey,
    GraphMetaField, IdReverse, IdReverseCfKey, Pending, VecId, VecMeta, VecMetaCfKey, VectorCfKey,
    Vectors,
};
use super::super::search::SearchStrategy;
use super::super::Storage;
use crate::rocksdb::{ColumnFamily, ColumnFamilySerde, HotColumnFamilyRecord};

/// Validate that the stored SpecHash matches the current EmbeddingSpec.
///
/// This detects configuration drift — when the EmbeddingSpec has changed
/// since the HNSW index was built, requiring a rebuild.
///
/// Legacy indexes without a stored SpecHash emit a warning but succeed.
pub(crate) fn validate_spec_hash(
    storage: &Storage,
    embedding_code: EmbeddingCode,
) -> Result<()> {
    let txn_db = storage.transaction_db()?;

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
        tracing::warn!(
            embedding = embedding_code,
            "Legacy index without SpecHash - drift check skipped"
        );
    }

    Ok(())
}

/// Exact HNSW search with IdReverse tombstone filtering.
///
/// Performs:
/// 1. Overfetch HNSW search (2x k)
/// 2. Batch IdReverse lookup to filter deleted vectors
/// 3. Resolve external IDs
/// 4. Truncate to k results
///
/// Used by `Processor::search()` for the simple exact-distance path.
pub(crate) fn search_exact(
    storage: &Storage,
    index: &hnsw::Index,
    embedding_code: EmbeddingCode,
    query: &[f32],
    k: usize,
    ef_search: usize,
) -> Result<Vec<SearchResult>> {
    // Overfetch to handle tombstones
    let overfetch_k = k * 2;
    let effective_ef = ef_search.max(overfetch_k);
    let raw_results = index.search(storage, query, overfetch_k, effective_ef)?;

    // Filter deleted vectors using batched IdReverse lookup
    filter_and_resolve(storage, &raw_results, embedding_code, k)
}

/// Strategy-dispatched search with IdReverse + VecMeta tombstone filtering.
///
/// Dispatches to exact or RaBitQ search based on strategy, then applies
/// defense-in-depth filtering (IdReverse + VecMeta lifecycle check).
///
/// Used by `Processor::search_with_config()`.
pub(crate) fn search_with_strategy(
    storage: &Storage,
    index: &hnsw::Index,
    encoder: Option<&RaBitQ>,
    code_cache: &BinaryCodeCache,
    embedding_code: EmbeddingCode,
    strategy: &SearchStrategy,
    query: &[f32],
    k: usize,
    ef: usize,
    rerank_factor: usize,
) -> Result<Vec<SearchResult>> {
    let overfetch_k = k * 2;
    let effective_ef = ef.max(overfetch_k);

    let raw_results = match strategy {
        SearchStrategy::Exact => {
            index.search(storage, query, overfetch_k, effective_ef)?
        }
        SearchStrategy::RaBitQ { use_cache } => {
            if *use_cache {
                let enc = encoder
                    .ok_or_else(|| anyhow::anyhow!("RaBitQ encoder not available"))?;
                index.search_with_rabitq_cached(
                    storage,
                    query,
                    enc,
                    code_cache,
                    overfetch_k,
                    effective_ef,
                    rerank_factor,
                )?
            } else {
                // RaBitQ without cache - fall back to exact
                index.search(storage, query, overfetch_k, effective_ef)?
            }
        }
    };

    // Filter with defense-in-depth (IdReverse + VecMeta)
    filter_and_resolve_with_meta(storage, &raw_results, embedding_code, k)
}

/// Scan pending vectors with brute-force distance computation.
///
/// Iterates the Pending CF for the given embedding and computes exact
/// distances for up to `limit` pending vectors. Results can be merged
/// with HNSW results to ensure immediate searchability of newly inserted
/// vectors.
///
/// Returns (distance, vec_id, external_key) tuples sorted by distance ascending.
pub(crate) fn scan_pending_vectors(
    storage: &Storage,
    embedding: &Embedding,
    query: &[f32],
    limit: usize,
) -> Result<Vec<(f32, VecId, ExternalKey)>> {
    if limit == 0 {
        return Ok(Vec::new());
    }

    let embedding_code = embedding.code();
    let txn_db = storage.transaction_db()?;

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

        // Check VecMeta lifecycle - skip if deleted
        let meta_key = VecMetaCfKey(embedding_code, vec_id);
        if let Some(meta_bytes) = txn_db.get_cf(&meta_cf, VecMeta::key_to_bytes(&meta_key))? {
            let meta = VecMeta::value_from_bytes(&meta_bytes)?.0;
            if meta.is_deleted() {
                continue;
            }
        }

        // Load vector data from Vectors CF using proper storage_type
        let vec_key = VectorCfKey(embedding_code, vec_id);
        let vec_bytes = match txn_db.get_cf(&vectors_cf, Vectors::key_to_bytes(&vec_key))? {
            Some(bytes) => bytes,
            None => continue,
        };
        let vector_data = Vectors::value_from_bytes_typed(&vec_bytes, storage_type)?;

        // Compute exact distance
        let distance = embedding.compute_distance(query, &vector_data);

        // Look up external key via IdReverse
        let reverse_key = IdReverseCfKey(embedding_code, vec_id);
        let external_key =
            match txn_db.get_cf(&reverse_cf, IdReverse::key_to_bytes(&reverse_key))? {
                Some(bytes) => IdReverse::value_from_bytes(&bytes)?.0,
                None => continue,
            };

        results.push((distance, vec_id, external_key));
    }

    // Sort by distance ascending
    results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    Ok(results)
}

/// Merge HNSW results with pending scan results, deduplicate, and truncate.
///
/// Combines results from HNSW index search and pending vector scan,
/// removes duplicates (preferring HNSW results), re-sorts by distance,
/// and truncates to k results.
pub(crate) fn merge_results(
    hnsw_results: &mut Vec<SearchResult>,
    pending_results: Vec<(f32, VecId, ExternalKey)>,
    embedding_code: EmbeddingCode,
    k: usize,
) {
    if pending_results.is_empty() {
        hnsw_results.truncate(k);
        return;
    }

    // Track vec_ids already in HNSW results to avoid duplicates
    let hnsw_vec_ids: std::collections::HashSet<VecId> =
        hnsw_results.iter().map(|r| r.vec_id).collect();

    // Add pending results that aren't duplicates
    for (distance, vec_id, external_key) in pending_results {
        if !hnsw_vec_ids.contains(&vec_id) {
            hnsw_results.push(SearchResult {
                embedding_code,
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

    hnsw_results.truncate(k);
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Filter raw HNSW results using IdReverse lookup only.
///
/// Simple tombstone filter: vectors without an IdReverse entry are deleted.
/// Used by `search_exact()`.
fn filter_and_resolve(
    storage: &Storage,
    raw_results: &[(f32, VecId)],
    embedding_code: EmbeddingCode,
    k: usize,
) -> Result<Vec<SearchResult>> {
    let txn_db = storage.transaction_db()?;
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
    let key_refs: Vec<_> = keys
        .iter()
        .map(|k: &Vec<u8>| (&reverse_cf, k.as_slice()))
        .collect();
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
                embedding_code,
                external_key,
                vec_id,
                distance,
            });
        }
        // Skip deleted vectors (IdReverse missing or error)
    }

    Ok(results)
}

/// Filter raw HNSW results using IdReverse + VecMeta defense-in-depth.
///
/// Two-layer tombstone filter:
/// 1. IdReverse: deleted vectors have no reverse mapping
/// 2. VecMeta: defense-in-depth lifecycle check (catches edge cases)
///
/// Used by `search_with_strategy()`.
fn filter_and_resolve_with_meta(
    storage: &Storage,
    raw_results: &[(f32, VecId)],
    embedding_code: EmbeddingCode,
    k: usize,
) -> Result<Vec<SearchResult>> {
    let txn_db = storage.transaction_db()?;
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

    // Batch fetch VecMeta for defense-in-depth
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

    // Filter results
    let mut results = Vec::with_capacity(k * 2);
    for (i, value_result) in reverse_values.into_iter().enumerate() {
        if let Ok(Some(bytes)) = value_result {
            // Primary filter passed: IdReverse exists
            // Defense-in-depth: check VecMeta lifecycle
            if let Ok(Some(meta_bytes)) = &meta_values[i] {
                if let Ok(meta) = VecMeta::value_from_bytes(meta_bytes) {
                    if meta.0.is_deleted() {
                        continue;
                    }
                }
            }

            let external_key = IdReverse::value_from_bytes(&bytes)?.0;
            let (distance, vec_id) = raw_results[i];
            results.push(SearchResult {
                embedding_code,
                external_key,
                vec_id,
                distance,
            });
        }
    }

    Ok(results)
}
