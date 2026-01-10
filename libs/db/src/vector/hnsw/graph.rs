//! HNSW graph operations: neighbors, distances, and batch operations.

use anyhow::{Context, Result};
use roaring::RoaringBitmap;

use super::HnswIndex;
use crate::rocksdb::ColumnFamily;
use crate::vector::distance::Distance;
use crate::vector::merge::EdgeOp;
use crate::vector::schema::{
    BinaryCodeCfKey, BinaryCodes, EdgeCfKey, Edges, HnswLayer, VecId, VectorCfKey, Vectors,
};
use crate::vector::Storage;

// ============================================================================
// Distance Functions
// ============================================================================

/// Compute L2 (Euclidean) squared distance between two vectors.
#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

/// Compute Cosine distance between two vectors.
/// Returns 1 - cosine_similarity, so 0 = identical, 2 = opposite.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0; // undefined, return neutral distance
    }

    1.0 - (dot / (norm_a * norm_b))
}

/// Compute negative dot product distance (for similarity ranking).
/// More negative = more similar (for MaxSim use cases).
/// We return negative so that lower values = more similar (consistent with L2/Cosine).
#[inline]
pub fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    -dot // Negate so lower = more similar
}

/// Compute distance between two vectors using the configured metric.
#[inline]
pub fn compute_distance_metric(distance: Distance, a: &[f32], b: &[f32]) -> f32 {
    match distance {
        Distance::L2 => l2_distance(a, b),
        Distance::Cosine => cosine_distance(a, b),
        Distance::DotProduct => dot_product_distance(a, b),
    }
}

// ============================================================================
// Neighbor Operations
// ============================================================================

/// Get neighbors of a node at a specific layer.
///
/// # Arguments
/// * `use_cache` - If true, uses NavigationCache for edge caching (for search).
///                 If false, reads directly from RocksDB (for index build).
///
/// Cache behavior when `use_cache` is true:
/// - Upper layers (>= 2): Full caching in HashMap
/// - Lower layers (0-1): FIFO hot cache with bounded size
pub fn get_neighbors(
    index: &HnswIndex,
    storage: &Storage,
    vec_id: VecId,
    layer: HnswLayer,
    use_cache: bool,
) -> Result<RoaringBitmap> {
    if use_cache {
        index.nav_cache().get_neighbors_cached(
            index.embedding(),
            vec_id,
            layer,
            || get_neighbors_uncached(index, storage, vec_id, layer),
        )
    } else {
        get_neighbors_uncached(index, storage, vec_id, layer)
    }
}

/// Get neighbors directly from RocksDB (no cache).
fn get_neighbors_uncached(
    index: &HnswIndex,
    storage: &Storage,
    vec_id: VecId,
    layer: HnswLayer,
) -> Result<RoaringBitmap> {
    let txn_db = storage.transaction_db()?;
    let cf = txn_db
        .cf_handle(Edges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Edges CF not found"))?;

    let key = Edges::key_to_bytes(&EdgeCfKey(index.embedding(), vec_id, layer));

    match txn_db.get_cf(&cf, &key)? {
        Some(bytes) => {
            let bitmap = RoaringBitmap::deserialize_from(&bytes[..])
                .context("Failed to deserialize edge bitmap")?;
            Ok(bitmap)
        }
        None => Ok(RoaringBitmap::new()),
    }
}

/// Get the number of neighbors (degree) for a node at a specific layer.
///
/// This uses `RoaringBitmap::len()` which is O(1) since the cardinality
/// is stored in the bitmap structure. Used for degree-based pruning checks.
pub fn get_neighbor_count(
    index: &HnswIndex,
    storage: &Storage,
    vec_id: VecId,
    layer: HnswLayer,
) -> Result<u64> {
    // Always uncached - this is typically called during build
    let neighbors = get_neighbors(index, storage, vec_id, layer, false)?;
    Ok(neighbors.len())
}

/// Connect a node to its neighbors bidirectionally using merge operators.
pub fn connect_neighbors(
    index: &HnswIndex,
    txn_db: &rocksdb::TransactionDB,
    vec_id: VecId,
    neighbors: &[(f32, VecId)],
    layer: HnswLayer,
) -> Result<()> {
    let cf = txn_db
        .cf_handle(Edges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Edges CF not found"))?;

    // Add forward edges (vec_id -> neighbors)
    let neighbor_ids: Vec<u32> = neighbors.iter().map(|(_, id)| *id).collect();
    let forward_key = Edges::key_to_bytes(&EdgeCfKey(index.embedding(), vec_id, layer));
    let forward_op = EdgeOp::AddBatch(neighbor_ids.clone());
    txn_db.merge_cf(&cf, forward_key, forward_op.to_bytes())?;

    // Add reverse edges (each neighbor -> vec_id)
    for &neighbor_id in &neighbor_ids {
        let reverse_key = Edges::key_to_bytes(&EdgeCfKey(index.embedding(), neighbor_id, layer));
        let reverse_op = EdgeOp::Add(vec_id);
        txn_db.merge_cf(&cf, reverse_key, reverse_op.to_bytes())?;
    }

    // Note: No cache invalidation needed - index build uses uncached paths,
    // and the cache is only populated during search operations.

    // TODO: Prune if degree exceeds M_max
    // This will be added when we implement degree-based pruning

    Ok(())
}

// ============================================================================
// Distance Computation
// ============================================================================

/// Compute distance between query vector and stored vector.
pub fn distance(index: &HnswIndex, storage: &Storage, query: &[f32], vec_id: VecId) -> Result<f32> {
    let vector = get_vector(index, storage, vec_id)?;
    Ok(compute_distance_metric(index.distance_metric(), query, &vector))
}

/// Load a vector from storage.
fn get_vector(index: &HnswIndex, storage: &Storage, vec_id: VecId) -> Result<Vec<f32>> {
    let txn_db = storage.transaction_db()?;
    let cf = txn_db
        .cf_handle(Vectors::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;

    let key = Vectors::key_to_bytes(&VectorCfKey(index.embedding(), vec_id));
    let bytes = txn_db
        .get_cf(&cf, &key)?
        .ok_or_else(|| anyhow::anyhow!("Vector {} not found", vec_id))?;

    Vectors::value_from_bytes_typed(&bytes, index.storage_type())
}

// ============================================================================
// Layer Search Operations
// ============================================================================

/// Greedy search at a single layer (for descent phase).
///
/// Finds the single closest node by following edges greedily.
/// Uses batch_distances when neighbor count exceeds threshold.
///
/// # Arguments
/// * `use_cache` - If true, uses edge cache (for search). If false, uncached (for build).
pub fn greedy_search_layer(
    index: &HnswIndex,
    storage: &Storage,
    query: &[f32],
    entry: VecId,
    entry_dist: f32,
    layer: HnswLayer,
    use_cache: bool,
) -> Result<(VecId, f32)> {
    let mut current = entry;
    let mut current_dist = entry_dist;

    loop {
        let neighbors = get_neighbors(index, storage, current, layer, use_cache)?;
        if neighbors.is_empty() {
            break;
        }

        let mut improved = false;

        // Use batch for larger neighbor sets, individual for small
        // Threshold is configurable - default 64 effectively disables batching
        if neighbors.len() as u64 >= index.config().batch_threshold as u64 {
            let neighbor_ids: Vec<VecId> = neighbors.iter().collect();
            let distances = batch_distances(index, storage, query, &neighbor_ids)?;
            for (neighbor, dist) in distances {
                if dist < current_dist {
                    current = neighbor;
                    current_dist = dist;
                    improved = true;
                }
            }
        } else {
            for neighbor in neighbors.iter() {
                let dist = distance(index, storage, query, neighbor)?;
                if dist < current_dist {
                    current = neighbor;
                    current_dist = dist;
                    improved = true;
                }
            }
        }

        if !improved {
            break;
        }
    }

    Ok((current, current_dist))
}

/// Search at a single layer with ef candidates.
///
/// Returns up to ef candidates sorted by distance.
///
/// # Arguments
/// * `use_cache` - If true, uses edge cache (for search). If false, uncached (for build).
pub fn search_layer(
    index: &HnswIndex,
    storage: &Storage,
    query: &[f32],
    entry: VecId,
    ef: usize,
    layer: HnswLayer,
    use_cache: bool,
) -> Result<Vec<(f32, VecId)>> {
    // For single-node graph, just return the entry
    let neighbors = get_neighbors(index, storage, entry, layer, use_cache)?;
    if neighbors.is_empty() {
        let dist = distance(index, storage, query, entry)?;
        return Ok(vec![(dist, entry)]);
    }

    // Use beam search
    let results = super::search::beam_search(index, storage, query, entry, ef, layer, use_cache)?;
    Ok(results)
}

// ============================================================================
// Batch Operations
// ============================================================================

/// Batch fetch vectors using RocksDB's multi_get_cf.
///
/// Returns vectors in the same order as input vec_ids. Missing vectors
/// are represented as None.
pub fn get_vectors_batch(
    index: &HnswIndex,
    storage: &Storage,
    vec_ids: &[VecId],
) -> Result<Vec<Option<Vec<f32>>>> {
    if vec_ids.is_empty() {
        return Ok(Vec::new());
    }

    let txn_db = storage.transaction_db()?;
    let cf = txn_db
        .cf_handle(Vectors::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;

    // Build keys
    let keys: Vec<Vec<u8>> = vec_ids
        .iter()
        .map(|&id| Vectors::key_to_bytes(&VectorCfKey(index.embedding(), id)))
        .collect();

    // Batch fetch using multi_get_cf
    let results: Vec<std::result::Result<Option<Vec<u8>>, rocksdb::Error>> =
        txn_db.multi_get_cf(keys.iter().map(|k| (&cf, k.as_slice())));

    // Deserialize vectors using configured storage type
    let storage_type = index.storage_type();
    results
        .into_iter()
        .map(|result| {
            result
                .map_err(|e| anyhow::anyhow!("RocksDB error: {}", e))?
                .map(|bytes| Vectors::value_from_bytes_typed(&bytes, storage_type))
                .transpose()
        })
        .collect()
}

/// Batch compute distances from query to multiple vectors.
///
/// Fetches all vectors in a single multi_get_cf call, then computes distances
/// in parallel using rayon. Uses the configured distance metric (L2, Cosine, DotProduct).
/// Returns (vec_id, distance) pairs for vectors that exist.
pub fn batch_distances(
    index: &HnswIndex,
    storage: &Storage,
    query: &[f32],
    vec_ids: &[VecId],
) -> Result<Vec<(VecId, f32)>> {
    use rayon::prelude::*;

    let vectors = get_vectors_batch(index, storage, vec_ids)?;

    // Collect (id, vector) pairs for parallel processing
    let id_vec_pairs: Vec<(VecId, Vec<f32>)> = vec_ids
        .iter()
        .copied()
        .zip(vectors.into_iter())
        .filter_map(|(id, vec_opt)| vec_opt.map(|v| (id, v)))
        .collect();

    // Parallel distance computation (CPU-bound, benefits from rayon)
    let distance_metric = index.distance_metric();
    Ok(id_vec_pairs
        .par_iter()
        .map(|(id, v)| (*id, compute_distance_metric(distance_metric, query, v)))
        .collect())
}

/// Batch fetch neighbors for multiple nodes at a layer.
///
/// Uses multi_get_cf for efficient batch lookup. Returns (vec_id, bitmap) pairs.
pub fn get_neighbors_batch(
    index: &HnswIndex,
    storage: &Storage,
    vec_ids: &[VecId],
    layer: HnswLayer,
) -> Result<Vec<(VecId, RoaringBitmap)>> {
    if vec_ids.is_empty() {
        return Ok(Vec::new());
    }

    let txn_db = storage.transaction_db()?;
    let cf = txn_db
        .cf_handle(Edges::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Edges CF not found"))?;

    // Build keys
    let keys: Vec<Vec<u8>> = vec_ids
        .iter()
        .map(|&id| Edges::key_to_bytes(&EdgeCfKey(index.embedding(), id, layer)))
        .collect();

    // Batch fetch
    let results: Vec<std::result::Result<Option<Vec<u8>>, rocksdb::Error>> =
        txn_db.multi_get_cf(keys.iter().map(|k| (&cf, k.as_slice())));

    // Deserialize bitmaps
    vec_ids
        .iter()
        .copied()
        .zip(results.into_iter())
        .map(|(id, result)| {
            let bitmap = result
                .map_err(|e| anyhow::anyhow!("RocksDB error: {}", e))?
                .map(|bytes| {
                    RoaringBitmap::deserialize_from(&bytes[..])
                        .context("Failed to deserialize edge bitmap")
                })
                .transpose()?
                .unwrap_or_default();
            Ok((id, bitmap))
        })
        .collect()
}

// ============================================================================
// Binary Code Operations
// ============================================================================

/// Get a binary code for a vector.
pub fn get_binary_code(index: &HnswIndex, storage: &Storage, vec_id: VecId) -> Result<Option<Vec<u8>>> {
    let txn_db = storage.transaction_db()?;
    let cf = txn_db
        .cf_handle(BinaryCodes::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("BinaryCodes CF not found"))?;

    let key = BinaryCodeCfKey(index.embedding(), vec_id);
    let key_bytes = BinaryCodes::key_to_bytes(&key);

    match txn_db.get_cf(&cf, key_bytes)? {
        Some(bytes) => Ok(Some(bytes.to_vec())),
        None => Ok(None),
    }
}

/// Batch retrieve binary codes for multiple vectors.
pub fn get_binary_codes_batch(
    index: &HnswIndex,
    storage: &Storage,
    vec_ids: &[VecId],
) -> Result<Vec<Option<Vec<u8>>>> {
    let txn_db = storage.transaction_db()?;
    let cf = txn_db
        .cf_handle(BinaryCodes::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("BinaryCodes CF not found"))?;

    let keys: Vec<Vec<u8>> = vec_ids
        .iter()
        .map(|&id| BinaryCodes::key_to_bytes(&BinaryCodeCfKey(index.embedding(), id)))
        .collect();

    let results: Vec<std::result::Result<Option<Vec<u8>>, rocksdb::Error>> =
        txn_db.multi_get_cf(keys.iter().map(|k| (&cf, k.as_slice())));

    results
        .into_iter()
        .map(|r| r.map(|opt| opt.map(|b| b.to_vec())).map_err(|e| anyhow::anyhow!("RocksDB error: {}", e)))
        .collect()
}
