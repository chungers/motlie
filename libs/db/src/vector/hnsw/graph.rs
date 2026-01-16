//! HNSW graph operations: neighbors, distances, and batch operations.

use anyhow::{Context, Result};
use roaring::RoaringBitmap;
use std::collections::HashMap;

use super::Index;
use crate::rocksdb::ColumnFamily;
use crate::vector::distance::Distance;
use crate::vector::merge::EdgeOp;
use crate::vector::schema::{
    BinaryCodeCfKey, BinaryCodes, EdgeCfKey, Edges, HnswLayer, VecId, VectorCfKey, Vectors,
};
use crate::vector::Storage;

// ============================================================================
// Batch Edge Cache
// ============================================================================

/// In-memory cache for edges added during a batch transaction.
///
/// RocksDB's `Transaction::get()` does NOT include pending merge operands,
/// so edges written via `txn.merge_cf()` are not visible to reads within
/// the same transaction. This cache tracks edges added during batch insert
/// so that later inserts in the same batch can see edges from earlier inserts.
///
/// # Usage
///
/// ```ignore
/// let mut batch_cache = BatchEdgeCache::new();
///
/// for vector in vectors {
///     // Insert using the cache
///     let update = insert_in_txn_for_batch_with_cache(
///         index, txn, txn_db, storage, vec_id, vector, &mut batch_cache
///     )?;
///     update.apply(nav_cache);
/// }
/// ```
#[derive(Default)]
pub struct BatchEdgeCache {
    /// Forward and reverse edges: (vec_id, layer) -> neighbors
    edges: HashMap<(VecId, HnswLayer), RoaringBitmap>,
}

impl BatchEdgeCache {
    /// Create a new empty batch edge cache.
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
        }
    }

    /// Record edges added for a node at a layer (both directions).
    ///
    /// This records both forward edges (vec_id -> neighbors) and reverse edges
    /// (each neighbor -> vec_id) since HNSW connections are bidirectional.
    pub fn add_edges(&mut self, vec_id: VecId, layer: HnswLayer, neighbors: &[VecId]) {
        // Add forward edges
        let entry = self
            .edges
            .entry((vec_id, layer))
            .or_insert_with(RoaringBitmap::new);
        for &neighbor in neighbors {
            entry.insert(neighbor);
        }

        // Add reverse edges (bidirectional graph)
        for &neighbor in neighbors {
            let rev_entry = self
                .edges
                .entry((neighbor, layer))
                .or_insert_with(RoaringBitmap::new);
            rev_entry.insert(vec_id);
        }
    }

    /// Get cached edges for a node at a layer (if any).
    pub fn get_edges(&self, vec_id: VecId, layer: HnswLayer) -> Option<&RoaringBitmap> {
        self.edges.get(&(vec_id, layer))
    }

    /// Clear the cache (call after batch commit).
    pub fn clear(&mut self) {
        self.edges.clear();
    }
}

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
    index: &Index,
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
    index: &Index,
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

/// Get neighbors with batch edge cache support.
///
/// This function merges edges from RocksDB with edges from the batch cache.
/// Used during batch insert where edges written via `txn.merge_cf()` are not
/// visible to reads within the same transaction.
///
/// # Arguments
/// * `index` - The HNSW index
/// * `storage` - Storage for committed edge reads
/// * `vec_id` - The vector ID to get neighbors for
/// * `layer` - The HNSW layer
/// * `batch_cache` - Cache of edges added during the current batch
pub fn get_neighbors_with_batch_cache(
    index: &Index,
    storage: &Storage,
    vec_id: VecId,
    layer: HnswLayer,
    batch_cache: &BatchEdgeCache,
) -> Result<RoaringBitmap> {
    // Get committed edges from DB (uncached - build path)
    let mut neighbors = get_neighbors_uncached(index, storage, vec_id, layer)?;

    // Merge with cached edges from current batch
    if let Some(cached) = batch_cache.get_edges(vec_id, layer) {
        neighbors |= cached;
    }

    Ok(neighbors)
}

/// Get the number of neighbors (degree) for a node at a specific layer.
///
/// This uses `RoaringBitmap::len()` which is O(1) since the cardinality
/// is stored in the bitmap structure. Used for degree-based pruning checks.
pub fn get_neighbor_count(
    index: &Index,
    storage: &Storage,
    vec_id: VecId,
    layer: HnswLayer,
) -> Result<u64> {
    // Always uncached - this is typically called during build
    let neighbors = get_neighbors(index, storage, vec_id, layer, false)?;
    Ok(neighbors.len())
}

/// Connect a node to its neighbors bidirectionally within a transaction.
///
/// This is the recommended API for production use. All edge writes are
/// performed within the provided transaction for atomic commit.
///
/// # Arguments
/// * `index` - The HNSW index
/// * `txn` - RocksDB transaction to write within
/// * `txn_db` - TransactionDB for CF handle lookup
/// * `vec_id` - Vector ID to connect
/// * `neighbors` - Neighbor (distance, vec_id) pairs
/// * `layer` - HNSW layer for these edges
pub fn connect_neighbors_in_txn(
    index: &Index,
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
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
    txn.merge_cf(&cf, forward_key, forward_op.to_bytes())?;

    // Add reverse edges (each neighbor -> vec_id)
    for &neighbor_id in &neighbor_ids {
        let reverse_key = Edges::key_to_bytes(&EdgeCfKey(index.embedding(), neighbor_id, layer));
        let reverse_op = EdgeOp::Add(vec_id);
        txn.merge_cf(&cf, reverse_key, reverse_op.to_bytes())?;
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
pub fn distance(index: &Index, storage: &Storage, query: &[f32], vec_id: VecId) -> Result<f32> {
    let vector = get_vector(index, storage, vec_id)?;
    Ok(compute_distance_metric(index.distance_metric(), query, &vector))
}

/// Load a vector from storage.
fn get_vector(index: &Index, storage: &Storage, vec_id: VecId) -> Result<Vec<f32>> {
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

/// Compute distance using transaction for uncommitted reads.
///
/// Used during batch insert where vectors may not be committed yet.
pub fn distance_in_txn(
    index: &Index,
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    query: &[f32],
    vec_id: VecId,
) -> Result<f32> {
    let vector = get_vector_in_txn(index, txn, txn_db, vec_id)?;
    Ok(compute_distance_metric(index.distance_metric(), query, &vector))
}

/// Load a vector from transaction (can read uncommitted data).
///
/// Used during batch insert where vectors may not be committed yet.
fn get_vector_in_txn(
    index: &Index,
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    vec_id: VecId,
) -> Result<Vec<f32>> {
    let cf = txn_db
        .cf_handle(Vectors::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;

    let key = Vectors::key_to_bytes(&VectorCfKey(index.embedding(), vec_id));
    // Use txn.get_cf() to read uncommitted data from the transaction
    let bytes = txn
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
    index: &Index,
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
    index: &Index,
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
// Transaction-Aware Layer Search (for batch insert)
// ============================================================================

/// Greedy search at a single layer using transaction for uncommitted reads.
///
/// Used during batch insert where vectors may not be committed yet.
pub fn greedy_search_layer_in_txn(
    index: &Index,
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    storage: &Storage,
    query: &[f32],
    entry: VecId,
    entry_dist: f32,
    layer: HnswLayer,
) -> Result<(VecId, f32)> {
    let mut current = entry;
    let mut current_dist = entry_dist;

    loop {
        // use_cache: false for index build
        let neighbors = get_neighbors(index, storage, current, layer, false)?;
        if neighbors.is_empty() {
            break;
        }

        let mut improved = false;
        for neighbor in neighbors.iter() {
            let dist = distance_in_txn(index, txn, txn_db, query, neighbor)?;
            if dist < current_dist {
                current = neighbor;
                current_dist = dist;
                improved = true;
            }
        }

        if !improved {
            break;
        }
    }

    Ok((current, current_dist))
}

/// Search at a single layer with ef candidates using transaction.
///
/// Used during batch insert where vectors may not be committed yet.
/// This is a simplified version that doesn't use beam search for speed
/// during index build - it uses a simple greedy expansion instead.
pub fn search_layer_in_txn(
    index: &Index,
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    storage: &Storage,
    query: &[f32],
    entry: VecId,
    ef: usize,
    layer: HnswLayer,
) -> Result<Vec<(f32, VecId)>> {
    use std::collections::{BinaryHeap, HashSet};

    // For single-node graph, just return the entry
    let neighbors = get_neighbors(index, storage, entry, layer, false)?;
    if neighbors.is_empty() {
        let dist = distance_in_txn(index, txn, txn_db, query, entry)?;
        return Ok(vec![(dist, entry)]);
    }

    // Simple greedy search with ef candidates
    let entry_dist = distance_in_txn(index, txn, txn_db, query, entry)?;
    let mut visited = HashSet::new();
    visited.insert(entry);

    // Min-heap for candidates (closest first)
    let mut candidates: BinaryHeap<std::cmp::Reverse<(ordered_float::OrderedFloat<f32>, VecId)>> =
        BinaryHeap::new();
    candidates.push(std::cmp::Reverse((ordered_float::OrderedFloat(entry_dist), entry)));

    // Max-heap for results (furthest first for easy pruning)
    let mut results: BinaryHeap<(ordered_float::OrderedFloat<f32>, VecId)> = BinaryHeap::new();
    results.push((ordered_float::OrderedFloat(entry_dist), entry));

    while let Some(std::cmp::Reverse((dist, current))) = candidates.pop() {
        // Stop if current candidate is further than worst result
        if results.len() >= ef {
            if let Some(&(worst_dist, _)) = results.peek() {
                if dist > worst_dist {
                    break;
                }
            }
        }

        // Explore neighbors
        let neighbors = get_neighbors(index, storage, current, layer, false)?;
        for neighbor in neighbors.iter() {
            if visited.insert(neighbor) {
                let neighbor_dist = distance_in_txn(index, txn, txn_db, query, neighbor)?;

                // Add to results if better than worst
                let should_add = if results.len() < ef {
                    true
                } else if let Some(&(worst_dist, _)) = results.peek() {
                    ordered_float::OrderedFloat(neighbor_dist) < worst_dist
                } else {
                    true
                };

                if should_add {
                    candidates.push(std::cmp::Reverse((
                        ordered_float::OrderedFloat(neighbor_dist),
                        neighbor,
                    )));
                    results.push((ordered_float::OrderedFloat(neighbor_dist), neighbor));
                    if results.len() > ef {
                        results.pop(); // Remove worst
                    }
                }
            }
        }
    }

    // Convert to sorted vec
    let mut result_vec: Vec<(f32, VecId)> =
        results.into_iter().map(|(d, id)| (d.0, id)).collect();
    result_vec.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    Ok(result_vec)
}

// ============================================================================
// Batch-Cache-Aware Layer Search (for batch insert with edge visibility)
// ============================================================================

/// Greedy search at a single layer using transaction reads and batch edge cache.
///
/// This version merges edges from the batch cache so that later inserts in a
/// batch can see edges created by earlier inserts (which are not visible via
/// `txn.get_cf()` since they are pending merge operations).
pub fn greedy_search_layer_with_batch_cache(
    index: &Index,
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    storage: &Storage,
    query: &[f32],
    entry: VecId,
    entry_dist: f32,
    layer: HnswLayer,
    batch_cache: &BatchEdgeCache,
) -> Result<(VecId, f32)> {
    let mut current = entry;
    let mut current_dist = entry_dist;

    loop {
        // Use batch-cache-aware neighbor lookup
        let neighbors = get_neighbors_with_batch_cache(index, storage, current, layer, batch_cache)?;
        if neighbors.is_empty() {
            break;
        }

        let mut improved = false;
        for neighbor in neighbors.iter() {
            let dist = distance_in_txn(index, txn, txn_db, query, neighbor)?;
            if dist < current_dist {
                current = neighbor;
                current_dist = dist;
                improved = true;
            }
        }

        if !improved {
            break;
        }
    }

    Ok((current, current_dist))
}

/// Search at a single layer with ef candidates using transaction and batch edge cache.
///
/// This version merges edges from the batch cache so that later inserts in a
/// batch can see edges created by earlier inserts (which are not visible via
/// `txn.get_cf()` since they are pending merge operations).
pub fn search_layer_with_batch_cache(
    index: &Index,
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    storage: &Storage,
    query: &[f32],
    entry: VecId,
    ef: usize,
    layer: HnswLayer,
    batch_cache: &BatchEdgeCache,
) -> Result<Vec<(f32, VecId)>> {
    use std::collections::{BinaryHeap, HashSet};

    // For single-node graph, just return the entry
    let neighbors = get_neighbors_with_batch_cache(index, storage, entry, layer, batch_cache)?;
    if neighbors.is_empty() {
        let dist = distance_in_txn(index, txn, txn_db, query, entry)?;
        return Ok(vec![(dist, entry)]);
    }

    // Simple greedy search with ef candidates
    let entry_dist = distance_in_txn(index, txn, txn_db, query, entry)?;
    let mut visited = HashSet::new();
    visited.insert(entry);

    // Min-heap for candidates (closest first)
    let mut candidates: BinaryHeap<std::cmp::Reverse<(ordered_float::OrderedFloat<f32>, VecId)>> =
        BinaryHeap::new();
    candidates.push(std::cmp::Reverse((ordered_float::OrderedFloat(entry_dist), entry)));

    // Max-heap for results (furthest first for easy pruning)
    let mut results: BinaryHeap<(ordered_float::OrderedFloat<f32>, VecId)> = BinaryHeap::new();
    results.push((ordered_float::OrderedFloat(entry_dist), entry));

    while let Some(std::cmp::Reverse((dist, current))) = candidates.pop() {
        // Stop if current candidate is further than worst result
        if results.len() >= ef {
            if let Some(&(worst_dist, _)) = results.peek() {
                if dist > worst_dist {
                    break;
                }
            }
        }

        // Explore neighbors using batch-cache-aware lookup
        let neighbors = get_neighbors_with_batch_cache(index, storage, current, layer, batch_cache)?;
        for neighbor in neighbors.iter() {
            if visited.insert(neighbor) {
                let neighbor_dist = distance_in_txn(index, txn, txn_db, query, neighbor)?;

                // Add to results if better than worst
                let should_add = if results.len() < ef {
                    true
                } else if let Some(&(worst_dist, _)) = results.peek() {
                    ordered_float::OrderedFloat(neighbor_dist) < worst_dist
                } else {
                    true
                };

                if should_add {
                    candidates.push(std::cmp::Reverse((
                        ordered_float::OrderedFloat(neighbor_dist),
                        neighbor,
                    )));
                    results.push((ordered_float::OrderedFloat(neighbor_dist), neighbor));
                    if results.len() > ef {
                        results.pop(); // Remove worst
                    }
                }
            }
        }
    }

    // Convert to sorted vec
    let mut result_vec: Vec<(f32, VecId)> =
        results.into_iter().map(|(d, id)| (d.0, id)).collect();
    result_vec.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    Ok(result_vec)
}

// ============================================================================
// Batch Operations
// ============================================================================

/// Batch fetch vectors using RocksDB's multi_get_cf.
///
/// Returns vectors in the same order as input vec_ids. Missing vectors
/// are represented as None.
pub fn get_vectors_batch(
    index: &Index,
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
    index: &Index,
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
    index: &Index,
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
pub fn get_binary_code(index: &Index, storage: &Storage, vec_id: VecId) -> Result<Option<Vec<u8>>> {
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
    index: &Index,
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
