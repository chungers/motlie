//! HNSW (Hierarchical Navigable Small World) graph algorithm.
//!
//! This module implements the core HNSW algorithm for approximate nearest neighbor search:
//! - Insert: Add vectors with hierarchical layer assignment
//! - Search: Layer descent with beam search at layer 0
//!
//! # Algorithm Overview
//!
//! ```text
//! Insert(v):
//!   1. Assign random layer l ~ exp(-l / m_L)
//!   2. Descend from entry point to layer l+1
//!   3. At each layer l..0: find neighbors, connect bidirectionally
//!   4. Update entry point if l > max_layer
//!
//! Search(q, k, ef):
//!   1. Start at entry point (highest layer)
//!   2. Greedy descent: at each layer, find closest neighbor
//!   3. At layer 0: beam search with ef candidates
//!   4. Return top-k from beam search results
//! ```
//!
//! # References
//!
//! - [HNSW Paper](https://arxiv.org/abs/1603.09320)
//! - [HNSW2 Optimization](examples/vector/HNSW2.md)

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use anyhow::{Context, Result};
use ordered_float::OrderedFloat;
use roaring::RoaringBitmap;

use super::config::HnswConfig;
use super::merge::EdgeOp;
use super::navigation::{NavigationCache, NavigationLayerInfo};
use super::schema::{
    EdgeCfKey, Edges, EmbeddingCode, GraphMeta, GraphMetaCfKey, GraphMetaCfValue, GraphMetaField,
    HnswLayer, VecId, VecMeta, VecMetaCfKey, VecMetaCfValue, VecMetadata, VectorCfKey, Vectors,
};
use super::Storage;
use crate::rocksdb::{ColumnFamily, HotColumnFamilyRecord};

// ============================================================================
// HNSW Index
// ============================================================================

/// HNSW index for a single embedding space.
///
/// Provides insert and search operations backed by RocksDB storage.
pub struct HnswIndex {
    /// The embedding space code this index manages
    embedding: EmbeddingCode,
    /// HNSW configuration (M, ef_construction, etc.)
    config: HnswConfig,
    /// Navigation cache for fast layer traversal
    nav_cache: Arc<NavigationCache>,
}

impl HnswIndex {
    /// Create a new HNSW index for an embedding space.
    pub fn new(
        embedding: EmbeddingCode,
        config: HnswConfig,
        nav_cache: Arc<NavigationCache>,
    ) -> Self {
        Self {
            embedding,
            config,
            nav_cache,
        }
    }

    /// Get the embedding code.
    pub fn embedding(&self) -> EmbeddingCode {
        self.embedding
    }

    /// Get the configuration.
    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    // =========================================================================
    // Insert
    // =========================================================================

    /// Insert a vector into the HNSW index.
    ///
    /// # Arguments
    /// * `storage` - RocksDB storage handle
    /// * `vec_id` - Internal vector ID (already allocated)
    /// * `vector` - The vector data (already stored in Vectors CF)
    ///
    /// # Algorithm
    /// 1. Assign random layer using HNSW distribution
    /// 2. Get or initialize navigation info
    /// 3. If graph is empty, just set as entry point
    /// 4. Otherwise: greedy descent from entry point to target layer
    /// 5. At each layer from target down to 0: find and connect neighbors
    /// 6. Update entry point if this node has higher layer
    pub fn insert(&self, storage: &Storage, vec_id: VecId, vector: &[f32]) -> Result<()> {
        let txn_db = storage.transaction_db()?;

        // 1. Assign random layer
        let mut rng = rand::thread_rng();
        let node_layer = self
            .nav_cache
            .get(self.embedding)
            .map(|info| info.random_layer(&mut rng))
            .unwrap_or(0);

        // 2. Store node metadata
        self.store_vec_meta(txn_db, vec_id, node_layer)?;

        // 3. Get current navigation state
        let nav_info = self.get_or_init_navigation(storage)?;

        // 4. Handle empty graph case
        if nav_info.is_empty() {
            self.init_first_node(storage, vec_id, node_layer)?;
            return Ok(());
        }

        // 5. Greedy descent to find entry point at target layer
        let entry_point = nav_info.entry_point().unwrap();
        let max_layer = nav_info.max_layer;

        let mut current = entry_point;
        let mut current_dist = self.distance(storage, vector, current)?;

        // Descend from max_layer to node_layer + 1 (greedy, single candidate)
        for layer in (node_layer + 1..=max_layer).rev() {
            (current, current_dist) =
                self.greedy_search_layer(storage, vector, current, current_dist, layer)?;
        }

        // 6. At each layer from node_layer down to 0: find neighbors and connect
        for layer in (0..=node_layer).rev() {
            // Search for neighbors at this layer
            let neighbors =
                self.search_layer(storage, vector, current, self.config.ef_construction, layer)?;

            // Select M neighbors (simple heuristic: take closest)
            let m = if layer == 0 {
                self.config.m * 2
            } else {
                self.config.m
            };
            let selected: Vec<_> = neighbors.into_iter().take(m as usize).collect();

            // Connect bidirectionally
            self.connect_neighbors(txn_db, vec_id, &selected, layer)?;

            // Update current for next layer
            if let Some(&(dist, id)) = selected.first() {
                current = id;
                current_dist = dist;
            }
        }

        // 7. Update entry point if needed
        if node_layer > max_layer {
            self.update_entry_point(storage, vec_id, node_layer)?;
        }

        // Update navigation cache
        self.nav_cache.update(self.embedding, self.config.m, |info| {
            info.maybe_update_entry(vec_id, node_layer);
            info.increment_layer_count(node_layer);
        });

        Ok(())
    }

    /// Initialize the first node in an empty graph.
    fn init_first_node(
        &self,
        storage: &Storage,
        vec_id: VecId,
        node_layer: HnswLayer,
    ) -> Result<()> {
        // Update navigation cache
        self.nav_cache.update(self.embedding, self.config.m, |info| {
            info.maybe_update_entry(vec_id, node_layer);
            info.increment_layer_count(0);
        });

        // Persist to GraphMeta
        self.update_entry_point(storage, vec_id, node_layer)?;

        Ok(())
    }

    /// Store vector metadata (layer assignment).
    fn store_vec_meta(
        &self,
        txn_db: &rocksdb::TransactionDB,
        vec_id: VecId,
        max_layer: HnswLayer,
    ) -> Result<()> {
        let key = VecMetaCfKey(self.embedding, vec_id);
        let value = VecMetaCfValue(VecMetadata {
            max_layer,
            flags: 0,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        });

        let cf = txn_db
            .cf_handle(VecMeta::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("VecMeta CF not found"))?;

        txn_db.put_cf(
            &cf,
            VecMeta::key_to_bytes(&key),
            VecMeta::value_to_bytes(&value)?,
        )?;

        Ok(())
    }

    /// Connect a node to its neighbors bidirectionally using merge operators.
    fn connect_neighbors(
        &self,
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
        let forward_key = Edges::key_to_bytes(&EdgeCfKey(self.embedding, vec_id, layer));
        let forward_op = EdgeOp::AddBatch(neighbor_ids.clone());
        txn_db.merge_cf(&cf, forward_key, forward_op.to_bytes())?;

        // Add reverse edges (each neighbor -> vec_id)
        for &neighbor_id in &neighbor_ids {
            let reverse_key = Edges::key_to_bytes(&EdgeCfKey(self.embedding, neighbor_id, layer));
            let reverse_op = EdgeOp::Add(vec_id);
            txn_db.merge_cf(&cf, reverse_key, reverse_op.to_bytes())?;
        }

        // TODO: Prune if degree exceeds M_max
        // This will be added when we implement degree-based pruning

        Ok(())
    }

    /// Update the global entry point.
    fn update_entry_point(
        &self,
        storage: &Storage,
        vec_id: VecId,
        layer: HnswLayer,
    ) -> Result<()> {
        let txn_db = storage.transaction_db()?;
        let cf = txn_db
            .cf_handle(GraphMeta::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("GraphMeta CF not found"))?;

        // Store entry point
        let ep_key = GraphMetaCfKey::entry_point(self.embedding);
        let ep_value = GraphMetaCfValue(GraphMetaField::EntryPoint(vec_id));
        txn_db.put_cf(&cf, GraphMeta::key_to_bytes(&ep_key), GraphMeta::value_to_bytes(&ep_value))?;

        // Store max level
        let level_key = GraphMetaCfKey::max_level(self.embedding);
        let level_value = GraphMetaCfValue(GraphMetaField::MaxLevel(layer));
        txn_db.put_cf(
            &cf,
            GraphMeta::key_to_bytes(&level_key),
            GraphMeta::value_to_bytes(&level_value),
        )?;

        Ok(())
    }

    // =========================================================================
    // Search
    // =========================================================================

    /// Search for k nearest neighbors.
    ///
    /// # Arguments
    /// * `storage` - RocksDB storage handle
    /// * `query` - Query vector
    /// * `k` - Number of results to return
    /// * `ef` - Search beam width (ef >= k for good recall)
    ///
    /// # Returns
    /// Vector of (distance, vec_id) pairs, sorted by distance ascending.
    pub fn search(
        &self,
        storage: &Storage,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<(f32, VecId)>> {
        let nav_info = self
            .nav_cache
            .get(self.embedding)
            .or_else(|| self.load_navigation(storage).ok())
            .ok_or_else(|| anyhow::anyhow!("No navigation info for embedding {}", self.embedding))?;

        if nav_info.is_empty() {
            return Ok(Vec::new());
        }

        let entry_point = nav_info.entry_point().unwrap();
        let max_layer = nav_info.max_layer;

        // Start from entry point
        let mut current = entry_point;
        let mut current_dist = self.distance(storage, query, current)?;

        // Greedy descent through layers (from max_layer down to 1)
        for layer in (1..=max_layer).rev() {
            (current, current_dist) =
                self.greedy_search_layer(storage, query, current, current_dist, layer)?;
        }

        // Beam search at layer 0
        let results = self.beam_search_layer0(storage, query, current, k, ef)?;

        Ok(results)
    }

    /// Greedy search at a single layer (for descent phase).
    ///
    /// Finds the single closest node by following edges greedily.
    /// Uses batch_distances for efficient I/O.
    fn greedy_search_layer(
        &self,
        storage: &Storage,
        query: &[f32],
        entry: VecId,
        entry_dist: f32,
        layer: HnswLayer,
    ) -> Result<(VecId, f32)> {
        let mut current = entry;
        let mut current_dist = entry_dist;

        loop {
            let neighbors = self.get_neighbors(storage, current, layer)?;
            if neighbors.is_empty() {
                break;
            }

            // Batch compute distances for all neighbors
            let neighbor_ids: Vec<VecId> = neighbors.iter().collect();
            let distances = self.batch_distances(storage, query, &neighbor_ids)?;

            // Find the closest neighbor
            let mut improved = false;
            for (neighbor, dist) in distances {
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

    /// Search at a single layer with ef candidates.
    ///
    /// Returns up to ef candidates sorted by distance.
    fn search_layer(
        &self,
        storage: &Storage,
        query: &[f32],
        entry: VecId,
        ef: usize,
        layer: HnswLayer,
    ) -> Result<Vec<(f32, VecId)>> {
        // For single-node graph, just return the entry
        let neighbors = self.get_neighbors(storage, entry, layer)?;
        if neighbors.is_empty() {
            let dist = self.distance(storage, query, entry)?;
            return Ok(vec![(dist, entry)]);
        }

        // Use beam search
        let results = self.beam_search(storage, query, entry, ef, layer)?;
        Ok(results)
    }

    /// Beam search at layer 0 for final candidate expansion.
    fn beam_search_layer0(
        &self,
        storage: &Storage,
        query: &[f32],
        entry: VecId,
        k: usize,
        ef: usize,
    ) -> Result<Vec<(f32, VecId)>> {
        self.beam_search(storage, query, entry, ef, 0)
            .map(|mut results| {
                results.truncate(k);
                results
            })
    }

    /// Batch size for beam search candidate processing.
    /// Larger batches = fewer I/O round trips, but may do more work.
    const BEAM_BATCH_SIZE: usize = 8;

    /// Generic beam search at any layer (batched version).
    ///
    /// Uses batch operations to reduce I/O round trips:
    /// - get_neighbors_batch: 1 I/O for all candidates in batch
    /// - batch_distances: 1 I/O for all neighbor vectors
    fn beam_search(
        &self,
        storage: &Storage,
        query: &[f32],
        entry: VecId,
        ef: usize,
        layer: HnswLayer,
    ) -> Result<Vec<(f32, VecId)>> {
        // Min-heap for candidates (closest first)
        let mut candidates: BinaryHeap<Reverse<(OrderedFloat<f32>, VecId)>> = BinaryHeap::new();
        // Max-heap for results (farthest first, for pruning)
        let mut results: BinaryHeap<(OrderedFloat<f32>, VecId)> = BinaryHeap::new();
        let mut visited = RoaringBitmap::new();

        let entry_dist = self.distance(storage, query, entry)?;
        candidates.push(Reverse((OrderedFloat(entry_dist), entry)));
        results.push((OrderedFloat(entry_dist), entry));
        visited.insert(entry);

        while !candidates.is_empty() {
            // Get current worst distance threshold
            let worst_dist = if results.len() >= ef {
                results.peek().map(|(OrderedFloat(d), _)| *d)
            } else {
                None
            };

            // Pop a batch of candidates, stopping if we hit the pruning threshold
            let mut batch: Vec<VecId> = Vec::with_capacity(Self::BEAM_BATCH_SIZE);
            while batch.len() < Self::BEAM_BATCH_SIZE {
                match candidates.pop() {
                    Some(Reverse((OrderedFloat(dist), node))) => {
                        // Check pruning condition
                        if let Some(worst) = worst_dist {
                            if dist > worst {
                                // This candidate and all remaining are too far, done
                                // Put it back and break out of the main loop
                                candidates.push(Reverse((OrderedFloat(dist), node)));
                                batch.clear(); // Clear batch to signal termination
                                break;
                            }
                        }
                        batch.push(node);
                    }
                    None => break,
                }
            }

            if batch.is_empty() {
                break;
            }

            // Batch fetch all neighbors for this batch
            let neighbor_maps = self.get_neighbors_batch(storage, &batch, layer)?;

            // Collect all unvisited neighbors
            let unvisited_neighbors: Vec<VecId> = neighbor_maps
                .iter()
                .flat_map(|(_, bitmap)| bitmap.iter())
                .filter(|&n| !visited.contains(n))
                .collect();

            // Mark as visited before fetching (to avoid duplicates in next batch)
            for &n in &unvisited_neighbors {
                visited.insert(n);
            }

            if unvisited_neighbors.is_empty() {
                continue;
            }

            // Batch compute distances for all unvisited neighbors
            let distances = self.batch_distances(storage, query, &unvisited_neighbors)?;

            // Update candidates and results
            for (neighbor, neighbor_dist) in distances {
                // Add to candidates if promising
                let should_add = results.len() < ef || {
                    let &(OrderedFloat(worst), _) = results.peek().unwrap();
                    neighbor_dist < worst
                };

                if should_add {
                    candidates.push(Reverse((OrderedFloat(neighbor_dist), neighbor)));
                    results.push((OrderedFloat(neighbor_dist), neighbor));

                    // Keep results bounded
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        // Convert to sorted vector
        let mut final_results: Vec<_> = results
            .into_iter()
            .map(|(OrderedFloat(d), id)| (d, id))
            .collect();
        final_results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        Ok(final_results)
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    /// Get or initialize navigation info from cache or storage.
    fn get_or_init_navigation(&self, storage: &Storage) -> Result<NavigationLayerInfo> {
        if let Some(info) = self.nav_cache.get(self.embedding) {
            return Ok(info);
        }

        // Try to load from storage
        if let Ok(info) = self.load_navigation(storage) {
            self.nav_cache.put(self.embedding, info.clone());
            return Ok(info);
        }

        // Initialize empty
        let info = NavigationLayerInfo::new(self.config.m);
        self.nav_cache.put(self.embedding, info.clone());
        Ok(info)
    }

    /// Load navigation info from GraphMeta CF.
    fn load_navigation(&self, storage: &Storage) -> Result<NavigationLayerInfo> {
        let txn_db = storage.transaction_db()?;
        let cf = txn_db
            .cf_handle(GraphMeta::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("GraphMeta CF not found"))?;

        // Load entry point
        let ep_key = GraphMetaCfKey::entry_point(self.embedding);
        let ep_bytes = txn_db
            .get_cf(&cf, GraphMeta::key_to_bytes(&ep_key))?
            .ok_or_else(|| anyhow::anyhow!("No entry point for embedding {}", self.embedding))?;
        let ep_value = GraphMeta::value_from_bytes(&ep_key, &ep_bytes)?;
        let entry_point = match ep_value.0 {
            GraphMetaField::EntryPoint(id) => id,
            _ => anyhow::bail!("Unexpected GraphMeta value type"),
        };

        // Load max level
        let level_key = GraphMetaCfKey::max_level(self.embedding);
        let level_bytes = txn_db
            .get_cf(&cf, GraphMeta::key_to_bytes(&level_key))?
            .ok_or_else(|| anyhow::anyhow!("No max level for embedding {}", self.embedding))?;
        let level_value = GraphMeta::value_from_bytes(&level_key, &level_bytes)?;
        let max_layer = match level_value.0 {
            GraphMetaField::MaxLevel(l) => l,
            _ => anyhow::bail!("Unexpected GraphMeta value type"),
        };

        // Build navigation info
        let mut info = NavigationLayerInfo::new(self.config.m);
        for _ in 0..=max_layer {
            info.entry_points.push(entry_point);
            info.layer_counts.push(0); // Counts not persisted yet
        }
        info.max_layer = max_layer;

        Ok(info)
    }

    /// Get neighbors of a node at a specific layer.
    ///
    /// For upper layers (>= min_cached_layer), uses the NavigationCache
    /// to avoid RocksDB reads during descent.
    fn get_neighbors(
        &self,
        storage: &Storage,
        vec_id: VecId,
        layer: HnswLayer,
    ) -> Result<RoaringBitmap> {
        // Use cached lookup for upper layers (Phase 3.5)
        self.nav_cache.get_neighbors_cached(
            self.embedding,
            vec_id,
            layer,
            || self.get_neighbors_uncached(storage, vec_id, layer),
        )
    }

    /// Get neighbors directly from RocksDB (no cache).
    fn get_neighbors_uncached(
        &self,
        storage: &Storage,
        vec_id: VecId,
        layer: HnswLayer,
    ) -> Result<RoaringBitmap> {
        let txn_db = storage.transaction_db()?;
        let cf = txn_db
            .cf_handle(Edges::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Edges CF not found"))?;

        let key = Edges::key_to_bytes(&EdgeCfKey(self.embedding, vec_id, layer));

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
        &self,
        storage: &Storage,
        vec_id: VecId,
        layer: HnswLayer,
    ) -> Result<u64> {
        let neighbors = self.get_neighbors(storage, vec_id, layer)?;
        Ok(neighbors.len())
    }

    /// Compute distance between query vector and stored vector.
    fn distance(&self, storage: &Storage, query: &[f32], vec_id: VecId) -> Result<f32> {
        let vector = self.get_vector(storage, vec_id)?;
        // Use L2 distance for now (TODO: use embedding's configured distance)
        Ok(l2_distance(query, &vector))
    }

    /// Load a vector from storage.
    fn get_vector(&self, storage: &Storage, vec_id: VecId) -> Result<Vec<f32>> {
        let txn_db = storage.transaction_db()?;
        let cf = txn_db
            .cf_handle(Vectors::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;

        let key = Vectors::key_to_bytes(&VectorCfKey(self.embedding, vec_id));
        let bytes = txn_db
            .get_cf(&cf, &key)?
            .ok_or_else(|| anyhow::anyhow!("Vector {} not found", vec_id))?;

        Vectors::value_from_bytes(&bytes).map(|v| v.0)
    }

    // =========================================================================
    // Batch Operations (Phase 3)
    // =========================================================================

    /// Batch fetch vectors using RocksDB's multi_get_cf.
    ///
    /// Returns vectors in the same order as input vec_ids. Missing vectors
    /// are represented as None.
    pub fn get_vectors_batch(
        &self,
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
            .map(|&id| Vectors::key_to_bytes(&VectorCfKey(self.embedding, id)))
            .collect();

        // Batch fetch using multi_get_cf
        let results: Vec<std::result::Result<Option<Vec<u8>>, rocksdb::Error>> =
            txn_db.multi_get_cf(keys.iter().map(|k| (&cf, k.as_slice())));

        // Deserialize vectors
        results
            .into_iter()
            .map(|result| {
                result
                    .map_err(|e| anyhow::anyhow!("RocksDB error: {}", e))?
                    .map(|bytes| Vectors::value_from_bytes(&bytes).map(|v| v.0))
                    .transpose()
            })
            .collect()
    }

    /// Batch compute distances from query to multiple vectors.
    ///
    /// Fetches all vectors in a single multi_get_cf call, then computes distances.
    /// Returns (vec_id, distance) pairs for vectors that exist.
    pub fn batch_distances(
        &self,
        storage: &Storage,
        query: &[f32],
        vec_ids: &[VecId],
    ) -> Result<Vec<(VecId, f32)>> {
        let vectors = self.get_vectors_batch(storage, vec_ids)?;

        Ok(vec_ids
            .iter()
            .copied()
            .zip(vectors.into_iter())
            .filter_map(|(id, vec_opt)| vec_opt.map(|v| (id, l2_distance(query, &v))))
            .collect())
    }

    /// Batch fetch neighbors for multiple nodes at a layer.
    ///
    /// Uses multi_get_cf for efficient batch lookup. Returns (vec_id, bitmap) pairs.
    pub fn get_neighbors_batch(
        &self,
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
            .map(|&id| Edges::key_to_bytes(&EdgeCfKey(self.embedding, id, layer)))
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
}

// ============================================================================
// Distance Functions
// ============================================================================

/// Compute L2 (Euclidean) squared distance between two vectors.
#[inline]
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((l2_distance(&a, &b) - 1.0).abs() < 0.0001);

        let c = vec![1.0, 1.0, 1.0];
        let d = vec![2.0, 2.0, 2.0];
        assert!((l2_distance(&c, &d) - 3.0).abs() < 0.0001);
    }

    #[test]
    fn test_hnsw_index_creation() {
        let config = HnswConfig::default();
        let nav_cache = Arc::new(NavigationCache::new());
        let index = HnswIndex::new(1, config.clone(), nav_cache);

        assert_eq!(index.embedding(), 1);
        assert_eq!(index.config().m, config.m);
    }

    // Integration tests require RocksDB storage
    // See integration test files for full end-to-end testing
}
