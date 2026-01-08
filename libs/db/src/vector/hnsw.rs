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
use super::rabitq::RaBitQ;
use super::schema::{
    BinaryCodeCfKey, BinaryCodes, EdgeCfKey, Edges, EmbeddingCode, GraphMeta, GraphMetaCfKey,
    GraphMetaCfValue, GraphMetaField, HnswLayer, VecId, VecMeta, VecMetaCfKey, VecMetaCfValue,
    VecMetadata, VectorCfKey, Vectors,
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
        // use_cache: false - index build should not use cache
        for layer in (node_layer + 1..=max_layer).rev() {
            (current, current_dist) =
                self.greedy_search_layer(storage, vector, current, current_dist, layer, false)?;
        }

        // 6. At each layer from node_layer down to 0: find neighbors and connect
        for layer in (0..=node_layer).rev() {
            // Search for neighbors at this layer (use_cache: false for build)
            let neighbors =
                self.search_layer(storage, vector, current, self.config.ef_construction, layer, false)?;

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

        // Note: No cache invalidation needed - index build uses uncached paths,
        // and the cache is only populated during search operations.

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
        // use_cache: true - search benefits from edge caching
        for layer in (1..=max_layer).rev() {
            (current, current_dist) =
                self.greedy_search_layer(storage, query, current, current_dist, layer, true)?;
        }

        // Beam search at layer 0 (use_cache: true for search)
        let results = self.beam_search_layer0(storage, query, current, k, ef, true)?;

        Ok(results)
    }

    /// Greedy search at a single layer (for descent phase).
    ///
    /// Finds the single closest node by following edges greedily.
    /// Uses batch_distances when neighbor count exceeds threshold.
    ///
    /// # Arguments
    /// * `use_cache` - If true, uses edge cache (for search). If false, uncached (for build).
    fn greedy_search_layer(
        &self,
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
            let neighbors = self.get_neighbors(storage, current, layer, use_cache)?;
            if neighbors.is_empty() {
                break;
            }

            let mut improved = false;

            // Use batch for larger neighbor sets, individual for small
            // Threshold is configurable - default 64 effectively disables batching
            if neighbors.len() as u64 >= self.config.batch_threshold as u64 {
                let neighbor_ids: Vec<VecId> = neighbors.iter().collect();
                let distances = self.batch_distances(storage, query, &neighbor_ids)?;
                for (neighbor, dist) in distances {
                    if dist < current_dist {
                        current = neighbor;
                        current_dist = dist;
                        improved = true;
                    }
                }
            } else {
                for neighbor in neighbors.iter() {
                    let dist = self.distance(storage, query, neighbor)?;
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
    fn search_layer(
        &self,
        storage: &Storage,
        query: &[f32],
        entry: VecId,
        ef: usize,
        layer: HnswLayer,
        use_cache: bool,
    ) -> Result<Vec<(f32, VecId)>> {
        // For single-node graph, just return the entry
        let neighbors = self.get_neighbors(storage, entry, layer, use_cache)?;
        if neighbors.is_empty() {
            let dist = self.distance(storage, query, entry)?;
            return Ok(vec![(dist, entry)]);
        }

        // Use beam search
        let results = self.beam_search(storage, query, entry, ef, layer, use_cache)?;
        Ok(results)
    }

    /// Beam search at layer 0 for final candidate expansion.
    ///
    /// # Arguments
    /// * `use_cache` - If true, uses edge cache (for search). If false, uncached (for build).
    fn beam_search_layer0(
        &self,
        storage: &Storage,
        query: &[f32],
        entry: VecId,
        k: usize,
        ef: usize,
        use_cache: bool,
    ) -> Result<Vec<(f32, VecId)>> {
        self.beam_search(storage, query, entry, ef, 0, use_cache)
            .map(|mut results| {
                results.truncate(k);
                results
            })
    }

    /// Generic beam search at any layer.
    ///
    /// Uses batch_distances for efficient neighbor distance computation
    /// while maintaining correct sequential candidate processing.
    ///
    /// # Arguments
    /// * `use_cache` - If true, uses edge cache (for search). If false, uncached (for build).
    fn beam_search(
        &self,
        storage: &Storage,
        query: &[f32],
        entry: VecId,
        ef: usize,
        layer: HnswLayer,
        use_cache: bool,
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

        while let Some(Reverse((OrderedFloat(dist), node))) = candidates.pop() {
            // If this candidate is farther than the worst result, stop
            if results.len() >= ef {
                if let Some(&(OrderedFloat(worst), _)) = results.peek() {
                    if dist > worst {
                        break;
                    }
                }
            }

            // Get neighbors for this candidate
            let neighbors = self.get_neighbors(storage, node, layer, use_cache)?;

            // Collect unvisited neighbors
            let unvisited: Vec<VecId> = neighbors
                .iter()
                .filter(|&n| !visited.contains(n))
                .collect();

            // Mark as visited
            for &n in &unvisited {
                visited.insert(n);
            }

            if unvisited.is_empty() {
                continue;
            }

            // Compute distances - use batch for larger sets
            // Threshold is configurable - default 64 effectively disables batching
            let distances: Vec<(VecId, f32)> = if unvisited.len() >= self.config.batch_threshold {
                self.batch_distances(storage, query, &unvisited)?
            } else {
                unvisited
                    .iter()
                    .filter_map(|&n| {
                        self.distance(storage, query, n)
                            .ok()
                            .map(|d| (n, d))
                    })
                    .collect()
            };

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
    /// # Arguments
    /// * `use_cache` - If true, uses NavigationCache for edge caching (for search).
    ///                 If false, reads directly from RocksDB (for index build).
    ///
    /// Cache behavior when `use_cache` is true:
    /// - Upper layers (>= 2): Full caching in HashMap
    /// - Lower layers (0-1): FIFO hot cache with bounded size
    fn get_neighbors(
        &self,
        storage: &Storage,
        vec_id: VecId,
        layer: HnswLayer,
        use_cache: bool,
    ) -> Result<RoaringBitmap> {
        if use_cache {
            self.nav_cache.get_neighbors_cached(
                self.embedding,
                vec_id,
                layer,
                || self.get_neighbors_uncached(storage, vec_id, layer),
            )
        } else {
            self.get_neighbors_uncached(storage, vec_id, layer)
        }
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
        // Always uncached - this is typically called during build
        let neighbors = self.get_neighbors(storage, vec_id, layer, false)?;
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

    // =========================================================================
    // RaBitQ Methods
    // =========================================================================

    /// Get a binary code for a vector.
    pub fn get_binary_code(&self, storage: &Storage, vec_id: VecId) -> Result<Option<Vec<u8>>> {
        let txn_db = storage.transaction_db()?;
        let cf = txn_db
            .cf_handle(BinaryCodes::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("BinaryCodes CF not found"))?;

        let key = BinaryCodeCfKey(self.embedding, vec_id);
        let key_bytes = BinaryCodes::key_to_bytes(&key);

        match txn_db.get_cf(&cf, key_bytes)? {
            Some(bytes) => Ok(Some(bytes.to_vec())),
            None => Ok(None),
        }
    }

    /// Batch retrieve binary codes for multiple vectors.
    pub fn get_binary_codes_batch(
        &self,
        storage: &Storage,
        vec_ids: &[VecId],
    ) -> Result<Vec<Option<Vec<u8>>>> {
        let txn_db = storage.transaction_db()?;
        let cf = txn_db
            .cf_handle(BinaryCodes::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("BinaryCodes CF not found"))?;

        let keys: Vec<Vec<u8>> = vec_ids
            .iter()
            .map(|&id| BinaryCodes::key_to_bytes(&BinaryCodeCfKey(self.embedding, id)))
            .collect();

        let results: Vec<std::result::Result<Option<Vec<u8>>, rocksdb::Error>> =
            txn_db.multi_get_cf(keys.iter().map(|k| (&cf, k.as_slice())));

        results
            .into_iter()
            .map(|r| r.map(|opt| opt.map(|b| b.to_vec())).map_err(|e| anyhow::anyhow!("RocksDB error: {}", e)))
            .collect()
    }

    /// Two-phase search using RaBitQ for initial filtering.
    ///
    /// Phase 1: Use standard HNSW search to find candidates
    /// Phase 2: Re-rank candidates with exact distance
    ///
    /// # Arguments
    ///
    /// * `storage` - Vector storage
    /// * `query` - Query vector
    /// * `encoder` - RaBitQ encoder for the query
    /// * `k` - Number of results to return
    /// * `ef` - Search beam width
    /// * `rerank_factor` - Multiplier for candidates to fetch (typically 2-4)
    ///
    /// # Returns
    ///
    /// Top-k results sorted by exact distance.
    pub fn search_with_rabitq(
        &self,
        storage: &Storage,
        query: &[f32],
        encoder: &RaBitQ,
        k: usize,
        ef: usize,
        rerank_factor: usize,
    ) -> Result<Vec<(f32, VecId)>> {
        // Phase 1: Get more candidates using standard search
        let expanded_k = k * rerank_factor;
        let expanded_ef = ef.max(expanded_k);

        // Use standard HNSW search
        let candidates = self.search(storage, query, expanded_k, expanded_ef)?;

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // Encode query for potential Hamming-based filtering
        let _query_code = encoder.encode(query);

        // Phase 2: Re-rank with exact distances
        // Note: The standard search already returns exact distances,
        // so this is mainly for future optimization where Phase 1
        // would use Hamming distance for faster candidate selection.

        // For now, just return top-k from the exact distance results
        let mut results = candidates;
        results.truncate(k);

        Ok(results)
    }

    /// Experimental: Search using Hamming distance for candidate filtering.
    ///
    /// This is a more aggressive optimization that uses binary codes
    /// during the beam search phase. Currently not used by default
    /// as it requires additional tuning for good recall.
    #[allow(dead_code)]
    pub fn search_hamming_filter(
        &self,
        storage: &Storage,
        query: &[f32],
        encoder: &RaBitQ,
        k: usize,
        ef: usize,
        rerank_count: usize,
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

        // Encode query to binary
        let query_code = encoder.encode(query);

        // Greedy descent using exact distance (upper layers are small)
        let mut current = entry_point;
        let mut current_dist = self.distance(storage, query, current)?;

        for layer in (1..=max_layer).rev() {
            (current, current_dist) =
                self.greedy_search_layer(storage, query, current, current_dist, layer, true)?;
        }

        // Beam search at layer 0 using Hamming distance for candidate ranking
        let mut candidates = self.beam_search_layer0_hamming(
            storage,
            query,
            &query_code,
            encoder,
            current,
            ef,
        )?;

        // Re-rank top candidates with exact distance
        candidates.truncate(rerank_count);
        let vec_ids: Vec<VecId> = candidates.iter().map(|(_, id)| *id).collect();

        // Fetch actual vectors and compute exact distances
        let mut exact_results: Vec<(f32, VecId)> = Vec::with_capacity(vec_ids.len());
        for &vec_id in &vec_ids {
            let dist = self.distance(storage, query, vec_id)?;
            exact_results.push((dist, vec_id));
        }

        // Sort by exact distance and return top-k
        exact_results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        exact_results.truncate(k);

        Ok(exact_results)
    }

    /// Beam search at layer 0 using Hamming distance for candidate ranking.
    ///
    /// This is an experimental optimization that uses binary codes
    /// for faster distance computation during beam search.
    #[allow(dead_code)]
    fn beam_search_layer0_hamming(
        &self,
        storage: &Storage,
        _query: &[f32],
        query_code: &[u8],
        _encoder: &RaBitQ,
        entry: VecId,
        ef: usize,
    ) -> Result<Vec<(u32, VecId)>> {
        // Min-heap for candidates (by Hamming distance, ascending)
        // Max-heap for results (by Hamming distance, descending - worst at top)
        let mut candidates: BinaryHeap<Reverse<(u32, VecId)>> = BinaryHeap::new();
        let mut results: BinaryHeap<(u32, VecId)> = BinaryHeap::new();
        let mut visited = RoaringBitmap::new();

        // Get entry point's Hamming distance
        let entry_code = self.get_binary_code(storage, entry)?.unwrap_or_default();
        let entry_dist = RaBitQ::hamming_distance(query_code, &entry_code);

        candidates.push(Reverse((entry_dist, entry)));
        results.push((entry_dist, entry));
        visited.insert(entry);

        while let Some(Reverse((current_dist, current))) = candidates.pop() {
            // Stop if current is worse than worst result (when results are full)
            if results.len() >= ef {
                if let Some(&(worst_dist, _)) = results.peek() {
                    if current_dist > worst_dist {
                        break;
                    }
                }
            }

            // Expand neighbors
            let neighbors = self.get_neighbors(storage, current, 0, true)?;

            // Batch fetch binary codes for unvisited neighbors
            let unvisited: Vec<VecId> = neighbors
                .iter()
                .filter(|n| !visited.contains(*n))
                .collect();

            if unvisited.is_empty() {
                continue;
            }

            let codes = self.get_binary_codes_batch(storage, &unvisited)?;

            for (neighbor, code_opt) in unvisited.into_iter().zip(codes.into_iter()) {
                visited.insert(neighbor);

                let dist = if let Some(code) = code_opt {
                    RaBitQ::hamming_distance(query_code, &code)
                } else {
                    u32::MAX // No code, skip
                };

                // Add to results if better than worst or results not full
                if results.len() < ef {
                    candidates.push(Reverse((dist, neighbor)));
                    results.push((dist, neighbor));
                } else if let Some(&(worst_dist, _)) = results.peek() {
                    if dist < worst_dist {
                        candidates.push(Reverse((dist, neighbor)));
                        results.pop();
                        results.push((dist, neighbor));
                    }
                }
            }
        }

        // Extract results sorted by Hamming distance (ascending)
        let mut result_vec: Vec<(u32, VecId)> = results.into_iter().collect();
        result_vec.sort_by_key(|(dist, _)| *dist);

        Ok(result_vec)
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
