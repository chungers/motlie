//! Navigation layer structure for HNSW index.
//!
//! The navigation layer is the hierarchical structure that enables O(log N) search.
//! Each layer contains a subset of nodes, with layer 0 containing all nodes and
//! higher layers containing exponentially fewer nodes.
//!
//! # Layer Distribution
//!
//! At 1B scale with M=16:
//! - Layer 0: ~1B nodes (all nodes)
//! - Layer 1: ~63M nodes (1/16)
//! - Layer 2: ~4M nodes (1/256)
//! - Layer 3: ~250K nodes (1/4096)
//! - Layer 4+: <16K nodes

use rand::Rng;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

use super::schema::{EmbeddingCode, HnswLayer, VecId};

// ============================================================================
// NavigationLayerInfo
// ============================================================================

/// Navigation layer metadata for an embedding space.
///
/// Tracks entry points and statistics for each layer in the HNSW graph.
/// Persisted in GraphMeta CF and cached in memory for fast access.
#[derive(Archive, RkyvDeserialize, RkyvSerialize, Serialize, Deserialize, Debug, Clone)]
#[archive(check_bytes)]
pub struct NavigationLayerInfo {
    /// Entry point vec_id for each layer.
    /// Index 0 = layer 0 entry point, etc.
    /// The global entry point is at index `max_layer`.
    pub entry_points: Vec<VecId>,

    /// Number of nodes in each layer (for statistics/debugging).
    pub layer_counts: Vec<u64>,

    /// Maximum layer in the graph (0 for empty or single-layer).
    pub max_layer: HnswLayer,

    /// m_L parameter: controls probability of layer assignment.
    /// P(layer = l) = exp(-l / m_L), where m_L = 1/ln(M)
    pub m_l: f32,
}

impl NavigationLayerInfo {
    /// Create new navigation info for given M parameter.
    ///
    /// # Arguments
    /// * `m` - HNSW M parameter (max connections per node at layer 0)
    pub fn new(m: usize) -> Self {
        let m_l = 1.0 / (m as f32).ln();
        Self {
            entry_points: Vec::new(),
            layer_counts: Vec::new(),
            max_layer: 0,
            m_l,
        }
    }

    /// Assign a random layer for a new node.
    ///
    /// Uses the HNSW probability distribution: P(layer=l) ∝ exp(-l / m_L).
    /// Returns layer in range [0, current_max + 1].
    pub fn random_layer<R: Rng>(&self, rng: &mut R) -> HnswLayer {
        let r: f32 = rng.gen();
        // -ln(r) * m_L gives exponential distribution
        let level = (-r.ln() * self.m_l).floor() as u8;
        // Cap at one above current max (graph grows one layer at a time)
        level.min(self.max_layer.saturating_add(1))
    }

    /// Get the global entry point (highest layer entry).
    pub fn entry_point(&self) -> Option<VecId> {
        self.entry_points.last().copied()
    }

    /// Get entry point for a specific layer.
    pub fn entry_point_for_layer(&self, layer: HnswLayer) -> Option<VecId> {
        self.entry_points.get(layer as usize).copied()
    }

    /// Check if this is the first node being inserted.
    pub fn is_empty(&self) -> bool {
        self.entry_points.is_empty()
    }

    /// Update entry point if new node has higher layer.
    ///
    /// Returns true if the entry point was updated.
    pub fn maybe_update_entry(&mut self, vec_id: VecId, node_layer: HnswLayer) -> bool {
        if self.entry_points.is_empty() || node_layer > self.max_layer {
            // Extend entry_points to accommodate new layer
            while self.entry_points.len() <= node_layer as usize {
                self.entry_points.push(vec_id);
                self.layer_counts.push(0);
            }
            self.max_layer = node_layer;
            true
        } else {
            false
        }
    }

    /// Increment node count for a layer.
    pub fn increment_layer_count(&mut self, layer: HnswLayer) {
        if (layer as usize) < self.layer_counts.len() {
            self.layer_counts[layer as usize] += 1;
        }
    }

    /// Decrement node count for a layer.
    pub fn decrement_layer_count(&mut self, layer: HnswLayer) {
        if (layer as usize) < self.layer_counts.len() {
            self.layer_counts[layer as usize] = self.layer_counts[layer as usize].saturating_sub(1);
        }
    }

    /// Get the total node count (layer 0 count = total).
    pub fn total_nodes(&self) -> u64 {
        self.layer_counts.first().copied().unwrap_or(0)
    }

    /// Serialize to bytes for storage.
    pub fn to_bytes(&self) -> Vec<u8> {
        rmp_serde::to_vec(self).expect("NavigationLayerInfo serialization should never fail")
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, rmp_serde::decode::Error> {
        rmp_serde::from_slice(bytes)
    }
}

// ============================================================================
// NavigationCache (Task 2.9 + Task 3.5 Edge Caching)
// ============================================================================

use std::collections::HashMap;
use std::sync::RwLock;

use roaring::RoaringBitmap;

/// Configuration for the navigation cache.
#[derive(Debug, Clone)]
pub struct NavigationCacheConfig {
    /// Minimum layer to cache fully in memory.
    /// Layers below this use RocksDB with block cache.
    /// Default: 2 (cache layers 2+ in memory)
    pub min_cached_layer: HnswLayer,

    /// Maximum edges to cache per embedding space.
    /// Default: 1M edges across all cached layers.
    pub max_cached_edges: usize,

    /// Maximum entries in the layer 0-1 LRU cache.
    /// These layers are too large to cache fully, so we use LRU.
    /// Default: 10,000 entries
    pub hot_cache_size: usize,
}

impl Default for NavigationCacheConfig {
    fn default() -> Self {
        Self {
            min_cached_layer: 2,
            max_cached_edges: 1_000_000,
            hot_cache_size: 10_000,
        }
    }
}

/// Key for edge cache: (embedding_code, vec_id, layer)
type EdgeCacheKey = (EmbeddingCode, VecId, HnswLayer);

use std::collections::VecDeque;

/// In-memory cache for navigation layer data.
///
/// Caches upper layers (sparse) fully in memory to avoid disk I/O
/// during the O(log N) descent phase of search.
///
/// # Phase 3.5: Edge Caching
///
/// For layers >= min_cached_layer, edges are cached in memory.
/// This reduces RocksDB reads during upper layer traversal.
///
/// # Phase 3.6: Hot Node Cache
///
/// For layers 0-1 (too large to cache fully), a bounded FIFO cache
/// keeps frequently accessed nodes in memory.
pub struct NavigationCache {
    /// Per-embedding-space navigation info
    nav_info: RwLock<HashMap<EmbeddingCode, NavigationLayerInfo>>,

    /// Edge cache for upper layers (layer >= min_cached_layer)
    /// Key: (embedding_code, vec_id, layer) -> neighbor bitmap
    edge_cache: RwLock<HashMap<EdgeCacheKey, RoaringBitmap>>,

    /// Current edge cache size (sum of bitmap lengths)
    edge_cache_size: RwLock<usize>,

    /// Hot node cache for layers 0-1 (Phase 3.6)
    /// Uses FIFO eviction when cache is full
    hot_cache: RwLock<HashMap<EdgeCacheKey, RoaringBitmap>>,

    /// FIFO order for hot cache eviction
    hot_cache_order: RwLock<VecDeque<EdgeCacheKey>>,

    /// Configuration
    config: NavigationCacheConfig,
}

impl NavigationCache {
    /// Create a new navigation cache with default configuration.
    pub fn new() -> Self {
        Self::with_config(NavigationCacheConfig::default())
    }

    /// Create a new navigation cache with custom configuration.
    pub fn with_config(config: NavigationCacheConfig) -> Self {
        Self {
            nav_info: RwLock::new(HashMap::new()),
            edge_cache: RwLock::new(HashMap::new()),
            edge_cache_size: RwLock::new(0),
            hot_cache: RwLock::new(HashMap::new()),
            hot_cache_order: RwLock::new(VecDeque::new()),
            config,
        }
    }

    /// Get navigation info for an embedding space.
    ///
    /// Returns None if not cached. Caller should load from DB.
    pub fn get(&self, embedding: EmbeddingCode) -> Option<NavigationLayerInfo> {
        self.nav_info.read().ok()?.get(&embedding).cloned()
    }

    /// Update navigation info in cache.
    pub fn put(&self, embedding: EmbeddingCode, info: NavigationLayerInfo) {
        if let Ok(mut cache) = self.nav_info.write() {
            cache.insert(embedding, info);
        }
    }

    /// Update navigation info atomically with a closure.
    ///
    /// If no entry exists, creates one with default M=16.
    pub fn update<F>(&self, embedding: EmbeddingCode, m: usize, f: F)
    where
        F: FnOnce(&mut NavigationLayerInfo),
    {
        if let Ok(mut cache) = self.nav_info.write() {
            let info = cache
                .entry(embedding)
                .or_insert_with(|| NavigationLayerInfo::new(m));
            f(info);
        }
    }

    /// Remove navigation info from cache.
    pub fn remove(&self, embedding: EmbeddingCode) -> Option<NavigationLayerInfo> {
        self.nav_info.write().ok()?.remove(&embedding)
    }

    /// Get the cache configuration.
    pub fn config(&self) -> &NavigationCacheConfig {
        &self.config
    }

    /// Check if a layer should be cached in memory.
    pub fn should_cache_layer(&self, layer: HnswLayer) -> bool {
        layer >= self.config.min_cached_layer
    }

    // =========================================================================
    // Edge Caching (Phase 3.5)
    // =========================================================================

    /// Get cached neighbors for upper layers, or load via fetch_fn.
    ///
    /// For layers >= min_cached_layer (upper layers):
    /// - Returns cached bitmap if available
    /// - Otherwise calls fetch_fn and caches the result
    ///
    /// For layers < min_cached_layer (layers 0-1):
    /// - Uses bounded hot cache with FIFO eviction
    /// - Helps for repeated access to frequently queried nodes
    ///
    /// # Arguments
    /// * `embedding` - Embedding space code
    /// * `vec_id` - Vector ID
    /// * `layer` - HNSW layer
    /// * `fetch_fn` - Function to fetch edges from storage
    pub fn get_neighbors_cached<F, E>(
        &self,
        embedding: EmbeddingCode,
        vec_id: VecId,
        layer: HnswLayer,
        fetch_fn: F,
    ) -> Result<RoaringBitmap, E>
    where
        F: FnOnce() -> Result<RoaringBitmap, E>,
    {
        let key = (embedding, vec_id, layer);

        // Upper layers (>= min_cached_layer): full caching
        if layer >= self.config.min_cached_layer {
            // Check cache first (read lock)
            if let Ok(cache) = self.edge_cache.read() {
                if let Some(bitmap) = cache.get(&key) {
                    return Ok(bitmap.clone());
                }
            }

            // Cache miss - fetch from storage
            let bitmap = fetch_fn()?;

            // Try to insert into cache (write lock)
            if let Ok(mut cache) = self.edge_cache.write() {
                if let Ok(mut size) = self.edge_cache_size.write() {
                    let bitmap_len = bitmap.len() as usize;

                    // Only cache if we haven't exceeded max size
                    if *size + bitmap_len <= self.config.max_cached_edges {
                        cache.insert(key, bitmap.clone());
                        *size += bitmap_len;
                    }
                }
            }

            return Ok(bitmap);
        }

        // Lower layers (0-1): use hot cache with FIFO eviction
        // Check hot cache first (read lock)
        if let Ok(cache) = self.hot_cache.read() {
            if let Some(bitmap) = cache.get(&key) {
                return Ok(bitmap.clone());
            }
        }

        // Cache miss - fetch from storage
        let bitmap = fetch_fn()?;

        // Try to insert into hot cache (write lock)
        if let Ok(mut cache) = self.hot_cache.write() {
            if let Ok(mut order) = self.hot_cache_order.write() {
                // Evict oldest entry if cache is full
                while cache.len() >= self.config.hot_cache_size {
                    if let Some(old_key) = order.pop_front() {
                        cache.remove(&old_key);
                    } else {
                        break;
                    }
                }

                // Insert new entry
                cache.insert(key, bitmap.clone());
                order.push_back(key);
            }
        }

        Ok(bitmap)
    }

    /// Invalidate cached edges for a node (call after edge updates).
    ///
    /// Should be called when edges are modified during insert or delete.
    pub fn invalidate_edges(&self, embedding: EmbeddingCode, vec_id: VecId, layer: HnswLayer) {
        let key = (embedding, vec_id, layer);

        if layer >= self.config.min_cached_layer {
            // Upper layer cache
            if let Ok(mut cache) = self.edge_cache.write() {
                if let Some(removed) = cache.remove(&key) {
                    if let Ok(mut size) = self.edge_cache_size.write() {
                        *size = size.saturating_sub(removed.len() as usize);
                    }
                }
            }
        } else {
            // Hot cache (layers 0-1)
            if let Ok(mut cache) = self.hot_cache.write() {
                cache.remove(&key);
                // Note: We don't remove from hot_cache_order to avoid O(n) scan
                // The entry will be skipped during eviction if already removed
            }
        }
    }

    /// Get edge cache statistics.
    pub fn edge_cache_stats(&self) -> (usize, usize) {
        let entries = self
            .edge_cache
            .read()
            .map(|c| c.len())
            .unwrap_or(0);
        let size = self
            .edge_cache_size
            .read()
            .map(|s| *s)
            .unwrap_or(0);
        (entries, size)
    }

    /// Get hot cache statistics.
    pub fn hot_cache_stats(&self) -> usize {
        self.hot_cache.read().map(|c| c.len()).unwrap_or(0)
    }
}

impl Default for NavigationCache {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_navigation_layer_info_new() {
        let info = NavigationLayerInfo::new(16);
        assert!(info.entry_points.is_empty());
        assert!(info.layer_counts.is_empty());
        assert_eq!(info.max_layer, 0);
        // m_L = 1/ln(16) ≈ 0.361
        assert!((info.m_l - 0.361).abs() < 0.01);
    }

    #[test]
    fn test_random_layer_distribution() {
        let info = NavigationLayerInfo::new(16);
        let mut rng = rand::thread_rng();

        // Generate many layers and check distribution
        let mut counts = [0u32; 10];
        for _ in 0..10000 {
            let layer = info.random_layer(&mut rng) as usize;
            if layer < 10 {
                counts[layer] += 1;
            }
        }

        // Layer 0 should have most nodes
        assert!(counts[0] > counts[1]);
        assert!(counts[1] > counts[2]);
        // Most should be in layer 0
        assert!(counts[0] > 5000);
    }

    #[test]
    fn test_maybe_update_entry() {
        let mut info = NavigationLayerInfo::new(16);

        // First node always updates
        assert!(info.maybe_update_entry(100, 0));
        assert_eq!(info.entry_point(), Some(100));
        assert_eq!(info.max_layer, 0);

        // Same layer doesn't update
        assert!(!info.maybe_update_entry(200, 0));
        assert_eq!(info.entry_point(), Some(100));

        // Higher layer updates
        assert!(info.maybe_update_entry(300, 2));
        assert_eq!(info.entry_point(), Some(300));
        assert_eq!(info.max_layer, 2);
        assert_eq!(info.entry_points.len(), 3); // layers 0, 1, 2
    }

    #[test]
    fn test_layer_counts() {
        let mut info = NavigationLayerInfo::new(16);
        info.maybe_update_entry(100, 2);

        info.increment_layer_count(0);
        info.increment_layer_count(0);
        info.increment_layer_count(1);

        assert_eq!(info.layer_counts[0], 2);
        assert_eq!(info.layer_counts[1], 1);
        assert_eq!(info.total_nodes(), 2);

        info.decrement_layer_count(0);
        assert_eq!(info.total_nodes(), 1);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut info = NavigationLayerInfo::new(16);
        info.maybe_update_entry(42, 3);
        info.increment_layer_count(0);
        info.increment_layer_count(1);

        let bytes = info.to_bytes();
        let restored = NavigationLayerInfo::from_bytes(&bytes).unwrap();

        assert_eq!(restored.max_layer, info.max_layer);
        assert_eq!(restored.entry_points, info.entry_points);
        assert_eq!(restored.layer_counts, info.layer_counts);
        assert!((restored.m_l - info.m_l).abs() < 0.0001);
    }

    #[test]
    fn test_navigation_cache() {
        let cache = NavigationCache::new();

        assert!(cache.get(1).is_none());

        let info = NavigationLayerInfo::new(16);
        cache.put(1, info.clone());

        let retrieved = cache.get(1).unwrap();
        assert_eq!(retrieved.max_layer, info.max_layer);
    }

    #[test]
    fn test_navigation_cache_update() {
        let cache = NavigationCache::new();

        cache.update(1, 16, |info| {
            info.maybe_update_entry(100, 2);
        });

        let info = cache.get(1).unwrap();
        assert_eq!(info.entry_point(), Some(100));
        assert_eq!(info.max_layer, 2);
    }

    #[test]
    fn test_should_cache_layer() {
        let cache = NavigationCache::new();

        // Default min_cached_layer = 2
        assert!(!cache.should_cache_layer(0));
        assert!(!cache.should_cache_layer(1));
        assert!(cache.should_cache_layer(2));
        assert!(cache.should_cache_layer(3));
    }

    #[test]
    fn test_edge_caching() {
        let cache = NavigationCache::new();
        let mut call_count = 0;

        // Layer 0 - should use hot cache (Phase 3.6)
        let result: Result<RoaringBitmap, ()> = cache.get_neighbors_cached(1, 100, 0, || {
            call_count += 1;
            let mut bm = RoaringBitmap::new();
            bm.insert(1);
            bm.insert(2);
            Ok(bm)
        });
        assert!(result.is_ok());
        assert_eq!(call_count, 1);
        assert_eq!(cache.hot_cache_stats(), 1);

        // Layer 0 again - should return cached value from hot cache
        let result: Result<RoaringBitmap, ()> = cache.get_neighbors_cached(1, 100, 0, || {
            call_count += 1;
            let mut bm = RoaringBitmap::new();
            bm.insert(1);
            bm.insert(2);
            Ok(bm)
        });
        assert!(result.is_ok());
        assert_eq!(call_count, 1); // No new call - hot cache hit!
        let bm = result.unwrap();
        assert!(bm.contains(1));
        assert!(bm.contains(2));

        // Layer 2 - should cache in upper layer cache (>= min_cached_layer)
        let result: Result<RoaringBitmap, ()> = cache.get_neighbors_cached(1, 100, 2, || {
            call_count += 1;
            let mut bm = RoaringBitmap::new();
            bm.insert(10);
            bm.insert(20);
            Ok(bm)
        });
        assert!(result.is_ok());
        assert_eq!(call_count, 2);
        let (entries, size) = cache.edge_cache_stats();
        assert_eq!(entries, 1);
        assert_eq!(size, 2);

        // Layer 2 again - should return cached value (no new fetch_fn call)
        let result: Result<RoaringBitmap, ()> = cache.get_neighbors_cached(1, 100, 2, || {
            call_count += 1;
            Ok(RoaringBitmap::new())
        });
        assert!(result.is_ok());
        assert_eq!(call_count, 2); // No new call
        let bm = result.unwrap();
        assert!(bm.contains(10));
        assert!(bm.contains(20));
    }

    #[test]
    fn test_edge_cache_invalidation() {
        let cache = NavigationCache::new();

        // Cache an edge
        let _: Result<RoaringBitmap, ()> = cache.get_neighbors_cached(1, 100, 2, || {
            let mut bm = RoaringBitmap::new();
            bm.insert(10);
            Ok(bm)
        });

        let (entries, _) = cache.edge_cache_stats();
        assert_eq!(entries, 1);

        // Invalidate
        cache.invalidate_edges(1, 100, 2);

        let (entries, _) = cache.edge_cache_stats();
        assert_eq!(entries, 0);
    }

    #[test]
    fn test_hot_cache_eviction() {
        // Create cache with small hot_cache_size for testing
        let config = NavigationCacheConfig {
            hot_cache_size: 3,
            ..Default::default()
        };
        let cache = NavigationCache::with_config(config);

        // Fill the hot cache
        for vec_id in 0..3 {
            let _: Result<RoaringBitmap, ()> = cache.get_neighbors_cached(1, vec_id, 0, || {
                let mut bm = RoaringBitmap::new();
                bm.insert(vec_id);
                Ok(bm)
            });
        }
        assert_eq!(cache.hot_cache_stats(), 3);

        // Add one more - should evict the oldest (vec_id=0)
        let _: Result<RoaringBitmap, ()> = cache.get_neighbors_cached(1, 100, 0, || {
            let mut bm = RoaringBitmap::new();
            bm.insert(100);
            Ok(bm)
        });
        assert_eq!(cache.hot_cache_stats(), 3);

        // vec_id=0 should be evicted - fetch_fn will be called again
        let mut fetch_called = false;
        let result: Result<RoaringBitmap, ()> = cache.get_neighbors_cached(1, 0, 0, || {
            fetch_called = true;
            let mut bm = RoaringBitmap::new();
            bm.insert(0);
            Ok(bm)
        });
        assert!(result.is_ok());
        assert!(fetch_called); // Was evicted, so fetch_fn was called

        // vec_id=2 should still be cached (not evicted yet)
        // Note: After re-adding vec_id=0, the cache contains [1, 2, 100] -> [2, 100, 0]
        let mut fetch_called = false;
        let result: Result<RoaringBitmap, ()> = cache.get_neighbors_cached(1, 2, 0, || {
            fetch_called = true;
            Ok(RoaringBitmap::new())
        });
        assert!(result.is_ok());
        assert!(!fetch_called); // Still in cache
    }
}
