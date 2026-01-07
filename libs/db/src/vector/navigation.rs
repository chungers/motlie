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
// NavigationCache (Task 2.9)
// ============================================================================

use std::collections::HashMap;
use std::sync::RwLock;

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
}

impl Default for NavigationCacheConfig {
    fn default() -> Self {
        Self {
            min_cached_layer: 2,
            max_cached_edges: 1_000_000,
        }
    }
}

/// In-memory cache for navigation layer data.
///
/// Caches upper layers (sparse) fully in memory to avoid disk I/O
/// during the O(log N) descent phase of search.
pub struct NavigationCache {
    /// Per-embedding-space navigation info
    nav_info: RwLock<HashMap<EmbeddingCode, NavigationLayerInfo>>,

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
}
