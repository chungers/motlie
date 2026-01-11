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
//! # Module Structure
//!
//! - `mod.rs` - Index struct and public API
//! - `config.rs` - HNSW configuration (`Config`, `ConfigWarning`)
//! - `insert.rs` - Insert algorithm implementation
//! - `search.rs` - Search algorithms (standard and RaBitQ)
//! - `graph.rs` - Graph operations (neighbors, distances, batch)
//!
//! # References
//!
//! - [HNSW Paper](https://arxiv.org/abs/1603.09320)
//! - [HNSW2 Optimization](examples/vector/HNSW2.md)

pub mod config;
mod graph;
mod insert;
mod search;

// Re-export Config for public API as vector::hnsw::Config
pub use config::{Config, ConfigWarning};

use std::sync::Arc;

use anyhow::Result;

use crate::vector::cache::{BinaryCodeCache, NavigationCache};
use crate::vector::distance::Distance;
use crate::vector::rabitq::RaBitQ;
use crate::vector::schema::{EmbeddingCode, VecId, VectorElementType};
use crate::vector::Storage;

// Re-export for public API
pub use graph::{cosine_distance, dot_product_distance, l2_distance};

// ============================================================================
// HNSW Index
// ============================================================================

/// HNSW index for a single embedding space.
///
/// Provides insert and search operations backed by RocksDB storage.
pub struct Index {
    /// The embedding space code this index manages
    embedding: EmbeddingCode,
    /// Distance metric for this index (Cosine, L2, DotProduct)
    distance: Distance,
    /// Storage type for vectors (F32 or F16)
    storage_type: VectorElementType,
    /// HNSW configuration (M, ef_construction, etc.)
    config: Config,
    /// Navigation cache for fast layer traversal
    nav_cache: Arc<NavigationCache>,
}

impl Index {
    /// Create a new HNSW index for an embedding space.
    ///
    /// # Arguments
    /// * `embedding` - The embedding space code
    /// * `distance` - Distance metric to use (Cosine, L2, DotProduct)
    /// * `config` - HNSW configuration
    /// * `nav_cache` - Navigation cache for fast layer traversal
    pub fn new(
        embedding: EmbeddingCode,
        distance: Distance,
        config: Config,
        nav_cache: Arc<NavigationCache>,
    ) -> Self {
        Self {
            embedding,
            distance,
            storage_type: VectorElementType::default(), // F32 by default
            config,
            nav_cache,
        }
    }

    /// Create a new HNSW index with specified storage type.
    ///
    /// # Arguments
    /// * `embedding` - The embedding space code
    /// * `distance` - Distance metric to use (Cosine, L2, DotProduct)
    /// * `storage_type` - Storage type for vectors (F32 or F16)
    /// * `config` - HNSW configuration
    /// * `nav_cache` - Navigation cache for fast layer traversal
    pub fn with_storage_type(
        embedding: EmbeddingCode,
        distance: Distance,
        storage_type: VectorElementType,
        config: Config,
        nav_cache: Arc<NavigationCache>,
    ) -> Self {
        Self {
            embedding,
            distance,
            storage_type,
            config,
            nav_cache,
        }
    }

    /// Get the embedding code.
    pub fn embedding(&self) -> EmbeddingCode {
        self.embedding
    }

    /// Get the storage type.
    pub fn storage_type(&self) -> VectorElementType {
        self.storage_type
    }

    /// Get the configuration.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Get the distance metric.
    pub fn distance_metric(&self) -> Distance {
        self.distance
    }

    /// Get a reference to the navigation cache.
    pub fn nav_cache(&self) -> &Arc<NavigationCache> {
        &self.nav_cache
    }

    // =========================================================================
    // Insert (delegated to insert.rs)
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
        insert::insert(self, storage, vec_id, vector)
    }

    // =========================================================================
    // Search (delegated to search.rs)
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
        search::search(self, storage, query, k, ef)
    }

    /// Two-phase search using RaBitQ with in-memory cached binary codes.
    ///
    /// Uses `BinaryCodeCache` for binary codes instead of fetching from RocksDB,
    /// avoiding the "double I/O" problem that would make RaBitQ slower than L2.
    ///
    /// # Performance
    ///
    /// With cached codes:
    /// - Hamming distance: ~ns (SIMD popcount on in-memory bytes)
    /// - No RocksDB reads during beam search
    /// - Only re-ranking phase reads vectors from disk
    ///
    /// Expected speedup: 2-4x over standard L2 search.
    ///
    /// # Arguments
    ///
    /// * `storage` - Vector storage (only used for re-ranking)
    /// * `query` - Query vector
    /// * `encoder` - RaBitQ encoder
    /// * `code_cache` - In-memory cache of binary codes
    /// * `k` - Number of results to return
    /// * `ef` - Search beam width
    /// * `rerank_factor` - Multiplier for candidates to re-rank
    pub fn search_with_rabitq_cached(
        &self,
        storage: &Storage,
        query: &[f32],
        encoder: &RaBitQ,
        code_cache: &BinaryCodeCache,
        k: usize,
        ef: usize,
        rerank_factor: usize,
    ) -> Result<Vec<(f32, VecId)>> {
        search::search_with_rabitq_cached(
            self,
            storage,
            query,
            encoder,
            code_cache,
            k,
            ef,
            rerank_factor,
        )
    }

    // =========================================================================
    // Graph Operations (delegated to graph.rs)
    // =========================================================================

    /// Get the number of neighbors (degree) for a node at a specific layer.
    ///
    /// This uses `RoaringBitmap::len()` which is O(1) since the cardinality
    /// is stored in the bitmap structure. Used for degree-based pruning checks.
    pub fn get_neighbor_count(&self, storage: &Storage, vec_id: VecId, layer: u8) -> Result<u64> {
        graph::get_neighbor_count(self, storage, vec_id, layer)
    }

    /// Batch fetch vectors using RocksDB's multi_get_cf.
    ///
    /// Returns vectors in the same order as input vec_ids. Missing vectors
    /// are represented as None.
    pub fn get_vectors_batch(
        &self,
        storage: &Storage,
        vec_ids: &[VecId],
    ) -> Result<Vec<Option<Vec<f32>>>> {
        graph::get_vectors_batch(self, storage, vec_ids)
    }

    /// Batch compute distances from query to multiple vectors.
    ///
    /// Fetches all vectors in a single multi_get_cf call, then computes distances
    /// in parallel using rayon. Uses the configured distance metric (L2, Cosine, DotProduct).
    /// Returns (vec_id, distance) pairs for vectors that exist.
    pub fn batch_distances(
        &self,
        storage: &Storage,
        query: &[f32],
        vec_ids: &[VecId],
    ) -> Result<Vec<(VecId, f32)>> {
        graph::batch_distances(self, storage, query, vec_ids)
    }

    /// Batch fetch neighbors for multiple nodes at a layer.
    ///
    /// Uses multi_get_cf for efficient batch lookup. Returns (vec_id, bitmap) pairs.
    pub fn get_neighbors_batch(
        &self,
        storage: &Storage,
        vec_ids: &[VecId],
        layer: u8,
    ) -> Result<Vec<(VecId, roaring::RoaringBitmap)>> {
        graph::get_neighbors_batch(self, storage, vec_ids, layer)
    }

    /// Get a binary code for a vector.
    pub fn get_binary_code(&self, storage: &Storage, vec_id: VecId) -> Result<Option<Vec<u8>>> {
        graph::get_binary_code(self, storage, vec_id)
    }

    /// Batch retrieve binary codes for multiple vectors.
    pub fn get_binary_codes_batch(
        &self,
        storage: &Storage,
        vec_ids: &[VecId],
    ) -> Result<Vec<Option<Vec<u8>>>> {
        graph::get_binary_codes_batch(self, storage, vec_ids)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rocksdb::ColumnFamily;
    use crate::vector::{Distance, Embedding, SearchConfig, VectorCfKey, VectorCfValue, Vectors};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;
    use std::collections::HashSet;
    use tempfile::TempDir;

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
        let config = Config::default();
        let nav_cache = Arc::new(NavigationCache::new());
        let index = Index::new(1, Distance::L2, config.clone(), nav_cache);

        assert_eq!(index.embedding(), 1);
        assert_eq!(index.config().m, config.m);
    }

    // =========================================================================
    // Recall Tests
    // =========================================================================
    //
    // These tests verify HNSW recall characteristics at different scales
    // and with different parameter configurations.
    //
    // Recall@k = |HNSW_results ∩ Ground_truth| / k
    //
    // Expected recall targets:
    // - Small scale (100-1K vectors): 99%+ with default params
    // - Medium scale (1K-10K): 95%+ with default params
    // - Large scale (10K-100K): 90%+ with default, 95%+ with tuned params

    /// Helper: Create test storage
    fn create_test_storage() -> (TempDir, Storage) {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut storage = Storage::readwrite(temp_dir.path());
        storage.ready().expect("Failed to initialize storage");
        (temp_dir, storage)
    }

    /// Helper: Generate random vectors
    fn generate_random_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..count)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect()
    }

    /// Helper: Compute ground truth (brute force k-NN)
    fn brute_force_knn(query: &[f32], vectors: &[Vec<f32>], k: usize) -> Vec<usize> {
        let mut distances: Vec<(f32, usize)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (l2_distance(query, v), i))
            .collect();
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        distances.into_iter().take(k).map(|(_, i)| i).collect()
    }

    /// Helper: Compute recall@k
    fn compute_recall(hnsw_results: &[(f32, VecId)], ground_truth: &[usize]) -> f64 {
        let hnsw_set: HashSet<u32> = hnsw_results.iter().map(|(_, id)| *id).collect();
        let gt_set: HashSet<u32> = ground_truth.iter().map(|&i| i as u32).collect();
        let intersection = hnsw_set.intersection(&gt_set).count();
        intersection as f64 / ground_truth.len() as f64
    }

    /// Helper: Store vectors in RocksDB Vectors CF
    fn store_vectors(storage: &Storage, embedding: EmbeddingCode, vectors: &[Vec<f32>]) {
        let txn_db = storage.transaction_db().expect("Failed to get txn_db");
        let cf = txn_db
            .cf_handle(Vectors::CF_NAME)
            .expect("Vectors CF not found");

        for (i, vector) in vectors.iter().enumerate() {
            let key = Vectors::key_to_bytes(&VectorCfKey(embedding, i as VecId));
            let value = Vectors::value_to_bytes(&VectorCfValue(vector.clone()));
            txn_db
                .put_cf(&cf, key, value)
                .expect("Failed to store vector");
        }
    }

    /// Helper: Build HNSW index with vectors
    fn build_index(
        storage: &Storage,
        embedding: EmbeddingCode,
        distance: Distance,
        vectors: &[Vec<f32>],
        config: &Config,
    ) -> Index {
        // First, store vectors in RocksDB so HNSW can access them
        store_vectors(storage, embedding, vectors);

        // Now build the HNSW index
        let nav_cache = Arc::new(NavigationCache::new());
        let index = Index::new(embedding, distance, config.clone(), nav_cache);

        for (i, vector) in vectors.iter().enumerate() {
            index
                .insert(storage, i as VecId, vector)
                .expect("Insert failed");
        }

        index
    }

    #[test]
    fn test_recall_small_scale_100_vectors() {
        let (_temp_dir, storage) = create_test_storage();
        let dim = 32;
        let n_vectors = 100;
        let n_queries = 20;
        let k = 10;

        let vectors = generate_random_vectors(n_vectors, dim, 42);
        let queries = generate_random_vectors(n_queries, dim, 123);

        let config = Config {
            dim,
            m: 16,
            m_max: 32,
            m_max_0: 32,
            ef_construction: 100,
            ..Default::default()
        };

        let index = build_index(&storage, 1, Distance::L2, &vectors, &config);

        // Test recall over multiple queries
        let mut total_recall = 0.0;
        for query in &queries {
            let results = index.search(&storage, query, k, 50).expect("Search failed");
            let ground_truth = brute_force_knn(query, &vectors, k);
            total_recall += compute_recall(&results, &ground_truth);
        }

        let avg_recall = total_recall / n_queries as f64;
        assert!(
            avg_recall >= 0.95,
            "Recall@{} should be >= 95% at {} vectors, got {:.1}%",
            k,
            n_vectors,
            avg_recall * 100.0
        );
    }

    #[test]
    fn test_recall_medium_scale_1k_vectors() {
        let (_temp_dir, storage) = create_test_storage();
        let dim = 64;
        let n_vectors = 1000;
        let n_queries = 20;
        let k = 10;

        let vectors = generate_random_vectors(n_vectors, dim, 42);
        let queries = generate_random_vectors(n_queries, dim, 123);

        let config = Config {
            dim,
            m: 16,
            m_max: 32,
            m_max_0: 32,
            ef_construction: 200,
            ..Default::default()
        };

        let index = build_index(&storage, 1, Distance::L2, &vectors, &config);

        let mut total_recall = 0.0;
        for query in &queries {
            let results = index
                .search(&storage, query, k, 100)
                .expect("Search failed");
            let ground_truth = brute_force_knn(query, &vectors, k);
            total_recall += compute_recall(&results, &ground_truth);
        }

        let avg_recall = total_recall / n_queries as f64;
        assert!(
            avg_recall >= 0.90,
            "Recall@{} should be >= 90% at {} vectors, got {:.1}%",
            k,
            n_vectors,
            avg_recall * 100.0
        );
    }

    #[test]
    fn test_recall_vs_ef_tradeoff() {
        // Test how recall improves with higher ef values
        let (_temp_dir, storage) = create_test_storage();
        let dim = 32;
        let n_vectors = 500;
        let n_queries = 10;
        let k = 10;

        let vectors = generate_random_vectors(n_vectors, dim, 42);
        let queries = generate_random_vectors(n_queries, dim, 123);

        let config = Config {
            dim,
            m: 16,
            m_max: 32,
            m_max_0: 32,
            ef_construction: 100,
            ..Default::default()
        };

        let index = build_index(&storage, 1, Distance::L2, &vectors, &config);

        // Test different ef values
        let ef_values = [10, 20, 50, 100];
        let mut prev_recall = 0.0;

        for &ef in &ef_values {
            let mut total_recall = 0.0;
            for query in &queries {
                let results = index.search(&storage, query, k, ef).expect("Search failed");
                let ground_truth = brute_force_knn(query, &vectors, k);
                total_recall += compute_recall(&results, &ground_truth);
            }
            let avg_recall = total_recall / n_queries as f64;

            // Recall should generally increase with ef (allowing small variance)
            assert!(
                avg_recall >= prev_recall - 0.05,
                "Recall should not decrease significantly with higher ef. ef={}, recall={:.1}%, prev={:.1}%",
                ef, avg_recall * 100.0, prev_recall * 100.0
            );
            prev_recall = avg_recall;
        }

        // Final recall with ef=100 should be good
        assert!(
            prev_recall >= 0.85,
            "Recall with ef=100 should be >= 85%, got {:.1}%",
            prev_recall * 100.0
        );
    }

    #[test]
    fn test_recall_high_recall_config() {
        // Test the high_recall configuration preset
        let (_temp_dir, storage) = create_test_storage();
        let dim = 64;
        let n_vectors = 500;
        let n_queries = 10;
        let k = 10;

        let vectors = generate_random_vectors(n_vectors, dim, 42);
        let queries = generate_random_vectors(n_queries, dim, 123);

        let config = Config::high_recall(dim);
        let index = build_index(&storage, 1, Distance::L2, &vectors, &config);

        let mut total_recall = 0.0;
        for query in &queries {
            let results = index
                .search(&storage, query, k, 150)
                .expect("Search failed");
            let ground_truth = brute_force_knn(query, &vectors, k);
            total_recall += compute_recall(&results, &ground_truth);
        }

        let avg_recall = total_recall / n_queries as f64;
        assert!(
            avg_recall >= 0.95,
            "High-recall config should achieve >= 95% recall, got {:.1}%",
            avg_recall * 100.0
        );
    }

    #[test]
    fn test_recall_compact_config() {
        // Test the compact (memory-optimized) configuration
        let (_temp_dir, storage) = create_test_storage();
        let dim = 32;
        let n_vectors = 500;
        let n_queries = 10;
        let k = 10;

        let vectors = generate_random_vectors(n_vectors, dim, 42);
        let queries = generate_random_vectors(n_queries, dim, 123);

        let config = Config::compact(dim);
        let index = build_index(&storage, 1, Distance::L2, &vectors, &config);

        let mut total_recall = 0.0;
        for query in &queries {
            // Compact config needs higher ef to compensate for smaller M
            let results = index
                .search(&storage, query, k, 100)
                .expect("Search failed");
            let ground_truth = brute_force_knn(query, &vectors, k);
            total_recall += compute_recall(&results, &ground_truth);
        }

        let avg_recall = total_recall / n_queries as f64;
        // Compact config trades recall for memory, so lower threshold
        assert!(
            avg_recall >= 0.70,
            "Compact config should achieve >= 70% recall, got {:.1}%",
            avg_recall * 100.0
        );
    }

    #[test]
    fn test_recall_different_k_values() {
        // Test recall at different k values (1, 5, 10, 20)
        let (_temp_dir, storage) = create_test_storage();
        let dim = 32;
        let n_vectors = 500;
        let n_queries = 10;

        let vectors = generate_random_vectors(n_vectors, dim, 42);
        let queries = generate_random_vectors(n_queries, dim, 123);

        let config = Config {
            dim,
            m: 16,
            m_max: 32,
            m_max_0: 32,
            ef_construction: 100,
            ..Default::default()
        };

        let index = build_index(&storage, 1, Distance::L2, &vectors, &config);

        let k_values = [1, 5, 10, 20];
        for &k in &k_values {
            let ef = k.max(50); // ef should be at least k
            let mut total_recall = 0.0;

            for query in &queries {
                let results = index.search(&storage, query, k, ef).expect("Search failed");
                let ground_truth = brute_force_knn(query, &vectors, k);
                total_recall += compute_recall(&results, &ground_truth);
            }

            let avg_recall = total_recall / n_queries as f64;
            // k=1 is harder, so lower threshold
            let min_recall = if k == 1 { 0.80 } else { 0.85 };
            assert!(
                avg_recall >= min_recall,
                "Recall@{} should be >= {:.0}%, got {:.1}%",
                k,
                min_recall * 100.0,
                avg_recall * 100.0
            );
        }
    }

    #[test]
    fn test_recall_clustered_data() {
        // Test recall on clustered data (harder case)
        let (_temp_dir, storage) = create_test_storage();
        let dim = 32;
        let n_clusters = 10;
        let points_per_cluster = 50;
        let n_vectors = n_clusters * points_per_cluster;
        let n_queries = 10;
        let k = 10;

        // Generate clustered data
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let mut vectors = Vec::with_capacity(n_vectors);

        for _ in 0..n_clusters {
            // Random cluster center
            let center: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 10.0).collect();

            // Points around the center
            for _ in 0..points_per_cluster {
                let point: Vec<f32> = center
                    .iter()
                    .map(|&c| c + (rng.gen::<f32>() - 0.5) * 0.5)
                    .collect();
                vectors.push(point);
            }
        }

        let queries = generate_random_vectors(n_queries, dim, 123);

        let config = Config {
            dim,
            m: 16,
            m_max: 32,
            m_max_0: 32,
            ef_construction: 200,
            ..Default::default()
        };

        let index = build_index(&storage, 1, Distance::L2, &vectors, &config);

        let mut total_recall = 0.0;
        for query in &queries {
            let results = index
                .search(&storage, query, k, 100)
                .expect("Search failed");
            let ground_truth = brute_force_knn(query, &vectors, k);
            total_recall += compute_recall(&results, &ground_truth);
        }

        let avg_recall = total_recall / n_queries as f64;
        assert!(
            avg_recall >= 0.85,
            "Recall on clustered data should be >= 85%, got {:.1}%",
            avg_recall * 100.0
        );
    }

    // =========================================================================
    // Parameter Documentation Tests
    // =========================================================================
    //
    // These tests document the expected recall for different configurations.
    // Run with `cargo test -- --nocapture` to see the parameter recommendations.

    #[test]
    fn test_document_optimal_parameters() {
        // This test documents optimal parameters for different recall targets
        let (_temp_dir, storage) = create_test_storage();
        let dim = 64;
        let n_vectors = 1000;
        let n_queries = 20;
        let k = 10;

        let vectors = generate_random_vectors(n_vectors, dim, 42);
        let queries = generate_random_vectors(n_queries, dim, 123);

        // Test different M values
        let m_values = [8, 16, 32];
        let ef_values = [50, 100, 200];

        println!(
            "\n=== HNSW Parameter Recommendations ({}D, {} vectors) ===",
            dim, n_vectors
        );
        println!("| M | ef_search | Recall@{} |", k);
        println!("|---|-----------|----------|");

        for &m in &m_values {
            let config = Config {
                dim,
                m,
                m_max: 2 * m,
                m_max_0: 2 * m,
                ef_construction: 200,
                ..Default::default()
            };

            let index = build_index(&storage, 1, Distance::L2, &vectors, &config);

            for &ef in &ef_values {
                let mut total_recall = 0.0;
                for query in &queries {
                    let results = index.search(&storage, query, k, ef).expect("Search failed");
                    let ground_truth = brute_force_knn(query, &vectors, k);
                    total_recall += compute_recall(&results, &ground_truth);
                }
                let avg_recall = total_recall / n_queries as f64;
                println!("| {} | {} | {:.1}% |", m, ef, avg_recall * 100.0);
            }
        }

        println!("\nRecommendations:");
        println!("- For >95% recall: M=16-32, ef_search >= 100");
        println!("- For >90% recall: M=16, ef_search >= 50");
        println!("- Memory-constrained: M=8, ef_search >= 100");
    }

    // =========================================================================
    // SearchConfig Integration Tests
    // =========================================================================
    //
    // These tests verify SearchConfig works correctly with Index,
    // including strategy selection, distance metric consistency, and
    // embedding code validation.

    /// Helper: Create an Embedding for testing
    fn make_test_embedding(code: u64, dim: u32, distance: Distance) -> Embedding {
        Embedding::new(code, "test-model", dim, distance, None)
    }

    #[test]
    fn test_search_config_strategy_selection() {
        // Cosine should auto-select RaBitQ
        let cosine_emb = make_test_embedding(1, 128, Distance::Cosine);
        let cosine_config = SearchConfig::new(cosine_emb, 10);
        assert!(
            cosine_config.strategy().is_rabitq(),
            "Cosine should auto-select RaBitQ"
        );

        // L2 should auto-select Exact
        let l2_emb = make_test_embedding(2, 128, Distance::L2);
        let l2_config = SearchConfig::new(l2_emb, 10);
        assert!(
            l2_config.strategy().is_exact(),
            "L2 should auto-select Exact"
        );

        // DotProduct should auto-select Exact
        let dot_emb = make_test_embedding(3, 128, Distance::DotProduct);
        let dot_config = SearchConfig::new(dot_emb, 10);
        assert!(
            dot_config.strategy().is_exact(),
            "DotProduct should auto-select Exact"
        );
    }

    #[test]
    fn test_search_config_embedding_validation() {
        let emb = make_test_embedding(42, 128, Distance::L2);
        let config = SearchConfig::new(emb, 10);

        // Should pass for matching code
        assert!(config.validate_embedding_code(42).is_ok());

        // Should fail for mismatched code
        let err = config.validate_embedding_code(99).unwrap_err();
        assert!(err.to_string().contains("mismatch"));
        assert!(err.to_string().contains("42"));
        assert!(err.to_string().contains("99"));
    }

    #[test]
    fn test_search_config_compute_distance_l2() {
        let emb = make_test_embedding(1, 4, Distance::L2);
        let config = SearchConfig::new(emb, 10);

        let a = vec![0.0, 0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0, 0.0];

        // L2 distance = sqrt(9 + 16) = 5
        let dist = config.compute_distance(&a, &b);
        assert!((dist - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_search_config_compute_distance_cosine() {
        let emb = make_test_embedding(1, 4, Distance::Cosine);
        let config = SearchConfig::new(emb, 10);

        // Same vector should have distance 0
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let dist = config.compute_distance(&a, &a);
        assert!(
            dist.abs() < 0.001,
            "Same vector cosine distance should be ~0"
        );

        // Orthogonal vectors
        let b = vec![1.0, 0.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0, 0.0];
        let dist = config.compute_distance(&b, &c);
        assert!(
            (dist - 1.0).abs() < 0.001,
            "Orthogonal vectors cosine distance should be ~1"
        );
    }

    #[test]
    fn test_search_config_compute_distance_dot_product() {
        let emb = make_test_embedding(1, 4, Distance::DotProduct);
        let config = SearchConfig::new(emb, 10);

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];

        // DotProduct returns -dot(a, b) for "lower is better"
        // dot = 1 + 2 + 3 + 4 = 10, so distance = -10
        let dist = config.compute_distance(&a, &b);
        assert!((dist - (-10.0)).abs() < 0.001);
    }

    #[test]
    fn test_search_config_rabitq_only_for_cosine() {
        // RaBitQ should work for Cosine
        let cosine_emb = make_test_embedding(1, 128, Distance::Cosine);
        let result = SearchConfig::new(cosine_emb, 10).exact().rabitq();
        assert!(result.is_ok());

        // RaBitQ should fail for L2
        let l2_emb = make_test_embedding(2, 128, Distance::L2);
        let result = SearchConfig::new(l2_emb, 10).rabitq();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Cosine"));

        // RaBitQ should fail for DotProduct
        let dot_emb = make_test_embedding(3, 128, Distance::DotProduct);
        let result = SearchConfig::new(dot_emb, 10).rabitq();
        assert!(result.is_err());
    }

    // =========================================================================
    // SIFT 1K Integration Tests
    // =========================================================================
    //
    // These tests use a SIFT-like dataset (1K random vectors) to verify
    // SearchConfig works correctly with actual HNSW index operations.

    /// Helper: Generate SIFT-like normalized vectors (128D, unit length)
    fn generate_sift_like_vectors(count: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..count)
            .map(|_| {
                let mut vec: Vec<f32> = (0..128).map(|_| rng.gen::<f32>() - 0.5).collect();
                // Normalize to unit length for cosine distance
                let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for x in &mut vec {
                        *x /= norm;
                    }
                }
                vec
            })
            .collect()
    }

    /// Helper: Compute ground truth with configurable distance metric
    fn brute_force_knn_with_distance(
        query: &[f32],
        vectors: &[Vec<f32>],
        k: usize,
        distance: Distance,
    ) -> Vec<usize> {
        let mut distances: Vec<(f32, usize)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (distance.compute(query, v), i))
            .collect();
        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        distances.into_iter().take(k).map(|(_, i)| i).collect()
    }

    #[test]
    fn test_sift_1k_l2_with_search_config() {
        // Test SearchConfig with L2 distance on SIFT-like data
        let (_temp_dir, storage) = create_test_storage();
        let dim = 128;
        let n_vectors = 1000;
        let n_queries = 20;
        let k = 10;

        let vectors = generate_sift_like_vectors(n_vectors, 42);
        let queries = generate_sift_like_vectors(n_queries, 123);

        // Create embedding with L2 distance
        let embedding = make_test_embedding(1, dim as u32, Distance::L2);

        let config = Config {
            dim,
            m: 16,
            m_max: 32,
            m_max_0: 32,
            ef_construction: 200,
            ..Default::default()
        };

        let index = build_index(
            &storage,
            embedding.code(),
            embedding.distance(),
            &vectors,
            &config,
        );

        // Create SearchConfig - should auto-select Exact for L2
        let search_config = SearchConfig::new(embedding.clone(), k).with_ef(100);
        assert!(search_config.strategy().is_exact());

        // Validate embedding matches
        assert!(search_config
            .validate_embedding_code(index.embedding())
            .is_ok());

        // Test recall using SearchConfig's compute_distance
        let mut total_recall = 0.0;
        for query in &queries {
            // Use the existing search method (SearchConfig will be integrated later)
            let results = index
                .search(&storage, query, k, search_config.ef())
                .expect("Search failed");
            let ground_truth = brute_force_knn_with_distance(query, &vectors, k, Distance::L2);
            total_recall += compute_recall(&results, &ground_truth);
        }

        let avg_recall = total_recall / n_queries as f64;
        assert!(
            avg_recall >= 0.90,
            "L2 recall@{} should be >= 90% at {} vectors, got {:.1}%",
            k,
            n_vectors,
            avg_recall * 100.0
        );
    }

    #[test]
    fn test_sift_1k_cosine_with_search_config() {
        // Test SearchConfig with Cosine distance on SIFT-like data
        let (_temp_dir, storage) = create_test_storage();
        let dim = 128;
        let n_vectors = 1000;
        let n_queries = 20;
        let k = 10;

        let vectors = generate_sift_like_vectors(n_vectors, 42);
        let queries = generate_sift_like_vectors(n_queries, 123);

        // Create embedding with Cosine distance
        let embedding = make_test_embedding(1, dim as u32, Distance::Cosine);

        let config = Config {
            dim,
            m: 16,
            m_max: 32,
            m_max_0: 32,
            ef_construction: 200,
            ..Default::default()
        };

        let index = build_index(
            &storage,
            embedding.code(),
            embedding.distance(),
            &vectors,
            &config,
        );

        // Create SearchConfig - should auto-select RaBitQ for Cosine
        let search_config = SearchConfig::new(embedding.clone(), k).with_ef(100);
        assert!(search_config.strategy().is_rabitq());

        // Test can force Exact strategy
        let exact_config = SearchConfig::new(embedding.clone(), k).with_ef(100).exact();
        assert!(exact_config.strategy().is_exact());

        // Validate embedding matches
        assert!(search_config
            .validate_embedding_code(index.embedding())
            .is_ok());

        // Test recall using search_config's distance (currently uses L2 internally,
        // but we verify the config would use Cosine)
        let mut total_recall = 0.0;
        for query in &queries {
            let results = index
                .search(&storage, query, k, search_config.ef())
                .expect("Search failed");
            // Ground truth computed with Cosine distance
            let ground_truth = brute_force_knn_with_distance(query, &vectors, k, Distance::Cosine);
            total_recall += compute_recall(&results, &ground_truth);
        }

        let avg_recall = total_recall / n_queries as f64;
        // Note: Current search() uses L2 internally, so recall with Cosine ground truth may differ
        // This test primarily verifies SearchConfig setup is correct
        assert!(
            avg_recall >= 0.60,
            "Cosine recall@{} should be >= 60% (L2 approximation), got {:.1}%",
            k,
            avg_recall * 100.0
        );
    }

    #[test]
    fn test_search_config_embedding_mismatch_detection() {
        // Test that SearchConfig detects embedding code mismatches
        let (_temp_dir, storage) = create_test_storage();
        let dim = 64;
        let n_vectors = 100;

        let vectors = generate_random_vectors(n_vectors, dim, 42);

        // Create index with embedding code 1
        let config = Config::for_dim(dim);
        let index = build_index(&storage, 1, Distance::L2, &vectors, &config);

        // Create SearchConfig with different embedding code (2)
        let wrong_embedding = make_test_embedding(2, dim as u32, Distance::L2);
        let search_config = SearchConfig::new(wrong_embedding, 10);

        // Validation should fail
        let result = search_config.validate_embedding_code(index.embedding());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("mismatch"));
    }

    #[test]
    fn test_search_config_parameters_propagation() {
        // Test that SearchConfig parameters are accessible
        let embedding = make_test_embedding(1, 128, Distance::L2);
        let config = SearchConfig::new(embedding, 10)
            .with_ef(150)
            .with_rerank_factor(8)
            .with_k(20);

        assert_eq!(config.k(), 20);
        assert_eq!(config.ef(), 150);
        assert_eq!(config.rerank_factor(), 8);
        assert_eq!(config.distance(), Distance::L2);
        assert_eq!(config.embedding().code(), 1);
    }

    #[test]
    fn test_sift_1k_different_k_values() {
        // Test SearchConfig with different k values
        let (_temp_dir, storage) = create_test_storage();
        let dim = 128;
        let n_vectors = 1000;
        let n_queries = 10;

        let vectors = generate_sift_like_vectors(n_vectors, 42);
        let queries = generate_sift_like_vectors(n_queries, 123);

        let embedding = make_test_embedding(1, dim as u32, Distance::L2);
        let hnsw_config = Config {
            dim,
            m: 16,
            ef_construction: 200,
            ..Default::default()
        };

        let index = build_index(
            &storage,
            embedding.code(),
            embedding.distance(),
            &vectors,
            &hnsw_config,
        );

        // Test different k values
        for k in [1, 5, 10, 20] {
            let search_config = SearchConfig::new(embedding.clone(), k).with_ef(k.max(50));

            let mut total_recall = 0.0;
            for query in &queries {
                let results = index
                    .search(&storage, query, search_config.k(), search_config.ef())
                    .expect("Search failed");
                let ground_truth = brute_force_knn_with_distance(query, &vectors, k, Distance::L2);
                total_recall += compute_recall(&results, &ground_truth);
            }

            let avg_recall = total_recall / n_queries as f64;
            let min_recall = if k == 1 { 0.80 } else { 0.85 };
            assert!(
                avg_recall >= min_recall,
                "Recall@{} should be >= {:.0}%, got {:.1}%",
                k,
                min_recall * 100.0,
                avg_recall * 100.0
            );
        }
    }
}
