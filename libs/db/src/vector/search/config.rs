//! Search configuration with Embedding-driven validation.
//!
//! This module provides type-safe search configuration that uses `Embedding` as
//! the single source of truth for distance metrics and search strategies.
//!
//! # Design Philosophy
//!
//! The `Embedding` struct contains the distance metric used to build the index.
//! `SearchConfig` captures this and:
//! 1. Validates the embedding matches the index being searched
//! 2. Auto-selects optimal strategy based on distance metric
//! 3. Prevents incompatible configurations at construction time
//!
//! # Example
//!
//! ```ignore
//! // Cosine index - RaBitQ auto-selected
//! let config = SearchConfig::new(cosine_embedding.clone(), 10);
//! let results = index.search(&storage, &config, &query)?;
//!
//! // Force exact search if needed
//! let config = SearchConfig::new(cosine_embedding.clone(), 10).exact();
//! ```

use crate::vector::distance::Distance;
use crate::vector::embedding::Embedding;

// ============================================================================
// SearchStrategy
// ============================================================================

/// Search strategy - determines how distance computation is performed.
///
/// Auto-selected based on `embedding.distance()`:
/// - Cosine → RaBitQ (Hamming approximates angular distance)
/// - L2/DotProduct → Exact (Hamming not compatible)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchStrategy {
    /// Exact distance computation at all layers.
    /// Works with all distance metrics.
    Exact,

    /// RaBitQ: Hamming filtering at layer 0 + exact re-rank.
    /// Only valid when embedding.distance == Cosine.
    /// Requires BinaryCodeCache for good performance.
    RaBitQ {
        /// Use in-memory binary code cache (required for good performance)
        use_cache: bool,
    },
}

impl SearchStrategy {
    /// Check if this strategy uses RaBitQ.
    pub fn is_rabitq(&self) -> bool {
        matches!(self, SearchStrategy::RaBitQ { .. })
    }

    /// Check if this strategy uses exact distance computation.
    pub fn is_exact(&self) -> bool {
        matches!(self, SearchStrategy::Exact)
    }
}

impl std::fmt::Display for SearchStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SearchStrategy::Exact => write!(f, "Exact"),
            SearchStrategy::RaBitQ { use_cache: true } => write!(f, "RaBitQ(cached)"),
            SearchStrategy::RaBitQ { use_cache: false } => write!(f, "RaBitQ(uncached)"),
        }
    }
}

// ============================================================================
// SearchConfig
// ============================================================================

/// Search configuration derived from Embedding.
///
/// Enforces correct distance metric and search strategy by using `Embedding`
/// as the single source of truth.
///
/// # Validation
///
/// - Embedding code is validated against the index at search time
/// - RaBitQ strategy is only allowed for Cosine distance
/// - Distance computations use the embedding's configured metric
///
/// # Example
///
/// ```ignore
/// // Create config - strategy auto-selected based on distance metric
/// let config = SearchConfig::new(embedding.clone(), 10)
///     .with_ef(150)
///     .with_rerank_factor(4);
///
/// // Search using the config
/// let results = index.search(&storage, &config, &query)?;
/// ```
/// Default threshold for parallel re-ranking (3200 candidates).
///
/// Benchmark results from `--rabitq-benchmark` on 512D LAION-CLIP vectors
/// (January 2026, aarch64 NEON):
///
/// | Candidates | Sequential | Parallel | Speedup |
/// |------------|------------|----------|---------|
/// | 50         | 6.6ms      | 19.0ms   | 0.35x   |
/// | 100        | 11.4ms     | 30.0ms   | 0.38x   |
/// | 400        | 45.6ms     | 91.2ms   | 0.50x   |
/// | 800        | 106.1ms    | 139.2ms  | 0.76x   |
/// | 1600       | 188.3ms    | 205.8ms  | 0.91x   |
/// | 3200       | 414.4ms    | 319.3ms  | 1.30x   | ← crossover
///
/// Note: Crossover moved from 800 (original tuning) to 3200 because sequential
/// distance computation got 25-30% faster due to code/compiler optimizations,
/// while rayon overhead remained constant.
///
/// See: `libs/db/src/vector/BENCHMARK.md` for full analysis.
pub const DEFAULT_PARALLEL_RERANK_THRESHOLD: usize = 3200;

#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// Embedding space specification (source of truth)
    embedding: Embedding,
    /// Search strategy (derived from embedding.distance + user choice)
    strategy: SearchStrategy,
    /// Number of results to return
    k: usize,
    /// Search beam width (ef parameter)
    ef: usize,
    /// Re-rank factor for RaBitQ (ignored for Exact)
    rerank_factor: usize,
    /// Minimum candidate count to use parallel (rayon) re-ranking.
    ///
    /// Below this threshold, sequential re-ranking is faster due to rayon overhead.
    /// Default: 3200 (based on LAION-CLIP 512D benchmarks, January 2026).
    ///
    /// Set to 0 to always use parallel, or `usize::MAX` to always use sequential.
    ///
    /// See: `libs/db/src/vector/BENCHMARK.md` for benchmark methodology.
    parallel_rerank_threshold: usize,
}

impl SearchConfig {
    /// Create search config with automatic strategy selection.
    ///
    /// Strategy is chosen based on embedding's distance metric:
    /// - Cosine → RaBitQ (Hamming approximates angular distance)
    /// - L2/DotProduct → Exact (Hamming not compatible)
    ///
    /// # Arguments
    ///
    /// * `embedding` - The embedding space (source of truth for distance metric)
    /// * `k` - Number of results to return
    pub fn new(embedding: Embedding, k: usize) -> Self {
        let strategy = match embedding.distance() {
            Distance::Cosine => SearchStrategy::RaBitQ { use_cache: true },
            Distance::L2 | Distance::DotProduct => SearchStrategy::Exact,
        };
        Self {
            embedding,
            strategy,
            k,
            ef: 100,          // sensible default
            rerank_factor: 10, // 10x re-ranking for ~90% recall at 10K scale
            parallel_rerank_threshold: DEFAULT_PARALLEL_RERANK_THRESHOLD,
        }
    }

    /// Create config with exact search strategy (bypass auto-selection).
    ///
    /// Safe for all distance metrics.
    pub fn new_exact(embedding: Embedding, k: usize) -> Self {
        Self {
            embedding,
            strategy: SearchStrategy::Exact,
            k,
            ef: 100,
            rerank_factor: 10,
            parallel_rerank_threshold: DEFAULT_PARALLEL_RERANK_THRESHOLD,
        }
    }

    // ─────────────────────────────────────────────────────────────
    // Builder Methods
    // ─────────────────────────────────────────────────────────────

    /// Builder: Override to use exact search (disable RaBitQ).
    ///
    /// Safe for all distance metrics.
    pub fn exact(mut self) -> Self {
        self.strategy = SearchStrategy::Exact;
        self
    }

    /// Builder: Force RaBitQ strategy.
    ///
    /// # Errors
    ///
    /// Returns error if `embedding.distance() != Cosine` because
    /// Hamming distance only approximates angular (cosine) distance.
    pub fn rabitq(mut self) -> anyhow::Result<Self> {
        if self.embedding.distance() != Distance::Cosine {
            return Err(anyhow::anyhow!(
                "RaBitQ requires Cosine distance (Hamming ≈ angular), got {:?}",
                self.embedding.distance()
            ));
        }
        self.strategy = SearchStrategy::RaBitQ { use_cache: true };
        Ok(self)
    }

    /// Builder: Force RaBitQ without cache (not recommended).
    ///
    /// # Errors
    ///
    /// Returns error if `embedding.distance() != Cosine`.
    pub fn rabitq_uncached(mut self) -> anyhow::Result<Self> {
        if self.embedding.distance() != Distance::Cosine {
            return Err(anyhow::anyhow!(
                "RaBitQ requires Cosine distance (Hamming ≈ angular), got {:?}",
                self.embedding.distance()
            ));
        }
        self.strategy = SearchStrategy::RaBitQ { use_cache: false };
        Ok(self)
    }

    /// Builder: Set ef (beam width).
    ///
    /// Higher ef = better recall but slower search.
    /// Typical values: 50-200.
    pub fn with_ef(mut self, ef: usize) -> Self {
        self.ef = ef;
        self
    }

    /// Builder: Set re-rank factor (for RaBitQ).
    ///
    /// Number of candidates to re-rank = k * rerank_factor.
    /// Higher factor = better recall but more I/O.
    /// Typical values: 2-8.
    pub fn with_rerank_factor(mut self, factor: usize) -> Self {
        self.rerank_factor = factor;
        self
    }

    /// Builder: Set k (number of results).
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Builder: Set parallel re-ranking threshold.
    ///
    /// Controls when rayon parallel re-ranking is used vs sequential:
    /// - `candidates >= threshold` → parallel (rayon)
    /// - `candidates < threshold` → sequential
    ///
    /// Default: 3200 (based on LAION-CLIP 512D benchmarks, January 2026).
    ///
    /// # Tuning Guidance
    ///
    /// The crossover point depends on:
    /// - Vector dimensionality (higher D → more work per distance → lower threshold)
    /// - CPU core count (more cores → better parallel scaling)
    /// - Distance metric complexity
    ///
    /// Benchmark results (512D, NEON SIMD, January 2026):
    /// - 800 candidates: parallel is 0.76x (overhead)
    /// - 1600 candidates: parallel is 0.91x (near equal)
    /// - 3200 candidates: parallel is 1.30x (crossover)
    ///
    /// See: `libs/db/src/vector/BENCHMARK.md` for full analysis.
    pub fn with_parallel_rerank_threshold(mut self, threshold: usize) -> Self {
        self.parallel_rerank_threshold = threshold;
        self
    }

    // ─────────────────────────────────────────────────────────────
    // Accessors
    // ─────────────────────────────────────────────────────────────

    /// Get the embedding (source of truth).
    #[inline]
    pub fn embedding(&self) -> &Embedding {
        &self.embedding
    }

    /// Get the search strategy.
    #[inline]
    pub fn strategy(&self) -> SearchStrategy {
        self.strategy
    }

    /// Get k (number of results).
    #[inline]
    pub fn k(&self) -> usize {
        self.k
    }

    /// Get ef (beam width).
    #[inline]
    pub fn ef(&self) -> usize {
        self.ef
    }

    /// Get re-rank factor.
    #[inline]
    pub fn rerank_factor(&self) -> usize {
        self.rerank_factor
    }

    /// Get parallel re-ranking threshold.
    ///
    /// Returns the minimum candidate count to use parallel (rayon) re-ranking.
    /// Below this threshold, sequential is faster due to rayon overhead.
    #[inline]
    pub fn parallel_rerank_threshold(&self) -> usize {
        self.parallel_rerank_threshold
    }

    /// Check if parallel re-ranking should be used for the given candidate count.
    ///
    /// Returns `true` if `candidate_count >= parallel_rerank_threshold`.
    #[inline]
    pub fn should_use_parallel_rerank(&self, candidate_count: usize) -> bool {
        candidate_count >= self.parallel_rerank_threshold
    }

    /// Get the distance metric from the embedding.
    #[inline]
    pub fn distance(&self) -> Distance {
        self.embedding.distance()
    }

    // ─────────────────────────────────────────────────────────────
    // Distance Computation (delegated to Embedding)
    // ─────────────────────────────────────────────────────────────

    /// Compute distance using embedding's configured metric.
    ///
    /// This is the ONLY way distances should be computed during search,
    /// ensuring the metric matches what was used to build the index.
    #[inline]
    pub fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        self.embedding.compute_distance(a, b)
    }

    // ─────────────────────────────────────────────────────────────
    // Validation
    // ─────────────────────────────────────────────────────────────

    /// Validate that this config's embedding matches the given embedding code.
    ///
    /// Should be called at the start of search to ensure the config was
    /// created with the correct embedding for this index.
    #[inline]
    pub fn validate_embedding_code(&self, expected_code: u64) -> anyhow::Result<()> {
        if self.embedding.code() != expected_code {
            return Err(anyhow::anyhow!(
                "Embedding mismatch: index has code {}, config has {}",
                expected_code,
                self.embedding.code()
            ));
        }
        Ok(())
    }
}

impl std::fmt::Display for SearchConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SearchConfig {{ embedding: {}, strategy: {}, k: {}, ef: {}, rerank: {}, parallel_threshold: {} }}",
            self.embedding, self.strategy, self.k, self.ef, self.rerank_factor, self.parallel_rerank_threshold
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedding(distance: Distance) -> Embedding {
        Embedding::new(1, "test", 128, distance, None)
    }

    #[test]
    fn test_auto_strategy_cosine() {
        let emb = make_embedding(Distance::Cosine);
        let config = SearchConfig::new(emb, 10);

        assert!(config.strategy().is_rabitq());
        assert_eq!(config.k(), 10);
        assert_eq!(config.ef(), 100);
        assert_eq!(config.rerank_factor(), 10); // Tuned for ~90% recall at 10K scale
    }

    #[test]
    fn test_auto_strategy_l2() {
        let emb = make_embedding(Distance::L2);
        let config = SearchConfig::new(emb, 10);

        assert!(config.strategy().is_exact());
    }

    #[test]
    fn test_auto_strategy_dot_product() {
        let emb = make_embedding(Distance::DotProduct);
        let config = SearchConfig::new(emb, 10);

        assert!(config.strategy().is_exact());
    }

    #[test]
    fn test_builder_exact() {
        let emb = make_embedding(Distance::Cosine);
        let config = SearchConfig::new(emb, 10).exact();

        assert!(config.strategy().is_exact());
    }

    #[test]
    fn test_builder_rabitq_cosine_ok() {
        let emb = make_embedding(Distance::Cosine);
        let config = SearchConfig::new(emb, 10).exact().rabitq();

        assert!(config.is_ok());
        assert!(config.unwrap().strategy().is_rabitq());
    }

    #[test]
    fn test_builder_rabitq_l2_fails() {
        let emb = make_embedding(Distance::L2);
        let config = SearchConfig::new(emb, 10).rabitq();

        assert!(config.is_err());
        assert!(config.unwrap_err().to_string().contains("Cosine"));
    }

    #[test]
    fn test_builder_ef_and_rerank() {
        let emb = make_embedding(Distance::Cosine);
        let config = SearchConfig::new(emb, 10).with_ef(200).with_rerank_factor(8);

        assert_eq!(config.ef(), 200);
        assert_eq!(config.rerank_factor(), 8);
    }

    #[test]
    fn test_validate_embedding_code_ok() {
        let emb = make_embedding(Distance::Cosine);
        let config = SearchConfig::new(emb, 10);

        assert!(config.validate_embedding_code(1).is_ok());
    }

    #[test]
    fn test_validate_embedding_code_mismatch() {
        let emb = make_embedding(Distance::Cosine);
        let config = SearchConfig::new(emb, 10);

        let result = config.validate_embedding_code(999);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("mismatch"));
    }

    #[test]
    fn test_compute_distance_cosine() {
        let emb = make_embedding(Distance::Cosine);
        let config = SearchConfig::new(emb, 10);

        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];

        // Orthogonal vectors have cosine similarity 0, distance 1
        let dist = config.compute_distance(&a, &b);
        assert!((dist - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_distance_l2() {
        let emb = make_embedding(Distance::L2);
        let config = SearchConfig::new(emb, 10);

        let a = vec![0.0, 0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0, 0.0];

        // L2 distance = sqrt(9 + 16) = 5
        let dist = config.compute_distance(&a, &b);
        assert!((dist - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_new_exact() {
        let emb = make_embedding(Distance::Cosine);
        let config = SearchConfig::new_exact(emb, 10);

        // Even for Cosine, new_exact forces Exact strategy
        assert!(config.strategy().is_exact());
    }

    #[test]
    fn test_display() {
        let emb = make_embedding(Distance::Cosine);
        let config = SearchConfig::new(emb, 10);

        let display = format!("{}", config);
        assert!(display.contains("RaBitQ"));
        assert!(display.contains("k: 10"));
    }

    #[test]
    fn test_strategy_display() {
        assert_eq!(format!("{}", SearchStrategy::Exact), "Exact");
        assert_eq!(
            format!("{}", SearchStrategy::RaBitQ { use_cache: true }),
            "RaBitQ(cached)"
        );
        assert_eq!(
            format!("{}", SearchStrategy::RaBitQ { use_cache: false }),
            "RaBitQ(uncached)"
        );
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_rabitq_uncached_cosine_ok() {
        let emb = make_embedding(Distance::Cosine);
        let config = SearchConfig::new(emb, 10).rabitq_uncached();

        assert!(config.is_ok());
        let config = config.unwrap();
        assert!(matches!(
            config.strategy(),
            SearchStrategy::RaBitQ { use_cache: false }
        ));
    }

    #[test]
    fn test_rabitq_uncached_l2_fails() {
        let emb = make_embedding(Distance::L2);
        let config = SearchConfig::new(emb, 10).rabitq_uncached();

        assert!(config.is_err());
    }

    #[test]
    fn test_builder_chain_all() {
        let emb = make_embedding(Distance::Cosine);
        let config = SearchConfig::new(emb, 5)
            .with_ef(200)
            .with_rerank_factor(8)
            .with_k(20)
            .exact();

        assert_eq!(config.k(), 20);
        assert_eq!(config.ef(), 200);
        assert_eq!(config.rerank_factor(), 8);
        assert!(config.strategy().is_exact());
    }

    #[test]
    fn test_distance_accessor() {
        let emb = make_embedding(Distance::DotProduct);
        let config = SearchConfig::new(emb, 10);

        assert_eq!(config.distance(), Distance::DotProduct);
    }

    #[test]
    fn test_embedding_accessor() {
        let emb = make_embedding(Distance::L2);
        let config = SearchConfig::new(emb.clone(), 10);

        assert_eq!(config.embedding().code(), 1);
        assert_eq!(config.embedding().distance(), Distance::L2);
    }

    #[test]
    fn test_compute_distance_dot_product() {
        let emb = make_embedding(Distance::DotProduct);
        let config = SearchConfig::new(emb, 10);

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];

        // DotProduct returns -dot(a, b) = -(1+2+3+4) = -10
        let dist = config.compute_distance(&a, &b);
        assert!((dist - (-10.0)).abs() < 0.01);
    }

    #[test]
    fn test_k_zero() {
        let emb = make_embedding(Distance::Cosine);
        let config = SearchConfig::new(emb, 0);

        assert_eq!(config.k(), 0);
    }

    #[test]
    fn test_ef_zero() {
        let emb = make_embedding(Distance::Cosine);
        let config = SearchConfig::new(emb, 10).with_ef(0);

        assert_eq!(config.ef(), 0);
    }

    #[test]
    fn test_large_values() {
        let emb = make_embedding(Distance::Cosine);
        let config = SearchConfig::new(emb, 1_000_000)
            .with_ef(10_000)
            .with_rerank_factor(100);

        assert_eq!(config.k(), 1_000_000);
        assert_eq!(config.ef(), 10_000);
        assert_eq!(config.rerank_factor(), 100);
    }

    #[test]
    fn test_multiple_embeddings_different_codes() {
        // Test that different embedding codes are properly distinguished
        let emb1 = Embedding::new(1, "model1", 128, Distance::Cosine, None);
        let emb2 = Embedding::new(2, "model2", 768, Distance::L2, None);

        let config1 = SearchConfig::new(emb1, 10);
        let config2 = SearchConfig::new(emb2, 10);

        assert!(config1.validate_embedding_code(1).is_ok());
        assert!(config1.validate_embedding_code(2).is_err());

        assert!(config2.validate_embedding_code(2).is_ok());
        assert!(config2.validate_embedding_code(1).is_err());
    }

    #[test]
    fn test_strategy_is_rabitq() {
        assert!(!SearchStrategy::Exact.is_rabitq());
        assert!(SearchStrategy::RaBitQ { use_cache: true }.is_rabitq());
        assert!(SearchStrategy::RaBitQ { use_cache: false }.is_rabitq());
    }

    #[test]
    fn test_strategy_is_exact() {
        assert!(SearchStrategy::Exact.is_exact());
        assert!(!SearchStrategy::RaBitQ { use_cache: true }.is_exact());
        assert!(!SearchStrategy::RaBitQ { use_cache: false }.is_exact());
    }

    #[test]
    fn test_display_exact_strategy() {
        let emb = make_embedding(Distance::L2);
        let config = SearchConfig::new(emb, 10);

        let display = format!("{}", config);
        assert!(display.contains("Exact"));
        assert!(display.contains("k: 10"));
    }
}
