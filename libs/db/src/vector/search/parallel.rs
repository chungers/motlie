//! Parallel search utilities using rayon.
//!
//! This module provides CPU-parallelized versions of search operations,
//! particularly for the reranking phase where we compute exact distances
//! for candidate vectors.
//!
//! # Performance Threshold
//!
//! Rayon parallelism has overhead that only pays off for larger candidate sets.
//! Use [`rerank_adaptive`] which automatically selects sequential vs parallel
//! based on candidate count:
//!
//! | Candidates | Sequential | Parallel | Recommendation |
//! |------------|------------|----------|----------------|
//! | < 400      | Faster     | Slower   | Sequential     |
//! | 400-800    | ~Equal     | ~Equal   | Either         |
//! | > 800      | Slower     | Faster   | Parallel       |
//!
//! Benchmark: 512D LAION-CLIP vectors, NEON SIMD, aarch64.
//! See `libs/db/src/vector/ROADMAP.md` Task 4.20 for full analysis.
//!
//! # Thread Safety
//!
//! RocksDB `TransactionDB` supports concurrent reads from multiple threads:
//! - `get_cf()` and `multi_get_cf()` are thread-safe
//! - Each rayon worker gets its own snapshot view
//! - No locking needed for read-only operations
//!
//! # Example
//!
//! ```ignore
//! use motlie_db::vector::search::rerank_adaptive;
//! use motlie_db::vector::search::DEFAULT_PARALLEL_RERANK_THRESHOLD;
//!
//! // Smart reranking - auto-selects sequential vs parallel
//! let results = rerank_adaptive(
//!     &candidates,
//!     |vec_id| index.distance(&storage, query, vec_id).ok(),
//!     k,
//!     DEFAULT_PARALLEL_RERANK_THRESHOLD,
//! );
//! ```

use rayon::prelude::*;

use crate::vector::schema::VecId;
use super::config::DEFAULT_PARALLEL_RERANK_THRESHOLD;

/// Parallel reranking - compute exact distances for candidates.
///
/// Uses rayon's parallel iterator to distribute distance computation
/// across multiple CPU cores. Each worker thread has safe read access
/// to RocksDB through the provided closure.
///
/// # Arguments
///
/// * `candidates` - Vector IDs to compute distances for
/// * `distance_fn` - Closure that computes distance for a single vector ID.
///                   Returns `None` if vector not found (will be filtered out).
/// * `k` - Number of top results to return
///
/// # Returns
///
/// Top-k results sorted by distance (ascending).
///
/// # Example
///
/// ```ignore
/// let results = rerank_parallel(
///     &vec_ids,
///     |id| index.distance(&storage, query, id).ok(),
///     10,
/// );
/// ```
pub fn rerank_parallel<F>(candidates: &[VecId], distance_fn: F, k: usize) -> Vec<(f32, VecId)>
where
    F: Fn(VecId) -> Option<f32> + Sync,
{
    // Parallel distance computation
    let mut results: Vec<(f32, VecId)> = candidates
        .par_iter()
        .filter_map(|&id| distance_fn(id).map(|d| (d, id)))
        .collect();

    // Sort and truncate (sequential, but on smaller result set)
    results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

/// Sequential reranking - compute exact distances without parallelism.
///
/// Faster than parallel for small candidate sets (< 800) due to rayon overhead.
///
/// # Arguments
///
/// * `candidates` - Vector IDs to compute distances for
/// * `distance_fn` - Closure that computes distance for a single vector ID.
///                   Returns `None` if vector not found (will be filtered out).
/// * `k` - Number of top results to return
///
/// # Returns
///
/// Top-k results sorted by distance (ascending).
pub fn rerank_sequential<F>(candidates: &[VecId], distance_fn: F, k: usize) -> Vec<(f32, VecId)>
where
    F: Fn(VecId) -> Option<f32>,
{
    let mut results: Vec<(f32, VecId)> = candidates
        .iter()
        .filter_map(|&id| distance_fn(id).map(|d| (d, id)))
        .collect();

    results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

/// Adaptive reranking - automatically selects sequential or parallel.
///
/// Uses the threshold to decide:
/// - `candidates.len() >= threshold` → parallel (rayon)
/// - `candidates.len() < threshold` → sequential
///
/// Default threshold: 800 (see [`DEFAULT_PARALLEL_RERANK_THRESHOLD`]).
///
/// # Arguments
///
/// * `candidates` - Vector IDs to compute distances for
/// * `distance_fn` - Closure that computes distance for a single vector ID.
///                   Returns `None` if vector not found (will be filtered out).
/// * `k` - Number of top results to return
/// * `threshold` - Minimum candidate count to use parallel. Use
///                 [`DEFAULT_PARALLEL_RERANK_THRESHOLD`] for the tuned default.
///
/// # Returns
///
/// Top-k results sorted by distance (ascending).
///
/// # Example
///
/// ```ignore
/// use motlie_db::vector::search::rerank_adaptive;
/// use motlie_db::vector::search::DEFAULT_PARALLEL_RERANK_THRESHOLD;
///
/// // Auto-select based on candidate count
/// let results = rerank_adaptive(
///     &candidates,
///     |id| Some(compute_distance(query, id)),
///     10,
///     DEFAULT_PARALLEL_RERANK_THRESHOLD,
/// );
///
/// // Or use config's threshold
/// let results = rerank_adaptive(
///     &candidates,
///     |id| Some(compute_distance(query, id)),
///     config.k(),
///     config.parallel_rerank_threshold(),
/// );
/// ```
///
/// # Performance
///
/// Benchmark results (512D LAION-CLIP, NEON SIMD):
///
/// | Candidates | Speedup (parallel vs sequential) |
/// |------------|----------------------------------|
/// | 50         | 0.48x (sequential 2x faster)     |
/// | 400        | 0.94x (roughly equal)            |
/// | 800        | 1.04x (crossover point)          |
/// | 1600       | 1.41x                            |
/// | 6400       | 2.34x                            |
/// | 12800      | 3.15x                            |
///
/// See: `libs/db/src/vector/ROADMAP.md` Task 4.20 for full analysis.
pub fn rerank_adaptive<F>(
    candidates: &[VecId],
    distance_fn: F,
    k: usize,
    threshold: usize,
) -> Vec<(f32, VecId)>
where
    F: Fn(VecId) -> Option<f32> + Sync,
{
    if candidates.len() >= threshold {
        rerank_parallel(candidates, distance_fn, k)
    } else {
        rerank_sequential(candidates, distance_fn, k)
    }
}

/// Adaptive reranking using the default threshold.
///
/// Convenience wrapper for [`rerank_adaptive`] using
/// [`DEFAULT_PARALLEL_RERANK_THRESHOLD`] (800).
///
/// # Example
///
/// ```ignore
/// let results = rerank_auto(&candidates, |id| Some(distance(id)), 10);
/// ```
pub fn rerank_auto<F>(candidates: &[VecId], distance_fn: F, k: usize) -> Vec<(f32, VecId)>
where
    F: Fn(VecId) -> Option<f32> + Sync,
{
    rerank_adaptive(candidates, distance_fn, k, DEFAULT_PARALLEL_RERANK_THRESHOLD)
}

/// Parallel batch distance computation.
///
/// Computes distances from a query to multiple vectors in parallel.
/// Unlike `rerank_parallel`, this doesn't sort or truncate - it returns
/// all distances for further processing.
///
/// # Arguments
///
/// * `candidates` - Vector IDs to compute distances for
/// * `distance_fn` - Closure that computes distance for a single vector ID
///
/// # Returns
///
/// All (vec_id, distance) pairs where distance computation succeeded.
pub fn batch_distances_parallel<F>(candidates: &[VecId], distance_fn: F) -> Vec<(VecId, f32)>
where
    F: Fn(VecId) -> Option<f32> + Sync,
{
    candidates
        .par_iter()
        .filter_map(|&id| distance_fn(id).map(|d| (id, d)))
        .collect()
}

/// Parallel distance computation with pre-fetched vectors.
///
/// When vectors are already loaded (e.g., from batch fetch), this computes
/// distances in parallel without additional RocksDB access.
///
/// # Arguments
///
/// * `query` - Query vector
/// * `vectors` - Pre-fetched (vec_id, vector) pairs
/// * `distance_fn` - Distance function (e.g., L2, Cosine)
///
/// # Returns
///
/// All (distance, vec_id) pairs sorted by distance.
pub fn distances_from_vectors_parallel<F>(
    query: &[f32],
    vectors: &[(VecId, Vec<f32>)],
    distance_fn: F,
) -> Vec<(f32, VecId)>
where
    F: Fn(&[f32], &[f32]) -> f32 + Sync,
{
    let mut results: Vec<(f32, VecId)> = vectors
        .par_iter()
        .map(|(id, vec)| (distance_fn(query, vec), *id))
        .collect();

    results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rerank_parallel_basic() {
        // Simulate distance computation
        let candidates: Vec<VecId> = (0..100).collect();
        let results = rerank_parallel(&candidates, |id| Some(100.0 - id as f32), 10);

        assert_eq!(results.len(), 10);
        // Smallest distances should be at the end (100-99=1, 100-98=2, etc.)
        assert_eq!(results[0].1, 99); // distance = 1.0
        assert_eq!(results[1].1, 98); // distance = 2.0
    }

    #[test]
    fn test_rerank_parallel_with_missing() {
        let candidates: Vec<VecId> = (0..50).collect();
        // Skip even IDs
        let results = rerank_parallel(
            &candidates,
            |id| {
                if id % 2 == 0 {
                    None
                } else {
                    Some(id as f32)
                }
            },
            5,
        );

        assert_eq!(results.len(), 5);
        // Should only have odd IDs, sorted by distance
        assert_eq!(results[0].1, 1);
        assert_eq!(results[1].1, 3);
    }

    #[test]
    fn test_rerank_parallel_fewer_than_k() {
        let candidates: Vec<VecId> = (0..5).collect();
        let results = rerank_parallel(&candidates, |id| Some(id as f32), 10);

        assert_eq!(results.len(), 5); // Only 5 candidates
    }

    #[test]
    fn test_batch_distances_parallel() {
        let candidates: Vec<VecId> = (0..100).collect();
        let results = batch_distances_parallel(&candidates, |id| Some(id as f32 * 2.0));

        assert_eq!(results.len(), 100);
        // Results are not sorted
        assert!(results.iter().any(|(id, d)| *id == 50 && *d == 100.0));
    }

    #[test]
    fn test_distances_from_vectors_parallel() {
        let query = vec![1.0, 0.0, 0.0];
        let vectors: Vec<(VecId, Vec<f32>)> = vec![
            (0, vec![1.0, 0.0, 0.0]), // distance = 0
            (1, vec![0.0, 1.0, 0.0]), // distance = 2 (L2 squared)
            (2, vec![0.0, 0.0, 1.0]), // distance = 2
        ];

        let l2_squared = |a: &[f32], b: &[f32]| -> f32 {
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
        };

        let results = distances_from_vectors_parallel(&query, &vectors, l2_squared);

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].1, 0); // Closest
        assert!((results[0].0 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_rerank_sequential_basic() {
        let candidates: Vec<VecId> = (0..100).collect();
        let results = rerank_sequential(&candidates, |id| Some(100.0 - id as f32), 10);

        assert_eq!(results.len(), 10);
        assert_eq!(results[0].1, 99); // distance = 1.0
        assert_eq!(results[1].1, 98); // distance = 2.0
    }

    #[test]
    fn test_rerank_sequential_matches_parallel() {
        let candidates: Vec<VecId> = (0..1000).collect();
        let distance_fn = |id: VecId| Some((id as f32 * 17.0) % 100.0); // Pseudo-random

        let seq_results = rerank_sequential(&candidates, distance_fn, 10);
        let par_results = rerank_parallel(&candidates, distance_fn, 10);

        // Results should be identical
        assert_eq!(seq_results.len(), par_results.len());
        for (s, p) in seq_results.iter().zip(par_results.iter()) {
            assert_eq!(s.1, p.1); // Same IDs
            assert!((s.0 - p.0).abs() < 1e-6); // Same distances
        }
    }

    #[test]
    fn test_rerank_adaptive_uses_sequential_below_threshold() {
        let candidates: Vec<VecId> = (0..100).collect();
        let threshold = 500; // Above candidate count

        // Should use sequential path
        let results = rerank_adaptive(&candidates, |id| Some(id as f32), 10, threshold);
        assert_eq!(results.len(), 10);
        assert_eq!(results[0].1, 0); // Smallest distance
    }

    #[test]
    fn test_rerank_adaptive_uses_parallel_above_threshold() {
        let candidates: Vec<VecId> = (0..1000).collect();
        let threshold = 500; // Below candidate count

        // Should use parallel path
        let results = rerank_adaptive(&candidates, |id| Some(id as f32), 10, threshold);
        assert_eq!(results.len(), 10);
        assert_eq!(results[0].1, 0); // Smallest distance
    }

    #[test]
    fn test_rerank_adaptive_at_threshold() {
        let candidates: Vec<VecId> = (0..800).collect();
        let threshold = 800; // Exactly at threshold

        // Should use parallel (>= threshold)
        let results = rerank_adaptive(&candidates, |id| Some(id as f32), 10, threshold);
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_rerank_auto_uses_default_threshold() {
        let candidates: Vec<VecId> = (0..100).collect();

        // With default threshold of 800, 100 candidates should use sequential
        let results = rerank_auto(&candidates, |id| Some(id as f32), 10);
        assert_eq!(results.len(), 10);
        assert_eq!(results[0].1, 0);
    }

    #[test]
    fn test_default_parallel_rerank_threshold() {
        // Verify the default is 800 as documented
        assert_eq!(DEFAULT_PARALLEL_RERANK_THRESHOLD, 800);
    }
}
