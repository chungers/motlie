//! Parallel search utilities using rayon.
//!
//! This module provides CPU-parallelized versions of search operations,
//! particularly for the reranking phase where we compute exact distances
//! for candidate vectors.
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
//! use motlie_db::vector::parallel;
//!
//! // Parallel reranking of candidates
//! let results = parallel::rerank_parallel(
//!     &candidates,
//!     |vec_id| index.distance(&storage, query, vec_id).ok(),
//!     k,
//! );
//! ```

use rayon::prelude::*;

use super::schema::VecId;

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
}
