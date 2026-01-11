//! HNSW search algorithm implementations.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use anyhow::Result;
use ordered_float::OrderedFloat;
use roaring::RoaringBitmap;

use super::graph::{distance, get_neighbors, greedy_search_layer};
use super::insert::load_navigation;
use super::Index;
use crate::vector::cache::BinaryCodeCache;
use crate::vector::rabitq::RaBitQ;
use crate::vector::schema::{HnswLayer, VecId};
use crate::vector::Storage;

/// Search for k nearest neighbors.
///
/// # Arguments
/// * `index` - The HNSW index
/// * `storage` - RocksDB storage handle
/// * `query` - Query vector
/// * `k` - Number of results to return
/// * `ef` - Search beam width (ef >= k for good recall)
///
/// # Returns
/// Vector of (distance, vec_id) pairs, sorted by distance ascending.
pub fn search(
    index: &Index,
    storage: &Storage,
    query: &[f32],
    k: usize,
    ef: usize,
) -> Result<Vec<(f32, VecId)>> {
    let nav_info = index
        .nav_cache()
        .get(index.embedding())
        .or_else(|| load_navigation(index, storage).ok())
        .ok_or_else(|| anyhow::anyhow!("No navigation info for embedding {}", index.embedding()))?;

    if nav_info.is_empty() {
        return Ok(Vec::new());
    }

    let entry_point = nav_info.entry_point().unwrap();
    let max_layer = nav_info.max_layer;

    // Start from entry point
    let mut current = entry_point;
    let mut current_dist = distance(index, storage, query, current)?;

    // Greedy descent through layers (from max_layer down to 1)
    // use_cache: true - search benefits from edge caching
    for layer in (1..=max_layer).rev() {
        (current, current_dist) =
            greedy_search_layer(index, storage, query, current, current_dist, layer, true)?;
    }

    // Suppress unused variable warning
    let _ = current_dist;

    // Beam search at layer 0 (use_cache: true for search)
    let results = beam_search_layer0(index, storage, query, current, k, ef, true)?;

    Ok(results)
}

/// Beam search at layer 0 for final candidate expansion.
///
/// # Arguments
/// * `use_cache` - If true, uses edge cache (for search). If false, uncached (for build).
fn beam_search_layer0(
    index: &Index,
    storage: &Storage,
    query: &[f32],
    entry: VecId,
    k: usize,
    ef: usize,
    use_cache: bool,
) -> Result<Vec<(f32, VecId)>> {
    beam_search(index, storage, query, entry, ef, 0, use_cache)
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
pub(super) fn beam_search(
    index: &Index,
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

    let entry_dist = distance(index, storage, query, entry)?;
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
        let neighbors = get_neighbors(index, storage, node, layer, use_cache)?;

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
        let distances: Vec<(VecId, f32)> = if unvisited.len() >= index.config().batch_threshold {
            super::graph::batch_distances(index, storage, query, &unvisited)?
        } else {
            unvisited
                .iter()
                .filter_map(|&n| {
                    distance(index, storage, query, n)
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
/// * `index` - The HNSW index
/// * `storage` - Vector storage (only used for re-ranking)
/// * `query` - Query vector
/// * `encoder` - RaBitQ encoder
/// * `code_cache` - In-memory cache of binary codes
/// * `k` - Number of results to return
/// * `ef` - Search beam width
/// * `rerank_factor` - Multiplier for candidates to re-rank
pub fn search_with_rabitq_cached(
    index: &Index,
    storage: &Storage,
    query: &[f32],
    encoder: &RaBitQ,
    code_cache: &BinaryCodeCache,
    k: usize,
    ef: usize,
    rerank_factor: usize,
) -> Result<Vec<(f32, VecId)>> {
    let nav_info = index
        .nav_cache()
        .get(index.embedding())
        .or_else(|| load_navigation(index, storage).ok())
        .ok_or_else(|| anyhow::anyhow!("No navigation info for embedding {}", index.embedding()))?;

    if nav_info.is_empty() {
        return Ok(Vec::new());
    }

    let entry_point = nav_info.entry_point().unwrap();
    let max_layer = nav_info.max_layer;

    // Encode query to binary code
    let query_code = encoder.encode(query);

    // Phase 1: Greedy descent through upper layers using exact distance
    let mut current = entry_point;
    let mut current_dist = distance(index, storage, query, current)?;

    for layer in (1..=max_layer).rev() {
        (current, current_dist) =
            greedy_search_layer(index, storage, query, current, current_dist, layer, true)?;
    }

    // Suppress unused variable warning
    let _ = current_dist;

    // Phase 2: Beam search at layer 0 using CACHED Hamming distance
    let rerank_count = k * rerank_factor;
    let effective_ef = ef.max(rerank_count);
    let hamming_candidates = beam_search_layer0_hamming_cached(
        index,
        storage,
        &query_code,
        code_cache,
        current,
        effective_ef,
    )?;

    if hamming_candidates.is_empty() {
        return Ok(Vec::new());
    }

    // Phase 3: Re-rank top candidates with exact distance (PARALLEL)
    let candidates_to_rerank = hamming_candidates.len().min(rerank_count);
    let vec_ids: Vec<VecId> = hamming_candidates
        .iter()
        .take(candidates_to_rerank)
        .map(|(_, id)| *id)
        .collect();

    // Parallel reranking - each worker has thread-safe RocksDB access
    let exact_results = crate::vector::parallel::rerank_parallel(&vec_ids, |vec_id| {
        distance(index, storage, query, vec_id).ok()
    }, k);

    Ok(exact_results)
}

/// Beam search at layer 0 using in-memory cached binary codes.
///
/// This version uses `BinaryCodeCache` instead of RocksDB reads, making
/// Hamming distance computation essentially free (just SIMD popcount).
fn beam_search_layer0_hamming_cached(
    index: &Index,
    storage: &Storage,
    query_code: &[u8],
    code_cache: &BinaryCodeCache,
    entry: VecId,
    ef: usize,
) -> Result<Vec<(u32, VecId)>> {
    let mut candidates: BinaryHeap<Reverse<(u32, VecId)>> = BinaryHeap::new();
    let mut results: BinaryHeap<(u32, VecId)> = BinaryHeap::new();
    let mut visited = RoaringBitmap::new();

    // Get entry point's Hamming distance from cache
    let entry_code = code_cache
        .get(index.embedding(), entry)
        .unwrap_or_default();
    let entry_dist = RaBitQ::hamming_distance(query_code, &entry_code);

    candidates.push(Reverse((entry_dist, entry)));
    results.push((entry_dist, entry));
    visited.insert(entry);

    while let Some(Reverse((current_dist, current))) = candidates.pop() {
        // Stop if current is worse than worst result
        if results.len() >= ef {
            if let Some(&(worst_dist, _)) = results.peek() {
                if current_dist > worst_dist {
                    break;
                }
            }
        }

        // Expand neighbors (still need edge lookup from storage)
        let neighbors = get_neighbors(index, storage, current, 0, true)?;

        // Filter unvisited
        let unvisited: Vec<VecId> = neighbors
            .iter()
            .filter(|n| !visited.contains(*n))
            .collect();

        if unvisited.is_empty() {
            continue;
        }

        // Batch fetch binary codes from CACHE (not RocksDB!)
        let codes = code_cache.get_batch(index.embedding(), &unvisited);

        for (neighbor, code_opt) in unvisited.into_iter().zip(codes.into_iter()) {
            visited.insert(neighbor);

            let dist = if let Some(code) = code_opt {
                RaBitQ::hamming_distance(query_code, &code)
            } else {
                u32::MAX // No cached code, skip (shouldn't happen if cache is populated)
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
