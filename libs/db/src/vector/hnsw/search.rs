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
use crate::vector::quantization::RaBitQ;
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
    // Get navigation info from cache or storage
    // For empty index (no GraphMeta), return empty results instead of error
    let nav_info = match index.nav_cache().get(index.embedding()) {
        Some(info) => info,
        None => match load_navigation(index, storage) {
            Ok(info) => info,
            Err(_) => return Ok(Vec::new()), // Empty index → empty results
        },
    };

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
        let distances: Vec<(VecId, f32)> = if unvisited.len() >= index.batch_threshold() {
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
/// Uses ADC (Asymmetric Distance Computation) for beam search at layer 0,
/// replacing symmetric Hamming distance which has fundamental issues with
/// multi-bit quantization (see RABITQ.md Section 5.9).
///
/// # ADC vs Hamming Performance
///
/// | Mode | Recall@10 | Notes |
/// |------|-----------|-------|
/// | Hamming 4-bit | ~24% | Symmetric, loses numeric ordering |
/// | ADC 4-bit | ~92% | Asymmetric, preserves ordering |
///
/// ADC keeps the query as float32 and computes weighted dot products with
/// binary codes, using stored correction factors for accurate estimation.
///
/// # Performance
///
/// With cached codes:
/// - ADC distance: ~10-20ns per vector (decode + dot product)
/// - No RocksDB reads during beam search
/// - Only re-ranking phase reads vectors from disk
///
/// Expected: >90% recall with proper re-ranking.
///
/// # Arguments
///
/// * `index` - The HNSW index
/// * `storage` - Vector storage (only used for re-ranking)
/// * `query` - Query vector
/// * `encoder` - RaBitQ encoder
/// * `code_cache` - In-memory cache of binary codes with ADC corrections
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
    // Get navigation info from cache or storage
    // For empty index (no GraphMeta), return empty results instead of error
    let nav_info = match index.nav_cache().get(index.embedding()) {
        Some(info) => info,
        None => match load_navigation(index, storage) {
            Ok(info) => info,
            Err(_) => return Ok(Vec::new()), // Empty index → empty results
        },
    };

    if nav_info.is_empty() {
        return Ok(Vec::new());
    }

    let entry_point = nav_info.entry_point().unwrap();
    let max_layer = nav_info.max_layer;

    // ADC mode: rotate query (keep as float32), compute query norm
    let query_rotated = encoder.rotate_query(query);
    let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Phase 1: Greedy descent through upper layers using exact distance
    let mut current = entry_point;
    let mut current_dist = distance(index, storage, query, current)?;

    for layer in (1..=max_layer).rev() {
        (current, current_dist) =
            greedy_search_layer(index, storage, query, current, current_dist, layer, true)?;
    }

    // Suppress unused variable warning
    let _ = current_dist;

    // Phase 2: Beam search at layer 0 using CACHED ADC distance
    let rerank_count = k * rerank_factor;
    let effective_ef = ef.max(rerank_count);
    let adc_candidates = beam_search_layer0_adc_cached(
        index,
        storage,
        query,
        encoder,
        &query_rotated,
        query_norm,
        code_cache,
        current,
        effective_ef,
    )?;

    if adc_candidates.is_empty() {
        return Ok(Vec::new());
    }

    // Phase 3: Re-rank top candidates with exact distance (PARALLEL)
    let candidates_to_rerank = adc_candidates.len().min(rerank_count);
    let vec_ids: Vec<VecId> = adc_candidates
        .iter()
        .take(candidates_to_rerank)
        .map(|(_, id)| *id)
        .collect();

    // Parallel reranking - each worker has thread-safe RocksDB access
    let exact_results = crate::vector::search::rerank_parallel(&vec_ids, |vec_id| {
        distance(index, storage, query, vec_id).ok()
    }, k);

    Ok(exact_results)
}

/// Beam search at layer 0 using ADC distance with in-memory cached binary codes.
///
/// Uses ADC (Asymmetric Distance Computation) instead of symmetric Hamming distance.
/// ADC keeps the query as float32 and computes weighted dot products with binary codes,
/// achieving 91-99% recall vs Hamming's 10-24% for multi-bit quantization.
///
/// # Missing Code Handling
///
/// When a binary code is not in the cache (e.g., during incremental indexing or
/// partial cache warmup), the function falls back to exact distance computation.
/// This ensures correctness at the cost of occasional RocksDB reads.
///
/// # Arguments
///
/// * `index` - The HNSW index
/// * `storage` - Storage for edge lookups and exact distance fallback
/// * `query` - Original query vector (for exact distance fallback)
/// * `encoder` - RaBitQ encoder for ADC distance computation
/// * `query_rotated` - Pre-rotated query vector (from encoder.rotate_query())
/// * `query_norm` - L2 norm of original query
/// * `code_cache` - In-memory cache of (binary_code, AdcCorrection) tuples
/// * `entry` - Entry point for beam search
/// * `ef` - Search beam width
fn beam_search_layer0_adc_cached(
    index: &Index,
    storage: &Storage,
    query: &[f32],
    encoder: &RaBitQ,
    query_rotated: &[f32],
    query_norm: f32,
    code_cache: &BinaryCodeCache,
    entry: VecId,
    ef: usize,
) -> Result<Vec<(OrderedFloat<f32>, VecId)>> {
    let mut candidates: BinaryHeap<Reverse<(OrderedFloat<f32>, VecId)>> = BinaryHeap::new();
    let mut results: BinaryHeap<(OrderedFloat<f32>, VecId)> = BinaryHeap::new();
    let mut visited = RoaringBitmap::new();

    // Get entry point's ADC distance from cache (fallback to exact if missing)
    let entry_dist = if let Some(entry_code) = code_cache.get(index.embedding(), entry) {
        encoder.adc_distance(query_rotated, query_norm, &entry_code.code, &entry_code.correction)
    } else {
        // Fallback to exact distance when code is not cached
        // This happens during incremental indexing or partial cache warmup
        distance(index, storage, query, entry)?
    };

    candidates.push(Reverse((OrderedFloat(entry_dist), entry)));
    results.push((OrderedFloat(entry_dist), entry));
    visited.insert(entry);

    while let Some(Reverse((OrderedFloat(current_dist), current))) = candidates.pop() {
        // Stop if current is worse than worst result
        if results.len() >= ef {
            if let Some(&(OrderedFloat(worst_dist), _)) = results.peek() {
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

        // Batch fetch binary codes with corrections from CACHE (not RocksDB!)
        let codes = code_cache.get_batch(index.embedding(), &unvisited);

        for (neighbor, code_opt) in unvisited.into_iter().zip(codes.into_iter()) {
            visited.insert(neighbor);

            // Compute distance: use ADC if cached, fallback to exact if not
            let dist = if let Some(entry) = code_opt {
                encoder.adc_distance(query_rotated, query_norm, &entry.code, &entry.correction)
            } else {
                // Fallback to exact distance when code is not cached
                // This happens during incremental indexing or partial cache warmup
                match distance(index, storage, query, neighbor) {
                    Ok(d) => d,
                    Err(_) => continue, // Skip if we can't compute distance
                }
            };

            // Add to results if better than worst or results not full
            if results.len() < ef {
                candidates.push(Reverse((OrderedFloat(dist), neighbor)));
                results.push((OrderedFloat(dist), neighbor));
            } else if let Some(&(OrderedFloat(worst_dist), _)) = results.peek() {
                if dist < worst_dist {
                    candidates.push(Reverse((OrderedFloat(dist), neighbor)));
                    results.pop();
                    results.push((OrderedFloat(dist), neighbor));
                }
            }
        }
    }

    // Extract results sorted by ADC distance (ascending)
    let mut result_vec: Vec<(OrderedFloat<f32>, VecId)> = results.into_iter().collect();
    result_vec.sort_by_key(|(dist, _)| *dist);

    Ok(result_vec)
}
