//! Integration tests for vector benchmark infrastructure.
//!
//! Tests the complete search pipeline at 1K scale for both datasets:
//! - SIFT: L2 distance, exact HNSW search, NavigationCache
//! - LAION-CLIP: Cosine distance, RaBitQ + BinaryCodeCache + NavigationCache + parallel rerank
//!
//! These tests use synthetic data to avoid requiring external dataset downloads.

use std::sync::Arc;
use tempfile::TempDir;

use motlie_db::vector::benchmark::{
    compute_recall, SiftSubset, LaionSubset, SIFT_EMBEDDING_DIM, LAION_EMBEDDING_DIM,
};
use motlie_db::vector::{
    hnsw, BinaryCodeCache, Distance, HnswIndex, NavigationCache, Storage, VecId,
    VectorStorageType,
};
use motlie_db::vector::quantization::RaBitQ;
use motlie_db::vector::search::rerank_adaptive;
use motlie_db::vector::schema::Vectors;
use motlie_db::rocksdb::ColumnFamily;

/// Helper: Generate synthetic SIFT-like data (128D, L2 distance)
fn generate_sift_subset(num_vectors: usize, num_queries: usize) -> SiftSubset {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Generate base vectors (128D, integer values 0-127 like real SIFT)
    let db_vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| {
            (0..SIFT_EMBEDDING_DIM)
                .map(|_| rng.gen_range(0..128) as f32)
                .collect()
        })
        .collect();

    // Generate query vectors
    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|_| {
            (0..SIFT_EMBEDDING_DIM)
                .map(|_| rng.gen_range(0..128) as f32)
                .collect()
        })
        .collect();

    SiftSubset {
        db_vectors,
        queries,
        dim: SIFT_EMBEDDING_DIM,
    }
}

/// Helper: Generate synthetic LAION-CLIP-like data (512D, Cosine distance, normalized)
fn generate_laion_subset(num_vectors: usize, num_queries: usize) -> LaionSubset {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Generate normalized vectors
    let normalize = |v: &mut Vec<f32>| {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    };

    // Generate base vectors (512D, normalized for Cosine)
    let db_vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| {
            let mut v: Vec<f32> = (0..LAION_EMBEDDING_DIM)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();
            normalize(&mut v);
            v
        })
        .collect();

    // Generate query vectors
    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|_| {
            let mut v: Vec<f32> = (0..LAION_EMBEDDING_DIM)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();
            normalize(&mut v);
            v
        })
        .collect();

    // Ground truth indices (not used directly, we compute recall via brute force)
    let ground_truth: Vec<usize> = (0..num_queries).collect();

    LaionSubset {
        db_vectors,
        queries,
        ground_truth,
        dim: LAION_EMBEDDING_DIM,
    }
}

/// Build HNSW index with NavigationCache
fn build_index_with_navigation(
    storage: &Storage,
    vectors: &[Vec<f32>],
    distance: Distance,
    storage_type: VectorStorageType,
    m: usize,
    ef_construction: usize,
) -> anyhow::Result<(HnswIndex, Arc<NavigationCache>)> {
    let dim = vectors.first().map(|v| v.len()).unwrap_or(128);
    let nav_cache = Arc::new(NavigationCache::new());
    let embedding_code = 1;

    let config = hnsw::Config {
        dim,
        m,
        m_max: m * 2,
        m_max_0: m * 2,
        ef_construction,
        ..Default::default()
    };

    let index = HnswIndex::with_storage_type(
        embedding_code,
        distance,
        storage_type,
        config,
        nav_cache.clone(),
    );

    // Store vectors and build index
    let txn_db = storage.transaction_db()?;
    let vectors_cf = txn_db
        .cf_handle(Vectors::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;

    use motlie_db::vector::schema::VectorCfKey;

    for (i, vector) in vectors.iter().enumerate() {
        let vec_id = i as VecId;

        // Store vector in Vectors CF
        let key = VectorCfKey(embedding_code, vec_id);
        let value_bytes = Vectors::value_to_bytes_typed(vector, storage_type);
        txn_db.put_cf(&vectors_cf, Vectors::key_to_bytes(&key), value_bytes)?;

        // Insert into HNSW index
        index.insert(storage, vec_id, vector)?;
    }

    Ok((index, nav_cache))
}

/// SIFT Integration Test: L2 Distance + Exact HNSW + NavigationCache
///
/// Validates:
/// - SIFT dataset support in benchmark crate
/// - L2 (Euclidean) distance metric
/// - HNSW index build and search
/// - NavigationCache for layer traversal
/// - Recall@10 computation
#[test]
fn test_sift_l2_hnsw_with_navigation_cache() -> anyhow::Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("sift_test_db");

    // Generate synthetic SIFT data (1K vectors, 100 queries)
    let subset = generate_sift_subset(1000, 100);

    println!("=== SIFT L2 + HNSW + NavigationCache Test ===");
    println!("Vectors: {}, Queries: {}, Dim: {}",
             subset.num_vectors(), subset.num_queries(), subset.dim);

    // Compute ground truth using L2 distance
    let ground_truth = subset.compute_ground_truth_topk_with_distance(10, Distance::L2);

    // Initialize storage
    let mut storage = Storage::readwrite(&db_path);
    storage.ready()?;

    // Build HNSW index with L2 distance
    let (index, _nav_cache) = build_index_with_navigation(
        &storage,
        &subset.db_vectors,
        Distance::L2,
        VectorStorageType::F32,
        16,  // M
        100, // ef_construction
    )?;

    println!("Index built successfully");

    // Run searches
    let ef_search = 50;
    let k = 10;
    let mut search_results: Vec<Vec<usize>> = Vec::new();

    for query in &subset.queries {
        let results = index.search(&storage, query, ef_search, k)?;
        let result_ids: Vec<usize> = results.iter().map(|(_, id)| *id as usize).collect();
        search_results.push(result_ids);
    }

    // Compute recall
    let recall = compute_recall(&search_results, &ground_truth, k);

    println!("Recall@{}: {:.2}%", k, recall * 100.0);

    // SIFT with L2 should achieve high recall at 1K scale
    assert!(recall >= 0.80, "Expected recall >= 80%, got {:.2}%", recall * 100.0);

    println!("✓ SIFT L2 + NavigationCache test passed");

    Ok(())
}

/// LAION-CLIP Integration Test: Cosine + Exact HNSW + NavigationCache (NO RaBitQ)
///
/// Validates:
/// - LAION-CLIP dataset works with standard exact HNSW search
/// - Cosine distance metric for both index build and search
/// - NavigationCache for layer traversal
/// - No RaBitQ, no BinaryCodeCache, no reranking - just exact distance
#[test]
fn test_laion_clip_cosine_exact_hnsw() -> anyhow::Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("laion_exact_test_db");

    // Generate synthetic LAION-CLIP data (1K vectors, 100 queries)
    let subset = generate_laion_subset(1000, 100);

    println!("=== LAION-CLIP Cosine + Exact HNSW + NavigationCache Test ===");
    println!("Vectors: {}, Queries: {}, Dim: {}",
             subset.num_vectors(), subset.num_queries(), subset.dim);

    // Compute ground truth using Cosine distance
    let ground_truth = subset.compute_ground_truth_topk_with_distance(10, Distance::Cosine);

    // Initialize storage
    let mut storage = Storage::readwrite(&db_path);
    storage.ready()?;

    // Build HNSW index with Cosine distance
    // Note: 512D requires higher M and ef_construction than 128D for good recall
    let (index, _nav_cache) = build_index_with_navigation(
        &storage,
        &subset.db_vectors,
        Distance::Cosine,
        VectorStorageType::F16, // Use F16 like real LAION-CLIP
        32,  // M - higher for 512D
        200, // ef_construction - higher for 512D
    )?;

    println!("Index built successfully");

    // Run exact HNSW searches (no RaBitQ, no caching, no reranking)
    // Note: 512D requires higher ef_search than 128D SIFT for good recall
    let ef_search = 100;
    let k = 10;
    let mut search_results: Vec<Vec<usize>> = Vec::new();

    for query in &subset.queries {
        // Use standard search() - exact Cosine distance
        let results = index.search(&storage, query, ef_search, k)?;
        let result_ids: Vec<usize> = results.iter().map(|(_, id)| *id as usize).collect();
        search_results.push(result_ids);
    }

    // Compute recall
    let recall = compute_recall(&search_results, &ground_truth, k);

    println!("Recall@{}: {:.2}%", k, recall * 100.0);

    // LAION-CLIP with exact Cosine HNSW should achieve high recall at 1K scale
    // (similar to SIFT with L2)
    assert!(recall >= 0.80, "Expected recall >= 80%, got {:.2}%", recall * 100.0);

    println!("✓ LAION-CLIP Cosine + Exact HNSW test passed");

    Ok(())
}

/// LAION-CLIP Integration Test: Cosine + RaBitQ + BinaryCodeCache + NavigationCache
///
/// Validates:
/// - LAION-CLIP dataset support in benchmark crate
/// - Cosine distance metric
/// - RaBitQ binary quantization
/// - BinaryCodeCache for Hamming distance
/// - NavigationCache for layer traversal
/// - Adaptive parallel reranking
/// - Two-phase search: Hamming pre-filtering + exact rerank
#[test]
fn test_laion_clip_cosine_rabitq_with_caches() -> anyhow::Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("laion_test_db");

    // Generate synthetic LAION-CLIP data (1K vectors, 100 queries)
    let subset = generate_laion_subset(1000, 100);

    println!("=== LAION-CLIP Cosine + RaBitQ + Caches Test ===");
    println!("Vectors: {}, Queries: {}, Dim: {}",
             subset.num_vectors(), subset.num_queries(), subset.dim);

    // Compute ground truth using Cosine distance
    let ground_truth = subset.compute_ground_truth_topk_with_distance(10, Distance::Cosine);

    // Initialize storage
    let mut storage = Storage::readwrite(&db_path);
    storage.ready()?;

    // Build HNSW index with Cosine distance
    let (index, _nav_cache) = build_index_with_navigation(
        &storage,
        &subset.db_vectors,
        Distance::Cosine,
        VectorStorageType::F16, // Use F16 like real LAION-CLIP
        16,  // M
        100, // ef_construction
    )?;

    println!("HNSW index built");

    // Initialize RaBitQ encoder
    let rabitq = RaBitQ::new(LAION_EMBEDDING_DIM, 1, 42);

    // Build BinaryCodeCache
    let binary_cache = BinaryCodeCache::new();
    let embedding_code = 1;

    for (i, vector) in subset.db_vectors.iter().enumerate() {
        let vec_id = i as VecId;
        let code = rabitq.encode(vector);
        binary_cache.put(embedding_code, vec_id, code);
    }

    println!("RaBitQ encoding complete, BinaryCodeCache populated with {} entries",
             subset.db_vectors.len());

    // Run two-phase RaBitQ search
    let ef_search = 50;
    let k = 10;
    let rerank_factor = 4; // Re-rank 40 candidates for top-10
    let mut search_results: Vec<Vec<usize>> = Vec::new();

    for query in &subset.queries {
        let results = index.search_with_rabitq_cached(
            &storage,
            query,
            &rabitq,
            &binary_cache,
            k,
            ef_search,
            rerank_factor,
        )?;
        let result_ids: Vec<usize> = results.iter().map(|(_, id)| *id as usize).collect();
        search_results.push(result_ids);
    }

    // Compute recall
    let recall = compute_recall(&search_results, &ground_truth, k);

    println!("Recall@{}: {:.2}%", k, recall * 100.0);

    // LAION-CLIP with RaBitQ at 1K scale:
    // - 1-bit quantization has significant information loss
    // - At production scale (100K+), recall improves with proper tuning
    // - This test validates the pipeline works end-to-end
    // - Threshold of 40% ensures the search is working (not random ~10%)
    assert!(recall >= 0.40, "Expected recall >= 40%, got {:.2}%", recall * 100.0);

    println!("✓ LAION-CLIP Cosine + RaBitQ + BinaryCodeCache + NavigationCache test passed");

    Ok(())
}

/// Test that SIFT subset ground truth computation works correctly
#[test]
fn test_sift_subset_ground_truth_l2() {
    let subset = SiftSubset {
        db_vectors: vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0],
        ],
        queries: vec![vec![0.0, 0.0, 0.0]],
        dim: 3,
    };

    let gt = subset.compute_ground_truth_topk_with_distance(3, Distance::L2);

    assert_eq!(gt.len(), 1);
    assert_eq!(gt[0].len(), 3);
    // Closest to origin should be [0,0,0], then the axis vectors
    assert_eq!(gt[0][0], 0);
}

/// Test that LAION subset ground truth computation works correctly
#[test]
fn test_laion_subset_ground_truth_cosine() {
    let subset = LaionSubset {
        db_vectors: vec![
            vec![1.0, 0.0, 0.0],  // Normalized
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.707, 0.707, 0.0],  // 45 degrees in xy plane
        ],
        queries: vec![vec![1.0, 0.0, 0.0]],  // Same as first vector
        ground_truth: vec![0],
        dim: 3,
    };

    let gt = subset.compute_ground_truth_topk_with_distance(2, Distance::Cosine);

    assert_eq!(gt.len(), 1);
    assert_eq!(gt[0].len(), 2);
    // [1,0,0] query should match [1,0,0] vector first (cos_dist = 0)
    assert_eq!(gt[0][0], 0);
}

/// Test adaptive parallel reranking threshold behavior
#[test]
fn test_adaptive_parallel_rerank_threshold() {
    use motlie_db::vector::search::DEFAULT_PARALLEL_RERANK_THRESHOLD;

    // Default threshold should be 3200 (from benchmarking)
    assert_eq!(DEFAULT_PARALLEL_RERANK_THRESHOLD, 3200);

    // Test with small candidate set (below threshold - uses sequential)
    let candidates: Vec<VecId> = (0..100).collect();
    let results = rerank_adaptive(
        &candidates,
        |id| Some(id as f32), // Distance = id value
        10,
        DEFAULT_PARALLEL_RERANK_THRESHOLD,
    );

    assert_eq!(results.len(), 10);
    assert_eq!(results[0].1, 0); // Smallest distance

    // Test with large candidate set (above threshold - uses parallel)
    let candidates: Vec<VecId> = (0..5000).collect();
    let results = rerank_adaptive(
        &candidates,
        |id| Some(id as f32),
        10,
        DEFAULT_PARALLEL_RERANK_THRESHOLD,
    );

    assert_eq!(results.len(), 10);
    assert_eq!(results[0].1, 0);
}
