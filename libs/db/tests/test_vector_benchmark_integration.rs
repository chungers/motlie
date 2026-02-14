#![cfg(feature = "benchmark")]
//! Integration tests for vector benchmark infrastructure.
//!
//! Tests the complete search pipeline at 1K scale for both datasets:
//! - SIFT: L2 distance, exact HNSW search
//! - LAION-CLIP: Cosine distance, RaBitQ + BinaryCodeCache + parallel rerank
//!
//! These tests use synthetic data to avoid requiring external dataset downloads.
//!
//! ## Migration Note (Task 5.9)
//!
//! These tests have been migrated from Processor API to use only public
//! Runnable traits (mutation::Runnable and query::Runnable):
//! - `InsertVector::run(&writer)` for inserts
//! - `SearchKNN::run(&search_reader, timeout)` for searches
//! - External IDs used for recall computation (not vec_ids)
//!
//! This enables making Processor `pub(crate)`.

use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;

use motlie_db::vector::benchmark::{
    compute_recall, LaionSubset, SiftSubset, LAION_EMBEDDING_DIM, SIFT_EMBEDDING_DIM,
};
use motlie_db::vector::search::rerank_adaptive;
use motlie_db::vector::{
    create_reader_with_storage, create_writer,
    spawn_mutation_consumer_with_storage_autoreg, spawn_query_consumers_with_storage_autoreg,
    Distance, EmbeddingBuilder, ExternalKey, InsertVector, MutationRunnable, ReaderConfig, Runnable,
    SearchKNN, Storage, VecId, VectorElementType, WriterConfig,
};
use motlie_db::Id;

/// Helper: Generate synthetic SIFT-like data (128D, L2 distance)
fn generate_sift_subset(num_vectors: usize, num_queries: usize) -> SiftSubset {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Generate base vectors (128D, integer values 0-127 like real SIFT)
    let db_vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| {
            (0..SIFT_EMBEDDING_DIM)
                .map(|_| rng.random_range(0..128) as f32)
                .collect()
        })
        .collect();

    // Generate query vectors
    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|_| {
            (0..SIFT_EMBEDDING_DIM)
                .map(|_| rng.random_range(0..128) as f32)
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
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            normalize(&mut v);
            v
        })
        .collect();

    // Generate query vectors
    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|_| {
            let mut v: Vec<f32> = (0..LAION_EMBEDDING_DIM)
                .map(|_| rng.random_range(-1.0..1.0))
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

/// SIFT Integration Test: L2 Distance + Exact HNSW
///
/// Validates:
/// - SIFT dataset support in benchmark crate
/// - L2 (Euclidean) distance metric
/// - Runnable API for insert and search
/// - Recall@10 computation
///
/// Migrated from Processor API to Runnable traits (Task 5.9).
#[tokio::test]
async fn test_sift_l2_hnsw_with_runnable() -> anyhow::Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("sift_test_db");

    // Generate synthetic SIFT data (1K vectors, 100 queries)
    let subset = generate_sift_subset(1000, 100);

    println!("=== SIFT L2 + HNSW + Runnable API Test ===");
    println!(
        "Vectors: {}, Queries: {}, Dim: {}",
        subset.num_vectors(),
        subset.num_queries(),
        subset.dim
    );

    // Compute ground truth using L2 distance
    let ground_truth = subset.compute_ground_truth_topk_with_distance(10, Distance::L2);

    // Initialize storage
    let mut storage = Storage::readwrite(&db_path);
    storage.ready()?;
    let storage = Arc::new(storage);

    // Register embedding
    let registry = storage.cache().clone();
    registry.set_storage(storage.clone())?;
    let builder = EmbeddingBuilder::new("sift-test", SIFT_EMBEDDING_DIM as u32, Distance::L2)
        .with_hnsw_m(16)
        .with_hnsw_ef_construction(100);
    let embedding = registry.register(builder)?;

    // Create writer and spawn mutation consumer
    let (writer, writer_rx) = create_writer(WriterConfig::default());
    let _writer_handle = spawn_mutation_consumer_with_storage_autoreg(
        writer_rx,
        WriterConfig::default(),
        storage.clone(),
    );

    // Create search reader and spawn query consumers
    let (search_reader, reader_rx) =
        create_reader_with_storage(ReaderConfig::default());
    let _reader_handles = spawn_query_consumers_with_storage_autoreg(
        reader_rx,
        ReaderConfig::default(),
        storage.clone(),
        2,
    );

    // Insert vectors using InsertVector::run()
    println!("Inserting {} vectors via Runnable API...", subset.db_vectors.len());
    let mut external_ids: Vec<Id> = Vec::with_capacity(subset.db_vectors.len());
    for (i, vector) in subset.db_vectors.iter().enumerate() {
        let id = Id::new();
        external_ids.push(id);
        InsertVector::new(&embedding, ExternalKey::NodeId(id), vector.clone())
            .immediate()
            .run(&writer)
            .await?;
        if (i + 1) % 200 == 0 {
            println!("  Inserted {}/{}", i + 1, subset.db_vectors.len());
        }
    }
    // Flush to ensure all inserts are committed
    writer.flush().await?;
    println!("  Inserted all {} vectors", subset.db_vectors.len());

    // Run searches using SearchKNN::run()
    let k = 10;
    let timeout = Duration::from_secs(10);
    let mut search_results: Vec<Vec<usize>> = Vec::new();

    for query in &subset.queries {
        let results = SearchKNN::new(&embedding, query.clone(), k)
            .with_ef(50)
            .exact()
            .run(&search_reader, timeout)
            .await?;

        // Map external IDs back to original indices for recall computation
        let result_indices: Vec<usize> = results
            .iter()
            .filter_map(|r| external_ids.iter().position(|&id| id == r.node_id().expect("expected NodeId")))
            .collect();
        search_results.push(result_indices);
    }

    // Compute recall
    let recall = compute_recall(&search_results, &ground_truth, k);

    println!("Recall@{}: {:.2}%", k, recall * 100.0);

    // SIFT with L2 should achieve high recall at 1K scale
    assert!(
        recall >= 0.80,
        "Expected recall >= 80%, got {:.2}%",
        recall * 100.0
    );

    println!("✓ SIFT L2 + Runnable API test passed");

    Ok(())
}

/// LAION-CLIP Integration Test: Cosine + Exact HNSW (NO RaBitQ)
///
/// Validates:
/// - LAION-CLIP dataset works with standard exact HNSW search
/// - Cosine distance metric for both index build and search
/// - No RaBitQ, no reranking - just exact distance via SearchKNN::exact()
///
/// Migrated from Processor API to Runnable traits (Task 5.9).
#[tokio::test]
async fn test_laion_clip_cosine_exact_hnsw_with_runnable() -> anyhow::Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("laion_exact_test_db");

    // Generate synthetic LAION-CLIP data (1K vectors, 100 queries)
    let subset = generate_laion_subset(1000, 100);

    println!("=== LAION-CLIP Cosine + Exact HNSW + Runnable API Test ===");
    println!(
        "Vectors: {}, Queries: {}, Dim: {}",
        subset.num_vectors(),
        subset.num_queries(),
        subset.dim
    );

    // Compute ground truth using Cosine distance
    let ground_truth = subset.compute_ground_truth_topk_with_distance(10, Distance::Cosine);

    // Initialize storage
    let mut storage = Storage::readwrite(&db_path);
    storage.ready()?;
    let storage = Arc::new(storage);

    // Register embedding with higher M and ef_construction for 512D
    let registry = storage.cache().clone();
    registry.set_storage(storage.clone())?;
    let builder =
        EmbeddingBuilder::new("laion-clip-test", LAION_EMBEDDING_DIM as u32, Distance::Cosine)
            .with_hnsw_m(32)
            .with_hnsw_ef_construction(200);
    let embedding = registry.register(builder)?;

    // Create writer and spawn mutation consumer
    let (writer, writer_rx) = create_writer(WriterConfig::default());
    let _writer_handle = spawn_mutation_consumer_with_storage_autoreg(
        writer_rx,
        WriterConfig::default(),
        storage.clone(),
    );

    // Create search reader and spawn query consumers
    let (search_reader, reader_rx) =
        create_reader_with_storage(ReaderConfig::default());
    let _reader_handles = spawn_query_consumers_with_storage_autoreg(
        reader_rx,
        ReaderConfig::default(),
        storage.clone(),
        2,
    );

    // Insert vectors using InsertVector::run()
    println!(
        "Inserting {} vectors via Runnable API...",
        subset.db_vectors.len()
    );
    let mut external_ids: Vec<Id> = Vec::with_capacity(subset.db_vectors.len());
    for (i, vector) in subset.db_vectors.iter().enumerate() {
        let id = Id::new();
        external_ids.push(id);
        InsertVector::new(&embedding, ExternalKey::NodeId(id), vector.clone())
            .immediate()
            .run(&writer)
            .await?;
        if (i + 1) % 200 == 0 {
            println!("  Inserted {}/{}", i + 1, subset.db_vectors.len());
        }
    }
    writer.flush().await?;
    println!("  Inserted all {} vectors", subset.db_vectors.len());

    // Run exact HNSW searches using SearchKNN::exact()
    let k = 10;
    let timeout = Duration::from_secs(10);
    let mut search_results: Vec<Vec<usize>> = Vec::new();

    for query in &subset.queries {
        let results = SearchKNN::new(&embedding, query.clone(), k)
            .with_ef(100)
            .exact()
            .run(&search_reader, timeout)
            .await?;

        // Map external IDs back to original indices
        let result_indices: Vec<usize> = results
            .iter()
            .filter_map(|r| external_ids.iter().position(|&id| id == r.node_id().expect("expected NodeId")))
            .collect();
        search_results.push(result_indices);
    }

    // Compute recall
    let recall = compute_recall(&search_results, &ground_truth, k);

    println!("Recall@{}: {:.2}%", k, recall * 100.0);

    // LAION-CLIP with exact Cosine HNSW should achieve high recall at 1K scale
    assert!(
        recall >= 0.80,
        "Expected recall >= 80%, got {:.2}%",
        recall * 100.0
    );

    println!("✓ LAION-CLIP Cosine + Exact HNSW + Runnable API test passed");

    Ok(())
}

/// LAION-CLIP Integration Test: Cosine + RaBitQ + BinaryCodeCache
///
/// Validates:
/// - LAION-CLIP dataset support in benchmark crate
/// - Cosine distance metric with RaBitQ acceleration
/// - Automatic BinaryCodeCache population via InsertVector
/// - SearchKNN with auto-strategy (RaBitQ for Cosine)
/// - Two-phase search: ADC navigation + exact rerank
///
/// Migrated from Processor API to Runnable traits (Task 5.9).
#[tokio::test]
async fn test_laion_clip_cosine_rabitq_with_runnable() -> anyhow::Result<()> {
    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("laion_test_db");

    // Generate synthetic LAION-CLIP data (1K vectors, 100 queries)
    let subset = generate_laion_subset(1000, 100);

    println!("=== LAION-CLIP Cosine + RaBitQ + Runnable API Test ===");
    println!(
        "Vectors: {}, Queries: {}, Dim: {}",
        subset.num_vectors(),
        subset.num_queries(),
        subset.dim
    );

    // Compute ground truth using Cosine distance
    let ground_truth = subset.compute_ground_truth_topk_with_distance(10, Distance::Cosine);

    // Initialize storage
    let mut storage = Storage::readwrite(&db_path);
    storage.ready()?;
    let storage = Arc::new(storage);

    // Register embedding
    let registry = storage.cache().clone();
    registry.set_storage(storage.clone())?;
    let builder = EmbeddingBuilder::new(
        "laion-clip-rabitq-test",
        LAION_EMBEDDING_DIM as u32,
        Distance::Cosine,
    )
    .with_hnsw_m(16)
    .with_hnsw_ef_construction(100);
    let embedding = registry.register(builder)?;

    // Create writer and spawn mutation consumer
    let (writer, writer_rx) = create_writer(WriterConfig::default());
    let _writer_handle = spawn_mutation_consumer_with_storage_autoreg(
        writer_rx,
        WriterConfig::default(),
        storage.clone(),
    );

    // Create search reader and spawn query consumers
    let (search_reader, reader_rx) =
        create_reader_with_storage(ReaderConfig::default());
    let _reader_handles = spawn_query_consumers_with_storage_autoreg(
        reader_rx,
        ReaderConfig::default(),
        storage.clone(),
        2,
    );

    // Insert vectors (BinaryCodeCache populated automatically)
    println!(
        "Inserting {} vectors via Runnable API...",
        subset.db_vectors.len()
    );
    let mut external_ids: Vec<Id> = Vec::with_capacity(subset.db_vectors.len());
    for (i, vector) in subset.db_vectors.iter().enumerate() {
        let id = Id::new();
        external_ids.push(id);
        InsertVector::new(&embedding, ExternalKey::NodeId(id), vector.clone())
            .immediate()
            .run(&writer)
            .await?;
        if (i + 1) % 200 == 0 {
            println!("  Inserted {}/{}", i + 1, subset.db_vectors.len());
        }
    }
    writer.flush().await?;
    println!("  Inserted all {} vectors + BinaryCodeCache populated", subset.db_vectors.len());

    // Run two-phase RaBitQ search using SearchKNN (auto-selects RaBitQ for Cosine)
    let k = 10;
    let rerank_factor = 4;
    let timeout = Duration::from_secs(10);
    let mut search_results: Vec<Vec<usize>> = Vec::new();

    for query in &subset.queries {
        // SearchKNN without .exact() auto-selects RaBitQ for Cosine
        let results = SearchKNN::new(&embedding, query.clone(), k)
            .with_ef(50)
            .with_rerank_factor(rerank_factor)
            .run(&search_reader, timeout)
            .await?;

        // Map external IDs back to original indices
        let result_indices: Vec<usize> = results
            .iter()
            .filter_map(|r| external_ids.iter().position(|&id| id == r.node_id().expect("expected NodeId")))
            .collect();
        search_results.push(result_indices);
    }

    // Compute recall
    let recall = compute_recall(&search_results, &ground_truth, k);

    println!("Recall@{}: {:.2}%", k, recall * 100.0);

    // LAION-CLIP with RaBitQ at 1K scale:
    // - 1-bit quantization has significant information loss
    // - At production scale (100K+), recall improves with proper tuning
    // - This test validates the pipeline works end-to-end
    // - Threshold of 40% ensures the search is working (not random ~10%)
    assert!(
        recall >= 0.40,
        "Expected recall >= 40%, got {:.2}%",
        recall * 100.0
    );

    println!("✓ LAION-CLIP Cosine + RaBitQ + Runnable API test passed");

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
            vec![1.0, 0.0, 0.0], // Normalized
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.707, 0.707, 0.0], // 45 degrees in xy plane
        ],
        queries: vec![vec![1.0, 0.0, 0.0]], // Same as first vector
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
    let results = rerank_adaptive(&candidates, |id| Some(id as f32), 10, DEFAULT_PARALLEL_RERANK_THRESHOLD);

    assert_eq!(results.len(), 10);
    assert_eq!(results[0].1, 0);
}

// ============================================================================
// VectorElementType end-to-end tests
// ============================================================================

/// Helper: run the full insert→search pipeline with a given storage type.
/// Returns (recall, embedding_storage_type_observed).
async fn run_storage_type_pipeline(
    storage_type: VectorElementType,
    distance: Distance,
    dim: usize,
    num_vectors: usize,
    num_queries: usize,
) -> anyhow::Result<(f64, VectorElementType)> {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(99);

    let normalize = |v: &mut Vec<f32>| {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    };

    // Generate synthetic data
    let db_vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            if distance == Distance::Cosine {
                normalize(&mut v);
            }
            v
        })
        .collect();
    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            if distance == Distance::Cosine {
                normalize(&mut v);
            }
            v
        })
        .collect();

    // Brute-force ground truth
    let k = 10;
    let ground_truth: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| {
            let mut dists: Vec<(usize, f32)> = db_vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, distance.compute(q, v)))
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.iter().take(k).map(|(i, _)| *i).collect()
        })
        .collect();

    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("storage_type_test");

    // Initialize storage
    let mut storage = Storage::readwrite(&db_path);
    storage.ready()?;
    let storage = Arc::new(storage);

    // Register embedding with explicit storage type
    let registry = storage.cache().clone();
    registry.set_storage(storage.clone())?;
    let embedding = registry.register(
        EmbeddingBuilder::new("storage-type-test", dim as u32, distance)
            .with_hnsw_m(16)
            .with_hnsw_ef_construction(100)
            .with_storage_type(storage_type),
    )?;

    // Verify storage type propagated to embedding
    let observed_type = embedding.storage_type();
    assert_eq!(
        observed_type, storage_type,
        "Embedding storage_type mismatch: expected {:?}, got {:?}",
        storage_type, observed_type
    );

    // Create writer/reader
    let (writer, writer_rx) = create_writer(WriterConfig::default());
    let _writer_handle = spawn_mutation_consumer_with_storage_autoreg(
        writer_rx,
        WriterConfig::default(),
        storage.clone(),
    );
    let (search_reader, reader_rx) =
        create_reader_with_storage(ReaderConfig::default());
    let _reader_handles = spawn_query_consumers_with_storage_autoreg(
        reader_rx,
        ReaderConfig::default(),
        storage.clone(),
        2,
    );

    // Insert vectors
    let mut external_ids: Vec<Id> = Vec::with_capacity(num_vectors);
    for vector in &db_vectors {
        let id = Id::new();
        external_ids.push(id);
        InsertVector::new(&embedding, ExternalKey::NodeId(id), vector.clone())
            .immediate()
            .run(&writer)
            .await?;
    }
    writer.flush().await?;

    // Search
    let timeout = Duration::from_secs(10);
    let mut search_results: Vec<Vec<usize>> = Vec::new();
    for query in &queries {
        let results = SearchKNN::new(&embedding, query.clone(), k)
            .with_ef(50)
            .exact()
            .run(&search_reader, timeout)
            .await?;
        let result_indices: Vec<usize> = results
            .iter()
            .filter_map(|r| {
                external_ids
                    .iter()
                    .position(|&id| id == r.node_id().expect("expected NodeId"))
            })
            .collect();
        search_results.push(result_indices);
    }

    let recall = compute_recall(&search_results, &ground_truth, k);
    Ok((recall, observed_type))
}

/// End-to-end test: F16 storage type flows from EmbeddingBuilder through
/// insert and search with correct recall.
#[tokio::test]
async fn test_storage_type_f16_end_to_end() -> anyhow::Result<()> {
    let (recall, observed_type) = run_storage_type_pipeline(
        VectorElementType::F16,
        Distance::Cosine,
        128,
        500,
        50,
    )
    .await?;

    assert_eq!(observed_type, VectorElementType::F16);
    assert!(
        recall >= 0.80,
        "F16 recall@10 = {:.1}%, expected >= 80%",
        recall * 100.0
    );
    println!("F16 end-to-end: recall@10 = {:.1}%", recall * 100.0);
    Ok(())
}

/// End-to-end test: F32 storage type flows from EmbeddingBuilder through
/// insert and search with correct recall.
#[tokio::test]
async fn test_storage_type_f32_end_to_end() -> anyhow::Result<()> {
    let (recall, observed_type) = run_storage_type_pipeline(
        VectorElementType::F32,
        Distance::Cosine,
        128,
        500,
        50,
    )
    .await?;

    assert_eq!(observed_type, VectorElementType::F32);
    assert!(
        recall >= 0.80,
        "F32 recall@10 = {:.1}%, expected >= 80%",
        recall * 100.0
    );
    println!("F32 end-to-end: recall@10 = {:.1}%", recall * 100.0);
    Ok(())
}

/// End-to-end test: F16 storage type metadata round-trips through RocksDB.
///
/// Verifies that `storage_type` survives serialization/deserialization in the
/// EmbeddingSpecs column family by creating a fresh EmbeddingRegistry (empty
/// in-memory cache) and loading the spec back from RocksDB via `get_by_code`.
#[tokio::test]
async fn test_storage_type_metadata_persists() -> anyhow::Result<()> {
    use motlie_db::vector::EmbeddingRegistry;
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(77);
    let dim = 64;
    let num_vectors = 200;

    let normalize = |v: &mut Vec<f32>| {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    };

    let db_vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            normalize(&mut v);
            v
        })
        .collect();
    let query: Vec<f32> = {
        let mut v: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
        normalize(&mut v);
        v
    };

    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("persist_test");

    let mut storage = Storage::readwrite(&db_path);
    storage.ready()?;
    let storage = Arc::new(storage);

    // Register F16 embedding and insert vectors
    let registry = storage.cache().clone();
    registry.set_storage(storage.clone())?;
    let embedding = registry.register(
        EmbeddingBuilder::new("persist-test", dim as u32, Distance::Cosine)
            .with_hnsw_m(16)
            .with_hnsw_ef_construction(100)
            .with_storage_type(VectorElementType::F16),
    )?;
    assert_eq!(embedding.storage_type(), VectorElementType::F16);

    let (writer, writer_rx) = create_writer(WriterConfig::default());
    let _writer_handle = spawn_mutation_consumer_with_storage_autoreg(
        writer_rx,
        WriterConfig::default(),
        storage.clone(),
    );

    for vector in &db_vectors {
        let id = Id::new();
        InsertVector::new(&embedding, ExternalKey::NodeId(id), vector.clone())
            .immediate()
            .run(&writer)
            .await?;
    }
    writer.flush().await?;

    // Verify RocksDB round-trip: create a fresh registry with empty cache
    // and load the spec from storage. This simulates cold-start recovery.
    let fresh_registry = EmbeddingRegistry::new_without_storage();
    fresh_registry.set_storage(storage.clone())?;

    let recovered = fresh_registry
        .get_by_code(1)
        .expect("Embedding code 1 should exist in RocksDB");
    assert_eq!(
        recovered.storage_type(),
        VectorElementType::F16,
        "Storage type should round-trip as F16 through RocksDB"
    );
    assert_eq!(recovered.dim(), dim as u32);
    assert_eq!(recovered.distance(), Distance::Cosine);

    // Verify search still works with the recovered embedding
    let (search_reader, reader_rx) = create_reader_with_storage(ReaderConfig::default());
    let _reader_handles = spawn_query_consumers_with_storage_autoreg(
        reader_rx,
        ReaderConfig::default(),
        storage.clone(),
        2,
    );
    let timeout = Duration::from_secs(10);
    let results = SearchKNN::new(&recovered, query.clone(), 5)
        .with_ef(50)
        .exact()
        .run(&search_reader, timeout)
        .await?;
    assert!(
        !results.is_empty(),
        "Search should return results with recovered F16 embedding"
    );
    println!(
        "Persistence test: recovered F16 embedding from RocksDB, search returned {} results",
        results.len()
    );

    Ok(())
}
