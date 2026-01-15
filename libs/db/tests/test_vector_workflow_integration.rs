//! Integration test: Complete vector workflow with LAION-CLIP style data.
//!
//! This test demonstrates the full lifecycle:
//! 1. Register embedding with Cosine distance (required for RaBitQ)
//! 2. Insert 1K vectors with RaBitQ index building
//! 3. Search using both exact and quantized (RaBitQ) strategies
//! 4. Delete vectors and verify they don't appear in results
//!
//! Uses synthetic 512D normalized vectors to simulate LAION-CLIP embeddings.

use std::collections::HashSet;
use std::sync::Arc;

use motlie_db::vector::benchmark::LAION_EMBEDDING_DIM;
use motlie_db::vector::{
    Distance, EmbeddingBuilder, Processor, RaBitQConfig, SearchConfig, Storage,
    hnsw,
};
use motlie_db::Id;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use tempfile::TempDir;

const NUM_VECTORS: usize = 1000;
const NUM_QUERIES: usize = 10;
const K: usize = 10;

/// Generate synthetic LAION-CLIP style vectors (512D, normalized for Cosine distance).
fn generate_laion_vectors(num_vectors: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    (0..num_vectors)
        .map(|_| {
            let mut v: Vec<f32> = (0..LAION_EMBEDDING_DIM)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();
            // Normalize for Cosine distance
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in v.iter_mut() {
                    *x /= norm;
                }
            }
            v
        })
        .collect()
}

/// Compute brute-force ground truth top-k for recall calculation.
fn compute_ground_truth(
    queries: &[Vec<f32>],
    db_vectors: &[Vec<f32>],
    k: usize,
) -> Vec<Vec<usize>> {
    queries
        .iter()
        .map(|query| {
            let mut distances: Vec<(usize, f32)> = db_vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, Distance::Cosine.compute(query, v)))
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            distances.iter().take(k).map(|(i, _)| *i).collect()
        })
        .collect()
}

/// Calculate Recall@k: fraction of ground truth results found in actual results.
fn compute_recall(ground_truth: &[usize], results: &[usize]) -> f32 {
    let gt_set: HashSet<_> = ground_truth.iter().collect();
    let found = results.iter().filter(|r| gt_set.contains(r)).count();
    found as f32 / ground_truth.len() as f32
}

#[test]
fn test_vector_workflow_with_laion_clip_style_data() {
    // =========================================================================
    // Setup: Create storage with HNSW and RaBitQ enabled
    // =========================================================================
    let temp_dir = TempDir::new().expect("temp dir");
    let mut storage = Storage::readwrite(temp_dir.path());
    storage.ready().expect("storage ready");
    let storage = Arc::new(storage);

    // Get registry from storage cache
    let registry = storage.cache().clone();
    let txn_db = storage.transaction_db().expect("txn_db");

    // Register embedding with Cosine distance (required for RaBitQ)
    let builder = EmbeddingBuilder::new("clip-vit-b32", LAION_EMBEDDING_DIM as u32, Distance::Cosine);
    let embedding = registry.register(builder, &txn_db).expect("register embedding");

    println!(
        "Registered embedding: code={}, model={}, dim={}, distance={:?}",
        embedding.code(),
        embedding.model(),
        embedding.dim(),
        embedding.distance()
    );

    // Create processor with HNSW enabled and RaBitQ configured
    let mut hnsw_config = hnsw::Config::default();
    hnsw_config.enabled = true;
    hnsw_config.dim = LAION_EMBEDDING_DIM;
    hnsw_config.m = 16;
    hnsw_config.ef_construction = 100;

    let rabitq_config = RaBitQConfig::default(); // 4-bit quantization

    let processor = Processor::with_config(storage.clone(), registry.clone(), rabitq_config, hnsw_config);

    // =========================================================================
    // Step 1: Generate synthetic LAION-CLIP vectors
    // =========================================================================
    println!("\n=== Step 1: Generating {} synthetic 512D vectors ===", NUM_VECTORS);
    let db_vectors = generate_laion_vectors(NUM_VECTORS, 42);
    let query_vectors = generate_laion_vectors(NUM_QUERIES, 123);

    // =========================================================================
    // Step 2: Insert vectors with index building
    // =========================================================================
    println!("\n=== Step 2: Inserting {} vectors with RaBitQ index building ===", NUM_VECTORS);
    let mut external_ids: Vec<Id> = Vec::with_capacity(NUM_VECTORS);

    for (i, vector) in db_vectors.iter().enumerate() {
        let id = Id::new();
        external_ids.push(id);

        processor
            .insert_vector(&embedding, id, vector, true) // build_index = true
            .expect("insert vector");

        if (i + 1) % 200 == 0 {
            println!("  Inserted {}/{} vectors", i + 1, NUM_VECTORS);
        }
    }
    println!("  Inserted all {} vectors", NUM_VECTORS);

    // =========================================================================
    // Step 3: Compute brute-force ground truth for recall measurement
    // =========================================================================
    println!("\n=== Step 3: Computing brute-force ground truth ===");
    let ground_truth = compute_ground_truth(&query_vectors, &db_vectors, K);

    // =========================================================================
    // Step 4: Search using EXACT strategy
    // =========================================================================
    println!("\n=== Step 4: Searching with EXACT strategy ===");
    let exact_config = SearchConfig::new(embedding.clone(), K)
        .exact() // Force exact distance computation
        .with_ef(100);

    assert!(exact_config.strategy().is_exact(), "Should be exact strategy");
    println!("  Strategy: {}", exact_config.strategy());

    let mut exact_recalls = Vec::new();
    for (qi, query) in query_vectors.iter().enumerate() {
        let results = processor
            .search_with_config(&exact_config, query)
            .expect("exact search");

        // Convert results to vec_id indices (external_ids index matches vec_id for this test)
        let result_indices: Vec<usize> = results
            .iter()
            .filter_map(|r| external_ids.iter().position(|&id| id == r.id))
            .collect();

        let recall = compute_recall(&ground_truth[qi], &result_indices);
        exact_recalls.push(recall);

        if qi == 0 {
            println!("  Query 0 top-{} results:", K);
            for (i, r) in results.iter().take(5).enumerate() {
                println!("    {}. id={}, distance={:.4}", i + 1, r.id, r.distance);
            }
        }
    }

    let avg_exact_recall = exact_recalls.iter().sum::<f32>() / exact_recalls.len() as f32;
    println!("  Average Recall@{} (Exact): {:.2}%", K, avg_exact_recall * 100.0);

    // =========================================================================
    // Step 5: Search using RaBitQ (quantized) strategy
    // =========================================================================
    println!("\n=== Step 5: Searching with RaBitQ (quantized) strategy ===");

    // For Cosine distance, auto-selects RaBitQ
    let rabitq_config = SearchConfig::new(embedding.clone(), K)
        .with_ef(100)
        .with_rerank_factor(10); // Rerank top 100 candidates (k * 10)

    assert!(rabitq_config.strategy().is_rabitq(), "Should be RaBitQ strategy");
    println!("  Strategy: {}", rabitq_config.strategy());
    println!("  Rerank factor: {}", rabitq_config.rerank_factor());

    let mut rabitq_recalls = Vec::new();
    for (qi, query) in query_vectors.iter().enumerate() {
        let results = processor
            .search_with_config(&rabitq_config, query)
            .expect("rabitq search");

        let result_indices: Vec<usize> = results
            .iter()
            .filter_map(|r| external_ids.iter().position(|&id| id == r.id))
            .collect();

        let recall = compute_recall(&ground_truth[qi], &result_indices);
        rabitq_recalls.push(recall);

        if qi == 0 {
            println!("  Query 0 top-{} results:", K);
            for (i, r) in results.iter().take(5).enumerate() {
                println!("    {}. id={}, distance={:.4}", i + 1, r.id, r.distance);
            }
        }
    }

    let avg_rabitq_recall = rabitq_recalls.iter().sum::<f32>() / rabitq_recalls.len() as f32;
    println!("  Average Recall@{} (RaBitQ): {:.2}%", K, avg_rabitq_recall * 100.0);

    // Both strategies should have reasonable recall
    assert!(
        avg_exact_recall >= 0.5,
        "Exact recall should be at least 50%, got {:.2}%",
        avg_exact_recall * 100.0
    );
    assert!(
        avg_rabitq_recall >= 0.5,
        "RaBitQ recall should be at least 50%, got {:.2}%",
        avg_rabitq_recall * 100.0
    );

    // =========================================================================
    // Step 6: Delete some vectors and verify they don't appear in results
    // =========================================================================
    println!("\n=== Step 6: Deleting vectors and verifying ===");

    // Delete first 100 vectors
    let delete_count = 100;
    let deleted_ids: Vec<Id> = external_ids[..delete_count].to_vec();

    for (i, &id) in deleted_ids.iter().enumerate() {
        let result = processor
            .delete_vector(&embedding, id)
            .expect("delete vector");
        assert!(result.is_some(), "Delete should return vec_id");

        if (i + 1) % 50 == 0 {
            println!("  Deleted {}/{} vectors", i + 1, delete_count);
        }
    }
    println!("  Deleted {} vectors", delete_count);

    // Search again and verify deleted vectors don't appear
    println!("\n=== Step 7: Verifying deleted vectors don't appear in results ===");
    let deleted_id_set: HashSet<Id> = deleted_ids.iter().copied().collect();

    for (qi, query) in query_vectors.iter().enumerate() {
        // Test with exact strategy
        let results = processor
            .search_with_config(&exact_config, query)
            .expect("search after delete");

        for result in &results {
            assert!(
                !deleted_id_set.contains(&result.id),
                "Query {}: Deleted ID {} should not appear in exact search results",
                qi,
                result.id
            );
        }

        // Test with RaBitQ strategy
        let results = processor
            .search_with_config(&rabitq_config, query)
            .expect("rabitq search after delete");

        for result in &results {
            assert!(
                !deleted_id_set.contains(&result.id),
                "Query {}: Deleted ID {} should not appear in RaBitQ search results",
                qi,
                result.id
            );
        }
    }
    println!("  Verified: No deleted vectors appear in any search results");

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n=== Summary ===");
    println!("  Vectors inserted: {}", NUM_VECTORS);
    println!("  Vectors deleted: {}", delete_count);
    println!("  Queries executed: {} per strategy", NUM_QUERIES);
    println!("  Exact Recall@{}: {:.2}%", K, avg_exact_recall * 100.0);
    println!("  RaBitQ Recall@{}: {:.2}%", K, avg_rabitq_recall * 100.0);
    println!("  Deletion verification: PASSED");
    println!("\nTest completed successfully!");
}

/// Test batch insert workflow.
#[test]
fn test_vector_batch_insert_workflow() {
    let temp_dir = TempDir::new().expect("temp dir");
    let mut storage = Storage::readwrite(temp_dir.path());
    storage.ready().expect("storage ready");
    let storage = Arc::new(storage);

    let registry = storage.cache().clone();
    let txn_db = storage.transaction_db().expect("txn_db");

    // Register embedding
    let builder = EmbeddingBuilder::new("clip-vit-b32", LAION_EMBEDDING_DIM as u32, Distance::Cosine);
    let embedding = registry.register(builder, &txn_db).expect("register");

    // Create processor with HNSW
    let mut hnsw_config = hnsw::Config::default();
    hnsw_config.enabled = true;
    hnsw_config.dim = LAION_EMBEDDING_DIM;

    let processor = Processor::with_config(
        storage.clone(),
        registry,
        RaBitQConfig::default(),
        hnsw_config,
    );

    // Generate vectors
    let vectors = generate_laion_vectors(500, 42);
    let batch: Vec<(Id, Vec<f32>)> = vectors.into_iter().map(|v| (Id::new(), v)).collect();

    // Batch insert with index building
    println!("Batch inserting 500 vectors...");
    let vec_ids = processor
        .insert_batch(&embedding, &batch, true)
        .expect("batch insert");

    assert_eq!(vec_ids.len(), 500, "Should insert 500 vectors");
    println!("Batch insert successful: {} vectors", vec_ids.len());

    // Verify search works
    let query = generate_laion_vectors(1, 999)[0].clone();
    let config = SearchConfig::new(embedding.clone(), 10).with_ef(50);

    let results = processor
        .search_with_config(&config, &query)
        .expect("search");

    assert!(!results.is_empty(), "Should have search results");
    println!("Search returned {} results", results.len());
}

/// Test SearchReader API with both exact and RaBitQ strategies.
///
/// This test verifies the fix for the "SearchReader strategy gap" where
/// the original `search_knn()` always used exact search. The new
/// `search_with_config()` method allows strategy selection.
#[test]
fn test_search_reader_strategy_selection() {
    use motlie_db::vector::{create_search_reader, ReaderConfig};

    let temp_dir = TempDir::new().expect("temp dir");
    let mut storage = Storage::readwrite(temp_dir.path());
    storage.ready().expect("storage ready");
    let storage = Arc::new(storage);

    let registry = storage.cache().clone();
    let txn_db = storage.transaction_db().expect("txn_db");

    // Register Cosine embedding (required for RaBitQ)
    let builder = EmbeddingBuilder::new("clip-vit-b32", LAION_EMBEDDING_DIM as u32, Distance::Cosine);
    let embedding = registry.register(builder, &txn_db).expect("register");

    // Create processor with HNSW and RaBitQ
    let mut hnsw_config = hnsw::Config::default();
    hnsw_config.enabled = true;
    hnsw_config.dim = LAION_EMBEDDING_DIM;

    let processor = Arc::new(Processor::with_config(
        storage.clone(),
        registry,
        RaBitQConfig::default(),
        hnsw_config,
    ));

    // Create SearchReader
    let (search_reader, _receiver) = create_search_reader(ReaderConfig::default(), processor.clone());

    // Insert test vectors
    println!("Inserting 200 test vectors...");
    let vectors = generate_laion_vectors(200, 42);
    for (i, vector) in vectors.iter().enumerate() {
        processor
            .insert_vector(&embedding, Id::new(), vector, true)
            .expect("insert");
        if (i + 1) % 100 == 0 {
            println!("  Inserted {}/200", i + 1);
        }
    }

    let query = generate_laion_vectors(1, 999)[0].clone();

    // =========================================================================
    // Test 1: SearchReader with EXACT strategy
    // =========================================================================
    println!("\n=== Testing SearchReader::search_with_config (Exact) ===");
    let exact_config = SearchConfig::new(embedding.clone(), 10)
        .exact()
        .with_ef(50);

    assert!(exact_config.strategy().is_exact(), "Config should be exact");
    println!("  Strategy: {}", exact_config.strategy());

    let exact_results = search_reader
        .search_with_config(&exact_config, &query)
        .expect("exact search via SearchReader");

    assert!(!exact_results.is_empty(), "Exact search should return results");
    println!("  Exact search returned {} results", exact_results.len());

    // =========================================================================
    // Test 2: SearchReader with RaBitQ strategy
    // =========================================================================
    println!("\n=== Testing SearchReader::search_with_config (RaBitQ) ===");
    let rabitq_config = SearchConfig::new(embedding.clone(), 10)
        .with_ef(50)
        .with_rerank_factor(10);

    assert!(rabitq_config.strategy().is_rabitq(), "Config should be RaBitQ for Cosine");
    println!("  Strategy: {}", rabitq_config.strategy());

    let rabitq_results = search_reader
        .search_with_config(&rabitq_config, &query)
        .expect("RaBitQ search via SearchReader");

    assert!(!rabitq_results.is_empty(), "RaBitQ search should return results");
    println!("  RaBitQ search returned {} results", rabitq_results.len());

    // =========================================================================
    // Verify both strategies produce similar top results
    // =========================================================================
    println!("\n=== Comparing results ===");
    println!("  Exact top-3:");
    for (i, r) in exact_results.iter().take(3).enumerate() {
        println!("    {}. distance={:.4}", i + 1, r.distance);
    }
    println!("  RaBitQ top-3:");
    for (i, r) in rabitq_results.iter().take(3).enumerate() {
        println!("    {}. distance={:.4}", i + 1, r.distance);
    }

    // Top result should be similar (within tolerance for quantization)
    let exact_top = exact_results[0].distance;
    let rabitq_top = rabitq_results[0].distance;
    let diff = (exact_top - rabitq_top).abs();
    println!("  Top-1 distance difference: {:.4}", diff);
    assert!(diff < 0.1, "Top results should be similar (diff={:.4})", diff);

    println!("\nSearchReader strategy selection test PASSED!");
}
