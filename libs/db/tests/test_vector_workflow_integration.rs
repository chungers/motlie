#![cfg(feature = "benchmark")]
//! Integration test: Complete vector workflow with LAION-CLIP style data.
//!
//! This test demonstrates the full lifecycle:
//! 1. Register embedding with Cosine distance (required for RaBitQ)
//! 2. Insert 1K vectors with RaBitQ index building
//! 3. Search using both exact and quantized (RaBitQ) strategies
//! 4. Delete vectors and verify they don't appear in results
//!
//! Uses synthetic 512D normalized vectors to simulate LAION-CLIP embeddings.
//!
//! ## Migration Note (Task 5.9)
//!
//! This test has been migrated from Processor API to use only public
//! Runnable traits:
//! - `InsertVector::run(&writer)` for inserts
//! - `SearchKNN::run(&search_reader, timeout)` for searches
//! - `DeleteVector::run(&writer)` for soft-deletes (tombstone when HNSW enabled)
//!
//! This enables making Processor `pub(crate)`.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use motlie_db::vector::benchmark::LAION_EMBEDDING_DIM;
use motlie_db::vector::{
    create_search_reader_with_storage, create_writer,
    spawn_mutation_consumer_with_storage_autoreg, spawn_query_consumers_with_storage_autoreg,
    DeleteVector, Distance, EmbeddingBuilder, ExternalKey, InsertVector, MutationRunnable,
    ReaderConfig, Runnable, SearchKNN, Storage, WriterConfig,
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

#[tokio::test]
async fn test_vector_workflow_with_laion_clip_style_data() {
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
    let builder =
        EmbeddingBuilder::new("clip-vit-b32", LAION_EMBEDDING_DIM as u32, Distance::Cosine);
    let embedding = registry
        .register(builder, &txn_db)
        .expect("register embedding");

    println!(
        "Registered embedding: code={}, model={}, dim={}, distance={:?}",
        embedding.code(),
        embedding.model(),
        embedding.dim(),
        embedding.distance()
    );

    // Create writer and spawn mutation consumer
    let (writer, writer_rx) = create_writer(WriterConfig::default());
    let _writer_handle = spawn_mutation_consumer_with_storage_autoreg(
        writer_rx,
        WriterConfig::default(),
        storage.clone(),
    );

    // Create search reader and spawn query consumers
    let (search_reader, reader_rx) =
        create_search_reader_with_storage(ReaderConfig::default(), storage.clone());
    let _reader_handles = spawn_query_consumers_with_storage_autoreg(
        reader_rx,
        ReaderConfig::default(),
        storage.clone(),
        2,
    );

    let timeout = Duration::from_secs(10);

    // =========================================================================
    // Step 1: Generate synthetic LAION-CLIP vectors
    // =========================================================================
    println!(
        "\n=== Step 1: Generating {} synthetic 512D vectors ===",
        NUM_VECTORS
    );
    let db_vectors = generate_laion_vectors(NUM_VECTORS, 42);
    let query_vectors = generate_laion_vectors(NUM_QUERIES, 123);

    // =========================================================================
    // Step 2: Insert vectors with index building via InsertVector::run()
    // =========================================================================
    println!(
        "\n=== Step 2: Inserting {} vectors with RaBitQ index building ===",
        NUM_VECTORS
    );
    let mut external_ids: Vec<Id> = Vec::with_capacity(NUM_VECTORS);

    for (i, vector) in db_vectors.iter().enumerate() {
        let id = Id::new();
        external_ids.push(id);

        InsertVector::new(&embedding, ExternalKey::NodeId(id), vector.clone())
            .immediate()
            .run(&writer)
            .await
            .expect("insert vector");

        if (i + 1) % 200 == 0 {
            println!("  Inserted {}/{} vectors", i + 1, NUM_VECTORS);
        }
    }
    // Flush to ensure all inserts are committed
    writer.flush().await.expect("flush writer");
    println!("  Inserted all {} vectors", NUM_VECTORS);

    // =========================================================================
    // Step 3: Compute brute-force ground truth for recall measurement
    // =========================================================================
    println!("\n=== Step 3: Computing brute-force ground truth ===");
    let ground_truth = compute_ground_truth(&query_vectors, &db_vectors, K);

    // =========================================================================
    // Step 4: Search using EXACT strategy via SearchKNN::exact()
    // =========================================================================
    println!("\n=== Step 4: Searching with EXACT strategy ===");
    println!("  Strategy: Exact (forced via SearchKNN::exact())");

    let mut exact_recalls = Vec::new();
    for (qi, query) in query_vectors.iter().enumerate() {
        let results = SearchKNN::new(&embedding, query.clone(), K)
            .with_ef(100)
            .exact()
            .run(&search_reader, timeout)
            .await
            .expect("exact search");

        // Convert results to indices via external_ids
        let result_indices: Vec<usize> = results
            .iter()
            .filter_map(|r| external_ids.iter().position(|&id| id == r.node_id().expect("expected NodeId")))
            .collect();

        let recall = compute_recall(&ground_truth[qi], &result_indices);
        exact_recalls.push(recall);

        if qi == 0 {
            println!("  Query 0 top-{} results:", K);
            for (i, r) in results.iter().take(5).enumerate() {
                println!("    {}. id={}, distance={:.4}", i + 1, r.node_id().expect("expected NodeId"), r.distance);
            }
        }
    }

    let avg_exact_recall = exact_recalls.iter().sum::<f32>() / exact_recalls.len() as f32;
    println!(
        "  Average Recall@{} (Exact): {:.2}%",
        K,
        avg_exact_recall * 100.0
    );

    // =========================================================================
    // Step 5: Search using RaBitQ (quantized) strategy via SearchKNN (auto)
    // =========================================================================
    println!("\n=== Step 5: Searching with RaBitQ (quantized) strategy ===");
    println!("  Strategy: RaBitQ (auto-selected for Cosine)");
    println!("  Rerank factor: 10");

    let mut rabitq_recalls = Vec::new();
    for (qi, query) in query_vectors.iter().enumerate() {
        // SearchKNN without .exact() auto-selects RaBitQ for Cosine
        let results = SearchKNN::new(&embedding, query.clone(), K)
            .with_ef(100)
            .with_rerank_factor(10)
            .run(&search_reader, timeout)
            .await
            .expect("rabitq search");

        let result_indices: Vec<usize> = results
            .iter()
            .filter_map(|r| external_ids.iter().position(|&id| id == r.node_id().expect("expected NodeId")))
            .collect();

        let recall = compute_recall(&ground_truth[qi], &result_indices);
        rabitq_recalls.push(recall);

        if qi == 0 {
            println!("  Query 0 top-{} results:", K);
            for (i, r) in results.iter().take(5).enumerate() {
                println!("    {}. id={}, distance={:.4}", i + 1, r.node_id().expect("expected NodeId"), r.distance);
            }
        }
    }

    let avg_rabitq_recall = rabitq_recalls.iter().sum::<f32>() / rabitq_recalls.len() as f32;
    println!(
        "  Average Recall@{} (RaBitQ): {:.2}%",
        K,
        avg_rabitq_recall * 100.0
    );

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
    println!("\n=== Step 6: Deleting vectors via mutation API ===");

    // Delete first 100 vectors using DeleteVector::run(&writer)
    let delete_count = 100;
    let deleted_ids: Vec<Id> = external_ids[..delete_count].to_vec();

    for (i, &id) in deleted_ids.iter().enumerate() {
        DeleteVector::new(&embedding, ExternalKey::NodeId(id))
            .run(&writer)
            .await
            .expect("delete vector");

        if (i + 1) % 50 == 0 {
            println!("  Deleted {}/{} vectors", i + 1, delete_count);
        }
    }

    // Flush to ensure all deletes are committed
    writer.flush().await.expect("flush writer");
    println!("  Deleted {} vectors", delete_count);

    // Search again and verify deleted vectors don't appear
    println!("\n=== Step 7: Verifying deleted vectors don't appear in results ===");
    let deleted_id_set: HashSet<Id> = deleted_ids.iter().copied().collect();

    for (qi, query) in query_vectors.iter().enumerate() {
        // Test with exact strategy
        let results = SearchKNN::new(&embedding, query.clone(), K)
            .with_ef(100)
            .exact()
            .run(&search_reader, timeout)
            .await
            .expect("search after delete");

        for result in &results {
            assert!(
                !deleted_id_set.contains(&result.node_id().expect("expected NodeId")),
                "Query {}: Deleted ID {} should not appear in exact search results",
                qi,
                result.node_id().expect("expected NodeId")
            );
        }

        // Test with RaBitQ strategy
        let results = SearchKNN::new(&embedding, query.clone(), K)
            .with_ef(100)
            .with_rerank_factor(10)
            .run(&search_reader, timeout)
            .await
            .expect("rabitq search after delete");

        for result in &results {
            assert!(
                !deleted_id_set.contains(&result.node_id().expect("expected NodeId")),
                "Query {}: Deleted ID {} should not appear in RaBitQ search results",
                qi,
                result.node_id().expect("expected NodeId")
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

/// Test batch insert workflow via InsertVector::run() in a loop.
///
/// Note: The original test used Processor::insert_batch() directly,
/// but since we're migrating to Runnable traits, we use individual
/// InsertVector::run() calls. For true batching, use InsertVectorBatch.
#[tokio::test]
async fn test_vector_batch_insert_workflow() {
    let temp_dir = TempDir::new().expect("temp dir");
    let mut storage = Storage::readwrite(temp_dir.path());
    storage.ready().expect("storage ready");
    let storage = Arc::new(storage);

    let registry = storage.cache().clone();
    let txn_db = storage.transaction_db().expect("txn_db");

    // Register embedding
    let builder =
        EmbeddingBuilder::new("clip-vit-b32", LAION_EMBEDDING_DIM as u32, Distance::Cosine);
    let embedding = registry.register(builder, &txn_db).expect("register");

    // Create writer and spawn mutation consumer
    let (writer, writer_rx) = create_writer(WriterConfig::default());
    let _writer_handle = spawn_mutation_consumer_with_storage_autoreg(
        writer_rx,
        WriterConfig::default(),
        storage.clone(),
    );

    // Create search reader and spawn query consumers
    let (search_reader, reader_rx) =
        create_search_reader_with_storage(ReaderConfig::default(), storage.clone());
    let _reader_handles = spawn_query_consumers_with_storage_autoreg(
        reader_rx,
        ReaderConfig::default(),
        storage.clone(),
        2,
    );

    // Generate vectors
    let vectors = generate_laion_vectors(500, 42);

    // Insert via InsertVector::run()
    println!("Inserting 500 vectors via InsertVector::run()...");
    for vector in &vectors {
        InsertVector::new(&embedding, ExternalKey::NodeId(Id::new()), vector.clone())
            .immediate()
            .run(&writer)
            .await
            .expect("insert");
    }
    writer.flush().await.expect("flush");
    println!("Insert successful: {} vectors", vectors.len());

    // Verify search works
    let query = generate_laion_vectors(1, 999)[0].clone();
    let timeout = Duration::from_secs(5);

    let results = SearchKNN::new(&embedding, query, 10)
        .with_ef(50)
        .run(&search_reader, timeout)
        .await
        .expect("search");

    assert!(!results.is_empty(), "Should have search results");
    println!("Search returned {} results", results.len());
}

/// Test SearchReader API with both exact and auto-strategy modes.
///
/// This test verifies that:
/// - SearchKNN with `.exact()` forces exact distance computation
/// - SearchKNN without `.exact()` auto-selects strategy based on distance metric
///
/// Migrated from Processor::search_with_config() to SearchKNN::run().
#[tokio::test]
async fn test_search_reader_strategy_selection() {
    let temp_dir = TempDir::new().expect("temp dir");
    let mut storage = Storage::readwrite(temp_dir.path());
    storage.ready().expect("storage ready");
    let storage = Arc::new(storage);

    let registry = storage.cache().clone();
    let txn_db = storage.transaction_db().expect("txn_db");

    // Register Cosine embedding (required for RaBitQ)
    let builder =
        EmbeddingBuilder::new("clip-vit-b32", LAION_EMBEDDING_DIM as u32, Distance::Cosine);
    let embedding = registry.register(builder, &txn_db).expect("register");

    // Create writer and spawn mutation consumer
    let (writer, writer_rx) = create_writer(WriterConfig::default());
    let _writer_handle = spawn_mutation_consumer_with_storage_autoreg(
        writer_rx,
        WriterConfig::default(),
        storage.clone(),
    );

    // Create search reader and spawn query consumers
    let (search_reader, reader_rx) =
        create_search_reader_with_storage(ReaderConfig::default(), storage.clone());
    let _reader_handles = spawn_query_consumers_with_storage_autoreg(
        reader_rx,
        ReaderConfig::default(),
        storage.clone(),
        2,
    );

    let timeout = Duration::from_secs(5);

    // Insert test vectors
    println!("Inserting 200 test vectors...");
    let vectors = generate_laion_vectors(200, 42);
    for (i, vector) in vectors.iter().enumerate() {
        InsertVector::new(&embedding, ExternalKey::NodeId(Id::new()), vector.clone())
            .immediate()
            .run(&writer)
            .await
            .expect("insert");
        if (i + 1) % 100 == 0 {
            println!("  Inserted {}/200", i + 1);
        }
    }
    writer.flush().await.expect("flush");

    let query = generate_laion_vectors(1, 999)[0].clone();

    // =========================================================================
    // Test 1: SearchKNN with EXACT strategy
    // =========================================================================
    println!("\n=== Testing SearchKNN with .exact() ===");
    let exact_results = SearchKNN::new(&embedding, query.clone(), 10)
        .with_ef(50)
        .exact()
        .run(&search_reader, timeout)
        .await
        .expect("exact search");

    assert!(
        !exact_results.is_empty(),
        "Exact search should return results"
    );
    println!("  Exact search returned {} results", exact_results.len());

    // =========================================================================
    // Test 2: SearchKNN with auto-strategy (RaBitQ for Cosine)
    // =========================================================================
    println!("\n=== Testing SearchKNN (auto-strategy) ===");
    let rabitq_results = SearchKNN::new(&embedding, query.clone(), 10)
        .with_ef(50)
        .with_rerank_factor(10)
        .run(&search_reader, timeout)
        .await
        .expect("auto-strategy search");

    assert!(
        !rabitq_results.is_empty(),
        "Auto-strategy search should return results"
    );
    println!(
        "  Auto-strategy search returned {} results",
        rabitq_results.len()
    );

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
    assert!(
        diff < 0.1,
        "Top results should be similar (diff={:.4})",
        diff
    );

    println!("\nSearchKNN strategy selection test PASSED!");
}
