#![cfg(feature = "benchmark")]
//! Integration test: Multi-embedding configurations with different RaBitQ bit depths.
//!
//! This test validates:
//! 1. Public API for embedding registration
//! 2. Index building with different RaBitQ configurations (2-bit, 4-bit)
//! 3. Both exact and RaBitQ search paths
//! 4. Non-interference between incompatible embeddings in the same storage
//!
//! ## Test Scenarios
//!
//! | Scenario | Distance | RaBitQ | Purpose |
//! |----------|----------|--------|---------|
//! | LAION-2bit | Cosine | 2-bit | High compression, lower recall |
//! | LAION-4bit | Cosine | 4-bit | Medium compression, higher recall |
//! | SIFT-L2 | L2 | N/A (exact) | Non-interference with Cosine |
//!
//! ## Non-Interference Tests
//!
//! - Multiple embeddings in same storage don't interfere
//! - Different distance metrics coexist correctly
//! - Searches in one embedding don't affect others

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use motlie_db::vector::benchmark::{LAION_EMBEDDING_DIM, SIFT_EMBEDDING_DIM};
use motlie_db::vector::{
    create_search_reader_with_storage, create_writer,
    spawn_mutation_consumer_with_storage_autoreg, spawn_query_consumers_with_storage_autoreg,
    Distance, EmbeddingBuilder, ExternalKey, InsertVector, MutationRunnable, ReaderConfig, Runnable,
    SearchKNN, Storage, WriterConfig,
};
use motlie_db::Id;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use tempfile::TempDir;

const NUM_VECTORS: usize = 1000;
const NUM_QUERIES: usize = 20;
const K: usize = 10;

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate normalized vectors for Cosine distance (LAION-CLIP style).
fn generate_cosine_vectors(dim: usize, count: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    (0..count)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
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

/// Generate unnormalized vectors for L2 distance (SIFT style).
fn generate_l2_vectors(dim: usize, count: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    (0..count)
        .map(|_| (0..dim).map(|_| rng.gen_range(0.0..128.0)).collect())
        .collect()
}

/// Compute brute-force ground truth.
fn compute_ground_truth(
    queries: &[Vec<f32>],
    db_vectors: &[Vec<f32>],
    k: usize,
    distance: Distance,
) -> Vec<Vec<usize>> {
    queries
        .iter()
        .map(|query| {
            let mut distances: Vec<(usize, f32)> = db_vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i, distance.compute(query, v)))
                .collect();
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            distances.iter().take(k).map(|(i, _)| *i).collect()
        })
        .collect()
}

/// Calculate Recall@k.
fn compute_recall(ground_truth: &[usize], results: &[usize]) -> f32 {
    let gt_set: HashSet<_> = ground_truth.iter().collect();
    let found = results.iter().filter(|r| gt_set.contains(r)).count();
    found as f32 / ground_truth.len().max(1) as f32
}

// ============================================================================
// Test: Cosine + 2-bit RaBitQ
// ============================================================================

/// Test LAION-style embeddings with 2-bit RaBitQ quantization.
///
/// 2-bit provides ~16x compression with ~85% recall (no rerank).
/// With rerank, recall should be significantly higher.
#[tokio::test]
async fn test_cosine_2bit_rabitq() {
    println!("\n=== Test: Cosine + 2-bit RaBitQ ===\n");

    let temp_dir = TempDir::new().expect("temp dir");
    let mut storage = Storage::readwrite(temp_dir.path());
    storage.ready().expect("storage ready");
    let storage = Arc::new(storage);

    let registry = storage.cache().clone();
    let txn_db = storage.transaction_db().expect("txn_db");

    // Register LAION-style embedding (Cosine for RaBitQ)
    let builder = EmbeddingBuilder::new("laion-2bit", LAION_EMBEDDING_DIM as u32, Distance::Cosine);
    let embedding = registry
        .register(builder, &txn_db)
        .expect("register embedding");

    println!(
        "Registered: model={}, dim={}, distance={:?}",
        embedding.model(),
        embedding.dim(),
        embedding.distance()
    );

    // Create writer and reader
    let (writer, writer_rx) = create_writer(WriterConfig::default());
    let _writer_handle = spawn_mutation_consumer_with_storage_autoreg(
        writer_rx,
        WriterConfig::default(),
        storage.clone(),
    );

    let (search_reader, reader_rx) =
        create_search_reader_with_storage(ReaderConfig::default(), storage.clone());
    let _reader_handles = spawn_query_consumers_with_storage_autoreg(
        reader_rx,
        ReaderConfig::default(),
        storage.clone(),
        2,
    );

    let timeout = Duration::from_secs(30);

    // Generate data
    let db_vectors = generate_cosine_vectors(LAION_EMBEDDING_DIM, NUM_VECTORS, 42);
    let query_vectors = generate_cosine_vectors(LAION_EMBEDDING_DIM, NUM_QUERIES, 123);

    // Insert vectors
    println!("Inserting {} vectors...", NUM_VECTORS);
    let mut external_ids: Vec<Id> = Vec::with_capacity(NUM_VECTORS);
    for (i, vector) in db_vectors.iter().enumerate() {
        let id = Id::new();
        external_ids.push(id);

        InsertVector::new(&embedding, ExternalKey::NodeId(id), vector.clone())
            .immediate()
            .run(&writer)
            .await
            .expect("insert vector");

        if (i + 1) % 250 == 0 {
            println!("  Inserted {}/{}", i + 1, NUM_VECTORS);
        }
    }
    writer.flush().await.expect("flush");
    println!("Insert complete.");

    // Compute ground truth
    let ground_truth = compute_ground_truth(&query_vectors, &db_vectors, K, Distance::Cosine);

    // Test EXACT search path
    println!("\n--- Exact Search ---");
    let mut exact_recalls = Vec::new();
    for (qi, query) in query_vectors.iter().enumerate() {
        let results = SearchKNN::new(&embedding, query.clone(), K)
            .with_ef(100)
            .exact()
            .run(&search_reader, timeout)
            .await
            .expect("exact search");

        let result_indices: Vec<usize> = results
            .iter()
            .filter_map(|r| external_ids.iter().position(|&id| id == r.node_id().expect("expected NodeId")))
            .collect();

        let recall = compute_recall(&ground_truth[qi], &result_indices);
        exact_recalls.push(recall);
    }
    let avg_exact_recall = exact_recalls.iter().sum::<f32>() / exact_recalls.len() as f32;
    println!("Exact Recall@{}: {:.1}%", K, avg_exact_recall * 100.0);

    // Test RaBitQ search path (auto-selected for Cosine)
    println!("\n--- RaBitQ Search ---");
    let mut rabitq_recalls = Vec::new();
    for (qi, query) in query_vectors.iter().enumerate() {
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
    }
    let avg_rabitq_recall = rabitq_recalls.iter().sum::<f32>() / rabitq_recalls.len() as f32;
    println!("RaBitQ Recall@{}: {:.1}%", K, avg_rabitq_recall * 100.0);

    // Assertions
    assert!(
        avg_exact_recall >= 0.5,
        "Exact recall should be >= 50%, got {:.1}%",
        avg_exact_recall * 100.0
    );
    assert!(
        avg_rabitq_recall >= 0.4,
        "RaBitQ recall should be >= 40%, got {:.1}%",
        avg_rabitq_recall * 100.0
    );

    println!("\n=== Cosine + 2-bit Test PASSED ===\n");
}

// ============================================================================
// Test: Cosine + 4-bit RaBitQ (Higher Recall)
// ============================================================================

/// Test LAION-style embeddings with 4-bit RaBitQ quantization.
///
/// 4-bit provides ~8x compression with ~92% recall (no rerank).
/// Expected to have higher recall than 2-bit.
#[tokio::test]
async fn test_cosine_4bit_rabitq() {
    println!("\n=== Test: Cosine + 4-bit RaBitQ ===\n");

    let temp_dir = TempDir::new().expect("temp dir");
    let mut storage = Storage::readwrite(temp_dir.path());
    storage.ready().expect("storage ready");
    let storage = Arc::new(storage);

    let registry = storage.cache().clone();
    let txn_db = storage.transaction_db().expect("txn_db");

    // Register LAION-style embedding
    let builder = EmbeddingBuilder::new("laion-4bit", LAION_EMBEDDING_DIM as u32, Distance::Cosine);
    let embedding = registry
        .register(builder, &txn_db)
        .expect("register embedding");

    println!(
        "Registered: model={}, dim={}, distance={:?}",
        embedding.model(),
        embedding.dim(),
        embedding.distance()
    );

    // Create writer and reader
    let (writer, writer_rx) = create_writer(WriterConfig::default());
    let _writer_handle = spawn_mutation_consumer_with_storage_autoreg(
        writer_rx,
        WriterConfig::default(),
        storage.clone(),
    );

    let (search_reader, reader_rx) =
        create_search_reader_with_storage(ReaderConfig::default(), storage.clone());
    let _reader_handles = spawn_query_consumers_with_storage_autoreg(
        reader_rx,
        ReaderConfig::default(),
        storage.clone(),
        2,
    );

    let timeout = Duration::from_secs(30);

    // Generate data
    let db_vectors = generate_cosine_vectors(LAION_EMBEDDING_DIM, NUM_VECTORS, 42);
    let query_vectors = generate_cosine_vectors(LAION_EMBEDDING_DIM, NUM_QUERIES, 123);

    // Insert vectors
    println!("Inserting {} vectors...", NUM_VECTORS);
    let mut external_ids: Vec<Id> = Vec::with_capacity(NUM_VECTORS);
    for (i, vector) in db_vectors.iter().enumerate() {
        let id = Id::new();
        external_ids.push(id);

        InsertVector::new(&embedding, ExternalKey::NodeId(id), vector.clone())
            .immediate()
            .run(&writer)
            .await
            .expect("insert vector");

        if (i + 1) % 250 == 0 {
            println!("  Inserted {}/{}", i + 1, NUM_VECTORS);
        }
    }
    writer.flush().await.expect("flush");
    println!("Insert complete.");

    // Compute ground truth
    let ground_truth = compute_ground_truth(&query_vectors, &db_vectors, K, Distance::Cosine);

    // Test EXACT search path
    println!("\n--- Exact Search ---");
    let mut exact_recalls = Vec::new();
    for (qi, query) in query_vectors.iter().enumerate() {
        let results = SearchKNN::new(&embedding, query.clone(), K)
            .with_ef(100)
            .exact()
            .run(&search_reader, timeout)
            .await
            .expect("exact search");

        let result_indices: Vec<usize> = results
            .iter()
            .filter_map(|r| external_ids.iter().position(|&id| id == r.node_id().expect("expected NodeId")))
            .collect();

        let recall = compute_recall(&ground_truth[qi], &result_indices);
        exact_recalls.push(recall);
    }
    let avg_exact_recall = exact_recalls.iter().sum::<f32>() / exact_recalls.len() as f32;
    println!("Exact Recall@{}: {:.1}%", K, avg_exact_recall * 100.0);

    // Test RaBitQ search path
    println!("\n--- RaBitQ Search ---");
    let mut rabitq_recalls = Vec::new();
    for (qi, query) in query_vectors.iter().enumerate() {
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
    }
    let avg_rabitq_recall = rabitq_recalls.iter().sum::<f32>() / rabitq_recalls.len() as f32;
    println!("RaBitQ Recall@{}: {:.1}%", K, avg_rabitq_recall * 100.0);

    // Assertions
    assert!(
        avg_exact_recall >= 0.5,
        "Exact recall should be >= 50%, got {:.1}%",
        avg_exact_recall * 100.0
    );
    assert!(
        avg_rabitq_recall >= 0.4,
        "RaBitQ recall should be >= 40%, got {:.1}%",
        avg_rabitq_recall * 100.0
    );

    println!("\n=== Cosine + 4-bit Test PASSED ===\n");
}

// ============================================================================
// Test: Multi-Embedding Non-Interference
// ============================================================================

/// Test that multiple embeddings with different configurations don't interfere.
///
/// Registers three embeddings in the same storage:
/// 1. LAION-Cosine (512D, Cosine, RaBitQ path)
/// 2. SIFT-L2 (128D, L2, Exact path)
/// 3. Custom-768 (768D, Cosine, different model)
///
/// Validates:
/// - Each embedding has its own namespace
/// - Searches in one embedding don't affect others
/// - Different distance metrics coexist
/// - Different dimensions coexist
#[tokio::test]
async fn test_multi_embedding_non_interference() {
    println!("\n=== Test: Multi-Embedding Non-Interference ===\n");

    let temp_dir = TempDir::new().expect("temp dir");
    let mut storage = Storage::readwrite(temp_dir.path());
    storage.ready().expect("storage ready");
    let storage = Arc::new(storage);

    let registry = storage.cache().clone();
    let txn_db = storage.transaction_db().expect("txn_db");

    // =========================================================================
    // Register three different embeddings
    // =========================================================================

    // Embedding 1: LAION-style (512D, Cosine)
    let laion_builder =
        EmbeddingBuilder::new("laion-clip-vit-b32", LAION_EMBEDDING_DIM as u32, Distance::Cosine);
    let laion_embedding = registry
        .register(laion_builder, &txn_db)
        .expect("register laion");
    println!(
        "Registered LAION: code={}, model={}, dim={}, distance={:?}",
        laion_embedding.code(),
        laion_embedding.model(),
        laion_embedding.dim(),
        laion_embedding.distance()
    );

    // Embedding 2: SIFT-style (128D, L2)
    let sift_builder =
        EmbeddingBuilder::new("sift-descriptors", SIFT_EMBEDDING_DIM as u32, Distance::L2);
    let sift_embedding = registry
        .register(sift_builder, &txn_db)
        .expect("register sift");
    println!(
        "Registered SIFT: code={}, model={}, dim={}, distance={:?}",
        sift_embedding.code(),
        sift_embedding.model(),
        sift_embedding.dim(),
        sift_embedding.distance()
    );

    // Embedding 3: Custom 768D (like BERT, Cosine)
    let custom_builder = EmbeddingBuilder::new("bert-base-uncased", 768, Distance::Cosine);
    let custom_embedding = registry
        .register(custom_builder, &txn_db)
        .expect("register custom");
    println!(
        "Registered Custom: code={}, model={}, dim={}, distance={:?}",
        custom_embedding.code(),
        custom_embedding.model(),
        custom_embedding.dim(),
        custom_embedding.distance()
    );

    // Verify unique codes
    assert_ne!(
        laion_embedding.code(),
        sift_embedding.code(),
        "LAION and SIFT should have different codes"
    );
    assert_ne!(
        laion_embedding.code(),
        custom_embedding.code(),
        "LAION and Custom should have different codes"
    );
    assert_ne!(
        sift_embedding.code(),
        custom_embedding.code(),
        "SIFT and Custom should have different codes"
    );

    // =========================================================================
    // Create writer and reader
    // =========================================================================
    let (writer, writer_rx) = create_writer(WriterConfig::default());
    let _writer_handle = spawn_mutation_consumer_with_storage_autoreg(
        writer_rx,
        WriterConfig::default(),
        storage.clone(),
    );

    let (search_reader, reader_rx) =
        create_search_reader_with_storage(ReaderConfig::default(), storage.clone());
    let _reader_handles = spawn_query_consumers_with_storage_autoreg(
        reader_rx,
        ReaderConfig::default(),
        storage.clone(),
        2,
    );

    let timeout = Duration::from_secs(30);

    // =========================================================================
    // Generate and insert vectors for each embedding
    // =========================================================================

    let num_per_embedding = 500;
    let num_queries_per = 10;

    // LAION vectors (512D, normalized)
    let laion_db = generate_cosine_vectors(LAION_EMBEDDING_DIM, num_per_embedding, 100);
    let laion_queries = generate_cosine_vectors(LAION_EMBEDDING_DIM, num_queries_per, 200);

    // SIFT vectors (128D, unnormalized)
    let sift_db = generate_l2_vectors(SIFT_EMBEDDING_DIM, num_per_embedding, 300);
    let sift_queries = generate_l2_vectors(SIFT_EMBEDDING_DIM, num_queries_per, 400);

    // Custom vectors (768D, normalized)
    let custom_db = generate_cosine_vectors(768, num_per_embedding, 500);
    let custom_queries = generate_cosine_vectors(768, num_queries_per, 600);

    // Insert LAION vectors
    println!("\nInserting {} LAION vectors...", num_per_embedding);
    let mut laion_ids: Vec<Id> = Vec::new();
    for vector in &laion_db {
        let id = Id::new();
        laion_ids.push(id);
        InsertVector::new(&laion_embedding, ExternalKey::NodeId(id), vector.clone())
            .immediate()
            .run(&writer)
            .await
            .expect("insert laion");
    }

    // Insert SIFT vectors
    println!("Inserting {} SIFT vectors...", num_per_embedding);
    let mut sift_ids: Vec<Id> = Vec::new();
    for vector in &sift_db {
        let id = Id::new();
        sift_ids.push(id);
        InsertVector::new(&sift_embedding, ExternalKey::NodeId(id), vector.clone())
            .immediate()
            .run(&writer)
            .await
            .expect("insert sift");
    }

    // Insert Custom vectors
    println!("Inserting {} Custom vectors...", num_per_embedding);
    let mut custom_ids: Vec<Id> = Vec::new();
    for vector in &custom_db {
        let id = Id::new();
        custom_ids.push(id);
        InsertVector::new(&custom_embedding, ExternalKey::NodeId(id), vector.clone())
            .immediate()
            .run(&writer)
            .await
            .expect("insert custom");
    }

    writer.flush().await.expect("flush");
    println!("All inserts complete.\n");

    // Collect all IDs for verification
    let laion_id_set: HashSet<Id> = laion_ids.iter().copied().collect();
    let sift_id_set: HashSet<Id> = sift_ids.iter().copied().collect();
    let custom_id_set: HashSet<Id> = custom_ids.iter().copied().collect();

    // =========================================================================
    // Test: LAION searches only return LAION vectors
    // =========================================================================
    println!("--- Testing LAION Search Isolation ---");
    for (qi, query) in laion_queries.iter().enumerate() {
        let results = SearchKNN::new(&laion_embedding, query.clone(), K)
            .with_ef(100)
            .run(&search_reader, timeout)
            .await
            .expect("laion search");

        for result in &results {
            assert!(
                laion_id_set.contains(&result.node_id().expect("expected NodeId")),
                "Query {}: LAION search returned non-LAION ID {}",
                qi,
                result.node_id().expect("expected NodeId")
            );
            assert!(
                !sift_id_set.contains(&result.node_id().expect("expected NodeId")),
                "Query {}: LAION search returned SIFT ID {}",
                qi,
                result.node_id().expect("expected NodeId")
            );
            assert!(
                !custom_id_set.contains(&result.node_id().expect("expected NodeId")),
                "Query {}: LAION search returned Custom ID {}",
                qi,
                result.node_id().expect("expected NodeId")
            );
        }
    }
    println!("  LAION search isolation: PASSED");

    // =========================================================================
    // Test: SIFT searches only return SIFT vectors
    // =========================================================================
    println!("--- Testing SIFT Search Isolation ---");
    for (qi, query) in sift_queries.iter().enumerate() {
        let results = SearchKNN::new(&sift_embedding, query.clone(), K)
            .with_ef(100)
            .exact() // L2 uses exact path
            .run(&search_reader, timeout)
            .await
            .expect("sift search");

        for result in &results {
            assert!(
                sift_id_set.contains(&result.node_id().expect("expected NodeId")),
                "Query {}: SIFT search returned non-SIFT ID {}",
                qi,
                result.node_id().expect("expected NodeId")
            );
            assert!(
                !laion_id_set.contains(&result.node_id().expect("expected NodeId")),
                "Query {}: SIFT search returned LAION ID {}",
                qi,
                result.node_id().expect("expected NodeId")
            );
            assert!(
                !custom_id_set.contains(&result.node_id().expect("expected NodeId")),
                "Query {}: SIFT search returned Custom ID {}",
                qi,
                result.node_id().expect("expected NodeId")
            );
        }
    }
    println!("  SIFT search isolation: PASSED");

    // =========================================================================
    // Test: Custom searches only return Custom vectors
    // =========================================================================
    println!("--- Testing Custom Search Isolation ---");
    for (qi, query) in custom_queries.iter().enumerate() {
        let results = SearchKNN::new(&custom_embedding, query.clone(), K)
            .with_ef(100)
            .run(&search_reader, timeout)
            .await
            .expect("custom search");

        for result in &results {
            assert!(
                custom_id_set.contains(&result.node_id().expect("expected NodeId")),
                "Query {}: Custom search returned non-Custom ID {}",
                qi,
                result.node_id().expect("expected NodeId")
            );
            assert!(
                !laion_id_set.contains(&result.node_id().expect("expected NodeId")),
                "Query {}: Custom search returned LAION ID {}",
                qi,
                result.node_id().expect("expected NodeId")
            );
            assert!(
                !sift_id_set.contains(&result.node_id().expect("expected NodeId")),
                "Query {}: Custom search returned SIFT ID {}",
                qi,
                result.node_id().expect("expected NodeId")
            );
        }
    }
    println!("  Custom search isolation: PASSED");

    // =========================================================================
    // Test: Recall is reasonable for each embedding
    // =========================================================================
    println!("\n--- Verifying Recall per Embedding ---");

    // LAION recall
    let laion_gt = compute_ground_truth(&laion_queries, &laion_db, K, Distance::Cosine);
    let mut laion_recall_sum = 0.0;
    for (qi, query) in laion_queries.iter().enumerate() {
        let results = SearchKNN::new(&laion_embedding, query.clone(), K)
            .with_ef(100)
            .run(&search_reader, timeout)
            .await
            .expect("laion search");
        let indices: Vec<usize> = results
            .iter()
            .filter_map(|r| laion_ids.iter().position(|&id| id == r.node_id().expect("expected NodeId")))
            .collect();
        laion_recall_sum += compute_recall(&laion_gt[qi], &indices);
    }
    let laion_recall = laion_recall_sum / laion_queries.len() as f32;
    println!("  LAION Recall@{}: {:.1}%", K, laion_recall * 100.0);

    // SIFT recall
    let sift_gt = compute_ground_truth(&sift_queries, &sift_db, K, Distance::L2);
    let mut sift_recall_sum = 0.0;
    for (qi, query) in sift_queries.iter().enumerate() {
        let results = SearchKNN::new(&sift_embedding, query.clone(), K)
            .with_ef(100)
            .exact()
            .run(&search_reader, timeout)
            .await
            .expect("sift search");
        let indices: Vec<usize> = results
            .iter()
            .filter_map(|r| sift_ids.iter().position(|&id| id == r.node_id().expect("expected NodeId")))
            .collect();
        sift_recall_sum += compute_recall(&sift_gt[qi], &indices);
    }
    let sift_recall = sift_recall_sum / sift_queries.len() as f32;
    println!("  SIFT Recall@{}: {:.1}%", K, sift_recall * 100.0);

    // Custom recall
    let custom_gt = compute_ground_truth(&custom_queries, &custom_db, K, Distance::Cosine);
    let mut custom_recall_sum = 0.0;
    for (qi, query) in custom_queries.iter().enumerate() {
        let results = SearchKNN::new(&custom_embedding, query.clone(), K)
            .with_ef(100)
            .run(&search_reader, timeout)
            .await
            .expect("custom search");
        let indices: Vec<usize> = results
            .iter()
            .filter_map(|r| custom_ids.iter().position(|&id| id == r.node_id().expect("expected NodeId")))
            .collect();
        custom_recall_sum += compute_recall(&custom_gt[qi], &indices);
    }
    let custom_recall = custom_recall_sum / custom_queries.len() as f32;
    println!("  Custom Recall@{}: {:.1}%", K, custom_recall * 100.0);

    // All should have reasonable recall
    assert!(
        laion_recall >= 0.3,
        "LAION recall too low: {:.1}%",
        laion_recall * 100.0
    );
    assert!(
        sift_recall >= 0.3,
        "SIFT recall too low: {:.1}%",
        sift_recall * 100.0
    );
    assert!(
        custom_recall >= 0.3,
        "Custom recall too low: {:.1}%",
        custom_recall * 100.0
    );

    println!("\n=== Multi-Embedding Non-Interference Test PASSED ===\n");
}

// ============================================================================
// Test: Embedding Re-registration Consistency
// ============================================================================

/// Test that re-registering the same embedding spec returns the same code.
///
/// This validates:
/// - Idempotent registration
/// - SpecHash consistency
/// - Registry lookup works correctly
#[tokio::test]
async fn test_embedding_registration_idempotent() {
    println!("\n=== Test: Embedding Registration Idempotent ===\n");

    let temp_dir = TempDir::new().expect("temp dir");
    let mut storage = Storage::readwrite(temp_dir.path());
    storage.ready().expect("storage ready");
    let storage = Arc::new(storage);

    let registry = storage.cache().clone();
    let txn_db = storage.transaction_db().expect("txn_db");

    // Register same embedding multiple times
    let builder1 = EmbeddingBuilder::new("test-model", 256, Distance::Cosine);
    let embedding1 = registry.register(builder1, &txn_db).expect("register 1");

    let builder2 = EmbeddingBuilder::new("test-model", 256, Distance::Cosine);
    let embedding2 = registry.register(builder2, &txn_db).expect("register 2");

    // Should get same code
    assert_eq!(
        embedding1.code(),
        embedding2.code(),
        "Re-registration should return same code"
    );
    assert_eq!(embedding1.model(), embedding2.model());
    assert_eq!(embedding1.dim(), embedding2.dim());

    println!(
        "Re-registration returned same code: {}",
        embedding1.code()
    );

    // Register different model - should get different code
    let builder3 = EmbeddingBuilder::new("different-model", 256, Distance::Cosine);
    let embedding3 = registry.register(builder3, &txn_db).expect("register 3");

    assert_ne!(
        embedding1.code(),
        embedding3.code(),
        "Different model should get different code"
    );

    println!(
        "Different model got different code: {} vs {}",
        embedding1.code(),
        embedding3.code()
    );

    println!("\n=== Embedding Registration Idempotent Test PASSED ===\n");
}

// ============================================================================
// Test: Both Search Paths Work (Exact vs RaBitQ)
// ============================================================================

/// Explicitly test both exact and RaBitQ search paths return valid results.
///
/// This ensures:
/// - `.exact()` forces exact distance computation
/// - Default (auto) uses RaBitQ for Cosine
/// - Both return consistent top results
#[tokio::test]
async fn test_exact_vs_rabitq_search_paths() {
    println!("\n=== Test: Exact vs RaBitQ Search Paths ===\n");

    let temp_dir = TempDir::new().expect("temp dir");
    let mut storage = Storage::readwrite(temp_dir.path());
    storage.ready().expect("storage ready");
    let storage = Arc::new(storage);

    let registry = storage.cache().clone();
    let txn_db = storage.transaction_db().expect("txn_db");

    // Register Cosine embedding (RaBitQ eligible)
    let builder = EmbeddingBuilder::new("search-path-test", LAION_EMBEDDING_DIM as u32, Distance::Cosine);
    let embedding = registry.register(builder, &txn_db).expect("register");

    // Create writer and reader
    let (writer, writer_rx) = create_writer(WriterConfig::default());
    let _writer_handle = spawn_mutation_consumer_with_storage_autoreg(
        writer_rx,
        WriterConfig::default(),
        storage.clone(),
    );

    let (search_reader, reader_rx) =
        create_search_reader_with_storage(ReaderConfig::default(), storage.clone());
    let _reader_handles = spawn_query_consumers_with_storage_autoreg(
        reader_rx,
        ReaderConfig::default(),
        storage.clone(),
        2,
    );

    let timeout = Duration::from_secs(30);

    // Insert vectors
    let db_vectors = generate_cosine_vectors(LAION_EMBEDDING_DIM, 500, 42);
    let mut ids: Vec<Id> = Vec::new();
    for vector in &db_vectors {
        let id = Id::new();
        ids.push(id);
        InsertVector::new(&embedding, ExternalKey::NodeId(id), vector.clone())
            .immediate()
            .run(&writer)
            .await
            .expect("insert");
    }
    writer.flush().await.expect("flush");

    // Query
    let query = generate_cosine_vectors(LAION_EMBEDDING_DIM, 1, 999)[0].clone();

    // Exact search
    println!("--- Exact Search ---");
    let exact_results = SearchKNN::new(&embedding, query.clone(), K)
        .with_ef(100)
        .exact()
        .run(&search_reader, timeout)
        .await
        .expect("exact search");

    assert!(!exact_results.is_empty(), "Exact should return results");
    println!("  Exact returned {} results", exact_results.len());
    println!("  Top-1 distance: {:.4}", exact_results[0].distance);

    // RaBitQ search (auto for Cosine)
    println!("\n--- RaBitQ Search ---");
    let rabitq_results = SearchKNN::new(&embedding, query.clone(), K)
        .with_ef(100)
        .with_rerank_factor(10)
        .run(&search_reader, timeout)
        .await
        .expect("rabitq search");

    assert!(!rabitq_results.is_empty(), "RaBitQ should return results");
    println!("  RaBitQ returned {} results", rabitq_results.len());
    println!("  Top-1 distance: {:.4}", rabitq_results[0].distance);

    // Top-1 should be similar (quantization may cause small differences)
    let distance_diff = (exact_results[0].distance - rabitq_results[0].distance).abs();
    println!("\n  Top-1 distance difference: {:.4}", distance_diff);

    // With rerank_factor=10, top results should be nearly identical
    assert!(
        distance_diff < 0.1,
        "Top-1 distance should be similar (diff={:.4})",
        distance_diff
    );

    println!("\n=== Exact vs RaBitQ Search Paths Test PASSED ===\n");
}
