//! Integration tests: Vector MPSC/MPMC Channel Infrastructure (Phase 6)
//!
//! These tests validate the channel-based Writer/Reader/Consumer architecture:
//!
//! | Test | Description | Phase 6 Task |
//! |------|-------------|--------------|
//! | `test_mutation_via_writer_consumer` | End-to-end insert via Writer → Consumer → Storage | 6.2 |
//! | `test_query_via_reader_pool` | Search via SearchReader → ProcessorConsumer pool | 6.3/6.4 |
//! | `test_concurrent_queries_mpmc` | 100 concurrent searches via MPMC channel | 6.6 |
//! | `test_subsystem_start_lifecycle` | Subsystem::start() returns working handles | 6.5 |
//! | `test_writer_flush_semantics` | Flush guarantees visibility | 6.2 |
//! | `test_channel_close_propagation` | Writer/Reader detect channel close | 6.2/6.4 |
//! | `test_concurrent_deletes_vs_searches` | Deletes + searches don't return deleted | CODEX |

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use motlie_db::vector::{
    create_search_reader_with_storage, create_writer, spawn_mutation_consumer_with_storage_autoreg,
    spawn_query_consumers_with_storage_autoreg, DeleteVector, Distance, EmbeddingBuilder,
    ExternalKey, InsertVector, MutationRunnable, ReaderConfig, Runnable, SearchKNN, Storage,
    Subsystem, WriterConfig,
};
use motlie_db::Id;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use tempfile::TempDir;

const DIM: usize = 128;
const K: usize = 10;

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate normalized vectors.
fn generate_vectors(dim: usize, count: usize, seed: u64) -> Vec<Vec<f32>> {
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

// ============================================================================
// Test: Mutation via Writer → Consumer → Storage (Task 6.2)
// ============================================================================

/// Validates end-to-end mutation flow:
/// 1. Mutation sent via Writer
/// 2. Consumer receives and processes via Processor
/// 3. Data persisted to RocksDB and searchable
#[tokio::test]
async fn test_mutation_via_writer_consumer() {
    let temp_dir = TempDir::new().expect("temp dir");
    let mut storage = Storage::readwrite(temp_dir.path());
    storage.ready().expect("storage ready");
    let storage = Arc::new(storage);

    let registry = storage.cache().clone();
    let txn_db = storage.transaction_db().expect("txn_db");

    // Register embedding
    let embedding = registry
        .register(
            EmbeddingBuilder::new("test-channel", DIM as u32, Distance::Cosine),
            &txn_db,
        )
        .expect("register");

    // Create Writer with Consumer
    let (writer, writer_rx) = create_writer(WriterConfig::default());
    let _handle = spawn_mutation_consumer_with_storage_autoreg(
        writer_rx,
        WriterConfig::default(),
        storage.clone(),
    );

    // Create SearchReader for verification
    let (search_reader, reader_rx) =
        create_search_reader_with_storage(ReaderConfig::default(), storage.clone());
    let _reader_handles = spawn_query_consumers_with_storage_autoreg(
        reader_rx,
        ReaderConfig::default(),
        storage.clone(),
        2,
    );

    let timeout = Duration::from_secs(5);

    // Insert vectors via Writer
    let vectors = generate_vectors(DIM, 100, 42);
    let mut ids: Vec<Id> = Vec::new();

    for vector in &vectors {
        let id = Id::new();
        ids.push(id);

        InsertVector::new(&embedding, ExternalKey::NodeId(id), vector.clone())
            .immediate()
            .run(&writer)
            .await
            .expect("insert via writer");
    }

    // Flush to ensure visibility
    writer.flush().await.expect("flush");

    // Verify: Search should find inserted vectors
    let query = vectors[0].clone();
    let results = SearchKNN::new(&embedding, query, K)
        .with_ef(50)
        .exact()
        .run(&search_reader, timeout)
        .await
        .expect("search");

    // The query vector should match itself (index 0)
    assert!(!results.is_empty(), "Should find results after insert");
    assert_eq!(
        results[0].node_id().expect("expected NodeId"), ids[0],
        "First result should be the query vector itself"
    );
}

// ============================================================================
// Test: Query via SearchReader → ProcessorConsumer Pool (Task 6.3/6.4)
// ============================================================================

/// Validates query flow through MPMC channel:
/// 1. Query sent via SearchReader
/// 2. Any of N ProcessorConsumers picks it up (MPMC)
/// 3. Result returned via oneshot channel
#[tokio::test]
async fn test_query_via_reader_pool() {
    let temp_dir = TempDir::new().expect("temp dir");
    let mut storage = Storage::readwrite(temp_dir.path());
    storage.ready().expect("storage ready");
    let storage = Arc::new(storage);

    let registry = storage.cache().clone();
    let txn_db = storage.transaction_db().expect("txn_db");

    let embedding = registry
        .register(
            EmbeddingBuilder::new("test-reader-pool", DIM as u32, Distance::Cosine),
            &txn_db,
        )
        .expect("register");

    // Create Writer
    let (writer, writer_rx) = create_writer(WriterConfig::default());
    let _writer_handle = spawn_mutation_consumer_with_storage_autoreg(
        writer_rx,
        WriterConfig::default(),
        storage.clone(),
    );

    // Create SearchReader with 4-worker pool (tests MPMC distribution)
    let num_workers = 4;
    let (search_reader, reader_rx) =
        create_search_reader_with_storage(ReaderConfig::default(), storage.clone());
    let _reader_handles = spawn_query_consumers_with_storage_autoreg(
        reader_rx,
        ReaderConfig::default(),
        storage.clone(),
        num_workers,
    );

    let timeout = Duration::from_secs(5);

    // Insert test data
    let vectors = generate_vectors(DIM, 50, 99);
    for vector in &vectors {
        InsertVector::new(&embedding, ExternalKey::NodeId(Id::new()), vector.clone())
            .immediate()
            .run(&writer)
            .await
            .expect("insert");
    }
    writer.flush().await.expect("flush");

    // Send multiple queries - they should be distributed across workers
    for i in 0..10 {
        let query = vectors[i % vectors.len()].clone();
        let results = SearchKNN::new(&embedding, query, K)
            .with_ef(50)
            .exact()
            .run(&search_reader, timeout)
            .await
            .expect("search via pool");

        assert!(!results.is_empty(), "Query {} should return results", i);
    }
}

// ============================================================================
// Test: Concurrent Queries via MPMC (Task 6.6)
// ============================================================================

/// Stress test: 100 concurrent searches via MPMC channel.
/// Validates that multiple queries can be processed in parallel.
#[tokio::test]
async fn test_concurrent_queries_mpmc() {
    let temp_dir = TempDir::new().expect("temp dir");
    let mut storage = Storage::readwrite(temp_dir.path());
    storage.ready().expect("storage ready");
    let storage = Arc::new(storage);

    let registry = storage.cache().clone();
    let txn_db = storage.transaction_db().expect("txn_db");

    let embedding = registry
        .register(
            EmbeddingBuilder::new("test-concurrent", DIM as u32, Distance::Cosine),
            &txn_db,
        )
        .expect("register");

    // Create Writer
    let (writer, writer_rx) = create_writer(WriterConfig::default());
    let _writer_handle = spawn_mutation_consumer_with_storage_autoreg(
        writer_rx,
        WriterConfig::default(),
        storage.clone(),
    );

    // Create SearchReader with 8-worker pool for concurrent queries
    let num_workers = 8;
    let (search_reader, reader_rx) =
        create_search_reader_with_storage(ReaderConfig::default(), storage.clone());
    let _reader_handles = spawn_query_consumers_with_storage_autoreg(
        reader_rx,
        ReaderConfig::default(),
        storage.clone(),
        num_workers,
    );

    let timeout = Duration::from_secs(10);

    // Insert vectors
    let vectors = generate_vectors(DIM, 200, 77);
    for vector in &vectors {
        InsertVector::new(&embedding, ExternalKey::NodeId(Id::new()), vector.clone())
            .immediate()
            .run(&writer)
            .await
            .expect("insert");
    }
    writer.flush().await.expect("flush");

    // Launch 100 concurrent queries
    let num_queries = 100;
    let success_count = Arc::new(AtomicUsize::new(0));
    let error_count = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();
    for i in 0..num_queries {
        let search_reader = search_reader.clone();
        let query = vectors[i % vectors.len()].clone();
        let embedding = embedding.clone();
        let success = success_count.clone();
        let errors = error_count.clone();

        handles.push(tokio::spawn(async move {
            let result = SearchKNN::new(&embedding, query, K)
                .with_ef(50)
                .exact()
                .run(&search_reader, timeout)
                .await;

            match result {
                Ok(results) if !results.is_empty() => {
                    success.fetch_add(1, Ordering::Relaxed);
                }
                Ok(_) => {
                    // Empty results count as error for this test
                    errors.fetch_add(1, Ordering::Relaxed);
                }
                Err(_) => {
                    errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        }));
    }

    // Wait for all queries
    for handle in handles {
        handle.await.expect("task join");
    }

    let successes = success_count.load(Ordering::Relaxed);
    let failures = error_count.load(Ordering::Relaxed);

    println!(
        "Concurrent queries: {} success, {} errors ({:.1}% success)",
        successes,
        failures,
        (successes as f64 / num_queries as f64) * 100.0
    );

    // All queries should succeed
    assert_eq!(
        successes, num_queries,
        "All {} concurrent queries should succeed",
        num_queries
    );
}

// ============================================================================
// Test: Subsystem::start() Lifecycle (Task 6.5)
// ============================================================================

/// Validates Subsystem::start() returns working Writer + SearchReader.
/// This is the recommended lifecycle management pattern.
#[tokio::test]
async fn test_subsystem_start_lifecycle() {
    let temp_dir = TempDir::new().expect("temp dir");
    let mut storage = Storage::readwrite(temp_dir.path());
    storage.ready().expect("storage ready");
    let storage = Arc::new(storage);

    // Create subsystem and start
    let subsystem = Subsystem::new();
    let registry = subsystem.cache().clone();

    // Pre-warm manually (normally done by SubsystemProvider::on_ready)
    // In this test we skip that since we use standalone storage

    let (writer, search_reader) = subsystem.start(
        storage.clone(),
        WriterConfig::default(),
        ReaderConfig::default(),
        4, // 4 query workers
    );

    // Verify Writer is active
    assert!(!writer.is_closed(), "Writer should be active after start");

    // Verify SearchReader is active
    assert!(
        !search_reader.is_closed(),
        "SearchReader should be active after start"
    );

    let timeout = Duration::from_secs(5);

    // Register embedding and test end-to-end
    let txn_db = storage.transaction_db().expect("txn_db");
    let embedding = registry
        .register(
            EmbeddingBuilder::new("test-subsystem", DIM as u32, Distance::Cosine),
            &txn_db,
        )
        .expect("register");

    // Insert via Writer
    let vectors = generate_vectors(DIM, 10, 33);
    for vector in &vectors {
        InsertVector::new(&embedding, ExternalKey::NodeId(Id::new()), vector.clone())
            .immediate()
            .run(&writer)
            .await
            .expect("insert via subsystem writer");
    }
    writer.flush().await.expect("flush");

    // Search via SearchReader
    let results = SearchKNN::new(&embedding, vectors[0].clone(), K)
        .with_ef(50)
        .exact()
        .run(&search_reader, timeout)
        .await
        .expect("search via subsystem reader");

    assert!(!results.is_empty(), "Should find results via subsystem");
}

// ============================================================================
// Test: Writer Flush Semantics (Task 6.2)
// ============================================================================

/// Validates that flush() guarantees data visibility.
/// After flush returns, data must be searchable.
#[tokio::test]
async fn test_writer_flush_semantics() {
    let temp_dir = TempDir::new().expect("temp dir");
    let mut storage = Storage::readwrite(temp_dir.path());
    storage.ready().expect("storage ready");
    let storage = Arc::new(storage);

    let registry = storage.cache().clone();
    let txn_db = storage.transaction_db().expect("txn_db");

    let embedding = registry
        .register(
            EmbeddingBuilder::new("test-flush", DIM as u32, Distance::Cosine),
            &txn_db,
        )
        .expect("register");

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

    let timeout = Duration::from_secs(5);
    let vectors = generate_vectors(DIM, 1, 11);
    let id = Id::new();

    // Insert without flush
    InsertVector::new(&embedding, ExternalKey::NodeId(id), vectors[0].clone())
        .immediate()
        .run(&writer)
        .await
        .expect("insert");

    // Flush and verify immediate visibility
    writer.flush().await.expect("flush");

    // Search immediately after flush - should find the vector
    let results = SearchKNN::new(&embedding, vectors[0].clone(), K)
        .with_ef(50)
        .exact()
        .run(&search_reader, timeout)
        .await
        .expect("search after flush");

    assert!(!results.is_empty(), "Vector should be visible after flush");
    assert_eq!(results[0].node_id().expect("expected NodeId"), id, "Should find the inserted vector");
}

// ============================================================================
// Test: Channel Close Propagation (Task 6.2/6.4)
// ============================================================================

/// Validates that Writer and SearchReader detect channel close.
#[tokio::test]
async fn test_channel_close_propagation() {
    let temp_dir = TempDir::new().expect("temp dir");
    let mut storage = Storage::readwrite(temp_dir.path());
    storage.ready().expect("storage ready");
    let storage = Arc::new(storage);

    // Create Writer but don't spawn consumer (simulates consumer shutdown)
    let (writer, receiver) = create_writer(WriterConfig::default());

    assert!(!writer.is_closed(), "Writer should be active initially");

    // Drop receiver (simulates consumer shutdown)
    drop(receiver);

    // Writer should detect channel close
    // Note: is_closed() might take a moment to reflect
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(
        writer.is_closed(),
        "Writer should detect closed channel after receiver drop"
    );

    // Same test for SearchReader
    let (search_reader, query_receiver) =
        create_search_reader_with_storage(ReaderConfig::default(), storage.clone());

    assert!(
        !search_reader.is_closed(),
        "SearchReader should be active initially"
    );

    drop(query_receiver);

    tokio::time::sleep(Duration::from_millis(10)).await;
    assert!(
        search_reader.is_closed(),
        "SearchReader should detect closed channel after receiver drop"
    );
}

// ============================================================================
// Test: Concurrent Deletes vs Searches (CODEX Review)
// ============================================================================

/// Validates that concurrent deletes and searches work correctly:
/// 1. Deleted vectors are not returned in search results (tombstone filtering)
/// 2. No panics or data corruption under concurrent delete/search load
/// 3. Searches complete successfully even while deletes are in progress
#[tokio::test]
async fn test_concurrent_deletes_vs_searches() {
    let temp_dir = TempDir::new().expect("temp dir");
    let mut storage = Storage::readwrite(temp_dir.path());
    storage.ready().expect("storage ready");
    let storage = Arc::new(storage);

    let registry = storage.cache().clone();
    let txn_db = storage.transaction_db().expect("txn_db");

    let embedding = registry
        .register(
            EmbeddingBuilder::new("test-delete-search", DIM as u32, Distance::Cosine),
            &txn_db,
        )
        .expect("register");

    // Create Writer with Consumer
    let (writer, writer_rx) = create_writer(WriterConfig::default());
    let _writer_handle = spawn_mutation_consumer_with_storage_autoreg(
        writer_rx,
        WriterConfig::default(),
        storage.clone(),
    );

    // Create SearchReader with 4-worker pool
    let (search_reader, reader_rx) =
        create_search_reader_with_storage(ReaderConfig::default(), storage.clone());
    let _reader_handles = spawn_query_consumers_with_storage_autoreg(
        reader_rx,
        ReaderConfig::default(),
        storage.clone(),
        4,
    );

    let timeout = Duration::from_secs(5);

    // Insert 200 vectors - we'll delete half while searching
    let vectors = generate_vectors(DIM, 200, 123);
    let mut ids: Vec<Id> = Vec::new();

    for vector in &vectors {
        let id = Id::new();
        ids.push(id);

        InsertVector::new(&embedding, ExternalKey::NodeId(id), vector.clone())
            .immediate()
            .run(&writer)
            .await
            .expect("insert");
    }
    writer.flush().await.expect("flush after inserts");

    // Verify all vectors are searchable before deletion
    let pre_delete_results = SearchKNN::new(&embedding, vectors[0].clone(), 200)
        .with_ef(200)
        .exact()
        .run(&search_reader, timeout)
        .await
        .expect("pre-delete search");
    assert_eq!(
        pre_delete_results.len(),
        200,
        "All 200 vectors should be searchable before deletion"
    );

    // Track which IDs we'll delete (first 100)
    let ids_to_delete: HashSet<Id> = ids[..100].iter().copied().collect();
    let ids_to_keep: HashSet<Id> = ids[100..].iter().copied().collect();

    // Shared state for concurrent operations
    let stop_flag = Arc::new(AtomicBool::new(false));
    let deleted_count = Arc::new(AtomicUsize::new(0));
    let search_count = Arc::new(AtomicUsize::new(0));
    let found_deleted_count = Arc::new(AtomicUsize::new(0));

    // Spawn delete task - deletes vectors one by one
    let delete_writer = writer.clone();
    let delete_embedding = embedding.clone();
    let delete_ids: Vec<Id> = ids[..100].to_vec();
    let deleted_counter = Arc::clone(&deleted_count);
    let delete_handle = tokio::spawn(async move {
        for id in delete_ids {
            DeleteVector::new(&delete_embedding, ExternalKey::NodeId(id))
                .run(&delete_writer)
                .await
                .expect("delete");
            deleted_counter.fetch_add(1, Ordering::Relaxed);
            // Small delay to interleave with searches
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        delete_writer.flush().await.expect("flush deletes");
    });

    // Spawn multiple search tasks that run concurrently with deletes
    let mut search_handles = Vec::new();
    for searcher_id in 0..4 {
        let search_reader = search_reader.clone();
        let embedding = embedding.clone();
        let vectors = vectors.clone();
        let stop = Arc::clone(&stop_flag);
        let search_counter = Arc::clone(&search_count);
        let found_deleted = Arc::clone(&found_deleted_count);
        let deleted_ids = ids_to_delete.clone();

        search_handles.push(tokio::spawn(async move {
            let mut local_searches = 0u64;
            while !stop.load(Ordering::Relaxed) && local_searches < 50 {
                // Search using different query vectors
                let query_idx = (searcher_id * 13 + local_searches as usize) % vectors.len();
                let query = vectors[query_idx].clone();

                if let Ok(results) = SearchKNN::new(&embedding, query, K)
                    .with_ef(50)
                    .exact()
                    .run(&search_reader, timeout)
                    .await
                {
                    search_counter.fetch_add(1, Ordering::Relaxed);
                    local_searches += 1;

                    // Check if any deleted IDs appear in results
                    // Note: During the delete window, deleted IDs may still appear
                    // until the delete is fully committed and flushed
                    for result in &results {
                        if deleted_ids.contains(&result.node_id().expect("expected NodeId")) {
                            found_deleted.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        }));
    }

    // Wait for deletes to complete
    delete_handle.await.expect("delete task");

    // Signal searchers to stop
    stop_flag.store(true, Ordering::Relaxed);

    // Wait for all searchers
    for handle in search_handles {
        handle.await.expect("search task");
    }

    // Final flush to ensure all deletes are committed
    writer.flush().await.expect("final flush");

    // Wait a moment for tombstones to be visible
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Final verification: search should NOT return deleted IDs
    let post_delete_results = SearchKNN::new(&embedding, vectors[0].clone(), 200)
        .with_ef(200)
        .exact()
        .run(&search_reader, timeout)
        .await
        .expect("post-delete search");

    // Count how many deleted IDs are in final results
    let deleted_in_final: Vec<Id> = post_delete_results
        .iter()
        .filter_map(|r| r.node_id())
        .filter(|id| ids_to_delete.contains(id))
        .collect();

    let kept_in_final: Vec<Id> = post_delete_results
        .iter()
        .filter_map(|r| r.node_id())
        .filter(|id| ids_to_keep.contains(id))
        .collect();

    println!("\n=== test_concurrent_deletes_vs_searches Results ===");
    println!("Deleted: {} vectors", deleted_count.load(Ordering::Relaxed));
    println!(
        "Searches completed: {}",
        search_count.load(Ordering::Relaxed)
    );
    println!(
        "Found deleted during concurrent phase: {}",
        found_deleted_count.load(Ordering::Relaxed)
    );
    println!("Post-delete results: {} total", post_delete_results.len());
    println!("  - Deleted IDs found: {}", deleted_in_final.len());
    println!("  - Kept IDs found: {}", kept_in_final.len());

    // Assertions
    assert_eq!(
        deleted_count.load(Ordering::Relaxed),
        100,
        "Should have deleted 100 vectors"
    );

    assert!(
        search_count.load(Ordering::Relaxed) > 0,
        "Should have completed some searches"
    );

    // Critical assertion: After all deletes are committed and flushed,
    // deleted vectors should NOT appear in search results
    assert_eq!(
        deleted_in_final.len(),
        0,
        "Deleted vectors should not appear in final search results (tombstone filtering)"
    );

    // The kept vectors should still be searchable
    assert_eq!(
        kept_in_final.len(),
        100,
        "All 100 non-deleted vectors should still be searchable"
    );
}
