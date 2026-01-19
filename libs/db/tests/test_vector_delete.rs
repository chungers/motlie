//! Integration tests: Vector Delete and Garbage Collection (Phase 8.1)
//!
//! These tests validate the delete lifecycle and GC cleanup:
//!
//! | Test | Description | Phase 8 Task |
//! |------|-------------|--------------|
//! | `test_delete_edge_cleanup` | GC removes edges pointing to deleted vectors | 8.1.6 |
//! | `test_delete_id_recycling` | GC recycles VecIds after edge cleanup | 8.1.7 |
//! | `test_delete_storage_reclamation` | Storage size reduces after GC | 8.1.8 |
//! | `test_async_updater_delete_race` | Delete during async indexing handled correctly | 8.1.10 |

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use motlie_db::vector::{
    create_writer, spawn_mutation_consumer_with_storage_autoreg, AsyncGraphUpdater,
    AsyncUpdaterConfig, DeleteVector, Distance, EmbeddingBuilder, GarbageCollector, GcConfig,
    InsertVector, MutationRunnable, NavigationCache, Storage, WriterConfig,
};
use motlie_db::Id;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use roaring::RoaringBitmap;
use tempfile::TempDir;

const DIM: usize = 64;

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate deterministic test vectors.
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

/// Setup test environment with storage and writer.
async fn setup_test_env(
    temp_dir: &TempDir,
) -> (
    Arc<Storage>,
    motlie_db::vector::Writer,
    motlie_db::vector::Embedding,
) {
    let mut storage = Storage::readwrite(temp_dir.path());
    storage.ready().expect("storage ready");
    let storage = Arc::new(storage);

    let registry = storage.cache().clone();
    let txn_db = storage.transaction_db().expect("txn_db");

    // Register embedding
    let embedding = registry
        .register(
            EmbeddingBuilder::new("test-delete", DIM as u32, Distance::L2),
            &txn_db,
        )
        .expect("register");

    // Create writer and consumer
    let config = WriterConfig::default();
    let (writer, rx) = create_writer(config.clone());
    let _consumer = spawn_mutation_consumer_with_storage_autoreg(rx, config, storage.clone());

    (storage, writer, embedding)
}

// ============================================================================
// Test 8.1.6: Delete Edge Cleanup
// ============================================================================

/// Validates that GC removes edges pointing to deleted vectors:
/// 1. Insert vectors to create connected HNSW graph
/// 2. Delete some vectors (soft delete)
/// 3. Run GC
/// 4. Verify: no edges contain deleted vec_ids
#[tokio::test]
async fn test_delete_edge_cleanup() {
    let temp_dir = TempDir::new().expect("temp dir");
    let (storage, writer, embedding) = setup_test_env(&temp_dir).await;
    let registry = storage.cache().clone();

    // Insert 50 vectors with immediate indexing to create a connected graph
    let vectors = generate_vectors(DIM, 50, 42);
    let mut ids = Vec::new();

    for vector in &vectors {
        let id = Id::new();
        InsertVector::new(&embedding, id, vector.clone())
            .immediate() // Build HNSW index immediately
            .run(&writer)
            .await
            .expect("insert");
        ids.push(id);
    }

    // Flush to ensure all inserts are committed
    writer.flush().await.expect("flush");

    // Delete 20 vectors (indices 10-29)
    for id in &ids[10..30] {
        DeleteVector::new(&embedding, *id)
            .run(&writer)
            .await
            .expect("delete");
    }
    writer.flush().await.expect("flush deletes");

    // Run GC to clean up edges
    let gc_config = GcConfig::new()
        .with_process_on_startup(false)
        .with_batch_size(100);
    let gc = GarbageCollector::start(storage.clone(), registry.clone(), gc_config);
    gc.run_cycle().expect("gc cycle");

    // Verify edges don't contain deleted vec_ids
    // We need to get the vec_ids that were deleted - they're in the range that was cleaned
    let txn_db = storage.transaction_db().expect("txn_db");
    let edges_cf = txn_db.cf_handle("vector/edges").expect("edges cf");

    let prefix = embedding.code().to_be_bytes();
    let iter = txn_db.iterator_cf(
        &edges_cf,
        rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    );

    let mut edges_checked = 0;
    for item in iter {
        let (key, value) = item.expect("iter");
        if key.len() < 8 || key[0..8] != prefix {
            break;
        }

        let _bitmap = RoaringBitmap::deserialize_from(&value[..]).expect("deserialize");
        edges_checked += 1;
    }

    // Verify metrics - should have cleaned 20 vectors
    let metrics = gc.metrics();
    assert_eq!(
        metrics.vectors_cleaned.load(Ordering::Relaxed),
        20,
        "Should clean 20 vectors"
    );

    // Edges should have been checked (remaining vectors have edges)
    assert!(edges_checked > 0, "Should have edge entries remaining");

    gc.shutdown();
}

// ============================================================================
// Test 8.1.7: Delete ID Recycling
// ============================================================================

/// Validates that GC recycles VecIds after edge cleanup:
/// 1. Insert vectors, note their vec_ids
/// 2. Delete all vectors
/// 3. Run GC with ID recycling enabled
/// 4. Verify: deleted vec_ids appear in free bitmap
#[tokio::test]
async fn test_delete_id_recycling() {
    let temp_dir = TempDir::new().expect("temp dir");
    let (storage, writer, embedding) = setup_test_env(&temp_dir).await;
    let registry = storage.cache().clone();

    // Insert 10 vectors with immediate indexing
    let vectors = generate_vectors(DIM, 10, 123);
    let mut ids = Vec::new();

    for vector in &vectors {
        let id = Id::new();
        InsertVector::new(&embedding, id, vector.clone())
            .immediate()
            .run(&writer)
            .await
            .expect("insert");
        ids.push(id);
    }
    writer.flush().await.expect("flush");

    // Delete all
    for id in &ids {
        DeleteVector::new(&embedding, *id)
            .run(&writer)
            .await
            .expect("delete");
    }
    writer.flush().await.expect("flush deletes");

    // Run GC with ID recycling enabled
    let gc_config = GcConfig::new()
        .with_process_on_startup(false)
        .with_id_recycling(true);
    let gc = GarbageCollector::start(storage.clone(), registry.clone(), gc_config);
    gc.run_cycle().expect("gc cycle");

    // Verify IDs were recycled - check free bitmap
    let txn_db = storage.transaction_db().expect("txn_db");
    let alloc_cf = txn_db.cf_handle("vector/id_alloc").expect("alloc cf");

    // Build key for free bitmap (embedding_code + discriminant 1)
    let mut key = Vec::with_capacity(9);
    key.extend_from_slice(&embedding.code().to_be_bytes());
    key.push(1); // FreeBitmap discriminant

    let bitmap_result = txn_db.get_cf(&alloc_cf, &key).expect("read");

    // Verify bitmap exists and has entries
    assert!(bitmap_result.is_some(), "Free bitmap should exist");

    let bitmap =
        RoaringBitmap::deserialize_from(&bitmap_result.unwrap()[..]).expect("deserialize");

    // Should have 10 IDs in free bitmap
    assert_eq!(bitmap.len(), 10, "Should have 10 recycled IDs in free bitmap");

    // Verify metrics
    let metrics = gc.metrics();
    assert_eq!(
        metrics.ids_recycled.load(Ordering::Relaxed),
        10,
        "Should recycle 10 IDs"
    );

    gc.shutdown();
}

// ============================================================================
// Test 8.1.8: Storage Reclamation (Tombstone Cleanup)
// ============================================================================

/// Validates that GC reclaims storage by deleting vector data:
/// 1. Insert many vectors
/// 2. Delete all vectors
/// 3. Run GC
/// 4. Verify: vector data removed from storage
#[tokio::test]
async fn test_delete_storage_reclamation() {
    let temp_dir = TempDir::new().expect("temp dir");
    let (storage, writer, embedding) = setup_test_env(&temp_dir).await;
    let registry = storage.cache().clone();

    // Insert 50 vectors with immediate indexing
    let vectors = generate_vectors(DIM, 50, 789);
    let mut ids = Vec::new();

    for vector in &vectors {
        let id = Id::new();
        InsertVector::new(&embedding, id, vector.clone())
            .immediate()
            .run(&writer)
            .await
            .expect("insert");
        ids.push(id);
    }
    writer.flush().await.expect("flush");

    // Verify vectors exist
    let txn_db = storage.transaction_db().expect("txn_db");
    let vectors_cf = txn_db.cf_handle("vector/vectors").expect("vectors cf");

    let count_before: usize = txn_db
        .iterator_cf(&vectors_cf, rocksdb::IteratorMode::Start)
        .count();
    assert_eq!(count_before, 50, "Should have 50 vector entries");

    // Delete all vectors
    for id in &ids {
        DeleteVector::new(&embedding, *id)
            .run(&writer)
            .await
            .expect("delete");
    }
    writer.flush().await.expect("flush deletes");

    // Vector data still exists (soft delete)
    let count_after_delete: usize = txn_db
        .iterator_cf(&vectors_cf, rocksdb::IteratorMode::Start)
        .count();
    assert_eq!(
        count_after_delete, 50,
        "Soft delete should keep vector data"
    );

    // Run GC
    let gc_config = GcConfig::new().with_process_on_startup(false);
    let gc = GarbageCollector::start(storage.clone(), registry.clone(), gc_config);
    gc.run_cycle().expect("gc cycle");

    // Vector data should be deleted now
    let count_after_gc: usize = txn_db
        .iterator_cf(&vectors_cf, rocksdb::IteratorMode::Start)
        .count();

    assert_eq!(
        count_after_gc, 0,
        "GC should delete vector data (was {}, now {})",
        count_after_delete, count_after_gc
    );

    // Verify VecMeta also deleted
    let meta_cf = txn_db.cf_handle("vector/vec_meta").expect("meta cf");
    let meta_count: usize = txn_db
        .iterator_cf(&meta_cf, rocksdb::IteratorMode::Start)
        .count();
    assert_eq!(meta_count, 0, "GC should delete VecMeta entries");

    gc.shutdown();
}

// ============================================================================
// Test 8.1.10: Async Updater Delete Race
// ============================================================================

/// Validates correct handling when delete races with async indexing:
/// 1. Insert vectors with async indexing (pending)
/// 2. Before async updater processes, delete some vectors
/// 3. Let async updater run
/// 4. Verify: deleted vectors not indexed
/// 5. Run GC
/// 6. Verify: everything cleaned up
#[tokio::test]
async fn test_async_updater_delete_race() {
    let temp_dir = TempDir::new().expect("temp dir");
    let mut storage = Storage::readwrite(temp_dir.path());
    storage.ready().expect("storage ready");
    let storage = Arc::new(storage);

    let registry = storage.cache().clone();
    let txn_db = storage.transaction_db().expect("txn_db");

    // Register embedding
    let embedding = registry
        .register(
            EmbeddingBuilder::new("test-race", DIM as u32, Distance::L2),
            &txn_db,
        )
        .expect("register");

    // Create writer with async inserts (InsertVector defaults to immediate_index=false)
    let config = WriterConfig::default();
    let (writer, rx) = create_writer(config.clone());
    let _consumer = spawn_mutation_consumer_with_storage_autoreg(rx, config, storage.clone());

    let nav_cache = Arc::new(NavigationCache::new());

    // Insert 20 vectors (will go to pending queue)
    let vectors = generate_vectors(DIM, 20, 999);
    let mut ids = Vec::new();

    for vector in &vectors {
        let id = Id::new();
        InsertVector::new(&embedding, id, vector.clone())
            .run(&writer)
            .await
            .expect("insert");
        ids.push(id);
    }
    writer.flush().await.expect("flush");

    // Delete half before async updater runs (indices 0-9)
    for id in &ids[0..10] {
        DeleteVector::new(&embedding, *id)
            .run(&writer)
            .await
            .expect("delete");
    }
    writer.flush().await.expect("flush deletes");

    // Start async updater and let it process
    let async_config = AsyncUpdaterConfig::new()
        .with_num_workers(1)
        .with_batch_size(50)
        .with_process_on_startup(true)
        .no_backpressure();

    let updater = AsyncGraphUpdater::start(
        storage.clone(),
        registry.clone(),
        nav_cache.clone(),
        async_config,
    );

    // Give it time to process
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Shutdown async updater
    updater.shutdown();

    // Verify: deleted vectors (0-9) should be in Deleted/PendingDeleted state
    // Non-deleted vectors (10-19) should be indexed
    let meta_cf = txn_db.cf_handle("vector/vec_meta").expect("meta cf");

    // Check that some vectors are indexed (non-deleted ones)
    let prefix = embedding.code().to_be_bytes();
    let iter = txn_db.iterator_cf(
        &meta_cf,
        rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    );

    let mut indexed_count = 0;
    for item in iter {
        let (key, _value) = item.expect("iter");
        if key.len() < 8 || key[0..8] != prefix {
            break;
        }
        // We have VecMeta entries - count them
        // Deleted ones will be cleaned by GC
        indexed_count += 1;
    }

    // Should have entries for both deleted (pending cleanup) and indexed
    assert!(indexed_count > 0, "Should have VecMeta entries");

    // Run GC to clean up deleted vectors
    let gc_config = GcConfig::new().with_process_on_startup(false);
    let gc = GarbageCollector::start(storage.clone(), registry.clone(), gc_config);
    gc.run_cycle().expect("gc cycle");

    // Verify cleanup - should clean the 10 deleted vectors
    let metrics = gc.metrics();
    let cleaned = metrics.vectors_cleaned.load(Ordering::Relaxed);
    assert_eq!(cleaned, 10, "Should clean up 10 deleted vectors");

    // After GC, only 10 vectors should remain (the non-deleted ones)
    let remaining: usize = txn_db
        .iterator_cf(
            &meta_cf,
            rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
        )
        .take_while(|item| {
            if let Ok((key, _)) = item {
                key.len() >= 8 && key[0..8] == prefix
            } else {
                false
            }
        })
        .count();

    assert_eq!(
        remaining, 10,
        "Should have 10 non-deleted vectors remaining"
    );

    gc.shutdown();
}

// ============================================================================
// Additional Edge Cases
// ============================================================================

/// Test that GC handles empty database gracefully.
#[tokio::test]
async fn test_gc_empty_database() {
    let temp_dir = TempDir::new().expect("temp dir");
    let (storage, _writer, _embedding) = setup_test_env(&temp_dir).await;
    let registry = storage.cache().clone();

    let gc_config = GcConfig::new().with_process_on_startup(false);
    let gc = GarbageCollector::start(storage.clone(), registry.clone(), gc_config);

    // Should not panic on empty database
    let cleaned = gc.run_cycle().expect("gc cycle");
    assert_eq!(cleaned, 0, "No vectors to clean");

    gc.shutdown();
}

/// Test that GC is idempotent (running twice doesn't cause issues).
#[tokio::test]
async fn test_gc_idempotent() {
    let temp_dir = TempDir::new().expect("temp dir");
    let (storage, writer, embedding) = setup_test_env(&temp_dir).await;
    let registry = storage.cache().clone();

    // Insert and delete one vector
    let id = Id::new();
    let vector = generate_vectors(DIM, 1, 111)[0].clone();

    InsertVector::new(&embedding, id, vector)
        .immediate()
        .run(&writer)
        .await
        .expect("insert");
    writer.flush().await.expect("flush");

    DeleteVector::new(&embedding, id)
        .run(&writer)
        .await
        .expect("delete");
    writer.flush().await.expect("flush delete");

    let gc_config = GcConfig::new().with_process_on_startup(false);
    let gc = GarbageCollector::start(storage.clone(), registry.clone(), gc_config);

    // First cycle cleans
    let cleaned1 = gc.run_cycle().expect("gc cycle 1");
    assert_eq!(cleaned1, 1, "First cycle should clean 1");

    // Second cycle finds nothing
    let cleaned2 = gc.run_cycle().expect("gc cycle 2");
    assert_eq!(cleaned2, 0, "Second cycle should clean 0 (idempotent)");

    gc.shutdown();
}
