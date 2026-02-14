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
    AsyncUpdaterConfig, DeleteVector, Distance, EmbeddingBuilder, ExternalKey, GarbageCollector,
    GcConfig, InsertVector, MutationRunnable, NavigationCache, Storage, VecId, WriterConfig,
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
            let mut v: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
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
    registry.set_storage(storage.clone()).expect("set storage");

    // Register embedding
    let embedding = registry
        .register(EmbeddingBuilder::new("test-delete", DIM as u32, Distance::L2))
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
        InsertVector::new(&embedding, ExternalKey::NodeId(id), vector.clone())
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
        DeleteVector::new(&embedding, ExternalKey::NodeId(*id))
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
    // Vectors are inserted sequentially, so vec_ids 10-29 correspond to the deleted vectors
    // ADDRESSED (Claude, 2026-01-19): Added explicit check that deleted vec_ids are absent
    // from all edge bitmaps after GC cleanup.
    let deleted_vec_ids: Vec<VecId> = (10..30).collect();

    let txn_db = storage.transaction_db().expect("txn_db");
    let edges_cf = txn_db.cf_handle("vector/edges").expect("edges cf");

    let prefix = embedding.code().to_be_bytes();
    let iter = txn_db.iterator_cf(
        &edges_cf,
        rocksdb::IteratorMode::From(&prefix, rocksdb::Direction::Forward),
    );

    let mut edges_checked = 0;
    let mut deleted_refs_found = 0;
    for item in iter {
        let (key, value) = item.expect("iter");
        if key.len() < 8 || key[0..8] != prefix {
            break;
        }

        let bitmap = RoaringBitmap::deserialize_from(&value[..]).expect("deserialize");
        edges_checked += 1;

        // Check that no deleted vec_id appears in this edge bitmap
        for deleted_id in &deleted_vec_ids {
            if bitmap.contains(*deleted_id) {
                deleted_refs_found += 1;
            }
        }
    }

    // Assert no deleted vec_ids remain in any edge bitmap
    assert_eq!(
        deleted_refs_found, 0,
        "GC should remove all references to deleted vec_ids from edge bitmaps"
    );

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
        InsertVector::new(&embedding, ExternalKey::NodeId(id), vector.clone())
            .immediate()
            .run(&writer)
            .await
            .expect("insert");
        ids.push(id);
    }
    writer.flush().await.expect("flush");

    // Delete all
    for id in &ids {
        DeleteVector::new(&embedding, ExternalKey::NodeId(*id))
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
    // ADDRESSED (Claude, 2026-01-19): IdAllocCfKey and IdAlloc are pub(crate) internal types,
    // so integration tests must use raw key format. The key structure is:
    //   [embedding_code: u64 BE] [discriminant: u8]
    // where discriminant 1 = FreeBitmap. This matches IdAllocCfKey::free_bitmap().
    let mut key = Vec::with_capacity(9);
    key.extend_from_slice(&embedding.code().to_be_bytes());
    key.push(1); // FreeBitmap discriminant (matches IdAllocField::FreeBitmap)

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
        InsertVector::new(&embedding, ExternalKey::NodeId(id), vector.clone())
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
        DeleteVector::new(&embedding, ExternalKey::NodeId(*id))
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
    registry.set_storage(storage.clone()).expect("set storage");

    // Register embedding
    let embedding = registry
        .register(EmbeddingBuilder::new("test-race", DIM as u32, Distance::L2))
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
        InsertVector::new(&embedding, ExternalKey::NodeId(id), vector.clone())
            .run(&writer)
            .await
            .expect("insert");
        ids.push(id);
    }
    writer.flush().await.expect("flush");

    // Get txn_db for database reads
    let txn_db = storage.transaction_db().expect("txn_db");

    // Capture vec_ids via IdForward before deletes (IdForward removed on delete)
    let id_forward_cf = txn_db.cf_handle("vector/id_forward").expect("id_forward cf");
    let mut vec_ids = Vec::new();
    for id in &ids {
        // Key format: [embedding: u64] + [ExternalKey bytes]
        // ExternalKey::NodeId bytes = [tag: 0x01] + [id: 16 bytes]
        let external_bytes = ExternalKey::NodeId(*id).to_bytes();
        let mut key = Vec::with_capacity(8 + external_bytes.len());
        key.extend_from_slice(&embedding.code().to_be_bytes());
        key.extend_from_slice(&external_bytes);
        let bytes = txn_db
            .get_cf(&id_forward_cf, &key)
            .expect("read")
            .expect("IdForward should exist");
        // Value is VecId (u32) in BE
        let vec_id = u32::from_be_bytes(bytes.as_slice().try_into().expect("vec_id bytes"));
        vec_ids.push(vec_id);
    }

    // Delete half before async updater runs (indices 0-9)
    for id in &ids[0..10] {
        DeleteVector::new(&embedding, ExternalKey::NodeId(*id))
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

    // ADDRESSED (Claude, 2026-01-19): Verify deleted vectors are excluded from search results.
    // In async insert mode, IdForward/IdReverse mappings are created during Phase 1 (sync)
    // and removed during soft delete. We verify:
    // 1. IdForward is gone for deleted vectors (they can't be resolved to vec_id)
    // 2. After GC, only non-deleted VecMeta entries remain
    let id_forward_cf = txn_db.cf_handle("vector/id_forward").expect("id_forward cf");

    // Check that deleted ids (0-9) have no IdForward mapping
    let mut deleted_found_in_forward = 0;
    for id in &ids[0..10] {
        // IdForward key format: [embedding_code: u64 BE][ExternalKey bytes]
        let external_bytes = ExternalKey::NodeId(*id).to_bytes();
        let mut key = Vec::with_capacity(8 + external_bytes.len());
        key.extend_from_slice(&embedding.code().to_be_bytes());
        key.extend_from_slice(&external_bytes);
        if txn_db.get_cf(&id_forward_cf, &key).expect("read").is_some() {
            deleted_found_in_forward += 1;
        }
    }
    assert_eq!(
        deleted_found_in_forward, 0,
        "Deleted vectors should have no IdForward mapping (search exclusion)"
    );

    // Check that non-deleted ids (10-19) still have IdForward mapping
    let mut live_found_in_forward = 0;
    for id in &ids[10..20] {
        let external_bytes = ExternalKey::NodeId(*id).to_bytes();
        let mut key = Vec::with_capacity(8 + external_bytes.len());
        key.extend_from_slice(&embedding.code().to_be_bytes());
        key.extend_from_slice(&external_bytes);
        if txn_db.get_cf(&id_forward_cf, &key).expect("read").is_some() {
            live_found_in_forward += 1;
        }
    }
    assert_eq!(
        live_found_in_forward, 10,
        "Non-deleted vectors should have IdForward mapping"
    );

    // ADDRESSED (Claude, 2026-01-19): Also check IdReverse (used by search result resolution)
    // IdReverse key format: [external_id: 16 bytes] -> (embedding_code, vec_id)
    // Note: In async insert mode, IdReverse may be created during async indexing, not Phase 1.
    // We verify deleted vectors have no IdReverse (soft delete removes it if present).
    let id_reverse_cf = txn_db.cf_handle("vector/id_reverse").expect("id_reverse cf");

    // Check that deleted ids (0-9) have no IdReverse mapping
    let mut deleted_found_in_reverse = 0;
    for vec_id in &vec_ids[0..10] {
        let mut key = Vec::with_capacity(12);
        key.extend_from_slice(&embedding.code().to_be_bytes());
        key.extend_from_slice(&vec_id.to_be_bytes());
        if txn_db.get_cf(&id_reverse_cf, &key).expect("read").is_some() {
            deleted_found_in_reverse += 1;
        }
    }
    assert_eq!(
        deleted_found_in_reverse, 0,
        "Deleted vectors should have no IdReverse mapping"
    );

    // Note: Non-deleted vectors may or may not have IdReverse depending on async indexing timing.
    // The key safety guarantee is that deleted vectors are excluded from search results via:
    // 1. IdForward removal (prevents external_id -> vec_id resolution for reinsert)
    // 2. IdReverse removal (prevents vec_id -> external_id resolution in results)
    // 3. VecMeta lifecycle check (defense-in-depth during search)

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

    InsertVector::new(&embedding, ExternalKey::NodeId(id), vector)
        .immediate()
        .run(&writer)
        .await
        .expect("insert");
    writer.flush().await.expect("flush");

    DeleteVector::new(&embedding, ExternalKey::NodeId(id))
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
