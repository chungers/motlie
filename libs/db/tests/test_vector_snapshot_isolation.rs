#![cfg(feature = "benchmark")]
//! Snapshot Isolation Tests for Vector Subsystem (Task 8.2.1, 8.2.2).
//!
//! These tests validate RocksDB's snapshot isolation guarantees under concurrent access:
//!
//! | Test | Description | Validates |
//! |------|-------------|-----------|
//! | `test_snapshot_isolation_delete_during_search` | Snapshot count remains stable while Reader queries run during deletes | Isolation + end-to-end |
//! | `test_snapshot_isolation_insert_during_search` | Snapshot excludes concurrent inserts | Isolation |
//! | `test_transaction_conflict_resolution` | Pessimistic locking serializes conflicts | Conflict handling |
//! | `test_transaction_conflict_stress` | High-contention conflict handling | Stress |
//!
//! ## RocksDB Snapshot Isolation
//!
//! RocksDB provides snapshot isolation through:
//! 1. **Read snapshots**: `db.snapshot()` returns a point-in-time view
//! 2. **Pessimistic transactions**: TransactionDB serializes conflicting writes
//!
//! These tests verify that:
//! - Snapshot reads see consistent data under concurrent writes
//! - Concurrent modifications don't corrupt ongoing reads
//! - Transaction conflicts are properly detected and reported

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use motlie_db::vector::benchmark::ConcurrentMetrics;
use motlie_db::vector::hnsw;
use motlie_db::vector::schema::{EmbeddingCode, EmbeddingSpec, VectorCfKey, VectorCfValue, Vectors};
use motlie_db::vector::reader::{
    create_reader_with_storage, spawn_query_consumers_with_storage_autoreg, ReaderConfig,
};
use motlie_db::vector::{
    cache::NavigationCache, create_writer, spawn_mutation_consumer_with_storage_autoreg,
    DeleteVector, Distance, EmbeddingBuilder, ExternalKey, InsertVector, MutationRunnable, Storage,
    VectorElementType, WriterConfig,
};
use motlie_db::rocksdb::ColumnFamily;
use motlie_db::Id;
use rand::prelude::*;
use tempfile::TempDir;

const DIM: usize = 64;

/// Helper: Create test EmbeddingSpec for HNSW index
fn make_test_spec(dim: usize, m: usize, ef_construction: usize) -> EmbeddingSpec {
    EmbeddingSpec {
        model: "test".to_string(),
        dim: dim as u32,
        distance: Distance::L2,
        storage_type: VectorElementType::F32,
        hnsw_m: m as u16,
        hnsw_ef_construction: ef_construction as u16,
        rabitq_bits: 1,
        rabitq_seed: 42,
    }
}
const TEST_EMBEDDING: EmbeddingCode = 1;
const ID_REVERSE_CF: &str = "vector/id_reverse";

/// Helper: Create test storage
fn create_test_storage() -> (TempDir, Storage) {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let mut storage = Storage::readwrite(temp_dir.path());
    storage.ready().expect("Failed to initialize storage");
    (temp_dir, storage)
}

/// Helper: Generate deterministic vector from seed
fn seeded_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    (0..dim).map(|_| rng.random::<f32>()).collect()
}

/// Atomic insert result
#[derive(Debug)]
enum InsertResult {
    Success(hnsw::CacheUpdate),
    CommitFailed,
    HnswFailed,
}

/// Helper: Atomic vector + HNSW insert in a single transaction.
fn insert_vector_atomic(
    storage: &Storage,
    index: &hnsw::Index,
    embedding: EmbeddingCode,
    vec_id: u32,
    vector: &[f32],
) -> InsertResult {
    let Ok(txn_db) = storage.transaction_db() else {
        return InsertResult::CommitFailed;
    };
    let Some(cf) = txn_db.cf_handle(Vectors::CF_NAME) else {
        return InsertResult::CommitFailed;
    };

    let txn = txn_db.transaction();

    // Step 1: Write vector data to Vectors CF
    let key = VectorCfKey(embedding, vec_id);
    let value = VectorCfValue(vector.to_vec());
    if txn
        .put_cf(
            &cf,
            Vectors::key_to_bytes(&key),
            Vectors::value_to_bytes(&value),
        )
        .is_err()
    {
        return InsertResult::CommitFailed;
    }

    // Step 2: Insert into HNSW graph (same transaction)
    let cache_update = match hnsw::insert(index, &txn, &txn_db, storage, vec_id, vector) {
        Ok(update) => update,
        Err(_) => return InsertResult::HnswFailed,
    };

    // Step 3: Commit both operations atomically
    if txn.commit().is_ok() {
        InsertResult::Success(cache_update)
    } else {
        InsertResult::CommitFailed
    }
}

// ============================================================================
// Test 8.2.1: Snapshot Isolation Validation
// ============================================================================

/// Test: Snapshot isolation during concurrent deletes.
///
/// Validates that a search started with a snapshot sees consistent data
/// even when concurrent threads delete vectors.
///
/// 1. Insert 1000 vectors
/// 2. Start search with snapshot (captures point-in-time view)
/// 3. Concurrent: delete 500 vectors
/// 4. Search should see all 1000 (snapshot isolation)
/// 5. New search should see only ~500
#[test]
fn test_snapshot_isolation_delete_during_search() {
    let (_temp_dir, storage) = create_test_storage();
    let storage = Arc::new(storage);

    // Register embedding for Reader/Processor path
    let registry = storage.cache().clone();
    let txn_db = storage.transaction_db().expect("txn_db");
    let embedding = registry
        .register(
            EmbeddingBuilder::new("snapshot-test", DIM as u32, Distance::L2)
                .with_hnsw_m(16)
                .with_hnsw_ef_construction(100),
            &txn_db,
        )
        .expect("embedding register");

    let num_vectors = 1000u32;
    let mut inserts = Vec::with_capacity(num_vectors as usize);
    for vec_id in 0..num_vectors {
        let id = Id::new();
        let vector = seeded_vector(DIM, vec_id as u64);
        inserts.push((id, vector));
    }
    let ids: Vec<Id> = inserts.iter().map(|(id, _)| *id).collect();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");

    let (writer, writer_rx) = create_writer(WriterConfig {
        channel_buffer_size: 10_000,
    });
    let mutation_handle = {
        let _guard = runtime.enter();
        spawn_mutation_consumer_with_storage_autoreg(
            writer_rx,
            WriterConfig::default(),
            storage.clone(),
        )
    };

    println!("Inserting {} vectors...", num_vectors);
    runtime.block_on(async {
        for (id, vector) in inserts {
            InsertVector::new(&embedding, ExternalKey::NodeId(id), vector)
                .immediate()
                .run(&writer)
                .await
                .expect("insert");
        }
        writer.flush().await.expect("flush");
    });
    println!("Inserted {} vectors", num_vectors);

    let txn_db = storage.transaction_db().expect("txn_db");
    let snapshot = txn_db.snapshot();

    // Verify initial count via snapshot
    let reverse_cf = txn_db.cf_handle(ID_REVERSE_CF).expect("cf");
    let initial_count: usize = txn_db
        .iterator_cf(&reverse_cf, rocksdb::IteratorMode::Start)
        .count();
    assert_eq!(initial_count, num_vectors as usize, "Should have all vectors before delete");

    // Step 3: Concurrent delete of 500 vectors (ids 0-499)
    let delete_count = 500u32;
    let delete_ids = ids[..delete_count as usize].to_vec();
    let writer_clone = writer.clone();
    let embedding_clone = embedding.clone();
    let search_results = runtime.block_on(async {
        let delete_handle = tokio::spawn(async move {
            for id in delete_ids {
                DeleteVector::new(&embedding_clone, ExternalKey::NodeId(id))
                    .run(&writer_clone)
                    .await
                    .expect("delete");
            }
            writer_clone.flush().await.expect("flush");
            delete_count
        });

        let (search_reader, reader_rx) =
            create_reader_with_storage(ReaderConfig::default(), storage.clone());
        let query_handles = spawn_query_consumers_with_storage_autoreg(
            reader_rx,
            ReaderConfig::default(),
            storage.clone(),
            2,
        );

        let mut results = Vec::new();
        let timeout = std::time::Duration::from_secs(5);
        let query = seeded_vector(DIM, 4242);

        for _ in 0..5 {
            let result = search_reader
                .search_knn(&embedding, query.clone(), 10, 100, timeout)
                .await
                .expect("search");
            results.push(result);
        }

        drop(search_reader);
        for handle in query_handles {
            let _ = handle.await;
        }
        let deleted = delete_handle.await.expect("delete task");

        (results, deleted)
    });

    // Step 4: While deletes are happening, read using snapshot
    // Snapshot should see all 1000 vectors (point-in-time consistency)
    let snapshot_count: usize = {
        let iter = snapshot.iterator_cf(&reverse_cf, rocksdb::IteratorMode::Start);
        iter.count()
    };

    let (search_results, deleted) = search_results;
    println!("Deleted {} vectors", deleted);

    // Step 5: Verify snapshot isolation
    assert_eq!(
        snapshot_count, num_vectors as usize,
        "Snapshot should see all {} vectors (isolation), but saw {}",
        num_vectors, snapshot_count
    );

    // Step 6: New read (without snapshot) should see fewer vectors
    let current_count: usize = txn_db
        .iterator_cf(&reverse_cf, rocksdb::IteratorMode::Start)
        .count();

    assert!(
        current_count < num_vectors as usize,
        "Current view should see fewer vectors after delete: expected < {}, got {}",
        num_vectors, current_count
    );

    println!(
        "Snapshot isolation validated: snapshot saw {}, current sees {}",
        snapshot_count, current_count
    );

    for (idx, result) in search_results.iter().enumerate() {
        assert!(
            !result.is_empty(),
            "Reader result {} should not be empty during deletes",
            idx
        );
    }

    drop(writer);
    let _ = runtime.block_on(async { mutation_handle.await });
}

/// Test: Snapshot isolation during concurrent inserts.
///
/// Validates that a search started with a snapshot excludes vectors
/// inserted after the snapshot was taken.
#[test]
fn test_snapshot_isolation_insert_during_search() {
    let (_temp_dir, storage) = create_test_storage();
    let storage = Arc::new(storage);

    // Create HNSW index
    let nav_cache = Arc::new(NavigationCache::new());
    let spec = make_test_spec(DIM, 16, 100);
    let index = Arc::new(hnsw::Index::from_spec(
        TEST_EMBEDDING,
        &spec,
        64, // batch_threshold
        nav_cache,
    ));

    // Step 1: Insert initial 500 vectors
    let initial_count = 500u32;
    for vec_id in 0..initial_count {
        let vector = seeded_vector(DIM, vec_id as u64);
        match insert_vector_atomic(&storage, &index, TEST_EMBEDDING, vec_id, &vector) {
            InsertResult::Success(cache_update) => {
                cache_update.apply(index.nav_cache());
            }
            _ => panic!("Insert {} failed", vec_id),
        }
    }

    // Step 2: Take snapshot
    let txn_db = storage.transaction_db().expect("txn_db");
    let snapshot = txn_db.snapshot();
    let vectors_cf = txn_db.cf_handle(Vectors::CF_NAME).expect("cf");

    // Step 3: Insert 500 more vectors concurrently
    let storage_clone = Arc::clone(&storage);
    let index_clone = Arc::clone(&index);
    let insert_handle = thread::spawn(move || {
        let mut inserted = 0u32;
        for vec_id in initial_count..(initial_count + 500) {
            let vector = seeded_vector(DIM, vec_id as u64);
            match insert_vector_atomic(&storage_clone, &index_clone, TEST_EMBEDDING, vec_id, &vector) {
                InsertResult::Success(cache_update) => {
                    cache_update.apply(index_clone.nav_cache());
                    inserted += 1;
                }
                _ => {}
            }
        }
        inserted
    });

    // Step 4: Read using snapshot (should see only initial 500)
    let snapshot_count: usize = {
        let iter = snapshot.iterator_cf(&vectors_cf, rocksdb::IteratorMode::Start);
        iter.count()
    };

    // Wait for inserts to complete
    let inserted = insert_handle.join().expect("insert thread");
    println!("Inserted {} additional vectors", inserted);

    // Step 5: Verify snapshot isolation
    assert_eq!(
        snapshot_count, initial_count as usize,
        "Snapshot should see only initial {} vectors, but saw {}",
        initial_count, snapshot_count
    );

    // Step 6: Current view should see all vectors
    let current_count: usize = txn_db
        .iterator_cf(&vectors_cf, rocksdb::IteratorMode::Start)
        .count();

    assert!(
        current_count > initial_count as usize,
        "Current view should see more vectors after insert"
    );

    println!(
        "Snapshot isolation validated: snapshot saw {}, current sees {}",
        snapshot_count, current_count
    );
}

/// Test: Snapshot isolation with Reader during concurrent inserts.
///
/// Validates that snapshot count remains stable while Reader queries
/// run during inserts.
#[test]
fn test_snapshot_isolation_insert_during_search_reader() {
    let (_temp_dir, storage) = create_test_storage();
    let storage = Arc::new(storage);

    let registry = storage.cache().clone();
    let txn_db = storage.transaction_db().expect("txn_db");
    let embedding = registry
        .register(
            EmbeddingBuilder::new("snapshot-test-reader", DIM as u32, Distance::L2)
                .with_hnsw_m(16)
                .with_hnsw_ef_construction(100),
            &txn_db,
        )
        .expect("embedding register");

    let initial_count = 500u32;
    let mut inserts = Vec::with_capacity(initial_count as usize);
    for seed in 0..initial_count {
        inserts.push((Id::new(), seeded_vector(DIM, seed as u64)));
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");

    let (writer, writer_rx) = create_writer(WriterConfig {
        channel_buffer_size: 10_000,
    });
    let mutation_handle = {
        let _guard = runtime.enter();
        spawn_mutation_consumer_with_storage_autoreg(
            writer_rx,
            WriterConfig::default(),
            storage.clone(),
        )
    };

    runtime.block_on(async {
        for (id, vector) in inserts {
            InsertVector::new(&embedding, ExternalKey::NodeId(id), vector)
                .immediate()
                .run(&writer)
                .await
                .expect("insert");
        }
        writer.flush().await.expect("flush");
    });

    let txn_db = storage.transaction_db().expect("txn_db");
    let snapshot = txn_db.snapshot();
    let reverse_cf = txn_db.cf_handle(ID_REVERSE_CF).expect("cf");
    let snapshot_count: usize = snapshot
        .iterator_cf(&reverse_cf, rocksdb::IteratorMode::Start)
        .count();
    assert_eq!(
        snapshot_count, initial_count as usize,
        "Snapshot should see initial {} vectors",
        initial_count
    );

    let writer_clone = writer.clone();
    let embedding_clone = embedding.clone();
    let search_results = runtime.block_on(async {
        let insert_handle = tokio::spawn(async move {
            for seed in initial_count..(initial_count + 200) {
                let vector = seeded_vector(DIM, seed as u64);
                InsertVector::new(&embedding_clone, ExternalKey::NodeId(Id::new()), vector)
                    .immediate()
                    .run(&writer_clone)
                    .await
                    .expect("insert");
            }
            writer_clone.flush().await.expect("flush");
            initial_count + 200
        });

        let (search_reader, reader_rx) =
            create_reader_with_storage(ReaderConfig::default(), storage.clone());
        let query_handles = spawn_query_consumers_with_storage_autoreg(
            reader_rx,
            ReaderConfig::default(),
            storage.clone(),
            2,
        );

        let mut results = Vec::new();
        let timeout = std::time::Duration::from_secs(5);
        let query = seeded_vector(DIM, 4243);

        for _ in 0..3 {
            let result = search_reader
                .search_knn(&embedding, query.clone(), 10, 100, timeout)
                .await
                .expect("search");
            results.push(result);
        }

        drop(search_reader);
        for handle in query_handles {
            let _ = handle.await;
        }

        let inserted = insert_handle.await.expect("insert task");
        (results, inserted)
    });

    let (search_results, inserted) = search_results;
    let current_count: usize = txn_db
        .iterator_cf(&reverse_cf, rocksdb::IteratorMode::Start)
        .count();
    assert!(
        current_count >= inserted as usize,
        "Current view should see inserts after completion"
    );

    for (idx, result) in search_results.iter().enumerate() {
        assert!(
            !result.is_empty(),
            "Reader result {} should not be empty during inserts",
            idx
        );
    }

    drop(writer);
    let _ = runtime.block_on(async { mutation_handle.await });
}

// ============================================================================
// Test 8.2.2: Transaction Conflict Resolution
// ============================================================================

/// Test: Concurrent transaction serialization.
///
/// Validates that RocksDB's pessimistic locking properly serializes
/// concurrent transactions modifying the same key.
///
/// Note: RocksDB TransactionDB uses pessimistic locking by default,
/// which means the second transaction waits for the lock rather than
/// detecting a conflict at commit time. This is the correct behavior
/// for our use case.
#[test]
fn test_transaction_conflict_resolution() {
    let (_temp_dir, storage) = create_test_storage();
    let storage = Arc::new(storage);

    let num_threads = 4;
    let ops_per_thread = 50;

    // Single key that all threads compete for
    let shared_key = VectorCfKey(TEST_EMBEDDING, 0);
    let key_bytes = Vectors::key_to_bytes(&shared_key);

    // Track successful commits
    let commit_count = Arc::new(AtomicU32::new(0));

    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let storage = Arc::clone(&storage);
        let commit_count = Arc::clone(&commit_count);
        let key_bytes = key_bytes.clone();

        handles.push(thread::spawn(move || {
            let txn_db = storage.transaction_db().expect("txn_db");
            let vectors_cf = txn_db.cf_handle(Vectors::CF_NAME).expect("cf");

            let mut successes = 0u32;

            for i in 0..ops_per_thread {
                let txn = txn_db.transaction();

                // Write a unique value (thread_id + iteration)
                let vector = vec![(thread_id * 1000 + i) as f32; DIM];
                let value = VectorCfValue(vector);

                if txn.put_cf(&vectors_cf, &key_bytes, Vectors::value_to_bytes(&value)).is_ok() {
                    if txn.commit().is_ok() {
                        commit_count.fetch_add(1, Ordering::Relaxed);
                        successes += 1;
                    }
                }
            }

            (thread_id, successes)
        }));
    }

    // Collect results
    for handle in handles {
        let (thread_id, successes) = handle.join().expect("thread");
        println!("Thread {}: {} commits", thread_id, successes);
    }

    let total_commits = commit_count.load(Ordering::Relaxed);
    let expected_total = (num_threads * ops_per_thread) as u32;

    println!("\n=== Transaction Serialization Results ===");
    println!("Total commits: {} / {}", total_commits, expected_total);

    // With pessimistic locking, all commits should succeed (serialized)
    assert_eq!(
        total_commits, expected_total,
        "All transactions should commit with pessimistic locking"
    );

    println!("Transaction serialization validated");
}

/// Test: High-contention transaction stress test.
///
/// Multiple threads competing to modify overlapping keys validates
/// that pessimistic locking serializes access correctly without
/// data corruption.
#[test]
fn test_transaction_conflict_stress() {
    let (_temp_dir, storage) = create_test_storage();
    let storage = Arc::new(storage);

    let metrics = Arc::new(ConcurrentMetrics::new());
    let num_threads = 8;
    let ops_per_thread = 100;

    // Use a small key space to maximize contention
    let key_space = 10u32;

    let mut handles = Vec::new();

    for thread_id in 0..num_threads {
        let storage = Arc::clone(&storage);
        let metrics = Arc::clone(&metrics);

        handles.push(thread::spawn(move || {
            let txn_db = storage.transaction_db().expect("txn_db");
            let vectors_cf = txn_db.cf_handle(Vectors::CF_NAME).expect("cf");
            let mut rng = rand::rng();

            let mut successes = 0u32;
            let mut errors = 0u32;

            for op_idx in 0..ops_per_thread {
                // Pick a random key from the small key space
                let vec_id = rng.random_range(0..key_space);
                let key = VectorCfKey(TEST_EMBEDDING, vec_id);
                let key_bytes = Vectors::key_to_bytes(&key);

                let txn = txn_db.transaction();

                // Read-modify-write pattern
                let _ = txn.get_cf(&vectors_cf, &key_bytes);

                // Write unique value that can be verified
                let vector = vec![(thread_id * 10000 + op_idx) as f32; DIM];
                let value = VectorCfValue(vector);

                let start = Instant::now();
                if txn.put_cf(&vectors_cf, &key_bytes, Vectors::value_to_bytes(&value)).is_ok() {
                    match txn.commit() {
                        Ok(_) => {
                            metrics.record_insert(start.elapsed());
                            successes += 1;
                        }
                        Err(_) => {
                            metrics.record_error();
                            errors += 1;
                        }
                    }
                } else {
                    metrics.record_error();
                    errors += 1;
                }
            }

            (thread_id, successes, errors)
        }));
    }

    // Collect results
    let mut total_successes = 0u32;
    let mut total_errors = 0u32;

    for handle in handles {
        let (thread_id, successes, errors) = handle.join().expect("thread");
        println!(
            "Thread {}: {} successes, {} errors",
            thread_id, successes, errors
        );
        total_successes += successes;
        total_errors += errors;
    }

    let total_ops = (num_threads * ops_per_thread) as u32;
    let success_rate = total_successes as f64 / total_ops as f64;

    println!("\n=== High-Contention Stress Results ===");
    println!("Total operations: {}", total_ops);
    println!("Successes: {}", total_successes);
    println!("Errors: {}", total_errors);
    println!("Success rate: {:.1}%", success_rate * 100.0);
    println!("{}", metrics.summary());

    // With pessimistic locking, most operations should succeed
    assert!(
        success_rate > 0.95,
        "Success rate should be > 95% with pessimistic locking, got {:.1}%",
        success_rate * 100.0
    );

    // Verify no data corruption by reading all keys
    let txn_db = storage.transaction_db().expect("txn_db");
    let vectors_cf = txn_db.cf_handle(Vectors::CF_NAME).expect("cf");

    let mut valid_reads = 0;
    for vec_id in 0..key_space {
        let key = VectorCfKey(TEST_EMBEDDING, vec_id);
        let key_bytes = Vectors::key_to_bytes(&key);
        if let Ok(Some(value)) = txn_db.get_cf(&vectors_cf, &key_bytes) {
            // Verify value is valid (can be deserialized)
            if value.len() == DIM * 4 {
                valid_reads += 1;
            }
        }
    }

    println!("Valid keys after stress: {}/{}", valid_reads, key_space);
    assert!(valid_reads > 0, "Should have some valid keys after stress test");

    println!("High-contention stress test passed");
}
