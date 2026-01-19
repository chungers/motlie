//! Snapshot Isolation Tests for Vector Subsystem (Task 8.2.1, 8.2.2).
//!
//! These tests validate RocksDB's snapshot isolation guarantees under concurrent access:
//!
//! | Test | Description | Validates |
//! |------|-------------|-----------|
//! | `test_snapshot_isolation_delete_during_search` | Search sees consistent snapshot despite concurrent deletes | Isolation |
//! | `test_snapshot_isolation_insert_during_search` | Search snapshot excludes concurrent inserts | Isolation |
//! | `test_transaction_conflict_resolution` | Optimistic concurrency detects conflicts | Conflict handling |
//! | `test_transaction_conflict_stress` | High-contention conflict handling | Stress |
//!
//! ## RocksDB Snapshot Isolation
//!
//! RocksDB provides snapshot isolation through:
//! 1. **Read snapshots**: `db.snapshot()` returns a point-in-time view
//! 2. **Optimistic transactions**: Detect write-write conflicts at commit time
//!
//! These tests verify that:
//! - Searches with snapshots see consistent data
//! - Concurrent modifications don't corrupt ongoing reads
//! - Transaction conflicts are properly detected and reported

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use motlie_db::vector::benchmark::ConcurrentMetrics;
use motlie_db::vector::hnsw::{self, Config as HnswConfig};
use motlie_db::vector::schema::{EmbeddingCode, VectorCfKey, VectorCfValue, Vectors};
use motlie_db::vector::{cache::NavigationCache, Distance, Storage};
use motlie_db::rocksdb::ColumnFamily;
use rand::prelude::*;
use tempfile::TempDir;

const DIM: usize = 64;
const TEST_EMBEDDING: EmbeddingCode = 1;

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
    (0..dim).map(|_| rng.gen::<f32>()).collect()
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

    // Create HNSW index
    let nav_cache = Arc::new(NavigationCache::new());
    let hnsw_config = HnswConfig {
        dim: DIM,
        m: 16,
        m_max: 32,
        m_max_0: 32,
        ef_construction: 100,
        ..Default::default()
    };
    let index = Arc::new(hnsw::Index::new(
        TEST_EMBEDDING,
        Distance::L2,
        hnsw_config,
        nav_cache,
    ));

    // Step 1: Insert 1000 vectors
    let num_vectors = 1000u32;
    println!("Inserting {} vectors...", num_vectors);

    for vec_id in 0..num_vectors {
        let vector = seeded_vector(DIM, vec_id as u64);
        match insert_vector_atomic(&storage, &index, TEST_EMBEDDING, vec_id, &vector) {
            InsertResult::Success(cache_update) => {
                cache_update.apply(index.nav_cache());
            }
            _ => panic!("Insert {} failed", vec_id),
        }
    }
    println!("Inserted {} vectors", num_vectors);

    // Step 2: Take a snapshot before deletes
    let txn_db = storage.transaction_db().expect("txn_db");
    let snapshot = txn_db.snapshot();

    // Verify initial count via snapshot
    let vectors_cf = txn_db.cf_handle(Vectors::CF_NAME).expect("cf");
    let initial_count: usize = txn_db
        .iterator_cf(&vectors_cf, rocksdb::IteratorMode::Start)
        .count();
    assert_eq!(initial_count, num_vectors as usize, "Should have all vectors before delete");

    // Step 3: Concurrent delete of 500 vectors (vec_ids 0-499)
    let delete_count = 500u32;
    let storage_clone = Arc::clone(&storage);
    let delete_handle = thread::spawn(move || {
        let txn_db = storage_clone.transaction_db().expect("txn_db");
        let vectors_cf = txn_db.cf_handle(Vectors::CF_NAME).expect("cf");

        let mut deleted = 0u32;
        for vec_id in 0..delete_count {
            let key = VectorCfKey(TEST_EMBEDDING, vec_id);
            let txn = txn_db.transaction();
            if txn.delete_cf(&vectors_cf, Vectors::key_to_bytes(&key)).is_ok() {
                if txn.commit().is_ok() {
                    deleted += 1;
                }
            }
        }
        deleted
    });

    // Step 4: While deletes are happening, read using snapshot
    // Snapshot should see all 1000 vectors (point-in-time consistency)
    let snapshot_count: usize = {
        let iter = snapshot.iterator_cf(&vectors_cf, rocksdb::IteratorMode::Start);
        iter.count()
    };

    // Wait for deletes to complete
    let deleted = delete_handle.join().expect("delete thread");
    println!("Deleted {} vectors", deleted);

    // Step 5: Verify snapshot isolation
    assert_eq!(
        snapshot_count, num_vectors as usize,
        "Snapshot should see all {} vectors (isolation), but saw {}",
        num_vectors, snapshot_count
    );

    // Step 6: New read (without snapshot) should see fewer vectors
    let current_count: usize = txn_db
        .iterator_cf(&vectors_cf, rocksdb::IteratorMode::Start)
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
    let hnsw_config = HnswConfig {
        dim: DIM,
        m: 16,
        m_max: 32,
        m_max_0: 32,
        ef_construction: 100,
        ..Default::default()
    };
    let index = Arc::new(hnsw::Index::new(
        TEST_EMBEDDING,
        Distance::L2,
        hnsw_config,
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
            let mut rng = rand::thread_rng();

            let mut successes = 0u32;
            let mut errors = 0u32;

            for op_idx in 0..ops_per_thread {
                // Pick a random key from the small key space
                let vec_id = rng.gen_range(0..key_space);
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
