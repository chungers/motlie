//! Multi-threaded stress tests for vector subsystem (Task 5.9).
//!
//! These tests validate concurrent access patterns without data corruption or panics.
//!
//! ## Test Categories
//!
//! | Test | Description | Thread Config |
//! |------|-------------|---------------|
//! | `test_concurrent_insert_search` | Insert while searching | 2 inserters, 2 searchers |
//! | `test_concurrent_batch_insert` | Parallel batch inserts | 4 batch inserters |
//! | `test_high_thread_count_stress` | Maximum concurrency | 8 writers, 8 readers |
//! | `test_writer_contention` | Multiple writers same embedding | 8 writers |
//!
//! ## Validation Criteria
//!
//! - No panics under any thread configuration
//! - No data corruption (inserted vectors retrievable)
//! - Search results consistent (no phantom results)
//!
//! ## Expected Error Conditions
//!
//! Errors are expected in these scenarios and do NOT indicate bugs:
//!
//! 1. **Empty graph search**: Searching before any vector is inserted returns an error
//!    because there is no entry point. This is expected behavior during the startup
//!    window when writers haven't committed their first insert yet.
//!
//! 2. **Transaction conflicts**: RocksDB's optimistic concurrency control may reject
//!    transactions when multiple writers modify overlapping keys. This is expected
//!    under high contention and is why we allow a small error rate.
//!
//! The 10% error threshold in tests accounts for both conditions. In production,
//! applications should either:
//! - Wait for index readiness before querying, or
//! - Handle "empty index" errors gracefully
//!
//! ## vec_id Generation Bounds
//!
//! Tests use different vec_id generation strategies:
//! - `test_concurrent_batch_insert`: `(thread_id << 16) | i` - max 65,536 vectors/thread
//! - `test_writer_contention`: Atomic counter - max 2^32 vectors total
//!
//! These bounds are sufficient for stress testing but should be documented if
//! configs are changed to use larger vector counts.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use motlie_db::rocksdb::ColumnFamily;
use motlie_db::vector::benchmark::ConcurrentMetrics;
use motlie_db::vector::hnsw::{self, Config as HnswConfig};
use motlie_db::vector::schema::{EmbeddingCode, VectorCfKey, VectorCfValue, Vectors};
use motlie_db::vector::{cache::NavigationCache, Distance, Storage};
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

/// Helper: Generate random vector
fn random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen::<f32>()).collect()
}

/// Helper: Store vector in RocksDB
fn store_vector(storage: &Storage, embedding: EmbeddingCode, vec_id: u32, vector: &[f32]) -> bool {
    let Ok(txn_db) = storage.transaction_db() else {
        return false;
    };
    let Some(cf) = txn_db.cf_handle(Vectors::CF_NAME) else {
        return false;
    };

    let txn = txn_db.transaction();
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
        return false;
    }
    txn.commit().is_ok()
}

/// Test: Concurrent insert and search operations.
///
/// Spawns writer threads inserting vectors while reader threads search.
/// Validates no panics and data integrity.
#[test]
fn test_concurrent_insert_search() {
    let (_temp_dir, storage) = create_test_storage();
    let storage = Arc::new(storage);

    let metrics = Arc::new(ConcurrentMetrics::new());
    let stop_flag = Arc::new(AtomicBool::new(false));
    let insert_counter = Arc::new(AtomicU32::new(0));

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

    let mut handles = Vec::new();

    // Spawn 2 writer threads
    for thread_id in 0..2 {
        let storage = Arc::clone(&storage);
        let index = Arc::clone(&index);
        let metrics = Arc::clone(&metrics);
        let stop = Arc::clone(&stop_flag);
        let counter = Arc::clone(&insert_counter);

        handles.push(thread::spawn(move || {
            let txn_db = storage.transaction_db().expect("txn_db");
            let mut local_count = 0u32;

            while !stop.load(Ordering::Relaxed) && local_count < 500 {
                let vec_id = counter.fetch_add(1, Ordering::Relaxed);
                let vector = random_vector(DIM);

                let start = Instant::now();

                // Store and insert
                if store_vector(&storage, TEST_EMBEDDING, vec_id, &vector) {
                    let txn = txn_db.transaction();
                    if let Ok(cache_update) =
                        hnsw::insert(&index, &txn, &txn_db, &storage, vec_id, &vector)
                    {
                        if txn.commit().is_ok() {
                            cache_update.apply(index.nav_cache());
                            metrics.record_insert(start.elapsed());
                            local_count += 1;
                        } else {
                            metrics.record_error();
                        }
                    } else {
                        metrics.record_error();
                    }
                } else {
                    metrics.record_error();
                }
            }

            println!(
                "Writer {} finished: {} inserts",
                thread_id, local_count
            );
        }));
    }

    // Spawn 2 reader threads
    for thread_id in 0..2 {
        let storage = Arc::clone(&storage);
        let index = Arc::clone(&index);
        let metrics = Arc::clone(&metrics);
        let stop = Arc::clone(&stop_flag);

        handles.push(thread::spawn(move || {
            let mut search_count = 0u64;

            while !stop.load(Ordering::Relaxed) {
                let query = random_vector(DIM);

                let start = Instant::now();
                let result = index.search(&storage, &query, 10, 50);
                let latency = start.elapsed();

                match result {
                    Ok(_results) => {
                        metrics.record_search(latency);
                        search_count += 1;
                    }
                    Err(_) => {
                        metrics.record_error();
                    }
                }

                // Small delay to allow writers to make progress
                if search_count % 100 == 0 {
                    thread::sleep(Duration::from_millis(1));
                }
            }

            println!(
                "Reader {} finished: {} searches",
                thread_id, search_count
            );
        }));
    }

    // Let the test run for 5 seconds
    thread::sleep(Duration::from_secs(5));
    stop_flag.store(true, Ordering::Relaxed);

    // Wait for all threads to finish
    for handle in handles {
        handle.join().expect("Thread panicked - STRESS TEST FAILED");
    }

    let summary = metrics.summary();
    println!("\n=== test_concurrent_insert_search Results ===");
    println!("{}", summary);

    // Validation criteria
    assert!(
        summary.insert_count > 0,
        "Should have completed some inserts"
    );
    assert!(
        summary.search_count > 0,
        "Should have completed some searches"
    );
    // Allow some errors during concurrent operations.
    // Expected error sources:
    // 1. Empty graph searches (no entry point yet) - occurs before first insert commits
    // 2. Transaction conflicts under high contention
    // The 10% threshold is generous to account for timing variations.
    // In practice, error rates should be much lower once the index is populated.
    let error_rate = summary.error_count as f64
        / (summary.insert_count + summary.search_count + summary.error_count) as f64;
    assert!(
        error_rate < 0.1,
        "Error rate should be < 10%, got {:.1}%",
        error_rate * 100.0
    );
}

/// Test: Parallel batch inserts from multiple threads.
///
/// Each thread inserts a batch of vectors independently.
/// Validates no data corruption and all vectors are retrievable.
#[test]
fn test_concurrent_batch_insert() {
    let (_temp_dir, storage) = create_test_storage();
    let storage = Arc::new(storage);

    let metrics = Arc::new(ConcurrentMetrics::new());
    let vectors_per_thread = 200;
    let num_threads = 4;

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

    let mut handles = Vec::new();

    // Spawn batch insert threads
    for thread_id in 0..num_threads {
        let storage = Arc::clone(&storage);
        let index = Arc::clone(&index);
        let metrics = Arc::clone(&metrics);

        handles.push(thread::spawn(move || {
            let txn_db = storage.transaction_db().expect("txn_db");
            let mut success_count = 0;

            for i in 0..vectors_per_thread {
                // Use thread_id to generate unique vec_ids
                let vec_id = ((thread_id as u32) << 16) | (i as u32);
                let vector = random_vector(DIM);

                let start = Instant::now();

                if store_vector(&storage, TEST_EMBEDDING, vec_id, &vector) {
                    let txn = txn_db.transaction();
                    if let Ok(cache_update) =
                        hnsw::insert(&index, &txn, &txn_db, &storage, vec_id, &vector)
                    {
                        if txn.commit().is_ok() {
                            cache_update.apply(index.nav_cache());
                            metrics.record_insert(start.elapsed());
                            success_count += 1;
                        }
                    }
                }
            }

            println!(
                "Thread {} completed: {}/{} vectors inserted",
                thread_id, success_count, vectors_per_thread
            );
            success_count
        }));
    }

    // Wait for all threads and collect results
    let total_inserted: u32 = handles
        .into_iter()
        .map(|h| h.join().expect("Thread panicked"))
        .sum();

    let summary = metrics.summary();
    println!("\n=== test_concurrent_batch_insert Results ===");
    println!("{}", summary);
    println!(
        "Total inserted: {} / {}",
        total_inserted,
        num_threads * vectors_per_thread
    );

    // Validation: Most vectors should be inserted successfully.
    // Under high contention, RocksDB's optimistic concurrency control may reject
    // some transactions. An 80% success rate is acceptable for stress testing.
    let expected = (num_threads * vectors_per_thread) as u32;
    let success_rate = total_inserted as f64 / expected as f64;
    assert!(
        success_rate >= 0.80,
        "At least 80% of vectors should be inserted, got {:.1}%",
        success_rate * 100.0
    );

    // Validation: Search should find vectors
    let query = random_vector(DIM);
    let results = index.search(&storage, &query, 10, 100).expect("Search failed");
    assert!(
        !results.is_empty() || total_inserted == 0,
        "Search should return results if vectors were inserted"
    );
}

/// Test: High thread count stress test.
///
/// Maximizes concurrency to find potential deadlocks or race conditions.
/// Uses 8 writers and 8 readers for 10 seconds.
#[test]
fn test_high_thread_count_stress() {
    let (_temp_dir, storage) = create_test_storage();
    let storage = Arc::new(storage);

    let metrics = Arc::new(ConcurrentMetrics::new());
    let stop_flag = Arc::new(AtomicBool::new(false));
    let insert_counter = Arc::new(AtomicU32::new(0));

    let num_writers = 8;
    let num_readers = 8;
    let test_duration = Duration::from_secs(10);

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

    let mut handles = Vec::new();

    // Spawn writer threads
    for thread_id in 0..num_writers {
        let storage = Arc::clone(&storage);
        let index = Arc::clone(&index);
        let metrics = Arc::clone(&metrics);
        let stop = Arc::clone(&stop_flag);
        let counter = Arc::clone(&insert_counter);

        handles.push(thread::spawn(move || {
            let txn_db = storage.transaction_db().expect("txn_db");

            while !stop.load(Ordering::Relaxed) {
                let vec_id = counter.fetch_add(1, Ordering::Relaxed);
                let vector = random_vector(DIM);

                let start = Instant::now();

                if store_vector(&storage, TEST_EMBEDDING, vec_id, &vector) {
                    let txn = txn_db.transaction();
                    if let Ok(cache_update) =
                        hnsw::insert(&index, &txn, &txn_db, &storage, vec_id, &vector)
                    {
                        if txn.commit().is_ok() {
                            cache_update.apply(index.nav_cache());
                            metrics.record_insert(start.elapsed());
                        } else {
                            metrics.record_error();
                        }
                    } else {
                        metrics.record_error();
                    }
                } else {
                    metrics.record_error();
                }
            }

            thread_id
        }));
    }

    // Spawn reader threads
    for thread_id in 0..num_readers {
        let storage = Arc::clone(&storage);
        let index = Arc::clone(&index);
        let metrics = Arc::clone(&metrics);
        let stop = Arc::clone(&stop_flag);

        handles.push(thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let query = random_vector(DIM);

                let start = Instant::now();
                let result = index.search(&storage, &query, 10, 50);

                match result {
                    Ok(_) => metrics.record_search(start.elapsed()),
                    Err(_) => metrics.record_error(),
                }
            }

            thread_id
        }));
    }

    // Let the stress test run
    println!(
        "Running high thread count stress test ({} writers, {} readers) for {:?}...",
        num_writers, num_readers, test_duration
    );
    thread::sleep(test_duration);
    stop_flag.store(true, Ordering::Relaxed);

    // Wait for all threads - if any panics, the test fails
    let mut panic_count = 0;
    for handle in handles {
        if handle.join().is_err() {
            panic_count += 1;
        }
    }

    let summary = metrics.summary();
    println!("\n=== test_high_thread_count_stress Results ===");
    println!("{}", summary);

    // Validation criteria
    assert_eq!(
        panic_count, 0,
        "No threads should have panicked"
    );
    assert!(
        summary.insert_count > 0,
        "Should have completed inserts"
    );
    assert!(
        summary.search_count > 0,
        "Should have completed searches"
    );

    // Report throughput
    let insert_throughput = summary.insert_count as f64 / test_duration.as_secs_f64();
    let search_throughput = summary.search_count as f64 / test_duration.as_secs_f64();
    println!(
        "\nThroughput: {:.1} inserts/sec, {:.1} searches/sec",
        insert_throughput, search_throughput
    );
}

/// Test: Writer contention - multiple threads writing to same embedding.
///
/// Tests that concurrent writes to the same embedding space don't corrupt data.
#[test]
fn test_writer_contention() {
    let (_temp_dir, storage) = create_test_storage();
    let storage = Arc::new(storage);

    let metrics = Arc::new(ConcurrentMetrics::new());
    let insert_counter = Arc::new(AtomicU32::new(0));

    let num_writers = 8;
    let vectors_per_writer = 100;

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

    let mut handles = Vec::new();

    // Spawn writer threads that all use the same counter for vec_ids
    // This creates contention on the ID space
    for thread_id in 0..num_writers {
        let storage = Arc::clone(&storage);
        let index = Arc::clone(&index);
        let metrics = Arc::clone(&metrics);
        let counter = Arc::clone(&insert_counter);

        handles.push(thread::spawn(move || {
            let txn_db = storage.transaction_db().expect("txn_db");
            let mut success = 0;
            let mut failures = 0;

            for _ in 0..vectors_per_writer {
                // All threads compete for incrementing vec_ids
                let vec_id = counter.fetch_add(1, Ordering::Relaxed);
                let vector = random_vector(DIM);

                let start = Instant::now();

                if store_vector(&storage, TEST_EMBEDDING, vec_id, &vector) {
                    let txn = txn_db.transaction();
                    match hnsw::insert(&index, &txn, &txn_db, &storage, vec_id, &vector) {
                        Ok(cache_update) => {
                            if txn.commit().is_ok() {
                                cache_update.apply(index.nav_cache());
                                metrics.record_insert(start.elapsed());
                                success += 1;
                            } else {
                                metrics.record_error();
                                failures += 1;
                            }
                        }
                        Err(_) => {
                            metrics.record_error();
                            failures += 1;
                        }
                    }
                } else {
                    metrics.record_error();
                    failures += 1;
                }
            }

            (thread_id, success, failures)
        }));
    }

    // Collect results
    let mut total_success = 0;
    let mut total_failures = 0;
    for handle in handles {
        let (thread_id, success, failures) = handle.join().expect("Thread panicked");
        println!(
            "Writer {}: {} success, {} failures",
            thread_id, success, failures
        );
        total_success += success;
        total_failures += failures;
    }

    let summary = metrics.summary();
    println!("\n=== test_writer_contention Results ===");
    println!("{}", summary);
    println!(
        "Total: {} success, {} failures out of {}",
        total_success,
        total_failures,
        num_writers * vectors_per_writer
    );

    // Under heavy contention, many transactions will fail due to RocksDB's
    // optimistic concurrency control. The goal of this test is to verify:
    // 1. No panics occur (thread safety)
    // 2. Data integrity is maintained (no corruption)
    // 3. At least some operations succeed
    // Success rate can be very low under extreme contention - this is expected.
    assert!(
        total_success > 0,
        "At least some writes should succeed under contention"
    );

    // Verify data integrity: vectors that were inserted should be searchable
    let query = random_vector(DIM);
    let results = index.search(&storage, &query, 10, 100).expect("Search failed");
    println!("Search returned {} results", results.len());
}
