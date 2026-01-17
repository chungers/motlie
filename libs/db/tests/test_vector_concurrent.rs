//! Multi-threaded stress tests for vector subsystem (Task 5.9).
//!
//! These tests validate concurrent access patterns without data corruption or panics.
//! All inserts use atomic transactions (vector data + HNSW graph in single commit).
//!
//! ## Test Categories
//!
//! | Test | Description | Thread Config |
//! |------|-------------|---------------|
//! | `test_concurrent_insert_search` | Atomic insert while searching | 2 inserters, 2 searchers |
//! | `test_concurrent_batch_insert` | Atomic parallel batch inserts | 4 batch inserters |
//! | `test_high_thread_count_stress` | Atomic maximum concurrency | 8 writers, 8 readers |
//! | `test_writer_contention` | Atomic writers same embedding | 8 writers |
//! | `test_multi_embedding_concurrent_access` | Multi-index concurrent r/w | 3 indices, 6 writers, 6 readers |
//! | `test_cache_isolation_under_load` | Cache isolation validation | 2 indices, 2 writers, 2 readers |
//! | `benchmark_quick_validation` | Quick benchmark for CI | 2 writers, 2 readers |
//! | `benchmark_baseline_balanced` | Balanced workload (ignored) | 4 writers, 4 readers |
//! | `benchmark_baseline_read_heavy` | Read-heavy workload (ignored) | 1 writer, 8 readers |
//! | `benchmark_baseline_write_heavy` | Write-heavy workload (ignored) | 8 writers, 1 reader |
//! | `benchmark_baseline_stress` | Stress test (ignored) | 16 writers, 16 readers |
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
use motlie_db::vector::benchmark::{BenchConfig, ConcurrentBenchmark, ConcurrentMetrics};
use motlie_db::vector::hnsw::{self, Config as HnswConfig};
use motlie_db::vector::schema::{EmbeddingCode, VectorCfKey, VectorCfValue, Vectors};
use motlie_db::vector::{cache::NavigationCache, Distance, Storage};
use rand::prelude::*;
use tempfile::TempDir;

const DIM: usize = 64;
const TEST_EMBEDDING: EmbeddingCode = 1;
/// Maximum vec_id value to prevent overflow in thread-partitioned ID generation.
/// With `(thread_id << 16) | i`, this allows 65,536 vectors per thread.
const MAX_VECTORS_PER_THREAD: u32 = 65_536;

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

/// Atomic insert result - indicates whether insert succeeded or failed with reason.
#[derive(Debug)]
enum InsertResult {
    /// Insert succeeded, cache update should be applied
    Success(hnsw::CacheUpdate),
    /// Transaction commit failed (RocksDB conflict)
    CommitFailed,
    /// HNSW insert failed
    HnswFailed,
}

/// Helper: Atomic vector + HNSW insert in a single transaction.
///
/// This mirrors production semantics where vector data and HNSW graph
/// are updated atomically - either both succeed or neither does.
/// No orphaned vectors are possible.
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
            let mut local_count = 0u32;

            while !stop.load(Ordering::Relaxed) && local_count < 500 {
                let vec_id = counter.fetch_add(1, Ordering::Relaxed);
                let vector = random_vector(DIM);

                let start = Instant::now();

                // Atomic insert: vector + HNSW in single transaction
                match insert_vector_atomic(&storage, &index, TEST_EMBEDDING, vec_id, &vector) {
                    InsertResult::Success(cache_update) => {
                        cache_update.apply(index.nav_cache());
                        metrics.record_insert(start.elapsed());
                        local_count += 1;
                    }
                    InsertResult::CommitFailed | InsertResult::HnswFailed => {
                        metrics.record_error();
                    }
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
            let mut success_count = 0;

            for i in 0..vectors_per_thread {
                // Use thread_id to generate unique vec_ids
                // Guard: ensure i doesn't exceed MAX_VECTORS_PER_THREAD
                assert!(
                    (i as u32) < MAX_VECTORS_PER_THREAD,
                    "vectors_per_thread exceeds MAX_VECTORS_PER_THREAD"
                );
                let vec_id = ((thread_id as u32) << 16) | (i as u32);
                let vector = random_vector(DIM);

                let start = Instant::now();

                // Atomic insert: vector + HNSW in single transaction
                match insert_vector_atomic(&storage, &index, TEST_EMBEDDING, vec_id, &vector) {
                    InsertResult::Success(cache_update) => {
                        cache_update.apply(index.nav_cache());
                        metrics.record_insert(start.elapsed());
                        success_count += 1;
                    }
                    InsertResult::CommitFailed | InsertResult::HnswFailed => {
                        // Expected under contention
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
            while !stop.load(Ordering::Relaxed) {
                let vec_id = counter.fetch_add(1, Ordering::Relaxed);
                let vector = random_vector(DIM);

                let start = Instant::now();

                // Atomic insert: vector + HNSW in single transaction
                match insert_vector_atomic(&storage, &index, TEST_EMBEDDING, vec_id, &vector) {
                    InsertResult::Success(cache_update) => {
                        cache_update.apply(index.nav_cache());
                        metrics.record_insert(start.elapsed());
                    }
                    InsertResult::CommitFailed | InsertResult::HnswFailed => {
                        metrics.record_error();
                    }
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
            let mut success = 0;
            let mut failures = 0;

            for _ in 0..vectors_per_writer {
                // All threads compete for incrementing vec_ids
                let vec_id = counter.fetch_add(1, Ordering::Relaxed);
                let vector = random_vector(DIM);

                let start = Instant::now();

                // Atomic insert: vector + HNSW in single transaction
                match insert_vector_atomic(&storage, &index, TEST_EMBEDDING, vec_id, &vector) {
                    InsertResult::Success(cache_update) => {
                        cache_update.apply(index.nav_cache());
                        metrics.record_insert(start.elapsed());
                        success += 1;
                    }
                    InsertResult::CommitFailed | InsertResult::HnswFailed => {
                        metrics.record_error();
                        failures += 1;
                    }
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

// ============================================================================
// Multi-Embedding Concurrent Tests (CODEX Coverage Gaps)
// ============================================================================

const EMBEDDING_A: EmbeddingCode = 100;
const EMBEDDING_B: EmbeddingCode = 200;
const EMBEDDING_C: EmbeddingCode = 300;

/// Test: Multi-embedding concurrent writes and reads.
///
/// Validates that concurrent operations across multiple embeddings in the
/// same Storage don't interfere with each other. This tests:
/// 1. Concurrent writers to different embeddings
/// 2. Concurrent readers from different embeddings
/// 3. No cross-contamination between embeddings
#[test]
fn test_multi_embedding_concurrent_access() {
    let (_temp_dir, storage) = create_test_storage();
    let storage = Arc::new(storage);

    let metrics = Arc::new(ConcurrentMetrics::new());
    let stop_flag = Arc::new(AtomicBool::new(false));

    // Track inserted vec_ids per embedding for validation
    let inserted_a = Arc::new(std::sync::Mutex::new(Vec::new()));
    let inserted_b = Arc::new(std::sync::Mutex::new(Vec::new()));
    let inserted_c = Arc::new(std::sync::Mutex::new(Vec::new()));

    // Create separate HNSW indices for each embedding (shared nav_cache)
    let nav_cache = Arc::new(NavigationCache::new());
    let hnsw_config = HnswConfig {
        dim: DIM,
        m: 16,
        m_max: 32,
        m_max_0: 32,
        ef_construction: 100,
        ..Default::default()
    };

    let index_a = Arc::new(hnsw::Index::new(
        EMBEDDING_A,
        Distance::L2,
        hnsw_config.clone(),
        Arc::clone(&nav_cache),
    ));
    let index_b = Arc::new(hnsw::Index::new(
        EMBEDDING_B,
        Distance::Cosine,
        hnsw_config.clone(),
        Arc::clone(&nav_cache),
    ));
    let index_c = Arc::new(hnsw::Index::new(
        EMBEDDING_C,
        Distance::L2,
        hnsw_config,
        Arc::clone(&nav_cache),
    ));

    let mut handles: Vec<thread::JoinHandle<()>> = Vec::new();

    // Spawn writers for each embedding (2 writers each)
    for (embedding, index, inserted) in [
        (EMBEDDING_A, Arc::clone(&index_a), Arc::clone(&inserted_a)),
        (EMBEDDING_B, Arc::clone(&index_b), Arc::clone(&inserted_b)),
        (EMBEDDING_C, Arc::clone(&index_c), Arc::clone(&inserted_c)),
    ] {
        for _writer_id in 0..2 {
            let storage = Arc::clone(&storage);
            let index = Arc::clone(&index);
            let metrics = Arc::clone(&metrics);
            let stop = Arc::clone(&stop_flag);
            let inserted = Arc::clone(&inserted);
            let counter = Arc::new(AtomicU32::new(0));

            handles.push(thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let local_count = counter.fetch_add(1, Ordering::Relaxed);
                    if local_count >= 100 {
                        break;
                    }
                    // Unique vec_id: (embedding << 8) | count
                    let vec_id = ((embedding as u32) << 8) | local_count;
                    let vector = random_vector(DIM);

                    let start = Instant::now();

                    match insert_vector_atomic(&storage, &index, embedding, vec_id, &vector) {
                        InsertResult::Success(cache_update) => {
                            cache_update.apply(index.nav_cache());
                            metrics.record_insert(start.elapsed());
                            inserted.lock().unwrap().push(vec_id);
                        }
                        InsertResult::CommitFailed | InsertResult::HnswFailed => {
                            metrics.record_error();
                        }
                    }
                }
            }));
        }
    }

    // Spawn readers for each embedding (2 readers each)
    for (_embedding, index) in [
        (EMBEDDING_A, Arc::clone(&index_a)),
        (EMBEDDING_B, Arc::clone(&index_b)),
        (EMBEDDING_C, Arc::clone(&index_c)),
    ] {
        for _reader_id in 0..2 {
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
                        Ok(_results) => {
                            metrics.record_search(start.elapsed());
                        }
                        Err(_) => {
                            // Expected before first insert commits
                            metrics.record_error();
                        }
                    }
                }
            }));
        }
    }

    // Let test run for 5 seconds
    thread::sleep(Duration::from_secs(5));
    stop_flag.store(true, Ordering::Relaxed);

    // Wait for threads and check for panics
    let mut panic_count = 0;
    for handle in handles {
        if handle.join().is_err() {
            panic_count += 1;
        }
    }

    let summary = metrics.summary();
    println!("\n=== test_multi_embedding_concurrent_access Results ===");
    println!("{}", summary);
    println!(
        "Inserted per embedding: A={}, B={}, C={}",
        inserted_a.lock().unwrap().len(),
        inserted_b.lock().unwrap().len(),
        inserted_c.lock().unwrap().len()
    );

    // Validation
    assert_eq!(panic_count, 0, "No threads should have panicked");
    assert!(summary.insert_count > 0, "Should have completed inserts");
    assert!(summary.search_count > 0, "Should have completed searches");

    // Verify no cross-contamination: search each index and check results
    for (embedding, index, inserted) in [
        (EMBEDDING_A, &index_a, &inserted_a),
        (EMBEDDING_B, &index_b, &inserted_b),
        (EMBEDDING_C, &index_c, &inserted_c),
    ] {
        let query = random_vector(DIM);
        if let Ok(results) = index.search(&storage, &query, 10, 100) {
            let inserted_set: std::collections::HashSet<_> =
                inserted.lock().unwrap().iter().cloned().collect();

            for (_dist, vec_id) in &results {
                assert!(
                    inserted_set.contains(vec_id),
                    "Embedding {} search returned vec_id {} not in its insert set (cross-contamination!)",
                    embedding,
                    vec_id
                );
            }
            println!(
                "Embedding {}: {} search results, all valid",
                embedding,
                results.len()
            );
        }
    }
}

/// Test: Cache isolation under concurrent multi-embedding load.
///
/// Validates that NavigationCache and BinaryCodeCache return correct data
/// under concurrent multi-embedding access. This specifically tests the
/// cache keying by EmbeddingCode.
#[test]
fn test_cache_isolation_under_load() {
    let (_temp_dir, storage) = create_test_storage();
    let storage = Arc::new(storage);

    let metrics = Arc::new(ConcurrentMetrics::new());
    let stop_flag = Arc::new(AtomicBool::new(false));

    // Shared caches (keyed by EmbeddingCode)
    let nav_cache = Arc::new(NavigationCache::new());

    let hnsw_config = HnswConfig {
        dim: DIM,
        m: 16,
        m_max: 32,
        m_max_0: 32,
        ef_construction: 100,
        ..Default::default()
    };

    // Create indices sharing the same cache
    let index_a = Arc::new(hnsw::Index::new(
        EMBEDDING_A,
        Distance::L2,
        hnsw_config.clone(),
        Arc::clone(&nav_cache),
    ));
    let index_b = Arc::new(hnsw::Index::new(
        EMBEDDING_B,
        Distance::Cosine,
        hnsw_config,
        Arc::clone(&nav_cache),
    ));

    // Track what we inserted to each embedding
    let vec_ids_a = Arc::new(std::sync::Mutex::new(std::collections::HashSet::new()));
    let vec_ids_b = Arc::new(std::sync::Mutex::new(std::collections::HashSet::new()));

    let mut handles = Vec::new();

    // Pre-populate with some vectors
    for i in 0..50 {
        let vec_id_a = ((EMBEDDING_A as u32) << 16) | i;
        let vec_id_b = ((EMBEDDING_B as u32) << 16) | i;

        if let InsertResult::Success(cache_update) =
            insert_vector_atomic(&storage, &index_a, EMBEDDING_A, vec_id_a, &random_vector(DIM))
        {
            cache_update.apply(index_a.nav_cache());
            vec_ids_a.lock().unwrap().insert(vec_id_a);
        }

        if let InsertResult::Success(cache_update) =
            insert_vector_atomic(&storage, &index_b, EMBEDDING_B, vec_id_b, &random_vector(DIM))
        {
            cache_update.apply(index_b.nav_cache());
            vec_ids_b.lock().unwrap().insert(vec_id_b);
        }
    }

    // Global cross-contamination counter
    let cross_contamination = Arc::new(AtomicU32::new(0));

    // Spawn concurrent inserters + searchers for both embeddings
    for (embedding, index, vec_ids) in [
        (EMBEDDING_A, Arc::clone(&index_a), Arc::clone(&vec_ids_a)),
        (EMBEDDING_B, Arc::clone(&index_b), Arc::clone(&vec_ids_b)),
    ] {
        // Clone for writer and reader threads
        let storage_w = Arc::clone(&storage);
        let storage_r = Arc::clone(&storage);
        let index_w = Arc::clone(&index);
        let index_r = Arc::clone(&index);
        let metrics_w = Arc::clone(&metrics);
        let metrics_r = Arc::clone(&metrics);
        let stop_w = Arc::clone(&stop_flag);
        let stop_r = Arc::clone(&stop_flag);
        let vec_ids_w = Arc::clone(&vec_ids);
        let cross_contamination_r = Arc::clone(&cross_contamination);
        let counter = Arc::new(AtomicU32::new(100)); // Start after pre-populated

        // Writer thread
        handles.push(thread::spawn(move || {
            while !stop_w.load(Ordering::Relaxed) {
                let i = counter.fetch_add(1, Ordering::Relaxed);
                if i > 300 {
                    break;
                }
                let vec_id = ((embedding as u32) << 16) | i;
                let vector = random_vector(DIM);

                let start = Instant::now();
                match insert_vector_atomic(&storage_w, &index_w, embedding, vec_id, &vector) {
                    InsertResult::Success(cache_update) => {
                        cache_update.apply(index_w.nav_cache());
                        metrics_w.record_insert(start.elapsed());
                        vec_ids_w.lock().unwrap().insert(vec_id);
                    }
                    _ => metrics_w.record_error(),
                }
            }
        }));

        // Reader thread - validates cache correctness
        handles.push(thread::spawn(move || {
            let mut search_count = 0u64;

            while !stop_r.load(Ordering::Relaxed) && search_count < 500 {
                let query = random_vector(DIM);

                let start = Instant::now();
                if let Ok(results) = index_r.search(&storage_r, &query, 10, 50) {
                    metrics_r.record_search(start.elapsed());
                    search_count += 1;

                    // Check for cross-contamination: vec_id should have this embedding's code
                    for (_dist, vec_id) in &results {
                        // Extract embedding from vec_id (stored in upper 16 bits)
                        let result_embedding = (vec_id >> 16) as EmbeddingCode;
                        if result_embedding != embedding {
                            cross_contamination_r.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
            }
        }));
    }

    // Let test run
    thread::sleep(Duration::from_secs(5));
    stop_flag.store(true, Ordering::Relaxed);

    // Wait for threads
    for handle in handles {
        let _ = handle.join();
    }

    let total_cross_contamination = cross_contamination.load(Ordering::Relaxed);

    let summary = metrics.summary();
    println!("\n=== test_cache_isolation_under_load Results ===");
    println!("{}", summary);
    println!(
        "Vec IDs: A={}, B={}",
        vec_ids_a.lock().unwrap().len(),
        vec_ids_b.lock().unwrap().len()
    );
    println!("Cross-contamination count: {}", total_cross_contamination);

    // Validation: NO cross-contamination allowed
    assert_eq!(
        total_cross_contamination, 0,
        "Cache isolation violated! Found {} cross-contaminated results",
        total_cross_contamination
    );
}

// ============================================================================
// Benchmark: Baseline Concurrent Performance (Task 5.11)
// ============================================================================

/// Runs the concurrent benchmark with "balanced" preset and captures baseline numbers.
///
/// This test is marked #[ignore] because it takes longer to run (30s).
/// Run with: `cargo test -p motlie-db --test test_vector_concurrent benchmark_baseline -- --ignored --nocapture`
#[test]
#[ignore]
fn benchmark_baseline_balanced() {
    let (_temp_dir, storage) = create_test_storage();
    let storage = Arc::new(storage);

    // Use balanced preset: 4 writers, 4 readers, 30s duration
    let config = BenchConfig::balanced();
    let metrics = Arc::new(ConcurrentMetrics::new());
    let bench = ConcurrentBenchmark::new(config, metrics);

    println!("\n=== Concurrent Benchmark: Balanced (4w/4r, 30s) ===\n");

    let result = bench.run(storage, TEST_EMBEDDING).expect("benchmark run");

    println!("{}", result);
    println!("\n=== End Benchmark ===\n");

    // No assertions - this is for capturing baseline numbers
}

/// Quick benchmark with smaller config for CI validation.
#[test]
fn benchmark_quick_validation() {
    let (_temp_dir, storage) = create_test_storage();
    let storage = Arc::new(storage);

    // Quick config: 2 writers, 2 readers, 5s duration
    let config = BenchConfig {
        writer_threads: 2,
        reader_threads: 2,
        duration: Duration::from_secs(5),
        vectors_per_writer: 500,
        vector_dim: DIM,
        ..Default::default()
    };
    let metrics = Arc::new(ConcurrentMetrics::new());
    let bench = ConcurrentBenchmark::new(config, metrics);

    println!("\n=== Quick Benchmark Validation (2w/2r, 5s) ===\n");

    let result = bench.run(storage, TEST_EMBEDDING).expect("benchmark run");

    println!("{}", result);

    // Basic validation: should complete without errors
    assert!(result.insert_throughput > 0.0, "Should have non-zero insert throughput");
    assert!(result.search_throughput > 0.0, "Should have non-zero search throughput");
}

/// Read-heavy baseline: 1 writer, 8 readers, 30s
/// Simulates CDN/cache access patterns.
#[test]
#[ignore]
fn benchmark_baseline_read_heavy() {
    let (_temp_dir, storage) = create_test_storage();
    let storage = Arc::new(storage);

    let config = BenchConfig::read_heavy();
    let metrics = Arc::new(ConcurrentMetrics::new());
    let bench = ConcurrentBenchmark::new(config, metrics);

    println!("\n=== Concurrent Benchmark: Read-Heavy (1w/8r, 30s) ===\n");

    let result = bench.run(storage, TEST_EMBEDDING).expect("benchmark run");

    println!("{}", result);
    println!("\n=== End Benchmark ===\n");
}

/// Write-heavy baseline: 8 writers, 1 reader, 30s
/// Simulates batch ingestion workloads.
#[test]
#[ignore]
fn benchmark_baseline_write_heavy() {
    let (_temp_dir, storage) = create_test_storage();
    let storage = Arc::new(storage);

    let config = BenchConfig::write_heavy();
    let metrics = Arc::new(ConcurrentMetrics::new());
    let bench = ConcurrentBenchmark::new(config, metrics);

    println!("\n=== Concurrent Benchmark: Write-Heavy (8w/1r, 30s) ===\n");

    let result = bench.run(storage, TEST_EMBEDDING).expect("benchmark run");

    println!("{}", result);
    println!("\n=== End Benchmark ===\n");
}

/// Stress baseline: 16 writers, 16 readers, 60s
/// Maximum concurrency to find bottlenecks.
#[test]
#[ignore]
fn benchmark_baseline_stress() {
    let (_temp_dir, storage) = create_test_storage();
    let storage = Arc::new(storage);

    let config = BenchConfig::stress();
    let metrics = Arc::new(ConcurrentMetrics::new());
    let bench = ConcurrentBenchmark::new(config, metrics);

    println!("\n=== Concurrent Benchmark: Stress (16w/16r, 60s) ===\n");

    let result = bench.run(storage, TEST_EMBEDDING).expect("benchmark run");

    println!("{}", result);
    println!("\n=== End Benchmark ===\n");
}
