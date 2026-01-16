//! Integration tests for vector cache infrastructure.
//!
//! Tests the Arc-based BinaryCodeCache under concurrent access patterns:
//! - Multi-threaded reads (simulating parallel search)
//! - Validates zero-copy Arc sharing across threads
//! - Ensures recall is maintained with concurrent access
//!
//! Uses synthetic LAION-CLIP data (1K scale) to avoid external dataset dependencies.

use std::sync::Arc;
use std::thread;
use tempfile::TempDir;

use motlie_db::vector::benchmark::{compute_recall, LaionSubset, LAION_EMBEDDING_DIM};
use motlie_db::vector::quantization::RaBitQ;
use motlie_db::vector::schema::Vectors;
use motlie_db::vector::{
    hnsw, BinaryCodeCache, Distance, NavigationCache, Storage, VecId,
    VectorElementType,
};
use motlie_db::vector::hnsw::insert_in_txn;
use motlie_db::rocksdb::ColumnFamily;

/// Generate synthetic LAION-CLIP-like data (512D, Cosine distance, normalized)
fn generate_laion_subset(num_vectors: usize, num_queries: usize) -> LaionSubset {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let normalize = |v: &mut Vec<f32>| {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    };

    let db_vectors: Vec<Vec<f32>> = (0..num_vectors)
        .map(|_| {
            let mut v: Vec<f32> = (0..LAION_EMBEDDING_DIM)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();
            normalize(&mut v);
            v
        })
        .collect();

    let queries: Vec<Vec<f32>> = (0..num_queries)
        .map(|_| {
            let mut v: Vec<f32> = (0..LAION_EMBEDDING_DIM)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect();
            normalize(&mut v);
            v
        })
        .collect();

    let ground_truth: Vec<usize> = (0..num_queries).collect();

    LaionSubset {
        db_vectors,
        queries,
        ground_truth,
        dim: LAION_EMBEDDING_DIM,
    }
}

/// Build HNSW index with NavigationCache
fn build_index_with_navigation(
    storage: &Storage,
    vectors: &[Vec<f32>],
    distance: Distance,
    storage_type: VectorElementType,
    m: usize,
    ef_construction: usize,
) -> anyhow::Result<(hnsw::Index, Arc<NavigationCache>)> {
    let dim = vectors.first().map(|v| v.len()).unwrap_or(512);
    let nav_cache = Arc::new(NavigationCache::new());
    let embedding_code = 1;

    let config = hnsw::Config {
        dim,
        m,
        m_max: m * 2,
        m_max_0: m * 2,
        ef_construction,
        ..Default::default()
    };

    let index = hnsw::Index::with_storage_type(
        embedding_code,
        distance,
        storage_type,
        config,
        nav_cache.clone(),
    );

    let txn_db = storage.transaction_db()?;
    let vectors_cf = txn_db
        .cf_handle(Vectors::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("Vectors CF not found"))?;

    use motlie_db::vector::schema::VectorCfKey;

    for (i, vector) in vectors.iter().enumerate() {
        let vec_id = i as VecId;
        let key = VectorCfKey(embedding_code, vec_id);
        let value_bytes = Vectors::value_to_bytes_typed(vector, storage_type);

        // Use transaction for atomic vector storage + HNSW insert
        let txn = txn_db.transaction();
        txn.put_cf(&vectors_cf, Vectors::key_to_bytes(&key), value_bytes)?;
        let cache_update = insert_in_txn(&index, &txn, &txn_db, storage, vec_id, vector)?;
        txn.commit()?;
        cache_update.apply(index.nav_cache());
    }

    Ok((index, nav_cache))
}

/// Multi-threaded BinaryCodeCache Integration Test
///
/// Validates:
/// - Arc<BinaryCodeEntry> sharing across multiple threads
/// - Concurrent cache reads don't cause data races
/// - Recall is maintained with parallel access
/// - No performance degradation from Arc overhead
#[test]
fn test_binary_code_cache_multithreaded_access() -> anyhow::Result<()> {
    const NUM_VECTORS: usize = 1000;
    const NUM_QUERIES: usize = 100;
    const NUM_THREADS: usize = 4;

    let temp_dir = TempDir::new()?;
    let db_path = temp_dir.path().join("cache_mt_test_db");

    // Generate synthetic LAION-CLIP data
    let subset = generate_laion_subset(NUM_VECTORS, NUM_QUERIES);

    println!("=== Multi-threaded BinaryCodeCache Test ===");
    println!(
        "Vectors: {}, Queries: {}, Threads: {}",
        NUM_VECTORS, NUM_QUERIES, NUM_THREADS
    );

    // Compute ground truth
    let ground_truth = subset.compute_ground_truth_topk_with_distance(10, Distance::Cosine);

    // Initialize storage and build index
    let mut storage = Storage::readwrite(&db_path);
    storage.ready()?;

    let (index, _nav_cache) = build_index_with_navigation(
        &storage,
        &subset.db_vectors,
        Distance::Cosine,
        VectorElementType::F16,
        16,
        100,
    )?;

    // Initialize RaBitQ encoder and populate BinaryCodeCache
    let rabitq = Arc::new(RaBitQ::new(LAION_EMBEDDING_DIM, 2, 42)); // 2-bit for better recall
    let binary_cache = Arc::new(BinaryCodeCache::new());
    let embedding_code = 1u64;

    for (i, vector) in subset.db_vectors.iter().enumerate() {
        let vec_id = i as VecId;
        let (code, correction) = rabitq.encode_with_correction(vector);
        binary_cache.put(embedding_code, vec_id, code, correction);
    }

    let (cache_count, cache_bytes) = binary_cache.stats();
    println!(
        "BinaryCodeCache populated: {} entries, {:.2} KB",
        cache_count,
        cache_bytes as f64 / 1024.0
    );

    // Partition queries across threads
    let queries_per_thread = NUM_QUERIES / NUM_THREADS;
    let queries = Arc::new(subset.queries);
    let index = Arc::new(index);
    let storage = Arc::new(storage);

    // Spawn threads to run concurrent searches
    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            let binary_cache = binary_cache.clone();
            let rabitq = rabitq.clone();
            let queries = queries.clone();
            let index = index.clone();
            let storage = storage.clone();

            thread::spawn(move || {
                let start_idx = thread_id * queries_per_thread;
                let end_idx = start_idx + queries_per_thread;
                let mut results: Vec<Vec<usize>> = Vec::new();

                for query in queries[start_idx..end_idx].iter() {
                    let search_results = index
                        .search_with_rabitq_cached(
                            &storage,
                            query,
                            &rabitq,
                            &binary_cache,
                            10,  // k
                            50,  // ef_search
                            4,   // rerank_factor
                        )
                        .expect("Search failed");

                    let result_ids: Vec<usize> =
                        search_results.iter().map(|(_, id)| *id as usize).collect();
                    results.push(result_ids);
                }

                (thread_id, results)
            })
        })
        .collect();

    // Collect results from all threads
    let mut all_results: Vec<(usize, Vec<Vec<usize>>)> = handles
        .into_iter()
        .map(|h| h.join().expect("Thread panicked"))
        .collect();

    // Sort by thread_id to ensure consistent ordering
    all_results.sort_by_key(|(id, _)| *id);

    // Flatten results in query order
    let search_results: Vec<Vec<usize>> = all_results
        .into_iter()
        .flat_map(|(_, results)| results)
        .collect();

    assert_eq!(search_results.len(), NUM_QUERIES);

    // Compute recall
    let recall = compute_recall(&search_results, &ground_truth, 10);
    println!("Recall@10: {:.2}%", recall * 100.0);

    // With 2-bit RaBitQ and rerank_factor=4, expect reasonable recall
    assert!(
        recall >= 0.50,
        "Expected recall >= 50%, got {:.2}%",
        recall * 100.0
    );

    println!("✓ Multi-threaded BinaryCodeCache test passed");

    Ok(())
}

/// Test Arc sharing behavior under concurrent access
///
/// Validates:
/// - Multiple threads can hold Arc references simultaneously
/// - Arc::strong_count reflects concurrent usage
/// - No data corruption with concurrent reads
#[test]
fn test_arc_sharing_concurrent_reads() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    const NUM_VECTORS: usize = 100;
    const NUM_THREADS: usize = 8;
    const READS_PER_THREAD: usize = 1000;

    let cache = Arc::new(BinaryCodeCache::new());
    let embedding_code = 1u64;

    // Populate cache
    for vec_id in 0..NUM_VECTORS as VecId {
        let code = vec![vec_id as u8; 64]; // 64-byte code
        let correction = motlie_db::vector::schema::AdcCorrection::new(1.0, 1.0);
        cache.put(embedding_code, vec_id, code, correction);
    }

    println!("=== Arc Sharing Concurrent Reads Test ===");
    println!(
        "Vectors: {}, Threads: {}, Reads/thread: {}",
        NUM_VECTORS, NUM_THREADS, READS_PER_THREAD
    );

    let successful_reads = Arc::new(AtomicUsize::new(0));
    let total_arc_refs = Arc::new(AtomicUsize::new(0));

    // Spawn threads to do concurrent reads
    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|_| {
            let cache = cache.clone();
            let successful_reads = successful_reads.clone();
            let total_arc_refs = total_arc_refs.clone();

            thread::spawn(move || {
                for i in 0..READS_PER_THREAD {
                    let vec_id = (i % NUM_VECTORS) as VecId;
                    if let Some(entry) = cache.get(embedding_code, vec_id) {
                        // Verify data integrity
                        assert_eq!(entry.code.len(), 64);
                        assert_eq!(entry.code[0], vec_id as u8);

                        // Track Arc reference count
                        total_arc_refs.fetch_add(Arc::strong_count(&entry), Ordering::Relaxed);
                        successful_reads.fetch_add(1, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();

    // Wait for all threads
    for h in handles {
        h.join().expect("Thread panicked");
    }

    let total_reads = successful_reads.load(Ordering::Relaxed);
    let expected_reads = NUM_THREADS * READS_PER_THREAD;

    println!("Successful reads: {}/{}", total_reads, expected_reads);
    assert_eq!(total_reads, expected_reads);

    // Average Arc ref count should be > 1 (cache + at least one reader)
    let avg_refs = total_arc_refs.load(Ordering::Relaxed) as f64 / total_reads as f64;
    println!("Average Arc strong_count during reads: {:.2}", avg_refs);
    assert!(avg_refs >= 2.0, "Expected avg refs >= 2.0, got {:.2}", avg_refs);

    println!("✓ Arc sharing concurrent reads test passed");
}

/// Test cache get_batch under concurrent access
///
/// Validates:
/// - Batch reads work correctly with multiple threads
/// - All entries in batch are valid Arc references
#[test]
fn test_batch_reads_concurrent() {
    const NUM_VECTORS: usize = 500;
    const NUM_THREADS: usize = 4;
    const BATCH_SIZE: usize = 50;

    let cache = Arc::new(BinaryCodeCache::new());
    let embedding_code = 1u64;

    // Populate cache
    for vec_id in 0..NUM_VECTORS as VecId {
        let code = vec![(vec_id % 256) as u8; 32];
        let correction = motlie_db::vector::schema::AdcCorrection::new(
            vec_id as f32 / NUM_VECTORS as f32,
            0.9,
        );
        cache.put(embedding_code, vec_id, code, correction);
    }

    println!("=== Batch Reads Concurrent Test ===");
    println!(
        "Vectors: {}, Threads: {}, Batch size: {}",
        NUM_VECTORS, NUM_THREADS, BATCH_SIZE
    );

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            let cache = cache.clone();

            thread::spawn(move || {
                let mut total_entries = 0usize;

                // Each thread reads different batches
                for batch_start in (thread_id * BATCH_SIZE..NUM_VECTORS).step_by(NUM_THREADS * BATCH_SIZE) {
                    let batch_end = (batch_start + BATCH_SIZE).min(NUM_VECTORS);
                    let vec_ids: Vec<VecId> = (batch_start..batch_end).map(|i| i as VecId).collect();

                    let batch = cache.get_batch(embedding_code, &vec_ids);

                    for (i, entry_opt) in batch.into_iter().enumerate() {
                        if let Some(entry) = entry_opt {
                            let expected_vec_id = (batch_start + i) as VecId;
                            assert_eq!(entry.code[0], (expected_vec_id % 256) as u8);
                            total_entries += 1;
                        }
                    }
                }

                total_entries
            })
        })
        .collect();

    let total: usize = handles
        .into_iter()
        .map(|h| h.join().expect("Thread panicked"))
        .sum();

    println!("Total entries read across threads: {}", total);
    assert!(total > 0, "Expected some entries to be read");

    println!("✓ Batch reads concurrent test passed");
}

/// Stress test: High contention on same cache entries
///
/// Validates:
/// - Cache handles high contention gracefully
/// - No deadlocks with many readers
#[test]
fn test_high_contention_same_entries() {
    const NUM_HOT_ENTRIES: usize = 10;
    const NUM_THREADS: usize = 16;
    const READS_PER_THREAD: usize = 10000;

    let cache = Arc::new(BinaryCodeCache::new());
    let embedding_code = 1u64;

    // Create a small number of "hot" entries
    for vec_id in 0..NUM_HOT_ENTRIES as VecId {
        let code = vec![0xAB; 128];
        let correction = motlie_db::vector::schema::AdcCorrection::new(1.0, 1.0);
        cache.put(embedding_code, vec_id, code, correction);
    }

    println!("=== High Contention Test ===");
    println!(
        "Hot entries: {}, Threads: {}, Reads/thread: {}",
        NUM_HOT_ENTRIES, NUM_THREADS, READS_PER_THREAD
    );

    let start = std::time::Instant::now();

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|_| {
            let cache = cache.clone();

            thread::spawn(move || {
                for i in 0..READS_PER_THREAD {
                    let vec_id = (i % NUM_HOT_ENTRIES) as VecId;
                    let entry = cache.get(embedding_code, vec_id).expect("Entry should exist");
                    assert_eq!(entry.code.len(), 128);
                    assert_eq!(entry.code[0], 0xAB);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("Thread panicked");
    }

    let elapsed = start.elapsed();
    let total_reads = NUM_THREADS * READS_PER_THREAD;
    let reads_per_sec = total_reads as f64 / elapsed.as_secs_f64();

    println!(
        "Completed {} reads in {:.2}ms ({:.0} reads/sec)",
        total_reads,
        elapsed.as_secs_f64() * 1000.0,
        reads_per_sec
    );

    // Should complete reasonably fast (no deadlocks)
    assert!(
        elapsed.as_secs() < 10,
        "Test took too long, possible contention issue"
    );

    println!("✓ High contention test passed");
}
