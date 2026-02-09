//! Crash recovery tests for vector module.
//!
//! These tests verify that:
//! 1. Uncommitted transactions don't persist data (atomicity)
//! 2. Committed transactions survive storage close/reopen (durability)
//! 3. IdAllocator recovers its state after restart
//! 4. HNSW navigation info loads correctly from cold cache
//!
//! # Test Strategy
//!
//! Each test follows the pattern:
//! 1. Create temporary storage
//! 2. Perform operations
//! 3. Close storage (simulating crash/restart)
//! 4. Reopen storage with fresh caches
//! 5. Verify expected state

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tempfile::TempDir;

    use crate::rocksdb::ColumnFamily;
    use crate::vector::cache::NavigationCache;
    use crate::vector::hnsw::{self, insert};
    use crate::vector::id::IdAllocator;
    use crate::vector::schema::{
        EmbeddingCode, EmbeddingSpec, GraphMeta, GraphMetaCfKey, GraphMetaField, VecId,
        VectorCfKey, VectorElementType, Vectors,
    };
    use crate::vector::{Distance, Storage};

    /// Helper: Create test storage
    fn create_test_storage(path: &std::path::Path) -> Storage {
        let mut storage = Storage::readwrite(path);
        storage.ready().expect("Failed to initialize storage");
        storage
    }

    /// Helper: Generate a random vector
    fn random_vector(dim: usize, seed: u64) -> Vec<f32> {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha20Rng;
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..dim).map(|_| rng.gen::<f32>()).collect()
    }

    /// Helper: Create test EmbeddingSpec with default HNSW parameters
    fn make_test_spec(dim: usize, distance: Distance) -> EmbeddingSpec {
        EmbeddingSpec {
            code: 0,
            model: "test".to_string(),
            dim: dim as u32,
            distance,
            storage_type: VectorElementType::F32,
            hnsw_m: 16,
            hnsw_ef_construction: 200,
            rabitq_bits: 1,
            rabitq_seed: 42,
        }
    }

    // =========================================================================
    // Transaction Atomicity Tests
    // =========================================================================

    #[test]
    fn test_uncommitted_transaction_not_visible() {
        // Verify that uncommitted transactions don't persist data
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let embedding: EmbeddingCode = 1;

        // Phase 1: Create transaction but don't commit
        {
            let storage = create_test_storage(temp_dir.path());
            let txn_db = storage.transaction_db().expect("Failed to get txn_db");
            let txn = txn_db.transaction();

            // Write vector data in transaction (but don't commit)
            let cf = txn_db
                .cf_handle(Vectors::CF_NAME)
                .expect("Vectors CF not found");

            let key = Vectors::key_to_bytes(&VectorCfKey(embedding, 0));
            let vector = random_vector(128, 42);
            let value = Vectors::value_to_bytes_typed(
                &vector,
                crate::vector::schema::VectorElementType::F32,
            );

            txn.put_cf(&cf, key, value).expect("Failed to put in txn");

            // Transaction is dropped here WITHOUT commit (simulating crash)
            drop(txn);

            // Verify data is not visible even in same session
            let result = txn_db.get_cf(
                &cf,
                Vectors::key_to_bytes(&VectorCfKey(embedding, 0)),
            );
            assert!(
                result.expect("Failed to get").is_none(),
                "Uncommitted data should not be visible"
            );
        }

        // Phase 2: Reopen storage and verify data is still not present
        {
            let storage = create_test_storage(temp_dir.path());
            let txn_db = storage.transaction_db().expect("Failed to get txn_db");
            let cf = txn_db
                .cf_handle(Vectors::CF_NAME)
                .expect("Vectors CF not found");

            let result = txn_db.get_cf(
                &cf,
                Vectors::key_to_bytes(&VectorCfKey(embedding, 0)),
            );
            assert!(
                result.expect("Failed to get").is_none(),
                "Uncommitted data should not persist after restart"
            );
        }
    }

    #[test]
    fn test_committed_transaction_persists() {
        // Verify that committed transactions survive restart
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let embedding: EmbeddingCode = 1;
        let vec_id: VecId = 42;
        let vector = random_vector(128, 42);

        // Phase 1: Create and commit transaction
        {
            let storage = create_test_storage(temp_dir.path());
            let txn_db = storage.transaction_db().expect("Failed to get txn_db");
            let txn = txn_db.transaction();

            let cf = txn_db
                .cf_handle(Vectors::CF_NAME)
                .expect("Vectors CF not found");

            let key = Vectors::key_to_bytes(&VectorCfKey(embedding, vec_id));
            let value = Vectors::value_to_bytes_typed(
                &vector,
                crate::vector::schema::VectorElementType::F32,
            );

            txn.put_cf(&cf, key, value).expect("Failed to put in txn");
            txn.commit().expect("Failed to commit");
        }

        // Phase 2: Reopen and verify data persists
        {
            let storage = create_test_storage(temp_dir.path());
            let txn_db = storage.transaction_db().expect("Failed to get txn_db");
            let cf = txn_db
                .cf_handle(Vectors::CF_NAME)
                .expect("Vectors CF not found");

            let result = txn_db
                .get_cf(&cf, Vectors::key_to_bytes(&VectorCfKey(embedding, vec_id)))
                .expect("Failed to get");

            assert!(result.is_some(), "Committed data should persist after restart");

            // Verify the data is correct
            let bytes = result.unwrap();
            let recovered = Vectors::value_from_bytes(&bytes).expect("Failed to deserialize");
            assert_eq!(recovered.0.len(), 128, "Vector dimension should match");
        }
    }

    // =========================================================================
    // IdAllocator Recovery Tests
    // =========================================================================

    #[test]
    fn test_id_allocator_recovery() {
        // Verify IdAllocator recovers its state after restart
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let embedding: EmbeddingCode = 1;

        // Phase 1: Allocate IDs and persist
        let allocated_ids: Vec<VecId>;
        {
            let storage = create_test_storage(temp_dir.path());
            let txn_db = storage.transaction_db().expect("Failed to get txn_db");

            let allocator = IdAllocator::new();

            // Allocate 10 IDs
            allocated_ids = (0..10).map(|_| allocator.allocate_local()).collect();

            // Free ID 5
            allocator.free_local(5);

            // Persist state
            allocator.persist(&txn_db, embedding).expect("Failed to persist");

            assert_eq!(allocator.next_id(), 10);
            assert_eq!(allocator.free_count(), 1);
        }

        // Phase 2: Recover and verify state
        {
            let storage = create_test_storage(temp_dir.path());
            let txn_db = storage.transaction_db().expect("Failed to get txn_db");

            let recovered = IdAllocator::recover(&txn_db, embedding).expect("Failed to recover");

            assert_eq!(recovered.next_id(), 10, "next_id should be recovered");
            assert_eq!(recovered.free_count(), 1, "free_count should be recovered");

            // Next allocation should reuse freed ID 5
            let next_id = recovered.allocate_local();
            assert_eq!(next_id, 5, "Should reuse freed ID");

            // After that, should allocate fresh
            let fresh_id = recovered.allocate_local();
            assert_eq!(fresh_id, 10, "Should allocate fresh ID after reusing freed");
        }
    }

    #[test]
    fn test_id_allocator_transactional_recovery() {
        // Verify IdAllocator transactions work correctly
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let embedding: EmbeddingCode = 1;

        // Phase 1: Allocate using transactional API
        {
            let storage = create_test_storage(temp_dir.path());
            let txn_db = storage.transaction_db().expect("Failed to get txn_db");
            let txn = txn_db.transaction();

            let allocator = IdAllocator::new();

            // Allocate 5 IDs in transaction
            for _ in 0..5 {
                allocator
                    .allocate(&txn, &txn_db, embedding)
                    .expect("Failed to allocate in txn");
            }

            txn.commit().expect("Failed to commit");
        }

        // Phase 2: Verify recovery
        {
            let storage = create_test_storage(temp_dir.path());
            let txn_db = storage.transaction_db().expect("Failed to get txn_db");

            let recovered = IdAllocator::recover(&txn_db, embedding).expect("Failed to recover");
            assert_eq!(recovered.next_id(), 5, "Should have 5 IDs allocated");
        }
    }

    // =========================================================================
    // HNSW Navigation Recovery Tests
    // =========================================================================

    #[test]
    fn test_hnsw_navigation_cold_cache_recovery() {
        // Verify HNSW search works correctly after recovery with cold cache
        // Note: Search loads navigation info from storage on cache miss, but may not
        // cache it (performance optimization TBD). The key test is that search works.
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let embedding: EmbeddingCode = 1;
        let dim = 64;
        let n_vectors = 10;

        // Generate same vectors for both phases
        let vectors: Vec<Vec<f32>> = (0..n_vectors)
            .map(|i| random_vector(dim, i as u64))
            .collect();

        // Phase 1: Insert vectors with HNSW indexing
        {
            let storage = create_test_storage(temp_dir.path());

            // First store vectors in Vectors CF (using transaction)
            let txn_db = storage.transaction_db().expect("Failed to get txn_db");
            let cf = txn_db
                .cf_handle(Vectors::CF_NAME)
                .expect("Vectors CF not found");

            let txn = txn_db.transaction();
            for (i, vector) in vectors.iter().enumerate() {
                let key = Vectors::key_to_bytes(&VectorCfKey(embedding, i as VecId));
                let value = Vectors::value_to_bytes_typed(
                    vector,
                    crate::vector::schema::VectorElementType::F32,
                );
                txn.put_cf(&cf, key, value).expect("Failed to store vector");
            }
            txn.commit().expect("Failed to commit vectors");

            // Now build HNSW index
            let nav_cache = Arc::new(NavigationCache::new());
            let spec = make_test_spec(dim, Distance::L2);
            let index = hnsw::Index::from_spec(embedding, &spec, 64, nav_cache.clone());

            for (i, vector) in vectors.iter().enumerate() {
                let txn = txn_db.transaction();
                let cache_update =
                    insert(&index, &txn, &txn_db, &storage, i as VecId, vector)
                        .expect("Failed to insert");
                txn.commit().expect("Failed to commit");
                cache_update.apply(index.nav_cache());
            }

            // Verify navigation info is populated
            let nav_info = nav_cache.get(embedding).expect("Nav info should exist");
            assert!(nav_info.entry_point().is_some(), "Should have entry point");
        }

        // Phase 2: Reopen with fresh cache and verify search works
        {
            let storage = create_test_storage(temp_dir.path());
            let nav_cache = Arc::new(NavigationCache::new());

            // Cache should be empty (cold start)
            assert!(
                nav_cache.get(embedding).is_none(),
                "Fresh cache should be empty"
            );

            let spec = make_test_spec(dim, Distance::L2);
            let index = hnsw::Index::from_spec(embedding, &spec, 64, nav_cache);

            // Search should work - it loads navigation info from storage
            let results = index
                .search(&storage, &vectors[0], 5, 50)
                .expect("Search should succeed after recovery");

            // Should find results
            assert!(!results.is_empty(), "Should find some results");

            // First result should be vec_id 0 (searching for vector[0])
            let (dist, vec_id) = results[0];
            assert_eq!(vec_id, 0, "Should find exact match first");
            assert!(dist < 0.001, "Distance to self should be ~0");
        }
    }

    #[test]
    fn test_hnsw_entry_point_persists() {
        // Verify HNSW entry point and max_layer persist correctly
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let embedding: EmbeddingCode = 1;

        // Phase 1: Insert and verify entry point is stored
        let expected_entry_point: VecId;
        {
            let storage = create_test_storage(temp_dir.path());
            let txn_db = storage.transaction_db().expect("Failed to get txn_db");

            // Store a vector (using transaction)
            let cf = txn_db
                .cf_handle(Vectors::CF_NAME)
                .expect("Vectors CF not found");
            let key = Vectors::key_to_bytes(&VectorCfKey(embedding, 0));
            let vector = random_vector(64, 42);
            let value = Vectors::value_to_bytes_typed(
                &vector,
                crate::vector::schema::VectorElementType::F32,
            );
            let txn = txn_db.transaction();
            txn.put_cf(&cf, key, value).expect("Failed to store");
            txn.commit().expect("Failed to commit");

            // Build HNSW index
            let nav_cache = Arc::new(NavigationCache::new());
            let spec = make_test_spec(64, Distance::L2);
            let index = hnsw::Index::from_spec(embedding, &spec, 64, nav_cache.clone());

            let txn = txn_db.transaction();
            let cache_update = insert(&index, &txn, &txn_db, &storage, 0, &vector)
                .expect("Failed to insert");
            txn.commit().expect("Failed to commit");
            cache_update.apply(index.nav_cache());

            expected_entry_point = nav_cache
                .get(embedding)
                .expect("Nav info should exist")
                .entry_point()
                .expect("Should have entry point");
        }

        // Phase 2: Verify entry point persisted
        {
            let storage = create_test_storage(temp_dir.path());
            let txn_db = storage.transaction_db().expect("Failed to get txn_db");

            let cf = txn_db
                .cf_handle(GraphMeta::CF_NAME)
                .expect("GraphMeta CF not found");

            let ep_key = GraphMetaCfKey::entry_point(embedding);
            let bytes = txn_db
                .get_cf(&cf, GraphMeta::key_to_bytes(&ep_key))
                .expect("Failed to get")
                .expect("Entry point should exist");

            let value = GraphMeta::value_from_bytes(&ep_key, &bytes).expect("Failed to deserialize");
            match value.0 {
                GraphMetaField::EntryPoint(ep) => {
                    assert_eq!(ep, expected_entry_point, "Entry point should match");
                }
                _ => panic!("Expected EntryPoint field"),
            }
        }
    }

    // =========================================================================
    // Full Integration Test
    // =========================================================================

    #[test]
    fn test_full_crash_recovery_scenario() {
        // Simulates a full crash recovery scenario:
        // 1. Insert multiple vectors
        // 2. Build HNSW index
        // 3. "Crash" (close storage)
        // 4. Recover and verify search still works

        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let embedding: EmbeddingCode = 1;
        let dim = 64;
        let n_vectors = 50;

        // Generate test vectors (same seed = same vectors)
        let vectors: Vec<Vec<f32>> = (0..n_vectors)
            .map(|i| random_vector(dim, i as u64))
            .collect();

        // Phase 1: Build index
        {
            let storage = create_test_storage(temp_dir.path());

            // Store vectors (using transaction)
            let txn_db = storage.transaction_db().expect("Failed to get txn_db");
            let cf = txn_db
                .cf_handle(Vectors::CF_NAME)
                .expect("Vectors CF not found");

            let txn = txn_db.transaction();
            for (i, vector) in vectors.iter().enumerate() {
                let key = Vectors::key_to_bytes(&VectorCfKey(embedding, i as VecId));
                let value = Vectors::value_to_bytes_typed(
                    vector,
                    crate::vector::schema::VectorElementType::F32,
                );
                txn.put_cf(&cf, key, value).expect("Failed to store");
            }
            txn.commit().expect("Failed to commit vectors");

            // Build HNSW
            let nav_cache = Arc::new(NavigationCache::new());
            let spec = make_test_spec(dim, Distance::L2);
            let index = hnsw::Index::from_spec(embedding, &spec, 64, nav_cache);

            for (i, vector) in vectors.iter().enumerate() {
                let txn = txn_db.transaction();
                let cache_update =
                    insert(&index, &txn, &txn_db, &storage, i as VecId, vector)
                        .expect("Failed to insert");
                txn.commit().expect("Failed to commit");
                cache_update.apply(index.nav_cache());
            }
        }

        // Phase 2: "Crash" and recover
        {
            let storage = create_test_storage(temp_dir.path());
            let nav_cache = Arc::new(NavigationCache::new());
            let spec = make_test_spec(dim, Distance::L2);
            let index = hnsw::Index::from_spec(embedding, &spec, 64, nav_cache.clone());

            // Search for the first vector (should find itself)
            let results = index
                .search(&storage, &vectors[0], 5, 50)
                .expect("Search should succeed");

            assert!(!results.is_empty(), "Should find results after recovery");

            // The first result should be vec_id 0 (exact match)
            let (_, first_id) = results[0];
            assert_eq!(first_id, 0, "Should find exact match first");
        }
    }

    #[test]
    fn test_transaction_insert_with_recovery() {
        // Test using insert directly with recovery
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let embedding: EmbeddingCode = 1;
        let dim = 64;

        let vector = random_vector(dim, 42);

        // Phase 1: Insert using transaction API
        {
            let storage = create_test_storage(temp_dir.path());
            let txn_db = storage.transaction_db().expect("Failed to get txn_db");

            // First store the vector (using transaction)
            let cf = txn_db
                .cf_handle(Vectors::CF_NAME)
                .expect("Vectors CF not found");
            let key = Vectors::key_to_bytes(&VectorCfKey(embedding, 0));
            let value = Vectors::value_to_bytes_typed(
                &vector,
                crate::vector::schema::VectorElementType::F32,
            );
            {
                let txn = txn_db.transaction();
                txn.put_cf(&cf, key, value).expect("Failed to store");
                txn.commit().expect("Failed to commit vector");
            }

            // Now insert into HNSW using transaction API
            let nav_cache = Arc::new(NavigationCache::new());
            let spec = make_test_spec(dim, Distance::L2);
            let index = hnsw::Index::from_spec(embedding, &spec, 64, nav_cache.clone());

            let txn = txn_db.transaction();
            let cache_update =
                insert(&index, &txn, &txn_db, &storage, 0, &vector).expect("Insert failed");

            // Commit
            txn.commit().expect("Commit failed");

            // Apply cache update AFTER commit
            cache_update.apply(&nav_cache);
        }

        // Phase 2: Verify recovery
        {
            let storage = create_test_storage(temp_dir.path());
            let nav_cache = Arc::new(NavigationCache::new());
            let spec = make_test_spec(dim, Distance::L2);
            let index = hnsw::Index::from_spec(embedding, &spec, 64, nav_cache);

            // Search should find the vector
            let results = index
                .search(&storage, &vector, 1, 50)
                .expect("Search should succeed");

            assert_eq!(results.len(), 1, "Should find exactly one result");
            assert_eq!(results[0].1, 0, "Should find vec_id 0");
            assert!(results[0].0 < 0.001, "Distance should be ~0 for exact match");
        }
    }
}
