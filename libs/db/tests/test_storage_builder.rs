//! Integration tests for StorageBuilder with graph and vector providers.
//!
//! These tests verify that graph::Schema and vector::Schema can share
//! a single RocksDB TransactionDB instance via StorageBuilder.

use std::sync::Arc;

use motlie_db::graph;
use motlie_db::storage_builder::{CacheConfig, SharedStorage, StorageBuilder};
use motlie_db::vector;
use tempfile::TempDir;

/// Helper to create a shared storage with both graph and vector providers
fn create_shared_storage(temp_dir: &TempDir) -> SharedStorage {
    let db_path = temp_dir.path().join("shared_db");

    StorageBuilder::new(&db_path)
        .with_provider(Box::new(graph::Schema::new()))
        .with_provider(Box::new(vector::Schema::new()))
        .with_cache_size(64 * 1024 * 1024) // 64MB for tests
        .build()
        .expect("Failed to build shared storage")
}

#[test]
fn test_shared_storage_creation() {
    let temp_dir = TempDir::new().unwrap();
    let storage = create_shared_storage(&temp_dir);

    // Verify both providers are registered
    assert!(storage.get_provider("graph").is_some());
    assert!(storage.get_provider("vector").is_some());

    let names = storage.provider_names();
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"graph"));
    assert!(names.contains(&"vector"));
}

#[test]
fn test_shared_storage_all_cfs_created() {
    let temp_dir = TempDir::new().unwrap();
    let storage = create_shared_storage(&temp_dir);

    let db = storage.db().expect("Should have DB");

    // Check graph CFs exist
    assert!(db.cf_handle("names").is_some(), "names CF should exist");
    assert!(db.cf_handle("nodes").is_some(), "nodes CF should exist");
    assert!(
        db.cf_handle("forward_edges").is_some(),
        "forward_edges CF should exist"
    );
    assert!(
        db.cf_handle("reverse_edges").is_some(),
        "reverse_edges CF should exist"
    );
    assert!(
        db.cf_handle("node_fragments").is_some(),
        "node_fragments CF should exist"
    );
    assert!(
        db.cf_handle("edge_fragments").is_some(),
        "edge_fragments CF should exist"
    );
    assert!(
        db.cf_handle("node_summaries").is_some(),
        "node_summaries CF should exist"
    );
    assert!(
        db.cf_handle("edge_summaries").is_some(),
        "edge_summaries CF should exist"
    );

    // Check vector CFs exist
    assert!(
        db.cf_handle("vector/embedding_specs").is_some(),
        "vector/embedding_specs CF should exist"
    );
    assert!(
        db.cf_handle("vector/vectors").is_some(),
        "vector/vectors CF should exist"
    );
    assert!(
        db.cf_handle("vector/edges").is_some(),
        "vector/edges CF should exist"
    );
    assert!(
        db.cf_handle("vector/binary_codes").is_some(),
        "vector/binary_codes CF should exist"
    );
    assert!(
        db.cf_handle("vector/vec_meta").is_some(),
        "vector/vec_meta CF should exist"
    );
    assert!(
        db.cf_handle("vector/graph_meta").is_some(),
        "vector/graph_meta CF should exist"
    );
    assert!(
        db.cf_handle("vector/id_forward").is_some(),
        "vector/id_forward CF should exist"
    );
    assert!(
        db.cf_handle("vector/id_reverse").is_some(),
        "vector/id_reverse CF should exist"
    );
    assert!(
        db.cf_handle("vector/id_alloc").is_some(),
        "vector/id_alloc CF should exist"
    );
    assert!(
        db.cf_handle("vector/pending").is_some(),
        "vector/pending CF should exist"
    );
}

#[test]
fn test_shared_storage_cf_names_list() {
    let temp_dir = TempDir::new().unwrap();
    let storage = create_shared_storage(&temp_dir);

    let all_cfs = storage.all_cf_names();

    // Graph CFs (8)
    assert!(all_cfs.contains(&"names"));
    assert!(all_cfs.contains(&"nodes"));
    assert!(all_cfs.contains(&"forward_edges"));
    assert!(all_cfs.contains(&"reverse_edges"));
    assert!(all_cfs.contains(&"node_fragments"));
    assert!(all_cfs.contains(&"edge_fragments"));
    assert!(all_cfs.contains(&"node_summaries"));
    assert!(all_cfs.contains(&"edge_summaries"));

    // Vector CFs (10)
    assert!(all_cfs.contains(&"vector/embedding_specs"));
    assert!(all_cfs.contains(&"vector/vectors"));
    assert!(all_cfs.contains(&"vector/edges"));
    assert!(all_cfs.contains(&"vector/binary_codes"));
    assert!(all_cfs.contains(&"vector/vec_meta"));
    assert!(all_cfs.contains(&"vector/graph_meta"));
    assert!(all_cfs.contains(&"vector/id_forward"));
    assert!(all_cfs.contains(&"vector/id_reverse"));
    assert!(all_cfs.contains(&"vector/id_alloc"));
    assert!(all_cfs.contains(&"vector/pending"));

    // Total: 18 CFs
    assert_eq!(all_cfs.len(), 18);
}

#[test]
fn test_graph_schema_name_cache_accessible() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("shared_db");

    // Create with shared name cache
    let name_cache = Arc::new(graph::NameCache::new());
    let graph_schema = graph::Schema::with_name_cache(name_cache.clone());

    let storage = StorageBuilder::new(&db_path)
        .with_provider(Box::new(graph_schema))
        .with_provider(Box::new(vector::Schema::new()))
        .build()
        .expect("Failed to build shared storage");

    // Name cache should be accessible and empty (no data pre-warmed)
    assert_eq!(name_cache.len(), 0);

    // Verify storage is functional
    assert!(storage.get_provider("graph").is_some());
}

#[test]
fn test_vector_schema_registry_accessible() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("shared_db");

    // Create with shared registry
    let registry = Arc::new(vector::EmbeddingRegistry::new());
    let vector_schema = vector::Schema::with_registry(registry.clone());

    let storage = StorageBuilder::new(&db_path)
        .with_provider(Box::new(graph::Schema::new()))
        .with_provider(Box::new(vector_schema))
        .build()
        .expect("Failed to build shared storage");

    // Registry should be accessible and empty (no embeddings pre-warmed)
    assert!(registry.is_empty());

    // Verify storage is functional
    assert!(storage.get_provider("vector").is_some());
}

#[test]
fn test_cache_config_applied() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("shared_db");

    let config = CacheConfig {
        size_bytes: 128 * 1024 * 1024, // 128MB
        block_size: 8 * 1024,          // 8KB
    };

    let storage = StorageBuilder::new(&db_path)
        .with_provider(Box::new(graph::Schema::new()))
        .with_cache_config(config)
        .build()
        .expect("Failed to build storage");

    // Storage should be created successfully
    assert!(storage.get_provider("graph").is_some());
}

#[test]
fn test_db_clone_works() {
    let temp_dir = TempDir::new().unwrap();
    let storage = create_shared_storage(&temp_dir);

    // Clone should work
    let db1 = storage.db_clone().expect("Should have DB");
    let db2 = storage.db_clone().expect("Should have DB");

    // Both clones should access the same CFs
    assert!(db1.cf_handle("names").is_some());
    assert!(db2.cf_handle("vector/vectors").is_some());
}

#[test]
fn test_storage_path() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("my_shared_db");

    let storage = StorageBuilder::new(&db_path)
        .with_provider(Box::new(graph::Schema::new()))
        .build()
        .expect("Failed to build storage");

    assert_eq!(storage.path(), db_path);
}

#[test]
fn test_single_provider_works() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_only_db");

    // Graph only
    let storage = StorageBuilder::new(&db_path)
        .with_provider(Box::new(graph::Schema::new()))
        .build()
        .expect("Failed to build storage");

    assert!(storage.get_provider("graph").is_some());
    assert!(storage.get_provider("vector").is_none());

    let db = storage.db().expect("Should have DB");
    assert!(db.cf_handle("names").is_some());
    assert!(db.cf_handle("vector/vectors").is_none());
}

#[test]
fn test_vector_only_works() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("vector_only_db");

    // Vector only
    let storage = StorageBuilder::new(&db_path)
        .with_provider(Box::new(vector::Schema::new()))
        .build()
        .expect("Failed to build storage");

    assert!(storage.get_provider("vector").is_some());
    assert!(storage.get_provider("graph").is_none());

    let db = storage.db().expect("Should have DB");
    assert!(db.cf_handle("vector/vectors").is_some());
    assert!(db.cf_handle("names").is_none());
}
