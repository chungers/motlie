//! Integration test: Mixed ExternalKey type operations
//!
//! T9.2: Tests for polymorphic ExternalKey support across all variants.
//!
//! This test demonstrates that all ExternalKey variants work correctly:
//! 1. Insert vectors with different ExternalKey types (NodeId, NodeFragment, Edge, etc.)
//! 2. Search and verify results return the correct ExternalKey variant
//! 3. Delete vectors and verify the correct variant is removed
//! 4. Validate forward/reverse ID mapping consistency for all types

use std::sync::Arc;
use std::time::Duration;

use motlie_db::graph::{NameHash, SummaryHash};
use motlie_db::vector::{
    create_reader_with_storage, create_writer,
    spawn_mutation_consumer_with_storage_autoreg, spawn_query_consumers_with_storage_autoreg,
    DeleteVector, Distance, EmbeddingBuilder, ExternalKey, InsertVector, MutationRunnable,
    ReaderConfig, Runnable, SearchKNN, Storage, WriterConfig,
};
use motlie_db::{Id, TimestampMilli};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use tempfile::TempDir;

const DIM: u32 = 64; // Small dimension for fast tests
const SEARCH_TIMEOUT: Duration = Duration::from_secs(5);

/// Generate a random vector for testing.
fn generate_vector(dim: u32, rng: &mut impl Rng) -> Vec<f32> {
    (0..dim).map(|_| rng.gen::<f32>()).collect()
}

#[tokio::test]
async fn test_mixed_external_key_types() {
    // =========================================================================
    // Setup: Create storage
    // =========================================================================
    let temp_dir = TempDir::new().expect("temp dir");
    let mut storage = Storage::readwrite(temp_dir.path());
    storage.ready().expect("storage ready");
    let storage = Arc::new(storage);

    // Get registry from storage cache
    let registry = storage.cache().clone();
    registry.set_storage(storage.clone()).expect("set storage");

    // Register embedding
    let builder = EmbeddingBuilder::new("test-mixed-key", DIM, Distance::L2);
    let embedding = registry.register(builder).expect("register embedding");

    // Create writer and mutation consumer
    let (writer, writer_rx) = create_writer(WriterConfig::default());
    let _mutation_handle = spawn_mutation_consumer_with_storage_autoreg(
        writer_rx,
        WriterConfig::default(),
        storage.clone(),
    );

    // Create reader and consumers
    let (search_reader, search_rx) =
        create_reader_with_storage(ReaderConfig::default());
    let _query_handles = spawn_query_consumers_with_storage_autoreg(
        search_rx,
        ReaderConfig::default(),
        storage.clone(),
        1, // single reader
    );

    // =========================================================================
    // Generate test data with different ExternalKey types
    // =========================================================================
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Create test keys of each type
    let id1 = Id::new();
    let id2 = Id::new();
    let id3 = Id::new();

    // Vector 1: NodeId
    let node_id_key = ExternalKey::NodeId(id1);
    let node_id_vec = generate_vector(DIM, &mut rng);

    // Vector 2: NodeFragment
    let node_fragment_key = ExternalKey::NodeFragment(id2, TimestampMilli(1234567890));
    let node_fragment_vec = generate_vector(DIM, &mut rng);

    // Vector 3: Edge
    let edge_key = ExternalKey::Edge(id1, id2, NameHash::from_name("relates_to"));
    let edge_vec = generate_vector(DIM, &mut rng);

    // Vector 4: EdgeFragment
    let edge_fragment_key = ExternalKey::EdgeFragment(
        id2,
        id3,
        NameHash::from_name("contains"),
        TimestampMilli(9876543210),
    );
    let edge_fragment_vec = generate_vector(DIM, &mut rng);

    // Vector 5: NodeSummary
    let node_summary_key = ExternalKey::NodeSummary(
        SummaryHash::from_summary(&"Test node summary content".to_string()).unwrap(),
    );
    let node_summary_vec = generate_vector(DIM, &mut rng);

    // Vector 6: EdgeSummary
    let edge_summary_key = ExternalKey::EdgeSummary(
        SummaryHash::from_summary(&"Test edge summary content".to_string()).unwrap(),
    );
    let edge_summary_vec = generate_vector(DIM, &mut rng);

    // =========================================================================
    // Insert vectors with different key types
    // =========================================================================
    let all_keys = vec![
        (node_id_key.clone(), node_id_vec.clone()),
        (node_fragment_key.clone(), node_fragment_vec.clone()),
        (edge_key.clone(), edge_vec.clone()),
        (edge_fragment_key.clone(), edge_fragment_vec.clone()),
        (node_summary_key.clone(), node_summary_vec.clone()),
        (edge_summary_key.clone(), edge_summary_vec.clone()),
    ];

    for (key, vec) in &all_keys {
        InsertVector::new(&embedding, key.clone(), vec.clone())
            .immediate()
            .run(&writer)
            .await
            .expect(&format!("insert {:?}", key.variant_name()));
    }

    writer.flush().await.expect("flush");

    // =========================================================================
    // Verify all keys can be found via search
    // =========================================================================
    for (query_key, query_vec) in &all_keys {
        let results = SearchKNN::new(&embedding, query_vec.clone(), 10)
            .run(&search_reader, SEARCH_TIMEOUT)
            .await
            .expect(&format!("search for {:?}", query_key.variant_name()));

        // The exact match should be in the results
        let found = results.iter().any(|r| &r.external_key == query_key);
        assert!(
            found,
            "Expected to find {:?} in search results, got: {:?}",
            query_key,
            results
                .iter()
                .map(|r| &r.external_key)
                .collect::<Vec<_>>()
        );

        // Verify variant_name() method works correctly
        let first_result = results.first().expect("at least one result");
        let variant = first_result.external_key.variant_name();
        assert!(
            !variant.is_empty(),
            "variant_name() should return non-empty string"
        );
    }

    // =========================================================================
    // Test SearchResult.node_id() accessor for NodeId key
    // =========================================================================
    let results = SearchKNN::new(&embedding, node_id_vec.clone(), 1)
        .run(&search_reader, SEARCH_TIMEOUT)
        .await
        .expect("search for NodeId");

    let first = results.first().expect("at least one result");
    assert_eq!(
        first.external_key.variant_name(),
        "NodeId",
        "First result should be NodeId"
    );
    assert_eq!(
        first.node_id(),
        Some(id1),
        "node_id() should return the correct Id"
    );

    // =========================================================================
    // Test that node_id() returns None for non-NodeId keys
    // =========================================================================
    let results = SearchKNN::new(&embedding, edge_vec.clone(), 1)
        .run(&search_reader, SEARCH_TIMEOUT)
        .await
        .expect("search for Edge");

    let first = results.first().expect("at least one result");
    assert_eq!(
        first.external_key.variant_name(),
        "Edge",
        "First result should be Edge"
    );
    assert_eq!(
        first.node_id(),
        None,
        "node_id() should return None for Edge key"
    );

    // =========================================================================
    // Test delete with different key types
    // =========================================================================
    // Delete the NodeFragment
    DeleteVector::new(&embedding, node_fragment_key.clone())
        .run(&writer)
        .await
        .expect("delete NodeFragment");

    writer.flush().await.expect("flush after delete");

    // Verify it's no longer found
    let results = SearchKNN::new(&embedding, node_fragment_vec.clone(), 10)
        .run(&search_reader, SEARCH_TIMEOUT)
        .await
        .expect("search after delete");

    let found = results
        .iter()
        .any(|r| &r.external_key == &node_fragment_key);
    assert!(
        !found,
        "Deleted NodeFragment key should not appear in results"
    );
}

#[tokio::test]
async fn test_external_key_variant_names() {
    // Test that all variant names are correct
    let id1 = Id::new();
    let id2 = Id::new();
    let ts = TimestampMilli(12345);
    let name_hash = NameHash::from_name("test");
    let summary_hash = SummaryHash::from_summary(&"test content".to_string()).unwrap();

    let test_cases = vec![
        (ExternalKey::NodeId(id1), "NodeId"),
        (ExternalKey::NodeFragment(id1, ts), "NodeFragment"),
        (ExternalKey::Edge(id1, id2, name_hash), "Edge"),
        (
            ExternalKey::EdgeFragment(id1, id2, name_hash, ts),
            "EdgeFragment",
        ),
        (ExternalKey::NodeSummary(summary_hash), "NodeSummary"),
        (ExternalKey::EdgeSummary(summary_hash), "EdgeSummary"),
    ];

    for (key, expected_name) in test_cases {
        assert_eq!(
            key.variant_name(),
            expected_name,
            "variant_name() for {:?}",
            expected_name
        );
    }
}

#[tokio::test]
async fn test_external_key_serialization_roundtrip() {
    // Test that all ExternalKey variants serialize and deserialize correctly
    let id1 = Id::new();
    let id2 = Id::new();
    let ts = TimestampMilli(9876543210);
    let name_hash = NameHash::from_name("edge_name");
    let summary_hash = SummaryHash::from_summary(&"summary content".to_string()).unwrap();

    let test_keys = vec![
        ExternalKey::NodeId(id1),
        ExternalKey::NodeFragment(id1, ts),
        ExternalKey::Edge(id1, id2, name_hash),
        ExternalKey::EdgeFragment(id1, id2, name_hash, ts),
        ExternalKey::NodeSummary(summary_hash),
        ExternalKey::EdgeSummary(summary_hash),
    ];

    for key in test_keys {
        // Test binary serialization
        let bytes = key.to_bytes();
        let parsed = ExternalKey::from_bytes(&bytes).expect(&format!(
            "from_bytes should succeed for {:?}",
            key.variant_name()
        ));
        assert_eq!(key, parsed, "Binary roundtrip for {:?}", key.variant_name());

        // Test JSON serialization (via serde)
        let json = serde_json::to_string(&key).expect("JSON serialize");
        let from_json: ExternalKey = serde_json::from_str(&json).expect("JSON deserialize");
        assert_eq!(
            key, from_json,
            "JSON roundtrip for {:?}",
            key.variant_name()
        );
    }
}
