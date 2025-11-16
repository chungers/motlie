/// Tests for NodeNames and EdgeNames prefix scanning functionality
///
/// This test suite verifies that the current design correctly supports
/// true prefix scanning (e.g., "Shop" matches "Shopping", "Shopper", "Shop.com").
///
/// Key Design:
/// - NodeNamesCfKey: (NodeName, Id) where NodeName is variable-length String
/// - EdgeNamesCfKey: (EdgeName, Id, DstId, SrcId) where EdgeName is variable-length String
///
/// Layout: [name_bytes (variable)] + [id_bytes (16 or 48)]
///
/// The variable-length name at the START is REQUIRED for prefix scanning to work.
/// Keys are ordered lexicographically by name, allowing RocksDB to:
/// 1. B-tree seek to first key >= prefix (O(log N))
/// 2. Scan matching keys sequentially (O(k))
///
/// See: libs/db/docs/prefix-scanning-analysis-final.md

use motlie_db::{AddEdge, AddNode, Id, TimestampMilli};
use std::time::Duration;
use tempfile::TempDir;

/// Test comprehensive prefix scanning for node names
///
/// Verifies that searching for "Shop" returns all nodes with names starting with "Shop":
/// - "Shop" (exact match)
/// - "Shop.com" (prefix with punctuation)
/// - "Shopper Johnson" (longer name with space)
/// - "Shopping" (simple extension)
/// - "Shopping Mall" (extension with space)
/// - "Shoppify" (another variation)
///
/// Does NOT return:
/// - "Amazon" (different prefix)
/// - "Sho" (shorter than search prefix)
#[tokio::test]
async fn test_node_names_prefix_scan_comprehensive() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().to_path_buf();

    // Create writer
    let (writer, writer_rx) = motlie_db::create_mutation_writer(Default::default());

    // Spawn write consumer
    let writer_handle =
        motlie_db::spawn_graph_consumer(writer_rx, Default::default(), &db_path);

    // Create nodes with various name patterns
    let nodes = vec![
        ("Shop", Id::new()),                // Exact match
        ("Shop.com", Id::new()),            // With punctuation
        ("Shopper Johnson", Id::new()),     // Longer with space
        ("Shopping", Id::new()),            // Simple extension
        ("Shopping Mall", Id::new()),       // Extension with space
        ("Shoppify", Id::new()),            // Another variation
        ("Amazon", Id::new()),              // Different prefix (control)
        ("Sho", Id::new()),                 // Shorter than prefix (control)
        ("apple", Id::new()),               // Lowercase, different (control)
    ];

    // Add all nodes
    for (name, id) in &nodes {
        writer
            .add_node(AddNode {
                id: *id,
                name: name.to_string(),
                ts_millis: TimestampMilli(1000),
            })
            .await
            .expect("Failed to add node");
    }

    // Give time for writes to complete
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Drop writer and wait for consumer to finish
    drop(writer);
    writer_handle.await.unwrap().unwrap();

    println!("\n=== Testing NodeNames prefix scan with 'Shop' ===");

    // Create query consumer
    let (reader, reader_rx) = motlie_db::create_query_reader(Default::default());
    let _reader_handle =
        motlie_db::spawn_query_consumer(reader_rx, Default::default(), &db_path);

    // Query for nodes starting with "Shop"
    let result = reader
        .nodes_by_name("Shop".to_string(), None, Some(100), Duration::from_secs(5))
        .await
        .expect("Query should succeed");

    println!("✅ Query succeeded, found {} results:", result.len());
    for (name, _id) in &result {
        println!("  - {}", name);
    }

    // Verify all results start with "Shop"
    for (name, _) in &result {
        assert!(
            name.starts_with("Shop"),
            "Result '{}' does not start with 'Shop'",
            name
        );
    }

    // We should find exactly 6 nodes that start with "Shop"
    assert_eq!(
        result.len(),
        6,
        "Expected 6 results, but got {}",
        result.len()
    );

    // Verify specific names are present
    let result_names: Vec<String> = result.iter().map(|(name, _)| name.clone()).collect();
    assert!(result_names.contains(&"Shop".to_string()));
    assert!(result_names.contains(&"Shop.com".to_string()));
    assert!(result_names.contains(&"Shopper Johnson".to_string()));
    assert!(result_names.contains(&"Shopping".to_string()));
    assert!(result_names.contains(&"Shopping Mall".to_string()));
    assert!(result_names.contains(&"Shoppify".to_string()));

    // Verify control cases are NOT present
    assert!(!result_names.contains(&"Amazon".to_string()));
    assert!(!result_names.contains(&"Sho".to_string()));
    assert!(!result_names.contains(&"apple".to_string()));
}

/// Test edge cases for node name prefix scanning
#[tokio::test]
async fn test_node_names_edge_cases() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().to_path_buf();

    let (writer, writer_rx) = motlie_db::create_mutation_writer(Default::default());
    let writer_handle =
        motlie_db::spawn_graph_consumer(writer_rx, Default::default(), &db_path);

    // Edge cases
    let nodes = vec![
        ("a", Id::new()),               // Single character
        ("ab", Id::new()),              // Two characters
        ("abc", Id::new()),             // Three characters
        ("abcd", Id::new()),            // Four characters
        ("b", Id::new()),               // Different single char (control)
        ("VeryLongNodeNameWithManyCharactersToTestVariableLengthHandling", Id::new()), // Very long
        ("UTF8🔥test", Id::new()),      // UTF-8 with emoji
        ("UTF8🔥test2", Id::new()),     // UTF-8 variant
    ];

    for (name, id) in &nodes {
        writer
            .add_node(AddNode {
                id: *id,
                name: name.to_string(),
                ts_millis: TimestampMilli(1000),
            })
            .await
            .expect("Failed to add node");
    }

    tokio::time::sleep(Duration::from_millis(100)).await;
    drop(writer);
    writer_handle.await.unwrap().unwrap();

    let (reader, reader_rx) = motlie_db::create_query_reader(Default::default());
    let _reader_handle =
        motlie_db::spawn_query_consumer(reader_rx, Default::default(), &db_path);

    // Test 1: Single character prefix
    let result = reader
        .nodes_by_name("a".to_string(), None, Some(100), Duration::from_secs(5))
        .await
        .expect("Query should succeed");

    assert_eq!(result.len(), 4, "Should find 4 nodes starting with 'a'");
    let names: Vec<String> = result.iter().map(|(n, _)| n.clone()).collect();
    assert!(names.contains(&"a".to_string()));
    assert!(names.contains(&"ab".to_string()));
    assert!(names.contains(&"abc".to_string()));
    assert!(names.contains(&"abcd".to_string()));

    // Test 2: Very long name prefix
    let result = reader
        .nodes_by_name("VeryLong".to_string(), None, Some(100), Duration::from_secs(5))
        .await
        .expect("Query should succeed");

    assert_eq!(result.len(), 1, "Should find 1 node with very long name");

    // Test 3: UTF-8 with emoji
    let result = reader
        .nodes_by_name("UTF8🔥".to_string(), None, Some(100), Duration::from_secs(5))
        .await
        .expect("Query should succeed");

    assert_eq!(result.len(), 2, "Should find 2 nodes with UTF-8 emoji prefix");
}

/// Test pagination for node name prefix scanning
#[tokio::test]
async fn test_node_names_pagination() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().to_path_buf();

    let (writer, writer_rx) = motlie_db::create_mutation_writer(Default::default());
    let writer_handle =
        motlie_db::spawn_graph_consumer(writer_rx, Default::default(), &db_path);

    // Create 20 nodes with same prefix but different names
    for i in 0..20 {
        writer
            .add_node(AddNode {
                id: Id::new(),
                name: format!("test_{:03}", i),
                ts_millis: TimestampMilli(1000),
            })
            .await
            .expect("Failed to add node");
    }

    tokio::time::sleep(Duration::from_millis(100)).await;
    drop(writer);
    writer_handle.await.unwrap().unwrap();

    let (reader, reader_rx) = motlie_db::create_query_reader(Default::default());
    let _reader_handle =
        motlie_db::spawn_query_consumer(reader_rx, Default::default(), &db_path);

    // Page 1: Get first 10
    let page1 = reader
        .nodes_by_name("test_".to_string(), None, Some(10), Duration::from_secs(5))
        .await
        .expect("Query should succeed");

    println!("\n=== Page 1 ===");
    for (name, id) in &page1 {
        println!("  {} - {}", name, id);
    }

    assert_eq!(page1.len(), 10, "First page should have 10 results");

    // Page 2: Get next 10 using last result from page 1
    let (last_name, last_id) = page1.last().unwrap().clone();
    println!("\n=== Seeking from: {} - {} ===", last_name, last_id);

    let page2 = reader
        .nodes_by_name(
            "test_".to_string(),
            Some((last_name.clone(), last_id)),
            Some(10),
            Duration::from_secs(5),
        )
        .await
        .expect("Query should succeed");

    println!("\n=== Page 2 ===");
    for (name, id) in &page2 {
        println!("  {} - {}", name, id);
    }

    assert_eq!(page2.len(), 10, "Second page should have 10 results");

    // Verify no overlap between pages
    let page1_ids: Vec<Id> = page1.iter().map(|(_, id)| *id).collect();
    let page2_ids: Vec<Id> = page2.iter().map(|(_, id)| *id).collect();

    for id in &page2_ids {
        assert!(
            !page1_ids.contains(id),
            "Page 2 should not contain IDs from page 1. Found duplicate: {}",
            id
        );
    }

    // Verify page 2 names are lexicographically after page 1
    let last_page1_name = &page1.last().unwrap().0;
    let first_page2_name = &page2.first().unwrap().0;
    assert!(
        first_page2_name > last_page1_name,
        "Page 2 first name '{}' should be after page 1 last name '{}'",
        first_page2_name,
        last_page1_name
    );

    // Get all in one query to verify total
    let all = reader
        .nodes_by_name("test_".to_string(), None, Some(100), Duration::from_secs(5))
        .await
        .expect("Query should succeed");

    assert_eq!(all.len(), 20, "Should have 20 total results");

    // Verify all names start with the prefix
    for (name, _) in &all {
        assert!(name.starts_with("test_"));
    }
}

/// Test edge name prefix scanning
#[tokio::test]
async fn test_edge_names_prefix_scan_comprehensive() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().to_path_buf();

    let (writer, writer_rx) = motlie_db::create_mutation_writer(Default::default());
    let writer_handle =
        motlie_db::spawn_graph_consumer(writer_rx, Default::default(), &db_path);

    // Create nodes
    let src_node = Id::new();
    let dst_node = Id::new();

    writer
        .add_node(AddNode {
            id: src_node,
            name: "Source".to_string(),
            ts_millis: TimestampMilli(1000),
        })
        .await
        .expect("Failed to add node");

    writer
        .add_node(AddNode {
            id: dst_node,
            name: "Destination".to_string(),
            ts_millis: TimestampMilli(1000),
        })
        .await
        .expect("Failed to add node");

    // Create edges with various names
    let edges = vec![
        ("payment", Id::new()),             // Exact match
        ("payment_gateway", Id::new()),     // With underscore
        ("payment_processing", Id::new()),  // Longer variant
        ("pay", Id::new()),                 // Shorter than prefix
        ("payroll_system", Id::new()),      // Different word
        ("pays", Id::new()),                // Simple extension
        ("connection", Id::new()),          // Different prefix (control)
    ];

    for (name, id) in &edges {
        writer
            .add_edge(AddEdge {
                id: *id,
                source_node_id: src_node,
                target_node_id: dst_node,
                name: motlie_db::EdgeName(name.to_string()),
                ts_millis: TimestampMilli(2000),
            })
            .await
            .expect("Failed to add edge");
    }

    tokio::time::sleep(Duration::from_millis(100)).await;
    drop(writer);
    writer_handle.await.unwrap().unwrap();

    println!("\n=== Testing EdgeNames prefix scan with 'pay' ===");

    let (reader, reader_rx) = motlie_db::create_query_reader(Default::default());
    let _reader_handle =
        motlie_db::spawn_query_consumer(reader_rx, Default::default(), &db_path);

    // Query for edges starting with "pay"
    let result = reader
        .edges_by_name("pay".to_string(), None, Some(100), Duration::from_secs(5))
        .await
        .expect("Query should succeed");

    println!("✅ Query succeeded, found {} results:", result.len());
    for (name, _id) in &result {
        println!("  - {}", name.0);
    }

    // Verify all results start with "pay"
    for (name, _) in &result {
        assert!(
            name.0.starts_with("pay"),
            "Result '{}' does not start with 'pay'",
            name.0
        );
    }

    // Should find 6 edges starting with "pay"
    assert_eq!(
        result.len(),
        6,
        "Expected 6 results, but got {}",
        result.len()
    );

    // Verify specific names
    let result_names: Vec<String> = result.iter().map(|(name, _)| name.0.clone()).collect();
    assert!(result_names.contains(&"pay".to_string()));
    assert!(result_names.contains(&"payment".to_string()));
    assert!(result_names.contains(&"payment_gateway".to_string()));
    assert!(result_names.contains(&"payment_processing".to_string()));
    assert!(result_names.contains(&"payroll_system".to_string()));
    assert!(result_names.contains(&"pays".to_string()));

    // Control case should NOT be present
    assert!(!result_names.contains(&"connection".to_string()));
}

/// Test that demonstrates current design handles variable-length keys correctly
#[test]
fn test_key_serialization_correctness() {
    use motlie_db::Id;

    // Simulate how NodeNamesCfKey is serialized
    let serialize_node_key = |name: &str, id: Id| -> Vec<u8> {
        let name_bytes = name.as_bytes();
        let mut bytes = Vec::with_capacity(name_bytes.len() + 16);
        bytes.extend_from_slice(name_bytes);
        bytes.extend_from_slice(&id.into_bytes());
        bytes
    };

    let id1 = Id::new();
    let id2 = Id::new();
    let id3 = Id::new();

    let key1 = serialize_node_key("Shop", id1);
    let key2 = serialize_node_key("Shopping", id2);
    let key3 = serialize_node_key("Shopping Mall", id3);

    // All keys have different total lengths
    assert_eq!(key1.len(), 4 + 16); // "Shop" + ID
    assert_eq!(key2.len(), 8 + 16); // "Shopping" + ID
    assert_eq!(key3.len(), 13 + 16); // "Shopping Mall" + ID

    // Verify that keys are lexicographically ordered by name
    assert!(key1 < key2); // "Shop" < "Shopping"
    assert!(key2 < key3); // "Shopping" < "Shopping Mall"

    // Verify deserialization always works
    let deserialize_node_key = |bytes: &[u8]| -> (String, Id) {
        let name_end = bytes.len() - 16;
        let name = String::from_utf8(bytes[0..name_end].to_vec()).unwrap();
        let mut id_bytes = [0u8; 16];
        id_bytes.copy_from_slice(&bytes[name_end..name_end + 16]);
        (name, Id::from_bytes(id_bytes))
    };

    let (name1, extracted_id1) = deserialize_node_key(&key1);
    assert_eq!(name1, "Shop");
    assert_eq!(extracted_id1, id1);

    let (name2, extracted_id2) = deserialize_node_key(&key2);
    assert_eq!(name2, "Shopping");
    assert_eq!(extracted_id2, id2);

    let (name3, extracted_id3) = deserialize_node_key(&key3);
    assert_eq!(name3, "Shopping Mall");
    assert_eq!(extracted_id3, id3);
}

/// Test prefix scanning with empty prefix (should return all)
#[tokio::test]
async fn test_empty_prefix_returns_all() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().to_path_buf();

    let (writer, writer_rx) = motlie_db::create_mutation_writer(Default::default());
    let writer_handle =
        motlie_db::spawn_graph_consumer(writer_rx, Default::default(), &db_path);

    // Add some nodes
    for name in &["apple", "banana", "cherry"] {
        writer
            .add_node(AddNode {
                id: Id::new(),
                name: name.to_string(),
                ts_millis: TimestampMilli(1000),
            })
            .await
            .expect("Failed to add node");
    }

    tokio::time::sleep(Duration::from_millis(100)).await;
    drop(writer);
    writer_handle.await.unwrap().unwrap();

    let (reader, reader_rx) = motlie_db::create_query_reader(Default::default());
    let _reader_handle =
        motlie_db::spawn_query_consumer(reader_rx, Default::default(), &db_path);

    // Query with empty prefix
    let result = reader
        .nodes_by_name("".to_string(), None, Some(100), Duration::from_secs(5))
        .await
        .expect("Query should succeed");

    // Empty prefix matches all
    assert_eq!(result.len(), 3, "Empty prefix should match all nodes");
}

/// Test prefix scanning with non-existent prefix
#[tokio::test]
async fn test_nonexistent_prefix_returns_empty() {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().to_path_buf();

    let (writer, writer_rx) = motlie_db::create_mutation_writer(Default::default());
    let writer_handle =
        motlie_db::spawn_graph_consumer(writer_rx, Default::default(), &db_path);

    // Add some nodes
    for name in &["apple", "banana", "cherry"] {
        writer
            .add_node(AddNode {
                id: Id::new(),
                name: name.to_string(),
                ts_millis: TimestampMilli(1000),
            })
            .await
            .expect("Failed to add node");
    }

    tokio::time::sleep(Duration::from_millis(100)).await;
    drop(writer);
    writer_handle.await.unwrap().unwrap();

    let (reader, reader_rx) = motlie_db::create_query_reader(Default::default());
    let _reader_handle =
        motlie_db::spawn_query_consumer(reader_rx, Default::default(), &db_path);

    // Query with non-existent prefix
    let result = reader
        .nodes_by_name("zebra".to_string(), None, Some(100), Duration::from_secs(5))
        .await
        .expect("Query should succeed");

    assert_eq!(
        result.len(),
        0,
        "Non-existent prefix should return no results"
    );
}
