//! Tests verifying Tantivy's document handling behavior for fragment semantics.
//!
//! These tests document and verify critical Tantivy behaviors:
//! 1. Tantivy does NOT have a built-in unique document ID - add_document always appends
//! 2. Multiple documents with the same field values are all retained and searchable
//! 3. Fragment append-only semantics work correctly for node/edge fragments
//! 4. delete_term requires INDEXED fields to work
//!
//! These tests serve as documentation and regression tests for our fragment indexing design.

use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{doc, Index, IndexWriter, ReloadPolicy};

/// Test that multiple documents with the same "id" field value are ALL retained.
/// Tantivy does NOT deduplicate by any field - it's purely append-only.
#[test]
fn test_tantivy_append_only_behavior() {
    // Build a simple schema with id and content fields
    let mut schema_builder = Schema::builder();
    let id_field = schema_builder.add_bytes_field("id", STORED | FAST);
    let timestamp_field = schema_builder.add_u64_field("timestamp", STORED | INDEXED | FAST);
    let content_field = schema_builder.add_text_field("content", TEXT | STORED);
    let schema = schema_builder.build();

    // Create a temporary index
    let temp_dir = tempfile::TempDir::new().unwrap();
    let index = Index::create_in_dir(temp_dir.path(), schema.clone()).unwrap();
    let mut writer: IndexWriter = index.writer(50_000_000).unwrap();

    // Simulate a node ID (like ULID bytes)
    let node_id: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];

    // Add multiple fragments with the SAME node_id but different timestamps and content
    // This simulates AddNodeFragment being called multiple times for the same node
    let fragments = vec![
        (1000u64, "First fragment about rust"),
        (2000u64, "Second fragment about programming"),
        (3000u64, "Third fragment about systems"),
        (4000u64, "Fourth fragment about async"),
        (5000u64, "Fifth fragment about concurrency"),
    ];

    for (ts, content) in &fragments {
        let doc = doc!(
            id_field => node_id.clone(),
            timestamp_field => *ts,
            content_field => *content,
        );
        writer.add_document(doc).unwrap();
    }

    writer.commit().unwrap();

    // Create a reader and searcher
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommitWithDelay)
        .try_into()
        .unwrap();
    let searcher = reader.searcher();

    // Verify ALL 5 documents are in the index
    assert_eq!(searcher.num_docs(), 5, "All 5 fragments should be in the index");

    // Search for each fragment's unique content - each should be found
    let query_parser = QueryParser::for_index(&index, vec![content_field]);

    // Search for "rust" - should find first fragment
    let query = query_parser.parse_query("rust").unwrap();
    let top_docs = searcher.search(&query, &TopDocs::with_limit(10)).unwrap();
    assert_eq!(top_docs.len(), 1, "Should find exactly 1 document containing 'rust'");

    // Search for "programming" - should find second fragment
    let query = query_parser.parse_query("programming").unwrap();
    let top_docs = searcher.search(&query, &TopDocs::with_limit(10)).unwrap();
    assert_eq!(top_docs.len(), 1, "Should find exactly 1 document containing 'programming'");

    // Search for "fragment" - should find ALL 5 documents (common word)
    let query = query_parser.parse_query("fragment").unwrap();
    let top_docs = searcher.search(&query, &TopDocs::with_limit(10)).unwrap();
    assert_eq!(top_docs.len(), 5, "Should find all 5 documents containing 'fragment'");

    // Verify we can retrieve ID and timestamp from each document
    for (_score, doc_address) in &top_docs {
        let doc: TantivyDocument = searcher.doc(*doc_address).unwrap();

        // Get the stored id field
        let id_value = doc.get_first(id_field).unwrap();
        let stored_id = id_value.as_bytes().unwrap();
        assert_eq!(stored_id, node_id.as_slice(), "ID should match");

        // Get the stored timestamp
        let ts_value = doc.get_first(timestamp_field).unwrap();
        let stored_ts = ts_value.as_u64().unwrap();
        assert!(stored_ts >= 1000 && stored_ts <= 5000, "Timestamp should be in expected range");
    }

}

/// Test that we can reconstruct RocksDB keys from Tantivy search results.
/// For NodeFragments, the CfKey is (node_id, timestamp).
#[test]
fn test_can_reconstruct_rocksdb_keys() {
    let mut schema_builder = Schema::builder();
    let id_field = schema_builder.add_bytes_field("id", STORED | FAST);
    let timestamp_field = schema_builder.add_u64_field("timestamp", STORED | INDEXED | FAST);
    let doc_type_field = schema_builder.add_text_field("doc_type", STRING | STORED);
    let content_field = schema_builder.add_text_field("content", TEXT);
    let schema = schema_builder.build();

    let temp_dir = tempfile::TempDir::new().unwrap();
    let index = Index::create_in_dir(temp_dir.path(), schema.clone()).unwrap();
    let mut writer: IndexWriter = index.writer(50_000_000).unwrap();

    // Create test data simulating node fragments
    let node_id: Vec<u8> = vec![0x01, 0x8D, 0x9B, 0xC4, 0x5E, 0xF2, 0x00, 0x00,
                                0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]; // 16 bytes like ULID

    let fragment_data = vec![
        (1701619200000u64, "First observation about the subject"),  // specific timestamp
        (1701619260000u64, "Second observation with more detail"),  // 1 minute later
        (1701619320000u64, "Third observation concluding thoughts"), // 2 minutes later
    ];

    for (ts, content) in &fragment_data {
        let doc = doc!(
            id_field => node_id.clone(),
            timestamp_field => *ts,
            doc_type_field => "node_fragments",
            content_field => *content,
        );
        writer.add_document(doc).unwrap();
    }

    writer.commit().unwrap();

    let reader = index.reader().unwrap();
    let searcher = reader.searcher();

    // Search for "observation"
    let query_parser = QueryParser::for_index(&index, vec![content_field]);
    let query = query_parser.parse_query("observation").unwrap();
    let top_docs = searcher.search(&query, &TopDocs::with_limit(10)).unwrap();

    assert_eq!(top_docs.len(), 3, "Should find all 3 observation fragments");

    // For each result, extract the data needed to reconstruct RocksDB CfKey
    let mut reconstructed_keys: Vec<(Vec<u8>, u64, String)> = Vec::new();

    for (_score, doc_address) in &top_docs {
        let doc: TantivyDocument = searcher.doc(*doc_address).unwrap();

        let id_bytes = doc.get_first(id_field).unwrap().as_bytes().unwrap().to_vec();
        let timestamp = doc.get_first(timestamp_field).unwrap().as_u64().unwrap();
        let doc_type = doc.get_first(doc_type_field).unwrap().as_str().unwrap().to_string();

        reconstructed_keys.push((id_bytes, timestamp, doc_type));
    }

    // Verify we have all 3 unique keys
    assert_eq!(reconstructed_keys.len(), 3);

    // Sort by timestamp to verify all fragments are present
    reconstructed_keys.sort_by_key(|k| k.1);

    assert_eq!(reconstructed_keys[0].1, 1701619200000u64);
    assert_eq!(reconstructed_keys[1].1, 1701619260000u64);
    assert_eq!(reconstructed_keys[2].1, 1701619320000u64);

    // All should have same node_id
    for key in &reconstructed_keys {
        assert_eq!(key.0, node_id);
        assert_eq!(key.2, "node_fragments");
    }

}

/// Test edge fragments with composite key (src_id, dst_id, edge_name, timestamp)
#[test]
fn test_edge_fragments_composite_key() {
    let mut schema_builder = Schema::builder();
    let src_id_field = schema_builder.add_bytes_field("src_id", STORED | FAST);
    let dst_id_field = schema_builder.add_bytes_field("dst_id", STORED | FAST);
    let edge_name_field = schema_builder.add_text_field("edge_name", STRING | STORED);
    let timestamp_field = schema_builder.add_u64_field("timestamp", STORED | INDEXED | FAST);
    let content_field = schema_builder.add_text_field("content", TEXT);
    let schema = schema_builder.build();

    let temp_dir = tempfile::TempDir::new().unwrap();
    let index = Index::create_in_dir(temp_dir.path(), schema.clone()).unwrap();
    let mut writer: IndexWriter = index.writer(50_000_000).unwrap();

    let src_id: Vec<u8> = vec![1; 16];
    let dst_id: Vec<u8> = vec![2; 16];
    let edge_name = "works_with";

    // Add multiple fragments for the same edge
    let fragments = vec![
        (1000u64, "Initial collaboration notes"),
        (2000u64, "Updated collaboration status"),
        (3000u64, "Final collaboration summary"),
    ];

    for (ts, content) in &fragments {
        let doc = doc!(
            src_id_field => src_id.clone(),
            dst_id_field => dst_id.clone(),
            edge_name_field => edge_name,
            timestamp_field => *ts,
            content_field => *content,
        );
        writer.add_document(doc).unwrap();
    }

    writer.commit().unwrap();

    let reader = index.reader().unwrap();
    let searcher = reader.searcher();

    // All 3 fragments should be searchable
    let query_parser = QueryParser::for_index(&index, vec![content_field]);
    let query = query_parser.parse_query("collaboration").unwrap();
    let top_docs = searcher.search(&query, &TopDocs::with_limit(10)).unwrap();

    assert_eq!(top_docs.len(), 3, "All 3 edge fragments should be found");

    // Verify we can reconstruct composite key for each
    for (_score, doc_address) in &top_docs {
        let doc: TantivyDocument = searcher.doc(*doc_address).unwrap();

        let src = doc.get_first(src_id_field).unwrap().as_bytes().unwrap();
        let dst = doc.get_first(dst_id_field).unwrap().as_bytes().unwrap();
        let name = doc.get_first(edge_name_field).unwrap().as_str().unwrap();
        let ts = doc.get_first(timestamp_field).unwrap().as_u64().unwrap();

        assert_eq!(src, src_id.as_slice());
        assert_eq!(dst, dst_id.as_slice());
        assert_eq!(name, edge_name);
        assert!(ts >= 1000 && ts <= 3000);
    }

}

/// Demonstrate that Tantivy has no built-in document ID - delete_term works on INDEXED fields
///
/// IMPORTANT FINDINGS:
/// 1. delete_term only works on INDEXED fields
/// 2. Bytes fields with STORED | FAST are NOT indexed for term lookup!
/// 3. To delete by bytes field, it MUST be created with INDEXED
/// 4. IndexReader must be reloaded to see deletions
#[test]
fn test_tantivy_delete_by_term() {
    let mut schema_builder = Schema::builder();
    // CRITICAL: Must use INDEXED for delete_term to work!
    let id_field = schema_builder.add_bytes_field("id", STORED | FAST | INDEXED);
    let timestamp_field = schema_builder.add_u64_field("timestamp", STORED | INDEXED | FAST);
    let content_field = schema_builder.add_text_field("content", TEXT);
    let schema = schema_builder.build();

    let temp_dir = tempfile::TempDir::new().unwrap();
    let index = Index::create_in_dir(temp_dir.path(), schema.clone()).unwrap();
    let mut writer: IndexWriter = index.writer(50_000_000).unwrap();

    let node_id_a: Vec<u8> = vec![1; 16];
    let node_id_b: Vec<u8> = vec![2; 16];

    // Add fragments for two different nodes
    for ts in [1000u64, 2000, 3000] {
        writer.add_document(doc!(
            id_field => node_id_a.clone(),
            timestamp_field => ts,
            content_field => format!("Fragment for node A at {}", ts),
        )).unwrap();
    }

    for ts in [1000u64, 2000] {
        writer.add_document(doc!(
            id_field => node_id_b.clone(),
            timestamp_field => ts,
            content_field => format!("Fragment for node B at {}", ts),
        )).unwrap();
    }

    writer.commit().unwrap();

    // Create a reader with OnCommitWithDelay reload policy
    let reader = index
        .reader_builder()
        .reload_policy(ReloadPolicy::OnCommitWithDelay)
        .try_into()
        .unwrap();
    assert_eq!(reader.searcher().num_docs(), 5, "Should have 5 documents total");

    // Delete all documents for node A using delete_term
    let term = tantivy::Term::from_field_bytes(id_field, &node_id_a);
    writer.delete_term(term);
    writer.commit().unwrap();

    // IMPORTANT: Must reload the reader to see deletions!
    reader.reload().unwrap();

    let searcher = reader.searcher();

    // Only node B's fragments should remain
    assert_eq!(searcher.num_docs(), 2, "Should have 2 documents remaining (node B only)");

}
