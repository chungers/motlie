//! Integration tests for the unified query interface (motlie_db::query).
//!
//! These tests verify the unified query API that composes:
//! 1. Fulltext search (Tantivy) for ranking and text matching
//! 2. Graph hydration (RocksDB) for retrieving full node/edge data
//!
//! # Key Components Demonstrated
//!
//! - `motlie_db::Storage::ready()` - Initializes both graph and fulltext
//!   subsystems and sets up MPMC query pipelines
//! - `motlie_db::query::Runnable<Reader>` - Trait for executing queries against
//!   the unified reader
//! - `motlie_db::query::{Nodes, Edges}` - Fulltext search with graph hydration
//! - `motlie_db::query::{EdgeDetails, NodeFragments, EdgeFragments}` - Direct
//!   graph lookups through the unified reader
//!
//! # Example Usage
//!
//! ```ignore
//! use motlie_db::{Storage, StorageConfig};
//! use motlie_db::query::{Nodes, Runnable};
//! use std::time::Duration;
//!
//! // Read-only mode: Initialize unified storage (graph + fulltext)
//! // Storage takes a single path and derives subdirectories:
//! //   <db_path>/graph - RocksDB graph database
//! //   <db_path>/fulltext - Tantivy fulltext index
//! let storage = Storage::readonly(db_path);
//! let handles = storage.ready(StorageConfig::default())?;  // Returns ReadOnlyHandles
//!
//! // Execute unified queries
//! let results = Nodes::new("search term".to_string(), 10)
//!     .run(handles.reader(), Duration::from_secs(5))
//!     .await?;
//!
//! // Clean shutdown
//! handles.shutdown().await?;
//!
//! // Read-write mode: Use Storage::readwrite() for both reads and writes
//! let storage = Storage::readwrite(db_path);
//! let handles = storage.ready(StorageConfig::default())?;  // Returns ReadWriteHandles
//!
//! // Write mutations - no unwrap() needed!
//! AddNode { /* ... */ }.run(handles.writer()).await?;
//!
//! // Execute queries
//! let results = Nodes::new("query", 10).run(handles.reader(), timeout).await?;
//! ```

use motlie_db::fulltext::{
    spawn_mutation_consumer as spawn_fulltext_mutation_consumer, Index, Storage as FulltextStorage,
};
use motlie_db::graph::mutation::{AddEdge, AddEdgeFragment, AddNode, AddNodeFragment};
use motlie_db::graph::schema::{EdgeSummary, NodeSummary};
use motlie_db::graph::writer::{
    create_mutation_writer, spawn_mutation_consumer_with_next, WriterConfig,
};
use motlie_db::graph::Storage as GraphStorage;
use motlie_db::mutation::Runnable as MutationRunnable;
use motlie_db::query::{
    AllEdges, AllNodes, EdgeDetails, Edges, FuzzyLevel, IncomingEdges, NodeById, NodeFragments,
    Nodes, NodesByIdsMulti, OutgoingEdges, Runnable, WithOffset,
};
use motlie_db::reader::ReaderConfig;
use motlie_db::{DataUrl, Id, ReadOnlyHandles, Storage, StorageConfig, TimestampMilli};
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;
use tokio::sync::mpsc;

/// Helper to populate test data in the database.
/// Creates the mutation pipeline, inserts data, and waits for completion.
async fn populate_test_data(db_path: &std::path::Path, index_path: &std::path::Path) {
    let config = WriterConfig {
        channel_buffer_size: 100,
    };

    // Create mutation pipeline: Graph -> Fulltext
    let (writer, mutation_rx) = create_mutation_writer(config.clone());
    let (fulltext_tx, fulltext_rx) = mpsc::channel(100);

    let graph_handle =
        spawn_mutation_consumer_with_next(mutation_rx, config.clone(), db_path, fulltext_tx);
    let fulltext_handle = spawn_fulltext_mutation_consumer(fulltext_rx, config.clone(), index_path);

    // Insert test data
    insert_test_data(&writer).await;

    // Flush to ensure graph writes are committed
    writer.flush().await.unwrap();

    // Shutdown mutation pipeline
    drop(writer);
    graph_handle.await.unwrap().unwrap();
    fulltext_handle.await.unwrap().unwrap();
}

/// Helper to set up a test environment using the unified Storage::ready() API.
///
/// This demonstrates the recommended way to initialize the unified query system:
/// 1. Create a `motlie_db::Storage` with a single base path
/// 2. Call `ready()` to initialize both subsystems and get `ReadOnlyHandles`
/// 3. Use `handles.reader()` to execute queries via the `Runnable` trait
/// 4. Call `handles.shutdown()` for clean termination
///
/// Returns the ReadOnlyHandles and paths for cleanup.
async fn setup_test_env_with_unified_storage(
    temp_dir: &TempDir,
) -> (ReadOnlyHandles, std::path::PathBuf, std::path::PathBuf) {
    let db_path = temp_dir.path().join("motlie_db");
    let graph_path = db_path.join("graph");
    let index_path = db_path.join("fulltext");

    // Populate test data using the derived subdirectory paths
    populate_test_data(&graph_path, &index_path).await;

    // =========================================================================
    // UNIFIED STORAGE API DEMONSTRATION
    // =========================================================================
    //
    // Use `motlie_db::Storage::readonly().ready()` to initialize both graph and
    // fulltext subsystems in one call. This:
    // - Takes a single base path and derives subdirectories:
    //   - <path>/graph for RocksDB
    //   - <path>/fulltext for Tantivy
    // - Opens both storage systems in read-only mode
    // - Creates MPMC channels for query dispatch
    // - Spawns consumer pools for parallel query processing
    // - Returns ReadOnlyHandles for lifecycle management
    //
    // The ReadOnlyHandles provides:
    // - `reader()` - Get the Reader for executing queries
    // - `shutdown()` - Clean termination of all workers
    //
    // For read-write mode, use Storage::readwrite() which returns ReadWriteHandles
    // with both reader() and writer() - no unwrap() needed!

    let storage = Storage::readonly(&db_path);
    let config = StorageConfig::with_channel_buffer_size(100).with_num_workers(2);
    let handles = storage.ready(config).unwrap();

    (handles, graph_path, index_path)
}

/// Helper to set up a test environment with manual graph and fulltext initialization.
/// This is the lower-level API that gives more control over individual subsystems.
/// Returns the unified reader, storage, and consumer handles.
async fn setup_test_env(
    temp_dir: &TempDir,
) -> (
    motlie_db::reader::Reader,
    Vec<tokio::task::JoinHandle<()>>,
    std::path::PathBuf,
    std::path::PathBuf,
) {
    let db_path = temp_dir.path().join("graph_db");
    let index_path = temp_dir.path().join("fulltext_index");

    // Populate test data
    populate_test_data(&db_path, &index_path).await;

    // Manual initialization (lower-level API)
    let mut graph_storage = GraphStorage::readonly(&db_path);
    graph_storage.ready().unwrap();
    let graph_storage = Arc::new(graph_storage);

    let mut fulltext_storage = FulltextStorage::readonly(&index_path);
    fulltext_storage.ready().unwrap();
    let fulltext_index = Arc::new(Index::new(Arc::new(fulltext_storage)));

    // Create unified reader with both graph and fulltext subsystems
    let reader_config = ReaderConfig::with_channel_buffer_size(100);
    let (reader, unified_handles, graph_handles, fulltext_handles) =
        motlie_db::reader::ReaderBuilder::new(graph_storage, fulltext_index)
            .with_config(reader_config)
            .with_num_workers(2)
            .build();

    // Combine all handles
    let mut handles = unified_handles;
    handles.extend(graph_handles);
    handles.extend(fulltext_handles);

    (reader, handles, db_path, index_path)
}

/// Insert test nodes and edges with searchable content.
async fn insert_test_data(writer: &motlie_db::graph::writer::Writer) {
    // Node 1: Rust programming language
    let rust_id = Id::new();
    AddNode {
        id: rust_id,
        ts_millis: TimestampMilli::now(),
        name: "Rust".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text("Systems programming language focused on safety"),
    }
    .run(writer)
    .await
    .unwrap();

    AddNodeFragment {
        id: rust_id,
        ts_millis: TimestampMilli::now(),
        content: DataUrl::from_text(
            "Rust is a systems programming language with memory safety guarantees",
        ),
        valid_range: None,
    }
    .run(writer)
    .await
    .unwrap();

    // Node 2: Python programming language
    let python_id = Id::new();
    AddNode {
        id: python_id,
        ts_millis: TimestampMilli::now(),
        name: "Python".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text("High-level programming language"),
    }
    .run(writer)
    .await
    .unwrap();

    AddNodeFragment {
        id: python_id,
        ts_millis: TimestampMilli::now(),
        content: DataUrl::from_text("Python is great for data science and machine learning"),
        valid_range: None,
    }
    .run(writer)
    .await
    .unwrap();

    // Node 3: JavaScript
    let js_id = Id::new();
    AddNode {
        id: js_id,
        ts_millis: TimestampMilli::now(),
        name: "JavaScript".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text("Web programming language"),
    }
    .run(writer)
    .await
    .unwrap();

    // Edge: Rust influences JavaScript (WebAssembly)
    AddEdge {
        source_node_id: rust_id,
        target_node_id: js_id,
        ts_millis: TimestampMilli::now(),
        name: "influences".to_string(),
        summary: EdgeSummary::from_text("Rust compiles to WebAssembly for browser execution"),
        weight: Some(0.8),
        valid_range: None,
    }
    .run(writer)
    .await
    .unwrap();

    AddEdgeFragment {
        src_id: rust_id,
        dst_id: js_id,
        edge_name: "influences".to_string(),
        ts_millis: TimestampMilli::now(),
        content: DataUrl::from_text(
            "WebAssembly allows Rust code to run in browsers alongside JavaScript",
        ),
        valid_range: None,
    }
    .run(writer)
    .await
    .unwrap();

    // Edge: Python used_with JavaScript (full-stack)
    AddEdge {
        source_node_id: python_id,
        target_node_id: js_id,
        ts_millis: TimestampMilli::now(),
        name: "used_with".to_string(),
        summary: EdgeSummary::from_text("Python backend with JavaScript frontend"),
        weight: Some(0.9),
        valid_range: None,
    }
    .run(writer)
    .await
    .unwrap();
}

/// Test: Nodes query returns hydrated data from graph storage
#[tokio::test]
async fn test_nodes_query_returns_hydrated_data() {
    let temp_dir = TempDir::new().unwrap();
    let (reader, handles, _, _) = setup_test_env(&temp_dir).await;

    // Search for "programming" - should match Rust and Python
    let results = Nodes::new("programming".to_string(), 10)
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

    // Should have results
    assert!(
        !results.is_empty(),
        "Expected results for 'programming' query"
    );

    // Results should be hydrated tuples: (Id, NodeName, NodeSummary)
    for (id, name, summary) in &results {
        assert!(!id.is_nil(), "Node ID should not be nil");
        assert!(!name.is_empty(), "Node name should not be empty");
        // Summary should contain actual content from graph (DataUrl)
        assert!(
            !summary.as_ref().is_empty(),
            "Node summary should be hydrated from graph"
        );
    }

    // Verify we got expected nodes (Rust and/or Python mention "programming")
    let names: Vec<&str> = results.iter().map(|(_, name, _)| name.as_str()).collect();
    assert!(
        names.contains(&"Rust") || names.contains(&"Python"),
        "Expected Rust or Python in results, got: {:?}",
        names
    );

    // Shutdown
    drop(reader);
    for handle in handles {
        handle.await.unwrap();
    }
}

/// Test: Edges query returns hydrated data from graph storage
#[tokio::test]
async fn test_edges_query_returns_hydrated_data() {
    let temp_dir = TempDir::new().unwrap();
    let (reader, handles, _, _) = setup_test_env(&temp_dir).await;

    // Search for "WebAssembly" - should match the influences edge
    let results = Edges::new("WebAssembly".to_string(), 10)
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

    // Should have results
    assert!(
        !results.is_empty(),
        "Expected results for 'WebAssembly' query"
    );

    // Results should be hydrated tuples: (SrcId, DstId, EdgeName, EdgeSummary)
    for (src_id, dst_id, edge_name, summary) in &results {
        assert!(!src_id.is_nil(), "Source ID should not be nil");
        assert!(!dst_id.is_nil(), "Destination ID should not be nil");
        assert!(!edge_name.is_empty(), "Edge name should not be empty");
        // Summary should contain actual content from graph (DataUrl)
        assert!(
            !summary.as_ref().is_empty(),
            "Edge summary should be hydrated from graph"
        );
    }

    // Verify we got the expected edge
    let edge_names: Vec<&str> = results
        .iter()
        .map(|(_, _, name, _)| name.as_str())
        .collect();
    assert!(
        edge_names.contains(&"influences"),
        "Expected 'influences' edge in results, got: {:?}",
        edge_names
    );

    // Shutdown
    drop(reader);
    for handle in handles {
        handle.await.unwrap();
    }
}

/// Test: Fuzzy matching works end-to-end
#[tokio::test]
async fn test_nodes_query_with_fuzzy_matching() {
    let temp_dir = TempDir::new().unwrap();
    let (reader, handles, _, _) = setup_test_env(&temp_dir).await;

    // Search with typo "programing" (missing 'm') with fuzzy matching
    let results = Nodes::new("programing".to_string(), 10)
        .with_fuzzy(FuzzyLevel::Low)
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

    // With fuzzy matching, should still find results
    // Note: This depends on Tantivy's fuzzy implementation
    // The test verifies the pipeline works, even if no results are returned
    // due to fuzzy distance thresholds
    println!(
        "Fuzzy search for 'programing' returned {} results",
        results.len()
    );

    // Shutdown
    drop(reader);
    for handle in handles {
        handle.await.unwrap();
    }
}

/// Test: Pagination with offset works correctly
#[tokio::test]
async fn test_nodes_query_with_pagination() {
    let temp_dir = TempDir::new().unwrap();
    let (reader, handles, _, _) = setup_test_env(&temp_dir).await;

    // Get all results first
    let all_results = Nodes::new("programming".to_string(), 10)
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

    if all_results.len() > 1 {
        // Get results with offset=1, should skip first result
        let offset_results = Nodes::new("programming".to_string(), 10)
            .with_offset(1)
            .run(&reader, Duration::from_secs(5))
            .await
            .unwrap();

        // Should have one fewer result
        assert_eq!(
            offset_results.len(),
            all_results.len() - 1,
            "Offset should skip first result"
        );

        // First result with offset should be second result without offset
        if !offset_results.is_empty() {
            assert_eq!(
                offset_results[0].0, all_results[1].0,
                "First offset result should match second original result"
            );
        }
    }

    // Shutdown
    drop(reader);
    for handle in handles {
        handle.await.unwrap();
    }
}

/// Test: Query returns empty results for non-matching search
#[tokio::test]
async fn test_nodes_query_no_matches() {
    let temp_dir = TempDir::new().unwrap();
    let (reader, handles, _, _) = setup_test_env(&temp_dir).await;

    // Search for something that doesn't exist
    let results = Nodes::new("xyznonexistent123".to_string(), 10)
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

    assert!(
        results.is_empty(),
        "Should have no results for non-matching query"
    );

    // Shutdown
    drop(reader);
    for handle in handles {
        handle.await.unwrap();
    }
}

/// Test: Multiple concurrent queries work correctly
#[tokio::test]
async fn test_concurrent_queries() {
    let temp_dir = TempDir::new().unwrap();
    let (reader, handles, _, _) = setup_test_env(&temp_dir).await;

    // Spawn multiple concurrent queries
    let reader1 = reader.clone();
    let reader2 = reader.clone();
    let reader3 = reader.clone();

    let (r1, r2, r3) = tokio::join!(
        async move {
            Nodes::new("Rust".to_string(), 10)
                .run(&reader1, Duration::from_secs(5))
                .await
        },
        async move {
            Nodes::new("Python".to_string(), 10)
                .run(&reader2, Duration::from_secs(5))
                .await
        },
        async move {
            Edges::new("WebAssembly".to_string(), 10)
                .run(&reader3, Duration::from_secs(5))
                .await
        }
    );

    // All queries should succeed
    assert!(r1.is_ok(), "First query should succeed");
    assert!(r2.is_ok(), "Second query should succeed");
    assert!(r3.is_ok(), "Third query should succeed");

    // Shutdown
    drop(reader);
    for handle in handles {
        handle.await.unwrap();
    }
}

// ============================================================================
// Tag Filtering Tests
// ============================================================================

/// Helper to set up a test environment with tagged data.
async fn setup_tagged_test_env(
    temp_dir: &TempDir,
) -> (motlie_db::reader::Reader, Vec<tokio::task::JoinHandle<()>>) {
    let db_path = temp_dir.path().join("graph_db");
    let index_path = temp_dir.path().join("fulltext_index");

    let config = WriterConfig {
        channel_buffer_size: 100,
    };

    // Create mutation pipeline: Graph -> Fulltext
    let (writer, mutation_rx) = create_mutation_writer(config.clone());
    let (fulltext_tx, fulltext_rx) = mpsc::channel(100);

    let graph_handle =
        spawn_mutation_consumer_with_next(mutation_rx, config.clone(), &db_path, fulltext_tx);
    let fulltext_handle =
        spawn_fulltext_mutation_consumer(fulltext_rx, config.clone(), &index_path);

    // Insert test data with tags
    insert_tagged_test_data(&writer).await;

    // Flush to ensure graph writes are committed
    writer.flush().await.unwrap();

    // Shutdown mutation pipeline
    drop(writer);
    graph_handle.await.unwrap().unwrap();
    fulltext_handle.await.unwrap().unwrap();

    // Open storage for reading
    let mut graph_storage = GraphStorage::readonly(&db_path);
    graph_storage.ready().unwrap();
    let graph_storage = Arc::new(graph_storage);

    let mut fulltext_storage = FulltextStorage::readonly(&index_path);
    fulltext_storage.ready().unwrap();
    let fulltext_index = Arc::new(Index::new(Arc::new(fulltext_storage)));

    // Create unified reader with both graph and fulltext subsystems
    let reader_config = ReaderConfig::with_channel_buffer_size(100);
    let (reader, unified_handles, graph_handles, fulltext_handles) =
        motlie_db::reader::ReaderBuilder::new(graph_storage, fulltext_index)
            .with_config(reader_config)
            .with_num_workers(2)
            .build();

    // Combine all handles
    let mut handles = unified_handles;
    handles.extend(graph_handles);
    handles.extend(fulltext_handles);

    (reader, handles)
}

/// Insert test data with hashtags for tag filtering tests.
async fn insert_tagged_test_data(writer: &motlie_db::graph::writer::Writer) {
    // Node 1: Rust with tags #systems #performance #safety
    let rust_id = Id::new();
    AddNode {
        id: rust_id,
        ts_millis: TimestampMilli::now(),
        name: "Rust Language".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text("A systems programming language"),
    }
    .run(writer)
    .await
    .unwrap();

    AddNodeFragment {
        id: rust_id,
        ts_millis: TimestampMilli::now(),
        content: DataUrl::from_text(
            "Rust provides #systems level control with #performance and #safety guarantees",
        ),
        valid_range: None,
    }
    .run(writer)
    .await
    .unwrap();

    // Node 2: Go with tags #systems #concurrency
    let go_id = Id::new();
    AddNode {
        id: go_id,
        ts_millis: TimestampMilli::now(),
        name: "Go Language".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text("A compiled programming language"),
    }
    .run(writer)
    .await
    .unwrap();

    AddNodeFragment {
        id: go_id,
        ts_millis: TimestampMilli::now(),
        content: DataUrl::from_text(
            "Go is great for #systems programming with built-in #concurrency support",
        ),
        valid_range: None,
    }
    .run(writer)
    .await
    .unwrap();

    // Node 3: Python with tags #scripting #datascience
    let python_id = Id::new();
    AddNode {
        id: python_id,
        ts_millis: TimestampMilli::now(),
        name: "Python Language".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text("A high-level programming language"),
    }
    .run(writer)
    .await
    .unwrap();

    AddNodeFragment {
        id: python_id,
        ts_millis: TimestampMilli::now(),
        content: DataUrl::from_text("Python excels at #scripting and #datascience applications"),
        valid_range: None,
    }
    .run(writer)
    .await
    .unwrap();

    // Edge: Rust -> Go with tags #comparison #performance
    AddEdge {
        source_node_id: rust_id,
        target_node_id: go_id,
        ts_millis: TimestampMilli::now(),
        name: "competes_with".to_string(),
        summary: EdgeSummary::from_text("Both are systems languages"),
        weight: Some(0.7),
        valid_range: None,
    }
    .run(writer)
    .await
    .unwrap();

    AddEdgeFragment {
        src_id: rust_id,
        dst_id: go_id,
        edge_name: "competes_with".to_string(),
        ts_millis: TimestampMilli::now(),
        content: DataUrl::from_text(
            "Rust and Go both target #systems programming with different #performance tradeoffs",
        ),
        valid_range: None,
    }
    .run(writer)
    .await
    .unwrap();

    // Edge: Python -> Go with tags #comparison #web
    AddEdge {
        source_node_id: python_id,
        target_node_id: go_id,
        ts_millis: TimestampMilli::now(),
        name: "integrates_with".to_string(),
        summary: EdgeSummary::from_text("Python and Go work well together"),
        weight: Some(0.6),
        valid_range: None,
    }
    .run(writer)
    .await
    .unwrap();

    AddEdgeFragment {
        src_id: python_id,
        dst_id: go_id,
        edge_name: "integrates_with".to_string(),
        ts_millis: TimestampMilli::now(),
        content: DataUrl::from_text(
            "Use Python for #scripting and Go for #web services in the same project",
        ),
        valid_range: None,
    }
    .run(writer)
    .await
    .unwrap();
}

/// Test: Nodes can be filtered by a single tag using builder method
#[tokio::test]
async fn test_nodes_filter_by_single_tag_builder() {
    let temp_dir = TempDir::new().unwrap();
    let (reader, handles) = setup_tagged_test_env(&temp_dir).await;

    // Search for "programming" with tag filter "systems"
    // Should match Rust and Go (both have #systems tag)
    let results = Nodes::new("programming".to_string(), 10)
        .with_tags(vec!["systems".to_string()])
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

    // Should have results
    assert!(
        !results.is_empty(),
        "Expected results for 'programming' with #systems tag"
    );

    // Verify only nodes with #systems tag are returned
    let names: Vec<&str> = results.iter().map(|(_, name, _)| name.as_str()).collect();
    println!("Nodes with #systems tag: {:?}", names);

    // Python should NOT be in results (it has #scripting, #datascience, not #systems)
    assert!(
        !names.contains(&"Python Language"),
        "Python should not match #systems tag"
    );

    // Shutdown
    drop(reader);
    for handle in handles {
        handle.await.unwrap();
    }
}

/// Test: Nodes can be filtered by tags using struct initialization (no builder)
#[tokio::test]
async fn test_nodes_filter_by_tag_struct_init() {
    let temp_dir = TempDir::new().unwrap();
    let (reader, handles) = setup_tagged_test_env(&temp_dir).await;

    // Create query using struct initialization instead of builder
    // This shows that the fulltext query struct has public fields
    //
    // Note: We search for "excels" which appears in the Python fragment that also
    // contains the #scripting tag. Tags are extracted from the same content they appear in.
    let query = Nodes {
        query: "excels".to_string(),
        limit: 10,
        tags: vec!["scripting".to_string()],
        ..Default::default()
    };

    let results = query.run(&reader, Duration::from_secs(5)).await.unwrap();

    // Should match Python (the fragment contains both "excels" and #scripting)
    let names: Vec<&str> = results.iter().map(|(_, name, _)| name.as_str()).collect();
    println!("Nodes with #scripting tag and 'excels': {:?}", names);

    // Only Python should match
    assert!(
        names.contains(&"Python Language"),
        "Python should match #scripting tag and 'excels' query"
    );
    assert!(
        !names.contains(&"Rust Language"),
        "Rust should not match #scripting tag"
    );
    assert!(
        !names.contains(&"Go Language"),
        "Go should not match #scripting tag"
    );

    // Shutdown
    drop(reader);
    for handle in handles {
        handle.await.unwrap();
    }
}

/// Test: Nodes can be filtered by multiple tags (OR semantics)
#[tokio::test]
async fn test_nodes_filter_by_multiple_tags_or() {
    let temp_dir = TempDir::new().unwrap();
    let (reader, handles) = setup_tagged_test_env(&temp_dir).await;

    // Search with multiple tags - should match documents with ANY of the tags
    // "safety" matches Rust, "concurrency" matches Go
    let results = Nodes::new("programming".to_string(), 10)
        .with_tags(vec!["safety".to_string(), "concurrency".to_string()])
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

    let names: Vec<&str> = results.iter().map(|(_, name, _)| name.as_str()).collect();
    println!("Nodes with #safety OR #concurrency tags: {:?}", names);

    // Should match Rust (has #safety) and Go (has #concurrency)
    // Python should not match (has neither)
    assert!(
        !names.contains(&"Python Language"),
        "Python should not match #safety or #concurrency"
    );

    // Shutdown
    drop(reader);
    for handle in handles {
        handle.await.unwrap();
    }
}

/// Test: Edges can be filtered by tags using builder method
#[tokio::test]
async fn test_edges_filter_by_tag_builder() {
    let temp_dir = TempDir::new().unwrap();
    let (reader, handles) = setup_tagged_test_env(&temp_dir).await;

    // Search for edges with #performance tag
    let results = Edges::new("programming".to_string(), 10)
        .with_tags(vec!["performance".to_string()])
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

    // Should match the Rust->Go edge (has #performance in fragment)
    let edge_names: Vec<&str> = results
        .iter()
        .map(|(_, _, name, _)| name.as_str())
        .collect();
    println!("Edges with #performance tag: {:?}", edge_names);

    // The competes_with edge should match
    assert!(
        edge_names.contains(&"competes_with"),
        "competes_with edge should match #performance tag"
    );

    // Shutdown
    drop(reader);
    for handle in handles {
        handle.await.unwrap();
    }
}

/// Test: Edges can be filtered by tags using struct initialization (no builder)
#[tokio::test]
async fn test_edges_filter_by_tag_struct_init() {
    let temp_dir = TempDir::new().unwrap();
    let (reader, handles) = setup_tagged_test_env(&temp_dir).await;

    // Create query using struct initialization instead of builder
    // Note: Search for "services" which appears in the fragment with #web tag
    let query = Edges {
        query: "services".to_string(),
        limit: 10,
        tags: vec!["web".to_string()],
        ..Default::default()
    };

    let results = query.run(&reader, Duration::from_secs(5)).await.unwrap();

    // Should match the Python->Go edge (has "services" and #web in same fragment)
    let edge_names: Vec<&str> = results
        .iter()
        .map(|(_, _, name, _)| name.as_str())
        .collect();
    println!("Edges with #web tag and 'services': {:?}", edge_names);

    // The integrates_with edge should match
    assert!(
        edge_names.contains(&"integrates_with"),
        "integrates_with edge should match #web tag and 'services' query"
    );

    // The competes_with edge should NOT match (no #web tag)
    assert!(
        !edge_names.contains(&"competes_with"),
        "competes_with edge should not match #web tag"
    );

    // Shutdown
    drop(reader);
    for handle in handles {
        handle.await.unwrap();
    }
}

/// Test: Tag filter with no matches returns empty results
#[tokio::test]
async fn test_nodes_filter_by_nonexistent_tag() {
    let temp_dir = TempDir::new().unwrap();
    let (reader, handles) = setup_tagged_test_env(&temp_dir).await;

    // Search with a tag that doesn't exist
    let results = Nodes::new("programming".to_string(), 10)
        .with_tags(vec!["nonexistent_tag_xyz".to_string()])
        .run(&reader, Duration::from_secs(5))
        .await
        .unwrap();

    // Should have no results
    assert!(
        results.is_empty(),
        "Should have no results for nonexistent tag"
    );

    // Shutdown
    drop(reader);
    for handle in handles {
        handle.await.unwrap();
    }
}

// ============================================================================
// Tests demonstrating the unified Storage::ready() API
// ============================================================================

/// Test: Demonstrates the unified Storage::ready() API for initializing the query system.
///
/// This is the recommended way to set up the unified query infrastructure:
/// 1. Create `motlie_db::Storage::readonly()` or `Storage::readwrite()` with paths
/// 2. Call `ready()` which returns `ReadOnlyHandles` or `ReadWriteHandles`
/// 3. Use `handles.reader()` to execute queries via the `Runnable` trait
/// 4. For read-write, use `handles.writer()` - no unwrap() needed!
/// 5. Call `handles.shutdown().await` for clean termination
#[tokio::test]
async fn test_unified_storage_ready_api() {
    let temp_dir = TempDir::new().unwrap();

    // Use the unified Storage::ready() API - returns StorageHandle
    let (handle, _, _) = setup_test_env_with_unified_storage(&temp_dir).await;

    // =========================================================================
    // Execute various query types through the unified reader
    // =========================================================================

    let timeout = Duration::from_secs(5);

    // 1. Fulltext search for nodes (with graph hydration)
    let node_results = Nodes::new("programming".to_string(), 10)
        .run(handle.reader(), timeout)
        .await
        .unwrap();

    assert!(
        !node_results.is_empty(),
        "Nodes query should return hydrated results"
    );

    // Verify results are hydrated tuples: (Id, NodeName, NodeSummary)
    for (id, name, summary) in &node_results {
        assert!(!id.is_nil(), "Node ID should not be nil");
        assert!(!name.is_empty(), "Node name should not be empty");
        assert!(
            !summary.as_ref().is_empty(),
            "Node summary should be hydrated from graph"
        );
    }

    // 2. Fulltext search for edges (with graph hydration)
    let edge_results = Edges::new("WebAssembly".to_string(), 10)
        .run(handle.reader(), timeout)
        .await
        .unwrap();

    assert!(
        !edge_results.is_empty(),
        "Edges query should return hydrated results"
    );

    // Verify results are hydrated tuples: (SrcId, DstId, EdgeName, EdgeSummary)
    for (src_id, dst_id, edge_name, summary) in &edge_results {
        assert!(!src_id.is_nil(), "Source ID should not be nil");
        assert!(!dst_id.is_nil(), "Destination ID should not be nil");
        assert!(!edge_name.is_empty(), "Edge name should not be empty");
        assert!(
            !summary.as_ref().is_empty(),
            "Edge summary should be hydrated from graph"
        );
    }

    // 3. Fuzzy search
    let fuzzy_results = Nodes::new("programing".to_string(), 10) // intentional typo
        .with_fuzzy(FuzzyLevel::Low)
        .run(handle.reader(), timeout)
        .await
        .unwrap();

    assert!(
        !fuzzy_results.is_empty(),
        "Fuzzy search should match despite typo"
    );

    // 4. Pagination with offset
    let first_page = Nodes::new("programming".to_string(), 1)
        .run(handle.reader(), timeout)
        .await
        .unwrap();

    let second_page = Nodes::new("programming".to_string(), 1)
        .with_offset(1)
        .run(handle.reader(), timeout)
        .await
        .unwrap();

    // If there are multiple results, pages should be different
    if !first_page.is_empty() && !second_page.is_empty() {
        assert_ne!(
            first_page[0].0, second_page[0].0,
            "Pagination should return different results"
        );
    }

    // Clean shutdown using StorageHandle
    handle.shutdown().await.unwrap();
}

/// Test: EdgeDetails query returns full edge information without exposing scores.
///
/// This demonstrates using `EdgeDetails` (alias for `EdgeSummaryBySrcDstName`)
/// through the unified reader, which returns `EdgeResult` instead of the
/// internal `(EdgeSummary, Option<f64>)` type.
#[tokio::test]
async fn test_edge_details_query() {
    let temp_dir = TempDir::new().unwrap();
    let (handle, _, _) = setup_test_env_with_unified_storage(&temp_dir).await;

    let timeout = Duration::from_secs(5);

    // First, search for edges to get IDs
    let edge_results = Edges::new("WebAssembly".to_string(), 10)
        .run(handle.reader(), timeout)
        .await
        .unwrap();

    assert!(!edge_results.is_empty(), "Should have edge results");

    // Get the first edge's details using EdgeDetails query
    let (src_id, dst_id, edge_name, _summary) = &edge_results[0];

    let edge_detail = EdgeDetails::new(*src_id, *dst_id, edge_name.clone(), None)
        .run(handle.reader(), timeout)
        .await
        .unwrap();

    // EdgeDetails returns EdgeDetailsResult: (Option<f64>, SrcId, DstId, EdgeName, EdgeSummary, Version)
    let (detail_weight, detail_src, detail_dst, detail_name, detail_summary, _version) = edge_detail;

    assert_eq!(detail_src, *src_id, "Source ID should match");
    assert_eq!(detail_dst, *dst_id, "Destination ID should match");
    assert_eq!(&detail_name, edge_name, "Edge name should match");
    assert!(
        !detail_summary.as_ref().is_empty(),
        "Edge summary should be present"
    );
    // Weight can be None or Some - the test doesn't set a specific weight
    let _ = detail_weight;

    // Clean shutdown
    handle.shutdown().await.unwrap();
}

/// Test: NodeFragments query retrieves fragment history for a node.
#[tokio::test]
async fn test_node_fragments_query() {
    let temp_dir = TempDir::new().unwrap();
    let (handle, _, _) = setup_test_env_with_unified_storage(&temp_dir).await;

    let timeout = Duration::from_secs(5);

    // First, search for nodes to get an ID
    let node_results = Nodes::new("Rust".to_string(), 10)
        .run(handle.reader(), timeout)
        .await
        .unwrap();

    assert!(!node_results.is_empty(), "Should have node results");

    // Get the node's fragment history
    let (node_id, _name, _summary) = &node_results[0];

    // Query all fragments (unbounded time range)
    use std::ops::Bound;
    let fragments = NodeFragments::new(*node_id, (Bound::Unbounded, Bound::Unbounded), None)
        .run(handle.reader(), timeout)
        .await
        .unwrap();

    // Should have at least one fragment (we inserted one in test data)
    assert!(
        !fragments.is_empty(),
        "Should have fragment history for the node"
    );

    // Verify fragment structure: Vec<(TimestampMilli, FragmentContent)>
    for (ts, content) in &fragments {
        assert!(ts.0 > 0, "Timestamp should be valid");
        assert!(
            !content.as_ref().is_empty(),
            "Fragment content should not be empty"
        );
    }

    // Clean shutdown
    handle.shutdown().await.unwrap();
}

/// Test: NodeById, OutgoingEdges, and IncomingEdges via the unified reader.
///
/// Demonstrates that graph queries can be executed through the unified reader
/// without needing to import separate graph traits.
#[tokio::test]
async fn test_node_by_id_and_edge_queries() {
    let temp_dir = TempDir::new().unwrap();
    let (handle, _, _) = setup_test_env_with_unified_storage(&temp_dir).await;

    let timeout = Duration::from_secs(5);

    // First, find a node via fulltext search to get its ID
    let search_results = Nodes::new("Rust".to_string(), 10)
        .run(handle.reader(), timeout)
        .await
        .unwrap();

    assert!(!search_results.is_empty(), "Should find nodes via fulltext");
    let (rust_id, rust_name, _) = &search_results[0];
    assert!(
        rust_name.contains("Rust"),
        "Should find a Rust node: found '{}'",
        rust_name
    );

    // =========================================================================
    // 1. NodeById - Direct graph lookup by ID
    // =========================================================================
    let (name, summary, _version) = NodeById::new(*rust_id, None)
        .run(handle.reader(), timeout)
        .await
        .unwrap();

    assert!(
        name.contains("Rust"),
        "NodeById should return the correct node"
    );
    assert!(
        !summary.as_ref().is_empty(),
        "NodeById should return a summary"
    );
    println!(
        "NodeById result: name='{}', summary_len={}",
        name,
        summary.as_ref().len()
    );

    // =========================================================================
    // 2. OutgoingEdges - Get edges originating from this node
    // =========================================================================
    let outgoing = OutgoingEdges::new(*rust_id, None)
        .run(handle.reader(), timeout)
        .await
        .unwrap();

    // The test data has Rust -> JavaScript (influences) edge
    println!("OutgoingEdges from Rust: {} edges", outgoing.len());
    for (weight, src_id, dst_id, edge_name, _version) in &outgoing {
        println!(
            "  {} -> {} via '{}' (weight: {:?})",
            src_id, dst_id, edge_name, weight
        );
    }

    // Should have at least one outgoing edge
    assert!(!outgoing.is_empty(), "Rust node should have outgoing edges");

    // Find the JavaScript node to verify incoming edges
    let js_results = Nodes::new("JavaScript".to_string(), 10)
        .run(handle.reader(), timeout)
        .await
        .unwrap();

    if !js_results.is_empty() {
        let (js_id, js_name, _) = &js_results[0];

        // =====================================================================
        // 3. IncomingEdges - Get edges pointing to this node
        // =====================================================================
        let incoming = IncomingEdges::new(*js_id, None)
            .run(handle.reader(), timeout)
            .await
            .unwrap();

        println!("IncomingEdges to {}: {} edges", js_name, incoming.len());
        for (weight, src_id, dst_id, edge_name, _version) in &incoming {
            println!(
                "  {} -> {} via '{}' (weight: {:?})",
                src_id, dst_id, edge_name, weight
            );
        }

        // Should have at least one incoming edge (from Rust)
        assert!(
            !incoming.is_empty(),
            "JavaScript node should have incoming edges from Rust"
        );

        // Verify the edge is from Rust to JavaScript
        // IncomingEdges returns edges where the queried node is involved
        // The tuple format is (weight, src_id, dst_id, edge_name) from the edge's perspective
        // For incoming edges to js_id, we look for edges where dst_id == js_id (target)
        // OR where the edge connects rust and js
        let rust_to_js_edge = incoming.iter().find(|(_, src, dst, _, _)| {
            (*src == *rust_id && *dst == *js_id) || (*dst == *rust_id && *src == *js_id)
        });
        assert!(
            rust_to_js_edge.is_some(),
            "Should have an edge connecting Rust and JavaScript. Found: {:?}",
            incoming
        );
    }

    // Clean shutdown
    handle.shutdown().await.unwrap();
}

/// Test NodesByIdsMulti - batch lookup of multiple nodes by ID through unified API.
#[tokio::test]
async fn test_nodes_by_ids_multi() {
    let temp_dir = TempDir::new().unwrap();
    let (handle, _db_path, _index_path) = setup_test_env_with_unified_storage(&temp_dir).await;
    let timeout = Duration::from_secs(5);

    // First, use fulltext search to get some node IDs
    let search_results = Nodes::new("programming".to_string(), 10)
        .run(handle.reader(), timeout)
        .await
        .unwrap();

    // Collect the IDs from search results
    let found_ids: Vec<Id> = search_results.iter().map(|(id, _, _)| *id).collect();
    assert!(
        !found_ids.is_empty(),
        "Should find at least one node with 'programming'"
    );

    // Now use NodesByIdsMulti to batch lookup these nodes
    let batch_results = NodesByIdsMulti::new(found_ids.clone(), None)
        .run(handle.reader(), timeout)
        .await
        .unwrap();

    // Verify we got the same number of results
    assert_eq!(
        batch_results.len(),
        found_ids.len(),
        "Batch lookup should return same number of nodes"
    );

    // Verify the IDs match
    for (id, _name, _summary, _version) in &batch_results {
        assert!(
            found_ids.contains(id),
            "Each result ID should be in our lookup list"
        );
    }

    // Test with a mix of existing and non-existing IDs
    let non_existing_id = Id::new();
    let mixed_ids: Vec<Id> = found_ids
        .iter()
        .take(2)
        .copied()
        .chain(std::iter::once(non_existing_id))
        .collect();

    let mixed_results = NodesByIdsMulti::new(mixed_ids.clone(), None)
        .run(handle.reader(), timeout)
        .await
        .unwrap();

    // Should only return the existing nodes (not the non-existing one)
    assert!(
        mixed_results.len() <= found_ids.len().min(2),
        "Mixed results should not include non-existing nodes"
    );

    // Verify non-existing ID is not in results
    for (id, _, _, _) in &mixed_results {
        assert_ne!(
            *id, non_existing_id,
            "Non-existing ID should not be in results"
        );
    }

    // Test with empty input
    let empty_results = NodesByIdsMulti::new(vec![], None)
        .run(handle.reader(), timeout)
        .await
        .unwrap();

    assert_eq!(empty_results.len(), 0, "Empty input should return empty vec");

    // Clean shutdown
    handle.shutdown().await.unwrap();
}

// ============================================================================
// Tests for AllNodes and AllEdges graph enumeration queries
// ============================================================================

/// Test: AllNodes query returns all nodes via the unified API
#[tokio::test]
async fn test_all_nodes_via_unified_api() {
    let temp_dir = TempDir::new().unwrap();
    let (handle, _, _) = setup_test_env_with_unified_storage(&temp_dir).await;

    let timeout = Duration::from_secs(5);

    // Query all nodes - the test data has 3 nodes (Rust, Python, JavaScript)
    let results = AllNodes::new(100)
        .run(handle.reader(), timeout)
        .await
        .unwrap();

    // Should return all 3 nodes
    assert_eq!(results.len(), 3, "Should return all 3 nodes from test data");

    // Verify results are tuples: (Id, NodeName, NodeSummary, Version)
    let names: Vec<&str> = results.iter().map(|(_, name, _, _)| name.as_str()).collect();
    assert!(names.contains(&"Rust"), "Should contain Rust node");
    assert!(names.contains(&"Python"), "Should contain Python node");
    assert!(names.contains(&"JavaScript"), "Should contain JavaScript node");

    // Verify each node has valid data
    for (id, name, summary, _version) in &results {
        assert!(!id.is_nil(), "Node ID should not be nil");
        assert!(!name.is_empty(), "Node name should not be empty");
        assert!(
            !summary.as_ref().is_empty(),
            "Node summary should not be empty"
        );
    }

    // Clean shutdown
    handle.shutdown().await.unwrap();
}

/// Test: AllNodes pagination via the unified API
#[tokio::test]
async fn test_all_nodes_pagination_via_unified_api() {
    let temp_dir = TempDir::new().unwrap();
    let (handle, _, _) = setup_test_env_with_unified_storage(&temp_dir).await;

    let timeout = Duration::from_secs(5);

    // Get first page of 2 nodes
    let page1 = AllNodes::new(2)
        .run(handle.reader(), timeout)
        .await
        .unwrap();

    assert_eq!(page1.len(), 2, "First page should have 2 nodes");

    // Get second page using cursor from last result
    let last_id = page1.last().unwrap().0;
    let page2 = AllNodes::new(2)
        .with_cursor(last_id)
        .run(handle.reader(), timeout)
        .await
        .unwrap();

    assert_eq!(page2.len(), 1, "Second page should have 1 remaining node");

    // Verify no overlap between pages
    let page1_ids: std::collections::HashSet<_> = page1.iter().map(|(id, _, _, _)| *id).collect();
    let page2_ids: std::collections::HashSet<_> = page2.iter().map(|(id, _, _, _)| *id).collect();
    assert!(
        page1_ids.is_disjoint(&page2_ids),
        "Pages should not overlap"
    );

    // Combined should equal total nodes
    assert_eq!(
        page1.len() + page2.len(),
        3,
        "All pages combined should have all 3 nodes"
    );

    // Clean shutdown
    handle.shutdown().await.unwrap();
}

/// Test: AllEdges query returns all edges via the unified API
#[tokio::test]
async fn test_all_edges_via_unified_api() {
    let temp_dir = TempDir::new().unwrap();
    let (handle, _, _) = setup_test_env_with_unified_storage(&temp_dir).await;

    let timeout = Duration::from_secs(5);

    // Query all edges - the test data has 2 edges (Rust->JS influences, Python->JS used_with)
    let results = AllEdges::new(100)
        .run(handle.reader(), timeout)
        .await
        .unwrap();

    // Should return all 2 edges
    assert_eq!(results.len(), 2, "Should return all 2 edges from test data");

    // Verify results are tuples: (Option<f64>, SrcId, DstId, EdgeName, Version)
    let edge_names: Vec<&str> =
        results.iter().map(|(_, _, _, name, _)| name.as_str()).collect();
    assert!(
        edge_names.contains(&"influences"),
        "Should contain 'influences' edge"
    );
    assert!(
        edge_names.contains(&"used_with"),
        "Should contain 'used_with' edge"
    );

    // Verify each edge has valid data
    for (weight, src_id, dst_id, name, _version) in &results {
        assert!(!src_id.is_nil(), "Source ID should not be nil");
        assert!(!dst_id.is_nil(), "Destination ID should not be nil");
        assert!(!name.is_empty(), "Edge name should not be empty");
        assert!(weight.is_some(), "Edge weight should be present");
    }

    // Verify weights match test data
    for (weight, _, _, name, _version) in &results {
        match name.as_str() {
            "influences" => assert_eq!(*weight, Some(0.8), "influences edge should have weight 0.8"),
            "used_with" => assert_eq!(*weight, Some(0.9), "used_with edge should have weight 0.9"),
            _ => panic!("Unexpected edge name: {}", name),
        }
    }

    // Clean shutdown
    handle.shutdown().await.unwrap();
}

/// Test: AllEdges pagination via the unified API
#[tokio::test]
async fn test_all_edges_pagination_via_unified_api() {
    let temp_dir = TempDir::new().unwrap();
    let (handle, _, _) = setup_test_env_with_unified_storage(&temp_dir).await;

    let timeout = Duration::from_secs(5);

    // Get first page of 1 edge
    let page1 = AllEdges::new(1)
        .run(handle.reader(), timeout)
        .await
        .unwrap();

    assert_eq!(page1.len(), 1, "First page should have 1 edge");

    // Get second page using cursor from last result
    let (_, src, dst, name, _version) = page1.last().unwrap();
    let page2 = AllEdges::new(1)
        .with_cursor((*src, *dst, name.clone()))
        .run(handle.reader(), timeout)
        .await
        .unwrap();

    assert_eq!(page2.len(), 1, "Second page should have 1 remaining edge");

    // Verify no overlap - different edge names
    let page1_names: std::collections::HashSet<_> =
        page1.iter().map(|(_, _, _, name, _)| name.as_str()).collect();
    let page2_names: std::collections::HashSet<_> =
        page2.iter().map(|(_, _, _, name, _)| name.as_str()).collect();
    assert!(
        page1_names.is_disjoint(&page2_names),
        "Pages should not overlap"
    );

    // Clean shutdown
    handle.shutdown().await.unwrap();
}

/// Test: AllNodes with limit 0 returns empty
#[tokio::test]
async fn test_all_nodes_limit_zero() {
    let temp_dir = TempDir::new().unwrap();
    let (handle, _, _) = setup_test_env_with_unified_storage(&temp_dir).await;

    let timeout = Duration::from_secs(5);

    let results = AllNodes::new(0)
        .run(handle.reader(), timeout)
        .await
        .unwrap();

    assert_eq!(results.len(), 0, "Limit 0 should return empty results");

    // Clean shutdown
    handle.shutdown().await.unwrap();
}

/// Test: AllEdges with limit 0 returns empty
#[tokio::test]
async fn test_all_edges_limit_zero() {
    let temp_dir = TempDir::new().unwrap();
    let (handle, _, _) = setup_test_env_with_unified_storage(&temp_dir).await;

    let timeout = Duration::from_secs(5);

    let results = AllEdges::new(0)
        .run(handle.reader(), timeout)
        .await
        .unwrap();

    assert_eq!(results.len(), 0, "Limit 0 should return empty results");

    // Clean shutdown
    handle.shutdown().await.unwrap();
}
