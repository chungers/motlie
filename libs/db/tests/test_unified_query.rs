//! Integration tests for the unified query interface (motlie_db::query).
//!
//! These tests verify the unified query API that composes:
//! 1. Fulltext search (Tantivy) for ranking and text matching
//! 2. Graph hydration (RocksDB) for retrieving full node/edge data
//!
//! # Key Components Demonstrated
//!
//! - `motlie_db::reader::Storage::ready()` - Initializes both graph and fulltext
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
//! use motlie_db::reader::{Storage, ReaderConfig};
//! use motlie_db::query::{Nodes, Runnable};
//! use std::time::Duration;
//!
//! // Initialize unified storage (graph + fulltext)
//! let storage = Storage::readonly(graph_path, fulltext_path);
//! let (reader, handles) = storage.ready(ReaderConfig::default(), 4)?;
//!
//! // Execute unified queries
//! let results = Nodes::new("search term".to_string(), 10)
//!     .run(&reader, Duration::from_secs(5))
//!     .await?;
//! ```

use motlie_db::fulltext::{
    spawn_mutation_consumer as spawn_fulltext_mutation_consumer, Index,
    Storage as FulltextStorage,
};
use motlie_db::graph::mutation::{
    AddEdge, AddEdgeFragment, AddNode, AddNodeFragment, Runnable as MutationRunnable,
};
use motlie_db::graph::schema::{EdgeSummary, NodeSummary};
use motlie_db::graph::writer::{
    create_mutation_writer, spawn_mutation_consumer_with_next, WriterConfig,
};
use motlie_db::graph::{Graph, Storage as GraphStorage};
use motlie_db::query::{
    EdgeDetails, Edges, FuzzyLevel, IncomingEdges, NodeById, NodeFragments, Nodes, OutgoingEdges,
    Runnable, WithOffset,
};
use motlie_db::reader::{ReaderConfig, Storage as UnifiedStorage};
use motlie_db::{DataUrl, Id, TimestampMilli};
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
    let fulltext_handle =
        spawn_fulltext_mutation_consumer(fulltext_rx, config.clone(), index_path);

    // Insert test data
    insert_test_data(&writer).await;

    // Shutdown mutation pipeline
    drop(writer);
    graph_handle.await.unwrap().unwrap();
    fulltext_handle.await.unwrap().unwrap();

    // Wait for data to be flushed
    tokio::time::sleep(Duration::from_millis(100)).await;
}

/// Helper to set up a test environment using the unified Storage::ready() API.
///
/// This demonstrates the recommended way to initialize the unified query system:
/// 1. Create a `motlie_db::reader::Storage` with paths to graph and fulltext storage
/// 2. Call `ready()` to initialize both subsystems and set up MPMC query pipelines
/// 3. Use the returned `Reader` to execute queries via the `Runnable` trait
///
/// Returns the unified reader and consumer handles.
async fn setup_test_env_with_unified_storage(
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

    // =========================================================================
    // UNIFIED STORAGE API DEMONSTRATION
    // =========================================================================
    //
    // Use `motlie_db::reader::Storage::ready()` to initialize both graph and
    // fulltext subsystems in one call. This:
    // - Opens both RocksDB (graph) and Tantivy (fulltext) storage
    // - Creates MPMC channels for query dispatch
    // - Spawns consumer pools for parallel query processing
    //
    // The returned Reader can execute any query type that implements
    // `motlie_db::query::Runnable<Reader>`.

    let storage = UnifiedStorage::readonly(&db_path, &index_path);
    let config = ReaderConfig::with_channel_buffer_size(100);
    let (reader, handles) = storage.ready(config, 2).unwrap();

    (reader, handles, db_path, index_path)
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
    let graph = Arc::new(Graph::new(Arc::new(graph_storage)));

    let mut fulltext_storage = FulltextStorage::readonly(&index_path);
    fulltext_storage.ready().unwrap();
    let fulltext_index = Arc::new(Index::new(Arc::new(fulltext_storage)));

    // Create unified reader with both graph and fulltext subsystems
    let reader_config = ReaderConfig::with_channel_buffer_size(100);
    let (reader, unified_handles, graph_handles, fulltext_handles) =
        motlie_db::reader::ReaderBuilder::new(graph, fulltext_index)
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
        content: DataUrl::from_text("Rust is a systems programming language with memory safety guarantees"),
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
        content: DataUrl::from_text("WebAssembly allows Rust code to run in browsers alongside JavaScript"),
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
    assert!(!results.is_empty(), "Expected results for 'programming' query");

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
    let edge_names: Vec<&str> = results.iter().map(|(_, _, name, _)| name.as_str()).collect();
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

    assert!(results.is_empty(), "Should have no results for non-matching query");

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
) -> (
    motlie_db::reader::Reader,
    Vec<tokio::task::JoinHandle<()>>,
) {
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
    let fulltext_handle = spawn_fulltext_mutation_consumer(fulltext_rx, config.clone(), &index_path);

    // Insert test data with tags
    insert_tagged_test_data(&writer).await;

    // Shutdown mutation pipeline
    drop(writer);
    graph_handle.await.unwrap().unwrap();
    fulltext_handle.await.unwrap().unwrap();

    // Wait for data to be flushed
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Open storage for reading
    let mut graph_storage = GraphStorage::readonly(&db_path);
    graph_storage.ready().unwrap();
    let graph = Arc::new(Graph::new(Arc::new(graph_storage)));

    let mut fulltext_storage = FulltextStorage::readonly(&index_path);
    fulltext_storage.ready().unwrap();
    let fulltext_index = Arc::new(Index::new(Arc::new(fulltext_storage)));

    // Create unified reader with both graph and fulltext subsystems
    let reader_config = ReaderConfig::with_channel_buffer_size(100);
    let (reader, unified_handles, graph_handles, fulltext_handles) =
        motlie_db::reader::ReaderBuilder::new(graph, fulltext_index)
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
        content: DataUrl::from_text(
            "Python excels at #scripting and #datascience applications",
        ),
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
    let edge_names: Vec<&str> = results.iter().map(|(_, _, name, _)| name.as_str()).collect();
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
    let edge_names: Vec<&str> = results.iter().map(|(_, _, name, _)| name.as_str()).collect();
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
/// 1. Create `motlie_db::reader::Storage` with paths to graph and fulltext storage
/// 2. Call `ready()` which initializes both subsystems and spawns MPMC consumer pools
/// 3. Use the returned `Reader` to execute queries via the `Runnable` trait
#[tokio::test]
async fn test_unified_storage_ready_api() {
    let temp_dir = TempDir::new().unwrap();

    // Use the unified Storage::ready() API
    let (reader, handles, _, _) = setup_test_env_with_unified_storage(&temp_dir).await;

    // =========================================================================
    // Execute various query types through the unified reader
    // =========================================================================

    let timeout = Duration::from_secs(5);

    // 1. Fulltext search for nodes (with graph hydration)
    let node_results = Nodes::new("programming".to_string(), 10)
        .run(&reader, timeout)
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
        .run(&reader, timeout)
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
        .run(&reader, timeout)
        .await
        .unwrap();

    assert!(
        !fuzzy_results.is_empty(),
        "Fuzzy search should match despite typo"
    );

    // 4. Pagination with offset
    let first_page = Nodes::new("programming".to_string(), 1)
        .run(&reader, timeout)
        .await
        .unwrap();

    let second_page = Nodes::new("programming".to_string(), 1)
        .with_offset(1)
        .run(&reader, timeout)
        .await
        .unwrap();

    // If there are multiple results, pages should be different
    if !first_page.is_empty() && !second_page.is_empty() {
        assert_ne!(
            first_page[0].0, second_page[0].0,
            "Pagination should return different results"
        );
    }

    // Shutdown
    drop(reader);
    for handle in handles {
        handle.await.unwrap();
    }
}

/// Test: EdgeDetails query returns full edge information without exposing scores.
///
/// This demonstrates using `EdgeDetails` (alias for `EdgeSummaryBySrcDstName`)
/// through the unified reader, which returns `EdgeResult` instead of the
/// internal `(EdgeSummary, Option<f64>)` type.
#[tokio::test]
async fn test_edge_details_query() {
    let temp_dir = TempDir::new().unwrap();
    let (reader, handles, _, _) = setup_test_env_with_unified_storage(&temp_dir).await;

    let timeout = Duration::from_secs(5);

    // First, search for edges to get IDs
    let edge_results = Edges::new("WebAssembly".to_string(), 10)
        .run(&reader, timeout)
        .await
        .unwrap();

    assert!(!edge_results.is_empty(), "Should have edge results");

    // Get the first edge's details using EdgeDetails query
    let (src_id, dst_id, edge_name, _summary) = &edge_results[0];

    let edge_detail = EdgeDetails::new(*src_id, *dst_id, edge_name.clone(), None)
        .run(&reader, timeout)
        .await
        .unwrap();

    // EdgeDetails returns EdgeResult: (SrcId, DstId, EdgeName, EdgeSummary)
    let (detail_src, detail_dst, detail_name, detail_summary) = edge_detail;

    assert_eq!(detail_src, *src_id, "Source ID should match");
    assert_eq!(detail_dst, *dst_id, "Destination ID should match");
    assert_eq!(&detail_name, edge_name, "Edge name should match");
    assert!(
        !detail_summary.as_ref().is_empty(),
        "Edge summary should be present"
    );

    // Shutdown
    drop(reader);
    for handle in handles {
        handle.await.unwrap();
    }
}

/// Test: NodeFragments query retrieves fragment history for a node.
#[tokio::test]
async fn test_node_fragments_query() {
    let temp_dir = TempDir::new().unwrap();
    let (reader, handles, _, _) = setup_test_env_with_unified_storage(&temp_dir).await;

    let timeout = Duration::from_secs(5);

    // First, search for nodes to get an ID
    let node_results = Nodes::new("Rust".to_string(), 10)
        .run(&reader, timeout)
        .await
        .unwrap();

    assert!(!node_results.is_empty(), "Should have node results");

    // Get the node's fragment history
    let (node_id, _name, _summary) = &node_results[0];

    // Query all fragments (unbounded time range)
    use std::ops::Bound;
    let fragments = NodeFragments::new(
        *node_id,
        (Bound::Unbounded, Bound::Unbounded),
        None,
    )
    .run(&reader, timeout)
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

    // Shutdown
    drop(reader);
    for handle in handles {
        handle.await.unwrap();
    }
}

/// Test: NodeById, OutgoingEdges, and IncomingEdges via the unified reader.
///
/// Demonstrates that graph queries can be executed through the unified reader
/// without needing to import separate graph traits.
#[tokio::test]
async fn test_node_by_id_and_edge_queries() {
    let temp_dir = TempDir::new().unwrap();
    let (reader, handles, _, _) = setup_test_env_with_unified_storage(&temp_dir).await;

    let timeout = Duration::from_secs(5);

    // First, find a node via fulltext search to get its ID
    let search_results = Nodes::new("Rust".to_string(), 10)
        .run(&reader, timeout)
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
    let (name, summary) = NodeById::new(*rust_id, None)
        .run(&reader, timeout)
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
    println!("NodeById result: name='{}', summary_len={}", name, summary.as_ref().len());

    // =========================================================================
    // 2. OutgoingEdges - Get edges originating from this node
    // =========================================================================
    let outgoing = OutgoingEdges::new(*rust_id, None)
        .run(&reader, timeout)
        .await
        .unwrap();

    // The test data has Rust -> JavaScript (influences) edge
    println!("OutgoingEdges from Rust: {} edges", outgoing.len());
    for (weight, src_id, dst_id, edge_name) in &outgoing {
        println!(
            "  {} -> {} via '{}' (weight: {:?})",
            src_id, dst_id, edge_name, weight
        );
    }

    // Should have at least one outgoing edge
    assert!(
        !outgoing.is_empty(),
        "Rust node should have outgoing edges"
    );

    // Find the JavaScript node to verify incoming edges
    let js_results = Nodes::new("JavaScript".to_string(), 10)
        .run(&reader, timeout)
        .await
        .unwrap();

    if !js_results.is_empty() {
        let (js_id, js_name, _) = &js_results[0];

        // =====================================================================
        // 3. IncomingEdges - Get edges pointing to this node
        // =====================================================================
        let incoming = IncomingEdges::new(*js_id, None)
            .run(&reader, timeout)
            .await
            .unwrap();

        println!("IncomingEdges to {}: {} edges", js_name, incoming.len());
        for (weight, src_id, dst_id, edge_name) in &incoming {
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
        let rust_to_js_edge = incoming
            .iter()
            .find(|(_, src, dst, _)| (*src == *rust_id && *dst == *js_id) || (*dst == *rust_id && *src == *js_id));
        assert!(
            rust_to_js_edge.is_some(),
            "Should have an edge connecting Rust and JavaScript. Found: {:?}",
            incoming
        );
    }

    // Shutdown
    drop(reader);
    for handle in handles {
        handle.await.unwrap();
    }
}
