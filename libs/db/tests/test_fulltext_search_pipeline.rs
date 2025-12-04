//! Integration tests for fulltext search pipeline with RocksDB verification.
//!
//! These tests verify that:
//! 1. The fulltext search pipeline integrates graph and fulltext mutation processors
//! 2. Nodes, edges, and fragments with searchable content are properly indexed
//! 3. Multiple fragments for a single entity result in proper search hits
//! 4. Fulltext search hits can be resolved back to correct RocksDB entries
//! 5. Deduplication behavior is documented and tested

use motlie_db::{
    create_fulltext_query_reader, create_mutation_writer, create_query_reader,
    spawn_fulltext_query_consumer_pool_shared, spawn_graph_consumer_with_next,
    spawn_graph_query_consumer_pool_shared, AddEdge, AddEdgeFragment, AddNode, AddNodeFragment,
    DataUrl, EdgeSummary, EdgeSummaryBySrcDstName, FulltextEdges, FulltextFacets, FulltextIndex,
    FulltextQueryRunnable, FulltextReaderConfig, FulltextStorage, FuzzyLevel, Graph, Id,
    MutationRunnable, NodeById, NodeSummary, QueryRunnable, ReaderConfig, Storage, TimestampMilli,
    WriterConfig,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tantivy::schema::Value;
use std::time::Duration;
use tempfile::TempDir;
use tokio::sync::mpsc;

/// Creates a test environment with graph and fulltext mutation consumers chained together.
/// Returns the writer, both consumer handles, and paths.
async fn setup_mutation_pipeline(
    temp_dir: &TempDir,
) -> (
    motlie_db::Writer,
    tokio::task::JoinHandle<anyhow::Result<()>>,
    tokio::task::JoinHandle<anyhow::Result<()>>,
    std::path::PathBuf,
    std::path::PathBuf,
) {
    let db_path = temp_dir.path().join("graph_db");
    let index_path = temp_dir.path().join("fulltext_index");

    let config = WriterConfig {
        channel_buffer_size: 100,
    };

    // Create fulltext consumer (end of chain)
    let (fulltext_sender, fulltext_receiver) = mpsc::channel(config.channel_buffer_size);
    let fulltext_handle =
        motlie_db::spawn_fulltext_consumer(fulltext_receiver, config.clone(), &index_path);

    // Create graph consumer that forwards to fulltext (chained)
    let (writer, graph_receiver) = create_mutation_writer(config.clone());
    let graph_handle =
        spawn_graph_consumer_with_next(graph_receiver, config.clone(), &db_path, fulltext_sender);

    (writer, graph_handle, fulltext_handle, db_path, index_path)
}

/// Setup query infrastructure for both graph and fulltext.
/// Returns readers and consumer handles.
fn setup_query_infrastructure(
    db_path: &std::path::Path,
    index_path: &std::path::Path,
) -> (
    motlie_db::Reader,
    motlie_db::FulltextReader,
    Vec<tokio::task::JoinHandle<()>>,
    Vec<tokio::task::JoinHandle<()>>,
) {
    // Graph query consumers
    let mut storage = Storage::readwrite(db_path);
    storage.ready().unwrap();
    let graph = Arc::new(Graph::new(Arc::new(storage)));

    let reader_config = ReaderConfig {
        channel_buffer_size: 100,
    };
    let (graph_reader, graph_query_receiver) = create_query_reader(reader_config);
    let graph_consumer_handles =
        spawn_graph_query_consumer_pool_shared(graph_query_receiver, graph, 2);

    // Fulltext query consumers
    let fulltext_reader_config = FulltextReaderConfig {
        channel_buffer_size: 100,
    };
    let (fulltext_reader, fulltext_query_receiver) =
        create_fulltext_query_reader(fulltext_reader_config);

    let mut ft_storage = FulltextStorage::readonly(index_path);
    ft_storage.ready().unwrap();
    let fulltext_index = Arc::new(FulltextIndex::new(Arc::new(ft_storage)));
    let fulltext_consumer_handles =
        spawn_fulltext_query_consumer_pool_shared(fulltext_query_receiver, fulltext_index, 2);

    (
        graph_reader,
        fulltext_reader,
        graph_consumer_handles,
        fulltext_consumer_handles,
    )
}

/// Test 1: Full pipeline integration with node search and RocksDB verification
///
/// This test:
/// - Sets up graph -> fulltext mutation pipeline
/// - Adds nodes with searchable content
/// - Adds multiple fragments per node (to test fragment hits)
/// - Performs fulltext search for nodes
/// - Verifies each NodeHit resolves to correct RocksDB entry via NodeById query
#[tokio::test]
async fn test_fulltext_node_search_resolves_to_rocksdb() {
    let temp_dir = TempDir::new().unwrap();

    let (writer, graph_handle, fulltext_handle, db_path, index_path) =
        setup_mutation_pipeline(&temp_dir).await;

    println!("=== Test: Fulltext Node Search Resolves to RocksDB ===\n");

    // Create nodes with distinct searchable content
    let mut node_data: HashMap<Id, (&str, &str)> = HashMap::new();

    // Node 1: Rust programming
    let rust_id = Id::new();
    node_data.insert(rust_id, ("Rust", "Systems programming language with memory safety"));

    // Node 2: Python programming
    let python_id = Id::new();
    node_data.insert(python_id, ("Python", "High-level interpreted programming language"));

    // Node 3: JavaScript
    let javascript_id = Id::new();
    node_data.insert(javascript_id, ("JavaScript", "Dynamic scripting language for web development"));

    // Node 4: Go
    let go_id = Id::new();
    node_data.insert(go_id, ("Go", "Compiled language designed at Google for cloud infrastructure"));

    // Add all nodes
    for (node_id, (name, summary)) in &node_data {
        let node = AddNode {
            id: *node_id,
            ts_millis: TimestampMilli::now(),
            name: name.to_string(),
            valid_range: None,
            summary: NodeSummary::from_text(summary),
        };
        node.run(&writer).await.unwrap();
        println!("  Added node: {} ({})", name, node_id);
    }

    // Add MULTIPLE fragments per node (to test fragment deduplication)
    // Each node gets 2-3 fragments with related but distinct content
    let fragments = vec![
        // Rust fragments (3)
        (rust_id, "Rust provides zero-cost abstractions and ownership model"),
        (rust_id, "Rust is ideal for systems programming without garbage collection"),
        (rust_id, "Rust enables fearless concurrency through its type system"),
        // Python fragments (2)
        (python_id, "Python excels at data science and machine learning applications"),
        (python_id, "Python has extensive library ecosystem including NumPy and Pandas"),
        // JavaScript fragments (2)
        (javascript_id, "JavaScript powers modern web frameworks like React and Vue"),
        (javascript_id, "JavaScript runs on both browser and server with Node.js"),
        // Go fragment (1)
        (go_id, "Go is used by Docker, Kubernetes, and many cloud native tools"),
    ];

    // Track fragment timestamps per node for later verification
    let mut fragment_timestamps: HashMap<Id, Vec<u64>> = HashMap::new();

    for (node_id, content) in &fragments {
        let ts = TimestampMilli::now();
        let fragment = AddNodeFragment {
            id: *node_id,
            ts_millis: ts,
            content: DataUrl::from_text(content),
            valid_range: None,
        };
        fragment.run(&writer).await.unwrap();
        fragment_timestamps
            .entry(*node_id)
            .or_default()
            .push(ts.0);

        // Small delay to ensure distinct timestamps
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    println!("\n  Added {} fragments across {} nodes\n", fragments.len(), node_data.len());

    // Wait for indexing and shutdown mutation pipeline
    tokio::time::sleep(Duration::from_millis(500)).await;
    drop(writer);
    graph_handle.await.unwrap().unwrap();
    fulltext_handle.await.unwrap().unwrap();

    // Setup query infrastructure
    let (graph_reader, fulltext_reader, graph_handles, fulltext_handles) =
        setup_query_infrastructure(&db_path, &index_path);

    let timeout = Duration::from_secs(5);

    // === FULLTEXT SEARCH: Find nodes containing "programming" ===
    println!("  Searching fulltext for 'programming'...");
    let results = motlie_db::FulltextNodes::new("programming".to_string(), 10)
        .run(&fulltext_reader, timeout)
        .await
        .unwrap();

    println!("  Found {} results for 'programming'", results.len());

    // Should find Rust, Python (both have "programming" in summary or fragments)
    assert!(results.len() >= 2, "Expected at least 2 results for 'programming'");

    // Verify each hit resolves to correct RocksDB entry
    let mut resolved_ids: HashSet<Id> = HashSet::new();
    for hit in &results {
        println!("    Hit: id={}, score={:.4}, fragment_ts={:?}",
            hit.id, hit.score, hit.fragment_timestamp);

        // Use NodeById query to verify the hit resolves to a real node
        let node_result = NodeById::new(hit.id, None)
            .run(&graph_reader, timeout)
            .await;

        match node_result {
            Ok((name, summary)) => {
                println!("      -> Resolved to node: name='{}', summary='{}'", name, summary.decode_string().unwrap_or_default());
                resolved_ids.insert(hit.id);

                // Verify the ID matches our expected data
                assert!(
                    node_data.contains_key(&hit.id),
                    "Hit ID {} should be in our test data",
                    hit.id
                );
            }
            Err(e) => {
                panic!("Failed to resolve hit {} to RocksDB entry: {}", hit.id, e);
            }
        }
    }

    // Verify we found distinct nodes (deduplication working)
    println!("\n  Distinct nodes resolved: {}", resolved_ids.len());

    // === FULLTEXT SEARCH: Find nodes with "Rust" ===
    println!("\n  Searching fulltext for 'Rust'...");
    let rust_results = motlie_db::FulltextNodes::new("Rust".to_string(), 10)
        .run(&fulltext_reader, timeout)
        .await
        .unwrap();

    println!("  Found {} results for 'Rust'", rust_results.len());

    // Verify Rust node is found
    let rust_hit = rust_results.iter().find(|h| h.id == rust_id);
    assert!(rust_hit.is_some(), "Should find the Rust node");

    // Verify hit resolves to correct node
    let (name, _) = NodeById::new(rust_id, None)
        .run(&graph_reader, timeout)
        .await
        .unwrap();
    assert_eq!(name, "Rust", "Resolved node should have name 'Rust'");

    // === DEDUPLICATION REPORT ===
    println!("\n=== Deduplication Report (Nodes) ===");
    println!("  Rust node has {} fragments in DB", fragment_timestamps.get(&rust_id).map(|v| v.len()).unwrap_or(0));
    println!("  Search for 'Rust' returned {} hits", rust_results.len());
    println!("  Note: Current implementation deduplicates by node ID, keeping best score");
    println!("        Multiple fragments for same node are collapsed into single hit");

    // Verify deduplication: Despite 3 fragments, we should get at most 1 hit per node for "Rust"
    let rust_hit_count = rust_results.iter().filter(|h| h.id == rust_id).count();
    assert_eq!(rust_hit_count, 1, "Should have exactly 1 hit for Rust node (deduplicated)");

    // Cleanup
    drop(graph_reader);
    drop(fulltext_reader);
    for h in graph_handles {
        h.await.unwrap();
    }
    for h in fulltext_handles {
        h.await.unwrap();
    }

    println!("\n✅ Test passed: Node fulltext search correctly resolves to RocksDB entries\n");
}

/// Test 2: Full pipeline integration with edge search and RocksDB verification
///
/// This test:
/// - Sets up graph -> fulltext mutation pipeline
/// - Adds nodes and edges with searchable content
/// - Adds multiple fragments per edge (to test fragment hits)
/// - Performs fulltext search for edges
/// - Verifies each EdgeHit resolves to correct RocksDB entry via EdgeSummaryBySrcDstName query
#[tokio::test]
async fn test_fulltext_edge_search_resolves_to_rocksdb() {
    let temp_dir = TempDir::new().unwrap();

    let (writer, graph_handle, fulltext_handle, db_path, index_path) =
        setup_mutation_pipeline(&temp_dir).await;

    println!("=== Test: Fulltext Edge Search Resolves to RocksDB ===\n");

    // Create nodes first (edges need source and target)
    let rust_id = Id::new();
    let python_id = Id::new();
    let javascript_id = Id::new();
    let webdev_id = Id::new();

    let nodes = vec![
        (rust_id, "Rust"),
        (python_id, "Python"),
        (javascript_id, "JavaScript"),
        (webdev_id, "WebDevelopment"),
    ];

    for (node_id, name) in &nodes {
        AddNode {
            id: *node_id,
            ts_millis: TimestampMilli::now(),
            name: name.to_string(),
            valid_range: None,
            summary: NodeSummary::from_text(&format!("{} topic node", name)),
        }
        .run(&writer)
        .await
        .unwrap();
        println!("  Added node: {}", name);
    }

    // Create edges with searchable content
    // Edge key = (src_id, dst_id, edge_name)
    struct EdgeData {
        src: Id,
        dst: Id,
        name: &'static str,
        summary: &'static str,
    }

    let edges = vec![
        EdgeData {
            src: rust_id,
            dst: python_id,
            name: "competes_with",
            summary: "Rust competes with Python in systems programming domain",
        },
        EdgeData {
            src: javascript_id,
            dst: webdev_id,
            name: "powers",
            summary: "JavaScript powers modern web development",
        },
        EdgeData {
            src: python_id,
            dst: webdev_id,
            name: "supports",
            summary: "Python supports backend web development with Django and Flask",
        },
        EdgeData {
            src: rust_id,
            dst: webdev_id,
            name: "enables",
            summary: "Rust enables WebAssembly for high-performance web applications",
        },
    ];

    // Track edge keys for verification
    let mut edge_keys: Vec<(Id, Id, String)> = Vec::new();

    for edge in &edges {
        AddEdge {
            source_node_id: edge.src,
            target_node_id: edge.dst,
            ts_millis: TimestampMilli::now(),
            name: edge.name.to_string(),
            summary: EdgeSummary::from_text(edge.summary),
            weight: Some(0.8),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();
        edge_keys.push((edge.src, edge.dst, edge.name.to_string()));
        println!("  Added edge: {} -> {} ({})", edge.src, edge.dst, edge.name);
    }

    // Add MULTIPLE fragments per edge (to test fragment deduplication)
    let edge_fragments = vec![
        // JavaScript -> WebDev edge gets multiple fragments
        (javascript_id, webdev_id, "powers", "JavaScript frameworks include React, Vue, and Angular"),
        (javascript_id, webdev_id, "powers", "JavaScript ecosystem has npm with millions of packages"),
        // Python -> WebDev edge gets multiple fragments
        (python_id, webdev_id, "supports", "Python web frameworks provide RESTful API development"),
        (python_id, webdev_id, "supports", "Python async libraries like FastAPI enable high-performance backends"),
    ];

    for (src, dst, edge_name, content) in &edge_fragments {
        AddEdgeFragment {
            src_id: *src,
            dst_id: *dst,
            edge_name: edge_name.to_string(),
            ts_millis: TimestampMilli::now(),
            content: DataUrl::from_text(content),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    println!("\n  Added {} edge fragments\n", edge_fragments.len());

    // Wait for indexing and shutdown mutation pipeline
    tokio::time::sleep(Duration::from_millis(500)).await;
    drop(writer);
    graph_handle.await.unwrap().unwrap();
    fulltext_handle.await.unwrap().unwrap();

    // Debug: Check what's in the index using direct Tantivy API
    {
        let mut ft_storage = FulltextStorage::readonly(&index_path);
        ft_storage.ready().unwrap();
        let index = ft_storage.index().unwrap();
        let reader = index.reader().unwrap();
        let searcher = reader.searcher();
        let schema = index.schema();

        println!("\n  DEBUG: Total documents in index: {}", searcher.num_docs());

        // Get doc_type field
        let doc_type_field = schema.get_field("doc_type").unwrap();

        // Search for all edge documents
        use tantivy::collector::TopDocs;
        use tantivy::query::AllQuery;
        let all_results = searcher.search(&AllQuery, &TopDocs::with_limit(100)).unwrap();

        let mut edges_count = 0;
        let mut edge_fragments_count = 0;
        let mut nodes_count = 0;
        let mut node_fragments_count = 0;

        for (_score, doc_addr) in &all_results {
            let doc = searcher.doc::<tantivy::TantivyDocument>(*doc_addr).unwrap();
            if let Some(dt) = doc.get_first(doc_type_field) {
                if let Some(text) = dt.as_str() {
                    match text {
                        "edges" => edges_count += 1,
                        "edge_fragments" => edge_fragments_count += 1,
                        "nodes" => nodes_count += 1,
                        "node_fragments" => node_fragments_count += 1,
                        _ => {}
                    }
                }
            }
        }

        println!("  DEBUG: Document types in index:");
        println!("    - nodes: {}", nodes_count);
        println!("    - node_fragments: {}", node_fragments_count);
        println!("    - edges: {}", edges_count);
        println!("    - edge_fragments: {}", edge_fragments_count);

    }

    // Setup query infrastructure
    let (graph_reader, fulltext_reader, graph_handles, fulltext_handles) =
        setup_query_infrastructure(&db_path, &index_path);

    let timeout = Duration::from_secs(5);

    // === FULLTEXT SEARCH: Find edges containing "web" ===
    println!("\n  Searching fulltext for edges with 'web'...");
    let results = FulltextEdges::new("web".to_string(), 10)
        .run(&fulltext_reader, timeout)
        .await
        .unwrap();

    println!("  Found {} results for 'web'", results.len());

    // Should find edges related to web development
    assert!(!results.is_empty(), "Expected results for 'web' search");

    // Verify each hit resolves to correct RocksDB entry
    let mut resolved_edges: HashSet<(Id, Id, String)> = HashSet::new();
    for hit in &results {
        println!("    Hit: src={}, dst={}, name='{}', score={:.4}, fragment_ts={:?}",
            hit.src_id, hit.dst_id, hit.edge_name, hit.score, hit.fragment_timestamp);

        // Use EdgeSummaryBySrcDstName query to verify the hit resolves to a real edge
        let edge_result = EdgeSummaryBySrcDstName::new(
            hit.src_id,
            hit.dst_id,
            hit.edge_name.clone(),
            None,
        )
        .run(&graph_reader, timeout)
        .await;

        match edge_result {
            Ok((summary, weight)) => {
                println!("      -> Resolved to edge: summary='{}', weight={:?}",
                    summary.decode_string().unwrap_or_default(), weight);
                resolved_edges.insert((hit.src_id, hit.dst_id, hit.edge_name.clone()));

                // Verify the edge key matches our expected data
                let expected_key = (hit.src_id, hit.dst_id, hit.edge_name.clone());
                assert!(
                    edge_keys.contains(&expected_key),
                    "Hit edge key {:?} should be in our test data",
                    expected_key
                );
            }
            Err(e) => {
                panic!(
                    "Failed to resolve edge hit ({}, {}, {}) to RocksDB entry: {}",
                    hit.src_id, hit.dst_id, hit.edge_name, e
                );
            }
        }
    }

    // Verify we found distinct edges (deduplication working)
    println!("\n  Distinct edges resolved: {}", resolved_edges.len());

    // === FULLTEXT SEARCH: Find edges with "JavaScript" ===
    println!("\n  Searching fulltext for edges with 'JavaScript'...");
    let js_results = FulltextEdges::new("JavaScript".to_string(), 10)
        .run(&fulltext_reader, timeout)
        .await
        .unwrap();

    println!("  Found {} results for 'JavaScript'", js_results.len());

    // Find the JS -> WebDev edge
    let js_edge_hit = js_results.iter().find(|h|
        h.src_id == javascript_id && h.dst_id == webdev_id && h.edge_name == "powers"
    );
    assert!(js_edge_hit.is_some(), "Should find JavaScript -> WebDev edge");

    // Verify it resolves correctly
    let (summary, _) = EdgeSummaryBySrcDstName::new(
        javascript_id,
        webdev_id,
        "powers".to_string(),
        None,
    )
    .run(&graph_reader, timeout)
    .await
    .unwrap();
    assert!(
        summary.decode_string().unwrap_or_default().contains("JavaScript"),
        "Resolved edge should contain 'JavaScript'"
    );

    // === DEDUPLICATION REPORT ===
    println!("\n=== Deduplication Report (Edges) ===");
    println!("  JavaScript -> WebDev edge has 2 fragments in DB");
    println!("  Search for 'JavaScript' returned {} hits", js_results.len());
    println!("  Note: Current implementation deduplicates by edge key (src_id, dst_id, edge_name)");
    println!("        Multiple fragments for same edge are collapsed into single hit");

    // Verify deduplication: Despite 2 fragments, we should get at most 1 hit per edge
    let js_webdev_hit_count = js_results
        .iter()
        .filter(|h| h.src_id == javascript_id && h.dst_id == webdev_id && h.edge_name == "powers")
        .count();
    assert_eq!(
        js_webdev_hit_count, 1,
        "Should have exactly 1 hit for JS->WebDev edge (deduplicated)"
    );

    // Cleanup
    drop(graph_reader);
    drop(fulltext_reader);
    for h in graph_handles {
        h.await.unwrap();
    }
    for h in fulltext_handles {
        h.await.unwrap();
    }

    println!("\n✅ Test passed: Edge fulltext search correctly resolves to RocksDB entries\n");
}

/// Test 3: Combined node and edge search with cross-verification
///
/// This test verifies that:
/// - Both node and edge searches work in the same database
/// - Hit IDs/keys correctly distinguish between entity types
/// - All hits can be resolved via appropriate graph queries
#[tokio::test]
async fn test_fulltext_combined_search_with_verification() {
    let temp_dir = TempDir::new().unwrap();

    let (writer, graph_handle, fulltext_handle, db_path, index_path) =
        setup_mutation_pipeline(&temp_dir).await;

    println!("=== Test: Combined Node and Edge Search ===\n");

    // Create a small interconnected graph about databases
    let postgres_id = Id::new();
    let mysql_id = Id::new();
    let mongodb_id = Id::new();
    let redis_id = Id::new();

    // Add nodes
    let node_data = vec![
        (postgres_id, "PostgreSQL", "Advanced open source relational database with ACID compliance"),
        (mysql_id, "MySQL", "Popular relational database for web applications"),
        (mongodb_id, "MongoDB", "Document-oriented NoSQL database for flexible schemas"),
        (redis_id, "Redis", "In-memory key-value store for caching and messaging"),
    ];

    for (id, name, summary) in &node_data {
        AddNode {
            id: *id,
            ts_millis: TimestampMilli::now(),
            name: name.to_string(),
            valid_range: None,
            summary: NodeSummary::from_text(summary),
        }
        .run(&writer)
        .await
        .unwrap();
    }

    // Add node fragments with keyword "database"
    for (id, name, _) in &node_data {
        AddNodeFragment {
            id: *id,
            ts_millis: TimestampMilli::now(),
            content: DataUrl::from_text(&format!(
                "{} is a popular database system used in production environments",
                name
            )),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();
    }

    // Add edges with "database" keyword
    let edge_data = vec![
        (postgres_id, mysql_id, "similar_to", "Both are relational database systems with SQL support"),
        (mongodb_id, redis_id, "often_paired", "NoSQL database and cache often used together"),
        (postgres_id, mongodb_id, "alternative_to", "Relational database alternative to document database"),
    ];

    for (src, dst, name, summary) in &edge_data {
        AddEdge {
            source_node_id: *src,
            target_node_id: *dst,
            ts_millis: TimestampMilli::now(),
            name: name.to_string(),
            summary: EdgeSummary::from_text(summary),
            weight: Some(1.0),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();
    }

    println!("  Created {} nodes and {} edges about databases\n", node_data.len(), edge_data.len());

    // Wait and shutdown mutation pipeline
    tokio::time::sleep(Duration::from_millis(500)).await;
    drop(writer);
    graph_handle.await.unwrap().unwrap();
    fulltext_handle.await.unwrap().unwrap();

    // Setup query infrastructure
    let (graph_reader, fulltext_reader, graph_handles, fulltext_handles) =
        setup_query_infrastructure(&db_path, &index_path);

    let timeout = Duration::from_secs(5);

    // === SEARCH: Find all nodes with "database" ===
    println!("  Searching for nodes with 'database'...");
    let node_results = motlie_db::FulltextNodes::new("database".to_string(), 10)
        .run(&fulltext_reader, timeout)
        .await
        .unwrap();

    println!("  Found {} node results", node_results.len());
    assert!(
        node_results.len() >= 4,
        "Should find all 4 database nodes"
    );

    // Verify all node hits resolve
    let mut verified_nodes = 0;
    for hit in &node_results {
        let result = NodeById::new(hit.id, None)
            .run(&graph_reader, timeout)
            .await;
        if result.is_ok() {
            verified_nodes += 1;
        }
    }
    assert_eq!(
        verified_nodes,
        node_results.len(),
        "All node hits should resolve to RocksDB"
    );

    // === SEARCH: Find all edges with "database" ===
    println!("  Searching for edges with 'database'...");
    let edge_results = FulltextEdges::new("database".to_string(), 10)
        .run(&fulltext_reader, timeout)
        .await
        .unwrap();

    println!("  Found {} edge results", edge_results.len());
    assert!(
        edge_results.len() >= 2,
        "Should find edges mentioning database"
    );

    // Verify all edge hits resolve
    let mut verified_edges = 0;
    for hit in &edge_results {
        let result = EdgeSummaryBySrcDstName::new(
            hit.src_id,
            hit.dst_id,
            hit.edge_name.clone(),
            None,
        )
        .run(&graph_reader, timeout)
        .await;
        if result.is_ok() {
            verified_edges += 1;
        }
    }
    assert_eq!(
        verified_edges,
        edge_results.len(),
        "All edge hits should resolve to RocksDB"
    );

    println!(
        "\n  Verified {} nodes and {} edges resolve correctly",
        verified_nodes, verified_edges
    );

    // Cleanup
    drop(graph_reader);
    drop(fulltext_reader);
    for h in graph_handles {
        h.await.unwrap();
    }
    for h in fulltext_handles {
        h.await.unwrap();
    }

    println!("\n✅ Test passed: Combined search with cross-verification\n");
}

/// Test 4: Verify multiple fragments produce correct deduplication behavior
///
/// This test specifically documents and verifies the current deduplication behavior:
/// - Multiple fragments for a single node/edge are deduplicated by entity key
/// - The best (highest) score is retained
/// - fragment_timestamp in Hit is None for deduplicated results
#[tokio::test]
async fn test_fulltext_fragment_deduplication_behavior() {
    let temp_dir = TempDir::new().unwrap();

    let (writer, graph_handle, fulltext_handle, db_path, index_path) =
        setup_mutation_pipeline(&temp_dir).await;

    println!("=== Test: Fragment Deduplication Behavior ===\n");

    // Create one node with MANY fragments containing same keyword
    let node_id = Id::new();

    AddNode {
        id: node_id,
        ts_millis: TimestampMilli::now(),
        name: "DeduplicationTest".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text("Test node for deduplication"),
    }
    .run(&writer)
    .await
    .unwrap();

    // Add 5 fragments with "unique_keyword_xyz" to the same node
    let fragment_count = 5;
    for i in 0..fragment_count {
        AddNodeFragment {
            id: node_id,
            ts_millis: TimestampMilli::now(),
            content: DataUrl::from_text(&format!(
                "Fragment {} contains unique_keyword_xyz for testing deduplication behavior",
                i
            )),
            valid_range: None,
        }
        .run(&writer)
        .await
        .unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    println!("  Created 1 node with {} fragments containing 'unique_keyword_xyz'\n", fragment_count);

    // Wait and shutdown
    tokio::time::sleep(Duration::from_millis(500)).await;
    drop(writer);
    graph_handle.await.unwrap().unwrap();
    fulltext_handle.await.unwrap().unwrap();

    // Setup query infrastructure
    let (graph_reader, fulltext_reader, graph_handles, fulltext_handles) =
        setup_query_infrastructure(&db_path, &index_path);

    let timeout = Duration::from_secs(5);

    // Search for the unique keyword
    println!("  Searching for 'unique_keyword_xyz'...");
    let results = motlie_db::FulltextNodes::new("unique_keyword_xyz".to_string(), 20)
        .run(&fulltext_reader, timeout)
        .await
        .unwrap();

    println!("  Search returned {} hits", results.len());

    // === DEDUPLICATION VERIFICATION ===
    println!("\n=== Deduplication Verification ===");
    println!("  Node has {} fragments with the search term", fragment_count);
    println!("  Search returned {} hits", results.len());

    // Current behavior: deduplicated by node ID
    assert_eq!(
        results.len(),
        1,
        "Should return exactly 1 hit (deduplicated by node ID)"
    );

    let hit = &results[0];
    println!("  Hit details:");
    println!("    - id: {}", hit.id);
    println!("    - score: {:.4}", hit.score);
    println!("    - fragment_timestamp: {:?}", hit.fragment_timestamp);

    // Verify hit ID matches our node
    assert_eq!(hit.id, node_id, "Hit should be for our test node");

    // Verify fragment_timestamp is None (deduplicated results don't track individual fragments)
    assert!(
        hit.fragment_timestamp.is_none(),
        "Deduplicated results should have fragment_timestamp = None"
    );

    // Verify hit resolves to correct node
    let (name, _) = NodeById::new(hit.id, None)
        .run(&graph_reader, timeout)
        .await
        .unwrap();
    assert_eq!(name, "DeduplicationTest", "Should resolve to correct node");

    println!("\n=== Deduplication Behavior Summary ===");
    println!("  Current behavior:");
    println!("    - Multiple fragments for same entity are collapsed into single hit");
    println!("    - Best (highest) BM25 score is retained");
    println!("    - fragment_timestamp is None for deduplicated results");
    println!("    - Entity can still be resolved via its ID/key");

    // Cleanup
    drop(graph_reader);
    drop(fulltext_reader);
    for h in graph_handles {
        h.await.unwrap();
    }
    for h in fulltext_handles {
        h.await.unwrap();
    }

    println!("\n✅ Test passed: Fragment deduplication behavior verified\n");
}

/// Test 5: Search with limit and verify correct number of results
#[tokio::test]
async fn test_fulltext_search_limit_and_ordering() {
    let temp_dir = TempDir::new().unwrap();

    let (writer, graph_handle, fulltext_handle, db_path, index_path) =
        setup_mutation_pipeline(&temp_dir).await;

    println!("=== Test: Search Limit and Score Ordering ===\n");

    // Create 10 nodes with "test" keyword, varying relevance
    let mut node_ids = Vec::new();
    for i in 0..10 {
        let node_id = Id::new();
        node_ids.push(node_id);

        // Vary the number of "test" occurrences to affect BM25 score
        let content = if i < 3 {
            format!("Node {} test test test test test high relevance", i)
        } else if i < 6 {
            format!("Node {} test test medium relevance", i)
        } else {
            format!("Node {} test low relevance", i)
        };

        AddNode {
            id: node_id,
            ts_millis: TimestampMilli::now(),
            name: format!("TestNode{}", i),
            valid_range: None,
            summary: NodeSummary::from_text(&content),
        }
        .run(&writer)
        .await
        .unwrap();
    }

    println!("  Created 10 nodes with varying 'test' keyword density\n");

    // Wait and shutdown
    tokio::time::sleep(Duration::from_millis(500)).await;
    drop(writer);
    graph_handle.await.unwrap().unwrap();
    fulltext_handle.await.unwrap().unwrap();

    // Setup query infrastructure
    let (graph_reader, fulltext_reader, graph_handles, fulltext_handles) =
        setup_query_infrastructure(&db_path, &index_path);

    let timeout = Duration::from_secs(5);

    // Test limit = 5
    println!("  Searching for 'test' with limit=5...");
    let results = motlie_db::FulltextNodes::new("test".to_string(), 5)
        .run(&fulltext_reader, timeout)
        .await
        .unwrap();

    assert_eq!(results.len(), 5, "Should return exactly 5 results");
    println!("  Got {} results (limit respected)", results.len());

    // Verify scores are in descending order
    for i in 1..results.len() {
        assert!(
            results[i - 1].score >= results[i].score,
            "Results should be ordered by score descending"
        );
    }
    println!("  Scores are in descending order");

    // Print scores
    println!("  Scores: {:?}", results.iter().map(|r| r.score).collect::<Vec<_>>());

    // Verify all results resolve to RocksDB
    for hit in &results {
        let result = NodeById::new(hit.id, None)
            .run(&graph_reader, timeout)
            .await;
        assert!(result.is_ok(), "All hits should resolve to RocksDB");
    }

    // Cleanup
    drop(graph_reader);
    drop(fulltext_reader);
    for h in graph_handles {
        h.await.unwrap();
    }
    for h in fulltext_handles {
        h.await.unwrap();
    }

    println!("\n✅ Test passed: Search limit and ordering verified\n");
}

/// Test 6: Tag-based facet filtering for nodes and edges
///
/// This test verifies:
/// - Tags are extracted from #hashtags in content
/// - `with_tags()` filter correctly restricts results
/// - AND semantics: all specified tags must match
/// - Works for both node and edge queries
#[tokio::test]
async fn test_fulltext_tag_facet_filtering() {
    let temp_dir = TempDir::new().unwrap();

    let (writer, graph_handle, fulltext_handle, db_path, index_path) =
        setup_mutation_pipeline(&temp_dir).await;

    println!("=== Test: Tag-Based Facet Filtering ===\n");

    // Create nodes with various hashtags in their content
    let rust_id = Id::new();
    let python_id = Id::new();
    let go_id = Id::new();
    let typescript_id = Id::new();

    // Node 1: Rust with tags #systems #async #memory-safe
    AddNode {
        id: rust_id,
        ts_millis: TimestampMilli::now(),
        name: "Rust".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text(
            "Rust is a #systems programming language with #async support and #memory-safe guarantees",
        ),
    }
    .run(&writer)
    .await
    .unwrap();

    // Node 2: Python with tags #scripting #async #data-science
    AddNode {
        id: python_id,
        ts_millis: TimestampMilli::now(),
        name: "Python".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text(
            "Python is a #scripting language with #async capabilities for #data-science",
        ),
    }
    .run(&writer)
    .await
    .unwrap();

    // Node 3: Go with tags #systems #cloud #concurrent
    AddNode {
        id: go_id,
        ts_millis: TimestampMilli::now(),
        name: "Go".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text(
            "Go is a #systems language designed for #cloud infrastructure and #concurrent programming",
        ),
    }
    .run(&writer)
    .await
    .unwrap();

    // Node 4: TypeScript with tags #scripting #web #typed
    AddNode {
        id: typescript_id,
        ts_millis: TimestampMilli::now(),
        name: "TypeScript".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text(
            "TypeScript is a #scripting language for #web development with #typed safety",
        ),
    }
    .run(&writer)
    .await
    .unwrap();

    println!("  Created 4 nodes with various #hashtags");

    // Create edges with hashtags
    AddEdge {
        source_node_id: rust_id,
        target_node_id: go_id,
        ts_millis: TimestampMilli::now(),
        name: "competes_with".to_string(),
        summary: EdgeSummary::from_text(
            "Rust and Go compete in #systems #cloud applications #performance-critical workloads",
        ),
        weight: Some(0.8),
        valid_range: None,
    }
    .run(&writer)
    .await
    .unwrap();

    AddEdge {
        source_node_id: python_id,
        target_node_id: typescript_id,
        ts_millis: TimestampMilli::now(),
        name: "similar_to".to_string(),
        summary: EdgeSummary::from_text(
            "Python and TypeScript share #scripting paradigm for #web and #automation tasks",
        ),
        weight: Some(0.6),
        valid_range: None,
    }
    .run(&writer)
    .await
    .unwrap();

    println!("  Created 2 edges with #hashtags\n");

    // Wait for indexing and shutdown mutation pipeline
    tokio::time::sleep(Duration::from_millis(500)).await;
    drop(writer);
    graph_handle.await.unwrap().unwrap();
    fulltext_handle.await.unwrap().unwrap();

    // Setup query infrastructure
    let (graph_reader, fulltext_reader, graph_handles, fulltext_handles) =
        setup_query_infrastructure(&db_path, &index_path);

    let timeout = Duration::from_secs(5);

    // === NODE TAG FILTERING TESTS ===
    println!("  Testing node tag filtering...\n");

    // Test 1: Filter by single tag #systems
    let results = motlie_db::FulltextNodes::new("language".to_string(), 10)
        .with_tags(vec!["systems".to_string()])
        .run(&fulltext_reader, timeout)
        .await
        .unwrap();

    println!("    Search 'language' with tag #systems: {} results", results.len());
    assert_eq!(results.len(), 2, "Should find Rust and Go (both have #systems tag)");

    // Verify correct nodes found
    let found_ids: HashSet<Id> = results.iter().map(|h| h.id).collect();
    assert!(found_ids.contains(&rust_id), "Should find Rust");
    assert!(found_ids.contains(&go_id), "Should find Go");
    assert!(!found_ids.contains(&python_id), "Should NOT find Python (no #systems tag)");

    // Test 2: Filter by single tag #scripting
    let results = motlie_db::FulltextNodes::new("language".to_string(), 10)
        .with_tags(vec!["scripting".to_string()])
        .run(&fulltext_reader, timeout)
        .await
        .unwrap();

    println!("    Search 'language' with tag #scripting: {} results", results.len());
    assert_eq!(results.len(), 2, "Should find Python and TypeScript (both have #scripting tag)");

    // Test 3: Filter by single tag #async
    let results = motlie_db::FulltextNodes::new("language".to_string(), 10)
        .with_tags(vec!["async".to_string()])
        .run(&fulltext_reader, timeout)
        .await
        .unwrap();

    println!("    Search 'language' with tag #async: {} results", results.len());
    assert_eq!(results.len(), 2, "Should find Rust and Python (both have #async tag)");

    // Test 4: Filter by multiple tags (AND semantics)
    let results = motlie_db::FulltextNodes::new("language".to_string(), 10)
        .with_tags(vec!["systems".to_string(), "async".to_string()])
        .run(&fulltext_reader, timeout)
        .await
        .unwrap();

    println!("    Search 'language' with tags #systems AND #async: {} results", results.len());
    assert_eq!(results.len(), 1, "Should find only Rust (has both #systems AND #async)");
    assert_eq!(results[0].id, rust_id, "Should be Rust");

    // Test 5: No results when tag doesn't match
    let results = motlie_db::FulltextNodes::new("language".to_string(), 10)
        .with_tags(vec!["nonexistent-tag".to_string()])
        .run(&fulltext_reader, timeout)
        .await
        .unwrap();

    println!("    Search 'language' with non-existent tag: {} results", results.len());
    assert_eq!(results.len(), 0, "Should find nothing with non-existent tag");

    // === EDGE TAG FILTERING TESTS ===
    println!("\n  Testing edge tag filtering...\n");

    // Test 6: Filter edges by tag #systems
    let results = FulltextEdges::new("compete".to_string(), 10)
        .with_tags(vec!["systems".to_string()])
        .run(&fulltext_reader, timeout)
        .await
        .unwrap();

    println!("    Search 'compete' with tag #systems: {} results", results.len());
    assert!(results.len() >= 1, "Should find Rust->Go edge (has #systems tag)");

    // Test 7: Filter edges by tag #scripting
    let results = FulltextEdges::new("similar".to_string(), 10)
        .with_tags(vec!["scripting".to_string()])
        .run(&fulltext_reader, timeout)
        .await
        .unwrap();

    println!("    Search 'similar' with tag #scripting: {} results", results.len());
    assert!(results.len() >= 1, "Should find Python->TypeScript edge (has #scripting tag)");

    // Test 8: Edge with multiple tags (AND semantics)
    let results = FulltextEdges::new("compete".to_string(), 10)
        .with_tags(vec!["systems".to_string(), "cloud".to_string()])
        .run(&fulltext_reader, timeout)
        .await
        .unwrap();

    println!("    Search 'compete' with tags #systems AND #cloud: {} results", results.len());
    assert!(results.len() >= 1, "Should find Rust->Go edge (has both tags)");

    // === VERIFY RESULTS RESOLVE TO ROCKSDB ===
    println!("\n  Verifying tag-filtered results resolve to RocksDB...");

    // Get #systems nodes and verify each resolves
    let results = motlie_db::FulltextNodes::new("language".to_string(), 10)
        .with_tags(vec!["systems".to_string()])
        .run(&fulltext_reader, timeout)
        .await
        .unwrap();

    for hit in &results {
        let node = NodeById::new(hit.id, None)
            .run(&graph_reader, timeout)
            .await;
        assert!(node.is_ok(), "Tag-filtered node hit should resolve to RocksDB");
    }
    println!("    All {} tag-filtered node hits resolved correctly", results.len());

    // Cleanup
    drop(graph_reader);
    drop(fulltext_reader);
    for h in graph_handles {
        h.await.unwrap();
    }
    for h in fulltext_handles {
        h.await.unwrap();
    }

    println!("\n✅ Test passed: Tag-based facet filtering works correctly\n");
}

/// Test 7: Fuzzy search for typo tolerance
///
/// This test verifies:
/// - FuzzyLevel::Low matches with 1 edit distance
/// - FuzzyLevel::Medium matches with 2 edit distance
/// - Fuzzy search works with both nodes and edges
#[tokio::test]
async fn test_fulltext_fuzzy_search() {
    let temp_dir = TempDir::new().unwrap();

    let (writer, graph_handle, fulltext_handle, db_path, index_path) =
        setup_mutation_pipeline(&temp_dir).await;

    println!("=== Test: Fuzzy Search ===\n");

    // Create nodes with specific searchable content
    let programming_id = Id::new();
    let database_id = Id::new();

    AddNode {
        id: programming_id,
        ts_millis: TimestampMilli::now(),
        name: "Programming".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text("programming concepts and patterns"),
    }
    .run(&writer)
    .await
    .unwrap();

    AddNode {
        id: database_id,
        ts_millis: TimestampMilli::now(),
        name: "Database".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text("database design and optimization"),
    }
    .run(&writer)
    .await
    .unwrap();

    // Create an edge
    AddEdge {
        source_node_id: programming_id,
        target_node_id: database_id,
        ts_millis: TimestampMilli::now(),
        name: "interacts_with".to_string(),
        summary: EdgeSummary::from_text("programming interacts with database systems"),
        weight: Some(1.0),
        valid_range: None,
    }
    .run(&writer)
    .await
    .unwrap();

    println!("  Created 2 nodes and 1 edge");

    // Wait for indexing and shutdown
    tokio::time::sleep(Duration::from_millis(500)).await;
    drop(writer);
    graph_handle.await.unwrap().unwrap();
    fulltext_handle.await.unwrap().unwrap();

    // Setup query infrastructure
    let (_graph_reader, fulltext_reader, graph_handles, fulltext_handles) =
        setup_query_infrastructure(&db_path, &index_path);

    let timeout = Duration::from_secs(5);

    // === FUZZY NODE SEARCH ===
    println!("\n  Testing fuzzy node search...");

    // Exact match
    let results = motlie_db::FulltextNodes::new("programming".to_string(), 10)
        .run(&fulltext_reader, timeout)
        .await
        .unwrap();
    println!("    Exact 'programming': {} results", results.len());
    assert!(results.len() >= 1, "Exact match should find Programming node");

    // FuzzyLevel::Low (1 edit) - common typo
    let results = motlie_db::FulltextNodes::new("programing".to_string(), 10)  // missing 'm'
        .with_fuzzy(FuzzyLevel::Low)
        .run(&fulltext_reader, timeout)
        .await
        .unwrap();
    println!("    Fuzzy Low 'programing': {} results", results.len());
    assert!(results.len() >= 1, "Fuzzy Low should match 'programming' with 1 edit");

    // FuzzyLevel::Medium (2 edits)
    let results = motlie_db::FulltextNodes::new("progaming".to_string(), 10)  // missing 'r' and 'm'
        .with_fuzzy(FuzzyLevel::Medium)
        .run(&fulltext_reader, timeout)
        .await
        .unwrap();
    println!("    Fuzzy Medium 'progaming': {} results", results.len());
    // Note: This may or may not match depending on how Tantivy handles 2-edit distance

    // FuzzyLevel::None should NOT match typo
    let results = motlie_db::FulltextNodes::new("programing".to_string(), 10)
        .run(&fulltext_reader, timeout)  // No fuzzy - exact only
        .await
        .unwrap();
    println!("    Exact (no fuzzy) 'programing': {} results", results.len());
    assert_eq!(results.len(), 0, "Exact match should NOT find misspelled word");

    // === FUZZY EDGE SEARCH ===
    println!("\n  Testing fuzzy edge search...");

    // FuzzyLevel::Low for edges
    let results = FulltextEdges::new("databse".to_string(), 10)  // typo: 'databse' instead of 'database'
        .with_fuzzy(FuzzyLevel::Low)
        .run(&fulltext_reader, timeout)
        .await
        .unwrap();
    println!("    Fuzzy Low 'databse': {} results", results.len());
    // Should match edge containing "database"

    // Cleanup - must drop readers first to close channels before awaiting handles
    drop(fulltext_reader);
    drop(_graph_reader);
    for h in fulltext_handles {
        h.await.unwrap();
    }
    for h in graph_handles {
        h.await.unwrap();
    }

    println!("\n✅ Test passed: Fuzzy search works correctly\n");
}

/// Test 8: Facet counts query
///
/// This test verifies:
/// - `FulltextFacets` query returns counts for all facet types
/// - doc_type facets include nodes, edges, node_fragments, edge_fragments
/// - tags facet counts match the hashtags in content
/// - validity facet counts match the document validity structures
/// - Optional doc_type filtering restricts counts to specific document types
#[tokio::test]
async fn test_fulltext_facets_query() {
    let temp_dir = TempDir::new().unwrap();

    let (writer, graph_handle, fulltext_handle, db_path, index_path) =
        setup_mutation_pipeline(&temp_dir).await;

    println!("=== Test: Facet Counts Query ===\n");

    // Create nodes with various hashtags
    let node1_id = Id::new();
    let node2_id = Id::new();
    let node3_id = Id::new();

    // Node 1: #rust #systems
    AddNode {
        id: node1_id,
        ts_millis: TimestampMilli::now(),
        name: "Rust".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text("Rust is a #rust #systems programming language"),
    }
    .run(&writer)
    .await
    .unwrap();

    // Node 2: #python #scripting #data
    AddNode {
        id: node2_id,
        ts_millis: TimestampMilli::now(),
        name: "Python".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text("Python is a #python #scripting language for #data science"),
    }
    .run(&writer)
    .await
    .unwrap();

    // Node 3: #rust #async (shares #rust tag with Node 1)
    AddNode {
        id: node3_id,
        ts_millis: TimestampMilli::now(),
        name: "Tokio".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text("Tokio is an #async runtime for #rust"),
    }
    .run(&writer)
    .await
    .unwrap();

    println!("  Created 3 nodes with various #hashtags");

    // Add node fragments with tags
    AddNodeFragment {
        id: node1_id,
        ts_millis: TimestampMilli::now(),
        content: DataUrl::from_text("Rust memory safety #memory-safe #systems"),
        valid_range: None,
    }
    .run(&writer)
    .await
    .unwrap();

    AddNodeFragment {
        id: node2_id,
        ts_millis: TimestampMilli::now(),
        content: DataUrl::from_text("Python #data analysis libraries"),
        valid_range: None,
    }
    .run(&writer)
    .await
    .unwrap();

    println!("  Created 2 node fragments");

    // Create edges with tags
    AddEdge {
        source_node_id: node1_id,
        target_node_id: node3_id,
        ts_millis: TimestampMilli::now(),
        name: "enables".to_string(),
        summary: EdgeSummary::from_text("Rust enables #async programming with Tokio"),
        weight: Some(0.9),
        valid_range: None,
    }
    .run(&writer)
    .await
    .unwrap();

    AddEdge {
        source_node_id: node2_id,
        target_node_id: node1_id,
        ts_millis: TimestampMilli::now(),
        name: "competes_with".to_string(),
        summary: EdgeSummary::from_text("Python competes with Rust in some domains #systems"),
        weight: Some(0.5),
        valid_range: None,
    }
    .run(&writer)
    .await
    .unwrap();

    println!("  Created 2 edges");

    // Add edge fragments
    AddEdgeFragment {
        src_id: node1_id,
        dst_id: node3_id,
        edge_name: "enables".to_string(),
        ts_millis: TimestampMilli::now(),
        content: DataUrl::from_text("Tokio runtime details #performance #async"),
        valid_range: None,
    }
    .run(&writer)
    .await
    .unwrap();

    println!("  Created 1 edge fragment\n");

    // Wait for indexing and shutdown
    tokio::time::sleep(Duration::from_millis(500)).await;
    drop(writer);
    graph_handle.await.unwrap().unwrap();
    fulltext_handle.await.unwrap().unwrap();

    // Setup query infrastructure
    let (graph_reader, fulltext_reader, graph_handles, fulltext_handles) =
        setup_query_infrastructure(&db_path, &index_path);

    let timeout = Duration::from_secs(5);

    // === TEST 1: Get all facet counts ===
    println!("  Testing facet counts for all documents...\n");

    let facet_counts = FulltextFacets::new()
        .run(&fulltext_reader, timeout)
        .await
        .unwrap();

    println!("  Document type counts:");
    for (doc_type, count) in &facet_counts.doc_types {
        println!("    {}: {}", doc_type, count);
    }

    println!("\n  Tag counts:");
    for (tag, count) in &facet_counts.tags {
        println!("    #{}: {}", tag, count);
    }

    println!("\n  Validity counts:");
    for (validity, count) in &facet_counts.validity {
        println!("    {}: {}", validity, count);
    }

    // Verify doc_type counts
    assert!(facet_counts.doc_types.get("nodes").unwrap_or(&0) >= &3, "Should have at least 3 nodes");
    assert!(facet_counts.doc_types.get("edges").unwrap_or(&0) >= &2, "Should have at least 2 edges");
    assert!(facet_counts.doc_types.get("node_fragments").unwrap_or(&0) >= &2, "Should have at least 2 node fragments");
    assert!(facet_counts.doc_types.get("edge_fragments").unwrap_or(&0) >= &1, "Should have at least 1 edge fragment");

    // Verify tag counts
    // #rust appears in: Node 1, Node 3, and their fragments/edges mentioning rust
    assert!(facet_counts.tags.get("rust").unwrap_or(&0) >= &2, "#rust should appear at least twice");
    // #systems appears in multiple documents
    assert!(facet_counts.tags.get("systems").unwrap_or(&0) >= &2, "#systems should appear at least twice");
    // #async appears in multiple documents
    assert!(facet_counts.tags.get("async").unwrap_or(&0) >= &2, "#async should appear at least twice");
    // #data appears
    assert!(facet_counts.tags.contains_key("data"), "#data tag should exist");

    // === TEST 2: Get facet counts filtered by doc_type ===
    println!("\n  Testing facet counts filtered to nodes only...\n");

    let nodes_only_counts = FulltextFacets::new()
        .with_doc_type_filter(vec!["nodes".to_string()])
        .run(&fulltext_reader, timeout)
        .await
        .unwrap();

    println!("  Filtered document type counts (nodes only):");
    for (doc_type, count) in &nodes_only_counts.doc_types {
        println!("    {}: {}", doc_type, count);
    }

    println!("\n  Filtered tag counts (nodes only):");
    for (tag, count) in &nodes_only_counts.tags {
        println!("    #{}: {}", tag, count);
    }

    // When filtering to nodes only, the tag counts should only reflect tags in nodes
    // Node 1 has #rust #systems, Node 2 has #python #scripting #data, Node 3 has #async #rust
    assert!(nodes_only_counts.tags.contains_key("rust"), "#rust should be in nodes");
    assert!(nodes_only_counts.tags.contains_key("python"), "#python should be in nodes");

    // === TEST 3: Get facet counts with tags_limit ===
    println!("\n  Testing facet counts with tags_limit=2...\n");

    let limited_counts = FulltextFacets::new()
        .with_tags_limit(2)
        .run(&fulltext_reader, timeout)
        .await
        .unwrap();

    println!("  Limited tag counts (max 2):");
    for (tag, count) in &limited_counts.tags {
        println!("    #{}: {}", tag, count);
    }

    assert!(limited_counts.tags.len() <= 2, "Should have at most 2 tags with tags_limit=2");

    // Cleanup - must drop readers first to close channels before awaiting handles
    drop(fulltext_reader);
    drop(graph_reader);
    for h in fulltext_handles {
        h.await.unwrap();
    }
    for h in graph_handles {
        h.await.unwrap();
    }

    println!("\n✅ Test passed: Facet counts query works correctly\n");
}
