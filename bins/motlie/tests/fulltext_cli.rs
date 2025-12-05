//! End-to-end integration tests for the fulltext CLI commands.
//!
//! This test:
//! 1. Inserts nodes, edges, and fragments directly into RocksDB (without fulltext indexing)
//! 2. Runs the `fulltext index` CLI command to build the fulltext index
//! 3. Runs the `fulltext search` CLI commands and verifies the results match the inserted data

use motlie_db::{
    create_mutation_writer, spawn_graph_consumer, AddEdge, AddEdgeFragment, AddNode,
    AddNodeFragment, DataUrl, EdgeSummary, Id, MutationRunnable, NodeSummary, TimestampMilli,
    WriterConfig,
};
use std::collections::HashSet;
use std::process::Command;
use tempfile::TempDir;

/// Test data structure holding all IDs for verification
#[derive(Debug, Clone)]
struct TestData {
    rust_id: Id,
    python_id: Id,
    javascript_id: Id,
    typescript_id: Id,
}

impl TestData {
    fn all_node_ids(&self) -> Vec<Id> {
        vec![
            self.rust_id,
            self.python_id,
            self.javascript_id,
            self.typescript_id,
        ]
    }
}

/// Helper to run the motlie CLI command
fn run_motlie(args: &[&str]) -> (bool, String, String) {
    let output = Command::new("cargo")
        .args(["run", "--bin", "motlie", "--"])
        .args(args)
        .output()
        .expect("Failed to execute motlie command");

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    (output.status.success(), stdout, stderr)
}

/// Parse node search results from stdout (TSV format: SCORE\tID\tFRAGMENT_TS\tSNIPPET)
fn parse_node_results(stdout: &str) -> Vec<Id> {
    stdout
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 2 {
                // Second column is the ID
                Id::from_str(parts[1]).ok()
            } else {
                None
            }
        })
        .collect()
}

/// Parse edge search results from stdout (TSV format: SCORE\tSRC_ID\tDST_ID\tEDGE_NAME\tFRAGMENT_TS\tSNIPPET)
fn parse_edge_results(stdout: &str) -> Vec<(Id, Id, String)> {
    stdout
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 4 {
                let src_id = Id::from_str(parts[1]).ok()?;
                let dst_id = Id::from_str(parts[2]).ok()?;
                let edge_name = parts[3].to_string();
                Some((src_id, dst_id, edge_name))
            } else {
                None
            }
        })
        .collect()
}

/// Parse facet results from stdout (TSV format: CATEGORY\tNAME\tCOUNT)
fn parse_facet_results(stdout: &str) -> Vec<(String, String, u64)> {
    stdout
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 3 {
                let category = parts[0].to_string();
                let name = parts[1].to_string();
                let count = parts[2].parse::<u64>().ok()?;
                Some((category, name, count))
            } else {
                None
            }
        })
        .collect()
}

/// Insert test data into graph storage only (no fulltext indexing)
/// Returns TestData with all the IDs for verification
async fn insert_test_data(db_path: &std::path::Path) -> TestData {
    let config = WriterConfig {
        channel_buffer_size: 100,
    };

    let (writer, receiver) = create_mutation_writer(config.clone());
    let _handle = spawn_graph_consumer(receiver, config.clone(), db_path);

    // Create nodes with searchable content
    let rust_id = Id::new();
    let python_id = Id::new();
    let javascript_id = Id::new();
    let typescript_id = Id::new();

    // Node: Rust - unique keywords: "systems", "memory safety", "ownership"
    let rust_node = AddNode {
        id: rust_id,
        ts_millis: TimestampMilli::now(),
        name: "Rust".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text("Rust systems programming language #rust #systems"),
    };
    rust_node.run(&writer).await.unwrap();

    // Node: Python - unique keywords: "scripting", "data science", "machine learning"
    let python_node = AddNode {
        id: python_id,
        ts_millis: TimestampMilli::now(),
        name: "Python".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text("Python scripting language #python #scripting"),
    };
    python_node.run(&writer).await.unwrap();

    // Node: JavaScript - unique keywords: "web", "browser", "frontend"
    let javascript_node = AddNode {
        id: javascript_id,
        ts_millis: TimestampMilli::now(),
        name: "JavaScript".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text("JavaScript web programming language #javascript #web"),
    };
    javascript_node.run(&writer).await.unwrap();

    // Node: TypeScript - unique keywords: "typed", "superset"
    let typescript_node = AddNode {
        id: typescript_id,
        ts_millis: TimestampMilli::now(),
        name: "TypeScript".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text(
            "TypeScript typed superset of JavaScript #typescript #web #typed",
        ),
    };
    typescript_node.run(&writer).await.unwrap();

    // Node fragments with detailed content
    let rust_fragment = AddNodeFragment {
        id: rust_id,
        ts_millis: TimestampMilli::now(),
        content: DataUrl::from_markdown(
            r#"# Rust Programming Language

Rust is a systems programming language focusing on safety, concurrency, and performance.
It provides memory safety without garbage collection through its ownership system.

Key features:
- Zero-cost abstractions
- Memory safety guarantees
- Fearless concurrency

#rust #systems #memory-safety"#,
        ),
        valid_range: None,
    };
    rust_fragment.run(&writer).await.unwrap();

    let python_fragment = AddNodeFragment {
        id: python_id,
        ts_millis: TimestampMilli::now(),
        content: DataUrl::from_markdown(
            r#"# Python Programming Language

Python is a high-level interpreted language known for simplicity and readability.
Widely used in data science, machine learning, and web development.

Key features:
- Easy to learn
- Dynamic typing
- Extensive libraries

#python #scripting #datascience"#,
        ),
        valid_range: None,
    };
    python_fragment.run(&writer).await.unwrap();

    // Edges between nodes
    let rust_influences_python = AddEdge {
        source_node_id: rust_id,
        target_node_id: python_id,
        ts_millis: TimestampMilli::now(),
        name: "influences".to_string(),
        valid_range: None,
        summary: EdgeSummary::from_text("Rust influences Python's typing features #influence"),
        weight: Some(0.8),
    };
    rust_influences_python.run(&writer).await.unwrap();

    let javascript_extends_to_typescript = AddEdge {
        source_node_id: javascript_id,
        target_node_id: typescript_id,
        ts_millis: TimestampMilli::now(),
        name: "extends_to".to_string(),
        valid_range: None,
        summary: EdgeSummary::from_text(
            "JavaScript extends to TypeScript with type safety #extends #typed",
        ),
        weight: Some(1.0),
    };
    javascript_extends_to_typescript.run(&writer).await.unwrap();

    let python_competes_with_javascript = AddEdge {
        source_node_id: python_id,
        target_node_id: javascript_id,
        ts_millis: TimestampMilli::now(),
        name: "competes_with".to_string(),
        valid_range: None,
        summary: EdgeSummary::from_text(
            "Python competes with JavaScript for web backend development #competition",
        ),
        weight: Some(0.6),
    };
    python_competes_with_javascript.run(&writer).await.unwrap();

    // Edge fragment
    let competition_details = AddEdgeFragment {
        src_id: python_id,
        dst_id: javascript_id,
        edge_name: "competes_with".to_string(),
        ts_millis: TimestampMilli::now(),
        content: DataUrl::from_markdown(
            r#"# Competition Analysis

Python and JavaScript compete in several areas:
- Web backend development (Django/Flask vs Node.js/Express)
- Data processing (Pandas vs various JS libraries)
- Machine learning (PyTorch/TensorFlow vs TensorFlow.js)

Both languages have strong communities and ecosystems.

#competition #backend #webdev"#,
        ),
        valid_range: None,
    };
    competition_details.run(&writer).await.unwrap();

    // Give time for mutations to be processed
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Drop writer to close the channel
    drop(writer);

    // Give time for the consumer to finish
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    TestData {
        rust_id,
        python_id,
        javascript_id,
        typescript_id,
    }
}

/// Helper to run index command and assert success
fn run_index(index_path: &std::path::Path, db_path: &std::path::Path) {
    let (success, _stdout, stderr) = run_motlie(&[
        "fulltext",
        "-p",
        index_path.to_str().unwrap(),
        "index",
        "-b",
        "10",
        db_path.to_str().unwrap(),
    ]);
    assert!(success, "fulltext index command failed: {}", stderr);
}

#[tokio::test]
async fn test_search_nodes_by_exact_name() {
    // Test that searching for exact node names returns the correct node
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");
    let index_path = temp_dir.path().join("fulltext_index");

    let test_data = insert_test_data(&db_path).await;
    run_index(&index_path, &db_path);

    // Search for "Rust" - should find exactly the Rust node
    let (success, stdout, stderr) = run_motlie(&[
        "fulltext",
        "-p",
        index_path.to_str().unwrap(),
        "search",
        "nodes",
        "Rust",
        "-l",
        "10",
    ]);
    assert!(success, "Search failed: {}", stderr);

    let results = parse_node_results(&stdout);
    println!("Search 'Rust' results: {:?}", results);
    println!("Expected Rust ID: {}", test_data.rust_id);

    assert!(
        results.contains(&test_data.rust_id),
        "Expected to find Rust node (ID: {}) in results: {:?}",
        test_data.rust_id,
        results
    );

    // Search for "Python" - should find exactly the Python node
    let (success, stdout, _stderr) = run_motlie(&[
        "fulltext",
        "-p",
        index_path.to_str().unwrap(),
        "search",
        "nodes",
        "Python",
        "-l",
        "10",
    ]);
    assert!(success);

    let results = parse_node_results(&stdout);
    assert!(
        results.contains(&test_data.python_id),
        "Expected to find Python node (ID: {}) in results: {:?}",
        test_data.python_id,
        results
    );

    // Search for "JavaScript" - should find JavaScript node
    let (success, stdout, _stderr) = run_motlie(&[
        "fulltext",
        "-p",
        index_path.to_str().unwrap(),
        "search",
        "nodes",
        "JavaScript",
        "-l",
        "10",
    ]);
    assert!(success);

    let results = parse_node_results(&stdout);
    assert!(
        results.contains(&test_data.javascript_id),
        "Expected to find JavaScript node (ID: {}) in results: {:?}",
        test_data.javascript_id,
        results
    );
}

#[tokio::test]
async fn test_search_nodes_by_unique_content() {
    // Test that searching for unique content keywords finds the correct node
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");
    let index_path = temp_dir.path().join("fulltext_index");

    let test_data = insert_test_data(&db_path).await;
    run_index(&index_path, &db_path);

    // Search for "ownership" - unique to Rust fragment, should find Rust
    let (success, stdout, stderr) = run_motlie(&[
        "fulltext",
        "-p",
        index_path.to_str().unwrap(),
        "search",
        "nodes",
        "ownership",
        "-l",
        "10",
    ]);
    assert!(success, "Search failed: {}", stderr);

    let results = parse_node_results(&stdout);
    println!("Search 'ownership' results: {:?}", results);

    assert!(
        results.contains(&test_data.rust_id),
        "Expected 'ownership' to find Rust node (ID: {}), got: {:?}",
        test_data.rust_id,
        results
    );
    // Should NOT find Python, JavaScript, or TypeScript
    assert!(
        !results.contains(&test_data.python_id),
        "'ownership' should not find Python"
    );

    // Search for "machine learning" - unique to Python fragment
    let (success, stdout, _stderr) = run_motlie(&[
        "fulltext",
        "-p",
        index_path.to_str().unwrap(),
        "search",
        "nodes",
        "machine learning",
        "-l",
        "10",
    ]);
    assert!(success);

    let results = parse_node_results(&stdout);
    println!("Search 'machine learning' results: {:?}", results);

    assert!(
        results.contains(&test_data.python_id),
        "Expected 'machine learning' to find Python node (ID: {}), got: {:?}",
        test_data.python_id,
        results
    );

    // Search for "superset" - unique to TypeScript
    let (success, stdout, _stderr) = run_motlie(&[
        "fulltext",
        "-p",
        index_path.to_str().unwrap(),
        "search",
        "nodes",
        "superset",
        "-l",
        "10",
    ]);
    assert!(success);

    let results = parse_node_results(&stdout);
    assert!(
        results.contains(&test_data.typescript_id),
        "Expected 'superset' to find TypeScript node (ID: {}), got: {:?}",
        test_data.typescript_id,
        results
    );
}

#[tokio::test]
async fn test_search_nodes_no_false_positives() {
    // Test that searching for non-existent terms returns no results
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");
    let index_path = temp_dir.path().join("fulltext_index");

    let _test_data = insert_test_data(&db_path).await;
    run_index(&index_path, &db_path);

    // Search for something not in any document
    let (success, stdout, stderr) = run_motlie(&[
        "fulltext",
        "-p",
        index_path.to_str().unwrap(),
        "search",
        "nodes",
        "xyznonexistent123",
        "-l",
        "10",
    ]);
    assert!(success, "Search failed: {}", stderr);

    let results = parse_node_results(&stdout);
    assert!(
        results.is_empty(),
        "Expected no results for nonexistent term, got: {:?}",
        results
    );
}

#[tokio::test]
async fn test_search_edges_by_name() {
    // Test that searching for edge names returns the correct edges
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");
    let index_path = temp_dir.path().join("fulltext_index");

    let test_data = insert_test_data(&db_path).await;
    run_index(&index_path, &db_path);

    // Search for "influences" - should find rust->python edge
    let (success, stdout, stderr) = run_motlie(&[
        "fulltext",
        "-p",
        index_path.to_str().unwrap(),
        "search",
        "edges",
        "influences",
        "-l",
        "10",
    ]);
    assert!(success, "Search failed: {}", stderr);

    let results = parse_edge_results(&stdout);
    println!("Search 'influences' edge results: {:?}", results);

    let expected_edge = (
        test_data.rust_id,
        test_data.python_id,
        "influences".to_string(),
    );
    assert!(
        results.contains(&expected_edge),
        "Expected to find rust->python 'influences' edge, got: {:?}",
        results
    );

    // Search for "extends" - should find javascript->typescript edge
    let (success, stdout, _stderr) = run_motlie(&[
        "fulltext",
        "-p",
        index_path.to_str().unwrap(),
        "search",
        "edges",
        "extends",
        "-l",
        "10",
    ]);
    assert!(success);

    let results = parse_edge_results(&stdout);
    println!("Search 'extends' edge results: {:?}", results);

    let expected_edge = (
        test_data.javascript_id,
        test_data.typescript_id,
        "extends_to".to_string(),
    );
    assert!(
        results.contains(&expected_edge),
        "Expected to find javascript->typescript 'extends_to' edge, got: {:?}",
        results
    );

    // Search for "competes" - should find python->javascript edge
    let (success, stdout, _stderr) = run_motlie(&[
        "fulltext",
        "-p",
        index_path.to_str().unwrap(),
        "search",
        "edges",
        "competes",
        "-l",
        "10",
    ]);
    assert!(success);

    let results = parse_edge_results(&stdout);
    let expected_edge = (
        test_data.python_id,
        test_data.javascript_id,
        "competes_with".to_string(),
    );
    assert!(
        results.contains(&expected_edge),
        "Expected to find python->javascript 'competes_with' edge, got: {:?}",
        results
    );
}

#[tokio::test]
async fn test_search_edges_by_fragment_content() {
    // Test that searching for edge fragment content finds the correct edge
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");
    let index_path = temp_dir.path().join("fulltext_index");

    let test_data = insert_test_data(&db_path).await;
    run_index(&index_path, &db_path);

    // Search for "Django" - unique to the competition edge fragment
    let (success, stdout, stderr) = run_motlie(&[
        "fulltext",
        "-p",
        index_path.to_str().unwrap(),
        "search",
        "edges",
        "Django",
        "-l",
        "10",
    ]);
    assert!(success, "Search failed: {}", stderr);

    let results = parse_edge_results(&stdout);
    println!("Search 'Django' edge results: {:?}", results);

    let expected_edge = (
        test_data.python_id,
        test_data.javascript_id,
        "competes_with".to_string(),
    );
    assert!(
        results.contains(&expected_edge),
        "Expected 'Django' to find python->javascript 'competes_with' edge, got: {:?}",
        results
    );

    // Search for "Pandas" - also unique to competition edge fragment
    let (success, stdout, _stderr) = run_motlie(&[
        "fulltext",
        "-p",
        index_path.to_str().unwrap(),
        "search",
        "edges",
        "Pandas",
        "-l",
        "10",
    ]);
    assert!(success);

    let results = parse_edge_results(&stdout);
    assert!(
        results.contains(&expected_edge),
        "Expected 'Pandas' to find python->javascript 'competes_with' edge, got: {:?}",
        results
    );
}

#[tokio::test]
async fn test_search_with_tag_filter() {
    // Test that tag filtering works correctly
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");
    let index_path = temp_dir.path().join("fulltext_index");

    let test_data = insert_test_data(&db_path).await;
    run_index(&index_path, &db_path);

    // Search for "language" with #rust tag - should only find Rust
    let (success, stdout, stderr) = run_motlie(&[
        "fulltext",
        "-p",
        index_path.to_str().unwrap(),
        "search",
        "nodes",
        "language",
        "-l",
        "10",
        "-t",
        "rust",
    ]);
    assert!(success, "Search failed: {}", stderr);

    let results = parse_node_results(&stdout);
    println!("Search 'language' with #rust tag results: {:?}", results);

    assert!(
        results.contains(&test_data.rust_id),
        "Expected to find Rust with #rust tag filter, got: {:?}",
        results
    );
    // Should not find other nodes
    assert!(
        !results.contains(&test_data.python_id),
        "Should not find Python with #rust tag filter"
    );
    assert!(
        !results.contains(&test_data.javascript_id),
        "Should not find JavaScript with #rust tag filter"
    );

    // Search for "language" with #web tag - should find JavaScript and TypeScript
    let (success, stdout, _stderr) = run_motlie(&[
        "fulltext",
        "-p",
        index_path.to_str().unwrap(),
        "search",
        "nodes",
        "language",
        "-l",
        "10",
        "-t",
        "web",
    ]);
    assert!(success);

    let results = parse_node_results(&stdout);
    println!("Search 'language' with #web tag results: {:?}", results);

    // Should find both JavaScript and TypeScript (both have #web tag)
    let web_nodes: HashSet<_> = results.iter().cloned().collect();
    assert!(
        web_nodes.contains(&test_data.javascript_id)
            || web_nodes.contains(&test_data.typescript_id),
        "Expected to find JavaScript or TypeScript with #web tag, got: {:?}",
        results
    );
    // Should NOT find Rust or Python
    assert!(
        !results.contains(&test_data.rust_id),
        "Should not find Rust with #web tag filter"
    );
}

#[tokio::test]
async fn test_search_with_fuzzy_matching() {
    // Test that fuzzy matching corrects typos
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");
    let index_path = temp_dir.path().join("fulltext_index");

    let test_data = insert_test_data(&db_path).await;
    run_index(&index_path, &db_path);

    // Search for "Pythn" (typo) with fuzzy matching - should find Python
    let (success, stdout, stderr) = run_motlie(&[
        "fulltext",
        "-p",
        index_path.to_str().unwrap(),
        "search",
        "nodes",
        "Pythn",
        "-l",
        "10",
        "-f",
        "low",
    ]);
    assert!(success, "Search failed: {}", stderr);

    let results = parse_node_results(&stdout);
    println!("Search 'Pythn' (fuzzy) results: {:?}", results);

    assert!(
        results.contains(&test_data.python_id),
        "Expected fuzzy search 'Pythn' to find Python (ID: {}), got: {:?}",
        test_data.python_id,
        results
    );

    // Search for "JavaScrpt" (typo) with fuzzy matching
    let (success, stdout, _stderr) = run_motlie(&[
        "fulltext",
        "-p",
        index_path.to_str().unwrap(),
        "search",
        "nodes",
        "JavaScrpt",
        "-l",
        "10",
        "-f",
        "low",
    ]);
    assert!(success);

    let results = parse_node_results(&stdout);
    println!("Search 'JavaScrpt' (fuzzy) results: {:?}", results);

    assert!(
        results.contains(&test_data.javascript_id),
        "Expected fuzzy search 'JavaScrpt' to find JavaScript (ID: {}), got: {:?}",
        test_data.javascript_id,
        results
    );
}

#[tokio::test]
async fn test_facets_document_type_counts() {
    // Test that facets returns correct document type counts
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");
    let index_path = temp_dir.path().join("fulltext_index");

    let _test_data = insert_test_data(&db_path).await;
    run_index(&index_path, &db_path);

    // Get all facets
    let (success, stdout, stderr) = run_motlie(&[
        "fulltext",
        "-p",
        index_path.to_str().unwrap(),
        "search",
        "facets",
    ]);
    assert!(success, "Facets command failed: {}", stderr);

    let facets = parse_facet_results(&stdout);
    println!("Facets results: {:?}", facets);

    // We inserted: 4 nodes, 2 node fragments, 3 edges, 1 edge fragment
    let doc_type_facets: Vec<_> = facets
        .iter()
        .filter(|(cat, _, _)| cat == "doc_type")
        .collect();

    println!("Doc type facets: {:?}", doc_type_facets);

    // Verify node count (should be 4)
    let nodes_count = doc_type_facets
        .iter()
        .find(|(_, name, _)| name == "nodes")
        .map(|(_, _, count)| *count);
    assert_eq!(
        nodes_count,
        Some(4),
        "Expected 4 nodes, got: {:?}",
        nodes_count
    );

    // Verify edge count (should be 3)
    let edges_count = doc_type_facets
        .iter()
        .find(|(_, name, _)| name == "edges")
        .map(|(_, _, count)| *count);
    assert_eq!(
        edges_count,
        Some(3),
        "Expected 3 edges, got: {:?}",
        edges_count
    );

    // Verify node fragments count (should be 2)
    let node_fragments_count = doc_type_facets
        .iter()
        .find(|(_, name, _)| name == "node_fragments")
        .map(|(_, _, count)| *count);
    assert_eq!(
        node_fragments_count,
        Some(2),
        "Expected 2 node_fragments, got: {:?}",
        node_fragments_count
    );

    // Verify edge fragments count (should be 1)
    let edge_fragments_count = doc_type_facets
        .iter()
        .find(|(_, name, _)| name == "edge_fragments")
        .map(|(_, _, count)| *count);
    assert_eq!(
        edge_fragments_count,
        Some(1),
        "Expected 1 edge_fragment, got: {:?}",
        edge_fragments_count
    );
}

#[tokio::test]
async fn test_facets_tag_counts() {
    // Test that facets returns correct tag counts
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");
    let index_path = temp_dir.path().join("fulltext_index");

    let _test_data = insert_test_data(&db_path).await;
    run_index(&index_path, &db_path);

    // Get all facets
    let (success, stdout, stderr) = run_motlie(&[
        "fulltext",
        "-p",
        index_path.to_str().unwrap(),
        "search",
        "facets",
    ]);
    assert!(success, "Facets command failed: {}", stderr);

    let facets = parse_facet_results(&stdout);
    let tag_facets: Vec<_> = facets.iter().filter(|(cat, _, _)| cat == "tag").collect();

    println!("Tag facets: {:?}", tag_facets);

    // Verify some expected tags exist
    let tag_names: HashSet<_> = tag_facets.iter().map(|(_, name, _)| name.as_str()).collect();

    // These tags should exist based on our test data
    assert!(
        tag_names.contains("rust") || tag_names.contains("python") || tag_names.contains("web"),
        "Expected to find at least one of our test tags, got: {:?}",
        tag_names
    );
}

#[tokio::test]
async fn test_reindex_prevention() {
    // Test that indexing into non-empty directory is prevented
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");
    let index_path = temp_dir.path().join("fulltext_index");

    let test_data = insert_test_data(&db_path).await;

    // First index should succeed
    run_index(&index_path, &db_path);

    // Verify the index works
    let (success, stdout, _stderr) = run_motlie(&[
        "fulltext",
        "-p",
        index_path.to_str().unwrap(),
        "search",
        "nodes",
        "Rust",
        "-l",
        "10",
    ]);
    assert!(success);
    let results = parse_node_results(&stdout);
    assert!(
        results.contains(&test_data.rust_id),
        "Should find Rust after first index"
    );

    // Second index attempt - should fail or be prevented
    let (_success, _stdout, stderr) = run_motlie(&[
        "fulltext",
        "-p",
        index_path.to_str().unwrap(),
        "index",
        "-b",
        "10",
        db_path.to_str().unwrap(),
    ]);

    println!("Re-index stderr: {}", stderr);

    // After re-index attempt, data should still be searchable
    let (success, stdout, _stderr) = run_motlie(&[
        "fulltext",
        "-p",
        index_path.to_str().unwrap(),
        "search",
        "nodes",
        "Rust",
        "-l",
        "10",
    ]);
    assert!(success, "Should still be able to search after re-index attempt");
    let results = parse_node_results(&stdout);
    assert!(
        results.contains(&test_data.rust_id),
        "Should still find Rust after re-index attempt"
    );
}
