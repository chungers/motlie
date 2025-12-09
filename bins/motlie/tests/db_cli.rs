//! End-to-end integration tests for the `motlie db` CLI commands.
//!
//! This test:
//! 1. Inserts nodes, edges, and fragments directly into RocksDB
//! 2. Runs the `db list` and `db scan` CLI commands
//! 3. Verifies the output matches the inserted data

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
#[allow(dead_code)]
struct TestData {
    alice_id: Id,
    bob_id: Id,
    charlie_id: Id,
    alice_fragment_ts: TimestampMilli,
    bob_fragment_ts: TimestampMilli,
    alice_bob_edge_ts: TimestampMilli,
    bob_charlie_edge_ts: TimestampMilli,
    edge_fragment_ts: TimestampMilli,
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

    // Filter out tracing log lines from stdout (they contain ANSI escape codes and "INFO")
    let filtered_stdout = stdout
        .lines()
        .filter(|line| !line.contains("INFO") && !line.contains("\x1b["))
        .collect::<Vec<_>>()
        .join("\n");

    (output.status.success(), filtered_stdout, stderr)
}

/// Insert test data into graph storage
/// Returns TestData with all the IDs and timestamps for verification
async fn insert_test_data(db_path: &std::path::Path) -> TestData {
    let config = WriterConfig {
        channel_buffer_size: 100,
    };

    let (writer, receiver) = create_mutation_writer(config.clone());
    let _handle = spawn_graph_consumer(receiver, config.clone(), db_path);

    // Create nodes
    let alice_id = Id::new();
    let bob_id = Id::new();
    let charlie_id = Id::new();

    // Node: Alice
    let alice_node = AddNode {
        id: alice_id,
        ts_millis: TimestampMilli::now(),
        name: "Alice".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text("Alice is a software engineer"),
    };
    alice_node.run(&writer).await.unwrap();

    // Node: Bob
    let bob_node = AddNode {
        id: bob_id,
        ts_millis: TimestampMilli::now(),
        name: "Bob".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text("Bob is a data scientist"),
    };
    bob_node.run(&writer).await.unwrap();

    // Node: Charlie
    let charlie_node = AddNode {
        id: charlie_id,
        ts_millis: TimestampMilli::now(),
        name: "Charlie".to_string(),
        valid_range: None,
        summary: NodeSummary::from_text("Charlie is a product manager"),
    };
    charlie_node.run(&writer).await.unwrap();

    // Node fragments
    let alice_fragment_ts = TimestampMilli::now();
    let alice_fragment = AddNodeFragment {
        id: alice_id,
        ts_millis: alice_fragment_ts,
        content: DataUrl::from_markdown("# Alice\n\nAlice works on backend systems."),
        valid_range: None,
    };
    alice_fragment.run(&writer).await.unwrap();

    let bob_fragment_ts = TimestampMilli::now();
    let bob_fragment = AddNodeFragment {
        id: bob_id,
        ts_millis: bob_fragment_ts,
        content: DataUrl::from_markdown("# Bob\n\nBob specializes in machine learning."),
        valid_range: None,
    };
    bob_fragment.run(&writer).await.unwrap();

    // Edges
    let alice_bob_edge_ts = TimestampMilli::now();
    let alice_bob_edge = AddEdge {
        source_node_id: alice_id,
        target_node_id: bob_id,
        ts_millis: alice_bob_edge_ts,
        name: "works_with".to_string(),
        valid_range: None,
        summary: EdgeSummary::from_text("Alice works with Bob"),
        weight: Some(0.9),
    };
    alice_bob_edge.run(&writer).await.unwrap();

    let bob_charlie_edge_ts = TimestampMilli::now();
    let bob_charlie_edge = AddEdge {
        source_node_id: bob_id,
        target_node_id: charlie_id,
        ts_millis: bob_charlie_edge_ts,
        name: "reports_to".to_string(),
        valid_range: None,
        summary: EdgeSummary::from_text("Bob reports to Charlie"),
        weight: Some(1.0),
    };
    bob_charlie_edge.run(&writer).await.unwrap();

    // Edge fragment
    let edge_fragment_ts = TimestampMilli::now();
    let edge_fragment = AddEdgeFragment {
        src_id: alice_id,
        dst_id: bob_id,
        edge_name: "works_with".to_string(),
        ts_millis: edge_fragment_ts,
        content: DataUrl::from_markdown("# Collaboration\n\nAlice and Bob collaborate on ML projects."),
        valid_range: None,
    };
    edge_fragment.run(&writer).await.unwrap();

    // Give time for mutations to be processed
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Drop writer to close the channel
    drop(writer);

    // Give time for the consumer to finish
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    TestData {
        alice_id,
        bob_id,
        charlie_id,
        alice_fragment_ts,
        bob_fragment_ts,
        alice_bob_edge_ts,
        bob_charlie_edge_ts,
        edge_fragment_ts,
    }
}

// ============================================================================
// List Command Tests
// ============================================================================

#[tokio::test]
async fn test_list_column_families() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");

    let _test_data = insert_test_data(&db_path).await;

    let (success, stdout, stderr) = run_motlie(&["db", "-p", db_path.to_str().unwrap(), "list"]);

    assert!(success, "List command failed: {}", stderr);

    // Verify all column families are listed
    assert!(
        stdout.contains("nodes"),
        "Expected 'nodes' in output: {}",
        stdout
    );
    assert!(
        stdout.contains("node-fragments"),
        "Expected 'node-fragments' in output: {}",
        stdout
    );
    assert!(
        stdout.contains("edge-fragments"),
        "Expected 'edge-fragments' in output: {}",
        stdout
    );
    assert!(
        stdout.contains("outgoing-edges"),
        "Expected 'outgoing-edges' in output: {}",
        stdout
    );
    assert!(
        stdout.contains("incoming-edges"),
        "Expected 'incoming-edges' in output: {}",
        stdout
    );
    assert!(
        stdout.contains("node-names"),
        "Expected 'node-names' in output: {}",
        stdout
    );
    assert!(
        stdout.contains("edge-names"),
        "Expected 'edge-names' in output: {}",
        stdout
    );
}

// ============================================================================
// Scan Nodes Tests
// ============================================================================

#[tokio::test]
async fn test_scan_nodes_tsv() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");

    let test_data = insert_test_data(&db_path).await;

    let (success, stdout, stderr) = run_motlie(&[
        "db",
        "-p",
        db_path.to_str().unwrap(),
        "scan",
        "nodes",
        "--limit",
        "10",
        "--format",
        "tsv",
    ]);

    assert!(success, "Scan nodes failed: {}", stderr);

    // Parse TSV output - format: SINCE\tUNTIL\tID\tNAME
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 3, "Expected 3 nodes, got: {:?}", lines);

    // Extract IDs from output
    let output_ids: HashSet<String> = lines
        .iter()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 3 {
                Some(parts[2].to_string())
            } else {
                None
            }
        })
        .collect();

    // Verify all node IDs are present
    assert!(
        output_ids.contains(&test_data.alice_id.to_string()),
        "Expected Alice ID {} in output",
        test_data.alice_id
    );
    assert!(
        output_ids.contains(&test_data.bob_id.to_string()),
        "Expected Bob ID {} in output",
        test_data.bob_id
    );
    assert!(
        output_ids.contains(&test_data.charlie_id.to_string()),
        "Expected Charlie ID {} in output",
        test_data.charlie_id
    );

    // Verify names are present
    assert!(
        stdout.contains("Alice"),
        "Expected 'Alice' in output: {}",
        stdout
    );
    assert!(
        stdout.contains("Bob"),
        "Expected 'Bob' in output: {}",
        stdout
    );
    assert!(
        stdout.contains("Charlie"),
        "Expected 'Charlie' in output: {}",
        stdout
    );
}

#[tokio::test]
async fn test_scan_nodes_table_format() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");

    let _test_data = insert_test_data(&db_path).await;

    let (success, stdout, stderr) = run_motlie(&[
        "db",
        "-p",
        db_path.to_str().unwrap(),
        "scan",
        "nodes",
        "-f",
        "table",
    ]);

    assert!(success, "Scan nodes table format failed: {}", stderr);

    // Table format should have headers
    assert!(
        stdout.contains("SINCE"),
        "Expected 'SINCE' header: {}",
        stdout
    );
    assert!(
        stdout.contains("UNTIL"),
        "Expected 'UNTIL' header: {}",
        stdout
    );
    assert!(stdout.contains("ID"), "Expected 'ID' header: {}", stdout);
    assert!(
        stdout.contains("NAME"),
        "Expected 'NAME' header: {}",
        stdout
    );

    // Should have separator line with dashes
    assert!(
        stdout.contains("---"),
        "Expected separator line: {}",
        stdout
    );
}

#[tokio::test]
async fn test_scan_nodes_with_limit() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");

    let _test_data = insert_test_data(&db_path).await;

    // Scan with limit of 1 using TSV format to avoid header lines
    let (success, stdout, stderr) = run_motlie(&[
        "db",
        "-p",
        db_path.to_str().unwrap(),
        "scan",
        "nodes",
        "--limit",
        "1",
        "--format",
        "tsv",
    ]);

    assert!(success, "Scan nodes with limit failed: {}", stderr);

    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 1, "Expected 1 node with limit=1, got: {:?}", lines);
}

#[tokio::test]
async fn test_scan_nodes_reverse() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");

    let _test_data = insert_test_data(&db_path).await;

    // Get forward order
    let (success, stdout_forward, _) = run_motlie(&[
        "db",
        "-p",
        db_path.to_str().unwrap(),
        "scan",
        "nodes",
        "--limit",
        "10",
    ]);
    assert!(success);

    // Get reverse order
    let (success, stdout_reverse, stderr) = run_motlie(&[
        "db",
        "-p",
        db_path.to_str().unwrap(),
        "scan",
        "nodes",
        "--limit",
        "10",
        "--reverse",
    ]);
    assert!(success, "Scan nodes reverse failed: {}", stderr);

    // Extract IDs in order
    let forward_ids: Vec<String> = stdout_forward
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 3 {
                Some(parts[2].to_string())
            } else {
                None
            }
        })
        .collect();

    let reverse_ids: Vec<String> = stdout_reverse
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 3 {
                Some(parts[2].to_string())
            } else {
                None
            }
        })
        .collect();

    // Reverse should be opposite order
    let mut expected_reverse = forward_ids.clone();
    expected_reverse.reverse();
    assert_eq!(
        reverse_ids, expected_reverse,
        "Reverse order should be opposite of forward order"
    );
}

#[tokio::test]
async fn test_scan_nodes_pagination() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");

    let test_data = insert_test_data(&db_path).await;

    // Get first page using TSV format for parsing
    let (success, stdout_page1, _) = run_motlie(&[
        "db",
        "-p",
        db_path.to_str().unwrap(),
        "scan",
        "nodes",
        "--limit",
        "1",
        "--format",
        "tsv",
    ]);
    assert!(success);

    let first_id = stdout_page1
        .lines()
        .next()
        .unwrap()
        .split('\t')
        .nth(2)
        .unwrap();

    // Get second page using cursor
    let (success, stdout_page2, stderr) = run_motlie(&[
        "db",
        "-p",
        db_path.to_str().unwrap(),
        "scan",
        "nodes",
        "--limit",
        "1",
        "--last",
        first_id,
        "--format",
        "tsv",
    ]);
    assert!(success, "Pagination failed: {}", stderr);

    let second_id = stdout_page2
        .lines()
        .next()
        .unwrap()
        .split('\t')
        .nth(2)
        .unwrap();

    // First and second page should have different IDs
    assert_ne!(first_id, second_id, "Pagination should return different nodes");

    // Both should be from our test data
    let all_ids: HashSet<String> = vec![
        test_data.alice_id.to_string(),
        test_data.bob_id.to_string(),
        test_data.charlie_id.to_string(),
    ]
    .into_iter()
    .collect();

    assert!(all_ids.contains(first_id), "First ID should be from test data");
    assert!(all_ids.contains(second_id), "Second ID should be from test data");
}

// ============================================================================
// Scan Node Fragments Tests
// ============================================================================

#[tokio::test]
async fn test_scan_node_fragments() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");

    let test_data = insert_test_data(&db_path).await;

    let (success, stdout, stderr) = run_motlie(&[
        "db",
        "-p",
        db_path.to_str().unwrap(),
        "scan",
        "node-fragments",
        "--limit",
        "10",
        "--format",
        "tsv",
    ]);

    assert!(success, "Scan node-fragments failed: {}", stderr);

    // Format: SINCE\tUNTIL\tNODE_ID\tTIMESTAMP\tMIME\tCONTENT
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 2, "Expected 2 node fragments, got: {:?}", lines);

    // Extract node IDs
    let output_node_ids: HashSet<String> = lines
        .iter()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 3 {
                Some(parts[2].to_string())
            } else {
                None
            }
        })
        .collect();

    // Verify Alice and Bob fragments are present
    assert!(
        output_node_ids.contains(&test_data.alice_id.to_string()),
        "Expected Alice's fragment"
    );
    assert!(
        output_node_ids.contains(&test_data.bob_id.to_string()),
        "Expected Bob's fragment"
    );

    // Verify MIME type is markdown
    assert!(
        stdout.contains("text/markdown"),
        "Expected 'text/markdown' MIME type: {}",
        stdout
    );

    // Verify content preview is present
    assert!(
        stdout.contains("Alice") || stdout.contains("Bob"),
        "Expected content preview: {}",
        stdout
    );
}

// ============================================================================
// Scan Edge Fragments Tests
// ============================================================================

#[tokio::test]
async fn test_scan_edge_fragments() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");

    let test_data = insert_test_data(&db_path).await;

    let (success, stdout, stderr) = run_motlie(&[
        "db",
        "-p",
        db_path.to_str().unwrap(),
        "scan",
        "edge-fragments",
        "--limit",
        "10",
        "--format",
        "tsv",
    ]);

    assert!(success, "Scan edge-fragments failed: {}", stderr);

    // Format: SINCE\tUNTIL\tSRC_ID\tDST_ID\tTIMESTAMP\tEDGE_NAME\tMIME\tCONTENT
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 1, "Expected 1 edge fragment, got: {:?}", lines);

    let parts: Vec<&str> = lines[0].split('\t').collect();
    assert!(parts.len() >= 6, "Expected at least 6 columns");

    // Verify source and destination IDs
    assert_eq!(
        parts[2],
        test_data.alice_id.to_string(),
        "Expected Alice as source"
    );
    assert_eq!(
        parts[3],
        test_data.bob_id.to_string(),
        "Expected Bob as destination"
    );

    // Verify edge name
    assert!(
        stdout.contains("works_with"),
        "Expected 'works_with' edge name: {}",
        stdout
    );
}

// ============================================================================
// Scan Outgoing Edges Tests
// ============================================================================

#[tokio::test]
async fn test_scan_outgoing_edges() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");

    let test_data = insert_test_data(&db_path).await;

    let (success, stdout, stderr) = run_motlie(&[
        "db",
        "-p",
        db_path.to_str().unwrap(),
        "scan",
        "outgoing-edges",
        "--limit",
        "10",
        "--format",
        "tsv",
    ]);

    assert!(success, "Scan outgoing-edges failed: {}", stderr);

    // Format: SINCE\tUNTIL\tSRC_ID\tDST_ID\tEDGE_NAME\tWEIGHT
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 2, "Expected 2 outgoing edges, got: {:?}", lines);

    // Verify edge names are present
    assert!(
        stdout.contains("works_with"),
        "Expected 'works_with' edge: {}",
        stdout
    );
    assert!(
        stdout.contains("reports_to"),
        "Expected 'reports_to' edge: {}",
        stdout
    );

    // Verify weights are present
    assert!(
        stdout.contains("0.9") || stdout.contains("0.9000"),
        "Expected weight 0.9: {}",
        stdout
    );
    assert!(
        stdout.contains("1.0") || stdout.contains("1.0000"),
        "Expected weight 1.0: {}",
        stdout
    );

    // Verify specific edges exist
    let alice_bob_found = lines.iter().any(|line| {
        line.contains(&test_data.alice_id.to_string())
            && line.contains(&test_data.bob_id.to_string())
            && line.contains("works_with")
    });
    assert!(alice_bob_found, "Expected Alice->Bob 'works_with' edge");

    let bob_charlie_found = lines.iter().any(|line| {
        line.contains(&test_data.bob_id.to_string())
            && line.contains(&test_data.charlie_id.to_string())
            && line.contains("reports_to")
    });
    assert!(bob_charlie_found, "Expected Bob->Charlie 'reports_to' edge");
}

// ============================================================================
// Scan Incoming Edges Tests
// ============================================================================

#[tokio::test]
async fn test_scan_incoming_edges() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");

    let test_data = insert_test_data(&db_path).await;

    let (success, stdout, stderr) = run_motlie(&[
        "db",
        "-p",
        db_path.to_str().unwrap(),
        "scan",
        "incoming-edges",
        "--limit",
        "10",
        "--format",
        "tsv",
    ]);

    assert!(success, "Scan incoming-edges failed: {}", stderr);

    // Format: SINCE\tUNTIL\tDST_ID\tSRC_ID\tEDGE_NAME
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 2, "Expected 2 incoming edges, got: {:?}", lines);

    // Verify Bob has incoming edge from Alice
    let bob_incoming = lines.iter().any(|line| {
        let parts: Vec<&str> = line.split('\t').collect();
        parts.len() >= 5
            && parts[2] == test_data.bob_id.to_string()
            && parts[3] == test_data.alice_id.to_string()
            && parts[4] == "works_with"
    });
    assert!(bob_incoming, "Expected incoming edge to Bob from Alice");

    // Verify Charlie has incoming edge from Bob
    let charlie_incoming = lines.iter().any(|line| {
        let parts: Vec<&str> = line.split('\t').collect();
        parts.len() >= 5
            && parts[2] == test_data.charlie_id.to_string()
            && parts[3] == test_data.bob_id.to_string()
            && parts[4] == "reports_to"
    });
    assert!(charlie_incoming, "Expected incoming edge to Charlie from Bob");
}

// ============================================================================
// Scan Node Names Tests
// ============================================================================

#[tokio::test]
async fn test_scan_node_names() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");

    let test_data = insert_test_data(&db_path).await;

    let (success, stdout, stderr) = run_motlie(&[
        "db",
        "-p",
        db_path.to_str().unwrap(),
        "scan",
        "node-names",
        "--limit",
        "10",
        "--format",
        "tsv",
    ]);

    assert!(success, "Scan node-names failed: {}", stderr);

    // Format: SINCE\tUNTIL\tNODE_ID\tNAME
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 3, "Expected 3 node names, got: {:?}", lines);

    // Verify all names are present
    assert!(stdout.contains("Alice"), "Expected 'Alice' in names");
    assert!(stdout.contains("Bob"), "Expected 'Bob' in names");
    assert!(stdout.contains("Charlie"), "Expected 'Charlie' in names");

    // Verify name->ID mappings
    let alice_entry = lines.iter().any(|line| {
        line.contains(&test_data.alice_id.to_string()) && line.contains("Alice")
    });
    assert!(alice_entry, "Expected Alice name->ID mapping");
}

// ============================================================================
// Scan Edge Names Tests
// ============================================================================

#[tokio::test]
async fn test_scan_edge_names() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");

    let test_data = insert_test_data(&db_path).await;

    let (success, stdout, stderr) = run_motlie(&[
        "db",
        "-p",
        db_path.to_str().unwrap(),
        "scan",
        "edge-names",
        "--limit",
        "10",
        "--format",
        "tsv",
    ]);

    assert!(success, "Scan edge-names failed: {}", stderr);

    // Format: SINCE\tUNTIL\tSRC_ID\tDST_ID\tNAME
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 2, "Expected 2 edge names, got: {:?}", lines);

    // Verify edge names are present
    assert!(
        stdout.contains("works_with"),
        "Expected 'works_with' in edge names"
    );
    assert!(
        stdout.contains("reports_to"),
        "Expected 'reports_to' in edge names"
    );

    // Verify works_with edge name entry
    let works_with_entry = lines.iter().any(|line| {
        line.contains(&test_data.alice_id.to_string())
            && line.contains(&test_data.bob_id.to_string())
            && line.contains("works_with")
    });
    assert!(works_with_entry, "Expected 'works_with' edge name entry");
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[tokio::test]
async fn test_invalid_database_path() {
    let (success, _stdout, _stderr) = run_motlie(&[
        "db",
        "-p",
        "/nonexistent/path/to/db",
        "scan",
        "nodes",
    ]);

    assert!(!success, "Should fail with invalid database path");
}

#[tokio::test]
async fn test_invalid_cursor_format() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");

    let _test_data = insert_test_data(&db_path).await;

    // Invalid cursor for node fragments (should be node_id:timestamp)
    let (success, _stdout, _stderr) = run_motlie(&[
        "db",
        "-p",
        db_path.to_str().unwrap(),
        "scan",
        "node-fragments",
        "--last",
        "invalid-cursor",
    ]);

    // The command should fail with an invalid cursor format
    assert!(!success, "Should fail with invalid cursor format");
}

// ============================================================================
// Output Format Tests
// ============================================================================

#[tokio::test]
async fn test_output_format_table_default() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");

    let _test_data = insert_test_data(&db_path).await;

    // Default format should be table
    let (success, stdout, _) = run_motlie(&[
        "db",
        "-p",
        db_path.to_str().unwrap(),
        "scan",
        "nodes",
    ]);
    assert!(success);

    // Table format should have header row with column names
    let first_line = stdout.lines().next().unwrap();
    assert!(
        first_line.contains("SINCE"),
        "Default format should be table with SINCE header"
    );
    assert!(
        first_line.contains("ID"),
        "Default format should be table with ID header"
    );
}

#[tokio::test]
async fn test_output_format_table() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");

    let _test_data = insert_test_data(&db_path).await;

    let (success, stdout, _) = run_motlie(&[
        "db",
        "-p",
        db_path.to_str().unwrap(),
        "scan",
        "nodes",
        "-f",
        "table",
    ]);
    assert!(success);

    // Table format should have header row with column names
    let lines: Vec<&str> = stdout.lines().collect();
    assert!(lines.len() >= 3, "Table should have header, separator, and data rows");

    // First line should be headers
    let header_line = lines[0];
    assert!(header_line.contains("SINCE"), "Missing SINCE header");
    assert!(header_line.contains("UNTIL"), "Missing UNTIL header");
    assert!(header_line.contains("ID"), "Missing ID header");
    assert!(header_line.contains("NAME"), "Missing NAME header");

    // Second line should be separators
    let separator_line = lines[1];
    assert!(
        separator_line.contains("---"),
        "Missing separator line with dashes"
    );
}

// ============================================================================
// Pagination Cursor Format Tests
// ============================================================================

#[tokio::test]
async fn test_node_fragment_pagination_cursor() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");

    let _test_data = insert_test_data(&db_path).await;

    // Get first fragment using TSV format for parsing
    let (success, stdout, _) = run_motlie(&[
        "db",
        "-p",
        db_path.to_str().unwrap(),
        "scan",
        "node-fragments",
        "--limit",
        "1",
        "--format",
        "tsv",
    ]);
    assert!(success);

    let first_line = stdout.lines().next().unwrap();
    let parts: Vec<&str> = first_line.split('\t').collect();
    let node_id = parts[2];
    let timestamp = parts[3];

    // Use cursor format: node_id:timestamp
    let cursor = format!("{}:{}", node_id, timestamp);
    let (success, stdout_page2, stderr) = run_motlie(&[
        "db",
        "-p",
        db_path.to_str().unwrap(),
        "scan",
        "node-fragments",
        "--limit",
        "1",
        "--last",
        &cursor,
        "--format",
        "tsv",
    ]);
    assert!(success, "Pagination with cursor failed: {}", stderr);

    // Should get different fragment or empty (if only 2 fragments)
    if !stdout_page2.is_empty() {
        let second_line = stdout_page2.lines().next().unwrap();
        let parts2: Vec<&str> = second_line.split('\t').collect();
        let node_id2 = parts2[2];
        let timestamp2 = parts2[3];

        // Should be different from first
        assert!(
            node_id != node_id2 || timestamp != timestamp2,
            "Pagination should return different fragment"
        );
    }
}

#[tokio::test]
async fn test_outgoing_edge_pagination_cursor() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("graph_db");

    let _test_data = insert_test_data(&db_path).await;

    // Get first edge using TSV format for parsing
    let (success, stdout, _) = run_motlie(&[
        "db",
        "-p",
        db_path.to_str().unwrap(),
        "scan",
        "outgoing-edges",
        "--limit",
        "1",
        "--format",
        "tsv",
    ]);
    assert!(success);

    let first_line = stdout.lines().next().unwrap();
    let parts: Vec<&str> = first_line.split('\t').collect();
    let src_id = parts[2];
    let dst_id = parts[3];
    let edge_name = parts[4];

    // Use cursor format: src_id:dst_id:edge_name
    let cursor = format!("{}:{}:{}", src_id, dst_id, edge_name);
    let (success, stdout_page2, stderr) = run_motlie(&[
        "db",
        "-p",
        db_path.to_str().unwrap(),
        "scan",
        "outgoing-edges",
        "--limit",
        "1",
        "--last",
        &cursor,
        "--format",
        "tsv",
    ]);
    assert!(success, "Pagination with cursor failed: {}", stderr);

    // Should get second edge
    let lines: Vec<&str> = stdout_page2.lines().collect();
    assert_eq!(lines.len(), 1, "Should get exactly 1 edge after cursor");
}
