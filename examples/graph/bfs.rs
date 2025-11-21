/// Breadth-First Search (BFS) Implementation Comparison
///
/// This example demonstrates BFS traversal using both petgraph (in-memory)
/// and motlie_db (persistent graph database), comparing correctness and performance.
///
/// BFS is commonly used for:
/// - Finding shortest paths (unweighted graphs)
/// - Level-order traversal
/// - Finding connected components
///
/// Usage: bfs <db_path>

// Include the common module
#[path = "common.rs"]
mod common;

use anyhow::Result;
use common::{build_graph, compute_hash, get_disk_metrics, measure_time_and_memory, measure_time_and_memory_async, parse_scale_factor, GraphEdge, GraphMetrics, GraphNode, Implementation};
use motlie_db::{Id, OutgoingEdges, QueryRunnable};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::Bfs;
use std::collections::{HashMap, HashSet, VecDeque};
use std::env;
use std::path::Path;
use tokio::time::Duration;

/// Create a test graph for BFS traversal
/// Creates a connected tree-like graph suitable for demonstrating BFS level-order traversal
/// The scale parameter determines the total number of nodes (scale * 9)
fn create_test_graph(scale: usize) -> (Vec<GraphNode>, Vec<GraphEdge>) {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut all_node_ids = Vec::new();

    let base_names = vec!["Root", "L1_A", "L1_B", "L2_A", "L2_B", "L2_C", "L2_D", "L3_A", "L3_B", "L3_C"];
    let total_nodes = scale * base_names.len();

    // Create all nodes
    for i in 0..total_nodes {
        let id = Id::new();
        let node_name = if scale == 1 {
            base_names[i % base_names.len()].to_string()
        } else {
            format!("N{}", i)
        };
        all_node_ids.push(id);
        nodes.push(GraphNode {
            id,
            name: node_name,
        });
    }

    // Create edges forming a connected tree-like structure
    // Each "level" has nodes that fan out to the next level
    for i in 0..total_nodes {
        // Each node connects to 2-3 nodes ahead to maintain tree-like structure
        let children = vec![
            i * 2 + 1,
            i * 2 + 2,
            i * 3 + 1,
        ];

        for &child in &children {
            if child < total_nodes {
                edges.push(GraphEdge {
                    source: all_node_ids[i],
                    target: all_node_ids[child],
                    name: format!("edge_{}_{}", i, child),
                    weight: Some(1.0),
                });
            }
        }
    }

    (nodes, edges)
}

/// BFS implementation using petgraph
fn bfs_petgraph(start_node: NodeIndex, graph: &DiGraph<String, f64>) -> Vec<String> {
    let mut visited = Vec::new();
    let mut bfs = Bfs::new(&graph, start_node);

    while let Some(node_idx) = bfs.next(&graph) {
        visited.push(graph[node_idx].clone());
    }

    visited
}

/// BFS implementation using motlie_db
async fn bfs_motlie(
    start_node: Id,
    reader: &motlie_db::Reader,
    timeout: Duration,
) -> Result<Vec<String>> {
    let mut visited = HashSet::new();
    let mut visit_order = Vec::new();
    let mut queue = VecDeque::new();

    queue.push_back(start_node);
    visited.insert(start_node);

    while let Some(current_id) = queue.pop_front() {
        // Get node name
        let (name, _summary) = motlie_db::NodeById::new(current_id, None)
            .run(reader, timeout)
            .await?;

        visit_order.push(name);

        // Get outgoing edges
        let edges = OutgoingEdges::new(current_id, None)
            .run(reader, timeout)
            .await?;

        // Add unvisited neighbors to queue
        for (_weight, _src, dst, _name) in edges {
            if !visited.contains(&dst) {
                visited.insert(dst);
                queue.push_back(dst);
            }
        }
    }

    Ok(visit_order)
}

/// Calculate the distance (level) from start node to each visited node
async fn bfs_with_levels(
    start_node: Id,
    reader: &motlie_db::Reader,
    timeout: Duration,
) -> Result<HashMap<String, usize>> {
    let mut levels = HashMap::new();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    queue.push_back((start_node, 0_usize));
    visited.insert(start_node);

    while let Some((current_id, level)) = queue.pop_front() {
        // Get node name
        let (name, _summary) = motlie_db::NodeById::new(current_id, None)
            .run(reader, timeout)
            .await?;

        levels.insert(name, level);

        // Get outgoing edges
        let edges = OutgoingEdges::new(current_id, None)
            .run(reader, timeout)
            .await?;

        // Add unvisited neighbors to queue with incremented level
        for (_weight, _src, dst, _name) in edges {
            if !visited.contains(&dst) {
                visited.insert(dst);
                queue.push_back((dst, level + 1));
            }
        }
    }

    Ok(levels)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 4 {
        eprintln!("Usage: {} <implementation> <db_path> <scale_factor>", args[0]);
        eprintln!();
        eprintln!("Arguments:");
        eprintln!("  implementation - 'reference' or 'motlie_db'");
        eprintln!("  db_path        - Path to RocksDB directory");
        eprintln!("  scale_factor   - Positive integer to scale the graph size");
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} reference /tmp/bfs_test_db 1      # petgraph, 9 nodes", args[0]);
        eprintln!("  {} motlie_db /tmp/bfs_test_db 1      # motlie_db, 9 nodes", args[0]);
        eprintln!("  {} reference /tmp/bfs_test_db 1000   # petgraph, 9000 nodes", args[0]);
        eprintln!("  {} motlie_db /tmp/bfs_test_db 1000   # motlie_db, 9000 nodes", args[0]);
        std::process::exit(1);
    }

    let implementation = Implementation::from_str(&args[1])?;
    let db_path = Path::new(&args[2]);
    let scale = parse_scale_factor(&args[3])?;

    // Generate test graph
    let (nodes, edges) = create_test_graph(scale);
    let num_nodes = nodes.len();
    let num_edges = edges.len();

    match implementation {
        Implementation::Reference => {
            // Run BFS with petgraph (reference implementation)
            let mut pg_graph = DiGraph::new();
            let mut pg_node_map = HashMap::new();

            // Add nodes to petgraph
            for node in &nodes {
                let idx = pg_graph.add_node(node.name.clone());
                pg_node_map.insert(node.name.clone(), idx);
            }

            // Add edges to petgraph
            for edge in &edges {
                let src_name = nodes.iter().find(|n| n.id == edge.source).unwrap().name.as_str();
                let dst_name = nodes.iter().find(|n| n.id == edge.target).unwrap().name.as_str();
                let src_idx = pg_node_map[src_name];
                let dst_idx = pg_node_map[dst_name];
                pg_graph.add_edge(src_idx, dst_idx, edge.weight.unwrap_or(1.0));
            }

            // Run BFS starting from first node
            let start_name = nodes[0].name.clone();
            let start_idx = pg_node_map[&start_name];
            let (result, time_ms, memory) = measure_time_and_memory(|| bfs_petgraph(start_idx, &pg_graph));
            let result_hash = Some(compute_hash(&result));

            let metrics = GraphMetrics {
                algorithm_name: "BFS".to_string(),
                implementation: Implementation::Reference,
                scale,
                num_nodes,
                num_edges,
                execution_time_ms: time_ms,
                memory_usage_bytes: memory,
                result_hash,
                disk_files: None,       // Reference implementation doesn't use disk
                disk_size_bytes: None,
            };

            metrics.print_csv();
        }
        Implementation::MotlieDb => {
            // Run BFS with motlie_db
            let (reader, name_to_id, _query_handle) = build_graph(db_path, nodes.clone(), edges).await?;
            let start_name = nodes[0].name.clone();
            let start_id = name_to_id[&start_name];
            let timeout = Duration::from_secs(60); // Longer timeout for large graphs

            let (result, time_ms, memory) = measure_time_and_memory_async(|| bfs_motlie(start_id, &reader, timeout)).await;
            let result = result?;
            let result_hash = Some(compute_hash(&result));

            // Measure disk usage after algorithm completes
            let (disk_files, disk_size) = get_disk_metrics(db_path).unwrap_or((0, 0));

            let metrics = GraphMetrics {
                algorithm_name: "BFS".to_string(),
                implementation: Implementation::MotlieDb,
                scale,
                num_nodes,
                num_edges,
                execution_time_ms: time_ms,
                memory_usage_bytes: memory,
                result_hash,
                disk_files: Some(disk_files),
                disk_size_bytes: Some(disk_size),
            };

            metrics.print_csv();
        }
    }

    Ok(())
}
