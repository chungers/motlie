/// Depth-First Search (DFS) Implementation Comparison
///
/// This example demonstrates DFS traversal using both petgraph (in-memory)
/// and motlie_db (persistent graph database), comparing correctness and performance.
///
/// Usage: dfs <db_path>
///
/// The program:
/// 1. Generates a test graph
/// 2. Runs DFS using petgraph
/// 3. Runs DFS using motlie_db
/// 4. Compares results and metrics

// Include the common module
#[path = "common.rs"]
mod common;

use anyhow::Result;
use common::{build_graph, compute_hash, get_disk_metrics, measure_time_and_memory, measure_time_and_memory_async, parse_scale_factor, GraphEdge, GraphMetrics, GraphNode, Implementation};
use motlie_db::{Id, OutgoingEdges, QueryRunnable};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::Dfs;
use std::collections::{HashMap, HashSet};
use std::env;
use std::path::Path;
use tokio::time::Duration;

/// Create a test graph for DFS traversal
/// Creates a connected directed graph suitable for DFS exploration
/// The scale parameter determines the total number of nodes (scale * 8)
fn create_test_graph(scale: usize) -> (Vec<GraphNode>, Vec<GraphEdge>) {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut all_node_ids = Vec::new();

    let base_names = vec!["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"];
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

    // Create edges to form a connected graph with interesting structure
    // Pattern: Create clusters with inter-cluster connections
    let cluster_size = base_names.len();

    for cluster in 0..scale {
        let cluster_start = cluster * cluster_size;

        // Edges within each cluster (following base pattern)
        let base_edge_pattern = vec![
            (0, 1), (0, 2), (1, 3), (1, 4),
            (2, 4), (3, 5), (3, 6), (4, 6), (6, 7),
            (5, 8), (7, 9),
        ];

        for (src_offset, dst_offset) in base_edge_pattern {
            edges.push(GraphEdge {
                source: all_node_ids[cluster_start + src_offset],
                target: all_node_ids[cluster_start + dst_offset],
                name: format!("edge_c{}_{}_{}", cluster, src_offset, dst_offset),
                weight: Some(1.0 + (src_offset as f64) * 0.5),
            });
        }

        // Connect clusters together to ensure full connectivity
        if cluster < scale - 1 {
            let next_cluster_start = (cluster + 1) * cluster_size;
            // Connect last node of current cluster to first node of next cluster
            edges.push(GraphEdge {
                source: all_node_ids[cluster_start + cluster_size - 1],
                target: all_node_ids[next_cluster_start],
                name: format!("bridge_{}_to_{}", cluster, cluster + 1),
                weight: Some(1.0),
            });
            // Add a back edge for interesting cycles
            edges.push(GraphEdge {
                source: all_node_ids[next_cluster_start + 2],
                target: all_node_ids[cluster_start + 1],
                name: format!("back_{}_to_{}", cluster + 1, cluster),
                weight: Some(2.0),
            });
        }
    }

    (nodes, edges)
}

/// DFS implementation using petgraph
fn dfs_petgraph(start_node: NodeIndex, graph: &DiGraph<String, f64>) -> Vec<String> {
    let mut visited = Vec::new();
    let mut dfs = Dfs::new(&graph, start_node);

    while let Some(node_idx) = dfs.next(&graph) {
        visited.push(graph[node_idx].clone());
    }

    visited
}

/// DFS implementation using motlie_db
async fn dfs_motlie(
    start_node: Id,
    reader: &motlie_db::Reader,
    timeout: Duration,
) -> Result<Vec<String>> {
    let mut visited = HashSet::new();
    let mut visit_order = Vec::new();
    let mut stack = vec![start_node];

    while let Some(current_id) = stack.pop() {
        if visited.contains(&current_id) {
            continue;
        }

        visited.insert(current_id);

        // Get node name
        let (name, _summary) = motlie_db::NodeById::new(current_id, None)
            .run(reader, timeout)
            .await?;

        visit_order.push(name);

        // Get outgoing edges
        let edges = OutgoingEdges::new(current_id, None)
            .run(reader, timeout)
            .await?;

        // Add neighbors to stack (in reverse order to maintain left-to-right traversal)
        for (_weight, _src, dst, _name) in edges.iter().rev() {
            if !visited.contains(dst) {
                stack.push(*dst);
            }
        }
    }

    Ok(visit_order)
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
        eprintln!("  {} reference /tmp/dfs_test_db 1      # petgraph, 8 nodes", args[0]);
        eprintln!("  {} motlie_db /tmp/dfs_test_db 1      # motlie_db, 8 nodes", args[0]);
        eprintln!("  {} reference /tmp/dfs_test_db 1000   # petgraph, 8000 nodes", args[0]);
        eprintln!("  {} motlie_db /tmp/dfs_test_db 1000   # motlie_db, 8000 nodes", args[0]);
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
            // Run DFS with petgraph (reference implementation)
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

            // Run DFS starting from first node
            let start_name = nodes[0].name.clone();
            let start_idx = pg_node_map[&start_name];
            let (result, time_ms, memory) = measure_time_and_memory(|| dfs_petgraph(start_idx, &pg_graph));
            let result_hash = Some(compute_hash(&result));

            let metrics = GraphMetrics {
                algorithm_name: "DFS".to_string(),
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
            // Run DFS with motlie_db
            let (reader, name_to_id, _query_handle) = build_graph(db_path, nodes.clone(), edges).await?;
            let start_name = nodes[0].name.clone();
            let start_id = name_to_id[&start_name];
            let timeout = Duration::from_secs(60); // Longer timeout for large graphs

            let (result, time_ms, memory) = measure_time_and_memory_async(|| dfs_motlie(start_id, &reader, timeout)).await;
            let result = result?;
            let result_hash = Some(compute_hash(&result));

            // Measure disk usage after algorithm completes
            let (disk_files, disk_size) = get_disk_metrics(db_path).unwrap_or((0, 0));

            let metrics = GraphMetrics {
                algorithm_name: "DFS".to_string(),
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
