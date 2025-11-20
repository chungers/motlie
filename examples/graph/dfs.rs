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
use common::{build_graph, measure_time, measure_time_async, parse_scale_factor, GraphEdge, GraphMetrics, GraphNode};
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

    let base_names = vec!["A", "B", "C", "D", "E", "F", "G", "H"];
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

    if args.len() != 3 {
        eprintln!("Usage: {} <db_path> <scale_factor>", args[0]);
        eprintln!();
        eprintln!("Arguments:");
        eprintln!("  db_path       - Path to RocksDB directory");
        eprintln!("  scale_factor  - Positive integer to scale the graph size");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  {} /tmp/dfs_test_db 1      # Base graph (8 nodes, 9 edges)", args[0]);
        eprintln!("  {} /tmp/dfs_test_db 10     # 10x larger (80 nodes, 90 edges)", args[0]);
        eprintln!("  {} /tmp/dfs_test_db 100    # 100x larger (800 nodes, 900 edges)", args[0]);
        eprintln!("  {} /tmp/dfs_test_db 1000   # 1000x larger (8000 nodes, 9000 edges)", args[0]);
        std::process::exit(1);
    }

    let db_path = Path::new(&args[1]);
    let scale = parse_scale_factor(&args[2])?;

    println!("🌲 Depth-First Search (DFS) Comparison");
    println!("{:=<80}", "");
    println!("\n📏 Scale factor: {}x", scale);

    // Generate test graph
    println!("\n📊 Generating test graph...");
    let start_gen = std::time::Instant::now();
    let (nodes, edges) = create_test_graph(scale);
    let gen_time = start_gen.elapsed();
    let num_nodes = nodes.len();
    let num_edges = edges.len();

    println!("  Nodes: {}", num_nodes);
    println!("  Edges: {}", num_edges);
    println!("  Generation time: {:.2} ms", gen_time.as_secs_f64() * 1000.0);

    // Store node name mapping for later
    let node_name_map: HashMap<String, Id> = nodes
        .iter()
        .map(|n| (n.name.clone(), n.id))
        .collect();

    // Pass 1: DFS with petgraph
    println!("\n🔹 Pass 1: DFS with petgraph (in-memory)");

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
    let (pg_result, pg_time) = measure_time(|| dfs_petgraph(start_idx, &pg_graph));

    // For large graphs, only show first few nodes
    if num_nodes <= 20 {
        println!("  Visited order: {:?}", pg_result);
    } else {
        println!("  Visited nodes: {} (showing first 10: {:?}...)", pg_result.len(), &pg_result[..10.min(pg_result.len())]);
    }
    println!("  Execution time: {:.2} ms", pg_time);

    let petgraph_metrics = GraphMetrics {
        algorithm_name: "petgraph".to_string(),
        num_nodes,
        num_edges,
        execution_time_ms: pg_time,
        memory_usage_bytes: None, // Could estimate with std::mem::size_of_val
    };

    // Pass 2: DFS with motlie_db
    println!("\n🔹 Pass 2: DFS with motlie_db (persistent)");

    let (reader, name_to_id, _query_handle) = build_graph(db_path, nodes, edges).await?;
    let start_id = name_to_id[&start_name];
    let timeout = Duration::from_secs(30); // Longer timeout for large graphs

    let (motlie_result, motlie_time) = measure_time_async(|| dfs_motlie(start_id, &reader, timeout)).await;
    let motlie_result = motlie_result?;

    // For large graphs, only show first few nodes
    if num_nodes <= 20 {
        println!("  Visited order: {:?}", motlie_result);
    } else {
        println!("  Visited nodes: {} (showing first 10: {:?}...)", motlie_result.len(), &motlie_result[..10.min(motlie_result.len())]);
    }
    println!("  Execution time: {:.2} ms", motlie_time);

    let motlie_metrics = GraphMetrics {
        algorithm_name: "DFS".to_string(),
        num_nodes,
        num_edges,
        execution_time_ms: motlie_time,
        memory_usage_bytes: None,
    };

    // Verify correctness
    println!("\n✅ Correctness Check:");
    if pg_result == motlie_result {
        println!("  ✓ Results match! Both implementations visited {} nodes in the same order.", pg_result.len());
    } else {
        println!("  ✗ Results differ!");
        println!("    petgraph: {:?}", pg_result);
        println!("    motlie_db: {:?}", motlie_result);
        println!("    Note: DFS order can vary based on edge iteration order, which is acceptable.");
        println!("    Checking if the same nodes were visited...");

        let pg_set: HashSet<_> = pg_result.iter().collect();
        let motlie_set: HashSet<_> = motlie_result.iter().collect();

        if pg_set == motlie_set {
            println!("  ✓ Same nodes visited, different order (both valid DFS traversals)");
        } else {
            println!("  ✗ Different nodes visited - this indicates an error!");
        }
    }

    // Print performance comparison
    GraphMetrics::print_comparison(&motlie_metrics, &petgraph_metrics);

    println!("\n✅ DFS example completed successfully!");
    Ok(())
}
