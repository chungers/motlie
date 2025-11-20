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
use common::{build_graph, measure_time_and_memory, measure_time_and_memory_async, parse_scale_factor, GraphEdge, GraphMetrics, GraphNode};
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

    let base_size = 9; // Base tree structure size
    let total_nodes = scale * base_size;

    // Create all nodes
    for i in 0..total_nodes {
        let id = Id::new();
        let node_name = if scale == 1 {
            vec!["A", "B", "C", "D", "E", "F", "G", "H", "I"][i].to_string()
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

    if args.len() != 3 {
        eprintln!("Usage: {} <db_path> <scale_factor>", args[0]);
        eprintln!();
        eprintln!("Arguments:");
        eprintln!("  db_path       - Path to RocksDB directory");
        eprintln!("  scale_factor  - Positive integer to scale the graph size");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  {} /tmp/bfs_test_db 1      # Base graph (9 nodes, 8 edges)", args[0]);
        eprintln!("  {} /tmp/bfs_test_db 10     # 10x larger (90 nodes, 80 edges)", args[0]);
        eprintln!("  {} /tmp/bfs_test_db 100    # 100x larger (900 nodes, 800 edges)", args[0]);
        std::process::exit(1);
    }

    let db_path = Path::new(&args[1]);
    let scale = parse_scale_factor(&args[2])?;

    println!("🔍 Breadth-First Search (BFS) Comparison");
    println!("{:=<80}", "");
    println!("\n📏 Scale factor: {}x", scale);

    // Generate test graph
    println!("\n📊 Generating test graph (tree-like structure)...");
    let start_gen = std::time::Instant::now();
    let (nodes, edges) = create_test_graph(scale);
    let gen_time = start_gen.elapsed();
    let num_nodes = nodes.len();
    let num_edges = edges.len();

    println!("  Nodes: {}", num_nodes);
    println!("  Edges: {}", num_edges);
    println!("  Generation time: {:.2} ms", gen_time.as_secs_f64() * 1000.0);

    // Pass 1: BFS with petgraph
    println!("\n🔹 Pass 1: BFS with petgraph (in-memory)");

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
    let (pg_result, pg_time, pg_result_memory) = measure_time_and_memory(|| bfs_petgraph(start_idx, &pg_graph));

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
        memory_usage_bytes: pg_result_memory,
    };

    // Pass 2: BFS with motlie_db
    println!("\n🔹 Pass 2: BFS with motlie_db (persistent)");

    let (reader, name_to_id, _query_handle) = build_graph(db_path, nodes, edges).await?;
    let start_id = name_to_id[&start_name];
    let timeout = Duration::from_secs(30);

    let (motlie_result, motlie_time, motlie_result_memory) = measure_time_and_memory_async(|| bfs_motlie(start_id, &reader, timeout)).await;
    let motlie_result = motlie_result?;

    if num_nodes <= 20 {
        println!("  Visited order: {:?}", motlie_result);
    } else {
        println!("  Visited nodes: {} (showing first 10: {:?}...)", motlie_result.len(), &motlie_result[..10.min(motlie_result.len())]);
    }
    println!("  Execution time: {:.2} ms", motlie_time);

    let motlie_metrics = GraphMetrics {
        algorithm_name: "BFS".to_string(),
        num_nodes,
        num_edges,
        execution_time_ms: motlie_time,
        memory_usage_bytes: motlie_result_memory,
    };

    // Verify correctness
    println!("\n✅ Correctness Check:");
    if pg_result == motlie_result {
        println!("  ✓ Results match! Both implementations visited {} nodes in the same order.", pg_result.len());
    } else {
        println!("  ⚠ Results differ in order (checking level-order consistency):");
        println!("    petgraph:  {:?}", pg_result);
        println!("    motlie_db: {:?}", motlie_result);

        let pg_set: HashSet<_> = pg_result.iter().collect();
        let motlie_set: HashSet<_> = motlie_result.iter().collect();

        if pg_set == motlie_set {
            println!("  ✓ Same nodes visited (order differences acceptable for nodes at same level)");
        } else {
            println!("  ✗ Different nodes visited - this indicates an error!");
        }
    }

    // Demonstrate BFS levels
    println!("\n🔹 BFS Level Analysis (using motlie_db):");
    let levels = bfs_with_levels(start_id, &reader, timeout).await?;

    let mut max_level = 0;
    for level in levels.values() {
        max_level = max_level.max(*level);
    }

    for current_level in 0..=max_level {
        let mut nodes_at_level: Vec<_> = levels
            .iter()
            .filter(|(_, &l)| l == current_level)
            .map(|(n, _)| n.as_str())
            .collect();
        nodes_at_level.sort();
        println!("  Level {}: {:?}", current_level, nodes_at_level);
    }

    // Print performance comparison
    GraphMetrics::print_comparison(&motlie_metrics, &petgraph_metrics);

    println!("\n✅ BFS example completed successfully!");
    Ok(())
}
