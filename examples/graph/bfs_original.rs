/// Breadth-First Search (BFS) Implementation - ORIGINAL (unoptimized)
///
/// This is the original implementation using individual NodeById calls.
/// Used for benchmarking comparison against the NodesByIdsMulti optimized version.

// Include the common module
#[path = "common.rs"]
mod common;

use anyhow::Result;
use common::{
    build_graph, compute_hash, get_disk_metrics, measure_time_and_memory,
    measure_time_and_memory_async, parse_scale_factor, GraphEdge, GraphMetrics, GraphNode,
    Implementation,
};
use motlie_db::query::{NodeById, OutgoingEdges, Runnable as QueryRunnable};
use motlie_db::Id;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::Bfs;
use std::collections::{HashMap, HashSet, VecDeque};
use std::env;
use std::path::Path;
use tokio::time::Duration;

/// Create a test graph for BFS traversal
fn create_test_graph(scale: usize) -> (Vec<GraphNode>, Vec<GraphEdge>) {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut all_node_ids = Vec::new();

    let base_names = vec!["Root", "L1_A", "L1_B", "L2_A", "L2_B", "L2_C", "L2_D", "L3_A", "L3_B", "L3_C"];
    let total_nodes = scale * base_names.len();

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

    for i in 0..total_nodes {
        let children = vec![i * 2 + 1, i * 2 + 2, i * 3 + 1];
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

fn bfs_petgraph(start_node: NodeIndex, graph: &DiGraph<String, f64>) -> Vec<String> {
    let mut visited = Vec::new();
    let mut bfs = Bfs::new(&graph, start_node);
    while let Some(node_idx) = bfs.next(&graph) {
        visited.push(graph[node_idx].clone());
    }
    visited
}

/// ORIGINAL BFS implementation - uses individual NodeById calls
async fn bfs_motlie_original(
    start_node: Id,
    reader: &motlie_db::graph::reader::Reader,
    timeout: Duration,
) -> Result<Vec<String>> {
    let mut visited = HashSet::new();
    let mut visit_order = Vec::new();
    let mut queue = VecDeque::new();

    queue.push_back(start_node);
    visited.insert(start_node);

    while let Some(current_id) = queue.pop_front() {
        // Individual NodeById call for each node
        let (name, _summary) = motlie_db::graph::query::NodeById::new(current_id, None)
            .run(reader, timeout)
            .await?;

        visit_order.push(name);

        let edges = OutgoingEdges::new(current_id, None)
            .run(reader, timeout)
            .await?;

        for (_weight, _src, dst, _name) in edges {
            if !visited.contains(&dst) {
                visited.insert(dst);
                queue.push_back(dst);
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
        std::process::exit(1);
    }

    let implementation = Implementation::from_str(&args[1])?;
    let db_path = Path::new(&args[2]);
    let scale = parse_scale_factor(&args[3])?;

    let (nodes, edges) = create_test_graph(scale);
    let num_nodes = nodes.len();
    let num_edges = edges.len();

    match implementation {
        Implementation::Reference => {
            let mut pg_graph = DiGraph::new();
            let mut pg_node_map = HashMap::new();

            for node in &nodes {
                let idx = pg_graph.add_node(node.name.clone());
                pg_node_map.insert(node.name.clone(), idx);
            }

            for edge in &edges {
                let src_name = nodes.iter().find(|n| n.id == edge.source).unwrap().name.as_str();
                let dst_name = nodes.iter().find(|n| n.id == edge.target).unwrap().name.as_str();
                let src_idx = pg_node_map[src_name];
                let dst_idx = pg_node_map[dst_name];
                pg_graph.add_edge(src_idx, dst_idx, edge.weight.unwrap_or(1.0));
            }

            let start_name = nodes[0].name.clone();
            let start_idx = pg_node_map[&start_name];
            let (result, time_ms, memory) = measure_time_and_memory(|| bfs_petgraph(start_idx, &pg_graph));
            let result_hash = Some(compute_hash(&result));

            let metrics = GraphMetrics {
                algorithm_name: "BFS_Original".to_string(),
                implementation: Implementation::Reference,
                scale,
                num_nodes,
                num_edges,
                execution_time_ms: time_ms,
                memory_usage_bytes: memory,
                result_hash,
                disk_files: None,
                disk_size_bytes: None,
            };

            metrics.print_csv();
        }
        Implementation::MotlieDb => {
            let (reader, name_to_id, _handles) = build_graph(db_path, nodes.clone(), edges).await?;
            let start_name = nodes[0].name.clone();
            let start_id = name_to_id[&start_name];
            let timeout = Duration::from_secs(120);

            let (result, time_ms, memory) = measure_time_and_memory_async(|| bfs_motlie_original(start_id, reader.graph(), timeout)).await;
            let result = result?;
            let result_hash = Some(compute_hash(&result));

            let (disk_files, disk_size) = get_disk_metrics(db_path).unwrap_or((0, 0));

            let metrics = GraphMetrics {
                algorithm_name: "BFS_Original".to_string(),
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
