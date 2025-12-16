/// Topological Sort Implementation Comparison
///
/// This example demonstrates topological sorting using both petgraph (in-memory)
/// and motlie_db (persistent graph database), comparing correctness and performance.
///
/// Topological sort is used for:
/// - Task scheduling with dependencies
/// - Build systems (e.g., Makefile, Cargo)
/// - Course prerequisite ordering
/// - Dependency resolution
///
/// Usage: toposort <db_path>

// Include the common module
#[path = "common.rs"]
mod common;

use anyhow::Result;
use common::{build_graph, compute_hash, get_disk_metrics, measure_time_and_memory, measure_time_and_memory_async, parse_scale_factor, GraphEdge, GraphMetrics, GraphNode, Implementation};
use motlie_db::query::{OutgoingEdges, Runnable as QueryRunnable};
use motlie_db::Id;
use petgraph::algo::toposort as petgraph_toposort;
use petgraph::graph::DiGraph;
use std::collections::{HashMap, VecDeque};
use std::env;
use std::path::Path;
use tokio::time::Duration;

/// Create a DAG (Directed Acyclic Graph) for topological sort
/// Represents a connected build dependency graph with multiple modules
/// The scale parameter determines the total number of tasks (scale * 8)
fn create_test_graph(scale: usize) -> (Vec<GraphNode>, Vec<GraphEdge>) {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut all_node_ids = Vec::new();

    let base_names = vec!["Start", "Task_A", "Task_B", "Task_C", "Task_D", "Task_E", "Task_F", "Task_G", "Task_H", "End"];
    let total_nodes = scale * base_names.len();

    // Create all nodes
    for i in 0..total_nodes {
        let id = Id::new();
        let node_name = if scale == 1 {
            base_names[i % base_names.len()].to_string()
        } else {
            format!("task_{}", i)
        };
        all_node_ids.push(id);
        nodes.push(GraphNode {
            id,
            name: node_name,
        });
    }

    // Create a connected DAG with proper dependency structure
    // Each module (cluster of 10 tasks) has internal dependencies
    // and dependencies on previous modules to form a coherent build pipeline
    let module_size = base_names.len();

    for module in 0..scale {
        let module_start = module * module_size;

        // Internal dependencies within each module (following task pipeline pattern)
        let internal_deps = vec![
            (0, 1), // Start -> Task_A
            (0, 2), // Start -> Task_B
            (1, 3), // Task_A -> Task_C
            (2, 3), // Task_B -> Task_C
            (1, 4), // Task_A -> Task_D
            (3, 5), // Task_C -> Task_E
            (4, 6), // Task_D -> Task_F
            (5, 7), // Task_E -> Task_G
            (6, 8), // Task_F -> Task_H
            (7, 9), // Task_G -> End
            (8, 9), // Task_H -> End
        ];

        for (src_offset, dst_offset) in internal_deps {
            edges.push(GraphEdge {
                source: all_node_ids[module_start + src_offset],
                target: all_node_ids[module_start + dst_offset],
                name: format!("dep_m{}_{}_{}", module, src_offset, dst_offset),
                weight: Some(1.0),
            });
        }

        // Connect to next module: End of current module -> Start of next module
        // This creates a linear dependency chain of modules
        if module < scale - 1 {
            let next_module_start = (module + 1) * module_size;
            edges.push(GraphEdge {
                source: all_node_ids[module_start + 9], // End
                target: all_node_ids[next_module_start], // next Start
                name: format!("bridge_m{}_to_m{}", module, module + 1),
                weight: Some(1.0),
            });
        }
    }

    (nodes, edges)
}

/// Topological sort using Kahn's algorithm with motlie_db
async fn toposort_motlie(
    all_nodes: &[Id],
    reader: &motlie_db::graph::reader::Reader,
    timeout: Duration,
) -> Result<Vec<String>> {
    // Calculate in-degrees for all nodes
    let mut in_degree: HashMap<Id, usize> = HashMap::new();
    let mut name_map: HashMap<Id, String> = HashMap::new();

    for &node_id in all_nodes {
        in_degree.insert(node_id, 0);

        // Get node name
        let (name, _summary) = motlie_db::graph::query::NodeById::new(node_id, None)
            .run(reader, timeout)
            .await?;
        name_map.insert(node_id, name);
    }

    // Count incoming edges for each node
    for &node_id in all_nodes {
        let outgoing = OutgoingEdges::new(node_id, None)
            .run(reader, timeout)
            .await?;

        for (_weight, _src, dst, _name) in outgoing {
            *in_degree.entry(dst).or_insert(0) += 1;
        }
    }

    // Start with nodes that have no incoming edges
    let mut queue: VecDeque<Id> = in_degree
        .iter()
        .filter(|(_, &deg)| deg == 0)
        .map(|(&id, _)| id)
        .collect();

    let mut sorted = Vec::new();

    while let Some(current_id) = queue.pop_front() {
        sorted.push(name_map[&current_id].clone());

        // Get outgoing edges
        let outgoing = OutgoingEdges::new(current_id, None)
            .run(reader, timeout)
            .await?;

        // Decrease in-degree of neighbors
        for (_weight, _src, dst, _name) in outgoing {
            if let Some(degree) = in_degree.get_mut(&dst) {
                *degree -= 1;
                if *degree == 0 {
                    queue.push_back(dst);
                }
            }
        }
    }

    // Check if all nodes were processed (cycle detection)
    if sorted.len() != all_nodes.len() {
        anyhow::bail!("Graph contains a cycle - topological sort not possible");
    }

    Ok(sorted)
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
        eprintln!("  {} reference /tmp/toposort_test_db 1      # petgraph, 8 nodes", args[0]);
        eprintln!("  {} motlie_db /tmp/toposort_test_db 1      # motlie_db, 8 nodes", args[0]);
        eprintln!("  {} reference /tmp/toposort_test_db 1000   # petgraph, 8000 nodes", args[0]);
        eprintln!("  {} motlie_db /tmp/toposort_test_db 1000   # motlie_db, 8000 nodes", args[0]);
        std::process::exit(1);
    }

    let implementation = Implementation::from_str(&args[1])?;
    let db_path = Path::new(&args[2]);
    let scale = parse_scale_factor(&args[3])?;

    // Generate test graph
    let (nodes, edges) = create_test_graph(scale);
    let num_nodes = nodes.len();
    let num_edges = edges.len();

    let node_ids: Vec<Id> = nodes.iter().map(|n| n.id).collect();

    match implementation {
        Implementation::Reference => {
            // Run topological sort with petgraph (reference implementation)
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

            // Run topological sort
            let (result, time_ms, memory) = measure_time_and_memory(|| {
                petgraph_toposort(&pg_graph, None)
                    .map(|sorted| {
                        sorted
                            .iter()
                            .map(|&idx| pg_graph[idx].clone())
                            .collect::<Vec<_>>()
                    })
                    .map_err(|_| anyhow::anyhow!("Graph contains a cycle"))
            });

            let result = result?;
            let result_hash = Some(compute_hash(&result));

            let metrics = GraphMetrics {
                algorithm_name: "Topological Sort".to_string(),
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
            // Run topological sort with motlie_db
            let (reader, _name_to_id, _handles) = build_graph(db_path, nodes, edges).await?;
            let timeout = Duration::from_secs(60); // Longer timeout for large graphs

            let (result, time_ms, memory) = measure_time_and_memory_async(|| toposort_motlie(&node_ids, reader.graph(), timeout)).await;
            let result = result?;
            let result_hash = Some(compute_hash(&result));

            // Measure disk usage after algorithm completes
            let (disk_files, disk_size) = get_disk_metrics(db_path).unwrap_or((0, 0));

            let metrics = GraphMetrics {
                algorithm_name: "Topological Sort".to_string(),
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
