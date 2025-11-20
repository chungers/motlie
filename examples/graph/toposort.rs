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
use common::{build_graph, measure_time_and_memory, measure_time_and_memory_async, parse_scale_factor, GraphEdge, GraphMetrics, GraphNode};
use motlie_db::{Id, OutgoingEdges, QueryRunnable};
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

    let base_size = 8; // Base tasks per module
    let total_nodes = scale * base_size;

    // Create all nodes
    for i in 0..total_nodes {
        let id = Id::new();
        let node_name = if scale == 1 {
            vec!["checkout_code", "install_deps", "compile", "run_tests",
                 "lint", "package", "generate_docs", "deploy"][i].to_string()
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
    // Each module (cluster of 8 tasks) has internal dependencies
    // and dependencies on previous modules to form a coherent build pipeline
    let module_size = base_size;

    for module in 0..scale {
        let module_start = module * module_size;

        // Internal dependencies within each module (following build pipeline pattern)
        let internal_deps = vec![
            (0, 1), // checkout -> install_deps
            (0, 6), // checkout -> generate_docs
            (1, 2), // install_deps -> compile
            (2, 3), // compile -> run_tests
            (2, 4), // compile -> lint
            (3, 5), // run_tests -> package
            (4, 5), // lint -> package
            (6, 5), // generate_docs -> package
            (5, 7), // package -> deploy
        ];

        for (src_offset, dst_offset) in internal_deps {
            edges.push(GraphEdge {
                source: all_node_ids[module_start + src_offset],
                target: all_node_ids[module_start + dst_offset],
                name: format!("dep_m{}_{}_{}", module, src_offset, dst_offset),
                weight: Some(1.0),
            });
        }

        // Connect to next module: deploy of current module -> checkout of next module
        // This creates a linear dependency chain of modules
        if module < scale - 1 {
            let next_module_start = (module + 1) * module_size;
            edges.push(GraphEdge {
                source: all_node_ids[module_start + 7], // deploy
                target: all_node_ids[next_module_start], // next checkout
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
    reader: &motlie_db::Reader,
    timeout: Duration,
) -> Result<Vec<String>> {
    // Calculate in-degrees for all nodes
    let mut in_degree: HashMap<Id, usize> = HashMap::new();
    let mut name_map: HashMap<Id, String> = HashMap::new();

    for &node_id in all_nodes {
        in_degree.insert(node_id, 0);

        // Get node name
        let (name, _summary) = motlie_db::NodeById::new(node_id, None)
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

/// Verify that a topological ordering is valid
async fn verify_toposort(
    order: &[String],
    name_to_id: &HashMap<String, Id>,
    reader: &motlie_db::Reader,
    timeout: Duration,
) -> Result<bool> {
    // Create position map for the ordering
    let position: HashMap<&str, usize> = order
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();

    // Check that for every edge u -> v, u comes before v in the ordering
    for name in order {
        let node_id = name_to_id[name];

        let outgoing = OutgoingEdges::new(node_id, None)
            .run(reader, timeout)
            .await?;

        for (_weight, src, dst, _edge_name) in outgoing {
            let src_name = order.iter().find(|n| name_to_id[*n] == src).unwrap();
            let dst_name = order.iter().find(|n| name_to_id[*n] == dst).unwrap();

            let src_pos = position[src_name.as_str()];
            let dst_pos = position[dst_name.as_str()];

            if src_pos >= dst_pos {
                println!(
                    "  ✗ Violation: {} (pos {}) -> {} (pos {})",
                    src_name, src_pos, dst_name, dst_pos
                );
                return Ok(false);
            }
        }
    }

    Ok(true)
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
        eprintln!("  {} /tmp/toposort_test_db 1     # Base graph (8 tasks, 9 dependencies)", args[0]);
        eprintln!("  {} /tmp/toposort_test_db 100   # 100x larger (800 tasks, 900 dependencies)", args[0]);
        std::process::exit(1);
    }

    let db_path = Path::new(&args[1]);
    let scale = parse_scale_factor(&args[2])?;

    println!("📋 Topological Sort Comparison");
    println!("{:=<80}", "");
    println!("\n📏 Scale factor: {}x", scale);

    // Generate test graph
    println!("\n📊 Generating test DAG (build dependency graph)...");
    let start_gen = std::time::Instant::now();
    let (nodes, edges) = create_test_graph(scale);
    let gen_time = start_gen.elapsed();
    let num_nodes = nodes.len();
    let num_edges = edges.len();

    println!("  Nodes (tasks): {}", num_nodes);
    println!("  Edges (dependencies): {}", num_edges);
    println!("  Generation time: {:.2} ms", gen_time.as_secs_f64() * 1000.0);

    if num_nodes <= 20 {
        println!("\n  Build tasks:");
        for node in &nodes {
            println!("    - {}", node.name);
        }
    }

    let node_ids: Vec<Id> = nodes.iter().map(|n| n.id).collect();

    // Pass 1: Topological sort with petgraph
    println!("\n🔹 Pass 1: Topological sort with petgraph (in-memory)");

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
    let (pg_result, pg_time, pg_result_memory) = measure_time_and_memory(|| {
        petgraph_toposort(&pg_graph, None)
            .map(|sorted| {
                sorted
                    .iter()
                    .map(|&idx| pg_graph[idx].clone())
                    .collect::<Vec<_>>()
            })
            .map_err(|_| anyhow::anyhow!("Graph contains a cycle"))
    });

    let pg_result = pg_result?;

    if num_nodes <= 20 {
        println!("  Execution order:");
        for (i, task) in pg_result.iter().enumerate() {
            println!("    {}. {}", i + 1, task);
        }
    } else {
        println!("  Execution order: {} tasks (showing first 10)", pg_result.len());
        for (i, task) in pg_result.iter().take(10).enumerate() {
            println!("    {}. {}", i + 1, task);
        }
        println!("    ...");
    }
    println!("  Execution time: {:.2} ms", pg_time);

    let petgraph_metrics = GraphMetrics {
        algorithm_name: "petgraph".to_string(),
        num_nodes,
        num_edges,
        execution_time_ms: pg_time,
        memory_usage_bytes: pg_result_memory,
    };

    // Pass 2: Topological sort with motlie_db
    println!("\n🔹 Pass 2: Topological sort with motlie_db (persistent)");

    let (reader, name_to_id, _query_handle) = build_graph(db_path, nodes, edges).await?;
    let timeout = Duration::from_secs(30);

    let (motlie_result, motlie_time, motlie_result_memory) =
        measure_time_and_memory_async(|| toposort_motlie(&node_ids, &reader, timeout)).await;
    let motlie_result = motlie_result?;

    if num_nodes <= 20 {
        println!("  Execution order:");
        for (i, task) in motlie_result.iter().enumerate() {
            println!("    {}. {}", i + 1, task);
        }
    } else {
        println!("  Execution order: {} tasks (showing first 10)", motlie_result.len());
        for (i, task) in motlie_result.iter().take(10).enumerate() {
            println!("    {}. {}", i + 1, task);
        }
        println!("    ...");
    }
    println!("  Execution time: {:.2} ms", motlie_time);

    let motlie_metrics = GraphMetrics {
        algorithm_name: "Topological Sort".to_string(),
        num_nodes,
        num_edges,
        execution_time_ms: motlie_time,
        memory_usage_bytes: motlie_result_memory,
    };

    // Verify correctness
    println!("\n✅ Correctness Check:");

    // Verify petgraph result
    let pg_valid = verify_toposort(&pg_result, &name_to_id, &reader, timeout).await?;
    println!("  petgraph ordering: {}", if pg_valid { "✓ Valid" } else { "✗ Invalid" });

    // Verify motlie result
    let motlie_valid = verify_toposort(&motlie_result, &name_to_id, &reader, timeout).await?;
    println!("  motlie_db ordering: {}", if motlie_valid { "✓ Valid" } else { "✗ Invalid" });

    if pg_result == motlie_result {
        println!("\n  ✓ Both implementations produced identical orderings!");
    } else {
        println!("\n  ⚠ Different orderings (both may be valid for DAGs):");
        println!("    Note: Multiple valid topological orderings can exist for a DAG.");

        if pg_valid && motlie_valid {
            println!("    ✓ Both orderings are valid topological sorts!");
        }
    }

    // Print performance comparison
    GraphMetrics::print_comparison(&motlie_metrics, &petgraph_metrics);

    println!("\n✅ Topological sort example completed successfully!");
    Ok(())
}
