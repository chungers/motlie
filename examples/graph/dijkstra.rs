/// Dijkstra's Shortest Path Algorithm Implementation Comparison
///
/// This example demonstrates Dijkstra's algorithm using both the pathfinding crate (in-memory)
/// and motlie_db (persistent graph database), comparing correctness and performance.
///
/// Dijkstra's algorithm finds the shortest path from a source to all other nodes in a
/// weighted graph with non-negative edge weights.
///
/// Common applications:
/// - GPS navigation and routing
/// - Network routing protocols
/// - Airline route planning
/// - Game AI pathfinding
///
/// Usage: dijkstra <db_path>

// Include the common module
#[path = "common.rs"]
mod common;

use anyhow::Result;
use common::{build_graph, measure_time, measure_time_async, parse_scale_factor, GraphEdge, GraphMetrics, GraphNode};
use motlie_db::{Id, OutgoingEdges, QueryRunnable};
use pathfinding::prelude::dijkstra as pathfinding_dijkstra;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::env;
use std::path::Path;
use tokio::time::Duration;

/// Create a weighted graph suitable for shortest path demonstration
/// Represents a connected city road network with distances
/// The scale parameter determines the total number of locations (scale * 8)
fn create_test_graph(scale: usize) -> (Vec<GraphNode>, Vec<GraphEdge>) {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut all_node_ids = Vec::new();

    let base_size = 8; // Base locations per district
    let total_nodes = scale * base_size;

    // Create all nodes
    for i in 0..total_nodes {
        let id = Id::new();
        let node_name = if scale == 1 {
            vec!["Home", "School", "Library", "Park", "Mall", "Hospital", "Station", "Airport"][i].to_string()
        } else {
            format!("L{}", i)
        };
        all_node_ids.push(id);
        nodes.push(GraphNode {
            id,
            name: node_name,
        });
    }

    // Create a connected weighted graph representing a road network
    // Each district (cluster of 8 locations) has internal roads
    // and highways connecting to adjacent districts
    let district_size = base_size;

    for district in 0..scale {
        let district_start = district * district_size;

        // Internal roads within each district (following base pattern)
        let internal_roads = vec![
            (0, 1, 2.0), (0, 2, 5.0), (1, 2, 1.0), (1, 3, 4.0),
            (2, 3, 2.0), (2, 4, 3.0), (3, 5, 3.0), (3, 4, 1.0),
            (4, 5, 2.0), (4, 6, 4.0), (5, 6, 3.0), (6, 7, 5.0),
            (0, 4, 8.0), // Longer direct route
        ];

        for (src_offset, dst_offset, weight) in internal_roads {
            edges.push(GraphEdge {
                source: all_node_ids[district_start + src_offset],
                target: all_node_ids[district_start + dst_offset],
                name: format!("road_d{}_{}_{}", district, src_offset, dst_offset),
                weight: Some(weight),
            });
        }

        // Connect districts with highways (bidirectional for shortest path)
        if district < scale - 1 {
            let next_district_start = (district + 1) * district_size;
            // Highway from current district's last location to next district's first
            edges.push(GraphEdge {
                source: all_node_ids[district_start + 7],
                target: all_node_ids[next_district_start],
                name: format!("highway_{}_{}", district, district + 1),
                weight: Some(3.0),
            });
            // Reverse highway
            edges.push(GraphEdge {
                source: all_node_ids[next_district_start],
                target: all_node_ids[district_start + 7],
                name: format!("highway_{}_{}_rev", district + 1, district),
                weight: Some(3.0),
            });
        }
    }

    (nodes, edges)
}

/// State for priority queue in Dijkstra's algorithm
#[derive(Copy, Clone, PartialEq)]
struct State {
    cost: f64,
    node: Id,
}

impl Eq for State {}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (BinaryHeap is a max-heap)
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Dijkstra's algorithm implementation using motlie_db
async fn dijkstra_motlie(
    start: Id,
    end: Id,
    reader: &motlie_db::Reader,
    timeout: Duration,
) -> Result<Option<(f64, Vec<String>)>> {
    let mut dist: HashMap<Id, f64> = HashMap::new();
    let mut prev: HashMap<Id, Id> = HashMap::new();
    let mut heap = BinaryHeap::new();

    dist.insert(start, 0.0);
    heap.push(State {
        cost: 0.0,
        node: start,
    });

    while let Some(State { cost, node }) = heap.pop() {
        // If we reached the end node, reconstruct path
        if node == end {
            let mut path = Vec::new();
            let mut current = end;

            // Get end node name
            let (name, _) = motlie_db::NodeById::new(current, None)
                .run(reader, timeout)
                .await?;
            path.push(name);

            while let Some(&prev_node) = prev.get(&current) {
                let (name, _) = motlie_db::NodeById::new(prev_node, None)
                    .run(reader, timeout)
                    .await?;
                path.push(name);
                current = prev_node;
            }

            path.reverse();
            return Ok(Some((cost, path)));
        }

        // Skip if we found a better path already
        if let Some(&best_cost) = dist.get(&node) {
            if cost > best_cost {
                continue;
            }
        }

        // Check all neighbors
        let edges = OutgoingEdges::new(node, None).run(reader, timeout).await?;

        for (weight_opt, _src, dst, _name) in edges {
            let weight = weight_opt.unwrap_or(1.0);
            let next_cost = cost + weight;

            // If this path is better than any previous path to dst
            if next_cost < *dist.get(&dst).unwrap_or(&f64::INFINITY) {
                dist.insert(dst, next_cost);
                prev.insert(dst, node);
                heap.push(State {
                    cost: next_cost,
                    node: dst,
                });
            }
        }
    }

    Ok(None) // No path found
}

/// Get all shortest paths from a source node using motlie_db
async fn dijkstra_all_paths(
    start: Id,
    reader: &motlie_db::Reader,
    timeout: Duration,
) -> Result<HashMap<String, f64>> {
    let mut dist: HashMap<Id, f64> = HashMap::new();
    let mut heap = BinaryHeap::new();
    let mut name_map: HashMap<Id, String> = HashMap::new();

    dist.insert(start, 0.0);
    heap.push(State {
        cost: 0.0,
        node: start,
    });

    // Get start node name
    let (name, _) = motlie_db::NodeById::new(start, None)
        .run(reader, timeout)
        .await?;
    name_map.insert(start, name);

    while let Some(State { cost, node }) = heap.pop() {
        // Skip if we found a better path already
        if let Some(&best_cost) = dist.get(&node) {
            if cost > best_cost {
                continue;
            }
        }

        // Check all neighbors
        let edges = OutgoingEdges::new(node, None).run(reader, timeout).await?;

        for (weight_opt, _src, dst, _name) in edges {
            // Get destination node name if not already cached
            if !name_map.contains_key(&dst) {
                let (dst_name, _) = motlie_db::NodeById::new(dst, None)
                    .run(reader, timeout)
                    .await?;
                name_map.insert(dst, dst_name);
            }

            let weight = weight_opt.unwrap_or(1.0);
            let next_cost = cost + weight;

            // If this path is better than any previous path to dst
            if next_cost < *dist.get(&dst).unwrap_or(&f64::INFINITY) {
                dist.insert(dst, next_cost);
                heap.push(State {
                    cost: next_cost,
                    node: dst,
                });
            }
        }
    }

    // Convert Id -> distance map to name -> distance map
    Ok(dist
        .into_iter()
        .filter_map(|(id, cost)| name_map.get(&id).map(|name| (name.clone(), cost)))
        .collect())
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
        eprintln!("  {} /tmp/dijkstra_test_db 1     # Base graph (8 locations, 13 roads)", args[0]);
        eprintln!("  {} /tmp/dijkstra_test_db 100   # 100x larger (800 locations, 1300 roads)", args[0]);
        std::process::exit(1);
    }

    let db_path = Path::new(&args[1]);
    let scale = parse_scale_factor(&args[2])?;

    println!("🗺️  Dijkstra's Shortest Path Algorithm Comparison");
    println!("{:=<80}", "");
    println!("\n📏 Scale factor: {}x", scale);

    // Generate test graph
    println!("\n📊 Generating test graph (city road network)...");
    let start_gen = std::time::Instant::now();
    let (nodes, edges) = create_test_graph(scale);
    let gen_time = start_gen.elapsed();
    let num_nodes = nodes.len();
    let num_edges = edges.len();

    println!("  Locations: {}", num_nodes);
    println!("  Roads: {}", num_edges);
    println!("  Generation time: {:.2} ms", gen_time.as_secs_f64() * 1000.0);

    let name_to_id: HashMap<String, Id> = nodes
        .iter()
        .map(|n| (n.name.clone(), n.id))
        .collect();

    // For small graph, use Home->Airport. For large, use first and last node
    let start_name = if scale == 1 { "Home".to_string() } else { nodes[0].name.clone() };
    let end_name = if scale == 1 { "Airport".to_string() } else { nodes[nodes.len() - 1].name.clone() };

    println!("\n  Finding shortest path from {} to {}", start_name, end_name);

    // Pass 1: Dijkstra with pathfinding crate
    println!("\n🔹 Pass 1: Dijkstra with pathfinding crate (in-memory)");

    // Build adjacency list for pathfinding crate
    let mut adjacency: HashMap<String, Vec<(String, u32)>> = HashMap::new();

    for node in &nodes {
        adjacency.insert(node.name.clone(), Vec::new());
    }

    for edge in &edges {
        let src_name = nodes.iter().find(|n| n.id == edge.source).unwrap().name.clone();
        let dst_name = nodes.iter().find(|n| n.id == edge.target).unwrap().name.clone();
        let weight = (edge.weight.unwrap_or(1.0) * 10.0) as u32; // Convert to integer

        adjacency
            .get_mut(&src_name)
            .unwrap()
            .push((dst_name, weight));
    }

    let (pf_result, pf_time) = measure_time(|| {
        pathfinding_dijkstra(
            &start_name,
            |n| adjacency[n].iter().cloned(),
            |n| n == &end_name,
        )
    });

    match &pf_result {
        Some((path, cost)) => {
            if num_nodes <= 20 {
                println!("  Path: {:?}", path);
            } else {
                println!("  Path: {} nodes", path.len());
            }
            println!("  Cost: {:.1} km", *cost as f64 / 10.0);
            println!("  Execution time: {:.2} ms", pf_time);
        }
        None => {
            println!("  No path found!");
        }
    }

    let pathfinding_metrics = GraphMetrics {
        algorithm_name: "pathfinding".to_string(),
        num_nodes,
        num_edges,
        execution_time_ms: pf_time,
        memory_usage_bytes: None,
    };

    // Pass 2: Dijkstra with motlie_db
    println!("\n🔹 Pass 2: Dijkstra with motlie_db (persistent)");

    let (reader, name_to_id, _query_handle) = build_graph(db_path, nodes, edges).await?;
    let start_id = name_to_id[&start_name];
    let end_id = name_to_id[&end_name];
    let timeout = Duration::from_secs(60); // Longer timeout for large graphs

    let (motlie_result, motlie_time) =
        measure_time_async(|| dijkstra_motlie(start_id, end_id, &reader, timeout)).await;
    let motlie_result = motlie_result?;

    match &motlie_result {
        Some((cost, path)) => {
            if num_nodes <= 20 {
                println!("  Path: {:?}", path);
            } else {
                println!("  Path: {} nodes", path.len());
            }
            println!("  Cost: {:.1} km", cost);
            println!("  Execution time: {:.2} ms", motlie_time);
        }
        None => {
            println!("  No path found!");
        }
    }

    let motlie_metrics = GraphMetrics {
        algorithm_name: "Dijkstra".to_string(),
        num_nodes,
        num_edges,
        execution_time_ms: motlie_time,
        memory_usage_bytes: None,
    };

    // Verify correctness
    println!("\n✅ Correctness Check:");
    match (&pf_result, &motlie_result) {
        (Some((pf_path, pf_cost)), Some((motlie_cost, motlie_path))) => {
            let pf_cost_f64 = *pf_cost as f64 / 10.0;
            let cost_diff = (pf_cost_f64 - motlie_cost).abs();

            if cost_diff < 0.1 {
                println!("  ✓ Path costs match: {:.1} km", motlie_cost);
            } else {
                println!("  ✗ Path costs differ:");
                println!("    pathfinding: {:.1} km", pf_cost_f64);
                println!("    motlie_db: {:.1} km", motlie_cost);
            }

            if pf_path == motlie_path {
                println!("  ✓ Paths are identical!");
            } else {
                println!("  ⚠ Different paths found (may both be valid if costs are equal):");
                println!("    pathfinding: {:?}", pf_path);
                println!("    motlie_db: {:?}", motlie_path);
            }
        }
        (None, None) => {
            println!("  ✓ Both correctly found no path");
        }
        _ => {
            println!("  ✗ One implementation found a path, the other didn't!");
        }
    }

    // Demonstrate all shortest paths from Home
    println!("\n🔹 All shortest paths from {} (using motlie_db):", start_name);
    let all_paths = dijkstra_all_paths(start_id, &reader, timeout).await?;

    let mut sorted_paths: Vec<_> = all_paths.iter().collect();
    sorted_paths.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());

    for (dest, cost) in sorted_paths {
        println!("  {} → {}: {:.1} km", start_name, dest, cost);
    }

    // Print performance comparison
    GraphMetrics::print_comparison(&motlie_metrics, &pathfinding_metrics);

    println!("\n✅ Dijkstra's algorithm example completed successfully!");
    Ok(())
}
