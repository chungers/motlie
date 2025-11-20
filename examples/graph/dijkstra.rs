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
use common::{build_graph, compute_hash_f64, compute_hash_f64_motlie, measure_time_and_memory, measure_time_and_memory_async, parse_scale_factor, GraphEdge, GraphMetrics, GraphNode, Implementation};
use motlie_db::{Id, OutgoingEdges, QueryRunnable};
use pathfinding::prelude::dijkstra as pathfinding_dijkstra;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
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

    let base_names = vec!["Hub", "N", "S", "E", "W", "NE", "NW", "SE", "SW", "Center"];
    let total_nodes = scale * base_names.len();

    // Create all nodes
    for i in 0..total_nodes {
        let id = Id::new();
        let node_name = if scale == 1 {
            base_names[i % base_names.len()].to_string()
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
    // Each district (cluster of 10 locations) has internal roads
    // and highways connecting to adjacent districts
    let district_size = base_names.len();

    for district in 0..scale {
        let district_start = district * district_size;

        // Internal roads within each district (following hub-and-spoke pattern)
        let internal_roads = vec![
            (0, 1, 2.0), (0, 2, 2.0), (0, 3, 2.0), (0, 4, 2.0), // Hub to cardinal directions
            (0, 5, 3.0), (0, 6, 3.0), (0, 7, 3.0), (0, 8, 3.0), // Hub to diagonal directions
            (1, 5, 1.5), (1, 6, 1.5), // North to NE, NW
            (2, 7, 1.5), (2, 8, 1.5), // South to SE, SW
            (3, 5, 1.5), (3, 7, 1.5), // East to NE, SE
            (4, 6, 1.5), (4, 8, 1.5), // West to NW, SW
            (5, 9, 2.0), (6, 9, 2.0), (7, 9, 2.0), (8, 9, 2.0), // Diagonals to Center
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
            // Highway from current district's Center to next district's Hub
            edges.push(GraphEdge {
                source: all_node_ids[district_start + 9],
                target: all_node_ids[next_district_start],
                name: format!("highway_{}_{}", district, district + 1),
                weight: Some(3.0),
            });
            // Reverse highway
            edges.push(GraphEdge {
                source: all_node_ids[next_district_start],
                target: all_node_ids[district_start + 9],
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
        eprintln!("  {} reference /tmp/dijkstra_test_db 1      # pathfinding, 8 nodes", args[0]);
        eprintln!("  {} motlie_db /tmp/dijkstra_test_db 1      # motlie_db, 8 nodes", args[0]);
        eprintln!("  {} reference /tmp/dijkstra_test_db 1000   # pathfinding, 8000 nodes", args[0]);
        eprintln!("  {} motlie_db /tmp/dijkstra_test_db 1000   # motlie_db, 8000 nodes", args[0]);
        std::process::exit(1);
    }

    let implementation = Implementation::from_str(&args[1])?;
    let db_path = Path::new(&args[2]);
    let scale = parse_scale_factor(&args[3])?;

    // Generate test graph
    let (nodes, edges) = create_test_graph(scale);
    let num_nodes = nodes.len();
    let num_edges = edges.len();

    let name_to_id: HashMap<String, Id> = nodes
        .iter()
        .map(|n| (n.name.clone(), n.id))
        .collect();

    // For small graph, use Hub->Center. For large, use first and last node
    let start_name = if scale == 1 { "Hub".to_string() } else { nodes[0].name.clone() };
    let end_name = if scale == 1 { "Center".to_string() } else { nodes[nodes.len() - 1].name.clone() };

    match implementation {
        Implementation::Reference => {
            // Run Dijkstra with pathfinding crate (reference implementation)
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

            let (result, time_ms, memory) = measure_time_and_memory(|| {
                pathfinding_dijkstra(
                    &start_name,
                    |n| adjacency[n].iter().cloned(),
                    |n| n == &end_name,
                )
            });
            let result_hash = Some(compute_hash_f64(&result));

            let metrics = GraphMetrics {
                algorithm_name: "Dijkstra".to_string(),
                implementation: Implementation::Reference,
                scale,
                num_nodes,
                num_edges,
                execution_time_ms: time_ms,
                memory_usage_bytes: memory,
                result_hash,
            };

            metrics.print_csv();
        }
        Implementation::MotlieDb => {
            // Run Dijkstra with motlie_db
            let (reader, name_to_id, _query_handle) = build_graph(db_path, nodes, edges).await?;
            let start_id = name_to_id[&start_name];
            let end_id = name_to_id[&end_name];
            let timeout = Duration::from_secs(60); // Longer timeout for large graphs

            let (result, time_ms, memory) = measure_time_and_memory_async(|| dijkstra_motlie(start_id, end_id, &reader, timeout)).await;
            let result = result?;
            let result_hash = Some(compute_hash_f64_motlie(&result));

            let metrics = GraphMetrics {
                algorithm_name: "Dijkstra".to_string(),
                implementation: Implementation::MotlieDb,
                scale,
                num_nodes,
                num_edges,
                execution_time_ms: time_ms,
                memory_usage_bytes: memory,
                result_hash,
            };

            metrics.print_csv();
        }
    }

    Ok(())
}
