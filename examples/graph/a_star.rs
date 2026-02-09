/// A* Shortest Path Algorithm Implementation Comparison
///
/// This example demonstrates the A* algorithm using both the pathfinding crate (in-memory)
/// and motlie_db (persistent graph database), comparing correctness and performance.
///
/// A* is an informed search algorithm that uses a heuristic function to guide the search,
/// making it more efficient than Dijkstra's algorithm when a good heuristic is available.
///
/// Common applications:
/// - Game AI pathfinding (NPCs, units)
/// - Robot navigation and motion planning
/// - GPS navigation with estimated distances
/// - Puzzle solving (e.g., sliding puzzle, Rubik's cube)
///
/// The algorithm combines:
/// - g(n): Actual cost from start to current node (like Dijkstra)
/// - h(n): Heuristic estimate from current node to goal
/// - f(n) = g(n) + h(n): Total estimated cost through this node
///
/// # Unified API Usage
///
/// This example uses the **unified motlie_db API** (porcelain layer):
/// - Storage: `motlie_db::{Storage, StorageConfig, ReadWriteHandles}`
/// - Queries: `motlie_db::query::{OutgoingEdges, Runnable}`
/// - Reader: `motlie_db::reader::Reader`
///
/// The unified API provides type-safe handles and a consistent interface
/// for both graph (RocksDB) and fulltext (Tantivy) operations.
///
/// Usage: a_star <implementation> <db_path> <scale_factor>

// Include the common module
#[path = "common.rs"]
mod common;

use anyhow::Result;
use common::{
    build_graph, compute_hash_f64, compute_hash_f64_motlie, get_disk_metrics,
    measure_time_and_memory, measure_time_and_memory_async, parse_scale_factor, GraphEdge,
    GraphMetrics, GraphNode, Implementation,
};
use motlie_db::query::{OutgoingEdges, Runnable as QueryRunnable};
use motlie_db::Id;
use pathfinding::prelude::astar as pathfinding_astar;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::env;
use std::path::Path;
use tokio::time::Duration;

/// Grid position for heuristic calculation
/// In our test graph, nodes are arranged in a 2D grid pattern
#[derive(Clone, Copy, Debug)]
struct Position {
    x: i32,
    y: i32,
}

impl Position {
    /// Manhattan distance heuristic (admissible for grid movement)
    fn manhattan_distance(&self, other: &Position) -> f64 {
        ((self.x - other.x).abs() + (self.y - other.y).abs()) as f64
    }

    /// Euclidean distance heuristic (admissible, tighter than Manhattan for diagonal movement)
    fn euclidean_distance(&self, other: &Position) -> f64 {
        let dx = (self.x - other.x) as f64;
        let dy = (self.y - other.y) as f64;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Create a weighted graph suitable for A* demonstration
/// Represents a 2D grid-like map with varying terrain costs
/// The scale parameter determines the grid size (scale * 5 x scale * 5)
fn create_test_graph(scale: usize) -> (Vec<GraphNode>, Vec<GraphEdge>, HashMap<Id, Position>) {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut positions: HashMap<Id, Position> = HashMap::new();

    let grid_size = scale * 5; // Grid dimension
    let mut grid_ids: Vec<Vec<Id>> = Vec::new();

    // Create all nodes in a grid pattern
    for y in 0..grid_size {
        let mut row = Vec::new();
        for x in 0..grid_size {
            let id = Id::new();
            let node_name = format!("Node_{}_{}", x, y);

            positions.insert(
                id,
                Position {
                    x: x as i32,
                    y: y as i32,
                },
            );

            nodes.push(GraphNode {
                id,
                name: node_name,
                summary: None,
            });
            row.push(id);
        }
        grid_ids.push(row);
    }

    // Create edges with varying weights (simulating terrain)
    // Connect each node to its 4 cardinal neighbors and 4 diagonal neighbors
    let directions = [
        (0, 1, 1.0),   // Right
        (1, 0, 1.0),   // Down
        (0, -1, 1.0),  // Left
        (-1, 0, 1.0),  // Up
        (1, 1, 1.4),   // Diagonal: down-right (sqrt(2) ≈ 1.414)
        (1, -1, 1.4),  // Diagonal: down-left
        (-1, 1, 1.4),  // Diagonal: up-right
        (-1, -1, 1.4), // Diagonal: up-left
    ];

    for y in 0..grid_size {
        for x in 0..grid_size {
            for (dy, dx, base_weight) in directions.iter() {
                let ny = y as i32 + dy;
                let nx = x as i32 + dx;

                if ny >= 0 && ny < grid_size as i32 && nx >= 0 && nx < grid_size as i32 {
                    let src_id = grid_ids[y][x];
                    let dst_id = grid_ids[ny as usize][nx as usize];

                    // Add some terrain variation based on position
                    // This creates "hills" that the algorithm should route around
                    let terrain_factor = if (x + y) % 7 == 0 { 2.0 } else { 1.0 };
                    let weight = base_weight * terrain_factor;

                    edges.push(GraphEdge {
                        source: src_id,
                        target: dst_id,
                        name: format!("edge_{}_{}_{}", x, y, directions.iter().position(|d| d == &(*dy, *dx, *base_weight)).unwrap()),
                        weight: Some(weight),
                        summary: None,
                    });
                }
            }
        }
    }

    (nodes, edges, positions)
}

/// State for priority queue in A* algorithm
#[derive(Clone)]
struct AStarState {
    f_score: f64, // f(n) = g(n) + h(n)
    g_score: f64, // Actual cost from start
    node: Id,
}

impl Eq for AStarState {}

impl PartialEq for AStarState {
    fn eq(&self, other: &Self) -> bool {
        self.f_score == other.f_score && self.node == other.node
    }
}

impl Ord for AStarState {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (BinaryHeap is a max-heap)
        other
            .f_score
            .partial_cmp(&self.f_score)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for AStarState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A* algorithm implementation using motlie_db
async fn astar_motlie(
    start: Id,
    end: Id,
    positions: &HashMap<Id, Position>,
    reader: &motlie_db::graph::reader::Reader,
    timeout: Duration,
) -> Result<Option<(f64, Vec<String>)>> {
    let end_pos = positions.get(&end).ok_or_else(|| anyhow::anyhow!("End node not in positions"))?;

    let mut g_scores: HashMap<Id, f64> = HashMap::new();
    let mut prev: HashMap<Id, Id> = HashMap::new();
    let mut heap = BinaryHeap::new();

    // Initialize start node
    let start_pos = positions.get(&start).ok_or_else(|| anyhow::anyhow!("Start node not in positions"))?;
    let h_start = start_pos.euclidean_distance(end_pos);

    g_scores.insert(start, 0.0);
    heap.push(AStarState {
        f_score: h_start,
        g_score: 0.0,
        node: start,
    });

    while let Some(AStarState {
        f_score: _,
        g_score,
        node,
    }) = heap.pop()
    {
        // If we reached the end node, reconstruct path
        if node == end {
            let mut path = Vec::new();
            let mut current = end;

            // Get end node name
            let (name, _, _) = motlie_db::graph::query::NodeById::new(current, None)
                .run(reader, timeout)
                .await?;
            path.push(name);

            while let Some(&prev_node) = prev.get(&current) {
                let (name, _, _) = motlie_db::graph::query::NodeById::new(prev_node, None)
                    .run(reader, timeout)
                    .await?;
                path.push(name);
                current = prev_node;
            }

            path.reverse();
            return Ok(Some((g_score, path)));
        }

        // Skip if we've found a better path already
        if let Some(&best_g) = g_scores.get(&node) {
            if g_score > best_g {
                continue;
            }
        }

        // Check all neighbors
        let edges = OutgoingEdges::new(node, None).run(reader, timeout).await?;

        for (weight_opt, _src, dst, _name, _version) in edges {
            let weight = weight_opt.unwrap_or(1.0);
            let tentative_g = g_score + weight;

            // If this path is better than any previous path to dst
            if tentative_g < *g_scores.get(&dst).unwrap_or(&f64::INFINITY) {
                g_scores.insert(dst, tentative_g);
                prev.insert(dst, node);

                // Calculate heuristic for dst
                let dst_pos = positions.get(&dst).unwrap();
                let h = dst_pos.euclidean_distance(end_pos);
                let f = tentative_g + h;

                heap.push(AStarState {
                    f_score: f,
                    g_score: tentative_g,
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
        eprintln!(
            "  {} reference /tmp/astar_test_db 1      # pathfinding, 25 nodes (5x5 grid)",
            args[0]
        );
        eprintln!(
            "  {} motlie_db /tmp/astar_test_db 1      # motlie_db, 25 nodes (5x5 grid)",
            args[0]
        );
        eprintln!(
            "  {} reference /tmp/astar_test_db 10     # pathfinding, 2500 nodes (50x50 grid)",
            args[0]
        );
        eprintln!(
            "  {} motlie_db /tmp/astar_test_db 10     # motlie_db, 2500 nodes (50x50 grid)",
            args[0]
        );
        std::process::exit(1);
    }

    let implementation = Implementation::from_str(&args[1])?;
    let db_path = Path::new(&args[2]);
    let scale = parse_scale_factor(&args[3])?;

    // Generate test graph
    let (nodes, edges, positions) = create_test_graph(scale);
    let num_nodes = nodes.len();
    let num_edges = edges.len();

    // Build name-to-position map for reference implementation
    let name_to_id: HashMap<String, Id> = nodes.iter().map(|n| (n.name.clone(), n.id)).collect();
    let name_to_pos: HashMap<String, Position> = nodes
        .iter()
        .map(|n| (n.name.clone(), *positions.get(&n.id).unwrap()))
        .collect();

    // Start from top-left corner, end at bottom-right corner
    let grid_size = scale * 5;
    let start_name = "Node_0_0".to_string();
    let end_name = format!("Node_{}_{}", grid_size - 1, grid_size - 1);

    match implementation {
        Implementation::Reference => {
            // Run A* with pathfinding crate (reference implementation)
            // Build adjacency list for pathfinding crate
            let mut adjacency: HashMap<String, Vec<(String, u32)>> = HashMap::new();

            for node in &nodes {
                adjacency.insert(node.name.clone(), Vec::new());
            }

            for edge in &edges {
                let src_name = nodes
                    .iter()
                    .find(|n| n.id == edge.source)
                    .unwrap()
                    .name
                    .clone();
                let dst_name = nodes
                    .iter()
                    .find(|n| n.id == edge.target)
                    .unwrap()
                    .name
                    .clone();
                let weight = (edge.weight.unwrap_or(1.0) * 10.0) as u32; // Convert to integer

                adjacency.get_mut(&src_name).unwrap().push((dst_name, weight));
            }

            let end_pos = name_to_pos[&end_name];

            let (result, time_ms, memory) = measure_time_and_memory(|| {
                pathfinding_astar(
                    &start_name,
                    |n| adjacency[n].iter().cloned(),
                    |n| {
                        // Heuristic: Euclidean distance (scaled to match weight scale)
                        let pos = name_to_pos[n];
                        (pos.euclidean_distance(&end_pos) * 10.0) as u32
                    },
                    |n| n == &end_name,
                )
            });
            let result_hash = Some(compute_hash_f64(&result));

            let metrics = GraphMetrics {
                algorithm_name: "A*".to_string(),
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
            // Run A* with motlie_db
            let (reader, name_to_id_built, _handles) =
                build_graph(db_path, nodes, edges).await?;
            let start_id = name_to_id_built[&start_name];
            let end_id = name_to_id_built[&end_name];

            // Rebuild positions map with IDs from build_graph
            let positions_rebuilt: HashMap<Id, Position> = name_to_id_built
                .iter()
                .map(|(name, &id)| (id, name_to_pos[name]))
                .collect();

            let timeout = Duration::from_secs(120); // Longer timeout for large graphs

            let (result, time_ms, memory) = measure_time_and_memory_async(|| {
                astar_motlie(start_id, end_id, &positions_rebuilt, reader.graph(), timeout)
            })
            .await;
            let result = result?;
            let result_hash = Some(compute_hash_f64_motlie(&result));

            // Measure disk usage after algorithm completes
            let (disk_files, disk_size) = get_disk_metrics(db_path).unwrap_or((0, 0));

            let metrics = GraphMetrics {
                algorithm_name: "A*".to_string(),
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
