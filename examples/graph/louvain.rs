/// Louvain Community Detection Algorithm Implementation Comparison
///
/// This example demonstrates the Louvain algorithm for community detection using both
/// an in-memory reference implementation and motlie_db (persistent graph database).
///
/// The Louvain algorithm is a greedy optimization method for detecting communities
/// in large networks. It maximizes modularity, a measure of the density of links
/// inside communities compared to links between communities.
///
/// Common applications:
/// - Social network analysis (finding groups of friends)
/// - Biological network analysis (protein interaction networks)
/// - Citation network clustering
/// - Customer segmentation
/// - Fraud detection networks
///
/// The algorithm works in two phases repeated iteratively:
/// 1. Local optimization: Move nodes between communities to maximize modularity gain
/// 2. Network aggregation: Build a new network of communities
///
/// Usage: louvain <implementation> <db_path> <scale_factor>

// Include the common module
#[path = "common.rs"]
mod common;

use anyhow::Result;
use common::{
    build_graph, compute_hash, get_disk_metrics, measure_time_and_memory,
    measure_time_and_memory_async, parse_scale_factor, GraphEdge, GraphMetrics, GraphNode,
    Implementation,
};
use motlie_db::{Id, OutgoingEdges, QueryRunnable};
use std::collections::{HashMap, HashSet};
use std::env;
use std::path::Path;
use tokio::time::Duration;

/// Create a test graph suitable for community detection
/// Creates multiple distinct communities with dense internal connections
/// and sparse inter-community connections
/// The scale parameter determines the number of communities (scale * 3)
fn create_test_graph(scale: usize) -> (Vec<GraphNode>, Vec<GraphEdge>) {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut all_node_ids = Vec::new();

    let community_size = 8; // Nodes per community
    let num_communities = scale * 3;
    let total_nodes = num_communities * community_size;

    // Create all nodes
    for i in 0..total_nodes {
        let id = Id::new();
        let community = i / community_size;
        let node_in_community = i % community_size;
        let node_name = format!("C{}_N{}", community, node_in_community);
        all_node_ids.push(id);
        nodes.push(GraphNode {
            id,
            name: node_name,
        });
    }

    // Create dense intra-community edges
    // Each community forms a near-complete subgraph
    for community in 0..num_communities {
        let start = community * community_size;

        // Connect each pair of nodes within the community (undirected = 2 directed edges)
        for i in 0..community_size {
            for j in (i + 1)..community_size {
                let src_idx = start + i;
                let dst_idx = start + j;

                // Forward edge
                edges.push(GraphEdge {
                    source: all_node_ids[src_idx],
                    target: all_node_ids[dst_idx],
                    name: format!("intra_c{}_{}_{}", community, i, j),
                    weight: Some(1.0),
                });

                // Backward edge (undirected graph)
                edges.push(GraphEdge {
                    source: all_node_ids[dst_idx],
                    target: all_node_ids[src_idx],
                    name: format!("intra_c{}_{}_{}_rev", community, j, i),
                    weight: Some(1.0),
                });
            }
        }
    }

    // Create sparse inter-community edges
    // Connect adjacent communities with just 1-2 bridge edges
    for community in 0..(num_communities - 1) {
        let current_start = community * community_size;
        let next_start = (community + 1) * community_size;

        // Bridge node: last node of current community to first node of next
        edges.push(GraphEdge {
            source: all_node_ids[current_start + community_size - 1],
            target: all_node_ids[next_start],
            name: format!("bridge_{}_{}", community, community + 1),
            weight: Some(1.0),
        });

        // Reverse bridge
        edges.push(GraphEdge {
            source: all_node_ids[next_start],
            target: all_node_ids[current_start + community_size - 1],
            name: format!("bridge_{}_{}_rev", community + 1, community),
            weight: Some(1.0),
        });

        // Second bridge (middle nodes) for larger graphs
        if community_size > 4 {
            let mid = community_size / 2;
            edges.push(GraphEdge {
                source: all_node_ids[current_start + mid],
                target: all_node_ids[next_start + mid],
                name: format!("bridge2_{}_{}", community, community + 1),
                weight: Some(1.0),
            });

            edges.push(GraphEdge {
                source: all_node_ids[next_start + mid],
                target: all_node_ids[current_start + mid],
                name: format!("bridge2_{}_{}_rev", community + 1, community),
                weight: Some(1.0),
            });
        }
    }

    // Connect first and last communities (circular) for larger graphs
    if num_communities > 2 {
        let first_start = 0;
        let last_start = (num_communities - 1) * community_size;

        edges.push(GraphEdge {
            source: all_node_ids[first_start],
            target: all_node_ids[last_start + community_size - 1],
            name: format!("bridge_circular_0_{}", num_communities - 1),
            weight: Some(1.0),
        });

        edges.push(GraphEdge {
            source: all_node_ids[last_start + community_size - 1],
            target: all_node_ids[first_start],
            name: format!("bridge_circular_{}_0_rev", num_communities - 1),
            weight: Some(1.0),
        });
    }

    (nodes, edges)
}

/// Community assignment result
#[derive(Clone, Debug)]
struct CommunityResult {
    /// Maps node name to community ID
    assignments: HashMap<String, usize>,
    /// Final modularity score
    modularity: f64,
    /// Number of communities found
    num_communities: usize,
}

impl std::hash::Hash for CommunityResult {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash the sorted community counts for consistency
        let mut community_counts: Vec<usize> = {
            let mut counts: HashMap<usize, usize> = HashMap::new();
            for &c in self.assignments.values() {
                *counts.entry(c).or_insert(0) += 1;
            }
            counts.values().cloned().collect()
        };
        community_counts.sort();
        community_counts.hash(state);
        self.num_communities.hash(state);
        // Hash modularity as bits to avoid floating point issues
        ((self.modularity * 10000.0) as i64).hash(state);
    }
}

/// Calculate modularity of a community assignment
/// Q = (1/2m) * Σ[Aij - (ki*kj)/(2m)] * δ(ci, cj)
/// where:
/// - m is the total number of edges
/// - Aij is the adjacency matrix
/// - ki is the degree of node i
/// - δ(ci, cj) is 1 if nodes i and j are in the same community
fn calculate_modularity_reference(
    adjacency: &HashMap<String, Vec<(String, f64)>>,
    communities: &HashMap<String, usize>,
) -> f64 {
    let mut total_weight = 0.0;
    let mut degrees: HashMap<String, f64> = HashMap::new();

    // Calculate total weight and degrees
    for (node, neighbors) in adjacency {
        let degree: f64 = neighbors.iter().map(|(_, w)| w).sum();
        degrees.insert(node.clone(), degree);
        total_weight += degree;
    }

    // total_weight counts each edge twice (for undirected), so m = total_weight / 2
    let m = total_weight / 2.0;
    if m == 0.0 {
        return 0.0;
    }

    let mut modularity = 0.0;

    for (node_i, neighbors) in adjacency {
        let ki = degrees[node_i];
        let ci = communities[node_i];

        for (node_j, weight) in neighbors {
            let kj = degrees[node_j];
            let cj = communities[node_j];

            if ci == cj {
                // Nodes in same community
                modularity += weight - (ki * kj) / (2.0 * m);
            }
        }
    }

    modularity / (2.0 * m)
}

/// Calculate modularity gain from moving a node to a new community
fn modularity_gain(
    node: &str,
    target_community: usize,
    adjacency: &HashMap<String, Vec<(String, f64)>>,
    communities: &HashMap<String, usize>,
    degrees: &HashMap<String, f64>,
    community_weights: &HashMap<usize, f64>,
    m: f64,
) -> f64 {
    let ki = degrees[node];
    let current_community = communities[node];

    if current_community == target_community {
        return 0.0;
    }

    // Sum of weights of edges from node to target community
    let mut ki_in = 0.0;
    // Sum of weights of edges from node to current community
    let mut ki_out = 0.0;

    for (neighbor, weight) in &adjacency[node] {
        let neighbor_community = communities[neighbor];
        if neighbor_community == target_community {
            ki_in += weight;
        }
        if neighbor_community == current_community && neighbor != node {
            ki_out += weight;
        }
    }

    // Total weight of edges in target community
    let sigma_in = community_weights.get(&target_community).unwrap_or(&0.0);
    // Total weight of edges in current community (excluding node's contribution)
    let sigma_out = community_weights.get(&current_community).unwrap_or(&0.0) - ki;

    // Modularity gain formula
    let gain_to = ki_in - (sigma_in * ki) / (2.0 * m);
    let loss_from = ki_out - (sigma_out * ki) / (2.0 * m);

    (gain_to - loss_from) / m
}

/// Reference implementation of Louvain algorithm
fn louvain_reference(adjacency: &HashMap<String, Vec<(String, f64)>>) -> CommunityResult {
    let nodes: Vec<String> = adjacency.keys().cloned().collect();

    // Initialize: each node in its own community
    let mut communities: HashMap<String, usize> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        communities.insert(node.clone(), i);
    }

    // Calculate degrees and total weight
    let mut degrees: HashMap<String, f64> = HashMap::new();
    let mut total_weight = 0.0;

    for (node, neighbors) in adjacency {
        let degree: f64 = neighbors.iter().map(|(_, w)| w).sum();
        degrees.insert(node.clone(), degree);
        total_weight += degree;
    }

    let m = total_weight / 2.0;
    if m == 0.0 {
        return CommunityResult {
            assignments: communities,
            modularity: 0.0,
            num_communities: nodes.len(),
        };
    }

    // Phase 1: Local optimization
    let max_iterations = 100;
    for _iteration in 0..max_iterations {
        let mut improved = false;

        // Calculate community weights
        let mut community_weights: HashMap<usize, f64> = HashMap::new();
        for (node, &degree) in &degrees {
            let c = communities[node];
            *community_weights.entry(c).or_insert(0.0) += degree;
        }

        // Try moving each node
        for node in &nodes {
            let current_community = communities[node];

            // Find neighboring communities
            let mut neighbor_communities: HashSet<usize> = HashSet::new();
            neighbor_communities.insert(current_community); // Can stay in current

            for (neighbor, _) in &adjacency[node] {
                neighbor_communities.insert(communities[neighbor]);
            }

            // Find best community
            let mut best_gain = 0.0;
            let mut best_community = current_community;

            for &target in &neighbor_communities {
                let gain = modularity_gain(
                    node,
                    target,
                    adjacency,
                    &communities,
                    &degrees,
                    &community_weights,
                    m,
                );

                if gain > best_gain {
                    best_gain = gain;
                    best_community = target;
                }
            }

            // Move node if beneficial
            if best_community != current_community && best_gain > 1e-10 {
                // Update community weights
                let ki = degrees[node];
                *community_weights.get_mut(&current_community).unwrap() -= ki;
                *community_weights.entry(best_community).or_insert(0.0) += ki;

                communities.insert(node.clone(), best_community);
                improved = true;
            }
        }

        if !improved {
            break;
        }
    }

    // Renumber communities to be consecutive
    let unique_communities: HashSet<usize> = communities.values().cloned().collect();
    let mut community_map: HashMap<usize, usize> = HashMap::new();
    for (new_id, old_id) in unique_communities.iter().enumerate() {
        community_map.insert(*old_id, new_id);
    }

    let mut final_communities: HashMap<String, usize> = HashMap::new();
    for (node, old_c) in &communities {
        final_communities.insert(node.clone(), community_map[old_c]);
    }

    let modularity = calculate_modularity_reference(adjacency, &final_communities);
    let num_communities = unique_communities.len();

    CommunityResult {
        assignments: final_communities,
        modularity,
        num_communities,
    }
}

/// Louvain algorithm implementation using motlie_db
async fn louvain_motlie(
    all_nodes: &[Id],
    id_to_name: &HashMap<Id, String>,
    reader: &motlie_db::Reader,
    timeout: Duration,
) -> Result<CommunityResult> {
    // Initialize: each node in its own community
    let mut communities: HashMap<Id, usize> = HashMap::new();
    for (i, &node_id) in all_nodes.iter().enumerate() {
        communities.insert(node_id, i);
    }

    // Calculate degrees and total weight
    let mut degrees: HashMap<Id, f64> = HashMap::new();
    let mut total_weight = 0.0;

    // Build adjacency from database
    let mut adjacency: HashMap<Id, Vec<(Id, f64)>> = HashMap::new();

    for &node_id in all_nodes {
        let outgoing = OutgoingEdges::new(node_id, None)
            .run(reader, timeout)
            .await?;

        let mut neighbors = Vec::new();
        let mut degree = 0.0;

        for (weight_opt, _src, dst, _name) in outgoing {
            let weight = weight_opt.unwrap_or(1.0);
            neighbors.push((dst, weight));
            degree += weight;
        }

        adjacency.insert(node_id, neighbors);
        degrees.insert(node_id, degree);
        total_weight += degree;
    }

    let m = total_weight / 2.0;
    if m == 0.0 {
        let assignments: HashMap<String, usize> = all_nodes
            .iter()
            .enumerate()
            .map(|(i, &id)| (id_to_name[&id].clone(), i))
            .collect();

        return Ok(CommunityResult {
            assignments,
            modularity: 0.0,
            num_communities: all_nodes.len(),
        });
    }

    // Phase 1: Local optimization
    let max_iterations = 100;
    for _iteration in 0..max_iterations {
        let mut improved = false;

        // Calculate community weights
        let mut community_weights: HashMap<usize, f64> = HashMap::new();
        for (&node_id, &degree) in &degrees {
            let c = communities[&node_id];
            *community_weights.entry(c).or_insert(0.0) += degree;
        }

        // Try moving each node
        for &node_id in all_nodes {
            let current_community = communities[&node_id];

            // Find neighboring communities
            let mut neighbor_communities: HashSet<usize> = HashSet::new();
            neighbor_communities.insert(current_community);

            for (neighbor_id, _) in &adjacency[&node_id] {
                if let Some(&c) = communities.get(neighbor_id) {
                    neighbor_communities.insert(c);
                }
            }

            // Find best community
            let mut best_gain = 0.0;
            let mut best_community = current_community;

            let ki = degrees[&node_id];

            for &target in &neighbor_communities {
                if target == current_community {
                    continue;
                }

                // Sum of weights to target community
                let mut ki_in = 0.0;
                let mut ki_out = 0.0;

                for (neighbor_id, weight) in &adjacency[&node_id] {
                    if let Some(&nc) = communities.get(neighbor_id) {
                        if nc == target {
                            ki_in += weight;
                        }
                        if nc == current_community && *neighbor_id != node_id {
                            ki_out += weight;
                        }
                    }
                }

                let sigma_in = community_weights.get(&target).unwrap_or(&0.0);
                let sigma_out = community_weights.get(&current_community).unwrap_or(&0.0) - ki;

                let gain_to = ki_in - (sigma_in * ki) / (2.0 * m);
                let loss_from = ki_out - (sigma_out * ki) / (2.0 * m);
                let gain = (gain_to - loss_from) / m;

                if gain > best_gain {
                    best_gain = gain;
                    best_community = target;
                }
            }

            // Move node if beneficial
            if best_community != current_community && best_gain > 1e-10 {
                *community_weights.get_mut(&current_community).unwrap() -= ki;
                *community_weights.entry(best_community).or_insert(0.0) += ki;

                communities.insert(node_id, best_community);
                improved = true;
            }
        }

        if !improved {
            break;
        }
    }

    // Renumber communities
    let unique_communities: HashSet<usize> = communities.values().cloned().collect();
    let mut community_map: HashMap<usize, usize> = HashMap::new();
    for (new_id, old_id) in unique_communities.iter().enumerate() {
        community_map.insert(*old_id, new_id);
    }

    let mut final_assignments: HashMap<String, usize> = HashMap::new();
    for (&node_id, &old_c) in &communities {
        let name = id_to_name[&node_id].clone();
        final_assignments.insert(name, community_map[&old_c]);
    }

    // Calculate modularity
    let mut modularity = 0.0;
    for (&node_i, neighbors) in &adjacency {
        let ki = degrees[&node_i];
        let ci = community_map[&communities[&node_i]];

        for (node_j, weight) in neighbors {
            if let Some(&cj_old) = communities.get(node_j) {
                let kj = degrees[node_j];
                let cj = community_map[&cj_old];

                if ci == cj {
                    modularity += weight - (ki * kj) / (2.0 * m);
                }
            }
        }
    }
    modularity /= 2.0 * m;

    Ok(CommunityResult {
        assignments: final_assignments,
        modularity,
        num_communities: unique_communities.len(),
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 4 {
        eprintln!(
            "Usage: {} <implementation> <db_path> <scale_factor>",
            args[0]
        );
        eprintln!();
        eprintln!("Arguments:");
        eprintln!("  implementation - 'reference' or 'motlie_db'");
        eprintln!("  db_path        - Path to RocksDB directory");
        eprintln!("  scale_factor   - Positive integer to scale the graph size");
        eprintln!();
        eprintln!("Examples:");
        eprintln!(
            "  {} reference /tmp/louvain_test_db 1      # reference, 24 nodes (3 communities)",
            args[0]
        );
        eprintln!(
            "  {} motlie_db /tmp/louvain_test_db 1      # motlie_db, 24 nodes (3 communities)",
            args[0]
        );
        eprintln!(
            "  {} reference /tmp/louvain_test_db 10     # reference, 240 nodes (30 communities)",
            args[0]
        );
        eprintln!(
            "  {} motlie_db /tmp/louvain_test_db 10     # motlie_db, 240 nodes (30 communities)",
            args[0]
        );
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
    let id_to_name: HashMap<Id, String> = nodes.iter().map(|n| (n.id, n.name.clone())).collect();

    match implementation {
        Implementation::Reference => {
            // Run Louvain with reference implementation
            // Build adjacency list
            let mut adjacency: HashMap<String, Vec<(String, f64)>> = HashMap::new();
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
                let weight = edge.weight.unwrap_or(1.0);

                adjacency.get_mut(&src_name).unwrap().push((dst_name, weight));
            }

            let (result, time_ms, memory) = measure_time_and_memory(|| louvain_reference(&adjacency));
            let result_hash = Some(compute_hash(&result));

            let metrics = GraphMetrics {
                algorithm_name: "Louvain".to_string(),
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
            // Run Louvain with motlie_db
            let (reader, name_to_id, _query_handle) = build_graph(db_path, nodes, edges).await?;

            // Rebuild id_to_name with IDs from build_graph
            let id_to_name_rebuilt: HashMap<Id, String> = name_to_id
                .iter()
                .map(|(name, &id)| (id, name.clone()))
                .collect();

            let node_ids_rebuilt: Vec<Id> = name_to_id.values().cloned().collect();

            let timeout = Duration::from_secs(120);

            let (result, time_ms, memory) = measure_time_and_memory_async(|| {
                louvain_motlie(&node_ids_rebuilt, &id_to_name_rebuilt, &reader, timeout)
            })
            .await;
            let result = result?;
            let result_hash = Some(compute_hash(&result));

            // Measure disk usage
            let (disk_files, disk_size) = get_disk_metrics(db_path).unwrap_or((0, 0));

            let metrics = GraphMetrics {
                algorithm_name: "Louvain".to_string(),
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
