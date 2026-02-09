/// PageRank Algorithm - ORIGINAL (unoptimized)
///
/// This is the original implementation using individual NodeById calls.
/// Used for benchmarking comparison against the NodesByIdsMulti optimized version.
///
/// # Unified API Usage
///
/// This example uses the **unified motlie_db API** (porcelain layer):
/// - Storage: `motlie_db::{Storage, StorageConfig, ReadWriteHandles}`
/// - Queries: `motlie_db::query::{IncomingEdges, OutgoingEdges, Runnable}`, `motlie_db::graph::query::NodeById`
/// - Reader: `motlie_db::reader::Reader`
///
/// Unlike the optimized pagerank.rs, this version makes individual NodeById calls
/// for each node during initialization (N calls instead of 1 batch call).

// Include the common module
#[path = "common.rs"]
mod common;

use anyhow::Result;
use common::{build_graph, compute_hash_pagerank, get_disk_metrics, measure_time_and_memory, measure_time_and_memory_async, parse_scale_factor, GraphEdge, GraphMetrics, GraphNode, Implementation};
use motlie_db::query::{IncomingEdges, OutgoingEdges, Runnable as QueryRunnable};
use motlie_db::Id;
use std::collections::HashMap;
use std::env;
use std::path::Path;
use tokio::time::Duration;

fn create_test_graph(scale: usize) -> (Vec<GraphNode>, Vec<GraphEdge>) {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut all_node_ids = Vec::new();

    let base_names = vec!["Page_A", "Page_B", "Page_C", "Page_D", "Page_E", "Page_F", "Page_G", "Page_H", "Page_I", "Page_J"];
    let total_nodes = scale * base_names.len();

    for i in 0..total_nodes {
        let id = Id::new();
        let node_name = if scale == 1 {
            base_names[i % base_names.len()].to_string()
        } else {
            format!("Page{}", i)
        };
        all_node_ids.push(id);
        nodes.push(GraphNode {
            id,
            name: node_name,
            summary: None,
        });
    }

    let website_size = base_names.len();

    for website in 0..scale {
        let website_start = website * website_size;

        let internal_links = vec![
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5),
            (1, 0), (1, 4), (1, 6),
            (2, 0), (2, 3), (2, 7),
            (3, 0), (3, 2), (3, 1), (3, 8),
            (4, 0), (4, 9),
            (5, 0), (5, 4),
            (6, 0), (6, 7),
            (7, 0), (7, 6),
            (8, 0), (8, 9),
            (9, 0),
        ];

        for (src_offset, dst_offset) in internal_links {
            edges.push(GraphEdge {
                source: all_node_ids[website_start + src_offset],
                target: all_node_ids[website_start + dst_offset],
                name: format!("link_w{}_{}_{}", website, src_offset, dst_offset),
                weight: Some(1.0),
                summary: None,
            });
        }

        if website < scale - 1 {
            let next_website_start = (website + 1) * website_size;
            edges.push(GraphEdge {
                source: all_node_ids[website_start],
                target: all_node_ids[next_website_start],
                name: format!("external_{}_{}", website, website + 1),
                weight: Some(1.0),
                summary: None,
            });
            edges.push(GraphEdge {
                source: all_node_ids[website_start + 3],
                target: all_node_ids[next_website_start + 3],
                name: format!("hub_link_{}_{}", website, website + 1),
                weight: Some(1.0),
                summary: None,
            });
            edges.push(GraphEdge {
                source: all_node_ids[next_website_start],
                target: all_node_ids[website_start],
                name: format!("backlink_{}_{}", website + 1, website),
                weight: Some(1.0),
                summary: None,
            });
        }
    }

    (nodes, edges)
}

/// ORIGINAL PageRank - uses individual NodeById calls during initialization
async fn pagerank_motlie_original(
    all_nodes: &[Id],
    reader: &motlie_db::graph::reader::Reader,
    timeout: Duration,
    damping_factor: f64,
    iterations: usize,
) -> Result<HashMap<String, f64>> {
    let n = all_nodes.len() as f64;
    let mut ranks: HashMap<Id, f64> = HashMap::new();
    let mut new_ranks: HashMap<Id, f64> = HashMap::new();
    let mut name_map: HashMap<Id, String> = HashMap::new();
    let mut outgoing_counts: HashMap<Id, usize> = HashMap::new();

    let initial_rank = 1.0 / n;

    // ORIGINAL: Individual NodeById call for each node
    for &node_id in all_nodes {
        ranks.insert(node_id, initial_rank);
        new_ranks.insert(node_id, 0.0);

        let (name, _summary, _version) = motlie_db::graph::query::NodeById::new(node_id, None)
            .run(reader, timeout)
            .await?;
        name_map.insert(node_id, name);

        let outgoing = OutgoingEdges::new(node_id, None)
            .run(reader, timeout)
            .await?;
        outgoing_counts.insert(node_id, outgoing.len());
    }

    for _iteration in 0..iterations {
        for rank in new_ranks.values_mut() {
            *rank = 0.0;
        }

        for &node_id in all_nodes {
            let incoming = IncomingEdges::new(node_id.into(), None)
                .run(reader, timeout)
                .await?;

            let mut rank_sum = 0.0;
            for (_weight, _dst, src, _name, _version) in incoming {
                let src_id = src;
                let src_rank = ranks[&src_id];
                let src_out_count = outgoing_counts[&src_id] as f64;

                if src_out_count > 0.0 {
                    rank_sum += src_rank / src_out_count;
                }
            }

            new_ranks.insert(node_id, (1.0 - damping_factor) / n + damping_factor * rank_sum);
        }

        std::mem::swap(&mut ranks, &mut new_ranks);
    }

    Ok(ranks
        .into_iter()
        .filter_map(|(id, rank)| name_map.get(&id).map(|name| (name.clone(), rank)))
        .collect())
}

fn pagerank_reference(
    adjacency: &HashMap<String, Vec<String>>,
    damping_factor: f64,
    iterations: usize,
) -> HashMap<String, f64> {
    let n = adjacency.len() as f64;
    let mut ranks: HashMap<String, f64> = HashMap::new();
    let mut new_ranks: HashMap<String, f64> = HashMap::new();

    let mut incoming: HashMap<String, Vec<String>> = HashMap::new();
    for node in adjacency.keys() {
        incoming.insert(node.clone(), Vec::new());
        ranks.insert(node.clone(), 1.0 / n);
        new_ranks.insert(node.clone(), 0.0);
    }

    for (src, dests) in adjacency {
        for dest in dests {
            incoming.get_mut(dest).unwrap().push(src.clone());
        }
    }

    for _ in 0..iterations {
        for (node, rank) in new_ranks.iter_mut() {
            let mut rank_sum = 0.0;
            if let Some(in_nodes) = incoming.get(node) {
                for in_node in in_nodes {
                    let in_rank = ranks[in_node];
                    let out_count = adjacency[in_node].len() as f64;
                    if out_count > 0.0 {
                        rank_sum += in_rank / out_count;
                    }
                }
            }
            *rank = (1.0 - damping_factor) / n + damping_factor * rank_sum;
        }
        std::mem::swap(&mut ranks, &mut new_ranks);
    }

    ranks
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
    let node_ids: Vec<Id> = nodes.iter().map(|n| n.id).collect();

    let damping_factor = 0.85;
    let iterations = 50;

    match implementation {
        Implementation::Reference => {
            let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
            for node in &nodes {
                adjacency.insert(node.name.clone(), Vec::new());
            }

            for edge in &edges {
                let src_name = nodes.iter().find(|n| n.id == edge.source).unwrap().name.clone();
                let dst_name = nodes.iter().find(|n| n.id == edge.target).unwrap().name.clone();
                adjacency.get_mut(&src_name).unwrap().push(dst_name);
            }

            let (result, time_ms, memory) = measure_time_and_memory(|| {
                pagerank_reference(&adjacency, damping_factor, iterations)
            });
            let result_hash = Some(compute_hash_pagerank(&result));

            let metrics = GraphMetrics {
                algorithm_name: "PageRank_Original".to_string(),
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
            let (reader, _name_to_id, _handles) = build_graph(db_path, nodes, edges).await?;
            let timeout = Duration::from_secs(300);

            let (result, time_ms, memory) = measure_time_and_memory_async(|| {
                pagerank_motlie_original(&node_ids, reader.graph(), timeout, damping_factor, iterations)
            })
            .await;
            let result = result?;
            let result_hash = Some(compute_hash_pagerank(&result));

            let (disk_files, disk_size) = get_disk_metrics(db_path).unwrap_or((0, 0));

            let metrics = GraphMetrics {
                algorithm_name: "PageRank_Original".to_string(),
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
