/// PageRank Algorithm Implementation
///
/// This example demonstrates the PageRank algorithm using motlie_db.
/// PageRank measures the importance of nodes in a directed graph based on
/// the structure of incoming links.
///
/// Developed by Google founders Larry Page and Sergey Brin, PageRank is used for:
/// - Web page ranking in search engines
/// - Social network influence analysis
/// - Citation analysis in academic papers
/// - Recommendation systems
///
/// The algorithm iteratively computes a score for each node based on:
/// - The number of incoming links
/// - The importance of nodes providing those links
/// - A damping factor (usually 0.85) modeling random surfer behavior
///
/// Usage: pagerank <db_path>

// Include the common module
#[path = "common.rs"]
mod common;

use anyhow::Result;
use common::{build_graph, measure_time_and_memory, measure_time_and_memory_async, parse_scale_factor, GraphEdge, GraphNode};
use motlie_db::{Id, IncomingEdges, OutgoingEdges, QueryRunnable};
use std::collections::HashMap;
use std::env;
use std::path::Path;
use tokio::time::Duration;

/// Create a test graph suitable for PageRank demonstration
/// Represents a connected web page link structure across multiple sites
/// The scale parameter determines the total number of pages (scale * 8)
fn create_test_graph(scale: usize) -> (Vec<GraphNode>, Vec<GraphEdge>) {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut all_node_ids = Vec::new();

    let base_size = 8; // Base pages per website
    let total_nodes = scale * base_size;

    // Create all nodes
    for i in 0..total_nodes {
        let id = Id::new();
        let node_name = if scale == 1 {
            vec!["Homepage", "About", "Products", "Blog", "Contact", "FAQ", "Privacy", "Terms"][i].to_string()
        } else {
            format!("Page{}", i)
        };
        all_node_ids.push(id);
        nodes.push(GraphNode {
            id,
            name: node_name,
        });
    }

    // Create a connected link structure representing web pages
    // Each website (cluster of 8 pages) has internal links
    // and external links to other websites
    let website_size = base_size;

    for website in 0..scale {
        let website_start = website * website_size;

        // Internal links within each website (following base pattern)
        // Homepage is a hub linking to many pages
        let internal_links = vec![
            (0, 1), (0, 2), (0, 3), (0, 4), // Homepage links out
            (1, 0), (1, 4), // About links
            (2, 0), (2, 3), // Products links
            (3, 0), (3, 2), (3, 1), // Blog links (hub)
            (4, 0), // Contact links
            (5, 0), (5, 4), // FAQ links
            (6, 0), (6, 7), // Privacy links
            (7, 0), (7, 6), // Terms links
        ];

        for (src_offset, dst_offset) in internal_links {
            edges.push(GraphEdge {
                source: all_node_ids[website_start + src_offset],
                target: all_node_ids[website_start + dst_offset],
                name: format!("link_w{}_{}_{}", website, src_offset, dst_offset),
                weight: Some(1.0),
            });
        }

        // External links between websites
        // Create cross-links to make the entire web connected
        if website < scale - 1 {
            let next_website_start = (website + 1) * website_size;

            // Homepage of current site links to next site's homepage
            edges.push(GraphEdge {
                source: all_node_ids[website_start],
                target: all_node_ids[next_website_start],
                name: format!("external_{}_{}", website, website + 1),
                weight: Some(1.0),
            });

            // Blog of current site links to next site's blog
            edges.push(GraphEdge {
                source: all_node_ids[website_start + 3],
                target: all_node_ids[next_website_start + 3],
                name: format!("blog_link_{}_{}", website, website + 1),
                weight: Some(1.0),
            });

            // Next site's homepage links back to current site's homepage (reciprocal link)
            edges.push(GraphEdge {
                source: all_node_ids[next_website_start],
                target: all_node_ids[website_start],
                name: format!("backlink_{}_{}", website + 1, website),
                weight: Some(1.0),
            });
        }
    }

    (nodes, edges)
}

/// PageRank implementation using motlie_db
///
/// The PageRank formula:
/// PR(A) = (1-d) + d * Σ(PR(Ti) / C(Ti))
///
/// Where:
/// - PR(A) is the PageRank of page A
/// - d is the damping factor (typically 0.85)
/// - Ti are pages that link to A
/// - C(Ti) is the number of outbound links from Ti
async fn pagerank_motlie(
    all_nodes: &[Id],
    reader: &motlie_db::Reader,
    timeout: Duration,
    damping_factor: f64,
    iterations: usize,
) -> Result<HashMap<String, f64>> {
    let n = all_nodes.len() as f64;
    let mut ranks: HashMap<Id, f64> = HashMap::new();
    let mut new_ranks: HashMap<Id, f64> = HashMap::new();
    let mut name_map: HashMap<Id, String> = HashMap::new();
    let mut outgoing_counts: HashMap<Id, usize> = HashMap::new();

    // Initialize all ranks to 1/N
    let initial_rank = 1.0 / n;
    for &node_id in all_nodes {
        ranks.insert(node_id, initial_rank);
        new_ranks.insert(node_id, 0.0);

        // Get node name
        let (name, _summary) = motlie_db::NodeById::new(node_id, None)
            .run(reader, timeout)
            .await?;
        name_map.insert(node_id, name);

        // Count outgoing edges
        let outgoing = OutgoingEdges::new(node_id, None)
            .run(reader, timeout)
            .await?;
        outgoing_counts.insert(node_id, outgoing.len());
    }

    // Iterate to convergence
    for iteration in 0..iterations {
        // Reset new ranks
        for rank in new_ranks.values_mut() {
            *rank = 0.0;
        }

        // Calculate new ranks based on incoming links
        for &node_id in all_nodes {
            let incoming = IncomingEdges::new(node_id.into(), None)
                .run(reader, timeout)
                .await?;

            let mut rank_sum = 0.0;
            for (_weight, _dst, src, _name) in incoming {
                let src_id = src;
                let src_rank = ranks[&src_id];
                let src_out_count = outgoing_counts[&src_id] as f64;

                if src_out_count > 0.0 {
                    rank_sum += src_rank / src_out_count;
                }
            }

            // Apply PageRank formula
            new_ranks.insert(node_id, (1.0 - damping_factor) / n + damping_factor * rank_sum);
        }

        // Swap old and new ranks
        std::mem::swap(&mut ranks, &mut new_ranks);

        // Print progress every 10 iterations
        if (iteration + 1) % 10 == 0 {
            let total: f64 = ranks.values().sum();
            println!("  Iteration {}: Total PageRank = {:.4}", iteration + 1, total);
        }
    }

    // Convert to name-based map
    Ok(ranks
        .into_iter()
        .filter_map(|(id, rank)| name_map.get(&id).map(|name| (name.clone(), rank)))
        .collect())
}

/// Simple reference PageRank implementation (in-memory)
fn pagerank_reference(
    adjacency: &HashMap<String, Vec<String>>,
    damping_factor: f64,
    iterations: usize,
) -> HashMap<String, f64> {
    let n = adjacency.len() as f64;
    let mut ranks: HashMap<String, f64> = HashMap::new();
    let mut new_ranks: HashMap<String, f64> = HashMap::new();

    // Build reverse adjacency (incoming links)
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

    // Iterate
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

    if args.len() != 3 {
        eprintln!("Usage: {} <db_path> <scale_factor>", args[0]);
        eprintln!();
        eprintln!("Arguments:");
        eprintln!("  db_path       - Path to RocksDB directory");
        eprintln!("  scale_factor  - Positive integer to scale the graph size");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  {} /tmp/pagerank_test_db 1     # Base graph (8 pages, 18 links)", args[0]);
        eprintln!("  {} /tmp/pagerank_test_db 100   # 100x larger (800 pages, 1800 links)", args[0]);
        std::process::exit(1);
    }

    let db_path = Path::new(&args[1]);
    let scale = parse_scale_factor(&args[2])?;

    println!("📊 PageRank Algorithm Demonstration");
    println!("{:=<80}", "");
    println!("\n📏 Scale factor: {}x", scale);

    // Generate test graph
    println!("\n📊 Generating test graph (web page link structure)...");
    let start_gen = std::time::Instant::now();
    let (nodes, edges) = create_test_graph(scale);
    let gen_time = start_gen.elapsed();
    let num_nodes = nodes.len();
    let num_edges = edges.len();

    println!("  Pages: {}", num_nodes);
    println!("  Links: {}", num_edges);
    println!("  Generation time: {:.2} ms", gen_time.as_secs_f64() * 1000.0);

    let node_ids: Vec<Id> = nodes.iter().map(|n| n.id).collect();

    // Build adjacency list for reference implementation
    let mut adjacency: HashMap<String, Vec<String>> = HashMap::new();
    for node in &nodes {
        adjacency.insert(node.name.clone(), Vec::new());
    }

    for edge in &edges {
        let src_name = nodes.iter().find(|n| n.id == edge.source).unwrap().name.clone();
        let dst_name = nodes.iter().find(|n| n.id == edge.target).unwrap().name.clone();
        adjacency.get_mut(&src_name).unwrap().push(dst_name);
    }

    let damping_factor = 0.85;
    let iterations = 50;

    // Pass 1: Reference implementation
    println!("\n🔹 Pass 1: Reference PageRank (in-memory)");
    println!("  Damping factor: {}", damping_factor);
    println!("  Iterations: {}", iterations);

    let (ref_result, ref_time, ref_result_memory) = measure_time_and_memory(|| {
        pagerank_reference(&adjacency, damping_factor, iterations)
    });

    println!("\n  Results:");
    let mut sorted_ref: Vec<_> = ref_result.iter().collect();
    sorted_ref.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    if num_nodes <= 20 {
        for (page, rank) in &sorted_ref {
            println!("    {:<12} {:.6}", page, rank);
        }
    } else {
        println!("    Top 10 pages:");
        for (page, rank) in sorted_ref.iter().take(10) {
            println!("      {:<25} {:.6}", page, rank);
        }
    }
    println!("  Execution time: {:.2} ms", ref_time);

    // Pass 2: PageRank with motlie_db
    println!("\n🔹 Pass 2: PageRank with motlie_db (persistent)");
    println!("  Damping factor: {}", damping_factor);
    println!("  Iterations: {}", iterations);

    let (reader, _name_to_id, _query_handle) = build_graph(db_path, nodes, edges).await?;
    let timeout = Duration::from_secs(120); // PageRank needs more time for iterations and large graphs

    let (motlie_result, motlie_time, motlie_result_memory) = measure_time_and_memory_async(|| {
        pagerank_motlie(&node_ids, &reader, timeout, damping_factor, iterations)
    })
    .await;
    let motlie_result = motlie_result?;

    println!("\n  Results:");
    let mut sorted_motlie: Vec<_> = motlie_result.iter().collect();
    sorted_motlie.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    if num_nodes <= 20 {
        for (page, rank) in &sorted_motlie {
            println!("    {:<12} {:.6}", page, rank);
        }
    } else {
        println!("    Top 10 pages:");
        for (page, rank) in sorted_motlie.iter().take(10) {
            println!("      {:<25} {:.6}", page, rank);
        }
    }
    println!("  Execution time: {:.2} ms", motlie_time);

    // Verify correctness
    println!("\n✅ Correctness Check:");
    let mut max_diff = 0.0_f64;
    let mut all_match = true;

    for (page, ref_rank) in &ref_result {
        if let Some(motlie_rank) = motlie_result.get(page) {
            let diff = (ref_rank - motlie_rank).abs();
            max_diff = max_diff.max(diff);

            if diff > 0.001 {
                // Tolerance for floating point differences
                println!(
                    "  ⚠ {} - difference: {:.6} (ref: {:.6}, motlie: {:.6})",
                    page, diff, ref_rank, motlie_rank
                );
                all_match = false;
            }
        }
    }

    if all_match {
        println!("  ✓ All PageRank scores match (max difference: {:.6})", max_diff);
    } else {
        println!("  ⚠ Some scores differ beyond tolerance");
    }

    // Verify ranking order
    println!("\n🔹 Ranking Comparison:");
    println!("  Reference ranking:");
    for (i, (page, rank)) in sorted_ref.iter().take(5).enumerate() {
        println!("    {}. {} ({:.6})", i + 1, page, rank);
    }

    println!("\n  motlie_db ranking:");
    for (i, (page, rank)) in sorted_motlie.iter().take(5).enumerate() {
        println!("    {}. {} ({:.6})", i + 1, page, rank);
    }

    // Performance comparison
    println!("\n{:=<80}", "");
    println!("📊 Performance Comparison");
    println!("{:=<80}", "");
    println!("  Reference:  {:.2} ms", ref_time);
    println!("  motlie_db:  {:.2} ms", motlie_time);

    let speedup = ref_time / motlie_time;
    if speedup > 1.0 {
        println!("  Speedup:    {:.2}x (motlie_db is faster)", speedup);
    } else {
        println!("  Slowdown:   {:.2}x (reference is faster)", 1.0 / speedup);
    }

    // Memory comparison
    if let (Some(ref_mem), Some(motlie_mem)) = (ref_result_memory, motlie_result_memory) {
        println!("\n💾 Memory Usage (delta):");

        fn format_bytes(bytes: usize) -> String {
            if bytes >= 1024 * 1024 {
                format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
            } else if bytes >= 1024 {
                format!("{:.2} KB", bytes as f64 / 1024.0)
            } else {
                format!("{} bytes", bytes)
            }
        }

        println!("  Reference:  {}", format_bytes(ref_mem));
        println!("  motlie_db:  {}", format_bytes(motlie_mem));

        if ref_mem > 0 {
            let ratio = motlie_mem as f64 / ref_mem as f64;
            if ratio < 1.0 {
                println!("  Savings:    {:.2}x less memory used by motlie_db", 1.0 / ratio);
            } else {
                println!("  Overhead:   {:.2}x more memory used by motlie_db", ratio);
            }
        }
    }

    println!("\n{:=<80}", "");
    println!("\n✅ PageRank example completed successfully!");
    Ok(())
}
