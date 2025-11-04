use anyhow::{Context, Result};
use csv::ReaderBuilder;
use motlie_db::{
    create_mutation_writer, spawn_fulltext_consumer, spawn_graph_consumer_with_next, AddEdgeArgs,
    AddFragmentArgs, AddNodeArgs, Id, WriterConfig,
};
use std::collections::HashMap;
use std::io;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging to see the mutation processing order
    env_logger::init();

    println!("Motlie CSV Processor - Demonstrating Graph → FullText Chaining");
    println!("===================================================================");
    println!("Mutations will flow: Writer → Graph → FullText");
    println!("Watch the logs to see Graph processing happens before FullText");
    println!();
    println!("Reading CSV from stdin...");
    println!("Format:");
    println!("  node1,fragment for node1");
    println!("  node2,fragment for node2");
    println!("  node1,node2,edge_name,edge fragment for edge1");
    println!();

    // Configuration for the consumer chain
    let config = WriterConfig {
        channel_buffer_size: 1000,
    };

    // Create the FullText consumer (end of chain)
    println!("Setting up consumer chain:");
    println!("  1. Creating FullText consumer (end of chain)");
    let (fulltext_sender, fulltext_receiver) = mpsc::channel(config.channel_buffer_size);
    let fulltext_handle = spawn_fulltext_consumer(fulltext_receiver, config.clone());

    // Create the Graph consumer that forwards to FullText
    println!("  2. Creating Graph consumer (forwards to FullText)");
    let (writer, graph_receiver) = create_mutation_writer(config.clone());
    let graph_handle = spawn_graph_consumer_with_next(
        graph_receiver,
        config,
        Path::new("/tmp/motlie_graph_db"),
        fulltext_sender,
    );

    println!("  3. Consumer chain ready: Writer → Graph → FullText");
    println!();

    // Keep track of node name to ID mapping for edges
    let mut node_ids: HashMap<String, Id> = HashMap::new();

    // Process CSV from stdin
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(io::stdin());

    let mut line_count = 0;

    for result in reader.records() {
        let record = result.context("Failed to read CSV record")?;
        line_count += 1;

        match record.len() {
            2 => {
                // node,fragment - create vertex and fragment
                let node_name = record.get(0).unwrap().trim();
                let fragment_text = record.get(1).unwrap().trim();

                if node_name.is_empty() {
                    println!("Warning: Empty node name on line {}, skipping", line_count);
                    continue;
                }

                // Generate or reuse ID for this node
                let node_id = node_ids
                    .entry(node_name.to_string())
                    .or_insert_with(Id::new)
                    .clone();

                let current_time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64;

                // Create vertex
                let vertex_args = AddNodeArgs {
                    id: node_id.clone(),
                    ts_millis: current_time,
                    name: node_name.to_string(),
                };

                // Create fragment
                let fragment_args = AddFragmentArgs {
                    id: node_id,
                    ts_millis: current_time,
                    content: fragment_text.to_string(),
                };

                // Send to Graph consumer (which will forward to FullText)
                writer
                    .add_vertex(vertex_args)
                    .await
                    .context("Failed to send vertex to consumer chain")?;
                writer
                    .add_fragment(fragment_args)
                    .await
                    .context("Failed to send fragment to consumer chain")?;

                println!(
                    "Sent vertex '{}' with fragment ({} chars) to chain",
                    node_name,
                    fragment_text.len()
                );
            }
            4 => {
                // source,target,edge_name,edge_fragment - create edge with fragment
                let source_name = record.get(0).unwrap().trim();
                let target_name = record.get(1).unwrap().trim();
                let edge_name = record.get(2).unwrap().trim();
                let edge_fragment = record.get(3).unwrap().trim();

                if source_name.is_empty() || target_name.is_empty() {
                    println!(
                        "Warning: Empty source or target on line {}, skipping",
                        line_count
                    );
                    continue;
                }

                // Get or create IDs for source and target nodes
                let source_id = node_ids
                    .entry(source_name.to_string())
                    .or_insert_with(Id::new)
                    .clone();
                let target_id = node_ids
                    .entry(target_name.to_string())
                    .or_insert_with(Id::new)
                    .clone();

                // Generate a unique ID for this edge
                let edge_id = Id::new();

                let current_time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64;

                let edge_args = AddEdgeArgs {
                    id: edge_id.clone(),
                    source_vertex_id: source_id,
                    target_vertex_id: target_id,
                    ts_millis: current_time,
                    name: edge_name.to_string(),
                };

                // Create fragment for the edge
                let fragment_args = AddFragmentArgs {
                    id: edge_id,
                    ts_millis: current_time,
                    content: edge_fragment.to_string(),
                };

                // Send to Graph consumer (which will forward to FullText)
                writer
                    .add_edge(edge_args)
                    .await
                    .context("Failed to send edge to consumer chain")?;
                writer
                    .add_fragment(fragment_args)
                    .await
                    .context("Failed to send edge fragment to consumer chain")?;

                println!(
                    "Sent edge '{}' -> '{}' (name: '{}', fragment: {} chars) to chain",
                    source_name,
                    target_name,
                    edge_name,
                    edge_fragment.len()
                );
            }
            _ => {
                println!(
                    "Warning: Invalid CSV format on line {} (expected 2 or 4 fields, got {})",
                    line_count,
                    record.len()
                );
                continue;
            }
        }
    }

    println!("\nProcessed {} lines from stdin", line_count);
    println!("Created {} unique nodes", node_ids.len());
    println!("Shutting down consumer chain...");

    // Close writer to signal shutdown (this will cascade through the chain)
    drop(writer);

    // Wait for Graph consumer to finish (which will close FullText's channel)
    println!("  1. Waiting for Graph consumer to finish...");
    graph_handle
        .await
        .context("Graph consumer task failed")?
        .context("Graph consumer failed")?;
    println!("  2. Graph consumer finished");

    // Wait for FullText consumer to finish
    println!("  3. Waiting for FullText consumer to finish...");
    fulltext_handle
        .await
        .context("FullText consumer task failed")?
        .context("FullText consumer failed")?;
    println!("  4. FullText consumer finished");

    println!("\nAll consumers shut down successfully");
    println!("Check the logs above - you should see [Graph] messages before [FullText] messages");

    Ok(())
}
