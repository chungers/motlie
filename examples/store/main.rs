use anyhow::{Context, Result};
use csv::ReaderBuilder;
use motlie_db::{
    create_mutation_writer, spawn_fulltext_consumer, spawn_graph_consumer_with_next, AddEdge,
    AddFragment, AddNode, Id, TimestampMilli, WriterConfig,
};
use rocksdb::DB;
use std::collections::{HashMap, HashSet};
use std::env;
use std::io;
use std::path::Path;
use tokio::sync::mpsc;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging to see the mutation processing order
    env_logger::init();

    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();

    let (verify_mode, db_path) = match args.len() {
        2 => (false, &args[1]),
        3 if args[1] == "--verify" => (true, &args[2]),
        _ => {
            eprintln!("Usage: {} [--verify] <db_path>", args[0]);
            eprintln!();
            eprintln!("Modes:");
            eprintln!("  Store mode:   Read CSV from stdin and store in database");
            eprintln!("  Verify mode:  Read CSV from stdin and verify against database");
            eprintln!();
            eprintln!("Arguments:");
            eprintln!("  --verify   Enable verification mode");
            eprintln!("  <db_path>  Path to RocksDB database directory");
            eprintln!();
            eprintln!("Examples:");
            eprintln!("  cat input.csv | target/release/examples/store /tmp/motlie_graph_db");
            eprintln!("  cat input.csv | target/release/examples/store --verify /tmp/motlie_graph_db");
            std::process::exit(1);
        }
    };

    if verify_mode {
        verify_mode_main(db_path)?;
    } else {
        store_mode_main(db_path).await?;
    }

    Ok(())
}

async fn store_mode_main(db_path: &str) -> Result<()> {
    println!("Motlie CSV Processor - Demonstrating Graph → FullText Chaining");
    println!("===================================================================");
    println!("Mutations will flow: Writer → Graph → FullText");
    println!("Watch the logs to see Graph processing happens before FullText");
    println!();
    println!("Database path: {}", db_path);
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
        Path::new(db_path),
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

                let current_time = TimestampMilli::now();

                // Create vertex
                let vertex_args = AddNode {
                    id: node_id.clone(),
                    ts_millis: current_time,
                    name: node_name.to_string(),
                };

                // Create fragment
                let fragment_args = AddFragment {
                    id: node_id,
                    ts_millis: current_time.0,
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

                let current_time = TimestampMilli::now();

                let edge_args = AddEdge {
                    id: edge_id.clone(),
                    source_node_id: source_id,
                    target_node_id: target_id,
                    ts_millis: current_time,
                    name: edge_name.to_string(),
                };

                // Create fragment for the edge
                let fragment_args = AddFragment {
                    id: edge_id,
                    ts_millis: current_time.0,
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

fn verify_mode_main(db_path: &str) -> Result<()> {
    println!("Motlie Store Verifier");
    println!("====================");
    println!();
    println!("Database: {}", db_path);
    println!("Reading CSV from stdin...");
    println!();

    let db_path_obj = Path::new(db_path);

    // Check if database exists
    if !db_path_obj.exists() {
        eprintln!("❌ Database does not exist at {}", db_path);
        eprintln!("Run in store mode first:");
        eprintln!("  cat input.csv | target/release/examples/store {}", db_path);
        std::process::exit(1);
    }

    // Parse CSV from stdin
    println!("📄 Parsing CSV from stdin...");
    let expected_data = parse_csv_from_stdin()?;
    println!("   Nodes: {}", expected_data.nodes.len());
    println!("   Edges: {}", expected_data.edges.len());
    println!("   Total fragments: {}", expected_data.node_fragments.len() + expected_data.edge_fragments.len());
    println!();

    // Open database in read-only mode
    let column_families = vec!["nodes", "edges", "fragments", "forward_edges", "reverse_edges"];
    let db = DB::open_cf_for_read_only(
        &rocksdb::Options::default(),
        db_path_obj,
        &column_families,
        false,
    )
    .context("Failed to open database")?;

    println!("✓ Database opened successfully\n");

    // Verify data
    let mut all_ok = true;

    println!("🔍 Verifying Nodes...");
    if !verify_nodes(&db, &expected_data)? {
        all_ok = false;
    }

    println!("\n🔍 Verifying Edges...");
    if !verify_edges(&db, &expected_data)? {
        all_ok = false;
    }

    println!("\n🔍 Verifying Fragments...");
    if !verify_fragments(&db, &expected_data)? {
        all_ok = false;
    }

    println!();
    if all_ok {
        println!("✅ All verification checks passed!");
        println!("   The database contents match the CSV input.");
    } else {
        println!("❌ Some verification checks failed!");
        println!("   The database contents do not fully match the CSV input.");
        std::process::exit(1);
    }

    Ok(())
}

// Verification data structures

#[derive(Debug)]
struct ExpectedData {
    nodes: HashMap<String, String>, // name -> fragment
    edges: Vec<EdgeData>,
    node_fragments: HashMap<String, String>, // node_name -> fragment
    edge_fragments: HashMap<(String, String, String), String>, // (source, target, name) -> fragment
}

#[derive(Debug)]
struct EdgeData {
    source: String,
    target: String,
    name: String,
    fragment: String,
}

fn parse_csv_from_stdin() -> Result<ExpectedData> {
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(io::stdin());

    let mut nodes = HashMap::new();
    let mut edges = Vec::new();
    let mut node_fragments = HashMap::new();
    let mut edge_fragments = HashMap::new();

    for result in reader.records() {
        let record = result.context("Failed to read CSV record")?;

        match record.len() {
            2 => {
                // Node with fragment: name, fragment
                let name = record.get(0).unwrap().trim().to_string();
                let fragment = record.get(1).unwrap().trim().to_string();
                nodes.insert(name.clone(), fragment.clone());
                node_fragments.insert(name, fragment);
            }
            4 => {
                // Edge with fragment: source, target, edge_name, fragment
                let source = record.get(0).unwrap().trim().to_string();
                let target = record.get(1).unwrap().trim().to_string();
                let edge_name = record.get(2).unwrap().trim().to_string();
                let fragment = record.get(3).unwrap().trim().to_string();

                // Ensure source and target nodes exist (they might not have explicit fragments)
                nodes.entry(source.clone()).or_insert_with(|| String::new());
                nodes.entry(target.clone()).or_insert_with(|| String::new());

                edge_fragments.insert((source.clone(), target.clone(), edge_name.clone()), fragment.clone());
                edges.push(EdgeData {
                    source,
                    target,
                    name: edge_name,
                    fragment,
                });
            }
            _ => {
                // Skip invalid rows
            }
        }
    }

    Ok(ExpectedData {
        nodes,
        edges,
        node_fragments,
        edge_fragments,
    })
}

fn verify_nodes(db: &DB, expected: &ExpectedData) -> Result<bool> {
    let cf = db.cf_handle("nodes").context("Nodes CF not found")?;

    // Count nodes in database
    let mut db_node_count = 0;
    let iter = db.iterator_cf(cf, rocksdb::IteratorMode::Start);
    let mut db_node_names = HashSet::new();

    for item in iter {
        let (_key, value) = item.context("Failed to read from nodes CF")?;
        db_node_count += 1;

        // Deserialize to extract node name
        if let Ok(node_value) = deserialize_node_value(&value) {
            // Extract name from markdown summary (format: "# name\n")
            if let Some(name_line) = node_value.lines().nth(1) {
                let name = name_line.trim_start_matches("# ").trim();
                db_node_names.insert(name.to_string());
            }
        }
    }

    let expected_count = expected.nodes.len();
    println!("   Expected: {} nodes", expected_count);
    println!("   Found:    {} nodes", db_node_count);

    let mut all_ok = true;

    if db_node_count == expected_count {
        println!("   ✓ Node count matches");
    } else {
        println!("   ✗ Node count mismatch!");
        all_ok = false;
    }

    // Check if all expected node names are present
    let mut missing_nodes = Vec::new();
    for expected_name in expected.nodes.keys() {
        if !db_node_names.contains(expected_name) {
            missing_nodes.push(expected_name);
        }
    }

    if missing_nodes.is_empty() {
        println!("   ✓ All expected nodes found in database");
    } else {
        println!("   ✗ Missing nodes: {:?}", missing_nodes);
        all_ok = false;
    }

    Ok(all_ok)
}

fn verify_edges(db: &DB, expected: &ExpectedData) -> Result<bool> {
    let cf = db.cf_handle("edges").context("Edges CF not found")?;

    // Count edges in database
    let mut db_edge_count = 0;
    let iter = db.iterator_cf(cf, rocksdb::IteratorMode::Start);

    for item in iter {
        let _ = item.context("Failed to read from edges CF")?;
        db_edge_count += 1;
    }

    let expected_count = expected.edges.len();
    println!("   Expected: {} edges", expected_count);
    println!("   Found:    {} edges", db_edge_count);

    let all_ok = if db_edge_count == expected_count {
        println!("   ✓ Edge count matches");
        true
    } else {
        println!("   ✗ Edge count mismatch!");
        false
    };

    Ok(all_ok)
}

fn verify_fragments(db: &DB, expected: &ExpectedData) -> Result<bool> {
    let cf = db.cf_handle("fragments").context("Fragments CF not found")?;

    // Count fragments in database
    let mut db_fragment_count = 0;
    let mut db_fragments_content = HashSet::new();
    let iter = db.iterator_cf(cf, rocksdb::IteratorMode::Start);

    for item in iter {
        let (_, value) = item.context("Failed to read from fragments CF")?;
        db_fragment_count += 1;

        if let Ok(content) = deserialize_fragment_value(&value) {
            db_fragments_content.insert(content);
        }
    }

    let expected_count = expected.node_fragments.len() + expected.edge_fragments.len();
    println!("   Expected: at least {} fragments", expected_count);
    println!("   Found:    {} fragments", db_fragment_count);

    let mut all_ok = true;

    if db_fragment_count >= expected_count {
        println!("   ✓ Fragment count OK (database may have additional fragments for implicit nodes)");
    } else {
        println!("   ✗ Fragment count too low!");
        all_ok = false;
    }

    // Check if expected fragments are present
    let mut missing_fragments = 0;
    for fragment in expected.node_fragments.values() {
        if !fragment.is_empty() && !db_fragments_content.contains(fragment) {
            missing_fragments += 1;
        }
    }
    for fragment in expected.edge_fragments.values() {
        if !db_fragments_content.contains(fragment) {
            missing_fragments += 1;
        }
    }

    if missing_fragments == 0 {
        println!("   ✓ All expected fragments found in database");
    } else {
        println!("   ✗ {} expected fragments not found", missing_fragments);
        all_ok = false;
    }

    Ok(all_ok)
}

// Deserialization helpers

fn deserialize_node_value(bytes: &[u8]) -> Result<String> {
    #[derive(serde::Deserialize)]
    struct NodeCfValue(NodeSummary);

    #[derive(serde::Deserialize)]
    struct NodeSummary(DataUrl);

    #[derive(serde::Deserialize)]
    struct DataUrl(String);

    let value: NodeCfValue = rmp_serde::from_slice(bytes).context("Failed to deserialize node value")?;

    // Decode the data URL to get the actual content
    let data_url_str = &value.0.0.0;
    let parsed = data_url::DataUrl::process(data_url_str)
        .context("Failed to parse data URL")?;
    let (body, _) = parsed.decode_to_vec()
        .context("Failed to decode data URL")?;
    let content = String::from_utf8(body)
        .context("Failed to convert bytes to UTF-8")?;

    Ok(content)
}

fn deserialize_fragment_value(bytes: &[u8]) -> Result<String> {
    #[derive(serde::Deserialize)]
    struct FragmentCfValue(FragmentContent);

    #[derive(serde::Deserialize)]
    struct FragmentContent(DataUrl);

    #[derive(serde::Deserialize)]
    struct DataUrl(String);

    let value: FragmentCfValue = rmp_serde::from_slice(bytes).context("Failed to deserialize fragment value")?;

    // Decode the data URL to get the actual content
    let data_url_str = &value.0.0.0;
    let parsed = data_url::DataUrl::process(data_url_str)
        .context("Failed to parse data URL")?;
    let (body, _) = parsed.decode_to_vec()
        .context("Failed to decode data URL")?;
    let content = String::from_utf8(body)
        .context("Failed to convert bytes to UTF-8")?;

    Ok(content)
}
