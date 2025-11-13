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
use std::time::Instant;
use tokio::sync::mpsc;

// Helper type for deserialization
type IdBytes = [u8; 16];

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
            eprintln!(
                "  cat input.csv | target/release/examples/store --verify /tmp/motlie_graph_db"
            );
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
    let total_start = Instant::now();

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
    let setup_start = Instant::now();
    let (fulltext_sender, fulltext_receiver) = mpsc::channel(config.channel_buffer_size);
    let fulltext_handle = spawn_fulltext_consumer(fulltext_receiver, config.clone());

    // Create the Graph consumer that forwards to FullText with batching
    println!("  2. Creating Batched Graph consumer (forwards to FullText)");
    let (writer, graph_receiver) = create_mutation_writer(config.clone());
    let graph_handle =
        spawn_graph_consumer_with_next(graph_receiver, config, Path::new(db_path), fulltext_sender);

    let setup_duration = setup_start.elapsed();
    println!("  3. Consumer chain ready: Writer → Graph → FullText");
    println!(
        "     ⏱️  Setup time: {:.2}ms",
        setup_duration.as_secs_f64() * 1000.0
    );
    println!();

    // Keep track of node name to ID mapping for edges
    let mut node_ids: HashMap<String, Id> = HashMap::new();

    // Process CSV from stdin
    let csv_start = Instant::now();
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(io::stdin());

    let mut line_count = 0;
    let mut node_count = 0;
    let mut edge_count = 0;

    // Timing instrumentation
    let mut total_csv_parse_time = std::time::Duration::ZERO;
    let mut total_writer_send_time = std::time::Duration::ZERO;

    for result in reader.records() {
        let record_start = Instant::now();
        let record = result.context("Failed to read CSV record")?;
        line_count += 1;

        match record.len() {
            2 => {
                // node,fragment - create node and fragment
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

                // Create node
                let node_args = AddNode {
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

                // Time the writer sends
                let send_start = Instant::now();

                // Send to Graph consumer (which will forward to FullText)
                writer
                    .add_node(node_args)
                    .await
                    .context("Failed to send node to consumer chain")?;
                writer
                    .add_fragment(fragment_args)
                    .await
                    .context("Failed to send fragment to consumer chain")?;

                total_writer_send_time += send_start.elapsed();
                total_csv_parse_time += record_start.elapsed();

                node_count += 1;
                if line_count <= 10 || line_count % 1000 == 0 {
                    println!(
                        "Sent node '{}' with fragment ({} chars) to chain",
                        node_name,
                        fragment_text.len()
                    );
                }
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

                // Time the writer sends
                let send_start = Instant::now();

                // Send to Graph consumer (which will forward to FullText)
                writer
                    .add_edge(edge_args)
                    .await
                    .context("Failed to send edge to consumer chain")?;
                writer
                    .add_fragment(fragment_args)
                    .await
                    .context("Failed to send edge fragment to consumer chain")?;

                total_writer_send_time += send_start.elapsed();
                total_csv_parse_time += record_start.elapsed();

                edge_count += 1;
                if line_count <= 10 || line_count % 1000 == 0 {
                    println!(
                        "Sent edge '{}' -> '{}' (name: '{}', fragment: {} chars) to chain",
                        source_name,
                        target_name,
                        edge_name,
                        edge_fragment.len()
                    );
                }
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

    let csv_duration = csv_start.elapsed();

    println!("\nProcessed {} lines from stdin", line_count);
    println!("  Nodes: {}", node_count);
    println!("  Edges: {}", edge_count);
    println!("  Unique node names: {}", node_ids.len());
    println!(
        "  ⏱️  CSV processing time: {:.2}ms",
        csv_duration.as_secs_f64() * 1000.0
    );
    println!("\n⏱️  Detailed Timing Breakdown:");
    println!(
        "  CSV Parsing:        {:.2}ms ({:.1}%)",
        total_csv_parse_time.as_secs_f64() * 1000.0,
        (total_csv_parse_time.as_secs_f64() / csv_duration.as_secs_f64()) * 100.0
    );
    println!(
        "  Writer Send Time:   {:.2}ms ({:.1}%)",
        total_writer_send_time.as_secs_f64() * 1000.0,
        (total_writer_send_time.as_secs_f64() / csv_duration.as_secs_f64()) * 100.0
    );
    println!("\nShutting down consumer chain...");

    // Close writer to signal shutdown (this will cascade through the chain)
    let shutdown_start = Instant::now();
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

    let shutdown_duration = shutdown_start.elapsed();
    let total_duration = total_start.elapsed();

    println!("\n✅ All consumers shut down successfully");
    println!(
        "   ⏱️  Shutdown time: {:.2}ms",
        shutdown_duration.as_secs_f64() * 1000.0
    );

    // Get database size

    let db_size = get_directory_size(Path::new(db_path))?;

    println!("\n📊 Performance Summary:");
    println!(
        "   Setup:              {:.2}ms",
        setup_duration.as_secs_f64() * 1000.0
    );
    println!(
        "   CSV Processing:     {:.2}ms",
        csv_duration.as_secs_f64() * 1000.0
    );
    println!(
        "   Shutdown:           {:.2}ms",
        shutdown_duration.as_secs_f64() * 1000.0
    );
    println!("   ─────────────────────────────");
    println!(
        "   Total:              {:.2}ms ({:.2}s)",
        total_duration.as_secs_f64() * 1000.0,
        total_duration.as_secs_f64()
    );

    println!("\n📈 Throughput:");
    println!(
        "   Items stored:       {} ({} nodes + {} edges)",
        node_count + edge_count,
        node_count,
        edge_count
    );
    let items_per_sec = (node_count + edge_count) as f64 / total_duration.as_secs_f64();
    println!("   Rate:               {:.0} items/sec", items_per_sec);

    println!("\n💾 Database Size:");
    println!("   Total:              {}", format_bytes(db_size));
    println!(
        "   Per item:           {}",
        format_bytes(db_size / (node_count + edge_count) as u64)
    );

    Ok(())
}

fn get_directory_size(path: &Path) -> Result<u64> {
    let mut total_size = 0u64;

    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            if metadata.is_file() {
                total_size += metadata.len();
            } else if metadata.is_dir() {
                total_size += get_directory_size(&entry.path())?;
            }
        }
    }

    Ok(total_size)
}

fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} bytes", bytes)
    }
}

fn verify_mode_main(db_path: &str) -> Result<()> {
    let total_start = Instant::now();

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
        eprintln!(
            "  cat input.csv | target/release/examples/store {}",
            db_path
        );
        std::process::exit(1);
    }

    // Parse CSV from stdin
    println!("📄 Parsing CSV from stdin...");
    let parse_start = Instant::now();
    let expected_data = parse_csv_from_stdin()?;
    let parse_duration = parse_start.elapsed();

    println!("   Nodes: {}", expected_data.nodes.len());
    println!("   Edges: {}", expected_data.edges.len());
    println!(
        "   Total fragments: {}",
        expected_data.node_fragments.len() + expected_data.edge_fragments.len()
    );
    println!(
        "   ⏱️  Parse time: {:.2}ms",
        parse_duration.as_secs_f64() * 1000.0
    );
    println!();

    // Open database in read-only mode
    let db_open_start = Instant::now();
    let column_families = vec![
        "nodes",
        "edges",
        "fragments",
        "forward_edges",
        "reverse_edges",
    ];
    let db = DB::open_cf_for_read_only(
        &rocksdb::Options::default(),
        db_path_obj,
        &column_families,
        false,
    )
    .context("Failed to open database")?;
    let db_open_duration = db_open_start.elapsed();

    println!("✓ Database opened successfully");
    println!(
        "   ⏱️  DB open time: {:.2}ms\n",
        db_open_duration.as_secs_f64() * 1000.0
    );

    // Verify data
    let mut all_ok = true;

    println!("🔍 Verifying Nodes...");
    let verify_nodes_start = Instant::now();
    let nodes_ok = verify_nodes(&db, &expected_data)?;
    let verify_nodes_duration = verify_nodes_start.elapsed();
    println!(
        "   ⏱️  Verification time: {:.2}ms",
        verify_nodes_duration.as_secs_f64() * 1000.0
    );
    if !nodes_ok {
        all_ok = false;
    }

    println!("\n🔍 Verifying Edges...");
    let verify_edges_start = Instant::now();
    let edges_ok = verify_edges(&db, &expected_data)?;
    let verify_edges_duration = verify_edges_start.elapsed();
    println!(
        "   ⏱️  Verification time: {:.2}ms",
        verify_edges_duration.as_secs_f64() * 1000.0
    );
    if !edges_ok {
        all_ok = false;
    }

    println!("\n🔍 Verifying Fragments...");
    let verify_fragments_start = Instant::now();
    let fragments_ok = verify_fragments(&db, &expected_data)?;
    let verify_fragments_duration = verify_fragments_start.elapsed();
    println!(
        "   ⏱️  Verification time: {:.2}ms",
        verify_fragments_duration.as_secs_f64() * 1000.0
    );
    if !fragments_ok {
        all_ok = false;
    }

    let total_duration = total_start.elapsed();

    println!();
    println!("📊 Performance Summary:");
    println!(
        "   CSV Parse:          {:.2}ms",
        parse_duration.as_secs_f64() * 1000.0
    );
    println!(
        "   DB Open:            {:.2}ms",
        db_open_duration.as_secs_f64() * 1000.0
    );
    println!(
        "   Verify Nodes:       {:.2}ms",
        verify_nodes_duration.as_secs_f64() * 1000.0
    );
    println!(
        "   Verify Edges:       {:.2}ms",
        verify_edges_duration.as_secs_f64() * 1000.0
    );
    println!(
        "   Verify Fragments:   {:.2}ms",
        verify_fragments_duration.as_secs_f64() * 1000.0
    );
    println!("   ─────────────────────────────");
    println!(
        "   Total:              {:.2}ms ({:.2}s)",
        total_duration.as_secs_f64() * 1000.0,
        total_duration.as_secs_f64()
    );

    // Calculate throughput
    let total_items = expected_data.nodes.len() + expected_data.edges.len();
    let items_per_sec = total_items as f64 / total_duration.as_secs_f64();
    println!("\n📈 Throughput:");
    println!("   Items verified:     {}", total_items);
    println!("   Rate:               {:.0} items/sec", items_per_sec);

    // Get database size
    let db_size = get_directory_size(db_path_obj)?;
    println!("\n💾 Database Size:");
    println!("   Total:              {}", format_bytes(db_size));
    if total_items > 0 {
        println!(
            "   Per item:           {}",
            format_bytes(db_size / total_items as u64)
        );
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

                edge_fragments.insert(
                    (source.clone(), target.clone(), edge_name.clone()),
                    fragment.clone(),
                );
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

    // Build map of node ID -> node name and collect all node data
    let mut db_node_count = 0;
    let iter = db.iterator_cf(cf, rocksdb::IteratorMode::Start);
    let mut db_node_names: HashSet<String> = HashSet::new(); // Just collect names

    for item in iter {
        let (_key, value) = item.context("Failed to read from nodes CF")?;
        db_node_count += 1;

        // Deserialize to extract node name (now directly from the tuple)
        if let Ok((node_name, _markdown_content)) = deserialize_node_value(&value) {
            db_node_names.insert(node_name);
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
        println!("   ✓ All expected node names found in database");
    } else {
        println!("   ✗ Missing node names: {:?}", missing_nodes);
        all_ok = false;
    }

    Ok(all_ok)
}

fn verify_edges(db: &DB, expected: &ExpectedData) -> Result<bool> {
    // First build a map of node IDs to names
    let nodes_cf = db.cf_handle("nodes").context("Nodes CF not found")?;
    let mut id_to_name: HashMap<IdBytes, String> = HashMap::new();

    for item in db.iterator_cf(nodes_cf, rocksdb::IteratorMode::Start) {
        let (key, value) = item.context("Failed to read from nodes CF")?;
        let node_id = deserialize_node_id(&key)?;
        if let Ok((node_name, _markdown_content)) = deserialize_node_value(&value) {
            id_to_name.insert(node_id, node_name);
        }
    }

    // Now verify edges using forward_edges CF which has (source_id, target_id, edge_name) as key
    let forward_edges_cf = db
        .cf_handle("forward_edges")
        .context("ForwardEdges CF not found")?;

    let mut db_edge_count = 0;
    let mut db_edges: HashSet<(String, String, String)> = HashSet::new(); // (source_name, target_name, edge_name)

    for item in db.iterator_cf(forward_edges_cf, rocksdb::IteratorMode::Start) {
        let (key, _value) = item.context("Failed to read from forward_edges CF")?;
        db_edge_count += 1;

        // Deserialize key to get source_id, target_id, edge_name
        // The key contains the canonical edge information
        let (source_id, target_id, edge_name) = deserialize_forward_edge_key(&key)?;

        // Map IDs to names
        if let (Some(source_name), Some(target_name)) =
            (id_to_name.get(&source_id), id_to_name.get(&target_id))
        {
            db_edges.insert((source_name.clone(), target_name.clone(), edge_name));
        }
    }

    let expected_count = expected.edges.len();
    println!("   Expected: {} edges", expected_count);
    println!("   Found:    {} edges", db_edge_count);

    let mut all_ok = true;

    if db_edge_count == expected_count {
        println!("   ✓ Edge count matches");
    } else {
        println!("   ✗ Edge count mismatch!");
        all_ok = false;
    }

    // Check if all expected edges are present
    let mut missing_edges = Vec::new();

    for edge in &expected.edges {
        let edge_tuple = (edge.source.clone(), edge.target.clone(), edge.name.clone());
        if !db_edges.contains(&edge_tuple) {
            missing_edges.push(format!(
                "{} -> {} ({})",
                edge.source, edge.target, edge.name
            ));
        }
    }

    if missing_edges.is_empty() {
        println!("   ✓ All expected edges found with correct source, target, and name");
    } else {
        println!("   ✗ Missing or mismatched edges:");
        for edge in &missing_edges {
            println!("      - {}", edge);
        }
        all_ok = false;
    }

    Ok(all_ok)
}

fn verify_fragments(db: &DB, expected: &ExpectedData) -> Result<bool> {
    let cf = db
        .cf_handle("fragments")
        .context("Fragments CF not found")?;

    // Count fragments in database and collect all content
    let mut db_fragment_count = 0;
    let mut db_fragments_content = HashSet::new();
    let iter = db.iterator_cf(cf, rocksdb::IteratorMode::Start);

    for item in iter {
        let (_, value) = item.context("Failed to read from fragments CF")?;
        db_fragment_count += 1;

        if let Ok(content) = deserialize_fragment_value(&value) {
            // Fragments are stored as markdown, but we want to compare the raw content
            // The content should match what was in the CSV
            db_fragments_content.insert(content);
        }
    }

    let expected_count = expected.node_fragments.len() + expected.edge_fragments.len();
    println!("   Expected: at least {} fragments", expected_count);
    println!("   Found:    {} fragments", db_fragment_count);

    let mut all_ok = true;

    if db_fragment_count >= expected_count {
        println!(
            "   ✓ Fragment count OK (database may have additional fragments for implicit nodes)"
        );
    } else {
        println!("   ✗ Fragment count too low!");
        all_ok = false;
    }

    // Check if expected fragment content is present
    let mut missing_fragments = Vec::new();

    for (node_name, fragment) in &expected.node_fragments {
        if !fragment.is_empty() && !db_fragments_content.contains(fragment) {
            missing_fragments.push(format!(
                "Node '{}': {:?}",
                node_name,
                if fragment.len() > 50 {
                    &fragment[..50]
                } else {
                    fragment
                }
            ));
        }
    }

    for ((source, target, edge_name), fragment) in &expected.edge_fragments {
        if !db_fragments_content.contains(fragment) {
            missing_fragments.push(format!(
                "Edge '{} -> {} ({})': {:?}",
                source,
                target,
                edge_name,
                if fragment.len() > 50 {
                    &fragment[..50]
                } else {
                    fragment
                }
            ));
        }
    }

    if missing_fragments.is_empty() {
        println!("   ✓ All expected fragment content found in database");
    } else {
        println!(
            "   ✗ {} expected fragments not found:",
            missing_fragments.len()
        );
        for (i, frag) in missing_fragments.iter().take(5).enumerate() {
            println!("      {}. {}", i + 1, frag);
        }
        if missing_fragments.len() > 5 {
            println!("      ... and {} more", missing_fragments.len() - 5);
        }
        all_ok = false;
    }

    Ok(all_ok)
}

// Deserialization helpers

fn deserialize_node_id(bytes: &[u8]) -> Result<IdBytes> {
    // Keys are now stored as raw 16-byte arrays, not MessagePack
    if bytes.len() != 16 {
        anyhow::bail!("Invalid node ID length: expected 16, got {}", bytes.len());
    }
    let mut id_bytes = [0u8; 16];
    id_bytes.copy_from_slice(bytes);
    Ok(id_bytes)
}

fn deserialize_node_value(bytes: &[u8]) -> Result<(String, String)> {
    #[derive(serde::Deserialize)]
    struct NodeCfValue(String, NodeSummary);

    #[derive(serde::Deserialize)]
    struct NodeSummary(DataUrl);

    #[derive(serde::Deserialize)]
    struct DataUrl(String);

    // Decompress LZ4 first
    let decompressed = lz4::block::decompress(bytes, None)
        .context("Failed to decompress node value with LZ4")?;

    let value: NodeCfValue =
        rmp_serde::from_slice(&decompressed).context("Failed to deserialize node value")?;

    let node_name = value.0;
    // Decode the data URL to get the actual content
    let data_url_str = &value.1 .0 .0;
    let parsed = data_url::DataUrl::process(data_url_str).context("Failed to parse data URL")?;
    let (body, _) = parsed
        .decode_to_vec()
        .context("Failed to decode data URL")?;
    let content = String::from_utf8(body).context("Failed to convert bytes to UTF-8")?;

    Ok((node_name, content))
}

fn deserialize_forward_edge_key(bytes: &[u8]) -> Result<(IdBytes, IdBytes, String)> {
    // Keys are now stored as: [src_id (16)] + [dst_id (16)] + [name UTF-8 bytes]
    if bytes.len() < 32 {
        anyhow::bail!(
            "Invalid forward edge key length: expected >= 32, got {}",
            bytes.len()
        );
    }

    let mut src_id_bytes = [0u8; 16];
    src_id_bytes.copy_from_slice(&bytes[0..16]);

    let mut dst_id_bytes = [0u8; 16];
    dst_id_bytes.copy_from_slice(&bytes[16..32]);

    let name_bytes = &bytes[32..];
    let name = String::from_utf8(name_bytes.to_vec()).context("Invalid UTF-8 in edge name")?;

    Ok((src_id_bytes, dst_id_bytes, name))
}

fn deserialize_edge_value(bytes: &[u8]) -> Result<String> {
    #[derive(serde::Deserialize)]
    struct EdgeCfValue(EdgeSummary);

    #[derive(serde::Deserialize)]
    struct EdgeSummary(DataUrl);

    #[derive(serde::Deserialize)]
    struct DataUrl(String);

    // Decompress LZ4 first
    let decompressed = lz4::block::decompress(bytes, None)
        .context("Failed to decompress edge value with LZ4")?;

    let value: EdgeCfValue =
        rmp_serde::from_slice(&decompressed).context("Failed to deserialize edge value")?;

    // Decode the data URL to get the actual content
    let data_url_str = &value.0 .0 .0;
    let parsed = data_url::DataUrl::process(data_url_str).context("Failed to parse data URL")?;
    let (body, _) = parsed
        .decode_to_vec()
        .context("Failed to decode data URL")?;
    let content = String::from_utf8(body).context("Failed to convert bytes to UTF-8")?;

    Ok(content)
}

fn deserialize_fragment_value(bytes: &[u8]) -> Result<String> {
    #[derive(serde::Deserialize)]
    struct FragmentCfValue(FragmentContent);

    #[derive(serde::Deserialize)]
    struct FragmentContent(DataUrl);

    #[derive(serde::Deserialize)]
    struct DataUrl(String);

    // Decompress LZ4 first
    let decompressed = lz4::block::decompress(bytes, None)
        .context("Failed to decompress fragment value with LZ4")?;

    let value: FragmentCfValue =
        rmp_serde::from_slice(&decompressed).context("Failed to deserialize fragment value")?;

    // Decode the data URL to get the actual content
    let data_url_str = &value.0 .0 .0;
    let parsed = data_url::DataUrl::process(data_url_str).context("Failed to parse data URL")?;
    let (body, _) = parsed
        .decode_to_vec()
        .context("Failed to decode data URL")?;
    let content = String::from_utf8(body).context("Failed to convert bytes to UTF-8")?;

    Ok(content)
}
