use anyhow::{Context, Result};
use csv::ReaderBuilder;
use rocksdb::DB;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs::File;
use std::path::Path;

fn main() -> Result<()> {
    println!("Motlie Store Verifier");
    println!("====================");
    println!();

    // Parse command-line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: {} <csv_file>", args[0]);
        eprintln!();
        eprintln!("Example:");
        eprintln!("  cargo run --example verify_store /tmp/test_data.csv");
        eprintln!();
        eprintln!("This will verify that the data in /tmp/motlie_graph_db");
        eprintln!("matches what was sent from the CSV file.");
        std::process::exit(1);
    }

    let csv_path = &args[1];
    let db_path = Path::new("/tmp/motlie_graph_db");

    println!("CSV file: {}", csv_path);
    println!("Database: /tmp/motlie_graph_db");
    println!();

    // Check if CSV file exists
    if !Path::new(csv_path).exists() {
        eprintln!("❌ CSV file does not exist: {}", csv_path);
        std::process::exit(1);
    }

    // Check if database exists
    if !db_path.exists() {
        eprintln!("❌ Database does not exist at /tmp/motlie_graph_db");
        eprintln!("Run the store example first:");
        eprintln!("  cat {} | cargo run --example store", csv_path);
        std::process::exit(1);
    }

    // Parse CSV file to extract expected data
    println!("📄 Parsing CSV file...");
    let expected_data = parse_csv(csv_path)?;
    println!("   Nodes: {}", expected_data.nodes.len());
    println!("   Edges: {}", expected_data.edges.len());
    println!("   Total fragments: {}", expected_data.node_fragments.len() + expected_data.edge_fragments.len());
    println!();

    // Open database in read-only mode
    let column_families = vec!["nodes", "edges", "fragments", "forward_edges", "reverse_edges"];
    let db = DB::open_cf_for_read_only(
        &rocksdb::Options::default(),
        db_path,
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

fn parse_csv(csv_path: &str) -> Result<ExpectedData> {
    let file = File::open(csv_path).context("Failed to open CSV file")?;
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(file);

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
    struct NodeSummary(String);

    let value: NodeCfValue = rmp_serde::from_slice(bytes).context("Failed to deserialize node value")?;
    Ok(value.0.0)
}

fn deserialize_fragment_value(bytes: &[u8]) -> Result<String> {
    #[derive(serde::Deserialize)]
    struct FragmentCfValue(FragmentContent);

    #[derive(serde::Deserialize)]
    struct FragmentContent(String);

    let value: FragmentCfValue = rmp_serde::from_slice(bytes).context("Failed to deserialize fragment value")?;
    Ok(value.0.0)
}
