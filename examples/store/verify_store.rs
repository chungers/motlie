use anyhow::{Context, Result};
use motlie_db::{Id, TimestampMilli};
use rocksdb::{IteratorMode, DB};
use std::path::Path;

fn main() -> Result<()> {
    println!("Motlie Store Verifier");
    println!("====================");
    println!("Reading from: /tmp/motlie_graph_db");
    println!();

    let db_path = Path::new("/tmp/motlie_graph_db");

    // Check if database exists
    if !db_path.exists() {
        println!("❌ Database does not exist at /tmp/motlie_graph_db");
        println!("Run the store example first with some input data:");
        println!("  echo 'alice,Alice is a researcher' | cargo run --example store");
        return Ok(());
    }

    // Define column families
    let column_families = vec!["nodes", "edges", "fragments", "forward_edges", "reverse_edges"];

    // Open database in read-only mode
    let db = DB::open_cf_for_read_only(
        &rocksdb::Options::default(),
        db_path,
        &column_families,
        false,
    )
    .context("Failed to open database")?;

    println!("✓ Database opened successfully\n");

    // Verify Nodes column family
    println!("📦 Nodes Column Family:");
    println!("─────────────────────────");
    verify_nodes(&db)?;

    // Verify Edges column family
    println!("\n📦 Edges Column Family:");
    println!("──────────────────────────");
    verify_edges(&db)?;

    // Verify Fragments column family
    println!("\n📦 Fragments Column Family:");
    println!("──────────────────────────────");
    verify_fragments(&db)?;

    // Verify ForwardEdges column family
    println!("\n📦 Forward Edges Column Family:");
    println!("──────────────────────────────────");
    verify_forward_edges(&db)?;

    // Verify ReverseEdges column family
    println!("\n📦 Reverse Edges Column Family:");
    println!("──────────────────────────────────");
    verify_reverse_edges(&db)?;

    println!("\n✓ Verification complete!");

    Ok(())
}

fn verify_nodes(db: &DB) -> Result<()> {
    let cf = db
        .cf_handle("nodes")
        .context("Nodes column family not found")?;

    let mut count = 0;
    let iter = db.iterator_cf(cf, IteratorMode::Start);

    for item in iter {
        let (key, value) = item.context("Failed to read from nodes CF")?;
        count += 1;

        // Try to deserialize the key as an Id
        match deserialize_id(&key) {
            Ok(id) => {
                // Try to deserialize value as NodeSummary
                match deserialize_node_value(&value) {
                    Ok(summary) => {
                        println!("  Node #{}: ID={}", count, id.as_str());
                        println!("    Value: {}", truncate(&summary, 100));
                    }
                    Err(_) => {
                        println!("  Node #{}: ID={}", count, id.as_str());
                        println!("    Value: {} bytes (raw)", value.len());
                    }
                }
            }
            Err(_) => {
                println!("  Node #{}: Key={} bytes, Value={} bytes", count, key.len(), value.len());
            }
        }
    }

    if count == 0 {
        println!("  (empty)");
    } else {
        println!("\n  Total: {} nodes", count);
    }

    Ok(())
}

fn verify_edges(db: &DB) -> Result<()> {
    let cf = db
        .cf_handle("edges")
        .context("Edges column family not found")?;

    let mut count = 0;
    let iter = db.iterator_cf(cf, IteratorMode::Start);

    for item in iter {
        let (key, value) = item.context("Failed to read from edges CF")?;
        count += 1;

        match deserialize_id(&key) {
            Ok(id) => {
                match deserialize_edge_value(&value) {
                    Ok(summary) => {
                        println!("  Edge #{}: ID={}", count, id.as_str());
                        println!("    Value: {}", truncate(&summary, 100));
                    }
                    Err(_) => {
                        println!("  Edge #{}: ID={}", count, id.as_str());
                        println!("    Value: {} bytes (raw)", value.len());
                    }
                }
            }
            Err(_) => {
                println!("  Edge #{}: Key={} bytes, Value={} bytes", count, key.len(), value.len());
            }
        }
    }

    if count == 0 {
        println!("  (empty)");
    } else {
        println!("\n  Total: {} edges", count);
    }

    Ok(())
}

fn verify_fragments(db: &DB) -> Result<()> {
    let cf = db
        .cf_handle("fragments")
        .context("Fragments column family not found")?;

    let mut count = 0;
    let iter = db.iterator_cf(cf, IteratorMode::Start);

    for item in iter {
        let (key, value) = item.context("Failed to read from fragments CF")?;
        count += 1;

        match deserialize_fragment_key(&key) {
            Ok((id, timestamp)) => {
                match deserialize_fragment_value(&value) {
                    Ok(content) => {
                        println!("  Fragment #{}: ID={}, Timestamp={}", count, id.as_str(), timestamp.0);
                        println!("    Content: {}", truncate(&content, 100));
                    }
                    Err(_) => {
                        println!("  Fragment #{}: ID={}, Timestamp={}", count, id.as_str(), timestamp.0);
                        println!("    Content: {} bytes (raw)", value.len());
                    }
                }
            }
            Err(_) => {
                println!("  Fragment #{}: Key={} bytes, Value={} bytes", count, key.len(), value.len());
            }
        }
    }

    if count == 0 {
        println!("  (empty)");
    } else {
        println!("\n  Total: {} fragments", count);
    }

    Ok(())
}

fn verify_forward_edges(db: &DB) -> Result<()> {
    let cf = db
        .cf_handle("forward_edges")
        .context("ForwardEdges column family not found")?;

    let mut count = 0;
    let iter = db.iterator_cf(cf, IteratorMode::Start);

    for item in iter {
        let (key, value) = item.context("Failed to read from forward_edges CF")?;
        count += 1;

        match deserialize_forward_edge_key(&key) {
            Ok((source, dest, name)) => {
                match deserialize_edge_value(&value) {
                    Ok(summary) => {
                        println!("  ForwardEdge #{}: {} -> {} ({})", count, source.as_str(), dest.as_str(), name);
                        println!("    Value: {}", truncate(&summary, 100));
                    }
                    Err(_) => {
                        println!("  ForwardEdge #{}: {} -> {} ({})", count, source.as_str(), dest.as_str(), name);
                        println!("    Value: {} bytes (raw)", value.len());
                    }
                }
            }
            Err(_) => {
                println!("  ForwardEdge #{}: Key={} bytes, Value={} bytes", count, key.len(), value.len());
            }
        }
    }

    if count == 0 {
        println!("  (empty)");
    } else {
        println!("\n  Total: {} forward edges", count);
    }

    Ok(())
}

fn verify_reverse_edges(db: &DB) -> Result<()> {
    let cf = db
        .cf_handle("reverse_edges")
        .context("ReverseEdges column family not found")?;

    let mut count = 0;
    let iter = db.iterator_cf(cf, IteratorMode::Start);

    for item in iter {
        let (key, value) = item.context("Failed to read from reverse_edges CF")?;
        count += 1;

        match deserialize_reverse_edge_key(&key) {
            Ok((dest, source, name)) => {
                match deserialize_edge_value(&value) {
                    Ok(summary) => {
                        println!("  ReverseEdge #{}: {} <- {} ({})", count, dest.as_str(), source.as_str(), name);
                        println!("    Value: {}", truncate(&summary, 100));
                    }
                    Err(_) => {
                        println!("  ReverseEdge #{}: {} <- {} ({})", count, dest.as_str(), source.as_str(), name);
                        println!("    Value: {} bytes (raw)", value.len());
                    }
                }
            }
            Err(_) => {
                println!("  ReverseEdge #{}: Key={} bytes, Value={} bytes", count, key.len(), value.len());
            }
        }
    }

    if count == 0 {
        println!("  (empty)");
    } else {
        println!("\n  Total: {} reverse edges", count);
    }

    Ok(())
}

// Deserialization helpers using MessagePack (rmp_serde)

fn deserialize_id(bytes: &[u8]) -> Result<Id> {
    #[derive(serde::Deserialize)]
    struct NodeCfKey(Id);

    let key: NodeCfKey = rmp_serde::from_slice(bytes).context("Failed to deserialize ID")?;
    Ok(key.0)
}

fn deserialize_node_value(bytes: &[u8]) -> Result<String> {
    #[derive(serde::Deserialize)]
    struct NodeCfValue(NodeSummary);

    #[derive(serde::Deserialize)]
    struct NodeSummary(String);

    let value: NodeCfValue = rmp_serde::from_slice(bytes).context("Failed to deserialize node value")?;
    Ok(value.0.0)
}

fn deserialize_edge_value(bytes: &[u8]) -> Result<String> {
    #[derive(serde::Deserialize)]
    struct EdgeCfValue(EdgeSummary);

    #[derive(serde::Deserialize)]
    struct EdgeSummary(String);

    let value: EdgeCfValue = rmp_serde::from_slice(bytes).context("Failed to deserialize edge value")?;
    Ok(value.0.0)
}

fn deserialize_fragment_key(bytes: &[u8]) -> Result<(Id, TimestampMilli)> {
    #[derive(serde::Deserialize)]
    struct FragmentCfKey(Id, TimestampMilli);

    let key: FragmentCfKey = rmp_serde::from_slice(bytes).context("Failed to deserialize fragment key")?;
    Ok((key.0, key.1))
}

fn deserialize_fragment_value(bytes: &[u8]) -> Result<String> {
    #[derive(serde::Deserialize)]
    struct FragmentCfValue(FragmentContent);

    #[derive(serde::Deserialize)]
    struct FragmentContent(String);

    let value: FragmentCfValue = rmp_serde::from_slice(bytes).context("Failed to deserialize fragment value")?;
    Ok(value.0.0)
}

fn deserialize_forward_edge_key(bytes: &[u8]) -> Result<(Id, Id, String)> {
    #[derive(serde::Deserialize)]
    struct ForwardEdgeCfKey(EdgeSourceId, EdgeDestinationId, EdgeName);

    #[derive(serde::Deserialize)]
    struct EdgeSourceId(Id);

    #[derive(serde::Deserialize)]
    struct EdgeDestinationId(Id);

    #[derive(serde::Deserialize)]
    struct EdgeName(String);

    let key: ForwardEdgeCfKey = rmp_serde::from_slice(bytes).context("Failed to deserialize forward edge key")?;
    Ok((key.0.0, key.1.0, key.2.0))
}

fn deserialize_reverse_edge_key(bytes: &[u8]) -> Result<(Id, Id, String)> {
    #[derive(serde::Deserialize)]
    struct ReverseEdgeCfKey(EdgeDestinationId, EdgeSourceId, EdgeName);

    #[derive(serde::Deserialize)]
    struct EdgeSourceId(Id);

    #[derive(serde::Deserialize)]
    struct EdgeDestinationId(Id);

    #[derive(serde::Deserialize)]
    struct EdgeName(String);

    let key: ReverseEdgeCfKey = rmp_serde::from_slice(bytes).context("Failed to deserialize reverse edge key")?;
    Ok((key.0.0, key.1.0, key.2.0))
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}
