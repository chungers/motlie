//! build_graph - Create a sample graph database with versioned nodes, edges, and active periods.
//!
//! Populates a RocksDB database with an evolving org chart to demonstrate
//! `motlie db list` and `motlie db scan` with temporal filtering and version history.
//!
//! Usage:
//!   cargo run --manifest-path bins/motlie/examples/build_graph/Cargo.toml -- <db_path>

use anyhow::Result;
use motlie_db::graph::mutation::{AddEdge, AddNode, UpdateEdge, UpdateNode};
use motlie_db::graph::schema::{ActivePeriod, EdgeSummary, NodeSummary};
use motlie_db::graph::writer::{spawn_mutation_consumer_with_storage, WriterConfig};
use motlie_db::graph::Storage;
use motlie_db::writer::Runnable;
use motlie_db::{Id, TimestampMilli};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

// Well-known dates as milliseconds since Unix epoch
const JAN_2025: u64 = 1_735_689_600_000; // 2025-01-01T00:00:00Z
const APR_2025: u64 = 1_743_465_600_000; // 2025-04-01T00:00:00Z
const JUL_2025: u64 = 1_751_328_000_000; // 2025-07-01T00:00:00Z

fn ts(millis: u64) -> TimestampMilli {
    TimestampMilli(millis)
}

#[tokio::main]
async fn main() -> Result<()> {
    let db_path = std::env::args()
        .nth(1)
        .expect("Usage: build_graph <db_path>");

    // Setup storage
    let mut storage = Storage::readwrite(Path::new(&db_path));
    storage.ready()?;
    let storage = Arc::new(storage);

    let config = WriterConfig { channel_buffer_size: 100 };
    let (writer, writer_handle) = spawn_mutation_consumer_with_storage(storage.clone(), config);

    let pause = Duration::from_millis(50);

    // ── Phase 1: Initial org (Jan 2025) ────────────────────────────────

    eprintln!("Phase 1: Creating initial organization (Jan 2025)");

    let alice_id = Id::new();
    let bob_id = Id::new();
    let carol_id = Id::new();

    // Alice: Engineer, active from Jan 2025 onwards
    AddNode {
        id: alice_id,
        ts_millis: TimestampMilli::now(),
        name: "Alice".to_string(),
        valid_range: ActivePeriod::active_from(ts(JAN_2025)),
        summary: NodeSummary::from_text("Software Engineer - Backend team"),
    }
    .run(&writer)
    .await?;
    writer.flush().await?;
    tokio::time::sleep(pause).await;

    // Bob: Manager, always active (no active period constraint)
    AddNode {
        id: bob_id,
        ts_millis: TimestampMilli::now(),
        name: "Bob".to_string(),
        valid_range: None, // Always active
        summary: NodeSummary::from_text("Engineering Manager"),
    }
    .run(&writer)
    .await?;
    writer.flush().await?;
    tokio::time::sleep(pause).await;

    // Carol: Intern, active only during internship (Jan-Jun 2025)
    AddNode {
        id: carol_id,
        ts_millis: TimestampMilli::now(),
        name: "Carol".to_string(),
        valid_range: ActivePeriod::active_between(ts(JAN_2025), ts(JUL_2025)),
        summary: NodeSummary::from_text("Summer Intern - Backend team"),
    }
    .run(&writer)
    .await?;
    writer.flush().await?;
    tokio::time::sleep(pause).await;

    // Edges: reporting structure
    AddEdge {
        source_node_id: alice_id,
        target_node_id: bob_id,
        ts_millis: TimestampMilli::now(),
        name: "reports_to".to_string(),
        summary: EdgeSummary::from_text("Direct report"),
        weight: Some(1.0),
        valid_range: ActivePeriod::active_from(ts(JAN_2025)),
    }
    .run(&writer)
    .await?;
    writer.flush().await?;
    tokio::time::sleep(pause).await;

    AddEdge {
        source_node_id: carol_id,
        target_node_id: bob_id,
        ts_millis: TimestampMilli::now(),
        name: "reports_to".to_string(),
        summary: EdgeSummary::from_text("Intern supervisor"),
        weight: Some(1.0),
        valid_range: ActivePeriod::active_between(ts(JAN_2025), ts(JUL_2025)),
    }
    .run(&writer)
    .await?;
    writer.flush().await?;
    tokio::time::sleep(pause).await;

    AddEdge {
        source_node_id: alice_id,
        target_node_id: carol_id,
        ts_millis: TimestampMilli::now(),
        name: "mentors".to_string(),
        summary: EdgeSummary::from_text("Technical mentorship"),
        weight: Some(0.8),
        valid_range: ActivePeriod::active_between(ts(JAN_2025), ts(JUL_2025)),
    }
    .run(&writer)
    .await?;
    writer.flush().await?;
    tokio::time::sleep(pause).await;

    eprintln!("  Created 3 nodes (Alice, Bob, Carol) and 3 edges");

    // ── Phase 2: Alice promoted (Apr 2025) ─────────────────────────────

    eprintln!("Phase 2: Alice promoted to Senior Engineer (Apr 2025)");

    UpdateNode {
        id: alice_id,
        expected_version: 1,
        new_active_period: None, // No change
        new_summary: Some(NodeSummary::from_text(
            "Senior Software Engineer - Backend team lead",
        )),
    }
    .run(&writer)
    .await?;
    writer.flush().await?;
    tokio::time::sleep(pause).await;

    // Stronger reporting relationship
    UpdateEdge {
        src_id: alice_id,
        dst_id: bob_id,
        name: "reports_to".to_string(),
        expected_version: 1,
        new_weight: Some(Some(0.9)),
        new_active_period: None,
        new_summary: Some(EdgeSummary::from_text("Senior direct report")),
    }
    .run(&writer)
    .await?;
    writer.flush().await?;
    tokio::time::sleep(pause).await;

    eprintln!("  Updated Alice (v1->v2) and Alice->Bob edge (v1->v2)");

    // ── Phase 3: Carol leaves, Dave joins (Jul 2025) ───────────────────

    eprintln!("Phase 3: Dave joins, Alice becomes Tech Lead (Jul 2025)");

    let dave_id = Id::new();

    // Dave: New hire, active from Jul 2025
    AddNode {
        id: dave_id,
        ts_millis: TimestampMilli::now(),
        name: "Dave".to_string(),
        valid_range: ActivePeriod::active_from(ts(JUL_2025)),
        summary: NodeSummary::from_text("Junior Engineer - Backend team"),
    }
    .run(&writer)
    .await?;
    writer.flush().await?;
    tokio::time::sleep(pause).await;

    // Alice becomes Tech Lead (v3), active period widened to include Apr start
    UpdateNode {
        id: alice_id,
        expected_version: 2,
        new_active_period: Some(ActivePeriod::active_from(ts(APR_2025))),
        new_summary: Some(NodeSummary::from_text("Tech Lead - Backend team")),
    }
    .run(&writer)
    .await?;
    writer.flush().await?;
    tokio::time::sleep(pause).await;

    // Dave reports to Alice
    AddEdge {
        source_node_id: dave_id,
        target_node_id: alice_id,
        ts_millis: TimestampMilli::now(),
        name: "reports_to".to_string(),
        summary: EdgeSummary::from_text("Direct report to tech lead"),
        weight: Some(1.0),
        valid_range: ActivePeriod::active_from(ts(JUL_2025)),
    }
    .run(&writer)
    .await?;
    writer.flush().await?;
    tokio::time::sleep(pause).await;

    eprintln!("  Created Dave, updated Alice (v2->v3), added Dave->Alice edge");

    // Print IDs for reference (to stdout so the bash script can capture them)
    println!("ALICE={}", alice_id);
    println!("BOB={}", bob_id);
    println!("CAROL={}", carol_id);
    println!("DAVE={}", dave_id);

    // Shutdown writer
    drop(writer);
    writer_handle.await??;

    eprintln!();
    eprintln!("Database written to: {}", db_path);

    Ok(())
}
