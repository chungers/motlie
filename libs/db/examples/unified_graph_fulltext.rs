use std::time::Duration;

use anyhow::Result;
use motlie_db::mutation::{
    AddEdge, AddNode, EdgeSummary, NodeSummary, Runnable as MutationRunnable,
};
use motlie_db::query::{NodeById, Nodes, Runnable as QueryRunnable};
use motlie_db::{Id, Storage, StorageConfig, TimestampMilli};

#[tokio::main]
async fn main() -> Result<()> {
    let base_path = std::env::temp_dir().join(format!(
        "motlie-db-unified-example-{}",
        Id::new()
    ));

    let storage = Storage::readwrite(&base_path);
    let handles = storage.ready(StorageConfig::default())?;

    let alice_id = Id::new();
    let bob_id = Id::new();

    AddNode {
        id: alice_id,
        ts_millis: TimestampMilli::now(),
        name: "Alice".to_string(),
        summary: NodeSummary::from_text("Graph engineer working on search"),
        valid_range: None,
    }
    .run(handles.writer())
    .await?;

    AddNode {
        id: bob_id,
        ts_millis: TimestampMilli::now(),
        name: "Bob".to_string(),
        summary: NodeSummary::from_text("Database engineer building indexing pipelines"),
        valid_range: None,
    }
    .run(handles.writer())
    .await?;

    AddEdge {
        source_node_id: alice_id,
        target_node_id: bob_id,
        ts_millis: TimestampMilli::now(),
        name: "collaborates_with".to_string(),
        summary: EdgeSummary::from_text("Alice and Bob collaborate on search infrastructure"),
        weight: Some(1.0),
        valid_range: None,
    }
    .run(handles.writer())
    .await?;

    // Guarantees graph commit visibility.
    handles.writer().flush().await?;

    let timeout = Duration::from_secs(5);
    let (name, summary, version) = NodeById::new(alice_id, None)
        .run(handles.reader(), timeout)
        .await?;
    println!(
        "NodeById => id={}, name={}, summary='{}', version={}",
        alice_id,
        name,
        summary.as_ref(),
        version
    );

    // Fulltext indexing is asynchronous in the root pipeline, so retry briefly.
    let mut fulltext_nodes = Vec::new();
    for _ in 0..20 {
        fulltext_nodes = Nodes::new("search".to_string(), 10)
            .run(handles.reader(), timeout)
            .await?;
        if !fulltext_nodes.is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    println!(
        "Nodes(\"search\") => {} hydrated results",
        fulltext_nodes.len()
    );

    handles.shutdown().await?;

    if let Err(err) = std::fs::remove_dir_all(&base_path) {
        eprintln!(
            "warning: failed to remove temporary path {}: {}",
            base_path.display(),
            err
        );
    }

    Ok(())
}
