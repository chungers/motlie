use crate::common::concurrent_test_utils::{Metrics, TestContext};
use motlie_db::graph::mutation::{AddEdge, AddNode};
use motlie_db::graph::schema::{EdgeSummary, NodeSummary};
use motlie_db::graph::writer::{create_mutation_writer, spawn_mutation_consumer};
use motlie_db::writer::Runnable as MutationRunnable;
use motlie_db::{Id, TimestampMilli};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Writer task: inserts nodes and edges
///
/// This task creates a mutation writer and inserts the specified number of nodes,
/// with each node having the specified number of edges.
///
/// # Arguments
/// * `db_path` - Path to the database
/// * `context` - Shared test context for coordination
/// * `num_nodes` - Number of nodes to insert
/// * `num_edges_per_node` - Number of edges to create for each node
///
/// # Returns
/// Metrics collected during the write operations
pub async fn writer_task(
    db_path: std::path::PathBuf,
    context: Arc<TestContext>,
    num_nodes: usize,
    num_edges_per_node: usize,
) -> Metrics {
    let mut metrics = Metrics::new();

    // Create writer
    let (writer, writer_rx) = create_mutation_writer(Default::default());

    // Spawn write consumer
    let consumer_handle = spawn_mutation_consumer(writer_rx, Default::default(), &db_path);

    // Insert nodes
    for i in 0..num_nodes {
        if context.should_stop() {
            break;
        }

        let node_id = Id::new();
        let node_name = format!("node_{}", i);

        let start = Instant::now();
        let result = AddNode {
            id: node_id,
            ts_millis: TimestampMilli::now(),
            name: node_name.clone(),
            valid_range: None,
            summary: NodeSummary::from_text(&format!("summary {}", i)),
        }
        .run(&writer)
        .await;

        let latency_us = start.elapsed().as_micros() as u64;

        match result {
            Ok(_) => {
                metrics.record_success(latency_us);
                context.add_written_node(node_id).await;

                // Insert edges from this node
                for j in 0..num_edges_per_node {
                    let target_id = Id::new(); // Random target for now
                    let edge_name = format!("edge_{}_{}", i, j);

                    let start = Instant::now();
                    let result = AddEdge {
                        source_node_id: node_id,
                        target_node_id: target_id,
                        ts_millis: TimestampMilli::now(),
                        name: edge_name.clone(),
                        valid_range: None,
                        summary: EdgeSummary::from_text(&format!("Edge summary {}", edge_name)),
                        weight: None,
                    }
                    .run(&writer)
                    .await;

                    let latency_us = start.elapsed().as_micros() as u64;

                    match result {
                        Ok(_) => {
                            metrics.record_success(latency_us);
                            // Note: edges don't have their own ID, they're identified by (src, dst, name)
                            // so we can't track edge_id anymore
                        }
                        Err(_) => metrics.record_error(),
                    }
                }
            }
            Err(_) => metrics.record_error(),
        }

        // Small delay to simulate realistic write pattern
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // Close writer and wait for consumer
    drop(writer);
    let _ = consumer_handle.await;

    metrics
}
