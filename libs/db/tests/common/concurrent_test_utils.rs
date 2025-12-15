/// Shared test utilities for concurrent read/write integration tests
///
/// This module provides common infrastructure for testing concurrent database operations:
/// - Metrics collection for read and write operations
/// - Test context for coordinating writer and readers
/// - Writer task implementation
///
/// Used by:
/// - test_concurrent_read_write.rs (readonly instances)
/// - test_concurrent_secondary.rs (secondary instances with catch-up)

use motlie_db::graph::mutation::{AddEdge, AddNode};
use motlie_db::writer::Runnable as MutationRunnable;
use motlie_db::graph::schema::{EdgeSummary, NodeSummary};
use motlie_db::graph::writer::{create_mutation_writer, spawn_mutation_consumer};
use motlie_db::{Id, TimestampMilli};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

/// Metrics collected during read or write operations
#[derive(Debug, Clone)]
pub struct Metrics {
    /// Number of successful operations
    pub success_count: u64,
    /// Number of failed operations
    pub error_count: u64,
    /// Sum of operation latencies in microseconds
    pub total_latency_us: u64,
    /// Minimum latency in microseconds
    pub min_latency_us: u64,
    /// Maximum latency in microseconds
    pub max_latency_us: u64,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            success_count: 0,
            error_count: 0,
            total_latency_us: 0,
            min_latency_us: u64::MAX,
            max_latency_us: 0,
        }
    }

    pub fn record_success(&mut self, latency_us: u64) {
        self.success_count += 1;
        self.total_latency_us += latency_us;
        self.min_latency_us = self.min_latency_us.min(latency_us);
        self.max_latency_us = self.max_latency_us.max(latency_us);
    }

    pub fn record_error(&mut self) {
        self.error_count += 1;
    }

    pub fn avg_latency_us(&self) -> f64 {
        if self.success_count > 0 {
            self.total_latency_us as f64 / self.success_count as f64
        } else {
            0.0
        }
    }

    pub fn throughput(&self, elapsed_secs: f64) -> f64 {
        if elapsed_secs > 0.0 {
            self.success_count as f64 / elapsed_secs
        } else {
            0.0
        }
    }
}

/// Shared test context for coordinating writer and readers
pub struct TestContext {
    /// IDs of nodes that have been written (shared between writer and readers)
    pub written_node_ids: Arc<Mutex<Vec<Id>>>,
    /// IDs of edges that have been written
    pub written_edge_ids: Arc<Mutex<Vec<Id>>>,
    /// Signal to stop all threads
    pub stop_signal: Arc<AtomicBool>,
}

impl TestContext {
    pub fn new() -> Self {
        Self {
            written_node_ids: Arc::new(Mutex::new(Vec::new())),
            written_edge_ids: Arc::new(Mutex::new(Vec::new())),
            stop_signal: Arc::new(AtomicBool::new(false)),
        }
    }

    pub async fn add_written_node(&self, id: Id) {
        self.written_node_ids.lock().await.push(id);
    }

    pub async fn add_written_edge(&self, id: Id) {
        self.written_edge_ids.lock().await.push(id);
    }

    pub async fn get_random_node_id(&self) -> Option<Id> {
        let nodes = self.written_node_ids.lock().await;
        if nodes.is_empty() {
            None
        } else {
            let idx = (Instant::now().elapsed().as_nanos() as usize) % nodes.len();
            Some(nodes[idx])
        }
    }

    pub async fn get_random_edge_id(&self) -> Option<Id> {
        let edges = self.written_edge_ids.lock().await;
        if edges.is_empty() {
            None
        } else {
            let idx = (Instant::now().elapsed().as_nanos() as usize) % edges.len();
            Some(edges[idx])
        }
    }

    pub async fn node_count(&self) -> usize {
        self.written_node_ids.lock().await.len()
    }

    pub async fn edge_count(&self) -> usize {
        self.written_edge_ids.lock().await.len()
    }

    pub fn should_stop(&self) -> bool {
        self.stop_signal.load(Ordering::Relaxed)
    }

    pub fn signal_stop(&self) {
        self.stop_signal.store(true, Ordering::Relaxed);
    }
}

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
