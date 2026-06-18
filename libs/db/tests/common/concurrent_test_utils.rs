/// Shared test utilities for concurrent read/write integration tests
///
/// This module provides common infrastructure for testing concurrent database operations:
/// - Metrics collection for read and write operations
/// - Test context for coordinating writer and readers
use motlie_db::Id;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
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
    /// Signal to stop all threads
    pub stop_signal: Arc<AtomicBool>,
}

impl TestContext {
    pub fn new() -> Self {
        Self {
            written_node_ids: Arc::new(Mutex::new(Vec::new())),
            stop_signal: Arc::new(AtomicBool::new(false)),
        }
    }

    pub async fn add_written_node(&self, id: Id) {
        self.written_node_ids.lock().await.push(id);
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

    pub async fn node_count(&self) -> usize {
        self.written_node_ids.lock().await.len()
    }

    pub fn should_stop(&self) -> bool {
        self.stop_signal.load(Ordering::Relaxed)
    }

    pub fn signal_stop(&self) {
        self.stop_signal.store(true, Ordering::Relaxed);
    }
}
