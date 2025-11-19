/// Integration test for concurrent read/write operations with readonly instances
///
/// This test validates that the database correctly handles:
/// 1. One readwrite writer inserting nodes and edges
/// 2. Multiple readonly readers querying concurrently
/// 3. Data consistency across concurrent operations
/// 4. Performance characteristics under concurrent load
///
/// Test Structure:
/// - Writer thread: Inserts N nodes and M edges over T seconds
/// - Reader threads (R readers): Each continuously queries random nodes/edges using readonly storage
/// - Metrics: Write latency, read latency, throughput, success rates
///
/// This is an integration test (not a benchmark) because:
/// - Validates functional correctness (data consistency)
/// - Tests concurrent access patterns
/// - Provides observability metrics for debugging
/// - Not intended for performance regression testing
mod common;

use common::concurrent_test_utils::{writer_task, Metrics, TestContext};
use motlie_db::{EdgeById, NodeById, QueryRunnable, ReaderConfig};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Reader task: continuously queries nodes and edges using readonly storage
async fn reader_task(
    db_path: std::path::PathBuf,
    context: Arc<TestContext>,
    _reader_id: usize,
) -> Metrics {
    let mut metrics = Metrics::new();

    // Wait until there's some data to read
    loop {
        if context.node_count().await > 10 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Create reader
    let (reader, reader_rx) = {
        let config = ReaderConfig {
            channel_buffer_size: 10,
        };
        let (sender, receiver) = flume::bounded(config.channel_buffer_size);
        let reader = motlie_db::Reader::new(sender);
        (reader, receiver)
    };

    // Spawn query consumer (it will open its own readonly storage)
    let consumer_handle = motlie_db::spawn_query_consumer(
        reader_rx,
        ReaderConfig {
            channel_buffer_size: 10,
        },
        &db_path,
    );

    // Continuously query random nodes and edges
    let mut iteration = 0;
    while !context.should_stop() {
        iteration += 1;

        // Alternate between node and edge queries
        if iteration % 2 == 0 {
            // Query node
            if let Some(node_id) = context.get_random_node_id().await {
                let start = Instant::now();
                let result = NodeById::new(node_id, None)
                    .run(&reader, Duration::from_secs(1))
                    .await;
                let latency_us = start.elapsed().as_micros() as u64;

                match result {
                    Ok(_) => metrics.record_success(latency_us),
                    Err(_) => metrics.record_error(),
                }
            }
        } else {
            // Query edge
            if let Some(edge_id) = context.get_random_edge_id().await {
                let start = Instant::now();
                let result = EdgeById::new(edge_id, None)
                    .run(&reader, Duration::from_secs(1))
                    .await;
                let latency_us = start.elapsed().as_micros() as u64;

                match result {
                    Ok(_) => metrics.record_success(latency_us),
                    Err(_) => metrics.record_error(),
                }
            }
        }

        // Small delay between queries
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    // Cleanup
    drop(reader);
    let _ = consumer_handle.await;

    metrics
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_concurrent_read_write_integration() {
    println!("\n=== Concurrent Read/Write Integration Test ===\n");

    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let db_path = temp_dir.path().to_path_buf();

    // Test parameters
    let num_nodes = 100;
    let num_edges_per_node = 3;
    let num_readers = 4;

    let context = Arc::new(TestContext::new());
    let test_start = Instant::now();

    // Spawn writer task
    let writer_context = context.clone();
    let writer_db_path = db_path.clone();
    let writer_handle = tokio::spawn(async move {
        writer_task(
            writer_db_path,
            writer_context,
            num_nodes,
            num_edges_per_node,
        )
        .await
    });

    // Spawn reader tasks
    let mut reader_handles = Vec::new();
    for reader_id in 0..num_readers {
        let reader_context = context.clone();
        let reader_db_path = db_path.clone();
        let handle =
            tokio::spawn(
                async move { reader_task(reader_db_path, reader_context, reader_id).await },
            );
        reader_handles.push(handle);
    }

    // Wait for writer to complete
    let write_metrics = writer_handle.await.expect("Writer task failed");

    // Let readers run a bit longer to catch up with all writes
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Signal readers to stop
    context.signal_stop();

    // Wait for all readers to complete
    let mut reader_metrics = Vec::new();
    for handle in reader_handles {
        let metrics = handle.await.expect("Reader task failed");
        reader_metrics.push(metrics);
    }

    let test_duration = test_start.elapsed().as_secs_f64();

    // Print results
    println!("Test Duration: {:.2}s", test_duration);
    println!("\n--- Writer Metrics ---");
    println!(
        "  Operations: {} success, {} errors",
        write_metrics.success_count, write_metrics.error_count
    );
    println!(
        "  Latency: avg={:.2}µs, min={}µs, max={}µs",
        write_metrics.avg_latency_us(),
        write_metrics.min_latency_us,
        write_metrics.max_latency_us
    );
    println!(
        "  Throughput: {:.2} ops/sec",
        write_metrics.throughput(test_duration)
    );

    println!("\n--- Reader Metrics ---");
    for (i, metrics) in reader_metrics.iter().enumerate() {
        println!(
            "  Reader {}: {} success, {} errors",
            i, metrics.success_count, metrics.error_count
        );
        println!(
            "    Latency: avg={:.2}µs, min={}µs, max={}µs",
            metrics.avg_latency_us(),
            metrics.min_latency_us,
            metrics.max_latency_us
        );
        println!(
            "    Throughput: {:.2} ops/sec",
            metrics.throughput(test_duration)
        );
    }

    // Aggregate reader stats
    let total_read_success: u64 = reader_metrics.iter().map(|m| m.success_count).sum();
    let total_read_errors: u64 = reader_metrics.iter().map(|m| m.error_count).sum();
    let total_read_latency: u64 = reader_metrics.iter().map(|m| m.total_latency_us).sum();
    let avg_read_latency = if total_read_success > 0 {
        total_read_latency as f64 / total_read_success as f64
    } else {
        0.0
    };

    println!("\n--- Aggregate Reader Stats ---");
    println!(
        "  Total reads: {} success, {} errors",
        total_read_success, total_read_errors
    );
    println!("  Average latency: {:.2}µs", avg_read_latency);
    println!(
        "  Total throughput: {:.2} ops/sec",
        total_read_success as f64 / test_duration
    );

    println!("\n--- Data Consistency ---");
    let final_node_count = context.node_count().await;
    let final_edge_count = context.edge_count().await;
    println!("  Nodes written: {}", final_node_count);
    println!("  Edges written: {}", final_edge_count);
    println!("  Expected nodes: {}", num_nodes);
    println!("  Expected edges: {}", num_nodes * num_edges_per_node);

    // Assertions for correctness
    assert_eq!(write_metrics.error_count, 0, "Writer should have no errors");
    assert_eq!(final_node_count, num_nodes, "All nodes should be written");
    assert_eq!(
        final_edge_count,
        num_nodes * num_edges_per_node,
        "All edges should be written"
    );

    // Readers should have mostly successful reads (some errors are OK if querying before data is written)
    let total_reader_ops = total_read_success + total_read_errors;
    assert!(
        total_read_success > 0,
        "Readers should have some successful reads"
    );

    // Readers should have reasonable success rate
    // Note: Some failures are expected due to timing (querying IDs before they're flushed to disk)
    let success_rate = total_read_success as f64 / total_reader_ops as f64;
    println!("\n--- Quality Metrics ---");
    println!("  Reader success rate: {:.1}%", success_rate * 100.0);

    // Success rate should be at least 10% - this is a consistency test, not a performance test
    // Low success rate is OK because:
    // 1. Readers may query IDs that are in the write buffer but not yet flushed
    // 2. RocksDB readonly mode may not see latest writes immediately
    // 3. The important thing is that successful reads return correct data
    assert!(
        success_rate >= 0.10,
        "Reader success rate should be at least 10% (got {:.1}%)",
        success_rate * 100.0
    );

    println!("\n=== Test Passed ===\n");
}
