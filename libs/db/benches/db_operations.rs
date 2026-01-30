// Comprehensive benchmarks for database operations
//
// This benchmark suite measures performance across multiple dimensions to evaluate
// planned optimizations including:
//
// ## Current Benchmarks (Baseline)
// 1. Write operations at various scales
// 2. Point lookup operations
// 3. Prefix scan operations (CRITICAL - tests direct encoding improvement)
// 4. Scan latency by node position (proves O(N) → O(K) improvement)
// 5. Scan latency by node degree (proves scaling with result set size)
//
// ## Optimization Evaluation Benchmarks
// 6. Serialization overhead (measures rmp_serde + LZ4 decompression cost)
// 7. Value size impact (measures blob separation benefit)
// 8. Transaction API performance (compares channel dispatch vs direct read)
// 9. Memory allocation profiling (measures allocation churn during scans)
//
// ## Planned Optimizations (from REVIEW.md)
// - Blob Separation: Split hot (topology/weights) from cold (summaries) data
// - Zero-Copy Serialization (rkyv): Eliminate deserialization allocations
// - Direct Read Path: Bypass channel dispatch for point lookups
// - Iterator-Based Scans: Return iterators instead of collected vectors
//
// ## Running Benchmarks
//
// ```bash
// # Run all benchmarks
// cargo bench --manifest-path libs/db/Cargo.toml
//
// # Run specific benchmark group
// cargo bench --manifest-path libs/db/Cargo.toml -- serialization_overhead
//
// # Compare with baseline (before optimization)
// cargo bench --manifest-path libs/db/Cargo.toml -- --save-baseline before
// # After implementing optimization...
// cargo bench --manifest-path libs/db/Cargo.toml -- --baseline before
//
// # Generate flamegraphs (requires cargo-flamegraph)
// cargo flamegraph --bench db_operations -- --bench
// ```
//
// ## Interpreting Results
//
// Key metrics to compare before/after optimization:
// - `serialization_overhead/*`: Should decrease with rkyv (target: 5-10x)
// - `value_size_impact/*`: Should show larger gains with blob separation
// - `transaction_vs_channel/*`: Should show transaction API competitive
// - `prefix_scans_*/*`: Should remain fast (already O(K))

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use motlie_db::graph::mutation::{AddEdge, AddNode};
use motlie_db::writer::Runnable as MutationRunnable;
use motlie_db::graph::query::{NodeById, OutgoingEdges};
use motlie_db::reader::Runnable as QueryRunnable;
use motlie_db::graph::reader::{create_query_reader, spawn_query_consumer, ReaderConfig};
use motlie_db::graph::schema::{EdgeSummary, NodeSummary};
use motlie_db::graph::writer::{create_mutation_writer, spawn_mutation_consumer, WriterConfig};
use motlie_db::vector::schema::ExternalKey;
use motlie_db::{Id, TimestampMilli};
use std::time::Duration;
use tempfile::TempDir;

// For serialization benchmarks
extern crate lz4;
extern crate rmp_serde;

/// Helper to create a test database with specified characteristics
async fn create_test_db(
    temp_dir: &TempDir,
    num_nodes: usize,
    avg_edges_per_node: usize,
) -> Vec<Id> {
    let writer_config = WriterConfig {
        channel_buffer_size: 10000,
    };

    let (writer, graph_receiver) = create_mutation_writer(writer_config.clone());
    let graph_handle = spawn_mutation_consumer(graph_receiver, writer_config, temp_dir.path());

    let mut node_ids = Vec::with_capacity(num_nodes);

    // Create nodes
    for i in 0..num_nodes {
        let id = Id::new();
        let node = AddNode {
            id,
            name: format!("node_{}", i),
            ts_millis: TimestampMilli::now(),
            valid_range: None,
            summary: NodeSummary::from_text(&format!("Benchmark node {}", i)),
        };

        node.run(&writer).await.unwrap();
        node_ids.push(id);
    }

    // Create edges
    for (i, &src_id) in node_ids.iter().enumerate() {
        for j in 0..avg_edges_per_node {
            let dst_idx = (i + j + 1) % num_nodes;
            let dst_id = node_ids[dst_idx];

            let edge = AddEdge {
                source_node_id: src_id,
                target_node_id: dst_id,
                name: format!("edge_{}", j),
                ts_millis: TimestampMilli::now(),
                valid_range: None,
                summary: EdgeSummary::from_text(&format!("Benchmark edge {}", j)),
                weight: None,
            };

            edge.run(&writer).await.unwrap();
        }
    }

    // Drop writer to signal completion, then wait for processing to finish
    drop(writer);
    graph_handle.await.unwrap().unwrap();

    node_ids
}

/// Benchmark 1: Write operations at various scales
fn bench_writes(c: &mut Criterion) {
    let mut group = c.benchmark_group("writes");
    group.measurement_time(Duration::from_secs(20));

    for size in [100, 1_000, 5_000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_nodes_10x_edges", size)),
            size,
            |b, &size| {
                b.to_async(tokio::runtime::Runtime::new().unwrap())
                    .iter(|| async {
                        let temp_dir = TempDir::new().unwrap();
                        let _ = create_test_db(&temp_dir, size, 10).await;
                    });
            },
        );
    }

    group.finish();
}

/// Benchmark 2: Point lookups by ID
fn bench_point_lookups(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let temp_dir = TempDir::new().unwrap();

    // Pre-populate database with 10K nodes, 100K edges
    let node_ids = rt.block_on(create_test_db(&temp_dir, 10_000, 10));

    // Create reader for querying
    let reader_config = ReaderConfig {
        channel_buffer_size: 1000,
    };
    let (reader, query_receiver) = create_query_reader(reader_config.clone());
    let _guard = rt.enter();
    let _query_handle = spawn_query_consumer(query_receiver, reader_config, temp_dir.path());

    let mut group = c.benchmark_group("point_lookups");
    group.measurement_time(Duration::from_secs(10));

    // Test various node positions
    for position in ["early", "middle", "late"].iter() {
        let node_idx = match *position {
            "early" => node_ids.len() / 10,
            "middle" => node_ids.len() / 2,
            "late" => node_ids.len() * 9 / 10,
            _ => unreachable!(),
        };
        let target_id = node_ids[node_idx];

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("node_by_id_{}", position)),
            position,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let result = NodeById::new(target_id, None)
                        .run(&reader, Duration::from_secs(5))
                        .await
                        .unwrap();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark 3: Prefix scans - CRITICAL test for direct encoding
///
/// This benchmark proves the O(N) → O(K) improvement from direct encoding.
/// With MessagePack keys:
/// - Scanning early nodes: fast (few keys to scan)
/// - Scanning late nodes: VERY slow (must scan entire CF)
///
/// With Direct encoding:
/// - All scans: O(K) - only scan matching keys, independent of position
fn bench_prefix_scans_by_position(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("prefix_scans_by_position");
    group.measurement_time(Duration::from_secs(15));

    // Test different database sizes
    for db_size in [1_000, 10_000].iter() {
        let temp_dir = TempDir::new().unwrap();
        let node_ids = rt.block_on(create_test_db(&temp_dir, *db_size, 10));

        // Create reader for querying
        let reader_config = ReaderConfig {
            channel_buffer_size: 1000,
        };
        let (reader, query_receiver) = create_query_reader(reader_config.clone());
        let _guard = rt.enter();
        let _query_handle = spawn_query_consumer(query_receiver, reader_config, temp_dir.path());

        // Test different positions in the key space
        for position in ["early", "middle", "late"].iter() {
            let node_idx = match *position {
                "early" => db_size / 10,
                "middle" => db_size / 2,
                "late" => db_size * 9 / 10,
                _ => unreachable!(),
            };
            let target_id = node_ids[node_idx];

            group.bench_with_input(
                BenchmarkId::from_parameter(format!("{}nodes_{}_position", db_size, position)),
                &(db_size, position),
                |b, _| {
                    b.to_async(&rt).iter(|| async {
                        let result = OutgoingEdges::new(target_id, None)
                            .run(&reader, Duration::from_secs(5))
                            .await
                            .unwrap();
                        black_box(result)
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark 4: Prefix scans by node degree
///
/// Tests that scan performance scales with result set size (K), not database size (N)
fn bench_prefix_scans_by_degree(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("prefix_scans_by_degree");
    group.measurement_time(Duration::from_secs(15));

    // Create databases with different edge densities
    for degree in [1, 10, 50].iter() {
        let temp_dir = TempDir::new().unwrap();
        let node_ids = rt.block_on(create_test_db(&temp_dir, 5_000, *degree));

        // Create reader for querying
        let reader_config = ReaderConfig {
            channel_buffer_size: 1000,
        };
        let (reader, query_receiver) = create_query_reader(reader_config.clone());
        let _guard = rt.enter();
        let _query_handle = spawn_query_consumer(query_receiver, reader_config, temp_dir.path());

        // Use a middle node to avoid position effects
        let target_id = node_ids[2_500];

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_edges", degree)),
            degree,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let result = OutgoingEdges::new(target_id, None)
                        .run(&reader, Duration::from_secs(5))
                        .await
                        .unwrap();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark 5: Scan all positions to prove position independence
///
/// With direct encoding, ALL positions should have similar performance.
/// With MessagePack, late positions would be dramatically slower.
fn bench_scan_position_independence(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let temp_dir = TempDir::new().unwrap();

    // Create 10K node database
    let node_ids = rt.block_on(create_test_db(&temp_dir, 10_000, 10));

    // Create reader for querying
    let reader_config = ReaderConfig {
        channel_buffer_size: 1000,
    };
    let (reader, query_receiver) = create_query_reader(reader_config.clone());
    let _guard = rt.enter();
    let _query_handle = spawn_query_consumer(query_receiver, reader_config, temp_dir.path());

    let mut group = c.benchmark_group("scan_position_independence");
    group.measurement_time(Duration::from_secs(10));

    // Test scanning nodes at different positions (percentiles)
    for position_pct in [0, 10, 25, 50, 75, 90, 99].iter() {
        let node_idx = (node_ids.len() * position_pct) / 100;
        let target_id = node_ids[node_idx];

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}pct", position_pct)),
            position_pct,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let result = OutgoingEdges::new(target_id, None)
                        .run(&reader, Duration::from_secs(5))
                        .await
                        .unwrap();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Optimization Evaluation Benchmarks
// ============================================================================
// These benchmarks are designed to measure the impact of planned optimizations
// from REVIEW.md. Run these before and after implementing optimizations.

/// Helper to create a test database with variable summary sizes
///
/// This enables testing the impact of blob separation by comparing
/// performance with small vs large summaries.
async fn create_test_db_with_summary_size(
    temp_dir: &TempDir,
    num_nodes: usize,
    avg_edges_per_node: usize,
    summary_size_bytes: usize,
) -> Vec<Id> {
    let writer_config = WriterConfig {
        channel_buffer_size: 10000,
    };

    let (writer, graph_receiver) = create_mutation_writer(writer_config.clone());
    let graph_handle = spawn_mutation_consumer(graph_receiver, writer_config, temp_dir.path());

    let mut node_ids = Vec::with_capacity(num_nodes);

    // Generate summary of specified size
    let summary_content: String = (0..summary_size_bytes)
        .map(|i| (b'A' + (i % 26) as u8) as char)
        .collect();

    // Create nodes with specified summary size
    for i in 0..num_nodes {
        let id = Id::new();
        let node = AddNode {
            id,
            name: format!("node_{}", i),
            ts_millis: TimestampMilli::now(),
            valid_range: None,
            summary: NodeSummary::from_text(&summary_content),
        };

        node.run(&writer).await.unwrap();
        node_ids.push(id);
    }

    // Create edges with specified summary size
    for (i, &src_id) in node_ids.iter().enumerate() {
        for j in 0..avg_edges_per_node {
            let dst_idx = (i + j + 1) % num_nodes;
            let dst_id = node_ids[dst_idx];

            let edge = AddEdge {
                source_node_id: src_id,
                target_node_id: dst_id,
                name: format!("edge_{}", j),
                ts_millis: TimestampMilli::now(),
                valid_range: None,
                summary: EdgeSummary::from_text(&summary_content),
                weight: Some((i * j) as f64),
            };

            edge.run(&writer).await.unwrap();
        }
    }

    // Drop writer to signal completion, then wait for processing to finish
    drop(writer);
    graph_handle.await.unwrap().unwrap();

    node_ids
}

/// Benchmark 6: Value Size Impact (Blob Separation Evaluation)
///
/// Measures scan performance with varying summary sizes to quantify
/// the benefit of blob separation (splitting hot topology from cold summaries).
///
/// Expected results:
/// - Current (mixed): Performance degrades significantly with larger summaries
/// - After blob separation: Performance independent of summary size for topology-only scans
fn bench_value_size_impact(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("value_size_impact");
    group.measurement_time(Duration::from_secs(15));

    // Test with different summary sizes (simulating hot vs cold data split)
    // Small = topology only (~30 bytes), Large = with summary (~500 bytes)
    for summary_size in [0, 100, 500, 2000].iter() {
        let temp_dir = TempDir::new().unwrap();
        let node_ids = rt.block_on(create_test_db_with_summary_size(
            &temp_dir,
            5_000,
            10,
            *summary_size,
        ));

        // Create reader for querying
        let reader_config = ReaderConfig {
            channel_buffer_size: 1000,
        };
        let (reader, query_receiver) = create_query_reader(reader_config.clone());
        let _guard = rt.enter();
        let _query_handle = spawn_query_consumer(query_receiver, reader_config, temp_dir.path());

        // Use middle node
        let target_id = node_ids[2_500];

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_bytes_summary", summary_size)),
            summary_size,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let result = OutgoingEdges::new(target_id, None)
                        .run(&reader, Duration::from_secs(5))
                        .await
                        .unwrap();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark 7: Serialization Overhead Measurement
///
/// Measures the overhead of the current serialization pipeline:
/// - Serialize: rmp_serde::to_vec -> LZ4 compress
/// - Deserialize: LZ4 decompress -> rmp_serde::from_slice
///
/// This benchmark isolates serialization cost to quantify rkyv benefits.
/// Run with different payload sizes to understand scaling.
fn bench_serialization_overhead(c: &mut Criterion) {
    // NodeSummary already imported at top of file

    let mut group = c.benchmark_group("serialization_overhead");
    group.measurement_time(Duration::from_secs(10));

    // Test serialization of different sized payloads
    for payload_size in [50, 200, 500, 2000].iter() {
        let content: String = (0..*payload_size)
            .map(|i| (b'A' + (i % 26) as u8) as char)
            .collect();

        // Benchmark DataUrl creation (used for summaries)
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("dataurl_create_{}_bytes", payload_size)),
            payload_size,
            |b, _| {
                b.iter(|| {
                    let summary = NodeSummary::from_text(&content);
                    black_box(summary)
                });
            },
        );

        // Create a summary to benchmark serialization
        let summary = NodeSummary::from_text(&content);

        // Benchmark rmp_serde serialization
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("rmp_serialize_{}_bytes", payload_size)),
            payload_size,
            |b, _| {
                b.iter(|| {
                    let bytes = rmp_serde::to_vec(&summary).unwrap();
                    black_box(bytes)
                });
            },
        );

        // Serialize for deserialization benchmark
        let serialized = rmp_serde::to_vec(&summary).unwrap();

        // Benchmark rmp_serde deserialization
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("rmp_deserialize_{}_bytes", payload_size)),
            payload_size,
            |b, _| {
                b.iter(|| {
                    let decoded: NodeSummary = rmp_serde::from_slice(&serialized).unwrap();
                    black_box(decoded)
                });
            },
        );

        // Benchmark LZ4 compression
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("lz4_compress_{}_bytes", payload_size)),
            payload_size,
            |b, _| {
                b.iter(|| {
                    let compressed = lz4::block::compress(&serialized, None, true).unwrap();
                    black_box(compressed)
                });
            },
        );

        // Compress for decompression benchmark
        let compressed = lz4::block::compress(&serialized, None, true).unwrap();

        // Benchmark LZ4 decompression
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("lz4_decompress_{}_bytes", payload_size)),
            payload_size,
            |b, _| {
                b.iter(|| {
                    let decompressed = lz4::block::decompress(&compressed, None).unwrap();
                    black_box(decompressed)
                });
            },
        );

        // Benchmark full pipeline (serialize + compress)
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("full_serialize_{}_bytes", payload_size)),
            payload_size,
            |b, _| {
                b.iter(|| {
                    let serialized = rmp_serde::to_vec(&summary).unwrap();
                    let compressed = lz4::block::compress(&serialized, None, true).unwrap();
                    black_box(compressed)
                });
            },
        );

        // Benchmark full pipeline (decompress + deserialize)
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("full_deserialize_{}_bytes", payload_size)),
            payload_size,
            |b, _| {
                b.iter(|| {
                    let decompressed = lz4::block::decompress(&compressed, None).unwrap();
                    let decoded: NodeSummary = rmp_serde::from_slice(&decompressed).unwrap();
                    black_box(decoded)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark 8: Transaction API vs Channel Dispatch
///
/// Compares performance of:
/// - Channel-based queries (current async pattern)
/// - Transaction-based queries (synchronous, no channel overhead)
///
/// This quantifies the benefit of the Transaction API for read-your-writes
/// and helps evaluate whether a separate RunDirect trait is needed.
fn bench_transaction_vs_channel(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let temp_dir = TempDir::new().unwrap();

    // Create test database with 5K nodes
    let node_ids = rt.block_on(create_test_db(&temp_dir, 5_000, 10));

    // Create reader for channel-based queries
    let reader_config = ReaderConfig {
        channel_buffer_size: 1000,
    };
    let (reader, query_receiver) = create_query_reader(reader_config.clone());
    let _guard = rt.enter();
    let _query_handle = spawn_query_consumer(query_receiver, reader_config, temp_dir.path());

    let mut group = c.benchmark_group("transaction_vs_channel");
    group.measurement_time(Duration::from_secs(15));

    // Test at different positions to ensure fair comparison
    for position in ["early", "middle", "late"].iter() {
        let node_idx = match *position {
            "early" => node_ids.len() / 10,
            "middle" => node_ids.len() / 2,
            "late" => node_ids.len() * 9 / 10,
            _ => unreachable!(),
        };
        let target_id = node_ids[node_idx];

        // Benchmark: Channel-based point lookup
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("channel_node_by_id_{}", position)),
            position,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let result = NodeById::new(target_id, None)
                        .run(&reader, Duration::from_secs(5))
                        .await
                        .unwrap();
                    black_box(result)
                });
            },
        );

        // Benchmark: Channel-based edge scan
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("channel_outgoing_edges_{}", position)),
            position,
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let result = OutgoingEdges::new(target_id, None)
                        .run(&reader, Duration::from_secs(5))
                        .await
                        .unwrap();
                    black_box(result)
                });
            },
        );
    }

    // Note: Transaction API benchmarks would be added here once Transaction
    // supports direct (non-write) operations. Currently, Transaction is
    // designed for read-your-writes patterns during mutation.

    group.finish();
}

/// Benchmark 9: Batch Scan Throughput
///
/// Measures throughput when scanning many edges in sequence.
/// This is the primary workload for graph algorithms (BFS, PageRank, etc.)
/// and is most affected by:
/// - Blob separation (cache efficiency)
/// - Zero-copy deserialization (allocation reduction)
fn bench_batch_scan_throughput(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("batch_scan_throughput");
    group.measurement_time(Duration::from_secs(20));

    // Test with different database sizes
    for db_size in [1_000, 5_000, 10_000].iter() {
        let temp_dir = TempDir::new().unwrap();
        let node_ids = rt.block_on(create_test_db(&temp_dir, *db_size, 10));

        // Create reader
        let reader_config = ReaderConfig {
            channel_buffer_size: 1000,
        };
        let (reader, query_receiver) = create_query_reader(reader_config.clone());
        let _guard = rt.enter();
        let _query_handle = spawn_query_consumer(query_receiver, reader_config, temp_dir.path());

        // Benchmark: Scan edges from multiple nodes (simulates BFS/PageRank)
        let num_scans = 100;
        let scan_targets: Vec<Id> = (0..num_scans)
            .map(|i| node_ids[(i * db_size / num_scans) % db_size])
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_nodes_{}_scans", db_size, num_scans)),
            &(db_size, num_scans),
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let mut total_edges = 0;
                    for &target_id in &scan_targets {
                        let edges = OutgoingEdges::new(target_id, None)
                            .run(&reader, Duration::from_secs(5))
                            .await
                            .unwrap();
                        total_edges += edges.len();
                    }
                    black_box(total_edges)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark 10: Write Throughput with Varying Payload Sizes
///
/// Measures write throughput with different summary sizes to understand
/// how blob separation would affect write performance.
fn bench_write_throughput_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("write_throughput_by_size");
    group.measurement_time(Duration::from_secs(20));

    for summary_size in [0, 100, 500].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_bytes_summary", summary_size)),
            summary_size,
            |b, &size| {
                b.to_async(tokio::runtime::Runtime::new().unwrap())
                    .iter(|| async {
                        let temp_dir = TempDir::new().unwrap();
                        let _ = create_test_db_with_summary_size(&temp_dir, 1_000, 10, size).await;
                    });
            },
        );
    }

    group.finish();
}

/// Benchmark 11: ExternalKey roundtrip (1M ops)
///
/// Measures encode+decode cost for ExternalKey (NodeId variant).
fn bench_external_key_roundtrip_1m(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_external_key");
    group.measurement_time(Duration::from_secs(5));

    let key = ExternalKey::NodeId(Id::new());
    group.bench_function("external_key_roundtrip_1m", |b| {
        b.iter_custom(|_| {
            let start = std::time::Instant::now();
            let mut bytes = Vec::new();
            for _ in 0..1_000_000 {
                bytes = key.to_bytes();
                let parsed = ExternalKey::from_bytes(black_box(&bytes)).unwrap();
                black_box(parsed);
            }
            start.elapsed()
        });
    });

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

criterion_group!(
    name = baseline_benches;
    config = Criterion::default().sample_size(50);
    targets =
        bench_writes,
        bench_point_lookups,
        bench_prefix_scans_by_position,
        bench_prefix_scans_by_degree,
        bench_scan_position_independence
);

criterion_group!(
    name = optimization_benches;
    config = Criterion::default().sample_size(30);
    targets =
        bench_serialization_overhead,
        bench_value_size_impact,
        bench_transaction_vs_channel,
        bench_batch_scan_throughput,
        bench_write_throughput_by_size,
        bench_external_key_roundtrip_1m
);

criterion_main!(baseline_benches, optimization_benches);
