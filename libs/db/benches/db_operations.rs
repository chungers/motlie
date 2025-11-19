// Comprehensive benchmarks for database operations
//
// This benchmark suite measures:
// 1. Write operations at various scales
// 2. Point lookup operations
// 3. Prefix scan operations (CRITICAL - tests direct encoding improvement)
// 4. Scan latency by node position (proves O(N) → O(K) improvement)
// 5. Scan latency by node degree (proves scaling with result set size)
//
// To run:
// ```
// cargo bench --manifest-path libs/db/Cargo.toml
// ```
//
// To compare with baseline:
// ```
// cargo bench --manifest-path libs/db/Cargo.toml -- --save-baseline before
// # Make changes...
// cargo bench --manifest-path libs/db/Cargo.toml -- --baseline before
// ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use motlie_db::{
    create_mutation_writer, create_query_reader, spawn_graph_consumer, spawn_query_consumer,
    AddEdge, AddNode, Id, MutationRunnable, NodeById, OutgoingEdges, QueryRunnable, ReaderConfig,
    TimestampMilli, WriterConfig,
};
use std::time::Duration;
use tempfile::TempDir;

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
    let graph_handle = spawn_graph_consumer(graph_receiver, writer_config, temp_dir.path());

    let mut node_ids = Vec::with_capacity(num_nodes);

    // Create nodes
    for i in 0..num_nodes {
        let id = Id::new();
        let node = AddNode {
            id,
            name: format!("node_{}", i),
            ts_millis: TimestampMilli::now(),
            temporal_range: None,
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
                id: Id::new(),
                source_node_id: src_id,
                target_node_id: dst_id,
                name: format!("edge_{}", j),
                ts_millis: TimestampMilli::now(),
                temporal_range: None,
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

criterion_group!(
    benches,
    bench_writes,
    bench_point_lookups,
    bench_prefix_scans_by_position,
    bench_prefix_scans_by_degree,
    bench_scan_position_independence
);
criterion_main!(benches);
