# Database Performance Benchmark Plan

## Overview

Benchmarks to compare MessagePack vs Direct Encoding for keys across various database sizes and query patterns.

## Benchmark Structure

### Using Criterion (Rust standard)

**File**: `libs/db/benches/db_operations.rs`

### Benchmark Scenarios

#### 1. **Write Operations**
- **Small**: 100 nodes, 500 edges (5 edges/node avg)
- **Medium**: 1,000 nodes, 10,000 edges (10 edges/node avg)
- **Large**: 10,000 nodes, 100,000 edges (10 edges/node avg)
- **XLarge**: 50,000 nodes, 500,000 edges (10 edges/node avg)

**Measures**:
- Time to write all nodes
- Time to write all edges
- Total throughput (ops/sec)
- Database size on disk

#### 2. **Point Lookup Operations**
Test direct key lookups by ID:
- Node by ID
- Edge by ID
- Random vs sequential access patterns

**Measures**:
- Average lookup latency
- P50, P95, P99 percentiles
- Throughput (ops/sec)

#### 3. **Prefix Scan Operations** ⚡ CRITICAL
Test the main use case for direct encoding:
- Get all edges FROM a node (forward_edges scan)
- Get all edges TO a node (reverse_edges scan)
- Get all fragments for a node (fragments scan)

**Query patterns**:
- Hot nodes (high degree, many edges)
- Average nodes (moderate degree)
- Cold nodes (low degree, few edges)
- Early in database (low ID values)
- Late in database (high ID values)
- Middle of database

**Measures**:
- Scan latency by node position
- Scan latency by node degree
- Total scanned keys (verify correctness)
- Deserialization overhead

#### 4. **Mixed Workload**
Realistic mix:
- 70% reads (prefix scans + point lookups)
- 30% writes

### Comparison Matrix

| Encoding | Small | Medium | Large | XLarge |
|----------|-------|--------|-------|--------|
| MessagePack Keys | ✓ | ✓ | ✓ | ✓ |
| Direct Encoding | ✓ | ✓ | ✓ | ✓ |

## Implementation Approach

### Phase 1: Add Criterion Dependency

**File**: `libs/db/Cargo.toml`

```toml
[dev-dependencies]
tempfile = "3.14"
criterion = { version = "0.5", features = ["html_reports", "async_tokio"] }

[[bench]]
name = "db_operations"
harness = false
```

### Phase 2: Create Benchmark Harness

**File**: `libs/db/benches/db_operations.rs`

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use motlie_db::{Storage, WriterConfig, ReaderConfig, AddNode, AddEdge, Id};
use tempfile::TempDir;
use std::time::Duration;

// Helper to create test database
async fn create_test_db(
    temp_dir: &TempDir,
    num_nodes: usize,
    avg_edges_per_node: usize,
) -> Storage {
    let storage = Storage::new(temp_dir.path().to_path_buf(), false);
    storage.ready().await.unwrap();

    // Create nodes
    let node_ids: Vec<Id> = (0..num_nodes)
        .map(|i| {
            let id = Id::new();
            let node = AddNode {
                id,
                name: format!("node_{}", i),
                ts_millis: motlie_db::TimestampMilli::now(),
            };
            // Write node...
            id
        })
        .collect();

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
                ts_millis: motlie_db::TimestampMilli::now(),
            };
            // Write edge...
        }
    }

    storage.close().await.unwrap();
    storage
}

// Benchmark 1: Write operations
fn bench_writes(c: &mut Criterion) {
    let mut group = c.benchmark_group("writes");

    for size in [100, 1_000, 10_000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_nodes", size)),
            size,
            |b, &size| {
                b.to_async(tokio::runtime::Runtime::new().unwrap())
                    .iter(|| async {
                        let temp_dir = TempDir::new().unwrap();
                        create_test_db(&temp_dir, size, 10).await
                    });
            },
        );
    }

    group.finish();
}

// Benchmark 2: Point lookups
fn bench_point_lookups(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let temp_dir = TempDir::new().unwrap();

    // Pre-populate database
    let storage = rt.block_on(create_test_db(&temp_dir, 10_000, 10));

    let mut group = c.benchmark_group("point_lookups");

    group.bench_function("node_by_id", |b| {
        b.to_async(&rt).iter(|| async {
            // Query random node by ID
            black_box(/* query */)
        });
    });

    group.finish();
}

// Benchmark 3: Prefix scans (CRITICAL)
fn bench_prefix_scans(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    let mut group = c.benchmark_group("prefix_scans");
    group.measurement_time(Duration::from_secs(10));

    for db_size in [1_000, 10_000, 100_000].iter() {
        for node_position in ["early", "middle", "late"].iter() {
            for node_degree in [1, 10, 100].iter() {
                let param = format!("{}nodes_{}_{}_edges", db_size, node_position, node_degree);

                group.bench_with_input(
                    BenchmarkId::from_parameter(&param),
                    &(db_size, node_position, node_degree),
                    |b, &(size, pos, degree)| {
                        // Setup database with specific characteristics
                        let temp_dir = TempDir::new().unwrap();
                        let storage = rt.block_on(create_test_db(&temp_dir, *size, 10));

                        // Select target node based on position
                        let target_node_idx = match *pos {
                            "early" => size / 10,
                            "middle" => size / 2,
                            "late" => size * 9 / 10,
                            _ => unreachable!(),
                        };

                        b.to_async(&rt).iter(|| async {
                            // Scan edges from node
                            black_box(/* scan operation */)
                        });
                    },
                );
            }
        }
    }

    group.finish();
}

// Benchmark 4: Scan position impact
fn bench_scan_by_position(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let temp_dir = TempDir::new().unwrap();

    // Create 100K edge database
    let storage = rt.block_on(create_test_db(&temp_dir, 10_000, 10));

    let mut group = c.benchmark_group("scan_position");

    // Test scanning nodes at different positions in the key space
    for position_pct in [0, 10, 25, 50, 75, 90, 99].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}pct", position_pct)),
            position_pct,
            |b, &pct| {
                b.to_async(&rt).iter(|| async {
                    // Scan node at specific position
                    black_box(/* scan */)
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
    bench_prefix_scans,
    bench_scan_by_position
);
criterion_main!(benches);
```

### Phase 3: Add Comparison Benchmarks

Create two versions:
1. **Before**: Current MessagePack implementation
2. **After**: Direct encoding implementation

Run both and compare:
```bash
# Baseline (MessagePack)
cargo bench --bench db_operations -- --save-baseline msgpack

# After implementing direct encoding
cargo bench --bench db_operations -- --baseline msgpack
```

### Phase 4: Key Metrics to Track

#### Write Performance
- Throughput (edges/sec)
- Latency per operation
- Database size on disk

#### Read Performance - Point Lookups
- Latency (P50, P95, P99)
- Throughput (queries/sec)

#### Read Performance - Prefix Scans ⚡
- **Scan latency by database size** (expect O(N) → O(K) improvement)
- **Scan latency by node position** (expect early nodes slow, late nodes very slow → all O(K))
- **Keys scanned vs keys returned** (expect N scanned → K scanned)
- **Throughput** (scans/sec)

### Expected Results

#### MessagePack Keys (Before)

| DB Size | Node Position | Edges | Keys Scanned | Latency |
|---------|--------------|-------|--------------|---------|
| 100K    | Early (10%)  | 10    | ~10K         | ~10ms   |
| 100K    | Middle (50%) | 10    | ~50K         | ~50ms   |
| 100K    | Late (90%)   | 10    | ~90K         | ~90ms   |

**Problem**: Latency grows with node position in database (O(N))

#### Direct Encoding (After)

| DB Size | Node Position | Edges | Keys Scanned | Latency |
|---------|--------------|-------|--------------|---------|
| 100K    | Early (10%)  | 10    | ~10          | ~0.1ms  |
| 100K    | Middle (50%) | 10    | ~10          | ~0.1ms  |
| 100K    | Late (90%)   | 10    | ~10          | ~0.1ms  |

**Improvement**: Latency independent of node position (O(K))

### Visualization

Criterion will generate HTML reports at `target/criterion/`:
- Performance graphs
- Statistical analysis
- Comparison between baselines

## Additional Instrumentation

### Add counters to track:

```rust
struct ScanMetrics {
    keys_scanned: u64,
    keys_matched: u64,
    keys_deserialized: u64,
    scan_duration: Duration,
}
```

Include in benchmark output to verify:
- MessagePack: `keys_scanned >> keys_matched`
- Direct encoding: `keys_scanned == keys_matched`

## Running Benchmarks

```bash
# Run all benchmarks
cargo bench --bench db_operations

# Run specific benchmark
cargo bench --bench db_operations -- prefix_scans

# Save baseline
cargo bench --bench db_operations -- --save-baseline before

# Compare against baseline
cargo bench --bench db_operations -- --baseline before

# Generate flamegraphs (requires cargo-flamegraph)
cargo flamegraph --bench db_operations -- --bench
```

## Integration with CI

Optional: Add benchmark regression detection:
```yaml
# .github/workflows/benchmark.yml
- name: Run benchmarks
  run: cargo bench --bench db_operations -- --save-baseline main

- name: Check for regressions
  run: cargo bench --bench db_operations -- --baseline main
```

## Summary

This benchmark plan:
1. ✅ Tests various database sizes (100 to 500K edges)
2. ✅ Focuses on prefix scan performance (the critical improvement)
3. ✅ Measures position impact (proves O(N) → O(K) improvement)
4. ✅ Provides comparison framework (before/after)
5. ✅ Generates detailed reports and visualizations
6. ✅ Tracks key metrics (keys scanned, latency, throughput)

The benchmarks will provide concrete evidence of the performance improvement from switching to direct encoding for keys.
