# Database Performance Benchmark Plan

> **Last Updated:** December 27, 2025
> **Status:** ✅ Implemented with optimization evaluation framework

## Overview

This benchmark suite provides comprehensive performance measurement for:
1. **Baseline Operations** - Current read/write performance
2. **Optimization Evaluation** - Metrics to compare before/after optimization implementations

The benchmarks are designed to validate the performance improvements outlined in [REVIEW.md](../REVIEW.md).

## Planned Optimizations (from REVIEW.md)

| Optimization | Impact | Priority | Benchmark Coverage |
|--------------|--------|----------|-------------------|
| **Blob Separation** | 10-20x cache efficiency | 1 | `value_size_impact/*` |
| **Zero-Copy (rkyv)** | 2-5x scan throughput | 3 | `serialization_overhead/*` |
| **Direct Read Path** | 3-5x point lookup | 2 (deferred) | `transaction_vs_channel/*` |
| **Iterator Scans** | Ergonomic, minor perf | 5 | N/A (code quality) |
| **Fulltext Sync** | Consistency | 4 | N/A (correctness) |

## Benchmark Structure

### File: `libs/db/benches/db_operations.rs`

The benchmark file is organized into two groups:

#### Baseline Benchmarks (existing)
- `bench_writes` - Write throughput at various scales
- `bench_point_lookups` - Point lookup latency
- `bench_prefix_scans_by_position` - Scan performance by key position
- `bench_prefix_scans_by_degree` - Scan performance by result set size
- `bench_scan_position_independence` - Verify O(K) scan behavior

#### Optimization Evaluation Benchmarks (new)
- `bench_serialization_overhead` - Isolate rmp_serde + LZ4 cost
- `bench_value_size_impact` - Measure blob separation benefit
- `bench_transaction_vs_channel` - Compare dispatch overhead
- `bench_batch_scan_throughput` - Graph algorithm simulation
- `bench_write_throughput_by_size` - Write performance by payload size

---

## Baseline Benchmarks

### 1. Write Operations (`bench_writes`)

Tests write throughput at various scales.

| Scale | Nodes | Edges | Use Case |
|-------|-------|-------|----------|
| Small | 100 | 1,000 | Unit test equivalent |
| Medium | 1,000 | 10,000 | Integration test |
| Large | 5,000 | 50,000 | Realistic workload |

**Metrics:**
- Total write time
- Throughput (ops/sec)

### 2. Point Lookup Operations (`bench_point_lookups`)

Tests direct key lookups by ID at different database positions.

**Test Cases:**
- Node lookup at early position (10%)
- Node lookup at middle position (50%)
- Node lookup at late position (90%)

**Metrics:**
- Average lookup latency
- Position independence (should be O(1))

### 3. Prefix Scan Operations (`bench_prefix_scans_*`)

Tests the core use case for direct encoding: scanning edges from a node.

**Test Dimensions:**
- Database size: 1K, 10K nodes
- Node position: early, middle, late
- Node degree: 1, 10, 50 edges

**Metrics:**
- Scan latency by position (should be O(K), not O(N))
- Scan latency by degree (should scale with result set)

### 4. Scan Position Independence (`bench_scan_position_independence`)

Proves that scan performance is independent of key position.

**Test Cases:**
- Scan at 0%, 10%, 25%, 50%, 75%, 90%, 99% positions

**Expected Result:**
All positions should have similar latency (±10%).

---

## Optimization Evaluation Benchmarks

### 5. Serialization Overhead (`bench_serialization_overhead`)

Isolates the cost of the current serialization pipeline to quantify rkyv benefits.

**Pipeline Components:**
```
Write: rmp_serde::to_vec → LZ4 compress
Read:  LZ4 decompress → rmp_serde::from_slice
```

**Test Cases (by payload size):**
- DataUrl creation: 50, 200, 500, 2000 bytes
- MessagePack serialize/deserialize
- LZ4 compress/decompress
- Full pipeline (combined)

**Metrics:**
- Per-operation latency for each component
- Memory allocation count (via allocator profiling)

**Expected Results After rkyv:**
| Component | Current | After rkyv | Improvement |
|-----------|---------|------------|-------------|
| Deserialize 500 bytes | ~5µs | ~0.1µs | 50x |
| Deserialize 2000 bytes | ~15µs | ~0.1µs | 150x |
| Allocations per edge | 3-5 | 0 | ∞ |

### 6. Value Size Impact (`bench_value_size_impact`)

Measures how scan performance degrades with larger values, quantifying blob separation benefit.

**Test Cases:**
- 0 bytes summary (topology only)
- 100 bytes summary (small metadata)
- 500 bytes summary (typical content)
- 2000 bytes summary (large content)

**Metrics:**
- Scan latency per edge
- Throughput (edges/sec)

**Expected Results:**
| Summary Size | Current | After Blob Separation | Improvement |
|--------------|---------|----------------------|-------------|
| 0 bytes | ~50µs | ~50µs | 1x (baseline) |
| 500 bytes | ~150µs | ~50µs | 3x |
| 2000 bytes | ~400µs | ~50µs | 8x |

### 7. Transaction vs Channel Dispatch (`bench_transaction_vs_channel`)

Compares channel-based async queries with synchronous transaction reads.

**Test Cases:**
- Channel-based NodeById (early, middle, late)
- Channel-based OutgoingEdges (early, middle, late)
- Transaction-based reads (future, when Transaction supports direct reads)

**Metrics:**
- Per-query latency
- Channel overhead component

**Expected Results:**
| Query Type | Channel | Direct | Overhead |
|------------|---------|--------|----------|
| NodeById (cached) | ~60µs | ~5µs | 12x |
| OutgoingEdges | ~100µs | ~50µs | 2x |

**Note:** The REVIEW.md evaluation concluded that Direct Read Path should be deferred because:
1. Transaction API already provides sync reads for mutations
2. Channel overhead is negligible at 1B scale (0.01% of search latency)
3. Risk of misuse breaking transaction semantics

### 8. Batch Scan Throughput (`bench_batch_scan_throughput`)

Simulates graph algorithm workloads (BFS, PageRank) with sequential scans.

**Test Cases:**
- 1K nodes, 100 scans
- 5K nodes, 100 scans
- 10K nodes, 100 scans

**Metrics:**
- Total edges scanned
- Throughput (scans/sec)
- Latency distribution

**Expected Results After Blob Separation + rkyv:**
| DB Size | Current | Optimized | Improvement |
|---------|---------|-----------|-------------|
| 1K nodes | ~500ms | ~50ms | 10x |
| 10K nodes | ~5s | ~500ms | 10x |

### 9. Write Throughput by Size (`bench_write_throughput_by_size`)

Measures write performance impact of blob separation (dual CF writes).

**Test Cases:**
- 0 bytes summary
- 100 bytes summary
- 500 bytes summary

**Metrics:**
- Write throughput (nodes+edges per second)
- Comparison before/after blob separation

**Expected Results:**
Blob separation adds ~10-20% write overhead due to dual CF writes, but this is acceptable given the 10x+ read improvement.

---

## Running Benchmarks

### Full Suite
```bash
cargo bench --manifest-path libs/db/Cargo.toml
```

### Specific Benchmark Group
```bash
# Run baseline benchmarks only
cargo bench --manifest-path libs/db/Cargo.toml -- baseline_benches

# Run optimization evaluation benchmarks only
cargo bench --manifest-path libs/db/Cargo.toml -- optimization_benches

# Run specific benchmark
cargo bench --manifest-path libs/db/Cargo.toml -- serialization_overhead
cargo bench --manifest-path libs/db/Cargo.toml -- value_size_impact
```

### Before/After Comparison

#### Step 1: Establish Baseline
```bash
cargo bench --manifest-path libs/db/Cargo.toml -- --save-baseline before
```

#### Step 2: Implement Optimization
(Make code changes)

#### Step 3: Compare
```bash
cargo bench --manifest-path libs/db/Cargo.toml -- --baseline before
```

### Generate Flamegraphs
```bash
# Requires: cargo install flamegraph
cargo flamegraph --bench db_operations -- --bench
```

---

## Optimization Implementation Workflow

### Phase 1: Blob Separation

**Implementation Steps:**
1. Add new column families: `forward_edges_hot`, `edge_summaries`, `nodes_hot`, `node_summaries`
2. Modify `MutationExecutor` to write to both CFs
3. Modify scan queries to read hot CF only
4. Modify point lookups to join hot + cold when summary needed

**Benchmark Workflow:**
```bash
# Before
cargo bench --manifest-path libs/db/Cargo.toml -- value_size_impact --save-baseline before_blob_sep

# Implement blob separation

# After
cargo bench --manifest-path libs/db/Cargo.toml -- value_size_impact --baseline before_blob_sep
```

**Expected Improvement:**
- `value_size_impact/500_bytes_summary`: 3x faster
- `value_size_impact/2000_bytes_summary`: 8x faster
- `batch_scan_throughput/*`: 5-10x faster

### Phase 2: Zero-Copy Serialization (rkyv)

**Implementation Steps:**
1. Add `rkyv` dependency
2. Create hot CF schema with rkyv derives
3. Remove LZ4 compression for hot CFs
4. Keep LZ4 + rmp_serde for cold CFs

**Benchmark Workflow:**
```bash
# Before
cargo bench --manifest-path libs/db/Cargo.toml -- serialization_overhead --save-baseline before_rkyv

# Implement rkyv for hot CFs

# After
cargo bench --manifest-path libs/db/Cargo.toml -- serialization_overhead --baseline before_rkyv
```

**Expected Improvement:**
- `serialization_overhead/full_deserialize_*`: 10-50x faster
- `batch_scan_throughput/*`: Additional 2-5x improvement

---

## Key Metrics to Track

### Per-Optimization Metrics

| Metric | Benchmark | Blob Sep Target | rkyv Target |
|--------|-----------|-----------------|-------------|
| Edge scan latency | `value_size_impact` | 3-8x faster | 2-5x faster |
| Batch throughput | `batch_scan_throughput` | 5-10x faster | 2-5x faster |
| Deserialize time | `serialization_overhead` | No change | 10-50x faster |
| Write throughput | `write_throughput_by_size` | -10% to -20% | No change |

### Cache Efficiency Metrics

| Metric | Current | After Blob Sep | Calculation |
|--------|---------|----------------|-------------|
| Edge entry size | ~500 bytes | ~30 bytes | 16x smaller |
| Edges per 1GB cache | ~2M | ~33M | 16x more |
| Cache hit rate | Varies | Higher | Depends on workload |

---

## Interpreting Results

### Good Signs
- `value_size_impact`: Flat latency across all summary sizes after blob separation
- `serialization_overhead`: Significant reduction in `full_deserialize_*` after rkyv
- `batch_scan_throughput`: Linear scaling with database size
- `scan_position_independence`: All positions within ±10% of each other

### Warning Signs
- `value_size_impact`: Latency scales linearly with summary size (blob separation not working)
- `write_throughput_by_size`: >30% regression (dual CF write too expensive)
- Position-dependent scan latency (prefix extraction misconfigured)

---

## HTML Reports

Criterion generates detailed HTML reports at:
```
target/criterion/<benchmark_name>/report/index.html
```

These include:
- Performance graphs over time
- Statistical analysis (mean, median, std dev)
- Regression detection
- Comparison with baselines

---

## Integration with CI

### GitHub Actions Workflow (Optional)

```yaml
# .github/workflows/benchmark.yml
name: Benchmarks

on:
  push:
    branches: [main, feature/*]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-action@stable

      - name: Run benchmarks
        run: cargo bench --manifest-path libs/db/Cargo.toml -- --noplot

      - name: Upload benchmark results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: target/criterion/
```

### Regression Detection

To detect performance regressions on PRs:
```bash
# On main branch
cargo bench --manifest-path libs/db/Cargo.toml -- --save-baseline main

# On PR branch
cargo bench --manifest-path libs/db/Cargo.toml -- --baseline main
```

Criterion will report if any benchmark regressed significantly.

---

## Summary

This benchmark plan provides:
1. ✅ Baseline performance metrics for current implementation
2. ✅ Optimization-specific benchmarks to validate improvements
3. ✅ Before/after comparison framework
4. ✅ Clear expected results for each optimization
5. ✅ CI integration guidance

The benchmarks enable data-driven decisions about which optimizations to pursue and verify that implementations achieve expected gains.

### Priority Implementation Order

Based on REVIEW.md analysis:

| Priority | Optimization | Expected Gain | Effort | Benchmark |
|----------|--------------|---------------|--------|-----------|
| 1 | Blob Separation | 10-20x cache | Medium | `value_size_impact` |
| 2 | Zero-Copy (rkyv) | 2-5x scan | High | `serialization_overhead` |
| 3 | Direct Read Path | Deferred | - | See REVIEW.md §Evaluation |
| 4 | Fulltext Sync | Correctness | Low | N/A |
| 5 | Iterator Scans | Ergonomics | Medium | N/A |
