# Vector Subsystem Baseline Benchmarks

**Status:** Proposed
**Date:** January 16, 2026
**Purpose:** Establish reproducible performance baselines for regression tracking

---

## Overview

This document defines a comprehensive baseline benchmark suite for the vector subsystem. The goals are:

1. **Establish performance baselines** for key workload patterns
2. **Enable regression detection** in CI/CD pipelines
3. **Document expected performance** for capacity planning
4. **Validate concurrent correctness** under load

CODEX: Concurrent correctness is primarily validated by `test_vector_concurrent.rs`; baseline benchmarks currently do not assert correctness beyond successful completion.

### Minimum Requirements

All baseline benchmarks MUST meet these minimum requirements:

| Requirement | Minimum | Rationale |
|-------------|---------|-----------|
| **Vector count** | 10,000 per embedding | Realistic index size for HNSW graph structure |
| **Embedding spaces** | 2 | Validate multi-tenancy under load |
| **Duration** | 30 seconds | Statistical significance |
| **Environment docs** | Complete | Reproducibility |

CODEX: Current `ConcurrentBenchmark::run` benchmarks a single embedding. Minimum requirement "Embedding spaces = 2" is not satisfied by the existing benchmark tests.
CODEX: Default `BenchConfig` uses `vectors_per_writer=1000`, so 10k vectors/embedding requires explicit overrides in baseline tests.

---

## Benchmark Scenarios

### 1. Single-Threaded Baseline

Measure raw operation performance without concurrency overhead.

| Metric | Description | Target |
|--------|-------------|--------|
| Insert latency | Single vector insert (vector + HNSW) | <10ms |
| Search latency | Single KNN search (k=10, ef=50) | <1ms |
| Batch insert | 100 vectors in single transaction | <500ms |

**Configuration:**
```rust
BenchConfig {
    writer_threads: 1,
    reader_threads: 0,
    duration: Duration::from_secs(30),
    vectors_per_writer: 10_000,
    vector_dim: 128,
    hnsw_m: 16,
    hnsw_ef_construction: 100,
    k: 10,
    ef_search: 50,
}
```
CODEX: No single-threaded baseline test exists in `test_vector_concurrent.rs`. Add a dedicated ignored test or configure `BenchConfig` explicitly.

### 2. Read-Heavy Workload

Simulates CDN/cache access patterns with many readers, few writers.

| Metric | Target | Rationale |
|--------|--------|-----------|
| Search throughput | >5,000 ops/sec | High read concurrency |
| Search P99 | <20ms | Tail latency matters for UX |
| Insert throughput | >100 ops/sec | Background ingestion |
| Error rate | <1% | Stable under load |

**Configuration:**
```rust
BenchConfig::read_heavy()  // 1 writer, 8 readers
```
CODEX: Implemented as `benchmark_baseline_read_heavy`, but still single-embedding and default `vectors_per_writer` unless overridden.

### 3. Write-Heavy Workload

Simulates batch ingestion with high insert concurrency.

| Metric | Target | Rationale |
|--------|--------|-----------|
| Insert throughput | >500 ops/sec | Bulk ingestion |
| Insert P99 | <100ms | Acceptable for background jobs |
| Search throughput | >100 ops/sec | Verification queries |
| Error rate | <5% | Transaction conflicts expected |

**Configuration:**
```rust
BenchConfig::write_heavy()  // 8 writers, 1 reader
```
CODEX: Implemented as `benchmark_baseline_write_heavy`, but still single-embedding and default `vectors_per_writer` unless overridden.

### 4. Balanced Workload

Simulates mixed production traffic.

| Metric | Target | Rationale |
|--------|--------|-----------|
| Insert throughput | >200 ops/sec | Steady ingestion |
| Search throughput | >2,000 ops/sec | User queries |
| Insert P99 | <50ms | Interactive acceptable |
| Search P99 | <30ms | UX requirement |
| Error rate | <2% | Production stability |

**Configuration:**
```rust
BenchConfig::balanced()  // 4 writers, 4 readers
```
CODEX: Implemented as `benchmark_baseline_balanced`, but still single-embedding and default `vectors_per_writer` unless overridden.

### 5. Stress Test

Maximum concurrency to find bottlenecks and breaking points.

| Metric | Target | Rationale |
|--------|--------|-----------|
| No panics | 100% | Stability requirement |
| No data corruption | 100% | Correctness requirement |
| Error rate | <10% | Graceful degradation |
| Throughput | Measure only | Find limits |

**Configuration:**
```rust
BenchConfig::stress()  // 16 writers, 16 readers
```
CODEX: Implemented as `benchmark_baseline_stress`; still single-embedding.

---

## Metrics Collected

### Operation Metrics

| Metric | Type | Collection |
|--------|------|------------|
| `insert_count` | Counter | Total successful inserts |
| `search_count` | Counter | Total successful searches |
| `delete_count` | Counter | Total successful deletes |
| `error_count` | Counter | Failed operations |

CODEX: No delete workload is currently executed by `ConcurrentBenchmark`; delete metrics will remain zero unless a delete phase is added.

### Latency Metrics

| Metric | Type | Buckets |
|--------|------|---------|
| `insert_latency` | Histogram | Log2 (1µs - 1s) |
| `search_latency` | Histogram | Log2 (1µs - 1s) |
| `delete_latency` | Histogram | Log2 (1µs - 1s) |

### Derived Metrics

| Metric | Formula |
|--------|---------|
| Insert throughput | `insert_count / duration_sec` |
| Search throughput | `search_count / duration_sec` |
| Error rate | `error_count / total_ops` |
| P50/P95/P99 | Histogram percentile (upper bound) |

---

## Test Parameters

### Vector Configuration

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `vector_dim` | 128 | 64-1024 | Higher dims = slower |
| `vectors_per_writer` | 1000 | 100-10000 | Per-thread limit |

### HNSW Configuration

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `hnsw_m` | 16 | 8-32 | Higher = better recall, slower |
| `hnsw_ef_construction` | 100 | 50-200 | Build quality |
| `ef_search` | 50 | 20-200 | Search quality |
| `k` | 10 | 1-100 | Neighbors to return |

### Concurrency Configuration

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `writer_threads` | 4 | 1-16 | Insert concurrency |
| `reader_threads` | 4 | 1-16 | Search concurrency |
| `duration` | 30s | 5s-300s | Test duration |

---

## Environment Requirements

### Hardware Documentation Template

Every recorded baseline MUST document the complete hardware environment:

```
=== Hardware Environment ===
Machine Type: [Physical / VM / Cloud instance type]
CPU Model: [Full model name from /proc/cpuinfo or lscpu]
CPU Architecture: [x86_64 / aarch64 / etc.]
Physical Cores: [N]
Logical Cores: [N threads]
CPU Frequency: [Base / Boost GHz]
L1/L2/L3 Cache: [Sizes]
RAM Total: [N GB]
RAM Type: [DDR4/DDR5, Speed]
Storage Type: [NVMe SSD / SATA SSD / HDD]
Storage Model: [Model name if available]
Storage IOPS: [Advertised or measured]
```

### Software Configuration Template

```
=== Software Environment ===
OS: [Distribution and version]
Kernel: [uname -r output]
Rust Version: [rustc --version]
Cargo Version: [cargo --version]
LLVM Version: [if relevant]
RocksDB Version: [from Cargo.lock]
Build Profile: release
Build Flags: [Any custom RUSTFLAGS]
SIMD: [Enabled features: AVX2/NEON/etc.]
```

### Capture Environment Script

```bash
#!/bin/bash
echo "=== Hardware Environment ==="
echo "Machine Type: $(hostnamectl | grep 'Chassis' | cut -d: -f2 | xargs)"
echo "CPU Model: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
echo "CPU Architecture: $(uname -m)"
echo "Physical Cores: $(lscpu | grep '^CPU(s):' | cut -d: -f2 | xargs)"
echo "CPU Frequency: $(lscpu | grep 'MHz' | head -1 | cut -d: -f2 | xargs) MHz"
echo "RAM Total: $(free -h | grep Mem | awk '{print $2}')"
echo "Storage: $(df -h . | tail -1 | awk '{print $1}')"

echo ""
echo "=== Software Environment ==="
echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '"')"
echo "Kernel: $(uname -r)"
echo "Rust: $(rustc --version)"
echo "Cargo: $(cargo --version)"
```

### Pre-run Checklist

- [ ] Environment documented using template above
- [ ] System idle (no background load >5% CPU)
- [ ] Fresh storage directory (no pre-existing data)
- [ ] Release build (`cargo build --release`)
- [ ] Warmup run completed and discarded
- [ ] Multiple runs averaged (minimum 3 for variance)

---

## Running Benchmarks

### Quick Validation (CI)

```bash
# 5-second smoke test
cargo test -p motlie-db --test test_vector_concurrent benchmark_quick_validation -- --nocapture
```

### Full Baseline Suite

```bash
# Run all baseline benchmarks (ignored by default)
cargo test -p motlie-db --test test_vector_concurrent benchmark_baseline -- --ignored --nocapture

# Individual scenarios
cargo test -p motlie-db --test test_vector_concurrent benchmark_baseline_balanced -- --ignored --nocapture
cargo test -p motlie-db --test test_vector_concurrent benchmark_baseline_read_heavy -- --ignored --nocapture
cargo test -p motlie-db --test test_vector_concurrent benchmark_baseline_write_heavy -- --ignored --nocapture
cargo test -p motlie-db --test test_vector_concurrent benchmark_baseline_stress -- --ignored --nocapture
```

### Capturing Results

```bash
# Run with output capture
cargo test -p motlie-db --test test_vector_concurrent benchmark_baseline -- --ignored --nocapture 2>&1 | tee benchmark_results.txt

# Parse key metrics
grep -E "(Insert|Search|throughput|P99)" benchmark_results.txt
```

---

## Interpreting Results

### Throughput Analysis

| Observation | Likely Cause | Action |
|-------------|--------------|--------|
| Insert << target | Write contention | Reduce writers, increase batch size |
| Search << target | Lock contention | Check cache efficiency |
| High variance | GC/compaction | Longer test duration |

### Latency Analysis

| Observation | Likely Cause | Action |
|-------------|--------------|--------|
| P99 >> P50 | Tail latency spikes | Check for RocksDB compaction |
| Insert P99 high | HNSW graph updates | Reduce M, use background updates |
| Search P99 high | Cache misses | Increase cache size |

### Error Analysis

| Error Type | Cause | Mitigation |
|------------|-------|------------|
| Empty index | Search before insert | Gate on index readiness |
| Transaction conflict | Write contention | Retry with backoff |
| Timeout | Overload | Reduce concurrency |

---

## Regression Tracking

### Baseline Thresholds

Define acceptable regression margins:

| Metric | Baseline | Regression Threshold |
|--------|----------|---------------------|
| Insert throughput | X ops/sec | >20% drop |
| Search throughput | X ops/sec | >20% drop |
| Insert P99 | X ms | >50% increase |
| Search P99 | X ms | >50% increase |
| Error rate | X% | >2x increase |

### CI Integration

```yaml
# Example CI job
benchmark:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Run benchmarks
      run: |
        cargo test -p motlie-db --test test_vector_concurrent benchmark_quick_validation -- --nocapture
    - name: Check regression
      run: |
        # Compare against stored baseline
        ./scripts/check_benchmark_regression.sh
```

### Baseline Update Process

1. Run full benchmark suite on reference hardware
2. Review results for anomalies
3. Update baseline values in this document
4. Commit with benchmark run ID and date
5. Tag release with baseline version

---

## Recorded Baselines

### Baseline Template

Each recorded baseline must include:

1. **Environment** - Full hardware/software documentation (see templates above)
2. **Configuration** - Exact benchmark parameters used
3. **Results** - All metrics with statistical context
4. **Validation** - Confirmation of minimum requirements met

### Baseline v1 (Pending)

**Status:** Not yet recorded - requires full benchmark run

**Minimum Requirements Checklist:**
- [ ] 10,000+ vectors per embedding
- [ ] 2+ embedding spaces
- [ ] 30+ second duration per scenario
- [ ] Environment fully documented
- [ ] Release build

**Environment:**
```
=== Hardware Environment ===
Machine Type: [TBD]
CPU Model: [TBD]
CPU Architecture: [TBD]
Physical Cores: [TBD]
Logical Cores: [TBD]
CPU Frequency: [TBD]
RAM Total: [TBD]
Storage Type: [TBD]

=== Software Environment ===
OS: [TBD]
Kernel: [TBD]
Rust Version: [TBD]
RocksDB Version: [TBD]
Build Profile: release
SIMD: [TBD]
```

**Configuration:**
```
Embeddings: 2 (embedding_a: 128D L2, embedding_b: 128D Cosine)
Vectors per embedding: 10,000
Duration: 30s per scenario
HNSW: M=16, ef_construction=100, ef_search=50
```
CODEX: Multi-embedding benchmark harness is not implemented yet; this configuration is aspirational until `ConcurrentBenchmark` can run multiple embeddings.

**Results:** [TBD - run benchmark suite]

| Scenario | Embeddings | Vectors | Insert/s | Search/s | Insert P99 | Search P99 | Errors |
|----------|------------|---------|----------|----------|------------|------------|--------|
| Read-heavy | 2 | 20,000 | - | - | - | - | - |
| Write-heavy | 2 | 20,000 | - | - | - | - | - |
| Balanced | 2 | 20,000 | - | - | - | - | - |
| Stress | 2 | 20,000 | - | - | - | - | - |

### Historical Quick Validation (Reference Only)

*Note: This quick validation does NOT meet minimum requirements (insufficient scale/embeddings). Retained for reference only.*

**Config:** 2 writers, 2 readers, 5s, 64D, 1 embedding, ~500 vectors

| Metric | Value |
|--------|-------|
| Insert throughput | 77.6 ops/sec |
| Search throughput | 37,409 ops/sec |
| Insert P99 | 33ms |
| Search P99 | 32µs |

---

## Future Enhancements

### Batch Mode Benchmarks

Currently all benchmarks use per-vector transactions. Add batch mode:

```rust
BenchConfig::balanced()
    .with_batch_size(100)  // Commit every 100 vectors
```

**Priority:** Low
**Rationale:** Production often batches; current approach may understate throughput.

### Extended Multi-Embedding Benchmarks

Current baseline requires 2 embeddings. Future enhancement to test more:

```rust
BenchConfig::multi_embedding(5)  // 5 embeddings, balanced load each
```

**Priority:** Medium
**Rationale:** Some production deployments have many embedding models (5-10+).

### Recall Under Load

Measure search quality (recall@k) while under concurrent write load:

```rust
BenchResult {
    // ... existing fields
    recall_at_10: f64,  // vs ground truth
}
```

**Priority:** Medium
**Rationale:** Ensure concurrent writes don't degrade search quality.

---

## References

- [CONCURRENT.md](./CONCURRENT.md) - Concurrent operations implementation
- [ROADMAP.md](./ROADMAP.md) - Full implementation roadmap
- [PHASE5.md](./PHASE5.md) - Phase 5 task tracking
- [benchmark/concurrent.rs](./benchmark/concurrent.rs) - Benchmark implementation
