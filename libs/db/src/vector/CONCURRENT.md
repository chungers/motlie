# Phase 5.9-5.11: Concurrent Operations

**Status:** ✅ Complete
**Date:** January 16, 2026
**Tasks:** 5.9 (Stress Tests), 5.10 (Metrics), 5.11 (Benchmarks)

---

## Overview

This document tracks the implementation of concurrent operation testing and benchmarking for the vector subsystem. The goal is to validate thread safety, measure performance under concurrent load, and establish baseline metrics for production readiness.

---

## Task 5.9: Multi-Threaded Stress Tests

**Status:** ✅ Complete

**Goal:** Validate concurrent access patterns under load without data corruption or panics.

### Test Categories

| Test | Description | Thread Config | Status |
|------|-------------|---------------|--------|
| `test_concurrent_insert_search` | Atomic insert while searching | 2 inserters, 2 searchers | ✅ |
| `test_concurrent_batch_insert` | Atomic parallel batch inserts | 4 batch inserters | ✅ |
| `test_high_thread_count_stress` | Atomic maximum concurrency | 8 writers, 8 readers | ✅ |
| `test_writer_contention` | Atomic writers same embedding | 8 writers | ✅ |
| `test_multi_embedding_concurrent_access` | Multi-index concurrent r/w | 3 indices, 6 writers, 6 readers | ✅ |
| `test_cache_isolation_under_load` | Cache isolation validation | 2 indices, 2 writers, 2 readers | ✅ |

### Implementation

**File:** `libs/db/tests/test_vector_concurrent.rs`

All tests verify:
1. No thread panics (thread safety)
2. Data integrity maintained (inserted vectors searchable)
3. Operations complete successfully under concurrent load

### Validation Criteria

- [x] No panics under any thread configuration
- [x] No data corruption (inserted vectors retrievable)
- [x] Search results consistent (no phantom results)

---

## Task 5.10: Metrics Collection Infrastructure

**Status:** ✅ Complete

**Goal:** Add lock-free metrics collection for concurrent operation analysis.

### Metrics to Collect

| Metric | Type | Description |
|--------|------|-------------|
| `insert_count` | Counter | Total insert operations |
| `search_count` | Counter | Total search operations |
| `delete_count` | Counter | Total delete operations |
| `insert_latency_ns` | Histogram | Per-operation insert latency |
| `search_latency_ns` | Histogram | Per-operation search latency |
| `delete_latency_ns` | Histogram | Per-operation delete latency |
| `errors` | Counter | Operation failures |

### Implementation

**File:** `libs/db/src/vector/benchmark/concurrent.rs`

Implemented `ConcurrentMetrics` with:
- Atomic counters for lock-free updates (`AtomicU64`)
- Log2 histogram buckets (20 buckets from 1us to ~1s)
- Methods: `record_insert()`, `record_search()`, `record_delete()`, `record_error()`
- Percentile calculations: `insert_percentile()`, `search_percentile()`
- Summary generation: `summary() -> MetricsSummary`

### Histogram Bucket Design

```
Bucket 0:  0-1us
Bucket 1:  1-2us
Bucket 2:  2-4us
Bucket 3:  4-8us
...
Bucket 19: 512ms-1s+
```

Log2 buckets provide good resolution across the full latency range with minimal memory overhead.

**Interpreting Percentiles:** Since buckets are coarse (powers of 2), reported p50/p95/p99
values represent the **upper bound** of the bucket containing that percentile, not exact
latency values. For example, if p99 is reported as 8ms, the actual p99 latency is somewhere
in the 4-8ms range. This trade-off allows lock-free updates without storing individual samples.

---

## Task 5.11: Concurrent Benchmark Baseline

**Status:** ✅ Complete

**Goal:** Establish baseline metrics for concurrent operations.

### Benchmark Scenarios

| Scenario | Writers | Readers | Duration | Purpose |
|----------|---------|---------|----------|---------|
| Read-heavy | 1 | 8 | 30s | CDN/cache workload |
| Write-heavy | 8 | 1 | 30s | Ingestion workload |
| Balanced | 4 | 4 | 30s | Mixed workload baseline |
| Stress | 16 | 16 | 60s | Find bottlenecks |

### Implementation

**File:** `libs/db/src/vector/benchmark/concurrent.rs`

Implemented:
- `BenchConfig` with preset configurations:
  - `read_heavy()` - 1 writer, 8 readers
  - `write_heavy()` - 8 writers, 1 reader
  - `balanced()` - 4 writers, 4 readers
  - `stress()` - 16 writers, 16 readers
- `ConcurrentBenchmark::run()` - Spawns writer/reader threads, collects metrics
- `BenchResult` - Throughput, latencies (p50/p95/p99), error counts

**Exports:** All types exported via `motlie_db::vector::benchmark::*`

### Expected Baseline (Target)

| Scenario | Insert/s | Search/s | Insert P99 | Search P99 |
|----------|----------|----------|------------|------------|
| Read-heavy | 500 | 2000 | <50ms | <20ms |
| Write-heavy | 2000 | 200 | <50ms | <50ms |
| Balanced | 1000 | 1000 | <50ms | <30ms |
| Stress | 500 | 500 | <100ms | <50ms |

*Note: Targets based on 100K vectors, 128D, M=16, ef=100*

---

## Multi-Tenancy Verification

**Status:** ✅ Verified (January 16, 2026)

Multiple embedding indexes coexist safely in:
- **Same RocksDB database** (same directory path)
- **Same `Storage` instance** (single `Arc<Storage>`)
- **Same MPSC mutation/query interfaces** (single writer, single reader)

### Architecture Review

All components are correctly scoped by `EmbeddingCode`:

| Component | Key Structure | Multi-Tenant? |
|-----------|---------------|---------------|
| **Schema (all CFs)** | `(EmbeddingCode, ...)` prefix | ✅ |
| **NavigationCache** | `HashMap<EmbeddingCode, NavInfo>` + `(EmbeddingCode, VecId, Layer)` | ✅ |
| **BinaryCodeCache** | `HashMap<(EmbeddingCode, VecId), Entry>` | ✅ |
| **HNSW Index** | Each Index stores `embedding: EmbeddingCode` | ✅ |
| **RaBitQ Encoders** | `DashMap<EmbeddingCode, Arc<RaBitQ>>` | ✅ |
| **IdAllocators** | `DashMap<EmbeddingCode, IdAllocator>` | ✅ |

### Schema Key Prefixing

All column families use `EmbeddingCode` (8-byte big-endian) as the first key component:

```
Vectors CF:     (embedding_code, vec_id)           → vector data
Edges CF:       (embedding_code, vec_id, layer)    → HNSW neighbors
BinaryCodes CF: (embedding_code, vec_id)           → RaBitQ codes
VecMeta CF:     (embedding_code, vec_id)           → layer assignment
GraphMeta CF:   (embedding_code, field)            → entry points, max_layer
IdForward CF:   (embedding_code, ulid)             → ULID → VecId
IdReverse CF:   (embedding_code, vec_id)           → VecId → ULID
IdAlloc CF:     (embedding_code, field)            → next_id, free bitmap
```

Prefix extractors ensure efficient scans within each embedding's keyspace.

### Processor State Isolation

```rust
pub struct Processor {
    storage: Arc<Storage>,                              // Shared
    registry: Arc<EmbeddingRegistry>,                   // Shared
    id_allocators: DashMap<EmbeddingCode, IdAllocator>, // Per-embedding
    rabitq_encoders: DashMap<EmbeddingCode, Arc<RaBitQ>>, // Per-embedding
    hnsw_indices: DashMap<EmbeddingCode, hnsw::Index>,  // Per-embedding
    nav_cache: Arc<NavigationCache>,                    // Shared (embedding-keyed)
    code_cache: Arc<BinaryCodeCache>,                   // Shared (embedding-keyed)
}
```

### Integration Test Certification

**File:** `libs/db/tests/test_vector_multi_embedding.rs`

**Test:** `test_multi_embedding_non_interference`

Validates multi-tenancy with:
- **Single `TempDir`** → one RocksDB directory
- **Single `Storage`** → `Storage::readwrite(temp_dir.path())`
- **Single `Arc<Storage>`** → shared across all embeddings
- **Single writer channel** → all inserts through same MPSC
- **Single reader channel** → all queries through same MPSC
- **Three embeddings:**
  - LAION (512D, Cosine) - RaBitQ path
  - SIFT (128D, L2) - Exact path
  - BERT (768D, Cosine) - RaBitQ path

**Assertions:**
- [x] Each embedding gets unique code
- [x] LAION searches return ONLY LAION vectors
- [x] SIFT searches return ONLY SIFT vectors
- [x] BERT searches return ONLY BERT vectors
- [x] No cross-contamination between embeddings
- [x] Different dimensions coexist (128D, 512D, 768D)
- [x] Different distance metrics coexist (Cosine, L2)

### Additional Multi-Embedding Tests

| Test | Description | Status |
|------|-------------|--------|
| `test_cosine_2bit_rabitq` | Single embedding, exact + RaBitQ paths | ✅ |
| `test_cosine_4bit_rabitq` | Single embedding, exact + RaBitQ paths | ✅ |
| `test_multi_embedding_non_interference` | 3 embeddings, isolation verified | ✅ |
| `test_embedding_registration_idempotent` | Re-register returns same code | ✅ |
| `test_exact_vs_rabitq_search_paths` | Both search paths work | ✅ |

---

## Progress Log

### January 16, 2026

- [x] Created CONCURRENT.md plan document
- [x] Task 5.10: Implemented `ConcurrentMetrics` in `concurrent.rs`
- [x] Task 5.11: Implemented `ConcurrentBenchmark`, `BenchConfig`, `BenchResult`
- [x] Task 5.9: Implemented stress tests in `test_vector_concurrent.rs`
- [x] All unit tests passing
- [x] Exported types via `benchmark/mod.rs`
- [x] Multi-tenancy architecture review completed
- [x] Multi-embedding integration tests added (`test_vector_multi_embedding.rs`)

---

## Files Created/Modified

| File | Purpose | Status |
|------|---------|--------|
| `libs/db/tests/test_vector_concurrent.rs` | Stress tests (5.9) | ✅ |
| `libs/db/tests/test_vector_multi_embedding.rs` | Multi-tenancy certification | ✅ |
| `libs/db/src/vector/benchmark/concurrent.rs` | Metrics + Benchmarks (5.10, 5.11) | ✅ |
| `libs/db/src/vector/benchmark/mod.rs` | Export new types | ✅ |

---

## Future Work: Migration Opportunities

The new concurrent infrastructure can benefit existing tests and benchmarks. This section outlines migration candidates.

### 1. Graph Concurrent Tests → `ConcurrentMetrics`

**Current:** `libs/db/tests/common/concurrent_test_utils.rs`

```rust
// BEFORE: Non-atomic, single operation type
pub struct Metrics {
    pub success_count: u64,
    pub error_count: u64,
    pub total_latency_us: u64,
    pub min_latency_us: u64,
    pub max_latency_us: u64,
}
```

**Issues:**
- Not thread-safe (requires `&mut self` for updates)
- No histogram/percentile support
- No operation type differentiation (reads vs writes)

**Migration:**
- Replace with `ConcurrentMetrics` for lock-free updates
- Gains: Atomic counters, percentile calculations, separate read/write tracking

**Affected Tests:**
| Test File | Description |
|-----------|-------------|
| `test_concurrent_readonly.rs` | Readonly instance concurrent reads |
| `test_concurrent_secondary.rs` | Secondary instance catch-up |
| `test_concurrent_shared.rs` | Shared storage concurrent access |

**Priority:** Medium - Improves test reliability under high contention

---

### 2. Vector Benchmark Runner → Concurrent Scenarios

**Current:** `libs/db/src/vector/benchmark/runner.rs`

The existing `ExperimentConfig` and `run_single_experiment()` are single-threaded.

**Migration Opportunity:**
- Add `run_concurrent_experiment()` using `ConcurrentBenchmark`
- Measure recall under concurrent load (writer + searchers)
- Validate that recall doesn't degrade with concurrent writes

**New Tests to Add:**
| Test | Description |
|------|-------------|
| `test_recall_under_concurrent_writes` | Measure recall while inserting |
| `test_throughput_vs_recall_tradeoff` | Pareto frontier under load |

**Priority:** Low - Current benchmarks are single-threaded by design

---

### 3. Graph Concurrent Benchmarks (New)

**Opportunity:** Create concurrent benchmarks for graph operations using the new infrastructure.

**Proposed:**
```rust
// libs/db/src/graph/benchmark/concurrent.rs
pub struct GraphConcurrentBenchmark {
    config: BenchConfig,
    metrics: Arc<ConcurrentMetrics>,
}

impl GraphConcurrentBenchmark {
    pub fn run(&self, storage: Arc<Storage>) -> BenchResult {
        // Spawn writer threads (AddNode, AddEdge)
        // Spawn reader threads (NodeById, OutgoingEdges)
        // Collect metrics
    }
}
```

**Benefits:**
- Unified metrics infrastructure across graph and vector subsystems
- Comparable throughput/latency measurements
- Shared `BenchConfig` presets (read_heavy, write_heavy, etc.)

**Priority:** Low - Graph subsystem has different access patterns

---

### 4. Unified Metrics API (Future)

**Goal:** Single metrics interface for all subsystems.

```rust
// Potential future API
pub trait OperationMetrics {
    fn record_operation(&self, op_type: OpType, latency: Duration);
    fn record_error(&self, op_type: OpType);
    fn summary(&self) -> MetricsSummary;
}

pub enum OpType {
    // Vector ops
    VectorInsert,
    VectorSearch,
    VectorDelete,
    // Graph ops
    NodeInsert,
    NodeQuery,
    EdgeInsert,
    EdgeQuery,
}
```

**Priority:** Future - Consider when adding more subsystems

---

## Migration Checklist

| Item | Status | Priority |
|------|--------|----------|
| Graph concurrent tests → `ConcurrentMetrics` | ⏳ Pending | Medium |
| Vector runner + concurrent scenarios | ⏳ Pending | Low |
| Graph concurrent benchmarks | ⏳ Pending | Low |
| Unified metrics trait | ⏳ Future | Low |

---

## References

- [PHASE5.md](./PHASE5.md) - Phase 5 task tracking
- [ROADMAP.md](./ROADMAP.md) - Full implementation roadmap
- [API.md](./API.md) - Public API reference

---

## Usage Guidance

### When to Use Concurrent Tests vs Benchmarks

| Tool | Purpose | Use When |
|------|---------|----------|
| **Stress Tests** (`test_vector_concurrent.rs`) | Validate correctness under load | CI/CD, detecting race conditions, verifying thread safety |
| **Concurrent Benchmarks** (`ConcurrentBenchmark`) | Measure throughput/latency | Performance tuning, capacity planning, regression testing |
| **Integration Tests** (`test_vector_multi_embedding.rs`) | Validate API correctness | Feature development, multi-tenancy verification |

**Key Differences:**
- Stress tests prioritize **no panics** and **data integrity** over throughput numbers
- Benchmarks prioritize **accurate measurements** and **reproducibility**
- Integration tests prioritize **API contract validation** and **isolation guarantees**

### Error Rate Expectations

The 10% error threshold in stress tests accounts for expected transient conditions:

1. **Empty graph searches** - Before any insert commits, searches return an error (no entry point)
2. **Transaction conflicts** - RocksDB optimistic concurrency may reject concurrent writes

These are documented in `test_vector_concurrent.rs` module-level comments. In production:
- Gate searches until index reports ready, or
- Handle "empty index" errors gracefully in application code

### vec_id Generation Bounds

Test configurations use bounded vec_id generation:
- `(thread_id << 16) | i` → max 65,536 vectors per thread (4 threads = 262K max)
- Atomic counter → max 2^32 vectors total

These bounds are documented in test file headers. Increase thread_id shift if larger configs needed.

### Benchmark Transaction Modes

Current benchmarks use per-vector transactions for correctness validation. For pure throughput
measurement without transaction overhead, consider:

```rust
// Future: Batch mode for throughput benchmarks
let config = BenchConfig::balanced()
    .with_batch_size(100);  // Commit every 100 vectors
```

This is not yet implemented but noted for future enhancement.

---

## CODEX Review Notes (Post-sync)

**Status:** ✅ Addressed (January 16, 2026)

### Correctness / Reliability

- ✅ **Non-atomic insert in stress tests:** Fixed. Created `insert_vector_atomic()` helper that writes vector data and HNSW graph in a single transaction. All 4 original stress tests updated to use this atomic helper. No orphaned vectors possible.

- ✅ **Error threshold still broad:** Addressed via documentation. Module-level comments in `test_vector_concurrent.rs` now explicitly document the two expected error conditions (empty graph search, transaction conflicts) and production mitigation strategies.

- ✅ **vec_id bounds:** Added `MAX_VECTORS_PER_THREAD = 65_536` constant with overflow prevention guard in atomic insert helper.

### Performance / Measurement Quality

- ⏳ **Per-vector transactions** *(Priority: Low)*: Current benchmarks use per-vector transactions for correctness validation. Add optional `BenchConfig::with_batch_size()` for batch commit mode to measure throughput without per-transaction overhead. Rationale: Production workloads often batch commits; current approach may understate throughput.

### Coverage Gaps / Additional Tests

- ✅ **Multi-index concurrency under shared `Storage`:** Added `test_multi_embedding_concurrent_access` which spawns concurrent writers/readers across 3 embeddings (EMBEDDING_A, EMBEDDING_B, EMBEDDING_C) and asserts no cross-contamination.

- ✅ **Cache isolation under load:** Added `test_cache_isolation_under_load` which validates NavigationCache correctness under concurrent multi-embedding access. Uses vec_id encoding with embedding prefix to detect any cross-contamination (asserts count = 0).

- ✅ **Concurrent deletes vs searches:** Added `test_concurrent_deletes_vs_searches` in `test_vector_channel.rs`. Validates tombstone filtering under concurrent delete/search load. Post-delete searches correctly exclude deleted vectors.

- ⏳ **Mixed search strategies** *(Priority: Low)*: Deferred. Concurrent RaBitQ + exact search validation requires setup with >256 vectors per embedding (RaBitQ threshold). Rationale: RaBitQ only activates above 256 vectors; test requires larger dataset setup. Add when benchmark infrastructure supports mixed-strategy scenarios.

### Updated Test Inventory

| Test | File | Description | CODEX Item |
|------|------|-------------|------------|
| `test_concurrent_insert_search` | `test_vector_concurrent.rs` | Atomic insert while searching | Fixed (atomic) |
| `test_concurrent_batch_insert` | `test_vector_concurrent.rs` | Atomic parallel batch inserts | Fixed (atomic) |
| `test_high_thread_count_stress` | `test_vector_concurrent.rs` | Atomic 16-thread stress | Fixed (atomic) |
| `test_writer_contention` | `test_vector_concurrent.rs` | Atomic 8-writer contention | Fixed (atomic) |
| `test_multi_embedding_concurrent_access` | `test_vector_concurrent.rs` | 3-embedding concurrent access | NEW |
| `test_cache_isolation_under_load` | `test_vector_concurrent.rs` | Cache isolation validation | NEW |
| `test_concurrent_deletes_vs_searches` | `test_vector_channel.rs` | Delete + search tombstone filtering | NEW |
