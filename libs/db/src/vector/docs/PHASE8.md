# Phase 8: Production Hardening

**Status:** Not Started
**Date:** January 18, 2026
**Prerequisite:** Phase 7 (Async Graph Updater) - Complete

---

## Overview

Phase 8 focuses on production readiness: ensuring deletes are clean, concurrent access is safe, and the system scales to billion-vector workloads. This phase addresses technical debt from earlier phases and validates the system under realistic production conditions.

### Problem Statement

Several gaps remain before production deployment:

1. **Delete Cleanup**: Soft deletes leave HNSW edges pointing to deleted vectors. Over time, this degrades graph quality and wastes storage.

2. **Scale Validation**: Current benchmarks only cover up to 1M vectors. Production workloads may reach 10M-1B scale with different performance characteristics.

3. **Concurrent Access Stress**: While basic concurrent tests exist (Phase 5.9-5.11), production-grade stress testing with snapshot isolation validation is needed.

### Goals

- Clean delete path with edge pruning and optional tombstone compaction
- Validated performance at 10M, 100M, and 1B vector scales
- Production-grade concurrent access stress tests with failure injection
- Memory budget validation and profiling at scale

---

## Gap Analysis

### From CONCURRENT.md Review

See [CONCURRENT.md](./CONCURRENT.md) for full concurrent test inventory and baseline results.

**Existing Infrastructure (Phase 5.9-5.11):**

| Component | Location | Status |
|-----------|----------|--------|
| `ConcurrentMetrics` | `benchmark/concurrent.rs` | ✅ Lock-free counters + histograms |
| `ConcurrentBenchmark` | `benchmark/concurrent.rs` | ✅ Throughput measurement harness |
| `BenchConfig` presets | `benchmark/concurrent.rs` | ✅ read_heavy, write_heavy, balanced, stress |
| Stress tests | `test_vector_concurrent.rs` | ✅ 6 tests covering insert/search/delete |
| Baseline benchmarks | `test_vector_concurrent.rs` | ✅ 4 baseline scenarios (ignored by default) |

**Identified Gaps:**

| Gap | Current State | Phase 8 Action |
|-----|--------------|----------------|
| Per-vector transactions in benchmarks | All benchmarks use single-vector commits ([CONCURRENT.md §Benchmark Transaction Modes](./CONCURRENT.md#benchmark-transaction-modes)) | Add batch commit mode for throughput measurement |
| Mixed search strategies under load | Not tested ([CONCURRENT.md §Coverage Gaps](./CONCURRENT.md#coverage-gaps--additional-tests)) | Add concurrent RaBitQ + exact search validation |
| Graph concurrent test migration | Uses old `Metrics` struct ([CONCURRENT.md §Migration Opportunities](./CONCURRENT.md#future-work-migration-opportunities)) | Migrate to `ConcurrentMetrics` (low priority) |

### From Delete Implementation Review

| Gap | Current State | Phase 8 Action |
|-----|--------------|----------------|
| Edge cleanup | HNSW edges point to deleted vectors | Add edge pruning during compaction or background worker |
| VecId reuse | IDs never freed when HNSW enabled | Add safe ID recycling after edges are cleaned |
| Tombstone compaction | Deleted vector data kept forever | Add compaction filter or background cleanup |
| Search filtering | Relies on IdReverse removal | Add VecMeta lifecycle check as defense-in-depth |

### From Scale Analysis

| Gap | Current State | Phase 8 Action |
|-----|--------------|----------------|
| 10M+ benchmarks | Only 1M benchmarks exist | Add 10M, 100M, 1B benchmark configs |
| Memory profiling | No systematic memory tracking | Add memory budget validation |
| Performance regression | Manual benchmark comparison | Add automated regression detection |

---

## Task Breakdown

### Task 8.1: Delete Refinement

**Status:** Not Started

**Goal:** Complete the delete lifecycle with edge cleanup, ID recycling, and tombstone compaction.

**Files:**
- `libs/db/src/vector/ops/delete.rs` - Delete operation enhancements
- `libs/db/src/vector/compaction.rs` - New: compaction filter
- `libs/db/src/vector/gc.rs` - New: garbage collection worker
- `libs/db/src/vector/schema.rs` - Add tombstone tracking

#### Current Delete Behavior (HNSW Enabled)

```
Delete(id) →
  1. Remove IdForward mapping (id → vec_id)
  2. Remove IdReverse mapping (vec_id → id)
  3. Set VecMetadata lifecycle → Deleted/PendingDeleted
  4. Remove from Pending queue if present
  5. Keep vector data (for HNSW distance calculations)
  6. Keep HNSW edges intact (tombstone approach)
  7. Do NOT free VecId (prevents graph corruption)
```

**Problem:** Steps 5-7 accumulate garbage over time.

#### Target Delete Behavior

```
Delete(id) →
  [Sync - unchanged]
  1-4. Same as current

  [Async - new background worker]
  5. Edge pruning: remove deleted vec_id from neighbor lists
  6. After edge cleanup: delete vector data + binary codes
  7. After edge cleanup: return VecId to free list
  8. Compaction: RocksDB filter removes orphaned entries
```

#### Deliverables

- [ ] 8.1.1: Add `DeletedVectors` tracking CF or bitmap for pending cleanup
- [ ] 8.1.2: Implement edge pruning in background worker
- [ ] 8.1.3: Add safe VecId recycling after edge cleanup confirms no references
- [ ] 8.1.4: Implement RocksDB compaction filter for tombstone cleanup
- [ ] 8.1.5: Add VecMeta lifecycle check in HNSW search as defense-in-depth
- [ ] 8.1.6: Add `test_delete_edge_cleanup` integration test
- [ ] 8.1.7: Add `test_delete_id_recycling` integration test
- [ ] 8.1.8: Add `test_delete_compaction_removes_tombstones` integration test
- [ ] 8.1.9: Document delete lifecycle in API.md

**Effort:** 3-4 days

#### Design Considerations

**Edge Pruning Strategy Options:**

| Strategy | Pros | Cons |
|----------|------|------|
| **Lazy (during search)** | No background overhead | Degrades search latency |
| **Background worker** | Predictable cleanup | Additional complexity |
| **Compaction filter** | Leverages RocksDB | Limited flexibility |

**Recommendation:** Background worker with configurable interval. Edge pruning is expensive and should not impact search latency.

**ID Recycling Safety:**

VecIds can only be recycled after ALL edges referencing them are removed. This requires:
1. Tracking which VecIds are pending cleanup
2. Scanning all Edges CF entries (or maintaining reverse index)
3. Only freeing after zero references confirmed

For simplicity, Phase 8 may defer ID recycling to future work if edge cleanup alone provides sufficient storage recovery.

---

### Task 8.2: Concurrent Access Hardening

**Status:** Not Started

**Goal:** Production-grade concurrent access validation with snapshot isolation and failure injection.

**Files:**
- `libs/db/tests/test_vector_concurrent.rs` - Enhanced stress tests
- `libs/db/src/vector/benchmark/concurrent.rs` - Batch mode support
- `libs/db/tests/test_vector_snapshot_isolation.rs` - New: isolation tests

**Foundation:** [CONCURRENT.md](./CONCURRENT.md) - Tasks 5.9-5.11

#### Current Concurrent Test Coverage

From [CONCURRENT.md §Task 5.9](./CONCURRENT.md#task-59-multi-threaded-stress-tests):

| Test | File | Thread Config | Gap |
|------|------|---------------|-----|
| `test_concurrent_insert_search` | `test_vector_concurrent.rs:154` | 2 inserters, 2 searchers | No snapshot isolation |
| `test_concurrent_batch_insert` | `test_vector_concurrent.rs:300` | 4 batch inserters | - |
| `test_high_thread_count_stress` | `test_vector_concurrent.rs:409` | 8 writers, 8 readers | 10% error budget |
| `test_writer_contention` | `test_vector_concurrent.rs:542` | 8 writers | - |
| `test_multi_embedding_concurrent_access` | `test_vector_concurrent.rs:664` | 3 indices, 6W, 6R | No failure injection |
| `test_cache_isolation_under_load` | `test_vector_concurrent.rs:843` | 2 indices, 2W, 2R | - |
| `test_concurrent_deletes_vs_searches` | `test_vector_channel.rs:510` | Delete + search | Basic tombstone only |

From [CONCURRENT.md §Task 5.11](./CONCURRENT.md#task-511-concurrent-benchmark-baseline):

| Baseline | Writers | Readers | Duration |
|----------|---------|---------|----------|
| `baseline_full_balanced` | 4 | 4 | 30s |
| `baseline_full_read_heavy` | 1 | 8 | 30s |
| `baseline_full_write_heavy` | 8 | 1 | 30s |
| `baseline_full_stress` | 16 | 16 | 60s |

#### Deliverables

- [ ] 8.2.1: Add snapshot isolation validation test (extends `test_concurrent_insert_search`)
- [ ] 8.2.2: Add transaction conflict resolution stress test (extends `test_writer_contention`)
- [ ] 8.2.3: Reduce error budget from 10% to 1% in `test_high_thread_count_stress` ([CONCURRENT.md §Error Rate Expectations](./CONCURRENT.md#error-rate-expectations))
- [ ] 8.2.4: Add `BenchConfig::with_batch_size()` for batch commit mode ([CONCURRENT.md §Benchmark Transaction Modes](./CONCURRENT.md#benchmark-transaction-modes))
- [ ] 8.2.5: Add mixed search strategy test (RaBitQ + exact concurrent) - deferred from [CONCURRENT.md §Coverage Gaps](./CONCURRENT.md#coverage-gaps--additional-tests)
- [ ] 8.2.6: Add failure injection test (kill writer mid-batch, extends `test_multi_embedding_concurrent_access`)
- [ ] 8.2.7: Add long-running soak test (1hr+ continuous operation, uses `ConcurrentBenchmark`)
- [ ] 8.2.8: Document concurrent access guarantees in API.md

**Effort:** 2-3 days

#### Snapshot Isolation Validation

```rust
#[test]
fn test_snapshot_isolation() {
    // 1. Insert 1000 vectors
    // 2. Start search with snapshot
    // 3. Concurrent: delete 500 vectors
    // 4. Search should see all 1000 (snapshot isolation)
    // 5. New search should see only 500
}
```

#### Batch Commit Mode

From [CONCURRENT.md §Benchmark Transaction Modes](./CONCURRENT.md#benchmark-transaction-modes):

> Current benchmarks use per-vector transactions for correctness validation. For pure throughput
> measurement without transaction overhead, consider adding `BenchConfig::with_batch_size()`.

```rust
// Current: per-vector commit (ConcurrentBenchmark default)
for vec in batch {
    processor.insert_vector(...)?;  // Each commits
}

// New: batch commit mode
let config = BenchConfig::balanced()
    .with_batch_size(100);  // Commit every 100 vectors
```

This extends `ConcurrentBenchmark` in `libs/db/src/vector/benchmark/concurrent.rs`.

---

### Task 8.3: Scale Validation (10M - 1B)

**Status:** Not Started

**Goal:** Validate performance and resource usage at production scale.

**Files:**
- `libs/db/src/vector/benchmark/scale.rs` - New: scale benchmark infrastructure
- `libs/db/benches/bench_vector_scale.rs` - New: scale benchmark binary
- `libs/db/src/vector/docs/SCALE.md` - New: scale validation results

**Foundation:**
- [BASELINE.md](./BASELINE.md) - Current benchmark baselines (50K-1M)
- [CONCURRENT.md §Task 5.11](./CONCURRENT.md#task-511-concurrent-benchmark-baseline) - Throughput baseline methodology

#### Current Scale Coverage

| Scale | Status | Notes |
|-------|--------|-------|
| 50K | ✅ Benchmarked | LAION-CLIP 512D |
| 100K | ✅ Benchmarked | LAION-CLIP 512D |
| 500K | ✅ Benchmarked | LAION-CLIP 512D |
| 1M | ✅ Benchmarked | LAION-CLIP 512D |
| 10M | ❌ Not tested | - |
| 100M | ❌ Not tested | - |
| 1B | ❌ Not tested | - |

#### Memory Budget Analysis

Per-vector memory usage (current):

| Component | Size | Notes |
|-----------|------|-------|
| Vector data (f16) | 512D × 2B = 1KB | Or 2KB for f32 |
| Binary code | 64B (512D/8) | RaBitQ |
| HNSW edges | ~128B avg | M=16, ~8 layers avg |
| VecMeta | 16B | Lifecycle + layer |
| IdForward | 24B | ULID + vec_id |
| IdReverse | 24B | vec_id + ULID |
| **Total** | ~1.8KB/vector | For 512D f16 |

Scale projections:

| Scale | Vectors | Storage | Memory (cache) |
|-------|---------|---------|----------------|
| 1M | 1M | ~1.8GB | ~500MB |
| 10M | 10M | ~18GB | ~5GB |
| 100M | 100M | ~180GB | ~50GB |
| 1B | 1B | ~1.8TB | ~500GB |

#### Deliverables

- [ ] 8.3.1: Create scale benchmark infrastructure with progress reporting
- [ ] 8.3.2: Generate or source 10M vector dataset
- [ ] 8.3.3: Benchmark 10M scale (insert, search, memory)
- [ ] 8.3.4: Generate or source 100M vector dataset
- [ ] 8.3.5: Benchmark 100M scale (insert, search, memory)
- [ ] 8.3.6: Validate 1B scale feasibility (may require sampling)
- [ ] 8.3.7: Profile memory usage at each scale
- [ ] 8.3.8: Document scaling characteristics in SCALE.md
- [ ] 8.3.9: Add CI gate for regression detection at 1M scale

**Effort:** 1-2 weeks

#### Scale Benchmark Protocol

```bash
# 10M benchmark (estimated 3-4 hours)
cargo run --release -p motlie-db --bin bench_vector -- \
    scale --vectors 10000000 --dim 512 --output results/scale_10m.json

# 100M benchmark (estimated 1-2 days)
cargo run --release -p motlie-db --bin bench_vector -- \
    scale --vectors 100000000 --dim 512 --output results/scale_100m.json
```

#### Expected Performance Targets

| Scale | Insert (vec/s) | Search QPS | Search P99 | Recall@10 |
|-------|----------------|------------|------------|-----------|
| 10M | >50 | >50 | <100ms | >80% |
| 100M | >30 | >20 | <200ms | >75% |
| 1B | >10 | >5 | <500ms | >70% |

*Note: These are initial targets. Actual performance may vary based on hardware and HNSW parameters.*

---

## Validation Tests

### Task 8.1 Tests

```rust
#[test]
fn test_delete_edge_cleanup() {
    // 1. Insert 1000 vectors with HNSW
    // 2. Delete 500 vectors
    // 3. Run edge cleanup worker
    // 4. Verify: deleted vec_ids not in any neighbor lists
    // 5. Verify: vector data removed for deleted vectors
}

#[test]
fn test_delete_id_recycling() {
    // 1. Insert 100 vectors, note vec_ids
    // 2. Delete all 100 vectors
    // 3. Run cleanup to completion
    // 4. Insert 100 new vectors
    // 5. Verify: some vec_ids are recycled (not all new)
}

#[test]
fn test_delete_compaction_removes_tombstones() {
    // 1. Insert 1000 vectors
    // 2. Delete all 1000 vectors
    // 3. Force RocksDB compaction
    // 4. Verify: storage size significantly reduced
}
```

### Task 8.2 Tests

```rust
#[test]
fn test_snapshot_isolation_search() {
    // Verify search sees consistent snapshot despite concurrent deletes
}

#[test]
fn test_transaction_conflict_resolution() {
    // Two transactions updating same vector - verify one wins cleanly
}

#[test]
fn test_mixed_search_strategies_concurrent() {
    // Concurrent exact + RaBitQ searches on same embedding
}

#[test]
#[ignore] // Long-running
fn test_soak_1hr_continuous_operation() {
    // 1hr of continuous insert/search/delete with metrics
}
```

### Task 8.3 Tests

```rust
#[test]
#[ignore] // Requires large dataset
fn test_scale_10m_insert_search() {
    // Insert 10M vectors, measure throughput and memory
    // Search with recall validation
}

#[test]
#[ignore] // Requires very large dataset
fn test_scale_100m_insert_search() {
    // Insert 100M vectors, measure throughput and memory
}
```

---

## Dependencies

- **Phase 7:** Async Graph Updater (Complete) - provides pending queue infrastructure
- **Phase 5-6:** Mutation/Query API (Complete) - provides transactional foundation
- **RocksDB:** Compaction filters, snapshot isolation

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Edge cleanup too expensive | Degrades write performance | Background worker with rate limiting |
| Large dataset generation slow | Delays validation | Use synthetic data or sampling |
| Memory exceeds available RAM at scale | OOM crashes | Validate with memory profiling first |
| ID recycling introduces bugs | Data corruption | Defer to future phase if complex |

---

## Success Criteria

1. **Delete Cleanup:** Edge pruning removes >95% of references to deleted vectors within 1 hour
2. **Concurrent Access:** Stress tests pass with <1% error rate (down from 10%)
3. **Scale Validation:** 10M benchmark completes with documented performance
4. **Memory Budget:** Memory usage within 2x of projected estimates at each scale
5. **No Regressions:** 1M baseline performance maintained (within 10%)

---

## Estimated Effort

| Task | Subtasks | Effort |
|------|----------|--------|
| 8.1: Delete Refinement | 8.1.1 - 8.1.9 | 3-4 days |
| 8.2: Concurrent Access Hardening | 8.2.1 - 8.2.8 | 2-3 days |
| 8.3: Scale Validation | 8.3.1 - 8.3.9 | 1-2 weeks |
| **Total** | **26 subtasks** | **~2-3 weeks** |

---

## References

- [CONCURRENT.md](./CONCURRENT.md) - Concurrent operations (Tasks 5.9-5.11)
- [PHASE7.md](./PHASE7.md) - Async Graph Updater design
- [BASELINE.md](./BASELINE.md) - Current benchmark baselines
- [API.md](./API.md) - Public API reference
