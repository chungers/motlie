# Phase 8: Production Hardening

**Status:** In Progress (8.1, 8.2 Complete; 8.3 Pending)
**Date:** January 18, 2026 (Updated: January 19, 2026)
**Prerequisite:** Phase 7 (Async Graph Updater) - Complete

---

## Overview

Phase 8 focuses on production readiness: ensuring deletes are clean, concurrent access is safe, and the system scales to billion-vector workloads. This phase addresses technical debt from earlier phases and validates the system under realistic production conditions.
COMMENT (CODEX, 2026-01-18): "Billion-vector workloads" implies multi-TB storage and multi-100GB cache; confirm target hardware class to keep scale tasks realistic (single-node vs distributed).
RESPONSE (2026-01-18): Target is **single-node** deployment. 1B scale is aspirational validation only (may use sampling). Primary targets are 10M-100M on commodity hardware (64GB RAM, NVMe SSD). Updated Task 8.3 with hardware profile section.

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
COMMENT (CODEX, 2026-01-18): VecMeta checks already exist in pending scan; for HNSW traversal, clarify where to skip deleted nodes to avoid duplicate logic or inconsistent filtering.
RESPONSE (2026-01-18): VecMeta check for HNSW should be in **reranking phase** (after candidates collected, before returning results). Pending scan already has check at `processor.rs:1090`. HNSW traversal itself doesn't need check (edges may point to deleted nodes, but final results are filtered). Task 8.1.5 clarified to target reranking.

### From Scale Analysis

| Gap | Current State | Phase 8 Action |
|-----|--------------|----------------|
| 10M+ benchmarks | Only 1M benchmarks exist | Add 10M, 100M, 1B benchmark configs |
| Memory profiling | No systematic memory tracking | Add memory budget validation |
| Performance regression | Manual benchmark comparison | Add automated regression detection |

---

## Task Breakdown

### Task 8.1: Delete Refinement

**Status:** Complete (2026-01-18)

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

- [x] 8.1.1: ~~Add `DeletedVectors` tracking CF or bitmap for pending cleanup~~ **SKIPPED**
  - STATUS (Claude, 2026-01-18): Skipped after design review.
  - DISCUSSION: Considered adding new `DeletedVectors` CF to track vectors pending GC cleanup.
  - DECISION: Use existing `VecMeta` CF instead. VecMeta already tracks lifecycle state via `is_deleted()`.
    A background worker can scan VecMeta for deleted entries rather than maintaining redundant tracking.
  - RATIONALE: Simpler design (no schema change, no migration), VecMeta is source of truth for lifecycle.
  - TRADE-OFF: O(n) scan on VecMeta vs O(1) lookup in dedicated CF. Acceptable for background GC.
- [x] 8.1.2: Implement edge pruning in background worker (uses VecMeta scan) **COMPLETE**
  - STATUS (Claude, 2026-01-19): Implemented and tested.
  - FILES: `libs/db/src/vector/gc.rs` (new), `libs/db/src/vector/mod.rs` (exports)
  - IMPLEMENTATION:
    - `GarbageCollector` struct with background worker thread (follows AsyncGraphUpdater pattern)
    - `GcConfig` with configurable interval, batch_size, edge_scan_limit, id_recycling flag
    - `GcMetrics` for tracking vectors_cleaned, edges_pruned, cycles_completed
    - `find_deleted_vectors()`: Scans VecMeta CF for `is_deleted() == true`
    - `cleanup_vector()`: Prunes edges, deletes vector data, binary codes, VecMeta
    - `prune_edges()`: Uses merge operator with `EdgeOp::Remove` for lock-free edge removal
  - TESTS: 8 unit tests pass
    - `test_gc_config_defaults`, `test_gc_config_builder`
    - `test_find_deleted_vectors_empty`, `test_gc_finds_deleted_vectors`
    - `test_gc_cleanup_removes_vector_data`, `test_gc_edge_pruning`
    - `test_gc_startup_and_shutdown`, `test_gc_metrics`
  - RESULTS: Edge pruning verified - deleted vec_ids removed from all neighbor bitmaps
  - NOTE: VecId recycling deferred to 8.1.3 (`enable_id_recycling` config flag prepared)
- [x] 8.1.3: Add safe VecId recycling after edge cleanup confirms no references **COMPLETE**
  - STATUS (Claude, 2026-01-19): Implemented and tested.
  - IMPLEMENTATION:
    - Added `recycle_vec_id()` function in `gc.rs` that updates IdAlloc CF free bitmap directly
    - `GcConfig::enable_id_recycling` flag controls feature (default: false for safety)
    - `GcMetrics::ids_recycled` counter tracks recycled IDs
    - Direct RocksDB update bypasses in-memory allocator (acceptable for background GC)
  - TESTS: 2 new tests added
    - `test_gc_id_recycling` - verifies vec_id appears in free bitmap after GC with recycling enabled
    - `test_gc_id_recycling_disabled_by_default` - verifies no recycling when disabled
  - SAFETY: Recycling only occurs after `prune_edges()` confirms no references
  - NOTE: Conservative default (disabled) because in-memory allocator state may be stale
  - ADDRESSED (Claude, 2026-01-19): Added `fully_scanned` flag to `prune_edges()` return value;
    ID recycling now skipped when edge_scan_limit is hit (see 8.1.11)
- [x] 8.1.4: ~~Implement RocksDB compaction filter for tombstone cleanup~~ **DEFERRED**
  - STATUS (Claude, 2026-01-19): Deferred after design analysis.
  - DISCUSSION: Considered adding RocksDB compaction filter for additional tombstone cleanup.
  - DECISION: Rely on GC worker (8.1.2) instead of compaction filter.
  - RATIONALE:
    1. Compaction filters run per-key and cannot safely do cross-CF lookups
       (e.g., can't check if VecMeta exists when filtering Vectors CF)
    2. GC worker already handles complete cleanup transactionally:
       - Prunes edges, deletes vector data, binary codes, VecMeta
       - Runs on configurable interval with batch processing
    3. RocksDB's native tombstone handling is sufficient for key-level deletes
    4. Additional compaction filter complexity not justified by marginal benefit
  - ALTERNATIVE: If orphaned data is observed in production, implement
    a background validation pass that scans for orphaned entries (no compaction filter needed)
- [x] 8.1.5: Add VecMeta lifecycle check in HNSW search as defense-in-depth **COMPLETE**
  - STATUS (Claude, 2026-01-19): Implemented and tested.
  - LOCATION: `libs/db/src/vector/processor.rs:search_with_config()` step 7
  - IMPLEMENTATION:
    - Added batch VecMeta fetch alongside existing IdReverse batch fetch
    - Check `meta.0.is_deleted()` before creating SearchResult
    - Defense-in-depth: catches edge cases where IdReverse exists but VecMeta indicates deleted
    - Uses `multi_get_cf` for efficient batched reads (no extra round trips)
  - EXISTING CHECK: Pending scan already had VecMeta check at lines 1085-1093
  - TESTS: All 65 search tests pass, including `test_search_filters_deleted`
  - NOTE: Primary filter remains IdReverse removal; VecMeta check is secondary defense
- [x] 8.1.6: Add `test_delete_edge_cleanup` integration test **COMPLETE**
  - STATUS (Claude, 2026-01-19): Implemented.
  - FILE: `libs/db/tests/test_vector_delete.rs`
  - VERIFIES: GC removes deleted vec_ids from all neighbor edge bitmaps
- [x] 8.1.7: Add `test_delete_id_recycling` integration test **COMPLETE**
  - STATUS (Claude, 2026-01-19): Implemented.
  - FILE: `libs/db/tests/test_vector_delete.rs`
  - VERIFIES: GC with `enable_id_recycling=true` adds freed IDs to free bitmap
- [x] 8.1.8: Add `test_delete_storage_reclamation` integration test **COMPLETE**
  - STATUS (Claude, 2026-01-19): Implemented (renamed from compaction test).
  - FILE: `libs/db/tests/test_vector_delete.rs`
  - VERIFIES: GC deletes vector data, binary codes, VecMeta entries
- [x] 8.1.9: Document delete lifecycle in API.md **COMPLETE**
  - STATUS (Claude, 2026-01-19): Implemented.
  - FILE: `libs/db/src/vector/docs/API.md` (Part 4)
  - INCLUDES: Delete state machine diagram, GC configuration, search-time safety, best practices
- [x] 8.1.11: Skip VecId recycling when edge scan limit is hit **COMPLETE**
  - STATUS (Claude, 2026-01-19): Implemented.
  - FILE: `libs/db/src/vector/gc.rs`
  - CHANGES: Added `PruneResult.fully_scanned` flag; `cleanup_vector()` now defers ALL
    cleanup (data, VecMeta, ID recycling) when `fully_scanned=false`; vector stays in
    Deleted state for retry in next GC cycle; `GcMetrics.recycling_skipped_incomplete`
    tracks deferred cleanups
  - ADDRESSED (Claude, 2026-01-19): GC now returns early when edge scan incomplete,
    committing only partial edge prunes. Full cleanup deferred to prevent dangling edges.
- [x] 8.1.10: Add `test_async_updater_delete_race` integration test **COMPLETE**
  - STATUS (Claude, 2026-01-19): Implemented.
  - FILE: `libs/db/tests/test_vector_delete.rs`
  - VERIFIES: Delete during async indexing handled correctly (PendingDeleted state)

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

**Status:** Complete (2026-01-19)

**Goal:** Production-grade concurrent access validation with snapshot isolation and failure injection.

**Files:**
- `libs/db/tests/test_vector_concurrent.rs` - Enhanced stress tests
- `libs/db/src/vector/benchmark/concurrent.rs` - Batch mode support + backpressure benchmark
- `libs/db/tests/test_vector_snapshot_isolation.rs` - New: isolation tests
- `libs/db/src/vector/docs/API.md` - Concurrent access guarantees documentation

**Foundation:** [CONCURRENT.md](./CONCURRENT.md) - Tasks 5.9-5.11

#### Current Concurrent Test Coverage

From [CONCURRENT.md §Task 5.9](./CONCURRENT.md#task-59-multi-threaded-stress-tests):

| Test | File | Thread Config | Gap |
|------|------|---------------|-----|
| `test_concurrent_insert_search` | `test_vector_concurrent.rs:154` | 2 inserters, 2 searchers | No snapshot isolation |
| `test_concurrent_batch_insert` | `test_vector_concurrent.rs:300` | 4 batch inserters | - |
| `test_high_thread_count_stress` | `test_vector_concurrent.rs:409` | 8 writers, 8 readers | ~~10% error budget~~ 1% error budget |
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

- [x] 8.2.1: Add snapshot isolation validation test → `test_vector_snapshot_isolation.rs` (5 tests)
  - Added SearchReader-backed insert snapshot test for end-to-end validation.
- [x] 8.2.2: Add transaction conflict resolution stress test → `test_transaction_conflict_stress` (pessimistic locking validated)
- [x] 8.2.3: Reduce error budget from 10% to 1% in `test_high_thread_count_stress` (pre-populates 100 vectors)
- [x] 8.2.4: Add `BenchConfig::with_batch_size()` for batch commit mode
  ADDRESSED (CODEX, 2026-01-19): `batch_size` now drives `InsertVectorBatch`
  sends in `benchmark/concurrent.rs` via `insert_producer_workload()`.
- [x] 8.2.5: Add mixed search strategy test → `test_mixed_search_strategy_concurrent`
- [x] 8.2.6: Add failure injection test → `test_failure_injection_writer_crash`
- [x] 8.2.7: Add long-running soak test → `test_soak_one_hour` (ignored by default, 0.1% error budget)
- [x] 8.2.8: Document concurrent access guarantees in API.md § Part 5
- [x] 8.2.9: Add backpressure throughput impact benchmark → `measure_backpressure_impact()`, `test_backpressure_throughput_impact`

**Key Findings:**
- RocksDB TransactionDB uses **pessimistic locking** (not optimistic): concurrent writers block on lock acquisition
- Snapshot isolation provides repeatable reads; concurrent deletes don't affect existing snapshots
- Pre-populating index before stress test eliminates "empty graph" errors, enabling tighter error budgets
- Failure injection test validates system remains operational after simulated writer crashes

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

**Status:** In Progress (8.3.1, 8.3.10 Complete; 10K/100K/1M benchmarks running)

**Goal:** Validate performance and resource usage at production scale.

**Files:**
- `libs/db/src/vector/benchmark/scale.rs` - Scale benchmark infrastructure
- `bins/bench_vector/src/commands.rs` - `bench_vector scale` command
- `libs/db/src/vector/docs/BASELINE.md` - Scale benchmark results

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

- [x] 8.3.1: Create scale benchmark infrastructure with progress reporting
  - `benchmark/scale.rs`: `ScaleConfig`, `ScaleBenchmark`, `ScaleProgress`, `ScaleResult`
  - Real-time progress with throughput, ETA, memory tracking
  - Integrated into `bench_vector scale` CLI command (not separate test file)
  - Distance wiring added (`--distance cosine|l2|dot`), with Cosine normalization only
- [x] 8.3.2: Run baseline benchmarks (10K, 100K, 1M)
  - 10K: 224.3 vec/s insert, 623.7 QPS, 53 MB RSS
  - 100K: 92.0 vec/s insert, 392.3 QPS, 345 MB RSS
  - 1M: In progress (ETA ~2 hours)
- [ ] 8.3.3: Benchmark 10M scale (insert, search, memory)
- [ ] 8.3.4: Generate or source 100M vector dataset
- [ ] 8.3.5: Benchmark 100M scale (insert, search, memory)
- [ ] 8.3.6: Validate 1B scale feasibility (may require sampling)
- [x] 8.3.7: Profile memory usage at each scale
  - RSS tracking integrated into scale benchmark (`get_rss_bytes()`)
  - Nav cache size reported in results
- [x] 8.3.8: Document scaling characteristics in BASELINE.md
  - Added "Scale Benchmarks (Phase 8.3)" section
- [ ] 8.3.9: Add CI gate for regression detection at 1M scale
- [x] 8.3.10: Add reproducible synthetic dataset generator (seeded) for large-scale runs
  - `StreamingVectorGenerator`: Memory-efficient streaming generation
  - Deterministic via ChaCha8Rng seed for reproducibility
  - Unit-normalized vectors for Cosine distance

**Effort:** 1-2 weeks

#### Scale Benchmark Protocol

```bash
# Build release binary
cargo build --release --bin bench_vector

# Quick validation (10K vectors, ~1 min)
./target/release/bench_vector scale \
    --num-vectors 10000 --dim 128 --db-path /tmp/bench_10k

# Medium scale (100K vectors, ~18 min)
./target/release/bench_vector scale \
    --num-vectors 100000 --dim 128 --db-path /tmp/bench_100k \
    --output results/scale_100k.json

# Large scale (1M vectors, ~2 hours)
./target/release/bench_vector scale \
    --num-vectors 1000000 --dim 128 --batch-size 5000 \
    --db-path /tmp/bench_1m --output results/scale_1m.json

# 10M benchmark (estimated 20+ hours)
./target/release/bench_vector scale \
    --num-vectors 10000000 --dim 128 --batch-size 5000 \
    --db-path /tmp/bench_10m --output results/scale_10m.json
```

#### Expected Performance Targets

| Scale | Insert (vec/s) | Search QPS | Search P99 | Recall@10 |
|-------|----------------|------------|------------|-----------|
| 10M | >50 | >50 | <100ms | >80% |
| 100M | >30 | >20 | <200ms | >75% |
| 1B | >10 | >5 | <500ms | >70% |

*Note: These are initial targets. Actual performance may vary based on hardware and HNSW parameters.*
COMMENT (CODEX, 2026-01-18): Consider adding a "hardware profile" subsection (CPU/RAM/SSD) so target numbers remain interpretable and comparable.
RESPONSE (2026-01-18): Added Hardware Profile section below.

#### Hardware Profile (Reference)

Target hardware for benchmark reproducibility:

| Tier | CPU | RAM | Storage | Target Scale |
|------|-----|-----|---------|--------------|
| **Development** | 4+ cores | 16GB | SSD | 1M |
| **Production (Small)** | 8+ cores | 64GB | NVMe | 10M |
| **Production (Medium)** | 16+ cores | 128GB | NVMe | 100M |
| **Production (Large)** | 32+ cores | 256GB+ | NVMe RAID | 1B (sampling) |

All benchmarks should record: CPU model, core count, RAM size, storage type, OS version.

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
| 8.1: Delete Refinement | 8.1.1 - 8.1.10 | 3-4 days |
| 8.2: Concurrent Access Hardening | 8.2.1 - 8.2.9 | 2-3 days |
| 8.3: Scale Validation | 8.3.1 - 8.3.10 | 1-2 weeks |
| **Total** | **29 subtasks** | **~2-3 weeks** |

---

## References

- [CONCURRENT.md](./CONCURRENT.md) - Concurrent operations (Tasks 5.9-5.11)
- [PHASE7.md](./PHASE7.md) - Async Graph Updater design
- [BASELINE.md](./BASELINE.md) - Current benchmark baselines
- [API.md](./API.md) - Public API reference
