# Phase 7: Async Graph Updater

**Status:** In Progress (Tasks 7.1-7.4 Core Complete)
**Date:** January 17, 2026
**Prerequisite:** Phase 6 (MPSC/MPMC Public API) - Complete

---

## Overview

Phase 7 enables online updates by decoupling vector storage from HNSW graph construction. This two-phase insert pattern reduces insert latency from ~50ms P99 to <5ms while maintaining search quality.

### Problem Statement

Synchronous HNSW graph updates during insert create latency spikes:
- Building HNSW edges requires greedy search through existing graph
- P99 latency ~50ms is unacceptable for real-time applications
- Graph construction blocks the mutation consumer

### Solution: Two-Phase Insert

```
Phase 1 (Synchronous, <5ms):
  insert() → Store vector + metadata + binary code → Add to Pending Queue
            ↓
  Vector immediately searchable (brute-force fallback for pending items)

Phase 2 (Asynchronous, background):
  Worker threads drain Pending Queue → Greedy search → Build HNSW edges
```

---

## Vector Lifecycle States

Vectors transition through well-defined lifecycle states, tracked in the `VecMetadata.flags` field using a type-safe enum + bitmask hybrid design.

### VecLifecycle Enum (bits 0-1)

```rust
pub(crate) enum VecLifecycle {
    Indexed = 0b00,        // In HNSW graph, fully searchable
    Deleted = 0b01,        // Soft-deleted, skip in search
    Pending = 0b10,        // Awaiting async graph construction
    PendingDeleted = 0b11, // Deleted before graph construction completed
}
```

### State Machine

```
Insert(build_index=true)  ───────────────────────► Indexed
Insert(build_index=false) ──► Pending ──► Indexed
                                 │            │
                            set_deleted   set_deleted
                                 ▼            ▼
                           PendingDeleted  Deleted
                                 │
                           clear_pending
                                 ▼
                              Deleted
```

### Flags Byte Layout

```
flags byte: 0bRRRRRRLL
            ││││││└┴── Lifecycle (2 bits): VecLifecycle enum (mutually exclusive)
            │││││└──── Reserved (bit 2)
            ││││└───── Reserved (bit 3)
            │││└────── Reserved (bit 4)
            ││└─────── Reserved (bit 5)
            │└──────── Reserved (bit 6)
            └───────── REPLICATED (bit 7) - FUTURE: not yet implemented
```

### Design Rationale

1. **Type Safety**: `VecLifecycle` enum provides compile-time enforcement of valid states
2. **State Machine**: `set_deleted()` and `clear_pending()` handle transitions correctly
3. **Extensibility**: Upper 6 bits reserved for future orthogonal flags (e.g., REPLICATED for distributed deployment)
4. **Wire Compatibility**: Enum values match original FLAG_DELETED=0x01, FLAG_PENDING=0x02 constants

### Usage

```rust
// Create pending vector
let meta = VecMetadata::pending();
assert!(meta.is_pending());
assert_eq!(meta.lifecycle(), VecLifecycle::Pending);

// After async graph construction
meta.clear_pending();  // Pending → Indexed

// Delete handling
meta.set_deleted();    // Indexed → Deleted (or Pending → PendingDeleted)

// Check state
if meta.is_deleted() { /* skip in search */ }
if meta.is_pending() { /* brute-force fallback or async processing needed */ }
```

**File:** `libs/db/src/vector/schema.rs`

---

## Task Breakdown

### Task 7.1: Pending Queue Column Family

**Status:** ✅ Complete (existing implementation verified)

**Goal:** RocksDB column family to persist pending graph updates.

**File:** `libs/db/src/vector/schema.rs`

**Key Design:**
- Key: `[embedding_code: u64][timestamp_milli: u64][vec_id: u32]` = 20 bytes
- Value: empty (vector data already in Vectors CF)
- Timestamp enables FIFO ordering for fair processing within each embedding
- Survives crashes for recovery
- Prefix extractor on embedding_code (8 bytes) for efficient per-embedding scans

CODEX (2026-01-17): `Pending` CF already exists in `schema.rs` with key `(embedding_code, TimestampMilli, vec_id)`; align the design (timestamp units + naming) and avoid adding a duplicate CF.
RESPONSE: Confirmed. Using existing `Pending` CF with `TimestampMilli`. Added `key_now()` and `prefix_for_embedding()` helpers.

CODEX (2026-01-17): FIFO ordering is only per-embedding with the current key layout; clarify cross-embedding fairness or add a batching strategy that round-robins embeddings.
RESPONSE: Acknowledged. Cross-embedding fairness will be handled in Task 7.3 via round-robin iteration over embedding codes.

**Existing Implementation:**
- `Pending` struct with `CF_NAME = "vector/pending"`
- `PendingCfKey(EmbeddingCode, TimestampMilli, VecId)`
- `key_to_bytes()` / `key_from_bytes()` for serialization
- `key_now()` - create key with current timestamp (added)
- `prefix_for_embedding()` - get 8-byte prefix for iteration (added)
- Registered in `ALL_COLUMN_FAMILIES`
- Prefix extractor configured for embedding-based scans
CODEX (2026-01-17): Verified `schema.rs` changes; `key_now()` uses `SystemTime::now().as_millis()` which matches `TimestampMilli` and the key layout. No duplication found.
CODEX (2026-01-17): `prefix_for_embedding()` + RocksDB prefix extractor are consistent (8-byte embedding prefix). Correctness OK.

**Deliverables:**
- [x] 7.1.1: `Pending` column family struct (existing)
- [x] 7.1.2: `key_to_bytes()` and `key_from_bytes()` methods (existing)
- [x] 7.1.3: Registered in `ALL_COLUMN_FAMILIES` (existing)
- [x] 7.1.4: `test_pending_key_roundtrip` test (existing)
- [x] 7.1.5: Added `key_now()` helper with TimestampMilli
- [x] 7.1.6: Added `prefix_for_embedding()` for iteration
- [x] 7.1.7: Added `test_pending_key_now` and `test_pending_prefix_for_embedding` tests
CODEX (2026-01-17): All 7.1 deliverables implemented as claimed.

---

### Task 7.2: Async Updater Configuration

**Status:** ✅ Complete

**Goal:** Define configuration for the background graph updater.

**File:** `libs/db/src/vector/async_updater.rs`

**Configuration Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | usize | 100 | Max vectors per worker batch |
| `batch_timeout` | Duration | 100ms | Max wait time to fill batch |
| `num_workers` | usize | 2 | Background worker thread count |
| `ef_construction` | usize | 200 | ef parameter for greedy search |
| `process_on_startup` | bool | true | Drain pending queue on startup |
| `idle_sleep` | Duration | 10ms | Sleep when no pending items (added) |

**Deliverables:**
- [x] 7.2.1: Create `AsyncUpdaterConfig` struct with defaults
- [x] 7.2.2: Add builder methods (`with_batch_size`, `with_num_workers`, etc.)
- [x] 7.2.3: Document tuning guidance in doc comments
- [x] 7.2.4: Add `test_config_defaults` and `test_config_builder` tests
CODEX (2026-01-17): Verified config struct, builders, and tests in `async_updater.rs` are implemented as claimed. Task 7.2 is complete and ready for 7.3.

---

### Task 7.3: Async Updater Core Implementation

**Status:** ✅ Complete (process_insert placeholder for Task 7.4)

**Goal:** Implement the background worker that processes pending inserts.

**File:** `libs/db/src/vector/async_updater.rs`

**Architecture:**
```
AsyncGraphUpdater
  ├── storage: Arc<Storage>
  ├── config: AsyncUpdaterConfig
  ├── shutdown: Arc<AtomicBool>
  ├── embedding_counter: Arc<AtomicU64>  (for round-robin)
  ├── items_processed: Arc<AtomicU64>    (metrics)
  ├── batches_processed: Arc<AtomicU64>  (metrics)
  └── workers: Vec<JoinHandle<()>>
      └── worker_loop()
          ├── collect_batch() → Vec<(key, EmbeddingCode, VecId)>
          └── process_insert() → builds edges + removes pending (atomic)
```

**Deliverables:**
- [x] 7.3.1: Implement `AsyncGraphUpdater::start()` - spawn workers
- [x] 7.3.2: Implement `worker_loop()` - batch collection and processing
- [x] 7.3.3: Implement `collect_batch()` - read from pending CF with limit
- [x] 7.3.4: Implement `process_insert()` - completed under Task 7.4.3
- [x] 7.3.5: Remove pending entry inside `process_insert()` transaction
- [x] 7.3.6: Implement `drain_pending_static()` - startup recovery
- [x] 7.3.7: Implement `shutdown()` - graceful worker termination
- [x] 7.3.8: Add metrics: `items_processed()`, `batches_processed()`

**Key Implementation Notes:**
- Batch collection uses snapshot-based iterator for stability
- Round-robin fairness implemented: workers round-robin across embeddings to prevent starvation
- `discover_active_embeddings()` uses seek-based O(E) discovery (not O(N) scan)
- Failed inserts logged but not cleared from pending (retry on next batch)
- Shutdown waits for in-flight batches to complete
- Delete operations are idempotent (safe for concurrent workers / crash recovery)
CODEX (2026-01-17): Verified round-robin fairness implementation and snapshot-based iteration. Task 7.3 infrastructure is now complete; ready for Task 7.4.

CODEX (2026-01-17): `collect_batch()` iterates the entire Pending CF and ignores `embedding_counter`; round-robin is not implemented. Update the note or implement fairness before certifying 7.3.
RESPONSE: Fixed. Implemented seek-based `discover_active_embeddings()` that finds unique embedding codes in O(E), then round-robin selects one via `embedding_counter`, then prefix-scans only that embedding's items.

CODEX (2026-01-18): Verified pending deletion now happens inside `process_insert()` transaction; no separate `clear_processed()` remains.

CODEX (2026-01-17): `collect_batch()` uses a live iterator without a snapshot; if items are deleted concurrently, ensure iterator stability or tolerate missing keys (idempotent delete).
RESPONSE: Fixed. Now uses snapshot-based iterator (`iterator_cf_opt` with `ReadOptions::set_snapshot`). Delete is idempotent in RocksDB.

CODEX (2026-01-17): Prefix scan per-embedding can starve other embeddings; add a round-robin or global iterator over embedding codes for fairness.
RESPONSE: Implemented. `discover_active_embeddings()` discovers embeddings with pending items using efficient seeks, then `embedding_counter` selects via modulo for round-robin fairness.

CODEX (2026-01-17): Collect keys outside of write transactions and keep batch operations idempotent (skip if FLAG_PENDING cleared or FLAG_DELETED set) to tolerate crashes and retries.
RESPONSE: Keys collected via read-only iterator. Idempotency check will be in process_insert (Task 7.4).

---

### Task 7.4: Insert Path Integration

**Status:** ✅ Complete

**Goal:** Modify insert path to use two-phase pattern.

**Files:**
- `libs/db/src/vector/schema.rs` - VecLifecycle enum, VecMetadata
- `libs/db/src/vector/ops/insert.rs` - Two-phase insert logic
- `libs/db/src/vector/async_updater.rs` - process_insert() implementation

**Changes:**
1. `InsertVector` has `immediate_index: bool` flag (already existed)
2. When `immediate_index = false` (i.e., `build_index = false`):
   - Store vector data synchronously (Vectors CF)
   - Store VecMeta with `VecLifecycle::Pending` state
   - Add to Pending queue
   - Skip HNSW edge construction
3. `process_insert()` builds HNSW edges asynchronously:
   - Checks `meta.is_pending()` (idempotent via VecLifecycle enum)
   - Loads vector from Vectors CF
   - Builds HNSW graph via hnsw::insert
   - Transitions to `VecLifecycle::Indexed` (via hnsw::insert writing VecMeta)
4. Search handles pending vectors via bounded brute-force fallback:
   - `SearchConfig.pending_scan_limit` controls max pending vectors to scan (default: 1000)
   - `Processor::scan_pending_vectors()` iterates Pending CF and computes exact distances
   - Results merged with HNSW search results, deduplicated, sorted, and truncated to k
   - `.no_pending_fallback()` or `.with_pending_scan_limit(0)` disables the fallback
CODEX (2026-01-17): Brute-force fallback must be bounded (e.g., cap pending scan size or time) to avoid O(N) degradation when backlog grows; add an explicit limit and metrics.
RESPONSE (2026-01-18): Implemented. `SearchConfig.pending_scan_limit` defaults to 1000. Scan is bounded by this limit.

**Deliverables:**
- [x] 7.4.1: Implement VecLifecycle enum with type-safe state transitions
- [x] 7.4.2: Modify `ops::insert::vector()` and `ops::insert::batch()` for two-phase
- [x] 7.4.3: Implement `process_insert()` in async_updater.rs
- [x] 7.4.4: Update search to handle pending vectors (bounded brute-force fallback)

**Implementation Notes:**
- `AsyncGraphUpdater::start()` now requires `registry` and `nav_cache` parameters
- `process_insert()` creates an hnsw::Index per embedding using stored EmbeddingSpec
- hnsw::insert() writes VecMeta with `VecLifecycle::Indexed`, transitioning from Pending
- Pending queue cleared atomically within the same transaction (no separate `clear_processed()`)
- VecLifecycle enum provides type-safe lifecycle states (see "Vector Lifecycle States" section above)
CODEX (2026-01-17): Verified async insert path in `ops::insert` writes VecMeta with FLAG_PENDING and enqueues Pending. `process_insert()` is transactional and clears FLAG_PENDING via `hnsw::insert()`. Pending deletion is still outside the transaction; it is idempotent but leaves a retry window on crash.
RESPONSE (2026-01-18): Fixed. Pending deletion is now inside the `process_insert()` transaction - all operations (edge build, FLAG_PENDING clear, pending delete) commit atomically. No crash retry window.
CODEX (2026-01-17): `process_insert()` rebuilds a new `hnsw::Index` per item and reads EmbeddingSpec from both registry and CF. Consider reusing a per-embedding Index/cache or validating registry/CF consistency to avoid divergence.
RESPONSE (2026-01-18): Clarified. Registry provides runtime info (storage_type, distance). CF provides authoritative build params (hnsw_m). Both sources are consistent because registry is initialized from CF on startup. Index is rebuilt per-item but this is acceptable for async path; optimization deferred to future task if needed.
CODEX (2026-01-18): Verified pending deletion is inside `process_insert()` transaction; no crash window remains.
CODEX (2026-01-18): `scan_pending_vectors()` uses `Vectors::value_from_bytes` (f32) and ignores `storage_type`; this is wrong for f16 embeddings. Use `value_from_bytes_typed` with `embedding.storage_type()` to avoid corrupted distances.
RESPONSE (2026-01-18): Fixed. Now uses `Vectors::value_from_bytes_typed(&vec_bytes, storage_type)?` with `storage_type = embedding.storage_type()`.
CODEX (2026-01-18): Pending scan does not check VecMeta lifecycle; if a pending entry exists for a deleted vector, search may return deleted results. Either validate VecMeta state or ensure pending entries are always removed on delete.
RESPONSE (2026-01-18): Fixed. Added VecMeta lifecycle check in `scan_pending_vectors()` - skips vectors where `meta.is_deleted()` returns true (PendingDeleted state).
CODEX (2026-01-18): Verified both pending-scan fixes in `processor::scan_pending_vectors()` (storage_type-aware decode + lifecycle check).

---

### Task 7.5: Delete Handling

**Goal:** Implement delete with async cleanup pattern.

**Files:**
- `libs/db/src/vector/processor.rs`
- `libs/db/src/vector/schema.rs`

**Delete Flow:**
1. Phase 1 (sync): Call `meta.set_deleted()`, remove from pending queue, free ID
2. Phase 2 (lazy): Stale edges cleaned during search/compaction

**Lifecycle Transitions on Delete:**
```rust
// Delete transitions (handled by VecMetadata::set_deleted())
Indexed       → Deleted         // Normal indexed vector
Pending       → PendingDeleted  // Deleted before graph construction
Deleted       → Deleted         // Already deleted (idempotent)
PendingDeleted → PendingDeleted // Already deleted (idempotent)
```

**Note:** VecLifecycle enum (Task 7.4) already provides:
- `VecLifecycle::Deleted` and `VecLifecycle::PendingDeleted` states
- `meta.set_deleted()` for correct state transitions
- `meta.is_deleted()` to check if vector should be skipped

**Deliverables:**
- [x] 7.5.1: VecLifecycle enum with Deleted/PendingDeleted states (completed in 7.4)
- [x] 7.5.2: Implement `mark_deleted()` in Processor (via `ops::delete::mark_deleted`)
- [x] 7.5.3: Implement `remove_from_pending()` helper (in `ops::delete`)
- [x] 7.5.4: Update search to skip deleted nodes (via IdReverse removal)
- [x] 7.5.5: Add ID recycling to allocator (only when HNSW disabled)
- [x] 7.5.6: Guard ID reuse until stale edges cleaned (implemented by disallowing reuse when HNSW enabled)
CODEX (2026-01-18): Verified delete path in `ops::delete` + `processor::delete_vector` updates VecMeta lifecycle, removes pending entries, and avoids vec_id reuse when HNSW is enabled. This satisfies 7.5.2-7.5.6.

---

### Task 7.6: Testing & Crash Recovery

**Status:** ✅ Complete

**Goal:** Comprehensive testing including crash recovery scenarios.

#### Existing Infrastructure

The synchronous path already has crash recovery tests in:
- **File:** `libs/db/src/vector/crash_recovery_tests.rs`

| Existing Test | Coverage |
|---------------|----------|
| `test_uncommitted_transaction_not_visible` | Atomicity: uncommitted txns don't persist |
| `test_committed_transaction_persists` | Durability: committed txns survive restart |
| `test_id_allocator_recovery` | IdAllocator state persists across restart |
| `test_id_allocator_transactional_recovery` | Transactional ID allocation recovery |
| `test_hnsw_navigation_cold_cache_recovery` | Search works with cold NavigationCache |
| `test_hnsw_entry_point_persists` | HNSW entry point survives restart |
| `test_full_crash_recovery_scenario` | End-to-end: insert → crash → search works |
| `test_transaction_insert_with_recovery` | Insert via txn API → recovery → search |

These tests validate the **synchronous path** (Phase 5/6). Phase 7 requires **additional tests** for the async pending queue.

#### New Tests for Phase 7

**File:** `libs/db/src/vector/async_updater_tests.rs` (new, in-module tests)

| New Test | Description | Builds On |
|----------|-------------|-----------|
| `test_insert_async_immediate_searchability` | Vectors searchable before graph built | - |
| `test_pending_queue_drains` | Workers process all pending items | - |
| `test_pending_queue_crash_recovery` | Pending items survive restart | `test_full_crash_recovery_scenario` |
| `test_delete_removes_from_pending` | Delete clears pending entry | - |
| `test_concurrent_insert_and_worker` | No races between insert and worker | - |
| `test_shutdown_completes_in_flight` | Graceful shutdown finishes batch | - |
| `test_partial_batch_idempotent` | Re-processing partial batch is safe | `test_committed_transaction_persists` |

**Relationship to Existing Tests:**
- Existing tests verify RocksDB transactions work correctly (atomicity, durability)
- Phase 7 tests build on this foundation: pending queue uses same transaction API
- `test_pending_queue_crash_recovery` mirrors `test_full_crash_recovery_scenario` but for two-phase inserts

**Deliverables:**
- [x] 7.6.1: Implement immediate searchability test
- [x] 7.6.2: Implement pending queue drain test
- [x] 7.6.3: Implement pending queue crash recovery test
- [x] 7.6.4: Implement delete + pending interaction test
- [x] 7.6.5: Implement concurrent insert/worker test
- [x] 7.6.6: Implement graceful shutdown test
- [x] 7.6.7: Implement partial batch idempotency test
- [x] 7.6.8: Add backlog bound test (pending scan limit respected under large queue)

**Implementation Notes:**
- All tests implemented in `libs/db/src/vector/async_updater.rs` in the `tests` module
- Tests use `TempDir` for isolated test storage
- Helper functions: `setup_test_env()`, `register_embedding()`, `test_vector()`, `count_pending_items()`
- Crash recovery test uses two-phase approach: insert → drop storage → reopen → verify

---

### Task 7.7: Integration & Benchmarks

**Status:** ✅ Complete

**Goal:** Integrate async updater with existing infrastructure and benchmark.

**Deliverables:**
- [x] 7.7.1: Add `AsyncGraphUpdater` to `Storage` initialization
- [x] 7.7.2: Update `WriterConfig` with async updater options
- [x] 7.7.3: Add benchmark comparing sync vs async insert latency
- [x] 7.7.4: Document latency characteristics in BASELINE.md
- [x] 7.7.5: Update ROADMAP.md to mark Phase 7 complete

**Implementation Notes:**
- `Subsystem.start_with_async()` integrates AsyncGraphUpdater with managed lifecycle
- `Processor.new_with_nav_cache()` enables shared NavigationCache between Processor and AsyncGraphUpdater
- `compare_sync_async_latency()` benchmark function in `benchmark/concurrent.rs`
- `SyncAsyncLatencyResult` provides P50/P99 latency comparison and speedup factors
- Async insert latency section added to BASELINE.md

### Task 7.8: Backpressure, Metrics, and Observability

**Goal:** Prevent unbounded queue growth and expose health signals.

**Deliverables:**
- [ ] 7.8.1: Add pending queue size metric (gauge) and worker throughput counters
- [ ] 7.8.2: Enforce backpressure when pending queue exceeds threshold (configurable)
- [ ] 7.8.3: Surface backlog depth and drain rate in logs/metrics

---

## Consistency Guarantees

| Scenario | Guarantee |
|----------|-----------|
| Crash before Phase 2 | Vector in pending queue, recovered on restart |
| Crash during Phase 2 | Partial edges may exist; re-processing is idempotent |
| Search during Phase 2 | Vector found via brute-force fallback |
| Delete during Phase 2 | Removed from pending queue, edges cleaned lazily |

---

## Performance Targets

| Metric | Current (Sync) | Target (Async) |
|--------|----------------|----------------|
| Insert P50 | ~5ms | <1ms |
| Insert P99 | ~50ms | <5ms |
| Search (pending) | N/A | <10ms (brute-force) |
| Search (indexed) | ~2ms | ~2ms (unchanged) |
| Graph build rate | N/A | >1000 vec/s/worker |

---

## Dependencies

- **Phase 5:** Internal Mutation/Query API (Complete)
- **Phase 6:** MPSC/MPMC Public API (Complete)
- **RocksDB:** Column family, transactions, prefix iteration

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Pending queue grows unboundedly | Memory/disk | Add backpressure when queue > threshold |
| Brute-force fallback slow at scale | Latency | Limit pending scan to recent items |
| Worker starvation | Graph quality degrades | Monitor pending queue depth, alert |
| Race between delete and worker | Data inconsistency | Use transaction isolation |

---

## Estimated Effort

| Task | Subtasks | Effort |
|------|----------|--------|
| 7.1: Pending Queue CF | 7.1.1 - 7.1.4 | 0.5 day |
| 7.2: Configuration | 7.2.1 - 7.2.3 | 0.25 day |
| 7.3: Core Implementation | 7.3.1 - 7.3.7 | 1.5 days |
| 7.4: Insert Path Integration | 7.4.1 - 7.4.4 | 0.5 day |
| 7.5: Delete Handling | 7.5.1 - 7.5.5 | 0.5 day |
| 7.6: Testing & Crash Recovery | 7.6.1 - 7.6.7 | 1 day |
| 7.7: Integration & Benchmarks | 7.7.1 - 7.7.5 | 0.75 day |
| **Total** | **30 subtasks** | **~5 days** |

---

## Success Criteria

1. Insert P99 latency < 5ms (currently ~50ms)
2. All pending vectors searchable immediately
3. Crash recovery drains pending queue on restart
4. No data loss in any failure scenario
5. Benchmark demonstrates 10x latency improvement
