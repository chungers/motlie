TESTING PLAN (graph crate)

Goals
- Provide confidence in core graph correctness pre-VERSIONING.
- Provide versioning guarantees (history, restore, time travel, validity).
- Make regressions visible with fast, deterministic, high-signal tests.
- Define performance expectations with targeted benchmarks.

Scope
- Unit tests: encode/decode, key/value schemas, helper functions.
- Integration tests: db txns, CF interactions, indexing, GC.
- Subsystem lifecycle: start/stop, shutdown, background tasks.
- Metrics/info export: counters, gauges, info snapshots.
- Property tests: invariants across randomized ops.
- Benchmarks: hot paths, scan patterns, GC cost.
- Examples: kept minimal; not relied upon for correctness.

Test Infrastructure
- Time control: inject clock or use deterministic timestamps in tests.
- Deterministic ids: stable NodeId/EdgeId generation for assertions.
- Fixtures: helper builders for Node, Edge, Fragment, Summary.
- Storage: use temp dir per test, no shared state.
- Assertions: verify both forward and reverse CFs, summaries, indexes.

Baseline (pre-VERSIONING) Behavioral Tests (missing)
- Node CRUD: create, read, update, delete across txns.
- Edge CRUD: create, read, update, delete across txns.
- Summary writes: node/edge summary CF receives data on create/update.
- Summary index: ensure hash -> entity index entries exist and are unique.
- Reverse edge index: ensure dst -> incoming edges is correct.
- Forward + reverse consistency: single txn writes both sides.
- Fragment append: fragment CF appends without overwrite; idempotency.
- Fragment search: bm25/vector index integration at least smoke-level.
- Delete semantics: delete removes or tombstones and is discoverable.
- Error handling: missing nodes/edges fail gracefully with correct error type.
- Serialization: round-trip for keys/values, including optional fields.
- Lifecycle:
  - Graph start/stop is idempotent.
  - Background tasks exit on shutdown.
  - Subsystems join/flush before close.
- GC lifecycle: start/shutdown, signal_shutdown (owned GC; no shutdown_arc)
  - GC worker thread joins cleanly on shutdown (IMPLEMENTED gc.rs)
- Metrics/info:
  - Core counters increment (writes, reads, failures).
  - Info export returns expected snapshot fields.

VERSIONING Behavioral Tests (missing)
- VersionHistory:
  - Create Node/Edge writes history entry.
  - Update Node/Edge writes history entry with new ValidSince.
  - Delete writes history entry (tombstone or deletion state).
- Restore:
  - Restore by version returns correct snapshot.
  - Restore does not leave multiple CURRENT markers.
  - Restore reuses existing summary hash only if present.
- Time travel:
  - AsOf reads return correct version for Node/Edge.
  - AsOf reads respect ValidSince/ValidUntil.
  - AsOf reads across multiple updates ordered correctly.
- Active period:
  - ValidUntil updates do not mutate in place without history.
  - Range queries return edges active during interval.
- Weight/history:
  - Weight updates create new version (no in-place mutation).
  - Weight/temporal fields propagate to history and indexes.
- Restore/index:
  - Restore marks prior CURRENT summary index entry as STALE.
  - Restore does not leave multiple CURRENT markers.
  - RestoreEdges batch also marks prior CURRENT entries STALE and records orphan candidates.
  - RestoreNode/RestoreEdge fail if referenced summary hash is missing (GC’d).
- RestoreEdges dry_run validates and fails on missing summaries without writing.

Index and Denormalization Tests (missing)
- Reverse edge temporal denormalization:
  - Forward update also updates reverse cf value in same txn.
  - Reverse scan honors ValidSince/ValidUntil without extra read.
- Summary index marker:
  - CURRENT and STALE markers obey single-current invariant.
  - Marker updates happen atomically with summary write.
- Hash prefix scans:
  - Prefix scan returns all entities sharing summary hash.
  - Reverse scan with padded version ordering yields latest first.

GC and Storage Tests (missing)
- Orphan summary tracking:
  - On summary replace, old summary hash is tracked.
- GC retention:
  - GC does not delete summaries that are still referenced by any current entity.
  - Retention window prevents deletion of recent summaries.
- Shared summary safety:
  - OrphanSummaries GC does not delete summary still referenced by other entities sharing the same SummaryHash.
- Fragment GC:
  - No GC of fragments (append-only).
  - Fragment summaries updated without fragment loss.

Concurrency and Atomicity (missing)
- Concurrent updates:
  - Two writers update same node/edge; history is consistent.
  - Conflicts are detected or resolved as designed.
- Multi-step txn:
  - Forward + reverse + summary + index are atomic or rolled back.
- Idempotency:
  - Replaying the same mutation does not corrupt indexes.
 - Lifecycle under load:
   - Shutdown during writes does not corrupt CFs.
   - Metrics still export after heavy load.

Property/Invariant Tests (missing)
- For any sequence of ops:
  - Exactly one CURRENT summary per entity.
  - Every CURRENT summary index references existing summary CF entry.
  - Reverse edge entries correspond to an existing forward edge version.
  - VersionHistory is monotonically ordered by ValidSince.

Performance/Benchmarks (missing)
- Hot scans:
  - Incoming edges scan with temporal filter.
  - Summary hash prefix scans.
  - AsOf reads with binary search / seek.
- Write amplification:
  - Create/update cost (number of CF writes) for Node/Edge.
- GC cost:
  - Orphan summary scan + delete vs refcount-based delete.

Documentation/Examples Validation (missing)
- Examples align with VERSIONING semantics.
- Docs mention delete history and restore semantics.
- Schema diagrams match actual key/value encoding.

Recommended Implementation Order
1) Baseline CRUD + serialization tests.
2) Summary/index consistency tests.
3) Reverse edge denormalization tests.
4) VersionHistory + time travel tests.
5) Restore semantics tests.
6) GC/refcount tests.
7) Property tests.
8) Performance benches.

Acceptance Criteria
- All tests above implemented with deterministic timestamps/ids.
- Coverage includes failure paths and invariants.
- Benchmarks track key regressions over time.

(codex, 2026-02-07, eval: updated to include lifecycle/metrics coverage and new VERSIONING gaps for restore markers, shared-summary safety, and versioned temporal fields.)

Post-Refactor (ARCH2) Testing Adjustments
- Processor API tests: direct sync calls validate mutation/query behavior without async overhead.
- Consumer wiring: Writer/Reader consumers use shared `Arc<Processor>` and still behave identically.
- Facade removal: tests no longer instantiate `Graph` (or verify it is optional/removed).
- Cache ownership: NameCache lives in Processor; verify cache prewarm + shared access across consumers.
- Subsystem lifecycle: start/stop idempotence, GC join, no tokio-blocking in GC thread.
- Transaction API: Writer.transaction uses Processor and preserves read-your-writes.
- Metrics/info: Processor/Subsystem metrics exposed and remain stable after refactor.

---

## Test Coverage Analysis (claude, 2026-02-07)

### Current Test Inventory

| File | Test Count | Coverage Area |
|------|------------|---------------|
| `tests.rs` | 100 | Storage lifecycle, CRUD, summary, refcount, time-travel, versioning, concurrency |
| `query.rs` | 14 | Query execution, timeouts, pagination |
| `gc.rs` | 8 | GC config, lifecycle, metrics |
| `schema.rs` | 14 | CF serialization, key/value encoding |
| `mutation.rs` | 2 | Consumer wiring |
| `processor.rs` | 3 | Processor creation, cache |
| `subsystem.rs` | 6 | Subsystem config, lifecycle |
| `scan.rs` | 8 | Scan iteration, pagination |
| `name_hash.rs` | 20 | Name hashing, cache |
| `summary_hash.rs` | 10 | Summary hashing |
| `transaction.rs` | 1 | Transaction empty check |
| `repair.rs` | 5 | Repair config, metrics |
| `reader.rs` | 2 | Reader closed detection |
| `writer.rs` | 2 | Writer closed detection |

### Coverage Status by TESTING.md Section

#### Baseline (pre-VERSIONING) - Status

| Requirement | Status | Location | Notes |
|-------------|--------|----------|-------|
| Node CRUD | ✅ | tests.rs | test_add_node_with_summary_and_query |
| Edge CRUD | ✅ | tests.rs | test_add_edge_with_summary_and_query |
| Summary writes | ✅ | tests.rs | test_node_summaries_cf_round_trip |
| Summary index | ✅ | tests.rs | test_node_index_entry_current_marker |
| Reverse edge index | ✅ | tests.rs | test_reverse_edge_index_consistency |
| Forward + reverse consistency | ✅ | tests.rs | test_forward_reverse_atomic_commit |
| Fragment append | ✅ | tests.rs | test_fragment_append_idempotency |
| Fragment search | ❌ | MISSING | Need: bm25 smoke test |
| Delete semantics | ✅ | tests.rs | test_delete_node_marks_orphan |
| Error handling | ✅ | tests.rs | test_query_missing_node/edge_returns_error |
| Serialization | ✅ | tests.rs | test_rkyv_* tests (6 tests) |
| Lifecycle | ✅ | tests.rs | test_storage_lifecycle |
| GC lifecycle | ✅ | gc.rs | test_gc_start_shutdown_lifecycle |
| Metrics/info | ✅ | tests.rs | test_gc_metrics, test_repair_metrics |

#### VERSIONING - Status

| Requirement | Status | Location | Notes |
|-------------|--------|----------|-------|
| VersionHistory create | ✅ | tests.rs | test_node_initial_version |
| VersionHistory update | ✅ | tests.rs | test_node_update_creates_version_history |
| VersionHistory delete | ✅ | tests.rs | test_delete_writes_history_tombstone |
| Restore by version | ❌ | MISSING | Need: restore returns correct snapshot |
| Restore CURRENT marker | ❌ | MISSING | Need: no multiple CURRENT markers |
| Restore summary reuse | ❌ | MISSING | Need: reuses existing hash if present |
| Time travel AsOf | ✅ | tests.rs | test_point_in_time_node_query |
| Active period | ✅ | tests.rs | test_active_period_filtering |
| Weight history | ✅ | tests.rs | test_edge_weight_update_creates_version |
| RestoreEdges batch | ❌ | MISSING | Need: batch marks STALE, records orphans |
| Restore missing summary | ❌ | MISSING | Need: fail if summary GC'd |

#### Index and Denormalization - Status

| Requirement | Status | Location | Notes |
|-------------|--------|----------|-------|
| Reverse edge temporal | ✅ | tests.rs | test_reverse_edge_index_consistency |
| CURRENT/STALE markers | ✅ | tests.rs | test_node_index_entry_current_marker |
| Hash prefix scans | ✅ | tests.rs | test_summary_hash_prefix_scan |
| Padded version ordering | ✅ | tests.rs | test_version_scan_returns_latest_first |

#### GC and Storage - Status

| Requirement | Status | Location | Notes |
|-------------|--------|----------|-------|
| Orphan summary tracking | ✅ | tests.rs | test_delete_node_marks_orphan |
| GC retention | ❌ | MISSING | Need: GC respects retention window |
| Shared summary safety | ✅ | tests.rs | test_shared_summary_survives_partial_deletion |
| Fragment GC | ✅ | DESIGN | Fragments are append-only (no GC) |

#### Concurrency and Atomicity - Status

| Requirement | Status | Location | Notes |
|-------------|--------|----------|-------|
| Concurrent updates | ✅ | tests.rs | test_concurrent_writers_same_node |
| Multi-step txn atomicity | ✅ | tests.rs | test_forward_reverse_atomic_commit |
| Idempotency | ✅ | tests.rs | test_replay_mutation_idempotent |
| Shutdown under load | ✅ | tests.rs | test_shutdown_during_writes_no_corruption |

#### ARCH2 Post-Refactor - Status

| Requirement | Status | Location | Notes |
|-------------|--------|----------|-------|
| Processor API tests | ✅ | tests.rs | test_processor_implements_writer_processor |
| Consumer wiring | ✅ | tests.rs | test_graph_consumer_basic_processing |
| Facade removal | ✅ | VERIFIED | Graph struct removed |
| Cache ownership | ✅ | tests.rs | test_namecache_shared_across_consumers |
| Subsystem lifecycle | ✅ | subsystem.rs | test_subsystem_new |
| Transaction API | ✅ | tests.rs | test_transaction_read_your_writes |
| Metrics/info | ✅ | tests.rs | test_gc_metrics |

---

## Test Cleanup and Refactoring Proposal (claude, 2026-02-07)

### Phase 1: Fill Critical Gaps (Priority: HIGH)

1. **Reverse Edge Consistency Test** (`tests.rs`)
   - Add `test_reverse_edge_index_consistency`
   - Verify: add edge → forward CF has entry AND reverse CF has entry
   - Verify: IncomingEdges query returns added edge

2. **Forward+Reverse Atomic Test** (`tests.rs`)
   - Add `test_forward_reverse_atomic_commit`
   - Verify: single txn writes both CFs or neither

3. **Error Handling Tests** (`tests.rs`)
   - Add `test_query_missing_node_returns_error`
   - Add `test_query_missing_edge_returns_error`
   - Verify: NodeById with unknown ID returns descriptive error

4. **NameCache Shared Access Test** (`processor.rs`)
   - Add `test_processor_namecache_shared_across_consumers`
   - Verify: Writer and Reader share same cache via Processor

### Phase 2: VERSIONING Behavioral Validation (Priority: MEDIUM)

5. **VersionHistory Update Test** (`tests.rs`)
   - Add `test_node_update_creates_version_history`
   - Add `test_edge_update_creates_version_history`
   - Verify: UpdateNode → new history entry with ValidSince

6. **Restore Semantics Tests** (`tests.rs`)
   - Add `test_restore_node_returns_correct_version`
   - Add `test_restore_marks_prior_current_as_stale`
   - Add `test_restore_fails_if_summary_gc_deleted`

7. **Weight History Test** (`tests.rs`)
   - Add `test_edge_weight_update_creates_version`
   - Verify: UpdateEdge (with new_weight) → new version (no in-place mutation)

8. **RestoreEdges Batch Test** (`tests.rs`)
   - Add `test_restore_edges_batch_marks_stale`
   - Add `test_restore_edges_skips_missing_summaries`

### Phase 3: Index and Scan Validation (Priority: MEDIUM)

9. **Hash Prefix Scan Test** (`query.rs`)
   - Add `test_summary_hash_prefix_scan`
   - Verify: NodesBySummaryHash returns all entities with same hash

10. **Padded Version Ordering Test** (`scan.rs`)
    - Add `test_version_scan_returns_latest_first`
    - Verify: reverse scan with padded ordering

### Phase 4: Concurrency Tests (Priority: LOW)

11. **Concurrent Writers Test** (`tests.rs` or integration test)
    - Add `test_concurrent_writers_same_node`
    - Verify: two writers update same node → history consistent

12. **Idempotency Test** (`tests.rs`)
    - Add `test_replay_mutation_idempotent`
    - Verify: replaying AddNode with same ID doesn't corrupt

### Phase 5: Test Infrastructure Cleanup

13. **Fixture Helpers** - Create `test_fixtures.rs`:
    - `TestNode::builder()` - deterministic node creation
    - `TestEdge::builder()` - deterministic edge creation
    - `TestClock` - injectable clock for deterministic timestamps
    - `TestIdGenerator` - stable ID generation for assertions

14. **Consolidate Storage Helpers** - Reduce duplication:
    - Extract `setup_readwrite_storage()` helper
    - Extract `setup_processor_with_writer()` helper
    - Use temp_dir fixtures consistently

### Cleanup Actions

15. **Remove Redundant Tests**:
    - `test_storage_ready_multiple_times` duplicates `test_storage_ready_idempotency`
    - Merge `test_storage_readonly_basic` with `test_storage_multiple_readonly_instances`

16. **Rename for Clarity**:
    - `test_graph_consumer_*` → `test_mutation_consumer_*` (Graph removed)
    - `test_consumer_basic` (query.rs) → `test_query_consumer_basic`

17. **Add Missing Doc Comments**:
    - Each test should document what invariant it validates
    - Use `/// Validates: <invariant>` format

---

## Test Execution Groups

### Fast Tests (< 1s each, run on every commit)
- Serialization: rkyv, key/value encoding
- Config: defaults, builders
- Pure functions: hashing, name resolution

### Integration Tests (< 5s each, run on PR)
- CRUD operations
- Time travel queries
- Lifecycle tests

### Stress Tests (> 5s, run nightly)
- Concurrent writers
- Large batch mutations
- GC under load

---

## Acceptance Criteria (updated)

- [ ] All Baseline tests implemented (14 items)
- [ ] All VERSIONING tests implemented (11 items)
- [ ] All Index tests implemented (4 items)
- [ ] All GC tests implemented (4 items)
- [ ] All Concurrency tests implemented (4 items)
- [ ] All ARCH2 tests implemented (7 items)
- [ ] Test fixtures created and used consistently
- [ ] Redundant tests consolidated
- [ ] Tests renamed for clarity post-Graph removal
- [ ] Each test documents invariant validated
