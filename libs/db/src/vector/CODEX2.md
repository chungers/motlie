# CODEX2: Transaction-Only Write Path Plan

**Goal:** Make all RocksDB writes transactional, then remove/rename non-transactional APIs so the `_in_txn` suffix can be dropped cleanly.

## Current State Summary

Most production write paths already use `txn.put_cf` / `txn.merge_cf` via `ops::*` and `Processor`. However, a small number of **non-transactional writes** remain in production code and in tests/bench utilities. These prevent a clean removal of non-transactional APIs and the `_in_txn` naming.

## Recent Updates (Context)

- **Batch insert edge visibility** is now handled via `BatchEdgeCache` (`libs/db/src/vector/hnsw/graph.rs`, `libs/db/src/vector/hnsw/insert.rs`, `libs/db/src/vector/ops/insert.rs`).
- This addresses pending-merge visibility during batch inserts, but **does not change** the transactional migration requirements below.
- **Production non-transactional writes reduced:** `EmbeddingRegistry::register()` now uses a transaction, `connect_neighbors()` was removed, and `Processor::persist_allocators()` was deleted. `IdAllocator::persist()` remains for tests only.

## Non-Transactional Writes That Must Be Eliminated

### Production Code

1) **`EmbeddingRegistry::register()`**
   - **File:** `libs/db/src/vector/registry.rs`
   - **Write:** Uses an internal `db.transaction()` + `txn.put_cf(...)` (transactional but isolated).
   - **Impact:** Still not shareable with an outer transaction; registration cannot be grouped atomically with other writes.
   - **Migration:** Allow callers to provide a transaction (e.g., `register_in_txn()` or route through `ops::embedding::spec()`), then deprecate the standalone path.

2) **`IdAllocator::persist()`**
   - **File:** `libs/db/src/vector/id.rs`
   - **Write:** `db.put_cf(...)` (non-transactional)
   - **Impact:** Persisting allocator state is not atomic with other mutations.
   - **Migration:** Replace with transaction-backed persistence, or make this method internal-only and call `allocate_in_txn()` / `free_in_txn()` exclusively.

3) **Legacy HNSW Edge Writes**
   - **File:** `libs/db/src/vector/hnsw/graph.rs`
   - **Status:** `connect_neighbors()` removed; only `connect_neighbors_in_txn()` remains.
   - **Follow-up:** Keep this file transactional-only; avoid reintroducing non-transactional edge APIs.

### Test / Benchmark Helpers (still need cleanup for consistency)

4) **HNSW test helpers**
   - **File:** `libs/db/src/vector/hnsw/mod.rs` (tests)
   - **Write:** `store_vectors()` uses `txn_db.put_cf(...)`
   - **Impact:** Tests exercise non-transactional writes.
   - **Migration:** Wrap in a transaction or use existing `ops::insert` helpers.

5) **Benchmark runner**
   - **File:** `libs/db/src/vector/benchmark/runner.rs`
   - **Write:** `txn_db.put_cf(...)`
   - **Impact:** Benchmarks bypass transactional paths.
   - **Migration:** Use transactional insert helpers or explicit transaction per batch.

6) **Crash recovery tests**
   - **File:** `libs/db/src/vector/crash_recovery_tests.rs`
   - **Write:** Mixed direct `txn_db.put_cf(...)` for setup.
   - **Impact:** Acceptable for low-level tests but should be marked as legacy or moved behind explicit transactions for parity.

## `_in_txn` Functions: Candidates for Rename (after cleanup)

Once **all** writes are transactional and non-transactional APIs are removed or isolated, these names can be cleaned up:

- **Allocator**
  - `allocate_in_txn()` → `allocate()` (transaction-required)
  - `free_in_txn()` → `free()` (transaction-required)

- **HNSW Insert**
  - `insert_in_txn()` → `insert()`
  - `insert_in_txn_for_batch()` → `insert_for_batch()` (still transactional but batch-specific)

- **HNSW Graph Helpers**
  - `connect_neighbors_in_txn()` → `connect_neighbors()`
  - `distance_in_txn()` → `distance()` (for batch/txn variant)
  - `greedy_search_layer_in_txn()` → `greedy_search_layer()` (txn-aware)
  - `search_layer_in_txn()` → `search_layer()` (txn-aware)
  - **Note:** Batch-cache-aware helpers (e.g., `*_with_batch_cache`) should keep their suffixes to avoid ambiguity with non-batch variants.

- **Internal insert helpers**
  - `store_vec_meta_in_txn()` → `store_vec_meta()`
  - `update_entry_point_in_txn()` → `update_entry_point()`

## Suggested Migration Steps

1) **Make registration fully transactional**
   - Replace `EmbeddingRegistry::register()` internal write with a transaction-backed implementation (or route through `ops::embedding::spec()`).

2) **Eliminate non-transactional allocator persistence**
   - Deprecate `IdAllocator::persist()` (or make it `pub(crate)` + used only for tests).
   - Ensure allocator state always updated via `allocate_in_txn()` / `free_in_txn()`.

3) **Remove/rename legacy non-transactional HNSW APIs**
   - Keep only transactional HNSW edge updates and insert operations as the public path.

4) **Update tests/bench to use transaction-backed writes**
   - Align all setup helpers with transactional insert patterns.

5) **Rename `_in_txn` APIs once no non-transactional variants remain**
   - This is the final cleanup step to remove naming noise without ambiguity.

## Impacted Files (Summary)

- `libs/db/src/vector/registry.rs`
- `libs/db/src/vector/id.rs`
- `libs/db/src/vector/hnsw/graph.rs`
- `libs/db/src/vector/hnsw/insert.rs`
- `libs/db/src/vector/ops/insert.rs`
- `libs/db/src/vector/hnsw/mod.rs` (tests)
- `libs/db/src/vector/benchmark/runner.rs`
- `libs/db/src/vector/crash_recovery_tests.rs`

## Notes

Until the non-transactional code paths are removed or isolated, renaming `_in_txn` functions would be misleading because both transactional and non-transactional variants would coexist. The rename should be a follow-on cleanup after all writes are transaction-only.

## Rename Readiness Assessment (Post-01192dc)

**Not ready to drop `_in_txn` yet.** Recent changes made test/bench writes transactional, and removed legacy edge writes, but two API surface issues remain:

1) **Dual insert APIs:** `hnsw::insert()` still exists and creates its own transaction, while `insert_in_txn()` is the externally managed transaction version. Renaming would be ambiguous unless the internal-transaction helper is removed or renamed (e.g., `insert_with_auto_txn()`).

2) **Embedding registration not externally transactional:** `EmbeddingRegistry::register()` now uses an internal transaction but cannot participate in a caller transaction. To cleanly drop `_in_txn`, expose a `register_in_txn()` (or route through `ops::embedding::spec()`), then deprecate the standalone path.

**Remaining safe exception:** `IdAllocator::persist()` is `pub(crate)` and test-only; it should stay isolated, but does not block renaming if all public APIs are transaction-only.
