# CODEX2: Transaction-Only Write Path Plan

**Goal:** Make all RocksDB writes transactional, then remove/rename non-transactional APIs so the `_in_txn` suffix can be dropped cleanly.

## Current State Summary

Most production write paths already use `txn.put_cf` / `txn.merge_cf` via `ops::*` and `Processor`. However, a small number of **non-transactional writes** remain in production code and in tests/bench utilities. These prevent a clean removal of non-transactional APIs and the `_in_txn` naming.

## Recent Updates (Context)

- **Batch insert edge visibility** is now handled via `BatchEdgeCache` (`libs/db/src/vector/hnsw/graph.rs`, `libs/db/src/vector/hnsw/insert.rs`, `libs/db/src/vector/ops/insert.rs`).
- This addresses pending-merge visibility during batch inserts, but **does not change** the transactional migration requirements below.

## Non-Transactional Writes That Must Be Eliminated

### Production Code

1) **`EmbeddingRegistry::register()`**
   - **File:** `libs/db/src/vector/registry.rs`
   - **Write:** `db.put_cf(...)` (non-transactional)
   - **Impact:** Registration is not atomic with other writes; no shared transaction context.
   - **Migration:** Route through a transaction-backed path (e.g., `ops::embedding::spec()` or a new `register_in_txn()`), and update callers accordingly.

2) **`IdAllocator::persist()`**
   - **File:** `libs/db/src/vector/id.rs`
   - **Write:** `db.put_cf(...)` (non-transactional)
   - **Impact:** Persisting allocator state is not atomic with other mutations.
   - **Migration:** Replace with transaction-backed persistence, or make this method internal-only and call `allocate_in_txn()` / `free_in_txn()` exclusively.

3) **Legacy HNSW Edge Writes**
   - **File:** `libs/db/src/vector/hnsw/graph.rs`
   - **Write:** `connect_neighbors()` uses `txn_db.merge_cf(...)` (non-transactional)
   - **Impact:** Leaves a legacy non-transactional API in the core graph module.
   - **Migration:** Remove or restrict to tests (rename to `connect_neighbors_legacy` and keep `pub(crate)`), so only transactional edge writes are used.

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
