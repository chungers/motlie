# Phase 5: Internal Mutation/Query API

**Status:** In Progress (Task 5.0 Complete)
**Date:** January 14, 2026
**Commits:** `3e33c88`, `f672a2f`, `1524679`, `be0751a`

---

## Overview

Phase 5 implements the internal mutation and query API for vector operations, building on the foundation established in Phases 0-4. The primary goal is to ensure **atomic transactions** for all vector operations, enabling safe online serving.

---

## Task 5.0: HNSW Transaction Refactoring (COMPLETE)

### Problem Statement

The original HNSW implementation used raw `txn_db.put_cf()` and `txn_db.merge_cf()` calls instead of transactional writes:

```rust
// BEFORE: Non-atomic writes
txn_db.put_cf(&cf, key, value)?;   // Could crash mid-write
txn_db.merge_cf(&cf, key, value)?; // Leaves inconsistent state

// AFTER: Atomic writes within transaction
txn.put_cf(&cf, key, value)?;      // All-or-nothing
txn.merge_cf(&cf, key, value)?;    // Rolled back on failure
```

**Impact:** A crash during insert could leave the HNSW graph in an inconsistent state with orphaned nodes, missing edges, or corrupted entry points.

### Solution: Transaction-Aware APIs

#### 5.0.1: IdAllocator Transaction Support

Added transaction-aware allocation methods to `id.rs`:

```rust
impl IdAllocator {
    /// Allocate ID and persist state within transaction
    pub fn allocate_in_txn(
        &self,
        txn: &rocksdb::Transaction<'_, TransactionDB>,
        txn_db: &TransactionDB,
        embedding: EmbeddingCode,
    ) -> Result<VecId>;

    /// Free ID and persist bitmap within transaction
    pub fn free_in_txn(
        &self,
        txn: &rocksdb::Transaction<'_, TransactionDB>,
        txn_db: &TransactionDB,
        embedding: EmbeddingCode,
        id: VecId,
    ) -> Result<()>;
}
```

#### 5.0.2: HNSW Insert Refactoring

Refactored `hnsw/insert.rs` with a new transaction-aware API:

```rust
/// Deferred cache update - applied ONLY after commit
pub struct CacheUpdate {
    pub embedding: EmbeddingCode,
    pub vec_id: VecId,
    pub node_layer: HnswLayer,
    pub is_new_entry_point: bool,
    pub m: usize,
}

impl CacheUpdate {
    /// Apply after txn.commit() succeeds
    pub fn apply(self, nav_cache: &NavigationCache);
}

/// Transaction-aware insert (recommended for production)
pub fn insert_in_txn(
    index: &Index,
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    storage: &Storage,
    vec_id: VecId,
    vector: &[f32],
) -> Result<CacheUpdate>;
```

#### 5.0.3: Graph Edge Connections

Added `connect_neighbors_in_txn()` to `hnsw/graph.rs`:

```rust
pub fn connect_neighbors_in_txn(
    index: &Index,
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
    vec_id: VecId,
    neighbors: &[(f32, VecId)],
    layer: HnswLayer,
) -> Result<()>;
```

#### 5.0.4: Cache Update Timing

**Critical pattern:** Cache updates are deferred until AFTER transaction commit:

```rust
// WRONG: Cache updated before commit (caches uncommitted state)
nav_cache.update(...);
txn.commit()?;

// CORRECT: Cache updated after successful commit
txn.commit()?;
nav_cache.update(...);  // Only on success
```

This ensures the navigation cache never contains uncommitted graph state.

#### 5.0.5: Writer Integration

Updated `writer.rs` to use transactional HNSW operations:

```rust
async fn execute_mutations(&self, mutations: &[Mutation]) -> Result<()> {
    let txn_db = self.processor.storage().transaction_db()?;
    let txn = txn_db.transaction();

    // Collect cache updates during processing
    let mut cache_updates: Vec<CacheUpdate> = Vec::new();

    for mutation in mutations {
        if let Some(update) = self.execute_single(&txn, &txn_db, mutation)? {
            cache_updates.push(update);
        }
    }

    // Commit transaction - all mutations are atomic
    txn.commit()?;

    // Apply cache updates ONLY after successful commit
    for update in cache_updates {
        update.apply(self.processor.nav_cache());
    }

    Ok(())
}
```

#### 5.0.6: Crash Recovery Tests

Added comprehensive tests in `crash_recovery_tests.rs`:

| Test | Purpose |
|------|---------|
| `test_uncommitted_transaction_not_visible` | Verify uncommitted data doesn't persist |
| `test_committed_transaction_persists` | Verify committed data survives restart |
| `test_id_allocator_recovery` | Verify IdAllocator state recovery |
| `test_id_allocator_transactional_recovery` | Verify transactional allocation |
| `test_hnsw_navigation_cold_cache_recovery` | Verify search works with cold cache |
| `test_hnsw_entry_point_persists` | Verify entry point persistence |
| `test_full_crash_recovery_scenario` | End-to-end crash recovery |
| `test_transaction_insert_with_recovery` | Verify insert_in_txn with recovery |

---

## Files Modified

| File | Changes |
|------|---------|
| `id.rs` | Added `allocate_in_txn()`, `free_in_txn()` |
| `hnsw/insert.rs` | Added `CacheUpdate`, `insert_in_txn()`, transaction helpers |
| `hnsw/graph.rs` | Added `connect_neighbors_in_txn()` |
| `hnsw/mod.rs` | Export new APIs, `Index` derives `Clone` |
| `processor.rs` | Added `get_or_create_index()`, `nav_cache()`, `hnsw_config()`, `index_count()` |
| `writer.rs` | Collect and apply `CacheUpdate`s after commit |
| `mod.rs` | Added `crash_recovery_tests` module |
| `crash_recovery_tests.rs` | New file with 8 tests |

---

## Processor HNSW Management

The `Processor` now manages HNSW indices similar to RaBitQ encoders:

```rust
pub struct Processor {
    // ... existing fields ...

    /// Per-embedding HNSW indices (lazily created)
    hnsw_indices: DashMap<EmbeddingCode, hnsw::Index>,
    /// Shared navigation cache for all HNSW indices
    nav_cache: Arc<NavigationCache>,
    /// HNSW configuration (shared across all embeddings)
    hnsw_config: hnsw::Config,
}

impl Processor {
    /// Get or create an HNSW index for the given embedding space
    pub fn get_or_create_index(&self, embedding: EmbeddingCode) -> Option<hnsw::Index>;

    /// Get the shared navigation cache
    pub fn nav_cache(&self) -> &Arc<NavigationCache>;

    /// Get the HNSW configuration
    pub fn hnsw_config(&self) -> &hnsw::Config;

    /// Get the number of cached HNSW indices
    pub fn index_count(&self) -> usize;
}
```

---

## Transaction Flow

The complete transaction flow for vector insertion with HNSW indexing:

```
┌─────────────────────────────────────────────────────────────┐
│                    execute_mutations()                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. txn = txn_db.transaction()                              │
│                                                             │
│  2. For each mutation:                                      │
│     ├── IdForward: txn.put_cf()                            │
│     ├── IdReverse: txn.put_cf()                            │
│     ├── Vectors: txn.put_cf()                              │
│     ├── BinaryCodes: txn.put_cf()                          │
│     └── HNSW (if immediate_index):                         │
│         ├── VecMeta: txn.put_cf()                          │
│         ├── GraphMeta: txn.put_cf()                        │
│         ├── Edges: txn.merge_cf()                          │
│         └── Returns CacheUpdate                            │
│                                                             │
│  3. txn.commit()  ─────────────────── ATOMIC BOUNDARY ───  │
│                                                             │
│  4. For each CacheUpdate:                                   │
│     └── update.apply(nav_cache)  ── Only after commit ──   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## CODEX Concerns Addressed

Two concerns raised by CODEX during Phase 5 planning:

### 1. All Side Effects in Same Transaction

**Concern:** Ensure all vector insert side effects share the same RocksDB transaction.

**Solution:** All CF writes use the same `&Transaction`:
- IdForward, IdReverse (ID mapping)
- Vectors (vector data)
- BinaryCodes (RaBitQ codes)
- VecMeta (HNSW node metadata)
- GraphMeta (entry point, max level)
- Edges (HNSW graph edges)
- IdAlloc (allocator state)

### 2. Defer Cache Updates Until After Commit

**Concern:** Avoid caching uncommitted graph state.

**Solution:** `CacheUpdate` pattern:
- `insert_in_txn()` returns `CacheUpdate` instead of updating cache directly
- Cache updates collected during mutation processing
- Applied only after `txn.commit()` succeeds
- On transaction failure, cache remains unchanged

---

## Testing

All tests pass:
- 487 existing tests
- 8 new crash recovery tests
- Total: 495 tests

```bash
cargo test -p motlie-db --lib
# test result: ok. 495 passed; 0 failed
```

---

## CODEX Review (Post-sync)

**Assessment:** Overall direction is sound: HNSW writes are transaction-scoped, cache updates are deferred until commit, and crash recovery tests are comprehensive. Two correctness gaps remain that should be addressed before declaring Task 5.0 fully complete:

1) **IdAllocator transactional persistence not wired into the write path.**
   - The code introduces `allocate_in_txn()` / `free_in_txn()`, but `writer.rs` still calls `allocator.allocate()` and `allocator.free()` outside the transaction.
   - This means IdAlloc CF may lag behind committed inserts/deletes, and a crash could re-use IDs that were already committed (collision risk).
   - **Action:** Use `allocate_in_txn()` and `free_in_txn()` inside `execute_single()` so allocator state is persisted in the same transaction as vector writes.

2) **Layer count update behavior changed for the empty-graph case.**
   - `insert_in_txn()` returns a `CacheUpdate` that increments `layer_counts[node_layer]`.
   - In the prior empty-graph path, the cache incremented layer 0 unconditionally, ensuring `layer_counts[0]` equals total nodes even when `node_layer > 0`.
   - If `node_layer` is ever > 0 on the first insert, `layer_counts[0]` may remain 0 (affecting `total_nodes()` and diagnostics).
   - **Action:** Consider incrementing layer 0 for the first node (or updating `CacheUpdate::apply` logic for the empty-graph path).

If these are addressed, Task 5.0 aligns with the CODEX concerns raised earlier.

### Resolution (Commit `0cbd597`)

Both issues have been fixed:

1. **IdAllocator transactional persistence - FIXED**
   ```rust
   // BEFORE (writer.rs)
   let vec_id = allocator.allocate();
   allocator.free(vec_id);

   // AFTER (writer.rs)
   let vec_id = allocator.allocate_in_txn(txn, txn_db, op.embedding)?;
   allocator.free_in_txn(txn, txn_db, op.embedding, vec_id)?;
   ```

2. **Layer counts for all nodes - FIXED**
   ```rust
   // BEFORE (CacheUpdate::apply)
   info.increment_layer_count(self.node_layer);

   // AFTER (CacheUpdate::apply)
   // Increment for ALL layers from 0 to node_layer
   for layer in 0..=self.node_layer {
       info.increment_layer_count(layer);
   }
   ```

All 495 tests pass. Task 5.0 is now fully complete.

---

## CODEX Verification (Post-update)

- ✅ `writer.rs` now uses `allocate_in_txn()` and `free_in_txn()` so IdAlloc state is transactionally persisted alongside vector writes.
- ✅ `CacheUpdate::apply` now increments layer counts for all layers `0..=node_layer`, preserving `layer_counts[0]`/`total_nodes()` correctness even for the first node.
- ✅ Changes align with the CODEX concerns raised in the previous review; Task 5.0 looks complete from a correctness standpoint.

No additional follow-up items at this time.

---

## Task 5.1: Processor::insert_vector() (COMPLETE)

### Overview

Task 5.1 implements the high-level `insert_vector()` method on the `Processor` struct, providing a unified API for inserting vectors with full transactional guarantees.

### API Signature

```rust
impl Processor {
    /// Insert a vector into an embedding space.
    ///
    /// This method performs all necessary operations atomically:
    /// 1. Validates embedding exists and dimension matches
    /// 2. Allocates internal VecId
    /// 3. Stores ID mappings (forward: Id→VecId, reverse: VecId→Id)
    /// 4. Stores vector data
    /// 5. Encodes and stores binary codes (RaBitQ)
    /// 6. Optionally builds HNSW index
    ///
    /// # Arguments
    /// * `embedding` - Embedding space code
    /// * `id` - External ID for the vector
    /// * `vector` - Vector data (must match embedding dimension)
    /// * `build_index` - Whether to add to HNSW index immediately
    ///
    /// # Returns
    /// The allocated internal VecId on success.
    pub fn insert_vector(
        &self,
        embedding: EmbeddingCode,
        id: Id,
        vector: &[f32],
        build_index: bool,
    ) -> Result<VecId>;
}
```

### Implementation Details

#### Transaction Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    insert_vector()                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Validate embedding exists in registry                   │
│  2. Validate vector dimension matches spec                  │
│                                                             │
│  3. txn = txn_db.transaction()                              │
│                                                             │
│  4. vec_id = allocator.allocate_in_txn()                   │
│                                                             │
│  5. IdForward: txn.put_cf(id → vec_id)                     │
│  6. IdReverse: txn.put_cf(vec_id → id)                     │
│  7. Vectors: txn.put_cf(vec_id → vector_bytes)             │
│                                                             │
│  8. BinaryCodes (if RaBitQ encoder available):             │
│     └── txn.put_cf(vec_id → binary_code)                   │
│                                                             │
│  9. HNSW (if build_index && hnsw_config.enabled):          │
│     └── insert_in_txn() → CacheUpdate                      │
│                                                             │
│  10. txn.commit() ─────────────────── ATOMIC BOUNDARY ───  │
│                                                             │
│  11. cache_update.apply(nav_cache) ── Only after commit ── │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Key Features

1. **Validation First**: Embedding existence and dimension validated before transaction starts
2. **Single Transaction**: All writes (ID mappings, vector, codes, HNSW) in same transaction
3. **Deferred Cache Update**: HNSW cache updated only after commit succeeds
4. **Optional Indexing**: `build_index` flag allows deferring HNSW indexing for batch loads

### Tests

| Test | Purpose |
|------|---------|
| `test_insert_vector_unknown_embedding` | Verify error for unknown embedding code |
| `test_insert_vector_dimension_mismatch` | Verify error for wrong dimension |
| `test_insert_vector_integration` | Full insert with ID mappings, vector storage, binary codes |

### Files Modified

| File | Changes |
|------|---------|
| `processor.rs` | Added `insert_vector()` method (~100 lines) and 3 tests |

---

## CODEX Review (Task 5.1)

Overall, the implementation is solid: validation is up-front, all writes are within a single transaction, and cache updates are deferred. Two correctness gaps remain to address before relying on this API in production:

1) **Transactional IdAllocator persistence is incomplete for reused IDs.**
   - `allocate_in_txn()` persists `next_id` but does not persist the updated free bitmap after popping an ID from `free_ids`.
   - Crash scenario: if a reused ID is allocated and the process dies before any future `free_in_txn()` write, `IdAlloc` recovery will still include that ID in the free bitmap, allowing duplicate allocation.
   - **Suggested fix:** persist the free bitmap inside `allocate_in_txn()` whenever allocation consumes a freed ID (or persist the bitmap unconditionally).

2) **Duplicate external IDs are not checked.**
   - `insert_vector()` writes IdForward/IdReverse without checking if the external `Id` already exists.
   - This can overwrite IdForward to a new vec_id while leaving the old vec_id orphaned (and its IdReverse mapping stale).
   - **Suggested fix:** check `IdForward` before insert; either return an error or treat this as an upsert with a proper delete + insert flow.

Once these are addressed, Task 5.1 will be complete from a correctness perspective.

### Resolution (Commit `6ee252b`)

Both issues have been fixed:

1. **IdAllocator free bitmap persistence - FIXED**

   Refactored `allocate_in_txn()` to handle allocation and persistence atomically:
   ```rust
   // BEFORE: Only persisted next_id
   let id = self.allocate();  // May remove from bitmap
   // Persist only next_id...

   // AFTER: Inline allocation + conditional bitmap persistence
   let (id, reused) = {
       let mut free = self.free_ids.lock().unwrap();
       if let Some(freed_id) = free.iter().next() {
           free.remove(freed_id);
           (freed_id, true)
       } else {
           (self.next_id.fetch_add(1, Ordering::Relaxed), false)
       }
   };
   // Persist next_id...
   if reused {
       // Also persist free bitmap
   }
   ```

2. **Duplicate external ID check - FIXED**

   Added check before ID allocation in `insert_vector()`:
   ```rust
   // Check if external ID already exists (avoid duplicates)
   let forward_cf = txn_db.cf_handle(IdForward::CF_NAME)?;
   let forward_key = IdForwardCfKey(embedding, id);
   if txn_db.get_cf(&forward_cf, IdForward::key_to_bytes(&forward_key))?.is_some() {
       return Err(anyhow::anyhow!(
           "External ID {} already exists in embedding space {}",
           id, embedding
       ));
   }
   ```

   Added test `test_insert_vector_duplicate_id` to verify the behavior.

All 499 tests pass. Task 5.1 is now complete from a correctness perspective.

---

## CODEX Verification (Post-fix)

- ✅ IdAllocator reuse now persists the free bitmap inside `allocate_in_txn()`, addressing crash-reuse duplication risk.
- ✅ `insert_vector()` rejects duplicate external IDs and includes a regression test.
- ⚠️ **Remaining improvement:** the duplicate-ID check uses `txn_db.get_cf()` outside the transaction. This can race under concurrent inserts of the same external ID. Prefer `txn.get_cf()` (or `get_for_update`) so the read participates in the same transaction/locking semantics as the write to `IdForward`.

No other correctness issues found in the Task 5.1 implementation.

### Resolution (Race Condition Fix)

Changed `txn_db.get_cf()` to `txn.get_for_update_cf()`:

```rust
// BEFORE: Non-transactional read (can race)
if txn_db.get_cf(&forward_cf, key)?.is_some() { ... }

// AFTER: Transactional read with lock
if txn.get_for_update_cf(&forward_cf, key, true)?.is_some() { ... }
```

The `get_for_update_cf()` acquires a lock on the key, preventing concurrent inserts of the same external ID from racing. All 499 tests pass.

---

## Remaining Phase 5 Tasks

| Task | Description | Status |
|------|-------------|--------|
| 5.1 | Processor::insert_vector() | **Complete** |
| 5.2 | Processor::delete_vector() | Pending |
| 5.3 | Processor::search() | Pending |
| 5.4 | Storage layer search methods | Pending |
| 5.5 | Dispatch logic for search strategies | Pending |
| 5.6 | Migration utilities | Pending |
| 5.7 | Stress tests | Pending |
| 5.8 | Search metrics | Pending |
| 5.9 | Batch insert benchmark | Pending |
| 5.10 | Delete + compact benchmark | Pending |
| 5.11 | Documentation update | Pending |

---

## References

- [ROADMAP.md](./ROADMAP.md) - Full implementation roadmap
- [API.md](./API.md) - Public API reference
- [BENCHMARK.md](./BENCHMARK.md) - Performance results
