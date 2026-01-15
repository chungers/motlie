# Phase 5: Internal Mutation/Query API

**Status:** In Progress (Tasks 5.0-5.5 Complete; 5.6-5.11 Pending)
**Date:** January 14, 2026
**Commits:** `3e33c88`, `f672a2f`, `1524679`, `be0751a`, `c29dceb`, `6ee252b`, `0cbd597`, `dc4c3f9`

## Task Numbering Alignment with ROADMAP.md

| ROADMAP Task | Description | PHASE5 Status |
|--------------|-------------|---------------|
| 5.0 | HNSW Transaction Refactoring | ✅ Complete |
| 5.1 | Processor::insert_vector() | ✅ Complete |
| 5.2 | Processor::delete_vector() | ✅ Complete |
| 5.3 | Processor::insert_batch() | ✅ Complete |
| 5.4 | Storage/Processor Search | ✅ Complete (5.4a-5.4g) |
| 5.5 | Processor::add_embedding_spec() | ✅ Complete (via registry) |
| 5.6 | Mutation Dispatch | 🔲 Not Started |
| 5.7 | Query Dispatch | 🔲 Not Started |
| 5.8 | Migrate Examples/Integration Tests | 🔲 Not Started |
| 5.9 | Multi-Threaded Stress Tests | 🔲 Not Started |
| 5.10 | Metrics Collection Infrastructure | 🔲 Not Started |
| 5.11 | Concurrent Benchmark Baseline | 🔲 Not Started |

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

## CODEX Verification (Post-race-fix)

- ✅ Duplicate-ID check is now transactional via `get_for_update_cf`, removing the race noted in the prior review.
- ✅ No additional correctness gaps found in Task 5.1 after this change.

---

## Task 5.2: Processor::delete_vector() (COMPLETE)

### Overview

Task 5.2 implements the `delete_vector()` method on the `Processor` struct, providing atomic vector deletion with ID reuse.

### API Signature

```rust
impl Processor {
    /// Delete a vector from an embedding space.
    ///
    /// # Returns
    /// - `Ok(Some(vec_id))` - Vector was deleted, returns the freed VecId
    /// - `Ok(None)` - Vector did not exist (idempotent)
    /// - `Err(...)` - Storage or transaction error
    pub fn delete_vector(
        &self,
        embedding: EmbeddingCode,
        id: Id,
    ) -> Result<Option<VecId>>;
}
```

### Implementation Details

#### Transaction Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    delete_vector()                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. txn = txn_db.transaction()                              │
│                                                             │
│  2. vec_id = txn.get_for_update_cf(IdForward, id)          │
│     └── Return Ok(None) if not found (idempotent)          │
│                                                             │
│  3. txn.delete_cf(IdForward, id)                           │
│  4. txn.delete_cf(IdReverse, vec_id)                       │
│  5. txn.delete_cf(Vectors, vec_id)                         │
│  6. txn.delete_cf(BinaryCodes, vec_id) ── if RaBitQ ──    │
│  7. allocator.free_in_txn(vec_id)                          │
│                                                             │
│  8. txn.commit() ─────────────────── ATOMIC BOUNDARY ───   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Key Features

1. **Idempotent**: Deleting non-existent vector returns `Ok(None)` instead of error
2. **Race-safe**: Uses `get_for_update_cf()` to lock the key during lookup
3. **ID Reuse**: Freed VecIds are returned to allocator for reuse
4. **Atomic**: All deletes in single transaction

#### HNSW Note

HNSW graph edges are NOT cleaned up on delete. This is standard HNSW behavior:
- Deleted nodes are skipped during search
- Edges cleaned up lazily during compaction or rebuild

### Tests

| Test | Purpose |
|------|---------|
| `test_delete_vector_success` | Delete existing vector, verify mappings removed |
| `test_delete_vector_not_found` | Delete non-existent vector (idempotent) |
| `test_delete_vector_id_reuse` | Verify VecId reuse after delete |

### Files Modified

| File | Changes |
|------|---------|
| `processor.rs` | Added `delete_vector()` method (~65 lines) and 3 tests |

---

## CODEX Review (Task 5.2)

The transactional delete logic is clean, but there are two correctness risks if HNSW indexing is enabled:

1) **VecId reuse while HNSW edges still reference the old node.**
   - `delete_vector()` frees the VecId back to the allocator, but HNSW edges are left intact.
   - If the VecId is reused, existing edges now point to a *different* vector, corrupting graph semantics and search results.
   - **Suggested fix:** do not reuse VecIds for indexed embeddings until edges are cleaned; alternatively, add a tombstone flag and keep IDs reserved.

2) **Entry point / navigation can point to deleted nodes.**
   - `GraphMeta` entry point is not updated on delete, and vector data is removed.
   - Search starts from the entry point and calls `distance()`, which will error if the vector data is missing.
   - **Suggested fix:** keep vector data for deleted nodes (tombstone) or update search to skip missing vectors and repair entry points.

If deletes are intended only for non-indexed embeddings, document that constraint explicitly. Otherwise, the above needs addressing before Task 5.2 can be considered safe for indexed usage.

### Resolution

Both issues addressed by implementing **soft delete** when HNSW is enabled:

1. **VecId reuse prevention - FIXED**
   - Added `enabled` field to `hnsw::Config` (default: `true`)
   - When HNSW is enabled, `delete_vector()` does NOT free the VecId
   - This prevents graph corruption from VecId reuse

2. **Vector data preservation - FIXED**
   - When HNSW is enabled, vector data is kept (not deleted)
   - Entry point and edges can still compute distances
   - Only ID mappings (IdForward, IdReverse) are removed

**Behavior summary:**

| HNSW Enabled | ID Mappings | Vector Data | VecId | Binary Codes |
|--------------|-------------|-------------|-------|--------------|
| `false`      | Deleted     | Deleted     | Freed | Deleted      |
| `true`       | Deleted     | **Kept**    | **Reserved** | Kept    |

Updated docstring documents this "soft delete" behavior. Full cleanup requires index rebuild (future work).

All 502 tests pass.

---

## CODEX Verification (Post-soft-delete)

- ✅ Soft delete behavior addresses both VecId reuse corruption and entry-point distance errors for HNSW-enabled embeddings.
- ⚠️ **Remaining improvement:** deleted nodes can still appear in search results because the graph is unchanged and no tombstone filter exists. Consider adding a deleted-flag check (e.g., IdReverse existence or a VecMeta flag) during search/rerank to exclude deleted vectors from results.

### Design Note: Deleted Node Filtering (for Task 5.3)

**Problem:** Soft-deleted vectors remain in the HNSW graph and will appear in raw search results.

**Solution:** Filter deleted vectors during result processing in `Processor::search()`:

```rust
// After HNSW search returns raw VecId results:
// 1. Batch lookup IdReverse for all result VecIds
// 2. Filter out results where IdReverse is missing (deleted)
// 3. Return only live vectors with their external Ids

fn filter_deleted_results(
    results: Vec<(f32, VecId)>,
    embedding: EmbeddingCode,
    txn_db: &TransactionDB,
) -> Result<Vec<(f32, VecId, Id)>> {
    let reverse_cf = txn_db.cf_handle(IdReverse::CF_NAME)?;

    results.into_iter()
        .filter_map(|(dist, vec_id)| {
            // Check if IdReverse exists (not deleted)
            let key = IdReverseCfKey(embedding, vec_id);
            match txn_db.get_cf(&reverse_cf, IdReverse::key_to_bytes(&key)) {
                Ok(Some(bytes)) => {
                    let id = IdReverse::value_from_bytes(&bytes).ok()?.0;
                    Some(Ok((dist, vec_id, id)))
                }
                Ok(None) => None, // Deleted - skip
                Err(e) => Some(Err(e.into())),
            }
        })
        .collect()
}
```

This approach:
- Uses IdReverse as implicit tombstone (deleted = missing)
- Avoids extra storage for tombstone flags
- Batch-friendly with MultiGet optimization potential

---

## Task 5.3: Processor::insert_batch() (COMPLETE)

### Overview

Task 5.3 implements batch vector insertion for efficiency. More efficient than individual `insert_vector()` calls:
- Single transaction for all vectors (atomic commit)
- Spec hash validated once per batch
- Batched RocksDB writes
- HNSW graph built after all vectors stored (better connectivity)

### API Signature

```rust
impl Processor {
    /// Batch insert multiple vectors into an embedding space.
    pub fn insert_batch(
        &self,
        embedding: EmbeddingCode,
        vectors: &[(Id, Vec<f32>)],
        build_index: bool,
    ) -> Result<Vec<VecId>>;
}
```

### Implementation Details

#### Transaction Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    insert_batch()                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Validate embedding exists                               │
│  2. Validate ALL dimensions upfront (fail fast)            │
│  3. Check for duplicate IDs within batch                   │
│                                                             │
│  4. txn = txn_db.transaction()                              │
│                                                             │
│  5. Check for existing external IDs (acquire locks)        │
│  6. Validate/store spec hash (once per batch)              │
│                                                             │
│  7. For each vector:                                        │
│     ├── vec_id = allocator.allocate_in_txn()              │
│     ├── IdForward: txn.put_cf(id → vec_id)                │
│     ├── IdReverse: txn.put_cf(vec_id → id)                │
│     ├── Vectors: txn.put_cf(vec_id → vector_bytes)        │
│     └── BinaryCodes: txn.put_cf() (if RaBitQ)             │
│                                                             │
│  8. Build HNSW index (if build_index):                     │
│     └── For each vec_id: insert_in_txn() → CacheUpdate    │
│                                                             │
│  9. txn.commit() ─────────────────── ATOMIC BOUNDARY ───   │
│                                                             │
│  10. Apply nav_cache and code_cache updates                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Key Features

1. **Fail-Fast Validation**: All dimensions and IDs validated before transaction starts
2. **Duplicate Detection**: Checks both within-batch and against existing data
3. **Spec Hash Once**: Drift detection run only once for the entire batch
4. **Deferred HNSW Build**: All vectors stored before graph connections (better connectivity)
5. **Atomic Commit**: All-or-nothing - failed batch leaves no partial state

#### Error Cases

| Error | Condition |
|-------|-----------|
| Unknown embedding | Embedding code not in registry |
| Dimension mismatch | Any vector dimension != embedding spec |
| Duplicate in batch | Same external ID appears twice in input |
| Duplicate existing | External ID already exists in embedding space |
| Spec hash mismatch | EmbeddingSpec changed since previous insert |

### Tests

| Test | Purpose |
|------|---------|
| `test_insert_batch_basic` | Insert 10 vectors without index, verify VecIds |
| `test_insert_batch_with_index` | Insert 20 vectors with HNSW, verify search works |
| `test_insert_batch_empty` | Empty batch returns empty result |
| `test_insert_batch_dimension_mismatch` | Verify error on wrong dimension |
| `test_insert_batch_duplicate_in_batch` | Detect duplicate IDs within batch |
| `test_insert_batch_duplicate_existing` | Detect conflict with existing data |
| `test_insert_batch_unknown_embedding` | Verify error on unknown embedding |
| `test_insert_batch_atomicity` | Failed batch does not partially commit |

### Files Modified

| File | Changes |
|------|---------|
| `processor.rs` | Added `insert_batch()` method (~250 lines) and 8 tests |

---

## Task 5.3: CODEX Review - Batch Insert Assessment

**Overall:** Solid implementation with correct transactional semantics, spec-hash validation once per batch, and deferred HNSW insertion for better connectivity. Tests cover the critical failure modes and atomicity.

**Remaining issues / improvements**

1) **BinaryCodes CF handling is inconsistent with single insert.**  
   `insert_batch()` treats `BinaryCodes` CF as optional (`codes_cf` is `Option`), so when RaBitQ is enabled but the CF is missing it silently skips code writes. `insert_vector()` hard-fails in this case.  
   - **Risk:** silent perf regression and cache misses in search; violates “config implies data exists.”  
   - **Fix:** align with `insert_vector()` by requiring the CF when encoder is present, or explicitly disable RaBitQ in this path with a warning.

2) **Spec hash lookup uses `txn_db.get_cf` instead of `txn.get_cf`.**  
   This mirrors `insert_vector()`, so it’s not new, but it means the spec-hash read is outside the transaction snapshot.  
   - **Risk:** benign race if another writer updates spec hash between validation and commit (should be impossible if spec is immutable, but worth keeping consistent).  
   - **Fix:** use the transaction handle for all reads in the batch path for clarity.

No other correctness issues found in the batch implementation.

---

## Task 5.3: CODEX Review Fixes (COMPLETE)

Both issues identified in the CODEX review have been addressed:

### Fix 1: BinaryCodes CF Consistency

**Before:** `insert_batch()` treated BinaryCodes CF as optional, silently skipping writes if CF was missing.

**After:** Now fails with error if encoder exists but CF is missing, matching `insert_vector()` behavior.

```rust
// Before (buggy): silently skipped
let codes_cf = txn_db.cf_handle(BinaryCodes::CF_NAME);  // Option
if let Some(ref enc) = encoder {
    if let Some(ref cf) = codes_cf {  // silently skipped if None
        // ...
    }
}

// After (fixed): fails if CF missing when encoder present
if let Some(ref enc) = encoder {
    let codes_cf = txn_db
        .cf_handle(BinaryCodes::CF_NAME)
        .ok_or_else(|| anyhow::anyhow!("BinaryCodes CF not found"))?;
    // ...
}
```

### Fix 2: Spec Hash Transaction Consistency

**Before:** Used `txn_db.get_cf()` for spec hash lookup (outside transaction snapshot).

**After:** Uses `txn.get_cf()` for snapshot consistency within the transaction.

```rust
// Before: outside transaction snapshot
let spec_bytes = txn_db.get_cf(&specs_cf, ...)?;
if let Some(stored_bytes) = txn_db.get_cf(&graph_meta_cf, ...)? { ... }

// After: within transaction snapshot
let spec_bytes = txn.get_cf(&specs_cf, ...)?;
if let Some(stored_bytes) = txn.get_cf(&graph_meta_cf, ...)? { ... }
```

All 524 tests pass after fixes.

---

## Task 5.4a: Processor::search() (COMPLETE)

### Overview

Task 5.4a implements the unified search API that:
1. Performs HNSW approximate nearest neighbor search
2. Filters out soft-deleted vectors
3. Returns external IDs with distances

### API Signature

```rust
/// Search result with external ID and distance.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// External ID (ULID)
    pub id: Id,
    /// Internal vector ID
    pub vec_id: VecId,
    /// Distance to query vector
    pub distance: f32,
}

impl Processor {
    /// Search for nearest neighbors in an embedding space.
    ///
    /// # Arguments
    /// * `embedding` - Embedding space code
    /// * `query` - Query vector (must match embedding dimension)
    /// * `k` - Number of results to return
    /// * `ef_search` - Search beam width (higher = better recall, slower)
    ///
    /// # Returns
    /// Up to k nearest neighbors, sorted by distance (ascending).
    /// Soft-deleted vectors are automatically filtered out.
    pub fn search(
        &self,
        embedding: EmbeddingCode,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<SearchResult>>;
}
```

### Implementation Details

#### Search Flow

```
┌─────────────────────────────────────────────────────────────┐
│                       search()                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Validate embedding exists in registry                   │
│  2. Validate query dimension matches spec                   │
│  3. Check HNSW is enabled (error if disabled)              │
│                                                             │
│  4. Get or create HNSW index for embedding                  │
│                                                             │
│  5. HNSW search: index.search(query, k, ef_search)         │
│     └── Returns raw Vec<(distance, VecId)>                 │
│                                                             │
│  6. Filter deleted vectors:                                 │
│     For each (distance, vec_id):                           │
│       └── Check IdReverse exists (not deleted)             │
│       └── If missing → skip (soft-deleted)                 │
│       └── If present → include in results with external Id │
│                                                             │
│  7. Return Vec<SearchResult>                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Key Features

1. **Validation First**: Embedding existence, dimension, and HNSW enabled checked before search
2. **Tombstone Filtering**: Uses IdReverse existence as implicit tombstone check
3. **External ID Resolution**: Returns external Ids (not internal VecIds) for client use
4. **Sorted Results**: Results are sorted by distance ascending (from HNSW)

#### Error Cases

| Error | Condition |
|-------|-----------|
| Unknown embedding | Embedding code not in registry |
| Dimension mismatch | Query dimension != embedding spec dimension |
| HNSW disabled | `hnsw_config.enabled = false` |

### Tests

| Test | Purpose |
|------|---------|
| `test_search_basic` | Insert vectors, search, verify results sorted by distance |
| `test_search_filters_deleted` | Insert and delete vectors, verify deleted are filtered |
| `test_search_dimension_mismatch` | Verify error for wrong query dimension |
| `test_search_hnsw_disabled` | Verify error when HNSW is disabled |

### Files Modified

| File | Changes |
|------|---------|
| `processor.rs` | Added `SearchResult` struct, `search()` method (~60 lines), 4 tests |

---

## CODEX Review (Task 5.3)

Search works and correctly filters deleted vectors, but there are two practical improvements worth addressing:

1) **Underfilled results when many tombstones exist.**
   - The search runs HNSW with `k` and then filters deleted results. If many results are soft-deleted, callers may receive fewer than `k` results.
   - **Suggested fix:** overfetch candidates (e.g., `k * 2` or `k + deleted_slack`) and then filter down to `k`, or add a loop to increase `ef_search`/`k` until enough live results are found (bounded).

2) **IdReverse lookups are per-result, not batched.**
   - Current filter uses `txn_db.get_cf()` in a loop, which is fine for small `k` but becomes costly as `k` or ef_search grows.
   - **Suggested fix:** use `multi_get_cf` or a small batch read for IdReverse keys to reduce RocksDB overhead.

These are performance/UX improvements rather than correctness blockers, but they become important as the number of soft-deleted nodes grows.

### Resolution

Both improvements implemented:

1. **Overfetch with 2x multiplier - FIXED**
   ```rust
   // Overfetch by 2x to ensure we get enough live results after filtering
   let overfetch_k = k * 2;
   let raw_results = index.search(&self.storage, query, overfetch_k, ef_search)?;
   ```

2. **Batched IdReverse lookups using multi_get_cf - FIXED**
   ```rust
   // Build batch of keys for multi_get
   let keys: Vec<_> = raw_results.iter()
       .map(|(_, vec_id)| IdReverse::key_to_bytes(&IdReverseCfKey(embedding, *vec_id)))
       .collect();

   // Batch lookup all IdReverse keys at once
   let key_refs: Vec<_> = keys.iter().map(|k| (&reverse_cf, k.as_slice())).collect();
   let values = txn_db.multi_get_cf(key_refs);

   // Filter and resolve external IDs, truncate to k
   ```

All 506 tests pass. Task 5.3 improvements complete.

---

## CODEX Verification (Post-overfetch/batch)

- ✅ Overfetch + batched IdReverse lookup fixes the two performance/UX concerns raised for Task 5.3.
- ⚠️ **Remaining improvement:** if `ef_search < overfetch_k`, the HNSW search still returns at most `ef_search` candidates, so tombstones can still reduce results below `k`. Consider setting `effective_ef = max(ef_search, overfetch_k)` or scaling `ef_search` along with overfetch.

### Resolution

Fixed by ensuring `ef_search >= overfetch_k`:

```rust
let overfetch_k = k * 2;
let effective_ef = ef_search.max(overfetch_k);
let raw_results = index.search(&self.storage, query, overfetch_k, effective_ef)?;
```

This guarantees HNSW explores enough candidates to return `overfetch_k` results.

## CODEX Verification (Post-ef scaling)

- ✅ `effective_ef = max(ef_search, overfetch_k)` addresses the underfilled-results risk when tombstones are present.
- ✅ No further issues identified for Task 5.3 at this time.

---

## Task 5.4b: Storage Layer Search (COMPLETE)

### Overview

The ROADMAP specified `Storage::search()` as a convenience wrapper. This functionality is **already implemented** via `Processor::search()`, which is the recommended high-level API.

### Architecture Decision

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Processor  ◄─── Primary API for vector operations          │
│    ├── insert_vector()                                      │
│    ├── delete_vector()                                      │
│    └── search()  ◄─── Implements all ROADMAP 5.4 features   │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Storage = rocksdb::Storage<Subsystem>                      │
│    └── Low-level RocksDB wrapper                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### What Was Implemented (via Processor)

| ROADMAP Requirement | Implementation |
|---------------------|----------------|
| Search with external ID resolution | `Processor::search()` |
| `SearchResult` struct | `processor::SearchResult` (now re-exported) |
| Dimension validation | ✅ Validates query.len() vs spec.dim() |
| Tombstone filtering | ✅ Via IdReverse existence check |
| Batch lookups | ✅ Via `multi_get_cf` |
| Overfetch for tombstones | ✅ 2x overfetch with ef scaling |

### Export Added

```rust
// libs/db/src/vector/mod.rs
pub use processor::{Processor, SearchResult};
```

---

## Task 5.4c: Search Strategy Dispatch (COMPLETE)

### Overview

Task 5.5 implements strategy-based search dispatch via `Processor::search_with_config()`.

### Implementation

```rust
impl Processor {
    /// Search using a SearchConfig for strategy-based dispatch.
    pub fn search_with_config(
        &self,
        config: &SearchConfig,
        query: &[f32],
    ) -> Result<Vec<SearchResult>> {
        // Dispatch based on strategy
        match config.strategy() {
            SearchStrategy::Exact => {
                // Standard HNSW search with exact distance
                index.search(...)
            }
            SearchStrategy::RaBitQ { use_cache } => {
                if use_cache {
                    // Two-phase search with cached binary codes
                    index.search_with_rabitq_cached(...)
                } else {
                    // Fallback to exact (uncached RaBitQ defeats purpose)
                    index.search(...)
                }
            }
        }
    }
}
```

### Changes Made

| File | Changes |
|------|---------|
| `processor.rs` | Added `BinaryCodeCache` field, `code_cache()` getter, `search_with_config()` method |
| `processor.rs` | Updated `insert_vector()` to populate code_cache after commit |
| `processor.rs` | Added 2 tests: `test_search_with_config_exact`, `test_search_with_config_rabitq` |

### Strategy Selection

| Distance Metric | Auto-Selected Strategy |
|-----------------|------------------------|
| Cosine | RaBitQ (ADC approximates angular distance) |
| L2 | Exact (ADC not compatible) |
| DotProduct | Exact (ADC not compatible) |

### Code Cache Integration

Binary codes are now:
1. Persisted to `BinaryCodes` CF in RocksDB (for durability)
2. Cached in `BinaryCodeCache` (for fast RaBitQ search)

Both updates happen after transaction commit to maintain consistency.

---

## CODEX Review (API Consistency + Workflow)

Given the intended workflow (register embedding → build index → search), there are a few consistency checks and design improvements to consider so search is constrained by how the index was built:

1) **Validate SearchConfig vs registered embedding spec.**
   - `search_with_config()` uses `config.embedding().code()` but does not verify that the `SearchConfig`'s embedding (distance/dim/storage type) matches the registry spec for that code.
   - **Suggestion:** compare the registry spec to `config.embedding()` and error on mismatch to avoid stale or forged configs.

2) **Persist per-embedding HNSW/RaBitQ parameters.**
   - HNSW config (M, ef_construction, etc.) and RaBitQ config (bits_per_dim, rotation_seed) are global in `Processor`, but indexes are built per embedding.
   - If these configs change between build and search, the search config may no longer reflect how the index was built.
   - **Suggestion:** persist HNSW config (and RaBitQ params) in GraphMeta/EmbeddingSpec, and have `get_or_create_index()` use the persisted values. Then validate SearchConfig against those persisted params.

3) **RaBitQ cache lifecycle and fallback behavior.**
   - `BinaryCodeCache` is in-memory only and not warmed from `BinaryCodes` on restart.
   - `search_with_config()` will still work (falls back to exact distance on cache miss) but performance can degrade silently.
   - **Suggestion:** either prewarm the cache from `BinaryCodes` on startup, or detect low cache coverage and fall back to Exact with a warning/metric.

4) **Explicit behavior for RaBitQ without cache.**
   - `SearchStrategy::RaBitQ { use_cache: false }` currently falls back to Exact.
   - **Suggestion:** either document this clearly or return an error to avoid surprising behavior.

These are not correctness blockers today, but they tighten the guarantee that search reflects the registered embedding and built index parameters.

---

### Agreement Summary (Workflow Consistency)

We are aligned on the following approach for embedding → build → search consistency:

1) **Phase 1 (no overrides):** Do not allow `prefer_exact` / strategy override to avoid surprises. SearchStrategy is derived strictly from the registered embedding + persisted build config.

2) **Phase 2 (validated overrides):** Allow `prefer_exact` only after build-affecting configs are persisted and drift checks are implemented. Overrides must be validated against stored spec/graph metadata.

3) **Persist build config:** Store HNSW + RaBitQ build parameters (bits_per_dim, rotation_seed, etc.) so SearchStrategy construction and validation use the same source of truth as index build.

4) **Drift mitigation:** Store `spec_hash` in GraphMeta at build time. On search, compare registry spec hash to graph spec hash and error/warn on mismatch (require rebuild).

This keeps the search API consistent with how the index was built while allowing explicit, validated overrides later.

### Resolution

**Alignment: AGREE with all points.** Implemented immediate fix for #1:

1. **SearchConfig vs registry validation - FIXED**
   ```rust
   // 2. Validate SearchConfig embedding matches registry spec
   let config_embedding = config.embedding();
   if config_embedding.dim() != spec.dim() {
       return Err(anyhow!("SearchConfig embedding dimension mismatch..."));
   }
   if config_embedding.distance() != spec.distance() {
       return Err(anyhow!("SearchConfig embedding distance mismatch..."));
   }
   ```

2. **Persist per-embedding configs** - Agree. Deferred to future phase (requires GraphMeta schema changes).

3. **Cache warmup** - Agree. Currently undocumented gap. Options:
   - Prewarm from BinaryCodes on startup
   - Detect low cache coverage and warn/fallback
   - For now: document behavior

4. **RaBitQ fallback documentation** - Agree. `use_cache: false` falls back to Exact (documented in code comment).

**Phase 2 workflow consistency** - Aligned with proposed approach:
- No overrides in Phase 1 (current)
- Validated overrides after config persistence (future)
- Drift mitigation via spec_hash (future)

---

## Task 5.4d: Build Config Persistence (COMPLETE)

**Note:** This task was added via CODEX feedback to ensure build parameters are persisted and validated.

### Proposed Design

### Problem

Current implementation has global HNSW/RaBitQ configs in `Processor`. If configs change between index build and search, behavior is undefined.

### Design Principle

| Store | Purpose | Contents |
|-------|---------|----------|
| **EmbeddingSpec** | HOW to build/search | All configuration for the embedding space |
| **GraphMeta** | Runtime state | Graph state + drift detection hash |

EmbeddingSpec is the single source of truth for build configuration. GraphMeta only stores a hash to detect if the spec changed after build.

### Schema Changes

#### EmbeddingSpec (Extended)

```rust
// libs/db/src/vector/schema.rs

pub struct EmbeddingSpec {
    // Existing fields
    pub model: String,
    pub dim: u32,
    pub distance: Distance,
    pub storage_type: VectorElementType,

    // NEW: HNSW build parameters
    pub hnsw_m: u16,              // M parameter (default: 16)
    pub hnsw_ef_construction: u16, // ef_construction (default: 200)

    // NEW: RaBitQ parameters
    pub rabitq_bits: u8,          // bits_per_dim (default: 1)
    pub rabitq_seed: u64,         // rotation_seed (default: 42)
}
```

#### GraphMeta (Add SpecHash)

```rust
// libs/db/src/vector/schema.rs

pub enum GraphMetaField {
    // Existing
    EntryPoint(VecId),
    MaxLevel(u8),
    Count(u32),
    Config(Vec<u8>),  // deprecated, use EmbeddingSpec

    // NEW: Hash of EmbeddingSpec at build time
    SpecHash(u64),
}
```

### Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Register Embedding                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   EmbeddingBuilder::new("model", 768, Cosine)              │
│       .with_hnsw_m(16)                                      │
│       .with_hnsw_ef_construction(200)                       │
│       .with_rabitq_bits(1)                                  │
│       .with_rabitq_seed(42)                                 │
│   → Persists complete EmbeddingSpec to EmbeddingSpecs CF    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. First Insert (Index Build Start)                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   processor.insert_vector(embedding, id, vector, true)      │
│       → Compute spec_hash = hash(EmbeddingSpec)             │
│       → Store GraphMeta::SpecHash(spec_hash)                │
│       → Build index using EmbeddingSpec params              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Search (Drift Detection)                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   processor.search_with_config(config, query)               │
│       → Load stored_hash from GraphMeta::SpecHash           │
│       → Compute current_hash = hash(registry EmbeddingSpec) │
│       → If stored_hash != current_hash:                     │
│           ERROR: "Spec changed since build - rebuild index" │
│       → Else: proceed with search                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Hash Computation

```rust
fn compute_spec_hash(spec: &EmbeddingSpec) -> u64 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    spec.dim.hash(&mut hasher);
    spec.distance.hash(&mut hasher);
    spec.storage_type.hash(&mut hasher);
    spec.hnsw_m.hash(&mut hasher);
    spec.hnsw_ef_construction.hash(&mut hasher);
    spec.rabitq_bits.hash(&mut hasher);
    spec.rabitq_seed.hash(&mut hasher);
    hasher.finish()
}
```

### Migration Path

1. **Backward compatibility:** Existing EmbeddingSpecs without new fields use defaults
2. **No GraphMeta::SpecHash:** Skip drift check (legacy indexes)
3. **Future:** Require SpecHash for all new indexes

### Cache Warmup (Separate Concern)

BinaryCodeCache warmup is orthogonal to config persistence:

| Option | Implementation |
|--------|----------------|
| Lazy load | `ensure_cache_warmed(embedding)` before RaBitQ search |
| Startup prewarm | Iterate BinaryCodes CF on `Processor::new()` |
| Background | Spawn async task after startup |

Recommend: Lazy load with metrics for Phase 5, prewarm for Phase 6.

---

## CODEX Review (Build Config Persistence Plan)

I agree with the overall plan: EmbeddingSpec as source of truth, GraphMeta holding a spec hash for drift detection. A few improvements to make it robust:

1) **Hash must include model + distance + storage_type + HNSW + RaBitQ.**
   - The proposed hash includes dim/distance/storage and HNSW/RaBitQ; also include `model` to avoid mismatches when two models share dim/distance.

2) **SpecHash should be written exactly once (first index build).**
   - Enforce “first insert sets spec_hash, subsequent inserts must match.” This prevents partial drift if a user updates EmbeddingSpec mid-build.

3) **SearchConfig should be validated against registry spec and stored hash.**
   - On search, compare config.embedding() to registry spec *and* compare registry spec hash to GraphMeta::SpecHash. Reject on mismatch to force rebuild.

4) **Migration path should include a warning when no SpecHash exists.**
   - Legacy indexes without SpecHash should log a warning (or metric) that drift checks are skipped.

These are minor tweaks; the plan is sound.

### Resolution

**Aligned with all four points.** Updated plan:

1. **Hash includes model** - Updated `compute_spec_hash()`:
   ```rust
   fn compute_spec_hash(spec: &EmbeddingSpec) -> u64 {
       let mut hasher = DefaultHasher::new();
       spec.model.hash(&mut hasher);  // Added
       spec.dim.hash(&mut hasher);
       spec.distance.hash(&mut hasher);
       spec.storage_type.hash(&mut hasher);
       spec.hnsw_m.hash(&mut hasher);
       spec.hnsw_ef_construction.hash(&mut hasher);
       spec.rabitq_bits.hash(&mut hasher);
       spec.rabitq_seed.hash(&mut hasher);
       hasher.finish()
   }
   ```

2. **SpecHash immutable after first write** - In `insert_vector()`:
   ```rust
   // On first insert: store spec_hash
   if graph_meta.spec_hash.is_none() {
       let hash = compute_spec_hash(&spec);
       txn.put_cf(&cf, GraphMeta::spec_hash_key(embedding), hash.to_le_bytes())?;
   } else {
       // On subsequent inserts: validate hash matches
       let stored_hash = graph_meta.spec_hash.unwrap();
       let current_hash = compute_spec_hash(&spec);
       if stored_hash != current_hash {
           return Err(anyhow!("EmbeddingSpec changed since index build - rebuild required"));
       }
   }
   ```

3. **Double validation in search** - In `search_with_config()`:
   ```rust
   // Validate config vs registry spec
   validate_config_vs_spec(config, &registry_spec)?;

   // Validate registry spec hash vs stored graph hash
   if let Some(stored_hash) = graph_meta.spec_hash {
       let registry_hash = compute_spec_hash(&registry_spec);
       if stored_hash != registry_hash {
           return Err(anyhow!("Registry spec changed since build - rebuild required"));
       }
   } else {
       log::warn!("Legacy index without SpecHash - drift check skipped");
   }
   ```

4. **Legacy warning** - Implemented via `log::warn!` as shown above.

Implementation will proceed with schema changes to EmbeddingSpec and GraphMeta.

---

## CODEX Review: Workflow Consistency and Config Drift (post-75930c9)

This pass checks the workflow: **register embedding → build index (insert) → search** and whether build/search parameters remain consistent.

**What looks good now**

- **Spec hash drift detection is in place.** `EmbeddingSpec::compute_spec_hash()` includes model + build params, stored on first insert, validated on later inserts and searches. This is the right guardrail for spec changes once data exists.
- **SearchConfig validation is stronger.** `search_with_config()` now validates embedding code + dimension + distance against registry, and checks `SpecHash` vs `EmbeddingSpecs` in RocksDB. This addresses stale/forged configs and post-build drift.

**Gaps: build parameters are persisted but not actually used**

Right now, the build/search runtime still uses **global configs**, not the persisted `EmbeddingSpec` fields:

- `Processor::get_or_create_index()` builds HNSW with `self.hnsw_config` only. It ignores `EmbeddingSpec.hnsw_m` and `EmbeddingSpec.hnsw_ef_construction`.
- `Processor::get_or_create_encoder()` builds RaBitQ with `self.rabitq_config` only. It ignores `EmbeddingSpec.rabitq_bits` and `EmbeddingSpec.rabitq_seed`.

That means the **SpecHash can match while the index/encoder is built with different parameters** (i.e., config drift can still happen, just not *spec* drift). This is the core correctness issue for the workflow: the persisted spec is not yet the source of truth for build/search behavior.

**Implications**

- A cluster-wide config change (HNSW/RaBitQ) silently changes behavior while SpecHash remains valid.
- Search may run with a `SearchStrategy` consistent with the embedding but inconsistent with the actual index or binary codes on disk.
- Future migrations or rebuilds cannot be reliably compared to stored `SpecHash` if the runtime ignores it.

**Proposed fix (Phase 5 follow-up)**

1) **Use EmbeddingSpec to build HNSW and RaBitQ.**
   - When creating HNSW `Config`, override `m` and `ef_construction` from `EmbeddingSpec`.
   - When creating RaBitQ encoder, construct from `EmbeddingSpec.rabitq_bits` + `rabitq_seed` (and make `rabitq_config.enabled` gating explicit).

2) **Add API surface to set build params at registration time.**
   - `EmbeddingBuilder` should accept `hnsw_m`, `hnsw_ef_construction`, `rabitq_bits`, `rabitq_seed`.
   - `EmbeddingRegistry::register()` should persist those values instead of hard-coded defaults (currently `16/200/1/42`).
   - Without this, **SpecHash locks in defaults** on first insert and users cannot tune before building index.

3) **SpecHash timing: consider storing on first index build, not first insert.**
   - Today: SpecHash is set on first insert even if `build_index=false`.
   - Consequence: users cannot adjust build params after inserting raw vectors but before building HNSW.
   - Option A (strict): keep current behavior (spec immutable once data exists). If so, document it.
   - Option B (flex): set SpecHash only when HNSW is built the first time; allow spec tweaks before the first build.

4) **Validate storage_type too.**
   - `search_with_config()` checks dim + distance but not `storage_type`.
   - If `Embedding` in SearchConfig differs (e.g., stale cache), we should reject to avoid searching with the wrong representation.

**API design suggestion (search constrains to build spec)**

Phase 1 (safe, no surprises):
```rust
// New constructor that pulls spec from registry and derives strategy.
let config = SearchConfig::from_registry(&registry, embedding_code, k)
    .with_ef(200)
    .with_rerank_factor(4);
// Strategy is auto-selected and cannot be overridden here.
```

Phase 2 (allow overrides with validation):
```rust
let config = SearchConfig::from_registry(&registry, embedding_code, k)
    .prefer_exact() // Allowed, but validated against EmbeddingSpec + SpecHash
    .with_ef(200);
```

This keeps **SearchStrategy constrained by how the index was built**, while still allowing safe tuning.

**Bottom line**

Task 5.4d added the necessary **persisted spec + drift detection**, but **the runtime still ignores the spec** for index/encoder construction. This is the next correctness gap to close before the workflow can be considered fully consistent.

---

## Task 5.4e: Runtime Uses EmbeddingSpec (COMPLETE)

**Date:** January 14, 2026

### Problem Statement

Task 5.4d established config persistence and drift detection via `SpecHash`, but the runtime was **not using** the persisted build parameters:

```rust
// BEFORE: Ignored EmbeddingSpec build params
fn get_or_create_index(&self, embedding: EmbeddingCode) -> Option<hnsw::Index> {
    // Uses self.hnsw_config (from Processor constructor), NOT EmbeddingSpec
    let config = self.hnsw_config.clone();  // BUG: ignores persisted params
    hnsw::Index::with_storage_type(embedding, distance, storage_type, config, ...)
}

fn get_or_create_encoder(&self, embedding: EmbeddingCode) -> Option<Arc<RaBitQ>> {
    // Uses self.rabitq_config, NOT EmbeddingSpec
    RaBitQ::from_config(dim, &self.rabitq_config)  // BUG: ignores persisted params
}
```

**Impact:** SpecHash could match while index/encoder was built with different parameters, causing silent config drift.

### Solution: Read EmbeddingSpec from Storage

#### 5.7a: HNSW Index Uses EmbeddingSpec

Updated `get_or_create_index()` to read persisted build params:

```rust
fn get_or_create_index(&self, embedding: EmbeddingCode) -> Option<hnsw::Index> {
    // Read EmbeddingSpec from RocksDB
    let spec = self.get_embedding_spec(embedding)?;

    // Build HNSW config using EmbeddingSpec params (not self.hnsw_config)
    let m = spec.hnsw_m as usize;
    let config = hnsw::Config {
        enabled: self.hnsw_config.enabled,  // Runtime control
        dim,
        m,
        m_max: 2 * m,
        m_max_0: 2 * m,
        ef_construction: spec.hnsw_ef_construction as usize,
        m_l: 1.0 / (m as f32).ln(),
        max_level: self.hnsw_config.max_level,      // Runtime control
        batch_threshold: self.hnsw_config.batch_threshold,  // Runtime control
    };

    hnsw::Index::with_storage_type(embedding, distance, storage_type, config, ...)
}
```

#### 5.7b: RaBitQ Encoder Uses EmbeddingSpec

Updated `get_or_create_encoder()` to read persisted build params:

```rust
fn get_or_create_encoder(&self, embedding: EmbeddingCode) -> Option<Arc<RaBitQ>> {
    // Read EmbeddingSpec from RocksDB
    let spec = self.get_embedding_spec(embedding)?;

    // Create encoder using EmbeddingSpec params (not self.rabitq_config)
    RaBitQ::new(dim, spec.rabitq_bits, spec.rabitq_seed)
}
```

#### 5.7c: EmbeddingBuilder API for Build Params

Added builder methods for configuring build params at registration time:

```rust
let embedding = EmbeddingBuilder::new("gemma", 768, Distance::Cosine)
    .with_hnsw_m(32)              // Custom M (default: 16)
    .with_hnsw_ef_construction(400) // Custom ef_construction (default: 200)
    .with_rabitq_bits(2)           // Custom bits (default: 1)
    .with_rabitq_seed(12345)       // Custom seed (default: 42)
    .register(&registry)?;
```

These values are persisted to `EmbeddingSpecs` CF and used for all subsequent index/encoder creation.

#### 5.7d: storage_type Validation in Search

Added explicit storage_type validation in `search_with_config()`:

```rust
if config_embedding.storage_type() != spec.storage_type() {
    return Err(anyhow::anyhow!(
        "SearchConfig embedding storage_type mismatch: config has {:?}, registry has {:?}",
        config_embedding.storage_type(),
        spec.storage_type()
    ));
}
```

### SpecHash Timing: Strict Behavior

The SpecHash is stored on **first insert** and validated on all subsequent operations:

**On Insert:**
1. First insert → compute SpecHash, store in `GraphMeta::SpecHash`
2. Subsequent inserts → compute SpecHash, compare to stored, error on mismatch

**On Search:**
1. Compute SpecHash from current `EmbeddingSpec`
2. Compare to stored `GraphMeta::SpecHash`
3. Error on mismatch: "Rebuild required"
4. Warn on missing: "Legacy index without SpecHash"

**Implications:**
- Build params are **immutable after first insert**
- Changing `hnsw_m`, `rabitq_bits`, etc. after data exists requires rebuild
- This prevents "partial drift" where some vectors use old params and some use new

**Example Error:**
```
EmbeddingSpec changed since index build (hash 12345678 != 87654321). Rebuild required.
```

### Runtime vs Build Params

| Parameter | Source | Can Change After Build? |
|-----------|--------|-------------------------|
| `hnsw_m` | EmbeddingSpec | **No** - SpecHash |
| `hnsw_ef_construction` | EmbeddingSpec | **No** - SpecHash |
| `rabitq_bits` | EmbeddingSpec | **No** - SpecHash |
| `rabitq_seed` | EmbeddingSpec | **No** - SpecHash |
| `enabled` | Processor config | Yes - runtime toggle |
| `max_level` | Processor config | Yes - runtime override |
| `batch_threshold` | Processor config | Yes - runtime tuning |
| `ef_search` | SearchConfig | Yes - query-time param |

### Tests

Added tests in `processor.rs`:
- `test_spec_hash_stored_on_first_insert` - verifies hash storage
- `test_spec_hash_validated_on_subsequent_insert` - verifies drift detection
- `test_search_validates_spec_hash` - verifies search validation

Added tests in `embedding.rs`:
- `test_embedding_builder_with_build_params` - verifies builder API
- `test_embedding_builder_invalid_rabitq_bits` - verifies validation

---

## Task 5.4f: CODEX Review - Runtime EmbeddingSpec Assessment

**Overall:** This change closes the main correctness gap I flagged: runtime HNSW/RaBitQ construction now uses persisted `EmbeddingSpec` parameters, and `search_with_config()` validates `storage_type`. Good alignment with the register → build → search workflow and SpecHash drift detection.

**Remaining issues / improvements**

1) **HNSW M validation (correctness).**  
   `get_or_create_index()` computes `m_l = 1.0 / ln(m)`. If `hnsw_m <= 1`, `ln(m)` is `0` or `-inf`, yielding `inf`/`-0` and likely unstable layer assignment.  
   - **Fix:** validate `hnsw_m >= 2` at registration (prefer `Result`), or clamp/guard before computing `m_l`.  
   - Suggest using `hnsw::Config::is_valid()` plus a dedicated error for `m <= 1`.

2) **Builder validation uses `assert!` (API behavior).**  
   `EmbeddingBuilder::with_rabitq_bits()` panics on invalid bits. That’s consistent with `RaBitQ::new()` but is surprising for a public API; prefer returning `Result<Self>` or a `try_with_*` variant.  
   - If keeping panics, document it clearly in API.md and RABITQ.md.

3) **SpecHash timing is strict (documented).**  
   The new doc section confirms SpecHash is set on first insert even if `build_index=false`. That prevents tuning build params after inserting raw vectors.  
   - This is acceptable if intended; just ensure API docs explicitly call it out (users may expect to tune HNSW/RaBitQ before first build).

**Quality assessment**

This is a strong improvement: persisted build params now drive runtime behavior, closing a correctness hole and aligning the API with the declared workflow. Once HNSW M validation is enforced (and builder panics are either documented or converted to `Result`), the design is sound.

---

## Task 5.4g: CODEX Review - Validation Assessment

**Overall:** This addresses the two issues I raised:
- `hnsw_m >= 2` validation prevents invalid `m_l = 1/ln(m)` computation.
- `with_rabitq_bits()` no longer panics; validation moved to `EmbeddingBuilder::validate()` and invoked during registration.
- API.md now documents SpecHash timing and the immutability expectation.

**Remaining issues / improvements**

1) **Validation happens before idempotent fast-path.**  
   `EmbeddingRegistry::register()` now calls `builder.validate()` before checking if the spec already exists. Because the registry’s identity key is `(model, dim, distance)` only, a caller can supply invalid build params but still expect to retrieve the existing embedding.  
   - **Risk:** This breaks the “idempotent” expectation and makes read-only calls fail.  
   - **Fix:** Check `by_spec` first; if found, return existing embedding without validating build params (or validate only if caller explicitly requests a new spec with build params).

2) **Missing validation for `hnsw_ef_construction` and `rabitq_seed` ranges.**  
   - `hnsw_ef_construction` can be set to `0` (u16) and will produce an invalid HNSW config (`Config::is_valid()` requires > 0).  
   - **Fix:** enforce `hnsw_ef_construction >= 1` in `EmbeddingBuilder::validate()`.

**Quality assessment**

The changes align well with the workflow and close the key correctness hole. Once validation order is adjusted to preserve idempotent registration behavior, and `hnsw_ef_construction` is validated, this should be robust.

---

**Update (post-task numbering alignment / commit 8f93965):**

- **Idempotent registration preserved.** Validation now runs only on new registrations; existing embeddings return without failing on invalid builder params.
- **`hnsw_ef_construction` validated.** Builder rejects values `< 1`, preventing invalid HNSW configs.

No further issues from this review pass.

---

## Task 5.5: Processor::add_embedding_spec() (COMPLETE)

**Status:** ✅ Complete (via `EmbeddingRegistry::register()`)

Per ROADMAP.md, this task implements embedding spec registration. This functionality is provided via `EmbeddingRegistry::register()` which:
- Validates build parameters
- Persists to `EmbeddingSpecs` CF
- Returns registered `Embedding`

See `registry.rs` for implementation.

---

## Task 5.6: Mutation Dispatch (NOT STARTED)

**Status:** 🔲 Not Started

Per ROADMAP.md, this task connects `Mutation` enum to `Processor` methods:
- `Mutation::AddEmbeddingSpec` → `processor.add_embedding_spec()`
- `Mutation::InsertVector` → `processor.insert_vector()`
- `Mutation::DeleteVector` → `processor.delete_vector()`
- etc.

See ROADMAP.md Task 5.6 for full specification.

---

## Task 5.7: Query Dispatch (NOT STARTED)

**Status:** 🔲 Not Started

Per ROADMAP.md, this task connects `Query` enum to `Storage` methods:
- `Query::SearchKNN` → `storage.search()`
- `Query::GetVector` → `storage.get_vector()`
- etc.

See ROADMAP.md Task 5.7 for full specification.

---

## Remaining Phase 5 Tasks (Aligned with ROADMAP.md)

| Task | Description | Status |
|------|-------------|--------|
| 5.0 | HNSW Transaction Refactoring | ✅ **Complete** |
| 5.1 | Processor::insert_vector() | ✅ **Complete** |
| 5.2 | Processor::delete_vector() | ✅ **Complete** |
| 5.3 | Processor::insert_batch() | ✅ **Complete** |
| 5.4 | Processor::search() + Storage layer | ✅ **Complete** (5.4a-5.4g) |
| 5.5 | Processor::add_embedding_spec() | ✅ **Complete** (via registry) |
| 5.6 | Mutation Dispatch | 🔲 Not Started |
| 5.7 | Query Dispatch | 🔲 Not Started |
| 5.8 | Migrate Examples/Integration Tests | 🔲 Not Started |
| 5.9 | Multi-Threaded Stress Tests | 🔲 Not Started |
| 5.10 | Metrics Collection Infrastructure | 🔲 Not Started |
| 5.11 | Concurrent Benchmark Baseline | 🔲 Not Started |

---

## References

- [ROADMAP.md](./ROADMAP.md) - Full implementation roadmap
- [API.md](./API.md) - Public API reference
- [BENCHMARK.md](./BENCHMARK.md) - Performance results
