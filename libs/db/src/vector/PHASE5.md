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

## Remaining Phase 5 Tasks

| Task | Description | Status |
|------|-------------|--------|
| 5.1 | Processor::insert_vector() | Pending |
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
