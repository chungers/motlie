# feat(unified-api): Add read-after-write consistency for streaming mutations

## Summary

Add mechanisms to ensure mutations are immediately visible to subsequent reads within the same logical operation. This is **critical** for online index construction algorithms (HNSW, Vamana) that require reading recently-written edges during the same insert operation.

## Problem

The current MPSC-based writer architecture decouples mutation sends from RocksDB commits:

```
writer.send(mutations) ───► MPSC Channel ───► Background Consumer ───► RocksDB Commit
        │                                              │
        │ Returns immediately                          │ Async commit
        │                                              │
        └──── reader.query() ──────────────────────────┼─► May see stale data!
```

This creates a **race condition** where reads may not see recently-written data.

### Current Workaround

Vector search examples require a 10ms sleep between operations:

```rust
// examples/vector/hnsw.rs:625
tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
```

### Impact

| Metric | Current (with 10ms sleep) | Target (no sleep) | Improvement |
|--------|--------------------------|-------------------|-------------|
| Max insert throughput | 100/sec | 10,000+/sec | **100x** |
| Actual HNSW throughput | 30-68/sec | 5,000-10,000/sec | **150x** |
| Time to index 1M vectors | ~9 hours | ~2 minutes | **270x** |
| Sleep as % of insert time | **78%** | 0% | Eliminated |

This is the **single largest bottleneck** for vector search performance.

## Proposed Solutions

### Phase 1: Flush API (Quick Win)

Add a `flush()` method to wait for pending mutations to complete:

```rust
impl Writer {
    /// Wait for all pending mutations to be processed and committed.
    /// Returns when all mutations sent before this call are visible to readers.
    pub async fn flush(&self) -> Result<()>;
}
```

**Usage for batch operations:**
```rust
for batch in vectors.chunks(100) {
    for vector in batch {
        insert(writer, reader, vector).await?;
    }
    writer.flush().await?;  // Wait for batch to commit, then continue
}
```

**Expected improvement**: 20x throughput for batch operations (100×10ms → 1×flush).

### Phase 2: Transaction Scope API (Full Solution)

Expose RocksDB Transaction for true read-your-writes semantics:

```rust
impl Writer {
    /// Begin a transaction scope for read-your-writes semantics.
    /// All writes within the scope are immediately visible to reads within the same scope.
    pub fn begin_transaction(&self) -> TransactionScope;
}

impl TransactionScope {
    /// Write mutation (immediately visible to read() within this scope)
    pub fn write(&self, mutation: impl Into<Mutation>) -> Result<()>;

    /// Read with visibility of uncommitted writes in this scope
    pub fn read<Q: QueryExecutor>(&self, query: Q) -> Result<Q::Output>;

    /// Commit all changes atomically
    pub fn commit(self) -> Result<()>;
}
```

**Usage for HNSW insert:**
```rust
let txn = writer.begin_transaction();

// Write node
txn.write(AddNode { id, name, ... })?;
txn.write(AddNodeFragment { id, content: vector, ... })?;

// Greedy search sees uncommitted writes
let neighbors = greedy_search(&txn, query_vector, ef).await?;

// Add edges (also immediately visible within txn)
for (dist, neighbor_id) in neighbors {
    txn.write(AddEdge { src: id, dst: neighbor_id, ... })?;
    txn.write(AddEdge { src: neighbor_id, dst: id, ... })?;
}

txn.commit()?;  // Single atomic commit
```

**Expected improvement**: 100x theoretical throughput (limited only by RocksDB I/O).

## Technical Analysis

### RocksDB Transaction Support

RocksDB's `Transaction` natively supports read-your-writes:

> "This function will also read pending changes in this transaction."
> — [rust-rocksdb Transaction::get documentation](https://docs.rs/rocksdb/latest/rocksdb/struct.Transaction.html)

The current motlie_db architecture already uses `TransactionDB`:

```rust
// libs/db/src/graph/mod.rs:22
use rocksdb::{Options, TransactionDB, TransactionDBOptions, DB};
```

And mutations execute against transactions:

```rust
// libs/db/src/graph/writer.rs:31-35
fn execute(
    &self,
    txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
    txn_db: &rocksdb::TransactionDB,
) -> Result<()>;
```

The issue is that the **transaction scope is internal to the consumer**, not exposed to callers.

### Alternative: WriteBatchWithIndex

RocksDB's [WriteBatchWithIndex](https://github.com/facebook/rocksdb/wiki/Write-Batch-With-Index) provides read-your-writes for batched operations. However, rust-rocksdb may not expose this API directly.

## Implementation Considerations

### Phase 1 (Flush API)

1. Add a "flush marker" mutation type that carries a oneshot channel
2. When consumer processes the marker, send completion signal
3. `flush()` waits on the oneshot receiver

**Effort**: 3-5 days
**Risk**: Low (additive change)

### Phase 2 (Transaction Scope)

1. Add `TransactionScope` struct wrapping `rocksdb::Transaction`
2. Implement `write()` that calls `mutation.execute()` directly
3. Add `QueryExecutor::execute_with_txn()` trait method for transaction-aware reads
4. Integrate with fulltext pipeline (send to consumer after commit)

**Effort**: 1-2 weeks
**Risk**: Medium (new abstraction, requires query trait changes)

## Use Cases

1. **Vector Search (HNSW/Vamana)**: Insert + immediate neighbor lookup
2. **Graph Algorithms**: Create node + create edges in single operation
3. **Atomic Multi-Entity Updates**: Update node + related edges atomically
4. **Transactional Writes**: Rollback on error

## Labels

- `enhancement`
- `p0` (critical priority)
- `performance`

## Related Issues

- #17 OutgoingEdgesMulti - would benefit from transaction scope
- #19 Degree Queries - prune decisions within transaction

## References

- [RocksDB Transactions Wiki](https://github.com/facebook/rocksdb/wiki/Transactions)
- [WriteBatchWithIndex Blog](https://rocksdb.org/blog/2015/02/27/write-batch-with-index.html)
- [rust-rocksdb Transaction](https://docs.rs/rocksdb/latest/rocksdb/struct.Transaction.html)
- [PERF.md Bottleneck Analysis](./PERF.md#bottleneck-1-10ms-sleep-dominant-factor)
