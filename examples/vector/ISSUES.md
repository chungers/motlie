# Vector Search Performance: GitHub Issue Analysis

This document analyzes existing GitHub issues against vector search performance requirements
and evaluates solutions for critical bottlenecks.

**Last Updated**: 2025-12-24

---

## Executive Summary

| Category | Existing Issue | Status | Vector Search Priority |
|----------|----------------|--------|------------------------|
| **Degree Queries** | #19 | Open (Medium) | **Critical** - Prune decisions |
| **Batch Edge Traversal** | #17 | Open (P0) | High - Search performance |
| **Edge Name Filtering** | #21 | Open (Medium) | High - Layer queries |
| **Batch Reverse Traversal** | #18 | Open (Low-Medium) | Medium |
| **Graph Stats** | #20 | Open (Low) | Low |
| **Read-After-Write** | #26 | ✅ **IMPLEMENTED** | Flush API + Transaction API |
| **Batch Fragment Retrieval** | **#35** | Open (High) | High - Vector loading |

---

## Part 1: Existing Issues - Vector Search Evidence

### Issue #19: Degree Queries (OutgoingEdgeCount, IncomingEdgeCount)

**Current Priority**: Medium
**Recommended Priority**: **Critical (P0)**

#### Vector Search Use Case

During HNSW/Vamana index construction, every insert operation must check if neighbors
exceed the maximum connection limit (M_max):

```rust
// Current implementation (examples/vector/hnsw.rs:380-395)
async fn prune_connections(&self, ...) -> Result<()> {
    // PROBLEM: Fetches ALL edges just to count them
    let neighbors = get_neighbors(reader, node_id, Some(&layer_prefix)).await?;

    // O(degree) work just to get a number
    if neighbors.len() <= m_max {
        return Ok(());  // No pruning needed
    }

    // ... prune edges
}
```

#### Quantified Impact

| Metric | Current | With #19 | Improvement |
|--------|---------|----------|-------------|
| Prune check time | O(degree) | O(1) | **32x** at M_max0=32 |
| Calls per insert | ~64 (bidirectional) | ~64 | Same count, faster each |
| Time per insert | ~2ms prune overhead | ~0.1ms | **95% reduction** |
| Index build (10K) | 329s | ~280s | **15% faster** |

#### Supporting Evidence from Benchmarks

```
HNSW Parameters: M=16, M_max0=32, ef_construction=200
Per-insert breakdown (from PERF.md):
  - Sleep delay: 10ms (78%)
  - Greedy search: 1.5ms (12%)
  - Edge mutations: 1.0ms (8%)  ← Includes O(degree) prune checks
  - Vector storage: 0.3ms (2%)
```

The O(degree) prune check is hidden in "Edge mutations" but dominates the non-sleep time.

#### Recommendation

Add a comment to issue #19 documenting this use case and request priority escalation to P0.

---

### Issue #17: OutgoingEdgesMulti for Batch Edge Traversal

**Current Priority**: P0 (High)
**Recommended Priority**: P0 (confirmed)

#### Vector Search Use Case

During greedy search, HNSW/Vamana expand multiple candidate nodes simultaneously.
Currently this requires sequential edge fetches:

```rust
// Current search pattern (conceptual)
for candidate in candidates.iter() {
    let neighbors = OutgoingEdges::new(candidate.id, Some(layer_name))
        .run(&reader, timeout).await?;  // N sequential queries

    for (_, neighbor_id, weight) in neighbors {
        // ... process neighbor
    }
}
```

#### Quantified Impact (from issue #17)

| Scale | Current | With #17 | Speedup |
|-------|---------|----------|---------|
| 10 nodes | ~10ms | ~3ms | 3x |
| 100 nodes | ~100ms | ~15ms | 7x |
| 1000 nodes | ~1000ms | ~80ms | 12x |

For vector search with ef_search=50 (typical), batch fetching 50 neighbor sets
per search could reduce search latency from 8ms to ~2ms.

#### Recommendation

Confirm P0 priority. Add vector search as additional use case in the issue.

---

### Issue #21: Edge Name Filtering

**Current Priority**: Medium
**Recommended Priority**: High

#### Vector Search Use Case

HNSW uses multiple layers with different edge names (`hnsw_L0`, `hnsw_L1`, etc.).
During layer-specific traversal, we only need edges for one layer:

```rust
// Current: Fetch all edges, filter in Rust
let all_edges = OutgoingEdges::new(node_id, None).run(&reader, timeout).await?;
let layer_edges: Vec<_> = all_edges
    .into_iter()
    .filter(|(name, _, _)| name == "hnsw_L0")  // Wasteful post-filtering
    .collect();

// With #21: Filter at RocksDB level
let layer_edges = OutgoingEdges::new(node_id, None)
    .with_edge_filter("hnsw_L0")  // Skip deserialization of non-matching edges
    .run(&reader, timeout).await?;
```

#### Quantified Impact

For a node with edges across 3 layers (L0=32, L1=16, L2=8 = 56 total edges):

| Query | Current | With #21 | Improvement |
|-------|---------|----------|-------------|
| Layer 0 only | 56 edge deserializations | 32 | **43% less work** |
| Layer 2 only | 56 edge deserializations | 8 | **86% less work** |

#### Recommendation

Escalate to High priority. Add vector search use case.

---

## Part 2: Read-After-Write Consistency - ✅ IMPLEMENTED

### The 10ms Bottleneck Problem - SOLVED

**Issue #26**: Read-After-Write Consistency - **✅ IMPLEMENTED** (2025-12-21)

Both Phase 1 (Flush API) and Phase 2 (Transaction API) have been implemented:
- `writer.flush().await` - Wait for pending writes to commit
- `writer.send_sync()` - Send + flush in one call
- `writer.transaction()` - Full read-your-writes Transaction API

See [flush.md](../../libs/db/docs/flush.md) and [transaction.md](../../libs/db/docs/transaction.md) for implementation details.

#### Historical Root Cause Analysis

The current architecture:

```
User Code                    MPSC Channel              Background Task
─────────────────────────────────────────────────────────────────────────
writer.send(mutations) ──────► [buffer] ──────► Consumer.run() ──────► RocksDB
    │                                                    │
    │ Returns immediately                                │ Commits async
    │                                                    │
    └──── reader.query() ────────────────────────────────┼─► May see stale data!
                                                         │
                                      ┌──────────────────┘
                                      │ Race condition window
                                      ▼
                              Insert visibility uncertain
```

The HNSW insert operation needs to:
1. Write node + edges to RocksDB
2. **Immediately** read those edges back during greedy search
3. Use the edges to find more neighbors

Without waiting for the write to commit, the read may see stale data.

#### Previous Workaround (REMOVED)

```rust
// OLD - examples/vector/hnsw.rs:625 (no longer used)
// tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

// NEW - Proper API calls:
writer.flush().await?;  // Or use Transaction API for read-your-writes
```

The sleep workaround has been **removed** from all vector search examples.

#### Achieved Results

| Metric | Old (with sleep) | With Flush API | With Transaction API |
|--------|-----------------|----------------|---------------------|
| HNSW 1K throughput | 45.8/s | ~46/s | **103.6/s** (+126%) |
| HNSW 10K throughput | 36.8/s | ~37/s | **68.1/s** (+85%) |
| Correctness | Hopeful | **Guaranteed** | **Guaranteed** |

---

## Part 3: Solutions for Read-After-Write Consistency - HISTORICAL

> **Note**: This section documents the design options that were evaluated. Options B and C have been **implemented**.

### Option A: Synchronous Processing Mode

**Concept**: Bypass MPSC channel, execute mutations synchronously within caller's context.

```rust
// New API
impl Writer {
    /// Process mutations synchronously, blocking until committed
    pub async fn send_sync(&self, mutations: Vec<Mutation>) -> Result<()> {
        let txn = self.storage.transaction_db()?.transaction();
        for mutation in &mutations {
            mutation.execute(&txn, self.storage.transaction_db()?)?;
        }
        txn.commit()?;
        Ok(())
    }
}
```

**Pros**:
- Simple implementation
- Guaranteed immediate visibility
- No architecture change

**Cons**:
- Skips fulltext indexing pipeline
- Caller blocks on RocksDB I/O
- Not composable with existing async consumers

**Effort**: Low (1-2 days)
**Risk**: Medium (bypasses existing pipeline)

---

### Option B: Writer Flush API - ✅ IMPLEMENTED

**Concept**: Add `flush()` method that waits for all pending mutations to complete.

**Status**: ✅ **Implemented** (2025-12-21) - See [flush.md](../../libs/db/docs/flush.md)

```rust
impl Writer {
    /// Wait for all pending mutations to be processed
    pub async fn flush(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        self.sender.send(FlushRequest { completion: tx }).await?;
        rx.await?
    }
}
```

**Implementation**: Insert a "flush marker" mutation that signals completion via oneshot channel.

**Pros**:
- Works with existing architecture
- Non-invasive change
- Composable with batching

**Cons**:
- Still has round-trip latency (send → process → signal)
- Adds channel coordination overhead
- Doesn't help with read-your-writes within transaction

**Effort**: Medium (3-5 days)
**Risk**: Low

---

### Option C: Transaction Scope API (Read-Your-Writes) - ✅ IMPLEMENTED

**Concept**: Expose RocksDB Transaction directly to callers for read-your-writes.

**Status**: ✅ **Implemented** (2025-12-23) - See [transaction.md](../../libs/db/docs/transaction.md)

```rust
// New API
impl Writer {
    /// Begin a transaction for read-your-writes semantics
    pub fn begin_transaction(&self) -> TransactionScope {
        let txn = self.storage.transaction_db().transaction();
        TransactionScope { txn }
    }
}

impl TransactionScope {
    /// Write mutation to transaction (visible via read())
    pub fn write(&self, mutation: Mutation) -> Result<()> {
        mutation.execute(&self.txn, ...)?;
        Ok(())
    }

    /// Read from transaction (sees uncommitted writes)
    pub fn read<Q: QueryExecutor>(&self, query: Q) -> Result<Q::Output> {
        query.execute_with_txn(&self.txn)
    }

    /// Commit all changes atomically
    pub fn commit(self) -> Result<()> {
        self.txn.commit()
    }
}
```

**Pros**:
- True read-your-writes (no latency)
- Leverages RocksDB's native Transaction support
- Zero overhead for visibility

**Cons**:
- Requires new Query trait method (`execute_with_txn`)
- Bypasses MPSC pipeline (must integrate with fulltext later)
- More complex API surface
- Transaction lifetime management

**Effort**: High (1-2 weeks)
**Risk**: Medium (new abstraction)

---

### Option D: WriteBatchWithIndex (Hybrid)

**Concept**: Use RocksDB's WriteBatchWithIndex for in-memory buffering with read support.

```rust
// Conceptual API
impl Writer {
    pub fn begin_batch(&self) -> BatchScope {
        let batch = WriteBatchWithIndex::new();
        BatchScope { batch, storage: self.storage.clone() }
    }
}

impl BatchScope {
    pub fn put(&mut self, mutation: Mutation) -> Result<()> {
        mutation.write_to_batch(&mut self.batch)?;
        Ok(())
    }

    /// Read from batch first, fallback to DB
    pub fn get(&self, key: &[u8], cf: &str) -> Result<Option<Vec<u8>>> {
        self.batch.get_from_batch_and_db(&self.storage.db, cf, key)
    }

    pub fn commit(self) -> Result<()> {
        self.storage.db.write(self.batch.write_batch())?;
        Ok(())
    }
}
```

**Pros**:
- Native RocksDB support for read-your-writes
- Efficient append-only batch structure
- Can integrate with existing pipeline after commit

**Cons**:
- `rust-rocksdb` crate may not expose WriteBatchWithIndex
- Requires checking crate support
- Different API than Transaction

**Effort**: Medium-High (depends on crate support)
**Risk**: Medium (crate dependency)

---

### Option E: Optimistic Writer Overlay

**Concept**: Maintain in-memory overlay of pending writes, merge with reads.

```rust
struct OptimisticWriter {
    pending: Arc<RwLock<HashMap<Key, Value>>>,
    base_writer: Writer,
    base_reader: Reader,
}

impl OptimisticWriter {
    pub async fn write(&self, mutation: Mutation) -> Result<()> {
        // Buffer in memory
        let ops = mutation.to_kv_pairs()?;
        let mut pending = self.pending.write();
        for (key, value) in ops {
            pending.insert(key, value);
        }

        // Send to background pipeline
        self.base_writer.send(vec![mutation]).await
    }

    pub async fn read_with_pending<Q>(&self, query: Q) -> Result<Q::Output> {
        // Merge pending writes with base reader results
        let base_result = query.run(&self.base_reader, timeout).await?;
        let pending = self.pending.read();
        merge_results(base_result, &pending, query.affected_keys())
    }
}
```

**Pros**:
- Works without RocksDB changes
- Composable with existing architecture
- Configurable memory limit

**Cons**:
- Complex merge logic
- Memory overhead for pending writes
- Must handle all query types
- Potential consistency bugs

**Effort**: High (1-2 weeks)
**Risk**: High (complex merge logic)

---

## Recommended Solution: Combination Approach - ✅ IMPLEMENTED

### Phase 1: Quick Win (Option B - Flush API) - ✅ DONE

**Implemented**: 2025-12-21
**Impact**: Proper read-after-write consistency (replaces hopeful 10ms sleep)

```rust
// Usage pattern for batch indexing
for batch in vectors.chunks(100) {
    for vector in batch {
        insert(writer, reader, vector).await?;
    }
    writer.flush().await?;  // Wait for batch to commit
}
```

**Result**: Same throughput (~30-45/s with per-insert flush), but **guaranteed correctness** instead of hopeful sleep.

### Phase 2: Full Solution (Option C - Transaction Scope) - ✅ DONE

**Implemented**: 2025-12-23
**Impact**: True read-your-writes with 2x speedup at small scales

```rust
// Usage pattern for HNSW insert (now used in examples)
let mut txn = writer.transaction()?;

txn.write(AddNode { id, name, ... })?;
txn.write(AddNodeFragment { id, content: vector, ... })?;

// Greedy search within transaction (sees uncommitted writes)
let neighbors = greedy_search_txn(&txn, query_vector, ef).await?;

for (dist, neighbor_id) in neighbors.iter().take(m) {
    txn.write(AddEdge { src: id, dst: *neighbor_id, ... })?;
    txn.write(AddEdge { src: *neighbor_id, dst: id, ... })?;
}

txn.commit()?;  // Single atomic commit
```

**Result**:
- HNSW 1K: **2.26x faster** (103.6 vs 45.8 vec/s)
- HNSW 10K: **1.85x faster** (68.1 vs 36.8 vec/s)

See [PERF.md](./PERF.md#2025-12-24-re-benchmark-transaction-api-performance) for full benchmark results.

---

## Part 4: Issue #35 - Batch Fragment Retrieval

### The Problem

During vector search, we need to compute distances to multiple candidate vectors.
Currently this requires N sequential database reads:

```rust
// Current pattern
for candidate_id in candidates {
    let vector = get_vector(reader, candidate_id).await?;  // N reads
    let distance = compute_distance(query, &vector);
}
```

### Proposed Solution

Similar to #16 (NodesByIds - completed), add `NodeFragmentsByIdsMulti`:

```rust
// Proposed API
let fragments = NodeFragmentsByIdsMulti::new(
    vec![id1, id2, id3, ...],  // Batch of IDs
    time_range,
    reference_ts,
)
.run(reader, timeout).await?;

// Returns: HashMap<Id, Vec<(TimestampMilli, FragmentContent)>>
```

### Expected Impact

| Operation | Current | With Batch API | Improvement |
|-----------|---------|----------------|-------------|
| Fetch 32 vectors | 32 × ~0.5ms = 16ms | 1 × ~2ms = 2ms | **8x** |
| Search latency | 8.29ms | ~3ms | **2.8x** |
| QPS | 120 | ~330 | **2.8x** |

---

## Summary: Status & Remaining Actions

### ✅ Completed

1. **Issue #26**: Read-After-Write Consistency (Flush API + Transaction Scope)
   - ✅ Flush API implemented (2025-12-21)
   - ✅ Transaction API implemented (2025-12-23)
   - ✅ 10ms sleep workaround removed from all examples
   - ✅ Benchmarks validated: 2.26x speedup at 1K, 1.85x at 10K

### 🔶 In Progress / Pending

2. **Comment on #19** (Degree Queries): Add vector search evidence, request P0 priority
   - Still needed for O(1) prune decisions

3. **Comment on #17** (OutgoingEdgesMulti): Add vector search as use case
   - Still needed for batch edge traversal

4. **Issue #35 Created**: NodeFragmentsByIdsMulti (Batch Fragment Retrieval)
   - Priority: High
   - Enables 2.8x search speedup

5. **Comment on #21** (Edge Name Filtering): Add vector search use case

### Future Work

6. **Implement #19** (Degree Queries)
   - Unlocks O(1) prune decisions
   - Critical for >1000 inserts/sec target

---

## Appendix: Cost-Benefit Matrix

| Feature | Effort | Throughput Gain | Search Latency | Priority | Status |
|---------|--------|-----------------|----------------|----------|--------|
| Flush API | Low | - | - | P0 | ✅ Done |
| Transaction Scope | High | 2.26x (1K) | - | P0 | ✅ Done |
| Degree Queries (#19) | Medium | 15% | - | P0 | Pending |
| Batch Edges (#17) | Medium | - | 2-4x | P0 | Pending |
| Edge Filtering (#21) | Low | - | 40-85% per query | High | Pending |
| Batch Fragments (#35) | Medium | - | 2.8x | High | Issue Created |

---

*Originally Generated: 2025-12-21*
*Last Updated: 2026-01-02*
*Based on: HNSW/Vamana benchmarks, RocksDB Transaction documentation, motlie_db architecture analysis*
