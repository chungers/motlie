# Vector Search Performance: GitHub Issue Analysis

This document analyzes existing GitHub issues against vector search performance requirements
and evaluates solutions for critical bottlenecks.

---

## Executive Summary

| Category | Existing Issue | Status | Vector Search Priority |
|----------|----------------|--------|------------------------|
| **Degree Queries** | #19 | Open (Medium) | **Critical** - Prune decisions |
| **Batch Edge Traversal** | #17 | Open (P0) | High - Search performance |
| **Edge Name Filtering** | #21 | Open (Medium) | High - Layer queries |
| **Batch Reverse Traversal** | #18 | Open (Low-Medium) | Medium |
| **Graph Stats** | #20 | Open (Low) | Low |
| **Read-After-Write** | **NONE** | **MISSING** | **CRITICAL - 10ms bottleneck** |
| **Batch Fragment Retrieval** | **NONE** | **MISSING** | High - Vector loading |

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

## Part 2: Missing Issue - Read-After-Write Consistency

### The 10ms Bottleneck Problem

**No existing GitHub issue covers this critical requirement.**

#### Root Cause Analysis

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

#### Current Workaround

```rust
// examples/vector/hnsw.rs:625
tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
```

This sleep accounts for **78% of indexing time**.

#### Impact on Vector Search

| Metric | Current (with sleep) | Target (no sleep) | Improvement |
|--------|---------------------|-------------------|-------------|
| Max throughput | 100 inserts/sec | 10,000+ inserts/sec | **100x** |
| Actual throughput | 30-68 inserts/sec | 5,000-10,000/sec | **150x** |
| Index 1M vectors | ~9 hours | ~2 minutes | **270x** |

---

## Part 3: Solutions for Read-After-Write Consistency

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

### Option B: Writer Flush API

**Concept**: Add `flush()` method that waits for all pending mutations to complete.

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

### Option C: Transaction Scope API (Read-Your-Writes)

**Concept**: Expose RocksDB Transaction directly to callers for read-your-writes.

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

## Recommended Solution: Combination Approach

### Phase 1: Quick Win (Option B - Flush API)

**Timeline**: 3-5 days
**Impact**: Eliminates 10ms sleep for batch operations

```rust
// Usage pattern for batch indexing
for batch in vectors.chunks(100) {
    for vector in batch {
        insert(writer, reader, vector).await?;
    }
    writer.flush().await?;  // Wait for batch to commit
}
```

This reduces sleep overhead from 100×10ms = 1000ms to 1×flush = ~50ms per batch.
**Improvement: 20x throughput for batch operations.**

### Phase 2: Full Solution (Option C - Transaction Scope)

**Timeline**: 1-2 weeks
**Impact**: True read-your-writes, eliminates all visibility delays

```rust
// Usage pattern for HNSW insert
let txn = writer.begin_transaction();

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

**Improvement: Theoretical 100x throughput (limited only by RocksDB I/O).**

---

## Part 4: Missing Issue - Batch Fragment Retrieval

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

## Summary: Recommended Actions

### Immediate (This Sprint)

1. **Create new issue**: Read-After-Write Consistency (Flush API + Transaction Scope)
   - Priority: P0/Critical
   - Blocks 100x throughput improvement

2. **Comment on #19** (Degree Queries): Add vector search evidence, request P0 priority

3. **Comment on #17** (OutgoingEdgesMulti): Add vector search as use case

4. **Create new issue**: NodeFragmentsByIdsMulti (Batch Fragment Retrieval)
   - Priority: High
   - Enables 2.8x search speedup

### Short-term (Next Sprint)

5. **Implement Flush API** (Phase 1 of read-after-write solution)
   - 3-5 days effort
   - 20x batch throughput improvement

6. **Comment on #21** (Edge Name Filtering): Add vector search use case

### Medium-term (Following Sprints)

7. **Implement Transaction Scope API** (Phase 2 of read-after-write solution)
   - 1-2 weeks effort
   - 100x theoretical throughput

8. **Implement #19** (Degree Queries)
   - Unlocks O(1) prune decisions

---

## Appendix: Cost-Benefit Matrix

| Feature | Effort | Throughput Gain | Search Latency | Priority |
|---------|--------|-----------------|----------------|----------|
| Flush API | Low | 20x (batch) | - | P0 |
| Transaction Scope | High | 100x | - | P0 |
| Degree Queries (#19) | Medium | 15% | - | P0 |
| Batch Edges (#17) | Medium | - | 2-4x | P0 |
| Edge Filtering (#21) | Low | - | 40-85% per query | High |
| Batch Fragments | Medium | - | 2.8x | High |

---

*Generated: 2025-12-21*
*Based on: HNSW/Vamana benchmarks, RocksDB Transaction documentation, motlie_db architecture analysis*
