# Comment for Issue #19: Degree Queries

> **Request: Escalate to P0/Critical Priority**

## Vector Search Use Case

I've been implementing HNSW and Vamana vector search on top of motlie_db and discovered that degree queries are **critical** for index construction performance.

### The Problem

During HNSW insert, we must check if each neighbor exceeds M_max connections before pruning:

```rust
// Current implementation (examples/vector/hnsw.rs)
async fn prune_connections(&self, ...) -> Result<()> {
    // FETCHES ALL EDGES just to count them
    let neighbors = OutgoingEdges::new(node_id, Some(&layer_prefix))
        .run(&reader, timeout).await?;

    // O(degree) work just to get a count
    if neighbors.len() <= m_max {
        return Ok(());
    }

    // ... actual pruning logic
}
```

Every insert operation triggers **~64 prune checks** (bidirectional edges to ~32 neighbors), each requiring full edge enumeration.

### Quantified Impact

With HNSW parameters M=16, M_max0=32:

| Metric | Current (O(degree)) | With #19 (O(1)) | Improvement |
|--------|---------------------|-----------------|-------------|
| Prune check time | ~2ms per check | ~0.1ms | **95% reduction** |
| Checks per insert | ~64 | ~64 | Same count |
| Index build (10K) | 329s | ~280s | **15% faster** |

### Benchmark Evidence

From [PERF.md](./examples/vector/PERF.md), per-insert breakdown:

| Phase | Time | % of Total |
|-------|------|------------|
| Sleep delay | 10ms | 78% |
| Greedy search | 1.5ms | 12% |
| **Edge mutations** | **1.0ms** | **8%** |
| Vector storage | 0.3ms | 2% |

The O(degree) prune checks are hidden in "Edge mutations" but dominate the non-sleep compute time.

### Why This is Critical

Once the read-after-write issue is resolved (eliminating the 10ms sleep), prune overhead becomes the **primary bottleneck**. The 1ms currently masked by the 10ms sleep will become 10% of total insert time.

### Suggested Priority: P0

This feature directly enables:
- 15% faster HNSW index construction
- Foundation for O(1) pruning decisions
- Efficient graph density calculations during construction

Happy to provide more detailed benchmarks if helpful!
