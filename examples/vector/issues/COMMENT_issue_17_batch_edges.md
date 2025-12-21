# Comment for Issue #17: OutgoingEdgesMulti

> **Additional Use Case: Vector Search (HNSW/Vamana)**

## Vector Search Context

Implemented HNSW and Vamana vector search on motlie_db. This feature would significantly improve search performance.

### Current Pattern in Greedy Search

```rust
// examples/vector/hnsw.rs - search_layer()
async fn search_layer(&self, ...) -> Result<Vec<(f32, Id)>> {
    while let Some(candidate) = candidates.pop() {
        // Sequential edge fetch for each candidate
        let neighbors = OutgoingEdges::new(candidate.id, Some(&layer_prefix))
            .run(&reader, timeout).await?;

        for (_, neighbor_id, weight) in neighbors {
            // Process each neighbor...
        }
    }
}
```

With `ef_search=50` (typical), we fetch edges for ~50 candidates per search.

### Expected Impact on Vector Search

| Metric | Current | With #17 | Improvement |
|--------|---------|----------|-------------|
| Edge fetches per search | 50 sequential | 1 batch | **50x fewer round-trips** |
| HNSW search latency | 8.29ms | ~4ms | **2x** |
| HNSW QPS | 120 | ~240 | **2x** |

### Synergy with Other Features

This would combine well with:
- **NodeFragmentsByIdsMulti** (proposed): Batch vector loading
- **Edge name filtering** (#21): Fetch only layer-specific edges

Together, these could reduce HNSW search latency from 8ms to ~2ms (4x improvement).

### Current Benchmark Numbers

From [PERF.md](./examples/vector/PERF.md):

| Algorithm | Latency (avg) | P50 | P99 | QPS |
|-----------|---------------|-----|-----|-----|
| HNSW | 8.29ms | 7.44ms | 12.03ms | 120.6 |
| Vamana | 4.21ms | 3.45ms | 6.36ms | 237.3 |

Batch edge traversal would significantly improve both algorithms.

Confirming P0 priority is appropriate for this feature!
