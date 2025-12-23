# Comment for Issue #21: Edge Name Filtering

> **Use Case: HNSW Multi-Layer Graph Traversal**

## Vector Search Context

HNSW uses multiple graph layers with different edge names:
- `hnsw_L0` - Base layer (all nodes, max 32 edges each)
- `hnsw_L1` - Layer 1 (subset of nodes, max 16 edges each)
- `hnsw_L2` - Layer 2 (smaller subset)
- etc.

During search, we traverse one layer at a time.

### Current Pattern

```rust
// examples/vector/hnsw.rs
async fn get_neighbors_at_layer(&self, node_id: Id, layer: usize) -> Result<Vec<Edge>> {
    let layer_prefix = format!("hnsw_L{}", layer);

    // Fetches ALL edges, then filters in Rust
    let all_edges = OutgoingEdges::new(node_id, None)
        .run(&reader, timeout).await?;

    // Wasteful post-filtering
    all_edges.into_iter()
        .filter(|(name, _, _)| name == &layer_prefix)
        .collect()
}
```

### Quantified Impact

For a node with edges across 3 HNSW layers (L0=32, L1=16, L2=8 = 56 total):

| Query Target | Current | With #21 | Improvement |
|--------------|---------|----------|-------------|
| Layer 0 only | 56 edge deserializations | 32 | **43% less work** |
| Layer 1 only | 56 edge deserializations | 16 | **71% less work** |
| Layer 2 only | 56 edge deserializations | 8 | **86% less work** |

### Why This Matters for HNSW

HNSW search algorithm:
1. Start at entry point in top layer (L2)
2. Greedy search in L2 (only need L2 edges)
3. Move to L1, greedy search (only need L1 edges)
4. Move to L0, expand search (only need L0 edges)

At each layer transition, we currently fetch **all** edges and discard most of them.

### Suggested Priority: High

This feature:
- Reduces unnecessary deserialization by 40-85%
- Improves HNSW search latency
- Also benefits Vamana (single edge type, but still avoids other edge types if present)

Low implementation effort (key prefix matching) with meaningful performance gains.
