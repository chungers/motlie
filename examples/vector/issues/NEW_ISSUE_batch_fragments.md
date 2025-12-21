# feat(unified-api): Add NodeFragmentsByIdsMulti for batch fragment retrieval

## Summary

Add a batch query for retrieving node fragments from multiple nodes in a single RocksDB operation, similar to the completed `NodesByIdsMulti` (#16). This is critical for vector search where distance computation requires fetching multiple vectors efficiently.

## Problem

During vector search (HNSW/Vamana greedy traversal), we need to compute distances to multiple candidate vectors. Currently this requires N sequential database reads:

```rust
// Current pattern - N sequential reads
for candidate_id in candidates {
    // Each call is a separate RocksDB prefix scan
    let fragments = NodeFragmentsByIdTimeRange::new(candidate_id, time_range, ref_ts)
        .run(&reader, timeout).await?;

    if let Some((_, content)) = fragments.first() {
        let vector = parse_vector(content)?;
        let distance = compute_distance(query, &vector);
    }
}
```

With ef_search=50 (typical HNSW parameter), this means 50 sequential RocksDB reads per search query.

## Proposed Solution

Add `NodeFragmentsByIdsMulti` that batch-retrieves fragments using RocksDB's `multi_get_cf()`:

```rust
/// Batch query for retrieving node fragments from multiple nodes
pub struct NodeFragmentsByIdsMulti {
    /// Node IDs to retrieve fragments for
    node_ids: Vec<Id>,
    /// Time range for fragment filtering
    time_range: TimeRange,
    /// Reference timestamp for "as of" queries
    reference_ts_millis: Option<TimestampMilli>,
}

impl NodeFragmentsByIdsMulti {
    pub fn new(
        node_ids: Vec<Id>,
        time_range: TimeRange,
        reference_ts_millis: Option<TimestampMilli>,
    ) -> Self {
        Self { node_ids, time_range, reference_ts_millis }
    }
}
```

### Output Type

```rust
// Returns mapping from node ID to its fragments
type Output = HashMap<Id, Vec<(TimestampMilli, FragmentContent)>>;
```

Nodes without fragments are omitted from the result (consistent with `NodesByIdsMulti`).

## Expected Performance Impact

Based on the pattern established by `NodesByIdsMulti` (#16):

| Algorithm | Current | With Batch | Improvement |
|-----------|---------|------------|-------------|
| Vector search (ef=50) | 50 × ~0.5ms = 25ms | 1 × ~3ms | **8x** |
| HNSW search latency | 8.29ms | ~3ms | **2.8x** |
| HNSW QPS | 120 | ~330 | **2.8x** |
| Vamana search latency | 4.21ms | ~2ms | **2x** |
| Vamana QPS | 237 | ~500 | **2x** |

### Vector Search Benchmark Context

From [PERF.md](./PERF.md), current HNSW search performance:

| Metric | Value |
|--------|-------|
| Average Search Time | 8.29 ms |
| P50 Latency | 7.44 ms |
| P99 Latency | 12.03 ms |
| QPS | 120.6 |

A 2.8x improvement would bring HNSW to ~330 QPS and ~3ms average latency.

## Implementation

### RocksDB multi_get Pattern

```rust
impl QueryExecutor for NodeFragmentsByIdsMulti {
    type Output = HashMap<Id, Vec<(TimestampMilli, FragmentContent)>>;

    fn execute(&self, storage: &Storage, _timeout: Duration) -> Result<Self::Output> {
        let db = storage.db()?;
        let cf = db.cf_handle(NodeFragments::CF_NAME)?;

        // Build keys for all requested nodes
        let keys: Vec<Vec<u8>> = self.node_ids.iter()
            .map(|id| NodeFragments::key_prefix(id))
            .collect();

        // Batch fetch using RocksDB multi_get
        let results = db.multi_get_cf(
            keys.iter().map(|k| (&cf, k.as_slice()))
        );

        // Parse and collect results
        let mut output = HashMap::new();
        for (id, result) in self.node_ids.iter().zip(results.into_iter()) {
            if let Ok(Some(value)) = result {
                let fragments = parse_fragments(&value, &self.time_range)?;
                if !fragments.is_empty() {
                    output.insert(*id, fragments);
                }
            }
        }

        Ok(output)
    }
}
```

### Integration with Existing API

Follow the pattern from `NodesByIdsMulti`:

```rust
// Example usage in vector search
let candidate_ids: Vec<Id> = candidates.iter().map(|c| c.id).collect();

let fragments = NodeFragmentsByIdsMulti::new(
    candidate_ids,
    TimeRange::all(),
    Some(reference_ts),
)
.run(&reader, timeout).await?;

// Compute distances using batch-fetched vectors
for (id, frags) in fragments {
    if let Some((_, content)) = frags.first() {
        let vector = parse_vector(content)?;
        distances.insert(id, compute_distance(query, &vector));
    }
}
```

## Use Cases

1. **Vector Search**: Batch fetch candidate vectors during greedy traversal
2. **Graph Algorithms**: Retrieve node attributes for multiple nodes (PageRank, community detection)
3. **Export/Backup**: Efficiently dump node data in batches
4. **Validation**: Compare fragments across multiple nodes

## Implementation Checklist

- [ ] Add `NodeFragmentsByIdsMulti` struct in `query.rs`
- [ ] Implement `QueryExecutor` trait
- [ ] Add to unified query enum
- [ ] Add unit tests with multiple nodes
- [ ] Add integration tests with vector data
- [ ] Benchmark against sequential reads
- [ ] Update documentation
- [ ] Add example usage

## Labels

- `enhancement`
- `p0` (high priority)

## Related Issues

- #16 NodesByIdsMulti (completed) - similar pattern for nodes
- #17 OutgoingEdgesMulti - batch edge retrieval
- #18 IncomingEdgesMulti - batch reverse edge retrieval

## Priority

**High** - Directly impacts vector search latency by 2.8x.
