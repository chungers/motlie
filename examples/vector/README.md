# Vector Search Examples

This directory contains implementations of Approximate Nearest Neighbor (ANN) search algorithms using `motlie_db` as the underlying graph storage.

## Overview

We implement two popular ANN algorithms:
- **HNSW** (Hierarchical Navigable Small World) - Multi-layer graph for fast approximate search
- **Vamana** (DiskANN) - Single-layer graph optimized for disk-based search

Both algorithms leverage `motlie_db`'s graph primitives (nodes, edges, weights, temporal ranges) to build and query vector indices.

## Design: Mapping Vector Indices to Graph Structures

### Nodes = Vectors

Each vector in the dataset is stored as a **Node**:
- `Node.id`: Unique identifier (UUID)
- `Node.name`: Human-readable identifier (e.g., "vec_0", "vec_1", ...)
- `Node.summary`: Metadata (optional)
- **NodeFragment**: Stores the actual vector data as a serialized `Vec<f32>` (1024 dimensions)

```
NodeFragment Content Format:
- Data URL with MessagePack-encoded Vec<f32>
- 1024 dimensions, each value in [0.0, 1.0)
- Quantized to 1000 intervals (0.000, 0.001, 0.002, ..., 0.999)
```

### Edges = Graph Connections

Graph connections between vectors are stored as **Edges**:
- `Edge.source_node_id`: Source vector ID
- `Edge.target_node_id`: Target vector (neighbor) ID
- `Edge.name`: Encodes the **layer** information (for HNSW) or connection type
- `Edge.weight`: Stores the **distance** between vectors (for greedy search)
- `Edge.valid_range`: Used for **soft-deletion/pruning** (temporal visibility)

### Layer Encoding (HNSW)

HNSW uses multiple layers where higher layers have fewer nodes but longer-range connections.
We encode layers in edge names:

```
Edge naming convention:
- "hnsw_L0": Layer 0 (base layer, all nodes)
- "hnsw_L1": Layer 1 (subset of nodes)
- "hnsw_L2": Layer 2 (smaller subset)
- ...
- "hnsw_L{max_level}": Top layer (entry point)
```

### Edge Pruning via Temporal Ranges

When edges are pruned during index construction (e.g., RNG-style pruning in Vamana),
we use **temporal ranges** to "hide" edges rather than delete them:

```rust
// Prune an edge by setting valid_until to current time
UpdateEdgeValidSinceUntil {
    src_id: source,
    dst_id: target,
    name: edge_name,
    temporal_range: TemporalRange::valid_until(TimestampMilli::now()),
    reason: "pruned".to_string(),
}
```

Querying with a future timestamp sees all edges; querying with past timestamp
sees only non-pruned edges. This enables:
- **Index versioning**: Query the index at different points in time
- **Rollback**: Restore pruned edges by adjusting query timestamp
- **Debugging**: Analyze pruning decisions

## Synthetic Test Data

We generate synthetic vectors with the following properties:
- **Dimensions**: 1024
- **Value range**: [0.0, 1.0)
- **Quantization**: 1000 intervals (values: 0.000, 0.001, ..., 0.999)
- **Distribution**: Uniform random sampling

```rust
// Generate a single dimension value
fn generate_dimension_value(rng: &mut impl Rng) -> f32 {
    let interval = rng.gen_range(0..1000);
    interval as f32 / 1000.0
}

// Generate a 1024-dimensional vector
fn generate_vector(rng: &mut impl Rng) -> Vec<f32> {
    (0..1024).map(|_| generate_dimension_value(rng)).collect()
}
```

## HNSW Algorithm

### Structure

```
Layer L (top):     [entry_point] ─────────────────────► [node_x]
                        │
Layer L-1:         [entry_point] ───► [node_a] ───► [node_x] ───► [node_y]
                        │                 │             │
Layer 0 (base):    [all nodes connected with M neighbors each]
```

### Key Parameters
- `M`: Max connections per node per layer (default: 16)
- `M_max0`: Max connections in layer 0 (default: 2*M = 32)
- `ef_construction`: Search width during construction (default: 200)
- `ef_search`: Search width during query (default: 50)
- `m_L`: Level generation factor (default: 1/ln(M))

### Construction Algorithm

```
1. For each new vector v:
   a. Determine level L = floor(-ln(random()) * m_L)
   b. Find entry point ep at top layer
   c. Greedy search from ep down to layer L+1
   d. For each layer l from L down to 0:
      - Search for ef_construction nearest neighbors
      - Select M best neighbors using heuristic
      - Add bidirectional edges (prune if exceeds M_max)
```

### Search Algorithm

```
1. Start at entry point in top layer
2. Greedy descent through layers until layer 0
3. At layer 0, expand search to ef_search candidates
4. Return top-k nearest neighbors
```

## Vamana (DiskANN) Algorithm

### Structure

Single-layer graph with:
- **Navigability**: Any node reachable from any other in ~O(log n) hops
- **Low out-degree**: Bounded by R (max neighbors)
- **RNG pruning**: Removes redundant edges

```
     [node_a] ───────────────► [node_b]
        │ ╲                      │
        │  ╲                     │
        ▼   ╲                    ▼
     [node_c] ──────────────► [node_d]
```

### Key Parameters
- `R`: Max out-degree (default: 64)
- `L`: Search list size during construction (default: 100)
- `alpha`: RNG pruning threshold (default: 1.2)

### Construction Algorithm

```
1. Initialize: Random graph or empty
2. For each vector v (potentially multiple passes):
   a. Greedy search from medoid to find L candidates
   b. RNG-prune candidates to get R neighbors
   c. Add edges from v to pruned neighbors
   d. For each neighbor n in pruned set:
      - Add reverse edge n → v
      - If n exceeds R neighbors, prune n's edges
```

### RNG Pruning (Robust Neighborhood Graph)

Prune edge (v → u) if there exists w such that:
```
dist(v, w) < dist(v, u)  AND  dist(w, u) < dist(v, u) * alpha
```

This removes "redundant" long edges when a shorter path exists.

### Search Algorithm

```
1. Start at medoid (centroid node)
2. Maintain candidate list of size L
3. Greedily expand closest unvisited candidate
4. Continue until no improvement
5. Return top-k from candidates
```

## Distance Metrics

We support:
- **Euclidean (L2)**: `sqrt(sum((a[i] - b[i])^2))`
- **Cosine**: `1 - dot(a, b) / (norm(a) * norm(b))`
- **Inner Product**: `-dot(a, b)` (negated for min-heap)

Default: **Euclidean distance**

## File Structure

```
examples/vector/
├── README.md           # This file
├── common.rs           # Shared utilities (data generation, distance functions)
├── hnsw.rs             # HNSW implementation
└── vamana.rs           # Vamana (DiskANN) implementation
```

## Usage

```bash
# Build the examples
cargo build --release --examples

# Run HNSW example
./target/release/examples/hnsw <db_path> <num_vectors> <num_queries> <k>

# Run Vamana example
./target/release/examples/vamana <db_path> <num_vectors> <num_queries> <k>
```

## Current Limitations & Known Issues

### Recall Issue

The current implementation has a **recall of 0.0** during search. Root cause:

1. **Vector Cache Not Populated During Search**: The `compute_distance()` method relies on an in-memory `VectorCache` that only contains vectors added during index construction. When searching, we encounter neighbor node IDs from the database, but their vectors aren't in cache.

2. **`f32::MAX` Returned for Missing Vectors**: When a vector isn't in cache, `compute_distance()` returns `f32::MAX`, effectively breaking greedy search since all neighbors appear infinitely far away.

3. **No Async Vector Loading**: The distance computation is synchronous but vector retrieval from RocksDB requires async I/O.

### Fix Required

```rust
// Current (broken):
fn compute_distance(&self, query: &[f32], node_id: &Id) -> f32 {
    match self.cache.get(node_id) {
        Some(vector) => (self.distance_fn)(query, vector),
        None => f32::MAX,  // ← This breaks search!
    }
}

// Fix option 1: Pre-load all vectors into cache before search
// Fix option 2: Make compute_distance async and load on-demand
// Fix option 3: Store vectors inline in edge weights (but 1024 dims is too large)
```

---

## Required motlie_db Optimizations

The following optimizations would make `motlie_db` more suitable for high-performance vector search:

### 1. Batch Vector Retrieval API

**Current State**: `NodeFragments` query retrieves fragments one node at a time.

**Needed**: A `NodeFragmentsByIdsMulti` query that batch-retrieves fragments for multiple nodes in a single RocksDB `multi_get` call.

```rust
// Proposed API
let fragments = NodeFragmentsByIdsMulti::new(
    vec![id1, id2, id3, ...],
    time_range,
    reference_ts,
)
.run(reader, timeout)
.await?;
// Returns: HashMap<Id, Vec<(TimestampMilli, FragmentContent)>>
```

**Why**: During graph traversal, we need to compute distances to many neighbors. Batch retrieval reduces RocksDB round-trips from O(n) to O(1).

### 2. Inline Vector Storage in Edges (Small Vectors)

**Current State**: Edge values store `(TemporalRange, Option<f64>, EdgeSummary)`.

**Needed**: Option to store small payload data directly in edges, avoiding the fragment lookup.

```rust
// Proposed: EdgeFragment or inline payload
AddEdge {
    source_node_id,
    target_node_id,
    name: "hnsw_L0",
    weight: Some(distance),
    payload: Some(target_vector_bytes),  // New: inline small data
    ...
}
```

**Why**: For small vectors or quantized representations (e.g., PQ codes), storing data inline avoids a separate lookup. Not practical for 1024-dim f32 vectors (4KB each), but useful for:
- Product Quantization (PQ) codes: 32-128 bytes
- Scalar Quantization (SQ): 256-1024 bytes
- Graph metadata

### 3. Efficient Edge Enumeration by Prefix

**Current State**: `OutgoingEdges` returns all edges from a node, filtered by temporal range.

**Needed**: Prefix filtering by edge name pattern for layer-specific queries.

```rust
// Proposed API
OutgoingEdges::new(node_id, None)
    .with_name_prefix("hnsw_L0")  // Only layer 0 edges
    .run(reader, timeout)
    .await?;
```

**Why**: HNSW stores edges for multiple layers. Querying layer 0 shouldn't scan layer 1, 2, etc. edges.

### 4. Vector-Specific Column Family

**Current State**: Vectors stored as NodeFragments (text-oriented, fulltext-indexed).

**Needed**: A dedicated column family optimized for binary vector data:
- Skip fulltext indexing (vectors aren't searchable text)
- Optimized compression (vectors compress differently than text)
- Optional: SIMD-friendly memory layout

```rust
// Proposed: Dedicated vector storage
AddNodeVector {
    id: node_id,
    vector: Vec<f32>,  // Or &[u8] for quantized
    dimensions: 1024,
    dtype: VectorDType::F32,
}
```

### 5. Approximate Distance in Edge Weight

**Current State**: Edge weight stores exact distance as `f64`.

**Consideration**: For ANN, we often only need approximate distances for ranking. Could store:
- Quantized distance (u16 or u8)
- Distance bucket/tier

**Trade-off**: Saves space but loses precision. May affect recall for borderline neighbors.

### 6. Connection Count Tracking

**Current State**: To check if a node exceeds `M_max` connections, we must enumerate all edges and count.

**Needed**: Atomic connection count per node, updated on edge add/prune.

```rust
// Proposed: Maintain count in node metadata
struct NodeMetadata {
    connection_counts: HashMap<String, usize>,  // layer_name -> count
}

// Or: Separate counter column family
// Key: (node_id, layer_name)
// Value: count
```

**Why**: Pruning decisions require knowing current neighbor count. Counting via iteration is O(degree).

### 7. Bidirectional Edge Atomicity

**Current State**: Adding a bidirectional edge requires two separate mutations.

**Needed**: Atomic bidirectional edge insertion.

```rust
// Proposed
AddBidirectionalEdge {
    node_a: id1,
    node_b: id2,
    name: "hnsw_L0",
    weight: Some(distance),
}
// Atomically creates: id1 → id2 AND id2 → id1
```

**Why**: Graph consistency. If one direction fails, we have a broken graph.

### 8. Lazy/Streaming Fragment Content

**Current State**: `NodeFragments` query loads entire fragment content into memory.

**Needed**: Streaming or lazy loading for large fragments.

```rust
// For 1024-dim f32 = 4KB per vector
// 1M vectors = 4GB just for vector data
// Need to avoid loading all into memory

// Proposed: Streaming iterator
let stream = NodeFragmentsStream::new(node_id, time_range)
    .run(reader, timeout)
    .await?;

while let Some((ts, content)) = stream.next().await {
    // Process incrementally
}
```

---

## Performance Comparison: Current vs Optimized

| Operation | Current | With Optimizations |
|-----------|---------|-------------------|
| Load vector for 1 node | 1 RocksDB read | 1 RocksDB read |
| Load vectors for N nodes | N RocksDB reads | 1 batch read |
| Check neighbor count | O(degree) scan | O(1) counter lookup |
| Add bidirectional edge | 2 mutations | 1 atomic mutation |
| Query layer-specific edges | Scan all + filter | Prefix scan |
| Store 1M vectors | Fulltext indexed | Skip fulltext |

---

## Recommended Implementation Priority

1. **Batch Vector Retrieval** (High) - Biggest impact on search performance
2. **Connection Count Tracking** (High) - Required for correct pruning
3. **Edge Name Prefix Filtering** (Medium) - Useful for HNSW layers
4. **Skip Fulltext for Vectors** (Medium) - Reduces indexing overhead
5. **Bidirectional Edge Atomicity** (Medium) - Graph consistency
6. **Inline Edge Payload** (Low) - Only for quantized vectors
7. **Streaming Fragments** (Low) - Only for very large datasets

---

## References

- HNSW: [Malkov & Yashunin, 2018](https://arxiv.org/abs/1603.09320)
- DiskANN/Vamana: [Subramanya et al., 2019](https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html)
