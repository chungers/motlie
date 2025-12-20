# Vector Search Examples

This directory contains implementations of Approximate Nearest Neighbor (ANN) search algorithms using `motlie_db` as the underlying graph storage.

## Overview

We implement two popular ANN algorithms:
- **HNSW** (Hierarchical Navigable Small World) - Multi-layer graph for fast approximate search
- **Vamana** (DiskANN) - Single-layer graph optimized for disk-based search

Both algorithms leverage `motlie_db`'s graph primitives (nodes, edges, weights, temporal ranges) to build and query vector indices.

## Current Status

| Aspect | Status | Notes |
|--------|--------|-------|
| **Index Construction** | ✅ Working | Both HNSW and Vamana build correct graph structures |
| **Graph Storage** | ✅ Working | Nodes, edges, weights, temporal ranges all persist correctly |
| **HNSW Search** | ✅ Working | **100% recall@10** with async distance computation + DB fallback |
| **Vamana Search** | ✅ Working | ~60% recall@10 with async distance computation + DB fallback |
| **Memory Efficiency** | ✅ Working | Async distance loads vectors on-demand from DB |
| **Online Updates (HNSW)** | 🔶 Partial | Incremental insert works but has latency issues |
| **Online Updates (Vamana)** | ❌ Not Supported | Batch-only construction, requires full rebuild |
| **Concurrent Access** | ❌ Not Implemented | No read/write synchronization |
| **Vector Deletion** | ❌ Not Implemented | No delete support in either algorithm |

**Key Finding**: The `motlie_db` Storage API is architecturally suitable for disk-based vector search. Both HNSW and Vamana implementations now use async distance computation with DB fallback.

### Progress Summary

| Phase | Description | Status |
|-------|-------------|--------|
| **Phase 0** | Basic graph storage mapping | ✅ Complete |
| **Phase 1** | Fix recall issue (async distance) | ✅ Complete |
| **Phase 2** | Memory-efficient search | ✅ Complete (basic) |
| **Phase 3** | Online updates support | 🔶 In Progress |
| **Phase 4** | Production optimizations | ❌ Not Started |

### Performance Results

See [PERF.md](./PERF.md) for comprehensive benchmarks.

**Summary at 10K vectors**:

| Algorithm | Index Time | Throughput | Disk | Latency | QPS | Recall@10 |
|-----------|------------|------------|------|---------|-----|-----------|
| HNSW | 329.7s | 30/s | 257MB | 22ms | 45 | 82% |
| Vamana | 225.5s | 44/s | 281MB | 12ms | 82 | 27% |

**Key Bottleneck**: 10ms sleep between inserts limits throughput to ~30-68/sec. See [Why Indexing is Slow](#why-indexing-is-slow) below.

Note: HNSW achieves higher recall due to its hierarchical multi-layer structure with larger ef_construction parameter. Vamana's lower recall on uniform random data is expected - it's optimized for clustered real-world data.

## HNSW vs Vamana Comparison

| Aspect | HNSW | Vamana (DiskANN) |
|--------|------|------------------|
| **Graph Structure** | Multi-layer hierarchical | Single-layer flat |
| **Entry Point** | Top layer node (random level assignment) | Medoid (centroid of dataset) |
| **Layers** | L layers with decreasing node count | 1 layer with all nodes |
| **Edge Naming** | `hnsw_L0`, `hnsw_L1`, ... | `vamana` (single name) |
| **Max Neighbors** | M per layer, M_max0 at layer 0 | R (uniform across all nodes) |
| **Neighbor Selection** | Heuristic: keep M closest | RNG pruning: remove redundant |
| **Construction** | Incremental (one vector at a time) | Batch (multiple passes) |
| **Memory During Build** | Low (cache current neighbors) | Higher (need all vectors for medoid) |
| **Search Complexity** | O(log n) layers + O(M·ef) at L0 | O(log n) hops × O(R) neighbors |
| **Disk Optimization** | Not designed for disk | Designed for SSD (with PQ) |
| **Online Insert** | ✅ Native support | ❌ Requires FreshDiskANN variant |
| **Online Delete** | 🔶 Possible with tombstones | ❌ Complex, affects graph quality |
| **Best For** | In-memory, dynamic updates | Disk-based, static datasets |

### When to Use Which?

**Choose HNSW when:**
- Dataset fits in memory
- Need incremental updates (add/remove vectors)
- Want simpler implementation
- **Require online updates without index rebuild**

**Choose Vamana when:**
- Dataset exceeds memory
- Willing to invest in PQ/compression
- Building static index (batch construction OK)
- Targeting SSD-based deployment
- **Can afford periodic full index rebuilds**

## Design: Mapping Vector Indices to Graph Structures

### Nodes = Vectors

Each vector in the dataset is stored as a **Node**:
- `Node.id`: Unique identifier (UUID)
- `Node.name`: Human-readable identifier (e.g., "vec_0", "vec_1", ...)
- `Node.summary`: Metadata (optional)
- **NodeFragment**: Stores the actual vector data as a serialized `Vec<f32>` (1024 dimensions)

```
NodeFragment Content Format:
- Data URL with JSON-encoded Vec<f32> (JSON used for fulltext indexer UTF-8 compatibility)
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

### Implementation Details (vamana.rs)

#### Build Process Flow

```
Phase 1: Store vectors and compute medoid
├── store_vector() for each (Id, Vec<f32>)
├── cache.insert() for in-memory access
└── compute_medoid() - find node closest to centroid

Phase 2: Build graph (2 passes)
├── initialize_random_edges() - R random edges per node for connectivity
└── For each pass:
    └── For each node (shuffled order):
        ├── greedy_search() → L candidates
        ├── rng_prune() → ≤R neighbors
        ├── add_vector_edge() × 2 (bidirectional)
        └── maybe_prune_node() for each neighbor
```

#### Storage API Usage

| Operation | API Call | Notes |
|-----------|----------|-------|
| Store vector | `AddNode` + `AddNodeFragment` | JSON-encoded Vec<f32> |
| Add edge | `AddEdge` | name="vamana", weight=distance |
| Get neighbors | `OutgoingEdges` | Filters by name="vamana" |
| Prune edge | `UpdateEdgeValidSinceUntil` | Sets valid_until=now |
| Count neighbors | `OutgoingEdges` + `.len()` | O(degree) enumeration |

#### Code Structure

```rust
vamana.rs (648 lines)
├── VamanaParams { r, l, alpha }     // Configuration
├── SearchCandidate                   // Min-heap entry for greedy search
└── VamanaIndex
    ├── build()                       // Batch construction
    │   ├── store vectors + cache
    │   ├── compute_medoid()
    │   ├── initialize_random_edges()
    │   └── 2× insert_node() pass
    ├── insert_node()                 // Per-node graph update
    │   ├── greedy_search()
    │   ├── rng_prune()
    │   └── maybe_prune_node()
    ├── greedy_search()               // Core navigation algorithm
    ├── rng_prune()                   // Redundant edge removal
    ├── compute_medoid()              // Entry point selection
    └── search()                      // Query interface
```

### Comparison to Original DiskANN Paper

The original DiskANN (Subramanya et al., 2019) includes optimizations not yet implemented:

| Feature | DiskANN Paper | Current Implementation |
|---------|--------------|----------------------|
| **Vector Storage** | SSD with sector-aligned layout | RocksDB NodeFragments |
| **Compressed Search** | Product Quantization (PQ) codes | Full f32 vectors |
| **Distance Computation** | PQ fast approximate + rerank | Exact Euclidean only |
| **Beam Search** | W parallel paths | Single greedy path |
| **Graph Layout** | Cache-optimized neighbor ordering | RocksDB edge order |
| **Entry Point** | Medoid with cached neighbors | Medoid only |

#### DiskANN Optimizations for Future Implementation

**1. Product Quantization (PQ) for Fast Approximate Distance**
```
Original 1024-dim f32 (4KB) → 64 subvectors × 8-bit codes = 64 bytes
Speedup: ~50× less memory, ~20× faster distance computation
Trade-off: ~5-10% recall loss, mitigated by reranking
```

**2. Two-Phase Search**
```
Phase 1: Navigate graph using PQ distances (fast, approximate)
Phase 2: Rerank top candidates with full-precision vectors (accurate)
```

**3. Beam Search (W > 1)**
```
Instead of single greedy path, maintain W parallel candidates
W=4 typical for disk-based search
Reduces sensitivity to local minima
```

**4. Sector-Aligned Storage**
```
SSD reads are 4KB aligned - store node + neighbors + vector together
Current: Separate column families require multiple reads
Optimal: Single read per hop
```

### Vamana-Specific Issues in Current Implementation

#### Issue 1: Cache Miss During Search - ✅ FIXED

The original implementation returned `f32::MAX` on cache miss, breaking greedy search.

**Solution implemented**: Async distance computation with DB fallback:
```rust
// vamana.rs - compute_distance_async now loads from DB on cache miss
async fn compute_distance_async(&mut self, reader: &Reader, query: &[f32], node_id: &Id) -> f32 {
    if let Some(vector) = self.cache.get(node_id) {
        return (self.distance_fn)(query, vector);
    }
    // Load from database on cache miss
    match get_vector(reader, *node_id).await {
        Ok(vector) => {
            let dist = (self.distance_fn)(query, &vector);
            self.cache.insert(*node_id, vector);  // Cache for future use
            dist
        }
        Err(_) => f32::MAX,
    }
}
```

**Result**: Recall improved from 0.0 to ~60%.

#### Issue 2: O(degree) Neighbor Count Check

```rust
// vamana.rs:382-394
async fn maybe_prune_node(&self, writer: &Writer, reader: &Reader, node_id: Id) -> Result<()> {
    let neighbors = get_neighbors(reader, node_id, Some(edge_name)).await?;  // ← Fetches ALL edges
    if neighbors.len() <= self.params.r {  // Just need count, got all data
        return Ok(());
    }
    // ... pruning logic
}
```

**Impact**: Every `insert_node()` call triggers neighbor enumeration for each affected node.
**Fix**: Atomic neighbor count stored in node metadata or separate counter CF.

#### Issue 3: Sequential Bidirectional Edge Insertion

```rust
// vamana.rs:258-262
for (dist, neighbor_id) in &pruned {
    add_vector_edge(writer, node_id, *neighbor_id, edge_name, *dist).await?;  // Edge 1
    add_vector_edge(writer, *neighbor_id, node_id, edge_name, *dist).await?;  // Edge 2 (separate call)
    // ...
}
```

**Impact**: 2× mutations per edge, potential inconsistency if second fails.
**Fix**: `AddBidirectionalEdge` mutation for atomic A↔B insertion.

#### Issue 4: No Batch Vector Loading in Pruning

```rust
// vamana.rs:403-409
let candidates: Vec<(f32, Id)> = neighbors
    .iter()
    .map(|(_, id)| {
        let dist = self.compute_distance(&node_vector, id);  // ← Cache lookup, no DB fallback
        (dist, *id)
    })
    .collect();
```

**Impact**: If any neighbor vector isn't cached, distance = f32::MAX, corrupting pruning decisions.
**Fix**: Batch load missing vectors with `NodeFragmentsByIdsMulti` (proposed API).

### Recommended Vamana-Specific Enhancements

#### Phase 1: Fix Core Functionality

| Enhancement | Effort | Impact | Status |
|-------------|--------|--------|--------|
| Async distance with DB fallback | Medium | Fixes recall, on-demand loading | ✅ Done |
| Add neighbor count tracking | Medium | 10× faster pruning decisions | Pending |

#### Phase 2: DiskANN Features

| Enhancement | Effort | Impact |
|-------------|--------|--------|
| Product Quantization | High | 50× less memory for distances |
| Two-phase search (PQ + rerank) | High | True disk-based operation |
| Beam search (W > 1) | Medium | Better recall on sparse graphs |

#### Phase 3: Storage Optimizations

| Enhancement | Effort | Impact |
|-------------|--------|--------|
| Batch fragment retrieval API | Medium | Faster neighbor loading |
| Sector-aligned storage | High | Optimal SSD performance |
| Neighbor prefetching | Medium | Hide I/O latency |

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

### Recall Issue - ✅ FIXED

The original implementation had a recall of 0.0 during search due to cache miss handling.

**Root Cause (Historical)**:
1. `compute_distance()` returned `f32::MAX` when vectors weren't in cache
2. This broke greedy search since all neighbors appeared infinitely far away

**Solution Implemented**: Both HNSW and Vamana now use async distance computation with DB fallback:

```rust
// Fixed implementation in both hnsw.rs and vamana.rs:
async fn compute_distance_async(&mut self, reader: &Reader, query: &[f32], node_id: &Id) -> f32 {
    // Check cache first
    if let Some(vector) = self.cache.get(node_id) {
        return (self.distance_fn)(query, vector);
    }
    // Load from database on cache miss
    match get_vector(reader, *node_id).await {
        Ok(vector) => {
            let dist = (self.distance_fn)(query, &vector);
            self.cache.insert(*node_id, vector);  // Cache for future use
            dist
        }
        Err(_) => f32::MAX,
    }
}
```

**Results After Fix**:
- HNSW: **100% recall@10** (500 vectors, 100 queries)
- Vamana: **61% recall@10** (1000 vectors, 100 queries)

---

## Why Indexing is Slow

The current implementation achieves only 30-68 vectors/sec. Here's the detailed breakdown:

### Bottleneck Analysis

| Bottleneck | Impact | % of Time |
|------------|--------|-----------|
| **10ms sleep** | Limits to 100/sec theoretical max | **78%** |
| **Graph construction** | Greedy search + edge mutations | 12% |
| **RocksDB writes** | 34-81 mutations per vector | 8% |
| **JSON parsing** | 60% overhead vs binary | 2% |

### The 10ms Sleep Problem

```rust
// hnsw.rs:559 - The dominant bottleneck
tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
```

**Why it exists**: RocksDB writes are async. Each insert's greedy search needs to see edges from the previous insert. Without the sleep, read-after-write inconsistency causes graph corruption.

**Solutions** (see [PERF.md](./PERF.md) for details):
1. **Optimistic read-your-writes**: Maintain pending edges in memory overlay
2. **Write batching**: Buffer inserts, batch-flush periodically
3. **RocksDB transactions**: Use WriteBatchWithIndex for immediate visibility

### Write Amplification per Insert

| Mutation | Count (HNSW) | Description |
|----------|--------------|-------------|
| AddNode | 1 | Vector metadata |
| AddNodeFragment | 1 | Vector data (JSON) |
| AddEdge (forward) | ~32 | To neighbors |
| AddEdge (reverse) | ~32 | From neighbors |
| UpdateEdgeValidUntil | 0-16 | Pruning |
| **Total** | **34-81** | Per vector |

### Disk Usage Breakdown

At 10K vectors, disk usage is **25.7KB per vector**:

| Component | Size | % |
|-----------|------|---|
| Vector data (JSON) | 6.5KB | 25% |
| Edges (bidirectional) | 12.8KB | **50%** |
| RocksDB overhead | 6.1KB | 24% |
| Node metadata | 0.3KB | 1% |

**Key insight**: Graph edges consume 2× more space than vector data itself.

### Proposed Optimizations

| Optimization | Improvement | Status |
|--------------|-------------|--------|
| Remove 10ms sleep | **50× throughput** | Requires motlie_db changes |
| Binary vector storage | 40% less space | Ready to implement |
| Adjacency list edges | 50% less space | Requires schema change |
| Edge count tracking | O(1) prune decisions | Requires motlie_db changes |

See [PERF.md](./PERF.md) for the complete optimization roadmap.

---

## Online Updates Requirement

A critical requirement for production vector search is **online updates** - the ability to add, update, or delete vectors without rebuilding the entire index.

### Current Online Update Capabilities

| Capability | HNSW | Vamana | Notes |
|------------|------|--------|-------|
| **Insert** | 🔶 Partial | ❌ No | HNSW supports incremental insert but with latency issues |
| **Delete** | ❌ No | ❌ No | Neither implementation supports deletion |
| **Update** | ❌ No | ❌ No | Would require delete + insert |
| **Concurrent Read/Write** | ❌ No | ❌ No | No synchronization implemented |

### HNSW Online Insert Analysis

HNSW is designed for incremental updates, but the current implementation has issues:

**What Works:**
- Insert one vector at a time via `insert()` method
- Graph structure updated incrementally
- Entry point updated if new node has higher level

**Current Issues:**

1. **Write Visibility Latency** (Critical)
   ```rust
   // hnsw.rs: 10ms sleep between inserts!
   tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
   ```
   - Required because RocksDB writes are async
   - Limits insert throughput to ~100 vectors/sec
   - **Root cause**: No read-after-write consistency guarantee

2. **Pruning Side Effects**
   ```rust
   // Inserting vector A can prune edges of existing vector B
   self.prune_connections(writer, reader, *neighbor_id, l).await?;
   ```
   - Concurrent queries might see inconsistent graph state
   - No transactional boundary for insert + prune operations

3. **Entry Point Race Condition**
   ```rust
   // Entry point can change mid-search
   if level > self.max_level {
       self.entry_point = Some(node_id);  // Not atomic with writes
       self.max_level = level;
   }
   ```

4. **Cache Invalidation**
   - No mechanism to invalidate cached vectors on update/delete
   - Concurrent searches might use stale data

### Vamana Online Insert Limitations

Vamana requires fundamental changes for online updates:

1. **Medoid Dependency**: Entry point is centroid of all vectors
   - Adding new vector shifts centroid
   - Would need periodic medoid recalculation

2. **Batch Construction**: Algorithm makes multiple passes
   - Pass 1: Random edges for connectivity
   - Pass 2+: Refine with RNG pruning
   - Single insert doesn't fit this model

3. **FreshDiskANN Approach** (Not Implemented):
   - Maintain small "fresh" index for new vectors
   - Periodically merge into main index
   - Requires hybrid search across both indices

### Required Changes for Online Updates

#### API Requirements (motlie_db)

| Requirement | Priority | Why |
|-------------|----------|-----|
| **Read-after-write consistency** | Critical | Eliminate sleep delays, immediate edge visibility |
| **Atomic multi-edge transactions** | Critical | Insert + bidirectional edges + prune as single operation |
| **Connection count tracking** | High | O(1) prune decisions, critical for insert performance |
| **Soft-delete for nodes** | High | Vector deletion via temporal range |
| **Batch edge operations** | Medium | Efficient multi-edge insert/prune |

#### Algorithm Requirements

| Requirement | HNSW | Vamana |
|-------------|------|--------|
| **Concurrent read/write** | Add RwLock around entry_point, max_level | N/A (batch only) |
| **Insert transaction** | Group insert + edges + prune atomically | Implement FreshDiskANN |
| **Delete operation** | Soft-delete node + edges via temporal range | Full rebuild required |
| **Cache invalidation** | Subscribe to mutation events | N/A |

### Online Update Performance Targets

| Operation | Target Latency | Target Throughput |
|-----------|---------------|-------------------|
| Insert | < 10ms | > 1000 vectors/sec |
| Delete | < 5ms | > 2000 vectors/sec |
| Search during update | < 20ms (P99) | No degradation |

### Proposed Online Update Architecture

```
                    ┌─────────────────────────────────────┐
                    │         Vector Search Index          │
                    │                                     │
   Insert ─────────►│  ┌─────────┐    ┌──────────────┐  │
                    │  │ Write   │───►│ Transaction  │  │
   Delete ─────────►│  │ Queue   │    │ Processor    │  │
                    │  └─────────┘    └──────────────┘  │
                    │        │                │          │
                    │        ▼                ▼          │
                    │  ┌─────────────────────────┐      │
                    │  │    motlie_db (RocksDB)  │      │
                    │  │  ┌─────┐ ┌─────┐ ┌────┐│      │
                    │  │  │Nodes│ │Edges│ │Cnts││      │
                    │  │  └─────┘ └─────┘ └────┘│      │
                    │  └─────────────────────────┘      │
                    │        │                          │
   Search ─────────►│  ┌─────────────────────────┐      │
                    │  │   Graph Navigator       │      │
                    │  │   (async distance)      │      │
                    │  └─────────────────────────┘      │
                    └─────────────────────────────────────┘
```

---

## Viability Assessment: Memory-Efficient Vector Search

### What the Current API Already Supports

The `motlie_db` Storage API has several features that support memory-efficient patterns:

| Feature | Status | Description |
|---------|--------|-------------|
| **Lazy Loading** | ✅ Viable | Data fetched on-demand via `NodeFragments` query, not pre-loaded |
| **Streaming Iteration** | ✅ Viable | `AllNodeFragments` scan with visitor pattern for incremental processing |
| **Temporal Pruning** | ✅ Viable | Soft deletion via `valid_until` enables versioning without physical deletes |
| **Distance in Edges** | ✅ Viable | Edge weights store distances, avoiding recomputation during traversal |
| **MPMC Query Dispatch** | ✅ Viable | Worker pools handle concurrent queries efficiently |
| **Type-Safe Access Modes** | ✅ Viable | `Storage<ReadOnly>` vs `Storage<ReadWrite>` prevents misuse |

### What's NOT Viable Without API Changes

| Gap | Impact | Current Workaround |
|-----|--------|-------------------|
| **Batch Fragment Retrieval** | High | `get_vectors_batch()` loops N times instead of 1 `multi_get` |
| **Edge Name Prefix Filtering** | Medium | `OutgoingEdges` returns all edges, Rust filters by layer |
| **Connection Count Tracking** | Medium | Must enumerate all edges to count neighbors |
| **Bidirectional Edge Atomicity** | Low | Two separate mutations (potential inconsistency) |

### Fixing the Recall Issue: Practical Options

**Option A: Pre-load All Vectors (Simple, Memory-Heavy)**
```rust
// Before search, load entire database into cache
pub async fn preload_all_vectors(&mut self, reader: &Reader) -> Result<()> {
    let all_nodes = AllNodes::new(None).run(reader, timeout).await?;
    for (id, _name, _summary) in all_nodes {
        let vector = get_vector(reader, id).await?;
        self.cache.insert(id, vector);
    }
    Ok(())
}
```
- Pros: Works now, no API changes needed
- Cons: O(n) memory, defeats disk-based search purpose

**Option B: Async Distance Computation (Complex, Memory-Efficient)**
```rust
// Make compute_distance async with on-demand loading
pub async fn compute_distance_async(&self, reader: &Reader, query: &[f32], node_id: &Id) -> f32 {
    if let Some(vector) = self.cache.get(node_id) {
        return (self.distance_fn)(query, vector);
    }
    // Load from database on cache miss
    match get_vector(reader, node_id).await {
        Ok(vector) => {
            let dist = (self.distance_fn)(query, &vector);
            // Optionally cache for future access
            dist
        }
        Err(_) => f32::MAX,
    }
}
```
- Pros: True disk-based search, minimal memory
- Cons: Requires async propagation through search algorithms, latency per lookup

**Option C: LRU Cache with Async Loading (Balanced)**
```rust
pub struct LruVectorCache {
    cache: LruCache<Id, Vec<f32>>,
    max_size: usize,
}

impl LruVectorCache {
    pub async fn get_or_load(&mut self, reader: &Reader, id: &Id) -> Option<Vec<f32>> {
        if let Some(v) = self.cache.get(id) {
            return Some(v.clone());
        }
        if let Ok(vector) = get_vector(reader, *id).await {
            self.cache.put(*id, vector.clone());
            return Some(vector);
        }
        None
    }
}
```
- Pros: Bounded memory, amortized performance, works with current API
- Cons: Still O(1) DB lookup per cache miss

**Recommended Approach**: Option C (LRU Cache) is the best balance for current API. For true billion-scale search, Option B with batch fragment retrieval API would be optimal.

### Memory Budget Analysis

For a given memory budget, here's what's achievable:

| Memory Budget | Max Vectors (1024-dim f32) | Strategy |
|--------------|---------------------------|----------|
| 1 GB | ~250,000 | Full cache viable |
| 4 GB | ~1,000,000 | Full cache viable |
| 8 GB | ~2,000,000 | Full cache viable |
| 16 GB | ~4,000,000 | Full cache or LRU |
| > 16 GB | 10M+ | LRU + disk required |

For datasets exceeding available memory:
1. Use LRU cache sized to working set (e.g., 1M vectors)
2. Rely on graph structure for navigation (edges stay in RocksDB)
3. Accept ~1 disk read per search hop not in cache

---

## Required motlie_db Optimizations

The following optimizations would make `motlie_db` suitable for production vector search with **online updates**.

### Priority Legend

| Priority | Meaning |
|----------|---------|
| 🔴 **Critical** | Blocking for online updates |
| 🟠 **High** | Significant performance/functionality impact |
| 🟡 **Medium** | Quality of life improvement |
| 🟢 **Low** | Nice to have |

### 1. 🔴 Read-After-Write Consistency

**Current State**: Writer is async; reads may not see recent writes immediately.

**Problem**: HNSW insert requires 10ms sleep between operations:
```rust
// hnsw.rs - current workaround
tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
```

**Needed**: Synchronous write visibility or explicit flush/sync option.

```rust
// Option A: Sync write mode
writer.write_sync(AddEdge { ... }).await?;  // Blocks until visible

// Option B: Explicit flush
writer.flush().await?;  // Ensure all pending writes are visible

// Option C: Read-your-writes guarantee
let reader = writer.reader_with_writes();  // Sees own pending writes
```

**Impact**: Enables >1000 inserts/sec (currently limited to ~100/sec due to sleep).

### 2. 🔴 Connection Count Tracking

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

// API usage
let count = GetConnectionCount::new(node_id, "hnsw_L0")
    .run(reader, timeout).await?;
```

**Impact**: O(1) prune decisions instead of O(degree). Critical for fast online inserts.

### 3. 🔴 Atomic Multi-Edge Transactions

**Current State**: Each edge add/prune is a separate mutation. Insert operation requires multiple mutations that can partially fail.

**Needed**: Transaction support for grouping mutations.

```rust
// Proposed: Transaction API
let txn = writer.begin_transaction();
txn.add(AddNode { ... });
txn.add(AddNodeFragment { ... });
txn.add(AddEdge { source: a, target: b, ... });
txn.add(AddEdge { source: b, target: a, ... });  // Reverse edge
txn.add(UpdateEdgeValidUntil { ... });  // Prune old edge
txn.commit().await?;  // All or nothing
```

**Impact**: Graph consistency during inserts. No partial state visible to queries.

### 4. 🟠 Bidirectional Edge Atomicity

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

**Impact**: Graph consistency. If one direction fails, we have a broken graph.

### 5. 🟠 Batch Vector Retrieval API

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

**Impact**: During graph traversal, batch retrieval reduces RocksDB round-trips from O(n) to O(1).

### 6. 🟠 Soft-Delete for Nodes

**Current State**: No mechanism to mark nodes as deleted while preserving graph structure.

**Needed**: Temporal soft-delete for nodes that hides them from queries.

```rust
// Proposed: Soft-delete node
UpdateNodeValidUntil {
    id: node_id,
    valid_until: TimestampMilli::now(),
    reason: "deleted by user",
}

// Queries with future timestamp see node; past timestamp doesn't
```

**Impact**: Enables vector deletion without full index rebuild.

### 7. 🟡 Edge Name Prefix Filtering

**Current State**: `OutgoingEdges` returns all edges from a node, filtered by temporal range.

**Needed**: Prefix filtering by edge name pattern for layer-specific queries.

```rust
// Proposed API
OutgoingEdges::new(node_id, None)
    .with_name_prefix("hnsw_L0")  // Only layer 0 edges
    .run(reader, timeout)
    .await?;
```

**Impact**: HNSW stores edges for multiple layers. Querying layer 0 shouldn't scan layer 1, 2, etc. edges.

### 8. 🟡 Skip Fulltext Indexing for Vectors

**Current State**: Vectors stored as NodeFragments are fulltext-indexed.

**Needed**: Flag to skip fulltext indexing for binary/vector data.

```rust
// Proposed: Skip fulltext flag
AddNodeFragment {
    id: node_id,
    content: vector_data,
    skip_fulltext: true,  // Don't index as text
    ...
}
```

**Impact**: 40-60% faster vector storage. Vectors aren't searchable text anyway.

### 9. 🟢 Vector-Specific Column Family

**Current State**: Vectors stored as NodeFragments (text-oriented).

**Needed**: A dedicated column family optimized for binary vector data:
- Skip fulltext indexing
- Optimized compression
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

### 10. 🟢 Inline Edge Payload (Small Vectors)

**Current State**: Edge values store `(TemporalRange, Option<f64>, EdgeSummary)`.

**Needed**: Option to store small payload data directly in edges.

```rust
// Proposed: Inline small payload
AddEdge {
    source_node_id,
    target_node_id,
    name: "hnsw_L0",
    weight: Some(distance),
    payload: Some(pq_codes),  // 32-128 bytes for PQ codes
    ...
}
```

**Impact**: Useful for Product Quantization codes (32-128 bytes), not for full vectors (4KB).

### 11. 🟢 Streaming Fragment Content

**Current State**: `NodeFragments` query loads entire fragment content into memory.

**Needed**: Streaming or lazy loading for large fragments.

```rust
// Proposed: Streaming iterator
let stream = NodeFragmentsStream::new(node_id, time_range)
    .run(reader, timeout)
    .await?;

while let Some((ts, content)) = stream.next().await {
    // Process incrementally
}
```

**Impact**: Only needed for extremely large fragments or memory-constrained environments.

---

## Performance Comparison: Current vs Optimized

### Query Performance

| Operation | Current | With Optimizations |
|-----------|---------|-------------------|
| Load vector for 1 node | 1 RocksDB read | 1 RocksDB read |
| Load vectors for N nodes | N RocksDB reads | 1 batch read |
| Query layer-specific edges | Scan all + filter | Prefix scan |
| Search latency (P99) | ~15ms | ~10ms |

### Online Update Performance

| Operation | Current | With Phase 1 | With Phase 3 |
|-----------|---------|--------------|--------------|
| Insert throughput | ~100/sec (10ms sleep) | >1000/sec | >5000/sec |
| Insert latency (P99) | ~15ms | <10ms | <5ms |
| Check neighbor count | O(degree) scan | O(1) counter | O(1) counter |
| Add bidirectional edge | 2 mutations | 1 atomic | 1 atomic |
| Insert atomicity | None | Transaction | Transaction |
| Concurrent readers | Breaks | Works | Snapshot isolated |

### Storage Performance

| Operation | Current | With Optimizations |
|-----------|---------|-------------------|
| Store 1M vectors | Fulltext indexed | Skip fulltext (40-60% faster) |
| Vector storage size | JSON encoded | Binary (50% smaller) |
| PQ codes inline | Not possible | In edge payload |

---

## Recommended Implementation Priority

Priorities are ordered based on enabling **online updates** while maintaining search performance.

### ✅ Phase 0: Foundation (COMPLETED)

| Task | Status | Impact |
|------|--------|--------|
| Basic graph storage mapping | ✅ Done | Enables HNSW/Vamana on motlie_db |
| Fix recall issue (async distance) | ✅ Done | Working search with 60-100% recall |
| Memory-efficient search | ✅ Done | On-demand vector loading from DB |

### 🔶 Phase 1: Online Update Enablers (CRITICAL)

These are **blocking** requirements for production online updates:

| Optimization | Effort | Impact | Why Critical for Online |
|--------------|--------|--------|------------------------|
| **Read-after-write consistency** | High | Critical | Eliminate 10ms sleep delays, enable >1000 inserts/sec |
| **Connection Count Tracking** | Medium | Critical | O(1) prune decisions during insert (currently O(degree)) |
| **Atomic Multi-Edge Transactions** | Medium | Critical | Insert + bidirectional edges + prune atomically |
| **Bidirectional Edge Atomicity** | Low | High | Graph consistency during concurrent operations |

**Target Outcome**: HNSW inserts at >1000 vectors/sec with immediate visibility.

### Phase 2: Online Update Quality

| Optimization | Effort | Impact | Why |
|--------------|--------|--------|-----|
| **Soft-delete for Nodes** | Medium | High | Enable vector deletion via temporal range |
| **Batch Fragment Retrieval** | Medium | High | Fast neighbor loading during insert/search |
| **Cache Invalidation Events** | Medium | High | Notify caches when vectors are updated/deleted |
| **Edge Name Prefix Filtering** | Low | Medium | Faster HNSW layer queries during insert |

**Target Outcome**: Full CRUD operations on vector index.

### Phase 3: Concurrent Access

| Optimization | Effort | Impact | Why |
|--------------|--------|--------|-----|
| **RwLock for Index Metadata** | Low | High | Safe concurrent read during write |
| **Snapshot Isolation for Queries** | Medium | High | Consistent reads during mutations |
| **Write Queue with Batching** | Medium | Medium | Amortize transaction overhead |

**Target Outcome**: No search degradation during concurrent updates.

### Phase 4: Production Optimizations

| Optimization | Effort | Impact | Why |
|--------------|--------|--------|-----|
| **Skip Fulltext for Vectors** | Medium | Medium | 40-60% indexing speedup |
| **Product Quantization** | High | High | 50× less memory, faster distance |
| **Vector-Specific Column Family** | High | Medium | SIMD-aligned storage, binary format |
| **Streaming Fragments** | High | Low | Only for very large vectors |

### Implementation Roadmap Summary

```
Phase 0 ──────────────────────────────────────────────────── ✅ DONE
  └── Basic search working with 60-100% recall

Phase 1 ──────────────────────────────────────────────────── 🔶 NEXT
  └── Online inserts at >1000/sec
      ├── Read-after-write consistency (motlie_db)
      ├── Connection count tracking (motlie_db)
      └── Atomic multi-edge transactions (motlie_db)

Phase 2 ──────────────────────────────────────────────────── Planned
  └── Full CRUD support
      ├── Vector deletion
      ├── Batch operations
      └── Cache invalidation

Phase 3 ──────────────────────────────────────────────────── Planned
  └── Concurrent read/write
      ├── RwLock for metadata
      └── Snapshot isolation

Phase 4 ──────────────────────────────────────────────────── Future
  └── Production scale
      ├── PQ compression
      └── Optimized storage
```

### Key Metrics to Track

| Metric | Current | Phase 1 Target | Phase 3 Target |
|--------|---------|----------------|----------------|
| Insert throughput | ~100/sec (10ms sleep) | >1000/sec | >5000/sec |
| Insert latency (P99) | ~15ms | <10ms | <5ms |
| Search during insert | Untested | <20ms P99 | <15ms P99 |
| Concurrent writers | 1 | 1 | Multiple |
| Delete support | No | No | Yes |

---

## References

- HNSW: [Malkov & Yashunin, 2018](https://arxiv.org/abs/1603.09320)
- DiskANN/Vamana: [Subramanya et al., 2019](https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html)
