# POC: Current Vector Search Implementation

**Phase 1 - Proof of Concept**

This document captures the current implementation of vector search in motlie_db, serving as the foundation for future optimization phases.

**Last Updated**: 2025-12-24

---

## Document Hierarchy

```
REQUIREMENTS.md     ← Ground truth for all design decisions
    ↓
POC.md (this)       ← Phase 1: Current implementation (you are here)
    ↓
HNSW2.md            ← Phase 2: Optimized HNSW with roaring bitmaps
    ↓
IVFPQ.md            ← Phase 3: GPU-accelerated search (optional)
    ↓
HYBRID.md           ← Phase 4: Billion-scale production architecture
```

**Links**: [REQUIREMENTS.md](./REQUIREMENTS.md) → **POC.md** → [HNSW2.md](./HNSW2.md) → [IVFPQ.md](./IVFPQ.md) → [HYBRID.md](./HYBRID.md)

---

## 1. Current Architecture

### 1.1 RocksDB Schema

The current implementation uses **5 column families**:

| Column Family | Key Format | Value Format | Purpose |
|---------------|------------|--------------|---------|
| `nodes` | `[Id:16]` | `(ActivePeriod, Name, Summary)` | Vector node metadata |
| `node_fragments` | `[Id:16][Timestamp:8]` | `(ActivePeriod, Content)` | Vector embeddings (as DataUrl) |
| `forward_edges` | `[SrcId:16][DstId:16][Name:var]` | `(ActivePeriod, Weight, Summary)` | Graph edges (src → dst) |
| `reverse_edges` | `[DstId:16][SrcId:16][Name:var]` | `(ActivePeriod)` | Reverse index (dst → src) |
| `edge_fragments` | `[SrcId:16][DstId:16][Name:var][Ts:8]` | `(ActivePeriod, Content)` | Edge metadata |

**Key Design Decisions:**

1. **16-byte UUIDs** (ULID format) for node IDs
   - Globally unique, timestamp-sortable
   - Trade-off: 4x larger than u32 IDs (see [HNSW2.md](./HNSW2.md) for optimization)

2. **Prefix extractors** for efficient range scans
   - `forward_edges`: 16-byte prefix (all edges from a source)
   - `node_fragments`: 16-byte prefix (all fragments for a node)

3. **Temporal validity** on all records
   - Supports point-in-time queries
   - Enables soft deletes via `valid_until` timestamps

### 1.2 Vector Storage

Vectors are stored as `DataUrl` in `node_fragments`:

```rust
// libs/db/src/graph/mutation.rs:151-163
pub struct AddNodeFragment {
    pub id: Id,                    // Node ID
    pub ts_millis: TimestampMilli, // Timestamp for ordering
    pub content: DataUrl,          // Vector as base64-encoded JSON
    pub valid_range: Option<ActivePeriod>,
}
```

**Current Encoding**: Vectors are JSON-serialized and base64-encoded in DataUrl.

| Metric | Current | Target ([HYBRID.md](./HYBRID.md)) |
|--------|---------|-------------------|
| 128D vector size | ~600 bytes | 8 bytes (PQ compressed) |
| Compression | None | 75x with PQ ([STOR-4](./REQUIREMENTS.md#stor-4)) |

### 1.3 Graph Edge Storage

Edges are stored in both `forward_edges` and `reverse_edges`:

```rust
// libs/db/src/graph/schema.rs:77-85
pub struct ForwardEdgeCfValue(
    pub Option<ActivePeriod>, // Temporal validity
    pub Option<f64>,           // Edge weight (distance)
    pub EdgeSummary,           // Edge metadata
);
```

**HNSW Layer Encoding**: Layers are encoded in edge names (`hnsw_L0`, `hnsw_L1`, etc.)

---

## 2. Consistency APIs

### 2.1 Flush API ([CON-1](./REQUIREMENTS.md#con-1))

**Status**: ✅ Implemented (2025-12-21)

The Flush API provides read-after-write consistency by waiting for pending mutations to commit.

```rust
// libs/db/src/graph/mutation.rs:38-56
pub struct FlushMarker {
    completion: Mutex<Option<oneshot::Sender<()>>>,
}

// Usage:
writer.send(mutations).await?;
writer.flush().await?;  // Blocks until all pending writes are committed
// Now safe to read the written data
```

**Implementation Details:**
1. `flush()` creates a `FlushMarker` with oneshot channel
2. Marker is sent through mutation channel
3. Consumer signals completion after RocksDB commit
4. Caller blocks on oneshot receiver

**Requirements Addressed**: [CON-1](./REQUIREMENTS.md#con-1) (Read-after-write consistency)

### 2.2 Transaction API ([CON-1](./REQUIREMENTS.md#con-1))

**Status**: ✅ Implemented (2025-12-23)

The Transaction API provides true read-your-writes semantics within a single transaction.

```rust
// libs/db/src/graph/transaction.rs:90-103
pub struct Transaction<'a> {
    txn: Option<rocksdb::Transaction<'a, rocksdb::TransactionDB>>,
    txn_db: &'a rocksdb::TransactionDB,
    mutations: Vec<Mutation>,
    forward_to: Option<mpsc::Sender<Vec<Mutation>>>,
}
```

**Usage Pattern (HNSW Insert):**

```rust
let mut txn = writer.transaction()?;

// 1. Write node and vector
txn.write(AddNode { id, name, ... })?;
txn.write(AddNodeFragment { id, content: vector, ... })?;

// 2. Greedy search (sees uncommitted writes!)
let neighbors = greedy_search_txn(&txn, query_vector, ef).await?;

// 3. Add edges based on search results
for (dist, neighbor_id) in neighbors.iter().take(m) {
    txn.write(AddEdge { src: id, dst: *neighbor_id, ... })?;
    txn.write(AddEdge { src: *neighbor_id, dst: id, ... })?;  // Bidirectional
}

// 4. Atomic commit
txn.commit()?;
```

**Key Features:**
- Writes are immediately visible to reads within the same transaction
- Atomic commit (all-or-nothing)
- Auto-rollback on drop (if not committed)
- Optional forwarding to fulltext indexer

**Requirements Addressed**: [CON-1](./REQUIREMENTS.md#con-1) (Read-after-write consistency)

---

## 3. Algorithm Implementations

### 3.1 HNSW

**Location**: `examples/vector/hnsw.rs`

**Parameters** (from [REQUIREMENTS.md](./REQUIREMENTS.md) Section 6.1):

| Parameter | Current | Description |
|-----------|---------|-------------|
| M | 16 | Connections per node |
| M_max0 | 32 | Max connections at layer 0 |
| ef_construction | 200 | Build beam width |
| ef_search | 100 | Query beam width |

**Insert Algorithm:**
1. Assign random level (geometric distribution)
2. Greedy search from top layer to level 0
3. At each layer: find M nearest neighbors
4. Connect bidirectionally, prune if > M_max

### 3.2 Vamana (DiskANN)

**Location**: `examples/vector/vamana.rs`

**Parameters** (from [REQUIREMENTS.md](./REQUIREMENTS.md) Section 6.2):

| Parameter | Current | Description |
|-----------|---------|-------------|
| R | 64 | Max out-degree |
| L | 100-200 | Search list size |
| alpha | 1.2 | RNG pruning threshold |
| num_passes | 3 | Construction passes |

**Key Differences from HNSW:**
- Single-layer graph (simpler, more disk-friendly)
- Medoid-based entry point
- RNG (Relative Neighborhood Graph) pruning

---

## 4. Current Performance

### 4.1 Benchmark Results (from [PERF.md](./PERF.md))

| Scale | Algorithm | Recall@10 | Build (vec/s) | Search (ms) | QPS |
|-------|-----------|-----------|---------------|-------------|-----|
| 1K | HNSW | 52.6% | 103.6 | 8.6 | 117 |
| 10K | HNSW | 80.7% | 68.1 | 15.4 | 65 |
| 100K | HNSW | 81.7% | 49.8 | 19.0 | 53 |
| **1M** | **HNSW** | **95.3%** | 39.9 | 21.5 | **47** |
| 1K | Vamana | 61.0% | 72.7 | 4.7 | 213 |
| 10K | Vamana | 77.8% | 53.8 | 7.0 | 143 |
| 1M | Vamana (L=200) | 81.9% | 18.6 | 13.6 | 74 |

### 4.2 Requirements Status

| Requirement | Target | Current | Status |
|-------------|--------|---------|--------|
| [REC-1](./REQUIREMENTS.md#rec-1) (Recall@10 at 1M) | > 95% | 95.3% | ✅ Met |
| [LAT-1](./REQUIREMENTS.md#lat-1) (Search P50 at 1M) | < 20 ms | 21.5 ms | ⚠️ Close |
| [THR-1](./REQUIREMENTS.md#thr-1) (Insert throughput) | > 5,000/s | ~40/s | ❌ Gap: 125x |
| [THR-3](./REQUIREMENTS.md#thr-3) (Search QPS at 1M) | > 500 QPS | 47 QPS | ❌ Gap: 10x |

---

## 5. Known Limitations

### 5.1 Build Throughput Bottleneck

**Current**: ~40 inserts/sec at 1M scale
**Target** ([THR-1](./REQUIREMENTS.md#thr-1)): 5,000 inserts/sec

**Root Causes:**
1. **Per-insert flush overhead** (~20-25ms per flush)
2. **Sequential edge operations** (no batch API for edges)
3. **16-byte UUID overhead** (could be 4-byte u32)

**Solutions** (see [HNSW2.md](./HNSW2.md), [HYBRID.md](./HYBRID.md)):
- Batched flush patterns
- RocksDB merge operators for lock-free edge updates
- Compact ID allocation

### 5.2 Search QPS Bottleneck

**Current**: 47 QPS at 1M
**Target** ([THR-3](./REQUIREMENTS.md#thr-3)): 500 QPS

**Root Causes:**
1. **Sequential vector reads** (no batch fragment API)
2. **Full vector distance computation** (no PQ compression)
3. **No SIMD optimization** ([STOR-5](./REQUIREMENTS.md#stor-5))

**Solutions** (see [HYBRID.md](./HYBRID.md)):
- Batch NodeFragments API (Issue #17)
- Product Quantization ([STOR-4](./REQUIREMENTS.md#stor-4))
- SIMD AVX2 distance computation ([STOR-5](./REQUIREMENTS.md#stor-5))

---

## 6. API Reference

### 6.1 Mutations

| Mutation | Description | Column Families |
|----------|-------------|-----------------|
| `AddNode` | Create a vector node | `nodes` |
| `AddNodeFragment` | Store vector embedding | `node_fragments` |
| `AddEdge` | Create graph edge | `forward_edges`, `reverse_edges` |
| `UpdateEdgeWeight` | Update edge distance | `forward_edges` |
| `UpdateNodeActivePeriod` | Soft delete node | `nodes`, edges |

### 6.2 Queries

| Query | Description | Output |
|-------|-------------|--------|
| `NodeById` | Get node by ID | `(Name, Summary)` |
| `NodeFragment` | Get vector by ID | `Vec<(Timestamp, Content)>` |
| `OutgoingEdges` | Get edges from node | `Vec<(Name, Dst, Weight)>` |
| `IncomingEdges` | Get edges to node | `Vec<(Name, Src, Weight)>` |
| `AllNodes` | Scan all nodes | `Vec<(Id, Name, Summary)>` |
| `AllEdges` | Scan all edges | `Vec<(Src, Dst, Name, Weight)>` |

---

## 7. Next Steps

### Phase 2: HNSW Optimization ([HNSW2.md](./HNSW2.md))

Focus: **Build throughput** ([THR-1](./REQUIREMENTS.md#thr-1): 5,000 inserts/sec)

Key Changes:
- Roaring bitmap edge storage
- RocksDB merge operators
- u32 ID allocation
- In-memory graph layer

### Phase 3: GPU Acceleration ([IVFPQ.md](./IVFPQ.md))

Focus: **Search performance** ([THR-3](./REQUIREMENTS.md#thr-3): 500 QPS → 10,000+ QPS)

Key Changes:
- IVF-PQ indexing
- CAGRA GPU graph
- cuVS integration

### Phase 4: Production Scale ([HYBRID.md](./HYBRID.md))

Focus: **Billion-scale** ([SCALE-1](./REQUIREMENTS.md#scale-1): 1B vectors)

Key Changes:
- Hybrid memory/disk architecture
- PQ compression (< 64 GB RAM at 1B)
- Async batch updater

---

## Appendix: Code References

| Component | Location | Lines |
|-----------|----------|-------|
| RocksDB Schema | `libs/db/src/graph/schema.rs` | 1-425 |
| Mutations | `libs/db/src/graph/mutation.rs` | 1-827 |
| Transaction API | `libs/db/src/graph/transaction.rs` | 1-343 |
| HNSW Example | `examples/vector/hnsw.rs` | - |
| Vamana Example | `examples/vector/vamana.rs` | - |
| Flush API | `libs/db/src/graph/writer.rs` | - |

---

*Generated: 2025-12-24*
*Based on: motlie_db graph module, HNSW/Vamana examples, PERF.md benchmarks*
