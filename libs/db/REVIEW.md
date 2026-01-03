# Engineering Review: Motlie DB Architecture & API

**Reviewer:** Gemini Agent  
**Date:** December 24, 2025  
**Scope:** `libs/db` (Architecture, API, Implementation)

## Executive Summary

The `motlie_db` library implements a hybrid temporal graph database using RocksDB for primary storage and Tantivy for full-text search. The architecture is well-structured, utilizing strong Rust patterns (Type State, Command Pattern) to ensure safety and ergonomics. The unified storage abstraction effectively hides the complexity of managing two distinct storage engines.

Overall, the implementation is high-quality, idiomatic, and designed for correctness. Performance optimizations like column family layout and compression are already in place. The primary opportunities for improvement lie in reducing latency for point lookups and optimizing serialization for scan-heavy workloads.

## Architecture Assessment

### Strengths
- **Unified Storage Abstraction:** The `Storage` struct provides a single, cohesive entry point that manages the lifecycle of both RocksDB and Tantivy. The use of the Type State pattern (`Storage<ReadOnly>` vs `Storage<ReadWrite>`) provides excellent compile-time safety.
- **Mutation Pipeline:** The chaining of consumers (`Writer -> Graph Consumer -> Fulltext Consumer`) ensures that the full-text index is eventually consistent with the graph storage without complex coordination.
- **Transaction Support:** The `TransactionQueryExecutor` trait allows for "read-your-writes" semantics within a transaction, which is critical for complex consistency requirements.

### Considerations
- **Query Dispatch Overhead:** The current `Runnable` implementation forces all requests through a channel-based worker pool. For simple point lookups (e.g., `NodeById`), the overhead of channel passing and context switching may be significant compared to the raw RocksDB lookup time.
- **Fulltext Consistency:** `writer.flush()` currently only guarantees graph persistence.

## Detailed Recommendations & Implementation Guides

### 1. Zero-Copy Serialization with `rkyv`

**Problem:**
The current `rmp_serde` implementation performs a full copy deserialization for every object read from RocksDB. For scan-heavy workloads (e.g., `AllEdges`, `OutgoingEdges`), this results in massive allocation churn (`String`, `Vec`, `DataUrl`) even if only a subset of fields (like `weight`) is accessed.

**Proposal:**
Migrate value serialization from `rmp_serde` to [rkyv](https://github.com/rkyv/rkyv). `rkyv` guarantees total zero-copy deserialization by structuring the on-disk bytes such that they can be effectively cast to a reference of the Rust type.

**Implementation Details:**

1.  **Dependency:** Add `rkyv = { version = "0.7", features = ["validation"] }` to `Cargo.toml`.
2.  **Derive Macros:** Replace `serde::Serialize/Deserialize` with `rkyv::Archive, rkyv::Serialize, rkyv::Deserialize`.

**Code Example:**

```rust
// libs/db/src/graph/schema.rs

use rkyv::{Archive, Deserialize, Serialize};

#[derive(Archive, Deserialize, Serialize, Debug)]
#[archive(check_bytes)]
pub struct ForwardEdgeCfValue {
    // rkyv will layout these fields so they can be read directly from the byte buffer
    pub valid_range: Option<TemporalRange>,
    pub weight: Option<f64>,
    // String/DataUrl data is stored inline but accessed via pointer logic
    pub summary: EdgeSummary, 
}

// In libs/db/src/graph/reader.rs
pub fn get_edge_weight_zero_copy(bytes: &[u8]) -> Option<f64> {
    // 1. Validate the buffer (optional in trusted environments, but good for safety)
    let archived = rkyv::check_archived_root::<ForwardEdgeCfValue>(bytes)
        .expect("Data corruption");
    
    // 2. Access fields directly from the raw bytes. 
    // No allocations happen here. 'archived' is just a reference wrapper.
    if let Some(weight) = archived.weight {
        return Some(weight);
    }
    None
}
```

**Benefits:**
- **Zero Allocations:** Reading a node or edge becomes purely pointer arithmetic.
- **Partial Reads:** If you only need `weight`, you never pay the cost of processing the `summary` string.

### 2. Direct Read Path (Trait-Based)

**Problem:**
The current `Query` architecture uses the Actor pattern: `Query -> Channel -> Worker Thread -> RocksDB -> Oneshot Channel -> Result`.
For a simple `Get` operation which might take 5µs in RocksDB (cached), the channel overhead and context switching can add 50-100µs of latency.

**Proposal:**
Introduce a `RunDirect` trait. This mirrors the existing `Runnable` (async) trait but executes synchronously on the current thread, bypassing the worker pool. This maintains API consistency (`query.run(...)` vs `query.run_direct(...)`).

**Implementation Details:**

1.  **Define Trait:** Create `RunDirect` in `libs/db/src/reader.rs`.
2.  **Implement:** Implement `RunDirect<Reader>` for point lookup queries like `NodeById`.

**Code Example:**

```rust
// libs/db/src/reader.rs

/// Trait for queries that can be executed immediately on the current thread.
/// This bypasses the async worker pool for lower latency on simple point lookups.
pub trait RunDirect<R> {
    type Output;
    fn run_direct(self, reader: &R) -> Result<Self::Output>;
}

// libs/db/src/graph/query.rs

impl RunDirect<Reader> for NodeById {
    type Output = Option<(NodeName, NodeSummary)>;

    fn run_direct(self, reader: &Reader) -> Result<Self::Output> {
        // Access raw DB handle directly from the reader
        let db = reader.storage().graph.storage().db()?;
        
        let key = schema::NodeCfKey(self.id);
        let key_bytes = schema::Nodes::key_to_bytes(&key);
        
        // Blocking I/O call happens on caller's thread
        if let Some(val_bytes) = db.get_cf(reader.graph().nodes_cf(), key_bytes)? {
            let val = schema::Nodes::value_from_bytes(&val_bytes)?;
            
            // Perform temporal validity check inline
            let ref_time = self.reference_ts_millis.unwrap_or_else(TimestampMilli::now);
            if !schema::is_valid_at_time(&val.0, ref_time) {
                return Ok(None);
            }
            
            return Ok(Some((val.1, val.2)));
        }
        Ok(None)
    }
}
```

**Usage:**

```rust
// Async/Worker path (existing)
let node = NodeById::new(id, None).run(&reader).await?;

// Direct/Sync path (new)
// Significantly faster for cached reads; blocks the current thread.
let node = NodeById::new(id, None).run_direct(&reader)?;
```

**Benefits:**
- **API Consistency:** Keeps the "Command Pattern" where the struct defines the parameters.
- **Performance:** Eliminates channel RTT and context switching for hot-path lookups.

### 3. Blob Separation & Cache Locality

**Problem:**
The `ForwardEdges` column family stores `EdgeSummary` (potentially large text/DataUrl) alongside `weight` and `topology`.
When iterating edges (e.g., `OutgoingEdges` for BFS or PageRank), RocksDB loads the entire value into the block cache. Large summaries pollute the cache, evicting other edge data, and increase memory bandwidth usage.

**Proposal:**
Split `ForwardEdges` into two Column Families:
1.  **`ForwardEdges` (Hot):** `[src][dst][name] -> [temporal_range][weight]`
2.  **`EdgeSummaries` (Cold):** `[src][dst][name] -> [EdgeSummary]`

**Schema Design:**

```rust
// CF: "forward_edges" (Hot Data)
// Key: [src_id][dst_id][name]
// Value:
struct ForwardEdgeHot {
    valid_range: Option<TemporalRange>,
    weight: Option<f64>,
    // Optional: flag to indicate if summary exists
    has_summary: bool, 
}

// CF: "edge_summaries" (Cold Data)
// Key: [src_id][dst_id][name] (Same key structure, enabling parallel seek)
// Value:
struct EdgeSummaryBlob(EdgeSummary);
```

**Operational Changes:**
- **Mutations:** `AddEdge` writes to *both* CFs atomically (in the same WriteBatch).
- **Scanning:** `OutgoingEdges` only iterates `ForwardEdges`. It runs much faster as values are tiny (~20 bytes vs ~500 bytes).
- **Point Lookup:** `EdgeSummaryBySrcDstName` performs a `get` on `ForwardEdges` to check validity/weight, and then (if needed) a `get` on `EdgeSummaries`.

**Benefits:**
- **Cache Efficiency:** Block cache stores ~25x more edges per megabyte.
- **Scan Speed:** Iterators process significantly less data bandwidth.

### 4. Iterator-Based Scan API

**Problem:**
The visitor pattern (`accept(&mut visitor)`) is rigid and hard to compose. It forces the control flow inside the `scan` module.

**Proposal:**
Return standard Rust `Iterator`s (synchronous) or `Stream`s (asynchronous). This naturally supports pagination via `IteratorMode::From` (seek).

**Code Example:**

```rust
// libs/db/src/graph/scan.rs

pub struct NodeIterator<'a> {
    db_iter: rocksdb::DBIterator<'a>,
    // Optional: end bound or prefix to stop iteration
}

impl<'a> NodeIterator<'a> {
    pub fn new(db: &'a rocksdb::DB, start_from: Option<Id>) -> Self {
        let mode = match start_from {
            Some(id) => {
                 let key = schema::NodeCfKey(id);
                 let bytes = schema::Nodes::key_to_bytes(&key);
                 // Start strictly after the last ID (exclusive pagination)
                 // Note: Ideally, we seek to bytes + 1 or handle exclusive logic in next()
                 rocksdb::IteratorMode::From(&bytes, rocksdb::Direction::Forward)
            }
            None => rocksdb::IteratorMode::Start,
        };
        
        let db_iter = db.iterator_cf(db.cf_handle(schema::Nodes::CF_NAME).unwrap(), mode);
        Self { db_iter }
    }
}

impl<'a> Iterator for NodeIterator<'a> {
    type Item = Result<(Id, NodeName, NodeSummary)>;

    fn next(&mut self) -> Option<Self::Item> {
        let (key_bytes, val_bytes) = self.db_iter.next()?;
        // Deserialize and return...
    }
}

// Usage in AllNodes query
impl AllNodes {
    pub fn execute_iter(&self, storage: &Storage) -> Result<Vec<(Id, NodeName, NodeSummary)>> {
        let iter = NodeIterator::new(storage.db()?, self.last);
        
        // Skip the first element if it matches 'last' exactly (exclusive cursor)
        let iter = if self.last.is_some() {
             iter.skip_while(|res| res.as_ref().ok().map(|(id,_,_)| Some(*id) == self.last).unwrap_or(false))
        } else {
             iter
        };

        iter.take(self.limit).collect()
    }
}
```

**Pagination Impact:**
- **Efficient Seek:** RocksDB's `IteratorMode::From` performs a seek to the precise key, ensuring O(1) start time for any page, avoiding O(N) skips.
- **Composition:** Pagination logic becomes `iter.skip_while(...).take(limit)`.

### 5. Fulltext Sync Mechanism

**Problem:**
Tests and consistency-sensitive workflows are flaky because `writer.flush()` returns before Tantivy has indexed the data.

**Proposal:**
Enhance `FlushMarker` to support a secondary acknowledgment.

**Implementation:**

```rust
// libs/db/src/graph/mutation.rs

pub struct FlushMarker {
    // Current: Signals when Graph consumer is done
    pub graph_completion: Option<oneshot::Sender<()>>,
    // New: Signals when Fulltext consumer is done
    pub fulltext_completion: Option<oneshot::Sender<()>>,
}

// libs/db/src/writer.rs

impl Writer {
    pub async fn flush_all(&self) -> Result<()> {
        let (graph_tx, graph_rx) = oneshot::channel();
        let (ft_tx, ft_rx) = oneshot::channel();
        
        let marker = Mutation::Flush(FlushMarker { 
            graph_completion: Some(graph_tx),
            fulltext_completion: Some(ft_tx) 
        });
        
        self.send(vec![marker]).await?;
        
        // Wait for both
        let _ = tokio::join!(graph_rx, ft_rx);
        Ok(())
    }
}
```

## Projected Performance Impact

### 1. Graph Algorithms (BFS, PageRank, Louvain)

These algorithms rely heavily on scanning graph topology (`OutgoingEdges`) and weights, but rarely access metadata (EdgeSummary).

*   **Impact of Blob Separation:** This is the most critical optimization. Currently, scanning edges loads large summary blobs into the RocksDB block cache.
    *   *Current State:* An edge entry might be ~500 bytes (summary included). 1GB cache holds ~2 million edges.
    *   *Optimized State:* An edge entry in the "Hot" CF is ~20-30 bytes. 1GB cache holds ~40 million edges.
    *   *Result:* **10x-20x increase in effective cache capacity** for topology data, leading to significantly fewer disk seeks during traversal.
*   **Impact of Zero-Copy (`rkyv`):**
    *   *Current State:* Deserializing 1 million edges allocates 1 million `Vec`s/`String`s (via MessagePack).
    *   *Optimized State:* Zero allocations during scan.
    *   *Result:* **~2x-5x throughput improvement** for in-memory scans by eliminating allocator pressure.

**Overall Projection:** Complex graph algorithms on datasets larger than RAM could see **5x-10x end-to-end speedup**.

### 2. Vector Search (HNSW / Vamana)

Vector search involves traversing a graph (navigable small world) to find nearest neighbors. This requires many sequential point lookups (greedy search).

*   **Impact of Direct Read Path (`RunDirect`):**
    *   *Current State:* Each hop in the graph (lookup node -> get neighbors) incurs a channel round-trip (~10-50µs overhead) plus context switching.
    *   *Optimized State:* Cached reads are immediate function calls (~1-5µs).
    *   *Result:* For a search requiring 200 hops, latency drops from ~10ms (overhead dominated) to ~1ms (compute/read dominated).
*   **Impact of Blob Separation:**
    *   Similar to graph algorithms, HNSW traversal only needs neighbor IDs and potentially vectors (if stored in graph), not document content. Keeping the traversal graph compact ensures it stays in CPU/L3 cache.

**Overall Projection:** Vector search latency (p99) could decrease by **3x-5x**, enabling high-QPS low-latency applications.

## API Compatibility Assessment

The proposed improvements have been designed to maintain strict backward compatibility with the existing `Runnable` and `Mutation` APIs.

| Improvement | Impact Level | API Changes | Internal Changes |
| :--- | :--- | :--- | :--- |
| **Zero-Copy (`rkyv`)** | None | None. Public structs (`NodeSummary`, `Id`) remain standard Rust types. | Serialization logic switches from `rmp_serde` to `rkyv`. Conversion happens transparently in `Reader`/`Writer`. |
| **Direct Read Path** | Additive | Adds `RunDirect` trait. Existing `Runnable` implementation can be refactored to wrap `RunDirect` for backward compatibility, or remain independent. | New method implementations on `Reader`. |
| **Blob Separation** | None | None. `AddEdge` and `EdgeSummaryBySrcDstName` structs remain unchanged. | `MutationExecutor` writes to 2 CFs. `QueryExecutor` reads from 2 CFs (if needed). |
| **Iterator Scans** | None | None. The existing `AllNodes` query can internally utilize the new Iterator to collect results into a `Vec`, preserving the current `async run()` signature. | `scan.rs` refactored from Visitor to Iterator pattern. |
| **Fulltext Sync** | Additive | Adds `flush_all()`. `flush()` remains unchanged (graph-only). | `FlushMarker` updated to hold dual channels. |

---

## Independent Assessment

**Reviewer:** Claude Opus 4.5
**Date:** December 25, 2025
**Scope:** Validation of recommendations against actual implementation

### Executive Summary

The Gemini Agent review is **technically competent and largely accurate**. However, several recommendations have nuances that require consideration before implementation. The most impactful recommendation is Blob Separation (#3), while the rkyv recommendation (#1) is incomplete due to overlooking the existing LZ4 compression layer.

---

### Recommendation 1: Zero-Copy Serialization with `rkyv`

**Assessment: Partially Valid ⚠️**

The review correctly identifies that `rmp_serde` performs full-copy deserialization. However, it **overlooks a critical implementation detail**: the current codebase uses **LZ4 compression** on all values.

From `libs/db/src/graph/mod.rs:114-129`:

```rust
fn value_to_bytes(value: &Self::Value) -> Result<Vec<u8>, ...> {
    let msgpack_bytes = rmp_serde::to_vec(value)?;
    let compressed = lz4::block::compress(&msgpack_bytes, None, true)?;
    Ok(compressed)
}

fn value_from_bytes(bytes: &[u8]) -> Result<Self::Value, ...> {
    let decompressed = lz4::block::decompress(bytes, None)?;
    rmp_serde::from_slice(&decompressed)
}
```

**Issue**: Zero-copy deserialization with `rkyv` requires uncompressed data—you cannot cast compressed bytes directly to a struct reference. This creates a trade-off:

| Option | Trade-off |
|--------|-----------|
| Remove LZ4 compression | Storage size increases ~2-4x |
| Decompress then use rkyv | Negates zero-copy benefit (still allocates decompression buffer) |
| Selective compression | Complex: compress cold data (summaries), leave hot data (topology) uncompressed |

**Projected 2x-5x improvement**: Achievable only if the LZ4 trade-off is resolved. The review's code examples would not work on the current compressed storage format.

**Recommendation**: Combine with Blob Separation (#3)—compress cold EdgeSummary blobs, leave hot topology data uncompressed for zero-copy access.

---

### Recommendation 2: Direct Read Path (`RunDirect`)

**Assessment: Valid ✓**

The current architecture routes all queries through MPSC channels with oneshot result delivery, as shown in `libs/db/src/query.rs:103-117`.

**Accuracy Check**: The review's latency estimate of **50-100µs overhead** may be conservative. The codebase uses `flume` channels (per `Cargo.toml:21`), which typically have ~1-10µs overhead per operation. However, the full round-trip includes:
- Channel send (~1-5µs)
- Worker thread wake/context switch (~10-50µs)
- Oneshot receive (~1-5µs)

Total overhead of 20-60µs per lookup is realistic.

**Considerations**:
- Blocking synchronous I/O in async context requires `spawn_blocking` or risks starving the runtime
- The current architecture provides natural backpressure and batching opportunities
- Direct reads would need careful synchronization with `TransactionDB`

**Projected 3x-5x improvement for vector search**: Reasonable if point lookup latency is the dominant factor in HNSW traversal.

---

### Recommendation 3: Blob Separation & Cache Locality

**Assessment: Strong Recommendation ✓**

The review **correctly identifies** the schema layout issue. From `libs/db/src/graph/schema.rs:81-85`:

```rust
pub(crate) struct ForwardEdgeCfValue(
    pub(crate) Option<TemporalRange>,  // ~10-20 bytes
    pub(crate) Option<f64>,             // ~8 bytes
    pub(crate) EdgeSummary,             // DataUrl - variable, potentially large
);
```

`EdgeSummary` is defined as `DataUrl` (line 109), which can contain arbitrary text or binary data. During graph traversal algorithms (BFS, PageRank, Louvain), only topology and weights are needed—loading large summaries into the block cache is wasteful.

**Validation of Projections**:
- Current edge entry: ~500 bytes (with typical summary) → ~2M edges per 1GB cache
- Optimized hot entry: ~25-30 bytes → ~40M edges per 1GB cache
- **10x-20x cache efficiency improvement**: Mathematically sound

This is a **well-established optimization pattern** used in production graph databases (Neo4j property separation, DGraph predicate sharding).

---

### Recommendation 4: Iterator-Based Scan API

**Assessment: Minor Improvement ⚠️**

The current visitor pattern in `libs/db/src/graph/scan.rs` already implements efficient pagination using `IteratorMode::From` (line 246):

```rust
let mode = if seek_key.is_empty() {
    if reverse { IteratorMode::End } else { IteratorMode::Start }
} else {
    IteratorMode::From(&seek_key, direction)
};
```

The review's claim about O(1) seek time is **already achieved** in the current implementation.

**Primary Benefit**: Ergonomics, not performance. Iterator composition (`iter.filter().take().collect()`) is more idiomatic than visitor callbacks.

**Consideration**: RocksDB iterator lifetimes are tied to the DB handle, making iterator-based APIs tricky to expose safely across async boundaries.

**Projected impact**: Minimal performance gain. This is a code quality improvement, not a performance optimization.

---

### Recommendation 5: Fulltext Sync Mechanism

**Assessment: Valid Concern, but Design Intent Unclear ⚠️**

The review correctly identifies that `FlushMarker` has only one completion channel. From `libs/db/src/graph/mutation.rs:38-56`:

```rust
pub struct FlushMarker {
    completion: Mutex<Option<oneshot::Sender<()>>>,
}
```

**However**, examining the consumer chaining test (mutation.rs:922-1026) reveals **intentional eventual consistency**:

```rust
// Test explicitly validates that fulltext drops messages under load
assert!(
    fulltext_processed < num_mutations,
    "Fulltext processor should have processed fewer than {} mutations
     due to buffer overflow..."
);
```

The graph→fulltext chain uses `try_send` which drops mutations when the buffer is full. This appears to be a **deliberate design choice** for:
- Graph write latency not blocked by fulltext indexing
- Graceful degradation under load

**Assessment**: The recommendation is valid for use cases requiring **strong consistency** between graph and fulltext. However, it may conflict with the original design intent of eventual consistency.

**Recommendation**: Document the consistency model explicitly. If strong consistency is required, implement `flush_all()`. If eventual consistency is acceptable, add a background reconciliation mechanism instead.

---

### Performance Projections Validation

| Claim | Validation | Conditions |
|-------|------------|------------|
| **5x-10x for graph algorithms** | **Optimistic but achievable** | Requires blob separation AND dataset exceeding RAM. Cache locality is the primary driver. |
| **3x-5x for vector search** | **Reasonable** | Assumes point lookup latency dominates. Actual gain depends on cache hit rates. |
| **10x-20x cache efficiency** | **Mathematically sound** | Based on edge entry size reduction from ~500 to ~25 bytes. |
| **2x-5x scan throughput with rkyv** | **Conditional** | Only achievable if LZ4 compression is removed or restructured. |

---

### Prioritized Implementation Roadmap

Based on impact vs. effort analysis:

| Priority | Recommendation | Impact | Effort | Dependencies |
|----------|----------------|--------|--------|--------------|
| **1** | Blob Separation (#3) | High | Medium | None |
| **2** | Direct Read Path (#2) | Medium | Low | None |
| **3** | Zero-Copy with rkyv (#1) | High | High | Blob Separation (for selective compression) |
| **4** | Fulltext Sync (#5) | Low | Low | Clarify design intent first |
| **5** | Iterator Scans (#4) | Low | Medium | None (ergonomic improvement) |

---

### Conclusion

The Gemini Agent review demonstrates solid understanding of database internals and provides actionable recommendations. The primary gap is the incomplete analysis of the rkyv migration due to overlooking LZ4 compression.

**Recommended next steps**:
1. Implement Blob Separation first—it provides immediate cache benefits and enables selective compression for future rkyv adoption
2. Add `RunDirect` trait for latency-sensitive point lookups
3. Revisit rkyv after blob separation, applying zero-copy only to hot topology data

---

## Proposed Architecture: Hybrid Serialization Strategy

**Date:** December 25, 2025

### Overview

Combine blob separation with a hybrid serialization approach:

| Data Category | Column Family | Serialization | Compression | Access Pattern |
|---------------|---------------|---------------|-------------|----------------|
| **Hot** (topology, weights, temporal) | `forward_edges_hot`, `nodes_hot` | rkyv | None | Zero-copy, high-frequency traversal |
| **Cold** (summaries, content) | `edge_summaries`, `node_summaries` | rmp_serde | LZ4 | Full deser, infrequent access |
| **Fragments** (historical content) | `node_fragments`, `edge_fragments` | rmp_serde | LZ4 | Full deser, rare access |

This resolves the LZ4/rkyv conflict by applying each serialization strategy where it's most effective.

### Schema Design

#### Hot Column Families (rkyv, no compression)

```rust
// libs/db/src/graph/schema_hot.rs

use rkyv::{Archive, Deserialize, Serialize};

/// Hot edge data - optimized for graph traversal
/// Size: ~30 bytes per edge (vs ~500 bytes with summary)
#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[archive(check_bytes)]
pub struct ForwardEdgeHotValue {
    pub valid_range: Option<ArchivedTemporalRange>,
    pub weight: Option<f64>,
    pub has_summary: bool,  // Flag to indicate cold data exists
}

/// Hot node data - optimized for lookups
#[derive(Archive, Deserialize, Serialize, Debug, Clone)]
#[archive(check_bytes)]
pub struct NodeHotValue {
    pub valid_range: Option<ArchivedTemporalRange>,
    pub name: ArchivedString,  // rkyv's zero-copy string
    pub has_summary: bool,
}

/// rkyv-compatible temporal range
#[derive(Archive, Deserialize, Serialize, Debug, Clone, Copy)]
#[archive(check_bytes)]
pub struct ArchivedTemporalRange {
    pub start: Option<u64>,
    pub until: Option<u64>,
}
```

#### Cold Column Families (rmp_serde + LZ4)

```rust
// libs/db/src/graph/schema_cold.rs

use serde::{Deserialize, Serialize};

/// Cold edge data - summaries stored separately
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EdgeSummaryColdValue(pub DataUrl);

/// Cold node data - summaries stored separately
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NodeSummaryColdValue(pub DataUrl);
```

### Trait Design

```rust
// libs/db/src/graph/mod.rs

/// Trait for hot column families using rkyv (zero-copy)
pub(crate) trait HotColumnFamily {
    const CF_NAME: &'static str;
    type Key;
    type Value: rkyv::Archive + rkyv::Serialize<...>;

    fn key_to_bytes(key: &Self::Key) -> Vec<u8>;
    fn key_from_bytes(bytes: &[u8]) -> Result<Self::Key>;

    /// Zero-copy value access - returns archived reference
    fn value_archived(bytes: &[u8]) -> Result<&<Self::Value as Archive>::Archived> {
        rkyv::check_archived_root::<Self::Value>(bytes)
            .map_err(|e| anyhow::anyhow!("Archive validation failed: {}", e))
    }

    /// Full deserialization when mutation is needed
    fn value_from_bytes(bytes: &[u8]) -> Result<Self::Value> {
        let archived = Self::value_archived(bytes)?;
        Ok(archived.deserialize(&mut rkyv::Infallible)?)
    }

    fn value_to_bytes(value: &Self::Value) -> Result<rkyv::AlignedVec> {
        Ok(rkyv::to_bytes::<_, 256>(value)?)
    }
}

/// Trait for cold column families using rmp_serde + LZ4 (existing pattern)
pub(crate) trait ColdColumnFamily {
    const CF_NAME: &'static str;
    type Key;
    type Value: Serialize + DeserializeOwned;

    fn key_to_bytes(key: &Self::Key) -> Vec<u8>;
    fn key_from_bytes(bytes: &[u8]) -> Result<Self::Key>;

    fn value_to_bytes(value: &Self::Value) -> Result<Vec<u8>> {
        let msgpack = rmp_serde::to_vec(value)?;
        Ok(lz4::block::compress(&msgpack, None, true)?)
    }

    fn value_from_bytes(bytes: &[u8]) -> Result<Self::Value> {
        let decompressed = lz4::block::decompress(bytes, None)?;
        Ok(rmp_serde::from_slice(&decompressed)?)
    }
}
```

### Column Family Layout

```
Before (current):
├── nodes           (Id → TemporalRange + Name + Summary)  ~200-500 bytes
├── forward_edges   (Src+Dst+Name → TemporalRange + Weight + Summary)  ~200-500 bytes
├── reverse_edges   (Dst+Src+Name → TemporalRange)  ~30 bytes
├── node_fragments  (Id+Ts → Content)  variable
└── edge_fragments  (Src+Dst+Name+Ts → Content)  variable

After (proposed):
├── nodes_hot       (Id → TemporalRange + Name + has_summary)  ~50 bytes [rkyv]
├── node_summaries  (Id → Summary)  variable [rmp+lz4]
├── forward_edges_hot (Src+Dst+Name → TemporalRange + Weight + has_summary)  ~30 bytes [rkyv]
├── edge_summaries  (Src+Dst+Name → Summary)  variable [rmp+lz4]
├── reverse_edges   (Dst+Src+Name → TemporalRange)  ~30 bytes [rkyv]
├── node_fragments  (Id+Ts → Content)  variable [rmp+lz4]
└── edge_fragments  (Src+Dst+Name+Ts → Content)  variable [rmp+lz4]
```

### Read Path Examples

#### Graph Traversal (Hot Path - Zero Copy)

```rust
impl OutgoingEdges {
    pub fn execute_zero_copy(&self, storage: &Storage) -> Result<Vec<(Option<f64>, DstId)>> {
        let db = storage.db()?;
        let cf = db.cf_handle(ForwardEdgesHot::CF_NAME)?;

        let prefix = self.src_id.into_bytes();
        let mut results = Vec::new();

        for item in db.prefix_iterator_cf(cf, &prefix) {
            let (key_bytes, value_bytes) = item?;

            // Zero-copy: just validate and cast, no allocation
            let archived = ForwardEdgesHot::value_archived(&value_bytes)?;

            // Check temporal validity directly on archived data
            if !is_valid_archived(&archived.valid_range, self.reference_ts) {
                continue;
            }

            // Extract weight without full deserialization
            let weight = archived.weight;
            let key = ForwardEdgesHot::key_from_bytes(&key_bytes)?;

            results.push((weight, key.1)); // dst_id
        }

        Ok(results)
    }
}
```

#### Full Edge Details (Hot + Cold)

```rust
impl EdgeDetails {
    pub fn execute(&self, storage: &Storage) -> Result<EdgeDetailsResult> {
        let db = storage.db()?;

        // 1. Read hot data (zero-copy)
        let hot_cf = db.cf_handle(ForwardEdgesHot::CF_NAME)?;
        let key = ForwardEdgeHotKey(self.src, self.dst, self.name.clone());
        let key_bytes = ForwardEdgesHot::key_to_bytes(&key);

        let hot_bytes = db.get_cf(hot_cf, &key_bytes)?
            .ok_or_else(|| anyhow::anyhow!("Edge not found"))?;

        let archived = ForwardEdgesHot::value_archived(&hot_bytes)?;
        let weight = archived.weight;

        // 2. Read cold data only if needed (full deserialization)
        let summary = if archived.has_summary {
            let cold_cf = db.cf_handle(EdgeSummaries::CF_NAME)?;
            let cold_bytes = db.get_cf(cold_cf, &key_bytes)?
                .ok_or_else(|| anyhow::anyhow!("Summary missing"))?;
            EdgeSummaries::value_from_bytes(&cold_bytes)?.0
        } else {
            DataUrl::from_text("")  // Empty summary
        };

        Ok((weight, self.src, self.dst, self.name.clone(), summary))
    }
}
```

### Write Path

```rust
impl MutationExecutor for AddEdge {
    fn execute(&self, txn: &Transaction, txn_db: &TransactionDB) -> Result<()> {
        // 1. Write to hot CF (rkyv serialization)
        let hot_cf = txn_db.cf_handle(ForwardEdgesHot::CF_NAME)?;
        let hot_key = ForwardEdgeHotKey(self.source_node_id, self.target_node_id, self.name.clone());
        let hot_value = ForwardEdgeHotValue {
            valid_range: self.valid_range.map(Into::into),
            weight: self.weight,
            has_summary: !self.summary.as_ref().is_empty(),
        };
        txn.put_cf(hot_cf,
            ForwardEdgesHot::key_to_bytes(&hot_key),
            ForwardEdgesHot::value_to_bytes(&hot_value)?)?;

        // 2. Write to cold CF if summary exists (rmp + lz4)
        if !self.summary.as_ref().is_empty() {
            let cold_cf = txn_db.cf_handle(EdgeSummaries::CF_NAME)?;
            let cold_value = EdgeSummaryColdValue(self.summary.clone());
            txn.put_cf(cold_cf,
                ForwardEdgesHot::key_to_bytes(&hot_key),  // Same key
                EdgeSummaries::value_to_bytes(&cold_value)?)?;
        }

        // 3. Write reverse edge (hot only, no summary)
        let reverse_cf = txn_db.cf_handle(ReverseEdgesHot::CF_NAME)?;
        // ... similar pattern

        Ok(())
    }
}
```

### Performance Projections

| Operation | Current | With Hybrid | Improvement |
|-----------|---------|-------------|-------------|
| **Edge scan (1M edges)** | ~500ms (deserialize + decompress) | ~50ms (zero-copy) | **10x** |
| **Point lookup (cached)** | ~100µs (channel + deserialize) | ~5µs (direct + zero-copy) | **20x** |
| **BFS traversal (100K nodes)** | ~2s | ~200ms | **10x** |
| **Storage size** | 1x | ~1.1x (hot uncompressed, cold compressed) | ~10% increase |

### Migration Strategy

1. **Phase 1**: Add new `*_hot` and `*_summaries` column families alongside existing ones
2. **Phase 2**: Dual-write to both old and new CFs during transition
3. **Phase 3**: Background migration job to populate new CFs from old data
4. **Phase 4**: Switch reads to new CFs, deprecate old CFs
5. **Phase 5**: Remove old CFs in next major version

### Dependencies

Add to `Cargo.toml`:
```toml
rkyv = { version = "0.8", features = ["validation", "strict"] }
```

### Trade-offs

| Aspect | Benefit | Cost |
|--------|---------|------|
| **Serialization complexity** | Optimal format per data type | Two serialization paths to maintain |
| **Storage size** | Cold data still compressed | Hot data ~10% larger (no compression) |
| **Schema evolution** | rkyv requires careful versioning | Need migration tooling for hot CFs |
| **Code complexity** | Clear separation of concerns | More column families to manage |

### Conclusion

The hybrid serialization strategy maximizes performance where it matters most (graph traversal) while preserving storage efficiency for large, infrequently-accessed content. This approach directly addresses the LZ4/rkyv conflict identified in the original review assessment.

---

## Evaluation: Direct Read Path vs Transaction API for 1B Vector Scale

**Date:** December 25, 2025

### Context

The original review recommends a `RunDirect` trait to bypass MPSC channel dispatch for point lookups. However, this must be evaluated against:

1. **Existing Transaction API**: `libs/db/src/graph/transaction.rs` implements read-your-writes semantics via `Transaction<'a>`
2. **HNSW2 Design**: `examples/vector/HNSW2.md` explicitly requires `WriteBatchWithIndex` for read-your-writes during HNSW insertion
3. **HYBRID Architecture**: `examples/vector/HYBRID.md` targets 1B vectors with async graph updater

### Key Finding: Orthogonal Concerns

**Direct Read Path** and **Transaction API** solve different problems:

| Concern | Direct Read Path | Transaction API |
|---------|------------------|-----------------|
| **Problem** | Channel dispatch overhead (~1-3µs) | ACID read-your-writes semantics |
| **Use case** | Simple point queries outside transactions | Multi-step atomic operations |
| **Conflict?** | No—these can coexist | No—different contexts |

### Current Transaction API (Essential for HNSW)

From `libs/db/src/graph/transaction.rs:90-103`:

```rust
pub struct Transaction<'a> {
    txn: Option<rocksdb::Transaction<'a, rocksdb::TransactionDB>>,
    txn_db: &'a rocksdb::TransactionDB,
    mutations: Vec<Mutation>,
    forward_to: Option<mpsc::Sender<Vec<Mutation>>>,
}
```

The `read()` method (line 201-207) uses `TransactionQueryExecutor` to execute queries within transaction scope, enabling patterns like:

```rust
// HNSW insertion requires read-your-writes
txn.write(AddNode { id: node_id, ... })?;
let neighbors = txn.read(OutgoingEdges::new(node_id, Some("hnsw")))?;  // Sees uncommitted!
for neighbor in neighbors {
    txn.write(AddEdge { src: node_id, dst: neighbor, ... })?;
}
txn.commit()?;
```

### HNSW2 Design Requirements

From `examples/vector/HNSW2.md`, the optimized design explicitly requires read-your-writes:

> **WriteBatchWithIndex**: "GetFromBatchAndDB sees uncommitted write!"
>
> ```cpp
> WriteBatchWithIndex wbwi;
> wbwi.Put(node_key, node_data);
> auto neighbors = GetFromBatchAndDB(wbwi, edge_prefix);  // Sees uncommitted!
> wbwi.Put(edge_key, roaring_bitmap);
> db.Write(wbwi);  // Atomic commit
> ```

This is the exact pattern implemented in the current Transaction API.

### Performance Analysis at 1B Vector Scale

| Operation | Latency | Channel Overhead | Overhead % |
|-----------|---------|------------------|------------|
| HNSW search (1B vectors) | 10-50ms | ~1-3µs | **0.01-0.03%** |
| RaBitQ scan (HYBRID) | 5-20ms | ~1-3µs | **0.02-0.06%** |
| Insert (async graph updater) | <10ms p99 | N/A | **Decoupled** |
| Point lookup (cached) | ~20-60µs | ~1-3µs | **5-15%** |

**Observation**: Channel overhead is only significant for individual point lookups. At scale:
- Search operations dominate (10-50ms) — channel overhead is noise
- HYBRID architecture uses async graph updater — insert latency already decoupled
- Batch operations amortize channel overhead across many lookups

### Pros of Direct Read Path

| Benefit | Impact | Applicability |
|---------|--------|---------------|
| Reduced latency for point queries | ~20-50µs savings | **Moderate**: Only benefits single lookups outside transactions |
| Simpler debugging | Shorter stack traces | **Minor**: Developer convenience |
| Bulk migration performance | Faster sequential reads | **Moderate**: Useful for tooling |

### Cons and Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Breaks transaction semantics if misused** | **High** | Direct reads bypass transaction scope—developer might accidentally use during HNSW insertion, miss uncommitted writes, select wrong neighbors |
| **Dual code paths** | **Medium** | Same query needs both `RunDirect` and `TransactionQueryExecutor` implementations |
| **API confusion** | **Medium** | When to use `run_direct()` vs `run()` vs `txn.read()`? |
| **Negligible benefit at scale** | **Low** | 1-3µs overhead on 10-50ms operations is <0.1% |

### Risk Scenario: Incorrect HNSW Insertion

```rust
// DANGEROUS: Developer uses direct read by mistake
let mut txn = writer.transaction()?;
txn.write(AddNode { id: node_id, ... })?;

// BUG: run_direct() doesn't see uncommitted writes!
let neighbors = OutgoingEdges::new(node_id, Some("hnsw")).run_direct(&reader)?;
// neighbors will be EMPTY because node_id isn't committed yet

for neighbor in neighbors {
    txn.write(AddEdge { ... })?;  // Never executes!
}
txn.commit()?;
// Result: Orphan node with no HNSW edges
```

This bug would be **silent**—no error, just incorrect index structure.

### Recommendation: **Defer Implementation**

For 1B vector scale with HNSW/HYBRID architecture:

| Factor | Assessment |
|--------|------------|
| Performance gain | **Negligible** (<0.1% of search latency at scale) |
| Implementation cost | **Moderate** (dual code paths, API complexity) |
| Risk of misuse | **High** (breaks read-your-writes if used incorrectly) |
| Current bottleneck | **Not channel overhead** (async updater already decouples) |

**The Transaction API is essential and sufficient** for HNSW operations. The HYBRID architecture's async graph updater already addresses insert latency concerns.

### Alternative Recommendations

Instead of Direct Read Path, focus on optimizations with higher ROI:

| Optimization | Impact | Effort | Risk |
|--------------|--------|--------|------|
| **Blob Separation (#3)** | 10x-20x cache efficiency | Medium | Low |
| **Hybrid Serialization** | 10x scan throughput | Medium | Low |
| **RaBitQ integration** | 32x vector compression | High | Medium |
| **Roaring bitmaps for edges** | 10x-100x edge storage | Medium | Low |

### Conclusion

**Skip Direct Read Path for now.** The Transaction API provides the read-your-writes semantics essential for HNSW/Vamana algorithms. Channel dispatch overhead (~1-3µs) is negligible compared to search latency (10-50ms) at billion-scale. The risk of developers misusing direct reads and breaking transaction semantics outweighs the marginal latency savings.

**If point lookup latency becomes a bottleneck in the future**, consider:
1. Batching lookups to amortize channel overhead
2. Read-through caching layer for hot nodes
3. `RunDirect` only for read-only, non-transactional contexts (migrations, analytics)

---

## Clarification: rkyv Applicability to HNSW2 Edge Storage

**Date:** December 25, 2025

### Summary

The rkyv + blob separation optimization applies to **motlie_db graph storage** (nodes, edges with summaries). It does **not** apply to HNSW2's packed edge representation, which uses a different optimization strategy.

### Why rkyv + Blob Separation Works for Graph Storage

The graph storage schema stores heterogeneous data together:

```rust
// Current: Hot and cold data mixed
struct ForwardEdgeCfValue(
    Option<TemporalRange>,  // Hot: 10-20 bytes, accessed every traversal
    Option<f64>,             // Hot: 8 bytes, accessed every traversal
    EdgeSummary,             // Cold: variable, rarely accessed
);
```

**Blob separation** moves cold data to separate CFs:
- Hot CF with rkyv: zero-copy access to topology/weights
- Cold CF with rmp+lz4: compressed summaries

### Why rkyv Does NOT Apply to HNSW2 Edges

HNSW2 uses **Roaring Bitmaps** for edge adjacency lists (per `examples/vector/HNSW2.md`):

```rust
// HNSW2 edge storage
CF: edges
Key:   node_id | layer (5 bytes)
Value: RoaringBitmap serialized (~50-200 bytes)
```

**Roaring bitmaps are not a good fit for rkyv** because:

| Factor | rkyv Pattern | Roaring Pattern |
|--------|--------------|-----------------|
| **Access** | Field access: `archived.weight` | Method calls: `bitmap.contains(id)` |
| **Compression** | None (raw bytes) | Run-length + array containers |
| **Iteration** | Direct memory | Iterator over containers |
| **Updates** | Full replace | Merge operators |

Roaring bitmaps have their own optimized serialization that:
- Compresses integer sets (4-10x smaller than explicit lists)
- Supports O(1) membership tests
- Enables bitmap intersection for filtered search
- Works with RocksDB merge operators

### Where rkyv DOES Apply in HNSW2

rkyv can benefit **metadata structures** in HNSW2:

```rust
// Node metadata - CAN use rkyv
#[derive(Archive, Serialize, Deserialize)]
struct NodeMeta {
    max_layer: u8,
    flags: u8,
    created_at: u64,
}

// Graph metadata - CAN use rkyv
#[derive(Archive, Serialize, Deserialize)]
struct GraphMeta {
    entry_point: u32,
    max_level: u8,
    node_count: u64,
    ef_construction: u16,
    m: u16,
    m_max: u16,
}
```

These are small, fixed-size structures where zero-copy access provides marginal benefit but adds consistency with the hot CF pattern.

### Optimization Strategy by Component

| Component | Storage | Optimization | Rationale |
|-----------|---------|--------------|-----------|
| **Graph nodes (hot)** | `nodes_hot` | rkyv | Zero-copy field access |
| **Graph edges (hot)** | `forward_edges_hot` | rkyv | Zero-copy traversal |
| **Graph summaries** | `*_summaries` | rmp+lz4 | Compression for cold data |
| **HNSW vectors** | `vectors` | Raw f32 | Already minimal |
| **HNSW edges** | `edges` | Roaring bitmap | Native compression + set ops |
| **HNSW metadata** | `node_meta`, `graph_meta` | rkyv | Consistency with graph pattern |

### Conclusion

**For motlie_db graph storage**: rkyv + blob separation is the primary optimization (10x-20x cache efficiency).

**For HNSW2**: Roaring bitmaps provide equivalent optimization for edge storage. rkyv applies only to small metadata structures for consistency.

---
