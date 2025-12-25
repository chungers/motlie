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
