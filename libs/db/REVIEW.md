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
Return standard Rust `Iterator`s (synchronous) or `Stream`s (asynchronous).

**Code Example:**

```rust
// libs/db/src/graph/scan.rs

pub struct OutgoingEdgeIterator<'a> {
    db_iter: rocksdb::DBIterator<'a>,
    src_id: Id,
    // ...
}

impl<'a> Iterator for OutgoingEdgeIterator<'a> {
    type Item = Result<(DstId, EdgeName, Option<f64>)>;

    fn next(&mut self) -> Option<Self::Item> {
        // 1. Advance RocksDB iterator
        // 2. Check prefix (if we moved past src_id, return None)
        // 3. Deserialize Key/Value
        // 4. Return Item
    }
}

// Usage
let iter = graph.scan_outgoing(node_id);
for edge in iter.filter(|e| e.weight > Some(0.5)) {
    // ...
}
```

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
