# Flush API Design

> **Status:** Proposed
> **Issue:** [#26 - Read-after-write consistency](https://github.com/chungers/motlie/issues/26)
> **Priority:** P0/Critical
> **Estimated Effort:** 3-5 days

## Summary

Add a `flush()` method to Writer that waits for all pending mutations to be committed to RocksDB, enabling immediate read-after-write consistency without arbitrary sleep delays.

```rust
// Usage
writer.send(mutations).await?;
writer.flush().await?;  // Blocks until mutations are committed
// Reader now guaranteed to see the writes
```

---

## Problem Statement

### Current Architecture

The Writer uses an async MPSC channel to decouple mutation sends from RocksDB commits:

```
writer.send() ──► [MPSC queue] ──► Consumer.run() ──► txn.commit() ──► Reader sees it
     │                                    │
     │ returns                            │ processes async
     │ immediately                        │ on background task
     ▼                                    ▼
 caller continues               variable latency (0-10ms+)
```

### The 10ms Sleep Workaround

Vector search examples require a sleep between operations:

```rust
// examples/vector/hnsw.rs:625
index.insert(writer, &reader, node_id, vector, &mut rng).await?;
tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;  // Workaround
```

### Impact

| Metric | Current (10ms sleep) | Target (flush API) |
|--------|----------------------|-------------------|
| Max throughput | 100 inserts/sec | 1,000-10,000/sec |
| Sleep as % of time | **78%** | 0% |
| Index 1M vectors | ~9 hours | ~6-60 minutes |

---

## Architecture Verification

**Confirmed:** Reader and Writer share the **same TransactionDB** instance:

```rust
// storage.rs:350-366
let graph_storage = graph::Storage::readwrite(&graph_path);  // TransactionDB
let graph = Arc::new(graph::Graph::new(Arc::new(graph_storage)));

// Same Arc<Graph> passed to both
ReaderBuilder::new(graph.clone(), ...)  // Reader shares TransactionDB
WriterBuilder::new(graph, ...)          // Writer shares TransactionDB
```

This means **once the Consumer commits, the Reader sees writes instantly**. The only delay is the async MPSC processing time.

---

## Proposed Solution

### Design: Flush Marker with Oneshot Channel

1. Add a `Flush` mutation variant containing a oneshot sender
2. When Consumer processes the flush marker, it signals completion
3. `Writer::flush()` waits on the oneshot receiver

```
writer.flush()
     │
     ├──► send FlushMarker { completion: tx }
     │
     └──► rx.await  ◄───────────────────────────────┐
                                                     │
Consumer.run():                                      │
     │                                               │
     ├──► process regular mutations                  │
     ├──► txn.commit()                               │
     └──► completion.send(())  ──────────────────────┘
```

---

## Implementation Details

### Files to Modify

```
libs/db/src/
├── graph/
│   ├── writer.rs      # Add flush() method, modify Consumer
│   └── mutation.rs    # Add Flush variant to Mutation enum
├── writer.rs          # Add flush() to unified Writer
└── fulltext/
    └── writer.rs      # (Optional) Add flush support to fulltext
```

---

### Step 1: Add Flush Mutation Variant

**File:** `libs/db/src/graph/mutation.rs`

```rust
use std::sync::Mutex;
use tokio::sync::oneshot;

/// Marker for flush synchronization.
///
/// Contains a oneshot sender that signals when the flush completes.
/// Uses Mutex<Option<...>> to allow taking ownership from a shared reference.
pub struct FlushMarker {
    completion: Mutex<Option<oneshot::Sender<()>>>,
}

impl FlushMarker {
    /// Create a new flush marker with completion channel.
    pub fn new(completion: oneshot::Sender<()>) -> Self {
        Self {
            completion: Mutex::new(Some(completion)),
        }
    }

    /// Take the completion sender (can only be called once).
    pub fn take_completion(&self) -> Option<oneshot::Sender<()>> {
        self.completion.lock().unwrap().take()
    }
}

impl std::fmt::Debug for FlushMarker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FlushMarker").finish()
    }
}

// Add to Mutation enum
pub enum Mutation {
    AddNode(AddNode),
    AddNodeFragment(AddNodeFragment),
    AddEdge(AddEdge),
    AddEdgeFragment(AddEdgeFragment),
    UpdateNodeValidSinceUntil(UpdateNodeValidSinceUntil),
    UpdateEdgeValidSinceUntil(UpdateEdgeValidSinceUntil),
    UpdateEdgeWeight(UpdateEdgeWeight),

    /// Flush marker - signals completion via oneshot channel.
    /// Not persisted to storage, only used for synchronization.
    Flush(FlushMarker),
}
```

---

### Step 2: Add flush() to Graph Writer

**File:** `libs/db/src/graph/writer.rs`

```rust
use tokio::sync::oneshot;
use super::mutation::{FlushMarker, Mutation};

impl Writer {
    /// Send a batch of mutations to be processed.
    ///
    /// This method returns immediately after enqueueing.
    /// Use `flush()` to wait for mutations to be committed.
    pub async fn send(&self, mutations: Vec<Mutation>) -> Result<()> {
        self.sender
            .send(mutations)
            .await
            .context("Failed to send mutations to writer queue")
    }

    /// Flush all pending mutations and wait for commit.
    ///
    /// Returns when all mutations sent before this call are committed
    /// to RocksDB and visible to readers.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Send mutations
    /// writer.send(vec![AddNode { ... }]).await?;
    /// writer.send(vec![AddEdge { ... }]).await?;
    ///
    /// // Wait for all to be committed
    /// writer.flush().await?;
    ///
    /// // Now safe to read
    /// let result = query.run(&reader, timeout).await?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The writer channel is closed
    /// - The consumer task has panicked or been dropped
    pub async fn flush(&self) -> Result<()> {
        let (tx, rx) = oneshot::channel();

        // Send flush marker through the same channel as mutations
        self.sender
            .send(vec![Mutation::Flush(FlushMarker::new(tx))])
            .await
            .context("Failed to send flush marker - channel closed")?;

        // Wait for consumer to process it
        rx.await
            .context("Flush failed - consumer dropped completion channel")?;

        Ok(())
    }

    /// Check if the writer is still active.
    pub fn is_closed(&self) -> bool {
        self.sender.is_closed()
    }
}
```

---

### Step 3: Modify Graph Consumer

**File:** `libs/db/src/graph/writer.rs` (Consumer section)

```rust
impl<P: Processor> Consumer<P> {
    /// Process mutations continuously until the channel is closed.
    #[tracing::instrument(skip(self), name = "mutation_consumer")]
    pub async fn run(mut self) -> Result<()> {
        tracing::info!(config = ?self.config, "Starting mutation consumer");

        loop {
            match self.receiver.recv().await {
                Some(mutations) => {
                    self.process_batch(mutations)
                        .await
                        .with_context(|| "Failed to process mutation batch")?;
                }
                None => {
                    tracing::info!("Mutation consumer shutting down - channel closed");
                    return Ok(());
                }
            }
        }
    }

    /// Process a batch of mutations, handling flush markers.
    async fn process_batch(&self, mutations: Vec<Mutation>) -> Result<()> {
        // Separate flush markers from regular mutations
        let mut regular_mutations = Vec::with_capacity(mutations.len());
        let mut flush_completions = Vec::new();

        for mutation in mutations {
            match mutation {
                Mutation::Flush(marker) => {
                    if let Some(completion) = marker.take_completion() {
                        flush_completions.push(completion);
                    }
                }
                other => regular_mutations.push(other),
            }
        }

        // Process regular mutations (commits to RocksDB)
        if !regular_mutations.is_empty() {
            self.processor
                .process_mutations(&regular_mutations)
                .await
                .with_context(|| format!(
                    "Failed to process {} mutations",
                    regular_mutations.len()
                ))?;

            // Forward to next consumer in chain (e.g., fulltext)
            if let Some(next) = &self.next {
                if let Err(e) = next.try_send(regular_mutations) {
                    tracing::warn!(
                        error = %e,
                        "Failed to forward mutations to next consumer"
                    );
                }
            }
        }

        // Signal completion for all flush markers
        // This happens AFTER the RocksDB commit, ensuring visibility
        for completion in flush_completions {
            // Ignore error if receiver was dropped (caller gave up waiting)
            let _ = completion.send(());
        }

        Ok(())
    }
}
```

---

### Step 4: Add flush() to Unified Writer

**File:** `libs/db/src/writer.rs`

```rust
impl Writer {
    /// Send a batch of mutations to be processed.
    pub async fn send(&self, mutations: Vec<Mutation>) -> Result<()> {
        self.graph_writer.send(mutations).await
    }

    /// Flush all pending mutations to graph storage.
    ///
    /// Waits for all mutations sent before this call to be committed
    /// to RocksDB. After this returns, readers are guaranteed to see
    /// the writes.
    ///
    /// # Note
    ///
    /// This flushes the graph store only. Fulltext indexing may still
    /// be in progress. For most use cases (including vector search),
    /// graph consistency is sufficient.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use motlie_db::mutation::{AddNode, Runnable};
    ///
    /// AddNode { ... }.run(&writer).await?;
    /// writer.flush().await?;
    ///
    /// // Reads now see the new node
    /// let node = NodeById::new(id).run(&reader, timeout).await?;
    /// ```
    pub async fn flush(&self) -> Result<()> {
        self.graph_writer.flush().await
    }
}
```

---

### Step 5: Skip Flush in MutationExecutor

**File:** `libs/db/src/graph/mutation.rs`

The `Flush` variant should not be executed against RocksDB:

```rust
impl MutationExecutor for Mutation {
    fn execute(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<()> {
        match self {
            Mutation::AddNode(m) => m.execute(txn, txn_db),
            Mutation::AddNodeFragment(m) => m.execute(txn, txn_db),
            Mutation::AddEdge(m) => m.execute(txn, txn_db),
            // ... other variants ...

            // Flush is not a storage operation
            Mutation::Flush(_) => Ok(()),
        }
    }
}
```

---

## Usage Examples

### Per-Insert Flush (Maximum Consistency)

```rust
for vector in vectors {
    index.insert(writer, &reader, node_id, vector).await?;
    writer.flush().await?;  // Each insert immediately visible
}
```

### Batch Flush (Better Throughput)

```rust
for (i, vector) in vectors.iter().enumerate() {
    index.insert(writer, &reader, node_id, vector).await?;

    // Flush every 100 inserts
    if (i + 1) % 100 == 0 {
        writer.flush().await?;
    }
}
writer.flush().await?;  // Final flush
```

### Conditional Flush

```rust
// Only flush when we need to read
for vector in vectors {
    index.insert(writer, &reader, node_id, vector).await?;
}

// Flush before search
writer.flush().await?;
let results = index.search(&reader, query, k).await?;
```

---

## Performance Expectations

### Latency Comparison

| Approach | Per-Operation Overhead | Notes |
|----------|------------------------|-------|
| 10ms sleep | 10ms fixed | Current workaround |
| flush() per insert | ~0.1-1ms | Actual RocksDB commit time |
| flush() per batch (100) | ~0.01ms amortized | Best throughput |

### Throughput Comparison

| Approach | Throughput | Improvement |
|----------|------------|-------------|
| 10ms sleep | 100/sec | Baseline |
| flush() per insert | 1,000-10,000/sec | **10-100x** |
| flush() per batch | 10,000+/sec | **100x+** |

---

## Testing Plan

### Unit Tests

```rust
#[tokio::test]
async fn test_flush_returns_after_commit() {
    let (writer, receiver) = create_mutation_writer(Default::default());
    let consumer = spawn_mutation_consumer(receiver, processor);

    // Send mutation
    writer.send(vec![AddNode { ... }]).await.unwrap();

    // Flush should complete
    writer.flush().await.unwrap();

    // Verify mutation was processed
    assert!(processor.was_called());
}

#[tokio::test]
async fn test_flush_with_closed_channel() {
    let (writer, receiver) = create_mutation_writer(Default::default());
    drop(receiver);  // Close channel

    // Flush should return error
    assert!(writer.flush().await.is_err());
}

#[tokio::test]
async fn test_multiple_flushes() {
    // Verify multiple sequential flushes work correctly
}

#[tokio::test]
async fn test_concurrent_flushes() {
    // Verify concurrent flush calls are handled correctly
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_read_after_write_with_flush() {
    let storage = Storage::readwrite(path);
    let handles = storage.ready(config)?;

    let node_id = Id::new();
    AddNode { id: node_id, ... }.run(handles.writer()).await?;

    // Without flush, this might fail
    handles.writer().flush().await?;

    // Now read should succeed
    let node = NodeById::new(node_id)
        .run(handles.reader(), timeout)
        .await?;

    assert!(node.is_some());
}
```

### Vector Search Integration

```rust
#[tokio::test]
async fn test_hnsw_with_flush() {
    // Build index using flush instead of sleep
    for vector in vectors {
        index.insert(writer, &reader, id, vector).await?;
        writer.flush().await?;
    }

    // Search should find all vectors
    let results = index.search(&reader, query, k).await?;
    assert_eq!(results.len(), k);
}
```

---

## Implementation Checklist

- [ ] Add `FlushMarker` struct to `graph/mutation.rs`
- [ ] Add `Flush` variant to `Mutation` enum
- [ ] Update `MutationExecutor` impl to handle `Flush`
- [ ] Implement `flush()` on `graph::Writer`
- [ ] Modify `graph::Consumer::process_batch()` to handle flush markers
- [ ] Implement `flush()` on unified `Writer`
- [ ] Add unit tests for flush behavior
- [ ] Add integration test: write → flush → read
- [ ] Update vector examples to use flush instead of sleep
- [ ] Benchmark flush latency vs 10ms sleep
- [ ] Update documentation

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Oneshot channel overhead | Low - single allocation | Acceptable for correctness |
| Consumer blocked on slow commit | Medium | Already the case; flush makes it explicit |
| Deadlock if consumer panics | Medium | Oneshot receiver returns error |
| Breaking API change | None | Additive - `flush()` is new method |
| Fulltext not flushed | Low | Document behavior; add fulltext flush if needed |

---

## Batch API Enhancements

The `Vec<Mutation>` batch interface can be extended to provide cleaner read-after-write semantics. This section defines a unified API supporting four usage patterns.

### Unified Writer API

```rust
impl Writer {
    // ========================================================================
    // Pattern 1: Fire-and-Forget (Async)
    // ========================================================================

    /// Send mutations asynchronously without waiting for commit.
    ///
    /// Returns immediately after enqueueing. Use `flush()` to ensure visibility.
    /// Best for high-throughput scenarios where eventual consistency is acceptable.
    pub async fn send(&self, mutations: Vec<Mutation>) -> Result<()>;

    /// Wait for all pending mutations to be committed.
    pub async fn flush(&self) -> Result<()>;

    // ========================================================================
    // Pattern 2: Synchronous Send
    // ========================================================================

    /// Send mutations and wait for commit.
    ///
    /// Returns when all mutations are visible to readers.
    /// Equivalent to `send()` followed by `flush()`.
    pub async fn send_sync(&self, mutations: Vec<Mutation>) -> Result<()> {
        self.send(mutations).await?;
        self.flush().await
    }

    // ========================================================================
    // Pattern 3: Batch Builder
    // ========================================================================

    /// Start building a batch of mutations with fluent API.
    pub fn batch(&self) -> BatchBuilder<'_>;

    // ========================================================================
    // Pattern 4: Transaction Scope
    // ========================================================================

    /// Begin a transaction for read-your-writes semantics.
    ///
    /// Writes within the transaction are immediately visible to reads
    /// within the same transaction, before commit.
    pub fn transaction(&self) -> Transaction<'_>;
}
```

---

### Pattern 1: Fire-and-Forget (High Throughput)

Best for bulk ingestion where you don't need immediate reads.

```rust
// Send many batches without waiting
for batch in mutation_batches {
    writer.send(batch).await?;
}

// Flush once at the end
writer.flush().await?;

// Now all mutations are visible
let results = query.run(&reader, timeout).await?;
```

**Characteristics:**
- Lowest latency per send
- Highest throughput
- Must call `flush()` before reads
- Good for: bulk imports, batch ETL

---

### Pattern 2: Synchronous Send (Simple Consistency)

Best when you need immediate visibility after each batch.

```rust
// Each send waits for commit
writer.send_sync(vec![
    AddNode { id, name, ... }.into(),
    AddEdge { src, dst, ... }.into(),
]).await?;

// Immediately visible
let node = NodeById::new(id).run(&reader, timeout).await?;
assert!(node.is_some());
```

**Characteristics:**
- Simple mental model: send = visible
- Higher latency than fire-and-forget
- Good for: interactive operations, CRUD APIs

---

### Pattern 3: Batch Builder (Fluent API)

Best for readable, type-safe batch construction.

```rust
// Build and commit atomically
writer.batch()
    .add(AddNode { id: node1, name: "Alice".into(), ... })
    .add(AddNode { id: node2, name: "Bob".into(), ... })
    .add(AddEdge { src: node1, dst: node2, name: "knows".into(), ... })
    .commit().await?;

// All mutations visible
let edges = OutgoingEdges::new(node1, None).run(&reader, timeout).await?;
assert_eq!(edges.len(), 1);
```

**Or fire-and-forget:**

```rust
writer.batch()
    .add(AddNode { ... })
    .add(AddEdge { ... })
    .send().await?;  // Async, no wait

// Later...
writer.flush().await?;
```

**Implementation:**

```rust
pub struct BatchBuilder<'a> {
    writer: &'a Writer,
    mutations: Vec<Mutation>,
}

impl<'a> BatchBuilder<'a> {
    /// Add a mutation to the batch.
    pub fn add<M: Into<Mutation>>(mut self, mutation: M) -> Self {
        self.mutations.push(mutation.into());
        self
    }

    /// Add multiple mutations to the batch.
    pub fn add_all<I, M>(mut self, mutations: I) -> Self
    where
        I: IntoIterator<Item = M>,
        M: Into<Mutation>,
    {
        self.mutations.extend(mutations.into_iter().map(Into::into));
        self
    }

    /// Send batch asynchronously (fire-and-forget).
    pub async fn send(self) -> Result<()> {
        if self.mutations.is_empty() {
            return Ok(());
        }
        self.writer.send(self.mutations).await
    }

    /// Send batch and wait for commit (read-after-write safe).
    pub async fn commit(self) -> Result<()> {
        if self.mutations.is_empty() {
            return Ok(());
        }
        self.writer.send_sync(self.mutations).await
    }

    /// Get the number of mutations in the batch.
    pub fn len(&self) -> usize {
        self.mutations.len()
    }

    /// Check if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.mutations.is_empty()
    }
}
```

**Characteristics:**
- Readable, self-documenting code
- Type-safe mutation construction
- Explicit commit vs send semantics
- Good for: application code, complex operations

---

### Pattern 4: Transaction Scope (Read-Your-Writes)

Best for operations that need to read uncommitted writes.

```rust
// Begin transaction
let txn = writer.transaction();

// Write mutations (not yet committed)
txn.write(AddNode { id: node_id, name: "Alice".into(), ... })?;
txn.write(AddNodeFragment { id: node_id, content: vector_data, ... })?;

// Read within transaction sees uncommitted writes!
let node = txn.read(NodeById::new(node_id))?;
assert!(node.is_some());  // Visible within transaction

// More writes based on reads
let edges = txn.read(OutgoingEdges::new(other_node, None))?;
for (_, neighbor_id, _) in edges {
    txn.write(AddEdge { src: node_id, dst: neighbor_id, ... })?;
}

// Atomic commit - all or nothing
txn.commit().await?;

// Now visible to all readers
```

**Implementation:**

```rust
pub struct Transaction<'a> {
    writer: &'a Writer,
    storage: Arc<graph::Storage>,
    txn: rocksdb::Transaction<'a, rocksdb::TransactionDB>,
    mutations: Vec<Mutation>,  // Track for fulltext forwarding
}

impl<'a> Transaction<'a> {
    /// Write a mutation (visible to read() within this transaction).
    pub fn write<M: Into<Mutation>>(&mut self, mutation: M) -> Result<()> {
        let m = mutation.into();

        // Execute against RocksDB transaction (not committed yet)
        m.execute(&self.txn, self.storage.transaction_db()?)?;

        // Track for later fulltext forwarding
        self.mutations.push(m);

        Ok(())
    }

    /// Read with visibility of uncommitted writes in this transaction.
    ///
    /// Uses RocksDB transaction's read-your-writes capability.
    pub fn read<Q>(&self, query: Q) -> Result<Q::Output>
    where
        Q: QueryExecutor,
    {
        query.execute_in_transaction(&self.txn, self.storage.as_ref())
    }

    /// Commit all changes atomically.
    ///
    /// After commit returns, all mutations are visible to readers.
    pub async fn commit(self) -> Result<()> {
        // Commit RocksDB transaction
        self.txn.commit()?;

        // Forward mutations to fulltext consumer (async)
        if !self.mutations.is_empty() {
            self.writer.forward_to_fulltext(self.mutations).await?;
        }

        Ok(())
    }

    /// Rollback all changes.
    ///
    /// Discards all mutations in this transaction.
    pub fn rollback(self) -> Result<()> {
        self.txn.rollback()?;
        Ok(())
    }

    /// Get the number of mutations in this transaction.
    pub fn len(&self) -> usize {
        self.mutations.len()
    }
}

// Auto-rollback on drop if not committed
impl<'a> Drop for Transaction<'a> {
    fn drop(&mut self) {
        // Transaction will auto-rollback if not committed
        // RocksDB handles this internally
    }
}
```

**Required Trait Extension:**

```rust
pub trait QueryExecutor: Send + Sync {
    type Output;

    /// Execute query against storage.
    fn execute(&self, storage: &graph::Storage) -> Result<Self::Output>;

    /// Execute query within a transaction (sees uncommitted writes).
    fn execute_in_transaction(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        storage: &graph::Storage,
    ) -> Result<Self::Output>;
}
```

**Characteristics:**
- True read-your-writes (zero latency within txn)
- Atomic multi-mutation transactions
- Rollback support
- Good for: HNSW insert, graph algorithms, complex transactions

---

### Pattern Comparison

| Pattern | Latency | Throughput | Read-After-Write | Rollback | Complexity |
|---------|---------|------------|------------------|----------|------------|
| 1. Fire-and-forget | Lowest | Highest | After flush | No | Low |
| 2. Sync send | Medium | Medium | Per-batch | No | Low |
| 3. Batch builder | Medium | Medium | Per-batch | No | Low |
| 4. Transaction | Lowest | High | Within txn | Yes | Medium |

---

### Usage Recommendations

| Use Case | Recommended Pattern |
|----------|---------------------|
| Bulk data import | Pattern 1 (fire-and-forget + flush) |
| REST API CRUD | Pattern 2 (send_sync) |
| Application logic | Pattern 3 (batch builder) |
| HNSW/Vamana insert | Pattern 4 (transaction) |
| Graph algorithms | Pattern 4 (transaction) |
| Simple scripts | Pattern 2 (send_sync) |

---

### HNSW Insert with Transaction (Example)

```rust
impl HnswIndex {
    pub async fn insert(
        &mut self,
        writer: &Writer,
        node_id: Id,
        vector: Vec<f32>,
    ) -> Result<()> {
        let txn = writer.transaction();

        // Store vector
        txn.write(AddNode {
            id: node_id,
            name: format!("vec_{}", node_id),
            ...
        })?;
        txn.write(AddNodeFragment {
            id: node_id,
            content: FragmentContent::from_f32_vec(&vector),
            ...
        })?;

        // Find neighbors (reads see the new node!)
        let entry_point = self.get_entry_point(&txn)?;
        let neighbors = self.greedy_search(&txn, &vector, entry_point, self.params.ef_construction)?;

        // Add edges
        for (dist, neighbor_id) in neighbors.iter().take(self.params.m) {
            txn.write(AddEdge {
                src_id: node_id,
                dst_id: *neighbor_id,
                weight: *dist,
                name: "hnsw_L0".into(),
                ...
            })?;

            // Bidirectional
            txn.write(AddEdge {
                src_id: *neighbor_id,
                dst_id: node_id,
                weight: *dist,
                name: "hnsw_L0".into(),
                ...
            })?;

            // Prune if needed (reads current edges from txn)
            self.prune_connections(&txn, *neighbor_id, 0)?;
        }

        // Atomic commit
        txn.commit().await?;

        Ok(())
    }
}
```

**Result:** No sleep needed, true read-your-writes, 100x throughput improvement.

---

### Implementation Phases

#### Phase 1: Flush API (This Document)
- `flush()` method
- `send_sync()` convenience method
- Estimated: 3-5 days

#### Phase 2: Batch Builder
- `BatchBuilder` struct
- `batch()` method on Writer
- Estimated: 1-2 days

#### Phase 3: Transaction Scope
- `Transaction` struct
- `QueryExecutor::execute_in_transaction()` trait method
- Fulltext forwarding after commit
- Estimated: 1-2 weeks

---

### Extended Implementation Checklist

#### Phase 1: Flush
- [ ] Add `FlushMarker` struct
- [ ] Add `Flush` variant to Mutation enum
- [ ] Implement `flush()` on graph::Writer
- [ ] Implement `send_sync()` on graph::Writer
- [ ] Implement `flush()` and `send_sync()` on unified Writer
- [ ] Add tests

#### Phase 2: Batch Builder
- [ ] Add `BatchBuilder` struct
- [ ] Implement `batch()` on Writer
- [ ] Add `add()`, `add_all()`, `send()`, `commit()` methods
- [ ] Add tests

#### Phase 3: Transaction Scope
- [ ] Add `Transaction` struct
- [ ] Implement `transaction()` on Writer
- [ ] Add `QueryExecutor::execute_in_transaction()` trait method
- [ ] Implement for all query types (NodeById, OutgoingEdges, etc.)
- [ ] Handle fulltext forwarding after commit
- [ ] Add rollback support
- [ ] Add tests
- [ ] Update HNSW/Vamana examples

---

## Fulltext Flush

If fulltext consistency is also needed:

```rust
impl Writer {
    /// Flush both graph and fulltext stores.
    pub async fn flush_all(&self) -> Result<()> {
        self.graph_writer.flush().await?;
        self.fulltext_writer.flush().await?;
        Ok(())
    }
}
```

---

## References

- [Issue #26: Read-after-write consistency](https://github.com/chungers/motlie/issues/26)
- [RocksDB Transactions](https://github.com/facebook/rocksdb/wiki/Transactions)
- [Vector Search PERF.md](../../examples/vector/PERF.md)
- [Concurrency and Storage Modes](./concurrency-and-storage-modes.md)
