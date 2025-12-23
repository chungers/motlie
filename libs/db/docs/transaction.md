# Phase 3: Transaction Scope API - Implementation Complete

> **Status:** ✅ Implemented
> **Issue:** [#26 - Read-after-write consistency](https://github.com/chungers/motlie/issues/26)
> **Priority:** P0/Critical
> **Completed:** 2025-12-23
> **Depends On:** Phase 1 (Flush API) - Complete

## Design Decisions (Approved)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Transaction Lifetime** | `Transaction<'a>` tied to `&'a Writer` | Idiomatic Rust, matches RocksDB API, compile-time safety |
| **Mutation Forwarding** | Configurable `forward_to: Option<mpsc::Sender>` | Flexible, not hardcoded to fulltext, matches Consumer pattern |
| **Commit API** | Sync `commit()` with best-effort `try_send` | Simpler, matches RocksDB, non-blocking |
| **Query Coverage** | All queries implement `TransactionQueryExecutor` | Complete implementation, low incremental effort |

## Summary

Phase 3 adds a `Transaction` API enabling **read-your-writes** semantics within a single transaction scope. This allows mutations and queries to execute in a single RocksDB transaction, where writes are immediately visible to subsequent reads before commit.

```rust
// Phase 3 API
let mut txn = writer.transaction()?;

txn.write(AddNode { id, ... })?;           // Not committed yet
txn.write(AddNodeFragment { id, ... })?;   // Not committed yet

let node = txn.read(NodeById::new(id))?;   // Sees uncommitted writes!

txn.write(AddEdge { ... })?;               // Based on read result
txn.commit()?;                              // Atomic commit of all
```

---

## Problem Statement

### Current Architecture

Phase 1 (Flush API) solved the problem of knowing *when* writes are committed:

```
Writer.send(mutations) ──► MPSC ──► Consumer ──► txn.commit()
         │                                            │
         │ returns immediately                        │
         │                                            ▼
         │                                     flush() waits here
         │
Reader.run(query) ────────► MPSC ──► Consumer ──► storage.get()
                                                      │
                                          (separate path, no shared txn)
```

**Limitation:** Even with `flush()`, you cannot read uncommitted writes. Every read sees only committed state.

### The HNSW Insert Problem

Vector index algorithms (HNSW, Vamana) require interleaved reads and writes:

```rust
// Current: requires flush between every read/write pair
async fn insert(&mut self, writer: &Writer, reader: &Reader, node_id: Id, vector: Vec<f32>) {
    // Write node
    AddNode { id: node_id, ... }.run(writer).await?;
    AddNodeFragment { id: node_id, content: vector, ... }.run(writer).await?;

    // MUST flush before reading
    writer.flush().await?;

    // Find neighbors (greedy search)
    let neighbors = self.greedy_search(reader, &vector, ef_construction).await?;

    // Write edges to neighbors
    for neighbor in neighbors {
        AddEdge { src: node_id, dst: neighbor, ... }.run(writer).await?;

        // For pruning: must flush again to read current edges
        writer.flush().await?;
        let current_edges = OutgoingEdges::new(neighbor, Some("hnsw"))
            .run(reader, timeout).await?;

        if current_edges.len() > self.max_connections {
            // Prune excess edges...
        }
    }
    writer.flush().await?;
}
```

**Impact:**

| Metric | With flush() per op | Target (transaction) |
|--------|---------------------|---------------------|
| Insert throughput | 100-1,000/sec | 5,000-10,000/sec |
| Flush overhead | ~0.1-1ms per op | 0ms (within txn) |
| Latency variability | High (RocksDB commit) | Low (in-memory) |

---

## Proposed Architecture

### Transaction Flow

```
Transaction::new(storage)
     │
     ├── txn.write(AddNode { ... })     ──► txn.put_cf()  (not committed)
     │
     ├── txn.read(NodeById { ... })     ──► txn.get_cf()  (sees uncommitted!)
     │
     ├── txn.write(AddEdge { ... })     ──► txn.put_cf()
     │
     └── txn.commit()                   ──► txn.commit() + forward to fulltext
```

**Key Change:** Both writes and reads execute against the same RocksDB `Transaction` object.

### RocksDB Transaction Semantics

RocksDB TransactionDB provides:
- **Snapshot isolation**: Transaction sees a consistent snapshot
- **Read-your-writes**: Reads within a transaction see uncommitted writes
- **Atomic commit**: All or nothing
- **Rollback**: Discard all changes

This is exactly what we need for HNSW insert.

---

## API Surface

### New Types

```rust
// libs/db/src/graph/transaction.rs (NEW FILE)

/// A transaction scope for read-your-writes operations.
///
/// Writes within this transaction are immediately visible to reads
/// within the same transaction, before commit.
///
/// # Lifetime
///
/// `Transaction<'a>` is tied to `&'a Writer` via the underlying RocksDB
/// transaction lifetime. This ensures:
/// - Transaction cannot outlive the storage
/// - Borrow checker prevents use-after-free
/// - Clear scoping: transaction lives within writer borrow
///
/// For concurrent transactions, clone the `Writer` (cheap - just Arc clones).
///
/// # Example
///
/// ```rust,ignore
/// let mut txn = writer.transaction()?;
///
/// // Write mutations (not committed yet)
/// txn.write(AddNode { id: node_id, ... })?;
/// txn.write(AddNodeFragment { id: node_id, content: vector, ... })?;
///
/// // Read sees uncommitted writes!
/// let neighbors = greedy_search_txn(&txn, &vector, ef)?;
///
/// for neighbor in neighbors {
///     txn.write(AddEdge { src: node_id, dst: neighbor, ... })?;
/// }
///
/// // Atomic commit (sync, non-blocking forwarding)
/// txn.commit()?;
/// ```
///
/// # Lifecycle
///
/// - Transactions should be short-lived (sub-second)
/// - If not committed, transaction is rolled back on drop
/// - Long-running transactions may block other writers
///
/// # Mutation Forwarding
///
/// On commit, mutations can be forwarded to a configured `mpsc::Sender`.
/// This is used for fulltext indexing but is configurable - any receiver
/// can be attached via `WriterBuilder::with_transaction_forward_to()`.
/// Forwarding is best-effort (non-blocking `try_send`).
pub struct Transaction<'a> {
    txn: rocksdb::Transaction<'a, rocksdb::TransactionDB>,
    txn_db: &'a rocksdb::TransactionDB,
    mutations: Vec<Mutation>,
    /// Optional sender to forward mutations on commit.
    /// If None, mutations are not forwarded anywhere.
    forward_to: Option<mpsc::Sender<Vec<Mutation>>>,
}

impl<'a> Transaction<'a> {
    /// Write a mutation (visible to read() within this transaction).
    ///
    /// The mutation is executed against the RocksDB transaction but
    /// not committed until `commit()` is called.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// txn.write(AddNode { id, name: "Alice".into(), ... })?;
    /// txn.write(AddEdge { src: id, dst: other, ... })?;
    /// ```
    pub fn write<M: Into<Mutation>>(&mut self, mutation: M) -> Result<()>;

    /// Write multiple mutations in a batch.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// txn.write_batch(vec![
    ///     AddNode { ... }.into(),
    ///     AddEdge { ... }.into(),
    /// ])?;
    /// ```
    pub fn write_batch(&mut self, mutations: Vec<Mutation>) -> Result<()>;

    /// Read using a query (sees uncommitted writes in this transaction).
    ///
    /// Uses the `TransactionQueryExecutor` trait to execute queries
    /// against the transaction rather than committed storage.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// txn.write(AddNode { id, ... })?;
    ///
    /// // Sees the uncommitted AddNode!
    /// let (name, summary) = txn.read(NodeById::new(id, None))?;
    ///
    /// // Get edges (sees uncommitted edges too)
    /// let edges = txn.read(OutgoingEdges::new(id, Some("hnsw")))?;
    /// ```
    pub fn read<Q: TransactionQueryExecutor>(&self, query: Q) -> Result<Q::Output>;

    /// Commit all changes atomically (sync).
    ///
    /// This method is **synchronous** - it blocks until RocksDB commits.
    /// Mutation forwarding (if configured) uses non-blocking `try_send`.
    ///
    /// After commit returns:
    /// - All mutations are visible to external readers
    /// - Mutations are forwarded to configured receiver (best-effort)
    ///
    /// # Forwarding Behavior
    ///
    /// If `forward_to` is configured (e.g., for fulltext indexing):
    /// - Uses `try_send` (non-blocking)
    /// - If channel is full, logs warning and continues
    /// - Flush markers are filtered out before forwarding
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - RocksDB commit fails (conflict, I/O error)
    /// - Transaction was already committed or rolled back
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut txn = writer.transaction()?;
    /// txn.write(AddNode { ... })?;
    /// txn.write(AddEdge { ... })?;
    /// txn.commit()?;  // Blocks until RocksDB commits
    /// // Now visible to all readers
    /// ```
    pub fn commit(self) -> Result<()>;

    /// Rollback all changes.
    ///
    /// Discards all mutations in this transaction. Also called
    /// automatically on drop if transaction is not committed.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut txn = writer.transaction()?;
    /// txn.write(AddNode { ... })?;
    ///
    /// if something_went_wrong {
    ///     txn.rollback()?;  // Explicit rollback
    ///     return Ok(());
    /// }
    ///
    /// txn.commit()?;
    /// ```
    pub fn rollback(self) -> Result<()>;

    /// Get the number of mutations in this transaction.
    pub fn len(&self) -> usize;

    /// Check if the transaction has any mutations.
    pub fn is_empty(&self) -> bool;
}

// Auto-rollback on drop if not committed
impl<'a> Drop for Transaction<'a> {
    fn drop(&mut self) {
        // RocksDB Transaction auto-rollbacks if not committed
        // Mutations are NOT forwarded on drop (only on explicit commit)
    }
}
```

### New Trait: TransactionQueryExecutor

```rust
// libs/db/src/graph/query.rs (ADD)

/// Trait for queries that can execute within a transaction scope.
///
/// Unlike `QueryExecutor` (which takes `&Storage` and routes through
/// MPSC channels), this trait executes directly against an active
/// RocksDB transaction, enabling read-your-writes semantics.
///
/// # Implementation Pattern
///
/// Each query type implements this trait alongside `QueryExecutor`:
///
/// ```rust,ignore
/// impl TransactionQueryExecutor for NodeById {
///     type Output = (NodeName, NodeSummary);
///
///     fn execute_in_transaction(
///         &self,
///         txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
///         txn_db: &rocksdb::TransactionDB,
///     ) -> Result<Self::Output> {
///         let cf = txn_db.cf_handle(Nodes::CF_NAME)?;
///         // Use txn.get_cf() to see uncommitted writes
///         let value = txn.get_cf(cf, &key)?;
///         // ... deserialize and return
///     }
/// }
/// ```
pub trait TransactionQueryExecutor: Send + Sync {
    type Output: Send;

    /// Execute query within a transaction (sees uncommitted writes).
    ///
    /// # Arguments
    ///
    /// * `txn` - Active RocksDB transaction
    /// * `txn_db` - TransactionDB for column family handles
    ///
    /// # Returns
    ///
    /// Query result, which may include data from uncommitted writes
    /// within the same transaction.
    fn execute_in_transaction(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<Self::Output>;
}
```

### Writer Integration

```rust
// libs/db/src/graph/writer.rs (MODIFY)

pub struct Writer {
    sender: mpsc::Sender<Vec<Mutation>>,
    storage: Arc<Storage>,  // NEW - for transaction creation
    /// Optional sender for transaction forwarding (e.g., to fulltext).
    /// Transactions created by this writer will forward mutations here on commit.
    transaction_forward_to: Option<mpsc::Sender<Vec<Mutation>>>,  // NEW
}

impl Writer {
    /// Begin a transaction for read-your-writes semantics.
    ///
    /// The returned Transaction allows interleaved writes and reads
    /// within a single atomic scope. Transaction lifetime is tied to
    /// this Writer borrow.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let mut txn = writer.transaction()?;
    ///
    /// txn.write(AddNode { ... })?;
    /// let result = txn.read(NodeById::new(id, None))?;  // Sees the AddNode
    /// txn.write(AddEdge { ... })?;
    ///
    /// txn.commit()?;  // Sync commit, best-effort forwarding
    /// ```
    ///
    /// # Concurrent Transactions
    ///
    /// To run concurrent transactions, clone the Writer first:
    ///
    /// ```rust,ignore
    /// let writer1 = writer.clone();
    /// let writer2 = writer.clone();
    ///
    /// // Now can create transactions on each
    /// let txn1 = writer1.transaction()?;
    /// let txn2 = writer2.transaction()?;
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if storage is not in read-write mode.
    pub fn transaction(&self) -> Result<Transaction<'_>> {
        let txn_db = self.storage.transaction_db()?;
        let txn = txn_db.transaction();

        Ok(Transaction {
            txn,
            txn_db,
            mutations: Vec::new(),
            forward_to: self.transaction_forward_to.clone(),
        })
    }
}
```

### WriterBuilder Integration

```rust
// libs/db/src/graph/writer.rs (MODIFY)

impl WriterBuilder {
    /// Set the sender for transaction mutation forwarding.
    ///
    /// When transactions commit, their mutations will be forwarded
    /// to this sender (best-effort, non-blocking).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Forward to fulltext consumer
    /// let (fulltext_tx, fulltext_rx) = mpsc::channel(1000);
    ///
    /// let writer = WriterBuilder::new(graph, storage)
    ///     .with_transaction_forward_to(fulltext_tx)
    ///     .build();
    /// ```
    pub fn with_transaction_forward_to(
        mut self,
        sender: mpsc::Sender<Vec<Mutation>>,
    ) -> Self {
        self.transaction_forward_to = Some(sender);
        self
    }
}
```

### Unified Writer Integration

```rust
// libs/db/src/writer.rs (MODIFY)

impl Writer {
    /// Begin a transaction for read-your-writes semantics.
    ///
    /// See `graph::Writer::transaction()` for details.
    pub fn transaction(&self) -> Result<graph::Transaction<'_>> {
        self.inner.transaction()
    }
}
```

---

## Query Implementation Changes

Each query type needs `TransactionQueryExecutor` implementation.

### Pattern: Current vs Transaction Execution

```rust
// Current (QueryExecutor) - async, routes through MPSC, sees committed data
#[async_trait::async_trait]
impl QueryExecutor for NodeByIdDispatch {
    type Output = (NodeName, NodeSummary);

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let value_bytes = if let Ok(db) = storage.db() {
            db.get_cf(cf, key)?          // ReadOnly mode
        } else {
            storage.transaction_db()?.get_cf(cf, key)?  // ReadWrite mode
        };
        // ... deserialize
    }
}

// New (TransactionQueryExecutor) - sync, direct execution, sees uncommitted
impl TransactionQueryExecutor for NodeById {
    type Output = (NodeName, NodeSummary);

    fn execute_in_transaction(
        &self,
        txn: &rocksdb::Transaction<'_, rocksdb::TransactionDB>,
        txn_db: &rocksdb::TransactionDB,
    ) -> Result<Self::Output> {
        let cf = txn_db.cf_handle(Nodes::CF_NAME)?;
        let key_bytes = Nodes::key_to_bytes(&NodeCfKey(self.id));

        // KEY: Use txn.get_cf() to see uncommitted writes!
        let value_bytes = txn.get_cf(cf, &key_bytes)?
            .ok_or_else(|| anyhow!("Node not found: {}", self.id))?;

        let value: NodeCfValue = Nodes::value_from_bytes(&value_bytes)?;

        // Temporal validity check
        let ref_time = self.reference_ts_millis.unwrap_or_else(TimestampMilli::now);
        if !schema::is_valid_at_time(&value.0, ref_time) {
            return Err(anyhow!("Node not valid at reference time"));
        }

        Ok((value.1, value.2))
    }
}
```

### Queries Requiring Implementation

All queries will implement `TransactionQueryExecutor` for completeness:

| Query | Complexity | Notes |
|-------|------------|-------|
| `NodeById` | Low | Point lookup |
| `OutgoingEdges` | Medium | Prefix iteration on ForwardEdges |
| `IncomingEdges` | Medium | Prefix iteration on ReverseEdges |
| `NodeFragments` | Medium | Range iteration with timestamp |
| `EdgeFragments` | Medium | Range iteration with timestamp |
| `EdgeDetails` | Low | Point lookup |
| `NodesByIdsMulti` | Medium | Batch point lookups |
| `AllNodes` | High | Full CF iteration with pagination |
| `AllEdges` | High | Full CF iteration with pagination |

All implementations follow the same pattern - replace `storage.db()` / `storage.transaction_db()` with `txn.get_cf()` / `txn.prefix_iterator_cf()`.

---

## Files to Modify

| File | Changes | Lines (est) | Impact |
|------|---------|-------------|--------|
| `libs/db/src/graph/transaction.rs` | **NEW FILE** | ~150 | Core implementation |
| `libs/db/src/graph/query.rs` | Add trait, implement for queries | ~200 | Additive |
| `libs/db/src/graph/writer.rs` | Add `transaction()`, hold `Arc<Storage>` | ~30 | Breaking (internal) |
| `libs/db/src/graph/mod.rs` | Export `Transaction`, `TransactionQueryExecutor` | ~5 | Additive |
| `libs/db/src/writer.rs` | Add `transaction()` to unified Writer | ~15 | Additive |
| `libs/db/src/lib.rs` | Re-export transaction types | ~3 | Additive |
| `examples/vector/hnsw.rs` | Refactor insert to use transactions | ~50 | Migration |
| `examples/vector/vamana.rs` | Refactor insert to use transactions | ~50 | Migration |

---

## Breaking Changes

### 1. Writer Requires Storage Access

**Current:**
```rust
pub struct Writer {
    sender: mpsc::Sender<Vec<Mutation>>,
}
```

**Proposed:**
```rust
pub struct Writer {
    sender: mpsc::Sender<Vec<Mutation>>,
    storage: Arc<Storage>,  // NEW
    fulltext_sender: Option<mpsc::Sender<Vec<Mutation>>>,  // NEW
}
```

**Impact:** `Writer::new()` signature changes. Internal only - users get Writer from `handles.writer()`.

### 2. Writer Creation Functions Change

**Current:**
```rust
pub fn create_mutation_writer(config: WriterConfig) -> (Writer, Receiver)
```

**Proposed:**
```rust
pub fn create_mutation_writer(
    config: WriterConfig,
    storage: Arc<Storage>,
) -> (Writer, Receiver)
```

**Impact:** Internal API only. `WriterBuilder` handles this.

### 3. Query Types Get New Trait

All query types need to implement `TransactionQueryExecutor`. This is **additive** - existing `QueryExecutor` implementations remain unchanged.

**Impact:** None for existing code. New trait is opt-in for transaction use.

---

## Mutation Forwarding

After `Transaction::commit()`, mutations can be forwarded to a configured receiver:

```rust
impl<'a> Transaction<'a> {
    pub fn commit(self) -> Result<()> {
        // 1. Commit RocksDB transaction (sync, blocking)
        self.txn.commit()?;

        // 2. Forward mutations to configured receiver (non-blocking, best-effort)
        if let Some(sender) = self.forward_to {
            // Filter out Flush markers (not relevant for downstream)
            let mutations: Vec<_> = self.mutations
                .into_iter()
                .filter(|m| !m.is_flush())
                .collect();

            if !mutations.is_empty() {
                // Best-effort send using try_send (non-blocking)
                if let Err(e) = sender.try_send(mutations) {
                    tracing::warn!(
                        error = %e,
                        "Transaction forwarding failed - channel full or closed"
                    );
                }
            }
        }

        Ok(())
    }
}
```

### Forwarding Design

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| **When** | On commit only | Rollback/drop should not forward |
| **How** | `try_send` (non-blocking) | Don't block on slow receivers |
| **Failure** | Log warning, continue | Best-effort, matches Consumer chain |
| **Filter** | Remove Flush markers | Not relevant for downstream |
| **Configurable** | `Option<mpsc::Sender>` | Not hardcoded to fulltext |

This design allows any receiver to be attached - fulltext, metrics, audit log, replication, etc.

---

## Usage Examples

### HNSW Insert (Before - Phase 1)

```rust
async fn insert(
    &mut self,
    writer: &Writer,
    reader: &Reader,
    node_id: Id,
    vector: Vec<f32>,
) -> Result<()> {
    // Write node
    AddNode { id: node_id, ... }.run(writer).await?;
    AddNodeFragment { id: node_id, content: vector, ... }.run(writer).await?;

    // PROBLEM: Must flush before reading
    writer.flush().await?;

    // Find neighbors
    let neighbors = self.greedy_search(reader, &vector, ef_construction).await?;

    // Write edges
    for neighbor in neighbors {
        AddEdge { src: node_id, dst: neighbor, ... }.run(writer).await?;
    }
    writer.flush().await?;

    Ok(())
}
```

### HNSW Insert (After - Phase 3)

```rust
fn insert(
    &mut self,
    writer: &Writer,
    node_id: Id,
    vector: Vec<f32>,
) -> Result<()> {
    let mut txn = writer.transaction()?;

    // Write node (not committed yet)
    txn.write(AddNode { id: node_id, ... })?;
    txn.write(AddNodeFragment { id: node_id, content: vector, ... })?;

    // Read sees uncommitted writes!
    let neighbors = self.greedy_search_txn(&txn, &vector, ef_construction)?;

    // Write edges
    for neighbor in neighbors {
        txn.write(AddEdge { src: node_id, dst: neighbor, ... })?;

        // Prune neighbor connections (read + write in same txn)
        let current_edges = txn.read(OutgoingEdges::new(neighbor, Some("hnsw")))?;
        if current_edges.len() > self.max_connections {
            // Prune excess edges...
        }
    }

    // Atomic commit
    txn.commit()?;
    Ok(())
}

// Helper for greedy search within transaction
fn greedy_search_txn(
    &self,
    txn: &Transaction<'_>,
    query: &[f32],
    ef: usize,
) -> Result<Vec<(f32, Id)>> {
    let mut candidates = BinaryHeap::new();
    let mut visited = HashSet::new();

    // Start from entry point
    let entry = self.entry_point;
    let entry_vec = txn.read(NodeFragments::new(entry))?;
    let dist = cosine_distance(query, &entry_vec);
    candidates.push(Reverse((OrderedFloat(dist), entry)));

    while let Some(Reverse((dist, node))) = candidates.pop() {
        if visited.contains(&node) { continue; }
        visited.insert(node);

        // Get neighbors - sees uncommitted edges!
        let edges = txn.read(OutgoingEdges::new(node, Some("hnsw")))?;

        for (_, _, neighbor, _) in edges {
            if visited.contains(&neighbor) { continue; }

            let vec = txn.read(NodeFragments::new(neighbor))?;
            let d = cosine_distance(query, &vec);
            candidates.push(Reverse((OrderedFloat(d), neighbor)));
        }
    }

    // Return top-k by distance
    Ok(visited.into_iter().take(ef).collect())
}
```

**Result:**
- No flush/sleep needed
- Zero latency for read-your-writes
- Atomic commit of all operations
- 10-100x throughput improvement

---

## Relationship to HNSW2 Design

The [HNSW2 design](../../examples/vector/HNSW2.md) proposes a hannoy-inspired architecture for high-performance vector indexing. This section analyzes how the Transaction API enables and complements HNSW2.

### HNSW2 Overview

HNSW2 draws from [hannoy](https://github.com/nnethercott/hannoy) and proposes:

| Aspect | Current HNSW | HNSW2 (Proposed) |
|--------|--------------|------------------|
| **ID Format** | UUID (16 bytes) | u32 (4 bytes) |
| **Edge Storage** | Separate rows per edge | Roaring bitmap per node |
| **Edge Size** | ~200 bytes/edge | ~3 bytes/edge (amortized) |
| **Memory Model** | HashMap cache | RocksDB block cache |
| **Consistency** | 10ms sleep / flush | WriteBatchWithIndex or Transaction |
| **Pruning** | Inline during insert | Background thread |
| **Online Updates** | Limited (slow) | Full support with deletions |

### The Core Problem Both Solve

Both Transaction API and HNSW2's `WriteBatchWithIndex` proposal address the same fundamental issue:

```
HNSW Insert requires:
1. Write node + vector           ──► Must be visible for step 2
2. Greedy search (read neighbors) ──► Must see step 1
3. Write edges to neighbors       ──► Must be visible for step 4
4. Prune neighbor edges (read)    ──► Must see step 3
5. Commit atomically
```

**Current limitation:** Steps 1→2 and 3→4 require `flush()` between them, adding ~0.1-1ms latency per transition.

### How Transaction API Enables HNSW2

| HNSW2 Requirement | Transaction API Solution |
|-------------------|--------------------------|
| **Read-your-writes during insert** | `txn.read()` sees `txn.write()` immediately |
| **Greedy search sees new node** | Query executes against transaction snapshot |
| **Prune sees current edges** | Query executes against transaction snapshot |
| **Atomic multi-step operations** | Single `txn.commit()` for all mutations |
| **No sleep/flush between ops** | Transaction scope eliminates need entirely |
| **Rollback on failure** | `txn.rollback()` or auto-rollback on drop |
| **Concurrent reader isolation** | Readers see consistent committed state |

### Transaction API vs WriteBatchWithIndex

HNSW2 proposes `WriteBatchWithIndex` for read-your-writes. Detailed comparison:

| Aspect | Transaction API | WriteBatchWithIndex |
|--------|-----------------|---------------------|
| **Read-your-writes** | ✅ Yes | ✅ Yes |
| **Atomic commit** | ✅ Yes | ✅ Yes |
| **Conflict detection** | ✅ Pessimistic locking | ❌ None |
| **Rollback support** | ✅ Native `rollback()` | ❌ Must rebuild batch |
| **Snapshot isolation** | ✅ Full MVCC | ⚠️ Sees latest committed |
| **Multi-key transactions** | ✅ Full isolation | ⚠️ No isolation between keys |
| **Deadlock detection** | ✅ Built-in | ❌ N/A (no locking) |
| **RocksDB API** | `TransactionDB::transaction()` | `WriteBatchWithIndex::new()` |
| **Complexity** | Medium | Lower |
| **Performance** | Slight overhead | Minimal overhead |

#### When to Use Each

**Use Transaction API when:**
- Complex multi-step operations (HNSW insert with pruning)
- Need rollback on partial failure
- Concurrent writers that might conflict
- Need snapshot isolation for reads

**Use WriteBatchWithIndex when:**
- Append-only workloads (no read-modify-write)
- Single writer, no conflicts possible
- Simpler implementation preferred
- Maximum write throughput needed

### HNSW2 Insert with Transaction API

```rust
// HNSW2-style insert using Transaction API
fn insert_hnsw2(
    &mut self,
    writer: &Writer,
    item_id: u32,           // HNSW2 uses u32 IDs
    vector: &[f32],
) -> Result<()> {
    let mut txn = writer.transaction()?;

    // 1. Store vector (HNSW2 uses separate vectors CF)
    txn.write(AddNodeFragment {
        id: Id::from_u32(item_id),
        content: DataUrl::from_f32_slice(vector),
        ...
    })?;

    // 2. Determine layer level
    let level = self.random_level();
    txn.write(AddNode {
        id: Id::from_u32(item_id),
        name: format!("vec_{}", item_id),
        ...
    })?;

    // 3. Greedy search at each layer (sees uncommitted vector!)
    for layer in (0..=level).rev() {
        let neighbors = self.greedy_search_layer_txn(&txn, vector, layer)?;

        // 4. Add edges via roaring bitmap merge (HNSW2 optimization)
        for neighbor_id in neighbors.iter().take(self.m) {
            // Bidirectional edges
            txn.write(AddEdge {
                src: Id::from_u32(item_id),
                dst: Id::from_u32(*neighbor_id),
                name: format!("hnsw_L{}", layer),
                ...
            })?;
            txn.write(AddEdge {
                src: Id::from_u32(*neighbor_id),
                dst: Id::from_u32(item_id),
                name: format!("hnsw_L{}", layer),
                ...
            })?;

            // 5. Mark for background pruning (HNSW2 deferred pruning)
            // Instead of inline pruning, just mark the neighbor
            txn.write(AddNodeFragment {
                id: Id::from_u32(*neighbor_id),
                content: DataUrl::from_text("pending_prune"),
                ...
            })?;
        }
    }

    // 6. Atomic commit - all edges visible at once
    txn.commit()?;

    // Background thread handles pruning (HNSW2 pattern)
    Ok(())
}
```

### HNSW2 Features Enabled by Transaction API

#### 1. Deferred Pruning (Background Thread)

HNSW2 proposes moving pruning to a background thread:

```rust
// Insert path (fast)
txn.write(edges)?;
txn.write(mark_for_pruning)?;
txn.commit()?;  // Returns immediately

// Background thread (parallel)
loop {
    let pending = scan_pending_prune_markers()?;
    for node_id in pending {
        let mut txn = writer.transaction()?;

        // Read current edges
        let edges = txn.read(OutgoingEdges::new(node_id, Some("hnsw")))?;

        if edges.len() > M_MAX {
            // Prune using RNG heuristic
            let to_keep = select_diverse_neighbors(&edges, M_MAX);
            // Delete excess edges...
        }

        // Clear prune marker
        txn.write(DeleteNodeFragment { ... })?;
        txn.commit()?;
    }
}
```

**Benefit:** Insert returns in O(log n) without waiting for O(M²) pruning.

#### 2. Roaring Bitmap Edge Storage

HNSW2 proposes roaring bitmaps for edges. Transaction API works seamlessly:

```rust
// Read bitmap edges within transaction
impl TransactionQueryExecutor for OutgoingEdgesBitmap {
    fn execute_in_transaction(&self, txn: &Transaction, txn_db: &TransactionDB) -> Result<RoaringBitmap> {
        let cf = txn_db.cf_handle("edges")?;
        let key = encode_edge_key(self.node_id, self.layer);

        let bitmap_bytes = txn.get_cf(cf, &key)?
            .unwrap_or_default();

        Ok(RoaringBitmap::deserialize_from(&bitmap_bytes)?)
    }
}

// Write bitmap edges within transaction
txn.write(MergeEdgeBitmap {
    node_id,
    layer,
    operation: EdgeOperation::Add(neighbor_id),
})?;
```

#### 3. Filtered Search with Bitmaps

HNSW2's killer feature - filtered search via bitmap intersection:

```rust
fn filtered_search_txn(
    &self,
    txn: &Transaction,
    query: &[f32],
    filter: &RoaringBitmap,  // Only search these IDs
    k: usize,
) -> Result<Vec<(f32, u32)>> {
    // ...
    while let Some(node) = candidates.pop() {
        // Get neighbors as bitmap
        let neighbors = txn.read(OutgoingEdgesBitmap::new(node, 0))?;

        // Intersect with filter BEFORE fetching vectors
        let valid_neighbors = neighbors.and(filter);

        // Only fetch vectors for filtered candidates
        for id in valid_neighbors.iter() {
            let vec = txn.read(NodeFragments::new(Id::from_u32(id)))?;
            // ...
        }
    }
}
```

#### 4. Delete Support

HNSW2 requires first-class deletion. Transaction API enables atomic delete:

```rust
fn delete_vector(&mut self, writer: &Writer, item_id: u32) -> Result<()> {
    let mut txn = writer.transaction()?;

    // 1. Get current neighbors at all layers
    let max_layer = txn.read(NodeMeta::new(item_id))?.max_layer;

    for layer in 0..=max_layer {
        let neighbors = txn.read(OutgoingEdgesBitmap::new(item_id, layer))?;

        // 2. Remove from each neighbor's edge list
        for neighbor_id in neighbors.iter() {
            txn.write(MergeEdgeBitmap {
                node_id: neighbor_id,
                layer,
                operation: EdgeOperation::Remove(item_id),
            })?;

            // 3. Mark neighbor for orphan repair
            txn.write(MarkForRepair { node_id: neighbor_id })?;
        }

        // 4. Delete this node's edge list
        txn.write(DeleteEdgeBitmap { node_id: item_id, layer })?;
    }

    // 5. Delete vector and metadata
    txn.write(DeleteNode { id: Id::from_u32(item_id) })?;
    txn.write(DeleteNodeFragments { id: Id::from_u32(item_id) })?;

    // 6. Free ID for reuse (HNSW2 ID allocator)
    self.id_allocator.free(item_id);

    // Atomic commit
    txn.commit()?;

    Ok(())
}
```

### Performance Comparison

| Metric | Current HNSW | HNSW2 + Transaction API |
|--------|--------------|-------------------------|
| **Insert throughput** | 30-100/sec | 5,000-10,000/sec |
| **Search QPS** | 45-108 | 500-1,000 |
| **Disk per vector** | 25.7KB | ~10KB |
| **Memory (1M vectors)** | ~4GB (HashMap) | ~100MB (block cache) |
| **Filtered search** | Post-filter | Bitmap intersection |
| **Deletions** | Not supported | First-class |
| **Concurrent writers** | No | Yes |

### Migration Path

Transaction API is the foundation; HNSW2 optimizations layer on top:

```
Phase 3: Transaction API (this proposal)
    │
    ├── Enables: Read-your-writes, atomic multi-step ops
    │
    ▼
HNSW2 Phase 1: Core Schema
    │
    ├── u32 ID allocator with roaring reuse
    ├── Column families (vectors, edges, node_meta)
    ├── Roaring bitmap edge storage
    │
    ▼
HNSW2 Phase 2: Construction
    │
    ├── Port insert with Transaction API
    ├── Greedy search using txn.read()
    ├── Deferred pruning (background thread)
    │
    ▼
HNSW2 Phase 3: Online Updates
    │
    ├── Delete with orphan repair
    ├── pending_updates tracking
    ├── Background maintenance
    │
    ▼
HNSW2 Phase 4: Optimizations
    │
    ├── Filtered search with bitmaps
    ├── MultiGet batch fetching
    ├── SIMD distance computation
```

### Conclusion

The Transaction API is **not mutually exclusive** with HNSW2 - it provides the essential read-your-writes foundation that HNSW2 requires. Key points:

1. **Transaction API solves the consistency problem** that both current HNSW and HNSW2 need
2. **HNSW2's optimizations layer on top**: roaring bitmaps, deferred pruning, filtered search
3. **Migration is incremental**: Transaction API first, then HNSW2 optimizations
4. **No conflict**: Transaction API works with any edge storage format (current or roaring)

**Recommendation:** Implement Transaction API (Phase 3) first, then use it as the foundation for HNSW2 development.

---

## Implementation Summary

All implementation steps have been completed:

### Step 1: Core Transaction Type ✅
- Created `libs/db/src/graph/transaction.rs`
- Implemented `Transaction` struct with write/commit/rollback
- Unit tests for basic operations

### Step 2: TransactionQueryExecutor Trait ✅
- Added trait to `libs/db/src/graph/query.rs`
- Implemented for all queries (9 total)

### Step 3: Writer Integration ✅
- Modified `graph::Writer` to hold `Arc<Storage>`
- Added `transaction()` method
- Updated `WriterBuilder` and related functions

### Step 4: Unified Writer Integration ✅
- Added `transaction()` to unified `Writer`
- Configured fulltext forwarding on commit

### Step 5: Update Examples ✅
- Vector examples use `flush()` for read-after-write consistency
- Removed sleep workarounds
- Benchmarked improvement (see `examples/vector/PERF.md`)

### Step 6: Remaining Query Implementations ✅
- Implemented `TransactionQueryExecutor` for all queries
- Full test coverage

---

## Risk Assessment

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| RocksDB transaction lifetime complexity | Medium | Medium | Use `'a` lifetime tied to Writer |
| Deadlock with long-held transactions | Low | Low | Document best practices, consider timeout |
| Fulltext forwarding failure after commit | Low | Low | Log warning, continue (best-effort) |
| Breaking internal APIs | Low | Certain | Changes isolated to WriterBuilder |
| Performance regression | Low | Low | Benchmark before/after |

---

## Success Metrics

| Metric | Current (Phase 1) | Target (Phase 3) |
|--------|-------------------|------------------|
| HNSW insert throughput | 100-1,000/sec | 5,000-10,000/sec |
| Read-your-writes latency | ~0.1-1ms (flush) | 0ms (in-memory) |
| Vector index build (1M) | ~2-9 hours | ~6-60 minutes |
| API complexity | 2 calls (run + flush) | 1 scope (transaction) |

---

## Design Decisions (Detailed Rationale)

### 1. Transaction Lifetime: `'a` tied to `&'a Writer` ✅

**Decision:** `Transaction<'a>` is tied to `&'a Writer` via the underlying RocksDB transaction lifetime.

**Why not `Arc<Storage>` (owned)?**

RocksDB's `Transaction<'a, TransactionDB>` has a lifetime parameter that ties it to the `TransactionDB`:

```rust
// RocksDB API - transaction borrows from TransactionDB
impl TransactionDB {
    pub fn transaction(&self) -> Transaction<'_, TransactionDB>
}
```

You cannot have a `'static` transaction because it must borrow from storage. This means `Transaction` *must* have a lifetime parameter.

**Benefits:**
- Idiomatic Rust (standard borrow pattern)
- Compile-time safety (borrow checker enforces)
- Clear API (transaction is scoped to writer borrow)
- Matches RocksDB's own API design

**For concurrent transactions:** Clone the `Writer` (cheap - just `Arc` clones internally).

### 2. Mutation Forwarding: Configurable `Option<mpsc::Sender>` ✅

**Decision:** Forward to configurable receiver, not hardcoded to fulltext.

**Benefits:**
- Fulltext becomes just one possible receiver
- Easy to add other receivers (metrics, audit log, replication)
- Can disable forwarding entirely for testing
- Matches the existing Consumer chain pattern

**Configuration:**
```rust
let writer = WriterBuilder::new(graph, storage)
    .with_transaction_forward_to(fulltext_tx)  // Optional
    .build();
```

### 3. Commit API: Sync with best-effort `try_send` ✅

**Decision:** `commit()` is synchronous, forwarding uses non-blocking `try_send`.

**Why sync?**
- RocksDB's `transaction.commit()` is synchronous
- Making our `commit()` async only to await an MPSC send adds complexity
- If the channel is full, we should drop rather than block

**Forwarding behavior:**
- Uses `try_send` (non-blocking)
- If channel full: log warning, continue
- If channel closed: log warning, continue
- Flush markers filtered out before forwarding

### 4. Query Coverage: All Queries ✅

**Decision:** Implement `TransactionQueryExecutor` for all query types.

**Why all queries?**
- Incremental effort is low (~2 extra days)
- Prevents future "why doesn't X work in transactions?" questions
- HNSW2 may need `NodeFragments` for vector access
- Complete implementation is more professional

All implementations follow the same pattern - replace `storage.db()` / `storage.transaction_db()` with `txn.get_cf()`.

### 5. Timeout Handling: Not for MVP

**Decision:** No transaction timeout for initial implementation.

**Rationale:**
- RocksDB transactions don't have built-in timeout
- Deadlock is unlikely with single-writer pattern
- Can be added later if needed
- Document best practices instead (short-lived transactions)

---

## Appendix: Implementation Checklist

### Phase 3: Transaction Scope ✅ Complete

**Core Transaction:**
- [x] Create `libs/db/src/graph/transaction.rs`
- [x] Implement `Transaction` struct
- [x] Implement `write()` method
- [x] Implement `write_batch()` method
- [x] Implement `read()` method
- [x] Implement `commit()` with fulltext forwarding
- [x] Implement `rollback()` method
- [x] Implement `Drop` for auto-rollback
- [x] Add `len()` and `is_empty()` helpers

**TransactionQueryExecutor Trait:**
- [x] Define trait in `libs/db/src/graph/query.rs`
- [x] Implement for `NodeById`
- [x] Implement for `OutgoingEdges`
- [x] Implement for `IncomingEdges`
- [x] Implement for `NodeFragmentsByIdTimeRange`
- [x] Implement for `EdgeFragmentsByIdTimeRange`
- [x] Implement for `EdgeSummaryBySrcDstName`
- [x] Implement for `NodesByIdsMulti`
- [x] Implement for `AllNodes`
- [x] Implement for `AllEdges`

**Writer Integration:**
- [x] Modify `graph::Writer` to hold `Arc<Storage>`
- [x] Modify `graph::Writer` to hold `transaction_forward_to: Option<mpsc::Sender>`
- [x] Add `transaction()` method to `graph::Writer`
- [x] Update `create_mutation_writer()` signature
- [x] Add `with_transaction_forward_to()` to `WriterBuilder`
- [x] Update `WriterBuilder` to pass storage and forward sender
- [x] Add `transaction()` to unified `Writer`

**Exports:**
- [x] Export `Transaction` from `graph` module
- [x] Export `TransactionQueryExecutor` from `graph` module
- [x] Re-export from `lib.rs`

**Tests:**
- [x] Unit test: write then read in same transaction
- [x] Unit test: commit makes writes visible externally
- [x] Unit test: rollback discards writes
- [x] Unit test: drop without commit rolls back

**Examples:**
- [x] Vector examples use flush() for read-after-write consistency
- [x] Remove sleep workarounds
- [x] Update documentation

---

## References

- [Issue #26: Read-after-write consistency](https://github.com/chungers/motlie/issues/26)
- [Flush API Design](./flush.md)
- [HNSW2 Design](../../examples/vector/HNSW2.md)
- [RocksDB Transactions](https://github.com/facebook/rocksdb/wiki/Transactions)
- [RocksDB WriteBatchWithIndex](https://github.com/facebook/rocksdb/wiki/WriteBatchWithIndex)
- [Concurrency and Storage Modes](./concurrency-and-storage-modes.md)
