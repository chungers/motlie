# TODO - Unified Writer API

This document tracks open design questions and future improvements for the unified writer API.

## Open Questions

### 1. Mutation Confirmation

**Status**: Open

**Question**: Should mutations return confirmation that they were persisted?

Currently `writer::Runnable::run()` returns `Result<()>` which only indicates the mutation was sent to the channel, not that it was persisted to storage.

**Options**:

A. **Fire-and-forget (current)**: `run()` returns when mutation is queued
   - Pros: Simple, fast, non-blocking
   - Cons: No confirmation of persistence

B. **Confirmation channel**: Add optional oneshot channel for completion notification
   ```rust
   // Option B: With confirmation
   AddNode { ... }.run_confirmed(&writer).await?;  // Waits for persistence
   ```
   - Pros: Guarantees durability before returning
   - Cons: Higher latency, more complex

C. **Flush method**: Keep fire-and-forget but add `writer.flush().await?`
   ```rust
   AddNode { ... }.run(&writer).await?;
   writer.flush().await?;  // Wait for all pending mutations to complete
   ```
   - Pros: Batching-friendly, explicit sync points
   - Cons: Doesn't confirm specific mutations

**Use Cases to Consider**:
- Testing: Need to read-after-write
- Critical mutations: Need durability guarantee before proceeding
- Bulk imports: Fire-and-forget is preferred for throughput

---

### 2. Batch vs Streaming Mutations

**Status**: Open

**Question**: Should the unified writer support both batch and streaming patterns?

**Current State**:
- `MutationBatch` allows sending multiple mutations atomically
- Individual mutations use `run(&writer)` pattern

**Options**:

A. **Keep current approach**: Batch via `MutationBatch`, individual via `run()`
   ```rust
   // Individual
   AddNode { ... }.run(&writer).await?;

   // Batch
   MutationBatch::new()
       .add_node(...)
       .add_edge(...)
       .run(&writer)
       .await?;
   ```

B. **Add streaming API**: Writer accepts stream of mutations
   ```rust
   let stream = async_stream::stream! {
       for item in items {
           yield AddNode { ... };
       }
   };
   writer.send_stream(stream).await?;
   ```

C. **Add transaction API**: Explicit transaction boundaries
   ```rust
   let txn = writer.begin().await?;
   txn.add_node(...)?;
   txn.add_edge(...)?;
   txn.commit().await?;
   ```

**Considerations**:
- Atomicity: Should batched mutations be atomic (all-or-nothing)?
- Backpressure: How to handle slow consumers?
- Error recovery: What happens if one mutation in a batch fails?

---

## Resolved Questions

### 3. Error Handling in Pipeline (RESOLVED)

**Decision**: Log error and continue

If fulltext indexing fails after graph write succeeds:
- Log the error at WARN level
- Continue processing subsequent mutations
- Do NOT rollback the graph write (would require complex distributed transaction)

**Rationale**:
- Graph is source of truth; fulltext is a derived index
- Fulltext can be rebuilt from graph if needed
- Simpler architecture, more predictable behavior

### 4. Read-After-Write in StorageHandle (RESOLVED)

**Decision**: Yes, provide both `writer()` and `reader()` on `StorageHandle`

**Implementation**:
- Both writer and reader share the same `Arc<graph::Graph>` (and thus same RocksDB `TransactionDB`)
- RocksDB `TransactionDB` supports concurrent reads and writes
- Tests should wait for writes to flush before reading

**Design**:
```rust
pub struct StorageHandle {
    writer: Writer,
    reader: Reader,
    handles: Vec<JoinHandle<()>>,
}

impl StorageHandle {
    pub fn writer(&self) -> &Writer { &self.writer }
    pub fn reader(&self) -> &Reader { &self.reader }
    pub async fn shutdown(self) -> Result<()> { ... }
}
```

**Shared Storage Pattern**:
```
                    ┌─────────────────────────┐
                    │   Arc<graph::Graph>     │
                    │   (TransactionDB)       │
                    └───────────┬─────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
    ┌─────────────────┐ ┌─────────────┐ ┌─────────────────┐
    │ MutationConsumer│ │QueryConsumer│ │ QueryConsumer   │
    │   (writes)      │ │  (reads)    │ │   (reads)       │
    └─────────────────┘ └─────────────┘ └─────────────────┘
```

---

## Future Improvements

- [ ] Implement unified `writer::Storage` and `StorageHandle`
- [ ] Add `motlie_db::mutation` module with re-exports
- [ ] Update documentation and examples
- [ ] Add integration tests for read-after-write scenarios
