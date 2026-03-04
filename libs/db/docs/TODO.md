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
- `Vec<Mutation>` allows sending multiple mutations atomically
- Individual mutations use `run(&writer)` pattern

**Options**:

A. **Keep current approach**: Batch via `Vec<Mutation>`, individual via `run()`
   ```rust
   // Individual
   AddNode { ... }.run(&writer).await?;

   // Batch
   Vec<Mutation>::new()
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

- [x] Implement unified `writer::Storage` and `StorageHandle` (delivered as typed root `Storage<ReadWrite>` + `ReadWriteHandles` in `libs/db/src/storage.rs`)
- [x] Add `motlie_db::mutation` module with re-exports
- [x] Update documentation and examples (root rustdocs + new `libs/db/docs/getting-started.md` + runnable examples in `libs/db/examples/`)
- [x] Add integration tests for read-after-write scenarios (e.g., `libs/db/tests/test_transaction_api.rs`)

---

## API Consistency Audit (2026-02-26)

This section captures findings from a cross-subsystem API review focused on:

- root crate (`motlie_db`) unified API shape
- subsystem API consistency (`graph`, `vector`, `fulltext`)
- integration path from graph writes -> fulltext/vector queryability

### Key Findings

#### Critical

1. Root "unified" API is graph+fulltext only, not graph+fulltext+vector.
   - Root unified storage initializes only graph/fulltext paths:
     - `libs/db/src/storage.rs` (`<base>/graph`, `<base>/fulltext`)
   - Root unified query/mutation enums only cover graph/fulltext:
     - `libs/db/src/query.rs`
     - `libs/db/src/mutation.rs`

2. Root mutation facade is incomplete vs graph module capabilities.
   - `motlie_db::mutation` re-exports add/update mutations but not delete/restore:
     - `libs/db/src/mutation.rs`
   - graph root exports delete/restore:
     - `libs/db/src/graph/mod.rs`

3. Graph -> fulltext propagation is best-effort (drop on backpressure), and there is no equivalent graph -> vector ingestion path in unified root API.
   - forwarding uses `try_send` and can drop:
     - `libs/db/src/graph/writer.rs`
     - `libs/db/src/graph/transaction.rs`

#### High

4. Read-write standalone subsystem opening is not symmetric with read-only for mixed CF databases.
   - read-only has extra-CF fallback:
     - `libs/db/src/rocksdb/storage.rs` (readonly path)
   - read-write does not:
     - `libs/db/src/rocksdb/storage.rs` (readwrite path)

5. Two different "root" storage topologies exist:
   - unified `Storage`: `<base>/graph`, `<base>/fulltext`
   - `StorageBuilder`: `<base>/rocksdb`, `<base>/tantivy`
   - See:
     - `libs/db/src/storage.rs`
     - `libs/db/src/storage_builder.rs`

#### Medium

6. Unified hydrated fulltext queries drop ranking provenance (`score`, `match_source`) from fulltext hits.
   - raw hit structs include score/source:
     - `libs/db/src/fulltext/search.rs`
   - hydrated outputs are tuples without score/source:
     - `libs/db/src/query.rs`

7. Visibility mismatch in unified query internals.
   - `Query::execute(&CompositeStorage)` and `Reader::storage()` are public-facing while `CompositeStorage` is `pub(crate)`:
     - `libs/db/src/query.rs`
     - `libs/db/src/reader.rs`

8. Graph/vector request options and typed result ergonomics are close but not fully aligned.
   - graph mutation run-with-result supports `ExecOptions`:
     - `libs/db/src/graph/mutation.rs`
   - vector mutation run-with-result has no options parameter:
     - `libs/db/src/vector/mutation.rs`

#### Low

9. Root writer docs contain stale `unwrap()` usage despite non-optional writer handles.
   - stale example:
     - `libs/db/src/writer.rs`
   - actual API:
     - `libs/db/src/storage.rs` (`ReadWriteHandles::writer() -> &Writer`)

### TODO: Recommended API Unification Work

- [ ] Define `RootStorageV2` contract that includes graph, vector, and fulltext as first-class components.
- [ ] Expand root `mutation` facade to include full graph mutation set (including delete/restore) and vector mutation facade.
- [ ] Add root `query` facade for vector operations (`SearchKNN`, embedding lookup/list/resolve) with consistent `Runnable` ergonomics.
- [ ] Introduce deterministic indexing contract from graph writes:
  - graph -> fulltext: selectable consistency mode (`best_effort` vs `acknowledged`)
  - graph -> vector: explicit ingestion API and lifecycle hooks
- [ ] Unify storage directory model across `Storage` and `StorageBuilder` (or clearly split into "simple" vs "composable" modes with shared terminology).
- [ ] Align read-only/read-write multi-CF behavior in generic RocksDB storage.
- [ ] Preserve ranking metadata in hydrated fulltext query outputs (or provide alternate typed result including score/match_source).
- [ ] Clean up public/private visibility boundaries in unified query plumbing.
- [ ] Standardize mutation `run_with_result` options across graph/vector modules.
- [ ] Fix stale root API docs/examples to match current signatures.

### Suggested Integration Tests to Add

- [ ] Root-level end-to-end test: graph write -> fulltext searchable with strict acknowledgment mode.
- [ ] Root-level end-to-end test: graph write -> vector searchable path for node/edge content embeddings.
- [ ] Mixed root query test combining graph traversal + fulltext rank + vector semantic rank under one `Storage` handle.
- [ ] Backpressure behavior test validating no silent data loss in strict indexing mode.

---

## Graph Module: Batch Restore Improvements

**Status**: Open (added 2026-02-08, post-unify-request-envelopes refactor)

The `RestoreEdges` batch mutation was removed in favor of `Vec<RestoreEdge>`. While this simplifies the API, two features were lost that may be worth re-adding:

### 1. Batch Restore Validation Report

**Previous API**:
```rust
let report = RestoreEdges {
    src_id,
    name: Some("likes".to_string()),
    as_of: TimestampMilli::now(),
    dry_run: true,
}.run(&writer).await?;

// RestoreEdgesReport provided:
// - candidates: number of deleted edges considered
// - restorable: number of edges that would be restored
// - skipped_no_version: edges with no version at as_of
```

**Current API**:
```rust
// dry_run works but returns Vec<MutationResult>, not a structured report
vec![RestoreEdge { ... }.into()]
    .run_with_result(&writer, ExecOptions { dry_run: true })
    .await?;
```

**Proposal**: Add `RestoreEdgesReport` query or helper that scans edges from a source and returns validation info without requiring manual `Vec<RestoreEdge>` construction.

### 2. Convenience Helper for Batch Edge Restore

**Previous API**:
```rust
RestoreEdges {
    src_id: alice_id,
    name: None,  // All edge names from this source
    as_of: timestamp,
    dry_run: false,
}.run(&writer).await?;
```

**Current API** (manual construction required):
```rust
// User must query edges, build Vec<RestoreEdge>, then run
let edges = OutgoingEdges::new(alice_id, None).run(&reader, timeout).await?;
let restores: Vec<Mutation> = edges.iter()
    .map(|e| RestoreEdge {
        src_id: alice_id,
        dst_id: e.dst_id,
        name: e.name.clone(),
        as_of: timestamp,
        expected_version: None,
    }.into())
    .collect();
restores.run(&writer).await?;
```

**Proposal**: Add helper function in `graph::mutation`:
```rust
/// Build RestoreEdge mutations for all outgoing edges from a source.
pub async fn restore_outgoing_edges(
    reader: &Reader,
    src_id: Id,
    name_filter: Option<&str>,
    as_of: TimestampMilli,
    timeout: Duration,
) -> Result<Vec<Mutation>>
```

This maintains the unified `Vec<Mutation>` pattern while providing ergonomic batch construction.
