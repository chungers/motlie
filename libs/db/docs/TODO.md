# TODO - DB API Consistency

This document tracks open API consistency and integration work across the root crate (`motlie_db`) and subsystem crates (`graph`, `fulltext`, `vector`).

## Current Behavior Snapshot (Verified 2026-03-07)

### 1. Mutation Confirmation and Durability Semantics (Resolved + Clarified)

- `Runnable::run()` is enqueue-only for graph/root and vector mutation APIs. `Result<()>` means "accepted by writer channel", not "persisted."
- Confirmation paths are already implemented:
  - `Writer::flush()` waits until prior graph mutations are committed.
  - `Writer::send_sync()` performs send + flush.
  - `graph::mutation::RunnableWithResult::run_with_result(..., ExecOptions)` waits for execution and returns typed replies.
  - `vector::mutation::RunnableWithResult::run_with_result(...)` waits for execution and returns typed replies (no options parameter).
- Graph mutation replies now include version data for versioned operations:
  - node mutations return `(Id, Version)` for add/update/delete/restore.
  - edge mutations return `Version` for add/update/delete/restore.

### 2. Batch vs Transaction Semantics (Resolved + Clarified)

- Batch execution via `Vec<Mutation>` exists and is atomic at the graph layer (single RocksDB transaction in `graph::Processor::process_mutations_with_options`).
- Transaction API is available via `writer.transaction()` (graph transaction scope).
- Remaining caveat: atomicity is graph-scoped; fulltext indexing is still asynchronous/best-effort after graph commit.

### 3. Pipeline Error Behavior (Clarified)

- Graph commit is source-of-truth and is not rolled back if downstream indexing fails.
- Graph -> fulltext forwarding is non-blocking `try_send` and can drop on backpressure/closed channel.
- Fulltext consumer returns an error on indexing failure; that task exits unless externally restarted.
- Effective contract today is best-effort derived indexing, not strict acknowledged indexing.

### 4. Read-After-Write Handle Model (Resolved)

- Legacy `StorageHandle` naming is obsolete.
- Current root API:
  - `Storage<ReadWrite>::ready(...) -> ReadWriteHandles` (`writer()` + `reader()`).
  - `Storage<ReadOnly>::ready(...) -> ReadOnlyHandles` (`reader()` only).
- Read-after-write visibility for graph queries is achieved by calling `writer.flush()` or `writer.send_sync()` before querying.
- Fulltext visibility is eventual; `flush()`/`send_sync()` do not guarantee fulltext indexing completion.

---

## Completed Improvements

- [x] Implement typed root storage handles (`Storage<ReadWrite>` + `ReadWriteHandles`, plus read-only mode)
- [x] Add `motlie_db::mutation` module with root re-exports
- [x] Update documentation and examples (root rustdocs + `libs/db/docs/getting-started.md` + runnable examples in `libs/db/examples/`)
- [x] Add integration tests for read-after-write scenarios (e.g., `libs/db/tests/test_transaction_api.rs`)

---

## API Consistency Audit (Re-validated 2026-03-07)

This section captures findings from a cross-subsystem API review focused on:

- root crate (`motlie_db`) unified API shape
- subsystem API consistency (`graph`, `vector`, `fulltext`)
- integration path from graph writes -> fulltext/vector queryability

### Key Findings

#### Critical

1. Root "unified" API is graph+fulltext only, not graph+fulltext+vector.
   - Root unified storage initializes only graph/fulltext paths:
     - `libs/db/src/storage.rs` (`<base>/graph`, `<base>/fulltext`)
   - Root unified query facade covers graph+fulltext; root mutation facade is graph-only:
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

#### Low (Resolved Since Prior Audit)

9. Root writer stale-doc issue is fixed.
   - `libs/db/src/writer.rs` now uses non-optional `handles.writer()`.
   - `scripts/check_db_doc_drift.sh` now guards this regression pattern.

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
- [x] Fix stale root API docs/examples to match current signatures.

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
    .map(|(_, _src_id, dst_id, edge_name, _version)| RestoreEdge {
        src_id: alice_id,
        dst_id: *dst_id,
        name: edge_name.clone(),
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
