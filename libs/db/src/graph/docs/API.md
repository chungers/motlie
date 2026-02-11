# Graph Module API Reference

**Module:** `motlie_db::graph`
**Purpose:** Content-addressed, versioned graph storage with async mutation/query pipelines

---

## Table of Contents

1. [Overview](#overview)
2. [Two API Styles](#two-api-styles)
3. [Complete Operation Catalog](#complete-operation-catalog)
4. [CRUD Operations](#crud-operations)
5. [Update Operations](#update-operations)
6. [Version Queries & Time-Travel](#version-queries--time-travel)
7. [Edge Retargeting](#edge-retargeting)
8. [Rollback Operations](#rollback-operations)
9. [Transactions (Read-Your-Writes)](#transactions-read-your-writes)
10. [Storage Modes](#storage-modes)
11. [Subsystem Lifecycle](#subsystem-lifecycle)
12. [Concurrency Guarantees](#concurrency-guarantees)

---

## Overview

The graph module provides a content-addressed, versioned graph database on top of RocksDB. It supports:

- **Nodes and Edges** with optional summaries and fragments
- **Time-travel** via `ValidSince/ValidUntil` and `VersionHistory`
- **Rollback** via `RestoreNode/RestoreEdge` (batch via `Vec<Mutation>`)
- **Content-addressed summaries** with orphan-index GC
- **Two API styles**: Sync (Processor) and Async (Writer/Reader)

See `ARCH.md` for current architecture and `VERSIONING.md` for temporal/rollback semantics.
(codex, 2026-02-10, decision: validated examples against current API; updated for `as_of`, `SystemTimeMillis`, and `ActiveTimeMillis` usage.)

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Processor** | Synchronous core (Storage + NameCache) for direct RocksDB operations |
| **Writer** | Async mutation interface via MPSC channel with backpressure |
| **Reader** | Async query interface via MPMC channel with worker pool |
| **Fragments** | Append-only timestamped content events |
| **Summaries** | Content-addressed summaries indexed by hash |
| **Version** | Monotonic counter incremented on each mutation |
| **ActivePeriod** | Business/application validity interval |
| **ValidSince/ValidUntil** | System time version history for time-travel |

---

## Two API Styles

The graph module provides two complementary API styles:

### Style 1: Synchronous Processor API (crate-internal)

Direct, blocking operations on RocksDB. `Processor` is `pub(crate)` and not
accessible from external crates — use the async Writer/Reader API (Style 2)
for public usage. Shown here for internal reference.

```rust
use motlie_db::graph::processor::Processor;
use motlie_db::graph::Storage;
use motlie_db::graph::mutation::{AddEdge, AddNode, Mutation};
use motlie_db::graph::schema::{EdgeSummary, NodeSummary};
use motlie_db::{Id, TimestampMilli};
use std::sync::Arc;

// Setup
let mut storage = Storage::readwrite(&db_path);
storage.ready()?;
let processor = Processor::new(Arc::new(storage));

// Single mutation
let mutation: Mutation = AddNode {
    id: Id::new(),
    ts_millis: TimestampMilli::now(),
    name: "Alice".to_string(),
    summary: NodeSummary::from_text("A person"),
    valid_range: None,
}.into();
processor.execute_mutation(&mutation)?;

// Batch mutations (atomic)
let mutations: Vec<Mutation> = vec![
    AddNode {
        id: Id::new(),
        ts_millis: TimestampMilli::now(),
        name: "Bob".to_string(),
        summary: NodeSummary::from_text("A person"),
        valid_range: None,
    }.into(),
    AddEdge {
        source_node_id: Id::new(),
        target_node_id: Id::new(),
        ts_millis: TimestampMilli::now(),
        name: "knows".to_string(),
        summary: EdgeSummary::from_text("Met recently"),
        weight: None,
        valid_range: None,
    }.into(),
];
processor.process_mutations(&mutations)?;
```

**Processor API Methods:**

| Method | Purpose |
|--------|---------|
| `Processor::new(storage)` | Create Processor with Storage + NameCache |
| `Processor::with_cache(storage, cache)` | Create with explicit NameCache |
| `execute_mutation(&mutation)` | Execute single mutation (sync) |
| `process_mutations(&[mutations])` | Execute batch atomically (sync) |
| `storage()` | Access underlying Storage |
| `name_cache()` | Access NameCache |
| `transaction_db()` | Access RocksDB TransactionDB |

**Query execution (direct):**

- `query.execute_on(&processor).await?`
- `query.execute_with_processor(&processor).await?` (for `Query` enum dispatch)

### Style 2: Async Writer/Reader API

Channel-based async operations with backpressure. Use when:
- You need async/await integration
- You want automatic batching and backpressure
- You're building a concurrent application

```rust
use motlie_db::graph::mutation::AddNode;
use motlie_db::graph::query::NodeById;
use motlie_db::graph::writer::spawn_mutation_consumer_with_storage;
use motlie_db::graph::reader::spawn_query_consumers_with_storage;
use motlie_db::writer::Runnable as MutationRunnable;
use motlie_db::reader::Runnable as QueryRunnable;
use std::sync::Arc;
use std::time::Duration;

// Setup storage, writer, and reader
let mut storage = motlie_db::graph::Storage::readwrite(&db_path);
storage.ready()?;
let storage = Arc::new(storage);

let (writer, _w_handle) = spawn_mutation_consumer_with_storage(
    storage.clone(),
    WriterConfig::default()
);
let (reader, _r_handles) = spawn_query_consumers_with_storage(
    storage.clone(),
    ReaderConfig::default(),
    4  // number of query workers
);

// Mutations use writer::Runnable trait
let node_id = Id::new();
AddNode {
    id: node_id,
    ts_millis: TimestampMilli::now(),
    name: "Alice".to_string(),
    summary: NodeSummary::from_text("A person"),
    valid_range: None,
}
.run(&writer)
.await?;

// Queries use reader::Runnable trait
let timeout = Duration::from_secs(5);
let (name, summary, version) = NodeById::new(node_id, None)
    .run(&reader, timeout)
    .await?;
```

**Typed Query Replies (QueryReply)**

Graph queries implement a `QueryReply` mapping that binds each query type
to its specific output variant. The user experience is unchanged (still
`Runnable::run`), but the mapping is now explicit on the type:

```rust
use motlie_db::graph::query::NodeById;
use motlie_db::reader::Runnable as QueryRunnable;
use std::time::Duration;

let timeout = Duration::from_secs(5);
let (name, summary, version) = NodeById::new(node_id, None)
    .run(&reader, timeout)
    .await?;
```

**Typed Mutation Replies (run_with_result)**

Mutations define their reply type on the mutation itself. The reply
includes `request_id` and `elapsed_time` for tracing/metrics.

```rust
use motlie_db::graph::mutation::{UpdateNode, ExecOptions, RunnableWithResult};

let reply = UpdateNode { /* ... */ }
    .run_with_result(&writer, ExecOptions::default())
    .await?;
let (id, version) = reply.payload;
```

**Architecture:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    graph::Processor (sync core)                  │
├─────────────────────────────────────────────────────────────────┤
│ Owned State:                                                     │
│   - Arc<Storage>              // RocksDB access                  │
│   - Arc<NameCache>            // NameHash ↔ String deduplication │
└─────────────────────────────────────────────────────────────────┘
                          │
         ┌────────────────┴────────────────┐
         │                                 │
    ┌────▼─────┐                     ┌─────▼────┐
    │  Writer  │ (async MPSC)        │  Reader  │ (async MPMC)
    │ Runnable │                     │ Runnable │
    └────┬─────┘                     └─────┬────┘
         │                                 │
    ┌────▼─────┐                     ┌─────▼────┐
    │ Consumer │                     │ Consumer │
    │ Pool (1) │                     │ Pool (N) │
    └──────────┘                     └──────────┘
```

---

## Complete Operation Catalog

### Mutation Types

| Mutation | Purpose | Key Fields |
|----------|---------|------------|
| `AddNode` | Create a new node | `id`, `ts_millis`, `name`, `summary`, `valid_range?` |
| `AddEdge` | Create edge between nodes | `source_node_id`, `target_node_id`, `ts_millis`, `name`, `summary`, `weight?`, `valid_range?` |
| `AddNodeFragment` | Append timestamped content to node | `id`, `ts_millis`, `content`, `valid_range?` |
| `AddEdgeFragment` | Append timestamped content to edge | `src_id`, `dst_id`, `edge_name`, `ts_millis`, `content`, `valid_range?` |
| `UpdateNode` | Update node (active_period and/or summary) | `id`, `expected_version`, `new_active_period?`, `new_summary?` |
| `UpdateEdge` | Update edge (weight, active_period, and/or summary) | `src_id`, `dst_id`, `name`, `expected_version`, `new_weight?`, `new_active_period?`, `new_summary?` |
| `DeleteNode` | Tombstone a node | `id`, `expected_version` |
| `DeleteEdge` | Tombstone an edge | `src_id`, `dst_id`, `name`, `expected_version` |
| `RestoreNode` | Restore node to prior time | `id`, `as_of`, `expected_version?` |
| `RestoreEdge` | Restore edge to prior time | `src_id`, `dst_id`, `name`, `as_of`, `expected_version?` |

### Mutation Execution Matrix (Sync vs Async, Dry Run vs Execute)

| API | Input | Output | `dry_run=false` | `dry_run=true` |
|-----|-------|--------|----------------|----------------|
| `Processor::process_mutations(&[Mutation])` | mutations | `Result<()>` | Executes + commits | N/A |
| `Processor::process_mutations_with_options(&[Mutation], ExecOptions)` | mutations + options | `Result<ReplyEnvelope<Vec<MutationResult>>>` | Executes + commits | Validates only (no commit) |
| `Processor::execute_mutation(&Mutation)` | single mutation | `Result<()>` | Executes + commits | N/A |
| `Processor::execute_mutation_with_options(&Mutation, ExecOptions)` | single + options | `Result<ReplyEnvelope<MutationResult>>` | Executes + commits | Validates only (no commit) |
| `Mutation::run(&Writer)` | single mutation | `Result<()>` | Executes (async) | N/A |
| `Vec<Mutation>::run(&Writer)` | batch | `Result<()>` | Executes (async) | N/A |
| `Writer::send(Vec<Mutation>)` | batch | `Result<()>` | Enqueue (fire-and-forget) | N/A |
| `Writer::send_sync(Vec<Mutation>)` | batch | `Result<()>` | Enqueue + wait for commit | N/A |
| `Vec<Mutation>::run_with_result(&Writer, ExecOptions)` | batch + options | `Result<ReplyEnvelope<Vec<MutationResult>>>` | Executes + commits | Validates only (no commit) |
| `Writer::send_with_result(Vec<Mutation>, ExecOptions)` | batch + options | `Result<ReplyEnvelope<Vec<MutationResult>>>` | Executes + commits | Validates only (no commit) |
| `Mutation::run_with_result(&Writer, ExecOptions)` | single + options | `Result<ReplyEnvelope<T>>` | Executes + commits | Validates only (no commit) |

**Reply semantics (`MutationResult`):**
- `AddNode` / `UpdateNode` / `DeleteNode` / `RestoreNode`: `(id, version)`
- `AddEdge` / `UpdateEdge` / `DeleteEdge` / `RestoreEdge`: `version`
- Fragment mutations return unit `()`

### Query Types

| Query | Purpose | Key Fields |
|-------|---------|------------|
| `NodeById` | Get node by ID | `id`, `reference_ts_millis?`, `as_of?` |
| `NodesByIdsMulti` | Get multiple nodes by IDs | `ids`, `reference_ts_millis?`, `as_of?` |
| `OutgoingEdges` | Get edges from a node | `src_id`, `reference_ts_millis?`, `as_of?` |
| `IncomingEdges` | Get edges to a node | `dst_id`, `reference_ts_millis?`, `as_of?` |
| `EdgeSummaryBySrcDstName` | Get edge by topology | `src_id`, `dst_id`, `name`, `reference_ts_millis?`, `as_of?` |
| `NodeFragmentsByIdTimeRange` | Get node fragments in time range | `id`, `start`, `end`, `reference_ts_millis?` |
| `EdgeFragmentsByIdTimeRange` | Get edge fragments in time range | `src_id`, `dst_id`, `name`, `start`, `end`, `reference_ts_millis?` |
| `AllNodes` | Scan all nodes | `limit?`, `cursor?`, `reference_ts_millis?` |
| `AllEdges` | Scan all edges | `limit?`, `cursor?`, `reference_ts_millis?` |
| `NodesBySummaryHash` | Reverse lookup by summary hash | `hash` |
| `EdgesBySummaryHash` | Reverse lookup by summary hash | `hash` |

---

## CRUD Operations

### Create Node

```rust
use motlie_db::graph::mutation::AddNode;
use motlie_db::graph::schema::NodeSummary;
use motlie_db::{Id, TimestampMilli};

let node_id = Id::new();
AddNode {
    id: node_id,
    ts_millis: TimestampMilli::now(),
    name: "Alice".to_string(),
    summary: NodeSummary::from_text("Software engineer at Acme Corp"),
    valid_range: None,  // No business validity constraint
}
.run(&writer)
.await?;
```

### Create Edge

```rust
use motlie_db::graph::mutation::AddEdge;
use motlie_db::graph::schema::EdgeSummary;

AddEdge {
    source_node_id: alice_id,
    target_node_id: bob_id,
    ts_millis: TimestampMilli::now(),
    name: "knows".to_string(),
    summary: EdgeSummary::from_text("Met at conference 2024"),
    weight: Some(0.8),  // Optional relationship strength
    valid_range: None,
}
.run(&writer)
.await?;
```

### Read Node

```rust
use motlie_db::graph::query::NodeById;
use std::time::Duration;

let timeout = Duration::from_secs(5);

// Get current version
let (name, _summary, version) = NodeById::new(alice_id, None)
    .run(&reader, timeout)
    .await?;

println!("Name: {}, Version: {}", name, version);
```

### Read Edges

```rust
use motlie_db::graph::query::{OutgoingEdges, IncomingEdges, EdgeSummaryBySrcDstName};

// Get all outgoing edges from Alice
let outgoing = OutgoingEdges::new(alice_id, None)
    .run(&reader, timeout)
    .await?;

for (_weight, src_id, dst_id, name, _version) in &outgoing {
    println!("{} --{}-> {}", src_id, name, dst_id);
}

// Get all incoming edges to Bob
let incoming = IncomingEdges::new(bob_id, None)
    .run(&reader, timeout)
    .await?;

// Get specific edge by topology
let (summary, weight, version) = EdgeSummaryBySrcDstName::new(alice_id, bob_id, "knows".to_string(), None)
    .run(&reader, timeout)
    .await?;
```

### Delete Node

```rust
use motlie_db::graph::mutation::DeleteNode;

// First, get current version for optimistic locking
let (_name, _summary, version) = NodeById::new(alice_id, None).run(&reader, timeout).await?;

DeleteNode {
    id: alice_id,
    expected_version: version,
}
.run(&writer)
.await?;
```

### Delete Edge

```rust
use motlie_db::graph::mutation::DeleteEdge;

// Get current edge version
let (_summary, _weight, version) = EdgeSummaryBySrcDstName::new(alice_id, bob_id, "knows".to_string(), None)
    .run(&reader, timeout)
    .await?;

DeleteEdge {
    src_id: alice_id,
    dst_id: bob_id,
    name: "knows".to_string(),
    expected_version: version,
}
.run(&writer)
.await?;
```

---

## Update Operations

Updates use the `Option<Option<T>>` pattern to distinguish:
- `None` = no change to this field
- `Some(None)` = clear/reset this field
- `Some(Some(value))` = set to specific value

### Update Node Summary

```rust
use motlie_db::graph::mutation::UpdateNode;
use motlie_db::graph::schema::NodeSummary;

let (_name, _summary, version) = NodeById::new(alice_id, None).run(&reader, timeout).await?;

UpdateNode {
    id: alice_id,
    expected_version: version,
    new_active_period: None,  // No change
    new_summary: Some(NodeSummary::from_text("Senior engineer at Acme Corp")),
}
.run(&writer)
.await?;
```

### Update Node Active Period

```rust
use motlie_db::graph::schema::ActivePeriod;
use motlie_db::ActiveTimeMillis;

UpdateNode {
    id: alice_id,
    expected_version: version,
    new_active_period: Some(ActivePeriod::active_between(
        ActiveTimeMillis(1_771_286_400_000),  // 2026-02-17T00:00:00Z
        ActiveTimeMillis(1_771_459_200_000),  // 2026-02-19T00:00:00Z
    )),
    new_summary: None,  // No change
}
.run(&writer)
.await?;
```

### Clear Node Active Period

```rust
UpdateNode {
    id: alice_id,
    expected_version: version,
    new_active_period: Some(None),  // Clear the active period
    new_summary: None,
}
.run(&writer)
.await?;
```

### Update Edge Weight

```rust
use motlie_db::graph::mutation::UpdateEdge;

let (_summary, _weight, version) = EdgeSummaryBySrcDstName::new(alice_id, bob_id, "knows".to_string(), None)
    .run(&reader, timeout)
    .await?;

UpdateEdge {
    src_id: alice_id,
    dst_id: bob_id,
    name: "knows".to_string(),
    expected_version: version,
    new_weight: Some(Some(0.95)),  // Increase relationship strength
    new_active_period: None,
    new_summary: None,
}
.run(&writer)
.await?;
```

### Update Multiple Edge Fields

```rust
use motlie_db::ActiveTimeMillis;

UpdateEdge {
    src_id: alice_id,
    dst_id: bob_id,
    name: "knows".to_string(),
    expected_version: version,
    new_weight: Some(Some(1.0)),
    new_active_period: Some(ActivePeriod::active_from(
        ActiveTimeMillis(1_771_286_400_000),  // 2026-02-17T00:00:00Z
    )),
    new_summary: Some(EdgeSummary::from_text("Close friends since 2024")),
}
.run(&writer)
.await?;
```

---

## Version Queries & Time-Travel

Every mutation creates a new version with `ValidSince` timestamp. Time-travel queries use `as_of` to retrieve historical state.
(codex, 2026-02-10, decision: `as_of` uses `SystemTimeMillis`; `reference_ts_millis` uses `ActiveTimeMillis`.)

### Query Current Version

```rust
// as_of = None means "current version"
let (_name, _summary, version) = NodeById::new(alice_id, None)
    .run(&reader, timeout)
    .await?;

println!("Current version: {}", version);
```

### Query at Specific Point in Time

```rust
use motlie_db::{SystemTimeMillis, TimestampMilli};

// Query state as it was at a specific system timestamp
// Convert a wall-clock datetime to a system timestamp (ms since epoch).
// Example: 2023-11-15T00:00:00Z
let historical_timestamp = SystemTimeMillis(1_700_000_000_000);

let (name, _summary, version) = NodeById::as_of(alice_id, historical_timestamp, None)
    .run(&reader, timeout)
    .await?;

println!("Version at {}: {}", historical_timestamp.0, version);
println!("Name was: {}", name);
```

### Query Historical Edges

```rust
use motlie_db::{SystemTimeMillis, TimestampMilli};

// Example: 2023-11-15T00:00:00Z
let past = SystemTimeMillis(1_700_000_000_000);

// Get edges as they existed at that system time
let outgoing_then = OutgoingEdges::as_of(alice_id, past, None)
    .run(&reader, timeout)
    .await?;

// Get specific edge at that system time
let (summary_then, weight_then, version_then) = EdgeSummaryBySrcDstName::as_of(
    alice_id,
    bob_id,
    "knows".to_string(),
    past,
    None,
)
.run(&reader, timeout)
.await?;
```

### ActivePeriod + System Time for Edge Lists

Example: A promotion edge is **written now** (system time), but becomes
**active next week** (business time / ActivePeriod).

```rust
use motlie_db::graph::mutation::{AddNode, AddEdge};
use motlie_db::graph::Runnable;
use motlie_db::graph::query::{OutgoingEdges, IncomingEdges};
use motlie_db::graph::schema::{ActivePeriod, EdgeSummary, NodeSummary};
use motlie_db::{ActiveTimeMillis, Id, SystemTimeMillis};

// System time for write (explicit date → ms since epoch).
// Example: 2026-02-10T00:00:00Z
let now = SystemTimeMillis(1_770_681_600_000);
// Business-time window (explicit dates in ms since epoch).
// 2026-02-17T00:00:00Z .. 2026-02-19T00:00:00Z
let next_week = ActiveTimeMillis(1_771_286_400_000);
let next_week_end = ActiveTimeMillis(1_771_459_200_000);
let now_business = ActiveTimeMillis(now.0);

let store_id = Id::new();
let user_id = Id::new();

AddNode {
    id: store_id,
    ts_millis: now,
    name: "Store".into(),
    valid_range: None,
    summary: NodeSummary::from_text("Store profile"),
}.run(&writer).await?;

AddNode {
    id: user_id,
    ts_millis: now,
    name: "User".into(),
    valid_range: None,
    summary: NodeSummary::from_text("User profile"),
}.run(&writer).await?;

// Write edge now, but ActivePeriod is in the future.
AddEdge {
    source_node_id: store_id,
    target_node_id: user_id,
    ts_millis: now, // system time of insertion
    name: "promotion".into(),
    valid_range: ActivePeriod::active_between(next_week, next_week_end),
    summary: EdgeSummary::from_text("Promo active next week"),
    weight: None,
}.run(&writer).await?;

// 1) Current system time + current business time -> NOT visible yet.
let outgoing_now = OutgoingEdges::new(store_id, Some(now_business))
    .run(&reader, timeout)
    .await?;
assert!(outgoing_now.is_empty());

// 2) Current system time + business time in future window -> visible.
let outgoing_future = OutgoingEdges::new(store_id, Some(next_week))
    .run(&reader, timeout)
    .await?;
assert!(!outgoing_future.is_empty());

// 3) System time as-of insertion + business time outside window -> not visible.
let outgoing_as_of_insert = OutgoingEdges::as_of(store_id, now, Some(now_business))
    .run(&reader, timeout)
    .await?;
assert!(outgoing_as_of_insert.is_empty());

// 4) Incoming edges use the same semantics.
let incoming_future = IncomingEdges::new(user_id, Some(next_week))
    .run(&reader, timeout)
    .await?;
assert!(!incoming_future.is_empty());
```

### Version History Example

```rust
use motlie_db::SystemTimeMillis;
use motlie_db::graph::mutation::AddNode;
use motlie_db::graph::schema::NodeSummary;

// Create node
let node_id = Id::new();
AddNode {
    id: node_id,
    ts_millis: TimestampMilli::now(),
    name: "V1".into(),
    summary: NodeSummary::from_text("Initial"),
    valid_range: None,
}.run(&writer).await?;
// Example system timestamps (ms since epoch)
let t1 = SystemTimeMillis(1_771_228_900_000);

// Update node
std::thread::sleep(std::time::Duration::from_millis(10));
let (_, _, v_current) = NodeById::new(node_id, None).run(&reader, timeout).await?;
UpdateNode {
    id: node_id,
    expected_version: v_current,
    new_summary: Some(NodeSummary::from_text("Updated")),
    new_active_period: None,
}.run(&writer).await?;
let t2 = SystemTimeMillis(1_771_229_000_000);

// Query different points in time
let (_, _, v1) = NodeById::as_of(node_id, t1, None).run(&reader, timeout).await?;
let (_, _, v2) = NodeById::as_of(node_id, t2, None).run(&reader, timeout).await?;
let (_, _, current) = NodeById::new(node_id, None).run(&reader, timeout).await?;

assert_eq!(v1, 1);
assert_eq!(v2, 2);
assert_eq!(current, 2);
```

#### Promotion ActivePeriod Update + Point-in-Time + Rollback

Example: Update a promotion’s ActivePeriod, query old vs. new periods,
then roll back by restoring the prior ActivePeriod.

```rust
use motlie_db::graph::mutation::{AddEdge, UpdateEdge, RestoreEdge};
use motlie_db::graph::Runnable;
use motlie_db::graph::query::EdgeSummaryBySrcDstName;
use motlie_db::graph::schema::{ActivePeriod, EdgeSummary};
use motlie_db::{ActiveTimeMillis, Id, SystemTimeMillis};

// Use case setup:
// - A store offers a promotion to a user.
// - The promo is scheduled for a future window (business time).
// - Marketing later shifts the window, then decides to roll it back.
let store_id = Id::new();
let user_id = Id::new();
let edge_name = "promotion".to_string();

// System time for write (explicit date → ms since epoch).
// Example: 2026-02-10T00:00:00Z
let now = SystemTimeMillis(1_770_681_600_000);
// Business-time windows (explicit dates in ms since epoch).
// Week1: 2026-02-17T00:00:00Z .. 2026-02-19T00:00:00Z
let week1_start = ActiveTimeMillis(1_771_286_400_000);
let week1_end = ActiveTimeMillis(1_771_459_200_000);
// Week2: 2026-02-24T00:00:00Z .. 2026-02-26T00:00:00Z
let week2_start = ActiveTimeMillis(1_771_891_200_000);
let week2_end = ActiveTimeMillis(1_772_064_000_000);

// Create promotion edge with week1 ActivePeriod (business time)
// - system time: now (when we write)
// - ActivePeriod: next week (when it's valid for the business domain)
AddEdge {
    source_node_id: store_id,
    target_node_id: user_id,
    ts_millis: now,
    name: edge_name.clone(),
    valid_range: ActivePeriod::active_between(week1_start, week1_end),
    summary: EdgeSummary::from_text("Promo v1"),
    weight: None,
}.run(&writer).await?;

// Capture system time after v1 write (for point-in-time query)
// Point-in-time capture (explicit system times)
let t1 = SystemTimeMillis(1_770_681_660_000); // 2026-02-10T00:01:00Z

// Update ActivePeriod to week2 (new business-time window)
let (current_summary, current_weight, current_version) =
    EdgeSummaryBySrcDstName::new(store_id, user_id, edge_name.clone(), None)
    .run(&reader, timeout)
    .await?;

UpdateEdge {
    src_id: store_id,
    dst_id: user_id,
    name: edge_name.clone(),
    expected_version: current_version,
    new_active_period: Some(ActivePeriod::active_between(week2_start, week2_end)),
    new_summary: Some(EdgeSummary::from_text("Promo v2")),
    new_weight: None,
}.run(&writer).await?;

// Capture system time after v2 write (for point-in-time query)
let t2 = SystemTimeMillis(1_770_681_720_000); // 2026-02-10T00:02:00Z

// Point-in-time query (system time): as-of t1 should reflect week1
let (v1_summary, _v1_weight, _v1_version) =
    EdgeSummaryBySrcDstName::as_of(store_id, user_id, edge_name.clone(), t1, None)
    .run(&reader, timeout)
    .await?;

// Current query (system time): reflects week2
let (v2_summary, _v2_weight, _v2_version) =
    EdgeSummaryBySrcDstName::new(store_id, user_id, edge_name.clone(), None)
    .run(&reader, timeout)
    .await?;

assert!(v1_summary.decode_string().unwrap().contains("Promo v1"));
assert!(v2_summary.decode_string().unwrap().contains("Promo v2"));

// Rollback: restore to the earlier version (system time = t1)
// This creates a new current version that matches v1's content/ActivePeriod.
RestoreEdge {
    src_id: store_id,
    dst_id: user_id,
    name: edge_name.clone(),
    as_of: t1,
    expected_version: None,
}.run(&writer).await?;

// Current query should now reflect the rollback (week1)
let (rolled_back, _rb_weight, _rb_version) =
    EdgeSummaryBySrcDstName::new(store_id, user_id, edge_name, None)
        .run(&reader, timeout)
        .await?;
assert!(rolled_back.decode_string().unwrap().contains("Promo v1"));
```

### Version Queries (Current vs. Point-in-Time)

```rust
use motlie_db::{ActiveTimeMillis, SystemTimeMillis};

// Current version (system time = now; business time optional)
let (_name, _summary, current_version) = NodeById::new(node_id, None)
    .run(&reader, timeout)
    .await?;

// Example business-time instant: 2026-02-18T00:00:00Z
let business_time = ActiveTimeMillis(1_771_372_800_000);

// Point-in-time (system time) lookup
let (_name, _summary, v_at_t1) = NodeById::as_of(node_id, t1, None)
    .run(&reader, timeout)
    .await?;

// Business-time filter (ActivePeriod) combined with system-time as_of
let (_name, _summary, v_at_t1_with_business_time) =
    NodeById::as_of(node_id, t1, Some(business_time))
        .run(&reader, timeout)
        .await?;
```

**Not currently supported by public API:**
- List all versions for a node/edge (no VersionHistory query surface).
- Git-style relative traversal (current‑1, current‑2).

---

## Edge Retargeting

Edges are identified by topology `(src_id, dst_id, name)`. To "retarget" an edge to a different destination, delete the old edge and create a new one.

### Retarget Edge to New Destination

```rust
use motlie_db::graph::mutation::{DeleteEdge, AddEdge};

// Alice "knows" Bob, but we want to change it to Alice "knows" Charlie

// Step 1: Get current edge version
let (old_summary, old_weight, old_version) = EdgeSummaryBySrcDstName::new(
    alice_id,
    bob_id,
    "knows".to_string(),
    None
).run(&reader, timeout).await?;

// Step 2: Delete old edge
DeleteEdge {
    src_id: alice_id,
    dst_id: bob_id,
    name: "knows".to_string(),
    expected_version: old_version,
}
.run(&writer)
.await?;

// Step 3: Create new edge to Charlie
AddEdge {
    source_node_id: alice_id,
    target_node_id: charlie_id,  // New destination
    ts_millis: TimestampMilli::now(),
    name: "knows".to_string(),
    summary: old_summary.clone(),  // Preserve summary
    weight: old_weight,            // Preserve weight
    valid_range: None,
}
.run(&writer)
.await?;
```

### Retarget with Transaction (Atomic)

```rust
// Use transaction for atomic retarget
let mut txn = writer.transaction()?;

// Read current edge
let (old_summary, old_weight, old_version) = txn.read(EdgeSummaryBySrcDstName::new(
    alice_id, bob_id, "knows".to_string(), None
))?;

// Delete old
txn.write(DeleteEdge {
    src_id: alice_id,
    dst_id: bob_id,
    name: "knows".to_string(),
    expected_version: old_version,
})?;

// Create new
txn.write(AddEdge {
    source_node_id: alice_id,
    target_node_id: charlie_id,
    ts_millis: TimestampMilli::now(),
    name: "knows".to_string(),
    summary: old_summary,
    weight: old_weight,
    valid_range: None,
})?;

// Atomic commit
txn.commit()?;
```

### Rename Edge (Change Edge Name)

```rust
// Similar pattern: delete old name, create new name
let (old_summary, old_weight, old_version) = EdgeSummaryBySrcDstName::new(
    alice_id, bob_id, "knows".to_string(), None
).run(&reader, timeout).await?;

// Delete "knows"
DeleteEdge {
    src_id: alice_id,
    dst_id: bob_id,
    name: "knows".to_string(),
    expected_version: old_version,
}.run(&writer).await?;

// Create "friends_with"
AddEdge {
    source_node_id: alice_id,
    target_node_id: bob_id,
    ts_millis: TimestampMilli::now(),
    name: "friends_with".to_string(),  // New name
    summary: old_summary,
    weight: old_weight,
    valid_range: None,
}.run(&writer).await?;
```

---

## Rollback Operations

Rollback restores entities to their state at a previous point in time by creating a new version with historical data.

### Restore Node to Prior State

```rust
use motlie_db::graph::mutation::RestoreNode;
use motlie_db::SystemTimeMillis;

// Restore Alice to state as of November 2023
RestoreNode {
    id: alice_id,
    as_of: SystemTimeMillis(1_700_000_000_000),
    expected_version: None,
}
.run(&writer)
.await?;

// The node now has a new (incremented) version with old data
let (_name, _summary, restored_version) = NodeById::new(alice_id, None).run(&reader, timeout).await?;
// restored_version reflects the latest version (incremented)
// but data matches what it was at as_of time
```

### Restore Single Edge

```rust
use motlie_db::graph::mutation::RestoreEdge;
use motlie_db::SystemTimeMillis;

RestoreEdge {
    src_id: alice_id,
    dst_id: bob_id,
    name: "knows".to_string(),
    as_of: SystemTimeMillis(1_700_000_000_000),
    expected_version: None,
}
.run(&writer)
.await?;
```

### Batch Restore Edges (Vec<Mutation> of RestoreEdge)

```rust
use motlie_db::graph::mutation::RestoreEdge;
use motlie_db::SystemTimeMillis;

// Restore multiple edges in a single transaction
vec![
    RestoreEdge {
        src_id: alice_id,
        dst_id: bob_id,
        name: "knows".to_string(),
        as_of: SystemTimeMillis(1_700_000_000_000),
        expected_version: None,
    }.into(),
    RestoreEdge {
        src_id: alice_id,
        dst_id: carol_id,
        name: "knows".to_string(),
        as_of: SystemTimeMillis(1_700_000_000_000),
        expected_version: None,
    }.into(),
].run(&writer).await?;
```

### Validate Restore Before Executing

```rust
use motlie_db::graph::mutation::{ExecOptions, MutationBatchExt, MutationResult};
use motlie_db::SystemTimeMillis;

// Dry run to validate (no writes, still checks summaries + versions)
let reply = vec![
    RestoreEdge {
        src_id: alice_id,
        dst_id: bob_id,
        name: "knows".to_string(),
        as_of: SystemTimeMillis(1_700_000_000_000),
        expected_version: None,
    }.into(),
].run_with_result(&writer, ExecOptions { dry_run: true }).await?;

let version = reply
    .payload
    .first()
    .and_then(|r| match r { MutationResult::RestoreEdge { version } => Some(*version), _ => None });
println!("Restored version (if executed): {:?}", version);

// If satisfied, run for real
RestoreEdge {
    src_id: alice_id,
    dst_id: bob_id,
    name: "knows".to_string(),
    as_of: SystemTimeMillis(1_700_000_000_000),
    expected_version: None,
}
.run(&writer)
.await?;
```

### Restore Guard

Restore operations fail if the historical summary was garbage collected:

```rust
// This will error if the summary at as_of was GC'd
let result = RestoreNode {
    id: alice_id,
    as_of: very_old_timestamp,
    expected_version: None,
}.run(&writer).await;

match result {
    Err(e) if e.to_string().contains("summary") => {
        println!("Cannot restore: historical summary was garbage collected");
    }
    Err(e) => return Err(e),
    Ok(_) => println!("Restored successfully"),
}
```

---

## Transactions (Read-Your-Writes)

Transactions allow interleaved reads and writes with read-your-writes semantics.

```rust
// Create transaction from writer
let mut txn = writer.transaction()?;

// Write a node
let alice_id = Id::new();
txn.write(AddNode {
    id: alice_id,
    ts_millis: TimestampMilli::now(),
    name: "Alice".to_string(),
    summary: NodeSummary::from_text("New user"),
    valid_range: None,
})?;

// Read it back (sees uncommitted write)
let (name, _summary, _version) = txn.read(NodeById::new(alice_id, None))?;
assert_eq!(name, "Alice");

// Write an edge
let bob_id = Id::new();
txn.write(AddNode { id: bob_id, /* ... */ })?;
txn.write(AddEdge {
    source_node_id: alice_id,
    target_node_id: bob_id,
    name: "knows".to_string(),
    /* ... */
})?;

// Atomic commit
txn.commit()?;
```

**Transaction Requirements:**
- Writer must be created via spawn helpers or `Subsystem` (ensures Processor is attached)
- Storage must be read-write mode

---

## Storage Modes

```rust
use motlie_db::graph::Storage;

// ReadWrite mode (exclusive writer, required for mutations)
let mut storage = Storage::readwrite(&db_path);
storage.ready()?;

// ReadOnly mode (multiple readers, no mutations)
let mut storage = Storage::readonly(&db_path);
storage.ready()?;

// Secondary mode (replica/follower, catches up with primary)
let mut storage = Storage::secondary(&db_path, &secondary_path);
storage.ready()?;
storage.try_catch_up_with_primary()?;
```

---

## Subsystem Lifecycle

Subsystem manages end-to-end lifecycle (Writer, Reader, GC, caches):

```rust
use motlie_db::graph::{Subsystem, WriterConfig, ReaderConfig, GraphGcConfig};

let subsystem = Subsystem::new();

// Build storage with subsystem
let storage = StorageBuilder::new(path)
    .with_rocksdb(Box::new(subsystem))
    .build()?;

// Start writer, reader, and GC
let (writer, reader) = subsystem.start(
    storage.graph_storage().clone(),
    WriterConfig::default(),
    ReaderConfig::default(),
    4,  // query worker count
    Some(GraphGcConfig::default()),  // enable GC
);

// Use writer and reader...

// Shutdown (flush → consumers → GC)
subsystem.shutdown().await?;
```

---

## Concurrency Guarantees

| Component | Guarantee |
|-----------|-----------|
| Writer | Bounded MPSC channel with backpressure |
| Reader | MPMC flume with configurable worker pool |
| Processor | `Send + Sync`, shared via `Arc` |
| Transactions | RocksDB TransactionDB semantics (serializable) |
| Updates | Optimistic locking via `expected_version` |

### Optimistic Locking Example

```rust
// Two concurrent updates to same node
let (_name, _summary, version) = NodeById::new(alice_id, None).run(&reader, timeout).await?;

// Update 1 succeeds
UpdateNode {
    id: alice_id,
    expected_version: version,  // version = 1
    new_summary: Some(NodeSummary::from_text("Update 1")),
    new_active_period: None,
}.run(&writer).await?;

// Update 2 fails (version mismatch)
let result = UpdateNode {
    id: alice_id,
    expected_version: version,  // still 1, but current is now 2
    new_summary: Some(NodeSummary::from_text("Update 2")),
    new_active_period: None,
}.run(&writer).await;

assert!(result.is_err());  // VersionMismatch error
```

---

## Spawn Helpers Reference

### Writer Helpers

| Helper | Purpose |
|--------|---------|
| `create_mutation_writer(config)` | Create writer + receiver channels |
| `spawn_mutation_consumer(receiver, config, &db_path)` | Spawn consumer with new storage |
| `spawn_mutation_consumer_with_storage(storage, config)` | Spawn with existing storage |
| `spawn_mutation_consumer_with_receiver(receiver, config, processor)` | Spawn with shared processor |
| `spawn_mutation_consumer_with_next(receiver, config, storage, next)` | Chain to next consumer |

### Reader Helpers

| Helper | Purpose |
|--------|---------|
| `create_reader_with_storage(storage, config)` | Create reader + receiver with shared Processor |
| `spawn_query_consumer(receiver, config, &db_path)` | Spawn consumer with new storage |
| `spawn_query_consumers_with_storage(storage, config, count)` | Spawn pool with existing storage |

(codex, 2026-02-10, decision: processor-backed reader helpers are internal-only and intentionally omitted from public API surface.)

---
