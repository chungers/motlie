# Graph Module API Reference

**Module:** `motlie_db::graph`  
**Purpose:** Content-addressed, versioned graph storage with async mutation/query pipelines

---

## Table of Contents

1. [Overview](#overview)
2. [API Architecture](#api-architecture)
3. [Part 1: Storage + Processor](#part-1-storage--processor)
4. [Part 2: Mutations (Writer)](#part-2-mutations-writer)
5. [Part 3: Queries (Reader)](#part-3-queries-reader)
6. [Part 4: VERSIONING & Time-Travel](#part-4-versioning--time-travel)
7. [Part 5: Transactions (Read-Your-Writes)](#part-5-transactions-read-your-writes)
8. [Part 6: Garbage Collection](#part-6-garbage-collection)
9. [Part 7: Subsystem Lifecycle](#part-7-subsystem-lifecycle)
10. [Part 8: Concurrency Guarantees](#part-8-concurrency-guarantees)
11. [Complete Usage Flow](#complete-usage-flow)
12. [Public API Catalog](#public-api-catalog)

---

## Overview

The graph module provides a content-addressed, versioned graph database on top of RocksDB. It supports:

- **Nodes and Edges** with optional summaries and fragments
- **Time-travel** via `ValidSince/ValidUntil` and `VersionHistory`
- **Rollback** via `RestoreNode/RestoreEdge/RestoreEdges`
- **Content-addressed summaries** with orphan-index GC
- **Async public API** (Writer/Reader) with a sync internal Processor

### Key Concepts

| Concept | Description |
|--------|-------------|
| **Processor** | Synchronous core (Storage + NameCache) used by async consumers |
| **Writer** | Async mutation interface via MPSC channel |
| **Reader** | Async query interface via MPMC channel |
| **Fragments** | Append-only content events |
| **Summaries** | Content-addressed summaries indexed by hash |
| **VERSIONING** | System time (ValidSince/ValidUntil) + Application time (ActivePeriod) |

---

## API Architecture

The graph module separates **internal sync execution** from **public async APIs**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    graph::Processor                              │
├─────────────────────────────────────────────────────────────────┤
│ Owned State:                                                     │
│   - Arc<Storage>              // RocksDB access                  │
│   - Arc<NameCache>            // NameHash ↔ String deduplication │
├─────────────────────────────────────────────────────────────────┤
│ Sync API (public):                                               │
│   - process_mutations()       // Batch mutation execution        │
│   - execute_mutation()        // Single mutation                 │
│   - storage()                 // Get storage reference           │
│   - name_cache()              // Get cache reference             │
│   - transaction_db()          // Get TransactionDB               │
└─────────────────────────────────────────────────────────────────┘
                          │
         ┌────────────────┴────────────────┐
         │                                 │
    ┌────▼─────┐                     ┌─────▼────┐
    │  Writer  │ (pub)               │  Reader  │ (pub)
    │  MPSC    │                     │  MPMC    │
    │  async   │                     │  async   │
    └────┬─────┘                     └─────┬────┘
         │                                 │
    ┌────▼─────┐                     ┌─────▼────┐
    │ Consumer │                     │ Consumer │
    │holds Arc │                     │holds Arc │
    │<Processor>                     │<Processor>
    └──────────┘                     └──────────┘
```

**Public API:** `Writer` and `Reader` (async, channel-based).  
**Internal core:** `Processor` (sync, direct RocksDB operations).

---

## Internal Processor API (Synchronous)

This is the **internal, synchronous** API used by the async consumers. It is the source of truth for graph operations and is **not channel-buffered**.

### Methods

| Method | Purpose |
|--------|---------|
| `Processor::new(storage: Arc<Storage>)` | Create Processor with Storage + NameCache |
| `Processor::with_cache(storage: Arc<Storage>, name_cache: Arc<NameCache>)` | Create Processor with explicit cache |
| `Processor::storage(&self) -> &Arc<Storage>` | Access underlying storage |
| `Processor::name_cache(&self) -> &Arc<NameCache>` | Access NameCache |
| `Processor::transaction_db(&self) -> Result<&TransactionDB>` | Access TransactionDB |
| `Processor::process_mutations(&self, mutations: &[Mutation]) -> Result<()>` | Execute batch atomically (sync) |
| `Processor::execute_mutation(&self, mutation: &Mutation) -> Result<()>` | Execute single mutation (sync) |

### Internal Usage Examples

#### Direct synchronous batch write

```rust
use motlie_db::graph::{Processor, Storage};
use motlie_db::graph::mutation::Mutation;
use std::sync::Arc;

let mut storage = Storage::readwrite(&db_path);
storage.ready()?;
let processor = Processor::new(Arc::new(storage));

let mutations: Vec<Mutation> = vec![
    // AddNode { ... }.into(),
    // AddEdge { ... }.into(),
];

processor.process_mutations(&mutations)?;
```

#### Direct synchronous single mutation

```rust
use motlie_db::graph::{Processor, Storage};
use motlie_db::graph::mutation::Mutation;
use std::sync::Arc;

let mut storage = Storage::readwrite(&db_path);
storage.ready()?;
let processor = Processor::new(Arc::new(storage));

let mutation: Mutation = /* AddNode { ... }.into() */;
processor.execute_mutation(&mutation)?;
```

### Relationship to Async APIs

The async layer is a **thin, buffered wrapper**:

```
Writer (async MPSC)
  └─ Consumer (async loop)
      └─ Processor::process_mutations(...)  // sync core

Reader (async MPMC)
  └─ Consumer pool (async loops)
      └─ Query::process_and_send(...) → QueryExecutor::execute(storage)
         (Processor provides storage access)
```

The async APIs exist for **backpressure and concurrency**, not different semantics.

---

## Part 1: Storage + Processor

### Storage

```rust
use motlie_db::graph::Storage;

// ReadWrite mode (exclusive writer)
let mut storage = Storage::readwrite(&db_path);
storage.ready()?;

// ReadOnly mode (multiple readers)
let mut storage = Storage::readonly(&db_path);
storage.ready()?;

// Secondary mode (replica/follower)
let mut storage = Storage::secondary(&db_path, &secondary_path);
storage.ready()?;
storage.try_catch_up_with_primary()?;
```

### Processor

```rust
use motlie_db::graph::{Processor, Storage};
use std::sync::Arc;

let mut storage = Storage::readwrite(&db_path);
storage.ready()?;
let storage = Arc::new(storage);

let processor = Processor::new(storage);
processor.process_mutations(&mutations)?;
```

---

## Part 2: Mutations (Writer)

### Core Mutations

| Mutation | Purpose |
|---------|---------|
| `AddNode` | Create node with name/summary |
| `AddEdge` | Create edge between nodes |
| `AddNodeFragment` | Append node fragment |
| `AddEdgeFragment` | Append edge fragment |
| `UpdateNodeSummary` | Versioned summary update |
| `UpdateEdgeSummary` | Versioned summary update |
| `DeleteNode` | Tombstone node |
| `DeleteEdge` | Tombstone edge |
| `RestoreNode` | Restore node to prior time |
| `RestoreEdge` | Restore edge to prior time |
| `RestoreEdges` | Batch restore outgoing edges |
| `UpdateNodeValidSinceUntil` | Versioned ActivePeriod update |
| `UpdateEdgeValidSinceUntil` | Versioned ActivePeriod update |
| `UpdateEdgeWeight` | Versioned weight update |

### Sending Mutations (Async)

```rust
use motlie_db::graph::mutation::AddNode;
use motlie_db::graph::writer::WriterConfig;
use motlie_db::writer::Runnable;
use motlie_db::{Id, TimestampMilli};

let (writer, receiver) = motlie_db::graph::writer::create_mutation_writer(WriterConfig::default());

// Spawn consumer with storage (recommended)
let _handle = motlie_db::graph::writer::spawn_mutation_consumer(
    receiver,
    WriterConfig::default(),
    &db_path,
);

AddNode {
    id: Id::new(),
    ts_millis: TimestampMilli::now(),
    name: "Alice".to_string(),
    summary: motlie_db::graph::schema::NodeSummary::from_text("A person"),
    valid_range: None,
}
.run(&writer)
.await?;
```

### Spawn Helpers

```rust
use motlie_db::graph::writer::{
    spawn_mutation_consumer_with_storage,
    spawn_mutation_consumer_with_receiver,
};

// With storage (creates Processor internally)
let (writer, handle) = spawn_mutation_consumer_with_storage(storage.clone(), config);

// With shared Processor (e.g., in subsystem)
let handle = spawn_mutation_consumer_with_receiver(receiver, config, processor.clone());
```

---

## Part 3: Queries (Reader)

### Core Queries

| Query | Purpose |
|-------|---------|
| `NodeById` | Get node by ID (current or as_of) |
| `OutgoingEdges` | Edges from node |
| `IncomingEdges` | Edges to node |
| `EdgeSummaryBySrcDstName` | Edge lookup by (src,dst,name) |
| `NodeFragmentsByIdTimeRange` | Node fragments in time range |
| `EdgeFragmentsByIdTimeRange` | Edge fragments in time range |
| `NodesBySummaryHash` | Reverse lookup by summary hash |
| `EdgesBySummaryHash` | Reverse lookup by summary hash |

### Running Queries (Async)

```rust
use motlie_db::graph::query::NodeById;
use motlie_db::reader::Runnable as QueryRunnable;
use std::time::Duration;

let timeout = Duration::from_secs(5);
let result = NodeById::new(node_id, None)
    .run(&reader, timeout)
    .await?;
```

### Spawn Helpers

```rust
use motlie_db::graph::reader::{
    spawn_query_consumers_with_storage,
    spawn_query_consumer_with_processor,
};

// With storage (creates Processor internally)
let (reader, handles) = spawn_query_consumers_with_storage(storage.clone(), config, 4);

// With shared Processor (e.g., in subsystem)
let handle = spawn_query_consumer_with_processor(receiver, config, processor.clone());
```

---

## Part 4: VERSIONING & Time-Travel

The graph uses **system time** for version history and **application time** for ActivePeriod.

| Field | Meaning |
|-------|---------|
| `ValidSince` | When this version became current in the DB |
| `ValidUntil` | When this version was superseded |
| `ActivePeriod` | Business validity interval |

### Time-Travel Query

```rust
use motlie_db::graph::query::NodeById;
use motlie_db::TimestampMilli;

let as_of = Some(TimestampMilli(1_700_000_000_000));
let node = NodeById::new(node_id, as_of).run(&reader, timeout).await?;
```

### Restore to Prior Time

```rust
use motlie_db::graph::mutation::RestoreNode;

RestoreNode {
    id: node_id,
    as_of: TimestampMilli(1_700_000_000_000),
}
.run(&writer)
.await?;
```

**Restore Guard:** If the summary referenced by the historical version was GC’d, `RestoreNode/RestoreEdge` returns an error. `RestoreEdges` skips those edges and logs a warning.

---

## Part 5: Transactions (Read-Your-Writes)

Writer transactions allow interleaved reads/writes before commit.

```rust
let mut txn = writer.transaction()?;

txn.write(AddNode { ... })?;
let node = txn.read(NodeById::new(id, None))?; // sees uncommitted writes
txn.commit()?;
```

**Requirements:**
- Writer must be configured with a Processor (via `set_processor` or spawn helpers)
- Storage must be read-write

---

## Part 6: Garbage Collection

GC cleans:
- stale summary index entries
- tombstones (future)
- orphan summaries (VERSIONING)

**Lifecycle:**
- Start via `GraphGarbageCollector::start(storage, config)`
- Shutdown via owned `gc.shutdown()`

Orphan summaries are tracked in `OrphanSummaries` CF and deleted after retention if no CURRENT references remain.

---

## Part 7: Subsystem Lifecycle

Subsystem manages end-to-end lifecycle (Writer, Reader, GC, caches):

```rust
use motlie_db::graph::{Subsystem, WriterConfig, ReaderConfig, GraphGcConfig};

let subsystem = Subsystem::new();
let storage = StorageBuilder::new(path)
    .with_rocksdb(Box::new(subsystem))
    .build()?;

let (writer, reader) = subsystem.start(
    storage.graph_storage().clone(),
    WriterConfig::default(),
    ReaderConfig::default(),
    4,
    Some(GraphGcConfig::default()),
);
```

**Shutdown order:** flush → consumers → GC

---

## Part 8: Concurrency Guarantees

- Writer uses bounded MPSC channel → backpressure
- Reader uses MPMC flume → multiple query workers
- Processor is `Send + Sync`, shared by Arc
- Transactions use RocksDB TransactionDB semantics

---

## Complete Usage Flow

```rust
use motlie_db::graph::{Storage, WriterConfig, ReaderConfig};
use motlie_db::graph::writer::spawn_mutation_consumer_with_storage;
use motlie_db::graph::reader::spawn_query_consumers_with_storage;
use motlie_db::graph::mutation::AddNode;
use motlie_db::writer::Runnable;
use std::time::Duration;

let mut storage = Storage::readwrite(&db_path);
storage.ready()?;
let storage = Arc::new(storage);

let (writer, _w_handle) = spawn_mutation_consumer_with_storage(storage.clone(), WriterConfig::default());
let (reader, _q_handles) = spawn_query_consumers_with_storage(storage.clone(), ReaderConfig::default(), 4);

AddNode { ... }.run(&writer).await?;
let node = NodeById::new(id, None).run(&reader, Duration::from_secs(5)).await?;
```

---

## Public API Catalog

### Core Types
- `Storage`
- `Processor`
- `Writer`, `WriterConfig`
- `Reader`, `ReaderConfig`
- `Transaction`

### Mutation Types (selected)
- `AddNode`, `AddEdge`
- `AddNodeFragment`, `AddEdgeFragment`
- `UpdateNodeSummary`, `UpdateEdgeSummary`
- `DeleteNode`, `DeleteEdge`
- `RestoreNode`, `RestoreEdge`, `RestoreEdges`

### Query Types (selected)
- `NodeById`, `OutgoingEdges`, `IncomingEdges`
- `NodeFragmentsByIdTimeRange`, `EdgeFragmentsByIdTimeRange`
- `NodesBySummaryHash`, `EdgesBySummaryHash`

### Spawn Helpers
- `create_mutation_writer`
- `spawn_mutation_consumer`
- `spawn_mutation_consumer_with_next`
- `spawn_mutation_consumer_with_receiver`
- `spawn_mutation_consumer_with_storage`
- `create_query_reader`
- `spawn_query_consumer`
- `spawn_query_consumer_with_processor`
- `spawn_query_consumer_pool_shared`
- `spawn_query_consumers_with_storage`

---
