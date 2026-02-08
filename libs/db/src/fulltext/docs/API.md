# Fulltext Module API Reference

**Module:** `motlie_db::fulltext`
**Purpose:** Tantivy-backed fulltext search with async query pipeline and graph-backed indexing

---

## Overview

The fulltext module provides:
- **Search queries** for nodes, edges, and facet counts
- **Async query pipeline** via `Reader` and `Runnable`
- **Indexing via mutations** executed by the fulltext writer/consumer

The API is intentionally simple: query types map to typed results, while indexing is driven by
mutations (primarily from the graph module) flowing through the writer/consumer.

---

## Async Query API (Typed Replies)

Each fulltext query implements a `QueryReply` mapping that binds the query type to its specific
output variant. User code still calls `Runnable::run`, but the result mapping is now explicit on the type.

### Nodes Search

```rust
use motlie_db::fulltext::query::Nodes;
use motlie_db::reader::Runnable as QueryRunnable;
use std::time::Duration;

let timeout = Duration::from_secs(5);
let results = Nodes::new("rust".to_string(), 10)
    .run(&reader, timeout)
    .await?;
```

### Edges Search

```rust
use motlie_db::fulltext::query::Edges;
use motlie_db::reader::Runnable as QueryRunnable;
use std::time::Duration;

let timeout = Duration::from_secs(5);
let results = Edges::new("collaborates".to_string(), 10)
    .run(&reader, timeout)
    .await?;
```

### Facets

```rust
use motlie_db::fulltext::query::Facets;
use motlie_db::reader::Runnable as QueryRunnable;
use std::time::Duration;

let timeout = Duration::from_secs(5);
let facets = Facets::new().run(&reader, timeout).await?;
```

---

## Indexing (Mutation-Driven)

The fulltext index is updated by consuming mutations. These are typically graph mutations
(`AddNode`, `AddEdge`, `UpdateNode`, etc.) that are executed by the fulltext writer's consumer.

```rust
use motlie_db::fulltext::writer::spawn_fulltext_writer_with_storage;
use motlie_db::graph::mutation::AddNode;
use motlie_db::writer::Runnable as MutationRunnable;
use motlie_db::{Id, TimestampMilli};

let (writer, _handle) = spawn_fulltext_writer_with_storage(storage.clone(), Default::default());

AddNode {
    id: Id::new(),
    ts_millis: TimestampMilli::now(),
    name: "Alice".to_string(),
    summary: NodeSummary::from_text("A person"),
    valid_range: None,
}
.run(&writer)
.await?;
```

---

## Query Catalog

| Query | Output | Description |
|-------|--------|-------------|
| `Nodes` | `Vec<NodeHit>` | Fulltext search over nodes |
| `Edges` | `Vec<EdgeHit>` | Fulltext search over edges |
| `Facets` | `FacetCounts` | Faceted counts for tags/types |

