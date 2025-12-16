# motlie-db

A temporal graph database with integrated fulltext search, combining RocksDB (graph storage) and Tantivy (fulltext indexing).

## Overview

`motlie-db` provides two API layers:

1. **Unified API (Porcelain)** - High-level interface via `motlie_db::Storage`, `motlie_db::query`, and `motlie_db::mutation` modules. Recommended for most use cases.

2. **Subsystem APIs (Plumbing)** - Direct access to `motlie_db::graph` and `motlie_db::fulltext` modules for advanced usage, extension, and testing.

## Quick Start

### Read-Write Mode (Mutations + Queries)

```rust
use motlie_db::{Storage, StorageConfig};
use motlie_db::mutation::{AddNode, AddEdge, Runnable, NodeSummary, EdgeSummary};
use motlie_db::query::{Nodes, NodeById, Runnable as QueryRunnable};
use motlie_db::{Id, TimestampMilli};
use std::time::Duration;

// 1. Create storage in read-write mode
let storage = Storage::readwrite(graph_path, fulltext_path);

// 2. Initialize and get handles (ReadWriteHandles)
let handles = storage.ready(StorageConfig::default())?;

// 3. Execute mutations - writer() returns &Writer directly, no unwrap needed!
AddNode {
    id: Id::new(),
    ts_millis: TimestampMilli::now(),
    name: "Alice".to_string(),
    summary: NodeSummary::from_text("A person named Alice"),
    valid_range: None,
}
.run(handles.writer())
.await?;

// 4. Execute queries
let timeout = Duration::from_secs(5);
let results = Nodes::new("Alice".to_string(), 10)
    .run(handles.reader(), timeout)
    .await?;

// 5. Clean shutdown
handles.shutdown().await?;
```

### Read-Only Mode (Queries Only)

```rust
use motlie_db::{Storage, StorageConfig};
use motlie_db::query::{Nodes, NodeById, OutgoingEdges, Runnable};
use std::time::Duration;

// 1. Create storage in read-only mode
let storage = Storage::readonly(graph_path, fulltext_path);

// 2. Initialize and get handles (ReadOnlyHandles)
let handles = storage.ready(StorageConfig::default())?;
let timeout = Duration::from_secs(5);

// 3. Fulltext search (returns hydrated graph data)
let results = Nodes::new("rust programming".to_string(), 10)
    .run(handles.reader(), timeout)
    .await?;

// 4. Direct graph lookups
let (name, summary) = NodeById::new(node_id, None)
    .run(handles.reader(), timeout)
    .await?;

// 5. Clean shutdown
handles.shutdown().await?;

// Note: handles.writer() is not available on ReadOnlyHandles - compile error if you try!
```

## API Lifecycle

### Type-Safe Access Modes

The unified API uses Rust's type system to distinguish between read-only and read-write access at compile time:

```text
Storage::readonly(graph, fulltext)   ──► Storage<ReadOnly>  ──► ReadOnlyHandles
                                                                   └── reader() only

Storage::readwrite(graph, fulltext)  ──► Storage<ReadWrite> ──► ReadWriteHandles
                                                                   ├── reader()
                                                                   └── writer()
```

This eliminates runtime checks - if you have `ReadWriteHandles`, `writer()` is guaranteed to exist.

### Writer Lifecycle (Read-Write Mode)

```text
Storage::readwrite(graph_path, fulltext_path)
    │
    ▼
storage.ready(StorageConfig)  →  ReadWriteHandles
    │                                  │
    │                                  ├── handles.writer() → &Writer
    │                                  │       │
    │                                  │       └── mutation.run(writer)
    │                                  │
    │                                  ├── handles.reader() → &Reader
    │                                  │       │
    │                                  │       └── query.run(reader, timeout)
    │                                  │
    │                                  └── handles.shutdown().await
    │
    └── Sets up: Graph → Fulltext mutation pipeline + MPMC query pools
```

### Reader Lifecycle (Read-Only Mode)

```text
Storage::readonly(graph_path, fulltext_path)
    │
    ▼
storage.ready(StorageConfig)  →  ReadOnlyHandles
    │                                  │
    │                                  ├── handles.reader() → &Reader
    │                                  │       │
    │                                  │       └── query.run(reader, timeout)
    │                                  │
    │                                  └── handles.shutdown().await
    │
    └── Sets up: MPMC consumer pools for graph + fulltext + unified queries
```

## Public Type Catalog

### Core Types (`motlie_db`)

| Type | Description |
|------|-------------|
| `Id` | 128-bit ULID identifier with lexicographic ordering |
| `TimestampMilli` | Millisecond-precision timestamp |
| `TemporalRange` | Validity window (start, until) for temporal queries |
| `DataUrl` | RFC 2397 data URL for content storage (text, markdown, JSON, images) |

### Storage Types (`motlie_db`)

| Type | Description |
|------|-------------|
| `Storage<ReadOnly>` | Read-only storage, created via `Storage::readonly()` |
| `Storage<ReadWrite>` | Read-write storage, created via `Storage::readwrite()` |
| `StorageConfig` | Configuration for channels and worker counts |
| `ReadOnlyHandles` | Handles from read-only storage: `reader()` only |
| `ReadWriteHandles` | Handles from read-write storage: `reader()` + `writer()` |
| `ReadOnly` | Marker type for read-only mode |
| `ReadWrite` | Marker type for read-write mode |

### Reader Module (`motlie_db::reader`)

| Type | Description |
|------|-------------|
| `Reader` | Query interface routing to graph/fulltext backends |
| `ReaderConfig` | Configuration for channel buffer sizes |
| `ReaderBuilder` | Builder for manual reader setup (advanced) |
| `Runnable<R>` | Generic trait for queries: `query.run(&reader, timeout)` |

### Writer Module (`motlie_db::writer`)

| Type | Description |
|------|-------------|
| `Writer` | Mutation interface for the graph→fulltext pipeline |
| `WriterConfig` | Configuration for channel buffer sizes |
| `WriterBuilder` | Builder for manual writer setup (advanced) |
| `Runnable` | Trait for mutations: `mutation.run(&writer)` |

### Mutation Module (`motlie_db::mutation`)

| Type | Description |
|------|-------------|
| `AddNode` | Create a node with name and summary |
| `AddEdge` | Create an edge between nodes |
| `AddNodeFragment` | Add timestamped content to a node |
| `AddEdgeFragment` | Add timestamped content to an edge |
| `UpdateNodeValidSinceUntil` | Update node's temporal validity |
| `UpdateEdgeValidSinceUntil` | Update edge's temporal validity |
| `UpdateEdgeWeight` | Update edge weight |
| `MutationBatch` | Batch multiple mutations atomically |
| `Mutation` | Enum wrapping all mutation types |
| `NodeSummary` | Node summary content wrapper |
| `EdgeSummary` | Edge summary content wrapper |
| `NodeName` | Node name wrapper |
| `EdgeName` | Edge name wrapper |
| `Runnable` | Re-export of `writer::Runnable` |

### Query Module (`motlie_db::query`)

| Type | Description |
|------|-------------|
| `Nodes` | Fulltext search for nodes (returns hydrated results) |
| `Edges` | Fulltext search for edges (returns hydrated results) |
| `NodeById` | Lookup node by ID |
| `NodesByIdsMulti` | Batch lookup nodes by IDs |
| `OutgoingEdges` | Get edges from a node |
| `IncomingEdges` | Get edges to a node |
| `EdgeDetails` | Lookup edge by topology (src, dst, name) |
| `NodeFragments` | Get node fragment history |
| `EdgeFragments` | Get edge fragment history |
| `AllNodes` | Enumerate all nodes with pagination (for graph algorithms) |
| `AllEdges` | Enumerate all edges with pagination (for graph algorithms) |
| `FuzzyLevel` | Fuzzy search tolerance (None, Low, Medium, High) |
| `WithOffset` | Trait for pagination offset |
| `Runnable` | Re-export of `reader::Runnable` |

#### Result Types

| Type | Shape |
|------|-------|
| `NodeResult` | `(Id, NodeName, NodeSummary)` |
| `EdgeResult` | `(SrcId, DstId, EdgeName, EdgeSummary)` |
| `EdgeDetailsResult` | `(Option<f64>, SrcId, DstId, EdgeName, EdgeSummary)` |
| `OutgoingEdgesResult` | `Vec<(Option<f64>, SrcId, DstId, EdgeName)>` |
| `IncomingEdgesResult` | `Vec<(Option<f64>, DstId, SrcId, EdgeName)>` |
| `AllNodesResult` | `Vec<(Id, NodeName, NodeSummary)>` |
| `AllEdgesResult` | `Vec<(Option<f64>, SrcId, DstId, EdgeName)>` |

## Subsystem APIs (Advanced Usage)

For advanced use cases, testing, or extension, access subsystems directly via qualified paths:

### Graph Module (`motlie_db::graph`)

```rust
use motlie_db::graph::{Storage, Graph, Reader, Writer, WriterConfig, ReaderConfig};
use motlie_db::graph::mutation::{AddNode, Mutation};
use motlie_db::graph::query::{NodeById, OutgoingEdges, AllNodes, AllEdges};
use motlie_db::graph::schema::{NodeSummary, EdgeSummary};
use motlie_db::graph::scan;  // Low-level visitor-based scan API
```

Key types:
- `graph::Storage` - RocksDB storage (readonly, readwrite, secondary modes)
- `graph::Graph` - Mutation and query processor
- `graph::Writer` / `graph::Reader` - Channel-based async infrastructure
- `graph::query::AllNodes` / `graph::query::AllEdges` - Graph enumeration (also in unified API)
- `graph::scan::*` - Low-level visitor-based pagination (advanced use)

### Fulltext Module (`motlie_db::fulltext`)

```rust
use motlie_db::fulltext::{Storage, Index, Reader, ReaderConfig};
use motlie_db::fulltext::query::{Nodes, Edges, FuzzyLevel};
use motlie_db::fulltext::search::{NodeHit, EdgeHit, MatchSource};
```

Key types:
- `fulltext::Storage` - Tantivy index (readonly, readwrite modes)
- `fulltext::Index` - Search processor
- `fulltext::query::Nodes/Edges` - Return raw hits (scores, match sources)
- `fulltext::search::NodeHit/EdgeHit` - Raw search results with BM25 scores

### Consumer Functions

Both subsystems expose spawn functions for building custom pipelines:

```rust
// Graph consumers
use motlie_db::graph::{
    spawn_mutation_consumer,
    spawn_mutation_consumer_with_next,  // Chain to fulltext
    spawn_query_consumer_pool_shared,
    spawn_query_consumer_pool_readonly,
};

// Fulltext consumers
use motlie_db::fulltext::{
    spawn_mutation_consumer,
    spawn_mutation_consumer_with_next,
    spawn_query_consumer_pool_shared,
};
```

## Architecture

### Mutation Pipeline

```text
┌─────────────────────────────────────────────────────────────────┐
│                    Unified Writer                                │
│                                                                  │
│   mutation.run(handles.writer())  →  Vec<Mutation>              │
│              │                                                   │
│              ▼                                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              graph::MutationConsumer                     │   │
│   │              (persists to RocksDB)                       │   │
│   └─────────────────────────┬───────────────────────────────┘   │
│                             │ chains to                          │
│                             ▼                                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │            fulltext::MutationConsumer                    │   │
│   │            (indexes in Tantivy)                          │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Query Pipeline

```text
┌─────────────────────────────────────────────────────────────────┐
│                     Unified Reader                               │
│                                                                  │
│   query.run(handles.reader(), timeout)                          │
│              │                                                   │
│              ├── Nodes/Edges ──▶ Fulltext search + Graph hydrate │
│              │                                                   │
│              └── NodeById/OutgoingEdges/... ──▶ Graph direct    │
│                                                                  │
│   ┌─────────────────────┐    ┌─────────────────────┐            │
│   │   Graph Consumer    │    │  Fulltext Consumer  │            │
│   │   Pool (MPMC)       │    │  Pool (MPMC)        │            │
│   └─────────┬───────────┘    └─────────┬───────────┘            │
│             │                          │                         │
│             ▼                          ▼                         │
│   ┌─────────────────────┐    ┌─────────────────────┐            │
│   │   graph::Graph      │    │   fulltext::Index   │            │
│   │   (RocksDB)         │    │   (Tantivy)         │            │
│   └─────────────────────┘    └─────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

## Graph Algorithm Support

The unified API fully supports graph algorithms like PageRank, Louvain, BFS, and A* through:

### Graph Enumeration

Use `AllNodes` and `AllEdges` to enumerate the entire graph with pagination:

```rust
use motlie_db::query::{AllNodes, AllEdges, Runnable};
use std::time::Duration;

let timeout = Duration::from_secs(30);

// Get all nodes (paginated)
let mut all_nodes = Vec::new();
let mut cursor: Option<Id> = None;

loop {
    let mut query = AllNodes::new(1000);
    if let Some(last_id) = cursor {
        query = query.with_cursor(last_id);
    }

    let page = query.run(handles.reader(), timeout).await?;
    if page.is_empty() {
        break;
    }

    cursor = page.last().map(|(id, _, _)| *id);
    all_nodes.extend(page);
}

// Get all edges (paginated)
let mut all_edges = Vec::new();
let mut edge_cursor: Option<(SrcId, DstId, EdgeName)> = None;

loop {
    let mut query = AllEdges::new(1000);
    if let Some(last) = edge_cursor.clone() {
        query = query.with_cursor(last);
    }

    let page = query.run(handles.reader(), timeout).await?;
    if page.is_empty() {
        break;
    }

    edge_cursor = page.last().map(|(_, s, d, n)| (*s, *d, n.clone()));
    all_edges.extend(page);
}
```

### Graph Traversal

Use `OutgoingEdges` and `IncomingEdges` for BFS/DFS traversal:

```rust
use motlie_db::query::{OutgoingEdges, IncomingEdges, Runnable};

// Get neighbors for traversal algorithms
let outgoing = OutgoingEdges::new(node_id, None)
    .run(handles.reader(), timeout)
    .await?;

for (weight, _src, dst, edge_name) in outgoing {
    // Process neighbor: dst with edge weight
}
```

### Algorithm Examples

See the `examples/` directory for complete implementations:

- **BFS** (`examples/graph/bfs.rs`) - Breadth-first search
- **PageRank** (`examples/graph/pagerank.rs`) - PageRank algorithm using `AllNodes` + `IncomingEdges`
- **Louvain** (`examples/graph/louvain.rs`) - Community detection using `AllNodes` + `AllEdges`
- **A*** (`examples/graph/a_star.rs`) - Pathfinding with heuristics

## Directory Structure

```
src/
├── lib.rs              # Core types (Id, TimestampMilli, DataUrl, TemporalRange)
├── storage.rs          # Unified storage (Storage<Mode>, ReadOnlyHandles, ReadWriteHandles)
├── reader.rs           # Unified reader infrastructure (Reader, ReaderConfig, ReaderBuilder)
├── writer.rs           # Unified writer infrastructure (Writer, WriterConfig, WriterBuilder)
├── query.rs            # Unified query types with Runnable implementations
├── mutation.rs         # Unified mutation re-exports
├── graph/              # RocksDB graph storage subsystem
│   ├── mod.rs          # Storage, Graph, re-exports
│   ├── schema.rs       # Column family definitions
│   ├── mutation.rs     # Mutation types and execution
│   ├── writer.rs       # Writer infrastructure
│   ├── query.rs        # Query types and execution
│   ├── reader.rs       # Reader infrastructure
│   └── scan.rs         # Pagination API
└── fulltext/           # Tantivy fulltext search subsystem
    ├── mod.rs          # Storage, Index, re-exports
    ├── schema.rs       # Tantivy schema definition
    ├── mutation.rs     # Mutation processing
    ├── writer.rs       # Writer infrastructure
    ├── query.rs        # Query types (Nodes, Edges, Facets)
    ├── reader.rs       # Reader infrastructure
    ├── search.rs       # Low-level search utilities
    └── fuzzy.rs        # Fuzzy search implementation
```

## See Also

- [`docs/graph_algorithm_api_analysis.md`](docs/graph_algorithm_api_analysis.md) - Graph algorithm API analysis
- [`docs/query-api-guide.md`](docs/query-api-guide.md) - Query API guide
- [`docs/TODO.md`](docs/TODO.md) - Design decisions and open questions

## License

See workspace LICENSE file.
