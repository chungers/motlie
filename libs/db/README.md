# motlie-db

A graph database library built on RocksDB with async query/mutation processing and full-text search capabilities.

## Overview

`motlie-db` provides a high-performance, async graph database with the following key features:

- **Unified Query API**: Single `Reader` interface for both fulltext search and graph queries
- **Graph Storage**: Nodes and edges with bidirectional navigation
- **Fragment Management**: Time-series content fragments attached to entities
- **Full-Text Search**: Tantivy-based search index with graph hydration
- **Async Processing**: Query and mutation processing with timeout support
- **Optimized Key Encoding**: Direct byte concatenation for RocksDB keys (values use MessagePack)
- **Prefix Scan Optimization**: O(1) prefix seek with RocksDB prefix extractors and bloom filters
- **Type-Safe IDs**: ULID-based 128-bit identifiers with lexicographic ordering

## Quick Start: Unified Query API

The recommended way to use `motlie-db` is through the **Unified Query API**, which provides a single `Reader` interface for all query types:

```rust
use motlie_db::reader::{Storage, ReaderConfig, StorageHandle};
use motlie_db::query::{
    // Fulltext search queries (with graph hydration)
    Nodes, Edges,
    // Direct graph lookups
    NodeById, OutgoingEdges, IncomingEdges, EdgeDetails, NodeFragments, EdgeFragments,
    // Trait for execution
    Runnable,
};
use std::time::Duration;

// 1. Initialize unified storage (graph + fulltext)
//    ready() returns a StorageHandle that manages lifecycle
let storage = Storage::readonly(graph_path, fulltext_path);
let handle = storage.ready(ReaderConfig::default(), 4)?;

let timeout = Duration::from_secs(5);

// 2. Fulltext search with automatic graph hydration
// Returns Vec<(Id, NodeName, NodeSummary)> - full graph data, not just search hits
let results = Nodes::new("rust programming".to_string(), 10)
    .with_fuzzy(FuzzyLevel::Low)
    .with_tags(vec!["systems".to_string()])
    .run(handle.reader(), timeout)
    .await?;

// 3. Direct graph lookups - same unified reader!
let (name, summary) = NodeById::new(node_id, None)
    .run(handle.reader(), timeout)
    .await?;

let outgoing = OutgoingEdges::new(node_id, None)
    .run(handle.reader(), timeout)
    .await?;

let incoming = IncomingEdges::new(node_id, None)
    .run(handle.reader(), timeout)
    .await?;

// 4. Edge details and fragment history
let edge = EdgeDetails::new(src_id, dst_id, "relationship".to_string(), None)
    .run(handle.reader(), timeout)
    .await?;

use std::ops::Bound;
let fragments = NodeFragments::new(node_id, (Bound::Unbounded, Bound::Unbounded), None)
    .run(handle.reader(), timeout)
    .await?;

// 5. Concurrent queries with tokio::try_join!
let reader = handle.reader_clone();
let (nodes, edges, incoming) = tokio::try_join!(
    Nodes::new("search".to_string(), 10).run(&reader, timeout),
    OutgoingEdges::new(node_id, None).run(&reader, timeout),
    IncomingEdges::new(node_id, None).run(&reader, timeout)
)?;

// 6. Clean shutdown
handle.shutdown().await?;
```

### Key Benefits

| Feature | Description |
|---------|-------------|
| **Single Reader** | One `reader::Reader` for all query types - no need for separate fulltext/graph readers |
| **Lifecycle Management** | `StorageHandle` provides `shutdown()` for clean termination |
| **Automatic Hydration** | Fulltext results automatically hydrated with full graph data |
| **Consistent API** | All queries use `query.run(handle.reader(), timeout)` pattern |
| **Type Safety** | Return types determined by query type, not reader type |
| **Concurrent Queries** | Use `tokio::try_join!` with `handle.reader_clone()` for parallel execution |
| **MPMC Pipeline** | Efficient multi-producer multi-consumer query dispatch |

### Query Types

| Query | Output Type | Description |
|-------|-------------|-------------|
| `Nodes` | `Vec<(Id, NodeName, NodeSummary)>` | Fulltext search nodes with graph hydration |
| `Edges` | `Vec<(SrcId, DstId, EdgeName, EdgeSummary)>` | Fulltext search edges with graph hydration |
| `NodeById` | `(NodeName, NodeSummary)` | Direct node lookup by ID |
| `NodesByIdsMulti` | `Vec<(Id, NodeName, NodeSummary)>` | Batch lookup nodes by IDs (uses RocksDB MultiGet) |
| `OutgoingEdges` | `Vec<(Option<f64>, SrcId, DstId, EdgeName)>` | Edges from a node |
| `IncomingEdges` | `Vec<(Option<f64>, DstId, SrcId, EdgeName)>` | Edges to a node |
| `EdgeDetails` | `(Option<f64>, SrcId, DstId, EdgeName, EdgeSummary)` | Edge lookup by topology |
| `NodeFragments` | `Vec<(TimestampMilli, FragmentContent)>` | Node fragment history |
| `EdgeFragments` | `Vec<(TimestampMilli, FragmentContent)>` | Edge fragment history |

### StorageHandle: Lifecycle Management

The `StorageHandle` returned by `Storage::ready()` provides clean lifecycle management for the unified query system:

```rust
use motlie_db::reader::{Storage, ReaderConfig, StorageHandle};

// Initialize storage - ready() returns StorageHandle
let storage = Storage::readonly(graph_path, fulltext_path);
let handle: StorageHandle = storage.ready(ReaderConfig::default(), 4)?;

// Access the reader for queries
let reader = handle.reader();           // &Reader
let reader_clone = handle.reader_clone(); // Reader (for concurrent use)

// Check if system is still running
if handle.is_running() {
    // Execute queries...
}

// Clean shutdown - closes channels and awaits all workers
handle.shutdown().await?;

// Alternative: graceful shutdown (logs warnings instead of returning errors)
// handle.shutdown_graceful().await;
```

**StorageHandle API:**

| Method | Return Type | Description |
|--------|-------------|-------------|
| `reader()` | `&Reader` | Get a reference to the Reader |
| `reader_clone()` | `Reader` | Clone the Reader (for concurrent tasks) |
| `is_running()` | `bool` | Check if channels are still open |
| `shutdown()` | `Result<()>` | Clean termination - drops Reader, awaits worker handles |
| `shutdown_graceful()` | `()` | Same as shutdown but logs warnings instead of errors |

**Lifecycle Diagram:**

```
Storage::new()  →  Storage::ready()  →  StorageHandle  →  shutdown()
      │                   │                   │                │
      │                   │                   │                ├── Drops Reader (closes channels)
      │                   │                   │                └── Awaits all worker JoinHandles
      │                   │                   │
      │                   │                   ├── reader() → execute queries
      │                   │                   └── is_running() → check status
      │                   │
      │                   └── Initializes graph + fulltext subsystems
      │                       Spawns MPMC consumer pools
      │
      └── Creates Storage with paths to graph and fulltext databases
```

**Why StorageHandle?**

1. **RAII-style resource management**: Even if `shutdown()` isn't called, dropping the handle closes channels and workers terminate gracefully
2. **Proper ownership**: The Reader and worker handles are bundled together, preventing resource leaks
3. **Clean API**: Single point of control for start/stop lifecycle
4. **Idiomatic Rust**: Follows the pattern of `tokio::spawn()` returning a `JoinHandle`

## Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Application                                      │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
           ┌─────────────────────▼─────────────────────┐
           │           reader::Reader (unified)         │
           │  - Fulltext search with graph hydration   │
           │  - Direct graph lookups                   │
           │  - MPMC query dispatch                    │
           └─────┬───────────────────────┬─────────────┘
                 │                       │
    ┌────────────▼────────────┐  ┌──────▼──────────────────┐
    │   graph::Reader         │  │   fulltext::Reader       │
    │   (graph queries)       │  │   (fulltext + hydration) │
    └────────────┬────────────┘  └──────┬──────────────────┘
                 │                       │
    ┌────────────▼────────────┐  ┌──────▼──────────────────┐
    │  Query Consumer Pool    │  │  Query Consumer Pool     │
    │  (async tasks, MPMC)    │  │  (async tasks, MPMC)     │
    └────────────┬────────────┘  └──────┬──────────────────┘
                 │                       │
    ┌────────────▼────────────┐  ┌──────▼──────────────────┐
    │   graph::Graph          │  │   fulltext::Index        │
    │   (RocksDB)             │  │   (Tantivy)              │
    │  ┌─────────────────┐    │  └─────────────────────────┘
    │  │ Nodes │ Edges   │    │
    │  │ Fragments │ ... │    │
    │  └─────────────────┘    │
    └─────────────────────────┘

                    ┌────────────────────────────────────────┐
                    │           Writer API                    │
                    │   (Mutation System)                    │
                    └───────────────┬────────────────────────┘
                                    │
                    ┌───────────────▼────────────────────────┐
                    │        Mutation Consumer Chain          │
                    │  Graph Consumer → FullText Consumer     │
                    │  (batched transactions)                 │
                    └─────────────────────────────────────────┘
```

### Query Architecture: Params/Dispatch Pattern

`motlie-db` uses a consistent **Params/Dispatch** pattern across all query modules to separate user-facing API from internal infrastructure:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           User-Facing API                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Query Params Structs (public)                                        │   │
│  │  - fulltext::Nodes, fulltext::Edges, fulltext::Facets                │   │
│  │  - graph::NodeById, graph::OutgoingEdges, graph::IncomingEdges       │   │
│  │  - query::Nodes, query::Edges (type aliases to fulltext::*)          │   │
│  │                                                                       │   │
│  │  Features:                                                            │   │
│  │  - #[derive(Debug, Clone, PartialEq, Default)]                       │   │
│  │  - All fields public for struct initialization                        │   │
│  │  - Deserializable from JSON without copying                          │   │
│  │  - Builder methods for fluent API                                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    │ query.run(&reader, timeout)             │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Runnable<R> Trait                                                   │   │
│  │  - Generic over reader type R                                        │   │
│  │  - Same params, different return types per reader                    │   │
│  │  - fulltext::Reader → Vec<NodeHit> (raw hits with scores)           │   │
│  │  - reader::Reader → Vec<(Id, Name, Summary)> (hydrated data)        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     │ Internal (hidden from docs)
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Internal Infrastructure                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Dispatch Wrappers (pub(crate))                                      │   │
│  │  - NodesDispatch, EdgesDispatch, FacetsDispatch                      │   │
│  │  - NodeByIdDispatch, OutgoingEdgesDispatch, etc.                     │   │
│  │                                                                       │   │
│  │  Contents:                                                            │   │
│  │  - params: T           // The user-facing params struct               │   │
│  │  - timeout: Duration   // Execution timeout                           │   │
│  │  - result_tx: oneshot  // Channel for async result delivery          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Query/Search Enums (pub + #[doc(hidden)])                           │   │
│  │  - graph::Query { NodeById(..), OutgoingEdges(..), ... }             │   │
│  │  - fulltext::Search { Nodes(..), Edges(..), Facets(..) }             │   │
│  │  - query::Query { Nodes(..), Edges(..), NodeById(..), ... }          │   │
│  │                                                                       │   │
│  │  Public for function signatures, but users never construct directly   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Reader Infrastructure (pub + #[doc(hidden)])                        │   │
│  │  - create_query_reader(), spawn_consumer_pool_shared()               │   │
│  │  - Consumer structs, spawn_* functions                               │   │
│  │                                                                       │   │
│  │  MPMC channel-based query dispatch and processing                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Benefits of Params/Dispatch Separation

1. **Clean User API**: Users work with simple structs, not channel machinery
   ```rust
   // Struct initialization - all fields accessible
   let query = Nodes {
       query: "rust programming".to_string(),
       limit: 10,
       tags: vec!["systems".to_string()],
       ..Default::default()
   };

   // Or builder pattern
   let query = Nodes::new("rust", 10)
       .with_fuzzy(FuzzyLevel::Low)
       .with_tags(vec!["systems".to_string()]);
   ```

2. **Serialization-Friendly**: Params structs derive `Clone`, `PartialEq`, `Default`
   ```rust
   // Deserialize from JSON directly
   let query: Nodes = serde_json::from_str(r#"{"query": "rust", "limit": 10}"#)?;
   ```

3. **Type-Safe Multi-Reader**: Same query type, different result types
   ```rust
   // Fulltext reader: returns raw hits with scores
   let hits: Vec<NodeHit> = query.clone().run(&fulltext_reader, timeout).await?;

   // Unified reader: returns hydrated graph data
   let results: Vec<(Id, NodeName, NodeSummary)> = query.run(&unified_reader, timeout).await?;
   ```

4. **Testable**: Params structs can be compared, cloned, stored
   ```rust
   assert_eq!(query1, query2);  // PartialEq
   let queries: Vec<Nodes> = vec![query1.clone(), query2];  // Clone
   ```

#### Visibility Rules

| Component | Visibility | Rationale |
|-----------|------------|-----------|
| Params structs (`Nodes`, `NodeById`, etc.) | `pub` | User-facing API |
| `Runnable<R>` trait | `pub` | User-facing API |
| Dispatch wrappers (`*Dispatch`) | `pub(crate)` | Internal infrastructure |
| Query/Search enums | `pub` + `#[doc(hidden)]` | Appear in function signatures |
| Reader factory functions | `pub` + `#[doc(hidden)]` | Advanced use / testing |
| Consumer structs | `pub` + `#[doc(hidden)]` | Advanced use / testing |

### Reader API (Query System)

There are **three reader types** in `motlie-db`, each serving different use cases:

#### 1. Unified Reader (`reader::Reader`) - Recommended

The **unified reader** provides a single interface for all query types:

```rust
use motlie_db::reader::{Storage, ReaderConfig, StorageHandle};
use motlie_db::query::{Nodes, NodeById, OutgoingEdges, Runnable};
use std::time::Duration;

// Initialize both graph and fulltext subsystems
let storage = Storage::readonly(graph_path, fulltext_path);
let handle = storage.ready(ReaderConfig::default(), 4)?;

let timeout = Duration::from_secs(5);

// Fulltext search with automatic graph hydration
let results = Nodes::new("rust".to_string(), 10)
    .run(handle.reader(), timeout)
    .await?;

// Direct graph lookups through the same reader
let (name, summary) = NodeById::new(node_id, None)
    .run(handle.reader(), timeout)
    .await?;

let edges = OutgoingEdges::new(node_id, None)
    .run(handle.reader(), timeout)
    .await?;

// Clean shutdown
handle.shutdown().await?;
```

#### 2. Graph Reader (`graph::Reader`) - Graph-only queries

For direct graph access without fulltext:

```rust
use motlie_db::graph::{Storage, Graph, reader::{create_query_reader, spawn_query_consumer_pool_shared}};
use motlie_db::graph::query::{NodeById, OutgoingEdges, Runnable};

let (reader, receiver) = create_query_reader(config);
let handles = spawn_query_consumer_pool_shared(receiver, graph.clone(), 4);

let (name, summary) = NodeById::new(node_id, None)
    .run(&reader, timeout)
    .await?;
```

#### 3. Fulltext Reader (`fulltext::Reader`) - Raw search hits

For raw fulltext results without graph hydration:

```rust
use motlie_db::fulltext::{Storage, Index, reader::{create_query_reader, spawn_query_consumer_pool_shared}};
use motlie_db::fulltext::query::{Nodes, Runnable};

let (reader, receiver) = create_query_reader(config);
let handles = spawn_query_consumer_pool_shared(receiver, index.clone(), 4);

// Returns Vec<NodeHit> with scores (not hydrated graph data)
let hits = Nodes::new("search".to_string(), 10)
    .run(&reader, timeout)
    .await?;
```

**Key Features**:
- **Type-driven API** - Query type determines result type
- **Timeout at execution** - Timeout is a runtime parameter, not construction
- **No builder boilerplate** - Direct query construction
- **Composable queries** - Queries are values that can be stored and reused
- **Temporal queries** - Optional reference timestamp for historical queries
- **Concurrent execution** - Use tokio::join! for parallel queries
- Returns complete data (no need for joins)

See: [`docs/query-api-guide.md`](docs/query-api-guide.md) for comprehensive API documentation

### Writer API (Mutation System)

The `Writer` provides async mutation operations with automatic transaction batching:

```rust
use motlie_db::graph::writer::{create_mutation_writer, WriterConfig};
use motlie_db::graph::mutation::{AddNode, AddEdge, AddNodeFragment, Runnable};
use motlie_db::{Id, TimestampMilli, DataUrl};

// Create writer and receiver
let (writer, receiver) = create_mutation_writer(WriterConfig::default());

// Mutations implement the Runnable trait
// Use mutation.run(&writer).await to execute

// Add node
AddNode {
    id: Id::new(),
    name: "Alice".to_string(),
    ts_millis: TimestampMilli::now(),
    summary: DataUrl::from_text("Alice's profile"),
    valid_range: None,
}.run(&writer).await?;

// Add edge
AddEdge {
    src_id: alice_id,
    dst_id: bob_id,
    edge_name: "follows".to_string(),
    ts_millis: TimestampMilli::now(),
    summary: DataUrl::from_text("Alice follows Bob"),
    weight: Some(1.0),
    valid_range: None,
}.run(&writer).await?;

// Add fragment (time-series content)
AddNodeFragment {
    id: alice_id,
    ts_millis: TimestampMilli::now(),
    content: DataUrl::from_text("Status update: Hello world!"),
    valid_range: None,
}.run(&writer).await?;
```

**Key Features**:
- Non-blocking async writes via `Runnable` trait
- **Automatic transaction batching** - mutations are batched and committed in single RocksDB transactions
- MPSC channel-based with `Vec<Mutation>` batching support
- Consumer chaining (Graph → FullText)
- Automatic index updates
- Atomic multi-mutation commits

## RocksDB Schema

### Column Families

The database uses 5 column families optimized for different access patterns:

#### 1. **Nodes CF** - Node metadata
```rust
Key:   NodeCfKey(Id)                          // 16 bytes
Value: NodeCfValue(NodeName, NodeSummary)     // Variable (name + DataUrl)
```

**Purpose**: Store node name and summary
**Index**: By node ID (primary key)

#### 2. **Edges CF** - Edge metadata and topology
```rust
Key:   EdgeCfKey(Id)                                              // 16 bytes
Value: EdgeCfValue(SrcId, EdgeName, DstId, EdgeSummary)          // Variable
```

**Purpose**: Store edge topology and summary
**Index**: By edge ID (primary key)
**Note**: Contains complete edge information (source, destination, name, summary)

#### 3. **ForwardEdges CF** - Source → Destination index
```rust
Key:   ForwardEdgeCfKey(SrcId, DstId, EdgeName)    // 16 + 16 + var bytes
Value: ForwardEdgeCfValue(EdgeId)                  // 16 bytes
```

**Purpose**: Efficiently query all outgoing edges from a node
**Index**: By (source_id, dest_id, edge_name)
**Optimization**: Variable-length EdgeName at END for prefix scanning

#### 4. **ReverseEdges CF** - Destination → Source index
```rust
Key:   ReverseEdgeCfKey(DstId, SrcId, EdgeName)    // 16 + 16 + var bytes
Value: ReverseEdgeCfValue(EdgeId)                  // 16 bytes
```

**Purpose**: Efficiently query all incoming edges to a node
**Index**: By (dest_id, source_id, edge_name)
**Optimization**: Variable-length EdgeName at END for prefix scanning

#### 5. **Fragments CF** - Time-series content
```rust
Key:   FragmentCfKey(Id, TimestampMilli)     // 16 + 8 bytes
Value: FragmentCfValue(FragmentContent)      // Variable (DataUrl)
```

**Purpose**: Store timestamped content fragments
**Index**: By (entity_id, timestamp)
**Ordering**: Lexicographic by entity ID, then timestamp

### Key Encoding Optimization

**Critical Performance Feature**: Direct byte concatenation for keys enables RocksDB prefix extractors.

#### Why Not MessagePack for Keys?

MessagePack adds variable-length headers that break constant-length prefix requirements:
- `fixstr(0-31)`: 1 byte header
- `str8(32-255)`: 2 bytes header
- `str16(256-65535)`: 3 bytes header

This makes the prefix length **non-constant**, preventing RocksDB's prefix bloom filters and O(1) prefix seek.

#### Our Approach: Hybrid Encoding

- **Keys**: Direct byte concatenation (constant-length prefixes)
  - `ForwardEdgeCfKey`: `[src_id (16)] + [dst_id (16)] + [name UTF-8]` = **32-byte constant prefix**
  - `ReverseEdgeCfKey`: `[dst_id (16)] + [src_id (16)] + [name UTF-8]` = **32-byte constant prefix**
  - `FragmentCfKey`: `[id (16)] + [timestamp (8)]` = **16-byte constant prefix**

- **Values**: MessagePack serialization (self-describing, no prefix scanning needed)

#### Performance Impact

- **Before** (MessagePack keys): O(N) scan from start of column family
- **After** (Direct encoding): O(K) seek directly to prefix, where K = matching keys
- **Improvement**: 1,000-10,000× faster for large databases!

Example: Query edges for a node in a 100K edge database:
- **MessagePack**: Scan ~50K keys (node in middle) = ~50ms
- **Direct encoding**: Scan ~10 keys (actual edges) = ~0.1ms
- **500× faster!**

### Schema Design Principles

1. **Fixed-Length Prefixes**: All variable-length fields (strings) placed at END of key tuples
   - ✅ `ForwardEdgeCfKey(SrcId, DstId, EdgeName)` - name at end
   - ✅ `ReverseEdgeCfKey(DstId, SrcId, EdgeName)` - name at end

2. **Prefix Extractors**: Configured per column family for O(1) prefix seek
   - ForwardEdges: 16-byte prefix (source_id)
   - ReverseEdges: 16-byte prefix (destination_id)
   - Fragments: 16-byte prefix (entity_id)

3. **Denormalization**: Edge topology stored in both `Edges` CF and index CFs for fast access

4. **ULID-based IDs**: 128-bit identifiers with timestamp prefix for natural ordering

See:
- [`docs/option2-implementation-outline.md`](docs/option2-implementation-outline.md) - Implementation details
- [`docs/rocksdb-prefix-scan-bug-analysis.md`](docs/rocksdb-prefix-scan-bug-analysis.md) - MessagePack analysis

## Query Processing

### Architecture

Queries use an async consumer-processor pattern with MPMC (multi-producer multi-consumer) channels:

```
Query.run(&reader)  →  MPMC Channel  →  Consumer Pool  →  Storage
         ↓                                                    ↓
         └─────────── oneshot channel ←──── Result ──────────┘
```

### Query Types

All queries implement the `Runnable<R>` trait parameterized by reader type:

**Graph queries** (`graph::query`):
- `NodeById`: Lookup node by ID → `(NodeName, NodeSummary)`
- `NodesByIdsMulti`: Batch lookup nodes by IDs → `Vec<(Id, NodeName, NodeSummary)>` (uses RocksDB MultiGet)
- `OutgoingEdges`: Get edges from a node → `Vec<(Option<f64>, SrcId, DstId, EdgeName)>`
- `IncomingEdges`: Get edges to a node → `Vec<(Option<f64>, DstId, SrcId, EdgeName)>`
- `EdgeSummaryBySrcDstName`: Lookup edge by topology → `(EdgeSummary, Option<f64>)`
- `NodeFragmentsByIdTimeRange`: Get node fragments → `Vec<(TimestampMilli, FragmentContent)>`
- `EdgeFragmentsByIdTimeRange`: Get edge fragments → `Vec<(TimestampMilli, FragmentContent)>`

**Fulltext queries** (`fulltext::query`):
- `Nodes`: Search nodes → `Vec<NodeHit>` (with scores)
- `Edges`: Search edges → `Vec<EdgeHit>` (with scores)
- `Facets`: Get tag facet counts → `FacetCounts`

**Unified queries** (`query`): Re-exports with graph hydration
- `Nodes`: Search + hydrate → `Vec<(Id, NodeName, NodeSummary)>`
- `Edges`: Search + hydrate → `Vec<(SrcId, DstId, EdgeName, EdgeSummary)>`
- Plus all graph query types with same return types

**Note**: Name-based lookups (finding nodes/edges by name) are handled by the fulltext search module.

### Query Processor Traits

Each module has its own `Processor` trait for storage access:

**Graph Processor** (`graph::reader::Processor`):
```rust
pub trait Processor: Send + Sync {
    /// Get access to the underlying storage for query execution
    fn storage(&self) -> &Storage;
}
```

**Fulltext Processor** (`fulltext::reader::Processor`):
```rust
pub trait Processor {
    /// Get a reference to the underlying Storage
    fn storage(&self) -> &Storage;
}
```

Query execution uses the `QueryExecutor` trait which takes `&Storage` directly:

```rust
#[async_trait]
pub trait QueryExecutor: Send + Sync {
    type Output: Send;
    async fn execute(&self, storage: &Storage) -> Result<Self::Output>;
    fn timeout(&self) -> Duration;
}
```

**Implementations**:
- `graph::Graph`: Implements `graph::reader::Processor`
- `fulltext::Index`: Implements `fulltext::reader::Processor`

### Mutation Processor Trait

The mutation `Processor` trait defines the contract for batch mutation processing:

```rust
#[async_trait]
pub trait Processor: Send + Sync {
    /// Process a batch of mutations atomically
    async fn process_mutations(&self, mutations: &[Mutation]) -> Result<()>;
}
```

**Key Design**:
- Single method handles all mutation types
- Receives a slice of mutations for efficient batching
- Implementation determines transaction boundaries
- Supports both single mutation (slice of 1) and true batching

**Implementations**:
- `graph::Graph`: Processes all mutations in a single RocksDB transaction
- `fulltext::Index`: Indexes all mutations in batch
- Test processors for mocking

### Consumer Pattern

The `spawn_query_consumer_pool_shared()` function creates a pool of async tasks that:

1. Receive queries from MPMC channel (flume)
2. Execute query via `QueryExecutor::execute(&storage)`
3. Handle timeouts
4. Send result via oneshot channel
5. Continue processing until channel closed

**Benefits**:
- Concurrent query execution with configurable pool size
- Backpressure via bounded channels
- Clean shutdown semantics (drop Reader to close channels)
- Testable via mock processors

## Mutation Processing

### Architecture

Mutations use a chainable consumer pattern with automatic batching:

```
Writer → Vec<Mutation> → MPSC Channel → Consumer₁ → Consumer₂ → ... → Consumerₙ
                                            ↓            ↓               ↓
                                         Storage    FullText          Custom
                                       (batched)   (batched)        (batched)
```

**Batching**: Mutations are sent as `Vec<Mutation>` through MPSC channels, enabling:
- Single RocksDB transaction for multiple mutations
- Improved throughput for bulk operations
- Atomic commits across all operations in a batch

### Mutation Types

```rust
pub enum Mutation {
    AddNode(AddNode),
    AddEdge(AddEdge),
    AddNodeFragment(AddNodeFragment),
    AddEdgeFragment(AddEdgeFragment),
    UpdateNodeValidSinceUntil(UpdateNodeValidSinceUntil),
    UpdateEdgeValidSinceUntil(UpdateEdgeValidSinceUntil),
    UpdateEdgeWeight(UpdateEdgeWeight),
}
```

Each mutation type implements `Runnable` for direct execution via `mutation.run(&writer).await`.

### Consumer Chaining

Consumers can forward mutations to the next consumer in the chain:

```rust
// Create FullText consumer (end of chain)
let fulltext_handle = fulltext::spawn_mutation_consumer(
    receiver_fulltext,
    config.clone(),
    &fulltext_index_path,
);

// Create Graph consumer (forwards to FullText)
let graph_handle = graph::spawn_mutation_consumer_with_next(
    receiver_graph,
    config,
    &db_path,
    fulltext_tx,  // Forward to FullText
);
```

**Processing Order**: Graph → FullText
- Graph updates RocksDB first
- Then forwards to FullText for indexing
- Ensures consistency (index only after successful storage)

### Graph Mutation Processing

The `Graph` struct implements `writer::Processor` for batched mutation processing:

```rust
#[async_trait]
impl writer::Processor for Graph {
    async fn process_mutations(&self, mutations: &[Mutation]) -> Result<()> {
        // Process all mutations in a single RocksDB transaction
        // Each mutation generates column family operations via mutation.plan()
        // All operations committed atomically
    }
}
```

**Key Benefits**:
- **True batching**: 1 mutation or 1000 mutations = 1 RocksDB transaction
- **Atomic writes**: All operations in a batch succeed or fail together
- **Multi-index consistency**: Edges + ForwardEdges + ReverseEdges updated atomically
- **Performance**: Significant speedup for bulk operations (see benchmarks)

## Data Types

### Id
```rust
pub struct Id([u8; 16]);
```
- 128-bit ULID (Universally Unique Lexicographically Sortable Identifier)
- Timestamp prefix for natural ordering
- Base32 string representation
- Implements: `Ord`, `Hash`, `Serialize`, `Deserialize`

### Type Aliases
```rust
pub type SrcId = Id;        // Self-documenting source node ID
pub type DstId = Id;        // Self-documenting destination node ID
pub type NodeName = String;
```

### EdgeName
```rust
pub struct EdgeName(pub String);
```
- Newtype wrapper for edge labels
- Used in both keys (ForwardEdges, ReverseEdges) and values (EdgeCfValue)

### Summary Types
```rust
pub struct NodeSummary(DataUrl);    // Markdown content
pub struct EdgeSummary(DataUrl);    // Markdown content
pub struct FragmentContent(DataUrl); // Markdown content
```

All summaries use **DataUrl** encoding:
- Format: `data:<mime>;charset=utf-8;base64,<content>`
- OpenAI-compliant
- Supports text, markdown, JSON, images
- Base64-encoded payload

## Graph Algorithm Support

The unified query API is designed to support common graph algorithms. This section assesses the current capabilities and proposes improvements.

### Supported Algorithms

| Algorithm | Status | API Used |
|-----------|--------|----------|
| **BFS** | ✅ Fully supported | `OutgoingEdges`, `NodeById` |
| **DFS** | ✅ Fully supported | `OutgoingEdges`, `NodeById` |
| **Dijkstra** | ✅ Fully supported | `OutgoingEdges` (weights included) |
| **A\*** | ✅ Fully supported | `OutgoingEdges` + heuristics |
| **Topological Sort** | ✅ Fully supported | `OutgoingEdges`, `IncomingEdges` |
| **PageRank** | ✅ Fully supported | `OutgoingEdges`, `IncomingEdges`, `AllNodes` |
| **Louvain (Community)** | ✅ Fully supported | `OutgoingEdges`, `IncomingEdges`, `AllEdges` |
| **Connected Components** | ✅ Fully supported | `OutgoingEdges`, `IncomingEdges` |
| **Cycle Detection** | ✅ Fully supported | `OutgoingEdges` with visited tracking |

### Example: BFS Implementation

```rust
use motlie_db::query::{NodeById, OutgoingEdges, Runnable};
use motlie_db::reader::Reader;
use motlie_db::Id;
use std::collections::{HashSet, VecDeque};

async fn bfs(
    start: Id,
    reader: &Reader,
    timeout: Duration,
) -> Result<Vec<String>> {
    let mut visited = HashSet::new();
    let mut result = Vec::new();
    let mut queue = VecDeque::new();

    queue.push_back(start);
    visited.insert(start);

    while let Some(current) = queue.pop_front() {
        // Get node data
        let (name, _summary) = NodeById::new(current, None)
            .run(reader, timeout)
            .await?;
        result.push(name);

        // Get neighbors
        let edges = OutgoingEdges::new(current, None)
            .run(reader, timeout)
            .await?;

        for (_weight, _src, dst, _edge_name) in edges {
            if !visited.contains(&dst) {
                visited.insert(dst);
                queue.push_back(dst);
            }
        }
    }

    Ok(result)
}
```

### Current Limitations

| Limitation | Impact | Affected Algorithms |
|------------|--------|---------------------|
| No batch node lookup | N sequential queries for N nodes | BFS, DFS, PageRank |
| No batch edge traversal | N sequential queries for N nodes | All traversal algorithms |
| No degree/count queries | Must fetch all edges to count | Degree centrality, hub detection |
| No edge name filtering | Post-filter in application | Typed edge traversal |
| No graph statistics | Full scan required | Any algorithm needing node/edge counts |
| Scans not in unified API | Must use `graph::scan` directly | Algorithms needing full iteration |

### Proposed Improvements

#### 1. Batch Node Lookup (`NodesByIds`)

**Problem**: BFS/DFS visiting N nodes makes N sequential `NodeById` calls.

**Proposed API**:

```rust
/// Batch lookup of multiple nodes by ID.
///
/// More efficient than multiple `NodeById` calls as it can:
/// - Batch RocksDB reads with MultiGet
/// - Reduce channel round-trips
/// - Enable parallel key lookups
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NodesByIds {
    /// Node IDs to look up
    pub ids: Vec<Id>,

    /// Reference timestamp for temporal validity (None = current time)
    pub reference_ts_millis: Option<TimestampMilli>,
}

impl NodesByIds {
    pub fn new(ids: Vec<Id>, reference_ts_millis: Option<TimestampMilli>) -> Self {
        Self { ids, reference_ts_millis }
    }
}

// Output: Vec<(Id, NodeName, NodeSummary)>
// Missing IDs are silently omitted from results
```

**Usage**:

```rust
// Instead of N sequential calls:
let mut nodes = Vec::new();
for id in node_ids {
    let (name, summary) = NodeById::new(id, None).run(&reader, timeout).await?;
    nodes.push((id, name, summary));
}

// Single batch call:
let nodes = NodesByIds::new(node_ids, None)
    .run(&reader, timeout)
    .await?;
```

**Expected speedup**: 10-50x for large traversals (reduced channel overhead + RocksDB MultiGet).

---

#### 2. Batch Edge Traversal (`OutgoingEdgesMulti`)

**Problem**: Getting neighbors for N nodes requires N sequential `OutgoingEdges` calls.

**Proposed API**:

```rust
/// Batch retrieval of outgoing edges for multiple source nodes.
///
/// Enables efficient multi-source BFS, parallel frontier expansion,
/// and batch neighbor lookups.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OutgoingEdgesMulti {
    /// Source node IDs
    pub ids: Vec<Id>,

    /// Optional edge name filter (only return edges with this name)
    pub edge_name_filter: Option<String>,

    /// Reference timestamp for temporal validity
    pub reference_ts_millis: Option<TimestampMilli>,
}

impl OutgoingEdgesMulti {
    pub fn new(ids: Vec<Id>, reference_ts_millis: Option<TimestampMilli>) -> Self {
        Self {
            ids,
            edge_name_filter: None,
            reference_ts_millis,
        }
    }

    pub fn with_edge_filter(mut self, name: String) -> Self {
        self.edge_name_filter = Some(name);
        self
    }
}

// Output: HashMap<Id, Vec<(Option<f64>, SrcId, DstId, EdgeName)>>
// Keys are source IDs, values are their outgoing edges
```

**Usage**:

```rust
// BFS frontier expansion - instead of N calls:
let frontier: Vec<Id> = queue.drain(..).collect();
let all_neighbors = OutgoingEdgesMulti::new(frontier, None)
    .run(&reader, timeout)
    .await?;

for (src, edges) in all_neighbors {
    for (_weight, _src, dst, _name) in edges {
        if !visited.contains(&dst) {
            visited.insert(dst);
            next_frontier.push(dst);
        }
    }
}
```

**Expected speedup**: 5-20x for BFS/DFS (batch prefix scans + reduced round-trips).

---

#### 3. Incoming Edges Multi (`IncomingEdgesMulti`)

**Proposed API**:

```rust
/// Batch retrieval of incoming edges for multiple destination nodes.
///
/// Useful for reverse traversal, computing in-degree, and
/// algorithms like PageRank that need incoming edge information.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IncomingEdgesMulti {
    /// Destination node IDs
    pub ids: Vec<Id>,

    /// Optional edge name filter
    pub edge_name_filter: Option<String>,

    /// Reference timestamp for temporal validity
    pub reference_ts_millis: Option<TimestampMilli>,
}

// Output: HashMap<Id, Vec<(Option<f64>, SrcId, DstId, EdgeName)>>
```

---

#### 4. Degree Queries (`OutgoingEdgeCount`, `IncomingEdgeCount`)

**Problem**: Computing node degree requires fetching all edges, even if only the count is needed.

**Proposed API**:

```rust
/// Get the count of outgoing edges from a node without fetching edge data.
///
/// Efficient for:
/// - Degree centrality computation
/// - Hub/authority detection
/// - Graph statistics
#[derive(Debug, Clone, PartialEq)]
pub struct OutgoingEdgeCount {
    /// Node ID to count edges for
    pub id: Id,

    /// Optional edge name filter (only count edges with this name)
    pub edge_name_filter: Option<String>,

    /// Reference timestamp for temporal validity
    pub reference_ts_millis: Option<TimestampMilli>,
}

impl OutgoingEdgeCount {
    pub fn new(id: Id, reference_ts_millis: Option<TimestampMilli>) -> Self {
        Self {
            id,
            edge_name_filter: None,
            reference_ts_millis,
        }
    }

    pub fn with_edge_filter(mut self, name: String) -> Self {
        self.edge_name_filter = Some(name);
        self
    }
}

// Output: usize
```

**Batch version**:

```rust
/// Batch degree computation for multiple nodes.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OutgoingEdgeCountMulti {
    pub ids: Vec<Id>,
    pub edge_name_filter: Option<String>,
    pub reference_ts_millis: Option<TimestampMilli>,
}

// Output: HashMap<Id, usize>
```

**Usage**:

```rust
// Find high-degree nodes (hubs)
let degrees = OutgoingEdgeCountMulti::new(node_ids, None)
    .run(&reader, timeout)
    .await?;

let hubs: Vec<Id> = degrees
    .into_iter()
    .filter(|(_, count)| *count > 100)
    .map(|(id, _)| id)
    .collect();
```

---

#### 5. Graph Statistics (`GraphStats`)

**Problem**: Getting node/edge counts requires full table scans.

**Proposed API**:

```rust
/// Retrieve graph-level statistics.
///
/// Returns approximate counts that may be slightly stale but are
/// much faster than full scans. Useful for:
/// - Progress reporting during algorithms
/// - Capacity planning
/// - Algorithm parameter tuning
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GraphStats;

/// Graph statistics result
#[derive(Debug, Clone, PartialEq)]
pub struct GraphStatsResult {
    /// Approximate node count
    pub node_count: u64,

    /// Approximate edge count
    pub edge_count: u64,

    /// Approximate node fragment count
    pub node_fragment_count: u64,

    /// Approximate edge fragment count
    pub edge_fragment_count: u64,
}

// Implementation note: Use RocksDB's GetApproximateSizes or
// maintain counters in a metadata column family
```

---

#### 6. Edge Name Filtering on Traversal

**Problem**: Typed graphs need to filter edges by relationship type during traversal.

**Proposed enhancement to existing queries**:

```rust
/// Enhanced OutgoingEdges with edge name filtering
#[derive(Debug, Clone, PartialEq)]
pub struct OutgoingEdges {
    /// Source node ID
    pub id: Id,

    /// NEW: Optional edge name filter
    /// When set, only edges matching this name are returned
    pub edge_name_filter: Option<String>,

    /// Reference timestamp for temporal validity
    pub reference_ts_millis: Option<TimestampMilli>,
}

impl OutgoingEdges {
    pub fn new(id: Id, reference_ts_millis: Option<TimestampMilli>) -> Self {
        Self {
            id,
            edge_name_filter: None,
            reference_ts_millis,
        }
    }

    /// Filter to only return edges with the specified name
    pub fn with_edge_filter(mut self, name: String) -> Self {
        self.edge_name_filter = Some(name);
        self
    }
}
```

**Usage**:

```rust
// Only traverse "follows" relationships
let follows = OutgoingEdges::new(user_id, None)
    .with_edge_filter("follows".to_string())
    .run(&reader, timeout)
    .await?;

// Only traverse "owns" relationships
let owned = OutgoingEdges::new(user_id, None)
    .with_edge_filter("owns".to_string())
    .run(&reader, timeout)
    .await?;
```

---

#### 7. Scan Queries in Unified API (`AllNodes`, `AllEdges`)

**Problem**: Full graph iteration requires using `graph::scan` directly, bypassing the unified API.

**Proposed API**:

```rust
/// Scan all nodes with pagination support.
///
/// Used for algorithms requiring full graph iteration:
/// - PageRank initialization
/// - Connected components
/// - Graph export
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ScanNodes {
    /// Maximum nodes to return
    pub limit: usize,

    /// Cursor for pagination (last seen node ID)
    pub cursor: Option<Id>,

    /// Scan direction
    pub reverse: bool,

    /// Reference timestamp for temporal validity
    pub reference_ts_millis: Option<TimestampMilli>,
}

impl ScanNodes {
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            cursor: None,
            reverse: false,
            reference_ts_millis: None,
        }
    }

    pub fn with_cursor(mut self, cursor: Id) -> Self {
        self.cursor = Some(cursor);
        self
    }

    pub fn reverse(mut self) -> Self {
        self.reverse = true;
        self
    }
}

// Output: Vec<(Id, NodeName, NodeSummary)>
// Use cursor from last result for pagination
```

**Usage**:

```rust
// Iterate all nodes for PageRank initialization
let mut all_nodes = Vec::new();
let mut cursor = None;

loop {
    let mut query = ScanNodes::new(1000);
    if let Some(c) = cursor {
        query = query.with_cursor(c);
    }

    let batch = query.run(&reader, timeout).await?;
    if batch.is_empty() {
        break;
    }

    cursor = batch.last().map(|(id, _, _)| *id);
    all_nodes.extend(batch);
}
```

---

### Implementation Priority

| Feature | Priority | Effort | Impact |
|---------|----------|--------|--------|
| `NodesByIds` | 🔴 High | Medium | 10-50x speedup for traversals |
| `OutgoingEdgesMulti` | 🔴 High | Medium | 5-20x speedup for BFS/DFS |
| Edge name filtering | 🟡 Medium | Low | Cleaner typed graph traversal |
| `OutgoingEdgeCount` | 🟡 Medium | Low | Efficient degree computation |
| `ScanNodes`/`ScanEdges` | 🟡 Medium | Low | Unified API consistency |
| `IncomingEdgesMulti` | 🟢 Low | Medium | Reverse traversal batching |
| `GraphStats` | 🟢 Low | Medium | Progress reporting |

### Quality Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Type Safety** | ⭐⭐⭐⭐⭐ | Strong types, no raw strings for IDs |
| **API Consistency** | ⭐⭐⭐⭐⭐ | Same `Runnable::run(&reader, timeout)` pattern |
| **Documentation** | ⭐⭐⭐⭐ | Good module docs, comprehensive README |
| **Ergonomics** | ⭐⭐⭐⭐ | Builder pattern + struct init both work |
| **Algorithm Support** | ⭐⭐⭐ | All major algorithms work, but not optimally |
| **Batch Performance** | ⭐⭐ | Sequential queries only, no batching |

The unified API provides a solid foundation for graph algorithms. The proposed batch queries would significantly improve performance for production workloads while maintaining the clean, consistent API design.

## Storage Modes

### Graph Storage

```rust
use motlie_db::graph::Storage;

// ReadWrite mode - for mutations
let mut storage = Storage::new("/path/to/db");
storage.ready()?;  // Create/open database

// ReadOnly mode - for queries only
let storage = Storage::readonly("/path/to/db");

// Secondary mode - follows primary
let secondary = Storage::secondary("/path/to/db", "/path/to/secondary");
```

### Unified Storage

```rust
use motlie_db::reader::{Storage, ReaderConfig, StorageHandle};

// Initialize both graph and fulltext
let storage = Storage::readonly(graph_path, fulltext_path);
// Or for read-write: Storage::readwrite(graph_path, fulltext_path)

// Get handle with lifecycle management
let handle = storage.ready(ReaderConfig::default(), 4)?;

// Use reader for queries
let reader = handle.reader();

// Clean shutdown
handle.shutdown().await?;
```

**Concurrency**:
- 1 ReadWrite + N ReadOnly instances can coexist
- Secondary instances follow primary via catch-up
- ReadOnly instances are independent

## Full-Text Search

### Index

Maintains a Tantivy search index synchronized with graph mutations:

```rust
use motlie_db::fulltext::{Storage, Index};

let mut storage = Storage::new("/path/to/index");
storage.ready()?;

let index = Index::new(Arc::new(storage));
```

**Indexed Fields**:
- `id`: Entity ID (STRING, stored)
- `doc_type`: "node" | "edge" | "node_fragment" | "edge_fragment" (STRING, indexed)
- `name`: Node/edge name (TEXT, indexed + stored)
- `summary`: Summary content (TEXT, indexed + stored)
- `content`: Fragment content (TEXT, indexed + stored)
- `tags`: Extracted hashtags (FACET, indexed)
- `src_id`, `dst_id`: Edge endpoints (STRING, stored)
- `edge_name`: Edge type (STRING, stored)
- `valid_since`, `valid_until`: Temporal validity (U64, indexed)
- `score`: Relevance score (F64, stored)

**Mutation Handling**:
- `AddNode`: Index node name and summary
- `AddEdge`: Index edge with topology
- `AddNodeFragment`: Index fragment content with tags
- `AddEdgeFragment`: Index edge fragment content

**Consumer Chain**: Graph → FullText ensures search index stays in sync

## Testing

Run all tests:
```bash
cargo test --lib
```

Run specific test:
```bash
cargo test --lib test_forward_edges_keys_lexicographically_sortable
```

### Test Coverage

- ✅ **161 passing tests** (137 unit + 24 integration)
- Storage lifecycle (open, close, reopen)
- ReadOnly/ReadWrite/Secondary concurrency
- Query processing with timeouts
- **Transaction batching** (single, multi, large batches, atomicity)
- Mutation processing and chaining
- Schema serialization and ordering
- Full-text indexing
- Graph topology queries
- Concurrent readonly and secondary access

## Examples

See `examples/store/main.rs` for a complete example demonstrating:

1. CSV ingestion (nodes, edges, fragments)
2. Graph consumer → FullText consumer chaining
3. Database verification against input data

Run the example:
```bash
# Store data
cat input.csv | cargo run --release --example store /tmp/motlie_db

# Verify data
cat input.csv | cargo run --release --example store --verify /tmp/motlie_db
```

## Documentation

Comprehensive design documentation available in [`docs/`](docs/):

### Essential Reading

- **[Query API Guide](docs/query-api-guide.md)** ⭐ - Complete guide to the modern Query API (v0.2.0+)
- **[Concurrency & Storage Modes](docs/concurrency-and-storage-modes.md)** - Threading patterns and concurrent access
- **[Schema Design](docs/variable-length-fields-in-keys.md)** - RocksDB key design principles

### Design Documentation

- **API Evolution**: Query and mutation processor architecture
- **Performance**: Prefix scanning optimization and benchmarks
- **Implementation**: Trait-based execution patterns

See [`docs/README.md`](docs/README.md) for complete documentation index.

## Performance Characteristics

### Query Performance

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| `node_by_id()` | O(1) | Direct RocksDB get |
| `edge_by_id()` | O(1) | Direct RocksDB get, returns full topology |
| `outgoing_edges_by_id()` | O(N) | N = edges from node, uses prefix scan |
| `incoming_edges_by_id()` | O(N) | N = edges to node, uses prefix scan |
| `fragments_by_id()` | O(F) | F = fragments for entity, prefix scan |

### Mutation Performance

**Transaction Batching Benefits**:
- **Single mutation**: Wrapped in Vec, processed in 1 transaction
- **Batched mutations**: All mutations in Vec processed in 1 transaction
- **Performance gain**: ~500-1000% speedup for bulk operations

**Example**: Writing 100 nodes
- **Individual (100 transactions)**: ~500ms
- **Batched (1 transaction)**: ~50ms
- **Speedup**: 10x faster

See `test_transaction_batching_vs_individual_performance` for benchmarking code.

### Storage Overhead

- **Edge denormalization**: 3x storage (Edges + ForwardEdges + ReverseEdges)
  - Trade-off: Fast bidirectional navigation vs storage
  - IDs are only 16 bytes each
  - Acceptable overhead for query performance

### MessagePack Efficiency

- **Compact encoding**: ~33 bytes for typical edge tuple
- **No padding**: Tightly packed binary format
- **Self-describing**: Type information embedded in stream
- **Lexicographically sortable**: Natural ordering preserved

## Telemetry

`motlie-db` uses the `tracing` crate for structured logging and distributed tracing. Telemetry initialization helpers are provided by the `motlie-core` crate.

### Development (stderr logging)

For development, use the simple stderr subscriber:

```rust
use motlie_core::telemetry;

fn main() {
    // Simple stderr logging at DEBUG level
    telemetry::init_dev_subscriber();

    // Or with RUST_LOG environment variable support
    telemetry::init_dev_subscriber_with_env_filter();

    // Application code...
}
```

Control log level via environment variable:

```bash
# Show debug logs
RUST_LOG=debug cargo run

# Show only motlie_db debug logs, info for others
RUST_LOG=info,motlie_db=debug cargo run

# Show warnings and errors only
RUST_LOG=warn cargo run
```

### Production (OpenTelemetry)

For production deployments with distributed tracing, enable the `dtrace-otel` feature on `motlie-core`:

```toml
# Cargo.toml
[dependencies]
motlie-core = { version = "0.1", features = ["dtrace-otel"] }
```

Then initialize with OpenTelemetry:

```rust
use motlie_core::telemetry;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize OpenTelemetry with OTLP exporter
    telemetry::init_otel_subscriber("my-service", "http://localhost:4317")?;

    // Or with RUST_LOG environment variable support
    telemetry::init_otel_subscriber_with_env_filter("my-service", "http://localhost:4317")?;

    // Application code...
    Ok(())
}
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RUST_LOG` | Log level filter (e.g., `debug`, `info`, `warn`) | `debug` (dev) / `info` (otel) |
| `DTRACE_ENDPOINT` | OTLP collector endpoint URL | None (required for otel) |
| `DTRACE_SERVICE_NAME` | Service name for traces | Application-defined |

### Instrumented Spans

Key functions are instrumented with `#[tracing::instrument]` for automatic span creation. Each span includes relevant fields for filtering and analysis.

#### Graph Module

| Span Name | Function | Fields | Description |
|-----------|----------|--------|-------------|
| `graph::Storage::ready` | `Storage::ready()` | `path` | Opens RocksDB database |
| `graph::Graph::process_mutations` | `Graph::process_mutations()` | `mutation_count` | Processes mutation batch in single transaction |
| `mutation_consumer` | `Consumer::run()` | - | Long-running mutation consumer loop |
| `graph::Consumer::process_batch` | `Consumer::process_batch()` | `batch_size` | Processes individual mutation batch |
| `query_consumer` | `Consumer::run()` | - | Long-running query consumer loop |
| `graph::Consumer::process_query` | `Consumer::process_query()` | `query_type` | Processes individual query |

#### Fulltext Module

| Span Name | Function | Fields | Description |
|-----------|----------|--------|-------------|
| `fulltext::Storage::ready` | `Storage::ready()` | `path`, `mode` | Opens Tantivy index |
| `fulltext::Index::process_mutations` | `Index::process_mutations()` | `mutation_count` | Indexes mutation batch |
| `fulltext_query_consumer` | `Consumer::run()` | - | Long-running fulltext query consumer |

### Tracing Events

The library emits structured events at various log levels:

#### INFO Events

| Location | Event | Fields | Description |
|----------|-------|--------|-------------|
| `graph/mod.rs` | `[Storage] Ready` | - | Database opened successfully |
| `graph/mod.rs` | `[Graph] About to insert mutations` | `count` | Starting mutation batch |
| `graph/mod.rs` | `[Graph] Successfully committed mutations` | `count` | Mutation batch committed |
| `graph/writer.rs` | `Starting mutation consumer` | `config` | Consumer started |
| `graph/writer.rs` | `Mutation consumer shutting down` | - | Channel closed |
| `graph/reader.rs` | `Starting query consumer` | `config` | Consumer started |
| `graph/reader.rs` | `Query consumer shutting down` | - | Channel closed |
| `graph/reader.rs` | `Query worker starting` | `worker_id` | Pool worker started |
| `graph/reader.rs` | `Query worker shutting down` | `worker_id` | Pool worker stopped |
| `graph/reader.rs` | `Spawned query consumer workers` | `num_workers` | Worker pool created |
| `graph/mutation.rs` | `[UpdateNodeValidSinceUntil]` | `node_id`, `edge_count` | Temporal update |
| `fulltext/mod.rs` | `[FullText Storage] Opened index` | `path` | Index opened |
| `fulltext/writer.rs` | `[FullText] Processing mutations` | `count` | Starting indexing |
| `fulltext/writer.rs` | `[FullText] Successfully indexed` | `count` | Indexing complete |
| `fulltext/reader.rs` | `Starting fulltext query consumer` | `config` | Consumer started |
| `fulltext/reader.rs` | `Fulltext query worker starting` | `worker_id` | Pool worker started |

#### DEBUG Events

##### Graph Query Executors

| Location | Event | Fields | Description |
|----------|-------|--------|-------------|
| `graph/query.rs` | `Executing NodeById query` | `id` | Lookup node by ID |
| `graph/query.rs` | `Executing NodesByIdsMulti query` | `count` | Batch lookup nodes by IDs |
| `graph/query.rs` | `Executing NodeFragmentsByIdTimeRange query` | `id`, `time_range` | Query node fragments |
| `graph/query.rs` | `Executing EdgeFragmentsByIdTimeRange query` | `src_id`, `dst_id`, `edge_name`, `time_range` | Query edge fragments |
| `graph/query.rs` | `Executing EdgeSummaryBySrcDstName query` | `src_id`, `dst_id`, `name` | Lookup edge by topology |
| `graph/query.rs` | `Executing OutgoingEdges query` | `node_id` | Get edges from node |
| `graph/query.rs` | `Executing IncomingEdges query` | `node_id` | Get edges to node |

##### Graph Mutation Executors

| Location | Event | Fields | Description |
|----------|-------|--------|-------------|
| `graph/mutation.rs` | `Executing AddNode mutation` | `id`, `name` | Add node to graph |
| `graph/mutation.rs` | `Executing AddEdge mutation` | `src`, `dst`, `name` | Add edge to graph |
| `graph/mutation.rs` | `Executing AddNodeFragment mutation` | `id`, `ts`, `content_len` | Add node fragment |
| `graph/mutation.rs` | `Executing AddEdgeFragment mutation` | `src`, `dst`, `edge_name`, `ts`, `content_len` | Add edge fragment |
| `graph/mutation.rs` | `Executing UpdateEdgeValidSinceUntil mutation` | `src`, `dst`, `name`, `reason` | Update edge validity |
| `graph/mutation.rs` | `Executing UpdateEdgeWeight mutation` | `src`, `dst`, `name`, `weight` | Update edge weight |

##### Graph Scan Executors

| Location | Event | Fields | Description |
|----------|-------|--------|-------------|
| `graph/scan.rs` | `Executing AllNodes scan` | `limit`, `reverse`, `has_cursor` | Scan all nodes |
| `graph/scan.rs` | `Executing AllEdges scan` | `limit`, `reverse`, `has_cursor` | Scan forward edges |
| `graph/scan.rs` | `Executing AllReverseEdges scan` | `limit`, `reverse`, `has_cursor` | Scan reverse edges |
| `graph/scan.rs` | `Executing AllNodeFragments scan` | `limit`, `reverse`, `has_cursor` | Scan node fragments |
| `graph/scan.rs` | `Executing AllEdgeFragments scan` | `limit`, `reverse`, `has_cursor` | Scan edge fragments |

##### Graph Writer Processing

| Location | Event | Fields | Description |
|----------|-------|--------|-------------|
| `graph/writer.rs` | `Processing AddNode` | `id`, `name` | Node mutation received |
| `graph/writer.rs` | `Processing AddEdge` | `source`, `target`, `name` | Edge mutation received |
| `graph/writer.rs` | `Processing AddNodeFragment` | `id`, `body_len` | Fragment received |
| `graph/writer.rs` | `Processing AddEdgeFragment` | `src`, `dst`, `name`, `body_len` | Edge fragment received |
| `graph/writer.rs` | `Processing UpdateNodeValidSinceUntil` | `id`, `reason` | Temporal update received |
| `graph/writer.rs` | `Processing UpdateEdgeValidSinceUntil` | `src`, `dst`, `name`, `reason` | Edge temporal received |
| `graph/writer.rs` | `Processing UpdateEdgeWeight` | `src`, `dst`, `name`, `weight` | Weight update received |
| `graph/reader.rs` | `Processing query` | `query` | Query dispatched |

##### Fulltext Query Executors

| Location | Event | Fields | Description |
|----------|-------|--------|-------------|
| `fulltext/query.rs` | `Executing fulltext Nodes query` | `query`, `fuzzy_level`, `limit` | Search nodes |
| `fulltext/query.rs` | `[FulltextEdges] Executing query` | `query`, `index_docs` | Search edges |
| `fulltext/query.rs` | `[FulltextEdges] Search returned` | `result_count`, `query` | Search results |
| `fulltext/query.rs` | `Executing fulltext Facets query` | `doc_type_filter` | Get facet counts |

##### Fulltext Mutation Executors

| Location | Event | Fields | Description |
|----------|-------|--------|-------------|
| `fulltext/mutation.rs` | `[FullText] Indexed node` | `id`, `name`, `valid_range` | Node indexed |
| `fulltext/mutation.rs` | `[FullText] Indexed edge` | `src`, `dst`, `name`, `valid_range` | Edge indexed |
| `fulltext/mutation.rs` | `[FullText] Indexed node fragment` | `id`, `content_len`, `valid_range` | Fragment indexed |
| `fulltext/mutation.rs` | `[FullText] Indexed edge fragment` | `src`, `dst`, `name`, `content_len` | Edge fragment indexed |
| `fulltext/mutation.rs` | `[FullText] Deleted node documents` | `id`, `reason` | Node temporal update |
| `fulltext/mutation.rs` | `[FullText] Deleted edge documents` | `src`, `dst`, `name`, `reason` | Edge temporal update |
| `fulltext/mutation.rs` | `[FullText] Edge weight updated` | `src`, `dst`, `name`, `weight` | Weight update (no-op) |

#### WARN Events

| Location | Event | Fields | Description |
|----------|-------|--------|-------------|
| `graph/writer.rs` | `[BUFFER FULL] Next consumer busy` | `err`, `count` | Mutations dropped due to backpressure |

#### ERROR Events

| Location | Event | Fields | Description |
|----------|-------|--------|-------------|
| `graph/reader.rs` | `Query worker failed to ready storage` | `worker_id`, `err` | Worker initialization failed |
| `fulltext/reader.rs` | `Worker failed to ready storage` | `worker_id`, `err` | Worker initialization failed |

### Example Trace Output

#### Development (stderr)

```
2024-01-15T10:30:00.123Z DEBUG motlie_db::graph::mod path="/data/graph" [Storage] Ready
2024-01-15T10:30:00.125Z  INFO motlie_db::graph::writer config=WriterConfig { channel_buffer_size: 100 } Starting mutation consumer
2024-01-15T10:30:00.130Z  INFO motlie_db::graph::mod count=5 [Graph] About to insert mutations
2024-01-15T10:30:00.131Z DEBUG motlie_db::graph::writer id=01HQXYZ123 name="Alice" Processing AddNode
2024-01-15T10:30:00.132Z DEBUG motlie_db::graph::writer id=01HQXYZ456 name="Bob" Processing AddNode
2024-01-15T10:30:00.133Z DEBUG motlie_db::graph::writer source=01HQXYZ123 target=01HQXYZ456 name="follows" Processing AddEdge
2024-01-15T10:30:00.140Z  INFO motlie_db::graph::mod count=5 [Graph] Successfully committed mutations
2024-01-15T10:30:00.141Z  INFO motlie_db::fulltext::writer count=5 [FullText] Processing mutations for indexing
2024-01-15T10:30:00.150Z DEBUG motlie_db::fulltext::mutation id=01HQXYZ123 name="Alice" valid_range=None [FullText] Indexed node
2024-01-15T10:30:00.160Z  INFO motlie_db::fulltext::writer count=5 [FullText] Successfully indexed mutations
```

#### OpenTelemetry (Jaeger/Tempo)

With OpenTelemetry enabled, spans are exported with full context:

```
Trace: mutation_consumer (duration: 45ms)
├── process_batch (batch_size=5, duration: 40ms)
│   ├── Graph::process_mutations (mutation_count=5, duration: 15ms)
│   │   └── [Graph] About to insert mutations (count=5)
│   │   └── Processing AddNode (id=01HQXYZ123, name="Alice")
│   │   └── Processing AddNode (id=01HQXYZ456, name="Bob")
│   │   └── Processing AddEdge (source=01HQXYZ123, target=01HQXYZ456)
│   │   └── [Graph] Successfully committed mutations (count=5)
│   └── Index::process_mutations (mutation_count=5, duration: 20ms)
│       └── [FullText] Processing mutations for indexing (count=5)
│       └── [FullText] Indexed node (id=01HQXYZ123, name="Alice")
│       └── [FullText] Successfully indexed mutations (count=5)
```

### Filtering Traces

Use `RUST_LOG` to filter by module or level:

```bash
# All debug logs from motlie_db
RUST_LOG=motlie_db=debug

# Only graph module at debug, others at info
RUST_LOG=info,motlie_db::graph=debug

# Only fulltext indexing
RUST_LOG=warn,motlie_db::fulltext::writer=info

# Mutation processing only
RUST_LOG=warn,motlie_db::graph::writer=debug,motlie_db::graph::mod=info
```

## Dependencies

- **rocksdb** (0.24): Embedded key-value store
- **tokio**: Async runtime
- **flume** (0.11): MPMC channels for mutation pipeline
- **ferroid** (0.8): ULID implementation
- **rmp-serde** (1.3): MessagePack serialization
- **tantivy**: Full-text search engine
- **data-url** (0.3): DataUrl encoding/decoding
- **tracing** (0.1): Structured logging and tracing
- **async-trait**: Async trait support

### Telemetry Dependencies (in motlie-core)

Telemetry initialization is provided by `motlie-core`. When using the `dtrace-otel` feature:

- **tracing-subscriber** (0.3): Subscriber implementations
- **tracing-opentelemetry**: OpenTelemetry integration for tracing
- **opentelemetry**: OpenTelemetry API
- **opentelemetry-otlp**: OTLP exporter
- **opentelemetry_sdk**: OpenTelemetry SDK with Tokio runtime

## License

See workspace LICENSE file.
