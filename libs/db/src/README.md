# Source Code Organization

This document describes the design patterns and file organization used across the `graph` and `fulltext` modules.

## Directory Structure

```
src/
├── lib.rs              # Crate root - common types (Id, TimestampMilli, DataUrl, TemporalRange)
├── README.md           # This file
├── graph/              # RocksDB-based graph storage
│   ├── mod.rs          # Storage, Graph struct, module exports
│   ├── schema.rs       # Column family definitions, key/value types
│   ├── mutation.rs     # Mutation types (AddNode, AddEdge, etc.)
│   ├── writer.rs       # Writer/MutationConsumer infrastructure
│   ├── query.rs        # Query types (NodeById, OutgoingEdges, etc.)
│   ├── reader.rs       # Reader/QueryConsumer infrastructure
│   ├── scan.rs         # Pagination/iteration API
│   └── tests.rs        # Module-level tests
└── fulltext/           # Tantivy-based fulltext search
    ├── mod.rs          # Storage, Index struct, module exports
    ├── mutation.rs     # Mutation processing for index updates
    ├── writer.rs       # Writer/MutationConsumer infrastructure
    ├── query.rs        # Query types (Nodes fulltext search)
    ├── reader.rs       # Reader/QueryConsumer infrastructure
    ├── search.rs       # Low-level search utilities
    ├── fuzzy.rs        # Fuzzy search implementation
    └── README.md       # Fulltext-specific documentation
```

## Consistent Design Pattern

Both `graph` and `fulltext` modules follow the same architectural pattern for consistency and maintainability.

### File Responsibilities

| File | Purpose |
|------|---------|
| `mod.rs` | Module entry point with `Storage` and processor (`Graph`/`Index`) structs, re-exports |
| `schema.rs` | Data schema definitions (graph only - column families, key/value types) |
| `mutation.rs` | Mutation types and the `Runnable` trait for executing mutations |
| `writer.rs` | `Writer`, `WriterConfig`, mutation consumer spawn functions, `Processor` trait |
| `query.rs` | Query types and the `Runnable` trait for executing queries |
| `reader.rs` | `Reader`, `ReaderConfig`, query consumer spawn functions, `Processor` trait |
| `scan.rs` | Pagination API for iterating over records (graph only) |
| `search.rs` | Low-level search utilities (fulltext only) |
| `fuzzy.rs` | Fuzzy/typo-tolerant search (fulltext only) |

### Core Types Comparison

| Concept | Graph Module | Fulltext Module |
|---------|--------------|-----------------|
| **Storage Backend** | `graph::Storage` (RocksDB) | `fulltext::Storage` (Tantivy) |
| **Processor** | `graph::Graph` | `fulltext::Index` |
| **Writer** | `graph::Writer` | Uses graph `Writer` |
| **Reader** | `graph::Reader` | `fulltext::Reader` |
| **Mutation Trait** | `mutation::Runnable` | N/A (uses graph mutations) |
| **Query Trait** | `query::Runnable` | `query::Runnable` |

### Storage Layer

Both modules use a `Storage` struct that:
- Supports multiple access modes (readonly vs readwrite)
- Requires explicit `ready()` initialization
- Is wrapped in `Arc` for shared ownership

```rust
// Graph Storage
let mut storage = graph::Storage::readwrite(&db_path);
storage.ready()?;
let storage = Arc::new(storage);

// Fulltext Storage (identical pattern)
let mut storage = fulltext::Storage::readwrite(&index_path);
storage.ready()?;
let storage = Arc::new(storage);
```

### Processor Layer

Both modules wrap `Arc<Storage>` in a processor type that provides the execution interface:

```rust
// Graph
let graph = Graph::new(Arc::new(storage));

// Fulltext
let index = fulltext::Index::new(Arc::new(storage));
```

### Writer/Reader Infrastructure

Both modules use channel-based async processing:

```
Writer → mutations → MPSC Channel → MutationConsumer → Processor → Storage
Reader → queries   → MPMC Channel → QueryConsumer    → Processor → Storage
```

### Spawn Function Naming Convention

| Pattern | Graph | Fulltext |
|---------|-------|----------|
| Basic mutation consumer | `spawn_graph_consumer` | `spawn_fulltext_consumer` |
| Mutation consumer with chaining | `spawn_graph_consumer_with_next` | `spawn_fulltext_mutation_consumer_with_next` |
| Basic query consumer | `spawn_graph_query_consumer` | `spawn_fulltext_query_consumer` |
| Query consumer with processor | `spawn_graph_query_consumer_with_graph` | N/A |
| Shared pool query consumers | `spawn_graph_query_consumer_pool_shared` | `spawn_fulltext_query_consumer_pool_shared` |
| Readonly pool query consumers | `spawn_graph_query_consumer_pool_readonly` | `spawn_fulltext_query_consumer_pool_readonly` |

## Mutation Flow

Mutations flow through a pipeline from the client to storage:

```
                          ┌──────────────────┐
   Client                 │  Mutation Types  │
     │                    │  - AddNode       │
     │ mutations          │  - AddEdge       │
     ▼                    │  - AddNodeFrag   │
┌─────────┐               │  - AddEdgeFrag   │
│ Writer  │               └──────────────────┘
└────┬────┘
     │ Vec<Mutation>
     ▼
┌────────────────┐       ┌────────────────┐
│ Graph Consumer │──────▶│Fulltext Consumer│
│   (RocksDB)    │ chain │   (Tantivy)     │
└────────────────┘       └────────────────┘
```

### Consumer Chaining

The Graph consumer can forward mutations to the Fulltext consumer:

```rust
// Create fulltext consumer (end of chain)
let (fulltext_tx, fulltext_rx) = mpsc::channel(1000);
let fulltext_handle = spawn_fulltext_consumer(fulltext_rx, config, &index_path);

// Create graph consumer that chains to fulltext
let (writer, graph_rx) = create_mutation_writer(config);
let graph_handle = spawn_graph_consumer_with_next(
    graph_rx,
    config,
    &db_path,
    fulltext_tx,  // Forward to fulltext
);
```

## Query Flow

Queries flow from clients through readers to consumers:

```
   Client 1 ─┐
   Client 2 ─┼─▶ Reader ──▶ MPMC Channel ──▶ ┌─────────────────┐
   Client N ─┘                               │ Query Consumers │
                                             │  ┌─────────┐    │
                                             │  │Worker 1 │    │
                                             │  └────┬────┘    │
                                             │       ▼         │
                                             │  ┌─────────┐    │
                                             │  │ Graph/  │    │
                                             │  │ Index   │    │
                                             │  └─────────┘    │
                                             └─────────────────┘
```

### Shared vs Readonly Pools

- **Shared Pool** (`spawn_*_pool_shared`): All workers share one `Arc<Graph>` or `Arc<Index>`
- **Readonly Pool** (`spawn_*_pool_readonly`): Each worker has its own readonly storage instance

## lib.rs: Crate Root

The crate root (`lib.rs`) contains:

1. **Common Types**: `Id`, `TimestampMilli`, `DataUrl`, `TemporalRange`
2. **Module Declarations**: `pub mod graph` and `pub mod fulltext`
3. **Re-exports**: Flattened exports for ergonomic imports

```rust
// Users can import from crate root
use motlie_db::{
    // Common types
    Id, TimestampMilli, DataUrl, TemporalRange,
    // Graph types
    Graph, Storage, AddNode, AddEdge, NodeById, OutgoingEdges,
    // Fulltext types
    FulltextIndex, FulltextStorage, FulltextNodes,
    // Infrastructure
    Writer, WriterConfig, Reader, ReaderConfig,
    spawn_graph_consumer, spawn_fulltext_consumer,
};

// Or import from submodules directly
use motlie_db::graph::{Storage, Graph, schema::Nodes};
use motlie_db::fulltext::{Storage as FtStorage, Index};
```

## Testing Organization

Tests are organized as follows:

| Location | Purpose |
|----------|---------|
| `src/lib.rs` (bottom) | Common type tests (Id, DataUrl, etc.) |
| `src/graph/tests.rs` | Graph module integration tests |
| `src/graph/*.rs` (inline) | Unit tests for specific modules |
| `src/fulltext/*.rs` (inline) | Unit tests for specific modules |
| `tests/*.rs` | Integration tests across modules |

## Adding New Features

When adding a new feature, follow the established pattern:

1. **New Query Type**: Add to `query.rs`, implement `QueryExecutor`, add re-export in `mod.rs`
2. **New Mutation Type**: Add to `mutation.rs`, implement `Runnable`, add to `Mutation` enum
3. **New Consumer Function**: Add to `writer.rs` or `reader.rs`, add re-export in `mod.rs` and `lib.rs`
4. **New Schema**: Add column family to `schema.rs`, update `ALL_COLUMN_FAMILIES`

## See Also

- [`graph/README.md`](graph/README.md) - Graph module specifics (if exists)
- [`fulltext/README.md`](fulltext/README.md) - Fulltext module specifics
- [`../README.md`](../README.md) - Crate-level documentation
- [`../docs/`](../docs/) - Design documents
