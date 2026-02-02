# Graph Module

This module provides RocksDB-based graph storage with async query/mutation processing.

## File Organization

```
graph/
├── mod.rs       # Storage, Graph struct, module exports
├── schema.rs    # Column family definitions, key/value types
├── mutation.rs  # Mutation types (AddNode, AddEdge, etc.)
├── writer.rs    # Writer/MutationConsumer infrastructure
├── query.rs     # Query types (NodeById, OutgoingEdges, etc.)
├── reader.rs    # Reader/QueryConsumer infrastructure
├── scan.rs      # Pagination/iteration API
├── tests.rs     # Module-level integration tests
└── README.md    # This file
```

## Core Types

### Storage

`Storage` manages the RocksDB connection with support for multiple access modes:

```rust
use motlie_db::graph::Storage;

// ReadWrite mode (exclusive write access)
let mut storage = Storage::readwrite(&db_path);
storage.ready()?;

// ReadOnly mode (concurrent read access)
let mut storage = Storage::readonly(&db_path);
storage.ready()?;

// Secondary mode (follows primary)
let mut storage = Storage::secondary(&db_path, &secondary_path);
storage.ready()?;
storage.try_catch_up_with_primary()?;
```

### Graph

`Graph` wraps `Arc<Storage>` and provides the execution interface:

```rust
use motlie_db::{Graph, Storage};
use std::sync::Arc;

let mut storage = Storage::readwrite(&db_path);
storage.ready()?;
let graph = Graph::new(Arc::new(storage));
```

## Mutation Types

Defined in `mutation.rs`:

| Type | Description |
|------|-------------|
| `AddNode` | Create a new node with name and summary |
| `AddEdge` | Create an edge between two nodes |
| `AddNodeFragment` | Add timestamped content fragment to a node |
| `AddEdgeFragment` | Add timestamped content fragment to an edge |
| `UpdateNodeValidSinceUntil` | Update node's temporal validity range |
| `UpdateEdgeValidSinceUntil` | Update edge's temporal validity range |
| `UpdateEdgeWeight` | Update edge weight |

### Runnable Trait

All mutations implement `writer::Runnable` (defined in `motlie_db::writer`):

```rust
use motlie_db::graph::mutation::AddNode;
use motlie_db::writer::Runnable;  // Unified mutation trait
use motlie_db::{Id, TimestampMilli};
use motlie_db::graph::schema::NodeSummary;

let node = AddNode {
    id: Id::new(),
    ts_millis: TimestampMilli::now(),
    name: "Alice".to_string(),
    summary: NodeSummary::from_text("A person"),
    valid_range: None,
};

// Execute via writer
node.run(&writer).await?;
```

## Query Types

Defined in `query.rs`:

| Type | Description |
|------|-------------|
| `NodeById` | Get node by ID |
| `OutgoingEdges` | Get edges from a node |
| `IncomingEdges` | Get edges to a node |
| `EdgeSummaryBySrcDstName` | Get edge by (src, dst, name) |
| `NodeFragmentsByIdTimeRange` | Get node fragments in time range |
| `EdgeFragmentsByIdTimeRange` | Get edge fragments in time range |

**Note**: Name-based lookups (finding nodes/edges by name) are handled by the fulltext search module.

### Runnable Trait

All queries implement `reader::Runnable` (defined in `motlie_db::reader`):

```rust
use motlie_db::graph::query::NodeById;
use motlie_db::reader::Runnable;  // Unified query trait
use std::time::Duration;

let result = NodeById::new(node_id, None)
    .run(&reader, Duration::from_secs(5))
    .await?;
```

## Spawn Functions

### Mutation Consumers

| Function | Description |
|----------|-------------|
| `spawn_mutation_consumer(receiver, config, path)` | Creates storage and processes mutations |
| `spawn_mutation_consumer_with_next(receiver, config, path, next_tx)` | Chains to next consumer |
| `spawn_mutation_consumer_with_graph(receiver, config, graph)` | Uses existing Arc\<Graph\> |

### Query Consumers

| Function | Description |
|----------|-------------|
| `spawn_query_consumer(receiver, config, path)` | Creates storage and processes queries |
| `spawn_query_consumer_with_graph(receiver, config, graph)` | Uses existing Arc\<Graph\> |
| `spawn_query_consumer_pool_shared(receiver, graph, n)` | N workers sharing one Graph |
| `spawn_query_consumer_pool_readonly(receiver, config, path, n)` | N workers with own readonly storage |

## Schema

Defined in `schema.rs`. The graph uses 8 column families:

### Hot Column Families (frequently accessed)

| Column Family | Key | Value | Purpose |
|---------------|-----|-------|---------|
| `graph/names` | `(NameHash)` | `String` | Name interning |
| `graph/nodes` | `(Id)` | `(TemporalRange?, NameHash, SummaryHash?)` | Node metadata |
| `graph/forward_edges` | `(SrcId, DstId, NameHash)` | `(TemporalRange?, Weight?, SummaryHash?)` | Edges by source |
| `graph/reverse_edges` | `(DstId, SrcId, NameHash)` | `()` | Reverse edge index (empty value) |

### Cold Column Families (blob separation)

| Column Family | Key | Value | Purpose |
|---------------|-----|-------|---------|
| `graph/node_summaries` | `(SummaryHash)` | `NodeSummary` | Node summary content |
| `graph/edge_summaries` | `(SummaryHash)` | `EdgeSummary` | Edge summary content |

### Fragment Column Families (append-only)

| Column Family | Key | Value | Purpose |
|---------------|-----|-------|---------|
| `graph/node_fragments` | `(Id, TimestampMilli)` | `(TemporalRange?, FragmentContent)` | Node content fragments |
| `graph/edge_fragments` | `(SrcId, DstId, NameHash, TimestampMilli)` | `(TemporalRange?, FragmentContent)` | Edge content fragments |

**Note**: `reverse_edges` has an empty value - it's purely an index for "find edges TO this node". All edge details (TemporalRange, weight, summary) are in `forward_edges`.

**Note**: Name-based lookups (finding nodes/edges by name) are handled by the fulltext search module using Tantivy indexing.

### Key Encoding

Keys use direct byte concatenation (not MessagePack) for efficient prefix scanning:
- Fixed-length fields (Id, TimestampMilli) are serialized as big-endian bytes
- Variable-length fields (names) are placed at the end of keys
- This enables RocksDB's prefix bloom filters and O(1) prefix seek

### Value Encoding

Values use MessagePack serialization with LZ4 compression for space efficiency.

## Scan API

Defined in `scan.rs`. Provides pagination for iterating over column families:

```rust
use motlie_db::scan::{AllNodes, Visitable};

let scanner = AllNodes::new();
let mut visitor = MyVisitor::new();

scanner.visit(&graph.storage(), &mut visitor)?;
```

### Record Types

| Type | Fields |
|------|--------|
| `NodeRecord` | `id, name, summary, valid_range` |
| `EdgeRecord` | `src_id, dst_id, name, weight, summary, valid_range` |
| `NodeFragmentRecord` | `node_id, timestamp, content, valid_range` |
| `EdgeFragmentRecord` | `src_id, dst_id, edge_name, timestamp, content, valid_range` |

## Consumer Chaining

The graph consumer can forward mutations to other consumers (e.g., fulltext):

```rust
use motlie_db::graph::writer::{create_mutation_writer, spawn_mutation_consumer_with_next, WriterConfig};
use motlie_db::fulltext::spawn_mutation_consumer as spawn_fulltext_mutation_consumer;
use tokio::sync::mpsc;

let config = WriterConfig { channel_buffer_size: 1000 };

// Create fulltext consumer (end of chain)
let (fulltext_tx, fulltext_rx) = mpsc::channel(config.channel_buffer_size);
let fulltext_handle = spawn_fulltext_mutation_consumer(fulltext_rx, config.clone(), &index_path);

// Create graph consumer that chains to fulltext
let (writer, graph_rx) = create_mutation_writer(config.clone());
let graph_handle = spawn_mutation_consumer_with_next(
    graph_rx,
    config,
    &db_path,
    fulltext_tx,
);

// Mutations flow: writer -> graph -> fulltext
```

## See Also

- `tests/test_pipeline_integration.rs` - Complete pipeline tests
- `tests/test_secondary_api.rs` - Secondary storage tests
- `src/fulltext/` - Fulltext module with parallel design
- `src/README.md` - Module design patterns overview
- `docs/schema_design.md` - Detailed schema documentation
- `docs/query-api-guide.md` - Query API guide
- `docs/mutation-api-guide.md` - Mutation API guide
