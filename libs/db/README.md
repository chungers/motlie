# motlie-db

A graph database library built on RocksDB with async query/mutation processing and full-text search capabilities.

## Overview

`motlie-db` provides a high-performance, async graph database with the following key features:

- **Graph Storage**: Nodes and edges with bidirectional navigation
- **Fragment Management**: Time-series content fragments attached to entities
- **Full-Text Search**: Tantivy-based search index for content
- **Async Processing**: Query and mutation processing with timeout support
- **Optimized Key Encoding**: Direct byte concatenation for RocksDB keys (values use MessagePack)
- **Prefix Scan Optimization**: O(1) prefix seek with RocksDB prefix extractors and bloom filters
- **Type-Safe IDs**: ULID-based 128-bit identifiers with lexicographic ordering

## Architecture

### Core Components

```
┌──────────────────────────────────────────────────────────────┐
│                        Application                            │
└──────────────────┬───────────────────────┬───────────────────┘
                   │                       │
         ┌─────────▼─────────┐   ┌─────────▼──────────┐
         │     Reader API     │   │    Writer API      │
         │  (Query System)    │   │ (Mutation System)  │
         └─────────┬─────────┘   └─────────┬──────────┘
                   │                       │
         ┌─────────▼─────────┐   ┌─────────▼──────────┐
         │  Query Consumer   │   │ Mutation Consumer  │
         │   (async tasks)   │   │   (async tasks)    │
         └─────────┬─────────┘   └─────────┬──────────┘
                   │                       │
         ┌─────────▼────────────────────────▼──────────┐
         │        Graph Processor + Storage            │
         │  ┌──────────────────────────────────────┐   │
         │  │          RocksDB Storage             │   │
         │  │  ┌────────┬─────────┬─────────┐     │   │
         │  │  │ Nodes  │  Edges  │Fragments│ ... │   │
         │  │  └────────┴─────────┴─────────┘     │   │
         │  └──────────────────────────────────────┘   │
         └───────────────────┬─────────────────────────┘
                             │
                   ┌─────────▼──────────┐
                   │  FullText Processor│
                   │  (Tantivy index)   │
                   └────────────────────┘
```

### Reader API (Query System)

The `Reader` provides async query operations with a modern, type-driven API:

```rust
use motlie_db::{Reader, NodeByIdQuery, EdgeByIdQuery, Runnable, Id};
use std::time::Duration;

let timeout = Duration::from_secs(5);

// Query node by ID
let (name, summary) = NodeByIdQuery::new(node_id, None)
    .run(&reader, timeout)
    .await?;

// Query edge by ID (returns topology + summary)
let (src_id, dst_id, edge_name, summary) = EdgeByIdQuery::new(edge_id, None)
    .run(&reader, timeout)
    .await?;

// Query edges from a node (outgoing)
let edges = OutgoingEdgesQuery::new(node_id, None)
    .run(&reader, timeout)
    .await?;

// Query edges to a node (incoming)
let edges = IncomingEdgesQuery::new(node_id, None)
    .run(&reader, timeout)
    .await?;

// Query fragments by ID and time range
use std::ops::Bound;
let fragments = FragmentsByIdTimeRangeQuery::new(
    entity_id,
    (Bound::Unbounded, Bound::Unbounded),
    None
)
.run(&reader, timeout)
.await?;

// Concurrent queries using tokio::try_join!
use tokio::try_join;

let (node_info, outgoing, incoming) = try_join!(
    NodeByIdQuery::new(node_id, None).run(&reader, timeout),
    OutgoingEdgesQuery::new(node_id, None).run(&reader, timeout),
    IncomingEdgesQuery::new(node_id, None).run(&reader, timeout)
)?;
```

**Key Features**:
- **Type-driven API** - Query type determines result type
- **Timeout at execution** - Timeout is a runtime parameter, not construction
- **No builder boilerplate** - Direct query construction
- **Composable queries** - Queries are values that can be stored and reused
- **Temporal queries** - Optional reference timestamp for historical queries
- **Concurrent execution** - Use tokio::join! for parallel queries
- Returns complete data (no need for joins)

See: [`docs/query-api-guide.md`](docs/query-api-guide.md) ⭐ for comprehensive API documentation

### Writer API (Mutation System)

The `Writer` provides async mutation operations with automatic transaction batching:

```rust
use motlie_db::{Writer, WriterConfig, AddNode, AddEdge, AddFragment, TimestampMilli, Id};

// Create writer
let (writer, receiver) = create_mutation_writer(WriterConfig::default());

// Add node
writer.add_node(AddNode {
    id: Id::new(),
    name: "Alice".to_string(),
    ts_millis: TimestampMilli::now(),
}).await?;

// Add edge
writer.add_edge(AddEdge {
    id: Id::new(),
    source_node_id: alice_id,
    target_node_id: bob_id,
    name: "follows".to_string(),
    ts_millis: TimestampMilli::now(),
}).await?;

// Add fragment
writer.add_fragment(AddFragment {
    id: alice_id,
    content: "Fragment content".to_string(),
    ts_millis: TimestampMilli::now().0,
}).await?;
```

**Key Features**:
- Non-blocking async writes
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

Queries use an async consumer-processor pattern:

```
Reader → Query → MPMC Channel → Consumer → Processor → RocksDB
   ↓                                             ↓
   └─────── oneshot channel ←──── Result ────────┘
```

### Query Types

All queries implement the `QueryWithResult` trait:

- `NodeByIdQuery`: Lookup node by ID
- `EdgeByIdQuery`: Lookup edge with full topology by ID
- `EdgeSummaryBySrcDstNameQuery`: Lookup edge by (src, dst, name)
- `FragmentContentByIdQuery`: Retrieve all fragments for an entity
- `EdgesFromNodeByIdQuery`: Get all outgoing edges
- `EdgesToNodeByIdQuery`: Get all incoming edges

### Query Processor Trait

The query `Processor` trait defines the contract for query execution:

```rust
#[async_trait]
pub trait Processor: Send + Sync {
    async fn get_node_by_id(&self, query: &NodeByIdQuery)
        -> Result<(NodeName, NodeSummary)>;

    async fn get_edge_by_id(&self, query: &EdgeByIdQuery)
        -> Result<(SrcId, DstId, EdgeName, EdgeSummary)>;

    async fn get_outgoing_edges_by_id(&self, query: &EdgesFromNodeByIdQuery)
        -> Result<Vec<(SrcId, EdgeName, DstId)>>;

    async fn get_incoming_edges_by_id(&self, query: &EdgesToNodeByIdQuery)
        -> Result<Vec<(DstId, EdgeName, SrcId)>>;

    // ... more methods
}
```

**Implementations**:
- `Graph`: Executes queries against RocksDB
- Test processors for mocking

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
- `Graph`: Processes all mutations in a single RocksDB transaction
- `Backend`: Indexes all mutations in batch
- Test processors for mocking

### Consumer Pattern

The `spawn_graph_query_consumer()` function creates an async task that:

1. Receives queries from MPMC channel (flume)
2. Executes query via `Processor`
3. Handles timeouts
4. Sends result via oneshot channel
5. Continues processing until channel closed

**Benefits**:
- Concurrent query execution
- Backpressure via bounded channels
- Clean shutdown semantics
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
    AddFragment(AddFragment),
    Invalidate(InvalidateArgs),
}
```

### Consumer Chaining

Consumers can forward mutations to the next consumer in the chain:

```rust
// Create FullText consumer (end of chain)
let (fulltext_consumer, _) = spawn_fulltext_consumer(
    fulltext_processor,
    receiver_fulltext,
    None,  // No next consumer
);

// Create Graph consumer (forwards to FullText)
let (graph_consumer, _) = spawn_graph_consumer(
    graph_processor,
    receiver_graph,
    Some(writer_fulltext),  // Forward to FullText
);
```

**Processing Order**: Graph → FullText
- Graph updates RocksDB first
- Then forwards to FullText for indexing
- Ensures consistency (index only after successful storage)

### GraphProcessor

Implements `Processor` trait with batched mutation processing:

```rust
#[async_trait]
impl Processor for Graph {
    async fn process_mutations(&self, mutations: &[Mutation]) -> Result<()> {
        if mutations.is_empty() {
            return Ok(());
        }

        log::info!("[Graph] About to insert {} mutations", mutations.len());

        // Each mutation generates its own storage operations
        let mut operations = Vec::new();
        for mutation in mutations {
            operations.extend(mutation.plan()?);
        }

        // Execute all operations in a SINGLE transaction
        let txn_db = self.storage.transaction_db()?;
        let txn = txn_db.transaction();

        for op in operations {
            match op {
                StorageOperation::PutCf(PutCf(cf_name, (key, value))) => {
                    let cf = txn_db.cf_handle(cf_name)
                        .ok_or_else(|| anyhow!("Column family '{}' not found", cf_name))?;
                    txn.put_cf(cf, key, value)?;
                }
            }
        }

        // Single commit for all mutations
        txn.commit()?;

        log::info!("[Graph] Successfully committed {} mutations", mutations.len());
        Ok(())
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

## Storage Modes

### ReadWrite Mode
```rust
let mut storage = Storage::new("/path/to/db");
storage.ready().await?;  // Create/open database

// Write operations allowed
storage.execute_batch(ops).await?;
```

### ReadOnly Mode
```rust
let storage = Storage::new_readonly("/path/to/db")?;

// Only read operations
let value = storage.db()?.get_cf(cf, key)?;

// Multiple readonly instances can coexist
let reader1 = Storage::new_readonly("/path/to/db")?;
let reader2 = Storage::new_readonly("/path/to/db")?;
```

**Concurrency**:
- 1 ReadWrite + N ReadOnly instances can coexist
- ReadOnly instances are independent
- Closing ReadWrite doesn't affect ReadOnly

## Full-Text Search

### Backend

Maintains a Tantivy search index synchronized with graph mutations:

```rust
pub struct Backend {
    index: Index,
    schema: Schema,
    // Fields: id, entity_type, content, timestamp
}
```

**Indexed Fields**:
- `id`: Entity ID (STRING, stored)
- `entity_type`: "node" | "edge" | "fragment" (STRING, indexed)
- `content`: Searchable text (TEXT, indexed + stored)
- `timestamp`: Creation time (U64, indexed + stored)

**Mutation Handling**:
- `AddNode`: Index node summary
- `AddEdge`: Index edge summary
- `AddFragment`: Index fragment content

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
| `graph/query.rs` | `Executing NodeFragmentsByIdTimeRange query` | `id`, `time_range` | Query node fragments |
| `graph/query.rs` | `Executing EdgeFragmentsByIdTimeRange query` | `src_id`, `dst_id`, `edge_name`, `time_range` | Query edge fragments |
| `graph/query.rs` | `Executing EdgeSummaryBySrcDstName query` | `src_id`, `dst_id`, `name` | Lookup edge by topology |
| `graph/query.rs` | `Executing OutgoingEdges query` | `node_id` | Get edges from node |
| `graph/query.rs` | `Executing IncomingEdges query` | `node_id` | Get edges to node |
| `graph/query.rs` | `Executing NodesByName query` | `name` | Search nodes by name |
| `graph/query.rs` | `Executing EdgesByName query` | `name` | Search edges by name |

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
| `graph/scan.rs` | `Executing AllNodeNames scan` | `limit`, `reverse`, `has_cursor` | Scan node name index |
| `graph/scan.rs` | `Executing AllEdgeNames scan` | `limit`, `reverse`, `has_cursor` | Scan edge name index |

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
