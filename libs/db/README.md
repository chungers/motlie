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

The `Reader` provides async query operations with timeout support:

```rust
use motlie_db::{Reader, ReaderConfig, Id};
use std::time::Duration;

// Create reader
let (reader, receiver) = create_query_reader(ReaderConfig::default());

// Query node by ID
let (name, summary) = reader
    .node_by_id(node_id, Duration::from_secs(5))
    .await?;

// Query edge by ID (returns topology + summary)
let (src_id, dst_id, edge_name, summary) = reader
    .edge_by_id(edge_id, Duration::from_secs(5))
    .await?;

// Query edges from a node (outgoing)
let edges = reader
    .edges_from_node_by_id(node_id, Duration::from_secs(5))
    .await?;

// Query edges to a node (incoming)
let edges = reader
    .edges_to_node_by_id(node_id, Duration::from_secs(5))
    .await?;

// Query fragments by ID
let fragments = reader
    .fragments_by_id(entity_id, Duration::from_secs(5))
    .await?;
```

**Key Features**:
- Point queries by ID
- Edge topology queries (neighbors)
- Fragment retrieval
- Timeout support for all operations
- Returns complete data (no need for joins)

See: [`docs/reader.md`](docs/reader.md) for detailed API documentation

### Writer API (Mutation System)

The `Writer` provides async mutation operations:

```rust
use motlie_db::{Writer, WriterConfig, AddNode, AddEdge, AddFragment};

// Create writer
let (writer, receiver) = create_writer(WriterConfig::default());

// Add node
let node = AddNode {
    id: Id::new(),
    name: "Alice".to_string(),
    ts_millis: TimestampMilli::now(),
};
writer.write(Mutation::AddNode(node)).await?;

// Add edge
let edge = AddEdge {
    id: Id::new(),
    source_node_id: alice_id,
    target_node_id: bob_id,
    name: "follows".to_string(),
    ts_millis: TimestampMilli::now(),
};
writer.write(Mutation::AddEdge(edge)).await?;

// Add fragment
let fragment = AddFragment {
    node_or_edge_id: alice_id,
    content: "Fragment content".to_string(),
    ts_millis: TimestampMilli::now(),
};
writer.write(Mutation::AddFragment(fragment)).await?;
```

**Key Features**:
- Non-blocking async writes
- MPMC channel-based
- Consumer chaining (Graph → FullText)
- Automatic index updates

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

### Processor Trait

The `Processor` trait defines the contract for query execution:

```rust
#[async_trait]
pub trait Processor: Send + Sync {
    async fn get_node_by_id(&self, query: &NodeByIdQuery)
        -> Result<(NodeName, NodeSummary)>;

    async fn get_edge_by_id(&self, query: &EdgeByIdQuery)
        -> Result<(SrcId, DstId, EdgeName, EdgeSummary)>;

    async fn get_edges_from_node_by_id(&self, query: &EdgesFromNodeByIdQuery)
        -> Result<Vec<(SrcId, EdgeName, DstId)>>;

    async fn get_edges_to_node_by_id(&self, query: &EdgesToNodeByIdQuery)
        -> Result<Vec<(DstId, EdgeName, SrcId)>>;

    // ... more methods
}
```

**Implementations**:
- `GraphProcessor`: Executes queries against RocksDB
- Test processors for mocking

### Consumer Pattern

The `spawn_query_consumer()` function creates an async task that:

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

Mutations use a chainable consumer pattern:

```
Writer → Mutation → MPMC Channel → Consumer₁ → Consumer₂ → ... → Consumerₙ
                                       ↓            ↓               ↓
                                    Storage    FullText          Custom
```

### Mutation Types

```rust
pub enum Mutation {
    AddNode(AddNode),
    AddEdge(AddEdge),
    AddFragment(AddFragment),
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

Implements `MutationProcessor` trait:

```rust
#[async_trait]
impl MutationProcessor for GraphProcessor {
    async fn process(&self, mutation: &Mutation) -> Result<()> {
        match mutation {
            Mutation::AddNode(n) => {
                // 1. Create operations
                let ops = Plan::create_node(n)?;

                // 2. Execute atomic batch
                self.storage.execute_batch(ops).await?;
            }
            Mutation::AddEdge(e) => {
                // Creates 3 operations:
                // - Put in Edges CF (topology + summary)
                // - Put in ForwardEdges CF (src → dst index)
                // - Put in ReverseEdges CF (dst → src index)
                let ops = Plan::create_edge(e)?;
                self.storage.execute_batch(ops).await?;
            }
            // ...
        }
    }
}
```

**Guarantees**:
- Atomic writes (RocksDB batch operations)
- Multi-index consistency (Edges + ForwardEdges + ReverseEdges)

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

### FullTextProcessor

Maintains a Tantivy search index synchronized with graph mutations:

```rust
pub struct FullTextProcessor {
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

- ✅ 123 passing tests
- Storage lifecycle (open, close, reopen)
- ReadOnly/ReadWrite concurrency
- Query processing with timeouts
- Mutation processing and chaining
- Schema serialization and ordering
- Full-text indexing
- Graph topology queries

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

- **API Design**: Reader API analysis and design decisions
- **Schema Design**: RocksDB schema and MessagePack serialization
- **Performance**: Prefix scanning and key ordering optimization

See [`docs/README.md`](docs/README.md) for complete documentation index.

## Performance Characteristics

### Query Performance

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| `node_by_id()` | O(1) | Direct RocksDB get |
| `edge_by_id()` | O(1) | Direct RocksDB get, returns full topology |
| `edges_from_node_by_id()` | O(N) | N = edges from node, uses prefix scan |
| `edges_to_node_by_id()` | O(N) | N = edges to node, uses prefix scan |
| `fragments_by_id()` | O(F) | F = fragments for entity, prefix scan |

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

## Dependencies

- **rocksdb** (0.24): Embedded key-value store
- **tokio**: Async runtime
- **flume** (0.11): MPMC channels for mutation pipeline
- **ferroid** (0.8): ULID implementation
- **rmp-serde** (1.3): MessagePack serialization
- **tantivy**: Full-text search engine
- **data-url** (0.3): DataUrl encoding/decoding
- **async-trait**: Async trait support

## License

See workspace LICENSE file.
