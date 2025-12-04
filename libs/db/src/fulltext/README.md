# Fulltext Module

This module provides full-text search capabilities using [Tantivy](https://github.com/quickwit-oss/tantivy), a Rust-native search engine library. The design closely mirrors the `graph` module to provide a consistent API across both storage backends.

## Design Pattern Comparison: Graph vs Fulltext

Both modules follow the same architectural pattern:

| Component | Graph Module | Fulltext Module |
|-----------|--------------|-----------------|
| **Storage** | `graph::Storage` (RocksDB) | `fulltext::Storage` (Tantivy) |
| **Processor** | `graph::Graph` | `fulltext::Index` |
| **Modes** | ReadOnly, ReadWrite, Secondary | ReadOnly, ReadWrite |
| **QueryExecutor** | `execute(&self, storage: &Storage)` | `execute(&self, storage: &Storage)` |
| **Processor trait** | `fn storage(&self) -> &Storage` | `fn storage(&self) -> &Storage` |

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

### Processor Layer (Graph/Index)

Both modules wrap `Arc<Storage>` in a processor type:

```rust
// Graph
let graph = Graph::new(Arc::new(storage));

// Fulltext (identical pattern)
let index = fulltext::Index::new(Arc::new(storage));
```

**Clone Behavior**: Both `Graph` and `fulltext::Index` clone the `Arc<Storage>`, preserving full read/write capability.

### QueryExecutor Pattern

Both modules implement queries that execute against `Storage`:

```rust
// Graph QueryExecutor
#[async_trait]
impl QueryExecutor for NodeById {
    async fn execute(&self, storage: &graph::Storage) -> Result<Self::Output> { ... }
}

// Fulltext QueryExecutor (same pattern)
#[async_trait]
impl QueryExecutor for Nodes {
    async fn execute(&self, storage: &fulltext::Storage) -> Result<Self::Output> { ... }
}
```

## Spawn Functions

### Graph Module

| Function | Description |
|----------|-------------|
| **Mutation Consumers** | |
| `spawn_graph_consumer(receiver, config, path)` | Single mutation consumer |
| `spawn_graph_consumer_with_next(receiver, config, path, next_tx)` | Mutation consumer that chains to next |
| `spawn_graph_consumer_with_graph(receiver, config, graph)` | Mutation consumer with existing Graph |
| **Query Consumers** | |
| `spawn_graph_query_consumer(receiver, config, path)` | Single query consumer with new Graph |
| `spawn_graph_query_consumer_with_graph(receiver, config, graph)` | Query consumer with existing Graph |
| `spawn_graph_query_consumer_pool_shared(receiver, graph, n)` | Pool sharing one Arc\<Graph\> |
| `spawn_graph_query_consumer_pool_readonly(receiver, config, path, n)` | Pool with individual readonly Graphs |

### Fulltext Module

| Function | Description |
|----------|-------------|
| **Mutation Consumers** | |
| `spawn_fulltext_consumer(receiver, config, path)` | Single mutation consumer |
| `spawn_fulltext_mutation_consumer_with_next(receiver, config, path, next_tx)` | Mutation consumer that chains to next |
| `spawn_fulltext_consumer_with_params(receiver, config, path, params)` | Mutation consumer with index params |
| `spawn_fulltext_consumer_with_params_and_next(receiver, config, path, params, next_tx)` | Mutation consumer with params and chaining |
| **Query Consumers** | |
| `spawn_fulltext_query_consumer(receiver, config, path)` | Single query consumer with new Index |
| `spawn_fulltext_query_consumer_pool_shared(receiver, index, n)` | Pool sharing one Arc\<Index\> |
| `spawn_fulltext_query_consumer_pool_readonly(receiver, config, path, n)` | Pool with individual readonly Indexes |

## Usage Examples

### Basic Setup

```rust
use motlie_db::{
    create_mutation_writer, create_query_reader, create_fulltext_query_reader,
    spawn_graph_query_consumer_pool_shared, spawn_fulltext_query_consumer_pool_shared,
    Graph, Storage, FulltextIndex, FulltextStorage,
    WriterConfig, ReaderConfig, FulltextReaderConfig,
};
use std::sync::Arc;

// Paths
let db_path = Path::new("/data/graph");
let index_path = Path::new("/data/fulltext");

// === Mutation Pipeline: Graph -> Fulltext ===

let writer_config = WriterConfig { channel_buffer_size: 1000 };
let (writer, mutation_receiver) = create_mutation_writer(writer_config.clone());

// Graph mutation consumer (receives mutations first, chains to fulltext)
let (fulltext_tx, fulltext_rx) = tokio::sync::mpsc::channel(1000);
let graph_mutation_handle = spawn_graph_consumer_with_next(
    mutation_receiver,
    writer_config.clone(),
    &db_path,
    fulltext_tx,
);

// Fulltext mutation consumer (receives mutations chained from graph)
let fulltext_mutation_handle = spawn_fulltext_consumer(
    fulltext_rx,
    writer_config,
    &index_path,
);

// Alternative: Fulltext first, then chain to graph
let (graph_tx, graph_rx) = tokio::sync::mpsc::channel(1000);
let fulltext_mutation_handle = spawn_fulltext_mutation_consumer_with_next(
    mutation_receiver,
    writer_config.clone(),
    &index_path,
    graph_tx,
);
let graph_mutation_handle = spawn_graph_consumer(
    graph_rx,
    writer_config,
    &db_path,
);
```

### Multiple Query Consumers

```rust
// === Graph Query Consumers (2 workers) ===

let graph_config = ReaderConfig { channel_buffer_size: 100 };
let (graph_reader, graph_query_receiver) = create_query_reader(graph_config);

// Create shared readonly Graph
let mut storage = Storage::readonly(&db_path);
storage.ready()?;
let graph = Arc::new(Graph::new(Arc::new(storage)));

// Spawn 2 graph query consumers sharing the same Graph
let graph_handles = spawn_graph_query_consumer_pool_shared(
    graph_query_receiver,
    graph,
    2,  // 2 workers
);

// === Fulltext Query Consumers (2 workers) ===

let fulltext_config = FulltextReaderConfig { channel_buffer_size: 100 };
let (fulltext_reader, fulltext_query_receiver) = create_fulltext_query_reader(fulltext_config);

// Create shared readonly Index
let mut ft_storage = FulltextStorage::readonly(&index_path);
ft_storage.ready()?;
let fulltext_index = Arc::new(FulltextIndex::new(Arc::new(ft_storage)));

// Spawn 2 fulltext query consumers sharing the same Index
let fulltext_handles = spawn_fulltext_query_consumer_pool_shared(
    fulltext_query_receiver,
    fulltext_index,
    2,  // 2 workers
);
```

### Concurrent Clients with Mixed Queries

Multiple clients can concurrently issue both graph and fulltext queries:

```rust
use motlie_db::{
    NodeById, NodesByName, OutgoingEdges, FulltextNodes,
    QueryRunnable, FulltextQueryRunnable,
};
use std::time::Duration;

// Client 1: Graph queries
let graph_reader_clone = graph_reader.clone();
let client1 = tokio::spawn(async move {
    // Query node by ID
    let node = NodeById::new(node_id, None)
        .run(&graph_reader_clone, Duration::from_secs(5))
        .await?;

    // Query nodes by name prefix
    let nodes = NodesByName::new("User".to_string(), 10, None)
        .run(&graph_reader_clone, Duration::from_secs(5))
        .await?;

    // Query outgoing edges
    let edges = OutgoingEdges::new(node_id, None, None)
        .run(&graph_reader_clone, Duration::from_secs(5))
        .await?;

    Ok::<_, anyhow::Error>(())
});

// Client 2: Fulltext queries
let fulltext_reader_clone = fulltext_reader.clone();
let client2 = tokio::spawn(async move {
    // Search for nodes containing "rust programming"
    let results = FulltextNodes::new("rust programming".to_string(), 10)
        .run(&fulltext_reader_clone, Duration::from_secs(5))
        .await?;

    for result in results {
        println!("Found: {} (score: {})", result.id, result.score);
    }

    Ok::<_, anyhow::Error>(())
});

// Client 3: Mixed queries (both graph and fulltext)
let graph_reader_clone = graph_reader.clone();
let fulltext_reader_clone = fulltext_reader.clone();
let client3 = tokio::spawn(async move {
    // First, fulltext search to find relevant nodes
    let search_results = FulltextNodes::new("database".to_string(), 5)
        .run(&fulltext_reader_clone, Duration::from_secs(5))
        .await?;

    // Then, for each found node, query its graph relationships
    for result in search_results {
        let edges = OutgoingEdges::new(result.id, None, None)
            .run(&graph_reader_clone, Duration::from_secs(5))
            .await?;

        println!("Node {} has {} outgoing edges", result.id, edges.len());
    }

    Ok::<_, anyhow::Error>(())
});

// Wait for all clients
let (r1, r2, r3) = tokio::join!(client1, client2, client3);
```

## Architecture Diagram

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    Mutation Pipeline                      │
                    │                                                           │
   Client           │   ┌─────────┐      ┌─────────────┐      ┌─────────────┐ │
     │              │   │         │      │   Graph     │      │  Fulltext   │ │
     │ mutations    │   │ Writer  │─────▶│  Consumer   │─────▶│  Consumer   │ │
     └─────────────▶│   │         │      │ (RocksDB)   │      │  (Tantivy)  │ │
                    │   └─────────┘      └─────────────┘      └─────────────┘ │
                    └─────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────────────────────────┐
                    │                     Query Pipeline                        │
                    │                                                           │
                    │   ┌─────────────┐    ┌─────────────────────────────────┐│
   Client 1 ───────▶│   │   Graph     │    │      Graph Query Consumers      ││
   (graph queries)  │   │   Reader    │───▶│  ┌─────────┐    ┌─────────┐    ││
                    │   │             │    │  │Worker 1 │    │Worker 2 │    ││
                    │   └─────────────┘    │  └─────────┘    └─────────┘    ││
                    │                      │         │              │        ││
                    │                      │         ▼              ▼        ││
                    │                      │    ┌─────────────────────┐      ││
                    │                      │    │   Arc<Graph>        │      ││
                    │                      │    │   (shared Storage)  │      ││
                    │                      │    └─────────────────────┘      ││
                    │                      └─────────────────────────────────┘│
                    │                                                         │
                    │   ┌─────────────┐    ┌─────────────────────────────────┐│
   Client 2 ───────▶│   │  Fulltext   │    │    Fulltext Query Consumers     ││
   (fulltext)       │   │   Reader    │───▶│  ┌─────────┐    ┌─────────┐    ││
                    │   │             │    │  │Worker 1 │    │Worker 2 │    ││
                    │   └─────────────┘    │  └─────────┘    └─────────┘    ││
                    │                      │         │              │        ││
                    │                      │         ▼              ▼        ││
                    │                      │    ┌─────────────────────┐      ││
                    │                      │    │  Arc<Index>         │      ││
                    │                      │    │  (shared Storage)   │      ││
                    │                      │    └─────────────────────┘      ││
                    │                      └─────────────────────────────────┘│
                    │                                                         │
   Client 3 ───────▶│   (can use both Graph Reader and Fulltext Reader)       │
   (mixed)          │                                                         │
                    └─────────────────────────────────────────────────────────┘
```

## Key Design Principles

1. **Separation of Storage and Processor**: `Storage` handles the underlying database, while `Graph`/`Index` provides the query/mutation interface.

2. **Readonly vs Readwrite Modes**: Readonly mode allows multiple concurrent readers. Readwrite mode provides exclusive write access.

3. **Arc-based Sharing**: Multiple query consumers share a single `Arc<Graph>` or `Arc<Index>`, minimizing memory usage and ensuring consistency.

4. **Channel-based Communication**: MPMC (multi-producer, multi-consumer) channels enable concurrent query processing.

5. **Self-executing Queries**: Each query type implements `QueryExecutor::execute()`, keeping query logic with the query type rather than in a central processor.

## Fulltext-Specific Features

- **BM25 Scoring**: Results ranked by relevance using BM25 algorithm
- **Faceted Search**: Filter by document type, validity structure, and user-defined tags
- **Fuzzy Search**: Typo-tolerant queries with configurable edit distance
- **Tag Extraction**: Automatic extraction of `#hashtags` from content as facets
- **Temporal Filtering**: Range queries on creation_timestamp and validity fields

## Tantivy Query Syntax

Tantivy uses a query parser similar to Lucene. Here are examples of supported query syntax:

### Basic Term Queries

```
rust                        # Single term - matches documents containing "rust"
rust programming            # Multiple terms (implicit OR) - matches either term
```

### Phrase Queries

```
"rust programming"          # Exact phrase - terms must appear adjacent and in order
"rust programming"~2        # Phrase with slop - terms within 2 positions of each other
```

### Boolean Operators

```
rust AND programming        # Both terms required
rust OR golang              # Either term matches
rust NOT python             # Must contain "rust", must not contain "python"
+rust -python               # Shorthand: + means required, - means excluded
+rust +programming -beginner  # Complex boolean
```

### Field-Specific Queries

```
name:Alice                  # Search only in the "name" field
content:"machine learning"  # Phrase search in "content" field
name:Bob AND content:engineer  # Multi-field query
```

### Wildcard Queries

```
prog*                       # Prefix wildcard - matches "programming", "progress", etc.
pro?ram                     # Single character wildcard - matches "program"
*gramming                   # Suffix wildcard (expensive, use sparingly)
```

### Fuzzy Queries

```
progrmaing~1                # Fuzzy with edit distance 1 - matches "programming"
rust~2                      # Fuzzy with edit distance 2
```

### Range Queries

```
timestamp:[2024-01-01 TO 2024-12-31]   # Inclusive range
weight:{0.5 TO 1.0}                     # Exclusive range
weight:[0.5 TO 1.0}                     # Mixed: inclusive start, exclusive end
timestamp:[* TO 2024-06-01]             # Open-ended range
```

### Boosting

```
rust^2 programming          # Boost "rust" by factor of 2
"machine learning"^1.5      # Boost phrase matches
```

### Grouping

```
(rust OR golang) AND web    # Parentheses for grouping
name:(Alice OR Bob)         # Field grouping
```

### Escaping Special Characters

Special characters that need escaping: `+ - && || ! ( ) { } [ ] ^ " ~ * ? : \ /`

```
\(parentheses\)             # Escaped parentheses
path:\/usr\/local           # Escaped forward slashes
```

### Example Queries for This Index

```rust
// Search nodes by name
"name:Alice"

// Search content for a phrase
"content:\"systems programming\""

// Search for nodes created in a time range
"doc_type:node AND timestamp:[1700000000000 TO 1710000000000]"

// Search edges by name
"doc_type:edge AND edge_name:works_with"

// Fuzzy search for typo tolerance
"content:progrmming~1"

// Combine multiple conditions
"(content:rust OR content:golang) AND doc_type:node_fragment"
```

## Schema and Indexed Fields

The fulltext index uses the following schema:

| Field | Type | Description |
|-------|------|-------------|
| `id_field` | BYTES | Node ID (16 bytes ULID) |
| `src_id_field` | BYTES | Edge source node ID |
| `dst_id_field` | BYTES | Edge destination node ID |
| `node_name_field` | TEXT | Node name (tokenized, searchable) |
| `edge_name_field` | TEXT | Edge name (tokenized, searchable) |
| `content_field` | TEXT | Main content field (summaries, fragments) |
| `doc_type_field` | TEXT | Document type: "nodes", "forward_edges", "node_fragments", "edge_fragments" |
| `creation_timestamp_field` | U64 | Creation timestamp in milliseconds |
| `valid_since_field` | U64 | Validity start timestamp (optional) |
| `valid_until_field` | U64 | Validity end timestamp (optional) |
| `weight_field` | F64 | Edge weight (optional) |

### Facet Fields

| Facet | Path Pattern | Description |
|-------|--------------|-------------|
| `doc_type_facet` | `/type/nodes`, `/type/forward_edges`, etc. | Document type categorization |
| `validity_facet` | `/validity/unbounded`, `/validity/bounded`, etc. | Temporal validity structure |
| `tags_facet` | `/tag/rust`, `/tag/programming`, etc. | User-defined hashtags |

### Note on Temporal Queries

Time-based filtering should use **range queries** on `creation_timestamp`, `valid_since`, and `valid_until` fields at query time, not facets. This allows proper comparison against the query timestamp rather than the index timestamp. See `mod.rs` tests for examples.

## Fragment Indexing Semantics

Node and Edge fragments in motlie-db are **append-only** - each fragment represents a timestamped observation about a node or edge, and multiple fragments form a timeline. This section documents how Tantivy handles these semantics.

### Key Findings

1. **Tantivy is Append-Only by Default**
   - `IndexWriter::add_document()` **always appends** a new document
   - Tantivy has no built-in concept of a "document ID" that would trigger an update/replace
   - Multiple calls to `AddNodeFragment` with the same `node_id` create separate Tantivy documents
   - This matches our append-only fragment semantics perfectly

2. **Multiple Fragments Are Fully Searchable**
   ```rust
   // Adding 5 fragments for the same node
   for ts in [1000, 2000, 3000, 4000, 5000] {
       AddNodeFragment {
           id: same_node_id,
           ts_millis: ts,
           content: format!("Fragment at {}", ts)
       }.run(&writer).await?;
   }
   // All 5 documents are indexed and searchable
   // Search for "Fragment" returns all 5 results
   ```

3. **RocksDB Key Reconstruction**
   - Search results contain stored fields to reconstruct the RocksDB CfKey:
     - **NodeFragments**: `(id_field, creation_timestamp_field)` → RocksDB key `(node_id, timestamp)`
     - **EdgeFragments**: `(src_id_field, dst_id_field, edge_name_field, creation_timestamp_field)` → RocksDB key `(src_id, dst_id, edge_name, timestamp)`
   - The `doc_type_field` indicates which column family to query

4. **Delete Operations Require INDEXED Fields**
   - `IndexWriter::delete_term()` only works on INDEXED fields
   - ID fields (`id_field`, `src_id_field`, `dst_id_field`) are defined with `STORED | FAST | INDEXED`
   - This enables `UpdateNodeValidSinceUntil` and `UpdateEdgeValidSinceUntil` to delete documents

### Fragment vs Node/Edge Deletion Behavior

| Mutation | Behavior |
|----------|----------|
| `AddNodeFragment` | Appends new document (never overwrites) |
| `AddEdgeFragment` | Appends new document (never overwrites) |
| `UpdateNodeValidSinceUntil` | Deletes **all** documents with matching `id_field` |
| `UpdateEdgeValidSinceUntil` | Deletes **all** documents with matching `src_id_field` |

**Important**: `UpdateNodeValidSinceUntil` deletes the node **and** all its fragments from the fulltext index. This is by design - when a node's validity changes, all its indexed content should be removed from search results. The authoritative data remains in RocksDB.

### Consistency with RocksDB

The fulltext index is a **secondary index** - RocksDB is the source of truth:

1. **Fragments are always append-only** in both RocksDB and Tantivy
2. **Deletions are eventually consistent** - Tantivy deletions are applied on commit
3. **Reader visibility** - Tantivy readers must `reload()` to see deletions
4. **Recovery** - If the fulltext index is lost, it can be rebuilt by replaying mutations from RocksDB

### Example: Fragment Timeline

```rust
// Create a node with multiple observations over time
let node_id = Id::new();

// Day 1: Initial observation
AddNodeFragment {
    id: node_id,
    ts_millis: TimestampMilli(day1),
    content: DataUrl::from_text("Initial status: healthy"),
}.run(&writer).await?;

// Day 2: Second observation
AddNodeFragment {
    id: node_id,
    ts_millis: TimestampMilli(day2),
    content: DataUrl::from_text("Status update: minor issue detected"),
}.run(&writer).await?;

// Day 3: Third observation
AddNodeFragment {
    id: node_id,
    ts_millis: TimestampMilli(day3),
    content: DataUrl::from_text("Status resolved: back to healthy"),
}.run(&writer).await?;

// All 3 fragments are searchable:
// - Search "healthy" → finds Day 1 and Day 3 fragments
// - Search "issue" → finds Day 2 fragment
// - Search "status" → finds all 3 fragments

// Each result includes (node_id, timestamp) for RocksDB lookup
```

## Current Query Types

### `Nodes` Query

Searches for nodes and node fragments, returning deduplicated results by node ID:

```rust
use motlie_db::{FulltextNodes, FuzzyLevel, FulltextQueryRunnable};

// Basic search - returns top 10 results
let results = FulltextNodes::new("rust programming".to_string(), 10)
    .run(&reader, Duration::from_secs(5))
    .await?;

for result in results {
    println!("ID: {}, Score: {}", result.id, result.score);
}
```

#### Fuzzy Search

Enable typo-tolerant matching with `FuzzyLevel`:

```rust
// Search with typo tolerance (1 edit distance)
let results = FulltextNodes::new("progrmming".to_string(), 10)
    .with_fuzzy(FuzzyLevel::Low)  // matches "programming"
    .run(&reader, Duration::from_secs(5))
    .await?;

// Higher tolerance (2 edit distance)
let results = FulltextNodes::new("progrmaing".to_string(), 10)
    .with_fuzzy(FuzzyLevel::Medium)  // matches "programming"
    .run(&reader, Duration::from_secs(5))
    .await?;
```

**FuzzyLevel options:**
- `FuzzyLevel::None` - Exact matching only (default)
- `FuzzyLevel::Low` - 1 character edit (insert, delete, substitute, transpose)
- `FuzzyLevel::Medium` - 2 character edits

#### Tag Filtering

Filter results by hashtags extracted from content:

```rust
// Search with tag filter - all specified tags must match
let results = FulltextNodes::new("systems".to_string(), 10)
    .with_tags(vec!["rust".to_string(), "async".to_string()])
    .run(&reader, Duration::from_secs(5))
    .await?;

// Combined: fuzzy search with tag filtering
let results = FulltextNodes::new("concurency".to_string(), 10)
    .with_fuzzy(FuzzyLevel::Low)
    .with_tags(vec!["programming".to_string()])
    .run(&reader, Duration::from_secs(5))
    .await?;
```

Tags are automatically extracted from content during indexing using `#hashtag` syntax.
See the "Tag Extraction" section in schema.rs for supported tag formats.

### `Edges` Query

Searches for edges and edge fragments, returning deduplicated results by edge key (src_id, dst_id, edge_name):

```rust
use motlie_db::{FulltextEdges, FuzzyLevel, FulltextQueryRunnable};

// Basic search
let results = FulltextEdges::new("collaborates".to_string(), 10)
    .run(&reader, Duration::from_secs(5))
    .await?;

for result in results {
    println!("Edge: {} -> {} ({}), Score: {}",
        result.src_id, result.dst_id, result.edge_name, result.score);
}

// With fuzzy matching
let results = FulltextEdges::new("colaborates".to_string(), 10)
    .with_fuzzy(FuzzyLevel::Low)
    .run(&reader, Duration::from_secs(5))
    .await?;

// With tag filtering
let results = FulltextEdges::new("partnership".to_string(), 10)
    .with_tags(vec!["important".to_string()])
    .run(&reader, Duration::from_secs(5))
    .await?;
```

### Query Result Types

#### `NodeHit`

```rust
pub struct NodeHit {
    /// BM25 relevance score
    pub score: f32,
    /// Node ID (use to look up full node in RocksDB)
    pub id: Id,
    /// Fragment timestamp (None = node match, Some = fragment match)
    pub fragment_timestamp: Option<u64>,
    /// Optional text snippet (for future highlighting support)
    pub snippet: Option<String>,
}
```

#### `EdgeHit`

```rust
pub struct EdgeHit {
    /// BM25 relevance score
    pub score: f32,
    /// Source node ID
    pub src_id: Id,
    /// Destination node ID
    pub dst_id: Id,
    /// Edge name
    pub edge_name: String,
    /// Fragment timestamp (None = edge match, Some = fragment match)
    pub fragment_timestamp: Option<u64>,
    /// Optional text snippet (for future highlighting support)
    pub snippet: Option<String>,
}
```

The `fragment_timestamp` field distinguishes between entity and fragment matches:
- `None` - The match came from the node/edge itself
- `Some(ts)` - The match came from a fragment; use with ID to look up in RocksDB

## TODO: Planned Query Types

The following query types are planned but not yet implemented:

### `FacetCounts` Query
Get counts of documents by facet values:

```rust
// TODO: Not yet implemented
// Count documents by tag
let tag_counts = FacetCounts::by_tag()
    .run(&reader, Duration::from_secs(5))
    .await?;
// Returns: [("rust", 42), ("programming", 38), ("database", 25), ...]

// Count documents by type
let type_counts = FacetCounts::by_doc_type()
    .run(&reader, Duration::from_secs(5))
    .await?;
// Returns: [("nodes", 500), ("forward_edges", 1200), ("node_fragments", 300), ...]

// Count by validity structure
let validity_counts = FacetCounts::by_validity()
    .run(&reader, Duration::from_secs(5))
    .await?;
// Returns: [("unbounded", 400), ("bounded", 100), ("since_only", 50), ...]
```

### `Aggregations` Query
Aggregate statistics over search results:

```rust
// TODO: Not yet implemented
let stats = Aggregations::new("programming".to_string())
    .count_by_tag()
    .count_by_validity()
    .run(&reader, Duration::from_secs(5))
    .await?;

println!("Total matches: {}", stats.total_count);
println!("Tag distribution: {:?}", stats.tag_counts);
println!("Validity distribution: {:?}", stats.validity_counts);
```

### `MoreLikeThis` Query
Find similar documents based on an existing document:

```rust
// TODO: Not yet implemented
let similar = MoreLikeThis::new(node_id)
    .min_term_freq(1)
    .max_query_terms(25)
    .run(&reader, Duration::from_secs(5))
    .await?;
```

## Implementation Checklist

### Completed
- [x] Basic `Nodes` query with BM25 ranking
- [x] Basic `Edges` query with BM25 ranking
- [x] Node and node fragment deduplication
- [x] Edge and edge fragment deduplication
- [x] Facet indexing (doc_type, validity, tags)
- [x] Fuzzy search support via `FuzzyLevel` (integrated into queries)
- [x] Tag filtering via `with_tags()` builder method
- [x] Multiple query consumers with shared Index
- [x] Readonly/Readwrite storage modes
- [x] Temporal validity fields (valid_since, valid_until)
- [x] Time range queries on creation_timestamp and validity fields

### Planned
- [ ] `FacetCounts` query for facet statistics
- [ ] `Aggregations` query for search analytics
- [ ] `MoreLikeThis` query for similarity search
- [ ] Highlight support (return matching snippets via `snippet` field)
- [ ] Pagination with search_after for deep pagination
- [ ] Custom scoring/boosting per field

## File Organization

```
fulltext/
├── mod.rs       # Storage, Index struct, module exports
├── schema.rs    # Tantivy schema definition and field handles
├── mutation.rs  # MutationExecutor impls for index updates
├── writer.rs    # Writer/MutationConsumer infrastructure
├── query.rs     # Query types (Nodes, Edges), FuzzyLevel enum
├── reader.rs    # Reader/QueryConsumer infrastructure
├── search.rs    # Search result types (NodeHit, EdgeHit, FacetCounts)
└── README.md    # This file
```

## Integration Tests

The `tests/test_fulltext_search_pipeline.rs` file contains comprehensive integration tests that verify the fulltext search functionality with the graph storage backend. Each test demonstrates specific query capabilities:

### Test Coverage

| Test Name | Features Demonstrated |
|-----------|----------------------|
| `test_fulltext_node_search_resolves_to_rocksdb` | Basic `FulltextNodes` query, BM25 scoring, result deduplication by node ID, verification via `NodeById` graph query |
| `test_fulltext_edge_search_resolves_to_rocksdb` | Basic `FulltextEdges` query, BM25 scoring, result deduplication by edge key (src, dst, name), verification via `EdgeSummaryBySrcDstName` graph query |
| `test_fulltext_combined_search_with_verification` | Mixed node/edge searches, cross-verification between fulltext hits and RocksDB entries, fragment indexing |
| `test_fulltext_fragment_deduplication_behavior` | Multiple fragments per entity, deduplication semantics, best score retention |
| `test_fulltext_search_limit_and_ordering` | Result limit enforcement, BM25 score ordering, pagination behavior |
| `test_fulltext_tag_facet_filtering` | `#hashtag` extraction from content, `with_tags()` filter, AND semantics for multiple tags, works for both nodes and edges |
| `test_fulltext_fuzzy_search` | `FuzzyLevel::Low` (1 edit distance), `FuzzyLevel::Medium` (2 edits), typo tolerance for nodes and edges |

### Running the Tests

```bash
# Run all fulltext search pipeline tests
cargo test --test test_fulltext_search_pipeline

# Run a specific test with output
cargo test --test test_fulltext_search_pipeline test_fulltext_tag_facet_filtering -- --nocapture

# Run tests matching a pattern
cargo test --test test_fulltext_search_pipeline fuzzy -- --nocapture
```

### Test Architecture

Each test follows this pattern:

1. **Setup mutation pipeline** - Graph and fulltext consumers chained together
2. **Add test data** - Nodes, edges, and fragments with searchable content
3. **Run fulltext queries** - Using `FulltextNodes` or `FulltextEdges`
4. **Verify against RocksDB** - Each hit is resolved via graph queries (`NodeById`, `EdgeSummaryBySrcDstName`)
5. **Cleanup** - Drop readers and await consumer handles

Example from `test_fulltext_tag_facet_filtering`:

```rust
// Create nodes with hashtags in content
AddNode {
    id: rust_id,
    ts_millis: TimestampMilli::now(),
    name: "Rust".to_string(),
    summary: NodeSummary::from_text(
        "Rust is a #systems programming language with #async support"
    ),
    valid_range: None,
}
.run(&writer)
.await?;

// Query with tag filter
let results = FulltextNodes::new("language".to_string(), 10)
    .with_tags(vec!["systems".to_string()])  // Filter by #systems tag
    .run(&fulltext_reader, timeout)
    .await?;

// Verify results resolve to RocksDB
for hit in results {
    let node = NodeById::new(hit.id, None)
        .run(&graph_reader, timeout)
        .await?;
    assert!(node.is_some());
}
```

## See Also

- `tests/test_fulltext_search_pipeline.rs` - **Main integration tests** demonstrating all query features
- `tests/test_pipeline_integration.rs` - Complete integration tests demonstrating all patterns
- `tests/test_fulltext_integration.rs` - Fulltext-specific integration tests
- `tests/test_fulltext_chaining.rs` - Graph-to-fulltext chaining tests
- `src/graph/` - Graph module with parallel design
- `src/README.md` - Module design patterns overview
- [Tantivy Query Parser Documentation](https://docs.rs/tantivy/latest/tantivy/query/struct.QueryParser.html)
