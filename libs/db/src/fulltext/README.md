# Fulltext Module

This module provides full-text search capabilities using [Tantivy](https://github.com/quickwit-oss/tantivy), a Rust-native search engine library. The design closely mirrors the `graph` module to provide a consistent API across both storage backends.

## Design Pattern Comparison: Graph vs Fulltext

Both modules follow the same architectural pattern:

| Component | Graph Module | Fulltext Module |
|-----------|--------------|-----------------|
| **Storage** | `graph::Storage` (RocksDB) | `fulltext::Storage` (Tantivy) |
| **Processor** | `graph::Processor` | `fulltext::Index` |
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

### Processor Layer (Processor/Index)

Both modules wrap `Arc<Storage>` in a processor type:

```rust
// Graph
let processor = graph::Processor::new(Arc::new(storage));

// Fulltext (identical pattern)
let index = fulltext::Index::new(Arc::new(storage));
```

**Clone Behavior**: Both `graph::Processor` and `fulltext::Index` clone the `Arc<Storage>`, preserving full read/write capability.

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
| `spawn_mutation_consumer(receiver, config, path)` | Single mutation consumer |
| `spawn_mutation_consumer_with_next(receiver, config, path, next_tx)` | Mutation consumer that chains to next |
| `spawn_mutation_consumer_with_receiver(receiver, config, processor)` | Mutation consumer with existing Processor |
| **Query Consumers** | |
| `spawn_query_consumer(receiver, config, path)` | Single query consumer with new Processor |
| `spawn_query_consumer_with_processor(receiver, config, processor)` | Query consumer with existing Processor |
| `spawn_query_consumer_pool_shared(receiver, processor, n)` | Pool sharing one Arc\<Processor\> |
| `spawn_query_consumer_pool_readonly(receiver, config, path, n)` | Pool with individual readonly Processors |

### Fulltext Module

| Function | Description |
|----------|-------------|
| **Mutation Consumers** | |
| `spawn_mutation_consumer(receiver, config, path)` | Single mutation consumer |
| `spawn_mutation_consumer_with_next(receiver, config, path, next_tx)` | Mutation consumer that chains to next |
| `spawn_mutation_consumer_with_params(receiver, config, path, params)` | Mutation consumer with index params |
| `spawn_mutation_consumer_with_params_and_next(receiver, config, path, params, next_tx)` | Mutation consumer with params and chaining |
| **Query Consumers** | |
| `spawn_query_consumer(receiver, config, path)` | Single query consumer with new Index |
| `spawn_query_consumer_pool_shared(receiver, index, n)` | Pool sharing one Arc\<Index\> |
| `spawn_query_consumer_pool_readonly(receiver, config, path, n)` | Pool with individual readonly Indexes |

## Usage Examples

### Basic Setup

```rust
use motlie_db::graph::{
    create_query_reader, spawn_query_consumer_pool_shared,
    Processor, Storage, WriterConfig, ReaderConfig,
};
use motlie_db::fulltext::{
    create_query_reader as create_fulltext_query_reader,
    spawn_query_consumer_pool_shared as spawn_fulltext_query_consumer_pool_shared,
    Index as FulltextIndex, Storage as FulltextStorage, ReaderConfig as FulltextReaderConfig,
};
use motlie_db::graph::writer::create_mutation_writer;
use std::sync::Arc;

// Paths
let db_path = Path::new("/data/graph");
let index_path = Path::new("/data/fulltext");

// === Mutation Pipeline: Graph -> Fulltext ===

let writer_config = WriterConfig { channel_buffer_size: 1000 };
let (writer, mutation_receiver) = create_mutation_writer(writer_config.clone());

// Graph mutation consumer (receives mutations first, chains to fulltext)
let (fulltext_tx, fulltext_rx) = tokio::sync::mpsc::channel(1000);
let graph_mutation_handle = spawn_mutation_consumer_with_next(
    mutation_receiver,
    writer_config.clone(),
    &db_path,
    fulltext_tx,
);

// Fulltext mutation consumer (receives mutations chained from graph)
let fulltext_mutation_handle = motlie_db::fulltext::spawn_mutation_consumer(
    fulltext_rx,
    writer_config,
    &index_path,
);

// Alternative: Fulltext first, then chain to graph
let (graph_tx, graph_rx) = tokio::sync::mpsc::channel(1000);
let fulltext_mutation_handle = motlie_db::fulltext::spawn_mutation_consumer_with_next(
    mutation_receiver,
    writer_config.clone(),
    &index_path,
    graph_tx,
);
let graph_mutation_handle = spawn_mutation_consumer(
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

// Create shared readonly Processor
let mut storage = Storage::readonly(&db_path);
storage.ready()?;
let processor = Arc::new(graph::Processor::new(Arc::new(storage)));

// Spawn 2 graph query consumers sharing the same Processor
let graph_handles = spawn_query_consumer_pool_shared(
    graph_query_receiver,
    processor,
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
    fulltext_index.clone(),
    2,  // 2 workers
);
```

### Concurrent Clients with Mixed Queries

Multiple clients can concurrently issue both graph and fulltext queries:

```rust
use motlie_db::graph::query::{NodeById, OutgoingEdges};
use motlie_db::fulltext::Nodes as FulltextNodes;
use motlie_db::reader::Runnable as QueryRunnable;  // Unified trait for all queries
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
                    │                      │    │  Arc<Processor>     │      ││
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

1. **Separation of Storage and Processor**: `Storage` handles the underlying database, while `Processor`/`Index` provides the query/mutation interface.

2. **Readonly vs Readwrite Modes**: Readonly mode allows multiple concurrent readers. Readwrite mode provides exclusive write access.

3. **Arc-based Sharing**: Multiple query consumers share a single `Arc<Processor>` or `Arc<Index>`, minimizing memory usage and ensuring consistency.

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
| `doc_type_field` | TEXT | Document type: "nodes", "edges", "node_fragments", "edge_fragments" |
| `creation_timestamp_field` | U64 | Creation timestamp in milliseconds |
| `valid_since_field` | U64 | Validity start timestamp (optional) |
| `valid_until_field` | U64 | Validity end timestamp (optional) |
| `weight_field` | F64 | Edge weight (optional) |

### Facet Fields

| Facet | Path Pattern | Description |
|-------|--------------|-------------|
| `doc_type_facet` | `/type/nodes`, `/type/edges`, etc. | Document type categorization |
| `validity_facet` | `/validity/unbounded`, `/validity/bounded`, etc. | Active period structure |
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
   - This enables `UpdateNodeActivePeriod` and `UpdateEdgeActivePeriod` to delete documents

### Fragment vs Node/Edge Deletion Behavior

| Mutation | Behavior |
|----------|----------|
| `AddNodeFragment` | Appends new document (never overwrites) |
| `AddEdgeFragment` | Appends new document (never overwrites) |
| `UpdateNodeActivePeriod` | Deletes **all** documents with matching `id_field` |
| `UpdateEdgeActivePeriod` | Deletes **all** documents with matching `src_id_field` |

**Important**: `UpdateNodeActivePeriod` deletes the node **and** all its fragments from the fulltext index. This is by design - when a node's validity changes, all its indexed content should be removed from search results. The authoritative data remains in RocksDB.

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
use motlie_db::fulltext::{Nodes as FulltextNodes, FuzzyLevel};
use motlie_db::reader::Runnable;  // Unified query trait

// Basic search - returns top 10 results
let results = FulltextNodes::new("rust programming".to_string(), 10)
    .run(&reader, Duration::from_secs(5))
    .await?;

for result in results {
    println!("ID: {}, Score: {}", result.id, result.score);
}
```

#### Wildcard Search

Wildcard patterns (`*` and `?`) are automatically detected and converted to regex queries:

```rust
// Prefix search - finds "likes", "likely", "liked"
let results = FulltextNodes::new("lik*".to_string(), 10)
    .run(&reader, Duration::from_secs(5))
    .await?;

// Suffix search - finds "backend", "frontend"
let results = FulltextNodes::new("*end".to_string(), 10)
    .run(&reader, Duration::from_secs(5))
    .await?;

// Single character wildcard - finds "likes", "liked"
let results = FulltextNodes::new("like?".to_string(), 10)
    .run(&reader, Duration::from_secs(5))
    .await?;

// Contains search - finds anything with "base" in it
let results = FulltextNodes::new("*base*".to_string(), 10)
    .run(&reader, Duration::from_secs(5))
    .await?;
```

**Wildcard patterns:**
- `*` - Matches zero or more characters
- `?` - Matches exactly one character

**Note:** When wildcards are detected, the query bypasses the standard QueryParser and uses Tantivy's `RegexQuery` for pattern matching.

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
// Search with tag filter - matches documents with ANY of the specified tags (OR semantics)
let results = FulltextNodes::new("systems".to_string(), 10)
    .with_tags(vec!["rust".to_string(), "golang".to_string()])
    .run(&reader, Duration::from_secs(5))
    .await?;
// Returns documents tagged with #rust OR #golang

// Combined: fuzzy search with tag filtering
let results = FulltextNodes::new("concurency".to_string(), 10)
    .with_fuzzy(FuzzyLevel::Low)
    .with_tags(vec!["programming".to_string()])
    .run(&reader, Duration::from_secs(5))
    .await?;
```

**Tag Filter Semantics (OR)**:
- When multiple tags are specified, documents matching **ANY** of the tags are returned
- Example: `.with_tags(vec!["rust", "python"])` returns documents with #rust OR #python
- This enables "find documents related to any of these topics" queries
- For AND semantics (all tags required), issue separate queries and intersect results

Tags are automatically extracted from content during indexing using `#hashtag` syntax.
See the "Tag Extraction" section in schema.rs for supported tag formats.

### `Edges` Query

Searches for edges and edge fragments, returning deduplicated results by edge key (src_id, dst_id, edge_name):

```rust
use motlie_db::fulltext::{Edges as FulltextEdges, FuzzyLevel};
use motlie_db::reader::Runnable;  // Unified query trait

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

#### `MatchSource`

Indicates what field/document type the search matched against:

```rust
pub enum MatchSource {
    /// Match came from a node's name field
    NodeName,
    /// Match came from a node fragment's content field
    NodeFragment,
    /// Match came from an edge's name field
    EdgeName,
    /// Match came from an edge fragment's content field
    EdgeFragment,
}
```

#### `NodeHit`

```rust
pub struct NodeHit {
    /// BM25 relevance score
    pub score: f32,
    /// Node ID (use to look up full node in RocksDB)
    pub id: Id,
    /// Fragment timestamp (None = node match, Some = fragment match)
    pub fragment_timestamp: Option<u64>,
    /// What field/document type the match came from
    pub match_source: MatchSource,
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
    /// What field/document type the match came from
    pub match_source: MatchSource,
}
```

The `match_source` field indicates where the match came from:
- `NodeName` / `EdgeName` - The match came from the entity's name field
- `NodeFragment` / `EdgeFragment` - The match came from a fragment's content

## TODO: Planned Query Types

### `FulltextFacets` Query

Get counts of documents by facet values:

```rust
// Get all facet counts across all documents
let counts = FulltextFacets::new()
    .run(&reader, Duration::from_secs(5))
    .await?;

// Document type counts: nodes, edges, node_fragments, edge_fragments
println!("Nodes: {}", counts.doc_types.get("nodes").unwrap_or(&0));
println!("Edges: {}", counts.doc_types.get("edges").unwrap_or(&0));

// Tag counts (from #hashtags in content)
for (tag, count) in &counts.tags {
    println!("#{}: {}", tag, count);
}

// Validity structure counts: unbounded, bounded, since_only, until_only
for (validity, count) in &counts.validity {
    println!("{}: {}", validity, count);
}
```

#### Filtering by Document Type

```rust
// Get facet counts only for nodes
let nodes_counts = FulltextFacets::new()
    .with_doc_type_filter(vec!["nodes".to_string()])
    .run(&reader, Duration::from_secs(5))
    .await?;

// Get facet counts for edges and edge fragments
let edge_counts = FulltextFacets::new()
    .with_doc_type_filter(vec!["edges".to_string(), "edge_fragments".to_string()])
    .run(&reader, Duration::from_secs(5))
    .await?;
```

#### Limiting Tag Results

```rust
// Limit tag facets to top 10
let counts = FulltextFacets::new()
    .with_tags_limit(10)
    .run(&reader, Duration::from_secs(5))
    .await?;

// counts.tags will have at most 10 entries
assert!(counts.tags.len() <= 10);
```

The following query types are planned but not yet implemented:

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
- [x] Active period fields (valid_since, valid_until)
- [x] Time range queries on creation_timestamp and validity fields
- [x] `FulltextFacets` query for facet statistics (doc_types, tags, validity)
- [x] `MatchSource` enum to indicate where the match came from (name vs fragment)
- [x] Wildcard search support (`*` and `?` patterns via RegexQuery)

### Planned
- [ ] `Aggregations` query for search analytics
- [ ] `MoreLikeThis` query for similarity search
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
├── search.rs    # Search result types (MatchSource, NodeHit, EdgeHit, FacetCounts)
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
| `test_fulltext_tag_facet_filtering` | `#hashtag` extraction from content, `with_tags()` filter, OR semantics for multiple tags, works for both nodes and edges |
| `test_fulltext_tag_or_semantics` | Explicit verification of OR semantics: documents with mutually exclusive tags, multiple tags return union of matches |
| `test_fulltext_fuzzy_search` | `FuzzyLevel::Low` (1 edit distance), `FuzzyLevel::Medium` (2 edits), typo tolerance for nodes and edges |
| `test_fulltext_facets_query` | `FulltextFacets` query for aggregating facet counts, doc_type filtering, tags_limit option, validity facet counts |

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

Example from `test_fulltext_tag_or_semantics`:

```rust
// Create nodes with mutually exclusive tags
AddNode {
    id: node_a_id,
    name: "NodeA".to_string(),
    summary: NodeSummary::from_text("This document has only the #alpha tag"),
    // ...
}.run(&writer).await?;

AddNode {
    id: node_b_id,
    name: "NodeB".to_string(),
    summary: NodeSummary::from_text("This document has only the #beta tag"),
    // ...
}.run(&writer).await?;

// Query with multiple tags - OR semantics means EITHER tag matches
let results = FulltextNodes::new("document".to_string(), 10)
    .with_tags(vec!["alpha".to_string(), "beta".to_string()])
    .run(&fulltext_reader, timeout)
    .await?;

// With OR semantics: returns 2 results (NodeA has #alpha, NodeB has #beta)
// With AND semantics: would return 0 results (no node has both tags)
assert_eq!(results.len(), 2);

// Verify results contain both nodes
let result_ids: Vec<_> = results.iter().map(|r| r.id).collect();
assert!(result_ids.contains(&node_a_id));  // Has #alpha
assert!(result_ids.contains(&node_b_id));  // Has #beta
```

## See Also

- `tests/test_fulltext_search_pipeline.rs` - **Main integration tests** demonstrating all query features
- `tests/test_pipeline_integration.rs` - Complete integration tests demonstrating all patterns
- `tests/test_fulltext_integration.rs` - Fulltext-specific integration tests
- `tests/test_fulltext_chaining.rs` - Graph-to-fulltext chaining tests
- `src/graph/` - Graph module with parallel design
- `src/README.md` - Module design patterns overview
- [Tantivy Query Parser Documentation](https://docs.rs/tantivy/latest/tantivy/query/struct.QueryParser.html)
