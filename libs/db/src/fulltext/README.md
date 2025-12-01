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
| **Query Consumers** | |
| `spawn_query_consumer(receiver, config, path)` | Single query consumer with new Graph |
| `spawn_query_consumer_pool_shared(receiver, Arc<Graph>, n)` | Pool sharing one Graph |
| `spawn_query_consumer_pool_readonly(receiver, config, path, n)` | Pool with individual Graphs |

### Fulltext Module

| Function | Description |
|----------|-------------|
| **Mutation Consumers** | |
| `spawn_fulltext_consumer(receiver, config, path)` | Single mutation consumer |
| `spawn_fulltext_mutation_consumer_with_next(receiver, config, path, next_tx)` | Mutation consumer that chains to next |
| **Query Consumers** | |
| `spawn_fulltext_query_consumer(receiver, config, path)` | Single query consumer with new Index |
| `spawn_fulltext_query_consumer_pool_shared(receiver, Arc<Index>, n)` | Pool sharing one Index |
| `spawn_fulltext_query_consumer_pool_readonly(receiver, config, path, n)` | Pool with individual Indexes |

## Usage Examples

### Basic Setup

```rust
use motlie_db::{
    create_mutation_writer, create_query_reader, create_fulltext_query_reader,
    spawn_query_consumer_pool_shared, spawn_fulltext_query_consumer_pool_shared,
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
let graph_handles = spawn_query_consumer_pool_shared(
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
        println!("Found: {} (score: {})", result.name, result.score);
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

        println!("Node {} has {} outgoing edges", result.name, edges.len());
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
- **Faceted Search**: Filter by document type, time buckets, weight ranges, and user-defined tags
- **Fuzzy Search**: Typo-tolerant queries with configurable edit distance
- **Tag Extraction**: Automatic extraction of `#hashtags` from content as facets

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
| `doc_type_field` | TEXT | Document type: "node", "edge", "node_fragment", "edge_fragment" |
| `timestamp_field` | I64 | Timestamp in milliseconds |
| `weight_field` | F64 | Edge weight (optional) |

### Facet Fields

| Facet | Path Pattern | Description |
|-------|--------------|-------------|
| `doc_type_facet` | `/type/node`, `/type/edge`, etc. | Document type categorization |
| `time_bucket_facet` | `/time/2024/01`, `/time/2024/02`, etc. | Monthly time buckets |
| `weight_range_facet` | `/weight/0.0-0.5`, `/weight/0.5-1.0`, etc. | Weight range buckets |
| `tags_facet` | `/tag/rust`, `/tag/programming`, etc. | User-defined hashtags |

## Current Query Types

### `Nodes` Query

Searches for nodes and node fragments, returning deduplicated results by node ID:

```rust
let results = FulltextNodes::new("rust programming".to_string(), 10)
    .run(&reader, Duration::from_secs(5))
    .await?;

for result in results {
    println!("ID: {}, Name: {}, Score: {}", result.id, result.name, result.score);
}
```

## TODO: Planned Query Types

The following query types are planned but not yet implemented:

### `Edges` Query
Search for edges by content, similar to `Nodes`:

```rust
// TODO: Not yet implemented
let results = FulltextEdges::new("collaborates".to_string(), 10)
    .run(&reader, Duration::from_secs(5))
    .await?;
```

### `FacetCounts` Query
Get counts of documents by facet values:

```rust
// TODO: Not yet implemented
// Count documents by tag
let tag_counts = FacetCounts::by_tag()
    .run(&reader, Duration::from_secs(5))
    .await?;
// Returns: [("rust", 42), ("programming", 38), ("database", 25), ...]

// Count documents by time bucket
let time_counts = FacetCounts::by_time_bucket()
    .run(&reader, Duration::from_secs(5))
    .await?;
// Returns: [("2024/01", 150), ("2024/02", 200), ("2024/03", 175), ...]

// Count documents by type
let type_counts = FacetCounts::by_doc_type()
    .run(&reader, Duration::from_secs(5))
    .await?;
// Returns: [("node", 500), ("edge", 1200), ("node_fragment", 300), ...]

// Count by weight range (for edges)
let weight_counts = FacetCounts::by_weight_range()
    .run(&reader, Duration::from_secs(5))
    .await?;
// Returns: [("0.0-0.5", 100), ("0.5-1.0", 250), ("1.0-2.0", 75), ...]
```

### `NodesWithFacets` Query
Search nodes with facet filtering:

```rust
// TODO: Not yet implemented
let results = NodesWithFacets::new("machine learning".to_string(), 10)
    .filter_tag("ai")
    .filter_time_range("2024/01", "2024/06")
    .run(&reader, Duration::from_secs(5))
    .await?;
```

### `EdgesWithFacets` Query
Search edges with facet filtering and weight constraints:

```rust
// TODO: Not yet implemented
let results = EdgesWithFacets::new("collaboration".to_string(), 10)
    .filter_weight_range(0.5, 1.0)
    .filter_time_range("2024/01", "2024/12")
    .run(&reader, Duration::from_secs(5))
    .await?;
```

### `Aggregations` Query
Aggregate statistics over search results:

```rust
// TODO: Not yet implemented
let stats = Aggregations::new("programming".to_string())
    .count_by_tag()
    .count_by_time_bucket()
    .avg_weight()  // For edges
    .run(&reader, Duration::from_secs(5))
    .await?;

println!("Total matches: {}", stats.total_count);
println!("Tag distribution: {:?}", stats.tag_counts);
println!("Monthly distribution: {:?}", stats.time_bucket_counts);
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
- [x] Node and node fragment deduplication
- [x] Facet indexing (doc_type, time_bucket, weight_range, tags)
- [x] Fuzzy search support via `FuzzySearchOptions`
- [x] Multiple query consumers with shared Index
- [x] Readonly/Readwrite storage modes

### In Progress
- [ ] `Edges` query (search edges and edge fragments)

### Planned
- [ ] `FacetCounts` query for facet statistics
- [ ] `NodesWithFacets` query with facet filtering
- [ ] `EdgesWithFacets` query with facet filtering
- [ ] `Aggregations` query for search analytics
- [ ] `MoreLikeThis` query for similarity search
- [ ] Highlight support (return matching snippets)
- [ ] Pagination with search_after for deep pagination
- [ ] Custom scoring/boosting per field
- [ ] Temporal filtering (valid_since/valid_until support)

## See Also

- `tests/test_pipeline_integration.rs` - Complete integration tests demonstrating all patterns
- `src/graph.rs` - Graph module with parallel design
- `src/fulltext/query.rs` - Fulltext query implementation
- `src/fulltext/search.rs` - Low-level search utilities
- `src/fulltext/fuzzy.rs` - Fuzzy search implementation
- [Tantivy Query Parser Documentation](https://docs.rs/tantivy/latest/tantivy/query/struct.QueryParser.html)
