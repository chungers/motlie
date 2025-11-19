# Query API Guide

**Status**: ✅ Current (as of 2025-11-17)

This guide covers the modern Query API introduced in v0.2.0, which provides a clean, type-driven interface for querying the motlie graph database.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Query Types](#query-types)
- [Common Patterns](#common-patterns)
- [Advanced Usage](#advanced-usage)
- [Migration Guide](#migration-guide)

## Overview

The Query API follows a simple, idiomatic Rust pattern:

```rust
QueryType::new(...params, reference_timestamp)
    .run(&reader, timeout)
    .await?
```

### Key Features

1. **Type-Driven** - Query type determines result type
2. **Timeout at Execution** - Timeout is a runtime parameter, not construction
3. **No Builder Boilerplate** - Direct query construction
4. **Async-First** - Built on `tokio` for concurrent queries
5. **Composable** - Queries are values that can be stored and reused

## Quick Start

```rust
use motlie_db::{Reader, NodeById, Runnable, Id, TimestampMilli};
use std::time::Duration;

async fn example(reader: &Reader) -> anyhow::Result<()> {
    let node_id = Id::new();
    let timeout = Duration::from_secs(5);

    // Simple query
    let (name, summary) = NodeById::new(node_id, None)
        .run(reader, timeout)
        .await?;

    println!("Node: {} - {:?}", name, summary);
    Ok(())
}
```

## Core Concepts

### Reader

The `Reader` is a handle for executing queries against the database:

```rust
use motlie_db::{Reader, spawn_query_consumer};

// Create a reader (typically during app initialization)
let reader = spawn_query_consumer(receiver, storage);
```

The `Reader` is:
- **Cloneable** - Can be shared across tasks/threads
- **Lightweight** - Just a channel sender
- **Thread-safe** - Safe to use from multiple tokio tasks

### Queries

Queries are lightweight value types that describe what data to fetch:

```rust
// Construct a query (no I/O, just data)
let query = NodeById::new(node_id, None);

// Execute it (performs I/O)
let result = query.run(&reader, timeout).await?;
```

### Runnable Trait

All queries implement the `Runnable` trait:

```rust
pub trait Runnable {
    type Output: Send + 'static;
    async fn run(self, reader: &Reader, timeout: Duration) -> Result<Self::Output>;
}
```

This enables generic query execution and composability.

### Timeout

Timeout is an execution parameter, not a query property:

```rust
let query = NodeById::new(id, None);

// Try with short timeout
match query.run(&reader, Duration::from_millis(100)).await {
    Ok(result) => println!("Fast query succeeded"),
    Err(_) => {
        // Retry with longer timeout
        let query2 = NodeById::new(id, None);
        let result = query2.run(&reader, Duration::from_secs(5)).await?;
    }
}
```

### Temporal Queries

All queries support temporal validity filtering via an optional reference timestamp:

```rust
let ref_time = TimestampMilli::now();

// Query as-of a specific time
let (name, summary) = NodeById::new(node_id, Some(ref_time))
    .run(&reader, timeout)
    .await?;

// Query current state (None = current time)
let (name, summary) = NodeById::new(node_id, None)
    .run(&reader, timeout)
    .await?;
```

## Query Types

### 1. NodeById

Fetch a node by its ID:

```rust
let (name, summary) = NodeById::new(id, ref_ts)
    .run(&reader, timeout)
    .await?;
```

**Returns**: `(NodeName, NodeSummary)`
- `NodeName` - String name of the node
- `NodeSummary` - DataUrl containing node data

### 2. EdgeSummaryBySrcDstName

Fetch an edge by its topology (source, destination, name):

```rust
let (summary, weight) = EdgeSummaryBySrcDstName::new(
    source_id,
    dest_id,
    edge_name,
    ref_ts
)
.run(&reader, timeout)
.await?;
```

**Returns**: `(EdgeSummary, Option<f64>)`
- `EdgeSummary` - DataUrl containing edge data
- `Option<f64>` - Optional edge weight (None if not weighted)

### 3. FragmentsByIdTimeRange

Fetch time-series fragments for an entity:

```rust
use std::ops::Bound;

let start = TimestampMilli::now();
let end = TimestampMilli(start.0 + 86400000); // +24h

let fragments = FragmentsByIdTimeRange::new(
    entity_id,
    (Bound::Included(start), Bound::Excluded(end)),
    ref_ts
)
.run(&reader, timeout)
.await?;

// Returns: Vec<(TimestampMilli, FragmentContent)>
for (ts, content) in fragments {
    println!("Fragment at {}: {}", ts.0, content.decode_string()?);
}
```

**Returns**: `Vec<(TimestampMilli, FragmentContent)>`

### 4. OutgoingEdges

Get all outgoing edges from a node with weights:

```rust
let outgoing = OutgoingEdges::new(node_id, ref_ts)
    .run(&reader, timeout)
    .await?;

// Returns: Vec<(Option<f64>, SrcId, DstId, EdgeName)>
// Note: Weight comes first (metadata before topology)
for (weight, src, dst, edge_name) in outgoing {
    match weight {
        Some(w) => println!("{} --{}({})-> {}", src, edge_name, w, dst),
        None => println!("{} --{}-> {}", src, edge_name, dst),
    }
}
```

**Returns**: `Vec<(Option<f64>, SrcId, DstId, EdgeName)>`
- Tuple ordering: weight-first design (metadata before topology)
- Weight is `None` for unweighted edges

### 5. IncomingEdges

Get all incoming edges to a node with weights:

```rust
let incoming = IncomingEdges::new(node_id, ref_ts)
    .run(&reader, timeout)
    .await?;

// Returns: Vec<(Option<f64>, DstId, SrcId, EdgeName)>
// Note: Weight comes first (metadata before topology)
for (weight, dst, src, edge_name) in incoming {
    match weight {
        Some(w) => println!("{} --{}({})-> {}", src, edge_name, w, dst),
        None => println!("{} --{}-> {}", src, edge_name, dst),
    }
}
```

**Returns**: `Vec<(Option<f64>, DstId, SrcId, EdgeName)>`
- Tuple ordering: weight-first design (metadata before topology)
- Weight is `None` for unweighted edges

### 6. NodesByName

Prefix search for nodes with pagination:

```rust
let nodes = NodesByName::new(
    "user_".to_string(),    // prefix
    None,                    // start cursor (for pagination)
    Some(100),              // page size
    ref_ts
)
.run(&reader, timeout)
.await?;

// Returns: Vec<(NodeName, Id)>
for (name, id) in &nodes {
    println!("Node: {} ({})", name, id);
}
```

**Returns**: `Vec<(NodeName, Id)>`

**Pagination Example**:
```rust
let mut cursor = None;

loop {
    let page = NodesByName::new(
        "user_".to_string(),
        cursor.clone(),
        Some(100),
        None
    )
    .run(&reader, timeout)
    .await?;

    if page.is_empty() {
        break;
    }

    // Process page...
    for (name, id) in &page {
        println!("{}: {}", name, id);
    }

    // Set cursor to last item for next page
    cursor = page.last().cloned();
}
```

### 7. EdgesByName

Prefix search for edges with pagination:

```rust
let edges = EdgesByName::new(
    "follows_".to_string(),  // prefix
    None,                     // start cursor
    Some(100),               // page size
    ref_ts
)
.run(&reader, timeout)
.await?;

// Returns: Vec<(EdgeName, Id)>
```

**Returns**: `Vec<(EdgeName, Id)>`

## Common Patterns

### Pattern 1: Simple Point Query

```rust
async fn get_node(
    reader: &Reader,
    node_id: Id
) -> anyhow::Result<(NodeName, NodeSummary)> {
    NodeById::new(node_id, None)
        .run(reader, Duration::from_secs(5))
        .await
}
```

### Pattern 2: Multiple Concurrent Queries

```rust
use tokio::try_join;

async fn get_node_neighborhood(
    reader: &Reader,
    node_id: Id,
    timeout: Duration
) -> anyhow::Result<()> {
    // Launch all queries concurrently
    let (node_info, outgoing, incoming) = try_join!(
        NodeById::new(node_id, None).run(reader, timeout),
        OutgoingEdges::new(node_id, None).run(reader, timeout),
        IncomingEdges::new(node_id, None).run(reader, timeout)
    )?;

    let (name, summary) = node_info;
    println!("Node {} has {} outgoing and {} incoming edges",
        name, outgoing.len(), incoming.len());

    Ok(())
}
```

### Pattern 3: Batch Queries

```rust
use futures::future::try_join_all;

async fn get_multiple_nodes(
    reader: &Reader,
    node_ids: Vec<Id>,
    timeout: Duration
) -> anyhow::Result<Vec<(NodeName, NodeSummary)>> {
    // Create all queries
    let queries = node_ids.into_iter()
        .map(|id| NodeById::new(id, None).run(reader, timeout));

    // Execute concurrently
    try_join_all(queries).await
}
```

### Pattern 4: Temporal Query

```rust
async fn get_historical_state(
    reader: &Reader,
    node_id: Id,
    timestamp: TimestampMilli,
    timeout: Duration
) -> anyhow::Result<(NodeName, NodeSummary)> {
    NodeById::new(node_id, Some(timestamp))
        .run(reader, timeout)
        .await
}
```

### Pattern 5: Pagination

```rust
async fn list_all_users(
    reader: &Reader,
    timeout: Duration
) -> anyhow::Result<Vec<(NodeName, Id)>> {
    let mut all_users = Vec::new();
    let mut cursor = None;

    loop {
        let page = NodesByName::new(
            "user_".to_string(),
            cursor.clone(),
            Some(100),
            None
        )
        .run(reader, timeout)
        .await?;

        if page.is_empty() {
            break;
        }

        all_users.extend(page.iter().cloned());
        cursor = page.last().cloned();
    }

    Ok(all_users)
}
```

### Pattern 6: Graph Traversal

```rust
async fn get_friends_of_friends(
    reader: &Reader,
    user_id: Id,
    timeout: Duration
) -> anyhow::Result<Vec<Id>> {
    // Get immediate friends
    let friends = OutgoingEdges::new(user_id, None)
        .run(reader, timeout)
        .await?;

    let mut fof = Vec::new();

    // Get friends of each friend concurrently
    for (weight, src, friend_id, edge_name) in friends {
        let friend_edges = OutgoingEdges::new(friend_id, None)
            .run(reader, timeout)
            .await?;

        for (_, _, fof_id, _) in friend_edges {
            if fof_id != user_id {  // Skip original user
                fof.push(fof_id);
            }
        }
    }

    // Deduplicate
    fof.sort();
    fof.dedup();

    Ok(fof)
}
```

### Pattern 7: Error Handling

```rust
async fn safe_query(
    reader: &Reader,
    node_id: Id,
    timeout: Duration
) -> anyhow::Result<Option<(NodeName, NodeSummary)>> {
    match NodeById::new(node_id, None).run(reader, timeout).await {
        Ok(result) => Ok(Some(result)),
        Err(e) if e.to_string().contains("not found") => Ok(None),
        Err(e) if e.to_string().contains("timeout") => {
            eprintln!("Query timed out, retrying...");
            // Retry with longer timeout
            NodeById::new(node_id, None)
                .run(reader, Duration::from_secs(30))
                .await
                .map(Some)
        }
        Err(e) => Err(e),
    }
}
```

## Advanced Usage

### Generic Query Execution

```rust
use motlie_db::Runnable;

async fn execute_with_retry<Q>(
    query: Q,
    reader: &Reader,
    max_attempts: usize,
    timeout: Duration
) -> anyhow::Result<Q::Output>
where
    Q: Runnable + Clone,
{
    for attempt in 1..=max_attempts {
        match query.clone().run(reader, timeout).await {
            Ok(result) => return Ok(result),
            Err(e) if attempt < max_attempts => {
                eprintln!("Attempt {} failed: {}", attempt, e);
                tokio::time::sleep(Duration::from_millis(100 * attempt as u64)).await;
            }
            Err(e) => return Err(e),
        }
    }
    unreachable!()
}

// Usage
let query = NodeById::new(id, None);
let result = execute_with_retry(query, &reader, 3, timeout).await?;
```

### Query Composition

```rust
async fn get_edge_with_endpoints(
    reader: &Reader,
    src_id: Id,
    dst_id: Id,
    edge_name: EdgeName,
    timeout: Duration
) -> anyhow::Result<EdgeData> {
    // Get edge info and endpoint node data concurrently
    let ((edge_summary, weight), (src_name, src_summary), (dst_name, dst_summary)) = try_join!(
        EdgeSummaryBySrcDstName::new(src_id, dst_id, edge_name.clone(), None).run(reader, timeout),
        NodeById::new(src_id, None).run(reader, timeout),
        NodeById::new(dst_id, None).run(reader, timeout)
    )?;

    Ok(EdgeData {
        edge_name,
        edge_summary,
        weight,
        source: (src_id, src_name, src_summary),
        destination: (dst_id, dst_name, dst_summary),
    })
}

struct EdgeData {
    edge_name: EdgeName,
    edge_summary: EdgeSummary,
    weight: Option<f64>,
    source: (Id, NodeName, NodeSummary),
    destination: (Id, NodeName, NodeSummary),
}
```

### Custom Timeout Strategy

```rust
async fn query_with_adaptive_timeout<Q>(
    query: Q,
    reader: &Reader,
    initial_timeout: Duration,
) -> anyhow::Result<Q::Output>
where
    Q: Runnable,
{
    let timeouts = [
        initial_timeout,
        initial_timeout * 2,
        initial_timeout * 4,
    ];

    for (attempt, timeout) in timeouts.iter().enumerate() {
        match query.run(reader, *timeout).await {
            Ok(result) => {
                if attempt > 0 {
                    eprintln!("Query succeeded on attempt {} with timeout {:?}",
                        attempt + 1, timeout);
                }
                return Ok(result);
            }
            Err(e) if e.to_string().contains("timeout") && attempt < timeouts.len() - 1 => {
                eprintln!("Timeout with {:?}, retrying with longer timeout", timeout);
                continue;
            }
            Err(e) => return Err(e),
        }
    }

    unreachable!()
}
```

## Migration Guide

### From Deprecated Reader API (v0.1.x)

**Before (Deprecated)**:
```rust
// Old API - timeout in method call
let (name, summary) = reader
    .node_by_id(id, Some(ref_time), Duration::from_secs(5))
    .await?;
```

**After (New API)**:
```rust
// New API - timeout in run()
let (name, summary) = NodeById::new(id, Some(ref_time))
    .run(&reader, Duration::from_secs(5))
    .await?;
```

### Migration Checklist

1. ✅ Import query types:
   ```rust
   use motlie_db::{NodeById, EdgeSummaryBySrcDstName, Runnable};
   ```

2. ✅ Update method calls:
   ```rust
   // Old (deprecated): reader.node_by_id(id, ref_ts, timeout)
   // New: NodeById::new(id, ref_ts).run(&reader, timeout)
   ```

   **Common method migrations:**
   ```rust
   // NodeById
   // Old: reader.node_by_id(id, ref_ts, timeout).await?
   NodeById::new(id, ref_ts).run(&reader, timeout).await?

   // EdgeSummaryBySrcDstName
   // Old: reader.edge_summary_by_src_dst_name(src, dst, name, ref_ts, timeout).await?
   EdgeSummaryBySrcDstName::new(src, dst, name, ref_ts).run(&reader, timeout).await?

   // FragmentsByIdTimeRange
   // Old: reader.fragments_by_id_time_range(id, range, ref_ts, timeout).await?
   FragmentsByIdTimeRange::new(id, range, ref_ts).run(&reader, timeout).await?

   // OutgoingEdges
   // Old: reader.edges_from_node_by_id(node_id, ref_ts, timeout).await?
   OutgoingEdges::new(node_id, ref_ts).run(&reader, timeout).await?

   // IncomingEdges
   // Old: reader.edges_to_node_by_id(node_id, ref_ts, timeout).await?
   IncomingEdges::new(node_id, ref_ts).run(&reader, timeout).await?

   // NodesByName
   // Old: reader.nodes_by_name(prefix, cursor, limit, ref_ts, timeout).await?
   NodesByName::new(prefix, cursor, limit, ref_ts).run(&reader, timeout).await?

   // EdgesByName
   // Old: reader.edges_by_name(prefix, cursor, limit, ref_ts, timeout).await?
   EdgesByName::new(prefix, cursor, limit, ref_ts).run(&reader, timeout).await?
   ```

3. ✅ Parameter order changed:
   - Old: `(id, reference_ts, timeout)`
   - New: `(id, reference_ts)` + `timeout` in `run()`

4. ✅ Enable new patterns:
   - Store queries as values
   - Retry with different timeouts
   - Generic query execution

### Compatibility

The old Reader methods (`reader.node_by_id()`, etc.) are **deprecated but still functional**. You can migrate gradually:

```rust
// Both work during migration period:

// Old API (deprecated)
#[allow(deprecated)]
let result1 = reader.node_by_id(id, None, timeout).await?;

// New API (recommended)
let result2 = NodeById::new(id, None).run(&reader, timeout).await?;

assert_eq!(result1, result2);
```

## Best Practices

1. **Use Appropriate Timeouts**
   - Point queries: 1-5 seconds
   - Range scans: 5-30 seconds
   - Large scans: Consider streaming or pagination

2. **Leverage Concurrency**
   - Use `tokio::try_join!` for independent queries
   - Launch multiple queries in parallel when safe
   - Avoid sequential queries when not necessary

3. **Handle Errors Gracefully**
   - Distinguish between "not found" and "timeout"
   - Retry on timeout with exponential backoff
   - Log query failures for debugging

4. **Use Pagination for Large Results**
   - Set reasonable page sizes (100-1000 items)
   - Avoid loading entire datasets into memory
   - Use cursor-based pagination for consistency

5. **Consider Temporal Queries**
   - Pass `None` for current state (most common)
   - Use specific timestamps for historical queries
   - Remember: temporal validity is optional (None = always valid)

## Performance Considerations

### Query Performance

- **Point lookups** (`NodeById`, `EdgeSummaryBySrcDstName`): O(1) - ~1-10µs
- **Prefix scans** (`NodesByName`, `EdgesByName`): O(K) where K = results - ~10-100µs per page
- **Range scans** (`FragmentsByIdTimeRange`): O(K) where K = results - ~10-100µs per page
- **Topology queries** (`OutgoingEdges`, `IncomingEdges`): O(degree) - ~10µs + 1µs per edge

### Concurrency

The Query API is designed for high concurrency:
- Reader is cloneable and thread-safe
- Multiple queries can execute concurrently
- Internal query queue handles backpressure
- Timeouts prevent indefinite blocking

### Caching

Consider caching for frequently accessed data:
```rust
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;

struct CachedReader {
    reader: Reader,
    cache: Arc<RwLock<HashMap<Id, (NodeName, NodeSummary)>>>,
}

impl CachedReader {
    async fn get_node(&self, id: Id, timeout: Duration) -> anyhow::Result<(NodeName, NodeSummary)> {
        // Check cache first
        {
            let cache = self.cache.read().await;
            if let Some(result) = cache.get(&id) {
                return Ok(result.clone());
            }
        }

        // Query database
        let result = NodeById::new(id, None)
            .run(&self.reader, timeout)
            .await?;

        // Update cache
        {
            let mut cache = self.cache.write().await;
            cache.insert(id, result.clone());
        }

        Ok(result)
    }
}
```

## See Also

- [Reader API Reference](reader.md) - Complete API documentation
- [Concurrency Guide](concurrency-and-storage-modes.md) - Threading and storage modes
- [Schema Design](variable-length-fields-in-keys.md) - Database internals
- [Main README](../README.md) - Library overview
