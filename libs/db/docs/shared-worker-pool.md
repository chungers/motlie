# Shared Worker Pool for Query Processing

## Overview

This document describes the shared worker pool pattern for query processing in Motlie database, which provides high consistency guarantees by allowing multiple worker threads to share a single TransactionDB instance via `Arc<Graph>`.

## Background

RocksDB's TransactionDB is thread-safe and uses MVCC (Multi-Version Concurrency Control) to provide consistent snapshots across concurrent readers. When multiple threads share the same TransactionDB instance:

- All readers see consistent snapshots based on the sequence number at snapshot creation time
- Changes are immediately visible to new snapshots after commit
- Expected consistency: **99%+** for concurrent reads on shared TransactionDB

In contrast, when each worker opens a separate readonly Storage instance:

- Each instance maintains its own view of the database
- Updates from the write instance may not propagate immediately to readonly instances
- Expected consistency: **25-30%** for separate readonly instances

## API

### `spawn_query_consumer_pool_shared`

Creates a pool of worker threads that share a single Graph instance via Arc.

```rust
pub fn spawn_query_consumer_pool_shared(
    receiver: flume::Receiver<crate::query::Query>,
    graph: Arc<Graph>,
    num_workers: usize,
) -> Vec<JoinHandle<()>>
```

**Parameters:**
- `receiver`: MPMC channel receiver for incoming queries
- `graph`: Arc-wrapped Graph instance to share across all workers
- `num_workers`: Number of worker threads to spawn

**Returns:**
- Vector of join handles for all worker threads

**Consistency:** 99%+ - All workers share the same TransactionDB instance

**Usage:**
```rust
use std::sync::Arc;
use motlie_db::{Graph, spawn_query_consumer_pool_shared};

// Create a Graph with readwrite storage
let graph = Arc::new(Graph::with_storage(
    Storage::readwrite(&db_path, &config)?
));

// Create query channel
let (query_sender, query_receiver) = flume::unbounded();

// Spawn worker pool with shared Graph
let handles = spawn_query_consumer_pool_shared(
    query_receiver,
    graph.clone(),
    num_cpus::get()
);

// Send queries to workers
query_sender.send(query)?;

// Shutdown
drop(query_sender);
for handle in handles {
    handle.await?;
}
```

### `spawn_query_consumer_pool_readonly`

Creates a pool of worker threads where each opens its own readonly Storage instance.

```rust
pub fn spawn_query_consumer_pool_readonly(
    receiver: flume::Receiver<crate::query::Query>,
    config: crate::ReaderConfig,
    db_path: &Path,
    num_workers: usize,
) -> Vec<JoinHandle<()>>
```

**Parameters:**
- `receiver`: MPMC channel receiver for incoming queries
- `config`: Reader configuration
- `db_path`: Path to the RocksDB database
- `num_workers`: Number of worker threads to spawn

**Returns:**
- Vector of join handles for all worker threads

**Consistency:** 25-30% - Each worker has separate readonly Storage instance

**⚠️ Warning:** This function is provided for archival/historical use cases only. For production workloads requiring high consistency, use `spawn_query_consumer_pool_shared` instead.

## Architecture Comparison

### Shared TransactionDB Pattern (Recommended)

```
┌─────────────────────────────────────────────┐
│           Query Senders (Multiple)          │
└────────────────┬────────────────────────────┘
                 │ queries
                 ▼
         ┌──────────────┐
         │ MPMC Channel │
         └──────┬───────┘
                │
       ┌────────┴────────┐
       ▼                 ▼
┌─────────────┐   ┌─────────────┐
│  Worker 1   │   │  Worker N   │
│             │   │             │
│ Arc<Graph>  │   │ Arc<Graph>  │
└──────┬──────┘   └──────┬──────┘
       │                 │
       └────────┬────────┘
                ▼
    ┌──────────────────────┐
    │   TransactionDB      │
    │   (Shared, MVCC)     │
    │   99%+ Consistency   │
    └──────────────────────┘
```

**Advantages:**
- High consistency (99%+)
- Immediate visibility of committed changes
- Efficient resource usage (single DB instance)
- Native MVCC guarantees from RocksDB

**Disadvantages:**
- Requires readwrite mode (cannot use readonly mode)
- All workers share the same sequence number space

### Separate Readonly Instances (Not Recommended)

```
┌─────────────────────────────────────────────┐
│           Query Senders (Multiple)          │
└────────────────┬────────────────────────────┘
                 │ queries
                 ▼
         ┌──────────────┐
         │ MPMC Channel │
         └──────┬───────┘
                │
       ┌────────┴────────┐
       ▼                 ▼
┌─────────────┐   ┌─────────────┐
│  Worker 1   │   │  Worker N   │
│             │   │             │
│   Graph     │   │   Graph     │
└──────┬──────┘   └──────┬──────┘
       │                 │
       ▼                 ▼
┌─────────────┐   ┌─────────────┐
│  Storage    │   │  Storage    │
│ (readonly)  │   │ (readonly)  │
│             │   │             │
│   25-30%    │   │   25-30%    │
│ Consistency │   │ Consistency │
└─────────────┘   └─────────────┘
```

**Advantages:**
- Can use readonly mode
- Workers are isolated

**Disadvantages:**
- Low consistency (25-30%)
- Updates may not propagate to readonly instances
- Higher memory overhead (multiple DB instances)
- Not suitable for production workloads

## Migration Guide

### From Single Consumer to Shared Worker Pool

**Before:**
```rust
let graph = Graph::with_storage(Storage::readonly(&db_path, &config)?);
let handle = spawn_query_consumer(receiver, graph);
```

**After:**
```rust
// Change to readwrite mode
let graph = Arc::new(Graph::with_storage(
    Storage::readwrite(&db_path, &config)?
));

// Use shared worker pool
let handles = spawn_query_consumer_pool_shared(
    receiver,
    graph.clone(),
    num_cpus::get()
);
```

### From Readonly Pool to Shared Pool

**Before:**
```rust
let handles = spawn_query_consumer_pool_readonly(
    receiver,
    config,
    &db_path,
    num_workers
);
```

**After:**
```rust
// Create shared Graph instance
let graph = Arc::new(Graph::with_storage(
    Storage::readwrite(&db_path, &config)?
));

// Use shared worker pool
let handles = spawn_query_consumer_pool_shared(
    receiver,
    graph.clone(),
    num_workers
);
```

## Performance Considerations

### Number of Workers

Choose the number of workers based on your workload:

- **CPU-bound queries:** Use `num_cpus::get()` or `num_cpus::get_physical()`
- **I/O-bound queries:** Can use more workers than CPU cores
- **Mixed workload:** Start with `num_cpus::get()` and tune based on profiling

### Memory Usage

- **Shared pattern:** One Graph instance + thread overhead × N workers
- **Readonly pattern:** Full Storage instance × N workers (much higher)

### Throughput

The shared worker pool provides:
- Linear scaling up to the number of CPU cores for CPU-bound queries
- Better-than-linear scaling for I/O-bound queries (up to RocksDB's I/O limits)
- Consistent query results across all workers

## Testing

The shared worker pool pattern is tested in the integration tests. See:
- `libs/db/src/graph_tests.rs`

To run the tests:
```bash
cargo test --package motlie-db --lib
```

## See Also

- [Concurrency and Storage Modes](concurrency-and-storage-modes.md) - Detailed analysis of consistency guarantees
- [Graph API Documentation](../src/graph.rs) - Full Graph API reference
- [Query Processing](../src/query.rs) - Query types and processing
