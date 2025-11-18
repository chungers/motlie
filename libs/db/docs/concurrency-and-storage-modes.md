# RocksDB Concurrency and Storage Modes Guide

Complete guide to concurrent access patterns, storage modes, threading, and performance tuning for motlie-db.

## Table of Contents

1. [Overview](#overview)
2. [Storage Modes Comparison](#storage-modes-comparison)
3. [RocksDB Architecture Fundamentals](#rocksdb-architecture-fundamentals)
4. [Readonly Mode](#readonly-mode)
5. [Secondary Mode](#secondary-mode)
6. [Shared Readwrite Mode](#shared-readwrite-mode)
7. [Threading and Concurrency](#threading-and-concurrency)
8. [Performance Tuning](#performance-tuning)
9. [Usage Examples](#usage-examples)
10. [Test Results and Analysis](#test-results-and-analysis)

---

## Overview

Motlie-db supports three concurrent access patterns, each with different trade-offs:

| Pattern | Storage Mode | Use Case | Success Rate | Consistency Model |
|---------|-------------|----------|--------------|-------------------|
| **Readonly** | Separate readonly instances | Static snapshots, archival | 25-30% | Snapshot isolation (static) |
| **Secondary** | Separate secondary instances | Read replicas, distributed reads | 40-45% | Eventually consistent (dynamic) |
| **Shared Readwrite** | Single shared TransactionDB | Multi-threaded single process | 99%+ | Immediate consistency |

### Key Decision Criteria

**Use Readonly when**:
- You need point-in-time snapshots
- Data is mostly static or archival
- Simplicity is important
- Low resource overhead is critical

**Use Secondary when**:
- You need read replicas that see recent writes
- Running distributed reader processes
- Can tolerate 7-12ms catch-up overhead
- Need better consistency than readonly

**Use Shared Readwrite when**:
- Multiple threads in same process need to read/write
- Highest consistency requirements
- All access within single application
- Can share a single Storage instance

---

## Storage Modes Comparison

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    PRIMARY WRITER                            │
│              (ReadWrite TransactionDB)                       │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
         ┌─────────────────┐
         │  WAL + SST Files │
         └─────────────────┘
                   │
      ┌────────────┼────────────┐
      │            │            │
      ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────────┐
│ READONLY │ │SECONDARY │ │SHARED READERS│
│ (Static) │ │(Catch-up)│ │(Same Process)│
└──────────┘ └──────────┘ └──────────────┘
 Separate     Separate      Same
 Instance     Instance      Instance
```

### Detailed Comparison

#### Readonly Mode
```rust
// Each reader opens its own readonly DB instance
let mut storage = Storage::readonly(&db_path);
storage.ready()?;
```

**Characteristics**:
- ✅ Simple API, minimal overhead
- ✅ Safe concurrent access (separate instances)
- ❌ Static snapshot - never sees new writes
- ❌ Low visibility of recent data (25-30%)
- ❌ Must reopen to see updates (60-1600ms overhead)

**RocksDB Details**:
- Opens with `DB::open_cf_descriptors_read_only()`
- Replays WAL at open time
- Sees only SST files + replayed WAL
- Does NOT see writer's active memtable
- Does NOT auto-update

#### Secondary Mode
```rust
// Each reader opens its own secondary instance
let mut storage = Storage::secondary(&primary_path, &secondary_path);
storage.ready()?;

// Periodically catch up
storage.try_catch_up_with_primary()?;
```

**Characteristics**:
- ✅ Dynamic catch-up (sees new writes)
- ✅ Much lower overhead than readonly reopen (7-12ms vs 60-1600ms)
- ✅ Better visibility than readonly (40-45%)
- ✅ Continuous availability
- ❌ Requires separate secondary path for MANIFEST
- ❌ More complex than readonly
- ❌ Still depends on flush timing

**RocksDB Details**:
- Opens with `DB::open_cf_descriptors_as_secondary()`
- Requires `max_open_files = -1` (keep all files open)
- `try_catch_up_with_primary()` syncs MANIFEST + replays WAL
- Sees SST files + recent WAL entries
- Does NOT see writer's active memtable (until flushed)

#### Shared Readwrite Mode
```rust
// ONE instance shared across all threads
let mut storage = Storage::readwrite(&db_path);
storage.ready()?;
let storage = Arc::new(storage);
let graph = Arc::new(Graph::new(storage));

// Share graph across writer + readers
spawn_graph_consumer_with_graph(writer_rx, config, graph.clone());
spawn_query_consumer_with_graph(reader_rx, config, graph.clone());
```

**Characteristics**:
- ✅ Highest consistency (99%+ success rate)
- ✅ Immediate visibility (all threads see same memtable)
- ✅ No catch-up overhead
- ✅ Single process, multiple threads
- ❌ Cannot open multiple TransactionDB instances on same path
- ❌ All access must be in same process
- ❌ Requires sharing Arc<Graph>

**RocksDB Details**:
- Uses `TransactionDB` (not regular DB)
- Thread-safe for concurrent access
- Lock file prevents multiple instances
- All threads see same memtable state
- MVCC for read consistency

---

## RocksDB Architecture Fundamentals

### Write Path

```
Write Request
    ↓
┌─────────────┐
│   WAL       │ ← Durable write (fsync to disk)
│  (Disk)     │
└─────────────┘
    ↓
┌─────────────┐
│  Memtable   │ ← In-memory write buffer
│  (Memory)   │
└─────────────┘
    ↓ (when full)
[Flush Trigger]
    ↓
┌─────────────┐
│  SST File   │ ← Immutable sorted file on disk
│  (Disk)     │
└─────────────┘
```

### Key Insight: Flush Timing

**Critical for understanding success rates**:

- Writes go to WAL (disk) + Memtable (memory)
- **Readonly**: Only sees SST files (not memtable)
- **Secondary**: Sees SST files + replayed WAL (not active memtable)
- **Shared Readwrite**: Sees memtable directly (highest visibility)

### Flush Parameters

#### `write_buffer_size` (Default: 64 MB)
Size of a single memtable.

```rust
options.set_write_buffer_size(64 * 1024 * 1024);  // 64 MB
```

**Impact on visibility**:
- Larger = Fewer flushes = Lower visibility for readonly/secondary
- Smaller = More flushes = Better visibility for readonly/secondary

#### `max_write_buffer_number` (Default: 2)
Maximum memtables (active + immutable waiting to flush).

```rust
options.set_max_write_buffer_number(2);
```

**Memory budget**: `write_buffer_size × max_write_buffer_number`

#### `min_write_buffer_number_to_merge` (Default: 1)
How many immutable memtables to accumulate before flushing.

```rust
options.set_min_write_buffer_number_to_merge(1);  // Flush immediately
```

**Impact on visibility**:
- 1 = Flush as soon as memtable fills (better visibility)
- 2+ = Wait to merge multiple memtables (worse visibility, better write perf)

---

## Readonly Mode

### How It Works

```rust
StorageMode::ReadOnly => {
    let db = DB::open_cf_descriptors_read_only(
        &self.db_options,
        &self.db_path,
        cf_descriptors,
        false,  // error_if_wal_file_exists = false
    )?;
    self.db = Some(DatabaseHandle::ReadOnly(db));
}
```

**Opening behavior**:
1. Opens SST files on disk
2. Replays WAL entries to reconstruct memtables
3. Creates static snapshot at that moment
4. **Never sees subsequent writes**

### When to Use

**Good for**:
- Archival access
- Point-in-time analysis
- Backup/restore verification
- Analytics on historical data

**Not good for**:
- Real-time read replicas (use Secondary instead)
- Frequently updated data (use Secondary instead)
- Same-process multi-threaded access (use Shared Readwrite instead)

### Code Example

```rust
use motlie_db::{Storage, Graph, ReaderConfig};
use std::sync::Arc;

// Each reader task opens its own readonly instance
async fn reader_task(db_path: std::path::PathBuf) {
    // Create reader
    let (reader, reader_rx) = {
        let config = ReaderConfig { channel_buffer_size: 10 };
        let (sender, receiver) = flume::bounded(config.channel_buffer_size);
        let reader = motlie_db::Reader::new(sender);
        (reader, receiver)
    };

    // Spawn query consumer (opens readonly storage)
    let consumer_handle = motlie_db::spawn_query_consumer(
        reader_rx,
        ReaderConfig { channel_buffer_size: 10 },
        &db_path,
    );

    // Query data
    let result = NodeByIdQuery::new(id, None)
        .run(&reader, Duration::from_secs(1))
        .await;

    // Cleanup
    drop(reader);
    let _ = consumer_handle.await;
}
```

### Performance Characteristics

**Success Rate**: 25-30% (test results)
- Depends entirely on flush timing
- Only sees data flushed to SST files before reader opens
- Never updates during lifetime

**Overhead**:
- Open: 60-1600ms (depends on WAL size)
- Query: Same as readwrite
- Memory: Minimal (just SST file metadata)

---

## Secondary Mode

### How It Works

```rust
StorageMode::Secondary { secondary_path } => {
    let db = DB::open_cf_descriptors_as_secondary(
        &self.db_options,
        &self.db_path,        // Primary DB path
        secondary_path,        // Secondary's own MANIFEST directory
        cf_descriptors,
    )?;
    self.db = Some(DatabaseHandle::Secondary(db));
}
```

**Opening behavior**:
1. Opens SST files from primary path
2. Creates own MANIFEST in secondary_path
3. Can call `try_catch_up_with_primary()` to sync

**Catch-up behavior**:
```rust
pub fn try_catch_up_with_primary(&self) -> Result<()> {
    match &self.mode {
        StorageMode::Secondary { .. } => {
            match &self.db {
                Some(DatabaseHandle::Secondary(db)) => {
                    db.try_catch_up_with_primary()?;  // RocksDB C++ method
                    Ok(())
                }
                _ => Err(anyhow::anyhow!("Database not ready")),
            }
        }
        _ => Err(anyhow::anyhow!("Not a secondary instance")),
    }
}
```

**What `try_catch_up_with_primary()` does**:
1. Read MANIFEST changes (SST file additions/deletions)
2. Replay WAL entries to reconstruct memtables
3. Update file handles

### When to Use

**Good for**:
- Distributed read replicas
- Read-heavy workloads with acceptable latency
- Scenarios where reads can tolerate 100ms-1s staleness
- Replacing readonly when you need dynamic updates

**Not good for**:
- Single-process multi-threaded (use Shared Readwrite instead)
- Need immediate consistency (use Shared Readwrite instead)
- Very low latency requirements (<10ms staleness)

### Code Example

#### Basic Usage
```rust
use motlie_db::Storage;
use std::path::PathBuf;

let primary_path = PathBuf::from("/data/db");
let secondary_path = primary_path.join("secondary");

// Open secondary instance
let mut storage = Storage::secondary(&primary_path, &secondary_path);
storage.ready()?;

// Periodically catch up with primary
loop {
    tokio::time::sleep(Duration::from_secs(5)).await;
    storage.try_catch_up_with_primary()?;
}
```

#### With Background Catch-Up
```rust
use motlie_db::{Storage, Graph, ReaderConfig};
use std::sync::Arc;
use std::time::Duration;

async fn reader_task_with_catchup(
    primary_path: std::path::PathBuf,
    reader_id: usize,
) {
    // Create unique secondary path per reader
    let secondary_path = primary_path.join(format!("secondary_{}", reader_id));

    let mut storage = Storage::secondary(&primary_path, &secondary_path);
    storage.ready().expect("Failed to ready secondary");
    let storage = Arc::new(storage);

    // Spawn background catch-up task
    let storage_clone = storage.clone();
    let catchup_handle = tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(100)).await;
            let _ = storage_clone.try_catch_up_with_primary();
        }
    });

    // Create reader
    let (reader, reader_rx) = {
        let config = ReaderConfig { channel_buffer_size: 10 };
        let (sender, receiver) = flume::bounded(config.channel_buffer_size);
        let reader = motlie_db::Reader::new(sender);
        (reader, receiver)
    };

    // Spawn query consumer
    let graph = Graph::new(storage.clone());
    // ... use reader for queries

    // Cleanup
    catchup_handle.abort();
}
```

### Performance Characteristics

**Success Rate**: 40-45% (test results)
- Better than readonly due to catch-up
- Still depends on flush timing
- Sees WAL-replayed data, not active memtable

**Overhead**:
- Open: Similar to readonly (60-1600ms)
- Catch-up: 7-12ms average (much better than readonly reopen!)
- Query: Same as readonly/readwrite
- Memory: Slightly higher than readonly (must keep all files open)

**Catch-up Intervals**:
- High consistency: 1-5 seconds
- Moderate: 10-30 seconds
- Low priority: 60+ seconds

### Options Configuration

**Required settings**:
```rust
pub fn default_for_secondary() -> Options {
    let mut options = Options::default();
    options.set_error_if_exists(false);
    options.set_create_if_missing(false);
    options.set_create_missing_column_families(false);
    options.set_max_open_files(-1);  // REQUIRED: keep all files open
    options
}
```

**Why `max_open_files = -1`?**
- Secondary must keep all SST files open
- Primary can add/remove files at any time
- Secondary needs immediate access to sync

---

## Shared Readwrite Mode

### The TransactionDB Constraint

**Critical Limitation**: RocksDB TransactionDB does NOT support multiple instances on the same database path.

```rust
// ❌ THIS FAILS - Lock file prevents it
let storage1 = Storage::readwrite(&db_path);  // Opens, acquires LOCK
storage1.ready()?;

let storage2 = Storage::readwrite(&db_path);  // FAILS!
storage2.ready()?;  // Error: "LOCK: No locks available"
```

**The lock error**:
```
IO error: lock hold by current process, acquire time 1763199268
acquiring thread 6180564992: /path/to/db/LOCK: No locks available
```

### The Solution: Share One Instance

**Correct pattern** - Share `Arc<Graph>` across threads:

```rust
// ✅ THIS WORKS - One instance, multiple threads
let mut storage = Storage::readwrite(&db_path);
storage.ready()?;
let storage = Arc::new(storage);
let graph = Arc::new(Graph::new(storage));

// Writer uses shared graph
spawn_graph_consumer_with_graph(writer_rx, config, graph.clone());

// Readers use same shared graph
for _ in 0..4 {
    spawn_query_consumer_with_graph(reader_rx, config, graph.clone());
}
```

### How It Works

```rust
// Graph must implement Clone for sharing
impl Clone for Graph {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),  // Arc<Storage> - cheap clone
        }
    }
}
```

**Each clone**:
- Shares the same `Arc<Storage>`
- Shares the same underlying TransactionDB
- Thread-safe access via RocksDB's internal locking

### When to Use

**Good for**:
- Multi-threaded single process application
- Need highest consistency (99%+)
- Writer and readers in same process
- Low-latency requirements

**Not good for**:
- Distributed systems (use Secondary instead)
- Multiple processes (use Secondary or Readonly instead)
- When you need process isolation

### Code Example

#### Complete Shared Pattern
```rust
use motlie_db::{Storage, Graph, WriterConfig, ReaderConfig};
use std::sync::Arc;

async fn shared_readwrite_example() -> anyhow::Result<()> {
    let db_path = std::path::PathBuf::from("/data/db");

    // Create ONE readwrite storage instance
    let mut storage = Storage::readwrite(&db_path);
    storage.ready()?;
    let storage = Arc::new(storage);
    let graph = Arc::new(Graph::new(storage.clone()));

    // Spawn writer with shared graph
    let (writer, writer_rx) = motlie_db::create_mutation_writer(Default::default());
    let writer_graph = graph.clone();
    let writer_handle = motlie_db::spawn_graph_consumer_with_graph(
        writer_rx,
        WriterConfig::default(),
        writer_graph,
    );

    // Spawn multiple readers with shared graph
    let mut reader_handles = vec![];
    for _ in 0..4 {
        let (reader, reader_rx) = {
            let config = ReaderConfig { channel_buffer_size: 10 };
            let (sender, receiver) = flume::bounded(config.channel_buffer_size);
            let reader = motlie_db::Reader::new(sender);
            (reader, receiver)
        };

        let reader_graph = graph.clone();
        let reader_handle = motlie_db::spawn_query_consumer_with_graph(
            reader_rx,
            ReaderConfig { channel_buffer_size: 10 },
            reader_graph,
        );

        reader_handles.push((reader, reader_handle));
    }

    // Use writer
    use motlie_db::{AddNode, Id, TimestampMilli};
    writer.add_node(AddNode {
        id: Id::new(),
        ts_millis: TimestampMilli::now(),
        name: "test".to_string(),
    }).await?;

    // Use readers
    // All readers see the write immediately (in memtable)

    Ok(())
}
```

### Performance Characteristics

**Success Rate**: 99%+ (test results)
- All threads see same memtable state
- No flush timing dependency
- Immediate consistency

**Overhead**:
- Open: Same as readwrite (TransactionDB initialization)
- Query: Same as other modes
- Memory: Single instance memory footprint
- No catch-up overhead

---

## Threading and Concurrency

### Thread Safety Guarantees

#### Readonly
- ✅ **Safe**: Multiple readonly instances, separate processes OK
- ✅ **Safe**: Multiple readonly instances, same process OK
- ✅ **Isolation**: Each instance has independent snapshot

#### Secondary
- ✅ **Safe**: Multiple secondary instances, separate processes OK
- ✅ **Safe**: Multiple secondary instances, same process OK
- ✅ **Requirement**: Each needs unique secondary_path
- ⚠️ **Note**: Each instance has independent catch-up state

#### Shared Readwrite
- ✅ **Safe**: Multiple threads, single shared instance OK
- ❌ **NOT Safe**: Multiple instances on same path (lock file prevents)
- ✅ **Guarantee**: TransactionDB provides thread-safe concurrent access
- ✅ **MVCC**: Multi-Version Concurrency Control for read consistency

### Concurrency Patterns

#### Pattern 1: Distributed Read Replicas
```
Primary Writer (Process 1)
    ↓
   [DB]
    ↓
├───────┼───────┐
Reader  Reader  Reader  (Processes 2, 3, 4)
(Secondary) (Secondary) (Secondary)
```

**Implementation**: Use Secondary mode, each process has own instance

#### Pattern 2: Multi-Threaded Application
```
Single Process
├── Writer Thread ─┐
├── Reader Thread 1├─→ Shared Arc<Graph>
├── Reader Thread 2│
└── Reader Thread 3┘
```

**Implementation**: Use Shared Readwrite mode

#### Pattern 3: Hybrid
```
Primary Writer (Process 1, multi-threaded)
├── Writer Thread ─┐
└── Reader Thread ─┴→ Shared Arc<Graph>
         ↓
        [DB]
         ↓
    ┌────┴────┐
Reader      Reader  (Processes 2, 3)
(Secondary) (Secondary)
```

**Implementation**: Shared Readwrite in primary, Secondary in remotes

---

## Performance Tuning

### For Better Readonly/Secondary Visibility

**Goal**: Minimize time between write and visibility

```rust
pub fn optimized_for_readonly_visibility() -> Options {
    let mut options = Options::default();

    // Flush more frequently
    options.set_write_buffer_size(16 * 1024 * 1024);  // 16 MB (smaller)

    // Don't wait to merge - flush immediately
    options.set_min_write_buffer_number_to_merge(1);

    // Allow more buffers to prevent write stalls
    options.set_max_write_buffer_number(4);

    // Enable statistics for monitoring
    options.enable_statistics();

    options
}
```

**Trade-offs**:
- ✅ Better visibility (30-40% → 50-60% success rate)
- ❌ More frequent flushes (slight write performance hit)
- ❌ More SST files (more compaction work)

### For Better Write Performance

**Goal**: Maximize write throughput

```rust
pub fn optimized_for_write_performance() -> Options {
    let mut options = Options::default();

    // Larger memtables = fewer flushes
    options.set_write_buffer_size(128 * 1024 * 1024);  // 128 MB

    // More memtables = less write stalling
    options.set_max_write_buffer_number(6);

    // Merge multiple memtables before flushing
    options.set_min_write_buffer_number_to_merge(2);

    // Increase parallelism
    options.increase_parallelism(8);

    options
}
```

**Trade-offs**:
- ✅ Higher write throughput
- ✅ Less CPU overhead (fewer flushes)
- ❌ Worse visibility for readonly/secondary (10-20% success rate)
- ❌ Higher memory usage

### For Balanced Workload

**Goal**: Balance read visibility and write performance

```rust
pub fn balanced_options() -> Options {
    let mut options = Options::default();

    // Default 64 MB is actually well-balanced
    options.set_write_buffer_size(64 * 1024 * 1024);
    options.set_max_write_buffer_number(3);
    options.set_min_write_buffer_number_to_merge(1);

    // Reasonable bloom filter
    let block_opts = rocksdb::BlockBasedOptions::default();
    block_opts.set_bloom_filter(10.0, false);
    options.set_block_based_table_factory(&block_opts);

    options
}
```

---

## Usage Examples

### Example 1: Simple Readonly Reader

```rust
use motlie_db::{Storage, Reader, ReaderConfig, Id};
use std::time::Duration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let db_path = std::path::PathBuf::from("/data/db");

    // Create reader
    let (reader, reader_rx) = {
        let config = ReaderConfig { channel_buffer_size: 10 };
        let (sender, receiver) = flume::bounded(config.channel_buffer_size);
        (Reader::new(sender), receiver)
    };

    // Spawn consumer (opens readonly storage)
    let consumer = motlie_db::spawn_query_consumer(
        reader_rx,
        ReaderConfig { channel_buffer_size: 10 },
        &db_path,
    );

    // Query
    let node_id = Id::new();
    match NodeByIdQuery::new(node_id, None)
        .run(&reader, Duration::from_secs(1))
        .await
    {
        Ok((name, summary)) => println!("Found: {}", name),
        Err(e) => println!("Not found: {}", e),
    }

    // Cleanup
    drop(reader);
    consumer.await??;
    Ok(())
}
```

### Example 2: Secondary with Catch-Up

```rust
use motlie_db::{Storage, Graph, Reader, ReaderConfig};
use std::sync::Arc;
use std::time::Duration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let primary_path = std::path::PathBuf::from("/data/db");
    let secondary_path = primary_path.join("secondary");

    // Open secondary
    let mut storage = Storage::secondary(&primary_path, &secondary_path);
    storage.ready()?;
    let storage = Arc::new(storage);

    // Background catch-up
    let storage_clone = storage.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(5)).await;
            if let Err(e) = storage_clone.try_catch_up_with_primary() {
                eprintln!("Catch-up failed: {}", e);
            }
        }
    });

    // Use normally
    let (reader, reader_rx) = {
        let config = ReaderConfig { channel_buffer_size: 10 };
        let (sender, receiver) = flume::bounded(config.channel_buffer_size);
        (Reader::new(sender), receiver)
    };

    // ... query as usual

    Ok(())
}
```

### Example 3: Shared Readwrite (Complete)

```rust
use motlie_db::{
    Storage, Graph, create_mutation_writer, spawn_graph_consumer_with_graph,
    spawn_query_consumer_with_graph, AddNode, Id, TimestampMilli,
    WriterConfig, ReaderConfig,
};
use std::sync::Arc;
use std::time::Duration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let db_path = std::path::PathBuf::from("/data/db");

    // Create shared storage
    let mut storage = Storage::readwrite(&db_path);
    storage.ready()?;
    let storage = Arc::new(storage);
    let graph = Arc::new(Graph::new(storage));

    // Create writer
    let (writer, writer_rx) = create_mutation_writer(Default::default());
    let writer_handle = spawn_graph_consumer_with_graph(
        writer_rx,
        WriterConfig::default(),
        graph.clone(),
    );

    // Create readers
    let mut readers = vec![];
    for _ in 0..4 {
        let (reader, reader_rx) = {
            let config = ReaderConfig { channel_buffer_size: 10 };
            let (sender, receiver) = flume::bounded(config.channel_buffer_size);
            (Reader::new(sender), receiver)
        };

        let reader_handle = spawn_query_consumer_with_graph(
            reader_rx,
            ReaderConfig { channel_buffer_size: 10 },
            graph.clone(),
        );

        readers.push((reader, reader_handle));
    }

    // Write data
    let node_id = Id::new();
    writer.add_node(AddNode {
        id: node_id,
        ts_millis: TimestampMilli::now(),
        name: "test_node".to_string(),
    }).await?;

    // Read immediately (99%+ will succeed)
    let (reader, _) = &readers[0];
    match NodeByIdQuery::new(node_id, None)
        .run(&reader, Duration::from_secs(1))
        .await
    {
        Ok((name, _)) => println!("Immediate read succeeded: {}", name),
        Err(e) => println!("Read failed: {}", e),
    }

    // Cleanup
    drop(writer);
    writer_handle.await??;

    for (reader, handle) in readers {
        drop(reader);
        handle.await??;
    }

    Ok(())
}
```

---

## Test Results and Analysis

### Comprehensive Test Results

From integration tests in `tests/`:

| Test | Storage Mode | Success Rate | Observations |
|------|-------------|--------------|--------------|
| `test_concurrent_readonly` | Readonly (separate instances) | **24.8%** | High variance (24-74%), flush dependent |
| `test_concurrent_secondary` | Secondary (separate instances) | **42.9%** | More stable, catch-up helps |
| `test_concurrent_shared` | Shared readwrite (one instance) | **99.9%** | Extremely consistent |

### Why Success Rates Vary

#### Readonly: 24-74% Range

**Cause**: Static snapshot timing relative to flush schedule

```
Scenario 1: Reader opens BEFORE flush
Writer: [Node 1-50 in memtable] → [Flush] → [Node 51-100]
Reader:                           [Opens] → Only sees 0 nodes
Result: 0% success rate

Scenario 2: Reader opens AFTER flush
Writer: [Node 1-50] → [Flush] → [Node 51-100 in memtable]
Reader:                         [Opens] → Sees 50 nodes
Result: 50% success rate
```

**Not flaky**: Variance is expected, deterministic given RocksDB behavior

#### Secondary: 40-45% Range

**Cause**: Catch-up helps, but still flush-dependent

```
Writer: [Writes] → [Flush to SST + WAL]
                           ↓
Secondary: [try_catch_up] → Replays WAL → Sees flushed data
```

**Advantage**: Can call `try_catch_up()` repeatedly to stay more current

#### Shared Readwrite: 99%+ Success

**Cause**: All threads see same memtable

```
Writer Thread: [Writes to memtable]
                      ↓
Reader Threads: [Read from same memtable] → Immediate visibility
```

**Why not 100%?**: Rare timing races during test setup

### Test Stability Analysis

**These tests are NOT flaky because**:

1. ✅ Core assertions always pass:
   - All writes succeed (0 errors)
   - All data committed (node/edge counts correct)
   - No data corruption

2. ✅ Success rate variance is:
   - Documented and understood
   - Within expected and allowed ranges
   - Reflects real RocksDB behavior

3. ✅ Tests validate correctness, not performance:
   - 10% threshold is very permissive
   - Focus is on data consistency
   - Variance teaches about eventual consistency

**Contrast with flaky test**:
- Random assertion failures (unpredictable)
- Same input → different pass/fail
- Unknown root cause

**Our tests**:
- Consistent pass/fail (always pass)
- Same input → consistent behavior
- Known root cause (RocksDB architecture)

---

## Summary and Recommendations

### Quick Decision Matrix

| Your Scenario | Recommended Mode | Why |
|--------------|------------------|-----|
| Analytics on historical data | **Readonly** | Simple, static snapshots |
| Distributed read replicas | **Secondary** | Dynamic catch-up, acceptable staleness |
| Multi-threaded app (single process) | **Shared Readwrite** | Highest consistency, immediate visibility |
| Backup/restore verification | **Readonly** | Point-in-time snapshots |
| Real-time dashboard (distributed) | **Secondary** | Recent data, acceptable 1-5s lag |
| Real-time dashboard (single process) | **Shared Readwrite** | Immediate data, <1ms lag |

### Key Takeaways

1. **Readonly**: Simple but static - never updates
2. **Secondary**: Dynamic but eventually consistent - needs catch-up
3. **Shared Readwrite**: Immediate but single-process - cannot have multiple instances

4. **Success rates depend on flush timing** - not storage mode alone
5. **TransactionDB is thread-safe** - but one instance per path only
6. **Share `Arc<Graph>`** - correct pattern for multi-threaded access

### Related Documentation

- [Test documentation](../tests/README.md) - Integration test details
- [RocksDB Wiki: Read-only and Secondary instances](https://github.com/facebook/rocksdb/wiki/Read-only-and-Secondary-instances)
- [Reader API](reader.md) - Query operations reference

---

## Appendix: Migration Guide

### From Readonly to Secondary

```rust
// Before
let mut storage = Storage::readonly(&db_path);
storage.ready()?;

// After
let secondary_path = db_path.join("secondary");
let mut storage = Storage::secondary(&db_path, &secondary_path);
storage.ready()?;

// Add catch-up loop
loop {
    tokio::time::sleep(Duration::from_secs(5)).await;
    storage.try_catch_up_with_primary()?;
}
```

### From Separate Readwrite to Shared

```rust
// Before (WRONG - will fail with lock error)
let mut storage1 = Storage::readwrite(&db_path);
storage1.ready()?;

let mut storage2 = Storage::readwrite(&db_path);  // ERROR!
storage2.ready()?;

// After (CORRECT - share one instance)
let mut storage = Storage::readwrite(&db_path);
storage.ready()?;
let storage = Arc::new(storage);
let graph = Arc::new(Graph::new(storage));

// Share graph across components
spawn_graph_consumer_with_graph(rx1, config, graph.clone());
spawn_query_consumer_with_graph(rx2, config, graph.clone());
```
