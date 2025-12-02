# Integration Tests

This directory contains integration tests for the motlie-db library.

## Test Organization

### Concurrent Access Tests
Tests for concurrent read/write patterns with different storage modes. All share common utilities from `common/concurrent_test_utils.rs`.

#### `test_concurrent_readonly.rs`
**Pattern**: Readonly readers (separate DB instances)
- 1 readwrite writer + 4 readonly readers
- Each reader opens its own readonly Storage instance
- **Success rate**: ~25-30% (depends on flush timing)
- **Use case**: Static snapshot reads, archival access

**Key test**: `test_concurrent_read_write_integration()`

#### `test_concurrent_secondary.rs`
**Pattern**: Secondary readers (separate DB instances with catch-up)
- 1 readwrite writer + 4 secondary readers
- Each reader opens its own secondary Storage instance
- Background catch-up task syncs with primary every 100ms
- **Success rate**: ~40-45% (depends on flush timing + catch-up)
- **Catch-up overhead**: 7-12ms average
- **Use case**: Dynamic read replicas, distributed reads

**Key test**: `test_concurrent_read_write_with_secondary()`

#### `test_concurrent_shared.rs`
**Pattern**: Shared readwrite (single TransactionDB instance)
- 1 writer + 4 readers sharing ONE TransactionDB instance
- All access same `Arc<Graph>` wrapping shared Storage
- **Success rate**: ~99%+ (immediate consistency within process)
- **Use case**: Multi-threaded access in single process, highest consistency

**Key test**: `test_concurrent_read_write_with_readwrite_readers()`

**Note**: RocksDB TransactionDB does NOT support multiple instances on same path (lock file prevents). This test demonstrates the correct pattern: share a single instance across threads.

### API Tests

#### `test_secondary_api.rs`
Basic API tests for secondary instance functionality (4 tests):
- `test_secondary_instance_basic()` - Open secondary, catch-up, query
- `test_secondary_catch_up_sees_new_writes()` - Catch-up after new writes
- `test_secondary_instance_creation()` - Instance creation validation
- `test_try_catch_up_on_non_secondary_fails()` - Error handling

### Fulltext Integration Tests

#### `test_fulltext_integration.rs`
Tests for fulltext search functionality (3 tests):
- Basic indexing and search
- Fuzzy search with typo tolerance
- Multi-document search and ranking

#### `test_fulltext_chaining.rs`
Tests for graph-to-fulltext consumer chaining (3 tests):
- Mutation flow from graph to fulltext
- Index consistency after chained writes
- Shutdown and flush behavior

### Pipeline Integration Tests

#### `test_pipeline_integration.rs`
End-to-end pipeline tests (4 tests):
- Complete mutation pipeline with consumers
- Query processing across graph and fulltext
- Concurrent client scenarios
- Mixed query workloads

### Validation Tests

#### `test_prefix_scan_bug.rs`
Validates RocksDB prefix scanning with direct byte encoding (7 tests).
Tests that variable-length fields in keys work correctly with prefix extractors.

## Shared Test Utilities

### `common/concurrent_test_utils.rs`
Shared infrastructure for concurrent tests:
- `Metrics` - Tracks success/error counts, latency statistics
- `TestContext` - Coordinates writers and readers (shared node/edge IDs, stop signal)
- `writer_task()` - Common writer implementation

**Benefits**:
- ~370 lines of duplication eliminated
- Consistent metrics collection
- Easy to add new concurrent test variants

## Running Tests

```bash
# Run all tests
cargo test

# Run specific test category
cargo test test_concurrent        # All concurrent tests
cargo test test_secondary         # Secondary-related tests

# Run with output
cargo test test_concurrent_shared -- --nocapture

# Run specific test
cargo test test_concurrent_read_write_integration -- --nocapture
```

## Success Rate Comparison

| Test | Storage Mode | Success Rate | Key Characteristic |
|------|-------------|--------------|-------------------|
| Readonly | Separate readonly instances | 25-30% | Static snapshots, flush-dependent |
| Secondary | Separate secondary instances | 40-45% | Dynamic catch-up, 7-12ms overhead |
| Shared | ONE shared TransactionDB | **99%+** | Immediate consistency, thread-safe |

## Key Learnings

### TransactionDB Concurrency Model
- ❌ **Does NOT support**: Multiple TransactionDB instances on same path (lock file prevents)
- ✅ **Does support**: Thread-safe concurrent access to single instance
- **Correct pattern**: Share `Arc<Graph>` (wrapping `Arc<Storage>`) across threads

### API Functions for Shared Storage
- `spawn_graph_query_consumer_with_graph()` - Share Storage across query consumers
- `spawn_graph_consumer_with_graph()` - Share Storage for mutation consumer
- Both require `Graph` to implement `Clone` (shallow clone of Arc)

### Success Rate Dependencies
All concurrent tests show that success rates depend primarily on **flush timing** (when data hits disk), not just storage mode:
- Writes go to memtable first
- Readers can't see data until it's flushed to SST files
- Shared readwrite has highest rate because all threads see same memtable state

## Test Metrics

Each test reports:
- **Writer metrics**: Operations, latency, throughput
- **Reader metrics**: Per-reader and aggregate success/error counts, latency, throughput
- **Data consistency**: Expected vs actual node/edge counts
- **Quality metrics**: Success rate percentage

### Additional Metrics (Secondary Test)
- **Catch-up metrics**: Total catch-ups, failures, average/min/max time

## Future Enhancements

Potential additional tests:
- Concurrent writes from multiple writers (requires different pattern)
- Read-heavy vs write-heavy workloads
- Large dataset stress tests
- Failure injection and recovery
