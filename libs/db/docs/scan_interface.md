# Scan API Design

## Overview

This document describes the design for a synchronous scan API that allows external consumers (like CLI tools) to iterate over column families with pagination support, without exposing internal schema types.

## Problem Statement

The async query API is channel-based, designed for concurrent query processing. For CLI utilities that need simple, synchronous iteration over column families:

1. The async machinery is overkill
2. Internal schema types (`NodeCfKey`, `NodeCfValue`, etc.) are `pub(crate)` and shouldn't be exposed
3. We need pagination support (`--last`, `--limit`) for large datasets

## Design Goals

1. **Minimal public API surface**: Don't expose internal CF key/value types
2. **Visitor pattern**: Let consumers define what to do with each record
3. **Pagination**: Support cursor-based pagination with `last` and `limit`
4. **Reusable logic**: Share pagination/iteration logic across all scan types
5. **Type safety**: Visitors receive strongly-typed, user-friendly record types

## Implementation

### Record Types

User-facing record types that hide internal schema details:

| Record Type | Fields | Description |
|-------------|--------|-------------|
| `NodeRecord` | `id`, `name`, `summary`, `valid_range` | Node metadata |
| `EdgeRecord` | `src_id`, `dst_id`, `name`, `summary`, `weight`, `valid_range` | Forward edge (outgoing) |
| `ReverseEdgeRecord` | `dst_id`, `src_id`, `name`, `valid_range` | Reverse edge index (incoming) |
| `NodeFragmentRecord` | `node_id`, `timestamp`, `content`, `valid_range` | Node content fragment |
| `EdgeFragmentRecord` | `src_id`, `dst_id`, `edge_name`, `timestamp`, `content`, `valid_range` | Edge content fragment |

### Visitor Trait

```rust
/// Visitor trait for processing scanned records.
/// Return `true` to continue scanning, `false` to stop early.
pub trait Visitor<R> {
    fn visit(&mut self, record: &R) -> bool;
}

// Blanket impl for closures
impl<R, F> Visitor<R> for F
where
    F: FnMut(&R) -> bool,
{
    fn visit(&mut self, record: &R) -> bool {
        self(record)
    }
}
```

### Scan Types

Each column family has a corresponding scan type with pagination parameters:

| Scan Type | Cursor Type | Description |
|-----------|-------------|-------------|
| `AllNodes` | `Option<Id>` | Scan all nodes |
| `AllEdges` | `Option<(SrcId, DstId, EdgeName)>` | Scan forward edges |
| `AllReverseEdges` | `Option<(DstId, SrcId, EdgeName)>` | Scan reverse edges |
| `AllNodeFragments` | `Option<(Id, TimestampMilli)>` | Scan node fragments |
| `AllEdgeFragments` | `Option<(SrcId, DstId, EdgeName, TimestampMilli)>` | Scan edge fragments |

All scan types support the following fields:
- `last`: Optional cursor for pagination (type depends on scan type)
- `limit`: Maximum number of records to return
- `reverse`: Scan in reverse direction (from end to start)
- `reference_ts_millis`: Optional reference timestamp for temporal validity filtering (if Some, only records valid at this time are returned)

### Visitable Trait

```rust
pub trait Visitable {
    type Record;

    fn accept<V: Visitor<Self::Record>>(
        &self,
        storage: &Storage,
        visitor: &mut V,
    ) -> Result<usize>;
}
```

## Usage Examples

### Basic Scan

```rust
use motlie_db::{Storage, scan::{AllNodes, NodeRecord, Visitable}};

let mut storage = Storage::readonly(db_path);
storage.ready()?;

let scan = AllNodes { last: None, limit: 100, reverse: false };
scan.accept(&storage, &mut |record: &NodeRecord| {
    println!("{}\t{}", record.id, record.name);
    true // continue scanning
})?;
```

### Pagination

```rust
// First page
let scan = AllNodes { last: None, limit: 10, ..Default::default() };
let mut last_id = None;
scan.accept(&storage, &mut |record: &NodeRecord| {
    println!("{}", record.id);
    last_id = Some(record.id);
    true
})?;

// Next page (using last_id as cursor)
let scan = AllNodes { last: last_id, limit: 10, ..Default::default() };
scan.accept(&storage, &mut |record: &NodeRecord| {
    println!("{}", record.id);
    true
})?;
```

### Reverse Direction

```rust
// Scan nodes from end to start
let scan = AllNodes { last: None, limit: 100, reverse: true };
scan.accept(&storage, &mut |record: &NodeRecord| {
    println!("{}\t{}", record.id, record.name);
    true
})?;

// Reverse pagination (from last page to first)
let scan = AllNodes { last: Some(cursor_id), limit: 10, reverse: true };
scan.accept(&storage, &mut |record: &NodeRecord| {
    println!("{}", record.id);
    true
})?;
```

### Temporal Filtering

```rust
use motlie_db::TimestampMilli;

// Only return records valid at a specific point in time
let reference_time = TimestampMilli(1704067200000); // 2024-01-01 00:00:00 UTC
let scan = AllNodes {
    last: None,
    limit: 100,
    reverse: false,
    reference_ts_millis: Some(reference_time),
};
scan.accept(&storage, &mut |record: &NodeRecord| {
    // Only records with valid_range containing reference_time are returned
    println!("{}\t{}", record.id, record.name);
    true
})?;
```

### Early Termination

```rust
let scan = AllNodes { last: None, limit: 1000, ..Default::default() };
let mut count = 0;
scan.accept(&storage, &mut |_record: &NodeRecord| {
    count += 1;
    count < 5 // Stop after 5 records
})?;
```

### Scanning Edges

```rust
use motlie_db::scan::{AllEdges, AllReverseEdges, EdgeRecord, ReverseEdgeRecord};

// Scan outgoing edges (forward edges)
let scan = AllEdges { last: None, limit: 100, ..Default::default() };
scan.accept(&storage, &mut |record: &EdgeRecord| {
    println!("{} -> {} via {}", record.src_id, record.dst_id, record.name);
    true
})?;

// Scan incoming edges (reverse edges)
let scan = AllReverseEdges { last: None, limit: 100, ..Default::default() };
scan.accept(&storage, &mut |record: &ReverseEdgeRecord| {
    println!("{} <- {} via {}", record.dst_id, record.src_id, record.name);
    true
})?;
```

## CLI Usage

The `motlie` CLI tool uses the scan API:

```bash
# List available column families
motlie db -p /path/to/db list

# Dump nodes (default: TSV format)
motlie db -p /path/to/db dump nodes

# Dump with pagination
motlie db -p /path/to/db dump nodes --limit 10
motlie db -p /path/to/db dump nodes --limit 10 --last <last_id>

# Table format with aligned columns
motlie db -p /path/to/db dump nodes -f table

# Scan in reverse direction (from end to start)
motlie db -p /path/to/db dump nodes --reverse
motlie db -p /path/to/db dump nodes -r  # short flag

# Temporal validity filtering (only return records valid at the specified time)
motlie db -p /path/to/db dump nodes 2024-01-01              # Date only (midnight)
motlie db -p /path/to/db dump nodes 2024-01-01-12:30:45     # Date and time

# Scan edges
motlie db -p /path/to/db dump outgoing-edges
motlie db -p /path/to/db dump incoming-edges

# Scan fragments (shows content preview for text types)
motlie db -p /path/to/db dump node-fragments -f table

# Combine options
motlie db -p /path/to/db dump nodes 2024-06-15 --limit 10 --reverse -f table
```

### Output Formats

- **`tsv`** (default): Tab-separated values, suitable for piping
- **`table`**: Aligned columns with headers, suitable for human reading

### Scan Direction

- **Forward (default)**: Scans from the start of the column family
- **Reverse (`--reverse` or `-r`)**: Scans from the end of the column family

### Temporal Filtering

When a datetime argument is provided after the column family name, only records that are temporally valid at that time are returned:

- **Format**: `YYYY-MM-DD` or `YYYY-MM-DD-HH:mm:ss`
- Records with `valid_range = None` are always considered valid
- Records with a `valid_range` are checked: `start <= reference_time < until`

## API Surface

### Public Types (from `motlie_db::scan`)

**Record Types:**
- `NodeRecord`
- `EdgeRecord`
- `ReverseEdgeRecord`
- `NodeFragmentRecord`
- `EdgeFragmentRecord`

**Scan Types:**
- `AllNodes`
- `AllEdges`
- `AllReverseEdges`
- `AllNodeFragments`
- `AllEdgeFragments`

**Traits:**
- `Visitor<R>`
- `Visitable`

**Re-exports:**
- `ActivePeriod`

### Internal Types (remain `pub(crate)`)

- `ColumnFamilyRecord` trait
- `Nodes`, `NodeCfKey`, `NodeCfValue`
- `ForwardEdges`, `ReverseEdges`, `EdgeFragments`, etc.
- All other schema types

## Design Decisions

### Why Visitor Pattern?

The visitor pattern was chosen over an iterator pattern because:

1. **Simpler lifetime management**: No need to tie iterator lifetime to Storage
2. **Explicit control flow**: Return `bool` to stop is clear and simple
3. **Matches use case**: "Scan and process" is the primary pattern

### Why Not Expose Schema Types?

1. **Encapsulation**: Internal types may change without breaking external code
2. **Simplicity**: User-facing types have cleaner field names (`src_id` vs `ForwardEdgeCfKey.0`)
3. **Safety**: Prevents accidental misuse of internal serialization formats
