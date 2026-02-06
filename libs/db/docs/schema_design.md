# Motlie Database Schema Design

**Status**: ✅ **CURRENT** (as of 2025-11-19)

This document describes the complete schema for all column families in the motlie graph database.

## Overview

Motlie is a temporal graph database built on RocksDB, using column families for efficient storage and querying of nodes, edges, and fragments with temporal validity tracking and optional edge weights.

## Column Families

### 1. Nodes Column Family

**Purpose:** Store node metadata and summaries.

**Key Layout:**
```
[node_id: 16 bytes]
```

**Value Layout (MessagePack + LZ4):**
```rust
NodeCfValue(
    Option<ActivePeriod>,  // Temporal validity
    NodeName,                     // String name
    NodeSummary,                  // DataUrl (markdown content)
)
```

**Key Encoding:** Raw 16-byte UUID
**Value Encoding:** MessagePack serialization + LZ4 compression

**Query Patterns:**
- Point lookup by node ID
- No prefix scanning needed

---

### 2. ForwardEdges Column Family (Enhanced with Weight + Summary)

**Purpose:** Primary storage for edges with full data (topology, summary, weight).

**Key Layout:**
```
[src_id: 16 bytes] + [dst_id: 16 bytes] + [edge_name: UTF-8 variable length]
```

**Value Layout (MessagePack + LZ4):**
```rust
ForwardEdgeCfValue(
    Option<ActivePeriod>,  // Field 0: Temporal validity
    Option<f64>,                  // Field 1: Optional weight
    EdgeSummary,                  // Field 2: Edge summary (DataUrl)
)
```

**Field Ordering Rationale:**
1. `Option<ActivePeriod>` - Small, frequently accessed for temporal filtering
2. `Option<f64>` - Small (9 bytes), frequently accessed by graph algorithms
3. `EdgeSummary` - Large (100-1000 bytes), accessed when edge details needed

This ordering optimizes deserialization: algorithms can read temporal range + weight without deserializing the large EdgeSummary if not needed.

**Key Encoding:** Raw bytes (no MessagePack)
**Value Encoding:** MessagePack serialization + LZ4 compression
**Prefix Extraction:** 16-byte fixed prefix (source node ID)

**Query Patterns:**
- Prefix scan by source node ID → all outgoing edges
- Point lookup by (src_id, dst_id, name) → specific edge with summary and weight
- Lexicographic ordering: (src, dst, name)

**Storage Cost:** 110-1026 bytes per edge (9-byte overhead for weight)

---

### 3. ReverseEdges Column Family (Simplified Index)

**Purpose:** Index for efficient incoming edge queries (edges TO a node).

**Key Layout:**
```
[dst_id: 16 bytes] + [src_id: 16 bytes] + [edge_name: UTF-8 variable length]
```

**Value Layout (MessagePack + LZ4):**
```rust
ReverseEdgeCfValue(
    Option<ActivePeriod>,  // Temporal validity only
)
```

**Rationale:** Pure index for incoming edge queries - no data duplication needed. Edge data is retrieved from ForwardEdges when needed.

**Key Encoding:** Raw bytes (no MessagePack)
**Value Encoding:** MessagePack serialization + LZ4 compression
**Prefix Extraction:** 16-byte fixed prefix (destination node ID)

**Query Patterns:**
- Prefix scan by destination node ID → all incoming edges
- Point lookup by (dst_id, src_id, name) → verify edge existence
- Lexicographic ordering: (dst, src, name)

**Storage Cost:** 1-17 bytes per edge

---

### 4. NodeFragments Column Family

**Purpose:** Store versioned fragments (document chunks) associated with nodes.

**Key Layout:**
```
[node_id: 16 bytes] + [timestamp: 8 bytes big-endian]
```

**Value Layout (MessagePack + LZ4):**
```rust
NodeFragmentCfValue(
    Option<ActivePeriod>,  // Temporal validity
    FragmentContent,              // DataUrl (markdown/text/etc)
)
```

**Key Encoding:** Raw bytes (no MessagePack)
**Value Encoding:** MessagePack serialization + LZ4 compression
**Prefix Extraction:** 16-byte fixed prefix (node ID)

**Query Patterns:**
- Prefix scan by node ID → all fragments for a node (time-ordered)
- Range scan by node ID + time range → fragments in time window
- Lexicographic ordering: (node_id, timestamp)

---

### 5. EdgeFragments Column Family

**Purpose:** Store versioned fragments (document chunks) associated with edges.

**Key Layout:**
```
[src_id: 16 bytes] + [dst_id: 16 bytes] + [edge_name: UTF-8 variable] + [timestamp: 8 bytes big-endian]
```

**Value Layout (MessagePack + LZ4):**
```rust
EdgeFragmentCfValue(
    Option<ActivePeriod>,  // Temporal validity
    FragmentContent,              // DataUrl (markdown/text/etc)
)
```

**Key Encoding:** Raw bytes (no MessagePack)
**Value Encoding:** MessagePack serialization + LZ4 compression
**Prefix Extraction:** Variable-length prefix (src_id + dst_id + edge_name, minimum 32 bytes)

**Query Patterns:**
- Prefix scan by (src_id, dst_id, edge_name) → all fragments for an edge (time-ordered)
- Range scan by edge topology + time range → fragments in time window
- Lexicographic ordering: (src_id, dst_id, edge_name, timestamp)

**Variable-Length Key:** The prefix (src_id + dst_id + edge_name) is variable length, followed by fixed 8-byte timestamp.

---

## Edge Identity Model

### Key Change: Edges Identified by Topology

**Before Migration:**
- Edges had unique UUIDs (edge_id)
- Required separate "Edges" CF for full edge data
- Edge lookups required 2 CF reads (index → Edges CF)

**After Migration:**
- Edges identified by topology: (src_id, dst_id, edge_name)
- ForwardEdges CF contains all edge data
- Edge lookups require 1 CF read

**Benefits:**
- **33% storage reduction** (2 CFs per edge vs 3)
- **2x faster EdgeSummaryBySrcDstName** (1 lookup vs 2)
- **50% faster writes** (2 CF writes vs 3)
- Simpler API: edges identified by meaningful topology

**Trade-offs:**
- Cannot have multiple edges with same (src, dst, name) tuple
- Edge updates require topology lookup, not UUID
- This is acceptable for graph workloads where edge identity is topological

---

## Temporal Validity

All column families support optional temporal validity ranges:

```rust
pub struct ActivePeriod(
    pub Option<TimestampMilli>,  // valid_since (None = beginning of time)
    pub Option<TimestampMilli>,  // valid_until (None = end of time)
);
```

**Temporal Filtering Rules:**
- Record is valid at time T if: `valid_since <= T < valid_until`
- `None` for `valid_since` means valid from beginning of time
- `None` for `valid_until` means valid until end of time
- Records without temporal range (`None`) are always valid

**Query Behavior:**
- All queries accept optional `reference_ts_millis` parameter
- If `None`, defaults to current time
- Records are filtered by temporal validity during query execution

---

## Storage Optimizations

### MessagePack + LZ4 Compression

All values use MessagePack serialization followed by LZ4 compression:

1. **Serialize** to MessagePack binary format
2. **Compress** with LZ4 (fast compression/decompression)
3. **Store** compressed bytes in RocksDB

**Benefits:**
- Compact binary format (~50% smaller than JSON)
- Fast compression/decompression (LZ4)
- Schema evolution support (MessagePack)
- Type safety (Rust serde)

### Raw Key Encoding

All keys use raw byte encoding (no MessagePack):

**Fixed-length fields:**
- UUIDs: 16 bytes raw
- Timestamps: 8 bytes big-endian

**Variable-length fields:**
- Strings: UTF-8 bytes (no length prefix)

**Benefits:**
- Lexicographic ordering preserved
- Efficient prefix scanning
- Compact representation

### Field Ordering in Values

Value fields are ordered by access pattern:
1. **Frequently accessed small fields first** (temporal range, weight)
2. **Rarely accessed large fields last** (summaries, content)

**Benefits:**
- Can deserialize only needed fields
- Reduces deserialization overhead for queries that only need metadata

---

## Query Patterns

### Point Lookups

**Node by ID:**
```rust
NodeById::new(node_id, None).run(&reader, timeout).await
// Returns: (NodeName, NodeSummary)
```

**Edge by Topology:**
```rust
EdgeSummaryBySrcDstName::new(src_id, dst_id, edge_name, None).run(&reader, timeout).await
// Returns: (EdgeSummary, Option<f64>)
```

### Prefix Scans

**Outgoing Edges:**
```rust
OutgoingEdges::new(node_id, None).run(&reader, timeout).await
// Returns: Vec<(Option<f64>, SrcId, DstId, EdgeName)>
```

**Incoming Edges:**
```rust
IncomingEdges::new(node_id, None).run(&reader, timeout).await
// Returns: Vec<(Option<f64>, DstId, SrcId, EdgeName)>
```

**Node Fragments:**
```rust
NodeFragmentsByIdTimeRange::new(node_id, time_range, None).run(&reader, timeout).await
// Returns: Vec<(TimestampMilli, FragmentContent)>
```

> **Note:** For searching nodes and edges by name, use the fulltext search module which provides
> efficient text-based lookups via Tantivy indexing.

---

## Mutation Operations

### Creating Entities

**Add Node:**
```rust
AddNode {
    id: Id::new(),
    ts_millis: TimestampMilli::now(),
    name: "node_name".to_string(),
    valid_range: None,
}.run(&writer).await
```

**Add Edge:**
```rust
AddEdge {
    source_node_id,
    target_node_id,
    ts_millis: TimestampMilli::now(),
    name: "edge_name".to_string(),
    summary: EdgeSummary::from_text("Edge description"),
    weight: Some(1.5),
    valid_range: None,
}.run(&writer).await
```

**Add Node Fragment:**
```rust
AddNodeFragment {
    id: node_id,
    ts_millis: TimestampMilli::now(),
    content: DataUrl::from_markdown("Fragment content"),
    valid_range: None,
}.run(&writer).await
```

**Add Edge Fragment:**
```rust
AddEdgeFragment {
    src_id,
    dst_id,
    edge_name: "edge_name".to_string(),
    ts_millis: TimestampMilli::now(),
    content: DataUrl::from_markdown("Fragment content"),
    valid_range: None,
}.run(&writer).await
```

### Updating Entities

**Update Edge Weight:**
```rust
UpdateEdgeWeight {
    src_id,
    dst_id,
    name: "edge_name".to_string(),
    weight: 2.5,
}.run(&writer).await
```

**Update Edge Temporal Range:**
```rust
UpdateEdgeValidSinceUntil {
    src_id,
    dst_id,
    name: "edge_name".to_string(),
    temporal_range: ActivePeriod(Some(start), Some(end)),
    reason: "Invalidating old edge".to_string(),
}.run(&writer).await
```

**Update Node Temporal Range:**
```rust
UpdateNodeValidSinceUntil {
    id: node_id,
    temporal_range: ActivePeriod(Some(start), Some(end)),
    reason: "Invalidating old node".to_string(),
}.run(&writer).await
```

---

## Performance Characteristics

### Storage Costs (Per Edge)

**ForwardEdges:**
- Temporal range: 1-17 bytes
- Weight: 9 bytes
- Summary: 100-1000 bytes
- **Total:** 110-1026 bytes

**ReverseEdges:**
- Temporal range: 1-17 bytes
- **Total:** 1-17 bytes

**Combined:** 111-1043 bytes per edge

### Write Performance

**Edge Creation:**
- 2 CF writes (ForwardEdges + ReverseEdges)
- 50% faster than old 3-CF approach

**Edge Weight Update:**
- 1 CF write (ForwardEdges only)
- No need to update ReverseEdges (index only)

### Read Performance

**EdgeSummaryBySrcDstName:**
- 1 CF read (ForwardEdges)
- 2x faster than old 2-CF approach (ForwardEdges → Edges)

**OutgoingEdges:**
- Prefix scan of ForwardEdges
- Can skip deserializing weight and summary for topology-only queries

---

## Migration from Previous Schema

### Removed

- ❌ **Edges CF** - Edge data now in ForwardEdges
- ❌ **edge_id field** - Edges identified by topology
- ❌ **EdgeById query** - Use EdgeSummaryBySrcDstName instead

### Added

- ✅ **EdgeFragments CF** - Separate fragments for edges
- ✅ **Edge weights** - Optional f64 weight per edge
- ✅ **Edge summary in ForwardEdges** - Moved from Edges CF
- ✅ **UpdateEdgeWeight mutation** - Update edge weights

### Changed

- ⚠️ **Fragments CF** → **NodeFragments CF** - Renamed for clarity
- ⚠️ **AddEdge** - Now includes summary and weight fields
- ⚠️ **UpdateEdgeValidSinceUntil** - Uses topology instead of edge_id
- ⚠️ **EdgeSummaryBySrcDstName** - Returns (summary, weight) instead of (id, summary)

### Benefits

- **33% storage reduction** (2 CFs per edge vs 3)
- **2x faster edge queries** (1 lookup vs 2)
- **50% faster edge writes** (2 CF writes vs 3)
- **Simpler API** - Edges identified by meaningful topology
- **Support for weighted graphs** - Optional edge weights

---

## Column Family Summary Table

| CF Name | Key Structure | Value Structure | Purpose | Prefix Scannable |
|---------|---------------|-----------------|---------|------------------|
| `nodes` | node_id (16) | temporal, name, summary | Node storage | No |
| `node_fragments` | node_id (16) + ts (8) | temporal, content | Node fragments | Yes (by node_id) |
| `edge_fragments` | src (16) + dst (16) + name (var) + ts (8) | temporal, content | Edge fragments | Yes (by edge topology) |
| `forward_edges` | src (16) + dst (16) + name (var) | temporal, weight, summary | Primary edge storage | Yes (by src_id) |
| `reverse_edges` | dst (16) + src (16) + name (var) | temporal | Incoming edge index | Yes (by dst_id) |

**Total:** 5 column families

> **Note:** Name-based lookups (previously provided by `node_names` and `edge_names` column families)
> are now handled by the fulltext search module using Tantivy indexing for more flexible text search.

---

## See Also

- [Query API Guide](query-api-guide.md) - Detailed query patterns and usage
- [Mutation API Guide](mutation-api-guide.md) - Detailed mutation patterns and usage
- [Concurrency and Storage Modes](concurrency-and-storage-modes.md) - Read/write modes and transactions
