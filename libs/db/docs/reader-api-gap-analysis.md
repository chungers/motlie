# Reader API Gap Analysis for Store Verification

## Executive Summary

The current `reader::Reader` API **cannot** fully replace direct RocksDB access in `examples/store/main.rs --verify` mode. The verification code requires **bulk iteration** capabilities that are fundamentally different from the point-query pattern the Reader API was designed for.

## Current Reader API Capabilities

The `reader::Reader` provides the following query methods (all point queries by ID):

1. ✅ `node_by_id(id, timeout)` → `(NodeName, NodeSummary)`
2. ✅ `edge_summary_by_id(id, timeout)` → `EdgeSummary`
3. ✅ `edge_by_src_dst_name(src_id, dst_id, name, timeout)` → `(Id, EdgeSummary)`
4. ✅ `fragments_by_id(id, timeout)` → `Vec<(TimestampMilli, FragmentContent)>`
5. ✅ `edges_from_node_by_id(id, timeout)` → `Vec<(SrcId, EdgeName, DstId)>`
6. ✅ `edges_to_node_by_id(id, timeout)` → `Vec<(DstId, EdgeName, SrcId)>`

**Pattern:** All methods require an `Id` as input parameter.

## Verification Requirements

The verification code in `examples/store/main.rs` performs these operations:

### 1. Verify Nodes (lines 415-462)
**Operation:** Iterate over ALL nodes to:
- Count total nodes in database
- Extract all node names into a set
- Compare against expected nodes from CSV

**Current Implementation:**
```rust
let cf = db.cf_handle("nodes").context("Nodes CF not found")?;
let iter = db.iterator_cf(cf, rocksdb::IteratorMode::Start);
for item in iter {
    let (_key, value) = item.context("Failed to read from nodes CF")?;
    db_node_count += 1;
    if let Ok((node_name, _markdown_content)) = deserialize_node_value(&value) {
        db_node_names.insert(node_name);
    }
}
```

**API Gap:** ❌ No way to iterate over all nodes without knowing their IDs in advance.

### 2. Verify Edges (lines 464-538)
**Operation:** Two-phase iteration:
- **Phase 1:** Iterate over ALL nodes to build `id_to_name` map
- **Phase 2:** Iterate over ALL forward_edges to:
  - Count total edges
  - Deserialize edge keys to extract (source_id, target_id, edge_name)
  - Map IDs to names using `id_to_name`
  - Build set of edges as (source_name, target_name, edge_name)
  - Compare against expected edges

**Current Implementation:**
```rust
// Phase 1: Build ID to name mapping
for item in db.iterator_cf(nodes_cf, rocksdb::IteratorMode::Start) {
    let (key, value) = item.context("Failed to read from nodes CF")?;
    let node_id = deserialize_node_id(&key)?;
    if let Ok((node_name, _markdown_content)) = deserialize_node_value(&value) {
        id_to_name.insert(node_id, node_name);
    }
}

// Phase 2: Iterate all edges
for item in db.iterator_cf(forward_edges_cf, rocksdb::IteratorMode::Start) {
    let (key, _value) = item.context("Failed to read from forward_edges CF")?;
    db_edge_count += 1;
    let (source_id, target_id, edge_name) = deserialize_forward_edge_key(&key)?;
    // Map to names and verify...
}
```

**API Gap:** ❌ No way to iterate over all edges without knowing IDs. Even if we had node IDs, we'd need to query edges for each node individually (inefficient N+1 pattern).

### 3. Verify Fragments (lines 540-594)
**Operation:** Iterate over ALL fragments to:
- Count total fragments
- Extract fragment content
- Compare against expected fragments

**Current Implementation:**
```rust
let iter = db.iterator_cf(cf, rocksdb::IteratorMode::Start);
for item in iter {
    let (_, value) = item.context("Failed to read from fragments CF")?;
    db_fragment_count += 1;
    if let Ok(content) = deserialize_fragment_value(&value) {
        db_fragments_content.insert(content);
    }
}
```

**API Gap:** ❌ No way to iterate over all fragments without knowing their IDs.

## Missing API Capabilities

To fully replace RocksDB direct access in verification, the Reader API would need:

### 1. Bulk Iteration APIs

#### A. `list_all_nodes(timeout) → Vec<(Id, NodeName, NodeSummary)>`
- Returns all nodes in the database
- Use case: Verification, admin tools, debugging
- Implementation: Scan the nodes CF and return all key-value pairs

#### B. `list_all_edges(timeout) → Vec<(Id, SrcId, EdgeName, DstId, EdgeSummary)>`
- Returns all edges in the database
- Use case: Verification, admin tools, graph analysis
- Implementation: Scan the edges CF and/or forward_edges CF

#### C. `list_all_fragments(timeout) → Vec<(Id, Vec<(TimestampMilli, FragmentContent)>)>`
- Returns all fragments grouped by ID
- Use case: Verification, content export
- Implementation: Scan the fragments CF

### 2. Pagination/Streaming APIs

For large databases, full scans might not be feasible. Consider:

#### A. `stream_nodes(offset, limit, timeout) → Vec<(Id, NodeName, NodeSummary)>`
- Returns paginated node results
- Allows processing large datasets in chunks

#### B. `stream_edges(offset, limit, timeout) → Vec<(Id, SrcId, EdgeName, DstId, EdgeSummary)>`
- Returns paginated edge results

#### C. `stream_fragments(offset, limit, timeout) → Vec<(Id, Vec<(TimestampMilli, FragmentContent)>)>`
- Returns paginated fragment results

### 3. Count/Statistics APIs

For quick verification without full iteration:

#### A. `count_nodes(timeout) → usize`
- Returns total count of nodes
- Faster than full iteration if only count is needed

#### B. `count_edges(timeout) → usize`
- Returns total count of edges

#### C. `count_fragments(timeout) → usize`
- Returns total count of fragments

### 4. Lookup by Name APIs

Since the verification works with names (not IDs), add:

#### A. `node_by_name(name: String, timeout) → Option<(Id, NodeSummary)>`
- Find node by name instead of ID
- Requires maintaining a name→ID index or scanning

#### B. `edge_by_names(src_name: String, dst_name: String, edge_name: String, timeout) → Option<(Id, EdgeSummary)>`
- Find edge by source/dest/edge names instead of IDs
- Requires name resolution first

## Architectural Considerations

### Design Philosophy Conflict

The current Reader API is designed around:
- **Point queries by ID**: Fast, efficient, targeted access
- **Bounded query time**: All queries have explicit timeouts
- **Single-entity focus**: Query one thing at a time

Verification requires:
- **Bulk operations**: Scan entire column families
- **Potentially unbounded**: Full scans can't have tight timeouts
- **Multi-entity aggregation**: Need to join data across CFs

### Alternative Approaches

#### Option 1: Add Scan APIs to Reader (Recommended with caveats)
**Pros:**
- Consistent API surface
- Controlled access patterns
- Can still use readonly RocksDB mode

**Cons:**
- Breaks the current design philosophy
- Need to handle large result sets
- Timeout semantics unclear for bulk operations

**Implementation:**
```rust
impl Reader {
    pub async fn scan_all_nodes(&self, timeout: Duration)
        -> Result<Vec<(Id, NodeName, NodeSummary)>> {
        // Implementation details...
    }
}
```

#### Option 2: Create Separate Admin/Verification API
**Pros:**
- Keeps Reader focused on point queries
- Clear separation of concerns
- Can optimize bulk operations differently

**Cons:**
- Two different APIs to learn/maintain
- Still needs access to database

**Implementation:**
```rust
pub struct DatabaseScanner {
    storage: Storage, // readonly mode
}

impl DatabaseScanner {
    pub fn scan_all_nodes(&self) -> Result<Vec<(Id, NodeName, NodeSummary)>> {
        // Direct RocksDB iteration but encapsulated
    }
}
```

#### Option 3: Keep Direct RocksDB for Verification (Current approach)
**Pros:**
- Simple, direct, efficient
- Verification is an administrative task, not normal operation
- No need to design/maintain bulk APIs

**Cons:**
- Bypasses abstraction layer
- Direct coupling to RocksDB
- Example code needs to understand schema internals

## Recommendation

### For Now: Keep Direct RocksDB Access in Verification

**Rationale:**
1. Verification is an **administrative/testing operation**, not a runtime query pattern
2. The Reader API is designed for **application queries**, not database introspection
3. Adding bulk scan APIs would complicate the Reader design without clear runtime benefit
4. The verification code is in `examples/`, not production code

### For Future: Consider DatabaseScanner/Inspector API

If bulk operations become a common need (beyond just verification), create a separate API:

```rust
// In libs/db/src/inspector.rs
pub struct DatabaseInspector {
    storage: Storage,
}

impl DatabaseInspector {
    pub fn new(db_path: impl AsRef<Path>) -> Result<Self> {
        // Open in readonly mode
    }

    pub fn list_all_nodes(&self) -> Result<Vec<(Id, NodeName, NodeSummary)>> {
        // Scan nodes CF
    }

    pub fn list_all_edges(&self) -> Result<Vec<EdgeInfo>> {
        // Scan edges/forward_edges CFs
    }

    pub fn stats(&self) -> DatabaseStats {
        // Count nodes, edges, fragments
    }
}
```

This would:
- Keep Reader focused on point queries
- Provide encapsulated bulk operations
- Support verification, debugging, admin tools
- Still hide RocksDB implementation details

## Summary Table

| Verification Need | Current Reader API | Gap? | Workaround |
|-------------------|-------------------|------|------------|
| Count all nodes | ❌ None | YES | Direct RocksDB iteration |
| List all node names | ❌ None | YES | Direct RocksDB iteration |
| Count all edges | ❌ None | YES | Direct RocksDB iteration |
| List all edges with names | ❌ None | YES | Direct RocksDB iteration + deserialization |
| Count all fragments | ❌ None | YES | Direct RocksDB iteration |
| List all fragment content | ❌ None | YES | Direct RocksDB iteration |
| Verify specific node exists | ✅ `node_by_id()` if ID known | PARTIAL | Need ID first, can't search by name |
| Verify specific edge exists | ✅ `edge_by_src_dst_name()` if IDs known | PARTIAL | Need source/dest IDs |

## Conclusion

**The Reader API cannot currently replace direct RocksDB access in verification mode.** The fundamental issue is that verification requires **bulk iteration** over all entities, while Reader provides **point queries by ID**.

**Recommendation:** Keep the current approach (direct RocksDB access in verification) as it's appropriate for an administrative/testing tool. If bulk operations become a runtime requirement, design a separate `DatabaseInspector` API rather than complicating the Reader.
