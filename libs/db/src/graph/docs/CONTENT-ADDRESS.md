# CONTENT-ADDRESS: Reverse Index with Versioning, Optimistic Locking, and GC

## Problem

1. **No reverse index**: `SummaryHash` from vector search cannot be resolved to graph entities without full scan (codex, 2026-02-02, validated)
2. **No optimistic locking**: Blind upserts can silently lose concurrent updates (codex, 2026-02-02, validated)
3. **No GC for stale content**: Old summaries accumulate without cleanup (codex, 2026-02-02, validated)

**Breaking change:** This work changes on-disk schema; no migration or deprecation is required for this project phase. (codex, 2026-02-03, validated)

## Core Goal

**Reverse index:** `SummaryHash → Vec<(EntityKey, Version)>`

Given a content hash from vector search, resolve to all entity keys that reference it.

---

# Part 1: Schema Changes

## 1.0 Version Type

```rust
/// Entity version number for optimistic locking.
/// u32 provides 4 billion versions per entity - sufficient for 136 years at 1 update/sec.
pub type Version = u32;

pub const VERSION_MAX: Version = u32::MAX;
```

| Type | Max Value | Overflow at 1 update/sec | Size Savings vs u64 |
|------|-----------|--------------------------|---------------------|
| `u32` | 4.2 billion | 136 years | 4 bytes/entry |

**Overflow Policy:**
- If `version == VERSION_MAX`, reject further updates with `Error::VersionOverflow`
- This is extremely unlikely (136 years at 1 update/sec per entity)
- If encountered, options: (1) delete and recreate entity, (2) upgrade to u64 in future schema version
(codex, 2026-02-02, validated) (claude, 2026-02-02, implemented in schema.rs)

## 1.1 Entity Column Families (HOT)

### Nodes

```rust
pub struct NodeCfKey(pub Id);  // 16 bytes

pub struct NodeCfValue(
    pub Option<TemporalRange>,
    pub NameHash,
    pub Option<SummaryHash>,  // Content hash for vector search matching
    pub Version,               // Version (monotonic, starts at 1) [NEW]
    pub bool,                  // deleted flag (tombstone) [NEW]
);
```
(codex, 2026-02-02, validated) (claude, 2026-02-02, implemented in schema.rs)

### ForwardEdges

```rust
pub struct ForwardEdgeCfKey(
    pub SrcId,      // 16 bytes
    pub DstId,      // 16 bytes
    pub NameHash,   // 8 bytes
);  // Total: 40 bytes

pub struct ForwardEdgeCfValue(
    pub Option<TemporalRange>,
    pub Option<f64>,           // weight
    pub Option<SummaryHash>,   // Content hash
    pub Version,               // Version [NEW]
    pub bool,                  // deleted flag (tombstone) [NEW]
);
```
(codex, 2026-02-02, validated) (claude, 2026-02-02, implemented in schema.rs)

**Tombstone Semantics:**
- `deleted = true`: Entity is logically deleted but retained for audit/time-travel
- Versioned summaries and index entries are preserved until GC
- `current_*_for_summary()` filters out deleted entities
- GC policy determines tombstone retention period
(codex, 2026-02-02, validated)

**Note:** Edge identity is `(src, dst, name)`. `name` is immutable; renames are modeled as delete+insert. (codex, 2026-02-02, validated)

### ReverseEdges

```rust
pub struct ReverseEdgeCfKey(pub DstId, pub SrcId, pub NameHash);  // 40 bytes

pub struct ReverseEdgeCfValue(pub Option<TemporalRange>);
// ReverseEdges is primarily an index for "find edges TO this node".
// TemporalRange is denormalized here for fast inbound scans with time filtering.
// Other edge details (weight, summary, version) remain authoritative in ForwardEdges.

**Consistency rule:** whenever an edge's temporal range is updated, update both ForwardEdges and ReverseEdges in the same transaction to keep this denormalized field in sync.
(codex, 2026-02-02, validated)
```

---

## 1.2 Content Column Families (COLD)

**Currently content-addressed (SummaryHash).** Versioned summaries are planned for GC support. (codex, 2026-02-03, validated)

### NodeSummaries

```rust
// CURRENT (implemented): content-addressed
pub struct NodeSummaryCfKey(pub SummaryHash);  // 8 bytes

/// Reference count for content-addressed summaries.
/// Use u32 to bound storage overhead while allowing billions of references.
pub type RefCount = u32;

// Value stores refcount + summary to enable safe GC of shared content
pub struct NodeSummaryCfValue(pub RefCount, pub NodeSummary); // (refcount, summary)
```
(codex, 2026-02-03, validated)

### EdgeSummaries

```rust
// CURRENT (implemented): content-addressed
pub struct EdgeSummaryCfKey(pub SummaryHash);  // 8 bytes

// Value stores refcount + summary to enable safe GC of shared content
pub struct EdgeSummaryCfValue(pub RefCount, pub EdgeSummary); // (refcount, summary)
```
(codex, 2026-02-03, validated)

---

## 1.3 Reverse Index Column Families [NEW]

### NodeSummaryIndex

```rust
pub struct NodeSummaryIndexCfKey(
    pub SummaryHash,  // 8 bytes - prefix for hash lookup
    pub Id,           // 16 bytes - node_id
    pub Version,      // 4 bytes - version
);  // Total: 28 bytes

/// 1-byte marker: 0x01 = current, 0x00 = stale
pub struct NodeSummaryIndexCfValue(pub u8);

impl NodeSummaryIndexCfValue {
    pub const CURRENT: u8 = 0x01;
    pub const STALE: u8 = 0x00;

    pub fn is_current(&self) -> bool { self.0 == Self::CURRENT }
}
```
(codex, 2026-02-02, validated) (claude, 2026-02-02, implemented in schema.rs)

### EdgeSummaryIndex

```rust
pub struct EdgeSummaryIndexCfKey(
    pub SummaryHash,  // 8 bytes - prefix for hash lookup
    pub SrcId,        // 16 bytes
    pub DstId,        // 16 bytes
    pub NameHash,     // 8 bytes
    pub Version,      // 4 bytes - version
);  // Total: 52 bytes

/// 1-byte marker: 0x01 = current, 0x00 = stale
pub struct EdgeSummaryIndexCfValue(pub u8);
```
(codex, 2026-02-02, validated) (claude, 2026-02-02, implemented in schema.rs)

**Marker Bit Semantics:**
- `0x01` (CURRENT): This (entity, version) is the current version
- `0x00` (STALE): Entity has been updated to a newer version, or deleted

This eliminates point-reads in `current_*_for_summary()` - just filter by marker during prefix scan.
(codex, 2026-02-02, validated)

**EntityKey (edge):** `(src_id, dst_id, name_hash)`. This matches the forward edge key. (codex, 2026-02-02, validated)

---

## 1.4 Fragment Column Families (Unchanged)

Fragments use timestamp for append-only semantics. No GC. (codex, 2026-02-02, validated)

```rust
pub struct NodeFragmentCfKey(pub Id, pub TimestampMilli);  // 24 bytes
pub struct NodeFragmentCfValue(pub Option<TemporalRange>, pub FragmentContent);

pub struct EdgeFragmentCfKey(pub SrcId, pub DstId, pub NameHash, pub TimestampMilli);  // 48 bytes
pub struct EdgeFragmentCfValue(pub Option<TemporalRange>, pub FragmentContent);
```

---

## 1.5 Column Family Registry

```rust
pub(crate) const ALL_COLUMN_FAMILIES: &[&str] = &[
    // Names (interning)
    "graph/names",

    // Entities (HOT)
    "graph/nodes",
    "graph/forward_edges",
    "graph/reverse_edges",

    // Content (COLD) - entity+version keyed
    "graph/node_summaries",
    "graph/edge_summaries",

    // Fragments - timestamp keyed
    "graph/node_fragments",
    "graph/edge_fragments",

    // Reverse indexes [NEW]
    "graph/node_summary_index",
    "graph/edge_summary_index",

    // Graph-level metadata (GC cursors, future config) [NEW]
    "graph/meta",
];
```
(codex, 2026-02-02, validated) (claude, 2026-02-02, partially implemented - index CFs added, meta CF pending)

---

# Part 2: Reverse Lookup API and Behavior

## 2.1 Critical Insight: Multi-Version Resolution

A `SummaryHash` lookup can return **multiple results**, including:

| Scenario | What's Returned |
|----------|-----------------|
| Different entities with same content | All entities that share the hash |
| Same entity at different versions | Multiple (entity, version) pairs |
| Old version after entity was updated | The old (entity, version) — now stale |
(codex, 2026-02-02, validated)

### Use Cases

| Use Case | Query Type |
|----------|------------|
| Vector search resolution | Current versions only |
| Time-travel query | Specific version |
| Audit/debugging | All versions |
| Content history | All versions of specific entity |
(codex, 2026-02-02, validated)

---

## 2.2 Node Reverse Lookup API
(claude, 2026-02-02, implemented in query.rs - NodesBySummaryHash, EdgesBySummaryHash)

### All Matches (Including Old Versions)

```rust
/// Returns ALL (node_id, version) pairs that have this hash.
/// Includes old versions that may no longer be current.
pub fn all_nodes_for_summary(&self, hash: SummaryHash) -> Result<Vec<(Id, Version)>>;
```

### Current Versions Only

```rust
/// Returns only node_ids where this hash is the CURRENT version.
/// Filters out old versions using marker bit (CURRENT vs STALE),
/// and optionally excludes tombstoned entities.
pub fn current_nodes_for_summary(&self, hash: SummaryHash) -> Result<Vec<Id>>;
```

### Versions of Specific Node

```rust
/// Returns all versions of a specific node that had this hash.
pub fn node_versions_for_summary(&self, hash: SummaryHash, node_id: Id) -> Result<Vec<Version>>;
```

### Get Summary by Version

```rust
/// Get summary for specific version, or current if version=None.
pub fn get_node_summary(&self, id: Id, version: Option<Version>) -> Result<Option<NodeSummary>>;
```
(codex, 2026-02-02, validated)

---

## 2.3 Edge Reverse Lookup API

### All Matches (Including Old Versions)

```rust
/// Returns ALL (edge_key, version) pairs that have this hash.
pub fn all_edges_for_summary(&self, hash: SummaryHash) -> Result<Vec<(ForwardEdgeCfKey, Version)>>;
```

### Current Versions Only

```rust
/// Returns only edge keys where this hash is the CURRENT version.
/// Filters out old versions using marker bit (CURRENT vs STALE),
/// and optionally excludes tombstoned entities.
pub fn current_edges_for_summary(&self, hash: SummaryHash) -> Result<Vec<ForwardEdgeCfKey>>;
```

### Versions of Specific Edge

```rust
/// Returns all versions of a specific edge that had this hash.
pub fn edge_versions_for_summary(
    &self,
    hash: SummaryHash,
    src: Id,
    dst: Id,
    name: &str,
) -> Result<Vec<Version>>;
```

### Get Summary by Version

```rust
/// Get edge summary for specific version, or current if version=None.
pub fn get_edge_summary(
    &self,
    src: Id,
    dst: Id,
    name: &str,
    version: Option<Version>,
) -> Result<Option<EdgeSummary>>;
```
(codex, 2026-02-02, validated)

---

## 2.4 Index Prefix Scan Capabilities

### NodeSummaryIndex

| Prefix | Bytes | Returns |
|--------|-------|---------|
| `(hash)` | 8 | All nodes with this hash (any version) |
| `(hash, node_id)` | 24 | All versions of specific node with this hash |

**Content-based search:** scanning by `(hash)` enables "find all entities with identical content" without any entity-specific filters. (codex, 2026-02-02, validated)

### EdgeSummaryIndex

| Prefix | Bytes | Returns |
|--------|-------|---------|
| `(hash)` | 8 | All edges with this hash (any version) |
| `(hash, src)` | 24 | All edges from `src` with this hash |
| `(hash, src, dst)` | 40 | All edges between `src→dst` with this hash |
| `(hash, src, dst, name)` | 48 | All versions of specific edge with this hash |
(codex, 2026-02-02, validated)

---

## 2.5 Node Example: Multi-Version Resolution

```
Timeline:
  t1: Insert Node A, summary="Person", hash=0xAAA, version=1
  t2: Insert Node B, summary="Person", hash=0xAAA, version=1
  t3: Update Node A, summary="Employee", hash=0xBBB, version=2
  t4: Insert Node C, summary="Person", hash=0xAAA, version=1
  t5: Update Node B, summary="Manager", hash=0xCCC, version=2
  t6: Update Node C, summary="Contractor", hash=0xDDD, version=2

Current State:
  Nodes CF:
    A → (hash=0xBBB, version=2)  // "Employee"
    B → (hash=0xCCC, version=2)  // "Manager"
    C → (hash=0xDDD, version=2)  // "Contractor"

  NodeSummaryIndex CF (entries for hash 0xAAA):
    (0xAAA, A, 1) → STALE   // A changed to 0xBBB at v2
    (0xAAA, B, 1) → STALE   // B changed to 0xCCC at v2
    (0xAAA, C, 1) → STALE   // C changed to 0xDDD at v2

Query Results for hash 0xAAA:
  all_nodes_for_summary(0xAAA):
    [(A, 1), (B, 1), (C, 1)]  // All three had it at v1

  current_nodes_for_summary(0xAAA):
    []  // Empty! None currently have 0xAAA
```
(codex, 2026-02-02, illustrative)

---

## 2.6 Edge Example: Multi-Version Resolution

```
Timeline:
  t1: Insert Edge (A→B, "knows"), summary="Friends", hash=0xAAA, version=1
  t2: Insert Edge (C→D, "knows"), summary="Friends", hash=0xAAA, version=1
  t3: Insert Edge (E→F, "works_with"), summary="Friends", hash=0xAAA, version=1
  t4: Update Edge (A→B, "knows"), summary="Close friends", hash=0xBBB, version=2
  t5: Update Edge (E→F, "works_with"), summary="Colleagues", hash=0xCCC, version=2

Current State:
  ForwardEdges CF:
    (A, B, "knows")      → (hash=0xBBB, version=2)  // Changed
    (C, D, "knows")      → (hash=0xAAA, version=1)  // Still "Friends"
    (E, F, "works_with") → (hash=0xCCC, version=2)  // Changed

  EdgeSummaryIndex CF (entries for hash 0xAAA):
    (0xAAA, A, B, "knows", 1)      → STALE   // Updated at v2
    (0xAAA, C, D, "knows", 1)      → CURRENT // Still current
    (0xAAA, E, F, "works_with", 1) → STALE   // Updated at v2

Query Results for hash 0xAAA:
  all_edges_for_summary(0xAAA):
    [
      ((A, B, "knows"), 1),        // Had it at v1, now stale
      ((C, D, "knows"), 1),        // Still current
      ((E, F, "works_with"), 1),   // Had it at v1, now stale
    ]

  current_edges_for_summary(0xAAA):
    [(C, D, "knows")]  // Only edge still at v1 with this hash
```
(codex, 2026-02-02, illustrative)

**Note:** It is expected that ANN search can return vectors for multiple versions of the same edge (e.g., 0xAAA for v1 and 0xBBB for v2). In that case, reverse lookup yields distinct versioned results for the same edge. The reranker should deduplicate to the latest version and rank accordingly. (codex, 2026-02-03, validated)

---

## 2.7 Result Types

```rust
/// Node lookup result with version info
pub struct NodeSummaryLookupResult {
    pub node_id: Id,
    pub version: Version,
    pub is_current: bool,  // version == entity's current version
}

/// Edge lookup result with version info
pub struct EdgeSummaryLookupResult {
    pub src: Id,
    pub dst: Id,
    pub name_hash: NameHash,
    pub version: Version,
    pub is_current: bool,
}
```

---

# Part 3: Implementation Tasks
(codex, 2026-02-03, partial) (claude, 2026-02-02, in-progress)

## 3.1 Write Path: Insert
(claude, 2026-02-02, implemented in mutation.rs - AddNode and AddEdge now write index entries)
(codex, 2026-02-03, validated)

### Insert Node (version = 1)

```rust
fn insert_node(&self, id: Id, name: &str, summary: NodeSummary) -> Result<()> {
    let txn = self.txn_db.transaction();

    // 1. Check doesn't exist
    if txn.get_cf(nodes_cf, id)?.is_some() {
        return Err(Error::AlreadyExists(id));
    }

    let version: Version = 1;
    let deleted = false;
    let name_hash = NameHash::from_name(name);
    let summary_hash = SummaryHash::from_summary(&summary)?;

    // 2. Write name interning
    txn.put_cf(names_cf, NameCfKey(name_hash), name)?;

    // 3. Write node (HOT) with version and deleted flag
    let node_value = NodeCfValue(None, name_hash, Some(summary_hash), version, deleted);
    txn.put_cf(nodes_cf, NodeCfKey(id), node_value)?;

    // 4. Write summary (COLD) - content-addressed
    let summary_key = NodeSummaryCfKey(summary_hash);
    txn.put_cf(node_summaries_cf, summary_key, summary)?;

    // 5. Write reverse index entry with CURRENT marker
    let index_key = NodeSummaryIndexCfKey(summary_hash, id, version);
    let index_value = NodeSummaryIndexCfValue(NodeSummaryIndexCfValue::CURRENT);
    txn.put_cf(node_summary_index_cf, index_key, index_value)?;

    txn.commit()
}
```

### Insert Edge (version = 1)

```rust
fn insert_edge(
    &self,
    src: Id,
    dst: Id,
    name: &str,
    summary: EdgeSummary,
    weight: Option<f64>,
) -> Result<()> {
    let txn = self.txn_db.transaction();

    let name_hash = NameHash::from_name(name);
    let edge_key = ForwardEdgeCfKey(src, dst, name_hash);

    // 1. Check doesn't exist
    if txn.get_cf(forward_edges_cf, &edge_key)?.is_some() {
        return Err(Error::AlreadyExists(edge_key));
    }

    let version: Version = 1;
    let deleted = false;
    let summary_hash = SummaryHash::from_summary(&summary)?;

    // 2. Write name interning
    txn.put_cf(names_cf, NameCfKey(name_hash), name)?;

    // 3. Write forward edge (HOT) with version and deleted flag
    let edge_value = ForwardEdgeCfValue(None, weight, Some(summary_hash), version, deleted);
    txn.put_cf(forward_edges_cf, edge_key, edge_value)?;

    // 4. Write reverse edge (HOT) - denormalized TemporalRange for inbound scans
    let reverse_key = ReverseEdgeCfKey(dst, src, name_hash);
    let reverse_value = ReverseEdgeCfValue(None);
    txn.put_cf(reverse_edges_cf, reverse_key, reverse_value)?;

    // 5. Write summary (COLD) - content-addressed
    let summary_key = EdgeSummaryCfKey(summary_hash);
    txn.put_cf(edge_summaries_cf, summary_key, summary)?;

    // 6. Write reverse index entry with CURRENT marker
    let index_key = EdgeSummaryIndexCfKey(summary_hash, src, dst, name_hash, version);
    let index_value = EdgeSummaryIndexCfValue(EdgeSummaryIndexCfValue::CURRENT);
    txn.put_cf(edge_summary_index_cf, index_key, index_value)?;

    txn.commit()
}
```

---

## 3.2 Write Path: Update (Optimistic Locking)
(codex, 2026-02-03, validated)

### Refcount Handling for Content-Addressed Summaries

When summaries are keyed by `SummaryHash`, multiple entities can share the same
summary row. To safely GC these rows, `NodeSummaryCfValue` and
`EdgeSummaryCfValue` store a refcount.

**Rules (must occur in the same transaction as the entity update):**

- **Insert (new entity with summary hash H):** increment refcount(H), create row if missing.
- **Update (old hash H1 → new hash H2):**
  - increment refcount(H2)
  - decrement refcount(H1)
  - if refcount(H1) reaches 0, delete the summary row.
- **Delete (tombstone, hash H):** decrement refcount(H); delete row if it reaches 0.

(codex, 2026-02-03, planned)

### Update Node

```rust
fn update_node(
    &self,
    id: Id,
    new_summary: NodeSummary,
    expected_version: Version,
) -> Result<Version> {
    let txn = self.txn_db.transaction();

    // 1. Read current node
    let current = txn.get_cf(nodes_cf, id)?
        .ok_or(Error::NotFound(id))?;
    let current = Nodes::value_from_bytes(&current)?;
    let current_version = current.3;
    let old_hash = current.2; // Save old hash for index update

    // 2. Optimistic lock check
    if current_version != expected_version {
        return Err(Error::VersionMismatch {
            expected: expected_version,
            actual: current_version,
        });
    }

    // 3. Version overflow check
    if current_version == VERSION_MAX {
        return Err(Error::VersionOverflow(id));
    }

    // 4. Compute new version and hash
    let new_version = current_version + 1;
    let new_hash = SummaryHash::from_summary(&new_summary)?;

    // 5. Write updated node (HOT) - preserve deleted flag
    let new_value = NodeCfValue(current.0, current.1, Some(new_hash), new_version, current.4);
    txn.put_cf(nodes_cf, NodeCfKey(id), new_value)?;

    // 6. Write new summary (COLD) - content-addressed
    let summary_key = NodeSummaryCfKey(new_hash);
    txn.put_cf(node_summaries_cf, summary_key, new_summary)?;

    // 7. Flip old index entry to STALE
    if let Some(old_h) = old_hash {
        let old_index_key = NodeSummaryIndexCfKey(old_h, id, current_version);
        let stale_value = NodeSummaryIndexCfValue(NodeSummaryIndexCfValue::STALE);
        txn.put_cf(node_summary_index_cf, old_index_key, stale_value)?;
    }

    // 8. Write new index entry with CURRENT marker
    let new_index_key = NodeSummaryIndexCfKey(new_hash, id, new_version);
    let current_value = NodeSummaryIndexCfValue(NodeSummaryIndexCfValue::CURRENT);
    txn.put_cf(node_summary_index_cf, new_index_key, current_value)?;

    txn.commit()?;
    Ok(new_version)
}
```

### Update Edge

```rust
fn update_edge(
    &self,
    src: Id,
    dst: Id,
    name: &str,
    new_summary: EdgeSummary,
    expected_version: Version,
) -> Result<Version> {
    let txn = self.txn_db.transaction();

    let name_hash = NameHash::from_name(name);
    let edge_key = ForwardEdgeCfKey(src, dst, name_hash);

    // 1. Read current edge
    let current = txn.get_cf(forward_edges_cf, &edge_key)?
        .ok_or(Error::NotFound(edge_key))?;
    let current = ForwardEdges::value_from_bytes(&current)?;
    let current_version = current.3;
    let old_hash = current.2; // Save old hash for index update

    // 2. Optimistic lock check
    if current_version != expected_version {
        return Err(Error::VersionMismatch {
            expected: expected_version,
            actual: current_version,
        });
    }

    // 3. Version overflow check
    if current_version == VERSION_MAX {
        return Err(Error::VersionOverflow(edge_key));
    }

    // 4. Compute new version and hash
    let new_version = current_version + 1;
    let new_hash = SummaryHash::from_summary(&new_summary)?;

    // 5. Write updated forward edge (HOT) - preserve deleted flag
    let new_value = ForwardEdgeCfValue(current.0, current.1, Some(new_hash), new_version, current.4);
    txn.put_cf(forward_edges_cf, edge_key, new_value)?;

    // Note: Reverse edge key unchanged. TemporalRange is updated via temporal mutations.

    // 6. Write new summary (COLD) - content-addressed
    let summary_key = EdgeSummaryCfKey(new_hash);
    txn.put_cf(edge_summaries_cf, summary_key, new_summary)?;

    // 7. Flip old index entry to STALE
    if let Some(old_h) = old_hash {
        let old_index_key = EdgeSummaryIndexCfKey(old_h, src, dst, name_hash, current_version);
        let stale_value = EdgeSummaryIndexCfValue(EdgeSummaryIndexCfValue::STALE);
        txn.put_cf(edge_summary_index_cf, old_index_key, stale_value)?;
    }

    // 8. Write new index entry with CURRENT marker
    let new_index_key = EdgeSummaryIndexCfKey(new_hash, src, dst, name_hash, new_version);
    let current_value = EdgeSummaryIndexCfValue(EdgeSummaryIndexCfValue::CURRENT);
    txn.put_cf(edge_summary_index_cf, new_index_key, current_value)?;

    txn.commit()?;
    Ok(new_version)
}
```

### Delete Node (Tombstone)

```rust
fn delete_node(&self, id: Id, expected_version: Version) -> Result<Version> {
    let txn = self.txn_db.transaction();

    // 1. Read current node
    let current = txn.get_cf(nodes_cf, id)?
        .ok_or(Error::NotFound(id))?;
    let current = Nodes::value_from_bytes(&current)?;
    let current_version = current.3;

    // 2. Optimistic lock check
    if current_version != expected_version {
        return Err(Error::VersionMismatch {
            expected: expected_version,
            actual: current_version,
        });
    }

    // 3. Already deleted?
    if current.4 {
        return Err(Error::AlreadyDeleted(id));
    }

    // 4. Increment version and set deleted flag
    let new_version = current_version + 1;
    let new_value = NodeCfValue(current.0, current.1, current.2, new_version, true);
    txn.put_cf(nodes_cf, NodeCfKey(id), new_value)?;

    // 5. Flip current index entry to STALE
    if let Some(hash) = current.2 {
        let index_key = NodeSummaryIndexCfKey(hash, id, current_version);
        let stale_value = NodeSummaryIndexCfValue(NodeSummaryIndexCfValue::STALE);
        txn.put_cf(node_summary_index_cf, index_key, stale_value)?;
    }

    // Note: No new index entry for tombstoned entity
    // Tombstone versions do not add a new summary row; latest summary remains at prior version.
    // Versioned summaries are preserved for audit/time-travel until GC.

    txn.commit()?;
    Ok(new_version)
}
```

### Delete Edge (Tombstone)

```rust
fn delete_edge(
    &self,
    src: Id,
    dst: Id,
    name: &str,
    expected_version: Version,
) -> Result<Version> {
    let txn = self.txn_db.transaction();

    let name_hash = NameHash::from_name(name);
    let edge_key = ForwardEdgeCfKey(src, dst, name_hash);

    // 1. Read current edge
    let current = txn.get_cf(forward_edges_cf, &edge_key)?
        .ok_or(Error::NotFound(edge_key))?;
    let current = ForwardEdges::value_from_bytes(&current)?;
    let current_version = current.3;

    // 2. Optimistic lock check
    if current_version != expected_version {
        return Err(Error::VersionMismatch {
            expected: expected_version,
            actual: current_version,
        });
    }

    // 3. Already deleted?
    if current.4 {
        return Err(Error::AlreadyDeleted(edge_key));
    }

    // 4. Increment version and set deleted flag
    let new_version = current_version + 1;
    let new_value = ForwardEdgeCfValue(current.0, current.1, current.2, new_version, true);
    txn.put_cf(forward_edges_cf, edge_key, new_value)?;

    // 5. Flip current index entry to STALE
    if let Some(hash) = current.2 {
        let index_key = EdgeSummaryIndexCfKey(hash, src, dst, name_hash, current_version);
        let stale_value = EdgeSummaryIndexCfValue(EdgeSummaryIndexCfValue::STALE);
        txn.put_cf(edge_summary_index_cf, index_key, stale_value)?;
    }

    // Note: Reverse edge entry stays (could remove if needed)
    // Tombstone versions do not add a new summary row; latest summary remains at prior version.
    // Versioned summaries are preserved for audit/time-travel until GC.

    txn.commit()?;
    Ok(new_version)
}
```

---

## 3.3 Read Path: Query Implementation

### current_nodes_for_summary

```rust
pub fn current_nodes_for_summary(&self, hash: SummaryHash) -> Result<Vec<Id>> {
    let prefix = hash.as_bytes();
    let mut results = Vec::new();

    for (key, value) in self.prefix_scan(node_summary_index_cf, prefix)? {
        // Filter by marker bit - no point-read needed!
        let marker = NodeSummaryIndexCfValue::from_bytes(&value)?;
        if !marker.is_current() {
            continue;
        }

        let index_key = NodeSummaryIndex::key_from_bytes(&key)?;
        let node_id = index_key.1;

        // Optional: verify entity not tombstoned (if audit queries need different behavior)
        if let Some(node) = self.get_node(node_id)? {
            if !node.deleted {
                results.push(node_id);
            }
        }
    }

    Ok(results)
}
```

### current_edges_for_summary

```rust
pub fn current_edges_for_summary(&self, hash: SummaryHash) -> Result<Vec<ForwardEdgeCfKey>> {
    let prefix = hash.as_bytes();
    let mut results = Vec::new();

    for (key, value) in self.prefix_scan(edge_summary_index_cf, prefix)? {
        // Filter by marker bit - no point-read needed!
        let marker = EdgeSummaryIndexCfValue::from_bytes(&value)?;
        if !marker.is_current() {
            continue;
        }

        let index_key = EdgeSummaryIndex::key_from_bytes(&key)?;
        let edge_key = ForwardEdgeCfKey(index_key.1, index_key.2, index_key.3);

        // Optional: verify entity not tombstoned
        if let Some(edge) = self.get_edge(&edge_key)? {
            if !edge.deleted {
                results.push(edge_key);
            }
        }
    }

    Ok(results)
}
```

---

## 3.4 Garbage Collection
(codex, 2026-02-02, planned) (claude, 2026-02-02, implemented in gc.rs - GraphGarbageCollector with cursor-based incremental processing)

### GcConfig

```rust
pub struct GraphGcConfig {
    /// Interval between GC cycles
    pub interval: Duration,  // default: 60s

    /// Max entries to process per cycle (bounds latency)
    pub batch_size: usize,  // default: 1000

    /// Number of summary versions to retain per entity
    pub versions_to_keep: usize,  // default: 2

    /// Tombstone retention period before hard delete
    pub tombstone_retention: Duration,  // default: 7 days

    /// Run GC on startup
    pub process_on_startup: bool,  // default: true
}
```

### GC Cursor (Incremental Processing)

Cursors are persisted using the `GraphMeta` pattern from the vector crate — a discriminated enum where the key discriminant selects the field type.

```rust
/// Stored in "graph/meta" CF — reuses singleton metadata pattern from vector::GraphMeta
pub struct GraphMeta;
impl ColumnFamily for GraphMeta {
    const CF_NAME: &'static str = "graph/meta";
}

/// Discriminated enum — discriminant byte in key, payload in value
/// Pattern borrowed from vector::schema::GraphMetaField
#[derive(Debug, Clone)]
pub enum GraphMetaField {
    // GC cursors: last processed key bytes (empty = start from beginning)
    GcCursorNodeSummaries(Vec<u8>),      // 0x00
    GcCursorEdgeSummaries(Vec<u8>),      // 0x01
    GcCursorNodeSummaryIndex(Vec<u8>),   // 0x02
    GcCursorEdgeSummaryIndex(Vec<u8>),   // 0x03
    GcCursorNodeTombstones(Vec<u8>),     // 0x04
    GcCursorEdgeTombstones(Vec<u8>),     // 0x05
    // Future: other graph-level metadata
}

impl GraphMetaField {
    pub fn discriminant(&self) -> u8 {
        match self {
            Self::GcCursorNodeSummaries(_) => 0x00,
            Self::GcCursorEdgeSummaries(_) => 0x01,
            Self::GcCursorNodeSummaryIndex(_) => 0x02,
            Self::GcCursorEdgeSummaryIndex(_) => 0x03,
            Self::GcCursorNodeTombstones(_) => 0x04,
            Self::GcCursorEdgeTombstones(_) => 0x05,
        }
    }

    pub fn from_discriminant(d: u8) -> Self {
        match d {
            0x00 => Self::GcCursorNodeSummaries(vec![]),
            0x01 => Self::GcCursorEdgeSummaries(vec![]),
            0x02 => Self::GcCursorNodeSummaryIndex(vec![]),
            0x03 => Self::GcCursorEdgeSummaryIndex(vec![]),
            0x04 => Self::GcCursorNodeTombstones(vec![]),
            0x05 => Self::GcCursorEdgeTombstones(vec![]),
            _ => panic!("Unknown GraphMetaField discriminant: {}", d),
        }
    }
}

/// Key: just the discriminant byte (1 byte total)
pub struct GraphMetaCfKey(pub GraphMetaField);

/// Value: field-dependent payload (cursor bytes for GC cursors)
pub struct GraphMetaCfValue(pub GraphMetaField);

impl GraphMeta {
    pub fn key_to_bytes(key: &GraphMetaCfKey) -> Vec<u8> {
        vec![key.0.discriminant()]
    }

    pub fn key_from_bytes(bytes: &[u8]) -> Result<GraphMetaCfKey> {
        if bytes.len() != 1 {
            anyhow::bail!("Invalid GraphMetaCfKey length");
        }
        Ok(GraphMetaCfKey(GraphMetaField::from_discriminant(bytes[0])))
    }

    pub fn value_to_bytes(value: &GraphMetaCfValue) -> Vec<u8> {
        match &value.0 {
            GraphMetaField::GcCursorNodeSummaries(v)
            | GraphMetaField::GcCursorEdgeSummaries(v)
            | GraphMetaField::GcCursorNodeSummaryIndex(v)
            | GraphMetaField::GcCursorEdgeSummaryIndex(v)
            | GraphMetaField::GcCursorNodeTombstones(v)
            | GraphMetaField::GcCursorEdgeTombstones(v) => v.clone(),
        }
    }

    pub fn value_from_bytes(key: &GraphMetaCfKey, bytes: &[u8]) -> Result<GraphMetaCfValue> {
        let field = match key.0.discriminant() {
            0x00 => GraphMetaField::GcCursorNodeSummaries(bytes.to_vec()),
            0x01 => GraphMetaField::GcCursorEdgeSummaries(bytes.to_vec()),
            0x02 => GraphMetaField::GcCursorNodeSummaryIndex(bytes.to_vec()),
            0x03 => GraphMetaField::GcCursorEdgeSummaryIndex(bytes.to_vec()),
            0x04 => GraphMetaField::GcCursorNodeTombstones(bytes.to_vec()),
            0x05 => GraphMetaField::GcCursorEdgeTombstones(bytes.to_vec()),
            d => anyhow::bail!("Unknown discriminant: {}", d),
        };
        Ok(GraphMetaCfValue(field))
    }
}
```
(codex, 2026-02-02, planned)

**Pattern Benefits (from vector::GraphMeta):**
- Single CF for all graph-level metadata
- Extensible: add new fields with new discriminants
- Compact keys: 1 byte per field
- Type-safe: enum variants enforce value types
- Same pattern as vector subsystem for consistency

**Persistence & Recovery:**
- On GC cycle completion: write cursor to `graph/meta` CF in same transaction
- On startup: read cursor from `graph/meta` CF, resume from last position
- If cursor points to deleted key, RocksDB iterator advances to next valid key

**Incremental GC Behavior:**
- Each cycle: read cursor, scan next `batch_size` entries, update cursor
- When cursor reaches end (iterator exhausted), delete cursor to reset
- Bounds work per cycle, amortizes cost over time

### GC Targets

| Target | Key Pattern | Retention Policy |
|--------|-------------|------------------|
| NodeSummaries | `(SummaryHash)` | Content-addressed; delete when refcount reaches 0 |
| EdgeSummaries | `(SummaryHash)` | Content-addressed; delete when refcount reaches 0 |
| NodeSummaryIndex | `(Hash, Id, version)` | Delete if marker=STALE and version not in retained set |
| EdgeSummaryIndex | `(Hash, EdgeKey, version)` | Delete if marker=STALE and version not in retained set |
| Tombstones (Nodes) | `(Id)` where deleted=true | Hard delete after `tombstone_retention` period |
| Tombstones (Edges) | `(EdgeKey)` where deleted=true | Hard delete after `tombstone_retention` period |
(codex, 2026-02-03, validated)

### GcMetrics

```rust
#[derive(Default)]
pub struct GcMetrics {
    pub node_summaries_deleted: AtomicU64,
    pub edge_summaries_deleted: AtomicU64,
    pub node_index_entries_deleted: AtomicU64,
    pub edge_index_entries_deleted: AtomicU64,
    pub cycles_completed: AtomicU64,
}
```

### GC Implementation

```rust
pub struct GraphGarbageCollector {
    storage: Arc<Storage>,
    config: GraphGcConfig,
    shutdown: Arc<AtomicBool>,
    metrics: Arc<GcMetrics>,
}

impl GraphGarbageCollector {
    pub fn start(storage: Arc<Storage>, config: GraphGcConfig) -> Self;
    pub fn shutdown(self);
    pub fn run_cycle(&self) -> Result<GcMetrics>;
}
```

### GC Logic: Node Summaries

**Note:** This GC logic assumes versioned summaries keyed by `(Id, version)`. It is not applicable while summaries remain content-addressed by `SummaryHash`. (codex, 2026-02-03, validated)

```rust
fn gc_node_summaries(&self) -> Result<u64> {
    let txn = self.txn_db.transaction();
    let mut deleted = 0u64;
    let n = self.config.versions_to_keep as Version;

    for (node_id, node_value) in self.scan_nodes_batch()? {
        let current_version = node_value.version;
        let min_keep = current_version.saturating_sub(n - 1);

        // Scan summaries for this node
        let prefix = node_id.into_bytes();
        for (key, _) in self.prefix_scan(node_summaries_cf, &prefix)? {
            let summary_key = NodeSummaries::key_from_bytes(&key)?;
            let version = summary_key.1;

            if version < min_keep {
                txn.delete_cf(node_summaries_cf, &key)?;
                deleted += 1;
            }
        }
    }

    txn.commit()?;
    Ok(deleted)
}
```

### GC Logic: Reverse Index (with Cursor)

```rust
fn gc_node_summary_index(&self) -> Result<u64> {
    let txn = self.txn_db.transaction();
    let mut deleted = 0u64;
    let mut processed = 0usize;
    let n = self.config.versions_to_keep as Version;

    // 1. Load cursor from graph/meta CF using GraphMeta pattern
    let cursor_key = GraphMetaCfKey(GraphMetaField::GcCursorNodeSummaryIndex(vec![]));
    let cursor_key_bytes = GraphMeta::key_to_bytes(&cursor_key);
    let start_key = txn.get_cf(graph_meta_cf, &cursor_key_bytes)?
        .map(|v| GraphMeta::value_from_bytes(&cursor_key, &v).unwrap().0)
        .and_then(|f| match f {
            GraphMetaField::GcCursorNodeSummaryIndex(cursor) => Some(cursor),
            _ => None,
        })
        .unwrap_or_default();

    // 2. Cache: node_id → (current_version, deleted)
    let version_cache: HashMap<Id, (Version, bool)> = self.scan_nodes_batch()?
        .map(|(id, val)| (id, (val.version, val.deleted)))
        .collect();

    // 3. Incremental scan from cursor position
    let iter = if start_key.is_empty() {
        txn.iterator_cf(node_summary_index_cf, IteratorMode::Start)
    } else {
        txn.iterator_cf(node_summary_index_cf, IteratorMode::From(&start_key, Direction::Forward))
    };

    let mut last_key: Option<Vec<u8>> = None;
    for item in iter {
        if processed >= self.config.batch_size {
            break; // Batch limit reached
        }

        let (key, value) = item?;
        last_key = Some(key.to_vec());
        processed += 1;

        let index_key = NodeSummaryIndex::key_from_bytes(&key)?;
        let (_, node_id, version) = (index_key.0, index_key.1, index_key.2);
        let marker = NodeSummaryIndexCfValue::from_bytes(&value)?;

        // Only GC stale entries
        if !marker.is_current() {
            match version_cache.get(&node_id) {
                Some(&(current_version, node_deleted)) => {
                    let min_keep = current_version.saturating_sub(n - 1);
                    // Delete if version too old OR node is tombstoned
                    if version < min_keep || node_deleted {
                        txn.delete_cf(node_summary_index_cf, &key)?;
                        deleted += 1;
                    }
                }
                None => {
                    // Node hard-deleted entirely
                    txn.delete_cf(node_summary_index_cf, &key)?;
                    deleted += 1;
                }
            }
        }
    }

    // 4. Persist cursor for next cycle using GraphMeta pattern
    if let Some(key) = last_key {
        let cursor_value = GraphMetaCfValue(GraphMetaField::GcCursorNodeSummaryIndex(key));
        txn.put_cf(graph_meta_cf, &cursor_key_bytes, GraphMeta::value_to_bytes(&cursor_value))?;
    } else {
        // Iterator exhausted - delete cursor to start fresh next cycle
        txn.delete_cf(graph_meta_cf, &cursor_key_bytes)?;
    }

    txn.commit()?;
    Ok(deleted)
}
```

---

## 3.5 Reverse Index Repair Task
(codex, 2026-02-02, planned)

Periodic integrity check to detect and fix inconsistencies between forward edges and reverse index.

### RepairConfig

```rust
pub struct RepairConfig {
    /// Interval between repair cycles
    pub interval: Duration,  // default: 1 hour

    /// Max entries to check per cycle
    pub batch_size: usize,  // default: 10000

    /// Auto-fix inconsistencies (vs. report only)
    pub auto_fix: bool,  // default: false
}
```

### Repair Logic

```rust
fn repair_forward_reverse_consistency(&self) -> Result<RepairMetrics> {
    let mut metrics = RepairMetrics::default();

    // 1. Scan forward edges, ensure reverse entries exist
    for (fwd_key, _) in self.scan_forward_edges_batch()? {
        let rev_key = ReverseEdgeCfKey(fwd_key.1, fwd_key.0, fwd_key.2);
        if self.get_reverse_edge(&rev_key)?.is_none() {
            metrics.missing_reverse += 1;
            if self.config.auto_fix {
                self.create_reverse_edge(&rev_key)?;
            }
        }
    }

    // 2. Scan reverse edges, drop orphans pointing to missing forward edges
    for (rev_key, _) in self.scan_reverse_edges_batch()? {
        let fwd_key = ForwardEdgeCfKey(rev_key.1, rev_key.0, rev_key.2);
        if self.get_forward_edge(&fwd_key)?.is_none() {
            metrics.orphan_reverse += 1;
            if self.config.auto_fix {
                self.delete_reverse_edge(&rev_key)?;
            }
        }
    }

    Ok(metrics)
}
```

---

# Part 4: File Changes Summary

| File | Changes |
|------|---------|
| **schema.rs** | Add `version: Version` and `deleted: bool` to entity values; add index CfValue with marker byte; change summary CF keys to `(EntityId, Version)`; add `NodeSummaryIndex` and `EdgeSummaryIndex` CFs; add `GraphMeta` CF with discriminated enum pattern (borrowed from vector::GraphMeta); update `ALL_COLUMN_FAMILIES` |
| **mutation.rs** | Update `AddNode`/`AddEdge` to set version=1, deleted=false, and write index with CURRENT marker; add `UpdateNode`/`UpdateEdge` with optimistic locking (flip old index to STALE); add `DeleteNode`/`DeleteEdge` tombstone mutations |
| **query.rs** | Add `all_nodes_for_summary()`, `current_nodes_for_summary()`, `node_versions_for_summary()`, and edge equivalents; filter by marker byte in current queries; filter out deleted entities |
| **gc.rs** | New file: `GraphGcConfig` with cursor support and tombstone retention; incremental GC with resume cursor; `GraphGarbageCollector` with background worker |
| **repair.rs** | New file: `RepairConfig`, forward↔reverse consistency checker |
| **tests.rs** | Test optimistic locking (version mismatch); test multi-version resolution; test tombstone semantics; test GC cleanup; test marker bit filtering |

---

# Part 5: Summary

| Aspect | Design |
|--------|--------|
| **Reverse index** | `(SummaryHash, EntityKey, Version)` → 1-byte marker (CURRENT/STALE) |
| **Multi-version support** | Index returns all versions; query API provides `all_*` and `current_*` variants |
| **Content storage** | Content-addressed by `SummaryHash` with refcounted summaries |
| **Optimistic locking** | Version in entity value; reject update if mismatch |
| **Delete semantics** | Tombstone flag in entity value; retained for audit until GC |
| **Version overflow** | Reject writes at VERSION_MAX; documented policy |
| **GC** | Incremental with cursor (persisted via `GraphMeta` pattern); delete old versions; delete stale index entries; hard delete tombstones after retention |
| **Repair** | Periodic forward↔reverse consistency check |
(codex, 2026-02-03, partial)

**Approval:** Design is ready to implement, with noted planned components. (codex, 2026-02-02, approved)

**Implementation Status** (claude, 2026-02-02):
- [x] Version type and entity value fields - schema.rs
- [x] NodeSummaryIndex and EdgeSummaryIndex CFs - schema.rs, subsystem.rs
- [x] Insert write paths with CURRENT marker - mutation.rs (AddNode, AddEdge)
- [x] Reverse lookup query APIs - query.rs (NodesBySummaryHash, EdgesBySummaryHash)
- [x] GraphMeta CF for GC cursors - schema.rs, subsystem.rs
- [x] Update/Delete mutations with optimistic locking - mutation.rs (UpdateNodeSummary, UpdateEdgeSummary, DeleteNode, DeleteEdge)
- [x] GC implementation - gc.rs (GraphGarbageCollector with cursor-based incremental GC)
- [x] GC for orphaned summaries - gc.rs (gc_node_summaries, gc_edge_summaries)
- [x] Reverse index repair task - repair.rs (GraphRepairer with forward↔reverse consistency checking)
- [x] Fulltext index updates for Update/Delete - fulltext/mutation.rs
- [x] Unit tests - tests.rs (content_address_tests module)

---

# Part 6: Performance Analysis

## 6.1 Write Amplification

### Baseline (Current Design)

| Operation | CF Writes | Details |
|-----------|-----------|---------|
| AddNode | 3 puts | Names, NodeSummaries, Nodes |
| AddEdge | 4 puts | Names, EdgeSummaries, ForwardEdges, ReverseEdges |
| UpdateEdgeWeight | 1 get + 1 put | Read-modify-write on ForwardEdges |
(codex, 2026-02-02, validated)
**Note:** Summary writes are conditional on non-empty summaries; counts assume summaries are present. (codex, 2026-02-02, validated)

### New Design

| Operation | CF Writes | Delta | Notes |
|-----------|-----------|-------|-------|
| InsertNode | 4 puts | **+1** | +NodeSummaryIndex (CURRENT marker) |
| InsertEdge | 5 puts | **+1** | +EdgeSummaryIndex (CURRENT marker) |
| UpdateNode | 1 get + 4 puts | **+1 get, +2 puts** | Optimistic read + STALE/CURRENT markers |
| UpdateEdge | 1 get + 4 puts | **+2 puts** | STALE/CURRENT markers |
| DeleteNode | 1 get + 2 puts | **new op** | Tombstone + STALE marker |
| DeleteEdge | 1 get + 2 puts | **new op** | Tombstone + STALE marker |
(codex, 2026-02-02, derived)

### Bytes per Operation

| Operation | Key Bytes | Value Bytes | Total |
|-----------|-----------|-------------|-------|
| NodeSummaryIndex entry | 28 | 1 | 29 |
| EdgeSummaryIndex entry | 52 | 1 | 53 |
| Version field (entity) | 0 | 4 | 4 |
| Deleted flag (entity) | 0 | 1 | 1 |
(codex, 2026-02-02, derived)

---

## 6.2 Read Amplification

### Reverse Lookup: `current_nodes_for_summary(hash)`

| Step | Operation | Count |
|------|-----------|-------|
| 1 | Prefix scan NodeSummaryIndex | N entries |
| 2 | Filter by marker (CURRENT) | in-scan |
| 3 | Get Nodes (tombstone check) | M gets (M ≤ N) |

**Optimization:** Skip step 3 if tombstone filtering not required.
(codex, 2026-02-02, derived)

### Reverse Lookup: `all_nodes_for_summary(hash)`

| Step | Operation | Count |
|------|-----------|-------|
| 1 | Prefix scan NodeSummaryIndex | N entries |

No additional reads — returns all (entity, version) pairs.
(codex, 2026-02-02, derived)

### Point Reads (Unchanged)

| Query | Operations |
|-------|------------|
| NodeById | 1 get |
| OutgoingEdges | 1 prefix scan |
| IncomingEdges | 1 prefix scan |
(codex, 2026-02-02, validated)

---

## 6.3 QPS Impact Estimates

### By Workload Type

| Workload | Write Impact | Read Impact | Net |
|----------|--------------|-------------|-----|
| Insert-heavy (90% insert) | -20% to -25% | 0% | **-18% to -23%** |
| Update-heavy (50% update) | -33% to -50% | 0% | **-17% to -25%** |
| Read-heavy (90% read) | -20% | 0% | **-2%** |
| Mixed balanced | -25% | 0% | **-12%** |
(codex, 2026-02-02, estimated)

### Benchmark Reference Points

From existing benchmarks:

| Metric | Value | Source |
|--------|-------|--------|
| Vector insert throughput | ~140 ops/sec | throughput_balanced.csv |
| Graph DFS 10K nodes | 0.1417ms | performance_metrics.csv |
| Graph BFS 10K nodes | 0.1792ms | performance_metrics.csv |
| RocksDB single put | ~7μs | typical SSD |
**Note:** RocksDB single-put latency is a rough estimate and depends on hardware/configuration. (codex, 2026-02-02, estimated)
(codex, 2026-02-02, partially-validated)

**Estimated index write overhead:** ~7-14μs per mutation (1-2 additional puts). (codex, 2026-02-02, estimated)

---

## 6.4 Mitigation Strategies

### 1. Batch Writes

Group mutations to amortize WAL overhead:

```rust
// Bad: individual writes (3 WAL syncs)
node1.run(&writer).await?;
node2.run(&writer).await?;
node3.run(&writer).await?;

// Good: batched (1 WAL sync)
mutations![node1, node2, node3].run(&writer).await?;
```

**Impact:** 2-3x throughput improvement for bulk operations.
(codex, 2026-02-02, estimated)

### 2. Async Index Updates (Optional)

For write-heavy workloads tolerating eventual consistency:

```rust
// HOT path: entity write only (synchronous)
txn.put_cf(nodes_cf, key, value)?;
txn.commit()?;

// COLD path: index update (async, best-effort)
index_queue.send(IndexUpdate::Node { id, version, hash })?;
```

**Trade-off:** Reverse lookups may miss recently written entities until index catches up.
(codex, 2026-02-02, validated)

### 3. Skip Tombstone Check

For use cases where deleted entities are acceptable in results:

```rust
pub fn current_nodes_for_summary_fast(&self, hash: SummaryHash) -> Result<Vec<Id>> {
    // Only prefix scan + marker filter
    // No Nodes CF get (skip tombstone check)
}
```

**Impact:** 50% reduction in read amplification for reverse lookups.
(codex, 2026-02-02, estimated)

### 4. Tuned RocksDB Settings

Recommended CF-specific tuning:

| CF | Block Cache | Bloom Filter | Compression |
|----|-------------|--------------|-------------|
| Nodes, ForwardEdges | Large (HOT) | 10 bits | LZ4 |
| NodeSummaryIndex | Medium | 10 bits | None (small values) |
| NodeSummaries | Small (COLD) | None | ZSTD |
(codex, 2026-02-02, suggested)

---

## 6.5 Cost-Benefit Summary

| Cost | Benefit |
|------|---------|
| +1 put per insert | Reverse lookup: hash → entities |
| +2 puts per update | Multi-version tracking |
| +1 get per update | Optimistic locking (no lost updates) |
| +5 new CFs | Clean separation HOT/COLD/INDEX |
| ~15-25% write throughput | Vector search → graph entity resolution |
(codex, 2026-02-02, estimated)

**Storage cost note:** If summaries were per-entity/per-version (instead of content-addressed), templated edge summaries (e.g., “Friends”, “co-workers”) would duplicate across many edges and versions. Content-addressed summaries avoid this duplication but require refcounting for safe GC. (codex, 2026-02-03, validated)

**Conclusion:** The reverse index capability enables the core use case (vector search results resolving to graph entities). The write amplification cost is acceptable and can be mitigated through batching. (codex, 2026-02-02, estimated)
