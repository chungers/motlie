# CONTENT-ADDRESS: Reverse Index with Versioning, Optimistic Locking, and GC

## Problem

1. **No reverse index**: `SummaryHash` from vector search cannot be resolved to graph entities without full scan
2. **No optimistic locking**: Blind upserts can silently lose concurrent updates
3. **No GC for stale content**: Old summaries accumulate without cleanup

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
```

| Type | Max Value | Overflow at 1 update/sec | Size Savings vs u64 |
|------|-----------|--------------------------|---------------------|
| `u32` | 4.2 billion | 136 years | 4 bytes/entry |

## 1.1 Entity Column Families (HOT)

### Nodes

```rust
pub struct NodeCfKey(pub Id);  // 16 bytes

pub struct NodeCfValue(
    pub Option<TemporalRange>,
    pub NameHash,
    pub Option<SummaryHash>,  // Content hash for vector search matching
    pub Version,               // Version (monotonic, starts at 1) [NEW]
);
```

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
);
```

**Note:** Edge identity is `(src, dst, name)`. `name` is immutable; renames are modeled as delete+insert.

### ReverseEdges

```rust
pub struct ReverseEdgeCfKey(pub DstId, pub SrcId, pub NameHash);  // 40 bytes

// Value: empty (no ReverseEdgeCfValue type)
// ReverseEdges is purely an index for "find edges TO this node".
// All edge details (TemporalRange, weight, summary, version) live in ForwardEdges.
// This avoids data duplication and consistency bugs between forward/reverse CFs.
```

---

## 1.2 Content Column Families (COLD)

**Changed from content-addressed to entity+version keyed.** Enables clean GC.

### NodeSummaries

```rust
// OLD: NodeSummaryCfKey(SummaryHash)  // content-addressed
// NEW: Entity+version keyed
pub struct NodeSummaryCfKey(
    pub Id,      // 16 bytes - node_id
    pub Version, // 4 bytes - version
);  // Total: 20 bytes

pub struct NodeSummaryCfValue(pub NodeSummary);
```

### EdgeSummaries

```rust
// OLD: EdgeSummaryCfKey(SummaryHash)  // content-addressed
// NEW: Entity+version keyed
pub struct EdgeSummaryCfKey(
    pub SrcId,    // 16 bytes
    pub DstId,    // 16 bytes
    pub NameHash, // 8 bytes
    pub Version,  // 4 bytes - version
);  // Total: 44 bytes

pub struct EdgeSummaryCfValue(pub EdgeSummary);
```

---

## 1.3 Reverse Index Column Families [NEW]

### NodeSummaryIndex

```rust
pub struct NodeSummaryIndexCfKey(
    pub SummaryHash,  // 8 bytes - prefix for hash lookup
    pub Id,           // 16 bytes - node_id
    pub Version,      // 4 bytes - version
);  // Total: 28 bytes

// Value: empty (existence is the data)
```

### EdgeSummaryIndex

```rust
pub struct EdgeSummaryIndexCfKey(
    pub SummaryHash,  // 8 bytes - prefix for hash lookup
    pub SrcId,        // 16 bytes
    pub DstId,        // 16 bytes
    pub NameHash,     // 8 bytes
    pub Version,      // 4 bytes - version
);  // Total: 52 bytes

// Value: empty
```

---

## 1.4 Fragment Column Families (Unchanged)

Fragments use timestamp for append-only semantics. No GC.

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
];
```

---

# Part 2: Reverse Lookup API and Behavior

## 2.1 Critical Insight: Multi-Version Resolution

A `SummaryHash` lookup can return **multiple results**, including:

| Scenario | What's Returned |
|----------|-----------------|
| Different entities with same content | All entities that share the hash |
| Same entity at different versions | Multiple (entity, version) pairs |
| Old version after entity was updated | The old (entity, version) — now stale |

### Use Cases

| Use Case | Query Type |
|----------|------------|
| Vector search resolution | Current versions only |
| Time-travel query | Specific version |
| Audit/debugging | All versions |
| Content history | All versions of specific entity |

---

## 2.2 Node Reverse Lookup API

### All Matches (Including Old Versions)

```rust
/// Returns ALL (node_id, version) pairs that have this hash.
/// Includes old versions that may no longer be current.
pub fn all_nodes_for_summary(&self, hash: SummaryHash) -> Result<Vec<(Id, Version)>>;
```

### Current Versions Only

```rust
/// Returns only node_ids where this hash is the CURRENT version.
/// Filters out old versions by checking entity's current state.
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

---

## 2.4 Index Prefix Scan Capabilities

### NodeSummaryIndex

| Prefix | Bytes | Returns |
|--------|-------|---------|
| `(hash)` | 8 | All nodes with this hash (any version) |
| `(hash, node_id)` | 24 | All versions of specific node with this hash |

### EdgeSummaryIndex

| Prefix | Bytes | Returns |
|--------|-------|---------|
| `(hash)` | 8 | All edges with this hash (any version) |
| `(hash, src)` | 24 | All edges from `src` with this hash |
| `(hash, src, dst)` | 40 | All edges between `src→dst` with this hash |
| `(hash, src, dst, name)` | 48 | All versions of specific edge with this hash |

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
    (0xAAA, A, 1) → empty   // Stale: A changed to 0xBBB at v2
    (0xAAA, B, 1) → empty   // Stale: B changed to 0xCCC at v2
    (0xAAA, C, 1) → empty   // Stale: C changed to 0xDDD at v2

Query Results for hash 0xAAA:
  all_nodes_for_summary(0xAAA):
    [(A, 1), (B, 1), (C, 1)]  // All three had it at v1

  current_nodes_for_summary(0xAAA):
    []  // Empty! None currently have 0xAAA
```

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
    (0xAAA, A, B, "knows", 1)      → empty  // Stale
    (0xAAA, C, D, "knows", 1)      → empty  // Current
    (0xAAA, E, F, "works_with", 1) → empty  // Stale

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

## 3.1 Write Path: Insert

### Insert Node (version = 1)

```rust
fn insert_node(&self, id: Id, name: &str, summary: NodeSummary) -> Result<()> {
    let txn = self.txn_db.transaction();

    // 1. Check doesn't exist
    if txn.get_cf(nodes_cf, id)?.is_some() {
        return Err(Error::AlreadyExists(id));
    }

    let version: Version = 1;
    let name_hash = NameHash::from_name(name);
    let summary_hash = SummaryHash::from_summary(&summary)?;

    // 2. Write name interning
    txn.put_cf(names_cf, NameCfKey(name_hash), name)?;

    // 3. Write node (HOT)
    let node_value = NodeCfValue(None, name_hash, Some(summary_hash), version);
    txn.put_cf(nodes_cf, NodeCfKey(id), node_value)?;

    // 4. Write versioned summary (COLD)
    let summary_key = NodeSummaryCfKey(id, version);
    txn.put_cf(node_summaries_cf, summary_key, summary)?;

    // 5. Write reverse index entry
    let index_key = NodeSummaryIndexCfKey(summary_hash, id, version);
    txn.put_cf(node_summary_index_cf, index_key, &[])?;

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
    let summary_hash = SummaryHash::from_summary(&summary)?;

    // 2. Write name interning
    txn.put_cf(names_cf, NameCfKey(name_hash), name)?;

    // 3. Write forward edge (HOT)
    let edge_value = ForwardEdgeCfValue(None, weight, Some(summary_hash), version);
    txn.put_cf(forward_edges_cf, edge_key, edge_value)?;

    // 4. Write reverse edge index (HOT) - empty value, just for reverse lookup
    let reverse_key = ReverseEdgeCfKey(dst, src, name_hash);
    txn.put_cf(reverse_edges_cf, reverse_key, &[])?;

    // 5. Write versioned summary (COLD)
    let summary_key = EdgeSummaryCfKey(src, dst, name_hash, version);
    txn.put_cf(edge_summaries_cf, summary_key, summary)?;

    // 6. Write reverse index entry
    let index_key = EdgeSummaryIndexCfKey(summary_hash, src, dst, name_hash, version);
    txn.put_cf(edge_summary_index_cf, index_key, &[])?;

    txn.commit()
}
```

---

## 3.2 Write Path: Update (Optimistic Locking)

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

    // 2. Optimistic lock check
    if current_version != expected_version {
        return Err(Error::VersionMismatch {
            expected: expected_version,
            actual: current_version,
        });
    }

    // 3. Compute new version and hash
    let new_version = current_version + 1;
    let new_hash = SummaryHash::from_summary(&new_summary)?;

    // 4. Write updated node (HOT)
    let new_value = NodeCfValue(current.0, current.1, Some(new_hash), new_version);
    txn.put_cf(nodes_cf, NodeCfKey(id), new_value)?;

    // 5. Write new versioned summary (COLD)
    // Old version stays until GC
    let summary_key = NodeSummaryCfKey(id, new_version);
    txn.put_cf(node_summaries_cf, summary_key, new_summary)?;

    // 6. Write new reverse index entry
    // Old index entry stays until GC
    let index_key = NodeSummaryIndexCfKey(new_hash, id, new_version);
    txn.put_cf(node_summary_index_cf, index_key, &[])?;

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

    // 2. Optimistic lock check
    if current_version != expected_version {
        return Err(Error::VersionMismatch {
            expected: expected_version,
            actual: current_version,
        });
    }

    // 3. Compute new version and hash
    let new_version = current_version + 1;
    let new_hash = SummaryHash::from_summary(&new_summary)?;

    // 4. Write updated forward edge (HOT)
    let new_value = ForwardEdgeCfValue(current.0, current.1, Some(new_hash), new_version);
    txn.put_cf(forward_edges_cf, edge_key, new_value)?;

    // Note: Reverse edge index unchanged (key stays same, value is empty)

    // 5. Write new versioned summary (COLD)
    let summary_key = EdgeSummaryCfKey(src, dst, name_hash, new_version);
    txn.put_cf(edge_summaries_cf, summary_key, new_summary)?;

    // 6. Write new reverse index entry
    let index_key = EdgeSummaryIndexCfKey(new_hash, src, dst, name_hash, new_version);
    txn.put_cf(edge_summary_index_cf, index_key, &[])?;

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

    for (key, _) in self.prefix_scan(node_summary_index_cf, prefix)? {
        let index_key = NodeSummaryIndex::key_from_bytes(&key)?;
        let (_, node_id, version) = (index_key.0, index_key.1, index_key.2);

        // Verify this version is still current
        if let Some(node) = self.get_node(node_id)? {
            if node.version == version && node.summary_hash == Some(hash) {
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

    for (key, _) in self.prefix_scan(edge_summary_index_cf, prefix)? {
        let index_key = EdgeSummaryIndex::key_from_bytes(&key)?;
        let edge_key = ForwardEdgeCfKey(index_key.1, index_key.2, index_key.3);
        let version = index_key.4;

        // Verify this version is still current
        if let Some(edge) = self.get_edge(&edge_key)? {
            if edge.version == version && edge.summary_hash == Some(hash) {
                results.push(edge_key);
            }
        }
    }

    Ok(results)
}
```

---

## 3.4 Garbage Collection

### GcConfig

```rust
pub struct GraphGcConfig {
    /// Interval between GC cycles
    pub interval: Duration,  // default: 60s

    /// Max entities to process per cycle
    pub batch_size: usize,  // default: 1000

    /// Number of summary versions to retain per entity
    pub versions_to_keep: usize,  // default: 2

    /// Run GC on startup
    pub process_on_startup: bool,  // default: true
}
```

### GC Targets

| Target | Key Pattern | Retention Policy |
|--------|-------------|------------------|
| NodeSummaries | `(Id, version)` | Keep versions ≥ (current - N + 1) |
| EdgeSummaries | `(EdgeKey, version)` | Keep versions ≥ (current - N + 1) |
| NodeSummaryIndex | `(Hash, Id, version)` | Delete if version not in retained set |
| EdgeSummaryIndex | `(Hash, EdgeKey, version)` | Delete if version not in retained set |

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

### GC Logic: Reverse Index

```rust
fn gc_node_summary_index(&self) -> Result<u64> {
    let txn = self.txn_db.transaction();
    let mut deleted = 0u64;
    let n = self.config.versions_to_keep as Version;

    // Cache: node_id → current_version
    let version_cache: HashMap<Id, Version> = self.scan_nodes_batch()?
        .map(|(id, val)| (id, val.version))
        .collect();

    for (key, _) in self.full_scan(node_summary_index_cf)? {
        let index_key = NodeSummaryIndex::key_from_bytes(&key)?;
        let (_, node_id, version) = (index_key.0, index_key.1, index_key.2);

        match version_cache.get(&node_id) {
            Some(&current_version) => {
                let min_keep = current_version.saturating_sub(n - 1);
                if version < min_keep {
                    txn.delete_cf(node_summary_index_cf, &key)?;
                    deleted += 1;
                }
            }
            None => {
                // Node deleted entirely
                txn.delete_cf(node_summary_index_cf, &key)?;
                deleted += 1;
            }
        }
    }

    txn.commit()?;
    Ok(deleted)
}
```


# Part 4: File Changes Summary

| File | Changes |
|------|---------|
| **schema.rs** | Add `version: Version` to entity values; change summary CF keys to `(EntityId, Version)`; add `NodeSummaryIndex` and `EdgeSummaryIndex` CFs; update `ALL_COLUMN_FAMILIES` |
| **mutation.rs** | Update `AddNode`/`AddEdge` to set version=1 and write index; add `UpdateNode`/`UpdateEdge` with optimistic locking |
| **query.rs** | Add `all_nodes_for_summary()`, `current_nodes_for_summary()`, `node_versions_for_summary()`, and edge equivalents; update `get_node_summary()`/`get_edge_summary()` for versioned keys |
| **gc.rs** | New file: `GraphGcConfig`, `GraphGcMetrics`, `GraphGarbageCollector` with background worker |
| **tests.rs** | Test optimistic locking (version mismatch); test multi-version resolution; test GC cleanup |

---

# Part 5: Summary

| Aspect | Design |
|--------|--------|
| **Reverse index** | `(SummaryHash, EntityKey, Version)` → empty |
| **Multi-version support** | Index returns all versions; query API provides `all_*` and `current_*` variants |
| **Content storage** | Keyed by `(EntityId, Version)` — enables clean GC |
| **Optimistic locking** | Version in entity value; reject update if mismatch |
| **GC** | Delete old versions beyond retention; delete stale index entries |

---

# Alternative Design

Author: Codex

This section proposes targeted changes to reduce read amplification, bound GC cost, and clarify migration/deletion behaviors for the CONTENT-ADDRESS design.

---

## 1) Current-Only Reverse Index (Reduce Read Amplification)

**Problem:** `current_*_for_summary()` scans all index entries for a hash and then point-reads each entity to check if the version is current. This is O(k) reads for k index entries, which can be large for popular hashes.

### Option A: Maintain a “Current” Index

Add a second CF that only contains *current* versions. This avoids per-entry validation reads for the common path (vector search resolution).

**Schema:**

```rust
// Node current-only index
pub struct NodeSummaryCurrentIndexCfKey(
    pub SummaryHash,  // 8 bytes
    pub Id,           // 16 bytes
);

// Edge current-only index
pub struct EdgeSummaryCurrentIndexCfKey(
    pub SummaryHash,  // 8 bytes
    pub SrcId,        // 16 bytes
    pub DstId,        // 16 bytes
    pub NameHash,     // 8 bytes
);
```

**Write path:**

- **Insert:** write to both `*_summary_index` and `*_summary_current_index`.
- **Update:**
  - Insert new `(hash, entity)` into current index.
  - Remove old `(old_hash, entity)` from current index.
- **Delete:** remove `(hash, entity)` from current index.

**Read path:**

- `current_nodes_for_summary()` becomes a prefix scan on the current index only.
- `all_nodes_for_summary()` continues to use the versioned index.

**Pros:**
- Eliminates N point reads in the hot path.
- Keeps full history without slowing vector resolution.

**Cons:**
- Adds write amplification (1 extra CF write per update).
- Requires tracking old hash on update (already present in entity value).

### Option B: “Current Marker” Bit in Index Value

Keep one index CF but store a 1-byte value indicating current or stale. Update flips old entry to stale instead of deleting. This avoids extra CF and keeps write costs smaller than Option A.

**Pros:**
- Single CF.
- Enables cheap current filtering without point reads.

**Cons:**
- Requires random reads of index values during scans.
- Still keeps stale entries; GC must remove stale ones eventually.

---

## 2) Bounded GC: Incremental, Partitioned, and Time-Sliced

**Problem:** Full scans over large CFs can be expensive and cause latency spikes.

### Option A: Incremental Scan with Resume Cursor

Persist a cursor (last processed key) per CF. Each GC cycle scans at most `batch_size` entries and resumes next cycle.

**Schema (for GC metadata):**

```rust
pub struct GcCursorCfKey(pub GcTarget); // e.g., "node_summary_index"
pub struct GcCursorCfValue(pub Vec<u8>); // last key
```

**Behavior:**
- Each cycle: read cursor, scan next `batch_size`, update cursor.
- This bounds work per cycle and amortizes cost.

### Option B: Time-Partitioned Index Keys

Add a short time bucket prefix (e.g., daily) so GC can target old buckets without scanning recent hot partitions.

**Schema:**

```rust
pub struct NodeSummaryIndexCfKey(
    pub DayBucket,    // u32, days since epoch
    pub SummaryHash,
    pub Id,
    pub Version,
);
```

**Behavior:**
- GC can delete entire buckets once all entries are older than retention.
- Avoids full scan by key range deletion if RocksDB supports range delete.

---

## 3) Migration Strategy (Content-Addressed → Entity+Version)

**Problem:** Existing content-addressed summaries must be preserved and made queryable without downtime.

### Recommended Migration Plan

1. **Dual-read:**
   - Read `NodeSummaryCfKey(SummaryHash)` if versioned key not found.
   - This keeps lookups working during migration.
2. **Backfill job:**
   - For each node/edge entity, compute current `summary_hash` and write versioned summary + index entries.
3. **Dual-write:**
   - During migration, write both old and new summary formats (temporary).
4. **Cutover:**
   - Once backfill is complete, disable old format writes and reads.
5. **Cleanup:**
   - Drop old CFs after a safety period.

---

## 4) Delete Semantics and Tombstones

**Problem:** Deletes are not explicitly defined, which affects GC and reverse lookup semantics.

### Option A: Hard Deletes

- Remove entity from HOT CF.
- Remove entries from current index immediately.
- GC removes historic summaries and index entries in time.

**Pros:** simple; storage reclaimable.
**Cons:** loses audit/history unless retained by versioned content and index.

### Option B: Tombstones

Add a `deleted` flag in HOT values:

```rust
pub struct NodeCfValue(
    pub Option<TemporalRange>,
    pub NameHash,
    pub Option<SummaryHash>,
    pub Version,
    pub bool, // deleted
);
```

**Behavior:**
- Tombstoned entities remain queryable for audit/time-travel.
- `current_*_for_summary` filters out deleted entries.
- GC policy decides how long to keep tombstones.

---

## 5) Version Overflow and High-Rate Updates

**Problem:** `u32` is likely sufficient, but extreme update rates could overflow.

### Mitigation Options

- **Overflow policy:** if `Version::MAX` reached, freeze writes or rotate entity key.
- **Widen to `u64`:** small storage increase, removes ambiguity.
- **Modulo + epoch:** store `(epoch, version)` to keep size bounded.

---

## 6) Reverse Index Integrity Checks

**Problem:** With empty values for reverse edges, corruption or partial writes can create inconsistencies.

### Mitigation

Add a periodic repair task:

- Scan forward edges, ensure reverse index entries exist.
- Optionally, drop reverse entries that point to missing forward edges.

This can run at low priority and uses bounded batching (similar to GC).

---

## 7) Summary of Recommended Changes

- Add a **current-only index** CF to eliminate point-read validation in `current_*` lookups.
- Implement **incremental GC with cursors** to bound work per cycle.
- Define a **migration plan** with dual-read/dual-write + backfill.
- Document **delete semantics** (hard delete vs tombstone) and their impact on reverse index and GC.
- Specify **version overflow policy** or upgrade to `u64` if needed.
- Add **reverse index repair** task for integrity.
