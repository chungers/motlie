# VERSIONING: Temporal Graph with Time-Travel and Rollback

## BREAKING CHANGE

This document describes a **breaking schema change** that modifies edge and node key structures. Migration required.

---

## Table of Contents

1. [Overview](#overview)
2. [Design Goals](#design-goals)
3. [Schema Changes](#schema-changes)
4. [Mutation API](#mutation-api)
5. [Query API](#query-api)
6. [Examples: Edges](#examples-edges)
7. [Examples: Nodes](#examples-nodes)
8. [Examples: Fragments](#examples-fragments)
9. [Performance Analysis](#performance-analysis)
10. [Pros and Cons](#pros-and-cons)
11. [Migration Path](#migration-path)

---

## Overview

Enable temporal versioning for the graph database:

| Capability | Before | After |
|------------|--------|-------|
| Time-travel queries | No | Yes |
| Topology rollback | No | Yes |
| Content rollback | No | Yes |
| Multi-edge support | Implicit | Explicit |
| Audit history | Partial | Full |

---

## Design Goals

1. **Time-travel**: Query graph state at any past timestamp
2. **Rollback**: Restore entities to previous state (topology or content)
3. **Audit**: Preserve full history of all changes
4. **Deduplication**: Content-addressed summaries avoid storage blowup
5. **Explicit API**: Disambiguate "add second edge" vs "retarget edge"

---

## Schema Changes

### BREAKING: ValidSince Added to Keys

```rust
// ============================================================
// EDGES - OLD SCHEMA
// ============================================================
ForwardEdgeCfKey(SrcId, DstId, NameHash)              // 40 bytes
ReverseEdgeCfKey(DstId, SrcId, NameHash)              // 40 bytes

// ============================================================
// EDGES - NEW SCHEMA (BREAKING)
// ============================================================
ForwardEdgeCfKey(SrcId, DstId, NameHash, ValidSince)  // 48 bytes (+8)
ReverseEdgeCfKey(DstId, SrcId, NameHash, ValidSince)  // 48 bytes (+8)

// ============================================================
// NODES - OLD SCHEMA
// ============================================================
NodeCfKey(Id)                                          // 16 bytes

// ============================================================
// NODES - NEW SCHEMA (BREAKING)
// ============================================================
NodeCfKey(Id, ValidSince)                              // 24 bytes (+8)
```

### Complete CF Schema

```rust
/// Forward edges with temporal key
ForwardEdges {
    key: (SrcId, DstId, NameHash, ValidSinceMilli),  // 48 bytes
    val: (ValidUntilMilli, Weight, SummaryHash, Version, Deleted),
}

/// Reverse edges for incoming queries
ReverseEdges {
    key: (DstId, SrcId, NameHash, ValidSinceMilli),  // 48 bytes
    val: (ValidUntilMilli),
}

/// Nodes with temporal key
Nodes {
    key: (Id, ValidSinceMilli),                       // 24 bytes
    val: (ValidUntilMilli, NameHash, SummaryHash, Version, Deleted),
}

/// Edge summaries (content-addressed, append-only for rollback)
EdgeSummaries {
    key: SummaryHash,                                 // 8 bytes
    val: EdgeSummary,                                 // No RefCount decrement
}

/// Node summaries (content-addressed, append-only for rollback)
NodeSummaries {
    key: SummaryHash,                                 // 8 bytes
    val: NodeSummary,                                 // No RefCount decrement
}

/// Edge version history (for content rollback)
EdgeVersionHistory {
    key: (SrcId, DstId, NameHash, ValidSince, Version),  // 52 bytes
    val: SummaryHash,                                     // 8 bytes
}

/// Node version history (for content rollback)
NodeVersionHistory {
    key: (Id, ValidSince, Version),                   // 28 bytes
    val: SummaryHash,                                 // 8 bytes
}

/// Edge fragments (unchanged - already temporal via timestamp)
EdgeFragments {
    key: (SrcId, DstId, NameHash, TimestampMilli),    // 48 bytes
    val: (TemporalRange, FragmentContent),
}

/// Node fragments (unchanged - already temporal via timestamp)
NodeFragments {
    key: (Id, TimestampMilli),                        // 24 bytes
    val: (TemporalRange, FragmentContent),
}
```

### Temporal Semantics

| Field | Type | Meaning |
|-------|------|---------|
| `valid_since` | `u64` | Timestamp when entity became valid (part of key) |
| `valid_until` | `Option<u64>` | Timestamp when entity stopped being valid (`None` = still valid) |
| `version` | `u32` | Monotonic counter for content changes within a temporal range |

**Entity is current if:** `valid_until.is_none() || valid_until > now()`

**Entity is valid at time T if:** `valid_since <= T && (valid_until.is_none() || valid_until > T)`

---

## Mutation API

### Edge Mutations

```rust
/// Create a new edge.
/// Fails if (src, dst, name) already has a current edge.
pub struct AddEdge {
    pub src: Id,
    pub dst: Id,
    pub name: String,
    pub summary: EdgeSummary,
    pub weight: Option<f64>,
    pub valid_since: Option<TimestampMilli>,  // default: now
}

/// Update edge: change topology (dst/name) and/or content (summary/weight).
/// If topology changes: closes old edge, creates new edge.
/// If only content changes: updates in place.
/// All operations in single transaction.
pub struct UpdateEdge {
    // Identity of edge to change
    pub src: Id,
    pub dst: Id,
    pub name: String,

    // Topology changes (None = keep current)
    pub new_dst: Option<Id>,
    pub new_name: Option<String>,

    // Content changes (None = keep current)
    pub new_summary: Option<EdgeSummary>,
    pub new_weight: Option<Option<f64>>,  // Some(None) = clear weight

    // Optimistic locking
    pub expected_version: Version,
}

/// Soft delete: sets valid_until = now.
/// Edge remains queryable via time-travel.
pub struct DeleteEdge {
    pub src: Id,
    pub dst: Id,
    pub name: String,
    pub expected_version: Version,
}

/// Restore edge to state at a previous time.
/// Creates NEW edge with topology/content from the past.
pub struct RestoreEdge {
    pub src: Id,
    pub dst: Id,
    pub name: String,
    pub as_of: TimestampMilli,
}

/// Rollback all outgoing edges from src to state at a previous time.
pub struct RollbackEdges {
    pub src: Id,
    pub name: Option<String>,  // None = all edge names
    pub as_of: TimestampMilli,
}
```

### Node Mutations

```rust
/// Create a new node.
pub struct AddNode {
    pub id: Id,
    pub name: String,
    pub summary: NodeSummary,
    pub valid_since: Option<TimestampMilli>,
}

/// Update node content.
pub struct UpdateNode {
    pub id: Id,
    pub new_name: Option<String>,
    pub new_summary: Option<NodeSummary>,
    pub expected_version: Version,
}

/// Soft delete node.
pub struct DeleteNode {
    pub id: Id,
    pub expected_version: Version,
}

/// Restore node to state at a previous time.
pub struct RestoreNode {
    pub id: Id,
    pub as_of: TimestampMilli,
}
```

### Fragment Mutations (Unchanged)

Fragments are already temporal via their timestamp key:

```rust
/// Add fragment to edge (append-only, no versioning needed)
pub struct AddEdgeFragment {
    pub src: Id,
    pub dst: Id,
    pub name: String,
    pub content: FragmentContent,
    pub valid_range: Option<TemporalRange>,
}

/// Add fragment to node (append-only, no versioning needed)
pub struct AddNodeFragment {
    pub id: Id,
    pub content: FragmentContent,
    pub valid_range: Option<TemporalRange>,
}
```

### Transaction Logic: UpdateEdge

```rust
fn execute_update_edge(mutation: UpdateEdge) -> Result<Version> {
    let txn = db.transaction();
    let now = now();

    // 1. Find current edge
    let old_edge = find_current_edge(txn, mutation.src, mutation.dst, mutation.name)?;

    // 2. Optimistic lock check
    if old_edge.version != mutation.expected_version {
        return Err(Error::VersionMismatch);
    }

    let topology_changed = mutation.new_dst.is_some() || mutation.new_name.is_some();
    let new_dst = mutation.new_dst.unwrap_or(mutation.dst);
    let new_name = mutation.new_name.unwrap_or(mutation.name);
    let new_summary = mutation.new_summary.unwrap_or(old_edge.summary);
    let new_hash = hash(&new_summary);

    if topology_changed {
        // TOPOLOGY CHANGE: close old, create new

        // Close old edges
        txn.put(ForwardEdges,
            key: (mutation.src, mutation.dst, mutation.name, old_edge.valid_since),
            val: (valid_until: now, ...old_edge));
        txn.put(ReverseEdges,
            key: (mutation.dst, mutation.src, mutation.name, old_edge.valid_since),
            val: (valid_until: now));

        // Create new edges
        txn.put(ForwardEdges,
            key: (mutation.src, new_dst, new_name, now),  // NEW key
            val: (valid_until: None, new_hash, version: 1));
        txn.put(ReverseEdges,
            key: (new_dst, mutation.src, new_name, now),
            val: (valid_until: None));

        // Write history for new edge
        txn.put(EdgeVersionHistory,
            key: (mutation.src, new_dst, new_name, now, 1),
            val: new_hash);

        new_version = 1;  // New edge starts at version 1
    } else {
        // CONTENT ONLY: update in place

        let new_version = old_edge.version + 1;
        txn.put(ForwardEdges,
            key: (mutation.src, mutation.dst, mutation.name, old_edge.valid_since),
            val: (valid_until: None, new_hash, version: new_version, ...));

        // Write history
        txn.put(EdgeVersionHistory,
            key: (mutation.src, mutation.dst, mutation.name, old_edge.valid_since, new_version),
            val: new_hash);
    }

    // Put summary (idempotent, content-addressed)
    txn.put(EdgeSummaries, new_hash, new_summary);

    // Update summary index
    txn.put(EdgeSummaryIndex, (old_edge.hash, ..., old_edge.version), STALE);
    txn.put(EdgeSummaryIndex, (new_hash, ..., new_version), CURRENT);

    txn.commit()?;
    Ok(new_version)
}
```

---

## Query API

### Current State Queries

```rust
/// Get node by ID (current state)
pub struct NodeById { pub id: Id }

/// Get outgoing edges from node (current state)
pub struct OutgoingEdges {
    pub src: Id,
    pub name: Option<String>,
}

/// Get incoming edges to node (current state)
pub struct IncomingEdges {
    pub dst: Id,
    pub name: Option<String>,
}
```

### Time-Travel Queries

```rust
/// Get node at specific time
pub struct NodeByIdAt {
    pub id: Id,
    pub at: TimestampMilli,
}

/// Get outgoing edges at specific time
pub struct OutgoingEdgesAt {
    pub src: Id,
    pub name: Option<String>,
    pub at: TimestampMilli,
}

/// Get incoming edges at specific time
pub struct IncomingEdgesAt {
    pub dst: Id,
    pub name: Option<String>,
    pub at: TimestampMilli,
}

/// Get edge content at specific version
pub struct EdgeAtVersion {
    pub src: Id,
    pub dst: Id,
    pub name: String,
    pub version: Version,
}

/// Get full history of an edge
pub struct EdgeHistory {
    pub src: Id,
    pub dst: Id,
    pub name: String,
}
// Returns: Vec<(ValidSince, ValidUntil, Version, Summary)>

/// Get fragments in time range
pub struct EdgeFragmentsInRange {
    pub src: Id,
    pub dst: Id,
    pub name: String,
    pub start: TimestampMilli,
    pub end: TimestampMilli,
}
```

---

## Examples: Edges

### Example 1: Multi-Edge (Alice Knows Both Bob AND Carol)

Alice creates friendships with both Bob and Carol (separate edges):

```
t=1000: AddEdge { src: Alice, dst: Bob, name: "knows", summary: "college friends" }
t=2000: AddEdge { src: Alice, dst: Carol, name: "knows", summary: "work friends" }

=== ForwardEdges CF ===
(Alice, Bob,   "knows", 1000) → (until=NULL, hash=0xAAA, v=1)
(Alice, Carol, "knows", 2000) → (until=NULL, hash=0xBBB, v=1)

=== EdgeSummaries CF ===
0xAAA → "college friends"
0xBBB → "work friends"

=== Query: OutgoingEdges { src: Alice, name: "knows" } ===
Result: [
  { dst: Bob,   summary: "college friends" },
  { dst: Carol, summary: "work friends" },
]
// Alice knows BOTH Bob and Carol
```

### Example 2: Topology Change (Alice Retargets from Bob to Carol)

Alice's "best_friend" edge moves from Bob to Carol:

```
t=1000: AddEdge { src: Alice, dst: Bob, name: "best_friend", summary: "besties" }
t=2000: UpdateEdge { src: Alice, dst: Bob, name: "best_friend",
                     new_dst: Some(Carol), expected_version: 1 }

=== ForwardEdges CF ===
(Alice, Bob,   "best_friend", 1000) → (until=2000, hash=0xAAA, v=1)  // CLOSED
(Alice, Carol, "best_friend", 2000) → (until=NULL, hash=0xAAA, v=1)  // CURRENT

=== ReverseEdges CF ===
(Bob,   Alice, "best_friend", 1000) → (until=2000)  // CLOSED
(Carol, Alice, "best_friend", 2000) → (until=NULL)  // CURRENT

=== EdgeSummaries CF ===
0xAAA → "besties"  // Shared by both (content unchanged)

=== Query: OutgoingEdges { src: Alice, name: "best_friend" } ===
Result: [{ dst: Carol, summary: "besties" }]  // Only Carol now

=== Time Travel: OutgoingEdgesAt { src: Alice, name: "best_friend", at: 1500 } ===
Result: [{ dst: Bob, summary: "besties" }]  // Bob at t=1500
```

### Example 3: Content Update (Same Topology, New Summary)

Alice updates her relationship description with Bob:

```
t=1000: AddEdge { src: Alice, dst: Bob, name: "knows", summary: "acquaintances" }
t=2000: UpdateEdge { src: Alice, dst: Bob, name: "knows",
                     new_summary: Some("close friends"), expected_version: 1 }
t=3000: UpdateEdge { src: Alice, dst: Bob, name: "knows",
                     new_summary: Some("best friends"), expected_version: 2 }

=== ForwardEdges CF ===
(Alice, Bob, "knows", 1000) → (until=NULL, hash=0xCCC, v=3)  // Same key, updated

=== EdgeVersionHistory CF ===
(Alice, Bob, "knows", 1000, v=1) → 0xAAA
(Alice, Bob, "knows", 1000, v=2) → 0xBBB
(Alice, Bob, "knows", 1000, v=3) → 0xCCC

=== EdgeSummaries CF (append-only) ===
0xAAA → "acquaintances"   // Preserved for rollback
0xBBB → "close friends"   // Preserved for rollback
0xCCC → "best friends"    // Current

=== Query: EdgeAtVersion { src: Alice, dst: Bob, name: "knows", version: 1 } ===
Result: { summary: "acquaintances" }  // Content at v1
```

### Example 4: Delete and Restore

```
t=1000: AddEdge { src: Alice, dst: Bob, name: "knows", summary: "friends" }
t=2000: DeleteEdge { src: Alice, dst: Bob, name: "knows", expected_version: 1 }
t=3000: RestoreEdge { src: Alice, dst: Bob, name: "knows", as_of: 1500 }

=== ForwardEdges CF ===
(Alice, Bob, "knows", 1000) → (until=2000, hash=0xAAA, v=1)  // Deleted at t=2000
(Alice, Bob, "knows", 3000) → (until=NULL, hash=0xAAA, v=1)  // Restored at t=3000

=== Timeline ===
t=1500: OutgoingEdges → [Bob]     // Before delete
t=2500: OutgoingEdges → []        // After delete
t=3500: OutgoingEdges → [Bob]     // After restore
```

### Example 5: Topology Rollback

Alice's best_friend changed multiple times, then rolls back:

```
t=1000: AddEdge { src: Alice, dst: Bob, name: "best_friend", summary: "besties" }
t=2000: UpdateEdge { new_dst: Some(Carol) }   // Bob → Carol
t=3000: UpdateEdge { new_dst: Some(Dave) }    // Carol → Dave
t=4000: RollbackEdges { src: Alice, name: "best_friend", as_of: 1500 }

=== ForwardEdges CF after t=4000 ===
(Alice, Bob,   "best_friend", 1000) → (until=2000, v=1)  // Historical
(Alice, Carol, "best_friend", 2000) → (until=3000, v=1)  // Historical
(Alice, Dave,  "best_friend", 3000) → (until=4000, v=1)  // Closed by rollback
(Alice, Bob,   "best_friend", 4000) → (until=NULL, v=1)  // NEW: Rollback restored Bob!

=== Timeline ===
t=1500: best_friend → Bob    // Original
t=2500: best_friend → Carol  // After first retarget
t=3500: best_friend → Dave   // After second retarget
t=4500: best_friend → Bob    // After rollback!
```

### Example 6: Content Rollback

Alice wants to revert her description back to an earlier version:

```
t=1000: AddEdge { src: Alice, dst: Bob, name: "knows", summary: "acquaintances" }
t=2000: UpdateEdge { new_summary: Some("friends") }
t=3000: UpdateEdge { new_summary: Some("enemies") }  // Oops!
t=4000: RestoreEdge { src: Alice, dst: Bob, name: "knows", as_of: 2500 }

=== EdgeVersionHistory ===
(Alice, Bob, "knows", 1000, v=1) → 0xAAA ("acquaintances")
(Alice, Bob, "knows", 1000, v=2) → 0xBBB ("friends")
(Alice, Bob, "knows", 1000, v=3) → 0xCCC ("enemies")
(Alice, Bob, "knows", 1000, v=4) → 0xBBB ("friends")  // Rollback reuses old hash!

=== EdgeSummaries ===
0xAAA → "acquaintances"  // Still exists
0xBBB → "friends"        // Reused by rollback
0xCCC → "enemies"        // Orphan, GC will clean up

=== Current State at t=4500 ===
Alice knows Bob: "friends" (v=4)
```

### Example 7: Combined Topology + Content Change

```
t=1000: AddEdge { src: Alice, dst: Bob, name: "knows", summary: "friends" }
t=2000: UpdateEdge { src: Alice, dst: Bob, name: "knows",
                     new_dst: Some(Carol),
                     new_summary: Some("close friends"),
                     expected_version: 1 }

=== ForwardEdges CF ===
(Alice, Bob,   "knows", 1000) → (until=2000, hash=0xAAA, v=1)  // Closed
(Alice, Carol, "knows", 2000) → (until=NULL, hash=0xBBB, v=1)  // New topology + content

=== EdgeSummaries ===
0xAAA → "friends"
0xBBB → "close friends"
```

---

## Examples: Nodes

### Example 8: Node Content Update

```
t=1000: AddNode { id: Alice, name: "person", summary: { bio: "Student" } }
t=2000: UpdateNode { id: Alice, new_summary: Some({ bio: "Engineer" }), expected_version: 1 }
t=3000: UpdateNode { id: Alice, new_summary: Some({ bio: "Manager" }), expected_version: 2 }

=== Nodes CF ===
(Alice, 1000) → (until=NULL, name="person", hash=0xCCC, v=3)

=== NodeVersionHistory ===
(Alice, 1000, v=1) → 0xAAA
(Alice, 1000, v=2) → 0xBBB
(Alice, 1000, v=3) → 0xCCC

=== NodeSummaries ===
0xAAA → { bio: "Student" }
0xBBB → { bio: "Engineer" }
0xCCC → { bio: "Manager" }

=== Time Travel: NodeByIdAt { id: Alice, at: 1500 } ===
Result: { name: "person", bio: "Student" }  // v1 at t=1500
```

### Example 9: Node Delete and Restore

```
t=1000: AddNode { id: Alice, name: "person", summary: { bio: "Engineer" } }
t=2000: DeleteNode { id: Alice, expected_version: 1 }
t=3000: RestoreNode { id: Alice, as_of: 1500 }

=== Nodes CF ===
(Alice, 1000) → (until=2000, v=1)  // Deleted
(Alice, 3000) → (until=NULL, v=1)  // Restored

=== Timeline ===
t=1500: NodeById(Alice) → { bio: "Engineer" }
t=2500: NodeById(Alice) → None (deleted)
t=3500: NodeById(Alice) → { bio: "Engineer" } (restored)
```

---

## Examples: Fragments

Fragments are append-only with timestamp keys. They don't need the ValidSince pattern because each fragment is a distinct event.

### Example 10: Edge Fragments Over Time

```
t=1000: AddEdge { src: Alice, dst: Bob, name: "knows", summary: "friends" }
t=1500: AddEdgeFragment { src: Alice, dst: Bob, name: "knows",
                          content: "Met at conference" }
t=2000: AddEdgeFragment { src: Alice, dst: Bob, name: "knows",
                          content: "Worked on project together" }
t=2500: AddEdgeFragment { src: Alice, dst: Bob, name: "knows",
                          content: "Started company" }

=== EdgeFragments CF ===
(Alice, Bob, "knows", 1500) → "Met at conference"
(Alice, Bob, "knows", 2000) → "Worked on project together"
(Alice, Bob, "knows", 2500) → "Started company"

=== Query: EdgeFragmentsInRange { src: Alice, dst: Bob, name: "knows",
                                   start: 1000, end: 2200 } ===
Result: [
  (1500, "Met at conference"),
  (2000, "Worked on project together"),
]
// Fragment at 2500 excluded (after range)

=== Fragments survive topology changes ===
t=3000: UpdateEdge { new_dst: Some(Carol) }  // Retarget to Carol

// Old fragments still queryable for (Alice, Bob, "knows"):
EdgeFragmentsInRange { src: Alice, dst: Bob, name: "knows", start: 0, end: 9999 }
Result: [all 3 fragments]  // Fragments tied to OLD topology
```

### Example 11: Node Fragments with Time Travel

```
t=1000: AddNode { id: Alice, name: "person", summary: { bio: "Student" } }
t=1500: AddNodeFragment { id: Alice, content: "Graduated college" }
t=2000: UpdateNode { id: Alice, new_summary: Some({ bio: "Engineer" }) }
t=2500: AddNodeFragment { id: Alice, content: "Got first job" }
t=3000: AddNodeFragment { id: Alice, content: "Promoted to senior" }

=== NodeFragments CF ===
(Alice, 1500) → "Graduated college"
(Alice, 2500) → "Got first job"
(Alice, 3000) → "Promoted to senior"

=== Time Travel Query at t=2200 ===
Node: { bio: "Engineer" } (v=2, updated at t=2000)
Fragments up to t=2200: ["Graduated college", "Got first job"]
```

---

## Performance Analysis

### Write Amplification

| Operation | Old Schema | New Schema | Delta |
|-----------|-----------|------------|-------|
| **AddEdge** | 4 puts | 5 puts | +1 (VersionHistory) |
| **UpdateEdge (content only)** | 3 puts | 4 puts | +1 (VersionHistory) |
| **UpdateEdge (topology)** | N/A | 7 puts | New capability |
| **DeleteEdge** | 2 puts | 2 puts | Same |
| **AddNode** | 3 puts | 4 puts | +1 (VersionHistory) |
| **UpdateNode** | 3 puts | 4 puts | +1 (VersionHistory) |

**Detailed breakdown for UpdateEdge (topology change):**
```
1. Update old ForwardEdge (set valid_until)
2. Update old ReverseEdge (set valid_until)
3. Insert new ForwardEdge (new key)
4. Insert new ReverseEdge (new key)
5. Put EdgeSummary (idempotent if unchanged)
6. Put EdgeVersionHistory
7. Update EdgeSummaryIndex (2 entries: STALE + CURRENT)
```

### Read Amplification

| Query | Old Schema | New Schema | Delta |
|-------|-----------|------------|-------|
| **OutgoingEdges** | 1 scan | 1 scan + filter | +filter cost |
| **OutgoingEdgesAt(T)** | N/A | 1 scan + filter | New capability |
| **NodeById** | 1 get | 1 scan + filter | Scan vs get |
| **EdgeHistory** | N/A | 1 scan | New capability |

**Filter cost:** Each edge scan now filters by `valid_until`:
```rust
// Before: just scan
prefix_scan(src)

// After: scan + filter
prefix_scan(src).filter(|e| e.valid_until.is_none() || e.valid_until > now)
```

### Storage Overhead

| Component | Old Size | New Size | Delta |
|-----------|---------|----------|-------|
| **Edge key** | 40 bytes | 48 bytes | +8 bytes (+20%) |
| **Node key** | 16 bytes | 24 bytes | +8 bytes (+50%) |
| **EdgeVersionHistory** | N/A | 52 + 8 = 60 bytes/version | New |
| **NodeVersionHistory** | N/A | 28 + 8 = 36 bytes/version | New |
| **Summaries** | Cleaned inline | Append-only | More storage until GC |

**Example storage for 1M edges with avg 3 versions:**
```
Edge keys:        1M × 8 bytes = 8 MB additional
VersionHistory:   3M × 60 bytes = 180 MB
Total overhead:   ~188 MB for 1M edges
```

### GC Changes

| Aspect | Old (RefCount) | New (Orphan Scan) |
|--------|---------------|-------------------|
| Summary cleanup | Inline (immediate) | Background GC |
| Write cost | +1 get/put per update | None |
| Orphan accumulation | None | Until GC runs |
| Rollback support | No | Yes |

**Orphan GC scan cost:**
```
For each summary hash:
  - Check if ANY index entry references it
  - If none: delete summary
  - O(summaries × index_scan)
```

---

## Pros and Cons

### Pros

| Benefit | Description |
|---------|-------------|
| **Time-travel queries** | Query graph at any past timestamp |
| **Topology rollback** | Restore edge destinations to past state |
| **Content rollback** | Restore summaries to past versions |
| **Full audit history** | All changes preserved with timestamps |
| **Explicit API** | Clear distinction between multi-edge and retarget |
| **Content deduplication** | Summaries still content-addressed |

### Cons

| Drawback | Description | Mitigation |
|----------|-------------|------------|
| **Breaking schema change** | Requires migration | One-time migration script |
| **Larger keys** | +8 bytes per edge/node key | Acceptable for capabilities gained |
| **More write ops** | +1 VersionHistory write per mutation | Batching amortizes cost |
| **Scan vs get for nodes** | NodeById now requires scan | Usually few valid_since values per ID |
| **Orphan accumulation** | Old summaries linger until GC | Background GC, acceptable latency |
| **Query filter overhead** | Every scan filters by valid_until | Minimal CPU cost |

### When NOT to Use

- **High-frequency updates**: If edges update 1000s of times/sec, history accumulates fast
- **Storage-constrained**: History consumes storage until GC
- **No audit requirements**: If you don't need rollback/time-travel, simpler schema is better

---

## Migration Path

### Phase 1: Schema Migration (Breaking)

```rust
// For each existing edge:
let old_key = ForwardEdgeCfKey(src, dst, name);
let old_val = db.get(old_key);

let new_key = ForwardEdgeCfKey(src, dst, name, 0);  // valid_since=0 (epoch)
let new_val = ForwardEdgeCfValue {
    valid_until: None,  // Currently valid
    ...old_val,
};

db.delete(old_key);
db.put(new_key, new_val);

// Same for ReverseEdges, Nodes
```

### Phase 2: New Mutations

1. Implement `UpdateEdge` with topology change support
2. Implement `DeleteEdge` with soft delete
3. Implement `RestoreEdge` and `RollbackEdges`
4. Update all queries to filter by `valid_until`

### Phase 3: GC Updates

1. Remove RefCount decrement from mutations
2. Implement orphan summary GC scan
3. Add VersionHistory writes to all mutations

---

## Summary

| Feature | Status |
|---------|--------|
| Time-travel queries | Enabled via ValidSince in key |
| Topology rollback | Enabled via close-old/create-new pattern |
| Content rollback | Enabled via VersionHistory + append-only summaries |
| Multi-edge support | Explicit via AddEdge (fails if exists) |
| Retarget support | Explicit via UpdateEdge with new_dst |
| Fragments | Unchanged (already temporal) |
| Breaking change | Yes (key structure) |

(claude, 2026-02-04, designed)
