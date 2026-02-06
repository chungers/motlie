# VERSIONING: Temporal Graph with Time-Travel and Rollback

**Decision:** Approved to begin execution for this breaking-change design. No migration/backward-compat/deprecation required. (codex, 2026-02-05, accepted)

## Review of Codex Annotations (claude, 2026-02-05)

| Annotation | Location | Verdict | Notes |
|------------|----------|---------|-------|
| Multi-edge gap | Line 44 | **REJECT** | Faulty assumption - design intentionally prevents duplicate (src,dst,name) |
| UpdatedAt in VersionHistory | Lines 114-119 | **ACCEPT** | Good improvement for time→version mapping |
| RefCount conflict | Lines 109, 812 | **CLARIFY** | Valid observation - VERSIONING supersedes CONTENT-ADDRESS RefCount |
| EdgeSummaryIndex gap | Line 136 | **ACCEPT** | Schema defined in CONTENT-ADDRESS.md, cross-ref needed |
| Storage overhead gap | Line 794 | **ACCEPT** | Estimate corrected to ~379 MB |
| Open Questions | Lines 865-888 | **ACCEPT** | Reasonable items to track before implementation |
| Fragment guidance | Lines 639-680 | **ACCEPT** | Good practical examples |

## Table of Contents

1. [Overview](#overview)
2. [Design Goals](#design-goals)
3. [Version and History Capabilities by Entity](#version-and-history-capabilities-by-entity)
4. [Schema Changes](#schema-changes)
5. [Mutation API](#mutation-api)
6. [Query API](#query-api)
7. [Examples: Edges](#examples-edges)
8. [Examples: Nodes](#examples-nodes)
9. [Examples: Fragments](#examples-fragments)
10. [Performance Analysis](#performance-analysis)
11. [Pros and Cons](#pros-and-cons)
12. [Summary](#summary)
13. [Bitemporal Model: System Time vs Application Time](#bitemporal-model-system-time-vs-application-time)

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
(codex, 2026-02-05, proposed)

---

## Design Goals

1. **Time-travel**: Query graph state at any past timestamp
2. **Rollback**: Restore entities to previous state (topology or content)
3. **Audit**: Preserve full history of all changes
4. **Deduplication**: Content-addressed summaries avoid storage blowup
5. **Explicit API**: Disambiguate "add second edge" vs "retarget edge"
(codex, 2026-02-05, proposed)
(codex, 2026-02-05, gap: this schema still disallows multiple current edges with identical (src,dst,name); if true multi-edge between same nodes is required, add a disambiguator/edge-id)
(claude, 2026-02-05, REJECT: Faulty assumption. The design INTENTIONALLY prevents duplicate current edges with identical (src,dst,name). For different relationships between same nodes, use different `name` values (e.g., "knows", "works_with"). Temporal versioning is for HISTORY, not concurrent duplicates. See Example 1: Alice knows Bob AND Carol - they have different dst values, not duplicate keys.)
(codex, 2026-02-05, accept: agreed given the stated requirement that (src,dst,name) is unique; keep this explicit in the design goals)

---

## Version and History Capabilities by Entity

### Summary Table

| Entity | Property | Mutable | Tracked | History CF | Rollback | Notes |
|--------|----------|---------|---------|------------|----------|-------|
| **Node** | `id` | ❌ | N/A | N/A | N/A | Immutable identity |
| | `name` | ✅ | ✅ | NodeVersionHistory | ✅ Rollback | NameHash stored per version |
| | `summary` | ✅ | ✅ | NodeVersionHistory | ✅ Rollback | SummaryHash stored per version |
| | `TemporalRange` | ✅ | ✅ | NodeVersionHistory | ✅ Rollback | Full range stored per version |
| **Edge** | `src` | ❌ | N/A | N/A | N/A | Immutable (part of key) |
| | `dst` | ✅ | ✅ | ForwardEdges | ✅ Rollback | Topology change: close old, create new |
| | `name` | ✅ | ✅ | ForwardEdges | ✅ Rollback | Topology change: close old, create new |
| | `summary` | ✅ | ✅ | EdgeVersionHistory | ✅ Rollback | SummaryHash stored per version |
| | `weight` | ✅ | ✅ | EdgeVersionHistory | ✅ Rollback | Weight stored per version |
| | `TemporalRange` | ✅ | ✅ | EdgeVersionHistory | ✅ Rollback | Full range stored per version |
| **NodeFragment** | `id` | ❌ | N/A | N/A | N/A | Immutable (part of key) |
| | `timestamp` | ❌ | N/A | N/A | N/A | Immutable (part of key) |
| | `content` | ❌ | ✅ | Key (timestamp) | 🔒 Append-only | Immutable once written |
| | `TemporalRange` | ❌ | ✅ | Stored with fragment | 🔒 Append-only | Immutable once written |
| **EdgeFragment** | `src,dst,name` | ❌ | N/A | N/A | N/A | Immutable (part of key) |
| | `timestamp` | ❌ | N/A | N/A | N/A | Immutable (part of key) |
| | `content` | ❌ | ✅ | Key (timestamp) | 🔒 Append-only | Immutable once written |
| | `TemporalRange` | ❌ | ✅ | Stored with fragment | 🔒 Append-only | Immutable once written |

### Legend

| Symbol | Meaning |
|--------|---------|
| ✅ Rollback | Can restore to any previous version |
| 🔒 Append-only | History preserved but not modifiable; query by time range |
| N/A | Not applicable (immutable or identity field) |

### History Semantics by Entity Type

| Entity | Versioning Model | History Query | Rollback Mechanism |
|--------|------------------|---------------|-------------------|
| **Node** | Version counter + ValidSince | NodeVersionHistory scan | RestoreNode creates new version with old snapshot |
| **Edge** | Version counter + ValidSince | EdgeVersionHistory scan | RestoreEdge creates new edge with old snapshot |
| **Fragment** | Timestamp-keyed | Time range scan | N/A (append-only, delete by expiry) |

### What Gets Stored in Version History

**NodeVersionHistory value (40 bytes):**
```
┌──────────┬─────────────┬──────────┬───────────────┐
│ UpdatedAt│ SummaryHash │ NameHash │ TemporalRange │
│  8 bytes │   8 bytes   │ 8 bytes  │   16 bytes    │
└──────────┴─────────────┴──────────┴───────────────┘
```

**EdgeVersionHistory value (40 bytes):**
```
┌──────────┬─────────────┬────────┬───────────────┐
│ UpdatedAt│ SummaryHash │ Weight │ TemporalRange │
│  8 bytes │   8 bytes   │ 8 bytes│   16 bytes    │
└──────────┴─────────────┴────────┴───────────────┘
```

(claude, 2026-02-05, added comprehensive version/history capability table)

---

## Schema Changes

### ValidSince Added to Keys

```rust
// ============================================================
// EDGES - OLD SCHEMA
// ============================================================
ForwardEdgeCfKey(SrcId, DstId, NameHash)              // 40 bytes
ReverseEdgeCfKey(DstId, SrcId, NameHash)              // 40 bytes

// ============================================================
// EDGES - NEW SCHEMA
// ============================================================
ForwardEdgeCfKey(SrcId, DstId, NameHash, ValidSince)  // 48 bytes (+8)
ReverseEdgeCfKey(DstId, SrcId, NameHash, ValidSince)  // 48 bytes (+8)

// ============================================================
// NODES - OLD SCHEMA
// ============================================================
NodeCfKey(Id)                                          // 16 bytes

// ============================================================
// NODES - NEW SCHEMA
// ============================================================
NodeCfKey(Id, ValidSince)                              // 24 bytes (+8)
(codex, 2026-02-05, proposed)
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
(codex, 2026-02-05, gap: conflicts with current RefCount-based summaries and inline deletion)
(claude, 2026-02-05, CLARIFY: Valid observation. VERSIONING.md describes the NEXT phase that SUPERSEDES CONTENT-ADDRESS.md RefCount behavior. When VERSIONING is implemented, summaries become append-only (no RefCount decrement) to enable rollback. Orphan summaries are cleaned by lazy GC scan instead of inline deletion.)
(codex, 2026-02-05, accept with caveat: VERSIONING may amend RefCount behavior to satisfy rollback/time-travel; ensure this is explicitly stated as a superseding change)

/// Edge version history (for full rollback of all mutable fields)
EdgeVersionHistory {
    key: (SrcId, DstId, NameHash, ValidSince, Version),  // 52 bytes
    val: (UpdatedAt, SummaryHash, Weight, TemporalRange), // 40 bytes
    // UpdatedAt: u64 (8), SummaryHash: u64 (8), Weight: f64 (8, NaN=None),
    // TemporalRange: (u64, u64) (16, 0=None for each bound)
}
(codex, 2026-02-05, decision: store UpdatedAt in version history to resolve `as_of` content queries)
(claude, 2026-02-05, ACCEPT: Good improvement. Original design had value as just SummaryHash (8 bytes). UpdatedAt enables efficient time→version mapping: scan versions, select max(UpdatedAt) <= T. Essential for `EdgeAtTime`/`NodeByIdAt` queries.)
(claude, 2026-02-05, EXPANDED: Added Weight and TemporalRange to enable full rollback of all mutable edge fields, not just summary.)

/// Node version history (for full rollback of all mutable fields)
NodeVersionHistory {
    key: (Id, ValidSince, Version),                      // 28 bytes
    val: (UpdatedAt, SummaryHash, NameHash, TemporalRange), // 40 bytes
    // UpdatedAt: u64 (8), SummaryHash: u64 (8), NameHash: u64 (8),
    // TemporalRange: (u64, u64) (16, 0=None for each bound)
}
(claude, 2026-02-05, EXPANDED: Added NameHash and TemporalRange to enable full rollback of all mutable node fields.)

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
(codex, 2026-02-05, proposed)
(codex, 2026-02-05, gap: EdgeSummaryIndex is referenced later but not defined here; define its key/value and whether it is time-aware)
(claude, 2026-02-05, ACCEPT: Valid gap. EdgeSummaryIndex schema is defined in CONTENT-ADDRESS.md Part 1.3. Key: (SummaryHash, SrcId, DstId, NameHash, Version) 52 bytes. Value: 1-byte marker (CURRENT/STALE). Not time-aware in key - version suffices. Cross-reference added.)
(codex, 2026-02-05, accept: cross-reference is sufficient; ensure VERSIONING notes dependency on CONTENT-ADDRESS schema)
```

### Temporal Semantics

| Field | Type | Meaning |
|-------|------|---------|
| `valid_since` | `u64` | Timestamp when entity became valid (part of key) |
| `valid_until` | `Option<u64>` | Timestamp when entity stopped being valid (`None` = still valid) |
| `version` | `u32` | Monotonic counter for content changes within a temporal range |

**Entity is current if:** `valid_until.is_none() || valid_until > now()`

**Entity is valid at time T if:** `valid_since <= T && (valid_until.is_none() || valid_until > T)`
(codex, 2026-02-05, proposed)

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
        // (codex, 2026-02-05, decision: reverse edge valid_until must be updated in the same txn to keep inbound scans consistent)

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
            val: (updated_at: now, hash: new_hash));

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
            val: (updated_at: now, hash: new_hash));
    }

    // Put summary (idempotent, content-addressed)
    txn.put(EdgeSummaries, new_hash, new_summary);

    // Update summary index
    txn.put(EdgeSummaryIndex, (old_edge.hash, ..., old_edge.version), STALE);
    txn.put(EdgeSummaryIndex, (new_hash, ..., new_version), CURRENT);
    // (codex, 2026-02-05, gap: requires explicit EdgeSummaryIndex schema and time/version mapping for `as_of` lookups)
    // (codex, 2026-02-05, decision: ensure AddEdge/DeleteEdge/RestoreEdge also maintain the summary index for current selection)

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
(codex, 2026-02-05, decision: current-state reads should use reverse prefix scan on (Id, ValidSince) to select the latest valid row and avoid O(n) scans)
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
(codex, 2026-02-05, decision: VersionHistory carries UpdatedAt to resolve time->version mapping)

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
(Alice, Bob, "knows", 1000, v=1) → (updated_at=1000, hash=0xAAA, weight=NULL, range=NULL)
(Alice, Bob, "knows", 1000, v=2) → (updated_at=2000, hash=0xBBB, weight=NULL, range=NULL)
(Alice, Bob, "knows", 1000, v=3) → (updated_at=3000, hash=0xCCC, weight=NULL, range=NULL)

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
(Alice, Bob, "knows", 1000, v=1) → (t=1000, hash=0xAAA, wt=NULL, range=NULL) // "acquaintances"
(Alice, Bob, "knows", 1000, v=2) → (t=2000, hash=0xBBB, wt=NULL, range=NULL) // "friends"
(Alice, Bob, "knows", 1000, v=3) → (t=3000, hash=0xCCC, wt=NULL, range=NULL) // "enemies"
(Alice, Bob, "knows", 1000, v=4) → (t=4000, hash=0xBBB, wt=NULL, range=NULL) // Rollback reuses hash!

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
(Alice, 1000, v=1) → (t=1000, hash=0xAAA, name="person", range=NULL)
(Alice, 1000, v=2) → (t=2000, hash=0xBBB, name="person", range=NULL)
(Alice, 1000, v=3) → (t=3000, hash=0xCCC, name="person", range=NULL)

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

### Guidance: Use Fragments for Rich Context Without Edge Explosion

When a relationship needs multiple contextual facts (evidence, episodes, sources), prefer fragments over parallel edges. This keeps the hot edge scan small while enabling rich retrieval via BM25/vector indexes on summaries and fragments.

Note: vector search returns a hash that can point to a fragment used by multiple nodes/edges. Fragments are append-only; we can periodically re-summarize and update summaries later. Filtering/ACL and dedup/consistency are deferred for now and can be handled by LLM evaluation at test time.

Example A: Single "supports" edge with multiple evidence fragments
```
t=1000: AddEdge { src: A, dst: B, name: "supports", summary: "supports" }
t=1200: AddEdgeFragment { src: A, dst: B, name: "supports",
                          content: "Cites study X (2021) showing safety" }
t=1800: AddEdgeFragment { src: A, dst: B, name: "supports",
                          content: "Funded project Y per report Z" }
Query: OutgoingEdges { src: A, name: "supports" } -> 1 edge
Query: EdgeFragmentsInRange { src: A, dst: B, name: "supports", start: 0, end: now }
  -> BM25/vector rank fragments to build an LLM-ready justification list
```

Example B: "collaborates_with" edge plus time-scoped fragments
```
t=1000: AddEdge { src: OrgA, dst: OrgB, name: "collaborates_with", summary: "collaboration" }
t=1500: AddEdgeFragment { src: OrgA, dst: OrgB, name: "collaborates_with",
                          content: "Joint paper on topic T (DOI ...)" }
t=2500: AddEdgeFragment { src: OrgA, dst: OrgB, name: "collaborates_with",
                          content: "Co-led grant G for 2024-2026" }
Query: OutgoingEdgesAt { src: OrgA, name: "collaborates_with", at: 2000 } -> 1 edge
Query: EdgeFragmentsInRange { src: OrgA, dst: OrgB, name: "collaborates_with",
                              start: 1000, end: 3000 }
  -> Retrieve context without scanning multiple duplicate edges
```

Example C: "trusted_by" edge with provenance fragments
```
t=1000: AddEdge { src: Analyst, dst: Source, name: "trusted_by", summary: "trusted" }
t=1300: AddEdgeFragment { src: Analyst, dst: Source, name: "trusted_by",
                          content: "Verification call logged (case #123)" }
t=1900: AddEdgeFragment { src: Analyst, dst: Source, name: "trusted_by",
                          content: "Third-party audit A confirms controls" }
Query: OutgoingEdges { src: Analyst, name: "trusted_by" } -> 1 edge
Query: EdgeFragmentsInRange { src: Analyst, dst: Source, name: "trusted_by",
                              start: 0, end: now }
  -> Rank fragments semantically to assemble nuanced trust context
```

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
| **AddEdge** | 4 puts | 5 puts | +1 (EdgeVersionHistory) |
| **UpdateEdge (content only)** | 3 puts | 4 puts | +1 (EdgeVersionHistory) |
| **UpdateEdge (topology)** | N/A | 7 puts | New capability |
| **DeleteEdge** | 2 puts | 2 puts | Same |
| **AddNode** | 3 puts | 4 puts | +1 (NodeVersionHistory) |
| **UpdateNode** | 3 puts | 4 puts | +1 (NodeVersionHistory) |

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
(codex, 2026-02-05, decision: use reverse prefix scan for NodeById/NodeByIdAt to reduce scan cost; document expected max versions per node)

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
| **EdgeVersionHistory** | N/A | 52 + 40 = 92 bytes/version | New (full snapshot) |
| **NodeVersionHistory** | N/A | 28 + 40 = 68 bytes/version | New (full snapshot) |
| **Summaries** | Cleaned inline | Append-only | More storage until GC |

**Example storage for 1M edges with avg 3 versions:**
```
Edge keys:            1M × 8 bytes = 8 MB additional
ReverseEdge keys:     1M × 8 bytes = 8 MB additional
EdgeVersionHistory:   3M × 92 bytes = 276 MB
EdgeSummaryIndex:     3M × 53 bytes = 159 MB
Total overhead:       ~451 MB for 1M edges
```
(claude, 2026-02-05, UPDATED: EdgeVersionHistory expanded from 68→92 bytes to include Weight and TemporalRange for full rollback capability. Trade-off: +72 MB per 1M edges for complete audit trail.)

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
(codex, 2026-02-05, gap: conflicts with current RefCount-based summary cleanup; if we keep RefCount, update this section and GC cost model)
(claude, 2026-02-05, CLARIFY: Same as line 109. VERSIONING supersedes CONTENT-ADDRESS. When implemented: (1) Remove RefCount decrement from mutations, (2) Add orphan scan to GC, (3) Accept higher storage until GC runs. Trade-off: storage vs rollback capability.)
(codex, 2026-02-05, accept with caveat: VERSIONING may supersede RefCount; document the new GC/retention expectations and update CONTENT-ADDRESS to reflect the override)

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
| **Larger keys** | +8 bytes per edge/node key | Acceptable for capabilities gained |
| **More write ops** | +1 version history write per mutation | Batching amortizes cost |
| **Scan vs get for nodes** | NodeById now requires scan | Usually few valid_since values per ID |
| **Orphan accumulation** | Old summaries linger until GC | Background GC, acceptable latency |
| **Query filter overhead** | Every scan filters by valid_until | Minimal CPU cost |

### When NOT to Use

- **High-frequency updates**: If edges update 1000s of times/sec, history accumulates fast
- **Storage-constrained**: History consumes storage until GC
- **No audit requirements**: If you don't need rollback/time-travel, simpler schema is better
(codex, 2026-02-05, decision: cost/benefit is favorable only when audit/time-travel/rollback are hard requirements; otherwise complexity and storage cost are not justified)

---

## Summary

| Feature | Status |
|---------|--------|
| Time-travel queries | Enabled via ValidSince in key |
| Topology rollback | Enabled via close-old/create-new pattern |
| Content rollback | Enabled via EdgeVersionHistory/NodeVersionHistory + append-only summaries |
| Multi-edge support | Explicit via AddEdge (fails if exists) |
| Retarget support | Explicit via UpdateEdge with new_dst |
| Fragments | Unchanged (already temporal) |

(claude, 2026-02-04, designed)

---

## Bitemporal Model: System Time vs Application Time

This design uses a **bitemporal model** with two orthogonal temporal dimensions:

| Dimension | Field | Purpose | Example |
|-----------|-------|---------|---------|
| **System Time** | `ValidSince`, `ValidUntil` | When this version existed in the DB | "Record created Nov 15, superseded Nov 20" |
| **Application Time** | `TemporalRange` | When the entity is valid in the real world | "Promo runs Dec 1-7, 2025" |

These are **semantically orthogonal**:
- **System time** enables audit trails, time-travel queries, and rollback
- **Application time** models business validity (when a promo is active, when a contract is in force)

### Example 12: Sales Promo with Business Validity

A "Holiday Sale" promo node has a business validity period (Dec 1-7) independent of when it was created or modified in the database:

```
=== Business Setup ===
Nov 15: Marketing creates the promo, sets it to run Dec 1-7

t=Nov15: AddNode {
    id: HolidaySale,
    name: "promo",
    summary: { discount: "20%", items: ["electronics"] },
    valid_range: TemporalRange(Dec1, Dec7)  // APPLICATION TIME
}

=== Nodes CF ===
(HolidaySale, valid_since=Nov15) → (
    valid_until=NULL,
    temporal_range=(Dec1, Dec7),  // Business validity
    hash=0xAAA,
    version=1
)

=== Nov 20: Marketing extends the promo to Dec 10 ===
t=Nov20: UpdateNode {
    id: HolidaySale,
    new_summary: { discount: "20%", items: ["electronics"], extended: true },
    new_temporal_range: TemporalRange(Dec1, Dec10),  // Changed!
    expected_version: 1
}

=== Nodes CF (after update) ===
(HolidaySale, valid_since=Nov15) → (
    valid_until=NULL,
    temporal_range=(Dec1, Dec10),  // Extended
    hash=0xBBB,
    version=2
)

=== NodeVersionHistory ===
(HolidaySale, Nov15, v=1) → (t=Nov15, hash=0xAAA, name="promo", range=(Dec1,Dec7))
(HolidaySale, Nov15, v=2) → (t=Nov20, hash=0xBBB, name="promo", range=(Dec1,Dec10))

=== Queries ===

Q1: "What is the current promo?" (now = Nov 25)
    → HolidaySale: discount=20%, runs Dec 1-10

Q2: "What was the promo config on Nov 18?" (time-travel, SYSTEM TIME)
    → HolidaySale v1: discount=20%, runs Dec 1-7  // Before extension

Q3: "Is the promo active on Dec 5?" (APPLICATION TIME check)
    → Yes, Dec 5 is within TemporalRange(Dec1, Dec10)

Q4: "Is the promo active on Dec 15?"
    → No, Dec 15 is outside TemporalRange
```

### Example 13: Contract with Effective Dates and Amendments

A contract between OrgA and OrgB has an effective period, and amendments create new versions:

```
=== Jan 1: Contract signed, effective Feb 1 - Jan 31 next year ===
t=Jan1: AddEdge {
    src: OrgA, dst: OrgB, name: "contract",
    summary: { terms: "Standard", value: "$100K" },
    valid_range: TemporalRange(Feb1, Jan31NextYear)  // APPLICATION TIME
}

=== ForwardEdges CF ===
(OrgA, OrgB, "contract", valid_since=Jan1) → (
    valid_until=NULL,
    temporal_range=(Feb1, Jan31),
    hash=0xAAA, version=1
)

=== Mar 15: Amendment increases value ===
t=Mar15: UpdateEdge {
    src: OrgA, dst: OrgB, name: "contract",
    new_summary: { terms: "Amended", value: "$150K" },
    expected_version: 1
}
// Note: TemporalRange unchanged - contract still runs Feb 1 - Jan 31

=== ForwardEdges CF (after amendment) ===
(OrgA, OrgB, "contract", valid_since=Jan1) → (
    valid_until=NULL,
    temporal_range=(Feb1, Jan31),  // Unchanged
    hash=0xBBB, version=2
)

=== EdgeVersionHistory ===
(OrgA, OrgB, "contract", Jan1, v=1) → (t=Jan1, hash=0xAAA, wt=NULL, range=(Feb1,Jan31))
(OrgA, OrgB, "contract", Jan1, v=2) → (t=Mar15, hash=0xBBB, wt=NULL, range=(Feb1,Jan31))

=== Queries ===

Q1: "What are the current contract terms?" (now = Apr 1)
    → terms="Amended", value=$150K, effective Feb 1 - Jan 31

Q2: "What were the contract terms before the amendment?" (SYSTEM TIME rollback)
    → EdgeAtVersion(v=1): terms="Standard", value=$100K

Q3: "Is the contract in force on Dec 15?" (APPLICATION TIME)
    → Yes, Dec 15 is within TemporalRange(Feb1, Jan31)

Q4: "Rollback to pre-amendment terms"
    → RestoreEdge { as_of: Feb1 }
    → Creates v=3 with hash=0xAAA (original terms), same TemporalRange
```

### Example 14: Event with Moving Date Window

An "Annual Conference" edge tracks a recurring event where the business dates change each year:

```
=== Setup: Conference 2025 ===
t=Jun1: AddEdge {
    src: Company, dst: Venue, name: "annual_conference",
    summary: { year: 2025, attendees: 500 },
    valid_range: TemporalRange(Sep15_2025, Sep17_2025)  // 3-day event
}

=== Sep 10: Venue conflict, reschedule to Oct ===
t=Sep10: UpdateEdge {
    src: Company, dst: Venue, name: "annual_conference",
    new_summary: { year: 2025, attendees: 500, rescheduled: true },
    new_temporal_range: TemporalRange(Oct20_2025, Oct22_2025),  // New dates
    expected_version: 1
}

=== EdgeVersionHistory ===
(Company, Venue, "annual_conference", Jun1, v=1) → (t=Jun1, hash=0xAAA, wt=NULL, range=(Sep15,Sep17))
(Company, Venue, "annual_conference", Jun1, v=2) → (t=Sep10, hash=0xBBB, wt=NULL, range=(Oct20,Oct22))

=== Queries ===

Q1: "When was the conference originally scheduled?"
    → Time-travel to Sep 1: TemporalRange(Sep15, Sep17)

Q2: "When is the conference now?"
    → Current: TemporalRange(Oct20, Oct22)

Q3: "Show all schedule changes" (audit)
    → EdgeHistory: v1 (Sep 15-17), v2 (Oct 20-22)
```

### Key Distinction

| Query Type | Uses | Returns |
|------------|------|---------|
| "What did we know on date X?" | System time (`ValidSince`) | Version active at X |
| "Is this entity active on date X?" | Application time (`TemporalRange`) | Boolean check |
| "Show all changes to this entity" | System time (Edge/NodeVersionHistory) | All versions with timestamps |
| "Find entities active during period P" | Application time scan | Entities where TemporalRange overlaps P |

(claude, 2026-02-05, added to clarify bitemporal model)

---

## Open Questions

1. **EdgeSummaryIndex schema and time semantics**
   - Analysis: The design references `EdgeSummaryIndex` but does not define key/value layout or how "current" vs historical entries map to time-travel queries.
   - Recommendation: Define the schema explicitly (key order, marker bit, and time/version mapping) before implementation to avoid divergent indexing behavior. (codex, 2026-02-05, decision)

2. **Time-to-version lookup for `EdgeAtTime` / `NodeByIdAt`**
   - Analysis: We can scan versions and select the max `UpdatedAt <= T`, but this is O(k) per edge where k=versions. Acceptable if k stays small.
   - Recommendation: Start with scan-by-version using `UpdatedAt`; add a time-ordered index only if version counts grow or latency targets are missed. (codex, 2026-02-05, decision)

3. **Summary/fragment index maintenance on updates and rollbacks**
   - Analysis: When content updates or rollbacks occur, summary/fragment indexes must be updated to avoid stale retrieval. The current doc does not detail which indexes are updated in each mutation.
   - Recommendation: Enumerate index maintenance steps per mutation (Add/Update/Delete/Restore) before implementation to keep retrieval consistent. (codex, 2026-02-05, decision)

4. **History retention/GC policy**
   - Analysis: The design supports full history but does not define retention bounds or GC triggers, which affects storage and compliance.
   - Recommendation: Treat history as unbounded for MVP; add optional retention policies later once workload patterns are known. (codex, 2026-02-05, decision)

5. **Concurrency semantics and conflict resolution**
   - Analysis: The design uses optimistic version checks, but does not specify behavior for concurrent writers updating topology vs content at the same time.
   - Recommendation: Define conflict rules (e.g., version mismatch aborts; topology changes always close prior current row) to keep transactions deterministic. (codex, 2026-02-05, decision)

6. **Restore/rollback interval behavior**
   - Analysis: Restores currently create new `valid_since` intervals; reopening an old interval is not described.
   - Recommendation: Keep "new interval on restore" as the rule for audit clarity, and document it explicitly. (codex, 2026-02-05, decision)
