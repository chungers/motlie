# VERSIONING: Temporal Edges with Time-Travel and Rollback

## Overview

This document describes the temporal versioning system for graph edges, enabling:
- **Time-travel queries**: Query graph state at any past timestamp
- **Topology rollback**: Restore edge topology to a previous state
- **Summary rollback**: Restore edge content to a previous version
- **Audit history**: Full history of all changes preserved

## Problem Statement

### Current Limitations

1. **No time-travel**: Queries only return current state
2. **No topology rollback**: Changing edge destination loses history
3. **API ambiguity**: Can't distinguish "add second edge" from "retarget edge"

### Example: The Ambiguity Problem

```rust
// Alice knows Bob
AddEdge { src: Alice, dst: Bob, name: "knows" }

// What does this mean?
AddEdge { src: Alice, dst: Carol, name: "knows" }

// A) Alice now knows BOTH Bob AND Carol (multi-edge)
// B) Alice retargeted from Bob to Carol (topology change)
```

Current API treats this as (A). For (B), we need an explicit `RetargetEdge` mutation.

---

## Schema Design

### Key Insight: ValidSince in Key

To support multiple temporal ranges for the same topology, include `ValidSince` in the edge key:

```rust
// OLD: One entry per topology
ForwardEdgeCfKey(SrcId, DstId, NameHash)

// NEW: Multiple entries per topology (different time ranges)
ForwardEdgeCfKey(SrcId, DstId, NameHash, ValidSinceMilli)
```

### Column Families

```rust
/// Forward edges with temporal key
/// Key: (SrcId, DstId, NameHash, ValidSinceMilli) - 48 bytes
/// Value: (ValidUntilMilli, Weight, SummaryHash, Version, Deleted)
ForwardEdges

/// Reverse edges with temporal key (for incoming edge queries)
/// Key: (DstId, SrcId, NameHash, ValidSinceMilli) - 48 bytes
/// Value: (ValidUntilMilli)
ReverseEdges

/// Edge summaries (content-addressed, append-only)
/// Key: SummaryHash - 8 bytes
/// Value: EdgeSummary (no RefCount decrement, GC scans for orphans)
EdgeSummaries

/// Edge version history (for rollback)
/// Key: (SrcId, DstId, NameHash, ValidSinceMilli, Version) - 52 bytes
/// Value: SummaryHash - 8 bytes
EdgeVersionHistory

/// Summary reverse index (for vector search resolution)
/// Key: (SummaryHash, SrcId, DstId, NameHash, ValidSinceMilli, Version)
/// Value: CURRENT | STALE marker
EdgeSummaryIndex
```

### Temporal Semantics

| Field | Meaning |
|-------|---------|
| `valid_since` | Timestamp when edge became valid (part of key) |
| `valid_until` | Timestamp when edge stopped being valid (NULL = still valid) |
| `version` | Monotonic counter for summary changes within a temporal range |

An edge is **current** if: `valid_until IS NULL OR valid_until > now()`

An edge is **valid at time T** if: `valid_since <= T AND (valid_until IS NULL OR valid_until > T)`

---

## Mutation API

### Public Mutations (User-Facing)

```rust
/// Create a new edge.
/// Fails if (src, dst, name) already has a current (non-closed) edge.
pub struct AddEdge {
    pub src: Id,
    pub dst: Id,
    pub name: String,
    pub summary: EdgeSummary,
    pub weight: Option<f64>,
    pub valid_since: Option<TimestampMilli>,  // default: now
}

/// Update summary/weight of existing current edge.
/// Topology unchanged. Uses optimistic locking.
pub struct UpdateEdgeSummary {
    pub src: Id,
    pub dst: Id,
    pub name: String,
    pub new_summary: EdgeSummary,
    pub new_weight: Option<Option<f64>>,  // None = unchanged, Some(None) = clear, Some(v) = set
    pub expected_version: Version,
}

/// Change edge topology (destination and/or name).
/// Atomically closes old edge and creates new edge.
/// At least one of new_dst or new_name must be Some.
pub struct UpdateEdgeTopology {
    // Identity of edge to change
    pub src: Id,
    pub dst: Id,
    pub name: String,

    // What to change (at least one must be Some)
    pub new_dst: Option<Id>,      // None = keep current dst
    pub new_name: Option<String>, // None = keep current name

    // Optional: update summary in same operation
    pub new_summary: Option<EdgeSummary>,  // None = copy from old edge
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
/// Creates NEW edge with topology/summary from the past.
pub struct RestoreEdge {
    pub src: Id,
    pub dst: Id,
    pub name: String,
    pub as_of: TimestampMilli,  // restore state as of this time
}

/// Rollback all outgoing edges from src to state at a previous time.
pub struct RollbackEdgeTopology {
    pub src: Id,
    pub name: Option<String>,  // None = all edge names
    pub as_of: TimestampMilli,
}
```

### Internal Operations

Public mutations translate to these internal CF operations:

| Public Mutation | Internal Operations |
|-----------------|---------------------|
| `AddEdge` | Insert ForwardEdge, ReverseEdge; Put Summary; Write VersionHistory |
| `UpdateEdgeSummary` | Update ForwardEdge value (same key); Put Summary; Write VersionHistory |
| `UpdateEdgeTopology` | Close old ForwardEdge/ReverseEdge (valid_until=now); Insert new edges with new dst/name |
| `DeleteEdge` | Update valid_until=now on ForwardEdge and ReverseEdge |
| `RestoreEdge` | Query past state; Insert new edges with old topology/summary |

---

## Query API

### Current State Queries

```rust
/// Get current outgoing edges from a node.
pub struct OutgoingEdges {
    pub src: Id,
    pub name: Option<String>,  // filter by name
}
// Internally: scan ForwardEdges prefix (src), filter valid_until=NULL or > now

/// Get current incoming edges to a node.
pub struct IncomingEdges {
    pub dst: Id,
    pub name: Option<String>,
}
// Internally: scan ReverseEdges prefix (dst), filter valid_until=NULL or > now
```

### Time-Travel Queries

```rust
/// Get outgoing edges at a specific point in time.
pub struct OutgoingEdgesAt {
    pub src: Id,
    pub name: Option<String>,
    pub at: TimestampMilli,
}
// Internally: scan prefix (src), filter valid_since <= at AND (valid_until=NULL OR valid_until > at)

/// Get incoming edges at a specific point in time.
pub struct IncomingEdgesAt {
    pub dst: Id,
    pub name: Option<String>,
    pub at: TimestampMilli,
}

/// Get full history of an edge topology.
pub struct EdgeHistory {
    pub src: Id,
    pub dst: Id,
    pub name: String,
}
// Returns: Vec<(ValidSince, ValidUntil, Version, SummaryHash)>
```

---

## Examples

### Example 1: Multi-Edge (Alice Knows Both Bob and Carol)

```
t=1000: AddEdge { src: Alice, dst: Bob, name: "knows", summary: "college friends" }
t=2000: AddEdge { src: Alice, dst: Carol, name: "knows", summary: "work friends" }

ForwardEdges after t=2000:
  (Alice, Bob,   "knows", 1000) → (until=NULL, "college friends", v1)
  (Alice, Carol, "knows", 2000) → (until=NULL, "work friends", v1)

Query: OutgoingEdges { src: Alice, name: "knows" }
Result: [Bob, Carol]  // Alice knows BOTH
```

### Example 2: Topology Change (Alice Stops Knowing Bob, Starts Knowing Carol)

```
t=1000: AddEdge { src: Alice, dst: Bob, name: "knows", summary: "friends" }
t=2000: UpdateEdgeTopology { src: Alice, dst: Bob, name: "knows", new_dst: Some(Carol) }

ForwardEdges after t=2000:
  (Alice, Bob,   "knows", 1000) → (until=2000, "friends", v1)  // CLOSED
  (Alice, Carol, "knows", 2000) → (until=NULL, "friends", v1)  // CURRENT

Query: OutgoingEdges { src: Alice, name: "knows" }
Result: [Carol]  // Only Carol

Query: OutgoingEdgesAt { src: Alice, name: "knows", at: 1500 }
Result: [Bob]  // Time travel to before retarget
```

### Example 3: Summary Update (Same Topology, New Content)

```
t=1000: AddEdge { src: Alice, dst: Bob, name: "knows", summary: "acquaintances" }
t=2000: UpdateEdgeSummary { src: Alice, dst: Bob, name: "knows",
                            new_summary: "close friends", expected_version: 1 }

ForwardEdges after t=2000:
  (Alice, Bob, "knows", 1000) → (until=NULL, hash=0xBBB, v2)  // same key, updated value

EdgeVersionHistory:
  (Alice, Bob, "knows", 1000, v1) → 0xAAA ("acquaintances")
  (Alice, Bob, "knows", 1000, v2) → 0xBBB ("close friends")

EdgeSummaries (append-only, no decrement):
  0xAAA → "acquaintances"  // preserved for rollback
  0xBBB → "close friends"
```

### Example 4: Topology Rollback

```
Timeline:
  t=1000: AddEdge Alice→Bob "knows"
  t=2000: UpdateEdgeTopology Alice: Bob→Carol
  t=3000: UpdateEdgeTopology Alice: Carol→Dave
  t=4000: RollbackEdgeTopology { src: Alice, name: "knows", as_of: 1500 }

State after each operation:

After t=1000:
  (Alice, Bob, "knows", 1000) → (until=NULL)  // current

After t=2000:
  (Alice, Bob,   "knows", 1000) → (until=2000)  // closed
  (Alice, Carol, "knows", 2000) → (until=NULL)  // current

After t=3000:
  (Alice, Bob,   "knows", 1000) → (until=2000)  // closed
  (Alice, Carol, "knows", 2000) → (until=3000)  // closed
  (Alice, Dave,  "knows", 3000) → (until=NULL)  // current

After t=4000 (rollback to t=1500):
  (Alice, Bob,   "knows", 1000) → (until=2000)  // historical
  (Alice, Carol, "knows", 2000) → (until=3000)  // historical
  (Alice, Dave,  "knows", 3000) → (until=4000)  // closed by rollback
  (Alice, Bob,   "knows", 4000) → (until=NULL)  // NEW: rollback restored Bob!

Query at t=4500: OutgoingEdges { src: Alice, name: "knows" }
Result: [Bob]  // Rolled back to Bob

Query at t=2500: OutgoingEdgesAt { src: Alice, name: "knows", at: 2500 }
Result: [Carol]  // Time travel still works
```

### Example 5: Delete and Restore

```
t=1000: AddEdge Alice→Bob "knows"
t=2000: DeleteEdge { src: Alice, dst: Bob, name: "knows", expected_version: 1 }
t=3000: RestoreEdge { src: Alice, dst: Bob, name: "knows", as_of: 1500 }

ForwardEdges:
  (Alice, Bob, "knows", 1000) → (until=2000, v1)  // deleted at t=2000
  (Alice, Bob, "knows", 3000) → (until=NULL, v1)  // restored at t=3000

Query at t=1500: [Bob]
Query at t=2500: []      // deleted
Query at t=3500: [Bob]   // restored
```

---

## Summary Changes: Append-Only with Lazy GC

### Why No RefCount Decrement?

To enable summary rollback, old summaries must be preserved:

```
Current (with decrement):
  Update summary A→B: decrement A, increment B
  If A reaches 0: DELETE A  ← Can't rollback!

New (append-only):
  Update summary A→B: just write B (A unchanged)
  A persists until GC determines no index entries reference it
```

### GC for Orphan Summaries

```rust
fn gc_orphan_summaries(&self) -> Result<u64> {
    let mut deleted = 0;

    for (hash, _summary) in self.scan_summaries()? {
        // Check if ANY index entry (current or stale) references this hash
        let has_reference = self.has_any_index_entry(hash)?;

        if !has_reference {
            txn.delete(summaries_cf, hash)?;
            deleted += 1;
        }
    }

    Ok(deleted)
}
```

---

## Node Versioning

The same temporal pattern applies to nodes:

```rust
// Node key includes ValidSince
NodeCfKey(Id, ValidSinceMilli)

// Mutations
AddNode { id, name, summary, valid_since }
UpdateNodeSummary { id, new_summary, expected_version }
DeleteNode { id, expected_version }
RestoreNode { id, as_of }

// Time-travel queries
NodeByIdAt { id, at: TimestampMilli }
```

---

## Migration Path

### Phase 1: Schema Change (Breaking)
1. Add `ValidSinceMilli` to edge keys
2. Migrate existing edges with `valid_since = 0` (epoch)
3. Update all edge queries to handle temporal keys

### Phase 2: New Mutations
1. Implement `RetargetEdge`, `RenameEdge`, `RestoreEdge`
2. Update `AddEdge` to check for existing current edge
3. Implement `RollbackEdgeTopology`

### Phase 3: Remove RefCount Decrement
1. Remove decrement logic from `UpdateEdgeSummary`, `DeleteEdge`
2. Add orphan summary GC scan
3. Update documentation

---

## Summary

| Feature | Before | After |
|---------|--------|-------|
| Time-travel queries | No | Yes |
| Topology rollback | No | Yes |
| Summary rollback | No | Yes |
| Multi-edge support | Implicit | Explicit (`AddEdge`) |
| Topology change | Ambiguous | Explicit (`RetargetEdge`) |
| Summary cleanup | Inline (RefCount) | Lazy (GC scan) |
| Key size (edges) | 40 bytes | 48 bytes |

(claude, 2026-02-04, designed)
