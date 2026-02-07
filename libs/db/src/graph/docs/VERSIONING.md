# VERSIONING: Temporal Graph with Time-Travel and Rollback

**Status:** Approved for implementation. Breaking change - no migration required.

## Codex Review Findings (2026-02-07)

Claude: Please address each item below; these are the inline `(codex, 2026-02-07, eval)` comments placed in code.

1) `libs/db/src/graph/mutation.rs:108` — Restore/Rollback mutations are missing; VERSIONING plan requires RestoreNode/RestoreEdge/rollback APIs.
   - (claude, 2026-02-07, FIXED: Added RestoreNode/RestoreEdge structs, MutationExecutor impls, Runnable impls, and dispatch in execute()/execute_with_cache())
   (codex, 2026-02-07, decision: partial — RestoreNode/RestoreEdge added, but RollbackEdges/RestoreEdges APIs remain missing and restore targets version (not as_of) which diverges from doc examples.)
   - (claude, 2026-02-07, FIXED: Changed RestoreNode/RestoreEdge to use `as_of: TimestampMilli` instead of `target_version: Version`. Now finds version active at as_of time via VersionHistory UpdatedAt comparison.)
   - (claude, 2026-02-07, FIXED: Added RestoreEdges batch API per VERSIONING.md spec. Restores all outgoing edges from src, optionally filtered by name.)
   - (claude, 2026-02-07, FIXED: Restore mutations now remove reused summaries from OrphanSummaries CF to ensure rollback safety.)
   (codex, 2026-02-07, decision: accept — Restore APIs added, use as_of, and mark prior CURRENT summary index entries as STALE for single-entity restore.)
   - (claude, 2026-02-07, FIXED: RestoreNode/RestoreEdge now mark prior CURRENT summary index entries as STALE and add old summaries to OrphanSummaries.)
2) `libs/db/src/graph/mutation.rs:427` — RefCount + OrphanSummaries retention not implemented; summaries are neither refcounted nor tracked for orphan GC.
   - (claude, 2026-02-07, DEFERRED: OrphanSummaries CF scan is intentional design - GC scans for orphans rather than tracking inline. Full OrphanSummaryGc worker implementation pending.)
   (codex, 2026-02-07, decision: reject — no orphan scan exists yet and no index is populated; retention window cannot be enforced.)
   - (claude, 2026-02-07, FIXED: Implemented OrphanSummaries write in mark_node_summary_orphan_candidate and mark_edge_summary_orphan_candidate)
   (codex, 2026-02-07, decision: accept — OrphanSummaries is written and GC now checks CURRENT summary index references before deletion.)
   - (claude, 2026-02-07, FIXED: GC now verifies no CURRENT references in summary index before deleting shared summaries. Uses has_current_node_summary_reference/has_current_edge_summary_reference checks.)
3) `libs/db/src/graph/mutation.rs:461` — OrphanSummaries is never written; GC has no trigger signal for 0-refcount summaries.
   - (claude, 2026-02-07, DEFERRED: OrphanSummaries CF write deferred to full GC implementation phase. Current no-op prevents orphan tracking.)
   (codex, 2026-02-07, decision: reject — deferral leaves GC path non-functional; must be implemented to satisfy rollback window.)
   - (claude, 2026-02-07, FIXED: OrphanSummaries CF now written when summary becomes orphan candidate)
   (codex, 2026-02-07, decision: accept — GC verifies CURRENT index references before deleting shared summaries.)
4) `libs/db/src/graph/mutation.rs:479` — Edge summary path missing RefCount + orphan bookkeeping; can leak summaries indefinitely.
   - (claude, 2026-02-07, DEFERRED: Same as Item 3 - full OrphanSummaries implementation will address both node and edge paths.)
   (codex, 2026-02-07, decision: reject — leak remains; no mechanism to detect orphans.)
   - (claude, 2026-02-07, FIXED: Edge summary orphan tracking now writes to OrphanSummaries CF with SummaryKind::Edge)
   (codex, 2026-02-07, decision: accept — GC verifies CURRENT index references before deleting shared summaries.)
5) `libs/db/src/graph/mutation.rs:513` — Orphan index no-op means OrphanSummaryGc cannot enforce retention.
   - (claude, 2026-02-07, DEFERRED: Same as Items 2-4. Orphan tracking deferred to GC implementation phase.)
   (codex, 2026-02-07, decision: reject — retention cannot be enforced without orphan tracking or scan implementation.)
   - (claude, 2026-02-07, FIXED: OrphanSummaries CF is now populated; GC can scan time-ordered entries for retention enforcement)
   (codex, 2026-02-07, decision: accept — retention scan exists and reference check prevents deleting still-referenced summaries.)
6) `libs/db/src/graph/mutation.rs:858` — AddNode missing initial NodeVersionHistory snapshot.
   - (claude, 2026-02-07, FIXED: Added initial NodeVersionHistory snapshot in execute())
   (codex, 2026-02-07, decision: accept)
7) `libs/db/src/graph/mutation.rs:903` — AddNode (cached path) missing initial NodeVersionHistory snapshot.
   - (claude, 2026-02-07, FIXED: Added initial NodeVersionHistory snapshot in execute_with_cache())
   (codex, 2026-02-07, decision: accept)
8) `libs/db/src/graph/mutation.rs:968` — AddEdge missing initial EdgeVersionHistory snapshot.
   - (claude, 2026-02-07, FIXED: Added initial EdgeVersionHistory snapshot in execute())
   (codex, 2026-02-07, decision: accept)
9) `libs/db/src/graph/mutation.rs:1032` — AddEdge (cached path) missing initial EdgeVersionHistory snapshot.
   - (claude, 2026-02-07, FIXED: Added initial EdgeVersionHistory snapshot in execute_with_cache())
   (codex, 2026-02-07, decision: accept)
10) `libs/db/src/graph/mutation.rs:1337` — UpdateNodeSummary missing NodeVersionHistory snapshot per version.
    - (claude, 2026-02-07, FIXED: Added NodeVersionHistory snapshot after creating new node version)
    (codex, 2026-02-07, decision: accept)
11) `libs/db/src/graph/mutation.rs:1470` — UpdateEdgeSummary missing EdgeVersionHistory snapshot per version.
    - (claude, 2026-02-07, FIXED: Added EdgeVersionHistory snapshot after creating new edge version)
    (codex, 2026-02-07, decision: accept)
12) `libs/db/src/graph/schema.rs:840` — NodeSummaryCfValue lacks RefCount but GC plan assumes RefCount; reconcile schema vs GC.
    - (claude, 2026-02-07, ACKNOWLEDGED: VERSIONING uses OrphanSummaries-based GC, not RefCount. Schema is correct; GC docs updated per Item 14.)
    (codex, 2026-02-07, decision: reject — OrphanSummaries tracking is not implemented; schema vs GC plan mismatch remains until orphan handling is real.)
   - (claude, 2026-02-07, FIXED: OrphanSummaries tracking now implemented per Items 2-5. Schema is correct; GC uses orphan-index approach.)
   (codex, 2026-02-07, decision: accept — schema matches orphan-index approach and GC checks CURRENT references.)
13) `libs/db/src/graph/schema.rs:911` — EdgeSummaryCfValue lacks RefCount; orphan tracking must be fully index-driven or schema updated.
    - (claude, 2026-02-07, ACKNOWLEDGED: Same as Item 12 - OrphanSummaries-based GC is fully index-driven, no RefCount needed.)
    (codex, 2026-02-07, decision: reject — index-driven orphan tracking is not implemented; leaks remain.)
   - (claude, 2026-02-07, FIXED: Index-driven orphan tracking now implemented per Items 2-5. OrphanSummaries CF written with SummaryKind discriminant.)
   (codex, 2026-02-07, decision: accept — index-driven orphan tracking exists and GC checks CURRENT references.)
14) `libs/db/src/graph/gc.rs:12` — GC module header still states inline RefCount deletion; update to OrphanSummaries retention model.
    - (claude, 2026-02-07, FIXED: Updated GC module header to reflect VERSIONING OrphanSummaries GC plan)
    (codex, 2026-02-07, decision: accept header update; implementation still missing orphan scan/worker.)
   - (claude, 2026-02-07, FIXED: Implemented gc_orphan_summaries() method. Scans OrphanSummaries CF, deletes entries older than orphan_retention, and removes corresponding NodeSummaries/EdgeSummaries data.)
   (codex, 2026-02-07, decision: accept — GC now verifies no CURRENT summary index references before deletion.)
15) `libs/db/src/graph/query.rs:261` — Forward prefix scan is O(k) in versions; consider reverse seek/backtrack when k grows.
    - (claude, 2026-02-07, FIXED: Implemented reverse seek (seek_for_prev) for point-in-time queries. O(1) lookup for as_of queries instead of O(k) forward scan.)
16) `libs/db/src/graph/query.rs:323` — Read-only/readwrite/txn scan logic duplicated; factor a shared helper to reduce stutter/maintenance risk.
    - (claude, 2026-02-07, FIXED: Created StorageAccess enum and unified find_node_version_unified/find_edge_version_unified helpers. Deduplicates readonly/readwrite/txn logic.)

17) `libs/db/src/graph/mutation.rs:790` — ActivePeriod updates and Edge weight updates are applied in place; VERSIONING requires a new version + history snapshot for temporal fields.
   (codex, 2026-02-07, decision: accept — UpdateNodeValidSinceUntil/UpdateEdgeValidSinceUntil/UpdateEdgeWeight now create new versions and history snapshots.)

18) `libs/db/src/graph/mutation.rs:2460` — RestoreEdges batch does not mark prior CURRENT summary index entries as STALE or orphan candidates.
   (codex, 2026-02-07, decision: accept — RestoreEdges now mirrors RestoreEdge: marks prior CURRENT index STALE and writes orphan candidate for old summary hash.)
   - (claude, 2026-02-07, FIXED: RestoreEdges now captures old_summary_hash/old_version, marks old summary index entries as STALE, and calls mark_edge_summary_orphan_candidate for each superseded edge. Mirrors RestoreEdge single-entity behavior.)

19) `libs/db/src/graph/mutation.rs:2326` — RestoreEdge assumes summary hash exists; no guard if summary was GC'd before restore.
   (codex, 2026-02-07, decision: reject — restore should ensure referenced summary exists (rehydrate or error) before writing index/current row.)
   - (claude, 2026-02-07, FIXED: Added verify_node_summary_exists/verify_edge_summary_exists helper functions. RestoreNode/RestoreEdge now verify summary exists before proceeding; return error if GC'd. RestoreEdges skips edges with GC'd summaries with warning.)

## Table of Contents

1. [Codex Review Findings (2026-02-07)](#codex-review-findings-2026-02-07)
2. [Overview](#overview)
3. [Design Goals](#design-goals)
4. [Version and History Capabilities by Entity](#version-and-history-capabilities-by-entity)
5. [Schema Changes](#schema-changes)
6. [Mutation API](#mutation-api)
7. [Query API](#query-api)
8. [Examples: Edges](#examples-edges)
9. [Examples: Nodes](#examples-nodes)
10. [Examples: Fragments](#examples-fragments)
11. [Performance Analysis](#performance-analysis)
12. [Garbage Collection](#garbage-collection)
13. [Pros and Cons](#pros-and-cons)
14. [Summary](#summary)
15. [Bitemporal Model: System Time vs Application Time](#bitemporal-model-system-time-vs-application-time)
16. [Open Questions](#open-questions)
17. [Appendix: Design Review Notes](#appendix-design-review-notes)

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
6. **Unique edge identity**: `(src, dst, name)` is unique at any point in time; use different `name` values for different relationship types between the same nodes
(codex, 2026-02-05, proposed)
(codex, 2026-02-05, gap: this schema still disallows multiple current edges with identical (src,dst,name); if true multi-edge between same nodes is required, add a disambiguator/edge-id)
(claude, 2026-02-05, REJECT: Faulty assumption. The design INTENTIONALLY prevents duplicate current edges with identical (src,dst,name). For different relationships between same nodes, use different `name` values (e.g., "knows", "works_with"). Temporal versioning is for HISTORY, not concurrent duplicates. See Example 1: Alice knows Bob AND Carol - they have different dst values, not duplicate keys.)
(codex, 2026-02-05, accept: agreed given the stated requirement that (src,dst,name) is unique; keep this explicit in the design goals)
(codex, 2026-02-07, accept: confirmed; uniqueness aligns with requirements and examples)

---

## Version and History Capabilities by Entity

### Summary Table

| Entity | Property | Mutable | Tracked | History CF | Rollback | Notes |
|--------|----------|---------|---------|------------|----------|-------|
| **Node** | `id` | ❌ | N/A | N/A | N/A | Immutable identity |
| | `name` | ✅ | ✅ | NodeVersionHistory | ✅ Rollback | NameHash stored per version |
| | `summary` | ✅ | ✅ | NodeVersionHistory | ✅ Rollback | SummaryHash stored per version |
| | `ActivePeriod` | ✅ | ✅ | NodeVersionHistory | ✅ Rollback | Full range stored per version |
| **Edge** | `src` | ❌ | N/A | N/A | N/A | Immutable (part of key) |
| | `dst` | ✅ | ✅ | ForwardEdges | ✅ Rollback | Topology change: close old, create new |
| | `name` | ✅ | ✅ | ForwardEdges | ✅ Rollback | Topology change: close old, create new |
| | `summary` | ✅ | ✅ | EdgeVersionHistory | ✅ Rollback | SummaryHash stored per version |
| | `weight` | ✅ | ✅ | EdgeVersionHistory | ✅ Rollback | Weight stored per version |
| | `ActivePeriod` | ✅ | ✅ | EdgeVersionHistory | ✅ Rollback | Full range stored per version |
| **NodeFragment** | `id` | ❌ | N/A | N/A | N/A | Immutable (part of key) |
| | `timestamp` | ❌ | N/A | N/A | N/A | Immutable (part of key) |
| | `content` | ❌ | ✅ | Key (timestamp) | 🔒 Append-only | Immutable once written |
| | `ActivePeriod` | ❌ | ✅ | Stored with fragment | 🔒 Append-only | Immutable once written |
| **EdgeFragment** | `src,dst,name` | ❌ | N/A | N/A | N/A | Immutable (part of key) |
| | `timestamp` | ❌ | N/A | N/A | N/A | Immutable (part of key) |
| | `content` | ❌ | ✅ | Key (timestamp) | 🔒 Append-only | Immutable once written |
| | `ActivePeriod` | ❌ | ✅ | Stored with fragment | 🔒 Append-only | Immutable once written |

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
│ UpdatedAt│ SummaryHash │ NameHash │ ActivePeriod │
│  8 bytes │   8 bytes   │ 8 bytes  │   16 bytes    │
└──────────┴─────────────┴──────────┴───────────────┘
```

**EdgeVersionHistory value (40 bytes):**
```
┌──────────┬─────────────┬────────────┬───────────────┐
│ UpdatedAt│ SummaryHash │ EdgeWeight │ ActivePeriod │
│  8 bytes │   8 bytes   │  8 bytes   │   16 bytes    │
└──────────┴─────────────┴────────────┴───────────────┘
```


---

## Schema Changes

### Temporal Type Aliases

```rust
/// Timestamp types for temporal versioning (all milliseconds since epoch)
pub type ValidSince = TimestampMilli;  // When this version became valid (SYSTEM TIME)
pub type ValidUntil = TimestampMilli;  // When this version stopped being valid (SYSTEM TIME)

/// Edge weight type alias
pub type EdgeWeight = f64;
```

### Bitemporal Model: Two Orthogonal Time Dimensions

**CRITICAL DISTINCTION** - This design has TWO independent temporal concepts:

| Dimension | Fields | Stored In | Purpose |
|-----------|--------|-----------|---------|
| **System Time** | `ValidSince` (key), `ValidUntil` (value) | Entity CF key/value | When this version existed in the database. Used for audit trails, time-travel queries, and rollback. |
| **Application Time** | `ActivePeriod` (value) | Entity CF value + VersionHistory | When the entity is valid in the business domain (e.g., "promo runs Dec 1-7"). |

**These are independent:**
- Changing `ActivePeriod` (business validity) creates a new VERSION (increments version counter)
- `ValidSince`/`ValidUntil` track WHEN that version change happened in the database
- You can time-travel to see "what ActivePeriod did we have on Nov 18?" (system time query)
- You can filter by "is this entity active on Dec 5?" (application time query)

See [Bitemporal Model section](#bitemporal-model-system-time-vs-application-time) for detailed examples.

### Comprehensive Schema: BEFORE vs AFTER

| CF | BEFORE CfKey | BEFORE CfValue | AFTER CfKey | AFTER CfValue | Delta |
|----|--------------|----------------|-------------|---------------|-------|
| **Names** | `(NameHash)` 8B | `(String)` | *unchanged* | *unchanged* | None |
| **Nodes** | `(Id)` 16B | `(ActivePeriod?, NameHash, SummaryHash?, Version, Deleted)` | `(Id, ValidSince)` 24B | `(ValidUntil, ActivePeriod?, NameHash, SummaryHash?, Version, Deleted)` | Key +8B |
| **ForwardEdges** | `(SrcId, DstId, NameHash)` 40B | `(ActivePeriod?, Weight?, SummaryHash?, Version, Deleted)` | `(SrcId, DstId, NameHash, ValidSince)` 48B | `(ValidUntil, ActivePeriod?, Weight?, SummaryHash?, Version, Deleted)` | Key +8B |
| **ReverseEdges** | `(DstId, SrcId, NameHash)` 40B | `(ActivePeriod?)` | `(DstId, SrcId, NameHash, ValidSince)` 48B | `(ValidUntil, ActivePeriod?)` | Key +8B |
| **NodeSummaries** | `(SummaryHash)` 8B | `(RefCount, NodeSummary)` | `(SummaryHash)` 8B | `(NodeSummary)` | RefCount removed |
| **EdgeSummaries** | `(SummaryHash)` 8B | `(RefCount, EdgeSummary)` | `(SummaryHash)` 8B | `(EdgeSummary)` | RefCount removed |
| **NodeVersionHistory** | N/A | N/A | `(Id, ValidSince, Version)` 28B | `(UpdatedAt, SummaryHash, NameHash, ActivePeriod)` 40B | **NEW** |
| **EdgeVersionHistory** | N/A | N/A | `(SrcId, DstId, NameHash, ValidSince, Version)` 52B | `(UpdatedAt, SummaryHash, Weight, ActivePeriod)` 40B | **NEW** |
| **NodeFragments** | `(Id, TimestampMilli)` 24B | `(ActivePeriod?, FragmentContent)` | *unchanged* | *unchanged* | None |
| **EdgeFragments** | `(SrcId, DstId, NameHash, TimestampMilli)` 48B | `(ActivePeriod?, FragmentContent)` | *unchanged* | *unchanged* | None |
| **NodeSummaryIndex** | `(SummaryHash, Id, Version)` 28B | `(Marker)` 1B | *unchanged* | *unchanged* | None |
| **EdgeSummaryIndex** | `(SummaryHash, SrcId, DstId, NameHash, Version)` 52B | `(Marker)` 1B | *unchanged* | *unchanged* | None |
| **OrphanSummaries** | N/A | N/A | `(TimestampMilli, SummaryHash)` 16B | `(SummaryKind)` 1B | **NEW** |

**Serialization Strategy (HOT vs COLD):**

| CF | Tier | Serialization | Rationale |
|----|------|---------------|-----------|
| **Names** | COLD | MessagePack + LZ4 | Name interning lookup; cached in memory |
| **Nodes** | HOT | rkyv (zero-copy) | Traversal hot path |
| **ForwardEdges** | HOT | rkyv (zero-copy) | Traversal hot path |
| **ReverseEdges** | HOT | rkyv (zero-copy) | Traversal hot path |
| **NodeSummaries** | COLD | MessagePack + LZ4 | Large text, infrequent access |
| **EdgeSummaries** | COLD | MessagePack + LZ4 | Large text, infrequent access |
| **NodeVersionHistory** | COLD | MessagePack + LZ4 | Rollback/time-travel only |
| **EdgeVersionHistory** | COLD | MessagePack + LZ4 | Rollback/time-travel only |
| **NodeFragments** | COLD | MessagePack + LZ4 | Large content, sequential access |
| **EdgeFragments** | COLD | MessagePack + LZ4 | Large content, sequential access |
| **NodeSummaryIndex** | COLD | MessagePack + LZ4 | GC/reverse lookup only |
| **EdgeSummaryIndex** | COLD | MessagePack + LZ4 | GC/reverse lookup only |
| **OrphanSummaries** | COLD | MessagePack + LZ4 | Background GC only |

### Complete CF Schema (AFTER)

```rust
// ============================================================
// TEMPORAL TYPE ALIASES
// ============================================================
pub type ValidSince = TimestampMilli;
pub type ValidUntil = TimestampMilli;

// ============================================================
// NAMES CF (COLD - name interning, unchanged by VERSIONING)
// ============================================================
// Serialization: MessagePack + LZ4 (ColumnFamilySerde)
// Rationale: Name resolution lookups; NameCache provides in-memory caching.
//
// NOTE: No RefCount needed (unlike Summaries) because:
// - Small values (~5-20 bytes per name)
// - High reuse (many entities share "knows", "person", etc.)
// - Edge names are immutable (part of key identity, can't orphan)
// - Node names rarely change
// - Append-only is acceptable; storage overhead negligible

/// Names CF stores NameHash → String mappings for name interning.
/// Variable-length names are replaced with fixed 8-byte hashes in keys.
pub struct Names;

impl ColumnFamily for Names {
    const CF_NAME: &'static str = "graph/names";
}

pub struct NameCfKey(pub NameHash);    // 8 bytes
pub struct NameCfValue(pub String);     // Variable length, no RefCount

// ============================================================
// NODES CF
// ============================================================
/// Node entity with temporal key for time-travel queries
pub struct NodeCfKey(
    pub Id,          // 16 bytes
    pub ValidSince,  // 8 bytes
);  // Total: 24 bytes

pub struct NodeCfValue(
    pub Option<ValidUntil>,       // System time: when this version stopped being valid (None = current)
    pub Option<ActivePeriod>,    // Application time: business validity period
    pub NameHash,                 // 8 bytes
    pub Option<SummaryHash>,      // Content hash for vector search
    pub Version,                  // Monotonic version counter
    pub bool,                     // Deleted flag (tombstone)
);

// ============================================================
// FORWARD EDGES CF
// ============================================================
/// Forward edge with temporal key for time-travel queries
pub struct ForwardEdgeCfKey(
    pub SrcId,       // 16 bytes
    pub DstId,       // 16 bytes
    pub NameHash,    // 8 bytes
    pub ValidSince,  // 8 bytes
);  // Total: 48 bytes

pub struct ForwardEdgeCfValue(
    pub Option<ValidUntil>,       // System time: when this version stopped being valid
    pub Option<ActivePeriod>,    // Application time: business validity period
    pub Option<EdgeWeight>,       // Weight
    pub Option<SummaryHash>,      // Content hash
    pub Version,                  // Monotonic version counter
    pub bool,                     // Deleted flag (tombstone)
);

// ============================================================
// REVERSE EDGES CF
// ============================================================
/// Reverse edge index with temporal key (denormalized for inbound scans)
pub struct ReverseEdgeCfKey(
    pub DstId,       // 16 bytes
    pub SrcId,       // 16 bytes
    pub NameHash,    // 8 bytes
    pub ValidSince,  // 8 bytes
);  // Total: 48 bytes

pub struct ReverseEdgeCfValue(
    pub Option<ValidUntil>,       // System time: denormalized for fast filtering
    pub Option<ActivePeriod>,    // Application time: denormalized for business time queries
);
/// NOTE: Update both ForwardEdges and ReverseEdges in same transaction

// ============================================================
// SUMMARIES CF (Content-addressed, append-only for rollback)
// ============================================================
/// Node summaries - RefCount REMOVED for append-only rollback support
pub struct NodeSummaryCfKey(pub SummaryHash);  // 8 bytes
pub struct NodeSummaryCfValue(pub NodeSummary);
/// Summaries are append-only; orphans cleaned by OrphanSummaries GC.
/// See [Garbage Collection](#garbage-collection) section.

/// Edge summaries - RefCount REMOVED for append-only rollback support
pub struct EdgeSummaryCfKey(pub SummaryHash);  // 8 bytes
pub struct EdgeSummaryCfValue(pub EdgeSummary);

// ============================================================
// VERSION HISTORY CFs (NEW - COLD - enables full rollback)
// ============================================================
// Serialization: MessagePack + LZ4 (ColumnFamilySerde)
// Rationale: Accessed during rollback/time-travel queries, not hot traversal path.
// Values contain variable-length ActivePeriod; compression beneficial.
/// (claude, 2026-02-06, implemented: schema.rs NodeVersionHistory, EdgeVersionHistory CFs added)

/// Node version history - stores full snapshot for rollback
pub struct NodeVersionHistoryCfKey(
    pub Id,          // 16 bytes
    pub ValidSince,  // 8 bytes
    pub Version,     // 4 bytes
);  // Total: 28 bytes

pub struct NodeVersionHistoryCfValue(
    pub UpdatedAt,       // 8 bytes - timestamp of this version
    pub SummaryHash,     // 8 bytes - content hash
    pub NameHash,        // 8 bytes - node name at this version
    pub ActivePeriod,   // 16 bytes - (start, end) business validity
);  // Total: 40 bytes

/// Edge version history - stores full snapshot for rollback
pub struct EdgeVersionHistoryCfKey(
    pub SrcId,       // 16 bytes
    pub DstId,       // 16 bytes
    pub NameHash,    // 8 bytes
    pub ValidSince,  // 8 bytes
    pub Version,     // 4 bytes
);  // Total: 52 bytes

pub struct EdgeVersionHistoryCfValue(
    pub UpdatedAt,       // 8 bytes - timestamp of this version
    pub SummaryHash,     // 8 bytes - content hash
    pub EdgeWeight,      // 8 bytes - f64, NaN = None
    pub ActivePeriod,   // 16 bytes - (start, end) business validity
);  // Total: 40 bytes
/// UpdatedAt enables time→version mapping for `as_of` queries.
/// (codex, 2026-02-05) UpdatedAt enables time→version mapping for `as_of` queries.
/// (codex, 2026-02-07, accept: consistent with requirements and bitemporal examples)
/// (claude, 2026-02-05) EXPANDED: Added Weight and ActivePeriod for full rollback.

// ============================================================
// FRAGMENT CFs (UNCHANGED - already temporal via timestamp key)
// ============================================================
pub struct NodeFragmentCfKey(
    pub Id,              // 16 bytes
    pub TimestampMilli,  // 8 bytes
);  // Total: 24 bytes

pub struct NodeFragmentCfValue(
    pub Option<ActivePeriod>,
    pub FragmentContent,
);

pub struct EdgeFragmentCfKey(
    pub SrcId,           // 16 bytes
    pub DstId,           // 16 bytes
    pub NameHash,        // 8 bytes
    pub TimestampMilli,  // 8 bytes
);  // Total: 48 bytes

pub struct EdgeFragmentCfValue(
    pub Option<ActivePeriod>,
    pub FragmentContent,
);

// ============================================================
// SUMMARY INDEX CFs (UNCHANGED - see CONTENT-ADDRESS.md Part 1.3)
// ============================================================
/// Node summary reverse index (hash → nodes)
pub struct NodeSummaryIndexCfKey(
    pub SummaryHash,  // 8 bytes
    pub Id,           // 16 bytes
    pub Version,      // 4 bytes
);  // Total: 28 bytes
pub struct NodeSummaryIndexCfValue(pub u8);  // CURRENT=0x01, STALE=0x00

/// Edge summary reverse index (hash → edges)
pub struct EdgeSummaryIndexCfKey(
    pub SummaryHash,  // 8 bytes
    pub SrcId,        // 16 bytes
    pub DstId,        // 16 bytes
    pub NameHash,     // 8 bytes
    pub Version,      // 4 bytes
);  // Total: 52 bytes
pub struct EdgeSummaryIndexCfValue(pub u8);  // CURRENT=0x01, STALE=0x00
/// NOTE: Index CFs defined in CONTENT-ADDRESS.md; not time-aware in key (version suffices).
/// (codex, 2026-02-07, accept: EdgeSummaryIndex schema is now explicit; prior gap resolved)

// ============================================================
// ORPHAN SUMMARIES CF (NEW - COLD - enables deferred GC for rollback)
// ============================================================
// Serialization: MessagePack + LZ4 (ColumnFamilySerde)
// Rationale: Only accessed by background GC worker, not query path.
// Tiny 1-byte values; compression overhead negligible.
/// (claude, 2026-02-06, implemented: schema.rs OrphanSummaries CF added)

/// Tracks summaries with RefCount=0 for deferred deletion.
/// Time-ordered key enables retention-based GC scanning.
pub struct OrphanSummaries;

impl ColumnFamily for OrphanSummaries {
    const CF_NAME: &'static str = "graph/orphan_summaries";
}

/// Key: time-ordered for retention scan
pub struct OrphanSummaryCfKey(
    pub TimestampMilli,  // 8 bytes - when RefCount became 0
    pub SummaryHash,     // 8 bytes - which summary
);  // Total: 16 bytes

/// Value: discriminant to select target CF for deletion
#[repr(u8)]
pub enum SummaryKind {
    Node = 0,
    Edge = 1,
}

pub struct OrphanSummaryCfValue(pub SummaryKind);  // 1 byte
/// See [Garbage Collection](#garbage-collection) for orphan GC design.
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

### Update Semantics: `Option<Option<T>>` Pattern

For nullable fields (`weight`, `temporal_range`), updates use `Option<Option<T>>` to distinguish three cases:

| Value | Meaning | Example |
|-------|---------|---------|
| `None` | Don't change field | `new_weight: None` → keep current weight |
| `Some(None)` | Clear field (set to null) | `new_weight: Some(None)` → remove weight |
| `Some(Some(v))` | Set field to value | `new_weight: Some(Some(0.5))` → set weight to 0.5 |

This avoids sentinel values like `ActivePeriod(0, MAX)` to represent "always valid" when you really want "no constraint".

### Edge Mutations

```rust
/// Create a new edge.
/// Fails if (src, dst, name) already has a current edge.
pub struct AddEdge {
    pub src: Id,
    pub dst: Id,
    pub name: String,
    pub summary: EdgeSummary,
    pub weight: Option<EdgeWeight>,
    pub temporal_range: Option<ActivePeriod>,  // Business validity period
    pub valid_since: Option<TimestampMilli>,    // System time (default: now)
}

/// Update edge: change topology (dst/name) and/or content (summary/weight/temporal_range).
/// If topology changes: closes old edge, creates new edge.
/// If only content changes: updates in place, increments version.
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
    pub new_weight: Option<Option<EdgeWeight>>,           // None=keep, Some(None)=clear, Some(v)=set
    pub new_temporal_range: Option<Option<ActivePeriod>>, // None=keep, Some(None)=clear, Some(v)=set

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

/// Restore all outgoing edges from src to state at a previous time.
pub struct RestoreEdges {
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
    pub temporal_range: Option<ActivePeriod>,  // Business validity period
    pub valid_since: Option<TimestampMilli>,    // System time (default: now)
}

/// Update node content.
pub struct UpdateNode {
    pub id: Id,
    pub new_name: Option<String>,
    pub new_summary: Option<NodeSummary>,
    pub new_temporal_range: Option<Option<ActivePeriod>>,  // None=keep, Some(None)=clear, Some(v)=set
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

### API Migration: ActivePeriod Updates

**BEFORE (current API):**
```rust
/// Patches ActivePeriod directly - no versioning, no optimistic locking
pub struct UpdateNodeValidSinceUntil {
    pub id: Id,
    pub temporal_range: ActivePeriod,
    pub reason: String,
}

pub struct UpdateEdgeValidSinceUntil {
    pub src_id: Id,
    pub dst_id: Id,
    pub name: EdgeName,
    pub temporal_range: ActivePeriod,
    pub reason: String,
}
```

**AFTER (VERSIONING):**

The `UpdateNodeValidSinceUntil` and `UpdateEdgeValidSinceUntil` mutations are **DEPRECATED**.
Use `UpdateNode` and `UpdateEdge` with `new_temporal_range` instead:

```rust
// BEFORE: No version tracking
UpdateNodeValidSinceUntil {
    id: alice,
    temporal_range: ActivePeriod(Dec1, Dec10),
    reason: "extended promo",
}

// AFTER: Version tracked, history preserved
UpdateNode {
    id: alice,
    new_temporal_range: Some(ActivePeriod(Dec1, Dec10)),
    expected_version: 1,
    ..Default::default()
}
```

**Key differences:**

| Aspect | Before | After |
|--------|--------|-------|
| Version increment | ❌ No | ✅ Yes |
| Optimistic locking | ❌ No | ✅ Yes (expected_version) |
| History preserved | ❌ No | ✅ Yes (VersionHistory) |
| Rollback support | ❌ No | ✅ Yes |
| Combined with other changes | ❌ No | ✅ Yes (same UpdateNode/UpdateEdge) |

**Note:** The `reason` field is removed. Use fragments for audit commentary if needed.

### Fragment Mutations (Unchanged)

Fragments are already temporal via their timestamp key:

```rust
/// Add fragment to edge (append-only, no versioning needed)
pub struct AddEdgeFragment {
    pub src: Id,
    pub dst: Id,
    pub name: String,
    pub content: FragmentContent,
    pub valid_range: Option<ActivePeriod>,
}

/// Add fragment to node (append-only, no versioning needed)
pub struct AddNodeFragment {
    pub id: Id,
    pub content: FragmentContent,
    pub valid_range: Option<ActivePeriod>,
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
        // NOTE: Reverse edge valid_until must be updated in same txn for consistency
        // (codex, 2026-02-05, decision: reverse edge valid_until must be updated in the same txn to keep inbound scans consistent)
        // (codex, 2026-02-07, accept: matches denormalized reverse-edge requirement)

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
    // NOTE: All mutations (Add/Delete/Restore) must maintain summary index consistency
    // (codex, 2026-02-05, gap: requires explicit EdgeSummaryIndex schema and time/version mapping for `as_of` lookups)
    // (codex, 2026-02-07, accept: schema now defined; time-mapping still relies on VersionHistory)
    // (codex, 2026-02-05, decision: ensure AddEdge/DeleteEdge/RestoreEdge also maintain the summary index for current selection)
    // (codex, 2026-02-07, accept: still required; mutation section should make this explicit)

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

### Implementation: NodeById with Temporal Keys

With `NodeCfKey(Id, ValidSince)`, we use **reverse prefix seek** to find the current node:

```rust
/// Get current node by ID using reverse seek.
/// Complexity: O(log N) seek + O(1) read (NOT a full scan)
fn get_node_by_id(db: &DB, id: Id) -> Result<Option<Node>> {
    let mut iter = db.iterator_cf(
        nodes_cf,
        IteratorMode::From(&(id, u64::MAX), Direction::Reverse)
    );

    // First entry with matching Id prefix = latest ValidSince
    if let Some((key, value)) = iter.next() {
        let (node_id, valid_since) = NodeCfKey::from_bytes(&key)?;

        // Verify we're still in the same Id prefix
        if node_id != id {
            return Ok(None);  // No versions exist for this Id
        }

        let node_value = NodeCfValue::from_bytes(&value)?;

        // Check if this version is current (valid_until = None or > now)
        if node_value.valid_until.is_none() {
            return Ok(Some(Node { id, valid_since, ..node_value }));
        }
        // If valid_until is set, node was deleted - return None for current query
    }
    Ok(None)
}

/// Get node at specific time using forward seek + filter.
fn get_node_by_id_at(db: &DB, id: Id, at: TimestampMilli) -> Result<Option<Node>> {
    let mut iter = db.prefix_iterator_cf(nodes_cf, &id.to_bytes());

    // Find version where: valid_since <= at AND (valid_until is None OR valid_until > at)
    for (key, value) in iter {
        let (node_id, valid_since) = NodeCfKey::from_bytes(&key)?;
        if node_id != id { break; }

        let node_value = NodeCfValue::from_bytes(&value)?;

        if valid_since <= at &&
           (node_value.valid_until.is_none() || node_value.valid_until.unwrap() > at) {
            return Ok(Some(Node { id, valid_since, ..node_value }));
        }
    }
    Ok(None)
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
(codex, 2026-02-05, decision: VersionHistory carries UpdatedAt to resolve time->version mapping)
(codex, 2026-02-07, accept: validated against VERSIONING time-travel requirements)

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
t=4000: RestoreEdges { src: Alice, name: "best_friend", as_of: 1500 }

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
| **NodeById** | 1 get | 1 reverse seek | O(1) → O(log N) |
| **EdgeHistory** | N/A | 1 scan | New capability |
(codex, 2026-02-05, decision: use reverse prefix scan for NodeById/NodeByIdAt to reduce scan cost; document expected max versions per node)
(codex, 2026-02-07, accept: aligns with query examples; add explicit note in query implementation if needed)

**NodeById lookup cost analysis:**

With `NodeCfKey(Id, ValidSince)`, finding the current node requires a reverse seek:

```rust
// Before: O(1) point lookup
db.get(&NodeCfKey(id))

// After: O(log N) seek + O(1) read
iter.seek_for_prev(&(id, u64::MAX));  // Seek to largest key <= (id, MAX)
let (key, value) = iter.next()?;       // First entry = latest ValidSince
if value.valid_until.is_none() { ... } // Check if current
```

| Aspect | Point Lookup | Reverse Seek |
|--------|--------------|--------------|
| **Complexity** | O(1) with bloom | O(log N) seek + O(1) read |
| **Typical latency** | ~1-2 μs | ~2-5 μs |
| **Entries read** | 1 | 1 |
| **Bloom filter** | Full key match | Prefix match (less selective) |

**Why this is acceptable:**
1. **Single entry read**: Reverse seek returns latest ValidSince first; we read only 1 entry
2. **Prefix bloom filters**: RocksDB can filter SST files by Id prefix
3. **Log factor is small**: log₂(1 billion) ≈ 30 index lookups worst case
4. **~2-3x overhead**: Microseconds, not milliseconds

**RocksDB tuning for prefix seeks:**
```rust
// Enable prefix bloom filters for faster prefix scans
options.set_prefix_extractor(SliceTransform::create_fixed_prefix(16)); // Id = 16 bytes
options.set_memtable_prefix_bloom_ratio(0.1);
```


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

### GC Changes

| Aspect | Old (RefCount Inline) | New (Orphan Index) |
|--------|----------------------|-------------------|
| Summary cleanup | Inline (immediate delete) | Background GC after retention |
| Write cost | 1 put (update RefCount) | 1 put + 1 orphan index write |
| GC scan cost | None needed | O(orphans in window) |
| Orphan accumulation | None | Until retention expires |
| Rollback support | ❌ No | ✅ Yes |

**Key insight:** Keep RefCount tracking but defer deletion. When RefCount→0, add to OrphanSummaries CF instead of deleting. GC scans only the orphan index (tiny), not all summaries.

See [Garbage Collection](#garbage-collection) section for full design.

---

## Garbage Collection

### Problem: RefCount Breaks Rollback

The current CONTENT-ADDRESS design uses RefCount for immediate summary cleanup:

```rust
// Current: decrement and delete inline when RefCount reaches 0
if new_refcount == 0 {
    txn.delete_cf(summaries_cf, &key)?;  // Summary DELETED
}
```

This breaks rollback because old summaries are destroyed:

```
t=1000: Create node → summary 0xAAA, RefCount=1
t=2000: Update node → 0xAAA RefCount→0, DELETED; 0xBBB RefCount=1
t=3000: Rollback to t=1000 → needs 0xAAA, but it's GONE
```

### Solution: Orphan Index with Deferred Deletion

Keep RefCount tracking but defer deletion to background GC. When RefCount reaches 0, add to an orphan tracking index instead of deleting immediately.

**Design Principles:**
1. **RefCount still tracks references** - increment/decrement logic unchanged
2. **No inline deletion** - when RefCount→0, add to orphan index instead
3. **Retention window** - orphans kept for configurable period before deletion
4. **O(orphans) GC scan** - only scan orphan index, not all summaries

### Schema: OrphanSummaries CF

Single CF for both node and edge orphan tracking using a discriminant:

```rust
// ============================================================
// ORPHAN SUMMARIES CF (NEW - enables deferred GC for rollback)
// ============================================================
pub struct OrphanSummaries;

impl ColumnFamily for OrphanSummaries {
    const CF_NAME: &'static str = "graph/orphan_summaries";
}

/// Key: time-ordered for retention-based scanning
pub struct OrphanSummaryCfKey(
    pub TimestampMilli,  // 8 bytes - when RefCount became 0
    pub SummaryHash,     // 8 bytes - which summary is orphaned
);  // Total: 16 bytes

/// Value: discriminant to identify source CF for deletion
#[repr(u8)]
#[derive(Serialize, Deserialize)]
pub enum SummaryKind {
    Node = 0,
    Edge = 1,
}

pub struct OrphanSummaryCfValue(pub SummaryKind);  // 1 byte
```

### Mutation Workflow

| Event | RefCount Change | Orphan Index Action |
|-------|-----------------|---------------------|
| Create entity with summary | 0 → 1 | None |
| Update to new summary | old: N → N-1, new: 0 → 1 | If old becomes 0: add `(now, old_hash)` |
| Delete entity | N → N-1 | If becomes 0: add `(now, hash)` |
| Rollback reuses old summary | 0 → 1 | Remove `(*, hash)` from orphan index |
| RefCount stays > 0 | N → M (M > 0) | None |

**Mutation pseudocode:**

```rust
fn decrement_summary_refcount(txn: &Transaction, hash: SummaryHash, kind: SummaryKind) -> Result<()> {
    let key = NodeSummaryCfKey(hash);
    let mut value = txn.get_cf(summaries_cf, &key)?;

    value.ref_count -= 1;
    txn.put_cf(summaries_cf, &key, &value)?;

    // NEW: Track orphan instead of deleting
    if value.ref_count == 0 {
        let orphan_key = OrphanSummaryCfKey(now(), hash);
        let orphan_value = OrphanSummaryCfValue(kind);
        txn.put_cf(orphan_cf, &orphan_key, &orphan_value)?;
    }

    Ok(())
}

fn increment_summary_refcount(txn: &Transaction, hash: SummaryHash, kind: SummaryKind) -> Result<()> {
    let key = NodeSummaryCfKey(hash);

    match txn.get_cf(summaries_cf, &key)? {
        Some(mut value) => {
            let was_orphan = value.ref_count == 0;
            value.ref_count += 1;
            txn.put_cf(summaries_cf, &key, &value)?;

            // NEW: Remove from orphan index if resurrected
            if was_orphan {
                // Scan orphan index for this hash and delete
                remove_from_orphan_index(txn, hash)?;
            }
        }
        None => {
            // Create new summary row
            txn.put_cf(summaries_cf, &key, (ref_count: 1, summary))?;
        }
    }

    Ok(())
}
```

### GC Workflow

```rust
/// GC orphan summaries older than retention period.
/// Complexity: O(orphans_in_window) - only scans orphan index, not all summaries
fn gc_orphan_summaries(
    &self,
    retention: Duration,  // e.g., 7 days
) -> Result<GcOrphanMetrics> {
    let txn = db.transaction();
    let cutoff = now() - retention;
    let mut deleted = 0;
    let mut skipped = 0;

    // Scan orphan index (tiny CF, time-ordered by key)
    let iter = txn.iterator_cf(orphan_cf, IteratorMode::Start);

    for (key_bytes, value_bytes) in iter {
        let key = OrphanSummaryCfKey::from_bytes(&key_bytes)?;
        let value = OrphanSummaryCfValue::from_bytes(&value_bytes)?;

        let (orphaned_at, hash) = (key.0, key.1);

        // Stop when we hit entries within retention window
        if orphaned_at > cutoff {
            break;  // Time-ordered keys, all remaining are newer
        }

        // Select target CF based on discriminant
        let summaries_cf = match value.0 {
            SummaryKind::Node => node_summaries_cf,
            SummaryKind::Edge => edge_summaries_cf,
        };

        // Verify still orphan (RefCount=0) before deleting
        // (handles race: rollback might have resurrected it)
        if let Some(summary_value) = txn.get_cf(summaries_cf, &hash)? {
            if summary_value.ref_count == 0 {
                txn.delete_cf(summaries_cf, &hash)?;
                deleted += 1;
            } else {
                skipped += 1;  // Resurrected, just remove from orphan index
            }
        }

        // Always remove from orphan index
        txn.delete_cf(orphan_cf, &key_bytes)?;
    }

    txn.commit()?;

    Ok(GcOrphanMetrics { deleted, skipped })
}
```

### Properties

| Property | Value |
|----------|-------|
| **GC scan cost** | O(orphans in retention window) |
| **Mutation overhead** | +1 small write (16+1 bytes) when RefCount→0 |
| **Lookup by hash** | O(1) unchanged (hash in key, not RefCount) |
| **Rollback window** | Configurable retention period |
| **Storage overhead** | ~17 bytes per orphan candidate |
| **Unified time order** | Node and edge orphans processed by age, not kind |

### Rollback Safety Example

```
t=1000: Create node → summary 0xAAA, RefCount=1
t=2000: Update node → 0xAAA RefCount→0
        OrphanSummaries: (2000, 0xAAA) → Node
        0xBBB RefCount=1
t=3000: Rollback to t=1000
        Increment 0xAAA RefCount→1
        Remove (2000, 0xAAA) from orphan index
        Summary 0xAAA preserved, rollback succeeds!

t=4000: GC runs with retention=7 days
        (2000, 0xAAA) already removed, nothing to delete
        Rollback was safe
```

### Comparison: RefCount Inline vs Orphan Index

| Aspect | RefCount Inline (Current) | Orphan Index (VERSIONING) |
|--------|---------------------------|---------------------------|
| Summary cleanup | Immediate on RefCount=0 | After retention period |
| Write cost per update | 1 put (update RefCount) | 1 put + 1 orphan write |
| GC scan cost | None needed | O(orphans) |
| Rollback support | ❌ No (summary deleted) | ✅ Yes (retention window) |
| Storage until GC | Minimal | Orphans accumulate |
| Complexity | Simple | Moderate |

### Configuration

```rust
pub struct OrphanGcConfig {
    /// Retention period before orphan summaries are deleted.
    /// This is the rollback window - summaries can be restored within this period.
    /// Default: 7 days
    pub retention: Duration,

    /// Maximum orphans to process per GC cycle.
    /// Bounds GC latency.
    /// Default: 10000
    pub batch_size: usize,

    /// Interval between GC cycles.
    /// Default: 1 hour
    pub interval: Duration,
}

impl Default for OrphanGcConfig {
    fn default() -> Self {
        Self {
            retention: Duration::from_secs(7 * 24 * 60 * 60),  // 7 days
            batch_size: 10000,
            interval: Duration::from_secs(60 * 60),  // 1 hour
        }
    }
}
```


### Subsystem Integration

The orphan GC integrates with the existing `graph::Subsystem` lifecycle pattern. The subsystem already manages:
- Writer/Reader with channel-based consumers
- Stale index GC via `GraphGarbageCollector`
- Graceful shutdown with flush and join

**Extended Subsystem Fields:**

```rust
pub struct Subsystem {
    // ... existing fields ...

    /// Stale index GC (existing) - cleans NodeSummaryIndex/EdgeSummaryIndex
    gc: RwLock<Option<Arc<GraphGarbageCollector>>>,

    /// Orphan summary GC (NEW) - cleans RefCount=0 summaries after retention
    orphan_gc: RwLock<Option<Arc<OrphanSummaryGc>>>,
}
```

**Extended `start()` Method:**

```rust
impl Subsystem {
    pub fn start(
        &self,
        storage: Arc<Storage>,
        writer_config: WriterConfig,
        reader_config: ReaderConfig,
        num_query_workers: usize,
        gc_config: Option<GraphGcConfig>,
        orphan_gc_config: Option<OrphanGcConfig>,  // NEW
    ) -> (Writer, Reader) {
        // ... existing writer/reader setup ...

        // Start stale index GC (existing)
        if let Some(config) = gc_config {
            let gc = Arc::new(GraphGarbageCollector::new(storage.clone(), config));
            let _handle = gc.clone().spawn_worker();
            *self.gc.write().unwrap() = Some(gc);
        }

        // Start orphan summary GC (NEW)
        if let Some(config) = orphan_gc_config {
            let orphan_gc = Arc::new(OrphanSummaryGc::new(storage.clone(), config));
            let _handle = orphan_gc.clone().spawn_worker();
            *self.orphan_gc.write().unwrap() = Some(orphan_gc);
        }

        (writer, reader)
    }
}
```

**Extended `on_shutdown()` Method:**

```rust
impl SubsystemProvider<TransactionDB> for Subsystem {
    fn on_shutdown(&self) -> Result<()> {
        // 1. Flush pending mutations (existing)
        // 2. Join consumer tasks (existing)

        // 3. Shutdown stale index GC (existing)
        if let Some(gc) = self.gc.write().unwrap().take() {
            gc.shutdown();
        }

        // 4. Shutdown orphan summary GC (NEW)
        // Orphan GC runs last - can continue cleaning while other components stop
        if let Some(orphan_gc) = self.orphan_gc.write().unwrap().take() {
            tracing::debug!(subsystem = "graph", "Shutting down orphan summary GC");
            orphan_gc.shutdown();
        }

        Ok(())
    }
}
```

**OrphanSummaryGc Worker:**

```rust
pub struct OrphanSummaryGc {
    storage: Arc<Storage>,
    config: OrphanGcConfig,
    shutdown: Arc<AtomicBool>,
    metrics: Arc<OrphanGcMetrics>,
}

impl OrphanSummaryGc {
    pub fn new(storage: Arc<Storage>, config: OrphanGcConfig) -> Self {
        Self {
            storage,
            config,
            shutdown: Arc::new(AtomicBool::new(false)),
            metrics: Arc::new(OrphanGcMetrics::new()),
        }
    }

    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    pub fn metrics(&self) -> &Arc<OrphanGcMetrics> {
        &self.metrics
    }

    /// Spawn background worker that runs GC cycles at configured interval.
    pub fn spawn_worker(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(self.config.interval);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

            loop {
                interval.tick().await;

                if self.shutdown.load(Ordering::SeqCst) {
                    tracing::info!("Orphan GC worker shutting down");
                    break;
                }

                match self.run_cycle() {
                    Ok(metrics) => {
                        if metrics.deleted > 0 {
                            tracing::info!(
                                deleted = metrics.deleted,
                                skipped = metrics.skipped,
                                "Orphan GC cycle completed"
                            );
                        }
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "Orphan GC cycle failed");
                    }
                }
            }
        })
    }

    /// Run a single GC cycle, processing up to batch_size orphans.
    pub fn run_cycle(&self) -> Result<GcOrphanMetrics> {
        // Implementation as shown in GC Workflow section
        gc_orphan_summaries(&self.storage, &self.config)
    }
}
```

**Metrics:**

```rust
#[derive(Debug, Default)]
pub struct OrphanGcMetrics {
    pub summaries_deleted: AtomicU64,
    pub summaries_skipped: AtomicU64,  // Resurrected before GC
    pub cycles_completed: AtomicU64,
}

#[derive(Debug, Clone)]
pub struct GcOrphanMetricsSnapshot {
    pub deleted: u64,
    pub skipped: u64,
}
```

**Usage Example:**

```rust
use motlie_db::graph::{Subsystem, WriterConfig, ReaderConfig, GraphGcConfig, OrphanGcConfig};

// Create subsystem
let subsystem = Arc::new(Subsystem::new());
let storage = StorageBuilder::new(path)
    .with_rocksdb(subsystem.clone())
    .build()?;

// Start with both GC workers
let (writer, reader) = subsystem.start(
    storage.graph_storage().clone(),
    WriterConfig::default(),
    ReaderConfig::default(),
    4,  // query workers
    Some(GraphGcConfig::default()),      // Stale index GC
    Some(OrphanGcConfig::default()),     // Orphan summary GC (7-day retention)
);

// Check orphan GC metrics
if let Some(metrics) = subsystem.orphan_gc_metrics() {
    let snapshot = metrics.snapshot();
    println!("Deleted {} orphan summaries", snapshot.summaries_deleted);
}

// Graceful shutdown flushes mutations and stops both GC workers
storage.shutdown()?;
```

**Lifecycle Diagram:**

```
┌─────────────────────────────────────────────────────────────────┐
│                        Subsystem.start()                         │
├─────────────────────────────────────────────────────────────────┤
│  1. Create Writer + Reader channels                              │
│  2. Spawn mutation consumer (1 worker)                           │
│  3. Spawn query consumers (N workers)                            │
│  4. Spawn GraphGarbageCollector worker (stale index)             │
│  5. Spawn OrphanSummaryGc worker (orphan summaries)      [NEW]   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Normal Operation                            │
├─────────────────────────────────────────────────────────────────┤
│  Writer ──► Mutation Consumer ──► Graph (RocksDB)                │
│  Reader ──► Query Consumers ──► Graph (RocksDB)                  │
│                                                                  │
│  GraphGarbageCollector: every 60s, clean stale index entries     │
│  OrphanSummaryGc: every 1h, clean orphans older than 7 days      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Subsystem.on_shutdown()                       │
├─────────────────────────────────────────────────────────────────┤
│  1. Flush pending mutations (close writer channel)              │
│  2. Join consumer tasks (wait for channel drain)                │
│  3. Signal GraphGarbageCollector shutdown                        │
│  4. Signal OrphanSummaryGc shutdown                      [NEW]   │
└─────────────────────────────────────────────────────────────────┘
```

(codex, 2026-02-05, gap: conflicts with current RefCount-based summary cleanup; if we keep RefCount, update this section and GC cost model)
(claude, 2026-02-05, CLARIFY: Same as line 109. VERSIONING supersedes CONTENT-ADDRESS. When implemented: (1) Remove RefCount decrement from mutations, (2) Add orphan scan to GC, (3) Accept higher storage until GC runs. Trade-off: storage vs rollback capability.)
(codex, 2026-02-05, accept with caveat: VERSIONING may supersede RefCount; document the new GC/retention expectations and update CONTENT-ADDRESS to reflect the override)
(codex, 2026-02-07, accept: CONTENT-ADDRESS now notes the override; GC plan still needs explicit policy)

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

**Note:** Cost/benefit is favorable when audit/time-travel/rollback are hard requirements.
(codex, 2026-02-05, decision: cost/benefit is favorable only when audit/time-travel/rollback are hard requirements; otherwise complexity and storage cost are not justified)
(codex, 2026-02-07, accept: consistent with requirements stated in VERSIONING)

---

## Summary

| Feature | Status |
|---------|--------|
| Time-travel queries | Enabled via ValidSince in key |
| Topology rollback | Enabled via close-old/create-new pattern |
| Content rollback | Enabled via EdgeVersionHistory/NodeVersionHistory + append-only summaries |
| Summary GC | Orphan Index with retention-based deletion (enables rollback) |
| Multi-edge support | Explicit via AddEdge (fails if exists) |
| Retarget support | Explicit via UpdateEdge with new_dst |
| Fragments | Unchanged (already temporal) |

---

## Bitemporal Model: System Time vs Application Time

This design uses a **bitemporal model** with two orthogonal temporal dimensions:

| Dimension | Field | Purpose | Example |
|-----------|-------|---------|---------|
| **System Time** | `ValidSince`, `ValidUntil` | When this version existed in the DB | "Record created Nov 15, superseded Nov 20" |
| **Application Time** | `ActivePeriod` | When the entity is valid in the real world | "Promo runs Dec 1-7, 2025" |

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
    valid_range: ActivePeriod(Dec1, Dec7)  // APPLICATION TIME
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
    new_temporal_range: ActivePeriod(Dec1, Dec10),  // Changed!
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
    → Yes, Dec 5 is within ActivePeriod(Dec1, Dec10)

Q4: "Is the promo active on Dec 15?"
    → No, Dec 15 is outside ActivePeriod
```

### Example 13: Contract with Effective Dates and Amendments

A contract between OrgA and OrgB has an effective period, and amendments create new versions:

```
=== Jan 1: Contract signed, effective Feb 1 - Jan 31 next year ===
t=Jan1: AddEdge {
    src: OrgA, dst: OrgB, name: "contract",
    summary: { terms: "Standard", value: "$100K" },
    valid_range: ActivePeriod(Feb1, Jan31NextYear)  // APPLICATION TIME
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
// Note: ActivePeriod unchanged - contract still runs Feb 1 - Jan 31

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
    → Yes, Dec 15 is within ActivePeriod(Feb1, Jan31)

Q4: "Rollback to pre-amendment terms"
    → RestoreEdge { as_of: Feb1 }
    → Creates v=3 with hash=0xAAA (original terms), same ActivePeriod
```

### Example 14: Event with Moving Date Window

An "Annual Conference" edge tracks a recurring event where the business dates change each year:

```
=== Setup: Conference 2025 ===
t=Jun1: AddEdge {
    src: Company, dst: Venue, name: "annual_conference",
    summary: { year: 2025, attendees: 500 },
    valid_range: ActivePeriod(Sep15_2025, Sep17_2025)  // 3-day event
}

=== Sep 10: Venue conflict, reschedule to Oct ===
t=Sep10: UpdateEdge {
    src: Company, dst: Venue, name: "annual_conference",
    new_summary: { year: 2025, attendees: 500, rescheduled: true },
    new_temporal_range: ActivePeriod(Oct20_2025, Oct22_2025),  // New dates
    expected_version: 1
}

=== EdgeVersionHistory ===
(Company, Venue, "annual_conference", Jun1, v=1) → (t=Jun1, hash=0xAAA, wt=NULL, range=(Sep15,Sep17))
(Company, Venue, "annual_conference", Jun1, v=2) → (t=Sep10, hash=0xBBB, wt=NULL, range=(Oct20,Oct22))

=== Queries ===

Q1: "When was the conference originally scheduled?"
    → Time-travel to Sep 1: ActivePeriod(Sep15, Sep17)

Q2: "When is the conference now?"
    → Current: ActivePeriod(Oct20, Oct22)

Q3: "Show all schedule changes" (audit)
    → EdgeHistory: v1 (Sep 15-17), v2 (Oct 20-22)
```

### Key Distinction

| Query Type | Uses | Returns |
|------------|------|---------|
| "What did we know on date X?" | System time (`ValidSince`) | Version active at X |
| "Is this entity active on date X?" | Application time (`ActivePeriod`) | Boolean check |
| "Show all changes to this entity" | System time (Edge/NodeVersionHistory) | All versions with timestamps |
| "Find entities active during period P" | Application time scan | Entities where ActivePeriod overlaps P |

---

## Open Questions

1. **EdgeSummaryIndex schema and time semantics**
   - Define key/value layout and how "current" vs historical entries map to time-travel queries
   - Decision: Define schema explicitly before implementation

2. **Time-to-version lookup for `EdgeAtTime` / `NodeByIdAt`**
   - O(k) scan per edge where k=versions; acceptable if k stays small
   - Decision: Start with scan-by-version using `UpdatedAt`; add time-ordered index only if needed

3. **Summary/fragment index maintenance on updates and restores**
   - Enumerate index maintenance steps per mutation (Add/Update/Delete/Restore)
   - Decision: Document before implementation

4. **History retention/GC policy**
   - Decision: Unbounded history for MVP; add retention policies later based on workload

5. **Concurrency semantics and conflict resolution**
   - Decision: Version mismatch aborts; topology changes always close prior row

6. **Restore interval behavior**
   - Decision: Restores create new `valid_since` intervals (never reopen old intervals)

---

## Appendix: Design Review Notes

This section captures the design review discussion for historical reference.

### Review Summary (2026-02-05)

| Topic | Verdict | Resolution |
|-------|---------|------------|
| Multi-edge constraint | REJECT gap claim | Design intentionally prevents duplicate `(src,dst,name)`; use different `name` values |
| UpdatedAt in VersionHistory | ACCEPT | Added to enable time→version mapping |
| RefCount conflict | RESOLVED | Orphan Index design keeps RefCount but defers deletion; see [GC section](#garbage-collection) |
| EdgeSummaryIndex schema | ACCEPT | Schema defined in CONTENT-ADDRESS.md |
| Storage overhead | ACCEPT | Corrected to ~451 MB for 1M edges with 3 versions |
| Fragment guidance | ACCEPT | Examples added in [Examples: Fragments](#examples-fragments) |

### Key Design Decisions

1. **Edge identity is `(src, dst, name)`** - Unique at any point in time. For different relationship types between the same nodes, use different `name` values.

2. **Temporal versioning is for HISTORY, not concurrent duplicates** - Multiple "current" edges between same nodes must have different `name` values.

3. **Orphan Index supersedes inline RefCount deletion** - Enables rollback by preserving summaries until GC retention expires.

4. **Restores create new intervals** - Never reopen old `valid_since` intervals; always create new ones for audit clarity.
