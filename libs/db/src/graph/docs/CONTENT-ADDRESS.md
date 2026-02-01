# CONTENT-ADDRESS: Reverse Index for SummaryHash → Id

## Problem

Graph stores content-addressed summaries:
- `NodeSummaryCfKey(SummaryHash)` → `NodeSummary`
- `EdgeSummaryCfKey(SummaryHash)` → `EdgeSummary`

<!-- @Codex: See schema.rs:705-763 for NodeSummaries CF, schema.rs:765-830 for EdgeSummaries CF -->

But there is no reverse mapping from `SummaryHash` back to the owning
`NodeId` / `Edge` keys. This makes it impossible to resolve a content-hash
returned by vector search into concrete graph entities without a full scan.

## Goals

- Add an efficient reverse index: `SummaryHash → Vec<Id>`
- Avoid large value updates (no `Vec<Id>` stored as values)
- Keep write amplification low
- Support prefix/range scans by `SummaryHash`

## Design

Use **multimap keys** in new CFs. Each `(SummaryHash, EntityKey)` pair is a
distinct key with an empty value.

### Column Families

<!-- @Codex: Follow naming conventions in schema.rs:1-44 -->
<!-- @Codex: Use ColumnFamilySerde trait (not HotColumnFamilyRecord) since values are empty -->

1) `NodeSummaryIndex`
   - **Key:** `[summary_hash: 8] + [node_id: 16] = 24 bytes`
   - **Value:** empty
   - Purpose: reverse lookup for nodes referencing a summary hash

2) `EdgeSummaryIndex`
   - **Key:** `[summary_hash: 8] + [forward_edge_key: 40] = 48 bytes`
   - **Value:** empty
   - Purpose: reverse lookup for edges referencing a summary hash
(codex, 2026-02-01 10:58:19 -0800, question) If delete/update flows only have a reverse edge key, do we need an extra lookup to compute the forward key for cleanup?
(codex, 2026-02-01 11:26:11 -0800, status) ReverseEdgeCfKey stores (dst_id, src_id, name_hash), so the forward key can be computed by swapping (src,dst) without a DB lookup.

All keys are **lexicographically ordered by `summary_hash` first**, enabling
prefix scans for a single hash.

<!-- @Codex: Use 8-byte prefix extractor for SummaryHash prefix scans, similar to schema.rs:531 -->

### Encoding Source of Truth

Reuse existing `HotColumnFamilyRecord` key encoders in `graph/schema.rs`:
- `NodeCfKey` for node IDs → `Nodes::key_to_bytes()` at schema.rs:331-333
- `ForwardEdgeCfKey` for edges → `ForwardEdges::key_to_bytes()` at schema.rs:568-576
- `SummaryHash::as_bytes()` → summary_hash.rs:84

No custom serialization is introduced; the index key is a concatenation of
existing canonical encodings.

## Write Path Changes

<!-- @Codex: Modify AddNode::execute() at mutation.rs:469-505 -->
<!-- @Codex: Modify AddEdge::execute() at mutation.rs:545-593 -->

### Node insert/update

When writing `NodeCfValue` with a `summary_hash`:
1) Insert `NodeSummaryIndex` key `(summary_hash, node_id)`
2) If summary hash changed, delete the old `(old_hash, node_id)` key
(codex, 2026-02-01 10:58:19 -0800, question) Are these index writes guaranteed to be in the same write batch/WAL as the primary CF write so readers never see a partial update?
(codex, 2026-02-01 11:26:11 -0800, status) AddNode/AddEdge already use a RocksDB Transaction and write all CFs via the same txn, so commit should be atomic across CFs.

<!-- @Codex: Step 2 requires reading old value first via txn.get_cf() before writing -->

### Edge insert/update

When writing `ForwardEdgeCfValue` with a `summary_hash`:
1) Insert `EdgeSummaryIndex` key `(summary_hash, forward_edge_key)`
2) If summary hash changed, delete the old `(old_hash, forward_edge_key)` key
(codex, 2026-02-01 10:58:19 -0800, question) Do all callers have the prior summary hash available? If some updates are blind overwrites, this can leave stale index keys.
(codex, 2026-02-01 11:26:11 -0800, status) Current mutations don't read the existing node/edge; AddNode/AddEdge are blind upserts, so you'd need a txn.get_cf() read to remove old index keys if summary can change.

### Deletes

On node/edge delete:
- Remove the corresponding index key(s) for the stored summary hash.

<!-- @Codex: No DeleteNode/DeleteEdge mutations exist yet - see OPEN QUESTIONS -->

## Read / Query API

<!-- @Codex: Add to query.rs, follow pattern of resolve_node_summary() at query.rs:117-147 -->

Add graph query helpers to resolve a summary hash back to concrete entities.

### Exact lookup (materialize Vec<Id>)

```rust
pub fn node_ids_for_summary(
    &self,
    summary: SummaryHash,
) -> Result<Vec<Id>>;

pub fn edges_for_summary(
    &self,
    summary: SummaryHash,
) -> Result<Vec<ForwardEdgeCfKey>>;
```

### Prefix scan (streaming / iterator)

```rust
pub fn node_ids_for_summary_iter(
    &self,
    summary: SummaryHash,
) -> Result<impl Iterator<Item = Result<Id>>>;

pub fn edges_for_summary_iter(
    &self,
    summary: SummaryHash,
) -> Result<impl Iterator<Item = Result<ForwardEdgeCfKey>>>;
```

### Prefix bytes (for scan utilities)

```rust
pub fn node_summary_index_prefix(summary: SummaryHash) -> [u8; 8];
pub fn edge_summary_index_prefix(summary: SummaryHash) -> [u8; 8];
```
(codex, 2026-02-01 10:58:19 -0800, status) This hard-codes the SummaryHash length; consider calling out the invariant explicitly or deriving the length from `SummaryHash::to_bytes()` if it might change.

## Schema / File Changes

**Core**
- `libs/db/src/graph/schema.rs`
  - Add `NodeSummaryIndex` and `EdgeSummaryIndex` CFs
  - Define key types and `key_to_bytes` / `key_from_bytes`
  - CF options (block cache, bloom) as in other small index CFs
  - Register in `ALL_COLUMN_FAMILIES` array

<!-- @Codex: Reference patterns:
  - CF struct: schema.rs:78 (Names)
  - ColumnFamily impl: schema.rs:86-88
  - ColumnFamilyConfig impl: schema.rs:90-116
  - ColumnFamilySerde impl: schema.rs:118-139
  - Prefix extractor: schema.rs:531 (ForwardEdges uses 16-byte prefix)
-->

**Mutation**
- `libs/db/src/graph/mutation.rs`
  - Update node/edge write paths to maintain index entries
  - Track old summary hashes for update/delete cleanup

<!-- @Codex: Reference patterns:
  - AddNode::execute(): mutation.rs:469-505
  - AddEdge::execute(): mutation.rs:545-593
  - txn.put_cf() usage: mutation.rs:493, 583
  - txn.get_cf() for reads: mutation.rs:347-349
-->

**Query**
- `libs/db/src/graph/query.rs`
  - Add lookup + iterator APIs above

<!-- @Codex: Reference patterns:
  - resolve_node_summary(): query.rs:117-147
  - iterate_cf! macro: query.rs:334-371
  - prefix iteration: use db.prefix_iterator_cf() or iterator_cf with From mode
-->

**Tests**
- `libs/db/src/graph/tests.rs`
  - Insert node/edge with summary hash; verify reverse lookup
  - Update summary hash; ensure old index removed
  - Delete node/edge; ensure index removed (defer if Q1 = Option A)

<!-- @Codex: Reference test patterns at tests.rs:18-61 -->

## Performance / Write Amplification

- Adds **one extra small write** per node/edge with a summary hash
- Uses multimap keys (no large value updates)
- Deletes on summary change add one extra delete

This overhead is expected to be negligible relative to node/edge writes.

## Notes

- Index enables resolving `ExternalKey::NodeSummary` / `EdgeSummary` results
  into concrete graph entities without a full scan.
- If multiple nodes/edges share the same summary hash, the index returns all
  of them.

---

## OPEN QUESTIONS (for Codex review)

### Q1: Delete Mutations

The plan references "On node/edge delete" but **no `DeleteNode` or `DeleteEdge` mutations exist** in the current codebase (see `mutation.rs` Mutation enum at line 95-113).

**Options:**
- A) Defer delete index cleanup to future work when delete mutations are added
- B) Add `DeleteNode` / `DeleteEdge` mutations as part of this PR
(codex, 2026-02-01 11:26:11 -0800, status) Based on the current Mutation enum, there are no delete mutations to hook into, so A is consistent unless you plan to expand the mutation API in this PR.
- C) Use soft deletes via `UpdateNodeValidSinceUntil` (set `until` to past) - index stays but entity is logically deleted

**Recommendation:** Option A - defer. Index entries for deleted nodes are harmless (lookup returns entity not found).

---

### Q2: Update Semantics - Read Before Write

Step 2 of write path says "If summary hash changed, delete old index entry." This requires **reading the old value first** to get the previous `summary_hash`.

**Trade-offs:**
- **Pro:** Clean index, no stale entries
- **Con:** Adds one read per write (txn.get_cf before put_cf)

**Options:**
- A) Always read old value, delete old index entry if hash changed (clean index)
- B) Skip read, allow duplicate index entries (periodic cleanup job)
- C) Only read if this is a known update (caller provides `is_update` flag)

**Recommendation:** Option A - the read is cheap (same transaction, likely cached), and a clean index avoids subtle bugs.

---

### Q3: CF Registration Location

New CFs must be added to `ALL_COLUMN_FAMILIES` at `schema.rs:834`. Should the index CFs be:
- A) Grouped with other index CFs (after `ReverseEdges`)
- B) Grouped with summary CFs (`NodeSummaries`, `EdgeSummaries`)

**Recommendation:** Option B - logically related to summary storage.

---

### Q4: Empty Value Representation

Index entries have empty values. Options for value type:
- A) `()` unit type with custom serde (0 bytes)
- B) `struct EmptyValue;` marker type
- C) Use raw `put_cf(key, &[])` without value type

**Recommendation:** Option A or C - minimal overhead. Option C is simplest if we bypass `ColumnFamilySerde::value_to_bytes`.
