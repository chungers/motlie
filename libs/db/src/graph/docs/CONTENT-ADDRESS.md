# CONTENT-ADDRESS: Reverse Index for SummaryHash → Id

## Problem

Graph stores content-addressed summaries:
- `NodeSummaryCfKey(SummaryHash)` → `NodeSummary`
- `EdgeSummaryCfKey(SummaryHash)` → `EdgeSummary`

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

1) `NodeSummaryIndex`
   - **Key:** `[summary_hash: 8] + [node_id: 16] = 24 bytes`
   - **Value:** empty
   - Purpose: reverse lookup for nodes referencing a summary hash

2) `EdgeSummaryIndex`
   - **Key:** `[summary_hash: 8] + [forward_edge_key: 40] = 48 bytes`
   - **Value:** empty
   - Purpose: reverse lookup for edges referencing a summary hash

All keys are **lexicographically ordered by `summary_hash` first**, enabling
prefix scans for a single hash.

### Encoding Source of Truth

Reuse existing `HotColumnFamilyRecord` key encoders in `graph/schema.rs`:
- `NodeCfKey` for node IDs
- `ForwardEdgeCfKey` for edges
- `SummaryHash::to_bytes()` for hash bytes

No custom serialization is introduced; the index key is a concatenation of
existing canonical encodings.

## Write Path Changes

### Node insert/update

When writing `NodeCfValue` with a `summary_hash`:
1) Insert `NodeSummaryIndex` key `(summary_hash, node_id)`
2) If summary hash changed, delete the old `(old_hash, node_id)` key

### Edge insert/update

When writing `ForwardEdgeCfValue` with a `summary_hash`:
1) Insert `EdgeSummaryIndex` key `(summary_hash, forward_edge_key)`
2) If summary hash changed, delete the old `(old_hash, forward_edge_key)` key

### Deletes

On node/edge delete:
- Remove the corresponding index key(s) for the stored summary hash.

## Read / Query API

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

## Schema / File Changes

**Core**
- `libs/db/src/graph/schema.rs`
  - Add `NodeSummaryIndex` and `EdgeSummaryIndex` CFs
  - Define key types and `key_to_bytes` / `key_from_bytes`
  - CF options (block cache, bloom) as in other small index CFs

**Mutation**
- `libs/db/src/graph/mutation.rs`
  - Update node/edge write paths to maintain index entries
  - Track old summary hashes for update/delete cleanup

**Query**
- `libs/db/src/graph/query.rs`
  - Add lookup + iterator APIs above

**Tests**
- `libs/db/src/graph/tests.rs`
  - Insert node/edge with summary hash; verify reverse lookup
  - Update summary hash; ensure old index removed
  - Delete node/edge; ensure index removed

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
