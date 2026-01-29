# IDMAP: Polymorphic External ID Mapping for Vector Embeddings

## Problem

`vector::IdForward` / `vector::IdReverse` only map between `vector::VecId` and
`graph::Id` (node IDs). This is insufficient for embeddings of:

- `NodeSummary` (content-addressed by `SummaryHash`)
- `NodeFragments` (keyed by `NodeId + Timestamp`)
- `EdgeSummary` (content-addressed by `SummaryHash`)
- `EdgeFragments` (keyed by `SrcId + DstId + NameHash + Timestamp`)

We need a **single mapping layer** that can associate a vector with any graph
entity key while preserving fast lookups and simple reverse resolution in search.

## Goals

- Support **typed external keys** (node, edge, fragment, summary, etc.)
- Preserve **fast lookup** by embedding + external key → vec_id
- Preserve **fast reverse mapping** vec_id → external key
- Maintain **prefix locality** by embedding code for range scans
- Provide **backward compatibility** for existing Node ID mappings

## Non-Goals

- Not changing HNSW storage or vector payload schema
- Not introducing multi-map (one vec_id → many external keys)
- Not altering graph schema keys

## External Key Types

Graph keys are already defined in `libs/db/src/graph/schema.rs`. We will reuse
their byte encodings and add a **type tag**.

| Type | Source Key | Byte Size (payload) | Notes |
|------|------------|---------------------|-------|
| `NodeId` | `NodeCfKey(Id)` | 16 | Node identity |
| `NodeFragment` | `NodeFragmentCfKey(Id, TimestampMilli)` | 24 | Node fragment |
| `Edge` | `ForwardEdgeCfKey(SrcId, DstId, NameHash)` | 40 | Edge identity (directional) |
| `EdgeFragment` | `EdgeFragmentCfKey(SrcId, DstId, NameHash, TimestampMilli)` | 48 | Edge fragment |
| `NodeSummary` | `NodeSummaryCfKey(SummaryHash)` | 8 | Content-addressed |
| `EdgeSummary` | `EdgeSummaryCfKey(SummaryHash)` | 8 | Content-addressed |

**Note:** `NodeSummary`/`EdgeSummary` are content-addressed. Multiple nodes/edges
may share the same summary hash. If uniqueness is required, use `NodeId`/`Edge`.

## Key Encoding

We follow the `GraphMetaField` pattern: a **tagged union** with a stable
discriminant.

```text
IdMapKey := [tag: u8] + [payload: fixed bytes]
```

Proposed tags (stable, do not renumber):

```
0x01 NodeId
0x02 NodeFragment
0x03 Edge
0x04 EdgeFragment
0x05 NodeSummary
0x06 EdgeSummary
```

**Forward map key** (embedding scoped):

```
IdForwardKey := [embedding_code: u64] + IdMapKey
```

**Reverse map key** (embedding scoped):

```
IdReverseKey := [embedding_code: u64] + [vec_id: u32]
IdReverseVal := IdMapKey
```

This preserves prefix locality on `embedding_code` for range scans and keeps
reverse lookup O(1).

## Column Families

**Option A (DECISION):** replace `IdForward` and `IdReverse` schemas (breaking)
- Pros: no extra CFs, clean data model
- Cons: **full migration required** (breaking change)

**Option B:** add new CFs (`IdMapForward`, `IdMapReverse`) and deprecate old
- Pros: migration can be lazy; easier rollback
- Cons: additional CF complexity

Decision: **Option A**. This is a breaking change and requires full migration
of existing IdForward/IdReverse records.

## API Changes

Add a new public type:

```rust
pub enum ExternalKey {
    NodeId(Id),
    NodeFragment(Id, TimestampMilli),
    Edge(Id, Id, NameHash),
    EdgeFragment(Id, Id, NameHash, TimestampMilli),
    NodeSummary(SummaryHash),
    EdgeSummary(SummaryHash),
}
```

Changes:
- `InsertVector` should accept `ExternalKey` instead of `Id` (with a helper
  `ExternalKey::NodeId` for backward compatibility).
- `DeleteVector` should accept `ExternalKey` instead of `Id`.
- Search result should return `ExternalKey` (typed), not `Id` only.
- Bench tools should use `ExternalKey::NodeId` for synthetic datasets.

### Serde & Wire Format

- `ExternalKey` needs **manual binary encoding** for RocksDB keys/values:
  - `ExternalKey::to_bytes()` / `from_bytes()` using `[tag: u8] + payload`
- Add `serde::{Serialize, Deserialize}` for JSON outputs (bench/admin/query).
- Wire format must be stable; **do not reorder tags**.

### Cache Impact

- `NavigationCache` and `BinaryCodeCache` remain keyed by `(embedding, vec_id)`,
  so **no changes** required.
- Any ID mapping caches must change to:
  - `(embedding, ExternalKey) -> vec_id`
  - `(embedding, vec_id) -> ExternalKey`
  (Optional; RocksDB-only lookups are acceptable initially.)

### Public MPSC Channel API

- Mutation messages should carry `ExternalKey` instead of `Id`:
  - `InsertVector`, `InsertVectorBatch`
  - `DeleteVector`
- Query result payloads should include `ExternalKey` instead of plain `Id`.
- Provide convenience constructors for node-only use:
  - `InsertVector::new_node_id(...)`
  - `DeleteVector::new_node_id(...)`

### Processor API

- Lookup helpers must be updated:
  - `vec_id_for_external(embedding, ExternalKey) -> VecId`
  - `external_for_vec_id(embedding, VecId) -> ExternalKey`
- Insert/Delete must write/remove **both** forward and reverse mappings for
  the new key type.
- Search results should resolve `ExternalKey` from reverse map values.

### SearchKNN / SearchResult Signatures

Use a union type in the public API for polymorphic external keys.

**New public type:**

```rust
pub enum ExternalKey {
    NodeId(Id),
    NodeFragment(Id, TimestampMilli),
    Edge(Id, Id, NameHash),
    EdgeFragment(Id, Id, NameHash, TimestampMilli),
    NodeSummary(SummaryHash),
    EdgeSummary(SummaryHash),
}
```

**Processor::search results:**

```rust
pub struct SearchResult {
    pub external: ExternalKey, // typed external key
    pub vec_id: VecId,
    pub distance: f32,
}

pub fn search(
    &self,
    embedding: &Embedding,
    query: &[f32],
    k: usize,
    config: SearchConfig,
) -> Result<Vec<SearchResult>>;
```

**SearchKNN public API:**

```rust
pub struct SearchKNN { /* unchanged builder */ }

impl SearchKNN {
    pub fn execute_with_processor(
        &self,
        processor: &Processor,
    ) -> Result<Vec<SearchResult>>;
}
```

**Convenience accessors (optional):**

```rust
impl SearchResult {
    pub fn node_id(&self) -> Option<Id> { /* only for ExternalKey::NodeId */ }
}
```

**Reader API / channel dispatch:**

```rust
pub async fn search(&self, query: SearchKNN) -> Result<Vec<SearchResult>>;
```

## Admin & Validation

Admin tools must:
- Display the **external key type** and payload in `admin vectors`.
- Validate that forward/reverse mappings serialize/deserialize for all key types.
- Provide summary counts by external key type in `admin stats`.

## Bench Tools

`bench_vector` should:
- Store mappings as `ExternalKey::NodeId` for the random dataset.
- Print typed external keys in query results if `--json` is used.

## Caching Considerations

Mapping more entity types increases the number of IdForward/IdReverse records.
Add a tuning parameter to the vector block cache configuration to avoid cache
thrash at scale.

**Proposed config:**
- `VectorBlockCacheConfig::id_map_cache_fraction` (default: `0.10`)
- Applied to IdForward/IdReverse CF options to reserve a fraction of the shared
  cache for ID mapping lookups.

This should be adjustable in `VectorConfig` for deployments with heavy mapping
traffic (e.g., many fragments per node/edge).

## Tests

Add tests for:
- Serialization round-trips for each `ExternalKey` variant
- Forward + reverse mapping consistency for all variants
- Admin validation on mixed key types

## Impacted Files

**Core**
- `libs/db/src/vector/schema.rs` (new CFs or updated IdForward/IdReverse)
- `libs/db/src/vector/ops/insert.rs` (write mapping)
- `libs/db/src/vector/ops/delete.rs` (remove mapping)
- `libs/db/src/vector/processor.rs` (search result mapping)
- `libs/db/src/vector/query.rs` (public query result types)
- `libs/db/src/vector/admin.rs` (vector info, validation)

**Graph integration**
- `libs/db/src/graph/schema.rs` (import key encodings; no changes expected)

**Bench/CLI**
- `bins/bench_vector/src/commands.rs` (mapping type, output)

**Tests**
- `libs/db/tests/test_vector_*` (insert/delete/search mappings)
- `bins/bench_vector` tests if present

## Migration Plan (Option A, Breaking)

1. Add new `ExternalKey` encoding and update IdForward/IdReverse schemas in-place.
2. Provide a one-time offline migration tool to rewrite all IdForward/IdReverse
   keys/values to the new tagged format.
3. Bump storage version and require migration on startup (fail fast if old format).
4. Update all read/write paths to use the new format only.
5. Remove any fallback logic after migration window closes.

## Open Questions

**Decisions:**

1) **Edge mapping**: use **ForwardEdge only**. Reverse edges are derivable.
2) **Summary mapping**: keep **1:1 per embedding** (one ExternalKey per vec_id)
   for simplicity and determinism within an embedding space.
