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
- Maintain a clean model for **new databases** (no legacy format support)

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

**Invariant:** Each vec_id must map to **exactly one** `ExternalKey`. This design
does **not** support multi-map (one vec_id → many external keys). If you need
both `NodeId` and `NodeSummary` to point to the same vector, you must choose a
single canonical `ExternalKey` and document/encode that choice at insert time.
The reverse lookup always returns that canonical key.

**Lookup flow for summaries:** Summary embeddings are addressed by
`SummaryHash`, not by `NodeId`/`Edge`. The vector layer treats `NodeSummary` and
`EdgeSummary` as *independent* key types and does not traverse graph tables.
Callers must perform the join explicitly:

1) **Resolve summary hash** from graph storage (e.g., `NodeSummaryCf` /
   `EdgeSummaryCf` or whatever table exposes the summary hash for a node/edge).
2) **Construct ExternalKey** as `ExternalKey::NodeSummary(summary_hash)` or
   `ExternalKey::EdgeSummary(summary_hash)`.
3) **Query vector** using the summary external key.

If a caller only has a `NodeId`/`Edge` and does not resolve the summary hash,
the vector lookup will not find a mapping and must return a not-found error.

**Encoding source of truth:** Vector key serialization will **reuse the
`HotColumnFamilyRecord` implementations in `graph/schema.rs`** to
encode/decode the corresponding `*CfKey` types. Canonical encoders/decoders:

- `Nodes` → `NodeCfKey`
  - `HotColumnFamilyRecord::key_to_bytes`
  - `HotColumnFamilyRecord::key_from_bytes`
- `NodeFragments` → `NodeFragmentCfKey`
  - `HotColumnFamilyRecord::key_to_bytes`
  - `HotColumnFamilyRecord::key_from_bytes`
- `ForwardEdges` → `ForwardEdgeCfKey`
  - `HotColumnFamilyRecord::key_to_bytes`
  - `HotColumnFamilyRecord::key_from_bytes`
- `EdgeFragments` → `EdgeFragmentCfKey`
  - `HotColumnFamilyRecord::key_to_bytes`
  - `HotColumnFamilyRecord::key_from_bytes`
- `NodeSummaries` → `NodeSummaryCfKey`
  - `HotColumnFamilyRecord::key_to_bytes`
  - `HotColumnFamilyRecord::key_from_bytes`
- `EdgeSummaries` → `EdgeSummaryCfKey`
  - `HotColumnFamilyRecord::key_to_bytes`
  - `HotColumnFamilyRecord::key_from_bytes`

This keeps encoding canonical with the graph layer and avoids duplicating byte
layouts.

**Serde strategy note (non-risk):** Graph CFs use a mix of rkyv and rmp *for
values*. The `HotColumnFamilyRecord` key helpers are **manual, fixed-layout
encoders** and do **not** depend on the value serde. IDMAP will continue to use
manual byte layouts (`[tag: u8] + payload`) for both forward and reverse
entries. Therefore the rkyv/rmp split in graph **does not affect** IDMAP key
serialization or `ExternalKey` storage.

**Risk boundary:** The only coupling risk is if graph **key** layouts change
(byte order/size). That is already a storage-breaking change and should be
versioned at the graph/vector layer together.

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
- If per-CF cache partitioning is available, apply to IdForward/IdReverse CFs to
  reserve a fraction of the shared cache for ID mapping lookups.
- If per-CF quotas are not supported, implement by allocating a **dedicated**
  block cache for IdForward/IdReverse or drop this tuning knob.

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

## Migration Plan

**Breaking change.** Existing databases using legacy `IdForward`/`IdReverse`
must be migrated or rebuilt. For now, this is acceptable in heavy development:
no compatibility guarantees are provided for legacy data.

## Open Questions

**Decisions:**

1) **Edge mapping**: use **ForwardEdge only**. Reverse edges are derivable.
2) **Summary mapping**: keep **1:1 per embedding** (one ExternalKey per vec_id)
   for simplicity and determinism within an embedding space.

---

## Gap Analysis

The following gaps and ambiguities have been identified during implementation review:

### G1: Storage Version Tracking ~~(HIGH)~~ N/A

**Gap:** ~~Document mentions "bump storage version" but lacks specifics.~~

**Resolution:** Not needed during heavy development. No existing databases to version.

### G2: Duplicate ExternalKey Definition (LOW)

**Gap:** The `ExternalKey` enum is defined twice in the document (API Changes
section and SearchKNN section). Should consolidate to a single definition.

**Action:** Remove duplicate definition from SearchKNN section.

### G3: Graph Schema Key Dependencies (MEDIUM)

**Gap:** Document references graph schema key types but doesn't verify they exist
or specify import paths.

**Required imports from `libs/db/src/graph/schema.rs`:**
- `NodeCfKey` (exists) → verify encoding matches
- `NodeFragmentCfKey` (exists) → verify encoding matches
- `ForwardEdgeCfKey` (exists) → verify 40-byte encoding
- `EdgeFragmentCfKey` (exists) → verify 48-byte encoding
- `SummaryHash` → verify this type exists (may need to add)

**Action:** Confirm byte sizes in `graph/schema.rs` before relying on the size
table above. If sizes differ, update this document and the serialization tests.

### G4: Migration Tool Specification ~~(HIGH)~~ N/A

~~**Gap:** Migration tool is mentioned but lacks implementation details.~~

**Resolution:** Not needed. Heavy development - no existing databases.

### G5: Error Handling Types (MEDIUM)

**Gap:** No explicit error types for key deserialization failures.

**Required error variants:**
```rust
pub enum IdMapError {
    UnknownTag(u8),
    InvalidKeyLength { expected: usize, actual: usize },
    KeyTypeMismatch { expected: &'static str, actual: &'static str },
    DecodeError(String),
}
```

### G6: Test Coverage Matrix (MEDIUM)

**Gap:** Tests section lacks specific edge cases.

**Additional tests needed:**
- Negative: invalid tag byte (0xFF)
- Negative: truncated key payloads
- Boundary: maximum valid timestamp (u64::MAX)
- Boundary: maximum valid NameHash
- Performance: 1M key serialization/deserialization benchmark

### G7: Memory Impact Estimation (LOW)

**Gap:** No quantitative analysis of memory overhead.

**Analysis:**
| Type | Old Size | New Size | Overhead |
|------|----------|----------|----------|
| NodeId | 16 | 17 (+tag) | +6.25% |
| NodeFragment | N/A | 25 | N/A |
| Edge | N/A | 41 | N/A |
| EdgeFragment | N/A | 49 | N/A |
| NodeSummary | N/A | 9 | N/A |
| EdgeSummary | N/A | 9 | N/A |

For 1M vectors with 80% NodeId keys:
- Old: 1M × 16 = 16 MB
- New: 800K × 17 + 200K × 25 (avg) = 18.6 MB (+16%)

### G8: Rollback Strategy ~~(HIGH)~~ N/A

~~**Gap:** No explicit rollback plan if migration fails mid-way.~~

**Resolution:** Not needed. Heavy development - no existing databases.

### G9: Admin Stats Implementation (LOW)

**Gap:** "Summary counts by external key type" mentioned but not detailed.

**Specification:**
```json
{
  "embedding_code": 12345,
  "total_vectors": 100000,
  "by_key_type": {
    "NodeId": 80000,
    "NodeFragment": 15000,
    "Edge": 3000,
    "EdgeFragment": 1500,
    "NodeSummary": 300,
    "EdgeSummary": 200
  }
}
```

### G10: Cache Fraction Tuning Guidance (LOW)

**Gap:** Default 0.10 cache fraction mentioned but no tuning guidance.

**Recommendation:**
- Light workload (mostly NodeId): 0.05
- Mixed workload (fragments): 0.10 (default)
- Heavy fragment workload: 0.15-0.20
- Add observability metric: `idmap_cache_hit_ratio`

### G11: Serde Strategy Mismatch (RESOLVED)

**Concern:** Graph CFs use rkyv/rmp for values while IDMAP uses manual bytes.

**Resolution:** IDMAP reuses only **key encoders** (`HotColumnFamilyRecord`
`key_to_bytes` / `key_from_bytes`), which are fixed-layout and do not depend on
value serde. IDMAP forward/reverse entries remain manual bytes. This is not a
risk unless graph **key** layouts change, which is already a storage-breaking
event that must be versioned.

---

## Implementation Tasks

### Phase 1: Core Type Definitions

- [ ] **T1.1**: Define `ExternalKey` enum in `libs/db/src/vector/schema.rs`
  - Add tag constants (0x01-0x06)
  - Implement `to_bytes()` / `from_bytes()`
  - Add unit tests for round-trip serialization

- [ ] **T1.2**: Define `IdMapError` error type in `libs/db/src/vector/error.rs`

- [ ] **T1.3**: Verify graph schema key imports
  - Confirm `NodeFragmentCfKey`, `ForwardEdgeCfKey`, `EdgeFragmentCfKey` exist
  - Add `SummaryHash` type if missing

- [x] ~~**T1.4**: Add `StorageVersion` field to `GraphMetaField` enum~~ N/A - no versioning needed

### Phase 2: Schema Changes

- [ ] **T2.1**: Update `IdForwardCfKey` to use `ExternalKey` instead of `Id`
  - Key size changes from 24 to 8 + (1 + payload_size)

- [ ] **T2.2**: Update `IdReverseCfValue` to store `ExternalKey` instead of `Id`
  - Value size changes from 16 to (1 + payload_size)

- [ ] **T2.3**: Update `IdForward::key_to_bytes()` / `key_from_bytes()`

- [ ] **T2.4**: Update `IdReverse::value_to_bytes()` / `value_from_bytes()`

### Phase 3: API Changes

- [ ] **T3.1**: Update `InsertVector` mutation to accept `ExternalKey`
  - Add `ExternalKey::NodeId(id)` convenience constructor
  - Preserve backward compatibility via `InsertVector::new_node()`

- [ ] **T3.2**: Update `DeleteVector` mutation to accept `ExternalKey`

- [ ] **T3.3**: Update `SearchResult` to return `ExternalKey`
  - Add `SearchResult::node_id()` convenience accessor

- [ ] **T3.4**: Update `GetInternalId` query to accept `ExternalKey`

- [ ] **T3.5**: Update `GetExternalId` query to return `ExternalKey`

- [ ] **T3.6**: Update `ResolveIds` query to return `Vec<Option<ExternalKey>>`

### Phase 4: Processor Updates

- [ ] **T4.1**: Update `processor.rs` lookup helpers
  - `vec_id_for_external(embedding, ExternalKey) -> VecId`
  - `external_for_vec_id(embedding, VecId) -> ExternalKey`

- [ ] **T4.2**: Update insert operation to write new key format

- [ ] **T4.3**: Update delete operation to handle new key format

- [ ] **T4.4**: Update search result resolution

### Phase 5: Migration Tool - N/A

~~No migration needed during heavy development. Implement new format directly.~~

- [x] ~~**T5.1-T5.5**: All migration tasks~~ N/A

### Phase 6: Admin & Validation

- [ ] **T6.1**: Update `admin.rs` to display `ExternalKey` type in vector info

- [ ] **T6.2**: Add `admin stats --by-key-type` for counts by ExternalKey variant

- [ ] **T6.3**: Add validation for forward/reverse mapping consistency

### Phase 7: Bench Tools

- [ ] **T7.1**: Update `bench_vector` to use `ExternalKey::NodeId`

- [ ] **T7.2**: Add `--json` output for typed external keys

### Phase 8: Cache Tuning

- [ ] **T8.1**: Add `id_map_cache_fraction` to `VectorBlockCacheConfig`

- [ ] **T8.2**: Apply cache fraction to IdForward/IdReverse CF options

- [ ] **T8.3**: Add `idmap_cache_hit_ratio` metric

### Phase 9: Tests

- [ ] **T9.1**: Unit tests for `ExternalKey` serialization (all variants)

- [ ] **T9.2**: Integration tests for mixed key type operations

- [x] ~~**T9.3**: Migration tests (old → new format)~~ N/A

- [ ] **T9.4**: Negative tests (invalid tags, truncated payloads)

- [ ] **T9.5**: Performance benchmark: 1M key operations

### Phase 10: Documentation

- [ ] **T10.1**: Update ROADMAP.md with new schema

- [ ] **T10.2**: Update any other docs referencing IdForward/IdReverse

- [ ] **T10.3**: Remove duplicate ExternalKey definition from this document

---

## Dependencies

```
T1.1 ─┬─► T2.1 ─┬─► T3.1 ─┬─► T4.2
      │         │         │
T1.2 ─┤   T2.2 ─┤   T3.2 ─┤   T4.3
      │         │         │
T1.3 ─┘   T2.3 ─┤   T3.3 ─┤   T4.4
          T2.4 ─┘   T3.4 ─┤
                    T3.5 ─┤
                    T3.6 ─┘

T4.* ─► T6.* (admin needs updated processor)
T4.* ─► T7.* (bench needs updated processor)
T2.* ─► T8.* (cache config applies to CFs)

All ─► T9.* (tests after implementation)
All ─► T10.* (docs after implementation)
```

## Estimated Effort

| Phase | Tasks | Est. Effort |
|-------|-------|-------------|
| 1 - Core Types | 3 | Small |
| 2 - Schema | 4 | Medium |
| 3 - API | 6 | Medium |
| 4 - Processor | 4 | Medium |
| 5 - Migration | ~~5~~ 0 | ~~Large~~ N/A |
| 6 - Admin | 3 | Small |
| 7 - Bench | 2 | Small |
| 8 - Cache | 3 | Small |
| 9 - Tests | 4 | Medium |
| 10 - Docs | 3 | Small |
| **Total** | **32** | |

## Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Performance regression | MEDIUM | MEDIUM | Benchmark before/after |
| API breaking changes | LOW | HIGH | Expected in dev phase |
| Cache thrash at scale | LOW | MEDIUM | Tunable cache fraction |
