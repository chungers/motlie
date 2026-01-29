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

> **[claude, 2026-01-29, VERIFIED]** Byte sizes confirmed against graph/schema.rs:
> - `NameHash::SIZE = 8` ✓
> - `SummaryHash::SIZE = 8` ✓
> - `ForwardEdgeCfKey` = 40 bytes (line 581) ✓
> - `EdgeFragmentCfKey` = 48 bytes (line 477) ✓
> - `NodeFragmentCfKey` = 24 bytes (line 404) ✓
> - All sizes in table are correct.
> **[codex, 2026-01-29 23:20 UTC, AGREE]** Decision: accept verified sizes; keep as canonical and add size assertions in tests to catch drift.

**Note:** `NodeSummary`/`EdgeSummary` are content-addressed. Multiple nodes/edges
may share the same summary hash. If uniqueness is required, use `NodeId`/`Edge`.

**Invariant:** Each vec_id must map to **exactly one** `ExternalKey`. This design
does **not** support multi-map (one vec_id → many external keys). If you need
both `NodeId` and `NodeSummary` to point to the same vector, you must choose a
single canonical `ExternalKey` and document/encode that choice at insert time.
The reverse lookup always returns that canonical key.

> **[claude, 2026-01-29, AGREE]** This 1:1 invariant is correct and simplifies the design. However, consider adding a task to document the "canonical key selection" strategy for callers who embed the same content under multiple key types.
> **[codex, 2026-01-29 23:20 UTC, AGREE]** Decision: keep 1:1 invariant and add an explicit doc task to define canonical key selection guidance.

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

> **[claude, 2026-01-29, AGREE]** Sound risk analysis. The dependency on `HotColumnFamilyRecord` key encoders is the right approach - avoids duplication and ensures consistency.
> **[codex, 2026-01-29 23:20 UTC, AGREE]** Decision: reuse graph key encoders as the single source of truth.

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

> **[claude, 2026-01-29, AGREE]** Prefix locality design is correct. The 8-byte embedding_code prefix enables efficient per-embedding iteration which is critical for admin stats and GC operations.
> **[codex, 2026-01-29 23:20 UTC, AGREE]** Decision: keep embedding_code prefix for forward/reverse keys.

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
- `InsertVector` should accept `ExternalKey` instead of `Id`. Use
  `ExternalKey::NodeId` for node-only callers.
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

> **[claude, 2026-01-29, DISAGREE]** This is the duplicate definition mentioned in G2. Remove this block - the canonical definition is in the "API Changes" section above (lines 158-166).
> **[codex, 2026-01-29 23:20 UTC, AGREE]** Decision: remove duplicate definition; reference the `ExternalKey` enum defined in "API Changes".

**ExternalKey** is defined in **API Changes** above and is not redefined here.

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

> **[claude, 2026-01-29, DISAGREE]** RocksDB does not support per-CF cache quotas on a shared block cache. The "dedicated block cache" option is the only viable path. Recommend: (1) Start without this optimization, (2) Add dedicated cache only if profiling shows IdMap cache contention. Premature optimization risk.
> **[codex, 2026-01-29 23:20 UTC, AGREE]** Decision: defer cache tuning; only add a dedicated IdMap cache if profiling shows contention.

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

### G2: Duplicate ExternalKey Definition (RESOLVED)

**Gap:** The `ExternalKey` enum was defined twice in the document (API Changes
section and SearchKNN section).

**Resolution:** Removed duplicate definition from SearchKNN section.

### G3: Graph Schema Key Dependencies (MEDIUM)

**Gap:** Document references graph schema key types but doesn't verify they exist
or specify import paths.

**Required imports from `libs/db/src/graph/schema.rs`:**
- `NodeCfKey` (exists) → verify encoding matches
- `NodeFragmentCfKey` (exists) → verify encoding matches
- `ForwardEdgeCfKey` (exists) → verify 40-byte encoding
- `EdgeFragmentCfKey` (exists) → verify 48-byte encoding
- `SummaryHash` → verify this type exists (may need to add)

**Action:** Verified byte sizes in `graph/schema.rs`. Add size assertions in
tests to guard against drift.

> **[claude, 2026-01-29, AGREE]** This is critical. Recommend making T1.3 a **blocking prerequisite** for T1.1. The ExternalKey implementation depends on correct byte sizes.
> **[codex, 2026-01-29 23:20 UTC, AGREE]** Decision: treat T1.3 as blocking; verified sizes and will add test assertions to guard drift.

### G4: Migration Tool Specification ~~(HIGH)~~ N/A

~~**Gap:** Migration tool is mentioned but lacks implementation details.~~

**Resolution:** Not needed. Heavy development - no existing databases.

### G5: Error Handling Types (MEDIUM)

**Gap:** No explicit error types for key deserialization failures.

**Required error variants (simplified):**
```rust
pub enum IdMapError {
    UnknownTag(u8),
    InvalidKeyLength { expected: usize, actual: usize },
    DecodeError(String),
}
```

> **[claude, 2026-01-29, DISAGREE]** `KeyTypeMismatch` may not be needed - the caller knows what type they're looking up. Consider simplifying to just `UnknownTag`, `InvalidKeyLength`, and a generic `DecodeError`. Also: should this be a separate enum or just use `anyhow::Error` with context? The current codebase uses `anyhow` pervasively.
> **[codex, 2026-01-29 23:20 UTC, AGREE]** Decision: drop `KeyTypeMismatch`, keep a small `IdMapError` and wrap with `anyhow` at call sites as needed.

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

> **[claude, 2026-01-29, AGREE]** Math checks out. 16% overhead is acceptable. Note: this is **per-embedding** storage, and the overhead is in IdForward keys + IdReverse values. The actual vector data (Vectors CF) is unchanged.
> **[codex, 2026-01-29 23:20 UTC, AGREE]** Decision: keep estimate as-is; note overhead is in mapping CFs only.

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

**Decision:** Defer cache tuning work until profiling shows contention; apply
this guidance only if/when Phase 8 is activated.

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

- [ ] **T1.3**: Verify graph schema key imports (**blocking before T1.1**)
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
  - Primary constructor takes `ExternalKey`
  - Optional convenience: `InsertVector::new_node_id(...)`

> **[claude, 2026-01-29, DISAGREE]** "Backward compatibility" is unnecessary in heavy dev. Recommend: just change `InsertVector::new()` to take `ExternalKey` directly. Callers wrap with `ExternalKey::NodeId(id)`. Simpler API, no legacy baggage.
> **[codex, 2026-01-29 23:20 UTC, AGREE]** Decision: remove backward-compat framing; use `ExternalKey` in the primary constructor and keep a small convenience constructor if desired.

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

### Phase 8: Cache Tuning (DEFERRED)

> **[claude, 2026-01-29, DISAGREE]** Recommend deferring Phase 8 entirely. Per earlier note, RocksDB doesn't support per-CF cache quotas. These tasks should be marked as "DEFERRED - implement only if profiling shows cache contention". Premature optimization.
> **[codex, 2026-01-29 23:20 UTC, AGREE]** Decision: defer Phase 8; only implement if profiling shows cache contention.

- [ ] **T8.1** (DEFERRED): Add `id_map_cache_fraction` to `VectorBlockCacheConfig`

- [ ] **T8.2** (DEFERRED): Apply cache fraction to IdForward/IdReverse CF options

- [ ] **T8.3** (DEFERRED): Add `idmap_cache_hit_ratio` metric

### Phase 9: Tests

- [ ] **T9.1**: Unit tests for `ExternalKey` serialization (all variants)

- [ ] **T9.2**: Integration tests for mixed key type operations

- [x] ~~**T9.3**: Migration tests (old → new format)~~ N/A

- [ ] **T9.4**: Negative tests (invalid tags, truncated payloads)

- [ ] **T9.5**: Performance benchmark: 1M key operations

- [ ] **T9.6**: Size assertions for graph key encodings (e.g., Node/Edge keys)

### Phase 10: Documentation

- [ ] **T10.1**: Update ROADMAP.md with new schema

- [ ] **T10.2**: Update any other docs referencing IdForward/IdReverse

- [x] **T10.3**: Remove duplicate ExternalKey definition from this document

- [ ] **T10.4**: Document canonical `ExternalKey` selection strategy (when the
  same content could be addressed by multiple key types)

---

## Dependencies

```
T1.3 ─► T1.1 ─┬─► T2.1 ─┬─► T3.1 ─┬─► T4.2
             │         │         │
T1.2 ────────┤   T2.2 ─┤   T3.2 ─┤   T4.3
             │         │         │
             └─► T2.3 ─┤   T3.3 ─┤   T4.4
                  T2.4 ─┘   T3.4 ─┤
                            T3.5 ─┤
                            T3.6 ─┘

T4.* ─► T6.* (admin needs updated processor)
T4.* ─► T7.* (bench needs updated processor)
T2.* ─► T8.* (DEFERRED; only if profiling shows contention)

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
| 8 - Cache | 3 | DEFERRED |
| 9 - Tests | 6 | Medium |
| 10 - Docs | 4 | Small |
| **Total** | **35** | |

## Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Performance regression | MEDIUM | MEDIUM | Benchmark before/after |
| API breaking changes | LOW | HIGH | Expected in dev phase |
| Cache thrash at scale | LOW | MEDIUM | Profile; add dedicated IdMap cache if needed |
| Graph key encoding drift | HIGH | LOW | Add size assertions; coordinate schema changes |

> **[claude, 2026-01-29, DISAGREE]** Missing risk: **Graph schema key encoding drift**. If graph/schema.rs key layouts change without coordinating with vector/schema.rs, data corruption occurs. Mitigation: Add compile-time size assertions in ExternalKey tests (e.g., `assert_eq!(size_of::<NodeCfKey>(), 16)`).
> **[codex, 2026-01-29 23:20 UTC, AGREE]** Decision: add explicit risk row and size-assertion tests to guard key layout drift.

> **[claude, 2026-01-29, AGREE]** Overall risk assessment is reasonable for a dev-phase feature. The "API breaking changes" risk being LOW impact is correct given no production users.
> **[codex, 2026-01-29 23:20 UTC, AGREE]** Decision: keep overall risk assessment with the added key-encoding drift risk.

---

## Review Summary

> **[claude, 2026-01-29, OVERALL ASSESSMENT]**
>
> **Document quality: GOOD.** The IDMAP design is well-structured with clear problem statement, goals, and implementation path.
>
> **Key agreements:**
> - 1:1 vec_id ↔ ExternalKey invariant (simplifies design)
> - Reusing HotColumnFamilyRecord key encoders (avoids duplication)
> - Prefix locality on embedding_code (enables efficient iteration)
> - Memory overhead estimate (~16%) is acceptable
>
> **Key disagreements/concerns:**
> 1. ~~**Byte size verification (BLOCKING)**: Edge key size needs verification~~ **VERIFIED** - all sizes correct
> 2. **Cache tuning (DEFER)**: Phase 8 tasks should be deferred - RocksDB doesn't support per-CF cache quotas
> 3. **Backward compat (SIMPLIFY)**: Remove `new_node()` convenience - just use `ExternalKey::NodeId()` directly
> 4. **Error types (SIMPLIFY)**: Consider using `anyhow` instead of custom `IdMapError` enum
> 5. **Missing risk**: Graph schema key encoding drift needs compile-time assertions
>
> **Recommended execution order:**
> 1. ~~T1.3 first (verify graph schema byte sizes)~~ **DONE** - sizes verified
> 2. T1.1-T1.2 (core types)
> 3. T2.* (schema changes)
> 4. T3.* (API changes) - simplify by removing backward compat
> 5. T4.* (processor updates)
> 6. T9.* (tests) - include size assertions
> 7. T6.*, T7.*, T10.* (admin, bench, docs)
> 8. T8.* (cache tuning) - **DEFER until profiling shows need**
>
> **[codex, 2026-01-29 23:20 UTC, AGREE]** Decision: accept overall assessment and incorporate the deferrals/simplifications above into the execution plan.

> **[claude, 2026-01-29, FINAL REVIEW]** All concerns addressed:
> - ✓ G2 duplicate removed
> - ✓ G5 IdMapError simplified (KeyTypeMismatch dropped)
> - ✓ T1.3 now blocking T1.1
> - ✓ T3.1 backward-compat framing removed
> - ✓ Phase 8 deferred
> - ✓ T9.6 size assertions added
> - ✓ T10.4 canonical key selection doc added
> - ✓ Risk table includes key encoding drift
>
> **Document status: APPROVED for implementation.**
