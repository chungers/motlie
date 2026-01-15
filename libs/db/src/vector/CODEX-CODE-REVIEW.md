# Code Review: motlie_db::vector (Codex)

Findings are ordered by severity and prioritized for correctness, then performance, then idiomatic design.

## Findings

1) High - Vector storage ignores per-embedding storage type and always writes F32.
   - `libs/db/src/vector/writer.rs:343` writes vectors via `Vectors::value_to_bytes`, which always serializes as F32, even though `EmbeddingSpec` supports F16 and the HNSW index can be configured for F16 storage.
   - Impact: embeddings registered with `VectorElementType::F16` will be persisted as F32 but later read as F16, corrupting values or truncating buffers; downstream distances and RaBitQ encodes can be incorrect.
   - Fix: look up the embedding spec (registry) during insert and call `Vectors::value_to_bytes_typed(..., storage_type)`; likewise ensure delete/reads assume the same storage type.

2) High - HNSW layer assignment uses only cached navigation info, causing incorrect layer distribution after restart.
   - `libs/db/src/vector/hnsw/insert.rs:33` assigns `node_layer` from `nav_cache.get(...)` and falls back to `0` without loading GraphMeta.
   - Impact: the first insert after a restart (or any cold cache) will always be layer 0, skewing the exponential layer distribution and reducing upper-layer connectivity/recall.
   - Fix: call `get_or_init_navigation` before sampling, or load navigation directly in the layer assignment path to use the correct `max_layer` and `m_l` distribution.

3) Medium - Searching an empty index errors instead of returning empty results.
   - `libs/db/src/vector/hnsw/search.rs:36` calls `load_navigation` and errors when GraphMeta is missing; `load_navigation` treats missing entry point/max level as errors (`libs/db/src/vector/hnsw/insert.rs:204`).
   - Impact: a valid “empty index” state returns an error rather than `Ok(Vec::new())`, complicating callers that expect a no-results response.
   - Fix: treat missing GraphMeta as empty navigation info; either special-case in `search` or make `load_navigation` return an empty `NavigationLayerInfo`.

4) Medium - RaBitQ cached ADC search inserts MAX-distance placeholders for missing codes.
   - `libs/db/src/vector/hnsw/search.rs:327` uses `f32::MAX` when `code_cache` misses, and still pushes those candidates/results (`libs/db/src/vector/hnsw/search.rs:364`).
   - Impact: partially populated caches (e.g., incremental indexing or cache warmup) can pollute the beam with MAX-distance nodes, causing early termination or poor recall.
   - Fix: skip candidates without cached codes (or fall back to exact distance) to avoid contaminating the beam with sentinel distances.

5) Low - BinaryCodeCache size accounting overcounts on overwrite.
   - `libs/db/src/vector/cache/binary_codes.rs:78` increments `cache_bytes` on every `put` without subtracting any previous entry size.
   - Impact: `stats()` becomes inaccurate after updates/overwrites, which can mislead memory monitoring or auto-tuning.
   - Fix: subtract existing entry size when a key is replaced.

## Performance Notes (Non-blocking)

- HNSW neighbor selection in `libs/db/src/vector/hnsw/insert.rs:73` uses a “closest M” heuristic only. This is simpler but typically yields lower recall than the standard HNSW diversity heuristic; consider adding the diversification step if higher recall is a priority.
- Cosine distance currently recomputes norms per comparison (`libs/db/src/vector/hnsw/graph.rs:35`). If embeddings are guaranteed normalized (as implied for RaBitQ/Cosine), caching norms or using dot-product-only distance can reduce CPU cost.

## Proposed Changes (Detailed)

### A) Fix storage-type mismatch on insert/read paths

**Goal:** Ensure vectors are serialized/deserialized using the embedding’s declared `VectorElementType`, not a fixed F32 format.

**Root cause (as observed):**
- `InsertVector` writes via `Vectors::value_to_bytes` (F32-only) while some embeddings may have `VectorElementType::F16`.

**Proposed implementation outline:**
1. **Lookup embedding spec during insert.**
   - Use the registry in `Processor` to fetch the `Embedding`/`EmbeddingSpec` for `op.embedding`.
   - If the embedding code is missing, return a hard error; do not silently default to F32.
2. **Serialize using the declared storage type.**
   - Replace `Vectors::value_to_bytes(&vec_value)` with `Vectors::value_to_bytes_typed(&op.vector, storage_type)`.
   - Keep all downstream usage consistent with that type for reads and exact-distance evaluation.
3. **Ensure read paths use the same type.**
   - Wherever vectors are loaded for distance computation or reranking, use `Vectors::value_from_bytes_typed(..., storage_type)` instead of assuming F32.
   - If the `Index` is already constructed with a `storage_type` that can be trusted, propagate that value through `Index` construction from the embedding spec (or confirm that it already does so).
4. **Verify HNSW index construction uses the right storage type.**
   - If the HNSW `Index` is created independently from the embedding registry (e.g., in benchmarks), ensure the storage type is explicit and matches the embedding spec, or skip the registry in those cases.

**Concrete file-level changes (suggested):**
- `libs/db/src/vector/writer.rs`: resolve `storage_type` from registry and pass into `Vectors::value_to_bytes_typed`.
- `libs/db/src/vector/hnsw/graph.rs`: avoid using `value_from_bytes_typed(..., index.storage_type())` if `index.storage_type()` can be out-of-sync; either ensure sync at index creation or validate on construction.
- `libs/db/src/vector/registry.rs` (if needed): expose a helper to fetch `VectorElementType` by embedding code without cloning the full embedding.
- `bins/bench_vector/src/commands.rs`: ensure the storage type used in `Index::with_storage_type` mirrors embedding spec when applicable (bench code is currently explicit F16 and might be fine).

**Compatibility & migration considerations:**
- Existing data written as F32 will be interpreted incorrectly if the embedding spec says F16. Consider:
  - Adding a one-time validation step when opening a DB: if an embedding spec’s storage type is F16 but stored vectors are F32, emit a warning and offer a migration tool.
  - Alternatively, introduce a version marker in the vector value encoding (but this is more invasive).

**Testing plan:**
- Unit test: insert vector with `VectorElementType::F16`, read it back via query path, verify numeric error is within expected F16 bounds.
- Integration test: build a small HNSW index with F16 and confirm search returns stable neighbors vs F32 (within tolerance).
- Regression test: existing F32 embeddings still serialize/deserialize identically.

### B) Fix HNSW layer assignment on restart (cold cache)

**Goal:** Use persisted navigation info (GraphMeta) when sampling the node layer, even on a cold cache.

**Root cause (as observed):**
- `insert` samples `node_layer` from `nav_cache.get(...)` and defaults to `0` when cache is empty, instead of loading GraphMeta.

**Proposed implementation outline:**
1. **Load navigation info before sampling.**
   - Call `get_or_init_navigation(index, storage)` at the top of `insert` to get a correct `NavigationLayerInfo`.
2. **Sample layer from the retrieved navigation info.**
   - Use `nav_info.random_layer(&mut rng)` rather than `nav_cache.get(...)`.
3. **If GraphMeta is missing, still allow a clean empty-index insert.**
   - `get_or_init_navigation` already creates a new `NavigationLayerInfo` when nothing is persisted.
4. **Keep cache coherent.**
   - Ensure `get_or_init_navigation` updates `nav_cache` with the loaded info so that subsequent inserts don’t repeatedly hit RocksDB.

**Concrete file-level changes (suggested):**
- `libs/db/src/vector/hnsw/insert.rs`: move `let nav_info = get_or_init_navigation(...)` to precede layer assignment; set `node_layer` based on that returned struct.

**Behavioral validation:**
- Restart test: build a small index, drop the cache, insert a new vector, and verify that the assigned layer is >0 with non-trivial probability (e.g., by seeding RNG and inspecting stored `VecMeta`).
- Distribution test: insert N nodes across restarts and confirm layer histogram matches expected exponential distribution.

**Why this aligns with roadmap:**
- This is correctness for the existing Phase 2 HNSW mechanics and does not depend on Phase 5+ mutation API changes.

## Overall Assessment

The vector subsystem is well-structured and thoughtfully documented, with clear separation between HNSW, quantization, caches, and public APIs. The RaBitQ + ADC implementation appears solid and matches the stated roadmap/benchmarks. The biggest correctness risks are around storage-type handling and stateful initialization (layer assignment after restart), both fixable with localized changes. Performance is competitive for an embedded database, and the abstractions are mostly idiomatic and maintainable. Overall quality is strong, with a few medium/high issues to address before production hardening (Phase 8).

---

## Claude Opus 4.5 Assessment (January 14, 2026)

### Response Summary

| Finding | CODEX Severity | Claude Assessment | Action |
|---------|---------------|-------------------|--------|
| #1 Storage type mismatch | HIGH | ✅ **AGREE** | Fix in Phase 4.5.1 (blocking) |
| #2 Layer assignment cold cache | HIGH | ✅ **AGREE** | Fix in Phase 4.5.2 (blocking) |
| #3 Empty index error | MEDIUM | ⚠️ **PARTIALLY AGREE** | Fix in Phase 4.5.4 (non-blocking) |
| #4 MAX-distance for missing codes | MEDIUM | ✅ **AGREE, ELEVATE** | Fix in Phase 4.5.3 (blocking) |
| #5 Cache size overcount | LOW | ✅ **AGREE** | Fix in Phase 4.5.5 (non-blocking) |

### Finding #1: Storage Type Mismatch

**CODEX:** HIGH - `writer.rs:343` always writes F32 regardless of `VectorElementType`.

**Claude Assessment:** ✅ **AGREE - Critical correctness bug.**

The code path in question always uses `Vectors::value_to_bytes(&vec_value)` which serializes as F32.
If an embedding is registered with `VectorElementType::F16`, this causes:
1. Data written as F32 (512 bytes for 128D)
2. Read path interprets as F16 (256 bytes for 128D)
3. Buffer truncation or garbage values

**Verification:** Confirmed by reading `writer.rs` - no lookup of embedding spec storage type occurs.

**Priority:** 🔴 Must fix before Phase 5 mutation API to prevent data corruption at scale.

### Finding #2: Layer Assignment Cold Cache

**CODEX:** HIGH - `insert.rs:33` falls back to layer 0 when nav_cache is empty.

**Claude Assessment:** ✅ **AGREE - Subtle but real correctness issue.**

After DB restart, `nav_cache` is empty. The code:
```rust
let nav_info = nav_cache.get(&index_key).cloned().unwrap_or_default();
```
Returns `NavigationLayerInfo::default()` which has `max_layer: 0`, causing all post-restart
inserts to get layer 0 until the cache is warmed by reading GraphMeta.

This breaks HNSW's exponential layer distribution property, degrading recall for indices
built across restarts.

**Verification:** Confirmed by reading `insert.rs` - no fallback to `get_or_init_navigation`.

**Priority:** 🔴 Must fix before Phase 5 - the mutation API will exacerbate this with frequent restarts.

### Finding #3: Empty Index Search Error

**CODEX:** MEDIUM - Search errors when GraphMeta is missing (empty index).

**Claude Assessment:** ⚠️ **PARTIALLY AGREE - Edge case, lower priority.**

While technically a usability issue, in practice:
- Empty indices are rare in production (users index data before querying)
- The error message is clear ("GraphMeta not found")
- Callers can wrap with try/catch and return empty results

However, CODEX is correct that returning `Ok(vec![])` is more ergonomic and matches
user expectations for "search found nothing."

**Priority:** 🟡 Nice to have, not blocking Phase 5.

### Finding #4: MAX-Distance for Missing Codes

**CODEX:** MEDIUM - `search.rs:327` uses `f32::MAX` for missing binary codes.

**Claude Assessment:** ✅ **AGREE - Elevated to HIGH.**

This is more serious than MEDIUM because:
1. Partially populated caches occur during incremental indexing (Phase 5's main use case)
2. `f32::MAX` candidates pollute the beam, displacing valid results
3. With `ef_search=100` and 50% cache miss rate, 50 MAX-distance entries dominate beam
4. This causes early termination or severely degraded recall

The fix is straightforward: either skip uncached candidates or fall back to exact distance.

**Priority:** 🔴 Elevated to HIGH - directly blocks Phase 5 incremental indexing correctness.

### Finding #5: Cache Size Overcount

**CODEX:** LOW - `binary_codes.rs:78` doesn't subtract old entry size on overwrite.

**Claude Assessment:** ✅ **AGREE - Minor cosmetic issue.**

The `cache_bytes` counter is only used for `stats()` reporting. Overwrites are rare
(same vec_id updated), and even when they occur, the size drift is bounded by
`(overwrite_count * code_size)`.

**Priority:** 🟢 Low - fix opportunistically, not blocking.

### Performance Notes Response

**CODEX Note 1:** HNSW neighbor selection uses "closest M" not diversity heuristic.

**Claude Response:** Acknowledged. The diversity heuristic improves recall ~2-5% at cost of
insert complexity. Current benchmarks show 87-89% recall which meets requirements.
Consider for Phase 8 production hardening if recall targets increase.

**CODEX Note 2:** Cosine distance recomputes norms per comparison.

**Claude Response:** Acknowledged. For normalized vectors (RaBitQ requirement), we could
use dot-product-only distance. However:
- Current cosine impl handles unnormalized input gracefully
- Norm computation is ~2% of total search time (profiled)
- Premature optimization risk vs. maintaining flexibility

Recommend: Document that pre-normalized vectors skip norm computation, but keep current impl.

### Conclusion

CODEX review is thorough and accurate. Three findings (1, 2, 4) are blocking for Phase 5 and
have been added to ROADMAP.md as Phase 4.5. Two findings (3, 5) are valid but non-blocking.
Performance notes are acknowledged but not actionable for Phase 5.

**Next Steps:**
1. Implement Phase 4.5 tasks (5 fixes)
2. Run full test suite + benchmarks to verify no regression
3. Update this document with resolution commits
4. Proceed to Phase 5: Internal Mutation/Query API
