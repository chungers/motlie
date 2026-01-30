# CONFIG: Persisted vs Runtime Configuration

## Status: COMPLETE ✅

This design has been fully implemented. `hnsw::Config` has been **deleted** (not just
deprecated) and `EmbeddingSpec` is now the single source of truth for all HNSW parameters.

## Original Problem (Solved)

The vector subsystem had configuration drift risk across process boundaries:

1. **Redundant config types**: `hnsw::Config` duplicated fields from `EmbeddingSpec`
2. **Runtime fields with no purpose**: `enabled` and `max_level` were always the same
3. **Scattered derivation logic**: `m_max = 2*m` computed in multiple places

This created maintenance burden and potential for silent behavior drift.

## What is Persisted (Source of Truth)

`EmbeddingSpec` in `EmbeddingSpecs` CF (protected by `SpecHash`):

| Field | Purpose | Used By |
|-------|---------|---------|
| `model` | Embedding model name | Identification |
| `dim` | Vector dimensionality | HNSW, RaBitQ, storage |
| `distance` | Distance metric | HNSW search |
| `storage_type` | F16/F32 | Vector serialization |
| `hnsw_m` | HNSW M parameter | Graph connectivity |
| `hnsw_ef_construction` | Build beam width | Graph quality |
| `rabitq_bits` | Quantization bits | Compression |
| `rabitq_seed` | Rotation seed | Deterministic encoding |

**This is the single source of truth.** All HNSW parameters should derive from here.

## Solved Problem: Redundant `hnsw::Config` (DELETED)

`hnsw::Config` existed as a separate type with these fields:

| Field | Source | Resolution |
|-------|--------|------------|
| `enabled` | Runtime | ✅ **DELETED** - was always `true` in production |
| `dim` | `spec.dim` | ✅ **DELETED** - derived from EmbeddingSpec |
| `m` | `spec.hnsw_m` | ✅ **DELETED** - derived from EmbeddingSpec |
| `m_max` | `2 * m` | ✅ **DELETED** - now inlined in `Index` |
| `m_max_0` | `2 * m` | ✅ **DELETED** - now inlined in `Index` |
| `ef_construction` | `spec.hnsw_ef_construction` | ✅ **DELETED** - derived from EmbeddingSpec |
| `m_l` | `1.0 / ln(m)` | ✅ **DELETED** - now inlined in `Index` |
| `max_level` | Runtime | ✅ **DELETED** - was always `None` (auto) |
| `batch_threshold` | Runtime | ✅ **MOVED** - to `Index.batch_threshold` |

**All redundant fields deleted.** `batch_threshold` moved to `Index` as a runtime knob.

## Analysis of `enabled` and `max_level` (Historical)

### `enabled` Field (DELETED)

**Former usage:**
- `processor.rs` - Returned error if false during search
- `ops/delete.rs` - Skipped HNSW cleanup if false
- Tests set `false` to isolate non-HNSW code paths

**Resolution:** Field deleted. Tests now use `#[cfg(test)] skip_hnsw_for_testing` flag.

### `max_level` Field (DELETED)

**Two different concepts existed with same name:**
- `hnsw::Config.max_level: Option<u8>` = Runtime **cap** on layers
- `GraphMeta::MaxLevel` = Persisted **actual** max layer in graph

**Resolution:** Config field deleted. Only `GraphMeta::MaxLevel` remains (correct behavior).

## Solution: Eliminate `hnsw::Config` ✅ COMPLETE

Since `hnsw::Config` was fully derivable from `EmbeddingSpec`, it has been **deleted entirely**.

> (codex, 2026-01-30 05:37 UTC, ACCEPT) This solves the cross-process drift risk by making persisted `EmbeddingSpec` the only source for structure-affecting params; OK as long as any remaining runtime knobs (e.g., batch sizing, caches) are explicitly separated and never fed into structural HNSW construction.
> (codex, 2026-01-30 06:31 UTC, PARTIAL) Implementation keeps `hnsw::Config` and deprecated constructors/exports, so drift is still possible for callers who bypass `Index::from_spec()`; internal Processor/AsyncUpdater path is aligned, but the doc should acknowledge this residual risk.
> (claude, 2026-01-30 07:15 UTC, RESOLVED) **`hnsw::Config` struct has been fully deleted.** `Index::from_spec()` is now the only constructor. Zero residual drift risk - there is no way to bypass `EmbeddingSpec` because `hnsw::Config` no longer exists.
> (codex, 2026-01-30 15:40 UTC, ACCEPT) Verified in code: `hnsw::Config` and legacy constructors are removed, and all index construction paths derive params from `EmbeddingSpec`. Drift risk eliminated for all call paths.

### Design Principles

1. **`EmbeddingSpec` is the single source of truth** - no redundant config types
2. **Derive values where needed** - compute `m_max`, `m_l` inline
3. **Remove useless fields** - `enabled` and `max_level` serve no purpose
4. **Keep runtime knobs at `Processor` level** - `batch_threshold` is process config

### Target Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        EmbeddingSpec                            │
│  (persisted, protected by SpecHash)                             │
│                                                                 │
│  model, dim, distance, storage_type,                            │
│  hnsw_m, hnsw_ef_construction, rabitq_bits, rabitq_seed         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Processor                               │
│  (runtime, not persisted)                                       │
│                                                                 │
│  batch_threshold: usize  // Only runtime knob                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      HNSW Algorithm                             │
│  (uses EmbeddingSpec directly, derives m_max/m_l inline)        │
│                                                                 │
│  fn insert(spec: &EmbeddingSpec, ...) {                         │
│      let m = spec.hnsw_m as usize;                              │
│      let m_max = 2 * m;                                         │
│      let m_l = 1.0 / (m as f32).ln();                           │
│      // ... use values directly                                 │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
```

### What Was Deleted ✅

All redundant types and APIs have been **fully deleted** (no backward compatibility).

| Component | Status |
|-----------|--------|
| `hnsw::Config` struct | ✅ **DELETED** |
| `hnsw::Config::enabled` | ✅ **DELETED** |
| `hnsw::Config::max_level` | ✅ **DELETED** |
| `hnsw::Config::dim` | ✅ **DELETED** |
| `hnsw::Config::m` | ✅ **DELETED** |
| `hnsw::Config::m_max` | ✅ **DELETED** |
| `hnsw::Config::m_max_0` | ✅ **DELETED** |
| `hnsw::Config::ef_construction` | ✅ **DELETED** |
| `hnsw::Config::m_l` | ✅ **DELETED** |
| `hnsw::Config::batch_threshold` | ✅ **MOVED** - to `Index.batch_threshold` |
| `Index::new()` | ✅ **DELETED** |
| `Index::with_storage_type()` | ✅ **DELETED** |
| `Processor::with_config()` | ✅ **DELETED** |
| `Processor::with_config_and_nav_cache()` | ✅ **DELETED** |
| `VectorConfig.hnsw` field | ✅ **DELETED** |
| `VectorConfig::dim_*()` presets | ✅ **DELETED** |
| `AsyncUpdaterConfig.ef_construction` | ✅ **DELETED** |

> (codex, 2026-01-30 06:31 UTC, REJECT) Current code did not delete `hnsw::Config` or `VectorConfig.hnsw`; both remain (deprecated). This section is not accurate as written.
> (claude, 2026-01-30 07:15 UTC, RESOLVED) **All items in this table have been fully deleted.** Backward compatibility was explicitly not required per user instruction. Grep for `hnsw::Config` now returns zero code references.
> (codex, 2026-01-30 15:40 UTC, ACCEPT) Confirmed: `hnsw::Config`, `Index::new()`, `VectorConfig.hnsw`, and `AsyncUpdaterConfig.ef_construction` are removed in code.

### What Was Added ✅

```rust
// In Processor
pub struct Processor {
    // ... existing fields ...

    /// Batch threshold for HNSW neighbor fetching.
    /// Performance knob - can vary per process without affecting index integrity.
    batch_threshold: usize,

    /// Test-only flag to skip HNSW operations.
    #[cfg(test)]
    skip_hnsw_for_testing: bool,
}

impl Processor {
    pub fn new(storage: Arc<Storage>, registry: Arc<EmbeddingRegistry>) -> Self {
        Self::with_batch_threshold(storage, registry, 64) // Default
    }

    pub fn with_batch_threshold(
        storage: Arc<Storage>,
        registry: Arc<EmbeddingRegistry>,
        batch_threshold: usize,
    ) -> Self {
        // ...
    }

    pub fn with_rabitq_config_and_batch_threshold(
        storage: Arc<Storage>,
        registry: Arc<EmbeddingRegistry>,
        rabitq_config: RaBitQConfig,
        batch_threshold: usize,
    ) -> Self {
        // ...
    }
}
```

```rust
// In hnsw::Index - the only constructor
impl Index {
    pub fn from_spec(
        embedding: EmbeddingCode,
        spec: &EmbeddingSpec,
        batch_threshold: usize,
        nav_cache: Arc<NavigationCache>,
    ) -> Self {
        // All HNSW params derived from spec
        let m = spec.m();
        let m_max = spec.m_max();
        let m_max_0 = spec.m_max_0();
        let m_l = spec.m_l();
        let ef_construction = spec.ef_construction();
        // ...
    }
}
```

> (codex, 2026-01-30 05:37 UTC, ACCEPT) Ensure `batch_threshold` is exposed via a public constructor or builder so callers can set it per process, and validate that no structural fields can be overridden outside `EmbeddingSpec`.
> (claude, 2026-01-30 07:15 UTC, VERIFIED) ✅ `batch_threshold` exposed via `Processor::with_batch_threshold()`. Structural fields cannot be overridden - `Index::from_spec()` is the only constructor and it derives all structural params from `EmbeddingSpec`.
> (codex, 2026-01-30 15:40 UTC, ACCEPT) `batch_threshold` is wired end-to-end (Processor → Index → search). No remaining overrides for structural params.

### Helper Methods on EmbeddingSpec

Add convenience methods for derived values:

```rust
impl EmbeddingSpec {
    /// HNSW M parameter (number of bidirectional links per node)
    pub fn m(&self) -> usize {
        self.hnsw_m as usize
    }

    /// Maximum links per node at layers > 0 (always 2*M)
    pub fn m_max(&self) -> usize {
        2 * self.m()
    }

    /// Maximum links at layer 0 (always 2*M)
    pub fn m_max_0(&self) -> usize {
        2 * self.m()
    }

    /// Layer probability multiplier (always 1/ln(M))
    pub fn m_l(&self) -> f32 {
        1.0 / (self.m() as f32).ln()
    }

    /// ef_construction parameter
    pub fn ef_construction(&self) -> usize {
        self.hnsw_ef_construction as usize
    }
}
```

### Test-Only HNSW Disable

For tests that need to skip HNSW, add a test helper instead of a config field:

```rust
#[cfg(test)]
impl Processor {
    /// Create processor that skips HNSW operations (test-only).
    ///
    /// Use this to test non-HNSW code paths in isolation.
    pub fn new_without_hnsw_for_testing(
        storage: Arc<Storage>,
        registry: Arc<EmbeddingRegistry>,
    ) -> Self {
        Self {
            skip_hnsw_for_testing: true,
            ..Self::new(storage, registry)
        }
    }
}
```

## Impact on Existing Code

### Files to Modify

| File | Change |
|------|--------|
| `hnsw/config.rs` | Delete `Config` struct, keep `ConfigWarning` for validation |
| `hnsw/mod.rs` | Remove `Config` from public exports |
| `hnsw/index.rs` | Change constructor to take `&EmbeddingSpec` + `batch_threshold` |
| `hnsw/insert.rs` | Use `spec.m()`, `spec.m_max()`, etc. directly |
| `hnsw/search.rs` | Use `spec.ef_construction()` directly |
| `config.rs` | Remove `hnsw` field from `VectorConfig` |
| `processor.rs` | Replace `hnsw_config: hnsw::Config` with `batch_threshold: usize` |
| `processor.rs` | Update `get_or_create_index()` to pass spec directly |
| `async_updater.rs` | Update HNSW index creation |
| `benchmark/*.rs` | Update to new API |

### Migration Path

1. Add helper methods to `EmbeddingSpec` (`m()`, `m_max()`, etc.)
2. Update HNSW code to use `EmbeddingSpec` directly
3. Update `Processor` to hold `batch_threshold` instead of `hnsw_config`
4. Delete `hnsw::Config` struct
5. Update tests to use `new_without_hnsw_for_testing()` helper

---

## Implementation Tasks

### Phase 1: Add EmbeddingSpec Helper Methods ✅

- [x] **T1.1**: Add derived value methods to `EmbeddingSpec`
  - File: `libs/db/src/vector/schema.rs`
  - Added: `m()`, `m_max()`, `m_max_0()`, `m_l()`, `ef_construction()`
  - These replace inline derivation scattered across codebase

- [x] **T1.2**: Add unit tests for derived methods
  - Verified `m_max() == 2 * m()`
  - Verified `m_l() == 1.0 / ln(m())`

### Phase 2: Update HNSW Module ✅

- [x] **T2.1**: Update `hnsw::Index` to take `&EmbeddingSpec` + `batch_threshold`
  - File: `libs/db/src/vector/hnsw/mod.rs`
  - Added: `Index::from_spec(embedding, spec, batch_threshold, nav_cache)`
  - Uses `spec.m()`, `spec.m_max()`, etc. internally
  - Old constructors marked deprecated

- [x] **T2.2**: Update `hnsw::insert` to use spec methods
  - HNSW insert uses Index which derives params from EmbeddingSpec

- [x] **T2.3**: Update `hnsw::search` to use spec methods
  - HNSW search uses Index which derives params from EmbeddingSpec

- [x] **T2.4**: Delete `hnsw::Config` struct entirely
  - Config struct **fully deleted** (not just deprecated)
  - `Index::from_spec()` is now the **only** constructor
  - Old constructors `Index::new()` and `Index::with_storage_type()` deleted

### Phase 3: Update Processor ✅

- [x] **T3.1**: Replace `hnsw_config` field with `batch_threshold`
  - File: `libs/db/src/vector/processor.rs`
  - Changed: `hnsw_config: hnsw::Config` → `batch_threshold: usize`
  - Added: `with_batch_threshold()` constructor
  - Old `with_config()` marked deprecated

- [x] **T3.2**: Update `get_or_create_index()` to pass spec
  - Uses `Index::from_spec()` with EmbeddingSpec from storage
  - Passes `self.batch_threshold` as runtime knob

- [x] **T3.3**: Remove HNSW enabled checks
  - Changed: `if !self.hnsw_config.enabled` → `#[cfg(test)] if self.skip_hnsw_for_testing`
  - HNSW is always enabled in production

- [x] **T3.4**: Add test helper for disabled-HNSW scenarios
  - Added: `#[cfg(test)] fn new_without_hnsw_for_testing()`
  - Updated tests that set `enabled = false`

### Phase 4: Update Related Code ✅

- [x] **T4.1**: Update `AsyncUpdater`
  - File: `libs/db/src/vector/async_updater.rs`
  - Uses `Index::from_spec()` for HNSW index creation
  - Reads EmbeddingSpec from storage (single source of truth)
  - **DELETED**: `AsyncUpdaterConfig.ef_construction` field removed entirely
  > (codex, 2026-01-30 06:31 UTC, PARTIAL) `AsyncUpdaterConfig.ef_construction` is now unused (index builds from spec). Either remove the knob or route it into spec-driven building; otherwise this is an API no-op.
  > (claude, 2026-01-30 07:15 UTC, RESOLVED) ✅ `ef_construction` field has been deleted from `AsyncUpdaterConfig`.

- [x] **T4.2**: Update `VectorConfig`
  - File: `libs/db/src/vector/config.rs`
  - **DELETED**: `hnsw` field removed entirely
  - **DELETED**: All `dim_*()` preset methods removed
  - Keep `rabitq` config as-is (only remaining field)

- [x] **T4.3**: Update benchmarks
  - Files: `libs/db/src/vector/benchmark/*.rs`
  - Updated `build_hnsw_index()` to take individual params instead of `hnsw::Config`
  - Creates `EmbeddingSpec` internally and uses `Index::from_spec()`

- [x] **T4.4**: Update `ops/delete.rs`
  - HNSW cleanup always happens (soft-delete)
  - Removed HNSW enabled check

### Phase 5: Documentation ✅

- [x] **T5.1**: Update this document
  - Marked all tasks complete
  - Added implementation notes

- [ ] **T5.2**: Update API.md (if exists)
  - Document new `Processor` constructor signatures
  - Document deprecation of `hnsw::Config`

---

## Verification Checklist

After implementation, verify:

### Scenario 1: Process Restart

1. Start Process A, insert 1000 vectors
2. Kill Process A
3. Start Process B (same binary)
4. Verify: Search returns same results
5. Verify: New inserts work correctly

### Scenario 2: Different batch_threshold

1. Process A with `batch_threshold = 64`, insert vectors
2. Restart as Process B with `batch_threshold = 8`
3. Verify: Search results identical (structural params from EmbeddingSpec)
4. Verify: Only performance differs (expected)

### Scenario 3: SpecHash Validation

1. Insert vectors with `hnsw_m = 16`
2. Corrupt DB: change `EmbeddingSpec.hnsw_m` to `32`
3. Attempt insert
4. Verify: Fails with SpecHash mismatch

### Scenario 4: No hnsw::Config Usage ✅ VERIFIED

1. Grep codebase for `hnsw::Config`
2. Verify: Zero references (struct deleted)
3. Verify: All HNSW code uses `EmbeddingSpec` methods

> (codex, 2026-01-30 06:31 UTC, REJECT) `hnsw::Config` is still referenced across code/tests/docs (grep shows multiple hits); this scenario is not satisfied.
> (claude, 2026-01-30 07:15 UTC, RESOLVED) ✅ **Scenario 4 now passes.** `hnsw::Config` struct has been fully deleted. `grep -r "hnsw::Config" --include="*.rs"` returns zero hits in code. Only documentation references remain (this file, describing history).

---

## Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| API breakage | MEDIUM | HIGH | Update all callers; breaking change is intentional |
| Test breakage | LOW | HIGH | Add `new_without_hnsw_for_testing()` helper |
| HNSW module changes | MEDIUM | MEDIUM | Careful refactoring; maintain behavior |
| Benchmark updates | LOW | HIGH | Straightforward signature changes |

---

## Summary

**Problem:** `hnsw::Config` was redundant - all fields were derived from `EmbeddingSpec`
or served no purpose (`enabled`, `max_level`).

**Solution:** Deleted `hnsw::Config` entirely. `EmbeddingSpec` is now the single source of
truth. `batch_threshold` moved to `Index` as the only runtime knob.

**Benefits:**
- Zero configuration drift risk (impossible by design - `hnsw::Config` no longer exists)
- Simpler codebase (one config type, not two)
- Clear separation: `EmbeddingSpec` = persisted truth, `batch_threshold` = runtime knob
- Tests use `#[cfg(test)] skip_hnsw_for_testing` instead of config flag

> (codex, 2026-01-30 06:31 UTC, PARTIAL) Drift risk is eliminated only for Processor/AsyncUpdater paths; deprecated `hnsw::Config` and `Index::new()` still allow drift if used externally.
> (claude, 2026-01-30 07:15 UTC, RESOLVED) ✅ **Full drift risk eliminated.** `hnsw::Config` struct and `Index::new()` have been **deleted**. There is no external bypass possible.

**Actual effort:** ~2 days - touched ~15 files, all changes mechanical.

> (codex, 2026-01-30 05:37 UTC, ACCEPT) The proposal addresses the stated persistence/correctness concerns; removing `enabled`/`max_level` is acceptable if tests use a helper and no production flows depended on these toggles.
> (claude, 2026-01-30 07:15 UTC, VERIFIED) ✅ Implementation complete. Tests use `skip_hnsw_for_testing` flag. No production flows depended on `enabled`/`max_level` toggles.
