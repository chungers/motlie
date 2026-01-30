# CONFIG: Persisted vs Runtime Configuration

## Problem

The vector subsystem has configuration drift risk across process boundaries:

1. **Redundant config types**: `hnsw::Config` duplicates fields from `EmbeddingSpec`
2. **Runtime fields with no purpose**: `enabled` and `max_level` are always the same
3. **Scattered derivation logic**: `m_max = 2*m` computed in multiple places

This creates maintenance burden and potential for silent behavior drift.

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

## Current Problem: Redundant `hnsw::Config`

`hnsw::Config` exists as a separate type with these fields:

| Field | Source | Status |
|-------|--------|--------|
| `enabled` | Runtime | ❌ **Useless** - always `true` in production |
| `dim` | `spec.dim` | ❌ **Redundant** - already in EmbeddingSpec |
| `m` | `spec.hnsw_m` | ❌ **Redundant** - already in EmbeddingSpec |
| `m_max` | `2 * m` | ❌ **Derived** - compute inline |
| `m_max_0` | `2 * m` | ❌ **Derived** - compute inline |
| `ef_construction` | `spec.hnsw_ef_construction` | ❌ **Redundant** - already in EmbeddingSpec |
| `m_l` | `1.0 / ln(m)` | ❌ **Derived** - compute inline |
| `max_level` | Runtime | ❌ **Useless** - always `None` (auto) |
| `batch_threshold` | Runtime | ✅ **Valid** - performance knob |

**8 of 9 fields are redundant or useless.** Only `batch_threshold` serves a purpose.

## Analysis of `enabled` and `max_level`

### `enabled` Field

**Current usage:**
- `processor.rs:753,976` - Returns error if false during search
- `ops/delete.rs:144` - Skips HNSW cleanup if false
- Tests set `false` to isolate non-HNSW code paths

**Reality:** Production always has `enabled = true`. The field exists only for tests.

**Problem:** If we "fix drift" by hardcoding `true`, the field serves no purpose.

### `max_level` Field

**Two different concepts with same name:**
- `hnsw::Config.max_level: Option<u8>` = Runtime **cap** on layers
- `GraphMeta::MaxLevel` = Persisted **actual** max layer in graph

**Reality:** Config's `max_level` is always `None` (auto) in all presets and production.

**Problem:** If we "fix drift" by hardcoding `None`, the field serves no purpose.

## Solution: Eliminate `hnsw::Config`

Since `hnsw::Config` is fully derivable from `EmbeddingSpec`, eliminate it entirely.

> (codex, 2026-01-30 05:37 UTC, ACCEPT) This solves the cross-process drift risk by making persisted `EmbeddingSpec` the only source for structure-affecting params; OK as long as any remaining runtime knobs (e.g., batch sizing, caches) are explicitly separated and never fed into structural HNSW construction.
> (codex, 2026-01-30 06:31 UTC, PARTIAL) Implementation keeps `hnsw::Config` and deprecated constructors/exports, so drift is still possible for callers who bypass `Index::from_spec()`; internal Processor/AsyncUpdater path is aligned, but the doc should acknowledge this residual risk.

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

### What Gets Removed

| Component | Action |
|-----------|--------|
| `hnsw::Config` struct | **DELETE** |
| `hnsw::Config::enabled` | **DELETE** - always true |
| `hnsw::Config::max_level` | **DELETE** - always None (auto) |
| `hnsw::Config::dim` | **DELETE** - use `spec.dim` |
| `hnsw::Config::m` | **DELETE** - use `spec.hnsw_m` |
| `hnsw::Config::m_max` | **DELETE** - compute `2 * m` inline |
| `hnsw::Config::m_max_0` | **DELETE** - compute `2 * m` inline |
| `hnsw::Config::ef_construction` | **DELETE** - use `spec.hnsw_ef_construction` |
| `hnsw::Config::m_l` | **DELETE** - compute `1.0 / ln(m)` inline |
| `hnsw::Config::batch_threshold` | **MOVE** - to `Processor.batch_threshold` |
| `Processor.hnsw_config` field | **DELETE** |
| `VectorConfig.hnsw` field | **DELETE** or simplify |

> (codex, 2026-01-30 06:31 UTC, REJECT) Current code did not delete `hnsw::Config` or `VectorConfig.hnsw`; both remain (deprecated). This section is not accurate as written.

### What Gets Added

```rust
// In Processor
pub struct Processor {
    // ... existing fields ...

    /// Batch threshold for HNSW neighbor fetching.
    /// Performance knob - can vary per process without affecting index integrity.
    batch_threshold: usize,
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
}
```

> (codex, 2026-01-30 05:37 UTC, ACCEPT) Ensure `batch_threshold` is exposed via a public constructor or builder so callers can set it per process, and validate that no structural fields can be overridden outside `EmbeddingSpec`.

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

- [x] **T2.4**: Keep `hnsw::Config` struct (deprecated)
  - Config struct kept for backward compatibility
  - `Index::from_spec()` is the preferred constructor
  - Old constructors marked deprecated

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
  > (codex, 2026-01-30 06:31 UTC, PARTIAL) `AsyncUpdaterConfig.ef_construction` is now unused (index builds from spec). Either remove the knob or route it into spec-driven building; otherwise this is an API no-op.

- [x] **T4.2**: Update `VectorConfig`
  - File: `libs/db/src/vector/config.rs`
  - Marked `hnsw` field as deprecated
  - Added deprecation warnings to `dim_*()` presets
  - Keep `rabitq` config as-is

- [x] **T4.3**: Update benchmarks
  - Files: `libs/db/src/vector/benchmark/*.rs`
  - Added `#[allow(deprecated)]` for benchmark code

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

### Scenario 4: No hnsw::Config Usage

1. Grep codebase for `hnsw::Config`
2. Verify: Zero references (struct deleted)
3. Verify: All HNSW code uses `EmbeddingSpec` methods

> (codex, 2026-01-30 06:31 UTC, REJECT) `hnsw::Config` is still referenced across code/tests/docs (grep shows multiple hits); this scenario is not satisfied.

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

**Problem:** `hnsw::Config` is redundant - all fields are derived from `EmbeddingSpec`
or serve no purpose (`enabled`, `max_level`).

**Solution:** Delete `hnsw::Config`. Use `EmbeddingSpec` directly as the source of
truth. Move `batch_threshold` to `Processor` as a simple runtime knob.

**Benefits:**
- Zero configuration drift risk (impossible by design)
- Simpler codebase (one config type, not two)
- Clear separation: `EmbeddingSpec` = persisted truth, `batch_threshold` = runtime knob
- Tests use explicit helper instead of config flag

> (codex, 2026-01-30 06:31 UTC, PARTIAL) Drift risk is eliminated only for Processor/AsyncUpdater paths; deprecated `hnsw::Config` and `Index::new()` still allow drift if used externally.

**Estimated effort:** Medium (1-2 days) - touches many files but changes are mechanical.

> (codex, 2026-01-30 05:37 UTC, ACCEPT) The proposal addresses the stated persistence/correctness concerns; removing `enabled`/`max_level` is acceptable if tests use a helper and no production flows depended on these toggles.
