# CONFIG: Persisted vs Runtime Configuration

## Problem

The vector subsystem currently accepts in-memory configuration (e.g., `hnsw::Config`,
`VectorConfig`) that **is not fully persisted**. On process restart, the registry
is prewarmed from `EmbeddingSpecs`, but runtime HNSW config can diverge if callers
reconstruct it differently. This creates silent behavior drift:

- Inserts after restart may use different HNSW parameters than the index was built with.
- Search behavior can change without any stored signal or validation.

Some drift is prevented by `SpecHash`, but it only covers fields already stored
in `EmbeddingSpecs`.

## What is Persisted Today

Stored in `EmbeddingSpecs` (and protected by `SpecHash` in `GraphMeta`):

- `model`
- `dim`
- `distance`
- `storage_type`
- `hnsw_m`
- `hnsw_ef_construction`
- `rabitq_bits`
- `rabitq_seed`

These fields are stable across process boundaries and already validated at insert/search.

## What is Not Persisted (Problematic)

Runtime HNSW/Vector config fields (currently in `hnsw::Config` or `VectorConfig`)
are not stored, and can drift across processes:

- `enabled` (HNSW on/off)
- `m_max`, `m_max_0` (derived but not stored)
- `m_l` (derived but not stored)
- `max_level` (runtime override)
- `batch_threshold` (runtime optimization knob)

Because these are not persisted, a new process can build a `Processor` using
different values than the process that built the index.

## Required Fix

The runtime config should be **fully derivable** from persisted spec, or else
persisted alongside it.

### Option A (Preferred): Derive from EmbeddingSpec

Define a deterministic conversion from `EmbeddingSpec` → `hnsw::Config`:

- `enabled`: per-embedding, must be persisted if you want to disable indexing
- `dim`: from `EmbeddingSpec.dim`
- `m`: from `EmbeddingSpec.hnsw_m`
- `m_max`: `2 * m`
- `m_max_0`: `2 * m`
- `ef_construction`: from `EmbeddingSpec.hnsw_ef_construction`
- `m_l`: `1.0 / ln(m)`
- `max_level`: `None` (auto)
- `batch_threshold`: fixed default (e.g., 64) or derived from a global config

This ensures all processes compute the same HNSW runtime config from persisted spec.

### Option B: Expand EmbeddingSpec (Persist Additional Fields)

If per-embedding overrides are required, expand `EmbeddingSpec` with:

- `hnsw_enabled: bool`
- `hnsw_m_max: u16` (optional override)
- `hnsw_m_max_0: u16` (optional override)
- `hnsw_max_level: Option<u8>`

These must be:
1) persisted in `EmbeddingSpecs` CF,
2) included in `SpecHash`,
3) validated in `Processor::search_with_config` and insert path.

## Required Code Changes (Option A)

1) **Centralize config derivation**
   - Add a helper:
     ```rust
     impl EmbeddingSpec {
         pub fn to_hnsw_config(&self) -> hnsw::Config { ... }
     }
     ```
2) **Use spec-derived config in all index creation**
   - `Processor::get_or_create_index`
   - `AsyncUpdater` HNSW index creation
   - Any benchmark/index builders

3) **Persist or derive structural HNSW fields**
   - Structural fields that must be stable across restarts:
     - `enabled`, `m`, `m_max`, `m_max_0`, `ef_construction`, `max_level`
   - Derived field (do not persist): `m_l`
   - Performance-only (do not persist): `batch_threshold`

4) **Remove or de-emphasize ad-hoc `VectorConfig` for per-embedding HNSW**
   - If `VectorConfig` remains, it should only wrap **global** knobs (e.g. batch_threshold)
     or be applied consistently on startup and stored in `GraphMeta`.

## Required Code Changes (Option B)

1) Expand `EmbeddingSpec` and builder (`EmbeddingBuilder`) to accept new fields.
2) Include new fields in spec hashing.
3) Ensure registry prewarm loads these fields.
4) Replace any in-memory HNSW config creation with `spec.to_hnsw_config()`.

## Recommendation

**Option A** is strongly preferred. It minimizes persisted surface area and ensures
deterministic config reconstruction from persisted spec fields. Only structural
HNSW parameters should be persisted; performance knobs like `batch_threshold`
remain global per process.

## Design: User-Mutable Runtime Knobs (Batch Threshold)

To allow user-tunable performance knobs **without** breaking structural
consistency across restarts, construct HNSW config from persisted spec and then
apply a small, explicit runtime override.

### Proposed Builder Pattern

```rust
pub struct HnswConfigBuilder {
    base: hnsw::Config,   // derived from EmbeddingSpec
}

impl HnswConfigBuilder {
    pub fn from_spec(spec: &EmbeddingSpec) -> Self {
        Self { base: spec.to_hnsw_config() }
    }

    // Runtime-only knobs (safe to vary per process)
    pub fn with_batch_threshold(mut self, threshold: usize) -> Self {
        self.base.batch_threshold = threshold;
        self
    }

    pub fn build(self) -> hnsw::Config {
        self.base
    }
}
```

### Usage

```rust
let cfg = HnswConfigBuilder::from_spec(&spec)
    .with_batch_threshold(8)
    .build();
```

### Guarantees

- **Structural fields** always come from `EmbeddingSpec` and remain stable.
- **Performance fields** are explicitly overridden per process.
- No hidden drift across restarts.

If desired, the override can be plumbed through `VectorConfig` as a **global**
runtime setting applied uniformly at startup.

## Structural vs Runtime Types (Recommended)

To make drift impossible, separate **persisted structural config** from
**process-local runtime config** with explicit types and a deterministic merge.

### Structural Types (Derived from EmbeddingSpec)

```rust
pub struct StructuralHnswConfig {
    pub enabled: bool,
    pub dim: usize,
    pub m: usize,
    pub m_max: usize,
    pub m_max_0: usize,
    pub ef_construction: usize,
    pub max_level: Option<u8>,
}

impl EmbeddingSpec {
    pub fn structural_hnsw(&self) -> StructuralHnswConfig {
        let m = self.hnsw_m as usize;
        StructuralHnswConfig {
            enabled: self.hnsw_enabled,
            dim: self.dim as usize,
            m,
            m_max: (self.hnsw_m_max.unwrap_or((2 * self.hnsw_m) as u16)) as usize,
            m_max_0: (self.hnsw_m_max_0.unwrap_or((2 * self.hnsw_m) as u16)) as usize,
            ef_construction: self.hnsw_ef_construction as usize,
            max_level: self.hnsw_max_level,
        }
    }
}
```

### Runtime Types (Process-Local)

```rust
pub struct HnswRuntimeConfig {
    pub batch_threshold: usize,
}

pub struct RabitqRuntimeConfig {
    pub use_cache: bool,
}
```

### Merge (Deterministic Composition)

```rust
impl StructuralHnswConfig {
    pub fn to_hnsw_config(self, runtime: HnswRuntimeConfig) -> hnsw::Config {
        let m_l = 1.0 / (self.m as f32).ln();
        hnsw::Config {
            enabled: self.enabled,
            dim: self.dim,
            m: self.m,
            m_max: self.m_max,
            m_max_0: self.m_max_0,
            ef_construction: self.ef_construction,
            m_l,
            max_level: self.max_level,
            batch_threshold: runtime.batch_threshold,
        }
    }
}
```

### Notes

- Structural fields are **persisted** (via `EmbeddingSpec`) and protected by `SpecHash`.
- Runtime fields are **not persisted** and are explicitly applied per process.
- The merge is deterministic and keeps index integrity stable across restarts.
