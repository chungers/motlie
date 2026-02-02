# Vector Module

Vector storage subsystem for HNSW-based approximate nearest neighbor search with RaBitQ quantization.

## Overview

The vector module provides:
- **Embedding Registry**: Runtime registry for embedding spaces (model, dim, distance metric)
- **HNSW Graph Storage**: Layered graph structure for approximate nearest neighbor search
- **RaBitQ Quantization**: Binary codes for fast approximate distance computation
- **ID Mapping**: Bidirectional mapping between external ULIDs and internal u32 vec_ids

## Design Docs

All vector design docs live in `docs/` and are kept close to the implementation for easier review.

- [docs/ROADMAP.md](docs/ROADMAP.md) - Primary implementation roadmap and phase history (kept up to date as phases close).
- [docs/PHASE5.md](docs/PHASE5.md) - Phase 5 execution notes and certification trail.
- [docs/CONCURRENT.md](docs/CONCURRENT.md) - Concurrency workstream details (tests, metrics, stress plan).
- [docs/BASELINE.md](docs/BASELINE.md) - Baseline benchmark protocol and recorded results.
- [docs/BENCHMARK.md](docs/BENCHMARK.md) - Large-scale benchmark results and tuning guidance.
- [docs/BENCHMARK2.md](docs/BENCHMARK2.md) - Secondary benchmark runs and sweeps (historical comparisons).
- [docs/API.md](docs/API.md) - Public API reference and usage patterns.
- [docs/RABITQ.md](docs/RABITQ.md) - RaBitQ design rationale and ADC details.
- [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) - Design constraints and invariants (DATA-1, etc).
- [docs/CODEX-CODE-REVIEW.md](docs/CODEX-CODE-REVIEW.md) - CODEX review findings and fixes (audit trail).
- [docs/GEMINI-CODE-REVIEW.md](docs/GEMINI-CODE-REVIEW.md) - Gemini review notes and follow-ups.
- [docs/GEMINI-BENCHMARK.md](docs/GEMINI-BENCHMARK.md) - Benchmark tooling proposal and status.
- [docs/GEMINI-REVIEW.md](docs/GEMINI-REVIEW.md) - Misc. Gemini design review notes.

## Schema

All column families use the `vector/` prefix. Keys use direct byte serialization for RocksDB prefix extraction; values use MessagePack where appropriate.

### Common Types

```rust,ignore
pub(crate) type EmbeddingCode = u64;         // FK to EmbeddingSpecs (primary key)
pub(crate) type VecId = u32;                 // Internal vector index
pub(crate) type HnswLayer = u8;              // HNSW layer index (0 = base)
pub(crate) type RoaringBitmapBytes = Vec<u8>; // Serialized RoaringBitmap
pub(crate) type RabitqCode = Vec<u8>;        // RaBitQ quantized binary code
// TimestampMilli from crate root
```

### Column Families

| CF | CfKey | CfValue | Description |
|---|---|---|---|
| `EmbeddingSpecs` | `(EmbeddingCode)` | `EmbeddingSpec` | Embedding space definitions |
| `Vectors` | `(EmbeddingCode, VecId)` | `Vec<f32>` | Raw vector data |
| `Edges` | `(EmbeddingCode, VecId, HnswLayer)` | `RoaringBitmapBytes` | HNSW graph edges |
| `BinaryCodes` | `(EmbeddingCode, VecId)` | `RabitqCode` | RaBitQ quantized codes |
| `VecMeta` | `(EmbeddingCode, VecId)` | `VecMetadata` | Vector metadata (level, created_at) |
| `GraphMeta` | `(EmbeddingCode, GraphMetaField)` | `GraphMetaValue` | Graph metadata (entry point, max level, count, spec hash) |
| `IdForward` | `(EmbeddingCode, Id)` | `VecId` | ULID → vec_id mapping |
| `IdReverse` | `(EmbeddingCode, VecId)` | `Id` | vec_id → ULID mapping |
| `IdAlloc` | `(EmbeddingCode, IdAllocField)` | `IdAllocValue` | ID allocator state (next_id, free bitmap) |
| `Pending` | `(EmbeddingCode, TimestampMilli, VecId)` | `()` | Async updater pending queue |

### Naming Convention

For a domain entity `Foo`:

1. **Column family marker**: `Foos` (plural) - unit struct
2. **Key type**: `FooCfKey` (singular + CfKey) - tuple struct
3. **Value type**: `FooCfValue` (singular + CfValue) - wraps domain type
4. **Domain struct**: `Foo` - if value has >2 fields

## Usage

```rust,ignore
use motlie_db::vector::{Storage, EmbeddingBuilder, Distance};

// Create storage
let mut storage = Storage::readwrite(&path);
storage.ready()?;

// Register embedding space
let embedding = storage.cache().register(
    EmbeddingBuilder::new("gemma", 768, Distance::Cosine),
    storage.db()?,
)?;

// Use embedding.code() as EmbeddingCode for all operations
```

---

## Appendix A: Union Pattern for Polymorphic Key/Values

Some column families store multiple field types under a single embedding space, where each field has a different value type. Rather than using magic `u8` discriminant constants, we use a **union enum pattern** that provides type safety and eliminates synchronization issues.

### Problem

Traditional approach with magic constants:

```rust,ignore
// Magic constants that must stay in sync
pub mod graph_meta_field {
    pub const ENTRY_POINT: u8 = 0;
    pub const MAX_LEVEL: u8 = 1;
}

// Key uses raw u8
pub struct GraphMetaCfKey(EmbeddingCode, u8);

// Value is a separate enum
pub enum GraphMetaCfValue {
    EntryPoint(VecId),
    MaxLevel(HnswLayer),
}

// Deserialization requires passing discriminant separately
fn value_from_bytes(field: u8, bytes: &[u8]) -> Result<GraphMetaCfValue>
```

Issues:
- Magic constants can get out of sync with enum variants
- Key construction requires knowing the right constant
- Discriminant passed separately to `value_from_bytes`

### Solution: Union Enum Pattern

Single enum type used for both key discrimination and value storage:

```rust,ignore
/// Single enum for both key discrimination and value storage.
///
/// For keys: variant determines which field, inner value is ignored.
/// For values: variant determines type, inner value is the actual data.
#[derive(Debug, Clone)]
pub(crate) enum GraphMetaField {
    EntryPoint(VecId),
    MaxLevel(HnswLayer),
    Count(u32),
    SpecHash(u64),
}

impl GraphMetaField {
    /// Discriminant derived from match - no magic constants
    fn discriminant(&self) -> u8 {
        match self {
            Self::EntryPoint(_) => 0,
            Self::MaxLevel(_) => 1,
            Self::Count(_) => 2,
            Self::SpecHash(_) => 3,
        }
    }
}

/// Key uses the enum (inner value ignored for serialization)
pub(crate) struct GraphMetaCfKey(pub(crate) EmbeddingCode, pub(crate) GraphMetaField);

/// Type alias distinguishes "value" semantics from "key discriminant" semantics
pub(crate) type GraphMetaValue = GraphMetaField;

/// Value wraps the enum with actual data
pub(crate) struct GraphMetaCfValue(pub(crate) GraphMetaValue);
```

### Key Construction Helpers

Helper methods hide placeholder values used in keys:

```rust,ignore
impl GraphMetaCfKey {
    pub fn entry_point(embedding_code: EmbeddingCode) -> Self {
        Self(embedding_code, GraphMetaField::EntryPoint(0)) // placeholder
    }

    pub fn max_level(embedding_code: EmbeddingCode) -> Self {
        Self(embedding_code, GraphMetaField::MaxLevel(0)) // placeholder
    }

    pub fn count(embedding_code: EmbeddingCode) -> Self {
        Self(embedding_code, GraphMetaField::Count(0)) // placeholder
    }

    pub fn spec_hash(embedding_code: EmbeddingCode) -> Self {
        Self(embedding_code, GraphMetaField::SpecHash(0)) // placeholder
    }
}
```

### Serialization

Key serialization extracts only the discriminant:

```rust,ignore
pub fn key_to_bytes(key: &GraphMetaCfKey) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(9);
    bytes.extend_from_slice(&key.0.to_be_bytes());
    bytes.push(key.1.discriminant()); // Only discriminant, not inner value
    bytes
}

pub fn key_from_bytes(bytes: &[u8]) -> Result<GraphMetaCfKey> {
    let embedding_code = u64::from_be_bytes(bytes[0..8].try_into()?);
    let discriminant = bytes[8];
    // Create field with placeholder value based on discriminant
    let field = match discriminant {
        0 => GraphMetaField::EntryPoint(0),
        1 => GraphMetaField::MaxLevel(0),
        2 => GraphMetaField::Count(0),
        3 => GraphMetaField::SpecHash(0),
        _ => bail!("Unknown discriminant: {}", discriminant),
    };
    Ok(GraphMetaCfKey(embedding_code, field))
}
```

Value deserialization uses the key's variant for type info:

```rust,ignore
pub fn value_from_bytes(key: &GraphMetaCfKey, bytes: &[u8]) -> Result<GraphMetaCfValue> {
    let field = match &key.1 {
        GraphMetaField::EntryPoint(_) => {
            GraphMetaField::EntryPoint(u32::from_be_bytes(bytes.try_into()?))
        }
        GraphMetaField::MaxLevel(_) => {
            GraphMetaField::MaxLevel(bytes[0])
        }
        // ... other variants
    };
    Ok(GraphMetaCfValue(field))
}
```

### Usage Examples

**Writing a value:**

```rust,ignore
let key = GraphMetaCfKey::entry_point(embedding_code);
let value = GraphMetaCfValue(GraphMetaField::EntryPoint(42)); // actual vec_id

let key_bytes = GraphMeta::key_to_bytes(&key);
let value_bytes = GraphMeta::value_to_bytes(&value);
db.put_cf(&cf, key_bytes, value_bytes)?;
```

**Reading a value:**

```rust,ignore
let key = GraphMetaCfKey::entry_point(embedding_code);
let key_bytes = GraphMeta::key_to_bytes(&key);

if let Some(value_bytes) = db.get_cf(&cf, &key_bytes)? {
    let value = GraphMeta::value_from_bytes(&key, &value_bytes)?;
    match value.0 {
        GraphMetaField::EntryPoint(vec_id) => println!("entry_point: {}", vec_id),
        _ => unreachable!(), // key determines variant
    }
}
```

**Iterating all fields for an embedding:**

```rust,ignore
let prefix = embedding_code.to_be_bytes();
for (key_bytes, value_bytes) in db.prefix_iterator_cf(&cf, &prefix) {
    let key = GraphMeta::key_from_bytes(&key_bytes)?;
    let value = GraphMeta::value_from_bytes(&key, &value_bytes)?;
    match value.0 {
        GraphMetaField::EntryPoint(v) => { /* ... */ }
        GraphMetaField::MaxLevel(v) => { /* ... */ }
        GraphMetaField::Count(v) => { /* ... */ }
        GraphMetaField::SpecHash(v) => { /* ... */ }
    }
}
```

### Benefits

1. **Single source of truth**: One enum defines all variants - no constants to sync
2. **Type-safe keys**: `GraphMetaCfKey::entry_point(code)` vs `GraphMetaCfKey(code, 0)`
3. **Compiler-checked exhaustiveness**: Adding a variant forces updates everywhere
4. **Self-documenting**: Key helpers show available fields
5. **No magic numbers**: Discriminant derived from `match`, not hardcoded

### Column Families Using This Pattern

| CF | Field Enum | Variants |
|---|---|---|
| `GraphMeta` | `GraphMetaField` | `EntryPoint(VecId)`, `MaxLevel(HnswLayer)`, `Count(u32)`, `SpecHash(u64)` |
| `IdAlloc` | `IdAllocField` | `NextId(VecId)`, `FreeBitmap(RoaringBitmapBytes)` |
