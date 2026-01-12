# RocksDB Storage Infrastructure

This module provides generic RocksDB storage infrastructure that eliminates
boilerplate across subsystems (graph, vector) while allowing subsystem-specific
behavior through the `StorageSubsystem` trait.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          StorageSubsystem Trait                             │
│                                                                             │
│  Defines per-subsystem:                                                     │
│  • Column families and RocksDB options                                      │
│  • In-memory cache type (NameCache, EmbeddingRegistry)                      │
│  • Pre-warm configuration and logic                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ implements
                    ┌───────────────┼───────────────┐
                    │               │               │
       ┌────────────┴────┐   ┌──────┴──────┐   ┌────┴────────────┐
       │ graph::Subsystem│   │vector::Sub. │   │ (future modules)│
       │                 │   │             │   │                 │
       │ Cache=NameCache │   │ Cache=Reg.  │   │                 │
       │ CFs: graph/*    │   │ CFs: vec/*  │   │                 │
       └─────────────────┘   └─────────────┘   └─────────────────┘
               │                     │
               ▼                     ▼
       ┌───────────────┐     ┌───────────────┐
       │Storage<Graph> │     │Storage<Vector>│     ← Generic Storage<S>
       │               │     │               │
       │ • readonly()  │     │ • readonly()  │
       │ • readwrite() │     │ • readwrite() │
       │ • ready()     │     │ • ready()     │
       │ • cache()     │     │ • cache()     │
       └───────────────┘     └───────────────┘
```

## Module Structure

```
rocksdb/
├── mod.rs          # Module exports and documentation
├── README.md       # This file
├── cf_traits.rs    # Column family trait hierarchy
├── config.rs       # BlockCacheConfig (shared configuration)
├── handle.rs       # DatabaseHandle, StorageMode, StorageOptions
├── storage.rs      # Generic Storage<S: StorageSubsystem>
└── subsystem.rs    # StorageSubsystem trait, ComponentWrapper, DbAccess
```

## Column Family Trait Hierarchy

The module defines a trait hierarchy for column family serialization:

```
                       ColumnFamily
                       (CF_NAME only)
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
         ▼                  ▼                  ▼
  ColumnFamilyConfig  ColumnFamilySerde  HotColumnFamilyRecord
       <C>            (MessagePack+LZ4)        (rkyv)
                            │
                            │ (used by)
                            ▼
                      MutationCodec
                   (mutation marshaling)
```

### Trait Descriptions

| Trait | Purpose | Used For |
|-------|---------|----------|
| `ColumnFamily` | Base marker with CF_NAME (single source of truth) | All CFs |
| `ColumnFamilyConfig<C>` | RocksDB options with domain-specific config | All CFs |
| `ColumnFamilySerde` | Cold CF serialization (MessagePack + LZ4) | Cold CFs: Names, Summaries, Fragments |
| `HotColumnFamilyRecord` | Hot CF zero-copy access using rkyv | Hot CFs: Nodes, Edges |
| `MutationCodec` | Marshals mutation structs to CF records | Cold CF mutations |

### Separation of Concerns

The hierarchy separates:
- **Storage infrastructure** (`ColumnFamily`, `ColumnFamilyConfig`, serde traits)
- **Mutation marshaling** (`MutationCodec` - converts business mutations to CF records)

This allows CFs like `Names` to implement `ColumnFamilySerde` for prewarm/iteration
without needing a mutation type, while mutation types own their marshaling logic.

### Hot vs Cold Column Families

**Hot CFs** (Nodes, ForwardEdges, ReverseEdges):
- Use `HotColumnFamilyRecord` with rkyv for zero-copy access
- Small, frequently-accessed during graph traversal
- Inherent `record_from`/`create_bytes` methods for mutation execution

**Cold CFs** (Names, Summaries, Fragments):
- Use `ColumnFamilySerde` with MessagePack + LZ4
- Larger, infrequently-accessed data
- `MutationCodec` for mutation marshaling (where applicable)

### Example: Using the Traits

**ColumnFamilySerde for cold CF serialization:**
```rust
impl ColumnFamilySerde for Names {
    type Key = NameCfKey;
    type Value = NameCfValue;

    fn key_to_bytes(key: &Self::Key) -> Vec<u8> {
        key.0.as_bytes().to_vec()
    }

    fn key_from_bytes(bytes: &[u8]) -> Result<Self::Key> {
        // Direct byte deserialization for fixed-size key
    }
    // value_to_bytes, value_from_bytes use default impl (MessagePack + LZ4)
}
```

**MutationCodec for mutation marshaling (cold CFs):**
```rust
// In mutation.rs
impl MutationCodec for AddNodeFragment {
    type Cf = NodeFragments;

    fn to_record(&self) -> (NodeFragmentCfKey, NodeFragmentCfValue) {
        let key = NodeFragmentCfKey(self.id, self.ts_millis);
        let value = NodeFragmentCfValue(self.valid_range.clone(), self.content.clone());
        (key, value)
    }
}

// Usage:
let (key_bytes, value_bytes) = add_fragment.to_cf_bytes()?;
db.put_cf(&cf, key_bytes, value_bytes)?;
```

**HotColumnFamilyRecord for hot CF zero-copy access:**
```rust
impl HotColumnFamilyRecord for Nodes {
    type Key = NodeCfKey;
    type Value = NodeCfValue;

    fn key_to_bytes(key: &Self::Key) -> Vec<u8> {
        key.0.into_bytes().to_vec()
    }

    fn key_from_bytes(bytes: &[u8]) -> Result<Self::Key> { ... }
    // value_to_bytes, value_from_bytes use rkyv
}

// Zero-copy hot path:
let archived = Nodes::value_archived(&value_bytes)?;
if archived.0.as_ref().map_or(true, |tr| tr.is_valid_at(now)) { ... }
```

**Generic prewarm helper:**
```rust
prewarm_cf::<Names, _>(db, 1000, |key, value| {
    cache.intern(key.0, &value.0);
    Ok(())
})?;
```

## Core Types

### `StorageSubsystem` Trait

The central abstraction for defining a storage subsystem:

```rust
pub trait StorageSubsystem: Send + Sync + 'static {
    /// Subsystem name ("graph", "vector")
    const NAME: &'static str;

    /// Column family names with subsystem prefix
    const COLUMN_FAMILIES: &'static [&'static str];

    /// Pre-warm configuration type
    type PrewarmConfig: Default + Clone + Send + Sync;

    /// In-memory cache type
    type Cache: Send + Sync;

    /// Create new cache instance
    fn create_cache() -> Arc<Self::Cache>;

    /// Build CF descriptors with shared block cache
    fn cf_descriptors(
        block_cache: &Cache,
        config: &BlockCacheConfig,
    ) -> Vec<ColumnFamilyDescriptor>;

    /// Pre-warm cache from database
    fn prewarm(
        db: &dyn DbAccess,
        cache: &Self::Cache,
        config: &Self::PrewarmConfig,
    ) -> Result<usize>;
}
```

### `Storage<S>` Generic Type

Generic storage that works with any subsystem:

```rust
pub struct Storage<S: StorageSubsystem> {
    // Common RocksDB infrastructure
    db: Option<DatabaseHandle>,
    mode: StorageMode,
    block_cache: Option<rocksdb::Cache>,

    // Subsystem-specific
    cache: Arc<S::Cache>,
    prewarm_config: S::PrewarmConfig,
}
```

### `RocksdbSubsystem` Trait

Extension trait for subsystems used with `StorageBuilder` (shared storage):

```rust
pub trait RocksdbSubsystem: SubsystemProvider<TransactionDB> {
    fn id(&self) -> &'static str;           // Short lookup ID ("graph", "vector")
    fn cf_names(&self) -> &'static [&'static str];
    fn cf_descriptors(&self, cache: &Cache, config: &BlockCacheConfig) -> Vec<ColumnFamilyDescriptor>;
}
```

See [SUBSYSTEM.md](../../SUBSYSTEM.md) for full trait hierarchy documentation.

## Usage Patterns

### Pattern 1: Standalone Storage

For applications using a single subsystem:

```rust
use motlie_db::graph;

// Type alias defined in graph module
// pub type Storage = rocksdb::Storage<graph::Subsystem>;

let mut storage = graph::Storage::readonly(path);
storage.ready()?;

// Access subsystem-specific cache
let name_cache: &Arc<NameCache> = storage.cache();

// Common operations
let txn_db = storage.transaction_db()?;
storage.try_catch_up_with_primary()?;
```

### Pattern 2: Shared Storage (StorageBuilder)

For applications sharing one TransactionDB across multiple subsystems:

```rust
use motlie_db::{graph, vector};
use motlie_db::storage_builder::StorageBuilder;

// Create subsystems and get cache references before boxing
let graph_subsystem = graph::Subsystem::new();
let name_cache = graph_subsystem.cache().clone();

let vector_subsystem = vector::Subsystem::new();
let registry = vector_subsystem.cache().clone();

let shared = StorageBuilder::new(path)
    .with_rocksdb(Box::new(graph_subsystem))
    .with_rocksdb(Box::new(vector_subsystem))
    .build()?;

// Access shared TransactionDB
let db = shared.db().unwrap();

// Access subsystems by ID
let graph = shared.get_component("graph");
let vector = shared.get_component("vector");

// name_cache and registry are still accessible via the cloned Arcs
assert_eq!(name_cache.len(), 0);  // Empty initially
assert!(registry.is_empty());
```

## Implementing a New Subsystem

To add a new storage subsystem:

### 1. Define the Subsystem Type

```rust
// my_module/subsystem.rs

use motlie_db::rocksdb::{BlockCacheConfig, DbAccess, StorageSubsystem};

pub struct Subsystem;

#[derive(Default, Clone)]
pub struct PrewarmConfig {
    pub limit: usize,
}

impl StorageSubsystem for Subsystem {
    const NAME: &'static str = "my_module";
    const COLUMN_FAMILIES: &'static [&'static str] = &[
        "my_module/data",
        "my_module/index",
    ];

    type PrewarmConfig = PrewarmConfig;
    type Cache = MyCache;

    fn create_cache() -> Arc<Self::Cache> {
        Arc::new(MyCache::new())
    }

    fn cf_descriptors(
        block_cache: &rocksdb::Cache,
        config: &BlockCacheConfig,
    ) -> Vec<ColumnFamilyDescriptor> {
        vec![
            // Define CF options with block cache
        ]
    }

    fn prewarm(
        db: &dyn DbAccess,
        cache: &Self::Cache,
        config: &Self::PrewarmConfig,
    ) -> Result<usize> {
        // Iterate CFs and populate cache
    }
}
```

### 2. Export Type Aliases

```rust
// my_module/mod.rs

pub mod subsystem;

pub use subsystem::{PrewarmConfig, Subsystem};

/// Standalone storage
pub type Storage = motlie_db::rocksdb::Storage<Subsystem>;

/// Component for StorageBuilder
pub type Component = motlie_db::rocksdb::ComponentWrapper<Subsystem>;

/// Convenience constructor
pub fn component() -> Component {
    Component::new()
}
```

### 3. Use the Subsystem

```rust
// Standalone
let mut storage = my_module::Storage::readwrite(path);
storage.ready()?;

// With StorageBuilder
let shared = StorageBuilder::new(path)
    .with_component(Box::new(my_module::component()))
    .build()?;
```

## Design Rationale

### Why Generic `Storage<S>` Instead of Inheritance?

Rust doesn't have inheritance. The generic approach provides:

1. **Zero-cost abstraction**: No virtual dispatch for common operations
2. **Type safety**: `storage.cache()` returns the correct cache type
3. **Code reuse**: ~250 lines of boilerplate eliminated per subsystem

### Why Two Traits: `StorageSubsystem` and `RocksdbSubsystem`?

Two different usage patterns require different trait designs:

1. **`StorageSubsystem`** (static dispatch, for `Storage<S>`):
   - Uses associated types (`type Cache`) for zero-cost abstractions
   - Uses associated constants (`const NAME`) for compile-time values
   - Not object-safe, but provides type-safe cache access

2. **`RocksdbSubsystem`** (dynamic dispatch, for `StorageBuilder`):
   - Object-safe trait for heterogeneous collections
   - Uses methods instead of associated types/constants
   - Enables `Vec<Box<dyn RocksdbSubsystem>>` in builder

```rust
// StorageSubsystem: static dispatch, not object-safe
impl StorageSubsystem for graph::Subsystem {
    type Cache = NameCache;           // Associated type
    const NAME: &'static str = "graph"; // Associated constant
}

// RocksdbSubsystem: dynamic dispatch, object-safe
impl RocksdbSubsystem for graph::Subsystem {
    fn id(&self) -> &'static str { "graph" }  // Method, not constant
}
```

Both traits are implemented by the same struct, allowing seamless use in either pattern.

### Pre-warming Architecture

Each subsystem manages its own cache pre-warming:

1. `StorageSubsystem::prewarm()` is called after database opens
2. Uses `DbAccess` trait to abstract over `DB` and `TransactionDB`
3. Populates subsystem-specific cache (NameCache, EmbeddingRegistry)
4. Configuration via `PrewarmConfig` (e.g., limit number of entries)

This isolation means:
- Graph module doesn't know about vector's EmbeddingRegistry
- Vector module doesn't know about graph's NameCache
- Each subsystem controls its own initialization

## Relationship with StorageBuilder

| Component | Purpose | Output |
|-----------|---------|--------|
| `Storage<S>` | Single subsystem, standalone | Owns DB + cache |
| `StorageBuilder` | Multiple subsystems, shared DB | `SharedStorage` |
| `Subsystem` structs | Implement both traits | Works in either pattern |

Both patterns use the same `Subsystem` struct. Implementing both
`StorageSubsystem` and `RocksdbSubsystem` ensures consistency between
standalone and shared deployments. See [SUBSYSTEM.md](../../SUBSYSTEM.md).
