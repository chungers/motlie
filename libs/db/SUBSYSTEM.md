# Subsystem Architecture

This document describes the subsystem trait hierarchy used in `motlie_db` for modular, composable storage backends.

## Overview

The subsystem architecture enables multiple storage modules (graph, vector, fulltext) to:

- Share a single RocksDB `TransactionDB` instance efficiently
- Share a single Tantivy index for full-text search
- Maintain module isolation (graph doesn't know about vector CFs)
- Provide consistent lifecycle hooks for initialization and shutdown
- Integrate with the telemetry system for observability

## Trait Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    motlie_core::telemetry                               │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ SubsystemInfo                                                     │  │
│  │   fn name() -> &'static str           // Display name             │  │
│  │   fn info_lines() -> Vec<(label, value)>  // Config for `info`    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ extends
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         motlie_db                                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ SubsystemProvider<B>                                              │  │
│  │   fn on_ready(&self, backend: &B) -> Result<()>                   │  │
│  │   fn on_shutdown(&self) -> Result<()>                             │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                    │                               │                    │
│         ┌─────────┴─────────┐           ┌─────────┴─────────┐          │
│         ▼                   ▼           ▼                   │          │
│  ┌─────────────────┐  ┌─────────────────┐                   │          │
│  │ RocksdbSubsystem│  │FulltextSubsystem│                   │          │
│  │ (B=TransactionDB)  │ (B=tantivy::Index)                  │          │
│  │                 │  │                 │                   │          │
│  │ fn id()         │  │ fn id()         │                   │          │
│  │ fn cf_names()   │  │ fn schema()     │                   │          │
│  │ fn cf_descriptors()│ fn writer_heap_size()              │          │
│  └─────────────────┘  └─────────────────┘                   │          │
│                                                             │          │
│  ┌──────────────────────────────────────────────────────────┘          │
│  │ StorageSubsystem (for standalone Storage<S>)                        │
│  │   const NAME: &'static str                                          │
│  │   const COLUMN_FAMILIES: &'static [&'static str]                    │
│  │   type PrewarmConfig                                                │
│  │   type Cache                                                        │
│  │   fn create_cache() -> Arc<Cache>                                   │
│  │   fn cf_descriptors(...)                                            │
│  │   fn prewarm(...)                                                   │
│  └─────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────┘
```

## Design Rationale

### Why Two Identifier Methods?

Each subsystem has two identifiers:

| Method | Purpose | Example |
|--------|---------|---------|
| `SubsystemInfo::name()` | Human-readable display name for UI/logs | `"Graph Database (RocksDB)"` |
| `id()` | Short programmatic identifier for lookup | `"graph"` |

This separation allows:
- Clean API: `storage.get_component("graph")` vs verbose display names
- Consistent display: Telemetry shows descriptive names
- Stable lookups: Short IDs don't change with display text updates

### Why Backend-Specific Extension Traits?

Extension traits (`RocksdbSubsystem`, `FulltextSubsystem`) are colocated with their backends because:

1. **Discoverability**: Developers find RocksDB traits in the `rocksdb` module
2. **Type safety**: Each trait is bound to a specific backend type
3. **Separation of concerns**: Base lifecycle is separate from backend-specific config

## Trait Details

### SubsystemInfo (motlie_core)

The base identity trait for all subsystems. Used for:
- Telemetry and observability
- The `info` command output
- Logging and tracing

```rust
use motlie_core::telemetry::SubsystemInfo;

impl SubsystemInfo for MySubsystem {
    fn name(&self) -> &'static str {
        "My Subsystem (RocksDB)"
    }

    fn info_lines(&self) -> Vec<(&'static str, String)> {
        vec![
            ("Cache Size", format!("{} MB", self.cache_size / (1024 * 1024))),
            ("Column Families", self.cf_count.to_string()),
        ]
    }
}
```

### SubsystemProvider\<B\>

The lifecycle trait, generic over backend type `B`.

```rust
use motlie_db::SubsystemProvider;

impl SubsystemProvider<TransactionDB> for MySubsystem {
    fn on_ready(&self, db: &TransactionDB) -> Result<()> {
        // Called after DB is opened
        // - Pre-warm caches from stored data
        // - Validate schema compatibility
        // - Initialize readers/writers
        let count = prewarm_cf::<MySpecs, _>(db, 1000, |key, value| {
            self.cache.insert(key, value);
            Ok(())
        })?;
        tracing::info!(count, "Pre-warmed cache");
        Ok(())
    }

    fn on_shutdown(&self) -> Result<()> {
        // Called before DB is closed
        // - Flush pending writes
        // - Persist in-memory state
        tracing::info!("Shutting down");
        Ok(())
    }
}
```

### RocksdbSubsystem

Extension for RocksDB-based subsystems. Used with `StorageBuilder` for shared storage.

```rust
use motlie_db::rocksdb::RocksdbSubsystem;

impl RocksdbSubsystem for MySubsystem {
    fn id(&self) -> &'static str {
        "my-subsystem"  // Used for storage.get_component("my-subsystem")
    }

    fn cf_names(&self) -> &'static [&'static str] {
        &["my-subsystem/data", "my-subsystem/index"]
    }

    fn cf_descriptors(
        &self,
        block_cache: &Cache,
        config: &BlockCacheConfig,
    ) -> Vec<ColumnFamilyDescriptor> {
        vec![
            ColumnFamilyDescriptor::new(
                "my-subsystem/data",
                Self::data_cf_options(block_cache, config),
            ),
            ColumnFamilyDescriptor::new(
                "my-subsystem/index",
                Self::index_cf_options(block_cache, config),
            ),
        ]
    }
}
```

### FulltextSubsystem

Extension for Tantivy-based subsystems.

```rust
use motlie_db::fulltext::FulltextSubsystem;

impl FulltextSubsystem for MyFulltextSchema {
    fn id(&self) -> &'static str {
        "my-fulltext"
    }

    fn schema(&self) -> tantivy::schema::Schema {
        let mut builder = Schema::builder();
        builder.add_text_field("content", TEXT | STORED);
        builder.add_facet_field("category", FacetOptions::default());
        builder.build()
    }

    fn writer_heap_size(&self) -> usize {
        100_000_000  // 100MB for bulk indexing
    }
}
```

### StorageSubsystem

For subsystems that manage their own standalone RocksDB instance via `Storage<S>`.

```rust
use motlie_db::rocksdb::StorageSubsystem;

impl StorageSubsystem for MySubsystem {
    const NAME: &'static str = "my-subsystem";
    const COLUMN_FAMILIES: &'static [&'static str] = &["my-subsystem/data"];

    type PrewarmConfig = MyPrewarmConfig;
    type Cache = MyCache;

    fn create_cache() -> Arc<Self::Cache> {
        Arc::new(MyCache::new())
    }

    fn cf_descriptors(
        block_cache: &Cache,
        config: &BlockCacheConfig,
    ) -> Vec<ColumnFamilyDescriptor> {
        // Return CF descriptors
    }

    fn prewarm(
        db: &dyn DbAccess,
        cache: &Self::Cache,
        config: &Self::PrewarmConfig,
    ) -> Result<usize> {
        // Pre-warm cache from DB
        Ok(loaded_count)
    }
}
```

## Lifecycle

```
┌─────────────────────────────────────────────────────────────────┐
│                        Startup                                  │
├─────────────────────────────────────────────────────────────────┤
│  1. Subsystem created with configuration                        │
│  2. StorageBuilder collects CF descriptors from all subsystems  │
│  3. Backend opened (RocksDB/Tantivy)                            │
│  4. on_ready(backend) called for each subsystem                 │
│     - Pre-warm caches                                           │
│     - Validate schema                                           │
│     - Initialize resources                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Application Runs                           │
│  - Subsystems process queries and mutations                     │
│  - Shared DB/Index accessed through SharedStorage               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Shutdown                                 │
├─────────────────────────────────────────────────────────────────┤
│  1. on_shutdown() called for each subsystem                     │
│     - Flush pending writes                                      │
│     - Persist in-memory state                                   │
│  2. Backend closed                                              │
└─────────────────────────────────────────────────────────────────┘
```

## Composition with StorageBuilder

`StorageBuilder` composes multiple subsystems into a single shared storage:

```rust
use motlie_db::storage_builder::StorageBuilder;
use motlie_db::{graph, vector, fulltext};

// Create subsystems (keep references for cache access)
let graph_subsystem = graph::Subsystem::new();
let name_cache = graph_subsystem.cache().clone();

let vector_subsystem = vector::Subsystem::new();
let embedding_registry = vector_subsystem.cache().clone();

// Build shared storage
let storage = StorageBuilder::new(&base_path)
    .with_rocksdb(Box::new(graph_subsystem))
    .with_rocksdb(Box::new(vector_subsystem))
    .with_fulltext(Box::new(fulltext::Schema::new()))
    .with_cache_size(512 * 1024 * 1024)  // 512MB shared block cache
    .build()?;

// Access components
let db = storage.db().expect("DB should exist");
let index = storage.index().expect("Index should exist");

// Access subsystems by ID
let graph = storage.get_component("graph");
let vector = storage.get_component("vector");
let fulltext = storage.get_fulltext("fulltext");

// List all component IDs
println!("RocksDB components: {:?}", storage.component_names());
println!("Fulltext components: {:?}", storage.fulltext_names());
println!("All CFs: {:?}", storage.all_cf_names());
```

### Directory Structure

StorageBuilder creates a structured directory layout:

```
<base_path>/
├── rocksdb/           # RocksDB TransactionDB
│   ├── 000003.log
│   ├── CURRENT
│   ├── LOCK
│   └── ...
└── tantivy/           # Tantivy fulltext index
    ├── meta.json
    ├── .managed.json
    └── ...
```

## Implementing a New Subsystem

### Step 1: Define the Subsystem Struct

```rust
// libs/db/src/mymodule/subsystem.rs

use std::sync::Arc;
use anyhow::Result;
use rocksdb::{Cache, ColumnFamilyDescriptor, TransactionDB};

use crate::rocksdb::{BlockCacheConfig, RocksdbSubsystem, StorageSubsystem};
use crate::SubsystemProvider;
use motlie_core::telemetry::SubsystemInfo;

/// My module's storage subsystem.
pub struct Subsystem {
    cache: Arc<MyCache>,
    config: MyConfig,
}

impl Subsystem {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(MyCache::new()),
            config: MyConfig::default(),
        }
    }

    pub fn with_config(mut self, config: MyConfig) -> Self {
        self.config = config;
        self
    }

    pub fn cache(&self) -> &Arc<MyCache> {
        &self.cache
    }
}

impl Default for Subsystem {
    fn default() -> Self {
        Self::new()
    }
}
```

### Step 2: Implement SubsystemInfo

```rust
impl SubsystemInfo for Subsystem {
    fn name(&self) -> &'static str {
        "My Module (RocksDB)"
    }

    fn info_lines(&self) -> Vec<(&'static str, String)> {
        vec![
            ("Prewarm Limit", self.config.prewarm_limit.to_string()),
            ("Column Families", ALL_COLUMN_FAMILIES.len().to_string()),
        ]
    }
}
```

### Step 3: Implement SubsystemProvider

```rust
impl SubsystemProvider<TransactionDB> for Subsystem {
    fn on_ready(&self, db: &TransactionDB) -> Result<()> {
        // Pre-warm cache from stored data
        let count = prewarm_cf::<MySpecs, _>(db, self.config.prewarm_limit, |key, value| {
            self.cache.insert(key.0, value.0);
            Ok(())
        })?;
        tracing::info!(subsystem = "my-module", count, "Pre-warmed cache");
        Ok(())
    }

    fn on_shutdown(&self) -> Result<()> {
        tracing::info!(subsystem = "my-module", "Shutting down");
        Ok(())
    }
}
```

### Step 4: Implement RocksdbSubsystem

```rust
impl RocksdbSubsystem for Subsystem {
    fn id(&self) -> &'static str {
        "my-module"
    }

    fn cf_names(&self) -> &'static [&'static str] {
        ALL_COLUMN_FAMILIES
    }

    fn cf_descriptors(
        &self,
        block_cache: &Cache,
        config: &BlockCacheConfig,
    ) -> Vec<ColumnFamilyDescriptor> {
        Self::build_cf_descriptors(block_cache, config)
    }
}
```

### Step 5: (Optional) Implement StorageSubsystem

If your module can also run standalone:

```rust
impl StorageSubsystem for Subsystem {
    const NAME: &'static str = "my-module";
    const COLUMN_FAMILIES: &'static [&'static str] = ALL_COLUMN_FAMILIES;

    type PrewarmConfig = MyConfig;
    type Cache = MyCache;

    fn create_cache() -> Arc<Self::Cache> {
        Arc::new(MyCache::new())
    }

    fn cf_descriptors(
        block_cache: &Cache,
        config: &BlockCacheConfig,
    ) -> Vec<ColumnFamilyDescriptor> {
        Self::build_cf_descriptors(block_cache, config)
    }

    fn prewarm(
        db: &dyn DbAccess,
        cache: &Self::Cache,
        config: &Self::PrewarmConfig,
    ) -> Result<usize> {
        prewarm_cf::<MySpecs, _>(db, config.prewarm_limit, |key, value| {
            cache.insert(key.0, value.0);
            Ok(())
        })
    }
}
```

## Telemetry Integration

Subsystems integrate with `motlie_core::telemetry` for observability:

### Using format_subsystem_info

```rust
use motlie_core::telemetry::{format_subsystem_info, SubsystemInfo};

// In your info command or startup logging
let graph = storage.get_component("graph").unwrap();
println!("{}", format_subsystem_info(graph));

// Output:
// [Graph Database (RocksDB)]
//   Prewarm Limit:       1000
//   Column Families:     8
```

### Startup Logging

```rust
use motlie_core::telemetry::{log_build_info, format_subsystem_info};

fn main() -> Result<()> {
    // Initialize telemetry
    motlie_core::telemetry::init_dev_subscriber();

    // Log build info
    log_build_info();

    // Build storage
    let storage = StorageBuilder::new(&path)
        .with_rocksdb(Box::new(graph::Subsystem::new()))
        .with_rocksdb(Box::new(vector::Subsystem::new()))
        .build()?;

    // Log subsystem info
    for id in storage.component_names() {
        if let Some(subsystem) = storage.get_component(id) {
            tracing::info!("{}", format_subsystem_info(subsystem));
        }
    }

    Ok(())
}
```

## Existing Subsystems

| Module | ID | Backend | Column Families |
|--------|-----|---------|-----------------|
| `graph` | `"graph"` | RocksDB | `graph/names`, `graph/nodes`, `graph/forward_edges`, `graph/reverse_edges`, `graph/node_fragments`, `graph/edge_fragments`, `graph/node_summaries`, `graph/edge_summaries` |
| `vector` | `"vector"` | RocksDB | `vector/embedding_specs`, `vector/vectors`, `vector/edges`, `vector/binary_codes`, `vector/vec_meta`, `vector/graph_meta`, `vector/id_forward`, `vector/id_reverse`, `vector/id_alloc`, `vector/pending` |
| `fulltext` | `"fulltext"` | Tantivy | N/A (Tantivy manages its own structure) |

## Best Practices

1. **Prefix column family names** with your module name (e.g., `vector/vectors`) to avoid conflicts
2. **Pre-warm critical data** in `on_ready()` for fast startup performance
3. **Keep `id()` short and stable** - it's used for programmatic lookup
4. **Use `SubsystemInfo::name()` for display** - it can be more descriptive
5. **Log subsystem activity** using the subsystem ID in tracing fields
6. **Clone Arc references before boxing** when you need to retain access to caches:
   ```rust
   let subsystem = vector::Subsystem::new();
   let registry = subsystem.cache().clone();  // Keep reference
   storage_builder.with_rocksdb(Box::new(subsystem));
   // `registry` still accessible
   ```
