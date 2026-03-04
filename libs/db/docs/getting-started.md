# Getting Started with motlie-db

This guide helps you pick the right API surface and get productive quickly.

## API Surfaces

| Use case | Recommended API |
|---|---|
| Graph writes + graph/fulltext reads from one handle | `motlie_db::Storage` + `motlie_db::mutation` + `motlie_db::query` |
| Semantic/vector indexing and ANN search | `motlie_db::vector::*` |
| Shared initialization for graph + vector + fulltext subsystems | `motlie_db::storage_builder::StorageBuilder` |

Important: the root unified API currently composes **graph + fulltext**. Vector APIs are currently used through `motlie_db::vector`.

## 1) Unified Graph + Fulltext

Use root `Storage` when your primary flow is graph mutations and graph/fulltext queries.

- Base path layout:
  - `<base>/graph` (RocksDB)
  - `<base>/fulltext` (Tantivy)
- Create `Storage::readwrite(...)` or `Storage::readonly(...)`
- Call `.ready(StorageConfig::default())` to get typed handles
- Use `handles.writer()` for mutations and `handles.reader()` for queries

Runnable example:

- `libs/db/examples/unified_graph_fulltext.rs`

Run:

```bash
cargo run -p motlie-db --example unified_graph_fulltext
```

## 2) Vector Subsystem

Use `motlie_db::vector` for embedding registration, vector insert/delete, and KNN search.

Typical setup:

1. Open `vector::Storage`.
2. Initialize and wrap in `Arc`.
3. Bind registry storage (`registry.set_storage(...)`) and register an embedding (`EmbeddingBuilder`).
4. Spawn writer/query consumers.
5. Use `InsertVector` and `SearchKNN`.

Runnable example:

- `libs/db/examples/vector_basic.rs`

Run:

```bash
cargo run -p motlie-db --example vector_basic
```

## 3) Shared Subsystem Initialization

Use `StorageBuilder` when you need graph + vector to share one RocksDB `TransactionDB`, with fulltext initialized from the same base directory.

- Base path layout:
  - `<base>/rocksdb`
  - `<base>/tantivy`
- Register subsystems explicitly:
  - `graph::Subsystem`
  - `vector::Subsystem`
  - `fulltext::Schema`

Runnable example:

- `libs/db/examples/storage_builder_multi_subsystem.rs`

Run:

```bash
cargo run -p motlie-db --example storage_builder_multi_subsystem
```

## Operational Notes

1. `flush()` on graph/vector writers confirms backend commit visibility for those subsystems.
2. In the root unified graph -> fulltext pipeline, fulltext indexing is asynchronous relative to graph commit.
3. Use subsystem APIs directly when you need lower-level control over workers, channels, or lifecycle.
