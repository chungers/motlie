# motlie CLI

Command-line tools for inspecting Motlie graph/vector RocksDB data and building/querying a fulltext index.

## What This CLI Covers

- `motlie db`: inspect graph + vector column families from a RocksDB database
- `motlie fulltext`: build/search a Tantivy index over graph content
- `motlie info`: print build/subsystem configuration

For high-throughput vector benchmarking and ANN experiments, use `bench_vector` (see `bins/bench_vector/README.md`).

## Build and Install

### Build

```bash
# Standard build
cargo build --release --bin motlie

# Build with OpenTelemetry support
cargo build --release --bin motlie --features dtrace-otel
```

### Install from source

```bash
cargo install --path . --bin motlie
```

## Quick Start

### 1) Check build/runtime capabilities

```bash
motlie info
```

### 2) Inspect graph and vector data in RocksDB

```bash
# List available column families
motlie db -p /path/to/db list

# Scan graph nodes
motlie db -p /path/to/db scan graph/nodes --limit 10

# Scan forward edges
motlie db -p /path/to/db scan graph/forward_edges --limit 10

# Scan vector embedding specs
motlie db -p /path/to/db scan vector/embedding_specs --limit 10

# Time-filter graph records (YYYY-MM-DD or YYYY-MM-DD-HH:mm:ss)
motlie db -p /path/to/db scan graph/nodes 2026-02-14 --limit 50
```

### 3) Build and query fulltext index

```bash
# Build fulltext index from graph DB
motlie fulltext -p /path/to/index index /path/to/db

# Search nodes
motlie fulltext -p /path/to/index search nodes "query text"

# Search edges with fuzzy matching
motlie fulltext -p /path/to/index search edges "influenc" -f low

# Show facets
motlie fulltext -p /path/to/index search facets
```

## Graph, Fulltext, and Vector Workflow

1. Graph data lives in RocksDB (`graph/*` CFs).
2. Fulltext index is built from graph records into a separate directory.
3. Vector ANN data lives in RocksDB (`vector/*` CFs), and can be inspected with `motlie db scan`.
4. For vector index/query benchmarking and parameter sweeps, use `bench_vector`.

## Commands

| Command | Purpose | Docs |
|---|---|---|
| `motlie info` | Print build + subsystem info | - |
| `motlie db ...` | List/scan graph + vector column families | [docs/db.md](docs/db.md) |
| `motlie fulltext ...` | Build and query fulltext index | [docs/fulltext.md](docs/fulltext.md) |

## Output Formats

- `motlie db` uses `-f, --format` with values `table` (default) or `tsv`.
- `motlie fulltext search` and `motlie fulltext index` use `-o, --format` with values `table` (default) or `tsv`.

Examples:

```bash
motlie db -p /path/to/db scan graph/nodes -f tsv
motlie fulltext -p /path/to/index search -o table nodes "rust"
```

## Environment Variables

### Logging

| Variable | Description | Default |
|---|---|---|
| `RUST_LOG` | Tracing filter | `debug` |

```bash
RUST_LOG=info motlie db -p /path/to/db scan graph/nodes --limit 5
```

### OpenTelemetry (`dtrace-otel` builds)

| Variable | Description | Default |
|---|---|---|
| `DTRACE_ENDPOINT` | OTLP collector endpoint URL | unset |
| `DTRACE_SERVICE_NAME` | Service name | `motlie` |

```bash
DTRACE_ENDPOINT=http://localhost:4317 \
DTRACE_SERVICE_NAME=motlie \
RUST_LOG=info \
motlie db -p /path/to/db scan graph/nodes
```

If `DTRACE_ENDPOINT` is not set, the CLI falls back to stderr logging.

## Exit Behavior

- `motlie db` returns non-zero on scan/list failures.
- `motlie fulltext` surfaces runtime failures in logs; if you are scripting around it, check stderr/log output in addition to process exit status.

## Tests

```bash
cargo test --test db_cli
cargo test --test fulltext_cli
```
