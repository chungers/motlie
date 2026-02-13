# motlie CLI

Command-line utility for working with Motlie graph databases.

## Installation

```bash
cargo install --path bins/motlie
```

Or build from source:

```bash
cargo build --release --bin motlie
```

## Build Features

| Feature | Description |
|---------|-------------|
| (default) | Standard build with stderr tracing |
| `dtrace-otel` | Enables OpenTelemetry distributed tracing support |

### Building with OpenTelemetry Support

To enable distributed tracing via OpenTelemetry:

```bash
cargo build --release --bin motlie --features dtrace-otel
```

Or install with the feature:

```bash
cargo install --path bins/motlie --features dtrace-otel
```

## Commands

| Command | Description | Documentation |
|---------|-------------|---------------|
| `db` | Database inspection and scanning | [docs/db.md](docs/db.md) |
| `fulltext` | Fulltext search indexing and querying | [docs/fulltext.md](docs/fulltext.md) |

## Quick Start

### Database Inspection

Scan and inspect the graph database contents:

```bash
# List column families
motlie db -p /path/to/graph-db list

# Scan nodes (latest versions)
motlie db -p /path/to/graph-db scan graph/nodes --limit 10

# Scan nodes active at a specific date (active period filtering)
motlie db -p /path/to/graph-db scan graph/nodes 2025-03-15 -f table

# Scan forward edges (outbound)
motlie db -p /path/to/graph-db scan graph/forward_edges --limit 10 -f table

# Reverse scan (latest first)
motlie db -p /path/to/graph-db scan graph/forward_edges --limit 10 --reverse -f table
```

See [docs/db.md](docs/db.md) for full documentation.

### Fulltext Search

Build a fulltext index and search:

```bash
# Build fulltext index from graph database
motlie fulltext -p /path/to/index index /path/to/graph-db

# Search for nodes
motlie fulltext -p /path/to/index search nodes "search query"

# Search for edges
motlie fulltext -p /path/to/index search edges "relationship"

# Get facet counts
motlie fulltext -p /path/to/index search facets
```

See [docs/fulltext.md](docs/fulltext.md) for full documentation.

## Output Formats

Both commands support two output formats:

| Format | Flag | Description |
|--------|------|-------------|
| TSV | `-f tsv` (default) | Tab-separated values for piping to other tools |
| Table | `-f table` | Aligned columns with headers for human reading |

## Environment Variables

The CLI uses `tracing` for structured logging and distributed tracing.

### Logging

| Variable | Description | Default |
|----------|-------------|---------|
| `RUST_LOG` | Log level filter | `debug` |

```bash
# Show debug logs
RUST_LOG=debug motlie db -p /path/to/db scan nodes

# Show info and above
RUST_LOG=info motlie fulltext -p /path/to/index search nodes "query"

# Show only warnings and errors
RUST_LOG=warn motlie db -p /path/to/db list
```

### Distributed Tracing (requires dtrace-otel feature)

When built with `--features dtrace-otel`, the CLI can export traces to an OpenTelemetry collector:

| Variable | Description | Default |
|----------|-------------|---------|
| `DTRACE_ENDPOINT` | OTLP collector endpoint URL (e.g., `http://localhost:4317`) | None |
| `DTRACE_SERVICE_NAME` | Service name for traces | `motlie` |

```bash
# Export traces to local Jaeger/Tempo/OTEL collector
DTRACE_ENDPOINT=http://localhost:4317 \
DTRACE_SERVICE_NAME=motlie-prod \
RUST_LOG=info \
motlie db -p /path/to/db scan nodes
```

When `DTRACE_ENDPOINT` is not set, the CLI falls back to stderr logging regardless of whether the feature is enabled.

### Example Trace Output

```bash
$ RUST_LOG=debug motlie db -p /tmp/test-db scan nodes --limit 2
2024-01-15T10:30:00.123Z DEBUG motlie_db::graph::mod path="/tmp/test-db" [Storage] Ready
2024-01-15T10:30:00.125Z  INFO motlie_db::graph::reader config=ReaderConfig { ... } Starting query consumer
2024-01-15T10:30:00.130Z DEBUG motlie_db::graph::reader query=ScanNodes Processing query
2024-01-15T10:30:00.135Z  INFO motlie starting
01HQXYZ123  Alice   data:text/plain;base64,...
01HQXYZ456  Bob     data:text/plain;base64,...
```

See [libs/db/README.md](../../libs/db/README.md#telemetry) for instrumentation details or [libs/core/src/telemetry.rs](../../libs/core/src/telemetry.rs) for telemetry initialization functions.

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Error |

## Tests

Integration tests are available for the fulltext commands:

```bash
cargo test --test fulltext_cli
```

See [docs/fulltext.md](docs/fulltext.md#tests) for test coverage details.

## Examples

Runnable demos are available under `bins/motlie/examples`:

### Vector DB Scan Demo

Creates a temporary vector database and scans all vector column families.

```bash
./bins/motlie/examples/demo_db_scan.sh
```

### Graph DB Scan Demo (Versioned + Active Periods)

Builds a small versioned graph with active periods, then scans all graph CFs.

```bash
./bins/motlie/examples/demo_db_scan_graph.sh
```

### Build Graph Tool

The graph scan demo uses a standalone builder:

```bash
cargo build --manifest-path bins/motlie/examples/build_graph/Cargo.toml
./bins/motlie/examples/build_graph/target/debug/build_graph /tmp/graphdb
```

## DB Scan Highlights

Common patterns when scanning large CFs:

```bash
# List column families
motlie db -p /path/to/db list

# Scan with pagination (vector/vectors)
motlie db -p /path/to/db scan vector/vectors --limit 5 -f tsv
motlie db -p /path/to/db scan vector/vectors --limit 5 --last "<EMBEDDING:VEC_ID>" -f tsv

# Reverse scan for newest entries
motlie db -p /path/to/db scan vector/vec_meta --limit 5 --reverse -f table
```

See [docs/db.md](docs/db.md) for full CLI usage, column families, and output columns.
