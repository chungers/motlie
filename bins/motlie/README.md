# motlie CLI

Command-line utility for working with Motlie graph databases.

## Installation

### macOS (Apple Silicon / Intel)

#### 1. Install HDF5 Dependency (for benchmark features)

If you want to build with the `benchmark` feature (for vector search benchmarking with standard datasets):

```bash
# Install HDF5 via Homebrew (supports HDF5 1.8.4 through 2.0.0)
brew install hdf5
```

#### 2. Build Options

**Standard build (no benchmark features):**

```bash
cargo build --release --bin motlie
```

**Build with SIMD-optimized vector operations:**

```bash
cargo build --release --bin motlie --features simd-native
```

**Build with benchmark and SIMD features (requires HDF5):**

```bash
cargo build --release --bin motlie --features benchmark,simd-native
```

**Apple Silicon (M1/M2/M3/M4)**: Uses NEON SIMD instructions automatically.

**Intel Mac**: Uses AVX2 or SSE4.2 depending on CPU capabilities.

### Linux

```bash
# Ubuntu/Debian (for benchmark feature)
sudo apt-get install libhdf5-dev

# Build with SIMD auto-detection
cargo build --release --bin motlie --features simd-native

# Or with benchmark features
cargo build --release --bin motlie --features benchmark,simd-native
```

### Install from Source

```bash
cargo install --path bins/motlie
```

## Build Features

| Feature | Description |
|---------|-------------|
| (default) | Standard build with stderr tracing |
| `dtrace-otel` | Enables OpenTelemetry distributed tracing support |
| `benchmark` | Enables vector benchmark infrastructure (requires HDF5) |
| `simd-native` | Auto-detect best SIMD at compile time (recommended) |
| `simd-runtime` | Runtime SIMD dispatch for portable binaries |
| `simd-neon` | Force NEON instructions (Apple Silicon) |
| `simd-avx2` | Force AVX2 instructions (Intel/AMD 2013+) |
| `simd-avx512` | Force AVX-512 instructions (Intel Xeon/Ice Lake+) |
| `simd-none` | Scalar fallback (debugging, compatibility) |

### Building with OpenTelemetry Support

To enable distributed tracing via OpenTelemetry:

```bash
cargo build --release --bin motlie --features dtrace-otel
```

Or install with the feature:

```bash
cargo install --path bins/motlie --features dtrace-otel
```

### Building with Multiple Features

Combine features as needed:

```bash
# SIMD + OpenTelemetry
cargo build --release --bin motlie --features simd-native,dtrace-otel

# All features (requires HDF5)
HDF5_DIR=/opt/homebrew/opt/hdf5@1.10 cargo build --release --bin motlie --features benchmark,simd-native,dtrace-otel
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

# Scan nodes
motlie db -p /path/to/graph-db scan nodes --limit 10

# Scan with table formatting
motlie db -p /path/to/graph-db scan nodes -f table
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
