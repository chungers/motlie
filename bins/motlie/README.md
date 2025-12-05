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

## Environment

The CLI uses `tracing` for logging. Set the `RUST_LOG` environment variable to control log output:

```bash
RUST_LOG=debug motlie db -p /path/to/db scan nodes
RUST_LOG=info motlie fulltext -p /path/to/index search nodes "query"
```

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
