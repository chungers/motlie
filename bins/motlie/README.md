# motlie CLI

Command-line utility for working with Motlie graph databases.

## Installation

```bash
cargo install --path bins/motlie
```

Or build from source:

```bash
cargo build --release -p motlie
```

## Usage

```
motlie <COMMAND>

Commands:
  db    Database inspection commands
  help  Print this message or the help of the given subcommand(s)
```

## Database Commands

The `db` subcommand provides tools for inspecting and dumping Motlie graph database contents.

```
motlie db -p <DB_DIR> <COMMAND>

Options:
  -p, --db-dir <DB_DIR>  Path to the RocksDB database directory

Commands:
  list  List column families
  dump  Dump contents of a column family
```

### List Column Families

Display all available column families in the database schema:

```bash
motlie db -p /path/to/db list
```

Output:
```
Column families:
  nodes
  node-fragments
  edge-fragments
  outgoing-edges
  incoming-edges
  node-names
  edge-names
```

### Dump Column Family Contents

Dump records from a specific column family with pagination and filtering support.

```
motlie db -p <DB_DIR> dump <CF> [DATETIME] [OPTIONS]

Arguments:
  <CF>        Column family to dump
  [DATETIME]  Reference time for temporal validity filtering (optional)

Options:
      --last <LAST>      Cursor for pagination (exclusive start)
      --limit <LIMIT>    Maximum number of results [default: 100]
  -f, --format <FORMAT>  Output format [default: tsv] [possible values: tsv, table]
  -r, --reverse          Scan in reverse direction (from end to start)
```

#### Column Families

| Column Family | Description |
|--------------|-------------|
| `nodes` | Node metadata (ID, name, temporal validity) |
| `node-fragments` | Node content fragments with timestamps |
| `edge-fragments` | Edge content fragments with timestamps |
| `outgoing-edges` | Forward edges (source → destination) |
| `incoming-edges` | Reverse edge index (destination ← source) |
| `node-names` | Node name index for lookups by name |
| `edge-names` | Edge name index for lookups by name |

#### Output Columns

Each column family outputs different columns. All include `SINCE` and `UNTIL` columns for temporal validity:

| Column Family | Columns |
|--------------|---------|
| `nodes` | SINCE, UNTIL, ID, NAME |
| `node-fragments` | SINCE, UNTIL, NODE_ID, TIMESTAMP, MIME, CONTENT |
| `edge-fragments` | SINCE, UNTIL, SRC_ID, DST_ID, TIMESTAMP, EDGE_NAME, MIME, CONTENT |
| `outgoing-edges` | SINCE, UNTIL, SRC_ID, DST_ID, EDGE_NAME, WEIGHT |
| `incoming-edges` | SINCE, UNTIL, DST_ID, SRC_ID, EDGE_NAME |
| `node-names` | SINCE, UNTIL, NODE_ID, NAME |
| `edge-names` | SINCE, UNTIL, SRC_ID, DST_ID, NAME |

## Examples

### Basic Dump

```bash
# Dump first 100 nodes (default)
motlie db -p /path/to/db dump nodes

# Dump with custom limit
motlie db -p /path/to/db dump nodes --limit 10
```

### Output Formats

**TSV format** (default) - Tab-separated values suitable for piping to other tools:

```bash
motlie db -p /path/to/db dump nodes
```
```
-       -       01JGXYZ123456789ABCDEF       Alice
-       -       01JGXYZ123456789ABCDEG       Bob
```

**Table format** - Aligned columns with headers for human reading:

```bash
motlie db -p /path/to/db dump nodes -f table
```
```
SINCE                 UNTIL                 ID                          NAME
--------------------  --------------------  --------------------------  -----
-                     -                     01JGXYZ123456789ABCDEF      Alice
-                     -                     01JGXYZ123456789ABCDEG      Bob
```

### Temporal Filtering

Filter records by temporal validity. Only records valid at the specified time are returned.

```bash
# Records valid on January 1, 2024 (midnight)
motlie db -p /path/to/db dump nodes 2024-01-01

# Records valid at a specific time
motlie db -p /path/to/db dump nodes 2024-06-15-14:30:00
```

Datetime format:
- `YYYY-MM-DD` - Date only (assumes 00:00:00)
- `YYYY-MM-DD-HH:mm:ss` - Date and time

The `SINCE` and `UNTIL` columns show:
- In TSV format: milliseconds since Unix epoch
- In table format: `YYYY-MM-DD HH:mm:ss`
- `-` indicates no boundary (always valid from that direction)

### Reverse Scanning

Scan from end to start instead of start to end:

```bash
# Get the last 10 nodes added
motlie db -p /path/to/db dump nodes --limit 10 --reverse

# Short flag
motlie db -p /path/to/db dump nodes --limit 10 -r
```

### Pagination

For large datasets, use cursor-based pagination with the `--last` option:

```bash
# First page
motlie db -p /path/to/db dump nodes --limit 10

# Note the last ID from the output, then fetch the next page
motlie db -p /path/to/db dump nodes --limit 10 --last 01JGXYZ123456789ABCDEF
```

Cursor formats vary by column family:

| Column Family | Cursor Format | Example |
|--------------|---------------|---------|
| `nodes` | `<id>` | `01JGXYZ...` |
| `node-fragments` | `<node_id>:<timestamp>` | `01JGXYZ...:1704067200000` |
| `edge-fragments` | `<src_id>:<dst_id>:<edge_name>:<timestamp>` | `01JGX...:01JGY...:follows:1704067200000` |
| `outgoing-edges` | `<src_id>:<dst_id>:<edge_name>` | `01JGX...:01JGY...:follows` |
| `incoming-edges` | `<dst_id>:<src_id>:<edge_name>` | `01JGY...:01JGX...:follows` |
| `node-names` | `<name>:<node_id>` | `Alice:01JGXYZ...` |
| `edge-names` | `<name>:<src_id>:<dst_id>` | `follows:01JGX...:01JGY...` |

### Combined Options

Options can be combined for complex queries:

```bash
# Last 20 nodes valid on 2024-06-15, formatted as table
motlie db -p /path/to/db dump nodes 2024-06-15 --limit 20 --reverse -f table

# Page through edges with temporal filtering
motlie db -p /path/to/db dump outgoing-edges 2024-01-01 --limit 50 --last "01JGX...:01JGY...:follows"
```

### Dumping Fragments

Fragments contain the actual content attached to nodes and edges:

```bash
# Node content fragments
motlie db -p /path/to/db dump node-fragments -f table

# Edge content fragments
motlie db -p /path/to/db dump edge-fragments -f table
```

Fragment output includes:
- `MIME` - Content type (e.g., `text/plain`, `application/json`)
- `CONTENT` - Preview of text content (truncated to 60 chars for printable types)

### Dumping Indexes

Name indexes allow efficient lookups by name:

```bash
# Find nodes by name prefix (using pipe to grep)
motlie db -p /path/to/db dump node-names | grep "^Alice"

# Edge name index
motlie db -p /path/to/db dump edge-names -f table
```

## Piping and Scripting

TSV output is designed for integration with Unix tools:

```bash
# Count total nodes
motlie db -p /path/to/db dump nodes --limit 1000000 | wc -l

# Extract just node IDs
motlie db -p /path/to/db dump nodes | cut -f3

# Find nodes with specific name pattern
motlie db -p /path/to/db dump nodes | grep -i "pattern"

# Export to CSV
motlie db -p /path/to/db dump nodes | tr '\t' ',' > nodes.csv
```

## Exit Codes

- `0` - Success
- `1` - Error (database not found, invalid arguments, etc.)

## Environment

The CLI uses `tracing` for logging. Set the `RUST_LOG` environment variable to control log output:

```bash
RUST_LOG=debug motlie db -p /path/to/db dump nodes
```
