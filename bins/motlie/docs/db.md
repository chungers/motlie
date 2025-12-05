# Database Commands

The `motlie db` command provides tools for inspecting and scanning Motlie graph database contents stored in RocksDB.

## Overview

The database commands allow you to:
- List available column families in the database schema
- Scan and dump contents of any column family
- Filter records by temporal validity
- Paginate through large datasets
- Output in TSV or formatted table format

## Commands

### List Column Families

Display all available column families in the database schema.

```bash
motlie db -p <db_dir> list
```

**Example:**
```bash
motlie db -p /data/graph-db list
```

**Output:**
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

### Scan Column Family

Scan and dump records from a specific column family.

```bash
motlie db -p <db_dir> scan <column_family> [datetime] [OPTIONS]
```

**Arguments:**
- `<column_family>` - Column family to scan (see table below)
- `[datetime]` - Optional reference time for temporal validity filtering

**Options:**
- `-p, --db-dir <path>` - Path to the RocksDB database directory (required)
- `--last <cursor>` - Cursor for pagination (exclusive start)
- `--limit <n>` - Maximum number of results (default: 100)
- `-f, --format <format>` - Output format: `tsv` or `table` (default: tsv)
- `-r, --reverse` - Scan in reverse direction (from end to start)

## Column Families

| Column Family | CLI Value | Description |
|--------------|-----------|-------------|
| Nodes | `nodes` | Node metadata (ID, name, temporal validity) |
| Node Fragments | `node-fragments` | Node content fragments with timestamps |
| Edge Fragments | `edge-fragments` | Edge content fragments with timestamps |
| Outgoing Edges | `outgoing-edges` | Forward edges (source → destination) |
| Incoming Edges | `incoming-edges` | Reverse edge index (destination ← source) |
| Node Names | `node-names` | Node name index for lookups by name |
| Edge Names | `edge-names` | Edge name index for lookups by name |

## Output Columns

Each column family outputs different columns. All include `SINCE` and `UNTIL` columns for temporal validity.

### Nodes
```
SINCE    UNTIL    ID    NAME
```

### Node Fragments
```
SINCE    UNTIL    NODE_ID    TIMESTAMP    MIME    CONTENT
```

### Edge Fragments
```
SINCE    UNTIL    SRC_ID    DST_ID    TIMESTAMP    EDGE_NAME    MIME    CONTENT
```

### Outgoing Edges
```
SINCE    UNTIL    SRC_ID    DST_ID    EDGE_NAME    WEIGHT
```

### Incoming Edges
```
SINCE    UNTIL    DST_ID    SRC_ID    EDGE_NAME
```

### Node Names
```
SINCE    UNTIL    NODE_ID    NAME
```

### Edge Names
```
SINCE    UNTIL    SRC_ID    DST_ID    NAME
```

## Output Formats

### TSV (Tab-Separated Values)

Default format, suitable for piping to other tools:

```bash
motlie db -p /data/graph-db scan nodes
```

```
-       -       01JGXYZ123456789ABCDEF       Alice
-       -       01JGXYZ123456789ABCDEG       Bob
```

### Table

Human-readable format with aligned columns and headers:

```bash
motlie db -p /data/graph-db scan nodes -f table
```

```
SINCE                 UNTIL                 ID                          NAME
--------------------  --------------------  --------------------------  -----
-                     -                     01JGXYZ123456789ABCDEF      Alice
-                     -                     01JGXYZ123456789ABCDEG      Bob
```

## Temporal Filtering

Filter records by temporal validity. Only records valid at the specified time are returned.

### Datetime Formats

| Format | Example | Description |
|--------|---------|-------------|
| `YYYY-MM-DD` | `2024-01-01` | Date only (assumes 00:00:00) |
| `YYYY-MM-DD-HH:mm:ss` | `2024-06-15-14:30:00` | Date and time |

### Examples

```bash
# Records valid on January 1, 2024 (midnight)
motlie db -p /data/graph-db scan nodes 2024-01-01

# Records valid at a specific time
motlie db -p /data/graph-db scan nodes 2024-06-15-14:30:00
```

### Temporal Validity Display

The `SINCE` and `UNTIL` columns show:

| Format | SINCE/UNTIL Display |
|--------|---------------------|
| TSV | Milliseconds since Unix epoch |
| Table | `YYYY-MM-DD HH:mm:ss` |
| No boundary | `-` (always valid from that direction) |

## Pagination

For large datasets, use cursor-based pagination with the `--last` option.

### Basic Pagination

```bash
# First page
motlie db -p /data/graph-db scan nodes --limit 10

# Note the last ID from the output, then fetch the next page
motlie db -p /data/graph-db scan nodes --limit 10 --last 01JGXYZ123456789ABCDEF
```

### Cursor Formats by Column Family

| Column Family | Cursor Format | Example |
|--------------|---------------|---------|
| `nodes` | `<id>` | `01JGXYZ...` |
| `node-fragments` | `<node_id>:<timestamp>` | `01JGXYZ...:1704067200000` |
| `edge-fragments` | `<src_id>:<dst_id>:<edge_name>:<timestamp>` | `01JGX...:01JGY...:follows:1704067200000` |
| `outgoing-edges` | `<src_id>:<dst_id>:<edge_name>` | `01JGX...:01JGY...:follows` |
| `incoming-edges` | `<dst_id>:<src_id>:<edge_name>` | `01JGY...:01JGX...:follows` |
| `node-names` | `<name>:<node_id>` | `Alice:01JGXYZ...` |
| `edge-names` | `<name>:<src_id>:<dst_id>` | `follows:01JGX...:01JGY...` |

## Reverse Scanning

Scan from end to start instead of start to end:

```bash
# Get the last 10 nodes added
motlie db -p /data/graph-db scan nodes --limit 10 --reverse

# Short flag
motlie db -p /data/graph-db scan nodes --limit 10 -r
```

## Examples

### Basic Scanning

```bash
# Scan first 100 nodes (default)
motlie db -p /data/graph-db scan nodes

# Scan with custom limit
motlie db -p /data/graph-db scan nodes --limit 10

# Scan in table format
motlie db -p /data/graph-db scan nodes -f table
```

### Scanning Fragments

Fragments contain the actual content attached to nodes and edges:

```bash
# Node content fragments
motlie db -p /data/graph-db scan node-fragments -f table

# Edge content fragments
motlie db -p /data/graph-db scan edge-fragments -f table
```

Fragment output includes:
- `MIME` - Content type (e.g., `text/plain`, `application/json`, `text/markdown`)
- `CONTENT` - Preview of text content (truncated to 60 chars for printable types)

### Scanning Indexes

Name indexes allow efficient lookups by name:

```bash
# Scan node names index
motlie db -p /data/graph-db scan node-names -f table

# Scan edge names index
motlie db -p /data/graph-db scan edge-names -f table
```

### Combined Options

Options can be combined for complex queries:

```bash
# Last 20 nodes valid on 2024-06-15, formatted as table
motlie db -p /data/graph-db scan nodes 2024-06-15 --limit 20 --reverse -f table

# Page through edges with temporal filtering
motlie db -p /data/graph-db scan outgoing-edges 2024-01-01 --limit 50 --last "01JGX...:01JGY...:follows"
```

## Piping and Scripting

TSV output is designed for integration with Unix tools:

```bash
# Count total nodes
motlie db -p /data/graph-db scan nodes --limit 1000000 | wc -l

# Extract just node IDs (3rd column)
motlie db -p /data/graph-db scan nodes | cut -f3

# Find nodes with specific name pattern
motlie db -p /data/graph-db scan nodes | grep -i "pattern"

# Export to CSV
motlie db -p /data/graph-db scan nodes | tr '\t' ',' > nodes.csv

# Find nodes by name prefix
motlie db -p /data/graph-db scan node-names | grep "^Alice"
```

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| "Database not found" | Invalid db_dir path | Verify the path exists |
| "Invalid cursor format" | Wrong --last format | Use the correct format for the column family |
| "Invalid ID" | Malformed ULID in cursor | Verify the ID is a valid ULID |
| "Invalid datetime format" | Wrong date/time format | Use `YYYY-MM-DD` or `YYYY-MM-DD-HH:mm:ss` |

## Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Error (database not found, invalid arguments, etc.) |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RUST_LOG` | Control log verbosity (e.g., `debug`, `info`, `warn`) |

```bash
RUST_LOG=debug motlie db -p /data/graph-db scan nodes
```
