# Database Commands

The `motlie db` command provides tools for inspecting and scanning Motlie database contents stored in RocksDB, covering both graph and vector subsystems.

## Overview

The database commands allow you to:
- List available column families in the database schema
- Scan and dump contents of any column family (graph or vector)
- Filter graph records by active period
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
  graph/names
  graph/nodes
  graph/node_fragments
  graph/node_summaries
  graph/node_summary_index
  graph/node_version_history
  graph/edge_fragments
  graph/edge_summaries
  graph/edge_summary_index
  graph/edge_version_history
  graph/forward_edges
  graph/reverse_edges
  graph/orphan_summaries
  graph/meta
  vector/embedding_specs
  vector/vectors
  vector/edges
  vector/binary_codes
  vector/vec_meta
  vector/graph_meta
  vector/id_forward
  vector/id_reverse
  vector/id_alloc
  vector/pending
  vector/lifecycle_counts
```

### Scan Column Family

Scan and dump records from a specific column family.

```bash
motlie db -p <db_dir> scan <column_family> [datetime] [OPTIONS]
```

**Arguments:**
- `<column_family>` - Column family to scan (see tables below)
- `[datetime]` - Optional reference time for active period filtering (graph CFs only)

**Options:**
- `-p, --db-dir <path>` - Path to the RocksDB database directory (required)
- `--last <cursor>` - Cursor for pagination (exclusive start)
- `--limit <n>` - Maximum number of results (default: 100)
- `-f, --format <format>` - Output format: `tsv` or `table` (default: table)
- `-r, --reverse` - Scan in reverse direction (from end to start)

## Graph Column Families

| Column Family | CLI Value | Description |
|--------------|-----------|-------------|
| Nodes | `graph/nodes` | Node metadata (ID, name, active period) |
| Node Fragments | `graph/node_fragments` | Node content fragments with timestamps |
| Edge Fragments | `graph/edge_fragments` | Edge content fragments with timestamps |
| Outgoing Edges | `graph/forward_edges` | Forward edges (source → destination) |
| Incoming Edges | `graph/reverse_edges` | Reverse edge index (destination ← source) |
| Names | `graph/names` | Name hash → name string mapping |
| Node Summaries | `graph/node_summaries` | Content-addressed node summary blobs |
| Edge Summaries | `graph/edge_summaries` | Content-addressed edge summary blobs |
| Node Summary Index | `graph/node_summary_index` | Summary hash → node version index |
| Edge Summary Index | `graph/edge_summary_index` | Summary hash → edge version index |
| Node Version History | `graph/node_version_history` | Node version timeline |
| Edge Version History | `graph/edge_version_history` | Edge version timeline |
| Orphan Summaries | `graph/orphan_summaries` | Unreferenced summary cleanup tracking |
| Graph Meta | `graph/meta` | Database-level metadata |

## Vector Column Families

| Column Family | CLI Value | Description |
|--------------|-----------|-------------|
| Embedding Specs | `vector/embedding_specs` | Embedding space definitions (model, dim, distance) |
| Vectors | `vector/vectors` | Raw vector data (f32 arrays) |
| HNSW Edges | `vector/edges` | HNSW graph neighbor lists per layer |
| Binary Codes | `vector/binary_codes` | RaBitQ quantized codes with ADC corrections |
| Vec Metadata | `vector/vec_meta` | Per-vector metadata (layer, lifecycle, timestamps) |
| Graph Metadata | `vector/graph_meta` | Per-embedding HNSW graph metadata (entry point, levels) |
| ID Forward | `vector/id_forward` | External key → internal vec_id mapping |
| ID Reverse | `vector/id_reverse` | Internal vec_id → external key mapping |
| ID Alloc | `vector/id_alloc` | ID allocator state (next ID, free bitmap) |
| Pending | `vector/pending` | Pending async graph update queue |
| Lifecycle Counts | `vector/lifecycle_counts` | Per-embedding lifecycle statistics |

## Output Columns

### Graph CFs

Graph CFs with temporal records include `SINCE` and `UNTIL` columns for active period.

#### Nodes
```
SINCE    UNTIL    ID    NAME
```

#### Node Fragments
```
SINCE    UNTIL    NODE_ID    TIMESTAMP    MIME    CONTENT
```

#### Edge Fragments
```
SINCE    UNTIL    SRC_ID    DST_ID    TIMESTAMP    EDGE_NAME    MIME    CONTENT
```

#### Outgoing Edges
```
SINCE    UNTIL    SRC_ID    DST_ID    EDGE_NAME    WEIGHT
```

#### Incoming Edges
```
SINCE    UNTIL    DST_ID    SRC_ID    EDGE_NAME
```

#### Names
```
HASH    NAME
```

#### Node/Edge Summaries
```
HASH    MIME    CONTENT
```

#### Node/Edge Summary Index
```
HASH    NODE_ID/EDGE_KEY    VERSION    STATUS
```

#### Node/Edge Version History
```
NODE_ID/EDGE_KEY    VERSION    SUMMARY_HASH    CREATED_AT
```

#### Orphan Summaries
```
ORPHANED_AT    HASH    KIND
```

#### Graph Meta
```
FIELD    CURSOR_BYTES
```

### Vector CFs

#### Embedding Specs
```
CODE    MODEL    DIM    DISTANCE    STORAGE    M    EF    RABITQ_BITS
```

#### Vectors
```
EMBEDDING    VEC_ID    DIM    BYTES
```

#### HNSW Edges
```
EMBEDDING    VEC_ID    LAYER    NEIGHBOR_BYTES
```

#### Binary Codes
```
EMBEDDING    VEC_ID    CODE_LEN    NORM    QERR
```

#### Vec Metadata
```
EMBEDDING    VEC_ID    MAX_LAYER    LIFECYCLE    CREATED_AT
```

#### Graph Metadata
```
EMBEDDING    FIELD    VALUE
```

#### ID Forward
```
EMBEDDING    EXT_KEY_TYPE    EXT_KEY    VEC_ID
```

#### ID Reverse
```
EMBEDDING    VEC_ID    EXT_KEY_TYPE    EXT_KEY
```

#### ID Alloc
```
EMBEDDING    FIELD    VALUE
```

#### Pending
```
EMBEDDING    TIMESTAMP    VEC_ID
```

#### Lifecycle Counts
```
EMBEDDING    INDEXED    PENDING    DELETED    PENDING_DELETED
```

## Output Formats

### TSV (Tab-Separated Values)

Default format, suitable for piping to other tools:

```bash
motlie db -p /data/graph-db scan graph/nodes
```

```
-       -       01JGXYZ123456789ABCDEF       Alice
-       -       01JGXYZ123456789ABCDEG       Bob
```

### Table

Human-readable format with aligned columns and headers:

```bash
motlie db -p /data/graph-db scan graph/nodes -f table
```

```
SINCE                 UNTIL                 ID                          NAME
--------------------  --------------------  --------------------------  -----
-                     -                     01JGXYZ123456789ABCDEF      Alice
-                     -                     01JGXYZ123456789ABCDEG      Bob
```

## Temporal Filtering

Filter graph records by active period. Only records valid at the specified time are returned. This applies to graph CFs only (nodes, fragments, edges).

### Datetime Formats

| Format | Example | Description |
|--------|---------|-------------|
| `YYYY-MM-DD` | `2024-01-01` | Date only (assumes 00:00:00) |
| `YYYY-MM-DD-HH:mm:ss` | `2024-06-15-14:30:00` | Date and time |

### Examples

```bash
# Records valid on January 1, 2024 (midnight)
motlie db -p /data/graph-db scan graph/nodes 2024-01-01

# Records valid at a specific time
motlie db -p /data/graph-db scan graph/nodes 2024-06-15-14:30:00
```

### Active period Display

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
motlie db -p /data/graph-db scan graph/nodes --limit 10

# Note the last ID from the output, then fetch the next page
motlie db -p /data/graph-db scan graph/nodes --limit 10 --last 01JGXYZ123456789ABCDEF
```

### Cursor Formats

#### Graph CFs

| Column Family | Cursor Format | Example |
|--------------|---------------|---------|
| `graph/nodes` | `<id>` | `01JGXYZ...` |
| `graph/node_fragments` | `<node_id>:<timestamp>` | `01JGXYZ...:1704067200000` |
| `graph/edge_fragments` | `<src_id>:<dst_id>:<edge_name>:<timestamp>` | `01JGX...:01JGY...:follows:1704067200000` |
| `graph/forward_edges` | `<src_id>:<dst_id>:<edge_name>` | `01JGX...:01JGY...:follows` |
| `graph/reverse_edges` | `<dst_id>:<src_id>:<edge_name>` | `01JGY...:01JGX...:follows` |
| `graph/names` | `<hash_hex>` | `a1b2c3d4e5f60708` |
| `graph/node_summaries` | `<hash_hex>` | `a1b2c3d4e5f60708` |
| `graph/edge_summaries` | `<hash_hex>` | `a1b2c3d4e5f60708` |
| `graph/node_summary_index` | `<hash_hex>:<node_id>:<version>` | `a1b2c3d4e5f60708:01JGXYZ...:1` |
| `graph/edge_summary_index` | `<hash_hex>:<src_id>:<dst_id>:<name_hash>:<version>` | `a1b2....:01JGX...:01JGY...:a1b2...:1` |
| `graph/node_version_history` | `<node_id>:<version>` | `01JGXYZ...:1` |
| `graph/edge_version_history` | `<src_id>:<dst_id>:<name_hash>:<version>` | `01JGX...:01JGY...:a1b2...:1` |
| `graph/orphan_summaries` | `<timestamp>:<hash_hex>` | `1704067200000:a1b2c3d4e5f60708` |

#### Vector CFs

| Column Family | Cursor Format | Example |
|--------------|---------------|---------|
| `vector/embedding_specs` | `<embedding_code>` | `42` |
| `vector/vectors` | `<embedding_code>:<vec_id>` | `42:100` |
| `vector/edges` | `<embedding_code>:<vec_id>:<layer>` | `42:100:0` |
| `vector/binary_codes` | `<embedding_code>:<vec_id>` | `42:100` |
| `vector/vec_meta` | `<embedding_code>:<vec_id>` | `42:100` |
| `vector/graph_meta` | `<embedding_code>` | `42` |
| `vector/id_forward` | N/A (no cursor support) | — |
| `vector/id_reverse` | `<embedding_code>:<vec_id>` | `42:100` |
| `vector/id_alloc` | `<embedding_code>` | `42` |
| `vector/pending` | `<embedding_code>:<timestamp>:<vec_id>` | `42:1704067200000:100` |
| `vector/lifecycle_counts` | `<embedding_code>` | `42` |

## Reverse Scanning

Scan from end to start instead of start to end:

```bash
# Get the last 10 nodes added
motlie db -p /data/graph-db scan graph/nodes --limit 10 --reverse

# Short flag
motlie db -p /data/graph-db scan graph/nodes --limit 10 -r
```

## Examples

### Basic Scanning

```bash
# Scan first 100 nodes (default)
motlie db -p /data/graph-db scan graph/nodes

# Scan with custom limit
motlie db -p /data/graph-db scan graph/nodes --limit 10

# Scan in table format
motlie db -p /data/graph-db scan graph/nodes -f table
```

### Scanning Graph Fragments

Fragments contain the actual content attached to nodes and edges:

```bash
# Node content fragments
motlie db -p /data/graph-db scan graph/node_fragments -f table

# Edge content fragments
motlie db -p /data/graph-db scan graph/edge_fragments -f table
```

Fragment output includes:
- `MIME` - Content type (e.g., `text/plain`, `application/json`, `text/markdown`)
- `CONTENT` - Preview of text content (truncated to 60 chars for printable types)

### Scanning Vector Data

```bash
# List embedding spaces
motlie db -p /data/db scan vector/embedding_specs -f table

# Scan vector metadata
motlie db -p /data/db scan vector/vec_meta -f table --limit 20

# Check lifecycle counts per embedding
motlie db -p /data/db scan vector/lifecycle_counts -f table

# Scan ID mappings
motlie db -p /data/db scan vector/id_forward -f table --limit 50
```

### Combined Options

Options can be combined for complex queries:

```bash
# Last 20 nodes valid on 2024-06-15, formatted as table
motlie db -p /data/graph-db scan graph/nodes 2024-06-15 --limit 20 --reverse -f table

# Page through edges with temporal filtering
motlie db -p /data/graph-db scan graph/forward_edges 2024-01-01 --limit 50 --last "01JGX...:01JGY...:follows"
```

## Piping and Scripting

TSV output is designed for integration with Unix tools:

```bash
# Count total nodes
motlie db -p /data/graph-db scan graph/nodes --limit 1000000 | wc -l

# Extract just node IDs (3rd column)
motlie db -p /data/graph-db scan graph/nodes | cut -f3

# Find nodes with specific name pattern
motlie db -p /data/graph-db scan graph/nodes | grep -i "pattern"

# Export to CSV
motlie db -p /data/graph-db scan graph/nodes | tr '\t' ',' > nodes.csv

# Count vectors per embedding space
motlie db -p /data/db scan vector/lifecycle_counts | cut -f1,2
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
RUST_LOG=debug motlie db -p /data/graph-db scan graph/nodes
```
