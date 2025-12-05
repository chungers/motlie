# Fulltext Search Commands

The `motlie fulltext` command provides fulltext search capabilities for the graph database using Tantivy. It supports indexing nodes, edges, and their fragments, as well as searching with fuzzy matching, tag filtering, and faceted search.

## Overview

The fulltext search workflow consists of two phases:

1. **Indexing**: Scan the graph database (RocksDB) and build a Tantivy fulltext index
2. **Searching**: Query the fulltext index to find nodes and edges

## Commands

### Index Command

Build a fulltext index from an existing graph database.

```bash
motlie fulltext -p <index_dir> index [OPTIONS] <graph_db_dir>
```

**Arguments:**
- `<graph_db_dir>` - Path to the graph RocksDB directory (used with `motlie db` commands)

**Options:**
- `-p, --index-dir <path>` - Path to the fulltext index directory (required)
- `-b, --batch-size <n>` - Number of records to batch before sending to indexer (default: 100)
- `-o, --format <format>` - Output format: `tsv` or `table` (default: tsv)

**Example:**
```bash
# Index a graph database with default settings
motlie fulltext -p /data/fulltext-index index /data/graph-db

# Index with larger batch size for better performance
motlie fulltext -p /data/fulltext-index index -b 500 /data/graph-db

# Index with table output format
motlie fulltext -p /data/fulltext-index index -o table /data/graph-db
```

**Notes:**
- The index directory must be empty or non-existent. Re-indexing to an existing index directory will fail.
- The command indexes all nodes, edges, node fragments, and edge fragments.
- Progress is printed to stderr showing batch counts and totals.
- Records are printed to stdout as they are indexed.

### Search Commands

Search the fulltext index for nodes, edges, or get facet counts.

#### Search Nodes

```bash
motlie fulltext -p <index_dir> search nodes [OPTIONS] <query>
```

**Arguments:**
- `<query>` - The search query string

**Options:**
- `-l, --limit <n>` - Maximum number of results (default: 10)
- `-f, --fuzzy-level <level>` - Fuzzy matching: `none`, `low`, `medium` (default: none)
- `-t, --tags <tag>` - Filter by tags (can be repeated, matches ANY tag)
- `-o, --format <format>` - Output format: `tsv` or `table` (default: tsv)

**Output format (TSV):**
```
SCORE    ID    FRAGMENT_TS    SNIPPET
```

**Example:**
```bash
# Simple search
motlie fulltext -p /data/fulltext-index search nodes "programming language"

# Search with fuzzy matching for typo tolerance
motlie fulltext -p /data/fulltext-index search nodes "Pythn" -f low

# Search with tag filter
motlie fulltext -p /data/fulltext-index search nodes "language" -t rust

# Search with multiple tags (matches ANY)
motlie fulltext -p /data/fulltext-index search nodes "language" -t rust -t python

# Limit results
motlie fulltext -p /data/fulltext-index search nodes "programming" -l 5
```

#### Search Edges

```bash
motlie fulltext -p <index_dir> search edges [OPTIONS] <query>
```

**Arguments:**
- `<query>` - The search query string

**Options:**
- `-l, --limit <n>` - Maximum number of results (default: 10)
- `-f, --fuzzy-level <level>` - Fuzzy matching: `none`, `low`, `medium` (default: none)
- `-t, --tags <tag>` - Filter by tags (can be repeated, matches ANY tag)
- `-o, --format <format>` - Output format: `tsv` or `table` (default: tsv)

**Output format (TSV):**
```
SCORE    SRC_ID    DST_ID    EDGE_NAME    FRAGMENT_TS    SNIPPET
```

**Example:**
```bash
# Search for edges by name
motlie fulltext -p /data/fulltext-index search edges "influences"

# Search for edges by content in fragments
motlie fulltext -p /data/fulltext-index search edges "Django Flask"

# Search with tag filter
motlie fulltext -p /data/fulltext-index search edges "relationship" -t important
```

#### Search Facets

Get document counts by facet (document type, tags, validity).

```bash
motlie fulltext -p <index_dir> search facets [OPTIONS]
```

**Options:**
- `-d, --doc-type-filter <type>` - Filter by document types (can be repeated)
  - Valid values: `nodes`, `edges`, `node_fragments`, `edge_fragments`
- `-l, --tags-limit <n>` - Maximum number of tags to return (default: 50)
- `-o, --format <format>` - Output format: `tsv` or `table` (default: tsv)

**Output format (TSV):**
```
CATEGORY    NAME    COUNT
```

**Example:**
```bash
# Get all facet counts
motlie fulltext -p /data/fulltext-index search facets

# Get facet counts for nodes only
motlie fulltext -p /data/fulltext-index search facets -d nodes

# Get facet counts for edges and edge fragments
motlie fulltext -p /data/fulltext-index search facets -d edges -d edge_fragments

# Limit tag results
motlie fulltext -p /data/fulltext-index search facets -l 20
```

## Fuzzy Matching

Fuzzy matching allows finding results even with typos in the query:

| Level | Description | Example |
|-------|-------------|---------|
| `none` | Exact match only (default) | "Python" matches "Python" |
| `low` | 1 character edit distance | "Pythn" matches "Python" |
| `medium` | 2 character edit distance | "Pyton" matches "Python" |

## Tag Filtering

Tags are extracted from content using `#hashtag` syntax during indexing. When searching with tag filters:

- Multiple `-t` flags use OR logic (matches documents with ANY of the specified tags)
- Tags are case-sensitive
- Tags in content should use the format `#tagname` (alphanumeric, no spaces)

**Example content with tags:**
```markdown
# Rust Programming Language

Rust is a systems programming language.

#rust #systems #memory-safety
```

## Indexed Content

The fulltext index includes:

| Document Type | Indexed Fields |
|--------------|----------------|
| Nodes | Name, Summary |
| Node Fragments | Content (text extracted from DataUrl) |
| Edges | Name, Summary |
| Edge Fragments | Content (text extracted from DataUrl) |

## Output Formats

### TSV (Tab-Separated Values)

Default format, suitable for piping to other tools:

```bash
motlie fulltext -p /data/index search nodes "rust" | cut -f2  # Extract IDs only
```

### Table

Human-readable format with aligned columns:

```bash
motlie fulltext -p /data/index search nodes "rust" -o table
```

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| "Index directory is not empty" | Attempting to re-index | Use a new/empty directory |
| "Index directory does not exist" | Searching non-existent index | Run `index` command first |
| "Failed to execute index command" | Graph DB doesn't exist | Verify graph DB path |

## Tests

End-to-end integration tests are available in `bins/motlie/tests/fulltext_cli.rs`. These tests:

1. Insert test data directly into RocksDB (without fulltext indexing)
2. Run the `fulltext index` command to build the index
3. Run `fulltext search` commands and verify results match inserted data

### Running Tests

```bash
cargo test --test fulltext_cli
```

### Test Coverage

| Test | Description |
|------|-------------|
| `test_search_nodes_by_exact_name` | Verifies exact node name searches return correct IDs |
| `test_search_nodes_by_unique_content` | Verifies content-specific keywords find correct nodes |
| `test_search_nodes_no_false_positives` | Verifies non-existent terms return empty results |
| `test_search_edges_by_name` | Verifies edge name searches return correct (src, dst, name) tuples |
| `test_search_edges_by_fragment_content` | Verifies edge fragment content is searchable |
| `test_search_with_tag_filter` | Verifies tag filtering includes/excludes correct nodes |
| `test_search_with_fuzzy_matching` | Verifies typo correction with fuzzy matching |
| `test_facets_document_type_counts` | Verifies exact document type counts |
| `test_facets_tag_counts` | Verifies tag facets are returned |
| `test_reindex_prevention` | Verifies re-indexing to non-empty directory is prevented |

### Test Data

The tests use a sample dataset of programming languages:

**Nodes:**
- Rust (with #rust, #systems tags)
- Python (with #python, #scripting tags)
- JavaScript (with #javascript, #web tags)
- TypeScript (with #typescript, #web, #typed tags)

**Edges:**
- Rust → Python ("influences")
- JavaScript → TypeScript ("extends_to")
- Python → JavaScript ("competes_with")

**Fragments:**
- Rust node fragment with "ownership", "memory safety" content
- Python node fragment with "machine learning", "data science" content
- Competition edge fragment with "Django", "Pandas" content

This allows testing:
- Unique keyword searches (e.g., "ownership" only finds Rust)
- Tag filtering (e.g., #web finds JavaScript and TypeScript)
- Edge relationship verification (correct src/dst IDs)
- Facet counting (4 nodes, 3 edges, 2 node fragments, 1 edge fragment)
