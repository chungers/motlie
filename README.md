# Motlie

A high-performance Rust-based graph processing system with dual storage backends for both search indexing and persistent storage.

## Overview

Motlie provides a flexible architecture for processing graph mutations through multiple specialized consumers:

- **BM25 Consumer**: Full-text search indexing with configurable scoring parameters
- **RocksDB Consumer**: High-performance key-value storage for persistence

The system uses MPSC (multiple-producer, single-consumer) queues to efficiently process mutations across both storage backends simultaneously.

## Quick Start

The easiest way to get started is with the store example:

```bash
# Clone and build
git clone <repository-url>
cd motlie
cargo build

# Generate test data (~100 nodes, ~1000 edges)
python3 examples/store/generate_data.py --total-nodes 100 2>/dev/null > /tmp/test_data.csv

# Run the store example with generated data
cat /tmp/test_data.csv | cargo run --example store

# Verify data was written correctly to RocksDB
cargo run --example verify_store /tmp/test_data.csv
```

This generates a realistic graph with professional entities (people, companies, projects, artifacts), social connections, and creates a fully indexed RocksDB database. The verification tool compares the CSV input with the database contents to ensure all data was stored correctly.

## Architecture

The store example demonstrates consumer chaining, where mutations flow through multiple processors:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CSV Input     │───▶│  MutationWriter  │───▶│  Graph Consumer │
└─────────────────┘    └──────────────────┘    └────────┬────────┘
                                                         │
                                                         │ (forwards)
                                                         ▼
                                              ┌──────────────────────┐
                                              │  FullText Consumer   │
                                              │  (Search Index)      │
                                              └──────────────────────┘
                                                         │
                                                         ▼
                                              ┌──────────────────────┐
                                              │      RocksDB         │
                                              │  (Persistent Store)  │
                                              └──────────────────────┘
```

## Core Concepts

### Graph Elements
- **Vertices**: Nodes in the graph with unique IDs and names
- **Edges**: Connections between vertices with metadata
- **Fragments**: Text content associated with vertices or edges

### Mutations
- **AddVertex**: Create a new vertex
- **AddEdge**: Create a connection between vertices  
- **AddFragment**: Associate text content with a vertex/edge
- **Invalidate**: Remove or mark content as invalid

### Storage Backends
- **BM25**: Optimized for full-text search with TF-IDF scoring
- **RocksDB**: High-performance persistent key-value storage

## Workspace Structure

```
motlie/
├── libs/
│   ├── core/                    # Core data structures and utilities
│   ├── db/                      # Database abstraction and mutation processing
│   └── mcp/                     # MCP (Model Context Protocol) integration
├── examples/
│   └── store/
│       ├── main.rs              # CSV processor with consumer chaining
│       ├── verify_store.rs      # RocksDB verification tool
│       ├── test_input.csv       # Sample test data (3 nodes, 3 edges)
│       ├── generate_data.py     # Data generator for testing
│       ├── graph_viewer.html    # Interactive graph visualizer
│       └── README.md            # Detailed example documentation
└── README.md                    # This file
```

## CSV Format

The basic usage example supports two CSV formats:

### Node with Fragment (2 columns)
```csv
node_name,fragment_text
user1,Hello world from user1
```

### Edge with Fragment (4 columns)  
```csv
source_node,target_node,edge_name,edge_fragment
user1,user2,friendship,Connected as friends
```

## Running Examples

### Generate and Process Data
```bash
# Generate a dataset with 100 nodes
python3 examples/store/generate_data.py --total-nodes 100 2>/dev/null > /tmp/test_data.csv

# Process the data
cat /tmp/test_data.csv | cargo run --example store

# Verify data persistence and correctness
cargo run --example verify_store /tmp/test_data.csv
```

### With Logging
```bash
# See detailed processing logs
python3 examples/store/generate_data.py --total-nodes 50 2>/dev/null > /tmp/test_data.csv
RUST_LOG=info cat /tmp/test_data.csv | cargo run --example store
```

### Small Test Dataset
```bash
# Use the small test input file (3 nodes, 3 edges)
cat examples/store/test_input.csv | cargo run --example store
```

### Sample Output
```
Motlie CSV Processor - Demonstrating Graph → FullText Chaining
===================================================================

Sent vertex 'Helen Smith' with fragment (156 chars) to chain
Sent vertex 'Ivan Clark' with fragment (152 chars) to chain
...
Sent edge 'Helen Smith' -> 'TechStart Inc' (name: 'works_at', fragment: 89 chars) to chain
...

Processed 1063 lines from stdin
Created 99 unique nodes
Shutting down consumer chain...
All consumers shut down successfully
```

### Verification Output
```
Motlie Store Verifier
====================

CSV file: /tmp/test_data.csv
Database: /tmp/motlie_graph_db

📄 Parsing CSV file...
   Nodes: 99
   Edges: 964
   Total fragments: 1050

✓ Database opened successfully

🔍 Verifying Nodes...
   Expected: 99 nodes
   Found:    99 nodes
   ✓ Node count matches
   ✓ All expected nodes found in database

🔍 Verifying Edges...
   Expected: 964 edges
   Found:    964 edges
   ✓ Edge count matches

🔍 Verifying Fragments...
   Expected: at least 1050 fragments
   Found:    1063 fragments
   ✓ Fragment count OK (database may have additional fragments for implicit nodes)
   ✓ All expected fragments found in database

✅ All verification checks passed!
   The database contents match the CSV input.
```

## Development

### Prerequisites
- Rust 1.70+
- Cargo

### Building
```bash
# Build all workspace members
cargo build

# Run tests
cargo test

# Check all code
cargo check --workspace
```

### Adding New Consumers

The system is designed to be extensible. To add a new consumer:

1. Implement the `Processor` trait from `libs/db/src/mutation.rs`
2. Create specialized processing logic for each mutation type:
   - `process_add_vertex()` - Handle node creation
   - `process_add_edge()` - Handle edge creation
   - `process_add_fragment()` - Handle fragment content
   - `process_invalidate()` - Handle deletions/invalidations
3. Use the generic `Consumer<P>` with your processor
4. Spawn your consumer in a chain or standalone

See `libs/db/src/graph.rs` and `libs/db/src/fulltext.rs` for examples.

### RocksDB Schema and Column Families

The Graph consumer writes to 5 column families in RocksDB:

- **nodes**: Node records keyed by node ID
- **edges**: Edge records keyed by edge ID
- **fragments**: Fragment content keyed by (ID, timestamp)
- **forward_edges**: Edges indexed by (source → destination, name) for efficient traversal
- **reverse_edges**: Edges indexed by (destination ← source, name) for bidirectional queries

All data is serialized using MessagePack (`rmp-serde`) for compact storage.

## Features

- ✅ **Consumer Chaining**: Mutations flow through multiple processors in sequence
- ✅ **Persistent Storage**: RocksDB with 5 specialized column families for efficient queries
- ✅ **Type-Safe IDs**: ULID-based identifiers with timestamp information
- ✅ **Bidirectional Edges**: Forward and reverse edge indices for graph traversal
- ✅ **Graceful Shutdown**: Ensures all mutations are processed before exit
- ✅ **Rich Logging**: Detailed processing logs for debugging and monitoring
- ✅ **Error Handling**: Comprehensive error handling with context
- ✅ **Flexible Input**: CSV format supports mixed node and edge creation
- ✅ **Verification Tool**: Standalone tool to inspect RocksDB data
- ✅ **Full Test Coverage**: 101+ tests covering all mutation types and storage operations

## License

Licensed under either of:

- Apache License, Version 2.0
- MIT License

at your option.