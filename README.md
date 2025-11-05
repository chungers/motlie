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

# Run the store example with test data
cat examples/store/test_input.csv | cargo run --example store

# Verify data was written to RocksDB
cargo run --example verify_store
```

The test input (`examples/store/test_input.csv`) includes 3 nodes (alice, bob, charlie) and 3 edges (knows, collaborates, mentors) with associated fragments.

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

### Basic Processing
```bash
# Process test data
cat examples/store/test_input.csv | cargo run --example store
```

### With Logging
```bash
# See detailed processing logs
RUST_LOG=info cat examples/store/test_input.csv | cargo run --example store
```

### Verify Data Persistence
```bash
# After running store example, verify data was written to RocksDB
cargo run --example verify_store
```

### Sample Output
```
Motlie CSV Processor - Demonstrating Graph → FullText Chaining
===================================================================

Sent vertex 'alice' with fragment (52 chars) to chain
Sent vertex 'bob' with fragment (53 chars) to chain
Sent vertex 'charlie' with fragment (56 chars) to chain
Sent edge 'alice' -> 'bob' (name: 'knows', fragment: 76 chars) to chain

Processed 6 lines from stdin
Created 3 unique nodes
Shutting down consumer chain...
All consumers shut down successfully
```

### Verification Output
```
Motlie Store Verifier
====================
Reading from: /tmp/motlie_graph_db

✓ Database opened successfully

📦 Nodes Column Family:
  Total: 3 nodes

📦 Fragments Column Family:
  Total: 6 fragments

✓ Verification complete!
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