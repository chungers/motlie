# Motlie

A high-performance Rust-based graph processing system with dual storage backends for both search indexing and persistent storage.

## Overview

Motlie provides a flexible architecture for processing graph mutations through multiple specialized consumers:

- **BM25 Consumer**: Full-text search indexing with configurable scoring parameters
- **RocksDB Consumer**: High-performance key-value storage for persistence

The system uses MPSC (multiple-producer, single-consumer) queues to efficiently process mutations across both storage backends simultaneously.

## Quick Start

The easiest way to get started is with the basic usage example:

```bash
# Clone and build
git clone <repository-url>
cd motlie
cargo build

# Run the CSV processor example
echo "user1,Hello world from user1
user2,Greetings from user2
user1,user2,friendship,Connected as friends" | cargo run --example basic-usage

# Or process sample data
cargo run --example basic-usage < examples/sample_data.csv
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CSV Input     │───▶│  MutationWriter  │───▶│  MPSC Channels  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                               ┌─────────▼─────────┐
                                               │                   │
                                               ▼                   ▼
                                    ┌─────────────────┐ ┌─────────────────┐
                                    │ BM25 Consumer   │ │ Rocks Consumer  │
                                    │ (Search Index)  │ │ (Key-Value DB)  │
                                    └─────────────────┘ └─────────────────┘
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
│   ├── core/           # Core data structures and utilities
│   ├── db/             # Database abstraction and mutation processing
│   └── mcp/            # MCP (Model Context Protocol) integration
├── examples/
│   ├── basic-usage.rs  # CSV processor example
│   ├── sample_data.csv # Sample data for testing
│   └── README.md       # Example documentation
└── README.md           # This file
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
# Process simple data
echo "alice,Software engineer
bob,Data scientist
alice,bob,Collaborators" | cargo run --example basic-usage
```

### With Logging
```bash
# See detailed processing logs
RUST_LOG=info cargo run --example basic-usage < examples/sample_data.csv
```

### Sample Output
```
Motlie CSV Processor
Reading CSV from stdin...

Processed vertex 'alice' with fragment (17 chars)
Processed vertex 'bob' with fragment (14 chars)
Processed edge 'alice' -> 'bob' (name: 'collaboration', fragment: 13 chars)

Processed 3 lines from stdin
Created 2 unique nodes
Shutting down consumers...
All consumers shut down successfully
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

1. Implement the `Processor` trait
2. Create specialized processing logic for each mutation type
3. Use the generic `Consumer<P>` with your processor
4. Spawn your consumer alongside existing ones

See `libs/db/src/bm25.rs` and `libs/db/src/rocks.rs` for examples.

## Features

- ✅ **Concurrent Processing**: Multiple consumers process same mutations simultaneously
- ✅ **Type-Safe IDs**: UUID-based identifiers prevent mixing different entity types  
- ✅ **Configurable Batching**: Efficient batch processing with timeout controls
- ✅ **Graceful Shutdown**: Ensures all mutations are processed before exit
- ✅ **Rich Logging**: Detailed processing logs for debugging and monitoring
- ✅ **Error Handling**: Comprehensive error handling with context
- ✅ **Flexible Input**: CSV format supports mixed vertex and edge creation

## License

Licensed under either of:

- Apache License, Version 2.0
- MIT License

at your option.