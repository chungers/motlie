# Motlie MCP Server Library

MCP (Model Context Protocol) server implementation for the Motlie graph database, built with the [rmcp SDK](https://github.com/anthropics/rmcp).

## Overview

This library exposes the Motlie graph database as an MCP server, allowing AI assistants (Claude, etc.) to interact with the graph through standardized MCP tools. The server supports both stdio and HTTP transports.

## Features

- **15 MCP Tools**: 7 mutation operations + 8 query operations
- **Two Transports**: stdio (for local integration) and HTTP with SSE (for remote access)
- **Lazy Database Initialization**: Fast server startup with on-demand database connection
- **Fulltext Search**: Node and edge search powered by Tantivy
- **Temporal Queries**: Time-travel queries with `reference_ts_millis`
- **Automatic Schema Generation**: JSON schemas generated from Rust types via `schemars`

## Quick Start

### Using the Example Server

```bash
# Stdio transport (for Claude Desktop, Claude Code, Cursor, etc.)
cargo run --example mcp -- --db-path /path/to/db --transport stdio

# HTTP transport (for remote access)
cargo run --example mcp -- --db-path /path/to/db --transport http --port 8080
```

### Library Usage

```rust
use motlie_db::{Storage, StorageConfig};
use motlie_mcp::{LazyDatabase, MotlieMcpServer, stdio, ServiceExt};
use std::sync::Arc;
use std::time::Duration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let db_path = std::path::PathBuf::from("/path/to/db");

    // Create lazy database initializer
    let lazy_db = Arc::new(LazyDatabase::new(Box::new(move || {
        let storage = Storage::readwrite(&db_path);
        let handles = storage.ready(StorageConfig::default())?;
        Ok((handles.writer_clone(), handles.reader_clone()))
    })));

    // Create MCP server
    let server = MotlieMcpServer::new(lazy_db, Duration::from_secs(5));

    // Run with stdio transport
    server.serve(stdio()).await?;
    Ok(())
}
```

## MCP Tools

### Mutation Tools (7)

| Tool | Description |
|------|-------------|
| `add_node` | Create a new node in the graph with name and optional temporal validity |
| `add_edge` | Create an edge between two nodes with optional weight and temporal validity |
| `add_node_fragment` | Add a text content fragment to an existing node |
| `add_edge_fragment` | Add a text content fragment to an existing edge |
| `update_node_valid_range` | Update the temporal validity range of a node |
| `update_edge_valid_range` | Update the temporal validity range of an edge |
| `update_edge_weight` | Update the weight of an edge for graph algorithms |

### Query Tools (8)

| Tool | Description |
|------|-------------|
| `query_node_by_id` | Retrieve a node by its ID |
| `query_edge` | Retrieve an edge by its source, destination, and name |
| `query_outgoing_edges` | Get all outgoing edges from a node |
| `query_incoming_edges` | Get all incoming edges to a node |
| `query_nodes_by_name` | Search for nodes by name (fulltext search) |
| `query_edges_by_name` | Search for edges by name (fulltext search) |
| `query_node_fragments` | Get content fragments for a node within a time range |
| `query_edge_fragments` | Get content fragments for an edge within a time range |

## JSON Schema Reference

### Common Types

#### TemporalRangeParam

Used for specifying temporal validity ranges:

```json
{
  "type": "object",
  "properties": {
    "valid_since": {
      "type": "integer",
      "format": "uint64",
      "description": "Start of validity period (milliseconds since Unix epoch)"
    },
    "valid_until": {
      "type": "integer",
      "format": "uint64",
      "description": "End of validity period (milliseconds since Unix epoch)"
    }
  },
  "required": ["valid_since", "valid_until"]
}
```

### Mutation Tool Schemas

#### add_node

```json
{
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "description": "Human-readable node name"
    },
    "summary": {
      "type": "string",
      "description": "Node summary/description"
    },
    "ts_millis": {
      "type": "integer",
      "format": "uint64",
      "description": "Optional timestamp (defaults to current time)"
    },
    "temporal_range": {
      "$ref": "#/definitions/TemporalRangeParam",
      "description": "Optional temporal validity range"
    }
  },
  "required": ["name", "summary"]
}
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully added node 'Alice' with ID 01HGW2N7E1H9XVKVBP97ZTQFGR",
  "node_id": "01HGW2N7E1H9XVKVBP97ZTQFGR",
  "node_name": "Alice"
}
```

#### add_edge

```json
{
  "type": "object",
  "properties": {
    "source_node_id": {
      "type": "string",
      "description": "Source node UUID (base32-encoded ULID)"
    },
    "target_node_id": {
      "type": "string",
      "description": "Target node UUID (base32-encoded ULID)"
    },
    "name": {
      "type": "string",
      "description": "Edge name/type (e.g., 'knows', 'follows')"
    },
    "summary": {
      "type": "string",
      "description": "Edge summary/description"
    },
    "weight": {
      "type": "number",
      "format": "double",
      "description": "Optional edge weight for graph algorithms"
    },
    "ts_millis": {
      "type": "integer",
      "format": "uint64",
      "description": "Optional timestamp (defaults to current time)"
    },
    "temporal_range": {
      "$ref": "#/definitions/TemporalRangeParam"
    }
  },
  "required": ["source_node_id", "target_node_id", "name", "summary"]
}
```

#### add_node_fragment

```json
{
  "type": "object",
  "properties": {
    "id": {
      "type": "string",
      "description": "Node UUID"
    },
    "content": {
      "type": "string",
      "description": "Fragment content (text)"
    },
    "ts_millis": {
      "type": "integer",
      "format": "uint64"
    },
    "temporal_range": {
      "$ref": "#/definitions/TemporalRangeParam"
    }
  },
  "required": ["id", "content"]
}
```

#### add_edge_fragment

```json
{
  "type": "object",
  "properties": {
    "src_id": {
      "type": "string",
      "description": "Source node UUID"
    },
    "dst_id": {
      "type": "string",
      "description": "Destination node UUID"
    },
    "edge_name": {
      "type": "string",
      "description": "Edge name"
    },
    "content": {
      "type": "string",
      "description": "Fragment content (text)"
    },
    "ts_millis": {
      "type": "integer",
      "format": "uint64"
    },
    "temporal_range": {
      "$ref": "#/definitions/TemporalRangeParam"
    }
  },
  "required": ["src_id", "dst_id", "edge_name", "content"]
}
```

#### update_node_valid_range

```json
{
  "type": "object",
  "properties": {
    "id": {
      "type": "string",
      "description": "Node UUID"
    },
    "temporal_range": {
      "$ref": "#/definitions/TemporalRangeParam",
      "description": "New temporal validity range"
    },
    "reason": {
      "type": "string",
      "description": "Reason for the update"
    }
  },
  "required": ["id", "temporal_range", "reason"]
}
```

#### update_edge_valid_range

```json
{
  "type": "object",
  "properties": {
    "src_id": { "type": "string" },
    "dst_id": { "type": "string" },
    "name": { "type": "string" },
    "temporal_range": { "$ref": "#/definitions/TemporalRangeParam" },
    "reason": { "type": "string" }
  },
  "required": ["src_id", "dst_id", "name", "temporal_range", "reason"]
}
```

#### update_edge_weight

```json
{
  "type": "object",
  "properties": {
    "src_id": { "type": "string" },
    "dst_id": { "type": "string" },
    "name": { "type": "string" },
    "weight": {
      "type": "number",
      "format": "double",
      "description": "New weight value"
    }
  },
  "required": ["src_id", "dst_id", "name", "weight"]
}
```

### Query Tool Schemas

#### query_node_by_id

```json
{
  "type": "object",
  "properties": {
    "id": {
      "type": "string",
      "description": "Node UUID"
    },
    "reference_ts_millis": {
      "type": "integer",
      "format": "uint64",
      "description": "Reference timestamp for temporal validity checks"
    }
  },
  "required": ["id"]
}
```

**Response:**
```json
{
  "node_id": "01HGW2N7E1H9XVKVBP97ZTQFGR",
  "name": "Alice",
  "summary": "Software engineer at Acme Corp"
}
```

#### query_edge

```json
{
  "type": "object",
  "properties": {
    "source_id": { "type": "string" },
    "dest_id": { "type": "string" },
    "name": { "type": "string" },
    "reference_ts_millis": { "type": "integer", "format": "uint64" }
  },
  "required": ["source_id", "dest_id", "name"]
}
```

**Response:**
```json
{
  "source_id": "01HGW2N7E1H9XVKVBP97ZTQFGR",
  "dest_id": "01HGW2N8F2J0YWLWCQ08AURGHT",
  "name": "knows",
  "summary": "Met at tech conference 2023",
  "weight": 0.8
}
```

#### query_outgoing_edges / query_incoming_edges

```json
{
  "type": "object",
  "properties": {
    "id": { "type": "string" },
    "reference_ts_millis": { "type": "integer", "format": "uint64" }
  },
  "required": ["id"]
}
```

**Response:**
```json
{
  "node_id": "01HGW2N7E1H9XVKVBP97ZTQFGR",
  "edges": [
    { "target_id": "01HGW2N8F2J0YWLWCQ08AURGHT", "name": "knows", "weight": 0.8 },
    { "target_id": "01HGW2N9G3K1ZXMXDR19BVSIJU", "name": "follows", "weight": null }
  ],
  "count": 2
}
```

#### query_nodes_by_name / query_edges_by_name

```json
{
  "type": "object",
  "properties": {
    "name": {
      "type": "string",
      "description": "Name or prefix to search for (fulltext search)"
    },
    "limit": {
      "type": "integer",
      "description": "Maximum number of results (default: 100)"
    },
    "reference_ts_millis": { "type": "integer", "format": "uint64" }
  },
  "required": ["name"]
}
```

**Response (nodes):**
```json
{
  "search_name": "Alice",
  "nodes": [
    { "name": "Alice Smith", "id": "01HGW2N7E1H9XVKVBP97ZTQFGR" },
    { "name": "Alice Johnson", "id": "01HGW2N8F2J0YWLWCQ08AURGHT" }
  ],
  "count": 2
}
```

#### query_node_fragments / query_edge_fragments

```json
{
  "type": "object",
  "properties": {
    "id": { "type": "string" },
    "start_ts_millis": {
      "type": "integer",
      "format": "uint64",
      "description": "Start of time range (optional, unbounded if not specified)"
    },
    "end_ts_millis": {
      "type": "integer",
      "format": "uint64",
      "description": "End of time range (optional, unbounded if not specified)"
    },
    "reference_ts_millis": { "type": "integer", "format": "uint64" }
  },
  "required": ["id"]
}
```

**Response:**
```json
{
  "node_id": "01HGW2N7E1H9XVKVBP97ZTQFGR",
  "fragments": [
    { "timestamp_millis": 1702000000000, "content": "Promoted to Senior Engineer" },
    { "timestamp_millis": 1703000000000, "content": "Started leading the backend team" }
  ],
  "count": 2
}
```

## Transport Configuration

### Stdio Transport

Standard input/output for local process communication. Used by Claude Desktop, Claude Code, and Cursor.

```rust
use motlie_mcp::{stdio, ServiceExt};

server.serve(stdio()).await?;
```

### HTTP Transport

HTTP with Streamable HTTP protocol and Server-Sent Events (SSE) for long-lived connections.

```rust
use motlie_mcp::{serve_http, HttpConfig};

let config = HttpConfig::new("127.0.0.1:8080".parse()?)
    .with_mcp_path("/mcp")
    .with_sse_keep_alive(Some(Duration::from_secs(30)))
    .with_stateful_mode(true);

serve_http(server, config).await?;
```

#### HttpConfig Options

| Option | Default | Description |
|--------|---------|-------------|
| `addr` | `127.0.0.1:8080` | Address to bind to |
| `mcp_path` | `/mcp` | MCP endpoint path |
| `sse_keep_alive` | 30 seconds | SSE keep-alive interval (None to disable) |
| `stateful_mode` | `true` | Maintain session state across requests |

## Graph Modeling Guidelines

The server provides instructions to AI assistants on how to properly model data:

1. **Nodes are entities**: People, places, things, events, concepts
2. **Edges are relationships**: Actions, connections between entities
3. **Fragments are properties**: Attributes, state changes, context about entities/relationships
4. **No deletion**: Use temporal validity ranges to "invalidate" nodes/edges
5. **Avoid duplication**: Query by name before creating new nodes

### Example: Modeling "Johnny loves ice cream in summer"

**Good approach:**
- Node: Johnny (fragment: "7 years old")
- Node: Ice Cream
- Edge: Johnny --[loves]--> Ice Cream (fragment: "in summer")

**Bad approach:**
- Node: Johnny (fragment: "loves ice cream in summer") - Mixing concepts!

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         MCP Client                  │
                    │  (Claude Desktop, Claude Code, etc) │
                    └─────────────┬───────────────────────┘
                                  │ MCP Protocol
                    ┌─────────────▼───────────────────────┐
                    │      MotlieMcpServer (rmcp)         │
                    │  ┌───────────┐ ┌─────────────────┐  │
                    │  │ Mutations │ │    Queries      │  │
                    │  │  7 tools  │ │    8 tools      │  │
                    │  └─────┬─────┘ └────────┬────────┘  │
                    └────────┼────────────────┼───────────┘
                             │                │
              ┌──────────────▼──┐      ┌──────▼──────────────┐
              │     Writer      │      │      Reader         │
              │  (mutations)    │      │  (queries)          │
              └────────┬────────┘      └──────────┬──────────┘
                       │                          │
              ┌────────▼──────────────────────────▼────────┐
              │              Unified Storage               │
              │  ┌─────────────────┐ ┌──────────────────┐  │
              │  │  Graph (RocksDB)│ │ Fulltext(Tantivy)│  │
              │  └─────────────────┘ └──────────────────┘  │
              └────────────────────────────────────────────┘
```

## Design Decisions

### Text-Only Fragments

Fragment content is intentionally text-only to prevent context window bloat when AI agents interact with the graph. Binary content should be stored externally and referenced by URL.

### Lazy Database Initialization

The server starts immediately and initializes the database on first tool use. This enables fast startup for MCP clients that may not use all tools.

### Fulltext Search for Name Queries

`query_nodes_by_name` and `query_edges_by_name` use Tantivy fulltext search rather than prefix matching, providing better search results.

## Documentation

- [DESIGN.md](docs/DESIGN.md) - Detailed architecture and design decisions
- [motlie-db README](../db/README.md) - Core database documentation
- [MCP Specification](https://spec.modelcontextprotocol.io/) - Official MCP protocol specification

## License

MIT
