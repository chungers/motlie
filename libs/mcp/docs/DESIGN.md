# Motlie MCP Server Design

## Overview

This document describes the design and implementation of the Model Context Protocol (MCP) server for the Motlie graph database. The MCP server exposes the graph mutation and query APIs as MCP tools, enabling AI assistants to interact with the graph database through standardized protocols.

## Architecture

### Component Structure

```
motlie/
├── libs/mcp/                    # MCP server library
│   ├── src/
│   │   ├── lib.rs              # Core MCP server implementation
│   │   ├── auth.rs             # Authentication middleware
│   │   ├── tools/
│   │   │   ├── mutations.rs    # Mutation tool handlers
│   │   │   └── queries.rs      # Query tool handlers
│   │   └── types.rs            # MCP parameter types
│   └── docs/
│       └── DESIGN.md           # This document
└── examples/mcp/
    └── main.rs                 # Server entry point and CLI
```

### Key Components

1. **MotlieMcpServer**: Core server struct that wraps database access and authentication
2. **Tool Handlers**: Individual MCP tools for each mutation and query operation
3. **Authentication Middleware**: Token-based authentication layer
4. **Transport Layer**: Support for stdio and HTTP transports via rmcp SDK

## Dependencies

### External Crates

- **rust-mcp-sdk** (v0.7): Official Rust SDK for Model Context Protocol
- **rust-mcp-schema** (v0.7): MCP protocol schemas and types
- **schemars** (v0.8): JSON Schema generation from Rust types
- **tokio**: Async runtime for concurrent operations
- **serde/serde_json**: Serialization for parameters
- **clap**: Command-line argument parsing (example only)
- **anyhow**: Error handling
- **async-trait**: Async trait support for ServerHandler

### Internal Dependencies

- **motlie-db**: Core graph database library providing:
  - `Writer` and `Reader` for async mutation/query execution
  - `Storage` for database access
  - `Graph` processor for queries
  - All mutation types (`AddNode`, `AddEdge`, etc.)
  - All query types (`NodeById`, `OutgoingEdges`, etc.)

## Core Design Patterns

### 1. ServerHandler Pattern

The MCP server implements the `ServerHandler` trait from rust-mcp-sdk to handle MCP protocol messages:

```rust
use async_trait::async_trait;
use rust_mcp_sdk::schema::{
    CallToolRequest, CallToolResult, ListToolsRequest, ListToolsResult, RpcError,
    schema_utils::CallToolError,
};
use rust_mcp_sdk::{mcp_server::ServerHandler, McpServer};

#[async_trait]
impl ServerHandler for MotlieMcpServer {
    async fn handle_list_tools_request(
        &self,
        request: ListToolsRequest,
        runtime: Arc<dyn McpServer>,
    ) -> Result<ListToolsResult, RpcError> {
        // Return list of available tools
    }

    async fn handle_call_tool_request(
        &self,
        request: CallToolRequest,
        runtime: Arc<dyn McpServer>,
    ) -> Result<CallToolResult, CallToolError> {
        // Route to appropriate tool handler based on request.params.name
    }
}
```

Each tool handler:
- Is a method within the `MotlieMcpServer` impl block
- Accepts parsed parameters from the request
- Returns `Result<CallToolResult, CallToolError>`
- Manually generates tool definitions for the tools list

### 2. Authentication Flow

```
Client Request
    ↓
Extract auth_token from params
    ↓
self.authenticate(token) → Result<(), McpError>
    ↓
Execute core mutation/query logic
    ↓
Return result or error
```

Authentication is performed at the tool handler level, before any database operations.

### 3. Database Access Pattern

The server maintains separate channels for mutations and queries:

```rust
pub struct MotlieMcpServer {
    writer: Writer,           // For mutations (async channel to graph consumer)
    reader: Reader,           // For queries (async channel to query consumer)
    storage: Arc<Storage>,    // Readonly storage reference
    auth_token: Option<String>, // Expected authentication token
}
```

**Mutations** flow through:
1. Tool handler validates parameters and auth
2. Constructs mutation struct (e.g., `AddNode`)
3. Calls `.run(&self.writer).await`
4. Background consumer executes against TransactionDB
5. Returns success/error to client

**Queries** flow through:
1. Tool handler validates parameters and auth
2. Constructs query struct (e.g., `NodeById`)
3. Calls `.run(&self.reader, timeout).await`
4. Background consumer executes against Storage
5. Returns data to client

### 4. Error Handling

All errors are mapped to MCP-compatible errors using `RpcError` and `CallToolError`:

```rust
mutation.run(&self.writer)
    .await
    .map_err(|e| CallToolError::new(
        RpcError::internal_error()
            .with_message(format!("Mutation failed: {}", e))
    ))?;
```

Error types:
- **Authentication errors**: `CallToolError::new(anyhow::anyhow!("Invalid token"))`
- **Parameter validation errors**: `CallToolError::new(anyhow::anyhow!(message))`
- **Database errors**: `CallToolError::new(RpcError::internal_error().with_message(message))`

## Tool Definitions

### Mutation Tools

Each mutation type in `motlie_db::mutation` gets a corresponding MCP tool:

| MCP Tool | Mutation Type | Description |
|----------|---------------|-------------|
| `add_node` | `AddNode` | Create a new node with ID, name, and optional temporal range |
| `add_edge` | `AddEdge` | Create an edge between two nodes with optional weight |
| `add_node_fragment` | `AddNodeFragment` | Add content fragment to a node |
| `add_edge_fragment` | `AddEdgeFragment` | Add content fragment to an edge |
| `update_node_valid_range` | `UpdateNodeValidSinceUntil` | Update temporal validity of a node |
| `update_edge_valid_range` | `UpdateEdgeValidSinceUntil` | Update temporal validity of an edge |
| `update_edge_weight` | `UpdateEdgeWeight` | Update the weight of an edge |

### Query Tools

Each query type in `motlie_db::query` gets a corresponding MCP tool:

| MCP Tool | Query Type | Description |
|----------|------------|-------------|
| `query_node_by_id` | `NodeById` | Retrieve node by UUID |
| `query_edge_by_src_dst_name` | `EdgeSummaryBySrcDstName` | Retrieve edge by endpoints and name |
| `query_outgoing_edges` | `OutgoingEdges` | Get all outgoing edges from a node |
| `query_incoming_edges` | `IncomingEdges` | Get all incoming edges to a node |
| `query_nodes_by_name` | `NodesByName` | Search nodes by name prefix |
| `query_edges_by_name` | `EdgesByName` | Search edges by name prefix |
| `query_node_fragments` | `NodeFragmentsByIdTimeRange` | Get node fragments in time range |

## Parameter Types

All tool parameters follow a consistent pattern:

```rust
#[derive(Debug, Deserialize, Serialize)]
pub struct AddNodeParams {
    // Authentication (present in all tools)
    pub auth_token: Option<String>,

    // Operation-specific fields
    pub id: String,                           // UUID as base32 string
    pub name: String,
    pub ts_millis: Option<u64>,               // Defaults to current time
    pub temporal_range: Option<TemporalRangeParam>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TemporalRangeParam {
    pub valid_since: u64,
    pub valid_until: u64,
}
```

Key design decisions:
- **String UUIDs**: IDs are passed as base32-encoded strings for JSON compatibility
- **Optional timestamps**: Default to current time if not provided
- **Nested structures**: Complex types like temporal ranges use dedicated structs
- **Consistent auth**: All tools accept `auth_token` parameter

## JSON Schema Generation

The MCP server uses **schemars** for automatic JSON schema generation from Rust types. Tool definitions are created manually but reference the schemars-generated schemas.

### Automatic Schema Derivation

All parameter types must derive the `JsonSchema` trait:

```rust
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct AddNodeParams {
    pub auth_token: Option<String>,
    pub id: String,
    pub name: String,
    pub ts_millis: Option<u64>,
    pub temporal_range: Option<TemporalRangeParam>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct TemporalRangeParam {
    pub valid_since: u64,
    pub valid_until: u64,
}
```

**Required derives**:
- `Serialize` / `Deserialize`: For JSON serialization (serde)
- `JsonSchema`: For JSON schema generation (schemars)
- `Debug`: For logging and error messages

### Schema Generation Process

Tool definitions are created manually in the `handle_list_tools_request` method:

1. **Schema generation**: Call `schema_for!(ParamType)` to get JSON schema
2. **Tool definition**: Create `Tool` struct with name, description, and schema
3. **Tools list**: Return all tools in `ListToolsResult`
4. **Validation**: MCP clients use the schema to validate requests before sending

Example of generated schema for `AddNodeParams`:

```json
{
  "type": "object",
  "properties": {
    "auth_token": {
      "type": ["string", "null"],
      "description": "Authentication token"
    },
    "id": {
      "type": "string",
      "description": "UUID as base32-encoded string"
    },
    "name": {
      "type": "string",
      "description": "Name of the node"
    },
    "ts_millis": {
      "type": ["integer", "null"],
      "description": "Timestamp in milliseconds since Unix epoch"
    },
    "temporal_range": {
      "anyOf": [
        { "$ref": "#/definitions/TemporalRangeParam" },
        { "type": "null" }
      ]
    }
  },
  "required": ["id", "name"],
  "definitions": {
    "TemporalRangeParam": {
      "type": "object",
      "properties": {
        "valid_since": { "type": "integer" },
        "valid_until": { "type": "integer" }
      },
      "required": ["valid_since", "valid_until"]
    }
  }
}
```

### Schema Annotations

Schemars supports attributes to customize schema generation:

```rust
use schemars::JsonSchema;

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct AddNodeParams {
    /// Authentication token for secure access
    #[schemars(description = "Bearer token for authentication")]
    pub auth_token: Option<String>,

    /// Node UUID in base32 format
    #[schemars(
        description = "Unique identifier for the node (base32-encoded ULID)",
        example = "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    )]
    pub id: String,

    /// Human-readable node name
    #[schemars(
        description = "Human-readable name for the node",
        min_length = 1,
        max_length = 256
    )]
    pub name: String,

    /// Optional timestamp (defaults to current time)
    #[schemars(description = "Timestamp in milliseconds since Unix epoch")]
    pub ts_millis: Option<u64>,

    /// Optional temporal validity range
    pub temporal_range: Option<TemporalRangeParam>,
}
```

**Available annotations**:
- `description`: Field documentation for clients
- `example`: Example values for documentation
- `min_length` / `max_length`: String length constraints
- `minimum` / `maximum`: Numeric range constraints
- `pattern`: Regex pattern for string validation
- `default`: Default value if field is missing

### Benefits of Schemars Integration

1. **Type safety**: Schema stays in sync with Rust types (compile-time checked)
2. **Zero boilerplate**: No manual JSON schema writing
3. **Client validation**: MCP clients validate requests before sending
4. **Documentation**: Schemas serve as API documentation
5. **IDE support**: Editors can autocomplete based on schema

### Schema Validation Flow

```
Rust Type Definition
    ↓
derive(JsonSchema)
    ↓
#[tool] macro processes Parameters<T>
    ↓
T::json_schema() generates schema
    ↓
MCP Tool Definition includes schema
    ↓
Client receives schema via tools/list
    ↓
Client validates parameters before sending
    ↓
Server validates again on receipt
```

### Custom Schema Types

For complex domain types (like `Id`, `DataUrl`, `TimestampMilli`), we implement `JsonSchema` manually:

```rust
impl JsonSchema for Id {
    fn schema_name() -> String {
        "NodeId".to_string()
    }

    fn json_schema(gen: &mut schemars::gen::SchemaGenerator) -> schemars::schema::Schema {
        let mut schema = schemars::schema::SchemaObject::default();
        schema.instance_type = Some(schemars::schema::InstanceType::String.into());
        schema.metadata = Some(Box::new(schemars::schema::Metadata {
            description: Some("Base32-encoded ULID identifier".to_string()),
            examples: vec![serde_json::json!("01ARZ3NDEKTSV4RRFFQ69G5FAV")],
            ..Default::default()
        }));
        schema.string = Some(Box::new(schemars::schema::StringValidation {
            pattern: Some("^[0-9A-HJKMNP-TV-Z]{26}$".to_string()),
            ..Default::default()
        }));
        schema.into()
    }
}
```

This approach allows:
- Validation of base32 format via regex pattern
- Better error messages for invalid IDs
- Documentation of expected format

### Schema Evolution

When adding new fields to parameter types:

**Backward compatible** (safe):
```rust
pub struct AddNodeParams {
    pub id: String,
    pub name: String,
    pub new_optional_field: Option<String>, // Safe: optional field
}
```

**Breaking changes** (requires version bump):
```rust
pub struct AddNodeParams {
    pub id: String,
    pub name: String,
    pub new_required_field: String, // Breaking: required field
}
```

**Best practice**: Always use `Option<T>` for new fields to maintain compatibility.

## Transport Support

The server supports two transport mechanisms via the rmcp SDK:

### 1. Standard I/O (stdio)

**Use case**: Running as a child process, communication via stdin/stdout

```bash
cargo run --example mcp -- --db-path /tmp/motlie.db --transport stdio
```

**Implementation**:
```rust
use rmcp::transport::stdio;

let service = server.serve(stdio()).await?;
service.waiting().await?;
```

**Characteristics**:
- Bidirectional JSON-RPC over stdin/stdout
- Suitable for local process integration
- Used by Claude Code, Cursor, and other MCP clients

### 2. HTTP with Server-Sent Events (SSE)

**Note**: HTTP/SSE transport is planned for future implementation. The rust-mcp-sdk supports it but requires additional configuration not yet implemented in the example server.

**Future implementation**:
```bash
cargo run --example mcp -- \
    --db-path /tmp/motlie.db \
    --transport http \
    --port 8080
```

**Characteristics**:
- HTTP-based JSON-RPC communication
- Supports remote clients
- SSE for server-to-client push

## Authentication Design

### Token-Based Authentication

The server implements a simple but extensible token-based authentication system:

```rust
impl MotlieMcpServer {
    fn authenticate(&self, provided_token: Option<&str>) -> Result<(), McpError> {
        match (&self.auth_token, provided_token) {
            (Some(expected), Some(provided)) if expected == provided => Ok(()),
            (None, _) => Ok(()), // No auth required
            _ => Err(McpError::invalid_params("Invalid or missing authentication token")),
        }
    }
}
```

### Authentication Flow

1. **Server Initialization**: Optional `--auth-token` CLI argument sets expected token
2. **Client Request**: Each tool invocation includes `auth_token` in parameters
3. **Validation**: `authenticate()` method compares tokens before executing operation
4. **Error Response**: Returns MCP error if authentication fails

### Security Considerations

**Current Implementation**:
- Simple bearer token comparison
- Token passed in request parameters (visible in logs)
- No token rotation or expiration
- No rate limiting

**Recommended for**:
- Development and testing
- Trusted network environments
- Single-user scenarios

**NOT recommended for**:
- Public internet exposure
- Multi-tenant scenarios
- Sensitive data without additional security layers

### Future Enhancements

The authentication system can be extended to support:
- **JWT tokens**: Signed tokens with claims and expiration
- **OAuth 2.0**: Standard authorization framework
- **API keys with scopes**: Fine-grained permission model
- **TLS/mTLS**: Transport-level security
- **Rate limiting**: Per-token request limits

Implementation strategy:
1. Extract auth logic to `libs/mcp/src/auth.rs`
2. Define `AuthProvider` trait
3. Implement multiple providers (Token, JWT, OAuth)
4. Select provider via CLI argument or config file

## Database Architecture

### Dual-Instance Pattern

The MCP server uses separate database instances for read and write operations:

**Write Instance (Mutations)**:
- Type: `TransactionDB`
- Mode: Read-write with transaction support
- Location: Separate path from read instance (e.g., `{db_path}_write`)
- Purpose: Execute mutations atomically via background consumer

**Read Instance (Queries)**:
- Type: `DB` (readonly) or `TransactionDB` (readonly mode)
- Mode: Readonly
- Location: Primary database path
- Purpose: Execute queries without lock contention

### Rationale

RocksDB's transaction support requires exclusive write access. By separating read and write instances:
- Queries don't block on mutation transactions
- Multiple query consumers can run concurrently
- Write instance handles all mutations sequentially for consistency

### Background Consumers

Both mutations and queries are processed by background tokio tasks:

```rust
// Mutation consumer (runs transactions against TransactionDB)
let mutation_handle = spawn_graph_consumer(
    mutation_receiver,
    writer_config,
    &write_db_path,
);

// Query consumer (runs against readonly DB)
let query_handle = spawn_query_consumer(
    query_receiver,
    reader_config,
    graph_processor,
);
```

The MCP server communicates with these consumers via async channels:
- `Writer::send()` sends mutations to the mutation consumer
- `Reader::send_query()` sends queries to the query consumer
- Results are returned via oneshot channels

## Configuration

### Command-Line Interface

```bash
motlie-mcp-server [OPTIONS]

Options:
  -d, --db-path <PATH>         Path to RocksDB database (required)
  -t, --transport <TYPE>       Transport: stdio or http [default: stdio]
  -p, --port <PORT>            HTTP port (if transport=http) [default: 8080]
  -a, --auth-token <TOKEN>     Authentication token (optional)
  -h, --help                   Print help
```

### Environment Variables (Future)

Potential environment variable support:

```bash
MOTLIE_DB_PATH=/data/motlie.db
MOTLIE_AUTH_TOKEN=secret-token
MOTLIE_TRANSPORT=stdio
MOTLIE_LOG_LEVEL=info
```

### Configuration File (Future)

TOML-based configuration:

```toml
[database]
path = "/data/motlie.db"
write_path = "/data/motlie_write.db"

[server]
transport = "http"
port = 8080

[auth]
enabled = true
token = "secret-token"
# provider = "jwt"  # Future: JWT, OAuth, etc.

[channels]
mutation_buffer_size = 100
query_buffer_size = 100

[timeouts]
query_timeout_secs = 5
mutation_timeout_secs = 10
```

## Error Handling Strategy

### Error Types and Mapping

| Source Error | MCP Error Type | Example |
|--------------|----------------|---------|
| Invalid UUID parsing | `invalid_params` | `"Invalid ID: not a valid base32 ULID"` |
| Missing required parameter | `invalid_params` | `"Missing required field: name"` |
| Authentication failure | `invalid_params` | `"Invalid or missing authentication token"` |
| Database not found | `internal_error` | `"Database not ready"` |
| Mutation execution failure | `internal_error` | `"Mutation failed: RocksDB error"` |
| Query timeout | `internal_error` | `"Query timeout after 5s"` |
| Query not found | `internal_error` | `"Node not found: {id}"` |

### Error Response Format

MCP errors are returned as JSON-RPC errors:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32602,
    "message": "Invalid or missing authentication token"
  }
}
```

### Logging Strategy

- **Debug**: Parameter values, query execution details
- **Info**: Tool invocations, authentication events
- **Warn**: Retryable errors, deprecation warnings
- **Error**: Fatal errors, unhandled exceptions

Example:
```rust
log::info!("Tool invocation: add_node for node_name={}", params.name);
log::debug!("Parameters: {:?}", params);
log::error!("Mutation failed: {}", error);
```

## Testing Strategy

### Unit Tests

Located in `libs/mcp/src/lib.rs`:

```rust
#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn test_add_node_with_valid_auth() {
        // Setup server with auth token
        // Call add_node with correct token
        // Assert success
    }

    #[tokio::test]
    async fn test_add_node_with_invalid_auth() {
        // Setup server with auth token
        // Call add_node with wrong token
        // Assert error
    }

    #[tokio::test]
    async fn test_query_node_by_id() {
        // Add node via mutation
        // Query by ID
        // Assert data matches
    }
}
```

### Integration Tests

Located in `libs/mcp/tests/integration_tests.rs`:

- Test full server lifecycle (startup, request handling, shutdown)
- Test stdio transport communication
- Test HTTP transport communication
- Test mutation + query workflows
- Test error scenarios

### End-to-End Tests

Manual testing with MCP clients:
- Claude Code
- Cursor
- MCP Inspector tool

## Performance Considerations

### Concurrency Model

- **Mutations**: Sequential execution via single background consumer
- **Queries**: Concurrent execution (multiple consumers possible)
- **Transport**: Async I/O via tokio runtime

### Bottlenecks

Potential bottlenecks:
1. **Mutation throughput**: Limited by RocksDB transaction performance
2. **Channel buffer size**: Fixed buffer may cause backpressure
3. **Query timeout**: Fixed 5-second timeout may be too short for complex queries

### Optimization Opportunities

1. **Batch mutations**: Group multiple mutations in single transaction
2. **Connection pooling**: For HTTP transport
3. **Caching**: Cache frequently queried nodes/edges
4. **Configurable timeouts**: Per-query timeout settings
5. **Metrics**: Track latency, throughput, error rates

## Deployment Scenarios

### 1. Development (Local)

```bash
# Stdio transport, no auth
cargo run --example mcp -- \
    --db-path /tmp/dev.db \
    --transport stdio
```

**Use case**: Local development with Claude Code or Cursor

### 2. Production (Single Server)

```bash
# HTTP transport with auth
cargo run --release --example mcp -- \
    --db-path /data/production.db \
    --transport http \
    --port 8080 \
    --auth-token "${MOTLIE_AUTH_TOKEN}"
```

**Use case**: Single-instance deployment with remote clients

### 3. Docker Container

```dockerfile
FROM rust:1.70 as builder
WORKDIR /app
COPY . .
RUN cargo build --release --example mcp

FROM debian:bullseye-slim
COPY --from=builder /app/target/release/examples/mcp /usr/local/bin/motlie-mcp
CMD ["motlie-mcp", "--db-path", "/data/motlie.db", "--transport", "http", "--port", "8080"]
```

### 4. Systemd Service

```ini
[Unit]
Description=Motlie MCP Server
After=network.target

[Service]
Type=simple
User=motlie
Environment="MOTLIE_AUTH_TOKEN=secret"
ExecStart=/usr/local/bin/motlie-mcp \
    --db-path /var/lib/motlie/db \
    --transport http \
    --port 8080 \
    --auth-token "${MOTLIE_AUTH_TOKEN}"
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

## Future Enhancements

### Short-term

1. **Complete tool implementation**: Implement all 14 tools (7 mutations + 7 queries)
2. **Comprehensive tests**: Unit, integration, and end-to-end tests
3. **Error handling refinement**: Better error messages and context
4. **Documentation**: API documentation, usage examples

### Medium-term

1. **Configuration file support**: TOML-based configuration
2. **Metrics and monitoring**: Prometheus metrics, health endpoints
3. **Advanced authentication**: JWT, OAuth support
4. **Query optimization**: Caching, connection pooling
5. **Batch operations**: Batch mutations for better throughput

### Long-term

1. **Multi-tenancy**: Support multiple databases per server
2. **Replication**: Read replicas for query scaling
3. **gRPC transport**: Binary protocol for better performance
4. **GraphQL API**: Alternative query interface
5. **Real-time subscriptions**: WebSocket-based change notifications

## References

### External Documentation

- [Model Context Protocol Specification](https://spec.modelcontextprotocol.io/)
- [rust-mcp-sdk - Rust MCP SDK](https://github.com/rust-mcp-stack/rust-mcp-sdk)
- [rust-mcp-sdk Documentation](https://docs.rs/rust-mcp-sdk/latest/rust_mcp_sdk/)
- [schemars Documentation](https://docs.rs/schemars/latest/schemars/)

### Internal Documentation

- `libs/db/src/mutation.rs`: Mutation API documentation
- `libs/db/src/query.rs`: Query API documentation
- `libs/db/src/graph.rs`: Graph storage and processing
- `examples/*/README.md`: Usage examples

## Changelog

### Version 0.1.0 (Initial Design)

- MCP server architecture
- Authentication design
- Tool definitions for all mutations and queries
- Transport support (stdio, HTTP)
- Database access patterns
- Testing strategy
- Deployment scenarios
