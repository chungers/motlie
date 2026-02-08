# Motlie MCP Server Design (rmcp SDK)

## Overview

This document describes the design and implementation of the Model Context Protocol (MCP) server for the Motlie graph database using the **rmcp** SDK. The MCP server exposes the graph mutation and query APIs as MCP tools, enabling AI assistants to interact with the graph database through standardized protocols.

## Architecture

### Component Structure

```
motlie/
├── libs/mcp/                    # MCP server library
│   ├── src/
│   │   ├── lib.rs              # Core MCP server implementation using pmcp
│   │   └── types.rs            # MCP parameter types with JsonSchema
│   └── docs/
│       └── DESIGN.md           # This document
└── examples/mcp/
    ├── Cargo.toml              # Example binary configuration
    ├── main.rs                 # Server entry point and CLI
    └── README.md               # Usage documentation
```

### Key Components

1. **MotlieMcpServer**: Core server struct that wraps database access and implements the rmcp `ServerHandler` trait
2. **Tool Handlers**: MCP tools implemented via rmcp's `#[tool]` attribute macro
3. **LazyDatabase**: Deferred database initialization for fast server startup
4. **Transport Layer**: Support for stdio and HTTP transports via rmcp SDK

## Dependencies

### External Crates

- **rmcp** (v0.9): Official Rust SDK for Model Context Protocol
  - Provides: `ServerHandler`, `#[tool]` macro, `#[tool_router]`, transport implementations
- **schemars** (v1.0): JSON Schema generation from Rust types
- **tokio**: Async runtime for concurrent operations
- **serde/serde_json**: Serialization for parameters
- **clap**: Command-line argument parsing (example only)
- **anyhow**: Error handling
- **log/tracing**: Logging

### Internal Dependencies

- **motlie-db**: Core graph database library providing:
  - `Writer` and `Reader` for async mutation/query execution
  - `Storage` for database access
  - All mutation types (`AddNode`, `AddEdge`, etc.)
  - All query types (`NodeById`, `OutgoingEdges`, etc.)

## Key Design Decisions

### Text-Only Fragments (No Binary Content via MCP)

Fragment content in `add_node_fragment` and `add_edge_fragment` is intentionally **text-only**. This is a deliberate design decision to prevent context window bloat when AI agents interact with the graph.

**The Problem with Binary Content in MCP Tools:**

When an AI agent (like Claude Code) needs to store binary content (e.g., an image), it would need to:
1. Fetch the content from a URL into its context window
2. Base64 encode it (~33% size increase)
3. Pass the entire encoded blob as an MCP tool parameter
4. The MCP tool call becomes part of the conversation context

This creates several issues:
- **Context window exhaustion**: A 1MB image becomes ~1.33MB of text in context
- **Inefficient data flow**: The agent becomes an unnecessary middleman for binary data
- **No streaming**: The entire blob must be present in memory for the tool call
- **Cumulative impact**: Multiple images quickly exhaust the context window

**The Solution:**

Fragment tools accept only text content. The underlying `DataUrl` type in `motlie-db` supports various content types internally, but the MCP API intentionally restricts input to text.

For binary content, use one of these patterns:
1. **URL references**: Store the binary externally and reference it by URL in text fragments
2. **External storage**: Use a separate upload mechanism outside the MCP/chat flow
3. **Direct programmatic access**: Use `motlie-db` directly (not via MCP) for binary content

**Note**: The internal helper functions (`content_to_dataurl`, `DataUrl::from_raw`) remain available for direct library usage where context windows are not a concern.

## Core Design Patterns

### 1. ServerBuilder Pattern (pmcp)

The pmcp SDK uses a fluent builder API for server configuration:

```rust
let server = MotlieMcpServer::new(writer, reader)
    .build_server(auth_token)?;

// Internally:
ServerBuilder::new()
    .name("motlie-mcp-server")
    .version(env!("CARGO_PKG_VERSION"))
    .capabilities(ServerCapabilities { ... })
    .tool("add_node", TypedTool::new(...))
    .tool("add_edge", TypedTool::new(...))
    ...
    .build()?
```

### 2. Type-Safe Tool Implementation

Tools leverage Rust's type system with automatic schema generation via pmcp's `TypedTool`:

```rust
.tool(
    "add_node",
    TypedTool::new("add_node", {
        let server = Arc::clone(&server_self);
        move |args: AddNodeParams, extra| {
            let server = Arc::clone(&server);
            Box::pin(async move {
                // Authenticate using pmcp's auth context
                Self::authenticate(&extra)?;

                // Validate parameters
                let id = Id::from_str(&args.id).map_err(|e| {
                    pmcp::Error::validation(format!("Invalid node ID: {}", e))
                })?;

                // Execute mutation
                let mutation = AddNode { ... };
                mutation.run(&server.writer).await.map_err(|e| {
                    pmcp::Error::internal(format!("Failed to add node: {}", e))
                })?;

                // Return result
                Ok(json!({
                    "success": true,
                    "message": "Successfully added node...",
                    ...
                }))
            })
        }
    })
    .with_description("Create a new node in the graph...")
)
```

**Key advantages of pmcp's TypedTool**:
- Automatic JSON schema generation from Rust types via `schemars`
- Compile-time type safety for tool parameters
- Runtime validation against generated schemas
- Clean error handling with `pmcp::Error::validation()` and `pmcp::Error::internal()`

### 3. Authentication Flow

pmcp handles authentication at the **protocol level** through `RequestHandlerExtra`:

```
Client Request with Auth
    ↓
pmcp Transport Layer validates credentials
    ↓
RequestHandlerExtra.auth_context populated
    ↓
Tool handler calls authenticate(&extra)
    ↓
Validates token from extra.auth_context
    ↓
Execute core mutation/query logic
    ↓
Return result or error
```

**Implementation**:
```rust
fn authenticate(extra: &pmcp::RequestHandlerExtra) -> pmcp::Result<()> {
    // Check if auth is required
    if let Ok(expected_token) = std::env::var("MOTLIE_MCP_AUTH_TOKEN") {
        // Auth is required, verify the token from auth_context
        if let Some(auth_ctx) = &extra.auth_context {
            if let Some(provided_token) = &auth_ctx.token {
                if provided_token == &expected_token {
                    return Ok(());
                }
            }
            return Err(pmcp::Error::validation("Invalid authentication token"));
        }
        return Err(pmcp::Error::validation("Authentication required"));
    }
    // No auth required
    Ok(())
}
```

**Key points**:
- **No custom auth module needed** - pmcp provides `AuthContext`, `AuthInfo`, `AuthScheme`
- **Bearer token authentication** - Token flows through `RequestHandlerExtra.auth_context`
- **Environment variable configuration** - `MOTLIE_MCP_AUTH_TOKEN` for expected token
- **Tool parameters are clean** - No `auth_token` fields needed in parameter structs

### 4. Database Access Pattern

The server maintains separate channels for mutations and queries:

```rust
pub struct MotlieMcpServer {
    writer: Writer,           // For mutations (async channel to graph consumer)
    reader: Reader,           // For queries (async channel to query consumer)
    query_timeout: Duration,  // Configurable query timeout
}
```

**Mutations** flow through:
1. Tool handler validates parameters and auth
2. Constructs mutation struct (e.g., `AddNode`)
3. Calls `.run(&self.writer).await`
4. Background consumer executes against TransactionDB
5. Returns success/error via pmcp error types

**Queries** flow through:
1. Tool handler validates parameters and auth
2. Constructs query struct (e.g., `NodeById`)
3. Calls `.run(&self.reader, timeout).await`
4. Background consumer executes against Storage
5. Returns data to client via JSON

### 5. Error Handling (pmcp)

All errors are mapped to pmcp-compatible errors:

```rust
// Validation errors (client-side issues)
pmcp::Error::validation("Invalid node ID: ...")
pmcp::Error::validation("Authentication required")

// Internal errors (server-side issues)
pmcp::Error::internal("Failed to add node: ...")
pmcp::Error::internal("Database error: ...")
```

Error types:
- **Validation errors**: Invalid parameters, missing auth, malformed input
- **Internal errors**: Database failures, timeouts, unexpected server issues

## Tool Definitions

### Mutation Tools

| MCP Tool | Mutation Type | Description | Parameters |
|----------|---------------|-------------|------------|
| `add_node` | `AddNode` | Create a new node | `name`, `ts_millis?`, `temporal_range?` |
| `add_edge` | `AddEdge` | Create an edge | `source_node_id`, `target_node_id`, `name`, `summary`, `weight?`, `ts_millis?`, `temporal_range?` |
| `add_node_fragment` | `AddNodeFragment` | Add text content fragment to a node | `id`, `content`, `ts_millis?`, `temporal_range?` |
| `add_edge_fragment` | `AddEdgeFragment` | Add text content fragment to an edge | `src_id`, `dst_id`, `edge_name`, `content`, `ts_millis?`, `temporal_range?` |
| `update_node_valid_range` | `UpdateNodeActivePeriod` | Update active period of a node | `id`, `temporal_range`, `reason` |
| `update_edge_valid_range` | `UpdateEdgeActivePeriod` | Update active period of an edge | `src_id`, `dst_id`, `name`, `temporal_range`, `reason` |
| `update_edge_weight` | `UpdateEdgeWeight` | Update the weight of an edge | `src_id`, `dst_id`, `name`, `weight` |

### Query Tools

| MCP Tool | Query Type | Description | Parameters |
|----------|------------|-------------|------------|
| `query_node_by_id` | `NodeById` | Retrieve node by UUID | `id`, `reference_ts_millis?` |
| `query_edge` | `EdgeDetails` | Retrieve edge by endpoints and name | `source_id`, `dest_id`, `name`, `reference_ts_millis?` |
| `query_outgoing_edges` | `OutgoingEdges` | Get all outgoing edges from a node | `id`, `reference_ts_millis?` |
| `query_incoming_edges` | `IncomingEdges` | Get all incoming edges to a node | `id`, `reference_ts_millis?` |
| `query_nodes_by_name` | `Nodes` | Search nodes by name (fulltext) | `name`, `limit?`, `reference_ts_millis?` |
| `query_edges_by_name` | `Edges` | Search edges by name (fulltext) | `name`, `limit?`, `reference_ts_millis?` |
| `query_node_fragments` | `NodeFragments` | Get node fragments in time range | `id`, `start_ts_millis?`, `end_ts_millis?`, `reference_ts_millis?` |
| `query_edge_fragments` | `EdgeFragments` | Get edge fragments in time range | `src_id`, `dst_id`, `edge_name`, `start_ts_millis?`, `end_ts_millis?`, `reference_ts_millis?` |

## Parameter Types

All tool parameters follow a consistent pattern (authentication is **not included**):

```rust
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct AddNodeParams {
    // NO auth_token field - pmcp handles auth via RequestHandlerExtra

    #[schemars(description = "Unique identifier for the node (base32-encoded ULID)")]
    pub id: String,

    #[schemars(description = "Human-readable name for the node")]
    pub name: String,

    #[schemars(description = "Timestamp in milliseconds since Unix epoch")]
    pub ts_millis: Option<u64>,

    #[schemars(description = "Time range during which this node is valid")]
    pub temporal_range: Option<TemporalRangeParam>,
}
```

Key design decisions:
- **No auth fields** - pmcp handles authentication at the protocol level
- **String UUIDs** - IDs are passed as base32-encoded strings for JSON compatibility
- **Optional timestamps** - Default to current time if not provided
- **schemars annotations** - Provide rich documentation for AI clients

## JSON Schema Generation (schemars 1.0)

The MCP server uses **schemars 1.0** for automatic JSON schema generation, matching pmcp's dependency:

```rust
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct AddNodeParams {
    /// Node UUID in base32 format
    #[schemars(description = "Unique identifier for the node (base32-encoded ULID)")]
    pub id: String,

    /// Human-readable node name
    #[schemars(description = "Human-readable name for the node")]
    pub name: String,
    // ...
}
```

pmcp's `TypedTool` automatically:
1. Generates JSON schema from the parameter type via `schemars`
2. Includes the schema in MCP tool definitions
3. Validates incoming parameters against the schema at runtime
4. Provides compile-time type safety

## Transport Support

The server supports stdio transport via pmcp:

### Standard I/O (stdio)

```rust
server.run_stdio().await?;
```

**Characteristics**:
- Bidirectional JSON-RPC over stdin/stdout
- Suitable for local process integration
- Used by Claude Code, Claude Desktop, Cursor, and other MCP clients
- pmcp handles all protocol details automatically

## Authentication Design (pmcp)

### Built-in Authentication via pmcp

pmcp provides comprehensive authentication support through:

```rust
// In RequestHandlerExtra:
pub struct AuthContext {
    pub subject: String,           // User/client identifier
    pub token: Option<String>,     // Bearer token
    pub oauth: Option<OAuthInfo>,  // OAuth metadata
    pub scheme: AuthScheme,        // Auth method (Bearer, OAuth, etc.)
}
```

### Implementation in Motlie MCP Server

```rust
// Server initialization with optional auth token
let server = mcp_server.build_server(auth_token)?;

// Internally stores token in environment variable
if let Some(token) = auth_token {
    std::env::set_var("MOTLIE_MCP_AUTH_TOKEN", token);
}

// Tool handlers authenticate each request
fn authenticate(extra: &pmcp::RequestHandlerExtra) -> pmcp::Result<()> {
    if let Ok(expected_token) = std::env::var("MOTLIE_MCP_AUTH_TOKEN") {
        if let Some(auth_ctx) = &extra.auth_context {
            if let Some(provided_token) = &auth_ctx.token {
                if provided_token == &expected_token {
                    return Ok(());
                }
            }
        }
        return Err(pmcp::Error::validation("Authentication required"));
    }
    Ok(())
}
```

### Security Considerations

**Current Implementation**:
- Bearer token comparison via pmcp's auth context
- Token validated on every tool invocation
- No token rotation or expiration (can be added via middleware)
- No rate limiting (pmcp supports middleware for this)

**Recommended for**:
- Development and testing
- Trusted network environments
- Single-user scenarios

**NOT recommended for**:
- Public internet exposure without additional security
- Multi-tenant scenarios without isolation
- Sensitive data without encryption

### Future Enhancements

pmcp supports advanced authentication via middleware:

```rust
// Example: OAuth middleware (from pmcp examples)
use pmcp::server::tool_middleware::ToolMiddleware;

struct OAuthInjectionMiddleware {
    token_store: Arc<RwLock<HashMap<String, String>>>,
}

#[async_trait]
impl ToolMiddleware for OAuthInjectionMiddleware {
    async fn on_request(&self, tool_name: &str, args: &mut Value,
                        extra: &mut RequestHandlerExtra, context: &ToolContext)
        -> Result<()>
    {
        // Extract OAuth token from auth_context
        if let Some(auth_ctx) = &extra.auth_context {
            if let Some(token) = &auth_ctx.token {
                extra.set_metadata("oauth_token".to_string(), token.clone());
            }
        }
        Ok(())
    }
}
```

## Database Architecture

### Dual-Instance Pattern

The MCP server uses separate database instances for read and write operations:

**Write Instance (Mutations)**:
- Type: `TransactionDB`
- Mode: Read-write with transaction support
- Purpose: Execute mutations atomically via background consumer

**Read Instance (Queries)**:
- Type: `DB` (readonly)
- Mode: Readonly
- Purpose: Execute queries without lock contention

### Background Consumers

```rust
// Mutation consumer (runs transactions against TransactionDB)
let (writer, mutation_receiver) = create_mutation_writer(config);
spawn_graph_consumer(mutation_receiver, config, db_path);

// Query consumer (runs against readonly DB)
let (reader, query_receiver) = create_query_reader(config);
spawn_query_consumer(query_receiver, config, db_path);
```

## Performance Considerations

### Concurrency Model

- **Mutations**: Sequential execution via single background consumer
- **Queries**: Concurrent execution (multiple consumers possible)
- **Transport**: Async I/O via tokio runtime
- **pmcp**: Zero-copy parsing with optional SIMD support

### Optimization Opportunities

1. **Batch mutations**: pmcp's `MessageBatcher` utility
2. **Caching**: pmcp middleware for query caching
3. **Metrics**: pmcp middleware for latency tracking
4. **Retry logic**: pmcp's `RetryMiddleware`

## Future Enhancements

### Short-term

1. ✅ **pmcp SDK migration** - Complete (this version)
2. **Comprehensive tests**: Unit, integration, and end-to-end tests
3. **Metrics middleware**: Track latency, throughput, error rates using pmcp middleware

### Medium-term

1. **HTTP/SSE transport**: pmcp supports this natively
2. **OAuth support**: Use pmcp's OAuth middleware patterns
3. **Advanced validation**: pmcp's `garde` integration for complex validation
4. **Batch operations**: Use pmcp's `MessageBatcher`

### Long-term

1. **WebSocket transport**: pmcp supports this with feature flag
2. **Resource watching**: pmcp's file watching capabilities for database changes
3. **MCP resources**: Expose graph structure as MCP resources
4. **MCP prompts**: Pre-defined query templates as MCP prompts

## Migration from rust-mcp-sdk to pmcp

### Key Changes

| Aspect | rust-mcp-sdk (0.7) | pmcp (1.8) |
|--------|-------------------|------------|
| **Server builder** | `server_runtime::create_server()` | `ServerBuilder::new()` |
| **Tool handlers** | `ServerHandler` trait | `TypedTool` closures |
| **Auth** | Custom `AuthContext` module | Built-in `RequestHandlerExtra.auth_context` |
| **Errors** | `RpcError`, `CallToolError` | `pmcp::Error` |
| **Schema** | `rust-mcp-schema` crate | Built-in with `schemars` |
| **Transport** | `StdioTransport::new()` | `server.run_stdio()` |
| **Tool registration** | `handle_call_tool_request()` dispatch | `.tool()` builder method |
| **Parameter types** | Had `auth_token` fields | Clean - auth via protocol |

### Benefits of pmcp

1. **Cleaner API** - No ServerHandler boilerplate, direct tool registration
2. **Better type safety** - TypedTool provides compile-time parameter validation
3. **Built-in auth** - No need for custom auth module
4. **Zero-copy parsing** - Better performance
5. **Middleware support** - Extensible cross-cutting concerns
6. **Active development** - Maintained by Pragmatic AI Labs
7. **Better docs** - Comprehensive examples and guides

## References

### External Documentation

- [pmcp GitHub Repository](https://github.com/paiml/rust-mcp-sdk)
- [pmcp Documentation](https://docs.rs/pmcp/latest/pmcp/)
- [Model Context Protocol Specification](https://spec.modelcontextprotocol.io/)
- [schemars Documentation](https://docs.rs/schemars/latest/schemars/)
- [How to Build a stdio MCP Server in Rust](https://www.shuttle.dev/blog/2025/07/18/how-to-build-a-stdio-mcp-server-in-rust)
- [Pragmatic AI Labs MCP SDK Blog](https://paiml.com/blog/2025-08-04-rust-mcp-sdk/)

### Internal Documentation

- `libs/db/src/mutation.rs`: Mutation API documentation
- `libs/db/src/query.rs`: Query API documentation
- `examples/mcp/README.md`: Usage examples and setup guide

## Changelog

### Version 0.1.0 (pmcp Migration)

- **Migrated from rust-mcp-sdk to pmcp** - Complete rewrite using pmcp 1.8
- **Removed custom auth module** - Use pmcp's built-in auth context
- **Simplified tool handlers** - TypedTool with closures instead of trait implementation
- **Cleaner parameter types** - Removed redundant `auth_token` fields
- **Updated schemars** - Upgraded to 1.0 to match pmcp
- **Improved error handling** - Use `pmcp::Error` consistently
- **Better documentation** - Comprehensive design doc and examples
