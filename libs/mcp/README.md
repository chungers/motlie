# Motlie MCP Server Library

MCP (Model Context Protocol) server infrastructure for building composable MCP servers, built with the [rmcp SDK](https://github.com/modelcontextprotocol/rust-sdk).

## Overview

This library provides a modular architecture for building MCP servers with support for multiple tool domains. The core infrastructure enables:

- **Tool Composition**: Combine tools from multiple domains (db, tts, etc.) into a single server
- **Trait-Based Design**: Parameter types implement `ToolCall` trait, ensuring 1:1 correspondence with implementations
- **Resource Lifecycle Management**: `ManagedResource` + `ResourceLifecycle` trait for graceful shutdown
- **Lazy Initialization**: Fast server startup with on-demand resource initialization
- **Multiple Transports**: stdio (for local integration) and HTTP with SSE (for remote access)

## Architecture

```
libs/mcp/
├── src/
│   ├── lib.rs          # Core infrastructure: LazyResource, ManagedResource,
│   │                   # ResourceLifecycle trait, ToolCall trait, re-exports
│   ├── http.rs         # Generic HTTP transport for any ServerHandler
│   ├── db/             # Motlie graph database tools
│   │   ├── mod.rs      # LazyDb, DbResource, ResourceLifecycle impl, INSTRUCTIONS
│   │   ├── types.rs    # Parameter types with ToolCall implementations
│   │   └── server.rs   # MotlieMcpServer (ready-to-use server)
│   └── tts/            # Text-to-speech tools (macOS)
│       ├── mod.rs      # LazyTts, TtsResource, TtsEngine, ResourceLifecycle impl
│       ├── types.rs    # Parameter types with ToolCall implementations
│       └── server.rs   # TtsMcpServer (ready-to-use server)
└── examples/
    └── combined_server.rs  # Example: combining db + tts tools
```

### Key Components

| Component | Description |
|-----------|-------------|
| `LazyResource<R>` | Deferred resource initialization for fast MCP handshake |
| `ManagedResource<R>` | Wrapper providing full resource lifecycle (init + shutdown) |
| `ResourceLifecycle` | Trait for resources that require graceful shutdown |
| `ToolCall` trait | Binds parameter types to their execution logic |
| `db::MotlieMcpServer` | Ready-to-use server for database tools |
| `tts::TtsMcpServer` | Ready-to-use server for TTS tools (macOS) |
| `serve_http()` | Generic HTTP transport for any `ServerHandler` |

### The `ToolCall` Trait

Each parameter type implements `ToolCall`, binding it to its execution logic:

```rust
#[async_trait]
pub trait ToolCall: Sized + Send {
    /// The resource type required to execute this tool
    type Resource: Sync;

    /// Execute the tool against the resource
    async fn call(self, resource: &Self::Resource) -> Result<CallToolResult, McpError>;
}
```

This design ensures:
- **Compile-time verification**: Every parameter type must have a `ToolCall` implementation
- **1:1 correspondence**: No sprawling re-implementations across servers
- **Easy composition**: Any server can delegate via `params.call(&resource).await`

### Resource Lifecycle Management

Resources like databases require proper shutdown to prevent data corruption. The `ResourceLifecycle` trait and `ManagedResource` wrapper provide this:

```rust
/// Trait for resources that require graceful shutdown
#[async_trait]
pub trait ResourceLifecycle: Send + Sync + Sized {
    async fn shutdown(self) -> anyhow::Result<()>;
}

/// Wrapper managing the full lifecycle: lazy init → use → shutdown
pub struct ManagedResource<R: ResourceLifecycle + Send + Sync + 'static> {
    lazy: Arc<LazyResource<R>>,
}

impl<R: ResourceLifecycle + Send + Sync + 'static> ManagedResource<R> {
    /// Create a new managed resource with a synchronous init function
    pub fn new(init_fn: ResourceInitFn<R>) -> Self;

    /// Get a clone of the lazy resource Arc for passing to servers
    pub fn lazy(&self) -> Arc<LazyResource<R>>;

    /// Gracefully shut down the resource (waits for exclusive ownership)
    pub async fn shutdown(self) -> anyhow::Result<()>;
}
```

**Architecture**:

```
ManagedResource<R: ResourceLifecycle>
    │
    └── owns Arc<LazyResource<R>>
            │
            └── lazy init → R (resource)
                    │
                    └── R::shutdown(self) on ManagedResource::shutdown()
```

**Shutdown Semantics**:
- `ManagedResource::shutdown()` uses `Arc::try_unwrap` to ensure exclusive ownership
- If other Arc references exist, shutdown returns an error
- Caller is responsible for dropping server instances before calling shutdown

## Quick Start

### Using a Pre-built Server with Lifecycle Management

```rust
use motlie_mcp::db::MotlieMcpServer;
use motlie_mcp::{ManagedResource, serve_http, HttpConfig};
use motlie_db::{Storage, StorageConfig};
use std::time::Duration;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create managed resource with proper lifecycle
    let managed_db = ManagedResource::new(Box::new(|| {
        let storage = Storage::readwrite("/path/to/db");
        storage.ready(StorageConfig::default())
    }));

    let server = MotlieMcpServer::new(managed_db.lazy(), Duration::from_secs(30));

    // Run server until Ctrl+C
    tokio::select! {
        result = serve_http(server, HttpConfig::default()) => {
            result?;
        }
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("Shutting down...");
        }
    }

    // Graceful shutdown - critical for data integrity!
    managed_db.shutdown().await?;
    Ok(())
}
```

### Composing Tools from Multiple Domains

```rust
use motlie_mcp::{db, tts, ToolCall, ManagedResource};
use rmcp::{tool, tool_router, tool_handler, ServerHandler, model::*};

#[derive(Clone)]
struct CombinedServer {
    db_resource: Arc<db::DbResource>,
    tts_resource: Arc<tts::TtsResource>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl CombinedServer {
    #[tool(description = "Add a node to the graph")]
    async fn add_node(&self, Parameters(p): Parameters<db::AddNodeParams>) -> Result<CallToolResult, McpError> {
        p.call(&self.db_resource).await  // Delegate to ToolCall impl
    }

    #[tool(description = "Speak text aloud")]
    async fn say(&self, Parameters(p): Parameters<tts::SayParams>) -> Result<CallToolResult, McpError> {
        p.call(&self.tts_resource).await  // Delegate to ToolCall impl
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create managed resources
    let managed_db = ManagedResource::new(Box::new(|| { /* ... */ }));
    let managed_tts = ManagedResource::new(Box::new(|| TtsEngine::new()));

    let server = CombinedServer::new(managed_db.lazy(), managed_tts.lazy());

    // Run server...

    // Shutdown all resources (order matters for dependencies)
    managed_tts.shutdown().await?;  // TTS first (no dependencies)
    managed_db.shutdown().await?;   // DB last (may have pending writes)
    Ok(())
}
```

See `examples/mcp/all.rs` for a complete example.

## Tool Modules

### Database Tools (`db`)

15 tools for interacting with the Motlie graph database:

**Mutations (7)**
| Tool | Description |
|------|-------------|
| `add_node` | Create a new node with name and optional temporal validity |
| `add_edge` | Create an edge between two nodes |
| `add_node_fragment` | Add a text fragment to a node |
| `add_edge_fragment` | Add a text fragment to an edge |
| `update_node_valid_range` | Update node temporal validity |
| `update_edge_valid_range` | Update edge temporal validity |
| `update_edge_weight` | Update edge weight |

**Queries (8)**
| Tool | Description |
|------|-------------|
| `query_node_by_id` | Retrieve a node by ID |
| `query_edge` | Retrieve an edge by endpoints and name |
| `query_outgoing_edges` | Get outgoing edges from a node |
| `query_incoming_edges` | Get incoming edges to a node |
| `query_nodes_by_name` | Search nodes by name (fulltext) |
| `query_edges_by_name` | Search edges by name (fulltext) |
| `query_node_fragments` | Get node fragments by time range |
| `query_edge_fragments` | Get edge fragments by time range |

**Resource Lifecycle**: `ReadWriteHandles` implements `ResourceLifecycle`, calling the internal `shutdown()` method which flushes pending writes and closes the RocksDB database cleanly.

### TTS Tools (`tts`)

Text-to-speech tools using macOS speech synthesis:

| Tool | Description |
|------|-------------|
| `say` | Speak text aloud (supports multiple phrases, voice selection, rate control) |
| `list_voices` | List available system voices |

**Platform Support**: macOS only. On other platforms, tools return an error.

**Resource Lifecycle**: `TtsEngine` implements `ResourceLifecycle` as a no-op since it only holds a path string with no cleanup needed.

## Extending the Library

### Adding a New Tool Module

To add a new tool domain (e.g., `search`):

1. **Create the module structure**:
   ```
   src/search/
   ├── mod.rs      # LazySearch, SearchResource, ResourceLifecycle impl, INSTRUCTIONS
   ├── types.rs    # Parameter types with ToolCall implementations
   └── server.rs   # SearchMcpServer
   ```

2. **Define the resource type with lifecycle** (`mod.rs`):
   ```rust
   use crate::{LazyResource, ResourceLifecycle};
   use async_trait::async_trait;

   pub struct SearchEngine { /* ... */ }

   #[async_trait]
   impl ResourceLifecycle for SearchEngine {
       async fn shutdown(self) -> anyhow::Result<()> {
           // Flush indexes, close file handles, etc.
           self.flush().await?;
           self.close().await?;
           Ok(())
       }
   }

   pub type LazySearch = LazyResource<SearchEngine>;

   pub struct SearchResource {
       search: Arc<LazySearch>,
   }

   impl SearchResource {
       pub async fn engine(&self) -> Result<&SearchEngine, McpError> {
           self.search.resource().await
       }
   }
   ```

3. **Implement parameter types with `ToolCall`** (`types.rs`):
   ```rust
   #[derive(Deserialize, JsonSchema)]
   pub struct SearchParams {
       pub query: String,
       pub limit: Option<usize>,
   }

   #[async_trait]
   impl ToolCall for SearchParams {
       type Resource = SearchResource;

       async fn call(self, res: &SearchResource) -> Result<CallToolResult, McpError> {
           let engine = res.engine().await?;
           let results = engine.search(&self.query, self.limit.unwrap_or(10));
           Ok(CallToolResult::success(vec![Content::text(
               serde_json::to_string(&results).unwrap()
           )]))
       }
   }
   ```

4. **Create the server** (`server.rs`):
   ```rust
   #[derive(Clone)]
   pub struct SearchMcpServer {
       resource: Arc<SearchResource>,
       tool_router: ToolRouter<Self>,
   }

   #[tool_router]
   impl SearchMcpServer {
       #[tool(description = "Search for documents")]
       async fn search(&self, Parameters(p): Parameters<SearchParams>) -> Result<CallToolResult, McpError> {
           p.call(&self.resource).await
       }
   }
   ```

5. **Register the module** (`lib.rs`):
   ```rust
   pub mod search;
   ```

### Composing Multiple Modules

Once your module is defined, it can be composed with other modules:

```rust
struct MultiServer {
    db: Arc<db::DbResource>,
    tts: Arc<tts::TtsResource>,
    search: Arc<search::SearchResource>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl MultiServer {
    // Include tools from any/all modules
    #[tool(description = "Add a node")]
    async fn add_node(&self, Parameters(p): Parameters<db::AddNodeParams>) -> Result<CallToolResult, McpError> {
        p.call(&self.db).await
    }

    #[tool(description = "Search documents")]
    async fn search(&self, Parameters(p): Parameters<search::SearchParams>) -> Result<CallToolResult, McpError> {
        p.call(&self.search).await
    }
}
```

## Transports

### Stdio Transport

For local process communication (Claude Desktop, Claude Code, Cursor):

```rust
use motlie_mcp::{stdio, ServiceExt};

server.serve(stdio()).await?;
```

### HTTP Transport

HTTP with Streamable HTTP protocol and SSE for long-lived connections:

```rust
use motlie_mcp::{serve_http, HttpConfig};

let config = HttpConfig::new("127.0.0.1:8080".parse()?)
    .with_mcp_path("/mcp")
    .with_sse_keep_alive(Some(Duration::from_secs(30)))
    .with_stateful_mode(true);

serve_http(server, config).await?;
```

| Option | Default | Description |
|--------|---------|-------------|
| `addr` | `127.0.0.1:8080` | Address to bind to |
| `mcp_path` | `/mcp` | MCP endpoint path |
| `sse_keep_alive` | 30 seconds | SSE keep-alive interval |
| `stateful_mode` | `true` | Maintain session state |

## Design Decisions

### Trait-Based Tool Dispatch

The `ToolCall` trait ensures every parameter type has exactly one implementation. This prevents:
- Missing tool implementations
- Inconsistent implementations across servers
- Code duplication when composing servers

### Resource Lifecycle Management

The `ResourceLifecycle` trait + `ManagedResource` wrapper ensure proper cleanup:

| Problem | Solution |
|---------|----------|
| Database corruption from unclean shutdown | `ResourceLifecycle::shutdown()` flushes writes |
| Arc references preventing shutdown | `Arc::try_unwrap` in `ManagedResource::shutdown()` |
| Forgetting to call shutdown | Clear ownership model - `ManagedResource` owner calls shutdown |
| Resources with no cleanup needed | No-op `ResourceLifecycle` impl (e.g., `TtsEngine`) |

**Why `ManagedResource` instead of constraining `LazyResource<R: ResourceLifecycle>`?**

| Aspect | Constrained LazyResource | ManagedResource wrapper |
|--------|--------------------------|-------------------------|
| Arc::try_unwrap at call site | Caller must handle | Encapsulated |
| Separation of concerns | LazyResource handles init + shutdown | LazyResource = init, ManagedResource = lifecycle |
| Flexibility | All R must impl ResourceLifecycle | LazyResource usable without constraint |
| Ownership clarity | Awkward - Arc shared but shutdown needs ownership | Owner controls lifecycle (RAII-like) |

### Lazy Resource Initialization

`LazyResource<R>` defers expensive initialization (database connections, etc.) until first tool use. This enables fast MCP handshake completion, critical for stdio transport where slow startup causes timeouts.

### Module Separation

Each tool domain (`db`, `tts`, etc.) is self-contained with:
- **mod.rs**: Resource types, `ResourceLifecycle` impl, constants
- **types.rs**: Parameter types + `ToolCall` implementations (the source of truth)
- **server.rs**: Ready-to-use MCP server (convenience)

This separation enables both standalone servers and tool composition.

## Examples

| Example | Description |
|---------|-------------|
| `examples/mcp/motlie_db.rs` | Database-only server with graceful shutdown |
| `examples/mcp/motlie_tts.rs` | TTS-only server (macOS, HTTP default) |
| `examples/mcp/all.rs` | Combined server with 17 tools (15 db + 2 tts) |

Run examples:
```bash
# Database server
cargo run --example motlie_db -- --db-path /path/to/db

# TTS server (HTTP default)
cargo run --example motlie_tts

# Combined server
cargo run --example motlie_all -- --db-path /path/to/db
```

## Documentation

- [TTS Module README](src/tts/README.md) - TTS-specific documentation including stdio limitations
- [motlie-db README](../db/README.md) - Core database documentation
- [MCP Specification](https://spec.modelcontextprotocol.io/) - Official MCP protocol specification
- [rmcp SDK](https://docs.rs/rmcp/latest/rmcp/) - Rust MCP SDK documentation

## License

MIT
