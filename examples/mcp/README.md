# Motlie MCP Server (rmcp SDK)

This example demonstrates how to run the Motlie graph database as an MCP (Model Context Protocol) server using the **rmcp** SDK, enabling AI assistants like Claude to interact with your graph database.

**Key Features**:
- Built with rmcp SDK v0.9 (official Rust MCP SDK)
- Type-safe tools with automatic JSON schema generation via `schemars`
- **Dual transport support**: stdio (local) and HTTP (remote via Streamable HTTP)
- **Multi-threaded query processing**: Worker pool with shared TransactionDB for 99%+ consistency
- **Lazy database initialization**: Fast startup, database opens on first tool use
- 15 tools: 7 mutations + 8 queries

## Quick Start

### Build the Server

```bash
# From the project root
cargo build --release --example mcp
```

### Run with stdio Transport (Local)

```bash
# Basic usage with stdio transport (default)
cargo run --release --example mcp -- \
    --db-path /path/to/your/database

# Explicitly specify stdio transport
cargo run --release --example mcp -- \
    --db-path /path/to/your/database \
    --transport stdio
```

### Run with HTTP Transport (Remote)

The server supports HTTP transport using the Streamable HTTP protocol (SSE-based):

```bash
# HTTP server on default port 8080
cargo run --release --example mcp -- \
    --db-path /path/to/your/database \
    --transport http

# HTTP server on custom port
cargo run --release --example mcp -- \
    --db-path /path/to/your/database \
    --transport http \
    --port 3000

# HTTP server with custom host and port
cargo run --release --example mcp -- \
    --db-path /path/to/your/database \
    --transport http \
    --host 0.0.0.0 \
    --port 8080
```

The HTTP server uses rmcp's `StreamableHttpService` which provides:
- **Stateful mode** with session management via `LocalSessionManager`
- **SSE (Server-Sent Events)** for streaming responses
- **JSON-RPC over HTTP** for tool calls
- **Graceful shutdown** on Ctrl+C

**Important**: HTTP clients must send the `Accept: application/json, text/event-stream` header.

## Command-Line Options

| Option | Description | Default | Required |
|--------|-------------|---------|----------|
| `--db-path` / `-d` | Path to RocksDB database directory | - | Yes |
| `--transport` / `-t` | Transport protocol: `stdio` or `http` | `stdio` | No |
| `--port` / `-p` | Port for HTTP transport (ignored for stdio) | `8080` | No |
| `--host` | Host address for HTTP transport (ignored for stdio) | `127.0.0.1` | No |
| `--mcp-path` | MCP endpoint path for HTTP transport | `/mcp` | No |
| `--mutation-buffer-size` | Mutation channel buffer size | `100` | No |
| `--query-buffer-size` | Query channel buffer size | `100` | No |
| `--query-workers` | Number of concurrent query worker threads | CPU cores | No |

**Transport Options**:
- **stdio**: Standard input/output for local MCP clients (Claude Desktop, Claude Code, etc.)
- **http**: HTTP server with Streamable HTTP protocol for remote access

**Query Workers**:
- Controls parallel query processing across multiple CPU cores
- All workers share a single readwrite TransactionDB via Arc<Graph>
- Provides **99%+ read-after-write consistency** (vs 25-30% with separate readonly instances)
- Uses RocksDB's native MVCC for thread-safe concurrent access
- Default is the number of CPU cores for optimal throughput
- Example: `--query-workers 4` uses 4 worker threads

## Integration with Claude Code

Claude Code supports custom MCP servers through its configuration system.

### Method 1: Using the CLI (Recommended)

```bash
# Add Motlie MCP server to Claude Code
claude mcp add motlie-db -- \
    /path/to/motlie/target/release/examples/mcp \
    --db-path /path/to/your/database

# List configured MCP servers
claude mcp list

# Remove the server
claude mcp remove motlie-db
```

### Method 2: Direct Configuration File Editing

**Configuration file location:**
- **macOS**: `~/.claude/claude_desktop_config.json`
- **Windows**: `%USERPROFILE%\.claude\claude_desktop_config.json`
- **Linux**: `~/.claude/claude_desktop_config.json`

**Example configuration (stdio transport):**

```json
{
  "mcpServers": {
    "motlie-db": {
      "command": "/path/to/motlie/target/release/examples/mcp",
      "args": [
        "--db-path",
        "/Users/yourname/data/motlie.db"
      ]
    }
  }
}
```

**Example configuration (HTTP transport for remote access):**

For remote HTTP servers, configure your MCP client to connect to the HTTP endpoint:

```json
{
  "mcpServers": {
    "motlie-db-remote": {
      "type": "http",
      "url": "http://your-server.com:8080/mcp",
      "headers": {
        "Accept": "application/json, text/event-stream"
      }
    }
  }
}
```

### Method 3: Using Pre-built Binary

For better performance and easier deployment, use a pre-built binary:

```bash
# Build the binary once
cargo build --release --example mcp

# Copy to a permanent location
cp target/release/examples/mcp /usr/local/bin/motlie-mcp
```

**Configuration with binary:**

```json
{
  "mcpServers": {
    "motlie-db": {
      "command": "/usr/local/bin/motlie-mcp",
      "args": [
        "--db-path",
        "/Users/yourname/data/motlie.db"
      ]
    }
  }
}
```

### Verification

After adding the configuration:

1. **Restart Claude Code** completely
2. **Verify connection** using the `/mcp` command in Claude Code
   - This displays the connection status of each MCP server
   - Status will show either "connected" or "failed"
3. **List available tools** using `claude mcp list` in terminal

## Integration with Claude Desktop App

Claude Desktop supports local MCP servers through a configuration file.

### Configuration Steps

1. **Open Claude Desktop**
2. **Click Settings** (gear icon in lower-left corner)
3. **Select the Developer tab**
4. **Click "Edit Config"** - this opens `claude_desktop_config.json`

**Configuration file location:**
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

### Example Configuration

**Basic setup (stdio):**

```json
{
  "mcpServers": {
    "motlie-db": {
      "command": "/path/to/motlie/target/release/examples/mcp",
      "args": [
        "--db-path",
        "/Users/yourname/data/motlie.db"
      ],
      "env": {
        "RUST_LOG": "info"
      }
    }
  }
}
```

**Production setup with binary:**

```json
{
  "mcpServers": {
    "motlie-db": {
      "command": "/usr/local/bin/motlie-mcp",
      "args": [
        "--db-path",
        "/var/lib/motlie/production.db"
      ],
      "env": {
        "RUST_LOG": "warn",
        "RUST_BACKTRACE": "1"
      }
    }
  }
}
```

### Verification

1. **Save the configuration file**
2. **Restart Claude Desktop** completely (quit and reopen)
3. **Start a new conversation**
4. **Test the connection** by asking Claude:
   - "What MCP tools are available?"
   - "Can you add a node to the graph database?"

## Available MCP Tools

Once connected, the following tools are available:

### Mutation Tools

| Tool Name | Description |
|-----------|-------------|
| `add_node` | Create a new node with name and optional temporal range (ID auto-generated) |
| `add_edge` | Create an edge between two nodes with optional weight |
| `add_node_fragment` | Add content fragment to a node |
| `add_edge_fragment` | Add content fragment to an edge |
| `update_node_valid_range` | Update temporal validity of a node |
| `update_edge_valid_range` | Update temporal validity of an edge |
| `update_edge_weight` | Update the weight of an edge |

### Query Tools

| Tool Name | Description |
|-----------|-------------|
| `query_node_by_id` | Retrieve node by UUID |
| `query_edge` | Retrieve edge by source, destination, and name |
| `query_outgoing_edges` | Get all outgoing edges from a node |
| `query_incoming_edges` | Get all incoming edges to a node |
| `query_nodes_by_name` | Search nodes by name prefix |
| `query_edges_by_name` | Search edges by name prefix |
| `query_node_fragments` | Get node fragments in time range |
| `query_edge_fragments` | Get edge fragments in time range |

## Getting Started: Your First MCP Session

### Step 1: Build the Server

```bash
cd /path/to/motlie
cargo build --release --example mcp
```

### Step 2: Choose a Database Location

```bash
mkdir -p ~/motlie-databases/my-first-graph
```

### Step 3: Configure Claude Desktop

Add this configuration to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "motlie-graph": {
      "command": "/path/to/motlie/target/release/examples/mcp",
      "args": [
        "--db-path",
        "/Users/yourname/motlie-databases/my-first-graph"
      ],
      "env": {
        "RUST_LOG": "info"
      }
    }
  }
}
```

**Important**: Replace paths with your actual paths.

### Step 4: Verify Connection

After restarting Claude Desktop:

1. **Start a new conversation**
2. **Ask Claude**: "What MCP tools do you have available?"
3. **You should see** a list including tools like `add_node`, `query_node_by_id`, etc.

### Step 5: Build Your First Graph

**Create nodes:**
```
You: "Create nodes for Alice and Bob in the graph database"
```

Claude will use `add_node` and return the generated IDs.

**Create relationships:**
```
You: "Create a 'reports_to' edge from Alice to Bob"
```

**Query the graph:**
```
You: "Show me all outgoing edges from Alice"
```

## Example Usage Patterns

### Pattern 1: Building a Knowledge Graph from Conversation

```
You: "I'm going to tell you about my organization's structure."

[Claude confirms]

You: "Our company has three departments: Engineering, Sales, and Marketing.
Alice heads Engineering, Bob heads Sales."

[Claude creates nodes and edges]

You: "Show me everyone who reports to Alice"
```

### Pattern 2: Temporal Knowledge Tracking

```
You: "Create a node for our Product Roadmap and add today's features as fragments"

[Claude creates node and fragments with timestamps]

You: "Query all fragments from the last 7 days"
```

### Pattern 3: Research and Connection Discovery

```
You: "Create nodes for React, Vue, Angular, TypeScript, and JavaScript.
Create edges showing that React, Vue, and Angular all 'use' TypeScript."

[Claude builds the tech graph]

You: "What technologies depend on JavaScript?"
```

## HTTP Transport Details

### Streamable HTTP Protocol

The HTTP transport implements the MCP Streamable HTTP protocol:

- **Endpoint**: `http://host:port/mcp` (configurable via `--mcp-path`)
- **Method**: POST
- **Content-Type**: `application/json`
- **Accept**: Must include `application/json, text/event-stream`
- **Response**: SSE (Server-Sent Events) format

### Testing HTTP Transport

```bash
# Start the server
./target/release/examples/mcp --db-path /tmp/test-db --transport http --port 8080

# Test with curl (in another terminal)
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'
```

Expected response (SSE format):
```
data: {"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"rmcp","version":"0.9.1"},...}}
```

### Session Management

The HTTP transport uses `LocalSessionManager` for per-session state. Each connection maintains its own session, enabling:
- Multiple concurrent clients
- Session-specific state if needed
- Automatic cleanup on disconnect

## Troubleshooting

### Common Issues

#### "MCP server failed to connect"

1. **Verify the binary exists**
   ```bash
   ls -la /path/to/motlie/target/release/examples/mcp
   ```

2. **Test the server manually**
   ```bash
   /path/to/motlie/target/release/examples/mcp \
     --db-path ~/motlie-databases/test --help
   ```

3. **Check JSON configuration**
   ```bash
   python3 -m json.tool ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

#### HTTP transport returns "Not Acceptable"

The client must send the `Accept: application/json, text/event-stream` header.

#### "Invalid node ID" errors

Node IDs are auto-generated by `add_node`. Use the returned `node_id` for subsequent operations (edges, fragments, queries).

#### Server starts but tools don't work

1. **Check database permissions**
   ```bash
   touch ~/motlie-databases/test/test.txt && rm ~/motlie-databases/test/test.txt
   ```

2. **Run with debug logging**
   ```bash
   RUST_LOG=debug ./target/release/examples/mcp --db-path /tmp/test
   ```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `RUST_LOG` | Log level (error, warn, info, debug, trace) | `RUST_LOG=debug` |
| `RUST_BACKTRACE` | Enable backtraces on panic | `RUST_BACKTRACE=1` |

## Advanced Configurations

### Running as a systemd Service (Linux)

Create `/etc/systemd/system/motlie-mcp.service`:

```ini
[Unit]
Description=Motlie MCP Server
After=network.target

[Service]
Type=simple
User=motlie
Environment="RUST_LOG=info"
ExecStart=/usr/local/bin/motlie-mcp \
    --db-path /var/lib/motlie/db \
    --transport http \
    --host 0.0.0.0 \
    --port 8080
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Running in Docker

```dockerfile
FROM rust:1.70 as builder
WORKDIR /app
COPY . .
RUN cargo build --release --example mcp

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libssl3 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/examples/mcp /usr/local/bin/motlie-mcp
EXPOSE 8080
CMD ["motlie-mcp", "--db-path", "/data/motlie.db", "--transport", "http", "--host", "0.0.0.0", "--port", "8080"]
```

Build and run:

```bash
docker build -t motlie-mcp .
docker run -d -p 8080:8080 -v /path/to/db:/data motlie-mcp
```

### Multiple Database Support

Run multiple MCP servers for different databases:

```json
{
  "mcpServers": {
    "motlie-dev": {
      "command": "/usr/local/bin/motlie-mcp",
      "args": ["--db-path", "/data/dev.db"]
    },
    "motlie-prod": {
      "type": "http",
      "url": "http://prod-server:8080/mcp",
      "headers": {"Accept": "application/json, text/event-stream"}
    }
  }
}
```

## Common Questions

### Q: Do I need to generate IDs myself?

**A:** No. When you create a node with `add_node`, the ID is auto-generated and returned in the response.

### Q: Can I use the same database from multiple conversations?

**A:** Yes. All conversations using the same MCP server access the same database.

### Q: What's the difference between nodes and fragments?

**A:**
- **Nodes** represent entities
- **Edges** represent relationships between nodes
- **Fragments** are content attached to nodes or edges

### Q: Can I delete nodes or edges?

**A:** Not in the current version. Use temporal validity ranges to mark items as no longer valid.

### Q: Why are fragments text-only? Can I store images or binary data?

**A:** Fragments are intentionally **text-only** to prevent context window bloat.

When an AI agent (like Claude Code) fetches binary content (e.g., an image from a URL), it would need to:
1. Fetch the content into its context
2. Base64 encode it (~33% size increase)
3. Pass the entire encoded blob through the MCP tool call

This encoded content becomes part of the agent's conversation context, which is precious and limited. A 1MB image would consume ~1.33MB of context window space, and multiple images would quickly exhaust it.

The MCP tool API is designed for **structured data exchange**, not binary data transport. If you need to reference binary content, store it externally and reference it by URL or ID in your text fragments.

## References

### MCP Protocol Documentation
- [Model Context Protocol Specification](https://spec.modelcontextprotocol.io/)
- [rmcp SDK](https://crates.io/crates/rmcp)

### Claude Documentation
- [Claude Code MCP Guide](https://docs.claude.com/en/docs/claude-code/mcp)
- [Claude Desktop MCP Setup](https://support.claude.com/en/articles/10949351-getting-started-with-local-mcp-servers-on-claude-desktop)

### Internal Documentation
- [Motlie DB Documentation](../../libs/db/README.md)

## Version Information

- **Motlie MCP Server**: v0.1.0
- **rmcp SDK**: v0.9
- **MCP Protocol**: 2024-11-05
