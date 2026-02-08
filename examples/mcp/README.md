# Motlie MCP Servers

This directory contains MCP (Model Context Protocol) server examples built with the **rmcp** SDK, enabling AI assistants like Claude to interact with Motlie services.

## Available Servers

| Example | Description | Tools | Platform |
|---------|-------------|-------|----------|
| `motlie_db` | Graph database operations | 15 (7 mutations + 8 queries) | All |
| `motlie_tts` | Text-to-speech synthesis | 2 (say, list_voices) | macOS only |
| `motlie_all` | Combined DB + TTS | 17 | All (TTS tools error on non-macOS) |

**Key Features**:
- Built with rmcp SDK v0.9 (official Rust MCP SDK)
- Type-safe tools with automatic JSON schema generation via `schemars`
- **Dual transport support**: stdio (local) and HTTP (remote via Streamable HTTP)
- **Lazy resource initialization**: Fast startup, resources open on first tool use
- **Graceful shutdown**: Ctrl+C properly flushes database and waits for TTS completion

## Quick Start

### Build All Servers

```bash
# From the project root
cargo build --release --example motlie_db
cargo build --release --example motlie_tts
cargo build --release --example motlie_all
```

### Database Server (motlie_db)

```bash
# Stdio transport (default) - for Claude Code, Claude Desktop, Cursor
cargo run --release --example motlie_db -- --db-path /tmp/mydb

# HTTP transport - for remote access
cargo run --release --example motlie_db -- --db-path /tmp/mydb --transport http --port 8080
```

### TTS Server (motlie_tts) - macOS Only

```bash
# HTTP transport (default, recommended) - keeps connection open during speech
cargo run --release --example motlie_tts

# Stdio transport - for Claude Code integration
cargo run --release --example motlie_tts -- --transport stdio
```

**Note**: HTTP is the default for TTS because speech can take 10-30+ seconds. With stdio, if the client closes stdin before speech completes, the server terminates and speech may be interrupted.

### Combined Server (motlie_all)

```bash
# Stdio transport (default)
cargo run --release --example motlie_all -- --db-path /tmp/mydb

# HTTP transport
cargo run --release --example motlie_all -- --db-path /tmp/mydb --transport http --port 8080
```

## Command-Line Options

### motlie_db

| Option | Description | Default | Required |
|--------|-------------|---------|----------|
| `--db-path` / `-d` | Path to RocksDB database directory | - | Yes |
| `--transport` / `-t` | Transport: `stdio` or `http` | `stdio` | No |
| `--port` / `-p` | Port for HTTP transport | `8080` | No |
| `--host` | Host address for HTTP transport | `127.0.0.1` | No |
| `--mcp-path` | MCP endpoint path for HTTP | `/mcp` | No |
| `--query-timeout` | Query timeout in seconds | `30` | No |

### motlie_tts

| Option | Description | Default | Required |
|--------|-------------|---------|----------|
| `--transport` / `-t` | Transport: `stdio` or `http` | `http` | No |
| `--port` / `-p` | Port for HTTP transport | `8081` | No |
| `--host` | Host address for HTTP transport | `127.0.0.1` | No |
| `--mcp-path` | MCP endpoint path for HTTP | `/mcp` | No |

### motlie_all

| Option | Description | Default | Required |
|--------|-------------|---------|----------|
| `--db-path` / `-d` | Path to RocksDB database directory | - | Yes |
| `--transport` / `-t` | Transport: `stdio` or `http` | `stdio` | No |
| `--port` / `-p` | Port for HTTP transport | `8080` | No |
| `--host` | Host address for HTTP transport | `127.0.0.1` | No |
| `--mcp-path` | MCP endpoint path for HTTP | `/mcp` | No |
| `--query-timeout` | Query timeout in seconds | `30` | No |

## Integration with Claude Code

Claude Code supports MCP servers via the `claude mcp add` command.

### Adding Servers

```bash
# Database server
claude mcp add motlie-db -- \
    cargo run --release --example motlie_db -- \
    --db-path /path/to/your/database

# TTS server (macOS only)
claude mcp add motlie-tts -- \
    cargo run --release --example motlie_tts -- \
    --transport stdio

# Combined server (DB + TTS)
claude mcp add motlie-all -- \
    cargo run --release --example motlie_all -- \
    --db-path /path/to/your/database
```

### Using Pre-built Binaries (Recommended)

For better performance, build once and reference the binary directly:

```bash
# Build all examples
cargo build --release --example motlie_db
cargo build --release --example motlie_tts
cargo build --release --example motlie_all

# Add using the binary path
claude mcp add motlie-db -- \
    ./target/release/examples/motlie_db \
    --db-path /path/to/your/database

claude mcp add motlie-tts -- \
    ./target/release/examples/motlie_tts \
    --transport stdio

claude mcp add motlie-all -- \
    ./target/release/examples/motlie_all \
    --db-path /path/to/your/database
```

### Managing Servers

```bash
# List configured MCP servers
claude mcp list

# Remove a server
claude mcp remove motlie-db
```

### Verification

After adding a server:

1. **Restart Claude Code** completely (or start a new session)
2. **Verify connection** using the `/mcp` command in Claude Code
3. **Test tools** by asking Claude to use them (e.g., "List available voices" for TTS)

## Integration with Claude Desktop App

Claude Desktop supports MCP servers via a JSON configuration file.

**Configuration file location:**
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

### Example Configuration

```json
{
  "mcpServers": {
    "motlie-db": {
      "command": "/path/to/motlie/target/release/examples/motlie_db",
      "args": ["--db-path", "/path/to/your/database"],
      "env": {"RUST_LOG": "info"}
    },
    "motlie-tts": {
      "command": "/path/to/motlie/target/release/examples/motlie_tts",
      "args": ["--transport", "stdio"],
      "env": {"RUST_LOG": "info"}
    }
  }
}
```

### Verification

1. Save the configuration file
2. Restart Claude Desktop completely (quit and reopen)
3. Test by asking Claude: "What MCP tools are available?"

## Available MCP Tools

### Database Tools (motlie_db, motlie_all)

**Mutation Tools:**

| Tool | Description |
|------|-------------|
| `add_node` | Create a new node with name and optional active period |
| `add_edge` | Create an edge between two nodes with optional weight |
| `add_node_fragment` | Add content fragment to a node |
| `add_edge_fragment` | Add content fragment to an edge |
| `update_node_valid_range` | Update active period of a node |
| `update_edge_valid_range` | Update active period of an edge |
| `update_edge_weight` | Update the weight of an edge |

**Query Tools:**

| Tool | Description |
|------|-------------|
| `query_node_by_id` | Retrieve node by UUID |
| `query_edge` | Retrieve edge by source, destination, and name |
| `query_outgoing_edges` | Get all outgoing edges from a node |
| `query_incoming_edges` | Get all incoming edges to a node |
| `query_nodes_by_name` | Search nodes by name prefix |
| `query_edges_by_name` | Search edges by name prefix |
| `query_node_fragments` | Get node fragments in time range |
| `query_edge_fragments` | Get edge fragments in time range |

### TTS Tools (motlie_tts, motlie_all) - macOS Only

| Tool | Description |
|------|-------------|
| `say` | Speak text aloud using macOS speech synthesis |
| `list_voices` | List available system voices |

**Say tool options:**
- `phrases`: Array of strings to speak in sequence
- `voice`: Optional voice name (use `list_voices` to see available)
- `rate`: Optional speech rate in words per minute

## Example Usage

### Database Example

```
You: "Create nodes for Alice and Bob in the graph database"
Claude: [uses add_node twice, returns generated IDs]

You: "Create a 'reports_to' edge from Alice to Bob"
Claude: [uses add_edge with the node IDs]

You: "Show me all outgoing edges from Alice"
Claude: [uses query_outgoing_edges]
```

### TTS Example (macOS)

```
You: "List available voices"
Claude: [uses list_voices, shows Alex, Samantha, etc.]

You: "Say 'Hello world' using the Samantha voice"
Claude: [uses say with voice="Samantha"]
```

## HTTP Transport

The HTTP transport implements the MCP Streamable HTTP protocol:

- **Endpoint**: `http://host:port/mcp` (configurable via `--mcp-path`)
- **Method**: POST
- **Content-Type**: `application/json`
- **Accept**: Must include `application/json, text/event-stream`
- **Response**: SSE (Server-Sent Events) format

### Testing with curl

```bash
# Start the server
cargo run --release --example motlie_db -- --db-path /tmp/test-db --transport http

# Test with curl (in another terminal)
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'
```

## Troubleshooting

### "MCP server failed to connect"

```bash
# Verify the binary exists
ls -la target/release/examples/motlie_db

# Test the server manually
cargo run --release --example motlie_db -- --db-path /tmp/test --help
```

### Run with debug logging

```bash
RUST_LOG=debug cargo run --release --example motlie_db -- --db-path /tmp/test
```

### TTS tools return "platform not supported"

TTS tools only work on macOS. On other platforms, the tools return an error.

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `RUST_LOG` | Log level (error, warn, info, debug, trace) | `RUST_LOG=debug` |
| `RUST_BACKTRACE` | Enable backtraces on panic | `RUST_BACKTRACE=1` |

## FAQ

**Q: Do I need to generate IDs myself?**
No. Node IDs are auto-generated by `add_node` and returned in the response.

**Q: Can I use the same database from multiple conversations?**
Yes. All conversations using the same MCP server access the same database.

**Q: What's the difference between nodes, edges, and fragments?**
- **Nodes** represent entities
- **Edges** represent relationships between nodes
- **Fragments** are timestamped content attached to nodes or edges

**Q: Can I delete nodes or edges?**
No. Use active period ranges to mark items as no longer valid.

## References

- [Model Context Protocol Specification](https://spec.modelcontextprotocol.io/)
- [rmcp SDK](https://crates.io/crates/rmcp)
- [Motlie DB Documentation](../../libs/db/README.md)
