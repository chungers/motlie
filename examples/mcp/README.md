# Motlie MCP Server

This example demonstrates how to run the Motlie graph database as an MCP (Model Context Protocol) server, enabling AI assistants like Claude to interact with your graph database.

## Quick Start

### Build the Server

```bash
# From the project root
cargo build --release --example mcp
```

### Run with stdio Transport (Local)

```bash
# Basic usage with stdio transport
cargo run --release --example mcp -- \
    --db-path /path/to/your/database \
    --transport stdio

# With authentication
cargo run --release --example mcp -- \
    --db-path /path/to/your/database \
    --transport stdio \
    --auth-token "your-secret-token"
```

### Run with HTTP Transport (Remote)

**Note:** HTTP transport is planned for future implementation. Currently, only stdio transport is supported.

```bash
# Future: HTTP server on port 8080
# cargo run --release --example mcp -- \
#     --db-path /path/to/your/database \
#     --transport http \
#     --port 8080
```

## Command-Line Options

| Option | Description | Default | Required |
|--------|-------------|---------|----------|
| `--db-path` | Path to RocksDB database directory | - | Yes |
| `--auth-token` | Authentication token for secure access | None (no auth) | No |

Note: `--transport` and `--port` options are planned for future implementation.

## Integration with Claude Code

Claude Code (version 2025.x) supports custom MCP servers through its configuration system.

### Method 1: Using the CLI (Recommended)

```bash
# Add Motlie MCP server to Claude Code
claude mcp add motlie-db \
    cargo run --release --example mcp -- \
    --db-path /path/to/your/database \
    --transport stdio \
    --scope user

# List configured MCP servers
claude mcp list

# Remove the server
claude mcp remove motlie-db
```

### Method 2: Direct Configuration File Editing

For more control, you can directly edit Claude Code's configuration file:

**Configuration file location:**
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/Claude/claude_desktop_config.json`

**Example configuration (stdio transport, no auth):**

```json
{
  "mcpServers": {
    "motlie-db": {
      "type": "stdio",
      "command": "cargo",
      "args": [
        "run",
        "--release",
        "--example",
        "mcp",
        "--",
        "--db-path",
        "/Users/yourname/data/motlie.db",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

**Example configuration (stdio transport, with auth):**

```json
{
  "mcpServers": {
    "motlie-db": {
      "type": "stdio",
      "command": "cargo",
      "args": [
        "run",
        "--release",
        "--example",
        "mcp",
        "--",
        "--db-path",
        "/Users/yourname/data/motlie.db",
        "--transport",
        "stdio",
        "--auth-token",
        "your-secret-token-here"
      ],
      "env": {
        "RUST_LOG": "info"
      }
    }
  }
}
```

**Example configuration (HTTP transport for remote access):**

```json
{
  "mcpServers": {
    "motlie-db-remote": {
      "type": "http",
      "url": "http://your-server.com:8080",
      "headers": {
        "Authorization": "Bearer your-secret-token"
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

# Update Claude Code config to use the binary
```

**Configuration with binary:**

```json
{
  "mcpServers": {
    "motlie-db": {
      "type": "stdio",
      "command": "/usr/local/bin/motlie-mcp",
      "args": [
        "--db-path",
        "/Users/yourname/data/motlie.db",
        "--transport",
        "stdio"
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

Claude Desktop (version 2025.x) supports local MCP servers through a configuration file.

### Configuration Steps

1. **Open Claude Desktop**
2. **Click Settings** (gear icon in lower-left corner)
3. **Select the Developer tab**
4. **Click "Edit Config"** - this opens `claude_desktop_config.json`

**Configuration file location:**
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

### Example Configuration

**Basic setup (stdio, no auth):**

```json
{
  "mcpServers": {
    "motlie-db": {
      "command": "cargo",
      "args": [
        "run",
        "--release",
        "--example",
        "mcp",
        "--",
        "--db-path",
        "/Users/yourname/data/motlie.db",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

**With authentication:**

```json
{
  "mcpServers": {
    "motlie-db": {
      "command": "/usr/local/bin/motlie-mcp",
      "args": [
        "--db-path",
        "/Users/yourname/data/motlie.db",
        "--transport",
        "stdio",
        "--auth-token",
        "your-secret-token"
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
        "/var/lib/motlie/production.db",
        "--transport",
        "stdio"
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

If configured correctly, Claude will list the Motlie MCP tools and be able to interact with your database.

## Available MCP Tools

Once connected, the following tools are available:

### Mutation Tools

| Tool Name | Description |
|-----------|-------------|
| `add_node` | Create a new node with ID, name, and optional temporal range |
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
| `query_edge_by_src_dst_name` | Retrieve edge by endpoints and name |
| `query_outgoing_edges` | Get all outgoing edges from a node |
| `query_incoming_edges` | Get all incoming edges to a node |
| `query_nodes_by_name` | Search nodes by name prefix |
| `query_edges_by_name` | Search edges by name prefix |
| `query_node_fragments` | Get node fragments in time range |

## Example Usage in Claude

Once the MCP server is connected, you can interact with your graph database naturally:

**Creating nodes:**
```
User: "Add a new node to the database named 'Alice' with ID 01ARZ3NDEKTSV4RRFFQ69G5FAV"

Claude: [Uses add_node tool to create the node]
```

**Creating edges:**
```
User: "Create a 'knows' edge from Alice to Bob with weight 0.8"

Claude: [Uses add_edge tool to create the relationship]
```

**Querying:**
```
User: "Show me all outgoing edges from Alice"

Claude: [Uses query_outgoing_edges to fetch and display the edges]
```

**Complex workflows:**
```
User: "Create a social network with 5 people and random friendships between them"

Claude: [Uses multiple add_node and add_edge calls to build the network]
```

## Authentication

### Development (No Auth)

For local development, you can run without authentication:

```bash
cargo run --release --example mcp -- \
    --db-path /tmp/dev.db \
    --transport stdio
```

### Production (Token Auth)

For production deployments, always use authentication:

```bash
# Set token via environment variable
export MOTLIE_AUTH_TOKEN="$(openssl rand -base64 32)"

# Run with auth
cargo run --release --example mcp -- \
    --db-path /data/production.db \
    --transport http \
    --port 8080 \
    --auth-token "$MOTLIE_AUTH_TOKEN"
```

**Client configuration with auth:**

When using authentication, each tool call must include the `auth_token` parameter:

```json
{
  "tool": "add_node",
  "arguments": {
    "auth_token": "your-secret-token",
    "id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
    "name": "Alice"
  }
}
```

Claude will automatically include the token if configured in the connection settings.

## Troubleshooting

### Server Won't Start

**Error: Database not found**
```
Solution: Ensure the database path exists and has been initialized
```

**Error: Port already in use**
```
Solution: Choose a different port with --port option
```

### Claude Can't Connect

**"MCP server failed" in Claude Code**

1. Check the server is running: `ps aux | grep motlie-mcp`
2. Check Claude Code logs: `tail -f ~/Library/Logs/Claude/mcp.log` (macOS)
3. Verify JSON config is valid: Use a JSON validator
4. Try running the command manually to see error output

**"Tool not available" errors**

1. Restart Claude Code/Desktop completely
2. Verify server is in "connected" state with `/mcp` command
3. Check server logs for errors

### Permission Issues

**Error: Permission denied**
```bash
# Fix database permissions
chmod -R 755 /path/to/database

# Fix binary permissions
chmod +x /usr/local/bin/motlie-mcp
```

### JSON Configuration Errors

**Malformed JSON**

Use a JSON validator before restarting Claude:

```bash
# Validate JSON on macOS
python3 -m json.tool ~/Library/Application\ Support/Claude/claude_desktop_config.json

# Validate JSON on Linux
python3 -m json.tool ~/.config/Claude/claude_desktop_config.json
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `RUST_LOG` | Log level (error, warn, info, debug, trace) | `export RUST_LOG=debug` |
| `RUST_BACKTRACE` | Enable backtraces on panic | `export RUST_BACKTRACE=1` |
| `MOTLIE_AUTH_TOKEN` | Authentication token | `export MOTLIE_AUTH_TOKEN=secret` |

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
    --port 8080 \
    --auth-token "your-secret-token"
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable motlie-mcp
sudo systemctl start motlie-mcp
sudo systemctl status motlie-mcp
```

### Running in Docker

Create a `Dockerfile`:

```dockerfile
FROM rust:1.70 as builder
WORKDIR /app
COPY . .
RUN cargo build --release --example mcp

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libssl3 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/examples/mcp /usr/local/bin/motlie-mcp
EXPOSE 8080
CMD ["motlie-mcp", \
     "--db-path", "/data/motlie.db", \
     "--transport", "http", \
     "--port", "8080"]
```

Build and run:

```bash
# Build image
docker build -t motlie-mcp .

# Run container
docker run -d \
    -p 8080:8080 \
    -v /path/to/db:/data \
    -e RUST_LOG=info \
    motlie-mcp
```

### Multiple Database Support

Run multiple MCP servers for different databases:

```json
{
  "mcpServers": {
    "motlie-dev": {
      "command": "/usr/local/bin/motlie-mcp",
      "args": ["--db-path", "/data/dev.db", "--transport", "stdio"]
    },
    "motlie-staging": {
      "command": "/usr/local/bin/motlie-mcp",
      "args": ["--db-path", "/data/staging.db", "--transport", "stdio"]
    },
    "motlie-prod": {
      "type": "http",
      "url": "https://prod-mcp.example.com:8080"
    }
  }
}
```

## Performance Tuning

### Database Optimization

```bash
# Adjust RocksDB settings for large databases
export ROCKSDB_MAX_OPEN_FILES=1000

# Increase channel buffer sizes (requires code modification)
# Edit examples/mcp/main.rs:
# WriterConfig { channel_buffer_size: 1000 }
# ReaderConfig { channel_buffer_size: 1000 }
```

### Logging

Control log verbosity:

```bash
# Minimal logging (production)
RUST_LOG=error cargo run --release --example mcp -- ...

# Detailed logging (debugging)
RUST_LOG=debug cargo run --release --example mcp -- ...

# Module-specific logging
RUST_LOG=motlie_mcp=debug,motlie_db=info cargo run --release --example mcp -- ...
```

## Security Considerations

### Token Security

- **Never commit tokens** to version control
- **Use environment variables** for tokens in production
- **Rotate tokens regularly** (monthly or quarterly)
- **Use strong tokens**: `openssl rand -base64 32`

### Network Security

- **Use HTTPS** for remote access (set up reverse proxy with nginx/Caddy)
- **Firewall rules**: Restrict access to known IP addresses
- **VPN**: Consider VPN for remote access instead of public exposure

### Database Security

- **File permissions**: Ensure database files are not world-readable
- **Separate instances**: Use separate databases for dev/staging/prod
- **Backups**: Regular backups with encryption

## References

### Claude Code Documentation
- [Connect Claude Code to tools via MCP](https://docs.claude.com/en/docs/claude-code/mcp) - Official Claude Code MCP documentation
- [Add MCP Servers to Claude Code - Setup Guide](https://mcpcat.io/guides/adding-an-mcp-server-to-claude-code/) - Comprehensive setup guide
- [Configuring MCP Tools in Claude Code](https://scottspence.com/posts/configuring-mcp-tools-in-claude-code) - Direct configuration approach

### Claude Desktop Documentation
- [Getting Started with Local MCP Servers on Claude Desktop](https://support.claude.com/en/articles/10949351-getting-started-with-local-mcp-servers-on-claude-desktop) - Official Claude Desktop MCP guide
- [Ultimate Guide to Claude MCP Servers & Setup](https://generect.com/blog/claude-mcp/) - Complete setup guide for 2025

### MCP Protocol Documentation
- [Model Context Protocol Specification](https://spec.modelcontextprotocol.io/) - Official MCP specification
- [Connect to local MCP servers](https://modelcontextprotocol.io/docs/develop/connect-local-servers) - MCP local server guide

### Internal Documentation
- [Design Document](../../libs/mcp/docs/DESIGN.md) - Detailed architecture and design decisions
- [Motlie DB Documentation](../../libs/db/README.md) - Core database library documentation

## Support

For issues and questions:

- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check the design document for implementation details
- **Logs**: Always include logs when reporting issues (`RUST_LOG=debug`)

## Version Information

This documentation is for:
- **Motlie MCP Server**: v0.1.0
- **Claude Code**: 2025.x
- **Claude Desktop**: 2025.x
- **rust-mcp-sdk**: v0.7.3
- **MCP Protocol**: v1.0
