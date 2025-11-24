# Motlie MCP Server (pmcp SDK)

This example demonstrates how to run the Motlie graph database as an MCP (Model Context Protocol) server using the **pmcp** (Pragmatic Model Context Protocol) SDK, enabling AI assistants like Claude to interact with your graph database.

**Key Features**:
- Built with pmcp SDK v1.8 (Pragmatic AI Labs)
- Type-safe tools with automatic JSON schema generation via `schemars`
- Built-in authentication via pmcp's auth context
- **Dual transport support**: stdio (local) and HTTP (remote)
- **Multi-threaded query processing**: Worker pool with shared TransactionDB for 99%+ consistency
- 14 tools: 7 mutations + 7 queries

## Quick Start

### Build the Server

```bash
# From the project root
cargo build --release --example mcp
```

### Run with stdio Transport (Local)

```bash
# Basic usage with stdio transport (no authentication)
# stdio is the default transport, --transport flag is optional
cargo run --release --example mcp -- \
    --db-path /path/to/your/database

# Explicitly specify stdio transport
cargo run --release --example mcp -- \
    --db-path /path/to/your/database \
    --transport stdio

# With Bearer token authentication
cargo run --release --example mcp -- \
    --db-path /path/to/your/database \
    --transport stdio \
    --auth-token "your-secret-token"
```

**Note**: Authentication is handled by pmcp at the protocol level via `RequestHandlerExtra.auth_context`. Tool parameters do NOT include `auth_token` fields.

### Run with HTTP Transport (Remote)

The server now supports HTTP transport with Server-Sent Events (SSE) for remote access:

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

# HTTP server with authentication
cargo run --release --example mcp -- \
    --db-path /path/to/your/database \
    --transport http \
    --port 8080 \
    --auth-token "your-secret-token"
```

The HTTP server uses pmcp's `StreamableHttpServer` which provides:
- **Stateless mode** by default (no session management)
- **JSON-RPC over HTTP** for tool calls
- **Standard HTTP headers** for authentication
- **Compatible** with any HTTP MCP client

## Command-Line Options

| Option | Description | Default | Required |
|--------|-------------|---------|----------|
| `--db-path` / `-d` | Path to RocksDB database directory | - | Yes |
| `--transport` / `-t` | Transport protocol: `stdio` or `http` | `stdio` | No |
| `--port` / `-p` | Port for HTTP transport (ignored for stdio) | `8080` | No |
| `--host` | Host address for HTTP transport (ignored for stdio) | `127.0.0.1` | No |
| `--auth-token` / `-a` | Bearer token for authentication (pmcp auth context) | None (no auth) | No |
| `--mutation-buffer-size` | Mutation channel buffer size | `100` | No |
| `--query-buffer-size` | Query channel buffer size | `100` | No |
| `--query-workers` | Number of concurrent query worker threads | CPU cores | No |

**Transport Options**:
- **stdio**: Standard input/output for local MCP clients (Claude Desktop, Claude Code, etc.)
- **http**: HTTP server with JSON-RPC for remote access

**Query Workers**:
- Controls parallel query processing across multiple CPU cores
- All workers share a single readwrite TransactionDB via Arc<Graph>
- Provides **99%+ read-after-write consistency** (vs 25-30% with separate readonly instances)
- Uses RocksDB's native MVCC for thread-safe concurrent access
- Default is the number of CPU cores for optimal throughput
- Example: `--query-workers 4` uses 4 worker threads

**Authentication Note**: When `--auth-token` is provided, pmcp validates the token at the protocol level via `RequestHandlerExtra.auth_context`. Tools do not need `auth_token` in their parameters.

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
        "/Users/yourname/data/motlie.db"
      ]
    }
  }
}
```

**Note**: The `--transport stdio` argument is no longer needed - pmcp uses stdio by default when running with `server.run_stdio()`.

**Example configuration (stdio transport, with authentication):**

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

**Authentication with pmcp**: The Bearer token is validated at the protocol level. pmcp's `RequestHandlerExtra.auth_context` provides the authentication info to tool handlers. No need for `auth_token` in tool parameters!

**Example configuration (HTTP transport for remote access):**

The MCP server now supports HTTP transport! Configure your MCP client to connect to the HTTP endpoint:

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

**How it works**:
1. Start the server with `--transport http --port 8080 --auth-token "your-token"`
2. pmcp's `StreamableHttpServer` handles all HTTP protocol details
3. MCP clients send JSON-RPC requests via HTTP POST
4. Authentication headers are validated and populate `RequestHandlerExtra.auth_context`
5. Tools execute exactly the same way as stdio transport

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

## Getting Started: Your First MCP Session

This section walks you through setting up and using the Motlie MCP server with Claude Desktop from scratch.

### Step 1: Build the Server

```bash
# Navigate to the project root
cd /path/to/motlie

# Build the MCP server in release mode
cargo build --release --example mcp

# The binary will be at: target/release/examples/mcp
```

### Step 2: Choose a Database Location

```bash
# Create a directory for your database
mkdir -p ~/motlie-databases/my-first-graph

# Note the full path - you'll need it for the config
echo ~/motlie-databases/my-first-graph
```

### Step 3: Configure Claude Desktop

1. **Open Claude Desktop**
2. **Click the Settings icon** (gear in lower-left corner)
3. **Go to the Developer tab**
4. **Click "Edit Config"** to open `claude_desktop_config.json`

Add this configuration (update paths for your system):

```json
{
  "mcpServers": {
    "motlie-graph": {
      "command": "/Users/yourname/motlie/target/release/examples/mcp",
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

**Important**: Replace `/Users/yourname` with your actual home directory path!

4. **Save the file**
5. **Completely quit and restart Claude Desktop** (not just close the window)

### Step 4: Verify Connection

After Claude Desktop restarts:

1. **Start a new conversation**
2. **Ask Claude**: "What MCP tools do you have available?"
3. **You should see** a list including tools like `add_node`, `query_node_by_id`, etc.

If you see the Motlie tools, you're connected! 🎉

### Step 5: Build Your First Graph

Let's create a simple knowledge graph about a team:

**Step 5.1: Create Nodes for Team Members**

```
You: "I want to create a knowledge graph about my team. First, add a node for Alice
with ID 01HQEWRKZGPVX8Q9M7NKJT5W2E. Then add a node for Bob with ID
01HQEWRKZGPVX8Q9M7NKJT5W2F."
```

Claude will use the `add_node` tool twice to create these nodes.

**What happens behind the scenes:**
```json
Tool: add_node
Arguments: {
  "id": "01HQEWRKZGPVX8Q9M7NKJT5W2E",
  "name": "Alice"
}

Tool: add_node
Arguments: {
  "id": "01HQEWRKZGPVX8Q9M7NKJT5W2F",
  "name": "Bob"
}
```

**Step 5.2: Create Relationships**

```
You: "Create a 'reports_to' edge from Alice to Bob with weight 1.0"
```

Claude will use `add_edge`:
```json
Tool: add_edge
Arguments: {
  "src_id": "01HQEWRKZGPVX8Q9M7NKJT5W2E",
  "dst_id": "01HQEWRKZGPVX8Q9M7NKJT5W2F",
  "name": "reports_to",
  "weight": 1.0
}
```

**Step 5.3: Query the Graph**

```
You: "Show me all outgoing edges from Alice"
```

Claude will use `query_outgoing_edges` and show you that Alice reports to Bob.

**Step 5.4: Add Rich Content**

```
You: "Add a note to Alice's node saying 'Senior Software Engineer, joined 2023'"
```

Claude will use `add_node_fragment`:
```json
Tool: add_node_fragment
Arguments: {
  "node_id": "01HQEWRKZGPVX8Q9M7NKJT5W2E",
  "content": "Senior Software Engineer, joined 2023"
}
```

### Step 6: Advanced Example - Build a Complete Project Graph

Let's build a more complex graph representing a software project:

```
You: "Help me create a knowledge graph for a web application project. Create nodes for:
- The project itself (name: WebApp)
- Three developers: Alice, Bob, and Carol
- Two features: Authentication and Dashboard

Then create these relationships:
- Alice and Bob work_on Authentication
- Carol works_on Dashboard
- Authentication depends_on Dashboard (because auth needs the dashboard framework)

Use meaningful IDs and add notes about each developer's role."
```

**Claude will orchestrate multiple tool calls:**

1. Creates 6 nodes (project + 3 developers + 2 features)
2. Creates 5 edges for the relationships
3. Adds fragments with role descriptions

**Then you can query:**
```
You: "What features does Alice work on?"
You: "What are all the dependencies in this project?"
You: "Show me the complete graph structure"
```

## Example Usage Patterns

### Pattern 1: Building a Knowledge Graph from Conversation

```
You: "I'm going to tell you about my organization's structure, and I want you
to build a graph as we talk. Ready?"

[Claude confirms]

You: "Our company has three departments: Engineering, Sales, and Marketing.
Alice heads Engineering, Bob heads Sales, and Carol heads Marketing.
Under Alice, there are two teams: Frontend led by Dave and Backend led by Eve."

[Claude automatically creates nodes and edges representing this hierarchy]

You: "Now show me everyone who reports to Alice"

[Claude queries and displays the organizational structure]
```

### Pattern 2: Temporal Knowledge Tracking

```
You: "Create a node for our Product Roadmap with ID 01HQEX... and add today's
features as fragments with timestamps. Each feature should be a separate fragment."

[Claude adds multiple fragments with automatic timestamps]

You: "Now query all fragments from the last 7 days"

[Claude retrieves recent roadmap updates]
```

### Pattern 3: Research and Connection Discovery

```
You: "I'm researching connections between different technologies. Create nodes
for: React, Vue, Angular, TypeScript, and JavaScript. Then create edges showing
that React, Vue, and Angular all 'use' TypeScript, and TypeScript 'compiles_to'
JavaScript."

[Claude builds the tech graph]

You: "What are all the technologies that ultimately depend on JavaScript?"

[Claude traverses the graph to find all connections]
```

### Pattern 4: Iterative Graph Building

```
You: "Let's build a recipe graph. Start by creating a node for 'Chocolate Cake'"

[Claude creates the node]

You: "Good. Now add edges to ingredient nodes: flour, sugar, cocoa, eggs, butter.
Create those ingredient nodes if they don't exist."

[Claude creates nodes and edges]

You: "For each ingredient, add a fragment noting the quantity needed"

[Claude adds detailed fragments]

You: "Show me all recipes that use flour"

[Claude queries for connections]
```

## Example Usage in Claude

Once the MCP server is connected, you can interact with your graph database naturally:

**Creating nodes:**
```
You: "Add a new node to the database named 'Alice' with ID 01ARZ3NDEKTSV4RRFFQ69G5FAV"

Claude: I'll add a node for Alice to the graph database.
[Uses add_node tool]
✓ Successfully added node 'Alice' with ID 01ARZ3NDEKTSV4RRFFQ69G5FAV
```

**Creating edges:**
```
You: "Create a 'knows' edge from Alice to Bob with weight 0.8"

Claude: I'll create a relationship showing that Alice knows Bob.
[Uses add_edge tool]
✓ Successfully created 'knows' edge from Alice to Bob with weight 0.8
```

**Querying:**
```
You: "Show me all outgoing edges from Alice"

Claude: Let me query Alice's outgoing connections.
[Uses query_outgoing_edges tool]

Alice has the following outgoing edges:
- knows → Bob (weight: 0.8)
- works_with → Carol (weight: 1.0)
```

**Complex workflows:**
```
You: "Create a social network with 5 people and random friendships between them"

Claude: I'll create a small social network for you.
[Uses multiple add_node and add_edge calls to build the network]

Created 5 people: Alice, Bob, Carol, Dave, Eve
Created 7 friendships with varying strengths
Would you like me to visualize the connections?
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

### Common Issues and Solutions

#### Issue 1: "MCP server failed to connect"

**Symptoms:** Claude Desktop shows the server as "failed" or doesn't list your tools

**Solutions:**

1. **Verify the binary path is correct**
   ```bash
   # Test that the binary exists and runs
   /Users/yourname/motlie/target/release/examples/mcp --help

   # If you get "command not found", rebuild:
   cd /path/to/motlie
   cargo build --release --example mcp
   ```

2. **Check the database path exists**
   ```bash
   # Create the directory if it doesn't exist
   mkdir -p ~/motlie-databases/my-first-graph

   # Verify permissions
   ls -la ~/motlie-databases/
   ```

3. **Verify JSON configuration is valid**
   ```bash
   # On macOS, validate the JSON
   python3 -m json.tool ~/Library/Application\ Support/Claude/claude_desktop_config.json

   # Should print the JSON without errors
   ```

4. **Test the server manually**
   ```bash
   # Try running the server directly
   /Users/yourname/motlie/target/release/examples/mcp \
     --db-path ~/motlie-databases/my-first-graph

   # You should see:
   # [INFO] Initializing database...
   # [INFO] Starting Motlie MCP server...
   # [WARN] Running without authentication...

   # Press Ctrl+C to stop
   ```

5. **Check Claude Desktop logs**
   ```bash
   # On macOS
   tail -f ~/Library/Logs/Claude/mcp.log

   # On Windows
   # Check: %APPDATA%\Claude\logs\mcp.log

   # Look for errors related to "motlie-graph"
   ```

6. **Restart Claude Desktop completely**
   - Don't just close the window
   - Quit the application (Cmd+Q on macOS)
   - Wait 5 seconds
   - Reopen Claude Desktop

#### Issue 2: "Tools appear but don't work"

**Symptoms:** Claude sees the tools but gets errors when trying to use them

**Solutions:**

1. **Check if database is writable**
   ```bash
   # Verify you can write to the database directory
   touch ~/motlie-databases/my-first-graph/test.txt
   rm ~/motlie-databases/my-first-graph/test.txt
   ```

2. **Look for RocksDB errors in logs**
   ```bash
   # Run server with debug logging
   RUST_LOG=debug /Users/yourname/motlie/target/release/examples/mcp \
     --db-path ~/motlie-databases/my-first-graph
   ```

3. **Try a fresh database**
   ```bash
   # Rename the old database
   mv ~/motlie-databases/my-first-graph ~/motlie-databases/my-first-graph.backup

   # Create a new one
   mkdir -p ~/motlie-databases/my-first-graph

   # Restart Claude Desktop and try again
   ```

#### Issue 3: "Invalid node ID" errors

**Symptoms:** Error messages about invalid IDs when creating nodes

**Solutions:**

1. **Use proper ULID format**
   - IDs must be 26 characters, base32 encoded
   - Valid characters: 0-9, A-Z (excluding I, L, O, U)
   - Example valid ID: `01ARZ3NDEKTSV4RRFFQ69G5FAV`

2. **Let Claude generate IDs**
   ```
   You: "Create a node for Alice - generate an appropriate ID for me"

   [Claude will generate a valid ULID]
   ```

3. **Generate IDs online**
   - Visit: https://www.ulidgenerator.com/
   - Copy the generated ID into your request

#### Issue 4: Server crashes or stops responding

**Solutions:**

1. **Check for database corruption**
   ```bash
   # Back up and recreate database
   mv ~/motlie-databases/my-first-graph ~/motlie-databases/backup
   mkdir -p ~/motlie-databases/my-first-graph
   ```

2. **Increase channel buffer sizes**
   ```json
   {
     "mcpServers": {
       "motlie-graph": {
         "command": "/path/to/mcp",
         "args": [
           "--db-path", "/path/to/db",
           "--mutation-buffer-size", "1000",
           "--query-buffer-size", "1000"
         ]
       }
     }
   }
   ```

3. **Monitor system resources**
   ```bash
   # Check if you're running out of memory
   top -l 1 | grep -E "^PhysMem"
   ```

### Verification Checklist

Before asking for help, verify:

- [ ] Server binary exists at the configured path
- [ ] Database directory exists and is writable
- [ ] JSON config is valid (use a JSON validator)
- [ ] Claude Desktop has been completely restarted
- [ ] Server can run manually without errors
- [ ] Logs don't show RocksDB or permission errors
- [ ] IDs are valid 26-character ULIDs

## Common Questions

### Q: Do I need to generate IDs myself?

**A:** You can ask Claude to generate valid ULIDs for you. Just say "create a node for Alice, generate an appropriate ID" and Claude will create a valid ULID.

### Q: Can I use the same database from multiple Claude conversations?

**A:** Yes! All conversations using the same MCP server configuration will access the same database. This means you can build a graph in one conversation and query it in another.

### Q: What happens if I restart Claude Desktop?

**A:** The MCP server process restarts, but your database persists on disk. All your nodes, edges, and fragments remain intact.

### Q: Can I inspect the database outside of Claude?

**A:** Yes! You can use RocksDB tools or write a simple Rust program using the `motlie-db` library to read and query your database directly.

### Q: How do I back up my graph database?

**A:** Simply copy the database directory:
```bash
# Create a backup
cp -r ~/motlie-databases/my-first-graph ~/motlie-databases/backup-$(date +%Y%m%d)

# Restore from backup
cp -r ~/motlie-databases/backup-20250101 ~/motlie-databases/my-first-graph
```

### Q: Can I run multiple MCP servers for different databases?

**A:** Yes! Add multiple server configurations to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "motlie-personal": {
      "command": "/path/to/mcp",
      "args": ["--db-path", "/path/to/personal-graph"]
    },
    "motlie-work": {
      "command": "/path/to/mcp",
      "args": ["--db-path", "/path/to/work-graph"]
    }
  }
}
```

Claude will have access to both databases simultaneously.

### Q: How do I enable authentication?

**A:** Add the `--auth-token` argument:

```json
{
  "mcpServers": {
    "motlie-graph": {
      "command": "/path/to/mcp",
      "args": [
        "--db-path", "/path/to/db",
        "--auth-token", "your-secret-token"
      ]
    }
  }
}
```

**Note:** Store the token securely! Don't commit it to version control.

### Q: What's the difference between nodes and fragments?

**A:**
- **Nodes** represent entities (people, concepts, things)
- **Edges** represent relationships between nodes
- **Fragments** are pieces of content attached to nodes or edges (notes, documents, data)

Think of nodes as objects, edges as connections, and fragments as attributes or documentation.

### Q: Can I delete nodes or edges?

**A:** The current version supports adding and updating, but not deleting. Deletion will be added in a future release. For now, you can use temporal validity ranges to mark nodes/edges as no longer valid.

### Q: How do I visualize my graph?

**A:** Currently, visualization is not built into the MCP server. You can:
1. Ask Claude to describe the structure in text/markdown
2. Export the data and use tools like Graphviz
3. Build a custom visualization using the query tools

A built-in visualization feature is planned for a future release.

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
