# TTS Module (`mcp::tts`)

Text-to-speech MCP tools using macOS speech synthesis.

## Overview

This module provides MCP tools for text-to-speech on macOS using the system `say` command. It includes:

- **`say`**: Speak text aloud with support for multiple phrases, voice selection, and rate control
- **`list_voices`**: List available system voices

## Platform Support

**macOS only.** On other platforms, tool calls return an error:

```
TTS tools are only supported on macOS
```

Platform validation happens lazily on first tool invocation, allowing the MCP handshake to complete quickly.

## Usage

### Standalone TTS Server

```bash
# HTTP transport (default, recommended)
cargo run --example motlie_tts
cargo run --example motlie_tts -- --port 8081

# Stdio transport (client must keep stdin open until speech completes)
cargo run --example motlie_tts -- --transport stdio
```

### Composing into a Custom Server

```rust
use motlie_mcp::tts::{self, TtsResource, SayParams};
use motlie_mcp::ToolCall;

#[derive(Clone)]
struct MyServer {
    tts_resource: Arc<TtsResource>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl MyServer {
    #[tool(description = "Speak text aloud")]
    async fn say(&self, Parameters(p): Parameters<SayParams>) -> Result<CallToolResult, McpError> {
        p.call(&self.tts_resource).await
    }
}
```

## Tool Reference

### `say`

Speak text aloud using macOS text-to-speech.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `phrases` | `string[]` | Yes | Array of text strings to speak in sequence |
| `voice` | `string` | No | Voice name (e.g., "Alex", "Samantha"). Use `list_voices` to see available options |
| `rate` | `integer` | No | Speech rate in words per minute (default: ~175-200) |

**Example:**

```json
{
  "phrases": ["Hello, world!", "How are you today?"],
  "voice": "Samantha",
  "rate": 150
}
```

**Response:**

```json
{
  "success": true,
  "message": "Successfully spoke 2 phrase(s)",
  "spoken_count": 2,
  "total_phrases": 2,
  "errors": []
}
```

### `list_voices`

List available text-to-speech voices on the system.

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `filter` | `string` | No | Filter voices by name (case-insensitive substring match) |

**Example:**

```json
{
  "filter": "en"
}
```

**Response:**

```json
{
  "voices": [
    {"name": "Alex", "language": "en_US"},
    {"name": "Samantha", "language": "en_US"}
  ],
  "count": 2,
  "filter": "en"
}
```

## Known Limitations

### Stdio Transport and Long-Running Tools

When using **stdio transport**, there is an important limitation with long-running tools like `say`:

**Problem:** The MCP server terminates when stdin closes (EOF). If the client closes the connection before the `say` tool finishes speaking, the tool execution may be interrupted.

**Timeline of the issue:**

```
1. Client sends "say" request with multiple phrases
2. Server receives request, starts speaking (takes 10-30 seconds)
3. Client closes stdin (sends EOF)
4. Server receives EOF, initiates shutdown
5. Speech may be interrupted mid-sentence
```

**Workarounds:**

1. **Use HTTP transport (recommended):**
   ```bash
   cargo run --example motlie_tts -- --transport http --port 8081
   ```
   HTTP connections remain open until the response is sent, avoiding this issue.

2. **Keep stdin open on the client side:**
   The client must keep the stdin pipe open until receiving the response. For testing with shell scripts:
   ```bash
   {
     echo '{"jsonrpc":"2.0","id":1,"method":"initialize",...}'
     sleep 0.3
     echo '{"jsonrpc":"2.0","method":"notifications/initialized"}'
     sleep 0.3
     echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"say",...}}'
     sleep 30  # Keep stdin open while speech completes
   } | cargo run --example motlie_tts -- --transport stdio
   ```

3. **Use a proper MCP client:**
   MCP clients like Claude Desktop, Claude Code, and Cursor maintain the connection properly and wait for responses.

**Root Cause:**

This is a limitation of the [rmcp SDK's](https://github.com/modelcontextprotocol/rust-sdk) stdio transport, which terminates the service when stdin closes rather than waiting for in-flight requests to complete. A proper fix would require upstream changes to rmcp.

**This limitation does NOT affect:**
- HTTP transport
- Proper MCP clients that maintain connections
- Short-duration tools that complete before stdin closes

## Architecture

```
tts/
├── mod.rs      # LazyTts, TtsResource, TtsEngine, create_lazy_tts()
├── types.rs    # SayParams, ListVoicesParams with ToolCall implementations
├── server.rs   # TtsMcpServer (ready-to-use MCP server)
└── README.md   # This file
```

### Key Types

| Type | Description |
|------|-------------|
| `TtsEngine` | Wrapper around macOS `say` command with platform validation |
| `LazyTts` | `LazyResource<TtsEngine>` - deferred initialization |
| `TtsResource` | Resource context passed to `ToolCall::call()` |
| `TtsMcpServer` | Ready-to-use MCP server exposing both tools |
| `SayParams` | Parameters for `say` tool, implements `ToolCall` |
| `ListVoicesParams` | Parameters for `list_voices` tool, implements `ToolCall` |

### ToolCall Pattern

Each parameter type implements the `ToolCall` trait:

```rust
#[async_trait]
impl ToolCall for SayParams {
    type Resource = TtsResource;

    async fn call(self, res: &TtsResource) -> Result<CallToolResult, McpError> {
        let engine = res.engine().await?;
        // Execute macOS 'say' command for each phrase
        // ...
    }
}
```

This ensures compile-time verification that every parameter type has a corresponding implementation.
