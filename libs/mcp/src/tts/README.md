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

# Stdio transport (now reliable - see Architecture section)
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
  "message": "Successfully queued 2 phrase(s) for speaking",
  "queued_count": 2,
  "total_phrases": 2,
  "errors": []
}
```

**Note:** The response indicates phrases were *queued* for speaking, not that they have finished speaking. The persistent worker processes phrases in order and speech completes asynchronously.

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

## Architecture

The TTS module uses a **persistent shell worker** architecture to ensure reliable speech completion, even when the MCP server's stdin closes (common with stdio transport).

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│ MCP Server Process                                          │
│                                                             │
│  TtsEngine                                                  │
│  ├── worker_stdin (pipe) ──────┐                           │
│  └── worker_handle             │                           │
│                                │                           │
└────────────────────────────────┼───────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Persistent Shell Worker (/bin/sh)                           │
│                                                             │
│  while read phrase voice rate; do                          │
│      /usr/bin/say [args] -- "$phrase"                      │
│  done                                                       │
│                                                             │
│  (Continues running even if parent stdin closes)           │
└─────────────────────────────────────────────────────────────┘
```

1. **On first tool use**: A shell worker process is spawned running an inline script
2. **When `say` is called**: Phrases are written to the worker's stdin (tab-separated format)
3. **Worker processing**: The shell reads phrases one at a time and executes `/usr/bin/say` for each
4. **On shutdown**: The worker's stdin is closed, signaling EOF. The worker finishes speaking all queued phrases before exiting

### Why This Design?

The previous implementation spawned a new `say` process for each phrase. This had a critical problem:

**Problem:** When using stdio transport, the MCP server terminates when stdin closes (EOF). If the client closed the connection before the `say` command finished, speech would be interrupted.

**Solution:** The persistent worker approach ensures:
- All queued phrases are spoken to completion
- The worker survives even if the parent MCP server exits
- Graceful shutdown waits for speech to finish (with 120s timeout)

### Graceful Shutdown

When `ManagedResource::shutdown()` is called:

1. The worker's stdin pipe is closed (signals EOF)
2. The worker finishes speaking any remaining phrases
3. The worker exits naturally after the while loop completes
4. Shutdown waits up to 120 seconds for the worker to finish
5. If timeout is exceeded, the worker is killed

```rust
// Example shutdown flow
let managed_tts = ManagedResource::new(Box::new(|| TtsEngine::new()));
let server = TtsMcpServer::new(managed_tts.lazy());

// Run server...

// Graceful shutdown - waits for all speech to complete
managed_tts.shutdown().await?;
```

## File Structure

```
tts/
├── mod.rs      # TtsEngine with persistent worker, ResourceLifecycle impl
├── types.rs    # SayParams, ListVoicesParams with ToolCall implementations
├── server.rs   # TtsMcpServer (ready-to-use MCP server)
└── README.md   # This file
```

### Key Types

| Type | Description |
|------|-------------|
| `TtsEngine` | Manages persistent shell worker for TTS |
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
        // Queue phrases to the persistent worker
        for phrase in &self.phrases {
            engine.say(phrase, self.voice.as_deref(), self.rate).await?;
        }
        // ...
    }
}
```

This ensures compile-time verification that every parameter type has a corresponding implementation.

## Transport Behavior

### Stdio Transport

The stdio transport is commonly used by MCP clients like Claude Desktop, Claude Code, and Cursor.

**Behavior:**
- Server reads JSON-RPC messages from stdin, writes responses to stdout
- Server terminates when stdin closes (EOF)
- The persistent worker ensures speech completes even after stdin closes

**Test results:**
- Phrases queued before stdin closes are fully spoken
- Worker continues for 12+ seconds after stdin closes (if needed)
- Graceful shutdown waits for worker to finish

**Example test:**
```bash
#!/bin/bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'
sleep 0.3
echo '{"jsonrpc":"2.0","method":"notifications/initialized"}'
sleep 0.3
echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"say","arguments":{"phrases":["Phrase one.","Phrase two.","Phrase three."]}}}'
sleep 0.5  # Stdin closes almost immediately - speech still completes!
```

### HTTP Transport

The HTTP transport uses the MCP Streamable HTTP protocol with Server-Sent Events (SSE).

**Behavior:**
- Server listens on configurable port (default: 8081)
- Uses session management via `mcp-session-id` header
- Responses are SSE formatted (`data: {...}`)
- Tool calls return immediately after queueing (async speech)

**Required headers:**
```
Content-Type: application/json
Accept: application/json, text/event-stream
mcp-session-id: <session-id>  # Required after initialization
```

**Session flow:**
1. Send `initialize` request → receive session ID in `mcp-session-id` response header
2. Send `notifications/initialized` with session ID header
3. Send tool calls with session ID header

**Example test with curl:**
```bash
# Start server
cargo run --example motlie_tts -- --transport http --port 8082

# In another terminal:

# Step 1: Initialize and capture session ID
HEADERS=$(mktemp)
curl -s -D "$HEADERS" -X POST http://127.0.0.1:8082/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'

SESSION_ID=$(grep -i "mcp-session-id" "$HEADERS" | sed 's/.*: //' | tr -d '\r\n')

# Step 2: Send initialized notification
curl -s -X POST http://127.0.0.1:8082/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","method":"notifications/initialized"}'

# Step 3: Call say tool
curl -s -X POST http://127.0.0.1:8082/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"say","arguments":{"phrases":["Hello from HTTP!"]}}}'
```

**Response format:**
```
data: {"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"{\"success\":true,\"message\":\"Successfully queued 1 phrase(s) for speaking\",\"queued_count\":1,\"total_phrases\":1,\"errors\":[]}"}],"isError":false}}
```

### Transport Comparison

| Aspect | Stdio | HTTP |
|--------|-------|------|
| Default port | N/A (stdin/stdout) | 8081 |
| Session management | Implicit (single session) | Explicit (`mcp-session-id` header) |
| Response format | JSON-RPC | SSE (`data: {...}`) |
| Client disconnect | stdin EOF terminates server | Session can be reused |
| Recommended for | Claude Desktop, CLI tools | Remote access, web clients |
| Speech reliability | ✅ Persistent worker | ✅ Persistent worker |

Both transports now handle long-running speech correctly thanks to the persistent worker architecture.
