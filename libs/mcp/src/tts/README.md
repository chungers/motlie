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

## Testing

### Quick Test (Stdio)

```bash
#!/bin/bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'
sleep 0.3
echo '{"jsonrpc":"2.0","method":"notifications/initialized"}'
sleep 0.3
echo '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"say","arguments":{"phrases":["Hello from the TTS test!"]}}}'
sleep 0.5  # Can close stdin immediately - speech continues!
```

Save as `test.sh`, then run:
```bash
chmod +x test.sh && ./test.sh | cargo run --example motlie_tts -- --transport stdio
```

The speech will complete even though stdin closes almost immediately after sending the request.
