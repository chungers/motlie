//! Text-to-speech MCP tools (macOS).
//!
//! This module provides MCP tools for text-to-speech synthesis using the
//! macOS `say` command. The tools will return an error on non-macOS platforms.
//!
//! # Architecture
//!
//! - **types**: Parameter types (`SayParams`, `ListVoicesParams`) implementing `ToolCall`
//! - **server**: `TtsMcpServer` - a ready-to-use MCP server exposing TTS tools
//!
//! # Platform Support
//!
//! These tools are only supported on macOS. The `TtsEngine::new()` constructor
//! validates the platform at runtime and returns an error on unsupported systems.
//!
//! # Example: Using the default server with lifecycle management
//!
//! ```ignore
//! use motlie_mcp::tts::{TtsMcpServer, create_lazy_tts};
//! use motlie_mcp::ManagedResource;
//! use std::sync::Arc;
//!
//! let managed_tts = ManagedResource::new(Box::new(|| TtsEngine::new()));
//! let server = TtsMcpServer::new(managed_tts.lazy());
//!
//! // Run server...
//!
//! // Graceful shutdown (no-op for TTS, but follows the pattern)
//! managed_tts.shutdown().await?;
//! ```
//!
//! # Example: Composing into a custom server
//!
//! ```ignore
//! use motlie_mcp::tts::{TtsResource, SayParams};
//! use motlie_mcp::ToolCall;
//!
//! // In your custom server's #[tool] method:
//! #[tool(description = "Speak text")]
//! async fn say(&self, Parameters(params): Parameters<SayParams>) -> Result<CallToolResult, McpError> {
//!     params.call(&self.tts_resource).await
//! }
//! ```

pub mod server;
pub mod types;

pub use server::TtsMcpServer;
pub use types::*;

use crate::{LazyResource, ResourceLifecycle};
use async_trait::async_trait;
use rmcp::ErrorData as McpError;
use std::sync::Arc;

/// TTS engine handle for macOS speech synthesis.
///
/// This struct validates that the platform supports TTS (macOS only)
/// and provides access to the `say` command.
pub struct TtsEngine {
    /// Path to the 'say' command
    say_path: String,
}

impl TtsEngine {
    /// Create a new TTS engine, validating platform support.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Not running on macOS
    /// - The `say` command is not found at `/usr/bin/say`
    pub fn new() -> anyhow::Result<Self> {
        // Runtime platform check
        if cfg!(not(target_os = "macos")) {
            anyhow::bail!("TTS tools are only supported on macOS. Current platform is not macOS.");
        }

        // Verify 'say' command exists
        let say_path = "/usr/bin/say".to_string();
        if !std::path::Path::new(&say_path).exists() {
            anyhow::bail!(
                "macOS 'say' command not found at {}. Ensure you're running on a standard macOS installation.",
                say_path
            );
        }

        tracing::info!("TTS engine initialized with say command at {}", say_path);
        Ok(Self { say_path })
    }

    /// Get the path to the 'say' command.
    pub fn say_path(&self) -> &str {
        &self.say_path
    }
}

// ============================================================================
// ResourceLifecycle Implementation
// ============================================================================

#[async_trait]
impl ResourceLifecycle for TtsEngine {
    /// Shut down the TTS engine.
    ///
    /// This is a no-op since `TtsEngine` only holds a path string and doesn't
    /// have any resources that need cleanup. However, implementing the trait
    /// allows `TtsEngine` to be used with `ManagedResource` for consistency.
    async fn shutdown(self) -> anyhow::Result<()> {
        tracing::info!("TTS engine shutdown (no-op)");
        Ok(())
    }
}

/// Type alias for lazy TTS engine initialization.
pub type LazyTts = LazyResource<TtsEngine>;

/// Resource context for TTS tool execution.
///
/// This struct wraps the lazy TTS engine and is passed to each tool's
/// `ToolCall::call` method.
pub struct TtsResource {
    tts: Arc<LazyTts>,
}

impl TtsResource {
    /// Create a new TTS resource context.
    pub fn new(tts: Arc<LazyTts>) -> Self {
        Self { tts }
    }

    /// Get the TTS engine.
    pub async fn engine(&self) -> Result<&TtsEngine, McpError> {
        self.tts.resource().await
    }
}

/// Create a `LazyTts` with platform validation.
///
/// The actual platform validation happens lazily on first tool invocation,
/// allowing the MCP handshake to complete quickly.
///
/// # Example
///
/// ```ignore
/// use motlie_mcp::tts::create_lazy_tts;
/// use std::sync::Arc;
///
/// let tts = Arc::new(create_lazy_tts());
/// // Platform validation happens on first tool call, not here
/// ```
pub fn create_lazy_tts() -> LazyTts {
    LazyResource::new(Box::new(|| TtsEngine::new()))
}

/// Instructions for AI assistants using TTS tools.
pub const INSTRUCTIONS: &str = r#"
Text-to-speech tools using macOS speech synthesis.

## Available Tools

- **say**: Speak text aloud using the system voice
  - Supports multiple phrases spoken in sequence
  - Optional voice selection (use list_voices to see available voices)
  - Optional speech rate control (words per minute)

- **list_voices**: List all available system voices
  - Returns voice names that can be used with the 'say' tool

## Platform Support

These tools are only available on macOS systems. On other platforms,
tool calls will return an error indicating the platform is not supported.

## Example Usage

To speak a greeting:
```json
{"phrases": ["Hello, world!", "How are you today?"]}
```

To speak with a specific voice:
```json
{"phrases": ["Hello"], "voice": "Samantha"}
```

To speak slowly:
```json
{"phrases": ["This is spoken slowly"], "rate": 100}
```
"#;
