//! MCP integration layer for text-to-speech.
//!
//! This module provides the MCP server scaffolding for the `motlie-tts` engine.
//! The core TTS engine lives in the `motlie_tts` crate; this module adds:
//!
//! - `ResourceLifecycle` implementation for graceful shutdown
//! - `LazyTts` / `TtsResource` wrappers for lazy initialization
//! - `SayParams` / `ListVoicesParams` with `ToolCall` implementations
//! - `TtsMcpServer` ready-to-use MCP server
//!
//! # Example: Using the default server with lifecycle management
//!
//! ```ignore
//! use motlie_mcp::tts::{TtsMcpServer, TtsEngine};
//! use motlie_mcp::ManagedResource;
//!
//! let managed_tts = ManagedResource::new(Box::new(|| TtsEngine::new()));
//! let server = TtsMcpServer::new(managed_tts.lazy());
//!
//! // Run server...
//!
//! // Graceful shutdown - waits for speech to complete
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

// Re-export core engine types from motlie-tts
pub use motlie_tts::{TtsEngine, INSTRUCTIONS};

use crate::{LazyResource, ResourceLifecycle};
use async_trait::async_trait;
use rmcp::ErrorData as McpError;
use std::sync::Arc;

// ============================================================================
// ResourceLifecycle Implementation
// ============================================================================

#[async_trait]
impl ResourceLifecycle for TtsEngine {
    /// Shut down the TTS engine gracefully.
    ///
    /// Delegates to [`TtsEngine::close()`] which closes the worker's stdin
    /// and waits for the worker to finish speaking.
    async fn shutdown(self) -> anyhow::Result<()> {
        self.close().await
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
/// The actual platform validation and worker spawn happens lazily on first
/// tool invocation, allowing the MCP handshake to complete quickly.
pub fn create_lazy_tts() -> LazyTts {
    LazyResource::new(Box::new(|| TtsEngine::new()))
}
