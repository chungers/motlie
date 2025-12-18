//! MCP server exposing text-to-speech tools.
//!
//! This module provides `TtsMcpServer`, a ready-to-use MCP server that
//! exposes TTS tools for macOS. Each tool delegates to its parameter type's
//! `ToolCall::call` implementation.

use super::{types::*, LazyTts, TtsResource, INSTRUCTIONS};
use crate::ToolCall;
use rmcp::{
    handler::server::{tool::ToolRouter, wrapper::Parameters},
    model::*,
    tool, tool_handler, tool_router,
    ErrorData as McpError, ServerHandler,
};
use std::sync::Arc;

/// MCP Server for text-to-speech (macOS).
///
/// This server exposes TTS tools using the macOS `say` command.
/// On non-macOS platforms, tool calls will return an error.
///
/// # Example
///
/// ```ignore
/// use motlie_mcp::tts::{TtsMcpServer, create_lazy_tts};
/// use motlie_mcp::{serve_http, HttpConfig};
/// use std::sync::Arc;
///
/// let tts = Arc::new(create_lazy_tts());
/// let server = TtsMcpServer::new(tts);
///
/// // Serve over HTTP
/// serve_http(server, HttpConfig::default()).await?;
/// ```
#[derive(Clone)]
pub struct TtsMcpServer {
    resource: Arc<TtsResource>,
    tool_router: ToolRouter<Self>,
}

impl TtsMcpServer {
    /// Create a new TTS MCP server.
    ///
    /// # Arguments
    ///
    /// * `tts` - Lazy TTS engine (platform validation happens on first use)
    pub fn new(tts: Arc<LazyTts>) -> Self {
        Self {
            resource: Arc::new(TtsResource::new(tts)),
            tool_router: Self::tool_router(),
        }
    }

    /// Get the TTS resource for direct access.
    pub fn resource(&self) -> &TtsResource {
        &self.resource
    }
}

#[tool_router]
impl TtsMcpServer {
    #[tool(
        description = "Speak text aloud using macOS text-to-speech. Supports multiple phrases, voice selection, and rate control."
    )]
    async fn say(
        &self,
        Parameters(params): Parameters<SayParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.resource).await
    }

    #[tool(
        description = "List available text-to-speech voices on the system. Optionally filter by name."
    )]
    async fn list_voices(
        &self,
        Parameters(params): Parameters<ListVoicesParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.resource).await
    }
}

#[tool_handler]
impl ServerHandler for TtsMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation::from_build_env(),
            instructions: Some(INSTRUCTIONS.to_string()),
        }
    }
}
