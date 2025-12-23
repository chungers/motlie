//! MCP server exposing all Motlie database tools.
//!
//! This module provides `MotlieMcpServer`, a ready-to-use MCP server that
//! exposes all database tools. Each tool delegates to its parameter type's
//! `ToolCall::call` implementation.

use super::{types::*, DbResource, LazyDb, INSTRUCTIONS};
use crate::ToolCall;
use rmcp::{
    handler::server::{tool::ToolRouter, wrapper::Parameters},
    model::*,
    tool, tool_handler, tool_router,
    ErrorData as McpError, ServerHandler,
};
use std::sync::Arc;
use std::time::Duration;

/// MCP Server for Motlie graph database.
///
/// This server exposes all 15 database tools (7 mutations, 8 queries) via the
/// MCP protocol. Each tool is implemented by delegating to the corresponding
/// parameter type's `ToolCall::call` method.
///
/// # Example
///
/// ```ignore
/// use motlie_mcp::db::{MotlieMcpServer, LazyDb};
/// use motlie_mcp::{LazyResource, serve_http, HttpConfig};
/// use std::sync::Arc;
/// use std::time::Duration;
///
/// let db = Arc::new(LazyResource::new(Box::new(|| {
///     // Initialize database writer and reader
///     Ok((writer, reader))
/// })));
///
/// let server = MotlieMcpServer::new(db, Duration::from_secs(30));
///
/// // Serve over HTTP
/// serve_http(server, HttpConfig::default()).await?;
/// ```
#[derive(Clone)]
pub struct MotlieMcpServer {
    resource: Arc<DbResource>,
    tool_router: ToolRouter<Self>,
}

impl MotlieMcpServer {
    /// Create a new MCP server instance.
    ///
    /// # Arguments
    ///
    /// * `db` - Lazy database with writer and reader
    /// * `query_timeout` - Timeout for query operations
    pub fn new(db: Arc<LazyDb>, query_timeout: Duration) -> Self {
        Self {
            resource: Arc::new(DbResource::new(db, query_timeout)),
            tool_router: Self::tool_router(),
        }
    }

    /// Get the database resource for direct access.
    pub fn resource(&self) -> &DbResource {
        &self.resource
    }
}

#[tool_router]
impl MotlieMcpServer {
    // ==================== Mutation Tools ====================

    #[tool(
        description = "Create a new node in the graph with name and optional temporal validity range (ID is auto-generated)"
    )]
    async fn add_node(
        &self,
        Parameters(params): Parameters<AddNodeParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.resource).await
    }

    #[tool(
        description = "Create an edge between two nodes with optional weight and temporal validity"
    )]
    async fn add_edge(
        &self,
        Parameters(params): Parameters<AddEdgeParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.resource).await
    }

    #[tool(description = "Add a text content fragment to an existing node")]
    async fn add_node_fragment(
        &self,
        Parameters(params): Parameters<AddNodeFragmentParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.resource).await
    }

    #[tool(description = "Add a text content fragment to an existing edge")]
    async fn add_edge_fragment(
        &self,
        Parameters(params): Parameters<AddEdgeFragmentParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.resource).await
    }

    #[tool(description = "Update the temporal validity range of a node")]
    async fn update_node_valid_range(
        &self,
        Parameters(params): Parameters<UpdateNodeValidRangeParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.resource).await
    }

    #[tool(description = "Update the temporal validity range of an edge")]
    async fn update_edge_valid_range(
        &self,
        Parameters(params): Parameters<UpdateEdgeValidRangeParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.resource).await
    }

    #[tool(description = "Update the weight of an edge for graph algorithms")]
    async fn update_edge_weight(
        &self,
        Parameters(params): Parameters<UpdateEdgeWeightParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.resource).await
    }

    // ==================== Query Tools ====================

    #[tool(description = "Retrieve a node by its ID")]
    async fn query_node_by_id(
        &self,
        Parameters(params): Parameters<QueryNodeByIdParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.resource).await
    }

    #[tool(description = "Retrieve an edge by its source, destination, and name")]
    async fn query_edge(
        &self,
        Parameters(params): Parameters<QueryEdgeParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.resource).await
    }

    #[tool(description = "Get all outgoing edges from a node")]
    async fn query_outgoing_edges(
        &self,
        Parameters(params): Parameters<QueryOutgoingEdgesParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.resource).await
    }

    #[tool(description = "Get all incoming edges to a node")]
    async fn query_incoming_edges(
        &self,
        Parameters(params): Parameters<QueryIncomingEdgesParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.resource).await
    }

    #[tool(description = "Search for nodes by name or name prefix")]
    async fn query_nodes_by_name(
        &self,
        Parameters(params): Parameters<QueryNodesByNameParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.resource).await
    }

    #[tool(description = "Search for edges by name or name prefix")]
    async fn query_edges_by_name(
        &self,
        Parameters(params): Parameters<QueryEdgesByNameParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.resource).await
    }

    #[tool(description = "Get content fragments for a node within a time range")]
    async fn query_node_fragments(
        &self,
        Parameters(params): Parameters<QueryNodeFragmentsParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.resource).await
    }

    #[tool(description = "Get content fragments for an edge within a time range")]
    async fn query_edge_fragments(
        &self,
        Parameters(params): Parameters<QueryEdgeFragmentsParams>,
    ) -> Result<CallToolResult, McpError> {
        params.call(&self.resource).await
    }
}

#[tool_handler]
impl ServerHandler for MotlieMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation::from_build_env(),
            instructions: Some(INSTRUCTIONS.to_string()),
        }
    }
}
