//! MCP server library for Motlie graph database
//!
//! This library provides an MCP (Model Context Protocol) server that exposes
//! the Motlie graph database mutation and query APIs as MCP tools.

pub mod auth;
pub mod types;

use auth::AuthContext;
use async_trait::async_trait;
use motlie_db::{
    AddEdge, AddEdgeFragment, AddNode, AddNodeFragment, DataUrl, EdgeSummary,
    EdgesByName, EdgeSummaryBySrcDstName, Id, IncomingEdges, MutationRunnable,
    NodeById, NodeFragmentsByIdTimeRange, NodesByName, OutgoingEdges,
    QueryRunnable, Reader, TimestampMilli, UpdateEdgeValidSinceUntil,
    UpdateEdgeWeight, UpdateNodeValidSinceUntil, ValidTemporalRange, Writer,
};
use rust_mcp_sdk::schema::{
    schema_utils::CallToolError, CallToolRequest, CallToolResult, ListToolsRequest,
    ListToolsResult, RpcError, TextContent,
};
use rust_mcp_sdk::{mcp_server::ServerHandler, McpServer};
use serde_json::Value as JsonValue;
use std::ops::Bound;
use std::sync::Arc;
use std::time::Duration;
use types::*;

/// MCP Server for Motlie graph database
///
/// This server wraps the Motlie database's Writer and Reader interfaces
/// and exposes them as MCP tools.
pub struct MotlieMcpServer {
    writer: Writer,
    reader: Reader,
    auth: Arc<AuthContext>,
    query_timeout: Duration,
}

impl MotlieMcpServer {
    /// Create a new MCP server instance
    ///
    /// # Arguments
    /// * `writer` - Writer for mutation operations
    /// * `reader` - Reader for query operations
    /// * `auth_token` - Optional authentication token
    pub fn new(writer: Writer, reader: Reader, auth_token: Option<String>) -> Self {
        Self {
            writer,
            reader,
            auth: Arc::new(AuthContext::new(auth_token)),
            query_timeout: Duration::from_secs(5),
        }
    }

    /// Set custom query timeout
    pub fn with_query_timeout(mut self, timeout: Duration) -> Self {
        self.query_timeout = timeout;
        self
    }

    /// Authenticate a request
    fn authenticate(&self, provided_token: Option<&str>) -> Result<(), RpcError> {
        self.auth
            .authenticate(provided_token)
            .map_err(|e| RpcError::invalid_params().with_message(e.to_string()))
    }

    /// Convert temporal range parameter to schema type
    fn to_schema_temporal_range(
        param: TemporalRangeParam,
    ) -> ValidTemporalRange {
        ValidTemporalRange(
            Some(TimestampMilli(param.valid_since)),
            Some(TimestampMilli(param.valid_until)),
        )
    }

    /// Helper to extract auth token from JSON parameters
    fn extract_auth_token(params: &JsonValue) -> Option<String> {
        params
            .get("auth_token")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    /// Helper to parse JSON parameters into typed struct
    fn parse_params<T: serde::de::DeserializeOwned>(params: &JsonValue) -> Result<T, RpcError> {
        serde_json::from_value(params.clone())
            .map_err(|e| RpcError::invalid_params().with_message(format!("Failed to parse parameters: {}", e)))
    }

    /// Helper to create a simple text result
    fn text_result(message: String) -> Result<CallToolResult, CallToolError> {
        Ok(CallToolResult::text_content(vec![TextContent::from(message)]))
    }

    /// Helper to create an error result
    fn error_result(message: String) -> Result<CallToolResult, CallToolError> {
        Err(CallToolError::new(RpcError::internal_error().with_message(message)))
    }
}

#[async_trait]
impl ServerHandler for MotlieMcpServer {
    async fn handle_list_tools_request(
        &self,
        _request: ListToolsRequest,
        _runtime: Arc<dyn McpServer>,
    ) -> Result<ListToolsResult, RpcError> {
        // For now, return empty tools list - we'll add proper tool definitions later
        Ok(ListToolsResult {
            meta: None,
            next_cursor: None,
            tools: vec![],
        })
    }

    async fn handle_call_tool_request(
        &self,
        request: CallToolRequest,
        _runtime: Arc<dyn McpServer>,
    ) -> Result<CallToolResult, CallToolError> {
        let params = JsonValue::Object(request.params.arguments.unwrap_or_default());
        let auth_token = Self::extract_auth_token(&params);

        // Authenticate
        self.authenticate(auth_token.as_deref())
            .map_err(|e| CallToolError::new(e))?;

        match request.params.name.as_str() {
            "add_node" => self.handle_add_node(params).await,
            "add_edge" => self.handle_add_edge(params).await,
            "add_node_fragment" => self.handle_add_node_fragment(params).await,
            "add_edge_fragment" => self.handle_add_edge_fragment(params).await,
            "update_node_valid_range" => self.handle_update_node_valid_range(params).await,
            "update_edge_valid_range" => self.handle_update_edge_valid_range(params).await,
            "update_edge_weight" => self.handle_update_edge_weight(params).await,
            "query_node_by_id" => self.handle_query_node_by_id(params).await,
            "query_edge" => self.handle_query_edge(params).await,
            "query_outgoing_edges" => self.handle_query_outgoing_edges(params).await,
            "query_incoming_edges" => self.handle_query_incoming_edges(params).await,
            "query_nodes_by_name" => self.handle_query_nodes_by_name(params).await,
            "query_edges_by_name" => self.handle_query_edges_by_name(params).await,
            "query_node_fragments" => self.handle_query_node_fragments(params).await,
            _ => Self::error_result(format!("Unknown tool: {}", request.params.name)),
        }
    }
}

// Tool handler implementations
impl MotlieMcpServer {
    async fn handle_add_node(&self, params: JsonValue) -> Result<CallToolResult, CallToolError> {
        let params: AddNodeParams = Self::parse_params(&params)
            .map_err(|e| CallToolError::new(e))?;

        let id = Id::from_str(&params.id)
            .map_err(|e| CallToolError::new(RpcError::invalid_params().with_message(format!("Invalid node ID: {}", e))))?;

        let mutation = AddNode {
            id,
            name: params.name.clone(),
            ts_millis: TimestampMilli(params.ts_millis.unwrap_or_else(|| TimestampMilli::now().0)),
            temporal_range: params.temporal_range.map(Self::to_schema_temporal_range),
        };

        mutation
            .run(&self.writer)
            .await
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Failed to add node: {}", e))))?;

        log::info!("Added node: {} ({})", params.name, id);

        Self::text_result(format!(
            "Successfully added node '{}' with ID {}",
            params.name, params.id
        ))
    }

    async fn handle_add_edge(&self, params: JsonValue) -> Result<CallToolResult, CallToolError> {
        let params: AddEdgeParams = Self::parse_params(&params)
            .map_err(|e| CallToolError::new(e))?;

        let source_id = Id::from_str(&params.source_node_id)
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Invalid source node ID: {}", e))))?;
        let target_id = Id::from_str(&params.target_node_id)
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Invalid target node ID: {}", e))))?;

        let mutation = AddEdge {
            source_node_id: source_id,
            target_node_id: target_id,
            ts_millis: TimestampMilli(params.ts_millis.unwrap_or_else(|| TimestampMilli::now().0)),
            name: params.name.clone(),
            temporal_range: params.temporal_range.map(Self::to_schema_temporal_range),
            summary: EdgeSummary::from_text(&params.summary),
            weight: params.weight,
        };

        mutation
            .run(&self.writer)
            .await
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Failed to add edge: {}", e))))?;

        log::info!(
            "Added edge: {} -> {} ({})",
            params.source_node_id,
            params.target_node_id,
            params.name
        );

        Self::text_result(format!(
            "Successfully added edge '{}' from {} to {}",
            params.name, params.source_node_id, params.target_node_id
        ))
    }

    async fn handle_add_node_fragment(&self, params: JsonValue) -> Result<CallToolResult, CallToolError> {
        let params: AddNodeFragmentParams = Self::parse_params(&params)
            .map_err(|e| CallToolError::new(e))?;

        let id = Id::from_str(&params.id)
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Invalid node ID: {}", e))))?;

        let mutation = AddNodeFragment {
            id,
            ts_millis: TimestampMilli(params.ts_millis.unwrap_or_else(|| TimestampMilli::now().0)),
            content: DataUrl::from_text(&params.content),
            temporal_range: params.temporal_range.map(Self::to_schema_temporal_range),
        };

        mutation
            .run(&self.writer)
            .await
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Failed to add node fragment: {}", e))))?;

        log::info!("Added fragment to node: {}", params.id);

        Self::text_result(format!("Successfully added fragment to node {}", params.id))
    }

    async fn handle_add_edge_fragment(&self, params: JsonValue) -> Result<CallToolResult, CallToolError> {
        let params: AddEdgeFragmentParams = Self::parse_params(&params)
            .map_err(|e| CallToolError::new(e))?;

        let src_id = Id::from_str(&params.src_id)
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Invalid source node ID: {}", e))))?;
        let dst_id = Id::from_str(&params.dst_id)
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Invalid destination node ID: {}", e))))?;

        let mutation = AddEdgeFragment {
            src_id,
            dst_id,
            edge_name: params.edge_name.clone(),
            ts_millis: TimestampMilli(params.ts_millis.unwrap_or_else(|| TimestampMilli::now().0)),
            content: DataUrl::from_text(&params.content),
            temporal_range: params.temporal_range.map(Self::to_schema_temporal_range),
        };

        mutation
            .run(&self.writer)
            .await
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Failed to add edge fragment: {}", e))))?;

        log::info!(
            "Added fragment to edge: {} -> {} ({})",
            params.src_id,
            params.dst_id,
            params.edge_name
        );

        Self::text_result(format!(
            "Successfully added fragment to edge {} -> {} ({})",
            params.src_id, params.dst_id, params.edge_name
        ))
    }

    async fn handle_update_node_valid_range(&self, params: JsonValue) -> Result<CallToolResult, CallToolError> {
        let params: UpdateNodeValidRangeParams = Self::parse_params(&params)
            .map_err(|e| CallToolError::new(e))?;

        let id = Id::from_str(&params.id)
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Invalid node ID: {}", e))))?;

        let mutation = UpdateNodeValidSinceUntil {
            id,
            temporal_range: Self::to_schema_temporal_range(params.temporal_range),
            reason: params.reason.clone(),
        };

        mutation
            .run(&self.writer)
            .await
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Failed to update node validity: {}", e))))?;

        log::info!("Updated validity range for node: {} ({})", params.id, params.reason);

        Self::text_result(format!("Successfully updated validity range for node {}", params.id))
    }

    async fn handle_update_edge_valid_range(&self, params: JsonValue) -> Result<CallToolResult, CallToolError> {
        let params: UpdateEdgeValidRangeParams = Self::parse_params(&params)
            .map_err(|e| CallToolError::new(e))?;

        let src_id = Id::from_str(&params.src_id)
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Invalid source node ID: {}", e))))?;
        let dst_id = Id::from_str(&params.dst_id)
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Invalid destination node ID: {}", e))))?;

        let mutation = UpdateEdgeValidSinceUntil {
            src_id,
            dst_id,
            name: params.name.clone(),
            temporal_range: Self::to_schema_temporal_range(params.temporal_range),
            reason: params.reason.clone(),
        };

        mutation
            .run(&self.writer)
            .await
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Failed to update edge validity: {}", e))))?;

        log::info!(
            "Updated validity range for edge: {} -> {} ({}, {})",
            params.src_id,
            params.dst_id,
            params.name,
            params.reason
        );

        Self::text_result(format!(
            "Successfully updated validity range for edge {} -> {} ({})",
            params.src_id, params.dst_id, params.name
        ))
    }

    async fn handle_update_edge_weight(&self, params: JsonValue) -> Result<CallToolResult, CallToolError> {
        let params: UpdateEdgeWeightParams = Self::parse_params(&params)
            .map_err(|e| CallToolError::new(e))?;

        let src_id = Id::from_str(&params.src_id)
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Invalid source node ID: {}", e))))?;
        let dst_id = Id::from_str(&params.dst_id)
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Invalid destination node ID: {}", e))))?;

        let mutation = UpdateEdgeWeight {
            src_id,
            dst_id,
            name: params.name.clone(),
            weight: params.weight,
        };

        mutation
            .run(&self.writer)
            .await
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Failed to update edge weight: {}", e))))?;

        log::info!(
            "Updated weight for edge: {} -> {} ({}) = {}",
            params.src_id,
            params.dst_id,
            params.name,
            params.weight
        );

        Self::text_result(format!(
            "Successfully updated weight for edge {} -> {} ({}) to {}",
            params.src_id, params.dst_id, params.name, params.weight
        ))
    }

    async fn handle_query_node_by_id(&self, params: JsonValue) -> Result<CallToolResult, CallToolError> {
        let params: QueryNodeByIdParams = Self::parse_params(&params)
            .map_err(|e| CallToolError::new(e))?;

        let id = Id::from_str(&params.id)
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Invalid node ID: {}", e))))?;

        let query = NodeById::new(id, params.reference_ts_millis.map(TimestampMilli));

        let (name, summary) = query
            .run(&self.reader, self.query_timeout)
            .await
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Failed to query node: {}", e))))?;

        let summary_text = summary
            .decode_string()
            .unwrap_or_else(|_| "Unable to decode summary".to_string());

        log::info!("Queried node: {} ({})", params.id, name);

        Self::text_result(format!(
            "Node: {}\nName: {}\nSummary: {}",
            params.id, name, summary_text
        ))
    }

    async fn handle_query_edge(&self, params: JsonValue) -> Result<CallToolResult, CallToolError> {
        let params: QueryEdgeParams = Self::parse_params(&params)
            .map_err(|e| CallToolError::new(e))?;

        let source_id = Id::from_str(&params.source_id)
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Invalid source node ID: {}", e))))?;
        let dest_id = Id::from_str(&params.dest_id)
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Invalid destination node ID: {}", e))))?;

        let query = EdgeSummaryBySrcDstName::new(
            source_id,
            dest_id,
            params.name.clone(),
            params.reference_ts_millis.map(TimestampMilli),
        );

        let (summary, weight) = query
            .run(&self.reader, self.query_timeout)
            .await
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Failed to query edge: {}", e))))?;

        let summary_text = summary
            .decode_string()
            .unwrap_or_else(|_| "Unable to decode summary".to_string());

        let weight_text = weight.map_or("None".to_string(), |w| w.to_string());

        log::info!(
            "Queried edge: {} -> {} ({})",
            params.source_id,
            params.dest_id,
            params.name
        );

        Self::text_result(format!(
            "Edge: {} -> {} ({})\nSummary: {}\nWeight: {}",
            params.source_id, params.dest_id, params.name, summary_text, weight_text
        ))
    }

    async fn handle_query_outgoing_edges(&self, params: JsonValue) -> Result<CallToolResult, CallToolError> {
        let params: QueryOutgoingEdgesParams = Self::parse_params(&params)
            .map_err(|e| CallToolError::new(e))?;

        let id = Id::from_str(&params.id)
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Invalid node ID: {}", e))))?;

        let query = OutgoingEdges::new(id, params.reference_ts_millis.map(TimestampMilli));

        let edges = query
            .run(&self.reader, self.query_timeout)
            .await
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Failed to query outgoing edges: {}", e))))?;

        log::info!("Queried outgoing edges from node: {} (found {})", params.id, edges.len());

        let text = if edges.is_empty() {
            format!("No outgoing edges found for node {}", params.id)
        } else {
            let mut result = format!("Outgoing edges from node {}:\n\n", params.id);
            for (weight, _src, dst, name) in edges {
                let weight_text = weight.map_or("None".to_string(), |w| w.to_string());
                result.push_str(&format!(
                    "  -> {} ({}), weight: {}\n",
                    dst.as_str(),
                    name,
                    weight_text
                ));
            }
            result
        };

        Self::text_result(text)
    }

    async fn handle_query_incoming_edges(&self, params: JsonValue) -> Result<CallToolResult, CallToolError> {
        let params: QueryIncomingEdgesParams = Self::parse_params(&params)
            .map_err(|e| CallToolError::new(e))?;

        let id = Id::from_str(&params.id)
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Invalid node ID: {}", e))))?;

        let query = IncomingEdges::new(id, params.reference_ts_millis.map(TimestampMilli));

        let edges = query
            .run(&self.reader, self.query_timeout)
            .await
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Failed to query incoming edges: {}", e))))?;

        log::info!("Queried incoming edges to node: {} (found {})", params.id, edges.len());

        let text = if edges.is_empty() {
            format!("No incoming edges found for node {}", params.id)
        } else {
            let mut result = format!("Incoming edges to node {}:\n\n", params.id);
            for (weight, _dst, src, name) in edges {
                let weight_text = weight.map_or("None".to_string(), |w| w.to_string());
                result.push_str(&format!(
                    "  {} ({}) ->, weight: {}\n",
                    src.as_str(),
                    name,
                    weight_text
                ));
            }
            result
        };

        Self::text_result(text)
    }

    async fn handle_query_nodes_by_name(&self, params: JsonValue) -> Result<CallToolResult, CallToolError> {
        let params: QueryNodesByNameParams = Self::parse_params(&params)
            .map_err(|e| CallToolError::new(e))?;

        let query = NodesByName::new(
            params.name.clone(),
            None,
            params.limit,
            params.reference_ts_millis.map(TimestampMilli),
        );

        let nodes = query
            .run(&self.reader, self.query_timeout)
            .await
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Failed to query nodes by name: {}", e))))?;

        log::info!(
            "Queried nodes by name: '{}' (found {})",
            params.name,
            nodes.len()
        );

        let text = if nodes.is_empty() {
            format!("No nodes found matching '{}'", params.name)
        } else {
            let mut result = format!("Nodes matching '{}':\n\n", params.name);
            for (name, id) in nodes {
                result.push_str(&format!("  {} ({})\n", name, id.as_str()));
            }
            result
        };

        Self::text_result(text)
    }

    async fn handle_query_edges_by_name(&self, params: JsonValue) -> Result<CallToolResult, CallToolError> {
        let params: QueryEdgesByNameParams = Self::parse_params(&params)
            .map_err(|e| CallToolError::new(e))?;

        let query = EdgesByName::new(
            params.name.clone(),
            None,
            params.limit,
            params.reference_ts_millis.map(TimestampMilli),
        );

        let edges = query
            .run(&self.reader, self.query_timeout)
            .await
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Failed to query edges by name: {}", e))))?;

        log::info!(
            "Queried edges by name: '{}' (found {})",
            params.name,
            edges.len()
        );

        let text = if edges.is_empty() {
            format!("No edges found matching '{}'", params.name)
        } else {
            let mut result = format!("Edges matching '{}':\n\n", params.name);
            for (name, id) in edges {
                result.push_str(&format!("  {} (ID: {})\n", name, id.as_str()));
            }
            result
        };

        Self::text_result(text)
    }

    async fn handle_query_node_fragments(&self, params: JsonValue) -> Result<CallToolResult, CallToolError> {
        let params: QueryNodeFragmentsParams = Self::parse_params(&params)
            .map_err(|e| CallToolError::new(e))?;

        let id = Id::from_str(&params.id)
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Invalid node ID: {}", e))))?;

        let start_bound = params
            .start_ts_millis
            .map_or(Bound::Unbounded, |ts| Bound::Included(TimestampMilli(ts)));
        let end_bound = params
            .end_ts_millis
            .map_or(Bound::Unbounded, |ts| Bound::Included(TimestampMilli(ts)));

        let query = NodeFragmentsByIdTimeRange::new(
            id,
            (start_bound, end_bound),
            params.reference_ts_millis.map(TimestampMilli),
        );

        let fragments = query
            .run(&self.reader, self.query_timeout)
            .await
            .map_err(|e| CallToolError::new(RpcError::internal_error().with_message(format!("Failed to query node fragments: {}", e))))?;

        log::info!(
            "Queried node fragments for node: {} (found {})",
            params.id,
            fragments.len()
        );

        let text = if fragments.is_empty() {
            format!(
                "No fragments found for node {} in the specified time range",
                params.id
            )
        } else {
            let mut result = format!("Fragments for node {}:\n\n", params.id);
            for (ts, content) in fragments {
                let content_text = content
                    .decode_string()
                    .unwrap_or_else(|_| "Unable to decode content".to_string());
                result.push_str(&format!("  [{}] {}\n", ts.0, content_text));
            }
            result
        };

        Self::text_result(text)
    }
}
