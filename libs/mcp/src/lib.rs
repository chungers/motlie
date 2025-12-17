//! MCP server library for Motlie graph database
//!
//! This library provides an MCP (Model Context Protocol) server that exposes
//! the Motlie graph database mutation and query APIs as MCP tools using the rmcp SDK.
//!
//! # Transports
//!
//! The server supports two transport modes:
//! - **stdio**: Standard input/output for local process communication
//! - **http**: HTTP with Streamable HTTP protocol (using axum)

pub mod http;
pub mod types;

use motlie_db::mutation::{
    AddEdge, AddEdgeFragment, AddNode, AddNodeFragment, EdgeSummary, NodeSummary, Runnable as MutationRunnable,
    UpdateEdgeValidSinceUntil, UpdateEdgeWeight, UpdateNodeValidSinceUntil,
};
use motlie_db::query::{
    EdgeFragments, Edges, IncomingEdges, NodeById, NodeFragments, Nodes, OutgoingEdges,
    Runnable as QueryRunnable, EdgeDetails,
};
use motlie_db::reader::Reader;
use motlie_db::writer::Writer;
use motlie_db::{DataUrl, Id, TemporalRange, TimestampMilli};
use rmcp::{
    handler::server::tool::ToolRouter, handler::server::wrapper::Parameters, model::*, tool,
    tool_handler, tool_router, ErrorData as McpError, ServerHandler,
};
use serde_json::json;
use std::ops::Bound;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::OnceCell;
use types::*;

/// Database initialization function type
pub type DbInitFn = Box<dyn FnOnce() -> anyhow::Result<(Writer, Reader)> + Send + Sync>;

/// Lazy database holder for deferred initialization
pub struct LazyDatabase {
    init_fn: std::sync::Mutex<Option<DbInitFn>>,
    writer: OnceCell<Writer>,
    reader: OnceCell<Reader>,
}

impl LazyDatabase {
    /// Create a new lazy database with an initialization function
    pub fn new(init_fn: DbInitFn) -> Self {
        Self {
            init_fn: std::sync::Mutex::new(Some(init_fn)),
            writer: OnceCell::new(),
            reader: OnceCell::new(),
        }
    }

    /// Get or initialize the writer
    pub async fn writer(&self) -> Result<&Writer, McpError> {
        self.ensure_initialized().await?;
        self.writer
            .get()
            .ok_or_else(|| McpError::internal_error("Writer not initialized", None))
    }

    /// Get or initialize the reader
    pub async fn reader(&self) -> Result<&Reader, McpError> {
        self.ensure_initialized().await?;
        self.reader
            .get()
            .ok_or_else(|| McpError::internal_error("Reader not initialized", None))
    }

    async fn ensure_initialized(&self) -> Result<(), McpError> {
        // Check if already initialized
        if self.writer.initialized() && self.reader.initialized() {
            return Ok(());
        }

        // Take the init function (only runs once)
        let init_fn = {
            let mut guard = self.init_fn.lock().unwrap();
            guard.take()
        };

        if let Some(init_fn) = init_fn {
            tracing::info!("Initializing database (lazy initialization on first tool use)...");
            let (writer, reader) = init_fn().map_err(|e| {
                McpError::internal_error(format!("Database initialization failed: {}", e), None)
            })?;
            let _ = self.writer.set(writer);
            let _ = self.reader.set(reader);
            tracing::info!("Database initialization complete");
        }

        Ok(())
    }
}

/// MCP Server for Motlie graph database using rmcp SDK
#[derive(Clone)]
pub struct MotlieMcpServer {
    db: Arc<LazyDatabase>,
    query_timeout: Duration,
    tool_router: ToolRouter<Self>,
}

impl MotlieMcpServer {
    /// Convert temporal range parameter to schema type
    fn to_schema_temporal_range(param: TemporalRangeParam) -> TemporalRange {
        TemporalRange(
            Some(TimestampMilli(param.valid_since)),
            Some(TimestampMilli(param.valid_until)),
        )
    }

    /// Convert content with optional MIME type and base64 flag to DataUrl
    fn content_to_dataurl(content: &str, mime_type: Option<&str>, base64_encoded: bool) -> DataUrl {
        let mime = mime_type.unwrap_or("text/plain");

        if base64_encoded {
            // Content is already base64-encoded, construct data URL directly
            DataUrl::from_raw(format!("data:{};base64,{}", mime, content))
        } else {
            // Plain text content - use appropriate constructor based on MIME type
            match mime {
                "text/markdown" => DataUrl::from_markdown(content),
                "application/json" => DataUrl::from_json(content),
                "text/html" => DataUrl::from_html(content),
                _ => DataUrl::from_text(content),
            }
        }
    }
}

#[tool_router]
impl MotlieMcpServer {
    /// Create a new MCP server instance with lazy database initialization
    pub fn new(db: Arc<LazyDatabase>, query_timeout: Duration) -> Self {
        Self {
            db,
            query_timeout,
            tool_router: Self::tool_router(),
        }
    }

    #[tool(
        description = "Create a new node in the graph with name and optional temporal validity range (ID is auto-generated)"
    )]
    async fn add_node(
        &self,
        Parameters(params): Parameters<AddNodeParams>,
    ) -> Result<CallToolResult, McpError> {
        let writer = self.db.writer().await?;
        let id = Id::new();

        let mutation = AddNode {
            id,
            name: params.name.clone(),
            ts_millis: TimestampMilli(params.ts_millis.unwrap_or_else(|| TimestampMilli::now().0)),
            valid_range: params.temporal_range.map(Self::to_schema_temporal_range),
            summary: NodeSummary::from_text(&params.summary),
        };

        mutation
            .run(writer)
            .await
            .map_err(|e| McpError::internal_error(format!("Failed to add node: {}", e), None))?;

        tracing::info!("Added node: {} ({})", params.name, id);

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&json!({
                "success": true,
                "message": format!("Successfully added node '{}' with ID {}", params.name, id.as_str()),
                "node_id": id.as_str(),
                "node_name": params.name
            })).unwrap()
        )]))
    }

    #[tool(
        description = "Create an edge between two nodes with optional weight and temporal validity"
    )]
    async fn add_edge(
        &self,
        Parameters(params): Parameters<AddEdgeParams>,
    ) -> Result<CallToolResult, McpError> {
        let writer = self.db.writer().await?;

        let source_id = Id::from_str(&params.source_node_id).map_err(|e| {
            McpError::invalid_params(format!("Invalid source node ID: {}", e), None)
        })?;
        let target_id = Id::from_str(&params.target_node_id).map_err(|e| {
            McpError::invalid_params(format!("Invalid target node ID: {}", e), None)
        })?;

        let mutation = AddEdge {
            source_node_id: source_id,
            target_node_id: target_id,
            ts_millis: TimestampMilli(params.ts_millis.unwrap_or_else(|| TimestampMilli::now().0)),
            name: params.name.clone(),
            valid_range: params.temporal_range.map(Self::to_schema_temporal_range),
            summary: EdgeSummary::from_text(&params.summary),
            weight: params.weight,
        };

        mutation
            .run(writer)
            .await
            .map_err(|e| McpError::internal_error(format!("Failed to add edge: {}", e), None))?;

        tracing::info!(
            "Added edge: {} -> {} ({})",
            params.source_node_id,
            params.target_node_id,
            params.name
        );

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&json!({
                "success": true,
                "message": format!("Successfully added edge '{}' from {} to {}", params.name, params.source_node_id, params.target_node_id),
                "edge_name": params.name,
                "source_id": params.source_node_id,
                "target_id": params.target_node_id
            })).unwrap()
        )]))
    }

    #[tool(description = "Add a text content fragment to an existing node")]
    async fn add_node_fragment(
        &self,
        Parameters(params): Parameters<AddNodeFragmentParams>,
    ) -> Result<CallToolResult, McpError> {
        let writer = self.db.writer().await?;

        let id = Id::from_str(&params.id)
            .map_err(|e| McpError::invalid_params(format!("Invalid node ID: {}", e), None))?;

        let mutation = AddNodeFragment {
            id,
            ts_millis: TimestampMilli(params.ts_millis.unwrap_or_else(|| TimestampMilli::now().0)),
            content: DataUrl::from_text(&params.content),
            valid_range: params.temporal_range.map(Self::to_schema_temporal_range),
        };

        mutation.run(writer).await.map_err(|e| {
            McpError::internal_error(format!("Failed to add node fragment: {}", e), None)
        })?;

        tracing::info!("Added fragment to node: {}", params.id);

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&json!({
                "success": true,
                "message": format!("Successfully added fragment to node {}", params.id),
                "node_id": params.id
            }))
            .unwrap(),
        )]))
    }

    #[tool(description = "Add a text content fragment to an existing edge")]
    async fn add_edge_fragment(
        &self,
        Parameters(params): Parameters<AddEdgeFragmentParams>,
    ) -> Result<CallToolResult, McpError> {
        let writer = self.db.writer().await?;

        let src_id = Id::from_str(&params.src_id).map_err(|e| {
            McpError::invalid_params(format!("Invalid source node ID: {}", e), None)
        })?;
        let dst_id = Id::from_str(&params.dst_id).map_err(|e| {
            McpError::invalid_params(format!("Invalid destination node ID: {}", e), None)
        })?;

        let mutation = AddEdgeFragment {
            src_id,
            dst_id,
            edge_name: params.edge_name.clone(),
            ts_millis: TimestampMilli(params.ts_millis.unwrap_or_else(|| TimestampMilli::now().0)),
            content: DataUrl::from_text(&params.content),
            valid_range: params.temporal_range.map(Self::to_schema_temporal_range),
        };

        mutation.run(writer).await.map_err(|e| {
            McpError::internal_error(format!("Failed to add edge fragment: {}", e), None)
        })?;

        tracing::info!(
            "Added fragment to edge: {} -> {} ({})",
            params.src_id,
            params.dst_id,
            params.edge_name
        );

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&json!({
                "success": true,
                "message": format!("Successfully added fragment to edge {} -> {} ({})", params.src_id, params.dst_id, params.edge_name),
                "src_id": params.src_id,
                "dst_id": params.dst_id,
                "edge_name": params.edge_name
            })).unwrap()
        )]))
    }

    #[tool(description = "Update the temporal validity range of a node")]
    async fn update_node_valid_range(
        &self,
        Parameters(params): Parameters<UpdateNodeValidRangeParams>,
    ) -> Result<CallToolResult, McpError> {
        let writer = self.db.writer().await?;

        let id = Id::from_str(&params.id)
            .map_err(|e| McpError::invalid_params(format!("Invalid node ID: {}", e), None))?;

        let mutation = UpdateNodeValidSinceUntil {
            id,
            temporal_range: Self::to_schema_temporal_range(params.temporal_range),
            reason: params.reason.clone(),
        };

        mutation.run(writer).await.map_err(|e| {
            McpError::internal_error(format!("Failed to update node validity: {}", e), None)
        })?;

        tracing::info!(
            "Updated validity range for node: {} ({})",
            params.id,
            params.reason
        );

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&json!({
                "success": true,
                "message": format!("Successfully updated validity range for node {}", params.id),
                "node_id": params.id
            }))
            .unwrap(),
        )]))
    }

    #[tool(description = "Update the temporal validity range of an edge")]
    async fn update_edge_valid_range(
        &self,
        Parameters(params): Parameters<UpdateEdgeValidRangeParams>,
    ) -> Result<CallToolResult, McpError> {
        let writer = self.db.writer().await?;

        let src_id = Id::from_str(&params.src_id).map_err(|e| {
            McpError::invalid_params(format!("Invalid source node ID: {}", e), None)
        })?;
        let dst_id = Id::from_str(&params.dst_id).map_err(|e| {
            McpError::invalid_params(format!("Invalid destination node ID: {}", e), None)
        })?;

        let mutation = UpdateEdgeValidSinceUntil {
            src_id,
            dst_id,
            name: params.name.clone(),
            temporal_range: Self::to_schema_temporal_range(params.temporal_range),
            reason: params.reason.clone(),
        };

        mutation.run(writer).await.map_err(|e| {
            McpError::internal_error(format!("Failed to update edge validity: {}", e), None)
        })?;

        tracing::info!(
            "Updated validity range for edge: {} -> {} ({}, {})",
            params.src_id,
            params.dst_id,
            params.name,
            params.reason
        );

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&json!({
                "success": true,
                "message": format!("Successfully updated validity range for edge {} -> {} ({})", params.src_id, params.dst_id, params.name),
                "src_id": params.src_id,
                "dst_id": params.dst_id,
                "edge_name": params.name
            })).unwrap()
        )]))
    }

    #[tool(description = "Update the weight of an edge for graph algorithms")]
    async fn update_edge_weight(
        &self,
        Parameters(params): Parameters<UpdateEdgeWeightParams>,
    ) -> Result<CallToolResult, McpError> {
        let writer = self.db.writer().await?;

        let src_id = Id::from_str(&params.src_id).map_err(|e| {
            McpError::invalid_params(format!("Invalid source node ID: {}", e), None)
        })?;
        let dst_id = Id::from_str(&params.dst_id).map_err(|e| {
            McpError::invalid_params(format!("Invalid destination node ID: {}", e), None)
        })?;

        let mutation = UpdateEdgeWeight {
            src_id,
            dst_id,
            name: params.name.clone(),
            weight: params.weight,
        };

        mutation.run(writer).await.map_err(|e| {
            McpError::internal_error(format!("Failed to update edge weight: {}", e), None)
        })?;

        tracing::info!(
            "Updated weight for edge: {} -> {} ({}) = {}",
            params.src_id,
            params.dst_id,
            params.name,
            params.weight
        );

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&json!({
                "success": true,
                "message": format!("Successfully updated weight for edge {} -> {} ({}) to {}", params.src_id, params.dst_id, params.name, params.weight),
                "src_id": params.src_id,
                "dst_id": params.dst_id,
                "edge_name": params.name,
                "weight": params.weight
            })).unwrap()
        )]))
    }

    #[tool(description = "Retrieve a node by its ID")]
    async fn query_node_by_id(
        &self,
        Parameters(params): Parameters<QueryNodeByIdParams>,
    ) -> Result<CallToolResult, McpError> {
        let reader = self.db.reader().await?;

        let id = Id::from_str(&params.id)
            .map_err(|e| McpError::invalid_params(format!("Invalid node ID: {}", e), None))?;

        let query = NodeById::new(id, params.reference_ts_millis.map(TimestampMilli));

        let (name, summary) = query
            .run(reader, self.query_timeout)
            .await
            .map_err(|e| McpError::internal_error(format!("Failed to query node: {}", e), None))?;

        let summary_text = summary
            .decode_string()
            .unwrap_or_else(|_| "Unable to decode summary".to_string());

        tracing::info!("Queried node: {} ({})", params.id, name);

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&json!({
                "node_id": params.id,
                "name": name,
                "summary": summary_text
            }))
            .unwrap(),
        )]))
    }

    #[tool(description = "Retrieve an edge by its source, destination, and name")]
    async fn query_edge(
        &self,
        Parameters(params): Parameters<QueryEdgeParams>,
    ) -> Result<CallToolResult, McpError> {
        let reader = self.db.reader().await?;

        let source_id = Id::from_str(&params.source_id).map_err(|e| {
            McpError::invalid_params(format!("Invalid source node ID: {}", e), None)
        })?;
        let dest_id = Id::from_str(&params.dest_id).map_err(|e| {
            McpError::invalid_params(format!("Invalid destination node ID: {}", e), None)
        })?;

        let query = EdgeDetails::new(
            source_id,
            dest_id,
            params.name.clone(),
            params.reference_ts_millis.map(TimestampMilli),
        );

        let (weight, _src, _dst, _name, summary): (Option<f64>, Id, Id, String, EdgeSummary) = query
            .run(reader, self.query_timeout)
            .await
            .map_err(|e| McpError::internal_error(format!("Failed to query edge: {}", e), None))?;

        let summary_text = summary
            .decode_string()
            .unwrap_or_else(|_| "Unable to decode summary".to_string());

        tracing::info!(
            "Queried edge: {} -> {} ({})",
            params.source_id,
            params.dest_id,
            params.name
        );

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&json!({
                "source_id": params.source_id,
                "dest_id": params.dest_id,
                "name": params.name,
                "summary": summary_text,
                "weight": weight
            }))
            .unwrap(),
        )]))
    }

    #[tool(description = "Get all outgoing edges from a node")]
    async fn query_outgoing_edges(
        &self,
        Parameters(params): Parameters<QueryOutgoingEdgesParams>,
    ) -> Result<CallToolResult, McpError> {
        let reader = self.db.reader().await?;

        let id = Id::from_str(&params.id)
            .map_err(|e| McpError::invalid_params(format!("Invalid node ID: {}", e), None))?;

        let query = OutgoingEdges::new(id, params.reference_ts_millis.map(TimestampMilli));

        let edges: Vec<(Option<f64>, Id, Id, String)> =
            query.run(reader, self.query_timeout).await.map_err(|e| {
                McpError::internal_error(format!("Failed to query outgoing edges: {}", e), None)
            })?;

        tracing::info!(
            "Queried outgoing edges from node: {} (found {})",
            params.id,
            edges.len()
        );

        let edges_list: Vec<serde_json::Value> = edges
            .into_iter()
            .map(|(weight, _src, dst, name): (Option<f64>, Id, Id, String)| {
                json!({
                    "target_id": dst.as_str(),
                    "name": name,
                    "weight": weight
                })
            })
            .collect();

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&json!({
                "node_id": params.id,
                "edges": edges_list,
                "count": edges_list.len()
            }))
            .unwrap(),
        )]))
    }

    #[tool(description = "Get all incoming edges to a node")]
    async fn query_incoming_edges(
        &self,
        Parameters(params): Parameters<QueryIncomingEdgesParams>,
    ) -> Result<CallToolResult, McpError> {
        let reader = self.db.reader().await?;

        let id = Id::from_str(&params.id)
            .map_err(|e| McpError::invalid_params(format!("Invalid node ID: {}", e), None))?;

        let query = IncomingEdges::new(id, params.reference_ts_millis.map(TimestampMilli));

        let edges: Vec<(Option<f64>, Id, Id, String)> =
            query.run(reader, self.query_timeout).await.map_err(|e| {
                McpError::internal_error(format!("Failed to query incoming edges: {}", e), None)
            })?;

        tracing::info!(
            "Queried incoming edges to node: {} (found {})",
            params.id,
            edges.len()
        );

        let edges_list: Vec<serde_json::Value> = edges
            .into_iter()
            .map(|(weight, _dst, src, name): (Option<f64>, Id, Id, String)| {
                json!({
                    "source_id": src.as_str(),
                    "name": name,
                    "weight": weight
                })
            })
            .collect();

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&json!({
                "node_id": params.id,
                "edges": edges_list,
                "count": edges_list.len()
            }))
            .unwrap(),
        )]))
    }

    #[tool(description = "Search for nodes by name or name prefix")]
    async fn query_nodes_by_name(
        &self,
        Parameters(params): Parameters<QueryNodesByNameParams>,
    ) -> Result<CallToolResult, McpError> {
        let reader = self.db.reader().await?;

        let query = Nodes::new(params.name.clone(), params.limit.unwrap_or(100) as usize);

        let nodes: Vec<(Id, String, NodeSummary)> =
            query.run(reader, self.query_timeout).await.map_err(|e| {
                McpError::internal_error(format!("Failed to query nodes by name: {}", e), None)
            })?;

        tracing::info!(
            "Queried nodes by name: '{}' (found {})",
            params.name,
            nodes.len()
        );

        let nodes_list: Vec<serde_json::Value> = nodes
            .into_iter()
            .map(|(id, name, _summary)| {
                json!({
                    "name": name,
                    "id": id.as_str()
                })
            })
            .collect();

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&json!({
                "search_name": params.name,
                "nodes": nodes_list,
                "count": nodes_list.len()
            }))
            .unwrap(),
        )]))
    }

    #[tool(description = "Search for edges by name or name prefix")]
    async fn query_edges_by_name(
        &self,
        Parameters(params): Parameters<QueryEdgesByNameParams>,
    ) -> Result<CallToolResult, McpError> {
        let reader = self.db.reader().await?;

        let query = Edges::new(params.name.clone(), params.limit.unwrap_or(100) as usize);

        let edges: Vec<(Id, Id, String, EdgeSummary)> =
            query.run(reader, self.query_timeout).await.map_err(|e| {
                McpError::internal_error(format!("Failed to query edges by name: {}", e), None)
            })?;

        tracing::info!(
            "Queried edges by name: '{}' (found {})",
            params.name,
            edges.len()
        );

        let edges_list: Vec<serde_json::Value> = edges
            .into_iter()
            .map(|(src_id, dst_id, name, _summary)| {
                json!({
                    "source_id": src_id.as_str(),
                    "dest_id": dst_id.as_str(),
                    "name": name
                })
            })
            .collect();

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&json!({
                "search_name": params.name,
                "edges": edges_list,
                "count": edges_list.len()
            }))
            .unwrap(),
        )]))
    }

    #[tool(description = "Get content fragments for a node within a time range")]
    async fn query_node_fragments(
        &self,
        Parameters(params): Parameters<QueryNodeFragmentsParams>,
    ) -> Result<CallToolResult, McpError> {
        let reader = self.db.reader().await?;

        let id = Id::from_str(&params.id)
            .map_err(|e| McpError::invalid_params(format!("Invalid node ID: {}", e), None))?;

        let start_bound = params
            .start_ts_millis
            .map_or(Bound::Unbounded, |ts| Bound::Included(TimestampMilli(ts)));
        let end_bound = params
            .end_ts_millis
            .map_or(Bound::Unbounded, |ts| Bound::Included(TimestampMilli(ts)));

        let query = NodeFragments::new(
            id,
            (start_bound, end_bound),
            params.reference_ts_millis.map(TimestampMilli),
        );

        let fragments: Vec<(TimestampMilli, DataUrl)> =
            query.run(reader, self.query_timeout).await.map_err(|e| {
                McpError::internal_error(format!("Failed to query node fragments: {}", e), None)
            })?;

        tracing::info!(
            "Queried node fragments for node: {} (found {})",
            params.id,
            fragments.len()
        );

        let fragments_list: Vec<serde_json::Value> = fragments
            .into_iter()
            .map(|(ts, content): (TimestampMilli, DataUrl)| {
                let content_text = content
                    .decode_string()
                    .unwrap_or_else(|_| "Unable to decode content".to_string());
                json!({
                    "timestamp_millis": ts.0,
                    "content": content_text
                })
            })
            .collect();

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&json!({
                "node_id": params.id,
                "fragments": fragments_list,
                "count": fragments_list.len()
            }))
            .unwrap(),
        )]))
    }

    #[tool(description = "Get content fragments for an edge within a time range")]
    async fn query_edge_fragments(
        &self,
        Parameters(params): Parameters<QueryEdgeFragmentsParams>,
    ) -> Result<CallToolResult, McpError> {
        let reader = self.db.reader().await?;

        let src_id = Id::from_str(&params.src_id).map_err(|e| {
            McpError::invalid_params(format!("Invalid source node ID: {}", e), None)
        })?;
        let dst_id = Id::from_str(&params.dst_id).map_err(|e| {
            McpError::invalid_params(format!("Invalid destination node ID: {}", e), None)
        })?;

        let start_bound = params
            .start_ts_millis
            .map_or(Bound::Unbounded, |ts| Bound::Included(TimestampMilli(ts)));
        let end_bound = params
            .end_ts_millis
            .map_or(Bound::Unbounded, |ts| Bound::Included(TimestampMilli(ts)));

        let query = EdgeFragments::new(
            src_id,
            dst_id,
            params.edge_name.clone(),
            (start_bound, end_bound),
            params.reference_ts_millis.map(TimestampMilli),
        );

        let fragments: Vec<(TimestampMilli, DataUrl)> =
            query.run(reader, self.query_timeout).await.map_err(|e| {
                McpError::internal_error(format!("Failed to query edge fragments: {}", e), None)
            })?;

        tracing::info!(
            "Queried edge fragments for edge {} -> {} '{}' (found {})",
            params.src_id,
            params.dst_id,
            params.edge_name,
            fragments.len()
        );

        let fragments_list: Vec<serde_json::Value> = fragments
            .into_iter()
            .map(|(ts, content): (TimestampMilli, DataUrl)| {
                let content_text = content
                    .decode_string()
                    .unwrap_or_else(|_| "Unable to decode content".to_string());
                json!({
                    "timestamp_millis": ts.0,
                    "content": content_text
                })
            })
            .collect();

        Ok(CallToolResult::success(vec![Content::text(
            serde_json::to_string(&json!({
                "src_id": params.src_id,
                "dst_id": params.dst_id,
                "edge_name": params.edge_name,
                "fragments": fragments_list,
                "count": fragments_list.len()
            }))
            .unwrap(),
        )]))
    }
}

// Server handler implementation
#[tool_handler]
impl ServerHandler for MotlieMcpServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation::from_build_env(),
            instructions: Some(INFO_TEXT.to_string()),
        }
    }
}

pub const INFO_TEXT: &str = INFO_TEXT_V2;

const INFO_TEXT_V1: &str = "
This server provides tools to interact with the Motlie graph database. \
It supports 15 tools:
  - 7 mutation operations (add_node, add_edge, add_node_fragment,
  add_edge_fragment, update_node_valid_range, update_edge_valid_range, update_edge_weight
  - 8 query operations (query_node_by_id, query_edge, query_outgoing_edges,
  query_incoming_edges, query_nodes_by_name, query_edges_by_name, query_node_fragments,
  query_edge_fragments). \
  - The database does not support deletion of nodes or edges but allows setting validity
  time range (since and until timestamps). For example, if a node should disappear after
  a certain time, set its 'valid until' timestamp via the update_node_valid_range tool.\
  - Timestamps are in milliseconds since Unix epoch. \
  - To avoid duplication, it's useful to query an object by name first and confirm with the
  user before calling the add_node or add_edge tools.
  - Useful contexts about a node or edge can be added by calling the add_node_fragment or
  add_edge_fragment tools.
  - The fragments for a node or edge serve as a chronological history or a log of
  useful tidbits, events, or state changes associated with the node or edge.
  - If the content of a fragment is long and contains concepts and topics outside the scope
  of the node or edge, it's recommended to create a new node and a new edge as the
  relationship between the two. \
";

const INFO_TEXT_V2: &str = "
This server provides tools to interact with the Motlie graph database.
  - Timestamps are in milliseconds since Unix epoch. \
  - Nodes are entities, Edges are relationships.
    - To avoid duplication, ALWAYS query an object by name first (if its ID is not in the context).
    Confirm with theuser before calling the add_node tool to model a new entity unseen before.  Nodes
    cannot be deleted after creation.
    - Relationships (edges) can have different names but similar meanings. Reuse string names when possible.
  - Fragments model the properties, attributes, and state changes of entities (nodes) and relationships (edges).
  - Properties, attributes, state changes of an entity or relationship can be added as fragments,
  by calling the add_node_fragment or add_edge_fragment tools.
    - A node or edge's fragments form a queryable history by time range.
    - Fragments are ONLY for contexts, properties, and attributes about an entity or
    a relationship.v
  - No deletion of nodes or edges. ONLY invalidation by setting a validity time range (since and until timestamps).
    - For example, if an event as a node (NeurIPS 2023) should disappear after a certain time
    (e.g. 2024-01-01), set its 'valid until' timestamp via the update_node_valid_range tool.\
 IMPORTANT RULES ABOUT GRAPH MODEL
  - Think hard about how you model the user's text.  Consider the RDF model in analyzing the user's text:
  subject, predicate, object. Subject and object are entities (nodes), while predicate / action forms
  a relationship (edge) between them.
  - Fragments are ONLY for contexts, properties, and attributes (for example, age, location,
  time became active, time became inactive, etc.).
  - A fragment MUST NOT be used to store content that contains concepts outside the scope of
  the entity or relationship.  Prefer creating a new node and edge as the relationship
  between the two, over using fragments.
  - Example: Johnny Rabbit loves ice cream in the summer.  He is 7 years old. He started Reed Elementary School in Sept 2025.
      - [BEST] Graph 1
        - Node1 = Johnny, NodeFragment1_1 = 7 years old. NodeFragment1_2 = Last name is Rabbit,
        - Node2 = Ice Cream, Edge2 = Johnny --[ loves ]--> Ice Cream.  EdgeFragment2_1 = in the summer
        - Node3 = Reed Elementary School, Edge3 = Johnny -- [Started] --> Reed Elementary School, EdgeFragment3_1 = started in 09/2025
      - [ACCEPTABLE] Graph 2
        - Node1 = Johnny Rabbit, NodeFragment1_1 = 7 years old,
        - Node2 = Ice Cream, Edge2 = Johnny Rabbit --[ loves ]--> Ice Cream.  EdgeFragment2_1 = in the summer
        - Node3 = Reed Elementary School, Edge3 = Johnny Rabbit -- [Started] --> Reed Elementary School, EdgeFragment3_1 = 2025
      - [BAD] Graph 3
        - Node1 = Johnny Rabbit, NodeFragment1_1 = 7 years old. Loves ice cream and started Reed Elementary School in 2025,
  ";

pub use http::{serve_http, HttpConfig};
/// Re-export for convenience
pub use rmcp::{transport::stdio, ServiceExt};
