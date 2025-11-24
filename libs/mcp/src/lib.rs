//! MCP server library for Motlie graph database
//!
//! This library provides an MCP (Model Context Protocol) server that exposes
//! the Motlie graph database mutation and query APIs as MCP tools using the pmcp SDK.

pub mod types;

use motlie_db::{
    AddEdge, AddEdgeFragment, AddNode, AddNodeFragment, DataUrl, EdgeSummary,
    EdgesByName, EdgeSummaryBySrcDstName, Id, IncomingEdges, MutationRunnable,
    NodeById, NodeFragmentsByIdTimeRange, NodesByName, OutgoingEdges,
    QueryRunnable, Reader, TimestampMilli, UpdateEdgeValidSinceUntil,
    UpdateEdgeWeight, UpdateNodeValidSinceUntil, ValidTemporalRange, Writer,
};
use pmcp::{ServerBuilder, ServerCapabilities, TypedTool};
use serde_json::{json, Value as JsonValue};
use std::ops::Bound;
use std::sync::Arc;
use std::time::Duration;
use types::*;

/// MCP Server for Motlie graph database
///
/// This server wraps the Motlie database's Writer and Reader interfaces
/// and exposes them as MCP tools using the pmcp SDK.
pub struct MotlieMcpServer {
    writer: Writer,
    reader: Reader,
    query_timeout: Duration,
}

impl MotlieMcpServer {
    /// Create a new MCP server instance
    ///
    /// # Arguments
    /// * `writer` - Writer for mutation operations
    /// * `reader` - Reader for query operations
    pub fn new(writer: Writer, reader: Reader) -> Self {
        Self {
            writer,
            reader,
            query_timeout: Duration::from_secs(5),
        }
    }

    /// Set custom query timeout
    pub fn with_query_timeout(mut self, timeout: Duration) -> Self {
        self.query_timeout = timeout;
        self
    }

    /// Convert temporal range parameter to schema type
    fn to_schema_temporal_range(param: TemporalRangeParam) -> ValidTemporalRange {
        ValidTemporalRange(
            Some(TimestampMilli(param.valid_since)),
            Some(TimestampMilli(param.valid_until)),
        )
    }

    /// Build the pmcp server with all tools registered
    ///
    /// # Arguments
    /// * `auth_token` - Optional authentication token (if provided, Bearer auth will be required)
    pub fn build_server(
        self,
        auth_token: Option<String>,
    ) -> pmcp::Result<pmcp::Server> {
        let server_self = Arc::new(self);

        let mut builder = ServerBuilder::new()
            .name("motlie-mcp-server")
            .version(env!("CARGO_PKG_VERSION"))
            .capabilities(ServerCapabilities {
                tools: Some(pmcp::types::ToolCapabilities::default()),
                ..Default::default()
            });

        // If auth token is provided, tools will check it via RequestHandlerExtra.auth_context
        if let Some(token) = auth_token {
            log::info!("Authentication enabled with Bearer token");
            // Note: pmcp handles authentication at the transport/protocol level
            // The auth_context will be automatically populated in RequestHandlerExtra
            // Tools can access it via extra.auth_context
            // We store the expected token for validation in tool handlers
            std::env::set_var("MOTLIE_MCP_AUTH_TOKEN", token);
        }

        // Register mutation tools
        builder = builder
            .tool(
                "add_node",
                TypedTool::new("add_node", {
                    let server = Arc::clone(&server_self);
                    move |args: AddNodeParams, extra| {
                        let server = Arc::clone(&server);
                        Box::pin(async move {
                            Self::authenticate(&extra)?;

                            let id = Id::from_str(&args.id).map_err(|e| {
                                pmcp::Error::validation(format!("Invalid node ID: {}", e))
                            })?;

                            let mutation = AddNode {
                                id,
                                name: args.name.clone(),
                                ts_millis: TimestampMilli(
                                    args.ts_millis.unwrap_or_else(|| TimestampMilli::now().0),
                                ),
                                temporal_range: args.temporal_range.map(Self::to_schema_temporal_range),
                            };

                            mutation.run(&server.writer).await.map_err(|e| {
                                pmcp::Error::internal(format!("Failed to add node: {}", e))
                            })?;

                            log::info!("Added node: {} ({})", args.name, id);

                            Ok(json!({
                                "success": true,
                                "message": format!("Successfully added node '{}' with ID {}", args.name, args.id),
                                "node_id": args.id,
                                "node_name": args.name
                            }))
                        })
                    }
                })
                .with_description("Create a new node in the graph with ID, name, and optional temporal validity range"),
            )
            .tool(
                "add_edge",
                TypedTool::new("add_edge", {
                    let server = Arc::clone(&server_self);
                    move |args: AddEdgeParams, extra| {
                        let server = Arc::clone(&server);
                        Box::pin(async move {
                            Self::authenticate(&extra)?;

                            let source_id = Id::from_str(&args.source_node_id).map_err(|e| {
                                pmcp::Error::validation(format!("Invalid source node ID: {}", e))
                            })?;
                            let target_id = Id::from_str(&args.target_node_id).map_err(|e| {
                                pmcp::Error::validation(format!("Invalid target node ID: {}", e))
                            })?;

                            let mutation = AddEdge {
                                source_node_id: source_id,
                                target_node_id: target_id,
                                ts_millis: TimestampMilli(
                                    args.ts_millis.unwrap_or_else(|| TimestampMilli::now().0),
                                ),
                                name: args.name.clone(),
                                temporal_range: args.temporal_range.map(Self::to_schema_temporal_range),
                                summary: EdgeSummary::from_text(&args.summary),
                                weight: args.weight,
                            };

                            mutation.run(&server.writer).await.map_err(|e| {
                                pmcp::Error::internal(format!("Failed to add edge: {}", e))
                            })?;

                            log::info!(
                                "Added edge: {} -> {} ({})",
                                args.source_node_id,
                                args.target_node_id,
                                args.name
                            );

                            Ok(json!({
                                "success": true,
                                "message": format!(
                                    "Successfully added edge '{}' from {} to {}",
                                    args.name, args.source_node_id, args.target_node_id
                                ),
                                "edge_name": args.name,
                                "source_id": args.source_node_id,
                                "target_id": args.target_node_id
                            }))
                        })
                    }
                })
                .with_description("Create an edge between two nodes with optional weight and temporal validity"),
            )
            .tool(
                "add_node_fragment",
                TypedTool::new("add_node_fragment", {
                    let server = Arc::clone(&server_self);
                    move |args: AddNodeFragmentParams, extra| {
                        let server = Arc::clone(&server);
                        Box::pin(async move {
                            Self::authenticate(&extra)?;

                            let id = Id::from_str(&args.id).map_err(|e| {
                                pmcp::Error::validation(format!("Invalid node ID: {}", e))
                            })?;

                            let mutation = AddNodeFragment {
                                id,
                                ts_millis: TimestampMilli(
                                    args.ts_millis.unwrap_or_else(|| TimestampMilli::now().0),
                                ),
                                content: DataUrl::from_text(&args.content),
                                temporal_range: args.temporal_range.map(Self::to_schema_temporal_range),
                            };

                            mutation.run(&server.writer).await.map_err(|e| {
                                pmcp::Error::internal(format!("Failed to add node fragment: {}", e))
                            })?;

                            log::info!("Added fragment to node: {}", args.id);

                            Ok(json!({
                                "success": true,
                                "message": format!("Successfully added fragment to node {}", args.id),
                                "node_id": args.id
                            }))
                        })
                    }
                })
                .with_description("Add a content fragment to an existing node"),
            )
            .tool(
                "add_edge_fragment",
                TypedTool::new("add_edge_fragment", {
                    let server = Arc::clone(&server_self);
                    move |args: AddEdgeFragmentParams, extra| {
                        let server = Arc::clone(&server);
                        Box::pin(async move {
                            Self::authenticate(&extra)?;

                            let src_id = Id::from_str(&args.src_id).map_err(|e| {
                                pmcp::Error::validation(format!("Invalid source node ID: {}", e))
                            })?;
                            let dst_id = Id::from_str(&args.dst_id).map_err(|e| {
                                pmcp::Error::validation(format!("Invalid destination node ID: {}", e))
                            })?;

                            let mutation = AddEdgeFragment {
                                src_id,
                                dst_id,
                                edge_name: args.edge_name.clone(),
                                ts_millis: TimestampMilli(
                                    args.ts_millis.unwrap_or_else(|| TimestampMilli::now().0),
                                ),
                                content: DataUrl::from_text(&args.content),
                                temporal_range: args.temporal_range.map(Self::to_schema_temporal_range),
                            };

                            mutation.run(&server.writer).await.map_err(|e| {
                                pmcp::Error::internal(format!("Failed to add edge fragment: {}", e))
                            })?;

                            log::info!(
                                "Added fragment to edge: {} -> {} ({})",
                                args.src_id,
                                args.dst_id,
                                args.edge_name
                            );

                            Ok(json!({
                                "success": true,
                                "message": format!(
                                    "Successfully added fragment to edge {} -> {} ({})",
                                    args.src_id, args.dst_id, args.edge_name
                                ),
                                "src_id": args.src_id,
                                "dst_id": args.dst_id,
                                "edge_name": args.edge_name
                            }))
                        })
                    }
                })
                .with_description("Add a content fragment to an existing edge"),
            )
            .tool(
                "update_node_valid_range",
                TypedTool::new("update_node_valid_range", {
                    let server = Arc::clone(&server_self);
                    move |args: UpdateNodeValidRangeParams, extra| {
                        let server = Arc::clone(&server);
                        Box::pin(async move {
                            Self::authenticate(&extra)?;

                            let id = Id::from_str(&args.id).map_err(|e| {
                                pmcp::Error::validation(format!("Invalid node ID: {}", e))
                            })?;

                            let mutation = UpdateNodeValidSinceUntil {
                                id,
                                temporal_range: Self::to_schema_temporal_range(args.temporal_range),
                                reason: args.reason.clone(),
                            };

                            mutation.run(&server.writer).await.map_err(|e| {
                                pmcp::Error::internal(format!("Failed to update node validity: {}", e))
                            })?;

                            log::info!("Updated validity range for node: {} ({})", args.id, args.reason);

                            Ok(json!({
                                "success": true,
                                "message": format!("Successfully updated validity range for node {}", args.id),
                                "node_id": args.id
                            }))
                        })
                    }
                })
                .with_description("Update the temporal validity range of a node"),
            )
            .tool(
                "update_edge_valid_range",
                TypedTool::new("update_edge_valid_range", {
                    let server = Arc::clone(&server_self);
                    move |args: UpdateEdgeValidRangeParams, extra| {
                        let server = Arc::clone(&server);
                        Box::pin(async move {
                            Self::authenticate(&extra)?;

                            let src_id = Id::from_str(&args.src_id).map_err(|e| {
                                pmcp::Error::validation(format!("Invalid source node ID: {}", e))
                            })?;
                            let dst_id = Id::from_str(&args.dst_id).map_err(|e| {
                                pmcp::Error::validation(format!("Invalid destination node ID: {}", e))
                            })?;

                            let mutation = UpdateEdgeValidSinceUntil {
                                src_id,
                                dst_id,
                                name: args.name.clone(),
                                temporal_range: Self::to_schema_temporal_range(args.temporal_range),
                                reason: args.reason.clone(),
                            };

                            mutation.run(&server.writer).await.map_err(|e| {
                                pmcp::Error::internal(format!("Failed to update edge validity: {}", e))
                            })?;

                            log::info!(
                                "Updated validity range for edge: {} -> {} ({}, {})",
                                args.src_id,
                                args.dst_id,
                                args.name,
                                args.reason
                            );

                            Ok(json!({
                                "success": true,
                                "message": format!(
                                    "Successfully updated validity range for edge {} -> {} ({})",
                                    args.src_id, args.dst_id, args.name
                                ),
                                "src_id": args.src_id,
                                "dst_id": args.dst_id,
                                "edge_name": args.name
                            }))
                        })
                    }
                })
                .with_description("Update the temporal validity range of an edge"),
            )
            .tool(
                "update_edge_weight",
                TypedTool::new("update_edge_weight", {
                    let server = Arc::clone(&server_self);
                    move |args: UpdateEdgeWeightParams, extra| {
                        let server = Arc::clone(&server);
                        Box::pin(async move {
                            Self::authenticate(&extra)?;

                            let src_id = Id::from_str(&args.src_id).map_err(|e| {
                                pmcp::Error::validation(format!("Invalid source node ID: {}", e))
                            })?;
                            let dst_id = Id::from_str(&args.dst_id).map_err(|e| {
                                pmcp::Error::validation(format!("Invalid destination node ID: {}", e))
                            })?;

                            let mutation = UpdateEdgeWeight {
                                src_id,
                                dst_id,
                                name: args.name.clone(),
                                weight: args.weight,
                            };

                            mutation.run(&server.writer).await.map_err(|e| {
                                pmcp::Error::internal(format!("Failed to update edge weight: {}", e))
                            })?;

                            log::info!(
                                "Updated weight for edge: {} -> {} ({}) = {}",
                                args.src_id,
                                args.dst_id,
                                args.name,
                                args.weight
                            );

                            Ok(json!({
                                "success": true,
                                "message": format!(
                                    "Successfully updated weight for edge {} -> {} ({}) to {}",
                                    args.src_id, args.dst_id, args.name, args.weight
                                ),
                                "src_id": args.src_id,
                                "dst_id": args.dst_id,
                                "edge_name": args.name,
                                "weight": args.weight
                            }))
                        })
                    }
                })
                .with_description("Update the weight of an edge for graph algorithms"),
            );

        // Register query tools
        builder = builder
            .tool(
                "query_node_by_id",
                TypedTool::new("query_node_by_id", {
                    let server = Arc::clone(&server_self);
                    move |args: QueryNodeByIdParams, extra| {
                        let server = Arc::clone(&server);
                        Box::pin(async move {
                            Self::authenticate(&extra)?;

                            let id = Id::from_str(&args.id).map_err(|e| {
                                pmcp::Error::validation(format!("Invalid node ID: {}", e))
                            })?;

                            let query = NodeById::new(id, args.reference_ts_millis.map(TimestampMilli));

                            let (name, summary) = query
                                .run(&server.reader, server.query_timeout)
                                .await
                                .map_err(|e| {
                                    pmcp::Error::internal(format!("Failed to query node: {}", e))
                                })?;

                            let summary_text = summary
                                .decode_string()
                                .unwrap_or_else(|_| "Unable to decode summary".to_string());

                            log::info!("Queried node: {} ({})", args.id, name);

                            Ok(json!({
                                "node_id": args.id,
                                "name": name,
                                "summary": summary_text
                            }))
                        })
                    }
                })
                .with_description("Retrieve a node by its ID"),
            )
            .tool(
                "query_edge",
                TypedTool::new("query_edge", {
                    let server = Arc::clone(&server_self);
                    move |args: QueryEdgeParams, extra| {
                        let server = Arc::clone(&server);
                        Box::pin(async move {
                            Self::authenticate(&extra)?;

                            let source_id = Id::from_str(&args.source_id).map_err(|e| {
                                pmcp::Error::validation(format!("Invalid source node ID: {}", e))
                            })?;
                            let dest_id = Id::from_str(&args.dest_id).map_err(|e| {
                                pmcp::Error::validation(format!("Invalid destination node ID: {}", e))
                            })?;

                            let query = EdgeSummaryBySrcDstName::new(
                                source_id,
                                dest_id,
                                args.name.clone(),
                                args.reference_ts_millis.map(TimestampMilli),
                            );

                            let (summary, weight) = query
                                .run(&server.reader, server.query_timeout)
                                .await
                                .map_err(|e| {
                                    pmcp::Error::internal(format!("Failed to query edge: {}", e))
                                })?;

                            let summary_text = summary
                                .decode_string()
                                .unwrap_or_else(|_| "Unable to decode summary".to_string());

                            log::info!(
                                "Queried edge: {} -> {} ({})",
                                args.source_id,
                                args.dest_id,
                                args.name
                            );

                            Ok(json!({
                                "source_id": args.source_id,
                                "dest_id": args.dest_id,
                                "name": args.name,
                                "summary": summary_text,
                                "weight": weight
                            }))
                        })
                    }
                })
                .with_description("Retrieve an edge by its source, destination, and name"),
            )
            .tool(
                "query_outgoing_edges",
                TypedTool::new("query_outgoing_edges", {
                    let server = Arc::clone(&server_self);
                    move |args: QueryOutgoingEdgesParams, extra| {
                        let server = Arc::clone(&server);
                        Box::pin(async move {
                            Self::authenticate(&extra)?;

                            let id = Id::from_str(&args.id).map_err(|e| {
                                pmcp::Error::validation(format!("Invalid node ID: {}", e))
                            })?;

                            let query = OutgoingEdges::new(id, args.reference_ts_millis.map(TimestampMilli));

                            let edges = query
                                .run(&server.reader, server.query_timeout)
                                .await
                                .map_err(|e| {
                                    pmcp::Error::internal(format!("Failed to query outgoing edges: {}", e))
                                })?;

                            log::info!("Queried outgoing edges from node: {} (found {})", args.id, edges.len());

                            let edges_list: Vec<JsonValue> = edges
                                .into_iter()
                                .map(|(weight, _src, dst, name)| {
                                    json!({
                                        "target_id": dst.as_str(),
                                        "name": name,
                                        "weight": weight
                                    })
                                })
                                .collect();

                            Ok(json!({
                                "node_id": args.id,
                                "edges": edges_list,
                                "count": edges_list.len()
                            }))
                        })
                    }
                })
                .with_description("Get all outgoing edges from a node"),
            )
            .tool(
                "query_incoming_edges",
                TypedTool::new("query_incoming_edges", {
                    let server = Arc::clone(&server_self);
                    move |args: QueryIncomingEdgesParams, extra| {
                        let server = Arc::clone(&server);
                        Box::pin(async move {
                            Self::authenticate(&extra)?;

                            let id = Id::from_str(&args.id).map_err(|e| {
                                pmcp::Error::validation(format!("Invalid node ID: {}", e))
                            })?;

                            let query = IncomingEdges::new(id, args.reference_ts_millis.map(TimestampMilli));

                            let edges = query
                                .run(&server.reader, server.query_timeout)
                                .await
                                .map_err(|e| {
                                    pmcp::Error::internal(format!("Failed to query incoming edges: {}", e))
                                })?;

                            log::info!("Queried incoming edges to node: {} (found {})", args.id, edges.len());

                            let edges_list: Vec<JsonValue> = edges
                                .into_iter()
                                .map(|(weight, _dst, src, name)| {
                                    json!({
                                        "source_id": src.as_str(),
                                        "name": name,
                                        "weight": weight
                                    })
                                })
                                .collect();

                            Ok(json!({
                                "node_id": args.id,
                                "edges": edges_list,
                                "count": edges_list.len()
                            }))
                        })
                    }
                })
                .with_description("Get all incoming edges to a node"),
            )
            .tool(
                "query_nodes_by_name",
                TypedTool::new("query_nodes_by_name", {
                    let server = Arc::clone(&server_self);
                    move |args: QueryNodesByNameParams, extra| {
                        let server = Arc::clone(&server);
                        Box::pin(async move {
                            Self::authenticate(&extra)?;

                            let query = NodesByName::new(
                                args.name.clone(),
                                None,
                                args.limit,
                                args.reference_ts_millis.map(TimestampMilli),
                            );

                            let nodes = query
                                .run(&server.reader, server.query_timeout)
                                .await
                                .map_err(|e| {
                                    pmcp::Error::internal(format!("Failed to query nodes by name: {}", e))
                                })?;

                            log::info!(
                                "Queried nodes by name: '{}' (found {})",
                                args.name,
                                nodes.len()
                            );

                            let nodes_list: Vec<JsonValue> = nodes
                                .into_iter()
                                .map(|(name, id)| {
                                    json!({
                                        "name": name,
                                        "id": id.as_str()
                                    })
                                })
                                .collect();

                            Ok(json!({
                                "search_name": args.name,
                                "nodes": nodes_list,
                                "count": nodes_list.len()
                            }))
                        })
                    }
                })
                .with_description("Search for nodes by name or name prefix"),
            )
            .tool(
                "query_edges_by_name",
                TypedTool::new("query_edges_by_name", {
                    let server = Arc::clone(&server_self);
                    move |args: QueryEdgesByNameParams, extra| {
                        let server = Arc::clone(&server);
                        Box::pin(async move {
                            Self::authenticate(&extra)?;

                            let query = EdgesByName::new(
                                args.name.clone(),
                                None,
                                args.limit,
                                args.reference_ts_millis.map(TimestampMilli),
                            );

                            let edges = query
                                .run(&server.reader, server.query_timeout)
                                .await
                                .map_err(|e| {
                                    pmcp::Error::internal(format!("Failed to query edges by name: {}", e))
                                })?;

                            log::info!(
                                "Queried edges by name: '{}' (found {})",
                                args.name,
                                edges.len()
                            );

                            let edges_list: Vec<JsonValue> = edges
                                .into_iter()
                                .map(|(name, id)| {
                                    json!({
                                        "name": name,
                                        "id": id.as_str()
                                    })
                                })
                                .collect();

                            Ok(json!({
                                "search_name": args.name,
                                "edges": edges_list,
                                "count": edges_list.len()
                            }))
                        })
                    }
                })
                .with_description("Search for edges by name or name prefix"),
            )
            .tool(
                "query_node_fragments",
                TypedTool::new("query_node_fragments", {
                    let server = Arc::clone(&server_self);
                    move |args: QueryNodeFragmentsParams, extra| {
                        let server = Arc::clone(&server);
                        Box::pin(async move {
                            Self::authenticate(&extra)?;

                            let id = Id::from_str(&args.id).map_err(|e| {
                                pmcp::Error::validation(format!("Invalid node ID: {}", e))
                            })?;

                            let start_bound = args
                                .start_ts_millis
                                .map_or(Bound::Unbounded, |ts| Bound::Included(TimestampMilli(ts)));
                            let end_bound = args
                                .end_ts_millis
                                .map_or(Bound::Unbounded, |ts| Bound::Included(TimestampMilli(ts)));

                            let query = NodeFragmentsByIdTimeRange::new(
                                id,
                                (start_bound, end_bound),
                                args.reference_ts_millis.map(TimestampMilli),
                            );

                            let fragments = query
                                .run(&server.reader, server.query_timeout)
                                .await
                                .map_err(|e| {
                                    pmcp::Error::internal(format!("Failed to query node fragments: {}", e))
                                })?;

                            log::info!(
                                "Queried node fragments for node: {} (found {})",
                                args.id,
                                fragments.len()
                            );

                            let fragments_list: Vec<JsonValue> = fragments
                                .into_iter()
                                .map(|(ts, content)| {
                                    let content_text = content
                                        .decode_string()
                                        .unwrap_or_else(|_| "Unable to decode content".to_string());
                                    json!({
                                        "timestamp_millis": ts.0,
                                        "content": content_text
                                    })
                                })
                                .collect();

                            Ok(json!({
                                "node_id": args.id,
                                "fragments": fragments_list,
                                "count": fragments_list.len()
                            }))
                        })
                    }
                })
                .with_description("Get content fragments for a node within a time range"),
            );

        builder.build()
    }

    /// Authenticate a request using pmcp's built-in auth context
    ///
    /// This checks if an auth token is configured and validates it against
    /// the auth_context provided by pmcp in the RequestHandlerExtra.
    fn authenticate(extra: &pmcp::RequestHandlerExtra) -> pmcp::Result<()> {
        // Check if auth is required
        if let Ok(expected_token) = std::env::var("MOTLIE_MCP_AUTH_TOKEN") {
            // Auth is required, verify the token from auth_context
            if let Some(auth_ctx) = &extra.auth_context {
                if let Some(provided_token) = &auth_ctx.token {
                    if provided_token == &expected_token {
                        log::debug!("Authentication successful for subject: {}", auth_ctx.subject);
                        return Ok(());
                    }
                }
                log::warn!("Authentication failed: invalid token");
                return Err(pmcp::Error::validation("Invalid authentication token"));
            }
            log::warn!("Authentication failed: missing auth context");
            return Err(pmcp::Error::validation("Authentication required"));
        }
        // No auth required
        Ok(())
    }
}
