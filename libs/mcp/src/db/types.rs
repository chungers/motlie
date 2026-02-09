//! Parameter types and ToolCall implementations for Motlie database tools.
//!
//! Each parameter type implements the `ToolCall` trait, binding it to its
//! execution logic. This ensures compile-time verification that every
//! parameter type has a corresponding tool implementation.

use super::DbResource;
use crate::ToolCall;
use async_trait::async_trait;
use motlie_db::mutation::{
    AddEdge, AddEdgeFragment, AddNode, AddNodeFragment, EdgeSummary, NodeSummary,
    Runnable as MutationRunnable, UpdateEdge, UpdateNode,
};
use motlie_db::query::{
    EdgeDetails, EdgeFragments, Edges, IncomingEdges, NodeById, NodeFragments, Nodes,
    OutgoingEdges, Runnable as QueryRunnable,
};
use motlie_db::{ActivePeriod, DataUrl, Id, TimestampMilli};
use rmcp::{model::*, ErrorData as McpError};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::ops::Bound;

// ============================================================
// Shared Types
// ============================================================

/// Temporal validity range for nodes and edges
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TemporalRangeParam {
    /// Start of validity period (milliseconds since Unix epoch)
    pub valid_since: u64,
    /// End of validity period (milliseconds since Unix epoch)
    pub valid_until: u64,
}

impl TemporalRangeParam {
    fn to_schema(self) -> ActivePeriod {
        ActivePeriod(
            Some(TimestampMilli(self.valid_since)),
            Some(TimestampMilli(self.valid_until)),
        )
    }
}

// ============================================================
// Mutation Tools
// ============================================================

/// Parameters for adding a new node to the graph
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AddNodeParams {
    /// Human-readable node name
    pub name: String,
    /// Node summary/description
    pub summary: String,
    /// Optional timestamp (defaults to current time)
    pub ts_millis: Option<u64>,
    /// Optional temporal validity range
    pub temporal_range: Option<TemporalRangeParam>,
}

#[async_trait]
impl ToolCall for AddNodeParams {
    type Resource = DbResource;

    async fn call(self, res: &DbResource) -> Result<CallToolResult, McpError> {
        let writer = res.writer().await?;
        let id = Id::new();

        let mutation = AddNode {
            id,
            name: self.name.clone(),
            ts_millis: TimestampMilli(self.ts_millis.unwrap_or_else(|| TimestampMilli::now().0)),
            valid_range: self.temporal_range.map(|r| r.to_schema()),
            summary: NodeSummary::from_text(&self.summary),
        };

        mutation.run(writer).await.map_err(|e| {
            McpError::internal_error(format!("Failed to add node: {}", e), None)
        })?;

        tracing::info!("Added node: {} ({})", self.name, id);

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "success": true,
                "message": format!("Successfully added node '{}' with ID {}", self.name, id.as_str()),
                "node_id": id.as_str(),
                "node_name": self.name
            })
            .to_string(),
        )]))
    }
}

/// Parameters for adding an edge between two nodes
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AddEdgeParams {
    /// Source node UUID (base32-encoded ULID)
    pub source_node_id: String,
    /// Target node UUID (base32-encoded ULID)
    pub target_node_id: String,
    /// Edge name/type (e.g., 'knows', 'follows')
    pub name: String,
    /// Edge summary/description
    pub summary: String,
    /// Optional edge weight for graph algorithms
    pub weight: Option<f64>,
    /// Optional timestamp (defaults to current time)
    pub ts_millis: Option<u64>,
    /// Optional temporal validity range
    pub temporal_range: Option<TemporalRangeParam>,
}

#[async_trait]
impl ToolCall for AddEdgeParams {
    type Resource = DbResource;

    async fn call(self, res: &DbResource) -> Result<CallToolResult, McpError> {
        let writer = res.writer().await?;

        let source_id = Id::from_str(&self.source_node_id)
            .map_err(|e| McpError::invalid_params(format!("Invalid source node ID: {}", e), None))?;
        let target_id = Id::from_str(&self.target_node_id)
            .map_err(|e| McpError::invalid_params(format!("Invalid target node ID: {}", e), None))?;

        let mutation = AddEdge {
            source_node_id: source_id,
            target_node_id: target_id,
            ts_millis: TimestampMilli(self.ts_millis.unwrap_or_else(|| TimestampMilli::now().0)),
            name: self.name.clone(),
            valid_range: self.temporal_range.map(|r| r.to_schema()),
            summary: EdgeSummary::from_text(&self.summary),
            weight: self.weight,
        };

        mutation.run(writer).await.map_err(|e| {
            McpError::internal_error(format!("Failed to add edge: {}", e), None)
        })?;

        tracing::info!(
            "Added edge: {} -> {} ({})",
            self.source_node_id,
            self.target_node_id,
            self.name
        );

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "success": true,
                "message": format!("Successfully added edge '{}' from {} to {}", self.name, self.source_node_id, self.target_node_id),
                "edge_name": self.name,
                "source_id": self.source_node_id,
                "target_id": self.target_node_id
            })
            .to_string(),
        )]))
    }
}

/// Parameters for adding a fragment to a node
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AddNodeFragmentParams {
    /// Node UUID
    pub id: String,
    /// Fragment content (text)
    pub content: String,
    /// Optional timestamp (defaults to current time)
    pub ts_millis: Option<u64>,
    /// Optional temporal validity range
    pub temporal_range: Option<TemporalRangeParam>,
}

#[async_trait]
impl ToolCall for AddNodeFragmentParams {
    type Resource = DbResource;

    async fn call(self, res: &DbResource) -> Result<CallToolResult, McpError> {
        let writer = res.writer().await?;

        let id = Id::from_str(&self.id)
            .map_err(|e| McpError::invalid_params(format!("Invalid node ID: {}", e), None))?;

        let mutation = AddNodeFragment {
            id,
            ts_millis: TimestampMilli(self.ts_millis.unwrap_or_else(|| TimestampMilli::now().0)),
            content: DataUrl::from_text(&self.content),
            valid_range: self.temporal_range.map(|r| r.to_schema()),
        };

        mutation.run(writer).await.map_err(|e| {
            McpError::internal_error(format!("Failed to add node fragment: {}", e), None)
        })?;

        tracing::info!("Added fragment to node: {}", self.id);

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "success": true,
                "message": format!("Successfully added fragment to node {}", self.id),
                "node_id": self.id
            })
            .to_string(),
        )]))
    }
}

/// Parameters for adding a fragment to an edge
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AddEdgeFragmentParams {
    /// Source node UUID
    pub src_id: String,
    /// Destination node UUID
    pub dst_id: String,
    /// Edge name
    pub edge_name: String,
    /// Fragment content (text)
    pub content: String,
    /// Optional timestamp (defaults to current time)
    pub ts_millis: Option<u64>,
    /// Optional temporal validity range
    pub temporal_range: Option<TemporalRangeParam>,
}

#[async_trait]
impl ToolCall for AddEdgeFragmentParams {
    type Resource = DbResource;

    async fn call(self, res: &DbResource) -> Result<CallToolResult, McpError> {
        let writer = res.writer().await?;

        let src_id = Id::from_str(&self.src_id)
            .map_err(|e| McpError::invalid_params(format!("Invalid source node ID: {}", e), None))?;
        let dst_id = Id::from_str(&self.dst_id)
            .map_err(|e| McpError::invalid_params(format!("Invalid destination node ID: {}", e), None))?;

        let mutation = AddEdgeFragment {
            src_id,
            dst_id,
            edge_name: self.edge_name.clone(),
            ts_millis: TimestampMilli(self.ts_millis.unwrap_or_else(|| TimestampMilli::now().0)),
            content: DataUrl::from_text(&self.content),
            valid_range: self.temporal_range.map(|r| r.to_schema()),
        };

        mutation.run(writer).await.map_err(|e| {
            McpError::internal_error(format!("Failed to add edge fragment: {}", e), None)
        })?;

        tracing::info!(
            "Added fragment to edge: {} -> {} ({})",
            self.src_id,
            self.dst_id,
            self.edge_name
        );

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "success": true,
                "message": format!("Successfully added fragment to edge {} -> {} ({})", self.src_id, self.dst_id, self.edge_name),
                "src_id": self.src_id,
                "dst_id": self.dst_id,
                "edge_name": self.edge_name
            })
            .to_string(),
        )]))
    }
}

/// Parameters for updating a node (consolidated API)
///
/// Updates any combination of active period and summary.
/// At least one optional field must be provided.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct UpdateNodeParams {
    /// Node UUID
    pub id: String,
    /// Expected version for optimistic locking
    pub expected_version: u32,
    /// Optional: new temporal validity range (null to reset, omit for no change)
    pub new_active_period: Option<Option<TemporalRangeParam>>,
    /// Optional: new summary content
    pub new_summary: Option<String>,
}

#[async_trait]
impl ToolCall for UpdateNodeParams {
    type Resource = DbResource;

    async fn call(self, res: &DbResource) -> Result<CallToolResult, McpError> {
        let writer = res.writer().await?;

        let id = Id::from_str(&self.id)
            .map_err(|e| McpError::invalid_params(format!("Invalid node ID: {}", e), None))?;

        let mutation = UpdateNode {
            id,
            expected_version: self.expected_version,
            new_active_period: self.new_active_period.clone().map(|opt| opt.map(|r| r.to_schema())),
            new_summary: self.new_summary.as_ref().map(|s| NodeSummary::from_text(s)),
        };

        mutation.run(writer).await.map_err(|e| {
            McpError::internal_error(format!("Failed to update node: {}", e), None)
        })?;

        tracing::info!(
            "Updated node: {}, version {}",
            self.id,
            self.expected_version
        );

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "success": true,
                "message": format!("Successfully updated node {}", self.id),
                "node_id": self.id,
                "new_active_period": self.new_active_period.as_ref().map(|opt| opt.as_ref().map(|r| format!("{} - {}", r.valid_since, r.valid_until))),
                "new_summary": self.new_summary.as_ref().map(|s| if s.len() > 50 { format!("{}...", &s[..50]) } else { s.clone() })
            })
            .to_string(),
        )]))
    }
}

/// Parameters for updating an edge (consolidated API)
///
/// Updates any combination of weight, active period, and summary.
/// At least one optional field must be provided.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct UpdateEdgeParams {
    /// Source node UUID
    pub src_id: String,
    /// Destination node UUID
    pub dst_id: String,
    /// Edge name
    pub name: String,
    /// Expected version for optimistic locking
    pub expected_version: u32,
    /// Optional: new weight value
    pub new_weight: Option<f64>,
    /// Optional: new temporal validity range
    pub new_active_period: Option<TemporalRangeParam>,
    /// Optional: new summary content
    pub new_summary: Option<String>,
}

#[async_trait]
impl ToolCall for UpdateEdgeParams {
    type Resource = DbResource;

    async fn call(self, res: &DbResource) -> Result<CallToolResult, McpError> {
        let writer = res.writer().await?;

        let src_id = Id::from_str(&self.src_id)
            .map_err(|e| McpError::invalid_params(format!("Invalid source node ID: {}", e), None))?;
        let dst_id = Id::from_str(&self.dst_id)
            .map_err(|e| McpError::invalid_params(format!("Invalid destination node ID: {}", e), None))?;

        let mutation = UpdateEdge {
            src_id,
            dst_id,
            name: self.name.clone(),
            expected_version: self.expected_version,
            new_weight: self.new_weight.map(Some),
            new_active_period: self.new_active_period.clone().map(|r| Some(r.to_schema())),
            new_summary: self.new_summary.as_ref().map(|s| EdgeSummary::from_text(s)),
        };

        mutation.run(writer).await.map_err(|e| {
            McpError::internal_error(format!("Failed to update edge: {}", e), None)
        })?;

        tracing::info!(
            "Updated edge: {} -> {} ({}), version {}",
            self.src_id,
            self.dst_id,
            self.name,
            self.expected_version
        );

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "success": true,
                "message": format!("Successfully updated edge {} -> {} ({})", self.src_id, self.dst_id, self.name),
                "src_id": self.src_id,
                "dst_id": self.dst_id,
                "edge_name": self.name,
                "new_weight": self.new_weight,
                "new_active_period": self.new_active_period.map(|r| format!("{} - {}", r.valid_since, r.valid_until)),
                "new_summary": self.new_summary.as_ref().map(|s| if s.len() > 50 { format!("{}...", &s[..50]) } else { s.clone() })
            })
            .to_string(),
        )]))
    }
}

// ============================================================
// Query Tools
// ============================================================

/// Parameters for querying a node by ID
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct QueryNodeByIdParams {
    /// Node UUID
    pub id: String,
    /// Reference timestamp for temporal validity checks
    pub reference_ts_millis: Option<u64>,
}

#[async_trait]
impl ToolCall for QueryNodeByIdParams {
    type Resource = DbResource;

    async fn call(self, res: &DbResource) -> Result<CallToolResult, McpError> {
        let reader = res.reader().await?;

        let id = Id::from_str(&self.id)
            .map_err(|e| McpError::invalid_params(format!("Invalid node ID: {}", e), None))?;

        let query = NodeById::new(id, self.reference_ts_millis.map(TimestampMilli));

        let (name, summary, _version) = query
            .run(reader, res.query_timeout())
            .await
            .map_err(|e| McpError::internal_error(format!("Failed to query node: {}", e), None))?;

        let summary_text = summary
            .decode_string()
            .unwrap_or_else(|_| "Unable to decode summary".to_string());

        tracing::info!("Queried node: {} ({})", self.id, name);

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "node_id": self.id,
                "name": name,
                "summary": summary_text
            })
            .to_string(),
        )]))
    }
}

/// Parameters for querying an edge by endpoints and name
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct QueryEdgeParams {
    /// Source node UUID
    pub source_id: String,
    /// Destination node UUID
    pub dest_id: String,
    /// Edge name
    pub name: String,
    /// Reference timestamp for temporal validity checks
    pub reference_ts_millis: Option<u64>,
}

#[async_trait]
impl ToolCall for QueryEdgeParams {
    type Resource = DbResource;

    async fn call(self, res: &DbResource) -> Result<CallToolResult, McpError> {
        let reader = res.reader().await?;

        let source_id = Id::from_str(&self.source_id)
            .map_err(|e| McpError::invalid_params(format!("Invalid source node ID: {}", e), None))?;
        let dest_id = Id::from_str(&self.dest_id)
            .map_err(|e| McpError::invalid_params(format!("Invalid destination node ID: {}", e), None))?;

        let query = EdgeDetails::new(
            source_id,
            dest_id,
            self.name.clone(),
            self.reference_ts_millis.map(TimestampMilli),
        );

        let (weight, _src, _dst, _name, summary, _version) = query
            .run(reader, res.query_timeout())
            .await
            .map_err(|e| McpError::internal_error(format!("Failed to query edge: {}", e), None))?;

        let summary_text = summary
            .decode_string()
            .unwrap_or_else(|_| "Unable to decode summary".to_string());

        tracing::info!(
            "Queried edge: {} -> {} ({})",
            self.source_id,
            self.dest_id,
            self.name
        );

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "source_id": self.source_id,
                "dest_id": self.dest_id,
                "name": self.name,
                "summary": summary_text,
                "weight": weight
            })
            .to_string(),
        )]))
    }
}

/// Parameters for querying outgoing edges from a node
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct QueryOutgoingEdgesParams {
    /// Node UUID
    pub id: String,
    /// Reference timestamp for temporal validity checks
    pub reference_ts_millis: Option<u64>,
}

#[async_trait]
impl ToolCall for QueryOutgoingEdgesParams {
    type Resource = DbResource;

    async fn call(self, res: &DbResource) -> Result<CallToolResult, McpError> {
        let reader = res.reader().await?;

        let id = Id::from_str(&self.id)
            .map_err(|e| McpError::invalid_params(format!("Invalid node ID: {}", e), None))?;

        let query = OutgoingEdges::new(id, self.reference_ts_millis.map(TimestampMilli));

        let edges =
            query.run(reader, res.query_timeout()).await.map_err(|e| {
                McpError::internal_error(format!("Failed to query outgoing edges: {}", e), None)
            })?;

        tracing::info!(
            "Queried outgoing edges from node: {} (found {})",
            self.id,
            edges.len()
        );

        let edges_list: Vec<serde_json::Value> = edges
            .into_iter()
            .map(|(weight, _src, dst, name, _version)| {
                json!({
                    "target_id": dst.as_str(),
                    "name": name,
                    "weight": weight
                })
            })
            .collect();

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "node_id": self.id,
                "edges": edges_list,
                "count": edges_list.len()
            })
            .to_string(),
        )]))
    }
}

/// Parameters for querying incoming edges to a node
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct QueryIncomingEdgesParams {
    /// Node UUID
    pub id: String,
    /// Reference timestamp for temporal validity checks
    pub reference_ts_millis: Option<u64>,
}

#[async_trait]
impl ToolCall for QueryIncomingEdgesParams {
    type Resource = DbResource;

    async fn call(self, res: &DbResource) -> Result<CallToolResult, McpError> {
        let reader = res.reader().await?;

        let id = Id::from_str(&self.id)
            .map_err(|e| McpError::invalid_params(format!("Invalid node ID: {}", e), None))?;

        let query = IncomingEdges::new(id, self.reference_ts_millis.map(TimestampMilli));

        let edges =
            query.run(reader, res.query_timeout()).await.map_err(|e| {
                McpError::internal_error(format!("Failed to query incoming edges: {}", e), None)
            })?;

        tracing::info!(
            "Queried incoming edges to node: {} (found {})",
            self.id,
            edges.len()
        );

        let edges_list: Vec<serde_json::Value> = edges
            .into_iter()
            .map(|(weight, _dst, src, name, _version)| {
                json!({
                    "source_id": src.as_str(),
                    "name": name,
                    "weight": weight
                })
            })
            .collect();

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "node_id": self.id,
                "edges": edges_list,
                "count": edges_list.len()
            })
            .to_string(),
        )]))
    }
}

/// Parameters for querying nodes by name prefix
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct QueryNodesByNameParams {
    /// Name or prefix to search for
    pub name: String,
    /// Maximum number of results
    pub limit: Option<usize>,
    /// Reference timestamp for temporal validity checks
    pub reference_ts_millis: Option<u64>,
}

#[async_trait]
impl ToolCall for QueryNodesByNameParams {
    type Resource = DbResource;

    async fn call(self, res: &DbResource) -> Result<CallToolResult, McpError> {
        let reader = res.reader().await?;

        let query = Nodes::new(self.name.clone(), self.limit.unwrap_or(100));

        let nodes: Vec<(Id, String, NodeSummary)> =
            query.run(reader, res.query_timeout()).await.map_err(|e| {
                McpError::internal_error(format!("Failed to query nodes by name: {}", e), None)
            })?;

        tracing::info!(
            "Queried nodes by name: '{}' (found {})",
            self.name,
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
            json!({
                "search_name": self.name,
                "nodes": nodes_list,
                "count": nodes_list.len()
            })
            .to_string(),
        )]))
    }
}

/// Parameters for querying edges by name prefix
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct QueryEdgesByNameParams {
    /// Name or prefix to search for
    pub name: String,
    /// Maximum number of results
    pub limit: Option<usize>,
    /// Reference timestamp for temporal validity checks
    pub reference_ts_millis: Option<u64>,
}

#[async_trait]
impl ToolCall for QueryEdgesByNameParams {
    type Resource = DbResource;

    async fn call(self, res: &DbResource) -> Result<CallToolResult, McpError> {
        let reader = res.reader().await?;

        let query = Edges::new(self.name.clone(), self.limit.unwrap_or(100));

        let edges: Vec<(Id, Id, String, EdgeSummary)> =
            query.run(reader, res.query_timeout()).await.map_err(|e| {
                McpError::internal_error(format!("Failed to query edges by name: {}", e), None)
            })?;

        tracing::info!(
            "Queried edges by name: '{}' (found {})",
            self.name,
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
            json!({
                "search_name": self.name,
                "edges": edges_list,
                "count": edges_list.len()
            })
            .to_string(),
        )]))
    }
}

/// Parameters for querying node fragments by time range
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct QueryNodeFragmentsParams {
    /// Node UUID
    pub id: String,
    /// Start of time range (optional, unbounded if not specified)
    pub start_ts_millis: Option<u64>,
    /// End of time range (optional, unbounded if not specified)
    pub end_ts_millis: Option<u64>,
    /// Reference timestamp for temporal validity checks
    pub reference_ts_millis: Option<u64>,
}

#[async_trait]
impl ToolCall for QueryNodeFragmentsParams {
    type Resource = DbResource;

    async fn call(self, res: &DbResource) -> Result<CallToolResult, McpError> {
        let reader = res.reader().await?;

        let id = Id::from_str(&self.id)
            .map_err(|e| McpError::invalid_params(format!("Invalid node ID: {}", e), None))?;

        let start_bound = self
            .start_ts_millis
            .map_or(Bound::Unbounded, |ts| Bound::Included(TimestampMilli(ts)));
        let end_bound = self
            .end_ts_millis
            .map_or(Bound::Unbounded, |ts| Bound::Included(TimestampMilli(ts)));

        let query = NodeFragments::new(
            id,
            (start_bound, end_bound),
            self.reference_ts_millis.map(TimestampMilli),
        );

        let fragments: Vec<(TimestampMilli, DataUrl)> =
            query.run(reader, res.query_timeout()).await.map_err(|e| {
                McpError::internal_error(format!("Failed to query node fragments: {}", e), None)
            })?;

        tracing::info!(
            "Queried node fragments for node: {} (found {})",
            self.id,
            fragments.len()
        );

        let fragments_list: Vec<serde_json::Value> = fragments
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

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "node_id": self.id,
                "fragments": fragments_list,
                "count": fragments_list.len()
            })
            .to_string(),
        )]))
    }
}

/// Parameters for querying edge fragments by time range
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct QueryEdgeFragmentsParams {
    /// Source node UUID
    pub src_id: String,
    /// Destination node UUID
    pub dst_id: String,
    /// Edge name
    pub edge_name: String,
    /// Start of time range (optional, unbounded if not specified)
    pub start_ts_millis: Option<u64>,
    /// End of time range (optional, unbounded if not specified)
    pub end_ts_millis: Option<u64>,
    /// Reference timestamp for temporal validity checks
    pub reference_ts_millis: Option<u64>,
}

#[async_trait]
impl ToolCall for QueryEdgeFragmentsParams {
    type Resource = DbResource;

    async fn call(self, res: &DbResource) -> Result<CallToolResult, McpError> {
        let reader = res.reader().await?;

        let src_id = Id::from_str(&self.src_id)
            .map_err(|e| McpError::invalid_params(format!("Invalid source node ID: {}", e), None))?;
        let dst_id = Id::from_str(&self.dst_id)
            .map_err(|e| McpError::invalid_params(format!("Invalid destination node ID: {}", e), None))?;

        let start_bound = self
            .start_ts_millis
            .map_or(Bound::Unbounded, |ts| Bound::Included(TimestampMilli(ts)));
        let end_bound = self
            .end_ts_millis
            .map_or(Bound::Unbounded, |ts| Bound::Included(TimestampMilli(ts)));

        let query = EdgeFragments::new(
            src_id,
            dst_id,
            self.edge_name.clone(),
            (start_bound, end_bound),
            self.reference_ts_millis.map(TimestampMilli),
        );

        let fragments: Vec<(TimestampMilli, DataUrl)> =
            query.run(reader, res.query_timeout()).await.map_err(|e| {
                McpError::internal_error(format!("Failed to query edge fragments: {}", e), None)
            })?;

        tracing::info!(
            "Queried edge fragments for edge {} -> {} '{}' (found {})",
            self.src_id,
            self.dst_id,
            self.edge_name,
            fragments.len()
        );

        let fragments_list: Vec<serde_json::Value> = fragments
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

        Ok(CallToolResult::success(vec![Content::text(
            json!({
                "src_id": self.src_id,
                "dst_id": self.dst_id,
                "edge_name": self.edge_name,
                "fragments": fragments_list,
                "count": fragments_list.len()
            })
            .to_string(),
        )]))
    }
}
