//! MCP tool parameter types for Motlie graph database operations
//!
//! All parameter types derive Serialize, Deserialize, and JsonSchema for automatic
//! schema generation via the pmcp SDK. Authentication is handled at the protocol
//! level by pmcp and does not need to be included in tool parameters.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Temporal validity range for nodes and edges
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TemporalRangeParam {
    /// Start of validity period (milliseconds since Unix epoch)
    #[schemars(description = "Timestamp in milliseconds when validity begins")]
    pub valid_since: u64,

    /// End of validity period (milliseconds since Unix epoch)
    #[schemars(description = "Timestamp in milliseconds when validity ends")]
    pub valid_until: u64,
}

/// Parameters for adding a new node to the graph
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct AddNodeParams {
    /// Human-readable node name
    #[schemars(description = "Human-readable name for the node")]
    pub name: String,

    /// Optional timestamp (defaults to current time)
    #[schemars(description = "Timestamp in milliseconds since Unix epoch")]
    pub ts_millis: Option<u64>,

    /// Optional temporal validity range
    #[schemars(description = "Time range during which this node is valid")]
    pub temporal_range: Option<TemporalRangeParam>,
}

/// Parameters for adding an edge between two nodes
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct AddEdgeParams {
    /// Source node UUID
    #[schemars(description = "Source node identifier (base32-encoded ULID)")]
    pub source_node_id: String,

    /// Target node UUID
    #[schemars(description = "Target node identifier (base32-encoded ULID)")]
    pub target_node_id: String,

    /// Edge name/type
    #[schemars(description = "Name or type of the edge (e.g., 'knows', 'follows')")]
    pub name: String,

    /// Edge summary/description
    #[schemars(description = "Summary or description of the edge relationship")]
    pub summary: String,

    /// Optional edge weight for graph algorithms
    #[schemars(description = "Numeric weight for weighted graph algorithms")]
    pub weight: Option<f64>,

    /// Optional timestamp (defaults to current time)
    #[schemars(description = "Timestamp in milliseconds since Unix epoch")]
    pub ts_millis: Option<u64>,

    /// Optional temporal validity range
    #[schemars(description = "Time range during which this edge is valid")]
    pub temporal_range: Option<TemporalRangeParam>,
}

/// Parameters for adding a fragment to a node
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct AddNodeFragmentParams {
    /// Node UUID
    #[schemars(description = "Node identifier to attach fragment to")]
    pub id: String,

    /// Fragment content as data URL
    #[schemars(
        description = "Content in data URL format (e.g., 'data:text/plain;base64,...')"
    )]
    pub content: String,

    /// Optional timestamp (defaults to current time)
    #[schemars(description = "Timestamp in milliseconds since Unix epoch")]
    pub ts_millis: Option<u64>,

    /// Optional temporal validity range
    #[schemars(description = "Time range during which this fragment is valid")]
    pub temporal_range: Option<TemporalRangeParam>,
}

/// Parameters for adding a fragment to an edge
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct AddEdgeFragmentParams {
    /// Source node UUID
    #[schemars(description = "Source node identifier")]
    pub src_id: String,

    /// Destination node UUID
    #[schemars(description = "Destination node identifier")]
    pub dst_id: String,

    /// Edge name
    #[schemars(description = "Name of the edge")]
    pub edge_name: String,

    /// Fragment content as data URL
    #[schemars(
        description = "Content in data URL format (e.g., 'data:text/plain;base64,...')"
    )]
    pub content: String,

    /// Optional timestamp (defaults to current time)
    #[schemars(description = "Timestamp in milliseconds since Unix epoch")]
    pub ts_millis: Option<u64>,

    /// Optional temporal validity range
    #[schemars(description = "Time range during which this fragment is valid")]
    pub temporal_range: Option<TemporalRangeParam>,
}

/// Parameters for updating node temporal validity
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct UpdateNodeValidRangeParams {
    /// Node UUID
    #[schemars(description = "Node identifier to update")]
    pub id: String,

    /// New temporal validity range
    #[schemars(description = "New time range for this node's validity")]
    pub temporal_range: TemporalRangeParam,

    /// Reason for the update
    #[schemars(description = "Explanation for why the validity range is being updated")]
    pub reason: String,
}

/// Parameters for updating edge temporal validity
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct UpdateEdgeValidRangeParams {
    /// Source node UUID
    #[schemars(description = "Source node identifier")]
    pub src_id: String,

    /// Destination node UUID
    #[schemars(description = "Destination node identifier")]
    pub dst_id: String,

    /// Edge name
    #[schemars(description = "Name of the edge")]
    pub name: String,

    /// New temporal validity range
    #[schemars(description = "New time range for this edge's validity")]
    pub temporal_range: TemporalRangeParam,

    /// Reason for the update
    #[schemars(description = "Explanation for why the validity range is being updated")]
    pub reason: String,
}

/// Parameters for updating edge weight
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct UpdateEdgeWeightParams {
    /// Source node UUID
    #[schemars(description = "Source node identifier")]
    pub src_id: String,

    /// Destination node UUID
    #[schemars(description = "Destination node identifier")]
    pub dst_id: String,

    /// Edge name
    #[schemars(description = "Name of the edge")]
    pub name: String,

    /// New weight value
    #[schemars(description = "New numeric weight for graph algorithms")]
    pub weight: f64,
}

/// Parameters for querying a node by ID
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct QueryNodeByIdParams {
    /// Node UUID
    #[schemars(description = "Node identifier to query")]
    pub id: String,

    /// Reference timestamp for temporal validity checks
    #[schemars(description = "Timestamp to check validity against (defaults to current time)")]
    pub reference_ts_millis: Option<u64>,
}

/// Parameters for querying an edge by endpoints and name
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct QueryEdgeParams {
    /// Source node UUID
    #[schemars(description = "Source node identifier")]
    pub source_id: String,

    /// Destination node UUID
    #[schemars(description = "Destination node identifier")]
    pub dest_id: String,

    /// Edge name
    #[schemars(description = "Name of the edge")]
    pub name: String,

    /// Reference timestamp for temporal validity checks
    #[schemars(description = "Timestamp to check validity against (defaults to current time)")]
    pub reference_ts_millis: Option<u64>,
}

/// Parameters for querying outgoing edges from a node
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct QueryOutgoingEdgesParams {
    /// Node UUID
    #[schemars(description = "Node identifier to query outgoing edges from")]
    pub id: String,

    /// Reference timestamp for temporal validity checks
    #[schemars(description = "Timestamp to check validity against (defaults to current time)")]
    pub reference_ts_millis: Option<u64>,
}

/// Parameters for querying incoming edges to a node
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct QueryIncomingEdgesParams {
    /// Node UUID
    #[schemars(description = "Node identifier to query incoming edges to")]
    pub id: String,

    /// Reference timestamp for temporal validity checks
    #[schemars(description = "Timestamp to check validity against (defaults to current time)")]
    pub reference_ts_millis: Option<u64>,
}

/// Parameters for querying nodes by name prefix
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct QueryNodesByNameParams {
    /// Name or prefix to search for
    #[schemars(description = "Node name or prefix to search for")]
    pub name: String,

    /// Maximum number of results
    #[schemars(description = "Maximum number of results to return")]
    pub limit: Option<usize>,

    /// Reference timestamp for temporal validity checks
    #[schemars(description = "Timestamp to check validity against (defaults to current time)")]
    pub reference_ts_millis: Option<u64>,
}

/// Parameters for querying edges by name prefix
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct QueryEdgesByNameParams {
    /// Name or prefix to search for
    #[schemars(description = "Edge name or prefix to search for")]
    pub name: String,

    /// Maximum number of results
    #[schemars(description = "Maximum number of results to return")]
    pub limit: Option<usize>,

    /// Reference timestamp for temporal validity checks
    #[schemars(description = "Timestamp to check validity against (defaults to current time)")]
    pub reference_ts_millis: Option<u64>,
}

/// Parameters for querying node fragments by time range
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct QueryNodeFragmentsParams {
    /// Node UUID
    #[schemars(description = "Node identifier to query fragments from")]
    pub id: String,

    /// Start of time range (optional, unbounded if not specified)
    #[schemars(description = "Start timestamp in milliseconds (inclusive)")]
    pub start_ts_millis: Option<u64>,

    /// End of time range (optional, unbounded if not specified)
    #[schemars(description = "End timestamp in milliseconds (inclusive)")]
    pub end_ts_millis: Option<u64>,

    /// Reference timestamp for temporal validity checks
    #[schemars(description = "Timestamp to check validity against (defaults to current time)")]
    pub reference_ts_millis: Option<u64>,
}

/// Parameters for querying edge fragments by time range
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct QueryEdgeFragmentsParams {
    /// Source node UUID
    #[schemars(description = "Source node identifier")]
    pub src_id: String,

    /// Destination node UUID
    #[schemars(description = "Destination node identifier")]
    pub dst_id: String,

    /// Edge name
    #[schemars(description = "Name of the edge")]
    pub edge_name: String,

    /// Start of time range (optional, unbounded if not specified)
    #[schemars(description = "Start timestamp in milliseconds (inclusive)")]
    pub start_ts_millis: Option<u64>,

    /// End of time range (optional, unbounded if not specified)
    #[schemars(description = "End timestamp in milliseconds (inclusive)")]
    pub end_ts_millis: Option<u64>,

    /// Reference timestamp for temporal validity checks
    #[schemars(description = "Timestamp to check validity against (defaults to current time)")]
    pub reference_ts_millis: Option<u64>,
}
