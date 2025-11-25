//! MCP tool parameter types for Motlie graph database operations
//!
//! All parameter types derive Serialize, Deserialize, and JsonSchema for automatic
//! schema generation via the rmcp SDK.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Temporal validity range for nodes and edges
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TemporalRangeParam {
    /// Start of validity period (milliseconds since Unix epoch)
    pub valid_since: u64,

    /// End of validity period (milliseconds since Unix epoch)
    pub valid_until: u64,
}

/// Parameters for adding a new node to the graph
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct AddNodeParams {
    /// Human-readable node name
    pub name: String,

    /// Optional timestamp (defaults to current time)
    pub ts_millis: Option<u64>,

    /// Optional temporal validity range
    pub temporal_range: Option<TemporalRangeParam>,
}

/// Parameters for adding an edge between two nodes
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
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

/// Parameters for adding a fragment to a node
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
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

/// Parameters for adding a fragment to an edge
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
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

/// Parameters for updating node temporal validity
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct UpdateNodeValidRangeParams {
    /// Node UUID
    pub id: String,

    /// New temporal validity range
    pub temporal_range: TemporalRangeParam,

    /// Reason for the update
    pub reason: String,
}

/// Parameters for updating edge temporal validity
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct UpdateEdgeValidRangeParams {
    /// Source node UUID
    pub src_id: String,

    /// Destination node UUID
    pub dst_id: String,

    /// Edge name
    pub name: String,

    /// New temporal validity range
    pub temporal_range: TemporalRangeParam,

    /// Reason for the update
    pub reason: String,
}

/// Parameters for updating edge weight
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct UpdateEdgeWeightParams {
    /// Source node UUID
    pub src_id: String,

    /// Destination node UUID
    pub dst_id: String,

    /// Edge name
    pub name: String,

    /// New weight value
    pub weight: f64,
}

/// Parameters for querying a node by ID
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct QueryNodeByIdParams {
    /// Node UUID
    pub id: String,

    /// Reference timestamp for temporal validity checks
    pub reference_ts_millis: Option<u64>,
}

/// Parameters for querying an edge by endpoints and name
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
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

/// Parameters for querying outgoing edges from a node
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct QueryOutgoingEdgesParams {
    /// Node UUID
    pub id: String,

    /// Reference timestamp for temporal validity checks
    pub reference_ts_millis: Option<u64>,
}

/// Parameters for querying incoming edges to a node
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct QueryIncomingEdgesParams {
    /// Node UUID
    pub id: String,

    /// Reference timestamp for temporal validity checks
    pub reference_ts_millis: Option<u64>,
}

/// Parameters for querying nodes by name prefix
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct QueryNodesByNameParams {
    /// Name or prefix to search for
    pub name: String,

    /// Maximum number of results
    pub limit: Option<usize>,

    /// Reference timestamp for temporal validity checks
    pub reference_ts_millis: Option<u64>,
}

/// Parameters for querying edges by name prefix
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct QueryEdgesByNameParams {
    /// Name or prefix to search for
    pub name: String,

    /// Maximum number of results
    pub limit: Option<usize>,

    /// Reference timestamp for temporal validity checks
    pub reference_ts_millis: Option<u64>,
}

/// Parameters for querying node fragments by time range
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
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

/// Parameters for querying edge fragments by time range
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
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
