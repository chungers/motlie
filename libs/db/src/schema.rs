use crate::{AddEdgeArgs, AddFragmentArgs, AddNodeArgs, Id};
use serde::{Deserialize, Serialize};

/// Trait for column family record types that can create and serialize key-value pairs.
pub(crate) trait ColumnFamilyRecord {
    const CF_NAME: &'static str;

    /// The key type for this column family
    type Key: Serialize + for<'de> Deserialize<'de>;

    /// The value type for this column family
    type Value: Serialize + for<'de> Deserialize<'de>;

    /// The argument type for creating records
    type Args;

    /// Create a key-value pair from arguments
    fn record_from(args: &Self::Args) -> (Self::Key, Self::Value);

    /// Create and serialize to bytes using MessagePack
    fn create_bytes(args: &Self::Args) -> Result<(Vec<u8>, Vec<u8>), rmp_serde::encode::Error> {
        let (key, value) = Self::record_from(args);
        let key_bytes = rmp_serde::to_vec(&key)?;
        let value_bytes = rmp_serde::to_vec(&value)?;
        Ok((key_bytes, value_bytes))
    }

    /// Serialize the key to bytes using MessagePack
    fn key_to_bytes(key: &Self::Key) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        rmp_serde::to_vec(key)
    }

    /// Serialize the value to bytes using MessagePack
    fn value_to_bytes(value: &Self::Value) -> Result<Vec<u8>, rmp_serde::encode::Error> {
        rmp_serde::to_vec(value)
    }

    /// Deserialize the key from bytes using MessagePack
    fn key_from_bytes(bytes: &[u8]) -> Result<Self::Key, rmp_serde::decode::Error> {
        rmp_serde::from_slice(bytes)
    }

    /// Deserialize the value from bytes using MessagePack
    fn value_from_bytes(bytes: &[u8]) -> Result<Self::Value, rmp_serde::decode::Error> {
        rmp_serde::from_slice(bytes)
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct TimestampMillisecond(pub(crate) u64);

/// Nodes column family.
pub(crate) struct Nodes;

#[derive(Serialize, Deserialize)]
pub(crate) struct NodeCfKey(pub(crate) Id);

#[derive(Serialize, Deserialize)]
pub(crate) struct NodeCfValue(pub(crate) NodeSummary);

#[derive(Serialize, Deserialize)]
pub(crate) struct NodeSummary(pub(crate) String);

impl ColumnFamilyRecord for Nodes {
    const CF_NAME: &'static str = "nodes";
    type Key = NodeCfKey;
    type Value = NodeCfValue;
    type Args = AddNodeArgs;

    fn record_from(args: &AddNodeArgs) -> (NodeCfKey, NodeCfValue) {
        let key = NodeCfKey(args.id);
        let summary = format!(
            "[comment]:\\#<!-- id={} -->]\n# {}\n# Summary\n",
            args.id, args.name
        );
        let value = NodeCfValue(NodeSummary(summary));
        (key, value)
    }
}

pub(crate) struct Edges;

#[derive(Serialize, Deserialize)]
pub(crate) struct EdgeCfKey(pub(crate) Id);

#[derive(Serialize, Deserialize)]
pub(crate) struct EdgeCfValue(pub(crate) EdgeSummary);

#[derive(Serialize, Deserialize)]
pub(crate) struct EdgeSummary(pub(crate) String);

impl ColumnFamilyRecord for Edges {
    const CF_NAME: &'static str = "edges";
    type Key = EdgeCfKey;
    type Value = EdgeCfValue;
    type Args = AddEdgeArgs;

    fn record_from(args: &AddEdgeArgs) -> (EdgeCfKey, EdgeCfValue) {
        let key = EdgeCfKey(args.id);
        let summary = format!(
            "[comment]:\\#<!-- id={} -->]\n# {}\n# Summary\n",
            args.id, args.name
        );
        let value = EdgeCfValue(EdgeSummary(summary));
        (key, value)
    }
}

pub(crate) struct Fragments;

#[derive(Serialize, Deserialize)]
pub(crate) struct FragmentCfKey(pub(crate) Id, pub(crate) TimestampMillisecond);

#[derive(Serialize, Deserialize)]
pub(crate) struct FragmentCfValue(pub(crate) FragmentContent);

#[derive(Serialize, Deserialize)]
pub(crate) struct FragmentContent(pub(crate) String);

impl ColumnFamilyRecord for Fragments {
    const CF_NAME: &'static str = "fragments";
    type Key = FragmentCfKey;
    type Value = FragmentCfValue;
    type Args = AddFragmentArgs;

    fn record_from(args: &AddFragmentArgs) -> (FragmentCfKey, FragmentCfValue) {
        let key = FragmentCfKey(args.id, TimestampMillisecond(args.ts_millis));
        let content = FragmentContent(args.content.clone());
        let value = FragmentCfValue(content);
        (key, value)
    }
}

pub(crate) struct ForwardEdges;

#[derive(Serialize, Deserialize)]
pub(crate) struct ForwardEdgeCfKey(
    pub(crate) EdgeSourceId,
    pub(crate) EdgeDestinationId,
    pub(crate) EdgeName,
);

#[derive(Serialize, Deserialize)]
pub(crate) struct ForwardEdgeCfValue(pub(crate) EdgeSummary);

#[derive(Serialize, Deserialize)]
pub(crate) struct EdgeSourceId(pub(crate) Id);

#[derive(Serialize, Deserialize)]
pub(crate) struct EdgeDestinationId(pub(crate) Id);

#[derive(Serialize, Deserialize)]
pub(crate) struct EdgeName(pub(crate) String);

impl ColumnFamilyRecord for ForwardEdges {
    const CF_NAME: &'static str = "forward_edges";
    type Key = ForwardEdgeCfKey;
    type Value = ForwardEdgeCfValue;
    type Args = AddEdgeArgs;

    fn record_from(args: &AddEdgeArgs) -> (ForwardEdgeCfKey, ForwardEdgeCfValue) {
        let key = ForwardEdgeCfKey(
            EdgeSourceId(args.source_vertex_id),
            EdgeDestinationId(args.target_vertex_id),
            EdgeName(args.name.clone()),
        );
        let summary = format!(
            "[comment]:\\#<!-- id={} -->]\n# {}\n# Summary\n",
            args.id, args.name
        );
        let value = ForwardEdgeCfValue(EdgeSummary(summary));
        (key, value)
    }
}

pub(crate) struct ReverseEdges;

#[derive(Serialize, Deserialize)]
pub(crate) struct ReverseEdgeCfKey(
    pub(crate) EdgeDestinationId,
    pub(crate) EdgeSourceId,
    pub(crate) EdgeName,
);

#[derive(Serialize, Deserialize)]
pub(crate) struct ReverseEdgeCfValue(pub(crate) EdgeSummary);

impl ColumnFamilyRecord for ReverseEdges {
    const CF_NAME: &'static str = "reverse_edges";
    type Key = ReverseEdgeCfKey;
    type Value = ReverseEdgeCfValue;
    type Args = AddEdgeArgs;

    fn record_from(args: &AddEdgeArgs) -> (ReverseEdgeCfKey, ReverseEdgeCfValue) {
        let key = ReverseEdgeCfKey(
            EdgeDestinationId(args.source_vertex_id),
            EdgeSourceId(args.target_vertex_id),
            EdgeName(args.name.clone()),
        );
        let summary = format!(
            "[comment]:\\#<!-- id={} -->]\n# {}\n# Summary\n",
            args.id, args.name
        );
        let value = ReverseEdgeCfValue(EdgeSummary(summary));
        (key, value)
    }
}

/// All column families used in the database.
/// This is the authoritative list that should be used when opening the database.
pub(crate) const ALL_COLUMN_FAMILIES: &[&str] = &[
    Nodes::CF_NAME,
    Edges::CF_NAME,
    Fragments::CF_NAME,
    ForwardEdges::CF_NAME,
    ReverseEdges::CF_NAME,
];
