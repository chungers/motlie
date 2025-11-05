use crate::graph::{PutCf, StorageOperation};
use crate::TimestampMilli;
use crate::{AddEdge, AddFragment, AddNode, Id};
use serde::{Deserialize, Serialize};

pub(crate) struct Plan {}
impl Plan {
    pub(crate) fn create_node(
        op: &AddNode,
    ) -> Result<Vec<StorageOperation>, rmp_serde::encode::Error> {
        Ok(vec![StorageOperation::PutCf(PutCf(
            Nodes::CF_NAME,
            Nodes::create_bytes(op)?,
        ))])
    }
    pub(crate) fn create_edge(
        op: &AddEdge,
    ) -> Result<Vec<StorageOperation>, rmp_serde::encode::Error> {
        Ok(vec![
            StorageOperation::PutCf(PutCf(Edges::CF_NAME, Edges::create_bytes(op)?)),
            StorageOperation::PutCf(PutCf(
                ForwardEdges::CF_NAME,
                ForwardEdges::create_bytes(op)?,
            )),
            StorageOperation::PutCf(PutCf(
                ReverseEdges::CF_NAME,
                ReverseEdges::create_bytes(op)?,
            )),
        ])
    }
    pub(crate) fn create_fragment(
        op: &AddFragment,
    ) -> Result<Vec<StorageOperation>, rmp_serde::encode::Error> {
        Ok(vec![StorageOperation::PutCf(PutCf(
            Fragments::CF_NAME,
            Fragments::create_bytes(op)?,
        ))])
    }
}

/// Trait for column family record types that can create and serialize key-value pairs.
pub(crate) trait ColumnFamilyRecord {
    const CF_NAME: &'static str;

    /// The key type for this column family
    type Key: Serialize + for<'de> Deserialize<'de>;

    /// The value type for this column family
    type Value: Serialize + for<'de> Deserialize<'de>;

    /// The argument type for creating records
    type CreateOp;

    /// Create a key-value pair from arguments
    fn record_from(args: &Self::CreateOp) -> (Self::Key, Self::Value);

    /// Create and serialize to bytes using MessagePack
    fn create_bytes(args: &Self::CreateOp) -> Result<(Vec<u8>, Vec<u8>), rmp_serde::encode::Error> {
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
    type CreateOp = AddNode;

    fn record_from(args: &AddNode) -> (NodeCfKey, NodeCfValue) {
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
    type CreateOp = AddEdge;

    fn record_from(args: &AddEdge) -> (EdgeCfKey, EdgeCfValue) {
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
pub(crate) struct FragmentCfKey(pub(crate) Id, pub(crate) TimestampMilli);

#[derive(Serialize, Deserialize)]
pub(crate) struct FragmentCfValue(pub(crate) FragmentContent);

#[derive(Serialize, Deserialize)]
pub(crate) struct FragmentContent(pub(crate) String);

impl ColumnFamilyRecord for Fragments {
    const CF_NAME: &'static str = "fragments";
    type Key = FragmentCfKey;
    type Value = FragmentCfValue;
    type CreateOp = AddFragment;

    fn record_from(args: &AddFragment) -> (FragmentCfKey, FragmentCfValue) {
        let key = FragmentCfKey(args.id, TimestampMilli(args.ts_millis));
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
    type CreateOp = AddEdge;

    fn record_from(args: &AddEdge) -> (ForwardEdgeCfKey, ForwardEdgeCfValue) {
        let key = ForwardEdgeCfKey(
            EdgeSourceId(args.source_node_id),
            EdgeDestinationId(args.target_node_id),
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
    type CreateOp = AddEdge;

    fn record_from(args: &AddEdge) -> (ReverseEdgeCfKey, ReverseEdgeCfValue) {
        let key = ReverseEdgeCfKey(
            EdgeDestinationId(args.source_node_id),
            EdgeSourceId(args.target_node_id),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AddEdge, Id};

    #[test]
    fn test_forward_edges_keys_lexicographically_sortable() {
        // Create multiple edge arguments with different source/destination combinations
        // Using deterministic timestamps for stable test behavior
        let base_ts = 1700000000000u64; // Fixed base timestamp
        let edges = vec![
            AddEdge {
                id: Id::new(),
                source_node_id: Id::from_bytes([0u8; 16]),
                target_node_id: Id::from_bytes([0u8; 16]),
                ts_millis: TimestampMilli(base_ts),
                name: "edge_a".to_string(),
            },
            AddEdge {
                id: Id::new(),
                source_node_id: Id::from_bytes([0u8; 16]),
                target_node_id: Id::from_bytes([1u8; 16]),
                ts_millis: TimestampMilli(base_ts + 1000),
                name: "edge_b".to_string(),
            },
            AddEdge {
                id: Id::new(),
                source_node_id: Id::from_bytes([1u8; 16]),
                target_node_id: Id::from_bytes([0u8; 16]),
                ts_millis: TimestampMilli(base_ts + 2000),
                name: "edge_c".to_string(),
            },
            AddEdge {
                id: Id::new(),
                source_node_id: Id::from_bytes([1u8; 16]),
                target_node_id: Id::from_bytes([1u8; 16]),
                ts_millis: TimestampMilli(base_ts + 3000),
                name: "edge_d".to_string(),
            },
            // Add edge with same source and target but different name
            AddEdge {
                id: Id::new(),
                source_node_id: Id::from_bytes([0u8; 16]),
                target_node_id: Id::from_bytes([0u8; 16]),
                ts_millis: TimestampMilli(base_ts + 4000),
                name: "edge_z".to_string(),
            },
        ];

        // Generate key-value pairs and serialize the keys
        let serialized_keys: Vec<(Vec<u8>, String)> = edges
            .iter()
            .map(|args| {
                let (key, _value) = ForwardEdges::record_from(args);
                let key_bytes = ForwardEdges::key_to_bytes(&key).unwrap();
                (key_bytes, args.name.clone())
            })
            .collect();

        // Clone for comparison
        let mut sorted_keys = serialized_keys.clone();

        // Sort by the serialized byte representation (lexicographic order)
        sorted_keys.sort_by(|a, b| a.0.cmp(&b.0));

        // Verify that sorting by bytes doesn't change the order we expect:
        // - Keys should be ordered by (source_id, destination_id, name)
        // - MessagePack serialization should preserve this ordering

        // Expected order based on key structure (source, dest, name):
        // 1. ([0..], [0..], "edge_a")
        // 2. ([0..], [0..], "edge_z")
        // 3. ([0..], [1..], "edge_b")
        // 4. ([1..], [0..], "edge_c")
        // 5. ([1..], [1..], "edge_d")

        assert_eq!(sorted_keys[0].1, "edge_a");
        assert_eq!(sorted_keys[1].1, "edge_z");
        assert_eq!(sorted_keys[2].1, "edge_b");
        assert_eq!(sorted_keys[3].1, "edge_c");
        assert_eq!(sorted_keys[4].1, "edge_d");

        // Verify that the serialized keys are actually in lexicographic order
        for i in 0..sorted_keys.len() - 1 {
            assert!(
                sorted_keys[i].0 <= sorted_keys[i + 1].0,
                "Keys should be in lexicographic order: {:?} should be <= {:?}",
                sorted_keys[i].0,
                sorted_keys[i + 1].0
            );
        }
    }
}
