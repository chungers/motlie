use crate::graph::ColumnFamilyRecord;
use crate::query::{DstId, SrcId};
use crate::DataUrl;
use crate::TimestampMilli;
use crate::{AddEdge, AddFragment, AddNode, Id};
use serde::{Deserialize, Serialize};

/// Support for temporal queries
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ValidTemporalRange(pub Option<StartTimestamp>, pub Option<UntilTimestamp>);

pub type StartTimestamp = TimestampMilli;
pub type UntilTimestamp = TimestampMilli;

impl ValidTemporalRange {
    /// Create a new temporal range with no constraints (always valid)
    pub fn always_valid() -> Option<Self> {
        None
    }

    /// Create a temporal range valid from a start time (inclusive)
    pub fn valid_from(start: TimestampMilli) -> Option<Self> {
        Some(ValidTemporalRange(Some(start), None))
    }

    /// Create a temporal range valid until an end time (exclusive)
    pub fn valid_until(until: TimestampMilli) -> Option<Self> {
        Some(ValidTemporalRange(None, Some(until)))
    }

    /// Create a temporal range valid between start (inclusive) and until (exclusive)
    pub fn valid_between(start: TimestampMilli, until: TimestampMilli) -> Option<Self> {
        Some(ValidTemporalRange(Some(start), Some(until)))
    }

    /// Check if a timestamp is valid according to this temporal range
    pub fn is_valid_at(&self, query_time: TimestampMilli) -> bool {
        let after_start = match self.0 {
            None => true,
            Some(start) => query_time.0 >= start.0,
        };
        let before_until = match self.1 {
            None => true,
            Some(until) => query_time.0 < until.0,
        };
        after_start && before_until
    }
}

/// Helper function to check if a record is valid at a given time
/// Returns true if temporal_range is None (always valid) or if query_time falls within range
pub fn is_valid_at_time(
    temporal_range: &Option<ValidTemporalRange>,
    query_time: TimestampMilli,
) -> bool {
    match temporal_range {
        None => true, // No temporal constraint = always valid
        Some(range) => range.is_valid_at(query_time),
    }
}

/// Nodes column family.
pub(crate) struct Nodes;
#[derive(Serialize, Deserialize)]
pub(crate) struct NodeCfKey(pub(crate) Id);
#[derive(Serialize, Deserialize)]
pub(crate) struct NodeCfValue(
    pub(crate) Option<ValidTemporalRange>,
    pub(crate) NodeName,
    pub(crate) NodeSummary,
);

/// Edges column family.
pub(crate) struct Edges;
#[derive(Serialize, Deserialize)]
pub(crate) struct EdgeCfKey(pub(crate) Id);
#[derive(Serialize, Deserialize)]
pub(crate) struct EdgeCfValue(
    pub(crate) Option<ValidTemporalRange>, // temporal validity
    pub(crate) SrcId,                       // source_id
    pub(crate) EdgeName,                    // edge name
    pub(crate) DstId,                       // dest_id
    pub(crate) EdgeSummary,                 // edge summary
);

/// Fragments column family.
pub(crate) struct Fragments;
#[derive(Serialize, Deserialize)]
pub(crate) struct FragmentCfKey(pub(crate) Id, pub(crate) TimestampMilli);
#[derive(Serialize, Deserialize)]
pub(crate) struct FragmentCfValue(
    pub(crate) Option<ValidTemporalRange>,
    pub(crate) FragmentContent,
);

/// Forward edges column family.
pub(crate) struct ForwardEdges;
#[derive(Serialize, Deserialize)]
pub(crate) struct ForwardEdgeCfKey(
    pub(crate) EdgeSourceId,
    pub(crate) EdgeDestinationId,
    pub(crate) EdgeName,
);
#[derive(Serialize, Deserialize)]
pub(crate) struct ForwardEdgeCfValue(
    pub(crate) Option<ValidTemporalRange>,
    pub(crate) Id,
);
#[derive(Serialize, Deserialize)]
pub(crate) struct EdgeSourceId(pub(crate) Id);
#[derive(Serialize, Deserialize)]
pub(crate) struct EdgeDestinationId(pub(crate) Id);
#[derive(Serialize, Deserialize, Debug, Clone)]

/// Reverse edges column family.
pub(crate) struct ReverseEdges;
#[derive(Serialize, Deserialize)]
pub(crate) struct ReverseEdgeCfKey(
    pub(crate) EdgeDestinationId,
    pub(crate) EdgeSourceId,
    pub(crate) EdgeName,
);
#[derive(Serialize, Deserialize)]
pub(crate) struct ReverseEdgeCfValue(
    pub(crate) Option<ValidTemporalRange>,
    pub(crate) Id,
);

/// Node names column family.
pub(crate) struct NodeNames;
#[derive(Serialize, Deserialize)]
pub(crate) struct NodeNameCfKey(pub(crate) NodeName, pub(crate) Id);
#[derive(Serialize, Deserialize)]
pub(crate) struct NodeNameCfValue(pub(crate) Option<ValidTemporalRange>);

/// Edge names column family.
pub(crate) struct EdgeNames;
#[derive(Serialize, Deserialize)]
pub(crate) struct EdgeNameCfKey(
    pub(crate) EdgeName,
    pub(crate) Id, // edge id
    pub(crate) EdgeDestinationId,
    pub(crate) EdgeSourceId,
);
#[derive(Serialize, Deserialize)]
pub(crate) struct EdgeNameCfValue(pub(crate) Option<ValidTemporalRange>);

pub type NodeName = String;
pub type EdgeName = String;
pub type NodeSummary = DataUrl;
pub type EdgeSummary = DataUrl;
pub type FragmentContent = DataUrl;

impl ColumnFamilyRecord for Nodes {
    const CF_NAME: &'static str = "nodes";
    type Key = NodeCfKey;
    type Value = NodeCfValue;
    type CreateOp = AddNode;

    fn record_from(args: &AddNode) -> (NodeCfKey, NodeCfValue) {
        let key = NodeCfKey(args.id);
        let markdown = format!("<!-- id={} -->]\n# {}\n# Summary\n", args.id, args.name);
        let value = NodeCfValue(
            args.temporal_range.clone(),
            args.name.clone(),
            DataUrl::from_markdown(markdown),
        );
        (key, value)
    }

    fn key_to_bytes(key: &Self::Key) -> Vec<u8> {
        // NodeCfKey(Id) -> just the 16-byte Id
        key.0.into_bytes().to_vec()
    }

    fn key_from_bytes(bytes: &[u8]) -> Result<Self::Key, anyhow::Error> {
        if bytes.len() != 16 {
            anyhow::bail!("Invalid NodeCfKey length: expected 16, got {}", bytes.len());
        }
        let mut id_bytes = [0u8; 16];
        id_bytes.copy_from_slice(bytes);
        Ok(NodeCfKey(Id::from_bytes(id_bytes)))
    }

    fn column_family_options() -> rocksdb::Options {
        // Point lookups by Id only, no prefix scanning needed
        rocksdb::Options::default()
    }
}

impl ColumnFamilyRecord for Edges {
    const CF_NAME: &'static str = "edges";
    type Key = EdgeCfKey;
    type Value = EdgeCfValue;
    type CreateOp = AddEdge;

    fn record_from(args: &AddEdge) -> (EdgeCfKey, EdgeCfValue) {
        let key = EdgeCfKey(args.id);
        let markdown = format!("<!-- id={} -->]\n# {}\n# Summary\n", args.id, args.name);
        let value = EdgeCfValue(
            args.temporal_range.clone(),
            args.source_node_id,
            args.name.clone(),
            args.target_node_id,
            DataUrl::from_markdown(markdown),
        );
        (key, value)
    }

    fn key_to_bytes(key: &Self::Key) -> Vec<u8> {
        // EdgeCfKey(Id) -> just the 16-byte Id
        key.0.into_bytes().to_vec()
    }

    fn key_from_bytes(bytes: &[u8]) -> Result<Self::Key, anyhow::Error> {
        if bytes.len() != 16 {
            anyhow::bail!("Invalid EdgeCfKey length: expected 16, got {}", bytes.len());
        }
        let mut id_bytes = [0u8; 16];
        id_bytes.copy_from_slice(bytes);
        Ok(EdgeCfKey(Id::from_bytes(id_bytes)))
    }

    fn column_family_options() -> rocksdb::Options {
        // Point lookups by Id only, no prefix scanning needed
        rocksdb::Options::default()
    }
}

impl ColumnFamilyRecord for Fragments {
    const CF_NAME: &'static str = "fragments";
    type Key = FragmentCfKey;
    type Value = FragmentCfValue;
    type CreateOp = AddFragment;

    fn record_from(args: &AddFragment) -> (FragmentCfKey, FragmentCfValue) {
        let key = FragmentCfKey(args.id, args.ts_millis);
        let value = FragmentCfValue(args.temporal_range.clone(), args.content.clone());
        (key, value)
    }

    fn key_to_bytes(key: &Self::Key) -> Vec<u8> {
        // FragmentCfKey(Id, TimestampMilli)
        // Layout: [Id bytes (16)] + [timestamp big-endian (8)]
        let mut bytes = Vec::with_capacity(24);
        bytes.extend_from_slice(&key.0.into_bytes());
        bytes.extend_from_slice(&key.1 .0.to_be_bytes());
        bytes
    }

    fn key_from_bytes(bytes: &[u8]) -> Result<Self::Key, anyhow::Error> {
        if bytes.len() != 24 {
            anyhow::bail!(
                "Invalid FragmentCfKey length: expected 24, got {}",
                bytes.len()
            );
        }

        let mut id_bytes = [0u8; 16];
        id_bytes.copy_from_slice(&bytes[0..16]);

        let mut ts_bytes = [0u8; 8];
        ts_bytes.copy_from_slice(&bytes[16..24]);
        let timestamp = u64::from_be_bytes(ts_bytes);

        Ok(FragmentCfKey(
            Id::from_bytes(id_bytes),
            TimestampMilli(timestamp),
        ))
    }

    fn column_family_options() -> rocksdb::Options {
        use rocksdb::SliceTransform;

        let mut opts = rocksdb::Options::default();

        // Key layout: [Id (16 bytes)] + [TimestampMilli (8 bytes)]
        // Use 16-byte prefix to scan all fragments for a given Id
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(16));

        // Enable prefix bloom filter for fast prefix existence checks
        opts.set_memtable_prefix_bloom_ratio(0.2);

        opts
    }
}

impl ColumnFamilyRecord for ForwardEdges {
    const CF_NAME: &'static str = "forward_edges";
    type Key = ForwardEdgeCfKey;
    type Value = ForwardEdgeCfValue;
    type CreateOp = AddEdge;

    fn record_from(args: &AddEdge) -> (ForwardEdgeCfKey, ForwardEdgeCfValue) {
        let key = ForwardEdgeCfKey(
            EdgeSourceId(args.source_node_id),
            EdgeDestinationId(args.target_node_id),
            args.name.clone(),
        );
        let value = ForwardEdgeCfValue(args.temporal_range.clone(), args.id);
        (key, value)
    }

    fn key_to_bytes(key: &Self::Key) -> Vec<u8> {
        // ForwardEdgeCfKey(EdgeSourceId, EdgeDestinationId, EdgeName)
        // Layout: [src_id (16)] + [dst_id (16)] + [name UTF-8 bytes]
        let name_bytes = key.2.as_bytes();
        let mut bytes = Vec::with_capacity(32 + name_bytes.len());
        bytes.extend_from_slice(&key.0 .0.into_bytes());
        bytes.extend_from_slice(&key.1 .0.into_bytes());
        bytes.extend_from_slice(name_bytes);
        bytes
    }

    fn key_from_bytes(bytes: &[u8]) -> Result<Self::Key, anyhow::Error> {
        if bytes.len() < 32 {
            anyhow::bail!(
                "Invalid ForwardEdgeCfKey length: expected >= 32, got {}",
                bytes.len()
            );
        }

        let mut src_id_bytes = [0u8; 16];
        src_id_bytes.copy_from_slice(&bytes[0..16]);

        let mut dst_id_bytes = [0u8; 16];
        dst_id_bytes.copy_from_slice(&bytes[16..32]);

        let name_bytes = &bytes[32..];
        let name = String::from_utf8(name_bytes.to_vec())
            .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in EdgeName: {}", e))?;

        Ok(ForwardEdgeCfKey(
            EdgeSourceId(Id::from_bytes(src_id_bytes)),
            EdgeDestinationId(Id::from_bytes(dst_id_bytes)),
            name,
        ))
    }

    fn column_family_options() -> rocksdb::Options {
        use rocksdb::SliceTransform;

        let mut opts = rocksdb::Options::default();

        // Key layout: [src_id (16)] + [dst_id (16)] + [name (variable)]
        // Use 16-byte prefix to scan all edges from a source node
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(16));

        // Enable prefix bloom filter for O(1) prefix existence check
        opts.set_memtable_prefix_bloom_ratio(0.2);

        opts
    }
}

impl ColumnFamilyRecord for ReverseEdges {
    const CF_NAME: &'static str = "reverse_edges";
    type Key = ReverseEdgeCfKey;
    type Value = ReverseEdgeCfValue;
    type CreateOp = AddEdge;

    fn record_from(args: &AddEdge) -> (ReverseEdgeCfKey, ReverseEdgeCfValue) {
        let key = ReverseEdgeCfKey(
            EdgeDestinationId(args.target_node_id),
            EdgeSourceId(args.source_node_id),
            args.name.clone(),
        );
        let value = ReverseEdgeCfValue(args.temporal_range.clone(), args.id);
        (key, value)
    }

    fn key_to_bytes(key: &Self::Key) -> Vec<u8> {
        // ReverseEdgeCfKey(EdgeDestinationId, EdgeSourceId, EdgeName)
        // Layout: [dst_id (16)] + [src_id (16)] + [name UTF-8 bytes]
        let name_bytes = key.2.as_bytes();
        let mut bytes = Vec::with_capacity(32 + name_bytes.len());
        bytes.extend_from_slice(&key.0 .0.into_bytes());
        bytes.extend_from_slice(&key.1 .0.into_bytes());
        bytes.extend_from_slice(name_bytes);
        bytes
    }

    fn key_from_bytes(bytes: &[u8]) -> Result<Self::Key, anyhow::Error> {
        if bytes.len() < 32 {
            anyhow::bail!(
                "Invalid ReverseEdgeCfKey length: expected >= 32, got {}",
                bytes.len()
            );
        }

        let mut dst_id_bytes = [0u8; 16];
        dst_id_bytes.copy_from_slice(&bytes[0..16]);

        let mut src_id_bytes = [0u8; 16];
        src_id_bytes.copy_from_slice(&bytes[16..32]);

        let name_bytes = &bytes[32..];
        let name = String::from_utf8(name_bytes.to_vec())
            .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in EdgeName: {}", e))?;

        Ok(ReverseEdgeCfKey(
            EdgeDestinationId(Id::from_bytes(dst_id_bytes)),
            EdgeSourceId(Id::from_bytes(src_id_bytes)),
            name,
        ))
    }

    fn column_family_options() -> rocksdb::Options {
        use rocksdb::SliceTransform;

        let mut opts = rocksdb::Options::default();

        // Key layout: [dst_id (16)] + [src_id (16)] + [name (variable)]
        // Use 16-byte prefix to scan all edges to a destination node
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(16));

        // Enable prefix bloom filter for O(1) prefix existence check
        opts.set_memtable_prefix_bloom_ratio(0.2);

        opts
    }
}

impl ColumnFamilyRecord for NodeNames {
    const CF_NAME: &'static str = "node_names";
    type Key = NodeNameCfKey;
    type Value = NodeNameCfValue;
    type CreateOp = AddNode;

    fn record_from(args: &AddNode) -> (NodeNameCfKey, NodeNameCfValue) {
        let key = NodeNameCfKey(args.name.clone(), args.id);
        let value = NodeNameCfValue(args.temporal_range.clone());
        (key, value)
    }

    fn key_to_bytes(key: &Self::Key) -> Vec<u8> {
        // NodeNamesCfKey(NodeName, Id)
        // Layout: [name UTF-8 bytes] + [node_id (16)]
        let name_bytes = key.0.as_bytes();
        let mut bytes = Vec::with_capacity(name_bytes.len() + 16);
        bytes.extend_from_slice(name_bytes);
        bytes.extend_from_slice(&key.1.into_bytes());
        bytes
    }

    fn key_from_bytes(bytes: &[u8]) -> Result<Self::Key, anyhow::Error> {
        if bytes.len() < 16 {
            anyhow::bail!(
                "Invalid NodeNamesCfKey length: expected >= 16, got {}",
                bytes.len()
            );
        }

        // The name is everything before the last 16 bytes (which is the node_id)
        let name_end = bytes.len() - 16;
        let name_bytes = &bytes[0..name_end];
        let name = String::from_utf8(name_bytes.to_vec())
            .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in NodeName: {}", e))?;

        let mut node_id_bytes = [0u8; 16];
        node_id_bytes.copy_from_slice(&bytes[name_end..name_end + 16]);

        Ok(NodeNameCfKey(name, Id::from_bytes(node_id_bytes)))
    }

    fn column_family_options() -> rocksdb::Options {
        // Key layout: [name (variable)] + [node_id (16)]
        // No prefix extraction needed for this column family
        // as queries will typically be by name which is variable-length at the start
        rocksdb::Options::default()
    }
}

impl ColumnFamilyRecord for EdgeNames {
    const CF_NAME: &'static str = "edge_names";
    type Key = EdgeNameCfKey;
    type Value = EdgeNameCfValue;
    type CreateOp = AddEdge;

    fn record_from(args: &AddEdge) -> (EdgeNameCfKey, EdgeNameCfValue) {
        let key = EdgeNameCfKey(
            args.name.clone(),
            args.id,
            EdgeDestinationId(args.target_node_id),
            EdgeSourceId(args.source_node_id),
        );
        let value = EdgeNameCfValue(args.temporal_range.clone());
        (key, value)
    }

    fn key_to_bytes(key: &Self::Key) -> Vec<u8> {
        // EdgeNamesCfKey(EdgeName, Id, EdgeDestinationId, EdgeSourceId)
        // Layout: [name UTF-8 bytes] + [edge_id (16)] + [dst_id (16)] + [src_id (16)]
        let name_bytes = key.0.as_bytes();
        let mut bytes = Vec::with_capacity(name_bytes.len() + 48);
        bytes.extend_from_slice(name_bytes);
        bytes.extend_from_slice(&key.1.into_bytes());
        bytes.extend_from_slice(&key.2 .0.into_bytes());
        bytes.extend_from_slice(&key.3 .0.into_bytes());
        bytes
    }

    fn key_from_bytes(bytes: &[u8]) -> Result<Self::Key, anyhow::Error> {
        if bytes.len() < 48 {
            anyhow::bail!(
                "Invalid EdgeNamesCfKey length: expected >= 48, got {}",
                bytes.len()
            );
        }

        // The name is everything before the last 48 bytes (which are edge_id + dst_id + src_id)
        let name_end = bytes.len() - 48;
        let name_bytes = &bytes[0..name_end];
        let name = String::from_utf8(name_bytes.to_vec())
            .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in EdgeName: {}", e))?;

        let mut edge_id_bytes = [0u8; 16];
        edge_id_bytes.copy_from_slice(&bytes[name_end..name_end + 16]);

        let mut dst_id_bytes = [0u8; 16];
        dst_id_bytes.copy_from_slice(&bytes[name_end + 16..name_end + 32]);

        let mut src_id_bytes = [0u8; 16];
        src_id_bytes.copy_from_slice(&bytes[name_end + 32..name_end + 48]);

        Ok(EdgeNameCfKey(
            name,
            Id::from_bytes(edge_id_bytes),
            EdgeDestinationId(Id::from_bytes(dst_id_bytes)),
            EdgeSourceId(Id::from_bytes(src_id_bytes)),
        ))
    }

    fn column_family_options() -> rocksdb::Options {
        // Key layout: [name (variable)] + [dst_id (16)] + [src_id (16)] + [edge_id (16)]
        // No prefix extraction needed for this column family
        // as queries will typically be by name which is variable-length at the start
        rocksdb::Options::default()
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
    NodeNames::CF_NAME,
    EdgeNames::CF_NAME,
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
                temporal_range: None,
            },
            AddEdge {
                id: Id::new(),
                source_node_id: Id::from_bytes([0u8; 16]),
                target_node_id: Id::from_bytes([1u8; 16]),
                ts_millis: TimestampMilli(base_ts + 1000),
                name: "edge_b".to_string(),
                temporal_range: None,
            },
            AddEdge {
                id: Id::new(),
                source_node_id: Id::from_bytes([1u8; 16]),
                target_node_id: Id::from_bytes([0u8; 16]),
                ts_millis: TimestampMilli(base_ts + 2000),
                name: "edge_c".to_string(),
                temporal_range: None,
            },
            AddEdge {
                id: Id::new(),
                source_node_id: Id::from_bytes([1u8; 16]),
                target_node_id: Id::from_bytes([1u8; 16]),
                ts_millis: TimestampMilli(base_ts + 3000),
                name: "edge_d".to_string(),
                temporal_range: None,
            },
            // Add edge with same source and target but different name
            AddEdge {
                id: Id::new(),
                source_node_id: Id::from_bytes([0u8; 16]),
                target_node_id: Id::from_bytes([0u8; 16]),
                ts_millis: TimestampMilli(base_ts + 4000),
                name: "edge_z".to_string(),
                temporal_range: None,
            },
        ];

        // Generate key-value pairs and serialize the keys
        let serialized_keys: Vec<(Vec<u8>, EdgeName)> = edges
            .iter()
            .map(|args| {
                let (key, _value) = ForwardEdges::record_from(args);
                let key_bytes = ForwardEdges::key_to_bytes(&key);
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

    #[test]
    fn test_valid_temporal_range_always_valid() {
        let range = ValidTemporalRange::always_valid();
        assert!(range.is_none(), "always_valid should return None");
    }

    #[test]
    fn test_valid_temporal_range_valid_from() {
        let start = TimestampMilli(1000);
        let range = ValidTemporalRange::valid_from(start);

        assert!(range.is_some());
        let range = range.unwrap();
        assert_eq!(range.0, Some(TimestampMilli(1000)));
        assert_eq!(range.1, None);
    }

    #[test]
    fn test_valid_temporal_range_valid_until() {
        let until = TimestampMilli(2000);
        let range = ValidTemporalRange::valid_until(until);

        assert!(range.is_some());
        let range = range.unwrap();
        assert_eq!(range.0, None);
        assert_eq!(range.1, Some(TimestampMilli(2000)));
    }

    #[test]
    fn test_valid_temporal_range_valid_between() {
        let start = TimestampMilli(1000);
        let until = TimestampMilli(2000);
        let range = ValidTemporalRange::valid_between(start, until);

        assert!(range.is_some());
        let range = range.unwrap();
        assert_eq!(range.0, Some(TimestampMilli(1000)));
        assert_eq!(range.1, Some(TimestampMilli(2000)));
    }

    #[test]
    fn test_is_valid_at_with_start_only() {
        let range = ValidTemporalRange(Some(TimestampMilli(1000)), None);

        // Before start - invalid
        assert!(!range.is_valid_at(TimestampMilli(999)));

        // At start - valid (inclusive)
        assert!(range.is_valid_at(TimestampMilli(1000)));

        // After start - valid
        assert!(range.is_valid_at(TimestampMilli(1001)));
        assert!(range.is_valid_at(TimestampMilli(9999)));
    }

    #[test]
    fn test_is_valid_at_with_until_only() {
        let range = ValidTemporalRange(None, Some(TimestampMilli(2000)));

        // Before until - valid
        assert!(range.is_valid_at(TimestampMilli(0)));
        assert!(range.is_valid_at(TimestampMilli(1999)));

        // At until - invalid (exclusive)
        assert!(!range.is_valid_at(TimestampMilli(2000)));

        // After until - invalid
        assert!(!range.is_valid_at(TimestampMilli(2001)));
    }

    #[test]
    fn test_is_valid_at_with_both_boundaries() {
        let range = ValidTemporalRange(Some(TimestampMilli(1000)), Some(TimestampMilli(2000)));

        // Before start - invalid
        assert!(!range.is_valid_at(TimestampMilli(999)));

        // At start - valid (inclusive)
        assert!(range.is_valid_at(TimestampMilli(1000)));

        // Between start and until - valid
        assert!(range.is_valid_at(TimestampMilli(1500)));
        assert!(range.is_valid_at(TimestampMilli(1999)));

        // At until - invalid (exclusive)
        assert!(!range.is_valid_at(TimestampMilli(2000)));

        // After until - invalid
        assert!(!range.is_valid_at(TimestampMilli(2001)));
    }

    #[test]
    fn test_is_valid_at_with_no_boundaries() {
        let range = ValidTemporalRange(None, None);

        // Always valid regardless of timestamp
        assert!(range.is_valid_at(TimestampMilli(0)));
        assert!(range.is_valid_at(TimestampMilli(1000)));
        assert!(range.is_valid_at(TimestampMilli(u64::MAX)));
    }

    #[test]
    fn test_is_valid_at_time_with_none() {
        let temporal_range: Option<ValidTemporalRange> = None;

        // None means always valid
        assert!(is_valid_at_time(&temporal_range, TimestampMilli(0)));
        assert!(is_valid_at_time(&temporal_range, TimestampMilli(1000)));
        assert!(is_valid_at_time(&temporal_range, TimestampMilli(u64::MAX)));
    }

    #[test]
    fn test_is_valid_at_time_with_range() {
        let temporal_range = Some(ValidTemporalRange(
            Some(TimestampMilli(1000)),
            Some(TimestampMilli(2000)),
        ));

        // Before range
        assert!(!is_valid_at_time(&temporal_range, TimestampMilli(999)));

        // Within range
        assert!(is_valid_at_time(&temporal_range, TimestampMilli(1000)));
        assert!(is_valid_at_time(&temporal_range, TimestampMilli(1500)));
        assert!(is_valid_at_time(&temporal_range, TimestampMilli(1999)));

        // At/after end
        assert!(!is_valid_at_time(&temporal_range, TimestampMilli(2000)));
        assert!(!is_valid_at_time(&temporal_range, TimestampMilli(2001)));
    }

    #[test]
    fn test_is_valid_at_time_edge_cases() {
        // Test with start only
        let from_only = Some(ValidTemporalRange(Some(TimestampMilli(100)), None));
        assert!(!is_valid_at_time(&from_only, TimestampMilli(99)));
        assert!(is_valid_at_time(&from_only, TimestampMilli(100)));
        assert!(is_valid_at_time(&from_only, TimestampMilli(u64::MAX)));

        // Test with until only
        let until_only = Some(ValidTemporalRange(None, Some(TimestampMilli(200))));
        assert!(is_valid_at_time(&until_only, TimestampMilli(0)));
        assert!(is_valid_at_time(&until_only, TimestampMilli(199)));
        assert!(!is_valid_at_time(&until_only, TimestampMilli(200)));

        // Test with no constraints (Some with both None)
        let no_constraints = Some(ValidTemporalRange(None, None));
        assert!(is_valid_at_time(&no_constraints, TimestampMilli(0)));
        assert!(is_valid_at_time(&no_constraints, TimestampMilli(u64::MAX)));
    }

    #[test]
    fn test_valid_temporal_range_serialization() {
        // Test that ValidTemporalRange can be serialized and deserialized
        let range = ValidTemporalRange(Some(TimestampMilli(1000)), Some(TimestampMilli(2000)));

        let serialized = rmp_serde::to_vec(&range).expect("Should serialize");
        let deserialized: ValidTemporalRange =
            rmp_serde::from_slice(&serialized).expect("Should deserialize");

        assert_eq!(range, deserialized);
    }

    #[test]
    fn test_valid_temporal_range_clone_and_equality() {
        let range1 = ValidTemporalRange(Some(TimestampMilli(1000)), Some(TimestampMilli(2000)));
        let range2 = range1.clone();

        assert_eq!(range1, range2);

        let range3 = ValidTemporalRange(Some(TimestampMilli(1000)), Some(TimestampMilli(2001)));
        assert_ne!(range1, range3);
    }
}
