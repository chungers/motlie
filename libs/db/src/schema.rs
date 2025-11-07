use crate::graph::{ColumnFamilyRecord, PutCf, StorageOperation};
use crate::query::{DstId, SrcId};
use crate::DataUrl;
use crate::TimestampMilli;
use crate::{AddEdge, AddFragment, AddNode, Id};
use serde::{Deserialize, Serialize};

pub(crate) struct Plan {}
impl Plan {
    pub(crate) fn create_node(
        op: &AddNode,
    ) -> Result<Vec<StorageOperation>, rmp_serde::encode::Error> {
        Ok(vec![StorageOperation::PutCf(PutCf(
            Nodes::CF_NAME, // Nodes (id)
            Nodes::create_bytes(op)?,
        ))])
    }
    pub(crate) fn create_edge(
        op: &AddEdge,
    ) -> Result<Vec<StorageOperation>, rmp_serde::encode::Error> {
        Ok(vec![
            StorageOperation::PutCf(PutCf(
                Edges::CF_NAME, // Edges (id)
                Edges::create_bytes(op)?,
            )),
            StorageOperation::PutCf(PutCf(
                ForwardEdges::CF_NAME, // ForwardEdges (src, dst, name)
                ForwardEdges::create_bytes(op)?,
            )),
            StorageOperation::PutCf(PutCf(
                ReverseEdges::CF_NAME, // ReverseEdges (dst, src, name)
                ReverseEdges::create_bytes(op)?,
            )),
        ])
    }
    pub(crate) fn create_fragment(
        op: &AddFragment,
    ) -> Result<Vec<StorageOperation>, rmp_serde::encode::Error> {
        Ok(vec![StorageOperation::PutCf(PutCf(
            Fragments::CF_NAME, // Fragments (id)
            Fragments::create_bytes(op)?,
        ))])
    }
}

/// Nodes column family.
pub(crate) struct Nodes;

#[derive(Serialize, Deserialize)]
pub(crate) struct NodeCfKey(pub(crate) Id);

#[derive(Serialize, Deserialize)]
pub(crate) struct NodeCfValue(pub(crate) NodeName, pub(crate) NodeSummary);

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NodeSummary(pub(crate) DataUrl);

pub type NodeName = String;

impl NodeSummary {
    pub fn new(content: impl AsRef<str>) -> Self {
        NodeSummary(DataUrl::from_markdown(content.as_ref()))
    }

    pub fn content(&self) -> Result<String, crate::DataUrlError> {
        self.0.decode_string()
    }

    pub fn as_data_url(&self) -> &DataUrl {
        &self.0
    }
}

impl ColumnFamilyRecord for Nodes {
    const CF_NAME: &'static str = "nodes";
    type Key = NodeCfKey;
    type Value = NodeCfValue;
    type CreateOp = AddNode;

    fn record_from(args: &AddNode) -> (NodeCfKey, NodeCfValue) {
        let key = NodeCfKey(args.id);
        let markdown = format!("<!-- id={} -->]\n# {}\n# Summary\n", args.id, args.name);
        let value = NodeCfValue(args.name.clone(), NodeSummary::new(markdown));
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

pub(crate) struct Edges;

#[derive(Serialize, Deserialize)]
pub(crate) struct EdgeCfKey(pub(crate) Id);

#[derive(Serialize, Deserialize)]
pub(crate) struct EdgeCfValue(
    pub(crate) SrcId,       // source_id
    pub(crate) EdgeName,    // edge name
    pub(crate) DstId,       // dest_id
    pub(crate) EdgeSummary, // edge summary
);

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EdgeSummary(pub(crate) DataUrl);

impl EdgeSummary {
    pub fn new(content: impl AsRef<str>) -> Self {
        EdgeSummary(DataUrl::from_markdown(content.as_ref()))
    }

    pub fn content(&self) -> Result<String, crate::DataUrlError> {
        self.0.decode_string()
    }

    pub fn as_data_url(&self) -> &DataUrl {
        &self.0
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
            args.source_node_id,
            EdgeName(args.name.clone()),
            args.target_node_id,
            EdgeSummary::new(markdown),
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

pub(crate) struct Fragments;

#[derive(Serialize, Deserialize)]
pub(crate) struct FragmentCfKey(pub(crate) Id, pub(crate) TimestampMilli);

#[derive(Serialize, Deserialize)]
pub(crate) struct FragmentCfValue(pub(crate) FragmentContent);

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FragmentContent(pub(crate) DataUrl);

impl FragmentContent {
    pub fn new(content: impl AsRef<str>) -> Self {
        // Treat all fragment content as markdown per user requirement
        FragmentContent(DataUrl::from_markdown(content.as_ref()))
    }

    pub fn content(&self) -> Result<String, crate::DataUrlError> {
        self.0.decode_string()
    }

    pub fn as_data_url(&self) -> &DataUrl {
        &self.0
    }
}

impl ColumnFamilyRecord for Fragments {
    const CF_NAME: &'static str = "fragments";
    type Key = FragmentCfKey;
    type Value = FragmentCfValue;
    type CreateOp = AddFragment;

    fn record_from(args: &AddFragment) -> (FragmentCfKey, FragmentCfValue) {
        let key = FragmentCfKey(args.id, TimestampMilli(args.ts_millis));
        let value = FragmentCfValue(FragmentContent::new(&args.content));
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
            anyhow::bail!("Invalid FragmentCfKey length: expected 24, got {}", bytes.len());
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

pub(crate) struct ForwardEdges;

#[derive(Serialize, Deserialize)]
pub(crate) struct ForwardEdgeCfKey(
    pub(crate) EdgeSourceId,
    pub(crate) EdgeDestinationId,
    pub(crate) EdgeName,
);

#[derive(Serialize, Deserialize)]
pub(crate) struct ForwardEdgeCfValue(pub(crate) Id);

#[derive(Serialize, Deserialize)]
pub(crate) struct EdgeSourceId(pub(crate) Id);

#[derive(Serialize, Deserialize)]
pub(crate) struct EdgeDestinationId(pub(crate) Id);

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EdgeName(pub String);

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
        let value = ForwardEdgeCfValue(args.id);
        (key, value)
    }

    fn key_to_bytes(key: &Self::Key) -> Vec<u8> {
        // ForwardEdgeCfKey(EdgeSourceId, EdgeDestinationId, EdgeName)
        // Layout: [src_id (16)] + [dst_id (16)] + [name UTF-8 bytes]
        let name_bytes = key.2 .0.as_bytes();
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
            EdgeName(name),
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

pub(crate) struct ReverseEdges;

#[derive(Serialize, Deserialize)]
pub(crate) struct ReverseEdgeCfKey(
    pub(crate) EdgeDestinationId,
    pub(crate) EdgeSourceId,
    pub(crate) EdgeName,
);

#[derive(Serialize, Deserialize)]
pub(crate) struct ReverseEdgeCfValue(pub(crate) Id);

impl ColumnFamilyRecord for ReverseEdges {
    const CF_NAME: &'static str = "reverse_edges";
    type Key = ReverseEdgeCfKey;
    type Value = ReverseEdgeCfValue;
    type CreateOp = AddEdge;

    fn record_from(args: &AddEdge) -> (ReverseEdgeCfKey, ReverseEdgeCfValue) {
        let key = ReverseEdgeCfKey(
            EdgeDestinationId(args.target_node_id),
            EdgeSourceId(args.source_node_id),
            EdgeName(args.name.clone()),
        );
        let value = ReverseEdgeCfValue(args.id);
        (key, value)
    }

    fn key_to_bytes(key: &Self::Key) -> Vec<u8> {
        // ReverseEdgeCfKey(EdgeDestinationId, EdgeSourceId, EdgeName)
        // Layout: [dst_id (16)] + [src_id (16)] + [name UTF-8 bytes]
        let name_bytes = key.2 .0.as_bytes();
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
            EdgeName(name),
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
}
