use super::mutation::{AddEdge, AddEdgeFragment, AddNode, AddNodeFragment};
use super::name_hash::NameHash;
use super::summary_hash::SummaryHash;
use super::BlockCacheConfig;
use super::ColumnFamilyRecord;
use super::HotColumnFamilyRecord;
use super::ValidRangePatchable;
use crate::DataUrl;
use crate::Id;
use crate::TimestampMilli;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

// Re-export TemporalRange and related types from crate root for convenience
pub use crate::{is_valid_at_time, StartTimestamp, TemporalRange, UntilTimestamp};

// ============================================================================
// Names Column Family (for name interning)
// ============================================================================

/// Names column family for storing hash→name mappings.
///
/// This CF enables name interning: variable-length names are replaced with
/// fixed 8-byte hashes in edge and node keys, while the full names are
/// stored here for resolution.
///
/// Key: NameHash (8 bytes)
/// Value: String (the full name)
pub(crate) struct Names;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct NamesCfKey(pub(crate) NameHash);

#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct NamesCfValue(pub(crate) String);

impl Names {
    pub const CF_NAME: &'static str = "names";

    /// Serialize the key to bytes (8 bytes, fixed size).
    pub fn key_to_bytes(key: &NamesCfKey) -> Vec<u8> {
        key.0.as_bytes().to_vec()
    }

    /// Deserialize the key from bytes.
    pub fn key_from_bytes(bytes: &[u8]) -> Result<NamesCfKey, anyhow::Error> {
        if bytes.len() != NameHash::SIZE {
            anyhow::bail!(
                "Invalid NamesCfKey length: expected {}, got {}",
                NameHash::SIZE,
                bytes.len()
            );
        }
        let mut hash_bytes = [0u8; 8];
        hash_bytes.copy_from_slice(bytes);
        Ok(NamesCfKey(NameHash::from_bytes(hash_bytes)))
    }

    /// Serialize the value to bytes (LZ4 compressed MessagePack).
    pub fn value_to_bytes(value: &NamesCfValue) -> Result<Vec<u8>, anyhow::Error> {
        let msgpack_bytes = rmp_serde::to_vec(value)?;
        let compressed = lz4::block::compress(&msgpack_bytes, None, true)?;
        Ok(compressed)
    }

    /// Deserialize the value from bytes.
    pub fn value_from_bytes(bytes: &[u8]) -> Result<NamesCfValue, anyhow::Error> {
        let decompressed = lz4::block::decompress(bytes, None)?;
        let value: NamesCfValue = rmp_serde::from_slice(&decompressed)?;
        Ok(value)
    }

    /// Configure RocksDB options for this column family.
    ///
    /// Names CF is small and hot - optimize for point lookups.
    pub fn column_family_options() -> rocksdb::Options {
        let mut opts = rocksdb::Options::default();

        // Enable bloom filter for fast point lookups
        let mut block_opts = rocksdb::BlockBasedOptions::default();
        block_opts.set_bloom_filter(10.0, false); // 10 bits per key
        opts.set_block_based_table_factory(&block_opts);

        opts
    }

    /// Configure RocksDB options with shared block cache.
    ///
    /// Names CF is small and hot - use high cache priority to keep it in memory.
    pub fn column_family_options_with_cache(
        cache: &rocksdb::Cache,
        config: &BlockCacheConfig,
    ) -> rocksdb::Options {
        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();

        // Shared block cache
        block_opts.set_block_cache(cache);
        block_opts.set_block_size(config.graph_block_size);

        // Cache index and filter blocks for Names CF
        // Note: The Rust bindings don't expose high_priority setting, but setting
        // cache_index_and_filter_blocks=true keeps metadata in cache which helps.
        if config.cache_index_and_filter_blocks {
            block_opts.set_cache_index_and_filter_blocks(true);
        }
        if config.pin_l0_filter_and_index {
            block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
        }

        // Enable bloom filter for fast point lookups
        block_opts.set_bloom_filter(10.0, false); // 10 bits per key

        opts.set_block_based_table_factory(&block_opts);
        opts
    }
}

/// Nodes column family (HOT - optimized for graph traversal).
///
/// After Phase 2 blob separation:
/// - Value contains `Option<SummaryHash>` instead of inline `NodeSummary`
/// - Full summary content is stored in `NodeSummaries` CF (COLD)
/// - Value size reduced from ~200-500 bytes to ~26 bytes
pub(crate) struct Nodes;

#[derive(Serialize, Deserialize)]
pub(crate) struct NodeCfKey(pub(crate) Id);

/// Node value - optimized for graph traversal (hot path)
/// Size: ~26 bytes (vs ~200-500 bytes with inline summary)
#[derive(Archive, RkyvDeserialize, RkyvSerialize)]
#[derive(Serialize, Deserialize)]
#[archive(check_bytes)]
pub(crate) struct NodeCfValue(
    pub(crate) Option<TemporalRange>,
    pub(crate) NameHash,            // Node name hash; full name in Names CF
    pub(crate) Option<SummaryHash>, // Content hash; full summary in NodeSummaries CF
);

impl ValidRangePatchable for Nodes {
    fn patch_valid_range(
        &self,
        old_value: &[u8],
        new_range: TemporalRange,
    ) -> Result<Vec<u8>, anyhow::Error> {
        use crate::graph::HotColumnFamilyRecord;

        let mut value = Nodes::value_from_bytes(old_value)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;
        value.0 = Some(new_range);
        Nodes::value_to_bytes(&value)
            .map(|aligned_vec| aligned_vec.to_vec())
            .map_err(|e| anyhow::anyhow!("Failed to serialize value: {}", e))
    }
}

/// Id of edge source node
pub type SrcId = Id;
/// Id of edge destination node
pub type DstId = Id;

impl ValidRangePatchable for ForwardEdges {
    fn patch_valid_range(
        &self,
        old_value: &[u8],
        new_range: TemporalRange,
    ) -> Result<Vec<u8>, anyhow::Error> {
        use crate::graph::HotColumnFamilyRecord;

        let mut value = ForwardEdges::value_from_bytes(old_value)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;
        value.0 = Some(new_range);
        ForwardEdges::value_to_bytes(&value)
            .map(|aligned_vec| aligned_vec.to_vec())
            .map_err(|e| anyhow::anyhow!("Failed to serialize value: {}", e))
    }
}

impl ValidRangePatchable for ReverseEdges {
    fn patch_valid_range(
        &self,
        old_value: &[u8],
        new_range: TemporalRange,
    ) -> Result<Vec<u8>, anyhow::Error> {
        use crate::graph::HotColumnFamilyRecord;

        let mut value = ReverseEdges::value_from_bytes(old_value)
            .map_err(|e| anyhow::anyhow!("Failed to deserialize value: {}", e))?;
        value.0 = Some(new_range);
        ReverseEdges::value_to_bytes(&value)
            .map(|aligned_vec| aligned_vec.to_vec())
            .map_err(|e| anyhow::anyhow!("Failed to serialize value: {}", e))
    }
}

/// Forward edges column family (HOT - optimized for graph traversal).
///
/// After Phase 2 blob separation:
/// - Value contains `Option<SummaryHash>` instead of inline `EdgeSummary`
/// - Full summary content is stored in `EdgeSummaries` CF (COLD)
/// - Value size reduced from ~200-500 bytes to ~35 bytes
pub(crate) struct ForwardEdges;

#[derive(Serialize, Deserialize)]
pub(crate) struct ForwardEdgeCfKey(
    pub(crate) SrcId,
    pub(crate) DstId,
    pub(crate) NameHash, // Edge name hash; full name in Names CF
);

/// Forward edge value - optimized for graph traversal (hot path)
/// Size: ~35 bytes (vs ~200-500 bytes with inline summary)
#[derive(Archive, RkyvDeserialize, RkyvSerialize)]
#[derive(Serialize, Deserialize)]
#[archive(check_bytes)]
pub(crate) struct ForwardEdgeCfValue(
    pub(crate) Option<TemporalRange>, // Field 0: Temporal validity
    pub(crate) Option<f64>,           // Field 1: Optional weight
    pub(crate) Option<SummaryHash>,   // Field 2: Content hash; full summary in EdgeSummaries CF
);

/// Reverse edges column family (index only).
pub(crate) struct ReverseEdges;
#[derive(Serialize, Deserialize)]
pub(crate) struct ReverseEdgeCfKey(
    pub(crate) DstId,
    pub(crate) SrcId,
    pub(crate) NameHash, // Edge name stored as hash; full name in Names CF
);
/// Reverse edge value - minimal for reverse lookups
/// Size: ~17 bytes
#[derive(Archive, RkyvDeserialize, RkyvSerialize)]
#[derive(Serialize, Deserialize)]
#[archive(check_bytes)]
pub(crate) struct ReverseEdgeCfValue(pub(crate) Option<TemporalRange>);

/// Edge fragments column family.
pub(crate) struct EdgeFragments;
#[derive(Serialize, Deserialize)]
pub(crate) struct EdgeFragmentCfKey(
    pub(crate) SrcId,
    pub(crate) DstId,
    pub(crate) NameHash, // Edge name stored as hash; full name in Names CF
    pub(crate) TimestampMilli,
);
#[derive(Serialize, Deserialize)]
pub(crate) struct EdgeFragmentCfValue(pub(crate) Option<TemporalRange>, pub(crate) FragmentContent);

pub type NodeName = String;
pub type EdgeName = String;
pub type NodeSummary = DataUrl;
pub type EdgeSummary = DataUrl;
pub type FragmentContent = DataUrl;

impl Nodes {
    pub const CF_NAME: &'static str = "nodes";

    /// Create key-value pair from AddNode mutation.
    pub fn record_from(args: &AddNode) -> (NodeCfKey, NodeCfValue) {
        let key = NodeCfKey(args.id);
        let name_hash = NameHash::from_name(&args.name);

        // Compute summary hash if summary is non-empty
        let summary_hash = if !args.summary.is_empty() {
            SummaryHash::from_summary(&args.summary).ok()
        } else {
            None
        };

        let value = NodeCfValue(args.valid_range.clone(), name_hash, summary_hash);
        (key, value)
    }

    /// Create and serialize to bytes (key bytes, value bytes).
    pub fn create_bytes(args: &AddNode) -> anyhow::Result<(Vec<u8>, Vec<u8>)> {
        let (key, value) = Self::record_from(args);
        let key_bytes = <Self as HotColumnFamilyRecord>::key_to_bytes(&key);
        let value_bytes = <Self as HotColumnFamilyRecord>::value_to_bytes(&value)?.to_vec();
        Ok((key_bytes, value_bytes))
    }

    /// Configure RocksDB options with shared block cache.
    pub fn column_family_options_with_cache(
        cache: &rocksdb::Cache,
        config: &BlockCacheConfig,
    ) -> rocksdb::Options {
        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();

        // Shared block cache with graph block size
        block_opts.set_block_cache(cache);
        block_opts.set_block_size(config.graph_block_size);

        if config.cache_index_and_filter_blocks {
            block_opts.set_cache_index_and_filter_blocks(true);
        }
        if config.pin_l0_filter_and_index {
            block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
        }

        opts.set_block_based_table_factory(&block_opts);
        opts
    }
}

impl HotColumnFamilyRecord for Nodes {
    const CF_NAME: &'static str = "nodes";
    type Key = NodeCfKey;
    type Value = NodeCfValue;

    fn key_to_bytes(key: &Self::Key) -> Vec<u8> {
        key.0.into_bytes().to_vec()
    }

    fn key_from_bytes(bytes: &[u8]) -> anyhow::Result<Self::Key> {
        if bytes.len() != 16 {
            anyhow::bail!("Invalid NodeCfKey length: expected 16, got {}", bytes.len());
        }
        let mut id_bytes = [0u8; 16];
        id_bytes.copy_from_slice(bytes);
        Ok(NodeCfKey(Id::from_bytes(id_bytes)))
    }
}

/// Node fragments column family (renamed from Fragments for clarity).
pub(crate) struct NodeFragments;
#[derive(Serialize, Deserialize)]
pub(crate) struct NodeFragmentCfKey(pub(crate) Id, pub(crate) TimestampMilli);
#[derive(Serialize, Deserialize)]
pub(crate) struct NodeFragmentCfValue(pub(crate) Option<TemporalRange>, pub(crate) FragmentContent);

impl ColumnFamilyRecord for NodeFragments {
    const CF_NAME: &'static str = "node_fragments";
    type Key = NodeFragmentCfKey;
    type Value = NodeFragmentCfValue;
    type CreateOp = AddNodeFragment;

    fn record_from(args: &AddNodeFragment) -> (NodeFragmentCfKey, NodeFragmentCfValue) {
        let key = NodeFragmentCfKey(args.id, args.ts_millis);
        let value = NodeFragmentCfValue(args.valid_range.clone(), args.content.clone());
        (key, value)
    }

    fn key_to_bytes(key: &Self::Key) -> Vec<u8> {
        // NodeFragmentCfKey(Id, TimestampMilli)
        // Layout: [Id bytes (16)] + [timestamp big-endian (8)]
        let mut bytes = Vec::with_capacity(24);
        bytes.extend_from_slice(&key.0.into_bytes());
        bytes.extend_from_slice(&key.1 .0.to_be_bytes());
        bytes
    }

    fn key_from_bytes(bytes: &[u8]) -> Result<Self::Key, anyhow::Error> {
        if bytes.len() != 24 {
            anyhow::bail!(
                "Invalid NodeFragmentCfKey length: expected 24, got {}",
                bytes.len()
            );
        }

        let mut id_bytes = [0u8; 16];
        id_bytes.copy_from_slice(&bytes[0..16]);

        let mut ts_bytes = [0u8; 8];
        ts_bytes.copy_from_slice(&bytes[16..24]);
        let timestamp = u64::from_be_bytes(ts_bytes);

        Ok(NodeFragmentCfKey(
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

impl NodeFragments {
    /// Configure RocksDB options with shared block cache.
    ///
    /// Fragment CFs use larger block size for variable-length content.
    pub fn column_family_options_with_cache(
        cache: &rocksdb::Cache,
        config: &BlockCacheConfig,
    ) -> rocksdb::Options {
        use rocksdb::SliceTransform;

        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();

        // Shared block cache with fragment block size (larger for variable content)
        block_opts.set_block_cache(cache);
        block_opts.set_block_size(config.fragment_block_size);

        if config.cache_index_and_filter_blocks {
            block_opts.set_cache_index_and_filter_blocks(true);
        }
        if config.pin_l0_filter_and_index {
            block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
        }

        opts.set_block_based_table_factory(&block_opts);

        // Key layout: [Id (16 bytes)] + [TimestampMilli (8 bytes)]
        // Use 16-byte prefix to scan all fragments for a given Id
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(16));
        opts.set_memtable_prefix_bloom_ratio(0.2);

        opts
    }
}

impl ColumnFamilyRecord for EdgeFragments {
    const CF_NAME: &'static str = "edge_fragments";
    type Key = EdgeFragmentCfKey;
    type Value = EdgeFragmentCfValue;
    type CreateOp = AddEdgeFragment;

    fn record_from(args: &AddEdgeFragment) -> (EdgeFragmentCfKey, EdgeFragmentCfValue) {
        let name_hash = NameHash::from_name(&args.edge_name);
        let key = EdgeFragmentCfKey(
            args.src_id,
            args.dst_id,
            name_hash,
            args.ts_millis,
        );
        let value = EdgeFragmentCfValue(args.valid_range.clone(), args.content.clone());
        (key, value)
    }

    fn key_to_bytes(key: &Self::Key) -> Vec<u8> {
        // EdgeFragmentCfKey(SrcId, DstId, NameHash, TimestampMilli)
        // Layout: [src_id (16)] + [dst_id (16)] + [name_hash (8)] + [timestamp (8)] = 48 bytes FIXED
        let mut bytes = Vec::with_capacity(48);
        bytes.extend_from_slice(&key.0.into_bytes());
        bytes.extend_from_slice(&key.1.into_bytes());
        bytes.extend_from_slice(key.2.as_bytes());
        bytes.extend_from_slice(&key.3 .0.to_be_bytes());
        bytes
    }

    fn key_from_bytes(bytes: &[u8]) -> Result<Self::Key, anyhow::Error> {
        if bytes.len() != 48 {
            anyhow::bail!(
                "Invalid EdgeFragmentCfKey length: expected 48, got {}",
                bytes.len()
            );
        }

        let mut src_id_bytes = [0u8; 16];
        src_id_bytes.copy_from_slice(&bytes[0..16]);

        let mut dst_id_bytes = [0u8; 16];
        dst_id_bytes.copy_from_slice(&bytes[16..32]);

        let mut name_hash_bytes = [0u8; 8];
        name_hash_bytes.copy_from_slice(&bytes[32..40]);

        let mut ts_bytes = [0u8; 8];
        ts_bytes.copy_from_slice(&bytes[40..48]);
        let timestamp = u64::from_be_bytes(ts_bytes);

        Ok(EdgeFragmentCfKey(
            Id::from_bytes(src_id_bytes),
            Id::from_bytes(dst_id_bytes),
            NameHash::from_bytes(name_hash_bytes),
            TimestampMilli(timestamp),
        ))
    }

    fn column_family_options() -> rocksdb::Options {
        use rocksdb::SliceTransform;

        let mut opts = rocksdb::Options::default();

        // Key layout: [src_id (16)] + [dst_id (16)] + [name_hash (8)] + [timestamp (8)] = 48 bytes
        // Use 32-byte prefix (src_id + dst_id) to scan all fragments for a given edge topology
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(32));

        // Enable prefix bloom filter for fast prefix existence checks
        opts.set_memtable_prefix_bloom_ratio(0.2);

        opts
    }
}

impl EdgeFragments {
    /// Configure RocksDB options with shared block cache.
    ///
    /// Fragment CFs use larger block size for variable-length content.
    pub fn column_family_options_with_cache(
        cache: &rocksdb::Cache,
        config: &BlockCacheConfig,
    ) -> rocksdb::Options {
        use rocksdb::SliceTransform;

        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();

        // Shared block cache with fragment block size (larger for variable content)
        block_opts.set_block_cache(cache);
        block_opts.set_block_size(config.fragment_block_size);

        if config.cache_index_and_filter_blocks {
            block_opts.set_cache_index_and_filter_blocks(true);
        }
        if config.pin_l0_filter_and_index {
            block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
        }

        opts.set_block_based_table_factory(&block_opts);

        // Key layout: [src_id (16)] + [dst_id (16)] + [name_hash (8)] + [timestamp (8)] = 48 bytes
        // Use 32-byte prefix (src_id + dst_id) to scan all fragments for a given edge topology
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(32));
        opts.set_memtable_prefix_bloom_ratio(0.2);

        opts
    }
}

impl ForwardEdges {
    pub const CF_NAME: &'static str = "forward_edges";

    /// Create key-value pair from AddEdge mutation.
    pub fn record_from(args: &AddEdge) -> (ForwardEdgeCfKey, ForwardEdgeCfValue) {
        let name_hash = NameHash::from_name(&args.name);
        let key = ForwardEdgeCfKey(args.source_node_id, args.target_node_id, name_hash);

        // Compute summary hash if summary is non-empty
        let summary_hash = if !args.summary.is_empty() {
            SummaryHash::from_summary(&args.summary).ok()
        } else {
            None
        };

        let value = ForwardEdgeCfValue(args.valid_range.clone(), args.weight, summary_hash);
        (key, value)
    }

    /// Create and serialize to bytes (key bytes, value bytes).
    pub fn create_bytes(args: &AddEdge) -> anyhow::Result<(Vec<u8>, Vec<u8>)> {
        let (key, value) = Self::record_from(args);
        let key_bytes = <Self as HotColumnFamilyRecord>::key_to_bytes(&key);
        let value_bytes = <Self as HotColumnFamilyRecord>::value_to_bytes(&value)?.to_vec();
        Ok((key_bytes, value_bytes))
    }

    /// Configure RocksDB options with shared block cache.
    pub fn column_family_options_with_cache(
        cache: &rocksdb::Cache,
        config: &BlockCacheConfig,
    ) -> rocksdb::Options {
        use rocksdb::SliceTransform;

        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();

        // Shared block cache with graph block size
        block_opts.set_block_cache(cache);
        block_opts.set_block_size(config.graph_block_size);

        if config.cache_index_and_filter_blocks {
            block_opts.set_cache_index_and_filter_blocks(true);
        }
        if config.pin_l0_filter_and_index {
            block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
        }

        opts.set_block_based_table_factory(&block_opts);

        // Key layout: [src_id (16)] + [dst_id (16)] + [name_hash (8)] = 40 bytes
        // Use 16-byte prefix to scan all edges from a source node
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(16));
        opts.set_memtable_prefix_bloom_ratio(0.2);

        opts
    }
}

impl HotColumnFamilyRecord for ForwardEdges {
    const CF_NAME: &'static str = "forward_edges";
    type Key = ForwardEdgeCfKey;
    type Value = ForwardEdgeCfValue;

    fn key_to_bytes(key: &Self::Key) -> Vec<u8> {
        // ForwardEdgeCfKey(SrcId, DstId, NameHash)
        // Layout: [src_id (16)] + [dst_id (16)] + [name_hash (8)] = 40 bytes
        let mut bytes = Vec::with_capacity(40);
        bytes.extend_from_slice(&key.0.into_bytes());
        bytes.extend_from_slice(&key.1.into_bytes());
        bytes.extend_from_slice(key.2.as_bytes());
        bytes
    }

    fn key_from_bytes(bytes: &[u8]) -> anyhow::Result<Self::Key> {
        if bytes.len() != 40 {
            anyhow::bail!(
                "Invalid ForwardEdgeCfKey length: expected 40, got {}",
                bytes.len()
            );
        }

        let mut src_id_bytes = [0u8; 16];
        src_id_bytes.copy_from_slice(&bytes[0..16]);

        let mut dst_id_bytes = [0u8; 16];
        dst_id_bytes.copy_from_slice(&bytes[16..32]);

        let mut name_hash_bytes = [0u8; 8];
        name_hash_bytes.copy_from_slice(&bytes[32..40]);

        Ok(ForwardEdgeCfKey(
            Id::from_bytes(src_id_bytes),
            Id::from_bytes(dst_id_bytes),
            NameHash::from_bytes(name_hash_bytes),
        ))
    }
}

impl ReverseEdges {
    pub const CF_NAME: &'static str = "reverse_edges";

    /// Create key-value pair from AddEdge mutation.
    pub fn record_from(args: &AddEdge) -> (ReverseEdgeCfKey, ReverseEdgeCfValue) {
        let name_hash = NameHash::from_name(&args.name);
        let key = ReverseEdgeCfKey(args.target_node_id, args.source_node_id, name_hash);
        let value = ReverseEdgeCfValue(args.valid_range.clone());
        (key, value)
    }

    /// Create and serialize to bytes (key bytes, value bytes).
    pub fn create_bytes(args: &AddEdge) -> anyhow::Result<(Vec<u8>, Vec<u8>)> {
        let (key, value) = Self::record_from(args);
        let key_bytes = <Self as HotColumnFamilyRecord>::key_to_bytes(&key);
        let value_bytes = <Self as HotColumnFamilyRecord>::value_to_bytes(&value)?.to_vec();
        Ok((key_bytes, value_bytes))
    }

    /// Configure RocksDB options with shared block cache.
    pub fn column_family_options_with_cache(
        cache: &rocksdb::Cache,
        config: &BlockCacheConfig,
    ) -> rocksdb::Options {
        use rocksdb::SliceTransform;

        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();

        // Shared block cache with graph block size
        block_opts.set_block_cache(cache);
        block_opts.set_block_size(config.graph_block_size);

        if config.cache_index_and_filter_blocks {
            block_opts.set_cache_index_and_filter_blocks(true);
        }
        if config.pin_l0_filter_and_index {
            block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
        }

        opts.set_block_based_table_factory(&block_opts);

        // Key layout: [dst_id (16)] + [src_id (16)] + [name_hash (8)] = 40 bytes
        // Use 16-byte prefix to scan all edges to a destination node
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(16));
        opts.set_memtable_prefix_bloom_ratio(0.2);

        opts
    }
}

impl HotColumnFamilyRecord for ReverseEdges {
    const CF_NAME: &'static str = "reverse_edges";
    type Key = ReverseEdgeCfKey;
    type Value = ReverseEdgeCfValue;

    fn key_to_bytes(key: &Self::Key) -> Vec<u8> {
        // ReverseEdgeCfKey(DstId, SrcId, NameHash)
        // Layout: [dst_id (16)] + [src_id (16)] + [name_hash (8)] = 40 bytes
        let mut bytes = Vec::with_capacity(40);
        bytes.extend_from_slice(&key.0.into_bytes());
        bytes.extend_from_slice(&key.1.into_bytes());
        bytes.extend_from_slice(key.2.as_bytes());
        bytes
    }

    fn key_from_bytes(bytes: &[u8]) -> anyhow::Result<Self::Key> {
        if bytes.len() != 40 {
            anyhow::bail!(
                "Invalid ReverseEdgeCfKey length: expected 40, got {}",
                bytes.len()
            );
        }

        let mut dst_id_bytes = [0u8; 16];
        dst_id_bytes.copy_from_slice(&bytes[0..16]);

        let mut src_id_bytes = [0u8; 16];
        src_id_bytes.copy_from_slice(&bytes[16..32]);

        let mut name_hash_bytes = [0u8; 8];
        name_hash_bytes.copy_from_slice(&bytes[32..40]);

        Ok(ReverseEdgeCfKey(
            Id::from_bytes(dst_id_bytes),
            Id::from_bytes(src_id_bytes),
            NameHash::from_bytes(name_hash_bytes),
        ))
    }
}

// ============================================================================
// Cold Column Families for Blob Separation (Phase 2)
// ============================================================================

/// NodeSummaries column family (COLD - infrequently accessed).
///
/// Stores node summary content keyed by content hash (SummaryHash).
/// Content-addressable: identical summaries stored once, referenced by hash.
///
/// Key: SummaryHash (8 bytes, content-addressable)
/// Value: NodeSummary (DataUrl, rmp_serde + LZ4 compressed)
pub(crate) struct NodeSummaries;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct NodeSummaryCfKey(pub(crate) SummaryHash);

#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct NodeSummaryCfValue(pub(crate) NodeSummary);

impl NodeSummaries {
    pub const CF_NAME: &'static str = "node_summaries";

    /// Serialize the key to bytes (8 bytes, fixed size).
    pub fn key_to_bytes(key: &NodeSummaryCfKey) -> Vec<u8> {
        key.0.as_bytes().to_vec()
    }

    /// Deserialize the key from bytes.
    pub fn key_from_bytes(bytes: &[u8]) -> Result<NodeSummaryCfKey, anyhow::Error> {
        if bytes.len() != SummaryHash::SIZE {
            anyhow::bail!(
                "Invalid NodeSummaryCfKey length: expected {}, got {}",
                SummaryHash::SIZE,
                bytes.len()
            );
        }
        let mut hash_bytes = [0u8; 8];
        hash_bytes.copy_from_slice(bytes);
        Ok(NodeSummaryCfKey(SummaryHash::from_bytes(hash_bytes)))
    }

    /// Serialize the value to bytes (LZ4 compressed MessagePack).
    pub fn value_to_bytes(value: &NodeSummaryCfValue) -> Result<Vec<u8>, anyhow::Error> {
        let msgpack_bytes = rmp_serde::to_vec(value)?;
        let compressed = lz4::block::compress(&msgpack_bytes, None, true)?;
        Ok(compressed)
    }

    /// Deserialize the value from bytes.
    pub fn value_from_bytes(bytes: &[u8]) -> Result<NodeSummaryCfValue, anyhow::Error> {
        let decompressed = lz4::block::decompress(bytes, None)?;
        let value: NodeSummaryCfValue = rmp_serde::from_slice(&decompressed)?;
        Ok(value)
    }

    /// Configure RocksDB options for this column family.
    ///
    /// NodeSummaries is a COLD CF - optimize for space, not speed.
    pub fn column_family_options() -> rocksdb::Options {
        let mut opts = rocksdb::Options::default();

        // Enable bloom filter for point lookups by hash
        let mut block_opts = rocksdb::BlockBasedOptions::default();
        block_opts.set_bloom_filter(10.0, false);
        opts.set_block_based_table_factory(&block_opts);

        opts
    }

    /// Configure RocksDB options with shared block cache.
    pub fn column_family_options_with_cache(
        cache: &rocksdb::Cache,
        config: &BlockCacheConfig,
    ) -> rocksdb::Options {
        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();

        // Shared block cache with larger block size for cold data
        block_opts.set_block_cache(cache);
        block_opts.set_block_size(config.fragment_block_size); // Use fragment size for cold CFs

        if config.cache_index_and_filter_blocks {
            block_opts.set_cache_index_and_filter_blocks(true);
        }
        if config.pin_l0_filter_and_index {
            block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
        }

        // Enable bloom filter for point lookups
        block_opts.set_bloom_filter(10.0, false);

        opts.set_block_based_table_factory(&block_opts);
        opts
    }
}

/// EdgeSummaries column family (COLD - infrequently accessed).
///
/// Stores edge summary content keyed by content hash (SummaryHash).
/// Content-addressable: identical summaries stored once, referenced by hash.
///
/// Key: SummaryHash (8 bytes, content-addressable)
/// Value: EdgeSummary (DataUrl, rmp_serde + LZ4 compressed)
pub(crate) struct EdgeSummaries;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct EdgeSummaryCfKey(pub(crate) SummaryHash);

#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct EdgeSummaryCfValue(pub(crate) EdgeSummary);

impl EdgeSummaries {
    pub const CF_NAME: &'static str = "edge_summaries";

    /// Serialize the key to bytes (8 bytes, fixed size).
    pub fn key_to_bytes(key: &EdgeSummaryCfKey) -> Vec<u8> {
        key.0.as_bytes().to_vec()
    }

    /// Deserialize the key from bytes.
    pub fn key_from_bytes(bytes: &[u8]) -> Result<EdgeSummaryCfKey, anyhow::Error> {
        if bytes.len() != SummaryHash::SIZE {
            anyhow::bail!(
                "Invalid EdgeSummaryCfKey length: expected {}, got {}",
                SummaryHash::SIZE,
                bytes.len()
            );
        }
        let mut hash_bytes = [0u8; 8];
        hash_bytes.copy_from_slice(bytes);
        Ok(EdgeSummaryCfKey(SummaryHash::from_bytes(hash_bytes)))
    }

    /// Serialize the value to bytes (LZ4 compressed MessagePack).
    pub fn value_to_bytes(value: &EdgeSummaryCfValue) -> Result<Vec<u8>, anyhow::Error> {
        let msgpack_bytes = rmp_serde::to_vec(value)?;
        let compressed = lz4::block::compress(&msgpack_bytes, None, true)?;
        Ok(compressed)
    }

    /// Deserialize the value from bytes.
    pub fn value_from_bytes(bytes: &[u8]) -> Result<EdgeSummaryCfValue, anyhow::Error> {
        let decompressed = lz4::block::decompress(bytes, None)?;
        let value: EdgeSummaryCfValue = rmp_serde::from_slice(&decompressed)?;
        Ok(value)
    }

    /// Configure RocksDB options for this column family.
    ///
    /// EdgeSummaries is a COLD CF - optimize for space, not speed.
    pub fn column_family_options() -> rocksdb::Options {
        let mut opts = rocksdb::Options::default();

        // Enable bloom filter for point lookups by hash
        let mut block_opts = rocksdb::BlockBasedOptions::default();
        block_opts.set_bloom_filter(10.0, false);
        opts.set_block_based_table_factory(&block_opts);

        opts
    }

    /// Configure RocksDB options with shared block cache.
    pub fn column_family_options_with_cache(
        cache: &rocksdb::Cache,
        config: &BlockCacheConfig,
    ) -> rocksdb::Options {
        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();

        // Shared block cache with larger block size for cold data
        block_opts.set_block_cache(cache);
        block_opts.set_block_size(config.fragment_block_size); // Use fragment size for cold CFs

        if config.cache_index_and_filter_blocks {
            block_opts.set_cache_index_and_filter_blocks(true);
        }
        if config.pin_l0_filter_and_index {
            block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
        }

        // Enable bloom filter for point lookups
        block_opts.set_bloom_filter(10.0, false);

        opts.set_block_based_table_factory(&block_opts);
        opts
    }
}

/// All column families used in the database.
/// This is the authoritative list that should be used when opening the database.
pub(crate) const ALL_COLUMN_FAMILIES: &[&str] = &[
    Names::CF_NAME,
    Nodes::CF_NAME,
    NodeFragments::CF_NAME,
    NodeSummaries::CF_NAME,  // Phase 2: Cold CF for node summaries
    EdgeFragments::CF_NAME,
    EdgeSummaries::CF_NAME,  // Phase 2: Cold CF for edge summaries
    ForwardEdges::CF_NAME,
    ReverseEdges::CF_NAME,
];

// ============================================================================
// Schema - ColumnFamilyProvider Implementation
// ============================================================================

use crate::provider::ColumnFamilyProvider;
use rocksdb::{Cache, ColumnFamilyDescriptor, TransactionDB};
use std::sync::Arc;

/// Graph module schema implementing the ColumnFamilyProvider trait.
///
/// This enables the graph module to register its column families with
/// a shared RocksDB instance. The graph module manages:
/// - Hot CFs: Names, Nodes, ForwardEdges, ReverseEdges (optimized for graph traversal)
/// - Cold CFs: NodeSummaries, EdgeSummaries (content-addressable blob storage)
/// - Fragment CFs: NodeFragments, EdgeFragments (temporal content)
///
/// # Example
///
/// ```ignore
/// use motlie_db::graph::Schema;
///
/// let name_cache = Arc::new(NameCache::new());
/// let schema = Schema::with_name_cache(name_cache);
///
/// // Get CF descriptors for database opening
/// let descriptors = schema.column_family_descriptors(Some(&cache), 4096);
/// ```
pub struct Schema {
    /// Name cache for pre-warming
    name_cache: Arc<super::name_hash::NameCache>,
    /// Block size for fragment/cold CFs (larger for variable content)
    fragment_block_size: usize,
    /// Maximum names to pre-warm
    prewarm_limit: usize,
}

impl Schema {
    /// Default block size for graph/hot CFs
    const DEFAULT_GRAPH_BLOCK_SIZE: usize = 4 * 1024; // 4KB

    /// Default block size for fragment/cold CFs
    const DEFAULT_FRAGMENT_BLOCK_SIZE: usize = 16 * 1024; // 16KB

    /// Default prewarm limit
    const DEFAULT_PREWARM_LIMIT: usize = 1000;

    /// Create a new Schema with a new NameCache.
    pub fn new() -> Self {
        Self {
            name_cache: Arc::new(super::name_hash::NameCache::new()),
            fragment_block_size: Self::DEFAULT_FRAGMENT_BLOCK_SIZE,
            prewarm_limit: Self::DEFAULT_PREWARM_LIMIT,
        }
    }

    /// Create with a shared NameCache.
    pub fn with_name_cache(name_cache: Arc<super::name_hash::NameCache>) -> Self {
        Self {
            name_cache,
            fragment_block_size: Self::DEFAULT_FRAGMENT_BLOCK_SIZE,
            prewarm_limit: Self::DEFAULT_PREWARM_LIMIT,
        }
    }

    /// Set the fragment block size.
    pub fn with_fragment_block_size(mut self, size: usize) -> Self {
        self.fragment_block_size = size;
        self
    }

    /// Set the prewarm limit.
    pub fn with_prewarm_limit(mut self, limit: usize) -> Self {
        self.prewarm_limit = limit;
        self
    }

    /// Get the name cache.
    pub fn name_cache(&self) -> &Arc<super::name_hash::NameCache> {
        &self.name_cache
    }

    /// Configure CF options with optional shared cache.
    fn cf_options_with_cache(
        base_opts: rocksdb::Options,
        cache: Option<&Cache>,
        block_size: usize,
    ) -> rocksdb::Options {
        let mut opts = base_opts;
        if let Some(cache) = cache {
            let mut block_opts = rocksdb::BlockBasedOptions::default();
            block_opts.set_block_size(block_size);
            block_opts.set_block_cache(cache);
            block_opts.set_cache_index_and_filter_blocks(true);
            block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
            opts.set_block_based_table_factory(&block_opts);
        }
        opts
    }
}

impl Default for Schema {
    fn default() -> Self {
        Self::new()
    }
}

impl ColumnFamilyProvider for Schema {
    fn name(&self) -> &'static str {
        "graph"
    }

    fn column_family_descriptors(
        &self,
        cache: Option<&Cache>,
        block_size: usize,
    ) -> Vec<ColumnFamilyDescriptor> {
        // Use provided block_size for hot CFs, fragment_block_size for cold/fragment CFs
        let graph_block_size = if block_size > 0 {
            block_size
        } else {
            Self::DEFAULT_GRAPH_BLOCK_SIZE
        };

        vec![
            // Hot CFs - optimized for graph traversal
            ColumnFamilyDescriptor::new(
                Names::CF_NAME,
                Self::cf_options_with_cache(Names::column_family_options(), cache, graph_block_size),
            ),
            ColumnFamilyDescriptor::new(
                Nodes::CF_NAME,
                Self::cf_options_with_cache(rocksdb::Options::default(), cache, graph_block_size),
            ),
            ColumnFamilyDescriptor::new(
                ForwardEdges::CF_NAME,
                Self::cf_options_with_cache(rocksdb::Options::default(), cache, graph_block_size),
            ),
            ColumnFamilyDescriptor::new(
                ReverseEdges::CF_NAME,
                Self::cf_options_with_cache(rocksdb::Options::default(), cache, graph_block_size),
            ),
            // Fragment CFs - larger blocks for variable content
            ColumnFamilyDescriptor::new(
                NodeFragments::CF_NAME,
                Self::cf_options_with_cache(
                    NodeFragments::column_family_options(),
                    cache,
                    self.fragment_block_size,
                ),
            ),
            ColumnFamilyDescriptor::new(
                EdgeFragments::CF_NAME,
                Self::cf_options_with_cache(
                    EdgeFragments::column_family_options(),
                    cache,
                    self.fragment_block_size,
                ),
            ),
            // Cold CFs - content-addressable blob storage
            ColumnFamilyDescriptor::new(
                NodeSummaries::CF_NAME,
                Self::cf_options_with_cache(
                    NodeSummaries::column_family_options(),
                    cache,
                    self.fragment_block_size,
                ),
            ),
            ColumnFamilyDescriptor::new(
                EdgeSummaries::CF_NAME,
                Self::cf_options_with_cache(
                    EdgeSummaries::column_family_options(),
                    cache,
                    self.fragment_block_size,
                ),
            ),
        ]
    }

    fn on_ready(&self, db: &TransactionDB) -> anyhow::Result<()> {
        use rocksdb::IteratorMode;

        if self.prewarm_limit == 0 {
            return Ok(());
        }

        let names_cf = db
            .cf_handle(Names::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("Names CF not found"))?;

        let mut loaded = 0usize;
        for item in db.iterator_cf(&names_cf, IteratorMode::Start) {
            if loaded >= self.prewarm_limit {
                break;
            }

            let (key_bytes, value_bytes) = item?;

            // Parse key (NameHash) and value (name string)
            if key_bytes.len() == 8 {
                let mut hash_bytes = [0u8; 8];
                hash_bytes.copy_from_slice(&key_bytes);
                let hash = super::name_hash::NameHash::from_bytes(hash_bytes);

                if let Ok(value) = Names::value_from_bytes(&value_bytes) {
                    self.name_cache.insert(hash, value.0);
                    loaded += 1;
                }
            }
        }

        tracing::info!(
            count = loaded,
            limit = self.prewarm_limit,
            "[graph::Schema] Pre-warmed NameCache"
        );
        Ok(())
    }

    fn cf_names(&self) -> Vec<&'static str> {
        ALL_COLUMN_FAMILIES.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::mutation::AddEdge;
    use crate::Id;

    #[test]
    fn test_forward_edges_keys_lexicographically_sortable() {
        // Create multiple edge arguments with different source/destination combinations
        // Using deterministic timestamps for stable test behavior
        let base_ts = 1700000000000u64; // Fixed base timestamp
        let edges = vec![
            AddEdge {
                source_node_id: Id::from_bytes([0u8; 16]),
                target_node_id: Id::from_bytes([0u8; 16]),
                ts_millis: TimestampMilli(base_ts),
                name: "edge_a".to_string(),
                summary: EdgeSummary::from_text(""),
                weight: Some(1.0),
                valid_range: None,
            },
            AddEdge {
                source_node_id: Id::from_bytes([0u8; 16]),
                target_node_id: Id::from_bytes([1u8; 16]),
                ts_millis: TimestampMilli(base_ts + 1000),
                name: "edge_b".to_string(),
                summary: EdgeSummary::from_text(""),
                weight: Some(1.0),
                valid_range: None,
            },
            AddEdge {
                source_node_id: Id::from_bytes([1u8; 16]),
                target_node_id: Id::from_bytes([0u8; 16]),
                ts_millis: TimestampMilli(base_ts + 2000),
                name: "edge_c".to_string(),
                summary: EdgeSummary::from_text(""),
                weight: Some(1.0),
                valid_range: None,
            },
            AddEdge {
                source_node_id: Id::from_bytes([1u8; 16]),
                target_node_id: Id::from_bytes([1u8; 16]),
                ts_millis: TimestampMilli(base_ts + 3000),
                name: "edge_d".to_string(),
                summary: EdgeSummary::from_text(""),
                weight: Some(1.0),
                valid_range: None,
            },
            // Add edge with same source and target but different name
            AddEdge {
                source_node_id: Id::from_bytes([0u8; 16]),
                target_node_id: Id::from_bytes([0u8; 16]),
                ts_millis: TimestampMilli(base_ts + 4000),
                name: "edge_z".to_string(),
                summary: EdgeSummary::from_text(""),
                weight: Some(1.0),
                valid_range: None,
            },
        ];

        // Generate key-value pairs and serialize the keys
        // Note: With NameHash, names are hashed to 8-byte values, so sort order
        // is by hash value, not alphabetical string order.
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

        // Verify that sorting by bytes produces a consistent ordering:
        // - Keys should be ordered by (source_id, destination_id, name_hash)
        // - With NameHash, edge_a and edge_z may sort in any order based on their hash values

        // With NameHash, the first 32 bytes are still src_id + dst_id.
        // All edges from source [0..] should come before edges from source [1..]
        // Edges with same source sorted by dest, then by name_hash.

        // All [0..] source edges (indices 0,1,2 in original) should be first
        // Check that source IDs are properly grouped:
        let src0_edges: Vec<_> = sorted_keys
            .iter()
            .filter(|(k, _)| k[..16] == [0u8; 16])
            .collect();
        let src1_edges: Vec<_> = sorted_keys
            .iter()
            .filter(|(k, _)| k[..16] == [1u8; 16])
            .collect();

        // We should have 3 edges from source [0..] and 2 from source [1..]
        assert_eq!(src0_edges.len(), 3, "Should have 3 edges from source [0..]");
        assert_eq!(src1_edges.len(), 2, "Should have 2 edges from source [1..]");

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
    fn test_temporal_range_always_valid() {
        let range = TemporalRange::always_valid();
        assert!(range.is_none(), "always_valid should return None");
    }

    #[test]
    fn test_temporal_range_valid_from() {
        let start = TimestampMilli(1000);
        let range = TemporalRange::valid_from(start);

        assert!(range.is_some());
        let range = range.unwrap();
        assert_eq!(range.0, Some(TimestampMilli(1000)));
        assert_eq!(range.1, None);
    }

    #[test]
    fn test_temporal_range_valid_until() {
        let until = TimestampMilli(2000);
        let range = TemporalRange::valid_until(until);

        assert!(range.is_some());
        let range = range.unwrap();
        assert_eq!(range.0, None);
        assert_eq!(range.1, Some(TimestampMilli(2000)));
    }

    #[test]
    fn test_temporal_range_valid_between() {
        let start = TimestampMilli(1000);
        let until = TimestampMilli(2000);
        let range = TemporalRange::valid_between(start, until);

        assert!(range.is_some());
        let range = range.unwrap();
        assert_eq!(range.0, Some(TimestampMilli(1000)));
        assert_eq!(range.1, Some(TimestampMilli(2000)));
    }

    #[test]
    fn test_is_valid_at_with_start_only() {
        let range = TemporalRange(Some(TimestampMilli(1000)), None);

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
        let range = TemporalRange(None, Some(TimestampMilli(2000)));

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
        let range = TemporalRange(Some(TimestampMilli(1000)), Some(TimestampMilli(2000)));

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
        let range = TemporalRange(None, None);

        // Always valid regardless of timestamp
        assert!(range.is_valid_at(TimestampMilli(0)));
        assert!(range.is_valid_at(TimestampMilli(1000)));
        assert!(range.is_valid_at(TimestampMilli(u64::MAX)));
    }

    #[test]
    fn test_is_valid_at_time_with_none() {
        let temporal_range: Option<TemporalRange> = None;

        // None means always valid
        assert!(is_valid_at_time(&temporal_range, TimestampMilli(0)));
        assert!(is_valid_at_time(&temporal_range, TimestampMilli(1000)));
        assert!(is_valid_at_time(&temporal_range, TimestampMilli(u64::MAX)));
    }

    #[test]
    fn test_is_valid_at_time_with_range() {
        let temporal_range = Some(TemporalRange(
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
        let from_only = Some(TemporalRange(Some(TimestampMilli(100)), None));
        assert!(!is_valid_at_time(&from_only, TimestampMilli(99)));
        assert!(is_valid_at_time(&from_only, TimestampMilli(100)));
        assert!(is_valid_at_time(&from_only, TimestampMilli(u64::MAX)));

        // Test with until only
        let until_only = Some(TemporalRange(None, Some(TimestampMilli(200))));
        assert!(is_valid_at_time(&until_only, TimestampMilli(0)));
        assert!(is_valid_at_time(&until_only, TimestampMilli(199)));
        assert!(!is_valid_at_time(&until_only, TimestampMilli(200)));

        // Test with no constraints (Some with both None)
        let no_constraints = Some(TemporalRange(None, None));
        assert!(is_valid_at_time(&no_constraints, TimestampMilli(0)));
        assert!(is_valid_at_time(&no_constraints, TimestampMilli(u64::MAX)));
    }

    #[test]
    fn test_temporal_range_serialization() {
        // Test that TemporalRange can be serialized and deserialized
        let range = TemporalRange(Some(TimestampMilli(1000)), Some(TimestampMilli(2000)));

        let serialized = rmp_serde::to_vec(&range).expect("Should serialize");
        let deserialized: TemporalRange =
            rmp_serde::from_slice(&serialized).expect("Should deserialize");

        assert_eq!(range, deserialized);
    }

    #[test]
    fn test_temporal_range_clone_and_equality() {
        let range1 = TemporalRange(Some(TimestampMilli(1000)), Some(TimestampMilli(2000)));
        let range2 = range1.clone();

        assert_eq!(range1, range2);

        let range3 = TemporalRange(Some(TimestampMilli(1000)), Some(TimestampMilli(2001)));
        assert_ne!(range1, range3);
    }

    // =========================================================================
    // Schema Tests
    // =========================================================================

    #[test]
    fn test_schema_new() {
        let schema = Schema::new();
        assert_eq!(schema.name(), "graph");
        assert_eq!(schema.prewarm_limit, Schema::DEFAULT_PREWARM_LIMIT);
    }

    #[test]
    fn test_schema_with_name_cache() {
        let name_cache = Arc::new(super::super::name_hash::NameCache::new());
        let schema = Schema::with_name_cache(name_cache.clone());

        // Should share the same cache
        assert!(Arc::ptr_eq(&name_cache, schema.name_cache()));
    }

    #[test]
    fn test_schema_builder_methods() {
        let schema = Schema::new()
            .with_fragment_block_size(32 * 1024)
            .with_prewarm_limit(500);

        assert_eq!(schema.fragment_block_size, 32 * 1024);
        assert_eq!(schema.prewarm_limit, 500);
    }

    #[test]
    fn test_schema_cf_names() {
        let schema = Schema::new();
        let names = schema.cf_names();
        assert_eq!(names.len(), ALL_COLUMN_FAMILIES.len());

        // Verify all expected CFs are present
        assert!(names.contains(&Names::CF_NAME));
        assert!(names.contains(&Nodes::CF_NAME));
        assert!(names.contains(&ForwardEdges::CF_NAME));
        assert!(names.contains(&ReverseEdges::CF_NAME));
        assert!(names.contains(&NodeFragments::CF_NAME));
        assert!(names.contains(&EdgeFragments::CF_NAME));
        assert!(names.contains(&NodeSummaries::CF_NAME));
        assert!(names.contains(&EdgeSummaries::CF_NAME));
    }

    #[test]
    fn test_schema_column_family_descriptors() {
        use crate::provider::ColumnFamilyProvider;

        let schema = Schema::new();
        let descriptors = schema.column_family_descriptors(None, 4096);
        assert_eq!(descriptors.len(), ALL_COLUMN_FAMILIES.len());
    }

    #[test]
    fn test_schema_column_family_descriptors_with_cache() {
        use crate::provider::ColumnFamilyProvider;

        let cache = rocksdb::Cache::new_lru_cache(16 * 1024 * 1024);
        let schema = Schema::new();
        let descriptors = schema.column_family_descriptors(Some(&cache), 4096);
        assert_eq!(descriptors.len(), ALL_COLUMN_FAMILIES.len());
    }
}
