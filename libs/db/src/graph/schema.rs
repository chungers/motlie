//! All graph CFs use the `graph/` prefix to avoid collisions with other subsystem CFs.
//!
//! ## Naming Convention
//!
//! For a domain entity named `Foo`:
//!
//! 1. **Column family marker**: `Foos` (plural) - a unit struct marking the CF
//! 2. **Key type**: `FooCfKey` (singular + CfKey) - always a tuple struct for compound keys
//! 3. **Value type**: `FooCfValue` (singular + CfValue) - wraps or aliases the value type
//! 4. **Domain struct**: `Foo` - if the value has >2 fields, define a separate struct
//!
//! ### Example
//!
//! ```ignore
//! // Domain struct for rich data (>2 fields)
//! pub(crate) struct EmbeddingSpec {
//!     pub(crate) model: String,
//!     pub(crate) dim: u32,
//!     pub(crate) distance: Distance,
//! }
//!
//! // CF marker (plural)
//! pub(crate) struct EmbeddingSpecs;
//!
//! // Key (singular + CfKey) - always tuple
//! pub(crate) struct EmbeddingSpecCfKey(pub(crate) u64);
//!
//! // Value (singular + CfValue) - wraps domain struct
//! pub(crate) struct EmbeddingSpecCfValue(pub(crate) EmbeddingSpec);
//! ```
//!
//! ### Key Rules
//!
//! - Keys are **always tuples** (compound keys for RocksDB prefix extraction)
//! - Keys use direct byte serialization (not MessagePack) for prefix extractors
//!
//! ### Value Rules
//!
//! - If value has ≤2 fields: tuple struct is acceptable
//! - If value has >2 fields: define a named struct, wrap in `FooCfValue`
//! - Values use MessagePack serialization (self-describing, compact) or rkyv zero-copy if the
//! column family implements the HotColumnFamilyRecord trait.
//!
//!
use super::mutation::{AddEdge, AddNode};
use super::name_hash::NameHash;
use super::subsystem::GraphBlockCacheConfig;
use super::summary_hash::SummaryHash;
use super::ColumnFamily;
use super::ColumnFamilyConfig;
use super::ColumnFamilySerde;
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
// Version Type for Optimistic Locking (CONTENT-ADDRESS design)
// ============================================================================

/// Entity version number for optimistic locking.
/// u32 provides 4 billion versions per entity - sufficient for 136 years at 1 update/sec.
/// (claude, 2026-02-02, in-progress)
pub type Version = u32;

/// Maximum version value. If reached, reject further updates with Error::VersionOverflow.
pub const VERSION_MAX: Version = u32::MAX;

/// Reference count for content-addressed summaries.
/// u32 allows billions of references while bounding storage overhead to 4 bytes.
/// Used to enable safe GC of shared summary content.
pub type RefCount = u32;

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
pub(crate) struct NameCfKey(pub(crate) NameHash);

#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct NameCfValue(pub(crate) String);

impl ColumnFamily for Names {
    const CF_NAME: &'static str = "graph/names";
}

impl ColumnFamilyConfig<GraphBlockCacheConfig> for Names {
    /// Configure RocksDB options with shared block cache.
    ///
    /// Names CF is small and hot - use high cache priority to keep it in memory.
    fn cf_options(cache: &rocksdb::Cache, config: &GraphBlockCacheConfig) -> rocksdb::Options {
        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();

        // Shared block cache
        block_opts.set_block_cache(cache);
        block_opts.set_block_size(config.graph_block_size);

        // Cache index and filter blocks for Names CF
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

impl ColumnFamilySerde for Names {
    type Key = NameCfKey;
    type Value = NameCfValue;

    fn key_to_bytes(key: &Self::Key) -> Vec<u8> {
        key.0.as_bytes().to_vec()
    }

    fn key_from_bytes(bytes: &[u8]) -> anyhow::Result<Self::Key> {
        if bytes.len() != NameHash::SIZE {
            anyhow::bail!(
                "Invalid NameCfKey length: expected {}, got {}",
                NameHash::SIZE,
                bytes.len()
            );
        }
        let mut hash_bytes = [0u8; 8];
        hash_bytes.copy_from_slice(bytes);
        Ok(NameCfKey(NameHash::from_bytes(hash_bytes)))
    }
    // value_to_bytes and value_from_bytes use default impl (MessagePack + LZ4)
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
/// Size: ~31 bytes (vs ~200-500 bytes with inline summary)
/// (claude, 2026-02-02, in-progress) Added version and deleted for CONTENT-ADDRESS
#[derive(Archive, RkyvDeserialize, RkyvSerialize, Serialize, Deserialize)]
#[archive(check_bytes)]
pub(crate) struct NodeCfValue(
    pub(crate) Option<TemporalRange>,
    pub(crate) NameHash,            // Node name hash; full name in Names CF
    pub(crate) Option<SummaryHash>, // Content hash; full summary in NodeSummaries CF
    pub(crate) Version,             // Monotonic version for optimistic locking (starts at 1)
    pub(crate) bool,                // Deleted flag (tombstone) for soft deletes
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
/// Size: ~40 bytes (vs ~200-500 bytes with inline summary)
/// (claude, 2026-02-02, in-progress) Added version and deleted for CONTENT-ADDRESS
#[derive(Archive, RkyvDeserialize, RkyvSerialize, Serialize, Deserialize)]
#[archive(check_bytes)]
pub(crate) struct ForwardEdgeCfValue(
    pub(crate) Option<TemporalRange>, // Field 0: Temporal validity
    pub(crate) Option<f64>,           // Field 1: Optional weight
    pub(crate) Option<SummaryHash>,   // Field 2: Content hash; full summary in EdgeSummaries CF
    pub(crate) Version,               // Field 3: Monotonic version for optimistic locking
    pub(crate) bool,                  // Field 4: Deleted flag (tombstone)
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
#[derive(Archive, RkyvDeserialize, RkyvSerialize, Serialize, Deserialize)]
#[archive(check_bytes)]
pub(crate) struct ReverseEdgeCfValue(pub(crate) Option<TemporalRange>);

/// Edge fragments column family.
pub struct EdgeFragments;
#[derive(Serialize, Deserialize)]
pub struct EdgeFragmentCfKey(
    pub SrcId,
    pub DstId,
    pub NameHash, // Edge name stored as hash; full name in Names CF
    pub TimestampMilli,
);
#[derive(Serialize, Deserialize)]
pub struct EdgeFragmentCfValue(pub Option<TemporalRange>, pub FragmentContent);

pub type NodeName = String;
pub type EdgeName = String;
pub type NodeSummary = DataUrl;
pub type EdgeSummary = DataUrl;
pub type FragmentContent = DataUrl;

impl ColumnFamily for Nodes {
    const CF_NAME: &'static str = "graph/nodes";
}

impl ColumnFamilyConfig<GraphBlockCacheConfig> for Nodes {
    /// Configure RocksDB options with shared block cache.
    fn cf_options(cache: &rocksdb::Cache, config: &GraphBlockCacheConfig) -> rocksdb::Options {
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

impl Nodes {
    /// Create key-value pair from AddNode mutation.
    /// (claude, 2026-02-02, in-progress) Updated for CONTENT-ADDRESS: version=1, deleted=false
    pub fn record_from(args: &AddNode) -> (NodeCfKey, NodeCfValue) {
        let key = NodeCfKey(args.id);
        let name_hash = NameHash::from_name(&args.name);

        // Compute summary hash if summary is non-empty
        let summary_hash = if !args.summary.is_empty() {
            SummaryHash::from_summary(&args.summary).ok()
        } else {
            None
        };

        // New nodes start at version 1, not deleted
        let value = NodeCfValue(args.valid_range.clone(), name_hash, summary_hash, 1, false);
        (key, value)
    }

    /// Create and serialize to bytes (key bytes, value bytes).
    pub fn create_bytes(args: &AddNode) -> anyhow::Result<(Vec<u8>, Vec<u8>)> {
        let (key, value) = Self::record_from(args);
        let key_bytes = <Self as HotColumnFamilyRecord>::key_to_bytes(&key);
        let value_bytes = <Self as HotColumnFamilyRecord>::value_to_bytes(&value)?.to_vec();
        Ok((key_bytes, value_bytes))
    }
}

impl HotColumnFamilyRecord for Nodes {
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
pub struct NodeFragments;
#[derive(Serialize, Deserialize)]
pub struct NodeFragmentCfKey(pub Id, pub TimestampMilli);
#[derive(Serialize, Deserialize)]
pub struct NodeFragmentCfValue(pub Option<TemporalRange>, pub FragmentContent);

impl ColumnFamily for NodeFragments {
    const CF_NAME: &'static str = "graph/node_fragments";
}

impl ColumnFamilyConfig<GraphBlockCacheConfig> for NodeFragments {
    /// Configure RocksDB options with shared block cache.
    ///
    /// Fragment CFs use larger block size for variable-length content.
    fn cf_options(cache: &rocksdb::Cache, config: &GraphBlockCacheConfig) -> rocksdb::Options {
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

impl ColumnFamilySerde for NodeFragments {
    type Key = NodeFragmentCfKey;
    type Value = NodeFragmentCfValue;

    fn key_to_bytes(key: &Self::Key) -> Vec<u8> {
        // NodeFragmentCfKey(Id, TimestampMilli)
        // Layout: [Id bytes (16)] + [timestamp big-endian (8)]
        let mut bytes = Vec::with_capacity(24);
        bytes.extend_from_slice(&key.0.into_bytes());
        bytes.extend_from_slice(&key.1 .0.to_be_bytes());
        bytes
    }

    fn key_from_bytes(bytes: &[u8]) -> anyhow::Result<Self::Key> {
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
}

impl ColumnFamily for EdgeFragments {
    const CF_NAME: &'static str = "graph/edge_fragments";
}

impl ColumnFamilyConfig<GraphBlockCacheConfig> for EdgeFragments {
    /// Configure RocksDB options with shared block cache.
    ///
    /// Fragment CFs use larger block size for variable-length content.
    fn cf_options(cache: &rocksdb::Cache, config: &GraphBlockCacheConfig) -> rocksdb::Options {
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

impl ColumnFamilySerde for EdgeFragments {
    type Key = EdgeFragmentCfKey;
    type Value = EdgeFragmentCfValue;

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

    fn key_from_bytes(bytes: &[u8]) -> anyhow::Result<Self::Key> {
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
}

impl ColumnFamily for ForwardEdges {
    const CF_NAME: &'static str = "graph/forward_edges";
}

impl ColumnFamilyConfig<GraphBlockCacheConfig> for ForwardEdges {
    /// Configure RocksDB options with shared block cache.
    fn cf_options(cache: &rocksdb::Cache, config: &GraphBlockCacheConfig) -> rocksdb::Options {
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

impl ForwardEdges {
    /// Create key-value pair from AddEdge mutation.
    /// (claude, 2026-02-02, in-progress) Updated for CONTENT-ADDRESS: version=1, deleted=false
    pub fn record_from(args: &AddEdge) -> (ForwardEdgeCfKey, ForwardEdgeCfValue) {
        let name_hash = NameHash::from_name(&args.name);
        let key = ForwardEdgeCfKey(args.source_node_id, args.target_node_id, name_hash);

        // Compute summary hash if summary is non-empty
        let summary_hash = if !args.summary.is_empty() {
            SummaryHash::from_summary(&args.summary).ok()
        } else {
            None
        };

        // New edges start at version 1, not deleted
        let value = ForwardEdgeCfValue(args.valid_range.clone(), args.weight, summary_hash, 1, false);
        (key, value)
    }

    /// Create and serialize to bytes (key bytes, value bytes).
    pub fn create_bytes(args: &AddEdge) -> anyhow::Result<(Vec<u8>, Vec<u8>)> {
        let (key, value) = Self::record_from(args);
        let key_bytes = <Self as HotColumnFamilyRecord>::key_to_bytes(&key);
        let value_bytes = <Self as HotColumnFamilyRecord>::value_to_bytes(&value)?.to_vec();
        Ok((key_bytes, value_bytes))
    }
}

impl HotColumnFamilyRecord for ForwardEdges {
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

impl ColumnFamily for ReverseEdges {
    const CF_NAME: &'static str = "graph/reverse_edges";
}

impl ColumnFamilyConfig<GraphBlockCacheConfig> for ReverseEdges {
    /// Configure RocksDB options with shared block cache.
    fn cf_options(cache: &rocksdb::Cache, config: &GraphBlockCacheConfig) -> rocksdb::Options {
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

impl ReverseEdges {
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
}

impl HotColumnFamilyRecord for ReverseEdges {
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
/// RefCount enables safe GC when all references are removed.
///
/// Key: SummaryHash (8 bytes, content-addressable)
/// Value: (RefCount, NodeSummary) - refcount + content
pub(crate) struct NodeSummaries;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct NodeSummaryCfKey(pub(crate) SummaryHash);

/// Value stores refcount + summary to enable safe GC of shared content.
/// Field 0: RefCount - number of entities referencing this summary
/// Field 1: NodeSummary - the actual summary content
#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct NodeSummaryCfValue(pub(crate) RefCount, pub(crate) NodeSummary);

impl ColumnFamily for NodeSummaries {
    const CF_NAME: &'static str = "graph/node_summaries";
}

impl ColumnFamilyConfig<GraphBlockCacheConfig> for NodeSummaries {
    /// Configure RocksDB options with shared block cache.
    fn cf_options(cache: &rocksdb::Cache, config: &GraphBlockCacheConfig) -> rocksdb::Options {
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

impl ColumnFamilySerde for NodeSummaries {
    type Key = NodeSummaryCfKey;
    type Value = NodeSummaryCfValue;

    fn key_to_bytes(key: &Self::Key) -> Vec<u8> {
        key.0.as_bytes().to_vec()
    }

    fn key_from_bytes(bytes: &[u8]) -> anyhow::Result<Self::Key> {
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
    // value_to_bytes and value_from_bytes use default impl (MessagePack + LZ4)
}

/// EdgeSummaries column family (COLD - infrequently accessed).
///
/// Stores edge summary content keyed by content hash (SummaryHash).
/// Content-addressable: identical summaries stored once, referenced by hash.
/// RefCount enables safe GC when all references are removed.
///
/// Key: SummaryHash (8 bytes, content-addressable)
/// Value: (RefCount, EdgeSummary) - refcount + content
pub(crate) struct EdgeSummaries;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct EdgeSummaryCfKey(pub(crate) SummaryHash);

/// Value stores refcount + summary to enable safe GC of shared content.
/// Field 0: RefCount - number of entities referencing this summary
/// Field 1: EdgeSummary - the actual summary content
#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct EdgeSummaryCfValue(pub(crate) RefCount, pub(crate) EdgeSummary);

impl ColumnFamily for EdgeSummaries {
    const CF_NAME: &'static str = "graph/edge_summaries";
}

impl ColumnFamilyConfig<GraphBlockCacheConfig> for EdgeSummaries {
    /// Configure RocksDB options with shared block cache.
    fn cf_options(cache: &rocksdb::Cache, config: &GraphBlockCacheConfig) -> rocksdb::Options {
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

impl ColumnFamilySerde for EdgeSummaries {
    type Key = EdgeSummaryCfKey;
    type Value = EdgeSummaryCfValue;

    fn key_to_bytes(key: &Self::Key) -> Vec<u8> {
        key.0.as_bytes().to_vec()
    }

    fn key_from_bytes(bytes: &[u8]) -> anyhow::Result<Self::Key> {
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
    // value_to_bytes and value_from_bytes use default impl (MessagePack + LZ4)
}

// ============================================================================
// Reverse Index Column Families (CONTENT-ADDRESS design)
// (claude, 2026-02-02, in-progress)
// ============================================================================

/// NodeSummaryIndex column family - reverse lookup from SummaryHash to nodes.
///
/// Key: (SummaryHash, Id, Version) = 28 bytes
/// Value: 1-byte marker (CURRENT=0x01, STALE=0x00)
///
/// Enables: "find all nodes with this content hash"
pub(crate) struct NodeSummaryIndex;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct NodeSummaryIndexCfKey(
    pub(crate) SummaryHash, // 8 bytes - prefix for hash lookup
    pub(crate) Id,          // 16 bytes - node_id
    pub(crate) Version,     // 4 bytes - version
);

/// 1-byte marker: 0x01 = current, 0x00 = stale
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub(crate) struct NodeSummaryIndexCfValue(pub(crate) u8);

impl NodeSummaryIndexCfValue {
    pub const CURRENT: u8 = 0x01;
    pub const STALE: u8 = 0x00;

    pub fn current() -> Self {
        Self(Self::CURRENT)
    }

    pub fn stale() -> Self {
        Self(Self::STALE)
    }

    pub fn is_current(&self) -> bool {
        self.0 == Self::CURRENT
    }
}

impl ColumnFamily for NodeSummaryIndex {
    const CF_NAME: &'static str = "graph/node_summary_index";
}

impl ColumnFamilyConfig<GraphBlockCacheConfig> for NodeSummaryIndex {
    fn cf_options(cache: &rocksdb::Cache, config: &GraphBlockCacheConfig) -> rocksdb::Options {
        use rocksdb::SliceTransform;

        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();

        block_opts.set_block_cache(cache);
        block_opts.set_block_size(config.graph_block_size);

        if config.cache_index_and_filter_blocks {
            block_opts.set_cache_index_and_filter_blocks(true);
        }
        if config.pin_l0_filter_and_index {
            block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
        }

        // Enable bloom filter for prefix lookups
        block_opts.set_bloom_filter(10.0, false);

        opts.set_block_based_table_factory(&block_opts);

        // Key layout: [summary_hash (8)] + [node_id (16)] + [version (4)] = 28 bytes
        // Use 8-byte prefix to scan all nodes with a given hash
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(8));
        opts.set_memtable_prefix_bloom_ratio(0.2);

        opts
    }
}

impl ColumnFamilySerde for NodeSummaryIndex {
    type Key = NodeSummaryIndexCfKey;
    type Value = NodeSummaryIndexCfValue;

    fn key_to_bytes(key: &Self::Key) -> Vec<u8> {
        // Layout: [summary_hash (8)] + [node_id (16)] + [version (4)] = 28 bytes
        let mut bytes = Vec::with_capacity(28);
        bytes.extend_from_slice(key.0.as_bytes());
        bytes.extend_from_slice(&key.1.into_bytes());
        bytes.extend_from_slice(&key.2.to_be_bytes());
        bytes
    }

    fn key_from_bytes(bytes: &[u8]) -> anyhow::Result<Self::Key> {
        if bytes.len() != 28 {
            anyhow::bail!(
                "Invalid NodeSummaryIndexCfKey length: expected 28, got {}",
                bytes.len()
            );
        }

        let mut hash_bytes = [0u8; 8];
        hash_bytes.copy_from_slice(&bytes[0..8]);

        let mut id_bytes = [0u8; 16];
        id_bytes.copy_from_slice(&bytes[8..24]);

        let mut version_bytes = [0u8; 4];
        version_bytes.copy_from_slice(&bytes[24..28]);
        let version = u32::from_be_bytes(version_bytes);

        Ok(NodeSummaryIndexCfKey(
            SummaryHash::from_bytes(hash_bytes),
            Id::from_bytes(id_bytes),
            version,
        ))
    }

    fn value_to_bytes(value: &Self::Value) -> anyhow::Result<Vec<u8>> {
        Ok(vec![value.0])
    }

    fn value_from_bytes(bytes: &[u8]) -> anyhow::Result<Self::Value> {
        if bytes.len() != 1 {
            anyhow::bail!(
                "Invalid NodeSummaryIndexCfValue length: expected 1, got {}",
                bytes.len()
            );
        }
        Ok(NodeSummaryIndexCfValue(bytes[0]))
    }
}

/// EdgeSummaryIndex column family - reverse lookup from SummaryHash to edges.
///
/// Key: (SummaryHash, SrcId, DstId, NameHash, Version) = 52 bytes
/// Value: 1-byte marker (CURRENT=0x01, STALE=0x00)
///
/// Enables: "find all edges with this content hash"
pub(crate) struct EdgeSummaryIndex;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct EdgeSummaryIndexCfKey(
    pub(crate) SummaryHash, // 8 bytes - prefix for hash lookup
    pub(crate) SrcId,       // 16 bytes
    pub(crate) DstId,       // 16 bytes
    pub(crate) NameHash,    // 8 bytes
    pub(crate) Version,     // 4 bytes
);

/// 1-byte marker: 0x01 = current, 0x00 = stale
#[derive(Serialize, Deserialize, Debug, Clone, Copy)]
pub(crate) struct EdgeSummaryIndexCfValue(pub(crate) u8);

impl EdgeSummaryIndexCfValue {
    pub const CURRENT: u8 = 0x01;
    pub const STALE: u8 = 0x00;

    pub fn current() -> Self {
        Self(Self::CURRENT)
    }

    pub fn stale() -> Self {
        Self(Self::STALE)
    }

    pub fn is_current(&self) -> bool {
        self.0 == Self::CURRENT
    }
}

impl ColumnFamily for EdgeSummaryIndex {
    const CF_NAME: &'static str = "graph/edge_summary_index";
}

impl ColumnFamilyConfig<GraphBlockCacheConfig> for EdgeSummaryIndex {
    fn cf_options(cache: &rocksdb::Cache, config: &GraphBlockCacheConfig) -> rocksdb::Options {
        use rocksdb::SliceTransform;

        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();

        block_opts.set_block_cache(cache);
        block_opts.set_block_size(config.graph_block_size);

        if config.cache_index_and_filter_blocks {
            block_opts.set_cache_index_and_filter_blocks(true);
        }
        if config.pin_l0_filter_and_index {
            block_opts.set_pin_l0_filter_and_index_blocks_in_cache(true);
        }

        // Enable bloom filter for prefix lookups
        block_opts.set_bloom_filter(10.0, false);

        opts.set_block_based_table_factory(&block_opts);

        // Key layout: [summary_hash (8)] + [src_id (16)] + [dst_id (16)] + [name_hash (8)] + [version (4)] = 52 bytes
        // Use 8-byte prefix to scan all edges with a given hash
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(8));
        opts.set_memtable_prefix_bloom_ratio(0.2);

        opts
    }
}

impl ColumnFamilySerde for EdgeSummaryIndex {
    type Key = EdgeSummaryIndexCfKey;
    type Value = EdgeSummaryIndexCfValue;

    fn key_to_bytes(key: &Self::Key) -> Vec<u8> {
        // Layout: [summary_hash (8)] + [src_id (16)] + [dst_id (16)] + [name_hash (8)] + [version (4)] = 52 bytes
        let mut bytes = Vec::with_capacity(52);
        bytes.extend_from_slice(key.0.as_bytes());
        bytes.extend_from_slice(&key.1.into_bytes());
        bytes.extend_from_slice(&key.2.into_bytes());
        bytes.extend_from_slice(key.3.as_bytes());
        bytes.extend_from_slice(&key.4.to_be_bytes());
        bytes
    }

    fn key_from_bytes(bytes: &[u8]) -> anyhow::Result<Self::Key> {
        if bytes.len() != 52 {
            anyhow::bail!(
                "Invalid EdgeSummaryIndexCfKey length: expected 52, got {}",
                bytes.len()
            );
        }

        let mut hash_bytes = [0u8; 8];
        hash_bytes.copy_from_slice(&bytes[0..8]);

        let mut src_id_bytes = [0u8; 16];
        src_id_bytes.copy_from_slice(&bytes[8..24]);

        let mut dst_id_bytes = [0u8; 16];
        dst_id_bytes.copy_from_slice(&bytes[24..40]);

        let mut name_hash_bytes = [0u8; 8];
        name_hash_bytes.copy_from_slice(&bytes[40..48]);

        let mut version_bytes = [0u8; 4];
        version_bytes.copy_from_slice(&bytes[48..52]);
        let version = u32::from_be_bytes(version_bytes);

        Ok(EdgeSummaryIndexCfKey(
            SummaryHash::from_bytes(hash_bytes),
            Id::from_bytes(src_id_bytes),
            Id::from_bytes(dst_id_bytes),
            NameHash::from_bytes(name_hash_bytes),
            version,
        ))
    }

    fn value_to_bytes(value: &Self::Value) -> anyhow::Result<Vec<u8>> {
        Ok(vec![value.0])
    }

    fn value_from_bytes(bytes: &[u8]) -> anyhow::Result<Self::Value> {
        if bytes.len() != 1 {
            anyhow::bail!(
                "Invalid EdgeSummaryIndexCfValue length: expected 1, got {}",
                bytes.len()
            );
        }
        Ok(EdgeSummaryIndexCfValue(bytes[0]))
    }
}

// ============================================================================
// GraphMeta Column Family - Graph-level metadata (GC cursors, future config)
// (claude, 2026-02-02, implemented) CONTENT-ADDRESS GC cursor persistence
// ============================================================================

/// GraphMeta column family - stores graph-level metadata.
///
/// Key: discriminant byte (1 byte)
/// Value: field-dependent payload (cursor bytes for GC cursors)
///
/// Pattern borrowed from vector::GraphMeta for consistency.
pub(crate) struct GraphMeta;

/// GraphMeta field enum - discriminated union for different metadata fields.
///
/// Each variant stores a different type of graph-level metadata.
/// The discriminant byte is stored in the key, the value bytes depend on variant.
#[derive(Debug, Clone)]
pub(crate) enum GraphMetaField {
    /// GC cursor for NodeSummaries CF - stores last processed key bytes
    GcCursorNodeSummaries(Vec<u8>),      // 0x00
    /// GC cursor for EdgeSummaries CF
    GcCursorEdgeSummaries(Vec<u8>),      // 0x01
    /// GC cursor for NodeSummaryIndex CF
    GcCursorNodeSummaryIndex(Vec<u8>),   // 0x02
    /// GC cursor for EdgeSummaryIndex CF
    GcCursorEdgeSummaryIndex(Vec<u8>),   // 0x03
    /// GC cursor for Node tombstones
    GcCursorNodeTombstones(Vec<u8>),     // 0x04
    /// GC cursor for Edge tombstones
    GcCursorEdgeTombstones(Vec<u8>),     // 0x05
}

impl GraphMetaField {
    /// Get the discriminant byte for key serialization
    pub fn discriminant(&self) -> u8 {
        match self {
            Self::GcCursorNodeSummaries(_) => 0x00,
            Self::GcCursorEdgeSummaries(_) => 0x01,
            Self::GcCursorNodeSummaryIndex(_) => 0x02,
            Self::GcCursorEdgeSummaryIndex(_) => 0x03,
            Self::GcCursorNodeTombstones(_) => 0x04,
            Self::GcCursorEdgeTombstones(_) => 0x05,
        }
    }

    /// Create a field variant from discriminant (with empty payload)
    pub fn from_discriminant(d: u8) -> anyhow::Result<Self> {
        match d {
            0x00 => Ok(Self::GcCursorNodeSummaries(vec![])),
            0x01 => Ok(Self::GcCursorEdgeSummaries(vec![])),
            0x02 => Ok(Self::GcCursorNodeSummaryIndex(vec![])),
            0x03 => Ok(Self::GcCursorEdgeSummaryIndex(vec![])),
            0x04 => Ok(Self::GcCursorNodeTombstones(vec![])),
            0x05 => Ok(Self::GcCursorEdgeTombstones(vec![])),
            _ => anyhow::bail!("Unknown GraphMetaField discriminant: {}", d),
        }
    }

    /// Get the inner cursor bytes
    pub fn cursor_bytes(&self) -> &[u8] {
        match self {
            Self::GcCursorNodeSummaries(v)
            | Self::GcCursorEdgeSummaries(v)
            | Self::GcCursorNodeSummaryIndex(v)
            | Self::GcCursorEdgeSummaryIndex(v)
            | Self::GcCursorNodeTombstones(v)
            | Self::GcCursorEdgeTombstones(v) => v,
        }
    }
}

/// GraphMeta key: just the discriminant byte (1 byte total)
#[derive(Debug, Clone)]
pub(crate) struct GraphMetaCfKey(pub(crate) GraphMetaField);

impl GraphMetaCfKey {
    /// Create key for NodeSummaries GC cursor
    pub fn gc_cursor_node_summaries() -> Self {
        Self(GraphMetaField::GcCursorNodeSummaries(vec![]))
    }

    /// Create key for EdgeSummaries GC cursor
    pub fn gc_cursor_edge_summaries() -> Self {
        Self(GraphMetaField::GcCursorEdgeSummaries(vec![]))
    }

    /// Create key for NodeSummaryIndex GC cursor
    pub fn gc_cursor_node_summary_index() -> Self {
        Self(GraphMetaField::GcCursorNodeSummaryIndex(vec![]))
    }

    /// Create key for EdgeSummaryIndex GC cursor
    pub fn gc_cursor_edge_summary_index() -> Self {
        Self(GraphMetaField::GcCursorEdgeSummaryIndex(vec![]))
    }

    /// Create key for Node tombstones GC cursor
    pub fn gc_cursor_node_tombstones() -> Self {
        Self(GraphMetaField::GcCursorNodeTombstones(vec![]))
    }

    /// Create key for Edge tombstones GC cursor
    pub fn gc_cursor_edge_tombstones() -> Self {
        Self(GraphMetaField::GcCursorEdgeTombstones(vec![]))
    }
}

/// GraphMeta value: wraps the field with actual cursor data
#[derive(Debug, Clone)]
pub(crate) struct GraphMetaCfValue(pub(crate) GraphMetaField);

impl ColumnFamily for GraphMeta {
    const CF_NAME: &'static str = "graph/meta";
}

impl ColumnFamilyConfig<super::subsystem::GraphBlockCacheConfig> for GraphMeta {
    fn cf_options(cache: &rocksdb::Cache, config: &super::subsystem::GraphBlockCacheConfig) -> rocksdb::Options {
        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();

        block_opts.set_block_cache(cache);
        block_opts.set_block_size(config.graph_block_size);
        opts.set_block_based_table_factory(&block_opts);
        opts
    }
}

impl GraphMeta {
    /// Serialize key: just the discriminant byte
    pub fn key_to_bytes(key: &GraphMetaCfKey) -> Vec<u8> {
        vec![key.0.discriminant()]
    }

    /// Deserialize key from bytes
    pub fn key_from_bytes(bytes: &[u8]) -> anyhow::Result<GraphMetaCfKey> {
        if bytes.len() != 1 {
            anyhow::bail!("Invalid GraphMetaCfKey length: expected 1, got {}", bytes.len());
        }
        let field = GraphMetaField::from_discriminant(bytes[0])?;
        Ok(GraphMetaCfKey(field))
    }

    /// Serialize value: just the cursor bytes (variable length)
    pub fn value_to_bytes(value: &GraphMetaCfValue) -> Vec<u8> {
        value.0.cursor_bytes().to_vec()
    }

    /// Deserialize value using key's field variant for type info
    pub fn value_from_bytes(key: &GraphMetaCfKey, bytes: &[u8]) -> anyhow::Result<GraphMetaCfValue> {
        let field = match key.0.discriminant() {
            0x00 => GraphMetaField::GcCursorNodeSummaries(bytes.to_vec()),
            0x01 => GraphMetaField::GcCursorEdgeSummaries(bytes.to_vec()),
            0x02 => GraphMetaField::GcCursorNodeSummaryIndex(bytes.to_vec()),
            0x03 => GraphMetaField::GcCursorEdgeSummaryIndex(bytes.to_vec()),
            0x04 => GraphMetaField::GcCursorNodeTombstones(bytes.to_vec()),
            0x05 => GraphMetaField::GcCursorEdgeTombstones(bytes.to_vec()),
            d => anyhow::bail!("Unknown discriminant: {}", d),
        };
        Ok(GraphMetaCfValue(field))
    }
}

/// All column families used in the database.
/// This is the authoritative list that should be used when opening the database.
/// (claude, 2026-02-02, in-progress) Added reverse index CFs for CONTENT-ADDRESS
pub(crate) const ALL_COLUMN_FAMILIES: &[&str] = &[
    Names::CF_NAME,
    Nodes::CF_NAME,
    NodeFragments::CF_NAME,
    NodeSummaries::CF_NAME,      // Phase 2: Cold CF for node summaries
    NodeSummaryIndex::CF_NAME,   // CONTENT-ADDRESS: Reverse index hash→nodes
    EdgeFragments::CF_NAME,
    EdgeSummaries::CF_NAME,      // Phase 2: Cold CF for edge summaries
    EdgeSummaryIndex::CF_NAME,   // CONTENT-ADDRESS: Reverse index hash→edges
    ForwardEdges::CF_NAME,
    ReverseEdges::CF_NAME,
    GraphMeta::CF_NAME,          // CONTENT-ADDRESS: Graph-level metadata (GC cursors)
];

#[cfg(test)]
mod tests {
    use super::super::mutation::AddEdge;
    use super::*;
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
}
