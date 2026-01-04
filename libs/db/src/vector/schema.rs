//! Column family definitions for the vector module.
//!
//! All vector CFs use the `vector/` prefix to avoid collisions with graph CFs.
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
//! - Values use MessagePack serialization (self-describing, compact)
//!
//! See ROADMAP.md for complete schema documentation.

use crate::rocksdb::{ColumnFamily, ColumnFamilyConfig};
use crate::{Id, TimestampMilli};
use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::distance::Distance;
use super::subsystem::VectorBlockCacheConfig;

// ============================================================================
// Common Types
// ============================================================================

/// Identifier for an embedding space.
///
/// This is the primary key of the `EmbeddingSpecs` CF and serves as a
/// foreign key in all other vector CFs. Using a type alias makes the
/// relationship explicit without relying on comments.
pub(crate) type EmbeddingCode = u64;

/// Internal vector identifier within an embedding space.
///
/// Compact u32 index used for HNSW graph edges and storage.
/// Mapped bidirectionally to external ULIDs via IdForward/IdReverse CFs.
pub(crate) type VecId = u32;

/// HNSW graph layer index.
///
/// Layer 0 is the base layer (densest); higher layers are sparser.
/// Maximum layer is typically determined by `ln(N) / ln(M)`.
pub(crate) type HnswLayer = u8;

/// Serialized RoaringBitmap bytes.
///
/// RoaringBitmap serialization/deserialization is handled externally.
/// Used for HNSW edge neighbor lists and ID allocator free lists.
pub(crate) type RoaringBitmapBytes = Vec<u8>;

/// RaBitQ quantized binary code.
///
/// Fixed-size binary representation of a vector for fast approximate distance.
/// Size depends on vector dimensionality (e.g., 16 bytes for 128-dim 1-bit RaBitQ).
pub(crate) type RabitqCode = Vec<u8>;

// ============================================================================
// Column Family Names
// ============================================================================

/// All column family names for the vector module.
pub const ALL_COLUMN_FAMILIES: &[&str] = &[
    EmbeddingSpecs::CF_NAME,
    Vectors::CF_NAME,
    Edges::CF_NAME,
    BinaryCodes::CF_NAME,
    VecMeta::CF_NAME,
    GraphMeta::CF_NAME,
    IdForward::CF_NAME,
    IdReverse::CF_NAME,
    IdAlloc::CF_NAME,
    Pending::CF_NAME,
];

// ============================================================================
// EmbeddingSpecs CF
// ============================================================================

/// Embedding specifications column family.
///
/// Stores the persistent specification of each embedding space (model, dim, distance).
/// The runtime registry (`vector::EmbeddingRegistry`) pre-warms from this CF.
///
/// Key: [embedding_code: u64] = 8 bytes
/// Value: EmbeddingSpec (model, dim, distance)
pub(crate) struct EmbeddingSpecs;

/// Domain struct for embedding specification.
///
/// Rich struct with >2 fields, so defined separately per naming convention.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct EmbeddingSpec {
    /// Model name (e.g., "gemma", "qwen3", "ada-002")
    pub(crate) model: String,
    /// Vector dimensionality (e.g., 128, 768, 1536)
    pub(crate) dim: u32,
    /// Distance metric for similarity computation
    pub(crate) distance: Distance,
}

/// EmbeddingSpecs key: the embedding space identifier (primary key)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct EmbeddingSpecCfKey(pub(crate) EmbeddingCode);

/// EmbeddingSpecs value: wraps EmbeddingSpec
#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct EmbeddingSpecCfValue(pub(crate) EmbeddingSpec);

impl ColumnFamily for EmbeddingSpecs {
    const CF_NAME: &'static str = "vector/embedding_specs";
}

impl ColumnFamilyConfig<VectorBlockCacheConfig> for EmbeddingSpecs {
    fn cf_options(cache: &rocksdb::Cache, config: &VectorBlockCacheConfig) -> rocksdb::Options {
        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();
        block_opts.set_bloom_filter(10.0, false);
        block_opts.set_block_cache(cache);
        block_opts.set_block_size(config.default_block_size);
        opts.set_block_based_table_factory(&block_opts);
        opts
    }
}

impl EmbeddingSpecs {
    pub fn key_to_bytes(key: &EmbeddingSpecCfKey) -> Vec<u8> {
        key.0.to_be_bytes().to_vec()
    }

    pub fn key_from_bytes(bytes: &[u8]) -> Result<EmbeddingSpecCfKey> {
        if bytes.len() != 8 {
            anyhow::bail!(
                "Invalid EmbeddingSpecCfKey length: expected 8, got {}",
                bytes.len()
            );
        }
        let mut arr = [0u8; 8];
        arr.copy_from_slice(bytes);
        Ok(EmbeddingSpecCfKey(u64::from_be_bytes(arr)))
    }

    pub fn value_to_bytes(value: &EmbeddingSpecCfValue) -> Result<Vec<u8>> {
        let bytes = rmp_serde::to_vec(value)?;
        Ok(bytes)
    }

    pub fn value_from_bytes(bytes: &[u8]) -> Result<EmbeddingSpecCfValue> {
        let value = rmp_serde::from_slice(bytes)?;
        Ok(value)
    }
}

// ============================================================================
// Vectors CF
// ============================================================================

/// Vectors column family - raw f32 vector storage.
///
/// Key: [embedding: u64] + [vec_id: u32] = 12 bytes
/// Value: f32[dim] as raw bytes (e.g., 512 bytes for 128D)
pub(crate) struct Vectors;

/// Vectors key: (embedding_code, vec_id)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct VectorCfKey(pub(crate) EmbeddingCode, pub(crate) VecId);

/// Vectors value: raw f32 vector data
#[derive(Debug, Clone)]
pub(crate) struct VectorCfValue(pub(crate) Vec<f32>);

impl ColumnFamily for Vectors {
    const CF_NAME: &'static str = "vector/vectors";
}

impl ColumnFamilyConfig<VectorBlockCacheConfig> for Vectors {
    fn cf_options(cache: &rocksdb::Cache, config: &VectorBlockCacheConfig) -> rocksdb::Options {
        use rocksdb::SliceTransform;

        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();

        block_opts.set_block_cache(cache);
        block_opts.set_block_size(config.vector_block_size);
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(8));
        opts.set_block_based_table_factory(&block_opts);
        opts
    }
}

impl Vectors {
    pub fn key_to_bytes(key: &VectorCfKey) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(12);
        bytes.extend_from_slice(&key.0.to_be_bytes());
        bytes.extend_from_slice(&key.1.to_be_bytes());
        bytes
    }

    pub fn key_from_bytes(bytes: &[u8]) -> Result<VectorCfKey> {
        if bytes.len() != 12 {
            anyhow::bail!(
                "Invalid VectorCfKey length: expected 12, got {}",
                bytes.len()
            );
        }
        let embedding_code = u64::from_be_bytes(bytes[0..8].try_into()?);
        let vec_id = u32::from_be_bytes(bytes[8..12].try_into()?);
        Ok(VectorCfKey(embedding_code, vec_id))
    }

    /// Store vector as raw f32 bytes (no compression for fast access)
    pub fn value_to_bytes(value: &VectorCfValue) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(value.0.len() * 4);
        for &v in &value.0 {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    /// Load vector from raw f32 bytes
    pub fn value_from_bytes(bytes: &[u8]) -> Result<VectorCfValue> {
        if bytes.len() % 4 != 0 {
            anyhow::bail!(
                "Invalid vector bytes length: {} is not divisible by 4",
                bytes.len()
            );
        }
        let mut vector = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            vector.push(f32::from_le_bytes(chunk.try_into()?));
        }
        Ok(VectorCfValue(vector))
    }
}

// ============================================================================
// Edges CF
// ============================================================================

/// Edges column family - HNSW graph edges stored as RoaringBitmap.
///
/// Key: [embedding: u64] + [vec_id: u32] + [layer: u8] = 13 bytes
/// Value: RoaringBitmap serialized (~50-200 bytes)
pub(crate) struct Edges;

/// Edges key: (embedding_code, vec_id, layer)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct EdgeCfKey(pub(crate) EmbeddingCode, pub(crate) VecId, pub(crate) HnswLayer);

/// Edges value: serialized RoaringBitmap of neighbor vec_ids
#[derive(Debug, Clone)]
pub(crate) struct EdgeCfValue(pub(crate) RoaringBitmapBytes);

impl ColumnFamily for Edges {
    const CF_NAME: &'static str = "vector/edges";
}

impl ColumnFamilyConfig<VectorBlockCacheConfig> for Edges {
    fn cf_options(cache: &rocksdb::Cache, config: &VectorBlockCacheConfig) -> rocksdb::Options {
        use rocksdb::SliceTransform;

        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();

        block_opts.set_block_cache(cache);
        block_opts.set_block_size(config.default_block_size);
        opts.set_compression_type(rocksdb::DBCompressionType::None);
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(12));
        opts.set_block_based_table_factory(&block_opts);
        opts
    }
}

impl Edges {
    pub fn key_to_bytes(key: &EdgeCfKey) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(13);
        bytes.extend_from_slice(&key.0.to_be_bytes());
        bytes.extend_from_slice(&key.1.to_be_bytes());
        bytes.push(key.2);
        bytes
    }

    pub fn key_from_bytes(bytes: &[u8]) -> Result<EdgeCfKey> {
        if bytes.len() != 13 {
            anyhow::bail!(
                "Invalid EdgeCfKey length: expected 13, got {}",
                bytes.len()
            );
        }
        let embedding_code = u64::from_be_bytes(bytes[0..8].try_into()?);
        let vec_id = u32::from_be_bytes(bytes[8..12].try_into()?);
        let layer = bytes[12];
        Ok(EdgeCfKey(embedding_code, vec_id, layer))
    }

    /// Serialize edge value (RoaringBitmap bytes passthrough)
    pub fn value_to_bytes(value: &EdgeCfValue) -> Vec<u8> {
        value.0.clone()
    }

    /// Deserialize edge value (RoaringBitmap bytes passthrough)
    pub fn value_from_bytes(bytes: &[u8]) -> Result<EdgeCfValue> {
        Ok(EdgeCfValue(bytes.to_vec()))
    }
}

// ============================================================================
// BinaryCodes CF
// ============================================================================

/// Binary codes column family - RaBitQ quantized vectors.
///
/// Key: [embedding: u64] + [vec_id: u32] = 12 bytes
/// Value: u8[code_size] (e.g., 16 bytes for 128-dim 1-bit RaBitQ)
pub(crate) struct BinaryCodes;

/// BinaryCodes key: (embedding_code, vec_id)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct BinaryCodeCfKey(pub(crate) EmbeddingCode, pub(crate) VecId);

/// BinaryCodes value: RaBitQ quantized code
#[derive(Debug, Clone)]
pub(crate) struct BinaryCodeCfValue(pub(crate) RabitqCode);

impl ColumnFamily for BinaryCodes {
    const CF_NAME: &'static str = "vector/binary_codes";
}

impl ColumnFamilyConfig<VectorBlockCacheConfig> for BinaryCodes {
    fn cf_options(cache: &rocksdb::Cache, config: &VectorBlockCacheConfig) -> rocksdb::Options {
        use rocksdb::SliceTransform;

        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();

        block_opts.set_block_cache(cache);
        block_opts.set_block_size(config.default_block_size);
        opts.set_compression_type(rocksdb::DBCompressionType::None);
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(8));
        opts.set_block_based_table_factory(&block_opts);
        opts
    }
}

impl BinaryCodes {
    pub fn key_to_bytes(key: &BinaryCodeCfKey) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(12);
        bytes.extend_from_slice(&key.0.to_be_bytes());
        bytes.extend_from_slice(&key.1.to_be_bytes());
        bytes
    }

    pub fn key_from_bytes(bytes: &[u8]) -> Result<BinaryCodeCfKey> {
        if bytes.len() != 12 {
            anyhow::bail!(
                "Invalid BinaryCodeCfKey length: expected 12, got {}",
                bytes.len()
            );
        }
        let embedding_code = u64::from_be_bytes(bytes[0..8].try_into()?);
        let vec_id = u32::from_be_bytes(bytes[8..12].try_into()?);
        Ok(BinaryCodeCfKey(embedding_code, vec_id))
    }

    /// Serialize binary code value
    pub fn value_to_bytes(value: &BinaryCodeCfValue) -> Vec<u8> {
        value.0.clone()
    }

    /// Deserialize binary code value
    pub fn value_from_bytes(bytes: &[u8]) -> Result<BinaryCodeCfValue> {
        Ok(BinaryCodeCfValue(bytes.to_vec()))
    }
}

// ============================================================================
// VecMeta CF
// ============================================================================

/// Vector metadata column family.
///
/// Key: [embedding: u64] + [vec_id: u32] = 12 bytes
/// Value: VecMetadata (max_layer, flags, created_at)
pub(crate) struct VecMeta;

/// Domain struct for vector metadata.
///
/// Rich struct with >2 fields, so defined separately per naming convention.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct VecMetadata {
    /// Maximum HNSW layer this vector appears in
    pub(crate) max_layer: u8,
    /// Flags (reserved for future use: deleted, etc.)
    pub(crate) flags: u8,
    /// Timestamp when vector was created (millis since epoch)
    pub(crate) created_at: u64,
}

/// VecMeta key: (embedding_code, vec_id)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct VecMetaCfKey(pub(crate) EmbeddingCode, pub(crate) VecId);

/// VecMeta value: wraps VecMetadata
#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct VecMetaCfValue(pub(crate) VecMetadata);

impl ColumnFamily for VecMeta {
    const CF_NAME: &'static str = "vector/vec_meta";
}

impl ColumnFamilyConfig<VectorBlockCacheConfig> for VecMeta {
    fn cf_options(cache: &rocksdb::Cache, config: &VectorBlockCacheConfig) -> rocksdb::Options {
        use rocksdb::SliceTransform;

        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();

        block_opts.set_block_cache(cache);
        block_opts.set_block_size(config.default_block_size);
        opts.set_compression_type(rocksdb::DBCompressionType::None);
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(8));
        opts.set_block_based_table_factory(&block_opts);
        opts
    }
}

impl VecMeta {
    pub fn key_to_bytes(key: &VecMetaCfKey) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(12);
        bytes.extend_from_slice(&key.0.to_be_bytes());
        bytes.extend_from_slice(&key.1.to_be_bytes());
        bytes
    }

    pub fn key_from_bytes(bytes: &[u8]) -> Result<VecMetaCfKey> {
        if bytes.len() != 12 {
            anyhow::bail!(
                "Invalid VecMetaCfKey length: expected 12, got {}",
                bytes.len()
            );
        }
        let embedding_code = u64::from_be_bytes(bytes[0..8].try_into()?);
        let vec_id = u32::from_be_bytes(bytes[8..12].try_into()?);
        Ok(VecMetaCfKey(embedding_code, vec_id))
    }

    pub fn value_to_bytes(value: &VecMetaCfValue) -> Result<Vec<u8>> {
        Ok(rmp_serde::to_vec(value)?)
    }

    pub fn value_from_bytes(bytes: &[u8]) -> Result<VecMetaCfValue> {
        Ok(rmp_serde::from_slice(bytes)?)
    }
}

// ============================================================================
// GraphMeta CF
// ============================================================================

/// Graph-level metadata column family.
///
/// Key: [embedding: u64] + [field: u8] = 9 bytes
/// Value: varies by field
pub(crate) struct GraphMeta;

/// GraphMeta field enum - single type for both key discrimination and value storage.
///
/// For keys: variant determines which field, inner value is ignored.
/// For values: variant determines type, inner value is the actual data.
#[derive(Debug, Clone)]
pub(crate) enum GraphMetaField {
    /// Entry point vec_id for the HNSW graph
    EntryPoint(VecId),
    /// Maximum level in the HNSW graph
    MaxLevel(HnswLayer),
    /// Total vector count in this embedding space
    Count(u32),
    /// Serialized HNSW configuration
    Config(Vec<u8>),
}

impl GraphMetaField {
    /// Get the discriminant byte for key serialization
    fn discriminant(&self) -> u8 {
        match self {
            Self::EntryPoint(_) => 0,
            Self::MaxLevel(_) => 1,
            Self::Count(_) => 2,
            Self::Config(_) => 3,
        }
    }
}

/// GraphMeta key: (embedding_code, field)
#[derive(Debug, Clone)]
pub(crate) struct GraphMetaCfKey(pub(crate) EmbeddingCode, pub(crate) GraphMetaField);

impl GraphMetaCfKey {
    /// Create key for entry_point field
    pub fn entry_point(embedding_code: EmbeddingCode) -> Self {
        Self(embedding_code, GraphMetaField::EntryPoint(0))
    }

    /// Create key for max_level field
    pub fn max_level(embedding_code: EmbeddingCode) -> Self {
        Self(embedding_code, GraphMetaField::MaxLevel(0))
    }

    /// Create key for count field
    pub fn count(embedding_code: EmbeddingCode) -> Self {
        Self(embedding_code, GraphMetaField::Count(0))
    }

    /// Create key for config field
    pub fn config(embedding_code: EmbeddingCode) -> Self {
        Self(embedding_code, GraphMetaField::Config(Vec::new()))
    }
}

/// Type alias distinguishing "field as actual value" from "field as key discriminant"
pub(crate) type GraphMetaValue = GraphMetaField;

/// GraphMeta value: wraps the field enum with actual data
#[derive(Debug, Clone)]
pub(crate) struct GraphMetaCfValue(pub(crate) GraphMetaValue);

impl ColumnFamily for GraphMeta {
    const CF_NAME: &'static str = "vector/graph_meta";
}

impl ColumnFamilyConfig<VectorBlockCacheConfig> for GraphMeta {
    fn cf_options(cache: &rocksdb::Cache, config: &VectorBlockCacheConfig) -> rocksdb::Options {
        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();

        block_opts.set_block_cache(cache);
        block_opts.set_block_size(config.default_block_size);
        opts.set_block_based_table_factory(&block_opts);
        opts
    }
}

impl GraphMeta {
    pub fn key_to_bytes(key: &GraphMetaCfKey) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(9);
        bytes.extend_from_slice(&key.0.to_be_bytes());
        bytes.push(key.1.discriminant()); // Only discriminant, not inner value
        bytes
    }

    pub fn key_from_bytes(bytes: &[u8]) -> Result<GraphMetaCfKey> {
        if bytes.len() != 9 {
            anyhow::bail!(
                "Invalid GraphMetaCfKey length: expected 9, got {}",
                bytes.len()
            );
        }
        let embedding_code = u64::from_be_bytes(bytes[0..8].try_into()?);
        let discriminant = bytes[8];
        // Create field with placeholder value based on discriminant
        let field = match discriminant {
            0 => GraphMetaField::EntryPoint(0),
            1 => GraphMetaField::MaxLevel(0),
            2 => GraphMetaField::Count(0),
            3 => GraphMetaField::Config(Vec::new()),
            _ => anyhow::bail!("Unknown GraphMeta discriminant: {}", discriminant),
        };
        Ok(GraphMetaCfKey(embedding_code, field))
    }

    /// Serialize value based on variant type
    pub fn value_to_bytes(value: &GraphMetaCfValue) -> Vec<u8> {
        match &value.0 {
            GraphMetaField::EntryPoint(v) => v.to_be_bytes().to_vec(),
            GraphMetaField::MaxLevel(v) => vec![*v],
            GraphMetaField::Count(v) => v.to_be_bytes().to_vec(),
            GraphMetaField::Config(v) => v.clone(),
        }
    }

    /// Deserialize value using key's field variant for type info
    pub fn value_from_bytes(key: &GraphMetaCfKey, bytes: &[u8]) -> Result<GraphMetaCfValue> {
        let field = match &key.1 {
            GraphMetaField::EntryPoint(_) => {
                if bytes.len() != 4 {
                    anyhow::bail!("Invalid EntryPoint length: expected 4, got {}", bytes.len());
                }
                GraphMetaField::EntryPoint(u32::from_be_bytes(bytes.try_into()?))
            }
            GraphMetaField::MaxLevel(_) => {
                if bytes.len() != 1 {
                    anyhow::bail!("Invalid MaxLevel length: expected 1, got {}", bytes.len());
                }
                GraphMetaField::MaxLevel(bytes[0])
            }
            GraphMetaField::Count(_) => {
                if bytes.len() != 4 {
                    anyhow::bail!("Invalid Count length: expected 4, got {}", bytes.len());
                }
                GraphMetaField::Count(u32::from_be_bytes(bytes.try_into()?))
            }
            GraphMetaField::Config(_) => {
                GraphMetaField::Config(bytes.to_vec())
            }
        };
        Ok(GraphMetaCfValue(field))
    }
}

// ============================================================================
// IdForward CF
// ============================================================================

/// Forward ID mapping: ULID -> vec_id.
///
/// Key: [embedding: u64] + [ulid: 16] = 24 bytes
/// Value: [vec_id: u32] = 4 bytes
pub(crate) struct IdForward;

/// IdForward key: (embedding_code, id)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct IdForwardCfKey(pub(crate) EmbeddingCode, pub(crate) Id);

/// IdForward value: vec_id mapping
#[derive(Debug, Clone)]
pub(crate) struct IdForwardCfValue(pub(crate) VecId);

impl ColumnFamily for IdForward {
    const CF_NAME: &'static str = "vector/id_forward";
}

impl ColumnFamilyConfig<VectorBlockCacheConfig> for IdForward {
    fn cf_options(cache: &rocksdb::Cache, config: &VectorBlockCacheConfig) -> rocksdb::Options {
        use rocksdb::SliceTransform;

        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();

        block_opts.set_bloom_filter(10.0, false);
        block_opts.set_block_cache(cache);
        block_opts.set_block_size(config.default_block_size);
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(8));
        opts.set_block_based_table_factory(&block_opts);
        opts
    }
}

impl IdForward {
    pub fn key_to_bytes(key: &IdForwardCfKey) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(24);
        bytes.extend_from_slice(&key.0.to_be_bytes());
        bytes.extend_from_slice(key.1.as_bytes());
        bytes
    }

    pub fn key_from_bytes(bytes: &[u8]) -> Result<IdForwardCfKey> {
        if bytes.len() != 24 {
            anyhow::bail!(
                "Invalid IdForwardCfKey length: expected 24, got {}",
                bytes.len()
            );
        }
        let embedding_code = u64::from_be_bytes(bytes[0..8].try_into()?);
        let mut ulid_bytes = [0u8; 16];
        ulid_bytes.copy_from_slice(&bytes[8..24]);
        Ok(IdForwardCfKey(embedding_code, Id::from_bytes(ulid_bytes)))
    }

    pub fn value_to_bytes(value: &IdForwardCfValue) -> Vec<u8> {
        value.0.to_be_bytes().to_vec()
    }

    pub fn value_from_bytes(bytes: &[u8]) -> Result<IdForwardCfValue> {
        if bytes.len() != 4 {
            anyhow::bail!(
                "Invalid IdForwardCfValue length: expected 4, got {}",
                bytes.len()
            );
        }
        Ok(IdForwardCfValue(u32::from_be_bytes(bytes.try_into()?)))
    }
}

// ============================================================================
// IdReverse CF
// ============================================================================

/// Reverse ID mapping: vec_id -> ULID.
///
/// Key: [embedding: u64] + [vec_id: u32] = 12 bytes
/// Value: [ulid: 16] = 16 bytes
pub(crate) struct IdReverse;

/// IdReverse key: (embedding_code, vec_id)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct IdReverseCfKey(pub(crate) EmbeddingCode, pub(crate) VecId);

/// IdReverse value: ULID mapping
#[derive(Debug, Clone)]
pub(crate) struct IdReverseCfValue(pub(crate) Id);

impl ColumnFamily for IdReverse {
    const CF_NAME: &'static str = "vector/id_reverse";
}

impl ColumnFamilyConfig<VectorBlockCacheConfig> for IdReverse {
    fn cf_options(cache: &rocksdb::Cache, config: &VectorBlockCacheConfig) -> rocksdb::Options {
        use rocksdb::SliceTransform;

        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();

        block_opts.set_block_cache(cache);
        block_opts.set_block_size(config.default_block_size);
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(8));
        opts.set_block_based_table_factory(&block_opts);
        opts
    }
}

impl IdReverse {
    pub fn key_to_bytes(key: &IdReverseCfKey) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(12);
        bytes.extend_from_slice(&key.0.to_be_bytes());
        bytes.extend_from_slice(&key.1.to_be_bytes());
        bytes
    }

    pub fn key_from_bytes(bytes: &[u8]) -> Result<IdReverseCfKey> {
        if bytes.len() != 12 {
            anyhow::bail!(
                "Invalid IdReverseCfKey length: expected 12, got {}",
                bytes.len()
            );
        }
        let embedding_code = u64::from_be_bytes(bytes[0..8].try_into()?);
        let vec_id = u32::from_be_bytes(bytes[8..12].try_into()?);
        Ok(IdReverseCfKey(embedding_code, vec_id))
    }

    pub fn value_to_bytes(value: &IdReverseCfValue) -> Vec<u8> {
        value.0.as_bytes().to_vec()
    }

    pub fn value_from_bytes(bytes: &[u8]) -> Result<IdReverseCfValue> {
        if bytes.len() != 16 {
            anyhow::bail!(
                "Invalid IdReverseCfValue length: expected 16, got {}",
                bytes.len()
            );
        }
        let mut ulid_bytes = [0u8; 16];
        ulid_bytes.copy_from_slice(bytes);
        Ok(IdReverseCfValue(Id::from_bytes(ulid_bytes)))
    }
}

// ============================================================================
// IdAlloc CF
// ============================================================================

/// ID allocator state.
///
/// Key: [embedding: u64] + [field: u8] = 9 bytes
/// Value: u32 (next_id) or RoaringBitmap (free list)
pub(crate) struct IdAlloc;

/// IdAlloc field enum - single type for both key discrimination and value storage.
///
/// For keys: variant determines which field, inner value is ignored.
/// For values: variant determines type, inner value is the actual data.
#[derive(Debug, Clone)]
pub(crate) enum IdAllocField {
    /// Next available vec_id to allocate
    NextId(VecId),
    /// Serialized RoaringBitmap of freed vec_ids available for reuse
    FreeBitmap(RoaringBitmapBytes),
}

impl IdAllocField {
    /// Get the discriminant byte for key serialization
    fn discriminant(&self) -> u8 {
        match self {
            Self::NextId(_) => 0,
            Self::FreeBitmap(_) => 1,
        }
    }
}

/// IdAlloc key: (embedding_code, field)
#[derive(Debug, Clone)]
pub(crate) struct IdAllocCfKey(pub(crate) EmbeddingCode, pub(crate) IdAllocField);

impl IdAllocCfKey {
    /// Create key for next_id field
    pub fn next_id(embedding_code: EmbeddingCode) -> Self {
        Self(embedding_code, IdAllocField::NextId(0))
    }

    /// Create key for free_bitmap field
    pub fn free_bitmap(embedding_code: EmbeddingCode) -> Self {
        Self(embedding_code, IdAllocField::FreeBitmap(Vec::new()))
    }
}

/// Type alias distinguishing "field as actual value" from "field as key discriminant"
pub(crate) type IdAllocValue = IdAllocField;

/// IdAlloc value: wraps the field enum with actual data
#[derive(Debug, Clone)]
pub(crate) struct IdAllocCfValue(pub(crate) IdAllocValue);

impl ColumnFamily for IdAlloc {
    const CF_NAME: &'static str = "vector/id_alloc";
}

impl ColumnFamilyConfig<VectorBlockCacheConfig> for IdAlloc {
    fn cf_options(cache: &rocksdb::Cache, config: &VectorBlockCacheConfig) -> rocksdb::Options {
        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();

        block_opts.set_block_cache(cache);
        block_opts.set_block_size(config.default_block_size);
        opts.set_block_based_table_factory(&block_opts);
        opts
    }
}

impl IdAlloc {
    pub fn key_to_bytes(key: &IdAllocCfKey) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(9);
        bytes.extend_from_slice(&key.0.to_be_bytes());
        bytes.push(key.1.discriminant()); // Only discriminant, not inner value
        bytes
    }

    pub fn key_from_bytes(bytes: &[u8]) -> Result<IdAllocCfKey> {
        if bytes.len() != 9 {
            anyhow::bail!(
                "Invalid IdAllocCfKey length: expected 9, got {}",
                bytes.len()
            );
        }
        let embedding_code = u64::from_be_bytes(bytes[0..8].try_into()?);
        let discriminant = bytes[8];
        // Create field with placeholder value based on discriminant
        let field = match discriminant {
            0 => IdAllocField::NextId(0),
            1 => IdAllocField::FreeBitmap(Vec::new()),
            _ => anyhow::bail!("Unknown IdAlloc discriminant: {}", discriminant),
        };
        Ok(IdAllocCfKey(embedding_code, field))
    }

    /// Serialize value based on variant type
    pub fn value_to_bytes(value: &IdAllocCfValue) -> Vec<u8> {
        match &value.0 {
            IdAllocField::NextId(v) => v.to_be_bytes().to_vec(),
            IdAllocField::FreeBitmap(v) => v.clone(),
        }
    }

    /// Deserialize value using key's field variant for type info
    pub fn value_from_bytes(key: &IdAllocCfKey, bytes: &[u8]) -> Result<IdAllocCfValue> {
        let field = match &key.1 {
            IdAllocField::NextId(_) => {
                if bytes.len() != 4 {
                    anyhow::bail!("Invalid NextId length: expected 4, got {}", bytes.len());
                }
                IdAllocField::NextId(u32::from_be_bytes(bytes.try_into()?))
            }
            IdAllocField::FreeBitmap(_) => {
                IdAllocField::FreeBitmap(bytes.to_vec())
            }
        };
        Ok(IdAllocCfValue(field))
    }
}

// ============================================================================
// Pending CF
// ============================================================================

/// Async updater pending queue.
///
/// Key: [embedding: u64] + [timestamp: u64] + [vec_id: u32] = 20 bytes
/// Value: empty (vector already stored in vectors CF)
pub(crate) struct Pending;

/// Pending key: (embedding_code, timestamp, vec_id)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub(crate) struct PendingCfKey(pub(crate) EmbeddingCode, pub(crate) TimestampMilli, pub(crate) VecId);

/// Pending value: empty (presence in CF indicates pending status)
#[derive(Debug, Clone)]
pub(crate) struct PendingCfValue(pub(crate) ());

impl ColumnFamily for Pending {
    const CF_NAME: &'static str = "vector/pending";
}

impl ColumnFamilyConfig<VectorBlockCacheConfig> for Pending {
    fn cf_options(cache: &rocksdb::Cache, config: &VectorBlockCacheConfig) -> rocksdb::Options {
        use rocksdb::SliceTransform;

        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();

        block_opts.set_block_cache(cache);
        block_opts.set_block_size(config.default_block_size);
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(8));
        opts.set_block_based_table_factory(&block_opts);
        opts
    }
}

impl Pending {
    pub fn key_to_bytes(key: &PendingCfKey) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(20);
        bytes.extend_from_slice(&key.0.to_be_bytes());
        bytes.extend_from_slice(&key.1.0.to_be_bytes());
        bytes.extend_from_slice(&key.2.to_be_bytes());
        bytes
    }

    pub fn key_from_bytes(bytes: &[u8]) -> Result<PendingCfKey> {
        if bytes.len() != 20 {
            anyhow::bail!(
                "Invalid PendingCfKey length: expected 20, got {}",
                bytes.len()
            );
        }
        let embedding_code = u64::from_be_bytes(bytes[0..8].try_into()?);
        let timestamp = TimestampMilli(u64::from_be_bytes(bytes[8..16].try_into()?));
        let vec_id = u32::from_be_bytes(bytes[16..20].try_into()?);
        Ok(PendingCfKey(embedding_code, timestamp, vec_id))
    }

    pub fn value_to_bytes(_value: &PendingCfValue) -> Vec<u8> {
        Vec::new()
    }

    pub fn value_from_bytes(_bytes: &[u8]) -> Result<PendingCfValue> {
        Ok(PendingCfValue(()))
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_cf_names_have_prefix() {
        for cf_name in ALL_COLUMN_FAMILIES {
            assert!(
                cf_name.starts_with("vector/"),
                "CF {} should have 'vector/' prefix",
                cf_name
            );
        }
    }

    #[test]
    fn test_vectors_key_roundtrip() {
        let key = VectorCfKey(42, 123);
        let bytes = Vectors::key_to_bytes(&key);
        assert_eq!(bytes.len(), 12);
        let parsed = Vectors::key_from_bytes(&bytes).unwrap();
        assert_eq!(parsed.0, 42);
        assert_eq!(parsed.1, 123);
    }

    #[test]
    fn test_vectors_value_roundtrip() {
        let vector = vec![1.0f32, 2.0, 3.0, 4.0];
        let value = VectorCfValue(vector.clone());
        let bytes = Vectors::value_to_bytes(&value);
        assert_eq!(bytes.len(), 16); // 4 floats * 4 bytes
        let parsed = Vectors::value_from_bytes(&bytes).unwrap();
        assert_eq!(parsed.0, vector);
    }

    #[test]
    fn test_edges_key_roundtrip() {
        let key = EdgeCfKey(1, 42, 3);
        let bytes = Edges::key_to_bytes(&key);
        assert_eq!(bytes.len(), 13);
        let parsed = Edges::key_from_bytes(&bytes).unwrap();
        assert_eq!(parsed.0, 1);
        assert_eq!(parsed.1, 42);
        assert_eq!(parsed.2, 3);
    }

    #[test]
    fn test_id_forward_key_roundtrip() {
        let ulid = Id::new();
        let key = IdForwardCfKey(5, ulid);
        let bytes = IdForward::key_to_bytes(&key);
        assert_eq!(bytes.len(), 24);
        let parsed = IdForward::key_from_bytes(&bytes).unwrap();
        assert_eq!(parsed.0, 5);
        assert_eq!(parsed.1, ulid);
    }

    #[test]
    fn test_pending_key_roundtrip() {
        use crate::TimestampMilli;
        let key = PendingCfKey(1, TimestampMilli(1000), 42);
        let bytes = Pending::key_to_bytes(&key);
        assert_eq!(bytes.len(), 20);
        let parsed = Pending::key_from_bytes(&bytes).unwrap();
        assert_eq!(parsed.0, 1);
        assert_eq!(parsed.1, TimestampMilli(1000));
        assert_eq!(parsed.2, 42);
    }

}
