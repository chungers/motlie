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

use crate::rocksdb::{ColumnFamily, ColumnFamilyConfig, ColumnFamilySerde, HotColumnFamilyRecord};
use crate::{Id, TimestampMilli};
use anyhow::Result;
use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
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
pub type EmbeddingCode = u64;

/// Internal vector identifier within an embedding space.
///
/// Compact u32 index used for HNSW graph edges and storage.
/// Mapped bidirectionally to external ULIDs via IdForward/IdReverse CFs.
pub type VecId = u32;

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
/// Type alias for RaBitQ binary code
pub type RabitqCode = Vec<u8>;

/// ADC (Asymmetric Distance Computation) corrective factors.
///
/// Stores per-vector correction data needed for accurate distance estimation
/// in RaBitQ ADC mode. Unlike symmetric Hamming, ADC keeps the query as float32
/// and uses these factors to correct the binary approximation.
///
/// # Storage
///
/// 8 bytes per vector (2 × f32).
///
/// # Formula
///
/// Distance estimation uses:
/// ```text
/// ⟨q, v⟩_estimated = binary_dot(q_rotated, v_code) / quantization_error
/// ```
///
/// See `RABITQ.md` for detailed explanation of why ADC solves the multi-bit problem.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AdcCorrection {
    /// L2 norm of the vector: ||v||
    ///
    /// For normalized unit vectors this is ~1.0, but we store it for generality
    /// and to handle vectors that aren't perfectly normalized.
    pub vector_norm: f32,

    /// Quantization error correction: ⟨v_quantized_decoded, v_normalized⟩
    ///
    /// This captures how much the quantized representation deviates from the
    /// original vector. Used to unbias the inner product estimate.
    pub quantization_error: f32,
}

impl AdcCorrection {
    /// Create a new ADC correction.
    #[inline]
    pub fn new(vector_norm: f32, quantization_error: f32) -> Self {
        Self {
            vector_norm,
            quantization_error,
        }
    }
}

/// Vector element type.
///
/// Controls how vector elements are serialized in the Vectors CF.
/// All computation is done in f32 regardless of element type.
///
/// # Storage Savings
///
/// | Type | Bytes/Element | 512D Vector | Savings |
/// |------|---------------|-------------|---------|
/// | F32  | 4 bytes       | 2,048 bytes | -       |
/// | F16  | 2 bytes       | 1,024 bytes | **50%** |
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub enum VectorElementType {
    /// 32-bit IEEE 754 floating point (default, full precision)
    #[default]
    F32,
    /// 16-bit IEEE 754 half-precision (50% smaller, ~0.1% precision loss for normalized vectors)
    F16,
}

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
    LifecycleCounts::CF_NAME,
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
pub struct EmbeddingSpecs;

/// Domain struct for embedding specification.
///
/// Rich struct with >2 fields, so defined separately per naming convention.
/// This is the **single source of truth** for how to build and search an embedding space.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmbeddingSpec {
    /// Model name (e.g., "gemma", "qwen3", "ada-002")
    pub model: String,
    /// Vector dimensionality (e.g., 128, 768, 1536)
    pub dim: u32,
    /// Distance metric for similarity computation
    pub distance: Distance,
    /// Storage type for vector elements (default: F32 for backward compatibility)
    #[serde(default)]
    pub storage_type: VectorElementType,

    // =========================================================================
    // Build Parameters (Phase 5.6 - Config Persistence)
    // =========================================================================

    /// HNSW M parameter: number of bidirectional links per node.
    /// Higher M = better recall, more memory. Default: 16
    #[serde(default = "default_hnsw_m")]
    pub hnsw_m: u16,

    /// HNSW ef_construction: search beam width during index build.
    /// Higher = better graph quality, slower build. Default: 200
    #[serde(default = "default_hnsw_ef_construction")]
    pub hnsw_ef_construction: u16,

    /// RaBitQ bits per dimension for binary quantization.
    /// 1 = 32x compression, 2 = 16x, 4 = 8x. Default: 1
    #[serde(default = "default_rabitq_bits")]
    pub rabitq_bits: u8,

    /// RaBitQ rotation matrix seed for deterministic encoding.
    /// Same seed = same rotation matrix. Default: 42
    #[serde(default = "default_rabitq_seed")]
    pub rabitq_seed: u64,
}

// Serde default functions for backward compatibility
fn default_hnsw_m() -> u16 {
    16
}
fn default_hnsw_ef_construction() -> u16 {
    200
}
fn default_rabitq_bits() -> u8 {
    1
}
fn default_rabitq_seed() -> u64 {
    42
}

impl EmbeddingSpec {
    /// Compute a deterministic hash of all build-affecting parameters.
    ///
    /// This hash is stored in GraphMeta::SpecHash on first index build and validated
    /// on subsequent inserts and searches to detect configuration drift.
    ///
    /// Includes: model, dim, distance, storage_type, hnsw_m, hnsw_ef_construction,
    /// rabitq_bits, rabitq_seed
    pub fn compute_spec_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.model.hash(&mut hasher);
        self.dim.hash(&mut hasher);
        self.distance.hash(&mut hasher);
        self.storage_type.hash(&mut hasher);
        self.hnsw_m.hash(&mut hasher);
        self.hnsw_ef_construction.hash(&mut hasher);
        self.rabitq_bits.hash(&mut hasher);
        self.rabitq_seed.hash(&mut hasher);
        hasher.finish()
    }
}

/// EmbeddingSpecs key: the embedding space identifier (primary key)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmbeddingSpecCfKey(pub EmbeddingCode);

/// EmbeddingSpecs value: wraps EmbeddingSpec
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmbeddingSpecCfValue(pub EmbeddingSpec);

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

impl ColumnFamilySerde for EmbeddingSpecs {
    type Key = EmbeddingSpecCfKey;
    type Value = EmbeddingSpecCfValue;

    fn key_to_bytes(key: &Self::Key) -> Vec<u8> {
        key.0.to_be_bytes().to_vec()
    }

    fn key_from_bytes(bytes: &[u8]) -> Result<Self::Key> {
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

    // value_to_bytes, value_from_bytes use default impl (MessagePack + LZ4)
}

// ============================================================================
// Vectors CF
// ============================================================================

/// Vectors column family - raw f32 vector storage.
///
/// Key: [embedding: u64] + [vec_id: u32] = 12 bytes
/// Value: f32[dim] as raw bytes (e.g., 512 bytes for 128D)
pub struct Vectors;

/// Vectors key: (embedding_code, vec_id)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct VectorCfKey(pub EmbeddingCode, pub VecId);

/// Vectors value: raw f32 vector data
#[derive(Debug, Clone)]
pub struct VectorCfValue(pub Vec<f32>);

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

    /// Store vector as raw f32 bytes (backward compatible, uses F32 storage)
    pub fn value_to_bytes(value: &VectorCfValue) -> Vec<u8> {
        Self::value_to_bytes_typed(&value.0, VectorElementType::F32)
    }

    /// Load vector from raw f32 bytes (backward compatible, assumes F32 storage)
    pub fn value_from_bytes(bytes: &[u8]) -> Result<VectorCfValue> {
        Self::value_from_bytes_typed(bytes, VectorElementType::F32).map(VectorCfValue)
    }

    /// Store vector with specified storage type.
    ///
    /// # Arguments
    /// * `value` - Vector as f32 slice (source is always f32)
    /// * `storage_type` - Target storage format (F32 or F16)
    ///
    /// # Returns
    /// Serialized bytes in the specified format
    pub fn value_to_bytes_typed(value: &[f32], storage_type: VectorElementType) -> Vec<u8> {
        match storage_type {
            VectorElementType::F32 => {
                let mut bytes = Vec::with_capacity(value.len() * 4);
                for &v in value {
                    bytes.extend_from_slice(&v.to_le_bytes());
                }
                bytes
            }
            VectorElementType::F16 => {
                use half::f16;
                let mut bytes = Vec::with_capacity(value.len() * 2);
                for &v in value {
                    let h = f16::from_f32(v);
                    bytes.extend_from_slice(&h.to_le_bytes());
                }
                bytes
            }
        }
    }

    /// Load vector from bytes with specified storage type.
    ///
    /// # Arguments
    /// * `bytes` - Serialized vector bytes
    /// * `storage_type` - Storage format used when serializing
    ///
    /// # Returns
    /// Vector as Vec<f32> (always returns f32 for computation)
    pub fn value_from_bytes_typed(
        bytes: &[u8],
        storage_type: VectorElementType,
    ) -> Result<Vec<f32>> {
        match storage_type {
            VectorElementType::F32 => {
                if bytes.len() % 4 != 0 {
                    anyhow::bail!(
                        "Invalid F32 vector bytes length: {} is not divisible by 4",
                        bytes.len()
                    );
                }
                let mut vector = Vec::with_capacity(bytes.len() / 4);
                for chunk in bytes.chunks_exact(4) {
                    vector.push(f32::from_le_bytes(chunk.try_into()?));
                }
                Ok(vector)
            }
            VectorElementType::F16 => {
                use half::f16;
                if bytes.len() % 2 != 0 {
                    anyhow::bail!(
                        "Invalid F16 vector bytes length: {} is not divisible by 2",
                        bytes.len()
                    );
                }
                let mut vector = Vec::with_capacity(bytes.len() / 2);
                for chunk in bytes.chunks_exact(2) {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    vector.push(f16::from_bits(bits).to_f32());
                }
                Ok(vector)
            }
        }
    }

    /// Calculate storage size in bytes for a vector of given dimension.
    pub fn storage_size(dim: usize, storage_type: VectorElementType) -> usize {
        match storage_type {
            VectorElementType::F32 => dim * 4,
            VectorElementType::F16 => dim * 2,
        }
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
pub(crate) struct EdgeCfKey(
    pub(crate) EmbeddingCode,
    pub(crate) VecId,
    pub(crate) HnswLayer,
);

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

        // Enable bloom filter for fast point lookups (10 bits per key)
        // This is critical for HNSW get_neighbors performance
        block_opts.set_bloom_filter(10.0, false);

        opts.set_compression_type(rocksdb::DBCompressionType::None);
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(12));
        opts.set_block_based_table_factory(&block_opts);

        // Configure merge operator for concurrent edge updates
        opts.set_merge_operator_associative(
            super::merge::EDGE_MERGE_OPERATOR_NAME,
            super::merge::edge_full_merge,
        );

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
            anyhow::bail!("Invalid EdgeCfKey length: expected 13, got {}", bytes.len());
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

/// Binary codes column family - RaBitQ quantized vectors with ADC corrections.
///
/// Key: [embedding: u64] + [vec_id: u32] = 12 bytes
/// Value: [binary_code: N bytes] + [vector_norm: f32] + [quantization_error: f32]
///
/// The ADC correction factors (8 bytes) enable high-recall search:
/// - Hamming distance: ~24% recall (multi-bit broken due to Gray code wraparound)
/// - ADC distance: ~92% recall (preserves numeric ordering)
pub struct BinaryCodes;

/// BinaryCodes key: (embedding_code, vec_id)
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BinaryCodeCfKey(pub EmbeddingCode, pub VecId);

/// BinaryCodes value: RaBitQ quantized code with ADC correction factors.
#[derive(Debug, Clone)]
pub struct BinaryCodeCfValue {
    /// Binary quantized code
    pub code: RabitqCode,
    /// ADC correction factors for distance estimation
    pub correction: AdcCorrection,
}

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

    /// Serialize binary code value with ADC correction.
    ///
    /// Format: [code bytes][vector_norm: f32 LE][quantization_error: f32 LE]
    pub fn value_to_bytes(value: &BinaryCodeCfValue) -> Vec<u8> {
        let mut bytes = value.code.clone();
        bytes.extend_from_slice(&value.correction.vector_norm.to_le_bytes());
        bytes.extend_from_slice(&value.correction.quantization_error.to_le_bytes());
        bytes
    }

    /// Deserialize binary code value with ADC correction.
    pub fn value_from_bytes(bytes: &[u8]) -> Result<BinaryCodeCfValue> {
        if bytes.len() < 8 {
            anyhow::bail!(
                "Invalid BinaryCodeCfValue length: expected >= 8, got {}",
                bytes.len()
            );
        }
        let code_len = bytes.len() - 8;
        let code = bytes[..code_len].to_vec();
        let norm = f32::from_le_bytes(bytes[code_len..code_len + 4].try_into()?);
        let qerr = f32::from_le_bytes(bytes[code_len + 4..].try_into()?);
        Ok(BinaryCodeCfValue {
            code,
            correction: AdcCorrection::new(norm, qerr),
        })
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
// ============================================================================
// VecMetadata Flags Design
// ============================================================================
//
// The `flags` field in VecMetadata uses a hybrid enum + bitmask design:
//
// ```
// flags byte: 0bRRRRRRLL
//             ││││││└┴── Lifecycle (2 bits): mutually exclusive states
//             │││││└──── Reserved (bit 2)
//             ││││└───── Reserved (bit 3)
//             │││└────── Reserved (bit 4)
//             ││└─────── Reserved (bit 5)
//             │└──────── Reserved (bit 6)
//             └───────── REPLICATED (bit 7) - FUTURE: not yet implemented
// ```
//
// ## Lifecycle States (bits 0-1, mutually exclusive)
//
// The lower 2 bits represent the vector's lifecycle state as a type-safe enum:
// - 0b00 = Indexed: fully in HNSW graph, searchable
// - 0b01 = Deleted: soft-deleted, skip in search, awaiting cleanup
// - 0b10 = Pending: stored but awaiting async graph construction
// - 0b11 = PendingDeleted: deleted before graph construction completed
//
// ## Orthogonal Flags (bits 2-7, independent)
//
// Upper bits are reserved for future orthogonal capabilities that can be
// combined with any lifecycle state:
// - Bit 7 (REPLICATED): FUTURE capability for distributed replication.
//   When implemented, indicates vector data has been replicated to another
//   node in a distributed deployment. NOT YET IMPLEMENTED - reserved only.
// - Bits 2-6: Reserved for future use.
//
// ## Backward Compatibility
//
// This design maintains wire compatibility with the original FLAG_DELETED=0x01
// and FLAG_PENDING=0x02 constants:
// - Old Indexed (flags=0x00) → VecLifecycle::Indexed (0b00)
// - Old Deleted (flags=0x01) → VecLifecycle::Deleted (0b01)
// - Old Pending (flags=0x02) → VecLifecycle::Pending (0b10)
// - Old Pending+Deleted (flags=0x03) → VecLifecycle::PendingDeleted (0b11)
//
// ============================================================================

/// Vector lifecycle states (mutually exclusive, occupies lower 2 bits of flags).
///
/// Represents the vector's position in its lifecycle state machine:
/// ```text
/// Insert(build_index=false) ──► Pending ──► Indexed
///                                  │            │
///                                  ▼            ▼
///                            PendingDeleted  Deleted
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum VecLifecycle {
    /// Vector is fully indexed in the HNSW graph and searchable.
    Indexed = 0b00,
    /// Vector has been soft-deleted. Skip in search, awaiting cleanup.
    Deleted = 0b01,
    /// Vector is stored but awaiting async HNSW graph construction.
    /// Search should include via brute-force fallback if needed.
    Pending = 0b10,
    /// Vector was deleted before async graph construction completed.
    /// Still needs removal from Pending queue, but skip in search.
    PendingDeleted = 0b11,
}

impl VecLifecycle {
    /// Bitmask for lifecycle bits (lower 2 bits).
    const MASK: u8 = 0b0000_0011;

    /// Extract lifecycle state from flags byte.
    #[inline]
    fn from_flags(flags: u8) -> Self {
        match flags & Self::MASK {
            0b00 => Self::Indexed,
            0b01 => Self::Deleted,
            0b10 => Self::Pending,
            0b11 => Self::PendingDeleted,
            _ => unreachable!(), // Only 4 possible values with 2-bit mask
        }
    }

    /// Apply this lifecycle state to a flags byte, preserving orthogonal flags.
    #[inline]
    fn apply_to_flags(self, flags: u8) -> u8 {
        (flags & !Self::MASK) | (self as u8)
    }
}

/// Orthogonal flags for future capabilities (upper bits of flags byte).
///
/// These flags are independent of lifecycle state and can be combined with
/// any VecLifecycle value.
#[allow(non_snake_case)]
pub(crate) mod VecFlags {
    /// FUTURE CAPABILITY - NOT YET IMPLEMENTED.
    ///
    /// When implemented, this flag will indicate that the vector data has been
    /// replicated to another node in a distributed deployment. This enables:
    /// - Distributed search across multiple nodes
    /// - Fault tolerance through data redundancy
    /// - Geographic distribution for latency optimization
    ///
    /// Bit 7 is chosen to leave room for other orthogonal flags in bits 2-6.
    ///
    /// Usage (when implemented):
    /// ```ignore
    /// meta.set_replicated(true);  // Mark as replicated
    /// if meta.is_replicated() { ... }  // Check replication status
    /// ```
    #[allow(dead_code)]
    pub const REPLICATED: u8 = 0b1000_0000; // bit 7
}

/// Vector metadata stored in VecMeta column family.
///
/// Rich struct with >2 fields, so defined separately per naming convention.
/// Uses rkyv for zero-copy access during HNSW search (hot path reads max_layer).
///
/// See module-level documentation for flags byte layout.
#[derive(Archive, RkyvDeserialize, RkyvSerialize, Debug, Clone)]
#[archive(check_bytes)]
pub(crate) struct VecMetadata {
    /// Maximum HNSW layer this vector appears in (0 if pending).
    pub(crate) max_layer: u8,
    /// Combined lifecycle state (bits 0-1) and orthogonal flags (bits 2-7).
    /// Use `lifecycle()`, `is_pending()`, etc. instead of direct access.
    flags: u8,
    /// Timestamp when vector was created (millis since epoch).
    pub(crate) created_at: u64,
}

impl VecMetadata {
    /// Create metadata with specified max_layer and raw flags.
    fn new(max_layer: u8, flags: u8) -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        Self {
            max_layer,
            flags,
            created_at,
        }
    }

    /// Create metadata for a pending vector (awaiting async graph construction).
    pub(crate) fn pending() -> Self {
        Self::new(0, VecLifecycle::Pending as u8)
    }

    /// Create metadata for an indexed vector (in HNSW graph).
    pub(crate) fn indexed(max_layer: u8) -> Self {
        Self::new(max_layer, VecLifecycle::Indexed as u8)
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Lifecycle State (mutually exclusive, type-safe enum)
    // ─────────────────────────────────────────────────────────────────────────

    /// Get the current lifecycle state as a type-safe enum.
    #[inline]
    pub(crate) fn lifecycle(&self) -> VecLifecycle {
        VecLifecycle::from_flags(self.flags)
    }

    /// Set lifecycle state, preserving any orthogonal flags.
    #[inline]
    pub(crate) fn set_lifecycle(&mut self, state: VecLifecycle) {
        self.flags = state.apply_to_flags(self.flags);
    }

    /// Check if vector is pending graph construction.
    ///
    /// Returns true for both `Pending` and `PendingDeleted` states,
    /// as both require processing by the async graph updater.
    #[inline]
    pub(crate) fn is_pending(&self) -> bool {
        matches!(
            self.lifecycle(),
            VecLifecycle::Pending | VecLifecycle::PendingDeleted
        )
    }

    /// Check if vector is deleted (should skip in search).
    ///
    /// Returns true for both `Deleted` and `PendingDeleted` states.
    #[inline]
    pub(crate) fn is_deleted(&self) -> bool {
        matches!(
            self.lifecycle(),
            VecLifecycle::Deleted | VecLifecycle::PendingDeleted
        )
    }

    /// Check if vector is fully indexed and active.
    #[inline]
    pub(crate) fn is_indexed(&self) -> bool {
        self.lifecycle() == VecLifecycle::Indexed
    }

    /// Mark as deleted, handling state transition correctly.
    ///
    /// State transitions:
    /// - Indexed → Deleted
    /// - Pending → PendingDeleted (still needs async updater cleanup)
    /// - Deleted/PendingDeleted → unchanged
    pub(crate) fn set_deleted(&mut self) {
        let new_state = match self.lifecycle() {
            VecLifecycle::Indexed => VecLifecycle::Deleted,
            VecLifecycle::Pending => VecLifecycle::PendingDeleted,
            other => other, // Already deleted
        };
        self.set_lifecycle(new_state);
    }

    /// Clear pending state (after graph construction completes).
    ///
    /// State transitions:
    /// - Pending → Indexed
    /// - PendingDeleted → Deleted
    /// - Indexed/Deleted → unchanged
    pub(crate) fn clear_pending(&mut self) {
        let new_state = match self.lifecycle() {
            VecLifecycle::Pending => VecLifecycle::Indexed,
            VecLifecycle::PendingDeleted => VecLifecycle::Deleted,
            other => other,
        };
        self.set_lifecycle(new_state);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Orthogonal Flags (FUTURE - not yet implemented)
    // ─────────────────────────────────────────────────────────────────────────

    /// FUTURE CAPABILITY - NOT YET IMPLEMENTED.
    ///
    /// Check if vector has been replicated to another node.
    /// Always returns false until distributed replication is implemented.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn is_replicated(&self) -> bool {
        self.flags & VecFlags::REPLICATED != 0
    }

    /// FUTURE CAPABILITY - NOT YET IMPLEMENTED.
    ///
    /// Set the replicated flag. Has no effect until distributed
    /// replication is implemented.
    #[allow(dead_code)]
    pub(crate) fn set_replicated(&mut self, replicated: bool) {
        if replicated {
            self.flags |= VecFlags::REPLICATED;
        } else {
            self.flags &= !VecFlags::REPLICATED;
        }
    }
}

impl ArchivedVecMetadata {
    /// Get the current lifecycle state (zero-copy from archived data).
    #[inline]
    pub(crate) fn lifecycle(&self) -> VecLifecycle {
        VecLifecycle::from_flags(self.flags)
    }

    /// Check if vector is deleted (zero-copy from archived data).
    #[inline]
    pub(crate) fn is_deleted(&self) -> bool {
        matches!(
            self.lifecycle(),
            VecLifecycle::Deleted | VecLifecycle::PendingDeleted
        )
    }

    /// Check if vector is pending graph construction (zero-copy from archived data).
    #[inline]
    pub(crate) fn is_pending(&self) -> bool {
        matches!(
            self.lifecycle(),
            VecLifecycle::Pending | VecLifecycle::PendingDeleted
        )
    }
}

/// VecMeta key: (embedding_code, vec_id)
#[derive(Debug, Clone)]
pub(crate) struct VecMetaCfKey(pub(crate) EmbeddingCode, pub(crate) VecId);

/// VecMeta value: wraps VecMetadata (rkyv-serialized)
#[derive(Archive, RkyvDeserialize, RkyvSerialize, Debug, Clone)]
#[archive(check_bytes)]
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

impl HotColumnFamilyRecord for VecMeta {
    type Key = VecMetaCfKey;
    type Value = VecMetaCfValue;

    fn key_to_bytes(key: &Self::Key) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(12);
        bytes.extend_from_slice(&key.0.to_be_bytes());
        bytes.extend_from_slice(&key.1.to_be_bytes());
        bytes
    }

    fn key_from_bytes(bytes: &[u8]) -> Result<Self::Key> {
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

    // value_to_bytes, value_from_bytes, value_archived provided by trait default impl
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
    /// Serialized HNSW configuration (deprecated - use EmbeddingSpec)
    Config(Vec<u8>),
    /// Hash of EmbeddingSpec at index build time (for drift detection)
    /// Set on first insert, validated on subsequent inserts and searches
    SpecHash(u64),
}

impl GraphMetaField {
    /// Get the discriminant byte for key serialization
    fn discriminant(&self) -> u8 {
        match self {
            Self::EntryPoint(_) => 0,
            Self::MaxLevel(_) => 1,
            Self::Count(_) => 2,
            Self::Config(_) => 3,
            Self::SpecHash(_) => 4,
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

    /// Create key for spec_hash field
    pub fn spec_hash(embedding_code: EmbeddingCode) -> Self {
        Self(embedding_code, GraphMetaField::SpecHash(0))
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
            4 => GraphMetaField::SpecHash(0),
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
            GraphMetaField::SpecHash(v) => v.to_be_bytes().to_vec(),
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
            GraphMetaField::Config(_) => GraphMetaField::Config(bytes.to_vec()),
            GraphMetaField::SpecHash(_) => {
                if bytes.len() != 8 {
                    anyhow::bail!("Invalid SpecHash length: expected 8, got {}", bytes.len());
                }
                GraphMetaField::SpecHash(u64::from_be_bytes(bytes.try_into()?))
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
            IdAllocField::FreeBitmap(_) => IdAllocField::FreeBitmap(bytes.to_vec()),
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
pub(crate) struct PendingCfKey(
    pub(crate) EmbeddingCode,
    pub(crate) TimestampMilli,
    pub(crate) VecId,
);

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
    /// Create a pending key with the current timestamp.
    ///
    /// Keys are ordered by (embedding, timestamp, vec_id) for FIFO processing
    /// within each embedding. Use `prefix_for_embedding()` for iteration.
    pub fn key_now(embedding: EmbeddingCode, vec_id: VecId) -> PendingCfKey {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        PendingCfKey(embedding, TimestampMilli(timestamp), vec_id)
    }

    /// Get the 8-byte prefix for iterating all pending items for an embedding.
    ///
    /// Use with RocksDB prefix iterator to scan pending queue per embedding.
    pub fn prefix_for_embedding(embedding: EmbeddingCode) -> [u8; 8] {
        embedding.to_be_bytes()
    }

    pub fn key_to_bytes(key: &PendingCfKey) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(20);
        bytes.extend_from_slice(&key.0.to_be_bytes());
        bytes.extend_from_slice(&key.1 .0.to_be_bytes());
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

// ============================================================================
// LifecycleCounts CF
// ============================================================================

/// Lifecycle counters for O(1) stats queries.
///
/// This CF uses a merge operator for atomic counter updates, avoiding
/// read-modify-write cycles during mutations. Counters track the number
/// of vectors in each lifecycle state per embedding.
///
/// Key: [embedding: u64] = 8 bytes
/// Value: [indexed: u64] + [pending: u64] + [deleted: u64] + [pending_deleted: u64] = 32 bytes
///
/// Values are merged using signed deltas during mutations and resolved
/// to absolute counts during reads/compaction.
pub struct LifecycleCounts;

/// Lifecycle counts key: embedding_code
#[derive(Debug, Clone)]
pub struct LifecycleCountsCfKey(pub EmbeddingCode);

/// Lifecycle counts value: counters for each lifecycle state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LifecycleCountsValue {
    pub indexed: u64,
    pub pending: u64,
    pub deleted: u64,
    pub pending_deleted: u64,
}

/// Lifecycle counts value wrapper.
#[derive(Debug, Clone)]
pub struct LifecycleCountsCfValue(pub LifecycleCountsValue);

/// Delta operation for lifecycle counter updates via merge operator.
///
/// Each field represents a signed delta to apply to the corresponding counter.
/// Positive values increment, negative values decrement (clamped to 0).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LifecycleCountsDelta {
    pub indexed: i64,
    pub pending: i64,
    pub deleted: i64,
    pub pending_deleted: i64,
}

impl LifecycleCountsDelta {
    /// Create a delta that increments the pending counter.
    pub fn inc_pending() -> Self {
        Self { pending: 1, ..Default::default() }
    }

    /// Create a delta that increments the indexed counter.
    pub fn inc_indexed() -> Self {
        Self { indexed: 1, ..Default::default() }
    }

    /// Create a delta for transitioning from pending to indexed.
    pub fn pending_to_indexed() -> Self {
        Self { pending: -1, indexed: 1, ..Default::default() }
    }

    /// Create a delta for marking an indexed vector as deleted.
    pub fn indexed_to_deleted() -> Self {
        Self { indexed: -1, deleted: 1, ..Default::default() }
    }

    /// Create a delta for marking a pending vector as pending_deleted.
    pub fn pending_to_pending_deleted() -> Self {
        Self { pending: -1, pending_deleted: 1, ..Default::default() }
    }

    /// Create a delta for purging a deleted vector (GC).
    pub fn purge_deleted() -> Self {
        Self { deleted: -1, ..Default::default() }
    }

    /// Create a delta for purging a pending_deleted vector (GC).
    pub fn purge_pending_deleted() -> Self {
        Self { pending_deleted: -1, ..Default::default() }
    }

    /// Serialize to bytes for use as merge operand.
    pub fn to_bytes(&self) -> Vec<u8> {
        rmp_serde::to_vec(self).expect("LifecycleCountsDelta serialization should never fail")
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        rmp_serde::from_slice(bytes).map_err(|e| anyhow::anyhow!("Failed to deserialize delta: {}", e))
    }
}

impl LifecycleCountsValue {
    /// Apply a delta to this counter set, clamping to 0.
    pub fn apply_delta(&mut self, delta: &LifecycleCountsDelta) {
        self.indexed = (self.indexed as i64 + delta.indexed).max(0) as u64;
        self.pending = (self.pending as i64 + delta.pending).max(0) as u64;
        self.deleted = (self.deleted as i64 + delta.deleted).max(0) as u64;
        self.pending_deleted = (self.pending_deleted as i64 + delta.pending_deleted).max(0) as u64;
    }

    /// Serialize to bytes for storage.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(32);
        bytes.extend_from_slice(&self.indexed.to_be_bytes());
        bytes.extend_from_slice(&self.pending.to_be_bytes());
        bytes.extend_from_slice(&self.deleted.to_be_bytes());
        bytes.extend_from_slice(&self.pending_deleted.to_be_bytes());
        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != 32 {
            anyhow::bail!("Invalid LifecycleCountsValue length: expected 32, got {}", bytes.len());
        }
        Ok(Self {
            indexed: u64::from_be_bytes(bytes[0..8].try_into()?),
            pending: u64::from_be_bytes(bytes[8..16].try_into()?),
            deleted: u64::from_be_bytes(bytes[16..24].try_into()?),
            pending_deleted: u64::from_be_bytes(bytes[24..32].try_into()?),
        })
    }
}

impl ColumnFamily for LifecycleCounts {
    const CF_NAME: &'static str = "vector/lifecycle_counts";
}

impl ColumnFamilyConfig<VectorBlockCacheConfig> for LifecycleCounts {
    fn cf_options(cache: &rocksdb::Cache, config: &VectorBlockCacheConfig) -> rocksdb::Options {
        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();

        block_opts.set_block_cache(cache);
        block_opts.set_block_size(config.default_block_size);
        opts.set_block_based_table_factory(&block_opts);

        // Configure merge operator for atomic counter updates
        opts.set_merge_operator_associative(
            super::merge::LIFECYCLE_MERGE_OPERATOR_NAME,
            super::merge::lifecycle_full_merge,
        );

        opts
    }
}

impl LifecycleCounts {
    pub fn key_to_bytes(key: &LifecycleCountsCfKey) -> Vec<u8> {
        key.0.to_be_bytes().to_vec()
    }

    pub fn key_from_bytes(bytes: &[u8]) -> Result<LifecycleCountsCfKey> {
        if bytes.len() != 8 {
            anyhow::bail!("Invalid LifecycleCountsCfKey length: expected 8, got {}", bytes.len());
        }
        Ok(LifecycleCountsCfKey(u64::from_be_bytes(bytes.try_into()?)))
    }

    pub fn value_to_bytes(value: &LifecycleCountsCfValue) -> Vec<u8> {
        value.0.to_bytes()
    }

    pub fn value_from_bytes(bytes: &[u8]) -> Result<LifecycleCountsCfValue> {
        Ok(LifecycleCountsCfValue(LifecycleCountsValue::from_bytes(bytes)?))
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

    #[test]
    fn test_pending_key_now() {
        let embedding = 42u64;
        let vec_id = 100u32;

        let key1 = Pending::key_now(embedding, vec_id);
        std::thread::sleep(std::time::Duration::from_millis(2));
        let key2 = Pending::key_now(embedding, vec_id + 1);

        // Same embedding
        assert_eq!(key1.0, embedding);
        assert_eq!(key2.0, embedding);

        // Timestamps should be increasing (FIFO order)
        assert!(key2.1 .0 >= key1.1 .0, "timestamps should be monotonic");

        // Vec IDs preserved
        assert_eq!(key1.2, vec_id);
        assert_eq!(key2.2, vec_id + 1);
    }

    #[test]
    fn test_pending_prefix_for_embedding() {
        let embedding = 0x0102030405060708u64;
        let prefix = Pending::prefix_for_embedding(embedding);

        // Prefix should be big-endian embedding bytes
        assert_eq!(prefix, [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);

        // Key should start with this prefix
        let key = Pending::key_now(embedding, 42);
        let key_bytes = Pending::key_to_bytes(&key);
        assert_eq!(&key_bytes[0..8], &prefix);
    }

    // ========================================================================
    // VecLifecycle and VecMetadata Tests
    // ========================================================================

    #[test]
    fn test_vec_lifecycle_from_flags() {
        // Verify backward compatibility with old FLAG_* constants
        assert_eq!(VecLifecycle::from_flags(0b00), VecLifecycle::Indexed);
        assert_eq!(VecLifecycle::from_flags(0b01), VecLifecycle::Deleted);
        assert_eq!(VecLifecycle::from_flags(0b10), VecLifecycle::Pending);
        assert_eq!(VecLifecycle::from_flags(0b11), VecLifecycle::PendingDeleted);

        // Orthogonal flags in upper bits should not affect lifecycle
        assert_eq!(VecLifecycle::from_flags(0b1000_0000), VecLifecycle::Indexed);
        assert_eq!(VecLifecycle::from_flags(0b1000_0001), VecLifecycle::Deleted);
        assert_eq!(VecLifecycle::from_flags(0b1000_0010), VecLifecycle::Pending);
        assert_eq!(VecLifecycle::from_flags(0b1000_0011), VecLifecycle::PendingDeleted);
    }

    #[test]
    fn test_vec_lifecycle_apply_to_flags() {
        // Should preserve upper bits when changing lifecycle
        let flags_with_replicated = 0b1000_0010; // Pending + REPLICATED
        let new_flags = VecLifecycle::Indexed.apply_to_flags(flags_with_replicated);
        assert_eq!(new_flags, 0b1000_0000); // Indexed + REPLICATED preserved
    }

    #[test]
    fn test_vec_metadata_pending() {
        let meta = VecMetadata::pending();
        assert_eq!(meta.lifecycle(), VecLifecycle::Pending);
        assert!(meta.is_pending());
        assert!(!meta.is_deleted());
        assert!(!meta.is_indexed());
        assert_eq!(meta.max_layer, 0);
    }

    #[test]
    fn test_vec_metadata_indexed() {
        let meta = VecMetadata::indexed(5);
        assert_eq!(meta.lifecycle(), VecLifecycle::Indexed);
        assert!(!meta.is_pending());
        assert!(!meta.is_deleted());
        assert!(meta.is_indexed());
        assert_eq!(meta.max_layer, 5);
    }

    #[test]
    fn test_vec_metadata_state_transitions() {
        // Pending → Indexed (normal async completion)
        let mut meta = VecMetadata::pending();
        meta.clear_pending();
        assert_eq!(meta.lifecycle(), VecLifecycle::Indexed);

        // Indexed → Deleted (normal delete)
        meta.set_deleted();
        assert_eq!(meta.lifecycle(), VecLifecycle::Deleted);

        // Pending → PendingDeleted (delete before async completion)
        let mut meta2 = VecMetadata::pending();
        meta2.set_deleted();
        assert_eq!(meta2.lifecycle(), VecLifecycle::PendingDeleted);
        assert!(meta2.is_pending()); // Still needs async cleanup
        assert!(meta2.is_deleted()); // But skip in search

        // PendingDeleted → Deleted (async completes on deleted item)
        meta2.clear_pending();
        assert_eq!(meta2.lifecycle(), VecLifecycle::Deleted);
        assert!(!meta2.is_pending());
        assert!(meta2.is_deleted());
    }

    #[test]
    fn test_vec_metadata_orthogonal_flags_preserved() {
        let mut meta = VecMetadata::pending();

        // Set future REPLICATED flag
        meta.set_replicated(true);
        assert!(meta.is_replicated());
        assert_eq!(meta.lifecycle(), VecLifecycle::Pending); // Lifecycle unchanged

        // Lifecycle transition should preserve REPLICATED
        meta.clear_pending();
        assert!(meta.is_replicated()); // Still replicated
        assert_eq!(meta.lifecycle(), VecLifecycle::Indexed);

        // Another transition
        meta.set_deleted();
        assert!(meta.is_replicated()); // Still replicated
        assert_eq!(meta.lifecycle(), VecLifecycle::Deleted);
    }

    #[test]
    fn test_vec_metadata_wire_compatibility() {
        // Verify the enum values match old FLAG_* constants for wire compatibility
        // Old: FLAG_DELETED=0x01, FLAG_PENDING=0x02
        assert_eq!(VecLifecycle::Indexed as u8, 0x00);
        assert_eq!(VecLifecycle::Deleted as u8, 0x01);
        assert_eq!(VecLifecycle::Pending as u8, 0x02);
        assert_eq!(VecLifecycle::PendingDeleted as u8, 0x03);
    }
}
