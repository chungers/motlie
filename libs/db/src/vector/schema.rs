//! Column family definitions for the vector module.
//!
//! All vector CFs use the `vector/` prefix to avoid collisions with graph CFs.
//! Follows the same patterns as graph::schema for consistency:
//! - Tuple structs for CfKey and CfValue types
//! - pub(crate) visibility
//! - CF marker as unit struct
//!
//! See ROADMAP.md for complete schema documentation.

use crate::Id;
use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::distance::Distance;

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
/// Value: (model, dim, distance)
pub(crate) struct EmbeddingSpecs;

#[derive(Serialize, Deserialize)]
pub(crate) struct EmbeddingSpecsCfKey(pub(crate) u64);

/// EmbeddingSpecs value: (model, dim, distance)
#[derive(Serialize, Deserialize)]
pub(crate) struct EmbeddingSpecsCfValue(
    pub(crate) String,   // model
    pub(crate) u32,      // dim
    pub(crate) Distance, // distance
);

impl EmbeddingSpecs {
    pub const CF_NAME: &'static str = "vector/embedding_specs";

    pub fn key_to_bytes(key: &EmbeddingSpecsCfKey) -> Vec<u8> {
        key.0.to_be_bytes().to_vec()
    }

    pub fn key_from_bytes(bytes: &[u8]) -> Result<EmbeddingSpecsCfKey> {
        if bytes.len() != 8 {
            anyhow::bail!(
                "Invalid EmbeddingSpecsCfKey length: expected 8, got {}",
                bytes.len()
            );
        }
        let mut arr = [0u8; 8];
        arr.copy_from_slice(bytes);
        Ok(EmbeddingSpecsCfKey(u64::from_be_bytes(arr)))
    }

    pub fn value_to_bytes(value: &EmbeddingSpecsCfValue) -> Result<Vec<u8>> {
        let bytes = rmp_serde::to_vec(value)?;
        Ok(bytes)
    }

    pub fn value_from_bytes(bytes: &[u8]) -> Result<EmbeddingSpecsCfValue> {
        let value = rmp_serde::from_slice(bytes)?;
        Ok(value)
    }

    pub fn column_family_options() -> rocksdb::Options {
        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();
        block_opts.set_bloom_filter(10.0, false);
        opts.set_block_based_table_factory(&block_opts);
        opts
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
#[derive(Serialize, Deserialize)]
pub(crate) struct VectorsCfKey(
    pub(crate) u64, // embedding_code
    pub(crate) u32, // vec_id
);

impl Vectors {
    pub const CF_NAME: &'static str = "vector/vectors";

    pub fn key_to_bytes(key: &VectorsCfKey) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(12);
        bytes.extend_from_slice(&key.0.to_be_bytes());
        bytes.extend_from_slice(&key.1.to_be_bytes());
        bytes
    }

    pub fn key_from_bytes(bytes: &[u8]) -> Result<VectorsCfKey> {
        if bytes.len() != 12 {
            anyhow::bail!(
                "Invalid VectorsCfKey length: expected 12, got {}",
                bytes.len()
            );
        }
        let embedding_code = u64::from_be_bytes(bytes[0..8].try_into()?);
        let vec_id = u32::from_be_bytes(bytes[8..12].try_into()?);
        Ok(VectorsCfKey(embedding_code, vec_id))
    }

    /// Store vector as raw f32 bytes (no compression for fast access)
    pub fn value_to_bytes(vector: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(vector.len() * 4);
        for &v in vector {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    /// Load vector from raw f32 bytes
    pub fn value_from_bytes(bytes: &[u8]) -> Result<Vec<f32>> {
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
        Ok(vector)
    }

    pub fn column_family_options() -> rocksdb::Options {
        use rocksdb::SliceTransform;

        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();

        // 16KB blocks for vector data
        block_opts.set_block_size(16 * 1024);
        // LZ4 compression for vectors
        opts.set_compression_type(rocksdb::DBCompressionType::Lz4);

        // Prefix extractor for embedding_code (8 bytes)
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(8));
        opts.set_block_based_table_factory(&block_opts);
        opts
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
#[derive(Serialize, Deserialize)]
pub(crate) struct EdgesCfKey(
    pub(crate) u64, // embedding_code
    pub(crate) u32, // vec_id
    pub(crate) u8,  // layer
);

impl Edges {
    pub const CF_NAME: &'static str = "vector/edges";

    pub fn key_to_bytes(key: &EdgesCfKey) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(13);
        bytes.extend_from_slice(&key.0.to_be_bytes());
        bytes.extend_from_slice(&key.1.to_be_bytes());
        bytes.push(key.2);
        bytes
    }

    pub fn key_from_bytes(bytes: &[u8]) -> Result<EdgesCfKey> {
        if bytes.len() != 13 {
            anyhow::bail!(
                "Invalid EdgesCfKey length: expected 13, got {}",
                bytes.len()
            );
        }
        let embedding_code = u64::from_be_bytes(bytes[0..8].try_into()?);
        let vec_id = u32::from_be_bytes(bytes[8..12].try_into()?);
        let layer = bytes[12];
        Ok(EdgesCfKey(embedding_code, vec_id, layer))
    }

    pub fn column_family_options() -> rocksdb::Options {
        use rocksdb::SliceTransform;

        let mut opts = rocksdb::Options::default();
        // No compression - RoaringBitmap is already compact
        opts.set_compression_type(rocksdb::DBCompressionType::None);
        // Prefix: embedding_code (8) + vec_id (4) = 12 bytes
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(12));
        opts
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
#[derive(Serialize, Deserialize)]
pub(crate) struct BinaryCodesCfKey(
    pub(crate) u64, // embedding_code
    pub(crate) u32, // vec_id
);

impl BinaryCodes {
    pub const CF_NAME: &'static str = "vector/binary_codes";

    pub fn key_to_bytes(key: &BinaryCodesCfKey) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(12);
        bytes.extend_from_slice(&key.0.to_be_bytes());
        bytes.extend_from_slice(&key.1.to_be_bytes());
        bytes
    }

    pub fn key_from_bytes(bytes: &[u8]) -> Result<BinaryCodesCfKey> {
        if bytes.len() != 12 {
            anyhow::bail!(
                "Invalid BinaryCodesCfKey length: expected 12, got {}",
                bytes.len()
            );
        }
        let embedding_code = u64::from_be_bytes(bytes[0..8].try_into()?);
        let vec_id = u32::from_be_bytes(bytes[8..12].try_into()?);
        Ok(BinaryCodesCfKey(embedding_code, vec_id))
    }

    pub fn column_family_options() -> rocksdb::Options {
        use rocksdb::SliceTransform;

        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();

        // 4KB blocks for binary codes (small, dense data)
        block_opts.set_block_size(4 * 1024);
        // No compression - already compact binary data
        opts.set_compression_type(rocksdb::DBCompressionType::None);
        // Prefix: embedding_code (8 bytes)
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(8));
        opts.set_block_based_table_factory(&block_opts);
        opts
    }
}

// ============================================================================
// VecMeta CF
// ============================================================================

/// Vector metadata column family.
///
/// Key: [embedding: u64] + [vec_id: u32] = 12 bytes
/// Value: (max_layer, flags, created_at)
pub(crate) struct VecMeta;

/// VecMeta key: (embedding_code, vec_id)
#[derive(Serialize, Deserialize)]
pub(crate) struct VecMetaCfKey(
    pub(crate) u64, // embedding_code
    pub(crate) u32, // vec_id
);

/// VecMeta value: (max_layer, flags, created_at)
#[derive(Serialize, Deserialize)]
pub(crate) struct VecMetaCfValue(
    pub(crate) u8,  // max_layer
    pub(crate) u8,  // flags
    pub(crate) u64, // created_at
);

impl VecMeta {
    pub const CF_NAME: &'static str = "vector/vec_meta";

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

    pub fn column_family_options() -> rocksdb::Options {
        use rocksdb::SliceTransform;

        let mut opts = rocksdb::Options::default();
        // No compression for small metadata
        opts.set_compression_type(rocksdb::DBCompressionType::None);
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(8));
        opts
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

/// GraphMeta key: (embedding_code, field)
#[derive(Serialize, Deserialize)]
pub(crate) struct GraphMetaCfKey(
    pub(crate) u64, // embedding_code
    pub(crate) u8,  // field
);

/// Graph metadata field constants
pub mod graph_meta_field {
    pub const ENTRY_POINT: u8 = 0;
    pub const MAX_LEVEL: u8 = 1;
    pub const COUNT: u8 = 2;
    pub const CONFIG: u8 = 3;
}

impl GraphMeta {
    pub const CF_NAME: &'static str = "vector/graph_meta";

    pub fn key_to_bytes(key: &GraphMetaCfKey) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(9);
        bytes.extend_from_slice(&key.0.to_be_bytes());
        bytes.push(key.1);
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
        let field = bytes[8];
        Ok(GraphMetaCfKey(embedding_code, field))
    }

    pub fn column_family_options() -> rocksdb::Options {
        rocksdb::Options::default()
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

/// IdForward key: (embedding_code, ulid)
#[derive(Serialize, Deserialize)]
pub(crate) struct IdForwardCfKey(
    pub(crate) u64, // embedding_code
    pub(crate) Id,  // ulid
);

impl IdForward {
    pub const CF_NAME: &'static str = "vector/id_forward";

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

    pub fn value_to_bytes(vec_id: u32) -> Vec<u8> {
        vec_id.to_be_bytes().to_vec()
    }

    pub fn value_from_bytes(bytes: &[u8]) -> Result<u32> {
        if bytes.len() != 4 {
            anyhow::bail!(
                "Invalid IdForward value length: expected 4, got {}",
                bytes.len()
            );
        }
        Ok(u32::from_be_bytes(bytes.try_into()?))
    }

    pub fn column_family_options() -> rocksdb::Options {
        use rocksdb::SliceTransform;

        let mut opts = rocksdb::Options::default();
        let mut block_opts = rocksdb::BlockBasedOptions::default();
        block_opts.set_bloom_filter(10.0, false);
        // Prefix: embedding_code (8 bytes)
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(8));
        opts.set_block_based_table_factory(&block_opts);
        opts
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
#[derive(Serialize, Deserialize)]
pub(crate) struct IdReverseCfKey(
    pub(crate) u64, // embedding_code
    pub(crate) u32, // vec_id
);

impl IdReverse {
    pub const CF_NAME: &'static str = "vector/id_reverse";

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

    pub fn value_to_bytes(ulid: &Id) -> Vec<u8> {
        ulid.as_bytes().to_vec()
    }

    pub fn value_from_bytes(bytes: &[u8]) -> Result<Id> {
        if bytes.len() != 16 {
            anyhow::bail!(
                "Invalid IdReverse value length: expected 16, got {}",
                bytes.len()
            );
        }
        let mut ulid_bytes = [0u8; 16];
        ulid_bytes.copy_from_slice(bytes);
        Ok(Id::from_bytes(ulid_bytes))
    }

    pub fn column_family_options() -> rocksdb::Options {
        use rocksdb::SliceTransform;

        let mut opts = rocksdb::Options::default();
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(8));
        opts
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

/// IdAlloc key: (embedding_code, field)
#[derive(Serialize, Deserialize)]
pub(crate) struct IdAllocCfKey(
    pub(crate) u64, // embedding_code
    pub(crate) u8,  // field
);

/// ID allocator field constants
pub mod id_alloc_field {
    pub const NEXT_ID: u8 = 0;
    pub const FREE_BITMAP: u8 = 1;
}

impl IdAlloc {
    pub const CF_NAME: &'static str = "vector/id_alloc";

    pub fn key_to_bytes(key: &IdAllocCfKey) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(9);
        bytes.extend_from_slice(&key.0.to_be_bytes());
        bytes.push(key.1);
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
        let field = bytes[8];
        Ok(IdAllocCfKey(embedding_code, field))
    }

    pub fn column_family_options() -> rocksdb::Options {
        rocksdb::Options::default()
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
#[derive(Serialize, Deserialize)]
pub(crate) struct PendingCfKey(
    pub(crate) u64, // embedding_code
    pub(crate) u64, // timestamp
    pub(crate) u32, // vec_id
);

impl Pending {
    pub const CF_NAME: &'static str = "vector/pending";

    pub fn key_to_bytes(key: &PendingCfKey) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(20);
        bytes.extend_from_slice(&key.0.to_be_bytes());
        bytes.extend_from_slice(&key.1.to_be_bytes());
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
        let timestamp = u64::from_be_bytes(bytes[8..16].try_into()?);
        let vec_id = u32::from_be_bytes(bytes[16..20].try_into()?);
        Ok(PendingCfKey(embedding_code, timestamp, vec_id))
    }

    pub fn column_family_options() -> rocksdb::Options {
        use rocksdb::SliceTransform;

        let mut opts = rocksdb::Options::default();
        // Prefix: embedding_code (8 bytes) for scanning per-embedding pending items
        opts.set_prefix_extractor(SliceTransform::create_fixed_prefix(8));
        opts
    }
}

// ============================================================================
// Schema - ColumnFamilyProvider Implementation
// ============================================================================

use crate::provider::ColumnFamilyProvider;
use crate::vector::registry::EmbeddingRegistry as Registry;
use rocksdb::{Cache, ColumnFamilyDescriptor, TransactionDB};
use std::sync::Arc;

/// Vector module schema implementing the ColumnFamilyProvider trait.
///
/// This enables the vector module to register its column families with
/// the shared RocksDB instance without coupling to the graph module.
pub struct Schema {
    /// Embedding registry for pre-warming
    registry: Arc<Registry>,
}

impl Schema {
    /// Create a new Schema.
    pub fn new() -> Self {
        Self {
            registry: Arc::new(Registry::new()),
        }
    }

    /// Create with a custom registry (for sharing).
    pub fn with_registry(registry: Arc<Registry>) -> Self {
        Self { registry }
    }

    /// Get the embedding registry.
    pub fn registry(&self) -> &Arc<Registry> {
        &self.registry
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
        "vector"
    }

    fn column_family_descriptors(
        &self,
        cache: Option<&Cache>,
        block_size: usize,
    ) -> Vec<ColumnFamilyDescriptor> {
        vec![
            ColumnFamilyDescriptor::new(
                EmbeddingSpecs::CF_NAME,
                Self::cf_options_with_cache(
                    EmbeddingSpecs::column_family_options(),
                    cache,
                    block_size,
                ),
            ),
            ColumnFamilyDescriptor::new(
                Vectors::CF_NAME,
                Self::cf_options_with_cache(Vectors::column_family_options(), cache, 16 * 1024),
            ),
            ColumnFamilyDescriptor::new(
                Edges::CF_NAME,
                Self::cf_options_with_cache(Edges::column_family_options(), cache, block_size),
            ),
            ColumnFamilyDescriptor::new(
                BinaryCodes::CF_NAME,
                Self::cf_options_with_cache(BinaryCodes::column_family_options(), cache, 4 * 1024),
            ),
            ColumnFamilyDescriptor::new(
                VecMeta::CF_NAME,
                Self::cf_options_with_cache(VecMeta::column_family_options(), cache, block_size),
            ),
            ColumnFamilyDescriptor::new(
                GraphMeta::CF_NAME,
                Self::cf_options_with_cache(GraphMeta::column_family_options(), cache, block_size),
            ),
            ColumnFamilyDescriptor::new(
                IdForward::CF_NAME,
                Self::cf_options_with_cache(IdForward::column_family_options(), cache, block_size),
            ),
            ColumnFamilyDescriptor::new(
                IdReverse::CF_NAME,
                Self::cf_options_with_cache(IdReverse::column_family_options(), cache, block_size),
            ),
            ColumnFamilyDescriptor::new(
                IdAlloc::CF_NAME,
                Self::cf_options_with_cache(IdAlloc::column_family_options(), cache, block_size),
            ),
            ColumnFamilyDescriptor::new(
                Pending::CF_NAME,
                Self::cf_options_with_cache(Pending::column_family_options(), cache, block_size),
            ),
        ]
    }

    fn on_ready(&self, db: &TransactionDB) -> Result<()> {
        // Pre-warm the embedding registry
        let count = self.registry.prewarm(db)?;
        tracing::info!(count, "[vector::Schema] Pre-warmed EmbeddingRegistry");
        Ok(())
    }

    fn cf_names(&self) -> Vec<&'static str> {
        ALL_COLUMN_FAMILIES.to_vec()
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
        let key = VectorsCfKey(42, 123);
        let bytes = Vectors::key_to_bytes(&key);
        assert_eq!(bytes.len(), 12);
        let parsed = Vectors::key_from_bytes(&bytes).unwrap();
        assert_eq!(parsed.0, 42);
        assert_eq!(parsed.1, 123);
    }

    #[test]
    fn test_vectors_value_roundtrip() {
        let vector = vec![1.0f32, 2.0, 3.0, 4.0];
        let bytes = Vectors::value_to_bytes(&vector);
        assert_eq!(bytes.len(), 16); // 4 floats * 4 bytes
        let parsed = Vectors::value_from_bytes(&bytes).unwrap();
        assert_eq!(parsed, vector);
    }

    #[test]
    fn test_edges_key_roundtrip() {
        let key = EdgesCfKey(1, 42, 3);
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
        let key = PendingCfKey(1, 1000, 42);
        let bytes = Pending::key_to_bytes(&key);
        assert_eq!(bytes.len(), 20);
        let parsed = Pending::key_from_bytes(&bytes).unwrap();
        assert_eq!(parsed.0, 1);
        assert_eq!(parsed.1, 1000);
        assert_eq!(parsed.2, 42);
    }

    #[test]
    fn test_schema_new() {
        let schema = Schema::new();
        assert_eq!(schema.name(), "vector");
    }

    #[test]
    fn test_schema_cf_names() {
        let schema = Schema::new();
        let names = schema.cf_names();
        assert_eq!(names.len(), ALL_COLUMN_FAMILIES.len());
        for name in &names {
            assert!(name.starts_with("vector/"));
        }
    }

    #[test]
    fn test_schema_column_family_descriptors() {
        let schema = Schema::new();
        let descriptors = schema.column_family_descriptors(None, 4096);
        assert_eq!(descriptors.len(), ALL_COLUMN_FAMILIES.len());
    }

    #[test]
    fn test_schema_registry_access() {
        let schema = Schema::new();
        let registry = schema.registry();
        assert!(registry.is_empty());
    }
}
