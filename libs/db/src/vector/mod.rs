//! Vector module - HNSW-based vector storage and search.
//!
//! This module provides vector similarity search with:
//! - HNSW graph index for approximate nearest neighbor search
//! - RaBitQ binary quantization for memory efficiency
//! - Multi-embedding support (same document in multiple spaces)
//! - Shared RocksDB with graph module
//!
//! ## Module Structure
//!
//! - `mod.rs` - Module exports and VectorStorage
//! - `config.rs` - Configuration types (HnswConfig, RaBitQConfig, VectorConfig)
//! - `distance.rs` - Distance metrics (Cosine, L2, DotProduct)
//! - `embedding.rs` - Embedding type and Embedder trait
//! - `registry.rs` - EmbeddingRegistry for managing embedding spaces
//! - `schema.rs` - RocksDB column family definitions
//! - `error.rs` - Error handling utilities
//!
//! ## Design Documents
//!
//! - `ROADMAP.md` - Implementation roadmap and phase details
//! - `REQUIREMENTS.md` - Functional and architectural requirements

// Submodules
pub mod api;
pub mod config;
pub mod distance;
pub mod embedding;
mod error;
pub mod hnsw;
pub mod id;
pub mod merge;
pub mod mutation;
pub mod navigation;
pub mod parallel;
pub mod processor;
pub mod query;
pub mod rabitq;
pub mod reader;
pub mod registry;
pub mod schema;
pub mod search_config;
pub mod subsystem;
pub mod writer;

// Re-exports for public API
pub use config::{ConfigWarning, HnswConfig, RaBitQConfig, VectorConfig};
pub use distance::Distance;
pub use embedding::{Embedder, Embedding, EmbeddingBuilder};
pub use hnsw::HnswIndex;
pub use id::IdAllocator;
pub use navigation::{BinaryCodeCache, NavigationCache, NavigationCacheConfig, NavigationLayerInfo};
pub use processor::Processor;
pub use rabitq::RaBitQ;
pub use registry::{EmbeddingFilter, EmbeddingRegistry};
pub use schema::{
    ALL_COLUMN_FAMILIES, BinaryCodeCfKey, BinaryCodeCfValue, BinaryCodes, EmbeddingCode,
    EmbeddingSpec, VecId, VectorCfKey, VectorCfValue, VectorStorageType, Vectors,
};
pub use search_config::{SearchConfig, SearchStrategy};

// Subsystem exports for use with rocksdb::Storage<S> and StorageBuilder
pub use subsystem::{EmbeddingRegistryConfig, Subsystem, VectorBlockCacheConfig};

/// Storage type alias using generic rocksdb::Storage
pub type Storage = crate::rocksdb::Storage<Subsystem>;

/// BlockCacheConfig re-export (alias to VectorBlockCacheConfig for backwards compat)
pub type BlockCacheConfig = VectorBlockCacheConfig;

/// Component type for use with StorageBuilder
pub type Component = crate::rocksdb::ComponentWrapper<Subsystem>;

/// Convenience constructor for vector component
pub fn component() -> Component {
    Component::new()
}

// Internal re-exports
pub(crate) use error::Result;

// ============================================================================
// SystemInfo - Telemetry
// ============================================================================

/// Static configuration info for the vector database subsystem.
///
/// Used by the `motlie info` command to display vector DB settings.
/// Implements [`motlie_core::telemetry::SubsystemInfo`] for consistent formatting.
///
/// # Example
///
/// ```ignore
/// use motlie_db::vector::SystemInfo;
/// use motlie_core::telemetry::{format_subsystem_info, SubsystemInfo};
///
/// let info = SystemInfo::default();
/// println!("{}", format_subsystem_info(&info));
/// ```
#[derive(Debug, Clone)]
pub struct SystemInfo {
    /// Block cache configuration
    pub block_cache_config: BlockCacheConfig,
    /// EmbeddingRegistry pre-warming configuration
    pub registry_config: EmbeddingRegistryConfig,
    /// List of column families
    pub column_families: Vec<&'static str>,
}

impl Default for SystemInfo {
    fn default() -> Self {
        Self {
            block_cache_config: BlockCacheConfig::default(),
            registry_config: EmbeddingRegistryConfig::default(),
            column_families: ALL_COLUMN_FAMILIES.to_vec(),
        }
    }
}

impl motlie_core::telemetry::SubsystemInfo for SystemInfo {
    fn name(&self) -> &'static str {
        "Vector Database (RocksDB)"
    }

    fn info_lines(&self) -> Vec<(&'static str, String)> {
        vec![
            ("Block Cache Size", format_bytes(self.block_cache_config.cache_size_bytes)),
            ("Default Block Size", format_bytes(self.block_cache_config.default_block_size)),
            ("Vector Block Size", format_bytes(self.block_cache_config.vector_block_size)),
            ("Registry Prewarm", self.registry_config.prewarm_limit.to_string()),
            ("Column Families", self.column_families.join(", ")),
        ]
    }
}

/// Format a byte count as a human-readable string.
fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{} GB", bytes / (1024 * 1024 * 1024))
    } else if bytes >= 1024 * 1024 {
        format!("{} MB", bytes / (1024 * 1024))
    } else if bytes >= 1024 {
        format!("{} KB", bytes / 1024)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify public types are accessible
        let _config = VectorConfig::default();
        let _distance = Distance::Cosine;
        let _filter = EmbeddingFilter::default();
    }

    #[test]
    fn test_cf_names() {
        // All CFs should have vector/ prefix
        for cf_name in ALL_COLUMN_FAMILIES {
            assert!(
                cf_name.starts_with("vector/"),
                "CF {} missing vector/ prefix",
                cf_name
            );
        }
    }

    #[test]
    fn test_hnsw_config_presets() {
        let default = HnswConfig::default();
        assert_eq!(default.dim, 128);
        assert_eq!(default.m, 16);

        let high_recall = HnswConfig::high_recall(768);
        assert_eq!(high_recall.dim, 768);
        assert_eq!(high_recall.m, 32);

        let compact = HnswConfig::compact(768);
        assert_eq!(compact.dim, 768);
        assert_eq!(compact.m, 8);
    }

    #[test]
    fn test_vector_config_presets() {
        let c128 = VectorConfig::dim_128();
        assert_eq!(c128.hnsw.dim, 128);

        let c768 = VectorConfig::dim_768();
        assert_eq!(c768.hnsw.dim, 768);

        let c1024 = VectorConfig::dim_1024();
        assert_eq!(c1024.hnsw.dim, 1024);

        let c1536 = VectorConfig::dim_1536();
        assert_eq!(c1536.hnsw.dim, 1536);
    }

    #[test]
    fn test_distance_metrics() {
        assert_eq!(Distance::Cosine.as_str(), "cosine");
        assert_eq!(Distance::L2.as_str(), "l2");
        assert_eq!(Distance::DotProduct.as_str(), "dot");

        // All metrics: lower = more similar
        assert!(Distance::Cosine.is_lower_better());
        assert!(Distance::L2.is_lower_better());
        assert!(Distance::DotProduct.is_lower_better());
    }

    #[test]
    fn test_embedding_builder() {
        let builder = EmbeddingBuilder::new("gemma", 768, Distance::Cosine);
        assert_eq!(builder.model(), "gemma");
        assert_eq!(builder.dim(), 768);
        assert_eq!(builder.distance(), Distance::Cosine);
    }

    #[test]
    fn test_embedding_filter() {
        let filter = EmbeddingFilter::default()
            .model("gemma")
            .dim(768)
            .distance(Distance::Cosine);

        assert_eq!(filter.model, Some("gemma".to_string()));
        assert_eq!(filter.dim, Some(768));
        assert_eq!(filter.distance, Some(Distance::Cosine));
    }
}
