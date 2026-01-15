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
//! ### Core Types (flat)
//! - `config.rs` - Configuration types (RaBitQConfig, VectorConfig)
//! - `hnsw/config.rs` - HNSW configuration (`hnsw::Config`)
//! - `distance.rs` - Distance metrics (Cosine, L2, DotProduct)
//! - `embedding.rs` - Embedding type and Embedder trait
//! - `registry.rs` - EmbeddingRegistry for managing embedding spaces
//! - `schema.rs` - RocksDB column family definitions
//! - `error.rs` - Error handling utilities
//! - `id.rs` - ID allocation
//! - `merge.rs` - RocksDB merge operators
//! - `subsystem.rs` - Storage subsystem configuration
//!
//! ### IO Operations (flat)
//! - `mutation.rs` - Write operations (Mutation enum, MutationExecutor trait)
//! - `query.rs` - Read operations (Query enum, QueryExecutor trait)
//! - `reader.rs` - Reader implementation (channel-based query dispatch)
//! - `writer.rs` - Writer implementation (channel-based mutation dispatch)
//! - `processor.rs` - Core operations (insert, delete, search with transactions)
//!
//! ### Nested Modules
//! - `hnsw/` - HNSW index implementation
//! - `cache/` - Caching infrastructure (NavigationCache, BinaryCodeCache)
//! - `quantization/` - Vector quantization (RaBitQ)
//! - `search/` - Search configuration and parallel utilities
//! - `benchmark/` - Benchmark infrastructure (LAION dataset, metrics)
//!
//! ## Design Documents
//!
//! - `BENCHMARK.md` - Performance benchmarks and comparison with Faiss
//! - `ROADMAP.md` - Implementation roadmap and phase details
//! - `REQUIREMENTS.md` - Functional and architectural requirements

// Flat modules - Core types
pub mod config;
pub mod distance;
pub mod embedding;
mod error;
pub mod id;
pub mod merge;
pub mod mutation;
pub mod processor;
pub mod query;
pub mod reader;
pub mod registry;
pub mod schema;
pub mod subsystem;
pub mod writer;

// Nested modules
pub mod benchmark;
pub mod cache;
pub mod hnsw;
pub mod ops;
pub mod quantization;
pub mod search;

// Test modules
#[cfg(test)]
mod crash_recovery_tests;

// Legacy module aliases for backwards compatibility during transition
// TODO: Remove these after updating all imports
pub mod navigation {
    //! Re-export from cache module for backwards compatibility.
    pub use crate::vector::cache::{
        BinaryCodeCache, NavigationCache, NavigationCacheConfig, NavigationLayerInfo,
    };
}
pub mod rabitq {
    //! Re-export from quantization module for backwards compatibility.
    pub use crate::vector::quantization::RaBitQ;
}
pub mod parallel {
    //! Re-export from search module for backwards compatibility.
    pub use crate::vector::search::{
        batch_distances_parallel, distances_from_vectors_parallel, rerank_adaptive, rerank_auto,
        rerank_parallel, rerank_sequential,
    };
}
pub mod search_config {
    //! Re-export from search module for backwards compatibility.
    pub use crate::vector::search::{
        SearchConfig, SearchStrategy, DEFAULT_PARALLEL_RERANK_THRESHOLD,
    };
}

// Re-exports for public API
pub use cache::{BinaryCodeCache, BinaryCodeEntry, NavigationCache, NavigationCacheConfig, NavigationLayerInfo};
pub use config::{RaBitQConfig, RaBitQConfigWarning, VectorConfig};
pub use distance::Distance;
pub use embedding::{Embedder, Embedding, EmbeddingBuilder};
pub use hnsw::ConfigWarning;
pub use id::IdAllocator;
pub use processor::{Processor, SearchResult};
pub use quantization::RaBitQ;
pub use registry::{EmbeddingFilter, EmbeddingRegistry};
pub use schema::{
    AdcCorrection, BinaryCodeCfKey, BinaryCodeCfValue, BinaryCodes, EmbeddingCode, EmbeddingSpec,
    VecId, VectorCfKey, VectorCfValue, VectorElementType, Vectors, ALL_COLUMN_FAMILIES,
};
pub use search::{SearchConfig, SearchStrategy, DEFAULT_PARALLEL_RERANK_THRESHOLD};

// Mutation types and infrastructure (following graph::mutation pattern)
// Note: UpdateEdges, UpdateGraphMeta, EdgeOperation, GraphMetaUpdate are internal-only
// Graph repair requires full rebuild - no partial repair API exposed
pub use mutation::{
    AddEmbeddingSpec, DeleteVector, FlushMarker, InsertVector, InsertVectorBatch, Mutation,
};
pub use writer::{
    create_writer, spawn_consumer as spawn_mutation_consumer, Consumer as MutationConsumer,
    MutationCacheUpdate, MutationExecutor, MutationProcessor, Writer, WriterConfig,
};

// Query types and infrastructure (following graph::query pattern)
pub use query::{
    GetExternalId, GetInternalId, GetVector, Query, QueryExecutor, QueryProcessor,
    QueryWithTimeout, ResolveIds, SearchKNN,
};
pub use reader::{
    create_reader, spawn_consumer as spawn_query_consumer,
    spawn_consumer_with_processor as spawn_query_consumer_with_processor,
    spawn_consumers as spawn_query_consumers,
    spawn_consumers_with_processor as spawn_query_consumers_with_processor,
    Consumer as QueryConsumer, ProcessorConsumer as ProcessorQueryConsumer, Reader, ReaderConfig,
};

// Subsystem exports for use with rocksdb::Storage<S> and StorageBuilder
pub use subsystem::{EmbeddingRegistryConfig, Subsystem, VectorBlockCacheConfig};

/// Storage type alias using generic rocksdb::Storage
pub type Storage = crate::rocksdb::Storage<Subsystem>;

/// BlockCacheConfig re-export (alias to VectorBlockCacheConfig for backwards compat)
pub type BlockCacheConfig = VectorBlockCacheConfig;

// Note: SystemInfo functionality is now in Subsystem which implements SubsystemInfo

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
        let default = hnsw::Config::default();
        assert_eq!(default.dim, 128);
        assert_eq!(default.m, 16);

        let high_recall = hnsw::Config::high_recall(768);
        assert_eq!(high_recall.dim, 768);
        assert_eq!(high_recall.m, 32);

        let compact = hnsw::Config::compact(768);
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
