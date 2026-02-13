//! Benchmark infrastructure for vector search.
//!
//! This module provides standardized benchmarking tools for HNSW and RaBitQ
//! performance evaluation using real-world datasets.
//!
//! ## Supported Datasets
//!
//! | Dataset | Dimensions | Distance | Format | Use Case |
//! |---------|------------|----------|--------|----------|
//! | LAION-CLIP | 512D | Cosine | NPY (float16) | RaBitQ, semantic search |
//! | SIFT1M | 128D | L2/Euclidean | fvecs/ivecs | Standard HNSW, ANN benchmarks |
//! | GIST-960 | 960D | L2/Euclidean | fvecs/ivecs | High-dim ANN benchmarks |
//! | Cohere Wiki | 768D | Cosine | Parquet | Semantic search (parquet feature) |
//! | GloVe-100 | 100D | Cosine | HDF5 | Word embeddings (hdf5 feature) |
//! | Random | configurable | Cosine | in-memory | Synthetic worst-case |
//!
//! ## Components
//!
//! - [`Dataset`] - Unified trait for all benchmark datasets
//! - [`dataset`] - LAION-CLIP, GIST, Cohere, GloVe, Random dataset loaders
//! - [`sift`] - SIFT dataset (fvecs/ivecs format)
//! - [`runner`] - Experiment runner with configurable parameters
//! - [`metrics`] - Recall, latency, and QPS computation
//!
//! ## Example: Generic Dataset Usage
//!
//! ```ignore
//! use motlie_db::vector::benchmark::{Dataset, SiftDataset};
//!
//! let sift = SiftDataset::load(&data_dir, 10_000, 1000)?;
//! let subset = sift.subset(1000, 100);
//!
//! // Use via the Dataset trait
//! println!("{}: {}D, {:?}", subset.name(), subset.dim(), subset.distance());
//! let gt = subset.ground_truth(10); // None for subsets, Some for full datasets with GT
//! ```
//!
//! ## Example: LAION-CLIP with RaBitQ
//!
//! ```ignore
//! use motlie_db::vector::benchmark::{LaionDataset, ExperimentConfig, run_experiment};
//!
//! // Load LAION dataset (Cosine distance, RaBitQ compatible)
//! let dataset = LaionDataset::load(&data_dir, 100_000)?;
//! let subset = dataset.subset(50_000, 1000);
//!
//! // Run with RaBitQ + BinaryCodeCache + NavigationCache
//! let config = ExperimentConfig::default()
//!     .with_scales(vec![50_000, 100_000])
//!     .with_ef_search(vec![50, 100, 200]);
//!
//! let results = run_experiment(&dataset, &config)?;
//! ```

use crate::vector::Distance;

/// Unified interface for benchmark datasets.
///
/// Every subset struct (`LaionSubset`, `SiftSubset`, `GistSubset`, etc.)
/// implements this trait, enabling `runner.rs` and the `bench_vector` CLI
/// to work generically over any dataset.
///
/// Existing concrete methods on each struct remain unchanged — the trait
/// is additive and non-breaking.
pub trait Dataset {
    /// Human-readable name (e.g. "SIFT-1M", "LAION-CLIP").
    fn name(&self) -> &str;

    /// Vector dimensionality.
    fn dim(&self) -> usize;

    /// Distance metric for this dataset.
    fn distance(&self) -> Distance;

    /// Database vectors (the corpus to index).
    fn vectors(&self) -> &[Vec<f32>];

    /// Query vectors.
    fn queries(&self) -> &[Vec<f32>];

    /// Pre-computed ground truth for the given k, if available.
    ///
    /// Returns `Some(top_k_indices_per_query)` when the dataset ships with
    /// pre-computed neighbors and the stored depth is >= k.
    /// Returns `None` when ground truth must be computed at runtime
    /// (e.g. LAION, Random, or subsets smaller than the full dataset).
    fn ground_truth(&self, k: usize) -> Option<Vec<Vec<usize>>>;
}

pub mod concurrent;
pub mod dataset;
pub mod metadata;
pub mod metrics;
pub mod runner;
pub mod scale;
pub mod sift;

// Re-exports for public API - LAION
pub use dataset::{DatasetConfig, LaionDataset, LaionSubset, NpyLoader, LAION_EMBEDDING_DIM};

// Re-exports for public API - GIST
pub use dataset::{
    GistDataset, GistSubset, GIST_BASE_VECTORS, GIST_EMBEDDING_DIM, GIST_QUERIES,
};

// Re-exports for public API - Random (synthetic)
pub use dataset::{compute_ground_truth_parallel, RandomDataset};

// Re-exports for public API - Parquet datasets (optional)
#[cfg(feature = "parquet")]
pub use dataset::{
    load_parquet_embeddings, CohereWikipediaDataset, CohereWikipediaSubset,
    COHERE_WIKI_DIM, COHERE_WIKI_VECTORS,
};

// Re-exports for public API - HDF5 datasets (optional)
#[cfg(feature = "hdf5")]
pub use dataset::{
    load_hdf5_embeddings, load_hdf5_ground_truth, GloveDataset, GloveSubset,
    GLOVE_DIM, GLOVE_QUERIES, GLOVE_VECTORS,
};

// Re-exports for public API - SIFT
pub use sift::{
    read_fvecs, read_fvecs_limited, read_ivecs, read_ivecs_limited, SiftDataset, SiftSubset,
    SIFT_EMBEDDING_DIM,
};

// Re-exports for public API - Metrics and Runner
pub use metrics::{
    compute_pareto_frontier, compute_pareto_frontier_for_k, compute_recall,
    compute_rotated_variance, percentile, print_pareto_frontier, print_rotation_stats,
    LatencyStats, ParetoInput, ParetoPoint, RecallMetrics, RotationStats,
};
pub use runner::{
    build_hnsw_index, run_all_experiments, run_flat_baseline, run_rabitq_experiments,
    run_single_experiment, save_rabitq_results_csv, ExperimentConfig, ExperimentResult,
    RabitqExperimentResult,
};

// Re-exports for public API - Metadata (incremental builds)
pub use metadata::{BenchmarkMetadata, GroundTruthCache};

// Re-exports for public API - Concurrent benchmarks
pub use concurrent::{
    compare_sync_async_latency, measure_backpressure_impact, measure_backpressure_impact_quick,
    save_benchmark_results_csv, BackpressureResult, BackpressureSample, BenchConfig, BenchResult,
    ConcurrentBenchmark, ConcurrentMetrics, DatasetSource, MetricsSummary, SearchMode,
    SyncAsyncLatencyResult,
};

// Re-exports for public API - Scale benchmarks
pub use scale::{
    get_rss_bytes, AsyncMetrics, ScaleBenchmark, ScaleConfig, ScaleProgress, ScaleResult,
    StreamingVectorGenerator,
};
