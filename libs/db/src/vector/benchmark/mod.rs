//! Benchmark infrastructure for vector search.
//!
//! This module provides standardized benchmarking tools for HNSW and RaBitQ
//! performance evaluation using real-world datasets (LAION-400M CLIP embeddings).
//!
//! ## Components
//!
//! - [`dataset`] - Dataset loading and management (LAION-CLIP, NPY format)
//! - [`runner`] - Experiment runner with configurable parameters
//! - [`metrics`] - Recall, latency, and QPS computation
//!
//! ## Example
//!
//! ```ignore
//! use motlie_db::vector::benchmark::{LaionDataset, ExperimentConfig, run_experiment};
//!
//! // Load dataset
//! let dataset = LaionDataset::load(&data_dir, 100_000)?;
//! let subset = dataset.subset(50_000, 1000);
//!
//! // Configure experiment
//! let config = ExperimentConfig::default()
//!     .with_scales(vec![50_000, 100_000])
//!     .with_ef_search(vec![50, 100, 200]);
//!
//! // Run and collect results
//! let results = run_experiment(&dataset, &config)?;
//! ```

pub mod dataset;
pub mod metrics;
pub mod runner;

// Re-exports for public API
pub use dataset::{DatasetConfig, LaionDataset, LaionSubset, NpyLoader, LAION_EMBEDDING_DIM};
pub use metrics::{compute_recall, percentile, LatencyStats, RecallMetrics};
pub use runner::{
    build_hnsw_index, run_all_experiments, run_flat_baseline, run_single_experiment,
    ExperimentConfig, ExperimentResult,
};
