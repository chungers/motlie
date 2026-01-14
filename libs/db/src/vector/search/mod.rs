//! Search configuration and parallel processing utilities.
//!
//! This module provides:
//!
//! - `SearchConfig`: Type-safe search configuration driven by `Embedding`
//! - `SearchStrategy`: Automatic strategy selection (Exact vs RaBitQ)
//! - Parallel reranking utilities for large candidate sets
//!
//! # Design Philosophy
//!
//! The `Embedding` struct is the single source of truth for distance metrics.
//! `SearchConfig` captures this and auto-selects the optimal search strategy:
//!
//! - Cosine → RaBitQ (ADC approximates angular distance)
//! - L2/DotProduct → Exact (ADC not compatible)

mod config;
mod parallel;

pub use config::{SearchConfig, SearchStrategy, DEFAULT_PARALLEL_RERANK_THRESHOLD};
pub use parallel::{
    batch_distances_parallel, distances_from_vectors_parallel, rerank_adaptive, rerank_auto,
    rerank_parallel, rerank_sequential,
};
