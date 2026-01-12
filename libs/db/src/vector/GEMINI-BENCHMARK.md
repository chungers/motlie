# Benchmark Infrastructure Improvement Proposal

**Date:** January 12, 2026
**Author:** Gemini Agent
**Scope:** `libs/db/src/vector/benchmark`

## 1. Executive Summary

The current benchmark infrastructure (`libs/db/src/vector/benchmark`) is well-structured for standard HNSW evaluation but lacks native support for tuning **RaBitQ** parameters (`rerank_factor`, `bits_per_dim`). It also relies heavily on LAION-CLIP and SIFT, missing high-dimensional unnormalized datasets (GIST) or other angular datasets (GloVe, Cohere) that could validate RaBitQ's behavior on different distributions.

This proposal outlines additions to the benchmark crate to enable:
1.  **RaBitQ Tuning**: Grid search for `rerank_factor` vs `ef_search`.
2.  **Dataset Expansion**: Support for GIST, GloVe, and Cohere/OpenAI formats.
3.  **Pareto Analysis**: Automated Recall vs QPS frontier generation.

## 2. Infrastructure Improvements

### 2.1 ExperimentConfig Expansion

Modify `ExperimentConfig` in `runner.rs` to include RaBitQ-specific parameters:

```rust
pub struct ExperimentConfig {
    // Existing fields...
    
    // New RaBitQ fields
    pub use_rabitq: bool,
    pub rabitq_bits: Vec<u8>,          // e.g., [1, 2, 4]
    pub rerank_factors: Vec<usize>,    // e.g., [1, 4, 10, 20]
    pub binary_cache: bool,            // Test with/without cache (optional)
}
```

### 2.2 Runner Logic Update

Update `run_single_experiment` in `runner.rs` to support the RaBitQ search path:

-   Detect if `use_rabitq` is true.
-   If true, instantiate `RaBitQ` encoder and `BinaryCodeCache`.
-   Iterate over `rerank_factors`.
-   Call `index.search_with_rabitq_cached()` instead of `index.search()`.
-   Record `rerank_factor` in `ExperimentResult`.

### 2.3 New Datasets (`dataset.rs`)

Add support for the following datasets to validate RaBitQ robustness:

| Dataset | Dimensions | Distance | Justification |
|---------|------------|----------|---------------|
| **GIST-1M** | 960 | L2 | High-dimensional, unnormalized. Tests RaBitQ's limits on non-unit vectors (should fail or require high rerank). |
| **GloVe-100** | 100 | Cosine | Angular distance, standard NLP benchmark. |
| **Cohere/OpenAI** | 768/1536 | Cosine | Modern high-dim embedding distributions (often clustered). Can simulated via synthetic generator or loaded from HDF5/Parquet. |
| **Random (Unit)**| 1024 | Cosine | "Worst case" for ranking (equidistant). Already used in `examples/vector2`, move to lib. |

### 2.4 Metrics Enhancements

-   **Recall vs Latency Pareto Frontier**: Output a JSON/CSV that allows plotting the optimal `(ef_search, rerank_factor)` pairs for a given recall target.
-   **Distribution Metrics**: Compute and log the variance of vector components *after* RaBitQ rotation to verify the $\sqrt{D}$ scaling fix is working for new datasets.

## 3. Implementation Plan

### Step 1: Promote Random Dataset
Move `RandomDataset` generation logic from `examples/vector2/benchmark.rs` to `libs/db/src/vector/benchmark/dataset.rs`.

### Step 2: HDF5/Parquet Support
Many modern benchmarks (ann-benchmarks) use HDF5. Add optional `hdf5` feature to load these directly.

### Step 3: RaBitQ Runner
Implement `run_rabitq_experiment` that loops over:
1.  `bits_per_dim` (1, 2, 4)
2.  `ef_search`
3.  `rerank_factor`

### Step 4: CLI Tool
Create a binary `bins/bench_vector` that exposes these configs via CLI, replacing the ad-hoc `examples/vector2` and `examples/laion_benchmark` scripts.

## 4. Proposed Benchmark Suite

Once implemented, run the following suite to finalize tuning:

1.  **RaBitQ Bits vs Recall**:
    -   Dataset: LAION-CLIP (512D)
    -   Sweep: Bits {1, 2, 4}, Rerank {1..50}
    -   Goal: Determine if 2-bit or 4-bit allows reducing `rerank_factor` significantly compared to 1-bit.

2.  **High-Dim Validation**:
    -   Dataset: Random-1024D (Unit)
    -   Sweep: Bits {4}, Rerank {10..100}
    -   Goal: Confirm recall > 90% is achievable with reasonable QPS.

3.  **Cache Sizing**:
    -   Vary `BinaryCodeCache` capacity.
    -   Goal: Measure performance cliff when cache doesn't fit in RAM.
