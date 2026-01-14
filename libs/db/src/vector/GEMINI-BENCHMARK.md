# Benchmark Infrastructure Improvement Proposal

**Date:** January 12, 2026
**Author:** Gemini Agent
**Scope:** `libs/db/src/vector/benchmark`

## 0. Implementation Status Summary (2026-01-12)

**Overall Status:** Mostly Complete. The core infrastructure, CLI tool, and major datasets are implemented.

**Missing Items & Inconsistencies:**
1.  **Random Dataset (Inconsistency):** `BENCHMARK2.md` includes "Random (Unit)" in the Experiment Matrix and results, but the `bench_vector` CLI **does not support** the `random` dataset. The synthetic generator exists only in `examples/vector2`, not in the shared library.
2.  **Distribution Metrics:** Logic to compute/log post-rotation vector component variance is missing.

---

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

**Status: ✅ Implemented.** `ExperimentConfig` now includes `use_rabitq`, `rabitq_bits`, `rerank_factors`, and `use_binary_cache`.

### 2.2 Runner Logic Update

Update `run_single_experiment` in `runner.rs` to support the RaBitQ search path:

-   Detect if `use_rabitq` is true.
-   If true, instantiate `RaBitQ` encoder and `BinaryCodeCache`.
-   Iterate over `rerank_factors`.
-   Call `index.search_with_rabitq_cached()` instead of `index.search()`.
-   Record `rerank_factor` in `ExperimentResult`.

**Status: ✅ Implemented.** `runner.rs` includes `run_rabitq_experiments` which handles the specific RaBitQ loop and uses `search_with_rabitq_cached`.

### 2.3 New Datasets (`dataset.rs`)

Add support for the following datasets to validate RaBitQ robustness:

| Dataset | Dimensions | Distance | Justification | Status |
|---------|------------|----------|---------------|--------|
| **GIST-1M** | 960 | L2 | High-dimensional, unnormalized. | ✅ Implemented |
| **GloVe-100** | 100 | Cosine | Angular distance, standard NLP benchmark. | ✅ Implemented (hdf5) |
| **Cohere/OpenAI** | 768/1536 | Cosine | Modern high-dim embedding distributions. | ✅ Implemented (parquet) |
| **Random (Unit)**| 1024 | Cosine | "Worst case" for ranking (equidistant). | ✅ Implemented |

**Status: ✅ Complete.** All datasets including synthetic `RandomDataset` are now implemented.

### 2.4 Metrics Enhancements

-   **Recall vs Latency Pareto Frontier**: Output a JSON/CSV that allows plotting the optimal `(ef_search, rerank_factor)` pairs for a given recall target.
-   **Distribution Metrics**: Compute and log the variance of vector components *after* RaBitQ rotation to verify the $\sqrt{D}$ scaling fix is working for new datasets.

**Status: ⚠️ Partial.** CSV output for Pareto analysis is implemented (`rabitq_results.csv`), but distribution metrics (component variance) are missing.

## 3. Implementation Plan

### Step 1: Promote Random Dataset
Move `RandomDataset` generation logic from `examples/vector2/benchmark.rs` to `libs/db/src/vector/benchmark/dataset.rs`.
**Status: ✅ Complete.** `RandomDataset` added to `dataset.rs` with parallel ground truth computation.

### Step 2: HDF5/Parquet Support
Many modern benchmarks (ann-benchmarks) use HDF5. Add optional `hdf5` feature to load these directly.
**Status: ✅ Complete.** Features `hdf5` and `parquet` are added.

### Step 3: RaBitQ Runner
Implement `run_rabitq_experiment` that loops over:
1.  `bits_per_dim` (1, 2, 4)
2.  `ef_search`
3.  `rerank_factor`
**Status: ✅ Complete.** `run_rabitq_experiments` implements this loop.

### Step 4: CLI Tool
Create a binary `bins/bench_vector` that exposes these configs via CLI, replacing the ad-hoc `examples/vector2` and `examples/laion_benchmark` scripts.
**Status: ✅ Complete.** `bench_vector` binary is implemented with `sweep`, `download`, `index`, `query` commands.

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

4.  **Gray Code Verification (Issue #43)**:
    -   **Objective:** Confirm that 2-bit/4-bit recall improves after Gray Code implementation.
    -   **Test:** Compare `bits=1` vs `bits=2` vs `bits=4` on Random-1024D.
    -   **Success Criteria:** Recall must increase with bits. (Current broken state: Recall decreases).

## 5. Post-Implementation Recommendations

1.  ~~**Implement `RandomDataset`:**~~ ✅ **DONE** - `RandomDataset` added to `libs/db/src/vector/benchmark/dataset.rs` with CLI support via `--dataset random`.
2.  **Add `check-distribution` Command:** Add a CLI command to sample vectors, apply RaBitQ rotation, and report component variance. This is critical to verifying the $\sqrt{D}$ scaling fix works on novel distributions (like GIST or Cohere).
3.  **Automated Pareto Reporter:** Add a tool (or CLI subcommand) that consumes the `rabitq_results.csv` and outputs the optimal configuration for specific recall targets (e.g., "Best config for 95% recall: bits=4, rerank=12").

