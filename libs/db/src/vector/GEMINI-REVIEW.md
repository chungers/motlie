# Gemini Code Review: Vector Search Subsystem

**Date:** January 12, 2026
**Reviewer:** Gemini Agent
**Scope:** `libs/db/src/vector`

## 1. Documentation & Implementation Consistency

The documentation (`README.md`, `ROADMAP.md`, `REQUIREMENTS.md`) is comprehensive and generally aligns with the codebase structure. However, there is a critical discrepancy regarding the "completion" status of HNSW indexing.

*   **Discrepancy:** `ROADMAP.md` claims "Phase 2: HNSW2 Core + Navigation Layer" is "Complete". While the core HNSW logic exists in `hnsw/insert.rs` and `hnsw/search.rs`, it is **not connected** to the write path in `writer.rs`.
*   **Impact:** The `InsertVector` mutation currently stores the vector data, ID mappings, and RaBitQ codes, but **does not insert the vector into the HNSW graph**. The graph remains empty, rendering the HNSW search non-functional (it will return empty results or fall back to brute force if implemented, though the search logic relies on the graph).
*   **Union Pattern:** The "Union Pattern" described in `README.md` Appendix A is correctly implemented in `schema.rs` for `GraphMeta` and `IdAlloc`.
*   **RaBitQ:** The implementation generates and stores binary codes as described, but there are mathematical concerns (see below).

### 1.1 Integration Test Analysis
Analysis of `libs/db/tests/test_vector_benchmark_integration.rs` confirms that the core algorithms (HNSW, RaBitQ) are functional but reveals why the system disconnect was not caught:
*   **Manual Indexing:** The tests manually invoke `index.insert()` and populate caches (e.g., `BinaryCodeCache`), bypassing the `writer.rs` mutation consumer.
*   **Validation Scope:** This validates the *components* (math, search logic, graph traversal) but fails to validate the *system write path*. The production-intent path (`api::insert` -> `writer` -> `consumer`) is indeed broken for indexing.
*   **1-bit Bias:** Tests primarily use 1-bit RaBitQ, which masks the threshold scaling issue present in 2-bit/4-bit modes.

## 2. Design & Algorithm Assessment

### 2.1 RaBitQ Quantization (Critical Issue)
The RaBitQ implementation in `quantization/rabitq.rs` uses a random orthonormal rotation followed by quantization. This is a valid approach for "training-free" quantization (DATA-1 compliant). However, the quantization thresholds are incorrect for the intended use case (unit vectors/Cosine distance).

*   **Issue:** The `quantize_2bit` and `quantize_4bit` functions use thresholds based on a standard normal distribution (variance=1), e.g., `[-0.5, 0.0, 0.5]`.
*   **Math:** For high-dimensional unit vectors (dimension $D$), the components are approximately normally distributed with variance $1/D$. For $D=128$, standard deviation $\sigma \approx 0.088$.
*   **Result:** A threshold of $0.5$ is $>5\sigma$. Practically all values will fall into the central bins (effectively just sign quantization). The 2-bit and 4-bit modes waste storage without providing additional precision over 1-bit quantization.
*   **Fix:** Scale the random rotation matrix by $\sqrt{D}$ during generation, or scale the input/thresholds appropriate to the dimension.

**Update (2026-01-12): Fixed**
*   **Status:** Resolved in commit `f533140`.
*   **Resolution:** The `generate_rotation_matrix` function was updated to scale the rotation matrix by $\sqrt{D}$. This ensures rotated unit vector components have variance $\approx 1$, matching the standard normal quantization thresholds.
*   **Verification:** New unit tests (`test_rotated_unit_vector_variance`, `test_2bit_uses_all_levels`) confirm that the variance is correct and that all quantization levels are now effectively utilized (e.g., 2-bit mode uses all 4 levels, whereas before it only used 2).

### 2.2 HNSW Graph Storage
The use of `RoaringBitmap` for edge lists in `schema.rs` is an excellent design choice for minimizing storage overhead. The implementation of `edge_merge_operator` in `merge.rs` correctly handles concurrent updates.

### 2.3 Distance Metrics
`distance.rs` correctly delegates to SIMD-optimized core implementations. The negation of `DotProduct` is a good practice for consistent "lower is better" semantics.

## 3. Actionable Improvements

### High Priority (Correctness)

1.  **Fix RaBitQ Thresholds:**
    *   **File:** `libs/db/src/vector/quantization/rabitq.rs`
    *   **Action:** Modify `generate_rotation_matrix` to scale the resulting matrix by $\sqrt{D}$ (multiply all elements by `dim as f32`). This ensures the rotated vector components have unit variance, matching the fixed quantization thresholds. Alternatively, adjust the thresholds in `quantize_*` functions.

2.  **Hook up HNSW Indexing:**
    *   **File:** `libs/db/src/vector/writer.rs`
    *   **Action:** In `execute_single` for `Mutation::InsertVector`, call `hnsw::insert` (or a wrapper in `Processor`) to update the HNSW graph.
    *   **Detail:** This should likely be conditional on `op.immediate_index` or performed asynchronously if an `AsyncUpdater` is implemented. Currently, no graph is built.
    *   **Refinement:** Ensure `Processor` exposes the `Index` or a method to access it, as `writer.rs` currently only has `Processor`.

### Medium Priority (Performance/Feature)

3.  **Implement Async Graph Updater (Phase 7):**
    *   **File:** `libs/db/src/vector/async_updater.rs` (to be created)
    *   **Action:** Create a background worker that consumes from the `Pending` CF (or `writer` queue) and applies HNSW updates. This is necessary to decouple write latency from graph construction cost (which can be high). Until this is done, `immediate_index: true` is the only way to make data searchable via HNSW.
