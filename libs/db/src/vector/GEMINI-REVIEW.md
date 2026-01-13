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

---

## 4. Additional Findings (Post-Review)

### 4.1 RaBitQ √D Scaling Fix (Issue #42) - RESOLVED

**Date:** January 12, 2026
**Discovered by:** Gemini (see Section 2.1)
**Implemented by:** Claude

The RaBitQ threshold scaling issue identified in Section 2.1 has been fixed:
- **Commit:** f533140
- **Fix:** Scale rotation matrix by √D in `generate_rotation_matrix()`
- **Validation:** Added tests `test_rotated_unit_vector_variance`, `test_2bit_uses_all_levels`, `test_4bit_uses_many_levels`

### 4.2 Multi-bit Hamming Distance Incompatibility (Issue #43) - RESOLVED

**Date:** January 12, 2026
**Discovered by:** Claude (during benchmark verification of RaBitQ recall claims)
**Status:** ✅ Resolved - Gray code encoding implemented

#### Discovery Context

During verification of the recall claims in `API.md` (2-bit ~85%, 4-bit ~92%), Claude ran benchmarks using `examples/vector2` with the LAION-400M-CLIP512 dataset (100K vectors). The results revealed a critical bug:

#### Benchmark Results

| Bits | Recall@10 (rerank=10) | Expected | Status |
|------|----------------------|----------|--------|
| 1-bit | **50.8%** | ~50% | ✅ Working |
| 2-bit | **45.4%** | ~85% | ❌ **Worse than 1-bit** |
| 4-bit | **37.6%** | ~92% | ❌ **Much worse than 1-bit** |

```
Test configuration:
- Dataset: LAION-400M-CLIP512 (hdf5, 100K vectors)
- Query count: 1000
- ef_search: 100
- rerank_factor: 10
- k: 10
```

#### Root Cause Analysis

**Problem:** Binary encoding + Hamming distance is fundamentally incompatible with multi-bit quantization.

For 2-bit quantization with 4 levels (0, 1, 2, 3):
- Binary encoding: 00, 01, 10, 11
- Level 1 (01) to Level 2 (10): Hamming distance = **2** (maximum!)
- Level 0 (00) to Level 3 (11): Hamming distance = **2** (maximum)
- Level 0 (00) to Level 1 (01): Hamming distance = 1
- Level 1 (01) to Level 3 (11): Hamming distance = 1

**Result:** Adjacent quantization levels have maximum Hamming distance, while distant levels have minimum distance. This inverts the distance semantics, causing recall to **decrease** as more bits are used.

#### Why 1-bit Works

1-bit quantization (sign-only) is mathematically sound because:
- Sign quantization directly captures the direction of each rotated component
- For unit vectors under Cosine distance, sign agreement approximates angular similarity
- Hamming distance on sign bits ≈ normalized angular distance

#### Proposed Fix: Gray Code Encoding

Use Gray code instead of binary encoding for multi-bit quantization:

| Level | Binary | Gray Code |
|-------|--------|-----------|
| 0 | 00 | 00 |
| 1 | 01 | 01 |
| 2 | 10 | **11** |
| 3 | 11 | **10** |

Gray code ensures adjacent levels differ by exactly 1 bit, preserving distance semantics.

**Implementation changes required:**
- `quantize_2bit()`: Map levels [0,1,2,3] → Gray codes [0b00, 0b01, 0b11, 0b10]
- `quantize_4bit()`: Map levels 0-15 → 4-bit Gray codes
- No changes to Hamming distance computation needed

#### Impact Assessment (Pre-Fix)

- **1-bit mode:** Unaffected, continues to work correctly
- **2-bit/4-bit modes:** Currently unusable, return worse recall than 1-bit
- **Documentation:** `API.md` updated with warning (commit pending)
- **GitHub Issue:** #43 tracks the fix

#### Resolution

**Date:** January 12, 2026
**Implemented by:** Claude

The fix adds Gray code encoding to `quantize_2bit()` and `quantize_4bit()` functions:

```rust
/// Convert binary value to Gray code.
/// Formula: gray = n ^ (n >> 1)
#[inline]
const fn to_gray_code(n: u8) -> u8 {
    n ^ (n >> 1)
}
```

**Changes:**
- Added `to_gray_code()` helper function in `rabitq.rs`
- Modified `quantize_2bit()` to use Gray code encoding
- Modified `quantize_4bit()` to use Gray code encoding
- Added unit tests: `test_gray_code_values`, `test_gray_code_adjacent_differ_by_one_bit`, `test_2bit_gray_code_distance_ordering`, `test_4bit_gray_code_distance_ordering`

**Benchmark Results (Post-Fix):**

| Bits | Recall@10 (rerank=10) | Before Fix | Status |
|------|----------------------|------------|--------|
| 1-bit | 20.0% | 50.8% | Baseline |
| 2-bit | **33.6%** | 45.4% | ✅ **Better than 1-bit** |
| 4-bit | **30.6%** | 37.6% | ✅ **Better than 1-bit** |

*Note: Absolute recall is lower due to random dataset (vs LAION-CLIP). Key metric is relative ordering.*

**Verification (Random Data):** Recall now **increases** with more bits (2-bit > 1-bit), confirming Gray code encoding is correct.

#### Extended Verification: LAION-CLIP 512D

| Bits | Recall@10 (rerank=10) | Status |
|------|----------------------|--------|
| 1-bit | **43.5%** | Baseline |
| 2-bit | 24.2% | ❌ Worse than 1-bit |
| 4-bit | 11.4% | ❌ Much worse than 1-bit |

**Key Insight:** Gray code fixes the **local** Hamming distance problem (adjacent levels differ by 1 bit), but multi-bit quantization + symmetric Hamming distance has a **fundamental limitation**:

| 4-bit Levels | Gray Code | Hamming Distance | Value Difference |
|--------------|-----------|------------------|------------------|
| 0 → 1 | 0000 → 0001 | 1 | 1 |
| 0 → 8 | 0000 → 1100 | 2 | 8 |
| 0 → 15 | 0000 → 1000 | **1** | **15** |

Distant quantization levels can have **lower** Hamming distance than nearby levels, corrupting the similarity measure. This explains why:
- **Random data** (uniformly distributed): Gray code helps
- **LAION-CLIP** (semantically clustered): Multi-bit makes recall worse

**Conclusion:**
- Gray code implementation is **correct** for the stated issue #43
- However, symmetric Hamming distance is **fundamentally unsuited** for multi-bit quantization
- **Recommendation:** Use 1-bit quantization for RaBitQ (mathematically sound: sign bits ≈ angular distance)
- For multi-bit precision gains, use **asymmetric distance** (ADC) or **product quantization** instead

#### References

- Issue #43: https://github.com/chungers/motlie/issues/43
- Gray code: https://en.wikipedia.org/wiki/Gray_code

### 4.3 Evaluation of Issue #43 (Gray Code)

**Reviewer:** Gemini Agent
**Date:** January 12, 2026

*   **Assessment:** **AGREE**. The claim is technically correct. Standard binary encoding fails to preserve local similarity in Hamming space for values > 1 bit.
    *   *Proof:* In standard binary, `1 (01)` and `2 (10)` have Hamming distance 2. In Gray code, `1 (01)` and `2 (11)` have Hamming distance 1.
    *   *Impact:* The observed degradation in recall for 2-bit/4-bit modes (vs 1-bit) is directly explained by this. The "precision" added by extra bits is effectively scrambling the distance metric for adjacent bins.
*   **Recommendation:** Implement Gray Code encoding immediately. This is a standard technique for binary quantization (e.g., used in ITQ and PQ).
*   **Validation Plan:**
    1.  **Unit Test:** Verify `hamming_distance(encode(i), encode(i+1)) == 1` for all adjacent integer levels $i$.
    2.  **Benchmark:** Re-run the `vector2` benchmark on Random-1024D (or SIFT). We expect 2-bit recall to significantly exceed 1-bit recall (should be >80% with re-ranking).

## 5. Summary of Recommendations

1.  **Implement Gray Code:** Fixes Issue #43 (Critical for 2/4-bit).
2.  **Tune RaBitQ:** Use the new benchmark infrastructure (GEMINI-BENCHMARK.md) to find optimal `rerank_factor` for the fixed 2/4-bit modes.
## 6. Gemini Review of RABITQ.md (ADC Findings)

**Date:** January 12, 2026

I have reviewed the new `RABITQ.md` document and independently verified its claims regarding Symmetric Hamming Distance vs. Asymmetric Distance Computation (ADC).

### 6.1 Summary of Findings

1.  **Symmetric Hamming is Flawed for Multi-bit:**
    *   While 1-bit (sign) Hamming distance works well as a proxy for angular distance, multi-bit Hamming distance (even with Gray codes) fails to preserve the metric properties required for accurate search.
    *   **Reason:** The distance between quantization levels in Hamming space does not correlate monotonically with their numeric difference (e.g., in 4-bit Gray code, levels 0 and 15 are numerically far but have Hamming distance 1).
    *   **Impact:** This explains why 2-bit/4-bit quantization previously yielded worse recall than 1-bit on structured data like LAION-CLIP.

2.  **ADC is the Correct Implementation:**
    *   **Asymmetric Distance Computation (ADC)** avoids binarizing the query vector. Instead, it computes the dot product between the *float32 query* (rotated) and the *reconstructed centroid* of the binary code.
    *   This preserves the full precision of the query and the correct numeric ordering of the quantized data levels.
    *   Independent research confirms this is the standard approach in Product Quantization (PQ) and the method used in the original RaBitQ paper.

3.  **Gray Code Implementation Verified:**
    *   The recent Gray Code implementation in `rabitq.rs` is **correct** (`n ^ (n >> 1)`).
    *   It solves the *local adjacency* problem but cannot solve the *global metric* problem inherent to Hamming distance.

### 6.2 Recommendations

1.  **Adopt ADC:** Proceed with the plan to implement ADC for RaBitQ. This is necessary to unlock the recall benefits of >1 bit quantization.
2.  **Benchmark Strategy:** Future benchmarks should focus on tuning `rerank_factor` with ADC, as it should provide much better candidate quality than Symmetric Hamming, potentially allowing for lower re-ranking costs.
3.  **Documentation:** `RABITQ.md` accurately reflects the state of the art and the path forward. See **Appendix C** in that document for the detailed independent review.

