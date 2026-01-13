# Gemini Code Review: Vector Subsystem

**Date:** January 12, 2026
**Scope:** `libs/db/src/vector`
**Focus:** Correctness of RaBitQ/ADC implementation and HNSW integration.

## 1. Summary of Findings

The vector subsystem has been successfully updated to implement **Asymmetric Distance Computation (ADC)** for RaBitQ, addressing the fundamental limitations of Symmetric Hamming distance for multi-bit quantization.

### 1.1 Correctness Verified
*   **RaBitQ Logic:** The implementation in `quantization/rabitq.rs` correctly computes:
    *   **Gray Codes:** `n ^ (n >> 1)` for local adjacency.
    *   **ADC Encoding:** `encode_with_correction` correctly calculates quantization error and vector norms.
    *   **ADC Distance:** `adc_distance` implements the estimator $\langle q, v \rangle \approx C \cdot \sum (Rq)_i \cdot \bar{v}_i$ via weighted dot products (1/2/4-bit) and applies the corrective factors ($\|v-c\|$ and quantization error) to estimate Euclidean/Cosine distance.
*   **HNSW Integration:** `hnsw/search.rs` uses `beam_search_layer0_adc_cached` which:
    1.  Rotates the query vector *once* (keeping it as `float32`).
    2.  Computes ADC distance to cached binary codes during graph traversal.
    3.  Avoids the "wraparound" metric error of Hamming distance.
*   **Cache:** `BinaryCodeCache` properly stores the `AdcCorrection` struct alongside the binary code.

### 1.2 Roadmap Status
*   **Phase 4 (RaBitQ):** ✅ **Complete.** The critical "ADC HNSW Integration" (Task 4.24) is implemented and verified.
*   **Documentation:** `RABITQ.md` status table has been updated to reflect this completion.

## 2. Code Quality & Design

*   **Modularization:** The separation of `quantization` logic from `hnsw` graph traversal is clean. The `RaBitQ` struct encapsulates all bit-twiddling and math.
*   **Performance:**
    *   **SIMD:** `binary_dot_product` uses manual loop unrolling/optimization. Future work could optimize this further with AVX-512 VNNI/VPOPCNT intrinsics, but the current implementation is solid.
    *   **Caching:** In-memory caching of binary codes + corrections eliminates disk I/O during the high-volume candidate filtering phase.
*   **Safety:** Configuration validation ensures RaBitQ is only used with Cosine distance (as required by the rotation math).

## 3. Remaining Risks & Next Steps

*   **Write Path:** As noted in previous reviews, `InsertVector` still needs to be connected to the HNSW graph updater (Phase 5). Currently, the index is static after initial bulk load/test.
*   **Persistence:** Verify that `AdcCorrection` factors are correctly serialized/deserialized from RocksDB (CF `BinaryCodes`) on restart.
*   **Tuning:** While ADC is correct, optimal `rerank_factor` and `ef_search` values for 4-bit ADC on 1M+ datasets need empirical tuning using the new benchmark infrastructure.

## 4. Conclusion

## 5. Future Optimization: SIMD for Quantized Dot Products

**Review:** The current `binary_dot_product` implementations in `rabitq.rs` use manual loops. While the 1-bit version is optimized algebraically, significant performance gains can be achieved by moving these mixed-precision operations into `motlie_core::distance` where they can leverage AVX2/AVX-512/NEON intrinsics.

### 5.1 Proposed Design (`motlie_core::distance::quantized`)

Create a new module `libs/core/src/distance/quantized.rs` exposing hardware-accelerated primitives for computing dot products between high-precision float vectors and low-precision packed binary codes.

```rust
pub mod quantized {
    /// Compute dot product between float vector and 1-bit packed binary code.
    /// Corresponds to: sum(q[i] * (if bit[i]==1 { 1.0 } else { -1.0 }))
    /// Optimization: 2.0 * sum(q where bit=1) - sum(q)
    pub fn dot_1bit(query: &[f32], code: &[u8]) -> f32;

    /// Compute dot product between float vector and 2-bit packed code (Gray coded).
    /// `values` maps the 4 possible 2-bit levels (0..3) to float weights.
    /// E.g. for RaBitQ: [-1.5, -0.5, 0.5, 1.5]
    pub fn dot_2bit_lookup(query: &[f32], code: &[u8], values: &[f32; 4]) -> f32;

    /// Compute dot product between float vector and 4-bit packed code (Gray coded).
    /// Since RaBitQ uses a linear mapping for 4-bit, a scale/offset version is efficient.
    /// value = (level * scale) + offset
    pub fn dot_4bit_linear(query: &[f32], code: &[u8], scale: f32, offset: f32) -> f32;
}
```

### 5.2 Implementation Strategy

*   **1-bit (`dot_1bit`):**
    *   **AVX2:** Broadcast `u8` bits to `__m256` mask. Use `vmaskmovps` or bitwise AND to select elements, then FMA.
    *   **NEON:** Similar approach with 128-bit vectors.
*   **2-bit (`dot_2bit_lookup`):**
    *   **AVX2:** Expand packed 2-bit indices to 32-bit integers. Use `vpermps` (shuffle) to look up the float value from the 4-element table stored in a register. FMA with query.
*   **4-bit (`dot_4bit_linear`):**
    *   **AVX2:** Expand packed nibbles to 32-bit integers. Convert to float (`vcvtdq2ps`). Apply scale/offset (`vfmadd213ps`). FMA with query.

### 5.3 Migration Plan

1.  **Implement `motlie_core` primitives:** Add the module and SIMD implementations (with scalar fallback) to `libs/core`.
2.  **Verify Correctness:** Add unit tests in `motlie_core` matching the logic in `rabitq.rs`.
3.  **Refactor `rabitq.rs`:** Replace manual loops with calls to `motlie_core`.

**Example Migration (`binary_dot_2bit`):**

```rust
// Current (libs/db/src/vector/quantization/rabitq.rs)
fn binary_dot_2bit(&self, query: &[f32], code: &[u8]) -> f32 {
    const LEVEL_VALUES: [f32; 4] = [-1.5, -0.5, 0.5, 1.5];
    let mut sum = 0.0f32;
    // ... manual loop decoding Gray code ...
    sum
}

// Proposed
fn binary_dot_2bit(&self, query: &[f32], code: &[u8]) -> f32 {
    const LEVEL_VALUES: [f32; 4] = [-1.5, -0.5, 0.5, 1.5];
    motlie_core::distance::quantized::dot_2bit_lookup(query, code, &LEVEL_VALUES)
}
```
