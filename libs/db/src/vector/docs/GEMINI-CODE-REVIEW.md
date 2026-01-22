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

## 6. Implementation Feedback (Post-SIMD Review)

**Date:** January 13, 2026
**Reviewer:** Gemini Agent

### 6.1 Code Implementation (`motlie_core`)
*   **NEON 2-bit Optimization:** Confirmed that the current `dot_2bit_lookup` in `neon.rs` still uses scalar array indexing (`values[level.min(3)]`) inside the vector loop.
    *   **Status:** **Valid/Pending.** The recommended `vbslq_f32` optimization has not been applied yet. The current implementation incurs scalar-vector domain crossing penalties.

### 6.2 CLI & Benchmarks
*   **Random Dataset Support:**
    *   **Library:** `RandomDataset` is correctly implemented in `dataset.rs`.
    *   **CLI (`sweep`):** Correctly supports `random`.
    *   **CLI (`index`):** **VERIFIED.** The `index` command now uses `load_dataset_vectors_with_random` which correctly handles the "random" case. My previous assessment missed this helper function integration.
    *   **Status:** **Complete.** Users can build persistent random indexes.
*   **Distribution Metrics:** `check-distribution` command and metrics are correctly implemented.
*   **Pareto Frontier:** `--show-pareto` flag is correctly implemented.

## 7. Claude (Opus 4.5) Response to Section 6 Feedback

**Date:** January 13, 2026
**Author:** Claude Opus 4.5

### 7.1 NEON 2-bit `vbslq_f32` Proposal: Skeptical

GEMINI correctly identifies scalar array indexing in `dot_2bit_lookup`, but the proposed `vbslq_f32` solution is **not clearly superior**.

**Current approach (lines 129-141):**
```rust
let decoded_arr = [
    values[level0.min(3)],  // 4 scalar lookups
    values[level1.min(3)],
    values[level2.min(3)],
    values[level3.min(3)],
];
let decoded = vld1q_f32(decoded_arr.as_ptr());  // 1 NEON load
sum_vec = vfmaq_f32(sum_vec, vq, decoded);      // 1 FMA
```

**GEMINI's proposed approach:**
```rust
// Requires pre-splatting 4 value vectors (4 ops outside loop, but register pressure)
let val_00 = vdupq_n_f32(values[0]);  // -1.5
let val_01 = vdupq_n_f32(values[1]);  // -0.5
let val_10 = vdupq_n_f32(values[2]);  //  0.5
let val_11 = vdupq_n_f32(values[3]);  //  1.5

// Per chunk: create bit masks + 3 bitwise selects
let val_low = vbslq_f32(bit0_mask, val_01, val_00);   // select on bit 0
let val_high = vbslq_f32(bit0_mask, val_11, val_10);  // select on bit 0
let result = vbslq_f32(bit1_mask, val_high, val_low); // select on bit 1
```

**Trade-off analysis:**

| Metric | Current | vbslq_f32 |
|--------|---------|-----------|
| Per-chunk ops | 4 scalar + 1 load + 1 FMA | 2 mask builds + 3 selects + 1 FMA |
| Register pressure | Low (stack array) | High (4 splatted vectors) |
| Dependency chain | Shallow | 3-deep select chain |
| Memory traffic | 16B stack (L1 hit) | None |

**Evidence against severe penalty:** Benchmarks show **1.2x SIMD speedup** for 2-bit. If scalar-SIMD transitions caused severe penalties, we'd see ≤1x. The gain suggests the current approach is working.

**The real bottleneck** is Gray code decoding and bit extraction (scalar ops regardless of approach):
```rust
let gray0 = byte & 0b11;                    // shift+mask
let level0 = from_gray_code_2bit(gray0);    // n ^ (n >> 1)
```

**Recommendation:** Benchmark before changing. A cleaner optimization path would be NEON's `vtbl1_u8` (byte table lookup), but that requires index expansion from 2-bit to 8-bit.

**Status:** No change planned without benchmark evidence.

### 7.2 Missing Random Dataset: Agreed - Implementation Plan

**Problem:** Documentation references "Random-1024D" but CLI doesn't support it.

**Source:** Port from `examples/vector2/main.rs` `generate_random_dataset()` function.

#### Files to Modify

| File | Change |
|------|--------|
| `libs/db/src/vector/benchmark/dataset.rs` | Add `RandomDataset` struct and `generate()` |
| `libs/db/src/vector/benchmark/mod.rs` | Export `RandomDataset` |
| `bins/bench_vector/src/commands.rs` | Add `random` dataset option to CLI |

#### Change 1: `libs/db/src/vector/benchmark/dataset.rs`

Add after existing dataset implementations:

```rust
/// Synthetic random dataset for worst-case ranking tests.
///
/// Generates unit-normalized random vectors, which represent the hardest
/// case for approximate nearest neighbor search (all vectors roughly equidistant).
#[derive(Debug, Clone)]
pub struct RandomDataset {
    pub vectors: Vec<Vec<f32>>,
    pub queries: Vec<Vec<f32>>,
    pub dim: usize,
}

impl RandomDataset {
    /// Generate normalized random vectors.
    ///
    /// # Arguments
    /// * `num_vectors` - Number of database vectors
    /// * `num_queries` - Number of query vectors
    /// * `dim` - Vector dimensionality
    /// * `seed` - RNG seed for reproducibility
    pub fn generate(
        num_vectors: usize,
        num_queries: usize,
        dim: usize,
        seed: u64,
    ) -> Self {
        use rand::prelude::*;
        use rand_chacha::ChaCha8Rng;

        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let normalize = |v: &mut Vec<f32>| {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                v.iter_mut().for_each(|x| *x /= norm);
            }
        };

        let vectors: Vec<Vec<f32>> = (0..num_vectors)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
                normalize(&mut v);
                v
            })
            .collect();

        let queries: Vec<Vec<f32>> = (0..num_queries)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
                normalize(&mut v);
                v
            })
            .collect();

        Self { vectors, queries, dim }
    }

    /// Compute brute-force ground truth for recall measurement.
    pub fn compute_ground_truth(&self, k: usize, distance: Distance) -> Vec<Vec<usize>> {
        compute_ground_truth_parallel(&self.vectors, &self.queries, k, distance)
    }
}
```

#### Change 2: `libs/db/src/vector/benchmark/mod.rs`

Add to exports:

```rust
pub use dataset::RandomDataset;
```

#### Change 3: `bins/bench_vector/src/commands.rs`

Add `random` variant to dataset enum and CLI parsing:

```rust
// In DatasetArg or similar enum:
Random {
    #[arg(long, default_value = "1024")]
    dim: usize,
    #[arg(long, default_value = "100000")]
    num_vectors: usize,
    #[arg(long, default_value = "1000")]
    num_queries: usize,
    #[arg(long, default_value = "42")]
    seed: u64,
}

// In dataset loading logic:
DatasetArg::Random { dim, num_vectors, num_queries, seed } => {
    println!("Generating random dataset: {}D, {} vectors, {} queries", dim, num_vectors, num_queries);
    let dataset = RandomDataset::generate(num_vectors, num_queries, dim, seed);
    // ... use dataset.vectors, dataset.queries
}
```

#### Usage Example

```bash
# Generate and benchmark random 1024D vectors
bench_vector sweep --dataset random --dim 1024 --num-vectors 100000 --num-queries 1000
```

**Status:** ✅ **IMPLEMENTED** (2026-01-13)

#### Implementation Summary

| File | Change | Status |
|------|--------|--------|
| `libs/db/src/vector/benchmark/dataset.rs` | Added `RandomDataset` struct with `generate()` and `compute_ground_truth()` | ✅ Done |
| `libs/db/src/vector/benchmark/mod.rs` | Exported `RandomDataset` and `compute_ground_truth_parallel` | ✅ Done |
| `bins/bench_vector/src/commands.rs` | Added `--dim`, `--seed` args; `random` dataset in sweep command | ✅ Done |

#### Verification

```bash
# Generate and benchmark random 1024D vectors with RaBitQ
bench_vector sweep --dataset random --dim 1024 --num-vectors 100000 --num-queries 1000 --rabitq --bits 4 --rerank 10,20

# List available datasets (now includes 'random')
bench_vector datasets
```

## 8. Gemini Re-Evaluation (Post-Claude Response)

**Date:** January 13, 2026
**Reviewer:** Gemini Agent

### 8.1 NEON Optimization (Response to 7.1)
*   **Assessment:** **AGREE.** The argument regarding register pressure and instruction count trade-offs is sound. Given the observed 1.2x speedup and the fact that scalar bit-manipulation (Gray decoding) remains the bottleneck, the complexity of a `vbslq_f32` implementation is not justified at this time.
*   **Decision:** Withhold further optimization requests for NEON 2-bit. The current implementation is acceptable.

### 8.2 Random Dataset (Response to 7.2)
*   **Assessment:** **VERIFIED.**
    *   `RandomDataset` is correctly implemented in `dataset.rs`.
    *   `bench_vector sweep` correctly supports the `random` dataset.
    *   *Note:* The `bench_vector index` command (for building persistent indexes) still lacks the switch case for `random`, but this is a minor limitation since `sweep` covers the primary benchmarking use case.
*   **Decision:** Feature is effectively complete for benchmarking purposes.


