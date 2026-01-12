# RaBitQ Implementation Guide

## Overview

RaBitQ (Random Bit Quantization) is a training-free binary quantization method for approximate nearest neighbor search. This document covers our implementation, research findings, and design decisions.

**Key Properties:**
- Training-free: Uses random orthonormal rotation (DATA-1 compliant)
- Incremental: Works with streaming inserts
- Theoretical bound: O(1/sqrt(D)) error for any data distribution

**References:**
- [RaBitQ Paper (SIGMOD 2024)](https://arxiv.org/abs/2405.12497)
- [Elasticsearch RaBitQ 101](https://www.elastic.co/search-labs/blog/rabitq-explainer-101)
- [LanceDB RaBitQ](https://lancedb.com/blog/feature-rabitq-quantization/)
- [GitHub Reference Implementation](https://github.com/gaoj0017/RaBitQ)

---

## Current Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Random rotation matrix | ✅ Complete | √D scaled for unit vectors (issue #42) |
| 1-bit quantization | ✅ Complete | Sign quantization, works well |
| 2-bit/4-bit quantization | ✅ Complete | Gray code encoding (issue #43) |
| Symmetric Hamming distance | ✅ Complete | Current search method |
| Asymmetric Distance (ADC) | ❌ Not implemented | **This document's focus** |
| Corrective factors storage | ❌ Not implemented | Required for ADC |

---

## Part 1: Research Findings

### 1.1 Symmetric Hamming Distance Limitations

Our current implementation uses **symmetric Hamming distance**:
- Both query and data vectors are encoded to binary
- Distance = popcount(query_code XOR data_code)

**Problem discovered (Issue #43):** Multi-bit quantization performs *worse* than 1-bit on real-world data.

| Dataset | 1-bit | 2-bit | 4-bit |
|---------|-------|-------|-------|
| Random 10K | 14.2% | **21.5%** | **22.2%** |
| LAION-CLIP 50K | **43.5%** | 24.2% | 11.4% |

**Root Cause:** Even with Gray code, distant quantization levels can have lower Hamming distance than nearby levels:

```
4-bit Gray code examples:
Level 0  → Level 1:  Hamming = 1, Value diff = 1   ✓
Level 0  → Level 8:  Hamming = 2, Value diff = 8
Level 0  → Level 15: Hamming = 1, Value diff = 15  ✗ (wraparound)
```

This corrupts similarity on structured/clustered data (semantic embeddings).

### 1.2 How RaBitQ Actually Works

The RaBitQ paper does **NOT** use symmetric Hamming distance. It uses **Asymmetric Distance Computation (ADC)** with corrective factors.

**Key insight from [Elasticsearch explainer](https://www.elastic.co/search-labs/blog/rabitq-explainer-101):**

> "RaBitQ does **not use Hamming distance**. Instead, it employs a sophisticated approximation technique based on corrective factors."

#### The Actual RaBitQ Formula

For estimating distance between query `q` and stored vector `v`:

```
dist(v, q) = √[||v-c||² + ||q-c||² - 2×||v-c||×||q-c||×⟨q,v⟩_estimated]
```

Where:
- `c` = centroid (can be origin for single-centroid mode)
- `||v-c||` = stored corrective factor #1 (float32)
- `⟨v_q, v_n⟩` = stored corrective factor #2 (float32)

**Inner product estimation (asymmetric):**
```
⟨q, v⟩ ≈ (q_rotated · v̄) / (v_correction · v̄)
```

Where:
- `q_rotated` = rotated query (float32, NOT binarized)
- `v̄` = binary code of stored vector
- `v_correction` = stored dot product correction

### 1.3 Industry Implementations

| System | Approach | Multi-bit |
|--------|----------|-----------|
| [Elasticsearch](https://www.elastic.co/search-labs/blog/rabitq-explainer-101) | Full ADC with corrective factors | 1-bit only |
| [LanceDB](https://lancedb.com/blog/feature-rabitq-quantization/) | ADC with 2 corrective factors | 1-bit only ("extended-RaBitQ not yet available") |
| [Qdrant](https://qdrant.tech/articles/binary-quantization/) | Symmetric Hamming + aggressive oversampling | 1-bit only |
| **Our current impl** | Symmetric Hamming + reranking | 1/2/4-bit (2/4-bit broken on real data) |

**Key observation:** Major vector databases ship 1-bit only. Multi-bit is research-only.

### 1.4 Alternative Approaches Considered

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **True ADC** | Store corrective factors, use formula | Proper RaBitQ, >95% recall | +8 bytes/vector |
| **Weighted Hamming** | Per-bit weights | Better than uniform | Fundamentally limited |
| **Product Quantization** | Segment vectors, codebook | Well-proven | Training required (violates DATA-1) |
| **1-bit + aggressive rerank** | Current, increase rerank_factor | Simple, works | Lower recall ceiling |

### 1.5 Why ADC Solves the Multi-bit Problem

The multi-bit Hamming problem exists because symmetric Hamming distance doesn't preserve numeric ordering of quantization levels. **ADC avoids this entirely by not using Hamming distance.**

#### The Fundamental Difference

| Approach | Query | Data | Distance Computation |
|----------|-------|------|---------------------|
| Symmetric Hamming | Binary | Binary | `popcount(query XOR data)` |
| ADC | **Float32** | Binary + corrections | `weighted_dot_product(query, data)` |

#### How ADC Computes Similarity

In ADC, the query vector is **rotated but never binarized**. Distance is computed as a weighted dot product:

```
dot_product = Σ query[i] × decode(level[i])
```

Where `decode(level)` returns the actual numeric value:
- 2-bit: levels 0,1,2,3 → values -1.5, -0.5, +0.5, +1.5
- 4-bit: levels 0-15 → values -2.0 to +2.0 linearly

#### Concrete Example

**ADC (correct):**
```
Query component: 0.3 (float, NOT quantized)

Data A: level 2 (value +0.5) → contribution = 0.3 × 0.5  = +0.15
Data B: level 0 (value -1.5) → contribution = 0.3 × -1.5 = -0.45
Data C: level 3 (value +1.5) → contribution = 0.3 × 1.5  = +0.45

Similarity order: C > A > B  ✓ (correct - matches level ordering)
```

**Symmetric Hamming (broken):**
```
Query quantized to level 1 (Gray code: 01)

Data A: level 2 (Gray: 11) → Hamming = 1
Data B: level 0 (Gray: 00) → Hamming = 1  ← SAME distance!
Data C: level 3 (Gray: 10) → Hamming = 2

Problem: A and B have same Hamming distance, but level 2 is much closer to level 1!
```

#### The Gray Code Wraparound Problem

With 4-bit Gray code:
- Level 0 (0000) to Level 15 (1000): Hamming = 1
- Level 0 (0000) to Level 1 (0001): Hamming = 1

Symmetric Hamming treats these as equally similar, but they're at opposite ends of the value range!

**ADC avoids this because:**
```
Query component: 0.5

Level 0  (value -2.0): contribution = 0.5 × -2.0 = -1.0
Level 15 (value +2.0): contribution = 0.5 × +2.0 = +1.0
Level 1  (value -1.73): contribution = 0.5 × -1.73 = -0.87

Result: Level 15 >> Level 1 > Level 0  ✓ (correct ordering preserved)
```

#### Summary

| Property | Symmetric Hamming | ADC |
|----------|-------------------|-----|
| Query encoding | Binarized | Float32 (rotated only) |
| Distance metric | XOR + popcount | Weighted dot product |
| Level ordering | Broken (Gray wraparound) | Preserved (numeric values) |
| Multi-bit support | ❌ Degrades recall | ✅ Improves recall |

**ADC works because it never loses the numeric relationship between quantization levels.**

---

## Part 2: Design Proposal

### 2.1 Goals

1. **Support both modes:** Symmetric Hamming (simple/fast) and ADC (accurate)
2. **Configurable at index build time:** Mode determines what's stored
3. **Configurable at search time:** Can use exact search regardless of mode
4. **Backward compatible:** Existing indexes work (default to Symmetric)

### 2.2 RaBitQ Mode Enum

```rust
/// RaBitQ distance computation mode.
///
/// Determines how binary codes are used during search and what
/// additional data is stored per vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RaBitQMode {
    /// Symmetric Hamming distance (current implementation).
    ///
    /// - Query and data both encoded to binary
    /// - Distance = popcount(query XOR data)
    /// - Fast, simple, no extra storage
    /// - **Only works well with 1-bit** (multi-bit broken on real data)
    ///
    /// Storage: binary code only
    #[default]
    Symmetric,

    /// Asymmetric Distance Computation (true RaBitQ).
    ///
    /// - Query remains float32 (rotated but not binarized)
    /// - Data stored as binary + 2 corrective factors
    /// - Uses inner product estimation formula
    /// - Works well with 1/2/4-bit quantization
    ///
    /// Storage: binary code + 2×f32 (8 bytes extra per vector)
    ADC,
}
```

### 2.3 Configuration Changes

#### RaBitQConfig (Index Build Time)

```rust
/// RaBitQ binary quantization parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RaBitQConfig {
    /// Bits per dimension (1, 2, or 4).
    pub bits_per_dim: u8,

    /// Seed for rotation matrix generation.
    pub rotation_seed: u64,

    /// Enable RaBitQ quantization.
    pub enabled: bool,

    /// Distance computation mode (NEW).
    ///
    /// - `Symmetric`: Fast Hamming, only use with 1-bit
    /// - `ADC`: True RaBitQ with corrective factors, works with multi-bit
    ///
    /// Default: `Symmetric` for backward compatibility.
    pub mode: RaBitQMode,
}
```

**Validation rules:**
- `mode == Symmetric && bits_per_dim > 1` → Warning (known to degrade recall)
- `mode == ADC && bits_per_dim == 1` → Valid but unnecessary overhead

#### SearchStrategy (Search Time)

```rust
/// Search strategy for RaBitQ.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchStrategy {
    /// Exact distance computation.
    Exact,

    /// RaBitQ with mode-aware distance computation.
    RaBitQ {
        /// Use in-memory binary code cache.
        use_cache: bool,

        /// Distance mode (must match index mode for ADC).
        mode: RaBitQMode,
    },
}
```

### 2.4 Storage Schema Changes

#### Current BinaryCodeCfValue

```rust
// Current: binary code only
pub struct BinaryCodeCfValue(pub RabitqCode);  // Vec<u8>
```

#### Proposed BinaryCodeCfValue

```rust
/// RaBitQ stored data for a vector.
///
/// Layout depends on RaBitQMode:
/// - Symmetric: binary code only
/// - ADC: binary code + corrective factors
#[derive(Debug, Clone)]
pub struct BinaryCodeCfValue {
    /// Binary quantized code.
    pub code: RabitqCode,

    /// Corrective factors for ADC mode (None for Symmetric).
    pub correction: Option<AdcCorrection>,
}

/// ADC corrective factors (8 bytes total).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AdcCorrection {
    /// ||v - c|| : distance from vector to centroid
    pub norm_to_centroid: f32,

    /// ⟨v_quantized, v_normalized⟩ : dot product correction
    pub quantization_error: f32,
}
```

**Storage overhead:**
| Mode | 128D 1-bit | 128D 2-bit | 128D 4-bit |
|------|------------|------------|------------|
| Symmetric | 16 bytes | 32 bytes | 64 bytes |
| ADC | 24 bytes | 40 bytes | 72 bytes |

### 2.5 RaBitQ Encoder Changes

```rust
impl RaBitQ {
    /// Encode vector to binary code (current behavior).
    pub fn encode(&self, vector: &[f32]) -> Vec<u8>;

    /// Encode vector with ADC corrective factors (NEW).
    ///
    /// Returns binary code and correction factors for ADC mode.
    pub fn encode_with_correction(
        &self,
        vector: &[f32],
        centroid: &[f32],  // Can be zero vector for single-centroid
    ) -> (Vec<u8>, AdcCorrection) {
        let rotated = self.rotate(vector);
        let code = self.quantize(&rotated);

        // Compute corrective factors
        let v_normalized = normalize(vector);
        let v_centered = subtract(vector, centroid);
        let norm_to_centroid = l2_norm(&v_centered);

        // Quantization error: how much the quantized code deviates
        let v_quantized_reconstructed = self.decode_approximate(&code);
        let quantization_error = dot(&v_quantized_reconstructed, &v_normalized);

        (code, AdcCorrection { norm_to_centroid, quantization_error })
    }

    /// Compute ADC distance estimate (NEW).
    ///
    /// Uses the RaBitQ formula:
    /// dist ≈ √[||v-c||² + ||q-c||² - 2×||v-c||×||q-c||×⟨q,v⟩_est]
    pub fn adc_distance(
        &self,
        query_rotated: &[f32],     // Rotated query (NOT binarized)
        data_code: &[u8],          // Binary code
        data_correction: &AdcCorrection,
        query_norm_to_centroid: f32,
    ) -> f32 {
        // Compute binary dot product (query_rotated · decode(data_code))
        let binary_dot = self.binary_dot_product(query_rotated, data_code);

        // Estimate inner product using correction
        let inner_product_est = binary_dot / data_correction.quantization_error;

        // Apply distance formula
        let v_norm = data_correction.norm_to_centroid;
        let q_norm = query_norm_to_centroid;

        let dist_sq = v_norm * v_norm + q_norm * q_norm
                    - 2.0 * v_norm * q_norm * inner_product_est;

        dist_sq.max(0.0).sqrt()
    }

    /// Compute dot product between float query and binary code.
    ///
    /// This is the key ADC operation: query is NOT binarized.
    fn binary_dot_product(&self, query: &[f32], code: &[u8]) -> f32 {
        // For 1-bit: sum query[i] where code bit i is set, subtract where not set
        // For 2/4-bit: weight by quantization level
        match self.bits_per_dim {
            1 => self.binary_dot_1bit(query, code),
            2 => self.binary_dot_2bit(query, code),
            4 => self.binary_dot_4bit(query, code),
            _ => unreachable!(),
        }
    }
}
```

### 2.6 Search Pipeline Changes

```rust
impl HnswIndex {
    pub fn search_rabitq(
        &self,
        query: &[f32],
        config: &SearchConfig,
    ) -> Result<Vec<(f32, Id)>> {
        match config.strategy() {
            SearchStrategy::RaBitQ { mode: RaBitQMode::Symmetric, .. } => {
                self.search_symmetric_hamming(query, config)
            }
            SearchStrategy::RaBitQ { mode: RaBitQMode::ADC, .. } => {
                self.search_adc(query, config)
            }
            SearchStrategy::Exact => {
                self.search_exact(query, config)
            }
        }
    }

    /// Symmetric Hamming search (current implementation).
    fn search_symmetric_hamming(&self, query: &[f32], config: &SearchConfig) -> Result<...> {
        // 1. Encode query to binary
        let query_code = self.rabitq.encode(query);

        // 2. HNSW navigation using Hamming distance
        let candidates = self.hnsw_search_hamming(&query_code, config.ef())?;

        // 3. Re-rank with exact distance
        self.rerank_exact(query, candidates, config)
    }

    /// ADC search (NEW - true RaBitQ).
    fn search_adc(&self, query: &[f32], config: &SearchConfig) -> Result<...> {
        // 1. Rotate query (but don't binarize!)
        let query_rotated = self.rabitq.rotate(query);
        let query_norm = l2_norm(&subtract(query, &self.centroid));

        // 2. HNSW navigation using ADC distance
        let candidates = self.hnsw_search_adc(
            &query_rotated,
            query_norm,
            config.ef()
        )?;

        // 3. Re-rank with exact distance
        self.rerank_exact(query, candidates, config)
    }
}
```

### 2.7 Migration Path

**Phase 1: Add ADC infrastructure (no breaking changes)**
1. Add `RaBitQMode` enum with `Default = Symmetric`
2. Add `mode` field to `RaBitQConfig`
3. Add `AdcCorrection` struct
4. Extend `BinaryCodeCfValue` to optionally store correction
5. Add `encode_with_correction()` to RaBitQ

**Phase 2: Implement ADC distance computation**
1. Add `binary_dot_product()` for 1/2/4-bit
2. Add `adc_distance()` method
3. Add `search_adc()` to search pipeline
4. Update `SearchStrategy` to include mode

**Phase 3: Testing and tuning**
1. Benchmark ADC vs Symmetric on LAION-CLIP
2. Tune ADC parameters (centroid, correction formula)
3. Document performance characteristics

### 2.8 API Examples

```rust
// Option A: Simple 1-bit with symmetric Hamming (current default)
let config = VectorConfig {
    rabitq: RaBitQConfig {
        bits_per_dim: 1,
        mode: RaBitQMode::Symmetric,  // Default
        ..Default::default()
    },
    ..Default::default()
};

// Option B: Multi-bit with ADC (true RaBitQ)
let config = VectorConfig {
    rabitq: RaBitQConfig {
        bits_per_dim: 4,              // 4-bit for higher precision
        mode: RaBitQMode::ADC,        // Required for multi-bit to work
        ..Default::default()
    },
    ..Default::default()
};

// Search with mode auto-detection
let search_config = SearchConfig::new(embedding.clone(), 10);
// Mode is inferred from index metadata

// Force exact search (ignores RaBitQ mode)
let search_config = SearchConfig::new(embedding.clone(), 10).exact();
```

---

## Part 3: Recommendations

### 3.1 Short-term (Current Release)

1. **Use 1-bit Symmetric mode** (default, already works well)
2. **Document multi-bit limitations** in API.md (done)
3. **Increase rerank_factor** for higher recall (10 → 50 for >90%)

### 3.2 Medium-term (Next Phase)

1. **Implement ADC mode** per this design
2. **Benchmark ADC vs Symmetric** on real datasets
3. **Enable multi-bit** only with ADC mode

### 3.3 Long-term Considerations

1. **Product Quantization**: If DATA-1 compliance relaxed, PQ offers better compression
2. **Multi-centroid**: For >100M vectors, partition by centroids
3. **Hardware acceleration**: AVX-512 VPOPCNT for Hamming, VNNI for ADC

---

## Appendix A: ADC Binary Dot Product Implementation

For ADC mode, we need to compute the dot product between a float32 query and a binary code without fully decoding the binary code.

### 1-bit Binary Dot Product

```rust
/// Compute dot product: query · decode_1bit(code)
///
/// For 1-bit, decode maps: bit=0 → -1, bit=1 → +1
/// So: dot = sum(query[i] * sign(bit[i]))
///         = sum(query where bit=1) - sum(query where bit=0)
fn binary_dot_1bit(&self, query: &[f32], code: &[u8]) -> f32 {
    let mut positive_sum = 0.0f32;
    let mut negative_sum = 0.0f32;

    for (i, &q) in query.iter().enumerate() {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        if (code[byte_idx] >> bit_idx) & 1 == 1 {
            positive_sum += q;
        } else {
            negative_sum += q;
        }
    }

    positive_sum - negative_sum
}
```

### 2-bit/4-bit Binary Dot Product

```rust
/// For multi-bit, we need to decode the Gray code level and multiply.
///
/// 2-bit levels (Gray decoded): 0, 1, 2, 3 → values: -1.5, -0.5, 0.5, 1.5
/// 4-bit levels: 0..15 → values: -2.0 to +2.0 linearly
fn binary_dot_2bit(&self, query: &[f32], code: &[u8]) -> f32 {
    let level_values = [-1.5, -0.5, 0.5, 1.5];
    let mut sum = 0.0f32;

    for (i, &q) in query.iter().enumerate() {
        let bit_offset = i * 2;
        let byte_idx = bit_offset / 8;
        let shift = bit_offset % 8;
        let gray = (code[byte_idx] >> shift) & 0b11;
        let level = from_gray_code(gray);  // Decode Gray → binary
        sum += q * level_values[level as usize];
    }

    sum
}
```

---

## Part 4: Should ADC Replace Symmetric Hamming?

If ADC is validated to work well, could we simplify the codebase by removing symmetric Hamming entirely?

### 4.1 Performance Comparison

#### Candidate Selection Phase (per vector, 512D)

| Operation | Symmetric Hamming | ADC |
|-----------|-------------------|-----|
| Data loaded | 64 bytes (1-bit code) | 72 bytes (code + 2 floats) |
| Computation | XOR + popcount | 512 float multiply-adds |
| SIMD efficiency | Excellent (VPOPCNT) | Good (FMA) |
| Instructions | ~4-8 (AVX-512) | ~32-64 (AVX-512) |

**Rough estimate:** Symmetric Hamming is **5-10x faster** for candidate filtering.

#### The Reranking Tradeoff

ADC produces **better candidates**, requiring fewer of them:

| Mode | Candidates for 95% recall@10 | Reranking cost |
|------|------------------------------|----------------|
| Symmetric 1-bit | ~500 (high rerank_factor) | High |
| ADC 1-bit | ~100 (lower rerank_factor) | Low |
| ADC 4-bit | ~50 (even better candidates) | Very low |

Since reranking (exact distance) often dominates latency, better candidates can offset slower filtering.

### 4.2 Pros of Single ADC Codepath

| Pro | Impact |
|-----|--------|
| **Simpler codebase** | One distance function, easier to maintain/test |
| **Consistent behavior** | All bit depths work identically |
| **Multi-bit works** | 2-bit/4-bit become real options |
| **Higher recall ceiling** | Better candidates → better results |
| **Easier to explain** | No "use 1-bit only" caveats in docs |
| **Future-proof** | Aligns with industry direction |

### 4.3 Cons of Removing Symmetric Hamming

| Con | Impact |
|-----|--------|
| **Slower candidate filtering** | 5-10x more FLOPs per candidate |
| **+8 bytes storage per vector** | 100M vectors = 800MB extra |
| **More memory bandwidth** | Must load corrections for every candidate |
| **Hamming is embarrassingly parallel** | XOR+popcount perfect for SIMD batch |
| **Breaking change** | Existing indexes need migration |

### 4.4 1-bit ADC Optimization

For 1-bit specifically, ADC can be optimized significantly:

```rust
// Naive: 512 branches + multiply-adds
dot = Σ query[i] * (bit[i] ? +1 : -1)

// Optimized: precompute query_sum once per query
// Then just sum query elements where bit=1
dot = 2 * Σ(query where bit=1) - query_sum

// SIMD: use masked load/add with bitmask
dot = 2 * masked_sum_simd(query, bitmask) - query_sum
```

This reduces the gap considerably, but Hamming likely remains faster.

### 4.5 Recommendation

**Keep both, but simplify the mental model:**

```rust
pub enum RaBitQMode {
    /// Fast mode: 1-bit symmetric Hamming.
    /// Maximum QPS, good recall with high rerank_factor.
    /// Best for: latency-sensitive applications.
    Fast,

    /// Accurate mode: ADC with corrective factors.
    /// Works with 1/2/4-bit, better candidates, less reranking needed.
    /// Best for: recall-sensitive applications.
    Accurate,
}
```

**Default behavior:**
- `bits_per_dim == 1` → Auto-select `Fast` (backward compatible)
- `bits_per_dim > 1` → Require `Accurate` (multi-bit needs ADC)

**If forced to choose ONE:**

Choose **ADC** because:
1. Works correctly for all bit depths
2. Simpler user experience (no caveats)
3. Performance gap acceptable after SIMD optimization
4. Industry trending toward accuracy over raw filtering speed

But **benchmark first** before deciding.

### 4.6 Validation Plan

Before making a final decision:

1. **Implement ADC** with SIMD-optimized 1-bit dot product
2. **Benchmark on LAION-CLIP 100K:**
   - Measure QPS at fixed 95% recall (vary rerank_factor)
   - Compare memory bandwidth utilization
   - Profile candidate filtering vs reranking time
3. **Decision criteria:**
   - ADC within 2x of Hamming at same recall → **Drop Hamming** (simplicity wins)
   - ADC 3x+ slower at same recall → **Keep both** with clear guidance
4. **If keeping both:** Document when to use each mode clearly

### 4.7 Migration Path

If we decide to deprecate Symmetric Hamming:

**Phase 1: Add ADC (current plan)**
- Implement ADC alongside Symmetric
- Default remains Symmetric for backward compatibility

**Phase 2: Validate**
- Benchmark both modes extensively
- Gather user feedback

**Phase 3: Deprecate (if validated)**
- Log warning when `RaBitQMode::Fast` used
- Document migration path
- Provide tool to add corrections to existing indexes

**Phase 4: Remove (major version)**
- Remove Symmetric codepath
- Simplify to single ADC implementation

---

## Appendix B: Expected Performance

Based on industry benchmarks and RaBitQ paper:

| Mode | Bits | Recall@10 (100K) | QPS | Storage/vec (512D) |
|------|------|------------------|-----|-------------------|
| Symmetric | 1 | ~85% (rerank=50) | High | 64 bytes |
| Symmetric | 2 | ~50% (broken) | High | 128 bytes |
| Symmetric | 4 | ~30% (broken) | High | 256 bytes |
| ADC | 1 | ~90% (rerank=10) | Medium | 72 bytes |
| ADC | 2 | ~95% (rerank=10) | Medium | 136 bytes |
| ADC | 4 | ~97% (rerank=10) | Medium | 264 bytes |
| Exact | - | 100% | Low | 2048 bytes |

*Note: ADC estimates based on RaBitQ paper claims. Actual performance TBD.*

---

## Changelog

- **2026-01-12**: Initial document created from research findings
- **2026-01-12**: Added section 1.5 explaining why ADC solves multi-bit problem
- **2026-01-12**: Added Part 4 analyzing ADC vs Symmetric Hamming tradeoffs
- **Issue #42**: √D scaling fix for rotation matrix
- **Issue #43**: Gray code encoding for multi-bit (doesn't fix fundamental Hamming limitation)
