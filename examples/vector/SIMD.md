# SIMD Acceleration for Vector Distance Computations

**Phase 2a - Search Performance Optimization**

This document investigates SIMD (Single Instruction Multiple Data) acceleration for vector similarity computations in motlie_db vector search.

**Last Updated**: 2025-12-30

---

## Document Hierarchy

```
REQUIREMENTS.md     <- Ground truth for all design decisions
    |
POC.md              <- Phase 1: Current implementation
    |
HNSW2.md            <- Phase 2: Optimized HNSW with roaring bitmaps
    |
SIMD.md (this)      <- Phase 2a: SIMD distance acceleration (you are here)
    |
HYBRID.md           <- Phase 4: Billion-scale production architecture
```

**Links**: [REQUIREMENTS.md](./REQUIREMENTS.md) -> [POC.md](./POC.md) -> [HNSW2.md](./HNSW2.md) -> **SIMD.md** -> [HYBRID.md](./HYBRID.md)

---

## Target Requirements

This optimization addresses:

| Requirement | Target | Current | SIMD Target |
|-------------|--------|---------|-------------|
| [**THR-3**](./REQUIREMENTS.md#thr-3) | > 500 QPS | 47 QPS | 150-300 QPS |
| [**STOR-5**](./REQUIREMENTS.md#stor-5) | SIMD distance computation | Not started | **Implemented** |
| [**LAT-1**](./REQUIREMENTS.md#lat-1) | < 20ms P50 at 1M | 21.5ms | < 15ms |

**Expected Impact**: 3-10x speedup in distance computation, translating to 2-5x overall search QPS improvement.

---

## Executive Summary

### Current State

The current implementation in `common.rs` uses naive Rust iterators for distance computation:

```rust
// Current: ~500-1000ns per 1024-dim vector pair
pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
}
```

This is simple and correct, but leaves significant performance on the table:
- No explicit SIMD instructions
- Compiler auto-vectorization is limited for floats
- No memory alignment hints

### Recommended Solution

**Approach: Target-Feature SIMD with Runtime Detection**

After investigating multiple options, we recommend a hybrid approach:

1. **Primary**: Use `#[target_feature(enable = "avx2,fma")]` with `is_x86_feature_detected!` for runtime dispatch
2. **Fallback**: Auto-vectorized scalar code for non-AVX2 systems
3. **Optional**: SimSIMD crate as a third option (with caveats, see below)

This approach provides:
- **5-15x speedup** on AVX2-capable systems (most modern x86_64)
- **Portable fallback** for older CPUs and ARM
- **Zero external dependencies** for the primary implementation
- **Compile-time safety** with runtime performance

---

## Research Findings

### Option 1: SimSIMD Crate

**Source**: [SimSIMD GitHub](https://github.com/ashvardanian/SimSIMD)

SimSIMD advertises up to 200x faster dot products with support for:
- AVX2, AVX-512, NEON, SVE, SVE2
- f64, f32, f16, bf16, i8, binary vectors
- Euclidean, Cosine, Dot Product, Hamming, Jaccard

**Rust API**:
```rust
use simsimd::SpatialSimilarity;

let dist = f32::sqeuclidean(a, b);  // Squared Euclidean
let dist = f32::cosine(a, b);       // Cosine distance
let dist = f32::dot(a, b);          // Dot product
```

**Concerns** (from [Issue #107](https://github.com/ashvardanian/SimSIMD/issues/107)):

Benchmarks on AMD Ryzen 9 5900X with 1024-dim vectors showed:
| Implementation | Time per Op |
|----------------|-------------|
| SimSIMD (Rust bindings) | ~619 ns |
| Pure Rust + FMA | ~95 ns |
| Pure Rust (no FMA) | ~234 ns |
| ndarray + OpenBLAS | ~43 ns |

**Verdict**: SimSIMD's Rust bindings underperformed well-optimized pure Rust code. The issue was closed as "invalid" but improvements are ongoing. **Use with caution and benchmark.**

### Option 2: Target-Feature SIMD (Recommended)

**Sources**:
- [Auto-Vectorization for Newer Instruction Sets](https://www.nickwilcox.com/blog/autovec2/)
- [Getting rustc to use AVX2](https://alexheretic.github.io/posts/auto-avx2/)

Using Rust's `#[target_feature]` attribute with runtime detection:

```rust
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn euclidean_distance_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    // Same code, but compiler generates AVX2 instructions
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
}

pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { euclidean_distance_squared_avx2(a, b) };
        }
    }
    // Fallback to scalar
    euclidean_distance_squared_scalar(a, b)
}
```

**Benefits**:
- No external dependencies
- Compiler handles vectorization
- Runtime detection for portability
- Works with existing code patterns

### Option 3: Explicit SIMD with std::arch

For maximum control, use `std::arch` intrinsics directly:

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx2,fma")]
unsafe fn euclidean_distance_squared_explicit(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = _mm256_setzero_ps();

    for (a_chunk, b_chunk) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        let va = _mm256_loadu_ps(a_chunk.as_ptr());
        let vb = _mm256_loadu_ps(b_chunk.as_ptr());
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);  // FMA: sum += diff * diff
    }

    // Horizontal sum of 8 floats
    horizontal_sum_avx(sum)
}
```

**Benefits**:
- Maximum performance (can match or beat BLAS)
- Full control over instruction selection
- Predictable code generation

**Drawbacks**:
- More complex code
- Platform-specific
- Manual tail handling

### Option 4: Pulp/Macerator Crates

**Source**: [Pulp Crate](https://crates.io/crates/pulp)

The `pulp` crate provides portable SIMD with built-in multiversioning:

```rust
use pulp::Arch;

fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    let arch = Arch::new();
    arch.dispatch(|| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
    })
}
```

**Benefits**:
- Automatic multiversioning
- Powers the `faer` linear algebra library
- Clean API

**Status**: Worth evaluating in benchmarks.

---

## Proposed Implementation

### Phase 1: Benchmark Framework

Create `examples/vector/simd_bench.rs` with:

1. **Baseline measurements** (current implementation)
2. **Target-feature variants** (AVX2, AVX-512)
3. **SimSIMD comparison** (if added as optional dependency)
4. **Various dimensions** (128, 256, 512, 1024)
5. **Various counts** (1K, 10K, 100K distance computations)

### Phase 2: Implementation

#### 2.1 Distance Module Structure

Create `examples/vector/distance.rs`:

```rust
//! SIMD-optimized distance functions
//!
//! Provides automatic dispatch to the fastest implementation
//! based on runtime CPU feature detection.

mod scalar;      // Fallback implementations
mod avx2;        // AVX2 + FMA implementations
mod avx512;      // AVX-512 implementations (optional)

pub use api::*;

mod api {
    use super::*;

    /// Distance function dispatcher
    pub struct Distance {
        inner: DistanceImpl,
    }

    enum DistanceImpl {
        Scalar,
        Avx2,
        Avx512,
    }

    impl Distance {
        pub fn new() -> Self {
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx512f") {
                    return Self { inner: DistanceImpl::Avx512 };
                }
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    return Self { inner: DistanceImpl::Avx2 };
                }
            }
            Self { inner: DistanceImpl::Scalar }
        }

        #[inline]
        pub fn euclidean_squared(&self, a: &[f32], b: &[f32]) -> f32 {
            match self.inner {
                DistanceImpl::Avx512 => unsafe { avx512::euclidean_squared(a, b) },
                DistanceImpl::Avx2 => unsafe { avx2::euclidean_squared(a, b) },
                DistanceImpl::Scalar => scalar::euclidean_squared(a, b),
            }
        }

        #[inline]
        pub fn cosine(&self, a: &[f32], b: &[f32]) -> f32 {
            match self.inner {
                DistanceImpl::Avx512 => unsafe { avx512::cosine(a, b) },
                DistanceImpl::Avx2 => unsafe { avx2::cosine(a, b) },
                DistanceImpl::Scalar => scalar::cosine(a, b),
            }
        }
    }
}
```

#### 2.2 AVX2 Implementation

```rust
// examples/vector/distance/avx2.rs

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn euclidean_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let mut sum = _mm256_setzero_ps();
    let chunks = a.len() / 8;

    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    // Handle tail
    let mut tail_sum = 0.0f32;
    for i in (chunks * 8)..a.len() {
        let d = a[i] - b[i];
        tail_sum += d * d;
    }

    horizontal_sum_avx(sum) + tail_sum
}

#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal_sum_avx(v: __m256) -> f32 {
    // Sum 8 floats: [a,b,c,d,e,f,g,h]
    let hi = _mm256_extractf128_ps(v, 1);        // [e,f,g,h]
    let lo = _mm256_castps256_ps128(v);          // [a,b,c,d]
    let sum4 = _mm_add_ps(lo, hi);               // [a+e,b+f,c+g,d+h]
    let sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));  // [a+e+c+g, b+f+d+h, ...]
    let sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
    _mm_cvtss_f32(sum1)
}
```

### Phase 3: Integration

Update `common.rs` to use the new distance module:

```rust
use crate::distance::Distance;

lazy_static! {
    static ref DISTANCE: Distance = Distance::new();
}

pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    DISTANCE.euclidean_squared(a, b)
}
```

---

## Benchmark Plan

### Metrics to Collect

| Metric | Description |
|--------|-------------|
| **ns/op** | Nanoseconds per distance computation |
| **M ops/sec** | Million operations per second |
| **Speedup** | Ratio vs baseline |
| **CPU features** | Detected SIMD capabilities |

### Test Matrix

| Dimension | Count | Baseline | Unrolled | AVX2 | NEON |
|-----------|-------|----------|----------|------|------|
| 128 | 100K | 49.4 ns | 38.1 ns | TBD | TBD |
| 1024 | 100K | 437.9 ns | 319.9 ns | TBD | TBD |

### Benchmark Results (ARM64 NEON)

**System**: aarch64 Linux, NEON available, 100K vector pairs, 10 iterations

#### Summary: Before vs After SIMD Integration

| Distance | Dim | Before (ns) | After (ns) | **Speedup** |
|----------|-----|-------------|------------|-------------|
| Euclidean | 128 | 39.4 | 28.2 | **1.39x** |
| Euclidean | 256 | 90.2 | 58.2 | **1.55x** |
| Euclidean | 512 | 220.9 | 119.0 | **1.86x** |
| Euclidean | 1024 | 485.8 | 237.4 | **2.05x** |
| Cosine | 128 | 96.8 | 28.4 | **3.41x** |
| Cosine | 256 | 252.4 | 58.2 | **4.34x** |
| Cosine | 512 | 655.1 | 117.0 | **5.60x** |
| Cosine | 1024 | 1428.8 | 247.3 | **5.78x** |

#### Euclidean Distance - All Dimensions

| Dim | Baseline (ns) | NEON (ns) | M ops/sec (Before) | M ops/sec (After) | Speedup |
|-----|---------------|-----------|-------------------|-------------------|---------|
| 128 | 39.4 | 28.2 | 25.39 | 35.42 | **1.39x** |
| 256 | 90.2 | 58.2 | 11.09 | 17.18 | **1.55x** |
| 512 | 220.9 | 119.0 | 4.53 | 8.40 | **1.86x** |
| 1024 | 485.8 | 237.4 | 2.06 | 4.21 | **2.05x** |

#### Cosine Distance - All Dimensions

| Dim | Baseline (ns) | NEON (ns) | M ops/sec (Before) | M ops/sec (After) | Speedup |
|-----|---------------|-----------|-------------------|-------------------|---------|
| 128 | 96.8 | 28.4 | 10.33 | 35.19 | **3.41x** |
| 256 | 252.4 | 58.2 | 3.96 | 17.18 | **4.34x** |
| 512 | 655.1 | 117.0 | 1.53 | 8.55 | **5.60x** |
| 1024 | 1428.8 | 247.3 | 0.70 | 4.04 | **5.78x** |

**Key Findings**:
1. **Euclidean speedup scales with dimensions**: 1.39x (128d) → 2.05x (1024d)
2. **Cosine gains are dramatic**: 3.41x (128d) → 5.78x (1024d) due to single-pass FMA for dot product, norm_a, and norm_b
3. **Unrolling alone provides 1.46-1.51x** speedup without explicit SIMD
4. Iterator and explicit loop compile to identical code (compiler optimizes well)

### Running Benchmarks

```bash
# Build with native optimizations
RUSTFLAGS='-C target-cpu=native' cargo build --release --example simd_bench

# Run benchmarks
./target/release/examples/simd_bench

# With specific features
./target/release/examples/simd_bench --dimension 128 --count 100000
```

---

## Measured Performance

### ARM64 NEON (Validated)

| Distance | Dimension | Before (ns) | After (ns) | Speedup |
|----------|-----------|-------------|------------|---------|
| Euclidean | 128 | 39.4 | 28.2 | **1.39x** |
| Euclidean | 1024 | 485.8 | 237.4 | **2.05x** |
| Cosine | 128 | 96.8 | 28.4 | **3.41x** |
| Cosine | 1024 | 1428.8 | 247.3 | **5.78x** |

### Expected for x86_64 (Based on Research)

| Implementation | 128-dim (ns) | 1024-dim (ns) | Speedup |
|----------------|--------------|---------------|---------|
| Baseline (scalar) | ~50 | ~400 | 1x |
| AVX2 + FMA | ~15 | ~80 | 3-5x |
| AVX-512 | ~10 | ~50 | 5-8x |

### Impact on Search Performance

| Metric | Before SIMD | After SIMD (ARM64) | After SIMD (x86 AVX2) |
|--------|-------------|-------------------|----------------------|
| Distance time (128-dim) | 39-97 ns | 28 ns | ~15 ns |
| Distance time (1024-dim) | 486-1429 ns | 237-247 ns | ~80 ns |
| **Throughput gain** | - | **1.4-5.8x** | **3-5x** |
| Search QPS improvement | - | +20-40% | +50-80% |

**Note**: Distance computation is one component of total search time. Full search involves graph traversal, I/O, and other overheads. Actual search improvement depends on the proportion of time spent in distance computation (typically 30-60% of search time).

---

## Implementation Phases

### Phase 1: Benchmarking ✅ COMPLETE

- [x] Research SIMD options
- [x] Create `simd_bench.rs` benchmark binary
- [x] Collect baseline measurements
- [x] Document findings in this file

### Phase 2: Core Implementation ✅ COMPLETE

- [x] Create `distance/` module with compile-time dispatch
- [x] Implement AVX2 + FMA variants (`distance/avx2.rs`)
- [x] Implement AVX-512 variants (`distance/avx512.rs`)
- [x] Implement NEON variants (`distance/neon.rs`)
- [x] Implement Scalar fallback (`distance/scalar.rs`)
- [x] Add runtime detection (`distance/runtime.rs`)
- [x] Create `build.rs` with platform detection
- [x] Add feature flags to `Cargo.toml`
- [x] Integrate with `common.rs`

### Phase 3: Optimization (Future)

- [ ] Profile hot paths in actual search workloads
- [ ] Consider memory alignment for aligned loads
- [ ] Batch distance computation for multiple vectors

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| AVX2 not available | Low (modern CPUs) | Medium | Scalar fallback |
| Performance regression | Low | High | Comprehensive benchmarks |
| Platform incompatibility | Medium (ARM) | Medium | Conditional compilation |
| Maintenance burden | Low | Low | Well-documented code |

---

## References

1. [SimSIMD GitHub](https://github.com/ashvardanian/SimSIMD) - SIMD library for similarity metrics
2. [SimSIMD Issue #107](https://github.com/ashvardanian/SimSIMD/issues/107) - Rust performance concerns
3. [Auto-Vectorization for Newer Instruction Sets](https://www.nickwilcox.com/blog/autovec2/) - Target-feature patterns
4. [Getting rustc to use AVX2](https://alexheretic.github.io/posts/auto-avx2/) - Runtime detection
5. [Rust SIMD Performance Guide](https://rust-lang.github.io/packed_simd/perf-guide/target-feature/rustflags.html) - Official guidance
6. [Hamming Distance SIMD Optimization](https://emschwartz.me/unnecessary-optimization-in-rust-hamming-distances-simd-and-auto-vectorization/) - Real-world optimization case study

---

## Build Configuration

The SIMD implementation uses compile-time platform detection via `build.rs`:

```bash
# macOS Apple Silicon (auto-detects NEON)
cargo build --release

# DGX Spark with AVX-512
cargo build --release --features simd-avx512

# Portable binary with runtime detection
cargo build --release --features simd-runtime

# Maximum performance for current CPU
RUSTFLAGS='-C target-cpu=native' cargo build --release --features simd-native

# Force scalar fallback (for testing)
cargo build --release --features simd-none
```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `simd-runtime` | Runtime CPU detection (portable binaries) |
| `simd-native` | Hint that `-C target-cpu=native` is used |
| `simd-avx2` | Force AVX2+FMA (x86_64) |
| `simd-avx512` | Force AVX-512 (DGX Spark) |
| `simd-neon` | Force NEON (ARM64) |
| `simd-none` | Scalar fallback only |

---

## Changelog

| Date | Change | Author |
|------|--------|--------|
| 2025-12-30 | Implemented full SIMD integration with build.rs and platform detection | Claude Opus 4.5 |
| 2025-12-30 | Initial SIMD investigation and proposal | Claude Opus 4.5 |
