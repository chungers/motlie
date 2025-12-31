# SIMD Distance Computation Module

Hardware-accelerated vector distance functions for `motlie-core`.

**Last Updated**: 2025-12-31

---

## Overview

The `motlie_core::distance` module provides SIMD-optimized distance computation with automatic platform detection:

| Platform | SIMD Level | Register Width | Elements/Op |
|----------|------------|----------------|-------------|
| x86_64 + AVX-512 | AVX-512 | 512-bit | 16 f32s |
| x86_64 + AVX2 | AVX2+FMA | 256-bit | 8 f32s |
| ARM64 (Apple Silicon, Graviton) | NEON | 128-bit | 4 f32s |
| Other | Scalar | - | 1 f32 |

---

## Quick Start

```rust
use motlie_core::distance::{euclidean_squared, cosine, dot, simd_level};

let vec_a = vec![1.0, 2.0, 3.0, 4.0];
let vec_b = vec![5.0, 6.0, 7.0, 8.0];

// Use module-level functions (idiomatic API)
let dist = euclidean_squared(&vec_a, &vec_b);
let cos_dist = cosine(&vec_a, &vec_b);
let dot_prod = dot(&vec_a, &vec_b);

// Check which SIMD level is active
println!("Using SIMD: {}", simd_level());  // e.g., "NEON", "AVX2+FMA"
```

---

## API Reference

### Distance Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `euclidean_squared(a, b)` | Sum of squared differences | `Σ(a[i] - b[i])²` |
| `euclidean(a, b)` | Euclidean (L2) distance | `√Σ(a[i] - b[i])²` |
| `cosine(a, b)` | Cosine distance | `1 - (a·b)/(‖a‖·‖b‖)` |
| `dot(a, b)` | Dot product | `Σ(a[i] · b[i])` |
| `simd_level()` | Active SIMD implementation | `&'static str` |

---

## Build Configuration

### Feature Flags

| Feature | Description | Use Case |
|---------|-------------|----------|
| `simd-runtime` | Runtime CPU detection | Portable binaries |
| `simd-native` | Hint for `-C target-cpu=native` | Maximum local performance |
| `simd-avx2` | Force AVX2+FMA | x86_64 servers |
| `simd-avx512` | Force AVX-512 | DGX Spark, Intel Xeon |
| `simd-neon` | Force NEON | Apple Silicon, AWS Graviton |
| `simd-none` | Scalar fallback only | Testing/debugging |
| `simd-simsimd` | SimSIMD C library | Cross-validation |

### Build Examples

```bash
# Auto-detect platform (recommended for most cases)
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

---

## Performance

### Benchmark Results (ARM64 NEON)

| Distance | Dimension | Baseline (ns) | SIMD (ns) | Speedup |
|----------|-----------|---------------|-----------|---------|
| Euclidean | 128 | 39.4 | 28.2 | **1.39x** |
| Euclidean | 256 | 90.2 | 58.2 | **1.55x** |
| Euclidean | 512 | 220.9 | 119.0 | **1.86x** |
| Euclidean | 1024 | 485.8 | 237.4 | **2.05x** |
| Cosine | 128 | 96.8 | 28.4 | **3.41x** |
| Cosine | 256 | 252.4 | 58.2 | **4.34x** |
| Cosine | 512 | 655.1 | 117.0 | **5.60x** |
| Cosine | 1024 | 1428.8 | 247.3 | **5.78x** |

### Expected Performance (x86_64)

| Implementation | 128-dim (ns) | 1024-dim (ns) | Speedup |
|----------------|--------------|---------------|---------|
| Baseline | ~50 | ~400 | 1x |
| AVX2 + FMA | ~15 | ~80 | 3-5x |
| AVX-512 | ~10 | ~50 | 5-8x |

### Running Benchmarks

```bash
# Run the distance benchmark
cargo bench -p motlie-core

# With native optimizations
RUSTFLAGS='-C target-cpu=native' cargo bench -p motlie-core
```

---

## Implementation Details

### Module Structure

```
libs/core/src/distance/
├── mod.rs       # Main dispatcher with compile-time SIMD selection
├── scalar.rs    # Portable fallback (auto-vectorized)
├── neon.rs      # ARM64 NEON intrinsics
├── avx2.rs      # x86_64 AVX2+FMA intrinsics
├── avx512.rs    # x86_64 AVX-512 intrinsics
├── runtime.rs   # Runtime CPU detection
└── tests.rs     # Comprehensive test suite (26 tests)
```

### Dispatch Strategy

The `build.rs` script detects the target platform and sets `simd_level` cfg:

1. Check explicit feature flags (`simd-avx512`, `simd-neon`, etc.)
2. Check target features from `RUSTFLAGS`
3. Auto-detect based on target architecture
4. Fall back to runtime detection or scalar

### Test Coverage

| Category | Tests | Description |
|----------|-------|-------------|
| Cross-Implementation | 3 | SIMD matches scalar for all dimensions |
| Edge Cases | 7 | Empty, single, zero, identical, opposite, orthogonal |
| Mathematical Properties | 3 | Symmetry, non-negativity, scaling invariance |
| SIMD Boundaries | 3 | Tail handling for NEON(4), AVX2(8), AVX-512(16) |
| Stress Tests | 2 | Large vectors (10K dims), 100 random pairs |

```bash
# Run all tests
cargo test -p motlie-core

# With SimSIMD cross-validation
cargo test -p motlie-core --features simd-simsimd
```

---

## Design Decisions

### Why Custom Intrinsics over SimSIMD?

We evaluated SimSIMD and found equivalent performance:

| Implementation | 1024-dim Cosine | Notes |
|----------------|-----------------|-------|
| Our NEON | 207.1 ns | Zero dependencies |
| SimSIMD | 205.5 ns | Requires C toolchain |

We chose custom intrinsics because:
1. **Zero external dependencies** - pure Rust, no FFI
2. **Full control** - can optimize for specific patterns
3. **Simpler build** - no C toolchain required
4. **Equivalent performance** - within 1-2% of SimSIMD

### When to Use SimSIMD Instead

- Need Hamming/Jaccard distance for binary vectors
- Prefer battle-tested library over maintaining SIMD code
- Need ARM SVE support (not just NEON)

---

## Troubleshooting

### Check Active SIMD Level

```rust
use motlie_core::distance::DISTANCE;
println!("Using: {}", DISTANCE.level());
```

### Force Specific SIMD Level

```bash
# Force scalar for debugging
cargo build --features simd-none

# Check build output for SIMD detection
cargo build -vv 2>&1 | grep "SIMD:"
```

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "Scalar" on x86_64 | AVX2 not enabled | Add `--features simd-runtime` or use RUSTFLAGS |
| Build error on ARM | Missing NEON feature | Usually auto-detected; check target triple |
| Performance regression | Wrong SIMD level | Check `DISTANCE.level()` at runtime |

---

## References

1. [Rust SIMD Performance Guide](https://rust-lang.github.io/packed_simd/perf-guide/)
2. [Auto-Vectorization for Newer Instruction Sets](https://www.nickwilcox.com/blog/autovec2/)
3. [SimSIMD GitHub](https://github.com/ashvardanian/SimSIMD)
