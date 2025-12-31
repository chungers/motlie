# SIMD Acceleration for Vector Distance Computations

> **Note**: The SIMD distance computation module has been moved to `libs/core/src/distance/`.
> See [libs/core/docs/SIMD.md](../../libs/core/docs/SIMD.md) for the full documentation.

## Quick Reference

The SIMD-optimized distance functions are now available via `motlie_core::distance`:

```rust
use motlie_core::distance::{DISTANCE, euclidean_squared, cosine, dot};

// Use the global dispatcher
let dist = DISTANCE.euclidean_squared(&vec_a, &vec_b);
println!("Using SIMD level: {}", DISTANCE.level());

// Or use convenience functions
let dist = euclidean_squared(&vec_a, &vec_b);
let cos_dist = cosine(&vec_a, &vec_b);
let dot_prod = dot(&vec_a, &vec_b);
```

## Feature Flags

SIMD dispatch is controlled via feature flags on `motlie-core`:

```bash
# macOS Apple Silicon (auto-detects NEON)
cargo build --release

# DGX Spark (AVX-512)
cargo build --release --features simd-avx512

# Portable binary with runtime detection
cargo build --release --features simd-runtime

# Maximum performance for current CPU
RUSTFLAGS='-C target-cpu=native' cargo build --release --features simd-native
```

## Benchmarks

Run the distance benchmark:

```bash
# Run benchmark from libs/core
cargo bench -p motlie-core

# Run the standalone simd_bench example
cargo run --release --example simd_bench
```

## Performance Summary (ARM64 NEON)

| Distance | Dim | Baseline (ns) | SIMD (ns) | Speedup |
|----------|-----|---------------|-----------|---------|
| Euclidean | 1024 | 485.8 | 237.4 | **2.05x** |
| Cosine | 1024 | 1428.8 | 247.3 | **5.78x** |

See the full documentation at [libs/core/docs/SIMD.md](../../libs/core/docs/SIMD.md).
