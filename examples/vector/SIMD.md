# SIMD Acceleration for Vector Distance Computations

> **Note**: The SIMD distance module is now part of `motlie-core`.
> See [libs/core/src/distance/README.md](../../libs/core/src/distance/README.md) for full documentation.

## Quick Start

```rust
use motlie_core::distance::{euclidean_squared, cosine, dot, simd_level};

let a = vec![1.0, 2.0, 3.0, 4.0];
let b = vec![5.0, 6.0, 7.0, 8.0];

// Functions automatically use optimal SIMD for your platform
let dist = euclidean_squared(&a, &b);
let cos_dist = cosine(&a, &b);
let dot_prod = dot(&a, &b);

println!("Using: {}", simd_level());  // "NEON", "AVX2+FMA", "AVX-512", etc.
```

## Benchmarks

```bash
# Run the motlie-core benchmark
cargo bench -p motlie-core

# Run the standalone example benchmark
cargo run --release --example simd_bench
```

## Documentation

- **Full documentation**: [libs/core/src/distance/README.md](../../libs/core/src/distance/README.md)
- **Benchmark details**: [libs/core/benches/README.md](../../libs/core/benches/README.md)
