# motlie-core Benchmarks

Performance benchmarks for `motlie-core` modules.

## Distance Benchmarks

The `distance.rs` benchmark measures SIMD-optimized distance computation performance across different vector dimensions.

### Quick Start

```bash
# Run all benchmarks with default settings
cargo bench -p motlie-core

# Run with native CPU optimizations (recommended)
RUSTFLAGS='-C target-cpu=native' cargo bench -p motlie-core

# Filter to specific benchmark
cargo bench -p motlie-core -- euclidean
cargo bench -p motlie-core -- cosine
```

### Feature Flags

Control SIMD implementation selection:

| Flag | Description | Command |
|------|-------------|---------|
| (default) | Auto-detect platform | `cargo bench -p motlie-core` |
| `simd-avx512` | Force AVX-512 | `cargo bench -p motlie-core --features simd-avx512` |
| `simd-avx2` | Force AVX2+FMA | `cargo bench -p motlie-core --features simd-avx2` |
| `simd-neon` | Force NEON | `cargo bench -p motlie-core --features simd-neon` |
| `simd-none` | Scalar only | `cargo bench -p motlie-core --features simd-none` |
| `simd-runtime` | Runtime detection | `cargo bench -p motlie-core --features simd-runtime` |
| `simd-simsimd` | Include SimSIMD | `cargo bench -p motlie-core --features simd-simsimd` |

### Platform-Specific Examples

```bash
# macOS Apple Silicon (NEON auto-detected)
cargo bench -p motlie-core

# Linux x86_64 with AVX2
RUSTFLAGS='-C target-cpu=native' cargo bench -p motlie-core

# DGX Spark / Intel Xeon with AVX-512
cargo bench -p motlie-core --features simd-avx512

# Cross-validate against SimSIMD
cargo bench -p motlie-core --features simd-simsimd
```

### Benchmark Output

The benchmark produces markdown-formatted tables:

```
## Euclidean Distance - 1024 dimensions

| Implementation | ns/op | M ops/sec | Speedup |
|----------------|-------|-----------|---------|
| Baseline (iterator) | 485.8 | 2.06 | 1.00x |
| SIMD (NEON) | 237.4 | 4.21 | 2.05x |
```

### What's Measured

| Metric | Description |
|--------|-------------|
| **Baseline** | Simple iterator-based implementation (auto-vectorized) |
| **SIMD** | Platform-optimized implementation (NEON/AVX2/AVX-512) |
| **SimSIMD** | C library comparison (optional, with `simd-simsimd` feature) |

Dimensions tested: 128, 256, 512, 1024

### Interpreting Results

- **Speedup** is relative to baseline iterator implementation
- Higher dimensions show better SIMD utilization (more data to process in parallel)
- Cosine distance benefits more than Euclidean (3 dot products vs 1 accumulation)

### Troubleshooting

```bash
# Check which SIMD level is active
cargo bench -p motlie-core 2>&1 | grep "SIMD Level"

# Verbose build to see SIMD detection
cargo bench -p motlie-core -vv 2>&1 | grep "SIMD:"

# Force rebuild
cargo bench -p motlie-core --force
```

## Adding New Benchmarks

Benchmarks use a simple custom harness (not Criterion) for minimal dependencies:

```rust
// In benches/my_bench.rs
fn main() {
    // Generate test data
    let vectors = generate_vectors(...);

    // Run benchmark
    let result = benchmark("name", &vectors_a, &vectors_b, warmup, iterations, |a, b| {
        my_function(a, b)
    });

    // Print results
    println!("| {} | {:.1} | {:.2}x |", result.name, result.ns_per_op, speedup);
}
```

Add to `Cargo.toml`:

```toml
[[bench]]
name = "my_bench"
harness = false
```
