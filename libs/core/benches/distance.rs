//! SIMD Distance Computation Benchmark
//!
//! This benchmark measures the performance of different distance computation
//! implementations to evaluate SIMD optimization opportunities.
//!
//! ## Usage
//!
//! ```bash
//! # Build with native CPU optimizations
//! RUSTFLAGS='-C target-cpu=native' cargo bench -p motlie-core
//!
//! # Run specific benchmark
//! cargo bench -p motlie-core -- euclidean
//!
//! # With SimSIMD comparison
//! cargo bench -p motlie-core --features simd-simsimd
//! ```

use motlie_core::distance::{self, simd_level};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::time::{Duration, Instant};

// ============================================================================
// Benchmark Infrastructure
// ============================================================================

/// Generate random vectors for benchmarking
fn generate_vectors(rng: &mut ChaCha8Rng, count: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..count)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

/// Benchmark result for a single implementation
#[derive(Debug, Clone)]
struct BenchResult {
    name: String,
    dimension: usize,
    count: usize,
    total_time: Duration,
    ns_per_op: f64,
    ops_per_sec: f64,
}

impl BenchResult {
    fn speedup_vs(&self, baseline: &BenchResult) -> f64 {
        baseline.ns_per_op / self.ns_per_op
    }
}

/// Run a benchmark for a distance function
fn benchmark<F>(
    name: &str,
    vectors_a: &[Vec<f32>],
    vectors_b: &[Vec<f32>],
    warmup: usize,
    iterations: usize,
    f: F,
) -> BenchResult
where
    F: Fn(&[f32], &[f32]) -> f32,
{
    let dimension = vectors_a[0].len();
    let count = vectors_a.len();

    // Warmup
    for _ in 0..warmup {
        for (a, b) in vectors_a.iter().zip(vectors_b.iter()) {
            std::hint::black_box(f(a, b));
        }
    }

    // Benchmark
    let mut total = Duration::ZERO;
    for _ in 0..iterations {
        let start = Instant::now();
        for (a, b) in vectors_a.iter().zip(vectors_b.iter()) {
            std::hint::black_box(f(a, b));
        }
        total += start.elapsed();
    }

    let total_ops = count * iterations;
    let ns_per_op = total.as_nanos() as f64 / total_ops as f64;
    let ops_per_sec = 1_000_000_000.0 / ns_per_op;

    BenchResult {
        name: name.to_string(),
        dimension,
        count,
        total_time: total,
        ns_per_op,
        ops_per_sec,
    }
}

/// Baseline: Simple iterator-based implementation
#[inline(never)]
fn euclidean_squared_baseline(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
}

/// Cosine distance - baseline
#[inline(never)]
fn cosine_distance_baseline(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }

    1.0 - (dot / (norm_a * norm_b))
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let dimensions = [128, 256, 512, 1024];
    let count = 100_000;
    let warmup = 1000;
    let iterations = 10;

    // Print system info
    println!("# SIMD Distance Benchmark");
    println!();
    println!("## System Information");
    println!();
    println!("- **SIMD Level**: {}", simd_level());
    #[cfg(target_arch = "x86_64")]
    {
        println!("- **Architecture**: x86_64");
        println!(
            "- **AVX2**: {}",
            if is_x86_feature_detected!("avx2") {
                "Supported"
            } else {
                "Not available"
            }
        );
        println!(
            "- **AVX-512F**: {}",
            if is_x86_feature_detected!("avx512f") {
                "Supported"
            } else {
                "Not available"
            }
        );
    }
    #[cfg(target_arch = "aarch64")]
    {
        println!("- **Architecture**: aarch64 (ARM64)");
        println!("- **NEON**: Available");
    }
    println!("- **Count**: {} vector pairs", count);
    println!("- **Warmup**: {} iterations", warmup);
    println!("- **Benchmark iterations**: {}", iterations);
    println!();

    // Run benchmarks for each dimension
    for dim in dimensions {
        run_dimension_benchmark(dim, count, warmup, iterations);
    }
}

fn run_dimension_benchmark(dim: usize, count: usize, warmup: usize, iterations: usize) {
    println!("## Euclidean Distance - {} dimensions", dim);
    println!();

    // Generate test data
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let vectors_a = generate_vectors(&mut rng, count, dim);
    let vectors_b = generate_vectors(&mut rng, count, dim);

    // Run benchmarks
    let mut results = Vec::new();

    // Baseline (iterator)
    results.push(benchmark(
        "Baseline (iterator)",
        &vectors_a,
        &vectors_b,
        warmup,
        iterations,
        euclidean_squared_baseline,
    ));

    // SIMD dispatcher
    results.push(benchmark(
        &format!("SIMD ({})", simd_level()),
        &vectors_a,
        &vectors_b,
        warmup,
        iterations,
        |a, b| distance::euclidean_squared(a, b),
    ));

    // SimSIMD (optional)
    #[cfg(feature = "simd-simsimd")]
    {
        use simsimd::SpatialSimilarity;
        results.push(benchmark(
            "SimSIMD",
            &vectors_a,
            &vectors_b,
            warmup,
            iterations,
            |a, b| <f32 as SpatialSimilarity>::l2sq(a, b).unwrap_or(f64::MAX) as f32,
        ));
    }

    // Print results
    let baseline = &results[0];

    println!("| Implementation | ns/op | M ops/sec | Speedup |");
    println!("|----------------|-------|-----------|---------|");
    for r in &results {
        println!(
            "| {} | {:.1} | {:.2} | {:.2}x |",
            r.name,
            r.ns_per_op,
            r.ops_per_sec / 1_000_000.0,
            r.speedup_vs(baseline)
        );
    }
    println!();

    // Cosine distance benchmarks
    println!("## Cosine Distance - {} dimensions", dim);
    println!();

    let mut cosine_results = Vec::new();

    cosine_results.push(benchmark(
        "Baseline (iterator)",
        &vectors_a,
        &vectors_b,
        warmup,
        iterations,
        cosine_distance_baseline,
    ));

    cosine_results.push(benchmark(
        &format!("SIMD ({})", simd_level()),
        &vectors_a,
        &vectors_b,
        warmup,
        iterations,
        |a, b| distance::cosine(a, b),
    ));

    #[cfg(feature = "simd-simsimd")]
    {
        use simsimd::SpatialSimilarity;
        cosine_results.push(benchmark(
            "SimSIMD",
            &vectors_a,
            &vectors_b,
            warmup,
            iterations,
            |a, b| <f32 as SpatialSimilarity>::cos(a, b).unwrap_or(1.0) as f32,
        ));
    }

    let cosine_baseline = &cosine_results[0];

    println!("| Implementation | ns/op | M ops/sec | Speedup |");
    println!("|----------------|-------|-----------|---------|");
    for r in &cosine_results {
        println!(
            "| {} | {:.1} | {:.2} | {:.2}x |",
            r.name,
            r.ns_per_op,
            r.ops_per_sec / 1_000_000.0,
            r.speedup_vs(cosine_baseline)
        );
    }
    println!();
}
