//! SIMD Distance Computation Benchmark
//!
//! This benchmark measures the performance of different distance computation
//! implementations to evaluate SIMD optimization opportunities.
//!
//! ## Usage
//!
//! ```bash
//! # Build with native CPU optimizations
//! RUSTFLAGS='-C target-cpu=native' cargo build --release --example simd_bench
//!
//! # Run all benchmarks
//! ./target/release/examples/simd_bench
//!
//! # Run specific configuration
//! ./target/release/examples/simd_bench --dimension 128 --count 100000
//!
//! # Compare with target features
//! RUSTFLAGS='-C target-feature=+avx2,+fma' cargo build --release --example simd_bench
//!
//! # Run with Pulp crate (portable SIMD with auto-multiversioning)
//! cargo build --release --example simd_bench --features simd-pulp
//!
//! # Run with SimSIMD (C library bindings, highly optimized)
//! cargo build --release --example simd_bench --features simd-simsimd
//!
//! # Run with all SIMD libraries for full comparison
//! cargo build --release --example simd_bench --features simd-pulp,simd-simsimd
//! ```
//!
//! ## Output
//!
//! Results are printed in markdown table format for easy inclusion in SIMD.md.

use clap::Parser;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::time::{Duration, Instant};

// ============================================================================
// CLI Arguments
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "simd_bench")]
#[command(about = "Benchmark SIMD distance computation implementations")]
struct Args {
    /// Vector dimensions to test (comma-separated)
    #[arg(short, long, default_value = "128,256,512,1024")]
    dimensions: String,

    /// Number of vector pairs to compute distances for
    #[arg(short, long, default_value = "100000")]
    count: usize,

    /// Number of warmup iterations
    #[arg(short, long, default_value = "1000")]
    warmup: usize,

    /// Number of benchmark iterations for timing
    #[arg(short, long, default_value = "10")]
    iterations: usize,

    /// Output format: "table" or "csv"
    #[arg(short, long, default_value = "table")]
    format: String,
}

// ============================================================================
// Distance Implementations
// ============================================================================

/// Baseline: Simple iterator-based implementation (current code)
#[inline(never)]
fn euclidean_squared_baseline(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
}

/// Baseline with explicit loop (may help auto-vectorization)
#[inline(never)]
fn euclidean_squared_loop(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

/// Unrolled loop (4x unroll)
#[inline(never)]
fn euclidean_squared_unrolled(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let mut sum = 0.0f32;

    for i in 0..chunks {
        let idx = i * 4;
        let d0 = a[idx] - b[idx];
        let d1 = a[idx + 1] - b[idx + 1];
        let d2 = a[idx + 2] - b[idx + 2];
        let d3 = a[idx + 3] - b[idx + 3];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }

    // Handle tail
    for i in (chunks * 4)..len {
        let d = a[i] - b[i];
        sum += d * d;
    }

    sum
}

/// Target-feature enabled version (AVX2 + FMA)
/// This uses the same code but compiled with SIMD instructions
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
unsafe fn euclidean_squared_avx2_auto(a: &[f32], b: &[f32]) -> f32 {
    // Same iterator code, but compiler uses AVX2 instructions
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
}

/// Explicit AVX2 SIMD implementation
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
unsafe fn euclidean_squared_avx2_explicit(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 8;

    let mut sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    // Horizontal sum
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum4 = _mm_add_ps(lo, hi);
    let sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
    let sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
    let mut result = _mm_cvtss_f32(sum1);

    // Handle tail
    for i in (chunks * 8)..len {
        let d = a[i] - b[i];
        result += d * d;
    }

    result
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

/// Cosine distance - AVX2 explicit
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
unsafe fn cosine_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 8;

    let mut dot_sum = _mm256_setzero_ps();
    let mut norm_a_sum = _mm256_setzero_ps();
    let mut norm_b_sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
        norm_a_sum = _mm256_fmadd_ps(va, va, norm_a_sum);
        norm_b_sum = _mm256_fmadd_ps(vb, vb, norm_b_sum);
    }

    // Horizontal sums
    let dot = horizontal_sum_avx(dot_sum);
    let norm_a = horizontal_sum_avx(norm_a_sum);
    let norm_b = horizontal_sum_avx(norm_b_sum);

    // Handle tail
    let mut dot_tail = 0.0f32;
    let mut norm_a_tail = 0.0f32;
    let mut norm_b_tail = 0.0f32;
    for i in (chunks * 8)..len {
        dot_tail += a[i] * b[i];
        norm_a_tail += a[i] * a[i];
        norm_b_tail += b[i] * b[i];
    }

    let total_dot = dot + dot_tail;
    let total_norm_a = (norm_a + norm_a_tail).sqrt();
    let total_norm_b = (norm_b + norm_b_tail).sqrt();

    if total_norm_a == 0.0 || total_norm_b == 0.0 {
        return 1.0;
    }

    1.0 - (total_dot / (total_norm_a * total_norm_b))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal_sum_avx(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum4 = _mm_add_ps(lo, hi);
    let sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
    let sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
    _mm_cvtss_f32(sum1)
}

// ============================================================================
// ARM NEON Implementations
// ============================================================================

/// Explicit NEON SIMD implementation for ARM64
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline(never)]
unsafe fn euclidean_squared_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let len = a.len();
    let chunks = len / 4;

    let mut sum = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let va = vld1q_f32(a.as_ptr().add(i * 4));
        let vb = vld1q_f32(b.as_ptr().add(i * 4));
        let diff = vsubq_f32(va, vb);
        sum = vfmaq_f32(sum, diff, diff); // FMA: sum += diff * diff
    }

    // Horizontal sum of 4 floats
    let mut result = vaddvq_f32(sum);

    // Handle tail
    for i in (chunks * 4)..len {
        let d = a[i] - b[i];
        result += d * d;
    }

    result
}

/// Cosine distance - NEON explicit
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline(never)]
unsafe fn cosine_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::*;

    let len = a.len();
    let chunks = len / 4;

    let mut dot_sum = vdupq_n_f32(0.0);
    let mut norm_a_sum = vdupq_n_f32(0.0);
    let mut norm_b_sum = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let va = vld1q_f32(a.as_ptr().add(i * 4));
        let vb = vld1q_f32(b.as_ptr().add(i * 4));
        dot_sum = vfmaq_f32(dot_sum, va, vb);
        norm_a_sum = vfmaq_f32(norm_a_sum, va, va);
        norm_b_sum = vfmaq_f32(norm_b_sum, vb, vb);
    }

    // Horizontal sums
    let mut dot = vaddvq_f32(dot_sum);
    let mut norm_a = vaddvq_f32(norm_a_sum);
    let mut norm_b = vaddvq_f32(norm_b_sum);

    // Handle tail
    for i in (chunks * 4)..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let norm_a = norm_a.sqrt();
    let norm_b = norm_b.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }

    1.0 - (dot / (norm_a * norm_b))
}

// ============================================================================
// Pulp Crate Implementations (Optional)
// ============================================================================

/// Euclidean squared distance using Pulp's auto-dispatch SIMD
#[cfg(feature = "simd-pulp")]
#[inline(never)]
fn euclidean_squared_pulp(a: &[f32], b: &[f32]) -> f32 {
    use pulp::Arch;

    let arch = Arch::new();
    arch.dispatch(|| {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum::<f32>()
    })
}

/// Cosine distance using Pulp's auto-dispatch SIMD
#[cfg(feature = "simd-pulp")]
#[inline(never)]
fn cosine_distance_pulp(a: &[f32], b: &[f32]) -> f32 {
    use pulp::Arch;

    let arch = Arch::new();
    arch.dispatch(|| {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 1.0;
        }

        1.0 - (dot / (norm_a * norm_b))
    })
}

// ============================================================================
// SimSIMD Crate Implementations (Optional)
// ============================================================================

/// Euclidean squared distance using SimSIMD
#[cfg(feature = "simd-simsimd")]
#[inline(never)]
fn euclidean_squared_simsimd(a: &[f32], b: &[f32]) -> f32 {
    use simsimd::SpatialSimilarity;

    // SimSIMD trait methods are called as f32::l2sq(&[f32], &[f32]) -> Option<f64>
    <f32 as SpatialSimilarity>::l2sq(a, b).unwrap_or(f64::MAX) as f32
}

/// Cosine distance using SimSIMD
#[cfg(feature = "simd-simsimd")]
#[inline(never)]
fn cosine_distance_simsimd(a: &[f32], b: &[f32]) -> f32 {
    use simsimd::SpatialSimilarity;

    // SimSIMD returns Option<f64>, we need to convert
    <f32 as SpatialSimilarity>::cos(a, b).unwrap_or(1.0) as f32
}

/// Dot product using SimSIMD
#[cfg(feature = "simd-simsimd")]
#[allow(dead_code)]
#[inline(never)]
fn dot_product_simsimd(a: &[f32], b: &[f32]) -> f32 {
    use simsimd::SpatialSimilarity;

    <f32 as SpatialSimilarity>::dot(a, b).unwrap_or(0.0) as f32
}

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
    checksum: f32, // For validation
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
    let mut checksum = 0.0f32;
    for _ in 0..warmup {
        for (a, b) in vectors_a.iter().zip(vectors_b.iter()) {
            checksum += f(a, b);
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
        checksum,
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let args = Args::parse();

    // Parse dimensions
    let dimensions: Vec<usize> = args
        .dimensions
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    // Print system info
    println!("# SIMD Distance Benchmark");
    println!();
    println!("## System Information");
    println!();
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
            "- **FMA**: {}",
            if is_x86_feature_detected!("fma") {
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
        println!("- **NEON**: Assumed available");
    }
    println!("- **Count**: {} vector pairs", args.count);
    println!("- **Warmup**: {} iterations", args.warmup);
    println!("- **Benchmark iterations**: {}", args.iterations);

    // Show optional features
    println!();
    println!("### Optional Libraries");
    println!();
    #[cfg(feature = "simd-pulp")]
    println!("- **Pulp**: Enabled (auto-dispatch SIMD)");
    #[cfg(not(feature = "simd-pulp"))]
    println!("- **Pulp**: Disabled (use --features simd-pulp)");

    #[cfg(feature = "simd-simsimd")]
    println!("- **SimSIMD**: Enabled (C library bindings)");
    #[cfg(not(feature = "simd-simsimd"))]
    println!("- **SimSIMD**: Disabled (use --features simd-simsimd)");

    println!();

    // Run benchmarks for each dimension
    for dim in &dimensions {
        run_dimension_benchmark(*dim, args.count, args.warmup, args.iterations, &args.format);
    }
}

fn run_dimension_benchmark(
    dim: usize,
    count: usize,
    warmup: usize,
    iterations: usize,
    format: &str,
) {
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

    // Loop version
    results.push(benchmark(
        "Loop (explicit)",
        &vectors_a,
        &vectors_b,
        warmup,
        iterations,
        euclidean_squared_loop,
    ));

    // Unrolled version
    results.push(benchmark(
        "Unrolled (4x)",
        &vectors_a,
        &vectors_b,
        warmup,
        iterations,
        euclidean_squared_unrolled,
    ));

    // AVX2 versions (x86_64 only)
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        results.push(benchmark(
            "AVX2 (auto-vec)",
            &vectors_a,
            &vectors_b,
            warmup,
            iterations,
            |a, b| unsafe { euclidean_squared_avx2_auto(a, b) },
        ));

        results.push(benchmark(
            "AVX2 (explicit)",
            &vectors_a,
            &vectors_b,
            warmup,
            iterations,
            |a, b| unsafe { euclidean_squared_avx2_explicit(a, b) },
        ));
    }

    // NEON versions (ARM64 only)
    #[cfg(target_arch = "aarch64")]
    {
        results.push(benchmark(
            "NEON (explicit)",
            &vectors_a,
            &vectors_b,
            warmup,
            iterations,
            |a, b| unsafe { euclidean_squared_neon(a, b) },
        ));
    }

    // Pulp crate (optional - auto-dispatched SIMD)
    #[cfg(feature = "simd-pulp")]
    {
        results.push(benchmark(
            "Pulp (auto-dispatch)",
            &vectors_a,
            &vectors_b,
            warmup,
            iterations,
            euclidean_squared_pulp,
        ));
    }

    // SimSIMD crate (optional - C library bindings)
    #[cfg(feature = "simd-simsimd")]
    {
        results.push(benchmark(
            "SimSIMD",
            &vectors_a,
            &vectors_b,
            warmup,
            iterations,
            euclidean_squared_simsimd,
        ));
    }

    // Print results
    let baseline = &results[0];

    if format == "csv" {
        println!("implementation,dimension,count,ns_per_op,ops_per_sec,speedup");
        for r in &results {
            println!(
                "{},{},{},{:.2},{:.0},{:.2}x",
                r.name,
                r.dimension,
                r.count,
                r.ns_per_op,
                r.ops_per_sec,
                r.speedup_vs(baseline)
            );
        }
    } else {
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

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        cosine_results.push(benchmark(
            "AVX2 (explicit)",
            &vectors_a,
            &vectors_b,
            warmup,
            iterations,
            |a, b| unsafe { cosine_distance_avx2(a, b) },
        ));
    }

    #[cfg(target_arch = "aarch64")]
    {
        cosine_results.push(benchmark(
            "NEON (explicit)",
            &vectors_a,
            &vectors_b,
            warmup,
            iterations,
            |a, b| unsafe { cosine_distance_neon(a, b) },
        ));
    }

    // Pulp crate (optional - auto-dispatched SIMD)
    #[cfg(feature = "simd-pulp")]
    {
        cosine_results.push(benchmark(
            "Pulp (auto-dispatch)",
            &vectors_a,
            &vectors_b,
            warmup,
            iterations,
            cosine_distance_pulp,
        ));
    }

    // SimSIMD crate (optional - C library bindings)
    #[cfg(feature = "simd-simsimd")]
    {
        cosine_results.push(benchmark(
            "SimSIMD",
            &vectors_a,
            &vectors_b,
            warmup,
            iterations,
            cosine_distance_simsimd,
        ));
    }

    let cosine_baseline = &cosine_results[0];

    if format == "csv" {
        for r in &cosine_results {
            println!(
                "{},{},{},{:.2},{:.0},{:.2}x",
                r.name,
                r.dimension,
                r.count,
                r.ns_per_op,
                r.ops_per_sec,
                r.speedup_vs(cosine_baseline)
            );
        }
    } else {
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
    }
    println!();
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_euclidean_implementations_match() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let a: Vec<f32> = (0..128).map(|_| rng.gen()).collect();
        let b: Vec<f32> = (0..128).map(|_| rng.gen()).collect();

        let baseline = euclidean_squared_baseline(&a, &b);
        let loop_ver = euclidean_squared_loop(&a, &b);
        let unrolled = euclidean_squared_unrolled(&a, &b);

        assert!(approx_eq(baseline, loop_ver, 1e-5));
        assert!(approx_eq(baseline, unrolled, 1e-5));

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let avx2_auto = unsafe { euclidean_squared_avx2_auto(&a, &b) };
            let avx2_explicit = unsafe { euclidean_squared_avx2_explicit(&a, &b) };

            assert!(approx_eq(baseline, avx2_auto, 1e-4));
            assert!(approx_eq(baseline, avx2_explicit, 1e-4));
        }

        #[cfg(target_arch = "aarch64")]
        {
            let neon = unsafe { euclidean_squared_neon(&a, &b) };
            assert!(approx_eq(baseline, neon, 1e-4));
        }
    }

    #[test]
    fn test_cosine_implementations_match() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let a: Vec<f32> = (0..128).map(|_| rng.gen()).collect();
        let b: Vec<f32> = (0..128).map(|_| rng.gen()).collect();

        let baseline = cosine_distance_baseline(&a, &b);

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let avx2 = unsafe { cosine_distance_avx2(&a, &b) };
            assert!(approx_eq(baseline, avx2, 1e-4));
        }

        #[cfg(target_arch = "aarch64")]
        {
            let neon = unsafe { cosine_distance_neon(&a, &b) };
            assert!(approx_eq(baseline, neon, 1e-4));
        }
    }
}
