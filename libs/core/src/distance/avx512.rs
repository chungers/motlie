//! AVX-512 SIMD distance implementations
//!
//! Optimized for x86_64 processors with AVX-512 support (DGX Spark, newer Intel/AMD).
//! AVX-512 provides 512-bit vector registers processing 16 f32s at once.

#![allow(dead_code)]

use std::arch::x86_64::*;

/// Compute squared Euclidean distance using AVX-512
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn euclidean_squared(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 16;

    let mut sum = _mm512_setzero_ps();

    for i in 0..chunks {
        let va = _mm512_loadu_ps(a.as_ptr().add(i * 16));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i * 16));
        let diff = _mm512_sub_ps(va, vb);
        sum = _mm512_fmadd_ps(diff, diff, sum); // FMA: sum += diff * diff
    }

    // Reduce 16 floats to single sum
    let mut result = _mm512_reduce_add_ps(sum);

    // Handle tail (remaining elements not divisible by 16)
    for i in (chunks * 16)..len {
        let d = *a.get_unchecked(i) - *b.get_unchecked(i);
        result += d * d;
    }

    result
}

/// Compute cosine distance using AVX-512
///
/// Computes dot product, norm_a, and norm_b in a single pass for efficiency.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 16;

    let mut dot_sum = _mm512_setzero_ps();
    let mut norm_a_sum = _mm512_setzero_ps();
    let mut norm_b_sum = _mm512_setzero_ps();

    for i in 0..chunks {
        let va = _mm512_loadu_ps(a.as_ptr().add(i * 16));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i * 16));
        dot_sum = _mm512_fmadd_ps(va, vb, dot_sum);
        norm_a_sum = _mm512_fmadd_ps(va, va, norm_a_sum);
        norm_b_sum = _mm512_fmadd_ps(vb, vb, norm_b_sum);
    }

    // Reduce to scalar sums
    let mut dot = _mm512_reduce_add_ps(dot_sum);
    let mut norm_a = _mm512_reduce_add_ps(norm_a_sum);
    let mut norm_b = _mm512_reduce_add_ps(norm_b_sum);

    // Handle tail
    for i in (chunks * 16)..len {
        let x = *a.get_unchecked(i);
        let y = *b.get_unchecked(i);
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let norm_a = norm_a.sqrt();
    let norm_b = norm_b.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }

    1.0 - (dot / (norm_a * norm_b))
}

/// Compute dot product using AVX-512
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 16;

    let mut sum = _mm512_setzero_ps();

    for i in 0..chunks {
        let va = _mm512_loadu_ps(a.as_ptr().add(i * 16));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i * 16));
        sum = _mm512_fmadd_ps(va, vb, sum);
    }

    let mut result = _mm512_reduce_add_ps(sum);

    // Handle tail
    for i in (chunks * 16)..len {
        result += *a.get_unchecked(i) * *b.get_unchecked(i);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_euclidean_squared() {
        if !is_x86_feature_detected!("avx512f") {
            println!("Skipping AVX-512 test - CPU doesn't support it");
            return;
        }

        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..128).map(|i| (i + 1) as f32 * 0.01).collect();

        let avx512_result = unsafe { euclidean_squared(&a, &b) };
        let scalar_result: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum();

        assert!(
            approx_eq(avx512_result, scalar_result, 1e-4),
            "AVX-512: {}, Scalar: {}",
            avx512_result,
            scalar_result
        );
    }

    #[test]
    fn test_cosine() {
        if !is_x86_feature_detected!("avx512f") {
            println!("Skipping AVX-512 test - CPU doesn't support it");
            return;
        }

        let a: Vec<f32> = (0..16).map(|i| i as f32 + 1.0).collect();
        let b: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 2.0).collect(); // Same direction

        let result = unsafe { cosine(&a, &b) };
        assert!(approx_eq(result, 0.0, 1e-5), "Got {}", result);
    }

    #[test]
    fn test_dot() {
        if !is_x86_feature_detected!("avx512f") {
            println!("Skipping AVX-512 test - CPU doesn't support it");
            return;
        }

        let a: Vec<f32> = (0..16).map(|i| i as f32 + 1.0).collect();
        let b: Vec<f32> = vec![1.0; 16];

        let avx512_result = unsafe { dot(&a, &b) };
        let scalar_result: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        assert!(
            approx_eq(avx512_result, scalar_result, 1e-5),
            "AVX-512: {}, Scalar: {}",
            avx512_result,
            scalar_result
        );
    }
}
