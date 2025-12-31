//! AVX2+FMA SIMD distance implementations
//!
//! Optimized for x86_64 processors with AVX2 and FMA support.
//! AVX2 provides 256-bit vector registers processing 8 f32s at once.

#![allow(dead_code)]

use std::arch::x86_64::*;

/// Compute squared Euclidean distance using AVX2+FMA
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn euclidean_squared(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;

    let mut sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum); // FMA: sum += diff * diff
    }

    // Horizontal sum of 8 floats
    // First, add high 128 bits to low 128 bits
    let high = _mm256_extractf128_ps(sum, 1);
    let low = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(high, low);

    // Horizontal add within 128 bits
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let sum128 = _mm_hadd_ps(sum128, sum128);

    let mut result = _mm_cvtss_f32(sum128);

    // Handle tail
    for i in (chunks * 8)..len {
        let d = *a.get_unchecked(i) - *b.get_unchecked(i);
        result += d * d;
    }

    result
}

/// Compute cosine distance using AVX2+FMA
///
/// Computes dot product, norm_a, and norm_b in a single pass for efficiency.
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn cosine(a: &[f32], b: &[f32]) -> f32 {
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
    let dot = horizontal_sum(dot_sum);
    let mut norm_a = horizontal_sum(norm_a_sum);
    let mut norm_b = horizontal_sum(norm_b_sum);
    let mut dot = dot;

    // Handle tail
    for i in (chunks * 8)..len {
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

/// Compute dot product using AVX2+FMA
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;

    let mut sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    let mut result = horizontal_sum(sum);

    // Handle tail
    for i in (chunks * 8)..len {
        result += *a.get_unchecked(i) * *b.get_unchecked(i);
    }

    result
}

/// Horizontal sum of 8 f32s in a __m256
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn horizontal_sum(v: __m256) -> f32 {
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(high, low);
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let sum128 = _mm_hadd_ps(sum128, sum128);
    _mm_cvtss_f32(sum128)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_euclidean_squared() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            println!("Skipping AVX2 test - CPU doesn't support it");
            return;
        }

        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..128).map(|i| (i + 1) as f32 * 0.01).collect();

        let avx2_result = unsafe { euclidean_squared(&a, &b) };
        let scalar_result: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum();

        assert!(
            approx_eq(avx2_result, scalar_result, 1e-4),
            "AVX2: {}, Scalar: {}",
            avx2_result,
            scalar_result
        );
    }

    #[test]
    fn test_cosine() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            println!("Skipping AVX2 test - CPU doesn't support it");
            return;
        }

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]; // Same direction

        let result = unsafe { cosine(&a, &b) };
        assert!(approx_eq(result, 0.0, 1e-5), "Got {}", result);
    }

    #[test]
    fn test_dot() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            println!("Skipping AVX2 test - CPU doesn't support it");
            return;
        }

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let avx2_result = unsafe { dot(&a, &b) };
        let scalar_result: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        assert!(
            approx_eq(avx2_result, scalar_result, 1e-5),
            "AVX2: {}, Scalar: {}",
            avx2_result,
            scalar_result
        );
    }
}
