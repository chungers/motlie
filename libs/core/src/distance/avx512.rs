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

/// Compute Hamming distance using AVX-512 for loading + scalar popcnt.
///
/// Processes 64 bytes at a time using 512-bit XOR, then uses
/// _popcnt64 on 8 x u64 chunks.
#[target_feature(enable = "avx512f,popcnt")]
#[inline]
pub unsafe fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    let len = a.len();
    let chunks = len / 64;
    let mut result = 0u32;

    for i in 0..chunks {
        let offset = i * 64;
        // Load 64 bytes from each array
        let va = _mm512_loadu_si512(a.as_ptr().add(offset) as *const i32);
        let vb = _mm512_loadu_si512(b.as_ptr().add(offset) as *const i32);

        // XOR to find differing bits
        let xor_result = _mm512_xor_si512(va, vb);

        // Extract as 8 x u64 and count bits
        let mut xor_bytes = [0u64; 8];
        _mm512_storeu_si512(xor_bytes.as_mut_ptr() as *mut i32, xor_result);

        result += _popcnt64(xor_bytes[0] as i64) as u32;
        result += _popcnt64(xor_bytes[1] as i64) as u32;
        result += _popcnt64(xor_bytes[2] as i64) as u32;
        result += _popcnt64(xor_bytes[3] as i64) as u32;
        result += _popcnt64(xor_bytes[4] as i64) as u32;
        result += _popcnt64(xor_bytes[5] as i64) as u32;
        result += _popcnt64(xor_bytes[6] as i64) as u32;
        result += _popcnt64(xor_bytes[7] as i64) as u32;
    }

    // Handle remaining bytes
    for i in (chunks * 64)..len {
        result += (*a.get_unchecked(i) ^ *b.get_unchecked(i)).count_ones();
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

    #[test]
    fn test_hamming_distance_identical() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("popcnt") {
            println!("Skipping AVX-512 hamming test - CPU doesn't support it");
            return;
        }

        let a = vec![0xFF; 64];
        let b = vec![0xFF; 64];
        assert_eq!(unsafe { hamming_distance(&a, &b) }, 0);
    }

    #[test]
    fn test_hamming_distance_opposite() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("popcnt") {
            println!("Skipping AVX-512 hamming test - CPU doesn't support it");
            return;
        }

        let a = vec![0x00; 64];
        let b = vec![0xFF; 64];
        // 64 bytes * 8 bits = 512 differing bits
        assert_eq!(unsafe { hamming_distance(&a, &b) }, 512);
    }

    #[test]
    fn test_hamming_distance_with_tail() {
        if !is_x86_feature_detected!("avx512f") || !is_x86_feature_detected!("popcnt") {
            println!("Skipping AVX-512 hamming test - CPU doesn't support it");
            return;
        }

        // 80 bytes = 64 (one AVX-512 chunk) + 16 (tail)
        let a = vec![0x00; 80];
        let b = vec![0xFF; 80];
        // 80 bytes * 8 bits = 640 differing bits
        assert_eq!(unsafe { hamming_distance(&a, &b) }, 640);
    }
}
