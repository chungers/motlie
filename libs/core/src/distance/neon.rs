//! ARM NEON SIMD distance implementations
//!
//! Optimized for ARM64 processors (Apple Silicon, AWS Graviton, etc.)
//! NEON provides 128-bit vector registers processing 4 f32s at once.

#![allow(dead_code)]

use std::arch::aarch64::*;

/// Compute squared Euclidean distance using NEON
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn euclidean_squared(a: &[f32], b: &[f32]) -> f32 {
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
        let d = *a.get_unchecked(i) - *b.get_unchecked(i);
        result += d * d;
    }

    result
}

/// Compute cosine distance using NEON
///
/// Computes dot product, norm_a, and norm_b in a single pass for efficiency.
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn cosine(a: &[f32], b: &[f32]) -> f32 {
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

/// Compute dot product using NEON
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;

    let mut sum = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let va = vld1q_f32(a.as_ptr().add(i * 4));
        let vb = vld1q_f32(b.as_ptr().add(i * 4));
        sum = vfmaq_f32(sum, va, vb);
    }

    let mut result = vaddvq_f32(sum);

    // Handle tail
    for i in (chunks * 4)..len {
        result += *a.get_unchecked(i) * *b.get_unchecked(i);
    }

    result
}

/// Compute Hamming distance using NEON vcnt (popcount) instruction.
///
/// Processes 16 bytes at a time using 128-bit vectors.
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    let len = a.len();
    let chunks = len / 16;

    // Accumulator for popcount results (as u8 per lane, sum later)
    let mut total = vdupq_n_u8(0);

    for i in 0..chunks {
        let offset = i * 16;
        // Load 16 bytes from each array
        let va = vld1q_u8(a.as_ptr().add(offset));
        let vb = vld1q_u8(b.as_ptr().add(offset));

        // XOR to find differing bits
        let xor_result = veorq_u8(va, vb);

        // Count bits in each byte using vcnt
        let popcnt = vcntq_u8(xor_result);

        // Accumulate (will overflow if > 255*16 = 4080 bytes, but that's 32K bits)
        total = vaddq_u8(total, popcnt);
    }

    // Horizontal sum of all 16 bytes
    // vaddlvq_u8 sums all bytes into a u16, but we need u32
    let sum16 = vaddlvq_u8(total);
    let mut result = sum16 as u32;

    // Handle remaining bytes
    for i in (chunks * 16)..len {
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
        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..128).map(|i| (i + 1) as f32 * 0.01).collect();

        let neon_result = unsafe { euclidean_squared(&a, &b) };
        let scalar_result: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum();

        assert!(
            approx_eq(neon_result, scalar_result, 1e-4),
            "NEON: {}, Scalar: {}",
            neon_result,
            scalar_result
        );
    }

    #[test]
    fn test_cosine() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 4.0, 6.0, 8.0]; // Same direction

        let result = unsafe { cosine(&a, &b) };
        assert!(approx_eq(result, 0.0, 1e-5), "Got {}", result);
    }

    #[test]
    fn test_dot() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let neon_result = unsafe { dot(&a, &b) };
        let scalar_result: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        assert!(
            approx_eq(neon_result, scalar_result, 1e-5),
            "NEON: {}, Scalar: {}",
            neon_result,
            scalar_result
        );
    }
}
