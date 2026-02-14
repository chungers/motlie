//! ARM NEON SIMD implementations of quantized dot products.
//!
//! These implementations use 128-bit vector registers to process
//! 4 f32 elements at once on ARM64 processors (Apple Silicon, Graviton, etc.).

#![allow(dead_code)]

use std::arch::aarch64::*;

/// Decode Gray code to binary for values 0-3 (2-bit).
#[inline]
fn from_gray_code_2bit(gray: u8) -> u8 {
    let n = gray;
    n ^ (n >> 1)
}

/// Decode Gray code to binary for values 0-15 (4-bit).
#[inline]
fn from_gray_code_4bit(gray: u8) -> u8 {
    let mut n = gray;
    n ^= n >> 2;
    n ^= n >> 1;
    n
}

/// Compute dot product between float vector and 1-bit packed binary code using NEON.
///
/// Uses the algebraic optimization: `dot = 2 * sum(q where bit=1) - sum(q)`
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn dot_1bit(query: &[f32], code: &[u8]) -> f32 {
    let len = query.len();
    let chunks = len / 4;

    // Compute query_sum using NEON
    let mut query_sum_vec = vdupq_n_f32(0.0);
    for i in 0..chunks {
        let vq = vld1q_f32(query.as_ptr().add(i * 4));
        query_sum_vec = vaddq_f32(query_sum_vec, vq);
    }
    let mut query_sum = vaddvq_f32(query_sum_vec);

    // Handle query_sum tail
    for i in (chunks * 4)..len {
        query_sum += *query.get_unchecked(i);
    }

    // Compute positive_sum (sum of query[i] where bit[i] == 1)
    // Process 4 dimensions at a time
    let mut positive_sum_vec = vdupq_n_f32(0.0);

    // Process in chunks of 8 (one code byte)
    let byte_chunks = len / 8;
    for byte_idx in 0..byte_chunks {
        let code_byte = *code.get_unchecked(byte_idx);

        // First 4 bits
        let bit0 = (code_byte >> 0) & 1;
        let bit1 = (code_byte >> 1) & 1;
        let bit2 = (code_byte >> 2) & 1;
        let bit3 = (code_byte >> 3) & 1;

        // Create mask: -1.0 if bit set, 0.0 if not
        // Actually, we'll use a different approach: multiply query by bit value
        let mask_arr = [bit0 as f32, bit1 as f32, bit2 as f32, bit3 as f32];
        let mask = vld1q_f32(mask_arr.as_ptr());
        let vq = vld1q_f32(query.as_ptr().add(byte_idx * 8));
        positive_sum_vec = vfmaq_f32(positive_sum_vec, vq, mask);

        // Second 4 bits
        let bit4 = (code_byte >> 4) & 1;
        let bit5 = (code_byte >> 5) & 1;
        let bit6 = (code_byte >> 6) & 1;
        let bit7 = (code_byte >> 7) & 1;

        let mask_arr = [bit4 as f32, bit5 as f32, bit6 as f32, bit7 as f32];
        let mask = vld1q_f32(mask_arr.as_ptr());
        let vq = vld1q_f32(query.as_ptr().add(byte_idx * 8 + 4));
        positive_sum_vec = vfmaq_f32(positive_sum_vec, vq, mask);
    }

    let mut positive_sum = vaddvq_f32(positive_sum_vec);

    // Handle tail (remaining dimensions after byte_chunks * 8)
    for i in (byte_chunks * 8)..len {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        if (code[byte_idx] >> bit_idx) & 1 == 1 {
            positive_sum += *query.get_unchecked(i);
        }
    }

    2.0 * positive_sum - query_sum
}

/// Compute dot product between float vector and 2-bit packed code using NEON.
///
/// Decodes Gray codes and looks up values from the 4-element table.
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn dot_2bit_lookup(query: &[f32], code: &[u8], values: &[f32; 4]) -> f32 {
    let len = query.len();

    // Lookup table is in `values` array, accessed directly below
    let _lookup = vld1q_f32(values.as_ptr()); // Reserved for future NEON lookup optimization

    let mut sum_vec = vdupq_n_f32(0.0);

    // Process 4 dimensions at a time (1 byte = 4 x 2-bit values)
    let chunks = len / 4;

    for chunk in 0..chunks {
        let byte = *code.get_unchecked(chunk);

        // Extract 4 gray codes
        let gray0 = byte & 0b11;
        let gray1 = (byte >> 2) & 0b11;
        let gray2 = (byte >> 4) & 0b11;
        let gray3 = (byte >> 6) & 0b11;

        // Decode Gray to binary
        let level0 = from_gray_code_2bit(gray0) as usize;
        let level1 = from_gray_code_2bit(gray1) as usize;
        let level2 = from_gray_code_2bit(gray2) as usize;
        let level3 = from_gray_code_2bit(gray3) as usize;

        // Look up values using array indexing (NEON doesn't have vpermps equivalent)
        // Extract individual lanes from lookup vector
        let decoded_arr = [
            values[level0.min(3)],
            values[level1.min(3)],
            values[level2.min(3)],
            values[level3.min(3)],
        ];
        let decoded = vld1q_f32(decoded_arr.as_ptr());

        // Load query values
        let vq = vld1q_f32(query.as_ptr().add(chunk * 4));

        // FMA: sum += query * decoded
        sum_vec = vfmaq_f32(sum_vec, vq, decoded);
    }

    let mut result = vaddvq_f32(sum_vec);

    // Handle tail
    for i in (chunks * 4)..len {
        let bit_offset = i * 2;
        let byte_idx = bit_offset / 8;
        let bit_shift = bit_offset % 8;
        let gray = (*code.get_unchecked(byte_idx) >> bit_shift) & 0b11;
        let level = from_gray_code_2bit(gray) as usize;
        result += *query.get_unchecked(i) * values[level.min(3)];
    }

    result
}

/// Compute dot product between float vector and 4-bit packed code using NEON.
///
/// Uses the linear mapping: value = level * scale + offset
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn dot_4bit_linear(query: &[f32], code: &[u8], scale: f32, offset: f32) -> f32 {
    let len = query.len();
    let chunks = len / 4;

    let scale_vec = vdupq_n_f32(scale);
    let offset_vec = vdupq_n_f32(offset);

    let mut sum_vec = vdupq_n_f32(0.0);

    for chunk in 0..chunks {
        // Load 2 bytes (4 x 4-bit values)
        let byte0 = *code.get_unchecked(chunk * 2);
        let byte1 = *code.get_unchecked(chunk * 2 + 1);

        // Extract 4 gray codes
        let gray0 = byte0 & 0x0F;
        let gray1 = (byte0 >> 4) & 0x0F;
        let gray2 = byte1 & 0x0F;
        let gray3 = (byte1 >> 4) & 0x0F;

        // Decode Gray to binary and convert to float
        let level0 = from_gray_code_4bit(gray0) as f32;
        let level1 = from_gray_code_4bit(gray1) as f32;
        let level2 = from_gray_code_4bit(gray2) as f32;
        let level3 = from_gray_code_4bit(gray3) as f32;

        // Create level vector
        let levels_arr = [level0, level1, level2, level3];
        let levels = vld1q_f32(levels_arr.as_ptr());

        // Apply linear transform: value = level * scale + offset
        let decoded = vfmaq_f32(offset_vec, levels, scale_vec);

        // Load query values
        let vq = vld1q_f32(query.as_ptr().add(chunk * 4));

        // FMA: sum += query * decoded
        sum_vec = vfmaq_f32(sum_vec, vq, decoded);
    }

    let mut result = vaddvq_f32(sum_vec);

    // Handle tail
    for i in (chunks * 4)..len {
        let bit_offset = i * 4;
        let byte_idx = bit_offset / 8;
        let bit_shift = bit_offset % 8;
        let gray = if bit_shift == 0 {
            *code.get_unchecked(byte_idx) & 0x0F
        } else {
            (*code.get_unchecked(byte_idx) >> 4) & 0x0F
        };
        let level = from_gray_code_4bit(gray);
        let value = (level as f32 * scale) + offset;
        result += *query.get_unchecked(i) * value;
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
    fn test_dot_1bit_neon() {
        let query: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let code = vec![0xFF, 0xFF]; // All bits set

        let neon_result = unsafe { dot_1bit(&query, &code) };
        let expected: f32 = query.iter().sum();

        assert!(
            approx_eq(neon_result, expected, 1e-4),
            "NEON: {}, Expected: {}",
            neon_result,
            expected
        );
    }

    #[test]
    fn test_dot_1bit_alternating_neon() {
        let query: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let code = vec![0b10101010]; // bits 1,3,5,7 set

        let neon_result = unsafe { dot_1bit(&query, &code) };
        // positive_sum = 2 + 4 + 6 + 8 = 20
        // query_sum = 36
        // dot = 2 * 20 - 36 = 4
        let expected = 4.0f32;

        assert!(
            approx_eq(neon_result, expected, 1e-4),
            "NEON: {}, Expected: {}",
            neon_result,
            expected
        );
    }

    #[test]
    fn test_dot_2bit_neon() {
        const LEVELS: [f32; 4] = [-1.5, -0.5, 0.5, 1.5];
        let query = vec![1.0f32; 4];
        // Encode levels 0,1,2,3
        // Gray: 0,1,3,2 → packed: 0b10_11_01_00
        let code = vec![0b10110100];

        let neon_result = unsafe { dot_2bit_lookup(&query, &code, &LEVELS) };
        // -1.5 + -0.5 + 0.5 + 1.5 = 0
        let expected = 0.0f32;

        assert!(
            approx_eq(neon_result, expected, 1e-4),
            "NEON: {}, Expected: {}",
            neon_result,
            expected
        );
    }

    #[test]
    fn test_dot_4bit_neon() {
        const SCALE: f32 = 1.0 / 3.75;
        const OFFSET: f32 = -2.0;

        let query = vec![1.0f32; 4];
        // Level 0 → Gray 0, Level 15 → Gray 8
        // Packed: byte0=0x80 (level 0, level 15), byte1=0x80
        let code = vec![0x80, 0x80];

        let neon_result = unsafe { dot_4bit_linear(&query, &code, SCALE, OFFSET) };
        // 2 * (-2.0 + 2.0) = 0
        let expected = 0.0f32;

        assert!(
            approx_eq(neon_result, expected, 1e-3),
            "NEON: {}, Expected: {}",
            neon_result,
            expected
        );
    }

    #[test]
    fn test_neon_matches_scalar() {
        use super::super::scalar;
        use rand::{RngExt, SeedableRng};

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Test 1-bit
        for dim in [8, 16, 64, 128, 512] {
            let query: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            let code: Vec<u8> = (0..(dim + 7) / 8).map(|_| rng.random()).collect();

            let scalar_result = scalar::dot_1bit(&query, &code);
            let neon_result = unsafe { dot_1bit(&query, &code) };

            assert!(
                approx_eq(neon_result, scalar_result, 1e-3),
                "1-bit dim={}: NEON={}, Scalar={}",
                dim,
                neon_result,
                scalar_result
            );
        }

        // Test 2-bit
        const LEVELS: [f32; 4] = [-1.5, -0.5, 0.5, 1.5];
        for dim in [4, 8, 16, 64, 128, 512] {
            let query: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            let code: Vec<u8> = (0..(dim * 2 + 7) / 8).map(|_| rng.random()).collect();

            let scalar_result = scalar::dot_2bit_lookup(&query, &code, &LEVELS);
            let neon_result = unsafe { dot_2bit_lookup(&query, &code, &LEVELS) };

            assert!(
                approx_eq(neon_result, scalar_result, 1e-2),
                "2-bit dim={}: NEON={}, Scalar={}",
                dim,
                neon_result,
                scalar_result
            );
        }

        // Test 4-bit
        const SCALE: f32 = 1.0 / 3.75;
        const OFFSET: f32 = -2.0;
        for dim in [4, 8, 16, 64, 128, 512] {
            let query: Vec<f32> = (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            let code: Vec<u8> = (0..(dim + 1) / 2).map(|_| rng.random()).collect();

            let scalar_result = scalar::dot_4bit_linear(&query, &code, SCALE, OFFSET);
            let neon_result = unsafe { dot_4bit_linear(&query, &code, SCALE, OFFSET) };

            assert!(
                approx_eq(neon_result, scalar_result, 1e-2),
                "4-bit dim={}: NEON={}, Scalar={}",
                dim,
                neon_result,
                scalar_result
            );
        }
    }
}
