//! AVX2+FMA SIMD implementations of quantized dot products.
//!
//! These implementations use 256-bit vector registers to process
//! 8 f32 elements at once, providing significant speedup for large vectors.

#![allow(dead_code)]

use std::arch::x86_64::*;

/// Decode Gray code to binary for values 0-3 (2-bit).
///
/// Gray: 00=0, 01=1, 11=2, 10=3
/// Binary: 00=0, 01=1, 10=2, 11=3
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

/// Horizontal sum of 8 f32s in a __m256.
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

/// Compute dot product between float vector and 1-bit packed binary code using AVX2.
///
/// Uses the algebraic optimization: `dot = 2 * sum(q where bit=1) - sum(q)`
///
/// The SIMD implementation:
/// 1. Computes sum(q) using AVX2 FMA
/// 2. For each byte of code, expands 8 bits to masks and selects query elements
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn dot_1bit(query: &[f32], code: &[u8]) -> f32 {
    let len = query.len();
    let chunks = len / 8;

    // Compute query_sum using SIMD
    let mut query_sum_vec = _mm256_setzero_ps();
    for i in 0..chunks {
        let vq = _mm256_loadu_ps(query.as_ptr().add(i * 8));
        query_sum_vec = _mm256_add_ps(query_sum_vec, vq);
    }
    let mut query_sum = horizontal_sum(query_sum_vec);

    // Handle query_sum tail
    for i in (chunks * 8)..len {
        query_sum += *query.get_unchecked(i);
    }

    // Compute positive_sum (sum of query[i] where bit[i] == 1)
    // For each byte, we expand 8 bits into 8 floats using a bit expansion technique
    let mut positive_sum = 0.0f32;

    // Bit masks for expanding a byte to 8 integers
    let bit_masks = _mm256_set_epi32(128, 64, 32, 16, 8, 4, 2, 1);

    for i in 0..chunks {
        let code_byte = *code.get_unchecked(i);

        // Broadcast the code byte to all lanes
        let code_broadcast = _mm256_set1_epi32(code_byte as i32);

        // AND with bit masks to isolate each bit
        let bits = _mm256_and_si256(code_broadcast, bit_masks);

        // Compare to zero: produces 0xFFFFFFFF (-1) if bit is set, 0 if not
        let mask = _mm256_cmpeq_epi32(bits, bit_masks);

        // Convert mask to float: -1.0 or 0.0
        // We need to AND with the query values
        let mask_f = _mm256_castsi256_ps(mask);

        // Load query values and apply mask
        let vq = _mm256_loadu_ps(query.as_ptr().add(i * 8));
        let masked = _mm256_and_ps(vq, mask_f);

        positive_sum += horizontal_sum(masked);
    }

    // Handle tail (remaining dimensions)
    for i in (chunks * 8)..len {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        if (code[byte_idx] >> bit_idx) & 1 == 1 {
            positive_sum += *query.get_unchecked(i);
        }
    }

    2.0 * positive_sum - query_sum
}

/// Compute dot product between float vector and 2-bit packed code using AVX2.
///
/// Uses vpermps to look up values from a 4-element table based on decoded Gray codes.
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn dot_2bit_lookup(query: &[f32], code: &[u8], values: &[f32; 4]) -> f32 {
    let len = query.len();

    // Load lookup table into lower 4 elements of __m256
    // vpermps uses indices 0-7, so we replicate the 4 values
    let lookup = _mm256_set_ps(
        values[3], values[2], values[1], values[0], values[3], values[2], values[1], values[0],
    );

    let mut sum_vec = _mm256_setzero_ps();

    // Process 4 dimensions at a time (1 byte = 4 x 2-bit values)
    // But we want to process 8 at a time for AVX2 efficiency
    let chunks = len / 8;

    for chunk in 0..chunks {
        // Load 2 bytes (8 x 2-bit values)
        let byte0 = *code.get_unchecked(chunk * 2);
        let byte1 = *code.get_unchecked(chunk * 2 + 1);

        // Extract 8 gray codes
        let gray0 = byte0 & 0b11;
        let gray1 = (byte0 >> 2) & 0b11;
        let gray2 = (byte0 >> 4) & 0b11;
        let gray3 = (byte0 >> 6) & 0b11;
        let gray4 = byte1 & 0b11;
        let gray5 = (byte1 >> 2) & 0b11;
        let gray6 = (byte1 >> 4) & 0b11;
        let gray7 = (byte1 >> 6) & 0b11;

        // Decode Gray to binary
        let level0 = from_gray_code_2bit(gray0);
        let level1 = from_gray_code_2bit(gray1);
        let level2 = from_gray_code_2bit(gray2);
        let level3 = from_gray_code_2bit(gray3);
        let level4 = from_gray_code_2bit(gray4);
        let level5 = from_gray_code_2bit(gray5);
        let level6 = from_gray_code_2bit(gray6);
        let level7 = from_gray_code_2bit(gray7);

        // Create index vector for vpermps
        let indices = _mm256_set_epi32(
            level7 as i32,
            level6 as i32,
            level5 as i32,
            level4 as i32,
            level3 as i32,
            level2 as i32,
            level1 as i32,
            level0 as i32,
        );

        // Look up values using vpermps
        let decoded = _mm256_permutevar8x32_ps(lookup, indices);

        // Load query values
        let vq = _mm256_loadu_ps(query.as_ptr().add(chunk * 8));

        // FMA: sum += query * decoded
        sum_vec = _mm256_fmadd_ps(vq, decoded, sum_vec);
    }

    let mut result = horizontal_sum(sum_vec);

    // Handle tail
    for i in (chunks * 8)..len {
        let bit_offset = i * 2;
        let byte_idx = bit_offset / 8;
        let bit_shift = bit_offset % 8;
        let gray = (*code.get_unchecked(byte_idx) >> bit_shift) & 0b11;
        let level = from_gray_code_2bit(gray) as usize;
        result += *query.get_unchecked(i) * values[level.min(3)];
    }

    result
}

/// Compute dot product between float vector and 4-bit packed code using AVX2.
///
/// Uses the linear mapping: value = level * scale + offset
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn dot_4bit_linear(query: &[f32], code: &[u8], scale: f32, offset: f32) -> f32 {
    let len = query.len();
    let chunks = len / 8;

    let scale_vec = _mm256_set1_ps(scale);
    let offset_vec = _mm256_set1_ps(offset);

    let mut sum_vec = _mm256_setzero_ps();

    for chunk in 0..chunks {
        // Load 4 bytes (8 x 4-bit values)
        let byte0 = *code.get_unchecked(chunk * 4);
        let byte1 = *code.get_unchecked(chunk * 4 + 1);
        let byte2 = *code.get_unchecked(chunk * 4 + 2);
        let byte3 = *code.get_unchecked(chunk * 4 + 3);

        // Extract 8 gray codes (4 bits each)
        let gray0 = byte0 & 0x0F;
        let gray1 = (byte0 >> 4) & 0x0F;
        let gray2 = byte1 & 0x0F;
        let gray3 = (byte1 >> 4) & 0x0F;
        let gray4 = byte2 & 0x0F;
        let gray5 = (byte2 >> 4) & 0x0F;
        let gray6 = byte3 & 0x0F;
        let gray7 = (byte3 >> 4) & 0x0F;

        // Decode Gray to binary
        let level0 = from_gray_code_4bit(gray0) as i32;
        let level1 = from_gray_code_4bit(gray1) as i32;
        let level2 = from_gray_code_4bit(gray2) as i32;
        let level3 = from_gray_code_4bit(gray3) as i32;
        let level4 = from_gray_code_4bit(gray4) as i32;
        let level5 = from_gray_code_4bit(gray5) as i32;
        let level6 = from_gray_code_4bit(gray6) as i32;
        let level7 = from_gray_code_4bit(gray7) as i32;

        // Create level vector
        let levels_i32 =
            _mm256_set_epi32(level7, level6, level5, level4, level3, level2, level1, level0);

        // Convert to float
        let levels_f32 = _mm256_cvtepi32_ps(levels_i32);

        // Apply linear transform: value = level * scale + offset
        let decoded = _mm256_fmadd_ps(levels_f32, scale_vec, offset_vec);

        // Load query values
        let vq = _mm256_loadu_ps(query.as_ptr().add(chunk * 8));

        // FMA: sum += query * decoded
        sum_vec = _mm256_fmadd_ps(vq, decoded, sum_vec);
    }

    let mut result = horizontal_sum(sum_vec);

    // Handle tail
    for i in (chunks * 8)..len {
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
    fn test_dot_1bit_avx2() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            println!("Skipping AVX2 test - CPU doesn't support it");
            return;
        }

        let query: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let code = vec![0xFF, 0xFF]; // All bits set

        let avx2_result = unsafe { dot_1bit(&query, &code) };
        let expected: f32 = query.iter().sum(); // All +1, so dot = sum

        assert!(
            approx_eq(avx2_result, expected, 1e-4),
            "AVX2: {}, Expected: {}",
            avx2_result,
            expected
        );
    }

    #[test]
    fn test_dot_1bit_alternating_avx2() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            println!("Skipping AVX2 test - CPU doesn't support it");
            return;
        }

        let query: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let code = vec![0b10101010]; // bits 1,3,5,7 set

        let avx2_result = unsafe { dot_1bit(&query, &code) };
        // positive_sum = 2 + 4 + 6 + 8 = 20
        // query_sum = 36
        // dot = 2 * 20 - 36 = 4
        let expected = 4.0f32;

        assert!(
            approx_eq(avx2_result, expected, 1e-4),
            "AVX2: {}, Expected: {}",
            avx2_result,
            expected
        );
    }

    #[test]
    fn test_dot_2bit_avx2() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            println!("Skipping AVX2 test - CPU doesn't support it");
            return;
        }

        const LEVELS: [f32; 4] = [-1.5, -0.5, 0.5, 1.5];
        let query = vec![1.0f32; 8];
        // Encode levels 0,1,2,3,0,1,2,3
        // Gray: 0,1,3,2,0,1,3,2
        // Packed byte0: 0b10_11_01_00, byte1: 0b10_11_01_00
        let code = vec![0b10110100, 0b10110100];

        let avx2_result = unsafe { dot_2bit_lookup(&query, &code, &LEVELS) };
        // 2 * (-1.5 + -0.5 + 0.5 + 1.5) = 0
        let expected = 0.0f32;

        assert!(
            approx_eq(avx2_result, expected, 1e-4),
            "AVX2: {}, Expected: {}",
            avx2_result,
            expected
        );
    }

    #[test]
    fn test_dot_4bit_avx2() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            println!("Skipping AVX2 test - CPU doesn't support it");
            return;
        }

        const SCALE: f32 = 1.0 / 3.75;
        const OFFSET: f32 = -2.0;

        let query = vec![1.0f32; 8];
        // Level 0 → Gray 0, Level 15 → Gray 8
        // Alternate: 0, 15, 0, 15, ...
        // Packed: byte0=0x80, byte1=0x80, byte2=0x80, byte3=0x80
        let code = vec![0x80, 0x80, 0x80, 0x80];

        let avx2_result = unsafe { dot_4bit_linear(&query, &code, SCALE, OFFSET) };
        // 4 * (-2.0 + 2.0) = 0
        let expected = 0.0f32;

        assert!(
            approx_eq(avx2_result, expected, 1e-3),
            "AVX2: {}, Expected: {}",
            avx2_result,
            expected
        );
    }

    #[test]
    fn test_avx2_matches_scalar() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            println!("Skipping AVX2 test - CPU doesn't support it");
            return;
        }

        use super::super::scalar;
        use rand::{Rng, SeedableRng};

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        // Test 1-bit
        for dim in [8, 16, 64, 128, 512] {
            let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let code: Vec<u8> = (0..(dim + 7) / 8).map(|_| rng.gen()).collect();

            let scalar_result = scalar::dot_1bit(&query, &code);
            let avx2_result = unsafe { dot_1bit(&query, &code) };

            assert!(
                approx_eq(avx2_result, scalar_result, 1e-3),
                "1-bit dim={}: AVX2={}, Scalar={}",
                dim,
                avx2_result,
                scalar_result
            );
        }

        // Test 2-bit
        const LEVELS: [f32; 4] = [-1.5, -0.5, 0.5, 1.5];
        for dim in [8, 16, 64, 128, 512] {
            let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let code: Vec<u8> = (0..(dim * 2 + 7) / 8).map(|_| rng.gen()).collect();

            let scalar_result = scalar::dot_2bit_lookup(&query, &code, &LEVELS);
            let avx2_result = unsafe { dot_2bit_lookup(&query, &code, &LEVELS) };

            assert!(
                approx_eq(avx2_result, scalar_result, 1e-2),
                "2-bit dim={}: AVX2={}, Scalar={}",
                dim,
                avx2_result,
                scalar_result
            );
        }

        // Test 4-bit
        const SCALE: f32 = 1.0 / 3.75;
        const OFFSET: f32 = -2.0;
        for dim in [8, 16, 64, 128, 512] {
            let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let code: Vec<u8> = (0..(dim + 1) / 2).map(|_| rng.gen()).collect();

            let scalar_result = scalar::dot_4bit_linear(&query, &code, SCALE, OFFSET);
            let avx2_result = unsafe { dot_4bit_linear(&query, &code, SCALE, OFFSET) };

            assert!(
                approx_eq(avx2_result, scalar_result, 1e-2),
                "4-bit dim={}: AVX2={}, Scalar={}",
                dim,
                avx2_result,
                scalar_result
            );
        }
    }
}
