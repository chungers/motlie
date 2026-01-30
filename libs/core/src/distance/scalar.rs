//! Scalar (non-SIMD) distance implementations
//!
//! These are portable fallback implementations that work on any platform.
//! They may still benefit from auto-vectorization by the compiler.

#![allow(dead_code)] // Fallback implementations - may be unused when SIMD is available

/// Compute squared Euclidean distance
#[inline]
pub fn euclidean_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Compute cosine distance: 1 - cosine_similarity
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let norm_a = norm_a.sqrt();
    let norm_b = norm_b.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0; // Maximum distance for zero vectors
    }

    1.0 - (dot / (norm_a * norm_b))
}

/// Compute dot product
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute Hamming distance between two binary codes.
///
/// Processes 8 bytes at a time using u64 chunks for efficiency.
/// Uses `count_ones()` which compiles to efficient popcount instruction.
#[inline]
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len(), "Code length mismatch");

    let mut distance = 0u32;
    let len = a.len();
    let chunks = len / 8;

    // Process 8 bytes at a time using u64
    for i in 0..chunks {
        let offset = i * 8;
        // Safety: we've verified offset + 8 <= len via chunks calculation
        let a_chunk = u64::from_le_bytes([
            a[offset],
            a[offset + 1],
            a[offset + 2],
            a[offset + 3],
            a[offset + 4],
            a[offset + 5],
            a[offset + 6],
            a[offset + 7],
        ]);
        let b_chunk = u64::from_le_bytes([
            b[offset],
            b[offset + 1],
            b[offset + 2],
            b[offset + 3],
            b[offset + 4],
            b[offset + 5],
            b[offset + 6],
            b[offset + 7],
        ]);
        distance += (a_chunk ^ b_chunk).count_ones();
    }

    // Handle remaining bytes
    for i in (chunks * 8)..len {
        distance += (a[i] ^ b[i]).count_ones();
    }

    distance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_squared() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // (1-4)^2 + (2-5)^2 + (3-6)^2 = 9 + 9 + 9 = 27
        assert!((euclidean_squared(&a, &b) - 27.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_parallel() {
        let a = vec![1.0, 0.0];
        let b = vec![2.0, 0.0];
        assert!(cosine(&a, &b).abs() < 1e-5);
    }

    #[test]
    fn test_dot() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        assert!((dot(&a, &b) - 11.0).abs() < 1e-5);
    }

    #[test]
    fn test_hamming_distance_identical() {
        let a = vec![0xFF, 0x00, 0xAA, 0x55];
        let b = vec![0xFF, 0x00, 0xAA, 0x55];
        assert_eq!(hamming_distance(&a, &b), 0);
    }

    #[test]
    fn test_hamming_distance_opposite() {
        let a = vec![0x00, 0x00];
        let b = vec![0xFF, 0xFF];
        // 16 bits different
        assert_eq!(hamming_distance(&a, &b), 16);
    }

    #[test]
    fn test_hamming_distance_single_bit() {
        let a = vec![0b00000000];
        let b = vec![0b00000001];
        assert_eq!(hamming_distance(&a, &b), 1);
    }

    #[test]
    fn test_hamming_distance_large() {
        // 16 bytes (128 bits) - typical for 128D 1-bit quantization
        let a = vec![0u8; 16];
        let b = vec![0xFFu8; 16];
        assert_eq!(hamming_distance(&a, &b), 128);
    }
}
