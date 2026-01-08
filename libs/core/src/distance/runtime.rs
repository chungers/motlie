//! Runtime CPU feature detection for SIMD dispatch
//!
//! This module provides runtime detection of CPU SIMD capabilities,
//! allowing a single binary to run optimally on different hardware.

#![allow(dead_code)]

use super::scalar;

#[cfg(target_arch = "x86_64")]
use super::avx2;
#[cfg(target_arch = "x86_64")]
use super::avx512;

/// CPU SIMD capability variant detected at runtime
#[derive(Debug, Clone, Copy)]
pub enum Variant {
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    Scalar,
}

impl Variant {
    /// Returns the name of this SIMD variant
    pub fn name(&self) -> &'static str {
        match self {
            #[cfg(target_arch = "x86_64")]
            Variant::Avx512 => "AVX-512 (runtime)",
            #[cfg(target_arch = "x86_64")]
            Variant::Avx2 => "AVX2+FMA (runtime)",
            Variant::Scalar => "Scalar (runtime)",
        }
    }

    /// Compute squared Euclidean distance
    #[inline]
    pub fn euclidean_squared(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            #[cfg(target_arch = "x86_64")]
            Variant::Avx512 => unsafe { avx512::euclidean_squared(a, b) },
            #[cfg(target_arch = "x86_64")]
            Variant::Avx2 => unsafe { avx2::euclidean_squared(a, b) },
            Variant::Scalar => scalar::euclidean_squared(a, b),
        }
    }

    /// Compute cosine distance
    #[inline]
    pub fn cosine(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            #[cfg(target_arch = "x86_64")]
            Variant::Avx512 => unsafe { avx512::cosine(a, b) },
            #[cfg(target_arch = "x86_64")]
            Variant::Avx2 => unsafe { avx2::cosine(a, b) },
            Variant::Scalar => scalar::cosine(a, b),
        }
    }

    /// Compute dot product
    #[inline]
    pub fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            #[cfg(target_arch = "x86_64")]
            Variant::Avx512 => unsafe { avx512::dot(a, b) },
            #[cfg(target_arch = "x86_64")]
            Variant::Avx2 => unsafe { avx2::dot(a, b) },
            Variant::Scalar => scalar::dot(a, b),
        }
    }

    /// Compute Hamming distance between binary codes
    #[inline]
    pub fn hamming_distance(&self, a: &[u8], b: &[u8]) -> u32 {
        match self {
            #[cfg(target_arch = "x86_64")]
            Variant::Avx512 => unsafe { avx512::hamming_distance(a, b) },
            #[cfg(target_arch = "x86_64")]
            Variant::Avx2 => unsafe { avx2::hamming_distance(a, b) },
            Variant::Scalar => scalar::hamming_distance(a, b),
        }
    }
}

/// Detect the best available SIMD variant for the current CPU
pub fn detect() -> Variant {
    #[cfg(target_arch = "x86_64")]
    {
        // Check for AVX-512 first (best performance)
        if is_x86_feature_detected!("avx512f") {
            return Variant::Avx512;
        }

        // Check for AVX2 + FMA
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Variant::Avx2;
        }
    }

    // Fall back to scalar
    Variant::Scalar
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_runtime_detection() {
        let variant = detect();
        println!("Detected SIMD variant: {}", variant.name());

        // Verify the variant matches what we expect from feature detection
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                assert!(matches!(variant, Variant::Avx512));
            } else if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                assert!(matches!(variant, Variant::Avx2));
            } else {
                assert!(matches!(variant, Variant::Scalar));
            }
        }

        #[cfg(not(target_arch = "x86_64"))]
        {
            assert!(matches!(variant, Variant::Scalar));
        }
    }

    #[test]
    fn test_runtime_euclidean() {
        let variant = detect();

        let a: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let b: Vec<f32> = (0..128).map(|i| (i + 1) as f32 * 0.01).collect();

        let result = variant.euclidean_squared(&a, &b);
        let expected: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum();

        assert!(
            approx_eq(result, expected, 1e-4),
            "{}: got {}, expected {}",
            variant.name(),
            result,
            expected
        );
    }

    #[test]
    fn test_runtime_cosine() {
        let variant = detect();

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]; // Same direction

        let result = variant.cosine(&a, &b);
        assert!(
            approx_eq(result, 0.0, 1e-5),
            "{}: got {}",
            variant.name(),
            result
        );
    }

    #[test]
    fn test_runtime_dot() {
        let variant = detect();

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let result = variant.dot(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        assert!(
            approx_eq(result, expected, 1e-5),
            "{}: got {}, expected {}",
            variant.name(),
            result,
            expected
        );
    }

    #[test]
    fn test_runtime_hamming_identical() {
        let variant = detect();
        let a = vec![0xFF, 0x00, 0xAA, 0x55];
        let b = vec![0xFF, 0x00, 0xAA, 0x55];
        assert_eq!(variant.hamming_distance(&a, &b), 0);
    }

    #[test]
    fn test_runtime_hamming_opposite() {
        let variant = detect();
        let a = vec![0x00, 0x00];
        let b = vec![0xFF, 0xFF];
        assert_eq!(variant.hamming_distance(&a, &b), 16);
    }

    #[test]
    fn test_runtime_hamming_large() {
        let variant = detect();
        // 64 bytes (512 bits) - tests AVX-512 path
        let a = vec![0u8; 64];
        let b = vec![0xFFu8; 64];
        let result = variant.hamming_distance(&a, &b);
        assert_eq!(result, 512, "{}: got {}, expected 512", variant.name(), result);
    }
}
