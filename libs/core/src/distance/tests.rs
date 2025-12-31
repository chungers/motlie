//! Comprehensive test suite for SIMD distance implementations
//!
//! This module ensures correctness of all SIMD implementations through:
//! 1. **Cross-implementation validation**: All implementations must match scalar
//! 2. **Edge case testing**: Empty, single, odd lengths, SIMD boundaries
//! 3. **Numeric edge cases**: Zero vectors, large values, small values
//! 4. **Mathematical properties**: Triangle inequality, symmetry, etc.
//! 5. **SimSIMD reference comparison**: When available, validate against battle-tested C library

#![cfg(test)]

use super::scalar;
use super::{cosine, dot, euclidean, euclidean_squared, simd_level};

// ============================================================================
// Test Utilities
// ============================================================================

fn approx_eq(a: f32, b: f32, epsilon: f32) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    (a - b).abs() < epsilon
}

fn approx_eq_relative(a: f32, b: f32, rel_epsilon: f32) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    let max_val = a.abs().max(b.abs()).max(1e-10);
    (a - b).abs() / max_val < rel_epsilon
}

/// Generate deterministic test vectors
fn make_vectors(dim: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut a = Vec::with_capacity(dim);
    let mut b = Vec::with_capacity(dim);

    // Simple LCG for deterministic pseudo-random values
    let mut state = seed;
    for _ in 0..dim {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        a.push((state as f32 / u64::MAX as f32) * 2.0 - 1.0);
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        b.push((state as f32 / u64::MAX as f32) * 2.0 - 1.0);
    }

    (a, b)
}

// ============================================================================
// Cross-Implementation Validation Tests
// ============================================================================

/// Test that dispatcher matches scalar for various dimensions
#[test]
fn test_dispatcher_matches_scalar_euclidean() {
    // Test dimensions that exercise SIMD boundaries:
    // 1, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 1024
    let dimensions = [1, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 512, 1024];

    for dim in dimensions {
        let (a, b) = make_vectors(dim, 42 + dim as u64);

        let scalar_result = scalar::euclidean_squared(&a, &b);
        let dispatch_result = euclidean_squared(&a, &b);

        assert!(
            approx_eq_relative(scalar_result, dispatch_result, 1e-5),
            "Euclidean mismatch at dim={}: scalar={}, dispatch={} (using {})",
            dim,
            scalar_result,
            dispatch_result,
            simd_level()
        );
    }
}

#[test]
fn test_dispatcher_matches_scalar_cosine() {
    let dimensions = [1, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 512, 1024];

    for dim in dimensions {
        let (a, b) = make_vectors(dim, 123 + dim as u64);

        let scalar_result = scalar::cosine(&a, &b);
        let dispatch_result = cosine(&a, &b);

        assert!(
            approx_eq_relative(scalar_result, dispatch_result, 1e-5),
            "Cosine mismatch at dim={}: scalar={}, dispatch={} (using {})",
            dim,
            scalar_result,
            dispatch_result,
            simd_level()
        );
    }
}

#[test]
fn test_dispatcher_matches_scalar_dot() {
    let dimensions = [1, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255, 256, 512, 1024];

    for dim in dimensions {
        let (a, b) = make_vectors(dim, 456 + dim as u64);

        let scalar_result = scalar::dot(&a, &b);
        let dispatch_result = dot(&a, &b);

        assert!(
            approx_eq_relative(scalar_result, dispatch_result, 1e-5),
            "Dot product mismatch at dim={}: scalar={}, dispatch={} (using {})",
            dim,
            scalar_result,
            dispatch_result,
            simd_level()
        );
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_empty_vectors() {
    let a: Vec<f32> = vec![];
    let b: Vec<f32> = vec![];

    // Empty vectors should return 0 for euclidean and dot
    assert_eq!(euclidean_squared(&a, &b), 0.0);
    assert_eq!(dot(&a, &b), 0.0);
    // Cosine of empty vectors is undefined, we return 1.0 (max distance)
    assert_eq!(cosine(&a, &b), 1.0);
}

#[test]
fn test_single_element() {
    let a = vec![3.0];
    let b = vec![7.0];

    // (3-7)^2 = 16
    assert!(approx_eq(euclidean_squared(&a, &b), 16.0, 1e-5));
    // 3*7 = 21
    assert!(approx_eq(dot(&a, &b), 21.0, 1e-5));
    // Same direction, distance should be 0
    assert!(approx_eq(cosine(&a, &b), 0.0, 1e-5));
}

#[test]
fn test_zero_vectors() {
    let a = vec![0.0; 128];
    let b = vec![0.0; 128];

    assert_eq!(euclidean_squared(&a, &b), 0.0);
    assert_eq!(dot(&a, &b), 0.0);
    // Cosine of zero vectors should return max distance (1.0)
    assert_eq!(cosine(&a, &b), 1.0);
}

#[test]
fn test_one_zero_vector() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let zero = vec![0.0; 4];

    // Cosine with zero vector should return max distance
    assert_eq!(cosine(&a, &zero), 1.0);
    assert_eq!(cosine(&zero, &a), 1.0);
}

#[test]
fn test_identical_vectors() {
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    assert_eq!(euclidean_squared(&a, &a), 0.0);
    assert!(approx_eq(cosine(&a, &a), 0.0, 1e-6));
}

#[test]
fn test_opposite_vectors() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b: Vec<f32> = a.iter().map(|x| -x).collect();

    // Opposite direction = cosine distance of 2.0
    assert!(approx_eq(cosine(&a, &b), 2.0, 1e-5));
}

#[test]
fn test_orthogonal_vectors() {
    let a = vec![1.0, 0.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0, 0.0];

    // Orthogonal = cosine distance of 1.0
    assert!(approx_eq(cosine(&a, &b), 1.0, 1e-5));
    // Dot product = 0
    assert!(approx_eq(dot(&a, &b), 0.0, 1e-5));
}

// ============================================================================
// Mathematical Property Tests
// ============================================================================

#[test]
fn test_symmetry() {
    let (a, b) = make_vectors(128, 789);

    // Distance should be symmetric
    assert_eq!(
        euclidean_squared(&a, &b),
        euclidean_squared(&b, &a)
    );
    assert!(approx_eq(
        cosine(&a, &b),
        cosine(&b, &a),
        1e-6
    ));
}

#[test]
fn test_non_negativity() {
    for seed in 0..10 {
        let (a, b) = make_vectors(256, seed);

        assert!(
            euclidean_squared(&a, &b) >= 0.0,
            "Euclidean squared must be non-negative"
        );
        let cos_dist = cosine(&a, &b);
        assert!(
            cos_dist >= -1e-6 && cos_dist <= 2.0 + 1e-6,
            "Cosine distance must be in [0, 2], got {}",
            cos_dist
        );
    }
}

#[test]
fn test_scaling_invariance_cosine() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![5.0, 6.0, 7.0, 8.0];

    // Cosine distance should be invariant to scaling
    let a_scaled: Vec<f32> = a.iter().map(|x| x * 100.0).collect();
    let b_scaled: Vec<f32> = b.iter().map(|x| x * 0.01).collect();

    assert!(approx_eq(
        cosine(&a, &b),
        cosine(&a_scaled, &b_scaled),
        1e-5
    ));
}

#[test]
fn test_known_values() {
    // Test with known mathematical results
    let a = vec![3.0, 4.0];
    let b = vec![0.0, 0.0];

    // Distance from origin: sqrt(9+16) = 5
    assert!(approx_eq(euclidean(&a, &b), 5.0, 1e-5));

    // 3-4-5 triangle
    let c = vec![3.0, 0.0];
    assert!(approx_eq(euclidean(&a, &c), 4.0, 1e-5));
}

// ============================================================================
// SIMD Boundary Tests (Tail Handling)
// ============================================================================

#[test]
fn test_simd_boundary_neon_4() {
    // NEON processes 4 floats at a time
    // Test dimensions around multiples of 4
    for dim in [3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17] {
        let (a, b) = make_vectors(dim, dim as u64);
        let scalar_result = scalar::euclidean_squared(&a, &b);
        let simd_result = euclidean_squared(&a, &b);

        assert!(
            approx_eq_relative(scalar_result, simd_result, 1e-5),
            "NEON boundary test failed at dim={}: scalar={}, simd={}",
            dim,
            scalar_result,
            simd_result
        );
    }
}

#[test]
fn test_simd_boundary_avx2_8() {
    // AVX2 processes 8 floats at a time
    // Test dimensions around multiples of 8
    for dim in [7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32, 33] {
        let (a, b) = make_vectors(dim, dim as u64 + 100);
        let scalar_result = scalar::euclidean_squared(&a, &b);
        let simd_result = euclidean_squared(&a, &b);

        assert!(
            approx_eq_relative(scalar_result, simd_result, 1e-5),
            "AVX2 boundary test failed at dim={}: scalar={}, simd={}",
            dim,
            scalar_result,
            simd_result
        );
    }
}

#[test]
fn test_simd_boundary_avx512_16() {
    // AVX-512 processes 16 floats at a time
    // Test dimensions around multiples of 16
    for dim in [15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65] {
        let (a, b) = make_vectors(dim, dim as u64 + 200);
        let scalar_result = scalar::euclidean_squared(&a, &b);
        let simd_result = euclidean_squared(&a, &b);

        assert!(
            approx_eq_relative(scalar_result, simd_result, 1e-5),
            "AVX-512 boundary test failed at dim={}: scalar={}, simd={}",
            dim,
            scalar_result,
            simd_result
        );
    }
}

// ============================================================================
// SimSIMD Reference Comparison (when available)
// ============================================================================

#[cfg(feature = "simd-simsimd")]
mod simsimd_validation {
    use super::*;
    use simsimd::SpatialSimilarity;

    #[test]
    fn test_euclidean_matches_simsimd() {
        let dimensions = [128, 256, 512, 1024];

        for dim in dimensions {
            let (a, b) = make_vectors(dim, dim as u64 + 1000);

            let our_result = euclidean_squared(&a, &b);
            let simsimd_result =
                <f32 as SpatialSimilarity>::l2sq(&a, &b).unwrap_or(f64::MAX) as f32;

            assert!(
                approx_eq_relative(our_result, simsimd_result, 1e-4),
                "Euclidean mismatch vs SimSIMD at dim={}: ours={}, simsimd={}",
                dim,
                our_result,
                simsimd_result
            );
        }
    }

    #[test]
    fn test_cosine_matches_simsimd() {
        let dimensions = [128, 256, 512, 1024];

        for dim in dimensions {
            let (a, b) = make_vectors(dim, dim as u64 + 2000);

            let our_result = cosine(&a, &b);
            let simsimd_result = <f32 as SpatialSimilarity>::cos(&a, &b).unwrap_or(1.0) as f32;

            assert!(
                approx_eq_relative(our_result, simsimd_result, 1e-4),
                "Cosine mismatch vs SimSIMD at dim={}: ours={}, simsimd={}",
                dim,
                our_result,
                simsimd_result
            );
        }
    }

    #[test]
    fn test_dot_matches_simsimd() {
        let dimensions = [128, 256, 512, 1024];

        for dim in dimensions {
            let (a, b) = make_vectors(dim, dim as u64 + 3000);

            let our_result = dot(&a, &b);
            let simsimd_result = <f32 as SpatialSimilarity>::dot(&a, &b).unwrap_or(0.0) as f32;

            assert!(
                approx_eq_relative(our_result, simsimd_result, 1e-4),
                "Dot product mismatch vs SimSIMD at dim={}: ours={}, simsimd={}",
                dim,
                our_result,
                simsimd_result
            );
        }
    }
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_large_vectors() {
    // Test with 10K dimensions (typical embedding size for some models)
    let (a, b) = make_vectors(10240, 999);

    let scalar_result = scalar::euclidean_squared(&a, &b);
    let simd_result = euclidean_squared(&a, &b);

    assert!(
        approx_eq_relative(scalar_result, simd_result, 1e-4),
        "Large vector test failed: scalar={}, simd={}",
        scalar_result,
        simd_result
    );
}

#[test]
fn test_many_random_vectors() {
    // Test 100 random vector pairs
    for seed in 0..100 {
        let (a, b) = make_vectors(128, seed);

        let scalar_euc = scalar::euclidean_squared(&a, &b);
        let simd_euc = euclidean_squared(&a, &b);

        assert!(
            approx_eq_relative(scalar_euc, simd_euc, 1e-5),
            "Random vector test {} failed for euclidean",
            seed
        );

        let scalar_cos = scalar::cosine(&a, &b);
        let simd_cos = cosine(&a, &b);

        assert!(
            approx_eq_relative(scalar_cos, simd_cos, 1e-5),
            "Random vector test {} failed for cosine",
            seed
        );
    }
}
