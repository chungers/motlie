//! SIMD-optimized distance functions with compile-time and runtime dispatch
//!
//! This module provides hardware-accelerated vector distance computations:
//! - **AVX-512**: x86_64 servers with AVX-512 (DGX Spark)
//! - **AVX2+FMA**: x86_64 servers with AVX2 (most modern x86)
//! - **NEON**: ARM64 (macOS Apple Silicon, Linux ARM)
//! - **Scalar**: Portable fallback
//!
//! ## Dispatch Strategy
//!
//! The SIMD level is selected by `build.rs` based on:
//! 1. Explicit feature flags (`--features simd-avx512`)
//! 2. Target features (`RUSTFLAGS='-C target-feature=+avx2,+fma'`)
//! 3. Target architecture auto-detection
//!
//! ## Usage
//!
//! ```rust
//! use crate::distance::DISTANCE;
//!
//! let dist = DISTANCE.euclidean_squared(&vec_a, &vec_b);
//! println!("Using SIMD level: {}", DISTANCE.level());
//! ```

mod scalar;

#[cfg(all(target_arch = "x86_64", any(simd_level = "avx2", simd_level = "runtime")))]
mod avx2;

#[cfg(all(target_arch = "x86_64", any(simd_level = "avx512", simd_level = "runtime")))]
mod avx512;

#[cfg(all(target_arch = "aarch64", simd_level = "neon"))]
mod neon;

#[cfg(simd_level = "runtime")]
mod runtime;

use lazy_static::lazy_static;

// ============================================================================
// Distance Dispatcher
// ============================================================================

/// SIMD-optimized distance computation dispatcher
///
/// Automatically selects the best implementation based on compile-time
/// configuration set by `build.rs`.
pub struct Distance {
    #[cfg(simd_level = "runtime")]
    variant: runtime::Variant,

    #[cfg(not(simd_level = "runtime"))]
    _phantom: std::marker::PhantomData<()>,
}

impl Distance {
    /// Create a new distance dispatcher
    ///
    /// For compile-time dispatch, this is a no-op.
    /// For runtime dispatch, this detects CPU features.
    pub fn new() -> Self {
        #[cfg(simd_level = "runtime")]
        {
            Self {
                variant: runtime::detect(),
            }
        }

        #[cfg(not(simd_level = "runtime"))]
        {
            Self {
                _phantom: std::marker::PhantomData,
            }
        }
    }

    /// Compute squared Euclidean distance between two vectors
    ///
    /// Returns sum((a[i] - b[i])^2) for all i
    #[inline]
    pub fn euclidean_squared(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

        #[cfg(simd_level = "avx512")]
        {
            unsafe { avx512::euclidean_squared(a, b) }
        }

        #[cfg(simd_level = "avx2")]
        {
            unsafe { avx2::euclidean_squared(a, b) }
        }

        #[cfg(simd_level = "neon")]
        {
            unsafe { neon::euclidean_squared(a, b) }
        }

        #[cfg(simd_level = "scalar")]
        {
            scalar::euclidean_squared(a, b)
        }

        #[cfg(simd_level = "runtime")]
        {
            self.variant.euclidean_squared(a, b)
        }
    }

    /// Compute Euclidean distance between two vectors
    ///
    /// Returns sqrt(sum((a[i] - b[i])^2))
    #[inline]
    pub fn euclidean(&self, a: &[f32], b: &[f32]) -> f32 {
        self.euclidean_squared(a, b).sqrt()
    }

    /// Compute cosine distance between two vectors
    ///
    /// Returns 1 - (a · b) / (||a|| * ||b||)
    #[inline]
    pub fn cosine(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

        #[cfg(simd_level = "avx512")]
        {
            unsafe { avx512::cosine(a, b) }
        }

        #[cfg(simd_level = "avx2")]
        {
            unsafe { avx2::cosine(a, b) }
        }

        #[cfg(simd_level = "neon")]
        {
            unsafe { neon::cosine(a, b) }
        }

        #[cfg(simd_level = "scalar")]
        {
            scalar::cosine(a, b)
        }

        #[cfg(simd_level = "runtime")]
        {
            self.variant.cosine(a, b)
        }
    }

    /// Compute dot product of two vectors
    ///
    /// Returns sum(a[i] * b[i])
    #[inline]
    pub fn dot(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "Vector dimensions must match");

        #[cfg(simd_level = "avx512")]
        {
            unsafe { avx512::dot(a, b) }
        }

        #[cfg(simd_level = "avx2")]
        {
            unsafe { avx2::dot(a, b) }
        }

        #[cfg(simd_level = "neon")]
        {
            unsafe { neon::dot(a, b) }
        }

        #[cfg(simd_level = "scalar")]
        {
            scalar::dot(a, b)
        }

        #[cfg(simd_level = "runtime")]
        {
            self.variant.dot(a, b)
        }
    }

    /// Returns the SIMD level being used
    pub fn level(&self) -> &'static str {
        #[cfg(simd_level = "avx512")]
        {
            "AVX-512"
        }

        #[cfg(simd_level = "avx2")]
        {
            "AVX2+FMA"
        }

        #[cfg(simd_level = "neon")]
        {
            "NEON"
        }

        #[cfg(simd_level = "scalar")]
        {
            "Scalar"
        }

        #[cfg(simd_level = "runtime")]
        {
            self.variant.name()
        }
    }
}

impl Default for Distance {
    fn default() -> Self {
        Self::new()
    }
}

// Global singleton for distance computation
lazy_static! {
    /// Global distance computation dispatcher
    ///
    /// Uses the SIMD level determined at compile time by `build.rs`.
    pub static ref DISTANCE: Distance = Distance::new();
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Compute squared Euclidean distance using the global dispatcher
#[inline]
pub fn euclidean_squared(a: &[f32], b: &[f32]) -> f32 {
    DISTANCE.euclidean_squared(a, b)
}

/// Compute Euclidean distance using the global dispatcher
#[inline]
pub fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    DISTANCE.euclidean(a, b)
}

/// Compute cosine distance using the global dispatcher
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    DISTANCE.cosine(a, b)
}

/// Compute dot product using the global dispatcher
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    DISTANCE.dot(a, b)
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
    fn test_euclidean_squared() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        // Expected: (1-5)^2 + (2-6)^2 + (3-7)^2 + (4-8)^2 = 16 + 16 + 16 + 16 = 64
        let result = euclidean_squared(&a, &b);
        assert!(approx_eq(result, 64.0, 1e-5), "Got {}", result);
    }

    #[test]
    fn test_euclidean() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];

        // Expected: sqrt(9 + 16) = 5
        let result = euclidean(&a, &b);
        assert!(approx_eq(result, 5.0, 1e-5), "Got {}", result);
    }

    #[test]
    fn test_cosine_same_direction() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 4.0, 6.0]; // Same direction, different magnitude

        // Expected: 0 (same direction = no angular distance)
        let result = cosine(&a, &b);
        assert!(approx_eq(result, 0.0, 1e-5), "Got {}", result);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];

        // Expected: 1 (orthogonal = maximum angular distance)
        let result = cosine(&a, &b);
        assert!(approx_eq(result, 1.0, 1e-5), "Got {}", result);
    }

    #[test]
    fn test_dot() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        // Expected: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        let result = dot(&a, &b);
        assert!(approx_eq(result, 32.0, 1e-5), "Got {}", result);
    }

    #[test]
    fn test_simd_level() {
        let level = DISTANCE.level();
        println!("SIMD level: {}", level);
        assert!(!level.is_empty());
    }
}
