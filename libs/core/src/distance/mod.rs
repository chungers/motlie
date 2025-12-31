//! SIMD-optimized distance functions with compile-time and runtime dispatch
//!
//! This module provides hardware-accelerated vector distance computations:
//! - **AVX-512**: x86_64 servers with AVX-512 (DGX Spark)
//! - **AVX2+FMA**: x86_64 servers with AVX2 (most modern x86)
//! - **NEON**: ARM64 (macOS Apple Silicon, Linux ARM)
//! - **Scalar**: Portable fallback
//!
//! ## Usage
//!
//! ```rust
//! use motlie_core::distance::{euclidean_squared, cosine, dot, simd_level};
//!
//! let vec_a = vec![1.0, 2.0, 3.0, 4.0];
//! let vec_b = vec![5.0, 6.0, 7.0, 8.0];
//!
//! let dist = euclidean_squared(&vec_a, &vec_b);
//! let cos_dist = cosine(&vec_a, &vec_b);
//! let dot_prod = dot(&vec_a, &vec_b);
//!
//! println!("Using SIMD: {}", simd_level());
//! ```
//!
//! ## Dispatch Strategy
//!
//! The SIMD level is selected by `build.rs` based on:
//! 1. Explicit feature flags (`--features simd-avx512`)
//! 2. Target features (`RUSTFLAGS='-C target-feature=+avx2,+fma'`)
//! 3. Target architecture auto-detection
//!
//! ## Feature Flags
//!
//! | Feature | Description |
//! |---------|-------------|
//! | `simd-runtime` | Runtime CPU detection (portable binaries) |
//! | `simd-native` | Hint that `-C target-cpu=native` is used |
//! | `simd-avx2` | Force AVX2+FMA (x86_64) |
//! | `simd-avx512` | Force AVX-512 (DGX Spark) |
//! | `simd-neon` | Force NEON (ARM64) |
//! | `simd-none` | Scalar fallback only |
//! | `simd-simsimd` | SimSIMD C library for comparison |

pub mod scalar;

#[cfg(all(target_arch = "x86_64", any(simd_level = "avx2", simd_level = "runtime")))]
pub mod avx2;

#[cfg(all(target_arch = "x86_64", any(simd_level = "avx512", simd_level = "runtime")))]
pub mod avx512;

#[cfg(all(target_arch = "aarch64", simd_level = "neon"))]
pub mod neon;

#[cfg(simd_level = "runtime")]
pub mod runtime;

// Comprehensive test suite for all SIMD implementations
#[cfg(test)]
mod tests;

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

// Global singleton for distance computation (internal)
lazy_static! {
    static ref DISPATCHER: Distance = Distance::new();
}

// ============================================================================
// Public API
// ============================================================================

/// Compute squared Euclidean distance between two vectors
///
/// Returns `sum((a[i] - b[i])^2)` for all i.
///
/// # Example
///
/// ```
/// use motlie_core::distance::euclidean_squared;
///
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0, 6.0];
/// let dist = euclidean_squared(&a, &b);
/// assert!((dist - 27.0).abs() < 1e-5);
/// ```
#[inline]
pub fn euclidean_squared(a: &[f32], b: &[f32]) -> f32 {
    DISPATCHER.euclidean_squared(a, b)
}

/// Compute Euclidean (L2) distance between two vectors
///
/// Returns `sqrt(sum((a[i] - b[i])^2))`.
///
/// # Example
///
/// ```
/// use motlie_core::distance::euclidean;
///
/// let a = vec![0.0, 0.0];
/// let b = vec![3.0, 4.0];
/// let dist = euclidean(&a, &b);
/// assert!((dist - 5.0).abs() < 1e-5);
/// ```
#[inline]
pub fn euclidean(a: &[f32], b: &[f32]) -> f32 {
    DISPATCHER.euclidean(a, b)
}

/// Compute cosine distance between two vectors
///
/// Returns `1 - (a · b) / (||a|| * ||b||)`.
///
/// - Returns 0 for identical directions
/// - Returns 1 for orthogonal vectors
/// - Returns 2 for opposite directions
/// - Returns 1 for zero vectors (undefined, max distance)
///
/// # Example
///
/// ```
/// use motlie_core::distance::cosine;
///
/// let a = vec![1.0, 0.0];
/// let b = vec![1.0, 0.0];  // Same direction
/// assert!(cosine(&a, &b).abs() < 1e-5);
///
/// let c = vec![0.0, 1.0];  // Orthogonal
/// assert!((cosine(&a, &c) - 1.0).abs() < 1e-5);
/// ```
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    DISPATCHER.cosine(a, b)
}

/// Compute dot product of two vectors
///
/// Returns `sum(a[i] * b[i])`.
///
/// # Example
///
/// ```
/// use motlie_core::distance::dot;
///
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0, 6.0];
/// let product = dot(&a, &b);
/// assert!((product - 32.0).abs() < 1e-5);  // 1*4 + 2*5 + 3*6 = 32
/// ```
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    DISPATCHER.dot(a, b)
}

/// Returns the active SIMD implementation level
///
/// Possible values:
/// - `"AVX-512"` - x86_64 with AVX-512
/// - `"AVX2+FMA"` - x86_64 with AVX2 and FMA
/// - `"NEON"` - ARM64 NEON
/// - `"Scalar"` - Portable fallback
/// - `"AVX-512 (runtime)"` / `"AVX2+FMA (runtime)"` / `"Scalar (runtime)"` - Runtime detected
///
/// # Example
///
/// ```
/// use motlie_core::distance::simd_level;
///
/// println!("Using SIMD: {}", simd_level());
/// ```
#[inline]
pub fn simd_level() -> &'static str {
    DISPATCHER.level()
}

// Tests are in tests.rs - comprehensive test suite for all SIMD implementations
