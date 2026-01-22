//! Quantized dot product operations for RaBitQ and similar algorithms.
//!
//! This module provides hardware-accelerated primitives for computing dot products
//! between high-precision float vectors (queries) and low-precision packed binary
//! codes (quantized database vectors). This is the core operation for Asymmetric
//! Distance Computation (ADC).
//!
//! ## Usage
//!
//! ```rust
//! use motlie_core::distance::quantized::{dot_1bit, dot_2bit_lookup, dot_4bit_linear};
//!
//! let query = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
//!
//! // 1-bit code: 8 dimensions packed into 1 byte
//! let code_1bit = vec![0b10101010]; // bits 1,3,5,7 set
//! let dot1 = dot_1bit(&query, &code_1bit);
//!
//! // 2-bit code: 4 dimensions packed into 1 byte
//! let code_2bit = vec![0b10110100]; // Gray coded levels
//! let levels = [-1.5f32, -0.5, 0.5, 1.5];
//! let dot2 = dot_2bit_lookup(&query[..4], &code_2bit, &levels);
//!
//! // 4-bit code: 2 dimensions packed into 1 byte
//! let code_4bit = vec![0x80]; // Gray coded levels
//! let dot4 = dot_4bit_linear(&query[..2], &code_4bit, 1.0/3.75, -2.0);
//! ```
//!
//! ## Encoding Format
//!
//! All packed codes use LSB-first bit ordering:
//! - **1-bit**: 8 values per byte, bit 0 = first dimension
//! - **2-bit**: 4 values per byte, bits 0-1 = first dimension
//! - **4-bit**: 2 values per byte, bits 0-3 = first dimension
//!
//! Values are encoded using Gray code for adjacency properties.
//!
//! ## SIMD Dispatch
//!
//! Automatically selects the best implementation based on compile-time configuration:
//! - **AVX2**: 8-wide f32 processing with vpermps lookups (x86_64)
//! - **NEON**: 4-wide f32 processing (ARM64)
//! - **Scalar**: Portable fallback

mod scalar;

#[cfg(all(target_arch = "x86_64", any(simd_level = "avx2", simd_level = "runtime")))]
mod avx2;

#[cfg(all(target_arch = "aarch64", simd_level = "neon"))]
mod neon;

// ============================================================================
// Public API
// ============================================================================

/// Compute dot product between float vector and 1-bit packed binary code.
///
/// Encoding: bit=0 → -1.0, bit=1 → +1.0
///
/// This is optimized using the algebraic identity:
/// `dot = 2 * sum(q where bit=1) - sum(q)`
///
/// # Arguments
/// * `query` - Float32 query vector of length N
/// * `code` - Packed binary code of length ceil(N/8) bytes
///
/// # Returns
/// Weighted sum representing the quantized dot product
///
/// # Panics
/// Debug-only: panics if code length doesn't match query dimensions
///
/// # Example
///
/// ```
/// use motlie_core::distance::quantized::dot_1bit;
///
/// let query = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let code = vec![0xFF]; // All bits set → all +1
/// let result = dot_1bit(&query, &code);
/// assert!((result - 36.0).abs() < 1e-5); // sum of query
/// ```
#[inline]
pub fn dot_1bit(query: &[f32], code: &[u8]) -> f32 {
    debug_assert!(
        code.len() >= (query.len() + 7) / 8,
        "Code length {} too short for {} dimensions",
        code.len(),
        query.len()
    );

    #[cfg(all(target_arch = "x86_64", simd_level = "avx2"))]
    {
        unsafe { avx2::dot_1bit(query, code) }
    }

    #[cfg(all(target_arch = "aarch64", simd_level = "neon"))]
    {
        unsafe { neon::dot_1bit(query, code) }
    }

    #[cfg(simd_level = "scalar")]
    {
        scalar::dot_1bit(query, code)
    }

    #[cfg(simd_level = "runtime")]
    {
        // For runtime dispatch, use scalar for now
        // TODO: Add runtime CPU detection when needed
        scalar::dot_1bit(query, code)
    }
}

/// Compute dot product between float vector and 2-bit packed code (Gray coded).
///
/// Decodes each 2-bit Gray code to a level (0-3), looks up the corresponding
/// value from the provided table, and computes the weighted sum.
///
/// # Arguments
/// * `query` - Float32 query vector of length N
/// * `code` - Packed 2-bit codes of length ceil(N*2/8) bytes
/// * `values` - Lookup table for levels 0-3. For RaBitQ: `[-1.5, -0.5, 0.5, 1.5]`
///
/// # Returns
/// Weighted sum representing the quantized dot product
///
/// # Example
///
/// ```
/// use motlie_core::distance::quantized::dot_2bit_lookup;
///
/// const LEVELS: [f32; 4] = [-1.5, -0.5, 0.5, 1.5];
/// let query = vec![1.0, 1.0, 1.0, 1.0];
/// // Encode levels 0,1,2,3 with Gray codes: 00, 01, 11, 10
/// let code = vec![0b10_11_01_00];
/// let result = dot_2bit_lookup(&query, &code, &LEVELS);
/// // -1.5 + -0.5 + 0.5 + 1.5 = 0.0
/// assert!(result.abs() < 1e-5);
/// ```
#[inline]
pub fn dot_2bit_lookup(query: &[f32], code: &[u8], values: &[f32; 4]) -> f32 {
    debug_assert!(
        code.len() >= (query.len() * 2 + 7) / 8,
        "Code length {} too short for {} dimensions (2-bit)",
        code.len(),
        query.len()
    );

    #[cfg(all(target_arch = "x86_64", simd_level = "avx2"))]
    {
        unsafe { avx2::dot_2bit_lookup(query, code, values) }
    }

    #[cfg(all(target_arch = "aarch64", simd_level = "neon"))]
    {
        unsafe { neon::dot_2bit_lookup(query, code, values) }
    }

    #[cfg(simd_level = "scalar")]
    {
        scalar::dot_2bit_lookup(query, code, values)
    }

    #[cfg(simd_level = "runtime")]
    {
        scalar::dot_2bit_lookup(query, code, values)
    }
}

/// Compute dot product between float vector and 4-bit packed code (Gray coded).
///
/// RaBitQ uses a linear mapping for 4-bit: `value = level * scale + offset`
/// This is more efficient than a 16-element lookup table.
///
/// # Arguments
/// * `query` - Float32 query vector of length N
/// * `code` - Packed 4-bit codes of length ceil(N/2) bytes
/// * `scale` - Scale factor. For RaBitQ: `1.0/3.75 ≈ 0.2667`
/// * `offset` - Offset. For RaBitQ: `-2.0`
///
/// # Value Mapping
/// For RaBitQ (scale=1/3.75, offset=-2.0):
/// - Level 0 → -2.0
/// - Level 7 → -0.133...
/// - Level 8 → +0.133...
/// - Level 15 → +2.0
///
/// # Returns
/// Weighted sum representing the quantized dot product
///
/// # Example
///
/// ```
/// use motlie_core::distance::quantized::dot_4bit_linear;
///
/// let query = vec![1.0, 1.0];
/// // Lower nibble = 0 (level 0), upper nibble = 8 (Gray code for level 15)
/// let code = vec![0x80];
/// let result = dot_4bit_linear(&query, &code, 1.0/3.75, -2.0);
/// // level 0 → -2.0, level 15 → +2.0, sum = 0.0
/// assert!(result.abs() < 1e-4);
/// ```
#[inline]
pub fn dot_4bit_linear(query: &[f32], code: &[u8], scale: f32, offset: f32) -> f32 {
    debug_assert!(
        code.len() >= (query.len() + 1) / 2,
        "Code length {} too short for {} dimensions (4-bit)",
        code.len(),
        query.len()
    );

    #[cfg(all(target_arch = "x86_64", simd_level = "avx2"))]
    {
        unsafe { avx2::dot_4bit_linear(query, code, scale, offset) }
    }

    #[cfg(all(target_arch = "aarch64", simd_level = "neon"))]
    {
        unsafe { neon::dot_4bit_linear(query, code, scale, offset) }
    }

    #[cfg(simd_level = "scalar")]
    {
        scalar::dot_4bit_linear(query, code, scale, offset)
    }

    #[cfg(simd_level = "runtime")]
    {
        scalar::dot_4bit_linear(query, code, scale, offset)
    }
}

/// Constants for RaBitQ quantization.
///
/// These are the standard values used by the RaBitQ algorithm.
pub mod rabitq {
    /// 2-bit level values: Gray code level 0-3 → float value
    pub const LEVEL_VALUES_2BIT: [f32; 4] = [-1.5, -0.5, 0.5, 1.5];

    /// 4-bit scale factor: converts level to value via `value = level * SCALE_4BIT + OFFSET_4BIT`
    pub const SCALE_4BIT: f32 = 1.0 / 3.75;

    /// 4-bit offset: converts level to value via `value = level * SCALE_4BIT + OFFSET_4BIT`
    pub const OFFSET_4BIT: f32 = -2.0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_1bit_all_positive() {
        let query: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let code = vec![0xFF]; // All bits set
        let result = dot_1bit(&query, &code);
        let expected: f32 = query.iter().sum(); // 36.0
        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn test_dot_1bit_all_negative() {
        let query: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let code = vec![0x00]; // No bits set
        let result = dot_1bit(&query, &code);
        let expected: f32 = -query.iter().sum::<f32>(); // -36.0
        assert!((result - expected).abs() < 1e-5);
    }

    #[test]
    fn test_dot_2bit_symmetric() {
        let query = vec![1.0, 1.0, 1.0, 1.0];
        // Levels 0,1,2,3 → values -1.5, -0.5, 0.5, 1.5 sum to 0
        // Gray codes: 0→0, 1→1, 2→3, 3→2
        // Packed: 0b10_11_01_00
        let code = vec![0b10110100];
        let result = dot_2bit_lookup(&query, &code, &rabitq::LEVEL_VALUES_2BIT);
        assert!(result.abs() < 1e-5);
    }

    #[test]
    fn test_dot_4bit_extremes() {
        let query = vec![1.0, 1.0];
        // Level 0 → -2.0, Level 15 → +2.0
        // Level 15 Gray = 15 ^ 7 = 8
        let code = vec![0x80]; // nibble0=0 (level 0), nibble1=8 (level 15)
        let result = dot_4bit_linear(&query, &code, rabitq::SCALE_4BIT, rabitq::OFFSET_4BIT);
        assert!(result.abs() < 1e-4);
    }

    #[test]
    fn test_dot_4bit_uniform() {
        // All level 7 or 8 should give near-zero values
        let query = vec![1.0; 16];
        // Level 7: Gray = 7 ^ 3 = 4
        // Level 8: Gray = 8 ^ 4 = 12
        // Using all level 7: value = 7/3.75 - 2.0 = 1.8667 - 2.0 = -0.1333
        let code = vec![0x44; 8]; // All nibbles = 4 = Gray(7)
        let result = dot_4bit_linear(&query, &code, rabitq::SCALE_4BIT, rabitq::OFFSET_4BIT);
        let expected = 16.0 * (7.0 / 3.75 - 2.0);
        assert!((result - expected).abs() < 1e-4);
    }
}
