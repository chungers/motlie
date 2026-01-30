//! Scalar (portable) implementations of quantized dot product operations.
//!
//! These functions compute dot products between high-precision float vectors
//! and low-precision packed binary codes. Used as fallback when SIMD is unavailable.

#![allow(dead_code)] // Fallback implementations - may be unused when SIMD is available

/// Decode Gray code to binary.
///
/// Gray code is used in RaBitQ to ensure adjacent quantization levels differ by
/// exactly one bit, providing better quantization properties.
#[inline]
fn from_gray_code(gray: u8) -> u8 {
    let mut n = gray;
    n ^= n >> 4;
    n ^= n >> 2;
    n ^= n >> 1;
    n
}

/// Compute dot product between float vector and 1-bit packed binary code.
///
/// For 1-bit encoding: bit=0 → -1.0, bit=1 → +1.0
///
/// # Optimization
/// Instead of per-element branches, we use:
/// `dot = 2 * sum(q where bit=1) - sum(q)`
///
/// This is derived from:
/// - `dot = sum(q where bit=1) - sum(q where bit=0)`
/// - `sum(q where bit=0) = sum(q) - sum(q where bit=1)`
/// - Therefore: `dot = 2 * sum(q where bit=1) - sum(q)`
///
/// # Arguments
/// * `query` - Float32 query vector
/// * `code` - Packed binary code (8 bits per byte, LSB first)
///
/// # Returns
/// Dot product approximation
#[inline]
pub fn dot_1bit(query: &[f32], code: &[u8]) -> f32 {
    let query_sum: f32 = query.iter().sum();
    let mut positive_sum = 0.0f32;

    for (i, &q) in query.iter().enumerate() {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        if (code[byte_idx] >> bit_idx) & 1 == 1 {
            positive_sum += q;
        }
    }

    2.0 * positive_sum - query_sum
}

/// Compute dot product between float vector and 2-bit packed code (Gray coded).
///
/// Decodes each 2-bit Gray code to a level (0-3), looks up the corresponding
/// float value from the provided table, and computes the weighted sum.
///
/// # Arguments
/// * `query` - Float32 query vector
/// * `code` - Packed 2-bit codes (4 values per byte, LSB first)
/// * `values` - Lookup table mapping levels 0-3 to float values.
///              For RaBitQ: `[-1.5, -0.5, 0.5, 1.5]`
///
/// # Returns
/// Dot product approximation
#[inline]
pub fn dot_2bit_lookup(query: &[f32], code: &[u8], values: &[f32; 4]) -> f32 {
    let mut sum = 0.0f32;

    for (i, &q) in query.iter().enumerate() {
        let bit_offset = i * 2;
        let byte_idx = bit_offset / 8;
        let bit_shift = bit_offset % 8;
        let gray = (code[byte_idx] >> bit_shift) & 0b11;
        let level = from_gray_code(gray) as usize;
        sum += q * values[level.min(3)];
    }

    sum
}

/// Compute dot product between float vector and 4-bit packed code (Gray coded).
///
/// Since RaBitQ uses a linear mapping for 4-bit (level → value), this function
/// takes scale and offset parameters instead of a 16-element lookup table.
///
/// The value mapping is: `value = (level * scale) + offset`
///
/// For RaBitQ: `scale = 1/3.75 ≈ 0.2667`, `offset = -2.0`
/// This maps levels 0-15 to values -2.0 to +2.0
///
/// # Arguments
/// * `query` - Float32 query vector
/// * `code` - Packed 4-bit codes (2 values per byte, LSB first)
/// * `scale` - Scale factor for linear mapping
/// * `offset` - Offset for linear mapping
///
/// # Returns
/// Dot product approximation
#[inline]
pub fn dot_4bit_linear(query: &[f32], code: &[u8], scale: f32, offset: f32) -> f32 {
    let mut sum = 0.0f32;

    for (i, &q) in query.iter().enumerate() {
        let bit_offset = i * 4;
        let byte_idx = bit_offset / 8;
        let bit_shift = bit_offset % 8;
        let gray = if bit_shift == 0 {
            code[byte_idx] & 0x0F
        } else {
            (code[byte_idx] >> 4) & 0x0F
        };
        let level = from_gray_code(gray);
        let value = (level as f32 * scale) + offset;
        sum += q * value;
    }

    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gray_code_roundtrip() {
        // Verify Gray code decoding
        // Gray(0)=0, Gray(1)=1, Gray(2)=3, Gray(3)=2
        assert_eq!(from_gray_code(0b00), 0);
        assert_eq!(from_gray_code(0b01), 1);
        assert_eq!(from_gray_code(0b11), 2);
        assert_eq!(from_gray_code(0b10), 3);

        // 4-bit Gray codes
        for n in 0u8..16 {
            let gray = n ^ (n >> 1);
            assert_eq!(from_gray_code(gray), n);
        }
    }

    #[test]
    fn test_dot_1bit_basic() {
        // All bits = 1: should map to all +1s
        let query = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let code = vec![0xFF]; // All 8 bits set
        let result = dot_1bit(&query, &code);
        // All values contribute as +1, so dot = sum(query) = 36
        assert!((result - 36.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_1bit_alternating() {
        // Alternating bits: 0b10101010 = bits 1,3,5,7 set
        let query = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let code = vec![0b10101010];
        let result = dot_1bit(&query, &code);
        // positive_sum = 2 + 4 + 6 + 8 = 20
        // query_sum = 36
        // dot = 2 * 20 - 36 = 4
        assert!((result - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_2bit_lookup() {
        const LEVEL_VALUES: [f32; 4] = [-1.5, -0.5, 0.5, 1.5];
        // Encode level 0 (Gray=0), level 1 (Gray=1), level 2 (Gray=3), level 3 (Gray=2)
        // 4 values: 0, 1, 2, 3 → Gray: 00, 01, 11, 10
        // Packed: byte0 = 0b10_11_01_00 = 0b10110100 = 180
        let query = vec![1.0, 1.0, 1.0, 1.0];
        let code = vec![0b10110100];
        let result = dot_2bit_lookup(&query, &code, &LEVEL_VALUES);
        // levels 0,1,2,3 → values -1.5, -0.5, 0.5, 1.5
        // dot = 1*(-1.5) + 1*(-0.5) + 1*(0.5) + 1*(1.5) = 0.0
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_4bit_linear() {
        const SCALE: f32 = 1.0 / 3.75;
        const OFFSET: f32 = -2.0;

        // Single dimension test: level 0 → value = -2.0
        let query = vec![1.0, 1.0];
        // Two nibbles: level 0 (Gray=0), level 15 (Gray=8, since 15^7=8)
        // level 15 Gray code: 15 ^ (15>>1) = 15 ^ 7 = 8
        let code = vec![0x80]; // lower nibble=0 (level 0), upper nibble=8 (level 15)
        let result = dot_4bit_linear(&query, &code, SCALE, OFFSET);
        // level 0 → -2.0, level 15 → 15/3.75 - 2.0 = 4.0 - 2.0 = 2.0
        // dot = 1*(-2.0) + 1*(2.0) = 0.0
        assert!((result - 0.0).abs() < 1e-5);
    }
}
