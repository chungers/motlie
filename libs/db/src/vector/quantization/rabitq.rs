//! RaBitQ binary quantization encoder.
//!
//! RaBitQ is a training-free binary quantization method that uses random
//! orthonormal rotation to preserve distance relationships. Key properties:
//!
//! - **Training-free**: Uses deterministic random rotation (DATA-1 compliant)
//! - **Distance-preserving**: O(1/sqrt(D)) error bound for any data distribution
//! - **Incremental**: Works with streaming inserts (no batch retraining)
//!
//! # Algorithm
//!
//! 1. Generate orthonormal rotation matrix R from seed (Gram-Schmidt)
//! 2. Rotate vector: v' = R * v
//! 3. Quantize rotated vector to binary code
//!
//! # Compression Ratios (128D)
//!
//! | Bits | Code Size | Compression | Recall (no rerank) |
//! |------|-----------|-------------|-------------------|
//! | 1    | 16 bytes  | 32x         | ~50%              |
//! | 2    | 32 bytes  | 16x         | ~65%              |
//! | 4    | 64 bytes  | 8x          | ~80%              |
//!
//! Note: Multi-bit modes use Gray code encoding to ensure adjacent quantization
//! levels have Hamming distance 1 (see issue #43).
//!
//! # References
//!
//! - RaBitQ paper: <https://arxiv.org/abs/2405.12497>

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::vector::config::RaBitQConfig;
use motlie_core::distance::quantized as simd_quantized;

/// RaBitQ binary quantization encoder.
///
/// Encodes high-dimensional vectors to compact binary codes using
/// random orthonormal rotation followed by sign quantization.
#[derive(Clone)]
pub struct RaBitQ {
    /// Orthonormal rotation matrix [D x D] stored row-major.
    /// rotation[i * dim + j] = R[i][j]
    rotation: Vec<f32>,

    /// Vector dimensionality.
    dim: usize,

    /// Bits per dimension (1, 2, or 4).
    bits_per_dim: u8,

    /// Use SIMD-optimized dot products from motlie_core::distance::quantized.
    use_simd_dot: bool,
}

impl std::fmt::Debug for RaBitQ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RaBitQ")
            .field("dim", &self.dim)
            .field("bits_per_dim", &self.bits_per_dim)
            .field("rotation_size", &self.rotation.len())
            .field("use_simd_dot", &self.use_simd_dot)
            .finish()
    }
}

impl RaBitQ {
    /// Create a new RaBitQ encoder with SIMD enabled by default.
    ///
    /// # Arguments
    ///
    /// * `dim` - Vector dimensionality (must be > 0)
    /// * `bits_per_dim` - Bits per dimension (1, 2, or 4)
    /// * `seed` - Random seed for rotation matrix generation
    ///
    /// # Panics
    ///
    /// Panics if `dim` is 0 or `bits_per_dim` is not 1, 2, or 4.
    pub fn new(dim: usize, bits_per_dim: u8, seed: u64) -> Self {
        Self::with_options(dim, bits_per_dim, seed, true)
    }

    /// Create a new RaBitQ encoder with explicit SIMD option.
    ///
    /// # Arguments
    ///
    /// * `dim` - Vector dimensionality (must be > 0)
    /// * `bits_per_dim` - Bits per dimension (1, 2, or 4)
    /// * `seed` - Random seed for rotation matrix generation
    /// * `use_simd_dot` - Use SIMD-optimized dot products (true) or scalar (false)
    ///
    /// # Panics
    ///
    /// Panics if `dim` is 0 or `bits_per_dim` is not 1, 2, or 4.
    pub fn with_options(dim: usize, bits_per_dim: u8, seed: u64, use_simd_dot: bool) -> Self {
        assert!(dim > 0, "Dimension must be positive");
        assert!(
            bits_per_dim == 1 || bits_per_dim == 2 || bits_per_dim == 4,
            "bits_per_dim must be 1, 2, or 4"
        );

        let rotation = Self::generate_rotation_matrix(dim, seed);

        Self {
            rotation,
            dim,
            bits_per_dim,
            use_simd_dot,
        }
    }

    /// Create a RaBitQ encoder from configuration.
    pub fn from_config(dim: usize, config: &RaBitQConfig) -> Self {
        Self::with_options(dim, config.bits_per_dim, config.rotation_seed, config.use_simd_dot)
    }

    /// Get the dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the bits per dimension.
    pub fn bits_per_dim(&self) -> u8 {
        self.bits_per_dim
    }

    /// Get the code size in bytes.
    pub fn code_size(&self) -> usize {
        (self.dim * self.bits_per_dim as usize + 7) / 8
    }

    /// Generate a random orthonormal rotation matrix via Gram-Schmidt.
    ///
    /// Uses ChaCha20Rng for deterministic generation from seed.
    ///
    /// # Scaling for Unit Vectors
    ///
    /// The matrix is scaled by √D to ensure quantization thresholds work correctly
    /// for unit vectors. Without scaling, unit vector components after rotation have
    /// variance 1/D, causing 2-bit/4-bit thresholds to be ineffective (see issue #42).
    ///
    /// With √D scaling:
    /// - Rotated components have variance ≈ 1 (matching threshold assumptions)
    /// - 2-bit thresholds [-0.5, 0, 0.5] distribute values across all 4 levels
    /// - 4-bit thresholds [-2, 2] utilize all 16 levels effectively
    fn generate_rotation_matrix(dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        // Generate random matrix with normally distributed values
        let mut matrix: Vec<f32> = (0..dim * dim).map(|_| rng.gen::<f32>() - 0.5).collect();

        // Gram-Schmidt orthogonalization (column-wise)
        for i in 0..dim {
            // Subtract projections onto previous columns
            for j in 0..i {
                let dot = Self::column_dot(&matrix, dim, j, i);
                for k in 0..dim {
                    matrix[k * dim + i] -= dot * matrix[k * dim + j];
                }
            }

            // Normalize column i
            let norm = Self::column_norm(&matrix, dim, i);
            if norm > 1e-10 {
                for k in 0..dim {
                    matrix[k * dim + i] /= norm;
                }
            }
        }

        // Scale by √D so rotated unit vectors have component variance ≈ 1.
        // This ensures quantization thresholds (designed for variance=1) work correctly.
        // See issue #42 for detailed analysis.
        let scale = (dim as f32).sqrt();
        for val in &mut matrix {
            *val *= scale;
        }

        matrix
    }

    /// Compute dot product of two columns.
    fn column_dot(matrix: &[f32], dim: usize, col_a: usize, col_b: usize) -> f32 {
        (0..dim)
            .map(|row| matrix[row * dim + col_a] * matrix[row * dim + col_b])
            .sum()
    }

    /// Compute L2 norm of a column.
    fn column_norm(matrix: &[f32], dim: usize, col: usize) -> f32 {
        Self::column_dot(matrix, dim, col, col).sqrt()
    }

    /// Convert binary value to Gray code.
    ///
    /// Gray code ensures adjacent integers differ by exactly 1 bit,
    /// which is essential for multi-bit quantization with Hamming distance.
    ///
    /// Without Gray code, adjacent quantization levels can have maximum
    /// Hamming distance (e.g., level 1=01 and level 2=10 differ by 2 bits).
    ///
    /// # Examples
    ///
    /// - 2-bit: 0→00, 1→01, 2→11, 3→10
    /// - 4-bit: 0→0000, 1→0001, 2→0011, 3→0010, 4→0110, ...
    ///
    /// # Formula
    ///
    /// `gray = n ^ (n >> 1)`
    ///
    /// See issue #43 for detailed analysis.
    #[inline]
    const fn to_gray_code(n: u8) -> u8 {
        n ^ (n >> 1)
    }

    /// Apply rotation: rotated = R * vector.
    fn rotate(&self, vector: &[f32]) -> Vec<f32> {
        assert_eq!(
            vector.len(),
            self.dim,
            "Vector dimension mismatch: expected {}, got {}",
            self.dim,
            vector.len()
        );

        let mut rotated = vec![0.0f32; self.dim];

        // Matrix-vector multiplication: rotated[i] = sum_j(R[i][j] * v[j])
        for i in 0..self.dim {
            let mut sum = 0.0f32;
            for j in 0..self.dim {
                sum += self.rotation[i * self.dim + j] * vector[j];
            }
            rotated[i] = sum;
        }

        rotated
    }

    /// Encode a vector to a binary code.
    ///
    /// # Arguments
    ///
    /// * `vector` - Input vector (must match encoder dimension)
    ///
    /// # Returns
    ///
    /// Binary code as bytes. Size depends on `bits_per_dim`:
    /// - 1-bit: dim/8 bytes
    /// - 2-bit: dim/4 bytes
    /// - 4-bit: dim/2 bytes
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let rotated = self.rotate(vector);

        match self.bits_per_dim {
            1 => self.quantize_1bit(&rotated),
            2 => self.quantize_2bit(&rotated),
            4 => self.quantize_4bit(&rotated),
            _ => unreachable!("Invalid bits_per_dim"),
        }
    }

    /// 1-bit quantization: sign(x) -> {0, 1}.
    fn quantize_1bit(&self, rotated: &[f32]) -> Vec<u8> {
        let num_bytes = (self.dim + 7) / 8;
        let mut code = vec![0u8; num_bytes];

        for (i, &val) in rotated.iter().enumerate() {
            if val >= 0.0 {
                code[i / 8] |= 1 << (i % 8);
            }
        }

        code
    }

    /// 2-bit quantization: 4 levels based on value distribution.
    ///
    /// Thresholds at -0.5, 0.0, 0.5 (assuming roughly normal distribution).
    /// Uses Gray code encoding so adjacent levels have Hamming distance 1.
    ///
    /// Level mapping (with Gray code):
    /// - Level 0 (< -0.5): Gray 00
    /// - Level 1 (-0.5..0): Gray 01
    /// - Level 2 (0..0.5): Gray 11
    /// - Level 3 (>= 0.5): Gray 10
    fn quantize_2bit(&self, rotated: &[f32]) -> Vec<u8> {
        let num_bytes = (self.dim * 2 + 7) / 8;
        let mut code = vec![0u8; num_bytes];

        for (i, &val) in rotated.iter().enumerate() {
            // Map value to quantization level: 0 (< -0.5), 1 (-0.5..0), 2 (0..0.5), 3 (>= 0.5)
            let level = if val < -0.5 {
                0u8
            } else if val < 0.0 {
                1u8
            } else if val < 0.5 {
                2u8
            } else {
                3u8
            };

            // Convert to Gray code so adjacent levels have Hamming distance 1
            let gray = Self::to_gray_code(level);

            let bit_offset = i * 2;
            let byte_idx = bit_offset / 8;
            let bit_shift = bit_offset % 8;
            code[byte_idx] |= gray << bit_shift;
        }

        code
    }

    /// 4-bit quantization: 16 levels.
    ///
    /// Uniform quantization from -2.0 to 2.0 (clipped).
    /// Uses Gray code encoding so adjacent levels have Hamming distance 1.
    ///
    /// Level mapping: values are uniformly quantized to 0-15, then Gray coded.
    /// Gray code sequence: 0000, 0001, 0011, 0010, 0110, 0111, 0101, 0100,
    ///                     1100, 1101, 1111, 1110, 1010, 1011, 1001, 1000
    fn quantize_4bit(&self, rotated: &[f32]) -> Vec<u8> {
        let num_bytes = (self.dim * 4 + 7) / 8;
        let mut code = vec![0u8; num_bytes];

        for (i, &val) in rotated.iter().enumerate() {
            // Clamp to [-2.0, 2.0] and map to [0, 15]
            let clamped = val.clamp(-2.0, 2.0);
            let level = ((clamped + 2.0) * 3.75) as u8; // (val+2) / 4 * 15

            // Convert to Gray code so adjacent levels have Hamming distance 1
            let gray = Self::to_gray_code(level);

            let bit_offset = i * 4;
            let byte_idx = bit_offset / 8;
            let bit_shift = bit_offset % 8;

            // Handle nibble alignment
            if bit_shift == 0 {
                code[byte_idx] |= gray;
            } else {
                // Nibble spans two bytes (only for odd dimensions)
                code[byte_idx] |= gray << 4;
            }
        }

        code
    }

    /// Compute Hamming distance between two binary codes.
    ///
    /// Uses SIMD-optimized implementation from `motlie_core::distance`.
    /// Automatically selects best implementation (AVX-512, AVX2, NEON, scalar).
    #[inline]
    pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
        motlie_core::distance::hamming_distance(a, b)
    }

    /// Compute Hamming distance using SIMD-optimized implementation.
    ///
    /// Alias for `hamming_distance` - both now use SIMD.
    #[inline]
    pub fn hamming_distance_fast(a: &[u8], b: &[u8]) -> u32 {
        motlie_core::distance::hamming_distance(a, b)
    }

    /// Batch compute Hamming distances from a query code to multiple candidates.
    ///
    /// Returns distances paired with indices for sorting.
    pub fn batch_hamming_distances(query_code: &[u8], candidate_codes: &[&[u8]]) -> Vec<u32> {
        candidate_codes
            .iter()
            .map(|code| Self::hamming_distance_fast(query_code, code))
            .collect()
    }

    // =========================================================================
    // ADC (Asymmetric Distance Computation) Methods
    // =========================================================================

    /// Convert Gray code back to binary.
    ///
    /// Inverse of `to_gray_code()`. Used for decoding multi-bit quantization levels.
    #[inline]
    const fn from_gray_code(gray: u8) -> u8 {
        let mut n = gray;
        let mut mask = n >> 1;
        while mask != 0 {
            n ^= mask;
            mask >>= 1;
        }
        n
    }

    /// Non-const version for use in const contexts workaround.
    #[inline]
    fn from_gray_code_runtime(gray: u8) -> u8 {
        let mut n = gray;
        n ^= n >> 4;
        n ^= n >> 2;
        n ^= n >> 1;
        n
    }

    /// Rotate query vector without quantizing.
    ///
    /// In ADC mode, the query is rotated but kept as float32 (not binarized).
    /// This preserves full precision for the asymmetric distance computation.
    #[inline]
    pub fn rotate_query(&self, query: &[f32]) -> Vec<f32> {
        self.rotate(query)
    }

    /// Encode vector with ADC corrective factors.
    ///
    /// Returns both the binary code and the correction factors needed for
    /// accurate ADC distance estimation.
    ///
    /// # Arguments
    ///
    /// * `vector` - Input vector (should be normalized for best results)
    ///
    /// # Returns
    ///
    /// Tuple of (binary_code, AdcCorrection)
    pub fn encode_with_correction(
        &self,
        vector: &[f32],
    ) -> (Vec<u8>, crate::vector::schema::AdcCorrection) {
        use crate::vector::schema::AdcCorrection;

        // Compute vector norm
        let vector_norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Normalize vector for encoding
        let normalized: Vec<f32> = if vector_norm > 1e-10 {
            vector.iter().map(|x| x / vector_norm).collect()
        } else {
            vector.to_vec()
        };

        // Rotate and encode
        let rotated = self.rotate(&normalized);
        let code = match self.bits_per_dim {
            1 => self.quantize_1bit(&rotated),
            2 => self.quantize_2bit(&rotated),
            4 => self.quantize_4bit(&rotated),
            _ => unreachable!(),
        };

        // Compute quantization error: dot product of decoded quantized vector with rotated original
        // This measures how well the binary code represents the rotated vector
        let decoded = self.decode_to_float(&code);
        let quantization_error: f32 = rotated
            .iter()
            .zip(decoded.iter())
            .map(|(r, d)| r * d)
            .sum();

        // Ensure quantization_error is not zero to avoid division by zero
        let quantization_error = if quantization_error.abs() < 1e-10 {
            1.0
        } else {
            quantization_error
        };

        (code, AdcCorrection::new(vector_norm, quantization_error))
    }

    /// Decode binary code to float representation.
    ///
    /// Used for computing quantization error correction factor.
    fn decode_to_float(&self, code: &[u8]) -> Vec<f32> {
        match self.bits_per_dim {
            1 => self.decode_1bit(code),
            2 => self.decode_2bit(code),
            4 => self.decode_4bit(code),
            _ => unreachable!(),
        }
    }

    /// Decode 1-bit code to float: 0 → -1.0, 1 → +1.0
    fn decode_1bit(&self, code: &[u8]) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.dim);
        for i in 0..self.dim {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            let bit = (code[byte_idx] >> bit_idx) & 1;
            result.push(if bit == 1 { 1.0 } else { -1.0 });
        }
        result
    }

    /// Decode 2-bit Gray code to float: levels 0,1,2,3 → -1.5, -0.5, +0.5, +1.5
    fn decode_2bit(&self, code: &[u8]) -> Vec<f32> {
        const LEVEL_VALUES: [f32; 4] = [-1.5, -0.5, 0.5, 1.5];
        let mut result = Vec::with_capacity(self.dim);
        for i in 0..self.dim {
            let bit_offset = i * 2;
            let byte_idx = bit_offset / 8;
            let bit_shift = bit_offset % 8;
            let gray = (code[byte_idx] >> bit_shift) & 0b11;
            let level = Self::from_gray_code_runtime(gray) as usize;
            result.push(LEVEL_VALUES[level.min(3)]);
        }
        result
    }

    /// Decode 4-bit Gray code to float: levels 0-15 → -2.0 to +2.0
    fn decode_4bit(&self, code: &[u8]) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.dim);
        for i in 0..self.dim {
            let bit_offset = i * 4;
            let byte_idx = bit_offset / 8;
            let bit_shift = bit_offset % 8;
            let gray = if bit_shift == 0 {
                code[byte_idx] & 0x0F
            } else {
                (code[byte_idx] >> 4) & 0x0F
            };
            let level = Self::from_gray_code_runtime(gray);
            // Map level 0-15 to -2.0 to +2.0 (inverse of encode: level = (val+2)*3.75)
            let value = (level as f32 / 3.75) - 2.0;
            result.push(value);
        }
        result
    }

    /// Compute ADC distance estimate between query and stored vector.
    ///
    /// This is the core ADC operation that avoids the symmetric Hamming problem.
    /// The query remains as float32, and we compute a weighted dot product with
    /// the binary code, corrected by the stored factors.
    ///
    /// # Arguments
    ///
    /// * `query_rotated` - Rotated query vector (from `rotate_query()`)
    /// * `query_norm` - L2 norm of original query: ||q||
    /// * `code` - Binary code of stored vector
    /// * `correction` - ADC correction factors for stored vector
    ///
    /// # Returns
    ///
    /// Estimated distance (lower = more similar for cosine-like behavior)
    pub fn adc_distance(
        &self,
        query_rotated: &[f32],
        query_norm: f32,
        code: &[u8],
        correction: &crate::vector::schema::AdcCorrection,
    ) -> f32 {
        // Compute binary dot product: query_rotated · decode(code)
        let binary_dot = self.binary_dot_product(query_rotated, code);

        // Estimate inner product using correction factor
        // ⟨q, v⟩ ≈ binary_dot / quantization_error
        let inner_product_est = binary_dot / correction.quantization_error;

        // For cosine distance: 1 - cos(θ) = 1 - ⟨q,v⟩/(||q||·||v||)
        // We return a distance-like value where lower = more similar
        let cos_sim = inner_product_est / (query_norm * correction.vector_norm + 1e-10);

        // Clamp to valid range and convert to distance
        1.0 - cos_sim.clamp(-1.0, 1.0)
    }

    /// Compute binary dot product: float query × binary code.
    ///
    /// Dispatches to SIMD-optimized (motlie_core) or scalar implementation
    /// based on `use_simd_dot` configuration.
    #[inline]
    pub fn binary_dot_product(&self, query: &[f32], code: &[u8]) -> f32 {
        if self.use_simd_dot {
            // Use SIMD-optimized implementations from motlie_core
            match self.bits_per_dim {
                1 => simd_quantized::dot_1bit(query, code),
                2 => simd_quantized::dot_2bit_lookup(
                    query,
                    code,
                    &simd_quantized::rabitq::LEVEL_VALUES_2BIT,
                ),
                4 => simd_quantized::dot_4bit_linear(
                    query,
                    code,
                    simd_quantized::rabitq::SCALE_4BIT,
                    simd_quantized::rabitq::OFFSET_4BIT,
                ),
                _ => unreachable!(),
            }
        } else {
            // Use local scalar implementations
            match self.bits_per_dim {
                1 => self.binary_dot_1bit(query, code),
                2 => self.binary_dot_2bit(query, code),
                4 => self.binary_dot_4bit(query, code),
                _ => unreachable!(),
            }
        }
    }

    /// 1-bit binary dot product (optimized).
    ///
    /// For 1-bit: decode maps bit=0 → -1, bit=1 → +1
    /// So: dot = sum(q where bit=1) - sum(q where bit=0)
    ///         = 2 * sum(q where bit=1) - sum(all q)
    ///
    /// This avoids per-element branches.
    fn binary_dot_1bit(&self, query: &[f32], code: &[u8]) -> f32 {
        let query_sum: f32 = query.iter().sum();
        let mut positive_sum = 0.0f32;

        for i in 0..self.dim {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            if (code[byte_idx] >> bit_idx) & 1 == 1 {
                positive_sum += query[i];
            }
        }

        // dot = positive_sum - negative_sum
        //     = positive_sum - (query_sum - positive_sum)
        //     = 2 * positive_sum - query_sum
        2.0 * positive_sum - query_sum
    }

    /// 2-bit binary dot product.
    ///
    /// Decodes Gray code to level, maps to value, multiplies by query.
    fn binary_dot_2bit(&self, query: &[f32], code: &[u8]) -> f32 {
        const LEVEL_VALUES: [f32; 4] = [-1.5, -0.5, 0.5, 1.5];
        let mut sum = 0.0f32;

        for i in 0..self.dim {
            let bit_offset = i * 2;
            let byte_idx = bit_offset / 8;
            let bit_shift = bit_offset % 8;
            let gray = (code[byte_idx] >> bit_shift) & 0b11;
            let level = Self::from_gray_code_runtime(gray) as usize;
            sum += query[i] * LEVEL_VALUES[level.min(3)];
        }

        sum
    }

    /// 4-bit binary dot product.
    ///
    /// Decodes Gray code to level, maps to value in [-2, 2], multiplies by query.
    fn binary_dot_4bit(&self, query: &[f32], code: &[u8]) -> f32 {
        let mut sum = 0.0f32;

        for i in 0..self.dim {
            let bit_offset = i * 4;
            let byte_idx = bit_offset / 8;
            let bit_shift = bit_offset % 8;
            let gray = if bit_shift == 0 {
                code[byte_idx] & 0x0F
            } else {
                (code[byte_idx] >> 4) & 0x0F
            };
            let level = Self::from_gray_code_runtime(gray);
            // Map level 0-15 to -2.0 to +2.0 (inverse of encode: level = (val+2)*3.75)
            let value = (level as f32 / 3.75) - 2.0;
            sum += query[i] * value;
        }

        sum
    }

    /// Check if rotation matrix is scaled orthonormal (for testing).
    ///
    /// After √D scaling, the matrix satisfies R * R^T = D * I (not I).
    #[cfg(test)]
    fn is_scaled_orthonormal(&self, tolerance: f32) -> bool {
        // Check R * R^T = D * I (scaled identity due to √D scaling)
        let scale_sq = self.dim as f32; // D = (√D)²
        for i in 0..self.dim {
            for j in 0..self.dim {
                // Compute (R * R^T)[i][j] = sum_k R[i][k] * R[j][k]
                let mut dot = 0.0f32;
                for k in 0..self.dim {
                    dot += self.rotation[i * self.dim + k] * self.rotation[j * self.dim + k];
                }

                let expected = if i == j { scale_sq } else { 0.0 };
                if (dot - expected).abs() > tolerance * scale_sq {
                    return false;
                }
            }
        }
        true
    }

    /// Compute variance of rotated unit vector components (for testing).
    ///
    /// For a random unit vector, after √D-scaled rotation, components should
    /// have variance ≈ 1 (not 1/D as with unscaled orthonormal rotation).
    #[cfg(test)]
    fn rotated_component_variance(&self, num_samples: usize, seed: u64) -> f32 {
        use rand::Rng;
        let mut rng = ChaCha20Rng::seed_from_u64(seed);

        let mut all_components = Vec::with_capacity(num_samples * self.dim);

        for _ in 0..num_samples {
            // Generate random unit vector
            let mut v: Vec<f32> = (0..self.dim).map(|_| rng.gen::<f32>() - 0.5).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm;
            }

            // Rotate and collect components
            let rotated = self.rotate(&v);
            all_components.extend(rotated);
        }

        // Compute variance: E[X²] - E[X]²
        let n = all_components.len() as f32;
        let mean: f32 = all_components.iter().sum::<f32>() / n;
        let variance: f32 =
            all_components.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;

        variance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotation_matrix_is_scaled_orthonormal() {
        let encoder = RaBitQ::new(128, 1, 42);
        assert!(
            encoder.is_scaled_orthonormal(1e-4),
            "Rotation matrix should satisfy R*R^T = D*I after √D scaling"
        );
    }

    /// Verify that rotated unit vectors have component variance ≈ 1 (issue #42 fix).
    ///
    /// Before the fix, variance was 1/D ≈ 0.0078 for D=128.
    /// After √D scaling, variance should be ≈ 1.
    #[test]
    fn test_rotated_unit_vector_variance() {
        let encoder = RaBitQ::new(128, 1, 42);
        let variance = encoder.rotated_component_variance(1000, 123);

        // Variance should be close to 1.0 (within 20% tolerance for sampling noise)
        assert!(
            (0.8..1.2).contains(&variance),
            "Rotated unit vector variance should be ≈1.0, got {:.4}. \
             Before fix it would be ≈{:.4}",
            variance,
            1.0 / 128.0
        );
    }

    /// Verify 2-bit quantization uses all 4 levels for unit vectors (issue #42 fix).
    ///
    /// Before the fix, only levels 1 and 2 were used (sign only).
    /// After fix, all 4 levels should be populated.
    #[test]
    fn test_2bit_uses_all_levels() {
        use rand::Rng;
        let encoder = RaBitQ::new(128, 2, 42);
        let mut rng = ChaCha20Rng::seed_from_u64(999);

        let mut level_counts = [0u32; 4];

        // Encode many random unit vectors
        for _ in 0..500 {
            let mut v: Vec<f32> = (0..128).map(|_| rng.gen::<f32>() - 0.5).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm;
            }

            let code = encoder.encode(&v);

            // Count levels in the 2-bit code
            for byte in &code {
                for shift in (0..8).step_by(2) {
                    let level = (byte >> shift) & 0b11;
                    level_counts[level as usize] += 1;
                }
            }
        }

        // All 4 levels should have significant counts
        let total: u32 = level_counts.iter().sum();
        let min_expected = total / 20; // At least 5% in each level

        for (level, &count) in level_counts.iter().enumerate() {
            assert!(
                count >= min_expected,
                "2-bit level {} has only {} counts ({}%), expected at least {}%. \
                 Level distribution: {:?}. Before fix, levels 0 and 3 would be ~0%.",
                level,
                count,
                count * 100 / total,
                min_expected * 100 / total,
                level_counts
            );
        }
    }

    /// Verify 4-bit quantization uses many levels for unit vectors (issue #42 fix).
    ///
    /// Before the fix, only ~3 central levels were used.
    /// After fix, values should spread across many levels.
    #[test]
    fn test_4bit_uses_many_levels() {
        use rand::Rng;
        let encoder = RaBitQ::new(128, 4, 42);
        let mut rng = ChaCha20Rng::seed_from_u64(888);

        let mut level_counts = [0u32; 16];

        // Encode many random unit vectors
        for _ in 0..500 {
            let mut v: Vec<f32> = (0..128).map(|_| rng.gen::<f32>() - 0.5).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm;
            }

            let code = encoder.encode(&v);

            // Count levels in the 4-bit code
            for (i, byte) in code.iter().enumerate() {
                // Even indices: low nibble, odd indices: high nibble
                if i * 2 < 128 {
                    level_counts[(byte & 0x0F) as usize] += 1;
                }
                if i * 2 + 1 < 128 {
                    level_counts[((byte >> 4) & 0x0F) as usize] += 1;
                }
            }
        }

        // Count how many of the 16 levels have non-trivial usage (>1%)
        let total: u32 = level_counts.iter().sum();
        let threshold = total / 100; // 1% threshold
        let levels_used = level_counts.iter().filter(|&&c| c > threshold).count();

        assert!(
            levels_used >= 8,
            "4-bit should use at least 8 of 16 levels significantly, but only {} used. \
             Before fix, only ~3 levels would be used. Distribution: {:?}",
            levels_used,
            level_counts
        );
    }

    #[test]
    fn test_rotation_matrix_deterministic() {
        let encoder1 = RaBitQ::new(128, 1, 42);
        let encoder2 = RaBitQ::new(128, 1, 42);

        assert_eq!(
            encoder1.rotation, encoder2.rotation,
            "Same seed should produce same rotation matrix"
        );
    }

    #[test]
    fn test_rotation_matrix_different_seeds() {
        let encoder1 = RaBitQ::new(128, 1, 42);
        let encoder2 = RaBitQ::new(128, 1, 123);

        assert_ne!(
            encoder1.rotation, encoder2.rotation,
            "Different seeds should produce different matrices"
        );
    }

    #[test]
    fn test_encode_1bit_size() {
        let encoder = RaBitQ::new(128, 1, 42);
        let vector: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
        let code = encoder.encode(&vector);

        assert_eq!(code.len(), 16, "128D 1-bit code should be 16 bytes");
    }

    #[test]
    fn test_encode_2bit_size() {
        let encoder = RaBitQ::new(128, 2, 42);
        let vector: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
        let code = encoder.encode(&vector);

        assert_eq!(code.len(), 32, "128D 2-bit code should be 32 bytes");
    }

    #[test]
    fn test_encode_4bit_size() {
        let encoder = RaBitQ::new(128, 4, 42);
        let vector: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
        let code = encoder.encode(&vector);

        assert_eq!(code.len(), 64, "128D 4-bit code should be 64 bytes");
    }

    #[test]
    fn test_encode_deterministic() {
        let encoder = RaBitQ::new(128, 1, 42);
        let vector: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();

        let code1 = encoder.encode(&vector);
        let code2 = encoder.encode(&vector);

        assert_eq!(code1, code2, "Same vector should produce same code");
    }

    #[test]
    fn test_hamming_distance_identical() {
        let encoder = RaBitQ::new(128, 1, 42);
        let vector: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
        let code = encoder.encode(&vector);

        assert_eq!(
            RaBitQ::hamming_distance(&code, &code),
            0,
            "Identical codes should have distance 0"
        );
    }

    #[test]
    fn test_hamming_distance_opposite() {
        let code_a = vec![0x00u8; 16];
        let code_b = vec![0xFFu8; 16];

        assert_eq!(
            RaBitQ::hamming_distance(&code_a, &code_b),
            128,
            "Opposite codes should have max distance"
        );
    }

    #[test]
    fn test_similar_vectors_low_distance() {
        let encoder = RaBitQ::new(128, 1, 42);

        // Two similar vectors
        let v1: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
        let v2: Vec<f32> = (0..128).map(|i| (i as f32 / 128.0) + 0.01).collect();

        let code1 = encoder.encode(&v1);
        let code2 = encoder.encode(&v2);

        let distance = RaBitQ::hamming_distance(&code1, &code2);
        assert!(
            distance < 20,
            "Similar vectors should have low Hamming distance, got {}",
            distance
        );
    }

    #[test]
    fn test_dissimilar_vectors_high_distance() {
        let encoder = RaBitQ::new(128, 1, 42);

        // Two dissimilar vectors
        let v1: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
        let v2: Vec<f32> = (0..128).map(|i| -(i as f32) / 128.0).collect();

        let code1 = encoder.encode(&v1);
        let code2 = encoder.encode(&v2);

        let distance = RaBitQ::hamming_distance(&code1, &code2);
        assert!(
            distance > 40,
            "Dissimilar vectors should have high Hamming distance, got {}",
            distance
        );
    }

    #[test]
    fn test_hamming_distance_fast_matches_regular() {
        let code_a: Vec<u8> = (0..32).map(|i| i as u8).collect();
        let code_b: Vec<u8> = (0..32).map(|i| (i * 2) as u8).collect();

        let regular = RaBitQ::hamming_distance(&code_a, &code_b);
        let fast = RaBitQ::hamming_distance_fast(&code_a, &code_b);

        assert_eq!(regular, fast, "Fast and regular should match");
    }

    #[test]
    fn test_batch_hamming_distances() {
        let query = vec![0xAAu8; 16];
        let candidates: Vec<Vec<u8>> = vec![
            vec![0xAAu8; 16], // Same
            vec![0x55u8; 16], // Opposite bits
            vec![0x00u8; 16], // Half bits different
        ];
        let candidate_refs: Vec<&[u8]> = candidates.iter().map(|c| c.as_slice()).collect();

        let distances = RaBitQ::batch_hamming_distances(&query, &candidate_refs);

        assert_eq!(distances[0], 0, "Identical should be 0");
        assert_eq!(distances[1], 128, "Opposite should be 128");
        assert_eq!(distances[2], 64, "Half different should be 64");
    }

    #[test]
    fn test_from_config() {
        let config = RaBitQConfig {
            bits_per_dim: 2,
            rotation_seed: 123,
            enabled: true,
            ..Default::default()
        };

        let encoder = RaBitQ::from_config(768, &config);
        assert_eq!(encoder.dim(), 768);
        assert_eq!(encoder.bits_per_dim(), 2);
        assert_eq!(encoder.code_size(), 192); // 768 * 2 / 8
    }

    #[test]
    fn test_code_size() {
        assert_eq!(RaBitQ::new(128, 1, 0).code_size(), 16);
        assert_eq!(RaBitQ::new(128, 2, 0).code_size(), 32);
        assert_eq!(RaBitQ::new(128, 4, 0).code_size(), 64);

        assert_eq!(RaBitQ::new(768, 1, 0).code_size(), 96);
        assert_eq!(RaBitQ::new(1536, 1, 0).code_size(), 192);
    }

    // ==================== Gray Code Tests (Issue #43) ====================

    /// Verify Gray code conversion produces expected values.
    #[test]
    fn test_gray_code_values() {
        // 2-bit Gray code: 0→00, 1→01, 2→11, 3→10
        assert_eq!(RaBitQ::to_gray_code(0), 0b00);
        assert_eq!(RaBitQ::to_gray_code(1), 0b01);
        assert_eq!(RaBitQ::to_gray_code(2), 0b11);
        assert_eq!(RaBitQ::to_gray_code(3), 0b10);

        // 4-bit Gray code first 8 values
        assert_eq!(RaBitQ::to_gray_code(0), 0b0000);
        assert_eq!(RaBitQ::to_gray_code(1), 0b0001);
        assert_eq!(RaBitQ::to_gray_code(2), 0b0011);
        assert_eq!(RaBitQ::to_gray_code(3), 0b0010);
        assert_eq!(RaBitQ::to_gray_code(4), 0b0110);
        assert_eq!(RaBitQ::to_gray_code(5), 0b0111);
        assert_eq!(RaBitQ::to_gray_code(6), 0b0101);
        assert_eq!(RaBitQ::to_gray_code(7), 0b0100);
    }

    /// Gemini's validation test: adjacent levels must have Hamming distance 1.
    ///
    /// This is the key property that fixes issue #43. Without Gray code,
    /// level 1 (01) and level 2 (10) have Hamming distance 2.
    #[test]
    fn test_gray_code_adjacent_differ_by_one_bit() {
        // Test 2-bit range (0-3)
        for level in 0..3u8 {
            let gray_a = RaBitQ::to_gray_code(level);
            let gray_b = RaBitQ::to_gray_code(level + 1);
            let hamming = (gray_a ^ gray_b).count_ones();
            assert_eq!(
                hamming, 1,
                "2-bit: Adjacent levels {} and {} should have Hamming distance 1, got {}. \
                 Gray codes: {:02b} and {:02b}",
                level,
                level + 1,
                hamming,
                gray_a,
                gray_b
            );
        }

        // Test 4-bit range (0-15)
        for level in 0..15u8 {
            let gray_a = RaBitQ::to_gray_code(level);
            let gray_b = RaBitQ::to_gray_code(level + 1);
            let hamming = (gray_a ^ gray_b).count_ones();
            assert_eq!(
                hamming, 1,
                "4-bit: Adjacent levels {} and {} should have Hamming distance 1, got {}. \
                 Gray codes: {:04b} and {:04b}",
                level,
                level + 1,
                hamming,
                gray_a,
                gray_b
            );
        }
    }

    /// Verify that similar vectors have lower Hamming distance with Gray code.
    ///
    /// This test encodes vectors with gradually changing values and verifies
    /// that the Hamming distance increases proportionally.
    #[test]
    fn test_2bit_gray_code_distance_ordering() {
        let encoder = RaBitQ::new(128, 2, 42);

        // Create three vectors: base, slightly different, very different
        let base: Vec<f32> = (0..128).map(|i| (i as f32 / 128.0) - 0.5).collect();
        let similar: Vec<f32> = base.iter().map(|x| x + 0.1).collect();
        let different: Vec<f32> = base.iter().map(|x| -x).collect();

        let code_base = encoder.encode(&base);
        let code_similar = encoder.encode(&similar);
        let code_different = encoder.encode(&different);

        let dist_similar = RaBitQ::hamming_distance(&code_base, &code_similar);
        let dist_different = RaBitQ::hamming_distance(&code_base, &code_different);

        assert!(
            dist_similar < dist_different,
            "Similar vector should have lower Hamming distance than different vector. \
             Got similar={}, different={}",
            dist_similar,
            dist_different
        );
    }

    /// Verify that similar vectors have lower Hamming distance with 4-bit Gray code.
    #[test]
    fn test_4bit_gray_code_distance_ordering() {
        let encoder = RaBitQ::new(128, 4, 42);

        // Create three vectors: base, slightly different, very different
        let base: Vec<f32> = (0..128).map(|i| (i as f32 / 128.0) - 0.5).collect();
        let similar: Vec<f32> = base.iter().map(|x| x + 0.1).collect();
        let different: Vec<f32> = base.iter().map(|x| -x).collect();

        let code_base = encoder.encode(&base);
        let code_similar = encoder.encode(&similar);
        let code_different = encoder.encode(&different);

        let dist_similar = RaBitQ::hamming_distance(&code_base, &code_similar);
        let dist_different = RaBitQ::hamming_distance(&code_base, &code_different);

        assert!(
            dist_similar < dist_different,
            "Similar vector should have lower Hamming distance than different vector. \
             Got similar={}, different={}",
            dist_similar,
            dist_different
        );
    }
}
