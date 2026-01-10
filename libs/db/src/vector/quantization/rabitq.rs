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
//! | 1    | 16 bytes  | 32x         | ~70%              |
//! | 2    | 32 bytes  | 16x         | ~85%              |
//! | 4    | 64 bytes  | 8x          | ~92%              |
//!
//! # References
//!
//! - RaBitQ paper: <https://arxiv.org/abs/2405.12497>

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use crate::vector::config::RaBitQConfig;

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
}

impl std::fmt::Debug for RaBitQ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RaBitQ")
            .field("dim", &self.dim)
            .field("bits_per_dim", &self.bits_per_dim)
            .field("rotation_size", &self.rotation.len())
            .finish()
    }
}

impl RaBitQ {
    /// Create a new RaBitQ encoder.
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
        }
    }

    /// Create a RaBitQ encoder from configuration.
    pub fn from_config(dim: usize, config: &RaBitQConfig) -> Self {
        Self::new(dim, config.bits_per_dim, config.rotation_seed)
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
    fn quantize_2bit(&self, rotated: &[f32]) -> Vec<u8> {
        let num_bytes = (self.dim * 2 + 7) / 8;
        let mut code = vec![0u8; num_bytes];

        for (i, &val) in rotated.iter().enumerate() {
            // Map value to 2-bit code: 0 (< -0.5), 1 (-0.5..0), 2 (0..0.5), 3 (>= 0.5)
            let level = if val < -0.5 {
                0u8
            } else if val < 0.0 {
                1u8
            } else if val < 0.5 {
                2u8
            } else {
                3u8
            };

            let bit_offset = i * 2;
            let byte_idx = bit_offset / 8;
            let bit_shift = bit_offset % 8;
            code[byte_idx] |= level << bit_shift;
        }

        code
    }

    /// 4-bit quantization: 16 levels.
    ///
    /// Uniform quantization from -2.0 to 2.0 (clipped).
    fn quantize_4bit(&self, rotated: &[f32]) -> Vec<u8> {
        let num_bytes = (self.dim * 4 + 7) / 8;
        let mut code = vec![0u8; num_bytes];

        for (i, &val) in rotated.iter().enumerate() {
            // Clamp to [-2.0, 2.0] and map to [0, 15]
            let clamped = val.clamp(-2.0, 2.0);
            let level = ((clamped + 2.0) * 3.75) as u8; // (val+2) / 4 * 15

            let bit_offset = i * 4;
            let byte_idx = bit_offset / 8;
            let bit_shift = bit_offset % 8;

            // Handle nibble alignment
            if bit_shift == 0 {
                code[byte_idx] |= level;
            } else {
                // Nibble spans two bytes (only for odd dimensions)
                code[byte_idx] |= level << 4;
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

    /// Check if rotation matrix is orthonormal (for testing).
    #[cfg(test)]
    fn is_orthonormal(&self, tolerance: f32) -> bool {
        // Check R * R^T = I
        for i in 0..self.dim {
            for j in 0..self.dim {
                // Compute (R * R^T)[i][j] = sum_k R[i][k] * R[j][k]
                let mut dot = 0.0f32;
                for k in 0..self.dim {
                    dot += self.rotation[i * self.dim + k] * self.rotation[j * self.dim + k];
                }

                let expected = if i == j { 1.0 } else { 0.0 };
                if (dot - expected).abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotation_matrix_is_orthonormal() {
        let encoder = RaBitQ::new(128, 1, 42);
        assert!(
            encoder.is_orthonormal(1e-5),
            "Rotation matrix should be orthonormal"
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
}
