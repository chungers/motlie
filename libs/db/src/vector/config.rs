//! Configuration types for vector storage and indexing.
//!
//! ## Design Note
//!
//! HNSW structural parameters (m, ef_construction, etc.) should come from
//! `EmbeddingSpec` which is persisted and protected by SpecHash. The hnsw
//! field in `VectorConfig` is deprecated - use `EmbeddingSpec` as the single
//! source of truth for HNSW configuration.
//!
//! References:
//! - RaBitQ paper: https://arxiv.org/abs/2405.12497
//! - motlie design: libs/db/src/vector/ROADMAP.md

use serde::{Deserialize, Serialize};

// Re-export Config from hnsw module for VectorConfig (deprecated)
#[allow(deprecated)]
pub(crate) use crate::vector::hnsw::Config;

/// RaBitQ binary quantization parameters.
///
/// RaBitQ is a training-free binary quantization method that uses random
/// rotation to preserve distance relationships. Unlike Product Quantization,
/// it requires no training data (DATA-1 compliant).
///
/// # Bits per Dimension Trade-offs
///
/// | Bits | Bytes (128D) | Recall (no rerank) | Compression |
/// |------|--------------|---------------------|-------------|
/// | 1    | 16 bytes     | ~70%                | 32x         |
/// | 2    | 32 bytes     | ~85%                | 16x         |
/// | 4    | 64 bytes     | ~92%                | 8x          |
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RaBitQConfig {
    /// Bits per dimension (1, 2, 4, or 8).
    /// 1-bit: 32x compression, ~70% recall without rerank
    /// 2-bit: 16x compression, ~85% recall without rerank
    /// 4-bit: 8x compression, ~92% recall without rerank
    pub bits_per_dim: u8,

    /// Seed for rotation matrix generation (deterministic).
    /// Same seed produces same rotation matrix.
    pub rotation_seed: u64,

    /// Enable RaBitQ (can be disabled for full-precision search).
    pub enabled: bool,

    /// Use SIMD-optimized dot products from motlie_core::distance::quantized.
    /// When true (default), uses AVX2/NEON accelerated implementations.
    /// When false, uses local scalar implementation for debugging/comparison.
    #[serde(default = "default_use_simd")]
    pub use_simd_dot: bool,
}

fn default_use_simd() -> bool {
    true
}

impl Default for RaBitQConfig {
    fn default() -> Self {
        Self {
            bits_per_dim: 1,
            rotation_seed: 42,
            enabled: true,
            use_simd_dot: true,
        }
    }
}

/// Configuration warning for RaBitQ settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RaBitQConfigWarning {
    /// bits_per_dim must be 1, 2, 4, or 8
    InvalidBitsPerDim(u8),
}

impl std::fmt::Display for RaBitQConfigWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RaBitQConfigWarning::InvalidBitsPerDim(bits) => {
                write!(
                    f,
                    "bits_per_dim={} is invalid (must be 1, 2, 4, or 8)",
                    bits
                )
            }
        }
    }
}

impl RaBitQConfig {
    /// Calculate the code size in bytes for a given dimension.
    pub fn code_size(&self, dim: usize) -> usize {
        (dim * self.bits_per_dim as usize + 7) / 8
    }

    /// Validate configuration and return warnings for invalid settings.
    ///
    /// # Example
    ///
    /// ```
    /// use motlie_db::vector::RaBitQConfig;
    ///
    /// let config = RaBitQConfig { bits_per_dim: 3, ..Default::default() };
    /// let warnings = config.validate();
    /// assert!(!warnings.is_empty()); // Warning: invalid bits_per_dim
    /// ```
    pub fn validate(&self) -> Vec<RaBitQConfigWarning> {
        let mut warnings = Vec::new();

        // bits_per_dim must be 1, 2, 4, or 8
        if ![1, 2, 4, 8].contains(&self.bits_per_dim) {
            warnings.push(RaBitQConfigWarning::InvalidBitsPerDim(self.bits_per_dim));
        }

        warnings
    }

    /// Check if this configuration is valid (no critical issues).
    pub fn is_valid(&self) -> bool {
        [1, 2, 4, 8].contains(&self.bits_per_dim)
    }
}

/// Complete vector storage configuration.
///
/// ## Deprecation Note
///
/// The `hnsw` field is deprecated. HNSW structural parameters should be set
/// via `EmbeddingSpec` (the persisted source of truth). Only `rabitq` config
/// is still relevant here for runtime compression settings.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VectorConfig {
    /// DEPRECATED: Use EmbeddingSpec for HNSW parameters instead.
    /// This field is kept for backward compatibility but will be removed.
    #[deprecated(
        since = "0.2.0",
        note = "Use EmbeddingSpec for HNSW parameters - this is the persisted source of truth"
    )]
    #[allow(deprecated)]
    pub hnsw: Config,
    /// RaBitQ compression configuration.
    pub rabitq: RaBitQConfig,
}

#[allow(deprecated)]
impl VectorConfig {
    /// Configuration for 128-dimensional embeddings (e.g., SIFT).
    #[deprecated(
        since = "0.2.0",
        note = "Use EmbeddingSpec for HNSW parameters instead"
    )]
    pub fn dim_128() -> Self {
        Self {
            hnsw: Config::for_dim(128),
            ..Default::default()
        }
    }

    /// Configuration for 768-dimensional embeddings (e.g., BERT, Gemma).
    #[deprecated(
        since = "0.2.0",
        note = "Use EmbeddingSpec for HNSW parameters instead"
    )]
    pub fn dim_768() -> Self {
        Self {
            hnsw: Config::for_dim(768),
            ..Default::default()
        }
    }

    /// Configuration for 1024-dimensional embeddings (e.g., Qwen3).
    #[deprecated(
        since = "0.2.0",
        note = "Use EmbeddingSpec for HNSW parameters instead"
    )]
    pub fn dim_1024() -> Self {
        Self {
            hnsw: Config::for_dim(1024),
            ..Default::default()
        }
    }

    /// Configuration for 1536-dimensional embeddings (e.g., OpenAI ada-002).
    #[deprecated(
        since = "0.2.0",
        note = "Use EmbeddingSpec for HNSW parameters instead"
    )]
    pub fn dim_1536() -> Self {
        Self {
            hnsw: Config::for_dim(1536),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rabitq_code_size() {
        let config = RaBitQConfig::default();
        // 128 dims * 1 bit = 128 bits = 16 bytes
        assert_eq!(config.code_size(128), 16);
        // 768 dims * 1 bit = 768 bits = 96 bytes
        assert_eq!(config.code_size(768), 96);

        let config_2bit = RaBitQConfig {
            bits_per_dim: 2,
            ..Default::default()
        };
        // 128 dims * 2 bits = 256 bits = 32 bytes
        assert_eq!(config_2bit.code_size(128), 32);
    }

    #[test]
    #[allow(deprecated)]
    fn test_vector_config_presets() {
        let c128 = VectorConfig::dim_128();
        assert_eq!(c128.hnsw.dim, 128);

        let c768 = VectorConfig::dim_768();
        assert_eq!(c768.hnsw.dim, 768);

        let c1536 = VectorConfig::dim_1536();
        assert_eq!(c1536.hnsw.dim, 1536);
    }

    #[test]
    fn test_rabitq_validate_default_no_warnings() {
        let config = RaBitQConfig::default();
        let warnings = config.validate();
        assert!(
            warnings.is_empty(),
            "Default config should have no warnings"
        );
        assert!(config.is_valid());
    }

    #[test]
    fn test_rabitq_validate_invalid_bits() {
        let config = RaBitQConfig {
            bits_per_dim: 3, // Invalid: not 1, 2, 4, or 8
            ..Default::default()
        };
        let warnings = config.validate();
        assert!(warnings
            .iter()
            .any(|w| matches!(w, RaBitQConfigWarning::InvalidBitsPerDim(3))));
        assert!(!config.is_valid());
    }

    #[test]
    fn test_rabitq_validate_valid_bits() {
        for bits in [1, 2, 4, 8] {
            let config = RaBitQConfig {
                bits_per_dim: bits,
                ..Default::default()
            };
            assert!(config.is_valid(), "bits_per_dim={} should be valid", bits);
            assert!(config.validate().is_empty());
        }
    }

    #[test]
    fn test_rabitq_config_warning_display() {
        let warning = RaBitQConfigWarning::InvalidBitsPerDim(3);
        let display = format!("{}", warning);
        assert!(display.contains("bits_per_dim=3"));
        assert!(display.contains("invalid"));
    }
}
