//! Configuration types for vector storage and indexing.
//!
//! References:
//! - RaBitQ paper: https://arxiv.org/abs/2405.12497
//! - HNSW config: see `crate::vector::hnsw::Config`
//! - motlie design: libs/db/src/vector/ROADMAP.md

use serde::{Deserialize, Serialize};

// Re-export HnswConfig from hnsw module for VectorConfig
pub use crate::vector::hnsw::Config as HnswConfig;

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
}

impl Default for RaBitQConfig {
    fn default() -> Self {
        Self {
            bits_per_dim: 1,
            rotation_seed: 42,
            enabled: true,
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

/// Async graph updater configuration.
///
/// Controls background graph refinement for online updates.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AsyncUpdaterConfig {
    /// Enable async graph updates.
    pub enabled: bool,

    /// Maximum pending vectors before blocking inserts.
    pub max_pending: usize,

    /// Batch size for graph updates.
    pub batch_size: usize,

    /// Interval between update batches in milliseconds.
    pub interval_ms: u64,
}

impl Default for AsyncUpdaterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_pending: 10_000,
            batch_size: 100,
            interval_ms: 100,
        }
    }
}

/// Navigation cache configuration.
///
/// Caches top HNSW layers in memory for faster search.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NavigationCacheConfig {
    /// Enable navigation cache.
    pub enabled: bool,

    /// Maximum layers to cache (from top).
    pub max_cached_layers: u8,

    /// Maximum nodes per layer to cache.
    pub max_nodes_per_layer: usize,
}

impl Default for NavigationCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_cached_layers: 3,
            max_nodes_per_layer: 10_000,
        }
    }
}

/// Complete vector storage configuration.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VectorConfig {
    pub hnsw: HnswConfig,
    pub rabitq: RaBitQConfig,
    pub async_updater: AsyncUpdaterConfig,
    pub navigation_cache: NavigationCacheConfig,
}

impl VectorConfig {
    /// Configuration for 128-dimensional embeddings (e.g., SIFT).
    pub fn dim_128() -> Self {
        Self {
            hnsw: HnswConfig::for_dim(128),
            ..Default::default()
        }
    }

    /// Configuration for 768-dimensional embeddings (e.g., BERT, Gemma).
    pub fn dim_768() -> Self {
        Self {
            hnsw: HnswConfig::for_dim(768),
            ..Default::default()
        }
    }

    /// Configuration for 1024-dimensional embeddings (e.g., Qwen3).
    pub fn dim_1024() -> Self {
        Self {
            hnsw: HnswConfig::for_dim(1024),
            ..Default::default()
        }
    }

    /// Configuration for 1536-dimensional embeddings (e.g., OpenAI ada-002).
    pub fn dim_1536() -> Self {
        Self {
            hnsw: HnswConfig::for_dim(1536),
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
