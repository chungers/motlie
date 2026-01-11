//! HNSW algorithm configuration.
//!
//! References:
//! - HNSW paper: https://arxiv.org/abs/1603.09320
//! - motlie design: libs/db/src/vector/ROADMAP.md

use serde::{Deserialize, Serialize};

/// HNSW algorithm parameters.
///
/// HNSW (Hierarchical Navigable Small World) is a graph-based approximate
/// nearest neighbor search algorithm. These parameters control the trade-off
/// between index quality, memory usage, and search speed.
///
/// # Parameter Guidelines
///
/// | Scale | M | ef_construction | ef_search | Memory/Vector |
/// |-------|---|-----------------|-----------|---------------|
/// | 10K   | 16| 100             | 50        | ~0.5KB        |
/// | 100K  | 16| 200             | 100       | ~0.5KB        |
/// | 1M    | 16-32| 200-400      | 100-200   | ~0.5-1KB      |
/// | 1B    | 16| 200             | 100       | ~0.5KB        |
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    /// Vector dimensionality (must match embedding model).
    /// Common values: 128 (SIFT), 768 (BERT), 1536 (OpenAI ada-002)
    pub dim: usize,

    /// Number of bidirectional links per node at layer 0.
    /// Higher M = better recall, more memory, slower insert.
    /// Recommended: 16-64, Default: 16
    pub m: usize,

    /// Maximum links per node at layers > 0 (typically 2*M).
    /// Default: 2 * m
    pub m_max: usize,

    /// Maximum links per node at layer 0 (typically M or 2*M).
    /// Layer 0 is denser, so higher limit improves recall.
    /// Default: 2 * m
    pub m_max_0: usize,

    /// Search beam width during index construction.
    /// Higher ef_construction = better graph quality, slower build.
    /// Recommended: 100-500, Default: 200
    pub ef_construction: usize,

    /// Probability multiplier for layer assignment.
    /// P(layer = L) = exp(-L * m_l)
    /// Default: 1.0 / ln(m)
    pub m_l: f32,

    /// Maximum search layer (auto-determined based on N).
    /// Default: None (auto-calculate as floor(ln(N) * m_l))
    pub max_level: Option<u8>,

    /// Minimum neighbor set size to trigger batch distance computation.
    ///
    /// When fetching neighbors during search, if the neighbor count exceeds
    /// this threshold, vectors are fetched via MultiGet instead of individual
    /// lookups. Batch operations have overhead (Vec allocation, MultiGet
    /// coordination) that only pays off for large batches.
    ///
    /// **Tuning guidance:**
    /// - `64` (default): Effectively disables batching. Best for local NVMe/SSD
    ///   storage with small M values (M ≤ 32). Individual lookups are faster.
    /// - `4-8`: Enables batching. May help with high-latency storage (network,
    ///   remote DB) or very large M values (M ≥ 64).
    ///
    /// **Background:** Phase 3 benchmarks showed batching hurts performance
    /// at 100K scale with M=16 on local storage due to overhead exceeding
    /// the I/O savings. The code is retained for future use with different
    /// storage backends or configurations.
    pub batch_threshold: usize,
}

impl Default for Config {
    fn default() -> Self {
        let m = 16;
        Self {
            dim: 128,
            m,
            m_max: 2 * m,
            m_max_0: 2 * m,
            ef_construction: 200,
            m_l: 1.0 / (m as f32).ln(),
            max_level: None,
            batch_threshold: 64, // Effectively disables batching for local storage
        }
    }
}

/// Configuration warnings for suboptimal HNSW settings.
///
/// These are warnings, not errors. Advanced users may intentionally
/// use non-standard settings for specific use cases.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigWarning {
    /// M < 8 may cause poor recall at scale
    LowM(usize),
    /// M > 64 has diminishing returns and high memory cost
    HighM(usize),
    /// ef_construction should be >= 2*M for good graph quality
    LowEfConstruction {
        ef_construction: usize,
        recommended: usize,
    },
    /// ef_construction > 500 has diminishing returns
    HighEfConstruction(usize),
    /// Dimension is unusually low (< 32)
    LowDimension(usize),
    /// Dimension is unusually high (> 4096)
    HighDimension(usize),
}

impl std::fmt::Display for ConfigWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigWarning::LowM(m) => {
                write!(
                    f,
                    "M={} is low, may cause poor recall at scale (recommend >= 8)",
                    m
                )
            }
            ConfigWarning::HighM(m) => {
                write!(
                    f,
                    "M={} is high, diminishing returns and high memory (recommend <= 64)",
                    m
                )
            }
            ConfigWarning::LowEfConstruction {
                ef_construction,
                recommended,
            } => {
                write!(
                    f,
                    "ef_construction={} is low for M, recommend >= {} (2*M)",
                    ef_construction, recommended
                )
            }
            ConfigWarning::HighEfConstruction(ef) => {
                write!(
                    f,
                    "ef_construction={} is high, diminishing returns (recommend <= 500)",
                    ef
                )
            }
            ConfigWarning::LowDimension(dim) => {
                write!(f, "dim={} is unusually low (common: 128-1536)", dim)
            }
            ConfigWarning::HighDimension(dim) => {
                write!(f, "dim={} is unusually high (common: 128-1536)", dim)
            }
        }
    }
}

impl Config {
    /// Create config for specific dimension with default parameters.
    pub fn for_dim(dim: usize) -> Self {
        Self {
            dim,
            ..Default::default()
        }
    }

    /// Validate configuration and return warnings for suboptimal settings.
    ///
    /// Returns an empty vector if all settings are within recommended ranges.
    /// Warnings are informational - the config will still work.
    ///
    /// # Example
    ///
    /// ```
    /// use motlie_db::vector::hnsw::Config;
    ///
    /// let config = Config { m: 4, ..Default::default() };
    /// let warnings = config.validate();
    /// assert!(!warnings.is_empty()); // Warning: M is low
    /// ```
    pub fn validate(&self) -> Vec<ConfigWarning> {
        let mut warnings = Vec::new();

        // Check M bounds
        if self.m < 8 {
            warnings.push(ConfigWarning::LowM(self.m));
        }
        if self.m > 64 {
            warnings.push(ConfigWarning::HighM(self.m));
        }

        // Check ef_construction relative to M
        let recommended_ef = self.m * 2;
        if self.ef_construction < recommended_ef {
            warnings.push(ConfigWarning::LowEfConstruction {
                ef_construction: self.ef_construction,
                recommended: recommended_ef,
            });
        }
        if self.ef_construction > 500 {
            warnings.push(ConfigWarning::HighEfConstruction(self.ef_construction));
        }

        // Check dimension bounds (informational)
        if self.dim < 32 {
            warnings.push(ConfigWarning::LowDimension(self.dim));
        }
        if self.dim > 4096 {
            warnings.push(ConfigWarning::HighDimension(self.dim));
        }

        warnings
    }

    /// Check if this configuration is valid (no critical issues).
    ///
    /// Returns true even with warnings - only returns false for
    /// invalid configurations that would cause errors.
    pub fn is_valid(&self) -> bool {
        self.dim > 0 && self.m > 0 && self.ef_construction > 0
    }

    /// High-recall configuration (slower, more memory).
    /// Best for applications where recall is critical.
    pub fn high_recall(dim: usize) -> Self {
        Self {
            dim,
            m: 32,
            m_max: 64,
            m_max_0: 64,
            ef_construction: 400,
            m_l: 1.0 / 32f32.ln(),
            max_level: None,
            batch_threshold: 64, // May benefit from lower threshold with M=32
        }
    }

    /// Memory-optimized configuration (faster, less recall).
    /// Best for resource-constrained environments.
    pub fn compact(dim: usize) -> Self {
        Self {
            dim,
            m: 8,
            m_max: 16,
            m_max_0: 16,
            ef_construction: 100,
            m_l: 1.0 / 8f32.ln(),
            max_level: None,
            batch_threshold: 64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_defaults() {
        let config = Config::default();
        assert_eq!(config.dim, 128);
        assert_eq!(config.m, 16);
        assert_eq!(config.m_max, 32);
        assert_eq!(config.m_max_0, 32);
        assert_eq!(config.ef_construction, 200);
    }

    #[test]
    fn test_hnsw_for_dim() {
        let config = Config::for_dim(768);
        assert_eq!(config.dim, 768);
        assert_eq!(config.m, 16); // Other params stay default
    }

    #[test]
    fn test_hnsw_high_recall() {
        let config = Config::high_recall(768);
        assert_eq!(config.dim, 768);
        assert_eq!(config.m, 32);
        assert_eq!(config.ef_construction, 400);
    }

    #[test]
    fn test_hnsw_compact() {
        let config = Config::compact(768);
        assert_eq!(config.dim, 768);
        assert_eq!(config.m, 8);
        assert_eq!(config.ef_construction, 100);
    }

    // =========================================================================
    // Validation Tests
    // =========================================================================

    #[test]
    fn test_hnsw_validate_default_no_warnings() {
        let config = Config::default();
        let warnings = config.validate();
        assert!(
            warnings.is_empty(),
            "Default config should have no warnings"
        );
        assert!(config.is_valid());
    }

    #[test]
    fn test_hnsw_validate_low_m() {
        let config = Config {
            m: 4,
            ..Default::default()
        };
        let warnings = config.validate();
        assert!(warnings.iter().any(|w| matches!(w, ConfigWarning::LowM(4))));
    }

    #[test]
    fn test_hnsw_validate_high_m() {
        let config = Config {
            m: 128,
            ..Default::default()
        };
        let warnings = config.validate();
        assert!(warnings
            .iter()
            .any(|w| matches!(w, ConfigWarning::HighM(128))));
    }

    #[test]
    fn test_hnsw_validate_low_ef_construction() {
        let config = Config {
            m: 16,
            ef_construction: 16, // Should be >= 32 (2*M)
            ..Default::default()
        };
        let warnings = config.validate();
        assert!(warnings.iter().any(|w| matches!(
            w,
            ConfigWarning::LowEfConstruction {
                ef_construction: 16,
                recommended: 32
            }
        )));
    }

    #[test]
    fn test_hnsw_validate_high_ef_construction() {
        let config = Config {
            ef_construction: 1000,
            ..Default::default()
        };
        let warnings = config.validate();
        assert!(warnings
            .iter()
            .any(|w| matches!(w, ConfigWarning::HighEfConstruction(1000))));
    }

    #[test]
    fn test_hnsw_validate_low_dimension() {
        let config = Config {
            dim: 16,
            ..Default::default()
        };
        let warnings = config.validate();
        assert!(warnings
            .iter()
            .any(|w| matches!(w, ConfigWarning::LowDimension(16))));
    }

    #[test]
    fn test_hnsw_validate_high_dimension() {
        let config = Config {
            dim: 8192,
            ..Default::default()
        };
        let warnings = config.validate();
        assert!(warnings
            .iter()
            .any(|w| matches!(w, ConfigWarning::HighDimension(8192))));
    }

    #[test]
    fn test_config_warning_display() {
        let warning = ConfigWarning::LowM(4);
        let display = format!("{}", warning);
        assert!(display.contains("M=4"));
        assert!(display.contains("low"));
    }
}
