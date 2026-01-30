//! HNSW configuration warnings.
//!
//! The structural HNSW parameters (M, ef_construction, m_l) are now derived from
//! the persisted EmbeddingSpec, eliminating configuration drift. This module
//! provides warnings for suboptimal parameter values.
//!
//! References:
//! - HNSW paper: https://arxiv.org/abs/1603.09320
//! - motlie design: libs/db/src/vector/docs/CONFIG.md

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_warning_display() {
        let warning = ConfigWarning::LowM(4);
        let display = format!("{}", warning);
        assert!(display.contains("M=4"));
        assert!(display.contains("low"));
    }

    #[test]
    fn test_config_warning_low_ef_construction() {
        let warning = ConfigWarning::LowEfConstruction {
            ef_construction: 16,
            recommended: 32,
        };
        let display = format!("{}", warning);
        assert!(display.contains("16"));
        assert!(display.contains("32"));
    }
}
