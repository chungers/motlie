//! Distance metrics for vector similarity.
//!
//! Design rationale (ARCH-17):
//! - Distance metric is fixed per embedding space, not per-search
//! - HNSW graph structure is optimized for the metric used during construction
//! - Searching with a different metric would follow wrong paths

use serde::{Deserialize, Serialize};

/// Distance metric for vector similarity computation.
///
/// The metric is fixed when creating an embedding space and cannot be changed
/// per-search, because the HNSW graph structure is optimized for the specific
/// metric used during construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Distance {
    /// Cosine distance: 1 - cos(a, b)
    /// Range: [0, 2], lower is more similar
    /// Best for: normalized embeddings, semantic similarity
    Cosine,

    /// Euclidean (L2) distance: ||a - b||
    /// Range: [0, ∞), lower is more similar
    /// Best for: spatial data, image features
    L2,

    /// Dot product: -a · b (negated for min-heap compatibility)
    /// Range: (-∞, ∞), lower (more negative) is more similar
    /// Best for: unnormalized embeddings where magnitude matters
    DotProduct,
}

impl Distance {
    /// Compute distance between two vectors using this metric.
    ///
    /// Uses SIMD-optimized implementations from `motlie_core::distance`.
    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Distance::Cosine => motlie_core::distance::cosine(a, b),
            Distance::L2 => motlie_core::distance::euclidean(a, b),
            // Negate dot product so lower = more similar (consistent with other metrics)
            Distance::DotProduct => -motlie_core::distance::dot(a, b),
        }
    }

    /// Whether lower distance values mean more similar vectors.
    ///
    /// - `true` for Cosine and L2 (lower = more similar)
    /// - `false` for DotProduct (higher dot product = more similar,
    ///   but we negate it so lower = more similar in our implementation)
    pub fn is_lower_better(&self) -> bool {
        // All metrics are normalized so lower = more similar
        // DotProduct is negated in compute() for consistency
        true
    }

    /// String representation of the distance metric.
    pub fn as_str(&self) -> &'static str {
        match self {
            Distance::Cosine => "cosine",
            Distance::L2 => "l2",
            Distance::DotProduct => "dot",
        }
    }
}

impl std::fmt::Display for Distance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for Distance {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cosine" | "cos" => Ok(Distance::Cosine),
            "l2" | "euclidean" | "euclid" => Ok(Distance::L2),
            "dot" | "dotproduct" | "dot_product" | "inner" | "ip" => Ok(Distance::DotProduct),
            _ => Err(format!("Unknown distance metric: {}", s)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_as_str() {
        assert_eq!(Distance::Cosine.as_str(), "cosine");
        assert_eq!(Distance::L2.as_str(), "l2");
        assert_eq!(Distance::DotProduct.as_str(), "dot");
    }

    #[test]
    fn test_distance_from_str() {
        assert_eq!("cosine".parse::<Distance>().unwrap(), Distance::Cosine);
        assert_eq!("l2".parse::<Distance>().unwrap(), Distance::L2);
        assert_eq!("dot".parse::<Distance>().unwrap(), Distance::DotProduct);
        assert_eq!(
            "dotproduct".parse::<Distance>().unwrap(),
            Distance::DotProduct
        );
        assert!("invalid".parse::<Distance>().is_err());
    }

    #[test]
    fn test_is_lower_better() {
        assert!(Distance::Cosine.is_lower_better());
        assert!(Distance::L2.is_lower_better());
        assert!(Distance::DotProduct.is_lower_better());
    }
}
