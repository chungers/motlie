//! Scalar (non-SIMD) distance implementations
//!
//! These are portable fallback implementations that work on any platform.
//! They may still benefit from auto-vectorization by the compiler.

/// Compute squared Euclidean distance
#[inline]
pub fn euclidean_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Compute cosine distance: 1 - cosine_similarity
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let norm_a = norm_a.sqrt();
    let norm_b = norm_b.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0; // Maximum distance for zero vectors
    }

    1.0 - (dot / (norm_a * norm_b))
}

/// Compute dot product
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_squared() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        // (1-4)^2 + (2-5)^2 + (3-6)^2 = 9 + 9 + 9 = 27
        assert!((euclidean_squared(&a, &b) - 27.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_parallel() {
        let a = vec![1.0, 0.0];
        let b = vec![2.0, 0.0];
        assert!(cosine(&a, &b).abs() < 1e-5);
    }

    #[test]
    fn test_dot() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        assert!((dot(&a, &b) - 11.0).abs() < 1e-5);
    }
}
