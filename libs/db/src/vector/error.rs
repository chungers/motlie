//! Error handling for the vector module.
//!
//! Uses `anyhow::Result` for consistency with the rest of motlie_db.

use anyhow::{anyhow, bail};

/// Convenience re-export of anyhow::Result
pub use anyhow::Result;

/// Create a dimension mismatch error
pub fn dimension_mismatch(expected: u32, got: usize) -> anyhow::Error {
    anyhow!(
        "Dimension mismatch: expected {}, got {}",
        expected,
        got as u32
    )
}

/// Create an unknown embedding error
pub fn unknown_embedding(code: u64) -> anyhow::Error {
    anyhow!("Unknown embedding: code {}", code)
}

/// Create a no embedder error
pub fn no_embedder(model: &str) -> anyhow::Error {
    anyhow!("No embedder configured for model: {}", model)
}

/// Create an embedding not registered error
pub fn embedding_not_registered(model: &str, dim: u32, distance: &str) -> anyhow::Error {
    anyhow!(
        "Embedding not registered: {} (dim={}, distance={})",
        model,
        dim,
        distance
    )
}

/// Bail if vector dimension doesn't match expected
#[inline]
pub fn check_dimension(expected: u32, vector: &[f32]) -> Result<()> {
    if vector.len() != expected as usize {
        bail!(
            "Dimension mismatch: expected {}, got {}",
            expected,
            vector.len()
        );
    }
    Ok(())
}
