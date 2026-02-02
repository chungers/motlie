//! Error handling for the vector module.
//!
//! Uses `anyhow::Result` for consistency with the rest of motlie_db.

use anyhow::bail;

/// Convenience re-export of anyhow::Result
pub use anyhow::Result;

/// Create a no embedder error
pub fn no_embedder(model: &str) -> anyhow::Error {
    anyhow::anyhow!("No embedder configured for model: {}", model)
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
