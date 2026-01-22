//! Vector quantization methods for compression and fast search.
//!
//! This module provides quantization implementations:
//!
//! - `RaBitQ`: Training-free binary quantization using random orthonormal rotation
//!
//! # Quantization Overview
//!
//! | Method | Compression | Recall | Training |
//! |--------|-------------|--------|----------|
//! | RaBitQ 1-bit | 32x | ~70% | None |
//! | RaBitQ 2-bit | 16x | ~85% | None |
//! | RaBitQ 4-bit | 8x | ~92% | None |

mod rabitq;

pub use rabitq::RaBitQ;
