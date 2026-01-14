//! Caching infrastructure for HNSW index operations.
//!
//! This module provides in-memory caches that optimize search performance:
//!
//! - `NavigationCache`: Caches HNSW navigation metadata and edge lists
//! - `BinaryCodeCache`: Caches RaBitQ binary codes for Hamming distance
//!
//! # Performance Impact
//!
//! | Cache | Purpose | Impact |
//! |-------|---------|--------|
//! | NavigationCache | Edge list caching | 2-3x search speedup |
//! | BinaryCodeCache | Binary code caching | Enables RaBitQ speedup |

mod binary_codes;
mod navigation;

pub use binary_codes::{BinaryCodeCache, BinaryCodeEntry};
pub use navigation::{NavigationCache, NavigationCacheConfig, NavigationLayerInfo};
