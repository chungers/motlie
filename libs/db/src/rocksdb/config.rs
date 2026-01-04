//! Configuration types for RocksDB storage.
//!
//! Provides common configuration shared across all storage subsystems.

// ============================================================================
// BlockCacheConfig
// ============================================================================

/// Configuration for RocksDB block cache.
///
/// RocksDB's block cache stores uncompressed data blocks (~4KB each).
/// A shared cache across all column families allows RocksDB to dynamically
/// allocate memory based on access patterns.
///
/// See [RocksDB Block Cache Wiki](https://github.com/facebook/rocksdb/wiki/Block-Cache)
/// for detailed documentation.
#[derive(Debug, Clone)]
pub struct BlockCacheConfig {
    /// Total block cache size in bytes.
    /// Default: 256MB. Production deployments should tune to ~1/3 of available memory.
    pub cache_size_bytes: usize,

    /// Default block size for column families.
    /// Default: 4KB. Optimal for fixed-size keys (~100 keys per block).
    pub default_block_size: usize,

    /// Block size for large data (fragments, vectors).
    /// Default: 16KB. Better for variable-length or sequential access patterns.
    pub large_block_size: usize,

    /// Whether to cache index and filter blocks in the block cache.
    /// Default: true. Keeps hot metadata in cache for faster lookups.
    pub cache_index_and_filter_blocks: bool,

    /// Whether to pin L0 filter and index blocks in cache.
    /// Default: true. Prevents eviction of newest (most likely accessed) data.
    pub pin_l0_filter_and_index: bool,
}

impl Default for BlockCacheConfig {
    fn default() -> Self {
        Self {
            cache_size_bytes: 256 * 1024 * 1024,       // 256MB
            default_block_size: 4 * 1024,               // 4KB
            large_block_size: 16 * 1024,                // 16KB
            cache_index_and_filter_blocks: true,
            pin_l0_filter_and_index: true,
        }
    }
}

impl BlockCacheConfig {
    /// Create config with specified cache size, using defaults for other settings.
    pub fn with_cache_size(cache_size_bytes: usize) -> Self {
        Self {
            cache_size_bytes,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_cache_config_default() {
        let config = BlockCacheConfig::default();
        assert_eq!(config.cache_size_bytes, 256 * 1024 * 1024);
        assert_eq!(config.default_block_size, 4 * 1024);
        assert_eq!(config.large_block_size, 16 * 1024);
        assert!(config.cache_index_and_filter_blocks);
        assert!(config.pin_l0_filter_and_index);
    }

    #[test]
    fn test_block_cache_config_with_cache_size() {
        let config = BlockCacheConfig::with_cache_size(512 * 1024 * 1024);
        assert_eq!(config.cache_size_bytes, 512 * 1024 * 1024);
        // Other defaults preserved
        assert_eq!(config.default_block_size, 4 * 1024);
    }
}
