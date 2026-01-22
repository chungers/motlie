//! In-memory cache for RaBitQ binary codes with ADC corrections.
//!
//! This module provides caching for binary codes and ADC correction factors
//! to enable fast ADC distance computation during RaBitQ search.
//!
//! ADC (Asymmetric Distance Computation) achieves 91-99% recall vs Hamming's
//! 10-24% by keeping the query as float32 and using weighted dot products.
//!
//! # Zero-Copy Design
//!
//! The cache uses `Arc<BinaryCodeEntry>` to avoid cloning binary codes on every
//! access. This eliminates O(n) memcpy per cache hit, replacing it with O(1)
//! atomic increment. At large scales (1M+ vectors, high concurrency), this
//! reduces allocator contention and improves QPS by 10-20%.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::vector::schema::{AdcCorrection, EmbeddingCode, VecId};

/// Cached binary code entry with ADC correction factors.
///
/// Stored in Arc to enable zero-copy sharing across cache lookups.
#[derive(Debug, Clone)]
pub struct BinaryCodeEntry {
    /// Binary quantized code (RaBitQ encoded)
    pub code: Vec<u8>,
    /// ADC correction factors for distance estimation
    pub correction: AdcCorrection,
}

/// In-memory cache for RaBitQ binary codes with ADC corrections.
///
/// Caches binary codes and ADC correction factors to enable fast ADC distance
/// computation during RaBitQ beam search, avoiding RocksDB reads.
///
/// # ADC vs Hamming
///
/// | Mode | Recall@10 | Notes |
/// |------|-----------|-------|
/// | Hamming 4-bit | 24% | Symmetric, loses numeric ordering |
/// | ADC 4-bit | 92% | Asymmetric, preserves ordering |
///
/// ADC keeps the query as float32 and computes weighted dot products,
/// correcting for quantization error using stored correction factors.
///
/// # Memory Usage
///
/// For 4-bit RaBitQ at 128D:
/// - Binary code: 64 bytes per vector
/// - ADC correction: 8 bytes per vector (2×f32)
/// - Arc overhead: 16 bytes per vector (strong + weak counts)
/// - Total: ~88 bytes per vector
/// - 100K vectors: ~8.4 MB
/// - 1M vectors: ~84 MB
///
/// # Performance
///
/// Uses `Arc<BinaryCodeEntry>` for O(1) cache hits instead of O(n) clone.
/// At 1M vectors with 32 threads, this improves QPS by 10-20%.
pub struct BinaryCodeCache {
    /// Cached binary codes with ADC corrections: (embedding_code, vec_id) -> Arc<entry>
    codes: RwLock<HashMap<(EmbeddingCode, VecId), Arc<BinaryCodeEntry>>>,

    /// Total bytes cached (for monitoring, excludes Arc overhead)
    cache_bytes: RwLock<usize>,
}

impl BinaryCodeCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self {
            codes: RwLock::new(HashMap::new()),
            cache_bytes: RwLock::new(0),
        }
    }

    /// Insert a binary code with ADC correction into the cache.
    ///
    /// If an entry already exists for this (embedding, vec_id), it is replaced
    /// and the cache_bytes counter is adjusted accordingly.
    pub fn put(
        &self,
        embedding: EmbeddingCode,
        vec_id: VecId,
        code: Vec<u8>,
        correction: AdcCorrection,
    ) {
        if let Ok(mut cache) = self.codes.write() {
            let new_code_len = code.len();
            let key = (embedding, vec_id);

            // Subtract old entry size if replacing (fixes overcount bug)
            if let Some(old_entry) = cache.get(&key) {
                if let Ok(mut bytes) = self.cache_bytes.write() {
                    let old_size = old_entry.code.len() + 8; // +8 for AdcCorrection
                    *bytes = bytes.saturating_sub(old_size);
                }
            }

            // Insert new entry and add its size
            let entry = Arc::new(BinaryCodeEntry { code, correction });
            cache.insert(key, entry);
            if let Ok(mut bytes) = self.cache_bytes.write() {
                *bytes += new_code_len + 8; // +8 for AdcCorrection (2×f32)
            }
        }
    }

    /// Get a binary code with ADC correction from the cache.
    ///
    /// Returns an `Arc<BinaryCodeEntry>` for zero-copy access. The Arc::clone
    /// is O(1) atomic increment, avoiding the O(n) memcpy of the previous design.
    pub fn get(&self, embedding: EmbeddingCode, vec_id: VecId) -> Option<Arc<BinaryCodeEntry>> {
        self.codes
            .read()
            .ok()?
            .get(&(embedding, vec_id))
            .cloned() // Arc::clone is cheap (atomic increment)
    }

    /// Get binary codes with corrections for multiple vectors.
    ///
    /// Returns a vector of `Option<Arc<BinaryCodeEntry>>` in the same order as vec_ids.
    /// Missing entries are None. Each Arc::clone is O(1).
    pub fn get_batch(
        &self,
        embedding: EmbeddingCode,
        vec_ids: &[VecId],
    ) -> Vec<Option<Arc<BinaryCodeEntry>>> {
        if let Ok(cache) = self.codes.read() {
            vec_ids
                .iter()
                .map(|&id| cache.get(&(embedding, id)).cloned())
                .collect()
        } else {
            vec![None; vec_ids.len()]
        }
    }

    /// Check if a code is cached.
    pub fn contains(&self, embedding: EmbeddingCode, vec_id: VecId) -> bool {
        self.codes
            .read()
            .ok()
            .map(|c| c.contains_key(&(embedding, vec_id)))
            .unwrap_or(false)
    }

    /// Get cache statistics: (entry_count, total_bytes)
    pub fn stats(&self) -> (usize, usize) {
        let count = self.codes.read().map(|c| c.len()).unwrap_or(0);
        let bytes = self.cache_bytes.read().map(|b| *b).unwrap_or(0);
        (count, bytes)
    }

    /// Clear all cached codes.
    pub fn clear(&self) {
        if let Ok(mut cache) = self.codes.write() {
            cache.clear();
        }
        if let Ok(mut bytes) = self.cache_bytes.write() {
            *bytes = 0;
        }
    }

    /// Get number of cached entries for a specific embedding space.
    pub fn count_for_embedding(&self, embedding: EmbeddingCode) -> usize {
        self.codes
            .read()
            .map(|c| c.keys().filter(|(e, _)| *e == embedding).count())
            .unwrap_or(0)
    }
}

impl Default for BinaryCodeCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_correction() -> AdcCorrection {
        AdcCorrection::new(1.0, 1.0)
    }

    #[test]
    fn test_put_and_get() {
        let cache = BinaryCodeCache::new();
        let code = vec![0x12, 0x34, 0x56, 0x78];
        let correction = AdcCorrection::new(0.99, 0.85);

        cache.put(1, 100, code.clone(), correction);
        let retrieved = cache.get(1, 100);

        assert!(retrieved.is_some());
        let entry = retrieved.unwrap();
        assert_eq!(entry.code, code);
        assert_eq!(entry.correction.vector_norm, 0.99);
        assert_eq!(entry.correction.quantization_error, 0.85);
    }

    #[test]
    fn test_get_missing() {
        let cache = BinaryCodeCache::new();
        assert!(cache.get(1, 100).is_none());
    }

    #[test]
    fn test_get_batch() {
        let cache = BinaryCodeCache::new();
        cache.put(1, 10, vec![0x01], default_correction());
        cache.put(1, 20, vec![0x02], default_correction());
        cache.put(1, 30, vec![0x03], default_correction());

        let results = cache.get_batch(1, &[10, 20, 99, 30]);

        assert_eq!(results.len(), 4);
        assert!(results[0].is_some());
        assert_eq!(results[0].as_ref().unwrap().code, vec![0x01]);
        assert!(results[1].is_some());
        assert_eq!(results[1].as_ref().unwrap().code, vec![0x02]);
        assert!(results[2].is_none()); // 99 not in cache
        assert!(results[3].is_some());
        assert_eq!(results[3].as_ref().unwrap().code, vec![0x03]);
    }

    #[test]
    fn test_contains() {
        let cache = BinaryCodeCache::new();
        cache.put(1, 100, vec![0x12], default_correction());

        assert!(cache.contains(1, 100));
        assert!(!cache.contains(1, 200));
        assert!(!cache.contains(2, 100));
    }

    #[test]
    fn test_stats() {
        let cache = BinaryCodeCache::new();
        assert_eq!(cache.stats(), (0, 0));

        cache.put(1, 10, vec![0x01, 0x02, 0x03, 0x04], default_correction()); // 4 + 8 = 12 bytes
        cache.put(1, 20, vec![0x05, 0x06], default_correction()); // 2 + 8 = 10 bytes

        let (count, bytes) = cache.stats();
        assert_eq!(count, 2);
        assert_eq!(bytes, 22); // 12 + 10
    }

    #[test]
    fn test_clear() {
        let cache = BinaryCodeCache::new();
        cache.put(1, 10, vec![0x01], default_correction());
        cache.put(1, 20, vec![0x02], default_correction());

        assert_eq!(cache.stats().0, 2);

        cache.clear();

        assert_eq!(cache.stats(), (0, 0));
        assert!(cache.get(1, 10).is_none());
    }

    #[test]
    fn test_count_for_embedding() {
        let cache = BinaryCodeCache::new();
        cache.put(1, 10, vec![0x01], default_correction());
        cache.put(1, 20, vec![0x02], default_correction());
        cache.put(2, 10, vec![0x03], default_correction());
        cache.put(2, 20, vec![0x04], default_correction());
        cache.put(2, 30, vec![0x05], default_correction());

        assert_eq!(cache.count_for_embedding(1), 2);
        assert_eq!(cache.count_for_embedding(2), 3);
        assert_eq!(cache.count_for_embedding(3), 0);
    }

    #[test]
    fn test_different_embeddings_isolated() {
        let cache = BinaryCodeCache::new();
        cache.put(1, 100, vec![0x11], AdcCorrection::new(1.0, 0.9));
        cache.put(2, 100, vec![0x22], AdcCorrection::new(1.0, 0.8));

        let entry1 = cache.get(1, 100).unwrap();
        let entry2 = cache.get(2, 100).unwrap();

        assert_eq!(entry1.code, vec![0x11]);
        assert_eq!(entry2.code, vec![0x22]);
        assert_eq!(entry1.correction.quantization_error, 0.9);
        assert_eq!(entry2.correction.quantization_error, 0.8);
    }

    #[test]
    fn test_arc_sharing() {
        // Verify that Arc::clone doesn't copy the underlying data
        let cache = BinaryCodeCache::new();
        let code = vec![0x12, 0x34, 0x56, 0x78];
        cache.put(1, 100, code.clone(), default_correction());

        let entry1 = cache.get(1, 100).unwrap();
        let entry2 = cache.get(1, 100).unwrap();

        // Both entries should point to the same underlying allocation
        assert!(Arc::ptr_eq(&entry1, &entry2));
        // Strong count should be 3 (cache + entry1 + entry2)
        assert_eq!(Arc::strong_count(&entry1), 3);
    }

    #[test]
    fn test_overwrite_updates_stats_correctly() {
        // Regression test for CODEX Finding #5: cache_bytes overcounted on overwrite
        let cache = BinaryCodeCache::new();

        // Insert initial entry: 4 bytes code + 8 bytes correction = 12 bytes
        cache.put(1, 100, vec![0x01, 0x02, 0x03, 0x04], default_correction());
        let (count, bytes) = cache.stats();
        assert_eq!(count, 1);
        assert_eq!(bytes, 12);

        // Overwrite with smaller entry: 2 bytes code + 8 bytes correction = 10 bytes
        cache.put(1, 100, vec![0xAA, 0xBB], default_correction());
        let (count, bytes) = cache.stats();
        assert_eq!(count, 1); // Still 1 entry
        assert_eq!(bytes, 10); // Should be 10, not 22 (12 + 10)

        // Overwrite with larger entry: 8 bytes code + 8 bytes correction = 16 bytes
        cache.put(1, 100, vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08], default_correction());
        let (count, bytes) = cache.stats();
        assert_eq!(count, 1); // Still 1 entry
        assert_eq!(bytes, 16); // Should be 16, not 26 (10 + 16)
    }
}
