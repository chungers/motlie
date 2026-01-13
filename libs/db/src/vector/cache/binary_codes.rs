//! In-memory cache for RaBitQ binary codes with ADC corrections.
//!
//! This module provides caching for binary codes and ADC correction factors
//! to enable fast ADC distance computation during RaBitQ search.
//!
//! ADC (Asymmetric Distance Computation) achieves 91-99% recall vs Hamming's
//! 10-24% by keeping the query as float32 and using weighted dot products.

use std::collections::HashMap;
use std::sync::RwLock;

use crate::vector::schema::{AdcCorrection, EmbeddingCode, VecId};

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
/// - Total: 72 bytes per vector
/// - 100K vectors: ~7 MB
/// - 1M vectors: ~70 MB
pub struct BinaryCodeCache {
    /// Cached binary codes with ADC corrections: (embedding_code, vec_id) -> (code, correction)
    codes: RwLock<HashMap<(EmbeddingCode, VecId), (Vec<u8>, AdcCorrection)>>,

    /// Total bytes cached (for monitoring)
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
    pub fn put(
        &self,
        embedding: EmbeddingCode,
        vec_id: VecId,
        code: Vec<u8>,
        correction: AdcCorrection,
    ) {
        if let Ok(mut cache) = self.codes.write() {
            let code_len = code.len();
            cache.insert((embedding, vec_id), (code, correction));
            if let Ok(mut bytes) = self.cache_bytes.write() {
                *bytes += code_len + 8; // +8 for AdcCorrection (2×f32)
            }
        }
    }

    /// Get a binary code with ADC correction from the cache.
    pub fn get(
        &self,
        embedding: EmbeddingCode,
        vec_id: VecId,
    ) -> Option<(Vec<u8>, AdcCorrection)> {
        self.codes.read().ok()?.get(&(embedding, vec_id)).cloned()
    }

    /// Get binary codes with corrections for multiple vectors.
    ///
    /// Returns a vector of Option in the same order as vec_ids.
    /// Missing entries are None.
    pub fn get_batch(
        &self,
        embedding: EmbeddingCode,
        vec_ids: &[VecId],
    ) -> Vec<Option<(Vec<u8>, AdcCorrection)>> {
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
        let (ret_code, ret_corr) = retrieved.unwrap();
        assert_eq!(ret_code, code);
        assert_eq!(ret_corr.vector_norm, 0.99);
        assert_eq!(ret_corr.quantization_error, 0.85);
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
        assert_eq!(results[0].as_ref().unwrap().0, vec![0x01]);
        assert!(results[1].is_some());
        assert_eq!(results[1].as_ref().unwrap().0, vec![0x02]);
        assert!(results[2].is_none()); // 99 not in cache
        assert!(results[3].is_some());
        assert_eq!(results[3].as_ref().unwrap().0, vec![0x03]);
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

        let (code1, corr1) = cache.get(1, 100).unwrap();
        let (code2, corr2) = cache.get(2, 100).unwrap();

        assert_eq!(code1, vec![0x11]);
        assert_eq!(code2, vec![0x22]);
        assert_eq!(corr1.quantization_error, 0.9);
        assert_eq!(corr2.quantization_error, 0.8);
    }
}
