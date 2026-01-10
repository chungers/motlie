//! In-memory cache for RaBitQ binary codes.
//!
//! This module provides caching for binary codes to enable fast Hamming distance
//! computation during RaBitQ search, avoiding the "double I/O" problem.

use std::collections::HashMap;
use std::sync::RwLock;

use crate::vector::schema::{EmbeddingCode, VecId};

/// In-memory cache for RaBitQ binary codes.
///
/// Caches binary codes to avoid RocksDB reads during Hamming-based beam search.
/// This addresses the "double I/O" problem where RaBitQ is slower than standard
/// L2 search due to fetching both binary codes and vectors from disk.
///
/// # Background
///
/// Task 4.9 benchmarking revealed that RaBitQ with RocksDB-backed binary codes
/// is SLOWER at 100K scale because:
/// 1. Binary code fetch: RocksDB read for each candidate's code
/// 2. Vector fetch: RocksDB read for top candidates during re-ranking
///
/// With in-memory caching, Hamming distance computation becomes nearly free
/// (SIMD popcount on cached bytes), making RaBitQ viable for speedup.
///
/// # Memory Usage
///
/// For 1-bit RaBitQ at 128D:
/// - Binary code: 16 bytes per vector
/// - 100K vectors: ~1.6 MB
/// - 1M vectors: ~16 MB
///
/// This is small compared to the 512 bytes/vector for full vectors.
pub struct BinaryCodeCache {
    /// Cached binary codes: (embedding_code, vec_id) -> binary_code
    codes: RwLock<HashMap<(EmbeddingCode, VecId), Vec<u8>>>,

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

    /// Insert a binary code into the cache.
    pub fn put(&self, embedding: EmbeddingCode, vec_id: VecId, code: Vec<u8>) {
        if let Ok(mut cache) = self.codes.write() {
            let code_len = code.len();
            cache.insert((embedding, vec_id), code);
            if let Ok(mut bytes) = self.cache_bytes.write() {
                *bytes += code_len;
            }
        }
    }

    /// Get a binary code from the cache.
    pub fn get(&self, embedding: EmbeddingCode, vec_id: VecId) -> Option<Vec<u8>> {
        self.codes.read().ok()?.get(&(embedding, vec_id)).cloned()
    }

    /// Get binary codes for multiple vectors.
    ///
    /// Returns a vector of Option<Vec<u8>> in the same order as vec_ids.
    /// Missing codes are None.
    pub fn get_batch(&self, embedding: EmbeddingCode, vec_ids: &[VecId]) -> Vec<Option<Vec<u8>>> {
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

    #[test]
    fn test_put_and_get() {
        let cache = BinaryCodeCache::new();
        let code = vec![0x12, 0x34, 0x56, 0x78];

        cache.put(1, 100, code.clone());
        let retrieved = cache.get(1, 100);

        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap(), code);
    }

    #[test]
    fn test_get_missing() {
        let cache = BinaryCodeCache::new();
        assert!(cache.get(1, 100).is_none());
    }

    #[test]
    fn test_get_batch() {
        let cache = BinaryCodeCache::new();
        cache.put(1, 10, vec![0x01]);
        cache.put(1, 20, vec![0x02]);
        cache.put(1, 30, vec![0x03]);

        let results = cache.get_batch(1, &[10, 20, 99, 30]);

        assert_eq!(results.len(), 4);
        assert_eq!(results[0], Some(vec![0x01]));
        assert_eq!(results[1], Some(vec![0x02]));
        assert_eq!(results[2], None); // 99 not in cache
        assert_eq!(results[3], Some(vec![0x03]));
    }

    #[test]
    fn test_contains() {
        let cache = BinaryCodeCache::new();
        cache.put(1, 100, vec![0x12]);

        assert!(cache.contains(1, 100));
        assert!(!cache.contains(1, 200));
        assert!(!cache.contains(2, 100));
    }

    #[test]
    fn test_stats() {
        let cache = BinaryCodeCache::new();
        assert_eq!(cache.stats(), (0, 0));

        cache.put(1, 10, vec![0x01, 0x02, 0x03, 0x04]); // 4 bytes
        cache.put(1, 20, vec![0x05, 0x06]); // 2 bytes

        let (count, bytes) = cache.stats();
        assert_eq!(count, 2);
        assert_eq!(bytes, 6);
    }

    #[test]
    fn test_clear() {
        let cache = BinaryCodeCache::new();
        cache.put(1, 10, vec![0x01]);
        cache.put(1, 20, vec![0x02]);

        assert_eq!(cache.stats().0, 2);

        cache.clear();

        assert_eq!(cache.stats(), (0, 0));
        assert!(cache.get(1, 10).is_none());
    }

    #[test]
    fn test_count_for_embedding() {
        let cache = BinaryCodeCache::new();
        cache.put(1, 10, vec![0x01]);
        cache.put(1, 20, vec![0x02]);
        cache.put(2, 10, vec![0x03]);
        cache.put(2, 20, vec![0x04]);
        cache.put(2, 30, vec![0x05]);

        assert_eq!(cache.count_for_embedding(1), 2);
        assert_eq!(cache.count_for_embedding(2), 3);
        assert_eq!(cache.count_for_embedding(3), 0);
    }

    #[test]
    fn test_different_embeddings_isolated() {
        let cache = BinaryCodeCache::new();
        cache.put(1, 100, vec![0x11]);
        cache.put(2, 100, vec![0x22]);

        assert_eq!(cache.get(1, 100), Some(vec![0x11]));
        assert_eq!(cache.get(2, 100), Some(vec![0x22]));
    }
}
