//! Name interning module for compact, fixed-size name storage.
//!
//! This module provides [`NameHash`] for replacing variable-length string names
//! with fixed 8-byte hashes in RocksDB keys, and [`NameCache`] for efficient
//! hash-to-name resolution.
//!
//! # Motivation
//!
//! RocksDB block cache stores entire data blocks (~4KB). Variable-length keys
//! reduce cache density and prefix compression effectiveness. By using fixed
//! 8-byte hashes, we achieve:
//! - 35% more edges per cache block
//! - Excellent prefix compression (32-byte shared prefix for same src+dst)
//! - Name deduplication across edges
//!
//! # Design
//!
//! - [`NameHash`]: 8-byte xxHash64 of the name string
//! - [`NameCache`]: Thread-safe in-memory cache for hash→name resolution
//! - `Names` CF: Persistent storage of hash→name mappings in RocksDB

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;
use xxhash_rust::xxh64::xxh64;

/// 8-byte name hash for compact, fixed-size keys.
///
/// Uses xxHash64 for fast, non-cryptographic hashing. The 8-byte output
/// provides a 2^64 namespace, making collisions negligible for typical
/// graph workloads (millions of distinct names).
///
/// # Example
///
/// ```rust,ignore
/// use motlie_db::graph::NameHash;
///
/// let hash = NameHash::from_name("FOLLOWS");
/// assert_eq!(hash.as_bytes().len(), 8);
///
/// // Same name always produces same hash
/// let hash2 = NameHash::from_name("FOLLOWS");
/// assert_eq!(hash, hash2);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NameHash([u8; 8]);

impl NameHash {
    /// Create a NameHash from a string name using xxHash64.
    ///
    /// The hash is deterministic: the same name always produces the same hash.
    pub fn from_name(name: &str) -> Self {
        let hash = xxh64(name.as_bytes(), 0); // seed = 0
        NameHash(hash.to_be_bytes())
    }

    /// Get the raw bytes of the hash.
    ///
    /// Returns a reference to the internal 8-byte array.
    #[inline]
    pub fn as_bytes(&self) -> &[u8; 8] {
        &self.0
    }

    /// Create from raw bytes.
    ///
    /// Used when deserializing from RocksDB keys.
    #[inline]
    pub fn from_bytes(bytes: [u8; 8]) -> Self {
        NameHash(bytes)
    }

    /// Size of the hash in bytes (always 8).
    pub const SIZE: usize = 8;
}

impl fmt::Debug for NameHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display as hex for readability
        write!(f, "NameHash({:016x})", u64::from_be_bytes(self.0))
    }
}

impl fmt::Display for NameHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}", u64::from_be_bytes(self.0))
    }
}

/// Thread-safe in-memory cache for name hash resolution.
///
/// Provides fast lookups from hash to name without hitting RocksDB.
/// The cache is populated lazily on first access and never evicts
/// (names are small and finite in typical workloads).
///
/// # Thread Safety
///
/// Uses `DashMap` for lock-free concurrent access. Multiple threads
/// can read and write simultaneously without blocking.
///
/// # Example
///
/// ```rust,ignore
/// use motlie_db::graph::{NameHash, NameCache};
///
/// let cache = NameCache::new();
///
/// // Insert a name
/// let hash = NameHash::from_name("FOLLOWS");
/// cache.insert(hash, "FOLLOWS".to_string());
///
/// // Resolve it later
/// let name = cache.get(&hash);
/// assert_eq!(name.as_deref(), Some("FOLLOWS"));
/// ```
#[derive(Debug)]
pub struct NameCache {
    /// Hash → Name mapping for resolution
    hash_to_name: DashMap<NameHash, Arc<String>>,
    /// Name → Hash mapping for interning (avoids re-hashing)
    /// Uses String as key for O(1) lookup by &str
    name_to_hash: DashMap<String, NameHash>,
}

impl Default for NameCache {
    fn default() -> Self {
        Self::new()
    }
}

impl NameCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self {
            hash_to_name: DashMap::new(),
            name_to_hash: DashMap::new(),
        }
    }

    /// Create a cache with pre-allocated capacity.
    ///
    /// Use when you know approximately how many distinct names to expect.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            hash_to_name: DashMap::with_capacity(capacity),
            name_to_hash: DashMap::with_capacity(capacity),
        }
    }

    /// Intern a name: compute hash, cache both directions, return hash.
    ///
    /// If the name is already cached, returns the existing hash without
    /// recomputing. This is the primary write-path method.
    pub fn intern(&self, name: &str) -> NameHash {
        // Check if already interned
        if let Some(hash) = self.name_to_hash.get(name) {
            return *hash;
        }

        // Compute hash and cache
        let hash = NameHash::from_name(name);
        let name_owned = name.to_string();
        let name_arc = Arc::new(name_owned.clone());

        // Insert in both directions
        self.hash_to_name.insert(hash, name_arc);
        self.name_to_hash.insert(name_owned, hash);

        hash
    }

    /// Insert a hash→name mapping directly.
    ///
    /// Used when loading from RocksDB Names CF.
    pub fn insert(&self, hash: NameHash, name: String) {
        let name_arc = Arc::new(name.clone());
        self.hash_to_name.insert(hash, name_arc);
        self.name_to_hash.insert(name, hash);
    }

    /// Get the name for a hash, if cached.
    ///
    /// Returns `None` if the hash is not in the cache. The caller
    /// should fall back to RocksDB lookup and then call `insert`.
    pub fn get(&self, hash: &NameHash) -> Option<Arc<String>> {
        self.hash_to_name.get(hash).map(|r| r.value().clone())
    }

    /// Check if a hash is in the cache.
    pub fn contains(&self, hash: &NameHash) -> bool {
        self.hash_to_name.contains_key(hash)
    }

    /// Get the number of cached names.
    pub fn len(&self) -> usize {
        self.hash_to_name.len()
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.hash_to_name.is_empty()
    }

    /// Clear all cached entries.
    ///
    /// Typically only used in tests.
    pub fn clear(&self) {
        self.hash_to_name.clear();
        self.name_to_hash.clear();
    }

    /// Check if a name is already interned (by name string).
    ///
    /// Returns the hash if the name is already in the cache, None otherwise.
    /// This is useful for the write path to avoid redundant DB writes.
    pub fn get_hash(&self, name: &str) -> Option<NameHash> {
        self.name_to_hash.get(name).map(|r| *r)
    }

    /// Intern a name only if it's not already in the cache.
    ///
    /// Returns `(hash, is_new)` where `is_new` is true if this is a new entry.
    /// This is the primary write-path method that allows callers to skip
    /// redundant Names CF writes.
    pub fn intern_if_new(&self, name: &str) -> (NameHash, bool) {
        // Check if already interned
        if let Some(hash) = self.name_to_hash.get(name) {
            return (*hash, false);
        }

        // Compute hash and cache
        let hash = NameHash::from_name(name);
        let name_owned = name.to_string();
        let name_arc = Arc::new(name_owned.clone());

        // Insert in both directions
        self.hash_to_name.insert(hash, name_arc);
        self.name_to_hash.insert(name_owned, hash);

        (hash, true)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name_hash_deterministic() {
        let hash1 = NameHash::from_name("FOLLOWS");
        let hash2 = NameHash::from_name("FOLLOWS");
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_name_hash_different_names() {
        let hash1 = NameHash::from_name("FOLLOWS");
        let hash2 = NameHash::from_name("LIKES");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_name_hash_size() {
        let hash = NameHash::from_name("test");
        assert_eq!(hash.as_bytes().len(), 8);
        assert_eq!(NameHash::SIZE, 8);
    }

    #[test]
    fn test_name_hash_from_bytes_roundtrip() {
        let original = NameHash::from_name("test_name");
        let bytes = *original.as_bytes();
        let recovered = NameHash::from_bytes(bytes);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_name_hash_debug_format() {
        let hash = NameHash::from_name("test");
        let debug = format!("{:?}", hash);
        assert!(debug.starts_with("NameHash("));
        assert!(debug.ends_with(")"));
    }

    #[test]
    fn test_name_hash_display_format() {
        let hash = NameHash::from_name("test");
        let display = format!("{}", hash);
        assert_eq!(display.len(), 16); // 8 bytes = 16 hex chars
    }

    #[test]
    fn test_name_hash_serialization() {
        let hash = NameHash::from_name("FOLLOWS");
        let serialized = rmp_serde::to_vec(&hash).expect("serialize");
        let deserialized: NameHash = rmp_serde::from_slice(&serialized).expect("deserialize");
        assert_eq!(hash, deserialized);
    }

    #[test]
    fn test_name_cache_new() {
        let cache = NameCache::new();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_name_cache_intern() {
        let cache = NameCache::new();
        let hash = cache.intern("FOLLOWS");

        // Should be cached
        assert!(cache.contains(&hash));
        assert_eq!(cache.len(), 1);

        // Get should return the name
        let name = cache.get(&hash).expect("should be cached");
        assert_eq!(&*name, "FOLLOWS");
    }

    #[test]
    fn test_name_cache_intern_idempotent() {
        let cache = NameCache::new();

        let hash1 = cache.intern("FOLLOWS");
        let hash2 = cache.intern("FOLLOWS");

        assert_eq!(hash1, hash2);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_name_cache_multiple_names() {
        let cache = NameCache::new();

        let hash1 = cache.intern("FOLLOWS");
        let hash2 = cache.intern("LIKES");
        let hash3 = cache.intern("KNOWS");

        assert_ne!(hash1, hash2);
        assert_ne!(hash2, hash3);
        assert_eq!(cache.len(), 3);

        assert_eq!(cache.get(&hash1).as_deref().map(|s| s.as_str()), Some("FOLLOWS"));
        assert_eq!(cache.get(&hash2).as_deref().map(|s| s.as_str()), Some("LIKES"));
        assert_eq!(cache.get(&hash3).as_deref().map(|s| s.as_str()), Some("KNOWS"));
    }

    #[test]
    fn test_name_cache_insert_direct() {
        let cache = NameCache::new();
        let hash = NameHash::from_name("FOLLOWS");

        cache.insert(hash, "FOLLOWS".to_string());

        assert!(cache.contains(&hash));
        assert_eq!(cache.get(&hash).as_deref().map(|s| s.as_str()), Some("FOLLOWS"));
    }

    #[test]
    fn test_name_cache_get_missing() {
        let cache = NameCache::new();
        let hash = NameHash::from_name("UNKNOWN");

        assert!(!cache.contains(&hash));
        assert!(cache.get(&hash).is_none());
    }

    #[test]
    fn test_name_cache_clear() {
        let cache = NameCache::new();
        cache.intern("FOLLOWS");
        cache.intern("LIKES");

        assert_eq!(cache.len(), 2);

        cache.clear();

        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_name_cache_concurrent_access() {
        use std::thread;

        let cache = Arc::new(NameCache::new());
        let mut handles = vec![];

        // Spawn multiple threads interning names
        for i in 0..10 {
            let cache_clone = cache.clone();
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    let name = format!("edge_type_{}_{}", i, j);
                    cache_clone.intern(&name);
                }
            }));
        }

        // Wait for all threads
        for handle in handles {
            handle.join().expect("thread should complete");
        }

        // Should have 1000 unique entries
        assert_eq!(cache.len(), 1000);
    }

    #[test]
    fn test_name_hash_empty_string() {
        // Empty string should still produce a valid hash
        let hash = NameHash::from_name("");
        assert_eq!(hash.as_bytes().len(), 8);

        let cache = NameCache::new();
        let cached_hash = cache.intern("");
        assert_eq!(hash, cached_hash);
    }

    #[test]
    fn test_name_hash_unicode() {
        // Unicode names should work correctly
        let hash1 = NameHash::from_name("関係");
        let hash2 = NameHash::from_name("関係");
        let hash3 = NameHash::from_name("関連");

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);

        let cache = NameCache::new();
        let cached = cache.intern("関係");
        assert_eq!(hash1, cached);
        assert_eq!(cache.get(&cached).as_deref().map(|s| s.as_str()), Some("関係"));
    }

    #[test]
    fn test_name_hash_long_string() {
        // Very long strings should hash to same 8 bytes
        let long_name = "a".repeat(10000);
        let hash = NameHash::from_name(&long_name);
        assert_eq!(hash.as_bytes().len(), 8);

        let cache = NameCache::new();
        cache.intern(&long_name);
        assert_eq!(cache.get(&hash).as_deref().map(|s| s.len()), Some(10000));
    }

    #[test]
    fn test_name_cache_get_hash() {
        let cache = NameCache::new();

        // Not interned yet
        assert!(cache.get_hash("FOLLOWS").is_none());

        // Intern it
        let hash = cache.intern("FOLLOWS");

        // Now should be found
        assert_eq!(cache.get_hash("FOLLOWS"), Some(hash));
    }

    #[test]
    fn test_name_cache_intern_if_new() {
        let cache = NameCache::new();

        // First time - should be new
        let (hash1, is_new1) = cache.intern_if_new("FOLLOWS");
        assert!(is_new1, "First intern should be new");

        // Second time - should not be new
        let (hash2, is_new2) = cache.intern_if_new("FOLLOWS");
        assert!(!is_new2, "Second intern should not be new");
        assert_eq!(hash1, hash2, "Hash should be the same");

        // Different name - should be new
        let (hash3, is_new3) = cache.intern_if_new("LIKES");
        assert!(is_new3, "Different name should be new");
        assert_ne!(hash1, hash3, "Different names should have different hashes");

        // Cache should have 2 entries
        assert_eq!(cache.len(), 2);
    }
}
