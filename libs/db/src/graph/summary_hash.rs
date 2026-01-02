//! Summary content hashing for blob separation.
//!
//! This module provides [`SummaryHash`] for content-addressable storage of
//! summaries in cold column families. Unlike [`NameHash`] which hashes the
//! name string, `SummaryHash` hashes the serialized summary content.
//!
//! # Motivation
//!
//! Phase 2 blob separation moves variable-length summaries out of hot CFs
//! (nodes, forward_edges) into dedicated cold CFs (node_summaries, edge_summaries).
//! Using content hashing enables:
//! - Deduplication: identical summaries stored once
//! - Fixed 8-byte reference in hot CF values
//! - Content-addressable storage pattern
//!
//! # Design
//!
//! - [`SummaryHash`]: 8-byte xxHash64 of serialized summary content
//! - Summary CFs use `SummaryHash` as key (content-addressable)
//! - Hot CF values store `Option<SummaryHash>` instead of inline summary

use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};
use std::fmt;
use xxhash_rust::xxh64::xxh64;

/// 8-byte content hash for summary deduplication using xxHash64.
///
/// Unlike NameHash (which hashes the name string), SummaryHash hashes
/// the serialized summary content. This enables:
/// - Deduplication of identical summaries across nodes/edges
/// - Fixed 8-byte key for summary CFs
/// - Efficient content-addressable storage
///
/// # Example
///
/// ```rust,ignore
/// use motlie_db::graph::SummaryHash;
/// use motlie_db::DataUrl;
///
/// let summary = DataUrl::from_text("Person entity");
/// let hash = SummaryHash::from_summary(&summary).unwrap();
/// assert_eq!(hash.as_bytes().len(), 8);
///
/// // Same content always produces same hash
/// let summary2 = DataUrl::from_text("Person entity");
/// let hash2 = SummaryHash::from_summary(&summary2).unwrap();
/// assert_eq!(hash, hash2);
/// ```
/// 8-byte summary content hash (xxHash64).
/// Has rkyv derives for use in hot CF values.
#[derive(Archive, RkyvDeserialize, RkyvSerialize)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[archive(check_bytes)]
#[archive_attr(derive(Clone, Copy, PartialEq, Eq, Hash, Debug))]
pub struct SummaryHash([u8; 8]);

impl SummaryHash {
    /// Size of the hash in bytes (always 8).
    pub const SIZE: usize = 8;

    /// Create a SummaryHash from serialized bytes.
    ///
    /// The input should be the rmp_serde serialized summary content.
    /// This is the low-level method; prefer `from_summary` for typed usage.
    pub fn from_bytes_content(content: &[u8]) -> Self {
        let hash = xxh64(content, 0); // seed = 0
        SummaryHash(hash.to_be_bytes())
    }

    /// Create a SummaryHash from a serializable summary type.
    ///
    /// Serializes the summary using MessagePack, then hashes the bytes.
    /// This is the primary method for creating SummaryHash from summaries.
    pub fn from_summary<T: Serialize>(summary: &T) -> Result<Self, anyhow::Error> {
        let bytes = rmp_serde::to_vec(summary)?;
        Ok(Self::from_bytes_content(&bytes))
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
        SummaryHash(bytes)
    }
}

impl fmt::Debug for SummaryHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Display as hex for readability
        write!(f, "SummaryHash({:016x})", u64::from_be_bytes(self.0))
    }
}

impl fmt::Display for SummaryHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:016x}", u64::from_be_bytes(self.0))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DataUrl;

    #[test]
    fn test_summary_hash_deterministic() {
        let summary1 = DataUrl::from_text("Test summary content");
        let summary2 = DataUrl::from_text("Test summary content");

        let hash1 = SummaryHash::from_summary(&summary1).unwrap();
        let hash2 = SummaryHash::from_summary(&summary2).unwrap();

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_summary_hash_different_content() {
        let summary1 = DataUrl::from_text("Content A");
        let summary2 = DataUrl::from_text("Content B");

        let hash1 = SummaryHash::from_summary(&summary1).unwrap();
        let hash2 = SummaryHash::from_summary(&summary2).unwrap();

        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_summary_hash_size() {
        let summary = DataUrl::from_text("test");
        let hash = SummaryHash::from_summary(&summary).unwrap();

        assert_eq!(hash.as_bytes().len(), 8);
        assert_eq!(SummaryHash::SIZE, 8);
    }

    #[test]
    fn test_summary_hash_from_bytes_roundtrip() {
        let summary = DataUrl::from_text("test summary");
        let original = SummaryHash::from_summary(&summary).unwrap();

        let bytes = *original.as_bytes();
        let recovered = SummaryHash::from_bytes(bytes);

        assert_eq!(original, recovered);
    }

    #[test]
    fn test_summary_hash_debug_format() {
        let summary = DataUrl::from_text("test");
        let hash = SummaryHash::from_summary(&summary).unwrap();

        let debug = format!("{:?}", hash);
        assert!(debug.starts_with("SummaryHash("));
        assert!(debug.ends_with(")"));
    }

    #[test]
    fn test_summary_hash_display_format() {
        let summary = DataUrl::from_text("test");
        let hash = SummaryHash::from_summary(&summary).unwrap();

        let display = format!("{}", hash);
        assert_eq!(display.len(), 16); // 8 bytes = 16 hex chars
    }

    #[test]
    fn test_summary_hash_serialization() {
        let summary = DataUrl::from_text("serialize me");
        let hash = SummaryHash::from_summary(&summary).unwrap();

        let serialized = rmp_serde::to_vec(&hash).expect("serialize");
        let deserialized: SummaryHash = rmp_serde::from_slice(&serialized).expect("deserialize");

        assert_eq!(hash, deserialized);
    }

    #[test]
    fn test_summary_hash_empty_content() {
        let summary = DataUrl::from_text("");
        let hash = SummaryHash::from_summary(&summary).unwrap();

        assert_eq!(hash.as_bytes().len(), 8);
    }

    #[test]
    fn test_summary_hash_large_content() {
        let large_content = "x".repeat(100_000);
        let summary = DataUrl::from_text(&large_content);
        let hash = SummaryHash::from_summary(&summary).unwrap();

        assert_eq!(hash.as_bytes().len(), 8);
    }

    #[test]
    fn test_summary_hash_from_bytes_content() {
        let content = b"raw bytes content";
        let hash = SummaryHash::from_bytes_content(content);

        assert_eq!(hash.as_bytes().len(), 8);

        // Same content should produce same hash
        let hash2 = SummaryHash::from_bytes_content(content);
        assert_eq!(hash, hash2);
    }
}
