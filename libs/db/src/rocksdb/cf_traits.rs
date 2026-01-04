//! Column family trait definitions.
//!
//! This module defines the trait hierarchy for column family types:
//!
//! ```text
//!                     ColumnFamily
//!                     (CF_NAME only)
//!                          │
//!          ┌───────────────┼───────────────┐
//!          │               │               │
//!          ▼               ▼               ▼
//!   ColumnFamilyConfig  ColumnFamilyRecord  HotColumnFamilyRecord
//!       <C>            (MessagePack+LZ4)       (rkyv)
//! ```
//!
//! - `ColumnFamily`: Base marker trait with CF_NAME (single source of truth)
//! - `ColumnFamilyConfig<C>`: RocksDB options with domain-specific config type
//! - `ColumnFamilyRecord`: Cold CF serialization using MessagePack + LZ4
//! - `HotColumnFamilyRecord`: Hot CF zero-copy access using rkyv

use anyhow::Result;
use rkyv::validation::validators::DefaultValidator;
use rkyv::{Archive, CheckBytes, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};
use rocksdb::{Cache, Options};
use serde::{Deserialize, Serialize};

use super::DbAccess;

// ============================================================================
// Base Trait: ColumnFamily
// ============================================================================

/// Base marker trait for column family types.
///
/// Provides the single source of truth for CF_NAME. All other CF traits
/// require this as a supertrait, eliminating duplicate CF_NAME definitions.
///
/// # Example
///
/// ```rust,ignore
/// impl ColumnFamily for Nodes {
///     const CF_NAME: &'static str = "graph/nodes";
/// }
/// ```
pub trait ColumnFamily {
    /// Column family name (with prefix, e.g., "graph/nodes", "vector/vectors")
    const CF_NAME: &'static str;
}

// ============================================================================
// Configuration Trait: ColumnFamilyConfig<C>
// ============================================================================

/// RocksDB configuration trait with domain-specific config type.
///
/// The type parameter `C` allows each subsystem to use its own config type:
/// - Graph CFs use `GraphBlockCacheConfig`
/// - Vector CFs use `VectorBlockCacheConfig`
///
/// # Example
///
/// ```rust,ignore
/// impl ColumnFamilyConfig<GraphBlockCacheConfig> for Nodes {
///     fn cf_options(cache: &Cache, config: &GraphBlockCacheConfig) -> Options {
///         let mut opts = Options::default();
///         // Configure based on GraphBlockCacheConfig fields
///         opts
///     }
/// }
/// ```
pub trait ColumnFamilyConfig<C>: ColumnFamily {
    /// Create column family options with shared block cache and config.
    fn cf_options(cache: &Cache, config: &C) -> Options;
}

// ============================================================================
// Cold CF Serialization: ColumnFamilyRecord
// ============================================================================

/// Trait for cold column family record types using MessagePack + LZ4.
///
/// Cold CFs store larger, less frequently accessed data (fragments, summaries).
/// Values are serialized with MessagePack for self-describing format, then
/// compressed with LZ4 for space efficiency.
///
/// Keys use direct byte concatenation (not MessagePack) to enable RocksDB
/// prefix extractors with constant-length prefixes.
///
/// # Example
///
/// ```rust,ignore
/// impl ColumnFamilyRecord for NodeFragments {
///     type Key = NodeFragmentCfKey;
///     type Value = NodeFragmentCfValue;
///     type CreateOp = AddNodeFragment;
///
///     fn record_from(args: &Self::CreateOp) -> (Self::Key, Self::Value) {
///         // Create key-value pair from mutation args
///     }
///
///     fn key_to_bytes(key: &Self::Key) -> Vec<u8> {
///         // Direct byte serialization for prefix extraction
///     }
///
///     fn key_from_bytes(bytes: &[u8]) -> Result<Self::Key> {
///         // Direct byte deserialization
///     }
/// }
/// ```
pub trait ColumnFamilyRecord: ColumnFamily {
    /// The key type for this column family
    type Key: Serialize + for<'de> Deserialize<'de>;

    /// The value type for this column family
    type Value: Serialize + for<'de> Deserialize<'de>;

    /// The argument type for creating records (e.g., mutation type)
    type CreateOp;

    /// Create a key-value pair from arguments
    fn record_from(args: &Self::CreateOp) -> (Self::Key, Self::Value);

    /// Serialize the key to bytes using direct concatenation (no MessagePack).
    /// This enables constant-length prefixes for RocksDB prefix extractors.
    fn key_to_bytes(key: &Self::Key) -> Vec<u8>;

    /// Deserialize the key from bytes (direct format, no MessagePack).
    fn key_from_bytes(bytes: &[u8]) -> Result<Self::Key>;

    /// Serialize the value to bytes using MessagePack, then compress with LZ4.
    fn value_to_bytes(value: &Self::Value) -> Result<Vec<u8>> {
        let msgpack_bytes = rmp_serde::to_vec(value)?;
        let compressed = lz4::block::compress(&msgpack_bytes, None, true)
            .map_err(|e| anyhow::anyhow!("LZ4 compression failed: {}", e))?;
        Ok(compressed)
    }

    /// Decompress with LZ4, then deserialize the value from bytes using MessagePack.
    fn value_from_bytes(bytes: &[u8]) -> Result<Self::Value> {
        let decompressed = lz4::block::decompress(bytes, None)
            .map_err(|e| anyhow::anyhow!("LZ4 decompression failed: {}", e))?;
        let value = rmp_serde::from_slice(&decompressed)?;
        Ok(value)
    }

    /// Create and serialize to bytes using direct encoding for keys, compressed MessagePack for values.
    fn create_bytes(args: &Self::CreateOp) -> Result<(Vec<u8>, Vec<u8>)> {
        let (key, value) = Self::record_from(args);
        let key_bytes = Self::key_to_bytes(&key);
        let value_bytes = Self::value_to_bytes(&value)?;
        Ok((key_bytes, value_bytes))
    }
}

// ============================================================================
// Hot CF Zero-Copy: HotColumnFamilyRecord
// ============================================================================

/// Trait for hot column families using rkyv (zero-copy serialization).
///
/// Hot CFs store small, frequently-accessed data (topology, weights, temporal ranges).
/// Values are serialized with rkyv for zero-copy access during graph traversal.
///
/// # Example
///
/// ```rust,ignore
/// // Zero-copy access (hot path)
/// let archived = Nodes::value_archived(&value_bytes)?;
/// if archived.temporal_range.as_ref().map_or(true, |tr| tr.is_valid_at(now)) {
///     // Use archived data directly without allocation
/// }
///
/// // Full deserialization when ownership is needed (cold path)
/// let value: NodeCfValue = Nodes::value_from_bytes(&value_bytes)?;
/// ```
pub trait HotColumnFamilyRecord: ColumnFamily {
    /// The key type for this column family
    type Key;

    /// The value type for this column family (must implement rkyv traits)
    type Value: Archive + RkyvSerialize<rkyv::ser::serializers::AllocSerializer<256>>;

    /// Serialize key to bytes.
    fn key_to_bytes(key: &Self::Key) -> Vec<u8>;

    /// Deserialize key from bytes.
    fn key_from_bytes(bytes: &[u8]) -> Result<Self::Key>;

    /// Zero-copy value access - returns archived reference without allocation.
    ///
    /// This is the hot path for graph traversal. The returned reference
    /// is valid as long as the input bytes are valid.
    fn value_archived(bytes: &[u8]) -> Result<&<Self::Value as Archive>::Archived>
    where
        <Self::Value as Archive>::Archived: for<'a> CheckBytes<DefaultValidator<'a>>,
    {
        rkyv::check_archived_root::<Self::Value>(bytes)
            .map_err(|e| anyhow::anyhow!("rkyv validation failed: {}", e))
    }

    /// Full deserialization when mutation/ownership is needed.
    ///
    /// This allocates and copies data. Use sparingly - prefer value_archived()
    /// for read-only access.
    fn value_from_bytes(bytes: &[u8]) -> Result<Self::Value>
    where
        <Self::Value as Archive>::Archived: for<'a> CheckBytes<DefaultValidator<'a>>,
        <Self::Value as Archive>::Archived: RkyvDeserialize<Self::Value, rkyv::Infallible>,
    {
        let archived = Self::value_archived(bytes)?;
        Ok(archived.deserialize(&mut rkyv::Infallible).expect("Infallible"))
    }

    /// Serialize value to bytes using rkyv.
    fn value_to_bytes(value: &Self::Value) -> Result<rkyv::AlignedVec> {
        rkyv::to_bytes::<_, 256>(value)
            .map_err(|e| anyhow::anyhow!("rkyv serialization failed: {}", e))
    }
}

// ============================================================================
// Prewarm Helper
// ============================================================================

/// Generic prewarm helper for `ColumnFamilyRecord` CFs.
///
/// Iterates over a column family using trait methods for deserialization,
/// calling the visitor function for each record up to the specified limit.
///
/// # Example
///
/// ```rust,ignore
/// prewarm_cf::<EmbeddingSpecs, _>(db, 1000, |key, value| {
///     cache.register_from_db(key.0, &value.0.model, value.0.dim, value.0.distance);
///     Ok(())
/// })?;
/// ```
pub fn prewarm_cf<CF, F>(db: &dyn DbAccess, limit: usize, mut visitor: F) -> Result<usize>
where
    CF: ColumnFamilyRecord,
    F: FnMut(&CF::Key, &CF::Value) -> Result<()>,
{
    if limit == 0 {
        return Ok(0);
    }

    let iter = db.iterator_cf(CF::CF_NAME)?;
    let mut loaded = 0;

    for item in iter {
        if loaded >= limit {
            break;
        }

        let (key_bytes, value_bytes) = item?;

        let key = CF::key_from_bytes(&key_bytes)?;
        let value = CF::value_from_bytes(&value_bytes)?;
        visitor(&key, &value)?;
        loaded += 1;
    }

    Ok(loaded)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test that traits can be implemented
    struct TestCf;

    impl ColumnFamily for TestCf {
        const CF_NAME: &'static str = "test/cf";
    }

    #[test]
    fn test_column_family_cf_name() {
        assert_eq!(TestCf::CF_NAME, "test/cf");
    }
}
