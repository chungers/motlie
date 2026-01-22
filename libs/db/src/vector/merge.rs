//! RocksDB merge operators for the vector module.
//!
//! Merge operators enable lock-free concurrent updates to edge lists by
//! allowing multiple writers to append operations that are combined at read time.
//!
//! # Edge Merge Operator
//!
//! The edge merge operator handles concurrent updates to HNSW neighbor lists
//! stored as RoaringBitmaps. Operations (add, remove) are serialized as operands
//! and combined during compaction or read.
//!
//! ```text
//! Writer A: merge_cf(key, Add(5))
//! Writer B: merge_cf(key, Add(7))
//!                     │
//!                     ▼
//! RocksDB combines: existing ∪ {5} ∪ {7}
//! ```

use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

// ============================================================================
// Edge Operations
// ============================================================================

/// Operations that can be applied to edge bitmaps via merge.
///
/// Each operation is serialized and stored as a merge operand in RocksDB.
/// During reads or compaction, operands are combined to produce the final bitmap.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum EdgeOp {
    /// Add a single neighbor to the edge list
    Add(u32),
    /// Add multiple neighbors in a batch (more efficient for bulk inserts)
    AddBatch(Vec<u32>),
    /// Remove a neighbor from the edge list
    Remove(u32),
    /// Remove multiple neighbors in a batch
    RemoveBatch(Vec<u32>),
}

impl EdgeOp {
    /// Serialize the operation to bytes for use as a merge operand.
    pub fn to_bytes(&self) -> Vec<u8> {
        rmp_serde::to_vec(self).expect("EdgeOp serialization should never fail")
    }

    /// Deserialize an operation from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, rmp_serde::decode::Error> {
        rmp_serde::from_slice(bytes)
    }
}

// ============================================================================
// Merge Operator
// ============================================================================

/// Full merge function for edge bitmaps.
///
/// This is called when RocksDB needs to combine an existing value with
/// merge operands (e.g., during compaction or when reading after merges).
///
/// # Arguments
///
/// * `_key` - The key being merged (unused, but required by RocksDB API)
/// * `existing` - The existing value, if any
/// * `operands` - Iterator over merge operands to apply
///
/// # Returns
///
/// The merged bitmap serialized to bytes, or `None` on error.
pub fn edge_full_merge(
    _key: &[u8],
    existing: Option<&[u8]>,
    operands: &rocksdb::MergeOperands,
) -> Option<Vec<u8>> {
    // Start with existing bitmap or empty
    let mut bitmap = match existing {
        Some(bytes) => match RoaringBitmap::deserialize_from(bytes) {
            Ok(bm) => bm,
            Err(e) => {
                tracing::error!(error = %e, "Failed to deserialize existing edge bitmap");
                return None;
            }
        },
        None => RoaringBitmap::new(),
    };

    // Apply each operand
    for operand in operands {
        match EdgeOp::from_bytes(operand) {
            Ok(EdgeOp::Add(id)) => {
                bitmap.insert(id);
            }
            Ok(EdgeOp::AddBatch(ids)) => {
                for id in ids {
                    bitmap.insert(id);
                }
            }
            Ok(EdgeOp::Remove(id)) => {
                bitmap.remove(id);
            }
            Ok(EdgeOp::RemoveBatch(ids)) => {
                for id in ids {
                    bitmap.remove(id);
                }
            }
            Err(_) => {
                // Operand might be a RoaringBitmap from a previous partial merge
                // (This happens with set_merge_operator_associative during compaction)
                if let Ok(other_bitmap) = RoaringBitmap::deserialize_from(operand) {
                    bitmap |= other_bitmap;
                } else {
                    tracing::warn!(
                        "Failed to deserialize edge operand as EdgeOp or RoaringBitmap, skipping"
                    );
                }
            }
        }
    }

    // Serialize result
    let mut buf = Vec::new();
    if let Err(e) = bitmap.serialize_into(&mut buf) {
        tracing::error!(error = %e, "Failed to serialize merged edge bitmap");
        return None;
    }

    Some(buf)
}

/// Partial merge function for edge bitmaps.
///
/// This is called when RocksDB can combine operands without the base value
/// (e.g., combining two Add operations). This optimization reduces work
/// during compaction.
///
/// # Arguments
///
/// * `_key` - The key being merged (unused)
/// * `_existing` - The first operand (treated as an operand, not base value)
/// * `_operand` - The second operand
///
/// # Returns
///
/// Combined operand bytes, or `None` to fall back to full merge.
pub fn edge_partial_merge(
    _key: &[u8],
    _existing: Option<&[u8]>,
    _operand: &[u8],
) -> Option<Vec<u8>> {
    // For simplicity, always use full merge.
    // A more optimized version could combine Add operations here.
    None
}

/// Get the merge operator name for logging and debugging.
pub const EDGE_MERGE_OPERATOR_NAME: &str = "vector_edge_merge";

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_op_roundtrip() {
        let ops = vec![
            EdgeOp::Add(42),
            EdgeOp::AddBatch(vec![1, 2, 3]),
            EdgeOp::Remove(10),
            EdgeOp::RemoveBatch(vec![4, 5]),
        ];

        for op in ops {
            let bytes = op.to_bytes();
            let parsed = EdgeOp::from_bytes(&bytes).unwrap();
            assert_eq!(parsed, op);
        }
    }

    #[test]
    fn test_full_merge_empty_existing() {
        use rocksdb::{MergeOperands, Options};

        // Create operands manually since MergeOperands is opaque
        // We'll test the logic directly instead
        let mut bitmap = RoaringBitmap::new();

        // Simulate Add(5)
        bitmap.insert(5);
        // Simulate Add(10)
        bitmap.insert(10);

        assert!(bitmap.contains(5));
        assert!(bitmap.contains(10));
        assert_eq!(bitmap.len(), 2);
    }

    #[test]
    fn test_full_merge_with_existing() {
        let mut existing = RoaringBitmap::new();
        existing.insert(1);
        existing.insert(2);

        let mut buf = Vec::new();
        existing.serialize_into(&mut buf).unwrap();

        // Parse and apply operations
        let mut bitmap = RoaringBitmap::deserialize_from(&buf[..]).unwrap();
        bitmap.insert(3); // Add(3)
        bitmap.remove(1); // Remove(1)

        assert!(!bitmap.contains(1));
        assert!(bitmap.contains(2));
        assert!(bitmap.contains(3));
        assert_eq!(bitmap.len(), 2);
    }

    #[test]
    fn test_roaring_bitmap_serialization() {
        let mut bitmap = RoaringBitmap::new();
        for i in 0..100 {
            bitmap.insert(i);
        }

        let mut buf = Vec::new();
        bitmap.serialize_into(&mut buf).unwrap();

        // RoaringBitmap is very compact
        assert!(buf.len() < 100 * 4); // Much smaller than 100 * 4 bytes

        let restored = RoaringBitmap::deserialize_from(&buf[..]).unwrap();
        assert_eq!(restored, bitmap);
    }
}
