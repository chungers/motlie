//! ID allocation for vector storage.
//!
//! Provides a thread-safe ID allocator that:
//! - Allocates sequential u32 IDs for vectors within an embedding space
//! - Reuses freed IDs before allocating new ones
//! - Persists state to RocksDB for crash recovery
//!
//! # Design
//!
//! Each embedding space has its own ID allocator. The allocator maintains:
//! - `next_id`: The next fresh ID to allocate (monotonically increasing)
//! - `free_ids`: A RoaringBitmap of freed IDs available for reuse
//!
//! Allocation preferentially reuses freed IDs to maintain compact ID space.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

use anyhow::Result;
use roaring::RoaringBitmap;
use rocksdb::TransactionDB;

use super::schema::{
    EmbeddingCode, IdAlloc, IdAllocCfKey, IdAllocCfValue, IdAllocField, VecId,
};
use crate::rocksdb::ColumnFamily;

// ============================================================================
// IdAllocator
// ============================================================================

/// Thread-safe ID allocator with free list support.
///
/// Allocates compact u32 IDs for vectors within a single embedding space.
/// Freed IDs are tracked in a RoaringBitmap and reused before allocating
/// fresh IDs.
///
/// # Thread Safety
///
/// - `next_id` uses atomic operations for lock-free fresh ID allocation
/// - `free_ids` is protected by a mutex for bitmap operations
///
/// # Persistence
///
/// State is persisted to the `IdAlloc` column family:
/// - `(embedding, 0)` -> next_id (u32)
/// - `(embedding, 1)` -> free_ids (serialized RoaringBitmap)
pub struct IdAllocator {
    /// Next fresh ID to allocate (monotonically increasing)
    next_id: AtomicU32,
    /// Free IDs from deletions (protected by mutex for bitmap ops)
    free_ids: Mutex<RoaringBitmap>,
}

impl IdAllocator {
    /// Create a new allocator starting from ID 0.
    pub fn new() -> Self {
        Self {
            next_id: AtomicU32::new(0),
            free_ids: Mutex::new(RoaringBitmap::new()),
        }
    }

    /// Create an allocator with specific initial state.
    ///
    /// Used by `recover()` to restore state from storage.
    pub fn with_state(next_id: u32, free_ids: RoaringBitmap) -> Self {
        Self {
            next_id: AtomicU32::new(next_id),
            free_ids: Mutex::new(free_ids),
        }
    }

    /// Allocate a new u32 ID.
    ///
    /// First tries to reuse a freed ID from the free list.
    /// If no freed IDs are available, allocates a fresh ID.
    ///
    /// # Returns
    ///
    /// A unique u32 ID within this allocator's scope.
    pub fn allocate(&self) -> VecId {
        // First try to reuse a freed ID
        {
            let mut free = self.free_ids.lock().unwrap();
            if let Some(id) = free.iter().next() {
                free.remove(id);
                return id;
            }
        }

        // Otherwise allocate fresh
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Return an ID to the free list for reuse.
    ///
    /// The ID will be available for future allocations.
    /// Does not validate that the ID was previously allocated.
    pub fn free(&self, id: VecId) {
        self.free_ids.lock().unwrap().insert(id);
    }

    /// Get the current next_id value (for testing/debugging).
    pub fn next_id(&self) -> VecId {
        self.next_id.load(Ordering::Relaxed)
    }

    /// Get the number of free IDs available for reuse.
    pub fn free_count(&self) -> u64 {
        self.free_ids.lock().unwrap().len()
    }

    /// Persist allocator state to RocksDB.
    ///
    /// Stores both `next_id` and `free_ids` bitmap to the IdAlloc CF.
    /// Should be called periodically or on shutdown.
    pub fn persist(&self, db: &TransactionDB, embedding: EmbeddingCode) -> Result<()> {
        let cf = db
            .cf_handle(IdAlloc::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("CF {} not found", IdAlloc::CF_NAME))?;

        // Persist next_id
        let next_key = IdAllocCfKey::next_id(embedding);
        let next_val = IdAllocCfValue(IdAllocField::NextId(self.next_id.load(Ordering::SeqCst)));
        db.put_cf(
            &cf,
            IdAlloc::key_to_bytes(&next_key),
            IdAlloc::value_to_bytes(&next_val),
        )?;

        // Persist free bitmap
        let bitmap_key = IdAllocCfKey::free_bitmap(embedding);
        let free = self.free_ids.lock().unwrap();
        let mut bitmap_bytes = Vec::new();
        free.serialize_into(&mut bitmap_bytes)?;
        let bitmap_val = IdAllocCfValue(IdAllocField::FreeBitmap(bitmap_bytes));
        db.put_cf(
            &cf,
            IdAlloc::key_to_bytes(&bitmap_key),
            IdAlloc::value_to_bytes(&bitmap_val),
        )?;

        Ok(())
    }

    /// Recover allocator state from RocksDB.
    ///
    /// Loads `next_id` and `free_ids` from the IdAlloc CF.
    /// Returns a new allocator if no state exists.
    pub fn recover(db: &TransactionDB, embedding: EmbeddingCode) -> Result<Self> {
        let cf = db
            .cf_handle(IdAlloc::CF_NAME)
            .ok_or_else(|| anyhow::anyhow!("CF {} not found", IdAlloc::CF_NAME))?;

        // Load next_id
        let next_key = IdAllocCfKey::next_id(embedding);
        let next_id = match db.get_cf(&cf, IdAlloc::key_to_bytes(&next_key))? {
            Some(bytes) => {
                let val = IdAlloc::value_from_bytes(&next_key, &bytes)?;
                match val.0 {
                    IdAllocField::NextId(v) => v,
                    _ => 0,
                }
            }
            None => 0,
        };

        // Load free bitmap
        let bitmap_key = IdAllocCfKey::free_bitmap(embedding);
        let free_ids = match db.get_cf(&cf, IdAlloc::key_to_bytes(&bitmap_key))? {
            Some(bytes) => {
                let val = IdAlloc::value_from_bytes(&bitmap_key, &bytes)?;
                match val.0 {
                    IdAllocField::FreeBitmap(b) => RoaringBitmap::deserialize_from(&b[..])?,
                    _ => RoaringBitmap::new(),
                }
            }
            None => RoaringBitmap::new(),
        };

        Ok(Self::with_state(next_id, free_ids))
    }
}

impl Default for IdAllocator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_sequential_allocation() {
        let allocator = IdAllocator::new();
        let mut ids = Vec::new();

        for _ in 0..1000 {
            ids.push(allocator.allocate());
        }

        // All IDs should be unique
        let unique: HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), 1000);

        // Should be sequential (when no frees)
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(*id, i as u32);
        }
    }

    #[test]
    fn test_id_reuse() {
        let allocator = IdAllocator::new();

        let id1 = allocator.allocate(); // 0
        let id2 = allocator.allocate(); // 1
        let _id3 = allocator.allocate(); // 2

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);

        allocator.free(id2); // Free ID 1

        let id4 = allocator.allocate(); // Should reuse 1
        assert_eq!(id4, 1);

        let id5 = allocator.allocate(); // Should be 3 (next fresh)
        assert_eq!(id5, 3);
    }

    #[test]
    fn test_free_multiple_reuse() {
        let allocator = IdAllocator::new();

        // Allocate 10 IDs
        let ids: Vec<_> = (0..10).map(|_| allocator.allocate()).collect();
        assert_eq!(allocator.next_id(), 10);

        // Free IDs 2, 5, 7
        allocator.free(ids[2]);
        allocator.free(ids[5]);
        allocator.free(ids[7]);
        assert_eq!(allocator.free_count(), 3);

        // Next allocations should reuse freed IDs (order depends on bitmap iteration)
        let reused: HashSet<_> = (0..3).map(|_| allocator.allocate()).collect();
        assert!(reused.contains(&2));
        assert!(reused.contains(&5));
        assert!(reused.contains(&7));

        // Now should allocate fresh
        let fresh = allocator.allocate();
        assert_eq!(fresh, 10);
    }

    #[test]
    fn test_concurrent_allocation() {
        let allocator = Arc::new(IdAllocator::new());
        let mut handles = Vec::new();

        for _ in 0..10 {
            let alloc = Arc::clone(&allocator);
            handles.push(thread::spawn(move || {
                let mut ids = Vec::new();
                for _ in 0..100 {
                    ids.push(alloc.allocate());
                }
                ids
            }));
        }

        let all_ids: Vec<VecId> = handles.into_iter().flat_map(|h| h.join().unwrap()).collect();

        // All 1000 IDs should be unique
        let unique: HashSet<_> = all_ids.iter().collect();
        assert_eq!(unique.len(), 1000);
    }

    #[test]
    fn test_allocate_free_uniqueness_property() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let allocator = IdAllocator::new();
        let mut live_ids: HashSet<VecId> = HashSet::new();

        for _ in 0..10000 {
            if live_ids.is_empty() || rng.gen_bool(0.7) {
                // Allocate
                let id = allocator.allocate();
                assert!(
                    !live_ids.contains(&id),
                    "Duplicate ID allocated: {}",
                    id
                );
                live_ids.insert(id);
            } else {
                // Free random live ID
                let id = *live_ids.iter().next().unwrap();
                live_ids.remove(&id);
                allocator.free(id);
            }
        }
    }

    #[test]
    fn test_state_accessors() {
        let allocator = IdAllocator::new();

        assert_eq!(allocator.next_id(), 0);
        assert_eq!(allocator.free_count(), 0);

        allocator.allocate();
        allocator.allocate();
        assert_eq!(allocator.next_id(), 2);

        allocator.free(0);
        assert_eq!(allocator.free_count(), 1);
    }

    #[test]
    fn test_with_state() {
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(5);
        bitmap.insert(10);

        let allocator = IdAllocator::with_state(100, bitmap);

        assert_eq!(allocator.next_id(), 100);
        assert_eq!(allocator.free_count(), 2);

        // Should reuse freed IDs first
        let id1 = allocator.allocate();
        let id2 = allocator.allocate();
        assert!(id1 == 5 || id1 == 10);
        assert!(id2 == 5 || id2 == 10);
        assert_ne!(id1, id2);

        // Now should allocate fresh
        let id3 = allocator.allocate();
        assert_eq!(id3, 100);
    }
}
