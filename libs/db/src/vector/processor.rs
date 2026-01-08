//! Vector processor - central state management for vector operations.
//!
//! The Processor is the central component that mutation executors and query
//! executors use to access the database and manage per-embedding state.
//!
//! # Design
//!
//! The Processor follows the same pattern as `graph::mutation::Processor`:
//! - Holds shared storage and registry references
//! - Manages per-embedding ID allocators (lazily created)
//! - Provides helper methods for common operations
//!
//! # Thread Safety
//!
//! The Processor is designed for concurrent access:
//! - Storage is Arc-wrapped for shared ownership
//! - Registry is Arc-wrapped for shared lookups
//! - IdAllocators use DashMap for lock-free concurrent access

use std::sync::Arc;

use anyhow::Result;
use dashmap::DashMap;

use super::config::RaBitQConfig;
use super::id::IdAllocator;
use super::rabitq::RaBitQ;
use super::registry::EmbeddingRegistry;
use super::schema::EmbeddingCode;
use super::Storage;

// ============================================================================
// Processor
// ============================================================================

/// Vector processor - provides storage access and state management.
///
/// This is the central component that mutation executors and query executors
/// use to access the database and manage per-embedding state.
///
/// # Example
///
/// ```rust,ignore
/// use std::sync::Arc;
/// use motlie_db::vector::{Storage, Processor, EmbeddingRegistry};
///
/// let storage = Arc::new(Storage::readwrite(path)?);
/// let registry = storage.cache().clone();
/// let processor = Processor::new(storage, registry);
///
/// // Get or create an ID allocator for an embedding space
/// let allocator = processor.get_or_create_allocator(embedding_code);
/// let vec_id = allocator.allocate();
/// ```
pub struct Processor {
    /// Vector storage (RocksDB via generic Storage<Subsystem>)
    storage: Arc<Storage>,
    /// Embedding registry (pre-warmed on startup)
    registry: Arc<EmbeddingRegistry>,
    /// Per-embedding ID allocators (lazily created)
    id_allocators: DashMap<EmbeddingCode, IdAllocator>,
    /// Per-embedding RaBitQ encoders (lazily created)
    rabitq_encoders: DashMap<EmbeddingCode, Arc<RaBitQ>>,
    /// RaBitQ configuration (shared across all embeddings)
    rabitq_config: RaBitQConfig,
}

impl Processor {
    /// Create a new Processor with the given storage and registry.
    pub fn new(storage: Arc<Storage>, registry: Arc<EmbeddingRegistry>) -> Self {
        Self::with_rabitq_config(storage, registry, RaBitQConfig::default())
    }

    /// Create a Processor with custom RaBitQ configuration.
    pub fn with_rabitq_config(
        storage: Arc<Storage>,
        registry: Arc<EmbeddingRegistry>,
        rabitq_config: RaBitQConfig,
    ) -> Self {
        Self {
            storage,
            registry,
            id_allocators: DashMap::new(),
            rabitq_encoders: DashMap::new(),
            rabitq_config,
        }
    }

    /// Get access to the underlying storage.
    ///
    /// Query types use this to execute themselves via QueryExecutor::execute()
    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    /// Get access to the Arc-wrapped storage.
    pub fn storage_arc(&self) -> &Arc<Storage> {
        &self.storage
    }

    /// Get access to the embedding registry.
    pub fn registry(&self) -> &EmbeddingRegistry {
        &self.registry
    }

    /// Get access to the Arc-wrapped registry.
    pub fn registry_arc(&self) -> &Arc<EmbeddingRegistry> {
        &self.registry
    }

    /// Get or create an ID allocator for the given embedding space.
    ///
    /// If an allocator doesn't exist for this embedding, it will:
    /// 1. Try to recover state from storage
    /// 2. Fall back to creating a new allocator
    ///
    /// The returned reference is a DashMap RefMut, allowing allocation.
    pub fn get_or_create_allocator(
        &self,
        embedding: EmbeddingCode,
    ) -> dashmap::mapref::one::RefMut<'_, EmbeddingCode, IdAllocator> {
        self.id_allocators.entry(embedding).or_insert_with(|| {
            // Try to recover from storage, or create new
            match self.storage.transaction_db() {
                Ok(db) => IdAllocator::recover(&db, embedding).unwrap_or_default(),
                Err(_) => IdAllocator::new(),
            }
        })
    }

    /// Check if an allocator exists for the given embedding space.
    pub fn has_allocator(&self, embedding: EmbeddingCode) -> bool {
        self.id_allocators.contains_key(&embedding)
    }

    /// Persist all ID allocators to storage.
    ///
    /// Should be called periodically or on shutdown to ensure
    /// crash recovery works correctly.
    pub fn persist_allocators(&self) -> Result<()> {
        let db = self.storage.transaction_db()?;
        for entry in self.id_allocators.iter() {
            let embedding = *entry.key();
            let allocator = entry.value();
            allocator.persist(&db, embedding)?;
        }
        Ok(())
    }

    /// Get the number of registered embedding spaces with allocators.
    pub fn allocator_count(&self) -> usize {
        self.id_allocators.len()
    }

    // ========================================================================
    // RaBitQ Encoder Management
    // ========================================================================

    /// Get or create a RaBitQ encoder for the given embedding space.
    ///
    /// Returns None if:
    /// - RaBitQ is disabled in config
    /// - Embedding is not found in registry (can't determine dimension)
    ///
    /// Encoders are cached per embedding space for efficiency.
    pub fn get_or_create_encoder(&self, embedding: EmbeddingCode) -> Option<Arc<RaBitQ>> {
        if !self.rabitq_config.enabled {
            return None;
        }

        // Check cache first
        if let Some(encoder) = self.rabitq_encoders.get(&embedding) {
            return Some(encoder.clone());
        }

        // Look up embedding to get dimension
        let emb = self.registry.get_by_code(embedding)?;
        let dim = emb.dim() as usize;

        // Create and cache encoder
        let encoder = Arc::new(RaBitQ::from_config(dim, &self.rabitq_config));
        self.rabitq_encoders.insert(embedding, encoder.clone());
        Some(encoder)
    }

    /// Check if RaBitQ is enabled.
    pub fn rabitq_enabled(&self) -> bool {
        self.rabitq_config.enabled
    }

    /// Get the RaBitQ configuration.
    pub fn rabitq_config(&self) -> &RaBitQConfig {
        &self.rabitq_config
    }

    /// Get the number of cached RaBitQ encoders.
    pub fn encoder_count(&self) -> usize {
        self.rabitq_encoders.len()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Full integration tests require a RocksDB instance.
    // These unit tests verify the basic structure and logic.

    #[test]
    fn test_processor_allocator_management() {
        // Create a mock scenario without actual storage
        // This tests the DashMap-based allocator management

        let allocators: DashMap<EmbeddingCode, IdAllocator> = DashMap::new();

        // Simulate get_or_create_allocator
        let embedding: EmbeddingCode = 1;
        allocators.entry(embedding).or_insert_with(IdAllocator::new);

        assert!(allocators.contains_key(&1));
        assert!(!allocators.contains_key(&2));

        // Allocate some IDs
        {
            let mut allocator = allocators.get_mut(&1).unwrap();
            let id1 = allocator.allocate();
            let id2 = allocator.allocate();
            assert_eq!(id1, 0);
            assert_eq!(id2, 1);
        }

        // Verify state persists
        {
            let allocator = allocators.get(&1).unwrap();
            assert_eq!(allocator.next_id(), 2);
        }
    }

    #[test]
    fn test_allocator_count() {
        let allocators: DashMap<EmbeddingCode, IdAllocator> = DashMap::new();

        assert_eq!(allocators.len(), 0);

        allocators.insert(1, IdAllocator::new());
        assert_eq!(allocators.len(), 1);

        allocators.insert(2, IdAllocator::new());
        assert_eq!(allocators.len(), 2);

        // Inserting same key doesn't increase count
        allocators.insert(1, IdAllocator::new());
        assert_eq!(allocators.len(), 2);
    }
}
