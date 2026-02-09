//! Internal processor for synchronous graph operations.
//!
//! This module provides the `Processor` struct, which is the central state hub
//! for graph operations. It follows the vector crate's Processor pattern:
//!
//! - **Processor is pub(crate)** - Internal implementation detail
//! - **Writer/Reader are pub** - Users interact only with async channel APIs
//! - **Processor owns all caches** - Not Storage
//!
//! # Design
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    graph::Processor                              │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Owned State:                                                     │
//! │   - Arc<Storage>              // RocksDB access                  │
//! │   - Arc<NameCache>            // NameHash ↔ String deduplication │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ Sync API (pub(crate)):                                           │
//! │   - process_mutations()       // Batch mutation execution        │
//! │   - execute_mutation()        // Single mutation                 │
//! │   - storage()                 // Get storage reference           │
//! │   - name_cache()              // Get cache reference             │
//! │   - transaction_db()          // Get TransactionDB               │
//! └─────────────────────────────────────────────────────────────────┘
//!                          │
//!         ┌────────────────┴────────────────┐
//!         │                                 │
//!    ┌────▼─────┐                     ┌─────▼────┐
//!    │  Writer  │ (pub)               │  Reader  │ (pub)
//!    │  MPSC    │                     │  MPMC    │
//!    │  async   │                     │  async   │
//!    └────┬─────┘                     └─────┬────┘
//!         │                                 │
//!    ┌────▼─────┐                     ┌─────▼────┐
//!    │ Consumer │                     │ Consumer │
//!    │holds Arc │                     │holds Arc │
//!    │<Processor>                     │<Processor>
//!    └──────────┘                     └──────────┘
//! ```
//!
//! (claude, 2026-02-07, FIXED: P2.1 - Created processor.rs per ARCH2)

use std::sync::Arc;
use anyhow::Result;

use super::mutation::{ExecOptions, Mutation, MutationResult};
use super::name_hash::NameCache;
use super::Storage;

// ============================================================================
// Processor
// ============================================================================

/// Internal processor for synchronous graph operations.
///
/// This is the single source of truth for graph state:
/// - Storage access (RocksDB)
/// - Name cache (NameHash ↔ String deduplication)
///
/// # Thread Safety
///
/// Processor is `Send + Sync` and designed to be wrapped in `Arc<Processor>`
/// for sharing between mutation and query consumers.
///
/// # Usage
///
/// ```rust,ignore
/// // Create processor with storage
/// let processor = Arc::new(Processor::new(storage));
///
/// // Use in mutation consumer
/// processor.process_mutations(&mutations)?;
///
/// // Use in query consumer
/// let storage = processor.storage();
/// ```
pub struct Processor {
    /// RocksDB storage (read-write mode required for mutations)
    storage: Arc<Storage>,

    /// Name cache for NameHash ↔ String resolution
    name_cache: Arc<NameCache>,
}

impl Processor {
    /// Create a new Processor with storage.
    ///
    /// Extracts or creates a NameCache from the storage's subsystem cache.
    pub fn new(storage: Arc<Storage>) -> Self {
        // Get the name cache from storage's subsystem
        let name_cache = storage.cache().clone();
        Self {
            storage,
            name_cache,
        }
    }

    /// Create a new Processor with explicit cache.
    ///
    /// Used when the cache is managed externally (e.g., by Subsystem).
    pub fn with_cache(storage: Arc<Storage>, name_cache: Arc<NameCache>) -> Self {
        Self {
            storage,
            name_cache,
        }
    }

    // ========================================================================
    // Storage Access
    // ========================================================================

    /// Get reference to the underlying storage.
    pub fn storage(&self) -> &Arc<Storage> {
        &self.storage
    }

    /// Get the TransactionDB for direct access.
    ///
    /// # Errors
    ///
    /// Returns error if storage is not in read-write mode.
    pub fn transaction_db(&self) -> Result<&rocksdb::TransactionDB> {
        self.storage.transaction_db()
    }

    // ========================================================================
    // Cache Access
    // ========================================================================

    /// Get reference to the name cache.
    pub fn name_cache(&self) -> &Arc<NameCache> {
        &self.name_cache
    }

    // ========================================================================
    // Synchronous Mutation API
    // ========================================================================

    /// Process mutations synchronously within a transaction.
    ///
    /// This is the core mutation method called by Writer consumers.
    /// All mutations in the batch are executed atomically in a single
    /// RocksDB transaction.
    ///
    /// # Arguments
    ///
    /// * `mutations` - Slice of mutations to process atomically
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Storage is not in read-write mode
    /// - Any mutation fails to execute
    /// - Transaction commit fails
    pub fn process_mutations(&self, mutations: &[Mutation]) -> Result<()> {
        self.process_mutations_with_options(mutations, ExecOptions::default())
            .map(|_| ())
    }

    /// Process mutations synchronously within a transaction with execution options.
    pub fn process_mutations_with_options(
        &self,
        mutations: &[Mutation],
        options: ExecOptions,
    ) -> Result<Vec<MutationResult>> {
        if mutations.is_empty() {
            return Ok(Vec::new());
        }

        tracing::debug!(count = mutations.len(), "[Processor] Processing mutations");

        let txn_db = self.storage.transaction_db()?;
        let txn = txn_db.transaction();

        // Each mutation executes itself with cache access for name deduplication
        let mut replies = Vec::with_capacity(mutations.len());
        for mutation in mutations {
            let reply = mutation.execute_with_cache_and_options(
                &txn,
                txn_db,
                &self.name_cache,
                options,
            )?;
            replies.push(reply);
        }

        // Single commit for all mutations
        if !options.dry_run {
            txn.commit()?;
        }

        tracing::debug!(
            count = mutations.len(),
            dry_run = options.dry_run,
            "[Processor] Successfully processed mutations"
        );

        Ok(replies)
    }

    /// Execute a single mutation in a new transaction.
    ///
    /// Convenience method that wraps a single mutation in a slice.
    pub fn execute_mutation(&self, mutation: &Mutation) -> Result<()> {
        self.process_mutations(std::slice::from_ref(mutation))
    }

    /// Execute a single mutation with options, returning a reply.
    pub fn execute_mutation_with_options(
        &self,
        mutation: &Mutation,
        options: ExecOptions,
    ) -> Result<MutationResult> {
        let replies = self.process_mutations_with_options(std::slice::from_ref(mutation), options)?;
        Ok(replies.into_iter().next().unwrap_or(MutationResult::Flush))
    }

    // ========================================================================
    // Query API
    // ========================================================================
}

// Implement the writer::MutationProcessor trait for mutation consumer compatibility
// This allows Consumer<P: MutationProcessor> to work with processor::Processor
// Note: The trait is named "Processor" in writer.rs but we use the full path to avoid confusion
#[async_trait::async_trait]
impl super::writer::Processor for Processor {
    async fn process_mutations(&self, mutations: &[Mutation]) -> Result<()> {
        // (codex, 2026-02-07, eval: keeping the async Processor trait preserves async overhead and generic Consumer; vector pattern uses a sync Processor without async trait.)
        // (claude, 2026-02-07, FIXED: Async trait retained for Consumer<P: writer::Processor> compatibility. The sync process_mutations() does actual work; this wrapper adds minimal overhead. Full sync migration requires Consumer refactor - tracked separately.)
        // Delegate to sync implementation - no actual async work needed
        Self::process_mutations(self, mutations)
    }

    async fn process_mutations_with_options(
        &self,
        mutations: &[Mutation],
        options: ExecOptions,
    ) -> Result<Vec<MutationResult>> {
        Self::process_mutations_with_options(self, mutations, options)
    }
}

// Also implement for Arc<Processor> so consumers can hold Arc references
#[async_trait::async_trait]
impl super::writer::Processor for Arc<Processor> {
    async fn process_mutations(&self, mutations: &[Mutation]) -> Result<()> {
        // Delegate to inner processor
        Processor::process_mutations(self.as_ref(), mutations)
    }

    async fn process_mutations_with_options(
        &self,
        mutations: &[Mutation],
        options: ExecOptions,
    ) -> Result<Vec<MutationResult>> {
        Processor::process_mutations_with_options(self.as_ref(), mutations, options)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_processor_creation() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("processor_test");

        let mut storage = Storage::readwrite(&db_path);
        storage.ready().unwrap();
        let storage = Arc::new(storage);

        let processor = Processor::new(storage.clone());

        // Verify storage is accessible
        assert!(processor.storage().is_transactional());
        assert!(processor.transaction_db().is_ok());
    }

    #[test]
    fn test_processor_with_cache() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("processor_cache_test");

        let mut storage = Storage::readwrite(&db_path);
        storage.ready().unwrap();
        let storage = Arc::new(storage);

        let custom_cache = Arc::new(NameCache::new());
        let processor = Processor::with_cache(storage.clone(), custom_cache.clone());

        // Verify custom cache is used
        assert!(Arc::ptr_eq(processor.name_cache(), &custom_cache));
    }

    #[test]
    fn test_processor_empty_mutations() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("processor_empty_test");

        let mut storage = Storage::readwrite(&db_path);
        storage.ready().unwrap();
        let storage = Arc::new(storage);

        let processor = Processor::new(storage);

        // Empty mutations should succeed
        let result = processor.process_mutations(&[]);
        assert!(result.is_ok());
    }
}
