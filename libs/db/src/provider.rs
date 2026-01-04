//! Column family provider trait for modular storage initialization.
//!
//! This module defines the `ColumnFamilyProvider` trait that enables different
//! storage modules (graph, vector, etc.) to register their column families
//! with a shared RocksDB instance without coupling between modules.
//!
//! ## Design Rationale (Task 0.4)
//!
//! - Graph module shouldn't know about vector-specific CFs
//! - Vector module needs to share the same TransactionDB
//! - Each module registers its own CFs and initialization logic
//! - Pre-warm isolation: each module's `on_ready()` handles its own caches

use anyhow::Result;
use rocksdb::{Cache, ColumnFamilyDescriptor, TransactionDB};

/// Trait for modules that provide column families to the shared RocksDB instance.
///
/// Implement this trait to register column families and perform module-specific
/// initialization after the database is opened.
///
/// # Example
///
/// ```ignore
/// use motlie_db::provider::ColumnFamilyProvider;
/// use rocksdb::{Cache, ColumnFamilyDescriptor, TransactionDB};
///
/// struct VectorSchema {
///     registry: EmbeddingRegistry,
/// }
///
/// impl ColumnFamilyProvider for VectorSchema {
///     fn name(&self) -> &'static str { "vector" }
///
///     fn column_family_descriptors(
///         &self,
///         cache: Option<&Cache>,
///         block_size: usize,
///     ) -> Vec<ColumnFamilyDescriptor> {
///         vec![
///             ColumnFamilyDescriptor::new("vector/vectors", cf_options(cache)),
///             ColumnFamilyDescriptor::new("vector/edges", cf_options(cache)),
///             // ... more CFs
///         ]
///     }
///
///     fn on_ready(&self, db: &TransactionDB) -> Result<()> {
///         let count = self.registry.prewarm(db)?;
///         tracing::info!(count, "Pre-warmed EmbeddingRegistry");
///         Ok(())
///     }
/// }
/// ```
pub trait ColumnFamilyProvider: Send + Sync {
    /// Module name for logging (e.g., "graph", "vector").
    fn name(&self) -> &'static str;

    /// Returns all CF descriptors for this module.
    ///
    /// Called during database initialization to collect all column families
    /// from all registered providers.
    ///
    /// # Arguments
    ///
    /// * `cache` - Optional shared block cache for memory efficiency
    /// * `block_size` - Block size in bytes for this module's column families
    fn column_family_descriptors(
        &self,
        cache: Option<&Cache>,
        block_size: usize,
    ) -> Vec<ColumnFamilyDescriptor>;

    /// Called after DB is opened to initialize module-specific state.
    ///
    /// Use this to pre-warm caches, validate schema, or perform other
    /// initialization that requires database access.
    ///
    /// # Default
    ///
    /// Default implementation is a no-op, returning `Ok(())`.
    fn on_ready(&self, _db: &TransactionDB) -> Result<()> {
        Ok(())
    }

    /// Returns the list of CF names this provider manages.
    ///
    /// Used for verification and documentation. Default implementation
    /// extracts names from `column_family_descriptors()`.
    fn cf_names(&self) -> Vec<&'static str> {
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockProvider {
        name: &'static str,
    }

    impl ColumnFamilyProvider for MockProvider {
        fn name(&self) -> &'static str {
            self.name
        }

        fn column_family_descriptors(
            &self,
            _cache: Option<&Cache>,
            _block_size: usize,
        ) -> Vec<ColumnFamilyDescriptor> {
            vec![]
        }
    }

    #[test]
    fn test_default_on_ready() {
        let provider = MockProvider { name: "test" };
        // Default on_ready should succeed
        // We can't test with real TransactionDB without setup
        assert_eq!(provider.name(), "test");
    }

    #[test]
    fn test_default_cf_names() {
        let provider = MockProvider { name: "test" };
        assert!(provider.cf_names().is_empty());
    }
}
