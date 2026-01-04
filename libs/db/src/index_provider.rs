//! Index provider trait for modular Tantivy index initialization.
//!
//! This module defines the `IndexProvider` trait that enables different
//! index modules (fulltext, future semantic search) to register their
//! Tantivy schemas and initialization logic with StorageBuilder.
//!
//! ## Design Rationale
//!
//! - Parallel to `ColumnFamilyProvider` for RocksDB
//! - Each module provides its own Tantivy schema
//! - Pre-warm isolation: each module's `on_ready()` handles its own initialization
//! - Enables StorageBuilder to compose both RocksDB and Tantivy backends

use anyhow::Result;
use tantivy::Index;

/// Trait for modules that provide Tantivy index schemas.
///
/// Implement this trait to register a Tantivy schema and perform
/// module-specific initialization after the index is opened.
///
/// # Example
///
/// ```ignore
/// use motlie_db::index_provider::IndexProvider;
/// use tantivy::{Index, schema::Schema};
///
/// struct FulltextSchema;
///
/// impl IndexProvider for FulltextSchema {
///     fn name(&self) -> &'static str { "fulltext" }
///
///     fn schema(&self) -> Schema {
///         // Return Tantivy schema definition
///         let mut builder = Schema::builder();
///         builder.add_text_field("content", tantivy::schema::TEXT);
///         builder.build()
///     }
///
///     fn on_ready(&self, index: &Index) -> Result<()> {
///         // Perform initialization after index is opened
///         Ok(())
///     }
/// }
/// ```
pub trait IndexProvider: Send + Sync {
    /// Module name for logging (e.g., "fulltext", "semantic").
    fn name(&self) -> &'static str;

    /// Returns the Tantivy schema for this module.
    ///
    /// Called during index creation to define the document structure.
    fn schema(&self) -> tantivy::schema::Schema;

    /// Called after the index is opened to initialize module-specific state.
    ///
    /// Use this to pre-warm caches, create readers, or perform other
    /// initialization that requires index access.
    ///
    /// # Default
    ///
    /// Default implementation is a no-op, returning `Ok(())`.
    fn on_ready(&self, _index: &Index) -> Result<()> {
        Ok(())
    }

    /// Returns the writer heap size for this index.
    ///
    /// Default: 50MB. Override for different buffer sizes.
    fn writer_heap_size(&self) -> usize {
        50_000_000 // 50MB default
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tantivy::schema::{Schema, TEXT};

    struct MockIndexProvider;

    impl IndexProvider for MockIndexProvider {
        fn name(&self) -> &'static str {
            "mock"
        }

        fn schema(&self) -> Schema {
            let mut builder = Schema::builder();
            builder.add_text_field("test_field", TEXT);
            builder.build()
        }
    }

    #[test]
    fn test_default_on_ready() {
        let provider = MockIndexProvider;
        assert_eq!(provider.name(), "mock");
    }

    #[test]
    fn test_default_writer_heap_size() {
        let provider = MockIndexProvider;
        assert_eq!(provider.writer_heap_size(), 50_000_000);
    }

    #[test]
    fn test_schema_creation() {
        let provider = MockIndexProvider;
        let schema = provider.schema();
        assert!(schema.get_field("test_field").is_ok());
    }
}
