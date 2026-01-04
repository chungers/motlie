//! Fulltext-specific storage infrastructure.
//!
//! Provides Tantivy index storage with readonly/readwrite access patterns.
//!
//! # Example
//!
//! ```ignore
//! use motlie_db::fulltext::Storage;
//!
//! // Read-only access for queries
//! let mut storage = Storage::readonly(index_path);
//! storage.ready()?;
//!
//! // Read-write access for mutations
//! let mut storage = Storage::readwrite(index_path);
//! storage.ready()?;
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tantivy::IndexWriter;

use super::schema::{self, DocumentFields};

// ============================================================================
// Storage Mode
// ============================================================================

/// Storage mode for Tantivy index
#[derive(Clone, Copy, Debug)]
enum StorageMode {
    /// Read-only access - can have multiple instances
    ReadOnly,
    /// Read-write access - exclusive (only one IndexWriter per index)
    ReadWrite,
}

// ============================================================================
// Storage
// ============================================================================

/// Fulltext storage following the same pattern as graph::Storage.
///
/// Supports two modes:
/// - **ReadOnly**: Multiple instances can open the same index for searching.
///   Does not acquire the index lock. Use for query consumers.
/// - **ReadWrite**: Exclusive access with an IndexWriter for mutations.
///   Only one instance can hold the lock at a time. Use for mutation consumers.
///
/// # Example
/// ```no_run
/// use motlie_db::fulltext::Storage;
/// use std::path::Path;
/// use std::sync::Arc;
///
/// // For mutations (exclusive write access)
/// let mut write_storage = Storage::readwrite(Path::new("/data/fulltext"));
/// write_storage.ready().unwrap();
/// let write_storage = Arc::new(write_storage);
///
/// // For queries (multiple readers allowed)
/// let mut read_storage = Storage::readonly(Path::new("/data/fulltext"));
/// read_storage.ready().unwrap();
/// let read_storage = Arc::new(read_storage);
/// ```
pub struct Storage {
    index_path: PathBuf,
    mode: StorageMode,
    /// The Tantivy index (set after ready())
    index: Option<tantivy::Index>,
    /// The index writer - only present in readwrite mode, behind Mutex for thread-safe access
    writer: Option<tokio::sync::Mutex<IndexWriter>>,
    /// Field handles for the schema
    fields: Option<DocumentFields>,
}

impl Storage {
    /// Create a new Storage instance in readonly mode.
    ///
    /// Multiple readonly instances can access the same index simultaneously.
    /// Use this for query consumers.
    pub fn readonly(index_path: &Path) -> Self {
        Self {
            index_path: PathBuf::from(index_path),
            mode: StorageMode::ReadOnly,
            index: None,
            writer: None,
            fields: None,
        }
    }

    /// Create a new Storage instance in readwrite mode.
    ///
    /// Only one readwrite instance can access the index at a time due to
    /// Tantivy's exclusive IndexWriter lock.
    /// Use this for mutation consumers.
    pub fn readwrite(index_path: &Path) -> Self {
        Self {
            index_path: PathBuf::from(index_path),
            mode: StorageMode::ReadWrite,
            index: None,
            writer: None,
            fields: None,
        }
    }

    /// Ready the storage by opening the index.
    ///
    /// For ReadOnly mode: Opens the index without acquiring the writer lock.
    /// For ReadWrite mode: Opens the index and creates an exclusive IndexWriter.
    #[tracing::instrument(skip(self), fields(path = ?self.index_path, mode = ?self.mode))]
    pub fn ready(&mut self) -> Result<()> {
        if self.index.is_some() {
            return Ok(());
        }

        let (tantivy_schema, fields) = schema::build_schema();

        match self.mode {
            StorageMode::ReadOnly => {
                // ReadOnly: Just open the index, no writer needed
                if !self.index_path.exists() {
                    return Err(anyhow::anyhow!(
                        "Index path does not exist: {:?}. Create it first with a readwrite Storage.",
                        self.index_path
                    ));
                }
                let index = tantivy::Index::open_in_dir(&self.index_path)
                    .context("Failed to open Tantivy index in readonly mode")?;

                tracing::info!(
                    path = ?self.index_path,
                    "[FullText Storage] Opened index in READONLY mode"
                );

                self.index = Some(index);
                self.fields = Some(fields);
            }
            StorageMode::ReadWrite => {
                // ReadWrite: Create or open index with exclusive writer
                // Check for meta.json to determine if a valid Tantivy index exists
                let meta_path = self.index_path.join("meta.json");
                let index = if meta_path.exists() {
                    tantivy::Index::open_in_dir(&self.index_path)
                        .context("Failed to open existing Tantivy index")?
                } else {
                    std::fs::create_dir_all(&self.index_path)
                        .context("Failed to create index directory")?;
                    tantivy::Index::create_in_dir(&self.index_path, tantivy_schema)
                        .context("Failed to create Tantivy index")?
                };

                // Create index writer with 50MB buffer (acquires exclusive lock)
                let writer = index
                    .writer(50_000_000)
                    .context("Failed to create index writer")?;

                tracing::info!(
                    path = ?self.index_path,
                    "[FullText Storage] Opened index in READWRITE mode"
                );

                self.index = Some(index);
                self.writer = Some(tokio::sync::Mutex::new(writer));
                self.fields = Some(fields);
            }
        }

        Ok(())
    }

    /// Get a reference to the Tantivy index for searching.
    pub fn index(&self) -> Result<&tantivy::Index> {
        self.index
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("[FullText Storage] Not ready"))
    }

    /// Get the field handles.
    pub fn fields(&self) -> Result<&DocumentFields> {
        self.fields
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("[FullText Storage] Not ready"))
    }

    /// Get the index path.
    pub fn index_path(&self) -> &Path {
        &self.index_path
    }

    /// Check if this storage is in readwrite mode.
    pub fn is_readwrite(&self) -> bool {
        matches!(self.mode, StorageMode::ReadWrite)
    }

    /// Get a reference to the index writer (only available in readwrite mode).
    ///
    /// Returns None if in readonly mode.
    pub(crate) fn writer(&self) -> Option<&tokio::sync::Mutex<IndexWriter>> {
        self.writer.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_storage_readonly_create() {
        let storage = Storage::readonly(Path::new("/tmp/test"));
        assert!(!storage.is_readwrite());
    }

    #[test]
    fn test_storage_readwrite_create() {
        let storage = Storage::readwrite(Path::new("/tmp/test"));
        assert!(storage.is_readwrite());
    }

    #[tokio::test]
    async fn test_storage_readwrite_ready() {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("fulltext_index");

        let mut storage = Storage::readwrite(&index_path);
        storage.ready().expect("Failed to initialize storage");

        assert!(storage.index().is_ok());
        assert!(storage.fields().is_ok());
        assert!(storage.writer().is_some());
    }

    #[tokio::test]
    async fn test_storage_readonly_fails_without_existing_index() {
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("nonexistent");

        let mut storage = Storage::readonly(&index_path);
        let result = storage.ready();

        assert!(result.is_err());
    }
}
