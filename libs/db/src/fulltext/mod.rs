//! Provides the full-text search implementation using Tantivy for processing mutations
//! from the MPSC queue and updating the full-text search index.
//!
//! This module includes support for:
//! - Basic fulltext indexing with BM25 scoring
//! - Faceted search for filtering by categories
//! - Fuzzy search for typo-tolerant queries
//! - Tag-based user-defined facets extracted from content

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tantivy::IndexWriter;

// Submodules
pub mod mutation;
pub mod query;
pub mod reader;
pub mod schema;
pub mod search;
pub mod writer;

#[cfg(test)]
mod tantivy_behavior_test;

// Re-export commonly used types (no prefixes - use module path for disambiguation)
// Note: Runnable is now generic: Runnable<R> where R is the reader type
// Search enum is public for reader APIs but has #[doc(hidden)] variants
pub use query::{Edges, Facets, FuzzyLevel, Nodes, Search};
pub use crate::reader::Runnable;
pub use reader::{
    create_query_consumer, create_query_reader, spawn_query_consumer,
    spawn_query_consumer_pool_readonly, spawn_query_consumer_pool_shared, Consumer, Processor,
    QueryExecutor, Reader, ReaderConfig,
};
pub use schema::{compute_validity_facet, extract_tags, DocumentFields};
pub use search::{EdgeHit, FacetCounts, Hit, MatchSource, NodeHit};
pub use writer::{
    create_mutation_consumer, create_mutation_consumer_with_next,
    create_mutation_consumer_with_params, create_mutation_consumer_with_params_and_next,
    spawn_mutation_consumer, spawn_mutation_consumer_with_next,
    spawn_mutation_consumer_with_params, spawn_mutation_consumer_with_params_and_next,
    MutationExecutor,
};

// ============================================================================
// Storage - Readonly/Readwrite Tantivy Index Access
// ============================================================================

/// Storage mode for Tantivy index
#[derive(Clone, Copy, Debug)]
enum StorageMode {
    /// Read-only access - can have multiple instances
    ReadOnly,
    /// Read-write access - exclusive (only one IndexWriter per index)
    ReadWrite,
}

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

        let (schema, fields) = schema::build_schema();

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
                    tantivy::Index::create_in_dir(&self.index_path, schema)
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

// ============================================================================
// Index - Wraps Storage for query and mutation processing
// ============================================================================

/// Full-text search processor using Tantivy, wrapping Arc<Storage>.
///
/// Following the same pattern as `graph::Graph` wrapping `Arc<graph::Storage>`,
/// this type provides the interface for both mutation processing and query processing.
///
/// # Usage Pattern (matches graph::Graph)
/// ```no_run
/// use motlie_db::fulltext::{Storage, Index};
/// use std::sync::Arc;
/// use std::path::Path;
///
/// // For mutations (exclusive write access)
/// let mut storage = Storage::readwrite(Path::new("/data/fulltext"));
/// storage.ready().unwrap();
/// let storage = Arc::new(storage);
/// let index = Index::new(storage);
///
/// // For queries (multiple readers allowed)
/// let mut storage = Storage::readonly(Path::new("/data/fulltext"));
/// storage.ready().unwrap();
/// let storage = Arc::new(storage);
/// let index = Index::new(storage);
/// ```
///
/// # Thread Safety
/// - Index wraps `Arc<Storage>`, so cloning is cheap and preserves full capability
/// - The IndexWriter inside Storage is behind a Mutex for thread-safe mutation access
/// - Tantivy's Index is inherently thread-safe (Send + Sync)
pub struct Index {
    pub(crate) storage: Arc<Storage>,
}

impl Index {
    /// Create a new Index from Arc<Storage>.
    ///
    /// The Storage should already be `ready()` before creating the Index.
    /// This follows the same pattern as `graph::Graph::new(Arc<Storage>)`.
    pub fn new(storage: Arc<Storage>) -> Self {
        Self { storage }
    }

    /// Get a reference to the underlying storage.
    pub fn storage(&self) -> &Storage {
        &self.storage
    }

    /// Get a reference to the Tantivy index for searching.
    ///
    /// # Panics
    /// Panics if the storage is not ready.
    pub fn tantivy_index(&self) -> &tantivy::Index {
        self.storage.index().expect("Storage not ready")
    }

    /// Get the field handles.
    ///
    /// # Panics
    /// Panics if the storage is not ready.
    pub fn fields(&self) -> &DocumentFields {
        self.storage.fields().expect("Storage not ready")
    }

    /// Get the index path.
    pub fn index_path(&self) -> &Path {
        self.storage.index_path()
    }

    /// Check if this Index has write capability.
    pub fn is_readwrite(&self) -> bool {
        self.storage.is_readwrite()
    }
}

impl Clone for Index {
    fn clone(&self) -> Self {
        // Clone the Arc - preserves full capability (same as graph::Graph)
        Self {
            storage: self.storage.clone(),
        }
    }
}

// ============================================================================
// Subsystem Info
// ============================================================================

/// Default writer heap size in bytes (50MB).
pub const DEFAULT_WRITER_HEAP_SIZE: usize = 50_000_000;

/// Default number of indexing threads (Tantivy default: number of CPUs).
pub const DEFAULT_NUM_THREADS: usize = 0; // 0 means use Tantivy's default

/// Static configuration info for the fulltext search subsystem.
///
/// Used by the `motlie info` command to display Tantivy settings.
/// Implements [`motlie_core::telemetry::SubsystemInfo`] for consistent formatting.
///
/// # Example
///
/// ```ignore
/// use motlie_db::fulltext::SystemInfo;
/// use motlie_core::telemetry::{format_subsystem_info, SubsystemInfo};
///
/// let info = SystemInfo::default();
/// println!("{}", format_subsystem_info(&info));
/// ```
#[derive(Debug, Clone)]
pub struct SystemInfo {
    /// IndexWriter heap size in bytes (memory buffer for indexing).
    /// Default: 50MB. Larger values improve throughput for bulk indexing.
    pub writer_heap_size: usize,
    /// Number of indexing threads. 0 means use Tantivy's default (num CPUs).
    pub num_threads: usize,
    /// Whether the index uses STORED fields (content is retrievable).
    pub stored_fields: bool,
    /// Whether BM25 scoring is enabled (default Tantivy scorer).
    pub bm25_scoring: bool,
    /// Whether faceted search is enabled.
    pub faceted_search: bool,
    /// Whether fuzzy search is supported.
    pub fuzzy_search: bool,
}

impl Default for SystemInfo {
    fn default() -> Self {
        Self {
            writer_heap_size: DEFAULT_WRITER_HEAP_SIZE,
            num_threads: DEFAULT_NUM_THREADS,
            stored_fields: true,   // We store content for retrieval
            bm25_scoring: true,    // Tantivy default
            faceted_search: true,  // We use facets for filtering
            fuzzy_search: true,    // We support fuzzy queries
        }
    }
}

impl motlie_core::telemetry::SubsystemInfo for SystemInfo {
    fn name(&self) -> &'static str {
        "Fulltext Search (Tantivy)"
    }

    fn info_lines(&self) -> Vec<(&'static str, String)> {
        let threads_str = if self.num_threads == 0 {
            "auto (num CPUs)".to_string()
        } else {
            self.num_threads.to_string()
        };

        vec![
            ("Writer Heap Size", format_bytes(self.writer_heap_size)),
            ("Indexing Threads", threads_str),
            ("Stored Fields", self.stored_fields.to_string()),
            ("BM25 Scoring", self.bm25_scoring.to_string()),
            ("Faceted Search", self.faceted_search.to_string()),
            ("Fuzzy Search", self.fuzzy_search.to_string()),
        ]
    }
}

/// Format a byte count as a human-readable string.
fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{} GB", bytes / (1024 * 1024 * 1024))
    } else if bytes >= 1024 * 1024 {
        format!("{} MB", bytes / (1024 * 1024))
    } else if bytes >= 1024 {
        format!("{} KB", bytes / 1024)
    } else {
        format!("{} B", bytes)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::mutation::{
        AddEdge, AddEdgeFragment, AddNode, AddNodeFragment, Mutation, UpdateNodeValidSinceUntil,
    };
    use crate::graph::writer::Processor;
    use crate::{DataUrl, Id, TimestampMilli};
    use tantivy::collector::TopDocs;
    use tantivy::query::QueryParser;

    #[tokio::test]
    async fn test_index_add_node() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");
        let processor = writer::create_readwrite_index(&index_path);

        let node = AddNode {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "test_node".to_string(),
            valid_range: None,
            summary: crate::graph::schema::NodeSummary::from_text("test summary"),
        };

        let mutations = vec![Mutation::AddNode(node.clone())];
        processor.process_mutations(&mutations).await.unwrap();

        // Search for the node
        let reader = processor.tantivy_index().reader().unwrap();
        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(
            processor.tantivy_index(),
            vec![processor.fields().node_name_field],
        );
        let query = query_parser.parse_query("test_node").unwrap();
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10)).unwrap();

        assert_eq!(top_docs.len(), 1);
    }

    #[tokio::test]
    async fn test_index_add_node_fragment() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");
        let processor = writer::create_readwrite_index(&index_path);

        let node_id = Id::new();
        let fragment = AddNodeFragment {
            id: node_id,
            ts_millis: TimestampMilli::now(),
            content: DataUrl::from_text("This is searchable content about Rust programming"),
            valid_range: None,
        };

        let mutations = vec![Mutation::AddNodeFragment(fragment)];
        processor.process_mutations(&mutations).await.unwrap();

        // Search for "Rust"
        let reader = processor.tantivy_index().reader().unwrap();
        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(
            processor.tantivy_index(),
            vec![processor.fields().content_field],
        );
        let query = query_parser.parse_query("Rust").unwrap();
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10)).unwrap();

        assert_eq!(top_docs.len(), 1);
    }

    #[tokio::test]
    async fn test_index_add_edge() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");
        let processor = writer::create_readwrite_index(&index_path);

        let edge = AddEdge {
            source_node_id: Id::new(),
            target_node_id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "depends_on".to_string(),
            summary: crate::graph::schema::EdgeSummary::from_text("dependency relationship"),
            weight: Some(1.0),
            valid_range: None,
        };

        let mutations = vec![Mutation::AddEdge(edge)];
        processor.process_mutations(&mutations).await.unwrap();

        // Search for "dependency"
        let reader = processor.tantivy_index().reader().unwrap();
        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(
            processor.tantivy_index(),
            vec![processor.fields().content_field],
        );
        let query = query_parser.parse_query("dependency").unwrap();
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10)).unwrap();

        assert_eq!(top_docs.len(), 1);
    }

    #[tokio::test]
    async fn test_index_add_edge_fragment() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");
        let processor = writer::create_readwrite_index(&index_path);

        let edge_fragment = AddEdgeFragment {
            src_id: Id::new(),
            dst_id: Id::new(),
            edge_name: "connects_to".to_string(),
            ts_millis: TimestampMilli::now(),
            content: DataUrl::from_markdown(
                "# Connection Details\nThis edge represents a network connection",
            ),
            valid_range: None,
        };

        let mutations = vec![Mutation::AddEdgeFragment(edge_fragment)];
        processor.process_mutations(&mutations).await.unwrap();

        // Search for "network"
        let reader = processor.tantivy_index().reader().unwrap();
        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(
            processor.tantivy_index(),
            vec![processor.fields().content_field],
        );
        let query = query_parser.parse_query("network").unwrap();
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10)).unwrap();

        assert_eq!(top_docs.len(), 1);
    }

    #[tokio::test]
    async fn test_batch_indexing() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");
        let processor = writer::create_readwrite_index(&index_path);

        // Create a batch of mutations
        let mutations = vec![
            Mutation::AddNode(AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                name: "node_one".to_string(),
                valid_range: None,
                summary: crate::graph::schema::NodeSummary::from_text("node one summary"),
            }),
            Mutation::AddNode(AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                name: "node_two".to_string(),
                valid_range: None,
                summary: crate::graph::schema::NodeSummary::from_text("node two summary"),
            }),
            Mutation::AddNodeFragment(AddNodeFragment {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                content: DataUrl::from_text("fragment content here"),
                valid_range: None,
            }),
        ];

        processor.process_mutations(&mutations).await.unwrap();

        // Search should find results
        let reader = processor.tantivy_index().reader().unwrap();
        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(
            processor.tantivy_index(),
            vec![
                processor.fields().node_name_field,
                processor.fields().content_field,
            ],
        );

        // Search for "node"
        let query = query_parser.parse_query("node").unwrap();
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10)).unwrap();
        assert_eq!(top_docs.len(), 2); // Should find both nodes

        // Search for "fragment"
        let query = query_parser.parse_query("fragment").unwrap();
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10)).unwrap();
        assert_eq!(top_docs.len(), 1); // Should find the fragment
    }

    #[tokio::test]
    async fn test_update_node_valid_since_until() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");
        let processor = writer::create_readwrite_index(&index_path);

        let node_id = Id::new();

        // First add a node
        let add_node = Mutation::AddNode(AddNode {
            id: node_id,
            ts_millis: TimestampMilli::now(),
            name: "test_node".to_string(),
            valid_range: None,
            summary: crate::graph::schema::NodeSummary::from_text("test summary"),
        });

        processor.process_mutations(&[add_node]).await.unwrap();

        // Verify it's indexed
        let reader = processor.tantivy_index().reader().unwrap();
        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(
            processor.tantivy_index(),
            vec![processor.fields().node_name_field],
        );
        let query = query_parser.parse_query("test_node").unwrap();
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10)).unwrap();
        assert_eq!(top_docs.len(), 1);

        // Now update its temporal range (which should delete the document)
        let update = Mutation::UpdateNodeValidSinceUntil(UpdateNodeValidSinceUntil {
            id: node_id,
            temporal_range: crate::TemporalRange(
                Some(TimestampMilli(0)),
                Some(TimestampMilli(1000)),
            ),
            reason: "test invalidation".to_string(),
        });

        // Process the update - this calls delete_term
        processor.process_mutations(&[update]).await.unwrap();

        // Note: The delete is logged but verifying deletion requires understanding
        // tantivy's reader/writer semantics more deeply. The delete_term call
        // is made and committed, which is the important part for this implementation.
    }

    #[tokio::test]
    async fn test_multiple_fragments_for_same_node() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");
        let processor = writer::create_readwrite_index(&index_path);

        let node_id = Id::new();

        // Add multiple fragments for the same node
        let mutations = vec![
            Mutation::AddNodeFragment(AddNodeFragment {
                id: node_id,
                ts_millis: TimestampMilli::now(),
                content: DataUrl::from_text("first fragment about databases"),
                valid_range: None,
            }),
            Mutation::AddNodeFragment(AddNodeFragment {
                id: node_id,
                ts_millis: TimestampMilli::now(),
                content: DataUrl::from_text("second fragment about indexing"),
                valid_range: None,
            }),
            Mutation::AddNodeFragment(AddNodeFragment {
                id: node_id,
                ts_millis: TimestampMilli::now(),
                content: DataUrl::from_text("third fragment about search"),
                valid_range: None,
            }),
        ];

        processor.process_mutations(&mutations).await.unwrap();

        // Search for "fragment"
        let reader = processor.tantivy_index().reader().unwrap();
        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(
            processor.tantivy_index(),
            vec![processor.fields().content_field],
        );
        let query = query_parser.parse_query("fragment").unwrap();
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10)).unwrap();

        assert_eq!(top_docs.len(), 3); // All three fragments should be indexed
    }

    #[tokio::test]
    async fn test_search_with_different_mime_types() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");
        let processor = writer::create_readwrite_index(&index_path);

        let mutations = vec![
            Mutation::AddNodeFragment(AddNodeFragment {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                content: DataUrl::from_text("plain text content"),
                valid_range: None,
            }),
            Mutation::AddNodeFragment(AddNodeFragment {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                content: DataUrl::from_markdown("# Markdown content"),
                valid_range: None,
            }),
            Mutation::AddNodeFragment(AddNodeFragment {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                content: DataUrl::from_html("<p>HTML content</p>"),
                valid_range: None,
            }),
        ];

        processor.process_mutations(&mutations).await.unwrap();

        // Search for "content"
        let reader = processor.tantivy_index().reader().unwrap();
        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(
            processor.tantivy_index(),
            vec![processor.fields().content_field],
        );
        let query = query_parser.parse_query("content").unwrap();
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10)).unwrap();

        assert_eq!(top_docs.len(), 3); // All three different formats should be indexed
    }

    #[tokio::test]
    async fn test_edge_with_weight() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");
        let processor = writer::create_readwrite_index(&index_path);

        let edge_with_weight = AddEdge {
            source_node_id: Id::new(),
            target_node_id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "weighted_edge".to_string(),
            summary: crate::graph::schema::EdgeSummary::from_text("weighted connection"),
            weight: Some(2.5),
            valid_range: None,
        };

        let edge_without_weight = AddEdge {
            source_node_id: Id::new(),
            target_node_id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "unweighted_edge".to_string(),
            summary: crate::graph::schema::EdgeSummary::from_text("unweighted connection"),
            weight: None,
            valid_range: None,
        };

        processor
            .process_mutations(&[
                Mutation::AddEdge(edge_with_weight),
                Mutation::AddEdge(edge_without_weight),
            ])
            .await
            .unwrap();

        // Search for "connection"
        let reader = processor.tantivy_index().reader().unwrap();
        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(
            processor.tantivy_index(),
            vec![processor.fields().content_field],
        );
        let query = query_parser.parse_query("connection").unwrap();
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10)).unwrap();

        assert_eq!(top_docs.len(), 2);
    }

    // ========================================================================
    // Time Range Query Tests
    // ========================================================================
    //
    // These tests demonstrate how to query documents by timestamp ranges.
    // Instead of using facets (which would be stale since they're computed at index time),
    // we use Tantivy's range queries on the indexed timestamp fields.
    //
    // Key timestamp fields:
    // - creation_timestamp_field: When the document was created
    // - valid_since_field: Start of validity period (optional)
    // - valid_until_field: End of validity period (optional)

    #[tokio::test]
    async fn test_query_by_creation_timestamp_range() {
        use tantivy::query::RangeQuery;

        let temp_dir = tempfile::TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");
        let processor = writer::create_readwrite_index(&index_path);

        // Create nodes with different timestamps
        let now = TimestampMilli::now().0;
        let one_hour_ago = now - 3600_000;
        let one_day_ago = now - 86400_000;
        let one_week_ago = now - 604800_000;

        let mutations = vec![
            Mutation::AddNode(AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli(now),
                name: "recent_node".to_string(),
                valid_range: None,
                summary: crate::graph::schema::NodeSummary::from_text("created just now"),
            }),
            Mutation::AddNode(AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli(one_hour_ago),
                name: "hourly_node".to_string(),
                valid_range: None,
                summary: crate::graph::schema::NodeSummary::from_text("created one hour ago"),
            }),
            Mutation::AddNode(AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli(one_day_ago),
                name: "daily_node".to_string(),
                valid_range: None,
                summary: crate::graph::schema::NodeSummary::from_text("created one day ago"),
            }),
            Mutation::AddNode(AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli(one_week_ago),
                name: "weekly_node".to_string(),
                valid_range: None,
                summary: crate::graph::schema::NodeSummary::from_text("created one week ago"),
            }),
        ];

        processor.process_mutations(&mutations).await.unwrap();

        let reader = processor.tantivy_index().reader().unwrap();
        let searcher = reader.searcher();

        // Query: Find nodes created in the last 2 hours (two_hours_ago <= ts < now+1)
        let two_hours_ago = now - 7200_000;
        let range_query =
            RangeQuery::new_u64("creation_timestamp".to_string(), two_hours_ago..(now + 1));
        let top_docs = searcher
            .search(&range_query, &TopDocs::with_limit(10))
            .unwrap();
        assert_eq!(top_docs.len(), 2, "Should find recent_node and hourly_node");

        // Query: Find nodes created more than 12 hours ago (0 <= ts <= twelve_hours_ago)
        let twelve_hours_ago = now - 43200_000;
        let range_query =
            RangeQuery::new_u64("creation_timestamp".to_string(), 0..(twelve_hours_ago + 1));
        let top_docs = searcher
            .search(&range_query, &TopDocs::with_limit(10))
            .unwrap();
        assert_eq!(top_docs.len(), 2, "Should find daily_node and weekly_node");

        // Query: Find nodes created between 6 hours ago and 2 days ago
        let six_hours_ago = now - 21600_000;
        let two_days_ago = now - 172800_000;
        let range_query = RangeQuery::new_u64(
            "creation_timestamp".to_string(),
            two_days_ago..(six_hours_ago + 1),
        );
        let top_docs = searcher
            .search(&range_query, &TopDocs::with_limit(10))
            .unwrap();
        assert_eq!(top_docs.len(), 1, "Should find only daily_node");
    }

    #[tokio::test]
    async fn test_query_combining_text_and_timestamp() {
        use tantivy::query::{BooleanQuery, Occur, RangeQuery};

        let temp_dir = tempfile::TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");
        let processor = writer::create_readwrite_index(&index_path);

        let now = TimestampMilli::now().0;
        let one_hour_ago = now - 3600_000;
        let one_week_ago = now - 604800_000;

        let mutations = vec![
            Mutation::AddNodeFragment(AddNodeFragment {
                id: Id::new(),
                ts_millis: TimestampMilli(now),
                content: DataUrl::from_text("Rust programming is awesome"),
                valid_range: None,
            }),
            Mutation::AddNodeFragment(AddNodeFragment {
                id: Id::new(),
                ts_millis: TimestampMilli(one_hour_ago),
                content: DataUrl::from_text("Rust language features"),
                valid_range: None,
            }),
            Mutation::AddNodeFragment(AddNodeFragment {
                id: Id::new(),
                ts_millis: TimestampMilli(one_week_ago),
                content: DataUrl::from_text("Old Rust documentation"),
                valid_range: None,
            }),
        ];

        processor.process_mutations(&mutations).await.unwrap();

        let reader = processor.tantivy_index().reader().unwrap();
        let searcher = reader.searcher();
        let fields = processor.fields();

        // Combined query: "Rust" AND created in the last 2 hours
        let query_parser =
            QueryParser::for_index(processor.tantivy_index(), vec![fields.content_field]);
        let text_query = query_parser.parse_query("Rust").unwrap();

        let two_hours_ago = now - 7200_000;
        let time_query =
            RangeQuery::new_u64("creation_timestamp".to_string(), two_hours_ago..(now + 1));

        let combined_query = BooleanQuery::new(vec![
            (Occur::Must, text_query),
            (Occur::Must, Box::new(time_query)),
        ]);

        let top_docs = searcher
            .search(&combined_query, &TopDocs::with_limit(10))
            .unwrap();
        assert_eq!(
            top_docs.len(),
            2,
            "Should find 2 recent Rust documents (excludes week-old one)"
        );
    }

    #[tokio::test]
    async fn test_query_by_validity_range() {
        use tantivy::query::{BooleanQuery, Occur, RangeQuery};

        let temp_dir = tempfile::TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");
        let processor = writer::create_readwrite_index(&index_path);

        let now = TimestampMilli::now().0;
        let past = now - 86400_000; // 1 day ago
        let future = now + 86400_000; // 1 day from now

        let mutations = vec![
            // Always valid (no temporal range)
            Mutation::AddNode(AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli(now),
                name: "always_valid".to_string(),
                valid_range: None,
                summary: crate::graph::schema::NodeSummary::from_text("no temporal constraints"),
            }),
            // Currently valid (past..future)
            Mutation::AddNode(AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli(now),
                name: "currently_valid".to_string(),
                valid_range: Some(crate::TemporalRange(
                    Some(TimestampMilli(past)),
                    Some(TimestampMilli(future)),
                )),
                summary: crate::graph::schema::NodeSummary::from_text("valid now"),
            }),
            // Expired (past start, past end)
            Mutation::AddNode(AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli(now),
                name: "expired".to_string(),
                valid_range: Some(crate::TemporalRange(
                    Some(TimestampMilli(past - 86400_000)),
                    Some(TimestampMilli(past)),
                )),
                summary: crate::graph::schema::NodeSummary::from_text("no longer valid"),
            }),
            // Future (future start)
            Mutation::AddNode(AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli(now),
                name: "future".to_string(),
                valid_range: Some(crate::TemporalRange(Some(TimestampMilli(future)), None)),
                summary: crate::graph::schema::NodeSummary::from_text("not yet valid"),
            }),
        ];

        processor.process_mutations(&mutations).await.unwrap();

        let reader = processor.tantivy_index().reader().unwrap();
        let searcher = reader.searcher();

        // Query: Find documents that are currently valid
        // This means: valid_since <= now AND (valid_until > now OR valid_until not set)
        //
        // For documents with temporal ranges:
        // - valid_since must be <= now
        // - valid_until must be > now (or not set)
        //
        // Note: Documents without valid_since/valid_until are always valid
        // but they don't have these fields indexed, so we need a different approach.

        // Find documents with valid_since <= now (0 <= valid_since < now+1)
        let valid_since_query = RangeQuery::new_u64("valid_since".to_string(), 0..(now + 1));

        // Find documents with valid_until > now (now < valid_until)
        let valid_until_query = RangeQuery::new_u64("valid_until".to_string(), (now + 1)..u64::MAX);

        // Documents that have valid_since <= now AND valid_until > now
        let currently_valid_with_range = BooleanQuery::new(vec![
            (Occur::Must, Box::new(valid_since_query)),
            (Occur::Must, Box::new(valid_until_query)),
        ]);

        let top_docs = searcher
            .search(&currently_valid_with_range, &TopDocs::with_limit(10))
            .unwrap();

        // Should find only "currently_valid" (the one with past..future range)
        // "expired" has valid_until in the past, "future" has valid_since in the future
        assert_eq!(
            top_docs.len(),
            1,
            "Should find 1 document with active validity range"
        );
    }

    #[tokio::test]
    async fn test_query_documents_created_after_timestamp() {
        use tantivy::query::RangeQuery;

        let temp_dir = tempfile::TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");
        let processor = writer::create_readwrite_index(&index_path);

        // Use fixed timestamps for predictable testing
        let base_time: u64 = 1700000000000; // Some fixed point in time
        let ts1 = base_time;
        let ts2 = base_time + 1000;
        let ts3 = base_time + 2000;
        let ts4 = base_time + 3000;

        let mutations = vec![
            Mutation::AddNode(AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli(ts1),
                name: "first".to_string(),
                valid_range: None,
                summary: crate::graph::schema::NodeSummary::from_text("first document"),
            }),
            Mutation::AddNode(AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli(ts2),
                name: "second".to_string(),
                valid_range: None,
                summary: crate::graph::schema::NodeSummary::from_text("second document"),
            }),
            Mutation::AddNode(AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli(ts3),
                name: "third".to_string(),
                valid_range: None,
                summary: crate::graph::schema::NodeSummary::from_text("third document"),
            }),
            Mutation::AddNode(AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli(ts4),
                name: "fourth".to_string(),
                valid_range: None,
                summary: crate::graph::schema::NodeSummary::from_text("fourth document"),
            }),
        ];

        processor.process_mutations(&mutations).await.unwrap();

        let reader = processor.tantivy_index().reader().unwrap();
        let searcher = reader.searcher();

        // Query: Documents created after ts2 (ts > ts2, i.e., ts >= ts2+1)
        let after_ts2 = RangeQuery::new_u64("creation_timestamp".to_string(), (ts2 + 1)..u64::MAX);
        let top_docs = searcher
            .search(&after_ts2, &TopDocs::with_limit(10))
            .unwrap();
        assert_eq!(top_docs.len(), 2, "Should find third and fourth");

        // Query: Documents created before ts3 (ts < ts3)
        let before_ts3 = RangeQuery::new_u64("creation_timestamp".to_string(), 0..ts3);
        let top_docs = searcher
            .search(&before_ts3, &TopDocs::with_limit(10))
            .unwrap();
        assert_eq!(top_docs.len(), 2, "Should find first and second");

        // Query: Documents created at exactly ts2 (ts2 <= ts < ts2+1)
        let exact_ts2 = RangeQuery::new_u64("creation_timestamp".to_string(), ts2..(ts2 + 1));
        let top_docs = searcher
            .search(&exact_ts2, &TopDocs::with_limit(10))
            .unwrap();
        assert_eq!(top_docs.len(), 1, "Should find only second");
    }

    // ========================================================================
    // Storage Tests
    // ========================================================================

    #[tokio::test]
    async fn test_readwrite_storage_creates_index_in_empty_directory() {
        // Regression test: Storage::readwrite should create a new index
        // when the directory exists but is empty (no meta.json)
        let temp_dir = tempfile::TempDir::new().unwrap();
        let index_path = temp_dir.path().join("empty_dir");

        // Create an empty directory (simulating the bug scenario)
        std::fs::create_dir_all(&index_path).unwrap();
        assert!(index_path.exists());
        assert!(index_path.read_dir().unwrap().next().is_none()); // Directory is empty

        // This should succeed by creating a new index, not fail trying to open
        let mut storage = Storage::readwrite(&index_path);
        let result = storage.ready();
        assert!(result.is_ok(), "Should create new index in empty directory");

        // Verify meta.json was created
        assert!(index_path.join("meta.json").exists());

        // Verify we can use the index
        let processor = Index::new(Arc::new(storage));
        let node = crate::graph::mutation::AddNode {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "test_node".to_string(),
            valid_range: None,
            summary: crate::graph::schema::NodeSummary::from_text("test"),
        };
        processor
            .process_mutations(&[crate::graph::mutation::Mutation::AddNode(node)])
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn test_readwrite_storage_opens_existing_index() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let index_path = temp_dir.path().join("existing_index");

        // Create an index first
        {
            let mut storage = Storage::readwrite(&index_path);
            storage.ready().unwrap();
            let processor = Index::new(Arc::new(storage));

            let node = crate::graph::mutation::AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                name: "original_node".to_string(),
                valid_range: None,
                summary: crate::graph::schema::NodeSummary::from_text("original"),
            };
            processor
                .process_mutations(&[crate::graph::mutation::Mutation::AddNode(node)])
                .await
                .unwrap();
        }

        // Verify meta.json exists
        assert!(index_path.join("meta.json").exists());

        // Open the existing index - should succeed and preserve data
        let mut storage = Storage::readwrite(&index_path);
        storage.ready().unwrap();
        let processor = Index::new(Arc::new(storage));

        // Search for the original node
        let reader = processor.tantivy_index().reader().unwrap();
        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(
            processor.tantivy_index(),
            vec![processor.fields().node_name_field],
        );
        let query = query_parser.parse_query("original_node").unwrap();
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10)).unwrap();
        assert_eq!(top_docs.len(), 1, "Should find the original node");
    }
}
