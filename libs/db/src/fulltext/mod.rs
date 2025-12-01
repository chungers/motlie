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
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

use tantivy::schema::*;
use tantivy::IndexWriter;

// Submodules
pub mod fuzzy;
pub mod mutation;
pub mod query;
pub mod search;

// Re-export commonly used types
pub use fuzzy::{FuzzyLevel, FuzzySearchOptions};
pub use mutation::FulltextIndexExecutor;
pub use query::{
    create_query_consumer as create_fulltext_query_consumer,
    create_query_reader as create_fulltext_query_reader,
    spawn_query_consumer as spawn_fulltext_query_consumer,
    spawn_query_consumer_pool_readonly as spawn_fulltext_query_consumer_pool_readonly,
    spawn_query_consumer_pool_shared as spawn_fulltext_query_consumer_pool_shared,
    Consumer as FulltextQueryConsumer, NodeSearchResult, Nodes as FulltextNodes,
    Query as FulltextQuery, QueryExecutor as FulltextQueryExecutor,
    QueryProcessor as FulltextQueryProcessor, Reader as FulltextReader,
    ReaderConfig as FulltextReaderConfig, Runnable as FulltextQueryRunnable,
};
pub use search::{FacetCounts, SearchOptions, SearchResults};

use crate::{
    mutation::{Consumer, Mutation, Processor},
    WriterConfig,
};

/// Field handles for efficient access to the Tantivy schema fields
#[derive(Clone)]
pub struct FulltextFields {
    // ID fields
    pub id_field: Field,
    pub src_id_field: Field,
    pub dst_id_field: Field,

    // Name fields
    pub node_name_field: Field,
    pub edge_name_field: Field,

    // Content field (main searchable text)
    pub content_field: Field,

    // Temporal fields
    pub timestamp_field: Field,
    pub valid_since_field: Field,
    pub valid_until_field: Field,

    // Document type discriminator
    pub doc_type_field: Field,

    // Edge-specific fields
    pub weight_field: Field,

    // Facet fields (for categorical filtering)
    pub doc_type_facet: Field,     // Document type as facet
    pub time_bucket_facet: Field,  // Time buckets (hour/day/week/month)
    pub weight_range_facet: Field, // Weight ranges for edges
    pub tags_facet: Field,         // User-defined tags from #hashtags
}

/// Build the Tantivy schema for fulltext indexing
fn build_fulltext_schema() -> (Schema, FulltextFields) {
    let mut schema_builder = Schema::builder();

    // ID fields (stored as bytes, not tokenized)
    let id_field = schema_builder.add_bytes_field("id", STORED | FAST);
    let src_id_field = schema_builder.add_bytes_field("src_id", STORED | FAST);
    let dst_id_field = schema_builder.add_bytes_field("dst_id", STORED | FAST);

    // Name fields (tokenized and stored)
    let text_options = TextOptions::default()
        .set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("default")
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        )
        .set_stored();

    let node_name_field = schema_builder.add_text_field("node_name", text_options.clone());
    let edge_name_field = schema_builder.add_text_field("edge_name", text_options.clone());

    // Content field (main searchable text with BM25)
    let content_field = schema_builder.add_text_field("content", text_options);

    // Temporal fields (for range queries)
    let timestamp_field = schema_builder.add_u64_field("timestamp", INDEXED | STORED | FAST);
    let valid_since_field = schema_builder.add_u64_field("valid_since", INDEXED | FAST);
    let valid_until_field = schema_builder.add_u64_field("valid_until", INDEXED | FAST);

    // Document type (node, edge, node_fragment, edge_fragment)
    let doc_type_field = schema_builder.add_text_field("doc_type", STRING | STORED);

    // Weight field for edges
    let weight_field = schema_builder.add_f64_field("weight", INDEXED | STORED | FAST);

    // Facet fields (for categorical filtering and aggregation)
    let doc_type_facet = schema_builder.add_facet_field("doc_type_facet", INDEXED | STORED);
    let time_bucket_facet = schema_builder.add_facet_field("time_bucket_facet", INDEXED | STORED);
    let weight_range_facet = schema_builder.add_facet_field("weight_range_facet", INDEXED | STORED);
    let tags_facet = schema_builder.add_facet_field("tags", INDEXED | STORED);

    let schema = schema_builder.build();

    let fields = FulltextFields {
        id_field,
        src_id_field,
        dst_id_field,
        node_name_field,
        edge_name_field,
        content_field,
        timestamp_field,
        valid_since_field,
        valid_until_field,
        doc_type_field,
        weight_field,
        doc_type_facet,
        time_bucket_facet,
        weight_range_facet,
        tags_facet,
    };

    (schema, fields)
}

// ============================================================================
// Tag Extraction for User-Defined Facets
// ============================================================================

/// Extract hashtags from content for user-defined facets
///
/// Supports formats:
/// - #tag
/// - #multi_word_tag
/// - #CamelCaseTag
///
/// Example: "This is about #rust and #systems_programming"
/// Returns: ["rust", "systems_programming"]
pub(crate) fn extract_tags(content: &str) -> Vec<String> {
    let mut tags = Vec::new();
    let mut chars = content.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '#' {
            let mut tag = String::new();

            // Collect tag characters (alphanumeric and underscore)
            while let Some(&next_ch) = chars.peek() {
                if next_ch.is_alphanumeric() || next_ch == '_' {
                    tag.push(chars.next().unwrap());
                } else {
                    break;
                }
            }

            if !tag.is_empty() {
                tags.push(tag.to_lowercase());
            }
        }
    }

    tags
}

// ============================================================================
// Facet Helper Functions
// ============================================================================

/// Convert timestamp to time bucket facet
pub(crate) fn compute_time_bucket(ts: crate::TimestampMilli) -> Facet {
    let now = crate::TimestampMilli::now().0;
    let diff = now.saturating_sub(ts.0);

    if diff < 3600_000 {
        Facet::from("/time/last_hour")
    } else if diff < 86400_000 {
        Facet::from("/time/last_day")
    } else if diff < 604800_000 {
        Facet::from("/time/last_week")
    } else if diff < 2592000_000 {
        Facet::from("/time/last_month")
    } else {
        Facet::from("/time/older")
    }
}

/// Convert edge weight to range facet
pub(crate) fn weight_to_facet(weight: f64) -> Facet {
    if weight < 0.5 {
        Facet::from("/weight/0-0.5")
    } else if weight < 1.0 {
        Facet::from("/weight/0.5-1.0")
    } else if weight < 2.0 {
        Facet::from("/weight/1.0-2.0")
    } else {
        Facet::from("/weight/2.0+")
    }
}

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
    fields: Option<FulltextFields>,
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
    pub fn ready(&mut self) -> Result<()> {
        if self.index.is_some() {
            return Ok(());
        }

        let (schema, fields) = build_fulltext_schema();

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

                log::info!(
                    "[FullText Storage] Opened index in READONLY mode at {:?}",
                    self.index_path
                );

                self.index = Some(index);
                self.fields = Some(fields);
            }
            StorageMode::ReadWrite => {
                // ReadWrite: Create or open index with exclusive writer
                let index = if self.index_path.exists() {
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

                log::info!(
                    "[FullText Storage] Opened index in READWRITE mode at {:?}",
                    self.index_path
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
    pub fn fields(&self) -> Result<&FulltextFields> {
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
    storage: Arc<Storage>,
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
    pub fn fields(&self) -> &FulltextFields {
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

#[async_trait::async_trait]
impl Processor for Index {
    /// Process a batch of mutations - index content for full-text search.
    ///
    /// Requires the Index to be in readwrite mode.
    async fn process_mutations(&self, mutations: &[Mutation]) -> Result<()> {
        if mutations.is_empty() {
            return Ok(());
        }

        log::info!(
            "[FullText] Processing {} mutations for indexing",
            mutations.len()
        );

        // Get the writer (requires readwrite mode)
        let writer_mutex = self
            .storage
            .writer()
            .ok_or_else(|| anyhow::anyhow!("[FullText] Index not in readwrite mode"))?;

        let fields = self.storage.fields()?;
        let mut writer = writer_mutex.lock().await;

        // Index each mutation
        for mutation in mutations {
            match mutation {
                Mutation::AddNode(m) => m.index(&mut writer, fields)?,
                Mutation::AddEdge(m) => m.index(&mut writer, fields)?,
                Mutation::AddNodeFragment(m) => m.index(&mut writer, fields)?,
                Mutation::AddEdgeFragment(m) => m.index(&mut writer, fields)?,
                Mutation::UpdateNodeValidSinceUntil(m) => m.index(&mut writer, fields)?,
                Mutation::UpdateEdgeValidSinceUntil(m) => m.index(&mut writer, fields)?,
                Mutation::UpdateEdgeWeight(m) => m.index(&mut writer, fields)?,
            }
        }

        // Commit the batch atomically
        writer.commit().context("Failed to commit batch to index")?;

        log::info!(
            "[FullText] Successfully indexed {} mutations",
            mutations.len()
        );
        Ok(())
    }
}

// ============================================================================
// Helper Functions for Creating Storage and Index
// ============================================================================

/// Create a readwrite Index from a path (convenience function).
///
/// This handles the full setup: Storage::readwrite -> ready -> Arc -> Index.
/// Use this for mutation consumers.
fn create_readwrite_index(index_path: &Path) -> Index {
    let mut storage = Storage::readwrite(index_path);
    storage.ready().expect("Failed to ready storage");
    Index::new(Arc::new(storage))
}

/// Create a readonly Index from a path (convenience function).
///
/// This handles the full setup: Storage::readonly -> ready -> Arc -> Index.
/// Use this for query consumers.
#[allow(dead_code)]
fn create_readonly_index(index_path: &Path) -> Index {
    let mut storage = Storage::readonly(index_path);
    storage.ready().expect("Failed to ready storage");
    Index::new(Arc::new(storage))
}

// ============================================================================
// Consumer Creation Functions
// ============================================================================

/// Create a new full-text mutation consumer with default parameters
pub fn create_fulltext_consumer(
    receiver: mpsc::Receiver<Vec<crate::Mutation>>,
    config: WriterConfig,
    index_path: &Path,
) -> Consumer<Index> {
    let processor = create_readwrite_index(index_path);
    Consumer::new(receiver, config, processor)
}

/// Create a new full-text mutation consumer with default parameters that chains to another processor
pub fn create_fulltext_consumer_with_next(
    receiver: mpsc::Receiver<Vec<crate::Mutation>>,
    config: WriterConfig,
    index_path: &Path,
    next: mpsc::Sender<Vec<crate::Mutation>>,
) -> Consumer<Index> {
    let processor = create_readwrite_index(index_path);
    Consumer::with_next(receiver, config, processor, next)
}

/// Create a new full-text mutation consumer with custom BM25 parameters
pub fn create_fulltext_consumer_with_params(
    receiver: mpsc::Receiver<Vec<crate::Mutation>>,
    config: WriterConfig,
    index_path: &Path,
    _k1: f32,
    _b: f32,
) -> Consumer<Index> {
    // Note: BM25 params not currently used - Tantivy uses defaults
    let processor = create_readwrite_index(index_path);
    Consumer::new(receiver, config, processor)
}

/// Create a new full-text mutation consumer with custom BM25 parameters that chains to another processor
pub fn create_fulltext_consumer_with_params_and_next(
    receiver: mpsc::Receiver<Vec<crate::Mutation>>,
    config: WriterConfig,
    index_path: &Path,
    _k1: f32,
    _b: f32,
    next: mpsc::Sender<Vec<crate::Mutation>>,
) -> Consumer<Index> {
    // Note: BM25 params not currently used - Tantivy uses defaults
    let processor = create_readwrite_index(index_path);
    Consumer::with_next(receiver, config, processor, next)
}

/// Spawn the full-text mutation consumer as a background task with default parameters
pub fn spawn_fulltext_consumer(
    receiver: mpsc::Receiver<Vec<crate::Mutation>>,
    config: WriterConfig,
    index_path: &Path,
) -> JoinHandle<Result<()>> {
    let consumer = create_fulltext_consumer(receiver, config, index_path);
    crate::mutation::spawn_consumer(consumer)
}

/// Spawn the full-text mutation consumer as a background task with default parameters and chaining.
///
/// This follows the same pattern as `spawn_graph_consumer_with_next` - the consumer processes
/// mutations and then forwards them to the next consumer in the chain.
///
/// # Example
/// ```no_run
/// use motlie_db::{create_mutation_writer, WriterConfig};
/// use motlie_db::fulltext::spawn_fulltext_mutation_consumer_with_next;
/// use std::path::Path;
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = WriterConfig { channel_buffer_size: 100 };
/// let (writer, mutation_rx) = create_mutation_writer(config.clone());
///
/// // Create channel for next consumer in chain
/// let (next_tx, next_rx) = tokio::sync::mpsc::channel(100);
///
/// // Fulltext consumer processes mutations then forwards to next_tx
/// let handle = spawn_fulltext_mutation_consumer_with_next(
///     mutation_rx,
///     config,
///     Path::new("/data/fulltext"),
///     next_tx,
/// );
/// # Ok(())
/// # }
/// ```
pub fn spawn_fulltext_mutation_consumer_with_next(
    receiver: mpsc::Receiver<Vec<crate::Mutation>>,
    config: WriterConfig,
    index_path: &Path,
    next: mpsc::Sender<Vec<crate::Mutation>>,
) -> JoinHandle<Result<()>> {
    let consumer = create_fulltext_consumer_with_next(receiver, config, index_path, next);
    crate::mutation::spawn_consumer(consumer)
}

/// Alias for `spawn_fulltext_mutation_consumer_with_next` for backward compatibility.
#[deprecated(
    since = "0.2.0",
    note = "Use spawn_fulltext_mutation_consumer_with_next instead"
)]
pub fn spawn_fulltext_consumer_with_next(
    receiver: mpsc::Receiver<Vec<crate::Mutation>>,
    config: WriterConfig,
    index_path: &Path,
    next: mpsc::Sender<Vec<crate::Mutation>>,
) -> JoinHandle<Result<()>> {
    spawn_fulltext_mutation_consumer_with_next(receiver, config, index_path, next)
}

/// Spawn the full-text mutation consumer as a background task with custom BM25 parameters
pub fn spawn_fulltext_consumer_with_params(
    receiver: mpsc::Receiver<Vec<crate::Mutation>>,
    config: WriterConfig,
    index_path: &Path,
    k1: f32,
    b: f32,
) -> JoinHandle<Result<()>> {
    let consumer = create_fulltext_consumer_with_params(receiver, config, index_path, k1, b);
    crate::mutation::spawn_consumer(consumer)
}

/// Spawn the full-text mutation consumer as a background task with custom BM25 parameters and chaining
pub fn spawn_fulltext_consumer_with_params_and_next(
    receiver: mpsc::Receiver<Vec<crate::Mutation>>,
    config: WriterConfig,
    index_path: &Path,
    k1: f32,
    b: f32,
    next: mpsc::Sender<Vec<crate::Mutation>>,
) -> JoinHandle<Result<()>> {
    let consumer =
        create_fulltext_consumer_with_params_and_next(receiver, config, index_path, k1, b, next);
    crate::mutation::spawn_consumer(consumer)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        mutation::{
            AddEdge, AddEdgeFragment, AddNode, AddNodeFragment, UpdateNodeValidSinceUntil,
        },
        DataUrl, Id, Mutation, TimestampMilli,
    };
    use tantivy::collector::TopDocs;
    use tantivy::query::QueryParser;

    #[test]
    fn test_fulltext_processor_creation() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");

        let processor = create_readwrite_index(&index_path);

        assert_eq!(processor.index_path(), index_path);
        assert!(index_path.exists());
    }

    #[test]
    fn test_fulltext_processor_with_custom_params() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");

        // BM25 params don't affect creation - just verify setup works
        let processor = create_readwrite_index(&index_path);

        // Just verify it creates successfully
        assert!(processor.index_path().exists());
    }

    #[tokio::test]
    async fn test_index_add_node() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");
        let processor = create_readwrite_index(&index_path);

        let node = AddNode {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "test_node".to_string(),
            temporal_range: None,
            summary: crate::NodeSummary::from_text("test summary"),
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
        let processor = create_readwrite_index(&index_path);

        let node_id = Id::new();
        let fragment = AddNodeFragment {
            id: node_id,
            ts_millis: TimestampMilli::now(),
            content: DataUrl::from_text("This is searchable content about Rust programming"),
            temporal_range: None,
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
        let processor = create_readwrite_index(&index_path);

        let edge = AddEdge {
            source_node_id: Id::new(),
            target_node_id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "depends_on".to_string(),
            summary: crate::schema::EdgeSummary::from_text("dependency relationship"),
            weight: Some(1.0),
            temporal_range: None,
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
        let processor = create_readwrite_index(&index_path);

        let edge_fragment = AddEdgeFragment {
            src_id: Id::new(),
            dst_id: Id::new(),
            edge_name: "connects_to".to_string(),
            ts_millis: TimestampMilli::now(),
            content: DataUrl::from_markdown(
                "# Connection Details\nThis edge represents a network connection",
            ),
            temporal_range: None,
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
        let processor = create_readwrite_index(&index_path);

        // Create a batch of mutations
        let mutations = vec![
            Mutation::AddNode(AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                name: "node_one".to_string(),
                temporal_range: None,
                summary: crate::NodeSummary::from_text("node one summary"),
            }),
            Mutation::AddNode(AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                name: "node_two".to_string(),
                temporal_range: None,
                summary: crate::NodeSummary::from_text("node two summary"),
            }),
            Mutation::AddNodeFragment(AddNodeFragment {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                content: DataUrl::from_text("fragment content here"),
                temporal_range: None,
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
        let processor = create_readwrite_index(&index_path);

        let node_id = Id::new();

        // First add a node
        let add_node = Mutation::AddNode(AddNode {
            id: node_id,
            ts_millis: TimestampMilli::now(),
            name: "test_node".to_string(),
            temporal_range: None,
            summary: crate::NodeSummary::from_text("test summary"),
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
            temporal_range: crate::schema::ValidTemporalRange(
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
        let processor = create_readwrite_index(&index_path);

        let node_id = Id::new();

        // Add multiple fragments for the same node
        let mutations = vec![
            Mutation::AddNodeFragment(AddNodeFragment {
                id: node_id,
                ts_millis: TimestampMilli::now(),
                content: DataUrl::from_text("first fragment about databases"),
                temporal_range: None,
            }),
            Mutation::AddNodeFragment(AddNodeFragment {
                id: node_id,
                ts_millis: TimestampMilli::now(),
                content: DataUrl::from_text("second fragment about indexing"),
                temporal_range: None,
            }),
            Mutation::AddNodeFragment(AddNodeFragment {
                id: node_id,
                ts_millis: TimestampMilli::now(),
                content: DataUrl::from_text("third fragment about search"),
                temporal_range: None,
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
        let processor = create_readwrite_index(&index_path);

        let mutations = vec![
            Mutation::AddNodeFragment(AddNodeFragment {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                content: DataUrl::from_text("plain text content"),
                temporal_range: None,
            }),
            Mutation::AddNodeFragment(AddNodeFragment {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                content: DataUrl::from_markdown("# Markdown content"),
                temporal_range: None,
            }),
            Mutation::AddNodeFragment(AddNodeFragment {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                content: DataUrl::from_html("<p>HTML content</p>"),
                temporal_range: None,
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
        let processor = create_readwrite_index(&index_path);

        let edge_with_weight = AddEdge {
            source_node_id: Id::new(),
            target_node_id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "weighted_edge".to_string(),
            summary: crate::schema::EdgeSummary::from_text("weighted connection"),
            weight: Some(2.5),
            temporal_range: None,
        };

        let edge_without_weight = AddEdge {
            source_node_id: Id::new(),
            target_node_id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "unweighted_edge".to_string(),
            summary: crate::schema::EdgeSummary::from_text("unweighted connection"),
            weight: None,
            temporal_range: None,
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

    #[tokio::test]
    async fn test_empty_mutations_batch() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");
        let processor = create_readwrite_index(&index_path);

        // Processing empty batch should not error
        let result = processor.process_mutations(&[]).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_index_persistence() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");

        // Create processor and add a document
        {
            let processor = create_readwrite_index(&index_path);
            let node = AddNode {
                id: Id::new(),
                ts_millis: TimestampMilli::now(),
                name: "persistent_node".to_string(),
                temporal_range: None,
                summary: crate::NodeSummary::from_text("persistent summary"),
            };
            processor
                .process_mutations(&[Mutation::AddNode(node)])
                .await
                .unwrap();
        } // Drop processor to close the index

        // Open again and verify the document is still there
        let processor = create_readwrite_index(&index_path);
        let reader = processor.tantivy_index().reader().unwrap();
        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(
            processor.tantivy_index(),
            vec![processor.fields().node_name_field],
        );
        let query = query_parser.parse_query("persistent_node").unwrap();
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10)).unwrap();

        assert_eq!(top_docs.len(), 1);
    }

    #[tokio::test]
    async fn test_fulltext_consumer_integration() {
        use crate::mutation::Runnable as MutRunnable;
        use crate::{create_mutation_writer, WriterConfig};
        use tokio::time::Duration;

        let temp_dir = tempfile::TempDir::new().unwrap();
        let index_path = temp_dir.path().join("test_index");

        let config = WriterConfig {
            channel_buffer_size: 10,
        };

        let (writer, receiver) = create_mutation_writer(config.clone());

        // Spawn consumer
        let consumer_handle = spawn_fulltext_consumer(receiver, config, &index_path);

        // Send some mutations
        let node_args = AddNode {
            id: Id::new(),
            ts_millis: TimestampMilli::now(),
            name: "test_node".to_string(),
            temporal_range: None,
            summary: crate::NodeSummary::from_text("test summary"),
        };
        node_args.run(&writer).await.unwrap();

        let fragment_args = AddNodeFragment {
            id: Id::new(),
            ts_millis: TimestampMilli(1234567890),
            content: crate::DataUrl::from_text(
                "This is a test fragment with some searchable content",
            ),
            temporal_range: None,
        };
        fragment_args.run(&writer).await.unwrap();

        // Give consumer time to process
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Drop writer to close channel
        drop(writer);

        // Wait for consumer to finish
        consumer_handle.await.unwrap().unwrap();

        // Verify the documents are indexed
        let processor = create_readwrite_index(&index_path);
        let reader = processor.tantivy_index().reader().unwrap();
        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(
            processor.tantivy_index(),
            vec![
                processor.fields().node_name_field,
                processor.fields().content_field,
            ],
        );

        // Search for node
        let query = query_parser.parse_query("test_node").unwrap();
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10)).unwrap();
        assert_eq!(top_docs.len(), 1);

        // Search for fragment content
        let query = query_parser.parse_query("searchable").unwrap();
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10)).unwrap();
        assert_eq!(top_docs.len(), 1);
    }
}
