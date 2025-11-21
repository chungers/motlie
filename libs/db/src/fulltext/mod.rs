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
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

use tantivy::schema::*;
use tantivy::{doc, Index, IndexWriter, Term};

// Submodules
pub mod search;
pub mod fuzzy;

// Re-export commonly used types
pub use search::{SearchOptions, SearchResults, FacetCounts};
pub use fuzzy::{FuzzySearchOptions, FuzzyLevel};

use crate::{
    mutation::{
        AddEdge, AddEdgeFragment, AddNode, AddNodeFragment, Consumer, Mutation, Processor,
        UpdateEdgeValidSinceUntil, UpdateEdgeWeight, UpdateNodeValidSinceUntil,
    },
    WriterConfig,
};

/// Trait for mutations to index themselves into Tantivy.
///
/// This trait defines HOW to index the mutation into the fulltext search index.
/// Each mutation type knows how to extract and index its own searchable content.
///
/// Following the same pattern as MutationExecutor for graph storage.
pub trait FulltextIndexExecutor: Send + Sync {
    /// Index this mutation into the Tantivy index writer.
    /// Each mutation type knows how to extract and index its searchable content.
    fn index(&self, index_writer: &IndexWriter, fields: &FulltextFields) -> Result<()>;
}

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
    pub doc_type_facet: Field,        // Document type as facet
    pub time_bucket_facet: Field,     // Time buckets (hour/day/week/month)
    pub weight_range_facet: Field,    // Weight ranges for edges
    pub tags_facet: Field,            // User-defined tags from #hashtags
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
fn extract_tags(content: &str) -> Vec<String> {
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
fn compute_time_bucket(ts: crate::TimestampMilli) -> Facet {
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
fn weight_to_facet(weight: f64) -> Facet {
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
// FulltextIndexExecutor Implementations
// ============================================================================

impl FulltextIndexExecutor for AddNode {
    fn index(&self, index_writer: &IndexWriter, fields: &FulltextFields) -> Result<()> {
        let mut doc = doc!(
            fields.id_field => self.id.as_bytes().to_vec(),
            fields.node_name_field => self.name.clone(),
            fields.doc_type_field => "node",
            fields.timestamp_field => self.ts_millis.0,
        );

        // Add facets
        doc.add_facet(fields.doc_type_facet, Facet::from("/type/node"));
        doc.add_facet(fields.time_bucket_facet, compute_time_bucket(self.ts_millis));

        index_writer
            .add_document(doc)
            .context("Failed to index AddNode")?;

        log::debug!("[FullText] Indexed node: id={}, name={}", self.id, self.name);
        Ok(())
    }
}

impl FulltextIndexExecutor for AddEdge {
    fn index(&self, index_writer: &IndexWriter, fields: &FulltextFields) -> Result<()> {
        // Decode edge summary content
        let summary_text = self
            .summary
            .decode_string()
            .unwrap_or_else(|_| String::new());

        // Extract tags from summary
        let tags = extract_tags(&summary_text);

        let mut doc = doc!(
            fields.src_id_field => self.source_node_id.as_bytes().to_vec(),
            fields.dst_id_field => self.target_node_id.as_bytes().to_vec(),
            fields.edge_name_field => self.name.clone(),
            fields.content_field => summary_text,
            fields.doc_type_field => "edge",
            fields.timestamp_field => self.ts_millis.0,
        );

        // Add facets
        doc.add_facet(fields.doc_type_facet, Facet::from("/type/edge"));
        doc.add_facet(fields.time_bucket_facet, compute_time_bucket(self.ts_millis));

        // Add weight if present
        if let Some(weight) = self.weight {
            doc.add_f64(fields.weight_field, weight);
            doc.add_facet(fields.weight_range_facet, weight_to_facet(weight));
        }

        // Add user-defined tags as facets
        for tag in tags {
            doc.add_facet(fields.tags_facet, Facet::from(&format!("/tag/{}", tag)));
        }

        index_writer
            .add_document(doc)
            .context("Failed to index AddEdge")?;

        log::debug!(
            "[FullText] Indexed edge: src={}, dst={}, name={}",
            self.source_node_id,
            self.target_node_id,
            self.name
        );
        Ok(())
    }
}

impl FulltextIndexExecutor for AddNodeFragment {
    fn index(&self, index_writer: &IndexWriter, fields: &FulltextFields) -> Result<()> {
        // Decode DataUrl content
        let content_text = self
            .content
            .decode_string()
            .context("Failed to decode fragment content")?;

        // Extract user-defined tags from content
        let tags = extract_tags(&content_text);

        let mut doc = doc!(
            fields.id_field => self.id.as_bytes().to_vec(),
            fields.content_field => content_text,
            fields.doc_type_field => "node_fragment",
            fields.timestamp_field => self.ts_millis.0,
        );

        // Add facets
        doc.add_facet(fields.doc_type_facet, Facet::from("/type/node_fragment"));
        doc.add_facet(fields.time_bucket_facet, compute_time_bucket(self.ts_millis));

        // Add user-defined tags as facets
        for tag in tags {
            doc.add_facet(fields.tags_facet, Facet::from(&format!("/tag/{}", tag)));
        }

        index_writer
            .add_document(doc)
            .context("Failed to index AddNodeFragment")?;

        log::debug!(
            "[FullText] Indexed node fragment: id={}, content_len={}",
            self.id,
            self.content.as_ref().len()
        );
        Ok(())
    }
}

impl FulltextIndexExecutor for AddEdgeFragment {
    fn index(&self, index_writer: &IndexWriter, fields: &FulltextFields) -> Result<()> {
        // Decode DataUrl content
        let content_text = self
            .content
            .decode_string()
            .context("Failed to decode edge fragment content")?;

        // Extract user-defined tags from content
        let tags = extract_tags(&content_text);

        let mut doc = doc!(
            fields.src_id_field => self.src_id.as_bytes().to_vec(),
            fields.dst_id_field => self.dst_id.as_bytes().to_vec(),
            fields.edge_name_field => self.edge_name.clone(),
            fields.content_field => content_text,
            fields.doc_type_field => "edge_fragment",
            fields.timestamp_field => self.ts_millis.0,
        );

        // Add facets
        doc.add_facet(fields.doc_type_facet, Facet::from("/type/edge_fragment"));
        doc.add_facet(fields.time_bucket_facet, compute_time_bucket(self.ts_millis));

        // Add user-defined tags as facets
        for tag in tags {
            doc.add_facet(fields.tags_facet, Facet::from(&format!("/tag/{}", tag)));
        }

        index_writer
            .add_document(doc)
            .context("Failed to index AddEdgeFragment")?;

        log::debug!(
            "[FullText] Indexed edge fragment: src={}, dst={}, name={}, content_len={}",
            self.src_id,
            self.dst_id,
            self.edge_name,
            self.content.as_ref().len()
        );
        Ok(())
    }
}

impl FulltextIndexExecutor for UpdateNodeValidSinceUntil {
    fn index(&self, index_writer: &IndexWriter, fields: &FulltextFields) -> Result<()> {
        // Delete existing documents for this node ID
        let id_term = Term::from_field_bytes(fields.id_field, self.id.as_bytes());
        index_writer.delete_term(id_term);

        log::debug!(
            "[FullText] Deleted node documents for temporal update: id={}, reason={}",
            self.id,
            self.reason
        );
        Ok(())
    }
}

impl FulltextIndexExecutor for UpdateEdgeValidSinceUntil {
    fn index(&self, index_writer: &IndexWriter, fields: &FulltextFields) -> Result<()> {
        // Delete existing documents for this edge
        // We need to delete by composite key (src_id + dst_id + edge_name)
        // Tantivy doesn't support composite term deletion directly, so we delete by src_id
        // and let the search handle filtering
        let src_term = Term::from_field_bytes(fields.src_id_field, self.src_id.as_bytes());
        index_writer.delete_term(src_term);

        log::debug!(
            "[FullText] Deleted edge documents for temporal update: src={}, dst={}, name={}, reason={}",
            self.src_id,
            self.dst_id,
            self.name,
            self.reason
        );
        Ok(())
    }
}

impl FulltextIndexExecutor for UpdateEdgeWeight {
    fn index(&self, _index_writer: &IndexWriter, _fields: &FulltextFields) -> Result<()> {
        // For weight updates, we'd need to delete and re-index
        // For now, just log as this is primarily a graph operation
        log::debug!(
            "[FullText] Edge weight updated (no index change needed): src={}, dst={}, name={}, weight={}",
            self.src_id,
            self.dst_id,
            self.name,
            self.weight
        );
        Ok(())
    }
}

// ============================================================================
// FullTextProcessor
// ============================================================================

/// Full-text search mutation processor using Tantivy
pub struct FullTextProcessor {
    index_path: PathBuf,
    index: Arc<Index>,
    index_writer: Arc<Mutex<IndexWriter>>,
    fields: FulltextFields,
    // BM25 parameters (stored for reference, Tantivy uses defaults)
    k1: f32,
    b: f32,
}

impl FullTextProcessor {
    /// Create a new full-text processor with default BM25 parameters
    ///
    /// # Arguments
    /// * `index_path` - Directory path where the Tantivy index will be stored
    ///
    /// # Returns
    /// A new FullTextProcessor instance with the index ready for writes
    pub fn new(index_path: &Path) -> Result<Self> {
        Self::with_params(index_path, 1.2, 0.75)
    }

    /// Create a new full-text processor with custom BM25 parameters
    ///
    /// # Arguments
    /// * `index_path` - Directory path where the Tantivy index will be stored
    /// * `k1` - BM25 k1 parameter (controls term frequency saturation)
    /// * `b` - BM25 b parameter (controls document length normalization)
    pub fn with_params(index_path: &Path, k1: f32, b: f32) -> Result<Self> {
        // Build schema
        let (schema, fields) = build_fulltext_schema();

        // Create or open index
        let index = if index_path.exists() {
            Index::open_in_dir(index_path)
                .context("Failed to open existing Tantivy index")?
        } else {
            std::fs::create_dir_all(index_path)
                .context("Failed to create index directory")?;
            Index::create_in_dir(index_path, schema.clone())
                .context("Failed to create Tantivy index")?
        };

        // Create index writer with 50MB buffer
        let index_writer = index
            .writer(50_000_000)
            .context("Failed to create index writer")?;

        log::info!(
            "[FullText] Initialized Tantivy index at {:?} with BM25 params k1={}, b={}",
            index_path,
            k1,
            b
        );

        Ok(Self {
            index_path: PathBuf::from(index_path),
            index: Arc::new(index),
            index_writer: Arc::new(Mutex::new(index_writer)),
            fields,
            k1,
            b,
        })
    }

    /// Get a reference to the Tantivy index for searching
    pub fn index(&self) -> &Index {
        &self.index
    }

    /// Get the field handles
    pub fn fields(&self) -> &FulltextFields {
        &self.fields
    }

    /// Get the index path
    pub fn index_path(&self) -> &Path {
        &self.index_path
    }
}

impl Default for FullTextProcessor {
    fn default() -> Self {
        // Use a temporary directory for default
        let temp_dir = std::env::temp_dir().join("motlie_fulltext_index");
        Self::new(&temp_dir).expect("Failed to create default FullTextProcessor")
    }
}

#[async_trait::async_trait]
impl Processor for FullTextProcessor {
    /// Process a batch of mutations - index content for full-text search
    async fn process_mutations(&self, mutations: &[Mutation]) -> Result<()> {
        if mutations.is_empty() {
            return Ok(());
        }

        log::info!(
            "[FullText] Processing {} mutations for indexing",
            mutations.len()
        );

        // Lock the index writer for the entire batch
        let mut writer = self.index_writer.lock().await;

        // Index each mutation
        for mutation in mutations {
            match mutation {
                Mutation::AddNode(m) => m.index(&writer, &self.fields)?,
                Mutation::AddEdge(m) => m.index(&writer, &self.fields)?,
                Mutation::AddNodeFragment(m) => m.index(&writer, &self.fields)?,
                Mutation::AddEdgeFragment(m) => m.index(&writer, &self.fields)?,
                Mutation::UpdateNodeValidSinceUntil(m) => m.index(&writer, &self.fields)?,
                Mutation::UpdateEdgeValidSinceUntil(m) => m.index(&writer, &self.fields)?,
                Mutation::UpdateEdgeWeight(m) => m.index(&writer, &self.fields)?,
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
// Consumer Creation Functions
// ============================================================================

/// Create a new full-text mutation consumer with default parameters
pub fn create_fulltext_consumer(
    receiver: mpsc::Receiver<Vec<crate::Mutation>>,
    config: WriterConfig,
    index_path: &Path,
) -> Consumer<FullTextProcessor> {
    let processor = FullTextProcessor::new(index_path).expect("Failed to create FullTextProcessor");
    Consumer::new(receiver, config, processor)
}

/// Create a new full-text mutation consumer with default parameters that chains to another processor
pub fn create_fulltext_consumer_with_next(
    receiver: mpsc::Receiver<Vec<crate::Mutation>>,
    config: WriterConfig,
    index_path: &Path,
    next: mpsc::Sender<Vec<crate::Mutation>>,
) -> Consumer<FullTextProcessor> {
    let processor = FullTextProcessor::new(index_path).expect("Failed to create FullTextProcessor");
    Consumer::with_next(receiver, config, processor, next)
}

/// Create a new full-text mutation consumer with custom BM25 parameters
pub fn create_fulltext_consumer_with_params(
    receiver: mpsc::Receiver<Vec<crate::Mutation>>,
    config: WriterConfig,
    index_path: &Path,
    k1: f32,
    b: f32,
) -> Consumer<FullTextProcessor> {
    let processor = FullTextProcessor::with_params(index_path, k1, b)
        .expect("Failed to create FullTextProcessor");
    Consumer::new(receiver, config, processor)
}

/// Create a new full-text mutation consumer with custom BM25 parameters that chains to another processor
pub fn create_fulltext_consumer_with_params_and_next(
    receiver: mpsc::Receiver<Vec<crate::Mutation>>,
    config: WriterConfig,
    index_path: &Path,
    k1: f32,
    b: f32,
    next: mpsc::Sender<Vec<crate::Mutation>>,
) -> Consumer<FullTextProcessor> {
    let processor = FullTextProcessor::with_params(index_path, k1, b)
        .expect("Failed to create FullTextProcessor");
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

/// Spawn the full-text mutation consumer as a background task with default parameters and chaining
pub fn spawn_fulltext_consumer_with_next(
    receiver: mpsc::Receiver<Vec<crate::Mutation>>,
    config: WriterConfig,
    index_path: &Path,
    next: mpsc::Sender<Vec<crate::Mutation>>,
) -> JoinHandle<Result<()>> {
    let consumer = create_fulltext_consumer_with_next(receiver, config, index_path, next);
    crate::mutation::spawn_consumer(consumer)
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
#[path = "tests.rs"]
mod tests;
