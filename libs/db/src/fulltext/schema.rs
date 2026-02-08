//! Schema definitions for the fulltext index.
//!
//! This module defines the Tantivy schema and field handles used for fulltext indexing.

use tantivy::schema::{
    self, Facet, Field, IndexRecordOption, TextFieldIndexing, TextOptions, FAST, INDEXED, STORED,
    STRING,
};

/// Field handles for efficient access to the Tantivy schema fields.
///
/// This struct holds references to all fields in the fulltext index schema,
/// allowing efficient document construction and querying without repeated lookups.
#[derive(Clone)]
pub struct DocumentFields {
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
    pub creation_timestamp_field: Field,
    pub valid_since_field: Field,
    pub valid_until_field: Field,

    // Document type discriminator
    pub doc_type_field: Field,

    // Edge-specific fields
    pub weight_field: Field,

    // Facet fields (for categorical filtering)
    pub doc_type_facet: Field,  // Document type as facet
    pub tags_facet: Field,      // User-defined tags from #hashtags
    pub validity_facet: Field,  // Validity structure: unbounded, since_only, until_only, bounded
}

/// Build the Tantivy schema for fulltext indexing.
///
/// Returns a tuple of (tantivy::schema::Schema, DocumentFields) where:
/// - Schema is the Tantivy schema definition
/// - DocumentFields contains handles to all fields for efficient access
///
/// # Storage Strategy
///
/// Since the source of truth is RocksDB (graph storage), we minimize what's stored in Tantivy:
/// - **STORED**: Only fields needed to construct RocksDB lookup keys (IDs + timestamps for fragments)
/// - **INDEXED only**: Fields used for searching/filtering but retrievable from RocksDB
///
/// This reduces Tantivy's disk usage and memory footprint.
pub fn build_schema() -> (schema::Schema, DocumentFields) {
    let mut schema_builder = schema::Schema::builder();

    // ID fields - STORED for RocksDB lookups, INDEXED for delete_term operations
    // IMPORTANT: INDEXED is required for delete_term to work (e.g., UpdateNodeActivePeriod)
    let id_field = schema_builder.add_bytes_field("id", STORED | FAST | INDEXED);
    let src_id_field = schema_builder.add_bytes_field("src_id", STORED | FAST | INDEXED);
    let dst_id_field = schema_builder.add_bytes_field("dst_id", STORED | FAST | INDEXED);

    // Name fields - indexed for search
    // NOTE: node_name is NOT stored (retrieve from RocksDB via NodeById)
    // BUT edge_name MUST be stored because it's part of the edge key for deduplication and result construction
    let text_options = TextOptions::default().set_indexing_options(
        TextFieldIndexing::default()
            .set_tokenizer("default")
            .set_index_option(IndexRecordOption::WithFreqsAndPositions),
    );
    let text_options_stored = text_options.clone().set_stored();

    let node_name_field = schema_builder.add_text_field("node_name", text_options.clone());
    let edge_name_field = schema_builder.add_text_field("edge_name", text_options_stored);

    // Content field - indexed for search, NOT stored (retrieve from RocksDB)
    let content_field = schema_builder.add_text_field("content", text_options);

    // Creation timestamp - STORED because it's part of the fragment CfKey for RocksDB lookup
    let creation_timestamp_field =
        schema_builder.add_u64_field("creation_timestamp", INDEXED | STORED | FAST);

    // Validity fields - indexed for range queries, NOT stored (retrieve from RocksDB)
    let valid_since_field = schema_builder.add_u64_field("valid_since", INDEXED | FAST);
    let valid_until_field = schema_builder.add_u64_field("valid_until", INDEXED | FAST);

    // Document type - STORED to know which RocksDB column family to query
    let doc_type_field = schema_builder.add_text_field("doc_type", STRING | STORED);

    // Weight field - indexed for range queries, NOT stored (retrieve from RocksDB)
    let weight_field = schema_builder.add_f64_field("weight", INDEXED | FAST);

    // Facet fields - indexed for filtering, NOT stored (can be reconstructed from RocksDB data)
    let doc_type_facet = schema_builder.add_facet_field("doc_type_facet", INDEXED);
    let tags_facet = schema_builder.add_facet_field("tags", INDEXED);
    let validity_facet = schema_builder.add_facet_field("validity_facet", INDEXED);

    let schema = schema_builder.build();

    let fields = DocumentFields {
        id_field,
        src_id_field,
        dst_id_field,
        node_name_field,
        edge_name_field,
        content_field,
        creation_timestamp_field,
        valid_since_field,
        valid_until_field,
        doc_type_field,
        weight_field,
        doc_type_facet,
        tags_facet,
        validity_facet,
    };

    (schema, fields)
}

// ============================================================================
// Tag Extraction for User-Defined Facets
// ============================================================================

/// Extract hashtags from content for user-defined facets.
///
/// Supports formats:
/// - `#tag` - simple tag
/// - `#multi_word_tag` - underscores allowed
/// - `#this-is-a-tag` - hyphens allowed
/// - `#path/to/tag` - slashes allowed for hierarchical tags
/// - `#CamelCaseTag` - case is preserved
///
/// Tags preserve their original case (no lowercasing).
///
/// # Example
/// ```
/// use motlie_db::fulltext::schema::extract_tags;
///
/// let tags = extract_tags("This is about #Rust and #systems_programming");
/// assert_eq!(tags, vec!["Rust", "systems_programming"]);
///
/// let tags = extract_tags("#this-is-valid #path/to/tag");
/// assert_eq!(tags, vec!["this-is-valid", "path/to/tag"]);
/// ```
pub fn extract_tags(content: &str) -> Vec<String> {
    let mut tags = Vec::new();
    let mut chars = content.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '#' {
            let mut tag = String::new();

            // Collect tag characters (alphanumeric, underscore, hyphen, slash)
            while let Some(&next_ch) = chars.peek() {
                if next_ch.is_alphanumeric() || next_ch == '_' || next_ch == '-' || next_ch == '/'
                {
                    tag.push(chars.next().unwrap());
                } else {
                    break;
                }
            }

            if !tag.is_empty() {
                tags.push(tag);
            }
        }
    }

    tags
}

// ============================================================================
// Facet Helper Functions
// ============================================================================

/// Compute validity facet from temporal range.
///
/// This facet captures the **structure** of the temporal constraint, not its current state.
/// Time-relative validity (expired, not yet valid, currently valid) must be computed at
/// query time using range queries on `valid_since_field` and `valid_until_field`.
///
/// Validity facets are categorized as:
/// - `/validity/unbounded` - No temporal constraints (always valid)
/// - `/validity/since_only` - Has start time but no end time (valid from X onwards)
/// - `/validity/until_only` - Has end time but no start time (valid until X)
/// - `/validity/bounded` - Has both start and end times (valid from X to Y)
pub fn compute_validity_facet(temporal_range: &Option<crate::ActivePeriod>) -> Facet {
    match temporal_range {
        None => Facet::from("/validity/unbounded"),
        Some(crate::ActivePeriod(since, until)) => match (since, until) {
            (None, None) => Facet::from("/validity/unbounded"),
            (Some(_), None) => Facet::from("/validity/since_only"),
            (None, Some(_)) => Facet::from("/validity/until_only"),
            (Some(_), Some(_)) => Facet::from("/validity/bounded"),
        },
    }
}

// ============================================================================
// Schema - SubsystemProvider + FulltextSubsystem Implementation
// ============================================================================

use crate::SubsystemProvider;
use super::{format_bytes, FulltextSubsystem};
use motlie_core::telemetry::SubsystemInfo;

/// Fulltext module schema implementing the subsystem provider traits.
///
/// Implements:
/// - [`SubsystemInfo`] - Identity and observability
/// - [`SubsystemProvider<tantivy::Index>`] - Lifecycle hooks
/// - [`FulltextSubsystem`] - Tantivy-specific schema and configuration
///
/// This enables the fulltext module to register its Tantivy schema with
/// StorageBuilder for unified initialization.
///
/// # Example
///
/// ```ignore
/// use motlie_db::fulltext::Schema;
/// use motlie_db::storage_builder::StorageBuilder;
///
/// let fulltext_schema = Schema::new();
///
/// // Get Tantivy schema and fields
/// let tantivy_schema = fulltext_schema.tantivy_schema();
/// let fields = fulltext_schema.fields();
/// ```
pub struct Schema {
    /// Cached Tantivy schema
    tantivy_schema: schema::Schema,
    /// Field handles for document construction
    fields: DocumentFields,
    /// Writer heap size (default 50MB)
    writer_heap_size: usize,
}

impl Schema {
    /// Default writer heap size (50MB)
    pub const DEFAULT_WRITER_HEAP_SIZE: usize = 50_000_000;

    /// Create a new Schema with default settings.
    pub fn new() -> Self {
        let (tantivy_schema, fields) = build_schema();
        Self {
            tantivy_schema,
            fields,
            writer_heap_size: Self::DEFAULT_WRITER_HEAP_SIZE,
        }
    }

    /// Set custom writer heap size.
    pub fn with_writer_heap_size(mut self, size: usize) -> Self {
        self.writer_heap_size = size;
        self
    }

    /// Get the Tantivy schema.
    pub fn tantivy_schema(&self) -> &schema::Schema {
        &self.tantivy_schema
    }

    /// Get the field handles.
    pub fn fields(&self) -> &DocumentFields {
        &self.fields
    }
}

impl Default for Schema {
    fn default() -> Self {
        Self::new()
    }
}

// ----------------------------------------------------------------------------
// SubsystemInfo Implementation (from motlie_core::telemetry)
// ----------------------------------------------------------------------------

impl SubsystemInfo for Schema {
    fn name(&self) -> &'static str {
        "Fulltext Search (Tantivy)"
    }

    fn info_lines(&self) -> Vec<(&'static str, String)> {
        vec![
            ("Writer Heap Size", format_bytes(self.writer_heap_size)),
            ("Stored Fields", "true".to_string()),
            ("BM25 Scoring", "true".to_string()),
            ("Faceted Search", "true".to_string()),
            ("Fuzzy Search", "true".to_string()),
        ]
    }
}

// ----------------------------------------------------------------------------
// SubsystemProvider<tantivy::Index> Implementation
// ----------------------------------------------------------------------------

impl SubsystemProvider<tantivy::Index> for Schema {
    fn on_ready(&self, _index: &tantivy::Index) -> anyhow::Result<()> {
        tracing::info!(subsystem = "fulltext", "Index ready");
        Ok(())
    }

    fn on_shutdown(&self) -> anyhow::Result<()> {
        tracing::info!(subsystem = "fulltext", "Shutting down");
        Ok(())
    }
}

// ----------------------------------------------------------------------------
// FulltextSubsystem Implementation
// ----------------------------------------------------------------------------

impl FulltextSubsystem for Schema {
    fn id(&self) -> &'static str {
        "fulltext"
    }

    fn schema(&self) -> schema::Schema {
        self.tantivy_schema.clone()
    }

    fn writer_heap_size(&self) -> usize {
        self.writer_heap_size
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_schema() {
        let (schema, fields) = build_schema();

        // Verify all fields exist in schema
        assert!(schema.get_field("id").is_ok());
        assert!(schema.get_field("src_id").is_ok());
        assert!(schema.get_field("dst_id").is_ok());
        assert!(schema.get_field("node_name").is_ok());
        assert!(schema.get_field("edge_name").is_ok());
        assert!(schema.get_field("content").is_ok());
        assert!(schema.get_field("creation_timestamp").is_ok());
        assert!(schema.get_field("valid_since").is_ok());
        assert!(schema.get_field("valid_until").is_ok());
        assert!(schema.get_field("doc_type").is_ok());
        assert!(schema.get_field("weight").is_ok());
        assert!(schema.get_field("doc_type_facet").is_ok());
        assert!(schema.get_field("tags").is_ok());
        assert!(schema.get_field("validity_facet").is_ok());

        // Verify field handles match schema fields
        assert_eq!(fields.id_field, schema.get_field("id").unwrap());
        assert_eq!(fields.content_field, schema.get_field("content").unwrap());
    }

    // ========================================================================
    // Tag Extraction Tests - Document Supported Tag Formats
    // ========================================================================
    //
    // Tags support the following characters after the # symbol:
    // - Alphanumeric (a-z, A-Z, 0-9)
    // - Underscores (_)
    // - Hyphens (-)
    // - Forward slashes (/) for hierarchical tags
    //
    // Case is preserved (no lowercasing).
    // Tags are terminated by any other character (space, punctuation, etc.)

    #[test]
    fn test_extract_tags_simple_lowercase() {
        let tags = extract_tags("This is about #rust and #programming");
        assert_eq!(tags, vec!["rust", "programming"]);
    }

    #[test]
    fn test_extract_tags_with_numbers() {
        let tags = extract_tags("Version #v2 and #release123");
        assert_eq!(tags, vec!["v2", "release123"]);
    }

    #[test]
    fn test_extract_tags_underscore_separator() {
        // Underscores connect words within a tag
        let tags = extract_tags("#systems_programming #low_level_code #my_tag_123");
        assert_eq!(tags, vec!["systems_programming", "low_level_code", "my_tag_123"]);
    }

    #[test]
    fn test_extract_tags_hyphen_separator() {
        // Hyphens connect words within a tag (kebab-case)
        let tags = extract_tags("#this-is-valid #another-tag #my-tag-123");
        assert_eq!(tags, vec!["this-is-valid", "another-tag", "my-tag-123"]);
    }

    #[test]
    fn test_extract_tags_hierarchical_with_slashes() {
        // Slashes create hierarchical tag paths
        let tags = extract_tags("#topic/subtopic #lang/rust/async #a/b/c/d");
        assert_eq!(tags, vec!["topic/subtopic", "lang/rust/async", "a/b/c/d"]);
    }

    #[test]
    fn test_extract_tags_case_preserved() {
        // Case is preserved exactly as written
        let tags = extract_tags("#Rust #ALLCAPS #CamelCase #mixedCase");
        assert_eq!(tags, vec!["Rust", "ALLCAPS", "CamelCase", "mixedCase"]);
    }

    #[test]
    fn test_extract_tags_mixed_separators() {
        // Can mix underscores, hyphens, and slashes
        let tags = extract_tags("#my_tag-with/mixed_separators #a-b_c/d");
        assert_eq!(tags, vec!["my_tag-with/mixed_separators", "a-b_c/d"]);
    }

    #[test]
    fn test_extract_tags_adjacent_no_space() {
        // Tags can be adjacent without spaces
        let tags = extract_tags("#one#two#three");
        assert_eq!(tags, vec!["one", "two", "three"]);
    }

    #[test]
    fn test_extract_tags_terminated_by_punctuation() {
        // Tags end at punctuation
        let tags = extract_tags("#tag. #tag, #tag! #tag? #tag:");
        assert_eq!(tags, vec!["tag", "tag", "tag", "tag", "tag"]);
    }

    #[test]
    fn test_extract_tags_terminated_by_parentheses() {
        let tags = extract_tags("(#tag) [#another] {#third}");
        assert_eq!(tags, vec!["tag", "another", "third"]);
    }

    #[test]
    fn test_extract_tags_in_sentence() {
        let tags = extract_tags("Check out #rust-lang for #systems/programming info.");
        assert_eq!(tags, vec!["rust-lang", "systems/programming"]);
    }

    #[test]
    fn test_extract_tags_empty_content() {
        let tags = extract_tags("");
        assert!(tags.is_empty());
    }

    #[test]
    fn test_extract_tags_no_tags() {
        let tags = extract_tags("No hashtags here, just plain text.");
        assert!(tags.is_empty());
    }

    #[test]
    fn test_extract_tags_empty_hash_ignored() {
        // A lone # followed by space or end is not a tag
        let tags = extract_tags("Invalid # tag and #valid here");
        assert_eq!(tags, vec!["valid"]);
    }

    #[test]
    fn test_extract_tags_hash_at_end() {
        let tags = extract_tags("Text ending with #");
        assert!(tags.is_empty());
    }

    #[test]
    fn test_extract_tags_unicode_alphanumeric() {
        // Unicode letters are alphanumeric
        let tags = extract_tags("#café #日本語 #über");
        assert_eq!(tags, vec!["café", "日本語", "über"]);
    }

    #[test]
    fn test_compute_validity_facet_none() {
        assert_eq!(
            compute_validity_facet(&None).to_string(),
            "/validity/unbounded"
        );
    }

    #[test]
    fn test_compute_validity_facet_empty_range() {
        let range = Some(crate::ActivePeriod(None, None));
        assert_eq!(
            compute_validity_facet(&range).to_string(),
            "/validity/unbounded"
        );
    }

    #[test]
    fn test_compute_validity_facet_since_only() {
        let ts = crate::TimestampMilli(1000);
        let range = Some(crate::ActivePeriod(Some(ts), None));
        assert_eq!(
            compute_validity_facet(&range).to_string(),
            "/validity/since_only"
        );
    }

    #[test]
    fn test_compute_validity_facet_until_only() {
        let ts = crate::TimestampMilli(1000);
        let range = Some(crate::ActivePeriod(None, Some(ts)));
        assert_eq!(
            compute_validity_facet(&range).to_string(),
            "/validity/until_only"
        );
    }

    #[test]
    fn test_compute_validity_facet_bounded() {
        let start = crate::TimestampMilli(1000);
        let end = crate::TimestampMilli(2000);
        let range = Some(crate::ActivePeriod(Some(start), Some(end)));
        assert_eq!(
            compute_validity_facet(&range).to_string(),
            "/validity/bounded"
        );
    }
}
