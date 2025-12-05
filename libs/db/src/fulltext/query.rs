//! Fulltext query module providing search queries against Tantivy index.
//!
//! This module contains only business logic - query type definitions and their
//! QueryExecutor implementations. Infrastructure (traits, Reader, Consumer, spawn
//! functions) is in the `reader` module.
//!
//! # Fuzzy Search
//!
//! Both `Nodes` and `Edges` queries support fuzzy matching via `FuzzyLevel`:
//! - `FuzzyLevel::None` - Exact matching only (default)
//! - `FuzzyLevel::Low` - Allow 1 character edit (typos in short words)
//! - `FuzzyLevel::Medium` - Allow 2 character edits (more forgiving)
//!
//! Example:
//! ```ignore
//! // Exact search
//! Nodes::new("rust".to_string(), 10)
//!
//! // Fuzzy search with builder
//! Nodes::new("rast".to_string(), 10).with_fuzzy(FuzzyLevel::Low)
//! ```

use std::time::Duration;

use anyhow::Result;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::Value;
use tokio::sync::oneshot;

use super::reader::{Processor, QueryExecutor, QueryProcessor, Reader};
use super::search::{EdgeHit, NodeHit};
use super::Storage;
use crate::Id;

// ============================================================================
// Fuzzy Search Level
// ============================================================================

/// Fuzzy search level determining match strictness.
///
/// Uses Levenshtein distance to find matches even with spelling errors:
/// - "Rast" matches "Rust" with `Low` (1 character difference)
/// - "performence" matches "performance" with `Medium` (2 character difference)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FuzzyLevel {
    /// No fuzzy matching - exact match only (default)
    #[default]
    None,

    /// Allow 1 character edit (insert, delete, substitute, or transpose).
    /// Good for catching typos in short words.
    Low,

    /// Allow 2 character edits.
    /// More forgiving, good for longer words.
    Medium,
}

impl FuzzyLevel {
    /// Convert fuzzy level to Levenshtein edit distance
    pub fn to_distance(self) -> u8 {
        match self {
            FuzzyLevel::None => 0,
            FuzzyLevel::Low => 1,
            FuzzyLevel::Medium => 2,
        }
    }
}

// ============================================================================
// Query Enum
// ============================================================================

/// Query enum representing all possible fulltext query types
#[derive(Debug)]
pub enum Search {
    Nodes(Nodes),
    Edges(Edges),
    Facets(Facets),
}

impl std::fmt::Display for Search {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Search::Nodes(q) => write!(f, "FulltextNodes: query={}, limit={}", q.query, q.limit),
            Search::Edges(q) => write!(f, "FulltextEdges: query={}, limit={}", q.query, q.limit),
            Search::Facets(q) => write!(
                f,
                "FulltextFacets: doc_type_filter={:?}, tags_limit={}",
                q.doc_type_filter, q.tags_limit
            ),
        }
    }
}

// ============================================================================
// Runnable Trait
// ============================================================================

/// Trait for query builders that can be executed via a Reader.
/// This matches the graph query pattern for client-side API.
#[async_trait::async_trait]
pub trait Runnable {
    /// The output type this query produces
    type Output: Send + 'static;

    /// Execute this query against a FulltextReader with the specified timeout
    async fn run(self, reader: &Reader, timeout: Duration) -> Result<Self::Output>;
}

// ============================================================================
// Nodes Query - Search for nodes by text
// ============================================================================

/// Query to search for nodes by fulltext query, returning top K results.
/// This is the fulltext equivalent of searching nodes.
///
/// Supports:
/// - Fuzzy matching via `with_fuzzy()` builder method
/// - Tag filtering via `with_tags()` builder method
#[derive(Debug)]
pub struct Nodes {
    /// The search query string
    pub query: String,

    /// Maximum number of results to return
    pub limit: usize,

    /// Fuzzy matching level (default: None = exact match)
    pub fuzzy_level: FuzzyLevel,

    /// Filter by tags (documents must have ANY of the specified tags)
    pub tags: Vec<String>,

    /// Timeout for this query execution (only set when query has channel)
    pub(crate) timeout: Option<Duration>,

    /// Channel to send the result back to the client (only set when ready to execute)
    result_tx: Option<oneshot::Sender<Result<Vec<NodeHit>>>>,
}

impl Nodes {
    /// Create a new query request (public API - no channel, no timeout yet)
    /// Use `.run(reader, timeout)` to execute this query
    pub fn new(query: String, limit: usize) -> Self {
        Self {
            query,
            limit,
            fuzzy_level: FuzzyLevel::None,
            tags: Vec::new(),
            timeout: None,
            result_tx: None,
        }
    }

    /// Enable fuzzy matching with the specified level.
    ///
    /// Example:
    /// ```ignore
    /// Nodes::new("rast".to_string(), 10).with_fuzzy(FuzzyLevel::Low)
    /// ```
    pub fn with_fuzzy(mut self, level: FuzzyLevel) -> Self {
        self.fuzzy_level = level;
        self
    }

    /// Filter results to only include documents with ANY of the specified tags.
    ///
    /// Tags are extracted from content using #hashtag syntax during indexing.
    ///
    /// Example:
    /// ```ignore
    /// Nodes::new("programming".to_string(), 10)
    ///     .with_tags(vec!["rust".to_string(), "async".to_string()])
    /// ```
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Internal constructor used by the query execution machinery (has the channel)
    pub(crate) fn with_channel(
        query: String,
        limit: usize,
        fuzzy_level: FuzzyLevel,
        tags: Vec<String>,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<Vec<NodeHit>>>,
    ) -> Self {
        Self {
            query,
            limit,
            fuzzy_level,
            tags,
            timeout: Some(timeout),
            result_tx: Some(result_tx),
        }
    }

    /// Send the result back to the client (consumes self)
    pub(crate) fn send_result(self, result: Result<Vec<NodeHit>>) {
        // Ignore error if receiver was dropped (client timeout/cancellation)
        if let Some(tx) = self.result_tx {
            let _ = tx.send(result);
        }
    }
}

#[async_trait::async_trait]
impl Runnable for Nodes {
    type Output = Vec<NodeHit>;

    async fn run(self, reader: &Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = oneshot::channel();

        let query = Nodes::with_channel(
            self.query,
            self.limit,
            self.fuzzy_level,
            self.tags,
            timeout,
            result_tx,
        );

        reader.send_query(Search::Nodes(query)).await?;

        // Wait for result with timeout
        match tokio::time::timeout(timeout, result_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("Query channel closed")),
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
        }
    }
}

#[async_trait::async_trait]
impl QueryExecutor for Nodes {
    type Output = Vec<NodeHit>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        use std::collections::HashMap;
        use tantivy::query::{BooleanQuery, FuzzyTermQuery, Occur, TermQuery};
        use tantivy::schema::IndexRecordOption;
        use tantivy::Term;

        let index = storage.index()?;
        let fields = storage.fields()?;

        // Create a reader for searching
        let reader = index
            .reader()
            .map_err(|e| anyhow::anyhow!("Failed to create index reader: {}", e))?;
        let searcher = reader.searcher();

        // Build text query - either exact (QueryParser) or fuzzy (FuzzyTermQuery)
        let text_query: Box<dyn tantivy::query::Query> = if self.fuzzy_level == FuzzyLevel::None {
            // Exact match: use QueryParser for full query syntax support
            let query_parser =
                QueryParser::for_index(index, vec![fields.content_field, fields.node_name_field]);
            let parsed = query_parser
                .parse_query(&self.query)
                .map_err(|e| anyhow::anyhow!("Failed to parse query '{}': {}", self.query, e))?;
            parsed
        } else {
            // Fuzzy match: build FuzzyTermQuery for each term in both fields
            let distance = self.fuzzy_level.to_distance();
            let terms: Vec<&str> = self.query.split_whitespace().collect();

            if terms.is_empty() {
                return Ok(vec![]);
            }

            // For each term, create fuzzy queries against both content and node_name fields
            let mut term_queries: Vec<(Occur, Box<dyn tantivy::query::Query>)> = Vec::new();

            for term_str in terms {
                let term_lower = term_str.to_lowercase();

                // Fuzzy query on content field
                let content_term = Term::from_field_text(fields.content_field, &term_lower);
                let content_fuzzy = FuzzyTermQuery::new(content_term, distance, true);

                // Fuzzy query on node_name field
                let name_term = Term::from_field_text(fields.node_name_field, &term_lower);
                let name_fuzzy = FuzzyTermQuery::new(name_term, distance, true);

                // Either field can match for this term
                let term_query = BooleanQuery::new(vec![
                    (Occur::Should, Box::new(content_fuzzy)),
                    (Occur::Should, Box::new(name_fuzzy)),
                ]);

                // All terms must match (AND semantics)
                term_queries.push((Occur::Must, Box::new(term_query)));
            }

            Box::new(BooleanQuery::new(term_queries))
        };

        // Build doc_type filter: (doc_type = "nodes" OR doc_type = "node_fragments")
        let nodes_term = Term::from_field_text(fields.doc_type_field, "nodes");
        let fragments_term = Term::from_field_text(fields.doc_type_field, "node_fragments");

        let doc_type_filter = BooleanQuery::new(vec![
            (
                Occur::Should,
                Box::new(TermQuery::new(nodes_term, IndexRecordOption::Basic)),
            ),
            (
                Occur::Should,
                Box::new(TermQuery::new(fragments_term, IndexRecordOption::Basic)),
            ),
        ]);

        // Build combined query with all filters
        let mut query_clauses: Vec<(Occur, Box<dyn tantivy::query::Query>)> = vec![
            (Occur::Must, text_query),
            (Occur::Must, Box::new(doc_type_filter)),
        ];

        // Add tag filters if specified (ANY tag must match - OR semantics)
        // Tags are stored as /tag/{name}, so we construct the facet path accordingly
        if !self.tags.is_empty() {
            let mut tag_clauses: Vec<(Occur, Box<dyn tantivy::query::Query>)> = Vec::new();
            for tag in &self.tags {
                let facet = tantivy::schema::Facet::from(&format!("/tag/{}", tag));
                let tag_term = Term::from_facet(fields.tags_facet, &facet);
                tag_clauses.push((
                    Occur::Should,
                    Box::new(TermQuery::new(tag_term, IndexRecordOption::Basic)),
                ));
            }
            query_clauses.push((Occur::Must, Box::new(BooleanQuery::new(tag_clauses))));
        }

        let combined_query = BooleanQuery::new(query_clauses);

        // Search with TopDocs collector - get extra results since we'll dedupe by node ID
        let top_docs = searcher
            .search(&combined_query, &TopDocs::with_limit(self.limit * 3))
            .map_err(|e| anyhow::anyhow!("Search failed: {}", e))?;

        // Collect results, deduplicating by node ID and keeping best score
        // This handles both node documents and node_fragment documents
        let mut node_scores: HashMap<Id, f32> = HashMap::new();

        for (score, doc_address) in top_docs {
            let doc = searcher
                .doc::<tantivy::TantivyDocument>(doc_address)
                .map_err(|e| anyhow::anyhow!("Failed to retrieve document: {}", e))?;

            // Extract node ID
            let id = match doc.get_first(fields.id_field).and_then(|v| v.as_bytes()) {
                Some(bytes) => match Id::from_slice(bytes) {
                    Ok(id) => id,
                    Err(_) => continue, // Invalid ID format
                },
                None => continue, // No ID field
            };

            // Update with best score
            node_scores
                .entry(id)
                .and_modify(|existing_score| {
                    if score > *existing_score {
                        *existing_score = score;
                    }
                })
                .or_insert(score);
        }

        // Convert to results and sort by score
        let mut results: Vec<NodeHit> = node_scores
            .into_iter()
            .map(|(id, score)| NodeHit {
                score,
                id,
                fragment_timestamp: None, // Deduplicated results don't track individual fragments
                snippet: None,
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(self.limit);

        Ok(results)
    }

    fn timeout(&self) -> Duration {
        self.timeout
            .expect("Query must have timeout set when executing")
    }
}

// Use macro to implement QueryProcessor for query types
crate::impl_fulltext_query_processor!(Nodes);

// ============================================================================
// Edges Query - Search for edges by text
// ============================================================================

/// Query to search for edges by fulltext query, returning top K results.
/// This is the fulltext equivalent of searching edges.
///
/// Supports:
/// - Fuzzy matching via `with_fuzzy()` builder method
/// - Tag filtering via `with_tags()` builder method
#[derive(Debug)]
pub struct Edges {
    /// The search query string
    pub query: String,

    /// Maximum number of results to return
    pub limit: usize,

    /// Fuzzy matching level (default: None = exact match)
    pub fuzzy_level: FuzzyLevel,

    /// Filter by tags (documents must have ANY of the specified tags)
    pub tags: Vec<String>,

    /// Timeout for this query execution (only set when query has channel)
    pub(crate) timeout: Option<Duration>,

    /// Channel to send the result back to the client (only set when ready to execute)
    result_tx: Option<oneshot::Sender<Result<Vec<EdgeHit>>>>,
}

impl Edges {
    /// Create a new query request (public API - no channel, no timeout yet)
    /// Use `.run(reader, timeout)` to execute this query
    pub fn new(query: String, limit: usize) -> Self {
        Self {
            query,
            limit,
            fuzzy_level: FuzzyLevel::None,
            tags: Vec::new(),
            timeout: None,
            result_tx: None,
        }
    }

    /// Enable fuzzy matching with the specified level.
    ///
    /// Example:
    /// ```ignore
    /// Edges::new("dependz_on".to_string(), 10).with_fuzzy(FuzzyLevel::Low)
    /// ```
    pub fn with_fuzzy(mut self, level: FuzzyLevel) -> Self {
        self.fuzzy_level = level;
        self
    }

    /// Filter results to only include documents with ANY of the specified tags.
    ///
    /// Tags are extracted from content using #hashtag syntax during indexing.
    ///
    /// Example:
    /// ```ignore
    /// Edges::new("relationship".to_string(), 10)
    ///     .with_tags(vec!["important".to_string()])
    /// ```
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Internal constructor used by the query execution machinery (has the channel)
    pub(crate) fn with_channel(
        query: String,
        limit: usize,
        fuzzy_level: FuzzyLevel,
        tags: Vec<String>,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<Vec<EdgeHit>>>,
    ) -> Self {
        Self {
            query,
            limit,
            fuzzy_level,
            tags,
            timeout: Some(timeout),
            result_tx: Some(result_tx),
        }
    }

    /// Send the result back to the client (consumes self)
    pub(crate) fn send_result(self, result: Result<Vec<EdgeHit>>) {
        // Ignore error if receiver was dropped (client timeout/cancellation)
        if let Some(tx) = self.result_tx {
            let _ = tx.send(result);
        }
    }
}

#[async_trait::async_trait]
impl Runnable for Edges {
    type Output = Vec<EdgeHit>;

    async fn run(self, reader: &Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = oneshot::channel();

        let query = Edges::with_channel(
            self.query,
            self.limit,
            self.fuzzy_level,
            self.tags,
            timeout,
            result_tx,
        );

        reader.send_query(Search::Edges(query)).await?;

        // Wait for result with timeout
        match tokio::time::timeout(timeout, result_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("Query channel closed")),
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
        }
    }
}

#[async_trait::async_trait]
impl QueryExecutor for Edges {
    type Output = Vec<EdgeHit>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        use std::collections::HashMap;
        use tantivy::query::{BooleanQuery, FuzzyTermQuery, Occur, TermQuery};
        use tantivy::schema::IndexRecordOption;
        use tantivy::Term;

        let index = storage.index()?;
        let fields = storage.fields()?;

        // Create a reader for searching
        let reader = index
            .reader()
            .map_err(|e| anyhow::anyhow!("Failed to create index reader: {}", e))?;
        let searcher = reader.searcher();

        log::debug!(
            "[FulltextEdges] Executing query='{}', index docs={}, fields content={:?} edge_name={:?} doc_type={:?}",
            self.query,
            searcher.num_docs(),
            fields.content_field,
            fields.edge_name_field,
            fields.doc_type_field
        );

        // Build text query - either exact (QueryParser) or fuzzy (FuzzyTermQuery)
        let text_query: Box<dyn tantivy::query::Query> = if self.fuzzy_level == FuzzyLevel::None {
            // Exact match: use QueryParser for full query syntax support
            let query_parser =
                QueryParser::for_index(index, vec![fields.content_field, fields.edge_name_field]);
            let parsed = query_parser
                .parse_query(&self.query)
                .map_err(|e| anyhow::anyhow!("Failed to parse query '{}': {}", self.query, e))?;
            parsed
        } else {
            // Fuzzy match: build FuzzyTermQuery for each term in both fields
            let distance = self.fuzzy_level.to_distance();
            let terms: Vec<&str> = self.query.split_whitespace().collect();

            if terms.is_empty() {
                return Ok(vec![]);
            }

            // For each term, create fuzzy queries against both content and edge_name fields
            let mut term_queries: Vec<(Occur, Box<dyn tantivy::query::Query>)> = Vec::new();

            for term_str in terms {
                let term_lower = term_str.to_lowercase();

                // Fuzzy query on content field
                let content_term = Term::from_field_text(fields.content_field, &term_lower);
                let content_fuzzy = FuzzyTermQuery::new(content_term, distance, true);

                // Fuzzy query on edge_name field
                let name_term = Term::from_field_text(fields.edge_name_field, &term_lower);
                let name_fuzzy = FuzzyTermQuery::new(name_term, distance, true);

                // Either field can match for this term
                let term_query = BooleanQuery::new(vec![
                    (Occur::Should, Box::new(content_fuzzy)),
                    (Occur::Should, Box::new(name_fuzzy)),
                ]);

                // All terms must match (AND semantics)
                term_queries.push((Occur::Must, Box::new(term_query)));
            }

            Box::new(BooleanQuery::new(term_queries))
        };

        // Build doc_type filter: (doc_type = "edges" OR doc_type = "edge_fragments")
        let edges_term = Term::from_field_text(fields.doc_type_field, "edges");
        let fragments_term = Term::from_field_text(fields.doc_type_field, "edge_fragments");

        let doc_type_filter = BooleanQuery::new(vec![
            (
                Occur::Should,
                Box::new(TermQuery::new(edges_term, IndexRecordOption::Basic)),
            ),
            (
                Occur::Should,
                Box::new(TermQuery::new(fragments_term, IndexRecordOption::Basic)),
            ),
        ]);

        // Build combined query with all filters
        let mut query_clauses: Vec<(Occur, Box<dyn tantivy::query::Query>)> = vec![
            (Occur::Must, text_query),
            (Occur::Must, Box::new(doc_type_filter)),
        ];

        // Add tag filters if specified (ANY tag must match - OR semantics)
        // Tags are stored as /tag/{name}, so we construct the facet path accordingly
        if !self.tags.is_empty() {
            let mut tag_clauses: Vec<(Occur, Box<dyn tantivy::query::Query>)> = Vec::new();
            for tag in &self.tags {
                let facet = tantivy::schema::Facet::from(&format!("/tag/{}", tag));
                let tag_term = Term::from_facet(fields.tags_facet, &facet);
                tag_clauses.push((
                    Occur::Should,
                    Box::new(TermQuery::new(tag_term, IndexRecordOption::Basic)),
                ));
            }
            query_clauses.push((Occur::Must, Box::new(BooleanQuery::new(tag_clauses))));
        }

        let combined_query = BooleanQuery::new(query_clauses);

        // Search with TopDocs collector - get extra results since we'll dedupe by edge key
        let top_docs = searcher
            .search(&combined_query, &TopDocs::with_limit(self.limit * 3))
            .map_err(|e| anyhow::anyhow!("Search failed: {}", e))?;

        log::debug!(
            "[FulltextEdges] Search returned {} raw results for query='{}'",
            top_docs.len(),
            self.query
        );

        // Collect results, deduplicating by edge key (src_id, dst_id, edge_name) and keeping best score
        // Edge key is (src_id, dst_id, edge_name)
        let mut edge_scores: HashMap<(Id, Id, String), f32> = HashMap::new();

        for (score, doc_address) in top_docs {
            let doc = searcher
                .doc::<tantivy::TantivyDocument>(doc_address)
                .map_err(|e| anyhow::anyhow!("Failed to retrieve document: {}", e))?;

            // Extract edge IDs
            let src_id = match doc
                .get_first(fields.src_id_field)
                .and_then(|v| v.as_bytes())
            {
                Some(bytes) => match Id::from_slice(bytes) {
                    Ok(id) => id,
                    Err(_) => continue, // Invalid ID format
                },
                None => continue, // No src_id field
            };

            let dst_id = match doc
                .get_first(fields.dst_id_field)
                .and_then(|v| v.as_bytes())
            {
                Some(bytes) => match Id::from_slice(bytes) {
                    Ok(id) => id,
                    Err(_) => continue, // Invalid ID format
                },
                None => continue, // No dst_id field
            };

            // Extract edge name
            let edge_name = doc
                .get_first(fields.edge_name_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            // Skip if no edge name (shouldn't happen for valid edge docs)
            if edge_name.is_empty() {
                continue;
            }

            // Update with best score
            let key = (src_id, dst_id, edge_name);
            edge_scores
                .entry(key)
                .and_modify(|existing_score| {
                    if score > *existing_score {
                        *existing_score = score;
                    }
                })
                .or_insert(score);
        }

        // Convert to results and sort by score
        let mut results: Vec<EdgeHit> = edge_scores
            .into_iter()
            .map(|((src_id, dst_id, edge_name), score)| EdgeHit {
                score,
                src_id,
                dst_id,
                edge_name,
                fragment_timestamp: None, // Deduplicated results don't track individual fragments
                snippet: None,
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(self.limit);

        Ok(results)
    }

    fn timeout(&self) -> Duration {
        self.timeout
            .expect("Query must have timeout set when executing")
    }
}

crate::impl_fulltext_query_processor!(Edges);

// ============================================================================
// Facets Query - Get facet counts from the index
// ============================================================================

/// Query to retrieve facet counts from the fulltext index.
///
/// This query does not perform text search - it aggregates facet values
/// across all documents or a filtered subset to provide counts for:
/// - Document types (nodes, edges, fragments)
/// - User-defined tags (from #hashtags)
/// - Validity structure (unbounded, bounded, since_only, until_only)
///
/// # Example
/// ```ignore
/// // Get all facet counts
/// let counts = Facets::new()
///     .run(&reader, Duration::from_secs(5))
///     .await?;
///
/// // Get facet counts for documents matching a filter
/// let counts = Facets::new()
///     .with_doc_type_filter(vec!["nodes".to_string()])
///     .run(&reader, Duration::from_secs(5))
///     .await?;
/// ```
#[derive(Debug)]
pub struct Facets {
    /// Optional filter by document types (if empty, count all)
    pub doc_type_filter: Vec<String>,

    /// Limit for number of tag facets to return (default: 100)
    pub tags_limit: usize,

    /// Timeout for this query execution (only set when query has channel)
    pub(crate) timeout: Option<Duration>,

    /// Channel to send the result back to the client (only set when ready to execute)
    result_tx: Option<oneshot::Sender<Result<super::search::FacetCounts>>>,
}

impl Facets {
    /// Create a new facets query (counts all facets across all documents)
    pub fn new() -> Self {
        Self {
            doc_type_filter: Vec::new(),
            tags_limit: 100,
            timeout: None,
            result_tx: None,
        }
    }

    /// Filter facet counts to only include documents of specific types.
    ///
    /// Valid doc types: "nodes", "edges", "node_fragments", "edge_fragments"
    ///
    /// Example:
    /// ```ignore
    /// Facets::new().with_doc_type_filter(vec!["nodes".to_string()])
    /// ```
    pub fn with_doc_type_filter(mut self, doc_types: Vec<String>) -> Self {
        self.doc_type_filter = doc_types;
        self
    }

    /// Set the maximum number of tag facets to return (default: 100).
    ///
    /// Example:
    /// ```ignore
    /// Facets::new().with_tags_limit(50)
    /// ```
    pub fn with_tags_limit(mut self, limit: usize) -> Self {
        self.tags_limit = limit;
        self
    }

    /// Internal constructor used by the query execution machinery (has the channel)
    pub(crate) fn with_channel(
        doc_type_filter: Vec<String>,
        tags_limit: usize,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<super::search::FacetCounts>>,
    ) -> Self {
        Self {
            doc_type_filter,
            tags_limit,
            timeout: Some(timeout),
            result_tx: Some(result_tx),
        }
    }

    /// Send the result back to the client (consumes self)
    pub(crate) fn send_result(self, result: Result<super::search::FacetCounts>) {
        // Ignore error if receiver was dropped (client timeout/cancellation)
        if let Some(tx) = self.result_tx {
            let _ = tx.send(result);
        }
    }
}

impl Default for Facets {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl Runnable for Facets {
    type Output = super::search::FacetCounts;

    async fn run(self, reader: &Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = oneshot::channel();

        let query = Facets::with_channel(
            self.doc_type_filter,
            self.tags_limit,
            timeout,
            result_tx,
        );

        reader.send_query(Search::Facets(query)).await?;

        // Wait for result with timeout
        match tokio::time::timeout(timeout, result_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("Query channel closed")),
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
        }
    }
}

#[async_trait::async_trait]
impl QueryExecutor for Facets {
    type Output = super::search::FacetCounts;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        use tantivy::collector::FacetCollector;
        use tantivy::query::{AllQuery, BooleanQuery, Occur, TermQuery};
        use tantivy::schema::IndexRecordOption;
        use tantivy::Term;

        let index = storage.index()?;
        let fields = storage.fields()?;

        // Create a reader for searching
        let reader = index
            .reader()
            .map_err(|e| anyhow::anyhow!("Failed to create index reader: {}", e))?;
        let searcher = reader.searcher();

        // Build query - either all docs or filtered by doc_type
        let query: Box<dyn tantivy::query::Query> = if self.doc_type_filter.is_empty() {
            Box::new(AllQuery)
        } else {
            // Build doc_type filter
            let mut doc_type_clauses: Vec<(Occur, Box<dyn tantivy::query::Query>)> = Vec::new();
            for doc_type in &self.doc_type_filter {
                let term = Term::from_field_text(fields.doc_type_field, doc_type);
                doc_type_clauses.push((
                    Occur::Should,
                    Box::new(TermQuery::new(term, IndexRecordOption::Basic)),
                ));
            }
            Box::new(BooleanQuery::new(doc_type_clauses))
        };

        // Create facet collectors for each facet field
        let mut doc_type_collector = FacetCollector::for_field("doc_type_facet");
        doc_type_collector.add_facet(tantivy::schema::Facet::from("/type"));

        let mut tags_collector = FacetCollector::for_field("tags");
        tags_collector.add_facet(tantivy::schema::Facet::from("/tag"));

        let mut validity_collector = FacetCollector::for_field("validity_facet");
        validity_collector.add_facet(tantivy::schema::Facet::from("/validity"));

        // Execute search with all three collectors
        let (doc_type_counts, tags_counts, validity_counts) = searcher
            .search(
                &query,
                &(doc_type_collector, tags_collector, validity_collector),
            )
            .map_err(|e| anyhow::anyhow!("Facet search failed: {}", e))?;

        // Convert to FacetCounts structure
        let mut result = super::search::FacetCounts::new();

        // Extract doc_type facets (/type/nodes, /type/edges, etc.)
        for (facet, count) in doc_type_counts.get("/type") {
            // Extract the facet name (last component)
            let facet_str = facet.to_string();
            if let Some(name) = facet_str.strip_prefix("/type/") {
                result.doc_types.insert(name.to_string(), count);
            }
        }

        // Extract tag facets (/tag/rust, /tag/programming, etc.)
        let mut tag_count = 0;
        for (facet, count) in tags_counts.get("/tag") {
            if tag_count >= self.tags_limit {
                break;
            }
            let facet_str = facet.to_string();
            if let Some(name) = facet_str.strip_prefix("/tag/") {
                result.tags.insert(name.to_string(), count);
                tag_count += 1;
            }
        }

        // Extract validity facets (/validity/unbounded, /validity/bounded, etc.)
        for (facet, count) in validity_counts.get("/validity") {
            let facet_str = facet.to_string();
            if let Some(name) = facet_str.strip_prefix("/validity/") {
                result.validity.insert(name.to_string(), count);
            }
        }

        Ok(result)
    }

    fn timeout(&self) -> Duration {
        self.timeout
            .expect("Query must have timeout set when executing")
    }
}

crate::impl_fulltext_query_processor!(Facets);

#[async_trait::async_trait]
impl QueryProcessor for Search {
    async fn process_and_send<P: Processor + Sync>(self, processor: &P) {
        match self {
            Search::Nodes(q) => q.process_and_send(processor).await,
            Search::Edges(q) => q.process_and_send(processor).await,
            Search::Facets(q) => q.process_and_send(processor).await,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fulltext::reader::{create_query_reader, spawn_consumer, Consumer, ReaderConfig};
    use crate::fulltext::{Index, Storage};
    use crate::graph::mutation::{AddNode, AddNodeFragment, Runnable as MutRunnable};
    use crate::graph::schema::NodeSummary;
    use crate::graph::writer::{create_mutation_writer, WriterConfig};
    use crate::{spawn_fulltext_consumer, DataUrl, TimestampMilli};
    use std::sync::Arc;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_nodes_query_basic() {
        // Create temporary directory for fulltext index
        let temp_dir = TempDir::new().unwrap();
        let index_path = temp_dir.path().join("fulltext_index");

        // Setup mutation writer and fulltext consumer
        let writer_config = WriterConfig {
            channel_buffer_size: 10,
        };
        let (writer, receiver) = create_mutation_writer(writer_config.clone());
        let fulltext_handle = spawn_fulltext_consumer(receiver, writer_config, &index_path);

        // Create test nodes with searchable content
        let node1_id = Id::new();
        let node1 = AddNode {
            id: node1_id,
            ts_millis: TimestampMilli::now(),
            name: "RustLang".to_string(),
            valid_range: None,
            summary: crate::NodeSummary::from_text("Rust language summary"),
        };
        node1.run(&writer).await.unwrap();

        let fragment1 = AddNodeFragment {
            id: node1_id,
            ts_millis: TimestampMilli::now(),
            content: DataUrl::from_text("Rust is a systems programming language"),
            valid_range: None,
        };
        fragment1.run(&writer).await.unwrap();

        let node2_id = Id::new();
        let node2 = AddNode {
            id: node2_id,
            ts_millis: TimestampMilli::now(),
            name: "Python".to_string(),
            valid_range: None,
            summary: crate::NodeSummary::from_text("Python language summary"),
        };
        node2.run(&writer).await.unwrap();

        // Wait for indexing
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Shutdown mutation consumer
        drop(writer);
        fulltext_handle.await.unwrap().unwrap();

        // Now test the query consumer
        let reader_config = ReaderConfig {
            channel_buffer_size: 10,
        };
        let (reader, query_receiver) = create_query_reader(reader_config.clone());

        // Create processor from existing index (readonly for query consumer)
        let mut storage = Storage::readonly(&index_path);
        storage.ready().unwrap();
        let processor = Index::new(Arc::new(storage));
        let consumer = Consumer::new(query_receiver, reader_config, processor);
        let consumer_handle = spawn_consumer(consumer);

        // Search for "Rust"
        let results = Nodes::new("Rust".to_string(), 10)
            .run(&reader, Duration::from_secs(5))
            .await
            .unwrap();

        assert!(!results.is_empty(), "Should find at least one result");

        // Cleanup
        drop(reader);
        consumer_handle.await.unwrap().unwrap();
    }
}
