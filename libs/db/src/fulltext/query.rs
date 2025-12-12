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
use tantivy::query::{BooleanQuery, Occur, QueryParser, RegexQuery};
use tantivy::schema::{Field, Value};
use tokio::sync::oneshot;

use super::reader::{Processor, QueryExecutor, QueryProcessor, Reader};
use super::search::{EdgeHit, MatchSource, NodeHit};
use super::Storage;
use crate::Id;

// ============================================================================
// Wildcard Query Support
// ============================================================================

/// Check if a query term contains wildcard characters (* or ?)
fn has_wildcard(term: &str) -> bool {
    term.contains('*') || term.contains('?')
}

/// Convert a glob-style wildcard pattern to a regex pattern.
/// - `*` becomes `.*` (match any characters)
/// - `?` becomes `.` (match single character)
/// - Other regex special characters are escaped
fn glob_to_regex(glob: &str) -> String {
    let mut regex = String::with_capacity(glob.len() * 2);
    for c in glob.chars() {
        match c {
            '*' => regex.push_str(".*"),
            '?' => regex.push('.'),
            // Escape regex special characters
            '.' | '+' | '^' | '$' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '\\' => {
                regex.push('\\');
                regex.push(c);
            }
            _ => regex.push(c),
        }
    }
    regex
}

/// Build a query that handles wildcard patterns.
/// Returns Some(query) if the input contains wildcards, None otherwise.
fn build_wildcard_query(
    query_str: &str,
    fields: &[Field],
) -> Option<Box<dyn tantivy::query::Query>> {
    // Split query into terms
    let terms: Vec<&str> = query_str.split_whitespace().collect();

    // Check if any term has wildcards
    if !terms.iter().any(|t| has_wildcard(t)) {
        return None;
    }

    // Build queries for each term
    let mut term_queries: Vec<(Occur, Box<dyn tantivy::query::Query>)> = Vec::new();

    for term in terms {
        if has_wildcard(term) {
            // Convert glob pattern to regex and create RegexQuery for each field
            let regex_pattern = glob_to_regex(&term.to_lowercase());

            let mut field_queries: Vec<(Occur, Box<dyn tantivy::query::Query>)> = Vec::new();
            for &field in fields {
                if let Ok(regex_query) = RegexQuery::from_pattern(&regex_pattern, field) {
                    field_queries.push((Occur::Should, Box::new(regex_query)));
                }
            }

            if !field_queries.is_empty() {
                term_queries.push((Occur::Must, Box::new(BooleanQuery::new(field_queries))));
            }
        } else {
            // For non-wildcard terms, we'll let the main query parser handle them
            // by returning None and falling back to normal parsing
            // This is a simplification - in practice, we handle the whole query as wildcard
            // if any term has wildcards
            let term_lower = term.to_lowercase();
            let mut field_queries: Vec<(Occur, Box<dyn tantivy::query::Query>)> = Vec::new();
            for &field in fields {
                let tantivy_term = tantivy::Term::from_field_text(field, &term_lower);
                field_queries.push((
                    Occur::Should,
                    Box::new(tantivy::query::TermQuery::new(
                        tantivy_term,
                        tantivy::schema::IndexRecordOption::WithFreqs,
                    )),
                ));
            }
            if !field_queries.is_empty() {
                term_queries.push((Occur::Must, Box::new(BooleanQuery::new(field_queries))));
            }
        }
    }

    if term_queries.is_empty() {
        None
    } else {
        Some(Box::new(BooleanQuery::new(term_queries)))
    }
}

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

/// Query enum representing all possible fulltext query types.
/// Uses dispatch wrappers internally for channel/timeout handling.
///
/// **Note**: This is internal infrastructure for the query dispatch pipeline.
/// Users interact with `Nodes`/`Edges`/`Facets` directly via the `Runnable` trait.
/// This enum is public because it appears in reader function signatures, but users
/// should not construct variants directly.
#[derive(Debug)]
#[doc(hidden)]
#[allow(private_interfaces)]
pub enum Search {
    Nodes(NodesDispatch),
    Edges(EdgesDispatch),
    Facets(FacetsDispatch),
}

impl std::fmt::Display for Search {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Search::Nodes(q) => write!(f, "FulltextNodes: query={}, limit={}", q.params.query, q.params.limit),
            Search::Edges(q) => write!(f, "FulltextEdges: query={}, limit={}", q.params.query, q.params.limit),
            Search::Facets(q) => write!(
                f,
                "FulltextFacets: doc_type_filter={:?}, tags_limit={}",
                q.params.doc_type_filter, q.params.tags_limit
            ),
        }
    }
}

// ============================================================================
// Runnable Trait
// ============================================================================

/// Trait for query builders that can be executed via a Reader.
///
/// This trait is generic over the reader type `R`, which allows the same query type
/// (e.g., `Nodes`) to be executed against different readers with different return types:
///
/// - `Runnable<fulltext::Reader>` → returns raw fulltext hits (e.g., `Vec<NodeHit>`)
/// - `Runnable<reader::Reader>` → returns hydrated graph data (e.g., `Vec<NodeResult>`)
///
/// # Example
///
/// ```ignore
/// use motlie_db::fulltext::{Nodes, Runnable, Reader as FulltextReader};
/// use motlie_db::reader::Reader as UnifiedReader;
///
/// // Same Nodes type, different output based on reader
/// let hits: Vec<NodeHit> = Nodes::new("rust", 10).run(&fulltext_reader, timeout).await?;
/// let results: Vec<NodeResult> = Nodes::new("rust", 10).run(&unified_reader, timeout).await?;
/// ```
#[async_trait::async_trait]
pub trait Runnable<R> {
    /// The output type this query produces
    type Output: Send + 'static;

    /// Execute this query against the specified reader with the given timeout
    async fn run(self, reader: &R, timeout: Duration) -> Result<Self::Output>;
}

// ============================================================================
// Nodes Query - Search for nodes by text
// ============================================================================

/// Query parameters for searching nodes by fulltext query.
///
/// This struct contains only user-facing parameters and can be:
/// - Constructed via struct initialization with `..Default::default()`
/// - Deserialized from JSON or other formats
/// - Cloned and reused
///
/// # Examples
///
/// ```ignore
/// // Struct initialization
/// let query = Nodes {
///     query: "rust programming".to_string(),
///     limit: 10,
///     tags: vec!["systems".to_string()],
///     ..Default::default()
/// };
///
/// // Builder pattern
/// let query = Nodes::new("rust".to_string(), 10)
///     .with_fuzzy(FuzzyLevel::Low)
///     .with_tags(vec!["async".to_string()]);
///
/// // Deserialize from JSON
/// let query: Nodes = serde_json::from_str(r#"{"query": "rust", "limit": 10}"#)?;
/// ```
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Nodes {
    /// The search query string
    pub query: String,

    /// Maximum number of results to return
    pub limit: usize,

    /// Fuzzy matching level. If None, defaults to FuzzyLevel::None (exact match).
    pub fuzzy_level: Option<FuzzyLevel>,

    /// Filter by tags (documents must have ANY of the specified tags)
    pub tags: Vec<String>,
}

impl Nodes {
    /// Create a new query request.
    /// Use `.run(reader, timeout)` to execute this query.
    pub fn new(query: String, limit: usize) -> Self {
        Self {
            query,
            limit,
            fuzzy_level: None,
            tags: Vec::new(),
        }
    }

    /// Enable fuzzy matching with the specified level.
    ///
    /// Example:
    /// ```ignore
    /// Nodes::new("rast".to_string(), 10).with_fuzzy(FuzzyLevel::Low)
    /// ```
    pub fn with_fuzzy(mut self, level: FuzzyLevel) -> Self {
        self.fuzzy_level = Some(level);
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

    /// Execute this query directly against storage without dispatch machinery.
    ///
    /// This is useful for CLI tools or direct storage access. For concurrent
    /// queries through the MPMC pipeline, use `.run(reader, timeout)` instead.
    ///
    /// Example:
    /// ```ignore
    /// let results = Nodes::new("rust".to_string(), 10)
    ///     .execute(&storage)
    ///     .await?;
    /// ```
    pub async fn execute(&self, storage: &Storage) -> Result<Vec<NodeHit>> {
        NodesDispatch::execute_params(self, storage).await
    }
}

/// Internal dispatch wrapper for Nodes query execution.
/// Contains the query parameters plus channel/timeout for async dispatch.
#[derive(Debug)]
pub(crate) struct NodesDispatch {
    /// The query parameters
    pub(crate) params: Nodes,

    /// Timeout for this query execution
    pub(crate) timeout: Duration,

    /// Channel to send the result back to the client
    pub(crate) result_tx: oneshot::Sender<Result<Vec<NodeHit>>>,
}

impl NodesDispatch {
    /// Create a new dispatch wrapper
    pub(crate) fn new(
        params: Nodes,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<Vec<NodeHit>>>,
    ) -> Self {
        Self {
            params,
            timeout,
            result_tx,
        }
    }

    /// Send the result back to the client (consumes self)
    pub(crate) fn send_result(self, result: Result<Vec<NodeHit>>) {
        // Ignore error if receiver was dropped (client timeout/cancellation)
        let _ = self.result_tx.send(result);
    }

    /// Execute a Nodes query directly without dispatch machinery.
    /// This is used by the unified query module for composition.
    pub(crate) async fn execute_params(params: &Nodes, storage: &Storage) -> Result<Vec<NodeHit>> {
        // Create a temporary dispatch just to access the execute logic
        // We use a dummy oneshot channel that we won't actually use
        let (tx, _rx) = oneshot::channel();
        let dispatch = NodesDispatch {
            params: params.clone(),
            timeout: Duration::from_secs(0), // Not used
            result_tx: tx,
        };
        <NodesDispatch as super::reader::QueryExecutor>::execute(&dispatch, storage).await
    }
}

#[async_trait::async_trait]
impl Runnable<Reader> for Nodes {
    type Output = Vec<NodeHit>;

    async fn run(self, reader: &Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = oneshot::channel();

        let dispatch = NodesDispatch::new(self, timeout, result_tx);

        reader.send_query(Search::Nodes(dispatch)).await?;

        // Wait for result with timeout
        match tokio::time::timeout(timeout, result_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("Query channel closed")),
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
        }
    }
}

#[async_trait::async_trait]
impl QueryExecutor for NodesDispatch {
    type Output = Vec<NodeHit>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let params = &self.params;
        tracing::debug!(
            query = %params.query,
            fuzzy_level = ?params.fuzzy_level,
            limit = params.limit,
            "Executing fulltext Nodes query"
        );

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

        // Build text query - check for wildcards first, then exact or fuzzy
        let fuzzy_level = params.fuzzy_level.unwrap_or(FuzzyLevel::None);
        let text_query: Box<dyn tantivy::query::Query> = if fuzzy_level == FuzzyLevel::None {
            // Check for wildcard patterns (e.g., lik*, test?)
            if let Some(wildcard_query) = build_wildcard_query(
                &params.query,
                &[fields.content_field, fields.node_name_field],
            ) {
                wildcard_query
            } else {
                // No wildcards: use QueryParser for full query syntax support
                let query_parser = QueryParser::for_index(
                    index,
                    vec![fields.content_field, fields.node_name_field],
                );
                let parsed = query_parser
                    .parse_query(&params.query)
                    .map_err(|e| anyhow::anyhow!("Failed to parse query '{}': {}", params.query, e))?;
                parsed
            }
        } else {
            // Fuzzy match: build FuzzyTermQuery for each term in both fields
            let distance = fuzzy_level.to_distance();
            let terms: Vec<&str> = params.query.split_whitespace().collect();

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
        if !params.tags.is_empty() {
            let mut tag_clauses: Vec<(Occur, Box<dyn tantivy::query::Query>)> = Vec::new();
            for tag in &params.tags {
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
            .search(&combined_query, &TopDocs::with_limit(params.limit * 3))
            .map_err(|e| anyhow::anyhow!("Search failed: {}", e))?;

        // Collect results, deduplicating by node ID and keeping best score + match source
        // This handles both node documents and node_fragment documents
        let mut node_hits: HashMap<Id, (f32, MatchSource)> = HashMap::new();

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

            // Extract doc_type to determine match source
            let match_source = match doc.get_first(fields.doc_type_field).and_then(|v| v.as_str())
            {
                Some("nodes") => MatchSource::NodeName,
                Some("node_fragments") => MatchSource::NodeFragment,
                _ => MatchSource::NodeName, // Default fallback
            };

            // Update with best score and its match source
            node_hits
                .entry(id)
                .and_modify(|(existing_score, existing_source)| {
                    if score > *existing_score {
                        *existing_score = score;
                        *existing_source = match_source;
                    }
                })
                .or_insert((score, match_source));
        }

        // Convert to results and sort by score
        let mut results: Vec<NodeHit> = node_hits
            .into_iter()
            .map(|(id, (score, match_source))| NodeHit {
                score,
                id,
                fragment_timestamp: None, // Deduplicated results don't track individual fragments
                match_source,
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(params.limit);

        Ok(results)
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

// Use macro to implement QueryProcessor for dispatch types
crate::impl_fulltext_query_processor!(NodesDispatch);

// ============================================================================
// Edges Query - Search for edges by text
// ============================================================================

/// Query parameters for searching edges by fulltext query.
///
/// This struct contains only user-facing parameters and can be:
/// - Constructed via struct initialization with `..Default::default()`
/// - Deserialized from JSON or other formats
/// - Cloned and reused
///
/// # Examples
///
/// ```ignore
/// // Struct initialization
/// let query = Edges {
///     query: "depends_on".to_string(),
///     limit: 10,
///     tags: vec!["important".to_string()],
///     ..Default::default()
/// };
///
/// // Builder pattern
/// let query = Edges::new("relationship".to_string(), 10)
///     .with_fuzzy(FuzzyLevel::Low);
/// ```
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Edges {
    /// The search query string
    pub query: String,

    /// Maximum number of results to return
    pub limit: usize,

    /// Fuzzy matching level. If None, defaults to FuzzyLevel::None (exact match).
    pub fuzzy_level: Option<FuzzyLevel>,

    /// Filter by tags (documents must have ANY of the specified tags)
    pub tags: Vec<String>,
}

impl Edges {
    /// Create a new query request.
    /// Use `.run(reader, timeout)` to execute this query.
    pub fn new(query: String, limit: usize) -> Self {
        Self {
            query,
            limit,
            fuzzy_level: None,
            tags: Vec::new(),
        }
    }

    /// Enable fuzzy matching with the specified level.
    ///
    /// Example:
    /// ```ignore
    /// Edges::new("dependz_on".to_string(), 10).with_fuzzy(FuzzyLevel::Low)
    /// ```
    pub fn with_fuzzy(mut self, level: FuzzyLevel) -> Self {
        self.fuzzy_level = Some(level);
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

    /// Execute this query directly against storage without dispatch machinery.
    ///
    /// This is useful for CLI tools or direct storage access. For concurrent
    /// queries through the MPMC pipeline, use `.run(reader, timeout)` instead.
    ///
    /// Example:
    /// ```ignore
    /// let results = Edges::new("depends_on".to_string(), 10)
    ///     .execute(&storage)
    ///     .await?;
    /// ```
    pub async fn execute(&self, storage: &Storage) -> Result<Vec<EdgeHit>> {
        EdgesDispatch::execute_params(self, storage).await
    }
}

/// Internal dispatch wrapper for Edges query execution.
/// Contains the query parameters plus channel/timeout for async dispatch.
#[derive(Debug)]
pub(crate) struct EdgesDispatch {
    /// The query parameters
    pub(crate) params: Edges,

    /// Timeout for this query execution
    pub(crate) timeout: Duration,

    /// Channel to send the result back to the client
    pub(crate) result_tx: oneshot::Sender<Result<Vec<EdgeHit>>>,
}

impl EdgesDispatch {
    /// Create a new dispatch wrapper
    pub(crate) fn new(
        params: Edges,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<Vec<EdgeHit>>>,
    ) -> Self {
        Self {
            params,
            timeout,
            result_tx,
        }
    }

    /// Send the result back to the client (consumes self)
    pub(crate) fn send_result(self, result: Result<Vec<EdgeHit>>) {
        // Ignore error if receiver was dropped (client timeout/cancellation)
        let _ = self.result_tx.send(result);
    }

    /// Execute an Edges query directly without dispatch machinery.
    /// This is used by the unified query module for composition.
    pub(crate) async fn execute_params(params: &Edges, storage: &Storage) -> Result<Vec<EdgeHit>> {
        // Create a temporary dispatch just to access the execute logic
        // We use a dummy oneshot channel that we won't actually use
        let (tx, _rx) = oneshot::channel();
        let dispatch = EdgesDispatch {
            params: params.clone(),
            timeout: Duration::from_secs(0), // Not used
            result_tx: tx,
        };
        <EdgesDispatch as super::reader::QueryExecutor>::execute(&dispatch, storage).await
    }
}

#[async_trait::async_trait]
impl Runnable<Reader> for Edges {
    type Output = Vec<EdgeHit>;

    async fn run(self, reader: &Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = oneshot::channel();

        let dispatch = EdgesDispatch::new(self, timeout, result_tx);

        reader.send_query(Search::Edges(dispatch)).await?;

        // Wait for result with timeout
        match tokio::time::timeout(timeout, result_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("Query channel closed")),
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
        }
    }
}

#[async_trait::async_trait]
impl QueryExecutor for EdgesDispatch {
    type Output = Vec<EdgeHit>;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let params = &self.params;
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

        tracing::debug!(
            query = %params.query,
            index_docs = searcher.num_docs(),
            content_field = ?fields.content_field,
            edge_name_field = ?fields.edge_name_field,
            doc_type_field = ?fields.doc_type_field,
            "[FulltextEdges] Executing query"
        );

        // Build text query - check for wildcards first, then exact or fuzzy
        let fuzzy_level = params.fuzzy_level.unwrap_or(FuzzyLevel::None);
        let text_query: Box<dyn tantivy::query::Query> = if fuzzy_level == FuzzyLevel::None {
            // Check for wildcard patterns (e.g., lik*, test?)
            if let Some(wildcard_query) = build_wildcard_query(
                &params.query,
                &[fields.content_field, fields.edge_name_field],
            ) {
                wildcard_query
            } else {
                // No wildcards: use QueryParser for full query syntax support
                let query_parser = QueryParser::for_index(
                    index,
                    vec![fields.content_field, fields.edge_name_field],
                );
                let parsed = query_parser
                    .parse_query(&params.query)
                    .map_err(|e| anyhow::anyhow!("Failed to parse query '{}': {}", params.query, e))?;
                parsed
            }
        } else {
            // Fuzzy match: build FuzzyTermQuery for each term in both fields
            let distance = fuzzy_level.to_distance();
            let terms: Vec<&str> = params.query.split_whitespace().collect();

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
        if !params.tags.is_empty() {
            let mut tag_clauses: Vec<(Occur, Box<dyn tantivy::query::Query>)> = Vec::new();
            for tag in &params.tags {
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
            .search(&combined_query, &TopDocs::with_limit(params.limit * 3))
            .map_err(|e| anyhow::anyhow!("Search failed: {}", e))?;

        tracing::debug!(
            result_count = top_docs.len(),
            query = %params.query,
            "[FulltextEdges] Search returned raw results"
        );

        // Collect results, deduplicating by edge key (src_id, dst_id, edge_name) and keeping best score + match source
        // Edge key is (src_id, dst_id, edge_name)
        let mut edge_hits: HashMap<(Id, Id, String), (f32, MatchSource)> = HashMap::new();

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

            // Extract doc_type to determine match source
            let match_source = match doc.get_first(fields.doc_type_field).and_then(|v| v.as_str())
            {
                Some("edges") => MatchSource::EdgeName,
                Some("edge_fragments") => MatchSource::EdgeFragment,
                _ => MatchSource::EdgeName, // Default fallback
            };

            // Update with best score and its match source
            let key = (src_id, dst_id, edge_name);
            edge_hits
                .entry(key)
                .and_modify(|(existing_score, existing_source)| {
                    if score > *existing_score {
                        *existing_score = score;
                        *existing_source = match_source;
                    }
                })
                .or_insert((score, match_source));
        }

        // Convert to results and sort by score
        let mut results: Vec<EdgeHit> = edge_hits
            .into_iter()
            .map(|((src_id, dst_id, edge_name), (score, match_source))| EdgeHit {
                score,
                src_id,
                dst_id,
                edge_name,
                fragment_timestamp: None, // Deduplicated results don't track individual fragments
                match_source,
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(params.limit);

        Ok(results)
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

crate::impl_fulltext_query_processor!(EdgesDispatch);

// ============================================================================
// Facets Query - Get facet counts from the index
// ============================================================================

/// Query parameters for retrieving facet counts from the fulltext index.
///
/// This struct contains only user-facing parameters and can be:
/// - Constructed via struct initialization with `..Default::default()`
/// - Deserialized from JSON or other formats
/// - Cloned and reused
///
/// # Examples
///
/// ```ignore
/// // Struct initialization
/// let query = Facets {
///     doc_type_filter: vec!["nodes".to_string()],
///     tags_limit: 50,
/// };
///
/// // Builder pattern
/// let query = Facets::new()
///     .with_doc_type_filter(vec!["nodes".to_string()])
///     .with_tags_limit(50);
/// ```
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Facets {
    /// Optional filter by document types (if empty, count all)
    pub doc_type_filter: Vec<String>,

    /// Limit for number of tag facets to return (default: 100)
    pub tags_limit: usize,
}

impl Facets {
    /// Create a new facets query (counts all facets across all documents)
    pub fn new() -> Self {
        Self {
            doc_type_filter: Vec::new(),
            tags_limit: 100,
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

    /// Execute this query directly against storage without dispatch machinery.
    ///
    /// This is useful for CLI tools or direct storage access. For concurrent
    /// queries through the MPMC pipeline, use `.run(reader, timeout)` instead.
    ///
    /// Example:
    /// ```ignore
    /// let counts = Facets::new()
    ///     .with_doc_type_filter(vec!["nodes".to_string()])
    ///     .execute(&storage)
    ///     .await?;
    /// ```
    pub async fn execute(&self, storage: &Storage) -> Result<super::search::FacetCounts> {
        FacetsDispatch::execute_params(self, storage).await
    }
}

/// Internal dispatch wrapper for Facets query execution.
/// Contains the query parameters plus channel/timeout for async dispatch.
#[derive(Debug)]
pub(crate) struct FacetsDispatch {
    /// The query parameters
    pub(crate) params: Facets,

    /// Timeout for this query execution
    pub(crate) timeout: Duration,

    /// Channel to send the result back to the client
    pub(crate) result_tx: oneshot::Sender<Result<super::search::FacetCounts>>,
}

impl FacetsDispatch {
    /// Create a new dispatch wrapper
    pub(crate) fn new(
        params: Facets,
        timeout: Duration,
        result_tx: oneshot::Sender<Result<super::search::FacetCounts>>,
    ) -> Self {
        Self {
            params,
            timeout,
            result_tx,
        }
    }

    /// Send the result back to the client (consumes self)
    pub(crate) fn send_result(self, result: Result<super::search::FacetCounts>) {
        // Ignore error if receiver was dropped (client timeout/cancellation)
        let _ = self.result_tx.send(result);
    }

    /// Execute a Facets query directly without dispatch machinery.
    /// This is used by the unified query module for composition.
    pub(crate) async fn execute_params(
        params: &Facets,
        storage: &Storage,
    ) -> Result<super::search::FacetCounts> {
        // Create a temporary dispatch just to access the execute logic
        // We use a dummy oneshot channel that we won't actually use
        let (tx, _rx) = oneshot::channel();
        let dispatch = FacetsDispatch {
            params: params.clone(),
            timeout: Duration::from_secs(0), // Not used
            result_tx: tx,
        };
        <FacetsDispatch as super::reader::QueryExecutor>::execute(&dispatch, storage).await
    }
}

#[async_trait::async_trait]
impl Runnable<Reader> for Facets {
    type Output = super::search::FacetCounts;

    async fn run(self, reader: &Reader, timeout: Duration) -> Result<Self::Output> {
        let (result_tx, result_rx) = oneshot::channel();

        let dispatch = FacetsDispatch::new(self, timeout, result_tx);

        reader.send_query(Search::Facets(dispatch)).await?;

        // Wait for result with timeout
        match tokio::time::timeout(timeout, result_rx).await {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => Err(anyhow::anyhow!("Query channel closed")),
            Err(_) => Err(anyhow::anyhow!("Query timeout after {:?}", timeout)),
        }
    }
}

#[async_trait::async_trait]
impl QueryExecutor for FacetsDispatch {
    type Output = super::search::FacetCounts;

    async fn execute(&self, storage: &Storage) -> Result<Self::Output> {
        let params = &self.params;
        tracing::debug!(
            doc_type_filter = ?params.doc_type_filter,
            "Executing fulltext Facets query"
        );

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
        let query: Box<dyn tantivy::query::Query> = if params.doc_type_filter.is_empty() {
            Box::new(AllQuery)
        } else {
            // Build doc_type filter
            let mut doc_type_clauses: Vec<(Occur, Box<dyn tantivy::query::Query>)> = Vec::new();
            for doc_type in &params.doc_type_filter {
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
            if tag_count >= params.tags_limit {
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
    }
}

crate::impl_fulltext_query_processor!(FacetsDispatch);

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
    use crate::fulltext::reader::{
        create_query_reader, spawn_consumer as spawn_query_consumer, Consumer, ReaderConfig,
    };
    use crate::fulltext::writer::spawn_mutation_consumer;
    use crate::fulltext::{Index, Storage};
    use crate::graph::mutation::{AddNode, AddNodeFragment, Runnable as MutRunnable};
    use crate::graph::schema::NodeSummary;
    use crate::graph::writer::{create_mutation_writer, WriterConfig};
    use crate::{DataUrl, TimestampMilli};
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
        let fulltext_handle = spawn_mutation_consumer(receiver, writer_config, &index_path);

        // Create test nodes with searchable content
        let node1_id = Id::new();
        let node1 = AddNode {
            id: node1_id,
            ts_millis: TimestampMilli::now(),
            name: "RustLang".to_string(),
            valid_range: None,
            summary: crate::graph::schema::NodeSummary::from_text("Rust language summary"),
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
            summary: crate::graph::schema::NodeSummary::from_text("Python language summary"),
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
        let consumer_handle = spawn_query_consumer(consumer);

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
