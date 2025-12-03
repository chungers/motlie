//! Faceted search support for fulltext indexing
//!
//! This module provides types and functions for performing faceted search,
//! allowing users to filter results by categories (facets) and see aggregated
//! counts for each facet value.

use std::collections::HashMap;

/// Search options for fulltext queries with facet filtering
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// The search query string
    pub query: String,

    /// Facet filters to apply (field_name, facet_value)
    /// Example: [("type", "node"), ("time", "last_week")]
    pub facet_filters: Vec<(String, String)>,

    /// Whether to collect facet counts in results
    pub collect_facets: bool,

    /// Maximum number of results to return
    pub limit: usize,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            query: String::new(),
            facet_filters: vec![],
            collect_facets: false,
            limit: 10,
        }
    }
}

/// Search results with optional facet counts
#[derive(Debug)]
pub struct SearchResults {
    /// Matching documents
    pub documents: Vec<SearchResult>,

    /// Facet counts (if collect_facets was true)
    pub facets: Option<FacetCounts>,

    /// Total number of results
    pub total: usize,
}

/// Individual search result
#[derive(Debug)]
pub struct SearchResult {
    /// BM25 relevance score
    pub score: f32,

    /// Document ID
    pub id: Vec<u8>,

    /// Document type (node, edge, node_fragment, edge_fragment)
    pub doc_type: String,

    /// Optional text snippet showing match context
    pub snippet: Option<String>,
}

/// Aggregated facet counts
#[derive(Debug)]
pub struct FacetCounts {
    /// Document types: [("node", 45), ("edge", 23), ...]
    pub doc_types: HashMap<String, u64>,

    /// Time buckets: [("last_hour", 12), ("last_day", 34), ...]
    pub time_buckets: HashMap<String, u64>,

    /// Weight ranges: [("0-0.5", 5), ("1.0-2.0", 8), ...]
    pub weight_ranges: HashMap<String, u64>,

    /// User-defined tags: [("rust", 15), ("systems_programming", 8), ...]
    pub tags: HashMap<String, u64>,
}

impl Default for FacetCounts {
    fn default() -> Self {
        Self {
            doc_types: HashMap::new(),
            time_buckets: HashMap::new(),
            weight_ranges: HashMap::new(),
            tags: HashMap::new(),
        }
    }
}

impl FacetCounts {
    /// Create a new empty FacetCounts
    pub fn new() -> Self {
        Self::default()
    }

    /// Get total count across all facets
    pub fn total_facets(&self) -> usize {
        self.doc_types.len() + self.time_buckets.len() + self.weight_ranges.len() + self.tags.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_options_default() {
        let opts = SearchOptions::default();
        assert_eq!(opts.limit, 10);
        assert!(!opts.collect_facets);
        assert!(opts.facet_filters.is_empty());
    }

    #[test]
    fn test_facet_counts_new() {
        let counts = FacetCounts::new();
        assert_eq!(counts.total_facets(), 0);
    }

    #[test]
    fn test_facet_counts_total() {
        let mut counts = FacetCounts::new();
        counts.doc_types.insert("node".to_string(), 10);
        counts.tags.insert("rust".to_string(), 5);
        assert_eq!(counts.total_facets(), 2);
    }
}
