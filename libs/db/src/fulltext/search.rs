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
    /// Document types: [("nodes", 45), ("forward_edges", 23), ...]
    pub doc_types: HashMap<String, u64>,

    /// User-defined tags: [("rust", 15), ("systems_programming", 8), ...]
    pub tags: HashMap<String, u64>,

    /// Validity structure: [("unbounded", 50), ("bounded", 10), ...]
    pub validity: HashMap<String, u64>,
}

impl Default for FacetCounts {
    fn default() -> Self {
        Self {
            doc_types: HashMap::new(),
            tags: HashMap::new(),
            validity: HashMap::new(),
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
        self.doc_types.len() + self.tags.len() + self.validity.len()
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
