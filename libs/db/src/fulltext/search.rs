//! Faceted search support for fulltext indexing
//!
//! This module provides types and functions for performing faceted search,
//! allowing users to filter results by categories (facets) and see aggregated
//! counts for each facet value.

use std::collections::HashMap;
use std::fmt;

use crate::Id;

/// Indicates what field/document type the search matched against.
///
/// This allows callers to understand the nature of the match without storing
/// the actual matched text (which would bloat the index).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchSource {
    /// Match came from a node's name field
    NodeName,
    /// Match came from a node fragment's content field
    NodeFragment,
    /// Match came from an edge's name field
    EdgeName,
    /// Match came from an edge fragment's content field
    EdgeFragment,
}

impl fmt::Display for MatchSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatchSource::NodeName => write!(f, "name"),
            MatchSource::NodeFragment => write!(f, "fragment"),
            MatchSource::EdgeName => write!(f, "name"),
            MatchSource::EdgeFragment => write!(f, "fragment"),
        }
    }
}

/// Individual search result - represents a single matching document from the fulltext index.
///
/// This enum distinguishes between node-related and edge-related search results,
/// allowing callers to handle each type appropriately when reconstructing RocksDB keys.
#[derive(Debug, Clone)]
pub enum Hit {
    /// A node or node fragment matched the search
    NodeHit(NodeHit),
    /// An edge or edge fragment matched the search
    EdgeHit(EdgeHit),
}

impl Hit {
    /// Get the BM25 relevance score for this result
    pub fn score(&self) -> f32 {
        match self {
            Hit::NodeHit(hit) => hit.score,
            Hit::EdgeHit(hit) => hit.score,
        }
    }
}

/// A node or node fragment that matched the search query.
///
/// Contains the information needed to look up the full document in RocksDB.
/// The presence of `fragment_timestamp` distinguishes between node and fragment matches:
///
/// - **Node match** (`fragment_timestamp` is `None`): Use `id` to query the Nodes column family
/// - **Fragment match** (`fragment_timestamp` is `Some`): Use `id` + `fragment_timestamp`
///   to query the NodeFragments column family
#[derive(Debug, Clone)]
pub struct NodeHit {
    /// BM25 relevance score
    pub score: f32,

    /// Node ID (used as key in Nodes CF, or first part of NodeFragments key)
    pub id: Id,

    /// Fragment timestamp, if this hit came from a node fragment.
    ///
    /// - `None`: The match came from the node itself (Nodes CF)
    /// - `Some(ts)`: The match came from a fragment; combined with `id`, this forms
    ///   the complete NodeFragments CfKey
    pub fragment_timestamp: Option<u64>,

    /// What field/document type the match came from
    pub match_source: MatchSource,
}

/// An edge or edge fragment that matched the search query.
///
/// Contains the information needed to look up the full document in RocksDB.
/// The presence of `fragment_timestamp` distinguishes between edge and fragment matches:
///
/// - **Edge match** (`fragment_timestamp` is `None`): Use `src_id` + `dst_id` + `edge_name`
///   to query the ForwardEdges column family
/// - **Fragment match** (`fragment_timestamp` is `Some`): Use `src_id` + `dst_id` + `edge_name`
///   + `fragment_timestamp` to query the EdgeFragments column family
#[derive(Debug, Clone)]
pub struct EdgeHit {
    /// BM25 relevance score
    pub score: f32,

    /// Source node ID
    pub src_id: Id,

    /// Destination node ID
    pub dst_id: Id,

    /// Edge name
    pub edge_name: String,

    /// Fragment timestamp, if this hit came from an edge fragment.
    ///
    /// - `None`: The match came from the edge itself (ForwardEdges CF)
    /// - `Some(ts)`: The match came from a fragment; combined with `src_id` + `dst_id` +
    ///   `edge_name`, this forms the complete EdgeFragments CfKey
    pub fragment_timestamp: Option<u64>,

    /// What field/document type the match came from
    pub match_source: MatchSource,
}

/// Aggregated facet counts
#[derive(Debug)]
pub struct FacetCounts {
    /// Document types: [("nodes", 45), ("edges", 23), ...]
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
