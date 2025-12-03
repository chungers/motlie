//! Fuzzy search support for typo-tolerant queries
//!
//! This module provides fuzzy search capabilities using Levenshtein distance
//! to find matches even when the query contains spelling errors.
//!
//! Example:
//! - "Rast" matches "Rust" (1 character difference)
//! - "performence" matches "performance" (2 character difference)

/// Fuzzy search level determining match strictness
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FuzzyLevel {
    /// No fuzzy matching (exact match only)
    None,

    /// Allow 1 character edit (insert, delete, substitute, or transpose)
    /// Good for catching typos in short words
    Low,

    /// Allow 2 character edits
    /// More forgiving, good for longer words
    Medium,
}

impl FuzzyLevel {
    /// Convert fuzzy level to edit distance
    pub fn to_distance(self) -> u8 {
        match self {
            FuzzyLevel::None => 0,
            FuzzyLevel::Low => 1,
            FuzzyLevel::Medium => 2,
        }
    }
}

/// Fuzzy search options
#[derive(Debug, Clone)]
pub struct FuzzySearchOptions {
    /// The search query string
    pub query: String,

    /// Fuzzy match level
    pub level: FuzzyLevel,

    /// Number of prefix characters that must match exactly (for performance)
    /// Setting this to 2 means the first 2 characters must match exactly
    pub prefix_length: usize,

    /// Allow character transpositions (ab → ba)
    pub transpositions: bool,

    /// Maximum number of results to return
    pub limit: usize,

    /// Auto-fallback: if exact search returns few results, try fuzzy
    pub auto_fallback: bool,

    /// Threshold for auto-fallback (minimum exact results before trying fuzzy)
    pub fallback_threshold: usize,
}

impl Default for FuzzySearchOptions {
    fn default() -> Self {
        Self {
            query: String::new(),
            level: FuzzyLevel::Medium,
            prefix_length: 2,
            transpositions: true,
            limit: 10,
            auto_fallback: true,
            fallback_threshold: 3,
        }
    }
}

impl FuzzySearchOptions {
    /// Create options with no fuzzy matching (exact only)
    pub fn exact(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            level: FuzzyLevel::None,
            ..Default::default()
        }
    }

    /// Create options with low fuzzy matching (1 edit)
    pub fn low(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            level: FuzzyLevel::Low,
            ..Default::default()
        }
    }

    /// Create options with medium fuzzy matching (2 edits)
    pub fn medium(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            level: FuzzyLevel::Medium,
            ..Default::default()
        }
    }

    /// Enable auto-fallback to fuzzy if exact search fails
    pub fn with_auto_fallback(mut self, threshold: usize) -> Self {
        self.auto_fallback = true;
        self.fallback_threshold = threshold;
        self
    }

    /// Disable auto-fallback
    pub fn without_auto_fallback(mut self) -> Self {
        self.auto_fallback = false;
        self
    }

    /// Set prefix length (characters that must match exactly)
    pub fn with_prefix_length(mut self, length: usize) -> Self {
        self.prefix_length = length;
        self
    }

    /// Set whether to allow transpositions
    pub fn with_transpositions(mut self, allow: bool) -> Self {
        self.transpositions = allow;
        self
    }

    /// Set result limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuzzy_level_to_distance() {
        assert_eq!(FuzzyLevel::None.to_distance(), 0);
        assert_eq!(FuzzyLevel::Low.to_distance(), 1);
        assert_eq!(FuzzyLevel::Medium.to_distance(), 2);
    }

    #[test]
    fn test_fuzzy_options_exact() {
        let opts = FuzzySearchOptions::exact("test");
        assert_eq!(opts.level, FuzzyLevel::None);
        assert_eq!(opts.query, "test");
    }

    #[test]
    fn test_fuzzy_options_low() {
        let opts = FuzzySearchOptions::low("test");
        assert_eq!(opts.level, FuzzyLevel::Low);
        assert_eq!(opts.query, "test");
    }

    #[test]
    fn test_fuzzy_options_medium() {
        let opts = FuzzySearchOptions::medium("test");
        assert_eq!(opts.level, FuzzyLevel::Medium);
        assert_eq!(opts.query, "test");
    }

    #[test]
    fn test_fuzzy_options_builder() {
        let opts = FuzzySearchOptions::medium("test")
            .with_prefix_length(3)
            .with_transpositions(false)
            .with_limit(20)
            .without_auto_fallback();

        assert_eq!(opts.prefix_length, 3);
        assert!(!opts.transpositions);
        assert_eq!(opts.limit, 20);
        assert!(!opts.auto_fallback);
    }

    #[test]
    fn test_fuzzy_options_default() {
        let opts = FuzzySearchOptions::default();
        assert_eq!(opts.level, FuzzyLevel::Medium);
        assert_eq!(opts.prefix_length, 2);
        assert!(opts.transpositions);
        assert_eq!(opts.limit, 10);
        assert!(opts.auto_fallback);
        assert_eq!(opts.fallback_threshold, 3);
    }
}
