//! Domain name utilities for policy implementations.
//!
//! Provides base domain extraction, suffix matching, and label parsing
//! used by category policies and DNS exfiltration detectors.

/// Extract the base domain (last two labels) from an FQDN.
///
/// # Examples
///
/// ```
/// use motlie_policy::domain::extract_base_domain;
///
/// assert_eq!(extract_base_domain("a.b.c.attacker.com"), "attacker.com");
/// assert_eq!(extract_base_domain("archive.ubuntu.com"), "ubuntu.com");
/// assert_eq!(extract_base_domain("localhost"), "localhost");
/// ```
pub fn extract_base_domain(domain: &str) -> String {
    let labels: Vec<&str> = domain.split('.').collect();
    if labels.len() >= 2 {
        labels[labels.len() - 2..].join(".")
    } else {
        domain.to_string()
    }
}

/// Check if a domain matches a suffix pattern.
///
/// Supports exact match (`crates.io`) and suffix match (`*.ubuntu.com`).
///
/// # Examples
///
/// ```
/// use motlie_policy::domain::matches_suffix;
///
/// assert!(matches_suffix("crates.io", "crates.io"));
/// assert!(matches_suffix("archive.ubuntu.com", "*.ubuntu.com"));
/// assert!(!matches_suffix("evil-ubuntu.com", "*.ubuntu.com"));
/// ```
pub fn matches_suffix(domain: &str, pattern: &str) -> bool {
    if let Some(suffix) = pattern.strip_prefix("*.") {
        // Suffix match: domain must end with .suffix and have at least one
        // label before it (to avoid matching "ubuntu.com" against "*.ubuntu.com")
        domain.ends_with(suffix) && domain.len() > suffix.len() + 1
            && domain.as_bytes()[domain.len() - suffix.len() - 1] == b'.'
    } else {
        // Exact match
        domain == pattern
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_domain_extracts_last_two_labels() {
        assert_eq!(extract_base_domain("a.b.c.attacker.com"), "attacker.com");
        assert_eq!(extract_base_domain("archive.ubuntu.com"), "ubuntu.com");
        assert_eq!(extract_base_domain("github.com"), "github.com");
    }

    #[test]
    fn base_domain_single_label() {
        assert_eq!(extract_base_domain("localhost"), "localhost");
    }

    #[test]
    fn suffix_exact_match() {
        assert!(matches_suffix("crates.io", "crates.io"));
        assert!(!matches_suffix("evil-crates.io", "crates.io"));
    }

    #[test]
    fn suffix_wildcard_match() {
        assert!(matches_suffix("archive.ubuntu.com", "*.ubuntu.com"));
        assert!(matches_suffix("security.ubuntu.com", "*.ubuntu.com"));
    }

    #[test]
    fn suffix_wildcard_no_false_positive() {
        // "ubuntu.com" itself should NOT match "*.ubuntu.com"
        assert!(!matches_suffix("ubuntu.com", "*.ubuntu.com"));
        // "evil-ubuntu.com" should NOT match
        assert!(!matches_suffix("evil-ubuntu.com", "*.ubuntu.com"));
    }
}
