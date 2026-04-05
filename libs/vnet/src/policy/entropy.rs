//! Shannon entropy analysis for detecting encoded/encrypted data.
//!
//! **Connection phase: Phase 1 (DNS — "Intent")**
//!
//! Measures the randomness of a string in bits per character. Natural language
//! and readable hostnames have low entropy (1.0–2.5); base64/hex-encoded data
//! has high entropy (3.5–5.0).
//!
//! # Policy API integration
//!
//! Used in `EgressPolicy::on_dns_query()` with `DnsQueryContext`:
//! ```rust,ignore
//! fn on_dns_query(&self, ctx: &DnsQueryContext) -> PolicyAction {
//!     let suspicious = ctx.label_lengths.iter()
//!         .zip(ctx.domain.split('.'))
//!         .any(|(len, label)| *len > 30 && shannon_entropy(label) > 3.5);
//!     if suspicious {
//!         PolicyAction::Deny { errno: libc::EHOSTUNREACH, reason: ... }
//!     } else {
//!         PolicyAction::Allow { reason: None }
//!     }
//! }
//! ```
//!
//! # Use cases
//!
//! - **DNS exfiltration** (Phase 1): high-entropy subdomain labels carrying
//!   encoded payloads (e.g. `aGVsbG8gd29ybGQ.exfil.attacker.com`)
//!
//! # Entropy scale
//!
//! | Bits/char | Interpretation | Examples |
//! |-----------|----------------|----------|
//! | 1.0 – 2.5 | Natural language | `archive`, `ubuntu`, `github` |
//! | 2.5 – 3.5 | Mixed / ambiguous | `cdn-a1b2c3`, `edge-us-west` |
//! | 3.5 – 4.5 | Likely encoded | `aGVsbG8gd29ybGQ` (base64) |
//! | 4.5 – 5.0 | Near-random | `7f3a9c2e` (hex), encrypted blobs |

/// Shannon entropy of a string (bits per character).
///
/// Returns 0.0 for empty input. Higher values indicate more randomness.
/// A threshold of ~3.5 bits/char separates most legitimate hostnames
/// from base64/hex-encoded payloads.
pub fn shannon_entropy(s: &str) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    let mut freq = [0u32; 256];
    for &b in s.as_bytes() {
        freq[b as usize] += 1;
    }
    let len = s.len() as f64;
    freq.iter()
        .filter(|&&count| count > 0)
        .map(|&count| {
            let p = count as f64 / len;
            -p * p.log2()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_string_has_zero_entropy() {
        assert_eq!(shannon_entropy(""), 0.0);
    }

    #[test]
    fn single_char_repeated_has_zero_entropy() {
        assert_eq!(shannon_entropy("aaaaaaa"), 0.0);
    }

    #[test]
    fn natural_language_has_low_entropy() {
        let e = shannon_entropy("archive");
        assert!(e > 1.0 && e < 3.0, "expected low entropy, got {}", e);
    }

    #[test]
    fn base64_has_high_entropy() {
        let e = shannon_entropy("aGVsbG8gd29ybGQ");
        assert!(e > 3.0, "expected high entropy for base64, got {}", e);
    }

    #[test]
    fn hex_has_high_entropy() {
        let e = shannon_entropy("78e731027d8fd50ed642340b7c9a63b3");
        assert!(e > 3.5, "expected high entropy for hex, got {}", e);
    }

    #[test]
    fn short_cdn_hash_moderate_entropy() {
        let e = shannon_entropy("d1a2b3c4");
        assert!(e > 2.5, "expected moderate+ entropy for short hash, got {}", e);
    }
}
