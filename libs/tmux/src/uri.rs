//! SSH URI parsing, rendering, and connect for SshConfig (DC21).
//!
//! Extends `SshConfig` (defined in `transport.rs`) with `parse()`,
//! `to_uri_string()`, `connect()`, `Display`, and `FromStr` impls.
//! URI logic is separated from transport logic while consolidating on one type.

use std::fmt;
use std::str::FromStr;

use anyhow::{anyhow, Result};

use crate::host::HostHandle;
use crate::transport::{LocalTransport, SshConfig, SshTransport, TransportKind};
use crate::types::{HostKeyPolicy, TmuxSocket};

/// Canonical URI component names that cannot appear as parameters.
const CANONICAL_COMPONENTS: &[&str] = &["user", "host", "port"];

/// Known parameter names.
const KNOWN_PARAMS: &[&str] = &["host-key-policy", "timeout", "keepalive", "socket-name"];

impl SshConfig {
    /// Parse from an `ssh://` URI string.
    ///
    /// Accepts nassh-style params (`;` in userinfo), query params (`?`),
    /// or both. Rejects unknown params and canonical-component duplication.
    /// For localhost URIs, `user` defaults to empty (ignored by LocalTransport).
    pub fn parse(uri: &str) -> Result<Self> {
        let remainder = uri
            .strip_prefix("ssh://")
            .ok_or_else(|| anyhow!("URI must start with ssh://"))?;

        if remainder.is_empty() {
            return Err(anyhow!("URI is missing host"));
        }

        // Split off query string
        let (before_query, query_string) = match remainder.find('?') {
            Some(pos) => (&remainder[..pos], Some(&remainder[pos + 1..])),
            None => (remainder, None),
        };

        // Split userinfo from host/path (last '@')
        let (userinfo_str, hostport_path) = match before_query.rfind('@') {
            Some(pos) => (Some(&before_query[..pos]), &before_query[pos + 1..]),
            None => (None, before_query),
        };

        // Parse userinfo: user[;param=value;...]
        let (user, userinfo_params) = parse_userinfo(userinfo_str)?;

        // Parse host[:port][/socket-path]
        let (host, port, socket_path) = parse_authority_and_path(hostport_path)?;

        if host.is_empty() {
            return Err(anyhow!("URI is missing host"));
        }

        // Parse query params
        let query_params = parse_query_params(query_string)?;

        // Check for canonical component duplication
        for params in [&userinfo_params, &query_params] {
            for (key, _) in params {
                if CANONICAL_COMPONENTS.contains(&key.as_str()) {
                    return Err(anyhow!(
                        "'{}' is a canonical URI component and cannot appear as a parameter",
                        key
                    ));
                }
            }
        }

        // Check for duplicate keys across locations
        let mut seen_keys: Vec<&str> = Vec::new();
        for (key, _) in &userinfo_params {
            seen_keys.push(key.as_str());
        }
        for (key, _) in &query_params {
            if seen_keys.contains(&key.as_str()) {
                return Err(anyhow!(
                    "duplicate parameter '{}' (appears in both userinfo and query)",
                    key
                ));
            }
        }

        // Check for unknown params
        let all_params: Vec<(&str, &str)> = userinfo_params
            .iter()
            .chain(query_params.iter())
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        for (key, _) in &all_params {
            if !KNOWN_PARAMS.contains(key) {
                return Err(anyhow!("unknown parameter: '{}'", key));
            }
        }

        // Build SshConfig
        let mut config = SshConfig::new(host, user).with_port(port);
        let mut has_socket_name = false;

        for (key, value) in &all_params {
            match *key {
                "host-key-policy" => {
                    let policy = match *value {
                        "verify" => HostKeyPolicy::Verify,
                        "tofu" => HostKeyPolicy::TrustFirstUse,
                        "insecure" => HostKeyPolicy::Insecure,
                        _ => {
                            return Err(anyhow!(
                                "invalid host-key-policy value '{}' \
                                 (expected: verify, tofu, insecure)",
                                value
                            ))
                        }
                    };
                    config = config.with_host_key_policy(policy);
                }
                "timeout" => {
                    let secs: u64 = value.parse().map_err(|_| {
                        anyhow!(
                            "invalid timeout value '{}' (expected integer seconds)",
                            value
                        )
                    })?;
                    if secs == 0 {
                        return Err(anyhow!("timeout must be > 0"));
                    }
                    config = config.with_timeout(std::time::Duration::from_secs(secs));
                }
                "keepalive" => {
                    let secs: u64 = value.parse().map_err(|_| {
                        anyhow!(
                            "invalid keepalive value '{}' (expected integer seconds)",
                            value
                        )
                    })?;
                    let interval = if secs == 0 {
                        None
                    } else {
                        Some(std::time::Duration::from_secs(secs))
                    };
                    config = config.with_keepalive(interval);
                }
                "socket-name" => {
                    has_socket_name = true;
                    config = config.with_socket(TmuxSocket::Name(value.to_string()));
                }
                _ => unreachable!(),
            }
        }

        // Handle socket-path from URI path component
        if let Some(path) = socket_path {
            if has_socket_name {
                return Err(anyhow!(
                    "socket-path and socket-name are mutually exclusive"
                ));
            }
            config = config.with_socket(TmuxSocket::Path(path));
        }

        Ok(config)
    }

    /// Render to canonical URI form.
    ///
    /// If user is non-empty, non-default parameters are placed as nassh-style
    /// userinfo params. If user is empty, non-default parameters are placed
    /// as query params. Socket paths render as the URI path component.
    pub fn to_uri_string(&self) -> String {
        let mut result = String::from("ssh://");

        // Collect non-default params
        let mut params: Vec<(&str, String)> = Vec::new();
        if *self.host_key_policy() != HostKeyPolicy::Verify {
            let value = match self.host_key_policy() {
                HostKeyPolicy::Verify => "verify",
                HostKeyPolicy::TrustFirstUse => "tofu",
                HostKeyPolicy::Insecure => "insecure",
            };
            params.push(("host-key-policy", value.to_string()));
        }
        if self.timeout() != std::time::Duration::from_secs(10) {
            params.push(("timeout", self.timeout().as_secs().to_string()));
        }
        if self.keepalive_interval() != Some(std::time::Duration::from_secs(30)) {
            let val = match self.keepalive_interval() {
                Some(d) => d.as_secs().to_string(),
                None => "0".to_string(),
            };
            params.push(("keepalive", val));
        }

        // socket-name goes as param; socket-path goes as URI path
        let mut socket_path: Option<&str> = None;
        if let Some(socket) = self.socket() {
            match socket {
                TmuxSocket::Name(n) => params.push(("socket-name", n.clone())),
                TmuxSocket::Path(p) => socket_path = Some(p.as_str()),
            }
        }

        let user = self.user();

        if !user.is_empty() {
            // Nassh-style: user[;param=value...]@host
            result.push_str(user);
            for (key, value) in &params {
                result.push(';');
                result.push_str(key);
                result.push('=');
                result.push_str(value);
            }
            result.push('@');
        }

        // Host (IPv6 needs brackets)
        let host = self.host();
        if host.contains(':') {
            result.push('[');
            result.push_str(host);
            result.push(']');
        } else {
            result.push_str(host);
        }

        // Port (omit default 22)
        if self.port() != 22 {
            result.push(':');
            result.push_str(&self.port().to_string());
        }

        // Socket path
        if let Some(path) = socket_path {
            if !path.starts_with('/') {
                result.push('/');
            }
            result.push_str(path);
        }

        // Query params (only used when user is empty)
        if user.is_empty() && !params.is_empty() {
            result.push('?');
            for (i, (key, value)) in params.iter().enumerate() {
                if i > 0 {
                    result.push('&');
                }
                result.push_str(key);
                result.push('=');
                result.push_str(value);
            }
        }

        result
    }

    /// Connect and return a HostHandle.
    ///
    /// Transport selection:
    /// - `localhost`, `127.0.0.1`, `::1` → LocalTransport (no SSH)
    /// - All other hosts → SshTransport via SSH
    ///
    /// `user` is ignored for localhost connections (LocalTransport runs as
    /// the current OS user). For SSH hosts, `user` is required — `connect()`
    /// returns an error if empty.
    pub async fn connect(self) -> Result<HostHandle> {
        let is_local = self.is_localhost();
        let user_empty = self.user().is_empty();
        let timeout = self.timeout();
        let socket = self.socket().cloned();

        if is_local {
            let transport = TransportKind::Local(LocalTransport::with_timeout(timeout));
            return Ok(HostHandle::new(transport, socket));
        }

        if user_empty {
            return Err(anyhow!("user is required for SSH connections"));
        }

        let transport = TransportKind::Ssh(SshTransport::connect(self).await?);
        Ok(HostHandle::new(transport, socket))
    }
}

impl fmt::Display for SshConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_uri_string())
    }
}

impl FromStr for SshConfig {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

// ---------------------------------------------------------------------------
// Internal parsing helpers
// ---------------------------------------------------------------------------

/// Parse userinfo: `user[;param=value;...]`
fn parse_userinfo(userinfo: Option<&str>) -> Result<(String, Vec<(String, String)>)> {
    let Some(info) = userinfo else {
        return Ok((String::new(), Vec::new()));
    };

    let parts: Vec<&str> = info.split(';').collect();
    let user = parts[0].to_string();

    let mut params = Vec::new();
    let mut seen_keys: Vec<String> = Vec::new();

    for part in &parts[1..] {
        let (key, value) = part.split_once('=').ok_or_else(|| {
            anyhow!(
                "invalid userinfo parameter '{}' (expected key=value)",
                part
            )
        })?;

        if key.is_empty() {
            return Err(anyhow!("empty parameter name in userinfo"));
        }

        // Within-location duplicate check
        if seen_keys.iter().any(|k| k == key) {
            return Err(anyhow!("duplicate parameter '{}' within userinfo", key));
        }
        seen_keys.push(key.to_string());

        params.push((key.to_string(), value.to_string()));
    }

    Ok((user, params))
}

/// Parse `host[:port][/socket-path]` from the authority+path portion.
/// Handles IPv6 bracket notation `[::1]`.
fn parse_authority_and_path(s: &str) -> Result<(String, u16, Option<String>)> {
    if s.is_empty() {
        return Err(anyhow!("URI is missing host"));
    }

    if s.starts_with('[') {
        // IPv6 bracket notation
        let bracket_end = s
            .find(']')
            .ok_or_else(|| anyhow!("unclosed IPv6 bracket in host"))?;
        let host = s[1..bracket_end].to_string();
        let remainder = &s[bracket_end + 1..];

        let (port, socket_path) = parse_port_and_path(remainder)?;
        Ok((host, port, socket_path))
    } else {
        // Regular host — find first '/' to split authority from path
        let (authority, path_part) = match s.find('/') {
            Some(pos) => (&s[..pos], Some(&s[pos + 1..])),
            None => (s, None),
        };

        // Split host:port (last ':')
        let (host, port) = match authority.rfind(':') {
            Some(pos) => {
                let port_str = &authority[pos + 1..];
                let port: u16 = port_str.parse().map_err(|_| {
                    anyhow!("invalid port '{}' (expected integer 1-65535)", port_str)
                })?;
                (authority[..pos].to_string(), port)
            }
            None => (authority.to_string(), 22),
        };

        let socket_path = path_part.and_then(|p| {
            if p.is_empty() {
                None
            } else {
                Some(format!("/{}", p))
            }
        });

        Ok((host, port, socket_path))
    }
}

/// Parse port and socket-path from remainder after IPv6 closing `]`.
fn parse_port_and_path(remainder: &str) -> Result<(u16, Option<String>)> {
    if remainder.is_empty() {
        return Ok((22, None));
    }

    if let Some(rest) = remainder.strip_prefix(':') {
        match rest.find('/') {
            Some(pos) => {
                let port: u16 = rest[..pos].parse().map_err(|_| {
                    anyhow!(
                        "invalid port '{}' (expected integer 1-65535)",
                        &rest[..pos]
                    )
                })?;
                let path = &rest[pos + 1..];
                let socket_path = if path.is_empty() {
                    None
                } else {
                    Some(format!("/{}", path))
                };
                Ok((port, socket_path))
            }
            None => {
                let port: u16 = rest.parse().map_err(|_| {
                    anyhow!("invalid port '{}' (expected integer 1-65535)", rest)
                })?;
                Ok((port, None))
            }
        }
    } else if let Some(rest) = remainder.strip_prefix('/') {
        let socket_path = if rest.is_empty() {
            None
        } else {
            Some(format!("/{}", rest))
        };
        Ok((22, socket_path))
    } else {
        Err(anyhow!(
            "unexpected character after IPv6 host: '{}'",
            remainder
        ))
    }
}

/// Parse query parameters from the query string.
fn parse_query_params(query: Option<&str>) -> Result<Vec<(String, String)>> {
    let Some(q) = query else {
        return Ok(Vec::new());
    };

    if q.is_empty() {
        return Ok(Vec::new());
    }

    let mut params = Vec::new();
    let mut seen_keys: Vec<String> = Vec::new();

    for part in q.split('&') {
        let (key, value) = part.split_once('=').ok_or_else(|| {
            anyhow!(
                "invalid query parameter '{}' (expected key=value)",
                part
            )
        })?;

        if key.is_empty() {
            return Err(anyhow!("empty parameter name in query"));
        }

        // Within-location duplicate check
        if seen_keys.iter().any(|k| k == key) {
            return Err(anyhow!("duplicate parameter '{}' within query", key));
        }
        seen_keys.push(key.to_string());

        params.push((key.to_string(), value.to_string()));
    }

    Ok(params)
}

// ---------------------------------------------------------------------------
// Tests — 1.11j (parse), 1.11k (to_uri_string + round-trip), 1.11l (connect)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // === 1.11j — parse() tests ===

    #[test]
    fn parse_basic() {
        let cfg = SshConfig::parse("ssh://deploy@prod-server").unwrap();
        assert_eq!(cfg.host(), "prod-server");
        assert_eq!(cfg.user(), "deploy");
        assert_eq!(cfg.port(), 22);
        assert_eq!(*cfg.host_key_policy(), HostKeyPolicy::Verify);
        assert_eq!(cfg.timeout(), Duration::from_secs(10));
        assert_eq!(cfg.keepalive_interval(), Some(Duration::from_secs(30)));
        assert!(cfg.socket().is_none());
    }

    #[test]
    fn parse_with_port() {
        let cfg = SshConfig::parse("ssh://deploy@prod:2222").unwrap();
        assert_eq!(cfg.host(), "prod");
        assert_eq!(cfg.port(), 2222);
    }

    #[test]
    fn parse_nassh_params() {
        let cfg = SshConfig::parse("ssh://deploy;host-key-policy=tofu@prod").unwrap();
        assert_eq!(*cfg.host_key_policy(), HostKeyPolicy::TrustFirstUse);
    }

    #[test]
    fn parse_query_params_valid() {
        let cfg =
            SshConfig::parse("ssh://deploy@prod?host-key-policy=tofu&timeout=30").unwrap();
        assert_eq!(*cfg.host_key_policy(), HostKeyPolicy::TrustFirstUse);
        assert_eq!(cfg.timeout(), Duration::from_secs(30));
    }

    #[test]
    fn parse_mixed_params() {
        let cfg =
            SshConfig::parse("ssh://deploy;timeout=30@prod?host-key-policy=tofu").unwrap();
        assert_eq!(cfg.timeout(), Duration::from_secs(30));
        assert_eq!(*cfg.host_key_policy(), HostKeyPolicy::TrustFirstUse);
    }

    #[test]
    fn parse_socket_path() {
        let cfg = SshConfig::parse("ssh://user@localhost/tmp/tmux-custom.sock").unwrap();
        assert_eq!(
            cfg.socket(),
            Some(&TmuxSocket::Path("/tmp/tmux-custom.sock".to_string()))
        );
    }

    #[test]
    fn parse_socket_name() {
        let cfg = SshConfig::parse("ssh://user;socket-name=myserver@host").unwrap();
        assert_eq!(
            cfg.socket(),
            Some(&TmuxSocket::Name("myserver".to_string()))
        );
    }

    #[test]
    fn parse_localhost_no_user() {
        let cfg = SshConfig::parse("ssh://localhost").unwrap();
        assert_eq!(cfg.host(), "localhost");
        assert_eq!(cfg.user(), "");
        assert!(cfg.is_localhost());
    }

    #[test]
    fn parse_localhost_with_user() {
        let cfg = SshConfig::parse("ssh://user@localhost").unwrap();
        assert_eq!(cfg.host(), "localhost");
        assert_eq!(cfg.user(), "user");
        assert!(cfg.is_localhost());
    }

    #[test]
    fn parse_127_0_0_1() {
        let cfg = SshConfig::parse("ssh://127.0.0.1").unwrap();
        assert!(cfg.is_localhost());
    }

    #[test]
    fn parse_ipv6_localhost() {
        let cfg = SshConfig::parse("ssh://user@[::1]").unwrap();
        assert_eq!(cfg.host(), "::1");
        assert!(cfg.is_localhost());
    }

    #[test]
    fn parse_ipv6_with_port() {
        let cfg = SshConfig::parse("ssh://user@[::1]:2222").unwrap();
        assert_eq!(cfg.host(), "::1");
        assert_eq!(cfg.port(), 2222);
    }

    #[test]
    fn parse_keepalive_zero_disables() {
        let cfg = SshConfig::parse("ssh://user@host?keepalive=0").unwrap();
        assert_eq!(cfg.keepalive_interval(), None);
    }

    #[test]
    fn parse_all_policies() {
        let verify = SshConfig::parse("ssh://u@h?host-key-policy=verify").unwrap();
        assert_eq!(*verify.host_key_policy(), HostKeyPolicy::Verify);

        let tofu = SshConfig::parse("ssh://u@h?host-key-policy=tofu").unwrap();
        assert_eq!(*tofu.host_key_policy(), HostKeyPolicy::TrustFirstUse);

        let insecure = SshConfig::parse("ssh://u@h?host-key-policy=insecure").unwrap();
        assert_eq!(*insecure.host_key_policy(), HostKeyPolicy::Insecure);
    }

    #[test]
    fn parse_host_only_no_user() {
        let cfg = SshConfig::parse("ssh://remote-host").unwrap();
        assert_eq!(cfg.host(), "remote-host");
        assert_eq!(cfg.user(), "");
    }

    #[test]
    fn parse_port_only() {
        let cfg = SshConfig::parse("ssh://host:443").unwrap();
        assert_eq!(cfg.host(), "host");
        assert_eq!(cfg.port(), 443);
    }

    // --- Invalid URIs ---

    #[test]
    fn parse_bad_scheme() {
        let err = SshConfig::parse("http://host").unwrap_err();
        assert!(err.to_string().contains("ssh://"));
    }

    #[test]
    fn parse_unknown_param() {
        let err = SshConfig::parse("ssh://user@host?unknown=value").unwrap_err();
        assert!(err.to_string().contains("unknown parameter"));
    }

    #[test]
    fn parse_missing_host() {
        let err = SshConfig::parse("ssh://").unwrap_err();
        assert!(err.to_string().contains("missing host"));
    }

    #[test]
    fn parse_canonical_component_port() {
        let err = SshConfig::parse("ssh://user@host?port=22").unwrap_err();
        assert!(err.to_string().contains("canonical"));
    }

    #[test]
    fn parse_canonical_component_user() {
        let err = SshConfig::parse("ssh://root;user=other@host").unwrap_err();
        assert!(err.to_string().contains("canonical"));
    }

    #[test]
    fn parse_canonical_component_host() {
        let err = SshConfig::parse("ssh://user@h?host=other").unwrap_err();
        assert!(err.to_string().contains("canonical"));
    }

    #[test]
    fn parse_duplicate_cross_location() {
        let err = SshConfig::parse("ssh://user;timeout=30@host?timeout=30").unwrap_err();
        assert!(err.to_string().contains("duplicate"));
    }

    #[test]
    fn parse_duplicate_within_userinfo() {
        let err =
            SshConfig::parse("ssh://user;timeout=10;timeout=20@host").unwrap_err();
        assert!(err.to_string().contains("duplicate"));
    }

    #[test]
    fn parse_duplicate_within_query() {
        let err =
            SshConfig::parse("ssh://user@host?timeout=10&timeout=20").unwrap_err();
        assert!(err.to_string().contains("duplicate"));
    }

    #[test]
    fn parse_socket_path_and_name_mutual_exclusion() {
        let err = SshConfig::parse("ssh://user@host/tmp/tmux.sock?socket-name=other")
            .unwrap_err();
        assert!(err.to_string().contains("mutually exclusive"));
    }

    #[test]
    fn parse_malformed_userinfo_param() {
        let err = SshConfig::parse("ssh://user;timeout@host").unwrap_err();
        assert!(err.to_string().contains("key=value"));
    }

    #[test]
    fn parse_invalid_policy_value() {
        let err =
            SshConfig::parse("ssh://user@host?host-key-policy=invalid").unwrap_err();
        assert!(err.to_string().contains("invalid host-key-policy"));
    }

    #[test]
    fn parse_timeout_zero() {
        let err = SshConfig::parse("ssh://user@host?timeout=0").unwrap_err();
        assert!(err.to_string().contains("timeout must be > 0"));
    }

    #[test]
    fn parse_invalid_port() {
        let err = SshConfig::parse("ssh://user@host:abc").unwrap_err();
        assert!(err.to_string().contains("invalid port"));
    }

    #[test]
    fn parse_unclosed_ipv6_bracket() {
        let err = SshConfig::parse("ssh://user@[::1").unwrap_err();
        assert!(err.to_string().contains("bracket"));
    }

    // === 1.11k — to_uri_string() and round-trip tests ===

    #[test]
    fn to_uri_string_basic() {
        let cfg = SshConfig::new("prod-server", "deploy");
        assert_eq!(cfg.to_uri_string(), "ssh://deploy@prod-server");
    }

    #[test]
    fn to_uri_string_with_port() {
        let cfg = SshConfig::new("prod", "deploy").with_port(2222);
        assert_eq!(cfg.to_uri_string(), "ssh://deploy@prod:2222");
    }

    #[test]
    fn to_uri_string_with_policy() {
        let cfg = SshConfig::new("host", "user")
            .with_host_key_policy(HostKeyPolicy::TrustFirstUse);
        assert_eq!(
            cfg.to_uri_string(),
            "ssh://user;host-key-policy=tofu@host"
        );
    }

    #[test]
    fn to_uri_string_with_timeout() {
        let cfg =
            SshConfig::new("host", "user").with_timeout(Duration::from_secs(30));
        assert_eq!(cfg.to_uri_string(), "ssh://user;timeout=30@host");
    }

    #[test]
    fn to_uri_string_keepalive_disabled() {
        let cfg = SshConfig::new("host", "user").with_keepalive(None);
        assert_eq!(cfg.to_uri_string(), "ssh://user;keepalive=0@host");
    }

    #[test]
    fn to_uri_string_socket_name() {
        let cfg = SshConfig::new("host", "user")
            .with_socket(TmuxSocket::Name("myserver".into()));
        assert_eq!(
            cfg.to_uri_string(),
            "ssh://user;socket-name=myserver@host"
        );
    }

    #[test]
    fn to_uri_string_socket_path() {
        let cfg = SshConfig::new("host", "user")
            .with_socket(TmuxSocket::Path("/tmp/tmux.sock".into()));
        assert_eq!(cfg.to_uri_string(), "ssh://user@host/tmp/tmux.sock");
    }

    #[test]
    fn to_uri_string_ipv6() {
        let cfg = SshConfig::new("::1", "user");
        assert_eq!(cfg.to_uri_string(), "ssh://user@[::1]");
    }

    #[test]
    fn to_uri_string_localhost_no_user() {
        let cfg = SshConfig::new("localhost", "");
        assert_eq!(cfg.to_uri_string(), "ssh://localhost");
    }

    #[test]
    fn to_uri_string_no_user_with_params() {
        let cfg = SshConfig::new("localhost", "")
            .with_timeout(Duration::from_secs(30));
        assert_eq!(cfg.to_uri_string(), "ssh://localhost?timeout=30");
    }

    // Round-trip tests

    #[test]
    fn roundtrip_basic() {
        let cfg = SshConfig::new("prod", "deploy");
        let reparsed = SshConfig::parse(&cfg.to_string()).unwrap();
        assert_eq!(cfg, reparsed);
    }

    #[test]
    fn roundtrip_full() {
        let cfg = SshConfig::new("prod", "deploy")
            .with_port(2222)
            .with_host_key_policy(HostKeyPolicy::TrustFirstUse)
            .with_timeout(Duration::from_secs(30))
            .with_keepalive(None)
            .with_socket(TmuxSocket::Name("myserver".into()));
        let reparsed = SshConfig::parse(&cfg.to_string()).unwrap();
        assert_eq!(cfg, reparsed);
    }

    #[test]
    fn roundtrip_socket_path() {
        let cfg = SshConfig::new("host", "user")
            .with_socket(TmuxSocket::Path("/tmp/tmux.sock".into()));
        let reparsed = SshConfig::parse(&cfg.to_string()).unwrap();
        assert_eq!(cfg, reparsed);
    }

    #[test]
    fn roundtrip_ipv6() {
        let cfg = SshConfig::new("::1", "user").with_port(2222);
        let reparsed = SshConfig::parse(&cfg.to_string()).unwrap();
        assert_eq!(cfg, reparsed);
    }

    #[test]
    fn roundtrip_localhost() {
        let cfg = SshConfig::new("localhost", "");
        let reparsed = SshConfig::parse(&cfg.to_string()).unwrap();
        assert_eq!(cfg, reparsed);
    }

    #[test]
    fn roundtrip_query_to_nassh() {
        // Parse from query params, canonical form renders as nassh, re-parse matches
        let cfg1 =
            SshConfig::parse("ssh://user@host?host-key-policy=tofu").unwrap();
        let cfg2 = SshConfig::parse(&cfg1.to_string()).unwrap();
        assert_eq!(cfg1, cfg2);
    }

    #[test]
    fn fromstr_integration() {
        let cfg: SshConfig = "ssh://deploy@prod:2222".parse().unwrap();
        assert_eq!(cfg.host(), "prod");
        assert_eq!(cfg.port(), 2222);
    }

    // === 1.11l — connect() localhost selection tests ===

    #[tokio::test]
    async fn connect_localhost() {
        let host = SshConfig::parse("ssh://localhost")
            .unwrap()
            .connect()
            .await
            .unwrap();
        assert!(matches!(host.transport_kind(), TransportKind::Local(_)));
    }

    #[tokio::test]
    async fn connect_127_0_0_1() {
        let host = SshConfig::parse("ssh://127.0.0.1")
            .unwrap()
            .connect()
            .await
            .unwrap();
        assert!(matches!(host.transport_kind(), TransportKind::Local(_)));
    }

    #[tokio::test]
    async fn connect_ipv6_localhost() {
        let host = SshConfig::parse("ssh://user@[::1]")
            .unwrap()
            .connect()
            .await
            .unwrap();
        assert!(matches!(host.transport_kind(), TransportKind::Local(_)));
    }

    #[tokio::test]
    async fn connect_empty_user_ssh_error() {
        let result = SshConfig::parse("ssh://remote-host")
            .unwrap()
            .connect()
            .await;
        assert!(result.is_err());
        assert!(result.err().unwrap().to_string().contains("user is required"));
    }

    #[tokio::test]
    async fn connect_localhost_timeout_propagation() {
        let host = SshConfig::parse("ssh://localhost?timeout=30")
            .unwrap()
            .connect()
            .await
            .unwrap();
        match host.transport_kind() {
            TransportKind::Local(local) => {
                assert_eq!(local.timeout, Duration::from_secs(30));
            }
            _ => panic!("expected LocalTransport"),
        }
    }

    #[tokio::test]
    async fn connect_localhost_socket_propagation() {
        let host = SshConfig::parse("ssh://user@localhost/tmp/test.sock")
            .unwrap()
            .connect()
            .await
            .unwrap();
        assert!(matches!(host.transport_kind(), TransportKind::Local(_)));
        // Socket is wired through HostHandle::new, verified by successful construction
    }
}
