//! SSH URI parsing, rendering, and connect for SshConfig (DC21).
//!
//! Extends `SshConfig` (defined in `transport.rs`) with `parse()`,
//! `to_uri_string()`, `connect()`, `Display`, and `FromStr` impls.
//! URI logic is separated from transport logic while consolidating on one type.

use std::fmt;
use std::str::FromStr;

use crate::error::{Error, Result};

use crate::host::HostHandle;
use crate::transport::{LocalTransport, SshConfig, SshTransport, TransportKind, SSH_DEFAULT_PORT};
use crate::types::{HostKeyPolicy, TmuxSocket};

/// Canonical URI component names that cannot appear as parameters.
const CANONICAL_COMPONENTS: &[&str] = &["user", "host", "port"];

/// Known parameter names.
const KNOWN_PARAMS: &[&str] = &[
    "host-key-policy",
    "timeout",
    "inactivity-timeout",
    "keepalive",
    "socket-name",
    "identity-file",
];

/// Parameters restricted to query-string only (DC26).
/// These are rejected if they appear in nassh-style userinfo params.
const QUERY_ONLY_PARAMS: &[&str] = &["identity-file"];

/// Characters that are unsafe in URI user/host/parameter values because they
/// collide with URI or nassh syntax delimiters. `to_uri_string()` validates
/// against this set rather than percent-encoding, keeping the implementation
/// simple and the output human-readable.
const URI_RESERVED_CHARS: &[char] = &[';', '@', '?', '&', '=', '#', '[', ']'];

impl SshConfig {
    /// Parse from an `ssh://` URI string.
    ///
    /// Accepts nassh-style params (`;` in userinfo), query params (`?`),
    /// or both. Rejects unknown params, canonical-component duplication,
    /// and URI-reserved characters (`;@?&=#[]`) in user, host, and
    /// parameter values.
    ///
    /// For localhost URIs, `user` defaults to empty (ignored by LocalTransport).
    ///
    /// Round-trip guarantee: `parse(cfg.to_uri_string()) == cfg` for all
    /// configs produced by `parse()`. Builder-constructed configs with
    /// URI-reserved characters may not round-trip — use DNS-safe hostnames
    /// and POSIX usernames.
    pub fn parse(uri: &str) -> Result<Self> {
        let remainder = uri
            .strip_prefix("ssh://")
            .ok_or_else(|| Error::Parse("URI must start with ssh://".into()))?;

        if remainder.is_empty() {
            return Err(Error::Parse("URI is missing host".into()));
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
            return Err(Error::Parse("URI is missing host".into()));
        }

        // Validate user and host contain no URI-reserved characters
        if !user.is_empty() {
            validate_uri_safe(&user, "user")?;
        }
        validate_uri_safe(&host, "host")?;

        // Parse query params
        let query_params = parse_query_params(query_string)?;

        // Check for canonical component duplication
        for params in [&userinfo_params, &query_params] {
            for (key, _) in params {
                if CANONICAL_COMPONENTS.contains(&key.as_str()) {
                    return Err(Error::Parse(format!(
                        "'{}' is a canonical URI component and cannot appear as a parameter",
                        key
                    )));
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
                return Err(Error::Parse(format!(
                    "duplicate parameter '{}' (appears in both userinfo and query)",
                    key
                )));
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
                return Err(Error::Parse(format!("unknown parameter: '{}'", key)));
            }
        }

        // Reject query-only params that appear in userinfo (DC26)
        for (key, _) in &userinfo_params {
            if QUERY_ONLY_PARAMS.contains(&key.as_str()) {
                return Err(Error::Parse(format!(
                    "'{}' is a query-only parameter and cannot appear in userinfo",
                    key
                )));
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
                            return Err(Error::Parse(format!(
                                "invalid host-key-policy value '{}' \
                                 (expected: verify, tofu, insecure)",
                                value
                            )))
                        }
                    };
                    config = config.with_host_key_policy(policy);
                }
                "timeout" => {
                    let secs: u64 = value.parse().map_err(|_| {
                        Error::Parse(format!(
                            "invalid timeout value '{}' (expected integer seconds)",
                            value
                        ))
                    })?;
                    if secs == 0 {
                        return Err(Error::Parse("timeout must be > 0".into()));
                    }
                    config = config.with_timeout(std::time::Duration::from_secs(secs));
                }
                "inactivity-timeout" => {
                    let secs: u64 = value.parse().map_err(|_| {
                        Error::Parse(format!(
                            "invalid inactivity-timeout value '{}' (expected integer seconds)",
                            value
                        ))
                    })?;
                    let timeout = if secs == 0 {
                        None
                    } else {
                        Some(std::time::Duration::from_secs(secs))
                    };
                    config = config.with_inactivity_timeout(timeout);
                }
                "keepalive" => {
                    let secs: u64 = value.parse().map_err(|_| {
                        Error::Parse(format!(
                            "invalid keepalive value '{}' (expected integer seconds)",
                            value
                        ))
                    })?;
                    let interval = if secs == 0 {
                        None
                    } else {
                        Some(std::time::Duration::from_secs(secs))
                    };
                    config = config.with_keepalive(interval);
                }
                "socket-name" => {
                    if !crate::transport::is_valid_socket_name(value) {
                        return Err(Error::Parse(format!(
                            "invalid socket-name '{}': must match [A-Za-z0-9._-]+",
                            value
                        )));
                    }
                    has_socket_name = true;
                    config = config.with_socket(TmuxSocket::Name(value.to_string()))?;
                }
                "identity-file" => {
                    if value.is_empty() {
                        return Err(Error::Parse("identity-file value cannot be empty".into()));
                    }
                    let path = std::path::PathBuf::from(value);
                    if !path.is_absolute() {
                        return Err(Error::Parse(format!(
                            "identity-file must be an absolute path, got '{}'",
                            value
                        )));
                    }
                    config = config.with_identity_file(path)?;
                }
                _ => unreachable!(),
            }
        }

        // Handle socket-path from URI path component
        if let Some(path) = socket_path {
            if has_socket_name {
                return Err(Error::Parse(
                    "socket-path and socket-name are mutually exclusive".into(),
                ));
            }
            config = config.with_socket(TmuxSocket::Path(path))?;
        }

        Ok(config)
    }

    /// Render to canonical URI form.
    ///
    /// If user is non-empty, non-default parameters are placed as nassh-style
    /// userinfo params. If user is empty, non-default parameters are placed
    /// as query params. Socket paths render as the URI path component.
    ///
    /// This method never panics. Configs built from `parse()` always produce
    /// valid, round-trippable URIs. Builder-constructed configs with
    /// URI-reserved characters in user or host (`;@?&=#[]`) will produce
    /// URIs that may not re-parse correctly — use DNS-safe hostnames and
    /// POSIX usernames for round-trip safety.
    pub fn to_uri_string(&self) -> String {
        let mut result = String::from("ssh://");

        // Collect non-default params, split into nassh-eligible and query-only
        let mut nassh_params: Vec<(&str, String)> = Vec::new();
        let mut query_only_params: Vec<(&str, String)> = Vec::new();

        if *self.host_key_policy() != HostKeyPolicy::Verify {
            let value = match self.host_key_policy() {
                HostKeyPolicy::Verify => "verify",
                HostKeyPolicy::TrustFirstUse => "tofu",
                HostKeyPolicy::Insecure => "insecure",
            };
            nassh_params.push(("host-key-policy", value.to_string()));
        }
        if self.timeout() != std::time::Duration::from_secs(10) {
            nassh_params.push(("timeout", self.timeout().as_secs().to_string()));
        }
        if let Some(timeout) = self.inactivity_timeout() {
            nassh_params.push(("inactivity-timeout", timeout.as_secs().to_string()));
        }
        if self.keepalive_interval() != Some(std::time::Duration::from_secs(30)) {
            let val = match self.keepalive_interval() {
                Some(d) => d.as_secs().to_string(),
                None => "0".to_string(),
            };
            nassh_params.push(("keepalive", val));
        }

        // socket-name goes as nassh-eligible param; socket-path goes as URI path
        let mut socket_path: Option<&str> = None;
        if let Some(socket) = self.socket() {
            match socket {
                TmuxSocket::Name(n) => nassh_params.push(("socket-name", n.clone())),
                TmuxSocket::Path(p) => socket_path = Some(p.as_str()),
            }
        }

        // identity-file is always query-only (DC26)
        if let Some(path) = self.identity_file() {
            query_only_params.push(("identity-file", path.display().to_string()));
        }

        let user = self.user();

        if !user.is_empty() {
            // Nassh-style: user[;param=value...]@host
            result.push_str(user);
            for (key, value) in &nassh_params {
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

        // Query params: query-only params always, nassh params when user is empty
        let all_query: Vec<(&str, &str)> = if user.is_empty() {
            nassh_params
                .iter()
                .chain(query_only_params.iter())
                .map(|(k, v)| (*k, v.as_str()))
                .collect()
        } else {
            query_only_params
                .iter()
                .map(|(k, v)| (*k, v.as_str()))
                .collect()
        };

        if !all_query.is_empty() {
            result.push('?');
            for (i, (key, value)) in all_query.iter().enumerate() {
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

    /// Stable, human-readable endpoint identity for display and Fleet/output aliases.
    ///
    /// The alias includes SSH user, host, non-default port, and non-default tmux
    /// socket identity. It intentionally omits identity-file because key paths are
    /// sensitive and too noisy for labels.
    pub fn endpoint_alias(&self) -> String {
        let mut alias = String::new();

        if !self.user().is_empty() {
            alias.push_str(self.user());
            alias.push('@');
        }

        let host = self.host();
        if host.contains(':') {
            alias.push('[');
            alias.push_str(host);
            alias.push(']');
        } else {
            alias.push_str(host);
        }

        if self.port() != SSH_DEFAULT_PORT {
            alias.push(':');
            alias.push_str(&self.port().to_string());
        }

        if let Some(socket) = self.socket() {
            match socket {
                TmuxSocket::Name(name) => {
                    alias.push_str("/socket:");
                    alias.push_str(name);
                }
                TmuxSocket::Path(path) => {
                    alias.push_str("/socket-path:");
                    alias.push_str(path);
                }
            }
        }

        alias
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
        let alias = self.host().to_string();
        self.connect_with_alias(&alias).await
    }

    /// Connect using `endpoint_alias()` as the HostHandle alias.
    pub async fn connect_with_endpoint_alias(self) -> Result<HostHandle> {
        let alias = self.endpoint_alias();
        self.connect_with_alias(&alias).await
    }

    /// Connect and return a `HostHandle` using an explicit Fleet/output alias.
    ///
    /// This is useful when the transport host name (`localhost`, `amd2`, an IP)
    /// is not the caller's stable routing alias.
    pub async fn connect_with_alias(self, alias: &str) -> Result<HostHandle> {
        if alias.is_empty() {
            return Err(Error::Parse("host alias cannot be empty".to_string()));
        }
        let is_local = self.is_localhost();
        let user_empty = self.user().is_empty();
        let timeout = self.timeout();
        let socket = self.socket().cloned();

        if is_local {
            let transport = TransportKind::Local(LocalTransport::with_timeout(timeout));
            return Ok(HostHandle::with_alias(transport, socket, alias));
        }

        if user_empty {
            return Err(Error::Parse("user is required for SSH connections".into()));
        }

        let transport = TransportKind::Ssh(SshTransport::connect(self).await?);
        Ok(HostHandle::with_alias(transport, socket, alias))
    }
}

impl fmt::Display for SshConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_uri_string())
    }
}

impl FromStr for SshConfig {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        Self::parse(s)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Validate a parsed port is in the valid range 1-65535.
fn validate_port(port: u16) -> Result<u16> {
    if port == 0 {
        return Err(Error::Parse(
            "invalid port '0' (expected integer 1-65535)".into(),
        ));
    }
    Ok(port)
}

/// Check that a string contains no URI-reserved characters.
fn validate_uri_safe(value: &str, context: &str) -> Result<()> {
    for ch in URI_RESERVED_CHARS {
        if value.contains(*ch) {
            return Err(Error::Parse(format!(
                "{} contains URI-reserved character '{}' \
                 (disallowed: ; @ ? & = # [ ])",
                context, ch
            )));
        }
    }
    Ok(())
}

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
            Error::Parse(format!(
                "invalid userinfo parameter '{}' (expected key=value)",
                part
            ))
        })?;

        if key.is_empty() {
            return Err(Error::Parse("empty parameter name in userinfo".into()));
        }

        // Within-location duplicate check
        if seen_keys.iter().any(|k| k == key) {
            return Err(Error::Parse(format!(
                "duplicate parameter '{}' within userinfo",
                key
            )));
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
        return Err(Error::Parse("URI is missing host".into()));
    }

    if s.starts_with('[') {
        // IPv6 bracket notation
        let bracket_end = s
            .find(']')
            .ok_or_else(|| Error::Parse("unclosed IPv6 bracket in host".into()))?;
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
                    Error::Parse(format!(
                        "invalid port '{}' (expected integer 1-65535)",
                        port_str
                    ))
                })?;
                (authority[..pos].to_string(), validate_port(port)?)
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
                    Error::Parse(format!(
                        "invalid port '{}' (expected integer 1-65535)",
                        &rest[..pos]
                    ))
                })?;
                let port = validate_port(port)?;
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
                    Error::Parse(format!(
                        "invalid port '{}' (expected integer 1-65535)",
                        rest
                    ))
                })?;
                Ok((validate_port(port)?, None))
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
        Err(Error::Parse(format!(
            "unexpected character after IPv6 host: '{}'",
            remainder
        )))
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
            Error::Parse(format!(
                "invalid query parameter '{}' (expected key=value)",
                part
            ))
        })?;

        if key.is_empty() {
            return Err(Error::Parse("empty parameter name in query".into()));
        }

        // Within-location duplicate check
        if seen_keys.iter().any(|k| k == key) {
            return Err(Error::Parse(format!(
                "duplicate parameter '{}' within query",
                key
            )));
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
        assert_eq!(cfg.inactivity_timeout(), None);
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
        let cfg = SshConfig::parse("ssh://deploy@prod?host-key-policy=tofu&timeout=30").unwrap();
        assert_eq!(*cfg.host_key_policy(), HostKeyPolicy::TrustFirstUse);
        assert_eq!(cfg.timeout(), Duration::from_secs(30));
    }

    #[test]
    fn parse_inactivity_timeout() {
        let cfg =
            SshConfig::parse("ssh://deploy@prod?inactivity-timeout=120&keepalive=15").unwrap();
        assert_eq!(cfg.inactivity_timeout(), Some(Duration::from_secs(120)));
        assert_eq!(cfg.keepalive_interval(), Some(Duration::from_secs(15)));
    }

    #[test]
    fn parse_mixed_params() {
        let cfg = SshConfig::parse("ssh://deploy;timeout=30@prod?host-key-policy=tofu").unwrap();
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
    fn endpoint_alias_includes_user_host_and_non_default_port() {
        let cfg = SshConfig::parse("ssh://david@amd1:2222").unwrap();
        assert_eq!(cfg.endpoint_alias(), "david@amd1:2222");
    }

    #[test]
    fn endpoint_alias_distinguishes_users_on_same_host() {
        let david = SshConfig::parse("ssh://david@amd1").unwrap();
        let alice = SshConfig::parse("ssh://alice@amd1").unwrap();

        assert_eq!(david.endpoint_alias(), "david@amd1");
        assert_eq!(alice.endpoint_alias(), "alice@amd1");
        assert_ne!(david.endpoint_alias(), alice.endpoint_alias());
    }

    #[test]
    fn endpoint_alias_includes_socket_identity() {
        let named = SshConfig::parse("ssh://david;socket-name=build@amd1").unwrap();
        assert_eq!(named.endpoint_alias(), "david@amd1/socket:build");

        let path = SshConfig::parse("ssh://david@amd1/tmp/tmux.sock").unwrap();
        assert_eq!(
            path.endpoint_alias(),
            "david@amd1/socket-path:/tmp/tmux.sock"
        );
    }

    #[test]
    fn endpoint_alias_omits_identity_file() {
        let cfg = SshConfig::parse("ssh://david@amd1?identity-file=/Users/david/.ssh/motliehost")
            .unwrap();
        assert_eq!(cfg.endpoint_alias(), "david@amd1");
    }

    #[test]
    fn endpoint_alias_brackets_ipv6_hosts() {
        let cfg = SshConfig::parse("ssh://user@[::1]:2222").unwrap();
        assert_eq!(cfg.endpoint_alias(), "user@[::1]:2222");
    }

    #[tokio::test]
    async fn connect_with_endpoint_alias_sets_host_alias() {
        let host = SshConfig::parse("ssh://localhost")
            .unwrap()
            .connect_with_endpoint_alias()
            .await
            .unwrap();
        assert_eq!(host.host_alias(), "localhost");
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
        let err = SshConfig::parse("ssh://user;timeout=10;timeout=20@host").unwrap_err();
        assert!(err.to_string().contains("duplicate"));
    }

    #[test]
    fn parse_duplicate_within_query() {
        let err = SshConfig::parse("ssh://user@host?timeout=10&timeout=20").unwrap_err();
        assert!(err.to_string().contains("duplicate"));
    }

    #[test]
    fn parse_socket_path_and_name_mutual_exclusion() {
        let err = SshConfig::parse("ssh://user@host/tmp/tmux.sock?socket-name=other").unwrap_err();
        assert!(err.to_string().contains("mutually exclusive"));
    }

    #[test]
    fn parse_malformed_userinfo_param() {
        let err = SshConfig::parse("ssh://user;timeout@host").unwrap_err();
        assert!(err.to_string().contains("key=value"));
    }

    #[test]
    fn parse_invalid_policy_value() {
        let err = SshConfig::parse("ssh://user@host?host-key-policy=invalid").unwrap_err();
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

    #[test]
    fn parse_port_zero() {
        let err = SshConfig::parse("ssh://user@host:0").unwrap_err();
        assert!(err.to_string().contains("invalid port"));
    }

    #[test]
    fn parse_port_zero_ipv6() {
        let err = SshConfig::parse("ssh://user@[::1]:0").unwrap_err();
        assert!(err.to_string().contains("invalid port"));
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
        let cfg = SshConfig::new("host", "user").with_host_key_policy(HostKeyPolicy::TrustFirstUse);
        assert_eq!(cfg.to_uri_string(), "ssh://user;host-key-policy=tofu@host");
    }

    #[test]
    fn to_uri_string_with_timeout() {
        let cfg = SshConfig::new("host", "user").with_timeout(Duration::from_secs(30));
        assert_eq!(cfg.to_uri_string(), "ssh://user;timeout=30@host");
    }

    #[test]
    fn to_uri_string_with_inactivity_timeout() {
        let cfg =
            SshConfig::new("host", "user").with_inactivity_timeout(Some(Duration::from_secs(120)));
        assert_eq!(
            cfg.to_uri_string(),
            "ssh://user;inactivity-timeout=120@host"
        );
    }

    #[test]
    fn to_uri_string_keepalive_disabled() {
        let cfg = SshConfig::new("host", "user").with_keepalive(None);
        assert_eq!(cfg.to_uri_string(), "ssh://user;keepalive=0@host");
    }

    #[test]
    fn to_uri_string_socket_name() {
        let cfg = SshConfig::new("host", "user")
            .with_socket(TmuxSocket::Name("myserver".into()))
            .unwrap();
        assert_eq!(cfg.to_uri_string(), "ssh://user;socket-name=myserver@host");
    }

    #[test]
    fn to_uri_string_socket_path() {
        let cfg = SshConfig::new("host", "user")
            .with_socket(TmuxSocket::Path("/tmp/tmux.sock".into()))
            .unwrap();
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
        let cfg = SshConfig::new("localhost", "").with_timeout(Duration::from_secs(30));
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
            .with_socket(TmuxSocket::Name("myserver".into()))
            .unwrap();
        let reparsed = SshConfig::parse(&cfg.to_string()).unwrap();
        assert_eq!(cfg, reparsed);
    }

    #[test]
    fn roundtrip_socket_path() {
        let cfg = SshConfig::new("host", "user")
            .with_socket(TmuxSocket::Path("/tmp/tmux.sock".into()))
            .unwrap();
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
        let cfg1 = SshConfig::parse("ssh://user@host?host-key-policy=tofu").unwrap();
        let cfg2 = SshConfig::parse(&cfg1.to_string()).unwrap();
        assert_eq!(cfg1, cfg2);
    }

    #[test]
    fn fromstr_integration() {
        let cfg: SshConfig = "ssh://deploy@prod:2222".parse().unwrap();
        assert_eq!(cfg.host(), "prod");
        assert_eq!(cfg.port(), 2222);
    }

    #[test]
    fn parse_reserved_char_in_user() {
        let err = SshConfig::parse("ssh://user;semi@host").unwrap_err();
        // "user;semi" splits on ';' — "semi" is parsed as a param without '='
        assert!(err.to_string().contains("key=value"));
    }

    #[test]
    fn parse_reserved_char_at_in_user() {
        // ssh://user@host@evil — rfind('@') splits at last '@':
        // userinfo="user@host", host="evil". User "user@host" contains '@' → rejected.
        let err = SshConfig::parse("ssh://user@host@evil").unwrap_err();
        assert!(err.to_string().contains("URI-reserved"));
    }

    #[test]
    fn parse_reserved_hash_in_host() {
        let err = SshConfig::parse("ssh://user@host#frag").unwrap_err();
        assert!(err.to_string().contains("URI-reserved"));
    }

    #[test]
    fn parse_reserved_semicolon_in_host() {
        let err = SshConfig::parse("ssh://host;semi").unwrap_err();
        // No '@' so no userinfo split — entire "host;semi" is the authority
        // ';' is reserved → rejected
        assert!(err.to_string().contains("URI-reserved"));
    }

    #[test]
    fn parse_reserved_in_socket_name() {
        let err = SshConfig::parse("ssh://user;socket-name=my;server@host").unwrap_err();
        // ';' in socket-name value — ';' splits userinfo params so this
        // parses socket-name=my, then "server" as another param without '='
        let msg = err.to_string();
        assert!(msg.contains("key=value") || msg.contains("unknown"));
    }

    #[test]
    fn parse_invalid_socket_name_chars() {
        let err = SshConfig::parse("ssh://user@host?socket-name=my%20server").unwrap_err();
        assert!(err.to_string().contains("invalid socket-name"));
    }

    #[test]
    fn parse_valid_socket_name() {
        let cfg = SshConfig::parse("ssh://user@host?socket-name=my-server_v2.0").unwrap();
        assert_eq!(
            cfg.socket(),
            Some(&TmuxSocket::Name("my-server_v2.0".to_string()))
        );
    }

    #[test]
    fn builder_invalid_socket_name() {
        let result =
            SshConfig::new("host", "user").with_socket(TmuxSocket::Name("bad;name".into()));
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("invalid socket name"));
    }

    #[test]
    fn builder_valid_socket_name() {
        let cfg = SshConfig::new("host", "user")
            .with_socket(TmuxSocket::Name("good-name_v2.0".into()))
            .unwrap();
        assert_eq!(
            cfg.socket(),
            Some(&TmuxSocket::Name("good-name_v2.0".into()))
        );
    }

    #[test]
    fn to_uri_string_no_panic_on_reserved() {
        // Builder-constructed config with reserved chars — to_uri_string()
        // must not panic (produces best-effort, non-round-trippable URI)
        let cfg = SshConfig::new("host", "user;semi");
        let uri = cfg.to_uri_string(); // must not panic
        assert!(uri.contains("user;semi")); // emitted raw
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
    async fn connect_with_alias_sets_host_alias() {
        let host = SshConfig::parse("ssh://localhost")
            .unwrap()
            .connect_with_alias("local")
            .await
            .unwrap();
        assert!(matches!(host.transport_kind(), TransportKind::Local(_)));
        assert_eq!(host.host_alias(), "local");
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
        assert!(result
            .err()
            .unwrap()
            .to_string()
            .contains("user is required"));
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

    // === 1.15d — identity-file (DC26) tests ===

    #[test]
    fn parse_identity_file_query() {
        let cfg = SshConfig::parse("ssh://deploy@host?identity-file=/home/deploy/.ssh/id_ed25519")
            .unwrap();
        assert_eq!(
            cfg.identity_file(),
            Some(std::path::Path::new("/home/deploy/.ssh/id_ed25519"))
        );
    }

    #[test]
    fn parse_identity_file_nassh_rejected() {
        let err = SshConfig::parse("ssh://deploy;identity-file=/path/to/key@host").unwrap_err();
        assert!(err.to_string().contains("query-only"));
    }

    #[test]
    fn parse_identity_file_relative_path_rejected() {
        let err = SshConfig::parse("ssh://deploy@host?identity-file=relative/key").unwrap_err();
        assert!(err.to_string().contains("absolute path"));
    }

    #[test]
    fn parse_identity_file_empty_rejected() {
        let err = SshConfig::parse("ssh://deploy@host?identity-file=").unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn parse_identity_file_mixed_params() {
        // nassh timeout + query identity-file
        let cfg =
            SshConfig::parse("ssh://deploy;timeout=30@host?identity-file=/keys/deploy").unwrap();
        assert_eq!(cfg.timeout(), Duration::from_secs(30));
        assert_eq!(
            cfg.identity_file(),
            Some(std::path::Path::new("/keys/deploy"))
        );
    }

    #[test]
    fn roundtrip_identity_file_with_user() {
        let cfg = SshConfig::parse("ssh://deploy@host?identity-file=/keys/deploy").unwrap();
        let reparsed = SshConfig::parse(&cfg.to_string()).unwrap();
        assert_eq!(cfg, reparsed);
    }

    #[test]
    fn roundtrip_identity_file_no_user() {
        let cfg = SshConfig::parse("ssh://localhost?identity-file=/keys/deploy").unwrap();
        let reparsed = SshConfig::parse(&cfg.to_string()).unwrap();
        assert_eq!(cfg, reparsed);
    }

    #[test]
    fn roundtrip_identity_file_with_other_params() {
        let cfg =
            SshConfig::parse("ssh://deploy;timeout=30@host?identity-file=/keys/deploy").unwrap();
        let reparsed = SshConfig::parse(&cfg.to_string()).unwrap();
        assert_eq!(cfg, reparsed);
    }

    #[test]
    fn to_uri_string_identity_file_with_user() {
        // With user: nassh params in userinfo, identity-file in query
        let cfg = SshConfig::new("host", "deploy")
            .with_identity_file("/keys/deploy")
            .unwrap();
        let uri = cfg.to_uri_string();
        assert_eq!(uri, "ssh://deploy@host?identity-file=/keys/deploy");
    }

    #[test]
    fn to_uri_string_identity_file_no_user() {
        let cfg = SshConfig::new("localhost", "")
            .with_identity_file("/keys/deploy")
            .unwrap();
        let uri = cfg.to_uri_string();
        assert_eq!(uri, "ssh://localhost?identity-file=/keys/deploy");
    }

    #[test]
    fn to_uri_string_identity_file_with_nassh_params() {
        // Other params go to nassh, identity-file goes to query
        let cfg = SshConfig::new("host", "deploy")
            .with_timeout(Duration::from_secs(30))
            .with_identity_file("/keys/deploy")
            .unwrap();
        let uri = cfg.to_uri_string();
        assert_eq!(
            uri,
            "ssh://deploy;timeout=30@host?identity-file=/keys/deploy"
        );
    }

    #[test]
    fn to_uri_string_identity_file_no_user_with_other_params() {
        let cfg = SshConfig::new("localhost", "")
            .with_timeout(Duration::from_secs(30))
            .with_identity_file("/keys/deploy")
            .unwrap();
        let uri = cfg.to_uri_string();
        assert_eq!(uri, "ssh://localhost?timeout=30&identity-file=/keys/deploy");
    }

    #[test]
    fn builder_identity_file() {
        let cfg = SshConfig::new("host", "deploy")
            .with_identity_file("/keys/deploy")
            .unwrap();
        assert_eq!(
            cfg.identity_file(),
            Some(std::path::Path::new("/keys/deploy"))
        );
    }

    #[test]
    fn builder_identity_file_default_none() {
        let cfg = SshConfig::new("host", "deploy");
        assert_eq!(cfg.identity_file(), None);
    }

    #[test]
    fn builder_identity_file_duplicate_errors() {
        let err = SshConfig::new("host", "deploy")
            .with_identity_file("/keys/a")
            .unwrap()
            .with_identity_file("/keys/b")
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("/keys/a"));
        assert!(msg.contains("/keys/b"));
    }

    #[test]
    fn parse_then_builder_identity_file_duplicate_errors() {
        let err = SshConfig::parse("ssh://deploy@host?identity-file=/keys/a")
            .unwrap()
            .with_identity_file("/keys/b")
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("/keys/a"));
        assert!(msg.contains("/keys/b"));
    }

    #[tokio::test]
    async fn connect_localhost_with_identity_file() {
        // identity-file is silently ignored for localhost (LocalTransport)
        let host = SshConfig::parse("ssh://localhost?identity-file=/nonexistent/key")
            .unwrap()
            .connect()
            .await
            .unwrap();
        assert!(matches!(host.transport_kind(), TransportKind::Local(_)));
    }
}
