use anyhow::{anyhow, Result};
use regex::Regex;
use std::fmt;

/// Authoritative pane identifier using tmux's `#{pane_id}` (%<id>).
/// Display fields (session, window, pane) are metadata for human readability.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct PaneAddress {
    /// Stable tmux pane id, e.g. "%12"
    pub pane_id: String,
    /// Session name (display)
    pub session: String,
    /// Window index (display)
    pub window: u32,
    /// Pane index (display)
    pub pane: u32,
}

impl PaneAddress {
    /// Canonical tmux target string: "session:window.pane"
    pub fn to_tmux_target(&self) -> String {
        format!("{}:{}.{}", self.session, self.window, self.pane)
    }

    /// The stable pane_id for keying (e.g. "%12")
    pub fn id(&self) -> &str {
        &self.pane_id
    }

    /// Parse from tab-delimited tmux list-panes output.
    /// Expected format: `%id\tsession:window.pane`
    pub fn parse(pane_id: &str, address_str: &str) -> Result<Self> {
        // address_str is "session_name:window_index.pane_index"
        let colon_pos = address_str
            .rfind(':')
            .ok_or_else(|| anyhow!("invalid pane address: missing ':'"))?;
        let session = address_str[..colon_pos].to_string();
        let rest = &address_str[colon_pos + 1..];
        let dot_pos = rest
            .find('.')
            .ok_or_else(|| anyhow!("invalid pane address: missing '.'"))?;
        let window: u32 = rest[..dot_pos]
            .parse()
            .map_err(|_| anyhow!("invalid window index"))?;
        let pane: u32 = rest[dot_pos + 1..]
            .parse()
            .map_err(|_| anyhow!("invalid pane index"))?;

        Ok(PaneAddress {
            pane_id: pane_id.to_string(),
            session,
            window,
            pane,
        })
    }
}

impl fmt::Display for PaneAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}.{}", self.session, self.window, self.pane)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionInfo {
    pub name: String,
    pub id: String,
    pub created: u64,
    pub attached: bool,
    pub window_count: u32,
    pub group: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WindowInfo {
    pub session_id: String,
    pub session_name: String,
    pub index: u32,
    pub name: String,
    pub active: bool,
    pub pane_count: u32,
    pub layout: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PaneInfo {
    pub address: PaneAddress,
    pub title: String,
    pub current_command: String,
    pub pid: u32,
    pub width: u32,
    pub height: u32,
    pub active: bool,
}

/// Unified target address at any hierarchy level (DC16).
#[derive(Debug, Clone)]
pub enum TargetAddress {
    Session(SessionInfo),
    Window(WindowInfo),
    Pane(PaneAddress),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetLevel {
    Session,
    Window,
    Pane,
}

/// Builder for tmux target strings (DC17).
///
/// Fields are private to enforce the hierarchy invariant: pane requires window.
/// Use `session()`, `.window()/.window_name()`, `.pane()` builders or `parse()`.
#[derive(Debug, Clone)]
pub struct TargetSpec {
    session_name: String,
    window_sel: Option<String>,
    pane_idx: Option<u32>,
}

impl TargetSpec {
    pub fn session(name: &str) -> Self {
        TargetSpec {
            session_name: name.to_string(),
            window_sel: None,
            pane_idx: None,
        }
    }

    pub fn window(mut self, index: u32) -> Self {
        self.window_sel = Some(index.to_string());
        self
    }

    pub fn window_name(mut self, name: &str) -> Self {
        self.window_sel = Some(name.to_string());
        self
    }

    /// Set pane index. Panics if `window` has not been set — pane requires
    /// a window context (tmux target hierarchy: session:window.pane).
    pub fn pane(mut self, index: u32) -> Self {
        assert!(
            self.window_sel.is_some(),
            "TargetSpec::pane() requires window to be set first (use .window() or .window_name() before .pane())"
        );
        self.pane_idx = Some(index);
        self
    }

    // --- Accessors ---

    pub fn session_name(&self) -> &str {
        &self.session_name
    }

    pub fn window_selector(&self) -> Option<&str> {
        self.window_sel.as_deref()
    }

    pub fn pane_index(&self) -> Option<u32> {
        self.pane_idx
    }

    /// Parse a tmux target string: "session", "session:window", "session:window.pane"
    pub fn parse(target_str: &str) -> Result<Self> {
        if target_str.is_empty() {
            return Err(anyhow!("empty target string"));
        }

        let (session_part, rest) = match target_str.rfind(':') {
            Some(pos) => (&target_str[..pos], Some(&target_str[pos + 1..])),
            None => (target_str, None),
        };

        let (window_part, pane_part) = match rest {
            Some(rest) => match rest.find('.') {
                Some(dot) => (Some(&rest[..dot]), Some(&rest[dot + 1..])),
                None => (Some(rest), None),
            },
            None => (None, None),
        };

        let pane = match pane_part {
            Some(p) => Some(p.parse().map_err(|_| anyhow!("invalid pane index: {}", p))?),
            None => None,
        };

        Ok(TargetSpec {
            session_name: session_part.to_string(),
            window_sel: window_part.map(|w| w.to_string()),
            pane_idx: pane,
        })
    }

    pub fn to_target_string(&self) -> String {
        match (&self.window_sel, self.pane_idx) {
            (None, _) => self.session_name.clone(),
            (Some(w), None) => format!("{}:{}", self.session_name, w),
            (Some(w), Some(p)) => format!("{}:{}.{}", self.session_name, w, p),
        }
    }
}

impl fmt::Display for TargetSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_target_string())
    }
}

/// Tmux server socket selection.
#[derive(Debug, Clone)]
pub enum TmuxSocket {
    /// Named socket: `tmux -L <name>`
    Name(String),
    /// Explicit socket path: `tmux -S <path>`
    Path(String),
}

/// SSH host key verification policy (DC2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostKeyPolicy {
    /// Verify against ~/.ssh/known_hosts (default)
    Verify,
    /// Accept and persist on first connect, reject on mismatch
    TrustFirstUse,
    /// Accept all, log warning
    Insecure,
}

impl Default for HostKeyPolicy {
    fn default() -> Self {
        HostKeyPolicy::Verify
    }
}

/// Structured output from `Target::exec()` (DC19).
#[derive(Debug, Clone)]
pub struct ExecOutput {
    pub stdout: String,
    pub exit_code: i32,
}

impl ExecOutput {
    pub fn success(&self) -> bool {
        self.exit_code == 0
    }
}

/// On-demand scrollback sampling query.
pub enum ScrollbackQuery {
    /// Capture the last N lines.
    LastLines(usize),
    /// Scan backwards until pattern matches, up to max_lines.
    Until { pattern: Regex, max_lines: usize },
    /// Capture last N lines, stop early if pattern matches.
    LastLinesUntil { lines: usize, stop_pattern: Regex },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pane_address_roundtrip() {
        let addr = PaneAddress {
            pane_id: "%5".to_string(),
            session: "build".to_string(),
            window: 0,
            pane: 1,
        };
        assert_eq!(addr.to_tmux_target(), "build:0.1");
        assert_eq!(addr.id(), "%5");
        assert_eq!(addr.to_string(), "build:0.1");

        let parsed = PaneAddress::parse("%5", "build:0.1").unwrap();
        assert_eq!(parsed, addr);
    }

    #[test]
    fn pane_address_parse_errors() {
        assert!(PaneAddress::parse("%1", "noseparator").is_err());
        assert!(PaneAddress::parse("%1", "session:nodot").is_err());
        assert!(PaneAddress::parse("%1", "session:abc.1").is_err());
    }

    #[test]
    fn target_spec_session_only() {
        let spec = TargetSpec::session("build");
        assert_eq!(spec.to_target_string(), "build");
        assert_eq!(format!("{}", spec), "build");
    }

    #[test]
    fn target_spec_with_window() {
        let spec = TargetSpec::session("build").window(2);
        assert_eq!(spec.to_target_string(), "build:2");
    }

    #[test]
    fn target_spec_with_window_name() {
        let spec = TargetSpec::session("build").window_name("editor");
        assert_eq!(spec.to_target_string(), "build:editor");
    }

    #[test]
    fn target_spec_full() {
        let spec = TargetSpec::session("build").window(0).pane(3);
        assert_eq!(spec.to_target_string(), "build:0.3");
    }

    #[test]
    fn target_spec_parse_session() {
        let spec = TargetSpec::parse("mysession").unwrap();
        assert_eq!(spec.session_name(), "mysession");
        assert!(spec.window_selector().is_none());
        assert!(spec.pane_index().is_none());
    }

    #[test]
    fn target_spec_parse_session_window() {
        let spec = TargetSpec::parse("mysession:2").unwrap();
        assert_eq!(spec.session_name(), "mysession");
        assert_eq!(spec.window_selector(), Some("2"));
        assert!(spec.pane_index().is_none());
    }

    #[test]
    fn target_spec_parse_full() {
        let spec = TargetSpec::parse("mysession:1.3").unwrap();
        assert_eq!(spec.session_name(), "mysession");
        assert_eq!(spec.window_selector(), Some("1"));
        assert_eq!(spec.pane_index(), Some(3));
    }

    #[test]
    fn target_spec_parse_empty() {
        assert!(TargetSpec::parse("").is_err());
    }

    #[test]
    fn exec_output_success() {
        let out = ExecOutput {
            stdout: "hello".to_string(),
            exit_code: 0,
        };
        assert!(out.success());

        let out_fail = ExecOutput {
            stdout: String::new(),
            exit_code: 1,
        };
        assert!(!out_fail.success());
    }

    #[test]
    fn host_key_policy_default() {
        assert_eq!(HostKeyPolicy::default(), HostKeyPolicy::Verify);
    }

    #[test]
    #[should_panic(expected = "TargetSpec::pane() requires window")]
    fn target_spec_pane_without_window_panics() {
        let _ = TargetSpec::session("s").pane(0);
    }
}
