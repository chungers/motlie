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

    /// Set pane index. Returns `Err` if `window` has not been set — pane
    /// requires a window context (tmux target hierarchy: session:window.pane).
    pub fn pane(mut self, index: u32) -> Result<Self> {
        if self.window_sel.is_none() {
            return Err(anyhow!(
                "TargetSpec::pane() requires window to be set first \
                 (use .window() or .window_name() before .pane())"
            ));
        }
        self.pane_idx = Some(index);
        Ok(self)
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
#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// Accept and persist on first connect, reject on mismatch.
    ///
    /// **Fail-closed behavior**: if persisting the key fails (e.g.,
    /// `~/.ssh/known_hosts` is not writable), the connection is **rejected**
    /// and an error is logged. This prevents silently downgrading to
    /// accept-all semantics. Ensure `~/.ssh/known_hosts` is writable
    /// before using this policy.
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

/// Capture normalization mode (DC20).
///
/// Controls how captured pane content is processed before delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureNormalizeMode {
    /// No transformation. Uses `capture-pane -p` (tmux-rendered text, no ANSI).
    Raw,
    /// Canonical line endings, trim width-artifact trailing spaces.
    /// Uses `capture-pane -ep` to preserve ANSI/control sequences.
    ScreenStable,
    /// Explicit ANSI/control stripping for human/LLM text workflows.
    /// Uses `capture-pane -p`, then normalizes line endings.
    PlainText,
}

impl Default for CaptureNormalizeMode {
    fn default() -> Self {
        CaptureNormalizeMode::Raw
    }
}

/// Options for capture operations with fidelity metadata.
#[derive(Debug, Clone)]
pub struct CaptureOptions {
    /// Negative line offset for scrollback history (e.g. -100).
    /// `None` captures only the visible area.
    pub history_start: Option<i32>,
    /// Normalization mode for the captured content.
    pub normalize: CaptureNormalizeMode,
    /// Number of overlap lines for incremental sampling (Phase 1.9b).
    pub overlap_lines: usize,
    /// Whether to detect geometry reflow around capture (Phase 1.9b).
    pub detect_reflow: bool,
}

impl Default for CaptureOptions {
    fn default() -> Self {
        CaptureOptions {
            history_start: None,
            normalize: CaptureNormalizeMode::Raw,
            overlap_lines: 0,
            detect_reflow: false,
        }
    }
}

impl CaptureOptions {
    /// Create options for a specific mode with no history.
    pub fn with_mode(mode: CaptureNormalizeMode) -> Self {
        CaptureOptions {
            normalize: mode,
            ..Default::default()
        }
    }

    /// Create options with history scrollback.
    pub fn with_history(start: i32) -> Self {
        CaptureOptions {
            history_start: Some(start),
            ..Default::default()
        }
    }
}

/// Fidelity issue detected during capture (DC20).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FidelityIssue {
    /// Client set changed during capture window.
    ClientResize,
    /// Pane geometry changed during capture window.
    PaneResize,
    /// Scrollback history was truncated by tmux history limit.
    HistoryTruncated,
    /// Overlap region was ambiguous and required wider recapture.
    OverlapResync,
}

/// Fidelity metadata for captured content (DC20).
///
/// The hot-path clean case uses `issues: None` for zero heap allocation.
#[derive(Debug, Clone)]
pub struct OutputFidelity {
    /// True when any fidelity degradation was detected.
    pub degraded: bool,
    /// `None` is the hot-path clean case: no heap allocation for issue storage.
    pub issues: Option<Vec<FidelityIssue>>,
}

impl OutputFidelity {
    /// Clean fidelity — no degradation, zero allocation.
    pub fn clean() -> Self {
        OutputFidelity {
            degraded: false,
            issues: None,
        }
    }

    /// Degraded fidelity with one or more issues.
    pub fn degraded(issues: Vec<FidelityIssue>) -> Self {
        OutputFidelity {
            degraded: true,
            issues: Some(issues),
        }
    }
}

impl Default for OutputFidelity {
    fn default() -> Self {
        OutputFidelity::clean()
    }
}

/// Result of a capture operation with normalization and fidelity metadata (DC20).
#[derive(Debug, Clone)]
pub struct CaptureResult {
    /// Mode-specific public payload.
    /// - `Raw`: tmux-rendered text, no normalization
    /// - `ScreenStable`: ANSI-preserving normalized stream
    /// - `PlainText`: ANSI/control-stripped normalized text
    pub text: String,
    /// Exact tmux capture before mode-specific normalization.
    /// Present for `ScreenStable` (the raw `-ep` output); `None` for `Raw`/`PlainText`.
    pub raw_text: Option<String>,
    /// Fidelity metadata for this capture.
    pub fidelity: OutputFidelity,
}

/// Attached tmux client information for geometry detection (DC20, Phase 1.9b).
#[derive(Debug, Clone)]
pub struct ClientInfo {
    pub width: u32,
    pub height: u32,
    pub session: String,
}

/// Pane geometry and scrollback state for reflow detection (DC20, Phase 1.9b).
#[derive(Debug, Clone)]
pub struct PaneGeometry {
    pub pane_width: u32,
    pub pane_height: u32,
    pub history_size: u32,
    pub history_limit: u32,
}

/// Pre/post geometry snapshot for detecting instability during capture (DC20).
#[derive(Debug, Clone)]
pub struct GeometrySnapshot {
    pub clients: Vec<ClientInfo>,
    pub pane: PaneGeometry,
    /// Session name for target-scoped client filtering.
    pub session: String,
}

impl GeometrySnapshot {
    /// Compare two snapshots and return detected fidelity issues.
    ///
    /// Only compares clients attached to the same session as this snapshot's
    /// target, avoiding false-positive `ClientResize` from unrelated sessions.
    pub fn compare(&self, after: &GeometrySnapshot) -> Vec<FidelityIssue> {
        let mut issues = Vec::new();

        // Filter clients to only those attached to the target session
        let before_clients: Vec<(u32, u32)> = self
            .clients
            .iter()
            .filter(|c| c.session == self.session)
            .map(|c| (c.width, c.height))
            .collect();
        let after_clients: Vec<(u32, u32)> = after
            .clients
            .iter()
            .filter(|c| c.session == after.session)
            .map(|c| (c.width, c.height))
            .collect();
        if before_clients != after_clients {
            issues.push(FidelityIssue::ClientResize);
        }

        // Pane geometry changed
        if self.pane.pane_width != after.pane.pane_width
            || self.pane.pane_height != after.pane.pane_height
        {
            issues.push(FidelityIssue::PaneResize);
        }

        // History was truncated (history_size hit limit and wrapped)
        if after.pane.history_size < self.pane.history_size {
            issues.push(FidelityIssue::HistoryTruncated);
        }

        issues
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
        let spec = TargetSpec::session("build").window(0).pane(3).unwrap();
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
    fn target_spec_pane_without_window_errors() {
        let result = TargetSpec::session("s").pane(0);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("requires window"));
    }

    #[test]
    fn capture_normalize_mode_default_is_raw() {
        assert_eq!(CaptureNormalizeMode::default(), CaptureNormalizeMode::Raw);
    }

    #[test]
    fn capture_options_default() {
        let opts = CaptureOptions::default();
        assert!(opts.history_start.is_none());
        assert_eq!(opts.normalize, CaptureNormalizeMode::Raw);
        assert_eq!(opts.overlap_lines, 0);
        assert!(!opts.detect_reflow);
    }

    #[test]
    fn capture_options_with_mode() {
        let opts = CaptureOptions::with_mode(CaptureNormalizeMode::ScreenStable);
        assert_eq!(opts.normalize, CaptureNormalizeMode::ScreenStable);
        assert!(opts.history_start.is_none());
    }

    #[test]
    fn capture_options_with_history() {
        let opts = CaptureOptions::with_history(-200);
        assert_eq!(opts.history_start, Some(-200));
        assert_eq!(opts.normalize, CaptureNormalizeMode::Raw);
    }

    #[test]
    fn output_fidelity_clean_zero_alloc() {
        let f = OutputFidelity::clean();
        assert!(!f.degraded);
        assert!(f.issues.is_none());
    }

    #[test]
    fn output_fidelity_degraded() {
        let f = OutputFidelity::degraded(vec![
            FidelityIssue::ClientResize,
            FidelityIssue::HistoryTruncated,
        ]);
        assert!(f.degraded);
        let issues = f.issues.as_ref().unwrap();
        assert_eq!(issues.len(), 2);
        assert_eq!(issues[0], FidelityIssue::ClientResize);
        assert_eq!(issues[1], FidelityIssue::HistoryTruncated);
    }

    #[test]
    fn output_fidelity_default_is_clean() {
        let f = OutputFidelity::default();
        assert!(!f.degraded);
        assert!(f.issues.is_none());
    }

    // --- Geometry snapshot tests (Phase 1.9b) ---

    fn make_snapshot(
        client_sizes: &[(u32, u32)],
        pane_w: u32,
        pane_h: u32,
        hist_size: u32,
        hist_limit: u32,
    ) -> GeometrySnapshot {
        make_snapshot_for_session(client_sizes, pane_w, pane_h, hist_size, hist_limit, "test")
    }

    fn make_snapshot_for_session(
        client_sizes: &[(u32, u32)],
        pane_w: u32,
        pane_h: u32,
        hist_size: u32,
        hist_limit: u32,
        session: &str,
    ) -> GeometrySnapshot {
        GeometrySnapshot {
            clients: client_sizes
                .iter()
                .map(|&(w, h)| ClientInfo {
                    width: w,
                    height: h,
                    session: session.to_string(),
                })
                .collect(),
            pane: PaneGeometry {
                pane_width: pane_w,
                pane_height: pane_h,
                history_size: hist_size,
                history_limit: hist_limit,
            },
            session: session.to_string(),
        }
    }

    #[test]
    fn geometry_snapshot_no_change() {
        let before = make_snapshot(&[(200, 50)], 80, 24, 100, 2000);
        let after = make_snapshot(&[(200, 50)], 80, 24, 110, 2000);
        let issues = before.compare(&after);
        assert!(issues.is_empty());
    }

    #[test]
    fn geometry_snapshot_client_resize() {
        let before = make_snapshot(&[(200, 50)], 80, 24, 100, 2000);
        let after = make_snapshot(&[(180, 40)], 80, 24, 100, 2000);
        let issues = before.compare(&after);
        assert_eq!(issues, vec![FidelityIssue::ClientResize]);
    }

    #[test]
    fn geometry_snapshot_client_attach_detach() {
        let before = make_snapshot(&[(200, 50)], 80, 24, 100, 2000);
        let after = make_snapshot(&[(200, 50), (180, 40)], 80, 24, 100, 2000);
        let issues = before.compare(&after);
        assert_eq!(issues, vec![FidelityIssue::ClientResize]);
    }

    #[test]
    fn geometry_snapshot_pane_resize() {
        let before = make_snapshot(&[(200, 50)], 80, 24, 100, 2000);
        let after = make_snapshot(&[(200, 50)], 100, 30, 100, 2000);
        let issues = before.compare(&after);
        assert_eq!(issues, vec![FidelityIssue::PaneResize]);
    }

    #[test]
    fn geometry_snapshot_history_truncated() {
        // history_size decreased → history was evicted
        let before = make_snapshot(&[(200, 50)], 80, 24, 2000, 2000);
        let after = make_snapshot(&[(200, 50)], 80, 24, 1500, 2000);
        let issues = before.compare(&after);
        assert_eq!(issues, vec![FidelityIssue::HistoryTruncated]);
    }

    #[test]
    fn geometry_snapshot_multiple_issues() {
        let before = make_snapshot(&[(200, 50)], 80, 24, 2000, 2000);
        let after = make_snapshot(&[(180, 40)], 100, 30, 1500, 2000);
        let issues = before.compare(&after);
        assert_eq!(issues.len(), 3);
        assert!(issues.contains(&FidelityIssue::ClientResize));
        assert!(issues.contains(&FidelityIssue::PaneResize));
        assert!(issues.contains(&FidelityIssue::HistoryTruncated));
    }

    #[test]
    fn geometry_snapshot_no_clients_stable() {
        let before = make_snapshot(&[], 80, 24, 100, 2000);
        let after = make_snapshot(&[], 80, 24, 110, 2000);
        assert!(before.compare(&after).is_empty());
    }

    #[test]
    fn geometry_snapshot_unrelated_session_client_ignored() {
        // Clients attached to a different session should not trigger ClientResize
        let before = GeometrySnapshot {
            clients: vec![
                ClientInfo { width: 200, height: 50, session: "build".to_string() },
                ClientInfo { width: 180, height: 40, session: "other".to_string() },
            ],
            pane: PaneGeometry { pane_width: 80, pane_height: 24, history_size: 100, history_limit: 2000 },
            session: "build".to_string(),
        };
        let after = GeometrySnapshot {
            clients: vec![
                ClientInfo { width: 200, height: 50, session: "build".to_string() },
                // "other" session client resized — should not matter
                ClientInfo { width: 100, height: 20, session: "other".to_string() },
            ],
            pane: PaneGeometry { pane_width: 80, pane_height: 24, history_size: 100, history_limit: 2000 },
            session: "build".to_string(),
        };
        assert!(before.compare(&after).is_empty());
    }

    #[test]
    fn geometry_snapshot_same_session_client_resize_detected() {
        let before = GeometrySnapshot {
            clients: vec![
                ClientInfo { width: 200, height: 50, session: "build".to_string() },
            ],
            pane: PaneGeometry { pane_width: 80, pane_height: 24, history_size: 100, history_limit: 2000 },
            session: "build".to_string(),
        };
        let after = GeometrySnapshot {
            clients: vec![
                ClientInfo { width: 180, height: 40, session: "build".to_string() },
            ],
            pane: PaneGeometry { pane_width: 80, pane_height: 24, history_size: 100, history_limit: 2000 },
            session: "build".to_string(),
        };
        assert_eq!(before.compare(&after), vec![FidelityIssue::ClientResize]);
    }
}
