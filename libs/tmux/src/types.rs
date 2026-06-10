use crate::error::{Error, Result};
use regex::Regex;
use std::fmt;
use std::path::PathBuf;

/// Authoritative pane identifier using tmux's `#{pane_id}` (%<id>).
/// Display fields (session, window, pane) are metadata for human readability.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct PaneAddress {
    /// Stable tmux pane id, e.g. "%12"
    pub pane_id: String,
    /// Stable tmux session id, e.g. "$3", when known.
    #[serde(default)]
    pub session_id: Option<SessionId>,
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
            .ok_or_else(|| Error::Parse("invalid pane address: missing ':'".to_string()))?;
        let session = address_str[..colon_pos].to_string();
        let rest = &address_str[colon_pos + 1..];
        let dot_pos = rest
            .find('.')
            .ok_or_else(|| Error::Parse("invalid pane address: missing '.'".to_string()))?;
        let window: u32 = rest[..dot_pos]
            .parse()
            .map_err(|_| Error::Parse("invalid window index".to_string()))?;
        let pane: u32 = rest[dot_pos + 1..]
            .parse()
            .map_err(|_| Error::Parse("invalid pane index".to_string()))?;

        Ok(PaneAddress {
            pane_id: pane_id.to_string(),
            session_id: None,
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

/// Stable tmux session identifier using tmux's `#{session_id}` (`$<id>`).
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(transparent)]
pub struct SessionId(String);

impl SessionId {
    /// Create a non-empty session id.
    pub fn new(id: impl Into<String>) -> Result<Self> {
        let id = id.into();
        if id.is_empty() {
            return Err(Error::Parse("empty session id".to_string()));
        }
        Ok(Self(id))
    }

    /// Return the tmux id string for dispatch and stable keying.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    #[cfg(test)]
    pub(crate) fn for_test(id: &str) -> Self {
        Self::new(id).expect("test session id must be non-empty")
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl TryFrom<&str> for SessionId {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self> {
        Self::new(value)
    }
}

impl TryFrom<String> for SessionId {
    type Error = Error;

    fn try_from(value: String) -> Result<Self> {
        Self::new(value)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionInfo {
    pub name: String,
    pub id: SessionId,
    pub created: u64,
    pub attached_count: u32,
    pub window_count: u32,
    pub group: Option<String>,
    pub activity: u64,
}

impl SessionInfo {
    /// Return true when one or more tmux clients are attached to this session.
    pub fn is_attached(&self) -> bool {
        self.attached_count > 0
    }
}

/// Maximum supported value size for a session metadata tag.
///
/// tmux user-defined options can hold larger strings, but keeping the public
/// helper capped makes tags safe to poll and avoids accidentally putting large
/// blobs into option-format paths.
pub const SESSION_TAG_VALUE_MAX_BYTES: usize = 2 * 1024;

/// Maximum supported value size for a session environment variable.
///
/// tmux environment values are regular command arguments in this API. Keeping
/// them bounded avoids accidentally piping large blobs through command
/// construction paths.
pub const SESSION_ENV_VAR_VALUE_MAX_BYTES: usize = 8 * 1024;

/// Maximum supported tmux style string size.
///
/// Styles are passed as one tmux option value. Keeping them bounded avoids
/// accidentally piping large content through command construction paths.
pub const TMUX_STYLE_MAX_BYTES: usize = 512;
pub const STATUS_STYLE_MAX_BYTES: usize = TMUX_STYLE_MAX_BYTES;
pub const STATUS_LEFT_MAX_BYTES: usize = 512;
pub const STATUS_LEFT_LENGTH_MAX: u32 = 4096;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TmuxStyle(String);

impl TmuxStyle {
    /// Create a validated tmux style value, for example
    /// `bg=blue,fg=white`.
    ///
    /// Empty styles are rejected; use the relevant unset operation to remove a
    /// local style override.
    ///
    /// This intentionally validates only transport-safety properties and lets
    /// tmux validate style syntax.
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        validate_tmux_style(&value)?;
        Ok(Self(value))
    }

    pub(crate) fn from_tmux_value(value: String) -> Result<Self> {
        validate_tmux_style(&value)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

pub type StatusStyle = TmuxStyle;
pub type WindowStyle = TmuxStyle;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StatusLeft(String);

impl StatusLeft {
    /// Create a validated tmux status-left format string.
    ///
    /// Empty values are accepted because tmux uses an empty `status-left` as a
    /// valid "render no left status text" format. Use the session status API's
    /// unset operation when the inherited/global format should apply instead.
    ///
    /// This intentionally validates only transport-safety properties and lets
    /// tmux validate format syntax.
    pub fn new(value: impl Into<String>) -> Result<Self> {
        let value = value.into();
        validate_status_left(&value)?;
        Ok(Self(value))
    }

    pub(crate) fn from_tmux_value(value: String) -> Result<Self> {
        validate_status_left(&value)?;
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StatusLeftLength(u32);

impl StatusLeftLength {
    /// Create a validated tmux `status-left-length` value.
    ///
    /// tmux accepts a numeric cell budget. This API allows `0` so callers can
    /// intentionally hide the left status area, and caps values at
    /// [`STATUS_LEFT_LENGTH_MAX`] to avoid pathological option values.
    pub fn new(value: u32) -> Result<Self> {
        validate_status_left_length(value)?;
        Ok(Self(value))
    }

    pub fn as_u32(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub(crate) struct SessionTagPrefix(String);

impl SessionTagPrefix {
    pub(crate) fn new(prefix: impl Into<String>) -> Result<Self> {
        let prefix = prefix.into();
        validate_session_tag_component("tag prefix", &prefix)?;
        Ok(Self(prefix))
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }

    pub(crate) fn option_prefix(&self) -> String {
        format!("@{}/", self.0)
    }

    pub(crate) fn option_name(&self, key: &str) -> Result<String> {
        validate_session_tag_component("tag key", key)?;
        Ok(format!("@{}/{key}", self.0))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionTag {
    prefix: String,
    key: String,
    value: String,
}

impl SessionTag {
    /// Create a validated session metadata tag.
    ///
    /// For `prefix = "mmux"` and `key = "role"`, the tmux option name is
    /// `@mmux/role`.
    pub fn new(
        prefix: impl Into<String>,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Result<Self> {
        let prefix = prefix.into();
        let key = key.into();
        let value = value.into();
        validate_session_tag_component("tag prefix", &prefix)?;
        validate_session_tag_component("tag key", &key)?;
        validate_session_tag_value(&value)?;
        Ok(Self { prefix, key, value })
    }

    pub(crate) fn from_parts(prefix: &SessionTagPrefix, key: &str, value: String) -> Result<Self> {
        validate_session_tag_component("tag key", key)?;
        validate_session_tag_value(&value)?;
        Ok(Self {
            prefix: prefix.as_str().to_string(),
            key: key.to_string(),
            value,
        })
    }

    /// Namespace prefix, for example `mmux` in `@mmux/role`.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Tag key without the namespace prefix, for example `role` in `@mmux/role`.
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Raw user-defined option value.
    pub fn value(&self) -> &str {
        &self.value
    }

    /// Full tmux user-defined option name, for example `@mmux/role`.
    pub fn option_name(&self) -> String {
        format!("@{}/{}", self.prefix, self.key)
    }

    pub(crate) fn into_value(self) -> String {
        self.value
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SessionEnvVar {
    name: String,
    value: String,
}

impl SessionEnvVar {
    /// Create a validated session environment variable.
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Result<Self> {
        let name = name.into();
        let value = value.into();
        validate_session_env_var_name(&name)?;
        validate_session_env_var_value(&value)?;
        Ok(Self { name, value })
    }

    pub(crate) fn from_parts(name: &str, value: String) -> Result<Self> {
        validate_session_env_var_name(name)?;
        validate_session_env_var_value(&value)?;
        Ok(Self {
            name: name.to_string(),
            value,
        })
    }

    /// Environment variable name, for example `PATH`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Raw environment variable value.
    pub fn value(&self) -> &str {
        &self.value
    }

    pub(crate) fn into_value(self) -> String {
        self.value
    }
}

pub(crate) fn validate_session_tag_component(kind: &str, value: &str) -> Result<()> {
    if value.is_empty() {
        return Err(Error::Parse(format!("{kind} cannot be empty")));
    }
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        return Err(Error::Parse(format!(
            "{kind} must contain only ASCII letters, digits, '.', '_' or '-': {value:?}"
        )));
    }
    Ok(())
}

pub(crate) fn validate_session_tag_value(value: &str) -> Result<()> {
    if value.len() > SESSION_TAG_VALUE_MAX_BYTES {
        return Err(Error::Parse(format!(
            "tag value is {} bytes, exceeding {} byte limit",
            value.len(),
            SESSION_TAG_VALUE_MAX_BYTES
        )));
    }
    if value.chars().any(char::is_control) {
        return Err(Error::Parse(
            "tag value cannot contain control characters".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_session_env_var_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(Error::Parse(
            "environment variable name cannot be empty".to_string(),
        ));
    }
    let mut bytes = name.bytes();
    let first = bytes.next().expect("name is not empty");
    if !(first.is_ascii_alphabetic() || first == b'_') {
        return Err(Error::Parse(format!(
            "environment variable name must start with an ASCII letter or '_': {name:?}"
        )));
    }
    if !bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_') {
        return Err(Error::Parse(format!(
            "environment variable name must contain only ASCII letters, digits or '_': {name:?}"
        )));
    }
    Ok(())
}

pub(crate) fn validate_session_env_var_value(value: &str) -> Result<()> {
    if value.len() > SESSION_ENV_VAR_VALUE_MAX_BYTES {
        return Err(Error::Parse(format!(
            "environment variable value is {} bytes, exceeding {} byte limit",
            value.len(),
            SESSION_ENV_VAR_VALUE_MAX_BYTES
        )));
    }
    if value.chars().any(char::is_control) {
        return Err(Error::Parse(
            "environment variable value cannot contain control characters".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_tmux_style(value: &str) -> Result<()> {
    if value.is_empty() {
        return Err(Error::Parse("tmux style cannot be empty".to_string()));
    }
    if value.len() > TMUX_STYLE_MAX_BYTES {
        return Err(Error::Parse(format!(
            "tmux style is {} bytes, exceeding {} byte limit",
            value.len(),
            TMUX_STYLE_MAX_BYTES
        )));
    }
    if value.chars().any(char::is_control) {
        return Err(Error::Parse(
            "tmux style cannot contain control characters".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_status_left(value: &str) -> Result<()> {
    if value.len() > STATUS_LEFT_MAX_BYTES {
        return Err(Error::Parse(format!(
            "status-left is {} bytes, exceeding {} byte limit",
            value.len(),
            STATUS_LEFT_MAX_BYTES
        )));
    }
    if value.chars().any(char::is_control) {
        return Err(Error::Parse(
            "status-left cannot contain control characters".to_string(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_status_left_length(value: u32) -> Result<()> {
    if value > STATUS_LEFT_LENGTH_MAX {
        return Err(Error::Parse(format!(
            "status-left-length {value} exceeds maximum {STATUS_LEFT_LENGTH_MAX}"
        )));
    }
    Ok(())
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

impl fmt::Display for TargetLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TargetLevel::Session => f.write_str("session"),
            TargetLevel::Window => f.write_str("window"),
            TargetLevel::Pane => f.write_str("pane"),
        }
    }
}

/// Builder for tmux target strings (DC17).
///
/// Fields are private to enforce the hierarchy invariant: pane requires window.
/// Use `session()`, `.window()/.window_name()`, `.pane()` builders or `parse()`.
#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
enum SessionSelector {
    Name(String),
    Id(SessionId),
}

impl SessionSelector {
    fn as_target_str(&self) -> &str {
        match self {
            SessionSelector::Name(name) => name,
            SessionSelector::Id(id) => id.as_str(),
        }
    }

    fn id(&self) -> Option<&SessionId> {
        match self {
            SessionSelector::Name(_) => None,
            SessionSelector::Id(id) => Some(id),
        }
    }
}

#[derive(
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct TargetSpec {
    session: SessionSelector,
    window_sel: Option<String>,
    pane_idx: Option<u32>,
}

impl TargetSpec {
    pub fn session(name: &str) -> Self {
        TargetSpec {
            session: SessionSelector::Name(name.to_string()),
            window_sel: None,
            pane_idx: None,
        }
    }

    pub fn session_id(id: impl Into<String>) -> Result<Self> {
        Ok(TargetSpec {
            session: SessionSelector::Id(SessionId::new(id)?),
            window_sel: None,
            pane_idx: None,
        })
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
            return Err(Error::Parse(
                "TargetSpec::pane() requires window to be set first \
                 (use .window() or .window_name() before .pane())"
                    .to_string(),
            ));
        }
        self.pane_idx = Some(index);
        Ok(self)
    }

    // --- Accessors ---

    pub fn session_name(&self) -> &str {
        self.session.as_target_str()
    }

    pub fn session_id_selector(&self) -> Option<&SessionId> {
        self.session.id()
    }

    pub fn window_selector(&self) -> Option<&str> {
        self.window_sel.as_deref()
    }

    pub fn pane_index(&self) -> Option<u32> {
        self.pane_idx
    }

    /// Parse a tmux target string: "session", "session:window", "session:window.pane".
    ///
    /// A `$<digits>` session component is treated as a stable tmux session id.
    /// Use [`TargetSpec::session`] when a literal session name looks like `$7`.
    pub fn parse(target_str: &str) -> Result<Self> {
        if target_str.is_empty() {
            return Err(Error::Parse("empty target string".to_string()));
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
            Some(p) => Some(
                p.parse()
                    .map_err(|_| Error::Parse(format!("invalid pane index: {}", p)))?,
            ),
            None => None,
        };

        let session = if looks_like_session_id(session_part) {
            SessionSelector::Id(SessionId::new(session_part.to_string())?)
        } else {
            SessionSelector::Name(session_part.to_string())
        };

        Ok(TargetSpec {
            session,
            window_sel: window_part.map(|w| w.to_string()),
            pane_idx: pane,
        })
    }

    pub fn to_target_string(&self) -> String {
        match (&self.window_sel, self.pane_idx) {
            (None, _) => self.session.as_target_str().to_string(),
            (Some(w), None) => format!("{}:{}", self.session.as_target_str(), w),
            (Some(w), Some(p)) => format!("{}:{}.{}", self.session.as_target_str(), w, p),
        }
    }
}

fn looks_like_session_id(value: &str) -> bool {
    let Some(rest) = value.strip_prefix('$') else {
        return false;
    };
    !rest.is_empty() && rest.bytes().all(|byte| byte.is_ascii_digit())
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

impl TmuxSocket {
    /// Create a dedicated automation socket with a `motlie-` prefix (DC30).
    ///
    /// Produces `TmuxSocket::Name(format!("motlie-{}", scope))`. The scope
    /// is validated: non-empty, max 64 chars, `[A-Za-z0-9._-]` charset.
    /// This isolates automation workloads from the user's default tmux server.
    pub fn automation(scope: &str) -> Result<Self> {
        if scope.is_empty() {
            return Err(Error::Parse(
                "automation scope must not be empty".to_string(),
            ));
        }
        if scope.len() > 64 {
            return Err(Error::Parse(format!(
                "automation scope too long ({} chars, max 64): {}",
                scope.len(),
                scope
            )));
        }
        if !scope
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-')
        {
            return Err(Error::Parse(format!(
                "automation scope contains invalid characters: '{}' (allowed: [A-Za-z0-9._-])",
                scope
            )));
        }
        Ok(TmuxSocket::Name(format!("motlie-{}", scope)))
    }
}

/// SSH host key verification policy (DC2).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum HostKeyPolicy {
    /// Verify against ~/.ssh/known_hosts (default)
    #[default]
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

/// Typed command identity for tracked execution (DC31).
///
/// Wraps `uuid::Uuid` to provide a unique per-execution identity. The
/// `short_hex()` method produces an 8-char hex string matching the existing
/// exec sentinel format (`__ML<hex>__`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecId(uuid::Uuid);

impl ExecId {
    /// Create a new random execution ID.
    pub fn new() -> Self {
        ExecId(uuid::Uuid::new_v4())
    }

    /// 8-character hex marker for sentinel matching.
    pub fn short_hex(&self) -> String {
        self.0.to_string()[..8].to_string()
    }
}

impl fmt::Display for ExecId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for ExecId {
    fn default() -> Self {
        Self::new()
    }
}

/// Execution state for tracked commands (DC31).
#[derive(Debug, Clone)]
pub enum ExecState {
    /// Command is still running (sentinel not yet observed).
    Running,
    /// Sentinel observed — command completed with structured output.
    Completed(ExecOutput),
    /// Cannot determine completion status (e.g., connection lost, timeout).
    Unknown { reason: String },
}

impl ExecState {
    /// Whether this state is terminal (will not change again).
    pub fn is_terminal(&self) -> bool {
        !matches!(self, ExecState::Running)
    }
}

/// Capture normalization mode (DC20).
///
/// Controls how captured pane content is processed before delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CaptureNormalizeMode {
    /// No transformation. Uses `capture-pane -p` (tmux-rendered text, no ANSI).
    #[default]
    Raw,
    /// Canonical line endings, trim width-artifact trailing spaces.
    /// Uses `capture-pane -ep` to preserve ANSI/control sequences.
    ScreenStable,
    /// Explicit ANSI/control stripping for human/LLM text workflows.
    /// Uses `capture-pane -p`, then normalizes line endings.
    PlainText,
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

/// Options for session creation (DC22).
///
/// All fields are optional. `Default` produces the same behavior as the
/// pre-DC22 `create_session(name, None, None)` — no size override, no
/// history limit, no initial environment overrides, tmux server defaults apply.
///
/// `initial_environment` is the lifecycle hook for variables that must be
/// visible to the first shell or command in the session. Post-creation
/// [`SessionEnvironment`](crate::SessionEnvironment) writes update tmux's
/// session environment for processes tmux starts later; they cannot mutate
/// already-running pane processes.
///
/// If `history_limit` is set, two `set-option` commands are issued after
/// `new-session`: per-session (covers future panes) and per-pane (tmux 3.1+,
/// covers the initial pane created by `new-session`).
#[derive(Debug, Clone, Default)]
pub struct CreateSessionOptions {
    /// Initial window name (`-n` flag).
    pub window_name: Option<String>,
    /// Command to run in the initial pane.
    pub command: Option<String>,
    /// Initial window width (`-x` flag).
    pub width: Option<u16>,
    /// Initial window height (`-y` flag).
    pub height: Option<u16>,
    /// Scrollback history limit for the session.
    pub history_limit: Option<u32>,
    /// Environment variables passed to `tmux new-session -e`.
    ///
    /// These are applied before tmux starts the initial pane process, so they
    /// are visible to the session's first shell or command. Values are emitted
    /// in vector order; if the same variable name appears more than once, tmux
    /// applies the last value.
    pub initial_environment: Vec<SessionEnvVar>,
}

/// Options for creating a new tmux window as a child of a session (DC25).
#[derive(Debug, Clone, Default)]
pub struct CreateWindowOptions {
    /// Window name (`-n` flag).
    pub name: Option<String>,
    /// Command to run in the initial pane.
    pub command: Option<String>,
    /// Initial window width (`-x` flag).
    pub width: Option<u16>,
    /// Initial window height (`-y` flag).
    pub height: Option<u16>,
    /// Start directory for the new window (`-c` flag).
    pub start_directory: Option<PathBuf>,
}

/// Direction for `split-window` (DC25).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SplitDirection {
    /// Create panes side-by-side (`split-window -h`).
    Horizontal,
    /// Create panes stacked top/bottom (`split-window -v`).
    #[default]
    Vertical,
}

/// Size override for `split-window` (DC25).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitSize {
    /// Fixed size in cells (`-l` flag).
    Cells(u16),
    /// Percentage of the target pane (`-l <n>%` flag form).
    Percent(u8),
}

impl SplitSize {
    /// Create a checked percentage value for `split-window -l <n>%`.
    pub fn percent(value: u8) -> Result<Self> {
        if value == 0 || value > 100 {
            return Err(Error::Parse(format!(
                "split percentage must be in 1..=100, got {}",
                value
            )));
        }
        Ok(SplitSize::Percent(value))
    }
}

/// Options for splitting a tmux pane from a window/pane target (DC25).
#[derive(Debug, Clone, Default)]
pub struct SplitPaneOptions {
    /// Horizontal (`-h`) or vertical (`-v`) split. Default: vertical.
    pub direction: SplitDirection,
    /// Optional size override (`-l <n>` or `-l <n>%`).
    pub size: Option<SplitSize>,
    /// Command to run in the new pane.
    pub command: Option<String>,
    /// Start directory for the new pane (`-c` flag).
    pub start_directory: Option<PathBuf>,
}

/// Options for host-level file transfer (DC23).
///
/// Controls overwrite and recursive behavior for `HostHandle::upload()`
/// and `HostHandle::download()`. Default: `overwrite=true`, `recursive=false`.
///
/// **Directory overwrite semantics**: when `overwrite=true` and the destination
/// is an existing directory, the transfer uses **merge** semantics — conflicting
/// files are overwritten from the source, missing entries are created, and
/// destination-only extras are preserved. This matches `cp -r` behavior, not
/// `rsync --delete`.
///
/// **Directory placement**: follows `cp -r` semantics — if the destination
/// exists as a directory, the source is copied **into** it using the source
/// basename; if the destination does not exist, the source is copied **as**
/// that path.
#[derive(Debug, Clone)]
pub struct TransferOptions {
    /// If `true`, overwrite existing destination files/directories (merge for dirs).
    /// If `false`, return `Err` when the destination already exists.
    pub overwrite: bool,
    /// If `true`, recursively copy directory trees.
    /// If `false`, return `Err` when the source is a directory.
    pub recursive: bool,
}

impl Default for TransferOptions {
    fn default() -> Self {
        Self {
            overwrite: true,
            recursive: false,
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
    pub session_id: Option<String>,
    pub activity: u64,
    pub readonly: bool,
    pub tty: Option<String>,
}

/// Attached-client activity summary for one tmux session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionClientActivity {
    pub session: String,
    pub attached_clients: usize,
    pub writable_clients: usize,
    pub latest_client_activity: Option<u64>,
    pub latest_writable_client_activity: Option<u64>,
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
    /// Capture a bounded window of scrollback older than the most recent N lines.
    LinesRange {
        older_than_lines: usize,
        count: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pane_address_roundtrip() {
        let addr = PaneAddress {
            pane_id: "%5".to_string(),
            session_id: None,
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
    fn status_style_validates_transport_safe_values() {
        let style = StatusStyle::new("bg=blue,fg=white").unwrap();
        assert_eq!(style.as_str(), "bg=blue,fg=white");

        assert!(StatusStyle::new("").is_err());
        assert!(StatusStyle::new("bg=blue\nfg=white").is_err());
        assert!(StatusStyle::new("x".repeat(STATUS_STYLE_MAX_BYTES + 1)).is_err());
    }

    #[test]
    fn status_left_validates_transport_safe_values() {
        let left = StatusLeft::new("#{=40:session_name}").unwrap();
        assert_eq!(left.as_str(), "#{=40:session_name}");
        assert_eq!(StatusLeftLength::new(40).unwrap().as_u32(), 40);

        assert!(StatusLeft::new("").is_ok());
        assert!(StatusLeft::new("name\nbad").is_err());
        assert!(StatusLeft::new("x".repeat(STATUS_LEFT_MAX_BYTES + 1)).is_err());
        assert!(StatusLeftLength::new(STATUS_LEFT_LENGTH_MAX).is_ok());
        assert!(StatusLeftLength::new(STATUS_LEFT_LENGTH_MAX + 1).is_err());
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
    fn target_spec_parse_session_id() {
        let spec = TargetSpec::parse("$7").unwrap();
        assert_eq!(spec.session_name(), "$7");
        assert_eq!(spec.session_id_selector().unwrap().as_str(), "$7");
        assert_eq!(spec.to_target_string(), "$7");
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
    fn split_direction_default_is_vertical() {
        assert_eq!(SplitDirection::default(), SplitDirection::Vertical);
    }

    #[test]
    fn split_size_percent_accepts_valid_range() {
        assert_eq!(SplitSize::percent(1).unwrap(), SplitSize::Percent(1));
        assert_eq!(SplitSize::percent(100).unwrap(), SplitSize::Percent(100));
    }

    #[test]
    fn split_size_percent_rejects_zero() {
        assert!(SplitSize::percent(0).is_err());
    }

    #[test]
    fn split_pane_options_default_is_vertical_without_size() {
        let opts = SplitPaneOptions::default();
        assert_eq!(opts.direction, SplitDirection::Vertical);
        assert!(opts.size.is_none());
        assert!(opts.command.is_none());
        assert!(opts.start_directory.is_none());
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
                    session_id: None,
                    activity: 0,
                    readonly: false,
                    tty: None,
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
                ClientInfo {
                    width: 200,
                    height: 50,
                    session: "build".to_string(),
                    session_id: Some("$1".to_string()),
                    activity: 0,
                    readonly: false,
                    tty: None,
                },
                ClientInfo {
                    width: 180,
                    height: 40,
                    session: "other".to_string(),
                    session_id: Some("$2".to_string()),
                    activity: 0,
                    readonly: false,
                    tty: None,
                },
            ],
            pane: PaneGeometry {
                pane_width: 80,
                pane_height: 24,
                history_size: 100,
                history_limit: 2000,
            },
            session: "build".to_string(),
        };
        let after = GeometrySnapshot {
            clients: vec![
                ClientInfo {
                    width: 200,
                    height: 50,
                    session: "build".to_string(),
                    session_id: Some("$1".to_string()),
                    activity: 0,
                    readonly: false,
                    tty: None,
                },
                // "other" session client resized — should not matter
                ClientInfo {
                    width: 100,
                    height: 20,
                    session: "other".to_string(),
                    session_id: Some("$2".to_string()),
                    activity: 0,
                    readonly: false,
                    tty: None,
                },
            ],
            pane: PaneGeometry {
                pane_width: 80,
                pane_height: 24,
                history_size: 100,
                history_limit: 2000,
            },
            session: "build".to_string(),
        };
        assert!(before.compare(&after).is_empty());
    }

    #[test]
    fn geometry_snapshot_same_session_client_resize_detected() {
        let before = GeometrySnapshot {
            clients: vec![ClientInfo {
                width: 200,
                height: 50,
                session: "build".to_string(),
                session_id: Some("$1".to_string()),
                activity: 0,
                readonly: false,
                tty: None,
            }],
            pane: PaneGeometry {
                pane_width: 80,
                pane_height: 24,
                history_size: 100,
                history_limit: 2000,
            },
            session: "build".to_string(),
        };
        let after = GeometrySnapshot {
            clients: vec![ClientInfo {
                width: 180,
                height: 40,
                session: "build".to_string(),
                session_id: Some("$1".to_string()),
                activity: 0,
                readonly: false,
                tty: None,
            }],
            pane: PaneGeometry {
                pane_width: 80,
                pane_height: 24,
                history_size: 100,
                history_limit: 2000,
            },
            session: "build".to_string(),
        };
        assert_eq!(before.compare(&after), vec![FidelityIssue::ClientResize]);
    }

    #[test]
    fn transfer_options_default() {
        let opts = TransferOptions::default();
        assert!(opts.overwrite);
        assert!(!opts.recursive);
    }

    // --- TmuxSocket::automation tests (DC30) ---

    #[test]
    fn automation_socket_valid_scope() {
        let socket = TmuxSocket::automation("ci-build").unwrap();
        assert_eq!(socket, TmuxSocket::Name("motlie-ci-build".to_string()));
    }

    #[test]
    fn automation_socket_prefix_check() {
        let socket = TmuxSocket::automation("agent_1").unwrap();
        match socket {
            TmuxSocket::Name(name) => assert!(name.starts_with("motlie-")),
            _ => panic!("expected Name variant"),
        }
    }

    #[test]
    fn automation_socket_empty_scope_errors() {
        assert!(TmuxSocket::automation("").is_err());
    }

    #[test]
    fn automation_socket_invalid_chars_errors() {
        assert!(TmuxSocket::automation("bad/scope").is_err());
        assert!(TmuxSocket::automation("bad scope").is_err());
        assert!(TmuxSocket::automation("bad@scope").is_err());
    }

    #[test]
    fn automation_socket_too_long_errors() {
        let long_scope = "a".repeat(65);
        assert!(TmuxSocket::automation(&long_scope).is_err());
    }

    #[test]
    fn automation_socket_max_length_ok() {
        let scope = "a".repeat(64);
        assert!(TmuxSocket::automation(&scope).is_ok());
    }

    #[test]
    fn automation_socket_all_valid_chars() {
        assert!(TmuxSocket::automation("ABCxyz.012_3-4").is_ok());
    }

    // --- ExecId tests (DC31) ---

    #[test]
    fn exec_id_uniqueness() {
        let a = ExecId::new();
        let b = ExecId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn exec_id_short_hex_length() {
        let id = ExecId::new();
        assert_eq!(id.short_hex().len(), 8);
    }

    #[test]
    fn exec_id_display() {
        let id = ExecId::new();
        let display = format!("{}", id);
        // UUID v4 format: 8-4-4-4-12
        assert_eq!(display.len(), 36);
    }

    // --- ExecState tests (DC31) ---

    #[test]
    fn exec_state_running_not_terminal() {
        assert!(!ExecState::Running.is_terminal());
    }

    #[test]
    fn exec_state_completed_is_terminal() {
        let state = ExecState::Completed(ExecOutput {
            stdout: "ok".to_string(),
            exit_code: 0,
        });
        assert!(state.is_terminal());
    }

    #[test]
    fn exec_state_unknown_is_terminal() {
        let state = ExecState::Unknown {
            reason: "timed out".to_string(),
        };
        assert!(state.is_terminal());
    }
}
