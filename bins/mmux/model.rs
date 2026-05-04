use std::cmp::{min, Ordering};
use std::collections::HashMap;

use motlie_tmux::{HostHandle, SessionInfo};
use ratatui::style::{Color, Style};

use crate::consts::{
    DEFAULT_LEFT_PERCENT, DEFAULT_TOP_PERCENT, HOST_COLOR_PALETTE, HOST_COLOR_SQUARE,
    LANDSCAPE_MAX_LEFT_PERCENT, LANDSCAPE_MIN_LEFT_PERCENT, PORTRAIT_MAX_TOP_PERCENT,
    PORTRAIT_MIN_TOP_PERCENT, STATUS_BAR_BG, STATUS_BAR_FG,
};
use crate::detail::DetailSource;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayoutMode {
    Normal,
    Portrait,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Focus {
    List,
    Detail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Button {
    Cancel,
    Ok,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SessionKeyValueFocus {
    Row(usize),
    Key,
    Value,
    Ok,
    Cancel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SessionKeyValueKind {
    Tags,
    Environment,
}

impl SessionKeyValueKind {
    pub(crate) fn title(self) -> &'static str {
        match self {
            Self::Tags => " Session Tags ",
            Self::Environment => " Initial Environment ",
        }
    }

    pub(crate) fn empty_label(self) -> &'static str {
        match self {
            Self::Tags => "No tags",
            Self::Environment => "No environment",
        }
    }

    pub(crate) fn noun(self) -> &'static str {
        match self {
            Self::Tags => "tag",
            Self::Environment => "env var",
        }
    }

    pub(crate) fn supports_checked_row(self) -> bool {
        matches!(self, Self::Tags)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NewSessionFocus {
    Host,
    Name,
    EnvRow(usize),
    EnvKey,
    EnvValue,
    Cancel,
    Ok,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SendKeysFocus {
    Input,
    Ok,
    Cancel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NewSessionHostChoice {
    pub(crate) id: HostId,
    pub(crate) label: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SessionKeyValueRow {
    pub(crate) key: String,
    pub(crate) value: String,
}

#[derive(Debug, Clone)]
pub(crate) enum ModalState {
    NewSession {
        ui: NewSessionModalUi,
    },
    KillSession {
        session: SelectedSession,
        button: Button,
    },
    RenameSession {
        session: SelectedSession,
        input: String,
        button: Button,
    },
    SendKeys {
        session: SelectedSession,
        ui: SendKeysModalUi,
    },
    SessionKeyValues {
        session: SelectedSession,
        ui: SessionKeyValueModalUi,
    },
    Help,
}

#[derive(Debug, Clone)]
pub(crate) struct SendKeysModalUi {
    pub(crate) input: String,
    pub(crate) focus: SendKeysFocus,
}

#[derive(Debug, Clone)]
pub(crate) struct NewSessionModalUi {
    pub(crate) input: String,
    pub(crate) hosts: Vec<NewSessionHostChoice>,
    pub(crate) host_index: usize,
    pub(crate) env_rows: Vec<SessionKeyValueRow>,
    pub(crate) env_key_input: String,
    pub(crate) env_value_input: String,
    pub(crate) focus: NewSessionFocus,
    pub(crate) button: Button,
}

impl NewSessionModalUi {
    pub(crate) fn selected_host(&self) -> Option<&NewSessionHostChoice> {
        self.hosts.get(self.host_index)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SessionKeyValueModalUi {
    pub(crate) kind: SessionKeyValueKind,
    pub(crate) rows: Vec<SessionKeyValueRow>,
    pub(crate) selected_key: Option<String>,
    pub(crate) original_rows: Vec<SessionKeyValueRow>,
    pub(crate) original_selected_key: Option<String>,
    pub(crate) key_input: String,
    pub(crate) value_input: String,
    pub(crate) focus: SessionKeyValueFocus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModalView {
    pub(crate) title: &'static str,
    pub(crate) body: ModalBody,
    pub(crate) buttons: String,
    pub(crate) active_button: Option<Button>,
}

impl ModalView {
    #[cfg(test)]
    pub(crate) fn body_text(&self) -> String {
        match &self.body {
            ModalBody::Text(text) => text.clone(),
            ModalBody::NewSession {
                input,
                host_label,
                env_rows,
                env_key_input,
                env_value_input,
                ..
            } => {
                let fields = match host_label {
                    Some(host_label) => format!("Host\n{host_label}\nSession name\n{input}"),
                    None => format!("Session name\n{input}"),
                };
                let env = if env_rows.is_empty() {
                    "No environment".to_string()
                } else {
                    env_rows
                        .iter()
                        .map(|row| format!("{}    {}", row.key, row.value))
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                format!("{fields}\n{env}\n{env_key_input}    {env_value_input}")
            }
            ModalBody::RenameSession { input } => format!("Session Name\n{input}"),
            ModalBody::SendKeys { label, input, .. } => format!("{label}\n{input}"),
            ModalBody::SessionKeyValues {
                kind,
                rows,
                selected_key,
                key_input,
                value_input,
                ..
            } => {
                let rows = if rows.is_empty() {
                    kind.empty_label().to_string()
                } else {
                    rows.iter()
                        .map(|row| {
                            let marker = if kind.supports_checked_row()
                                && selected_key.as_deref() == Some(row.key.as_str())
                            {
                                "✓"
                            } else {
                                " "
                            };
                            format!("{}    {} {marker}", row.key, row.value)
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                format!("{rows}\n{key_input}    {value_input}")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ModalBody {
    Text(String),
    NewSession {
        input: String,
        host_label: Option<String>,
        host_count: usize,
        env_rows: Vec<SessionKeyValueRow>,
        env_key_input: String,
        env_value_input: String,
        focus: NewSessionFocus,
    },
    RenameSession {
        input: String,
    },
    SendKeys {
        label: String,
        input: String,
        focused: bool,
    },
    SessionKeyValues {
        kind: SessionKeyValueKind,
        rows: Vec<SessionKeyValueRow>,
        selected_key: Option<String>,
        key_input: String,
        value_input: String,
        focus: SessionKeyValueFocus,
    },
}

/// Stable identity for a target host across the binary. Either a normalized
/// SSH URI (for SSH targets) or the literal `"localhost"` (for the local host).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct HostId(String);

impl HostId {
    pub(crate) fn local() -> Self {
        Self("localhost".to_string())
    }

    pub(crate) fn from_ssh_uri(uri: &str) -> Self {
        Self(uri.to_string())
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for HostId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// One configured host: stable id, display label, URI alias, IP, and the
/// connected handle.
///
/// Recency math (activity, age) is computed in the binary against the
/// operator's wall clock. We assume host clocks are NTP-synced with the
/// operator (typical for any production deployment); wildly skewed clocks
/// will produce mildly inaccurate "age" text but the selector continues to
/// function. There is no portable, side-effect-free way to ask older
/// tmux for its server clock, so the lib does not expose one.
#[derive(Clone)]
pub(crate) struct HostEntry {
    pub(crate) id: HostId,
    /// Host-local name probed after connection; used for display.
    pub(crate) label: String,
    /// Hostname parsed from the SSH URI; used as the stable operator alias.
    pub(crate) alias: String,
    pub(crate) ip_address: String,
    pub(crate) handle: HostHandle,
}

impl HostEntry {
    pub(crate) fn diagnostic_label(&self) -> String {
        if self.alias == self.label {
            self.label.clone()
        } else {
            format!("{} ({})", self.label, self.alias)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum HostConnectionStatus {
    Connecting,
    Connected,
    Failed(HostConnectFailure),
}

impl HostConnectionStatus {
    pub(crate) fn is_failed(&self) -> bool {
        matches!(self, Self::Failed(_))
    }
}

/// Stable, user-visible equivalence class for SSH connection failures.
///
/// `PartialEq` intentionally compares both the typed phase and the display
/// message. That keeps repeated identical retry failures from forcing redraws,
/// while still updating the status if a host moves from one concrete failure
/// mode to another.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HostConnectFailure {
    phase: HostConnectFailurePhase,
    message: String,
}

impl HostConnectFailure {
    pub(crate) fn connect(message: String) -> Self {
        Self {
            phase: HostConnectFailurePhase::Connect,
            message,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum HostConnectFailurePhase {
    Connect,
}

#[derive(Clone)]
pub(crate) struct HostSlot {
    pub(crate) id: HostId,
    pub(crate) label: String,
    pub(crate) alias: String,
    pub(crate) ip_address: String,
    pub(crate) status: HostConnectionStatus,
}

impl HostSlot {
    pub(crate) fn connecting(id: HostId, label: String, alias: String) -> Self {
        Self {
            id,
            label,
            alias,
            ip_address: "unknown".to_string(),
            status: HostConnectionStatus::Connecting,
        }
    }

    pub(crate) fn connected(entry: &HostEntry) -> Self {
        Self {
            id: entry.id.clone(),
            label: entry.label.clone(),
            alias: entry.alias.clone(),
            ip_address: entry.ip_address.clone(),
            status: HostConnectionStatus::Connected,
        }
    }

    fn update_connected(&mut self, entry: &HostEntry) -> bool {
        let changed = self.label != entry.label
            || self.alias != entry.alias
            || self.ip_address != entry.ip_address
            || self.status != HostConnectionStatus::Connected;
        self.label = entry.label.clone();
        self.alias = entry.alias.clone();
        self.ip_address = entry.ip_address.clone();
        self.status = HostConnectionStatus::Connected;
        changed
    }
}

#[derive(Clone)]
pub(crate) struct HostLegendItem {
    pub(crate) color: Color,
    pub(crate) label: String,
    pub(crate) failed: bool,
}

/// Collection of one or more configured hosts. Always has at least one
/// configured host (localhost when no SSH URIs are specified). Multi-host mode
/// is implicit: `is_multi() == true` iff more than one host is configured.
///
/// This is **not** `motlie_tmux::Fleet` (`libs/tmux/src/fleet.rs`). That type
/// is a monitoring/automation registry — it owns a shared `OutputBus`,
/// manages per-host `MonitorHandle` lifecycle, and stores workstream alias→
/// `TargetSpec` bindings, which are orthogonal to the selector's needs.
/// `HostFleet` is the selector's *display-and-routing* fleet: a flat list of
/// hosts with per-row metadata (label, ip) and 1 Hz fan-out polling via
/// `HostHandle::list_sessions()`. See `docs/DESIGN.md` §Multi-host Mode →
/// §Internal data model for the full rationale.
///
/// `hosts` is the authoritative configured-host order used for presentation
/// and stable color assignment. `entries` is the connected-host routing cache
/// with owned `HostHandle`s, keyed by the same `HostId`s. Keep those
/// collections in sync only through `upsert_connected` and `mark_host_failed`.
#[derive(Clone, Default)]
pub(crate) struct HostFleet {
    hosts: Vec<HostSlot>,
    pub(crate) entries: Vec<HostEntry>,
}

impl HostFleet {
    pub(crate) fn from_entries(entries: Vec<HostEntry>) -> Self {
        let hosts = entries.iter().map(HostSlot::connected).collect();
        Self { hosts, entries }
    }

    pub(crate) fn from_configured_hosts(entries: Vec<HostEntry>, hosts: Vec<HostSlot>) -> Self {
        let mut fleet = Self {
            hosts,
            entries: Vec::new(),
        };
        for entry in entries {
            fleet.upsert_connected(entry);
        }
        fleet
    }

    pub(crate) fn is_multi(&self) -> bool {
        self.hosts.len() > 1
    }

    pub(crate) fn len(&self) -> usize {
        self.hosts.len()
    }

    pub(crate) fn entry(&self, id: &HostId) -> Option<&HostEntry> {
        self.entries.iter().find(|entry| &entry.id == id)
    }

    pub(crate) fn first(&self) -> Option<&HostEntry> {
        self.entries.first()
    }

    #[cfg(test)]
    pub(crate) fn host_slot(&self, id: &HostId) -> Option<&HostSlot> {
        self.hosts.iter().find(|host| &host.id == id)
    }

    pub(crate) fn upsert_connected(&mut self, entry: HostEntry) -> bool {
        let mut changed = false;
        match self
            .entries
            .iter_mut()
            .find(|existing| existing.id == entry.id)
        {
            Some(existing) => *existing = entry.clone(),
            None => {
                self.entries.push(entry.clone());
                changed = true;
            }
        }
        match self.hosts.iter_mut().find(|host| host.id == entry.id) {
            Some(host) => changed |= host.update_connected(&entry),
            None => {
                self.hosts.push(HostSlot::connected(&entry));
                changed = true;
            }
        }
        changed
    }

    pub(crate) fn mark_host_failed(&mut self, id: &HostId, failure: HostConnectFailure) -> bool {
        let status = HostConnectionStatus::Failed(failure);
        if let Some(host) = self.hosts.iter_mut().find(|host| &host.id == id) {
            if host.status == status {
                return false;
            }
            host.status = status;
            return true;
        }
        false
    }

    /// Color assigned to the given host in multi-host rows and legends.
    /// Returns `None` for single-host mode or an unknown host id.
    pub(crate) fn host_color(&self, id: &HostId) -> Option<Color> {
        if !self.is_multi() {
            return None;
        }
        self.hosts
            .iter()
            .position(|host| &host.id == id)
            .map(host_color_for_index)
    }

    /// Width of the compact host marker column when rendered in multi-host rows.
    /// Returns 0 for single-host (column is omitted).
    pub(crate) fn host_marker_width(&self) -> usize {
        if !self.is_multi() {
            return 0;
        }
        HOST_COLOR_SQUARE.chars().count()
    }

    /// Host-color legend shown in the multi-host top status bar.
    pub(crate) fn host_color_legend(&self) -> Option<Vec<HostLegendItem>> {
        if !self.is_multi() {
            return None;
        }
        Some(
            self.hosts
                .iter()
                .enumerate()
                .map(|(index, host)| HostLegendItem {
                    color: host_color_for_index(index),
                    label: host.label.clone(),
                    failed: host.status.is_failed(),
                })
                .collect(),
        )
    }

    pub(crate) fn host_sort_index(&self, id: &HostId) -> usize {
        self.hosts
            .iter()
            .position(|host| host.id == *id)
            .unwrap_or(usize::MAX)
    }
}

fn host_color_for_index(index: usize) -> Color {
    HOST_COLOR_PALETTE[index % HOST_COLOR_PALETTE.len()]
}

/// Identity of a session as returned to callers from the highlighted row.
/// Carries the host id and label so dispatch (attach/kill/monitor) can route
/// to the correct `HostHandle` and render messages with the host context.
#[derive(Debug, Clone)]
pub(crate) struct SelectedSession {
    pub(crate) host_id: HostId,
    pub(crate) host_label: String,
    pub(crate) info: SessionInfo,
}

impl SelectedSession {
    pub(crate) fn id(&self) -> &str {
        self.info.id.as_str()
    }

    pub(crate) fn name(&self) -> &str {
        &self.info.name
    }
}

/// One row in the merged session list.
///
/// Activity vs age have different clock semantics (see DESIGN §Clock handling):
/// * **Activity** ("recent change") is observer-relative —
///   `local_now - activity_observed_at_local`, where
///   `activity_observed_at_local` is the binary's wall clock the last time
///   the activity timestamp moved forward. Insensitive to host/operator
///   clock skew because we never compare host time to local time for it.
/// * **Age** ("session lifetime") is `local_now - session.created` under
///   the NTP-synced clock assumption. Wildly skewed host clocks produce
///   mildly inaccurate "age" text but no functional regression.
#[derive(Clone)]
pub(crate) struct SessionRow {
    pub(crate) host_id: HostId,
    pub(crate) host_label: String,
    pub(crate) local_now: u64,
    pub(crate) activity_observed_at_local: u64,
    pub(crate) session: SessionInfo,
    pub(crate) selected_tag: Option<SessionSelectedTag>,
}

impl SessionRow {
    pub(crate) fn displayed_tag_value(&self) -> Option<&str> {
        self.selected_tag
            .as_ref()
            .map(|tag| tag.value.as_str())
            .filter(|value| !value.is_empty())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SessionSelectedTag {
    pub(crate) key: String,
    pub(crate) value: String,
}

/// Per-session, per-host activity tracker.
///
/// At each refresh tick, for every observed `(host_id, session_id)`:
/// 1. If we've seen this session before and `session.activity` advanced from
///    the previous tick, record the move at `local_now`.
/// 2. If we've seen it and the activity timestamp didn't move, leave the
///    recorded "last observed change" alone — activity displays as the time
///    since the recorded mark.
/// 3. If we've never seen it (first sight), seed the recorded mark to
///    `local_now − max(0, local_now − session.activity)` so the displayed
///    recency immediately reflects how stale the activity timestamp is at
///    first sight, rather than starting at "now". Under the NTP-synced
///    clock assumption, host activity timestamps and `local_now` come from
///    the same wall clock, so the seeding math compares them directly.
///
/// State is keyed by `(HostId, SessionId)` and pruned when sessions disappear
/// from the merged listing.
#[derive(Default, Clone)]
pub(crate) struct ActivityTracker {
    last_seen: HashMap<(HostId, String), ActivityState>,
}

#[derive(Debug, Clone, Copy)]
struct ActivityState {
    activity_ts: u64,
    observed_at_local: u64,
}

impl ActivityTracker {
    /// Update tracker state for one observation and return the local-time
    /// mark to display ("when did we last see activity move?").
    pub(crate) fn observe(
        &mut self,
        host_id: &HostId,
        session_id: &str,
        activity_ts: u64,
        local_now: u64,
    ) -> u64 {
        let key = (host_id.clone(), session_id.to_string());
        match self.last_seen.get_mut(&key) {
            Some(state) => {
                if activity_ts > state.activity_ts {
                    state.activity_ts = activity_ts;
                    state.observed_at_local = local_now;
                }
                state.observed_at_local
            }
            None => {
                let host_age = local_now.saturating_sub(activity_ts);
                let observed_at_local = local_now.saturating_sub(host_age);
                self.last_seen.insert(
                    key,
                    ActivityState {
                        activity_ts,
                        observed_at_local,
                    },
                );
                observed_at_local
            }
        }
    }

    /// Drop tracker entries that aren't in `keep`. Bounds memory across
    /// session create/destroy cycles.
    pub(crate) fn retain(&mut self, keep: &std::collections::HashSet<(HostId, String)>) {
        self.last_seen.retain(|key, _| keep.contains(key));
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.last_seen.len()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RetainedUiState {
    layout_mode: LayoutMode,
    selected_key: Option<(HostId, String)>,
    selected_index: usize,
    focus: Focus,
    left_percent: u16,
    top_percent: u16,
    sort_mode: SessionSortMode,
}

impl Default for RetainedUiState {
    fn default() -> Self {
        Self::new(LayoutMode::Normal)
    }
}

impl RetainedUiState {
    pub(crate) fn new(layout_mode: LayoutMode) -> Self {
        Self {
            layout_mode,
            selected_key: None,
            selected_index: 0,
            focus: Focus::List,
            left_percent: DEFAULT_LEFT_PERCENT,
            top_percent: DEFAULT_TOP_PERCENT,
            sort_mode: SessionSortMode::Activity,
        }
    }

    pub(crate) fn apply_to(&self, app: &mut AppState) {
        app.layout.mode = self.layout_mode;
        app.session_list.selected = self.selected_index;
        app.layout.focus = self.focus;
        app.layout.left_percent = self
            .left_percent
            .clamp(LANDSCAPE_MIN_LEFT_PERCENT, LANDSCAPE_MAX_LEFT_PERCENT);
        app.layout.top_percent = self
            .top_percent
            .clamp(PORTRAIT_MIN_TOP_PERCENT, PORTRAIT_MAX_TOP_PERCENT);
        app.session_list.sort_mode = self.sort_mode;
    }

    pub(crate) fn update_from(&mut self, app: &AppState) {
        self.layout_mode = app.layout.mode;
        self.selected_key = app
            .selected_session()
            .map(|session| (session.host_id.clone(), session.id().to_string()));
        self.selected_index = app.session_list.selected;
        self.focus = app.layout.focus;
        self.left_percent = app.layout.left_percent;
        self.top_percent = app.layout.top_percent;
        self.sort_mode = app.session_list.sort_mode;
    }

    pub(crate) fn selected_session_key(&self) -> Option<(HostId, String)> {
        self.selected_key.clone()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct LayoutState {
    pub(crate) mode: LayoutMode,
    pub(crate) focus: Focus,
    pub(crate) left_percent: u16,
    pub(crate) top_percent: u16,
}

#[derive(Default, Clone)]
pub(crate) struct SessionListState {
    pub(crate) rows: Vec<SessionRow>,
    pub(crate) selected: usize,
    pub(crate) scroll: usize,
    pub(crate) sort_mode: SessionSortMode,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SessionSortMode {
    #[default]
    Activity,
    TagGroup,
}

impl SessionListState {
    /// Set the merged set of rows, sorted by visible activity recency across
    /// all hosts. Tie-breaks: name, then session id, then host id.
    ///
    /// Sort key is the same coarse recency bucket rendered in the row, not the
    /// raw host `session.activity` or exact observer timestamp. This keeps the
    /// list stable when several active sessions all display `now`; otherwise
    /// invisible one-second host refresh timing differences can reshuffle rows
    /// on every poll.
    pub(crate) fn set_rows_sorted_by_activity(&mut self, mut rows: Vec<SessionRow>) {
        sort_rows_by_activity(&mut rows);
        self.rows = rows;
    }

    pub(crate) fn set_rows_sorted(&mut self, rows: Vec<SessionRow>, fleet: &HostFleet) {
        match self.sort_mode {
            SessionSortMode::Activity => self.set_rows_sorted_by_activity(rows),
            SessionSortMode::TagGroup => self.set_rows_grouped_by_tag(rows, fleet),
        }
    }

    pub(crate) fn set_rows_grouped_by_tag(&mut self, mut rows: Vec<SessionRow>, fleet: &HostFleet) {
        sort_rows_by_tag_group(&mut rows, fleet);
        self.rows = rows;
    }

    pub(crate) fn toggle_sort_mode(&mut self) -> SessionSortMode {
        self.sort_mode = match self.sort_mode {
            SessionSortMode::Activity => SessionSortMode::TagGroup,
            SessionSortMode::TagGroup => SessionSortMode::Activity,
        };
        self.sort_mode
    }

    pub(crate) fn resort(&mut self, fleet: &HostFleet) {
        let rows = std::mem::take(&mut self.rows);
        self.set_rows_sorted(rows, fleet);
    }

    pub(crate) fn select_first(&mut self) {
        self.selected = 0;
        self.scroll = 0;
    }

    pub(crate) fn selected_session(&self) -> Option<SelectedSession> {
        self.rows.get(self.selected).map(|row| SelectedSession {
            host_id: row.host_id.clone(),
            host_label: row.host_label.clone(),
            info: row.session.clone(),
        })
    }

    /// Try to keep the selection on the same `(host_id, session_id)` pair
    /// across a refresh. Falls back to clamping the existing index.
    pub(crate) fn preserve_selection(&mut self, previous_key: Option<(HostId, String)>) {
        if self.rows.is_empty() {
            self.selected = 0;
            self.scroll = 0;
            return;
        }

        if let Some((host_id, session_id)) = previous_key {
            if let Some(pos) = self
                .rows
                .iter()
                .position(|row| row.host_id == host_id && row.session.id.as_str() == session_id)
            {
                self.selected = pos;
                return;
            }
        }
        self.selected = min(self.selected, self.rows.len().saturating_sub(1));
    }

    pub(crate) fn move_selection(&mut self, delta: isize) -> bool {
        if self.rows.is_empty() {
            return false;
        }
        let old = self.selected;
        let max_index = self.rows.len().saturating_sub(1) as isize;
        let next = (self.selected as isize + delta).clamp(0, max_index) as usize;
        self.selected = next;
        old != next
    }
}

fn sort_rows_by_activity(rows: &mut [SessionRow]) {
    rows.sort_by(activity_sort_order);
}

fn activity_sort_order(left: &SessionRow, right: &SessionRow) -> Ordering {
    activity_recency_bucket(left)
        .cmp(&activity_recency_bucket(right))
        .then_with(|| left.session.name.cmp(&right.session.name))
        .then_with(|| left.session.id.as_str().cmp(right.session.id.as_str()))
        .then_with(|| left.host_id.as_str().cmp(right.host_id.as_str()))
}

fn sort_rows_by_tag_group(rows: &mut [SessionRow], fleet: &HostFleet) {
    let group_activity = tag_group_activity_bucket(rows);
    rows.sort_by(|left, right| {
        tag_group_sort_order(left, right, &group_activity)
            .then_with(|| activity_recency_bucket(left).cmp(&activity_recency_bucket(right)))
            .then_with(|| {
                fleet
                    .host_sort_index(&left.host_id)
                    .cmp(&fleet.host_sort_index(&right.host_id))
            })
            .then_with(|| left.host_label.cmp(&right.host_label))
            .then_with(|| left.session.name.cmp(&right.session.name))
            .then_with(|| left.session.id.as_str().cmp(right.session.id.as_str()))
            .then_with(|| left.host_id.as_str().cmp(right.host_id.as_str()))
    });
}

fn activity_recency_bucket(row: &SessionRow) -> u64 {
    let seconds = row.local_now.saturating_sub(row.activity_observed_at_local);
    if seconds < 60 {
        0
    } else if seconds < 60 * 60 {
        1 + seconds / 60
    } else if seconds < 48 * 60 * 60 {
        61 + seconds / 60 / 60
    } else {
        109 + seconds / 60 / 60 / 24
    }
}

fn tag_group_activity_bucket(rows: &[SessionRow]) -> HashMap<String, u64> {
    let mut group_activity: HashMap<String, u64> = HashMap::new();
    for row in rows {
        let Some(value) = row.displayed_tag_value() else {
            continue;
        };
        let bucket = activity_recency_bucket(row);
        group_activity
            .entry(value.to_string())
            .and_modify(|activity| *activity = (*activity).min(bucket))
            .or_insert(bucket);
    }
    group_activity
}

fn tag_group_sort_order(
    left: &SessionRow,
    right: &SessionRow,
    group_activity: &HashMap<String, u64>,
) -> Ordering {
    match (left.displayed_tag_value(), right.displayed_tag_value()) {
        (Some(left_value), Some(right_value)) => {
            let left_activity = group_activity.get(left_value).copied().unwrap_or(0);
            let right_activity = group_activity.get(right_value).copied().unwrap_or(0);
            left_activity
                .cmp(&right_activity)
                .then_with(|| left_value.cmp(right_value))
        }
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

pub(crate) struct DetailState {
    pub(crate) lines: Vec<String>,
    pub(crate) scroll: usize,
    /// Render-feedback cache used by next-tick scroll math. The renderer owns
    /// the actual viewport size; input handling uses this last known value.
    pub(crate) last_known_view_height: usize,
    pub(crate) source: DetailSource,
    pub(crate) auto_tail: bool,
}

impl DetailState {
    pub(crate) fn set_text(&mut self, text: String) {
        self.lines = text.lines().map(|line| line.to_string()).collect();
        self.scroll = 0;
        self.auto_tail = true;
    }

    pub(crate) fn max_scroll(&self) -> usize {
        self.lines.len().saturating_sub(self.last_known_view_height)
    }

    pub(crate) fn scroll(&mut self, delta: isize) -> bool {
        let old = self.scroll;
        if delta < 0 {
            self.scroll = self.scroll.saturating_sub(delta.unsigned_abs());
        } else {
            self.scroll = self
                .scroll
                .saturating_add(delta as usize)
                .min(self.max_scroll());
        }
        self.auto_tail = self.scroll == 0;
        old != self.scroll
    }

    pub(crate) fn home(&mut self) {
        self.scroll = self.max_scroll();
        self.auto_tail = false;
    }

    pub(crate) fn end(&mut self) {
        self.scroll = 0;
        self.auto_tail = true;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StatusBanner {
    Loading(String),
    Info(String),
    Error(String),
}

impl StatusBanner {
    pub(crate) fn loading(text: impl Into<String>) -> Self {
        Self::Loading(text.into())
    }

    pub(crate) fn info(text: impl Into<String>) -> Self {
        Self::Info(text.into())
    }

    pub(crate) fn error(text: impl Into<String>) -> Self {
        Self::Error(text.into())
    }

    pub(crate) fn text(&self) -> &str {
        match self {
            StatusBanner::Loading(text) | StatusBanner::Info(text) | StatusBanner::Error(text) => {
                text
            }
        }
    }

    pub(crate) fn style(&self) -> Style {
        match self {
            StatusBanner::Loading(_) => Style::default().fg(STATUS_BAR_FG).bg(STATUS_BAR_BG),
            StatusBanner::Info(_) => Style::default().fg(Color::Yellow).bg(STATUS_BAR_BG),
            StatusBanner::Error(_) => Style::default().fg(Color::Red).bg(STATUS_BAR_BG),
        }
    }
}

pub(crate) struct AppState {
    pub(crate) fleet: HostFleet,
    pub(crate) layout: LayoutState,
    pub(crate) session_list: SessionListState,
    pub(crate) detail: DetailState,
    pub(crate) status: StatusBanner,
    pub(crate) modal: Option<ModalState>,
    pub(crate) activity_tracker: ActivityTracker,
}

impl AppState {
    /// Single-host test constructor. Builds a fleet with one entry from the
    /// given label/IP.
    #[cfg(test)]
    pub(crate) fn new(host_label: String, layout_mode: LayoutMode) -> Self {
        Self::new_with_host_ip(host_label, "unknown".to_string(), layout_mode)
    }

    #[cfg(test)]
    pub(crate) fn new_with_host_ip(
        host_label: String,
        host_ip_address: String,
        layout_mode: LayoutMode,
    ) -> Self {
        let entry = HostEntry {
            id: HostId::local(),
            alias: host_label.clone(),
            label: host_label,
            ip_address: host_ip_address,
            handle: motlie_tmux::HostHandle::local(),
        };
        let fleet = HostFleet::from_entries(vec![entry]);
        Self::with_fleet(fleet, layout_mode)
    }

    pub(crate) fn with_fleet(fleet: HostFleet, layout_mode: LayoutMode) -> Self {
        Self {
            fleet,
            layout: LayoutState {
                mode: layout_mode,
                focus: Focus::List,
                left_percent: DEFAULT_LEFT_PERCENT,
                top_percent: DEFAULT_TOP_PERCENT,
            },
            session_list: SessionListState::default(),
            detail: DetailState {
                lines: Vec::new(),
                scroll: 0,
                last_known_view_height: 1,
                source: DetailSource::sample(),
                auto_tail: true,
            },
            status: StatusBanner::loading("loading sessions"),
            modal: None,
            activity_tracker: ActivityTracker::default(),
        }
    }

    pub(crate) fn selected_session(&self) -> Option<SelectedSession> {
        self.session_list.selected_session()
    }

    pub(crate) fn preserve_selection(&mut self, previous: Option<(HostId, String)>) {
        self.session_list.preserve_selection(previous);
    }

    pub(crate) fn move_selection(&mut self, delta: isize) -> bool {
        self.session_list.move_selection(delta)
    }

    pub(crate) fn set_detail_text(&mut self, text: String) {
        self.detail.set_text(text);
    }

    pub(crate) fn scroll_detail(&mut self, delta: isize) {
        if self.detail.scroll(delta) {
            self.status = StatusBanner::info(format!("detail offset {}", self.detail.scroll));
        }
    }

    pub(crate) fn detail_home(&mut self) {
        self.detail.home();
    }

    pub(crate) fn detail_end(&mut self) {
        self.detail.end();
    }

    pub(crate) fn focus_next(&mut self) {
        self.layout.focus = match (self.layout.mode, self.layout.focus) {
            (LayoutMode::Normal, Focus::List) => Focus::Detail,
            (LayoutMode::Normal, Focus::Detail) => Focus::List,
            (LayoutMode::Portrait, Focus::List) => Focus::Detail,
            (LayoutMode::Portrait, Focus::Detail) => Focus::List,
        };
    }

    pub(crate) fn toggle_layout(&mut self) {
        self.layout.mode = match self.layout.mode {
            LayoutMode::Normal => LayoutMode::Portrait,
            LayoutMode::Portrait => LayoutMode::Normal,
        };
        self.status = StatusBanner::info("layout toggled");
    }
}
