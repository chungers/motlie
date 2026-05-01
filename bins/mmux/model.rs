use std::cmp::{max, min};
use std::collections::HashMap;

use motlie_tmux::{HostHandle, SessionInfo};
use ratatui::style::{Color, Style};

use crate::consts::{
    DEFAULT_LEFT_PERCENT, DEFAULT_TOP_PERCENT, LANDSCAPE_MAX_LEFT_PERCENT,
    LANDSCAPE_MIN_LEFT_PERCENT, MODAL_TEXT_FIELD_HEIGHT, PORTRAIT_MAX_TOP_PERCENT,
    PORTRAIT_MIN_TOP_PERCENT, STATUS_BAR_BG,
};
use crate::detail::DetailSource;

const SESSION_TAGS_INPUT_SECTION_HEIGHT: u16 = 2;
const SESSION_TAGS_LIST_MAX_ROWS: u16 = 5;
const SESSION_TAGS_EDIT_ROW_OVERHEAD: usize = 6;
const SESSION_TAGS_LIST_ROW_OVERHEAD: usize = 7;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayoutMode {
    Normal,
    Portrait,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Focus {
    Motd,
    List,
    Detail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Button {
    Cancel,
    Ok,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SessionTagsFocus {
    TagRow(usize),
    Key,
    Value,
    Cancel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SessionTagRow {
    pub(crate) key: String,
    pub(crate) value: String,
}

#[derive(Debug, Clone)]
pub(crate) enum ModalState {
    NewSession {
        input: String,
        button: Button,
    },
    KillSession {
        host_id: HostId,
        host_label: String,
        id: String,
        name: String,
        button: Button,
    },
    RenameSession {
        host_id: HostId,
        host_label: String,
        id: String,
        current_name: String,
        input: String,
        button: Button,
    },
    SessionTags {
        host_id: HostId,
        host_label: String,
        id: String,
        name: String,
        tags: Vec<SessionTagRow>,
        sort_key: Option<String>,
        key_input: String,
        value_input: String,
        focus: SessionTagsFocus,
    },
    Help,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModalView {
    pub(crate) title: &'static str,
    pub(crate) body: ModalBody,
    pub(crate) buttons: String,
    pub(crate) active_button: Button,
}

impl ModalView {
    #[cfg(test)]
    pub(crate) fn body_text(&self) -> String {
        match &self.body {
            ModalBody::Text(text) => text.clone(),
            ModalBody::NewSession { input } => format!("Session name\n{input}"),
            ModalBody::RenameSession { input } => format!("Session Name\n{input}"),
            ModalBody::SessionTags {
                tags,
                sort_key,
                key_input,
                value_input,
                ..
            } => {
                let rows = if tags.is_empty() {
                    "No tags".to_string()
                } else {
                    tags.iter()
                        .map(|tag| {
                            let marker = if sort_key.as_deref() == Some(tag.key.as_str()) {
                                "✓"
                            } else {
                                " "
                            };
                            format!("{}    {} {marker}", tag.key, tag.value)
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                };
                format!("{rows}\n{key_input}    {value_input}")
            }
        }
    }

    pub(crate) fn content_height(&self) -> u16 {
        match &self.body {
            ModalBody::Text(text) => max(1, text.lines().count()) as u16,
            ModalBody::NewSession { .. } => 1 + MODAL_TEXT_FIELD_HEIGHT,
            ModalBody::RenameSession { .. } => 1 + MODAL_TEXT_FIELD_HEIGHT,
            ModalBody::SessionTags { tags, .. } => {
                let rows = max(1, tags.len()) as u16;
                min(rows, SESSION_TAGS_LIST_MAX_ROWS) + SESSION_TAGS_INPUT_SECTION_HEIGHT
            }
        }
    }

    pub(crate) fn content_width(&self) -> u16 {
        let body_width = match &self.body {
            ModalBody::Text(text) => text
                .lines()
                .map(|line| line.chars().count())
                .max()
                .unwrap_or(0),
            ModalBody::NewSession { input } => {
                max("Session name".chars().count(), input.chars().count())
            }
            ModalBody::RenameSession { input } => {
                max("Session Name".chars().count(), input.chars().count())
            }
            ModalBody::SessionTags {
                tags,
                key_input,
                value_input,
                ..
            } => tags
                .iter()
                .map(|tag| {
                    tag.key.chars().count()
                        + tag.value.chars().count()
                        + SESSION_TAGS_LIST_ROW_OVERHEAD
                })
                .chain([
                    "No tags".chars().count() + SESSION_TAGS_LIST_ROW_OVERHEAD,
                    key_input.chars().count()
                        + value_input.chars().count()
                        + SESSION_TAGS_EDIT_ROW_OVERHEAD,
                ])
                .max()
                .unwrap_or(0),
        };
        max(body_width, self.buttons.chars().count()) as u16
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ModalBody {
    Text(String),
    NewSession {
        input: String,
    },
    RenameSession {
        input: String,
    },
    SessionTags {
        tags: Vec<SessionTagRow>,
        sort_key: Option<String>,
        key_input: String,
        value_input: String,
        focus: SessionTagsFocus,
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

/// One configured host: stable id, display label, IP, and the connected handle.
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
    pub(crate) label: String,
    pub(crate) ip_address: String,
    pub(crate) handle: HostHandle,
}

/// Collection of one or more configured hosts. Always has at least one entry
/// (localhost when no SSH URIs are specified). Multi-host mode is implicit:
/// `is_multi() == true` iff `len() > 1`.
///
/// This is **not** `motlie_tmux::Fleet` (`libs/tmux/src/fleet.rs`). That type
/// is a monitoring/automation registry — it owns a shared `OutputBus`,
/// manages per-host `MonitorHandle` lifecycle, and stores workstream alias→
/// `TargetSpec` bindings, which are orthogonal to the selector's needs.
/// `HostFleet` is the selector's *display-and-routing* fleet: a flat list of
/// hosts with per-row metadata (label, ip) and 1 Hz fan-out polling via
/// `HostHandle::list_sessions()`. See `docs/DESIGN.md` §Multi-host Mode →
/// §Internal data model for the full rationale.
#[derive(Clone, Default)]
pub(crate) struct HostFleet {
    pub(crate) entries: Vec<HostEntry>,
}

/// Cap for the hostname column width in multi-host row format. Longer labels
/// are truncated with an ellipsis to keep rows readable.
pub(crate) const HOST_LABEL_COLUMN_MAX: usize = 24;

impl HostFleet {
    pub(crate) fn from_entries(entries: Vec<HostEntry>) -> Self {
        Self { entries }
    }

    pub(crate) fn is_multi(&self) -> bool {
        self.entries.len() > 1
    }

    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(crate) fn entry(&self, id: &HostId) -> Option<&HostEntry> {
        self.entries.iter().find(|entry| &entry.id == id)
    }

    pub(crate) fn first(&self) -> Option<&HostEntry> {
        self.entries.first()
    }

    /// Width of the hostname column when rendered in multi-host rows.
    /// Returns 0 for single-host (column is omitted).
    pub(crate) fn host_label_width(&self) -> usize {
        if !self.is_multi() {
            return 0;
        }
        self.entries
            .iter()
            .map(|entry| entry.label.chars().count().min(HOST_LABEL_COLUMN_MAX))
            .max()
            .unwrap_or(0)
    }
}

/// Identity of a session as returned to callers from the highlighted row.
/// Carries the host id and label so dispatch (attach/kill/monitor) can route
/// to the correct `HostHandle` and render messages with the host context.
#[derive(Debug, Clone)]
pub(crate) struct SelectedSession {
    pub(crate) host_id: HostId,
    pub(crate) host_label: String,
    pub(crate) id: String,
    pub(crate) name: String,
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
        }
    }

    pub(crate) fn apply_to(&self, app: &mut AppState) {
        app.layout.mode = self.layout_mode;
        app.session_list.selected = self.selected_index;
        app.layout.focus = focus_for_layout(self.focus, app.layout.mode);
        app.layout.left_percent = self
            .left_percent
            .clamp(LANDSCAPE_MIN_LEFT_PERCENT, LANDSCAPE_MAX_LEFT_PERCENT);
        app.layout.top_percent = self
            .top_percent
            .clamp(PORTRAIT_MIN_TOP_PERCENT, PORTRAIT_MAX_TOP_PERCENT);
    }

    pub(crate) fn update_from(&mut self, app: &AppState) {
        self.layout_mode = app.layout.mode;
        self.selected_key = app
            .selected_session()
            .map(|session| (session.host_id, session.id));
        self.selected_index = app.session_list.selected;
        self.focus = app.layout.focus;
        self.left_percent = app.layout.left_percent;
        self.top_percent = app.layout.top_percent;
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

#[derive(Debug, Clone)]
pub(crate) struct MotdState {
    pub(crate) text: String,
    pub(crate) is_placeholder: bool,
}

#[derive(Default, Clone)]
pub(crate) struct SessionListState {
    pub(crate) rows: Vec<SessionRow>,
    pub(crate) selected: usize,
    pub(crate) scroll: usize,
}

impl SessionListState {
    /// Set the merged set of rows, sorted by activity descending across all
    /// hosts. Tie-breaks: name, then session id, then host id.
    ///
    /// Sort key is `activity_observed_at_local` (operator-side wall clock)
    /// not the raw host `session.activity`. This keeps the merged-list order
    /// consistent with the displayed activity column and is robust to host
    /// clock skew across hosts in multi-host mode — a host whose clock is
    /// minutes ahead of others doesn't pin its sessions to the top.
    pub(crate) fn set_rows_sorted_by_activity(&mut self, mut rows: Vec<SessionRow>) {
        rows.sort_by(|left, right| {
            right
                .activity_observed_at_local
                .cmp(&left.activity_observed_at_local)
                .then_with(|| left.session.name.cmp(&right.session.name))
                .then_with(|| left.session.id.as_str().cmp(right.session.id.as_str()))
                .then_with(|| left.host_id.as_str().cmp(right.host_id.as_str()))
        });
        self.rows = rows;
    }

    pub(crate) fn selected_session(&self) -> Option<SelectedSession> {
        self.rows.get(self.selected).map(|row| SelectedSession {
            host_id: row.host_id.clone(),
            host_label: row.host_label.clone(),
            id: row.session.id.as_str().to_string(),
            name: row.session.name.clone(),
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
            StatusBanner::Loading(_) => Style::default().fg(Color::White).bg(STATUS_BAR_BG),
            StatusBanner::Info(_) => Style::default().fg(Color::Yellow).bg(STATUS_BAR_BG),
            StatusBanner::Error(_) => Style::default().fg(Color::Red).bg(STATUS_BAR_BG),
        }
    }
}

pub(crate) struct AppState {
    pub(crate) fleet: HostFleet,
    pub(crate) layout: LayoutState,
    /// `None` in multi-host mode (MOTD pane is hidden when multiple hosts are
    /// listed). `Some` in single-host mode with the host's `/etc/motd` text or
    /// the motlie placeholder.
    pub(crate) motd: Option<MotdState>,
    pub(crate) session_list: SessionListState,
    pub(crate) detail: DetailState,
    pub(crate) status: StatusBanner,
    pub(crate) modal: Option<ModalState>,
    pub(crate) activity_tracker: ActivityTracker,
}

impl AppState {
    /// Single-host test constructor. Builds a fleet with one entry from the
    /// given label/IP and (optional) MOTD text.
    #[cfg(test)]
    pub(crate) fn new(
        host_label: String,
        layout_mode: LayoutMode,
        motd: String,
        placeholder: bool,
    ) -> Self {
        Self::new_with_host_ip(
            host_label,
            "unknown".to_string(),
            layout_mode,
            motd,
            placeholder,
        )
    }

    #[cfg(test)]
    pub(crate) fn new_with_host_ip(
        host_label: String,
        host_ip_address: String,
        layout_mode: LayoutMode,
        motd: String,
        placeholder: bool,
    ) -> Self {
        let entry = HostEntry {
            id: HostId::local(),
            label: host_label,
            ip_address: host_ip_address,
            handle: motlie_tmux::HostHandle::local(),
        };
        let fleet = HostFleet::from_entries(vec![entry]);
        Self::with_fleet(fleet, layout_mode, Some((motd, placeholder)))
    }

    /// Production constructor. `motd` should be `Some` only in single-host
    /// mode; multi-host mode passes `None` so the MOTD pane is hidden.
    pub(crate) fn with_fleet(
        fleet: HostFleet,
        layout_mode: LayoutMode,
        motd: Option<(String, bool)>,
    ) -> Self {
        Self {
            fleet,
            layout: LayoutState {
                mode: layout_mode,
                focus: Focus::List,
                left_percent: DEFAULT_LEFT_PERCENT,
                top_percent: DEFAULT_TOP_PERCENT,
            },
            motd: motd.map(|(text, is_placeholder)| MotdState {
                text,
                is_placeholder,
            }),
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
            (LayoutMode::Normal, Focus::Motd) => Focus::List,
            (LayoutMode::Normal, Focus::List) => Focus::Detail,
            // In multi-host (no MOTD) Normal mode, skip the MOTD focus state.
            (LayoutMode::Normal, Focus::Detail) if self.motd.is_some() => Focus::Motd,
            (LayoutMode::Normal, Focus::Detail) => Focus::List,
            (LayoutMode::Portrait, Focus::List) => Focus::Detail,
            (LayoutMode::Portrait, Focus::Detail) => Focus::List,
            (LayoutMode::Portrait, Focus::Motd) => Focus::List,
        };
    }

    pub(crate) fn toggle_layout(&mut self) {
        self.layout.mode = match self.layout.mode {
            LayoutMode::Normal => LayoutMode::Portrait,
            LayoutMode::Portrait => LayoutMode::Normal,
        };
        self.layout.focus = focus_for_layout(self.layout.focus, self.layout.mode);
        self.status = StatusBanner::info("layout toggled");
    }
}

pub(crate) fn focus_for_layout(focus: Focus, layout_mode: LayoutMode) -> Focus {
    match (layout_mode, focus) {
        (LayoutMode::Portrait, Focus::Motd) => Focus::List,
        _ => focus,
    }
}
