use std::cmp::min;
use std::collections::{HashMap, HashSet};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use motlie_tmux::{
    CreateSessionOptions, HostHandle, KeySequence, SessionEnvVar, SessionId, SessionInfo,
    SessionTag,
};

use crate::consts::{
    DEFAULT_DETAIL_LINES, LANDSCAPE_MAX_LEFT_PERCENT, LANDSCAPE_MIN_LEFT_PERCENT,
    PORTRAIT_MAX_TOP_PERCENT, PORTRAIT_MIN_TOP_PERCENT,
};
use crate::detail::{DetailMode, DetailSource, SessionDetailSource};
use crate::model::{
    ActivityTracker, AppState, Button, Focus, HostEntry, HostFleet, HostId, LayoutMode, ModalState,
    NewSessionFocus, NewSessionHostChoice, NewSessionModalUi, SelectedSession, SelectionKey,
    SendKeysFocus, SendKeysModalUi, SessionKeyValueFocus, SessionKeyValueKind,
    SessionKeyValueModalUi, SessionKeyValueRow, SessionRow, SessionSelectedTag, SessionSortMode,
    StatusBanner,
};

const TAG_PREFIX: &str = "mmux";
const SELECTED_TAG_KEY_OPTION: &str = "__selected-key";
const RESERVED_TAG_KEYS: &[&str] = &[SELECTED_TAG_KEY_OPTION];
const SEND_KEYS_THEN_ENTER_SUFFIX: &str = "$$";
const SEND_KEYS_ENTER_DELAY: Duration = Duration::from_millis(500);

pub(crate) struct HostRefreshResult {
    host_id: HostId,
    host_label: String,
    diagnostic_label: String,
    local_now: u64,
    result: std::result::Result<HostSessionSnapshot, String>,
}

struct HostSessionSnapshot {
    sessions: Vec<SessionInfo>,
    selected_tags: HashMap<SessionId, SessionSelectedTag>,
}

pub(crate) struct FleetRefresh {
    hosts: Vec<HostRefreshResult>,
}

/// Fan out `list_sessions()` across all configured hosts in parallel, merge
/// into a flat `Vec<SessionRow>`, run the activity-tracker over the
/// observations, and return rows for the caller to sort.
///
/// Per-host failures are tolerated: a failing host's rows simply do not
/// appear in the merged list, and the failure surfaces in the status banner.
/// One host failing does not block the others.
///
/// Activity vs age clock semantics (see DESIGN §Clock handling): activity
/// recency is observer-relative via `ActivityTracker`; age is computed
/// against `local_now` under the NTP-synced clock assumption. Tracker
/// state is pruned to drop entries for sessions that have disappeared from
/// the merged listing.
#[cfg(test)]
pub(crate) async fn fetch_fleet_rows(
    fleet: &HostFleet,
    tracker: &mut ActivityTracker,
) -> (Vec<SessionRow>, Vec<String>) {
    let refresh = fetch_fleet_refresh(fleet).await;
    let (rows, failures, keep_keys) = rows_from_fleet_refresh(tracker, &refresh, None);
    tracker.retain(&keep_keys);
    (rows, failures)
}

pub(crate) async fn fetch_fleet_refresh(fleet: &HostFleet) -> FleetRefresh {
    let hosts = futures::future::join_all(fleet.entries.iter().map(fetch_host_refresh)).await;
    FleetRefresh { hosts }
}

pub(crate) async fn fetch_host_refresh(entry: &HostEntry) -> HostRefreshResult {
    let local_now = local_epoch_seconds();
    let result = match entry.handle.list_sessions().await {
        Ok(sessions) => {
            let selected_tags = load_selected_session_tags(&entry.handle, &sessions).await;
            Ok(HostSessionSnapshot {
                sessions,
                selected_tags,
            })
        }
        Err(err) => Err(err.to_string()),
    };
    HostRefreshResult {
        host_id: entry.id.clone(),
        host_label: entry.label.clone(),
        diagnostic_label: entry.diagnostic_label(),
        local_now,
        result,
    }
}

fn rows_from_fleet_refresh(
    tracker: &mut ActivityTracker,
    refresh: &FleetRefresh,
    excluded: Option<&SelectionKey>,
) -> (Vec<SessionRow>, Vec<String>, HashSet<SelectionKey>) {
    let mut rows = Vec::new();
    let mut failures = Vec::new();
    let mut keep_keys: HashSet<SelectionKey> = HashSet::new();
    for host in &refresh.hosts {
        match &host.result {
            Ok(snapshot) => {
                for session in snapshot.sessions.iter().cloned() {
                    let key = (host.host_id.clone(), session.id.as_str().to_string());
                    if excluded.is_some_and(|(host_id, session_id)| {
                        host_id == &host.host_id && session_id.as_str() == session.id.as_str()
                    }) {
                        continue;
                    }
                    let activity_observed_at_local = tracker.observe(
                        &host.host_id,
                        session.id.as_str(),
                        session.activity,
                        host.local_now,
                    );
                    let selected_tag = snapshot.selected_tags.get(&session.id).cloned();
                    keep_keys.insert(key);
                    rows.push(SessionRow {
                        host_id: host.host_id.clone(),
                        host_label: host.host_label.clone(),
                        local_now: host.local_now,
                        activity_observed_at_local,
                        session,
                        selected_tag,
                    });
                }
            }
            Err(err) => {
                failures.push(format!("{}: {err}", host.diagnostic_label));
            }
        }
    }
    (rows, failures, keep_keys)
}

fn local_epoch_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

async fn load_selected_session_tags(
    host: &HostHandle,
    sessions: &[SessionInfo],
) -> HashMap<SessionId, SessionSelectedTag> {
    let Ok(tags_by_session) = host.list_tags_for_session_infos(TAG_PREFIX, sessions).await else {
        return HashMap::new();
    };
    tags_by_session
        .into_iter()
        .filter_map(|(id, tags)| selected_session_tag_from_tags(&tags).map(|tag| (id, tag)))
        .collect()
}

fn selected_session_tag_from_tags(tags: &[SessionTag]) -> Option<SessionSelectedTag> {
    let selected_key = selected_key_from_tags(tags)?;
    tags.iter()
        .find(|tag| tag.key() == selected_key.as_str())
        .map(|tag| SessionSelectedTag {
            key: tag.key().to_string(),
            value: tag.value().to_string(),
        })
}

pub(crate) async fn refresh_sessions(
    fleet: &HostFleet,
    app: &mut AppState,
    force_detail: bool,
) -> Result<()> {
    let previous = current_selection_key(app);
    refresh_sessions_preserving_with_status(fleet, app, force_detail, previous, true, None).await
}

pub(crate) async fn refresh_sessions_quiet(
    fleet: &HostFleet,
    app: &mut AppState,
    force_detail: bool,
) -> Result<()> {
    let previous = current_selection_key(app);
    refresh_sessions_preserving_with_status(fleet, app, force_detail, previous, false, None).await
}

#[cfg(test)]
pub(crate) async fn refresh_sessions_preserving(
    fleet: &HostFleet,
    app: &mut AppState,
    force_detail: bool,
    previous: Option<SelectionKey>,
) -> Result<()> {
    refresh_sessions_preserving_with_status(fleet, app, force_detail, previous, true, None).await
}

async fn refresh_sessions_excluding(
    fleet: &HostFleet,
    app: &mut AppState,
    force_detail: bool,
    excluded: SelectionKey,
) -> Result<()> {
    let previous = current_selection_key(app);
    refresh_sessions_preserving_with_status(
        fleet,
        app,
        force_detail,
        previous,
        true,
        Some(excluded),
    )
    .await
}

fn current_selection_key(app: &AppState) -> Option<SelectionKey> {
    app.selected_session()
        .map(|session| (session.host_id.clone(), session.id().to_string()))
}

async fn refresh_sessions_preserving_with_status(
    fleet: &HostFleet,
    app: &mut AppState,
    force_detail: bool,
    previous: Option<SelectionKey>,
    update_status: bool,
    excluded: Option<SelectionKey>,
) -> Result<()> {
    let refresh = fetch_fleet_refresh(fleet).await;
    apply_fleet_snapshot(
        fleet,
        app,
        refresh,
        RefreshApplyOptions::after_action(force_detail, previous, update_status, excluded),
    )
    .await
}

#[derive(Clone)]
pub(crate) struct RefreshApplyOptions {
    kind: RefreshApplyKind,
}

#[derive(Clone)]
enum RefreshApplyKind {
    Initial {
        previous: Option<SelectionKey>,
    },
    Periodic,
    HostConnected {
        previous: Option<SelectionKey>,
    },
    AfterAction {
        force_detail: bool,
        previous: Option<SelectionKey>,
        update_status: bool,
        excluded: Option<SelectionKey>,
    },
}

impl RefreshApplyOptions {
    pub(crate) fn initial(previous: Option<SelectionKey>) -> Self {
        Self {
            kind: RefreshApplyKind::Initial { previous },
        }
    }

    pub(crate) fn periodic() -> Self {
        Self {
            kind: RefreshApplyKind::Periodic,
        }
    }

    pub(crate) fn host_connected(previous: Option<SelectionKey>) -> Self {
        Self {
            kind: RefreshApplyKind::HostConnected { previous },
        }
    }

    pub(crate) fn after_action(
        force_detail: bool,
        previous: Option<SelectionKey>,
        update_status: bool,
        excluded: Option<SelectionKey>,
    ) -> Self {
        Self {
            kind: RefreshApplyKind::AfterAction {
                force_detail,
                previous,
                update_status,
                excluded,
            },
        }
    }

    pub(crate) fn allow_detail_refresh(&self) -> bool {
        !matches!(self.kind, RefreshApplyKind::Initial { .. })
    }

    pub(crate) fn clear_previous_selection(&mut self) {
        match &mut self.kind {
            RefreshApplyKind::Initial { previous }
            | RefreshApplyKind::HostConnected { previous }
            | RefreshApplyKind::AfterAction { previous, .. } => *previous = None,
            RefreshApplyKind::Periodic => {}
        }
    }

    fn previous(&self) -> Option<SelectionKey> {
        match &self.kind {
            RefreshApplyKind::Initial { previous }
            | RefreshApplyKind::HostConnected { previous }
            | RefreshApplyKind::AfterAction { previous, .. } => previous.clone(),
            RefreshApplyKind::Periodic => None,
        }
    }

    fn excluded(&self) -> Option<&SelectionKey> {
        match &self.kind {
            RefreshApplyKind::AfterAction { excluded, .. } => excluded.as_ref(),
            RefreshApplyKind::Initial { .. }
            | RefreshApplyKind::Periodic
            | RefreshApplyKind::HostConnected { .. } => None,
        }
    }

    fn update_status(&self) -> bool {
        match &self.kind {
            RefreshApplyKind::Initial { .. } => true,
            RefreshApplyKind::AfterAction { update_status, .. } => *update_status,
            RefreshApplyKind::Periodic | RefreshApplyKind::HostConnected { .. } => false,
        }
    }

    fn force_detail(&self) -> bool {
        match &self.kind {
            RefreshApplyKind::AfterAction { force_detail, .. } => *force_detail,
            RefreshApplyKind::Initial { .. }
            | RefreshApplyKind::Periodic
            | RefreshApplyKind::HostConnected { .. } => false,
        }
    }
}

/// Replace the merged row list with a full fleet snapshot. Hosts that failed
/// in this snapshot are removed from the merged list, so use this for complete
/// reconciliation paths such as initial startup and user-triggered actions.
pub(crate) async fn apply_fleet_snapshot(
    fleet: &HostFleet,
    app: &mut AppState,
    refresh: FleetRefresh,
    options: RefreshApplyOptions,
) -> Result<()> {
    let (rows, failures, keep_keys) =
        rows_from_fleet_refresh(&mut app.activity_tracker, &refresh, options.excluded());
    app.activity_tracker.retain(&keep_keys);
    apply_refreshed_rows(fleet, app, rows, failures, options).await
}

/// Apply a streaming subset of host results. Hosts that failed in this batch
/// keep their prior rows visible, so transient host failures do not make rows
/// blink out during the per-tick refresh path.
pub(crate) async fn apply_streaming_host_results(
    fleet: &HostFleet,
    app: &mut AppState,
    refreshes: Vec<HostRefreshResult>,
    options: RefreshApplyOptions,
) -> Result<()> {
    if refreshes.is_empty() {
        return Ok(());
    }
    let succeeded_hosts = refreshes
        .iter()
        .filter_map(|refresh| refresh.result.is_ok().then_some(refresh.host_id.clone()))
        .collect::<HashSet<_>>();
    let refresh = FleetRefresh { hosts: refreshes };
    let (host_rows, failures, _) =
        rows_from_fleet_refresh(&mut app.activity_tracker, &refresh, options.excluded());
    let mut rows = Vec::with_capacity(app.session_list.rows.len() + host_rows.len());
    rows.extend(
        app.session_list
            .rows
            .iter()
            .filter(|row| !succeeded_hosts.contains(&row.host_id))
            .cloned(),
    );
    rows.extend(host_rows);
    app.activity_tracker.retain(&row_keys(&rows));
    apply_refreshed_rows(fleet, app, rows, failures, options).await
}

async fn apply_refreshed_rows(
    fleet: &HostFleet,
    app: &mut AppState,
    rows: Vec<SessionRow>,
    failures: Vec<String>,
    options: RefreshApplyOptions,
) -> Result<()> {
    let closed_monitored = closed_monitored_session(app, &rows);
    let previous_key = options.previous().or_else(|| current_selection_key(app));
    let selection_key_before_refresh = previous_key.clone();
    app.session_list.set_rows_sorted(rows, fleet);
    app.preserve_selection(previous_key);
    if options.update_status() {
        app.status = build_status(app, &failures);
    }
    let selected_key = current_selection_key(app);
    let mut monitor_just_closed = false;
    if let Some((host_id, id, name)) = closed_monitored {
        if let Some(name) = stop_monitor_if_closed(app, &host_id, &id, name).await {
            app.status = StatusBanner::info(format!("monitored session {name} closed"));
            monitor_just_closed = true;
        }
    }
    // Only re-render the detail pane when something the session-refresh
    // path actually changed: the caller forced it (user-driven action),
    // the highlighted row moved to a different (host, session), or the
    // monitored session just closed and we need to repaint as Sample.
    //
    // Quiet refreshes that find nothing changed in detail must NOT call
    // `refresh_detail`. In `Monitor` mode the inner render unconditionally
    // recaptures the pane (no `force` check), and doing that on every
    // session-refresh tick blocks the next draw — so updated `list-sessions`
    // activity in the row list lands a tick or more late. The main loop
    // owns the monitor refresh cadence (its 750 ms `refresh_detail` call);
    // session refresh does not need to drive it.
    let selection_changed = selection_key_before_refresh != selected_key;
    if options.allow_detail_refresh()
        && (options.force_detail() || selection_changed || monitor_just_closed)
    {
        refresh_detail(fleet, app, true).await?;
    }
    Ok(())
}

fn row_keys(rows: &[SessionRow]) -> HashSet<SelectionKey> {
    rows.iter()
        .map(|row| (row.host_id.clone(), row.session.id.as_str().to_string()))
        .collect()
}

fn build_status(app: &AppState, failures: &[String]) -> StatusBanner {
    if !failures.is_empty() {
        let summary = failures.join("; ");
        return StatusBanner::error(format!("host unreachable: {summary}"));
    }
    if app.session_list.rows.is_empty() {
        StatusBanner::info("no sessions")
    } else {
        StatusBanner::info(format!("{} session(s)", app.session_list.rows.len()))
    }
}

/// Detect whether the currently-monitored session has dropped out of the
/// merged listing (closed, or its host became unreachable). Returns the
/// `(host_id, session_id, name)` triple needed to deactivate the monitor.
fn closed_monitored_session(
    app: &AppState,
    refreshed_rows: &[SessionRow],
) -> Option<(HostId, String, String)> {
    let monitored_id = app.detail.source.monitored_session_id()?.to_string();
    let monitored_host = app.detail.source.monitored_host_id()?.clone();
    if refreshed_rows
        .iter()
        .any(|row| row.host_id == monitored_host && row.session.id.as_str() == monitored_id)
    {
        return None;
    }
    let name = app
        .session_list
        .rows
        .iter()
        .find(|row| row.host_id == monitored_host && row.session.id.as_str() == monitored_id)
        .map(|row| row.session.name.clone())
        .unwrap_or_else(|| monitored_id.clone());
    Some((monitored_host, monitored_id, name))
}

pub(crate) async fn refresh_detail(
    fleet: &HostFleet,
    app: &mut AppState,
    force: bool,
) -> Result<()> {
    if app.session_list.rows.is_empty() {
        app.detail.lines.clear();
        return Ok(());
    }

    let Some(selected) = app.selected_session() else {
        return Ok(());
    };
    let Some(host) = fleet_host(fleet, &selected.host_id) else {
        app.detail.lines.clear();
        return Ok(());
    };

    match app.detail.source.mode() {
        DetailMode::Sample => {
            if force || app.detail.lines.is_empty() {
                let text = app
                    .detail
                    .source
                    .render(host, &selected)
                    .await
                    .unwrap_or_else(|err| format!("sample error: {err:#}"));
                app.set_detail_text(text);
            }
        }
        DetailMode::Monitor => {
            let text = app
                .detail
                .source
                .render(host, &selected)
                .await
                .unwrap_or_else(|err| format!("monitor error: {err:#}"));
            app.detail.lines = text.lines().map(|line| line.to_string()).collect();
            if app.detail.auto_tail {
                app.detail.scroll = 0;
            }
        }
    }
    Ok(())
}

fn fleet_host<'a>(fleet: &'a HostFleet, host_id: &HostId) -> Option<&'a HostHandle> {
    fleet.entry(host_id).map(|entry| &entry.handle)
}

pub(crate) async fn stop_monitor_if_closed(
    app: &mut AppState,
    host_id: &HostId,
    session_id: &str,
    name: String,
) -> Option<String> {
    if app.detail.source.monitored_session_id() == Some(session_id)
        && app.detail.source.monitored_host_id() == Some(host_id)
    {
        stop_detail_source(app).await;
        app.detail.source = DetailSource::sample();
        app.detail.lines.clear();
        Some(name)
    } else {
        None
    }
}

pub(crate) enum KeyOutcome {
    Continue,
    Select(SelectedSession),
    Cancel,
}

pub(crate) async fn handle_key(
    fleet: &HostFleet,
    app: &mut AppState,
    key: KeyEvent,
) -> Result<KeyOutcome> {
    if app.modal.is_some() {
        return handle_modal_key(fleet, app, key).await;
    }

    match (key.code, key.modifiers) {
        (KeyCode::Char('c'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
            return Ok(KeyOutcome::Cancel);
        }
        (KeyCode::Char('q'), _) => return Ok(KeyOutcome::Cancel),
        (KeyCode::Esc, _) => app.layout.focus = Focus::List,
        (KeyCode::Char('h'), _) => app.modal = Some(ModalState::Help),
        (KeyCode::Char('m'), _) => start_monitor(fleet, app).await?,
        (KeyCode::Char('g'), _) if app.layout.focus == Focus::List => {
            toggle_session_grouping(fleet, app).await?;
        }
        (KeyCode::Char('n'), _) => {
            app.modal = Some(new_session_modal_state(fleet, app));
        }
        (KeyCode::Char('k'), _) => {
            if let Some(selected) = app.selected_session() {
                app.modal = Some(ModalState::KillSession {
                    session: selected,
                    button: Button::Cancel,
                });
            } else {
                app.status = StatusBanner::info("no session selected");
            }
        }
        (KeyCode::Char('r'), _) if app.layout.focus == Focus::List => {
            if let Some(selected) = app.selected_session() {
                app.modal = Some(ModalState::RenameSession {
                    input: selected.name().to_string(),
                    session: selected,
                    button: Button::Ok,
                });
            } else {
                app.status = StatusBanner::info("no session selected");
            }
        }
        (KeyCode::Char('t'), _) => {
            if let Some(selected) = app.selected_session() {
                open_session_key_values_modal(fleet, app, selected, SessionKeyValueKind::Tags)
                    .await?;
            } else {
                app.status = StatusBanner::info("no session selected");
            }
        }
        (KeyCode::Char('p'), _) | (KeyCode::Char('@'), _) => {
            if let Some(selected) = app.selected_session() {
                app.modal = Some(ModalState::SendKeys {
                    session: selected,
                    ui: SendKeysModalUi {
                        input: String::new(),
                        focus: SendKeysFocus::Input,
                    },
                });
            } else {
                app.status = StatusBanner::info("no session selected");
            }
        }
        (KeyCode::Char('a'), _) => {
            if let Some(selected) = app.selected_session() {
                return Ok(KeyOutcome::Select(selected));
            }
            app.status = StatusBanner::info("no session selected");
        }
        (KeyCode::Enter, _) if app.layout.focus == Focus::List => {
            if app.selected_session().is_some() {
                reset_to_sample_detail(fleet, app).await?;
            } else {
                app.status = StatusBanner::info("no session selected");
            }
        }
        (KeyCode::Up, modifiers)
            if app.layout.mode == LayoutMode::Portrait && is_resize_modifier(modifiers) =>
        {
            app.layout.top_percent = app
                .layout
                .top_percent
                .saturating_sub(5)
                .max(PORTRAIT_MIN_TOP_PERCENT);
        }
        (KeyCode::Down, modifiers)
            if app.layout.mode == LayoutMode::Portrait && is_resize_modifier(modifiers) =>
        {
            app.layout.top_percent = app
                .layout
                .top_percent
                .saturating_add(5)
                .min(PORTRAIT_MAX_TOP_PERCENT);
        }
        (KeyCode::Left, modifiers)
            if app.layout.mode == LayoutMode::Normal && is_resize_modifier(modifiers) =>
        {
            app.layout.left_percent = app
                .layout
                .left_percent
                .saturating_sub(5)
                .max(LANDSCAPE_MIN_LEFT_PERCENT);
        }
        (KeyCode::Right, modifiers)
            if app.layout.mode == LayoutMode::Normal && is_resize_modifier(modifiers) =>
        {
            app.layout.left_percent = app
                .layout
                .left_percent
                .saturating_add(5)
                .min(LANDSCAPE_MAX_LEFT_PERCENT);
        }
        (KeyCode::Char('b'), modifiers)
            if app.layout.mode == LayoutMode::Normal && is_word_left_resize(modifiers) =>
        {
            app.layout.left_percent = app
                .layout
                .left_percent
                .saturating_sub(5)
                .max(LANDSCAPE_MIN_LEFT_PERCENT);
        }
        (KeyCode::Char('f'), modifiers)
            if app.layout.mode == LayoutMode::Normal && is_word_right_resize(modifiers) =>
        {
            app.layout.left_percent = app
                .layout
                .left_percent
                .saturating_add(5)
                .min(LANDSCAPE_MAX_LEFT_PERCENT);
        }
        (KeyCode::Tab | KeyCode::Char('\t'), _) => app.focus_next(),
        (KeyCode::Char('l'), _) => app.toggle_layout(),
        (KeyCode::Char('u'), KeyModifiers::NONE) | (KeyCode::Up, _) => {
            move_focused_pane_line(fleet, app, -1).await?;
        }
        (KeyCode::Char('b'), KeyModifiers::NONE) | (KeyCode::Down, _) => {
            move_focused_pane_line(fleet, app, 1).await?;
        }
        (KeyCode::PageUp, _) => match app.layout.focus {
            Focus::List => {
                if app.move_selection(-10) {
                    reset_to_sample_detail(fleet, app).await?;
                }
            }
            Focus::Detail => {
                fetch_older_detail(fleet, app).await?;
                app.scroll_detail(10);
            }
        },
        (KeyCode::PageDown, _) => match app.layout.focus {
            Focus::List => {
                if app.move_selection(10) {
                    reset_to_sample_detail(fleet, app).await?;
                }
            }
            Focus::Detail => app.scroll_detail(-10),
        },
        (KeyCode::Home, _) => match app.layout.focus {
            Focus::List => {
                if !app.session_list.rows.is_empty() {
                    app.session_list.selected = 0;
                    reset_to_sample_detail(fleet, app).await?;
                }
            }
            Focus::Detail => app.detail_home(),
        },
        (KeyCode::End, _) => match app.layout.focus {
            Focus::List => {
                if !app.session_list.rows.is_empty() {
                    app.session_list.selected = app.session_list.rows.len().saturating_sub(1);
                    reset_to_sample_detail(fleet, app).await?;
                }
            }
            Focus::Detail => app.detail_end(),
        },
        _ => {}
    }

    Ok(KeyOutcome::Continue)
}

async fn move_list_selection(fleet: &HostFleet, app: &mut AppState, delta: isize) -> Result<()> {
    if app.move_selection(delta) {
        reset_to_sample_detail(fleet, app).await?;
    }
    Ok(())
}

async fn move_focused_pane_line(fleet: &HostFleet, app: &mut AppState, delta: isize) -> Result<()> {
    match app.layout.focus {
        Focus::List => move_list_selection(fleet, app, delta).await?,
        Focus::Detail => app.scroll_detail(-delta),
    }
    Ok(())
}

fn is_resize_modifier(modifiers: KeyModifiers) -> bool {
    modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SHIFT)
}

async fn toggle_session_grouping(fleet: &HostFleet, app: &mut AppState) -> Result<()> {
    let previous = current_selection_key(app);
    let mode = app.session_list.toggle_sort_mode();
    app.session_list.resort(fleet);
    app.session_list.select_first();
    if previous != current_selection_key(app) {
        reset_to_sample_detail(fleet, app).await?;
    }
    app.status = StatusBanner::info(match mode {
        SessionSortMode::Activity => "sort: activity",
        SessionSortMode::TagGroup => "group: tag",
    });
    Ok(())
}

fn is_word_left_resize(modifiers: KeyModifiers) -> bool {
    modifiers.intersects(KeyModifiers::ALT | KeyModifiers::CONTROL)
}

fn is_word_right_resize(modifiers: KeyModifiers) -> bool {
    modifiers.intersects(KeyModifiers::ALT | KeyModifiers::CONTROL)
}

fn new_session_modal_state(fleet: &HostFleet, app: &AppState) -> ModalState {
    let selected_host_id = app.selected_session().map(|session| session.host_id);
    let hosts = fleet
        .entries
        .iter()
        .map(|entry| NewSessionHostChoice {
            id: entry.id.clone(),
            label: entry.label.clone(),
        })
        .collect::<Vec<_>>();
    let host_index = selected_host_id
        .and_then(|selected| hosts.iter().position(|host| host.id == selected))
        .unwrap_or(0);
    ModalState::NewSession {
        ui: NewSessionModalUi {
            input: String::new(),
            hosts,
            host_index,
            env_rows: Vec::new(),
            env_key_input: String::new(),
            env_value_input: String::new(),
            focus: NewSessionFocus::Name,
            button: Button::Ok,
        },
    }
}

async fn reset_to_sample_detail(fleet: &HostFleet, app: &mut AppState) -> Result<()> {
    stop_detail_source(app).await;
    app.detail.source = DetailSource::sample();
    refresh_detail(fleet, app, true).await
}

enum ModalAction {
    None,
    Close,
    CreateSession {
        input: String,
        host_id: Option<HostId>,
        env_rows: Vec<SessionKeyValueRow>,
    },
    StageNewSessionEnv {
        key: String,
        value: String,
    },
    UnsetNewSessionEnv {
        index: usize,
    },
    KillSession {
        session: SelectedSession,
    },
    RenameSession {
        session: SelectedSession,
        input: String,
    },
    SendKeys {
        session: SelectedSession,
        input: String,
        mode: SendKeysSubmitMode,
    },
    StageKeyValue {
        kind: SessionKeyValueKind,
        key: String,
        value: String,
    },
    StageUnsetKeyValue {
        kind: SessionKeyValueKind,
        key: String,
        index: usize,
        selected_key: Option<String>,
    },
    StageSelectTag {
        key: Option<String>,
        index: usize,
    },
    ApplyKeyValues {
        session: SelectedSession,
        ui: SessionKeyValueModalUi,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SendKeysSubmitMode {
    Exact,
    ThenEnterAfterDelay,
}

async fn handle_modal_key(
    fleet: &HostFleet,
    app: &mut AppState,
    key: KeyEvent,
) -> Result<KeyOutcome> {
    let action = match app.modal.as_mut() {
        Some(ModalState::NewSession { ui }) => handle_new_session_modal_key(key, ui),
        Some(ModalState::KillSession { session, button }) => match key.code {
            KeyCode::Esc => ModalAction::Close,
            KeyCode::Tab | KeyCode::Char('\t') | KeyCode::BackTab => {
                *button = match *button {
                    Button::Cancel => Button::Ok,
                    Button::Ok => Button::Cancel,
                };
                ModalAction::None
            }
            KeyCode::Left => {
                *button = Button::Cancel;
                ModalAction::None
            }
            KeyCode::Right => {
                *button = Button::Ok;
                ModalAction::None
            }
            KeyCode::Enter if *button == Button::Ok => ModalAction::KillSession {
                session: session.clone(),
            },
            KeyCode::Enter => ModalAction::Close,
            _ => ModalAction::None,
        },
        Some(ModalState::RenameSession {
            session,
            input,
            button,
        }) => match key.code {
            KeyCode::Esc => ModalAction::Close,
            KeyCode::Left => {
                *button = Button::Cancel;
                ModalAction::None
            }
            KeyCode::Right => {
                *button = Button::Ok;
                ModalAction::None
            }
            KeyCode::Enter if *button == Button::Ok => ModalAction::RenameSession {
                session: session.clone(),
                input: input.clone(),
            },
            KeyCode::Enter => ModalAction::Close,
            _ if edit_focused_text_field(&key, true, input) => ModalAction::None,
            _ => ModalAction::None,
        },
        Some(ModalState::SendKeys { session, ui }) => {
            handle_send_keys_modal_key(key, session.clone(), ui)
        }
        Some(ModalState::SessionKeyValues { session, ui }) => {
            handle_session_key_values_modal_key(key, session.clone(), ui)
        }
        Some(ModalState::Help) => match key.code {
            KeyCode::Esc | KeyCode::Enter => ModalAction::Close,
            _ => ModalAction::None,
        },
        None => ModalAction::None,
    };

    match action {
        ModalAction::None => {}
        ModalAction::Close => {
            let discard_status = discarded_key_value_status(app.modal.as_ref());
            app.modal = None;
            if let Some(status) = discard_status {
                app.status = StatusBanner::info(status);
            }
        }
        ModalAction::CreateSession {
            input,
            host_id,
            env_rows,
        } => {
            app.modal = None;
            create_session_from_modal(fleet, app, host_id, input, env_rows).await?;
        }
        ModalAction::StageNewSessionEnv { key, value } => {
            stage_new_session_env(app, key, value);
        }
        ModalAction::UnsetNewSessionEnv { index } => {
            unset_new_session_env(app, index);
        }
        ModalAction::KillSession { session } => {
            app.modal = None;
            kill_session_from_modal(fleet, app, session).await?;
        }
        ModalAction::RenameSession { session, input } => {
            app.modal = None;
            rename_session_from_modal(fleet, app, session, input).await?;
        }
        ModalAction::SendKeys {
            session,
            input,
            mode,
        } => {
            if send_keys_from_modal(fleet, app, session, input, mode).await? {
                app.modal = None;
            }
        }
        ModalAction::StageKeyValue { kind, key, value } => {
            stage_key_value_in_modal(app, kind, key, value);
        }
        ModalAction::StageUnsetKeyValue {
            kind,
            key,
            index,
            selected_key,
        } => {
            stage_unset_key_value_in_modal(app, kind, key, index, selected_key);
        }
        ModalAction::StageSelectTag { key, index } => {
            stage_select_tag_in_modal(app, key, index);
        }
        ModalAction::ApplyKeyValues { session, ui } => {
            app.modal = None;
            apply_session_key_values_modal(fleet, app, session, ui).await?;
        }
    }
    Ok(KeyOutcome::Continue)
}

fn discarded_key_value_status(modal: Option<&ModalState>) -> Option<String> {
    match modal {
        Some(ModalState::SessionKeyValues { session, ui })
            if ui.kind == SessionKeyValueKind::Tags && !tag_modal_changes(ui).is_empty() =>
        {
            Some(format!("discarded tag changes on {}", session.name()))
        }
        _ => None,
    }
}

fn handle_send_keys_modal_key(
    key: KeyEvent,
    session: SelectedSession,
    ui: &mut SendKeysModalUi,
) -> ModalAction {
    match key.code {
        KeyCode::Esc => ModalAction::Close,
        KeyCode::Tab | KeyCode::Char('\t') => {
            ui.focus = next_focus(ui.focus, &SEND_KEYS_FOCUS_ORDER);
            ModalAction::None
        }
        KeyCode::BackTab => {
            ui.focus = previous_focus(ui.focus, &SEND_KEYS_FOCUS_ORDER);
            ModalAction::None
        }
        KeyCode::Left => {
            ui.focus = SendKeysFocus::Cancel;
            ModalAction::None
        }
        KeyCode::Right => {
            ui.focus = SendKeysFocus::Ok;
            ModalAction::None
        }
        KeyCode::Enter => {
            submit_send_keys_modal(key.modifiers.contains(KeyModifiers::CONTROL), session, ui)
        }
        _ if edit_focused_text_field(&key, ui.focus == SendKeysFocus::Input, &mut ui.input) => {
            ModalAction::None
        }
        _ => ModalAction::None,
    }
}

fn submit_send_keys_modal(
    with_delayed_enter: bool,
    session: SelectedSession,
    ui: &SendKeysModalUi,
) -> ModalAction {
    if ui.focus == SendKeysFocus::Cancel {
        return ModalAction::Close;
    }
    let can_submit =
        ui.focus == SendKeysFocus::Ok || (ui.focus == SendKeysFocus::Input && !ui.input.is_empty());
    if !can_submit {
        return ModalAction::None;
    }
    ModalAction::SendKeys {
        session,
        input: ui.input.clone(),
        mode: if with_delayed_enter {
            SendKeysSubmitMode::ThenEnterAfterDelay
        } else {
            SendKeysSubmitMode::Exact
        },
    }
}

const SEND_KEYS_FOCUS_ORDER: [SendKeysFocus; 3] = [
    SendKeysFocus::Input,
    SendKeysFocus::Ok,
    SendKeysFocus::Cancel,
];

fn handle_new_session_modal_key(key: KeyEvent, ui: &mut NewSessionModalUi) -> ModalAction {
    let multi_host = ui.hosts.len() > 1;
    match key.code {
        KeyCode::Esc => ModalAction::Close,
        KeyCode::Tab | KeyCode::Char('\t') => {
            set_new_session_focus(
                ui,
                next_new_session_focus(ui.focus, multi_host, ui.env_rows.len()),
            );
            ModalAction::None
        }
        KeyCode::BackTab => {
            set_new_session_focus(
                ui,
                previous_new_session_focus(ui.focus, multi_host, ui.env_rows.len()),
            );
            ModalAction::None
        }
        KeyCode::Up
            if multi_host && matches!(ui.focus, NewSessionFocus::Host | NewSessionFocus::Name) =>
        {
            set_new_session_focus(ui, NewSessionFocus::Host);
            if ui.host_index == 0 {
                ui.host_index = ui.hosts.len().saturating_sub(1);
            } else {
                ui.host_index = ui.host_index.saturating_sub(1);
            }
            ModalAction::None
        }
        KeyCode::Down
            if multi_host && matches!(ui.focus, NewSessionFocus::Host | NewSessionFocus::Name) =>
        {
            set_new_session_focus(ui, NewSessionFocus::Host);
            ui.host_index = (ui.host_index + 1) % ui.hosts.len();
            ModalAction::None
        }
        KeyCode::Up => {
            if let Some(focus) = previous_new_session_env_row_focus(ui.focus, ui.env_rows.len()) {
                set_new_session_focus(ui, focus);
            }
            ModalAction::None
        }
        KeyCode::Char('u')
            if key.modifiers == KeyModifiers::NONE
                && matches!(ui.focus, NewSessionFocus::EnvRow(_)) =>
        {
            move_new_session_env_row_focus_up(ui);
            ModalAction::None
        }
        KeyCode::Down => {
            if let Some(focus) = next_new_session_env_row_focus(ui.focus, ui.env_rows.len()) {
                set_new_session_focus(ui, focus);
            }
            ModalAction::None
        }
        KeyCode::Char('b')
            if key.modifiers == KeyModifiers::NONE
                && matches!(ui.focus, NewSessionFocus::EnvRow(_)) =>
        {
            move_new_session_env_row_focus_down(ui);
            ModalAction::None
        }
        KeyCode::Left => {
            set_new_session_focus(ui, NewSessionFocus::Cancel);
            ModalAction::None
        }
        KeyCode::Right => {
            set_new_session_focus(ui, NewSessionFocus::Ok);
            ModalAction::None
        }
        KeyCode::Enter
            if matches!(
                ui.focus,
                NewSessionFocus::EnvKey | NewSessionFocus::EnvValue
            ) =>
        {
            ModalAction::StageNewSessionEnv {
                key: ui.env_key_input.trim().to_string(),
                value: ui.env_value_input.clone(),
            }
        }
        KeyCode::Enter if ui.focus == NewSessionFocus::Cancel || ui.button == Button::Cancel => {
            ModalAction::Close
        }
        KeyCode::Enter => ModalAction::CreateSession {
            input: ui.input.clone(),
            host_id: ui.selected_host().map(|host| host.id.clone()),
            env_rows: ui.env_rows.clone(),
        },
        KeyCode::Char('x') if matches!(ui.focus, NewSessionFocus::EnvRow(_)) => {
            let NewSessionFocus::EnvRow(index) = ui.focus else {
                return ModalAction::None;
            };
            ModalAction::UnsetNewSessionEnv { index }
        }
        KeyCode::Char('m')
            if key.modifiers == KeyModifiers::NONE
                && matches!(ui.focus, NewSessionFocus::EnvRow(_)) =>
        {
            let NewSessionFocus::EnvRow(index) = ui.focus else {
                return ModalAction::None;
            };
            if let Some(row) = ui.env_rows.get(index) {
                ui.env_key_input = row.key.clone();
                ui.env_value_input = row.value.clone();
                set_new_session_focus(ui, NewSessionFocus::EnvValue);
            }
            ModalAction::None
        }
        _ if edit_new_session_text_field(&key, ui) => ModalAction::None,
        _ => ModalAction::None,
    }
}

fn set_new_session_focus(ui: &mut NewSessionModalUi, focus: NewSessionFocus) {
    ui.focus = focus;
    ui.button = match focus {
        NewSessionFocus::Cancel => Button::Cancel,
        _ => Button::Ok,
    };
}

fn edit_new_session_text_field(key: &KeyEvent, ui: &mut NewSessionModalUi) -> bool {
    match ui.focus {
        NewSessionFocus::Name => edit_text_field(key, &mut ui.input),
        NewSessionFocus::EnvKey => edit_text_field(key, &mut ui.env_key_input),
        NewSessionFocus::EnvValue => edit_text_field(key, &mut ui.env_value_input),
        _ => false,
    }
}

fn move_new_session_env_row_focus_up(ui: &mut NewSessionModalUi) {
    let NewSessionFocus::EnvRow(index) = ui.focus else {
        return;
    };
    set_new_session_focus(ui, NewSessionFocus::EnvRow(index.saturating_sub(1)));
}

fn move_new_session_env_row_focus_down(ui: &mut NewSessionModalUi) {
    let NewSessionFocus::EnvRow(index) = ui.focus else {
        return;
    };
    set_new_session_focus(
        ui,
        NewSessionFocus::EnvRow(min(index + 1, ui.env_rows.len().saturating_sub(1))),
    );
}

fn edit_focused_text_field(key: &KeyEvent, focused: bool, input: &mut String) -> bool {
    focused && edit_text_field(key, input)
}

fn edit_text_field(key: &KeyEvent, input: &mut String) -> bool {
    match key.code {
        KeyCode::Backspace => {
            input.pop();
            true
        }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            input.push(c);
            true
        }
        _ => false,
    }
}

fn next_focus<T: Copy + Eq>(focus: T, order: &[T]) -> T {
    cycle_focus(focus, order, 1)
}

fn previous_focus<T: Copy + Eq>(focus: T, order: &[T]) -> T {
    cycle_focus(focus, order, order.len().saturating_sub(1))
}

fn cycle_focus<T: Copy + Eq>(focus: T, order: &[T], offset: usize) -> T {
    if order.is_empty() {
        return focus;
    }
    let Some(index) = order.iter().position(|candidate| *candidate == focus) else {
        return focus;
    };
    order[(index + offset) % order.len()]
}

fn next_new_session_focus(
    focus: NewSessionFocus,
    multi_host: bool,
    env_row_count: usize,
) -> NewSessionFocus {
    let order = new_session_focus_order(multi_host, env_row_count);
    next_focus(normalized_new_session_focus(focus), &order)
}

fn previous_new_session_focus(
    focus: NewSessionFocus,
    multi_host: bool,
    env_row_count: usize,
) -> NewSessionFocus {
    let order = new_session_focus_order(multi_host, env_row_count);
    let previous = previous_focus(normalized_new_session_focus(focus), &order);
    if matches!(previous, NewSessionFocus::EnvRow(_)) {
        NewSessionFocus::EnvRow(env_row_count.saturating_sub(1))
    } else {
        previous
    }
}

fn new_session_focus_order(multi_host: bool, env_row_count: usize) -> Vec<NewSessionFocus> {
    let mut order = Vec::with_capacity(7);
    if multi_host {
        order.push(NewSessionFocus::Host);
    }
    order.push(NewSessionFocus::Name);
    if env_row_count > 0 {
        order.push(NewSessionFocus::EnvRow(0));
    }
    order.extend([
        NewSessionFocus::EnvKey,
        NewSessionFocus::EnvValue,
        NewSessionFocus::Ok,
        NewSessionFocus::Cancel,
    ]);
    order
}

fn normalized_new_session_focus(focus: NewSessionFocus) -> NewSessionFocus {
    match focus {
        NewSessionFocus::EnvRow(_) => NewSessionFocus::EnvRow(0),
        _ => focus,
    }
}

fn previous_new_session_env_row_focus(
    focus: NewSessionFocus,
    env_row_count: usize,
) -> Option<NewSessionFocus> {
    match focus {
        NewSessionFocus::EnvRow(index) => Some(NewSessionFocus::EnvRow(index.saturating_sub(1))),
        NewSessionFocus::EnvKey | NewSessionFocus::EnvValue if env_row_count > 0 => {
            Some(NewSessionFocus::EnvRow(env_row_count - 1))
        }
        _ => None,
    }
}

fn next_new_session_env_row_focus(
    focus: NewSessionFocus,
    env_row_count: usize,
) -> Option<NewSessionFocus> {
    match focus {
        NewSessionFocus::EnvRow(index) if index + 1 < env_row_count => {
            Some(NewSessionFocus::EnvRow(index + 1))
        }
        NewSessionFocus::EnvRow(_) => Some(NewSessionFocus::EnvKey),
        _ => None,
    }
}

fn handle_session_key_values_modal_key(
    key: KeyEvent,
    session: SelectedSession,
    ui: &mut SessionKeyValueModalUi,
) -> ModalAction {
    if let Some(action) = handle_session_key_values_submit(&key, &session, ui) {
        return action;
    }
    if handle_session_key_values_navigation(&key, &ui.rows, &mut ui.focus) {
        return ModalAction::None;
    }
    if let Some(action) = handle_session_key_values_row_action(&key, ui) {
        return action;
    }
    handle_session_key_values_edit(&key, &mut ui.key_input, &mut ui.value_input, ui.focus);
    ModalAction::None
}

fn handle_session_key_values_submit(
    key: &KeyEvent,
    session: &SelectedSession,
    ui: &SessionKeyValueModalUi,
) -> Option<ModalAction> {
    match key.code {
        KeyCode::Esc => Some(ModalAction::Close),
        KeyCode::Enter => match ui.focus {
            SessionKeyValueFocus::Key | SessionKeyValueFocus::Value => {
                Some(ModalAction::StageKeyValue {
                    kind: ui.kind,
                    key: ui.key_input.trim().to_string(),
                    value: ui.value_input.clone(),
                })
            }
            SessionKeyValueFocus::Ok => Some(ModalAction::ApplyKeyValues {
                session: session.clone(),
                ui: ui.clone(),
            }),
            SessionKeyValueFocus::Cancel => Some(ModalAction::Close),
            _ => Some(ModalAction::None),
        },
        _ => None,
    }
}

fn handle_session_key_values_navigation(
    key: &KeyEvent,
    rows: &[SessionKeyValueRow],
    focus: &mut SessionKeyValueFocus,
) -> bool {
    match key.code {
        KeyCode::Tab | KeyCode::Char('\t') => {
            *focus = next_session_key_values_focus(*focus);
            true
        }
        KeyCode::BackTab => {
            *focus = previous_session_key_values_focus(*focus);
            true
        }
        KeyCode::Up => {
            move_session_key_value_focus_up(rows, focus);
            true
        }
        KeyCode::Char('u')
            if key.modifiers == KeyModifiers::NONE
                && matches!(*focus, SessionKeyValueFocus::Row(_)) =>
        {
            move_session_key_value_focus_up(rows, focus);
            true
        }
        KeyCode::Down => {
            move_session_key_value_focus_down(rows, focus);
            true
        }
        KeyCode::Char('b')
            if key.modifiers == KeyModifiers::NONE
                && matches!(*focus, SessionKeyValueFocus::Row(_)) =>
        {
            move_session_key_value_focus_down(rows, focus);
            true
        }
        _ => false,
    }
}

fn move_session_key_value_focus_up(rows: &[SessionKeyValueRow], focus: &mut SessionKeyValueFocus) {
    *focus = match *focus {
        SessionKeyValueFocus::Row(index) => SessionKeyValueFocus::Row(index.saturating_sub(1)),
        _ if !rows.is_empty() => SessionKeyValueFocus::Row(rows.len() - 1),
        _ => *focus,
    };
}

fn move_session_key_value_focus_down(
    rows: &[SessionKeyValueRow],
    focus: &mut SessionKeyValueFocus,
) {
    *focus = match *focus {
        SessionKeyValueFocus::Row(index) => {
            SessionKeyValueFocus::Row(min(index + 1, rows.len().saturating_sub(1)))
        }
        _ => *focus,
    };
}

fn handle_session_key_values_row_action(
    key: &KeyEvent,
    ui: &mut SessionKeyValueModalUi,
) -> Option<ModalAction> {
    let SessionKeyValueFocus::Row(index) = ui.focus else {
        return None;
    };
    match key.code {
        KeyCode::Char('c') if ui.kind.supports_checked_row() => ui.rows.get(index).map(|row| {
            let key = if ui.selected_key.as_deref() == Some(row.key.as_str()) {
                None
            } else {
                Some(row.key.clone())
            };
            ModalAction::StageSelectTag { key, index }
        }),
        KeyCode::Char('x') => ui
            .rows
            .get(index)
            .map(|row| ModalAction::StageUnsetKeyValue {
                kind: ui.kind,
                key: row.key.clone(),
                index,
                selected_key: ui.selected_key.clone(),
            }),
        KeyCode::Char('m') => {
            if let Some(row) = ui.rows.get(index) {
                ui.key_input = row.key.clone();
                ui.value_input = row.value.clone();
                ui.focus = SessionKeyValueFocus::Value;
            }
            Some(ModalAction::None)
        }
        _ => None,
    }
}

fn handle_session_key_values_edit(
    key: &KeyEvent,
    key_input: &mut String,
    value_input: &mut String,
    focus: SessionKeyValueFocus,
) -> bool {
    match focus {
        SessionKeyValueFocus::Key => edit_text_field(key, key_input),
        SessionKeyValueFocus::Value => edit_text_field(key, value_input),
        _ => false,
    }
}

fn next_session_key_values_focus(focus: SessionKeyValueFocus) -> SessionKeyValueFocus {
    match focus {
        SessionKeyValueFocus::Row(_) => SessionKeyValueFocus::Key,
        _ => next_focus(focus, &SESSION_KEY_VALUES_FOCUS_ORDER),
    }
}

fn previous_session_key_values_focus(focus: SessionKeyValueFocus) -> SessionKeyValueFocus {
    match focus {
        SessionKeyValueFocus::Row(_) => SessionKeyValueFocus::Cancel,
        _ => previous_focus(focus, &SESSION_KEY_VALUES_FOCUS_ORDER),
    }
}

const SESSION_KEY_VALUES_FOCUS_ORDER: [SessionKeyValueFocus; 4] = [
    SessionKeyValueFocus::Key,
    SessionKeyValueFocus::Value,
    SessionKeyValueFocus::Ok,
    SessionKeyValueFocus::Cancel,
];

async fn create_session_from_modal(
    fleet: &HostFleet,
    app: &mut AppState,
    host_id: Option<HostId>,
    input: String,
    env_rows: Vec<SessionKeyValueRow>,
) -> Result<()> {
    let Some(target_host) = host_id
        .as_ref()
        .and_then(|id| fleet.entry(id))
        .or_else(|| host_for_new_session(fleet, app))
    else {
        app.status = StatusBanner::error("no host available for new session");
        return Ok(());
    };
    let name = input.trim();
    if name.is_empty() {
        app.status = StatusBanner::info("new session name is empty");
        return Ok(());
    }
    let initial_environment = match session_env_vars_from_rows(&env_rows) {
        Ok(vars) => vars,
        Err(err) => {
            app.status = StatusBanner::error(format!("invalid env var: {err}"));
            return Ok(());
        }
    };
    let opts = CreateSessionOptions {
        initial_environment,
        ..Default::default()
    };
    match target_host.handle.create_session(name, &opts).await {
        Ok(_) => {
            let status = if fleet.is_multi() {
                format!("created session {name} on {}", target_host.label)
            } else {
                format!("created session {name}")
            };
            refresh_sessions(fleet, app, true).await?;
            app.status = StatusBanner::info(status);
        }
        Err(err) => {
            app.status = StatusBanner::error(format!("create failed: {err}"));
        }
    }
    Ok(())
}

fn stage_new_session_env(app: &mut AppState, key: String, value: String) {
    if key.is_empty() {
        app.status = StatusBanner::info("env var key is empty");
        return;
    }
    if value.is_empty() {
        app.status = StatusBanner::info("env var value is empty");
        return;
    }
    if let Err(err) = SessionEnvVar::new(key.clone(), value.clone()) {
        app.status = StatusBanner::error(format!("invalid env var: {err}"));
        return;
    }
    let Some(ModalState::NewSession { ui }) = app.modal.as_mut() else {
        return;
    };
    if let Some(row) = ui.env_rows.iter_mut().find(|row| row.key == key) {
        row.value = value;
    } else {
        ui.env_rows.push(SessionKeyValueRow {
            key: key.clone(),
            value,
        });
    }
    ui.env_rows.sort_by(|left, right| {
        left.key
            .cmp(&right.key)
            .then_with(|| left.value.cmp(&right.value))
    });
    let index = ui
        .env_rows
        .iter()
        .position(|row| row.key == key)
        .unwrap_or(0);
    ui.env_key_input.clear();
    ui.env_value_input.clear();
    set_new_session_focus(ui, NewSessionFocus::EnvRow(index));
    app.status = StatusBanner::info(format!("staged env var {key}"));
}

fn unset_new_session_env(app: &mut AppState, index: usize) {
    let Some(ModalState::NewSession { ui }) = app.modal.as_mut() else {
        return;
    };
    if index >= ui.env_rows.len() {
        return;
    }
    let removed = ui.env_rows.remove(index);
    if ui.env_rows.is_empty() {
        set_new_session_focus(ui, NewSessionFocus::EnvKey);
    } else {
        set_new_session_focus(
            ui,
            NewSessionFocus::EnvRow(min(index, ui.env_rows.len() - 1)),
        );
    }
    app.status = StatusBanner::info(format!("removed env var {}", removed.key));
}

fn session_env_vars_from_rows(rows: &[SessionKeyValueRow]) -> Result<Vec<SessionEnvVar>> {
    rows.iter()
        .map(|row| Ok(SessionEnvVar::new(row.key.clone(), row.value.clone())?))
        .collect()
}

async fn kill_session_from_modal(
    fleet: &HostFleet,
    app: &mut AppState,
    session: SelectedSession,
) -> Result<()> {
    let Some(host) = fleet_host(fleet, &session.host_id) else {
        app.status =
            StatusBanner::error(format!("host {} no longer connected", session.host_label));
        return Ok(());
    };
    let target = host.target_for_session_info(session.info.clone());
    match target.kill().await {
        Ok(()) => {
            let killed_key = (session.host_id.clone(), session.id().to_string());
            let status = if fleet.is_multi() {
                format!(
                    "killed session {} on {}",
                    session.name(),
                    session.host_label
                )
            } else {
                format!("killed session {}", session.name())
            };
            refresh_sessions_excluding(fleet, app, true, killed_key).await?;
            app.status = StatusBanner::info(status);
        }
        Err(err) => {
            app.status = StatusBanner::error(format!("kill failed: {err}"));
        }
    }
    Ok(())
}

async fn rename_session_from_modal(
    fleet: &HostFleet,
    app: &mut AppState,
    session: SelectedSession,
    input: String,
) -> Result<()> {
    let new_name = input.trim();
    if new_name.is_empty() {
        app.status = StatusBanner::info("session name is empty");
        return Ok(());
    }
    if new_name == session.name() {
        app.status = StatusBanner::info("session name unchanged");
        return Ok(());
    }
    let Some(host) = fleet_host(fleet, &session.host_id) else {
        app.status =
            StatusBanner::error(format!("host {} no longer connected", session.host_label));
        return Ok(());
    };
    match host.session_by_id(session.id()).await? {
        Some(target) => match target.rename(new_name).await {
            Ok(_) => {
                let status = if fleet.is_multi() {
                    format!(
                        "renamed session {} to {new_name} on {}",
                        session.name(),
                        session.host_label
                    )
                } else {
                    format!("renamed session {} to {new_name}", session.name())
                };
                refresh_sessions(fleet, app, true).await?;
                app.status = StatusBanner::info(status);
            }
            Err(err) => {
                app.status = StatusBanner::error(format!("rename failed: {err}"));
            }
        },
        None => {
            refresh_sessions(fleet, app, true).await?;
            app.status = StatusBanner::error(format!(
                "session {} disappeared before rename",
                session.name()
            ));
        }
    }
    Ok(())
}

async fn send_keys_from_modal(
    fleet: &HostFleet,
    app: &mut AppState,
    session: SelectedSession,
    input: String,
    mode: SendKeysSubmitMode,
) -> Result<bool> {
    let (input, mode) = normalize_send_keys_input(input, mode);
    let enter_keys = if mode == SendKeysSubmitMode::ThenEnterAfterDelay {
        Some(KeySequence::parse("{Enter}").expect("Enter is a valid KeySequence literal"))
    } else {
        None
    };
    if input.is_empty() && enter_keys.is_none() {
        app.status = StatusBanner::info("keys are empty");
        return Ok(false);
    }
    let keys = if input.is_empty() {
        None
    } else {
        match KeySequence::parse(&input) {
            Ok(keys) => Some(keys),
            Err(err) => {
                app.status = StatusBanner::error(format!("invalid keys: {err}"));
                return Ok(false);
            }
        }
    };
    let Some(host) = fleet_host(fleet, &session.host_id) else {
        app.status =
            StatusBanner::error(format!("host {} no longer connected", session.host_label));
        return Ok(true);
    };
    match host.session_by_id(session.id()).await? {
        Some(target) => {
            if let Some(keys) = keys.as_ref() {
                if let Err(err) = target.send_keys(keys).await {
                    app.status = StatusBanner::error(format!("send keys failed: {err}"));
                    return Ok(true);
                }
            }
            if let Some(enter_keys) = enter_keys.as_ref() {
                tokio::time::sleep(SEND_KEYS_ENTER_DELAY).await;
                if let Err(err) = target.send_keys(enter_keys).await {
                    app.status = StatusBanner::error(format!("send Enter failed: {err}"));
                    return Ok(true);
                }
            }
            app.status = StatusBanner::info(if fleet.is_multi() {
                format!("sent keys to {} on {}", session.name(), session.host_label)
            } else {
                format!("sent keys to {}", session.name())
            });
            if app.detail.source.mode() == DetailMode::Sample {
                refresh_detail(fleet, app, true).await?;
            }
        }
        None => {
            refresh_sessions(fleet, app, true).await?;
            app.status = StatusBanner::error(format!(
                "session {} disappeared before keys were sent",
                session.name()
            ));
        }
    }
    Ok(true)
}

fn normalize_send_keys_input(
    input: String,
    mode: SendKeysSubmitMode,
) -> (String, SendKeysSubmitMode) {
    if let Some(input) = input.strip_suffix(SEND_KEYS_THEN_ENTER_SUFFIX) {
        (input.to_string(), SendKeysSubmitMode::ThenEnterAfterDelay)
    } else {
        (input, mode)
    }
}

async fn open_session_key_values_modal(
    fleet: &HostFleet,
    app: &mut AppState,
    selected: SelectedSession,
    kind: SessionKeyValueKind,
) -> Result<()> {
    match load_key_value_state(fleet, &selected.host_id, selected.id(), kind).await? {
        Some(state) => {
            let focus = initial_session_key_values_focus(kind, &state.rows);
            let rows = state.rows;
            let selected_key = state.selected_key;
            app.modal = Some(ModalState::SessionKeyValues {
                session: selected,
                ui: SessionKeyValueModalUi {
                    kind,
                    original_rows: rows.clone(),
                    rows,
                    original_selected_key: selected_key.clone(),
                    selected_key,
                    key_input: String::new(),
                    value_input: String::new(),
                    focus,
                },
            });
        }
        None => {
            refresh_sessions(fleet, app, true).await?;
            app.status = StatusBanner::error(format!(
                "session {} disappeared before {} edit",
                selected.name(),
                kind.noun()
            ));
        }
    }
    Ok(())
}

fn stage_key_value_in_modal(
    app: &mut AppState,
    kind: SessionKeyValueKind,
    key: String,
    value: String,
) {
    if kind != SessionKeyValueKind::Tags {
        app.status = StatusBanner::error("session environment is set during session creation");
        return;
    }
    if key.is_empty() {
        app.status = StatusBanner::info(format!("{} key is empty", kind.noun()));
        return;
    }
    if kind.supports_checked_row() && is_reserved_tag_key(&key) {
        app.status = StatusBanner::info("tag key is reserved");
        return;
    }
    if value.is_empty() {
        app.status = StatusBanner::info(format!("{} value is empty", kind.noun()));
        return;
    }

    let status_text = {
        let Some(ModalState::SessionKeyValues { session, ui }) = app.modal.as_mut() else {
            return;
        };
        if ui.kind != kind {
            return;
        }
        let index = upsert_session_key_value_row(ui, key.clone(), value);
        ui.key_input.clear();
        ui.value_input.clear();
        ui.focus = SessionKeyValueFocus::Row(index);
        format!("staged {} {key} on {}", kind.noun(), session.name())
    };
    app.status = StatusBanner::info(status_text);
}

fn stage_unset_key_value_in_modal(
    app: &mut AppState,
    kind: SessionKeyValueKind,
    key: String,
    index: usize,
    selected_key: Option<String>,
) {
    if kind != SessionKeyValueKind::Tags {
        app.status = StatusBanner::error("session environment is set during session creation");
        return;
    }
    let status_text = {
        let Some(ModalState::SessionKeyValues { session, ui }) = app.modal.as_mut() else {
            return;
        };
        if ui.kind != kind {
            return;
        }
        if let Some(position) = ui.rows.iter().position(|row| row.key == key) {
            ui.rows.remove(position);
        }
        if selected_key.as_deref() == Some(key.as_str()) || ui.selected_key.as_deref() == Some(&key)
        {
            ui.selected_key = None;
        }
        ui.focus = if ui.rows.is_empty() {
            SessionKeyValueFocus::Key
        } else {
            SessionKeyValueFocus::Row(min(index, ui.rows.len() - 1))
        };
        format!("staged delete {} {key} on {}", kind.noun(), session.name())
    };
    app.status = StatusBanner::info(status_text);
}

fn stage_select_tag_in_modal(app: &mut AppState, key: Option<String>, index: usize) {
    let status_text = {
        let Some(ModalState::SessionKeyValues { session, ui }) = app.modal.as_mut() else {
            return;
        };
        if ui.kind != SessionKeyValueKind::Tags {
            return;
        }
        ui.selected_key = key.clone();
        ui.focus = if ui.rows.is_empty() {
            SessionKeyValueFocus::Key
        } else {
            SessionKeyValueFocus::Row(min(index, ui.rows.len() - 1))
        };
        match key {
            Some(key) => format!("staged selected tag {key} on {}", session.name()),
            None => format!("staged cleared selected tag on {}", session.name()),
        }
    };
    app.status = StatusBanner::info(status_text);
}

async fn apply_session_key_values_modal(
    fleet: &HostFleet,
    app: &mut AppState,
    session: SelectedSession,
    ui: SessionKeyValueModalUi,
) -> Result<()> {
    if ui.kind != SessionKeyValueKind::Tags {
        app.status = StatusBanner::error("session environment is set during session creation");
        return Ok(());
    }

    let changes = tag_modal_changes(&ui);
    if changes.is_empty() {
        app.status = StatusBanner::info(format!("no tag changes on {}", session.name()));
        return Ok(());
    }

    let Some(host) = fleet_host(fleet, &session.host_id) else {
        app.status =
            StatusBanner::error(format!("host {} no longer connected", session.host_label));
        return Ok(());
    };
    let Some(target) = host.session_by_id(session.id()).await? else {
        refresh_sessions(fleet, app, true).await?;
        app.status = StatusBanner::error(format!(
            "session {} disappeared before tag changes applied",
            session.name()
        ));
        return Ok(());
    };

    let tags = target.tags(TAG_PREFIX).await?;
    for (key, value) in &changes.set_rows {
        if let Err(err) = tags.set(key, value).await {
            app.status = StatusBanner::error(format!("apply tag changes failed: {err}"));
            return Ok(());
        }
    }
    for key in &changes.unset_keys {
        if let Err(err) = tags.unset(key).await {
            app.status = StatusBanner::error(format!("apply tag changes failed: {err}"));
            return Ok(());
        }
    }
    if ui.selected_key != ui.original_selected_key {
        let result = match ui.selected_key.as_deref() {
            Some(key) => tags.set(SELECTED_TAG_KEY_OPTION, key).await,
            None => tags.unset(SELECTED_TAG_KEY_OPTION).await,
        };
        if let Err(err) = result {
            app.status = StatusBanner::error(format!("apply tag selection failed: {err}"));
            return Ok(());
        }
    }

    app.status = StatusBanner::info(if fleet.is_multi() {
        format!(
            "applied tag changes on {} at {}",
            session.name(),
            session.host_label
        )
    } else {
        format!("applied tag changes on {}", session.name())
    });
    refresh_sessions_quiet(fleet, app, false).await?;
    Ok(())
}

fn upsert_session_key_value_row(
    ui: &mut SessionKeyValueModalUi,
    key: String,
    value: String,
) -> usize {
    if let Some(row) = ui.rows.iter_mut().find(|row| row.key == key) {
        row.value = value;
    } else {
        ui.rows.push(SessionKeyValueRow {
            key: key.clone(),
            value,
        });
    }
    sort_session_key_value_rows(&mut ui.rows);
    ui.rows.iter().position(|row| row.key == key).unwrap_or(0)
}

fn sort_session_key_value_rows(rows: &mut [SessionKeyValueRow]) {
    rows.sort_by(|left, right| {
        left.key
            .cmp(&right.key)
            .then_with(|| left.value.cmp(&right.value))
    });
}

#[derive(Debug, Default, PartialEq, Eq)]
struct TagModalChanges {
    set_rows: Vec<(String, String)>,
    unset_keys: Vec<String>,
    selected_key_changed: bool,
}

impl TagModalChanges {
    fn is_empty(&self) -> bool {
        self.set_rows.is_empty() && self.unset_keys.is_empty() && !self.selected_key_changed
    }
}

fn tag_modal_changes(ui: &SessionKeyValueModalUi) -> TagModalChanges {
    let original = session_key_value_row_map(&ui.original_rows);
    let current = session_key_value_row_map(&ui.rows);
    let mut set_rows = current
        .iter()
        .filter_map(|(key, value)| {
            if original.get(key) == Some(value) {
                None
            } else {
                Some((key.clone(), value.clone()))
            }
        })
        .collect::<Vec<_>>();
    set_rows.sort_by(|left, right| left.0.cmp(&right.0));

    let mut unset_keys = original
        .keys()
        .filter(|key| !current.contains_key(*key))
        .cloned()
        .collect::<Vec<_>>();
    unset_keys.sort();

    TagModalChanges {
        set_rows,
        unset_keys,
        selected_key_changed: ui.selected_key != ui.original_selected_key,
    }
}

fn session_key_value_row_map(rows: &[SessionKeyValueRow]) -> HashMap<String, String> {
    rows.iter()
        .map(|row| (row.key.clone(), row.value.clone()))
        .collect()
}

struct SessionKeyValueState {
    rows: Vec<SessionKeyValueRow>,
    selected_key: Option<String>,
}

async fn load_key_value_state(
    fleet: &HostFleet,
    host_id: &HostId,
    id: &str,
    kind: SessionKeyValueKind,
) -> Result<Option<SessionKeyValueState>> {
    let Some(host) = fleet_host(fleet, host_id) else {
        return Ok(None);
    };
    let Some(target) = host.session_by_id(id).await? else {
        return Ok(None);
    };
    match kind {
        SessionKeyValueKind::Tags => {
            let tags = target.tags(TAG_PREFIX).await?.list().await?;
            Ok(Some(tag_state_from_tags(tags)))
        }
        SessionKeyValueKind::Environment => Ok(Some(SessionKeyValueState {
            rows: Vec::new(),
            selected_key: None,
        })),
    }
}

fn tag_state_from_tags(tags: Vec<SessionTag>) -> SessionKeyValueState {
    let selected_key = selected_key_from_tags(&tags);
    let mut rows = tags
        .into_iter()
        .filter(|tag| !is_reserved_tag_key(tag.key()))
        .map(session_tag_row)
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        left.key
            .cmp(&right.key)
            .then_with(|| left.value.cmp(&right.value))
    });
    let selected_key = selected_key.filter(|key| rows.iter().any(|row| row.key == *key));
    SessionKeyValueState { rows, selected_key }
}

fn selected_key_from_tags(tags: &[SessionTag]) -> Option<String> {
    let value = tags
        .iter()
        .find(|tag| tag.key() == SELECTED_TAG_KEY_OPTION)?
        .value();
    if value.is_empty() || is_reserved_tag_key(value) {
        None
    } else {
        Some(value.to_string())
    }
}

fn is_reserved_tag_key(key: &str) -> bool {
    RESERVED_TAG_KEYS.contains(&key)
}

fn session_tag_row(tag: SessionTag) -> SessionKeyValueRow {
    SessionKeyValueRow {
        key: tag.key().to_string(),
        value: tag.value().to_string(),
    }
}

fn first_session_key_values_focus(rows: &[SessionKeyValueRow]) -> SessionKeyValueFocus {
    if rows.is_empty() {
        SessionKeyValueFocus::Key
    } else {
        SessionKeyValueFocus::Row(0)
    }
}

fn initial_session_key_values_focus(
    kind: SessionKeyValueKind,
    rows: &[SessionKeyValueRow],
) -> SessionKeyValueFocus {
    match kind {
        SessionKeyValueKind::Tags => first_session_key_values_focus(rows),
        SessionKeyValueKind::Environment => SessionKeyValueFocus::Key,
    }
}

/// Pick the host for a new-session request: the host of the highlighted row
/// in the merged list, or the only host in single-host mode, or the first
/// host as a sensible fallback.
fn host_for_new_session<'a>(fleet: &'a HostFleet, app: &AppState) -> Option<&'a HostEntry> {
    if let Some(selected) = app.selected_session() {
        if let Some(entry) = fleet.entry(&selected.host_id) {
            return Some(entry);
        }
    }
    fleet.first()
}

async fn start_monitor(fleet: &HostFleet, app: &mut AppState) -> Result<()> {
    let Some(selected) = app.selected_session() else {
        app.status = StatusBanner::info("no session selected");
        return Ok(());
    };
    let Some(host) = fleet_host(fleet, &selected.host_id) else {
        app.status =
            StatusBanner::error(format!("host {} no longer connected", selected.host_label));
        return Ok(());
    };

    stop_detail_source(app).await;
    let mut source = DetailSource::monitor();
    match source.activate(host, &selected).await {
        Ok(()) => {
            app.detail.source = source;
            app.status = StatusBanner::info(format!("monitoring {}", selected.name()));
            app.detail.lines.clear();
        }
        Err(err) => {
            app.detail.source = DetailSource::sample();
            app.status = StatusBanner::error(format!("monitor failed: {err}"));
            refresh_detail(fleet, app, true).await?;
        }
    }
    Ok(())
}

pub(crate) async fn stop_detail_source(app: &mut AppState) {
    let _ = app.detail.source.deactivate().await;
}

async fn fetch_older_detail(fleet: &HostFleet, app: &mut AppState) -> Result<()> {
    let Some(selected) = app.selected_session() else {
        return Ok(());
    };
    let Some(host) = fleet_host(fleet, &selected.host_id) else {
        return Ok(());
    };
    let older_than_lines = app.detail.lines.len();
    let text = app
        .detail
        .source
        .fetch_older(host, &selected, older_than_lines, DEFAULT_DETAIL_LINES)
        .await
        .unwrap_or_default();
    if !text.trim().is_empty() {
        let mut older = text
            .lines()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        older.append(&mut app.detail.lines);
        app.detail.lines = older;
        app.status = StatusBanner::info("fetched older scrollback");
    }
    Ok(())
}
