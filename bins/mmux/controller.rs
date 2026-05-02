use std::cmp::min;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use motlie_tmux::{CreateSessionOptions, HostHandle, SessionId, SessionInfo, SessionTag};

use crate::consts::{
    DEFAULT_DETAIL_LINES, LANDSCAPE_MAX_LEFT_PERCENT, LANDSCAPE_MIN_LEFT_PERCENT,
    MOTLIE_PLACEHOLDER, PORTRAIT_MAX_TOP_PERCENT, PORTRAIT_MIN_TOP_PERCENT,
};
use crate::detail::{DetailMode, DetailSource, SessionDetailSource};
use crate::model::{
    ActivityTracker, AppState, Button, Focus, HostEntry, HostFleet, HostId, LayoutMode, ModalState,
    SelectedSession, SessionRow, SessionSelectedTag, SessionSortMode, SessionTagRow,
    SessionTagsFocus, SessionTagsModalUi, StatusBanner,
};

const TAG_PREFIX: &str = "mmux";
const SELECTED_TAG_KEY_OPTION: &str = "__selected-key";
const RESERVED_TAG_KEYS: &[&str] = &[SELECTED_TAG_KEY_OPTION];

pub(crate) async fn load_motd(host: &HostHandle) -> (String, bool) {
    load_motd_from(host, Path::new("/etc/motd")).await
}

pub(crate) async fn load_motd_from(host: &HostHandle, path: &Path) -> (String, bool) {
    match host.read_text_file(path, 64 * 1024).await {
        Ok(text) if !text.trim().is_empty() => (text.trim_end().to_string(), false),
        _ => (MOTLIE_PLACEHOLDER.to_string(), true),
    }
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
pub(crate) async fn fetch_fleet_rows(
    fleet: &HostFleet,
    tracker: &mut ActivityTracker,
) -> (Vec<SessionRow>, Vec<String>) {
    let listings = futures::future::join_all(
        fleet
            .entries
            .iter()
            .map(|entry| async move { (entry, entry.handle.list_sessions().await) }),
    )
    .await;

    // Capture local_now once per refresh so all rows in this tick agree on
    // the operator-side observation time.
    let local_now = local_epoch_seconds();
    let mut rows = Vec::new();
    let mut failures = Vec::new();
    let mut keep_keys: HashSet<(HostId, String)> = HashSet::new();
    for (entry, listing) in listings {
        match listing {
            Ok(sessions) => {
                let selected_tags = load_selected_session_tags(&entry.handle, &sessions).await;
                for session in sessions {
                    let key = (entry.id.clone(), session.id.as_str().to_string());
                    let activity_observed_at_local = tracker.observe(
                        &entry.id,
                        session.id.as_str(),
                        session.activity,
                        local_now,
                    );
                    let selected_tag = selected_tags.get(&session.id).cloned();
                    keep_keys.insert(key);
                    rows.push(SessionRow {
                        host_id: entry.id.clone(),
                        host_label: entry.label.clone(),
                        local_now,
                        activity_observed_at_local,
                        session,
                        selected_tag,
                    });
                }
            }
            Err(err) => {
                failures.push(format!("{}: {err}", entry.label));
            }
        }
    }
    tracker.retain(&keep_keys);
    (rows, failures)
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
    refresh_sessions_preserving_with_status(fleet, app, force_detail, previous, true).await
}

pub(crate) async fn refresh_sessions_quiet(
    fleet: &HostFleet,
    app: &mut AppState,
    force_detail: bool,
) -> Result<()> {
    let previous = current_selection_key(app);
    refresh_sessions_preserving_with_status(fleet, app, force_detail, previous, false).await
}

pub(crate) async fn refresh_sessions_preserving(
    fleet: &HostFleet,
    app: &mut AppState,
    force_detail: bool,
    previous: Option<(HostId, String)>,
) -> Result<()> {
    refresh_sessions_preserving_with_status(fleet, app, force_detail, previous, true).await
}

fn current_selection_key(app: &AppState) -> Option<(HostId, String)> {
    app.selected_session()
        .map(|session| (session.host_id, session.id))
}

async fn refresh_sessions_preserving_with_status(
    fleet: &HostFleet,
    app: &mut AppState,
    force_detail: bool,
    previous: Option<(HostId, String)>,
    update_status: bool,
) -> Result<()> {
    let (rows, failures) = fetch_fleet_rows(fleet, &mut app.activity_tracker).await;
    let closed_monitored = closed_monitored_session(app, &rows);
    let previous_key = previous.clone();
    app.session_list.set_rows_sorted(rows, fleet);
    app.preserve_selection(previous);
    if update_status {
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
    let selection_changed = previous_key != selected_key;
    if force_detail || selection_changed || monitor_just_closed {
        refresh_detail(fleet, app, true).await?;
    }
    Ok(())
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
        (KeyCode::Char('s'), _) if app.layout.focus == Focus::List => {
            toggle_session_sort(fleet, app).await?;
        }
        (KeyCode::Char('n'), _) => {
            app.modal = Some(ModalState::NewSession {
                input: String::new(),
                button: Button::Ok,
            });
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
                    input: selected.name.clone(),
                    session: selected,
                    button: Button::Ok,
                });
            } else {
                app.status = StatusBanner::info("no session selected");
            }
        }
        (KeyCode::Char('t'), _) => {
            if let Some(selected) = app.selected_session() {
                open_session_tags_modal(fleet, app, selected).await?;
            } else {
                app.status = StatusBanner::info("no session selected");
            }
        }
        (KeyCode::Enter, _) => {
            if let Some(selected) = app.selected_session() {
                return Ok(KeyOutcome::Select(selected));
            }
            app.status = StatusBanner::info("no session selected");
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
        (KeyCode::Char('p'), _) => app.focus_next(),
        (KeyCode::Char('l'), _) => app.toggle_layout(),
        (KeyCode::Up, _) => match app.layout.focus {
            Focus::List => {
                if app.move_selection(-1) {
                    reset_to_sample_detail(fleet, app).await?;
                }
            }
            Focus::Detail => app.scroll_detail(1),
            Focus::Motd => {}
        },
        (KeyCode::Down, _) => match app.layout.focus {
            Focus::List => {
                if app.move_selection(1) {
                    reset_to_sample_detail(fleet, app).await?;
                }
            }
            Focus::Detail => app.scroll_detail(-1),
            Focus::Motd => {}
        },
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
            Focus::Motd => {}
        },
        (KeyCode::PageDown, _) => match app.layout.focus {
            Focus::List => {
                if app.move_selection(10) {
                    reset_to_sample_detail(fleet, app).await?;
                }
            }
            Focus::Detail => app.scroll_detail(-10),
            Focus::Motd => {}
        },
        (KeyCode::Home, _) => match app.layout.focus {
            Focus::List => {
                if !app.session_list.rows.is_empty() {
                    app.session_list.selected = 0;
                    reset_to_sample_detail(fleet, app).await?;
                }
            }
            Focus::Detail => app.detail_home(),
            Focus::Motd => {}
        },
        (KeyCode::End, _) => match app.layout.focus {
            Focus::List => {
                if !app.session_list.rows.is_empty() {
                    app.session_list.selected = app.session_list.rows.len().saturating_sub(1);
                    reset_to_sample_detail(fleet, app).await?;
                }
            }
            Focus::Detail => app.detail_end(),
            Focus::Motd => {}
        },
        _ => {}
    }

    Ok(KeyOutcome::Continue)
}

fn is_resize_modifier(modifiers: KeyModifiers) -> bool {
    modifiers.intersects(KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SHIFT)
}

async fn toggle_session_sort(fleet: &HostFleet, app: &mut AppState) -> Result<()> {
    let previous = current_selection_key(app);
    let mode = app.session_list.toggle_sort_mode();
    app.session_list.resort(fleet);
    app.session_list.select_first();
    if previous != current_selection_key(app) {
        reset_to_sample_detail(fleet, app).await?;
    }
    app.status = StatusBanner::info(match mode {
        SessionSortMode::Activity => "sort: activity",
        SessionSortMode::Tag => "sort: tag",
    });
    Ok(())
}

fn is_word_left_resize(modifiers: KeyModifiers) -> bool {
    modifiers.intersects(KeyModifiers::ALT | KeyModifiers::CONTROL)
}

fn is_word_right_resize(modifiers: KeyModifiers) -> bool {
    modifiers.intersects(KeyModifiers::ALT | KeyModifiers::CONTROL)
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
    },
    KillSession {
        session: SelectedSession,
    },
    RenameSession {
        session: SelectedSession,
        input: String,
    },
    SetTag {
        session: SelectedSession,
        key: String,
        value: String,
    },
    UnsetTag {
        session: SelectedSession,
        key: String,
        index: usize,
        selected_key: Option<String>,
    },
    SelectTag {
        session: SelectedSession,
        key: Option<String>,
        index: usize,
    },
}

async fn handle_modal_key(
    fleet: &HostFleet,
    app: &mut AppState,
    key: KeyEvent,
) -> Result<KeyOutcome> {
    let action = match app.modal.as_mut() {
        Some(ModalState::NewSession { input, button }) => match key.code {
            KeyCode::Esc => ModalAction::Close,
            KeyCode::Left => {
                *button = Button::Cancel;
                ModalAction::None
            }
            KeyCode::Right => {
                *button = Button::Ok;
                ModalAction::None
            }
            KeyCode::Enter if *button == Button::Ok => ModalAction::CreateSession {
                input: input.clone(),
            },
            KeyCode::Enter => ModalAction::Close,
            KeyCode::Backspace => {
                input.pop();
                ModalAction::None
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                input.push(c);
                ModalAction::None
            }
            _ => ModalAction::None,
        },
        Some(ModalState::KillSession { session, button }) => match key.code {
            KeyCode::Esc => ModalAction::Close,
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
            KeyCode::Backspace => {
                input.pop();
                ModalAction::None
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                input.push(c);
                ModalAction::None
            }
            _ => ModalAction::None,
        },
        Some(ModalState::SessionTags { session, ui }) => {
            handle_session_tags_modal_key(key, session.clone(), ui)
        }
        Some(ModalState::Help) => match key.code {
            KeyCode::Esc | KeyCode::Enter => ModalAction::Close,
            _ => ModalAction::None,
        },
        None => ModalAction::None,
    };

    match action {
        ModalAction::None => {}
        ModalAction::Close => app.modal = None,
        ModalAction::CreateSession { input } => {
            app.modal = None;
            create_session_from_modal(fleet, app, input).await?;
        }
        ModalAction::KillSession { session } => {
            app.modal = None;
            kill_session_from_modal(fleet, app, session).await?;
        }
        ModalAction::RenameSession { session, input } => {
            app.modal = None;
            rename_session_from_modal(fleet, app, session, input).await?;
        }
        ModalAction::SetTag {
            session,
            key,
            value,
        } => {
            set_tag_from_modal(fleet, app, session, key, value).await?;
        }
        ModalAction::UnsetTag {
            session,
            key,
            index,
            selected_key,
        } => {
            unset_tag_from_modal(fleet, app, session, key, index, selected_key).await?;
        }
        ModalAction::SelectTag {
            session,
            key,
            index,
        } => {
            select_tag_from_modal(fleet, app, session, key, index).await?;
        }
    }
    Ok(KeyOutcome::Continue)
}

fn handle_session_tags_modal_key(
    key: KeyEvent,
    session: SelectedSession,
    ui: &mut SessionTagsModalUi,
) -> ModalAction {
    if let Some(action) = handle_session_tags_submit(&key, &session, ui) {
        return action;
    }
    if handle_session_tags_navigation(&key, &ui.tags, &mut ui.focus) {
        return ModalAction::None;
    }
    if let Some(action) = handle_session_tags_row_action(&key, session, ui) {
        return action;
    }
    handle_session_tags_edit(&key, &mut ui.key_input, &mut ui.value_input, ui.focus);
    ModalAction::None
}

fn handle_session_tags_submit(
    key: &KeyEvent,
    session: &SelectedSession,
    ui: &SessionTagsModalUi,
) -> Option<ModalAction> {
    match key.code {
        KeyCode::Esc => Some(ModalAction::Close),
        KeyCode::Enter => match ui.focus {
            SessionTagsFocus::Key | SessionTagsFocus::Value => Some(ModalAction::SetTag {
                session: session.clone(),
                key: ui.key_input.trim().to_string(),
                value: ui.value_input.clone(),
            }),
            SessionTagsFocus::Cancel => Some(ModalAction::Close),
            _ => Some(ModalAction::None),
        },
        _ => None,
    }
}

fn handle_session_tags_navigation(
    key: &KeyEvent,
    tags: &[SessionTagRow],
    focus: &mut SessionTagsFocus,
) -> bool {
    match key.code {
        KeyCode::Tab | KeyCode::Char('\t') => {
            *focus = next_session_tags_focus(*focus);
            true
        }
        KeyCode::BackTab => {
            *focus = previous_session_tags_focus(*focus);
            true
        }
        KeyCode::Up => {
            *focus = match *focus {
                SessionTagsFocus::TagRow(index) => {
                    SessionTagsFocus::TagRow(index.saturating_sub(1))
                }
                _ if !tags.is_empty() => SessionTagsFocus::TagRow(tags.len() - 1),
                _ => *focus,
            };
            true
        }
        KeyCode::Down => {
            *focus = match *focus {
                SessionTagsFocus::TagRow(index) => {
                    SessionTagsFocus::TagRow(min(index + 1, tags.len().saturating_sub(1)))
                }
                _ => *focus,
            };
            true
        }
        _ => false,
    }
}

fn handle_session_tags_row_action(
    key: &KeyEvent,
    session: SelectedSession,
    ui: &mut SessionTagsModalUi,
) -> Option<ModalAction> {
    let SessionTagsFocus::TagRow(index) = ui.focus else {
        return None;
    };
    match key.code {
        KeyCode::Char('c') => ui.tags.get(index).map(|tag| {
            let key = if ui.selected_key.as_deref() == Some(tag.key.as_str()) {
                None
            } else {
                Some(tag.key.clone())
            };
            ModalAction::SelectTag {
                session,
                key,
                index,
            }
        }),
        KeyCode::Char('x') => ui.tags.get(index).map(|tag| ModalAction::UnsetTag {
            session,
            key: tag.key.clone(),
            index,
            selected_key: ui.selected_key.clone(),
        }),
        KeyCode::Char('u') => {
            if let Some(tag) = ui.tags.get(index) {
                ui.key_input = tag.key.clone();
                ui.value_input = tag.value.clone();
                ui.focus = SessionTagsFocus::Value;
            }
            Some(ModalAction::None)
        }
        _ => None,
    }
}

fn handle_session_tags_edit(
    key: &KeyEvent,
    key_input: &mut String,
    value_input: &mut String,
    focus: SessionTagsFocus,
) -> bool {
    match key.code {
        KeyCode::Backspace => {
            match focus {
                SessionTagsFocus::Key => {
                    key_input.pop();
                }
                SessionTagsFocus::Value => {
                    value_input.pop();
                }
                _ => {}
            }
            true
        }
        KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
            match focus {
                SessionTagsFocus::Key => key_input.push(c),
                SessionTagsFocus::Value => value_input.push(c),
                _ => {}
            }
            true
        }
        _ => false,
    }
}

fn next_session_tags_focus(focus: SessionTagsFocus) -> SessionTagsFocus {
    match focus {
        SessionTagsFocus::TagRow(_) => SessionTagsFocus::Key,
        SessionTagsFocus::Key => SessionTagsFocus::Value,
        SessionTagsFocus::Value => SessionTagsFocus::Cancel,
        SessionTagsFocus::Cancel => SessionTagsFocus::Key,
    }
}

fn previous_session_tags_focus(focus: SessionTagsFocus) -> SessionTagsFocus {
    match focus {
        SessionTagsFocus::TagRow(_) => SessionTagsFocus::Cancel,
        SessionTagsFocus::Key => SessionTagsFocus::Cancel,
        SessionTagsFocus::Value => SessionTagsFocus::Key,
        SessionTagsFocus::Cancel => SessionTagsFocus::Value,
    }
}

async fn create_session_from_modal(
    fleet: &HostFleet,
    app: &mut AppState,
    input: String,
) -> Result<()> {
    let Some(target_host) = host_for_new_session(fleet, app) else {
        app.status = StatusBanner::error("no host available for new session");
        return Ok(());
    };
    let name = input.trim();
    if name.is_empty() {
        app.status = StatusBanner::info("new session name is empty");
        return Ok(());
    }
    match target_host
        .handle
        .create_session(name, &CreateSessionOptions::default())
        .await
    {
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
    match host.session_by_id(&session.id).await? {
        Some(target) => match target.kill().await {
            Ok(()) => {
                let status = if fleet.is_multi() {
                    format!("killed session {} on {}", session.name, session.host_label)
                } else {
                    format!("killed session {}", session.name)
                };
                refresh_sessions(fleet, app, true).await?;
                app.status = StatusBanner::info(status);
            }
            Err(err) => {
                app.status = StatusBanner::error(format!("kill failed: {err}"));
            }
        },
        None => {
            refresh_sessions(fleet, app, true).await?;
            app.status =
                StatusBanner::error(format!("session {} disappeared before kill", session.name));
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
    if new_name == session.name {
        app.status = StatusBanner::info("session name unchanged");
        return Ok(());
    }
    let Some(host) = fleet_host(fleet, &session.host_id) else {
        app.status =
            StatusBanner::error(format!("host {} no longer connected", session.host_label));
        return Ok(());
    };
    match host.session_by_id(&session.id).await? {
        Some(target) => match target.rename(new_name).await {
            Ok(_) => {
                let status = if fleet.is_multi() {
                    format!(
                        "renamed session {} to {new_name} on {}",
                        session.name, session.host_label
                    )
                } else {
                    format!("renamed session {} to {new_name}", session.name)
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
                session.name
            ));
        }
    }
    Ok(())
}

async fn open_session_tags_modal(
    fleet: &HostFleet,
    app: &mut AppState,
    selected: SelectedSession,
) -> Result<()> {
    match load_tag_state(fleet, &selected.host_id, &selected.id).await? {
        Some(state) => {
            let focus = first_session_tags_focus(&state.rows);
            app.modal = Some(ModalState::SessionTags {
                session: selected,
                ui: SessionTagsModalUi {
                    tags: state.rows,
                    selected_key: state.selected_key,
                    key_input: String::new(),
                    value_input: String::new(),
                    focus,
                },
            });
        }
        None => {
            refresh_sessions(fleet, app, true).await?;
            app.status = StatusBanner::error(format!(
                "session {} disappeared before tag edit",
                selected.name
            ));
        }
    }
    Ok(())
}

async fn set_tag_from_modal(
    fleet: &HostFleet,
    app: &mut AppState,
    session: SelectedSession,
    key: String,
    value: String,
) -> Result<()> {
    if key.is_empty() {
        app.status = StatusBanner::info("tag key is empty");
        return Ok(());
    }
    if is_reserved_tag_key(&key) {
        app.status = StatusBanner::info("tag key is reserved");
        return Ok(());
    }
    if value.is_empty() {
        app.status = StatusBanner::info("tag value is empty");
        return Ok(());
    }
    let Some(host) = fleet_host(fleet, &session.host_id) else {
        app.status =
            StatusBanner::error(format!("host {} no longer connected", session.host_label));
        return Ok(());
    };
    match host.session_by_id(&session.id).await? {
        Some(target) => match target.set_tag(TAG_PREFIX, &key, &value).await {
            Ok(()) => {
                app.status = StatusBanner::info(if fleet.is_multi() {
                    format!(
                        "set tag {key} on {} at {}",
                        session.name, session.host_label
                    )
                } else {
                    format!("set tag {key} on {}", session.name)
                });
                reload_session_tags_modal(fleet, app, session, Some(key), None).await?;
                refresh_sessions_quiet(fleet, app, false).await?;
            }
            Err(err) => {
                app.status = StatusBanner::error(format!("set tag failed: {err}"));
            }
        },
        None => {
            app.modal = None;
            refresh_sessions(fleet, app, true).await?;
            app.status = StatusBanner::error(format!(
                "session {} disappeared before tag edit",
                session.name
            ));
        }
    }
    Ok(())
}

async fn unset_tag_from_modal(
    fleet: &HostFleet,
    app: &mut AppState,
    session: SelectedSession,
    key: String,
    index: usize,
    selected_key: Option<String>,
) -> Result<()> {
    let Some(host) = fleet_host(fleet, &session.host_id) else {
        app.status =
            StatusBanner::error(format!("host {} no longer connected", session.host_label));
        return Ok(());
    };
    match host.session_by_id(&session.id).await? {
        Some(target) => match target.unset_tag(TAG_PREFIX, &key).await {
            Ok(()) => {
                if selected_key.as_deref() == Some(key.as_str()) {
                    if let Err(err) = target.unset_tag(TAG_PREFIX, SELECTED_TAG_KEY_OPTION).await {
                        app.status = StatusBanner::error(format!(
                            "deleted tag {key}, but clearing selected tag failed: {err}"
                        ));
                        reload_session_tags_modal(fleet, app, session, None, Some(index)).await?;
                        refresh_sessions_quiet(fleet, app, false).await?;
                        return Ok(());
                    }
                }
                app.status = StatusBanner::info(if fleet.is_multi() {
                    format!(
                        "deleted tag {key} on {} at {}",
                        session.name, session.host_label
                    )
                } else {
                    format!("deleted tag {key} on {}", session.name)
                });
                reload_session_tags_modal(fleet, app, session, None, Some(index)).await?;
                refresh_sessions_quiet(fleet, app, false).await?;
            }
            Err(err) => {
                app.status = StatusBanner::error(format!("delete tag failed: {err}"));
            }
        },
        None => {
            app.modal = None;
            refresh_sessions(fleet, app, true).await?;
            app.status = StatusBanner::error(format!(
                "session {} disappeared before tag delete",
                session.name
            ));
        }
    }
    Ok(())
}

async fn select_tag_from_modal(
    fleet: &HostFleet,
    app: &mut AppState,
    session: SelectedSession,
    key: Option<String>,
    index: usize,
) -> Result<()> {
    let Some(host) = fleet_host(fleet, &session.host_id) else {
        app.status =
            StatusBanner::error(format!("host {} no longer connected", session.host_label));
        return Ok(());
    };
    match host.session_by_id(&session.id).await? {
        Some(target) => {
            let result = match key.as_deref() {
                Some(key) => {
                    target
                        .set_tag(TAG_PREFIX, SELECTED_TAG_KEY_OPTION, key)
                        .await
                }
                None => target.unset_tag(TAG_PREFIX, SELECTED_TAG_KEY_OPTION).await,
            };
            match result {
                Ok(()) => {
                    if let Some(key) = key.as_deref() {
                        app.status = StatusBanner::info(if fleet.is_multi() {
                            format!(
                                "selected tag {key} on {} at {}",
                                session.name, session.host_label
                            )
                        } else {
                            format!("selected tag {key} on {}", session.name)
                        });
                    } else {
                        app.status = StatusBanner::info(if fleet.is_multi() {
                            format!(
                                "cleared selected tag on {} at {}",
                                session.name, session.host_label
                            )
                        } else {
                            format!("cleared selected tag on {}", session.name)
                        });
                    }
                    reload_session_tags_modal(fleet, app, session, None, Some(index)).await?;
                    refresh_sessions_quiet(fleet, app, false).await?;
                }
                Err(err) => {
                    app.status = StatusBanner::error(if fleet.is_multi() {
                        format!(
                            "select tag failed on {} at {}: {err}",
                            session.name, session.host_label
                        )
                    } else {
                        format!("select tag failed on {}: {err}", session.name)
                    });
                }
            }
        }
        None => {
            app.modal = None;
            refresh_sessions(fleet, app, true).await?;
            app.status = StatusBanner::error(format!(
                "session {} disappeared before tag selection",
                session.name
            ));
        }
    }
    Ok(())
}

async fn reload_session_tags_modal(
    fleet: &HostFleet,
    app: &mut AppState,
    session: SelectedSession,
    preferred_key: Option<String>,
    preferred_index: Option<usize>,
) -> Result<()> {
    match load_tag_state(fleet, &session.host_id, &session.id).await? {
        Some(state) => {
            let tags = state.rows;
            let focus = if let Some(key) = preferred_key {
                tags.iter()
                    .position(|tag| tag.key == key)
                    .map(SessionTagsFocus::TagRow)
                    .unwrap_or_else(|| first_session_tags_focus(&tags))
            } else if let Some(index) = preferred_index {
                if tags.is_empty() {
                    SessionTagsFocus::Key
                } else {
                    SessionTagsFocus::TagRow(min(index, tags.len() - 1))
                }
            } else {
                first_session_tags_focus(&tags)
            };
            app.modal = Some(ModalState::SessionTags {
                session,
                ui: SessionTagsModalUi {
                    tags,
                    selected_key: state.selected_key,
                    key_input: String::new(),
                    value_input: String::new(),
                    focus,
                },
            });
        }
        None => {
            app.modal = None;
            refresh_sessions(fleet, app, true).await?;
            app.status = StatusBanner::error(format!(
                "session {} disappeared before tag reload",
                session.name
            ));
        }
    }
    Ok(())
}

struct SessionTagState {
    rows: Vec<SessionTagRow>,
    selected_key: Option<String>,
}

async fn load_tag_state(
    fleet: &HostFleet,
    host_id: &HostId,
    id: &str,
) -> Result<Option<SessionTagState>> {
    let Some(host) = fleet_host(fleet, host_id) else {
        return Ok(None);
    };
    let Some(target) = host.session_by_id(id).await? else {
        return Ok(None);
    };
    let tags = target.tags(TAG_PREFIX).await?.list().await?;
    Ok(Some(tag_state_from_tags(tags)))
}

fn tag_state_from_tags(tags: Vec<SessionTag>) -> SessionTagState {
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
    SessionTagState { rows, selected_key }
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

fn session_tag_row(tag: SessionTag) -> SessionTagRow {
    SessionTagRow {
        key: tag.key().to_string(),
        value: tag.value().to_string(),
    }
}

fn first_session_tags_focus(tags: &[SessionTagRow]) -> SessionTagsFocus {
    if tags.is_empty() {
        SessionTagsFocus::Key
    } else {
        SessionTagsFocus::TagRow(0)
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
            app.status = StatusBanner::info(format!("monitoring {}", selected.name));
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
