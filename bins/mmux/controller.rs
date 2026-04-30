use std::collections::HashSet;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use motlie_tmux::{CreateSessionOptions, HostHandle};

use crate::consts::{
    DEFAULT_DETAIL_LINES, LANDSCAPE_MAX_LEFT_PERCENT, LANDSCAPE_MIN_LEFT_PERCENT,
    MOTLIE_PLACEHOLDER, PORTRAIT_MAX_TOP_PERCENT, PORTRAIT_MIN_TOP_PERCENT,
};
use crate::detail::{DetailMode, DetailSource, SessionDetailSource};
use crate::model::{
    ActivityTracker, AppState, Button, Focus, HostEntry, HostFleet, HostId, LayoutMode, ModalState,
    SelectedSession, SessionRow, StatusBanner,
};

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
/// observations, and sort by activity descending.
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
                for session in sessions {
                    let key = (entry.id.clone(), session.id.as_str().to_string());
                    let activity_observed_at_local =
                        tracker.observe(&entry.id, session.id.as_str(), session.activity, local_now);
                    keep_keys.insert(key);
                    rows.push(SessionRow {
                        host_id: entry.id.clone(),
                        host_label: entry.label.clone(),
                        local_now,
                        activity_observed_at_local,
                        session,
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
    app.session_list.set_rows_sorted_by_activity(rows);
    app.preserve_selection(previous);
    if update_status {
        app.status = build_status(app, &failures);
    }
    let selected_key = current_selection_key(app);
    if let Some((host_id, id, name)) = closed_monitored {
        if let Some(name) = stop_monitor_if_closed(app, &host_id, &id, name).await {
            app.status = StatusBanner::info(format!("monitored session {name} closed"));
        }
    }
    refresh_detail(fleet, app, force_detail || previous_key != selected_key).await?;
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
        (KeyCode::Char('n'), _) => {
            app.modal = Some(ModalState::NewSession {
                input: String::new(),
                button: Button::Ok,
            });
        }
        (KeyCode::Char('k'), _) => {
            if let Some(selected) = app.selected_session() {
                app.modal = Some(ModalState::KillSession {
                    host_id: selected.host_id,
                    host_label: selected.host_label,
                    id: selected.id,
                    name: selected.name,
                    button: Button::Cancel,
                });
            } else {
                app.status = StatusBanner::info("no session selected");
            }
        }
        (KeyCode::Enter, _) | (KeyCode::Char('a'), _) => {
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

async fn handle_modal_key(
    fleet: &HostFleet,
    app: &mut AppState,
    key: KeyEvent,
) -> Result<KeyOutcome> {
    let mut close = false;
    let mut apply = false;
    match app.modal.as_mut() {
        Some(ModalState::NewSession { input, button }) => match key.code {
            KeyCode::Esc => close = true,
            KeyCode::Left => *button = Button::Cancel,
            KeyCode::Right => *button = Button::Ok,
            KeyCode::Enter => {
                apply = *button == Button::Ok;
                close = true;
            }
            KeyCode::Backspace => {
                input.pop();
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                input.push(c);
            }
            _ => {}
        },
        Some(ModalState::KillSession { button, .. }) => match key.code {
            KeyCode::Esc => close = true,
            KeyCode::Left => *button = Button::Cancel,
            KeyCode::Right => *button = Button::Ok,
            KeyCode::Enter => {
                apply = *button == Button::Ok;
                close = true;
            }
            _ => {}
        },
        Some(ModalState::Help) => match key.code {
            KeyCode::Esc | KeyCode::Enter => close = true,
            _ => {}
        },
        None => {}
    }

    if close {
        let modal = app.modal.take();
        if apply {
            match modal {
                Some(ModalState::NewSession { input, .. }) => {
                    let Some(target_host) = host_for_new_session(fleet, app) else {
                        app.status = StatusBanner::error("no host available for new session");
                        return Ok(KeyOutcome::Continue);
                    };
                    let name = input.trim();
                    if name.is_empty() {
                        app.status = StatusBanner::info("new session name is empty");
                    } else {
                        match target_host
                            .handle
                            .create_session(name, &CreateSessionOptions::default())
                            .await
                        {
                            Ok(_) => {
                                app.status = StatusBanner::info(if fleet.is_multi() {
                                    format!("created session {name} on {}", target_host.label)
                                } else {
                                    format!("created session {name}")
                                });
                                refresh_sessions(fleet, app, true).await?;
                            }
                            Err(err) => {
                                app.status = StatusBanner::error(format!("create failed: {err}"));
                            }
                        }
                    }
                }
                Some(ModalState::KillSession {
                    host_id,
                    host_label,
                    id,
                    name,
                    ..
                }) => {
                    let Some(host) = fleet_host(fleet, &host_id) else {
                        app.status =
                            StatusBanner::error(format!("host {host_label} no longer connected"));
                        return Ok(KeyOutcome::Continue);
                    };
                    match host.session_by_id(&id).await? {
                        Some(target) => match target.kill().await {
                            Ok(()) => {
                                app.status = StatusBanner::info(if fleet.is_multi() {
                                    format!("killed session {name} on {host_label}")
                                } else {
                                    format!("killed session {name}")
                                });
                                refresh_sessions(fleet, app, true).await?;
                            }
                            Err(err) => {
                                app.status = StatusBanner::error(format!("kill failed: {err}"));
                            }
                        },
                        None => {
                            app.status = StatusBanner::error(format!(
                                "session {name} disappeared before kill"
                            ));
                            refresh_sessions(fleet, app, true).await?;
                        }
                    }
                }
                Some(ModalState::Help) => {}
                None => {}
            }
        }
    }
    Ok(KeyOutcome::Continue)
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
