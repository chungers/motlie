use std::path::Path;

use anyhow::{Context, Result};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use motlie_tmux::{CreateSessionOptions, HostEvent, HostHandle};
use tokio::sync::mpsc;

use crate::consts::{
    DEFAULT_DETAIL_LINES, LANDSCAPE_MAX_LEFT_PERCENT, LANDSCAPE_MIN_LEFT_PERCENT,
    MOTLIE_PLACEHOLDER, PORTRAIT_MAX_TOP_PERCENT, PORTRAIT_MIN_TOP_PERCENT,
};
use crate::detail::{DetailMode, DetailSource, SessionDetailSource};
use crate::model::{
    AppState, Button, Focus, LayoutMode, ModalState, SelectedSession, StatusBanner,
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

pub(crate) async fn refresh_sessions(
    host: &HostHandle,
    app: &mut AppState,
    force_detail: bool,
) -> Result<()> {
    let previous = app.selected_session().map(|s| s.id);
    refresh_sessions_preserving(host, app, force_detail, previous).await
}

pub(crate) async fn refresh_sessions_preserving(
    host: &HostHandle,
    app: &mut AppState,
    force_detail: bool,
    previous: Option<String>,
) -> Result<()> {
    app.session_list.sessions = host.list_sessions().await.context("list tmux sessions")?;
    app.preserve_selection(previous);
    app.status = if app.session_list.sessions.is_empty() {
        StatusBanner::info("no sessions")
    } else {
        StatusBanner::info(format!("{} session(s)", app.session_list.sessions.len()))
    };
    refresh_detail(host, app, force_detail).await?;
    Ok(())
}

pub(crate) async fn refresh_detail(
    host: &HostHandle,
    app: &mut AppState,
    force: bool,
) -> Result<()> {
    if app.session_list.sessions.is_empty() {
        app.detail.lines.clear();
        return Ok(());
    }

    match app.detail.source.mode() {
        DetailMode::Sample => {
            if force || app.detail.lines.is_empty() {
                let Some(selected) = app.selected_session() else {
                    return Ok(());
                };
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
            let Some(selected) = app.selected_session() else {
                return Ok(());
            };
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

pub(crate) async fn drain_host_events(
    host: &HostHandle,
    app: &mut AppState,
    event_rx: &mut Option<mpsc::Receiver<HostEvent>>,
) -> Result<()> {
    let Some(rx) = event_rx.as_mut() else {
        return Ok(());
    };
    let mut should_refresh = false;
    let mut closed_monitored_session = None;
    loop {
        match rx.try_recv() {
            Ok(HostEvent::Disconnect { reason }) => {
                app.status = StatusBanner::error(format!("event stream degraded: {reason}"));
            }
            Ok(HostEvent::SessionClosed { id, name }) => {
                if closed_monitored_session.is_none() {
                    closed_monitored_session = stop_monitor_if_closed(app, id.as_str(), name).await;
                }
                should_refresh = true;
            }
            Ok(_) => should_refresh = true,
            Err(mpsc::error::TryRecvError::Empty) => break,
            Err(mpsc::error::TryRecvError::Disconnected) => {
                app.status =
                    StatusBanner::error("event stream disconnected; using manual refreshes");
                break;
            }
        }
    }
    if should_refresh {
        refresh_sessions(host, app, true).await?;
        if let Some(name) = closed_monitored_session {
            app.status = StatusBanner::info(format!("monitored session {name} closed"));
        }
    }
    Ok(())
}

pub(crate) async fn stop_monitor_if_closed(
    app: &mut AppState,
    id: &str,
    name: String,
) -> Option<String> {
    if app.detail.source.monitored_session_id() == Some(id) {
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
    host: &HostHandle,
    app: &mut AppState,
    key: KeyEvent,
) -> Result<KeyOutcome> {
    if app.modal.is_some() {
        return handle_modal_key(host, app, key).await;
    }

    match (key.code, key.modifiers) {
        (KeyCode::Char('c'), modifiers) if modifiers.contains(KeyModifiers::CONTROL) => {
            return Ok(KeyOutcome::Cancel);
        }
        (KeyCode::Char('q'), _) => return Ok(KeyOutcome::Cancel),
        (KeyCode::Esc, _) => app.layout.focus = Focus::List,
        (KeyCode::Char('h'), _) => app.modal = Some(ModalState::Help),
        (KeyCode::Char('m'), _) => start_monitor(host, app).await?,
        (KeyCode::Char('n'), _) => {
            app.modal = Some(ModalState::NewSession {
                input: String::new(),
                button: Button::Ok,
            });
        }
        (KeyCode::Char('k'), _) => {
            if let Some(selected) = app.selected_session() {
                app.modal = Some(ModalState::KillSession {
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
                    reset_to_sample_detail(host, app).await?;
                }
            }
            Focus::Detail => app.scroll_detail(1),
            Focus::Motd => {}
        },
        (KeyCode::Down, _) => match app.layout.focus {
            Focus::List => {
                if app.move_selection(1) {
                    reset_to_sample_detail(host, app).await?;
                }
            }
            Focus::Detail => app.scroll_detail(-1),
            Focus::Motd => {}
        },
        (KeyCode::PageUp, _) => match app.layout.focus {
            Focus::List => {
                if app.move_selection(-10) {
                    reset_to_sample_detail(host, app).await?;
                }
            }
            Focus::Detail => {
                fetch_older_detail(host, app).await?;
                app.scroll_detail(10);
            }
            Focus::Motd => {}
        },
        (KeyCode::PageDown, _) => match app.layout.focus {
            Focus::List => {
                if app.move_selection(10) {
                    reset_to_sample_detail(host, app).await?;
                }
            }
            Focus::Detail => app.scroll_detail(-10),
            Focus::Motd => {}
        },
        (KeyCode::Home, _) => match app.layout.focus {
            Focus::List => {
                if !app.session_list.sessions.is_empty() {
                    app.session_list.selected = 0;
                    reset_to_sample_detail(host, app).await?;
                }
            }
            Focus::Detail => app.detail_home(),
            Focus::Motd => {}
        },
        (KeyCode::End, _) => match app.layout.focus {
            Focus::List => {
                if !app.session_list.sessions.is_empty() {
                    app.session_list.selected = app.session_list.sessions.len().saturating_sub(1);
                    reset_to_sample_detail(host, app).await?;
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

async fn reset_to_sample_detail(host: &HostHandle, app: &mut AppState) -> Result<()> {
    stop_detail_source(app).await;
    app.detail.source = DetailSource::sample();
    refresh_detail(host, app, true).await
}

async fn handle_modal_key(
    host: &HostHandle,
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
                    let name = input.trim();
                    if name.is_empty() {
                        app.status = StatusBanner::info("new session name is empty");
                    } else {
                        match host
                            .create_session(name, &CreateSessionOptions::default())
                            .await
                        {
                            Ok(_) => {
                                app.status = StatusBanner::info(format!("created session {name}"));
                                refresh_sessions(host, app, true).await?;
                            }
                            Err(err) => {
                                app.status = StatusBanner::error(format!("create failed: {err}"));
                            }
                        }
                    }
                }
                Some(ModalState::KillSession { id, name, .. }) => {
                    match host.session_by_id(&id).await? {
                        Some(target) => match target.kill().await {
                            Ok(()) => {
                                app.status = StatusBanner::info(format!("killed session {name}"));
                                refresh_sessions(host, app, true).await?;
                            }
                            Err(err) => {
                                app.status = StatusBanner::error(format!("kill failed: {err}"));
                            }
                        },
                        None => {
                            app.status = StatusBanner::error(format!(
                                "session {name} disappeared before kill"
                            ));
                            refresh_sessions(host, app, true).await?;
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

async fn start_monitor(host: &HostHandle, app: &mut AppState) -> Result<()> {
    let Some(selected) = app.selected_session() else {
        app.status = StatusBanner::info("no session selected");
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
            refresh_detail(host, app, true).await?;
        }
    }
    Ok(())
}

pub(crate) async fn stop_detail_source(app: &mut AppState) {
    let _ = app.detail.source.deactivate().await;
}

async fn fetch_older_detail(host: &HostHandle, app: &mut AppState) -> Result<()> {
    let Some(selected) = app.selected_session() else {
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
