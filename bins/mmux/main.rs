mod cli;
mod consts;
mod controller;
mod detail;
mod forcecommand;
mod model;
mod render;
mod target_host;
mod terminal;

#[cfg(test)]
mod tests;

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use crossterm::event::{self, Event, KeyEventKind};
use ratatui::style::Color;
use tokio::sync::mpsc::{self, error::TrySendError, Receiver, Sender, UnboundedReceiver};
use tokio::time::sleep;

use cli::{select_layout, Cli};
use consts::{
    mmux_attach_status_style, mmux_attach_window_style, MMUX_ATTACH_STATUS_LEFT,
    MMUX_ATTACH_STATUS_LEFT_LENGTH,
};
use controller::{
    apply_host_refresh, fetch_host_refresh, handle_key, refresh_detail, stop_detail_source,
    HostRefreshResult, KeyOutcome, RefreshApplyOptions,
};
use forcecommand::maybe_run_forcecommand_bypass;
use model::{
    AppState, HostConnectFailure, HostEntry, HostFleet, HostId, LayoutMode, RetainedUiState,
    SelectedSession, StatusBanner,
};
use motlie_tmux::{
    AttachExit, SessionStatus, SessionStatusOverrides, SessionStatusSnapshot,
    SessionWindowStyleOverrides, SessionWindowStyles, SessionWindowStylesSnapshot, StatusLeft,
    StatusLeftLength, StatusStyle, Target, WindowStyle,
};
use target_host::{connect_initial_fleet, connect_ssh_spec, HostConnectSpec};
use terminal::TerminalSession;

#[derive(Debug)]
enum SelectorOutcome {
    Selected(SelectedSession),
    Cancelled,
}

struct AttachStyleSnapshot {
    status: Option<SessionStatusSnapshot>,
    window_styles: Option<SessionWindowStylesSnapshot>,
}

const HOST_CONNECT_RETRY_DELAY: Duration = Duration::from_secs(5);
const HOST_CONNECT_EVENT_BUFFER: usize = 64;

enum HostConnectEvent {
    Connected(HostEntry),
    Failed {
        id: HostId,
        failure: HostConnectFailure,
    },
}

struct PendingHostConnections {
    receiver: Receiver<HostConnectEvent>,
    tasks: Vec<tokio::task::JoinHandle<()>>,
}

#[derive(Default)]
struct HostConnectApply {
    connected: bool,
}

impl Drop for PendingHostConnections {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
    }
}

struct PendingSessionRefresh {
    receiver: UnboundedReceiver<HostRefreshResult>,
    tasks: Vec<tokio::task::JoinHandle<()>>,
    remaining: usize,
    options: RefreshApplyOptions,
}

struct SessionRefreshCompletion {
    /// True when detail refresh was intentionally suppressed during session
    /// list reconciliation. The caller extends the quiet period so the list can
    /// settle before the next routine detail capture.
    suppress_detail_refresh: bool,
}

#[tokio::main]
async fn main() {
    let code = match run().await {
        Ok(code) => code,
        Err(err) => {
            eprintln!("mmux: {err:#}");
            1
        }
    };
    std::process::exit(code);
}

async fn run() -> Result<i32> {
    if let Some(code) = maybe_run_forcecommand_bypass()? {
        return Ok(code);
    }

    let cli = Cli::parse();
    let initial_fleet = connect_initial_fleet(&cli).await?;
    let mut fleet = initial_fleet.fleet;
    let mut host_connections = start_host_connect_retries(initial_fleet.retry_specs);
    let layout = select_layout(cli.forced_layout());
    let mut ui_state = RetainedUiState::new(layout);

    loop {
        let outcome =
            run_selector_once(&mut fleet, &mut host_connections, layout, &mut ui_state).await?;
        let selected = match outcome {
            SelectorOutcome::Selected(selected) => selected,
            SelectorOutcome::Cancelled => return Ok(if cli.script { 1 } else { 0 }),
        };

        if cli.script {
            println!("{}", selected.name());
            return Ok(0);
        }

        let Some(host) = fleet.entry(&selected.host_id) else {
            eprintln!(
                "mmux: host {} no longer connected; returning to selector",
                selected.host_label
            );
            continue;
        };
        let target = match host.handle.session_by_id(selected.id()).await? {
            Some(target) => target,
            None => {
                eprintln!("mmux: selected session disappeared; returning to selector");
                continue;
            }
        };
        let exit =
            attach_current_pty_with_mmux_status(&target, fleet.host_color(&selected.host_id))
                .await?;
        let code = exit.shell_status();
        if should_reenter_after_attach(&fleet, &selected, &exit).await? {
            continue;
        }
        return Ok(code);
    }
}

async fn attach_current_pty_with_mmux_status(
    target: &Target,
    host_color: Option<Color>,
) -> Result<AttachExit> {
    let status = match target.status().await {
        Ok(status) => Some(status),
        Err(err) => {
            eprintln!("mmux: could not access tmux session status before attach: {err}");
            None
        }
    };
    let window_styles = match target.window_styles().await {
        Ok(styles) => Some(styles),
        Err(err) => {
            eprintln!("mmux: could not access tmux window styles before attach: {err}");
            None
        }
    };
    let snapshot = prepare_attach_styles(status.as_ref(), window_styles.as_ref(), host_color).await;
    let exit = target.attach_current_pty().await;
    restore_attach_styles(status.as_ref(), window_styles.as_ref(), snapshot).await;
    exit.map_err(Into::into)
}

async fn prepare_attach_styles(
    status: Option<&SessionStatus<'_>>,
    window_styles: Option<&SessionWindowStyles<'_>>,
    host_color: Option<Color>,
) -> AttachStyleSnapshot {
    AttachStyleSnapshot {
        status: match status {
            Some(status) => prepare_attach_status(status, host_color).await,
            None => None,
        },
        window_styles: match window_styles {
            Some(window_styles) => prepare_attach_window_styles(window_styles).await,
            None => None,
        },
    }
}

async fn prepare_attach_status(
    status: &SessionStatus<'_>,
    host_color: Option<Color>,
) -> Option<SessionStatusSnapshot> {
    let snapshot = match status.snapshot().await {
        Ok(snapshot) => snapshot,
        Err(err) => {
            eprintln!("mmux: could not read tmux status overrides before attach: {err}");
            return None;
        }
    };
    let style = StatusStyle::new(mmux_attach_status_style(host_color))
        .expect("mmux attach status style is a valid static tmux style");
    let left = StatusLeft::new(MMUX_ATTACH_STATUS_LEFT)
        .expect("mmux attach status-left is a valid static tmux format");
    let left_length = StatusLeftLength::new(MMUX_ATTACH_STATUS_LEFT_LENGTH)
        .expect("mmux attach status-left-length is within the supported range");
    let overrides = SessionStatusOverrides {
        style: Some(style),
        left: Some(left),
        left_length: Some(left_length),
    };
    if let Err(err) = status.apply(&overrides).await {
        eprintln!("mmux: could not set tmux status overrides before attach: {err}");
    }
    Some(snapshot)
}

async fn prepare_attach_window_styles(
    window_styles: &SessionWindowStyles<'_>,
) -> Option<SessionWindowStylesSnapshot> {
    let style = match mmux_attach_window_style() {
        Some(style) => {
            WindowStyle::new(style).expect("mmux attach window style is a valid static tmux style")
        }
        None => return None,
    };
    let snapshot = match window_styles.snapshot().await {
        Ok(snapshot) => snapshot,
        Err(err) => {
            eprintln!("mmux: could not read tmux window style overrides before attach: {err}");
            return None;
        }
    };
    let overrides = SessionWindowStyleOverrides {
        style: Some(style.clone()),
        active_style: Some(style),
    };
    if let Err(err) = window_styles.apply(&overrides).await {
        eprintln!("mmux: could not set tmux window style overrides before attach: {err}");
    }
    Some(snapshot)
}

async fn restore_attach_styles(
    status: Option<&SessionStatus<'_>>,
    window_styles: Option<&SessionWindowStyles<'_>>,
    snapshot: AttachStyleSnapshot,
) {
    if let (Some(window_styles), Some(snapshot)) = (window_styles, snapshot.window_styles) {
        restore_attach_window_styles(window_styles, Some(snapshot)).await;
    }
    if let (Some(status), Some(snapshot)) = (status, snapshot.status) {
        restore_attach_status(status, Some(snapshot)).await;
    }
}

async fn restore_attach_status(
    status: &SessionStatus<'_>,
    snapshot: Option<SessionStatusSnapshot>,
) {
    let Some(snapshot) = snapshot else {
        return;
    };
    if let Err(err) = status.restore(&snapshot).await {
        eprintln!("mmux: could not restore tmux status overrides after attach: {err}");
    }
}

async fn restore_attach_window_styles(
    window_styles: &SessionWindowStyles<'_>,
    snapshot: Option<SessionWindowStylesSnapshot>,
) {
    let Some(snapshot) = snapshot else {
        return;
    };
    if let Err(err) = window_styles.restore(&snapshot).await {
        eprintln!("mmux: could not restore tmux window style overrides after attach: {err}");
    }
}

async fn should_reenter_after_attach(
    fleet: &HostFleet,
    selected: &SelectedSession,
    exit: &AttachExit,
) -> Result<bool> {
    if exit.success() {
        return Ok(true);
    }
    let Some(entry) = fleet.entry(&selected.host_id) else {
        return Ok(false);
    };
    Ok(entry.handle.session_by_id(selected.id()).await?.is_some())
}

async fn run_selector_once(
    fleet: &mut HostFleet,
    host_connections: &mut PendingHostConnections,
    layout: LayoutMode,
    ui_state: &mut RetainedUiState,
) -> Result<SelectorOutcome> {
    let mut app = AppState::with_fleet(fleet.clone(), layout);
    ui_state.apply_to(&mut app);

    let mut terminal = TerminalSession::enter()?;
    let mut last_detail_refresh = Instant::now();
    let mut last_session_refresh = Instant::now();
    let mut pending_session_refresh = Some(start_session_refresh(
        fleet,
        RefreshApplyOptions {
            force_detail: false,
            previous: ui_state.selected_session_key(),
            update_status: true,
            excluded: None,
            allow_detail_refresh: false,
        },
    ));

    loop {
        let connect_apply = apply_finished_host_connects(fleet, &mut app, host_connections);
        if connect_apply.connected {
            abort_pending_session_refresh(&mut pending_session_refresh);
            pending_session_refresh = Some(start_session_refresh(
                fleet,
                RefreshApplyOptions {
                    force_detail: false,
                    previous: selected_session_key(&app),
                    update_status: false,
                    excluded: None,
                    allow_detail_refresh: true,
                },
            ));
            last_session_refresh = Instant::now();
        }
        if let Some(completion) =
            apply_finished_session_refresh(fleet, &mut app, &mut pending_session_refresh).await?
        {
            last_session_refresh = Instant::now();
            if completion.suppress_detail_refresh {
                last_detail_refresh = Instant::now();
            }
        }
        if pending_session_refresh.is_none()
            && last_session_refresh.elapsed() >= Duration::from_secs(1)
        {
            pending_session_refresh = Some(start_session_refresh(
                fleet,
                RefreshApplyOptions {
                    force_detail: false,
                    previous: None,
                    update_status: false,
                    excluded: None,
                    allow_detail_refresh: true,
                },
            ));
            last_session_refresh = Instant::now();
        }
        if last_detail_refresh.elapsed() >= Duration::from_millis(750) {
            refresh_detail(fleet, &mut app, false).await?;
            last_detail_refresh = Instant::now();
        }
        terminal.draw(&mut app)?;

        if event::poll(Duration::from_millis(100)).context("poll terminal event")? {
            let Event::Key(key) = event::read().context("read terminal event")? else {
                continue;
            };
            if key.kind == KeyEventKind::Release {
                continue;
            }
            clear_pending_previous_selection(&mut pending_session_refresh);
            match handle_key(fleet, &mut app, key).await? {
                KeyOutcome::Continue => {}
                KeyOutcome::Select(selected) => {
                    ui_state.update_from(&app);
                    abort_pending_session_refresh(&mut pending_session_refresh);
                    stop_detail_source(&mut app).await;
                    terminal.restore()?;
                    return Ok(SelectorOutcome::Selected(selected));
                }
                KeyOutcome::Cancel => {
                    ui_state.update_from(&app);
                    abort_pending_session_refresh(&mut pending_session_refresh);
                    stop_detail_source(&mut app).await;
                    terminal.restore()?;
                    return Ok(SelectorOutcome::Cancelled);
                }
            }
        }
    }
}

fn start_host_connect_retries(specs: Vec<HostConnectSpec>) -> PendingHostConnections {
    let (tx, receiver) = mpsc::channel(HOST_CONNECT_EVENT_BUFFER);
    let tasks = specs
        .into_iter()
        .map(|spec| {
            let tx = tx.clone();
            tokio::spawn(async move {
                retry_host_connect(spec, tx).await;
            })
        })
        .collect();
    drop(tx);
    PendingHostConnections { receiver, tasks }
}

async fn retry_host_connect(spec: HostConnectSpec, tx: Sender<HostConnectEvent>) {
    let mut last_sent_failure: Option<HostConnectFailure> = None;
    loop {
        match connect_ssh_spec(&spec)
            .await
            .with_context(|| format!("connect host '{}'", spec.uri))
        {
            Ok(entry) => {
                if let Err(err) = tx.send(HostConnectEvent::Connected(entry)).await {
                    tracing::debug!(
                        host = %spec.id,
                        error = %err,
                        "discarded successful host-connect event because selector exited"
                    );
                }
                return;
            }
            Err(err) => {
                let failure = HostConnectFailure::connect(format!("{err:#}"));
                if last_sent_failure.as_ref() != Some(&failure) {
                    match tx.try_send(HostConnectEvent::Failed {
                        id: spec.id.clone(),
                        failure: failure.clone(),
                    }) {
                        Ok(()) => last_sent_failure = Some(failure),
                        Err(TrySendError::Full(_)) => {
                            tracing::debug!(
                                host = %spec.id,
                                "host-connect event buffer full; coalescing failed retry event"
                            );
                        }
                        Err(TrySendError::Closed(_)) => {
                            tracing::debug!(
                                host = %spec.id,
                                "discarded failed host-connect event because selector exited"
                            );
                            return;
                        }
                    }
                }
                sleep(HOST_CONNECT_RETRY_DELAY).await;
            }
        }
    }
}

fn apply_finished_host_connects(
    fleet: &mut HostFleet,
    app: &mut AppState,
    connections: &mut PendingHostConnections,
) -> HostConnectApply {
    let mut changed = false;
    let mut applied = HostConnectApply::default();
    while let Ok(event) = connections.receiver.try_recv() {
        match event {
            HostConnectEvent::Connected(entry) => {
                changed |= fleet.upsert_connected(entry);
                applied.connected = true;
            }
            HostConnectEvent::Failed { id, failure } => {
                changed |= fleet.mark_host_failed(&id, failure);
            }
        }
    }
    if changed {
        app.fleet = fleet.clone();
    }
    applied
}

fn selected_session_key(app: &AppState) -> Option<(HostId, String)> {
    app.selected_session()
        .map(|session| (session.host_id.clone(), session.id().to_string()))
}

fn start_session_refresh(fleet: &HostFleet, options: RefreshApplyOptions) -> PendingSessionRefresh {
    let (tx, receiver) = mpsc::unbounded_channel();
    let tasks = fleet
        .entries
        .iter()
        .map(|entry| {
            let entry = entry.clone();
            let tx = tx.clone();
            tokio::spawn(async move {
                let result = fetch_host_refresh(&entry).await;
                let _ = tx.send(result);
            })
        })
        .collect::<Vec<_>>();
    let remaining = tasks.len();
    drop(tx);
    PendingSessionRefresh {
        receiver,
        tasks,
        remaining,
        options,
    }
}

fn abort_pending_session_refresh(pending: &mut Option<PendingSessionRefresh>) {
    if let Some(refresh) = pending.take() {
        for task in refresh.tasks {
            task.abort();
        }
    }
}

fn clear_pending_previous_selection(pending: &mut Option<PendingSessionRefresh>) {
    if let Some(refresh) = pending {
        refresh.options.previous = None;
    }
}

async fn apply_finished_session_refresh(
    fleet: &HostFleet,
    app: &mut AppState,
    pending: &mut Option<PendingSessionRefresh>,
) -> Result<Option<SessionRefreshCompletion>> {
    let Some(refresh) = pending.as_mut() else {
        return Ok(None);
    };
    let mut applied = false;
    let mut suppress_detail_refresh = false;
    while let Ok(result) = refresh.receiver.try_recv() {
        suppress_detail_refresh |= !refresh.options.allow_detail_refresh;
        apply_host_refresh(fleet, app, result, refresh.options.clone()).await?;
        refresh.remaining = refresh.remaining.saturating_sub(1);
        applied = true;
    }
    if refresh.receiver.is_closed() && refresh.remaining > 0 {
        refresh.remaining = 0;
        app.status = StatusBanner::error("session refresh task failed");
    }
    if refresh.remaining == 0 {
        pending.take();
    }
    if !applied {
        return Ok(None);
    }
    Ok(Some(SessionRefreshCompletion {
        suppress_detail_refresh,
    }))
}
