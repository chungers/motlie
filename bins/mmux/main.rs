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

use cli::{select_layout, Cli};
use consts::{MMUX_ATTACH_STATUS_LEFT, MMUX_ATTACH_STATUS_LEFT_LENGTH, MMUX_ATTACH_STATUS_STYLE};
use controller::{
    handle_key, load_motd, refresh_detail, refresh_sessions_preserving, refresh_sessions_quiet,
    stop_detail_source, KeyOutcome,
};
use forcecommand::maybe_run_forcecommand_bypass;
use model::{AppState, HostFleet, LayoutMode, RetainedUiState, SelectedSession, StatusBanner};
use motlie_tmux::{AttachExit, StatusLeft, StatusLeftLength, StatusStyle, Target};
use target_host::connect_fleet;
use terminal::TerminalSession;

#[derive(Debug)]
enum SelectorOutcome {
    Selected(SelectedSession),
    Cancelled,
}

#[derive(Debug)]
struct AttachStatusSnapshot {
    previous_style: Option<StatusStyle>,
    previous_left: Option<StatusLeft>,
    previous_left_length: Option<StatusLeftLength>,
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
    let fleet = connect_fleet(&cli).await?;
    let layout = select_layout(cli.forced_layout());
    let mut ui_state = RetainedUiState::new(layout);

    loop {
        let outcome = run_selector_once(&fleet, layout, &mut ui_state).await?;
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
        let exit = attach_current_pty_with_mmux_status(&target).await?;
        let code = exit.shell_status();
        if should_reenter_after_attach(&fleet, &selected, &exit).await? {
            continue;
        }
        return Ok(code);
    }
}

async fn attach_current_pty_with_mmux_status(target: &Target) -> Result<AttachExit> {
    let snapshot = prepare_attach_status(target).await;
    let exit = target.attach_current_pty().await;
    restore_attach_status(target, snapshot).await;
    exit.map_err(Into::into)
}

async fn prepare_attach_status(target: &Target) -> Option<AttachStatusSnapshot> {
    let previous_style = match target.read_local_status_style().await {
        Ok(previous) => previous,
        Err(err) => {
            eprintln!("mmux: could not read tmux status style before attach: {err}");
            return None;
        }
    };
    let previous_left = match target.read_local_status_left().await {
        Ok(previous) => previous,
        Err(err) => {
            eprintln!("mmux: could not read tmux status-left before attach: {err}");
            return None;
        }
    };
    let previous_left_length = match target.read_local_status_left_length().await {
        Ok(previous) => previous,
        Err(err) => {
            eprintln!("mmux: could not read tmux status-left-length before attach: {err}");
            return None;
        }
    };

    let style = StatusStyle::new(MMUX_ATTACH_STATUS_STYLE)
        .expect("mmux attach status style is a valid static tmux style");
    if let Err(err) = target.set_status_style(&style).await {
        eprintln!("mmux: could not set tmux status style before attach: {err}");
    }
    let left = StatusLeft::new(MMUX_ATTACH_STATUS_LEFT)
        .expect("mmux attach status-left is a valid static tmux format");
    if let Err(err) = target.set_status_left(&left).await {
        eprintln!("mmux: could not set tmux status-left before attach: {err}");
    }
    if let Err(err) = target
        .set_status_left_length(StatusLeftLength::new(MMUX_ATTACH_STATUS_LEFT_LENGTH))
        .await
    {
        eprintln!("mmux: could not set tmux status-left-length before attach: {err}");
    }

    Some(AttachStatusSnapshot {
        previous_style,
        previous_left,
        previous_left_length,
    })
}

async fn restore_attach_status(target: &Target, snapshot: Option<AttachStatusSnapshot>) {
    let Some(snapshot) = snapshot else {
        return;
    };
    let result = match snapshot.previous_style {
        Some(style) => target.set_status_style(&style).await,
        None => target.unset_status_style().await,
    };
    if let Err(err) = result {
        eprintln!("mmux: could not restore tmux status style after attach: {err}");
    }
    let result = match snapshot.previous_left {
        Some(left) => target.set_status_left(&left).await,
        None => target.unset_status_left().await,
    };
    if let Err(err) = result {
        eprintln!("mmux: could not restore tmux status-left after attach: {err}");
    }
    let result = match snapshot.previous_left_length {
        Some(length) => target.set_status_left_length(length).await,
        None => target.unset_status_left_length().await,
    };
    if let Err(err) = result {
        eprintln!("mmux: could not restore tmux status-left-length after attach: {err}");
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
    fleet: &HostFleet,
    layout: LayoutMode,
    ui_state: &mut RetainedUiState,
) -> Result<SelectorOutcome> {
    // MOTD is per-host; only meaningful in single-host mode. Multi-host mode
    // hides the MOTD pane entirely (issue #235).
    let motd = if fleet.is_multi() {
        None
    } else {
        let entry = fleet.first().expect("fleet has at least one entry");
        Some(load_motd(&entry.handle).await)
    };
    let mut app = AppState::with_fleet(fleet.clone(), layout, motd);
    ui_state.apply_to(&mut app);
    let previous_selection = ui_state.selected_session_key();
    refresh_sessions_preserving(fleet, &mut app, true, previous_selection).await?;

    let mut terminal = TerminalSession::enter()?;
    let mut last_detail_refresh = Instant::now();
    let mut last_session_refresh = Instant::now();

    loop {
        if last_session_refresh.elapsed() >= Duration::from_secs(1) {
            if let Err(err) = refresh_sessions_quiet(fleet, &mut app, false).await {
                app.status = StatusBanner::error(format!("session refresh failed: {err:#}"));
            }
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
            match handle_key(fleet, &mut app, key).await? {
                KeyOutcome::Continue => {}
                KeyOutcome::Select(selected) => {
                    ui_state.update_from(&app);
                    stop_detail_source(&mut app).await;
                    terminal.restore()?;
                    return Ok(SelectorOutcome::Selected(selected));
                }
                KeyOutcome::Cancel => {
                    ui_state.update_from(&app);
                    stop_detail_source(&mut app).await;
                    terminal.restore()?;
                    return Ok(SelectorOutcome::Cancelled);
                }
            }
        }
    }
}
