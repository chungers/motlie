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
use motlie_tmux::HostHandle;

use cli::{select_layout, Cli};
use controller::{
    drain_host_events, handle_key, load_motd, refresh_detail, refresh_sessions_preserving,
    refresh_sessions_quiet, stop_detail_source, KeyOutcome,
};
use forcecommand::maybe_run_forcecommand_bypass;
use model::{AppState, LayoutMode, RetainedUiState, SelectedSession};
use target_host::{connect_host, HostIdentity};
use terminal::TerminalSession;

#[derive(Debug)]
enum SelectorOutcome {
    Selected(SelectedSession),
    Cancelled,
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
    let (host, identity) = connect_host(&cli).await?;
    let layout = select_layout(cli.forced_layout());
    let mut ui_state = RetainedUiState::new(layout);

    loop {
        let outcome = run_selector_once(&host, &identity, layout, &mut ui_state).await?;
        let selected = match outcome {
            SelectorOutcome::Selected(selected) => selected,
            SelectorOutcome::Cancelled => return Ok(if cli.script { 1 } else { 0 }),
        };

        if cli.script {
            println!("{}", selected.name);
            return Ok(0);
        }

        let target = match host.session_by_id(&selected.id).await? {
            Some(target) => target,
            None => {
                eprintln!("mmux: selected session disappeared; returning to selector");
                continue;
            }
        };
        let exit = target.attach_current_pty().await?;
        let code = exit.shell_status();
        if should_reenter_after_attach(&host, &selected, &exit).await? {
            continue;
        }
        return Ok(code);
    }
}

async fn should_reenter_after_attach(
    host: &HostHandle,
    selected: &SelectedSession,
    exit: &motlie_tmux::AttachExit,
) -> Result<bool> {
    if exit.success() {
        return Ok(true);
    }
    Ok(host.session_by_id(&selected.id).await?.is_some())
}

async fn run_selector_once(
    host: &HostHandle,
    identity: &HostIdentity,
    layout: LayoutMode,
    ui_state: &mut RetainedUiState,
) -> Result<SelectorOutcome> {
    let (motd, placeholder) = load_motd(host).await;
    let mut app = AppState::new_with_host_ip(
        identity.hostname.clone(),
        identity.ip_address.clone(),
        layout,
        motd,
        placeholder,
    );
    ui_state.apply_to(&mut app);
    let previous_selection = ui_state.selected_session_id();
    refresh_sessions_preserving(host, &mut app, true, previous_selection).await?;

    let mut event_rx = host
        .watch_host_events()
        .await
        .ok()
        .map(motlie_tmux::HostEventStream::into_receiver);
    let mut terminal = TerminalSession::enter()?;
    let mut last_detail_refresh = Instant::now();
    let mut last_session_refresh = Instant::now();

    loop {
        drain_host_events(host, &mut app, &mut event_rx).await?;
        if last_session_refresh.elapsed() >= Duration::from_secs(1) {
            refresh_sessions_quiet(host, &mut app, false).await?;
            last_session_refresh = Instant::now();
        }
        if last_detail_refresh.elapsed() >= Duration::from_millis(750) {
            refresh_detail(host, &mut app, false).await?;
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
            match handle_key(host, &mut app, key).await? {
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
