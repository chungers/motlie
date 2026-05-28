//! Binary-local split-screen TUI mirror consumer (DC32).
//!
//! Renders a top mirror frame (watched remote session transcript) and a bottom
//! REPL frame (command input + output history). Uses `HistoryHandle` for the
//! mirror transcript and `ratatui` for terminal rendering.
//!
//! This module lives beside the REPL example, not in `libs/tmux`, consistent
//! with DC11 and DC32.

use motlie_tmux::{
    has_visible_text, overlap_deduplicate, CaptureNormalizeMode, CaptureOptions, FidelityIssue,
    HistoryHandle, HistoryOptions, HostHandle, KeySequence, LabelFormat, MonitorHealth,
    ScrollbackQuery, SessionMonitorHandle, SinkFilter, TargetSpec,
};

use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame, Terminal,
};

use ansi_to_tui::IntoText;
use std::collections::VecDeque;
use std::io;
use std::time::Duration;

const BG: Color = Color::Black;
const FG: Color = Color::White;
const ACCENT: Color = Color::Yellow;
const DIM: Color = Color::Gray;

/// Action returned when the TUI event loop exits.
pub enum TuiAction {
    /// User typed `tui off` — return to plain REPL mode.
    TuiOff,
    /// User typed `quit` or Ctrl-C — exit the program.
    Quit,
}

/// Active watch state: monitor + history subscription for one session.
struct WatchState {
    monitor_handle: SessionMonitorHandle,
    history_handle: HistoryHandle,
}

/// Polling-based stream mode (mirrors stream_pane modes).
enum StreamMode {
    /// Poll visible pane via `capture()`.  Replaces mirror on change.
    Visible,
    /// Poll scrollback via `sample_text(LastLines)` + overlap dedup.
    /// Appends new lines (like `tail -f`).
    Tail { lines: usize },
    /// Poll scrollback via `sample_text(Until)`.  Replaces mirror on change.
    Until {
        pattern: regex::Regex,
        max_lines: usize,
    },
    /// Poll via `capture_with_options()` with reflow detection.
    Fidelity,
    /// Poll via `capture_all()` — rendered pane state for all panes.
    /// Works correctly with TUI programs (htop, vim, etc.).
    Render,
}

/// Active polling stream state, set by the `stream` command.
struct StreamState {
    target: motlie_tmux::Target,
    mode: StreamMode,
    interval: Duration,
    previous: String,
    last_tick: tokio::time::Instant,
}

/// All mutable state for the TUI.
struct TuiState {
    watch: Option<WatchState>,
    stream: Option<StreamState>,
    mirror_text: String,
    /// Label shown in the mirror frame title (e.g. "watching: foo", "capture: bar").
    mirror_label: String,
    /// When true, `draw_mirror` parses ANSI escape sequences in `mirror_text`
    /// into styled ratatui spans (used by visible/render modes).
    mirror_ansi: bool,
    /// When true the event loop refreshes `mirror_text` from the active
    /// `HistoryHandle` every tick.  Commands like `capture` and `history`
    /// set this to false so their one-shot output stays visible.
    mirror_auto_refresh: bool,

    input: String,
    cursor_pos: usize,
    output_lines: VecDeque<String>,

    /// Command history (oldest first).
    cmd_history: Vec<String>,
    /// Position in history during Up/Down browsing (None = not browsing).
    history_pos: Option<usize>,
    /// Input line saved when the user starts browsing history.
    saved_input: String,

    /// Left pane width as a percentage (10..=90).  Adjusted with Ctrl+Arrow.
    split_pct: u16,
    host_uri: String,
}

impl TuiState {
    fn new(host_uri: &str) -> Self {
        Self {
            watch: None,
            stream: None,
            mirror_text: String::new(),
            mirror_label: String::new(),
            mirror_ansi: false,
            mirror_auto_refresh: true,
            input: String::new(),
            cursor_pos: 0,
            output_lines: VecDeque::with_capacity(100),
            cmd_history: Vec::new(),
            history_pos: None,
            saved_input: String::new(),
            split_pct: 30,
            host_uri: host_uri.to_string(),
        }
    }

    fn push_output(&mut self, line: &str) {
        self.output_lines.push_back(line.to_string());
        if self.output_lines.len() > 100 {
            self.output_lines.pop_front();
        }
    }
}

/// Enter split-screen TUI mode and run the event loop until `tui off` or `quit`.
pub async fn run(host: &HostHandle, host_uri: &str) -> anyhow::Result<TuiAction> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Install a panic hook that restores the terminal so the user's shell
    // isn't left in raw mode. The original hook is shared via Arc so it
    // can be restored on non-panic exit paths.
    let original_hook = std::sync::Arc::new(std::panic::take_hook());
    let hook_for_panic = std::sync::Arc::clone(&original_hook);
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        hook_for_panic(info);
    }));

    let mut state = TuiState::new(host_uri);
    state.push_output("tui: split-screen mode enabled; waiting for monitor target");

    let result = event_loop(&mut terminal, &mut state, host).await;

    // Restore terminal.
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    // Remove our panic hook and restore the original. Arc::try_unwrap
    // succeeds here because the panic hook closure (the other Arc ref)
    // was just removed by take_hook().
    let _ = std::panic::take_hook();
    if let Ok(hook) = std::sync::Arc::try_unwrap(original_hook) {
        std::panic::set_hook(hook);
    }

    // Tear down any active watch.
    if let Some(watch) = state.watch.take() {
        let bus = host.output_bus();
        let _ = bus.unsubscribe(watch.history_handle.id());
        let _ = watch.history_handle.join().await;
        let _ = watch.monitor_handle.shutdown().await;
    }

    result
}

// ---------------------------------------------------------------------------
// Event loop
// ---------------------------------------------------------------------------

async fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    state: &mut TuiState,
    host: &HostHandle,
) -> anyhow::Result<TuiAction> {
    loop {
        terminal.draw(|f| draw(f, state))?;

        // Poll for a crossterm event with a short timeout so the mirror
        // refreshes even when no keys are pressed.
        let maybe_event = tokio::task::spawn_blocking(|| {
            if event::poll(Duration::from_millis(150)).unwrap_or(false) {
                event::read().ok()
            } else {
                None
            }
        })
        .await?;

        if let Some(Event::Key(key)) = maybe_event {
            // Ignore key-release events on platforms that report them.
            if key.kind == event::KeyEventKind::Release {
                continue;
            }
            match key.code {
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    return Ok(TuiAction::Quit);
                }
                KeyCode::Char(c) => {
                    state.input.insert(state.cursor_pos, c);
                    state.cursor_pos += 1;
                }
                KeyCode::Backspace if state.cursor_pos > 0 => {
                    state.cursor_pos -= 1;
                    state.input.remove(state.cursor_pos);
                }
                // Any modifier + Arrow resizes the pane split.
                // (Ctrl+Arrow is often captured by macOS Mission Control,
                //  so Shift+Arrow and Alt+Arrow also work.)
                KeyCode::Left
                    if key.modifiers.intersects(
                        KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SHIFT,
                    ) =>
                {
                    state.split_pct = state.split_pct.saturating_sub(5).max(10);
                }
                KeyCode::Right
                    if key.modifiers.intersects(
                        KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SHIFT,
                    ) =>
                {
                    state.split_pct = (state.split_pct + 5).min(90);
                }
                KeyCode::Left => {
                    state.cursor_pos = state.cursor_pos.saturating_sub(1);
                }
                KeyCode::Right if state.cursor_pos < state.input.len() => {
                    state.cursor_pos += 1;
                }
                KeyCode::Up => {
                    if state.cmd_history.is_empty() {
                        // nothing
                    } else if let Some(pos) = state.history_pos {
                        if pos > 0 {
                            state.history_pos = Some(pos - 1);
                            state.input = state.cmd_history[pos - 1].clone();
                            state.cursor_pos = state.input.len();
                        }
                    } else {
                        state.saved_input = state.input.clone();
                        let pos = state.cmd_history.len() - 1;
                        state.history_pos = Some(pos);
                        state.input = state.cmd_history[pos].clone();
                        state.cursor_pos = state.input.len();
                    }
                }
                KeyCode::Down => {
                    if let Some(pos) = state.history_pos {
                        if pos + 1 < state.cmd_history.len() {
                            state.history_pos = Some(pos + 1);
                            state.input = state.cmd_history[pos + 1].clone();
                            state.cursor_pos = state.input.len();
                        } else {
                            state.history_pos = None;
                            state.input = state.saved_input.clone();
                            state.cursor_pos = state.input.len();
                        }
                    }
                }
                KeyCode::Enter => {
                    let cmd = state.input.clone();
                    state.input.clear();
                    state.cursor_pos = 0;
                    state.history_pos = None;
                    state.saved_input.clear();
                    if !cmd.trim().is_empty() {
                        state.cmd_history.push(cmd.trim().to_string());
                        state.push_output(&format!("repl> {}", cmd));
                        if let Some(action) = process_command(&cmd, state, host).await? {
                            return Ok(action);
                        }
                    }
                }
                _ => {}
            }
        }

        // Refresh mirror: polling stream takes priority, then watch auto-refresh.
        if state.stream.is_some() {
            tick_stream(state).await;
        } else if state.mirror_auto_refresh {
            if let Some(ref watch) = state.watch {
                state.mirror_text = watch.history_handle.render_text().await.replace('\r', "");
            }
        }
    }
}

/// Drive one tick of the active polling stream, updating `mirror_text`.
async fn tick_stream(state: &mut TuiState) {
    // Take the stream out so we can mutate mirror_text without borrow conflict.
    let mut ss = match state.stream.take() {
        Some(s) => s,
        None => return,
    };

    if ss.last_tick.elapsed() < ss.interval {
        state.stream = Some(ss);
        return;
    }
    ss.last_tick = tokio::time::Instant::now();

    match &ss.mode {
        StreamMode::Visible => {
            let opts = CaptureOptions::with_mode(CaptureNormalizeMode::ScreenStable);
            if let Ok(result) = ss.target.capture_with_options(&opts).await {
                if result.text != ss.previous {
                    state.mirror_text = result.text.clone();
                    ss.previous = result.text;
                }
            }
        }
        StreamMode::Tail { lines } => {
            let query = ScrollbackQuery::LastLines(*lines);
            if let Ok(current) = ss.target.sample_text(&query).await {
                if !current.is_empty() && current != ss.previous {
                    let (merged, _) = overlap_deduplicate(&ss.previous, &current, 5);
                    if merged != ss.previous {
                        if merged.starts_with(&ss.previous) {
                            state.mirror_text.push_str(&merged[ss.previous.len()..]);
                        } else {
                            state.mirror_text.push_str(&current);
                            if !current.ends_with('\n') {
                                state.mirror_text.push('\n');
                            }
                        }
                    }
                    ss.previous = merged;
                }
            }
            // Trim accumulated text to ~100 KB.
            const MAX_MIRROR: usize = 100_000;
            if state.mirror_text.len() > MAX_MIRROR {
                let mut trim = state.mirror_text.len() - MAX_MIRROR * 3 / 4;
                // Advance to a char boundary to avoid slicing inside a
                // multi-byte UTF-8 sequence (e.g. box-drawing '─').
                while trim < state.mirror_text.len() && !state.mirror_text.is_char_boundary(trim) {
                    trim += 1;
                }
                let boundary = state.mirror_text[trim..]
                    .find('\n')
                    .map(|p| trim + p + 1)
                    .unwrap_or(trim);
                state.mirror_text = state.mirror_text[boundary..].to_string();
            }
        }
        StreamMode::Until { pattern, max_lines } => {
            let query = ScrollbackQuery::Until {
                pattern: pattern.clone(),
                max_lines: *max_lines,
            };
            if let Ok(content) = ss.target.sample_text(&query).await {
                if content != ss.previous {
                    state.mirror_text = content.clone();
                    ss.previous = content;
                }
            }
        }
        StreamMode::Fidelity => {
            let opts = CaptureOptions {
                detect_reflow: true,
                normalize: CaptureNormalizeMode::PlainText,
                ..Default::default()
            };
            if let Ok(result) = ss.target.capture_with_options(&opts).await {
                if result.text != ss.previous {
                    let mut text = result.text.clone();
                    if result.fidelity.degraded {
                        if let Some(ref issues) = result.fidelity.issues {
                            let names: Vec<&str> = issues
                                .iter()
                                .map(|i| match i {
                                    FidelityIssue::ClientResize => "ClientResize",
                                    FidelityIssue::PaneResize => "PaneResize",
                                    FidelityIssue::HistoryTruncated => "HistoryTruncated",
                                    FidelityIssue::OverlapResync => "OverlapResync",
                                })
                                .collect();
                            text.push_str(&format!("\n[DEGRADED: {}]", names.join(", ")));
                        }
                    } else {
                        text.push_str("\n[FIDELITY: CLEAN]");
                    }
                    state.mirror_text = text;
                    ss.previous = result.text;
                }
            }
        }
        StreamMode::Render => {
            if let Ok(panes) = ss.target.capture_all().await {
                let session_name = ss.target.session_name();
                let mut pane_list: Vec<_> = panes.iter().collect();
                pane_list.sort_by_key(|(addr, _)| (addr.window, addr.pane));

                let mut rendered = String::new();
                for (addr, content) in pane_list {
                    if !has_visible_text(content) {
                        continue;
                    }
                    rendered.push_str(&format!("--- {}({}) ---\n", session_name, addr.pane_id));
                    rendered.push_str(content);
                    if !content.ends_with('\n') {
                        rendered.push('\n');
                    }
                }

                if rendered != ss.previous {
                    state.mirror_text = rendered.clone();
                    ss.previous = rendered;
                }
            }
        }
    }

    state.stream = Some(ss);
}

// ---------------------------------------------------------------------------
// Command processing (TUI-mode subset)
// ---------------------------------------------------------------------------

async fn process_command(
    cmd: &str,
    state: &mut TuiState,
    host: &HostHandle,
) -> anyhow::Result<Option<TuiAction>> {
    let parts: Vec<&str> = cmd.trim().splitn(3, ' ').collect();
    if parts.is_empty() || parts[0].is_empty() {
        return Ok(None);
    }

    match parts[0] {
        "tui" => {
            if parts.get(1).copied() == Some("off") {
                return Ok(Some(TuiAction::TuiOff));
            }
            state.push_output("already in tui mode; use 'tui off' to exit");
        }
        "quit" => return Ok(Some(TuiAction::Quit)),
        "help" => {
            show_help(parts.get(1).copied(), state);
        }
        "monitor" => {
            state.stream = None;
            if parts.len() < 2 {
                state.push_output("usage: monitor <session>");
                return Ok(None);
            }
            let session_name = parts[1];

            // Tear down any previous watch.
            if let Some(prev) = state.watch.take() {
                let bus = host.output_bus();
                let _ = bus.unsubscribe(prev.history_handle.id());
                let _ = prev.history_handle.join().await;
                let _ = prev.monitor_handle.shutdown().await;
                state.mirror_text.clear();
            }

            match host.start_monitoring_session(session_name).await {
                Ok(monitor_handle) => {
                    let bus = host.output_bus();
                    let filter = SinkFilter::for_session(session_name);
                    match bus.subscribe(vec![filter], 64) {
                        Ok(sub) => {
                            // Bound the history to roughly 2x the visible
                            // mirror frame so render_text() cost stays
                            // proportional to the terminal, not the total
                            // session output.
                            let (cols, rows) = crossterm::terminal::size().unwrap_or((120, 40));
                            let mirror_chars = (cols as usize) * (rows as usize * 3 / 4) * 2;
                            let history = sub.history(HistoryOptions {
                                max_entries: 500,
                                max_render_chars: mirror_chars,
                                label_format: LabelFormat::Bracketed,
                                include_omission_marker: true,
                                ..Default::default()
                            });
                            state.watch = Some(WatchState {
                                monitor_handle,
                                history_handle: history,
                            });
                            state.mirror_label = format!("watching: {}", session_name);
                            state.mirror_ansi = false;
                            state.mirror_auto_refresh = true;
                            state.push_output(&format!("watching session: {}", session_name));
                        }
                        Err(e) => {
                            let _ = monitor_handle.shutdown().await;
                            state.push_output(&format!("subscribe error: {}", e));
                        }
                    }
                }
                Err(e) => state.push_output(&format!("monitor error: {}", e)),
            }
        }
        "create" => {
            if parts.len() < 2 {
                state.push_output("usage: create <name>");
                return Ok(None);
            }
            match host.create_session(parts[1], &Default::default()).await {
                Ok(_) => state.push_output(&format!("created: {}", parts[1])),
                Err(e) => state.push_output(&format!("error: {}", e)),
            }
        }
        "kill" => {
            if parts.len() < 2 {
                state.push_output("usage: kill <target>");
                return Ok(None);
            }
            match resolve_target(host, parts[1]).await {
                Ok(target) => match target.kill().await {
                    Ok(()) => state.push_output(&format!("killed: {}", parts[1])),
                    Err(e) => state.push_output(&format!("error: {}", e)),
                },
                Err(e) => state.push_output(&e),
            }
        }
        "targets" => match host.list_sessions().await {
            Ok(sessions) => {
                if sessions.is_empty() {
                    state.push_output("  (no sessions)");
                }
                for s in &sessions {
                    let spec = match TargetSpec::parse(&s.name) {
                        Ok(sp) => sp,
                        Err(e) => {
                            state.push_output(&format!("  {} (parse error: {})", s.name, e));
                            continue;
                        }
                    };
                    let target = match host.target(&spec).await {
                        Ok(Some(t)) => t,
                        Ok(None) => {
                            state.push_output(&format!("  {} (not found)", s.name));
                            continue;
                        }
                        Err(e) => {
                            state.push_output(&format!("  {} (error: {})", s.name, e));
                            continue;
                        }
                    };
                    let windows = match target.children().await {
                        Ok(w) => w,
                        Err(e) => {
                            state.push_output(&format!("  {} (error: {})", s.name, e));
                            continue;
                        }
                    };
                    state.push_output(&format!(
                        "  {:<20} ({} window{})",
                        s.name,
                        windows.len(),
                        if windows.len() == 1 { "" } else { "s" }
                    ));
                    for w in &windows {
                        let panes = match w.children().await {
                            Ok(p) => p,
                            Err(_) => continue,
                        };
                        let winfo = w.window_info();
                        let wname = winfo.map(|i| i.name.as_str()).unwrap_or("?");
                        state.push_output(&format!(
                            "    {:<18} ('{}', {} pane{})",
                            w.target_string(),
                            wname,
                            panes.len(),
                            if panes.len() == 1 { "" } else { "s" }
                        ));
                        for p in &panes {
                            let pid = p.pane_address().map(|a| a.pane_id.as_str()).unwrap_or("?");
                            state.push_output(&format!(
                                "      {:<16} ({})",
                                p.target_string(),
                                pid
                            ));
                        }
                    }
                }
            }
            Err(e) => state.push_output(&format!("error: {}", e)),
        },
        "send" => {
            if parts.len() < 3 {
                state.push_output("usage: send <target> <text...>");
                return Ok(None);
            }
            match resolve_target(host, parts[1]).await {
                Ok(target) => {
                    let enter = KeySequence::parse("{Enter}").expect("static parse");
                    if let Err(e) = target.send_text(parts[2]).await {
                        state.push_output(&format!("error sending text: {}", e));
                    } else if let Err(e) = target.send_keys(&enter).await {
                        state.push_output(&format!("error sending Enter: {}", e));
                    } else {
                        state.push_output(&format!("sent to {}", parts[1]));
                    }
                }
                Err(e) => state.push_output(&e),
            }
        }
        "keys" => {
            if parts.len() < 3 {
                state.push_output("usage: keys <target> <keys...>");
                return Ok(None);
            }
            match KeySequence::parse(parts[2]) {
                Ok(keys) => match resolve_target(host, parts[1]).await {
                    Ok(target) => match target.send_keys(&keys).await {
                        Ok(()) => state.push_output(&format!("sent keys to {}", parts[1])),
                        Err(e) => state.push_output(&format!("error: {}", e)),
                    },
                    Err(e) => state.push_output(&e),
                },
                Err(e) => state.push_output(&format!("parse error: {}", e)),
            }
        }
        "capture" => {
            state.stream = None;
            if let Some(prev) = state.watch.take() {
                let bus = host.output_bus();
                let _ = bus.unsubscribe(prev.history_handle.id());
                let _ = prev.history_handle.join().await;
                let _ = prev.monitor_handle.shutdown().await;
            }
            if parts.len() < 3 {
                state.push_output("usage: capture <target> <n>");
                return Ok(None);
            }
            let target_str = parts[1];
            let n: usize = match parts[2].parse() {
                Ok(n) if n > 0 => n,
                _ => {
                    state.push_output("error: <n> must be a positive integer");
                    return Ok(None);
                }
            };
            match resolve_target(host, target_str).await {
                Ok(target) => {
                    let query = ScrollbackQuery::LastLines(n);
                    match target.sample_text(&query).await {
                        Ok(text) => {
                            if text.is_empty() {
                                state.mirror_text = "(empty)".to_string();
                            } else {
                                state.mirror_text = text;
                            }
                            state.mirror_label = format!("capture: {} (last {})", target_str, n);
                            state.mirror_ansi = false;
                            state.mirror_auto_refresh = false;
                            state.push_output(&format!("captured {} lines from {}", n, target_str));
                        }
                        Err(e) => state.push_output(&format!("error: {}", e)),
                    }
                }
                Err(e) => state.push_output(&e),
            }
        }
        "history" => {
            state.stream = None;
            if let Some(prev) = state.watch.take() {
                let bus = host.output_bus();
                let _ = bus.unsubscribe(prev.history_handle.id());
                let _ = prev.history_handle.join().await;
                let _ = prev.monitor_handle.shutdown().await;
            }
            let words: Vec<&str> = cmd.split_whitespace().collect();
            if words.len() < 2 {
                state.push_output("usage: history <session> [session...]");
                return Ok(None);
            }
            let session_names = &words[1..];
            let mut output = String::new();
            let mut captured: Vec<&str> = Vec::new();
            for &name in session_names {
                match host.session(name).await {
                    Ok(Some(target)) => match target.capture_all().await {
                        Ok(panes) => {
                            let mut pane_list: Vec<_> = panes.iter().collect();
                            pane_list.sort_by_key(|(addr, _)| (addr.window, addr.pane));
                            for (addr, content) in pane_list {
                                if !has_visible_text(content) {
                                    continue;
                                }
                                output.push_str(&format!("--- {}({}) ---\n", name, addr.pane_id));
                                output.push_str(content);
                                if !content.ends_with('\n') {
                                    output.push('\n');
                                }
                            }
                            captured.push(name);
                        }
                        Err(e) => {
                            state.push_output(&format!("error capturing {}: {}", name, e));
                        }
                    },
                    Ok(None) => {
                        state.push_output(&format!("session '{}' not found", name));
                    }
                    Err(e) => {
                        state.push_output(&format!("error resolving '{}': {}", name, e));
                    }
                }
            }
            if output.is_empty() {
                state.mirror_text = "(no visible content)".to_string();
            } else {
                state.mirror_text = output;
            }
            state.mirror_label = format!("history: {}", captured.join(", "));
            state.mirror_ansi = false;
            state.mirror_auto_refresh = false;
            if !captured.is_empty() {
                state.push_output(&format!("history: {}", captured.join(", ")));
            }
        }
        "stream" => {
            let words: Vec<&str> = cmd.split_whitespace().collect();
            if words.len() < 2 {
                state.push_output(
                    "usage: stream <target> [--mode visible|tail|until|fidelity|monitor|render] \
                     [--lines N] [--interval MS] [--pattern REGEX]",
                );
                return Ok(None);
            }
            let target_str = words[1];
            let mut mode_str = "tail";
            let mut lines = 50usize;
            let mut interval_ms = 200u64;
            let mut pattern_str = r"^\$ ";
            let mut i = 2;
            let mut parse_err = false;
            while i < words.len() {
                match words[i] {
                    "--mode" => {
                        i += 1;
                        match words.get(i) {
                            Some(v) => mode_str = v,
                            None => {
                                state.push_output("--mode requires a value");
                                parse_err = true;
                                break;
                            }
                        }
                    }
                    "--lines" => {
                        i += 1;
                        match words.get(i).and_then(|v| v.parse().ok()) {
                            Some(n) if n > 0 => lines = n,
                            _ => {
                                state.push_output("--lines must be a positive integer");
                                parse_err = true;
                                break;
                            }
                        }
                    }
                    "--interval" => {
                        i += 1;
                        match words.get(i).and_then(|v| v.parse().ok()) {
                            Some(n) if n > 0 => interval_ms = n,
                            _ => {
                                state.push_output("--interval must be a positive integer");
                                parse_err = true;
                                break;
                            }
                        }
                    }
                    "--pattern" => {
                        i += 1;
                        match words.get(i) {
                            Some(v) => pattern_str = v,
                            None => {
                                state.push_output("--pattern requires a value");
                                parse_err = true;
                                break;
                            }
                        }
                    }
                    other => {
                        state.push_output(&format!("unknown flag: {}", other));
                        parse_err = true;
                        break;
                    }
                }
                i += 1;
            }
            if parse_err {
                return Ok(None);
            }

            // Resolve target.
            let target = match resolve_target(host, target_str).await {
                Ok(t) => t,
                Err(e) => {
                    state.push_output(&e);
                    return Ok(None);
                }
            };

            match mode_str {
                "visible" | "tail" | "until" | "fidelity" | "render" => {
                    let mode = match mode_str {
                        "visible" => StreamMode::Visible,
                        "tail" => StreamMode::Tail { lines },
                        "until" => match regex::Regex::new(pattern_str) {
                            Ok(re) => StreamMode::Until {
                                pattern: re,
                                max_lines: lines,
                            },
                            Err(e) => {
                                state.push_output(&format!("invalid regex: {}", e));
                                return Ok(None);
                            }
                        },
                        "fidelity" => StreamMode::Fidelity,
                        "render" => StreamMode::Render,
                        _ => unreachable!(),
                    };

                    // Initial capture for tail mode.
                    let initial = if matches!(mode, StreamMode::Tail { .. }) {
                        let q = ScrollbackQuery::LastLines(lines);
                        target.sample_text(&q).await.unwrap_or_default()
                    } else {
                        String::new()
                    };

                    // Tear down any existing watch.
                    if let Some(prev) = state.watch.take() {
                        let bus = host.output_bus();
                        let _ = bus.unsubscribe(prev.history_handle.id());
                        let _ = prev.history_handle.join().await;
                        let _ = prev.monitor_handle.shutdown().await;
                    }

                    state.stream = Some(StreamState {
                        target,
                        mode,
                        interval: Duration::from_millis(interval_ms),
                        previous: initial.clone(),
                        last_tick: tokio::time::Instant::now(),
                    });
                    state.mirror_text = initial;
                    state.mirror_label = format!("stream: {} [{}]", target_str, mode_str);
                    state.mirror_ansi = mode_str == "visible";
                    state.mirror_auto_refresh = false;
                    state.push_output(&format!(
                        "streaming {} [mode={}, interval={}ms]",
                        target_str, mode_str, interval_ms
                    ));
                }
                "monitor" => {
                    // Event-driven mode — reuse the watch/HistoryHandle pipeline.
                    state.stream = None;
                    let session_name = target.session_name();

                    // Tear down any previous watch.
                    if let Some(prev) = state.watch.take() {
                        let bus = host.output_bus();
                        let _ = bus.unsubscribe(prev.history_handle.id());
                        let _ = prev.history_handle.join().await;
                        let _ = prev.monitor_handle.shutdown().await;
                        state.mirror_text.clear();
                    }

                    match host.start_monitoring_session(session_name).await {
                        Ok(monitor_handle) => {
                            let bus = host.output_bus();
                            let filter = SinkFilter::for_session(session_name);
                            match bus.subscribe(vec![filter], 64) {
                                Ok(sub) => {
                                    let (cols, rows) =
                                        crossterm::terminal::size().unwrap_or((120, 40));
                                    let mirror_chars =
                                        (cols as usize) * (rows as usize * 3 / 4) * 2;
                                    let history = sub.history(HistoryOptions {
                                        max_entries: 500,
                                        max_render_chars: mirror_chars,
                                        label_format: LabelFormat::Bracketed,
                                        include_omission_marker: true,
                                        ..Default::default()
                                    });
                                    state.watch = Some(WatchState {
                                        monitor_handle,
                                        history_handle: history,
                                    });
                                    state.mirror_label =
                                        format!("stream: {} [{}]", target_str, mode_str);
                                    state.mirror_ansi = false;
                                    state.mirror_auto_refresh = true;
                                    state.push_output(&format!(
                                        "streaming {} [mode={}, event-driven]",
                                        target_str, mode_str
                                    ));
                                }
                                Err(e) => {
                                    let _ = monitor_handle.shutdown().await;
                                    state.push_output(&format!("subscribe error: {}", e));
                                }
                            }
                        }
                        Err(e) => state.push_output(&format!("monitor error: {}", e)),
                    }
                }
                other => {
                    state.push_output(&format!(
                        "unknown mode: '{}' (visible|tail|until|fidelity|monitor|render)",
                        other
                    ));
                }
            }
        }
        other => {
            state.push_output(&format!(
                "unknown command: '{}'; type 'help' for available commands",
                other
            ));
        }
    }
    Ok(None)
}

/// Resolve a target spec string into a `Target`.
async fn resolve_target(
    host: &HostHandle,
    target_str: &str,
) -> Result<motlie_tmux::Target, String> {
    let spec = TargetSpec::parse(target_str)
        .map_err(|e| format!("invalid target '{}': {}", target_str, e))?;
    host.target(&spec)
        .await
        .map_err(|e| format!("error resolving '{}': {}", target_str, e))?
        .ok_or_else(|| format!("target '{}' not found", target_str))
}

// ---------------------------------------------------------------------------
// Per-command help
// ---------------------------------------------------------------------------

fn show_help(topic: Option<&str>, state: &mut TuiState) {
    match topic {
        None => {
            state.push_output("TUI mode commands (mirror pane):");
            state.push_output("  stream    Continuously stream pane content to mirror pane");
            state.push_output("  monitor   Event-driven session output to mirror pane");
            state.push_output("  capture   One-shot scrollback snapshot to mirror pane");
            state.push_output("  history   One-shot session capture to mirror pane");
            state.push_output("TUI mode commands (general):");
            state.push_output("  create    Create a tmux session");
            state.push_output("  kill      Kill a session/window/pane");
            state.push_output("  targets   List all sessions with target tree");
            state.push_output("  send      Send text + Enter to a target");
            state.push_output("  keys      Send a key sequence to a target");
            state.push_output("  tui off   Return to plain REPL mode");
            state.push_output("  quit      Exit the program");
            state.push_output("Type 'help <command>' for detailed usage.");
        }
        Some("stream") => {
            state.push_output("stream <target> [OPTIONS]");
            state.push_output("");
            state.push_output("  Continuously stream pane content to the mirror pane.");
            state.push_output("  Mirrors the stream_pane example — six capture strategies.");
            state.push_output("");
            state.push_output("OPTIONS:");
            state.push_output("  --mode MODE       Capture strategy [default: tail]");
            state.push_output(
                "  --lines N         Scrollback line count (tail/until) [default: 50]",
            );
            state.push_output(
                "  --interval MS     Poll interval in ms (polling modes) [default: 200]",
            );
            state.push_output("  --pattern REGEX   Regex for until mode [default: ^\\$ ]");
            state.push_output("");
            state.push_output("MODES (polling — output refreshes at --interval):");
            state.push_output("  tail      Poll scrollback with overlap dedup. Appends only new");
            state.push_output("            lines, like `tail -f`. Best for shells / log streams.");
            state.push_output("  visible   Poll visible pane via capture(). Replaces mirror on");
            state.push_output("            change. Best for TUI programs (htop, vim, top).");
            state.push_output("  until     Poll scrollback backwards until --pattern matches.");
            state.push_output("            Shows from match to end. Good for watching output");
            state.push_output("            since the last shell prompt.");
            state.push_output("  fidelity  Poll with reflow detection. Shows content + fidelity");
            state.push_output("            status (CLEAN or DEGRADED with issue names). Resize");
            state.push_output("            the target pane to see ClientResize/PaneResize.");
            state.push_output("  render    Poll via capture_all() — shows the rendered state of");
            state.push_output("            all panes with headers. Works correctly with TUI");
            state.push_output("            programs (htop, vim, etc.) unlike monitor mode.");
            state.push_output("");
            state.push_output("MODES (event-driven — --interval/--lines ignored):");
            state.push_output("  monitor   Real-time via tmux control mode. All panes in the");
            state.push_output("            session stream with source labels. No polling.");
            state.push_output("            Best for shells/logs; garbled for TUI programs.");
            state.push_output("");
            state.push_output("WHICH MODE TO USE:");
            state.push_output("  Mode      TUI programs    Shell / logs");
            state.push_output("  --------  --------------  ----------------");
            state.push_output("  visible   best (1 pane)   ok");
            state.push_output("  render    good (all panes) good");
            state.push_output("  tail      no              best (incremental)");
            state.push_output("  until     no              good (prompt-aware)");
            state.push_output("  fidelity  good            good (+ reflow info)");
            state.push_output("  monitor   garbled         real-time, no polling");
            state.push_output("");
            state.push_output("EXAMPLES:");
            state.push_output("  stream my_session                          # tail, 50 lines");
            state.push_output("  stream my_session --mode visible           # watch a TUI");
            state.push_output("  stream my_session --mode render            # all panes, rendered");
            state.push_output("  stream my_session --mode tail --lines 100 --interval 500");
            state.push_output("  stream my_session --mode until --pattern '^\\$ '");
            state.push_output("  stream my_session:0.1 --mode fidelity");
            state.push_output("  stream my_session --mode monitor           # shells only");
        }
        Some("monitor") => {
            state.push_output("monitor <session>");
            state.push_output("");
            state.push_output("  Start event-driven monitoring of a tmux session. Output");
            state.push_output("  streams to the mirror pane in real time via tmux control mode.");
            state.push_output("  Uses a HistoryHandle with rolling transcript — older entries");
            state.push_output("  are evicted when the render budget is exceeded.");
            state.push_output("");
            state.push_output("  Only one monitor can be active at a time. Running monitor");
            state.push_output("  again replaces the previous watch.");
            state.push_output("");
            state.push_output("  The mirror pane auto-refreshes every tick (~150ms). Use");
            state.push_output("  'capture' or 'history' to freeze a snapshot.");
            state.push_output("");
            state.push_output("EXAMPLES:");
            state.push_output("  monitor my_session");
        }
        Some("capture") => {
            state.push_output("capture <target> <n>");
            state.push_output("");
            state.push_output("  One-shot capture of the last N lines of scrollback from a");
            state.push_output("  target pane. Output is displayed in the mirror pane and");
            state.push_output("  frozen (auto-refresh paused). Any active stream is stopped.");
            state.push_output("");
            state.push_output("  <target> is a tmux target string:");
            state.push_output("    session              first pane of first window");
            state.push_output("    session:window       first pane of named window");
            state.push_output("    session:window.pane  specific pane");
            state.push_output("");
            state.push_output("EXAMPLES:");
            state.push_output("  capture my_session 50");
            state.push_output("  capture my_session:0.1 200");
        }
        Some("history") => {
            state.push_output("history <session> [session...]");
            state.push_output("");
            state.push_output("  One-shot capture of all panes across one or more sessions.");
            state.push_output("  Calls capture_all() on each session and renders every pane");
            state.push_output("  with a '--- session(pane_id) ---' header. Output is displayed");
            state.push_output("  in the mirror pane and frozen.");
            state.push_output("");
            state.push_output("  Empty panes (no visible text) are omitted. Panes are sorted");
            state.push_output("  by window index then pane index.");
            state.push_output("");
            state.push_output("EXAMPLES:");
            state.push_output("  history my_session");
            state.push_output("  history agent_a agent_b        # multi-session");
        }
        Some("create") => {
            state.push_output("create <name>");
            state.push_output("");
            state.push_output("  Create a new tmux session with the given name.");
            state.push_output("  Uses default options (no custom size or history limit).");
            state.push_output("");
            state.push_output("EXAMPLES:");
            state.push_output("  create my_session");
        }
        Some("kill") => {
            state.push_output("kill <target>");
            state.push_output("");
            state.push_output("  Kill a session, window, or pane identified by <target>.");
            state.push_output("");
            state.push_output("  <target> is a tmux target string:");
            state.push_output("    session              kill entire session");
            state.push_output("    session:window       kill a window");
            state.push_output("    session:window.pane  kill a specific pane");
            state.push_output("");
            state.push_output("EXAMPLES:");
            state.push_output("  kill my_session");
            state.push_output("  kill my_session:0.1");
        }
        Some("targets") => {
            state.push_output("targets");
            state.push_output("");
            state.push_output("  List all tmux sessions on the connected host with their");
            state.push_output("  full target tree: sessions > windows > panes.");
            state.push_output("  Shows window names and pane IDs.");
        }
        Some("send") => {
            state.push_output("send <target> <text...>");
            state.push_output("");
            state.push_output("  Send text followed by Enter to a target pane.");
            state.push_output("  The text is sent as-is (no escaping). Enter is appended.");
            state.push_output("");
            state.push_output("EXAMPLES:");
            state.push_output("  send my_session ls -la");
            state.push_output("  send my_session:0.1 echo hello");
        }
        Some("keys") => {
            state.push_output("keys <target> <keys...>");
            state.push_output("");
            state.push_output("  Send a key sequence to a target pane. Supports special");
            state.push_output("  keys in {braces}:");
            state.push_output("    {Enter}    Enter/Return");
            state.push_output("    {Escape}   Escape key");
            state.push_output("    {C-c}      Ctrl-C");
            state.push_output("    {C-d}      Ctrl-D");
            state.push_output("    {C-z}      Ctrl-Z");
            state.push_output("  Literal text and specials can be mixed.");
            state.push_output("");
            state.push_output("EXAMPLES:");
            state.push_output("  keys my_session {C-c}");
            state.push_output("  keys my_session {Escape}:wq{Enter}");
            state.push_output("  keys my_session echo hello{Enter}");
        }
        Some("tui") => {
            state.push_output("tui off");
            state.push_output("");
            state.push_output("  Return to plain REPL mode. The TUI alternate screen is");
            state.push_output("  restored and the REPL prompt resumes. Any active monitor");
            state.push_output("  or stream is torn down.");
        }
        Some("quit") => {
            state.push_output("quit");
            state.push_output("");
            state.push_output("  Disconnect from the host and exit the program.");
            state.push_output("  Also triggered by Ctrl-C.");
        }
        Some("help") => {
            state.push_output("help [command]");
            state.push_output("");
            state.push_output("  Without arguments, show all available commands.");
            state.push_output("  With a command name, show detailed usage for that command.");
        }
        Some(other) => {
            state.push_output(&format!(
                "no help for '{}'; type 'help' for command list",
                other
            ));
        }
    }
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

fn draw(f: &mut Frame, state: &TuiState) {
    let area = f.area();

    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(state.split_pct),
            Constraint::Percentage(100 - state.split_pct),
        ])
        .split(area);

    draw_repl(f, columns[0], state);
    draw_mirror(f, columns[1], state);
}

fn draw_mirror(f: &mut Frame, area: ratatui::layout::Rect, state: &TuiState) {
    let title = if state.mirror_label.is_empty() {
        " Mirror Frame ".to_string()
    } else {
        format!(" Mirror — {} ", state.mirror_label)
    };

    let content_str = if !state.mirror_text.is_empty() {
        state.mirror_text.as_str()
    } else if state.watch.is_some() {
        "(waiting for output...)"
    } else {
        "(use 'monitor', 'capture', or 'history')"
    };

    // Show the most recent lines (bottom of transcript).
    let inner_height = area.height.saturating_sub(2) as usize;
    let all_lines: Vec<&str> = content_str.lines().collect();
    let visible_start = all_lines.len().saturating_sub(inner_height);
    let visible_text: String = all_lines[visible_start..].join("\n");

    // Build status line for the bottom border.
    let stream_label = match &state.watch {
        Some(w) => match w.monitor_handle.health() {
            MonitorHealth::Streaming => " stream: active ",
            MonitorHealth::Reconnecting => " stream: reconnecting ",
            MonitorHealth::Failed => " stream: failed ",
            MonitorHealth::Stopped => " stream: stopped ",
        },
        None => " stream: idle ",
    };
    let status_style = if matches!(
        state.watch.as_ref().map(|w| w.monitor_handle.health()),
        Some(MonitorHealth::Streaming) | Some(MonitorHealth::Reconnecting)
    ) {
        Style::default().fg(ACCENT).bg(BG)
    } else {
        Style::default().fg(DIM).bg(BG)
    };
    let status_line = Line::from(vec![
        Span::styled(
            format!(" {} ", state.host_uri),
            Style::default().fg(DIM).bg(BG),
        ),
        Span::styled(stream_label, status_style),
    ]);

    let block = Block::default()
        .title(title)
        .title_style(Style::default().fg(ACCENT).bg(BG))
        .title_bottom(status_line)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ACCENT).bg(BG));

    // When mirror_ansi is set, parse ANSI escape sequences into styled spans.
    if state.mirror_ansi {
        let styled = visible_text
            .into_text()
            .unwrap_or_else(|_| ratatui::text::Text::raw(&state.mirror_text));
        let paragraph = Paragraph::new(styled)
            .style(Style::default().bg(BG))
            .block(block);
        f.render_widget(paragraph, area);
    } else {
        let paragraph = Paragraph::new(visible_text)
            .style(Style::default().fg(FG).bg(BG))
            .block(block)
            .wrap(Wrap { trim: false });
        f.render_widget(paragraph, area);
    }
}

fn draw_repl(f: &mut Frame, area: ratatui::layout::Rect, state: &TuiState) {
    let inner_height = area.height.saturating_sub(2) as usize;
    let output_height = inner_height.saturating_sub(1);

    let mut visible: Vec<Line> = Vec::new();
    let start = state.output_lines.len().saturating_sub(output_height);
    for line in state.output_lines.iter().skip(start) {
        visible.push(Line::from(Span::raw(line.as_str())));
    }

    // Input line with blinking cursor.
    visible.push(Line::from(vec![
        Span::styled("repl> ", Style::default().fg(ACCENT).bg(BG)),
        Span::styled(&state.input, Style::default().fg(FG).bg(BG)),
        Span::styled(
            "_",
            Style::default()
                .fg(ACCENT)
                .bg(BG)
                .add_modifier(Modifier::SLOW_BLINK),
        ),
    ]));

    let mode_hint = Span::styled(
        " mode: tui on ",
        Style::default()
            .fg(ACCENT)
            .bg(BG)
            .add_modifier(Modifier::ITALIC),
    );

    let paragraph = Paragraph::new(visible)
        .style(Style::default().fg(FG).bg(BG))
        .block(
            Block::default()
                .title(" REPL ")
                .title_style(Style::default().fg(ACCENT).bg(BG))
                .title_bottom(mode_hint)
                .borders(Borders::ALL)
                .border_style(Style::default().fg(ACCENT).bg(BG)),
        );
    f.render_widget(paragraph, area);
}
