//! Binary-local split-screen TUI mirror consumer (DC32).
//!
//! Renders a top mirror frame (watched remote session transcript) and a bottom
//! REPL frame (command input + output history). Uses `HistoryHandle` for the
//! mirror transcript and `ratatui` for terminal rendering.
//!
//! This module lives beside the REPL example, not in `libs/tmux`, consistent
//! with DC11 and DC32.

use motlie_tmux::{
    HistoryHandle, HistoryOptions, HostHandle, LabelFormat, SessionMonitorHandle, SinkFilter,
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

use std::collections::VecDeque;
use std::io;
use std::time::Duration;

/// Action returned when the TUI event loop exits.
pub enum TuiAction {
    /// User typed `tui off` — return to plain REPL mode.
    TuiOff,
    /// User typed `quit` or Ctrl-C — exit the program.
    Quit,
}

/// Active watch state: monitor + history subscription for one session.
struct WatchState {
    session_name: String,
    monitor_handle: SessionMonitorHandle,
    history_handle: HistoryHandle,
}

/// All mutable state for the TUI.
struct TuiState {
    watch: Option<WatchState>,
    mirror_text: String,

    input: String,
    cursor_pos: usize,
    output_lines: VecDeque<String>,

    host_uri: String,
}

impl TuiState {
    fn new(host_uri: &str) -> Self {
        Self {
            watch: None,
            mirror_text: String::new(),
            input: String::new(),
            cursor_pos: 0,
            output_lines: VecDeque::with_capacity(100),
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

    // Restore terminal on panic so the user's shell isn't left in raw mode.
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        original_hook(info);
    }));

    let mut state = TuiState::new(host_uri);
    state.push_output("tui: split-screen mode enabled; waiting for monitor target");

    let result = event_loop(&mut terminal, &mut state, host).await;

    // Restore terminal.
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    // Drop the panic hook installed above.
    let _ = std::panic::take_hook();

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
                KeyCode::Backspace => {
                    if state.cursor_pos > 0 {
                        state.cursor_pos -= 1;
                        state.input.remove(state.cursor_pos);
                    }
                }
                KeyCode::Left => {
                    state.cursor_pos = state.cursor_pos.saturating_sub(1);
                }
                KeyCode::Right => {
                    if state.cursor_pos < state.input.len() {
                        state.cursor_pos += 1;
                    }
                }
                KeyCode::Enter => {
                    let cmd = state.input.clone();
                    state.input.clear();
                    state.cursor_pos = 0;
                    if !cmd.trim().is_empty() {
                        state.push_output(&format!("repl> {}", cmd));
                        if let Some(action) = process_command(&cmd, state, host).await? {
                            return Ok(action);
                        }
                    }
                }
                _ => {}
            }
        }

        // Refresh mirror transcript.
        if let Some(ref watch) = state.watch {
            state.mirror_text = watch.history_handle.render_text().await.replace('\r', "");
        }
    }
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
            if parts.get(1).map(|s| *s) == Some("off") {
                return Ok(Some(TuiAction::TuiOff));
            }
            state.push_output("already in tui mode; use 'tui off' to exit");
        }
        "quit" => return Ok(Some(TuiAction::Quit)),
        "help" => {
            state.push_output("TUI mode commands:");
            state.push_output("  monitor <session>   Watch a session in the mirror frame");
            state.push_output("  tui off             Return to plain REPL mode");
            state.push_output("  quit                Exit the program");
            state.push_output("  help                Show this help");
        }
        "monitor" => {
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
                            let history = sub.history(HistoryOptions {
                                max_entries: 500,
                                max_render_chars: 0,
                                label_format: LabelFormat::Bracketed,
                                include_omission_marker: true,
                            });
                            state.watch = Some(WatchState {
                                session_name: session_name.to_string(),
                                monitor_handle,
                                history_handle: history,
                            });
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
        other => {
            state.push_output(&format!(
                "'{}' not available in TUI mode; use 'tui off' first",
                other
            ));
        }
    }
    Ok(None)
}

// ---------------------------------------------------------------------------
// Drawing
// ---------------------------------------------------------------------------

fn draw(f: &mut Frame, state: &TuiState) {
    let area = f.area();
    let repl_height = area.height.min(12).max(6) / 1;
    let repl_height = repl_height.min(area.height / 4).max(6);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(5),
            Constraint::Length(1),
            Constraint::Length(repl_height),
        ])
        .split(area);

    draw_mirror(f, chunks[0], state);
    draw_status(f, chunks[1], state);
    draw_repl(f, chunks[2], state);
}

fn draw_mirror(f: &mut Frame, area: ratatui::layout::Rect, state: &TuiState) {
    let title = match &state.watch {
        Some(w) => format!(" Mirror Frame — watching: {} ", w.session_name),
        None => " Mirror Frame ".to_string(),
    };

    let content = if state.watch.is_none() {
        "(no session watched — use 'monitor <session>' to start)".to_string()
    } else if state.mirror_text.is_empty() {
        "(waiting for output...)".to_string()
    } else {
        state.mirror_text.clone()
    };

    // Show the most recent lines (bottom of transcript).
    let inner_height = area.height.saturating_sub(2) as usize;
    let lines: Vec<&str> = content.lines().collect();
    let visible_start = lines.len().saturating_sub(inner_height);
    let visible_text: String = lines[visible_start..].join("\n");

    let paragraph = Paragraph::new(visible_text)
        .block(
            Block::default()
                .title(title)
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Blue)),
        )
        .wrap(Wrap { trim: false });
    f.render_widget(paragraph, area);
}

fn draw_status(f: &mut Frame, area: ratatui::layout::Rect, state: &TuiState) {
    let stream_label = match &state.watch {
        Some(_) => Span::styled(" stream: active ", Style::default().fg(Color::Green)),
        None => Span::styled(" stream: idle ", Style::default().fg(Color::DarkGray)),
    };

    let status = Line::from(vec![
        Span::styled(
            format!(" host: {} ", state.host_uri),
            Style::default().fg(Color::Cyan),
        ),
        Span::raw("|"),
        stream_label,
        Span::raw("|"),
        Span::styled(
            " render: history mirror ",
            Style::default().fg(Color::DarkGray),
        ),
    ]);

    let bar = Paragraph::new(status).style(Style::default().bg(Color::DarkGray).fg(Color::White));
    f.render_widget(bar, area);
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
        Span::styled("repl> ", Style::default().fg(Color::Green)),
        Span::raw(&state.input),
        Span::styled("_", Style::default().add_modifier(Modifier::SLOW_BLINK)),
    ]));

    let mode_hint = Span::styled(
        " mode: tui on ",
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::ITALIC),
    );

    let paragraph = Paragraph::new(visible).block(
        Block::default()
            .title(" REPL ")
            .title_bottom(mode_hint)
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Blue)),
    );
    f.render_widget(paragraph, area);
}
