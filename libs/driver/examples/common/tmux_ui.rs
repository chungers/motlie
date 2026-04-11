use std::collections::VecDeque;
use std::io;
use std::time::Duration;

use ansi_to_tui::IntoText;
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use motlie_driver::commands::tmux::{TmuxCommand, TmuxMirrorSnapshot, TmuxState};
use motlie_driver::{CommandEffect, CommandEngine};
use motlie_tmux::MonitorHealth;
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};

const BG: Color = Color::Black;
const FG: Color = Color::White;
const ACCENT: Color = Color::Yellow;
const DIM: Color = Color::Gray;
const OUTPUT_LIMIT: usize = 200;

pub enum TuiAction {
    ReturnToRepl,
    Quit,
}

struct TuiState {
    mirror: TmuxMirrorSnapshot,
    input: String,
    cursor_pos: usize,
    output_lines: VecDeque<String>,
    cmd_history: Vec<String>,
    history_pos: Option<usize>,
    saved_input: String,
    split_pct: u16,
    host_uri: String,
}

impl TuiState {
    fn new(host_uri: String, mirror: TmuxMirrorSnapshot) -> Self {
        Self {
            mirror,
            input: String::new(),
            cursor_pos: 0,
            output_lines: VecDeque::with_capacity(OUTPUT_LIMIT),
            cmd_history: Vec::new(),
            history_pos: None,
            saved_input: String::new(),
            split_pct: 30,
            host_uri,
        }
    }

    fn push_output(&mut self, line: impl Into<String>) {
        self.output_lines.push_back(line.into());
        while self.output_lines.len() > OUTPUT_LIMIT {
            self.output_lines.pop_front();
        }
    }
}

pub async fn run(engine: &mut CommandEngine<TmuxState, TmuxCommand>) -> anyhow::Result<TuiAction> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let original_hook = std::sync::Arc::new(std::panic::take_hook());
    let hook_for_panic = std::sync::Arc::clone(&original_hook);
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
        hook_for_panic(info);
    }));

    let host_uri = engine.context().host_uri.clone();
    let mirror = engine.context().mirror_snapshot();
    let mut state = TuiState::new(host_uri, mirror);
    state.push_output("tui: split-screen mode enabled");

    let result = event_loop(&mut terminal, &mut state, engine).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    let _ = std::panic::take_hook();
    if let Ok(hook) = std::sync::Arc::try_unwrap(original_hook) {
        std::panic::set_hook(hook);
    }

    engine.context_mut().shutdown_managed_state().await?;
    result
}

async fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    state: &mut TuiState,
    engine: &mut CommandEngine<TmuxState, TmuxCommand>,
) -> anyhow::Result<TuiAction> {
    loop {
        terminal.draw(|frame| draw(frame, state))?;

        let maybe_event = tokio::task::spawn_blocking(|| {
            if event::poll(Duration::from_millis(150)).unwrap_or(false) {
                event::read().ok()
            } else {
                None
            }
        })
        .await?;

        if let Some(Event::Key(key)) = maybe_event {
            if key.kind == event::KeyEventKind::Release {
                continue;
            }

            match key.code {
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    return Ok(TuiAction::Quit);
                }
                KeyCode::Char(ch) => {
                    state.input.insert(state.cursor_pos, ch);
                    state.cursor_pos += 1;
                }
                KeyCode::Backspace => {
                    if state.cursor_pos > 0 {
                        state.cursor_pos -= 1;
                        state.input.remove(state.cursor_pos);
                    }
                }
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
                KeyCode::Right => {
                    if state.cursor_pos < state.input.len() {
                        state.cursor_pos += 1;
                    }
                }
                KeyCode::Up => {
                    if state.cmd_history.is_empty() {
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
                    let command = state.input.clone();
                    state.input.clear();
                    state.cursor_pos = 0;
                    state.history_pos = None;
                    state.saved_input.clear();

                    let trimmed = command.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    if trimmed == "tui off" {
                        return Ok(TuiAction::ReturnToRepl);
                    }

                    state.cmd_history.push(trimmed.to_string());
                    state.push_output(format!("tmux> {trimmed}"));
                    let output = engine.run_line(trimmed).await?;
                    for line in output.lines {
                        state.push_output(line);
                    }
                    if output
                        .effects
                        .iter()
                        .any(|effect| matches!(effect, CommandEffect::ExitShell))
                    {
                        return Ok(TuiAction::Quit);
                    }
                }
                _ => {}
            }
        }

        engine.context_mut().refresh_mirror().await?;
        state.mirror = engine.context().mirror_snapshot();
    }
}

fn draw(frame: &mut Frame<'_>, state: &TuiState) {
    let area = frame.area();
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(state.split_pct),
            Constraint::Percentage(100 - state.split_pct),
        ])
        .split(area);

    draw_output(frame, columns[0], state);
    draw_mirror(frame, columns[1], state);
}

fn draw_output(frame: &mut Frame<'_>, area: ratatui::layout::Rect, state: &TuiState) {
    let inner_height = area.height.saturating_sub(2) as usize;
    let output_height = inner_height.saturating_sub(1);

    let mut visible = Vec::new();
    let start = state.output_lines.len().saturating_sub(output_height);
    for line in state.output_lines.iter().skip(start) {
        visible.push(Line::from(Span::raw(line.as_str())));
    }
    visible.push(Line::from(vec![
        Span::styled("tmux> ", Style::default().fg(ACCENT).bg(BG)),
        Span::styled(&state.input, Style::default().fg(FG).bg(BG)),
        Span::styled(
            "_",
            Style::default()
                .fg(ACCENT)
                .bg(BG)
                .add_modifier(Modifier::SLOW_BLINK),
        ),
    ]));

    let paragraph = Paragraph::new(visible)
        .style(Style::default().fg(FG).bg(BG))
        .block(
            Block::default()
                .title(" Driver REPL ")
                .title_style(Style::default().fg(ACCENT).bg(BG))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(ACCENT).bg(BG)),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);
}

fn draw_mirror(frame: &mut Frame<'_>, area: ratatui::layout::Rect, state: &TuiState) {
    let title = if state.mirror.label.is_empty() {
        " Mirror ".to_string()
    } else {
        format!(" Mirror — {} ", state.mirror.label)
    };

    let content = if !state.mirror.text.is_empty() {
        state.mirror.text.as_str()
    } else if state.mirror.watch_health.is_some() {
        "(waiting for output...)"
    } else {
        "(use 'monitor', 'stream', 'capture', or 'history')"
    };

    let inner_height = area.height.saturating_sub(2) as usize;
    let all_lines = content.lines().collect::<Vec<_>>();
    let visible_start = all_lines.len().saturating_sub(inner_height);
    let visible_text = all_lines[visible_start..].join("\n");

    let stream_label = match state.mirror.watch_health {
        Some(MonitorHealth::Streaming) => " stream: active ",
        Some(MonitorHealth::Reconnecting) => " stream: reconnecting ",
        Some(MonitorHealth::Failed) => " stream: failed ",
        Some(MonitorHealth::Stopped) => " stream: stopped ",
        None => " stream: idle ",
    };
    let status_style = if matches!(
        state.mirror.watch_health,
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

    if state.mirror.ansi {
        let styled = visible_text
            .into_text()
            .unwrap_or_else(|_| ratatui::text::Text::raw(visible_text.as_str()));
        let paragraph = Paragraph::new(styled)
            .style(Style::default().bg(BG))
            .block(block);
        frame.render_widget(paragraph, area);
    } else {
        let paragraph = Paragraph::new(visible_text)
            .style(Style::default().fg(FG).bg(BG))
            .block(block)
            .wrap(Wrap { trim: false });
        frame.render_widget(paragraph, area);
    }
}
