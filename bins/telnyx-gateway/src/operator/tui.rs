use std::io::{self, Stdout};
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use motlie_driver::{CommandEffect, CommandEngine};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Position};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Wrap};
use ratatui::Terminal;

use crate::operator::commands::{GatewayCommand, GatewayContext};
use crate::operator::state::{CallStatus, GatewayState, LogLevel, SharedState, TranscriptKind};

pub async fn run_tui(
    engine: &mut CommandEngine<GatewayContext, GatewayCommand>,
) -> anyhow::Result<()> {
    let mut terminal = TerminalGuard::enter()?;
    let mut input = String::new();
    let mut history = vec!["telnyx-gateway TUI ready".to_string()];

    loop {
        let state = engine.context().state.read().await.clone();
        terminal.draw(|frame| draw(frame, &state, &history, &input))?;

        if state.shutdown_requested {
            break;
        }

        if !event::poll(Duration::from_millis(100))? {
            continue;
        }

        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }

        match key.code {
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break,
            KeyCode::Char('d') if key.modifiers.contains(KeyModifiers::CONTROL) => break,
            KeyCode::Char(ch) => input.push(ch),
            KeyCode::Backspace => {
                let _ = input.pop();
            }
            KeyCode::Enter => {
                let command = input.trim().to_string();
                input.clear();
                if command.is_empty() {
                    continue;
                }
                history.push(format!("> {command}"));
                match engine.run_line(&command).await {
                    Ok(output) => {
                        history.extend(output.lines);
                        if output.effects.contains(&CommandEffect::ExitShell) {
                            break;
                        }
                    }
                    Err(error) => history.push(format!("error: {error}")),
                }
                while history.len() > 200 {
                    history.remove(0);
                }
            }
            KeyCode::Up => select_relative(&engine.context().state, -1).await,
            KeyCode::Down => select_relative(&engine.context().state, 1).await,
            KeyCode::Esc => break,
            _ => {}
        }
    }

    Ok(())
}

async fn select_relative(state: &SharedState, delta: isize) {
    let mut guard = state.write().await;
    if guard.calls.is_empty() {
        return;
    }

    let call_ids = guard.calls.keys().cloned().collect::<Vec<_>>();
    let current_index = guard
        .selected_call
        .as_ref()
        .and_then(|selected| call_ids.iter().position(|call_id| call_id == selected))
        .unwrap_or(0);
    let next_index = if delta < 0 {
        current_index.saturating_sub(1)
    } else {
        (current_index + 1).min(call_ids.len() - 1)
    };
    let selected = call_ids[next_index].clone();
    guard.selected_call = Some(selected.clone());
    if let Some(call) = guard.calls.get_mut(&selected) {
        call.unread_events = 0;
    }
}

fn draw(frame: &mut ratatui::Frame<'_>, state: &GatewayState, history: &[String], input: &str) {
    let root = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(44), Constraint::Percentage(56)])
        .split(frame.area());
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(36), Constraint::Percentage(64)])
        .split(root[1]);

    let shell_height = root[0].height.saturating_sub(2) as usize;
    let history_capacity = shell_height.saturating_sub(1);
    let mut shell_lines = history
        .iter()
        .rev()
        .take(history_capacity)
        .rev()
        .cloned()
        .map(Line::from)
        .collect::<Vec<_>>();
    shell_lines.push(Line::from(vec![
        Span::styled("> ", Style::default().fg(Color::Cyan)),
        Span::raw(input.to_string()),
    ]));
    frame.render_widget(
        Paragraph::new(shell_lines)
            .block(Block::default().title("Shell").borders(Borders::ALL))
            .wrap(Wrap { trim: false }),
        root[0],
    );
    set_shell_cursor(frame, root[0], history_capacity, history.len(), input);

    let call_items = if state.calls.is_empty() {
        vec![ListItem::new("no calls")]
    } else {
        state
            .calls
            .values()
            .map(|call| {
                let selected = state.selected_call.as_deref() == Some(&call.gateway_call_id);
                let style = match call.status {
                    CallStatus::PendingInbound => Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                    CallStatus::Failed => Style::default().fg(Color::Red),
                    CallStatus::Transcribing | CallStatus::MediaStarted => {
                        Style::default().fg(Color::Green)
                    }
                    _ => Style::default(),
                };
                let marker = if selected { "> " } else { "  " };
                ListItem::new(format!(
                    "{marker}{} {:<13} {} -> {} [{}]",
                    call.gateway_call_id,
                    call.status.label(),
                    call.from.as_deref().unwrap_or("?"),
                    call.to.as_deref().unwrap_or("?"),
                    call.unread_events
                ))
                .style(style)
            })
            .collect::<Vec<_>>()
    };
    frame.render_widget(
        List::new(call_items).block(Block::default().title("Calls").borders(Borders::ALL)),
        right[0],
    );

    let detail_lines = selected_detail_lines(state);
    frame.render_widget(
        Paragraph::new(detail_lines)
            .block(
                Block::default()
                    .title("Selected Call")
                    .borders(Borders::ALL),
            )
            .wrap(Wrap { trim: false }),
        right[1],
    );
}

fn set_shell_cursor(
    frame: &mut ratatui::Frame<'_>,
    area: ratatui::layout::Rect,
    history_capacity: usize,
    history_len: usize,
    input: &str,
) {
    if area.width <= 2 || area.height <= 2 {
        return;
    }

    let visible_history = history_len.min(history_capacity) as u16;
    let prompt_y = area.y + 1 + visible_history;
    let inner_width = area.width.saturating_sub(2);
    let prompt_offset = 2 + input.chars().count() as u16;
    let cursor_offset = prompt_offset.min(inner_width.saturating_sub(1));
    frame.set_cursor_position(Position::new(area.x + 1 + cursor_offset, prompt_y));
}

fn selected_detail_lines(state: &GatewayState) -> Vec<Line<'static>> {
    let Some(call_id) = state.selected_call.as_deref() else {
        return status_lines(state);
    };
    let Some(call) = state.calls.get(call_id) else {
        return status_lines(state);
    };

    let mut lines = vec![
        Line::from(format!("call: {}", call.gateway_call_id)),
        Line::from(format!("state: {}", call.status.label())),
        Line::from(format!("call_control_id: {}", call.ids.call_control_id)),
        Line::from(format!(
            "call_session_id: {}",
            call.ids.call_session_id.as_deref().unwrap_or("<none>")
        )),
        Line::from(format!(
            "call_leg_id: {}",
            call.ids.call_leg_id.as_deref().unwrap_or("<none>")
        )),
        Line::from(format!(
            "stream_id: {}",
            call.ids.stream_id.as_deref().unwrap_or("<none>")
        )),
        Line::from(format!(
            "media: {} {}Hz {}ch",
            call.media.encoding.as_deref().unwrap_or("<unknown>"),
            call.media
                .sample_rate_hz
                .map(|value| value.to_string())
                .unwrap_or_else(|| "?".to_string()),
            call.media
                .channels
                .map(|value| value.to_string())
                .unwrap_or_else(|| "?".to_string())
        )),
        Line::from(""),
        Line::from("transcript:"),
    ];

    for transcript in call.transcripts.iter().rev().take(12).rev() {
        let prefix = match transcript.kind {
            TranscriptKind::Partial => "partial",
            TranscriptKind::Final => "final",
        };
        lines.push(Line::from(format!("{prefix}: {}", transcript.text)));
    }
    if let Some(error) = &call.last_error {
        lines.push(Line::from(""));
        lines.push(Line::from(format!("error: {error}")));
    }
    lines
}

fn status_lines(state: &GatewayState) -> Vec<Line<'static>> {
    let mut lines = vec![
        Line::from(format!(
            "listener: {}",
            state
                .config
                .bind
                .map(|addr| addr.to_string())
                .unwrap_or_else(|| "<unknown>".to_string())
        )),
        Line::from(format!("inbound: {}", state.inbound_mode.label())),
        Line::from(format!(
            "webhook: {}",
            state
                .config
                .public_webhook_url
                .as_deref()
                .unwrap_or("<unset>")
        )),
        Line::from(format!(
            "media: {}",
            state
                .config
                .public_media_url
                .as_deref()
                .unwrap_or("<unset>")
        )),
        Line::from(""),
        Line::from("recent log:"),
    ];
    for log in state.logs.iter().rev().take(10).rev() {
        let level = match log.level {
            LogLevel::Info => "info",
            LogLevel::Warn => "warn",
            LogLevel::Error => "error",
        };
        lines.push(Line::from(format!("{level}: {}", log.message)));
    }
    lines
}

struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl TerminalGuard {
    fn enter() -> anyhow::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let terminal = Terminal::new(CrosstermBackend::new(stdout))?;
        Ok(Self { terminal })
    }

    fn draw<F>(&mut self, f: F) -> anyhow::Result<ratatui::CompletedFrame<'_>>
    where
        F: FnOnce(&mut ratatui::Frame<'_>),
    {
        Ok(self.terminal.draw(f)?)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}
