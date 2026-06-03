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
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Terminal;

use crate::operator::commands::{GatewayCommand, GatewayContext};
use crate::operator::session::{ordered_call_ids, OperatorSession};
use crate::operator::state::{CallStatus, GatewayState, LogLevel, TranscriptKind};

pub async fn run_tui(
    engine: &mut CommandEngine<GatewayContext, GatewayCommand>,
) -> anyhow::Result<()> {
    let mut terminal = TerminalGuard::enter()?;
    let mut input = String::new();
    let mut history = vec!["telnyx-gateway TUI ready".to_string()];

    loop {
        let state = engine.context().state.read().await.clone();
        engine.context_mut().session.ensure_valid_selection(&state);
        let session = engine.context().session.clone();
        terminal.draw(|frame| draw(frame, &state, &session, &history, &input))?;

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
            KeyCode::Up => select_relative(engine.context_mut(), -1).await,
            KeyCode::Down => select_relative(engine.context_mut(), 1).await,
            KeyCode::PageUp => engine.context_mut().session.scroll_detail(-8),
            KeyCode::PageDown => engine.context_mut().session.scroll_detail(8),
            KeyCode::Esc => break,
            _ => {}
        }
    }

    Ok(())
}

async fn select_relative(context: &mut GatewayContext, delta: isize) {
    let mut guard = context.state.write().await;
    let _ = context.session.move_selection(&mut guard, delta);
}

fn draw(
    frame: &mut ratatui::Frame<'_>,
    state: &GatewayState,
    session: &OperatorSession,
    history: &[String],
    input: &str,
) {
    let root = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(44), Constraint::Percentage(56)])
        .split(frame.area());
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(36), Constraint::Percentage(64)])
        .split(root[1]);

    let shell_inner_height = root[0].height.saturating_sub(2) as usize;
    let shell_inner_width = root[0].width.saturating_sub(2) as usize;
    let shell_view = build_shell_view(history, input, shell_inner_height, shell_inner_width);
    frame.render_widget(
        Paragraph::new(shell_view.lines)
            .block(Block::default().title("Shell").borders(Borders::ALL))
            .wrap(Wrap { trim: false }),
        root[0],
    );
    set_shell_cursor(frame, root[0], shell_view.prompt_row, input);

    let call_ids = ordered_call_ids(state);
    let call_items = if call_ids.is_empty() {
        vec![ListItem::new("no calls")]
    } else {
        call_ids
            .iter()
            .filter_map(|call_id| state.calls.get(call_id))
            .map(|call| {
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
                ListItem::new(format!(
                    "{} {:<13} {} -> {} [{}]",
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
    let selected_index = session
        .selected_call
        .as_ref()
        .and_then(|selected| call_ids.iter().position(|call_id| call_id == selected));
    let mut call_list_state = ListState::default();
    call_list_state.select(selected_index);
    frame.render_stateful_widget(
        List::new(call_items)
            .block(Block::default().title("Calls").borders(Borders::ALL))
            .highlight_symbol("> "),
        right[0],
        &mut call_list_state,
    );

    let detail_lines = selected_detail_lines(state, session);
    frame.render_widget(
        Paragraph::new(detail_lines)
            .block(
                Block::default()
                    .title("Selected Call")
                    .borders(Borders::ALL),
            )
            .wrap(Wrap { trim: false })
            .scroll((session.detail_scroll, 0)),
        right[1],
    );
}

struct ShellView {
    lines: Vec<Line<'static>>,
    prompt_row: u16,
}

fn build_shell_view(
    history: &[String],
    input: &str,
    inner_height: usize,
    inner_width: usize,
) -> ShellView {
    if inner_height == 0 {
        return ShellView {
            lines: Vec::new(),
            prompt_row: 0,
        };
    }

    let width = inner_width.max(1);
    let mut selected = Vec::new();
    let mut used_rows = 1usize;
    for line in history.iter().rev() {
        let rows = display_rows(line, width);
        if used_rows.saturating_add(rows) > inner_height {
            break;
        }
        selected.push(line.clone());
        used_rows += rows;
    }
    selected.reverse();

    let mut lines = selected.into_iter().map(Line::from).collect::<Vec<_>>();
    lines.push(Line::from(vec![
        Span::styled("> ", Style::default().fg(Color::Cyan)),
        Span::raw(input.to_string()),
    ]));

    ShellView {
        lines,
        prompt_row: used_rows.saturating_sub(1) as u16,
    }
}

fn display_rows(text: &str, width: usize) -> usize {
    let width = width.max(1);
    let chars = text.chars().count().max(1);
    chars.div_ceil(width)
}

fn set_shell_cursor(
    frame: &mut ratatui::Frame<'_>,
    area: ratatui::layout::Rect,
    prompt_row: u16,
    input: &str,
) {
    if area.width <= 2 || area.height <= 2 {
        return;
    }

    let inner_width = area.width.saturating_sub(2);
    let prompt_y = area.y + 1 + prompt_row.min(area.height.saturating_sub(3));
    let prompt_offset = 2 + input.chars().count() as u16;
    let cursor_offset = prompt_offset.min(inner_width.saturating_sub(1));
    frame.set_cursor_position(Position::new(area.x + 1 + cursor_offset, prompt_y));
}

fn selected_detail_lines(state: &GatewayState, session: &OperatorSession) -> Vec<Line<'static>> {
    let Some(call) = session.selected_call(state) else {
        return status_lines(state, session);
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
        Line::from(format!(
            "asr: {}",
            call.asr_backend
                .map(|backend| format!("{} ({})", backend.label(), backend.model_label()))
                .unwrap_or_else(|| "<unbound>".to_string())
        )),
    ];
    if let Some(reason) = &call.terminal_reason {
        lines.push(Line::from(format!("ended: {reason}")));
    }
    lines.extend([
        Line::from(""),
        Line::from("assembled transcript:"),
        Line::from(call.assembled_transcript_text()),
        Line::from(""),
        Line::from("recent events:"),
    ]);

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

fn status_lines(state: &GatewayState, session: &OperatorSession) -> Vec<Line<'static>> {
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
            "asr next: {} ({})",
            session.next_asr_backend.label(),
            session.next_asr_backend.model_label()
        )),
        Line::from(format!(
            "asr default: {} ({})",
            state.config.asr_backend.label(),
            state.config.asr_backend.model_label()
        )),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_rows_counts_wrapped_terminal_rows() {
        assert_eq!(display_rows("", 8), 1);
        assert_eq!(display_rows("12345678", 8), 1);
        assert_eq!(display_rows("123456789", 8), 2);
    }

    #[test]
    fn shell_prompt_row_accounts_for_wrapped_history() {
        let history = vec!["short".to_string(), "1234567890".to_string()];
        let view = build_shell_view(&history, "", 4, 5);

        assert_eq!(view.prompt_row, 3);
        assert_eq!(view.lines.len(), 3);
    }

    #[test]
    fn shell_prompt_stays_visible_when_recent_output_is_too_tall() {
        let history = vec!["12345678901234567890".to_string()];
        let view = build_shell_view(&history, "", 3, 4);

        assert_eq!(view.prompt_row, 0);
        assert_eq!(view.lines.len(), 1);
    }
}
