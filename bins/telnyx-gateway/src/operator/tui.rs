use std::io::{self, Stdout};
use std::time::Duration;

use chrono::{DateTime, Local, Utc};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use motlie_driver::{CommandEffect, CommandEngine};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap};
use ratatui::Terminal;

use crate::operator::commands::{GatewayCommand, GatewayContext};
use crate::operator::script::run_operator_line;
use crate::operator::session::{ordered_call_ids, OperatorSession};
use crate::operator::state::{CallStatus, GatewayState, LogLevel};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FocusedPane {
    Shell,
    Calls,
    Detail,
}

impl FocusedPane {
    fn next(self) -> Self {
        match self {
            Self::Shell => Self::Calls,
            Self::Calls => Self::Detail,
            Self::Detail => Self::Shell,
        }
    }
}

#[derive(Clone, Debug, Default)]
struct ShellInput {
    text: String,
    cursor_chars: usize,
}

impl ShellInput {
    fn as_str(&self) -> &str {
        &self.text
    }

    fn clear(&mut self) {
        self.text.clear();
        self.cursor_chars = 0;
    }

    fn insert(&mut self, ch: char) {
        let byte_index = char_to_byte_index(&self.text, self.cursor_chars);
        self.text.insert(byte_index, ch);
        self.cursor_chars = self.cursor_chars.saturating_add(1);
    }

    fn backspace(&mut self) {
        if self.cursor_chars == 0 {
            return;
        }
        let start = char_to_byte_index(&self.text, self.cursor_chars - 1);
        let end = char_to_byte_index(&self.text, self.cursor_chars);
        self.text.replace_range(start..end, "");
        self.cursor_chars -= 1;
    }

    fn delete(&mut self) {
        if self.cursor_chars >= self.text.chars().count() {
            return;
        }
        let start = char_to_byte_index(&self.text, self.cursor_chars);
        let end = char_to_byte_index(&self.text, self.cursor_chars + 1);
        self.text.replace_range(start..end, "");
    }

    fn move_left(&mut self) {
        self.cursor_chars = self.cursor_chars.saturating_sub(1);
    }

    fn move_right(&mut self) {
        self.cursor_chars = self
            .cursor_chars
            .saturating_add(1)
            .min(self.text.chars().count());
    }

    fn move_home(&mut self) {
        self.cursor_chars = 0;
    }

    fn move_end(&mut self) {
        self.cursor_chars = self.text.chars().count();
    }

    fn cursor_chars(&self) -> usize {
        self.cursor_chars
    }
}

fn char_to_byte_index(text: &str, cursor_chars: usize) -> usize {
    text.char_indices()
        .nth(cursor_chars)
        .map(|(index, _)| index)
        .unwrap_or(text.len())
}

pub async fn run_tui(
    engine: &mut CommandEngine<GatewayContext, GatewayCommand>,
) -> anyhow::Result<()> {
    let mut terminal = TerminalGuard::enter()?;
    let host = host_label();
    let mut input = ShellInput::default();
    let mut history = vec!["telnyx-gateway TUI ready".to_string()];
    let mut focus = FocusedPane::Shell;

    loop {
        let state = engine.context().state.read().await.clone();
        engine.context_mut().session.ensure_valid_selection(&state);
        let session = engine.context().session.clone();
        terminal.draw(|frame| draw(frame, &state, &session, &history, &input, focus, &host))?;

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
            KeyCode::Tab => focus = focus.next(),
            KeyCode::Enter if focus == FocusedPane::Shell => {
                let command = input.as_str().trim().to_string();
                input.clear();
                if command.is_empty() {
                    continue;
                }
                history.push(format!("> {command}"));
                match run_operator_line(engine, &command).await {
                    Ok(output) => {
                        history.extend(output.lines);
                        if output.effects.contains(&CommandEffect::ExitShell) {
                            break;
                        }
                    }
                    Err(error) => history.push(format!("error: {error}")),
                }
                trim_history(&mut history);
            }
            KeyCode::Char(ch) => match focus {
                FocusedPane::Shell
                    if !key
                        .modifiers
                        .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
                {
                    input.insert(ch);
                }
                FocusedPane::Calls if ch == 'a' => {
                    let message = attach_selected(engine.context_mut()).await;
                    history.push(message);
                    trim_history(&mut history);
                }
                FocusedPane::Detail if ch == 'u' => engine.context_mut().session.scroll_detail(-1),
                FocusedPane::Detail if ch == 'b' => engine.context_mut().session.scroll_detail(1),
                _ => {}
            },
            KeyCode::Backspace if focus == FocusedPane::Shell => {
                input.backspace();
            }
            KeyCode::Delete if focus == FocusedPane::Shell => {
                input.delete();
            }
            KeyCode::Left if focus == FocusedPane::Shell => {
                input.move_left();
            }
            KeyCode::Right if focus == FocusedPane::Shell => {
                input.move_right();
            }
            KeyCode::Home if focus == FocusedPane::Shell => {
                input.move_home();
            }
            KeyCode::End if focus == FocusedPane::Shell => {
                input.move_end();
            }
            KeyCode::Up => match focus {
                FocusedPane::Calls => select_relative(engine.context_mut(), -1).await,
                FocusedPane::Detail => engine.context_mut().session.scroll_detail(-1),
                FocusedPane::Shell => {}
            },
            KeyCode::Down => match focus {
                FocusedPane::Calls => select_relative(engine.context_mut(), 1).await,
                FocusedPane::Detail => engine.context_mut().session.scroll_detail(1),
                FocusedPane::Shell => {}
            },
            KeyCode::PageUp if focus == FocusedPane::Detail => {
                engine.context_mut().session.scroll_detail(-8);
            }
            KeyCode::PageDown if focus == FocusedPane::Detail => {
                engine.context_mut().session.scroll_detail(8);
            }
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

async fn attach_selected(context: &mut GatewayContext) -> String {
    let Some(call_id) = context.session.selected_call.clone() else {
        return "no call selected".to_string();
    };
    let mut guard = context.state.write().await;
    if context.session.select_call(&mut guard, &call_id) {
        format!("attached to {call_id}")
    } else {
        "selected call no longer exists".to_string()
    }
}

fn trim_history(history: &mut Vec<String>) {
    while history.len() > 200 {
        history.remove(0);
    }
}

fn draw(
    frame: &mut ratatui::Frame<'_>,
    state: &GatewayState,
    session: &OperatorSession,
    history: &[String],
    input: &ShellInput,
    focus: FocusedPane,
    host: &str,
) {
    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(1)])
        .split(frame.area());
    render_status_bar(frame, outer[0], state, session, host);

    let root = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(44), Constraint::Percentage(56)])
        .split(outer[1]);
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(36), Constraint::Percentage(64)])
        .split(root[1]);

    let shell_inner_height = root[0].height.saturating_sub(2) as usize;
    let shell_inner_width = root[0].width.saturating_sub(2) as usize;
    let shell_view = build_shell_view(
        history,
        input.as_str(),
        shell_inner_height,
        shell_inner_width,
    );
    frame.render_widget(
        Paragraph::new(shell_view.lines)
            .block(focused_block("Shell", FocusedPane::Shell, focus))
            .wrap(Wrap { trim: false }),
        root[0],
    );
    if focus == FocusedPane::Shell {
        set_shell_cursor(frame, root[0], shell_view.prompt_row, input);
    }

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
            .block(focused_block("Calls", FocusedPane::Calls, focus))
            .highlight_symbol("> "),
        right[0],
        &mut call_list_state,
    );

    let detail_lines = selected_detail_lines(state, session);
    frame.render_widget(
        Paragraph::new(detail_lines)
            .block(focused_block("Selected Call", FocusedPane::Detail, focus))
            .wrap(Wrap { trim: false })
            .scroll((session.detail_scroll, 0)),
        right[1],
    );
}

const STATUS_BAR_BG: Color = Color::Rgb(0, 43, 85);
const STATUS_BAR_FG: Color = Color::White;
const STATUS_BAR_KEY_FG: Color = Color::LightCyan;

fn render_status_bar(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    state: &GatewayState,
    session: &OperatorSession,
    host: &str,
) {
    let now = Local::now();
    let line = status_bar_line(state, session, host, now, area.width as usize);
    frame.render_widget(
        Paragraph::new(line).style(Style::default().fg(STATUS_BAR_FG).bg(STATUS_BAR_BG)),
        area,
    );
}

fn status_bar_line(
    state: &GatewayState,
    session: &OperatorSession,
    host: &str,
    now: DateTime<Local>,
    width: usize,
) -> Line<'static> {
    let right_text = status_bar_right_text(state, now);
    let right_text = truncate_chars(&right_text, right_text.chars().count().min(width));
    let max_left_width = width.saturating_sub(right_text.chars().count());
    let left_text = truncate_chars(&status_bar_left_text(state, session, host), max_left_width);
    let padding = width.saturating_sub(left_text.chars().count() + right_text.chars().count());

    Line::from(vec![
        Span::styled(
            left_text,
            Style::default()
                .fg(STATUS_BAR_FG)
                .bg(STATUS_BAR_BG)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            " ".repeat(padding),
            Style::default().fg(STATUS_BAR_FG).bg(STATUS_BAR_BG),
        ),
        Span::styled(
            right_text,
            Style::default().fg(STATUS_BAR_KEY_FG).bg(STATUS_BAR_BG),
        ),
    ])
}

fn status_bar_left_text(state: &GatewayState, session: &OperatorSession, host: &str) -> String {
    format!(
        " telnyx | {} | {} | {} | {} | {}",
        host,
        state
            .config
            .public_webhook_url
            .as_deref()
            .unwrap_or("<webhook unset>"),
        state
            .config
            .public_media_url
            .as_deref()
            .unwrap_or("<media unset>"),
        state.inbound_mode.label(),
        session.next_asr_backend.label()
    )
}

fn status_bar_right_text(state: &GatewayState, now: DateTime<Local>) -> String {
    let now_utc = now.with_timezone(&Utc);
    let age = format_age(now_utc.signed_duration_since(state.started_at));
    let started = state
        .started_at
        .with_timezone(&Local)
        .format("%m-%d %H:%M:%S")
        .to_string();
    format!(" {started} / {} / {age} ", now.format("%H:%M:%S"))
}

fn format_age(age: chrono::Duration) -> String {
    let total_seconds = age.num_seconds().max(0);
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    if hours > 0 {
        format!("{hours}h{minutes:02}m{seconds:02}s")
    } else if minutes > 0 {
        format!("{minutes}m{seconds:02}s")
    } else {
        format!("{seconds}s")
    }
}

fn truncate_chars(text: &str, width: usize) -> String {
    text.chars().take(width).collect()
}

fn host_label() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| {
            std::fs::read_to_string("/etc/hostname")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        })
        .unwrap_or_else(|| "<unknown>".to_string())
}

fn focused_block(title: &'static str, pane: FocusedPane, focus: FocusedPane) -> Block<'static> {
    let block = Block::default().title(title).borders(Borders::ALL);
    if pane == focus {
        block.border_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
    } else {
        block
    }
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
    area: Rect,
    prompt_row: u16,
    input: &ShellInput,
) {
    if area.width <= 2 || area.height <= 2 {
        return;
    }

    let inner_width = area.width.saturating_sub(2);
    let prompt_y = area.y + 1 + prompt_row.min(area.height.saturating_sub(3));
    let prompt_offset = 2 + input.cursor_chars() as u16;
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
        Line::from("assembled transcription:"),
        Line::from(call.assembled_transcript_text()),
    ]);
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
    use chrono::TimeZone;

    #[test]
    fn display_rows_counts_wrapped_terminal_rows() {
        assert_eq!(display_rows("", 8), 1);
        assert_eq!(display_rows("12345678", 8), 1);
        assert_eq!(display_rows("123456789", 8), 2);
    }

    #[test]
    fn focused_pane_cycles_through_tui_regions() {
        assert_eq!(FocusedPane::Shell.next(), FocusedPane::Calls);
        assert_eq!(FocusedPane::Calls.next(), FocusedPane::Detail);
        assert_eq!(FocusedPane::Detail.next(), FocusedPane::Shell);
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

    #[test]
    fn shell_input_edits_at_cursor() {
        let mut input = ShellInput::default();
        for ch in "abcd".chars() {
            input.insert(ch);
        }

        input.move_left();
        input.move_left();
        input.insert('X');

        assert_eq!(input.as_str(), "abXcd");
        assert_eq!(input.cursor_chars(), 3);
    }

    #[test]
    fn shell_input_backspace_and_delete_are_unicode_safe() {
        let mut input = ShellInput::default();
        for ch in "aé日b".chars() {
            input.insert(ch);
        }

        input.move_left();
        input.backspace();
        input.delete();

        assert_eq!(input.as_str(), "aé");
        assert_eq!(input.cursor_chars(), 2);
    }

    #[test]
    fn status_bar_left_text_uses_compact_unlabeled_fields() {
        let mut state = GatewayState::new("127.0.0.1:8080".parse().expect("valid addr"));
        state.config.public_webhook_url = Some("https://example.test/telnyx/webhooks".to_string());
        state.config.public_media_url = Some("wss://example.test/telnyx/media".to_string());
        state.config.selected_connection_id = Some("conn-1".to_string());
        state.config.selected_phone_number = Some("+15550000001".to_string());
        let session = OperatorSession::new(state.config.asr_backend);

        let text = status_bar_left_text(&state, &session, "host-a");

        assert_eq!(
            text,
            " telnyx | host-a | https://example.test/telnyx/webhooks | wss://example.test/telnyx/media | disabled | kroko-2025"
        );
        assert!(!text.contains("host="));
        assert!(!text.contains("bind="));
        assert!(!text.contains("webhook="));
        assert!(!text.contains("media="));
        assert!(!text.contains("conn-1"));
        assert!(!text.contains("+15550000001"));
    }

    #[test]
    fn status_bar_right_text_uses_start_now_age_order() {
        let mut state = GatewayState::new("127.0.0.1:8080".parse().expect("valid addr"));
        state.started_at = Utc
            .with_ymd_and_hms(2026, 6, 3, 20, 0, 0)
            .single()
            .expect("valid timestamp");
        let now = state.started_at.with_timezone(&Local) + chrono::Duration::seconds(65);

        let text = status_bar_right_text(&state, now);

        assert!(text.contains(" / "));
        assert!(text.ends_with(" / 1m05s "));
        assert!(!text.contains("start="));
        assert!(!text.contains("now="));
        assert!(!text.contains("age="));
    }

    #[test]
    fn status_age_is_compact() {
        assert_eq!(format_age(chrono::Duration::seconds(-1)), "0s");
        assert_eq!(format_age(chrono::Duration::seconds(9)), "9s");
        assert_eq!(format_age(chrono::Duration::seconds(65)), "1m05s");
        assert_eq!(format_age(chrono::Duration::seconds(3661)), "1h01m01s");
    }
}
