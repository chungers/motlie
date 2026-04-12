#[cfg(feature = "repl")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "tui")]
use std::{collections::VecDeque, io, time::Duration};

#[cfg(feature = "tui")]
use ansi_to_tui::IntoText;
#[cfg(feature = "tui")]
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
#[cfg(feature = "tui")]
use ratatui::{
    Frame, Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span as TuiSpan},
    widgets::{Block, Borders, Paragraph, Wrap},
};
#[cfg(feature = "repl")]
use reedline::{
    Completer, DefaultPrompt, DefaultPromptSegment, Reedline, Signal, Span, Suggestion,
};

#[cfg(feature = "repl")]
use crate::clap::analyze_completion;
#[cfg(feature = "tui")]
use crate::commands::tmux::TmuxMirrorSnapshot;
use crate::commands::tmux::{TmuxCommand, TmuxState};
#[cfg(feature = "repl")]
use crate::completion::{CompletionCandidate, CompletionRequest, dedup_sorted};
#[cfg(feature = "repl")]
use crate::engine::CommandSet;
use crate::engine::{CommandEffect, CommandEngine};
use crate::error::DriverResult;
use crate::term::asciicast::AsciicastRecorder;

#[cfg(feature = "repl")]
const HISTORY_PAGE_SIZE: usize = 32;

#[cfg(feature = "tui")]
const BG: Color = Color::Black;
#[cfg(feature = "tui")]
const FG: Color = Color::White;
#[cfg(feature = "tui")]
const ACCENT: Color = Color::Yellow;
#[cfg(feature = "tui")]
const DIM: Color = Color::Gray;
#[cfg(feature = "tui")]
const OUTPUT_LIMIT: usize = 200;

pub enum TuiAction {
    ReturnToRepl,
    Quit,
}

#[cfg(not(feature = "tui"))]
pub async fn run_tmux_tui(
    _engine: &mut CommandEngine<TmuxState, TmuxCommand>,
    _recorder: &mut Option<AsciicastRecorder>,
) -> DriverResult<TuiAction> {
    Err(crate::error::DriverError::message(
        "tmux TUI frontend requires the `tui` feature",
    ))
}

#[cfg(feature = "repl")]
pub struct TmuxCompleter {
    context: Arc<Mutex<<TmuxCommand as CommandSet<TmuxState>>::CompletionContext>>,
}

#[cfg(feature = "repl")]
impl TmuxCompleter {
    pub fn new(
        context: Arc<Mutex<<TmuxCommand as CommandSet<TmuxState>>::CompletionContext>>,
    ) -> Self {
        Self { context }
    }
}

#[cfg(feature = "repl")]
impl Completer for TmuxCompleter {
    fn complete(&mut self, line: &str, pos: usize) -> Vec<Suggestion> {
        let context = match self.context.lock() {
            Ok(guard) => guard,
            Err(_) => return Vec::new(),
        };

        let root = TmuxCommand::root_command();
        let completion = analyze_completion(&root, line, pos);
        let path_refs = completion
            .command_path
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let mut candidates = completion.static_candidates;

        if path_refs.is_empty() {
            for builtin in ["help", "quit"] {
                if builtin.starts_with(&completion.prefix) {
                    candidates.push(CompletionCandidate::new(builtin));
                }
            }
        }

        candidates.extend(TmuxCommand::complete(
            CompletionRequest {
                command_path: &path_refs,
                arg_id: completion.arg_id.as_deref(),
                prefix: &completion.prefix,
            },
            &context,
        ));

        let candidates = dedup_sorted(candidates);
        let start = line
            .get(..pos)
            .and_then(|prefix| {
                prefix
                    .rmatch_indices(char::is_whitespace)
                    .next()
                    .map(|(index, matched)| index + matched.len())
            })
            .unwrap_or(0);

        candidates
            .into_iter()
            .map(|candidate| Suggestion {
                value: candidate.value,
                description: candidate.help,
                style: None,
                extra: None,
                span: Span::new(start, pos),
                append_whitespace: true,
            })
            .collect()
    }
}

#[cfg(feature = "repl")]
pub async fn run_tmux_repl(
    engine: &mut CommandEngine<TmuxState, TmuxCommand>,
    recorder: &mut Option<AsciicastRecorder>,
    host_uri: &str,
) -> DriverResult<()> {
    let completion_context = Arc::new(Mutex::new(engine.completion_context()));
    let completer = Box::new(TmuxCompleter::new(completion_context.clone()));
    let mut line_editor = Reedline::create().with_completer(completer);
    let prompt = DefaultPrompt::new(
        DefaultPromptSegment::Basic("tmux> ".to_string()),
        DefaultPromptSegment::Empty,
    );
    record_output(recorder, &format!("Connected to {host_uri}\n"))?;

    loop {
        match line_editor.read_line(&prompt)? {
            Signal::Success(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                record_output(recorder, &format!("tmux> {trimmed}\n"))?;
                record_input(recorder, &format!("{trimmed}\n"))?;
                let output = engine.run_line(trimmed).await?;
                for line in &output.lines {
                    println!("{line}");
                    record_output(recorder, &format!("{line}\n"))?;
                }

                if let Ok(mut guard) = completion_context.lock() {
                    *guard = engine.completion_context();
                }

                if engine.context().has_live_follow() {
                    run_live_follow(engine, recorder).await?;
                    if let Ok(mut guard) = completion_context.lock() {
                        *guard = engine.completion_context();
                    }
                }

                if let Some(effect) = output.effects.first() {
                    match effect {
                        CommandEffect::EnterTui => match run_tmux_tui(engine, recorder).await? {
                            TuiAction::Quit => return Ok(()),
                            TuiAction::ReturnToRepl => {}
                        },
                        CommandEffect::ExitShell => return Ok(()),
                        CommandEffect::ExitTui => {}
                    }
                }
            }
            Signal::CtrlC | Signal::CtrlD => return Ok(()),
        }
    }
}

#[cfg(feature = "repl")]
async fn run_live_follow(
    engine: &mut CommandEngine<TmuxState, TmuxCommand>,
    recorder: &mut Option<AsciicastRecorder>,
) -> DriverResult<()> {
    println!("Live follow attached. Press Ctrl-C to stop and return to the prompt.");
    record_output(
        recorder,
        "Live follow attached. Press Ctrl-C to stop and return to the prompt.\n",
    )?;
    let mut previous = String::new();
    let mut cursor = None;

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                engine.context_mut().shutdown_managed_state().await?;
                println!();
                println!("Live follow stopped.");
                record_output(recorder, "^C\nLive follow stopped.\n")?;
                break;
            }
            _ = tokio::time::sleep(std::time::Duration::from_millis(150)) => {}
        }

        engine.context_mut().refresh_mirror().await?;
        let page = engine
            .context()
            .mirror_history_page(cursor, HISTORY_PAGE_SIZE);
        for record in page.items {
            render_incremental(&mut previous, &record.item.text);
            record_output(recorder, &record.item.text)?;
            if !record.item.text.ends_with('\n') {
                record_output(recorder, "\n")?;
            }
            cursor = Some(record.seq);
        }

        if !engine.context().has_live_follow() {
            break;
        }
    }
    Ok(())
}

#[cfg(feature = "tui")]
struct TuiState {
    mirror: TmuxMirrorSnapshot,
    input: String,
    cursor_chars: usize,
    output_lines: VecDeque<String>,
    cmd_history: Vec<String>,
    history_pos: Option<usize>,
    saved_input: String,
    split_pct: u16,
    host_uri: String,
}

#[cfg(feature = "tui")]
impl TuiState {
    fn new(host_uri: String, mirror: TmuxMirrorSnapshot) -> Self {
        Self {
            mirror,
            input: String::new(),
            cursor_chars: 0,
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

    fn replace_input(&mut self, new_input: String) {
        self.input = new_input;
        self.cursor_chars = self.input.chars().count();
    }
}

#[cfg(feature = "tui")]
pub async fn run_tmux_tui(
    engine: &mut CommandEngine<TmuxState, TmuxCommand>,
    recorder: &mut Option<AsciicastRecorder>,
) -> DriverResult<TuiAction> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    if let Ok(size) = terminal.size() {
        record_resize(recorder, size.width, size.height)?;
    }
    record_output(recorder, "[tui] entered split-screen mode\n")?;

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

    let result = tmux_tui_event_loop(&mut terminal, &mut state, engine, recorder).await;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    let _ = std::panic::take_hook();
    if let Ok(hook) = std::sync::Arc::try_unwrap(original_hook) {
        std::panic::set_hook(hook);
    }

    engine.context_mut().shutdown_managed_state().await?;
    record_output(recorder, "[tui] exited split-screen mode\n")?;
    result
}

#[cfg(feature = "tui")]
async fn tmux_tui_event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    state: &mut TuiState,
    engine: &mut CommandEngine<TmuxState, TmuxCommand>,
    recorder: &mut Option<AsciicastRecorder>,
) -> DriverResult<TuiAction> {
    loop {
        terminal.draw(|frame| draw_tmux_tui(frame, state))?;

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
                KeyCode::Char(ch) => insert_char(&mut state.input, &mut state.cursor_chars, ch),
                KeyCode::Backspace => {
                    delete_previous_char(&mut state.input, &mut state.cursor_chars)
                }
                KeyCode::Delete => delete_current_char(&mut state.input, &state.cursor_chars),
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
                KeyCode::Left => state.cursor_chars = state.cursor_chars.saturating_sub(1),
                KeyCode::Right => {
                    let max_chars = state.input.chars().count();
                    if state.cursor_chars < max_chars {
                        state.cursor_chars += 1;
                    }
                }
                KeyCode::Home => state.cursor_chars = 0,
                KeyCode::End => state.cursor_chars = state.input.chars().count(),
                KeyCode::Up => {
                    if state.cmd_history.is_empty() {
                    } else if let Some(pos) = state.history_pos {
                        if pos > 0 {
                            state.history_pos = Some(pos - 1);
                            state.replace_input(state.cmd_history[pos - 1].clone());
                        }
                    } else {
                        state.saved_input = state.input.clone();
                        let pos = state.cmd_history.len() - 1;
                        state.history_pos = Some(pos);
                        state.replace_input(state.cmd_history[pos].clone());
                    }
                }
                KeyCode::Down => {
                    if let Some(pos) = state.history_pos {
                        if pos + 1 < state.cmd_history.len() {
                            state.history_pos = Some(pos + 1);
                            state.replace_input(state.cmd_history[pos + 1].clone());
                        } else {
                            state.history_pos = None;
                            state.replace_input(state.saved_input.clone());
                        }
                    }
                }
                KeyCode::Enter => {
                    let command = state.input.clone();
                    state.input.clear();
                    state.cursor_chars = 0;
                    state.history_pos = None;
                    state.saved_input.clear();

                    let trimmed = command.trim();
                    if trimmed.is_empty() {
                        continue;
                    }

                    state.cmd_history.push(trimmed.to_string());
                    state.push_output(format!("tmux> {trimmed}"));
                    record_input(recorder, &format!("{trimmed}\n"))?;
                    let output = engine.run_line(trimmed).await?;
                    for line in output.lines {
                        state.push_output(line.clone());
                        record_output(recorder, &format!("{line}\n"))?;
                    }
                    if let Some(effect) = output.effects.first() {
                        match effect {
                            CommandEffect::ExitShell => return Ok(TuiAction::Quit),
                            CommandEffect::ExitTui => return Ok(TuiAction::ReturnToRepl),
                            CommandEffect::EnterTui => {}
                        }
                    }
                }
                _ => {}
            }
        }

        engine.context_mut().refresh_mirror().await?;
        state.mirror = engine.context().mirror_snapshot();
    }
}

#[cfg(feature = "tui")]
fn draw_tmux_tui(frame: &mut Frame<'_>, state: &TuiState) {
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

#[cfg(feature = "tui")]
fn draw_output(frame: &mut Frame<'_>, area: ratatui::layout::Rect, state: &TuiState) {
    let inner_height = area.height.saturating_sub(2) as usize;
    let output_height = inner_height.saturating_sub(1);

    let mut visible = Vec::new();
    let start = state.output_lines.len().saturating_sub(output_height);
    for line in state.output_lines.iter().skip(start) {
        visible.push(Line::from(TuiSpan::raw(line.as_str())));
    }
    visible.push(Line::from(vec![
        TuiSpan::styled("tmux> ", Style::default().fg(ACCENT).bg(BG)),
        TuiSpan::styled(&state.input, Style::default().fg(FG).bg(BG)),
        TuiSpan::styled(
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

#[cfg(feature = "tui")]
fn draw_mirror(frame: &mut Frame<'_>, area: ratatui::layout::Rect, state: &TuiState) {
    let title = if state.mirror.label.is_empty() {
        " Mirror ".to_string()
    } else {
        format!(" Mirror - {} ", state.mirror.label)
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
        Some(motlie_tmux::MonitorHealth::Streaming) => " stream: active ",
        Some(motlie_tmux::MonitorHealth::Reconnecting) => " stream: reconnecting ",
        Some(motlie_tmux::MonitorHealth::Failed) => " stream: failed ",
        Some(motlie_tmux::MonitorHealth::Stopped) => " stream: stopped ",
        None => " stream: idle ",
    };
    let status_style = if matches!(
        state.mirror.watch_health,
        Some(motlie_tmux::MonitorHealth::Streaming)
            | Some(motlie_tmux::MonitorHealth::Reconnecting)
    ) {
        Style::default().fg(ACCENT).bg(BG)
    } else {
        Style::default().fg(DIM).bg(BG)
    };
    let status_line = Line::from(vec![
        TuiSpan::styled(
            format!(" {} ", state.host_uri),
            Style::default().fg(DIM).bg(BG),
        ),
        TuiSpan::styled(stream_label, status_style),
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

#[cfg(feature = "repl")]
fn render_incremental(previous: &mut String, current: &str) {
    if current.is_empty() || current == previous {
        return;
    }

    let delta = if current.starts_with(previous.as_str()) {
        &current[previous.len()..]
    } else {
        current
    };

    if !delta.trim().is_empty() {
        print!("{delta}");
        if !delta.ends_with('\n') {
            println!();
        }
    }

    previous.clear();
    previous.push_str(current);
}

#[cfg(feature = "tui")]
fn char_to_byte_index(text: &str, char_index: usize) -> usize {
    text.char_indices()
        .map(|(idx, _)| idx)
        .nth(char_index)
        .unwrap_or(text.len())
}

#[cfg(feature = "tui")]
fn insert_char(text: &mut String, cursor_chars: &mut usize, ch: char) {
    let byte_index = char_to_byte_index(text, *cursor_chars);
    text.insert(byte_index, ch);
    *cursor_chars += 1;
}

#[cfg(feature = "tui")]
fn delete_previous_char(text: &mut String, cursor_chars: &mut usize) {
    if *cursor_chars == 0 {
        return;
    }
    *cursor_chars -= 1;
    let start = char_to_byte_index(text, *cursor_chars);
    let end = char_to_byte_index(text, *cursor_chars + 1);
    text.drain(start..end);
}

#[cfg(feature = "tui")]
fn delete_current_char(text: &mut String, cursor_chars: &usize) {
    if *cursor_chars >= text.chars().count() {
        return;
    }
    let start = char_to_byte_index(text, *cursor_chars);
    let end = char_to_byte_index(text, *cursor_chars + 1);
    text.drain(start..end);
}

fn record_output(recorder: &mut Option<AsciicastRecorder>, text: &str) -> DriverResult<()> {
    if let Some(recorder) = recorder.as_mut() {
        recorder.record_output(text)?;
    }
    Ok(())
}

fn record_input(recorder: &mut Option<AsciicastRecorder>, text: &str) -> DriverResult<()> {
    if let Some(recorder) = recorder.as_mut() {
        recorder.record_input(text)?;
    }
    Ok(())
}

#[cfg(feature = "tui")]
fn record_resize(
    recorder: &mut Option<AsciicastRecorder>,
    cols: u16,
    rows: u16,
) -> DriverResult<()> {
    if let Some(recorder) = recorder.as_mut() {
        recorder.record_resize(cols, rows)?;
    }
    Ok(())
}
