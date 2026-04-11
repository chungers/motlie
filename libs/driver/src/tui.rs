#[cfg(feature = "tui")]
use std::collections::VecDeque;
#[cfg(feature = "tui")]
use std::io;
#[cfg(feature = "tui")]
use std::time::Duration;

#[cfg(feature = "tui")]
use anyhow::Result;
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
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};

#[cfg(feature = "tui")]
use crate::engine::{CommandEffect, CommandEngine, CommandSet};

#[cfg(feature = "tui")]
const OUTPUT_HISTORY_LIMIT: usize = 200;

#[cfg(feature = "tui")]
pub struct TuiFrontend<C, S> {
    engine: CommandEngine<C, S>,
    title: String,
    prompt: String,
    output_lines: VecDeque<String>,
    input: String,
    cursor_pos: usize,
}

#[cfg(feature = "tui")]
impl<C, S> TuiFrontend<C, S>
where
    C: 'static,
    S: CommandSet<C> + Send + 'static,
{
    pub fn new(engine: CommandEngine<C, S>) -> Self {
        Self {
            engine,
            title: "motlie".to_string(),
            prompt: "> ".to_string(),
            output_lines: VecDeque::with_capacity(OUTPUT_HISTORY_LIMIT),
            input: String::new(),
            cursor_pos: 0,
        }
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }

    pub fn engine(&self) -> &CommandEngine<C, S> {
        &self.engine
    }

    pub fn engine_mut(&mut self) -> &mut CommandEngine<C, S> {
        &mut self.engine
    }

    pub async fn run(&mut self) -> Result<()> {
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

        let run_result = self.event_loop(&mut terminal).await;

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        let _ = std::panic::take_hook();
        if let Ok(hook) = std::sync::Arc::try_unwrap(original_hook) {
            std::panic::set_hook(hook);
        }

        run_result
    }

    async fn event_loop(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    ) -> Result<()> {
        loop {
            terminal.draw(|frame| self.draw(frame))?;

            let maybe_event = tokio::task::spawn_blocking(|| {
                if event::poll(Duration::from_millis(150)).unwrap_or(false) {
                    event::read().ok()
                } else {
                    None
                }
            })
            .await?;

            let Some(Event::Key(key)) = maybe_event else {
                continue;
            };

            if key.kind == event::KeyEventKind::Release {
                continue;
            }

            match key.code {
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break,
                KeyCode::Char(ch) => {
                    self.input.insert(self.cursor_pos, ch);
                    self.cursor_pos += 1;
                }
                KeyCode::Backspace => {
                    if self.cursor_pos > 0 {
                        self.cursor_pos -= 1;
                        self.input.remove(self.cursor_pos);
                    }
                }
                KeyCode::Delete => {
                    if self.cursor_pos < self.input.len() {
                        self.input.remove(self.cursor_pos);
                    }
                }
                KeyCode::Left => {
                    self.cursor_pos = self.cursor_pos.saturating_sub(1);
                }
                KeyCode::Right => {
                    if self.cursor_pos < self.input.len() {
                        self.cursor_pos += 1;
                    }
                }
                KeyCode::Home => self.cursor_pos = 0,
                KeyCode::End => self.cursor_pos = self.input.len(),
                KeyCode::Enter => {
                    let line = self.input.trim().to_string();
                    self.input.clear();
                    self.cursor_pos = 0;

                    if line.is_empty() {
                        continue;
                    }

                    self.push_output(format!("{}{}", self.prompt, line));
                    let output = self.engine.run_line(&line).await?;
                    for line in output.lines {
                        self.push_output(line);
                    }

                    if output
                        .effects
                        .iter()
                        .any(|effect| matches!(effect, CommandEffect::ExitShell))
                    {
                        break;
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    fn draw(&self, frame: &mut Frame<'_>) {
        let areas = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(3)])
            .split(frame.area());

        let output = self
            .output_lines
            .iter()
            .map(|line| Line::from(line.as_str()))
            .collect::<Vec<_>>();
        let output_widget = Paragraph::new(output)
            .block(
                Block::default()
                    .title(Span::styled(
                        self.title.as_str(),
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    ))
                    .borders(Borders::ALL),
            )
            .wrap(Wrap { trim: false });
        frame.render_widget(output_widget, areas[0]);

        let input_widget = Paragraph::new(self.input.as_str())
            .block(Block::default().title("Input").borders(Borders::ALL));
        frame.render_widget(input_widget, areas[1]);
        frame.set_cursor_position((areas[1].x + self.cursor_pos as u16 + 1, areas[1].y + 1));
    }

    fn push_output(&mut self, line: impl Into<String>) {
        self.output_lines.push_back(line.into());
        while self.output_lines.len() > OUTPUT_HISTORY_LIMIT {
            self.output_lines.pop_front();
        }
    }
}
