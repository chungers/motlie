use std::io;

use anyhow::{Context, Result};
use crossterm::cursor::{Hide, SetCursorStyle, Show};
use crossterm::event::{
    KeyboardEnhancementFlags, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use crate::model::{AppState, HostFleet};
use crate::render::draw;

pub(crate) struct TerminalSession {
    terminal: Terminal<CrosstermBackend<io::Stderr>>,
    active: bool,
    keyboard_enhanced: bool,
}

impl TerminalSession {
    pub(crate) fn enter() -> Result<Self> {
        enable_raw_mode().context("enable terminal raw mode")?;
        let mut stderr = io::stderr();
        execute!(
            stderr,
            PushKeyboardEnhancementFlags(keyboard_enhancement_flags()),
            EnterAlternateScreen,
            SetCursorStyle::BlinkingBar,
            Hide
        )
        .context("enter alternate screen")?;
        let backend = CrosstermBackend::new(stderr);
        let terminal = Terminal::new(backend).context("create terminal backend")?;
        Ok(Self {
            terminal,
            active: true,
            keyboard_enhanced: true,
        })
    }

    pub(crate) fn draw(&mut self, fleet: &HostFleet, app: &mut AppState) -> Result<()> {
        self.terminal.draw(|frame| draw(frame, fleet, app))?;
        Ok(())
    }

    pub(crate) fn restore(&mut self) -> Result<()> {
        if self.active {
            disable_raw_mode().context("disable terminal raw mode")?;
            if self.keyboard_enhanced {
                execute!(
                    self.terminal.backend_mut(),
                    PopKeyboardEnhancementFlags,
                    SetCursorStyle::DefaultUserShape,
                    Show,
                    LeaveAlternateScreen
                )
                .context("leave alternate screen")?;
                self.keyboard_enhanced = false;
            } else {
                execute!(
                    self.terminal.backend_mut(),
                    SetCursorStyle::DefaultUserShape,
                    Show,
                    LeaveAlternateScreen
                )
                .context("leave alternate screen")?;
            }
            self.active = false;
        }
        Ok(())
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        if self.active {
            let _ = disable_raw_mode();
            if self.keyboard_enhanced {
                let _ = execute!(
                    self.terminal.backend_mut(),
                    PopKeyboardEnhancementFlags,
                    SetCursorStyle::DefaultUserShape,
                    Show,
                    LeaveAlternateScreen
                );
            } else {
                let _ = execute!(
                    self.terminal.backend_mut(),
                    SetCursorStyle::DefaultUserShape,
                    Show,
                    LeaveAlternateScreen
                );
            }
        }
    }
}

fn keyboard_enhancement_flags() -> KeyboardEnhancementFlags {
    KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES
        | KeyboardEnhancementFlags::REPORT_ALL_KEYS_AS_ESCAPE_CODES
        | KeyboardEnhancementFlags::REPORT_ALTERNATE_KEYS
}
