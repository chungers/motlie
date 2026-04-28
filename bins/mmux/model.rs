use std::cmp::{max, min};

use motlie_tmux::SessionInfo;
use ratatui::style::{Color, Style};

use crate::consts::{
    DEFAULT_LEFT_PERCENT, DEFAULT_TOP_PERCENT, LANDSCAPE_MAX_LEFT_PERCENT,
    LANDSCAPE_MIN_LEFT_PERCENT, MODAL_TEXT_FIELD_HEIGHT, PORTRAIT_MAX_TOP_PERCENT,
    PORTRAIT_MIN_TOP_PERCENT, STATUS_BAR_BG,
};
use crate::detail::DetailSource;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayoutMode {
    Normal,
    Portrait,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Focus {
    Motd,
    List,
    Detail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Button {
    Cancel,
    Ok,
}

#[derive(Debug, Clone)]
pub(crate) enum ModalState {
    NewSession {
        input: String,
        button: Button,
    },
    KillSession {
        id: String,
        name: String,
        button: Button,
    },
    Help,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModalView {
    pub(crate) title: &'static str,
    pub(crate) body: ModalBody,
    pub(crate) buttons: String,
    pub(crate) active_button: Button,
}

impl ModalView {
    #[cfg(test)]
    pub(crate) fn body_text(&self) -> String {
        match &self.body {
            ModalBody::Text(text) => text.clone(),
            ModalBody::NewSession { input } => format!("Session name\n{input}"),
        }
    }

    pub(crate) fn content_height(&self) -> u16 {
        match &self.body {
            ModalBody::Text(text) => max(1, text.lines().count()) as u16,
            ModalBody::NewSession { .. } => 1 + MODAL_TEXT_FIELD_HEIGHT,
        }
    }

    pub(crate) fn content_width(&self) -> u16 {
        let body_width = match &self.body {
            ModalBody::Text(text) => text
                .lines()
                .map(|line| line.chars().count())
                .max()
                .unwrap_or(0),
            ModalBody::NewSession { input } => {
                max("Session name".chars().count(), input.chars().count())
            }
        };
        max(body_width, self.buttons.chars().count()) as u16
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ModalBody {
    Text(String),
    NewSession { input: String },
}

#[derive(Debug, Clone)]
pub(crate) struct SelectedSession {
    pub(crate) id: String,
    pub(crate) name: String,
}

#[derive(Debug, Clone)]
pub(crate) struct RetainedUiState {
    layout_mode: LayoutMode,
    selected_session_id: Option<String>,
    selected_index: usize,
    focus: Focus,
    left_percent: u16,
    top_percent: u16,
}

impl Default for RetainedUiState {
    fn default() -> Self {
        Self::new(LayoutMode::Normal)
    }
}

impl RetainedUiState {
    pub(crate) fn new(layout_mode: LayoutMode) -> Self {
        Self {
            layout_mode,
            selected_session_id: None,
            selected_index: 0,
            focus: Focus::List,
            left_percent: DEFAULT_LEFT_PERCENT,
            top_percent: DEFAULT_TOP_PERCENT,
        }
    }

    pub(crate) fn apply_to(&self, app: &mut AppState) {
        app.layout.mode = self.layout_mode;
        app.session_list.selected = self.selected_index;
        app.layout.focus = focus_for_layout(self.focus, app.layout.mode);
        app.layout.left_percent = self
            .left_percent
            .clamp(LANDSCAPE_MIN_LEFT_PERCENT, LANDSCAPE_MAX_LEFT_PERCENT);
        app.layout.top_percent = self
            .top_percent
            .clamp(PORTRAIT_MIN_TOP_PERCENT, PORTRAIT_MAX_TOP_PERCENT);
    }

    pub(crate) fn update_from(&mut self, app: &AppState) {
        self.layout_mode = app.layout.mode;
        self.selected_session_id = app.selected_session().map(|session| session.id);
        self.selected_index = app.session_list.selected;
        self.focus = app.layout.focus;
        self.left_percent = app.layout.left_percent;
        self.top_percent = app.layout.top_percent;
    }

    pub(crate) fn selected_session_id(&self) -> Option<String> {
        self.selected_session_id.clone()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct HostContext {
    pub(crate) label: String,
    pub(crate) ip_address: String,
}

#[derive(Debug, Clone)]
pub(crate) struct LayoutState {
    pub(crate) mode: LayoutMode,
    pub(crate) focus: Focus,
    pub(crate) left_percent: u16,
    pub(crate) top_percent: u16,
}

#[derive(Debug, Clone)]
pub(crate) struct MotdState {
    pub(crate) text: String,
    pub(crate) is_placeholder: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct SessionListState {
    pub(crate) sessions: Vec<SessionInfo>,
    pub(crate) now: u64,
    pub(crate) selected: usize,
    pub(crate) scroll: usize,
}

impl SessionListState {
    pub(crate) fn selected_session(&self) -> Option<SelectedSession> {
        self.sessions
            .get(self.selected)
            .map(|session| SelectedSession {
                id: session.id.as_str().to_string(),
                name: session.name.clone(),
            })
    }

    pub(crate) fn preserve_selection(&mut self, previous_id: Option<String>) {
        if self.sessions.is_empty() {
            self.selected = 0;
            self.scroll = 0;
            return;
        }

        if let Some(id) = previous_id {
            if let Some(pos) = self.sessions.iter().position(|s| s.id.as_str() == id) {
                self.selected = pos;
            } else {
                self.selected = min(self.selected, self.sessions.len().saturating_sub(1));
            }
        } else {
            self.selected = min(self.selected, self.sessions.len().saturating_sub(1));
        }
    }

    pub(crate) fn move_selection(&mut self, delta: isize) -> bool {
        if self.sessions.is_empty() {
            return false;
        }
        let old = self.selected;
        let max_index = self.sessions.len().saturating_sub(1) as isize;
        let next = (self.selected as isize + delta).clamp(0, max_index) as usize;
        self.selected = next;
        old != next
    }
}

pub(crate) struct DetailState {
    pub(crate) lines: Vec<String>,
    pub(crate) scroll: usize,
    /// Render-feedback cache used by next-tick scroll math. The renderer owns
    /// the actual viewport size; input handling uses this last known value.
    pub(crate) last_known_view_height: usize,
    pub(crate) source: DetailSource,
    pub(crate) auto_tail: bool,
}

impl DetailState {
    pub(crate) fn set_text(&mut self, text: String) {
        self.lines = text.lines().map(|line| line.to_string()).collect();
        self.scroll = 0;
        self.auto_tail = true;
    }

    pub(crate) fn max_scroll(&self) -> usize {
        self.lines.len().saturating_sub(self.last_known_view_height)
    }

    pub(crate) fn scroll(&mut self, delta: isize) -> bool {
        let old = self.scroll;
        if delta < 0 {
            self.scroll = self.scroll.saturating_sub(delta.unsigned_abs());
        } else {
            self.scroll = self
                .scroll
                .saturating_add(delta as usize)
                .min(self.max_scroll());
        }
        self.auto_tail = self.scroll == 0;
        old != self.scroll
    }

    pub(crate) fn home(&mut self) {
        self.scroll = self.max_scroll();
        self.auto_tail = false;
    }

    pub(crate) fn end(&mut self) {
        self.scroll = 0;
        self.auto_tail = true;
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StatusBanner {
    Loading(String),
    Info(String),
    Error(String),
}

impl StatusBanner {
    pub(crate) fn loading(text: impl Into<String>) -> Self {
        Self::Loading(text.into())
    }

    pub(crate) fn info(text: impl Into<String>) -> Self {
        Self::Info(text.into())
    }

    pub(crate) fn error(text: impl Into<String>) -> Self {
        Self::Error(text.into())
    }

    pub(crate) fn text(&self) -> &str {
        match self {
            StatusBanner::Loading(text) | StatusBanner::Info(text) | StatusBanner::Error(text) => {
                text
            }
        }
    }

    pub(crate) fn style(&self) -> Style {
        match self {
            StatusBanner::Loading(_) => Style::default().fg(Color::White).bg(STATUS_BAR_BG),
            StatusBanner::Info(_) => Style::default().fg(Color::Yellow).bg(STATUS_BAR_BG),
            StatusBanner::Error(_) => Style::default().fg(Color::Red).bg(STATUS_BAR_BG),
        }
    }
}

pub(crate) struct AppState {
    pub(crate) host: HostContext,
    pub(crate) layout: LayoutState,
    pub(crate) motd: MotdState,
    pub(crate) session_list: SessionListState,
    pub(crate) detail: DetailState,
    pub(crate) status: StatusBanner,
    pub(crate) modal: Option<ModalState>,
}

impl AppState {
    #[cfg(test)]
    pub(crate) fn new(
        host_label: String,
        layout_mode: LayoutMode,
        motd: String,
        placeholder: bool,
    ) -> Self {
        Self::new_with_host_ip(
            host_label,
            "unknown".to_string(),
            layout_mode,
            motd,
            placeholder,
        )
    }

    pub(crate) fn new_with_host_ip(
        host_label: String,
        host_ip_address: String,
        layout_mode: LayoutMode,
        motd: String,
        placeholder: bool,
    ) -> Self {
        Self {
            host: HostContext {
                label: host_label,
                ip_address: host_ip_address,
            },
            layout: LayoutState {
                mode: layout_mode,
                focus: Focus::List,
                left_percent: DEFAULT_LEFT_PERCENT,
                top_percent: DEFAULT_TOP_PERCENT,
            },
            motd: MotdState {
                text: motd,
                is_placeholder: placeholder,
            },
            session_list: SessionListState {
                sessions: Vec::new(),
                now: 0,
                selected: 0,
                scroll: 0,
            },
            detail: DetailState {
                lines: Vec::new(),
                scroll: 0,
                last_known_view_height: 1,
                source: DetailSource::sample(),
                auto_tail: true,
            },
            status: StatusBanner::loading("loading sessions"),
            modal: None,
        }
    }

    pub(crate) fn selected_session(&self) -> Option<SelectedSession> {
        self.session_list.selected_session()
    }

    pub(crate) fn preserve_selection(&mut self, previous_id: Option<String>) {
        self.session_list.preserve_selection(previous_id);
    }

    pub(crate) fn move_selection(&mut self, delta: isize) -> bool {
        self.session_list.move_selection(delta)
    }

    pub(crate) fn set_detail_text(&mut self, text: String) {
        self.detail.set_text(text);
    }

    pub(crate) fn scroll_detail(&mut self, delta: isize) {
        if self.detail.scroll(delta) {
            self.status = StatusBanner::info(format!("detail offset {}", self.detail.scroll));
        }
    }

    pub(crate) fn detail_home(&mut self) {
        self.detail.home();
    }

    pub(crate) fn detail_end(&mut self) {
        self.detail.end();
    }

    pub(crate) fn focus_next(&mut self) {
        self.layout.focus = match (self.layout.mode, self.layout.focus) {
            (LayoutMode::Normal, Focus::Motd) => Focus::List,
            (LayoutMode::Normal, Focus::List) => Focus::Detail,
            (LayoutMode::Normal, Focus::Detail) => Focus::Motd,
            (LayoutMode::Portrait, Focus::List) => Focus::Detail,
            (LayoutMode::Portrait, Focus::Detail) => Focus::List,
            (LayoutMode::Portrait, Focus::Motd) => Focus::List,
        };
    }

    pub(crate) fn toggle_layout(&mut self) {
        self.layout.mode = match self.layout.mode {
            LayoutMode::Normal => LayoutMode::Portrait,
            LayoutMode::Portrait => LayoutMode::Normal,
        };
        self.layout.focus = focus_for_layout(self.layout.focus, self.layout.mode);
        self.status = StatusBanner::info("layout toggled");
    }
}

pub(crate) fn focus_for_layout(focus: Focus, layout_mode: LayoutMode) -> Focus {
    match (layout_mode, focus) {
        (LayoutMode::Portrait, Focus::Motd) => Focus::List,
        _ => focus,
    }
}
