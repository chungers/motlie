use std::cmp::{max, min};

use ansi_to_tui::IntoText;
use motlie_tmux::{strip_ansi, SessionInfo};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span as TuiSpan, Text};
use ratatui::widgets::{
    Block, Borders, Clear, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap,
};
use ratatui::Frame;

use crate::consts::{
    BUILD_DATE, BUILD_GIT_SHA, COMPACT_MOTLIE_PLACEHOLDER, HELP_KEY_FUNCTIONS, MODAL_BUTTON_HEIGHT,
    MODAL_CONTENT_HORIZONTAL_PADDING, MODAL_CONTENT_VERTICAL_PADDING, MODAL_MIN_WIDTH,
    MODAL_OUTER_MARGIN, MODAL_SEPARATOR_HEIGHT, MODAL_TEXT_FIELD_HEIGHT, MOTLIE_PLACEHOLDER,
    NORMAL_STATUS_KEYS, PORTRAIT_STATUS_KEYS, STATUS_BAR_BG,
};
use crate::detail::{DetailMode, SessionDetailSource};
use crate::model::{AppState, Button, Focus, LayoutMode, ModalBody, ModalState, ModalView};

pub(crate) fn draw(frame: &mut Frame<'_>, app: &mut AppState) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .split(area);

    draw_top_status(frame, chunks[0], app);
    match app.layout.mode {
        LayoutMode::Normal => draw_normal(frame, chunks[1], app),
        LayoutMode::Portrait => draw_portrait(frame, chunks[1], app),
    }
    draw_status(frame, chunks[2], app);
    if let Some(modal) = &app.modal {
        draw_modal(frame, area, modal);
    }
}

fn draw_normal(frame: &mut Frame<'_>, area: Rect, app: &mut AppState) {
    let columns = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(app.layout.left_percent),
            Constraint::Percentage(100 - app.layout.left_percent),
        ])
        .split(area);

    let compact_placeholder = use_compact_placeholder(app, columns[0].width, columns[0].height);
    let motd_lines = motd_render_line_count(app, compact_placeholder);
    let max_motd = max(3, area.height.saturating_mul(30) / 100);
    let desired = motd_lines.saturating_add(2);
    let upper = if app.motd.is_placeholder {
        min(desired, columns[0].height.saturating_sub(3))
    } else {
        min(desired, max_motd)
    };
    let upper = max(3, min(upper, columns[0].height.saturating_sub(3)));
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(upper), Constraint::Min(3)])
        .split(columns[0]);

    draw_motd(frame, left[0], app);
    draw_sessions(frame, left[1], app);
    draw_detail(frame, columns[1], app, " Detail ");
}

fn draw_portrait(frame: &mut Frame<'_>, area: Rect, app: &mut AppState) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(app.layout.top_percent),
            Constraint::Percentage(100 - app.layout.top_percent),
        ])
        .split(area);
    draw_sessions(frame, rows[0], app);
    draw_detail(frame, rows[1], app, " Detail ");
}

fn focused_style(app: &AppState, focus: Focus) -> Style {
    if app.layout.focus == focus {
        Style::default().fg(Color::Green)
    } else {
        Style::default().fg(Color::DarkGray)
    }
}

fn use_compact_placeholder(app: &AppState, width: u16, height: u16) -> bool {
    if !app.motd.is_placeholder {
        return false;
    }
    let full_height = MOTLIE_PLACEHOLDER
        .lines()
        .count()
        .saturating_add(1)
        .saturating_add(2) as u16;
    width < 63 || height < full_height.saturating_add(3)
}

fn motd_render_line_count(app: &AppState, compact_placeholder: bool) -> u16 {
    if app.motd.is_placeholder && compact_placeholder {
        COMPACT_MOTLIE_PLACEHOLDER.lines().count().saturating_add(1) as u16
    } else if app.motd.is_placeholder {
        app.motd.text.lines().count().saturating_add(1) as u16
    } else {
        app.motd.text.lines().count() as u16
    }
}

pub(crate) fn motd_render_text(app: &AppState, area: Rect) -> String {
    if !app.motd.is_placeholder {
        return app.motd.text.clone();
    }
    if use_compact_placeholder(app, area.width, area.height) {
        format!("{COMPACT_MOTLIE_PLACEHOLDER}\n(no /etc/motd)")
    } else {
        format!("{}\n(no /etc/motd)", app.motd.text)
    }
}

fn draw_motd(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
    let text_style = if app.motd.is_placeholder {
        Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD)
    } else {
        Style::default().fg(Color::White)
    };
    let text = motd_render_text(app, area);
    let paragraph = Paragraph::new(text)
        .style(text_style)
        .block(
            Block::default()
                .title(" MOTD ")
                .borders(Borders::ALL)
                .border_style(focused_style(app, Focus::Motd)),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(paragraph, area);
}

pub(crate) fn sessions_title(app: &AppState) -> String {
    format!(" Sessions [{}] ", app.session_list.sessions.len())
}

fn draw_sessions(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
    let height = area.height.saturating_sub(2) as usize;
    let mut lines = Vec::new();
    if app.session_list.sessions.is_empty() {
        lines.push(Line::from(TuiSpan::styled(
            format!("(no sessions on {} - press n to create)", app.host.label),
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        let start = app
            .session_list
            .selected
            .saturating_sub(height.saturating_sub(1));
        for (idx, session) in app
            .session_list
            .sessions
            .iter()
            .enumerate()
            .skip(start)
            .take(height)
        {
            let selected = idx == app.session_list.selected;
            let style = if selected {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Green)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            lines.push(Line::from(TuiSpan::styled(
                session_list_line(session, selected),
                style,
            )));
        }
    }

    let block = Block::default()
        .title(sessions_title(app))
        .borders(Borders::ALL)
        .border_style(focused_style(app, Focus::List));
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

pub(crate) fn session_list_line(session: &SessionInfo, selected: bool) -> String {
    let marker = if selected { ">" } else { " " };
    let attached = if session.attached { "*" } else { " " };
    format!("{marker}{attached} {}", session.name)
}

fn draw_detail(frame: &mut Frame<'_>, area: Rect, app: &mut AppState, title: &str) {
    let height = area.height.saturating_sub(2) as usize;
    app.detail.last_known_view_height = max(1, height);
    app.detail.scroll = app.detail.scroll.min(app.detail.max_scroll());
    let total = app.detail.lines.len();
    let end = total.saturating_sub(app.detail.scroll);
    let start = end.saturating_sub(height);
    let visible = if start < end {
        app.detail.lines[start..end].join("\n")
    } else if app.session_list.sessions.is_empty() {
        "press n to create a session".to_string()
    } else {
        String::new()
    };
    let position = if total == 0 {
        "0/0".to_string()
    } else {
        format!("{}-{}/{}", start + 1, end, total)
    };
    let title = match app.detail.source.mode() {
        DetailMode::Sample => format!("{title} {position} "),
        DetailMode::Monitor => format!(" Detail - monitor {position} "),
    };
    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(focused_style(app, Focus::Detail));
    frame.render_widget(
        Paragraph::new(detail_text_for_render(&visible))
            .block(block)
            .wrap(Wrap { trim: false }),
        area,
    );

    if total > height {
        let mut scrollbar_state = ScrollbarState::new(total)
            .position(start)
            .viewport_content_length(height);
        frame.render_stateful_widget(
            Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .thumb_style(Style::default().fg(Color::Green))
                .track_style(Style::default().fg(Color::DarkGray)),
            area,
            &mut scrollbar_state,
        );
    }
}

pub(crate) fn detail_text_for_render(text: &str) -> Text<'_> {
    text.into_text()
        .unwrap_or_else(|_| Text::raw(strip_ansi(text)))
}

fn draw_status(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
    let text = status_line_text(app);
    let paragraph = Paragraph::new(Line::from(vec![
        TuiSpan::styled(text, Style::default().fg(Color::White).bg(STATUS_BAR_BG)),
        TuiSpan::styled(format!(" | {}", app.status.text()), app.status.style()),
    ]))
    .style(Style::default().bg(STATUS_BAR_BG));
    frame.render_widget(paragraph, area);
}

fn draw_top_status(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
    let time = chrono::Local::now().format("%H:%M:%S").to_string();
    let paragraph = Paragraph::new(top_status_line(app, &time, area.width as usize))
        .style(Style::default().bg(STATUS_BAR_BG));
    frame.render_widget(paragraph, area);
}

pub(crate) fn top_status_line(app: &AppState, time: &str, width: usize) -> Line<'static> {
    let time = format!(" {time} ");
    let time = truncate_chars(&time, min(time.chars().count(), width));
    let max_left_width = width.saturating_sub(time.chars().count());
    let left = if max_left_width == 0 {
        String::new()
    } else {
        truncate_chars(&top_status_host_text(app), max_left_width)
    };
    let left_width = left.chars().count();
    let time_width = time.chars().count();
    let padding_width = width.saturating_sub(left_width + time_width);

    Line::from(vec![
        TuiSpan::styled(
            left,
            Style::default()
                .fg(Color::White)
                .bg(STATUS_BAR_BG)
                .add_modifier(Modifier::BOLD),
        ),
        TuiSpan::styled(
            " ".repeat(padding_width),
            Style::default().bg(STATUS_BAR_BG),
        ),
        TuiSpan::styled(time, Style::default().fg(Color::White).bg(STATUS_BAR_BG)),
    ])
}

fn top_status_host_text(app: &AppState) -> String {
    format!(" {} | {} ", app.host.label, app.host.ip_address)
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    text.chars().take(max_chars).collect()
}

pub(crate) fn status_line_text(app: &AppState) -> String {
    let keys = if app.layout.mode == LayoutMode::Portrait {
        PORTRAIT_STATUS_KEYS
    } else {
        NORMAL_STATUS_KEYS
    };
    format!(" {} ", keys)
}

fn draw_modal(frame: &mut Frame<'_>, area: Rect, modal: &ModalState) {
    let view = modal_content(modal);
    let width = min(
        max(
            MODAL_MIN_WIDTH,
            view.content_width()
                .saturating_add(MODAL_CONTENT_HORIZONTAL_PADDING.saturating_mul(2))
                .saturating_add(2),
        ),
        area.width
            .saturating_sub(MODAL_OUTER_MARGIN.saturating_mul(2)),
    );
    let height = min(
        max(
            7,
            view.content_height()
                .saturating_add(MODAL_CONTENT_VERTICAL_PADDING.saturating_mul(2))
                .saturating_add(MODAL_SEPARATOR_HEIGHT)
                .saturating_add(MODAL_BUTTON_HEIGHT)
                .saturating_add(2),
        ),
        area.height
            .saturating_sub(MODAL_OUTER_MARGIN.saturating_mul(2)),
    );
    let x = area.x + area.width.saturating_sub(width) / 2;
    let y = area.y + area.height.saturating_sub(height) / 2;
    let rect = Rect::new(x, y, width, height);
    frame.render_widget(Clear, rect);

    let border = if view.active_button == Button::Ok {
        Color::Green
    } else {
        Color::Yellow
    };
    let block = Block::default()
        .title(view.title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border));
    frame.render_widget(block, rect);

    let inner = inset_rect(rect, 1, 1);
    if inner.height <= MODAL_SEPARATOR_HEIGHT + MODAL_BUTTON_HEIGHT || inner.width == 0 {
        return;
    }

    let button_y = inner.y + inner.height.saturating_sub(MODAL_BUTTON_HEIGHT);
    let separator_y = button_y.saturating_sub(MODAL_SEPARATOR_HEIGHT);
    let content_outer = Rect::new(
        inner.x,
        inner.y,
        inner.width,
        separator_y.saturating_sub(inner.y),
    );
    let content_rect = inset_rect(
        content_outer,
        MODAL_CONTENT_HORIZONTAL_PADDING,
        MODAL_CONTENT_VERTICAL_PADDING,
    );
    draw_modal_body(frame, content_rect, &view.body);

    let separator = "─".repeat(inner.width as usize);
    frame.render_widget(
        Paragraph::new(separator).style(Style::default().fg(Color::DarkGray)),
        Rect::new(inner.x, separator_y, inner.width, MODAL_SEPARATOR_HEIGHT),
    );
    draw_modal_buttons(
        frame,
        inset_rect(
            Rect::new(inner.x, button_y, inner.width, MODAL_BUTTON_HEIGHT),
            MODAL_CONTENT_HORIZONTAL_PADDING,
            0,
        ),
        &view.buttons,
    );
}

fn draw_modal_body(frame: &mut Frame<'_>, area: Rect, body: &ModalBody) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    match body {
        ModalBody::Text(text) => {
            frame.render_widget(
                Paragraph::new(text.as_str()).wrap(Wrap { trim: false }),
                area,
            );
        }
        ModalBody::NewSession { input } => {
            frame.render_widget(
                Paragraph::new("Session name"),
                Rect::new(area.x, area.y, area.width, 1),
            );
            if area.height <= 1 {
                return;
            }
            let input_rect = Rect::new(
                area.x,
                area.y.saturating_add(1),
                area.width,
                min(MODAL_TEXT_FIELD_HEIGHT, area.height.saturating_sub(1)),
            );
            frame.render_widget(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray)),
                input_rect,
            );
            let input_inner = inset_rect(input_rect, 1, 1);
            if input_inner.width > 0 && input_inner.height > 0 {
                frame.render_widget(Paragraph::new(input.as_str()), input_inner);
            }
        }
    }
}

fn draw_modal_buttons(frame: &mut Frame<'_>, area: Rect, buttons: &str) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let button_width = min(buttons.chars().count() as u16, area.width);
    let x = area.x + area.width.saturating_sub(button_width) / 2;
    frame.render_widget(
        Paragraph::new(buttons),
        Rect::new(x, area.y, button_width, area.height),
    );
}

fn inset_rect(rect: Rect, horizontal: u16, vertical: u16) -> Rect {
    let x_offset = min(horizontal, rect.width);
    let y_offset = min(vertical, rect.height);
    Rect::new(
        rect.x.saturating_add(x_offset),
        rect.y.saturating_add(y_offset),
        rect.width.saturating_sub(x_offset.saturating_mul(2)),
        rect.height.saturating_sub(y_offset.saturating_mul(2)),
    )
}

pub(crate) fn modal_content(modal: &ModalState) -> ModalView {
    match modal {
        ModalState::NewSession { input, button } => ModalView {
            title: " New Session ",
            body: ModalBody::NewSession {
                input: input.clone(),
            },
            buttons: format!(
                "{}   {}",
                button_text(*button, Button::Cancel),
                button_text(*button, Button::Ok)
            ),
            active_button: *button,
        },
        ModalState::KillSession { name, button, .. } => ModalView {
            title: " Kill Session ",
            body: ModalBody::Text(format!("Kill session {name}?")),
            buttons: format!(
                "{}   {}",
                button_text(*button, Button::Cancel),
                button_text(*button, Button::Ok)
            ),
            active_button: *button,
        },
        ModalState::Help => ModalView {
            title: " Help ",
            body: ModalBody::Text(format!(
                "{}\n\nBuild date: {}\nGit SHA: {}\n\n{}",
                MOTLIE_PLACEHOLDER,
                BUILD_DATE,
                short_build_git_sha(),
                HELP_KEY_FUNCTIONS
            )),
            buttons: "[Ok]".to_string(),
            active_button: Button::Ok,
        },
    }
}

pub(crate) fn short_build_git_sha() -> String {
    let mut chars = BUILD_GIT_SHA.chars().rev().take(8).collect::<Vec<_>>();
    chars.reverse();
    chars.into_iter().collect()
}

fn button_text(active: Button, button: Button) -> String {
    let label = match button {
        Button::Cancel => "Cancel",
        Button::Ok => "Ok",
    };
    if active == button {
        format!("[{label}]")
    } else {
        format!(" {label} ")
    }
}
