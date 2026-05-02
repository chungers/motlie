use std::cmp::{max, min};

use ansi_to_tui::IntoText;
use motlie_tmux::strip_ansi;
use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span as TuiSpan, Text};
use ratatui::widgets::{
    Block, Borders, Clear, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap,
};

use crate::consts::{
    BUILD_DATE, BUILD_GIT_SHA, COMPACT_MOTLIE_PLACEHOLDER, HELP_KEY_FUNCTIONS, MODAL_BUTTON_HEIGHT,
    MODAL_CONTENT_HORIZONTAL_PADDING, MODAL_CONTENT_VERTICAL_PADDING, MODAL_MIN_WIDTH,
    MODAL_OUTER_MARGIN, MODAL_SEPARATOR_HEIGHT, MODAL_TEXT_FIELD_HEIGHT, MOTLIE_PLACEHOLDER,
    STATUS_BAR_BG,
};
use crate::detail::{DetailMode, SessionDetailSource};
use crate::model::{
    AppState, Button, Focus, LayoutMode, ModalBody, ModalState, ModalView, MotdState, SessionRow,
    SessionTagsFocus,
};

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

    if app.motd.is_some() {
        let upper = normal_motd_height(app, columns[0]);
        let left = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(upper), Constraint::Min(3)])
            .split(columns[0]);
        draw_motd(frame, left[0], app);
        draw_sessions(frame, left[1], app);
    } else {
        // Multi-host mode: MOTD is hidden; left column is fully sessions.
        draw_sessions(frame, columns[0], app);
    }
    draw_detail(frame, columns[1], app, " Detail ");
}

pub(crate) fn normal_motd_height(app: &AppState, left_area: Rect) -> u16 {
    const MIN_WIDGET_HEIGHT: u16 = 3;
    const MIN_SESSION_LIST_HEIGHT: u16 = 3;

    let Some(motd) = app.motd.as_ref() else {
        return 0;
    };
    if left_area.height <= MIN_SESSION_LIST_HEIGHT {
        return left_area.height;
    }

    let available = left_area.height.saturating_sub(MIN_SESSION_LIST_HEIGHT);
    let compact_placeholder = motd.is_placeholder
        && (left_area.width < full_placeholder_widget_width()
            || available < full_placeholder_widget_height());
    let desired = motd_render_line_count(motd, compact_placeholder).saturating_add(2);
    let capped = if motd.is_placeholder {
        desired
    } else {
        min(
            desired,
            max(MIN_WIDGET_HEIGHT, left_area.height.saturating_mul(30) / 100),
        )
    };

    min(max(MIN_WIDGET_HEIGHT, capped), available)
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

pub(crate) fn use_compact_placeholder(app: &AppState, width: u16, height: u16) -> bool {
    let Some(motd) = app.motd.as_ref() else {
        return false;
    };
    if !motd.is_placeholder {
        return false;
    }
    width < full_placeholder_widget_width() || height < full_placeholder_widget_height()
}

fn full_placeholder_widget_width() -> u16 {
    MOTLIE_PLACEHOLDER
        .lines()
        .map(|line| line.chars().count())
        .max()
        .unwrap_or(0)
        .saturating_add(2) as u16
}

fn full_placeholder_widget_height() -> u16 {
    MOTLIE_PLACEHOLDER
        .lines()
        .count()
        .saturating_add(1)
        .saturating_add(2) as u16
}

fn motd_render_line_count(motd: &MotdState, compact_placeholder: bool) -> u16 {
    if motd.is_placeholder && compact_placeholder {
        COMPACT_MOTLIE_PLACEHOLDER.lines().count().saturating_add(1) as u16
    } else if motd.is_placeholder {
        motd.text.lines().count().saturating_add(1) as u16
    } else {
        motd.text.lines().count() as u16
    }
}

pub(crate) fn motd_render_text(app: &AppState, area: Rect) -> String {
    let Some(motd) = app.motd.as_ref() else {
        return String::new();
    };
    if !motd.is_placeholder {
        return motd.text.clone();
    }
    if use_compact_placeholder(app, area.width, area.height) {
        format!("{COMPACT_MOTLIE_PLACEHOLDER}\n(no /etc/motd)")
    } else {
        format!("{}\n(no /etc/motd)", motd.text)
    }
}

fn draw_motd(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
    let Some(motd) = app.motd.as_ref() else {
        return;
    };
    let text_style = if motd.is_placeholder {
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
    format!(" Sessions [{}] ", app.session_list.rows.len())
}

fn draw_sessions(frame: &mut Frame<'_>, area: Rect, app: &AppState) {
    let height = area.height.saturating_sub(2) as usize;
    let row_width = area.width.saturating_sub(2) as usize;
    let host_code_width = app.fleet.host_code_width();
    let mut lines = Vec::new();
    if app.session_list.rows.is_empty() {
        lines.push(Line::from(TuiSpan::styled(
            empty_session_list_message(app),
            Style::default().fg(Color::DarkGray),
        )));
    } else {
        let start = app
            .session_list
            .selected
            .saturating_sub(height.saturating_sub(1));
        for (idx, row) in app
            .session_list
            .rows
            .iter()
            .enumerate()
            .skip(start)
            .take(height)
        {
            let selected = idx == app.session_list.selected;
            let host_code = app.fleet.host_code(&row.host_id);
            let style = if selected {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Green)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::White)
            };
            lines.push(Line::from(TuiSpan::styled(
                session_list_line(
                    row,
                    selected,
                    host_code.as_deref(),
                    host_code_width,
                    row_width,
                ),
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

fn empty_session_list_message(app: &AppState) -> String {
    if app.fleet.is_multi() {
        let n = app.fleet.len();
        format!("(no sessions across {n} hosts - press n to create)")
    } else if let Some(entry) = app.fleet.first() {
        format!("(no sessions on {} - press n to create)", entry.label)
    } else {
        "(no sessions - press n to create)".to_string()
    }
}

/// Render a session row.
///
/// `host_code` and `host_code_width` control the compact multi-host code
/// column. Pass `None, 0` for single-host rows.
pub(crate) fn session_list_line(
    row: &SessionRow,
    selected: bool,
    host_code: Option<&str>,
    host_code_width: usize,
    width: usize,
) -> String {
    const MIN_METADATA_GAP: usize = 2;
    const RECENCY_RIGHT_MARGIN: usize = 2;

    if width == 0 {
        return String::new();
    }

    let marker = if selected { ">" } else { " " };
    let attached = if row.session.is_attached() { "*" } else { " " };
    let prefix = if host_code_width > 0 {
        let host = truncate_chars(host_code.unwrap_or(""), host_code_width);
        let host_padded = pad_to(&host, host_code_width);
        format!("{marker}{attached} {host_padded} ")
    } else {
        format!("{marker}{attached} ")
    };
    let metadata = session_recency_text(row);
    let prefix_width = char_width(&prefix);
    let metadata_width = char_width(&metadata);
    let metadata_gap = session_row_metadata_gap(row, &metadata, MIN_METADATA_GAP);
    let content_width = width.saturating_sub(RECENCY_RIGHT_MARGIN);
    if content_width < prefix_width + 1 + metadata_gap + metadata_width {
        let name_width = width.saturating_sub(prefix_width);
        let name = session_row_name_field(row, name_width);
        return pad_or_truncate(format!("{prefix}{name}"), width);
    }

    let name_width = content_width
        .saturating_sub(prefix_width)
        .saturating_sub(metadata_width)
        .saturating_sub(metadata_gap);
    let name = session_row_name_field(row, name_width);
    let left = format!("{prefix}{name}");
    let padding = content_width
        .saturating_sub(char_width(&left))
        .saturating_sub(metadata_width);
    format!(
        "{left}{}{metadata}{}",
        " ".repeat(padding),
        " ".repeat(width.saturating_sub(content_width))
    )
}

fn session_row_name_field(row: &SessionRow, width: usize) -> String {
    if width == 0 {
        return String::new();
    }
    let Some(value) = row.displayed_tag_value() else {
        return truncate_chars(&row.session.name, width);
    };
    if width < 3 {
        return truncate_chars(&row.session.name, width);
    }

    let value_budget = width.saturating_sub(2);
    let value = truncate_chars(value, value_budget);
    let value_width = char_width(&value);
    if value_width == 0 {
        return truncate_chars(&row.session.name, width);
    }

    let name_width = width.saturating_sub(value_width + 1);
    let name = truncate_chars(&row.session.name, name_width);
    let gap = width
        .saturating_sub(char_width(&name))
        .saturating_sub(value_width);
    format!("{name}{}{value}", " ".repeat(gap))
}

fn session_row_metadata_gap(row: &SessionRow, metadata: &str, default_gap: usize) -> usize {
    match row.displayed_tag_value() {
        Some(_) => {
            let activity_left_pad = metadata.chars().take_while(|ch| *ch == ' ').count();
            default_gap.saturating_sub(activity_left_pad)
        }
        _ => default_gap,
    }
}

pub(crate) fn session_recency_text(row: &SessionRow) -> String {
    const RECENCY_FIELD_WIDTH: usize = 5;

    // Activity is observer-relative: time since the binary last saw the
    // host's `session.activity` advance. Naturally insensitive to host/
    // operator clock skew because both endpoints come from local time.
    let active = compact_elapsed(row.local_now, row.activity_observed_at_local);
    // Age is `local_now - session.created` under the NTP-synced clock
    // assumption. Wildly skewed host clocks produce mildly inaccurate text
    // but no functional regression.
    let age = compact_elapsed(row.local_now, row.session.created);
    format!(
        "{active:>width$} / {age:>width$}",
        width = RECENCY_FIELD_WIDTH
    )
}

fn pad_to(text: &str, width: usize) -> String {
    let text_width = char_width(text);
    if text_width >= width {
        text.to_string()
    } else {
        format!("{text}{}", " ".repeat(width - text_width))
    }
}

fn compact_elapsed(now: u64, then: u64) -> String {
    let seconds = now.saturating_sub(then);
    if seconds < 60 {
        "now".to_string()
    } else if seconds < 60 * 60 {
        format!("{}m", seconds / 60)
    } else if seconds < 48 * 60 * 60 {
        format!("{}h", seconds / 60 / 60)
    } else {
        compact_days(seconds)
    }
}

fn compact_days(seconds: u64) -> String {
    let tenths = (seconds.saturating_mul(10).saturating_add(12 * 60 * 60)) / (24 * 60 * 60);
    let whole = tenths / 10;
    let decimal = tenths % 10;
    if decimal == 0 {
        format!("{whole}d")
    } else {
        format!("{whole}.{decimal}d")
    }
}

fn pad_or_truncate(text: String, width: usize) -> String {
    let text = truncate_chars(&text, width);
    let text_width = char_width(&text);
    if text_width >= width {
        text
    } else {
        format!("{text}{}", " ".repeat(width - text_width))
    }
}

fn char_width(text: &str) -> usize {
    text.chars().count()
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
    } else if app.session_list.rows.is_empty() {
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
    let mut line = status_line(app);
    line.spans.push(TuiSpan::styled(
        format!(" | {}", app.status.text()),
        app.status.style(),
    ));
    let paragraph = Paragraph::new(line).style(Style::default().bg(STATUS_BAR_BG));
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
    if app.fleet.is_multi() {
        match app.fleet.host_code_legend() {
            Some(legend) if !legend.is_empty() => format!("mmux {legend} "),
            _ => "mmux ".to_string(),
        }
    } else if let Some(entry) = app.fleet.first() {
        format!(" {} | {} ", entry.label, entry.ip_address)
    } else {
        " mmux ".to_string()
    }
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    text.chars().take(max_chars).collect()
}

#[cfg(test)]
pub(crate) fn status_line_text(app: &AppState) -> String {
    status_line(app)
        .spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect()
}

pub(crate) fn status_line(app: &AppState) -> Line<'static> {
    let mut spans = vec![status_span(" ↑/↓ sel | ")];
    push_status_command(&mut spans, "help", 'h');
    push_status_separator(&mut spans);
    push_status_command(&mut spans, "pane", 'p');
    push_status_separator(&mut spans);
    push_status_command(&mut spans, "monitor", 'm');
    push_status_separator(&mut spans);
    push_status_command(&mut spans, "attach", 'a');
    push_status_separator(&mut spans);
    push_status_command(&mut spans, "new", 'n');
    push_status_separator(&mut spans);
    push_status_command(&mut spans, "kill", 'k');
    push_status_separator(&mut spans);
    push_status_command(&mut spans, "rename", 'r');
    push_status_separator(&mut spans);
    push_status_command(&mut spans, "tags", 't');
    push_status_separator(&mut spans);
    push_status_command(&mut spans, "group", 'g');
    push_status_separator(&mut spans);
    push_status_command(&mut spans, "quit", 'q');
    push_status_separator(&mut spans);
    push_status_command(&mut spans, "layout", 'l');
    push_status_separator(&mut spans);
    spans.push(status_span(match app.layout.mode {
        LayoutMode::Portrait => "mod-↑/↓ resize ",
        LayoutMode::Normal => "mod-←/→ resize ",
    }));
    Line::from(spans)
}

fn push_status_separator(spans: &mut Vec<TuiSpan<'static>>) {
    spans.push(status_span(" | "));
}

fn push_status_command(spans: &mut Vec<TuiSpan<'static>>, label: &'static str, mnemonic: char) {
    let mut plain = String::new();
    let mut found = false;
    for ch in label.chars() {
        if !found && ch == mnemonic {
            if !plain.is_empty() {
                spans.push(status_span(plain));
                plain = String::new();
            }
            spans.push(TuiSpan::styled(
                ch.to_string(),
                status_base_style().add_modifier(Modifier::UNDERLINED),
            ));
            found = true;
        } else {
            plain.push(ch);
        }
    }
    if !plain.is_empty() {
        spans.push(status_span(plain));
    }
}

fn status_span(text: impl Into<std::borrow::Cow<'static, str>>) -> TuiSpan<'static> {
    TuiSpan::styled(text, status_base_style())
}

fn status_base_style() -> Style {
    Style::default().fg(Color::White).bg(STATUS_BAR_BG)
}

fn draw_modal(frame: &mut Frame<'_>, area: Rect, modal: &ModalState) {
    let view = modal_content(modal);
    let width = min(
        max(
            MODAL_MIN_WIDTH,
            modal_content_width(&view)
                .saturating_add(MODAL_CONTENT_HORIZONTAL_PADDING.saturating_mul(2))
                .saturating_add(2),
        ),
        area.width
            .saturating_sub(MODAL_OUTER_MARGIN.saturating_mul(2)),
    );
    let height = min(
        max(
            7,
            modal_content_height(&view)
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

fn modal_content_height(view: &ModalView) -> u16 {
    match &view.body {
        ModalBody::Text(text) => max(1, text.lines().count()) as u16,
        ModalBody::NewSession { .. } => 1 + MODAL_TEXT_FIELD_HEIGHT,
        ModalBody::RenameSession { .. } => 1 + MODAL_TEXT_FIELD_HEIGHT,
        ModalBody::SessionTags { tags, .. } => {
            let rows = max(1, tags.len()) as u16;
            min(rows, TAG_LIST_MAX_ROWS as u16) + TAG_INPUT_SECTION_HEIGHT
        }
    }
}

fn modal_content_width(view: &ModalView) -> u16 {
    let body_width = match &view.body {
        ModalBody::Text(text) => text
            .lines()
            .map(|line| line.chars().count())
            .max()
            .unwrap_or(0),
        ModalBody::NewSession { input } => {
            max("Session name".chars().count(), input.chars().count())
        }
        ModalBody::RenameSession { input } => {
            max("Session Name".chars().count(), input.chars().count())
        }
        ModalBody::SessionTags {
            tags,
            key_input,
            value_input,
            ..
        } => tags
            .iter()
            .map(|tag| tag.key.chars().count() + tag.value.chars().count() + TAG_LIST_ROW_OVERHEAD)
            .chain([
                "No tags".chars().count() + TAG_LIST_ROW_OVERHEAD,
                key_input.chars().count() + value_input.chars().count() + TAG_EDIT_ROW_OVERHEAD,
            ])
            .max()
            .unwrap_or(0),
    };
    max(body_width, view.buttons.chars().count()) as u16
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
            draw_labeled_text_field(frame, area, "Session name", input, true);
        }
        ModalBody::RenameSession { input } => {
            draw_labeled_text_field(frame, area, "Session Name", input, true);
        }
        ModalBody::SessionTags {
            tags,
            selected_key,
            key_input,
            value_input,
            focus,
        } => {
            draw_session_tags_body(
                frame,
                area,
                tags,
                selected_key,
                key_input,
                value_input,
                *focus,
            );
        }
    }
}

fn draw_session_tags_body(
    frame: &mut Frame<'_>,
    area: Rect,
    tags: &[crate::model::SessionTagRow],
    selected_key: &Option<String>,
    key_input: &str,
    value_input: &str,
    focus: SessionTagsFocus,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }

    let key_width = tag_key_column_width(area.width, tags, key_input);
    let indicator_width = tag_indicator_column_width(area.width);
    let prefix_width = tag_prefix_column_width(area.width);
    let value_width = area
        .width
        .saturating_sub(prefix_width + key_width + indicator_width);
    let columns = TagColumns {
        prefix_width,
        key_width,
        value_width,
        indicator_width,
    };
    let max_tag_rows = visible_session_tag_rows(area.height, tags.len());
    let selected_row = match focus {
        SessionTagsFocus::TagRow(index) => Some(index),
        _ => None,
    };
    let start = selected_row
        .map(|index| index.saturating_sub(max_tag_rows.saturating_sub(1)))
        .unwrap_or(0);
    let mut lines = Vec::new();

    if tags.is_empty() && max_tag_rows > 0 {
        lines.push(no_session_tags_line(columns));
    } else {
        for (index, tag) in tags.iter().enumerate().skip(start).take(max_tag_rows) {
            let focused = matches!(focus, SessionTagsFocus::TagRow(row) if row == index);
            lines.push(session_tag_line(tag, selected_key, columns, focused));
        }
    }

    if max_tag_rows > 0 {
        frame.render_widget(
            Paragraph::new(lines).wrap(Wrap { trim: false }),
            Rect::new(area.x, area.y, area.width, max_tag_rows as u16),
        );
    }

    let input_y = area.y.saturating_add(max_tag_rows as u16);
    if input_y < area.bottom() {
        let separator = "─".repeat(area.width as usize);
        frame.render_widget(
            Paragraph::new(separator).style(tag_input_separator_style()),
            Rect::new(area.x, input_y, area.width, 1),
        );
    }
    let row_y = input_y.saturating_add(1);
    if row_y < area.bottom() {
        draw_session_tag_input_row(
            frame,
            Rect::new(area.x, row_y, area.width, 1),
            key_input,
            value_input,
            focus,
            columns,
        );
    }
}

#[derive(Clone, Copy)]
struct TagColumns {
    prefix_width: u16,
    key_width: u16,
    value_width: u16,
    indicator_width: u16,
}

const TAG_INPUT_SECTION_HEIGHT: u16 = 2;
const TAG_LIST_MAX_ROWS: usize = 5;
const TAG_LIST_PREFIX_WIDTH: u16 = 2;
const TAG_KEY_COLUMN_PADDING: u16 = 4;
const TAG_EMPTY_KEY_COLUMN_PERCENT: u16 = 30;
const TAG_EDIT_ROW_OVERHEAD: usize = 6;
const TAG_LIST_ROW_OVERHEAD: usize = 7;

fn visible_session_tag_rows(area_height: u16, tag_count: usize) -> usize {
    let available = area_height.saturating_sub(TAG_INPUT_SECTION_HEIGHT) as usize;
    let row_count = max(1, tag_count);
    min(TAG_LIST_MAX_ROWS, min(row_count, available))
}

fn tag_prefix_column_width(area_width: u16) -> u16 {
    min(TAG_LIST_PREFIX_WIDTH, area_width)
}

fn tag_indicator_column_width(area_width: u16) -> u16 {
    const WIDTH: u16 = 1;
    min(WIDTH, area_width)
}

pub(crate) fn tag_key_column_width(
    area_width: u16,
    tags: &[crate::model::SessionTagRow],
    key_input: &str,
) -> u16 {
    let prefix_width = tag_prefix_column_width(area_width);
    let indicator_width = tag_indicator_column_width(area_width);
    let max_key_width = area_width
        .saturating_sub(prefix_width)
        .saturating_sub(indicator_width)
        .saturating_sub(1);
    if max_key_width == 0 {
        return 0;
    }
    let empty_tags_default = if tags.is_empty() {
        empty_session_tags_key_column_width(area_width, prefix_width)
    } else {
        0
    };
    let longest_key = tags
        .iter()
        .map(|tag| tag.key.chars().count())
        .chain(std::iter::once(key_input.chars().count()))
        .max()
        .unwrap_or(0)
        .saturating_add(TAG_KEY_COLUMN_PADDING as usize) as u16;
    min(max(longest_key, empty_tags_default), max_key_width)
}

fn empty_session_tags_key_column_width(area_width: u16, prefix_width: u16) -> u16 {
    let edit_strip_width = area_width.saturating_sub(prefix_width);
    if edit_strip_width == 0 {
        return 0;
    }
    max(
        1,
        ((edit_strip_width as u32 * TAG_EMPTY_KEY_COLUMN_PERCENT as u32) / 100) as u16,
    )
}

fn no_session_tags_line(columns: TagColumns) -> Line<'static> {
    tag_list_row_line(
        pad_or_truncate_owned(String::new(), columns.prefix_width as usize),
        pad_or_truncate_owned("No tags".to_string(), columns.key_width as usize),
        pad_or_truncate_owned(String::new(), columns.value_width as usize),
        pad_or_truncate_owned(String::new(), columns.indicator_width as usize),
        Style::default().fg(Color::DarkGray),
        Style::default().fg(Color::DarkGray),
        Style::default().fg(Color::DarkGray),
    )
}

fn session_tag_line(
    tag: &crate::model::SessionTagRow,
    selected_key: &Option<String>,
    columns: TagColumns,
    focused: bool,
) -> Line<'static> {
    let indicator = if selected_key.as_deref() == Some(tag.key.as_str()) {
        "✓"
    } else {
        " "
    };
    let style = if focused {
        selected_list_row_style()
    } else {
        Style::default().fg(Color::White)
    };
    let marker = if focused { ">" } else { " " };
    let prefix = format!("{marker} ");
    tag_list_row_line(
        pad_or_truncate_owned(prefix, columns.prefix_width as usize),
        pad_or_truncate_owned(tag.key.clone(), columns.key_width as usize),
        pad_or_truncate_owned(tag.value.clone(), columns.value_width as usize),
        pad_or_truncate_owned(indicator.to_string(), columns.indicator_width as usize),
        style,
        style,
        style,
    )
}

fn draw_session_tag_input_row(
    frame: &mut Frame<'_>,
    area: Rect,
    key_input: &str,
    value_input: &str,
    focus: SessionTagsFocus,
    columns: TagColumns,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    let key_style = tag_edit_cell_style(focus == SessionTagsFocus::Key);
    let value_style = tag_edit_cell_style(focus == SessionTagsFocus::Value);
    let value_width = columns.value_width.saturating_add(columns.indicator_width);
    let line = tag_input_row_line(
        pad_or_truncate_owned(String::new(), columns.prefix_width as usize),
        pad_or_truncate_owned(key_input.to_string(), columns.key_width as usize),
        pad_or_truncate_owned(value_input.to_string(), value_width as usize),
        key_style,
        value_style,
    );
    frame.render_widget(Paragraph::new(line).style(tag_input_row_style()), area);
}

fn tag_list_row_line(
    prefix: String,
    key: String,
    value: String,
    indicator: String,
    key_style: Style,
    value_style: Style,
    indicator_style: Style,
) -> Line<'static> {
    Line::from(vec![
        TuiSpan::styled(prefix, key_style),
        TuiSpan::styled(key, key_style),
        TuiSpan::styled(value, value_style),
        TuiSpan::styled(indicator, indicator_style),
    ])
}

fn tag_input_row_line(
    prefix: String,
    key: String,
    value: String,
    key_style: Style,
    value_style: Style,
) -> Line<'static> {
    Line::from(vec![
        TuiSpan::styled(prefix, key_style),
        TuiSpan::styled(key, key_style),
        TuiSpan::styled(value, value_style),
    ])
}

fn tag_input_separator_style() -> Style {
    Style::default().fg(Color::DarkGray)
}

fn selected_list_row_style() -> Style {
    Style::default()
        .fg(Color::Black)
        .bg(Color::Green)
        .add_modifier(Modifier::BOLD)
}

fn tag_input_row_style() -> Style {
    Style::default().fg(Color::White).bg(Color::DarkGray)
}

fn tag_edit_cell_style(focused: bool) -> Style {
    if focused {
        selected_list_row_style()
    } else {
        tag_input_row_style()
    }
}

fn pad_or_truncate_owned(text: String, width: usize) -> String {
    let truncated = truncate_chars(&text, width);
    let text_width = char_width(&truncated);
    if text_width >= width {
        truncated
    } else {
        format!("{truncated}{}", " ".repeat(width - text_width))
    }
}

fn draw_labeled_text_field(
    frame: &mut Frame<'_>,
    area: Rect,
    label: &str,
    value: &str,
    focused: bool,
) {
    if area.width == 0 || area.height == 0 {
        return;
    }
    frame.render_widget(
        Paragraph::new(label),
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
    let border = if focused {
        Color::Green
    } else {
        Color::DarkGray
    };
    frame.render_widget(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border)),
        input_rect,
    );
    let input_inner = inset_rect(input_rect, 1, 1);
    if input_inner.width > 0 && input_inner.height > 0 {
        frame.render_widget(Paragraph::new(value), input_inner);
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
        ModalState::KillSession {
            session, button, ..
        } => ModalView {
            title: " Kill Session ",
            body: ModalBody::Text(format!("Kill session {}?", session.name)),
            buttons: format!(
                "{}   {}",
                button_text(*button, Button::Cancel),
                button_text(*button, Button::Ok)
            ),
            active_button: *button,
        },
        ModalState::RenameSession { input, button, .. } => ModalView {
            title: " Rename Session ",
            body: ModalBody::RenameSession {
                input: input.clone(),
            },
            buttons: format!(
                "{}   {}",
                button_text(*button, Button::Cancel),
                button_text(*button, Button::Ok)
            ),
            active_button: *button,
        },
        ModalState::SessionTags { ui, .. } => {
            let active_button = if ui.focus == SessionTagsFocus::Cancel {
                Button::Cancel
            } else {
                Button::Ok
            };
            ModalView {
                title: " Session Tags ",
                body: ModalBody::SessionTags {
                    tags: ui.tags.clone(),
                    selected_key: ui.selected_key.clone(),
                    key_input: ui.key_input.clone(),
                    value_input: ui.value_input.clone(),
                    focus: ui.focus,
                },
                buttons: button_text(active_button, Button::Cancel),
                active_button,
            }
        }
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
