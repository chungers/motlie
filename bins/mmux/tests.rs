use clap::{CommandFactory, Parser};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use motlie_tmux::{
    transport::MockTransport, HostHandle, SessionId, SessionInfo, StatusLeft, StatusLeftLength,
    StatusStyle, TransportKind, WindowStyle, SSH_DEFAULT_PORT,
};
use ratatui::backend::{Backend, TestBackend};
use ratatui::buffer::Buffer;
use ratatui::layout::Position;
use ratatui::style::{Color, Modifier};
use ratatui::widgets::{Paragraph, Wrap};
use ratatui::Terminal;

use crate::cli::{is_portrait_pty, select_layout, Cli};
use crate::consts::{
    mmux_attach_status_style, mmux_attach_window_style, APP_BASE_BG, APP_BASE_FG, BUILD_DATE,
    BUILD_GIT_SHA, HELP_KEY_FUNCTIONS, HOST_COLOR_PALETTE, HOST_COLOR_SQUARE,
    HOST_CONNECTION_FAILED_FG, LANDSCAPE_MAX_LEFT_PERCENT, LANDSCAPE_MIN_LEFT_PERCENT,
    MMUX_ATTACH_STATUS_LEFT, MMUX_ATTACH_STATUS_LEFT_LENGTH, MODAL_CONTENT_HORIZONTAL_PADDING,
    MODAL_MIN_WIDTH, MOTLIE_PLACEHOLDER, PORTRAIT_MAX_TOP_PERCENT, PORTRAIT_MIN_TOP_PERCENT,
    STATUS_BAR_BG, STATUS_BAR_MNEMONIC_FG,
};
use crate::controller::{
    apply_fleet_snapshot, apply_streaming_host_results, fetch_fleet_refresh, fetch_host_refresh,
    handle_key, refresh_sessions_preserving, refresh_sessions_quiet, stop_monitor_if_closed,
    KeyOutcome, RefreshApplyOptions,
};
use crate::detail::{
    DetailMode, DetailSource, MonitorDetailSource, SampleDetailSource, SessionDetailSource,
};
use crate::model::{
    AppState, Button, Focus, HostConnectFailure, HostConnectionStatus, HostEntry, HostFleet,
    HostId, HostSlot, LayoutMode, ModalBody, ModalState, NewSessionFocus, NewSessionHostChoice,
    NewSessionModalUi, PendingListShortcut, SelectedSession, SendKeysFocus, SendKeysModalUi,
    SessionKeyValueFocus, SessionKeyValueKind, SessionKeyValueModalUi, SessionKeyValueRow,
    SessionRow, SessionSelectedTag, SessionSortMode,
};
use crate::render::{
    detail_lines_text_for_render, detail_text_for_render, detail_title, detail_total_wrapped_rows,
    draw, key_value_key_column_width, modal_content, session_key_values_footer_line,
    session_list_line, session_recency_text, sessions_title, short_build_git_sha, status_line,
    status_line_text, top_status_line,
};
use crate::target_host::resolve_ip_address;
use crate::{
    prepare_attach_status, prepare_attach_styles, restore_attach_status, restore_attach_styles,
    session_refresh_task_failure_status, PendingHostRefreshTask,
};

fn sid(id: &str) -> SessionId {
    SessionId::new(id).unwrap()
}

fn session(name: &str, id: &str) -> SessionInfo {
    session_with_times(name, id, 0, 0)
}

fn session_with_times(name: &str, id: &str, created: u64, activity: u64) -> SessionInfo {
    SessionInfo {
        name: name.to_string(),
        id: sid(id),
        created,
        attached_count: 0,
        window_count: 1,
        group: None,
        activity,
    }
}

fn app_with_session() -> AppState {
    let mut app = AppState::new(LayoutMode::Normal);
    app.session_list.rows = vec![make_row(session("dev", "$1"))];
    app
}

fn local_host_id() -> HostId {
    HostId::local()
}

/// Build a row with `activity_observed_at_local = session.activity`, so
/// helpers that don't care about wall-clock-anchored timing can compose
/// rows that sort by their `session.activity` values.
fn make_row(session: SessionInfo) -> SessionRow {
    make_row_at(session, u64::MAX)
}

/// Build a row as if just observed at `now` (operator wall clock; under the
/// NTP-synced clock assumption this is also effectively the host wall
/// clock). The `activity_observed_at_local` field is seeded as a first-sight
/// observation would be: `min(now, session.activity)`, so the displayed
/// activity reproduces the pre-tracker `now - session.activity` arithmetic.
fn make_row_at(session: SessionInfo, now: u64) -> SessionRow {
    let activity_observed_at_local = now.min(session.activity);
    SessionRow {
        host_id: local_host_id(),
        host_label: "host".to_string(),
        local_now: now,
        activity_observed_at_local,
        session,
        selected_tag: None,
    }
}

fn make_row_for_host(session: SessionInfo, host_id: HostId, host_label: &str) -> SessionRow {
    make_row_for_host_at(session, host_id, host_label, u64::MAX)
}

fn make_row_for_host_at(
    session: SessionInfo,
    host_id: HostId,
    host_label: &str,
    now: u64,
) -> SessionRow {
    let activity_observed_at_local = session.activity;
    SessionRow {
        host_id,
        host_label: host_label.to_string(),
        local_now: now,
        activity_observed_at_local,
        session,
        selected_tag: None,
    }
}

fn with_selected_tag(mut row: SessionRow, value: &str) -> SessionRow {
    row.selected_tag = Some(SessionSelectedTag {
        key: "owner".to_string(),
        value: value.to_string(),
    });
    row
}

fn to_rows(sessions: Vec<SessionInfo>) -> Vec<SessionRow> {
    sessions.into_iter().map(make_row).collect()
}

fn fleet_with(handle: HostHandle) -> HostFleet {
    HostFleet::from_entries(vec![HostEntry {
        id: local_host_id(),
        label: "host".to_string(),
        alias: "host".to_string(),
        ip_address: "unknown".to_string(),
        handle,
    }])
}

fn local_fleet() -> HostFleet {
    fleet_with(HostHandle::local())
}

fn fleet_with_label_ip(label: &str, ip_address: &str) -> HostFleet {
    HostFleet::from_entries(vec![HostEntry {
        id: local_host_id(),
        alias: label.to_string(),
        label: label.to_string(),
        ip_address: ip_address.to_string(),
        handle: HostHandle::local(),
    }])
}

fn make_selected(host_id: HostId, host_label: &str, id: &str, name: &str) -> SelectedSession {
    SelectedSession {
        host_id,
        host_label: host_label.to_string(),
        info: session(name, id),
    }
}

fn test_selected_session() -> SelectedSession {
    make_selected(local_host_id(), "host", "$1", "dev")
}

fn test_new_session_modal(input: &str, button: Button) -> ModalState {
    ModalState::NewSession {
        ui: NewSessionModalUi {
            input: input.to_string(),
            hosts: vec![NewSessionHostChoice {
                id: local_host_id(),
                label: "host".to_string(),
            }],
            host_index: 0,
            env_rows: Vec::new(),
            env_key_input: String::new(),
            env_value_input: String::new(),
            focus: NewSessionFocus::Name,
            button,
        },
    }
}

fn test_send_keys_modal(input: &str, focus: SendKeysFocus) -> ModalState {
    ModalState::SendKeys {
        session: test_selected_session(),
        ui: SendKeysModalUi {
            input: input.to_string(),
            focus,
        },
    }
}

fn test_session_tags_modal(
    tags: Vec<SessionKeyValueRow>,
    selected_key: Option<&str>,
    key_input: &str,
    value_input: &str,
    focus: SessionKeyValueFocus,
) -> ModalState {
    ModalState::SessionKeyValues {
        session: test_selected_session(),
        ui: SessionKeyValueModalUi {
            kind: SessionKeyValueKind::Tags,
            original_rows: tags.clone(),
            rows: tags,
            original_selected_key: selected_key.map(str::to_string),
            selected_key: selected_key.map(str::to_string),
            key_input: key_input.to_string(),
            value_input: value_input.to_string(),
            focus,
        },
    }
}

fn render_to_string(app: &mut AppState, width: u16, height: u16) -> String {
    render_to_lines(app, width, height).join("")
}

fn render_to_string_with_fleet(
    fleet: &HostFleet,
    app: &mut AppState,
    width: u16,
    height: u16,
) -> String {
    render_to_lines_with_fleet(fleet, app, width, height).join("")
}

fn render_to_lines(app: &mut AppState, width: u16, height: u16) -> Vec<String> {
    render_to_lines_and_cursor(app, width, height).0
}

fn render_to_lines_with_fleet(
    fleet: &HostFleet,
    app: &mut AppState,
    width: u16,
    height: u16,
) -> Vec<String> {
    render_to_lines_and_cursor_with_fleet(fleet, app, width, height).0
}

fn render_to_buffer(app: &mut AppState, width: u16, height: u16) -> Buffer {
    let fleet = local_fleet();
    render_to_buffer_with_fleet(&fleet, app, width, height)
}

fn render_to_buffer_with_fleet(
    fleet: &HostFleet,
    app: &mut AppState,
    width: u16,
    height: u16,
) -> Buffer {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| draw(frame, fleet, app)).unwrap();
    terminal.backend().buffer().clone()
}

fn render_to_lines_and_cursor(
    app: &mut AppState,
    width: u16,
    height: u16,
) -> (Vec<String>, Position) {
    let fleet = local_fleet();
    render_to_lines_and_cursor_with_fleet(&fleet, app, width, height)
}

fn render_to_lines_and_cursor_with_fleet(
    fleet: &HostFleet,
    app: &mut AppState,
    width: u16,
    height: u16,
) -> (Vec<String>, Position) {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| draw(frame, fleet, app)).unwrap();
    let cursor = terminal.backend_mut().get_cursor_position().unwrap();
    let lines = terminal
        .backend()
        .buffer()
        .content()
        .chunks(width as usize)
        .map(|row| row.iter().map(|cell| cell.symbol()).collect())
        .collect();
    (lines, cursor)
}

fn modal_border_width(lines: &[String], title: &str) -> usize {
    let border = lines
        .iter()
        .find(|line| line.contains(title) && line.contains('┌') && line.contains('┐'))
        .expect("expected modal top border");
    let left = border
        .chars()
        .position(|ch| ch == '┌')
        .expect("expected modal left corner");
    let right = border
        .chars()
        .enumerate()
        .filter_map(|(index, ch)| (ch == '┐').then_some(index))
        .last()
        .expect("expected modal right corner");
    right - left + 1
}

fn modal_border_left(lines: &[String], title: &str) -> usize {
    lines
        .iter()
        .find(|line| line.contains(title) && line.contains('┌') && line.contains('┐'))
        .and_then(|line| line.chars().position(|ch| ch == '┌'))
        .expect("expected modal left border")
}

fn modal_border_height(lines: &[String], title: &str) -> usize {
    let top = lines
        .iter()
        .position(|line| line.contains(title) && line.contains('┌') && line.contains('┐'))
        .expect("expected modal top border");
    let left = modal_border_left(lines, title);
    let bottom = lines
        .iter()
        .enumerate()
        .skip(top + 1)
        .find_map(|(index, line)| {
            let is_bottom = line.chars().nth(left) == Some('└') && line.contains('┘');
            is_bottom.then_some(index)
        })
        .expect("expected modal bottom border");
    bottom - top + 1
}

fn line_char_index(line: &str, needle: &str) -> usize {
    let byte_index = line.find(needle).expect("expected text in rendered line");
    line[..byte_index].chars().count()
}

#[test]
fn render_applies_configured_base_colors_to_unstyled_cells() {
    let mut app = app_with_session();
    let buffer = render_to_buffer(&mut app, 80, 24);
    let cell = &buffer[(70, 5)];

    match APP_BASE_FG {
        Some(fg) => assert_eq!(cell.fg, fg),
        None => assert_eq!(cell.fg, Color::Reset),
    }
    match APP_BASE_BG {
        Some(bg) => assert_eq!(cell.bg, bg),
        None => assert_eq!(cell.bg, Color::Reset),
    }
}

#[test]
fn modal_clear_area_preserves_configured_base_background() {
    let mut app = app_with_session();
    app.modal = Some(test_send_keys_modal("", SendKeysFocus::Input));
    let buffer = render_to_buffer(&mut app, 80, 24);

    if let Some(bg) = APP_BASE_BG {
        assert_eq!(buffer[(40, 12)].bg, bg);
        assert!(buffer.content().iter().all(|cell| cell.bg != Color::Reset));
    }
}

#[test]
fn attach_styles_are_derived_from_mmux_theme() {
    assert_eq!(mmux_attach_status_style(None), "bg=#002b55,fg=white");
    assert_eq!(
        mmux_attach_status_style(Some(HOST_COLOR_PALETTE[1])),
        "bg=#ffc107,fg=black"
    );
    assert_eq!(
        mmux_attach_window_style(),
        Some("bg=black,fg=white".to_string())
    );
}

#[test]
fn cli_accepts_script_and_rejects_removed_mode_flags() {
    let script = Cli::try_parse_from(["mmux", "--script"]).unwrap();
    assert!(script.script);

    let print_session = Cli::try_parse_from(["mmux", "--print-session"]);
    assert!(print_session.is_err());

    let dashboard = Cli::try_parse_from(["mmux", "--dashboard"]);
    assert!(dashboard.is_err());

    Cli::command().debug_assert();
}

#[test]
fn cli_accepts_layout_force_flags_and_rejects_old_short_flag() {
    let portrait = Cli::try_parse_from(["mmux", "--portrait"]).unwrap();
    assert!(portrait.portrait);
    assert_eq!(portrait.forced_layout(), Some(LayoutMode::Portrait));

    let portrait_short = Cli::try_parse_from(["mmux", "-p"]).unwrap();
    assert!(portrait_short.portrait);

    let landscape = Cli::try_parse_from(["mmux", "--landscape"]).unwrap();
    assert!(landscape.landscape);
    assert_eq!(landscape.forced_layout(), Some(LayoutMode::Normal));

    let landscape_short = Cli::try_parse_from(["mmux", "-l"]).unwrap();
    assert!(landscape_short.landscape);

    let old_short = Cli::try_parse_from(["mmux", "-s"]);
    assert!(old_short.is_err());

    let conflicting = Cli::try_parse_from(["mmux", "--portrait", "--landscape"]);
    assert!(conflicting.is_err());
}

#[test]
fn layout_auto_detection_uses_pty_aspect_ratio() {
    assert_eq!(
        select_layout(Some(LayoutMode::Portrait)),
        LayoutMode::Portrait
    );
    assert_eq!(select_layout(Some(LayoutMode::Normal)), LayoutMode::Normal);
    assert!(is_portrait_pty(64, 32));
    assert!(is_portrait_pty(60, 30));
    assert!(is_portrait_pty(66, 30));
    assert!(is_portrait_pty(80, 24));
    assert!(is_portrait_pty(100, 30));
    assert!(is_portrait_pty(160, 40));
    assert!(is_portrait_pty(40, 40));
    assert!(!is_portrait_pty(161, 40));
    assert!(!is_portrait_pty(200, 40));
}

#[test]
fn portrait_default_split_is_35_65() {
    let app = AppState::new(LayoutMode::Portrait);

    assert_eq!(app.layout.top_percent, 35);
}

#[test]
fn status_line_omits_layout_mode() {
    let normal = AppState::new(LayoutMode::Normal);
    let normal_status = status_line_text(&normal);
    assert!(normal_status.contains(" tab ↑/↓"));
    assert!(!normal_status.contains(" ↑/↓ sel"));
    assert!(!normal_status.contains("keys"));
    assert!(!normal_status.contains("host"));
    assert!(!normal_status.contains("(h)elp"));
    assert!(!normal_status.contains("(s)end"));
    assert!(!normal_status.contains("tab pane"));
    assert_status_order(
        &normal_status,
        &[
            "help",
            "monitor",
            "prompt",
            "attach",
            "new",
            "kill",
            "rename",
            "group",
            "layout",
            "quit",
            "mod-←/→ resize",
        ],
    );
    assert!(!normal_status.contains("normal"));
    assert!(!normal_status.contains("landscape"));
    assert!(!normal_status.contains("portrait"));

    let portrait = AppState::new(LayoutMode::Portrait);
    let portrait_status = status_line_text(&portrait);
    assert!(portrait_status.contains(" tab ↑/↓"));
    assert!(!portrait_status.contains(" ↑/↓ sel"));
    assert!(!portrait_status.contains("tab pane"));
    assert_status_order(
        &portrait_status,
        &[
            "help",
            "monitor",
            "prompt",
            "attach",
            "new",
            "kill",
            "rename",
            "group",
            "layout",
            "quit",
            "mod-↑/↓ resize",
        ],
    );
}

#[test]
fn status_line_styles_command_mnemonics() {
    let app = AppState::new(LayoutMode::Normal);
    let line = status_line(&app);
    let styled_mnemonics = line
        .spans
        .iter()
        .filter(|span| {
            span.style.add_modifier.contains(Modifier::BOLD)
                && span.style.fg == Some(STATUS_BAR_MNEMONIC_FG)
                && !span.style.add_modifier.contains(Modifier::UNDERLINED)
        })
        .map(|span| span.content.as_ref())
        .collect::<String>();

    assert_eq!(styled_mnemonics, "hmpankrglq");
    assert_eq!(
        status_line_text(&app),
        " tab ↑/↓ | help | monitor | prompt | attach | new | kill | rename | group | layout | quit | mod-←/→ resize "
    );
}

#[test]
fn session_tags_footer_styles_tag_command_mnemonics() {
    let line = session_key_values_footer_line(SessionKeyValueKind::Tags, "[Cancel]   Ok ");
    let text = line
        .spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>();
    let styled_mnemonics = line
        .spans
        .iter()
        .filter(|span| {
            span.style.add_modifier.contains(Modifier::BOLD)
                && span.style.fg == Some(STATUS_BAR_MNEMONIC_FG)
                && span.style.bg == Some(STATUS_BAR_BG)
        })
        .map(|span| span.content.as_ref())
        .collect::<String>();

    assert_eq!(text, "[Cancel]   Ok  | modify | x unset | check");
    assert_eq!(styled_mnemonics, "mxc");
}

fn assert_status_order(status: &str, tokens: &[&str]) {
    let mut last = 0;
    for token in tokens {
        let pos = status.find(token).unwrap_or_else(|| {
            panic!("missing status token {token:?} in {status:?}");
        });
        assert!(
            pos >= last,
            "status token {token:?} is out of order in {status:?}"
        );
        last = pos;
    }
}

#[test]
fn top_status_includes_bold_host_and_right_justified_time() {
    let fleet = fleet_with_label_ip("target-host", "192.0.2.10");
    let line = top_status_line(&fleet, "12:34:56", 40);
    let rendered = line
        .spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>();

    assert_eq!(rendered.chars().count(), 40);
    assert!(rendered.starts_with(" target-host | 192.0.2.10 "));
    assert!(line.spans[0].style.add_modifier.contains(Modifier::BOLD));
    assert!(rendered.ends_with(" 12:34:56 "));
}

#[test]
fn sessions_title_only_includes_count() {
    let mut app = AppState::new(LayoutMode::Normal);
    app.session_list.rows = to_rows(vec![session("dev", "$1"), session("build", "$2")]);

    assert_eq!(sessions_title(&app), " Sessions [2] ");
}

#[test]
fn session_list_line_hides_stable_id() {
    let line = session_list_line(&make_row(session("dev", "$42")), true, None, 0, 16);
    assert!(line.starts_with(">  dev"));
    assert!(!line.contains("$42"));
}

#[test]
fn session_list_line_right_justifies_selected_tag_value() {
    let mut row = make_row_at(session_with_times("dev", "$42", 100, 110), 120);
    row.selected_tag = Some(SessionSelectedTag {
        key: "owner".to_string(),
        value: "platform".to_string(),
    });

    let line = session_list_line(&row, true, None, 0, 48);
    let metadata = session_recency_text(&row);
    let tag_start = line.find("platform").unwrap();
    let activity = metadata.split(" / ").next().unwrap().trim();
    let activity_start = line.find(activity).unwrap();

    assert!(line.contains("dev"), "{line:?}");
    assert!(line.contains("platform"), "{line:?}");
    assert!(!line.contains("dev platform"), "{line:?}");
    assert_eq!(
        activity_start.saturating_sub(tag_start + "platform".len()),
        2
    );
    assert!(!line.contains("owner"));
}

#[tokio::test]
async fn refresh_sessions_loads_selected_tag_value_for_rows() {
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response(
            "display-message -p '__MOTLIE_TAGS__ $1'",
            "__MOTLIE_TAGS__ $1\n@mmux/__selected-key owner\n@mmux/owner platform\n",
        );
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = AppState::new(LayoutMode::Normal);

    refresh_sessions_quiet(&fleet, &mut app, false)
        .await
        .unwrap();

    assert!(matches!(
        app.session_list.rows.first().and_then(|row| row.selected_tag.as_ref()),
        Some(SessionSelectedTag { key, value }) if key == "owner" && value == "platform"
    ));
}

#[tokio::test]
async fn apply_fleet_snapshot_can_defer_initial_detail_capture() {
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_error("capture-pane", "initial detail must be deferred");
    let fleet = fleet_with(HostHandle::new(TransportKind::Mock(mock), None));
    let mut app = AppState::new(LayoutMode::Normal);

    let refresh = fetch_fleet_refresh(&fleet).await;
    apply_fleet_snapshot(
        &fleet,
        &mut app,
        refresh,
        RefreshApplyOptions::initial(None),
    )
    .await
    .unwrap();

    assert_eq!(
        app.selected_session()
            .map(|session| session.name().to_string()),
        Some("dev".to_string())
    );
    assert!(app.detail.lines.is_empty());
    assert_eq!(app.status.text(), "1 session(s)");
}

#[tokio::test]
async fn apply_fleet_snapshot_without_previous_preserves_current_selection() {
    let mock = MockTransport::new().with_response(
        "list-sessions",
        "__MOTLIE_S__ selected $1 10 0 1  100\n__MOTLIE_S__ fresh $2 20 0 1  500\n",
    );
    let fleet = fleet_with(HostHandle::new(TransportKind::Mock(mock), None));
    let mut app = AppState::new(LayoutMode::Normal);
    app.session_list.rows = to_rows(vec![
        session_with_times("selected", "$1", 10, 100),
        session_with_times("fresh", "$2", 20, 50),
    ]);
    app.session_list.selected = 0;

    let refresh = fetch_fleet_refresh(&fleet).await;
    apply_fleet_snapshot(&fleet, &mut app, refresh, RefreshApplyOptions::periodic())
        .await
        .unwrap();

    assert_eq!(
        app.session_list
            .rows
            .iter()
            .map(|row| row.session.name.as_str())
            .collect::<Vec<_>>(),
        vec!["fresh", "selected"]
    );
    assert_eq!(
        app.selected_session()
            .map(|session| session.name().to_string()),
        Some("selected".to_string())
    );
}

#[tokio::test]
async fn apply_streaming_host_results_batch_successes_and_keeps_failed_rows() {
    let a_entry = ssh_host_entry(
        "ssh://a",
        "alpha",
        "x",
        HostHandle::new(
            TransportKind::Mock(
                MockTransport::new()
                    .with_response("list-sessions", "__MOTLIE_S__ a-new $1 10 0 1  900\n"),
            ),
            None,
        ),
    );
    let b_entry = ssh_host_entry(
        "ssh://b",
        "beta",
        "y",
        HostHandle::new(
            TransportKind::Mock(
                MockTransport::new().with_error("list-sessions", "transport: connection refused"),
            ),
            None,
        ),
    );
    let fleet = HostFleet::from_entries(vec![a_entry.clone(), b_entry.clone()]);
    let mut app = AppState::new(LayoutMode::Normal);
    app.session_list.rows = vec![
        make_row_for_host(
            session_with_times("a-old", "$1", 10, 100),
            ssh_host_id("ssh://a"),
            "alpha",
        ),
        make_row_for_host(
            session_with_times("b-stale", "$2", 20, 200),
            ssh_host_id("ssh://b"),
            "beta",
        ),
    ];
    app.session_list.selected = 1;

    let a_refresh = fetch_host_refresh(&a_entry).await;
    let b_refresh = fetch_host_refresh(&b_entry).await;
    apply_streaming_host_results(
        &fleet,
        &mut app,
        vec![a_refresh, b_refresh],
        RefreshApplyOptions::after_action(false, None, true, None),
    )
    .await
    .unwrap();

    assert_eq!(
        app.session_list
            .rows
            .iter()
            .map(|row| (row.host_label.as_str(), row.session.name.as_str()))
            .collect::<Vec<_>>(),
        vec![("alpha", "a-new"), ("beta", "b-stale")]
    );
    assert_eq!(
        app.selected_session()
            .map(|session| session.name().to_string()),
        Some("b-stale".to_string())
    );
    assert!(app.status.text().contains("host unreachable: beta"));
}

#[tokio::test]
async fn failed_host_refresh_keeps_existing_rows_visible() {
    let entry = ssh_host_entry(
        "ssh://down",
        "down-host",
        "x",
        HostHandle::new(
            TransportKind::Mock(
                MockTransport::new().with_error("list-sessions", "transport: connection refused"),
            ),
            None,
        ),
    );
    let fleet = HostFleet::from_entries(vec![entry.clone()]);
    let mut app = AppState::new(LayoutMode::Normal);
    app.session_list.rows = vec![make_row_for_host(
        session_with_times("stale", "$1", 10, 100),
        ssh_host_id("ssh://down"),
        "down-host",
    )];

    let refresh = fetch_host_refresh(&entry).await;
    apply_streaming_host_results(
        &fleet,
        &mut app,
        vec![refresh],
        RefreshApplyOptions::after_action(false, None, true, None),
    )
    .await
    .unwrap();

    assert_eq!(app.session_list.rows.len(), 1);
    assert_eq!(app.session_list.rows[0].session.name, "stale");
    assert!(app.status.text().contains("host unreachable: down-host"));
}

#[tokio::test]
async fn session_refresh_task_failure_status_names_panicked_host() {
    let status = session_refresh_task_failure_status(vec![PendingHostRefreshTask {
        diagnostic_label: "alpha".to_string(),
        handle: tokio::spawn(async {
            panic!("boom");
        }),
    }])
    .await;

    assert!(status.contains("session refresh task failed"));
    assert!(status.contains("alpha panicked"));
}

#[test]
fn session_list_sorts_most_recent_activity_first() {
    let mut app = AppState::new(LayoutMode::Normal);

    app.session_list.set_rows_sorted_by_activity(vec![
        make_row_at(session_with_times("older", "$1", 10, 100), 400),
        make_row_at(session_with_times("fresh", "$2", 20, 300), 400),
        make_row_at(session_with_times("alpha", "$3", 30, 200), 400),
        make_row_at(session_with_times("beta", "$4", 40, 200), 400),
    ]);

    let names = app
        .session_list
        .rows
        .iter()
        .map(|row| row.session.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["fresh", "alpha", "beta", "older"]);
}

#[test]
fn activity_sort_is_stable_within_visible_recency_bucket() {
    let mut app = AppState::new(LayoutMode::Normal);

    app.session_list.set_rows_sorted_by_activity(vec![
        make_row_at(session_with_times("zeta", "$1", 10, 999), 1_000),
        make_row_at(session_with_times("alpha", "$2", 10, 950), 1_000),
    ]);

    let names = app
        .session_list
        .rows
        .iter()
        .map(|row| row.session.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["alpha", "zeta"]);
}

#[test]
fn session_list_tag_group_orders_groups_by_recent_activity() {
    let fleet = HostFleet::from_entries(vec![
        ssh_host_entry("ssh://a", "alpha", "x", HostHandle::local()),
        ssh_host_entry("ssh://b", "beta", "y", HostHandle::local()),
    ]);
    let mut app = AppState::new(LayoutMode::Normal);
    let rows = vec![
        make_row_for_host_at(
            session_with_times("no-tag-fresh", "$1", 10, 900),
            ssh_host_id("ssh://a"),
            "alpha",
            1_000,
        ),
        with_selected_tag(
            make_row_for_host_at(
                session_with_times("b-alpha-low", "$2", 10, 100),
                ssh_host_id("ssh://b"),
                "beta",
                1_000,
            ),
            "alpha",
        ),
        with_selected_tag(
            make_row_for_host_at(
                session_with_times("a-alpha-high", "$3", 10, 300),
                ssh_host_id("ssh://a"),
                "alpha",
                1_000,
            ),
            "alpha",
        ),
        with_selected_tag(
            make_row_for_host_at(
                session_with_times("b-same", "$4", 10, 400),
                ssh_host_id("ssh://b"),
                "beta",
                1_000,
            ),
            "same",
        ),
        with_selected_tag(
            make_row_for_host_at(
                session_with_times("z-same", "$5", 10, 400),
                ssh_host_id("ssh://a"),
                "alpha",
                1_000,
            ),
            "same",
        ),
        with_selected_tag(
            make_row_for_host_at(
                session_with_times("empty-selected-tag", "$7", 10, 800),
                ssh_host_id("ssh://a"),
                "alpha",
                1_000,
            ),
            "",
        ),
        make_row_for_host_at(
            session_with_times("no-tag-old", "$6", 10, 100),
            ssh_host_id("ssh://b"),
            "beta",
            1_000,
        ),
    ];

    app.session_list.set_rows_grouped_by_tag(rows, &fleet);

    let names = app
        .session_list
        .rows
        .iter()
        .map(|row| row.session.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        vec![
            "z-same",
            "b-same",
            "a-alpha-high",
            "b-alpha-low",
            "no-tag-fresh",
            "empty-selected-tag",
            "no-tag-old"
        ]
    );
}

#[test]
fn activity_sort_preserves_selection_by_stable_id() {
    let mut app = AppState::new(LayoutMode::Normal);
    app.session_list.rows = to_rows(vec![
        session_with_times("selected", "$1", 10, 100),
        session_with_times("other", "$2", 20, 200),
    ]);
    app.session_list.selected = 0;
    let previous = app
        .selected_session()
        .map(|session| (session.host_id.clone(), session.id().to_string()));

    app.session_list.set_rows_sorted_by_activity(to_rows(vec![
        session_with_times("selected", "$1", 10, 100),
        session_with_times("other", "$2", 20, 300),
    ]));
    app.preserve_selection(previous);

    assert_eq!(
        app.selected_session()
            .map(|session| session.name().to_string()),
        Some("selected".to_string())
    );
}

#[tokio::test]
async fn g_toggles_tag_grouping_from_list_focus_and_selects_top_row() {
    let fleet = local_fleet();
    let mut app = app_with_session();
    app.session_list.rows = vec![
        with_selected_tag(
            make_row(session_with_times("selected", "$1", 10, 300)),
            "zeta",
        ),
        with_selected_tag(
            make_row(session_with_times("other", "$2", 10, 400)),
            "alpha",
        ),
    ];
    app.session_list.selected = 0;
    app.layout.focus = Focus::Detail;

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('g'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert_eq!(app.session_list.sort_mode, SessionSortMode::Activity);
    assert_eq!(app.session_list.selected, 0);

    app.layout.focus = Focus::List;
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('g'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert_eq!(app.session_list.sort_mode, SessionSortMode::TagGroup);
    assert_eq!(app.status.text(), "group: tag");
    assert_eq!(app.session_list.selected, 0);
    assert_eq!(
        app.session_list
            .rows
            .iter()
            .map(|row| row.session.name.as_str())
            .collect::<Vec<_>>(),
        vec!["other", "selected"]
    );
    assert_eq!(
        app.selected_session()
            .map(|session| session.name().to_string()),
        Some("other".to_string())
    );

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('g'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert_eq!(app.session_list.sort_mode, SessionSortMode::Activity);
    assert_eq!(app.status.text(), "sort: activity");
    assert_eq!(app.session_list.selected, 0);
    assert_eq!(
        app.session_list
            .rows
            .iter()
            .map(|row| row.session.name.as_str())
            .collect::<Vec<_>>(),
        vec!["other", "selected"]
    );
    assert_eq!(
        app.selected_session()
            .map(|session| session.name().to_string()),
        Some("other".to_string())
    );
}

#[tokio::test]
async fn quiet_refresh_preserves_tag_group_mode() {
    let mock = MockTransport::new()
        .with_response(
            "list-sessions",
            "__MOTLIE_S__ zed $1 10 0 1  500\n__MOTLIE_S__ alpha $2 10 0 1  400\n",
        )
        .with_response(
            "display-message -p '__MOTLIE_TAGS__ $2'",
            "__MOTLIE_TAGS__ $1\n@mmux/__selected-key owner\n@mmux/owner zeta\n__MOTLIE_TAGS__ $2\n@mmux/__selected-key owner\n@mmux/owner alpha\n",
        );
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = AppState::new(LayoutMode::Normal);
    app.session_list.sort_mode = SessionSortMode::TagGroup;

    refresh_sessions_quiet(&fleet, &mut app, false)
        .await
        .unwrap();

    assert_eq!(
        app.session_list
            .rows
            .iter()
            .map(|row| row.session.name.as_str())
            .collect::<Vec<_>>(),
        vec!["alpha", "zed"]
    );
}

#[tokio::test]
async fn quiet_refresh_reorders_by_activity_without_overwriting_status() {
    let mock = MockTransport::new()
        .with_response(
            "list-sessions",
            "__MOTLIE_S__ older $1 10 0 1  100\n__MOTLIE_S__ fresh $2 20 0 1  400\n",
        )
        .with_response("capture-pane -ep", "fresh screen\n");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let mut app = AppState::new(LayoutMode::Normal);
    app.status = crate::model::StatusBanner::info("keep this");

    let fleet = fleet_with(host);
    refresh_sessions_quiet(&fleet, &mut app, false)
        .await
        .unwrap();

    let names = app
        .session_list
        .rows
        .iter()
        .map(|row| row.session.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["fresh", "older"]);
    assert_eq!(app.status.text(), "keep this");
}

#[tokio::test]
async fn refresh_forces_detail_when_selected_session_changes() {
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ replacement $2 20 0 1  400\n")
        .with_response("capture-pane -ep", "replacement screen\n");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let mut app = AppState::new(LayoutMode::Normal);
    app.session_list.rows = to_rows(vec![session_with_times("gone", "$1", 10, 300)]);
    app.session_list.selected = 0;
    app.set_detail_text("stale screen".to_string());

    let fleet = fleet_with(host.clone());
    refresh_sessions_preserving(
        &fleet,
        &mut app,
        false,
        Some((local_host_id(), "$1".to_string())),
    )
    .await
    .unwrap();

    assert_eq!(
        app.selected_session()
            .map(|session| session.name().to_string()),
        Some("replacement".to_string())
    );
    assert_eq!(app.detail.lines, vec!["replacement screen".to_string()]);
}

#[tokio::test]
async fn quiet_refresh_stops_monitor_when_monitored_session_closes() {
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ replacement $2 20 0 1  400\n")
        .with_response("capture-pane -ep", "replacement screen\n");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let mut app = AppState::new(LayoutMode::Normal);
    app.session_list.rows = to_rows(vec![
        session_with_times("watched", "$1", 10, 300),
        session_with_times("replacement", "$2", 20, 200),
    ]);
    app.detail.source = DetailSource::Monitor(Box::new(MonitorDetailSource {
        session_id: Some("$1".to_string()),
        host_id: Some(local_host_id()),
    }));
    app.detail.lines = vec!["live".to_string()];

    let fleet = fleet_with(host.clone());
    refresh_sessions_quiet(&fleet, &mut app, false)
        .await
        .unwrap();

    assert_eq!(app.detail.source.mode(), DetailMode::Sample);
    assert_eq!(app.status.text(), "monitored session watched closed");
    assert_eq!(app.detail.lines, vec!["replacement screen".to_string()]);
}

#[tokio::test]
async fn quiet_refresh_skips_monitor_recapture() {
    // Reviewer regression test: when monitor mode is active and a quiet
    // session refresh sees newer `session_activity`, the row list must
    // update without re-rendering the monitor detail. The previous
    // behavior unconditionally called `refresh_detail` after every quiet
    // refresh; in `Monitor` mode that recaptures the pane every tick,
    // blocking the next draw and making the row activity look stale.
    //
    // Setup: monitor is active on the highlighted session $1. list-sessions
    // returns a fresher `session_activity` for $1 (200 → 800). capture-pane
    // is set with_error so any call triggered by refresh_detail would land
    // a visible "monitor error" string in detail.lines.
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ live $1 10 1 1  800\n")
        .with_error("capture-pane", "this should not be called");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let mut app = AppState::new(LayoutMode::Normal);
    // Seed the row list with the older activity (200) and select $1.
    app.session_list.rows = to_rows(vec![session_with_times("live", "$1", 10, 200)]);
    app.session_list.selected = 0;
    // Activate monitor mode on $1 with non-empty pre-existing detail lines.
    app.detail.source = DetailSource::Monitor(Box::new(MonitorDetailSource {
        session_id: Some("$1".to_string()),
        host_id: Some(local_host_id()),
    }));
    app.detail.lines = vec!["pre-existing live capture".to_string()];

    let fleet = fleet_with(host);
    refresh_sessions_quiet(&fleet, &mut app, false)
        .await
        .unwrap();

    // Row activity reflects the newer list-sessions snapshot.
    assert_eq!(app.session_list.rows.len(), 1);
    assert_eq!(app.session_list.rows[0].session.activity, 800);
    // Monitor mode is still active and the detail lines are untouched —
    // refresh_detail was not called from the session-refresh path.
    assert_eq!(app.detail.source.mode(), DetailMode::Monitor);
    assert_eq!(
        app.detail.lines,
        vec!["pre-existing live capture".to_string()],
        "monitor detail must not be recaptured by a quiet session refresh"
    );
}

#[test]
fn session_list_line_right_aligns_active_and_age() {
    let now = 7_200;
    let first = session_list_line(
        &make_row_at(session_with_times("dev", "$1", 0, 7_020), now),
        true,
        None,
        0,
        42,
    );
    let second = session_list_line(
        &make_row_at(
            session_with_times("longer-build-name", "$2", 3_600, 4_560),
            now,
        ),
        false,
        None,
        0,
        42,
    );

    assert_eq!(first.chars().count(), 42);
    assert_eq!(second.chars().count(), 42);
    assert_eq!(first.find(" / "), second.find(" / "));
    assert!(first.ends_with("   3m /    2h  "));
    assert!(second.ends_with("  44m /    1h  "));
}

#[test]
fn session_recency_uses_now_minutes_and_hours() {
    let session = session_with_times("dev", "$1", 60, 3_590);

    assert_eq!(
        session_recency_text(&make_row_at(session.clone(), 3_600)),
        "  now /   59m"
    );
    assert_eq!(
        session_recency_text(&make_row_at(session, 7_260)),
        "   1h /    2h"
    );
}

#[test]
fn session_recency_uses_days_for_long_durations() {
    let now = 340 * 60 * 60;
    let session = session_with_times("dev", "$1", 0, now - 32 * 60 * 60);

    assert_eq!(
        session_recency_text(&make_row_at(session, now)),
        "  32h / 14.2d"
    );
}

#[test]
fn resolve_ip_address_preserves_literal_ip() {
    assert_eq!(
        resolve_ip_address("192.0.2.10", SSH_DEFAULT_PORT),
        "192.0.2.10"
    );
}

#[test]
fn selection_preserves_stable_id_after_rename() {
    let mut app = AppState::new(LayoutMode::Normal);
    app.session_list.rows = vec![make_row(session("old", "$1"))];
    app.session_list.selected = 0;
    app.session_list.rows = vec![make_row(session("new", "$1"))];
    app.preserve_selection(Some((local_host_id(), "$1".to_string())));
    assert_eq!(app.session_list.selected, 0);
    assert_eq!(
        app.selected_session().map(|s| s.name().to_string()),
        Some("new".to_string())
    );
}

#[test]
fn retained_ui_state_restores_selection_layout_and_split_on_reentry() {
    let mut app = AppState::new(LayoutMode::Normal);
    app.session_list.rows = to_rows(vec![session("shell", "$1"), session("build", "$2")]);
    app.session_list.selected = 1;
    app.layout.focus = Focus::Detail;
    app.layout.mode = LayoutMode::Portrait;
    app.layout.left_percent = 55;
    app.layout.top_percent = 45;

    let mut retained = crate::model::RetainedUiState::default();
    retained.update_from(&app);

    let mut reentered = AppState::new(LayoutMode::Normal);
    retained.apply_to(&mut reentered);
    reentered.session_list.rows = to_rows(vec![session("build", "$2"), session("shell", "$1")]);
    reentered.preserve_selection(retained.selected_session_key());

    assert_eq!(
        reentered
            .selected_session()
            .map(|session| session.name().to_string()),
        Some("build".to_string())
    );
    assert_eq!(reentered.layout.mode, LayoutMode::Portrait);
    assert_eq!(reentered.layout.focus, Focus::Detail);
    assert_eq!(reentered.layout.left_percent, 55);
    assert_eq!(reentered.layout.top_percent, 45);
}

#[test]
fn retained_ui_state_falls_back_to_previous_index_when_session_disappears() {
    let mut app = AppState::new(LayoutMode::Normal);
    app.session_list.rows = to_rows(vec![
        session("one", "$1"),
        session("gone", "$2"),
        session("three", "$3"),
    ]);
    app.session_list.selected = 1;

    let mut retained = crate::model::RetainedUiState::default();
    retained.update_from(&app);

    let mut reentered = AppState::new(LayoutMode::Normal);
    retained.apply_to(&mut reentered);
    reentered.session_list.rows = to_rows(vec![session("one", "$1"), session("three", "$3")]);
    reentered.preserve_selection(retained.selected_session_key());

    assert_eq!(reentered.session_list.selected, 1);
    assert_eq!(
        reentered
            .selected_session()
            .map(|session| session.name().to_string()),
        Some("three".to_string())
    );
}

#[test]
fn landscape_layout_renders_sessions_and_detail_without_motd() {
    let mut app = AppState::new(LayoutMode::Normal);
    app.session_list.rows = vec![make_row(session("dev", "$1"))];

    let rendered = render_to_string(&mut app, 120, 30);

    assert!(rendered.contains("Sessions [1]"));
    assert!(rendered.contains("Detail"));
    assert!(!rendered.contains("MOTD"));
    assert!(!rendered.contains(MOTLIE_PLACEHOLDER));
}

#[test]
fn portrait_layout_renders_sessions_and_detail_without_motd() {
    let mut app = AppState::new(LayoutMode::Portrait);
    app.session_list.rows = vec![make_row(session("dev", "$1"))];

    let rendered = render_to_string(&mut app, 100, 30);

    assert!(rendered.contains("Sessions [1]"));
    assert!(rendered.contains("Detail"));
    assert!(!rendered.contains("MOTD"));
    assert!(!rendered.contains(MOTLIE_PLACEHOLDER));
}

#[tokio::test]
async fn closed_monitored_session_resets_detail_source() {
    let mut app = AppState::new(LayoutMode::Normal);
    app.detail.source = DetailSource::Monitor(Box::new(MonitorDetailSource {
        session_id: Some("$1".to_string()),
        host_id: Some(local_host_id()),
    }));
    app.detail.lines = vec!["live".to_string()];

    let host_id = local_host_id();
    let closed = stop_monitor_if_closed(&mut app, &host_id, "$1", "dev".to_string()).await;

    assert_eq!(closed, Some("dev".to_string()));
    assert_eq!(app.detail.source.mode(), DetailMode::Sample);
    assert!(app.detail.lines.is_empty());
}

#[test]
fn detail_scroll_up_moves_toward_older_content() {
    let mut app = AppState::new(LayoutMode::Normal);
    app.detail.lines = (0..100).map(|n| format!("line {n}")).collect();
    app.detail.last_known_view_height = 10;
    app.detail.last_known_scroll_max = 90;

    app.scroll_detail(1);
    assert_eq!(app.detail.scroll, 1);
    assert!(!app.detail.auto_tail);

    app.scroll_detail(-1);
    assert_eq!(app.detail.scroll, 0);
    assert!(app.detail.auto_tail);
}

#[test]
fn detail_tail_view_accounts_for_wrapped_lines() {
    let mut app = AppState::new(LayoutMode::Normal);
    app.session_list.rows = vec![make_row(session("dev", "$1"))];
    app.detail.lines = vec![
        "older".to_string(),
        "wide-content ".repeat(8),
        "recent-1".to_string(),
        "recent-2".to_string(),
        "recent-3".to_string(),
    ];

    let rendered = render_to_string(&mut app, 42, 8);

    assert!(rendered.contains("recent-1"));
    assert!(rendered.contains("recent-2"));
    assert!(rendered.contains("recent-3"));
}

#[test]
fn detail_home_reaches_oldest_wrapped_content() {
    let mut app = AppState::new(LayoutMode::Normal);
    app.session_list.rows = vec![make_row(session("dev", "$1"))];
    app.detail.lines = vec![
        "oldest".to_string(),
        "wide-content ".repeat(8),
        "recent-1".to_string(),
        "recent-2".to_string(),
        "recent-3".to_string(),
    ];

    render_to_string(&mut app, 42, 8);
    app.detail_home();
    let rendered = render_to_string(&mut app, 42, 8);

    assert!(rendered.contains("oldest"));
}

#[test]
fn detail_wrapped_row_count_matches_ratatui_paragraph_wrap() {
    let lines = vec![
        "alpha beta gamma".to_string(),
        "wide \x1b[32mgreen\x1b[0m end".to_string(),
        "cjk 界界 done".to_string(),
    ];
    let width = 7;
    let expected_rows = detail_total_wrapped_rows(&lines, width);
    let backend = TestBackend::new(width as u16, (expected_rows + 3) as u16);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal
        .draw(|frame| {
            frame.render_widget(
                Paragraph::new(detail_lines_text_for_render(&lines)).wrap(Wrap { trim: false }),
                frame.area(),
            );
        })
        .unwrap();

    let rendered_rows = terminal
        .backend()
        .buffer()
        .content()
        .chunks(width)
        .map(|row| row.iter().map(|cell| cell.symbol()).collect::<String>())
        .filter(|row| !row.trim().is_empty())
        .count();

    assert_eq!(rendered_rows, expected_rows);
}

#[tokio::test]
async fn q_exits_like_ctrl_c() {
    let fleet = local_fleet();
    let mut app = AppState::new(LayoutMode::Normal);

    let outcome = handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(matches!(outcome, KeyOutcome::Cancel));
}

#[tokio::test]
async fn a_attaches_selected_session() {
    let fleet = local_fleet();
    let mut app = app_with_session();

    let outcome = handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(matches!(outcome, KeyOutcome::Select(selected) if selected.name() == "dev"));
}

#[tokio::test]
async fn u_and_b_move_list_selection_like_arrow_keys() {
    let listing = "__MOTLIE_S__ top $1 10 0 1  100\n\
                   __MOTLIE_S__ middle $2 10 0 1  100\n\
                   __MOTLIE_S__ bottom $3 10 0 1  100\n";
    let mock = MockTransport::new()
        .with_response("list-sessions", listing)
        .with_response("capture-pane -ep", "top screen\n")
        .with_response("list-sessions", listing)
        .with_response("capture-pane -ep", "middle screen\n");
    let fleet = fleet_with(HostHandle::new(TransportKind::Mock(mock), None));
    let mut app = AppState::new(LayoutMode::Normal);
    app.session_list.rows = to_rows(vec![
        session("top", "$1"),
        session("middle", "$2"),
        session("bottom", "$3"),
    ]);
    app.session_list.selected = 1;
    app.layout.focus = Focus::List;

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('u'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert_eq!(app.selected_session().unwrap().name(), "top");
    assert_eq!(app.detail.lines, vec!["top screen".to_string()]);

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('b'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert_eq!(app.selected_session().unwrap().name(), "middle");
    assert_eq!(app.detail.lines, vec!["middle screen".to_string()]);
}

#[tokio::test]
async fn u_and_b_scroll_detail_like_arrow_keys() {
    let fleet = local_fleet();
    let mut app = AppState::new(LayoutMode::Normal);
    app.layout.focus = Focus::Detail;
    app.detail.lines = (0..100).map(|n| format!("line {n}")).collect();
    app.detail.last_known_view_height = 10;
    app.detail.last_known_scroll_max = 90;

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('u'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert_eq!(app.detail.scroll, 1);
    assert!(!app.detail.auto_tail);

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('b'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert_eq!(app.detail.scroll, 0);
    assert!(app.detail.auto_tail);
}

#[tokio::test]
async fn enter_from_list_refreshes_sample_detail_without_attaching() {
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response("capture-pane -ep", "current screen\n");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = app_with_session();
    app.detail.source = DetailSource::Monitor(Box::new(MonitorDetailSource {
        session_id: Some("$1".to_string()),
        host_id: Some(local_host_id()),
    }));
    app.set_detail_text("stale monitor screen".to_string());

    let outcome = handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(matches!(outcome, KeyOutcome::Continue));
    assert_eq!(app.detail.source.mode(), DetailMode::Sample);
    assert_eq!(app.detail.lines, vec!["current screen".to_string()]);
}

#[tokio::test]
async fn attach_status_overrides_are_set_and_restored() {
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response(
            "show-option -q -t '$1' status-style",
            "status-style bg=green,fg=black\n",
        )
        .with_response(
            "show-option -q -t '$1' status-left-length",
            "status-left-length 32\n",
        )
        .with_response(
            "show-option -q -t '$1' status-left",
            "status-left \"session: #{session_name}\"\n",
        )
        .with_response(
            &format!(
                "set-option -t '$1' status-style '{}'",
                mmux_attach_status_style(None)
            ),
            "",
        )
        .with_response(
            &format!("set-option -t '$1' status-left-length {MMUX_ATTACH_STATUS_LEFT_LENGTH}"),
            "",
        )
        .with_response(
            &format!("set-option -t '$1' status-left '{MMUX_ATTACH_STATUS_LEFT}'"),
            "",
        )
        .with_response("set-option -t '$1' status-style 'bg=green,fg=black'", "")
        .with_response("set-option -t '$1' status-left-length 32", "")
        .with_response(
            "set-option -t '$1' status-left 'session: #{session_name}'",
            "",
        );
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let target = host.session_by_id("$1").await.unwrap().unwrap();

    let status = target.status().await.unwrap();
    let snapshot = prepare_attach_status(&status, None).await.unwrap();
    assert_eq!(
        snapshot.style,
        Some(StatusStyle::new("bg=green,fg=black").unwrap())
    );
    assert_eq!(
        snapshot.left,
        Some(StatusLeft::new("session: #{session_name}").unwrap())
    );
    assert_eq!(
        snapshot.left_length,
        Some(StatusLeftLength::new(32).unwrap())
    );
    restore_attach_status(&status, Some(snapshot)).await;
}

#[tokio::test]
async fn attach_status_uses_selected_host_color() {
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response("show-option -q -t '$1' status-style", "")
        .with_response("show-option -q -t '$1' status-left-length", "")
        .with_response("show-option -q -t '$1' status-left", "")
        .with_response("set-option -t '$1' status-style 'bg=#ffc107,fg=black'", "")
        .with_response(
            &format!("set-option -t '$1' status-left-length {MMUX_ATTACH_STATUS_LEFT_LENGTH}"),
            "",
        )
        .with_response(
            &format!("set-option -t '$1' status-left '{MMUX_ATTACH_STATUS_LEFT}'"),
            "",
        );
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let target = host.session_by_id("$1").await.unwrap().unwrap();

    let status = target.status().await.unwrap();
    let snapshot = prepare_attach_status(&status, Some(HOST_COLOR_PALETTE[1]))
        .await
        .unwrap();
    assert_eq!(snapshot.style, None);
    assert_eq!(snapshot.left, None);
    assert_eq!(snapshot.left_length, None);
}

#[tokio::test]
async fn attach_status_overrides_unset_when_no_previous_local_values() {
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response("show-option -q -t '$1' status-style", "")
        .with_response("show-option -q -t '$1' status-left-length", "")
        .with_response("show-option -q -t '$1' status-left", "")
        .with_response(
            &format!(
                "set-option -t '$1' status-style '{}'",
                mmux_attach_status_style(None)
            ),
            "",
        )
        .with_response(
            &format!("set-option -t '$1' status-left-length {MMUX_ATTACH_STATUS_LEFT_LENGTH}"),
            "",
        )
        .with_response(
            &format!("set-option -t '$1' status-left '{MMUX_ATTACH_STATUS_LEFT}'"),
            "",
        )
        .with_response("set-option -u -t '$1' status-style", "")
        .with_response("set-option -u -t '$1' status-left-length", "")
        .with_response("set-option -u -t '$1' status-left", "");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let target = host.session_by_id("$1").await.unwrap().unwrap();

    let status = target.status().await.unwrap();
    let snapshot = prepare_attach_status(&status, None).await.unwrap();
    assert_eq!(snapshot.style, None);
    assert_eq!(snapshot.left, None);
    assert_eq!(snapshot.left_length, None);
    restore_attach_status(&status, Some(snapshot)).await;
}

#[tokio::test]
async fn attach_styles_apply_status_and_window_theme_then_restore() {
    let window_style = mmux_attach_window_style().expect("mmux attach window style is configured");
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 2  100\n")
        .with_response("show-option -q -t '$1' status-style", "")
        .with_response("show-option -q -t '$1' status-left-length", "")
        .with_response("show-option -q -t '$1' status-left", "")
        .with_response("list-windows -t '$1'", "@2\n@5\n")
        .with_response(
            "show-option -q -t '@2' window-style",
            "window-style bg=green,fg=black\n",
        )
        .with_response("show-option -q -t '@2' window-active-style", "")
        .with_response("show-option -q -t '@5' window-style", "")
        .with_response(
            "show-option -q -t '@5' window-active-style",
            "window-active-style bg=yellow,fg=black\n",
        )
        .with_response(
            &format!(
                "set-option -t '$1' status-style '{}'",
                mmux_attach_status_style(None)
            ),
            "",
        )
        .with_response(
            &format!("set-option -t '$1' status-left-length {MMUX_ATTACH_STATUS_LEFT_LENGTH}"),
            "",
        )
        .with_response(
            &format!("set-option -t '$1' status-left '{MMUX_ATTACH_STATUS_LEFT}'"),
            "",
        )
        .with_response("list-windows -t '$1'", "@2\n@5\n")
        .with_response(
            &format!("set-option -t '@2' window-style '{window_style}'"),
            "",
        )
        .with_response(
            &format!("set-option -t '@2' window-active-style '{window_style}'"),
            "",
        )
        .with_response(
            &format!("set-option -t '@5' window-style '{window_style}'"),
            "",
        )
        .with_response(
            &format!("set-option -t '@5' window-active-style '{window_style}'"),
            "",
        )
        .with_response("set-option -t '@2' window-style 'bg=green,fg=black'", "")
        .with_response("set-option -u -t '@2' window-active-style", "")
        .with_response("set-option -u -t '@5' window-style", "")
        .with_response(
            "set-option -t '@5' window-active-style 'bg=yellow,fg=black'",
            "",
        )
        .with_response("set-option -u -t '$1' status-style", "")
        .with_response("set-option -u -t '$1' status-left-length", "")
        .with_response("set-option -u -t '$1' status-left", "");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let target = host.session_by_id("$1").await.unwrap().unwrap();
    let status = target.status().await.unwrap();
    let window_styles = target.window_styles().await.unwrap();

    let snapshot = prepare_attach_styles(Some(&status), Some(&window_styles), None).await;
    assert!(snapshot.status.is_some());
    assert!(snapshot.window_styles.is_some());
    assert_eq!(
        mmux_attach_window_style(),
        Some(WindowStyle::new(window_style).unwrap().as_str().to_string())
    );

    restore_attach_styles(Some(&status), Some(&window_styles), snapshot).await;
}

#[tokio::test]
async fn p_opens_prompt_without_grouping_sessions() {
    let fleet = local_fleet();
    let mut app = app_with_session();
    app.layout.focus = Focus::List;

    let outcome = handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(matches!(outcome, KeyOutcome::Continue));
    assert_eq!(app.session_list.sort_mode, SessionSortMode::Activity);
    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::SendKeys { session, ui })
            if session.name() == "dev" && ui.focus == SendKeysFocus::Input
    ));
}

#[tokio::test]
async fn h_opens_help_modal_and_enter_or_escape_closes_it() {
    let fleet = local_fleet();
    let mut app = app_with_session();

    let outcome = handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('h'), KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(matches!(outcome, KeyOutcome::Continue));
    assert!(matches!(app.modal.as_ref(), Some(ModalState::Help)));
    let view = modal_content(app.modal.as_ref().unwrap());
    assert_eq!(view.title, " Help ");
    assert_eq!(view.active_button, Some(Button::Ok));
    assert_eq!(view.buttons, "[Ok]");
    let body = view.body_text();
    assert!(body.contains(MOTLIE_PLACEHOLDER));
    assert!(body.contains(HELP_KEY_FUNCTIONS));
    assert!(body.contains("↑ (u) / ↓ (b) select session or scroll detail"));
    assert!(body.contains("Enter sample highlighted session (list pane)"));
    assert!(body.contains("  $0..$9 send digit to highlight"));
    assert!(body.contains("  $! send Escape to highlight"));
    assert!(body.contains("  Ctrl-Enter send keys, wait, Enter"));
    assert!(body.contains("  $$ suffix same delayed Enter"));
    assert!(body.contains("  ↑ (u) / ↓ (b) move env row"));
    assert!(body.contains("  m modify env row"));
    assert!(body.contains("  x unset env row"));
    assert!(body.contains("  ↑ (u) / ↓ (b) move focused tag"));
    assert!(body.contains("  m modify focused tag"));
    assert!(body.contains("  x unset focused tag"));
    assert!(body.contains("  c toggle sort tag"));
    assert!(!body.contains("PgUp/PgDn page current pane"));
    assert!(!body.contains("Home/End jump current pane"));
    assert!(body.contains(BUILD_DATE));
    assert!(body.contains(&format!("Git SHA: {}", short_build_git_sha())));
    if BUILD_GIT_SHA.chars().count() > 8 {
        assert!(!body.contains(&format!("Git SHA: {BUILD_GIT_SHA}")));
    }

    let logo_pos = body.find(MOTLIE_PLACEHOLDER).unwrap();
    let build_date_pos = body.find("Build date: ").unwrap();
    let git_sha_pos = body.find("Git SHA: ").unwrap();
    let keys_pos = body.find(HELP_KEY_FUNCTIONS).unwrap();
    assert!(logo_pos < build_date_pos);
    assert!(build_date_pos < git_sha_pos);
    assert!(git_sha_pos < keys_pos);

    let outcome = handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert!(matches!(outcome, KeyOutcome::Continue));
    assert!(app.modal.is_none());

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('h'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    let outcome = handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert!(matches!(outcome, KeyOutcome::Continue));
    assert!(app.modal.is_none());
}

#[tokio::test]
async fn kill_modal_tab_cycles_cancel_and_ok() {
    let fleet = local_fleet();
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('k'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::KillSession {
            button: Button::Cancel,
            ..
        })
    ));

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::KillSession {
            button: Button::Ok,
            ..
        })
    ));

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::KillSession {
            button: Button::Cancel,
            ..
        })
    ));

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT),
    )
    .await
    .unwrap();
    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::KillSession {
            button: Button::Ok,
            ..
        })
    ));
}

#[test]
fn modal_content_separates_body_from_button_bar() {
    let new_session = modal_content(&test_new_session_modal("dev", Button::Ok));
    assert_eq!(new_session.title, " New Session ");
    assert_eq!(new_session.active_button, Some(Button::Ok));
    assert_eq!(new_session.buttons, " Cancel    [Ok]");
    assert!(matches!(
        new_session.body,
        ModalBody::NewSession { ref input, host_label: None, .. } if input == "dev"
    ));
    assert!(!new_session.body_text().contains("[Ok]"));

    let kill = modal_content(&ModalState::KillSession {
        session: test_selected_session(),
        button: Button::Cancel,
    });
    assert_eq!(kill.title, " Kill Session ");
    assert_eq!(kill.active_button, Some(Button::Cancel));
    assert_eq!(kill.body_text(), "Kill session dev?");
    assert_eq!(kill.buttons, "[Cancel]    Ok ");

    let rename = modal_content(&ModalState::RenameSession {
        session: test_selected_session(),
        input: "dev".to_string(),
        button: Button::Ok,
    });
    assert_eq!(rename.title, " Rename Session ");
    assert_eq!(rename.active_button, Some(Button::Ok));
    assert_eq!(rename.buttons, " Cancel    [Ok]");
    assert!(matches!(
        rename.body,
        ModalBody::RenameSession { ref input } if input == "dev"
    ));
    assert!(!rename.body_text().contains("[Ok]"));

    let send_keys = modal_content(&test_send_keys_modal("echo hi", SendKeysFocus::Input));
    assert_eq!(send_keys.title, " Send Keys ");
    assert_eq!(send_keys.active_button, None);
    assert_eq!(send_keys.buttons, " Cancel     Ok ");
    assert!(matches!(
        send_keys.body,
        ModalBody::SendKeys { ref label, ref input, focused }
            if label == "To: dev on host" && input == "echo hi" && focused
    ));
    assert!(!send_keys.body_text().contains("[Ok]"));

    let send_keys_ok = modal_content(&test_send_keys_modal("echo hi", SendKeysFocus::Ok));
    assert_eq!(send_keys_ok.active_button, Some(Button::Ok));
    assert_eq!(send_keys_ok.buttons, " Cancel    [Ok]");

    let mut short_app = app_with_session();
    short_app.modal = Some(test_send_keys_modal("short", SendKeysFocus::Input));
    let short_lines = render_to_lines(&mut short_app, 100, 30);
    let title_line = short_lines
        .iter()
        .position(|line| line.contains("Send Keys") && line.contains('┌'))
        .expect("expected Send Keys modal title");
    let label_line = short_lines
        .iter()
        .position(|line| line.contains("To: dev on host"))
        .expect("expected Send Keys target label");
    let input_bottom_line = short_lines
        .iter()
        .enumerate()
        .skip(label_line + 1)
        .find_map(|(index, line)| (line.contains('└') && line.contains('┘')).then_some(index))
        .expect("expected Send Keys input bottom border");
    let separator_line = short_lines
        .iter()
        .enumerate()
        .skip(input_bottom_line + 1)
        .find_map(|(index, line)| {
            let is_separator = line.contains('─')
                && !line.contains('┌')
                && !line.contains('┐')
                && !line.contains('└');
            is_separator.then_some(index)
        })
        .expect("expected Send Keys separator");

    assert_eq!(label_line, title_line + 2);
    assert_eq!(separator_line, input_bottom_line + 2);

    let mut long_app = app_with_session();
    long_app.modal = Some(test_send_keys_modal(
        &"0123456789".repeat(12),
        SendKeysFocus::Input,
    ));
    let long_lines = render_to_lines(&mut long_app, 100, 30);

    assert_eq!(
        modal_border_width(&short_lines, "Send Keys"),
        modal_border_width(&long_lines, "Send Keys")
    );
    assert!(
        modal_border_height(&long_lines, "Send Keys")
            > modal_border_height(&short_lines, "Send Keys")
    );

    let tags = modal_content(&test_session_tags_modal(
        vec![SessionKeyValueRow {
            key: "owner".to_string(),
            value: "platform".to_string(),
        }],
        Some("owner"),
        "phase",
        "build",
        SessionKeyValueFocus::Value,
    ));
    assert_eq!(tags.title, " Session Tags ");
    assert_eq!(tags.active_button, None);
    assert_eq!(tags.buttons, " Cancel     Ok ");
    assert!(matches!(
        tags.body,
        ModalBody::SessionKeyValues { ref key_input, ref value_input, .. }
            if key_input == "phase" && value_input == "build"
    ));
    assert!(tags.body_text().contains("owner    platform ✓"));

    let tags_ok = modal_content(&test_session_tags_modal(
        vec![SessionKeyValueRow {
            key: "owner".to_string(),
            value: "platform".to_string(),
        }],
        Some("owner"),
        "",
        "",
        SessionKeyValueFocus::Ok,
    ));
    assert_eq!(tags_ok.active_button, Some(Button::Ok));
    assert_eq!(tags_ok.buttons, " Cancel    [Ok]");
}

#[test]
fn focused_modal_input_fields_place_terminal_cursor() {
    let mut rename_app = app_with_session();
    rename_app.modal = Some(ModalState::RenameSession {
        session: test_selected_session(),
        input: "dev".to_string(),
        button: Button::Ok,
    });
    let (rename_lines, rename_cursor) = render_to_lines_and_cursor(&mut rename_app, 80, 24);
    let rename_label_y = rename_lines
        .iter()
        .enumerate()
        .find_map(|(index, line)| line.contains("Session Name").then_some(index))
        .expect("expected rename label");
    let rename_y = rename_label_y + 2;
    let rename_line = &rename_lines[rename_y];
    assert_eq!(
        rename_cursor,
        Position::new(
            (line_char_index(rename_line, "dev") + "dev".len()) as u16,
            rename_y as u16
        )
    );

    let mut send_app = app_with_session();
    send_app.modal = Some(test_send_keys_modal("abc", SendKeysFocus::Input));
    let (send_lines, send_cursor) = render_to_lines_and_cursor(&mut send_app, 100, 30);
    let send_label_y = send_lines
        .iter()
        .enumerate()
        .find_map(|(index, line)| line.contains("To: dev on host").then_some(index))
        .expect("expected send keys label");
    let send_y = send_label_y + 2;
    let send_line = &send_lines[send_y];
    assert_eq!(
        send_cursor,
        Position::new(
            (line_char_index(send_line, "abc") + "abc".len()) as u16,
            send_y as u16
        )
    );

    let mut tags_app = app_with_session();
    tags_app.modal = Some(test_session_tags_modal(
        Vec::new(),
        None,
        "owner",
        "",
        SessionKeyValueFocus::Key,
    ));
    let (tags_lines, tags_cursor) = render_to_lines_and_cursor(&mut tags_app, 80, 24);
    let (tags_y, tags_line) = tags_lines
        .iter()
        .enumerate()
        .find(|(_, line)| line.contains("owner"))
        .expect("expected tag key input");
    assert_eq!(
        tags_cursor,
        Position::new(
            (line_char_index(tags_line, "owner") + "owner".len()) as u16,
            tags_y as u16
        )
    );

    let mut long_rename_app = app_with_session();
    long_rename_app.modal = Some(ModalState::RenameSession {
        session: test_selected_session(),
        input: "prefix-prefix-prefix-prefix-very-long-session-name-that-does-not-fit".to_string(),
        button: Button::Ok,
    });
    let (long_rename_lines, long_rename_cursor) =
        render_to_lines_and_cursor(&mut long_rename_app, 48, 18);
    let long_label_y = long_rename_lines
        .iter()
        .enumerate()
        .find_map(|(index, line)| line.contains("Session Name").then_some(index))
        .expect("expected long rename label");
    let long_input_y = long_label_y + 2;
    let long_input_line = &long_rename_lines[long_input_y];
    assert!(long_input_line.contains("me-that-does-not-fit"));
    assert!(!long_input_line.contains("very-long-session"));
    assert_eq!(long_rename_cursor.y, long_input_y as u16);
    assert!(long_rename_cursor.x > line_char_index(long_input_line, "me-that-does-not-fit") as u16);
}

#[tokio::test]
async fn new_session_modal_selects_host_in_multi_host_mode() {
    let fleet = HostFleet::from_entries(vec![
        ssh_host_entry("ssh://a", "alpha", "x", HostHandle::local()),
        ssh_host_entry("ssh://b", "beta", "y", HostHandle::local()),
    ]);
    let mut app = AppState::new(LayoutMode::Normal);
    app.session_list.rows = vec![make_row_for_host(
        session("dev", "$1"),
        ssh_host_id("ssh://b"),
        "beta",
    )];

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE),
    )
    .await
    .unwrap();

    let modal = app.modal.as_ref().expect("new session modal opened");
    let view = modal_content(modal);
    assert!(matches!(
        view.body,
        ModalBody::NewSession {
            ref host_label,
            focus: NewSessionFocus::Name,
            ..
        } if host_label.as_deref() == Some("beta")
    ));

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Up, KeyModifiers::NONE),
    )
    .await
    .unwrap();

    let modal = app.modal.as_ref().expect("new session modal remains open");
    let view = modal_content(modal);
    assert!(matches!(
        view.body,
        ModalBody::NewSession {
            ref host_label,
            focus: NewSessionFocus::Host,
            ..
        } if host_label.as_deref() == Some("alpha")
    ));
}

#[tokio::test]
async fn new_session_modal_creates_on_selected_multi_host() {
    let alpha = HostHandle::new(
        TransportKind::Mock(
            MockTransport::new()
                .with_error("new-session", "wrong host")
                .with_response("list-sessions", ""),
        ),
        None,
    );
    let beta = HostHandle::new(
        TransportKind::Mock(
            MockTransport::new()
                .with_response("new-session -d -s 'build'", "")
                .with_response("list-sessions", "__MOTLIE_S__ build $9 10 0 1  100\n")
                .with_response(
                    "display-message -p '__MOTLIE_TAGS__ $9'",
                    "__MOTLIE_TAGS__ $9\n",
                ),
        ),
        None,
    );
    let fleet = HostFleet::from_entries(vec![
        ssh_host_entry("ssh://a", "alpha", "x", alpha),
        ssh_host_entry("ssh://b", "beta", "y", beta),
    ]);
    let mut app = AppState::new(LayoutMode::Normal);
    app.session_list.rows = vec![make_row_for_host(
        session("selected", "$1"),
        ssh_host_id("ssh://b"),
        "beta",
    )];

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    for ch in "build".chars() {
        handle_key(
            &fleet,
            &mut app,
            KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE),
        )
        .await
        .unwrap();
    }
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert_eq!(app.status.text(), "created session build on beta");
    assert_eq!(
        app.session_list
            .rows
            .iter()
            .map(|row| (row.host_label.as_str(), row.session.name.as_str()))
            .collect::<Vec<_>>(),
        vec![("beta", "build")]
    );
}

#[tokio::test]
async fn kill_session_modal_clears_selected_multi_host_row() {
    let alpha = HostHandle::new(
        TransportKind::Mock(
            MockTransport::new()
                .with_response("list-sessions", "__MOTLIE_S__ shell $1 10 0 1  100\n")
                .with_error("kill-session", "wrong host"),
        ),
        None,
    );
    let beta = HostHandle::new(
        TransportKind::Mock(
            MockTransport::new()
                .with_response("kill-session -t '$7'", "")
                .with_response("list-sessions", "__MOTLIE_S__ build $7 10 0 1  100\n")
                .with_response(
                    "display-message -p '__MOTLIE_TAGS__ $7'",
                    "__MOTLIE_TAGS__ $7\n",
                ),
        ),
        None,
    );
    let fleet = HostFleet::from_entries(vec![
        ssh_host_entry("ssh://a", "alpha", "x", alpha),
        ssh_host_entry("ssh://b", "beta", "y", beta),
    ]);
    let mut app = AppState::new(LayoutMode::Normal);
    app.session_list.rows = vec![
        make_row_for_host(session("shell", "$1"), ssh_host_id("ssh://a"), "alpha"),
        make_row_for_host(session("build", "$7"), ssh_host_id("ssh://b"), "beta"),
    ];
    app.session_list.selected = 1;

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('k'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Right, KeyModifiers::NONE),
    )
    .await
    .unwrap();
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert_eq!(app.status.text(), "killed session build on beta");
    assert_eq!(
        app.session_list
            .rows
            .iter()
            .map(|row| (row.host_label.as_str(), row.session.name.as_str()))
            .collect::<Vec<_>>(),
        vec![("alpha", "shell")]
    );
}

#[test]
fn session_tags_modal_renders_list_and_distinct_input_row() {
    let mut app = app_with_session();
    app.modal = Some(test_session_tags_modal(
        vec![SessionKeyValueRow {
            key: "owner".to_string(),
            value: "platform".to_string(),
        }],
        Some("owner"),
        "phase",
        "build",
        SessionKeyValueFocus::Value,
    ));

    let screen_lines = render_to_lines(&mut app, 80, 24);
    let screen = screen_lines.join("");
    let footer = screen_lines
        .iter()
        .find(|line| line.contains("Cancel") && line.contains("modify"))
        .expect("expected session tags footer");

    assert_eq!(
        modal_border_width(&screen_lines, "Session Tags"),
        MODAL_MIN_WIDTH as usize
    );
    let cancel_start = footer
        .find("Cancel")
        .map(|byte_index| footer[..byte_index].chars().count());
    assert_eq!(
        cancel_start,
        Some(
            modal_border_left(&screen_lines, "Session Tags")
                + 1
                + MODAL_CONTENT_HORIZONTAL_PADDING as usize
                + 1
        )
    );
    assert!(screen.contains("owner"));
    assert!(screen.contains("platform"));
    assert!(screen.contains("phase"));
    assert!(screen.contains("build"));
    assert!(screen.contains("modify"));
    assert!(screen.contains("x unset"));
    assert!(screen.contains("check"));
    assert!(!screen.contains("+"));
    assert!(!screen.contains("┬"));
    assert!(!screen.contains("┼"));
    assert!(!screen.contains("┴"));
    assert!(!screen.contains("[✓]"));
    assert!(!screen.contains("[+]"));
    assert!(!screen.contains("[phase]"));
    assert!(!screen.contains("[build]"));
}

#[test]
fn session_key_values_empty_list_defaults_key_input_to_thirty_percent() {
    let rows = Vec::<SessionKeyValueRow>::new();

    assert_eq!(key_value_key_column_width(62, &rows, ""), 18);
    assert_eq!(key_value_key_column_width(62, &rows, "owner"), 18);
    assert_eq!(
        key_value_key_column_width(62, &rows, "abcdefghijklmnopqrstuvwxyz"),
        30
    );
}

#[test]
fn session_key_values_key_column_uses_longest_existing_key_when_rows_exist() {
    let tags = vec![SessionKeyValueRow {
        key: "owner".to_string(),
        value: "platform".to_string(),
    }];

    assert_eq!(key_value_key_column_width(62, &tags, ""), 9);
}

#[test]
fn session_tags_modal_list_caps_at_five_rows_and_scrolls() {
    let tags = (0..7)
        .map(|index| SessionKeyValueRow {
            key: format!("tag{index}"),
            value: format!("value{index}"),
        })
        .collect::<Vec<_>>();
    let mut app = app_with_session();
    app.modal = Some(test_session_tags_modal(
        tags.clone(),
        None,
        "new",
        "next",
        SessionKeyValueFocus::Row(0),
    ));

    let screen = render_to_string(&mut app, 80, 24);
    assert!(screen.contains("> tag0"));
    for index in 0..5 {
        assert!(screen.contains(&format!("tag{index}")));
    }
    assert!(!screen.contains("tag5"));
    assert!(!screen.contains("tag6"));

    if let Some(ModalState::SessionKeyValues { ui, .. }) = app.modal.as_mut() {
        ui.focus = SessionKeyValueFocus::Row(6);
    } else {
        panic!("expected session tags modal");
    }

    let screen = render_to_string(&mut app, 80, 24);
    assert!(screen.contains("> tag6"));
    assert!(!screen.contains("tag0"));
    assert!(!screen.contains("tag1"));
    for index in 2..7 {
        assert!(screen.contains(&format!("tag{index}")));
    }
}

#[tokio::test]
async fn r_opens_rename_only_from_session_list_focus() {
    let fleet = local_fleet();
    let mut app = app_with_session();
    app.layout.focus = Focus::Detail;

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('r'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert!(app.modal.is_none());

    app.layout.focus = Focus::List;
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('r'), KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::RenameSession {
            session,
            input,
            button,
            ..
        }) if session.id() == "$1" && session.name() == "dev" && input == "dev" && *button == Button::Ok
    ));
}

#[tokio::test]
async fn rename_modal_renames_changed_session_by_stable_id() {
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response("list-sessions", "__MOTLIE_S__ renamed $1 10 0 1  200\n")
        .with_response("rename-session -t '$1' 'renamed'", "")
        .with_response("capture-pane -ep", "renamed screen\n");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('r'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    if let Some(ModalState::RenameSession { input, .. }) = app.modal.as_mut() {
        *input = "renamed".to_string();
    } else {
        panic!("expected rename modal");
    }

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(app.modal.is_none());
    assert_eq!(
        app.selected_session()
            .map(|session| session.name().to_string()),
        Some("renamed".to_string())
    );
    assert_eq!(app.status.text(), "renamed session dev to renamed");
}

#[tokio::test]
async fn p_opens_send_keys_modal_for_selected_session() {
    let fleet = local_fleet();
    let mut app = app_with_session();
    app.layout.focus = Focus::Detail;

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::SendKeys { session, ui })
            if session.id() == "$1"
                && session.name() == "dev"
                && ui.input.is_empty()
                && ui.focus == SendKeysFocus::Input
    ));
}

#[tokio::test]
async fn at_opens_send_keys_modal_for_selected_session() {
    let fleet = local_fleet();
    let mut app = app_with_session();
    app.layout.focus = Focus::Detail;

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('@'), KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::SendKeys { session, ui })
            if session.id() == "$1"
                && session.name() == "dev"
                && ui.input.is_empty()
                && ui.focus == SendKeysFocus::Input
    ));
}

#[tokio::test]
async fn dollar_prefix_digit_sends_immediately_to_highlighted_session() {
    let mock = MockTransport::new()
        .with_error(
            "send-keys -t '$1' Enter",
            "digit shortcut should not send Enter",
        )
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response("send-keys -l -t '$1' 7", "")
        .with_response("capture-pane -ep", "updated screen\n");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('$'), KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert_eq!(
        app.pending_list_shortcut,
        Some(PendingListShortcut::SendKeysImmediate)
    );
    assert!(app.modal.is_none());

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('7'), KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert_eq!(app.pending_list_shortcut, None);
    assert!(app.modal.is_none());
    assert_eq!(app.status.text(), "sent keys to dev");
    assert_eq!(app.detail.lines, vec!["updated screen".to_string()]);
}

#[tokio::test]
async fn dollar_prefix_bang_sends_escape_immediately_to_highlighted_session() {
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response("send-keys -t '$1' Escape", "")
        .with_response("capture-pane -ep", "updated screen\n");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('$'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('!'), KeyModifiers::SHIFT),
    )
    .await
    .unwrap();

    assert_eq!(app.pending_list_shortcut, None);
    assert!(app.modal.is_none());
    assert_eq!(app.status.text(), "sent keys to dev");
    assert_eq!(app.detail.lines, vec!["updated screen".to_string()]);
}

#[tokio::test]
async fn dollar_prefix_invalid_shortcut_consumes_prefix_and_stays_in_main_view() {
    let fleet = local_fleet();
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('$'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert_eq!(app.pending_list_shortcut, None);
    assert!(app.modal.is_none());
    assert_eq!(app.status.text(), "invalid $ shortcut; use $0..$9 or $!");
}

#[tokio::test]
async fn send_keys_modal_tab_ok_sends_keys_and_closes() {
    let mock = MockTransport::new()
        .with_error("send-keys -t '$1' Enter", "should not send implicit Enter")
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response("send-keys -l -t '$1' 1", "")
        .with_response("capture-pane -ep", "updated screen\n");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    for ch in "1".chars() {
        handle_key(
            &fleet,
            &mut app,
            KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE),
        )
        .await
        .unwrap();
    }
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::SendKeys { ui, .. }) if ui.focus == SendKeysFocus::Ok
    ));

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(app.modal.is_none());
    assert_eq!(app.status.text(), "sent keys to dev");
    assert_eq!(app.detail.lines, vec!["updated screen".to_string()]);
}

#[tokio::test]
async fn send_keys_modal_enter_from_input_sends_keys_and_closes() {
    let mock = MockTransport::new()
        .with_error("send-keys -t '$1' Enter", "should not send implicit Enter")
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response("send-keys -l -t '$1' 1", "")
        .with_response("capture-pane -ep", "updated screen\n");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    for ch in "1".chars() {
        handle_key(
            &fleet,
            &mut app,
            KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE),
        )
        .await
        .unwrap();
    }
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(app.modal.is_none());
    assert_eq!(app.status.text(), "sent keys to dev");
    assert_eq!(app.detail.lines, vec!["updated screen".to_string()]);
}

#[tokio::test]
async fn send_keys_modal_does_not_force_refresh_in_monitor_mode() {
    let mock = MockTransport::new()
        .with_error("capture-pane -ep", "snapshot capture should not run")
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response("send-keys -l -t '$1' 1", "");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = app_with_session();
    app.detail.source = DetailSource::Monitor(Box::new(MonitorDetailSource {
        session_id: Some("$1".to_string()),
        host_id: Some(local_host_id()),
    }));
    app.set_detail_text("monitor screen".to_string());

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('1'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(app.modal.is_none());
    assert_eq!(app.status.text(), "sent keys to dev");
    assert_eq!(app.detail.source.mode(), DetailMode::Monitor);
    assert_eq!(app.detail.lines, vec!["monitor screen".to_string()]);
}

#[tokio::test]
async fn send_keys_modal_ctrl_enter_sends_keys_then_enter_and_closes() {
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response("send-keys -l -t '$1' 1", "")
        .with_response("send-keys -t '$1' Enter", "")
        .with_response("capture-pane -ep", "updated screen\n");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    for ch in "1".chars() {
        handle_key(
            &fleet,
            &mut app,
            KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE),
        )
        .await
        .unwrap();
    }
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Enter, KeyModifiers::CONTROL),
    )
    .await
    .unwrap();

    assert!(app.modal.is_none());
    assert_eq!(app.status.text(), "sent keys to dev");
}

#[tokio::test]
async fn send_keys_modal_suffix_shorthand_sends_keys_then_enter_and_closes() {
    let mock = MockTransport::new()
        .with_error("send-keys -l -t '$1' '1$$'", "should strip $$ suffix")
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response("send-keys -l -t '$1' 1", "")
        .with_response("send-keys -t '$1' Enter", "")
        .with_response("capture-pane -ep", "updated screen\n");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    for ch in "1$$".chars() {
        handle_key(
            &fleet,
            &mut app,
            KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE),
        )
        .await
        .unwrap();
    }
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(app.modal.is_none());
    assert_eq!(app.status.text(), "sent keys to dev");
}

#[tokio::test]
async fn send_keys_modal_suffix_only_sends_delayed_enter_and_closes() {
    let mock = MockTransport::new()
        .with_error("send-keys -l -t '$1'", "$$ should not send literal text")
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response("send-keys -t '$1' Enter", "")
        .with_response("capture-pane -ep", "updated screen\n");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    for ch in "$$".chars() {
        handle_key(
            &fleet,
            &mut app,
            KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE),
        )
        .await
        .unwrap();
    }
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(app.modal.is_none());
    assert_eq!(app.status.text(), "sent keys to dev");
}

#[tokio::test]
async fn send_keys_modal_suffix_shorthand_only_applies_at_end() {
    let mock = MockTransport::new()
        .with_error("send-keys -t '$1' Enter", "should not send implicit Enter")
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response("1$$2", "")
        .with_response("capture-pane -ep", "updated screen\n");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    for ch in "1$$2".chars() {
        handle_key(
            &fleet,
            &mut app,
            KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE),
        )
        .await
        .unwrap();
    }
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(app.modal.is_none());
    assert_eq!(app.status.text(), "sent keys to dev");
}

#[tokio::test]
async fn send_keys_modal_sends_explicit_special_key_sequence() {
    let mock = MockTransport::new()
        .with_error("send-keys -t '$1' Enter", "should not send implicit Enter")
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response("send-keys -l -t '$1' 1", "")
        .with_response("send-keys -t '$1' Tab", "")
        .with_response("capture-pane -ep", "updated screen\n");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    for ch in "1{Tab}".chars() {
        handle_key(
            &fleet,
            &mut app,
            KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE),
        )
        .await
        .unwrap();
    }
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(app.modal.is_none());
    assert_eq!(app.status.text(), "sent keys to dev");
}

#[tokio::test]
async fn send_keys_modal_escape_cancels_without_sending() {
    let mock = MockTransport::new().with_error("send-keys", "should not send keys");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(app.modal.is_none());
}

#[tokio::test]
async fn send_keys_modal_invalid_sequence_stays_open() {
    let mock = MockTransport::new().with_error("send-keys", "should not send invalid keys");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    for ch in "echo {Enter".chars() {
        handle_key(
            &fleet,
            &mut app,
            KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE),
        )
        .await
        .unwrap();
    }
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
    )
    .await
    .unwrap();
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(matches!(app.modal, Some(ModalState::SendKeys { .. })));
    assert!(app.status.text().starts_with("invalid keys:"));
}

#[tokio::test]
async fn t_opens_session_tags_modal_and_i_is_unassigned() {
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response(
            "show-options -t '$1'",
            "@mmux/b alpha\n@mmux/__selected-key a\n@mmux/a beta\n@other/a skip\n",
        );
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('i'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert!(app.modal.is_none());

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('t'), KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::SessionKeyValues { ui, .. })
            if ui.kind == SessionKeyValueKind::Tags
                && ui.rows
                == vec![
                    SessionKeyValueRow { key: "a".to_string(), value: "beta".to_string() },
                    SessionKeyValueRow { key: "b".to_string(), value: "alpha".to_string() },
                ]
                && ui.selected_key.as_deref() == Some("a")
                && ui.focus == SessionKeyValueFocus::Row(0)
    ));
}

#[tokio::test]
async fn new_session_modal_applies_staged_initial_environment() {
    let mock = MockTransport::new()
        .with_response(
            "new-session -d -s 'build' -e 'BUILD_ID=42' -e 'MOTLIE=enabled'",
            "",
        )
        .with_response("list-sessions", "__MOTLIE_S__ build $9 10 0 1  100\n")
        .with_response(
            "display-message -p '__MOTLIE_TAGS__ $9'",
            "__MOTLIE_TAGS__ $9\n",
        );
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = AppState::new(LayoutMode::Normal);

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    for ch in "build".chars() {
        handle_key(
            &fleet,
            &mut app,
            KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE),
        )
        .await
        .unwrap();
    }

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
    )
    .await
    .unwrap();
    for ch in "MOTLIE".chars() {
        handle_key(
            &fleet,
            &mut app,
            KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE),
        )
        .await
        .unwrap();
    }
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
    )
    .await
    .unwrap();
    for ch in "enabled".chars() {
        handle_key(
            &fleet,
            &mut app,
            KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE),
        )
        .await
        .unwrap();
    }
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::NewSession { ui })
            if ui.env_rows == vec![SessionKeyValueRow { key: "MOTLIE".to_string(), value: "enabled".to_string() }]
                && ui.focus == NewSessionFocus::EnvRow(0)
    ));

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
    )
    .await
    .unwrap();
    for ch in "BUILD_ID".chars() {
        handle_key(
            &fleet,
            &mut app,
            KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE),
        )
        .await
        .unwrap();
    }
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
    )
    .await
    .unwrap();
    for ch in "42".chars() {
        handle_key(
            &fleet,
            &mut app,
            KeyEvent::new(KeyCode::Char(ch), KeyModifiers::NONE),
        )
        .await
        .unwrap();
    }
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::NewSession { ui })
            if ui.env_rows == vec![
                SessionKeyValueRow { key: "BUILD_ID".to_string(), value: "42".to_string() },
                SessionKeyValueRow { key: "MOTLIE".to_string(), value: "enabled".to_string() },
            ]
    ));

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Right, KeyModifiers::NONE),
    )
    .await
    .unwrap();
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert_eq!(app.status.text(), "created session build");
}

#[tokio::test]
async fn new_session_modal_u_and_b_move_env_rows_and_m_updates() {
    let fleet = local_fleet();
    let mut app = AppState::new(LayoutMode::Normal);
    app.modal = Some(test_new_session_modal("build", Button::Ok));
    let Some(ModalState::NewSession { ui }) = app.modal.as_mut() else {
        panic!("new session modal should be open");
    };
    ui.env_rows = vec![
        SessionKeyValueRow {
            key: "BUILD_ID".to_string(),
            value: "42".to_string(),
        },
        SessionKeyValueRow {
            key: "MOTLIE".to_string(),
            value: "enabled".to_string(),
        },
    ];
    ui.focus = NewSessionFocus::EnvRow(0);

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('b'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::NewSession { ui }) if ui.focus == NewSessionFocus::EnvRow(1)
    ));

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('b'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::NewSession { ui }) if ui.focus == NewSessionFocus::EnvRow(1)
    ));

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('u'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::NewSession { ui }) if ui.focus == NewSessionFocus::EnvRow(0)
    ));

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('m'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::NewSession { ui })
            if ui.focus == NewSessionFocus::EnvValue
                && ui.env_key_input == "BUILD_ID"
                && ui.env_value_input == "42"
    ));
}

#[tokio::test]
async fn session_tags_modal_tab_cycles_edit_row_fields_ok_and_cancel() {
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response("show-options -t '$1'", "");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('t'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::SessionKeyValues { ui, .. }) if ui.focus == SessionKeyValueFocus::Key
    ));

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('\t'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::SessionKeyValues { ui, .. }) if ui.focus == SessionKeyValueFocus::Value
    ));

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::SessionKeyValues { ui, .. }) if ui.focus == SessionKeyValueFocus::Ok
    ));

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::SessionKeyValues { ui, .. }) if ui.focus == SessionKeyValueFocus::Cancel
    ));

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT),
    )
    .await
    .unwrap();
    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::SessionKeyValues { ui, .. }) if ui.focus == SessionKeyValueFocus::Ok
    ));
}

#[tokio::test]
async fn session_tags_modal_ok_from_keyboard_dismisses() {
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response("show-options -t '$1'", "");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('t'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
    )
    .await
    .unwrap();
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::SessionKeyValues { ui, .. }) if ui.focus == SessionKeyValueFocus::Ok
    ));

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(app.modal.is_none());
    assert_eq!(app.status.text(), "no tag changes on dev");
}

#[tokio::test]
async fn session_tags_modal_c_stages_selected_row() {
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response("show-options -t '$1'", "@mmux/a beta\n@mmux/b alpha\n");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('t'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('c'), KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::SessionKeyValues { ui, .. }) if ui.selected_key.as_deref() == Some("a")
    ));
    let view = modal_content(app.modal.as_ref().unwrap());
    assert!(view.body_text().contains("a    beta ✓"));

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('c'), KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::SessionKeyValues { ui, .. }) if ui.selected_key.is_none()
    ));
}

#[tokio::test]
async fn session_tags_modal_delete_updates_list() {
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response(
            "show-options -t '$1'",
            "@mmux/__selected-key a\n@mmux/a one\n@mmux/b two\n",
        );
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('t'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('x'), KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::SessionKeyValues { ui, .. })
            if ui.rows == vec![SessionKeyValueRow { key: "b".to_string(), value: "two".to_string() }]
                && ui.selected_key.is_none()
                && ui.focus == SessionKeyValueFocus::Row(0)
    ));
    assert_eq!(app.status.text(), "staged delete tag a on dev");
}

#[tokio::test]
async fn session_tags_modal_update_uses_bottom_fields() {
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response("show-options -t '$1'", "@mmux/owner old\n");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('t'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('m'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::SessionKeyValues { ui, .. })
            if ui.key_input == "owner" && ui.value_input == "old" && ui.focus == SessionKeyValueFocus::Value
    ));

    if let Some(ModalState::SessionKeyValues { ui, .. }) = app.modal.as_mut() {
        ui.value_input = "new".to_string();
        ui.focus = SessionKeyValueFocus::Key;
    } else {
        panic!("expected session tags modal");
    }
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::SessionKeyValues { ui, .. })
            if ui.rows == vec![SessionKeyValueRow { key: "owner".to_string(), value: "new".to_string() }]
                && ui.focus == SessionKeyValueFocus::Row(0)
    ));
    assert_eq!(app.status.text(), "staged tag owner on dev");
}

#[tokio::test]
async fn session_tags_modal_u_and_b_move_rows_and_m_updates() {
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response(
            "show-options -t '$1'",
            "@mmux/a one\n@mmux/b two\n@mmux/c three\n",
        );
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('t'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('b'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::SessionKeyValues { ui, .. }) if ui.focus == SessionKeyValueFocus::Row(1)
    ));

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('u'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::SessionKeyValues { ui, .. }) if ui.focus == SessionKeyValueFocus::Row(0)
    ));

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('m'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert_eq!(app.detail.source.mode(), DetailMode::Sample);
    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::SessionKeyValues { ui, .. })
            if ui.key_input == "a" && ui.value_input == "one" && ui.focus == SessionKeyValueFocus::Value
    ));
}

#[tokio::test]
async fn session_tags_modal_cancel_discards_staged_changes() {
    let mock = MockTransport::new()
        .with_error("set-option -t '$1'", "cancel should not write tags")
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response("show-options -t '$1'", "");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('t'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    if let Some(ModalState::SessionKeyValues { ui, .. }) = app.modal.as_mut() {
        ui.key_input = "owner".to_string();
        ui.value_input = "platform".to_string();
        ui.focus = SessionKeyValueFocus::Key;
    } else {
        panic!("expected session tags modal");
    }
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::SessionKeyValues { ui, .. })
            if ui.rows == vec![SessionKeyValueRow { key: "owner".to_string(), value: "platform".to_string() }]
    ));

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(app.modal.is_none());
    assert_eq!(app.status.text(), "discarded tag changes on dev");
}

#[tokio::test]
async fn session_tags_modal_ok_applies_staged_changes() {
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response(
            "display-message -p '__MOTLIE_TAGS__ $1'",
            "__MOTLIE_TAGS__ $1\n@mmux/__selected-key added\n@mmux/added yes\n@mmux/owner new\n",
        )
        .with_response("show-options -t '$1'", "@mmux/keep yes\n@mmux/owner old\n");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('t'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    if let Some(ModalState::SessionKeyValues { ui, .. }) = app.modal.as_mut() {
        ui.rows = vec![
            SessionKeyValueRow {
                key: "added".to_string(),
                value: "yes".to_string(),
            },
            SessionKeyValueRow {
                key: "owner".to_string(),
                value: "new".to_string(),
            },
        ];
        ui.selected_key = Some("added".to_string());
        ui.focus = SessionKeyValueFocus::Ok;
    } else {
        panic!("expected session tags modal");
    }

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(app.modal.is_none());
    assert_eq!(app.status.text(), "applied tag changes on dev");
    assert_eq!(
        app.session_list.rows[0]
            .selected_tag
            .as_ref()
            .map(|tag| (tag.key.as_str(), tag.value.as_str())),
        Some(("added", "yes"))
    );
}

#[tokio::test]
async fn session_tags_modal_empty_value_does_not_dispatch() {
    let mock = MockTransport::new()
        .with_error("set-option -t '$1'", "should not set empty tag values")
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response("show-options -t '$1'", "");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('t'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    if let Some(ModalState::SessionKeyValues { ui, .. }) = app.modal.as_mut() {
        ui.key_input = "owner".to_string();
        ui.value_input.clear();
        ui.focus = SessionKeyValueFocus::Value;
    } else {
        panic!("expected session tags modal");
    }

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(matches!(
        app.modal,
        Some(ModalState::SessionKeyValues { .. })
    ));
    assert_eq!(app.status.text(), "tag value is empty");
}

#[tokio::test]
async fn session_tags_modal_rejects_reserved_selected_key() {
    let mock = MockTransport::new()
        .with_error(
            "set-option -t '$1' @mmux/__selected-key",
            "should not overwrite selected tag marker from edit row",
        )
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response("show-options -t '$1'", "");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let fleet = fleet_with(host);
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('t'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    if let Some(ModalState::SessionKeyValues { ui, .. }) = app.modal.as_mut() {
        ui.key_input = "__selected-key".to_string();
        ui.value_input = "owner".to_string();
        ui.focus = SessionKeyValueFocus::Key;
    } else {
        panic!("expected session tags modal");
    }

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(matches!(
        app.modal,
        Some(ModalState::SessionKeyValues { .. })
    ));
    assert_eq!(app.status.text(), "tag key is reserved");
}

#[tokio::test]
async fn tab_cycles_landscape_panes() {
    let fleet = local_fleet();
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert_eq!(app.layout.focus, Focus::Detail);

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert_eq!(app.layout.focus, Focus::List);
}

#[tokio::test]
async fn tab_cycles_portrait_panes() {
    let fleet = local_fleet();
    let mut app = app_with_session();
    app.layout.mode = LayoutMode::Portrait;

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert_eq!(app.layout.focus, Focus::Detail);

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert_eq!(app.layout.focus, Focus::List);
}

#[tokio::test]
async fn l_toggles_layout_and_preserves_focus() {
    let fleet = local_fleet();
    let mut app = app_with_session();
    app.layout.focus = Focus::Detail;

    let outcome = handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('l'), KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(matches!(outcome, KeyOutcome::Continue));
    assert_eq!(app.layout.mode, LayoutMode::Portrait);
    assert_eq!(app.layout.focus, Focus::Detail);
    assert_eq!(app.status.text(), "layout toggled");

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('l'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert_eq!(app.layout.mode, LayoutMode::Normal);
    assert_eq!(app.layout.focus, Focus::Detail);
}

#[tokio::test]
async fn plain_left_and_right_do_not_cycle_panes() {
    let fleet = local_fleet();
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Left, KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert_eq!(app.layout.focus, Focus::List);

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Right, KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert_eq!(app.layout.focus, Focus::List);
}

#[tokio::test]
async fn modified_arrow_keys_resize_layouts() {
    let fleet = local_fleet();
    let mut landscape = AppState::new(LayoutMode::Normal);
    let initial = landscape.layout.left_percent;

    handle_key(
        &fleet,
        &mut landscape,
        KeyEvent::new(KeyCode::Left, KeyModifiers::SHIFT),
    )
    .await
    .unwrap();
    assert!(landscape.layout.left_percent < initial);

    landscape.layout.left_percent = 5;
    handle_key(
        &fleet,
        &mut landscape,
        KeyEvent::new(KeyCode::Left, KeyModifiers::SHIFT),
    )
    .await
    .unwrap();
    assert_eq!(landscape.layout.left_percent, LANDSCAPE_MIN_LEFT_PERCENT);

    landscape.layout.left_percent = 95;
    handle_key(
        &fleet,
        &mut landscape,
        KeyEvent::new(KeyCode::Right, KeyModifiers::SHIFT),
    )
    .await
    .unwrap();
    assert_eq!(landscape.layout.left_percent, LANDSCAPE_MAX_LEFT_PERCENT);

    let mut portrait = AppState::new(LayoutMode::Portrait);
    portrait.layout.top_percent = 5;
    handle_key(
        &fleet,
        &mut portrait,
        KeyEvent::new(KeyCode::Up, KeyModifiers::SHIFT),
    )
    .await
    .unwrap();
    assert_eq!(portrait.layout.top_percent, PORTRAIT_MIN_TOP_PERCENT);

    portrait.layout.top_percent = 95;
    handle_key(
        &fleet,
        &mut portrait,
        KeyEvent::new(KeyCode::Down, KeyModifiers::SHIFT),
    )
    .await
    .unwrap();
    assert_eq!(portrait.layout.top_percent, PORTRAIT_MAX_TOP_PERCENT);
}

#[tokio::test]
async fn word_arrow_fallback_sequences_resize_normal_layout() {
    let fleet = local_fleet();
    let mut app = AppState::new(LayoutMode::Normal);
    let initial = app.layout.left_percent;

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('b'), KeyModifiers::ALT),
    )
    .await
    .unwrap();
    assert!(app.layout.left_percent < initial);

    let shrunk = app.layout.left_percent;
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('f'), KeyModifiers::ALT),
    )
    .await
    .unwrap();
    assert!(app.layout.left_percent > shrunk);
}

#[test]
fn detail_uses_ansi_vte_parser_for_screen_content() {
    let text = detail_text_for_render("\x1b[31mred\x1b[0m");
    assert_eq!(text.lines[0].spans[0].content.as_ref(), "red");
    assert!(!text.lines[0].spans[0].content.contains('\x1b'));
    assert!(text
        .lines
        .iter()
        .flat_map(|line| line.spans.iter())
        .all(|span| span.style.fg != Some(Color::Reset) && span.style.bg != Some(Color::Reset)));
}

#[test]
fn detail_title_marks_snapshot_or_monitor_mode_in_bold() {
    let snapshot = detail_title(DetailMode::Sample, "1-3/3");
    assert_eq!(
        snapshot
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>(),
        " Detail snapshot 1-3/3 "
    );
    assert!(snapshot.spans[1]
        .style
        .add_modifier
        .contains(Modifier::BOLD));

    let monitor = detail_title(DetailMode::Monitor, "2-4/9");
    assert_eq!(
        monitor
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>(),
        " Detail monitor 2-4/9 "
    );
    assert!(monitor.spans[1].style.add_modifier.contains(Modifier::BOLD));
}

#[tokio::test]
async fn sample_detail_preserves_ansi_color_for_detail_pane() {
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ dev $7 0 0 1  0\n")
        .with_response("capture-pane -ep", "\x1b[34mBLUE\x1b[0m\n");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let selected = make_selected(local_host_id(), "host", "$7", "dev");
    let mut source = SampleDetailSource;

    let rendered = source.render(&host, &selected).await.unwrap();

    assert!(rendered.contains("\x1b[34mBLUE\x1b[0m"));
}

#[tokio::test]
async fn monitor_detail_captures_rendered_screen_with_ansi() {
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ dash $7 0 0 1  0\n")
        .with_response("list-panes", "%0 dash 0 0 main bash 100 80 24 1\n")
        .with_response("capture-pane -ep", "\x1b[32mREADY\x1b[0m\n");
    let host = HostHandle::new(TransportKind::Mock(mock), None);
    let selected = make_selected(local_host_id(), "host", "$7", "dash");
    let mut source = MonitorDetailSource::new();

    source.activate(&host, &selected).await.unwrap();
    let rendered = source.render(&host, &selected).await.unwrap();

    assert!(rendered.contains("\x1b[32mREADY\x1b[0m"));
    assert!(!rendered.contains("%output"));
}

// -----------------------------------------------------------------------
// Multi-host tests (issue #235)
// -----------------------------------------------------------------------

fn ssh_host_id(uri: &str) -> HostId {
    HostId::from_ssh_uri(uri)
}

fn ssh_host_entry(uri: &str, label: &str, ip: &str, handle: HostHandle) -> HostEntry {
    HostEntry {
        id: ssh_host_id(uri),
        label: label.to_string(),
        alias: uri
            .strip_prefix("ssh://")
            .unwrap_or(uri)
            .split(['/', '?', ':'])
            .next()
            .unwrap_or(uri)
            .to_string(),
        ip_address: ip.to_string(),
        handle,
    }
}

#[test]
fn cli_accepts_multiple_ssh_uris() {
    let single = Cli::try_parse_from(["mmux", "ssh://a.example.com"]).unwrap();
    assert_eq!(single.ssh_uris, vec!["ssh://a.example.com".to_string()]);

    let multi = Cli::try_parse_from([
        "mmux",
        "ssh://a.example.com",
        "ssh://b.example.com",
        "ssh://c.example.com",
    ])
    .unwrap();
    assert_eq!(multi.ssh_uris.len(), 3);
    assert_eq!(multi.ssh_uris[1], "ssh://b.example.com".to_string());

    let none = Cli::try_parse_from(["mmux"]).unwrap();
    assert!(none.ssh_uris.is_empty());
}

#[tokio::test]
async fn connect_initial_fleet_rejects_duplicate_ssh_uris() {
    use crate::target_host::connect_initial_fleet;

    let cli =
        Cli::try_parse_from(["mmux", "ssh://dchung@localhost", "ssh://dchung@localhost"]).unwrap();

    let err = match connect_initial_fleet(&cli).await {
        Ok(_) => panic!("expected duplicate SSH URI to be rejected"),
        Err(err) => err,
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("duplicate SSH URI"),
        "expected duplicate-URI rejection, got: {msg}"
    );
}

#[tokio::test]
async fn connect_initial_fleet_includes_localhost_and_defers_ssh_connect() {
    use crate::target_host::connect_initial_fleet;

    let cli = Cli::try_parse_from(["mmux", "ssh://remote.example.com"]).unwrap();

    let initial_fleet = connect_initial_fleet(&cli).await.unwrap();
    let fleet = initial_fleet.fleet;
    let specs = initial_fleet.retry_specs;
    let remote_id = ssh_host_id("ssh://remote.example.com");

    assert!(fleet.is_multi());
    assert_eq!(fleet.len(), 2);
    assert_eq!(specs.len(), 1);
    assert_eq!(specs[0].id, remote_id);
    assert!(fleet.entry(&local_host_id()).is_some());
    assert!(
        fleet.entry(&remote_id).is_none(),
        "SSH host should not block startup by connecting during initial fleet setup"
    );
    assert_eq!(
        fleet.host_slot(&remote_id).map(|slot| slot.label.as_str()),
        Some("remote.example.com")
    );
}

#[test]
fn fleet_is_multi_only_with_two_or_more_entries() {
    let single = HostFleet::from_entries(vec![HostEntry {
        id: local_host_id(),
        label: "h".to_string(),
        alias: "h".to_string(),
        ip_address: "x".to_string(),
        handle: HostHandle::local(),
    }]);
    assert!(!single.is_multi());
    assert_eq!(single.len(), 1);

    let multi = HostFleet::from_entries(vec![
        HostEntry {
            id: ssh_host_id("ssh://a"),
            label: "alpha".to_string(),
            alias: "a".to_string(),
            ip_address: "x".to_string(),
            handle: HostHandle::local(),
        },
        HostEntry {
            id: ssh_host_id("ssh://b"),
            label: "beta".to_string(),
            alias: "b".to_string(),
            ip_address: "y".to_string(),
            handle: HostHandle::local(),
        },
    ]);
    assert!(multi.is_multi());
    assert_eq!(multi.len(), 2);

    // host_marker_width is 0 in single-host (column omitted) and the width of
    // the compact square marker in multi-host.
    assert_eq!(single.host_marker_width(), 0);
    assert_eq!(single.host_color(&local_host_id()), None);
    assert_eq!(multi.host_marker_width(), HOST_COLOR_SQUARE.chars().count());
    assert_eq!(
        multi.host_color(&ssh_host_id("ssh://a")),
        Some(HOST_COLOR_PALETTE[0])
    );
    assert_eq!(
        multi.host_color(&ssh_host_id("ssh://b")),
        Some(HOST_COLOR_PALETTE[1])
    );
}

#[test]
fn fleet_can_show_pending_ssh_host_without_connected_entry() {
    let local = HostEntry {
        id: local_host_id(),
        label: "local".to_string(),
        alias: "local".to_string(),
        ip_address: "127.0.0.1".to_string(),
        handle: HostHandle::local(),
    };
    let remote_id = ssh_host_id("ssh://remote");
    let fleet = HostFleet::from_configured_hosts(
        vec![local.clone()],
        vec![
            HostSlot::connected(&local),
            HostSlot::connecting(
                remote_id.clone(),
                "remote".to_string(),
                "remote".to_string(),
            ),
        ],
    );

    assert!(fleet.is_multi());
    assert_eq!(fleet.len(), 2);
    assert!(fleet.entry(&local_host_id()).is_some());
    assert!(fleet.entry(&remote_id).is_none());
    assert_eq!(
        fleet.host_color(&local_host_id()),
        Some(HOST_COLOR_PALETTE[0])
    );
    assert_eq!(fleet.host_color(&remote_id), Some(HOST_COLOR_PALETTE[1]));
}

#[test]
fn connected_ssh_host_replaces_failed_slot_and_keeps_color_index() {
    let local = HostEntry {
        id: local_host_id(),
        label: "local".to_string(),
        alias: "local".to_string(),
        ip_address: "127.0.0.1".to_string(),
        handle: HostHandle::local(),
    };
    let remote_id = ssh_host_id("ssh://remote");
    let mut fleet = HostFleet::from_configured_hosts(
        vec![local.clone()],
        vec![
            HostSlot::connected(&local),
            HostSlot::connecting(
                remote_id.clone(),
                "remote".to_string(),
                "remote".to_string(),
            ),
        ],
    );
    fleet.mark_host_failed(
        &remote_id,
        HostConnectFailure::connect("network unreachable".to_string()),
    );

    let remote = ssh_host_entry(
        "ssh://remote",
        "remote-tmux",
        "10.0.0.8",
        HostHandle::local(),
    );
    fleet.upsert_connected(remote);

    assert_eq!(
        fleet.host_slot(&remote_id).map(|slot| &slot.status),
        Some(&HostConnectionStatus::Connected)
    );
    assert_eq!(
        fleet.host_color(&remote_id),
        Some(HOST_COLOR_PALETTE[1]),
        "reconnected host keeps its configured legend/row color"
    );
    assert_eq!(fleet.entry(&remote_id).unwrap().label, "remote-tmux");
}

#[test]
fn repeated_host_failure_with_same_error_is_not_a_state_change() {
    let local = HostEntry {
        id: local_host_id(),
        label: "local".to_string(),
        alias: "local".to_string(),
        ip_address: "127.0.0.1".to_string(),
        handle: HostHandle::local(),
    };
    let remote_id = ssh_host_id("ssh://remote");
    let mut fleet = HostFleet::from_configured_hosts(
        vec![local.clone()],
        vec![
            HostSlot::connected(&local),
            HostSlot::connecting(
                remote_id.clone(),
                "remote".to_string(),
                "remote".to_string(),
            ),
        ],
    );

    assert!(fleet.mark_host_failed(
        &remote_id,
        HostConnectFailure::connect("connection refused".to_string())
    ));
    assert!(
        !fleet.mark_host_failed(
            &remote_id,
            HostConnectFailure::connect("connection refused".to_string())
        ),
        "unchanged retry failure should not force a redraw"
    );
    assert!(fleet.mark_host_failed(
        &remote_id,
        HostConnectFailure::connect("network unreachable".to_string())
    ));
}

#[test]
fn fleet_entry_lookup_by_host_id() {
    let fleet = HostFleet::from_entries(vec![
        HostEntry {
            id: ssh_host_id("ssh://a"),
            label: "alpha".to_string(),
            alias: "a".to_string(),
            ip_address: "x".to_string(),
            handle: HostHandle::local(),
        },
        HostEntry {
            id: ssh_host_id("ssh://b"),
            label: "beta".to_string(),
            alias: "b".to_string(),
            ip_address: "y".to_string(),
            handle: HostHandle::local(),
        },
    ]);

    assert!(fleet.entry(&ssh_host_id("ssh://a")).is_some());
    assert!(fleet.entry(&ssh_host_id("ssh://b")).is_some());
    assert!(fleet.entry(&ssh_host_id("ssh://nope")).is_none());
    assert_eq!(
        fleet.entry(&ssh_host_id("ssh://b")).unwrap().label,
        "beta".to_string()
    );
}

#[test]
fn multi_host_top_status_shows_host_color_legend() {
    let fleet = HostFleet::from_entries(vec![
        ssh_host_entry("ssh://a", "alpha", "10.0.0.1", HostHandle::local()),
        ssh_host_entry("ssh://b", "beta", "10.0.0.2", HostHandle::local()),
        ssh_host_entry("ssh://c", "gamma", "10.0.0.3", HostHandle::local()),
    ]);
    let line = top_status_line(&fleet, "12:34:56", 60);
    let rendered = line
        .spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>();
    assert!(rendered.starts_with("mmux ■ alpha ■ beta ■ gamma"));
    let square_colors = line
        .spans
        .iter()
        .filter(|span| span.content.as_ref() == HOST_COLOR_SQUARE)
        .map(|span| span.style.fg)
        .collect::<Vec<_>>();
    assert_eq!(
        square_colors,
        vec![
            Some(HOST_COLOR_PALETTE[0]),
            Some(HOST_COLOR_PALETTE[1]),
            Some(HOST_COLOR_PALETTE[2]),
        ]
    );
    assert!(!rendered.contains("multi-host mode"));
    assert!(!rendered.contains("10.0.0"));
    assert!(rendered.ends_with(" 12:34:56 "));
}

#[test]
fn failed_multi_host_top_status_highlights_host_in_red() {
    let local = HostEntry {
        id: local_host_id(),
        label: "local".to_string(),
        alias: "local".to_string(),
        ip_address: "127.0.0.1".to_string(),
        handle: HostHandle::local(),
    };
    let remote_id = ssh_host_id("ssh://remote");
    let mut fleet = HostFleet::from_configured_hosts(
        vec![local.clone()],
        vec![
            HostSlot::connected(&local),
            HostSlot::connecting(
                remote_id.clone(),
                "remote".to_string(),
                "remote".to_string(),
            ),
        ],
    );
    fleet.mark_host_failed(
        &remote_id,
        HostConnectFailure::connect("connection refused".to_string()),
    );

    let line = top_status_line(&fleet, "12:34:56", 60);
    let rendered = line
        .spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>();
    assert!(rendered.starts_with("mmux ■ local ■ remote"));
    let remote_label = line
        .spans
        .iter()
        .find(|span| span.content.as_ref() == " remote ")
        .expect("failed host label is rendered");
    assert_eq!(remote_label.style.fg, Some(HOST_CONNECTION_FAILED_FG));

    let square_colors = line
        .spans
        .iter()
        .filter(|span| span.content.as_ref() == HOST_COLOR_SQUARE)
        .map(|span| span.style.fg)
        .collect::<Vec<_>>();
    assert_eq!(
        square_colors,
        vec![Some(HOST_COLOR_PALETTE[0]), Some(HOST_CONNECTION_FAILED_FG)]
    );
}

#[test]
fn multi_host_landscape_uses_same_sessions_detail_layout() {
    let mut app = AppState::new(LayoutMode::Normal);
    let fleet = HostFleet::from_entries(vec![
        ssh_host_entry("ssh://a", "alpha", "x", HostHandle::local()),
        ssh_host_entry("ssh://b", "beta", "y", HostHandle::local()),
    ]);
    app.session_list.rows = vec![make_row(session("dev", "$1"))];

    let rendered = render_to_string_with_fleet(&fleet, &mut app, 120, 30);
    assert!(!rendered.contains("MOTD"));
    assert!(!rendered.contains(MOTLIE_PLACEHOLDER));
    assert!(rendered.contains("mmux ■ alpha ■ beta"));
    assert!(rendered.contains("Sessions [1]"));
}

#[test]
fn host_marker_width_aligns_multi_host_rows() {
    let fleet = HostFleet::from_entries(vec![
        HostEntry {
            id: ssh_host_id("ssh://a"),
            label: "alpha".to_string(),
            alias: "a".to_string(),
            ip_address: "x".to_string(),
            handle: HostHandle::local(),
        },
        HostEntry {
            id: ssh_host_id("ssh://b"),
            label: "supercalifragilistic".to_string(),
            alias: "b".to_string(),
            ip_address: "y".to_string(),
            handle: HostHandle::local(),
        },
        HostEntry {
            id: ssh_host_id("ssh://c"),
            label: "beta".to_string(),
            alias: "c".to_string(),
            ip_address: "z".to_string(),
            handle: HostHandle::local(),
        },
    ]);
    assert_eq!(fleet.host_marker_width(), HOST_COLOR_SQUARE.chars().count());

    let alpha_row = make_row_for_host(session("dev", "$1"), ssh_host_id("ssh://a"), "alpha");
    let supercali_row = make_row_for_host(
        session("dev", "$2"),
        ssh_host_id("ssh://b"),
        "supercalifragilistic",
    );
    let beta_row = make_row_for_host(session("dev", "$3"), ssh_host_id("ssh://c"), "beta");

    let width = fleet.host_marker_width();
    let alpha_marker = fleet
        .host_color(&alpha_row.host_id)
        .map(|_| HOST_COLOR_SQUARE);
    let supercali_marker = fleet
        .host_color(&supercali_row.host_id)
        .map(|_| HOST_COLOR_SQUARE);
    let beta_marker = fleet
        .host_color(&beta_row.host_id)
        .map(|_| HOST_COLOR_SQUARE);
    let alpha_line = session_list_line(&alpha_row, false, alpha_marker, width, 80);
    let supercali_line = session_list_line(&supercali_row, false, supercali_marker, width, 80);
    let beta_line = session_list_line(&beta_row, false, beta_marker, width, 80);

    // The session-name token "dev" appears at the same column index in every
    // row regardless of which host produced the row.
    let alpha_dev_col = alpha_line.find("dev").unwrap();
    let supercali_dev_col = supercali_line.find("dev").unwrap();
    let beta_dev_col = beta_line.find("dev").unwrap();
    assert_eq!(
        alpha_dev_col, supercali_dev_col,
        "alpha row's session-name column aligns with supercalifragilistic row"
    );
    assert_eq!(
        beta_dev_col, supercali_dev_col,
        "beta row's session-name column aligns with supercalifragilistic row"
    );
}

#[test]
fn host_colors_cycle_through_five_color_palette() {
    let entries = (0..6)
        .map(|index| HostEntry {
            id: ssh_host_id(&format!("ssh://host{index}")),
            label: format!("host{index}"),
            alias: format!("host{index}"),
            ip_address: "x".to_string(),
            handle: HostHandle::local(),
        })
        .collect::<Vec<_>>();
    let fleet = HostFleet::from_entries(entries);

    assert_eq!(fleet.host_marker_width(), HOST_COLOR_SQUARE.chars().count());
    assert_eq!(
        fleet.host_color(&ssh_host_id("ssh://host0")),
        Some(HOST_COLOR_PALETTE[0])
    );
    assert_eq!(
        fleet.host_color(&ssh_host_id("ssh://host4")),
        Some(HOST_COLOR_PALETTE[4])
    );
    assert_eq!(
        fleet.host_color(&ssh_host_id("ssh://host5")),
        Some(HOST_COLOR_PALETTE[0])
    );
}

#[test]
fn multi_host_session_row_inserts_host_color_marker_column() {
    let row = make_row_for_host(session("dev", "$1"), ssh_host_id("ssh://a"), "alpha");
    let line = session_list_line(
        &row,
        false,
        Some(HOST_COLOR_SQUARE),
        HOST_COLOR_SQUARE.chars().count(),
        60,
    );
    // Format expected:  " * ■ dev"  (leading marker, attached, host square,
    // session name). The full hostname is reserved for the top legend.
    assert!(line.contains(HOST_COLOR_SQUARE));
    assert!(!line.contains("alpha"));
    assert!(line.contains("dev"));
    let code_pos = line.find(HOST_COLOR_SQUARE).expect("host marker rendered");
    let dev_pos = line.find("dev").expect("session name rendered");
    assert!(
        code_pos < dev_pos,
        "host marker appears before session name: {line:?}"
    );
}

#[test]
fn single_host_row_omits_hostname_column() {
    let row = make_row(session("dev", "$1"));
    let line = session_list_line(&row, false, None, 0, 30);
    // No host-marker column when host_marker_width = 0 (single-host).
    assert!(line.contains("dev"));
    assert!(!line.contains(HOST_COLOR_SQUARE));
}

#[test]
fn multi_host_sort_merges_rows_by_activity_across_hosts() {
    let mut app = AppState::new(LayoutMode::Normal);
    let rows = vec![
        make_row_for_host_at(
            session_with_times("alpha-old", "$1", 10, 100),
            ssh_host_id("ssh://a"),
            "alpha",
            1_000,
        ),
        make_row_for_host_at(
            session_with_times("beta-fresh", "$2", 20, 950),
            ssh_host_id("ssh://b"),
            "beta",
            1_000,
        ),
        make_row_for_host_at(
            session_with_times("alpha-fresh", "$3", 30, 800),
            ssh_host_id("ssh://a"),
            "alpha",
            1_000,
        ),
        make_row_for_host_at(
            session_with_times("beta-old", "$4", 40, 600),
            ssh_host_id("ssh://b"),
            "beta",
            1_000,
        ),
    ];
    app.session_list.set_rows_sorted_by_activity(rows);

    let names = app
        .session_list
        .rows
        .iter()
        .map(|row| row.session.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        vec!["beta-fresh", "alpha-fresh", "beta-old", "alpha-old"]
    );

    // Selection identity uses (host_id, session_id) so a row that re-orders
    // still resolves back to the same selected session.
    app.session_list.selected = 0;
    let selected = app.selected_session().unwrap();
    assert_eq!(selected.host_id, ssh_host_id("ssh://b"));
    assert_eq!(selected.id(), "$2");
}

#[test]
fn selection_preserves_host_and_session_after_multi_host_reorder() {
    let mut app = AppState::new(LayoutMode::Normal);
    app.session_list.set_rows_sorted_by_activity(vec![
        make_row_for_host_at(
            session_with_times("aa", "$1", 10, 900),
            ssh_host_id("ssh://a"),
            "alpha",
            1_000,
        ),
        make_row_for_host_at(
            session_with_times("bb", "$2", 20, 100),
            ssh_host_id("ssh://b"),
            "beta",
            1_000,
        ),
    ]);
    // First row by activity is alpha/$1.
    app.session_list.selected = 0;
    let key = app
        .selected_session()
        .map(|s| (s.host_id.clone(), s.id().to_string()))
        .unwrap();
    assert_eq!(key.0, ssh_host_id("ssh://a"));

    // Reorder: now bb is most active. Selection should preserve alpha/$1.
    app.session_list.set_rows_sorted_by_activity(vec![
        make_row_for_host_at(
            session_with_times("bb", "$2", 20, 950),
            ssh_host_id("ssh://b"),
            "beta",
            1_000,
        ),
        make_row_for_host_at(
            session_with_times("aa", "$1", 10, 900),
            ssh_host_id("ssh://a"),
            "alpha",
            1_000,
        ),
    ]);
    app.preserve_selection(Some(key));
    let still = app.selected_session().unwrap();
    assert_eq!(still.host_id, ssh_host_id("ssh://a"));
    assert_eq!(still.id(), "$1");
    assert_eq!(app.session_list.selected, 1);
}

#[tokio::test]
async fn fleet_fan_out_isolates_per_host_failure() {
    use crate::controller::fetch_fleet_rows;
    use crate::model::ActivityTracker;

    // Healthy host with one session. The post-#{epoch}-removal command shape
    // starts with `list-sessions`, so the mock matches on that prefix.
    let healthy_mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ running $1 10 0 1  400\n");
    let healthy = HostHandle::new(TransportKind::Mock(healthy_mock), None);

    // Failing host: mock returns a non-empty-state error, so list_sessions
    // surfaces the failure into the fan-out's failures vec rather than aborting.
    let failing_mock =
        MockTransport::new().with_error("list-sessions", "transport: connection refused");
    let failing = HostHandle::new(TransportKind::Mock(failing_mock), None);

    let fleet = HostFleet::from_entries(vec![
        ssh_host_entry("ssh://up", "up-host", "x", healthy),
        ssh_host_entry("ssh://down", "down-host", "y", failing),
    ]);

    let mut tracker = ActivityTracker::default();
    let (rows, failures) = fetch_fleet_rows(&fleet, &mut tracker).await;
    assert_eq!(rows.len(), 1, "healthy host's row(s) still listed");
    assert_eq!(rows[0].host_label, "up-host");
    assert_eq!(rows[0].session.name, "running");
    assert_eq!(failures.len(), 1, "failing host surfaces one failure entry");
    assert!(failures[0].contains("down-host"));
    assert_eq!(
        tracker.len(),
        1,
        "tracker holds state only for sessions seen on healthy hosts"
    );
}

#[test]
fn activity_tracker_first_sight_seeds_with_reported_age() {
    // First-sight semantics: when we observe a session for the first time,
    // the tracker seeds `activity_observed_at_local` to reflect how stale the
    // activity timestamp is *right now*, not "treat as fresh".
    use crate::model::ActivityTracker;
    let mut tracker = ActivityTracker::default();

    // NTP-synced clocks: activity_ts is on the same scale as local_now.
    let local_now = 1_000_000;
    let activity_ts = local_now - 30;

    let observed_at_local = tracker.observe(&local_host_id(), "$1", activity_ts, local_now);

    // Recency = local_now - observed_at_local should equal 30s — exactly the
    // reported staleness — not 0 ("now") just because we just saw it.
    assert_eq!(
        local_now - observed_at_local,
        30,
        "first-sight observed_at_local seeds with the reported age"
    );
}

#[test]
fn activity_tracker_advances_only_when_activity_moves_forward() {
    // Stationary activity → observed_at_local stays put across polls.
    // When activity advances, tracker resets to the new observation.
    use crate::model::ActivityTracker;
    let mut tracker = ActivityTracker::default();
    let id = local_host_id();

    let first = tracker.observe(&id, "$1", 100, 1_000);
    let second = tracker.observe(&id, "$1", 100, 1_005);
    assert_eq!(
        first, second,
        "activity_ts unchanged → observed_at_local unchanged across polls"
    );

    let third = tracker.observe(&id, "$1", 200, 1_010);
    assert_eq!(
        third, 1_010,
        "activity_ts advanced → observed_at_local resets to current local_now"
    );
}

#[test]
fn activity_tracker_retain_drops_disappeared_sessions() {
    use crate::model::ActivityTracker;
    use std::collections::HashSet;
    let mut tracker = ActivityTracker::default();
    let id = local_host_id();

    tracker.observe(&id, "$1", 100, 1_000);
    tracker.observe(&id, "$2", 200, 1_000);
    assert_eq!(tracker.len(), 2);

    let mut keep = HashSet::new();
    keep.insert((id.clone(), "$2".to_string()));
    tracker.retain(&keep);

    assert_eq!(tracker.len(), 1, "$1 dropped from tracker after retain");
}

#[test]
fn multi_host_kill_modal_carries_host_id() {
    // The kill modal must capture host_id at modal-open so dispatch routes
    // back to the correct host even if the highlighted row reorders later.
    let modal = ModalState::KillSession {
        session: make_selected(ssh_host_id("ssh://b"), "beta", "$42", "build"),
        button: Button::Cancel,
    };
    let view = modal_content(&modal);
    assert_eq!(view.title, " Kill Session ");
    assert_eq!(view.body_text(), "Kill session build?");
    // Confirms the modal carries the host info even though the rendered
    // confirmation prompt uses the session name only.
    if let ModalState::KillSession { session, .. } = modal {
        assert_eq!(session.host_id, ssh_host_id("ssh://b"));
        assert_eq!(session.host_label, "beta");
    } else {
        unreachable!();
    }
}
