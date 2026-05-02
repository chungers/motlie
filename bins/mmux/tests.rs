use clap::{CommandFactory, Parser};
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use motlie_tmux::{
    transport::MockTransport, HostHandle, SessionId, SessionInfo, TransportKind, SSH_DEFAULT_PORT,
};
use ratatui::backend::TestBackend;
use ratatui::layout::Rect;
use ratatui::style::Modifier;
use ratatui::Terminal;

use crate::cli::{is_portrait_pty, select_layout, Cli};
use crate::consts::{
    BUILD_DATE, BUILD_GIT_SHA, COMPACT_MOTLIE_PLACEHOLDER, HELP_KEY_FUNCTIONS,
    LANDSCAPE_MAX_LEFT_PERCENT, LANDSCAPE_MIN_LEFT_PERCENT, MODAL_MIN_WIDTH, MOTLIE_PLACEHOLDER,
    PORTRAIT_MAX_TOP_PERCENT, PORTRAIT_MIN_TOP_PERCENT,
};
use crate::controller::{
    handle_key, load_motd_from, refresh_sessions_preserving, refresh_sessions_quiet,
    stop_monitor_if_closed, KeyOutcome,
};
use crate::detail::{
    DetailMode, DetailSource, MonitorDetailSource, SampleDetailSource, SessionDetailSource,
};
use crate::model::{
    AppState, Button, Focus, HostEntry, HostFleet, HostId, LayoutMode, ModalBody, ModalState,
    SelectedSession, SessionRow, SessionSelectedTag, SessionSortMode, SessionTagRow,
    SessionTagsFocus, SessionTagsModalUi,
};
use crate::render::{
    detail_text_for_render, draw, modal_content, motd_render_text, normal_motd_height,
    session_list_line, session_recency_text, sessions_title, short_build_git_sha, status_line,
    status_line_text, tag_key_column_width, top_status_line, use_compact_placeholder,
};
use crate::target_host::resolve_ip_address;

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
    let mut app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );
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
    let activity_observed_at_local = session.activity;
    SessionRow {
        host_id,
        host_label: host_label.to_string(),
        local_now: u64::MAX,
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
        ip_address: "unknown".to_string(),
        handle,
    }])
}

fn local_fleet() -> HostFleet {
    fleet_with(HostHandle::local())
}

fn make_selected(host_id: HostId, host_label: &str, id: &str, name: &str) -> SelectedSession {
    SelectedSession {
        host_id,
        host_label: host_label.to_string(),
        id: id.to_string(),
        name: name.to_string(),
    }
}

fn test_selected_session() -> SelectedSession {
    make_selected(local_host_id(), "host", "$1", "dev")
}

fn test_session_tags_modal(
    tags: Vec<SessionTagRow>,
    selected_key: Option<&str>,
    key_input: &str,
    value_input: &str,
    focus: SessionTagsFocus,
) -> ModalState {
    ModalState::SessionTags {
        session: test_selected_session(),
        ui: SessionTagsModalUi {
            tags,
            selected_key: selected_key.map(str::to_string),
            key_input: key_input.to_string(),
            value_input: value_input.to_string(),
            focus,
        },
    }
}

fn unique_test_path(name: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("mmux-{name}-{}-{nanos}", std::process::id()))
}

async fn write_test_file(name: &str, content: impl AsRef<[u8]>) -> std::path::PathBuf {
    let path = unique_test_path(name);
    tokio::fs::write(&path, content).await.unwrap();
    path
}

async fn remove_test_file(path: &std::path::Path) {
    let _ = tokio::fs::remove_file(path).await;
}

fn render_to_string(app: &mut AppState, width: u16, height: u16) -> String {
    render_to_lines(app, width, height).join("")
}

fn render_to_lines(app: &mut AppState, width: u16, height: u16) -> Vec<String> {
    let backend = TestBackend::new(width, height);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal.draw(|frame| draw(frame, app)).unwrap();
    terminal
        .backend()
        .buffer()
        .content()
        .chunks(width as usize)
        .map(|row| row.iter().map(|cell| cell.symbol()).collect())
        .collect()
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
fn portrait_default_split_is_30_70() {
    let app = AppState::new(
        "host".to_string(),
        LayoutMode::Portrait,
        "motd".to_string(),
        false,
    );

    assert_eq!(app.layout.top_percent, 30);
}

#[test]
fn status_line_omits_layout_mode() {
    let normal = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );
    let normal_status = status_line_text(&normal);
    assert!(normal_status.contains(" ↑/↓ sel"));
    assert!(!normal_status.contains("keys"));
    assert!(!normal_status.contains("host"));
    assert!(!normal_status.contains("(h)elp"));
    assert!(!normal_status.contains("(p)ane"));
    assert!(normal_status.contains("pane"));
    assert_status_order(
        &normal_status,
        &[
            "help",
            "pane",
            "monitor",
            "enter/attach",
            "new",
            "kill",
            "rename",
            "tags",
            "quit",
            "layout",
            "mod-←/→ resize",
        ],
    );
    assert!(!normal_status.contains("normal"));
    assert!(!normal_status.contains("landscape"));
    assert!(!normal_status.contains("portrait"));

    let portrait = AppState::new(
        "host".to_string(),
        LayoutMode::Portrait,
        "motd".to_string(),
        false,
    );
    let portrait_status = status_line_text(&portrait);
    assert!(portrait_status.contains(" ↑/↓ sel"));
    assert!(portrait_status.contains("pane"));
    assert_status_order(
        &portrait_status,
        &[
            "help",
            "pane",
            "monitor",
            "enter/attach",
            "new",
            "kill",
            "rename",
            "tags",
            "quit",
            "layout",
            "mod-↑/↓ resize",
        ],
    );
}

#[test]
fn status_line_underlines_command_mnemonics() {
    let app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );
    let line = status_line(&app);
    let underlined = line
        .spans
        .iter()
        .filter(|span| span.style.add_modifier.contains(Modifier::UNDERLINED))
        .map(|span| span.content.as_ref())
        .collect::<String>();

    assert_eq!(underlined, "hpmankrtsql");
    assert_eq!(status_line_text(&app), " ↑/↓ sel | help | pane | monitor | enter/attach | new | kill | rename | tags | sort | quit | layout | mod-←/→ resize ");
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
    let app = AppState::new_with_host_ip(
        "target-host".to_string(),
        "192.0.2.10".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );
    let line = top_status_line(&app, "12:34:56", 40);
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
    let mut app = AppState::new_with_host_ip(
        "target-host".to_string(),
        "192.0.2.10".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );
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
    let mut app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );

    refresh_sessions_quiet(&fleet, &mut app, false)
        .await
        .unwrap();

    assert!(matches!(
        app.session_list.rows.first().and_then(|row| row.selected_tag.as_ref()),
        Some(SessionSelectedTag { key, value }) if key == "owner" && value == "platform"
    ));
}

#[test]
fn session_list_sorts_most_recent_activity_first() {
    let mut app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );

    app.session_list.set_rows_sorted_by_activity(to_rows(vec![
        session_with_times("older", "$1", 10, 100),
        session_with_times("fresh", "$2", 20, 300),
        session_with_times("alpha", "$3", 30, 200),
        session_with_times("beta", "$4", 40, 200),
    ]));

    let names = app
        .session_list
        .rows
        .iter()
        .map(|row| row.session.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["fresh", "alpha", "beta", "older"]);
}

#[test]
fn session_list_tag_sort_groups_checked_tags_before_missing_tags() {
    let fleet = HostFleet::from_entries(vec![
        ssh_host_entry("ssh://a", "alpha", "x", HostHandle::local()),
        ssh_host_entry("ssh://b", "beta", "y", HostHandle::local()),
    ]);
    let mut app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );
    let rows = vec![
        make_row_for_host(
            session_with_times("no-tag-fresh", "$1", 10, 900),
            ssh_host_id("ssh://a"),
            "alpha",
        ),
        with_selected_tag(
            make_row_for_host(
                session_with_times("b-alpha-low", "$2", 10, 100),
                ssh_host_id("ssh://b"),
                "beta",
            ),
            "alpha",
        ),
        with_selected_tag(
            make_row_for_host(
                session_with_times("a-alpha-high", "$3", 10, 300),
                ssh_host_id("ssh://a"),
                "alpha",
            ),
            "alpha",
        ),
        with_selected_tag(
            make_row_for_host(
                session_with_times("b-same", "$4", 10, 400),
                ssh_host_id("ssh://b"),
                "beta",
            ),
            "same",
        ),
        with_selected_tag(
            make_row_for_host(
                session_with_times("z-same", "$5", 10, 400),
                ssh_host_id("ssh://a"),
                "alpha",
            ),
            "same",
        ),
        with_selected_tag(
            make_row_for_host(
                session_with_times("empty-selected-tag", "$7", 10, 800),
                ssh_host_id("ssh://a"),
                "alpha",
            ),
            "",
        ),
        make_row_for_host(
            session_with_times("no-tag-old", "$6", 10, 100),
            ssh_host_id("ssh://b"),
            "beta",
        ),
    ];

    app.session_list.set_rows_sorted_by_tag(rows, &fleet);

    let names = app
        .session_list
        .rows
        .iter()
        .map(|row| row.session.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        vec![
            "a-alpha-high",
            "b-alpha-low",
            "z-same",
            "b-same",
            "no-tag-fresh",
            "empty-selected-tag",
            "no-tag-old"
        ]
    );
}

#[test]
fn activity_sort_preserves_selection_by_stable_id() {
    let mut app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );
    app.session_list.rows = to_rows(vec![
        session_with_times("selected", "$1", 10, 100),
        session_with_times("other", "$2", 20, 200),
    ]);
    app.session_list.selected = 0;
    let previous = app
        .selected_session()
        .map(|session| (session.host_id, session.id));

    app.session_list.set_rows_sorted_by_activity(to_rows(vec![
        session_with_times("selected", "$1", 10, 100),
        session_with_times("other", "$2", 20, 300),
    ]));
    app.preserve_selection(previous);

    assert_eq!(
        app.selected_session().map(|session| session.name),
        Some("selected".to_string())
    );
}

#[tokio::test]
async fn s_toggles_tag_sort_from_list_focus_and_selects_top_row() {
    let fleet = local_fleet();
    let mut app = app_with_session();
    app.session_list.rows = vec![
        with_selected_tag(
            make_row(session_with_times("selected", "$1", 10, 300)),
            "zeta",
        ),
        with_selected_tag(
            make_row(session_with_times("other", "$2", 10, 200)),
            "alpha",
        ),
    ];
    app.session_list.selected = 0;
    app.layout.focus = Focus::Detail;

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('s'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert_eq!(app.session_list.sort_mode, SessionSortMode::Activity);
    assert_eq!(app.session_list.selected, 0);

    app.layout.focus = Focus::List;
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('s'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert_eq!(app.session_list.sort_mode, SessionSortMode::Tag);
    assert_eq!(app.status.text(), "sort: tag");
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
        app.selected_session().map(|session| session.name),
        Some("other".to_string())
    );

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('s'), KeyModifiers::NONE),
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
        vec!["selected", "other"]
    );
    assert_eq!(
        app.selected_session().map(|session| session.name),
        Some("selected".to_string())
    );
}

#[tokio::test]
async fn quiet_refresh_preserves_tag_sort_mode() {
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
    let mut app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );
    app.session_list.sort_mode = SessionSortMode::Tag;

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
    let mut app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );
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
    let mut app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );
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
        app.selected_session().map(|session| session.name),
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
    let mut app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );
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
    let mut app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );
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
    let mut app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );
    app.session_list.rows = vec![make_row(session("old", "$1"))];
    app.session_list.selected = 0;
    app.session_list.rows = vec![make_row(session("new", "$1"))];
    app.preserve_selection(Some((local_host_id(), "$1".to_string())));
    assert_eq!(app.session_list.selected, 0);
    assert_eq!(
        app.selected_session().map(|s| s.name),
        Some("new".to_string())
    );
}

#[test]
fn retained_ui_state_restores_selection_layout_and_split_on_reentry() {
    let mut app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );
    app.session_list.rows = to_rows(vec![session("shell", "$1"), session("build", "$2")]);
    app.session_list.selected = 1;
    app.layout.focus = Focus::Detail;
    app.layout.mode = LayoutMode::Portrait;
    app.layout.left_percent = 55;
    app.layout.top_percent = 45;

    let mut retained = crate::model::RetainedUiState::default();
    retained.update_from(&app);

    let mut reentered = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );
    retained.apply_to(&mut reentered);
    reentered.session_list.rows = to_rows(vec![session("build", "$2"), session("shell", "$1")]);
    reentered.preserve_selection(retained.selected_session_key());

    assert_eq!(
        reentered.selected_session().map(|session| session.name),
        Some("build".to_string())
    );
    assert_eq!(reentered.layout.mode, LayoutMode::Portrait);
    assert_eq!(reentered.layout.focus, Focus::Detail);
    assert_eq!(reentered.layout.left_percent, 55);
    assert_eq!(reentered.layout.top_percent, 45);
}

#[test]
fn retained_ui_state_falls_back_to_previous_index_when_session_disappears() {
    let mut app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );
    app.session_list.rows = to_rows(vec![
        session("one", "$1"),
        session("gone", "$2"),
        session("three", "$3"),
    ]);
    app.session_list.selected = 1;

    let mut retained = crate::model::RetainedUiState::default();
    retained.update_from(&app);

    let mut reentered = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );
    retained.apply_to(&mut reentered);
    reentered.session_list.rows = to_rows(vec![session("one", "$1"), session("three", "$3")]);
    reentered.preserve_selection(retained.selected_session_key());

    assert_eq!(reentered.session_list.selected, 1);
    assert_eq!(
        reentered.selected_session().map(|session| session.name),
        Some("three".to_string())
    );
}

#[test]
fn motd_placeholder_uses_compact_graphic_when_narrow() {
    let app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        MOTLIE_PLACEHOLDER.to_string(),
        true,
    );
    let text = motd_render_text(&app, Rect::new(0, 0, 40, 5));
    assert!(text.contains("motlie"));
    assert!(text.contains("(no /etc/motd)"));
}

#[test]
fn motd_placeholder_renders_full_logo_when_wide() {
    let app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        MOTLIE_PLACEHOLDER.to_string(),
        true,
    );

    let text = motd_render_text(&app, Rect::new(0, 0, 100, 20));

    assert!(text.contains(MOTLIE_PLACEHOLDER));
    assert!(text.ends_with("(no /etc/motd)"));
    assert!(!use_compact_placeholder(&app, 100, 20));
}

#[test]
fn compact_placeholder_boundary_uses_embedded_logo_width() {
    let mut app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        MOTLIE_PLACEHOLDER.to_string(),
        true,
    );

    assert!(use_compact_placeholder(&app, 40, 20));
    assert!(!use_compact_placeholder(&app, 41, 20));

    if let Some(motd) = app.motd.as_mut() {
        motd.is_placeholder = false;
    }
    assert!(!use_compact_placeholder(&app, 30, 5));
}

#[test]
fn landscape_layout_renders_motd_pane_with_placeholder() {
    let mut app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        MOTLIE_PLACEHOLDER.to_string(),
        true,
    );
    app.session_list.rows = vec![make_row(session("dev", "$1"))];

    let rendered = render_to_string(&mut app, 120, 30);

    assert!(rendered.contains("MOTD"));
    assert!(rendered.contains("_ __ ___"));
    assert!(!rendered.contains(COMPACT_MOTLIE_PLACEHOLDER));
    assert!(rendered.contains("(no /etc/motd)"));
    assert!(rendered.contains("Sessions [1]"));
}

#[test]
fn landscape_layout_renders_motd_pane_with_host_content() {
    let mut app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        "Welcome to dev-host".to_string(),
        false,
    );
    app.session_list.rows = vec![make_row(session("dev", "$1"))];

    let rendered = render_to_string(&mut app, 120, 30);

    assert!(rendered.contains("MOTD"));
    assert!(rendered.contains("Welcome to dev-host"));
    assert!(rendered.contains("Sessions [1]"));
}

#[test]
fn landscape_motd_height_preserves_visible_placeholder_before_sessions() {
    let app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        MOTLIE_PLACEHOLDER.to_string(),
        true,
    );

    assert_eq!(normal_motd_height(&app, Rect::new(0, 0, 40, 30)), 4);
    assert_eq!(normal_motd_height(&app, Rect::new(0, 0, 41, 30)), 8);
    assert_eq!(normal_motd_height(&app, Rect::new(0, 0, 40, 6)), 3);
}

#[test]
fn portrait_layout_does_not_render_motd_widget() {
    let mut app = AppState::new(
        "host".to_string(),
        LayoutMode::Portrait,
        MOTLIE_PLACEHOLDER.to_string(),
        true,
    );
    app.session_list.rows = vec![make_row(session("dev", "$1"))];
    let rendered = render_to_string(&mut app, 100, 30);

    assert!(!rendered.contains(MOTLIE_PLACEHOLDER));
    assert!(!rendered.contains(COMPACT_MOTLIE_PLACEHOLDER));
    assert!(!rendered.contains("MOTD"));
}

#[tokio::test]
async fn load_motd_falls_back_to_placeholder_when_missing() {
    let host = HostHandle::local();
    let missing = unique_test_path("missing-motd");

    let (text, is_placeholder) = load_motd_from(&host, &missing).await;

    assert!(is_placeholder);
    assert_eq!(text, MOTLIE_PLACEHOLDER);
}

#[tokio::test]
async fn load_motd_falls_back_to_placeholder_when_empty() {
    let host = HostHandle::local();
    let path = write_test_file("empty-motd", b"").await;

    let (text, is_placeholder) = load_motd_from(&host, &path).await;
    remove_test_file(&path).await;

    assert!(is_placeholder);
    assert_eq!(text, MOTLIE_PLACEHOLDER);
}

#[tokio::test]
async fn load_motd_falls_back_to_placeholder_when_whitespace_only() {
    let host = HostHandle::local();
    let path = write_test_file("whitespace-motd", b"  \n\t\n  ").await;

    let (text, is_placeholder) = load_motd_from(&host, &path).await;
    remove_test_file(&path).await;

    assert!(is_placeholder);
    assert_eq!(text, MOTLIE_PLACEHOLDER);
}

#[tokio::test]
async fn load_motd_falls_back_to_placeholder_when_oversized() {
    let host = HostHandle::local();
    let path = write_test_file("oversized-motd", vec![b'x'; 70 * 1024]).await;

    let (text, is_placeholder) = load_motd_from(&host, &path).await;
    remove_test_file(&path).await;

    assert!(is_placeholder);
    assert_eq!(text, MOTLIE_PLACEHOLDER);
}

#[tokio::test]
async fn load_motd_returns_content_when_readable() {
    let host = HostHandle::local();
    let path = write_test_file("normal-motd", b"Welcome to dev-host\n\n").await;

    let (text, is_placeholder) = load_motd_from(&host, &path).await;
    remove_test_file(&path).await;

    assert!(!is_placeholder);
    assert_eq!(text, "Welcome to dev-host");
}

#[tokio::test]
async fn closed_monitored_session_resets_detail_source() {
    let mut app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );
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
    let mut app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );
    app.detail.lines = (0..100).map(|n| format!("line {n}")).collect();
    app.detail.last_known_view_height = 10;

    app.scroll_detail(1);
    assert_eq!(app.detail.scroll, 1);
    assert!(!app.detail.auto_tail);

    app.scroll_detail(-1);
    assert_eq!(app.detail.scroll, 0);
    assert!(app.detail.auto_tail);
}

#[tokio::test]
async fn q_exits_like_ctrl_c() {
    let fleet = local_fleet();
    let mut app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );

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
async fn a_attaches_like_enter() {
    let fleet = local_fleet();
    let mut app = app_with_session();

    let outcome = handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(matches!(
        outcome,
        KeyOutcome::Select(SelectedSession { name, .. }) if name == "dev"
    ));
}

#[tokio::test]
async fn g_no_longer_attaches() {
    let fleet = local_fleet();
    let mut app = app_with_session();

    let outcome = handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('g'), KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(matches!(outcome, KeyOutcome::Continue));
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
    assert_eq!(view.active_button, Button::Ok);
    assert_eq!(view.buttons, "[Ok]");
    let body = view.body_text();
    assert!(body.contains(MOTLIE_PLACEHOLDER));
    assert!(body.contains(HELP_KEY_FUNCTIONS));
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

#[test]
fn modal_content_separates_body_from_button_bar() {
    let new_session = modal_content(&ModalState::NewSession {
        input: "dev".to_string(),
        button: Button::Ok,
    });
    assert_eq!(new_session.title, " New Session ");
    assert_eq!(new_session.active_button, Button::Ok);
    assert_eq!(new_session.buttons, " Cancel    [Ok]");
    assert!(matches!(
        new_session.body,
        ModalBody::NewSession { ref input } if input == "dev"
    ));
    assert!(!new_session.body_text().contains("[Ok]"));

    let kill = modal_content(&ModalState::KillSession {
        session: test_selected_session(),
        button: Button::Cancel,
    });
    assert_eq!(kill.title, " Kill Session ");
    assert_eq!(kill.active_button, Button::Cancel);
    assert_eq!(kill.body_text(), "Kill session dev?");
    assert_eq!(kill.buttons, "[Cancel]    Ok ");

    let rename = modal_content(&ModalState::RenameSession {
        session: test_selected_session(),
        input: "dev".to_string(),
        button: Button::Ok,
    });
    assert_eq!(rename.title, " Rename Session ");
    assert_eq!(rename.active_button, Button::Ok);
    assert_eq!(rename.buttons, " Cancel    [Ok]");
    assert!(matches!(
        rename.body,
        ModalBody::RenameSession { ref input } if input == "dev"
    ));
    assert!(!rename.body_text().contains("[Ok]"));

    let tags = modal_content(&test_session_tags_modal(
        vec![SessionTagRow {
            key: "owner".to_string(),
            value: "platform".to_string(),
        }],
        Some("owner"),
        "phase",
        "build",
        SessionTagsFocus::Value,
    ));
    assert_eq!(tags.title, " Session Tags ");
    assert_eq!(tags.active_button, Button::Ok);
    assert_eq!(tags.buttons, " Cancel ");
    assert!(matches!(
        tags.body,
        ModalBody::SessionTags { ref key_input, ref value_input, .. }
            if key_input == "phase" && value_input == "build"
    ));
    assert!(tags.body_text().contains("owner    platform ✓"));
}

#[test]
fn session_tags_modal_renders_list_and_distinct_input_row() {
    let mut app = app_with_session();
    app.modal = Some(test_session_tags_modal(
        vec![SessionTagRow {
            key: "owner".to_string(),
            value: "platform".to_string(),
        }],
        Some("owner"),
        "phase",
        "build",
        SessionTagsFocus::Value,
    ));

    let screen_lines = render_to_lines(&mut app, 80, 24);
    let screen = screen_lines.join("");

    assert_eq!(
        modal_border_width(&screen_lines, "Session Tags"),
        MODAL_MIN_WIDTH as usize
    );
    assert!(screen.contains("owner"));
    assert!(screen.contains("platform"));
    assert!(screen.contains("phase"));
    assert!(screen.contains("build"));
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
fn session_tags_empty_list_defaults_key_input_to_thirty_percent() {
    let tags = Vec::<SessionTagRow>::new();

    assert_eq!(tag_key_column_width(62, &tags, ""), 18);
    assert_eq!(tag_key_column_width(62, &tags, "owner"), 18);
    assert_eq!(
        tag_key_column_width(62, &tags, "abcdefghijklmnopqrstuvwxyz"),
        30
    );
}

#[test]
fn session_tags_key_column_uses_longest_existing_key_when_tags_exist() {
    let tags = vec![SessionTagRow {
        key: "owner".to_string(),
        value: "platform".to_string(),
    }];

    assert_eq!(tag_key_column_width(62, &tags, ""), 9);
}

#[test]
fn session_tags_modal_list_caps_at_five_rows_and_scrolls() {
    let tags = (0..7)
        .map(|index| SessionTagRow {
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
        SessionTagsFocus::TagRow(0),
    ));

    let screen = render_to_string(&mut app, 80, 24);
    assert!(screen.contains("> tag0"));
    for index in 0..5 {
        assert!(screen.contains(&format!("tag{index}")));
    }
    assert!(!screen.contains("tag5"));
    assert!(!screen.contains("tag6"));

    if let Some(ModalState::SessionTags { ui, .. }) = app.modal.as_mut() {
        ui.focus = SessionTagsFocus::TagRow(6);
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
        }) if session.id == "$1" && session.name == "dev" && input == "dev" && *button == Button::Ok
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
        app.selected_session().map(|session| session.name),
        Some("renamed".to_string())
    );
    assert_eq!(app.status.text(), "renamed session dev to renamed");
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
        Some(ModalState::SessionTags { ui, .. })
            if ui.tags
                == vec![
                    SessionTagRow { key: "a".to_string(), value: "beta".to_string() },
                    SessionTagRow { key: "b".to_string(), value: "alpha".to_string() },
                ]
                && ui.selected_key.as_deref() == Some("a")
                && ui.focus == SessionTagsFocus::TagRow(0)
    ));
}

#[tokio::test]
async fn session_tags_modal_tab_cycles_edit_row_fields_and_cancel() {
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
        Some(ModalState::SessionTags { ui, .. }) if ui.focus == SessionTagsFocus::Key
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
        Some(ModalState::SessionTags { ui, .. }) if ui.focus == SessionTagsFocus::Value
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
        Some(ModalState::SessionTags { ui, .. }) if ui.focus == SessionTagsFocus::Cancel
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
        Some(ModalState::SessionTags { ui, .. }) if ui.focus == SessionTagsFocus::Value
    ));
}

#[tokio::test]
async fn session_tags_modal_c_persists_selected_row() {
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response("show-options -t '$1'", "@mmux/a beta\n@mmux/b alpha\n")
        .with_response(
            "show-options -t '$1'",
            "@mmux/__selected-key a\n@mmux/a beta\n@mmux/b alpha\n",
        )
        .with_response(
            "show-options -t '$1'",
            "@mmux/__selected-key a\n@mmux/a beta\n@mmux/b alpha\n",
        )
        .with_response("show-options -t '$1'", "@mmux/a beta\n@mmux/b alpha\n")
        .with_response("set-option -t '$1' @mmux/__selected-key a", "")
        .with_response("set-option -u -t '$1' @mmux/__selected-key", "");
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
        Some(ModalState::SessionTags { ui, .. }) if ui.selected_key.as_deref() == Some("a")
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
        Some(ModalState::SessionTags { ui, .. }) if ui.selected_key.is_none()
    ));
}

#[tokio::test]
async fn session_tags_modal_delete_updates_list() {
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response(
            "show-options -t '$1'",
            "@mmux/__selected-key a\n@mmux/a one\n@mmux/b two\n",
        )
        .with_response("show-options -t '$1'", "@mmux/b two\n")
        .with_response("set-option -u -t '$1' @mmux/a", "")
        .with_response("set-option -u -t '$1' @mmux/__selected-key", "");
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
        Some(ModalState::SessionTags { ui, .. })
            if ui.tags == vec![SessionTagRow { key: "b".to_string(), value: "two".to_string() }]
                && ui.selected_key.is_none()
                && ui.focus == SessionTagsFocus::TagRow(0)
    ));
    assert_eq!(app.status.text(), "deleted tag a on dev");
}

#[tokio::test]
async fn session_tags_modal_update_uses_bottom_fields() {
    let mock = MockTransport::new()
        .with_response("list-sessions", "__MOTLIE_S__ dev $1 10 0 1  100\n")
        .with_response("show-options -t '$1'", "@mmux/owner old\n")
        .with_response("show-options -t '$1'", "@mmux/owner new\n")
        .with_response("set-option -t '$1' @mmux/owner 'new'", "");
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
        KeyEvent::new(KeyCode::Char('u'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert!(matches!(
        app.modal.as_ref(),
        Some(ModalState::SessionTags { ui, .. })
            if ui.key_input == "owner" && ui.value_input == "old" && ui.focus == SessionTagsFocus::Value
    ));

    if let Some(ModalState::SessionTags { ui, .. }) = app.modal.as_mut() {
        ui.value_input = "new".to_string();
        ui.focus = SessionTagsFocus::Key;
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
        Some(ModalState::SessionTags { ui, .. })
            if ui.tags == vec![SessionTagRow { key: "owner".to_string(), value: "new".to_string() }]
                && ui.focus == SessionTagsFocus::TagRow(0)
    ));
    assert_eq!(app.status.text(), "set tag owner on dev");
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
    if let Some(ModalState::SessionTags { ui, .. }) = app.modal.as_mut() {
        ui.key_input = "owner".to_string();
        ui.value_input.clear();
        ui.focus = SessionTagsFocus::Value;
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

    assert!(matches!(app.modal, Some(ModalState::SessionTags { .. })));
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
    if let Some(ModalState::SessionTags { ui, .. }) = app.modal.as_mut() {
        ui.key_input = "__selected-key".to_string();
        ui.value_input = "owner".to_string();
        ui.focus = SessionTagsFocus::Key;
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

    assert!(matches!(app.modal, Some(ModalState::SessionTags { .. })));
    assert_eq!(app.status.text(), "tag key is reserved");
}

#[tokio::test]
async fn p_cycles_landscape_panes() {
    let fleet = local_fleet();
    let mut app = app_with_session();

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert_eq!(app.layout.focus, Focus::Detail);

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert_eq!(app.layout.focus, Focus::Motd);

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert_eq!(app.layout.focus, Focus::List);
}

#[tokio::test]
async fn p_cycles_portrait_panes_without_motd() {
    let fleet = local_fleet();
    let mut app = app_with_session();
    app.layout.mode = LayoutMode::Portrait;

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert_eq!(app.layout.focus, Focus::Detail);

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('p'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert_eq!(app.layout.focus, Focus::List);
}

#[tokio::test]
async fn l_toggles_layout_and_normalizes_motd_focus_for_portrait() {
    let fleet = local_fleet();
    let mut app = app_with_session();
    app.layout.focus = Focus::Motd;

    let outcome = handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('l'), KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert!(matches!(outcome, KeyOutcome::Continue));
    assert_eq!(app.layout.mode, LayoutMode::Portrait);
    assert_eq!(app.layout.focus, Focus::List);
    assert_eq!(app.status.text(), "layout toggled");

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Char('l'), KeyModifiers::NONE),
    )
    .await
    .unwrap();
    assert_eq!(app.layout.mode, LayoutMode::Normal);
    assert_eq!(app.layout.focus, Focus::List);
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
async fn motd_focus_does_not_scroll_or_change_selection() {
    let fleet = local_fleet();
    let mut app = app_with_session();
    app.layout.focus = Focus::Motd;
    app.detail.lines = (0..20).map(|idx| format!("line {idx}")).collect();
    app.detail.last_known_view_height = 5;

    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::Down, KeyModifiers::NONE),
    )
    .await
    .unwrap();
    handle_key(
        &fleet,
        &mut app,
        KeyEvent::new(KeyCode::PageUp, KeyModifiers::NONE),
    )
    .await
    .unwrap();

    assert_eq!(app.session_list.selected, 0);
    assert_eq!(app.detail.scroll, 0);
}

#[tokio::test]
async fn modified_arrow_keys_resize_layouts() {
    let fleet = local_fleet();
    let mut landscape = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );
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

    let mut portrait = AppState::new(
        "host".to_string(),
        LayoutMode::Portrait,
        "motd".to_string(),
        false,
    );
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
    let mut app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );
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
async fn connect_fleet_rejects_duplicate_ssh_uris() {
    use crate::target_host::connect_fleet;

    let cli =
        Cli::try_parse_from(["mmux", "ssh://dchung@localhost", "ssh://dchung@localhost"]).unwrap();

    let err = match connect_fleet(&cli).await {
        Ok(_) => panic!("expected duplicate SSH URI to be rejected"),
        Err(err) => err,
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("duplicate SSH URI"),
        "expected duplicate-URI rejection, got: {msg}"
    );
}

#[test]
fn fleet_is_multi_only_with_two_or_more_entries() {
    let single = HostFleet::from_entries(vec![HostEntry {
        id: local_host_id(),
        label: "h".to_string(),
        ip_address: "x".to_string(),
        handle: HostHandle::local(),
    }]);
    assert!(!single.is_multi());
    assert_eq!(single.len(), 1);

    let multi = HostFleet::from_entries(vec![
        HostEntry {
            id: ssh_host_id("ssh://a"),
            label: "alpha".to_string(),
            ip_address: "x".to_string(),
            handle: HostHandle::local(),
        },
        HostEntry {
            id: ssh_host_id("ssh://b"),
            label: "beta".to_string(),
            ip_address: "y".to_string(),
            handle: HostHandle::local(),
        },
    ]);
    assert!(multi.is_multi());
    assert_eq!(multi.len(), 2);

    // host_code_width is 0 in single-host (column omitted) and the width of
    // the largest assigned compact code in multi-host.
    assert_eq!(single.host_code_width(), 0);
    assert_eq!(single.host_code(&local_host_id()), None);
    assert_eq!(multi.host_code_width(), "[A]".len());
    assert_eq!(
        multi.host_code(&ssh_host_id("ssh://a")).as_deref(),
        Some("[A]")
    );
    assert_eq!(
        multi.host_code(&ssh_host_id("ssh://b")).as_deref(),
        Some("[B]")
    );
}

#[test]
fn fleet_entry_lookup_by_host_id() {
    let fleet = HostFleet::from_entries(vec![
        HostEntry {
            id: ssh_host_id("ssh://a"),
            label: "alpha".to_string(),
            ip_address: "x".to_string(),
            handle: HostHandle::local(),
        },
        HostEntry {
            id: ssh_host_id("ssh://b"),
            label: "beta".to_string(),
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
fn multi_host_top_status_shows_host_code_legend() {
    let mut app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );
    app.fleet = HostFleet::from_entries(vec![
        ssh_host_entry("ssh://a", "alpha", "10.0.0.1", HostHandle::local()),
        ssh_host_entry("ssh://b", "beta", "10.0.0.2", HostHandle::local()),
        ssh_host_entry("ssh://c", "gamma", "10.0.0.3", HostHandle::local()),
    ]);
    let line = top_status_line(&app, "12:34:56", 60);
    let rendered = line
        .spans
        .iter()
        .map(|span| span.content.as_ref())
        .collect::<String>();
    assert!(rendered.starts_with("mmux [A] alpha [B] beta [C] gamma"));
    assert!(!rendered.contains("multi-host mode"));
    assert!(!rendered.contains("10.0.0"));
    assert!(rendered.ends_with(" 12:34:56 "));
}

#[test]
fn multi_host_motd_pane_is_hidden() {
    let mut app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        MOTLIE_PLACEHOLDER.to_string(),
        true,
    );
    app.motd = None; // multi-host mode discards MOTD
    app.fleet = HostFleet::from_entries(vec![
        ssh_host_entry("ssh://a", "alpha", "x", HostHandle::local()),
        ssh_host_entry("ssh://b", "beta", "y", HostHandle::local()),
    ]);
    app.session_list.rows = vec![make_row(session("dev", "$1"))];

    assert_eq!(
        normal_motd_height(&app, Rect::new(0, 0, 80, 20)),
        0,
        "MOTD region collapses to 0 in multi-host"
    );

    let rendered = render_to_string(&mut app, 120, 30);
    assert!(!rendered.contains("MOTD"));
    assert!(!rendered.contains(MOTLIE_PLACEHOLDER));
    assert!(rendered.contains("mmux [A] alpha [B] beta"));
    assert!(rendered.contains("Sessions [1]"));
}

#[test]
fn host_code_width_pads_to_largest_host_code() {
    let fleet = HostFleet::from_entries(vec![
        HostEntry {
            id: ssh_host_id("ssh://a"),
            label: "alpha".to_string(),
            ip_address: "x".to_string(),
            handle: HostHandle::local(),
        },
        HostEntry {
            id: ssh_host_id("ssh://b"),
            label: "supercalifragilistic".to_string(),
            ip_address: "y".to_string(),
            handle: HostHandle::local(),
        },
        HostEntry {
            id: ssh_host_id("ssh://c"),
            label: "beta".to_string(),
            ip_address: "z".to_string(),
            handle: HostHandle::local(),
        },
    ]);
    assert_eq!(fleet.host_code_width(), "[A]".len());

    let alpha_row = make_row_for_host(session("dev", "$1"), ssh_host_id("ssh://a"), "alpha");
    let supercali_row = make_row_for_host(
        session("dev", "$2"),
        ssh_host_id("ssh://b"),
        "supercalifragilistic",
    );
    let beta_row = make_row_for_host(session("dev", "$3"), ssh_host_id("ssh://c"), "beta");

    let width = fleet.host_code_width();
    let alpha_code = fleet.host_code(&alpha_row.host_id);
    let supercali_code = fleet.host_code(&supercali_row.host_id);
    let beta_code = fleet.host_code(&beta_row.host_id);
    let alpha_line = session_list_line(&alpha_row, false, alpha_code.as_deref(), width, 80);
    let supercali_line =
        session_list_line(&supercali_row, false, supercali_code.as_deref(), width, 80);
    let beta_line = session_list_line(&beta_row, false, beta_code.as_deref(), width, 80);

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
fn host_codes_extend_after_z() {
    let entries = (0..27)
        .map(|index| HostEntry {
            id: ssh_host_id(&format!("ssh://host{index}")),
            label: format!("host{index}"),
            ip_address: "x".to_string(),
            handle: HostHandle::local(),
        })
        .collect::<Vec<_>>();
    let fleet = HostFleet::from_entries(entries);

    assert_eq!(fleet.host_code_width(), "[AA]".len());
    assert_eq!(
        fleet.host_code(&ssh_host_id("ssh://host0")).as_deref(),
        Some("[A]")
    );
    assert_eq!(
        fleet.host_code(&ssh_host_id("ssh://host25")).as_deref(),
        Some("[Z]")
    );
    assert_eq!(
        fleet.host_code(&ssh_host_id("ssh://host26")).as_deref(),
        Some("[AA]")
    );
}

#[test]
fn multi_host_session_row_inserts_host_code_column() {
    let row = make_row_for_host(session("dev", "$1"), ssh_host_id("ssh://a"), "alpha");
    let line = session_list_line(&row, false, Some("[A]"), "[A]".len(), 60);
    // Format expected:  " * [A] dev"  (leading marker, attached, host code,
    // session name). The full hostname is reserved for the top legend.
    assert!(line.contains("[A]"));
    assert!(!line.contains("alpha"));
    assert!(line.contains("dev"));
    let code_pos = line.find("[A]").expect("host code rendered");
    let dev_pos = line.find("dev").expect("session name rendered");
    assert!(
        code_pos < dev_pos,
        "host code appears before session name: {line:?}"
    );
}

#[test]
fn single_host_row_omits_hostname_column() {
    let row = make_row(session("dev", "$1"));
    let line = session_list_line(&row, false, None, 0, 30);
    // No host-code column when host_code_width = 0 (single-host).
    assert!(line.contains("dev"));
    assert!(!line.contains("[A]"));
}

#[test]
fn multi_host_sort_merges_rows_by_activity_across_hosts() {
    let mut app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );
    app.fleet = HostFleet::from_entries(vec![
        ssh_host_entry("ssh://a", "alpha", "x", HostHandle::local()),
        ssh_host_entry("ssh://b", "beta", "y", HostHandle::local()),
    ]);
    let rows = vec![
        make_row_for_host(
            session_with_times("alpha-old", "$1", 10, 100),
            ssh_host_id("ssh://a"),
            "alpha",
        ),
        make_row_for_host(
            session_with_times("beta-fresh", "$2", 20, 500),
            ssh_host_id("ssh://b"),
            "beta",
        ),
        make_row_for_host(
            session_with_times("alpha-fresh", "$3", 30, 400),
            ssh_host_id("ssh://a"),
            "alpha",
        ),
        make_row_for_host(
            session_with_times("beta-old", "$4", 40, 200),
            ssh_host_id("ssh://b"),
            "beta",
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
    assert_eq!(selected.id, "$2");
}

#[test]
fn selection_preserves_host_and_session_after_multi_host_reorder() {
    let mut app = AppState::new(
        "host".to_string(),
        LayoutMode::Normal,
        "motd".to_string(),
        false,
    );
    app.fleet = HostFleet::from_entries(vec![
        ssh_host_entry("ssh://a", "alpha", "x", HostHandle::local()),
        ssh_host_entry("ssh://b", "beta", "y", HostHandle::local()),
    ]);
    app.session_list.set_rows_sorted_by_activity(vec![
        make_row_for_host(
            session_with_times("aa", "$1", 10, 200),
            ssh_host_id("ssh://a"),
            "alpha",
        ),
        make_row_for_host(
            session_with_times("bb", "$2", 20, 100),
            ssh_host_id("ssh://b"),
            "beta",
        ),
    ]);
    // First row by activity is alpha/$1.
    app.session_list.selected = 0;
    let key = app.selected_session().map(|s| (s.host_id, s.id)).unwrap();
    assert_eq!(key.0, ssh_host_id("ssh://a"));

    // Reorder: now bb is most active. Selection should preserve alpha/$1.
    app.session_list.set_rows_sorted_by_activity(vec![
        make_row_for_host(
            session_with_times("bb", "$2", 20, 500),
            ssh_host_id("ssh://b"),
            "beta",
        ),
        make_row_for_host(
            session_with_times("aa", "$1", 10, 200),
            ssh_host_id("ssh://a"),
            "alpha",
        ),
    ]);
    app.preserve_selection(Some(key));
    let still = app.selected_session().unwrap();
    assert_eq!(still.host_id, ssh_host_id("ssh://a"));
    assert_eq!(still.id, "$1");
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
