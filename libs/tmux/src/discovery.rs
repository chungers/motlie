use crate::error::{Error, Result};
use regex::Regex;
use std::collections::HashMap;

use crate::transport::{shell_escape_arg, TransportKind};
use crate::types::{
    ClientInfo, GeometrySnapshot, PaneAddress, PaneGeometry, PaneInfo, SessionId, SessionInfo,
    WindowInfo,
};

/// Parse a line of tmux `q:`-escaped fields separated by spaces.
///
/// tmux's `#{q:field}` format modifier shell-escapes each value using
/// backslash sequences (e.g. `\ ` for space, `\<` for `<`). Fields are
/// separated by unescaped spaces. An empty field between two separators
/// (double space) produces an empty string.
pub fn parse_escaped_fields(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '\\' => {
                // Backslash-escaped character: take the next char literally
                if let Some(next) = chars.next() {
                    current.push(next);
                }
            }
            ' ' => {
                // Unescaped space = field separator
                fields.push(std::mem::take(&mut current));
            }
            _ => {
                current.push(ch);
            }
        }
    }
    // Don't forget the last field
    fields.push(current);

    fields
}

pub const LIST_CLIENTS_FMT: &str = "#{q:client_width} #{q:client_height} #{q:client_session} #{q:client_activity} #{q:client_readonly} #{q:client_tty}";

pub const PANE_GEOMETRY_FMT: &str =
    "#{q:pane_width} #{q:pane_height} #{q:history_size} #{q:history_limit}";

pub const LIST_SESSIONS_FMT: &str =
    "#{q:session_name} #{q:session_id} #{q:session_created} #{q:session_attached} #{q:session_windows} #{q:session_group} #{q:session_activity}";

/// Sentinels prepended to each row of the merged `list-sessions \;
/// list-windows -a` output so the parser can dispatch unambiguously.
///
/// Both row kinds carry an explicit tag rather than relying on "session
/// rows are whatever doesn't match the window prefix" — a session whose
/// `session_name` happened to start with the window prefix would otherwise
/// be silently treated as a window-activity row. With both rows tagged,
/// any line that lacks both prefixes is an unframed parse error and a
/// session named `__MOTLIE_WIN__` parses correctly because its name
/// arrives in a separate field after the `__MOTLIE_S__` tag.
///
/// See issue #237 for context on why we need to chain `list-windows -a`
/// (tmux's `session_activity` only tracks attached-client input; program
/// output advances `window_activity` per window).
const SESSION_LISTING_S_PREFIX: &str = "__MOTLIE_S__";
const SESSION_LISTING_WIN_PREFIX: &str = "__MOTLIE_WIN__";

pub const LIST_WINDOWS_FMT: &str =
    "#{q:session_id} #{q:session_name} #{q:window_index} #{q:window_name} #{q:window_active} #{q:window_panes} #{q:window_layout}";

pub const LIST_PANES_FMT: &str =
    "#{q:pane_id} #{q:session_name} #{q:window_index} #{q:pane_index} #{q:pane_title} #{q:pane_current_command} #{q:pane_pid} #{q:pane_width} #{q:pane_height} #{q:pane_active}";

/// List all tmux sessions using a caller-provided prefix (resolved binary + socket).
pub(crate) async fn list_sessions_with_prefix(
    transport: &TransportKind,
    prefix: &str,
) -> Result<Vec<SessionInfo>> {
    let cmd = list_sessions_with_windows_command(prefix);

    let output = match transport.exec(&cmd).await {
        Ok(o) => o,
        Err(e) => {
            let msg = e.to_string();
            if is_tmux_empty_state_error(&msg, &cmd, &["no server running", "no sessions"]) {
                return Ok(Vec::new());
            }
            return Err(e);
        }
    };

    parse_session_block(&output)
}

/// List windows in a session using a caller-provided prefix.
pub(crate) async fn list_windows_with_prefix(
    transport: &TransportKind,
    prefix: &str,
    session: &str,
) -> Result<Vec<WindowInfo>> {
    let cmd = format!(
        "{} list-windows -t '{}' -F '{}'",
        prefix,
        crate::transport::shell_escape_arg(session),
        LIST_WINDOWS_FMT
    );

    let output = transport.exec(&cmd).await?;
    parse_windows(&output)
}

/// List panes in a specific session using a caller-provided prefix.
pub(crate) async fn list_panes_in_session_with_prefix(
    transport: &TransportKind,
    prefix: &str,
    session: &str,
) -> Result<Vec<PaneInfo>> {
    let cmd = format!(
        "{} list-panes -s -t '{}' -F '{}'",
        prefix,
        crate::transport::shell_escape_arg(session),
        LIST_PANES_FMT
    );
    let output = transport.exec(&cmd).await?;
    parse_panes(&output, None)
}

/// List attached tmux clients using a caller-provided prefix (DC20, Phase 1.9b).
pub(crate) async fn list_clients_with_prefix(
    transport: &TransportKind,
    prefix: &str,
) -> Result<Vec<ClientInfo>> {
    let cmd = format!("{} list-clients -F '{}'", prefix, LIST_CLIENTS_FMT);

    let output = match transport.exec(&cmd).await {
        Ok(o) => o,
        Err(e) => {
            let msg = e.to_string();
            if is_tmux_empty_state_error(&msg, &cmd, &["no server running", "no clients"]) {
                return Ok(Vec::new());
            }
            return Err(e);
        }
    };

    parse_clients(&output)
}

/// Query pane geometry using a caller-provided prefix (DC20, Phase 1.9b).
pub(crate) async fn query_pane_geometry_with_prefix(
    transport: &TransportKind,
    prefix: &str,
    target: &str,
) -> Result<PaneGeometry> {
    let cmd = format!(
        "{} display-message -p -t '{}' '{}'",
        prefix,
        shell_escape_arg(target),
        PANE_GEOMETRY_FMT
    );

    let output = transport.exec(&cmd).await?;
    parse_pane_geometry(&output)
}

/// Take a full geometry snapshot using a caller-provided prefix (DC20, Phase 1.9b).
///
/// `target` is a tmux target string (e.g. "build:0.0"). The session name
/// is extracted from the target to scope client filtering.
pub(crate) async fn take_geometry_snapshot_with_prefix(
    transport: &TransportKind,
    prefix: &str,
    target: &str,
) -> Result<GeometrySnapshot> {
    let clients = list_clients_with_prefix(transport, prefix).await?;
    let pane = query_pane_geometry_with_prefix(transport, prefix, target).await?;

    // Extract session name from target string for session-scoped client filtering.
    // Target formats: "session", "session:window", "session:window.pane", "%pane_id"
    let session = if target.starts_with('%') {
        // Pane ID target — query tmux for session name
        let cmd = format!(
            "{} display-message -p -t '{}' '#{{session_name}}'",
            prefix,
            shell_escape_arg(target)
        );
        transport
            .exec(&cmd)
            .await
            .map(|s| s.trim().to_string())
            .map_err(|e| {
                Error::Command(format!(
                    "failed to resolve session for pane target '{}': {}",
                    target, e
                ))
            })?
    } else {
        // Parse session from target string (everything before first ':')
        target.split(':').next().unwrap_or(target).to_string()
    };

    Ok(GeometrySnapshot {
        clients,
        pane,
        session,
    })
}

pub(crate) fn is_tmux_empty_state_error(message: &str, command: &str, patterns: &[&str]) -> bool {
    if !message.starts_with("command failed (exit 1): ") {
        return false;
    }
    if !message.contains(command) {
        return false;
    }

    patterns.iter().any(|pattern| {
        message
            .lines()
            .any(|line| line.starts_with("stderr:") && line.contains(pattern))
    })
}

fn parse_exact_fields(line: &str, expected: usize, kind: &str) -> Result<Vec<String>> {
    let fields = parse_escaped_fields(line);
    if fields.len() != expected {
        return Err(Error::Command(format!(
            "malformed {} line (expected {} fields, got {}): {}",
            kind,
            expected,
            fields.len(),
            line
        )));
    }
    Ok(fields)
}

fn parse_u32_field(value: &str, field: &str, kind: &str, line: &str) -> Result<u32> {
    value.parse().map_err(|e| {
        Error::Parse(format!(
            "invalid {} {} value '{}': {} (line: {})",
            kind, field, value, e, line
        ))
    })
}

fn parse_client_dimension_field(value: &str, field: &str, line: &str) -> Result<u32> {
    if value.is_empty() {
        return Ok(0);
    }
    parse_u32_field(value, field, "client", line)
}

fn parse_u64_field(value: &str, field: &str, kind: &str, line: &str) -> Result<u64> {
    value.parse().map_err(|e| {
        Error::Parse(format!(
            "invalid {} {} value '{}': {} (line: {})",
            kind, field, value, e, line
        ))
    })
}

fn list_sessions_with_windows_command(prefix: &str) -> String {
    format!(
        "{} list-sessions -F '{} {}' \\; \
         list-windows -a -F '{} #{{q:session_id}} #{{q:window_activity}}'",
        prefix, SESSION_LISTING_S_PREFIX, LIST_SESSIONS_FMT, SESSION_LISTING_WIN_PREFIX
    )
}

/// Split the merged output of `list-sessions \; list-windows -a` into
/// session lines and window-activity lines using the explicit
/// `__MOTLIE_S__` / `__MOTLIE_WIN__` tags, then run the session parser
/// and apply per-session window-activity aggregation.
///
/// Any non-empty line that carries neither tag is an unframed parse error
/// — silent fallback would mask a real tmux/format regression.
fn parse_session_block(output: &str) -> Result<Vec<SessionInfo>> {
    let mut session_lines: Vec<&str> = Vec::new();
    let mut window_lines: Vec<&str> = Vec::new();
    for line in output.lines() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix(SESSION_LISTING_S_PREFIX) {
            session_lines.push(rest.trim_start());
        } else if line.starts_with(SESSION_LISTING_WIN_PREFIX) {
            window_lines.push(line);
        } else {
            return Err(Error::Command(format!(
                "unframed session-listing row (expected `{}` or `{}` prefix): {}",
                SESSION_LISTING_S_PREFIX, SESSION_LISTING_WIN_PREFIX, line
            )));
        }
    }
    let session_output = session_lines.join("\n");
    let mut sessions = parse_sessions(&session_output)?;
    let window_activity = parse_window_activity_block(&window_lines)?;
    apply_window_activity_aggregation(&mut sessions, &window_activity);
    Ok(sessions)
}

/// Build a `HashMap<session_id, max(window_activity)>` from the trailing
/// `__MOTLIE_WIN__ <session_id> <window_activity>` block. Tmux issues one row
/// per window across the entire server; we fold to a per-session max.
///
/// Malformed rows return `Error::Command`. Silent fallback would make
/// activity stale and hide parser/format regressions; if a tmux build
/// emits an unparseable row, the lib reports it instead of papering over.
fn parse_window_activity_block(lines: &[&str]) -> Result<HashMap<String, u64>> {
    let mut map: HashMap<String, u64> = HashMap::new();
    for line in lines {
        let rest = line
            .strip_prefix(SESSION_LISTING_WIN_PREFIX)
            .ok_or_else(|| {
                Error::Command(format!(
                    "window-activity row missing `{}` prefix: {}",
                    SESSION_LISTING_WIN_PREFIX, line
                ))
            })?;
        let trimmed = rest.trim_start();
        let fields = parse_escaped_fields(trimmed);
        if fields.len() != 2 {
            return Err(Error::Command(format!(
                "malformed window-activity row (expected 2 fields, got {}): {}",
                fields.len(),
                line
            )));
        }
        if fields[0].is_empty() {
            return Err(Error::Command(format!(
                "malformed window-activity row (empty session id): {}",
                line
            )));
        }
        let activity = fields[1].parse::<u64>().map_err(|err| {
            Error::Parse(format!(
                "invalid window_activity value '{}' in window row {:?}: {}",
                fields[1], line, err
            ))
        })?;
        map.entry(fields[0].clone())
            .and_modify(|current| {
                if activity > *current {
                    *current = activity;
                }
            })
            .or_insert(activity);
    }
    Ok(map)
}

/// Update each session's `activity` to `max(session_activity, max(window_activity))`.
///
/// `session_activity` from tmux only tracks attached-client input (see issue
/// #237 for the empirical confirmation). Program output writes only advance
/// `window_activity` per window. Aggregating to the session level here gives
/// `SessionInfo.activity` the "any-program-output OR any-client-input" meaning
/// that recency UIs actually want.
fn apply_window_activity_aggregation(
    sessions: &mut [SessionInfo],
    window_activity: &HashMap<String, u64>,
) {
    for session in sessions.iter_mut() {
        if let Some(&max_window) = window_activity.get(session.id.as_str()) {
            if max_window > session.activity {
                session.activity = max_window;
            }
        }
    }
}

fn parse_bool_field(value: &str, field: &str, kind: &str, line: &str) -> Result<bool> {
    match value {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err(Error::Parse(format!(
            "invalid {} {} flag '{}': expected 0 or 1 (line: {})",
            kind, field, value, line
        ))),
    }
}

fn parse_clients(output: &str) -> Result<Vec<ClientInfo>> {
    let mut clients = Vec::new();
    for line in output.lines() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        let fields = parse_exact_fields(line, 6, "client")?;
        clients.push(ClientInfo {
            width: parse_client_dimension_field(&fields[0], "width", line)?,
            height: parse_client_dimension_field(&fields[1], "height", line)?,
            session: fields[2].clone(),
            activity: parse_u64_field(&fields[3], "activity", "client", line)?,
            readonly: parse_bool_field(&fields[4], "readonly", "client", line)?,
            tty: if fields[5].is_empty() {
                None
            } else {
                Some(fields[5].clone())
            },
        });
    }
    Ok(clients)
}

fn parse_pane_geometry(output: &str) -> Result<PaneGeometry> {
    let line = output.trim_end_matches('\n').trim_end_matches('\r');
    let fields = parse_exact_fields(line, 4, "pane geometry")?;
    Ok(PaneGeometry {
        pane_width: parse_u32_field(&fields[0], "pane_width", "pane geometry", line)?,
        pane_height: parse_u32_field(&fields[1], "pane_height", "pane geometry", line)?,
        history_size: parse_u32_field(&fields[2], "history_size", "pane geometry", line)?,
        history_limit: parse_u32_field(&fields[3], "history_limit", "pane geometry", line)?,
    })
}

pub(crate) fn parse_sessions(output: &str) -> Result<Vec<SessionInfo>> {
    let mut sessions = Vec::new();
    for line in output.lines() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        let fields = parse_exact_fields(line, 7, "session")?;
        let attached_count = parse_u32_field(&fields[3], "attached", "session", line)?;
        sessions.push(SessionInfo {
            name: fields[0].clone(),
            id: SessionId::new(fields[1].clone())?,
            created: parse_u64_field(&fields[2], "created", "session", line)?,
            attached_count,
            window_count: parse_u32_field(&fields[4], "window_count", "session", line)?,
            group: if fields[5].is_empty() {
                None
            } else {
                Some(fields[5].clone())
            },
            activity: parse_u64_field(&fields[6], "activity", "session", line)?,
        });
    }
    Ok(sessions)
}

fn parse_windows(output: &str) -> Result<Vec<WindowInfo>> {
    let mut windows = Vec::new();
    for line in output.lines() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        let fields = parse_exact_fields(line, 7, "window")?;
        windows.push(WindowInfo {
            session_id: fields[0].clone(),
            session_name: fields[1].clone(),
            index: parse_u32_field(&fields[2], "index", "window", line)?,
            name: fields[3].clone(),
            active: parse_bool_field(&fields[4], "active", "window", line)?,
            pane_count: parse_u32_field(&fields[5], "pane_count", "window", line)?,
            layout: fields[6].clone(),
        });
    }
    Ok(windows)
}

fn parse_panes(output: &str, filter: Option<&Regex>) -> Result<Vec<PaneInfo>> {
    let mut panes = Vec::new();
    for line in output.lines() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() {
            continue;
        }
        let fields = parse_exact_fields(line, 10, "pane")?;

        // fields: pane_id, session_name, window_index, pane_index, pane_title,
        //         pane_current_command, pane_pid, pane_width, pane_height, pane_active
        let session_name = &fields[1];
        let window_index = parse_u32_field(&fields[2], "window_index", "pane", line)?;
        let pane_index = parse_u32_field(&fields[3], "pane_index", "pane", line)?;

        // Construct the address string for filter matching
        let address_str = format!("{}:{}.{}", session_name, window_index, pane_index);

        if let Some(re) = filter {
            if !re.is_match(&address_str) {
                continue;
            }
        }

        let address = PaneAddress {
            pane_id: fields[0].clone(),
            session: session_name.clone(),
            window: window_index,
            pane: pane_index,
        };

        panes.push(PaneInfo {
            address,
            title: fields[4].clone(),
            current_command: fields[5].clone(),
            pid: parse_u32_field(&fields[6], "pid", "pane", line)?,
            width: parse_u32_field(&fields[7], "width", "pane", line)?,
            height: parse_u32_field(&fields[8], "height", "pane", line)?,
            active: parse_bool_field(&fields[9], "active", "pane", line)?,
        });
    }
    Ok(panes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::MockTransport;

    #[test]
    fn parse_escaped_fields_simple() {
        let fields = parse_escaped_fields("build $0 123");
        assert_eq!(fields, vec!["build", "$0", "123"]);
    }

    #[test]
    fn parse_escaped_fields_escaped_spaces() {
        let fields = parse_escaped_fields("name\\ with\\ spaces $0");
        assert_eq!(fields, vec!["name with spaces", "$0"]);
    }

    #[test]
    fn parse_escaped_fields_empty_field() {
        let fields = parse_escaped_fields("build  123");
        assert_eq!(fields, vec!["build", "", "123"]);
    }

    #[test]
    fn parse_escaped_fields_escaped_special_chars() {
        let fields = parse_escaped_fields("name\\<\\|\\>split $0");
        assert_eq!(fields, vec!["name<|>split", "$0"]);
    }

    #[test]
    fn parse_escaped_fields_escaped_quotes() {
        let fields = parse_escaped_fields("name\\'s $0");
        assert_eq!(fields, vec!["name's", "$0"]);
    }

    #[test]
    fn parse_escaped_fields_trailing_empty() {
        // Empty last field (trailing space)
        let fields = parse_escaped_fields("build $0 ");
        assert_eq!(fields, vec!["build", "$0", ""]);
    }

    #[tokio::test]
    async fn list_sessions_parses_output() {
        let mock = MockTransport::new().with_response(
            "list-sessions",
            "__MOTLIE_S__ build $0 1709900000 1 3  1709900200\n\
             __MOTLIE_S__ test $1 1709900100 0 1 group1 1709900300\n",
        );
        let transport = TransportKind::Mock(mock);
        let sessions = list_sessions_with_prefix(&transport, "tmux").await.unwrap();
        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions[0].name, "build");
        assert_eq!(sessions[0].id.as_str(), "$0");
        assert_eq!(sessions[0].attached_count, 1);
        assert!(sessions[0].is_attached());
        assert_eq!(sessions[0].window_count, 3);
        assert!(sessions[0].group.is_none());
        assert_eq!(sessions[0].activity, 1709900200);
        assert_eq!(sessions[1].name, "test");
        assert_eq!(sessions[1].attached_count, 0);
        assert!(!sessions[1].is_attached());
        assert_eq!(sessions[1].group.as_deref(), Some("group1"));
        assert_eq!(sessions[1].activity, 1709900300);
    }

    #[tokio::test]
    async fn list_sessions_empty() {
        let mock = MockTransport::new().with_response("list-sessions", "");
        let transport = TransportKind::Mock(mock);
        let sessions = list_sessions_with_prefix(&transport, "tmux").await.unwrap();
        assert!(sessions.is_empty());
    }

    /// Issue #237: tmux's `session_activity` only tracks attached-client
    /// input. Program output advances `window_activity` per window. Fold the
    /// max window activity into `SessionInfo.activity` so the recency display
    /// reflects "any-output OR any-input."
    #[tokio::test]
    async fn session_activity_aggregates_max_window_activity() {
        // Session $0 has session_activity=200, window_activity=500 (output is
        // newer than the last input). Final activity should be 500.
        // Session $1 has session_activity=900, window_activity=300 (input
        // is newer than output). Final activity should stay at 900.
        let mock = MockTransport::new().with_response(
            "list-sessions",
            concat!(
                "__MOTLIE_S__ build $0 100 0 1  200\n",
                "__MOTLIE_S__ talky $1 100 0 1  900\n",
                "__MOTLIE_WIN__ $0 500\n",
                "__MOTLIE_WIN__ $1 300\n",
            ),
        );
        let transport = TransportKind::Mock(mock);

        let sessions = list_sessions_with_prefix(&transport, "tmux").await.unwrap();

        assert_eq!(sessions.len(), 2);
        let build = &sessions[0];
        let talky = &sessions[1];
        assert_eq!(build.name, "build");
        assert_eq!(
            build.activity, 500,
            "window_activity (500) > session_activity (200): aggregator picks 500"
        );
        assert_eq!(talky.name, "talky");
        assert_eq!(
            talky.activity, 900,
            "session_activity (900) > window_activity (300): aggregator preserves 900"
        );
    }

    #[tokio::test]
    async fn session_activity_uses_max_across_multiple_windows() {
        // A session with two windows: oldest, middle, newest. Aggregate to the
        // newest. session_activity is older than both — final should be newest.
        let mock = MockTransport::new().with_response(
            "list-sessions",
            concat!(
                "__MOTLIE_S__ multi $7 100 0 2  200\n",
                "__MOTLIE_WIN__ $7 300\n",
                "__MOTLIE_WIN__ $7 700\n",
                "__MOTLIE_WIN__ $7 500\n",
            ),
        );
        let transport = TransportKind::Mock(mock);

        let sessions = list_sessions_with_prefix(&transport, "tmux").await.unwrap();

        assert_eq!(sessions[0].activity, 700);
    }

    #[tokio::test]
    async fn session_activity_unchanged_when_no_window_rows_present() {
        // Defensive: if the chained list-windows somehow returns nothing for
        // a session id, the session falls back to its raw session_activity.
        let mock = MockTransport::new()
            .with_response("list-sessions", "__MOTLIE_S__ lonely $9 100 0 1  250\n");
        let transport = TransportKind::Mock(mock);

        let sessions = list_sessions_with_prefix(&transport, "tmux").await.unwrap();

        assert_eq!(sessions[0].activity, 250);
    }

    #[test]
    fn parse_window_activity_block_clean_input_succeeds() {
        let map = parse_window_activity_block(&[
            "__MOTLIE_WIN__ $0 700",
            "__MOTLIE_WIN__ $0 200",
            "__MOTLIE_WIN__ $1 350",
        ])
        .unwrap();
        assert_eq!(map.get("$0").copied(), Some(700));
        assert_eq!(map.get("$1").copied(), Some(350));
    }

    #[test]
    fn parse_window_activity_block_errors_on_unparseable_timestamp() {
        // Stricter contract (reviewer feedback): lib refuses to silently
        // drop unparseable rows because that would mask format regressions.
        let err = parse_window_activity_block(&["__MOTLIE_WIN__ $0 not-a-number"]).unwrap_err();
        assert!(err.to_string().contains("invalid window_activity value"));
    }

    #[test]
    fn parse_window_activity_block_errors_on_missing_timestamp() {
        let err = parse_window_activity_block(&["__MOTLIE_WIN__ $0"]).unwrap_err();
        assert!(err.to_string().contains("malformed window-activity row"));
    }

    #[test]
    fn parse_window_activity_block_errors_on_missing_session_id() {
        // After stripping the prefix and trimming whitespace, a row with
        // only a timestamp parses as one field — caught by the
        // expected-two-fields check.
        let err = parse_window_activity_block(&["__MOTLIE_WIN__  500"]).unwrap_err();
        assert!(err.to_string().contains("malformed window-activity row"));
    }

    #[test]
    fn parse_session_block_errors_on_unframed_row() {
        // A line that lacks both __MOTLIE_S__ and __MOTLIE_WIN__ tags is
        // an unframed parse error. This protects against the silent-drop
        // failure mode where a session named __MOTLIE_WIN__ would be
        // misclassified as a window-activity row.
        let err = parse_session_block("build $0 100 0 1  200\n").unwrap_err();
        assert!(err.to_string().contains("unframed session-listing row"));
    }

    #[test]
    fn parse_session_block_handles_session_named_motlie_win() {
        // Adversarial: a session whose tmux-reported name is literally
        // `__MOTLIE_WIN__` parses correctly because the name field comes
        // after the explicit `__MOTLIE_S__` framing tag. Pre-framing
        // partition-by-prefix would have silently dropped this session.
        let sessions =
            parse_session_block("__MOTLIE_S__ __MOTLIE_WIN__ $5 100 0 1  300\n").unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].name, "__MOTLIE_WIN__");
        assert_eq!(sessions[0].id.as_str(), "$5");
    }

    #[tokio::test]
    async fn list_sessions_with_prefix_aggregates_window_activity() {
        let mock = MockTransport::new().with_response(
            "list-sessions",
            concat!(
                "__MOTLIE_S__ build $0 100 0 1  200\n",
                "__MOTLIE_WIN__ $0 750\n",
            ),
        );
        let transport = TransportKind::Mock(mock);

        let sessions = list_sessions_with_prefix(&transport, "tmux").await.unwrap();

        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].activity, 750);
    }

    #[test]
    fn parse_sessions_rejects_empty_session_id() {
        let err = parse_sessions("build  1709900000 0 1  1709900001\n").unwrap_err();
        assert!(err.to_string().contains("empty session id"));
    }

    #[tokio::test]
    async fn list_sessions_malformed_returns_error() {
        // Properly framed but with too-few-fields content surfaces as the
        // session-line malformed error from the inner parse_sessions.
        let mock = MockTransport::new().with_response(
            "list-sessions",
            "__MOTLIE_S__ bad\n__MOTLIE_S__ ok $0 0 0 1 \n",
        );
        let transport = TransportKind::Mock(mock);
        let err = list_sessions_with_prefix(&transport, "tmux")
            .await
            .unwrap_err();
        assert!(err.to_string().contains("malformed session line"));
    }

    #[tokio::test]
    async fn list_sessions_true_empty_state_returns_empty() {
        let cmd = list_sessions_with_windows_command("tmux");
        let mock = MockTransport::new().with_error(
            "list-sessions",
            format!(
                "command failed (exit 1): {}\nstderr: no server running on /tmp/tmux-1000/default",
                cmd
            )
            .as_str(),
        );
        let transport = TransportKind::Mock(mock);
        let sessions = list_sessions_with_prefix(&transport, "tmux").await.unwrap();
        assert!(sessions.is_empty());
    }

    #[tokio::test]
    async fn list_sessions_non_tmux_error_is_not_collapsed() {
        let mock = MockTransport::new().with_error(
            "list-sessions",
            "SSH transport hiccup mentioning no sessions in passing",
        );
        let transport = TransportKind::Mock(mock);
        let err = list_sessions_with_prefix(&transport, "tmux")
            .await
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("SSH transport hiccup mentioning no sessions in passing"));
    }

    #[tokio::test]
    async fn list_sessions_attached_count_above_one_is_true() {
        let mock = MockTransport::new()
            .with_response("list-sessions", "__MOTLIE_S__ build $0 0 2 1  10\n");
        let transport = TransportKind::Mock(mock);
        let sessions = list_sessions_with_prefix(&transport, "tmux").await.unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].attached_count, 2);
        assert!(sessions[0].is_attached());
    }

    #[tokio::test]
    async fn list_windows_parses() {
        let mock = MockTransport::new().with_response(
            "list-windows",
            "$0 build 0 main 1 2 even-horizontal\n$0 build 1 editor 0 1 even-vertical\n",
        );
        let transport = TransportKind::Mock(mock);
        let windows = list_windows_with_prefix(&transport, "tmux", "build")
            .await
            .unwrap();
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].name, "main");
        assert!(windows[0].active);
        assert_eq!(windows[1].name, "editor");
        assert!(!windows[1].active);
    }

    #[tokio::test]
    async fn list_clients_parses() {
        let mock = MockTransport::new().with_response(
            "list-clients",
            "200 50 build 100 0 /dev/ttys001\n180 40 test 80 1 \n80  worker 120 0 \n",
        );
        let transport = TransportKind::Mock(mock);
        let clients = list_clients_with_prefix(&transport, "tmux").await.unwrap();
        assert_eq!(clients.len(), 3);
        assert_eq!(clients[0].width, 200);
        assert_eq!(clients[0].height, 50);
        assert_eq!(clients[0].session, "build");
        assert_eq!(clients[0].activity, 100);
        assert!(!clients[0].readonly);
        assert_eq!(clients[0].tty.as_deref(), Some("/dev/ttys001"));
        assert_eq!(clients[1].width, 180);
        assert_eq!(clients[1].height, 40);
        assert_eq!(clients[1].activity, 80);
        assert!(clients[1].readonly);
        assert_eq!(clients[1].tty, None);
        assert_eq!(clients[2].width, 80);
        assert_eq!(clients[2].height, 0);
        assert_eq!(clients[2].session, "worker");
        assert_eq!(clients[2].activity, 120);
        assert!(!clients[2].readonly);
        assert_eq!(clients[2].tty, None);
    }

    #[tokio::test]
    async fn list_clients_empty() {
        let mock = MockTransport::new().with_response("list-clients", "");
        let transport = TransportKind::Mock(mock);
        let clients = list_clients_with_prefix(&transport, "tmux").await.unwrap();
        assert!(clients.is_empty());
    }

    #[tokio::test]
    async fn list_clients_malformed_returns_error() {
        let mock = MockTransport::new().with_response("list-clients", "200 bad\n");
        let transport = TransportKind::Mock(mock);
        let err = list_clients_with_prefix(&transport, "tmux")
            .await
            .unwrap_err();
        assert!(err.to_string().contains("malformed client line"));
    }

    #[tokio::test]
    async fn query_pane_geometry_parses() {
        let mock = MockTransport::new().with_response("display-message", "80 24 150 2000\n");
        let transport = TransportKind::Mock(mock);
        let geo = query_pane_geometry_with_prefix(&transport, "tmux", "build:0.0")
            .await
            .unwrap();
        assert_eq!(geo.pane_width, 80);
        assert_eq!(geo.pane_height, 24);
        assert_eq!(geo.history_size, 150);
        assert_eq!(geo.history_limit, 2000);
    }

    #[tokio::test]
    async fn query_pane_geometry_invalid_number_returns_error() {
        let mock = MockTransport::new().with_response("display-message", "80 24 bad 2000\n");
        let transport = TransportKind::Mock(mock);
        let err = query_pane_geometry_with_prefix(&transport, "tmux", "build:0.0")
            .await
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("invalid pane geometry history_size value"));
    }

    #[tokio::test]
    async fn take_snapshot_combines_clients_and_pane() {
        let mock = MockTransport::new()
            .with_response("list-clients", "200 50 build 100 0 /dev/ttys001\n")
            .with_response("display-message", "80 24 100 2000\n");
        let transport = TransportKind::Mock(mock);
        let snap = take_geometry_snapshot_with_prefix(&transport, "tmux", "build:0.0")
            .await
            .unwrap();
        assert_eq!(snap.clients.len(), 1);
        assert_eq!(snap.pane.pane_width, 80);
        assert_eq!(snap.pane.history_size, 100);
        assert_eq!(snap.session, "build");
    }

    #[tokio::test]
    async fn list_windows_invalid_flag_returns_error() {
        let mock = MockTransport::new()
            .with_response("list-windows", "$0 build 0 main maybe 2 even-horizontal\n");
        let transport = TransportKind::Mock(mock);
        let err = list_windows_with_prefix(&transport, "tmux", "build")
            .await
            .unwrap_err();
        assert!(err.to_string().contains("invalid window active flag"));
    }

    #[tokio::test]
    async fn take_snapshot_pane_target_session_lookup_failure_is_error() {
        let mock = MockTransport::new()
            .with_response("list-clients", "200 50 build 100 0 /dev/ttys001\n")
            .with_response("display-message", "80 24 100 2000\n")
            .with_error("#{session_name}", "lookup failed");
        let transport = TransportKind::Mock(mock);
        let err = take_geometry_snapshot_with_prefix(&transport, "tmux", "%5")
            .await
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("failed to resolve session for pane target '%5'"));
    }
}
