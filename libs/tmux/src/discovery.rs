use anyhow::{anyhow, Result};
use regex::Regex;

use crate::transport::{shell_escape_arg, tmux_prefix, TransportKind};
use crate::types::{
    ClientInfo, GeometrySnapshot, PaneAddress, PaneGeometry, PaneInfo, SessionInfo, TmuxSocket,
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

pub const LIST_CLIENTS_FMT: &str =
    "#{q:client_width} #{q:client_height} #{q:client_session}";

pub const PANE_GEOMETRY_FMT: &str =
    "#{q:pane_width} #{q:pane_height} #{q:history_size} #{q:history_limit}";

pub const LIST_SESSIONS_FMT: &str =
    "#{q:session_name} #{q:session_id} #{q:session_created} #{q:session_attached} #{q:session_windows} #{q:session_group}";

pub const LIST_WINDOWS_FMT: &str =
    "#{q:session_id} #{q:session_name} #{q:window_index} #{q:window_name} #{q:window_active} #{q:window_panes} #{q:window_layout}";

pub const LIST_PANES_FMT: &str =
    "#{q:pane_id} #{q:session_name} #{q:window_index} #{q:pane_index} #{q:pane_title} #{q:pane_current_command} #{q:pane_pid} #{q:pane_width} #{q:pane_height} #{q:pane_active}";

/// List all tmux sessions (using bare "tmux" prefix).
pub async fn list_sessions(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
) -> Result<Vec<SessionInfo>> {
    list_sessions_with_prefix(transport, &tmux_prefix(socket)).await
}

/// List all tmux sessions using a caller-provided prefix (resolved binary + socket).
pub async fn list_sessions_with_prefix(
    transport: &TransportKind,
    prefix: &str,
) -> Result<Vec<SessionInfo>> {
    let cmd = format!("{} list-sessions -F '{}'", prefix, LIST_SESSIONS_FMT);

    let output = match transport.exec(&cmd).await {
        Ok(o) => o,
        Err(e) => {
            let msg = e.to_string();
            // "no server running" or "no sessions" → empty list
            if msg.contains("no server running") || msg.contains("no sessions") {
                return Ok(Vec::new());
            }
            return Err(e);
        }
    };

    parse_sessions(&output)
}

/// List windows in a session (using bare "tmux" prefix).
pub async fn list_windows(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    session: &str,
) -> Result<Vec<WindowInfo>> {
    list_windows_with_prefix(transport, &tmux_prefix(socket), session).await
}

/// List windows in a session using a caller-provided prefix.
pub async fn list_windows_with_prefix(
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

/// List all panes, optionally filtered by regex (using bare "tmux" prefix).
pub async fn list_panes(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    filter: Option<&Regex>,
) -> Result<Vec<PaneInfo>> {
    list_panes_with_prefix(transport, &tmux_prefix(socket), filter).await
}

/// List all panes using a caller-provided prefix.
pub async fn list_panes_with_prefix(
    transport: &TransportKind,
    prefix: &str,
    filter: Option<&Regex>,
) -> Result<Vec<PaneInfo>> {
    let cmd = format!("{} list-panes -a -F '{}'", prefix, LIST_PANES_FMT);
    let output = transport.exec(&cmd).await?;
    parse_panes(&output, filter)
}

/// List panes in a specific session (using bare "tmux" prefix).
pub async fn list_panes_in_session(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    session: &str,
) -> Result<Vec<PaneInfo>> {
    list_panes_in_session_with_prefix(transport, &tmux_prefix(socket), session).await
}

/// List panes in a specific session using a caller-provided prefix.
pub async fn list_panes_in_session_with_prefix(
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

/// List attached tmux clients (using bare "tmux" prefix).
pub async fn list_clients(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
) -> Result<Vec<ClientInfo>> {
    list_clients_with_prefix(transport, &tmux_prefix(socket)).await
}

/// List attached tmux clients using a caller-provided prefix (DC20, Phase 1.9b).
pub async fn list_clients_with_prefix(
    transport: &TransportKind,
    prefix: &str,
) -> Result<Vec<ClientInfo>> {
    let cmd = format!("{} list-clients -F '{}'", prefix, LIST_CLIENTS_FMT);

    let output = match transport.exec(&cmd).await {
        Ok(o) => o,
        Err(e) => {
            let msg = e.to_string();
            if msg.contains("no server running") || msg.contains("no clients") {
                return Ok(Vec::new());
            }
            return Err(e);
        }
    };

    parse_clients(&output)
}

/// Query pane geometry (using bare "tmux" prefix).
pub async fn query_pane_geometry(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
) -> Result<PaneGeometry> {
    query_pane_geometry_with_prefix(transport, &tmux_prefix(socket), target).await
}

/// Query pane geometry using a caller-provided prefix (DC20, Phase 1.9b).
pub async fn query_pane_geometry_with_prefix(
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

/// Take a full geometry snapshot (using bare "tmux" prefix).
pub async fn take_geometry_snapshot(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
) -> Result<GeometrySnapshot> {
    take_geometry_snapshot_with_prefix(transport, &tmux_prefix(socket), target).await
}

/// Take a full geometry snapshot using a caller-provided prefix (DC20, Phase 1.9b).
///
/// `target` is a tmux target string (e.g. "build:0.0"). The session name
/// is extracted from the target to scope client filtering.
pub async fn take_geometry_snapshot_with_prefix(
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
            .unwrap_or_default()
    } else {
        // Parse session from target string (everything before first ':')
        target
            .split(':')
            .next()
            .unwrap_or(target)
            .to_string()
    };

    Ok(GeometrySnapshot {
        clients,
        pane,
        session,
    })
}

fn parse_clients(output: &str) -> Result<Vec<ClientInfo>> {
    let mut clients = Vec::new();
    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields = parse_escaped_fields(line);
        if fields.len() < 3 {
            tracing::warn!("skipping malformed client line: {}", line);
            continue;
        }
        clients.push(ClientInfo {
            width: fields[0].parse().unwrap_or(0),
            height: fields[1].parse().unwrap_or(0),
            session: fields[2].clone(),
        });
    }
    Ok(clients)
}

fn parse_pane_geometry(output: &str) -> Result<PaneGeometry> {
    let line = output.trim();
    let fields = parse_escaped_fields(line);
    if fields.len() < 4 {
        return Err(anyhow!(
            "malformed pane geometry output (expected 4 fields): {}",
            line
        ));
    }
    Ok(PaneGeometry {
        pane_width: fields[0].parse().unwrap_or(0),
        pane_height: fields[1].parse().unwrap_or(0),
        history_size: fields[2].parse().unwrap_or(0),
        history_limit: fields[3].parse().unwrap_or(0),
    })
}

pub(crate) fn parse_sessions(output: &str) -> Result<Vec<SessionInfo>> {
    let mut sessions = Vec::new();
    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields = parse_escaped_fields(line);
        if fields.len() < 5 {
            tracing::warn!("skipping malformed session line: {}", line);
            continue;
        }
        sessions.push(SessionInfo {
            name: fields[0].clone(),
            id: fields[1].clone(),
            created: fields[2].parse().unwrap_or(0),
            attached: fields[3] == "1",
            window_count: fields[4].parse().unwrap_or(0),
            group: fields.get(5).and_then(|g| {
                if g.is_empty() {
                    None
                } else {
                    Some(g.clone())
                }
            }),
        });
    }
    Ok(sessions)
}

fn parse_windows(output: &str) -> Result<Vec<WindowInfo>> {
    let mut windows = Vec::new();
    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields = parse_escaped_fields(line);
        if fields.len() < 7 {
            tracing::warn!("skipping malformed window line: {}", line);
            continue;
        }
        windows.push(WindowInfo {
            session_id: fields[0].clone(),
            session_name: fields[1].clone(),
            index: fields[2].parse().unwrap_or(0),
            name: fields[3].clone(),
            active: fields[4] == "1",
            pane_count: fields[5].parse().unwrap_or(0),
            layout: fields[6].clone(),
        });
    }
    Ok(windows)
}

fn parse_panes(output: &str, filter: Option<&Regex>) -> Result<Vec<PaneInfo>> {
    let mut panes = Vec::new();
    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields = parse_escaped_fields(line);
        if fields.len() < 10 {
            tracing::warn!("skipping malformed pane line: {}", line);
            continue;
        }

        // fields: pane_id, session_name, window_index, pane_index, pane_title,
        //         pane_current_command, pane_pid, pane_width, pane_height, pane_active
        let session_name = &fields[1];
        let window_index: u32 = fields[2].parse().unwrap_or(0);
        let pane_index: u32 = fields[3].parse().unwrap_or(0);

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
            pid: fields[6].parse().unwrap_or(0),
            width: fields[7].parse().unwrap_or(0),
            height: fields[8].parse().unwrap_or(0),
            active: fields[9] == "1",
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
            "build $0 1709900000 1 3 \ntest $1 1709900100 0 1 group1\n",
        );
        let transport = TransportKind::Mock(mock);
        let sessions = list_sessions(&transport, None).await.unwrap();
        assert_eq!(sessions.len(), 2);
        assert_eq!(sessions[0].name, "build");
        assert_eq!(sessions[0].id, "$0");
        assert!(sessions[0].attached);
        assert_eq!(sessions[0].window_count, 3);
        assert!(sessions[0].group.is_none());
        assert_eq!(sessions[1].name, "test");
        assert!(!sessions[1].attached);
        assert_eq!(sessions[1].group.as_deref(), Some("group1"));
    }

    #[tokio::test]
    async fn list_sessions_empty() {
        let mock = MockTransport::new().with_response("list-sessions", "");
        let transport = TransportKind::Mock(mock);
        let sessions = list_sessions(&transport, None).await.unwrap();
        assert!(sessions.is_empty());
    }

    #[tokio::test]
    async fn list_sessions_malformed_skipped() {
        let mock = MockTransport::new()
            .with_response("list-sessions", "bad\nok $0 0 0 1 \n");
        let transport = TransportKind::Mock(mock);
        let sessions = list_sessions(&transport, None).await.unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].name, "ok");
    }

    #[tokio::test]
    async fn list_windows_parses() {
        let mock = MockTransport::new().with_response(
            "list-windows",
            "$0 build 0 main 1 2 even-horizontal\n$0 build 1 editor 0 1 even-vertical\n",
        );
        let transport = TransportKind::Mock(mock);
        let windows = list_windows(&transport, None, "build").await.unwrap();
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0].name, "main");
        assert!(windows[0].active);
        assert_eq!(windows[1].name, "editor");
        assert!(!windows[1].active);
    }

    #[tokio::test]
    async fn list_panes_parses() {
        // Fields: pane_id session_name window_index pane_index pane_title pane_current_command pane_pid pane_width pane_height pane_active
        let mock = MockTransport::new().with_response(
            "list-panes",
            "%0 build 0 0 title0 bash 1234 80 24 1\n%1 build 0 1 title1 vim 1235 80 24 0\n",
        );
        let transport = TransportKind::Mock(mock);
        let panes = list_panes(&transport, None, None).await.unwrap();
        assert_eq!(panes.len(), 2);
        assert_eq!(panes[0].address.pane_id, "%0");
        assert_eq!(panes[0].address.session, "build");
        assert_eq!(panes[0].address.window, 0);
        assert_eq!(panes[0].address.pane, 0);
        assert_eq!(panes[0].current_command, "bash");
        assert!(panes[0].active);
        assert_eq!(panes[1].address.pane_id, "%1");
        assert_eq!(panes[1].current_command, "vim");
        assert!(!panes[1].active);
    }

    #[tokio::test]
    async fn list_panes_with_filter() {
        // Fields: pane_id session_name window_index pane_index pane_title pane_current_command pane_pid pane_width pane_height pane_active
        let mock = MockTransport::new().with_response(
            "list-panes",
            "%0 build 0 0  bash 1234 80 24 1\n%1 test 0 0  sh 1235 80 24 1\n",
        );
        let transport = TransportKind::Mock(mock);
        let filter = Regex::new("^build").unwrap();
        let panes = list_panes(&transport, None, Some(&filter)).await.unwrap();
        assert_eq!(panes.len(), 1);
        assert_eq!(panes[0].address.session, "build");
    }

    #[tokio::test]
    async fn list_clients_parses() {
        let mock = MockTransport::new().with_response(
            "list-clients",
            "200 50 build\n180 40 test\n",
        );
        let transport = TransportKind::Mock(mock);
        let clients = list_clients(&transport, None).await.unwrap();
        assert_eq!(clients.len(), 2);
        assert_eq!(clients[0].width, 200);
        assert_eq!(clients[0].height, 50);
        assert_eq!(clients[0].session, "build");
        assert_eq!(clients[1].width, 180);
        assert_eq!(clients[1].height, 40);
    }

    #[tokio::test]
    async fn list_clients_empty() {
        let mock = MockTransport::new().with_response("list-clients", "");
        let transport = TransportKind::Mock(mock);
        let clients = list_clients(&transport, None).await.unwrap();
        assert!(clients.is_empty());
    }

    #[tokio::test]
    async fn query_pane_geometry_parses() {
        let mock = MockTransport::new()
            .with_response("display-message", "80 24 150 2000\n");
        let transport = TransportKind::Mock(mock);
        let geo = query_pane_geometry(&transport, None, "build:0.0")
            .await
            .unwrap();
        assert_eq!(geo.pane_width, 80);
        assert_eq!(geo.pane_height, 24);
        assert_eq!(geo.history_size, 150);
        assert_eq!(geo.history_limit, 2000);
    }

    #[tokio::test]
    async fn take_snapshot_combines_clients_and_pane() {
        let mock = MockTransport::new()
            .with_response("list-clients", "200 50 build\n")
            .with_response("display-message", "80 24 100 2000\n");
        let transport = TransportKind::Mock(mock);
        let snap = take_geometry_snapshot(&transport, None, "build:0.0")
            .await
            .unwrap();
        assert_eq!(snap.clients.len(), 1);
        assert_eq!(snap.pane.pane_width, 80);
        assert_eq!(snap.pane.history_size, 100);
        assert_eq!(snap.session, "build");
    }
}
