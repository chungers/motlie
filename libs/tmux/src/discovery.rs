use anyhow::{anyhow, Result};
use regex::Regex;

use crate::transport::{tmux_prefix, TransportKind};
use crate::types::{PaneAddress, PaneInfo, SessionInfo, TmuxSocket, WindowInfo};

pub const LIST_SESSIONS_FMT: &str =
    "#{session_name}\t#{session_id}\t#{session_created}\t#{session_attached}\t#{session_windows}\t#{session_group}";

pub const LIST_WINDOWS_FMT: &str =
    "#{session_id}\t#{session_name}\t#{window_index}\t#{window_name}\t#{window_active}\t#{window_panes}\t#{window_layout}";

pub const LIST_PANES_FMT: &str =
    "#{pane_id}\t#{session_name}:#{window_index}.#{pane_index}\t#{pane_title}\t#{pane_current_command}\t#{pane_pid}\t#{pane_width}\t#{pane_height}\t#{pane_active}";

/// List all tmux sessions.
pub async fn list_sessions(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
) -> Result<Vec<SessionInfo>> {
    let prefix = tmux_prefix(socket);
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

/// List windows in a session.
pub async fn list_windows(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    session: &str,
) -> Result<Vec<WindowInfo>> {
    let prefix = tmux_prefix(socket);
    let cmd = format!(
        "{} list-windows -t '{}' -F '{}'",
        prefix,
        crate::transport::shell_escape_arg(session),
        LIST_WINDOWS_FMT
    );

    let output = transport.exec(&cmd).await?;
    parse_windows(&output)
}

/// List all panes, optionally filtered by regex on session:window.pane.
pub async fn list_panes(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    filter: Option<&Regex>,
) -> Result<Vec<PaneInfo>> {
    let prefix = tmux_prefix(socket);
    let cmd = format!("{} list-panes -a -F '{}'", prefix, LIST_PANES_FMT);
    let output = transport.exec(&cmd).await?;
    parse_panes(&output, filter)
}

/// List panes in a specific session.
pub async fn list_panes_in_session(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    session: &str,
) -> Result<Vec<PaneInfo>> {
    let prefix = tmux_prefix(socket);
    let cmd = format!(
        "{} list-panes -s -t '{}' -F '{}'",
        prefix,
        crate::transport::shell_escape_arg(session),
        LIST_PANES_FMT
    );
    let output = transport.exec(&cmd).await?;
    parse_panes(&output, None)
}

fn parse_sessions(output: &str) -> Result<Vec<SessionInfo>> {
    let mut sessions = Vec::new();
    for line in output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 5 {
            tracing::warn!("skipping malformed session line: {}", line);
            continue;
        }
        sessions.push(SessionInfo {
            name: fields[0].to_string(),
            id: fields[1].to_string(),
            created: fields[2].parse().unwrap_or(0),
            attached: fields[3] == "1",
            window_count: fields[4].parse().unwrap_or(0),
            group: fields.get(5).and_then(|g| {
                if g.is_empty() {
                    None
                } else {
                    Some(g.to_string())
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
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 7 {
            tracing::warn!("skipping malformed window line: {}", line);
            continue;
        }
        windows.push(WindowInfo {
            session_id: fields[0].to_string(),
            session_name: fields[1].to_string(),
            index: fields[2].parse().unwrap_or(0),
            name: fields[3].to_string(),
            active: fields[4] == "1",
            pane_count: fields[5].parse().unwrap_or(0),
            layout: fields[6].to_string(),
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
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 8 {
            tracing::warn!("skipping malformed pane line: {}", line);
            continue;
        }

        // fields[0] = pane_id (%N), fields[1] = session:window.pane
        let address_str = fields[1];

        if let Some(re) = filter {
            if !re.is_match(address_str) {
                continue;
            }
        }

        let address = PaneAddress::parse(fields[0], address_str)
            .map_err(|e| anyhow!("failed to parse pane address '{}': {}", address_str, e))?;

        panes.push(PaneInfo {
            address,
            title: fields[2].to_string(),
            current_command: fields[3].to_string(),
            pid: fields[4].parse().unwrap_or(0),
            width: fields[5].parse().unwrap_or(0),
            height: fields[6].parse().unwrap_or(0),
            active: fields[7] == "1",
        });
    }
    Ok(panes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::MockTransport;

    #[tokio::test]
    async fn list_sessions_parses_output() {
        let mock = MockTransport::new().with_response(
            "list-sessions",
            "build\t$0\t1709900000\t1\t3\t\ntest\t$1\t1709900100\t0\t1\tgroup1\n",
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
            .with_response("list-sessions", "bad\nok\t$0\t0\t0\t1\t\n");
        let transport = TransportKind::Mock(mock);
        let sessions = list_sessions(&transport, None).await.unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].name, "ok");
    }

    #[tokio::test]
    async fn list_windows_parses() {
        let mock = MockTransport::new().with_response(
            "list-windows",
            "$0\tbuild\t0\tmain\t1\t2\teven-horizontal\n$0\tbuild\t1\teditor\t0\t1\teven-vertical\n",
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
        let mock = MockTransport::new().with_response(
            "list-panes",
            "%0\tbuild:0.0\ttitle0\tbash\t1234\t80\t24\t1\n%1\tbuild:0.1\ttitle1\tvim\t1235\t80\t24\t0\n",
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
        let mock = MockTransport::new().with_response(
            "list-panes",
            "%0\tbuild:0.0\t\tbash\t1234\t80\t24\t1\n%1\ttest:0.0\t\tsh\t1235\t80\t24\t1\n",
        );
        let transport = TransportKind::Mock(mock);
        let filter = Regex::new("^build").unwrap();
        let panes = list_panes(&transport, None, Some(&filter)).await.unwrap();
        assert_eq!(panes.len(), 1);
        assert_eq!(panes[0].address.session, "build");
    }
}
