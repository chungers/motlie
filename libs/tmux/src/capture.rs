use anyhow::Result;
use std::collections::HashMap;

use crate::discovery;
use crate::transport::{shell_escape_arg, tmux_prefix, TransportKind};
use crate::types::{PaneAddress, ScrollbackQuery, TmuxSocket};

/// Capture visible pane content.
pub async fn capture_pane(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
) -> Result<String> {
    let prefix = tmux_prefix(socket);
    let cmd = format!(
        "{} capture-pane -p -t '{}'",
        prefix,
        shell_escape_arg(target)
    );
    transport.exec(&cmd).await
}

/// Capture pane content with scrollback history.
/// `start` is negative to go into scrollback (e.g. -100 = 100 lines above visible area).
/// Captures through the end of the visible area.
pub async fn capture_pane_history(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
    start: i32,
) -> Result<String> {
    let prefix = tmux_prefix(socket);
    let cmd = format!(
        "{} capture-pane -p -t '{}' -S {}",
        prefix,
        shell_escape_arg(target),
        start,
    );
    transport.exec(&cmd).await
}

/// Capture all panes in a session.
pub async fn capture_session(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    session: &str,
) -> Result<HashMap<PaneAddress, String>> {
    let panes = discovery::list_panes_in_session(transport, socket, session).await?;
    let mut result = HashMap::new();
    for pane in panes {
        let target = pane.address.to_tmux_target();
        let content = capture_pane(transport, socket, &target).await?;
        result.insert(pane.address, content);
    }
    Ok(result)
}

/// Sample scrollback text according to a query.
pub async fn sample_text(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
    query: &ScrollbackQuery,
) -> Result<String> {
    match query {
        ScrollbackQuery::LastLines(n) => {
            let n = *n as i32;
            let content = capture_pane_history(transport, socket, target, -n).await?;
            // Trim trailing blank lines from visible area padding
            let trimmed = content.trim_end();
            Ok(trimmed.to_string())
        }
        ScrollbackQuery::Until { pattern, max_lines } => {
            let max = *max_lines as i32;
            let content =
                capture_pane_history(transport, socket, target, -max).await?;
            let lines: Vec<&str> = content.lines().collect();
            // Scan from end to find the pattern
            for (i, line) in lines.iter().enumerate().rev() {
                if pattern.is_match(line) {
                    return Ok(lines[i..].join("\n"));
                }
            }
            // Pattern not found — return all captured
            Ok(content)
        }
        ScrollbackQuery::LastLinesUntil {
            lines,
            stop_pattern,
        } => {
            let n = *lines as i32;
            let content =
                capture_pane_history(transport, socket, target, -n).await?;
            let lines: Vec<&str> = content.lines().collect();
            // Scan from end to find stop pattern
            for (i, line) in lines.iter().enumerate().rev() {
                if stop_pattern.is_match(line) {
                    return Ok(lines[i..].join("\n"));
                }
            }
            Ok(content)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::MockTransport;
    use regex::Regex;

    #[tokio::test]
    async fn capture_pane_basic() {
        let mock = MockTransport::new()
            .with_response("capture-pane", "line1\nline2\nline3\n");
        let transport = TransportKind::Mock(mock);
        let result = capture_pane(&transport, None, "build:0.0").await.unwrap();
        assert_eq!(result, "line1\nline2\nline3\n");
    }

    #[tokio::test]
    async fn sample_text_last_lines() {
        let mock = MockTransport::new()
            .with_response("capture-pane", "line1\nline2\nline3\nline4\nline5\n");
        let transport = TransportKind::Mock(mock);
        let query = ScrollbackQuery::LastLines(5);
        let result = sample_text(&transport, None, "build:0.0", &query)
            .await
            .unwrap();
        assert!(result.contains("line1"));
    }

    #[tokio::test]
    async fn sample_text_until_pattern() {
        let mock = MockTransport::new().with_response(
            "capture-pane",
            "old stuff\n$ prompt\ncommand output\nmore output\n",
        );
        let transport = TransportKind::Mock(mock);
        let query = ScrollbackQuery::Until {
            pattern: Regex::new(r"^\$ ").unwrap(),
            max_lines: 100,
        };
        let result = sample_text(&transport, None, "build:0.0", &query)
            .await
            .unwrap();
        assert!(result.starts_with("$ prompt"));
        assert!(result.contains("command output"));
    }

    #[tokio::test]
    async fn sample_text_last_lines_until() {
        let mock = MockTransport::new().with_response(
            "capture-pane",
            "irrelevant\n--- marker ---\nwanted line 1\nwanted line 2\n",
        );
        let transport = TransportKind::Mock(mock);
        let query = ScrollbackQuery::LastLinesUntil {
            lines: 100,
            stop_pattern: Regex::new(r"--- marker ---").unwrap(),
        };
        let result = sample_text(&transport, None, "test:0.0", &query)
            .await
            .unwrap();
        assert!(result.starts_with("--- marker ---"));
        assert!(result.contains("wanted line 1"));
    }
}
