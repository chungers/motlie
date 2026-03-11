use anyhow::Result;

use crate::keys::KeySequence;
use crate::transport::{shell_escape_arg, tmux_prefix, TransportKind};
use crate::types::TmuxSocket;

/// Shell-escape a string for safe interpolation into shell commands.
/// Uses POSIX single-quote wrapping: wrap in single quotes, escape
/// interior single quotes with '\'' (OC5).
pub fn shell_escape(s: &str) -> String {
    format!("'{}'", shell_escape_arg(s))
}

/// Create a new detached tmux session.
pub async fn create_session(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    name: &str,
    window_name: Option<&str>,
    command: Option<&str>,
) -> Result<()> {
    let prefix = tmux_prefix(socket);
    let mut cmd = format!("{} new-session -d -s {}", prefix, shell_escape(name));

    if let Some(wn) = window_name {
        cmd.push_str(&format!(" -n {}", shell_escape(wn)));
    }
    if let Some(c) = command {
        cmd.push_str(&format!(" {}", shell_escape(c)));
    }

    transport.exec(&cmd).await?;
    Ok(())
}

/// Kill a tmux session by name.
pub async fn kill_session(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    name: &str,
) -> Result<()> {
    let prefix = tmux_prefix(socket);
    let cmd = format!("{} kill-session -t {}", prefix, shell_escape(name));
    transport.exec(&cmd).await?;
    Ok(())
}

/// Kill a tmux window.
pub async fn kill_window(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
) -> Result<()> {
    let prefix = tmux_prefix(socket);
    let cmd = format!("{} kill-window -t {}", prefix, shell_escape(target));
    transport.exec(&cmd).await?;
    Ok(())
}

/// Kill a tmux pane.
pub async fn kill_pane(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
) -> Result<()> {
    let prefix = tmux_prefix(socket);
    let cmd = format!("{} kill-pane -t {}", prefix, shell_escape(target));
    transport.exec(&cmd).await?;
    Ok(())
}

/// Send a key sequence to a tmux target.
pub async fn send_keys(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
    keys: &KeySequence,
) -> Result<()> {
    let prefix = tmux_prefix(socket);
    let tmux_target = shell_escape(target);

    for args in keys.to_tmux_args(target) {
        // Build the full command: tmux [socket] send-keys [-l] -t target [text|key]
        // All arguments are shell-escaped to prevent injection via Raw key names.
        let has_literal = args.contains(&"-l".to_string());
        let mut cmd = prefix.clone();
        cmd.push_str(" send-keys");
        if has_literal {
            cmd.push_str(" -l");
        }
        cmd.push_str(&format!(" -t {}", tmux_target));
        // Last element is the value (text or key name)
        let value = &args[args.len() - 1];
        cmd.push_str(&format!(" {}", shell_escape(value)));
        transport.exec(&cmd).await?;
    }
    Ok(())
}

/// Send literal text to a tmux target (send-keys -l).
pub async fn send_text(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
    text: &str,
) -> Result<()> {
    let prefix = tmux_prefix(socket);
    let cmd = format!(
        "{} send-keys -l -t {} {}",
        prefix,
        shell_escape(target),
        shell_escape(text)
    );
    transport.exec(&cmd).await?;
    Ok(())
}

/// Rename a tmux session.
pub async fn rename_session(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    current_name: &str,
    new_name: &str,
) -> Result<()> {
    let prefix = tmux_prefix(socket);
    let cmd = format!(
        "{} rename-session -t {} {}",
        prefix,
        shell_escape(current_name),
        shell_escape(new_name)
    );
    transport.exec(&cmd).await?;
    Ok(())
}

/// Rename a tmux window.
pub async fn rename_window(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    session: &str,
    window_index: u32,
    new_name: &str,
) -> Result<()> {
    let prefix = tmux_prefix(socket);
    let target = format!("{}:{}", session, window_index);
    let cmd = format!(
        "{} rename-window -t {} {}",
        prefix,
        shell_escape(&target),
        shell_escape(new_name)
    );
    transport.exec(&cmd).await?;
    Ok(())
}

/// Set `history-limit` for a tmux session or globally (DC20, Phase 1.9b).
///
/// **Important**: `history-limit` only applies to panes created *after* this
/// call. Existing panes retain their creation-time limit. For automation,
/// call this before creating the session/window.
///
/// - `target`: session name for session-level, or `None` for global server-level.
pub async fn set_history_limit(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: Option<&str>,
    limit: u32,
) -> Result<()> {
    let prefix = tmux_prefix(socket);
    let cmd = match target {
        Some(t) => format!(
            "{} set-option -t {} history-limit {}",
            prefix,
            shell_escape(t),
            limit
        ),
        None => format!("{} set-option -g history-limit {}", prefix, limit),
    };
    transport.exec(&cmd).await?;
    Ok(())
}

/// Query the current `history-limit` for a session (or global default).
pub async fn get_history_limit(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: Option<&str>,
) -> Result<u32> {
    let prefix = tmux_prefix(socket);
    let cmd = match target {
        Some(t) => format!(
            "{} show-option -v -t {} history-limit",
            prefix,
            shell_escape(t)
        ),
        None => format!("{} show-option -gv history-limit", prefix),
    };
    let output = transport.exec(&cmd).await?;
    output
        .trim()
        .parse::<u32>()
        .map_err(|e| anyhow::anyhow!("failed to parse history-limit '{}': {}", output.trim(), e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::MockTransport;

    #[test]
    fn shell_escape_simple_string() {
        assert_eq!(shell_escape("hello"), "'hello'");
    }

    #[test]
    fn shell_escape_with_quotes() {
        assert_eq!(shell_escape("it's"), "'it'\\''s'");
    }

    #[test]
    fn shell_escape_injection_semicolon() {
        let input = "; rm -rf /";
        let escaped = shell_escape(input);
        assert_eq!(escaped, "'; rm -rf /'");
        // The entire string is inside single quotes — shell will not interpret ;
    }

    #[test]
    fn shell_escape_injection_backtick() {
        let input = "`whoami`";
        let escaped = shell_escape(input);
        assert_eq!(escaped, "'`whoami`'");
    }

    #[test]
    fn shell_escape_injection_dollar() {
        let input = "$(rm -rf /)";
        let escaped = shell_escape(input);
        assert_eq!(escaped, "'$(rm -rf /)'");
    }

    #[test]
    fn shell_escape_newline() {
        let input = "line1\nline2";
        let escaped = shell_escape(input);
        assert_eq!(escaped, "'line1\nline2'");
    }

    #[test]
    fn shell_escape_double_quotes() {
        let input = r#"say "hello""#;
        let escaped = shell_escape(input);
        assert_eq!(escaped, r#"'say "hello"'"#);
    }

    #[tokio::test]
    async fn create_session_basic() {
        let mock = MockTransport::new().with_default("");
        let transport = TransportKind::Mock(mock);
        create_session(&transport, None, "test", None, None)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn create_session_with_window_and_command() {
        let mock = MockTransport::new().with_default("");
        let transport = TransportKind::Mock(mock);
        create_session(&transport, None, "test", Some("main"), Some("vim"))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn kill_session_basic() {
        let mock = MockTransport::new().with_default("");
        let transport = TransportKind::Mock(mock);
        kill_session(&transport, None, "test").await.unwrap();
    }

    #[tokio::test]
    async fn rename_session_basic() {
        let mock = MockTransport::new().with_default("");
        let transport = TransportKind::Mock(mock);
        rename_session(&transport, None, "old", "new")
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn rename_window_basic() {
        let mock = MockTransport::new().with_default("");
        let transport = TransportKind::Mock(mock);
        rename_window(&transport, None, "build", 0, "editor")
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn send_text_basic() {
        let mock = MockTransport::new().with_default("");
        let transport = TransportKind::Mock(mock);
        send_text(&transport, None, "build:0.0", "echo hello")
            .await
            .unwrap();
    }

    #[test]
    fn shell_escape_null_byte() {
        let input = "before\0after";
        let escaped = shell_escape(input);
        // Null byte is inside single quotes — shell will not interpret it as terminator
        assert_eq!(escaped, "'before\0after'");
    }

    #[tokio::test]
    async fn set_history_limit_session() {
        let mock = MockTransport::new().with_default("");
        let transport = TransportKind::Mock(mock);
        set_history_limit(&transport, None, Some("build"), 50000)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn set_history_limit_global() {
        let mock = MockTransport::new().with_default("");
        let transport = TransportKind::Mock(mock);
        set_history_limit(&transport, None, None, 10000)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn get_history_limit_parses() {
        let mock = MockTransport::new()
            .with_response("show-option", "2000\n");
        let transport = TransportKind::Mock(mock);
        let limit = get_history_limit(&transport, None, None).await.unwrap();
        assert_eq!(limit, 2000);
    }
}
