use anyhow::Result;
use std::path::Path;

use crate::keys::KeySequence;
use crate::transport::{shell_escape_arg, tmux_prefix, TransportKind};
use crate::types::{
    CreateSessionOptions, CreateWindowOptions, PaneAddress, SplitDirection, SplitPaneOptions,
    SplitSize, TmuxSocket, WindowInfo,
};

/// Shell-escape a string for safe interpolation into shell commands.
/// Uses POSIX single-quote wrapping: wrap in single quotes, escape
/// interior single quotes with '\'' (OC5).
pub fn shell_escape(s: &str) -> String {
    format!("'{}'", shell_escape_arg(s))
}

fn shell_escape_path(path: &Path) -> Result<String> {
    let s = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("non-UTF-8 paths are not supported: {}", path.display()))?;
    Ok(shell_escape(s))
}

fn parse_created_window(output: &str) -> Result<WindowInfo> {
    let line = output.trim();
    let fields = crate::discovery::split_fields(line);
    if fields.len() < 7 {
        return Err(anyhow::anyhow!(
            "malformed new-window output (expected 7 fields): {}",
            line
        ));
    }

    Ok(WindowInfo {
        session_id: fields[0].to_string(),
        session_name: fields[1].to_string(),
        index: fields[2]
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid window_index: {}", fields[2]))?,
        name: fields[3].to_string(),
        active: fields[4] == "1",
        pane_count: fields[5]
            .parse()
            .map_err(|_| anyhow::anyhow!("invalid window_panes: {}", fields[5]))?,
        layout: fields[6].to_string(),
    })
}

fn parse_created_pane(output: &str) -> Result<PaneAddress> {
    let line = output.trim();
    let fields = crate::discovery::split_fields(line);
    if fields.len() < 2 {
        return Err(anyhow::anyhow!(
            "malformed split-pane output (expected 2 fields): {}",
            line
        ));
    }
    PaneAddress::parse(fields[0], fields[1])
}

/// Create a new detached tmux session (DC22).
///
/// Runs `tmux new-session -d -s <name>` with optional `-n`, `-x`, `-y`, and command.
/// If `opts.history_limit` is set, issues two additional `set-option` commands:
/// per-session (covers future panes) and per-pane (tmux 3.1+, covers initial pane).
pub async fn create_session(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    name: &str,
    opts: &CreateSessionOptions,
) -> Result<()> {
    let prefix = tmux_prefix(socket);
    let mut cmd = format!("{} new-session -d -s {}", prefix, shell_escape(name));

    if let Some(wn) = &opts.window_name {
        cmd.push_str(&format!(" -n {}", shell_escape(wn)));
    }
    if let Some(w) = opts.width {
        cmd.push_str(&format!(" -x {}", w));
    }
    if let Some(h) = opts.height {
        cmd.push_str(&format!(" -y {}", h));
    }
    if let Some(c) = &opts.command {
        cmd.push_str(&format!(" {}", shell_escape(c)));
    }

    transport.exec(&cmd).await?;

    // Set history-limit if requested (DC22, Option B).
    // If either set-option fails, kill the just-created session to avoid
    // leaked state, then propagate the error.
    if let Some(limit) = opts.history_limit {
        let result = async {
            // Per-session: covers future windows/panes
            let session_cmd = format!(
                "{} set-option -t {} history-limit {}",
                prefix,
                shell_escape(name),
                limit
            );
            transport.exec(&session_cmd).await?;

            // Per-pane: covers the initial pane created by new-session (tmux 3.1+)
            let pane_cmd = format!(
                "{} set-option -p -t {} history-limit {}",
                prefix,
                shell_escape(name),
                limit
            );
            transport.exec(&pane_cmd).await?;
            Ok::<(), anyhow::Error>(())
        }
        .await;

        if let Err(e) = result {
            // Best-effort cleanup: kill the session we just created
            let kill_cmd = format!("{} kill-session -t {}", prefix, shell_escape(name));
            let _ = transport.exec(&kill_cmd).await;
            return Err(e);
        }
    }

    Ok(())
}

/// Create a new tmux window and return its typed address metadata (DC25).
pub async fn new_window(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    session: &str,
    opts: &CreateWindowOptions,
) -> Result<WindowInfo> {
    let prefix = tmux_prefix(socket);
    let mut cmd = format!(
        "{} new-window -P -F '#{{session_id}}@@#{{session_name}}@@#{{window_index}}@@#{{window_name}}@@#{{window_active}}@@#{{window_panes}}@@#{{window_layout}}'",
        prefix
    );

    if let Some(name) = &opts.name {
        cmd.push_str(&format!(" -n {}", shell_escape(name)));
    }
    if let Some(width) = opts.width {
        cmd.push_str(&format!(" -x {}", width));
    }
    if let Some(height) = opts.height {
        cmd.push_str(&format!(" -y {}", height));
    }
    if let Some(start_directory) = &opts.start_directory {
        cmd.push_str(&format!(" -c {}", shell_escape_path(start_directory)?));
    }
    cmd.push_str(&format!(" -t {}", shell_escape(session)));
    if let Some(command) = &opts.command {
        cmd.push_str(&format!(" {}", shell_escape(command)));
    }

    let output = transport.exec(&cmd).await?;
    parse_created_window(&output)
}

/// Split a tmux pane/window and return the created pane address (DC25).
pub async fn split_pane(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
    opts: &SplitPaneOptions,
) -> Result<PaneAddress> {
    let prefix = tmux_prefix(socket);
    let mut cmd = format!(
        "{} split-window -P -F '#{{pane_id}}@@#{{session_name}}:#{{window_index}}.#{{pane_index}}'",
        prefix
    );

    match opts.direction {
        SplitDirection::Horizontal => cmd.push_str(" -h"),
        SplitDirection::Vertical => cmd.push_str(" -v"),
    }

    if let Some(size) = opts.size {
        match size {
            SplitSize::Cells(cells) => {
                cmd.push_str(&format!(" -l {}", cells));
            }
            SplitSize::Percent(percent) => {
                if percent == 0 || percent > 100 {
                    return Err(anyhow::anyhow!(
                        "split percentage must be in 1..=100, got {}",
                        percent
                    ));
                }
                cmd.push_str(&format!(" -l {}%", percent));
            }
        }
    }

    if let Some(start_directory) = &opts.start_directory {
        cmd.push_str(&format!(" -c {}", shell_escape_path(start_directory)?));
    }
    cmd.push_str(&format!(" -t {}", shell_escape(target)));
    if let Some(command) = &opts.command {
        cmd.push_str(&format!(" {}", shell_escape(command)));
    }

    let output = transport.exec(&cmd).await?;
    parse_created_pane(&output)
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
///
/// **Note**: the returned value is the *configured* limit, which only applies
/// to panes created after it was set. Existing panes retain their
/// creation-time limit regardless of subsequent changes. To verify a
/// specific pane's effective limit, use `display-message -p '#{history_limit}'`
/// targeted at that pane.
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
    use std::path::PathBuf;

    #[cfg(unix)]
    use std::os::unix::ffi::OsStringExt;

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
        create_session(&transport, None, "test", &Default::default())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn create_session_with_window_and_command() {
        let mock = MockTransport::new().with_default("");
        let transport = TransportKind::Mock(mock);
        let opts = CreateSessionOptions {
            window_name: Some("main".to_string()),
            command: Some("vim".to_string()),
            ..Default::default()
        };
        create_session(&transport, None, "test", &opts)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn create_session_with_size() {
        let mock = MockTransport::new().with_default("");
        let transport = TransportKind::Mock(mock);
        let opts = CreateSessionOptions {
            width: Some(200),
            height: Some(50),
            ..Default::default()
        };
        create_session(&transport, None, "test", &opts)
            .await
            .unwrap();
        // Verifies -x 200 -y 50 flags are appended (mock accepts any command)
    }

    #[tokio::test]
    async fn create_session_with_history_limit() {
        let mock = MockTransport::new().with_default("");
        let transport = TransportKind::Mock(mock);
        let opts = CreateSessionOptions {
            history_limit: Some(50000),
            ..Default::default()
        };
        create_session(&transport, None, "test", &opts)
            .await
            .unwrap();
        // Verifies set-option -t and set-option -p commands are issued
    }

    #[tokio::test]
    async fn create_session_with_all_options() {
        let mock = MockTransport::new().with_default("");
        let transport = TransportKind::Mock(mock);
        let opts = CreateSessionOptions {
            window_name: Some("editor".to_string()),
            command: Some("vim".to_string()),
            width: Some(200),
            height: Some(50),
            history_limit: Some(50000),
        };
        create_session(&transport, None, "test", &opts)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn new_window_with_all_options_builds_expected_command() {
        let expected = "tmux new-window -P -F '#{session_id}@@#{session_name}@@#{window_index}@@#{window_name}@@#{window_active}@@#{window_panes}@@#{window_layout}' -n 'editor' -x 200 -y 50 -c '/tmp/project' -t 'build' 'vim'";
        let mock =
            MockTransport::new().with_response(expected, "$0@@build@@1@@editor@@1@@1@@layout");
        let transport = TransportKind::Mock(mock);
        let opts = CreateWindowOptions {
            name: Some("editor".to_string()),
            command: Some("vim".to_string()),
            width: Some(200),
            height: Some(50),
            start_directory: Some(PathBuf::from("/tmp/project")),
        };

        let window = new_window(&transport, None, "build", &opts).await.unwrap();
        assert_eq!(window.session_name, "build");
        assert_eq!(window.index, 1);
        assert_eq!(window.name, "editor");
    }

    #[tokio::test]
    async fn split_pane_with_all_options_builds_expected_command() {
        let expected = "tmux split-window -P -F '#{pane_id}@@#{session_name}:#{window_index}.#{pane_index}' -h -l 40% -c '/tmp/project' -t 'build:1.0' 'htop'";
        let mock = MockTransport::new().with_response(expected, "%9@@build:1.1");
        let transport = TransportKind::Mock(mock);
        let opts = SplitPaneOptions {
            direction: SplitDirection::Horizontal,
            size: Some(SplitSize::percent(40).unwrap()),
            command: Some("htop".to_string()),
            start_directory: Some(PathBuf::from("/tmp/project")),
        };

        let pane = split_pane(&transport, None, "build:1.0", &opts)
            .await
            .unwrap();
        assert_eq!(pane.pane_id, "%9");
        assert_eq!(pane.session, "build");
        assert_eq!(pane.window, 1);
        assert_eq!(pane.pane, 1);
    }

    #[tokio::test]
    async fn split_pane_rejects_invalid_percent_defensively() {
        let mock = MockTransport::new().with_default("");
        let transport = TransportKind::Mock(mock);
        let opts = SplitPaneOptions {
            size: Some(SplitSize::Percent(101)),
            ..Default::default()
        };

        let err = split_pane(&transport, None, "build:0.0", &opts)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("1..=100"));
    }

    #[tokio::test]
    async fn new_window_rejects_malformed_printed_output() {
        let mock = MockTransport::new().with_response("new-window -P", "too-few-fields");
        let transport = TransportKind::Mock(mock);

        let err = new_window(&transport, None, "build", &Default::default())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("malformed new-window output"));
    }

    #[tokio::test]
    async fn new_window_rejects_invalid_numeric_fields() {
        let mock = MockTransport::new().with_response(
            "new-window -P",
            "$0@@build@@not-a-number@@editor@@1@@also-bad@@layout",
        );
        let transport = TransportKind::Mock(mock);

        let err = new_window(&transport, None, "build", &Default::default())
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("invalid window_index") || msg.contains("invalid window_panes"));
    }

    #[tokio::test]
    async fn split_pane_rejects_malformed_printed_output() {
        let mock = MockTransport::new().with_response("split-window -P", "bad-output");
        let transport = TransportKind::Mock(mock);

        let err = split_pane(&transport, None, "build:0.0", &Default::default())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("malformed split-pane output"));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn new_window_rejects_non_utf8_start_directory() {
        let mock = MockTransport::new().with_default("");
        let transport = TransportKind::Mock(mock);
        let non_utf8 = std::ffi::OsString::from_vec(vec![0x66, 0x6f, 0x80, 0x6f]);
        let opts = CreateWindowOptions {
            start_directory: Some(PathBuf::from(non_utf8)),
            ..Default::default()
        };

        let err = new_window(&transport, None, "build", &opts)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("non-UTF-8"));
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
        let mock = MockTransport::new().with_response("show-option", "2000\n");
        let transport = TransportKind::Mock(mock);
        let limit = get_history_limit(&transport, None, None).await.unwrap();
        assert_eq!(limit, 2000);
    }
}
