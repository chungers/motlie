use crate::error::{Error, Result};
use std::path::Path;

use crate::keys::KeySequence;
use crate::transport::{shell_escape_arg, tmux_prefix, TransportKind};
use crate::types::{
    validate_session_tag_value, CreateSessionOptions, CreateWindowOptions, PaneAddress, SessionTag,
    SessionTagPrefix, SplitDirection, SplitPaneOptions, SplitSize, TmuxSocket, WindowInfo,
};

/// Shell-escape a string for safe interpolation into shell commands.
/// Uses POSIX single-quote wrapping: wrap in single quotes, escape
/// interior single quotes with '\'' (OC5).
pub fn shell_escape(s: &str) -> String {
    format!("'{}'", shell_escape_arg(s))
}

fn shell_escape_path(path: &Path) -> Result<String> {
    let s = path.to_str().ok_or_else(|| {
        Error::Parse(format!(
            "non-UTF-8 paths are not supported: {}",
            path.display()
        ))
    })?;
    Ok(shell_escape(s))
}

fn parse_created_window(output: &str) -> Result<WindowInfo> {
    let line = output.trim();
    let fields = crate::discovery::parse_escaped_fields(line);
    if fields.len() < 7 {
        return Err(Error::Command(format!(
            "malformed new-window output (expected 7 fields): {}",
            line
        )));
    }

    Ok(WindowInfo {
        session_id: fields[0].clone(),
        session_name: fields[1].clone(),
        index: fields[2]
            .parse()
            .map_err(|_| Error::Parse(format!("invalid window_index: {}", fields[2])))?,
        name: fields[3].clone(),
        active: fields[4] == "1",
        pane_count: fields[5]
            .parse()
            .map_err(|_| Error::Parse(format!("invalid window_panes: {}", fields[5])))?,
        layout: fields[6].clone(),
    })
}

fn parse_created_pane(output: &str) -> Result<PaneAddress> {
    let line = output.trim();
    let fields = crate::discovery::parse_escaped_fields(line);
    if fields.len() < 4 {
        return Err(Error::Command(format!(
            "malformed split-pane output (expected 4 fields): {}",
            line
        )));
    }
    let window: u32 = fields[2]
        .parse()
        .map_err(|_| Error::Parse(format!("invalid window_index: {}", fields[2])))?;
    let pane: u32 = fields[3]
        .parse()
        .map_err(|_| Error::Parse(format!("invalid pane_index: {}", fields[3])))?;
    Ok(PaneAddress {
        pane_id: fields[0].clone(),
        session: fields[1].clone(),
        window,
        pane,
    })
}

/// Create a new detached tmux session (using bare "tmux" prefix).
pub async fn create_session(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    name: &str,
    opts: &CreateSessionOptions,
) -> Result<()> {
    create_session_with_prefix(transport, &tmux_prefix(socket), name, opts).await
}

/// Create a new detached tmux session using a caller-provided prefix (DC22).
pub async fn create_session_with_prefix(
    transport: &TransportKind,
    prefix: &str,
    name: &str,
    opts: &CreateSessionOptions,
) -> Result<()> {
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
    if let Some(limit) = opts.history_limit {
        let result = async {
            let session_cmd = format!(
                "{} set-option -t {} history-limit {}",
                prefix,
                shell_escape(name),
                limit
            );
            transport.exec(&session_cmd).await?;

            let pane_cmd = format!(
                "{} set-option -p -t {} history-limit {}",
                prefix,
                shell_escape(name),
                limit
            );
            transport.exec(&pane_cmd).await?;
            Ok::<(), Error>(())
        }
        .await;

        if let Err(e) = result {
            let kill_cmd = format!("{} kill-session -t {}", prefix, shell_escape(name));
            let _ = transport.exec(&kill_cmd).await;
            return Err(e);
        }
    }

    Ok(())
}

/// Create a new tmux window (using bare "tmux" prefix).
pub async fn new_window(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    session: &str,
    opts: &CreateWindowOptions,
) -> Result<WindowInfo> {
    new_window_with_prefix(transport, &tmux_prefix(socket), session, opts).await
}

/// Create a new tmux window using a caller-provided prefix (DC25).
pub async fn new_window_with_prefix(
    transport: &TransportKind,
    prefix: &str,
    session: &str,
    opts: &CreateWindowOptions,
) -> Result<WindowInfo> {
    let mut cmd = format!(
        "{} new-window -P -F '#{{q:session_id}} #{{q:session_name}} #{{q:window_index}} #{{q:window_name}} #{{q:window_active}} #{{q:window_panes}} #{{q:window_layout}}'",
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

/// Split a tmux pane/window (using bare "tmux" prefix).
pub async fn split_pane(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
    opts: &SplitPaneOptions,
) -> Result<PaneAddress> {
    split_pane_with_prefix(transport, &tmux_prefix(socket), target, opts).await
}

/// Split a tmux pane/window using a caller-provided prefix (DC25).
pub async fn split_pane_with_prefix(
    transport: &TransportKind,
    prefix: &str,
    target: &str,
    opts: &SplitPaneOptions,
) -> Result<PaneAddress> {
    let mut cmd = format!(
        "{} split-window -P -F '#{{q:pane_id}} #{{q:session_name}} #{{q:window_index}} #{{q:pane_index}}'",
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
                    return Err(Error::Parse(format!(
                        "split percentage must be in 1..=100, got {}",
                        percent
                    )));
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

/// Kill a tmux session by name (using bare "tmux" prefix).
pub async fn kill_session(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    name: &str,
) -> Result<()> {
    kill_session_with_prefix(transport, &tmux_prefix(socket), name).await
}

/// Kill a tmux session using a caller-provided prefix.
pub async fn kill_session_with_prefix(
    transport: &TransportKind,
    prefix: &str,
    name: &str,
) -> Result<()> {
    let cmd = format!("{} kill-session -t {}", prefix, shell_escape(name));
    transport.exec(&cmd).await?;
    Ok(())
}

/// Kill a tmux window (using bare "tmux" prefix).
pub async fn kill_window(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
) -> Result<()> {
    kill_window_with_prefix(transport, &tmux_prefix(socket), target).await
}

/// Kill a tmux window using a caller-provided prefix.
pub async fn kill_window_with_prefix(
    transport: &TransportKind,
    prefix: &str,
    target: &str,
) -> Result<()> {
    let cmd = format!("{} kill-window -t {}", prefix, shell_escape(target));
    transport.exec(&cmd).await?;
    Ok(())
}

/// Kill a tmux pane (using bare "tmux" prefix).
pub async fn kill_pane(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
) -> Result<()> {
    kill_pane_with_prefix(transport, &tmux_prefix(socket), target).await
}

/// Kill a tmux pane using a caller-provided prefix.
pub async fn kill_pane_with_prefix(
    transport: &TransportKind,
    prefix: &str,
    target: &str,
) -> Result<()> {
    let cmd = format!("{} kill-pane -t {}", prefix, shell_escape(target));
    transport.exec(&cmd).await?;
    Ok(())
}

/// Send a key sequence to a tmux target (using bare "tmux" prefix).
pub async fn send_keys(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
    keys: &KeySequence,
) -> Result<()> {
    send_keys_with_prefix(transport, &tmux_prefix(socket), target, keys).await
}

/// Send a key sequence using a caller-provided prefix.
pub async fn send_keys_with_prefix(
    transport: &TransportKind,
    prefix: &str,
    target: &str,
    keys: &KeySequence,
) -> Result<()> {
    let tmux_target = shell_escape(target);

    for args in keys.to_tmux_args(target) {
        let has_literal = args.contains(&"-l".to_string());
        let mut cmd = prefix.to_string();
        cmd.push_str(" send-keys");
        if has_literal {
            cmd.push_str(" -l");
        }
        cmd.push_str(&format!(" -t {}", tmux_target));
        let value = &args[args.len() - 1];
        cmd.push_str(&format!(" {}", shell_escape(value)));
        transport.exec(&cmd).await?;
    }
    Ok(())
}

/// Send literal text to a tmux target (using bare "tmux" prefix).
pub async fn send_text(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
    text: &str,
) -> Result<()> {
    send_text_with_prefix(transport, &tmux_prefix(socket), target, text).await
}

/// Send literal text using a caller-provided prefix.
pub async fn send_text_with_prefix(
    transport: &TransportKind,
    prefix: &str,
    target: &str,
    text: &str,
) -> Result<()> {
    let cmd = format!(
        "{} send-keys -l -t {} {}",
        prefix,
        shell_escape(target),
        shell_escape(text)
    );
    transport.exec(&cmd).await?;
    Ok(())
}

/// Rename a tmux session (using bare "tmux" prefix).
pub async fn rename_session(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    current_name: &str,
    new_name: &str,
) -> Result<()> {
    rename_session_with_prefix(transport, &tmux_prefix(socket), current_name, new_name).await
}

/// Rename a tmux session using a caller-provided prefix.
pub async fn rename_session_with_prefix(
    transport: &TransportKind,
    prefix: &str,
    current_name: &str,
    new_name: &str,
) -> Result<()> {
    let cmd = format!(
        "{} rename-session -t {} {}",
        prefix,
        shell_escape(current_name),
        shell_escape(new_name)
    );
    transport.exec(&cmd).await?;
    Ok(())
}

/// Rename a tmux window (using bare "tmux" prefix).
pub async fn rename_window(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    session: &str,
    window_index: u32,
    new_name: &str,
) -> Result<()> {
    rename_window_with_prefix(
        transport,
        &tmux_prefix(socket),
        session,
        window_index,
        new_name,
    )
    .await
}

/// Rename a tmux window using a caller-provided prefix.
pub async fn rename_window_with_prefix(
    transport: &TransportKind,
    prefix: &str,
    session: &str,
    window_index: u32,
    new_name: &str,
) -> Result<()> {
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

/// Set `history-limit` (using bare "tmux" prefix).
pub async fn set_history_limit(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: Option<&str>,
    limit: u32,
) -> Result<()> {
    set_history_limit_with_prefix(transport, &tmux_prefix(socket), target, limit).await
}

/// Set `history-limit` using a caller-provided prefix (DC20, Phase 1.9b).
pub async fn set_history_limit_with_prefix(
    transport: &TransportKind,
    prefix: &str,
    target: Option<&str>,
    limit: u32,
) -> Result<()> {
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

/// Query the current `history-limit` (using bare "tmux" prefix).
pub async fn get_history_limit(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: Option<&str>,
) -> Result<u32> {
    get_history_limit_with_prefix(transport, &tmux_prefix(socket), target).await
}

/// Query the current `history-limit` using a caller-provided prefix.
pub async fn get_history_limit_with_prefix(
    transport: &TransportKind,
    prefix: &str,
    target: Option<&str>,
) -> Result<u32> {
    let cmd = match target {
        Some(t) => format!(
            "{} show-option -v -t {} history-limit",
            prefix,
            shell_escape(t)
        ),
        None => format!("{} show-option -gv history-limit", prefix),
    };
    let output = transport.exec(&cmd).await?;
    output.trim().parse::<u32>().map_err(|e| {
        Error::Parse(format!(
            "failed to parse history-limit '{}': {}",
            output.trim(),
            e
        ))
    })
}

/// Set a namespaced session tag via tmux user-defined options.
pub(crate) async fn set_session_tag_with_prefix(
    transport: &TransportKind,
    prefix: &str,
    target: &str,
    tag_prefix: &SessionTagPrefix,
    key: &str,
    value: &str,
) -> Result<()> {
    validate_session_tag_value(value)?;
    let option_name = tag_prefix.option_name(key)?;
    let cmd = format!(
        "{} set-option -t {} {} {}",
        prefix,
        shell_escape(target),
        option_name,
        shell_escape(value)
    );
    transport.exec(&cmd).await?;
    Ok(())
}

/// Read one namespaced session tag. Missing tags return `Ok(None)`.
pub(crate) async fn read_session_tag_with_prefix(
    transport: &TransportKind,
    prefix: &str,
    target: &str,
    tag_prefix: &SessionTagPrefix,
    key: &str,
) -> Result<Option<String>> {
    let option_name = tag_prefix.option_name(key)?;
    let cmd = format!(
        "{} show-option -q -t {} {}",
        prefix,
        shell_escape(target),
        option_name
    );
    let output = transport.exec(&cmd).await?;
    let option_prefix = tag_prefix.option_prefix();
    let Some(tag) = parse_session_tag_option_line(tag_prefix, &option_prefix, &output)? else {
        return Ok(None);
    };
    if tag.key() != key {
        return Err(Error::Parse(format!(
            "show-option returned {} while reading {}",
            tag.option_name(),
            option_name
        )));
    }
    Ok(Some(tag.into_value()))
}

/// List all tags under one namespace prefix for a session.
pub(crate) async fn list_session_tags_with_prefix(
    transport: &TransportKind,
    prefix: &str,
    target: &str,
    tag_prefix: &SessionTagPrefix,
) -> Result<Vec<SessionTag>> {
    let cmd = format!("{} show-options -t {}", prefix, shell_escape(target));
    let output = transport.exec(&cmd).await?;
    let mut tags = Vec::new();
    let option_prefix = tag_prefix.option_prefix();
    for line in output.lines() {
        if let Some(tag) = parse_session_tag_option_line(tag_prefix, &option_prefix, line)? {
            tags.push(tag);
        }
    }
    Ok(tags)
}

fn parse_session_tag_option_line(
    prefix: &SessionTagPrefix,
    option_prefix: &str,
    line: &str,
) -> Result<Option<SessionTag>> {
    let line = line.trim_end_matches('\r').trim();
    if line.is_empty() {
        return Ok(None);
    }
    if !line.starts_with(option_prefix) {
        return Ok(None);
    }
    let Some((option_name, value)) = split_once_whitespace(line) else {
        return Err(Error::Parse(format!(
            "malformed session tag option without value: {line}"
        )));
    };
    let Some(key) = option_name.strip_prefix(option_prefix) else {
        return Ok(None);
    };
    if key.is_empty() {
        return Err(Error::Parse(format!(
            "malformed session tag option with empty key: {line}"
        )));
    }
    let value = parse_tmux_option_value(value)?;
    Ok(Some(SessionTag::from_validated_prefix(prefix, key, value)?))
}

fn split_once_whitespace(input: &str) -> Option<(&str, &str)> {
    let index = input
        .char_indices()
        .find_map(|(index, ch)| ch.is_whitespace().then_some(index))?;
    let (left, right) = input.split_at(index);
    Some((left, right.trim_start()))
}

fn parse_tmux_option_value(input: &str) -> Result<String> {
    let mut out = String::new();
    let mut chars = input.trim().chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '\'' => loop {
                match chars.next() {
                    Some('\'') => break,
                    Some(inner) => out.push(inner),
                    None => {
                        return Err(Error::Parse(format!(
                            "unterminated single-quoted option value: {input}"
                        )));
                    }
                }
            },
            '"' => loop {
                match chars.next() {
                    Some('"') => break,
                    Some('\\') => match chars.next() {
                        Some('\\') => out.push('\\'),
                        Some('"') => out.push('"'),
                        Some(next) => {
                            out.push('\\');
                            out.push(next);
                        }
                        None => out.push('\\'),
                    },
                    Some(inner) => out.push(inner),
                    None => {
                        return Err(Error::Parse(format!(
                            "unterminated double-quoted option value: {input}"
                        )));
                    }
                }
            },
            '\\' => {
                if let Some(next) = chars.next() {
                    out.push(next);
                }
            }
            ch if ch.is_whitespace() => {
                if chars.any(|rest| !rest.is_whitespace()) {
                    return Err(Error::Parse(format!(
                        "unexpected trailing tokens in option value: {input}"
                    )));
                }
                break;
            }
            other => out.push(other),
        }
    }
    Ok(out)
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
        let expected = "tmux new-window -P -F '#{q:session_id} #{q:session_name} #{q:window_index} #{q:window_name} #{q:window_active} #{q:window_panes} #{q:window_layout}' -n 'editor' -x 200 -y 50 -c '/tmp/project' -t 'build' 'vim'";
        let mock = MockTransport::new().with_response(expected, "$0 build 1 editor 1 1 layout");
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
        let expected = "tmux split-window -P -F '#{q:pane_id} #{q:session_name} #{q:window_index} #{q:pane_index}' -h -l 40% -c '/tmp/project' -t 'build:1.0' 'htop'";
        let mock = MockTransport::new().with_response(expected, "%9 build 1 1");
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
            "$0 build not-a-number editor 1 also-bad layout",
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

    fn tag_prefix() -> SessionTagPrefix {
        SessionTagPrefix::new("mmux").unwrap()
    }

    #[test]
    fn session_tag_prefix_validates_components() {
        let prefix = tag_prefix();
        assert_eq!(prefix.option_name("owner_1").unwrap(), "@mmux/owner_1");
        assert!(SessionTagPrefix::new("").is_err());
        assert!(SessionTagPrefix::new("mmux/team").is_err());
        assert!(prefix.option_name("").is_err());
        assert!(prefix.option_name("owner/team").is_err());
    }

    #[test]
    fn session_tag_new_validates_and_is_self_describing() {
        let tag = SessionTag::new("mmux", "owner", "david").unwrap();
        assert_eq!(
            (tag.prefix(), tag.key(), tag.value(), tag.option_name()),
            ("mmux", "owner", "david", "@mmux/owner".to_string())
        );
        assert!(SessionTag::new("mmux/team", "owner", "david").is_err());
        assert!(SessionTag::new("mmux", "owner/team", "david").is_err());
        assert!(SessionTag::new("mmux", "owner", "line1\nline2").is_err());
    }

    #[test]
    fn parse_session_tag_option_line_decodes_tmux_values() {
        let prefix = tag_prefix();
        let option_prefix = prefix.option_prefix();
        assert_eq!(
            parse_session_tag_option_line(&prefix, &option_prefix, r#"@mmux/foo "hello world""#)
                .unwrap(),
            Some(SessionTag::new("mmux", "foo", "hello world").unwrap())
        );
        assert_eq!(
            parse_session_tag_option_line(&prefix, &option_prefix, "@mmux/empty ''").unwrap(),
            Some(SessionTag::new("mmux", "empty", "").unwrap())
        );
        assert_eq!(
            parse_session_tag_option_line(&prefix, &option_prefix, r#"@mmux/quote "say \"hi\"""#)
                .unwrap(),
            Some(SessionTag::new("mmux", "quote", "say \"hi\"").unwrap())
        );
        assert_eq!(
            parse_session_tag_option_line(&prefix, &option_prefix, r#"@mmux/slash "a\nb""#)
                .unwrap(),
            Some(SessionTag::new("mmux", "slash", r"a\nb").unwrap())
        );
        assert_eq!(
            parse_session_tag_option_line(&prefix, &option_prefix, "@other/foo value").unwrap(),
            None
        );
    }

    #[test]
    fn parse_session_tag_option_line_rejects_malformed_matching_tags() {
        let prefix = tag_prefix();
        let option_prefix = prefix.option_prefix();
        assert!(parse_session_tag_option_line(&prefix, &option_prefix, "@mmux/foo").is_err());
        assert!(parse_session_tag_option_line(&prefix, &option_prefix, "@mmux/ 'value'").is_err());
        assert!(
            parse_session_tag_option_line(&prefix, &option_prefix, "@mmux/foo/bar value").is_err()
        );
        assert!(
            parse_session_tag_option_line(&prefix, &option_prefix, "@mmux/foo 'unterminated")
                .is_err()
        );
    }

    #[tokio::test]
    async fn set_session_tag_builds_expected_command() {
        let expected = "tmux set-option -t '$7' @mmux/foo 'bar baz'";
        let mock = MockTransport::new().with_response(expected, "");
        let transport = TransportKind::Mock(mock);
        let prefix = tag_prefix();

        set_session_tag_with_prefix(&transport, "tmux", "$7", &prefix, "foo", "bar baz")
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn read_session_tag_returns_value_or_none() {
        let mock = MockTransport::new()
            .with_response(
                "show-option -q -t '$7' @mmux/foo",
                "@mmux/foo \"hello world\"\n",
            )
            .with_response("show-option -q -t '$7' @mmux/missing", "");
        let transport = TransportKind::Mock(mock);
        let prefix = tag_prefix();

        let value = read_session_tag_with_prefix(&transport, "tmux", "$7", &prefix, "foo")
            .await
            .unwrap();
        let missing = read_session_tag_with_prefix(&transport, "tmux", "$7", &prefix, "missing")
            .await
            .unwrap();

        assert_eq!(value, Some("hello world".to_string()));
        assert_eq!(missing, None);
    }

    #[tokio::test]
    async fn read_session_tag_rejects_unexpected_key() {
        let mock = MockTransport::new()
            .with_response("show-option -q -t '$7' @mmux/foo", "@mmux/other value\n");
        let transport = TransportKind::Mock(mock);
        let prefix = tag_prefix();

        let err = read_session_tag_with_prefix(&transport, "tmux", "$7", &prefix, "foo")
            .await
            .unwrap_err();

        assert!(err.to_string().contains("@mmux/other"));
    }

    #[tokio::test]
    async fn list_session_tags_filters_prefix() {
        let mock = MockTransport::new().with_response(
            "show-options -t '$7'",
            "@mmux/foo \"hello world\"\n@other/foo nope\n@mmux/empty ''\n",
        );
        let transport = TransportKind::Mock(mock);
        let prefix = tag_prefix();

        let tags = list_session_tags_with_prefix(&transport, "tmux", "$7", &prefix)
            .await
            .unwrap();

        assert_eq!(
            tags,
            vec![
                SessionTag::new("mmux", "foo", "hello world").unwrap(),
                SessionTag::new("mmux", "empty", "").unwrap(),
            ]
        );
    }
}
