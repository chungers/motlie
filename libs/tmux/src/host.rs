use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::Duration;

use crate::capture;
use crate::control;
use crate::discovery;
use crate::keys::KeySequence;
use crate::transport::TransportKind;
use crate::types::*;

/// Internal state shared via Arc between HostHandle and Target.
struct HostHandleInner {
    transport: TransportKind,
    socket: Option<TmuxSocket>,
}

/// Handle to a tmux host (local or remote).
#[derive(Clone)]
pub struct HostHandle {
    inner: Arc<HostHandleInner>,
}

impl HostHandle {
    /// Create a HostHandle for localhost.
    pub fn local() -> Self {
        HostHandle {
            inner: Arc::new(HostHandleInner {
                transport: TransportKind::Local(
                    crate::transport::LocalTransport::new(),
                ),
                socket: None,
            }),
        }
    }

    /// Create a HostHandle with a specific transport and socket.
    pub fn new(transport: TransportKind, socket: Option<TmuxSocket>) -> Self {
        HostHandle {
            inner: Arc::new(HostHandleInner { transport, socket }),
        }
    }

    /// List all tmux sessions on this host.
    pub async fn list_sessions(&self) -> Result<Vec<SessionInfo>> {
        discovery::list_sessions(&self.inner.transport, self.inner.socket.as_ref()).await
    }

    /// Create a new tmux session. Returns a Target at session level.
    pub async fn create_session(
        &self,
        name: &str,
        window_name: Option<&str>,
        command: Option<&str>,
    ) -> Result<Target> {
        control::create_session(
            &self.inner.transport,
            self.inner.socket.as_ref(),
            name,
            window_name,
            command,
        )
        .await?;

        // Query the created session to get full info
        let sessions =
            discovery::list_sessions(&self.inner.transport, self.inner.socket.as_ref())
                .await?;
        let info = sessions
            .into_iter()
            .find(|s| s.name == name)
            .ok_or_else(|| anyhow!("session '{}' created but not found in list", name))?;

        Ok(Target {
            inner: self.inner.clone(),
            address: TargetAddress::Session(info),
            exec_mutex: Arc::new(Mutex::new(())),
        })
    }

    /// Get a Target for an existing session by name.
    pub async fn session(&self, name: &str) -> Result<Option<Target>> {
        let sessions =
            discovery::list_sessions(&self.inner.transport, self.inner.socket.as_ref())
                .await?;
        Ok(sessions.into_iter().find(|s| s.name == name).map(|info| {
            Target {
                inner: self.inner.clone(),
                address: TargetAddress::Session(info),
                exec_mutex: Arc::new(Mutex::new(())),
            }
        }))
    }

    /// Get a Target from a TargetSpec. Verifies the entity exists.
    pub async fn target(&self, spec: &TargetSpec) -> Result<Option<Target>> {
        // Check session exists
        let sessions =
            discovery::list_sessions(&self.inner.transport, self.inner.socket.as_ref())
                .await?;
        let session_info = match sessions.into_iter().find(|s| s.name == spec.session) {
            Some(s) => s,
            None => return Ok(None),
        };

        match (&spec.window, spec.pane) {
            (None, _) => Ok(Some(Target {
                inner: self.inner.clone(),
                address: TargetAddress::Session(session_info),
                exec_mutex: Arc::new(Mutex::new(())),
            })),
            (Some(window_str), None) => {
                let windows = discovery::list_windows(
                    &self.inner.transport,
                    self.inner.socket.as_ref(),
                    &spec.session,
                )
                .await?;
                let win = windows.into_iter().find(|w| {
                    w.index.to_string() == *window_str || w.name == *window_str
                });
                Ok(win.map(|w| Target {
                    inner: self.inner.clone(),
                    address: TargetAddress::Window(w),
                    exec_mutex: Arc::new(Mutex::new(())),
                }))
            }
            (Some(window_str), Some(pane_idx)) => {
                let panes = discovery::list_panes_in_session(
                    &self.inner.transport,
                    self.inner.socket.as_ref(),
                    &spec.session,
                )
                .await?;
                let window_index: Option<u32> = window_str.parse().ok();
                let pane = panes.into_iter().find(|p| {
                    let win_match = match window_index {
                        Some(idx) => p.address.window == idx,
                        None => false, // window by name not matched to pane directly
                    };
                    win_match && p.address.pane == pane_idx
                });
                Ok(pane.map(|p| Target {
                    inner: self.inner.clone(),
                    address: TargetAddress::Pane(p.address),
                    exec_mutex: Arc::new(Mutex::new(())),
                }))
            }
        }
    }
}

/// Unified target at any hierarchy level (DC16).
pub struct Target {
    inner: Arc<HostHandleInner>,
    address: TargetAddress,
    /// Per-target mutex for serializing exec() calls to the same pane (DC19).
    exec_mutex: Arc<Mutex<()>>,
}

impl Target {
    /// What level this target is at.
    pub fn level(&self) -> TargetLevel {
        match &self.address {
            TargetAddress::Session(_) => TargetLevel::Session,
            TargetAddress::Window(_) => TargetLevel::Window,
            TargetAddress::Pane(_) => TargetLevel::Pane,
        }
    }

    /// The tmux target string for commands.
    pub fn target_string(&self) -> String {
        match &self.address {
            TargetAddress::Session(s) => s.name.clone(),
            TargetAddress::Window(w) => format!("{}:{}", w.session_name, w.index),
            TargetAddress::Pane(p) => p.to_tmux_target(),
        }
    }

    /// Session info (available at any level).
    pub fn session_info(&self) -> Option<&SessionInfo> {
        match &self.address {
            TargetAddress::Session(s) => Some(s),
            _ => None,
        }
    }

    /// Session name (available at any level).
    pub fn session_name(&self) -> &str {
        match &self.address {
            TargetAddress::Session(s) => &s.name,
            TargetAddress::Window(w) => &w.session_name,
            TargetAddress::Pane(p) => &p.session,
        }
    }

    /// Window info (available at window and pane level).
    pub fn window_info(&self) -> Option<&WindowInfo> {
        match &self.address {
            TargetAddress::Window(w) => Some(w),
            _ => None,
        }
    }

    /// Pane address (available at pane level only).
    pub fn pane_address(&self) -> Option<&PaneAddress> {
        match &self.address {
            TargetAddress::Pane(p) => Some(p),
            _ => None,
        }
    }

    /// The underlying TargetAddress.
    pub fn address(&self) -> &TargetAddress {
        &self.address
    }

    // --- Navigation ---

    /// List children one level down.
    pub async fn children(&self) -> Result<Vec<Target>> {
        match &self.address {
            TargetAddress::Session(s) => {
                let windows = discovery::list_windows(
                    &self.inner.transport,
                    self.inner.socket.as_ref(),
                    &s.name,
                )
                .await?;
                Ok(windows
                    .into_iter()
                    .map(|w| Target {
                        inner: self.inner.clone(),
                        address: TargetAddress::Window(w),
                        exec_mutex: Arc::new(Mutex::new(())),
                    })
                    .collect())
            }
            TargetAddress::Window(w) => {
                let panes = discovery::list_panes_in_session(
                    &self.inner.transport,
                    self.inner.socket.as_ref(),
                    &w.session_name,
                )
                .await?;
                Ok(panes
                    .into_iter()
                    .filter(|p| p.address.window == w.index)
                    .map(|p| Target {
                        inner: self.inner.clone(),
                        address: TargetAddress::Pane(p.address),
                        exec_mutex: Arc::new(Mutex::new(())),
                    })
                    .collect())
            }
            TargetAddress::Pane(_) => Ok(Vec::new()),
        }
    }

    /// Navigate to a window by index (from session level).
    pub async fn window(&self, index: u32) -> Result<Option<Target>> {
        let session_name = self.session_name().to_string();
        let windows = discovery::list_windows(
            &self.inner.transport,
            self.inner.socket.as_ref(),
            &session_name,
        )
        .await?;
        Ok(windows.into_iter().find(|w| w.index == index).map(|w| {
            Target {
                inner: self.inner.clone(),
                address: TargetAddress::Window(w),
                exec_mutex: Arc::new(Mutex::new(())),
            }
        }))
    }

    /// Navigate to a pane by index.
    pub async fn pane(&self, index: u32) -> Result<Option<Target>> {
        let session_name = self.session_name().to_string();
        let panes = discovery::list_panes_in_session(
            &self.inner.transport,
            self.inner.socket.as_ref(),
            &session_name,
        )
        .await?;

        let window_filter = match &self.address {
            TargetAddress::Window(w) => Some(w.index),
            _ => None,
        };

        let pane = panes.into_iter().find(|p| {
            p.address.pane == index
                && window_filter.map_or(true, |wi| p.address.window == wi)
        });

        Ok(pane.map(|p| Target {
            inner: self.inner.clone(),
            address: TargetAddress::Pane(p.address),
            exec_mutex: Arc::new(Mutex::new(())),
        }))
    }

    /// Navigate to a pane by PaneAddress.
    pub fn pane_by_address(&self, address: &PaneAddress) -> Target {
        Target {
            inner: self.inner.clone(),
            address: TargetAddress::Pane(address.clone()),
            exec_mutex: Arc::new(Mutex::new(())),
        }
    }

    // --- I/O ---

    /// Send literal text.
    pub async fn send_text(&self, text: &str) -> Result<()> {
        control::send_text(
            &self.inner.transport,
            self.inner.socket.as_ref(),
            &self.target_string(),
            text,
        )
        .await
    }

    /// Send a key sequence.
    pub async fn send_keys(&self, keys: &KeySequence) -> Result<()> {
        control::send_keys(
            &self.inner.transport,
            self.inner.socket.as_ref(),
            &self.target_string(),
            keys,
        )
        .await
    }

    /// Capture visible pane content.
    pub async fn capture(&self) -> Result<String> {
        capture::capture_pane(
            &self.inner.transport,
            self.inner.socket.as_ref(),
            &self.target_string(),
        )
        .await
    }

    /// Capture with scrollback history. `start` is negative for scrollback lines.
    pub async fn capture_with_history(&self, start: i32) -> Result<String> {
        capture::capture_pane_history(
            &self.inner.transport,
            self.inner.socket.as_ref(),
            &self.target_string(),
            start,
        )
        .await
    }

    /// Sample recent scrollback.
    pub async fn sample_text(&self, query: &ScrollbackQuery) -> Result<String> {
        capture::sample_text(
            &self.inner.transport,
            self.inner.socket.as_ref(),
            &self.target_string(),
            query,
        )
        .await
    }

    /// Capture all panes under this target.
    pub async fn capture_all(&self) -> Result<HashMap<PaneAddress, String>> {
        capture::capture_session(
            &self.inner.transport,
            self.inner.socket.as_ref(),
            self.session_name(),
        )
        .await
    }

    // --- Lifecycle ---

    /// Kill this entity.
    pub async fn kill(&self) -> Result<()> {
        match &self.address {
            TargetAddress::Session(s) => {
                control::kill_session(
                    &self.inner.transport,
                    self.inner.socket.as_ref(),
                    &s.name,
                )
                .await
            }
            TargetAddress::Window(_) => {
                control::kill_window(
                    &self.inner.transport,
                    self.inner.socket.as_ref(),
                    &self.target_string(),
                )
                .await
            }
            TargetAddress::Pane(_) => {
                control::kill_pane(
                    &self.inner.transport,
                    self.inner.socket.as_ref(),
                    &self.target_string(),
                )
                .await
            }
        }
    }

    /// Rename this entity.
    pub async fn rename(&self, new_name: &str) -> Result<()> {
        match &self.address {
            TargetAddress::Session(s) => {
                control::rename_session(
                    &self.inner.transport,
                    self.inner.socket.as_ref(),
                    &s.name,
                    new_name,
                )
                .await
            }
            TargetAddress::Window(w) => {
                control::rename_window(
                    &self.inner.transport,
                    self.inner.socket.as_ref(),
                    &w.session_name,
                    w.index,
                    new_name,
                )
                .await
            }
            TargetAddress::Pane(_) => Err(anyhow!("cannot rename a pane")),
        }
    }

    // --- exec (DC19 sentinel mechanism) ---

    /// Execute a shell command in the target pane and capture output.
    ///
    /// Sends command with a UUID sentinel suffix, polls scrollback until
    /// the sentinel appears (or timeout), and extracts stdout + exit code.
    pub async fn exec(&self, command: &str, timeout: Duration) -> Result<ExecOutput> {
        let _guard = self.exec_mutex.lock().await;

        // Use a short hex ID to avoid line wrapping in narrow panes
        let id = &uuid::Uuid::new_v4().to_string()[..8];
        let marker = format!("__ML{}__", id);
        let target = self.target_string();
        let socket = self.inner.socket.as_ref();
        let transport = &self.inner.transport;

        // Detect shell for exit code variable
        let exit_var = detect_exit_var(transport, socket, &target).await;

        // Send command with sentinel.
        // The sentinel echo must NOT be inside single quotes so $? expands.
        // Format: <command> ; echo "<marker> $?"
        let sentinel_cmd = format!(
            "{} ; echo \"{} {}\"",
            command, marker, exit_var
        );
        let keys = KeySequence::literal(&sentinel_cmd).then_enter();
        control::send_keys(transport, socket, &target, &keys).await?;

        // Poll scrollback for sentinel
        let start_time = tokio::time::Instant::now();
        let poll_interval = Duration::from_millis(100);

        loop {
            if start_time.elapsed() > timeout {
                return Err(anyhow!(
                    "exec timed out after {:?} waiting for sentinel",
                    timeout
                ));
            }

            let content =
                capture::capture_pane(transport, socket, &target).await?;

            if let Some(result) = parse_sentinel_output(&content, &marker) {
                return Ok(result);
            }

            tokio::time::sleep(poll_interval).await;
        }
    }
}

/// Detect the shell exit code variable.
async fn detect_exit_var(
    transport: &TransportKind,
    socket: Option<&TmuxSocket>,
    target: &str,
) -> &'static str {
    // Try to detect fish shell
    let prefix = crate::transport::tmux_prefix(socket);
    let cmd = format!(
        "{} display-message -p -t '{}' '#{{pane_current_command}}'",
        prefix,
        crate::transport::shell_escape_arg(target)
    );
    if let Ok(shell) = transport.exec(&cmd).await {
        let shell = shell.trim().to_lowercase();
        if shell.contains("fish") {
            return "$status";
        }
    }
    "$?"
}

/// Parse sentinel output from captured scrollback.
fn parse_sentinel_output(content: &str, marker: &str) -> Option<ExecOutput> {
    let lines: Vec<&str> = content.lines().collect();

    // Find the sentinel output line: starts with the marker (not part of echo command).
    // The echo command line contains "echo" before the marker.
    // The sentinel output line starts with the marker directly.
    let mut sentinel_idx = None;
    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with(marker) {
            sentinel_idx = Some(i);
            break;
        }
    }

    let sentinel_idx = sentinel_idx?;
    let sentinel_line = lines[sentinel_idx].trim();

    // Extract exit code: marker followed by space and exit code
    let after_marker = sentinel_line.strip_prefix(marker)?.trim();
    let exit_code = after_marker.parse::<i32>().unwrap_or(-1);

    // Find the command echo line (before sentinel, contains the marker in a command)
    let mut cmd_echo_idx = None;
    for i in (0..sentinel_idx).rev() {
        if lines[i].contains(marker) {
            cmd_echo_idx = Some(i);
            break;
        }
    }

    // stdout is everything between command echo and sentinel
    let start = cmd_echo_idx.map_or(0, |i| i + 1);
    let stdout = lines[start..sentinel_idx].join("\n");
    let stdout = stdout.trim_end().to_string();

    Some(ExecOutput { stdout, exit_code })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::MockTransport;

    fn mock_host(mock: MockTransport) -> HostHandle {
        HostHandle::new(TransportKind::Mock(mock), None)
    }

    #[tokio::test]
    async fn create_session_returns_target() {
        let mock = MockTransport::new()
            .with_default("")
            .with_response("list-sessions", "test\t$0\t1700000000\t0\t1\t\n");
        let host = mock_host(mock);
        let target = host.create_session("test", None, None).await.unwrap();
        assert_eq!(target.level(), TargetLevel::Session);
        assert_eq!(target.target_string(), "test");
        assert_eq!(target.session_name(), "test");
    }

    #[tokio::test]
    async fn session_not_found() {
        let mock = MockTransport::new()
            .with_response("list-sessions", "other\t$0\t0\t0\t1\t\n");
        let host = mock_host(mock);
        let result = host.session("nonexistent").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn session_found() {
        let mock = MockTransport::new()
            .with_response("list-sessions", "build\t$0\t0\t1\t2\t\n");
        let host = mock_host(mock);
        let target = host.session("build").await.unwrap();
        assert!(target.is_some());
        let t = target.unwrap();
        assert_eq!(t.level(), TargetLevel::Session);
        assert_eq!(t.target_string(), "build");
    }

    #[tokio::test]
    async fn target_spec_session_level() {
        let mock = MockTransport::new()
            .with_response("list-sessions", "build\t$0\t0\t0\t1\t\n");
        let host = mock_host(mock);
        let spec = TargetSpec::session("build");
        let t = host.target(&spec).await.unwrap();
        assert!(t.is_some());
        assert_eq!(t.unwrap().level(), TargetLevel::Session);
    }

    #[tokio::test]
    async fn children_session_lists_windows() {
        let mock = MockTransport::new()
            .with_response("list-sessions", "build\t$0\t0\t0\t2\t\n")
            .with_response("list-windows", "$0\tbuild\t0\tmain\t1\t1\tlayout\n$0\tbuild\t1\teditor\t0\t1\tlayout\n");
        let host = mock_host(mock);
        let target = host.session("build").await.unwrap().unwrap();
        let children = target.children().await.unwrap();
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].level(), TargetLevel::Window);
        assert_eq!(children[0].target_string(), "build:0");
        assert_eq!(children[1].target_string(), "build:1");
    }

    #[tokio::test]
    async fn rename_pane_fails() {
        let addr = PaneAddress {
            pane_id: "%0".to_string(),
            session: "test".to_string(),
            window: 0,
            pane: 0,
        };
        let mock = MockTransport::new().with_default("");
        let host = mock_host(mock);
        let target = Target {
            inner: host.inner.clone(),
            address: TargetAddress::Pane(addr),
            exec_mutex: Arc::new(Mutex::new(())),
        };
        assert!(target.rename("new_name").await.is_err());
    }

    #[test]
    fn parse_sentinel_basic() {
        let marker = "__MLabc123__";
        let content = format!(
            "$ echo hello ; echo \"{} $?\"\nhello\n{} 0\n$",
            marker, marker
        );
        let result = parse_sentinel_output(&content, marker);
        assert!(result.is_some());
        let out = result.unwrap();
        assert_eq!(out.exit_code, 0);
        assert_eq!(out.stdout, "hello");
    }

    #[test]
    fn parse_sentinel_nonzero_exit() {
        let marker = "__MLxyz789__";
        let content = format!(
            "$ false ; echo \"{} $?\"\n{} 1\n$",
            marker, marker
        );
        let result = parse_sentinel_output(&content, marker);
        assert!(result.is_some());
        assert_eq!(result.unwrap().exit_code, 1);
    }

    #[test]
    fn parse_sentinel_not_found() {
        let result = parse_sentinel_output("just some output\n", "__MLnope__");
        assert!(result.is_none());
    }

    #[test]
    fn parse_sentinel_multiline_output() {
        let marker = "__MLtest42__";
        let content = format!(
            "$ ls ; echo \"{} $?\"\nfile1\nfile2\nfile3\n{} 0\n$",
            marker, marker
        );
        let result = parse_sentinel_output(&content, marker).unwrap();
        assert_eq!(result.exit_code, 0);
        assert_eq!(result.stdout, "file1\nfile2\nfile3");
    }
}
