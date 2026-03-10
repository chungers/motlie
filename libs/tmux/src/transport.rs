use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Mutex;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, Command};

use crate::types::TmuxSocket;

/// Static-dispatch transport for command execution (DC6).
pub enum TransportKind {
    Local(LocalTransport),
    Mock(MockTransport),
    // Ssh(SshTransport) — Phase 2a
}

impl TransportKind {
    /// Execute a command and return stdout.
    pub async fn exec(&self, command: &str) -> Result<String> {
        match self {
            TransportKind::Local(t) => t.exec(command).await,
            TransportKind::Mock(t) => t.exec(command).await,
        }
    }

    /// Open a persistent shell channel.
    pub async fn open_shell(&self) -> Result<ShellChannelKind> {
        match self {
            TransportKind::Local(t) => t.open_shell().await.map(ShellChannelKind::Local),
            TransportKind::Mock(t) => t.open_shell().await.map(ShellChannelKind::Mock),
        }
    }
}

/// Localhost transport — executes via subprocess.
pub struct LocalTransport {
    pub timeout: std::time::Duration,
}

impl LocalTransport {
    pub fn new() -> Self {
        LocalTransport {
            timeout: std::time::Duration::from_secs(10),
        }
    }

    pub fn with_timeout(timeout: std::time::Duration) -> Self {
        LocalTransport { timeout }
    }

    async fn exec(&self, command: &str) -> Result<String> {
        let output = tokio::time::timeout(
            self.timeout,
            Command::new("sh")
                .arg("-c")
                .arg(command)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output(),
        )
        .await
        .map_err(|_| anyhow!("command timed out after {:?}: {}", self.timeout, command))?
        .map_err(|e| anyhow!("failed to execute command: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow!(
                "command failed (exit {}): {}\nstderr: {}",
                output.status.code().unwrap_or(-1),
                command,
                stderr.trim()
            ));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    async fn open_shell(&self) -> Result<LocalShellChannel> {
        let child = Command::new("sh")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| anyhow!("failed to spawn shell: {}", e))?;

        Ok(LocalShellChannel { child })
    }
}

impl Default for LocalTransport {
    fn default() -> Self {
        Self::new()
    }
}

/// Mock transport for unit testing.
pub struct MockTransport {
    responses: Mutex<HashMap<String, Vec<String>>>,
    default_response: String,
}

impl MockTransport {
    pub fn new() -> Self {
        MockTransport {
            responses: Mutex::new(HashMap::new()),
            default_response: String::new(),
        }
    }

    /// Add a canned response for a command pattern.
    /// If multiple responses are added for the same pattern, they are
    /// returned in FIFO order. When exhausted, returns the last one.
    pub fn with_response(self, command_contains: &str, response: &str) -> Self {
        let mut responses = self.responses.lock().unwrap();
        responses
            .entry(command_contains.to_string())
            .or_default()
            .push(response.to_string());
        drop(responses);
        self
    }

    /// Set default response for unmatched commands.
    pub fn with_default(mut self, response: &str) -> Self {
        self.default_response = response.to_string();
        self
    }

    async fn exec(&self, command: &str) -> Result<String> {
        let mut responses = self.responses.lock().unwrap();
        for (pattern, queue) in responses.iter_mut() {
            if command.contains(pattern.as_str()) {
                if queue.len() > 1 {
                    return Ok(queue.remove(0));
                } else {
                    return Ok(queue[0].clone());
                }
            }
        }
        Ok(self.default_response.clone())
    }

    async fn open_shell(&self) -> Result<MockShellChannel> {
        Ok(MockShellChannel {
            data: Vec::new(),
            pos: 0,
        })
    }
}

impl Default for MockTransport {
    fn default() -> Self {
        Self::new()
    }
}

/// Shell channel — static dispatch.
pub enum ShellChannelKind {
    Local(LocalShellChannel),
    Mock(MockShellChannel),
}

impl ShellChannelKind {
    pub async fn write(&mut self, data: &[u8]) -> Result<()> {
        match self {
            ShellChannelKind::Local(ch) => ch.write(data).await,
            ShellChannelKind::Mock(ch) => ch.write(data).await,
        }
    }

    pub async fn read(&mut self) -> Option<ShellEvent> {
        match self {
            ShellChannelKind::Local(ch) => ch.read().await,
            ShellChannelKind::Mock(ch) => ch.read().await,
        }
    }
}

/// Events from a shell channel.
#[derive(Debug)]
pub enum ShellEvent {
    Data(Vec<u8>),
    Eof,
}

/// Localhost shell channel backed by a child process.
pub struct LocalShellChannel {
    child: Child,
}

impl LocalShellChannel {
    async fn write(&mut self, data: &[u8]) -> Result<()> {
        let stdin = self
            .child
            .stdin
            .as_mut()
            .ok_or_else(|| anyhow!("stdin not available"))?;
        stdin
            .write_all(data)
            .await
            .map_err(|e| anyhow!("write to shell failed: {}", e))
    }

    async fn read(&mut self) -> Option<ShellEvent> {
        let stdout = self.child.stdout.as_mut()?;
        let mut buf = vec![0u8; 4096];
        match stdout.read(&mut buf).await {
            Ok(0) => Some(ShellEvent::Eof),
            Ok(n) => {
                buf.truncate(n);
                Some(ShellEvent::Data(buf))
            }
            Err(_) => Some(ShellEvent::Eof),
        }
    }
}

/// Mock shell channel for testing.
pub struct MockShellChannel {
    data: Vec<Vec<u8>>,
    pos: usize,
}

impl MockShellChannel {
    async fn write(&mut self, _data: &[u8]) -> Result<()> {
        Ok(())
    }

    async fn read(&mut self) -> Option<ShellEvent> {
        if self.pos < self.data.len() {
            let d = self.data[self.pos].clone();
            self.pos += 1;
            Some(ShellEvent::Data(d))
        } else {
            Some(ShellEvent::Eof)
        }
    }
}

/// Build the tmux command prefix with optional socket args.
pub fn tmux_prefix(socket: Option<&TmuxSocket>) -> String {
    match socket {
        None => "tmux".to_string(),
        Some(TmuxSocket::Name(n)) => format!("tmux -L '{}'", shell_escape_arg(n)),
        Some(TmuxSocket::Path(p)) => format!("tmux -S '{}'", shell_escape_arg(p)),
    }
}

/// POSIX shell escape: single-quote wrapping with '\'' for interior quotes.
pub fn shell_escape_arg(s: &str) -> String {
    s.replace('\'', "'\\''")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_transport_canned_response() {
        let mock = MockTransport::new()
            .with_response("list-sessions", "session1\nsession2\n")
            .with_response("list-panes", "pane1\npane2\n");

        let result = mock.exec("tmux list-sessions").await.unwrap();
        assert_eq!(result, "session1\nsession2\n");

        let result = mock.exec("tmux list-panes -a").await.unwrap();
        assert_eq!(result, "pane1\npane2\n");
    }

    #[tokio::test]
    async fn mock_transport_default_response() {
        let mock = MockTransport::new().with_default("ok");
        let result = mock.exec("anything").await.unwrap();
        assert_eq!(result, "ok");
    }

    #[tokio::test]
    async fn local_transport_echo() {
        let local = LocalTransport::new();
        let result = local.exec("echo hello").await.unwrap();
        assert_eq!(result.trim(), "hello");
    }

    #[test]
    fn tmux_prefix_default() {
        assert_eq!(tmux_prefix(None), "tmux");
    }

    #[test]
    fn tmux_prefix_named() {
        let socket = TmuxSocket::Name("myserver".to_string());
        assert_eq!(tmux_prefix(Some(&socket)), "tmux -L 'myserver'");
    }

    #[test]
    fn tmux_prefix_path() {
        let socket = TmuxSocket::Path("/tmp/tmux.sock".to_string());
        assert_eq!(tmux_prefix(Some(&socket)), "tmux -S '/tmp/tmux.sock'");
    }

    #[test]
    fn shell_escape_simple() {
        assert_eq!(shell_escape_arg("hello"), "hello");
    }

    #[test]
    fn shell_escape_quotes() {
        assert_eq!(shell_escape_arg("it's"), "it'\\''s");
    }

    #[test]
    fn shell_escape_multiple_quotes() {
        assert_eq!(shell_escape_arg("a'b'c"), "a'\\''b'\\''c");
    }
}
