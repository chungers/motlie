use anyhow::{anyhow, Result};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::process::{Child, Command};

use crate::types::{HostKeyPolicy, TmuxSocket};

// ---------------------------------------------------------------------------
// Static-dispatch transport (DC6)
// ---------------------------------------------------------------------------

/// Static-dispatch transport for command execution (DC6).
pub enum TransportKind {
    Local(LocalTransport),
    Mock(MockTransport),
    Ssh(SshTransport),
}

impl TransportKind {
    /// Execute a command and return stdout.
    pub async fn exec(&self, command: &str) -> Result<String> {
        match self {
            TransportKind::Local(t) => t.exec(command).await,
            TransportKind::Mock(t) => t.exec(command).await,
            TransportKind::Ssh(t) => t.exec(command).await,
        }
    }

    /// Transport-agnostic health probe.
    ///
    /// - `Local`: always returns `true` (subprocess transport has no persistent connection).
    /// - `Mock`: always returns `true`.
    /// - `Ssh`: checks if the underlying SSH connection is closed via `SshTransport::is_closed()`.
    pub fn is_healthy(&self) -> bool {
        match self {
            TransportKind::Local(_) => true,
            TransportKind::Mock(_) => true,
            TransportKind::Ssh(t) => !t.is_closed(),
        }
    }

    /// Open a persistent shell channel.
    ///
    /// `cols` and `rows` control PTY dimensions for SSH transports (used by
    /// `SshTransport::open_shell()`). Local and Mock transports ignore them.
    pub async fn open_shell(&self, cols: u32, rows: u32) -> Result<ShellChannelKind> {
        match self {
            TransportKind::Local(t) => t.open_shell().await.map(ShellChannelKind::Local),
            TransportKind::Mock(t) => t.open_shell().await.map(ShellChannelKind::Mock),
            TransportKind::Ssh(t) => t.open_shell(cols, rows).await.map(ShellChannelKind::Ssh),
        }
    }
}

// ---------------------------------------------------------------------------
// LocalTransport
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// MockTransport
// ---------------------------------------------------------------------------

/// Mock transport for unit testing.
///
/// Patterns are matched in insertion order (first match wins), which makes
/// test behavior deterministic regardless of hash seed. Use `with_response()`
/// to add success responses and `with_error()` to add error responses.
pub struct MockTransport {
    /// Ordered (pattern, response_queue) pairs. First-match wins.
    responses: Mutex<Vec<(String, Vec<String>)>>,
    /// Ordered (pattern, error_message) pairs. Checked before `responses`.
    errors: Mutex<Vec<(String, String)>>,
    default_response: String,
}

impl MockTransport {
    pub fn new() -> Self {
        MockTransport {
            responses: Mutex::new(Vec::new()),
            errors: Mutex::new(Vec::new()),
            default_response: String::new(),
        }
    }

    /// Add a canned response for a command pattern.
    /// If multiple responses are added for the same pattern, they are
    /// returned in FIFO order. When exhausted, returns the last one.
    /// Patterns are matched in insertion order (first match wins).
    pub fn with_response(self, command_contains: &str, response: &str) -> Self {
        let mut responses = self.responses.lock().unwrap();
        if let Some(entry) = responses.iter_mut().find(|(p, _)| p == command_contains) {
            entry.1.push(response.to_string());
        } else {
            responses.push((command_contains.to_string(), vec![response.to_string()]));
        }
        drop(responses);
        self
    }

    /// Add a canned error for a command pattern.
    /// When `exec()` matches this pattern, it returns `Err(anyhow!(...))`.
    /// Error patterns are checked before response patterns.
    pub fn with_error(self, command_contains: &str, message: &str) -> Self {
        let mut errors = self.errors.lock().unwrap();
        errors.push((command_contains.to_string(), message.to_string()));
        drop(errors);
        self
    }

    /// Set default response for unmatched commands.
    pub fn with_default(mut self, response: &str) -> Self {
        self.default_response = response.to_string();
        self
    }

    async fn exec(&self, command: &str) -> Result<String> {
        // Check error patterns first (insertion order)
        {
            let errors = self.errors.lock().unwrap();
            for (pattern, message) in errors.iter() {
                if command.contains(pattern.as_str()) {
                    return Err(anyhow!("{}", message));
                }
            }
        }

        // Check response patterns (insertion order, first match wins)
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

// ---------------------------------------------------------------------------
// SshTransport (Phase 2a.1, DC2, DC6)
// ---------------------------------------------------------------------------

/// SSH/host connection configuration.
///
/// Constructed via builder (`SshConfig::new()`) or parsed from an SSH URI
/// string (`SshConfig::parse()`). Supports both nassh-style (`;` in userinfo)
/// and query-param (`?key=value`) syntax.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SshConfig {
    host: String,
    port: u16,
    user: String,
    host_key_policy: HostKeyPolicy,
    timeout: std::time::Duration,
    keepalive_interval: Option<std::time::Duration>,
    socket: Option<TmuxSocket>,
}

impl SshConfig {
    // --- Builder ---

    pub fn new(host: impl Into<String>, user: impl Into<String>) -> Self {
        SshConfig {
            host: host.into(),
            port: 22,
            user: user.into(),
            host_key_policy: HostKeyPolicy::default(),
            timeout: std::time::Duration::from_secs(10),
            keepalive_interval: Some(std::time::Duration::from_secs(30)),
            socket: None,
        }
    }

    pub fn with_port(mut self, port: u16) -> Self {
        self.port = port;
        self
    }

    pub fn with_host_key_policy(mut self, policy: HostKeyPolicy) -> Self {
        self.host_key_policy = policy;
        self
    }

    /// Set the per-command execution timeout for the SSH transport.
    ///
    /// This bounds each individual `exec()` call (channel open + exec + output
    /// collection). It is independent of `Target::exec()`'s sentinel-poll
    /// timeout, which bounds how long to wait for a user command's output
    /// to appear in scrollback. Keep this short (seconds) to catch hung
    /// connections; use a larger `Target::exec()` timeout for long-running
    /// user commands.
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_keepalive(mut self, interval: Option<std::time::Duration>) -> Self {
        self.keepalive_interval = interval;
        self
    }

    pub fn with_socket(mut self, socket: TmuxSocket) -> Self {
        self.socket = Some(socket);
        self
    }

    // --- Accessors ---

    pub fn host(&self) -> &str {
        &self.host
    }

    pub fn user(&self) -> &str {
        &self.user
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    pub fn host_key_policy(&self) -> &HostKeyPolicy {
        &self.host_key_policy
    }

    pub fn timeout(&self) -> std::time::Duration {
        self.timeout
    }

    pub fn keepalive_interval(&self) -> Option<std::time::Duration> {
        self.keepalive_interval
    }

    pub fn socket(&self) -> Option<&TmuxSocket> {
        self.socket.as_ref()
    }

    pub fn is_localhost(&self) -> bool {
        self.host == "localhost" || self.host == "127.0.0.1" || self.host == "::1"
    }
}

/// SSH client handler implementing host key verification (DC2).
struct SshHandler {
    host: String,
    port: u16,
    policy: HostKeyPolicy,
}

#[async_trait::async_trait]
impl russh::client::Handler for SshHandler {
    type Error = russh::Error;

    async fn check_server_key(
        &mut self,
        server_public_key: &russh_keys::key::PublicKey,
    ) -> Result<bool, Self::Error> {
        match &self.policy {
            HostKeyPolicy::Verify => {
                match russh_keys::known_hosts::check_known_hosts(&self.host, self.port, server_public_key) {
                    Ok(true) => Ok(true),
                    Ok(false) => {
                        tracing::error!(
                            host = %self.host,
                            port = self.port,
                            "SSH host key not found in known_hosts. \
                             Add the key with: ssh-keyscan -p {} {} >> ~/.ssh/known_hosts",
                            self.port,
                            self.host
                        );
                        Ok(false)
                    }
                    Err(russh_keys::Error::KeyChanged { line }) => {
                        tracing::error!(
                            host = %self.host,
                            port = self.port,
                            line,
                            "SSH HOST KEY HAS CHANGED — possible MITM attack. \
                             The key at ~/.ssh/known_hosts line {} does not match. \
                             If this is expected, remove the old entry and reconnect.",
                            line
                        );
                        Ok(false)
                    }
                    Err(e) => {
                        tracing::error!(
                            host = %self.host,
                            error = %e,
                            "Failed to check known_hosts"
                        );
                        Ok(false)
                    }
                }
            }
            HostKeyPolicy::TrustFirstUse => {
                match russh_keys::known_hosts::check_known_hosts(&self.host, self.port, server_public_key) {
                    Ok(true) => Ok(true),
                    Ok(false) => {
                        // First connection — learn the key
                        tracing::info!(
                            host = %self.host,
                            port = self.port,
                            "Trust-on-first-use: accepting and persisting new host key"
                        );
                        if let Err(e) = russh_keys::known_hosts::learn_known_hosts(
                            &self.host,
                            self.port,
                            server_public_key,
                        ) {
                            tracing::error!(
                                host = %self.host,
                                error = %e,
                                "TOFU: failed to persist host key — rejecting connection. \
                                 Check that ~/.ssh/known_hosts is writable."
                            );
                            return Ok(false);
                        }
                        Ok(true)
                    }
                    Err(russh_keys::Error::KeyChanged { line }) => {
                        tracing::error!(
                            host = %self.host,
                            port = self.port,
                            line,
                            "SSH HOST KEY HAS CHANGED — rejecting (TOFU policy). \
                             The key at ~/.ssh/known_hosts line {} does not match.",
                            line
                        );
                        Ok(false)
                    }
                    Err(e) => {
                        tracing::warn!(
                            host = %self.host,
                            error = %e,
                            "Failed to check known_hosts, attempting to learn key (TOFU)"
                        );
                        if let Err(e) = russh_keys::known_hosts::learn_known_hosts(
                            &self.host,
                            self.port,
                            server_public_key,
                        ) {
                            tracing::error!(
                                error = %e,
                                "TOFU: failed to persist host key — rejecting connection"
                            );
                            return Ok(false);
                        }
                        Ok(true)
                    }
                }
            }
            HostKeyPolicy::Insecure => {
                tracing::warn!(
                    host = %self.host,
                    port = self.port,
                    "Insecure mode: accepting SSH host key without verification"
                );
                Ok(true)
            }
        }
    }
}

/// SSH transport — executes commands on a remote host via russh (Phase 2a.1).
pub struct SshTransport {
    handle: Arc<tokio::sync::Mutex<russh::client::Handle<SshHandler>>>,
    config: SshConfig,
}

impl SshTransport {
    /// Connect to a remote host via SSH and authenticate using ssh-agent.
    ///
    /// Returns an error with actionable message if SSH_AUTH_SOCK is not set
    /// or the agent has no identities (OC3).
    pub async fn connect(config: SshConfig) -> Result<Self> {
        let ssh_config = russh::client::Config {
            inactivity_timeout: Some(config.timeout),
            keepalive_interval: config.keepalive_interval,
            ..<_>::default()
        };

        let handler = SshHandler {
            host: config.host.clone(),
            port: config.port,
            policy: config.host_key_policy.clone(),
        };

        let addr = format!("{}:{}", config.host, config.port);
        let mut handle = tokio::time::timeout(
            config.timeout,
            russh::client::connect(Arc::new(ssh_config), &addr, handler),
        )
        .await
        .map_err(|_| {
            anyhow!(
                "SSH connection to {}:{} timed out after {:?}",
                config.host,
                config.port,
                config.timeout
            )
        })?
        .map_err(|e| {
            anyhow!(
                "SSH connection to {}:{} failed: {}",
                config.host,
                config.port,
                e
            )
        })?;

        // Authenticate via ssh-agent
        Self::authenticate_agent(&mut handle, &config).await?;

        Ok(SshTransport {
            handle: Arc::new(tokio::sync::Mutex::new(handle)),
            config,
        })
    }

    /// Authenticate using ssh-agent keys.
    async fn authenticate_agent(
        handle: &mut russh::client::Handle<SshHandler>,
        config: &SshConfig,
    ) -> Result<()> {
        let mut agent = russh_keys::agent::client::AgentClient::connect_env()
            .await
            .map_err(|e| {
                anyhow!(
                    "Failed to connect to SSH agent (is SSH_AUTH_SOCK set?): {}. \
                     Ensure ssh-agent is running and SSH_AUTH_SOCK is exported.",
                    e
                )
            })?;

        let identities = agent.request_identities().await.map_err(|e| {
            anyhow!(
                "Failed to list SSH agent identities: {}. \
                 Ensure keys are loaded with ssh-add.",
                e
            )
        })?;

        if identities.is_empty() {
            return Err(anyhow!(
                "SSH agent has no identities. Add a key with: ssh-add ~/.ssh/id_ed25519"
            ));
        }

        // Try each agent key until one succeeds
        for key in &identities {
            let (returned_agent, auth_result) = handle
                .authenticate_future(&config.user, key.clone(), agent)
                .await;
            agent = returned_agent;

            match auth_result {
                Ok(true) => {
                    tracing::debug!(
                        host = %config.host,
                        user = %config.user,
                        "SSH authentication succeeded"
                    );
                    return Ok(());
                }
                Ok(false) => {
                    tracing::debug!("SSH key rejected, trying next identity");
                    continue;
                }
                Err(e) => {
                    tracing::debug!(error = %e, "SSH auth error with key, trying next");
                    continue;
                }
            }
        }

        Err(anyhow!(
            "SSH authentication failed for user '{}' on {}:{}. \
             None of the {} agent key(s) were accepted.",
            config.user,
            config.host,
            config.port,
            identities.len()
        ))
    }

    /// Execute a command on the remote host and return stdout.
    ///
    /// The full exec lifecycle — channel open, exec request, and output
    /// collection — is bounded by `config.timeout`. The SSH handle lock is
    /// held only during channel open (not the full command lifetime), allowing
    /// multiple concurrent execs on the same connection.
    async fn exec(&self, command: &str) -> Result<String> {
        // Single timeout boundary covering channel open + exec + output
        // collection, so a stalled server at any phase is caught.
        let (stdout, stderr, exit_code) = tokio::time::timeout(
            self.config.timeout,
            async {
                // Lock only to open the channel, then release. The Channel is
                // self-contained — its read/write operations don't need the Handle.
                let mut channel = {
                    let handle = self.handle.lock().await;
                    handle.channel_open_session().await.map_err(|e| {
                        anyhow!("SSH: failed to open session channel: {}", e)
                    })?
                };

                channel.exec(true, command).await.map_err(|e| {
                    anyhow!("SSH: failed to exec command: {}", e)
                })?;

                // Collect output
                let mut stdout = Vec::new();
                let mut stderr = Vec::new();
                let mut exit_code: Option<u32> = None;

                while let Some(msg) = channel.wait().await {
                    match msg {
                        russh::ChannelMsg::Data { ref data } => {
                            stdout.extend_from_slice(data);
                        }
                        russh::ChannelMsg::ExtendedData { ref data, ext } => {
                            if ext == 1 {
                                stderr.extend_from_slice(data);
                            }
                        }
                        russh::ChannelMsg::ExitStatus { exit_status } => {
                            exit_code = Some(exit_status);
                        }
                        russh::ChannelMsg::Eof | russh::ChannelMsg::Close => break,
                        _ => {}
                    }
                }

                Ok::<_, anyhow::Error>((stdout, stderr, exit_code))
            },
        )
        .await
        .map_err(|_| {
            anyhow!(
                "SSH command timed out after {:?}: {}",
                self.config.timeout,
                command
            )
        })??;

        let code = exit_code.unwrap_or(0);
        if code != 0 {
            let stderr_str = String::from_utf8_lossy(&stderr);
            return Err(anyhow!(
                "command failed (exit {}): {}\nstderr: {}",
                code,
                command,
                stderr_str.trim()
            ));
        }

        Ok(String::from_utf8_lossy(&stdout).to_string())
    }

    /// Open a persistent shell channel on the remote host with a PTY.
    ///
    /// `cols` and `rows` set the initial PTY dimensions. Use the target pane's
    /// geometry for accurate rendering, or pass `(80, 24)` as a safe default.
    async fn open_shell(&self, cols: u32, rows: u32) -> Result<SshShellChannel> {
        let channel = {
            let handle = self.handle.lock().await;
            handle.channel_open_session().await.map_err(|e| {
                anyhow!("SSH: failed to open session channel for shell: {}", e)
            })?
        };

        // Request a PTY for interactive shell use
        channel
            .request_pty(
                true,
                "xterm",
                cols,
                rows,
                0,    // pixel width
                0,    // pixel height
                &[],  // terminal modes
            )
            .await
            .map_err(|e| anyhow!("SSH: failed to request PTY: {}", e))?;

        channel
            .request_shell(true)
            .await
            .map_err(|e| anyhow!("SSH: failed to request shell: {}", e))?;

        Ok(SshShellChannel { channel })
    }

    /// Check if the SSH connection is still alive.
    pub fn is_closed(&self) -> bool {
        // Try to check without blocking — if we can't get the lock, assume alive
        match self.handle.try_lock() {
            Ok(handle) => handle.is_closed(),
            Err(_) => false,
        }
    }

    /// Get a reference to the SSH configuration.
    pub fn config(&self) -> &SshConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Shell channels
// ---------------------------------------------------------------------------

/// Shell channel — static dispatch.
pub enum ShellChannelKind {
    Local(LocalShellChannel),
    Mock(MockShellChannel),
    Ssh(SshShellChannel),
}

impl ShellChannelKind {
    pub async fn write(&mut self, data: &[u8]) -> Result<()> {
        match self {
            ShellChannelKind::Local(ch) => ch.write(data).await,
            ShellChannelKind::Mock(ch) => ch.write(data).await,
            ShellChannelKind::Ssh(ch) => ch.write(data).await,
        }
    }

    pub async fn read(&mut self) -> Option<ShellEvent> {
        match self {
            ShellChannelKind::Local(ch) => ch.read().await,
            ShellChannelKind::Mock(ch) => ch.read().await,
            ShellChannelKind::Ssh(ch) => ch.read().await,
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

/// SSH shell channel backed by a russh PTY session.
pub struct SshShellChannel {
    channel: russh::Channel<russh::client::Msg>,
}

impl SshShellChannel {
    async fn write(&mut self, data: &[u8]) -> Result<()> {
        self.channel
            .data(&data[..])
            .await
            .map_err(|e| anyhow!("SSH: write to shell failed: {}", e))
    }

    async fn read(&mut self) -> Option<ShellEvent> {
        match self.channel.wait().await {
            Some(russh::ChannelMsg::Data { data }) => {
                Some(ShellEvent::Data(data.to_vec()))
            }
            Some(russh::ChannelMsg::ExtendedData { data, .. }) => {
                Some(ShellEvent::Data(data.to_vec()))
            }
            Some(russh::ChannelMsg::Eof) | Some(russh::ChannelMsg::Close) | None => {
                Some(ShellEvent::Eof)
            }
            Some(_) => {
                // Other messages (ExitStatus, etc.) — skip and read again
                // Recurse via Box::pin to avoid stack growth
                Box::pin(self.read()).await
            }
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

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

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
    async fn mock_transport_error_response() {
        let mock = MockTransport::new()
            .with_error("kill-session", "session not found")
            .with_response("list-sessions", "ok\n");
        // Error pattern matches → Err
        let result = mock.exec("tmux kill-session -t foo").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("session not found"));
        // Non-error pattern still works
        let result = mock.exec("tmux list-sessions").await.unwrap();
        assert_eq!(result, "ok\n");
    }

    #[tokio::test]
    async fn mock_transport_error_before_response() {
        // Error patterns are checked before response patterns
        let mock = MockTransport::new()
            .with_response("cmd", "ok")
            .with_error("cmd", "fail");
        let result = mock.exec("cmd").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn mock_transport_insertion_order() {
        // First-match wins: "list" matches before "list-sessions"
        let mock = MockTransport::new()
            .with_response("list", "first")
            .with_response("list-sessions", "second");
        let result = mock.exec("tmux list-sessions").await.unwrap();
        assert_eq!(result, "first");
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

    #[test]
    fn ssh_config_defaults() {
        let cfg = SshConfig::new("example.com", "deploy");
        assert_eq!(cfg.host(), "example.com");
        assert_eq!(cfg.port(), 22);
        assert_eq!(cfg.user(), "deploy");
        assert_eq!(*cfg.host_key_policy(), HostKeyPolicy::Verify);
        assert_eq!(cfg.timeout(), std::time::Duration::from_secs(10));
        assert_eq!(
            cfg.keepalive_interval(),
            Some(std::time::Duration::from_secs(30))
        );
        assert!(cfg.socket().is_none());
    }

    #[test]
    fn ssh_config_builder() {
        let cfg = SshConfig::new("host", "user")
            .with_port(2222)
            .with_host_key_policy(HostKeyPolicy::Insecure)
            .with_timeout(std::time::Duration::from_secs(30))
            .with_keepalive(None)
            .with_socket(TmuxSocket::Name("test".into()));
        assert_eq!(cfg.port(), 2222);
        assert_eq!(*cfg.host_key_policy(), HostKeyPolicy::Insecure);
        assert_eq!(cfg.timeout(), std::time::Duration::from_secs(30));
        assert_eq!(cfg.keepalive_interval(), None);
        assert_eq!(cfg.socket(), Some(&TmuxSocket::Name("test".into())));
    }

    #[test]
    fn ssh_config_is_localhost() {
        assert!(SshConfig::new("localhost", "u").is_localhost());
        assert!(SshConfig::new("127.0.0.1", "u").is_localhost());
        assert!(SshConfig::new("::1", "u").is_localhost());
        assert!(!SshConfig::new("remote", "u").is_localhost());
    }
}
