//! SSH proxy: host-side ingress and programmatic guest control plane.
//!
//! ## Design (FR-6, FR-7, DESIGN.md)
//!
//! The SSH proxy serves two roles:
//!
//! 1. **User-facing ingress** — russh server on localhost, bridges SSH
//!    channels to guest sshd via CA-signed ephemeral certs over vsock.
//! 2. **Programmatic exec** — opens non-PTY SSH channels to execute
//!    commands inside guests and capture output, enabling automated
//!    validation without human intervention.
//!
//! ## Transport
//!
//! The guest connects OUT to the host via vsock (guest→host direction).
//! A socat bridge in the guest connects `VSOCK-CONNECT:2:<port>` to
//! `TCP:localhost:22` (sshd). The host accepts on the vsock UDS and
//! feeds the stream into russh::client::connect_stream(). Multiple SSH
//! sessions multiplex over this single connection via SSH channels.
//!
//! ## Auth model
//!
//! - Inbound: localhost trust (username = guest identity)
//! - Outbound: ephemeral Ed25519 cert signed by in-memory CA

use std::collections::{HashMap, VecDeque};
use std::io::ErrorKind;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use russh::server::{Auth, Msg, Session};
use russh::{Channel, ChannelMsg, ChannelWriteHalf, Pty};
use serde::Serialize;
use thiserror::Error;
use tokio::net::UnixListener;
use tokio::task::JoinHandle;
use tokio::time::{Instant, sleep, timeout as tokio_timeout};

use crate::ca::{CaError, SshCa};

#[cfg(feature = "debug-trace")]
macro_rules! debug_trace {
    ($($arg:tt)*) => {
        eprintln!($($arg)*);
    };
}

#[cfg(not(feature = "debug-trace"))]
macro_rules! debug_trace {
    ($($arg:tt)*) => {};
}

/// vsock port used for the guest→host SSH bridge.
pub const VSOCK_SSH_PORT: u32 = 2222;

#[derive(Debug, Error)]
pub enum SshProxyError {
    #[error("failed to connect to guest sshd for '{guest}': {reason}")]
    GuestConnection { guest: String, reason: String },
    #[error("exec failed on guest {guest}: {reason}")]
    ExecFailed { guest: String, reason: String },
    #[error("CA error: {0}")]
    Ca(#[from] CaError),
    #[error("SSH error: {0}")]
    Ssh(String),
    #[error("channel closed before command completed")]
    ChannelClosed,
    #[error("command on guest '{guest}' completed without an exit status: {command}")]
    MissingExitStatus { guest: String, command: String },
    #[error("unknown guest: {0}")]
    UnknownGuest(String),
    #[error("internal state poisoned: {0}")]
    StatePoisoned(&'static str),
    #[error("PTY operation timed out on guest '{guest}': {expectation}")]
    PtyTimeout { guest: String, expectation: String },
    #[error("unsupported SSH control-plane operation: {0}")]
    Unsupported(&'static str),
}

impl From<russh::Error> for SshProxyError {
    fn from(value: russh::Error) -> Self {
        Self::Ssh(value.to_string())
    }
}

/// Output from a programmatic command execution inside a guest VM.
#[derive(Debug, Clone, Serialize)]
pub struct ExecOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct PtyRequest {
    pub term: String,
    pub col_width: u32,
    pub row_height: u32,
    pub pix_width: u32,
    pub pix_height: u32,
    pub command: Option<String>,
}

impl Default for PtyRequest {
    fn default() -> Self {
        Self {
            term: "xterm-256color".to_string(),
            col_width: 80,
            row_height: 24,
            pix_width: 0,
            pix_height: 0,
            command: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum PtyTranscriptEvent {
    Sent(Vec<u8>),
    Received(Vec<u8>),
    Resized {
        col_width: u32,
        row_height: u32,
        pix_width: u32,
        pix_height: u32,
    },
    ExitStatus(u32),
    Eof,
    Close,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct PtyRead {
    pub output: String,
    pub bytes: Vec<u8>,
    pub exit_status: Option<u32>,
    pub eof: bool,
    pub closed: bool,
}

/// Configuration for the SSH proxy server.
#[derive(Debug, Clone)]
pub struct SshProxyConfig {
    pub listen: SocketAddr,
}

impl Default for SshProxyConfig {
    fn default() -> Self {
        Self {
            listen: SocketAddr::new(std::net::IpAddr::V4(Ipv4Addr::LOCALHOST), 2222),
        }
    }
}

// ---------------------------------------------------------------------------
// Shared guest registry
// ---------------------------------------------------------------------------

/// Per-guest connection info needed by the SSH proxy.
#[derive(Clone)]
pub struct GuestEndpoint {
    pub ssh_handle: Option<Arc<tokio::sync::Mutex<russh::client::Handle<GuestClientHandler>>>>,
}

impl std::fmt::Debug for GuestEndpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuestEndpoint")
            .field("ssh_handle", &self.ssh_handle.as_ref().map(|_| "..."))
            .finish()
    }
}

pub type GuestRegistry = Arc<Mutex<HashMap<String, GuestEndpoint>>>;

pub fn new_guest_registry() -> GuestRegistry {
    Arc::new(Mutex::new(HashMap::new()))
}

pub struct GuestBridgeHandle {
    guest_name: String,
    uds_path: PathBuf,
    registry: GuestRegistry,
    task: std::sync::Mutex<Option<JoinHandle<()>>>,
}

impl GuestBridgeHandle {
    pub async fn wait_ready(&self, timeout: Duration) -> Result<(), SshProxyError> {
        wait_for_guest_bridge_ready(&self.registry, &self.guest_name, timeout).await
    }

    pub async fn exec(
        &self,
        command: &str,
        timeout: Duration,
    ) -> Result<ExecOutput, SshProxyError> {
        exec_on_guest(&self.registry, &self.guest_name, command, timeout).await
    }

    pub async fn open_pty(
        &self,
        request: PtyRequest,
        timeout: Duration,
    ) -> Result<GuestPtySession, SshProxyError> {
        open_pty_on_guest(&self.registry, &self.guest_name, request, timeout).await
    }

    pub fn shutdown(&self) -> Result<(), SshProxyError> {
        if let Ok(mut task) = self.task.lock() {
            if let Some(task) = task.take() {
                task.abort();
            }
        }
        clear_guest_handle(&self.registry, &self.guest_name)?;
        if let Err(source) = std::fs::remove_file(&self.uds_path) {
            if source.kind() != ErrorKind::NotFound {
                return Err(SshProxyError::Ssh(format!(
                    "failed to remove guest bridge socket {}: {source}",
                    self.uds_path.display()
                )));
            }
        }
        Ok(())
    }

    pub fn uds_path(&self) -> &std::path::Path {
        &self.uds_path
    }
}

fn default_pty_modes() -> Vec<(Pty, u32)> {
    vec![
        (Pty::VINTR, 3),
        (Pty::VQUIT, 28),
        (Pty::VERASE, 127),
        (Pty::VKILL, 21),
        (Pty::VEOF, 4),
        (Pty::VWERASE, 23),
        (Pty::VLNEXT, 22),
        (Pty::VREPRINT, 18),
        (Pty::VSUSP, 26),
        (Pty::ICRNL, 1),
        (Pty::IXON, 1),
        (Pty::ISIG, 1),
        (Pty::ICANON, 1),
        (Pty::IEXTEN, 1),
        (Pty::ECHO, 1),
        (Pty::ECHOE, 1),
        (Pty::ECHOK, 1),
        (Pty::ECHOCTL, 1),
        (Pty::ECHOKE, 1),
        (Pty::OPOST, 1),
        (Pty::ONLCR, 1),
        (Pty::TTY_OP_ISPEED, 38_400),
        (Pty::TTY_OP_OSPEED, 38_400),
    ]
}

fn update_guest_handle(
    registry: &GuestRegistry,
    guest_name: &str,
    handle: Arc<tokio::sync::Mutex<russh::client::Handle<GuestClientHandler>>>,
) -> Result<(), SshProxyError> {
    let mut registry = registry
        .lock()
        .map_err(|_| SshProxyError::StatePoisoned("guest registry"))?;
    registry.insert(
        guest_name.to_string(),
        GuestEndpoint {
            ssh_handle: Some(handle),
        },
    );
    Ok(())
}

fn current_guest_handle(
    registry: &GuestRegistry,
    guest_name: &str,
) -> Result<Option<Arc<tokio::sync::Mutex<russh::client::Handle<GuestClientHandler>>>>, SshProxyError>
{
    let registry = registry
        .lock()
        .map_err(|_| SshProxyError::StatePoisoned("guest registry"))?;
    Ok(registry
        .get(guest_name)
        .and_then(|ep| ep.ssh_handle.clone()))
}

fn clear_guest_handle_if_matches(
    registry: &GuestRegistry,
    guest_name: &str,
    handle: &Arc<tokio::sync::Mutex<russh::client::Handle<GuestClientHandler>>>,
) -> Result<(), SshProxyError> {
    let mut registry = registry
        .lock()
        .map_err(|_| SshProxyError::StatePoisoned("guest registry"))?;
    if let Some(endpoint) = registry.get_mut(guest_name) {
        if endpoint
            .ssh_handle
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, handle))
        {
            endpoint.ssh_handle = None;
        }
    }
    Ok(())
}

fn clear_guest_handle(registry: &GuestRegistry, guest_name: &str) -> Result<(), SshProxyError> {
    let mut registry = registry
        .lock()
        .map_err(|_| SshProxyError::StatePoisoned("guest registry"))?;
    if let Some(endpoint) = registry.get_mut(guest_name) {
        endpoint.ssh_handle = None;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// vsock SSH bridge
// ---------------------------------------------------------------------------

pub fn vsock_ssh_uds_path(vsock_socket: &str) -> PathBuf {
    PathBuf::from(format!("{}_{}", vsock_socket, VSOCK_SSH_PORT))
}

pub fn bind_vsock_ssh_listener(
    vsock_socket: &str,
) -> Result<(UnixListener, PathBuf), SshProxyError> {
    let uds_path = vsock_ssh_uds_path(vsock_socket);
    if let Err(e) = std::fs::remove_file(&uds_path) {
        if e.kind() != ErrorKind::NotFound {
            tracing::debug!(
                "failed to remove stale SSH bridge socket {}: {e}",
                uds_path.display()
            );
        }
    }
    let listener = UnixListener::bind(&uds_path).map_err(|e| {
        SshProxyError::Ssh(format!(
            "failed to bind vsock SSH UDS {}: {e}",
            uds_path.display()
        ))
    })?;
    tracing::info!("Bound vsock SSH bridge UDS at {}", uds_path.display());
    Ok((listener, uds_path))
}

/// Accept one guest vsock SSH bridge connection and authenticate with a CA cert.
async fn accept_guest_ssh_once(
    listener: &UnixListener,
    uds_path: &std::path::Path,
    ca: &SshCa,
    guest_name: &str,
) -> Result<russh::client::Handle<GuestClientHandler>, SshProxyError> {
    tracing::info!(
        "Waiting for guest '{}' vsock SSH bridge on {}",
        guest_name,
        uds_path.display()
    );

    let (stream, _) = listener
        .accept()
        .await
        .map_err(|e| SshProxyError::GuestConnection {
            guest: guest_name.into(),
            reason: format!("vsock SSH accept failed: {e}"),
        })?;

    tracing::info!("Guest '{}' vsock SSH bridge connected", guest_name);

    let eph = ca.sign_ephemeral(guest_name)?;
    let config = Arc::new(russh::client::Config::default());

    let mut handle = russh::client::connect_stream(config, stream, GuestClientHandler)
        .await
        .map_err(|e| SshProxyError::GuestConnection {
            guest: guest_name.into(),
            reason: format!("SSH handshake failed: {e}"),
        })?;

    // Authenticate with CA-signed ephemeral certificate.
    let result = handle
        .authenticate_openssh_cert(guest_name, Arc::new(eph.key), eph.cert)
        .await
        .map_err(|e| SshProxyError::Ssh(format!("SSH cert auth failed: {e}")))?;

    if !result.success() {
        return Err(SshProxyError::GuestConnection {
            guest: guest_name.into(),
            reason: "guest sshd rejected CA-based auth".into(),
        });
    }

    tracing::info!("Guest '{}' SSH authenticated via CA cert", guest_name);
    Ok(handle)
}

/// Accept guest SSH bridge reconnects forever and keep the registry updated
/// with the most recent authenticated bridge handle.
pub async fn run_guest_ssh_bridge(
    listener: UnixListener,
    uds_path: PathBuf,
    ca: Arc<SshCa>,
    guest_name: String,
    registry: GuestRegistry,
) {
    loop {
        match accept_guest_ssh_once(&listener, &uds_path, &ca, &guest_name).await {
            Ok(handle) => {
                let handle = Arc::new(tokio::sync::Mutex::new(handle));
                match update_guest_handle(&registry, &guest_name, handle) {
                    Ok(()) => {
                        eprintln!("SSH bridge ready for guest '{guest_name}'");
                    }
                    Err(e) => {
                        eprintln!(
                            "SSH bridge registry update failed for guest '{guest_name}': {e}"
                        );
                        sleep(Duration::from_millis(250)).await;
                    }
                }
            }
            Err(e) => {
                eprintln!("SSH bridge failed for guest '{guest_name}': {e}");
                sleep(Duration::from_millis(250)).await;
            }
        }
    }
}

pub fn spawn_guest_ssh_bridge(
    vsock_socket: &str,
    ca: Arc<SshCa>,
    guest_name: String,
    registry: GuestRegistry,
) -> Result<GuestBridgeHandle, SshProxyError> {
    let (listener, uds_path) = bind_vsock_ssh_listener(vsock_socket)?;
    let task = tokio::spawn(run_guest_ssh_bridge(
        listener,
        uds_path.clone(),
        ca,
        guest_name.clone(),
        Arc::clone(&registry),
    ));
    Ok(GuestBridgeHandle {
        guest_name,
        uds_path,
        registry,
        task: std::sync::Mutex::new(Some(task)),
    })
}

async fn open_guest_session_with_retry(
    registry: &GuestRegistry,
    guest_name: &str,
    timeout: Duration,
) -> Result<Channel<russh::client::Msg>, SshProxyError> {
    let deadline = Instant::now() + timeout;
    let mut last_err: Option<String> = None;

    loop {
        let Some(handle) = current_guest_handle(registry, guest_name)? else {
            if Instant::now() >= deadline {
                return Err(SshProxyError::GuestConnection {
                    guest: guest_name.into(),
                    reason: last_err.unwrap_or_else(|| "SSH bridge not ready".into()),
                });
            }
            sleep(Duration::from_millis(100)).await;
            continue;
        };

        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(SshProxyError::GuestConnection {
                guest: guest_name.into(),
                reason: last_err.unwrap_or_else(|| "SSH channel open failed".into()),
            });
        }

        let open_result = tokio_timeout(remaining, async {
            let guard = handle.lock().await;
            guard.channel_open_session().await
        })
        .await
        .map_err(|_| SshProxyError::GuestConnection {
            guest: guest_name.into(),
            reason: "SSH channel open timed out".into(),
        })?;

        match open_result {
            Ok(channel) => return Ok(channel),
            Err(e) => {
                last_err = Some(e.to_string());
                clear_guest_handle_if_matches(registry, guest_name, &handle)?;
                if Instant::now() >= deadline {
                    return Err(SshProxyError::GuestConnection {
                        guest: guest_name.into(),
                        reason: last_err.unwrap_or_else(|| "SSH channel open failed".into()),
                    });
                }
                sleep(Duration::from_millis(100)).await;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// FR-7: Programmatic exec
// ---------------------------------------------------------------------------

pub async fn exec_on_handle(
    handle: &Arc<tokio::sync::Mutex<russh::client::Handle<GuestClientHandler>>>,
    guest_name: &str,
    command: &str,
) -> Result<ExecOutput, SshProxyError> {
    let channel = {
        let guard = handle.lock().await;
        guard.channel_open_session().await
    }
    .map_err(|e| SshProxyError::Ssh(e.to_string()))?;

    exec_on_channel(channel, guest_name, command).await
}

async fn exec_on_channel(
    mut channel: Channel<russh::client::Msg>,
    guest_name: &str,
    command: &str,
) -> Result<ExecOutput, SshProxyError> {
    channel
        .exec(true, command)
        .await
        .map_err(|e| SshProxyError::ExecFailed {
            guest: guest_name.into(),
            reason: e.to_string(),
        })?;

    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut exit_code: Option<u32> = None;

    while let Some(msg) = channel.wait().await {
        match msg {
            ChannelMsg::Data { ref data } => stdout.extend_from_slice(data),
            ChannelMsg::ExtendedData { ref data, ext } if ext == 1 => {
                stderr.extend_from_slice(data)
            }
            ChannelMsg::ExitStatus { exit_status } => exit_code = Some(exit_status),
            _ => {}
        }
    }

    Ok(ExecOutput {
        stdout: String::from_utf8_lossy(&stdout).into_owned(),
        stderr: String::from_utf8_lossy(&stderr).into_owned(),
        exit_code: exit_code.ok_or_else(|| SshProxyError::MissingExitStatus {
            guest: guest_name.into(),
            command: command.into(),
        })?,
    })
}

pub async fn wait_for_guest_bridge_ready(
    registry: &GuestRegistry,
    guest_name: &str,
    timeout: Duration,
) -> Result<(), SshProxyError> {
    let deadline = Instant::now() + timeout;
    loop {
        if current_guest_handle(registry, guest_name)?.is_some() {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(SshProxyError::GuestConnection {
                guest: guest_name.into(),
                reason: "SSH bridge not ready".into(),
            });
        }
        sleep(Duration::from_millis(100)).await;
    }
}

pub async fn exec_on_guest(
    registry: &GuestRegistry,
    guest_name: &str,
    command: &str,
    timeout: Duration,
) -> Result<ExecOutput, SshProxyError> {
    wait_for_guest_bridge_ready(registry, guest_name, timeout).await?;
    let started = Instant::now();
    let channel = open_guest_session_with_retry(registry, guest_name, timeout).await?;
    let remaining = timeout.saturating_sub(started.elapsed());
    if remaining.is_zero() {
        return Err(SshProxyError::ExecFailed {
            guest: guest_name.into(),
            reason: format!("command timed out after {:?}", timeout),
        });
    }
    tokio_timeout(remaining, exec_on_channel(channel, guest_name, command))
        .await
        .map_err(|_| SshProxyError::ExecFailed {
            guest: guest_name.into(),
            reason: format!("command timed out after {:?}", timeout),
        })?
}

pub struct GuestPtySession {
    guest_name: String,
    channel: Arc<tokio::sync::Mutex<Channel<russh::client::Msg>>>,
    transcript: Arc<Mutex<Vec<PtyTranscriptEvent>>>,
}

impl std::fmt::Debug for GuestPtySession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuestPtySession")
            .field("guest_name", &self.guest_name)
            .finish_non_exhaustive()
    }
}

impl GuestPtySession {
    pub async fn send(&self, data: &[u8]) -> Result<(), SshProxyError> {
        {
            let mut transcript = self
                .transcript
                .lock()
                .map_err(|_| SshProxyError::StatePoisoned("pty transcript"))?;
            transcript.push(PtyTranscriptEvent::Sent(data.to_vec()));
        }

        let channel = self.channel.lock().await;
        channel.data(data).await.map_err(SshProxyError::from)
    }

    pub async fn send_line(&self, line: &str) -> Result<(), SshProxyError> {
        let mut bytes = line.as_bytes().to_vec();
        bytes.push(b'\n');
        self.send(&bytes).await
    }

    pub async fn resize(
        &self,
        col_width: u32,
        row_height: u32,
        pix_width: u32,
        pix_height: u32,
    ) -> Result<(), SshProxyError> {
        {
            let mut transcript = self
                .transcript
                .lock()
                .map_err(|_| SshProxyError::StatePoisoned("pty transcript"))?;
            transcript.push(PtyTranscriptEvent::Resized {
                col_width,
                row_height,
                pix_width,
                pix_height,
            });
        }

        let channel = self.channel.lock().await;
        channel
            .window_change(col_width, row_height, pix_width, pix_height)
            .await
            .map_err(SshProxyError::from)
    }

    pub async fn read_for(&self, timeout: Duration) -> Result<PtyRead, SshProxyError> {
        let deadline = Instant::now() + timeout;
        let mut read = PtyRead::default();
        let mut output = Vec::new();

        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }

            let next = {
                let mut channel = self.channel.lock().await;
                tokio_timeout(remaining, channel.wait()).await
            };

            match next {
                Err(_) => break,
                Ok(None) => {
                    read.closed = true;
                    let mut transcript = self
                        .transcript
                        .lock()
                        .map_err(|_| SshProxyError::StatePoisoned("pty transcript"))?;
                    transcript.push(PtyTranscriptEvent::Close);
                    break;
                }
                Ok(Some(ChannelMsg::Data { ref data })) => {
                    output.extend_from_slice(data);
                    let mut transcript = self
                        .transcript
                        .lock()
                        .map_err(|_| SshProxyError::StatePoisoned("pty transcript"))?;
                    transcript.push(PtyTranscriptEvent::Received(data.to_vec()));
                }
                Ok(Some(ChannelMsg::ExtendedData { ref data, .. })) => {
                    output.extend_from_slice(data);
                    let mut transcript = self
                        .transcript
                        .lock()
                        .map_err(|_| SshProxyError::StatePoisoned("pty transcript"))?;
                    transcript.push(PtyTranscriptEvent::Received(data.to_vec()));
                }
                Ok(Some(ChannelMsg::ExitStatus { exit_status })) => {
                    read.exit_status = Some(exit_status);
                    let mut transcript = self
                        .transcript
                        .lock()
                        .map_err(|_| SshProxyError::StatePoisoned("pty transcript"))?;
                    transcript.push(PtyTranscriptEvent::ExitStatus(exit_status));
                }
                Ok(Some(ChannelMsg::Eof)) => {
                    read.eof = true;
                    let mut transcript = self
                        .transcript
                        .lock()
                        .map_err(|_| SshProxyError::StatePoisoned("pty transcript"))?;
                    transcript.push(PtyTranscriptEvent::Eof);
                    break;
                }
                Ok(Some(ChannelMsg::Close)) => {
                    read.closed = true;
                    let mut transcript = self
                        .transcript
                        .lock()
                        .map_err(|_| SshProxyError::StatePoisoned("pty transcript"))?;
                    transcript.push(PtyTranscriptEvent::Close);
                    break;
                }
                Ok(Some(_)) => {}
            }
        }

        read.bytes = output.clone();
        read.output = String::from_utf8_lossy(&output).into_owned();
        Ok(read)
    }

    pub async fn read_until_contains(
        &self,
        needle: &str,
        timeout: Duration,
    ) -> Result<PtyRead, SshProxyError> {
        let deadline = Instant::now() + timeout;
        let mut read = PtyRead::default();
        let mut output = Vec::new();

        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                read.bytes = output.clone();
                read.output = String::from_utf8_lossy(&output).into_owned();
                return Err(SshProxyError::PtyTimeout {
                    guest: self.guest_name.clone(),
                    expectation: format!("output containing '{needle}'"),
                });
            }

            let next = {
                let mut channel = self.channel.lock().await;
                tokio_timeout(remaining, channel.wait()).await
            };

            match next {
                Err(_) => {
                    read.output = String::from_utf8_lossy(&output).into_owned();
                    return Err(SshProxyError::PtyTimeout {
                        guest: self.guest_name.clone(),
                        expectation: format!("output containing '{needle}'"),
                    });
                }
                Ok(None) => {
                    read.closed = true;
                    let mut transcript = self
                        .transcript
                        .lock()
                        .map_err(|_| SshProxyError::StatePoisoned("pty transcript"))?;
                    transcript.push(PtyTranscriptEvent::Close);
                    break;
                }
                Ok(Some(ChannelMsg::Data { ref data })) => {
                    output.extend_from_slice(data);
                    let mut transcript = self
                        .transcript
                        .lock()
                        .map_err(|_| SshProxyError::StatePoisoned("pty transcript"))?;
                    transcript.push(PtyTranscriptEvent::Received(data.to_vec()));
                    let current = String::from_utf8_lossy(&output);
                    if current.contains(needle) {
                        read.bytes = output.clone();
                        read.output = current.into_owned();
                        return Ok(read);
                    }
                }
                Ok(Some(ChannelMsg::ExtendedData { ref data, .. })) => {
                    output.extend_from_slice(data);
                    let mut transcript = self
                        .transcript
                        .lock()
                        .map_err(|_| SshProxyError::StatePoisoned("pty transcript"))?;
                    transcript.push(PtyTranscriptEvent::Received(data.to_vec()));
                    let current = String::from_utf8_lossy(&output);
                    if current.contains(needle) {
                        read.bytes = output.clone();
                        read.output = current.into_owned();
                        return Ok(read);
                    }
                }
                Ok(Some(ChannelMsg::ExitStatus { exit_status })) => {
                    read.exit_status = Some(exit_status);
                    let mut transcript = self
                        .transcript
                        .lock()
                        .map_err(|_| SshProxyError::StatePoisoned("pty transcript"))?;
                    transcript.push(PtyTranscriptEvent::ExitStatus(exit_status));
                }
                Ok(Some(ChannelMsg::Eof)) => {
                    read.eof = true;
                    let mut transcript = self
                        .transcript
                        .lock()
                        .map_err(|_| SshProxyError::StatePoisoned("pty transcript"))?;
                    transcript.push(PtyTranscriptEvent::Eof);
                    break;
                }
                Ok(Some(ChannelMsg::Close)) => {
                    read.closed = true;
                    let mut transcript = self
                        .transcript
                        .lock()
                        .map_err(|_| SshProxyError::StatePoisoned("pty transcript"))?;
                    transcript.push(PtyTranscriptEvent::Close);
                    break;
                }
                Ok(Some(_)) => {}
            }
        }

        read.bytes = output.clone();
        read.output = String::from_utf8_lossy(&output).into_owned();
        Err(SshProxyError::PtyTimeout {
            guest: self.guest_name.clone(),
            expectation: format!("output containing '{needle}'"),
        })
    }

    pub fn transcript(&self) -> Result<Vec<PtyTranscriptEvent>, SshProxyError> {
        let transcript = self
            .transcript
            .lock()
            .map_err(|_| SshProxyError::StatePoisoned("pty transcript"))?;
        Ok(transcript.clone())
    }

    pub async fn close(&self) -> Result<(), SshProxyError> {
        {
            let mut transcript = self
                .transcript
                .lock()
                .map_err(|_| SshProxyError::StatePoisoned("pty transcript"))?;
            transcript.push(PtyTranscriptEvent::Close);
        }

        let channel = self.channel.lock().await;
        channel.close().await.map_err(SshProxyError::from)
    }
}

pub async fn open_pty_on_guest(
    registry: &GuestRegistry,
    guest_name: &str,
    request: PtyRequest,
    timeout: Duration,
) -> Result<GuestPtySession, SshProxyError> {
    wait_for_guest_bridge_ready(registry, guest_name, timeout).await?;
    let started = Instant::now();
    let channel = open_guest_session_with_retry(registry, guest_name, timeout).await?;
    let remaining = timeout.saturating_sub(started.elapsed());
    if remaining.is_zero() {
        return Err(SshProxyError::PtyTimeout {
            guest: guest_name.into(),
            expectation: "PTY open".into(),
        });
    }

    tokio_timeout(remaining, async {
        channel
            .request_pty(
                true,
                &request.term,
                request.col_width,
                request.row_height,
                request.pix_width,
                request.pix_height,
                &default_pty_modes(),
            )
            .await?;
        if let Some(command) = &request.command {
            channel.exec(true, command.clone()).await?;
        } else {
            channel.request_shell(true).await?;
        }
        Ok::<_, russh::Error>(channel)
    })
    .await
    .map_err(|_| SshProxyError::PtyTimeout {
        guest: guest_name.into(),
        expectation: "PTY request".into(),
    })?
    .map(|channel| GuestPtySession {
        guest_name: guest_name.to_string(),
        channel: Arc::new(tokio::sync::Mutex::new(channel)),
        transcript: Arc::new(Mutex::new(Vec::new())),
    })
    .map_err(SshProxyError::from)
}

// ---------------------------------------------------------------------------
// FR-6: SSH proxy server
// ---------------------------------------------------------------------------

pub async fn run_proxy(
    config: SshProxyConfig,
    registry: GuestRegistry,
) -> Result<(), SshProxyError> {
    let server_key =
        russh::keys::PrivateKey::random(&mut rand::rng(), russh::keys::Algorithm::Ed25519)
            .map_err(|e| SshProxyError::Ssh(format!("failed to generate server key: {e}")))?;

    let russh_config = Arc::new(russh::server::Config {
        keys: vec![server_key],
        ..Default::default()
    });

    let listener = tokio::net::TcpListener::bind(config.listen)
        .await
        .map_err(|e| SshProxyError::Ssh(format!("failed to bind {}: {e}", config.listen)))?;

    tracing::info!("SSH proxy listening on {}", config.listen);
    eprintln!("SSH proxy: listening on {}", config.listen);
    eprintln!("  ssh -p {} <guest>@localhost", config.listen.port());

    loop {
        let (stream, peer) = match listener.accept().await {
            Ok(conn) => conn,
            Err(e) => {
                tracing::warn!("SSH proxy accept error: {e}");
                continue;
            }
        };

        tracing::info!("SSH proxy: new connection from {peer}");

        let handler = ProxyHandler {
            registry: Arc::clone(&registry),
            username: None,
            guest_channels: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
        };

        let cfg = Arc::clone(&russh_config);
        tokio::spawn(async move {
            match russh::server::run_stream(cfg, stream, handler).await {
                Ok(_) => tracing::info!("SSH proxy: session from {peer} ended"),
                Err(e) => tracing::warn!("SSH proxy: session from {peer} failed: {e:?}"),
            }
        });
    }
}

/// Shared guest channel write half, set by channel_open_session,
/// used by handler methods for client→guest forwarding on the same SSH channel
/// whose read half is being forwarded back to the client.
#[derive(Debug, Clone, Copy)]
enum PendingReplyKind {
    Pty,
    Shell,
    Exec,
}

#[derive(Debug, Clone, Copy)]
struct PendingReply {
    client_channel: russh::ChannelId,
    #[cfg_attr(not(feature = "debug-trace"), allow(dead_code))]
    kind: PendingReplyKind,
}

#[derive(Default)]
struct ProxyChannelState {
    writer: Option<ChannelWriteHalf<russh::client::Msg>>,
    pending_replies: VecDeque<PendingReply>,
}

type SharedGuestChannels = Arc<tokio::sync::Mutex<HashMap<russh::ChannelId, ProxyChannelState>>>;

struct ProxyHandler {
    registry: GuestRegistry,
    username: Option<String>,
    /// Per-client-channel proxied guest state. Each inbound localhost SSH
    /// session channel gets its own guest SSH channel and ordered queue of
    /// pending want-reply requests whose actual Success/Failure arrives later
    /// from the guest channel. SSH replies are ordered per channel, so a
    /// FIFO queue is sufficient as long as one proxied guest channel maps to
    /// one inbound client channel.
    guest_channels: SharedGuestChannels,
}

const PROXY_LOGIN_SHELL_COMMAND: &str =
    r#"if [ -r /etc/motd ]; then cat /etc/motd; fi; exec "${SHELL:-/bin/bash}" -l"#;

impl russh::server::Handler for ProxyHandler {
    type Error = SshProxyError;

    fn auth_none(
        &mut self,
        user: &str,
    ) -> impl std::future::Future<Output = Result<Auth, Self::Error>> + Send {
        self.username = Some(user.to_string());
        tracing::info!("SSH proxy: auth_none user={user}");
        async { Ok(Auth::Accept) }
    }

    fn auth_publickey_offered(
        &mut self,
        user: &str,
        _public_key: &russh::keys::PublicKey,
    ) -> impl std::future::Future<Output = Result<Auth, Self::Error>> + Send {
        self.username = Some(user.to_string());
        tracing::info!("SSH proxy: auth_publickey_offered user={user}");
        async { Ok(Auth::Accept) }
    }

    fn auth_publickey(
        &mut self,
        user: &str,
        _public_key: &russh::keys::PublicKey,
    ) -> impl std::future::Future<Output = Result<Auth, Self::Error>> + Send {
        self.username = Some(user.to_string());
        tracing::info!("SSH proxy: auth_publickey user={user}");
        async { Ok(Auth::Accept) }
    }

    fn channel_open_session(
        &mut self,
        channel: Channel<Msg>,
        session: &mut Session,
    ) -> impl std::future::Future<Output = Result<bool, Self::Error>> + Send {
        let username = self.username.clone();
        let registry = Arc::clone(&self.registry);
        let guest_channels = Arc::clone(&self.guest_channels);
        let server_handle = session.handle();
        let client_ch_id = channel.id();

        async move {
            let username = match username {
                Some(u) => u,
                None => return Ok(false),
            };

            if current_guest_handle(&registry, &username)?.is_none() {
                let msg = format!(
                    "Guest '{}' is not launched or SSH bridge not ready.\r\n",
                    username
                );
                let _ = channel.data(msg.as_bytes()).await;
                let _ = channel.close().await;
                return Ok(false);
            }

            tracing::info!("SSH proxy: opening session to guest '{username}'");

            let guest_ch =
                match open_guest_session_with_retry(&registry, &username, Duration::from_secs(3))
                    .await
                {
                    Ok(ch) => ch,
                    Err(SshProxyError::GuestConnection { .. }) => {
                        let msg = format!(
                            "Guest '{}' is not launched or SSH bridge not ready.\r\n",
                            username
                        );
                        let _ = channel.data(msg.as_bytes()).await;
                        let _ = channel.close().await;
                        return Ok(false);
                    }
                    Err(e) => {
                        let _ = channel.data(&b"internal SSH proxy error\r\n"[..]).await;
                        let _ = channel.close().await;
                        return Err(e);
                    }
                };

            let (mut guest_read, guest_write) = guest_ch.split();
            let guest_channels_for_read = Arc::clone(&guest_channels);

            // Spawn guest→client forwarding via the server Handle.
            let ch_id = client_ch_id;
            tokio::spawn(async move {
                while let Some(msg) = guest_read.wait().await {
                    match msg {
                        ChannelMsg::Data { ref data } => {
                            let bytes = bytes::Bytes::copy_from_slice(data);
                            if server_handle.data(ch_id, bytes).await.is_err() {
                                break;
                            }
                        }
                        ChannelMsg::ExtendedData { ref data, ext } if ext == 1 => {
                            let bytes = bytes::Bytes::copy_from_slice(data);
                            if server_handle
                                .extended_data(ch_id, ext, bytes)
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                        ChannelMsg::Success => {
                            debug_trace!("SSH proxy: guest channel reported Success");
                            let pending = {
                                let mut states = guest_channels_for_read.lock().await;
                                states
                                    .get_mut(&ch_id)
                                    .and_then(|state| state.pending_replies.pop_front())
                            };
                            if let Some(pending) = pending {
                                debug_trace!(
                                    "SSH proxy: forwarding {:?} success to client channel {:?}",
                                    pending.kind,
                                    pending.client_channel
                                );
                                if server_handle
                                    .channel_success(pending.client_channel)
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                            }
                        }
                        ChannelMsg::Failure => {
                            debug_trace!("SSH proxy: guest channel reported Failure");
                            let pending = {
                                let mut states = guest_channels_for_read.lock().await;
                                states
                                    .get_mut(&ch_id)
                                    .and_then(|state| state.pending_replies.pop_front())
                            };
                            if let Some(pending) = pending {
                                debug_trace!(
                                    "SSH proxy: forwarding {:?} failure to client channel {:?}",
                                    pending.kind,
                                    pending.client_channel
                                );
                                if server_handle
                                    .channel_failure(pending.client_channel)
                                    .await
                                    .is_err()
                                {
                                    break;
                                }
                            }
                        }
                        ChannelMsg::ExitStatus { exit_status } => {
                            debug_trace!("SSH proxy: guest channel exit-status={exit_status}");
                            if server_handle
                                .exit_status_request(ch_id, exit_status)
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                        ChannelMsg::ExitSignal {
                            signal_name,
                            core_dumped,
                            ref error_message,
                            ref lang_tag,
                        } => {
                            debug_trace!("SSH proxy: guest channel exit-signal={signal_name:?}");
                            if server_handle
                                .exit_signal_request(
                                    ch_id,
                                    signal_name,
                                    core_dumped,
                                    error_message.to_string(),
                                    lang_tag.to_string(),
                                )
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                        ChannelMsg::Eof => {
                            if server_handle.eof(ch_id).await.is_err() {
                                break;
                            }
                        }
                        ChannelMsg::Close => {
                            if server_handle.close(ch_id).await.is_err() {
                                break;
                            }
                            break;
                        }
                        _ => {}
                    }
                }
                let mut states = guest_channels_for_read.lock().await;
                states.remove(&ch_id);
                tracing::debug!("SSH proxy: guest→client read loop ended");
            });

            let mut states = guest_channels.lock().await;
            states.insert(
                client_ch_id,
                ProxyChannelState {
                    writer: Some(guest_write),
                    pending_replies: VecDeque::new(),
                },
            );

            Ok(true)
        }
    }

    fn data(
        &mut self,
        channel: russh::ChannelId,
        data: &[u8],
        _session: &mut Session,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send {
        let guest_channels = Arc::clone(&self.guest_channels);
        let data = data.to_vec();
        async move {
            let states = guest_channels.lock().await;
            if let Some(state) = states.get(&channel) {
                if let Some(ref ch) = state.writer {
                    ch.data(&data[..]).await?;
                }
            }
            Ok(())
        }
    }

    fn pty_request(
        &mut self,
        channel: russh::ChannelId,
        term: &str,
        col_width: u32,
        row_height: u32,
        pix_width: u32,
        pix_height: u32,
        _modes: &[(russh::Pty, u32)],
        session: &mut Session,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send {
        let guest_channels = Arc::clone(&self.guest_channels);
        let term = term.to_string();
        #[cfg(feature = "debug-trace")]
        let incoming_mode_count = _modes
            .iter()
            .copied()
            .take_while(|(mode, _)| *mode != Pty::TTY_OP_END)
            .count();
        let modes = default_pty_modes();
        debug_trace!(
            "SSH proxy: forwarding pty term={} rows={} cols={} incoming_modes={} effective_modes={}",
            term,
            row_height,
            col_width,
            incoming_mode_count,
            modes.len()
        );
        let server_handle = session.handle();
        async move {
            let mut states = guest_channels.lock().await;
            if let Some(state) = states.get_mut(&channel) {
                if let Some(ref ch) = state.writer {
                    if let Err(e) = ch
                        .request_pty(
                            true, &term, col_width, row_height, pix_width, pix_height, &modes,
                        )
                        .await
                    {
                        let _ = server_handle.channel_failure(channel).await;
                        return Err(e.into());
                    }
                    state.pending_replies.push_back(PendingReply {
                        client_channel: channel,
                        kind: PendingReplyKind::Pty,
                    });
                } else {
                    let _ = server_handle.channel_failure(channel).await;
                }
            } else {
                let _ = server_handle.channel_failure(channel).await;
            }
            Ok(())
        }
    }

    fn shell_request(
        &mut self,
        channel: russh::ChannelId,
        session: &mut Session,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send {
        let guest_channels = Arc::clone(&self.guest_channels);
        let server_handle = session.handle();
        async move {
            let mut states = guest_channels.lock().await;
            if let Some(state) = states.get_mut(&channel) {
                if let Some(ref ch) = state.writer {
                    if let Err(e) = ch.exec(true, PROXY_LOGIN_SHELL_COMMAND).await {
                        let _ = server_handle.channel_failure(channel).await;
                        return Err(e.into());
                    }
                    state.pending_replies.push_back(PendingReply {
                        client_channel: channel,
                        kind: PendingReplyKind::Shell,
                    });
                } else {
                    let _ = server_handle.channel_failure(channel).await;
                }
            } else {
                let _ = server_handle.channel_failure(channel).await;
            }
            Ok(())
        }
    }

    fn exec_request(
        &mut self,
        channel: russh::ChannelId,
        data: &[u8],
        session: &mut Session,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send {
        let guest_channels = Arc::clone(&self.guest_channels);
        let data = data.to_vec();
        let server_handle = session.handle();
        async move {
            let mut states = guest_channels.lock().await;
            if let Some(state) = states.get_mut(&channel) {
                if let Some(ref ch) = state.writer {
                    if let Err(e) = ch.exec(true, &data[..]).await {
                        let _ = server_handle.channel_failure(channel).await;
                        return Err(e.into());
                    }
                    state.pending_replies.push_back(PendingReply {
                        client_channel: channel,
                        kind: PendingReplyKind::Exec,
                    });
                } else {
                    let _ = server_handle.channel_failure(channel).await;
                }
            } else {
                let _ = server_handle.channel_failure(channel).await;
            }
            Ok(())
        }
    }

    fn window_change_request(
        &mut self,
        channel: russh::ChannelId,
        col_width: u32,
        row_height: u32,
        pix_width: u32,
        pix_height: u32,
        _session: &mut Session,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send {
        let guest_channels = Arc::clone(&self.guest_channels);
        async move {
            let states = guest_channels.lock().await;
            if let Some(state) = states.get(&channel) {
                if let Some(ref ch) = state.writer {
                    ch.window_change(col_width, row_height, pix_width, pix_height)
                        .await?;
                }
            }
            Ok(())
        }
    }

    fn channel_eof(
        &mut self,
        channel: russh::ChannelId,
        _session: &mut Session,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send {
        let guest_channels = Arc::clone(&self.guest_channels);
        async move {
            let states = guest_channels.lock().await;
            if let Some(state) = states.get(&channel) {
                if let Some(ref ch) = state.writer {
                    ch.eof().await?;
                }
            }
            Ok(())
        }
    }

    fn channel_close(
        &mut self,
        channel: russh::ChannelId,
        _session: &mut Session,
    ) -> impl std::future::Future<Output = Result<(), Self::Error>> + Send {
        let guest_channels = Arc::clone(&self.guest_channels);
        async move {
            let mut states = guest_channels.lock().await;
            if let Some(mut state) = states.remove(&channel) {
                if let Some(ch) = state.writer.take() {
                    ch.close().await?;
                }
                state.pending_replies.clear();
            }
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Client handler for guest connections
// ---------------------------------------------------------------------------

pub struct GuestClientHandler;

impl russh::client::Handler for GuestClientHandler {
    type Error = SshProxyError;

    fn check_server_key(
        &mut self,
        _server_public_key: &russh::keys::PublicKey,
    ) -> impl std::future::Future<Output = Result<bool, Self::Error>> + Send {
        async { Ok(true) }
    }
}
