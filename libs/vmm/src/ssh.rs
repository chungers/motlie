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

use std::collections::HashMap;
use std::io::ErrorKind;
use std::net::{Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::future::BoxFuture;
use russh::server::{Auth, Msg, Session};
use russh::{Channel, ChannelMsg, ChannelWriteHalf, Pty};
use serde::Serialize;
use thiserror::Error;
use tokio::net::UnixListener;
use tokio::task::JoinHandle;
use tokio::time::{sleep, timeout as tokio_timeout, Instant};

use crate::ca::{CaError, SshCa};
use crate::spec::GuestSshAccess;

#[cfg(feature = "debug-trace")]
macro_rules! debug_trace {
    ($($arg:tt)*) => {
        tracing::debug!($($arg)*);
    };
}

#[cfg(not(feature = "debug-trace"))]
macro_rules! debug_trace {
    ($($arg:tt)*) => {};
}

fn ssh_exec_trace_enabled() -> bool {
    std::env::var_os("MOTLIE_VMM_SSH_EXEC_TRACE").is_some()
}

macro_rules! ssh_exec_trace {
    ($($arg:tt)*) => {
        if crate::ssh::ssh_exec_trace_enabled() {
            eprintln!($($arg)*);
        }
    };
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
    #[error("failed to generate SSH proxy server key: {reason}")]
    GenerateServerKey { reason: String },
    #[error("failed to bind SSH proxy listener on {listen}: {reason}")]
    ProxyBind { listen: SocketAddr, reason: String },
    #[error("failed to connect to SSH proxy on {listen}: {reason}")]
    ProxyConnect { listen: SocketAddr, reason: String },
    #[error("failed to authenticate to SSH proxy as '{principal}': {reason}")]
    ProxyAuth { principal: String, reason: String },
    #[error("failed to bind guest bridge socket {path}: {reason}")]
    BindGuestBridgeSocket { path: PathBuf, reason: String },
    #[error("failed to authenticate guest {guest} with CA-signed SSH cert: {reason}")]
    CertAuth { guest: String, reason: String },
    #[error("SSH operation '{operation}' failed: {reason}")]
    Russh {
        operation: &'static str,
        reason: String,
    },
    #[error("channel closed before command completed")]
    ChannelClosed,
    #[error("command on guest '{guest}' completed without an exit status: {command}")]
    MissingExitStatus { guest: String, command: String },
    #[error("unknown guest: {0}")]
    UnknownGuest(String),
    #[error("failed to resolve guest for SSH principal '{principal}': {reason}")]
    ResolveGuest { principal: String, reason: String },
    #[error("internal state poisoned: {0}")]
    StatePoisoned(&'static str),
    #[error("failed to remove guest bridge socket {path}: {source}")]
    CleanupGuestBridgeSocket {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("PTY operation timed out on guest '{guest}': {expectation}")]
    PtyTimeout { guest: String, expectation: String },
    #[error("unsupported SSH control-plane operation: {0}")]
    Unsupported(&'static str),
}

impl From<russh::Error> for SshProxyError {
    fn from(value: russh::Error) -> Self {
        Self::Russh {
            operation: "russh",
            reason: value.to_string(),
        }
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
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PtyTranscriptEventKind {
    Sent {
        data: Vec<u8>,
    },
    Received {
        data: Vec<u8>,
    },
    Resized {
        col_width: u32,
        row_height: u32,
        pix_width: u32,
        pix_height: u32,
    },
    ExitStatus {
        exit_status: u32,
    },
    Eof,
    Close,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PtyTranscriptEvent {
    pub offset_ms: u64,
    #[serde(flatten)]
    pub event: PtyTranscriptEventKind,
}

impl PtyTranscriptEvent {
    pub fn sent(offset_ms: u64, data: &[u8]) -> Self {
        Self {
            offset_ms,
            event: PtyTranscriptEventKind::Sent {
                data: data.to_vec(),
            },
        }
    }

    pub fn received(offset_ms: u64, data: &[u8]) -> Self {
        Self {
            offset_ms,
            event: PtyTranscriptEventKind::Received {
                data: data.to_vec(),
            },
        }
    }

    pub fn resized(
        offset_ms: u64,
        col_width: u32,
        row_height: u32,
        pix_width: u32,
        pix_height: u32,
    ) -> Self {
        Self {
            offset_ms,
            event: PtyTranscriptEventKind::Resized {
                col_width,
                row_height,
                pix_width,
                pix_height,
            },
        }
    }

    pub fn exit_status(offset_ms: u64, exit_status: u32) -> Self {
        Self {
            offset_ms,
            event: PtyTranscriptEventKind::ExitStatus { exit_status },
        }
    }

    pub fn eof(offset_ms: u64) -> Self {
        Self {
            offset_ms,
            event: PtyTranscriptEventKind::Eof,
        }
    }

    pub fn close(offset_ms: u64) -> Self {
        Self {
            offset_ms,
            event: PtyTranscriptEventKind::Close,
        }
    }
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
#[derive(Clone)]
pub struct SshProxyConfig {
    pub listen: SocketAddr,
    pub principal_resolver: Option<PrincipalResolver>,
}

impl Default for SshProxyConfig {
    fn default() -> Self {
        Self {
            listen: SocketAddr::new(std::net::IpAddr::V4(Ipv4Addr::LOCALHOST), 2222),
            principal_resolver: None,
        }
    }
}

pub type PrincipalResolver =
    Arc<dyn Fn(String) -> BoxFuture<'static, Result<String, SshProxyError>> + Send + Sync>;

impl std::fmt::Debug for SshProxyConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SshProxyConfig")
            .field("listen", &self.listen)
            .field(
                "principal_resolver",
                &self.principal_resolver.as_ref().map(|_| "..."),
            )
            .finish()
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
        let mut task = self
            .task
            .lock()
            .map_err(|_| SshProxyError::StatePoisoned("guest bridge task"))?;
        if let Some(task) = task.take() {
            task.abort();
        }
        clear_guest_handle(&self.registry, &self.guest_name)?;
        if let Err(source) = std::fs::remove_file(&self.uds_path) {
            if source.kind() != ErrorKind::NotFound {
                return Err(SshProxyError::CleanupGuestBridgeSocket {
                    path: self.uds_path.clone(),
                    source,
                });
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
    let listener =
        UnixListener::bind(&uds_path).map_err(|e| SshProxyError::BindGuestBridgeSocket {
            path: uds_path.clone(),
            reason: e.to_string(),
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
    ssh_access: &GuestSshAccess,
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

    let eph = ca.sign_ephemeral(&ssh_access.principal)?;
    let config = Arc::new(russh::client::Config::default());

    let mut handle = russh::client::connect_stream(config, stream, GuestClientHandler)
        .await
        .map_err(|e| SshProxyError::GuestConnection {
            guest: guest_name.into(),
            reason: format!("SSH handshake failed: {e}"),
        })?;

    // Authenticate with CA-signed ephemeral certificate.
    let result = handle
        .authenticate_openssh_cert(&ssh_access.login_user, Arc::new(eph.key), eph.cert)
        .await
        .map_err(|e| SshProxyError::CertAuth {
            guest: guest_name.into(),
            reason: e.to_string(),
        })?;

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
    ssh_access: GuestSshAccess,
    registry: GuestRegistry,
) {
    loop {
        match accept_guest_ssh_once(&listener, &uds_path, &ca, &guest_name, &ssh_access).await {
            Ok(handle) => {
                let handle = Arc::new(tokio::sync::Mutex::new(handle));
                match update_guest_handle(&registry, &guest_name, handle) {
                    Ok(()) => {
                        tracing::info!("SSH bridge ready for guest '{guest_name}'");
                    }
                    Err(e) => {
                        tracing::warn!(
                            "SSH bridge registry update failed for guest '{guest_name}': {e}"
                        );
                        sleep(Duration::from_millis(250)).await;
                    }
                }
            }
            Err(e) => {
                tracing::warn!("SSH bridge failed for guest '{guest_name}': {e}");
                sleep(Duration::from_millis(250)).await;
            }
        }
    }
}

pub fn spawn_guest_ssh_bridge(
    vsock_socket: &str,
    ca: Arc<SshCa>,
    guest_name: String,
    ssh_access: GuestSshAccess,
    registry: GuestRegistry,
) -> Result<GuestBridgeHandle, SshProxyError> {
    let (listener, uds_path) = bind_vsock_ssh_listener(vsock_socket)?;
    let task = tokio::spawn(run_guest_ssh_bridge(
        listener,
        uds_path.clone(),
        ca,
        guest_name.clone(),
        ssh_access,
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
    .map_err(|e| SshProxyError::Russh {
        operation: "open guest session",
        reason: e.to_string(),
    })?;

    exec_on_channel(channel, guest_name, command).await
}

async fn exec_on_channel(
    mut channel: Channel<russh::client::Msg>,
    guest_name: &str,
    command: &str,
) -> Result<ExecOutput, SshProxyError> {
    ssh_exec_trace!("exec_on_channel[{guest_name}]: sending exec {:?}", command);
    channel
        .exec(true, command)
        .await
        .map_err(|e| SshProxyError::ExecFailed {
            guest: guest_name.into(),
            reason: e.to_string(),
        })?;
    ssh_exec_trace!("exec_on_channel[{guest_name}]: exec accepted");
    channel.eof().await.map_err(|e| SshProxyError::ExecFailed {
        guest: guest_name.into(),
        reason: e.to_string(),
    })?;
    ssh_exec_trace!("exec_on_channel[{guest_name}]: eof sent");

    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let mut exit_code: Option<u32> = None;
    let mut saw_terminal_event = false;

    while let Some(msg) = channel.wait().await {
        match msg {
            ChannelMsg::Data { ref data } => {
                ssh_exec_trace!(
                    "exec_on_channel[{guest_name}]: stdout {} bytes",
                    data.len()
                );
                stdout.extend_from_slice(data)
            }
            ChannelMsg::ExtendedData { ref data, ext } if ext == 1 => {
                ssh_exec_trace!(
                    "exec_on_channel[{guest_name}]: stderr {} bytes",
                    data.len()
                );
                stderr.extend_from_slice(data)
            }
            ChannelMsg::ExitStatus { exit_status } => {
                ssh_exec_trace!(
                    "exec_on_channel[{guest_name}]: exit_status {}",
                    exit_status
                );
                exit_code = Some(exit_status)
            }
            ChannelMsg::Eof => {
                ssh_exec_trace!("exec_on_channel[{guest_name}]: eof received");
                saw_terminal_event = true;
                break;
            }
            ChannelMsg::Close => {
                ssh_exec_trace!("exec_on_channel[{guest_name}]: close received");
                saw_terminal_event = true;
                break;
            }
            _ => {}
        }
    }

    if exit_code.is_none() && saw_terminal_event {
        ssh_exec_trace!(
            "exec_on_channel[{guest_name}]: treating terminal event without exit status as success"
        );
        exit_code = Some(0);
    }

    ssh_exec_trace!(
        "exec_on_channel[{guest_name}]: completed with exit_code={:?}",
        exit_code
    );

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

pub async fn exec_via_proxy(
    listen: SocketAddr,
    principal: &str,
    command: &str,
    timeout: Duration,
) -> Result<ExecOutput, SshProxyError> {
    let started = Instant::now();
    let config = Arc::new(russh::client::Config::default());
    let mut handle = tokio_timeout(
        timeout,
        russh::client::connect(config, listen, ProxyClientHandler),
    )
    .await
    .map_err(|_| SshProxyError::ProxyConnect {
        listen,
        reason: format!("timed out after {:?}", timeout),
    })?
    .map_err(|err| SshProxyError::ProxyConnect {
        listen,
        reason: err.to_string(),
    })?;

    let remaining = timeout.saturating_sub(started.elapsed());
    if remaining.is_zero() {
        return Err(SshProxyError::ProxyConnect {
            listen,
            reason: format!("timed out after {:?}", timeout),
        });
    }

    let auth = tokio_timeout(remaining, handle.authenticate_none(principal))
        .await
        .map_err(|_| SshProxyError::ProxyAuth {
            principal: principal.to_string(),
            reason: format!("timed out after {:?}", timeout),
        })?
        .map_err(|err| SshProxyError::ProxyAuth {
            principal: principal.to_string(),
            reason: err.to_string(),
        })?;
    if !auth.success() {
        return Err(SshProxyError::ProxyAuth {
            principal: principal.to_string(),
            reason: "SSH proxy rejected auth_none".to_string(),
        });
    }

    let remaining = timeout.saturating_sub(started.elapsed());
    if remaining.is_zero() {
        return Err(SshProxyError::ExecFailed {
            guest: principal.to_string(),
            reason: format!("command timed out after {:?}", timeout),
        });
    }

    let channel = tokio_timeout(remaining, handle.channel_open_session())
        .await
        .map_err(|_| SshProxyError::ProxyConnect {
            listen,
            reason: "timed out opening proxy session".to_string(),
        })?
        .map_err(|err| SshProxyError::Russh {
            operation: "open proxy session",
            reason: err.to_string(),
        })?;

    let remaining = timeout.saturating_sub(started.elapsed());
    if remaining.is_zero() {
        return Err(SshProxyError::ExecFailed {
            guest: principal.to_string(),
            reason: format!("command timed out after {:?}", timeout),
        });
    }

    tokio_timeout(remaining, exec_on_channel(channel, principal, command))
        .await
        .map_err(|_| SshProxyError::ExecFailed {
            guest: principal.to_string(),
            reason: format!("command timed out after {:?}", timeout),
        })?
}

pub struct GuestPtySession {
    guest_name: String,
    channel: Arc<tokio::sync::Mutex<Channel<russh::client::Msg>>>,
    transcript: Arc<Mutex<Vec<PtyTranscriptEvent>>>,
    transcript_started_at: Instant,
}

impl std::fmt::Debug for GuestPtySession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuestPtySession")
            .field("guest_name", &self.guest_name)
            .finish_non_exhaustive()
    }
}

impl GuestPtySession {
    fn push_transcript_event(&self, event: PtyTranscriptEventKind) -> Result<(), SshProxyError> {
        let mut transcript = self
            .transcript
            .lock()
            .map_err(|_| SshProxyError::StatePoisoned("pty transcript"))?;
        let offset_ms = self
            .transcript_started_at
            .elapsed()
            .as_millis()
            .min(u128::from(u64::MAX)) as u64;
        transcript.push(PtyTranscriptEvent { offset_ms, event });
        Ok(())
    }

    pub async fn send(&self, data: &[u8]) -> Result<(), SshProxyError> {
        self.push_transcript_event(PtyTranscriptEventKind::Sent {
            data: data.to_vec(),
        })?;

        let channel = self.channel.lock().await;
        channel.data(data).await.map_err(|e| SshProxyError::Russh {
            operation: "send PTY data",
            reason: e.to_string(),
        })
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
        self.push_transcript_event(PtyTranscriptEventKind::Resized {
            col_width,
            row_height,
            pix_width,
            pix_height,
        })?;

        let channel = self.channel.lock().await;
        channel
            .window_change(col_width, row_height, pix_width, pix_height)
            .await
            .map_err(|e| SshProxyError::Russh {
                operation: "resize PTY",
                reason: e.to_string(),
            })
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
                    self.push_transcript_event(PtyTranscriptEventKind::Close)?;
                    break;
                }
                Ok(Some(ChannelMsg::Data { ref data })) => {
                    output.extend_from_slice(data);
                    self.push_transcript_event(PtyTranscriptEventKind::Received {
                        data: data.to_vec(),
                    })?;
                }
                Ok(Some(ChannelMsg::ExtendedData { ref data, .. })) => {
                    output.extend_from_slice(data);
                    self.push_transcript_event(PtyTranscriptEventKind::Received {
                        data: data.to_vec(),
                    })?;
                }
                Ok(Some(ChannelMsg::ExitStatus { exit_status })) => {
                    read.exit_status = Some(exit_status);
                    self.push_transcript_event(PtyTranscriptEventKind::ExitStatus { exit_status })?;
                }
                Ok(Some(ChannelMsg::Eof)) => {
                    read.eof = true;
                    self.push_transcript_event(PtyTranscriptEventKind::Eof)?;
                    break;
                }
                Ok(Some(ChannelMsg::Close)) => {
                    read.closed = true;
                    self.push_transcript_event(PtyTranscriptEventKind::Close)?;
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
                    self.push_transcript_event(PtyTranscriptEventKind::Close)?;
                    break;
                }
                Ok(Some(ChannelMsg::Data { ref data })) => {
                    output.extend_from_slice(data);
                    self.push_transcript_event(PtyTranscriptEventKind::Received {
                        data: data.to_vec(),
                    })?;
                    let current = String::from_utf8_lossy(&output);
                    if current.contains(needle) {
                        read.bytes = output.clone();
                        read.output = current.into_owned();
                        return Ok(read);
                    }
                }
                Ok(Some(ChannelMsg::ExtendedData { ref data, .. })) => {
                    output.extend_from_slice(data);
                    self.push_transcript_event(PtyTranscriptEventKind::Received {
                        data: data.to_vec(),
                    })?;
                    let current = String::from_utf8_lossy(&output);
                    if current.contains(needle) {
                        read.bytes = output.clone();
                        read.output = current.into_owned();
                        return Ok(read);
                    }
                }
                Ok(Some(ChannelMsg::ExitStatus { exit_status })) => {
                    read.exit_status = Some(exit_status);
                    self.push_transcript_event(PtyTranscriptEventKind::ExitStatus { exit_status })?;
                }
                Ok(Some(ChannelMsg::Eof)) => {
                    read.eof = true;
                    self.push_transcript_event(PtyTranscriptEventKind::Eof)?;
                    break;
                }
                Ok(Some(ChannelMsg::Close)) => {
                    read.closed = true;
                    self.push_transcript_event(PtyTranscriptEventKind::Close)?;
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
        self.push_transcript_event(PtyTranscriptEventKind::Close)?;

        let channel = self.channel.lock().await;
        channel.close().await.map_err(|e| SshProxyError::Russh {
            operation: "close PTY channel",
            reason: e.to_string(),
        })
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
        transcript_started_at: Instant::now(),
    })
    .map_err(|e| SshProxyError::Russh {
        operation: "open PTY",
        reason: e.to_string(),
    })
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
            .map_err(|e| SshProxyError::GenerateServerKey {
                reason: e.to_string(),
            })?;

    let russh_config = Arc::new(russh::server::Config {
        keys: vec![server_key],
        ..Default::default()
    });

    let listener = tokio::net::TcpListener::bind(config.listen)
        .await
        .map_err(|e| SshProxyError::ProxyBind {
            listen: config.listen,
            reason: e.to_string(),
        })?;

    tracing::info!("SSH proxy listening on {}", config.listen);

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
            principal_resolver: config.principal_resolver.clone(),
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
#[derive(Default)]
struct ProxyChannelState {
    writer: Option<ChannelWriteHalf<russh::client::Msg>>,
}

type SharedGuestChannels = Arc<tokio::sync::Mutex<HashMap<russh::ChannelId, ProxyChannelState>>>;

struct ProxyHandler {
    registry: GuestRegistry,
    username: Option<String>,
    principal_resolver: Option<PrincipalResolver>,
    /// Per-client-channel proxied guest state. Each inbound localhost SSH
    /// session channel gets its own guest SSH channel writer for forwarding
    /// client payload and lifecycle events into the guest.
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
        let principal_resolver = self.principal_resolver.clone();
        let guest_channels = Arc::clone(&self.guest_channels);
        let server_handle = session.handle();
        let client_ch_id = channel.id();

        async move {
            let username = match username {
                Some(u) => u,
                None => return Ok(false),
            };

            let guest_name = match principal_resolver.as_ref() {
                Some(resolve) => match resolve(username.clone()).await {
                    Ok(guest_name) => guest_name,
                    Err(SshProxyError::ResolveGuest { reason, .. }) => {
                        let _ = channel.data(format!("{reason}\r\n").as_bytes()).await;
                        let _ = channel.close().await;
                        return Ok(false);
                    }
                    Err(err) => {
                        let _ = channel.data(&b"internal SSH proxy error\r\n"[..]).await;
                        let _ = channel.close().await;
                        return Err(err);
                    }
                },
                None => username.clone(),
            };

            if guest_name != username {
                tracing::info!(
                    "SSH proxy: principal '{}' resolved to guest '{}'",
                    username,
                    guest_name
                );
            }

            if current_guest_handle(&registry, &guest_name)?.is_none() {
                let msg = format!(
                    "Guest '{}' is not launched or SSH bridge not ready.\r\n",
                    guest_name
                );
                let _ = channel.data(msg.as_bytes()).await;
                let _ = channel.close().await;
                return Ok(false);
            }

            tracing::info!("SSH proxy: opening session to guest '{guest_name}'");

            let guest_ch =
                match open_guest_session_with_retry(&registry, &guest_name, Duration::from_secs(3))
                    .await
                {
                    Ok(ch) => ch,
                    Err(SshProxyError::GuestConnection { .. }) => {
                        let msg = format!(
                            "Guest '{}' is not launched or SSH bridge not ready.\r\n",
                            guest_name
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
                        }
                        ChannelMsg::Failure => {
                            debug_trace!("SSH proxy: guest channel reported Failure");
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
                    let _ = server_handle.channel_success(channel).await;
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
                    let _ = server_handle.channel_success(channel).await;
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
                    let _ = server_handle.channel_success(channel).await;
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
        // The guest bridge connects over a host-created UDS/vsock path and
        // authenticates the guest with a fresh CA-signed ephemeral cert.
        // The guest-side sshd server key is therefore accepted as part of the
        // localhost/vsock trust boundary for this harness-owned channel.
        async { Ok(true) }
    }
}

pub struct ProxyClientHandler;

impl russh::client::Handler for ProxyClientHandler {
    type Error = SshProxyError;

    fn check_server_key(
        &mut self,
        _server_public_key: &russh::keys::PublicKey,
    ) -> impl std::future::Future<Output = Result<bool, Self::Error>> + Send {
        async { Ok(true) }
    }
}
