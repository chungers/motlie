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
use std::net::{Ipv4Addr, SocketAddr};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use russh::server::{Auth, Msg, Session};
use russh::{Channel, ChannelMsg};
use thiserror::Error;
use tokio::net::UnixListener;

use crate::ca::{CaError, SshCa};

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
    #[error("unknown guest: {0}")]
    UnknownGuest(String),
}

/// Output from a programmatic command execution inside a guest VM.
#[derive(Debug, Clone)]
pub struct ExecOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: u32,
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
    let _ = std::fs::remove_file(&uds_path);
    let listener = UnixListener::bind(&uds_path).map_err(|e| {
        SshProxyError::Ssh(format!(
            "failed to bind vsock SSH UDS {}: {e}",
            uds_path.display()
        ))
    })?;
    tracing::info!("Bound vsock SSH bridge UDS at {}", uds_path.display());
    Ok((listener, uds_path))
}

/// Accept the guest's vsock SSH bridge and authenticate with CA cert.
pub async fn accept_guest_ssh(
    listener: UnixListener,
    uds_path: &std::path::Path,
    ca: &SshCa,
    guest_name: &str,
) -> Result<russh::client::Handle<GuestClientHandler>, SshProxyError> {
    tracing::info!(
        "Waiting for guest '{}' vsock SSH bridge on {}",
        guest_name,
        uds_path.display()
    );

    let (stream, _) = listener.accept().await.map_err(|e| SshProxyError::GuestConnection {
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

// ---------------------------------------------------------------------------
// FR-7: Programmatic exec
// ---------------------------------------------------------------------------

pub async fn exec_on_handle(
    handle: &Arc<tokio::sync::Mutex<russh::client::Handle<GuestClientHandler>>>,
    guest_name: &str,
    command: &str,
) -> Result<ExecOutput, SshProxyError> {
    let guard = handle.lock().await;
    let mut channel = guard
        .channel_open_session()
        .await
        .map_err(|e| SshProxyError::Ssh(e.to_string()))?;
    drop(guard);

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
        exit_code: exit_code.unwrap_or(255),
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

struct ProxyHandler {
    registry: GuestRegistry,
    username: Option<String>,
}

impl russh::server::Handler for ProxyHandler {
    type Error = anyhow::Error;

    fn auth_none(&mut self, user: &str) -> impl std::future::Future<Output = Result<Auth, Self::Error>> + Send {
        self.username = Some(user.to_string());
        tracing::info!("SSH proxy: auth_none user={user}");
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
        _session: &mut Session,
    ) -> impl std::future::Future<Output = Result<bool, Self::Error>> + Send {
        let username = self.username.clone();
        let registry = Arc::clone(&self.registry);

        async move {
            let username = match username {
                Some(u) => u,
                None => return Ok(false),
            };

            let endpoint = {
                let reg = registry.lock().unwrap();
                reg.get(&username).cloned()
            };
            let ssh_handle = match endpoint.and_then(|ep| ep.ssh_handle) {
                Some(h) => h,
                None => {
                    let msg = format!(
                        "Guest '{}' is not launched or SSH bridge not ready.\r\n",
                        username
                    );
                    let _ = channel.data(msg.as_bytes()).await;
                    let _ = channel.close().await;
                    return Ok(false);
                }
            };

            tracing::info!("SSH proxy: opening channel to guest '{username}'");

            let guard = ssh_handle.lock().await;
            let guest_ch = guard.channel_open_session().await?;
            drop(guard);

            // Spawn guest→client forwarding task.
            let server_channel = channel;
            let mut guest_ch_read = guest_ch;
            tokio::spawn(async move {
                while let Some(msg) = guest_ch_read.wait().await {
                    match msg {
                        ChannelMsg::Data { ref data } => {
                            if server_channel.data(data.as_ref()).await.is_err() {
                                break;
                            }
                        }
                        ChannelMsg::ExtendedData { ref data, ext } if ext == 1 => {
                            if server_channel.extended_data(1, data.as_ref()).await.is_err() {
                                break;
                            }
                        }
                        ChannelMsg::Eof => {
                            let _ = server_channel.eof().await;
                            break;
                        }
                        ChannelMsg::Close => {
                            let _ = server_channel.close().await;
                            break;
                        }
                        _ => {}
                    }
                }
            });

            // Open a write channel for handler methods.
            let guard2 = ssh_handle.lock().await;
            let _write_ch = guard2.channel_open_session().await?;
            drop(guard2);
            // Can't set self.guest_channel here since we're in an async block
            // that doesn't have &mut self. Store it via a side channel.
            // For now, the write channel is unused — data forwarding happens
            // through the spawned task above. This is a known limitation;
            // full bidirectional bridging will be addressed in a follow-up.

            Ok(true)
        }
    }

    // Client→guest forwarding (data, pty, shell, exec, window_change) is
    // not yet wired because channel_open_session runs in an async block
    // that can't set self.guest_channel. The guest→client direction works
    // via the spawned forwarding task. Full bidirectional interactive SSH
    // proxy is a follow-up; programmatic exec via exec_on_handle works now.
}

// ---------------------------------------------------------------------------
// Client handler for guest connections
// ---------------------------------------------------------------------------

pub struct GuestClientHandler;

impl russh::client::Handler for GuestClientHandler {
    type Error = anyhow::Error;

    fn check_server_key(
        &mut self,
        _server_public_key: &russh::keys::PublicKey,
    ) -> impl std::future::Future<Output = Result<bool, Self::Error>> + Send {
        async { Ok(true) }
    }
}
