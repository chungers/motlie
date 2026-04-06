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
//! The guest connects OUT to the host via vsock (guest→host direction,
//! which CH's vhost-vsock supports). A socat bridge in the guest connects
//! `VSOCK-CONNECT:2:<port>` to `TCP:localhost:22` (sshd). The host
//! accepts on the vsock UDS (`$VSOCK_SOCKET_<port>`) and feeds the stream
//! into russh::client::connect_stream(). Multiple SSH sessions multiplex
//! over this single connection via SSH channels.
//!
//! This eliminates TAP and CAP_NET_ADMIN for ingress entirely.
//!
//! ## Auth model
//!
//! - Inbound: localhost trust (username = guest identity)
//! - Outbound: ephemeral Ed25519 cert signed by in-memory CA (60s TTL)

use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddr};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use russh::server::{Auth, Msg, Session};
use russh::{Channel, ChannelId, ChannelMsg, Pty};
use thiserror::Error;
use tokio::net::UnixListener;
use tracing;

use crate::ca::{CaError, SshCa};

/// vsock port used for the guest→host SSH bridge.
/// Guest socat connects to host CID 2 on this port.
/// Host listens on `$VSOCK_SOCKET_<PORT>`.
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
    /// Address to listen on for incoming SSH clients (default: 127.0.0.1:2222).
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
// Shared guest registry — the REPL updates this, the SSH proxy reads it
// ---------------------------------------------------------------------------

/// Per-guest connection info needed by the SSH proxy.
#[derive(Clone)]
pub struct GuestEndpoint {
    /// Authenticated russh client handle, established when the guest
    /// connects its vsock→sshd bridge after boot.
    pub ssh_handle: Option<Arc<tokio::sync::Mutex<russh::client::Handle<GuestClientHandler>>>>,
}

impl std::fmt::Debug for GuestEndpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuestEndpoint")
            .field("ssh_handle", &self.ssh_handle.as_ref().map(|_| "..."))
            .finish()
    }
}

/// Shared registry of launched guests.
pub type GuestRegistry = Arc<Mutex<HashMap<String, GuestEndpoint>>>;

/// Create a new empty guest registry.
pub fn new_guest_registry() -> GuestRegistry {
    Arc::new(Mutex::new(HashMap::new()))
}

// ---------------------------------------------------------------------------
// Accept guest's vsock SSH bridge connection
// ---------------------------------------------------------------------------

/// Derive the host-side UDS path for the vsock SSH bridge.
/// CH routes guest vsock connections to `$VSOCK_SOCKET_$PORT`.
pub fn vsock_ssh_uds_path(vsock_socket: &str) -> PathBuf {
    PathBuf::from(format!("{}_{}", vsock_socket, VSOCK_SSH_PORT))
}

/// Bind the vsock SSH bridge UDS listener BEFORE the guest boots.
///
/// Must be called before `execute_launch_script` so the UDS file
/// exists when the guest's socat tries to connect.
/// Pass the returned listener to `accept_guest_ssh`.
pub fn bind_vsock_ssh_listener(
    vsock_socket: &str,
) -> Result<(UnixListener, PathBuf), SshProxyError> {
    let uds_path = vsock_ssh_uds_path(vsock_socket);
    let _ = std::fs::remove_file(&uds_path);

    let listener = UnixListener::bind(&uds_path).map_err(|e| SshProxyError::Ssh(
        format!("failed to bind vsock SSH UDS {}: {e}", uds_path.display()),
    ))?;

    tracing::info!("Bound vsock SSH bridge UDS at {}", uds_path.display());
    Ok((listener, uds_path))
}

/// Accept the guest's vsock SSH bridge connection and authenticate.
///
/// The listener must be bound BEFORE the guest boots (via
/// `bind_vsock_ssh_listener`). After guest boot, the guest's socat
/// service connects to host CID 2 on VSOCK_SSH_PORT. CH routes this
/// to the UDS. We accept and authenticate to guest sshd over it.
///
/// Returns an authenticated russh client handle.
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

    // Authenticate to guest sshd over the accepted stream using the
    // CA-signed ephemeral certificate (not bare publickey).
    let eph = ca.sign_ephemeral(guest_name)?;

    // Debug: dump cert and key to /tmp for manual testing with ssh(1).
    let debug_dir = std::path::PathBuf::from(format!("/tmp/motlie-vmm-ssh-debug-{}", guest_name));
    let _ = std::fs::create_dir_all(&debug_dir);
    if let Ok(cert_str) = eph.cert.to_openssh() {
        let _ = std::fs::write(debug_dir.join("cert.pub"), &cert_str);
        eprintln!("SSH debug: wrote cert to {}/cert.pub", debug_dir.display());
    }
    if let Ok(key_str) = eph.key.to_openssh(ssh_key::LineEnding::LF) {
        let _ = std::fs::write(debug_dir.join("key"), AsRef::<[u8]>::as_ref(&key_str));
        let _ = std::fs::set_permissions(debug_dir.join("key"), std::fs::Permissions::from_mode(0o600));
        eprintln!("SSH debug: wrote key to {}/key", debug_dir.display());
    }
    if let Ok(ca_pub) = ca.public_key_openssh() {
        let _ = std::fs::write(debug_dir.join("ca.pub"), &ca_pub);
    }
    eprintln!("SSH debug: cert principals={:?} type={:?}", eph.cert.valid_principals(), eph.cert.cert_type());

    let config = Arc::new(russh::client::Config::default());

    let mut handle = russh::client::connect_stream(config, stream, GuestClientHandler)
        .await
        .map_err(|e| SshProxyError::GuestConnection {
            guest: guest_name.into(),
            reason: format!("SSH handshake failed: {e}"),
        })?;

    let authed = handle
        .authenticate_openssh_cert(guest_name, Arc::new(eph.key), eph.cert)
        .await
        .map_err(|e| SshProxyError::Ssh(format!("SSH cert auth failed: {e}")))?;

    if !authed {
        return Err(SshProxyError::GuestConnection {
            guest: guest_name.into(),
            reason: "guest sshd rejected CA-based auth".into(),
        });
    }

    tracing::info!("Guest '{}' SSH authenticated via CA cert", guest_name);
    Ok(handle)
}

// ---------------------------------------------------------------------------
// FR-7: Programmatic exec (over pre-established vsock SSH connection)
// ---------------------------------------------------------------------------

/// Execute a command inside a guest via its pre-established SSH connection.
///
/// Opens a new channel on the existing authenticated handle, runs the
/// command, and captures output. Multiple execs can run concurrently
/// via SSH channel multiplexing.
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

    drop(guard); // Release lock while command executes.

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
// FR-6: SSH proxy server (listen + bridge over pre-established vsock)
// ---------------------------------------------------------------------------

/// Start the SSH proxy server.
///
/// Listens on `config.listen` (TCP, localhost). For each incoming SSH
/// client, extracts the username as guest identity, looks up the
/// pre-established vsock SSH handle from the registry, and bridges
/// channels bidirectionally.
pub async fn run_proxy(
    config: SshProxyConfig,
    registry: GuestRegistry,
) -> Result<(), SshProxyError> {
    let server_key =
        russh_keys::PrivateKey::random(&mut rand::thread_rng(), ssh_key::Algorithm::Ed25519)
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
            guest_channel: None,
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
    guest_channel: Option<Channel<russh::client::Msg>>,
}

#[async_trait::async_trait]
impl russh::server::Handler for ProxyHandler {
    type Error = anyhow::Error;

    async fn auth_none(&mut self, user: &str) -> Result<Auth, Self::Error> {
        self.username = Some(user.to_string());
        tracing::info!("SSH proxy: auth_none user={user}");
        Ok(Auth::Accept)
    }

    async fn auth_publickey(
        &mut self,
        user: &str,
        _public_key: &ssh_key::PublicKey,
    ) -> Result<Auth, Self::Error> {
        self.username = Some(user.to_string());
        tracing::info!("SSH proxy: auth_publickey user={user}");
        Ok(Auth::Accept)
    }

    async fn channel_open_session(
        &mut self,
        channel: Channel<Msg>,
        _session: &mut Session,
    ) -> Result<bool, Self::Error> {
        let username = match &self.username {
            Some(u) => u.clone(),
            None => return Ok(false),
        };

        // Look up the pre-established SSH handle for this guest.
        let endpoint = {
            let reg = self.registry.lock().unwrap();
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

        // Open a new session channel on the guest's multiplexed SSH connection.
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

        // For client→guest, open another channel that handler methods write to.
        let guard2 = ssh_handle.lock().await;
        let write_ch = guard2.channel_open_session().await?;
        drop(guard2);
        self.guest_channel = Some(write_ch);

        Ok(true)
    }

    async fn data(
        &mut self,
        _channel: ChannelId,
        data: &[u8],
        _session: &mut Session,
    ) -> Result<(), Self::Error> {
        if let Some(ref guest_ch) = self.guest_channel {
            guest_ch.data(data).await?;
        }
        Ok(())
    }

    async fn pty_request(
        &mut self,
        _channel: ChannelId,
        term: &str,
        col_width: u32,
        row_height: u32,
        pix_width: u32,
        pix_height: u32,
        modes: &[(Pty, u32)],
        session: &mut Session,
    ) -> Result<(), Self::Error> {
        if let Some(ref guest_ch) = self.guest_channel {
            guest_ch
                .request_pty(true, term, col_width, row_height, pix_width, pix_height, modes)
                .await?;
        }
        session.channel_success(_channel)?;
        Ok(())
    }

    async fn shell_request(
        &mut self,
        _channel: ChannelId,
        session: &mut Session,
    ) -> Result<(), Self::Error> {
        if let Some(ref guest_ch) = self.guest_channel {
            guest_ch.request_shell(true).await?;
        }
        session.channel_success(_channel)?;
        Ok(())
    }

    async fn exec_request(
        &mut self,
        _channel: ChannelId,
        data: &[u8],
        session: &mut Session,
    ) -> Result<(), Self::Error> {
        if let Some(ref guest_ch) = self.guest_channel {
            guest_ch.exec(true, data).await?;
        }
        session.channel_success(_channel)?;
        Ok(())
    }

    async fn window_change_request(
        &mut self,
        _channel: ChannelId,
        col_width: u32,
        row_height: u32,
        pix_width: u32,
        pix_height: u32,
        _session: &mut Session,
    ) -> Result<(), Self::Error> {
        if let Some(ref guest_ch) = self.guest_channel {
            guest_ch
                .window_change(col_width, row_height, pix_width, pix_height)
                .await?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Client-side handler for guest connections
// ---------------------------------------------------------------------------

pub struct GuestClientHandler;

#[async_trait::async_trait]
impl russh::client::Handler for GuestClientHandler {
    type Error = anyhow::Error;

    async fn check_server_key(
        &mut self,
        _server_public_key: &ssh_key::PublicKey,
    ) -> Result<bool, Self::Error> {
        Ok(true)
    }
}
