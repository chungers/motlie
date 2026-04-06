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
//! The proxy reaches guest sshd over vsock (AF_VSOCK), not TCP/TAP.
//! The guest runs a socat bridge: vsock port 2222 → TCP localhost:22.
//! This eliminates the TAP NIC and CAP_NET_ADMIN requirement for ingress.
//!
//! ## Auth model
//!
//! - Inbound: localhost trust (username = guest identity)
//! - Outbound: ephemeral Ed25519 cert signed by in-memory CA (60s TTL)

use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::{Arc, Mutex};

use russh::server::{Auth, Msg, Session};
use russh::{Channel, ChannelId, ChannelMsg, Pty};
use russh_keys::key::PrivateKeyWithHashAlg;
use thiserror::Error;
use tokio_vsock::VsockStream;
use tracing;

use crate::ca::{CaError, SshCa};

/// vsock port where the guest's socat bridge listens (vsock:2222 → TCP:localhost:22).
pub const GUEST_VSOCK_SSH_PORT: u32 = 2222;

#[derive(Debug, Error)]
pub enum SshProxyError {
    #[error("failed to connect to guest sshd via vsock cid={cid} port={port}: {reason}")]
    GuestConnection { cid: u32, port: u32, reason: String },
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
    /// Address to listen on (default: 127.0.0.1:2222).
    pub listen: SocketAddr,
    /// vsock port on guest where the socat bridge listens (default: 2222).
    pub guest_vsock_ssh_port: u32,
}

impl Default for SshProxyConfig {
    fn default() -> Self {
        Self {
            listen: SocketAddr::new(std::net::IpAddr::V4(Ipv4Addr::LOCALHOST), 2222),
            guest_vsock_ssh_port: GUEST_VSOCK_SSH_PORT,
        }
    }
}

// ---------------------------------------------------------------------------
// Shared guest registry — the REPL updates this, the SSH proxy reads it
// ---------------------------------------------------------------------------

/// Per-guest connection info needed by the SSH proxy.
#[derive(Clone, Debug)]
pub struct GuestEndpoint {
    pub cid: u32,
    /// Admin TAP IP (kept for backward compat with `exec` REPL command).
    pub admin_ip: Option<Ipv4Addr>,
}

/// Shared registry of launched guests.
/// Updated by the REPL when guests are launched/shut down.
/// Read by the SSH proxy to resolve guest names.
pub type GuestRegistry = Arc<Mutex<HashMap<String, GuestEndpoint>>>;

/// Create a new empty guest registry.
pub fn new_guest_registry() -> GuestRegistry {
    Arc::new(Mutex::new(HashMap::new()))
}

// ---------------------------------------------------------------------------
// Connect to guest sshd via vsock
// ---------------------------------------------------------------------------

/// Connect to a guest's sshd over vsock and authenticate with an ephemeral cert.
/// Returns a russh client handle ready for channel operations.
async fn connect_guest_vsock(
    ca: &SshCa,
    guest_name: &str,
    cid: u32,
    vsock_ssh_port: u32,
) -> Result<russh::client::Handle<GuestClientHandler>, SshProxyError> {
    let eph = ca.sign_ephemeral(guest_name)?;

    let vsock_addr = tokio_vsock::VsockAddr::new(cid, vsock_ssh_port);
    let stream = VsockStream::connect(vsock_addr)
        .await
        .map_err(|e| SshProxyError::GuestConnection {
            cid,
            port: vsock_ssh_port,
            reason: e.to_string(),
        })?;

    let config = Arc::new(russh::client::Config::default());
    let mut handle = russh::client::connect_stream(config, stream, GuestClientHandler)
        .await
        .map_err(|e| SshProxyError::GuestConnection {
            cid,
            port: vsock_ssh_port,
            reason: e.to_string(),
        })?;

    let key_with_alg = PrivateKeyWithHashAlg::new(Arc::new(eph.key), None)
        .map_err(|e| SshProxyError::Ssh(e.to_string()))?;

    let authed = handle
        .authenticate_publickey(guest_name, key_with_alg)
        .await
        .map_err(|e| SshProxyError::Ssh(e.to_string()))?;

    if !authed {
        return Err(SshProxyError::GuestConnection {
            cid,
            port: vsock_ssh_port,
            reason: "authentication rejected by guest sshd".into(),
        });
    }

    Ok(handle)
}

// ---------------------------------------------------------------------------
// FR-7: Programmatic exec
// ---------------------------------------------------------------------------

/// Execute a command inside a guest VM via SSH over vsock.
///
/// Connects to the guest's socat vsock bridge → sshd, runs the command
/// on a non-PTY channel, and captures output.
pub async fn exec_vsock(
    ca: &SshCa,
    guest_name: &str,
    cid: u32,
    command: &str,
) -> Result<ExecOutput, SshProxyError> {
    let handle = connect_guest_vsock(ca, guest_name, cid, GUEST_VSOCK_SSH_PORT).await?;

    let mut channel = handle
        .channel_open_session()
        .await
        .map_err(|e| SshProxyError::Ssh(e.to_string()))?;

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
            ChannelMsg::Data { ref data } => {
                stdout.extend_from_slice(data);
            }
            ChannelMsg::ExtendedData { ref data, ext } => {
                if ext == 1 {
                    stderr.extend_from_slice(data);
                }
            }
            ChannelMsg::ExitStatus { exit_status } => {
                exit_code = Some(exit_status);
            }
            _ => {}
        }
    }

    Ok(ExecOutput {
        stdout: String::from_utf8_lossy(&stdout).into_owned(),
        stderr: String::from_utf8_lossy(&stderr).into_owned(),
        exit_code: exit_code.unwrap_or(255),
    })
}

/// Execute a command inside a guest VM via SSH over TCP (legacy TAP path).
/// Kept for backward compat during the transition; prefer exec_vsock.
pub async fn exec(
    ca: &SshCa,
    guest_name: &str,
    guest_ip: Ipv4Addr,
    guest_ssh_port: u16,
    command: &str,
) -> Result<ExecOutput, SshProxyError> {
    let eph = ca.sign_ephemeral(guest_name)?;
    let config = Arc::new(russh::client::Config::default());
    let addr = SocketAddr::new(std::net::IpAddr::V4(guest_ip), guest_ssh_port);

    let mut handle = russh::client::connect(config, addr, GuestClientHandler)
        .await
        .map_err(|e| SshProxyError::GuestConnection {
            cid: 0,
            port: guest_ssh_port as u32,
            reason: e.to_string(),
        })?;

    let key_with_alg = PrivateKeyWithHashAlg::new(Arc::new(eph.key), None)
        .map_err(|e| SshProxyError::Ssh(e.to_string()))?;

    let authed = handle
        .authenticate_publickey(guest_name, key_with_alg)
        .await
        .map_err(|e| SshProxyError::Ssh(e.to_string()))?;

    if !authed {
        return Err(SshProxyError::GuestConnection {
            cid: 0,
            port: guest_ssh_port as u32,
            reason: "authentication rejected by guest sshd".into(),
        });
    }

    let mut channel = handle
        .channel_open_session()
        .await
        .map_err(|e| SshProxyError::Ssh(e.to_string()))?;

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
            ChannelMsg::ExtendedData { ref data, ext } if ext == 1 => stderr.extend_from_slice(data),
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
// FR-6: SSH proxy server (listen + bridge over vsock)
// ---------------------------------------------------------------------------

/// Start the SSH proxy server.
///
/// Listens on `config.listen` (TCP, localhost) and for each incoming connection:
/// 1. Extracts the username as guest identity
/// 2. Looks up the guest CID from the registry
/// 3. Connects to guest sshd over vsock (AF_VSOCK)
/// 4. Signs an ephemeral cert and authenticates
/// 5. Bridges all channels bidirectionally
pub async fn run_proxy(
    config: SshProxyConfig,
    ca: Arc<SshCa>,
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
    eprintln!(
        "  ssh -p {} <guest>@localhost",
        config.listen.port()
    );

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
            ca: Arc::clone(&ca),
            registry: Arc::clone(&registry),
            guest_vsock_ssh_port: config.guest_vsock_ssh_port,
            username: None,
            guest_channel: None,
        };

        let cfg = Arc::clone(&russh_config);
        tokio::spawn(async move {
            match russh::server::run_stream(cfg, stream, handler).await {
                Ok(_session) => {
                    tracing::info!("SSH proxy: session from {peer} ended");
                }
                Err(e) => {
                    tracing::warn!("SSH proxy: session from {peer} failed: {e:?}");
                }
            }
        });
    }
}

/// Server-side handler for one SSH proxy connection.
struct ProxyHandler {
    ca: Arc<SshCa>,
    registry: GuestRegistry,
    guest_vsock_ssh_port: u32,
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
            None => {
                tracing::warn!("SSH proxy: channel_open_session with no username");
                return Ok(false);
            }
        };

        // Look up guest CID from the shared registry.
        let endpoint = {
            let reg = self.registry.lock().unwrap();
            reg.get(&username).cloned()
        };
        let endpoint = match endpoint {
            Some(ep) => ep,
            None => {
                tracing::warn!("SSH proxy: unknown or unlaunched guest '{username}'");
                let msg = format!(
                    "Guest '{username}' is not launched. Use the REPL to `launch {username}` first.\r\n"
                );
                let _ = channel.data(msg.as_bytes()).await;
                let _ = channel.close().await;
                return Ok(false);
            }
        };

        tracing::info!(
            "SSH proxy: connecting to guest '{username}' via vsock cid={} port={}",
            endpoint.cid,
            self.guest_vsock_ssh_port
        );

        // Connect to guest sshd over vsock.
        let client_handle = match connect_guest_vsock(
            &self.ca,
            &username,
            endpoint.cid,
            self.guest_vsock_ssh_port,
        )
        .await
        {
            Ok(h) => h,
            Err(e) => {
                tracing::warn!("SSH proxy: failed to connect to guest '{username}': {e}");
                let msg = format!("Failed to connect to guest '{username}': {e}\r\n");
                let _ = channel.data(msg.as_bytes()).await;
                let _ = channel.close().await;
                return Ok(false);
            }
        };

        let mut guest_ch = client_handle.channel_open_session().await?;
        tracing::info!("SSH proxy: session bridged for '{username}'");

        // Spawn guest→client forwarding task (reads from guest, writes to client).
        let server_channel = channel;
        tokio::spawn(async move {
            while let Some(msg) = guest_ch.wait().await {
                match msg {
                    ChannelMsg::Data { ref data } => {
                        if server_channel.data(data.as_ref()).await.is_err() {
                            break;
                        }
                    }
                    ChannelMsg::ExtendedData { ref data, ext } => {
                        if ext == 1 {
                            if server_channel.extended_data(1, data.as_ref()).await.is_err() {
                                break;
                            }
                        }
                    }
                    ChannelMsg::ExitStatus { exit_status } => {
                        tracing::debug!("SSH proxy: guest exit_status={exit_status}");
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
            tracing::debug!("SSH proxy: guest→client forward loop ended");
        });

        // For client→guest forwarding, we need a channel to write to.
        // Open a second session channel on the same client connection for
        // handler methods to use.
        let write_ch = client_handle.channel_open_session().await?;
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
// Client-side handler for guest connections (shared by exec + proxy)
// ---------------------------------------------------------------------------

struct GuestClientHandler;

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
