//! SSH proxy: host-side ingress and programmatic guest control plane.
//!
//! ## Design (FR-6, FR-7, DESIGN.md)
//!
//! The SSH proxy serves two roles:
//!
//! 1. **User-facing ingress** — russh server on localhost, bridges SSH
//!    channels to guest sshd via CA-signed ephemeral certs.
//! 2. **Programmatic exec** — opens non-PTY SSH channels to execute
//!    commands inside guests and capture output, enabling automated
//!    validation without human intervention.
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
use tracing;

use crate::ca::{CaError, SshCa};

#[derive(Debug, Error)]
pub enum SshProxyError {
    #[error("failed to connect to guest sshd at {addr}: {reason}")]
    GuestConnection { addr: SocketAddr, reason: String },
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
    /// Port on guest VMs where sshd listens (default: 22).
    pub guest_ssh_port: u16,
}

impl Default for SshProxyConfig {
    fn default() -> Self {
        Self {
            listen: SocketAddr::new(std::net::IpAddr::V4(Ipv4Addr::LOCALHOST), 2222),
            guest_ssh_port: 22,
        }
    }
}

// ---------------------------------------------------------------------------
// Shared guest registry — the REPL updates this, the SSH proxy reads it
// ---------------------------------------------------------------------------

/// Shared registry of launched guests and their admin IPs.
/// Updated by the REPL when guests are launched/shut down.
/// Read by the SSH proxy to resolve guest names to IPs.
pub type GuestRegistry = Arc<Mutex<HashMap<String, Ipv4Addr>>>;

/// Create a new empty guest registry.
pub fn new_guest_registry() -> GuestRegistry {
    Arc::new(Mutex::new(HashMap::new()))
}

// ---------------------------------------------------------------------------
// FR-7: Programmatic exec (client-side only)
// ---------------------------------------------------------------------------

/// Execute a command inside a guest VM via SSH exec channel.
///
/// Opens a connection to the guest's sshd using an ephemeral CA-signed
/// cert, runs the command on a non-PTY channel, and captures output.
pub async fn exec(
    ca: &SshCa,
    guest_name: &str,
    guest_ip: Ipv4Addr,
    guest_ssh_port: u16,
    command: &str,
) -> Result<ExecOutput, SshProxyError> {
    let eph = ca.sign_ephemeral(guest_name)?;

    let config = russh::client::Config::default();
    let addr = SocketAddr::new(std::net::IpAddr::V4(guest_ip), guest_ssh_port);

    let sh = GuestClientHandler;
    let mut handle = russh::client::connect(Arc::new(config), addr, sh)
        .await
        .map_err(|e| SshProxyError::GuestConnection {
            addr,
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
            addr,
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

// ---------------------------------------------------------------------------
// FR-6: SSH proxy server (listen + bridge)
// ---------------------------------------------------------------------------

/// Start the SSH proxy server.
///
/// Listens on `config.listen` and for each incoming connection:
/// 1. Extracts the username as guest identity
/// 2. Looks up the guest IP from the registry
/// 3. Signs an ephemeral cert and connects to guest sshd
/// 4. Bridges all channels bidirectionally
///
/// This function runs until the listener is dropped or an error occurs.
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
            guest_ssh_port: config.guest_ssh_port,
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
///
/// Implements russh::server::Handler. On channel open, connects to the
/// guest sshd and bridges all traffic bidirectionally.
struct ProxyHandler {
    ca: Arc<SshCa>,
    registry: GuestRegistry,
    guest_ssh_port: u16,
    username: Option<String>,
    /// The client-side channel to the guest sshd, set after channel_open_session.
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

        // Look up guest IP from the shared registry.
        let guest_ip = {
            let reg = self.registry.lock().unwrap();
            reg.get(&username).copied()
        };
        let guest_ip = match guest_ip {
            Some(ip) => ip,
            None => {
                tracing::warn!("SSH proxy: unknown or unlaunched guest '{username}'");
                // Send an error message to the client before rejecting.
                let msg = format!("Guest '{username}' is not launched. Use the REPL to `launch {username}` first.\r\n");
                let _ = channel.data(msg.as_bytes()).await;
                let _ = channel.close().await;
                return Ok(false);
            }
        };

        tracing::info!("SSH proxy: connecting to guest '{username}' at {guest_ip}:{}", self.guest_ssh_port);

        // Sign ephemeral cert and connect to guest sshd.
        let eph = self.ca.sign_ephemeral(&username)?;
        let client_config = Arc::new(russh::client::Config::default());
        let addr = SocketAddr::new(std::net::IpAddr::V4(guest_ip), self.guest_ssh_port);

        let mut client_handle =
            russh::client::connect(client_config, addr, GuestClientHandler).await?;

        let key_with_alg = PrivateKeyWithHashAlg::new(Arc::new(eph.key), None)
            .map_err(|e| anyhow::anyhow!("key error: {e}"))?;

        let authed = client_handle
            .authenticate_publickey(&username, key_with_alg)
            .await?;

        if !authed {
            tracing::warn!("SSH proxy: guest sshd rejected auth for '{username}'");
            let msg = format!("Guest sshd rejected CA-based auth for '{username}'.\r\n");
            let _ = channel.data(msg.as_bytes()).await;
            let _ = channel.close().await;
            return Ok(false);
        }

        let guest_ch = client_handle.channel_open_session().await?;
        tracing::info!("SSH proxy: session bridged for '{username}'");

        self.guest_channel = Some(guest_ch);

        // Spawn a task to forward guest→client data.
        // We clone the server-side channel to send data back.
        let server_channel = channel;
        if let Some(mut guest_ch_reader) = self.guest_channel.take() {
            // We need the guest channel for both reading (background task) and
            // writing (handler methods). Clone the writable half for the handler.
            // Unfortunately russh Channel doesn't impl Clone, so we split the
            // read loop into the background task and keep a reference for writes
            // via the client_handle.
            //
            // Approach: the background task owns the guest channel for reading.
            // Handler methods (data, pty_request, etc.) use client_handle to
            // open new operations. But that doesn't work for forwarding data.
            //
            // The practical approach for v1.3: spawn a background task that
            // reads from guest and writes to server_channel. The handler
            // forwards client→guest via storing the client_handle.
            let guest_channel_for_handler = client_handle.channel_open_session().await?;
            self.guest_channel = Some(guest_channel_for_handler);

            tokio::spawn(async move {
                while let Some(msg) = guest_ch_reader.wait().await {
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
                            // Can't easily forward exit status through the server channel
                            // in the same way. The session handler would need to call
                            // session.exit_status_request(). We'll handle this via EOF.
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
        }

        Ok(true)
    }

    async fn data(
        &mut self,
        _channel: ChannelId,
        data: &[u8],
        _session: &mut Session,
    ) -> Result<(), Self::Error> {
        // Forward client→guest data.
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

/// Minimal russh client handler for guest connections.
/// Accepts the guest's host key unconditionally.
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
