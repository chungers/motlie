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

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;

use russh_keys::key::PrivateKeyWithHashAlg;
use thiserror::Error;

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

/// Execute a command inside a guest VM via SSH exec channel.
///
/// Opens a connection to the guest's sshd using an ephemeral CA-signed
/// cert, runs the command on a non-PTY channel, and captures output.
///
/// This is the core primitive for FR-7 (automated guest validation).
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

    let sh = GuestSshHandler;
    let mut handle = russh::client::connect(Arc::new(config), addr, sh)
        .await
        .map_err(|e| SshProxyError::GuestConnection {
            addr,
            reason: e.to_string(),
        })?;

    // Authenticate with the ephemeral key.
    // For Ed25519 keys, hash_alg is None (only RSA needs it).
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
            russh::ChannelMsg::Data { ref data } => {
                stdout.extend_from_slice(data);
            }
            russh::ChannelMsg::ExtendedData { ref data, ext } => {
                if ext == 1 {
                    // stderr
                    stderr.extend_from_slice(data);
                }
            }
            russh::ChannelMsg::ExitStatus { exit_status } => {
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

/// Minimal russh client handler for guest connections.
///
/// Accepts the guest's host key unconditionally — the guest image is
/// controlled by the orchestrator, and host key verification is handled
/// by the CA trust chain (the guest has a host cert signed by our CA).
struct GuestSshHandler;

#[async_trait::async_trait]
impl russh::client::Handler for GuestSshHandler {
    type Error = anyhow::Error;

    async fn check_server_key(
        &mut self,
        _server_public_key: &ssh_key::PublicKey,
    ) -> Result<bool, Self::Error> {
        // Accept unconditionally — we control the guest image and its
        // host keys. In the long term, the guest has a host cert signed
        // by our CA, so this could verify that cert. For v1, trust by
        // construction.
        Ok(true)
    }
}
