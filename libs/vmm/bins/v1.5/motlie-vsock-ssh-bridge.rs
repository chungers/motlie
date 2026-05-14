//! v1.5 guest-side vsock-to-SSH bridge.
//!
//! This replaces the shell-era dependency on distro `socat` VSOCK support,
//! which is not portable across the Ubuntu/Debian packages used by CH/VZ
//! image experiments.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::net::TcpStream;
use tokio::time::sleep;
use tokio_vsock::{VsockAddr, VsockStream};

const DEFAULT_BACKEND_ENV: &str = "/etc/motlie/v1.5/backend.env";
const DEFAULT_HOST_CID: u32 = 2;
const DEFAULT_SSH_VSOCK_PORT: u32 = 2222;
const DEFAULT_SSH_TCP_ADDR: &str = "127.0.0.1:22";
const RETRY_DELAY: Duration = Duration::from_secs(1);

#[derive(Debug, Clone)]
struct BridgeConfig {
    host_cid: u32,
    vsock_port: u32,
    tcp_addr: String,
}

impl BridgeConfig {
    fn load(path: &Path) -> Self {
        let env = read_env_file(path).unwrap_or_default();
        Self {
            host_cid: env
                .get("MOTLIE_SSH_HOST_CID")
                .or_else(|| env.get("MOTLIE_VFS_HOST_CID"))
                .and_then(|value| value.parse().ok())
                .unwrap_or(DEFAULT_HOST_CID),
            vsock_port: env
                .get("MOTLIE_SSH_VSOCK_PORT")
                .and_then(|value| value.parse().ok())
                .unwrap_or(DEFAULT_SSH_VSOCK_PORT),
            tcp_addr: env
                .get("MOTLIE_SSH_TCP_ADDR")
                .cloned()
                .unwrap_or_else(|| DEFAULT_SSH_TCP_ADDR.to_string()),
        }
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let env_path =
        std::env::var("MOTLIE_BACKEND_ENV").unwrap_or_else(|_| DEFAULT_BACKEND_ENV.to_string());
    let env_path = Path::new(&env_path);

    loop {
        let config = BridgeConfig::load(env_path);
        eprintln!(
            "motlie-vsock-ssh-bridge: connecting cid={} port={} tcp={}",
            config.host_cid, config.vsock_port, config.tcp_addr
        );
        if let Err(error) = bridge_once(&config).await {
            eprintln!(
                "motlie-vsock-ssh-bridge: cid={} port={} tcp={} error={error:#}",
                config.host_cid, config.vsock_port, config.tcp_addr
            );
        }
        sleep(RETRY_DELAY).await;
    }
}

async fn bridge_once(config: &BridgeConfig) -> Result<()> {
    let mut tcp = TcpStream::connect(&config.tcp_addr)
        .await
        .with_context(|| format!("connect guest SSH at {}", config.tcp_addr))?;
    eprintln!("motlie-vsock-ssh-bridge: connected guest SSH");
    let addr = VsockAddr::new(config.host_cid, config.vsock_port);
    let mut vsock = VsockStream::connect(addr).await.with_context(|| {
        format!(
            "connect host vsock cid={} port={}",
            config.host_cid, config.vsock_port
        )
    })?;
    eprintln!("motlie-vsock-ssh-bridge: connected host vsock");
    let _ = tokio::io::copy_bidirectional(&mut tcp, &mut vsock)
        .await
        .context("bridge SSH TCP stream to host vsock")?;
    eprintln!("motlie-vsock-ssh-bridge: bridge stream closed");
    Ok(())
}

fn read_env_file(path: &Path) -> Result<BTreeMap<String, String>> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("read backend env {}", path.display()))?;
    let mut env = BTreeMap::new();
    for line in contents.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        env.insert(
            key.trim().to_string(),
            value.trim().trim_matches('"').to_string(),
        );
    }
    Ok(env)
}
