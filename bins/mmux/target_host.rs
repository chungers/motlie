use std::net::{IpAddr, ToSocketAddrs};
use std::process::Command;

use anyhow::{Context, Result};
use motlie_tmux::{HostHandle, SshConfig, SSH_DEFAULT_PORT};

use crate::cli::Cli;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct HostIdentity {
    pub(crate) hostname: String,
    pub(crate) ip_address: String,
}

impl HostIdentity {
    fn new(hostname: String, port: u16) -> Self {
        let ip_address = resolve_ip_address(&hostname, port);
        Self {
            hostname,
            ip_address,
        }
    }

    fn local() -> Self {
        let hostname = local_hostname();
        let ip_address = local_ip_address(&hostname)
            .unwrap_or_else(|| resolve_ip_address(&hostname, SSH_DEFAULT_PORT));
        Self {
            hostname,
            ip_address,
        }
    }
}

pub(crate) async fn connect_host(cli: &Cli) -> Result<(HostHandle, HostIdentity)> {
    match &cli.ssh_uri {
        Some(uri) => {
            let config = SshConfig::parse(uri).context("parse ssh target")?;
            let identity = HostIdentity::new(config.host().to_string(), config.port());
            let host = config.connect().await.context("connect ssh target")?;
            Ok((host, identity))
        }
        None => Ok((HostHandle::local(), HostIdentity::local())),
    }
}

fn local_hostname() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .or_else(|| command_first_token("hostname", &[]))
        .unwrap_or_else(|| "localhost".to_string())
}

fn local_ip_address(hostname: &str) -> Option<String> {
    command_first_token("hostname", &["-I"]).or_else(|| {
        let resolved = resolve_ip_address(hostname, SSH_DEFAULT_PORT);
        if resolved == "unknown" {
            None
        } else {
            Some(resolved)
        }
    })
}

fn command_first_token(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .next()
        .map(str::to_string)
        .filter(|value| !value.is_empty())
}

pub(crate) fn resolve_ip_address(hostname: &str, port: u16) -> String {
    if let Ok(ip) = hostname.parse::<IpAddr>() {
        return ip.to_string();
    }

    let Ok(addrs) = (hostname, port).to_socket_addrs() else {
        return "unknown".to_string();
    };
    let ips = addrs.map(|addr| addr.ip()).collect::<Vec<_>>();
    ips.iter()
        .find(|ip| matches!(ip, IpAddr::V4(_)))
        .or_else(|| ips.first())
        .map(ToString::to_string)
        .unwrap_or_else(|| "unknown".to_string())
}
