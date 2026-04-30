use std::net::{IpAddr, ToSocketAddrs};
use std::process::Command;

use anyhow::{anyhow, Context, Result};
use motlie_tmux::{HostHandle, SshConfig, SSH_DEFAULT_PORT};

use crate::cli::Cli;
use crate::model::{HostEntry, HostFleet, HostId};

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

/// Connect to all hosts named on the CLI. With zero URIs, returns a fleet
/// containing only the local host. With one or more URIs, connects to each
/// concurrently. A single-URI fleet stays in single-host UX; two or more
/// URIs implicitly activate multi-host mode in the renderer.
///
/// Per-host connect failure aborts startup if no hosts succeed; otherwise
/// the failures are surfaced via stderr but successful hosts proceed. (For
/// v1 we keep startup strict: all-or-nothing for explicit URIs to avoid
/// silently dropping hosts the operator named.)
///
/// Duplicate URIs are rejected up-front. `HostId` is derived from the URI
/// string; two entries with the same id would collapse `HostFleet::entry`
/// lookups, selection-preservation keys, and `ActivityTracker` keys onto
/// each other. Reject explicitly rather than silently degrade.
pub(crate) async fn connect_fleet(cli: &Cli) -> Result<HostFleet> {
    if cli.ssh_uris.is_empty() {
        let entry = HostEntry {
            id: HostId::local(),
            label: local_hostname(),
            ip_address: HostIdentity::local().ip_address,
            handle: HostHandle::local(),
        };
        return Ok(HostFleet::from_entries(vec![entry]));
    }

    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for uri in &cli.ssh_uris {
        if !seen.insert(uri.as_str()) {
            return Err(anyhow!(
                "duplicate SSH URI '{uri}' on the command line; each host must appear once"
            ));
        }
    }

    // Connect concurrently. try_join_all so a single bad URI fails fast and
    // surfaces clearly rather than silently dropping hosts the operator
    // named on the command line.
    let futures = cli.ssh_uris.iter().map(|uri| async move {
        connect_ssh_entry(uri)
            .await
            .with_context(|| format!("connect host '{uri}'"))
    });
    let entries = futures::future::try_join_all(futures).await?;
    if entries.is_empty() {
        return Err(anyhow!("no hosts configured"));
    }
    Ok(HostFleet::from_entries(entries))
}

async fn connect_ssh_entry(uri: &str) -> Result<HostEntry> {
    let config = SshConfig::parse(uri).context("parse ssh target")?;
    let identity = HostIdentity::new(config.host().to_string(), config.port());
    let handle = config.connect().await.context("connect ssh target")?;
    Ok(HostEntry {
        id: HostId::from_ssh_uri(uri),
        label: identity.hostname,
        ip_address: identity.ip_address,
        handle,
    })
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
