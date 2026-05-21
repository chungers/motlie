use std::net::{IpAddr, ToSocketAddrs};
use std::process::Command;

use anyhow::{anyhow, Context, Result};
use motlie_tmux::{HostHandle, SshConfig, SSH_DEFAULT_PORT};

use crate::cli::Cli;
use crate::model::{HostEntry, HostFleet, HostId, HostSlot};

#[derive(Debug, Clone)]
pub(crate) struct HostConnectSpec {
    pub(crate) id: HostId,
    pub(crate) uri: String,
    pub(crate) label: String,
    pub(crate) alias: String,
    label_override: Option<String>,
    config: SshConfig,
}

/// Initial selector state plus the retry work needed to complete it.
///
/// `fleet` is usable immediately by the TUI. `retry_specs` contains the SSH
/// hosts represented by connecting slots in that fleet; the caller starts
/// background retries for these specs before entering the selector loop.
pub(crate) struct InitialHostFleet {
    pub(crate) fleet: HostFleet,
    pub(crate) retry_specs: Vec<HostConnectSpec>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IpAddressSource {
    Local,
    Alias { port: u16 },
}

/// Build the selector's initial fleet. Localhost is connected immediately and
/// every explicit SSH URI is registered as a configured host for background
/// connection/retry.
///
/// Duplicate URIs are rejected up-front. `HostId` is derived from the URI
/// string; two entries with the same id would collapse `HostFleet::entry`
/// lookups, selection-preservation keys, and `ActivityTracker` keys onto
/// each other. Reject explicitly rather than silently degrade.
pub(crate) async fn connect_initial_fleet(cli: &Cli) -> Result<InitialHostFleet> {
    let specs = ssh_connect_specs(cli)?;
    let mut local = connect_local_entry().await;
    apply_label_override(&mut local, cli.host_alias_override(0));
    if specs.is_empty() {
        return Ok(InitialHostFleet {
            fleet: HostFleet::from_entries(vec![local]),
            retry_specs: Vec::new(),
        });
    }

    let mut hosts = Vec::with_capacity(specs.len() + 1);
    hosts.push(HostSlot::connected(&local));
    hosts.extend(
        specs.iter().map(|spec| {
            HostSlot::connecting(spec.id.clone(), spec.label.clone(), spec.alias.clone())
        }),
    );
    Ok(InitialHostFleet {
        fleet: HostFleet::from_configured_hosts(vec![local], hosts),
        retry_specs: specs,
    })
}

fn ssh_connect_specs(cli: &Cli) -> Result<Vec<HostConnectSpec>> {
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    cli.ssh_uris
        .iter()
        .enumerate()
        .map(|(index, uri)| {
            if !seen.insert(uri.as_str()) {
                return Err(anyhow!(
                    "duplicate SSH URI '{uri}' on the command line; each host must appear once"
                ));
            }
            let config = SshConfig::parse(uri).context("parse ssh target")?;
            let alias = config.host().to_string();
            let label_override = cli.host_alias_override(index + 1).map(str::to_string);
            let label = label_override.clone().unwrap_or_else(|| alias.clone());
            Ok(HostConnectSpec {
                id: HostId::from_ssh_uri(uri),
                uri: uri.clone(),
                label,
                alias,
                label_override,
                config,
            })
        })
        .collect()
}

pub(crate) async fn connect_local_entry() -> HostEntry {
    let handle = HostHandle::local();
    let fallback_label = local_hostname();
    let alias = handle.host_alias().to_string();
    build_host_entry(
        HostId::local(),
        handle,
        alias,
        fallback_label,
        IpAddressSource::Local,
    )
    .await
}

pub(crate) async fn connect_ssh_spec(spec: &HostConnectSpec) -> Result<HostEntry> {
    let mut entry = connect_ssh_config(spec.id.clone(), spec.config.clone()).await?;
    apply_label_override(&mut entry, spec.label_override.as_deref());
    Ok(entry)
}

async fn connect_ssh_config(id: HostId, config: SshConfig) -> Result<HostEntry> {
    let port = config.port();
    let handle = config.connect().await.context("connect ssh target")?;
    let alias = handle.host_alias().to_string();
    Ok(build_host_entry(
        id,
        handle,
        alias.clone(),
        alias,
        IpAddressSource::Alias { port },
    )
    .await)
}

async fn build_host_entry(
    id: HostId,
    handle: HostHandle,
    alias: String,
    fallback_label: String,
    ip_source: IpAddressSource,
) -> HostEntry {
    let label = handle
        .tmux_hostname()
        .await
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or(fallback_label);
    let ip_address = match ip_source {
        IpAddressSource::Local => {
            local_ip_address(&label).unwrap_or_else(|| resolve_ip_address(&label, SSH_DEFAULT_PORT))
        }
        IpAddressSource::Alias { port } => resolve_ip_address(&alias, port),
    };
    HostEntry {
        id,
        label,
        alias,
        ip_address,
        handle,
    }
}

fn apply_label_override(entry: &mut HostEntry, override_label: Option<&str>) {
    if let Some(label) = override_label {
        entry.label = label.to_string();
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
