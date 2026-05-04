use std::net::Ipv4Addr;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use motlie_vmm::backend::vz::egress::{
    run_vz_userspace_egress, VzHostForward, VzUserspaceEgressConfig,
};

fn parse_ipv4(label: &str, value: &str) -> Result<Ipv4Addr> {
    value
        .parse::<Ipv4Addr>()
        .with_context(|| format!("invalid {label}: {value}"))
}

fn parse_host_forward(value: &str) -> Result<VzHostForward> {
    let parts: Vec<&str> = value.split(':').collect();
    if parts.len() != 3 {
        anyhow::bail!("host forward must be host_addr:host_port:guest_port");
    }
    let host_addr = parse_ipv4("host forward address", parts[0])?;
    let host_port = parts[1]
        .parse::<u16>()
        .with_context(|| format!("invalid host port in {value}"))?;
    let guest_port = parts[2]
        .parse::<u16>()
        .with_context(|| format!("invalid guest port in {value}"))?;
    Ok(VzHostForward::tcp(host_addr, host_port, guest_port))
}

fn print_usage() {
    eprintln!(
        "usage: vz_egress_helper_v1_5 --socket-path <path> [--guest-ip <ipv4>] [--host-ip <ipv4>] [--netmask <ipv4>] [--dns-ip <ipv4>] [--host-forward-tcp <host_ip:host_port:guest_port>] [--log-frames]"
    );
}

fn main() -> Result<()> {
    let mut socket_path: Option<PathBuf> = None;
    let mut guest_ip = Ipv4Addr::new(10, 0, 2, 15);
    let mut host_ip = Ipv4Addr::new(10, 0, 2, 2);
    let mut netmask = Ipv4Addr::new(255, 255, 255, 0);
    let mut dns_ip = Ipv4Addr::new(10, 0, 2, 3);
    let mut forwards: Vec<VzHostForward> = Vec::new();
    let mut log_frames = false;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--socket-path" if i + 1 < args.len() => {
                socket_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--guest-ip" if i + 1 < args.len() => {
                guest_ip = parse_ipv4("guest ip", &args[i + 1])?;
                i += 2;
            }
            "--host-ip" if i + 1 < args.len() => {
                host_ip = parse_ipv4("host ip", &args[i + 1])?;
                i += 2;
            }
            "--netmask" if i + 1 < args.len() => {
                netmask = parse_ipv4("netmask", &args[i + 1])?;
                i += 2;
            }
            "--dns-ip" if i + 1 < args.len() => {
                dns_ip = parse_ipv4("dns ip", &args[i + 1])?;
                i += 2;
            }
            "--host-forward-tcp" if i + 1 < args.len() => {
                forwards.push(parse_host_forward(&args[i + 1])?);
                i += 2;
            }
            "--log-frames" => {
                log_frames = true;
                i += 1;
            }
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            other => anyhow::bail!("unknown argument: {other}"),
        }
    }

    if log_frames {
        let _ = tracing_subscriber::fmt()
            .with_target(false)
            .with_writer(std::io::stderr)
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("debug")),
            )
            .try_init();
    }

    let socket_path = socket_path.context("--socket-path is required")?;
    let config = VzUserspaceEgressConfig {
        socket_path,
        guest_ip,
        host_ip,
        netmask,
        dns_ip,
        host_forwards: forwards,
        log_frames,
    };

    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_flag = Arc::clone(&shutdown);
    ctrlc::set_handler(move || {
        shutdown_flag.store(true, Ordering::SeqCst);
    })?;

    let stats = run_vz_userspace_egress(config, shutdown, None)?;
    eprintln!(
        "vz_egress_helper_v1_5 summary: guest_to_host_frames={} host_to_guest_frames={}",
        stats.guest_to_host_frames, stats.host_to_guest_frames
    );
    Ok(())
}
