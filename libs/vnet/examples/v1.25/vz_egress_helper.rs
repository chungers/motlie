use std::fs;
use std::net::Ipv4Addr;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::io::AsRawFd;
use std::os::unix::net::{SocketAddr, UnixDatagram};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use anyhow::{Context, Result};
use motlie_vnet::slirp::{SlirpConfig, SlirpInstance};
use tracing::debug;

struct HostForward {
    host_addr: Ipv4Addr,
    host_port: u16,
    guest_port: u16,
}

fn hex_prefix(data: &[u8], limit: usize) -> String {
    data.iter()
        .take(limit)
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn parse_ipv4(label: &str, value: &str) -> Result<Ipv4Addr> {
    value
        .parse::<Ipv4Addr>()
        .with_context(|| format!("invalid {label}: {value}"))
}

fn parse_host_forward(value: &str) -> Result<HostForward> {
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
    Ok(HostForward {
        host_addr,
        host_port,
        guest_port,
    })
}

fn sockaddr_path(addr: &SocketAddr) -> Option<PathBuf> {
    let path = addr.as_pathname()?;
    let bytes = path.as_os_str().as_bytes();
    let trimmed = match bytes.iter().position(|b| *b == 0) {
        Some(idx) => &bytes[..idx],
        None => bytes,
    };
    if trimmed.is_empty() {
        None
    } else {
        Some(PathBuf::from(std::ffi::OsStr::from_bytes(trimmed)))
    }
}

fn print_usage() {
    eprintln!(
        "usage: vz_egress_helper_v1_25 --socket-path <path> [--guest-ip <ipv4>] [--host-ip <ipv4>] [--netmask <ipv4>] [--dns-ip <ipv4>] [--host-forward-tcp <host_ip:host_port:guest_port>] [--log-frames]"
    );
}

fn tune_unix_datagram_socket(sock: &UnixDatagram) -> Result<()> {
    let fd = sock.as_raw_fd();
    let sndbuf: libc::c_int = 256 * 1024;
    let rcvbuf: libc::c_int = 1024 * 1024;

    let set_opt = |optname, value: &libc::c_int| unsafe {
        libc::setsockopt(
            fd,
            libc::SOL_SOCKET,
            optname,
            value as *const _ as *const libc::c_void,
            std::mem::size_of::<libc::c_int>() as libc::socklen_t,
        )
    };

    if set_opt(libc::SO_SNDBUF, &sndbuf) != 0 {
        return Err(std::io::Error::last_os_error()).context("setsockopt SO_SNDBUF");
    }
    if set_opt(libc::SO_RCVBUF, &rcvbuf) != 0 {
        return Err(std::io::Error::last_os_error()).context("setsockopt SO_RCVBUF");
    }
    Ok(())
}

fn send_with_retry(sock: &UnixDatagram, frame: &[u8], peer_path: &PathBuf) -> Result<()> {
    loop {
        match sock.send_to(frame, peer_path) {
            Ok(_) => return Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(err)
                if matches!(
                    err.raw_os_error(),
                    Some(libc::ENOBUFS) | Some(libc::EAGAIN)
                ) =>
            {
                thread::sleep(Duration::from_millis(1));
            }
            Err(err) => {
                return Err(err)
                    .with_context(|| format!("send_to {}", peer_path.display()));
            }
        }
    }
}

fn main() -> Result<()> {
    let mut socket_path: Option<PathBuf> = None;
    let mut guest_ip = Ipv4Addr::new(10, 0, 2, 15);
    let mut host_ip = Ipv4Addr::new(10, 0, 2, 2);
    let mut netmask = Ipv4Addr::new(255, 255, 255, 0);
    let mut dns_ip = Ipv4Addr::new(10, 0, 2, 3);
    let mut forwards: Vec<HostForward> = Vec::new();
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
            other => {
                anyhow::bail!("unknown argument: {other}");
            }
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
    if let Some(parent) = socket_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let _ = fs::remove_file(&socket_path);

    let sock = UnixDatagram::bind(&socket_path)
        .with_context(|| format!("bind unix datagram {}", socket_path.display()))?;
    tune_unix_datagram_socket(&sock)?;
    sock.set_nonblocking(true)?;

    let config = SlirpConfig {
        guest_ipv4: guest_ip,
        host_ipv4: host_ip,
        netmask,
        dns: dns_ip,
    };
    let slirp = SlirpInstance::new(&config);
    for fwd in &forwards {
        slirp
            .add_hostfwd_tcp(fwd.host_addr, fwd.host_port, guest_ip, fwd.guest_port)
            .map_err(|port| anyhow::anyhow!("failed to bind hostfwd port {port}"))?;
    }

    eprintln!(
        "vz_egress_helper_v1_25 ready: socket={} guest={} host={} dns={}",
        socket_path.display(),
        guest_ip,
        host_ip,
        dns_ip
    );

    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_flag = shutdown.clone();
    ctrlc::set_handler(move || {
        shutdown_flag.store(true, Ordering::SeqCst);
    })?;

    let mut peer: Option<PathBuf> = None;
    let mut buf = vec![0u8; 65535];
    let mut guest_to_host_frames: u64 = 0;
    let mut host_to_guest_frames: u64 = 0;

    while !shutdown.load(Ordering::SeqCst) {
        loop {
            match sock.recv_from(&mut buf) {
                Ok((n, addr)) => {
                    if let Some(path) = sockaddr_path(&addr) {
                        peer = Some(path);
                    }
                    guest_to_host_frames += 1;
                    if log_frames && (guest_to_host_frames <= 10 || guest_to_host_frames % 100 == 0)
                    {
                        debug!(
                            frame = guest_to_host_frames,
                            bytes = n,
                            prefix = %hex_prefix(&buf[..n], 32),
                            "vz_egress_helper_v1_25 rx"
                        );
                    }
                    slirp.input(&buf[..n]);
                }
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(err) => return Err(err).context("recv_from unix datagram"),
            }
        }

        let frames = slirp.run_once_with_max_timeout(10);
        if let Some(peer_path) = peer.as_ref() {
            for frame in frames {
                host_to_guest_frames += 1;
                if log_frames && (host_to_guest_frames <= 10 || host_to_guest_frames % 100 == 0) {
                    debug!(
                        frame = host_to_guest_frames,
                        bytes = frame.len(),
                        prefix = %hex_prefix(&frame, 32),
                        "vz_egress_helper_v1_25 tx"
                    );
                }
                if let Err(err) = send_with_retry(&sock, &frame, peer_path) {
                    eprintln!("vz_egress_helper_v1_25 {err:#}");
                }
            }
        } else if frames.is_empty() {
            thread::sleep(Duration::from_millis(2));
        }
    }

    let _ = fs::remove_file(&socket_path);
    Ok(())
}
