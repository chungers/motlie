use std::fs;
use std::net::{Ipv4Addr, TcpListener};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::io::AsRawFd;
use std::os::unix::net::{SocketAddr, UnixDatagram};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

use motlie_vnet::slirp::{SlirpConfig, SlirpInstance};
use thiserror::Error;
use tracing::debug;

use crate::backend::BackendError;
use crate::network::EgressNetMode;
use crate::orchestrator::PreparedGuest;
use crate::spec::GuestRuntimePaths;

#[derive(Debug, Default, Clone, Copy)]
pub struct VzUserspaceEgressBacking;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VzHostForward {
    pub host_addr: Ipv4Addr,
    pub host_port: u16,
    pub guest_port: u16,
}

impl VzHostForward {
    pub fn tcp(host_addr: Ipv4Addr, host_port: u16, guest_port: u16) -> Self {
        Self {
            host_addr,
            host_port,
            guest_port,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VzUserspaceEgressConfig {
    pub socket_path: PathBuf,
    pub guest_ip: Ipv4Addr,
    pub host_ip: Ipv4Addr,
    pub netmask: Ipv4Addr,
    pub dns_ip: Ipv4Addr,
    pub host_forwards: Vec<VzHostForward>,
    pub log_frames: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VzUserspaceEgressStats {
    pub guest_to_host_frames: u64,
    pub host_to_guest_frames: u64,
}

pub struct VzUserspaceEgressHandle {
    socket_path: PathBuf,
    shutdown: Arc<AtomicBool>,
    worker: Mutex<Option<thread::JoinHandle<Result<VzUserspaceEgressStats, VzEgressError>>>>,
}

impl std::fmt::Debug for VzUserspaceEgressHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VzUserspaceEgressHandle")
            .field("socket_path", &self.socket_path)
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Error)]
pub enum VzEgressError {
    #[error(transparent)]
    Backend(#[from] BackendError),
    #[error("failed to reserve loopback TCP control port: {0}")]
    ReserveControlPort(std::io::Error),
    #[error("failed to write VZ control port file {path}: {source}")]
    WriteControlPortFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to create VZ egress runtime directory {path}: {source}")]
    CreateRuntimeDir {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to remove stale VZ egress socket {path}: {source}")]
    RemoveStaleSocket {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to bind VZ egress socket {path}: {source}")]
    BindSocket {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to tune VZ egress socket {path} option {option}: {source}")]
    TuneSocket {
        path: PathBuf,
        option: &'static str,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to set VZ egress socket nonblocking {path}: {source}")]
    SetNonblocking {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to clone VZ egress socket {path}: {source}")]
    CloneSocket {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to bind VZ egress host forward port {host_port}")]
    HostForwardBind { host_port: u16 },
    #[error("failed to spawn VZ egress worker thread: {source}")]
    SpawnThread {
        #[source]
        source: std::io::Error,
    },
    #[error("VZ egress worker failed before readiness: {reason}")]
    StartupFailed { reason: String },
    #[error("timed out waiting for VZ egress readiness")]
    StartupTimeout,
    #[error("VZ egress worker exited before reporting readiness")]
    StartupDisconnected,
    #[error("VZ egress worker state poisoned")]
    WorkerStatePoisoned,
    #[error("VZ egress worker panicked")]
    WorkerPanicked,
    #[error("VZ egress socket receive failed: {0}")]
    Recv(std::io::Error),
    #[error("VZ egress host-to-guest frame queue closed: {0}")]
    FrameQueueClosed(String),
    #[error("failed to send VZ egress frame to {path}: {source}")]
    SendTo {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

impl VzUserspaceEgressBacking {
    pub fn new() -> Self {
        Self
    }

    pub fn provision(
        &self,
        prepared: &PreparedGuest,
    ) -> Result<Option<VzUserspaceEgressHandle>, VzEgressError> {
        if !matches!(prepared.network_modes.egress, EgressNetMode::VzUserspace) {
            return Ok(None);
        }

        let control_port = reserve_loopback_tcp_port()?;
        let control_port_file = control_port_file(&prepared.runtime_paths);
        if let Some(parent) = control_port_file.parent() {
            fs::create_dir_all(parent).map_err(|source| VzEgressError::CreateRuntimeDir {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        fs::write(&control_port_file, format!("{control_port}\n")).map_err(|source| {
            VzEgressError::WriteControlPortFile {
                path: control_port_file.clone(),
                source,
            }
        })?;

        // VZ keeps the VM runner as the platform boundary, but the libslirp
        // egress backend is now VMM-owned runtime state, parallel to the CH
        // Motlie VNET handle. The launcher may only consume this socket.
        let config = VzUserspaceEgressConfig {
            socket_path: prepared.runtime_paths.vnet_socket.clone(),
            guest_ip: prepared.net_assignment.egress_ipv4.guest,
            host_ip: prepared.net_assignment.egress_ipv4.host,
            netmask: prepared.net_assignment.egress_ipv4.netmask,
            dns_ip: prepared.net_assignment.egress_ipv4.dns,
            host_forwards: vec![VzHostForward::tcp(Ipv4Addr::LOCALHOST, control_port, 22)],
            log_frames: std::env::var_os("MOTLIE_VZ_LOG_FRAMES").is_some(),
        };

        Ok(Some(VzUserspaceEgressHandle::start(config)?))
    }
}

impl VzUserspaceEgressHandle {
    pub fn start(
        config: VzUserspaceEgressConfig,
    ) -> Result<VzUserspaceEgressHandle, VzEgressError> {
        let socket_path = config.socket_path.clone();
        let shutdown = Arc::new(AtomicBool::new(false));
        let worker_shutdown = Arc::clone(&shutdown);
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<(), String>>(1);

        let worker = thread::Builder::new()
            .name("motlie-vz-egress".into())
            .spawn(move || run_vz_userspace_egress(config, worker_shutdown, Some(ready_tx)))
            .map_err(|source| VzEgressError::SpawnThread { source })?;

        match ready_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Ok(())) => Ok(Self {
                socket_path,
                shutdown,
                worker: Mutex::new(Some(worker)),
            }),
            Ok(Err(reason)) => {
                let _ = worker.join();
                Err(VzEgressError::StartupFailed { reason })
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                shutdown.store(true, Ordering::SeqCst);
                let _ = thread::Builder::new()
                    .name("motlie-vz-egress-timeout-cleanup".into())
                    .spawn(move || {
                        let _ = worker.join();
                    });
                Err(VzEgressError::StartupTimeout)
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let join_result = worker.join();
                match join_result {
                    Ok(Err(err)) => Err(VzEgressError::StartupFailed {
                        reason: err.to_string(),
                    }),
                    Ok(Ok(_)) => Err(VzEgressError::StartupDisconnected),
                    Err(_) => Err(VzEgressError::WorkerPanicked),
                }
            }
        }
    }

    pub fn shutdown(&mut self) -> Result<(), VzEgressError> {
        self.shutdown.store(true, Ordering::SeqCst);
        let worker = self
            .worker
            .lock()
            .map_err(|_| VzEgressError::WorkerStatePoisoned)?
            .take();
        if let Some(worker) = worker {
            match worker.join() {
                Ok(Ok(_stats)) => {}
                Ok(Err(err)) => return Err(err),
                Err(_) => return Err(VzEgressError::WorkerPanicked),
            }
        }
        let _ = fs::remove_file(&self.socket_path);
        Ok(())
    }

    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }
}

impl Drop for VzUserspaceEgressHandle {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

pub fn control_port_file(runtime_paths: &GuestRuntimePaths) -> PathBuf {
    runtime_paths.runtime_dir.join("control-port")
}

pub fn run_vz_userspace_egress(
    config: VzUserspaceEgressConfig,
    shutdown: Arc<AtomicBool>,
    startup_tx: Option<mpsc::SyncSender<Result<(), String>>>,
) -> Result<VzUserspaceEgressStats, VzEgressError> {
    let initialized = match initialize_egress(&config, &shutdown) {
        Ok(initialized) => {
            if let Some(tx) = startup_tx {
                let _ = tx.send(Ok(()));
            }
            initialized
        }
        Err(err) => {
            if let Some(tx) = startup_tx {
                let _ = tx.send(Err(err.to_string()));
            }
            return Err(err);
        }
    };

    run_egress_loop(config, initialized, shutdown)
}

struct InitializedEgress {
    sock: UnixDatagram,
    slirp: SlirpInstance,
    peer: Arc<Mutex<Option<PathBuf>>>,
    frame_tx: mpsc::Sender<Vec<u8>>,
    tx_thread: Option<thread::JoinHandle<()>>,
}

fn initialize_egress(
    config: &VzUserspaceEgressConfig,
    shutdown: &Arc<AtomicBool>,
) -> Result<InitializedEgress, VzEgressError> {
    if let Some(parent) = config.socket_path.parent() {
        fs::create_dir_all(parent).map_err(|source| VzEgressError::CreateRuntimeDir {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    if let Err(source) = fs::remove_file(&config.socket_path) {
        if source.kind() != std::io::ErrorKind::NotFound {
            return Err(VzEgressError::RemoveStaleSocket {
                path: config.socket_path.clone(),
                source,
            });
        }
    }

    let sock =
        UnixDatagram::bind(&config.socket_path).map_err(|source| VzEgressError::BindSocket {
            path: config.socket_path.clone(),
            source,
        })?;
    tune_unix_datagram_socket(&sock, &config.socket_path)?;
    sock.set_nonblocking(true)
        .map_err(|source| VzEgressError::SetNonblocking {
            path: config.socket_path.clone(),
            source,
        })?;

    let slirp_config = SlirpConfig {
        guest_ipv4: config.guest_ip,
        host_ipv4: config.host_ip,
        netmask: config.netmask,
        dns: config.dns_ip,
    };
    let slirp = SlirpInstance::new(&slirp_config);
    for fwd in &config.host_forwards {
        slirp
            .add_hostfwd_tcp(
                fwd.host_addr,
                fwd.host_port,
                config.guest_ip,
                fwd.guest_port,
            )
            .map_err(|host_port| VzEgressError::HostForwardBind { host_port })?;
    }

    let peer = Arc::new(Mutex::new(None::<PathBuf>));
    let tx_sock = sock
        .try_clone()
        .map_err(|source| VzEgressError::CloneSocket {
            path: config.socket_path.clone(),
            source,
        })?;
    let tx_peer = Arc::clone(&peer);
    let tx_shutdown = Arc::clone(shutdown);
    let (frame_tx, frame_rx) = mpsc::channel::<Vec<u8>>();
    let tx_thread = thread::Builder::new()
        .name("motlie-vz-egress-tx".into())
        .spawn(move || {
            while let Ok(frame) = frame_rx.recv() {
                loop {
                    if tx_shutdown.load(Ordering::SeqCst) {
                        break;
                    }
                    let peer_path = tx_peer.lock().ok().and_then(|guard| guard.clone());
                    let Some(peer_path) = peer_path else {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    };
                    if let Err(err) = send_with_retry(&tx_sock, &frame, &peer_path) {
                        tracing::warn!("VZ egress send failed: {err}");
                    }
                    break;
                }
            }
        })
        .map_err(|source| VzEgressError::SpawnThread { source })?;

    tracing::info!(
        socket = %config.socket_path.display(),
        guest = %config.guest_ip,
        host = %config.host_ip,
        dns = %config.dns_ip,
        "VZ userspace egress ready"
    );

    Ok(InitializedEgress {
        sock,
        slirp,
        peer,
        frame_tx,
        tx_thread: Some(tx_thread),
    })
}

fn run_egress_loop(
    config: VzUserspaceEgressConfig,
    mut initialized: InitializedEgress,
    shutdown: Arc<AtomicBool>,
) -> Result<VzUserspaceEgressStats, VzEgressError> {
    let mut buf = vec![0u8; 65535];
    let mut guest_to_host_frames: u64 = 0;
    let mut host_to_guest_frames: u64 = 0;

    while !shutdown.load(Ordering::SeqCst) {
        loop {
            match initialized.sock.recv_from(&mut buf) {
                Ok((n, addr)) => {
                    if let Some(path) = sockaddr_path(&addr) {
                        // The runner creates the peer datagram socket after
                        // boot. Record it lazily and send frames back to the
                        // most recent runner peer.
                        if let Ok(mut guard) = initialized.peer.lock() {
                            *guard = Some(path);
                        }
                    }
                    guest_to_host_frames += 1;
                    if config.log_frames
                        && (guest_to_host_frames <= 10 || guest_to_host_frames % 100 == 0)
                    {
                        debug!(
                            frame = guest_to_host_frames,
                            bytes = n,
                            prefix = %hex_prefix(&buf[..n], 32),
                            "VZ egress rx"
                        );
                    }
                    initialized.slirp.input(&buf[..n]);
                }
                Err(err) if err.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(err) => return Err(VzEgressError::Recv(err)),
            }
        }

        let frames = initialized.slirp.run_once_with_max_timeout(10);
        if !frames.is_empty() {
            for frame in frames {
                host_to_guest_frames += 1;
                if config.log_frames
                    && (host_to_guest_frames <= 10 || host_to_guest_frames % 100 == 0)
                {
                    debug!(
                        frame = host_to_guest_frames,
                        bytes = frame.len(),
                        prefix = %hex_prefix(&frame, 32),
                        "VZ egress tx"
                    );
                }
                initialized
                    .frame_tx
                    .send(frame)
                    .map_err(|err| VzEgressError::FrameQueueClosed(err.to_string()))?;
            }
        } else {
            thread::sleep(Duration::from_millis(2));
        }
    }

    drop(initialized.frame_tx);
    if let Some(tx_thread) = initialized.tx_thread.take() {
        let _ = tx_thread.join();
    }
    let _ = fs::remove_file(&config.socket_path);

    Ok(VzUserspaceEgressStats {
        guest_to_host_frames,
        host_to_guest_frames,
    })
}

fn reserve_loopback_tcp_port() -> Result<u16, VzEgressError> {
    let listener =
        TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).map_err(VzEgressError::ReserveControlPort)?;
    let port = listener
        .local_addr()
        .map_err(VzEgressError::ReserveControlPort)?
        .port();
    Ok(port)
}

fn tune_unix_datagram_socket(sock: &UnixDatagram, path: &Path) -> Result<(), VzEgressError> {
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
        return Err(VzEgressError::TuneSocket {
            path: path.to_path_buf(),
            option: "SO_SNDBUF",
            source: std::io::Error::last_os_error(),
        });
    }
    if set_opt(libc::SO_RCVBUF, &rcvbuf) != 0 {
        return Err(VzEgressError::TuneSocket {
            path: path.to_path_buf(),
            option: "SO_RCVBUF",
            source: std::io::Error::last_os_error(),
        });
    }
    Ok(())
}

fn send_with_retry(
    sock: &UnixDatagram,
    frame: &[u8],
    peer_path: &Path,
) -> Result<(), VzEgressError> {
    loop {
        match sock.send_to(frame, peer_path) {
            Ok(_) => return Ok(()),
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(err) if matches!(err.raw_os_error(), Some(libc::ENOBUFS) | Some(libc::EAGAIN)) => {
                thread::sleep(Duration::from_millis(1));
            }
            Err(source) => {
                return Err(VzEgressError::SendTo {
                    path: peer_path.to_path_buf(),
                    source,
                });
            }
        }
    }
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

fn hex_prefix(data: &[u8], limit: usize) -> String {
    data.iter()
        .take(limit)
        .map(|b| format!("{b:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}
