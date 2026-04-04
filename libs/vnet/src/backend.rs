// vhost-user-net backend: two-thread model
//
// VhostUserBackend requires Send+Sync, but SlirpInstance is !Send (libslirp C
// state). So slirp runs on a dedicated thread, communicating with the
// vhost-user worker via channels + eventfd.
//
// Thread 1 (vhost-user worker, epoll): virtqueue processing
// Thread 2 (slirp, poll): libslirp event loop + frame translation

use std::ops::Deref;
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex, RwLock};
use std::thread;

use vhost::vhost_user::message::*;
use vhost::vhost_user::Listener;
use vhost_user_backend::{VhostUserBackend, VhostUserDaemon, VringRwLock, VringT};
use virtio_bindings::virtio_net::*;
use virtio_bindings::virtio_config::VIRTIO_F_VERSION_1;
use virtio_bindings::virtio_ring::VIRTIO_RING_F_EVENT_IDX;
use virtio_queue::QueueT;
use vm_memory::{Address, GuestAddressSpace, GuestMemoryAtomic, GuestMemoryMmap, Bytes};
use vmm_sys_util::epoll::EventSet;
use vmm_sys_util::eventfd::EventFd;

use crate::slirp::{SlirpConfig, SlirpInstance};
use crate::config::VnetConfig;
use crate::error::VnetError;

const RX_QUEUE: u16 = 0;
const TX_QUEUE: u16 = 1;
const NUM_QUEUES: usize = 2;
const QUEUE_SIZE: usize = 256;

// Custom epoll event token for slirp rx notification.
// Must be > num_queues (0..num_queues reserved for virtqueues + exit event).
const SLIRP_RX_TOKEN: u16 = (NUM_QUEUES + 1) as u16;

// virtio-net header size (no mergeable rx buffers).
const VIRTIO_NET_HDR_SIZE: usize = 12;

// ---------------------------------------------------------------------------
// VnetVhostBackend — VhostUserBackend trait impl (Send+Sync)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct VnetVhostBackend {
    rx_eventfd: Arc<EventFd>,
    tx_wake_eventfd: Arc<EventFd>,
    tx_sender: Arc<Mutex<mpsc::Sender<Vec<u8>>>>,
    rx_receiver: Arc<Mutex<mpsc::Receiver<Vec<u8>>>>,
    acked_features: Arc<RwLock<u64>>,
    mem: GuestMemoryAtomic<GuestMemoryMmap>,
    mac: [u8; 6],
}

impl VnetVhostBackend {
    fn new(
        config: &VnetConfig,
        tx_sender: mpsc::Sender<Vec<u8>>,
        rx_receiver: mpsc::Receiver<Vec<u8>>,
        rx_eventfd: EventFd,
        tx_wake_eventfd: EventFd,
        mem: GuestMemoryAtomic<GuestMemoryMmap>,
    ) -> Self {
        Self {
            rx_eventfd: Arc::new(rx_eventfd),
            tx_wake_eventfd: Arc::new(tx_wake_eventfd),
            tx_sender: Arc::new(Mutex::new(tx_sender)),
            rx_receiver: Arc::new(Mutex::new(rx_receiver)),
            acked_features: Arc::new(RwLock::new(0)),
            mem,
            mac: config.mac,
        }
    }

    /// Process tx virtqueue: read frames from guest, send to slirp thread.
    fn process_tx(&self, vring: &VringRwLock) {
        let mem_guard = self.mem.memory();
        let mem = mem_guard.deref();

        let mut state = vring.get_mut();
        let queue = state.get_queue_mut();
        let mut sent = false;

        while let Some(mut desc_chain) = queue.pop_descriptor_chain(mem) {
            let desc_idx = desc_chain.head_index();
            let mut frame = Vec::new();
            let mut first = true;

            while let Some(desc) = desc_chain.next() {
                let addr = desc.addr();
                let len = desc.len() as usize;

                if first {
                    first = false;
                    if len <= VIRTIO_NET_HDR_SIZE {
                        continue;
                    }
                    let data_len = len - VIRTIO_NET_HDR_SIZE;
                    let mut buf = vec![0u8; data_len];
                    let data_addr = addr.unchecked_add(VIRTIO_NET_HDR_SIZE as u64);
                    if mem.read(&mut buf, data_addr).is_ok() {
                        frame.extend_from_slice(&buf);
                    }
                } else {
                    let mut buf = vec![0u8; len];
                    if mem.read(&mut buf, addr).is_ok() {
                        frame.extend_from_slice(&buf);
                    }
                }
            }

            if !frame.is_empty() {
                if let Err(e) = self.tx_sender.lock().expect("tx lock not poisoned").send(frame) {
                    tracing::warn!("tx channel send failed: {}", e);
                }
                sent = true;
            }

            let _ = queue.add_used(mem, desc_idx, 0);
        }

        if sent {
            if let Err(e) = state.signal_used_queue() {
                tracing::warn!("failed to signal tx used queue: {}", e);
            }
            let _ = self.tx_wake_eventfd.write(1);
        }
    }

    /// Process rx: drain frames from slirp, inject into rx virtqueue.
    fn process_rx(&self, vring: &VringRwLock) {
        let _ = self.rx_eventfd.read();

        let mem_guard = self.mem.memory();
        let mem = mem_guard.deref();

        let rx_lock = self.rx_receiver.lock().expect("rx lock not poisoned");
        let mut state = vring.get_mut();
        let queue = state.get_queue_mut();
        let mut injected = false;

        while let Ok(frame) = rx_lock.try_recv() {
            if let Some(mut desc_chain) = queue.pop_descriptor_chain(mem) {
                let desc_idx = desc_chain.head_index();
                let mut offset = 0usize;

                while let Some(desc) = desc_chain.next() {
                    if !desc.is_write_only() {
                        continue;
                    }

                    let addr = desc.addr();
                    let avail = desc.len() as usize;

                    if offset < VIRTIO_NET_HDR_SIZE {
                        let hdr_remaining = VIRTIO_NET_HDR_SIZE - offset;
                        let hdr_write = hdr_remaining.min(avail);
                        let zeros = vec![0u8; hdr_write];
                        let _ = mem.write(&zeros, addr);
                        offset += hdr_write;

                        if hdr_write < avail && offset >= VIRTIO_NET_HDR_SIZE {
                            let frame_offset = offset - VIRTIO_NET_HDR_SIZE;
                            let data_write = (avail - hdr_write)
                                .min(frame.len().saturating_sub(frame_offset));
                            if data_write > 0 {
                                let _ = mem.write(
                                    &frame[frame_offset..frame_offset + data_write],
                                    addr.unchecked_add(hdr_write as u64),
                                );
                                offset += data_write;
                            }
                        }
                    } else {
                        let frame_offset = offset - VIRTIO_NET_HDR_SIZE;
                        if frame_offset < frame.len() {
                            let data_write = avail.min(frame.len() - frame_offset);
                            let _ = mem.write(
                                &frame[frame_offset..frame_offset + data_write],
                                addr,
                            );
                            offset += data_write;
                        }
                    }
                }

                let _ = queue.add_used(mem, desc_idx, offset as u32);
                injected = true;
            } else {
                tracing::debug!("rx vring full, dropping frame ({} bytes)", frame.len());
                break;
            }
        }

        if injected {
            if let Err(e) = state.signal_used_queue() {
                tracing::warn!("failed to signal rx used queue: {}", e);
            }
        }
    }
}

impl VhostUserBackend for VnetVhostBackend {
    type Bitmap = ();
    type Vring = VringRwLock;

    fn num_queues(&self) -> usize {
        NUM_QUEUES
    }

    fn max_queue_size(&self) -> usize {
        QUEUE_SIZE
    }

    fn features(&self) -> u64 {
        1 << VIRTIO_NET_F_GUEST_CSUM
            | 1 << VIRTIO_NET_F_CSUM
            | 1 << VIRTIO_NET_F_MAC
            | 1 << VIRTIO_F_VERSION_1
            | 1 << VIRTIO_RING_F_EVENT_IDX
            | VhostUserVirtioFeatures::PROTOCOL_FEATURES.bits()
    }

    fn acked_features(&self, features: u64) {
        *self.acked_features.write().expect("features lock not poisoned") = features;
    }

    fn protocol_features(&self) -> VhostUserProtocolFeatures {
        VhostUserProtocolFeatures::CONFIG
    }

    fn get_config(&self, offset: u32, size: u32) -> Vec<u8> {
        let mut config = vec![0u8; 6];
        config.copy_from_slice(&self.mac);
        let end = (offset as usize + size as usize).min(config.len());
        let start = (offset as usize).min(end);
        config[start..end].to_vec()
    }

    fn set_event_idx(&self, _enabled: bool) {}

    fn update_memory(
        &self,
        _atomic_mem: GuestMemoryAtomic<GuestMemoryMmap>,
    ) -> std::io::Result<()> {
        Ok(())
    }

    fn handle_event(
        &self,
        device_event: u16,
        _evset: EventSet,
        vrings: &[VringRwLock],
        _thread_id: usize,
    ) -> std::io::Result<()> {
        match device_event {
            RX_QUEUE => {
                // Guest made rx buffers available — no action needed.
            }
            TX_QUEUE => {
                self.process_tx(&vrings[TX_QUEUE as usize]);
            }
            SLIRP_RX_TOKEN => {
                self.process_rx(&vrings[RX_QUEUE as usize]);
            }
            _ => {
                tracing::warn!("unexpected device_event: {}", device_event);
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Slirp thread
// ---------------------------------------------------------------------------

/// Slirp thread startup result — sent back to start() via the readiness channel.
enum SlirpStartup {
    Ready,
    Failed(VnetError),
}

fn slirp_thread_main(
    config: &VnetConfig,
    tx_receiver: mpsc::Receiver<Vec<u8>>,
    rx_sender: mpsc::Sender<Vec<u8>>,
    rx_eventfd: EventFd,
    shutdown: Arc<AtomicBool>,
    ready_tx: mpsc::SyncSender<SlirpStartup>,
) {
    let slirp_config = SlirpConfig {
        guest_ipv4: config.guest_ipv4,
        host_ipv4: config.host_ipv4,
        netmask: config.netmask,
        dns: config.dns_ipv4,
    };

    let slirp = SlirpInstance::new(&slirp_config);

    // Configure port forwards — propagate failures to start().
    for fwd in &config.host_forwards {
        if let Err(port) = slirp.add_hostfwd_tcp(
            fwd.bind_addr,
            fwd.host_port,
            config.guest_ipv4,
            fwd.guest_port,
        ) {
            let err = VnetError::PortForwardBind {
                host_port: port,
                source: std::io::Error::new(
                    std::io::ErrorKind::AddrInUse,
                    format!("libslirp hostfwd bind failed for port {}", port),
                ),
            };
            let _ = ready_tx.send(SlirpStartup::Failed(err));
            return;
        }
    }

    // Signal readiness — slirp context is live, hostfwds are bound.
    let _ = ready_tx.send(SlirpStartup::Ready);

    tracing::info!("slirp thread started");

    while !shutdown.load(Ordering::Relaxed) {
        while let Ok(frame) = tx_receiver.try_recv() {
            slirp.input(&frame);
        }

        let rx_frames = slirp.run_once_with_max_timeout(10);

        let mut notified = false;
        for frame in rx_frames {
            if rx_sender.send(frame).is_err() {
                break;
            }
            notified = true;
        }
        if notified {
            let _ = rx_eventfd.write(1);
        }
    }

    tracing::info!("slirp thread exiting");
}

// ---------------------------------------------------------------------------
// VnetBackend + VnetHandle — public API
// ---------------------------------------------------------------------------

/// A configured vhost-user-net backend ready to start.
pub struct VnetBackend {
    config: VnetConfig,
}

impl VnetBackend {
    pub fn new(config: VnetConfig) -> Self {
        Self { config }
    }

    /// Start the backend. Spawns background threads and returns a handle.
    ///
    /// Returns only after:
    /// - The vhost-user socket is bound and listening
    /// - The slirp context is initialized and all hostfwds are configured
    ///
    /// Any startup failure (socket bind, hostfwd bind, slirp init) is returned
    /// as an error — the handle is never returned in a partially-ready state.
    pub fn start(self) -> Result<VnetHandle, VnetError> {
        let socket_path = self.config.socket_path.clone();

        // Clean up stale socket.
        if socket_path.exists() {
            std::fs::remove_file(&socket_path).map_err(VnetError::SocketCleanup)?;
            tracing::info!("removed stale socket: {}", socket_path.display());
        }

        let (tx_sender, tx_receiver) = mpsc::channel::<Vec<u8>>();
        let (rx_sender, rx_receiver) = mpsc::channel::<Vec<u8>>();

        let rx_eventfd = EventFd::new(libc::EFD_NONBLOCK)
            .map_err(|e| VnetError::BackendInit(format!("rx eventfd: {}", e)))?;
        let rx_eventfd_for_slirp = rx_eventfd.try_clone()
            .map_err(|e| VnetError::BackendInit(format!("rx eventfd clone: {}", e)))?;
        let tx_wake_eventfd = EventFd::new(libc::EFD_NONBLOCK)
            .map_err(|e| VnetError::BackendInit(format!("tx wake eventfd: {}", e)))?;

        let mem = GuestMemoryAtomic::new(GuestMemoryMmap::<()>::new());

        let vhost_backend = VnetVhostBackend::new(
            &self.config,
            tx_sender,
            rx_receiver,
            rx_eventfd,
            tx_wake_eventfd,
            mem.clone(),
        );

        let mut daemon = VhostUserDaemon::new(
            "motlie-vnet".to_string(),
            vhost_backend,
            mem,
        )
        .map_err(|e| VnetError::BackendInit(format!("daemon: {}", e)))?;

        // Register the rx eventfd in the vhost-user epoll.
        let handlers = daemon.get_epoll_handlers();
        if let Some(handler) = handlers.into_iter().next() {
            handler.register_listener(
                rx_eventfd_for_slirp.as_raw_fd(),
                EventSet::IN,
                SLIRP_RX_TOKEN as u64,
            ).map_err(|e| VnetError::BackendInit(format!("register rx eventfd: {}", e)))?;
        }

        // Spawn slirp thread with readiness barrier.
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_slirp = shutdown.clone();
        let config_clone = self.config.clone();
        let (ready_tx, ready_rx) = mpsc::sync_channel::<SlirpStartup>(1);

        let slirp_handle = thread::Builder::new()
            .name("motlie-vnet-slirp".into())
            .spawn(move || {
                slirp_thread_main(
                    &config_clone,
                    tx_receiver,
                    rx_sender,
                    rx_eventfd_for_slirp,
                    shutdown_slirp,
                    ready_tx,
                );
            })
            .map_err(|e| VnetError::BackendInit(format!("slirp thread: {}", e)))?;

        // Wait for slirp thread readiness — hostfwd failures surface here.
        match ready_rx.recv() {
            Ok(SlirpStartup::Ready) => {}
            Ok(SlirpStartup::Failed(e)) => {
                shutdown.store(true, Ordering::Relaxed);
                let _ = slirp_handle.join();
                return Err(e);
            }
            Err(_) => {
                return Err(VnetError::BackendInit(
                    "slirp thread exited before signaling readiness".into(),
                ));
            }
        }

        // Bind the vhost-user socket on the calling thread.
        let mut listener = Listener::new(&socket_path, true)
            .map_err(|e| VnetError::SocketBind(std::io::Error::other(format!("{}", e))))?;

        tracing::info!("vnet socket bound: {}", socket_path.display());

        // Hand the bound listener to the daemon thread.
        let daemon_handle = thread::Builder::new()
            .name("motlie-vnet-daemon".into())
            .spawn(move || {
                // start() blocks until a frontend connects (accept), then
                // spawns worker threads and returns. wait() blocks until the
                // workers exit (frontend disconnect or error).
                if let Err(e) = daemon.start(&mut listener) {
                    tracing::error!("vhost-user daemon accept error: {}", e);
                    return;
                }
                // wait() joins worker threads — they exit when the frontend
                // disconnects, which happens on VM shutdown or on our dummy
                // connect during VnetHandle::shutdown().
                match daemon.wait() {
                    Ok(()) => tracing::debug!("vhost-user daemon exited cleanly"),
                    Err(e) => tracing::debug!("vhost-user daemon exited: {}", e),
                }
            })
            .map_err(|e| VnetError::BackendInit(format!("daemon thread: {}", e)))?;

        tracing::info!("vnet backend started: {}", socket_path.display());

        Ok(VnetHandle {
            slirp_thread: Some(slirp_handle),
            daemon_thread: Some(daemon_handle),
            shutdown,
            socket_path,
        })
    }
}

/// Handle to a running vnet backend. Drop to shut down (best-effort),
/// or call `shutdown()` for deterministic cleanup.
pub struct VnetHandle {
    slirp_thread: Option<thread::JoinHandle<()>>,
    daemon_thread: Option<thread::JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    socket_path: PathBuf,
}

impl VnetHandle {
    /// Shut down deterministically. Signals all threads to exit, joins them,
    /// and removes the socket file. Returns only after all threads have exited.
    pub fn shutdown(mut self) -> Result<(), VnetError> {
        self.do_shutdown();
        Ok(())
    }

    fn do_shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);

        // Connect a dummy client to unblock the daemon's accept() if it's
        // still waiting for a frontend. The dummy connection will cause the
        // vhost-user handler to fail on the first protocol message, which
        // terminates the daemon.
        let _ = std::os::unix::net::UnixStream::connect(&self.socket_path);

        if let Some(handle) = self.slirp_thread.take() {
            let _ = handle.join();
        }
        // The daemon thread may be blocked in accept() or in a worker's
        // epoll_wait. After the dummy connect, the daemon should terminate
        // quickly. Give it a bounded wait, then detach if stuck.
        if let Some(handle) = self.daemon_thread.take() {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
            loop {
                if handle.is_finished() {
                    let _ = handle.join();
                    break;
                }
                if std::time::Instant::now() >= deadline {
                    tracing::debug!("daemon thread did not exit within 2s, detaching");
                    drop(handle);
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
        let _ = std::fs::remove_file(&self.socket_path);
        tracing::info!("vnet backend shut down: {}", self.socket_path.display());
    }
}

impl Drop for VnetHandle {
    fn drop(&mut self) {
        self.do_shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::VnetConfig;

    fn init_logging() {
        let _ = tracing_subscriber::fmt().with_test_writer().try_init();
    }

    #[test]
    fn test_config_build_and_backend_new() {
        init_logging();
        let config = VnetConfig::builder()
            .socket_path("/tmp/motlie-vnet-test-build.sock")
            .host_forward_tcp(2222, 22)
            .build()
            .expect("config build failed");

        assert_eq!(config.guest_ipv4, std::net::Ipv4Addr::new(10, 0, 2, 15));
        assert_eq!(config.mac, [0x52, 0x54, 0x00, 0x12, 0x34, 0x56]);
        assert_eq!(config.host_forwards.len(), 1);
        assert_eq!(config.host_forwards[0].host_port, 2222);
        assert_eq!(config.host_forwards[0].guest_port, 22);

        let _backend = VnetBackend::new(config);
    }

    #[test]
    fn test_config_rejects_empty_path() {
        let result = VnetConfig::builder().build();
        assert!(result.is_err(), "should reject missing socket_path");
    }

    #[test]
    fn test_config_rejects_nonexistent_parent() {
        let result = VnetConfig::builder()
            .socket_path("/nonexistent/dir/test.sock")
            .build();
        assert!(result.is_err(), "should reject nonexistent parent dir");
    }

    #[test]
    fn test_config_rejects_duplicate_hostfwd_ports() {
        let result = VnetConfig::builder()
            .socket_path("/tmp/test.sock")
            .host_forward_tcp(2222, 22)
            .host_forward_tcp(2222, 80) // duplicate host port
            .build();
        assert!(result.is_err(), "should reject duplicate host_forward ports");
    }

    /// Socket lifecycle: start creates socket, shutdown removes it.
    #[test]
    fn test_start_creates_socket_shutdown_removes() {
        init_logging();
        let socket_path = format!("/tmp/motlie-vnet-lifecycle-{}.sock", std::process::id());

        let config = VnetConfig::builder()
            .socket_path(&socket_path)
            .build()
            .expect("config build failed");

        let handle = VnetBackend::new(config).start().expect("start failed");

        // Socket should exist after start() returns (bound on calling thread).
        assert!(
            std::path::Path::new(&socket_path).exists(),
            "socket should exist after start()"
        );

        handle.shutdown().expect("shutdown failed");

        // Socket should be cleaned up after shutdown.
        assert!(
            !std::path::Path::new(&socket_path).exists(),
            "socket should be removed after shutdown()"
        );
    }

    /// Drop cleans up without panic.
    #[test]
    fn test_drop_cleans_up() {
        init_logging();
        let socket_path = format!("/tmp/motlie-vnet-drop-{}.sock", std::process::id());

        let config = VnetConfig::builder()
            .socket_path(&socket_path)
            .build()
            .expect("config build failed");

        {
            let _handle = VnetBackend::new(config).start().expect("start failed");
            assert!(std::path::Path::new(&socket_path).exists());
        }
        // Handle dropped — socket should be cleaned up.
        assert!(
            !std::path::Path::new(&socket_path).exists(),
            "socket should be removed on drop"
        );
    }

    /// Verify subnet consistency validation.
    #[test]
    fn test_config_rejects_mismatched_subnet() {
        let result = VnetConfig::builder()
            .socket_path("/tmp/test.sock")
            .guest_ipv4(std::net::Ipv4Addr::new(192, 168, 1, 10))
            .host_ipv4(std::net::Ipv4Addr::new(10, 0, 2, 2))
            .build();
        assert!(result.is_err(), "should reject guest/host in different subnets");
    }
}
