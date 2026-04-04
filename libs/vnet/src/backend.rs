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
const SLIRP_RX_TOKEN: u16 = NUM_QUEUES as u16;

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
                        continue; // Skip header-only descriptor.
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
                if let Err(e) = self.tx_sender.lock().unwrap().send(frame) {
                    log::warn!("tx channel send failed: {}", e);
                }
                sent = true;
            }

            let _ = queue.add_used(mem, desc_idx, 0);
        }

        if sent {
            let _ = self.tx_wake_eventfd.write(1);
        }
    }

    /// Process rx: drain frames from slirp, inject into rx virtqueue.
    fn process_rx(&self, vring: &VringRwLock) {
        // Consume the eventfd notification.
        let _ = self.rx_eventfd.read();

        let mem_guard = self.mem.memory();
        let mem = mem_guard.deref();

        let rx_lock = self.rx_receiver.lock().unwrap();
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
                log::debug!("rx vring full, dropping frame ({} bytes)", frame.len());
                break;
            }
        }

        if injected {
            if let Err(e) = state.signal_used_queue() {
                log::warn!("failed to signal rx used queue: {}", e);
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
        *self.acked_features.write().unwrap() = features;
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
                // Guest made rx buffers available. No action needed — we inject
                // rx frames when notified by slirp via SLIRP_RX_TOKEN.
            }
            TX_QUEUE => {
                self.process_tx(&vrings[TX_QUEUE as usize]);
            }
            SLIRP_RX_TOKEN => {
                self.process_rx(&vrings[RX_QUEUE as usize]);
            }
            _ => {
                log::warn!("unexpected device_event: {}", device_event);
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Slirp thread
// ---------------------------------------------------------------------------

fn slirp_thread_main(
    config: &VnetConfig,
    tx_receiver: mpsc::Receiver<Vec<u8>>,
    rx_sender: mpsc::Sender<Vec<u8>>,
    rx_eventfd: EventFd,
    shutdown: Arc<AtomicBool>,
) {
    let slirp_config = SlirpConfig {
        guest_ipv4: config.guest_ipv4,
        host_ipv4: config.host_ipv4,
        netmask: config.netmask,
        dns: config.dns_ipv4,
    };

    let slirp = SlirpInstance::new(&slirp_config);

    for fwd in &config.host_forwards {
        if let Err(port) = slirp.add_hostfwd_tcp(
            fwd.bind_addr,
            fwd.host_port,
            config.guest_ipv4,
            fwd.guest_port,
        ) {
            log::error!("hostfwd failed for port {}", port);
        }
    }

    log::info!("slirp thread started");

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

    log::info!("slirp thread exiting");
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
    pub fn start(self) -> Result<VnetHandle, VnetError> {
        let socket_path = self.config.socket_path.clone();

        // Clean up stale socket.
        if socket_path.exists() {
            std::fs::remove_file(&socket_path).map_err(VnetError::SocketCleanup)?;
            log::info!("removed stale socket: {}", socket_path.display());
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

        // Spawn slirp thread.
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_slirp = shutdown.clone();
        let config_clone = self.config.clone();

        let slirp_handle = thread::Builder::new()
            .name("motlie-vnet-slirp".into())
            .spawn(move || {
                slirp_thread_main(
                    &config_clone,
                    tx_receiver,
                    rx_sender,
                    rx_eventfd_for_slirp,
                    shutdown_slirp,
                );
            })
            .map_err(|e| VnetError::BackendInit(format!("slirp thread: {}", e)))?;

        // Start daemon on its own thread.
        let socket_path_for_daemon = socket_path.clone();
        let daemon_handle = thread::Builder::new()
            .name("motlie-vnet-daemon".into())
            .spawn(move || {
                if let Err(e) = daemon.serve(&socket_path_for_daemon) {
                    log::error!("vhost-user daemon error: {}", e);
                }
            })
            .map_err(|e| VnetError::BackendInit(format!("daemon thread: {}", e)))?;

        log::info!("vnet backend started: {}", socket_path.display());

        Ok(VnetHandle {
            slirp_thread: Some(slirp_handle),
            daemon_thread: Some(daemon_handle),
            shutdown,
            socket_path,
        })
    }
}

/// Handle to a running vnet backend.
pub struct VnetHandle {
    slirp_thread: Option<thread::JoinHandle<()>>,
    daemon_thread: Option<thread::JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    socket_path: PathBuf,
}

impl VnetHandle {
    pub fn shutdown(mut self) -> Result<(), VnetError> {
        self.do_shutdown();
        Ok(())
    }

    fn do_shutdown(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);

        // Connect a dummy client to unblock the daemon's accept().
        let _ = std::os::unix::net::UnixStream::connect(&self.socket_path);

        if let Some(handle) = self.slirp_thread.take() {
            let _ = handle.join();
        }
        // The daemon thread may be stuck in the vhost-user protocol handler
        // loop even after accept returns. Detach rather than join to avoid
        // blocking shutdown indefinitely. The thread will exit when the
        // process exits or when the socket is removed.
        if let Some(handle) = self.daemon_thread.take() {
            drop(handle);
        }
        let _ = std::fs::remove_file(&self.socket_path);
        log::info!("vnet backend shut down: {}", self.socket_path.display());
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
        let _ = env_logger::builder().is_test(true).try_init();
    }

    /// Task 2.3.8: verify config builder + backend construction.
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

        // Backend construction should succeed.
        let _backend = VnetBackend::new(config);
    }

    /// Task 2.3.9: verify config validation rejects bad paths.
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

    // Note: full start/shutdown/socket lifecycle tests require a real
    // vhost-user frontend (CH). These are covered in Phase 3 e2e tests.
    // The daemon's serve() blocks on accept() until a frontend connects,
    // making unit-level lifecycle tests impractical.
}
