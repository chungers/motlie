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
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "debug-trace")]
use std::sync::atomic::AtomicUsize;
use std::sync::{mpsc, Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use vhost::vhost_user::message::*;
use vhost::vhost_user::{Error as VhostUserError, Listener};
use vhost_user_backend::{VhostUserBackend, VhostUserDaemon, VringRwLock, VringT};
use virtio_bindings::virtio_net::*;
use virtio_bindings::virtio_config::VIRTIO_F_VERSION_1;
use virtio_queue::QueueT;
use vm_memory::{Address, GuestAddressSpace, GuestMemoryAtomic, GuestMemoryMmap, Bytes};
use vmm_sys_util::epoll::EventSet;
use vmm_sys_util::event::{new_event_consumer_and_notifier, EventConsumer, EventFlag, EventNotifier};
use vmm_sys_util::eventfd::EventFd;

use crate::slirp::{SlirpConfig, SlirpInstance};
use crate::config::VnetConfig;
use crate::error::VnetError;

type VnetMemory = GuestMemoryAtomic<GuestMemoryMmap>;

/// Returns true for vhost-user errors that indicate a benign frontend
/// disconnect (guest shutdown, dummy-connect during our shutdown, etc.)
/// rather than a real runtime failure. Only these are silenced in the
/// daemon thread; everything else propagates to shutdown().
fn is_benign_vhost_error(e: &vhost_user_backend::Error) -> bool {
    use vhost_user_backend::Error as BE;
    matches!(
        e,
        BE::HandleRequest(VhostUserError::Disconnected)
            | BE::HandleRequest(VhostUserError::PartialMessage)
            | BE::HandleRequest(VhostUserError::SocketBroken(_))
    )
}

const RX_QUEUE: u16 = 0;
const TX_QUEUE: u16 = 1;
const NUM_QUEUES: usize = 2;
const QUEUE_SIZE: usize = 256;

// Custom epoll event token for slirp rx notification.
// Must be > num_queues (0..=num_queues reserved for virtqueues + exit event).
const SLIRP_RX_TOKEN: u16 = (NUM_QUEUES + 1) as u16;

// virtio-net header size used by the current CH/Linux guest combination.
// Live packet inspection shows the guest presents 12 bytes before the Ethernet
// frame on TX (two zero bytes followed by the expected L2 header), so the
// backend must strip/write 12 bytes here.
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
    pending_rx: Arc<Mutex<VecDeque<Vec<u8>>>>,
    acked_features: Arc<RwLock<u64>>,
    mem: Arc<RwLock<VnetMemory>>,
    mac: [u8; 6],
    /// Pre-cloned exit event pairs ready for the framework to consume.
    /// exit_event() pops from this — it cannot fail at call time because
    /// all cloning is done up front in the constructor.
    framework_exit_events: Arc<Mutex<Vec<(EventConsumer, EventNotifier)>>>,
    #[cfg(feature = "debug-trace")]
    tx_debug_budget: Arc<AtomicUsize>,
    #[cfg(feature = "debug-trace")]
    rx_debug_budget: Arc<AtomicUsize>,
    #[cfg(feature = "debug-trace")]
    tx_event_debug_budget: Arc<AtomicUsize>,
}

impl VnetVhostBackend {
    #[cfg(feature = "debug-trace")]
    fn debug_budget_tick(counter: &AtomicUsize) -> bool {
        counter
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |remaining| {
                remaining.checked_sub(1)
            })
            .is_ok()
    }

    fn new(
        config: &VnetConfig,
        tx_sender: mpsc::Sender<Vec<u8>>,
        rx_receiver: mpsc::Receiver<Vec<u8>>,
        rx_eventfd: EventFd,
        tx_wake_eventfd: EventFd,
        mem: VnetMemory,
    ) -> Result<(Self, EventNotifier), VnetError> {
        // Pre-create the exit event and clone all needed copies up front.
        // This ensures exit_event() cannot fail at call time.
        let (exit_consumer, exit_notifier) =
            new_event_consumer_and_notifier(EventFlag::empty())
                .map_err(|e| VnetError::BackendInit(format!("exit event: {}", e)))?;

        // Clone for the framework (returned by exit_event).
        let framework_consumer = exit_consumer
            .try_clone()
            .map_err(|e| VnetError::BackendInit(format!("exit consumer clone: {}", e)))?;
        let framework_notifier = exit_notifier
            .try_clone()
            .map_err(|e| VnetError::BackendInit(format!("exit notifier clone: {}", e)))?;

        // Clone for VnetHandle (used during shutdown to signal workers).
        let shutdown_notifier = exit_notifier
            .try_clone()
            .map_err(|e| VnetError::BackendInit(format!("shutdown notifier clone: {}", e)))?;

        let backend = Self {
            rx_eventfd: Arc::new(rx_eventfd),
            tx_wake_eventfd: Arc::new(tx_wake_eventfd),
            tx_sender: Arc::new(Mutex::new(tx_sender)),
            rx_receiver: Arc::new(Mutex::new(rx_receiver)),
            pending_rx: Arc::new(Mutex::new(VecDeque::new())),
            acked_features: Arc::new(RwLock::new(0)),
            mem: Arc::new(RwLock::new(mem)),
            mac: config.mac,
            framework_exit_events: Arc::new(Mutex::new(vec![
                (framework_consumer, framework_notifier),
            ])),
            #[cfg(feature = "debug-trace")]
            tx_debug_budget: Arc::new(AtomicUsize::new(8)),
            #[cfg(feature = "debug-trace")]
            rx_debug_budget: Arc::new(AtomicUsize::new(8)),
            #[cfg(feature = "debug-trace")]
            tx_event_debug_budget: Arc::new(AtomicUsize::new(16)),
        };

        Ok((backend, shutdown_notifier))
    }

    /// Process tx virtqueue: read frames from guest, send to slirp thread.
    /// Returns Err on fatal backend state (lock poison) to signal the
    /// framework to shut down this worker — prevents half-alive degradation.
    fn process_tx(&self, vring: &VringRwLock) -> std::io::Result<()> {
        let tx_lock = self.tx_sender.lock().map_err(|e| {
            tracing::error!("tx lock poisoned, backend fatally degraded: {}", e);
            std::io::Error::new(std::io::ErrorKind::Other, "tx lock poisoned")
        })?;

        let mem_atomic = self.mem.read().map_err(|e| {
            tracing::error!("memory lock poisoned, backend fatally degraded: {}", e);
            std::io::Error::new(std::io::ErrorKind::Other, "memory lock poisoned")
        })?;
        let mem_guard = mem_atomic.memory();
        let mem = mem_guard.deref();

        let mut state = vring.get_mut();
        #[cfg(feature = "debug-trace")]
        let call_present = state.get_call().is_some();
        let queue = state.get_queue_mut();
        let mut sent = false;
        let mut returned_any = false;
        #[cfg(feature = "debug-trace")]
        let mut saw_desc = false;

        #[cfg(feature = "debug-trace")]
        if Self::debug_budget_tick(&self.tx_event_debug_budget) {
            eprintln!("motlie-vnet tx-event: kick received call_present={call_present}");
        }

        while let Some(mut desc_chain) = queue.pop_descriptor_chain(mem) {
            #[cfg(feature = "debug-trace")]
            {
                saw_desc = true;
            }
            let desc_idx = desc_chain.head_index();
            let mut frame = Vec::new();
            let mut first = true;
            let mut read_failed = false;

            'tx_desc: while let Some(desc) = desc_chain.next() {
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
                    match mem.read(&mut buf, data_addr) {
                        Ok(_) => frame.extend_from_slice(&buf),
                        Err(e) => {
                            #[cfg(feature = "debug-trace")]
                            eprintln!(
                                "motlie-vnet tx-read-fail: desc={} first=true addr={} len={} err={}",
                                desc_idx,
                                data_addr.raw_value(),
                                data_len,
                                e
                            );
                            tracing::warn!("tx: guest memory read failed at {}: {}", data_addr.raw_value(), e);
                            read_failed = true;
                            break 'tx_desc;
                        }
                    }
                } else {
                    let mut buf = vec![0u8; len];
                    match mem.read(&mut buf, addr) {
                        Ok(_) => frame.extend_from_slice(&buf),
                        Err(e) => {
                            #[cfg(feature = "debug-trace")]
                            eprintln!(
                                "motlie-vnet tx-read-fail: desc={} first=false addr={} len={} err={}",
                                desc_idx,
                                addr.raw_value(),
                                len,
                                e
                            );
                            tracing::warn!("tx: guest memory read failed at {}: {}", addr.raw_value(), e);
                            read_failed = true;
                            break 'tx_desc;
                        }
                    }
                }
            }

            // Only send complete frames — drop truncated ones on read failure.
            if !read_failed && !frame.is_empty() {
                #[cfg(feature = "debug-trace")]
                if Self::debug_budget_tick(&self.tx_debug_budget) {
                    let ethertype = if frame.len() >= 14 {
                        u16::from_be_bytes([frame[12], frame[13]])
                    } else {
                        0
                    };
                    eprintln!(
                        "motlie-vnet tx: len={} ethertype=0x{:04x} dst={:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} src={:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
                        frame.len(),
                        ethertype,
                        frame.get(0).copied().unwrap_or(0),
                        frame.get(1).copied().unwrap_or(0),
                        frame.get(2).copied().unwrap_or(0),
                        frame.get(3).copied().unwrap_or(0),
                        frame.get(4).copied().unwrap_or(0),
                        frame.get(5).copied().unwrap_or(0),
                        frame.get(6).copied().unwrap_or(0),
                        frame.get(7).copied().unwrap_or(0),
                        frame.get(8).copied().unwrap_or(0),
                        frame.get(9).copied().unwrap_or(0),
                        frame.get(10).copied().unwrap_or(0),
                        frame.get(11).copied().unwrap_or(0),
                    );
                }
                if let Err(e) = tx_lock.send(frame) {
                    tracing::warn!("tx channel send failed: {}", e);
                }
                sent = true;
            }

            if let Err(e) = queue.add_used(mem, desc_idx, 0) {
                #[cfg(feature = "debug-trace")]
                eprintln!("motlie-vnet tx-add-used-fail: desc={} err={}", desc_idx, e);
                tracing::warn!("tx: add_used failed for desc {}: {}", desc_idx, e);
            } else {
                returned_any = true;
            }
        }

        #[cfg(feature = "debug-trace")]
        if Self::debug_budget_tick(&self.tx_event_debug_budget) {
            eprintln!(
                "motlie-vnet tx-event: saw_desc={} returned_any={} sent_any={} call_present={}",
                saw_desc,
                returned_any,
                sent,
                call_present
            );
        }

        if returned_any {
            if let Err(e) = state.signal_used_queue() {
                #[cfg(feature = "debug-trace")]
                eprintln!("motlie-vnet tx-signal-fail: err={}", e);
                tracing::warn!("tx: signal_used_queue failed: {}", e);
            } else {
                #[cfg(feature = "debug-trace")]
                if Self::debug_budget_tick(&self.tx_event_debug_budget) {
                    eprintln!("motlie-vnet tx-signal-ok: call_present={call_present}");
                }
            }
        }
        if sent {
            let _ = self.tx_wake_eventfd.write(1);
        }
        Ok(())
    }

    /// Process rx: drain frames from slirp, inject into rx virtqueue.
    fn process_rx(&self, vring: &VringRwLock) -> std::io::Result<()> {
        let _ = self.rx_eventfd.read();

        let rx_lock = self.rx_receiver.lock().map_err(|e| {
            tracing::error!("rx lock poisoned, backend fatally degraded: {}", e);
            std::io::Error::new(std::io::ErrorKind::Other, "rx lock poisoned")
        })?;
        let mut pending_rx = self.pending_rx.lock().map_err(|e| {
            tracing::error!("pending rx lock poisoned, backend fatally degraded: {}", e);
            std::io::Error::new(std::io::ErrorKind::Other, "pending rx lock poisoned")
        })?;

        while let Ok(frame) = rx_lock.try_recv() {
            pending_rx.push_back(frame);
        }

        let mem_atomic = self.mem.read().map_err(|e| {
            tracing::error!("memory lock poisoned, backend fatally degraded: {}", e);
            std::io::Error::new(std::io::ErrorKind::Other, "memory lock poisoned")
        })?;
        let mem_guard = mem_atomic.memory();
        let mem = mem_guard.deref();

        let mut state = vring.get_mut();
        let queue = state.get_queue_mut();
        let mut injected = false;

        while let Some(frame) = pending_rx.pop_front() {
            if let Some(mut desc_chain) = queue.pop_descriptor_chain(mem) {
                #[cfg(feature = "debug-trace")]
                if Self::debug_budget_tick(&self.rx_debug_budget) {
                    let ethertype = if frame.len() >= 14 {
                        u16::from_be_bytes([frame[12], frame[13]])
                    } else {
                        0
                    };
                    eprintln!(
                        "motlie-vnet rx: len={} ethertype=0x{:04x} dst={:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x} src={:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
                        frame.len(),
                        ethertype,
                        frame.get(0).copied().unwrap_or(0),
                        frame.get(1).copied().unwrap_or(0),
                        frame.get(2).copied().unwrap_or(0),
                        frame.get(3).copied().unwrap_or(0),
                        frame.get(4).copied().unwrap_or(0),
                        frame.get(5).copied().unwrap_or(0),
                        frame.get(6).copied().unwrap_or(0),
                        frame.get(7).copied().unwrap_or(0),
                        frame.get(8).copied().unwrap_or(0),
                        frame.get(9).copied().unwrap_or(0),
                        frame.get(10).copied().unwrap_or(0),
                        frame.get(11).copied().unwrap_or(0),
                    );
                }
                let desc_idx = desc_chain.head_index();
                let mut offset = 0usize;
                let mut write_failed = false;

                'desc: while let Some(desc) = desc_chain.next() {
                    if !desc.is_write_only() {
                        continue;
                    }

                    let addr = desc.addr();
                    let avail = desc.len() as usize;

                    if offset < VIRTIO_NET_HDR_SIZE {
                        let hdr_remaining = VIRTIO_NET_HDR_SIZE - offset;
                        let hdr_write = hdr_remaining.min(avail);
                        let zeros = vec![0u8; hdr_write];
                        if let Err(e) = mem.write(&zeros, addr) {
                            tracing::warn!("rx: header write failed at {}: {}", addr.raw_value(), e);
                            write_failed = true;
                            break 'desc;
                        }
                        offset += hdr_write;

                        if hdr_write < avail && offset >= VIRTIO_NET_HDR_SIZE {
                            let frame_offset = offset - VIRTIO_NET_HDR_SIZE;
                            let data_write = (avail - hdr_write)
                                .min(frame.len().saturating_sub(frame_offset));
                            if data_write > 0 {
                                let write_addr = addr.unchecked_add(hdr_write as u64);
                                if let Err(e) = mem.write(
                                    &frame[frame_offset..frame_offset + data_write],
                                    write_addr,
                                ) {
                                    tracing::warn!("rx: payload write failed at {}: {}", write_addr.raw_value(), e);
                                    write_failed = true;
                                    break 'desc;
                                }
                                offset += data_write;
                            }
                        }
                    } else {
                        let frame_offset = offset - VIRTIO_NET_HDR_SIZE;
                        if frame_offset < frame.len() {
                            let data_write = avail.min(frame.len() - frame_offset);
                            if let Err(e) = mem.write(
                                &frame[frame_offset..frame_offset + data_write],
                                addr,
                            ) {
                                tracing::warn!("rx: payload write failed at {}: {}", addr.raw_value(), e);
                                write_failed = true;
                                break 'desc;
                            }
                            offset += data_write;
                        }
                    }
                }

                // Publish only the bytes actually written. On write failure,
                // offset reflects the last successful write position — the
                // guest sees a truncated or empty frame rather than stale data.
                // Publish only the bytes actually written. On write failure,
                // offset reflects the last successful write position.
                let used_len = if write_failed && offset == 0 { 0 } else { offset as u32 };
                if let Err(e) = queue.add_used(mem, desc_idx, used_len) {
                    tracing::warn!("rx: add_used failed for desc {}: {}", desc_idx, e);
                }
                // Always notify — even zero-length returns must be signaled
                // so the guest recycles the descriptor and avoids rx stalls.
                injected = true;
            } else {
                tracing::debug!(
                    "rx vring empty/full, deferring frame ({} bytes, {} pending)",
                    frame.len(),
                    pending_rx.len() + 1
                );
                pending_rx.push_front(frame);
                break;
            }
        }

        if injected {
            if let Err(e) = state.signal_used_queue() {
                tracing::warn!("rx: signal_used_queue failed: {}", e);
            }
        }
        Ok(())
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
        1 << VIRTIO_NET_F_MAC
            | 1 << VIRTIO_F_VERSION_1
            | VhostUserVirtioFeatures::PROTOCOL_FEATURES.bits()
    }

    fn acked_features(&self, features: u64) {
        match self.acked_features.write() {
            Ok(mut guard) => *guard = features,
            Err(e) => tracing::error!("features lock poisoned, cannot ack: {}", e),
        }
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
        atomic_mem: VnetMemory,
    ) -> std::io::Result<()> {
        match self.mem.write() {
            Ok(mut guard) => {
                *guard = atomic_mem;
                Ok(())
            }
            Err(e) => {
                tracing::error!("memory lock poisoned during update_memory: {}", e);
                Err(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "memory lock poisoned",
                ))
            }
        }
    }

    fn exit_event(&self, _thread_index: usize) -> Option<(EventConsumer, EventNotifier)> {
        // Pop a pre-cloned pair. All cloning was done in the constructor
        // where errors propagate. Mutex poison is recovered (not silenced)
        // so the exit event is always delivered if one was prepared.
        let mut guard = self
            .framework_exit_events
            .lock()
            .unwrap_or_else(|poisoned| {
                tracing::error!("exit event mutex poisoned — recovering to deliver exit event");
                poisoned.into_inner()
            });
        let pair = guard.pop();
        if pair.is_none() {
            tracing::error!("exit_event called but no pre-cloned pair available — worker thread will not have an exit signal");
        }
        pair
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
                self.process_rx(&vrings[RX_QUEUE as usize])?;
            }
            TX_QUEUE => {
                self.process_tx(&vrings[TX_QUEUE as usize])?;
            }
            SLIRP_RX_TOKEN => {
                self.process_rx(&vrings[RX_QUEUE as usize])?;
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

        // Constructor returns (backend, shutdown_notifier). All exit-event
        // cloning is done inside — no silent clone failures at call time.
        let (backend, exit_notifier) = VnetVhostBackend::new(
            &self.config,
            tx_sender,
            rx_receiver,
            rx_eventfd,
            tx_wake_eventfd,
            mem.clone(),
        )?;

        let mut daemon = VhostUserDaemon::new(
            "motlie-vnet".to_string(),
            backend,
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

        // Helper: shut down the slirp thread and return its panic if one occurred.
        // Used in both pre-readiness and post-readiness failure paths.
        let stop_slirp = |shutdown: &Arc<AtomicBool>,
                          slirp_handle: JoinHandle<()>|
         -> Option<VnetError> {
            shutdown.store(true, Ordering::Relaxed);
            match slirp_handle.join() {
                Ok(()) => None,
                Err(panic_payload) => Some(VnetError::BackendInit(format!(
                    "slirp thread panicked: {:?}",
                    panic_payload
                ))),
            }
        };

        // Wait for slirp readiness. Prefer slirp panic over readiness error
        // if both occurred.
        match ready_rx.recv_timeout(Duration::from_secs(5)) {
            Ok(SlirpStartup::Ready) => {}
            Ok(SlirpStartup::Failed(e)) => {
                // Thread already exited (it sent Failed before returning) —
                // blocking join is safe and bounded.
                if let Some(panic_err) = stop_slirp(&shutdown, slirp_handle) {
                    return Err(panic_err);
                }
                return Err(e);
            }
            Err(e) => {
                // Timeout: the slirp thread may be hung in SlirpInstance::new()
                // or a libslirp C call. A blocking join would defeat the timeout.
                // Set the shutdown flag and spawn a detached cleanup thread
                // so start() returns within the documented timeout bound.
                shutdown.store(true, Ordering::Relaxed);
                let cleanup_failed = match thread::Builder::new()
                    .name("motlie-vnet-slirp-cleanup".into())
                    .spawn(move || {
                        match slirp_handle.join() {
                            Ok(()) => {}
                            Err(p) => tracing::error!("slirp thread panicked during timeout cleanup: {:?}", p),
                        }
                    }) {
                    Ok(_) => None,
                    Err(spawn_err) => {
                        tracing::error!(
                            "failed to spawn slirp cleanup thread — slirp thread leaked: {}",
                            spawn_err
                        );
                        Some(spawn_err)
                    }
                };
                let mut msg = format!("slirp readiness timeout: {}", e);
                if let Some(ce) = cleanup_failed {
                    msg.push_str(&format!(" (cleanup thread spawn also failed: {})", ce));
                }
                return Err(VnetError::BackendInit(msg));
            }
        }

        // Bind socket on calling thread — socket is live when start() returns.
        // Listener::new(path, true) does its own unlink-before-bind. If it fails,
        // we did NOT create the socket, so we must NOT unlink it (another process
        // may own that path).
        let mut listener = match Listener::new(&socket_path, true) {
            Ok(l) => l,
            Err(e) => {
                // Prefer slirp panic over bind error if both occurred.
                if let Some(slirp_err) = stop_slirp(&shutdown, slirp_handle) {
                    return Err(slirp_err);
                }
                return Err(VnetError::SocketBind(std::io::Error::other(format!("{}", e))));
            }
        };
        tracing::info!("vnet socket bound: {}", socket_path.display());

        // From this point on, WE own the socket file and must clean it up on failure.
        let daemon_handle: JoinHandle<Result<(), VnetError>> = match thread::Builder::new()
            .name("motlie-vnet-daemon".into())
            .spawn(move || {
                daemon.start(&mut listener).map_err(|e| {
                    VnetError::BackendInit(format!("daemon accept: {}", e))
                })?;
                match daemon.wait() {
                    Ok(()) => Ok(()),
                    Err(e) => {
                        if is_benign_vhost_error(&e) {
                            tracing::debug!("daemon exited on disconnect: {}", e);
                            Ok(())
                        } else {
                            Err(VnetError::BackendInit(format!("daemon: {}", e)))
                        }
                    }
                }
            }) {
            Ok(handle) => handle,
            Err(e) => {
                let primary_err = if let Some(slirp_err) = stop_slirp(&shutdown, slirp_handle) {
                    slirp_err
                } else {
                    VnetError::BackendInit(format!("daemon thread: {}", e))
                };
                if let Err(ce) = std::fs::remove_file(&socket_path) {
                    tracing::warn!("socket cleanup failed during daemon spawn recovery: {}", ce);
                }
                return Err(primary_err);
            }
        };

        tracing::info!("vnet backend started: {}", socket_path.display());

        Ok(VnetHandle {
            slirp_thread: Some(slirp_handle),
            daemon_thread: Some(daemon_handle),
            shutdown,
            socket_path,
            exit_notifier,
        })
    }
}

/// Handle to a running vnet backend. Drop to shut down (best-effort),
/// or call `shutdown()` for deterministic cleanup.
pub struct VnetHandle {
    slirp_thread: Option<JoinHandle<()>>,
    daemon_thread: Option<JoinHandle<Result<(), VnetError>>>,
    shutdown: Arc<AtomicBool>,
    socket_path: PathBuf,
    exit_notifier: EventNotifier,
}

impl VnetHandle {
    pub fn is_alive(&self) -> bool {
        let slirp_alive = self
            .slirp_thread
            .as_ref()
            .map(|h| !h.is_finished())
            .unwrap_or(false);
        let daemon_alive = self
            .daemon_thread
            .as_ref()
            .map(|h| !h.is_finished())
            .unwrap_or(false);
        slirp_alive && daemon_alive && self.socket_path.exists()
    }

    /// Shut down deterministically. Signals all threads to exit, joins them,
    /// and removes the socket file. Propagates daemon-thread errors.
    pub fn shutdown(&mut self) -> Result<(), VnetError> {
        self.shutdown.store(true, Ordering::Relaxed);

        // Signal the daemon worker's exit event to unblock epoll_wait.
        let _ = self.exit_notifier.notify();

        // Dummy-connect to unblock accept() if the daemon is waiting.
        let _ = std::os::unix::net::UnixStream::connect(&self.socket_path);

        // Join slirp thread — propagate panics.
        if let Some(handle) = self.slirp_thread.take() {
            if let Err(panic_payload) = handle.join() {
                // Still attempt daemon cleanup before returning the error.
                if let Some(dh) = self.daemon_thread.take() {
                    let _ = dh.join();
                }
                if let Err(e) = std::fs::remove_file(&self.socket_path) {
                    tracing::warn!("socket cleanup failed during slirp panic recovery: {}", e);
                }
                return Err(VnetError::BackendInit(format!(
                    "slirp thread panicked: {:?}", panic_payload
                )));
            }
        }

        // Join daemon thread — propagate non-benign errors and panics.
        if let Some(handle) = self.daemon_thread.take() {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    if let Err(ce) = std::fs::remove_file(&self.socket_path) {
                        tracing::warn!("socket cleanup failed during daemon error recovery: {}", ce);
                    }
                    return Err(e);
                }
                Err(panic_payload) => {
                    if let Err(ce) = std::fs::remove_file(&self.socket_path) {
                        tracing::warn!("socket cleanup failed during daemon panic recovery: {}", ce);
                    }
                    return Err(VnetError::BackendInit(format!(
                        "daemon thread panicked: {:?}", panic_payload
                    )));
                }
            }
        }

        // Socket cleanup — propagate failure.
        if self.socket_path.exists() {
            std::fs::remove_file(&self.socket_path).map_err(VnetError::SocketCleanup)?;
        }
        tracing::info!("vnet backend shut down: {}", self.socket_path.display());
        Ok(())
    }
}

impl Drop for VnetHandle {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::VnetConfig;
    use std::os::fd::FromRawFd;
    use std::os::unix::io::IntoRawFd;
    use virtio_queue::mock::MockSplitQueue;
    use virtio_queue::desc::split::Descriptor;
    use virtio_queue::desc::RawDescriptor;
    use vm_memory::GuestAddress;

    fn init_logging() {
        let _ = tracing_subscriber::fmt().with_test_writer().try_init();
    }

    fn create_test_memory() -> VnetMemory {
        GuestMemoryAtomic::new(
            GuestMemoryMmap::<()>::from_ranges(&[(GuestAddress(0), 0x20000)]).unwrap(),
        )
    }

    fn create_vring<M: vm_memory::GuestMemory>(
        mem: VnetMemory,
        queue: &MockSplitQueue<'_, M>,
    ) -> VringRwLock {
        let vring = VringRwLock::new(mem, QUEUE_SIZE as u16).unwrap();
        vring.set_queue_size(QUEUE_SIZE as u16);
        vring
            .set_queue_info(
                queue.desc_table_addr().raw_value(),
                queue.avail_addr().raw_value(),
                queue.used_addr().raw_value(),
            )
            .unwrap();
        vring.set_queue_ready(true);
        vring.set_enabled(true);
        vring
    }

    // -- Config tests --

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

        let _backend = VnetBackend::new(config);
    }

    #[test]
    fn test_config_rejects_empty_path() {
        assert!(VnetConfig::builder().build().is_err());
    }

    #[test]
    fn test_config_rejects_nonexistent_parent() {
        assert!(VnetConfig::builder()
            .socket_path("/nonexistent/dir/test.sock")
            .build()
            .is_err());
    }

    #[test]
    fn test_config_rejects_duplicate_hostfwd_ports() {
        assert!(VnetConfig::builder()
            .socket_path("/tmp/test.sock")
            .host_forward_tcp(2222, 22)
            .host_forward_tcp(2222, 80)
            .build()
            .is_err());
    }

    #[test]
    fn test_config_rejects_mismatched_subnet() {
        assert!(VnetConfig::builder()
            .socket_path("/tmp/test.sock")
            .guest_ipv4(std::net::Ipv4Addr::new(192, 168, 1, 10))
            .host_ipv4(std::net::Ipv4Addr::new(10, 0, 2, 2))
            .build()
            .is_err());
    }

    // -- Lifecycle tests --

    #[test]
    fn test_start_creates_socket_shutdown_removes() {
        init_logging();
        let dir = tempfile::tempdir().unwrap();
        let socket_path = dir.path().join("motlie-vnet.sock");

        let config = VnetConfig::builder()
            .socket_path(&socket_path)
            .build()
            .unwrap();

        let mut handle = VnetBackend::new(config).start().unwrap();
        assert!(socket_path.exists(), "socket should exist after start()");

        handle.shutdown().unwrap();
        assert!(!socket_path.exists(), "socket should be removed after shutdown()");
    }

    #[test]
    fn test_drop_cleans_up() {
        init_logging();
        let dir = tempfile::tempdir().unwrap();
        let socket_path = dir.path().join("motlie-vnet.sock");

        let config = VnetConfig::builder()
            .socket_path(&socket_path)
            .build()
            .unwrap();

        {
            let _handle = VnetBackend::new(config).start().unwrap();
            assert!(socket_path.exists());
        }
        assert!(!socket_path.exists(), "socket should be removed on drop");
    }

    // -- TX datapath test --

    #[test]
    fn test_process_tx_moves_frame_to_channel_and_notifies_guest() {
        let mem = create_test_memory();
        let mem_ref = mem.memory();
        let queue_mem = mem_ref.deref();
        let mock_queue = MockSplitQueue::new(queue_mem, QUEUE_SIZE as u16);

        // Write virtio-net header + payload into guest memory.
        let header_addr = GuestAddress(0x1000);
        let payload_addr = GuestAddress(0x2000);
        let payload = [0xAA, 0xBB, 0xCC, 0xDD];
        queue_mem.write(&[0u8; VIRTIO_NET_HDR_SIZE], header_addr).unwrap();
        queue_mem.write(&payload, payload_addr).unwrap();

        // Build a descriptor chain: header desc → payload desc.
        let descs = [
            RawDescriptor::from(Descriptor::new(
                header_addr.raw_value(),
                VIRTIO_NET_HDR_SIZE as u32,
                0,
                0,
            )),
            RawDescriptor::from(Descriptor::new(
                payload_addr.raw_value(),
                payload.len() as u32,
                0,
                0,
            )),
        ];
        mock_queue.build_desc_chain(&descs).unwrap();

        let vring = create_vring(mem.clone(), &mock_queue);

        // Attach a call notifier to detect guest notification.
        let (call_consumer, call_notifier) =
            new_event_consumer_and_notifier(EventFlag::empty()).unwrap();
        let call_file = unsafe { std::fs::File::from_raw_fd(call_notifier.into_raw_fd()) };
        vring.set_call(Some(call_file));

        // Create backend with channels.
        let (tx_sender, tx_receiver) = mpsc::channel();
        let (_rx_sender, rx_receiver) = mpsc::channel();
        let (backend, _exit_notifier) = VnetVhostBackend::new(
            &VnetConfig::builder().socket_path("/tmp/test.sock").build().unwrap(),
            tx_sender,
            rx_receiver,
            EventFd::new(libc::EFD_NONBLOCK).unwrap(),
            EventFd::new(libc::EFD_NONBLOCK).unwrap(),
            mem,
        )
        .unwrap();

        backend.process_tx(&vring).unwrap();

        // Verify: payload arrived in the tx channel (header stripped).
        assert_eq!(tx_receiver.try_recv().unwrap(), payload);
        // Verify: guest was notified (used ring signal).
        call_consumer.consume().unwrap();
        // Verify: used ring advanced.
        assert_eq!(mock_queue.used().idx().load(), 1);
    }

    // -- RX datapath test --

    #[test]
    fn test_process_rx_writes_frame_to_guest_and_notifies() {
        let mem = create_test_memory();
        let mem_ref = mem.memory();
        let queue_mem = mem_ref.deref();
        let mock_queue = MockSplitQueue::new(queue_mem, QUEUE_SIZE as u16);

        // Build write-only descriptor chain for rx: header + payload buffer.
        let hdr_addr = GuestAddress(0x3000);
        let payload_addr = GuestAddress(0x4000);
        let descs = [
            RawDescriptor::from(Descriptor::new(
                hdr_addr.raw_value(),
                VIRTIO_NET_HDR_SIZE as u32,
                virtio_bindings::virtio_ring::VRING_DESC_F_WRITE as u16,
                0,
            )),
            RawDescriptor::from(Descriptor::new(
                payload_addr.raw_value(),
                4,
                virtio_bindings::virtio_ring::VRING_DESC_F_WRITE as u16,
                0,
            )),
        ];
        mock_queue.build_desc_chain(&descs).unwrap();

        let vring = create_vring(mem.clone(), &mock_queue);

        let (call_consumer, call_notifier) =
            new_event_consumer_and_notifier(EventFlag::empty()).unwrap();
        let call_file = unsafe { std::fs::File::from_raw_fd(call_notifier.into_raw_fd()) };
        vring.set_call(Some(call_file));

        // Seed the rx channel and eventfd.
        let (rx_sender, rx_receiver) = mpsc::channel();
        let eventfd = EventFd::new(libc::EFD_NONBLOCK).unwrap();
        rx_sender.send(vec![1, 2, 3, 4]).unwrap();
        eventfd.write(1).unwrap();

        let (backend, _exit_notifier) = VnetVhostBackend::new(
            &VnetConfig::builder().socket_path("/tmp/test.sock").build().unwrap(),
            mpsc::channel().0,
            rx_receiver,
            eventfd,
            EventFd::new(libc::EFD_NONBLOCK).unwrap(),
            mem,
        )
        .unwrap();

        backend.process_rx(&vring).unwrap();

        // Verify: guest memory has zeroed header.
        let mut hdr = [0xFF; VIRTIO_NET_HDR_SIZE];
        queue_mem.read(&mut hdr, hdr_addr).unwrap();
        assert_eq!(hdr, [0u8; VIRTIO_NET_HDR_SIZE]);
        // Verify: guest memory has the frame payload.
        let mut payload_out = [0u8; 4];
        queue_mem.read(&mut payload_out, payload_addr).unwrap();
        assert_eq!(payload_out, [1, 2, 3, 4]);
        // Verify: guest was notified.
        call_consumer.consume().unwrap();
        // Verify: used ring advanced.
        assert_eq!(mock_queue.used().idx().load(), 1);
    }

    #[test]
    fn test_process_rx_defers_frame_until_guest_posts_buffers() {
        let mem = create_test_memory();
        let mem_ref = mem.memory();
        let queue_mem = mem_ref.deref();
        let mock_queue = MockSplitQueue::new(queue_mem, QUEUE_SIZE as u16);
        let vring = create_vring(mem.clone(), &mock_queue);

        let (call_consumer, call_notifier) =
            new_event_consumer_and_notifier(EventFlag::empty()).unwrap();
        let call_file = unsafe { std::fs::File::from_raw_fd(call_notifier.into_raw_fd()) };
        vring.set_call(Some(call_file));

        let (rx_sender, rx_receiver) = mpsc::channel();
        let eventfd = EventFd::new(libc::EFD_NONBLOCK).unwrap();
        rx_sender.send(vec![1, 2, 3, 4]).unwrap();
        eventfd.write(1).unwrap();

        let (backend, _exit_notifier) = VnetVhostBackend::new(
            &VnetConfig::builder().socket_path("/tmp/test.sock").build().unwrap(),
            mpsc::channel().0,
            rx_receiver,
            eventfd,
            EventFd::new(libc::EFD_NONBLOCK).unwrap(),
            mem,
        )
        .unwrap();

        // No RX descriptors yet: frame should be retained, not dropped.
        backend.process_rx(&vring).unwrap();
        assert_eq!(backend.pending_rx.lock().unwrap().len(), 1);
        assert_eq!(mock_queue.used().idx().load(), 0);

        // Guest posts buffers later.
        let hdr_addr = GuestAddress(0x3000);
        let payload_addr = GuestAddress(0x4000);
        let descs = [
            RawDescriptor::from(Descriptor::new(
                hdr_addr.raw_value(),
                VIRTIO_NET_HDR_SIZE as u32,
                virtio_bindings::virtio_ring::VRING_DESC_F_WRITE as u16,
                0,
            )),
            RawDescriptor::from(Descriptor::new(
                payload_addr.raw_value(),
                4,
                virtio_bindings::virtio_ring::VRING_DESC_F_WRITE as u16,
                0,
            )),
        ];
        mock_queue.build_desc_chain(&descs).unwrap();

        // RX queue kick should retry pending delivery.
        backend.handle_event(RX_QUEUE, EventSet::IN, &[vring.clone(), vring.clone()], 0).unwrap();

        let mut payload_out = [0u8; 4];
        queue_mem.read(&mut payload_out, payload_addr).unwrap();
        assert_eq!(payload_out, [1, 2, 3, 4]);
        call_consumer.consume().unwrap();
        assert_eq!(backend.pending_rx.lock().unwrap().len(), 0);
        assert_eq!(mock_queue.used().idx().load(), 1);
    }

    // -- TX read-failure regression test --

    /// Verify that a guest memory read failure drops the frame rather than
    /// sending a truncated packet to the slirp channel.
    #[test]
    fn test_process_tx_drops_frame_on_read_failure() {
        // Guest memory is only 0x20000 bytes (128KB).
        let mem = create_test_memory();
        let mem_ref = mem.memory();
        let queue_mem = mem_ref.deref();
        let mock_queue = MockSplitQueue::new(queue_mem, QUEUE_SIZE as u16);

        // First descriptor: valid header at a good address.
        let header_addr = GuestAddress(0x1000);
        queue_mem.write(&[0u8; VIRTIO_NET_HDR_SIZE], header_addr).unwrap();

        // Second descriptor: payload at an address PAST the end of guest memory.
        // mem.read() will fail on this address → frame should be dropped.
        let bad_addr = GuestAddress(0x30000); // beyond 0x20000

        let descs = [
            RawDescriptor::from(Descriptor::new(
                header_addr.raw_value(),
                (VIRTIO_NET_HDR_SIZE + 4) as u32, // header + some payload
                0,
                0,
            )),
            RawDescriptor::from(Descriptor::new(
                bad_addr.raw_value(),
                64,
                0,
                0,
            )),
        ];
        mock_queue.build_desc_chain(&descs).unwrap();

        let vring = create_vring(mem.clone(), &mock_queue);

        let (tx_sender, tx_receiver) = mpsc::channel();
        let (_rx_sender, rx_receiver) = mpsc::channel();
        let (backend, _exit_notifier) = VnetVhostBackend::new(
            &VnetConfig::builder().socket_path("/tmp/test.sock").build().unwrap(),
            tx_sender,
            rx_receiver,
            EventFd::new(libc::EFD_NONBLOCK).unwrap(),
            EventFd::new(libc::EFD_NONBLOCK).unwrap(),
            mem,
        )
        .unwrap();

        backend.process_tx(&vring).unwrap();

        // Frame should NOT have been sent — the read failure on the second
        // descriptor should have dropped the entire frame.
        // Note: the first descriptor includes header + 4 payload bytes in a
        // single buffer. The read of the first descriptor succeeds (valid addr),
        // but the second descriptor's read fails. Since we break on first failure,
        // the partial frame from the first descriptor is accumulated but the
        // read_failed flag prevents it from being sent.
        //
        // Actually, the first descriptor reads successfully (4 payload bytes
        // after header), then the second descriptor at bad_addr fails. The
        // read_failed flag is set and the frame is dropped.
        assert!(
            tx_receiver.try_recv().is_err(),
            "truncated frame should NOT be sent to slirp channel"
        );

        // Descriptor should still be returned to the used ring so the guest
        // doesn't wedge.
        assert_eq!(mock_queue.used().idx().load(), 1);
    }

    #[test]
    fn test_process_tx_read_failure_still_notifies_guest() {
        let mem = create_test_memory();
        let mem_ref = mem.memory();
        let queue_mem = mem_ref.deref();
        let mock_queue = MockSplitQueue::new(queue_mem, QUEUE_SIZE as u16);

        let header_addr = GuestAddress(0x1000);
        queue_mem.write(&[0u8; VIRTIO_NET_HDR_SIZE], header_addr).unwrap();
        let bad_addr = GuestAddress(0x30000);

        let descs = [
            RawDescriptor::from(Descriptor::new(
                header_addr.raw_value(),
                (VIRTIO_NET_HDR_SIZE + 4) as u32,
                0,
                0,
            )),
            RawDescriptor::from(Descriptor::new(
                bad_addr.raw_value(),
                64,
                0,
                0,
            )),
        ];
        mock_queue.build_desc_chain(&descs).unwrap();

        let vring = create_vring(mem.clone(), &mock_queue);

        let (call_consumer, call_notifier) =
            new_event_consumer_and_notifier(EventFlag::empty()).unwrap();
        let call_file = unsafe { std::fs::File::from_raw_fd(call_notifier.into_raw_fd()) };
        vring.set_call(Some(call_file));

        let (tx_sender, tx_receiver) = mpsc::channel();
        let (_rx_sender, rx_receiver) = mpsc::channel();
        let (backend, _exit_notifier) = VnetVhostBackend::new(
            &VnetConfig::builder().socket_path("/tmp/test.sock").build().unwrap(),
            tx_sender,
            rx_receiver,
            EventFd::new(libc::EFD_NONBLOCK).unwrap(),
            EventFd::new(libc::EFD_NONBLOCK).unwrap(),
            mem,
        )
        .unwrap();

        backend.process_tx(&vring).unwrap();

        assert!(tx_receiver.try_recv().is_err(), "truncated frame should be dropped");
        call_consumer.consume().unwrap();
        assert_eq!(mock_queue.used().idx().load(), 1);
    }
}
