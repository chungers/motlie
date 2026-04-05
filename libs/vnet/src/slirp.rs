// libslirp wrapper: event loop, frame I/O, config builder
//
// The libslirp Context is !Send (contains *mut Slirp). It must be created
// and used exclusively on one thread. We use Rc<RefCell<SlirpHandler>> so
// the handler state (rx frame queue, polled fds) is accessible both from
// libslirp callbacks and from our event loop driver.

use libslirp::context::{Context, Handler, PollEvents};
use std::cell::RefCell;
use std::collections::HashMap;
use std::io;
use std::net::Ipv4Addr;
use std::os::unix::io::RawFd;
use std::rc::Rc;
use std::time::Instant;

// ---------------------------------------------------------------------------
// SlirpHandler — implements libslirp::Handler
// ---------------------------------------------------------------------------

/// Timer handle returned to libslirp via the Handler trait. Holds only an ID;
/// the actual callback and expiry live in SlirpHandler's timer registry so
/// `run_once()` can fire expired timers without raw pointer gymnastics.
pub struct SlirpTimer {
    id: u64,
}

/// Internal timer state stored in the handler's registry.
struct TimerEntry {
    callback: Box<dyn FnMut()>,
    expire_ns: i64,
}

/// Handler state shared between the event loop and libslirp callbacks.
pub struct SlirpHandler {
    /// Monotonic clock base for `clock_get_ns`.
    epoch: Instant,
    /// Frames produced by libslirp (host→guest) via `send_packet`.
    rx_queue: Vec<Vec<u8>>,
    /// File descriptors that libslirp wants polled.
    poll_fds: HashMap<RawFd, ()>,
    /// Whether the event loop needs to be woken (e.g. timer fired).
    notified: bool,
    /// Active timer registry. Keyed by timer ID.
    timers: HashMap<u64, TimerEntry>,
    /// Next timer ID to assign.
    next_timer_id: u64,
}

impl SlirpHandler {
    pub fn new() -> Self {
        Self {
            epoch: Instant::now(),
            rx_queue: Vec::new(),
            poll_fds: HashMap::new(),
            notified: false,
            timers: HashMap::new(),
            next_timer_id: 0,
        }
    }

    /// Drain all queued rx frames (host→guest). Caller writes these into the
    /// virtqueue in Phase 2; in Phase 1 tests we inspect them directly.
    pub fn drain_rx_frames(&mut self) -> Vec<Vec<u8>> {
        std::mem::take(&mut self.rx_queue)
    }

    /// Returns the set of fds libslirp currently wants polled.
    pub fn polled_fds(&self) -> impl Iterator<Item = RawFd> + '_ {
        self.poll_fds.keys().copied()
    }

    /// Clear and return the notified flag.
    pub fn take_notified(&mut self) -> bool {
        std::mem::take(&mut self.notified)
    }

}

impl Handler for SlirpHandler {
    type Timer = SlirpTimer;

    fn clock_get_ns(&mut self) -> i64 {
        self.epoch.elapsed().as_nanos() as i64
    }

    fn send_packet(&mut self, buf: &[u8]) -> io::Result<usize> {
        let len = buf.len();
        if len == 0 {
            return Ok(0);
        }
        self.rx_queue.push(buf.to_vec());
        tracing::debug!("send_packet: queued {} byte frame ({} pending)", len, self.rx_queue.len());
        Ok(len)
    }

    fn register_poll_fd(&mut self, fd: RawFd) {
        self.poll_fds.insert(fd, ());
    }

    fn unregister_poll_fd(&mut self, fd: RawFd) {
        self.poll_fds.remove(&fd);
    }

    fn guest_error(&mut self, msg: &str) {
        tracing::warn!("libslirp guest error: {}", msg);
    }

    fn notify(&mut self) {
        self.notified = true;
    }

    fn timer_new(&mut self, func: Box<dyn FnMut()>) -> Box<SlirpTimer> {
        let id = self.next_timer_id;
        self.next_timer_id += 1;
        self.timers.insert(id, TimerEntry {
            callback: func,
            expire_ns: i64::MAX,
        });
        Box::new(SlirpTimer { id })
    }

    fn timer_mod(&mut self, timer: &mut Box<SlirpTimer>, expire_time: i64) {
        if let Some(entry) = self.timers.get_mut(&timer.id) {
            entry.expire_ns = expire_time;
        }
    }

    fn timer_free(&mut self, timer: Box<SlirpTimer>) {
        self.timers.remove(&timer.id);
    }
}

// ---------------------------------------------------------------------------
// SlirpConfig — builder for creating a configured Context
// ---------------------------------------------------------------------------

/// Configuration for creating a libslirp Context.
/// Maps to the public VnetConfig fields relevant to slirp initialization.
pub struct SlirpConfig {
    /// Guest IP assigned via DHCP. Maps to libslirp's `vdhcp_start`.
    /// Default: 10.0.2.15
    pub guest_ipv4: Ipv4Addr,
    /// Gateway IP inside the virtual network. Maps to libslirp's `vhost`.
    /// Default: 10.0.2.2
    pub host_ipv4: Ipv4Addr,
    /// Subnet mask. Used to compute the network base (`vnetwork = host_ipv4 & netmask`).
    /// Default: 255.255.255.0
    pub netmask: Ipv4Addr,
    /// DNS server IP. Maps to libslirp's `vnameserver`.
    /// Default: 10.0.2.3
    pub dns: Ipv4Addr,
}

impl SlirpConfig {
    /// Compute the network base address: `host_ipv4 & netmask`.
    /// This is libslirp's `vnetwork` parameter.
    fn vnetwork(&self) -> Ipv4Addr {
        let host = u32::from(self.host_ipv4);
        let mask = u32::from(self.netmask);
        Ipv4Addr::from(host & mask)
    }
}

impl Default for SlirpConfig {
    fn default() -> Self {
        Self {
            guest_ipv4: Ipv4Addr::new(10, 0, 2, 15),
            host_ipv4: Ipv4Addr::new(10, 0, 2, 2),
            netmask: Ipv4Addr::new(255, 255, 255, 0),
            dns: Ipv4Addr::new(10, 0, 2, 3),
        }
    }
}

/// A running libslirp instance. Wraps Context and the shared handler.
///
/// This type is `!Send` because `Context` contains a raw pointer to the C
/// libslirp state. It must be created and used on a single thread.
pub struct SlirpInstance {
    ctx: Context<Rc<RefCell<SlirpHandler>>>,
    handler: Rc<RefCell<SlirpHandler>>,
}

impl SlirpInstance {
    /// Create a new slirp instance with the given config.
    pub fn new(config: &SlirpConfig) -> Self {
        let handler = Rc::new(RefCell::new(SlirpHandler::new()));
        let vnetwork = config.vnetwork();
        let ctx = Context::new(
            false,                                          // restricted
            true,                                           // ipv4_enabled
            vnetwork,                                       // vnetwork (network base, e.g. 10.0.2.0)
            config.netmask,                                 // vnetmask
            config.host_ipv4,                               // vhost (gateway, e.g. 10.0.2.2)
            false,                                          // ipv6_enabled
            std::net::Ipv6Addr::UNSPECIFIED,                // vprefix_addr6
            0,                                              // vprefix_len
            std::net::Ipv6Addr::UNSPECIFIED,                // vhost6
            None,                                           // vhostname
            None,                                           // tftp_server_name
            None,                                           // tftp_path
            None,                                           // tftp_bootfile
            config.guest_ipv4,                              // vdhcp_start (guest's DHCP-assigned IP)
            config.dns,                                     // vnameserver
            std::net::Ipv6Addr::UNSPECIFIED,                // vnameserver6
            Vec::new(),                                     // vdnssearch
            None,                                           // vdomainname
            handler.clone(),
        );

        tracing::info!(
            "slirp context created: host={}, guest={}, dns={}",
            config.host_ipv4,
            config.guest_ipv4,
            config.dns,
        );

        Self { ctx, handler }
    }

    /// Feed an Ethernet frame from the guest into libslirp (guest→host tx path).
    pub fn input(&self, frame: &[u8]) {
        self.ctx.input(frame);
    }

    /// Run one iteration of the slirp event loop:
    /// 1. Ask slirp which fds need polling (`pollfds_fill`)
    /// 2. `poll()` those fds
    /// 3. Report results back (`pollfds_poll`)
    /// 4. Fire expired timer callbacks (TCP retransmits, DHCP leases, etc.)
    /// 5. Drain and return any rx frames produced by `send_packet` callbacks
    ///
    /// Returns the frames libslirp wants sent to the guest (host→guest).
    pub fn run_once(&self) -> Vec<Vec<u8>> {
        self.run_once_inner(None)
    }

    /// Like `run_once()`, but caps the poll timeout to `max_timeout_ms`.
    /// Used by the slirp thread to ensure it checks for tx frames and
    /// shutdown signals reasonably often.
    pub fn run_once_with_max_timeout(&self, max_timeout_ms: i32) -> Vec<Vec<u8>> {
        self.run_once_inner(Some(max_timeout_ms))
    }

    fn run_once_inner(&self, max_timeout: Option<i32>) -> Vec<Vec<u8>> {
        // Phase 1: collect fds slirp wants polled
        let mut timeout: u32 = u32::MAX;
        let mut poll_entries: Vec<(RawFd, PollEvents)> = Vec::new();

        self.ctx.pollfds_fill(&mut timeout, |fd, events| {
            let idx = poll_entries.len() as i32;
            poll_entries.push((fd, events));
            idx
        });

        // Phase 2: poll with libc::poll
        let mut pollfds: Vec<libc::pollfd> = poll_entries
            .iter()
            .map(|(fd, events)| libc::pollfd {
                fd: *fd,
                events: poll_events_to_poll_flags(events),
                revents: 0,
            })
            .collect();

        let mut timeout_ms = if timeout == u32::MAX { -1 } else { timeout as i32 };
        if let Some(max) = max_timeout {
            if timeout_ms < 0 || timeout_ms > max {
                timeout_ms = max;
            }
        }
        let poll_error = if !pollfds.is_empty() || timeout_ms >= 0 {
            // Safety: pollfds is a valid slice of libc::pollfd structs.
            let ret = unsafe {
                libc::poll(
                    pollfds.as_mut_ptr(),
                    pollfds.len() as libc::nfds_t,
                    timeout_ms,
                )
            };
            if ret < 0 {
                let err = std::io::Error::last_os_error();
                // EINTR is expected (signal interrupted poll) — not an error.
                if err.kind() != std::io::ErrorKind::Interrupted {
                    tracing::warn!("poll() failed: {}", err);
                }
                true
            } else {
                false
            }
        } else {
            false
        };

        // Phase 3: report poll results to slirp
        self.ctx.pollfds_poll(poll_error, |idx| {
            let revents = pollfds[idx as usize].revents;
            poll_flags_to_poll_events(revents)
        });

        // Phase 4: fire expired timers (TCP retransmits, DHCP leases, etc.)
        // This must be done at the SlirpInstance level, not inside a handler
        // borrow, because timer callbacks reenter the handler through
        // libslirp C → timer_mod_handler → Rc<RefCell<SlirpHandler>>::borrow_mut.
        self.fire_expired_timers();

        // Phase 5: drain rx frames queued by send_packet callbacks
        self.handler.borrow_mut().drain_rx_frames()
    }

    /// Fire expired timer callbacks with proper borrow scoping.
    ///
    /// Timer callbacks reenter the handler through the C FFI
    /// (libslirp C → timer_mod_handler → Rc<RefCell<SlirpHandler>>::borrow_mut),
    /// so we must not hold a handler borrow while invoking them.
    ///
    /// Strategy: borrow briefly to extract expired callback closures, drop the
    /// borrow, invoke callbacks (which may safely re-borrow), then re-borrow to
    /// put callbacks back for future re-arming.
    fn fire_expired_timers(&self) {
        // Step 1: extract expired callbacks under a short borrow.
        let to_fire: Vec<(u64, Box<dyn FnMut()>)> = {
            let mut handler = self.handler.borrow_mut();
            let now = handler.clock_get_ns();

            let expired_ids: Vec<u64> = handler
                .timers
                .iter()
                .filter(|(_, e)| e.expire_ns <= now)
                .map(|(id, _)| *id)
                .collect();

            let mut callbacks = Vec::with_capacity(expired_ids.len());
            for id in expired_ids {
                if let Some(entry) = handler.timers.get_mut(&id) {
                    entry.expire_ns = i64::MAX;
                    // Swap out the real callback, leaving a no-op placeholder.
                    let cb = std::mem::replace(&mut entry.callback, Box::new(|| {}));
                    callbacks.push((id, cb));
                }
            }
            callbacks
        }; // handler borrow dropped — callbacks can now safely reenter

        // Step 2: fire callbacks outside any handler borrow.
        let mut fired: Vec<(u64, Box<dyn FnMut()>)> = Vec::with_capacity(to_fire.len());
        for (id, mut cb) in to_fire {
            cb();
            fired.push((id, cb));
        }

        // Step 3: restore callbacks so they can fire again if re-armed.
        {
            let mut handler = self.handler.borrow_mut();
            for (id, cb) in fired {
                if let Some(entry) = handler.timers.get_mut(&id) {
                    entry.callback = cb;
                }
                // If the timer was freed during the callback, it's gone — the
                // callback is simply dropped.
            }
        }
    }

    /// Add a host→guest TCP port forward via libslirp's `slirp_add_hostfwd`.
    /// Returns Ok(()) on success, Err with the host_port on failure.
    ///
    /// This is an optional debug/demo helper — not the primary product ingress path.
    pub fn add_hostfwd_tcp(
        &self,
        host_addr: Ipv4Addr,
        host_port: u16,
        guest_addr: Ipv4Addr,
        guest_port: u16,
    ) -> Result<(), u16> {
        let ret = unsafe {
            libslirp_sys::slirp_add_hostfwd(
                self.ctx.inner.context,
                0, // is_udp = false
                host_addr.into(),
                host_port as i32,
                guest_addr.into(),
                guest_port as i32,
            )
        };
        if ret < 0 {
            tracing::warn!(
                "slirp_add_hostfwd failed: host={}:{} -> guest={}:{}",
                host_addr, host_port, guest_addr, guest_port,
            );
            Err(host_port)
        } else {
            tracing::info!(
                "hostfwd: {}:{} -> {}:{} (tcp)",
                host_addr, host_port, guest_addr, guest_port,
            );
            Ok(())
        }
    }

    /// Access the shared handler state (for tests and Phase 2 integration).
    pub fn handler(&self) -> std::cell::Ref<'_, SlirpHandler> {
        self.handler.borrow()
    }

    /// Get connection info string from libslirp (for debugging).
    pub fn connection_info(&self) -> String {
        self.ctx.connection_info().to_string()
    }
}

impl Drop for SlirpInstance {
    fn drop(&mut self) {
        tracing::info!("slirp context dropped");
    }
}

// ---------------------------------------------------------------------------
// PollEvents <-> libc::poll flag conversion
// ---------------------------------------------------------------------------

fn poll_events_to_poll_flags(events: &PollEvents) -> i16 {
    let mut flags: i16 = 0;
    if events.has_in() {
        flags |= libc::POLLIN as i16;
    }
    if events.has_out() {
        flags |= libc::POLLOUT as i16;
    }
    if events.has_pri() {
        flags |= libc::POLLPRI as i16;
    }
    if events.has_err() {
        flags |= libc::POLLERR as i16;
    }
    if events.has_hup() {
        flags |= libc::POLLHUP as i16;
    }
    flags
}

fn poll_flags_to_poll_events(flags: i16) -> PollEvents {
    let mut events = PollEvents::empty();
    if flags & libc::POLLIN != 0 {
        events |= PollEvents::poll_in();
    }
    if flags & libc::POLLOUT != 0 {
        events |= PollEvents::poll_out();
    }
    if flags & libc::POLLPRI != 0 {
        events |= PollEvents::poll_pri();
    }
    if flags & libc::POLLERR != 0 {
        events |= PollEvents::poll_err();
    }
    if flags & libc::POLLHUP != 0 {
        events |= PollEvents::poll_hup();
    }
    events
}

// ---------------------------------------------------------------------------
// Utility: parse DNS from /etc/resolv.conf
// ---------------------------------------------------------------------------

/// Parse the first `nameserver` entry from /etc/resolv.conf.
/// Returns the default (10.0.2.3 — libslirp's built-in DNS forwarder) if
/// parsing fails or the file is absent.
pub fn parse_host_dns() -> Ipv4Addr {
    let default_dns = Ipv4Addr::new(10, 0, 2, 3);

    let content = match std::fs::read_to_string("/etc/resolv.conf") {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("failed to read /etc/resolv.conf: {}, using default DNS {}", e, default_dns);
            return default_dns;
        }
    };

    for line in content.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("nameserver") {
            let addr_str = rest.trim();
            if let Ok(addr) = addr_str.parse::<Ipv4Addr>() {
                if addr.is_loopback() || addr.is_unspecified() {
                    tracing::warn!(
                        "ignoring loopback/unspecified host DNS {} from /etc/resolv.conf, using default DNS {}",
                        addr,
                        default_dns
                    );
                    continue;
                }
                return addr;
            }
            // Skip IPv6 nameservers
        }
    }

    tracing::warn!("no IPv4 nameserver found in /etc/resolv.conf, using default DNS {}", default_dns);
    default_dns
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn init_logging() {
        let _ = tracing_subscriber::fmt().with_test_writer().try_init();
    }

    #[test]
    fn test_handler_clock_advances() {
        let mut handler = SlirpHandler::new();
        let t0 = handler.clock_get_ns();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let t1 = handler.clock_get_ns();
        assert!(t1 > t0, "clock should advance: t0={}, t1={}", t0, t1);
    }

    #[test]
    fn test_handler_send_packet_queues_frames() {
        let mut handler = SlirpHandler::new();
        let frame = vec![0xFFu8; 64];
        let result = handler.send_packet(&frame);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 64);

        let frames = handler.drain_rx_frames();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].len(), 64);

        // Drain should empty the queue
        let frames = handler.drain_rx_frames();
        assert!(frames.is_empty());
    }

    #[test]
    fn test_handler_poll_fd_tracking() {
        let mut handler = SlirpHandler::new();
        handler.register_poll_fd(42);
        handler.register_poll_fd(99);
        assert_eq!(handler.polled_fds().count(), 2);

        handler.unregister_poll_fd(42);
        assert_eq!(handler.polled_fds().count(), 1);
    }

    #[test]
    fn test_slirp_instance_creates_context() {
        init_logging();
        let config = SlirpConfig::default();
        let instance = SlirpInstance::new(&config);
        // Context creation succeeded if we get here.
        // connection_info should return something (possibly empty for no connections).
        let _info = instance.connection_info();
    }

    /// Feed a DHCP discover frame to slirp, run the event loop, and verify
    /// slirp responds with a DHCP offer (task 1.2.7).
    #[test]
    fn test_dhcp_discover_gets_offer() {
        init_logging();
        let config = SlirpConfig::default();
        let instance = SlirpInstance::new(&config);

        // Construct a minimal DHCP Discover packet:
        // Ethernet header (14) + IP header (20) + UDP header (8) + BOOTP/DHCP (300)
        let mut frame = vec![0u8; 342];

        // -- Ethernet header --
        // dst: broadcast
        frame[0..6].copy_from_slice(&[0xFF; 6]);
        // src: guest MAC (QEMU convention)
        frame[6..12].copy_from_slice(&[0x52, 0x54, 0x00, 0x12, 0x34, 0x56]);
        // ethertype: IPv4
        frame[12] = 0x08;
        frame[13] = 0x00;

        // -- IP header (20 bytes) --
        let ip_start = 14;
        frame[ip_start] = 0x45; // version=4, ihl=5
        // total length = 328 (IP + UDP + DHCP)
        frame[ip_start + 2] = 0x01;
        frame[ip_start + 3] = 0x48;
        frame[ip_start + 8] = 64; // TTL
        frame[ip_start + 9] = 17; // protocol = UDP
        // src: 0.0.0.0
        // dst: 255.255.255.255
        frame[ip_start + 16..ip_start + 20].copy_from_slice(&[255, 255, 255, 255]);

        // IP checksum
        let checksum = ip_checksum(&frame[ip_start..ip_start + 20]);
        frame[ip_start + 10] = (checksum >> 8) as u8;
        frame[ip_start + 11] = (checksum & 0xFF) as u8;

        // -- UDP header (8 bytes) --
        let udp_start = ip_start + 20;
        // src port: 68 (DHCP client)
        frame[udp_start] = 0;
        frame[udp_start + 1] = 68;
        // dst port: 67 (DHCP server)
        frame[udp_start + 2] = 0;
        frame[udp_start + 3] = 67;
        // UDP length = 308
        frame[udp_start + 4] = 0x01;
        frame[udp_start + 5] = 0x34;

        // -- BOOTP/DHCP payload --
        let bootp_start = udp_start + 8;
        frame[bootp_start] = 1;      // op: BOOTREQUEST
        frame[bootp_start + 1] = 1;  // htype: Ethernet
        frame[bootp_start + 2] = 6;  // hlen: 6
        // xid
        frame[bootp_start + 4] = 0xDE;
        frame[bootp_start + 5] = 0xAD;
        frame[bootp_start + 6] = 0xBE;
        frame[bootp_start + 7] = 0xEF;
        // chaddr (client MAC)
        frame[bootp_start + 28..bootp_start + 34]
            .copy_from_slice(&[0x52, 0x54, 0x00, 0x12, 0x34, 0x56]);

        // DHCP magic cookie at offset 236
        let cookie_off = bootp_start + 236;
        frame[cookie_off] = 99;
        frame[cookie_off + 1] = 130;
        frame[cookie_off + 2] = 83;
        frame[cookie_off + 3] = 99;

        // DHCP option 53 (message type) = 1 (Discover)
        frame[cookie_off + 4] = 53;
        frame[cookie_off + 5] = 1;
        frame[cookie_off + 6] = 1;
        // End option
        frame[cookie_off + 7] = 255;

        // Feed the DHCP discover to slirp
        instance.input(&frame);

        // Run the event loop a few times to let slirp process and respond
        let mut all_rx = Vec::new();
        for _ in 0..10 {
            let rx = instance.run_once();
            all_rx.extend(rx);
            if !all_rx.is_empty() {
                break;
            }
        }

        // Slirp should have produced at least one response frame (DHCP offer)
        assert!(
            !all_rx.is_empty(),
            "expected DHCP offer response, got no frames"
        );

        // Verify it's a UDP packet on port 68 (DHCP client)
        let resp = &all_rx[0];
        assert!(resp.len() > 42, "response frame too short: {} bytes", resp.len());
        // Ethertype should be IPv4
        assert_eq!(resp[12], 0x08);
        assert_eq!(resp[13], 0x00);
        // IP protocol should be UDP (17)
        assert_eq!(resp[23], 17);
        // UDP dst port should be 68
        assert_eq!(resp[36], 0);
        assert_eq!(resp[37], 68);
    }

    /// Feed a DNS query to slirp, verify it produces a response or forwards it
    /// (task 1.2.8).
    #[test]
    fn test_dns_query_produces_response() {
        init_logging();
        let config = SlirpConfig::default();
        let instance = SlirpInstance::new(&config);

        // First, we need the guest to have an IP (simulate a minimal ARP +
        // direct UDP). For simplicity, just send a UDP DNS query from the
        // guest's expected IP to the DNS server IP and let slirp forward it.
        let mut frame = vec![0u8; 83]; // Eth(14) + IP(20) + UDP(8) + DNS(41)

        // -- Ethernet --
        frame[0..6].copy_from_slice(&[0x52, 0x54, 0x00, 0x00, 0x00, 0x02]); // dst: gateway MAC
        frame[6..12].copy_from_slice(&[0x52, 0x54, 0x00, 0x12, 0x34, 0x56]); // src: guest MAC
        frame[12] = 0x08;
        frame[13] = 0x00;

        // -- IP --
        let ip = 14;
        frame[ip] = 0x45;
        let ip_total = 69u16; // 20 + 8 + 41
        frame[ip + 2] = (ip_total >> 8) as u8;
        frame[ip + 3] = (ip_total & 0xFF) as u8;
        frame[ip + 8] = 64;  // TTL
        frame[ip + 9] = 17;  // UDP
        // src: 10.0.2.15 (guest)
        frame[ip + 12..ip + 16].copy_from_slice(&[10, 0, 2, 15]);
        // dst: 10.0.2.3 (DNS)
        frame[ip + 16..ip + 20].copy_from_slice(&[10, 0, 2, 3]);
        let cksum = ip_checksum(&frame[ip..ip + 20]);
        frame[ip + 10] = (cksum >> 8) as u8;
        frame[ip + 11] = (cksum & 0xFF) as u8;

        // -- UDP --
        let udp = ip + 20;
        // src port: 12345
        frame[udp] = 0x30;
        frame[udp + 1] = 0x39;
        // dst port: 53 (DNS)
        frame[udp] = (12345 >> 8) as u8;
        frame[udp + 1] = (12345 & 0xFF) as u8;
        frame[udp + 2] = 0;
        frame[udp + 3] = 53;
        let udp_len = 49u16; // 8 + 41
        frame[udp + 4] = (udp_len >> 8) as u8;
        frame[udp + 5] = (udp_len & 0xFF) as u8;

        // -- DNS query for example.com (type A) --
        let dns = udp + 8;
        // Transaction ID
        frame[dns] = 0xAB;
        frame[dns + 1] = 0xCD;
        // Flags: standard query
        frame[dns + 2] = 0x01;
        frame[dns + 3] = 0x00;
        // Questions: 1
        frame[dns + 4] = 0x00;
        frame[dns + 5] = 0x01;
        // Answer/Auth/Additional: 0
        // Query: example.com = 7example3com0
        let q = dns + 12;
        frame[q] = 7;
        frame[q + 1..q + 8].copy_from_slice(b"example");
        frame[q + 8] = 3;
        frame[q + 9..q + 12].copy_from_slice(b"com");
        frame[q + 12] = 0; // root
        // Type A (1)
        frame[q + 13] = 0;
        frame[q + 14] = 1;
        // Class IN (1)
        frame[q + 15] = 0;
        frame[q + 16] = 1;

        instance.input(&frame);

        // Run several iterations — slirp will try to forward the DNS query
        // to the host resolver via a real UDP socket. The response depends on
        // actual network connectivity, so we just verify slirp accepted the
        // frame without crashing and attempted to process it.
        for _ in 0..5 {
            instance.run_once();
        }
        // If we get here without panic, slirp processed the DNS query.
        // A full response test requires network connectivity and is covered
        // by e2e tests in Phase 3.
    }

    /// Feed an ARP who-has request for the slirp gateway and verify we get an
    /// ARP reply back. This isolates the most basic guest→gateway datapath that
    /// the v1.2 demo depends on before any IP/DNS traffic can work.
    #[test]
    fn test_arp_request_for_gateway_gets_reply() {
        init_logging();
        let config = SlirpConfig::default();
        let instance = SlirpInstance::new(&config);

        let guest_mac = [0x12, 0x34, 0x56, 0x78, 0x90, 0xAB];
        let guest_ip = config.guest_ipv4.octets();
        let gateway_ip = config.host_ipv4.octets();

        // Ethernet (14) + ARP (28)
        let mut frame = vec![0u8; 42];

        // Ethernet header
        frame[0..6].copy_from_slice(&[0xFF; 6]); // broadcast
        frame[6..12].copy_from_slice(&guest_mac); // sender
        frame[12] = 0x08;
        frame[13] = 0x06; // ARP

        // ARP payload
        frame[14] = 0x00;
        frame[15] = 0x01; // Ethernet
        frame[16] = 0x08;
        frame[17] = 0x00; // IPv4
        frame[18] = 6; // hlen
        frame[19] = 4; // plen
        frame[20] = 0x00;
        frame[21] = 0x01; // opcode=request
        frame[22..28].copy_from_slice(&guest_mac); // sender MAC
        frame[28..32].copy_from_slice(&guest_ip); // sender IP
        frame[32..38].copy_from_slice(&[0u8; 6]); // target MAC unknown
        frame[38..42].copy_from_slice(&gateway_ip); // target IP

        instance.input(&frame);

        let mut all_rx = Vec::new();
        for _ in 0..10 {
            let rx = instance.run_once();
            all_rx.extend(rx);
            if !all_rx.is_empty() {
                break;
            }
        }

        assert!(
            !all_rx.is_empty(),
            "expected ARP reply for gateway, got no frames"
        );

        let reply = &all_rx[0];
        assert_eq!(&reply[0..6], &guest_mac, "reply should target guest MAC");
        assert_eq!(reply[12], 0x08);
        assert_eq!(reply[13], 0x06, "reply should be ARP");
        assert_eq!(reply[20], 0x00);
        assert_eq!(reply[21], 0x02, "ARP opcode should be reply");
        assert_eq!(&reply[28..32], &gateway_ip, "ARP sender IP should be gateway");
        assert_eq!(&reply[38..42], &guest_ip, "ARP target IP should be guest");
    }

    /// Verify hostfwd binds a host-side port (task 1.2.9).
    #[test]
    fn test_hostfwd_binds_port() {
        init_logging();
        let config = SlirpConfig::default();
        let instance = SlirpInstance::new(&config);

        // Add a port forward: host 127.0.0.1:12222 -> guest 10.0.2.15:22
        let result = instance.add_hostfwd_tcp(
            Ipv4Addr::new(127, 0, 0, 1),
            12222,
            Ipv4Addr::new(10, 0, 2, 15),
            22,
        );
        assert!(result.is_ok(), "hostfwd should succeed");

        // Verify we can connect to the forwarded port
        let conn = std::net::TcpStream::connect_timeout(
            &std::net::SocketAddr::from(([127, 0, 0, 1], 12222)),
            std::time::Duration::from_secs(1),
        );
        assert!(conn.is_ok(), "should be able to connect to hostfwd port 12222");
    }

    #[test]
    fn test_parse_host_dns() {
        // Just verify it returns a valid IPv4 address without panicking.
        let dns = parse_host_dns();
        assert!(!dns.is_unspecified(), "DNS should not be 0.0.0.0");
    }

    /// Regression test: timer callbacks that reenter the handler through
    /// libslirp's C FFI must not cause a double-borrow panic.
    ///
    /// We exercise this by running a DHCP exchange which arms libslirp's
    /// internal timers (DHCP lease, TCP retransmit). If fire_expired_timers()
    /// held a handler borrow while invoking callbacks, the reentrancy through
    /// timer_mod_handler → Rc<RefCell<SlirpHandler>>::borrow_mut() would panic.
    #[test]
    fn test_timer_reentrancy_no_panic() {
        init_logging();
        let config = SlirpConfig::default();
        let instance = SlirpInstance::new(&config);

        // Feed a DHCP discover to arm libslirp's internal timers.
        let mut frame = vec![0u8; 342];
        frame[0..6].copy_from_slice(&[0xFF; 6]);
        frame[6..12].copy_from_slice(&[0x52, 0x54, 0x00, 0x12, 0x34, 0x56]);
        frame[12] = 0x08;
        frame[13] = 0x00;
        let ip = 14;
        frame[ip] = 0x45;
        frame[ip + 2] = 0x01;
        frame[ip + 3] = 0x48;
        frame[ip + 8] = 64;
        frame[ip + 9] = 17;
        frame[ip + 16..ip + 20].copy_from_slice(&[255, 255, 255, 255]);
        let cksum = ip_checksum(&frame[ip..ip + 20]);
        frame[ip + 10] = (cksum >> 8) as u8;
        frame[ip + 11] = (cksum & 0xFF) as u8;
        let udp = ip + 20;
        frame[udp + 1] = 68;
        frame[udp + 3] = 67;
        frame[udp + 4] = 0x01;
        frame[udp + 5] = 0x34;
        let bootp = udp + 8;
        frame[bootp] = 1;
        frame[bootp + 1] = 1;
        frame[bootp + 2] = 6;
        frame[bootp + 28..bootp + 34].copy_from_slice(&[0x52, 0x54, 0x00, 0x12, 0x34, 0x56]);
        let cookie = bootp + 236;
        frame[cookie..cookie + 4].copy_from_slice(&[99, 130, 83, 99]);
        frame[cookie + 4] = 53;
        frame[cookie + 5] = 1;
        frame[cookie + 6] = 1;
        frame[cookie + 7] = 255;

        instance.input(&frame);

        // Run many iterations to exercise timer arming, firing, and re-arming.
        // If the three-step borrow scoping is wrong, this panics with
        // "already borrowed: BorrowMutError".
        for _ in 0..50 {
            instance.run_once();
        }
    }

    /// Helper: compute IP header checksum.
    fn ip_checksum(header: &[u8]) -> u16 {
        let mut sum: u32 = 0;
        for i in (0..header.len()).step_by(2) {
            let word = if i + 1 < header.len() {
                ((header[i] as u32) << 8) | (header[i + 1] as u32)
            } else {
                (header[i] as u32) << 8
            };
            // Skip the checksum field (bytes 10-11)
            if i == 10 {
                continue;
            }
            sum += word;
        }
        while sum >> 16 != 0 {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }
        !sum as u16
    }
}
