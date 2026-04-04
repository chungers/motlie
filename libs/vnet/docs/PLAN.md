# motlie-vnet Delivery Plan

Derived from [DESIGN.md](./DESIGN.md). Implements an embeddable vhost-user-net
backend with libslirp for rootless guest networking.

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-04 | @claude-tl | Phase 2 implementation: two-thread vhost-user backend, public API (VnetConfig/VnetError/VnetBackend/VnetHandle), vm-memory pinned to 0.17 for vhost crate compat |
| 2026-04-03 | @claude-tl | Phase 1 implementation: update dep versions to latest (libslirp-dev 4.7.0, vhost 0.16, vhost-user-backend 0.22, virtio-queue 0.17, vm-memory 0.17) |
| 2026-04-03 | @codex | Address final PR review nits: align `VnetError` naming with DESIGN and remove stale `--net-mode` wording |
| 2026-04-03 | @codex | Address review follow-ups: expand public API tasks to match DESIGN, add epoll fallback spike, make short-term dual-NIC routing explicit, and document the long-term host SSH proxy path |
| 2026-04-02 | @claude | Address blocking review: SSH ingress tasks, guest migration tasks, crate layering, composed acceptance milestone, fix validation commands |
| 2026-04-01 | @chungers | Address PR #125 review: add fd registration task, logging task, crate validation, fix vnet-vfs coupling, fix stale paths |
| 2026-04-01 | @claude | Initial PLAN: 4 phases from bootstrap to CH integration |

## Status

All phases pending. No implementation yet.

---

## Phase 1: Bootstrap Crate and libslirp Wrapper

Design references: [Component Architecture](./DESIGN.md), [Key Crates](./DESIGN.md)

### 1.1 Crate Setup

- [x] 1.1.1 Create `libs/vnet/Cargo.toml` with the core dependencies.
  ```toml
  [package]
  name = "motlie-vnet"

  [dependencies]
  libslirp = "4"
  anyhow.workspace = true
  log = "0.4"
  ```
- [x] 1.1.2 Add `libs/vnet` to workspace `Cargo.toml` members.
- [x] 1.1.3 Create `src/lib.rs` with module skeleton: `pub mod slirp;`
- [x] 1.1.4 Verify `cargo check -p motlie-vnet` succeeds.
  ```bash
  # Requires: sudo apt install libslirp-dev
  cargo check -p motlie-vnet
  ```
- [x] 1.1.5 Validate crate versions on crates.io and workspace compatibility.
  Validated 2026-04-03: `libslirp 4.3.2` (Rust crate), `libslirp-dev 4.7.0`
  (system apt), `vhost 0.16.0`, `vhost-user-backend 0.22.0`,
  `virtio-queue 0.17.0`, `vm-memory 0.18.0`. Also requires `libglib2.0-dev`
  as a transitive system dependency of `libslirp-dev`.
- [x] 1.1.6 Run workspace `cargo check` to verify no breakage.

### 1.2 libslirp Wrapper (`src/slirp.rs`)

Design references: [Data Flow](./DESIGN.md), [Open Questions: libslirp thread safety](./DESIGN.md)

- [x] 1.2.1 Implement the `libslirp::Handler` trait on a `SlirpHandler` struct.
  Required methods:
  - `clock_get_ns() → i64` — return `Instant::now()` elapsed nanoseconds
  - `timer_new(cb) → Box<Timer>` — create timer with callback
  - `timer_mod(timer, expire_ns)` — schedule timer
  - `timer_free(timer)` — release timer
  - `send_packet(buf) → io::Result<usize>` — queue frame for guest rx
  - `guest_error(msg)` — log error
  - `register_poll_fd(fd)` / `unregister_poll_fd(fd)` — track polled fds
  - `notify()` — wake event loop
- [x] 1.2.2 Implement frame output: `SlirpHandler` holds a `Vec<Vec<u8>>` for
  queued rx frames. `send_packet()` pushes to it. Caller drains after
  `pollfds_poll()`.
- [x] 1.2.3 Implement the slirp event loop:
  ```rust
  pub fn run_once(ctxt: &mut Context<SlirpHandler>) → Vec<Vec<u8>> {
      let mut fds = Vec::new();
      let timeout = ctxt.pollfds_fill(&mut fds);
      // poll(fds, timeout)
      ctxt.pollfds_poll(&fds);
      ctxt.handler().drain_rx_frames()
  }
  ```
- [x] 1.2.4 Wrap `Context::new()` with a builder for config:
  ```rust
  pub struct SlirpConfig {
      pub guest_ipv4: Ipv4Addr,   // default: 10.0.2.15
      pub host_ipv4: Ipv4Addr,    // default: 10.0.2.2
      pub netmask: Ipv4Addr,      // default: 255.255.255.0
      pub dns: Ipv4Addr,          // default: from host /etc/resolv.conf (runtime)
      pub host_forwards: Vec<PortForward>,  // host→guest TCP port forwards
  }
  ```
- [x] 1.2.5 Verify libslirp is single-threaded safe before the wrapper is used by
  the backend: confirm the Rust bindings enforce `!Send` or document the
  pinning requirement.
- [x] 1.2.6 Implement `host_forward_tcp()` via libslirp's `slirp_add_hostfwd()`.
  Design ref: [FR-7](./DESIGN.md). Treat this as an optional standalone
  debug / demo helper, not as the primary composed runtime ingress path.
  Each `PortForward` entry calls
  `slirp_add_hostfwd(context, is_udp=false, host_addr, host_port, guest_addr, guest_port)`.
  libslirp binds a host-side TCP listener and forwards accepted connections
  to the guest IP inside the virtual network.
- [x] 1.2.7 Add test: create slirp context, feed a DHCP discover frame,
  verify slirp responds with a DHCP offer.
- [x] 1.2.8 Add test: feed a DNS query frame, verify slirp produces a
  response (forwarded to host resolver).
- [x] 1.2.9 Add test: configure `host_forward_tcp(12222, 22)`, verify host
  can connect to `127.0.0.1:12222` and the connection is accepted by libslirp.
  (Full SSH validation requires a guest — see Phase 3.)
- [x] 1.2.10 Set up `log` crate integration for the slirp wrapper.
  - `guest_error()` callback → `log::warn!`
  - Frame drops in `send_packet()` (e.g. queue full) → `log::debug!`
  - Context creation / teardown → `log::info!`
  This satisfies DESIGN NFR-2 (no panics — log and degrade gracefully).

---

## Phase 2: vhost-user-net Backend

Design references: [Data Flow](./DESIGN.md), [Threading Model](./DESIGN.md), [FR-1](./DESIGN.md)

### 2.1 Dependencies

- [x] 2.1.1 Add vhost-user dependencies to `Cargo.toml`:
  ```toml
  vhost = { version = "0.16", features = ["vhost-user-backend"] }
  vhost-user-backend = "0.22"
  virtio-queue = "0.17"
  virtio-bindings = "0.2"
  vm-memory = "0.18"
  ```
- [x] 2.1.2 Verify `cargo check -p motlie-vnet` with all deps.

### 2.2 Backend Implementation (`src/backend.rs`)

- [x] 2.2.1 Implement `VhostUserBackend` trait on `VnetBackend` struct.
  Key trait methods:
  - `num_queues() → usize` — return 2 (rx + tx)
  - `max_queue_size() → usize` — return 256
  - `features() → u64` — negotiate virtio-net features
  - `handle_event(device_event, evset, vrings, thread_backend)` —
    process virtqueue kicks
  - `set_backend_req_fd(backend_req)` — store backend request channel
- [x] 2.2.2 Implement virtio-net feature negotiation:
  ```rust
  // Minimal feature set for libslirp compatibility
  1 << VIRTIO_NET_F_GUEST_CSUM
  | 1 << VIRTIO_NET_F_CSUM
  | 1 << VIRTIO_NET_F_MAC
  | 1 << VIRTIO_F_VERSION_1
  | 1 << VIRTIO_F_RING_PACKED  // optional — negotiated only if CH offers
  ```
  Verify against CH's required features (ref: [Open Questions](./DESIGN.md)).
- [x] 2.2.3 Implement tx path (guest → host):
  - Read descriptor chain from tx virtqueue
  - Extract Ethernet frame bytes
  - Call `slirp_context.input(&frame)`
  - Mark descriptor as used
- [x] 2.2.4 Implement rx path (host → guest):
  - After `slirp.pollfds_poll()`, drain rx frames from `SlirpHandler`
  - For each frame, write to rx virtqueue descriptor chain
  - Signal guest via eventfd
- [x] 2.2.5 Register libslirp fds in the vhost-user epoll loop.
  libslirp's `register_poll_fd()` / `unregister_poll_fd()` callbacks add/remove
  host TCP/UDP socket fds. These must be registered with the `VhostUserDaemon`'s
  epoll alongside the virtqueue kick eventfds, so a single `epoll_wait()` drives
  both virtqueue processing and slirp's network I/O. This is the bridge between
  the vhost-user event loop and the slirp event loop from Phase 1.
  - First do a spike against `vhost-user-backend` to confirm whether the daemon
    can expose or register non-virtqueue fds cleanly.
  - If that integration is awkward or unsupported, fall back to a dedicated
    slirp thread that polls libslirp fds separately and hands rx work to the
    backend thread over an internal channel/eventfd. Document the chosen shape
    before implementation continues.
- [x] 2.2.6 Wire the slirp event loop into `handle_event()`:
  - On tx virtqueue kick: process all pending tx descriptors → slirp.input()
  - On timer/poll: run slirp poll cycle → drain rx → inject to rx virtqueue
- [x] 2.2.7 Add test: mock virtqueue with a tx descriptor containing an ARP
  request, verify slirp processes it.
- [x] 2.2.8 Add test: inject an rx frame via slirp, verify it appears in the
  rx virtqueue.

### 2.3 Public API (`src/lib.rs`)

Design references: [API Design](./DESIGN.md), [FR-6](./DESIGN.md)

- [x] 2.3.1 Implement `VnetBackend` builder:
  Design ref: [API Design](./DESIGN.md), [FR-7](./DESIGN.md)
  ```rust
  pub struct VnetBackendBuilder {
      socket_path: PathBuf,
      guest_ipv4: Ipv4Addr,
      host_ipv4: Ipv4Addr,
      host_forwards: Vec<PortForward>,  // host→guest TCP forwards
  }
  ```
  Builder methods include `.host_forward_tcp(host_port, guest_port)` which
  appends a `PortForward { bind_addr: 127.0.0.1, host_port, guest_port }`.
- [x] 2.3.2 Define and implement `PortForward` exactly as in DESIGN, including
  localhost-default binding semantics for the optional hostfwd helper.
- [x] 2.3.3 Implement host `/etc/resolv.conf` parsing for the default DNS
  resolver path, with explicit fallback/error behavior when no usable nameserver
  is present.
- [x] 2.3.4 Define `VnetError` and map builder/start/runtime failures into it
  (`SocketPath`, `SocketBind`, `SocketCleanup`, `SlirpInit`, `BackendInit`,
  `DnsResolver`, `PortForwardBind`).
- [x] 2.3.5 Implement `VnetBackend::start()` as the primary API from DESIGN.
  It should spawn the background thread, bind the socket, and return a
  `VnetHandle` for deterministic teardown.
- [x] 2.3.6 Implement `VnetHandle` with `shutdown()` and `Drop` semantics from
  DESIGN. `shutdown()` is the explicit cleanup path; `Drop` remains best-effort.
- [x] 2.3.7 Keep `VnetBackend::serve()` as the lower-level/blocking primitive:
  - Create `VhostUserDaemon` with the backend
  - Bind Unix socket at `socket_path`
  - Run daemon event loop (blocks until client disconnects)
- [x] 2.3.8 Add test: start backend on a socket, verify socket file is created.
- [x] 2.3.9 Add shutdown test: verify backend exits cleanly when socket is
  closed or `VnetHandle::shutdown()` is called.

---

## Phase 3: Guest Migration, CH Integration, and End-to-End Testing

Design references: [CH Integration](./DESIGN.md), [Guest Experience](./DESIGN.md),
[Guest Image and Launcher Migration](./DESIGN.md), [Zero-Host-Impact Model](./DESIGN.md)

### 3.1 Guest Image Changes

Based on current `libs/vfs/examples/v1/build-guest.sh`. Changes are additive
and do not break existing TAP or no-net modes.

- [ ] 3.1.1 Add `systemd-networkd` DHCP config for the vnet egress NIC to guest image build.
  Short-term migration keeps TAP admin SSH on one NIC and uses DHCP on a
  separate vnet egress NIC. Match the egress NIC by launcher-assigned MAC,
  not by interface name, so the guest does not depend on `eth1` ordering.
  The egress NIC owns the default route and DNS. The TAP admin NIC must stay
  limited to the host-reachable management subnet and must not install a
  competing default route. Drop a network unit file into the squashfs:
  ```ini
  # /etc/systemd/network/20-egress.network
  [Match]
  MACAddress=12:34:56:78:90:ab

  [Network]
  DHCP=ipv4

  [DHCPv4]
  UseDNS=yes
  RouteMetric=100
  ```
  This is a no-op when no egress NIC exists and preserves the TAP admin path.
- [ ] 3.1.2 Enable `systemd-networkd` in the guest image:
  ```bash
  chroot "$ROOTFS" systemctl enable systemd-networkd
  ```
- [ ] 3.1.3 Add validation packages to `build-guest.sh` package list.
  Current packages: `openssh-server bash coreutils tmux fuse3 libfuse3-3
  systemd systemd-sysv dbus iproute2`. Add:
  ```
  curl dnsutils
  ```
  These are needed for e2e network validation (3.4) and are small (~5MB).
- [ ] 3.1.4 Rebuild guest image and verify existing TAP mode still works.
  ```bash
  cd libs/vfs/examples/v1
  ./build-guest.sh
  ./launch-ch.sh --admin-net=tap --egress-net=tap
  # Inside guest: ip addr show eth0 should show 192.168.249.2 (static)
  ```

### 3.2 Launcher Migration

- [ ] 3.2.1 Split launcher controls into ingress vs egress instead of one `--net-mode`.
  Design ref: [Guest Image and Launcher Migration](./DESIGN.md).
  ```bash
  # --admin-net=none --egress-net=none
  # --admin-net=tap  --egress-net=tap         (legacy)
  # --admin-net=tap  --egress-net=vhost-user  (short-term migration)
  ```
  Default: `none` (preserves current `--no-net` behavior).
- [ ] 3.2.2 Keep `--no-net` as a backward-compatible alias for
  `--admin-net=none --egress-net=none`, document the migration path, and warn
  in help text once the split flags exist.
- [ ] 3.2.3 In `vhost-user` egress mode, add `shared=true` to `--memory` arg.
  Per [CH docs](https://github.com/cloud-hypervisor/cloud-hypervisor/blob/main/docs/memory.md),
  vhost-user devices require shared memory for guest RAM access.
- [ ] 3.2.4 In `vhost-user` egress mode, accept `--vnet-socket` path (default:
  `/tmp/motlie-vnet-0.sock`) and attach the egress NIC without disturbing
  the TAP admin NIC.
- [ ] 3.2.5 In short-term dual-NIC mode, make route ownership explicit:
  the admin TAP NIC must remain host-reachable only, while the vnet egress NIC
  owns the default route. Prefer configuring this by omitting the admin default
  gateway rather than relying on device-order-dependent route metrics.
- [ ] 3.2.6 Verify CH starts in each mode:
  ```bash
  ./launch-ch.sh --admin-net=none --egress-net=none
  ./launch-ch.sh --admin-net=tap --egress-net=tap
  ./launch-ch.sh --admin-net=tap --egress-net=vhost-user --vnet-socket=/tmp/motlie-vnet-0.sock
  ```

### 3.3 Crate Layering and Standalone Example

Design ref: [Component Architecture](./DESIGN.md)

- [ ] 3.3.1 Create `libs/vnet/examples/demo_host.rs` — standalone vnet demo.
  Starts the vnet backend, prints egress / CH launch instructions, waits for
  Ctrl-C. No vfs dependency and no requirement to own the SSH ingress story.
  ```rust
  // libs/vnet/examples/demo_host.rs
  fn main() -> anyhow::Result<()> {
      let config = motlie_vnet::VnetConfig::builder()
          .socket_path("/tmp/motlie-vnet-0.sock")
          .build()?;
      let handle = motlie_vnet::VnetBackend::new(config).start()?;
      eprintln!("vnet backend running.");
      eprintln!("Launch CH: ./launch-ch.sh --admin-net=none --egress-net=vhost-user");
      // wait for Ctrl-C
      handle.shutdown()
  }
  ```
- [ ] 3.3.2 Add `[[example]]` entry to `libs/vnet/Cargo.toml`.
- [ ] 3.3.3 Verify: `cargo run -p motlie-vnet --example demo_host` starts
  the backend, socket file is created, CH can connect.

### 3.4 End-to-End Validation (Standalone vnet)

All validation uses commands available in the updated base image
(3.1.3: `curl`, `dnsutils` added; `iproute2` already present).

- [ ] 3.4.1 Boot CH guest with vhost-user-net, verify DHCP:
  ```bash
  # Inside guest:
  ip addr
  ip -4 route show default
  # The launcher-assigned egress NIC should show 10.0.2.15/24 (libslirp DHCP)
  # and own the default route. In short-term dual-NIC mode the TAP admin NIC
  # remains only for the host-side management subnet.
  ```
- [ ] 3.4.2 Verify outbound TCP:
  ```bash
  curl -s -o /dev/null -w "%{http_code}" http://example.com
  # Should print: 200
  ```
- [ ] 3.4.3 Verify DNS resolution:
  ```bash
  host google.com
  # Should resolve to IP addresses
  ```
- [ ] 3.4.4 Verify `apt update` works inside the guest.
- [ ] 3.4.5 Verify zero host impact: no `CAP_NET_ADMIN` on CH binary,
  no TAP device created, no host routes modified. `ip link` on host
  shows no new interfaces.
- [ ] 3.4.6 Verify clean shutdown: stop vnet backend, confirm CH handles
  backend disconnect gracefully (guest sees link-down, no crash).

### 3.5 Composed Acceptance: VFS + SSH Ingress + Internet Egress

Design ref: [FR-7](./DESIGN.md). This is the short-term migration acceptance
target — proving that virtual filesystem, inbound SSH, and outbound Internet
all work in the same guest VM flow even while ingress and egress use different
designs. This phase still depends on the legacy TAP admin path, so it retains
the existing `CAP_NET_ADMIN` / host-networking prerequisites until the
long-term ingress proxy replaces guest-reachable TAP SSH.

- [ ] 3.5.1 Add `motlie-vnet` as a **dev-dependency** of `motlie-vfs`.
  `repl_host` is an `[[example]]`, so dev-dependencies suffice. This avoids
  coupling the vfs library to vnet at the crate level.
  ```toml
  # libs/vfs/Cargo.toml
  [dev-dependencies]
  motlie-vnet = { path = "../vnet" }
  ```
- [ ] 3.5.2 In `repl_host.rs`, add `--egress-net=vhost-user` CLI support that
  spawns vnet backend before entering REPL, while preserving the existing TAP
  admin / SSH path:
  ```rust
  if args.egress_net == "vhost-user" {
      let socket = format!("/tmp/motlie-vnet-{}.sock", tag);
      let config = motlie_vnet::VnetConfig::builder()
          .socket_path(&socket)
          .build()?;
      let handle = motlie_vnet::VnetBackend::new(config).start()?;
      eprintln!("vhost-user-net: {socket}");
      // TAP admin ingress remains unchanged in the short term
      // hold handle until repl_host exits
  }
  ```
- [ ] 3.5.3 Full composed flow test:
  ```bash
  # Terminal 1: start host with VFS + vnet egress
  cat setup-alice.sh.vfs | cargo run --example repl_host -- \
      --tag alice-home --egress-net=vhost-user

  # Terminal 2: launch CH with TAP admin ingress + vhost-user egress
  ./launch-ch.sh --admin-net=tap --egress-net=vhost-user

  # Terminal 3: verify all three capabilities
  # 1. SSH ingress over existing TAP/admin path
  ssh alice@192.168.249.2        # → interactive shell

  # 2. VFS mounts (inside guest SSH session)
  ls /home/alice/.ssh/            # → authorized_keys (from overlay)
  cat /home/alice/.env            # → ANTHROPIC_API_KEY (from overlay)

  # 3. Internet egress (inside guest SSH session)
  curl -s http://example.com      # → HTML response
  apt update                      # → package list refresh
  ```
- [ ] 3.5.4 Document the composed flow in `libs/vfs/examples/v1/README.md`
  as the recommended migration/development setup.

### 3.6 Long-Term Ingress: Host SSH Proxy

- [ ] 3.6.1 Add a DESIGN-level host runtime section showing SSH ingress
  terminating in the host process (`russhd` / REPL layer) instead of in
  libslirp host forwarding.
- [ ] 3.6.2 Define mechanism A explicitly: host `russhd` / REPL proxy
  terminates client SSH, bridges it over vsock to a guest PTY/control endpoint,
  and keeps `motlie-vnet` focused on outbound networking.
- [ ] 3.6.3 Document why this is the target architecture: host-controlled auth,
  CA/key rotation without guest image churn, session multiplexing, and removal
  of guest `openssh-server` from the steady-state attack surface.
- [ ] 3.6.4 Add a guest migration/deprecation plan: once host-proxied SSH is
  validated, stop requiring guest `openssh-server` in the image and remove the
  TAP-admin SSH dependency from the default launch path.
- [ ] 3.6.5 Add a future composed acceptance target:
  `ssh -p 2222 alice@127.0.0.1` reaches the guest via host proxy while
  outbound internet still uses `motlie-vnet`.

---

## Phase 4: Documentation and Cleanup

- [ ] 4.1.1 Update `libs/vfs/examples/v1/README.md` with vhost-user-net instructions
  and the composed VFS + SSH + Internet flow.
- [ ] 4.1.2 Update `libs/vfs/examples/v1/CH-HARNESS.md` prerequisites and launch flow
  for the split `--admin-net` / `--egress-net` launcher flags.
- [ ] 4.1.3 Remove `CAP_NET_ADMIN` / `setcap` references from all docs (TAP mode
  docs should note it as legacy requiring capabilities).
- [ ] 4.1.4 Add `libs/vnet/README.md` with usage examples (standalone and composed).
- [ ] 4.1.5 Remove `libs/vfs/docs/PLAN-v1.md` v1.5 networking section (now lives here).
- [ ] 4.1.6 Update DESIGN.md open questions with resolved answers.

---

## Delivery Order

1. **Phase 1** (1.1 → 1.2) — bootstrap + libslirp wrapper + optional hostfwd helper
2. **Phase 2** (2.1 → 2.2 → 2.3) — vhost-user backend + public API
3. **Phase 3** (3.1 → 3.2 → 3.3 → 3.4 → 3.5 → 3.6) — guest migration, launcher,
   standalone e2e, short-term composed acceptance, long-term ingress design
4. **Phase 4** — docs and cleanup

Phases 1 and 2 can be developed and tested independently of CH.
Phase 3.1–3.2 (guest image + launcher) can be done in parallel with Phases 1–2.
Phase 3.3–3.5 require both the vnet library (Phase 2) and the updated guest
image (Phase 3.1) to be ready.

## Dependencies

| Dependency | Type | Required for |
|------------|------|-------------|
| `libslirp-dev` 4.7.0 + `libglib2.0-dev` | System (apt) | Phase 1+ |
| `vhost` crate | Rust | Phase 2+ |
| `vhost-user-backend` crate | Rust | Phase 2+ |
| `curl`, `dnsutils` | Guest image (apt) | Phase 3.1+ |
| `systemd-networkd` config | Guest image | Phase 3.1+ |
| Updated `launch-ch.sh` | Script | Phase 3.2+ |
| CH with `--memory shared=true` | Runtime | Phase 3.3+ |
| `motlie-vfs` dev-dep on `motlie-vnet` | Rust | Phase 3.5 |
