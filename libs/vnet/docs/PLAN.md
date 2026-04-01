# motlie-vnet Delivery Plan

Derived from [DESIGN.md](./DESIGN.md). Implements an embeddable vhost-user-net
backend with libslirp for rootless guest networking.

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-03-31 | @claude | Initial PLAN: 4 phases from bootstrap to CH integration |

## Status

All phases pending. No implementation yet.

---

## Phase 1: Bootstrap Crate and libslirp Wrapper

Design references: [Component Architecture](./DESIGN.md), [Key Crates](./DESIGN.md)

### 1.1 Crate Setup

- [ ] 1.1.1 Create `libs/vnet/Cargo.toml` with feature-gated dependencies.
  ```toml
  [package]
  name = "motlie-vnet"

  [features]
  default = ["slirp"]
  slirp = ["dep:libslirp"]

  [dependencies]
  libslirp = { version = "4", optional = true }
  anyhow.workspace = true
  log = "0.4"
  ```
- [ ] 1.1.2 Add `libs/vnet` to workspace `Cargo.toml` members.
- [ ] 1.1.3 Create `src/lib.rs` with module skeleton: `pub mod slirp;`
- [ ] 1.1.4 Verify `cargo check -p motlie-vnet` succeeds.
  ```bash
  # Requires: sudo apt install libslirp-dev
  cargo check -p motlie-vnet
  ```
- [ ] 1.1.5 Run workspace `cargo check` to verify no breakage.

### 1.2 libslirp Wrapper (`src/slirp.rs`)

Design references: [Data Flow](./DESIGN.md), [Open Questions: libslirp thread safety](./DESIGN.md)

- [ ] 1.2.1 Implement the `libslirp::Handler` trait on a `SlirpHandler` struct.
  Required methods:
  - `clock_get_ns() → i64` — return `Instant::now()` elapsed nanoseconds
  - `timer_new(cb) → Box<Timer>` — create timer with callback
  - `timer_mod(timer, expire_ns)` — schedule timer
  - `timer_free(timer)` — release timer
  - `send_packet(buf) → io::Result<usize>` — queue frame for guest rx
  - `guest_error(msg)` — log error
  - `register_poll_fd(fd)` / `unregister_poll_fd(fd)` — track polled fds
  - `notify()` — wake event loop
- [ ] 1.2.2 Implement frame output: `SlirpHandler` holds a `Vec<Vec<u8>>` for
  queued rx frames. `send_packet()` pushes to it. Caller drains after
  `pollfds_poll()`.
- [ ] 1.2.3 Implement the slirp event loop:
  ```rust
  pub fn run_once(ctxt: &mut Context<SlirpHandler>) → Vec<Vec<u8>> {
      let mut fds = Vec::new();
      let timeout = ctxt.pollfds_fill(&mut fds);
      // poll(fds, timeout)
      ctxt.pollfds_poll(&fds);
      ctxt.handler().drain_rx_frames()
  }
  ```
- [ ] 1.2.4 Wrap `Context::new_with_opt()` with a builder for config:
  ```rust
  pub struct SlirpConfig {
      pub guest_ipv4: Ipv4Addr,   // default: 10.0.2.15
      pub host_ipv4: Ipv4Addr,    // default: 10.0.2.2
      pub netmask: Ipv4Addr,      // default: 255.255.255.0
      pub dns: Ipv4Addr,          // default: from host /etc/resolv.conf
  }
  ```
- [ ] 1.2.5 Add test: create slirp context, feed a DHCP discover frame,
  verify slirp responds with a DHCP offer.
- [ ] 1.2.6 Add test: feed a DNS query frame, verify slirp produces a
  response (forwarded to host resolver).
- [ ] 1.2.7 Verify libslirp is single-threaded safe: confirm the Rust bindings
  enforce `!Send` or document the pinning requirement.

---

## Phase 2: vhost-user-net Backend

Design references: [Data Flow](./DESIGN.md), [Threading Model](./DESIGN.md), [FR-1](./DESIGN.md)

### 2.1 Dependencies

- [ ] 2.1.1 Add vhost-user dependencies to `Cargo.toml`:
  ```toml
  vhost = { version = "0.12", features = ["vhost-user-backend"] }
  vhost-user-backend = "0.16"
  virtio-queue = "0.12"
  virtio-bindings = "0.2"
  vm-memory = "0.16"
  ```
- [ ] 2.1.2 Verify `cargo check -p motlie-vnet` with all deps.

### 2.2 Backend Implementation (`src/backend.rs`)

- [ ] 2.2.1 Implement `VhostUserBackend` trait on `VnetBackend` struct.
  Key trait methods:
  - `num_queues() → usize` — return 2 (rx + tx)
  - `max_queue_size() → usize` — return 256
  - `features() → u64` — negotiate virtio-net features
  - `handle_event(device_event, evset, vrings, thread_backend)` —
    process virtqueue kicks
  - `set_backend_req_fd(backend_req)` — store backend request channel
- [ ] 2.2.2 Implement virtio-net feature negotiation:
  ```rust
  // Minimal feature set for libslirp compatibility
  1 << VIRTIO_NET_F_GUEST_CSUM
  | 1 << VIRTIO_NET_F_CSUM
  | 1 << VIRTIO_NET_F_MAC
  | 1 << VIRTIO_F_VERSION_1
  ```
  Verify against CH's required features (ref: [Open Questions](./DESIGN.md)).
- [ ] 2.2.3 Implement tx path (guest → host):
  - Read descriptor chain from tx virtqueue
  - Extract Ethernet frame bytes
  - Call `slirp_context.input(&frame)`
  - Mark descriptor as used
- [ ] 2.2.4 Implement rx path (host → guest):
  - After `slirp.pollfds_poll()`, drain rx frames from `SlirpHandler`
  - For each frame, write to rx virtqueue descriptor chain
  - Signal guest via eventfd
- [ ] 2.2.5 Wire the slirp event loop into `handle_event()`:
  - On tx virtqueue kick: process all pending tx descriptors → slirp.input()
  - On timer/poll: run slirp poll cycle → drain rx → inject to rx virtqueue
- [ ] 2.2.6 Add test: mock virtqueue with a tx descriptor containing an ARP
  request, verify slirp processes it.
- [ ] 2.2.7 Add test: inject an rx frame via slirp, verify it appears in the
  rx virtqueue.

### 2.3 Public API (`src/lib.rs`)

Design references: [API Design](./DESIGN.md), [FR-6](./DESIGN.md)

- [ ] 2.3.1 Implement `VnetBackend` builder:
  ```rust
  pub struct VnetBackendBuilder {
      socket_path: PathBuf,
      guest_ipv4: Ipv4Addr,
      host_ipv4: Ipv4Addr,
  }
  ```
- [ ] 2.3.2 Implement `VnetBackend::serve()`:
  - Create `VhostUserDaemon` with the backend
  - Bind Unix socket at `socket_path`
  - Run daemon event loop (blocks until client disconnects)
- [ ] 2.3.3 Add test: start backend on a socket, verify socket file is created.
- [ ] 2.3.4 Add shutdown test: verify backend exits cleanly when socket is closed.

---

## Phase 3: CH Integration and End-to-End Testing

Design references: [CH Integration](./DESIGN.md), [Guest Experience](./DESIGN.md)

### 3.1 Shared Memory Requirement

- [ ] 3.1.1 Verify CH vhost-user requires `--memory shared=true`.
  Per [CH docs](https://github.com/cloud-hypervisor/cloud-hypervisor/blob/main/docs/memory.md),
  vhost-user devices need shared memory for guest RAM access.
  ```bash
  # launch-ch.sh must use:
  --memory size=512M,shared=true
  ```
- [ ] 3.1.2 Update `launch-ch.sh` to add `shared=true` when vhost-user net is used.
- [ ] 3.1.3 Test: verify CH starts with `shared=true` and vhost-user net socket.

### 3.2 Integration with repl_host

- [ ] 3.2.1 Add `motlie-vnet` as a dependency of `motlie-vfs` (optional, feature-gated).
  ```toml
  # libs/vfs/Cargo.toml
  motlie-vnet = { path = "../vnet", optional = true }

  [features]
  net-backend = ["dep:motlie-vnet"]
  ```
- [ ] 3.2.2 In `repl_host.rs`, spawn vnet backend thread before entering REPL:
  ```rust
  #[cfg(feature = "net-backend")]
  {
      let backend = motlie_vnet::VnetBackend::builder()
          .socket_path("/tmp/motlie-vhost-net.sock")
          .build()?;
      std::thread::spawn(move || {
          if let Err(e) = backend.serve() {
              eprintln!("vnet backend error: {e}");
          }
      });
      eprintln!("vhost-user-net backend listening on /tmp/motlie-vhost-net.sock");
  }
  ```
- [ ] 3.2.3 Update `launch-ch.sh` to detect socket and add `--net vhost_user=true,...`.
- [ ] 3.2.4 Update guest image to include DHCP client (already in Debian bookworm).

### 3.3 End-to-End Validation

- [ ] 3.3.1 Boot CH guest with vhost-user-net, verify `eth0` gets IP via DHCP.
  ```bash
  # Inside guest:
  ip addr show eth0
  # Should show: 10.0.2.15/24
  ```
- [ ] 3.3.2 Verify outbound TCP: `curl -s http://example.com`.
- [ ] 3.3.3 Verify DNS resolution: `host google.com`.
- [ ] 3.3.4 Verify `apt update` works inside the guest.
- [ ] 3.3.5 Verify no host capabilities needed: remove `CAP_NET_ADMIN` from CH,
  confirm guest still has internet.
- [ ] 3.3.6 Verify clean shutdown: stop repl_host, confirm CH handles
  backend disconnect gracefully.

---

## Phase 4: Documentation and Cleanup

- [ ] 4.1.1 Update `libs/vfs/examples/v1/README.md` with vhost-user-net instructions.
- [ ] 4.1.2 Update `libs/vfs/examples/v1/CH-HARNESS.md` prerequisites and launch flow.
- [ ] 4.1.3 Remove `CAP_NET_ADMIN` / `setcap` references from all docs.
- [ ] 4.1.4 Add `libs/vnet/README.md` with usage examples.
- [ ] 4.1.5 Remove `libs/vfs/docs/PLAN-v1.md` v1.5 networking section (now lives here).
- [ ] 4.1.6 Update DESIGN.md open questions with resolved answers.

---

## Delivery Order

1. **Phase 1** (1.1 → 1.2) — bootstrap + libslirp wrapper
2. **Phase 2** (2.1 → 2.2 → 2.3) — vhost-user backend
3. **Phase 3** (3.1 → 3.2 → 3.3) — CH integration + e2e
4. **Phase 4** — docs and cleanup

Phases 1 and 2 can be developed and tested independently of CH.
Phase 3 requires a running CH instance for e2e validation.

## Dependencies

| Dependency | Type | Required for |
|------------|------|-------------|
| `libslirp-dev` | System (apt) | Phase 1+ |
| `vhost` crate | Rust | Phase 2+ |
| `vhost-user-backend` crate | Rust | Phase 2+ |
| CH with `--memory shared=true` | Runtime | Phase 3 |
