# motlie-vnet: Embeddable User-Mode Networking for Virtual Machines

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-04-13 | @codex-vz | Add [`DESIGN_XBACKENDS.md`](./DESIGN_XBACKENDS.md) as the detailed cross-backend evolution plan: preserve the no-host-config-drift/all-userspace/ephemeral-lifetime requirements while splitting `motlie-vnet` into a reusable slirp/policy/flow core plus thin hypervisor adapters |
| 2026-04-06 | @codex | Re-baseline docs around the implemented `v1.2` harness: `motlie-vnet` owns the `v1.2+` example line, the current composed flow lives under `libs/vnet/examples/v1.2`, and the design now distinguishes current implementation from future standalone/demo targets |
| 2026-04-03 | @codex | Address final PR review nits: align `VnetError` variants with PLAN and remove stale `--net-mode` wording |
| 2026-04-03 | @codex | Address review follow-ups: document why long-term SSH ingress moves into a host-side proxy, make short-term dual-NIC route ownership explicit, and align migration/docs with the public `start()` / `VnetHandle` API |
| 2026-04-03 | @codex | Split SSH ingress from outbound egress: short-term migration keeps TAP admin SSH while `motlie-vnet` owns outbound internet, and the long-term target moves ingress into a host-side `russhd` / REPL proxy |
| 2026-04-02 | @claude | Address blocking review: restore SSH ingress as a system requirement, add guest migration plan, crate layering (src/examples/bins), composed acceptance milestone, and zero-host-impact model |
| 2026-04-01 | @chungers | Address PR #125 review: clarify RING_PACKED, DNS build-time, drop vs shutdown, MAC defaults, ICMP fallback, module re-exports, thiserror |
| 2026-04-01 | @claude | Complete API design: builder, start/shutdown, multi-guest, error types, repl_host integration. Resolve all open questions (features, shared memory, thread safety). Expand testing matrix |
| 2026-04-01 | @claude | Expand NFRs: instance model, isolation, socket parametrization, teardown, netns, performance on DGX Spark |
| 2026-04-01 | @claude | Initial DESIGN: problem statement, motivation, architecture, alternatives analysis |

## Problem Statement

Cloud Hypervisor guest VMs need internet access for development workflows
(package installation, git operations, API calls). The current networking
approach requires host-level privileges that conflict with the project's
goal of minimal-privilege VM operation.

### Current state (v1)

The v1 CH harness (`launch-ch.sh`) uses TAP networking:

```bash
cloud-hypervisor --net "tap=,ip=192.168.249.1,mask=255.255.255.0"
```

This requires **`CAP_NET_ADMIN`** on the `cloud-hypervisor` binary, granted via:

```bash
sudo setcap cap_net_admin+ep $(which cloud-hypervisor)
```

This is acceptable for development but introduces several problems:

1. **Security surface** — CAP_NET_ADMIN grants broad network manipulation capabilities
2. **Setup friction** — every new machine needs the setcap step
3. **Container/CI incompatibility** — many CI environments don't allow capability grants
4. **Multi-tenant risk** — shared machines shouldn't grant CAP_NET_ADMIN to user binaries

### Goal

Provide guest internet access with **zero host networking configuration** and
**no elevated privileges** — the host-side binary runs as a regular user with
no capabilities, no sudo, and no kernel module requirements beyond KVM and
vhost-vsock (which the user already has via the `kvm` group).

Cross-backend note:

- the detailed plan for preserving these same requirements while evolving toward
  Vz and future backends lives in
  [`DESIGN_XBACKENDS.md`](./DESIGN_XBACKENDS.md)
- that document keeps the same product constraints explicit:
  - no persistent host network configuration changes
  - all userspace
  - runtime networking state is ephemeral and tied to process lifetime

## Non-Goals

- High-performance networking (DPDK-class throughput)
- IPv6 support in v1 (guest-initiated IPv4 egress only)
- Network policy enforcement (firewall rules, traffic shaping)
- Multi-VM networking (VM-to-VM communication)
- General-purpose inbound port forwarding as a product requirement. v1 only
  needs system SSH/admin ingress, and the long-term ingress path is expected
  to terminate in the host runtime rather than in libslirp host forwarding.

## Harness Ownership and Source of Truth

The example / validation harness lineage now splits by subsystem ownership:

- `motlie-vfs` owns the historical `v1` and `v1.1` harnesses under
  `libs/vfs/examples/`
- `motlie-vnet` owns the `v1.2+` harness line under
  `libs/vnet/examples/v1.2/`

That ownership boundary is about the harness and runbook, not about eliminating
composition between the crates. The current `v1.2` host flow is intentionally
composed:

- `motlie-vnet` owns the split-network launcher, guest image, and validation
  story
- `motlie-vfs` remains the filesystem service used by the `v1.2` host REPL
- `libs/vnet/docs/{DESIGN,PLAN}.md` are the source of truth for `v1.2+`
  architecture and follow-up work

## Background: Alternatives Investigated

### Alternative 1: passt/pasta (separate process)

**passt** (Plug A Simple Socket Transport) provides user-mode networking by
translating L2 Ethernet frames to L4 host sockets. It runs as a separate
process and integrates with QEMU via a Unix domain socket (`-netdev stream`).

**Investigation (2026-03-31):**

We attempted to integrate pasta (passt's namespace variant) with Cloud
Hypervisor. pasta creates a network namespace with a TAP device and
translates traffic between the namespace and host sockets.

**Findings:**

1. **CH doesn't speak passt's protocol.** QEMU has `-netdev stream` for
   passt's Unix socket protocol. CH only supports TAP (`--net tap=`) and
   vhost-user (`--net vhost_user=true,socket=`). There is no equivalent
   stream socket backend.

2. **pasta namespace approach failed.** We tried running CH inside a pasta
   network namespace (`pasta --config-net -- cloud-hypervisor ...`). CH
   creates its own TAP (`vmtap0`) inside the namespace, but this is a
   **separate device** from pasta's TAP. Traffic between them requires
   bridging or routing, which isn't automatically configured.

3. **TAP fd sharing failed.** We tried opening pasta's TAP fd and passing
   it to CH via `--net fd=N`. This failed because pasta owns the TAP —
   the `TUNSETIFF` ioctl to attach to an existing TAP owned by another
   process is rejected.

4. **User namespace restrictions.** Ubuntu 24.04 blocks unprivileged user
   namespaces via AppArmor (`kernel.apparmor_restrict_unprivileged_userns=1`).
   pasta requires user namespaces, so it needs a sysctl override.

**Verdict:** passt/pasta cannot be used with CH without either (a) CH adding
a stream socket network backend, or (b) writing a bridge between pasta's
TAP and CH's TAP inside the namespace. Both are significant work for a
fragile multi-process architecture.

### Alternative 2: slirp4netns + TAP fd

**slirp4netns** creates a network namespace, opens a TAP inside it, and
provides SLIRP-based networking. The TAP fd could theoretically be passed
to CH via `--net fd=N`.

**Issues:**
- Same namespace + TAP ownership problems as pasta
- slirp4netns is a separate process with its own lifecycle
- SLIRP performance is poor (full TCP/IP stack emulation in userspace)

**Verdict:** Same multi-process fragility as passt, with worse performance.

### Alternative 3: Embedded vhost-user-net with libslirp (chosen)

**Approach:** Embed a vhost-user-net backend directly into the host-side
process (`repl_host` or equivalent VMM orchestrator). The backend uses
libslirp for user-mode TCP/IP translation.

**Why this wins:**

| Concern | passt | slirp4netns | Embedded vhost-user |
|---------|-------|-------------|---------------------|
| Process count | 2 (passt + CH) | 2 (slirp4netns + CH) | 1 (unified) |
| CH integration | No stream backend | TAP fd issues | Native vhost-user |
| Lifecycle | Must manage separately | Must manage separately | Dies with host process |
| Privileges | User namespaces | User namespaces | None beyond KVM |
| Performance | Good (L4 translation) | Poor (full stack) | Good (L4 via libslirp) |

## Functional Requirements

- **FR-1:** Guest VM gets a `virtio-net` device via vhost-user protocol
- **FR-2:** Guest obtains IP via DHCP (libslirp built-in DHCP server)
- **FR-3:** Guest can make outbound TCP/UDP connections to the internet
- **FR-4:** Guest DNS queries are forwarded to host resolver
- **FR-5:** No host-level capabilities, sudo, or kernel modules required
  (beyond KVM + vhost-vsock already needed for the VM)
- **FR-6:** The backend is embeddable as a library — callers spawn it as a
  thread, not a separate process
- **FR-7:** The overall host/guest system must preserve host-to-guest SSH
  ingress while `motlie-vnet` replaces TAP / passt for outbound internet
  egress.
  - **Short term migration:** preserve the existing TAP-based admin / SSH
    path from the `vfs/examples/v1.1` lineage while moving outbound guest
    internet to `motlie-vnet`.
  - **Long term target:** move SSH ingress above `motlie-vnet` into the
    composed host runtime (for example a `russhd` / REPL-hosted proxy),
    so `motlie-vnet` remains focused on DHCP, DNS, and outbound egress.
  - libslirp `hostfwd` may still exist as an optional standalone
    debugging / demo capability, but it is not the primary product
    architecture for SSH ingress.

## Non-Functional Requirements

### Embeddability and Error Handling

- **NFR-1:** Library-embeddable — the backend is a Rust struct that callers
  spawn on a thread, not a binary. No `main()`, no global state, no
  `std::process::exit()`. Multiple instances can coexist in one process.
- **NFR-2:** No panics in library code. All errors returned as `Result`.
  `unwrap()` / `expect()` / `assert!()` are forbidden in non-test code.
  libslirp C callbacks that cannot return errors must log and degrade
  gracefully (e.g. drop the frame) rather than panic.
- **NFR-3:** No unsafe code in the glue layer. `unsafe` is acceptable only
  inside libslirp C bindings and vm-memory mappings where the crate
  requires it.

### Instance Model: One Stack Per Guest

- **NFR-4:** Each guest VM gets its own `VnetBackend` instance with its own:
  - libslirp context (separate TCP/IP state, DHCP leases, DNS cache)
  - vhost-user Unix socket (unique path per guest)
  - background thread
  - No shared state between instances — full isolation by construction.

  This means an orchestrator managing N guests spawns N backend threads,
  each bound to a distinct socket. Example:

  ```
  Guest 0: /tmp/motlie-vnet-0.sock  →  VnetBackend thread 0
  Guest 1: /tmp/motlie-vnet-1.sock  →  VnetBackend thread 1
  Guest 2: /tmp/motlie-vnet-2.sock  →  VnetBackend thread 2
  ```

  The per-guest model is chosen over a shared/multiplexed model because:
  - libslirp is single-threaded and not thread-safe — sharing requires
    serialization which defeats the purpose
  - vhost-user is a 1:1 protocol (one frontend ↔ one backend per socket)
  - Isolation: a misbehaving guest cannot affect another guest's network
  - Simplicity: no routing, no NAT between guests, no shared IP space

- **NFR-5:** Socket path is caller-specified via the builder API:
  ```rust
  VnetBackend::builder()
      .socket_path("/tmp/motlie-vnet-0.sock")  // required, no default
      .build()?;
  ```
  The builder validates the path is writable and removes stale sockets
  before binding. Socket cleanup on drop is best-effort (unlink on
  `Drop`, but not guaranteed if the process is killed with SIGKILL).

### Lifecycle and Teardown

- **NFR-6:** Clean shutdown sequence:
  1. Caller drops `VnetBackend` or the `serve()` method returns
  2. vhost-user daemon closes the Unix socket
  3. CH detects backend disconnect → guest sees link-down on virtio-net
  4. libslirp context is dropped → all host sockets (TCP/UDP) are closed
  5. Socket file is unlinked (best-effort)
  6. Thread exits

- **NFR-7:** Abrupt termination (SIGKILL, process crash):
  - Stale socket file may remain — callers must handle this at startup
    (the builder unlinks existing sockets before binding)
  - Host TCP/UDP sockets are closed by the OS (fd cleanup)
  - No persistent state — restart is clean with no recovery needed
  - CH handles backend disappearance: guest sees link-down, no crash

### Network Namespace Requirements

- **NFR-8:** No network namespace required. The backend operates entirely
  in the host's network namespace using unprivileged sockets:
  - TCP connections: `connect()` as regular user
  - UDP: `sendto()` / `recvfrom()` as regular user
  - ICMP: libslirp uses unprivileged ICMP sockets (requires
    `net.ipv4.ping_group_range` to include the user's gid, which is
    the default on modern Linux). If the sysctl is restrictive, `ping`
    from the guest silently fails (ICMP socket creation returns EACCES,
    libslirp drops the packet). TCP/UDP are unaffected.
  - DNS: forwarded to host resolver via UDP socket

  If the orchestrator wants to isolate guests from each other at the
  network level (e.g. prevent guest A from connecting to guest B's
  host-side ports), it can run each backend in a separate network
  namespace. But this is not required and is outside the scope of
  this library.

  Cross-backend note:
  - the same no-persistent-host-config-drift requirement applies to the
    cross-backend split documented in
    [`DESIGN_XBACKENDS.md`](./DESIGN_XBACKENDS.md)

### Performance Characteristics

- **NFR-9:** Performance target: sufficient for interactive development
  workloads (package installation, git operations, API calls). Not
  designed for production data-plane or bulk transfer workloads.

  **Overhead model:**
  - Each Ethernet frame traverses: virtqueue → memcpy → libslirp L2 parse
    → L4 socket syscall. This is ~2 context switches per packet (one for
    the virtqueue kick eventfd, one for the socket syscall).
  - libslirp adds CPU overhead for TCP/IP stack emulation — roughly
    equivalent to QEMU's `-net user` mode, which sustains ~1-5 Gbps
    depending on packet size and host CPU.

  **Expected throughput (dev/test workloads):**

  | Workload | Expected | Notes |
  |----------|----------|-------|
  | `apt update` | Seconds | Small metadata downloads |
  | `git clone` (100MB repo) | ~10-30s | Limited by libslirp TCP window |
  | `curl` download (1GB) | ~100-500 MB/s | CPU-bound in libslirp |
  | ICMP ping RTT | <1ms | Host-local translation |

  **Scaling on DGX Spark:**
  - Each backend thread consumes ~1-5% of one CPU core at idle (epoll wait)
  - Active network I/O: ~10-30% of one core per guest (libslirp processing)
  - Memory: ~2-5 MB per backend instance (libslirp buffers + virtqueue)
  - A DGX Spark with 20 cores and 128GB RAM can comfortably run 10-20
    concurrent guests with active network I/O, or 50+ mostly-idle guests
  - The bottleneck is CPU for libslirp, not memory or fd limits

  **Not at line rate.** Line rate (100 Gbps on DGX Spark) requires
  kernel bypass (DPDK, io_uring, XDP). libslirp is fundamentally
  limited by syscall overhead and single-threaded TCP processing.
  For production networking, the v2 path would use a DPDK-based
  vhost-user backend instead of libslirp.

## High-Level System Design

### Data Flow

```
Guest VM (virtio-net driver)
    │
    ▼
Cloud Hypervisor (vhost-user frontend)
    │  virtio ring shared memory
    ▼
┌──────────────────────────────────┐
│  motlie-vnet backend (Rust)      │
│                                  │
│  ┌────────────┐  ┌────────────┐  │
│  │ tx virtq   │  │ rx virtq   │  │
│  │ guest→host │  │ host→guest │  │
│  └─────┬──────┘  └──────▲─────┘  │
│        │                │        │
│        ▼                │        │
│  ┌─────────────────────────────┐ │
│  │  libslirp                   │ │
│  │  L2 frame ↔ L4 socket      │ │
│  │  DHCP server (10.0.2.2)    │ │
│  │  DNS forwarder              │ │
│  └─────────────────────────────┘ │
│        │                ▲        │
│        ▼                │        │
│  host TCP/UDP sockets (unprivileged)
└──────────────────────────────────┘
```

### Component Architecture

```
libs/vnet/
├── src/                              # Panic-free reusable library (no main(), no process::exit())
│   ├── lib.rs                        # Public API: re-exports VnetConfig, VnetBackend, VnetHandle, VnetError
│   ├── backend.rs                    # VhostUserBackend trait implementation
│   ├── slirp.rs                      # libslirp wrapper: event loop, frame I/O
│   └── virtq.rs                      # Virtqueue ↔ slirp frame bridging
├── examples/                         # Validation harnesses owned by vnet
│   └── v1.2/repl_host.rs             # Current canonical composed v1.2 host flow (depends on motlie-vfs as a dev-dependency)
├── bins/                             # Host-side binaries (layered on top of library)
│   └── (future: composed host runtime combining vfs + vnet)
├── docs/
│   ├── DESIGN.md                     # This document
│   └── PLAN.md                       # Delivery plan
└── Cargo.toml
```

**Layering principle** (following `libs/vfs` pattern):
- **`src/`** — All reusable logic. No panics (`unwrap`/`expect`/`assert!`
  forbidden in non-test code). No global state. Multiple instances coexist.
- **`examples/`** — Validation harnesses layered on top of the library.
  `v1.2` is intentionally a composed `motlie-vnet` + `motlie-vfs` harness, and
  `repl_host_v1_2` is the only current host-side example entry point.
- **`bins/`** — Production host-side binaries. May combine `motlie-vnet` +
  `motlie-vfs` + CLI parsing. Layered above library — no library code depends
  on bins.

### Key Crates

| Crate | Version | Role |
|-------|---------|------|
| `vhost` | 0.12+ | vhost-user protocol (shared memory, fd passing) |
| `vhost-user-backend` | 0.16+ | Framework for building vhost-user backends |
| `virtio-queue` | 0.12+ | Virtqueue descriptor chain processing |
| `virtio-bindings` | 0.2+ | Virtio feature flag constants |
| `libslirp` | 4+ | Rust bindings for libslirp (C library) |

**System dependency:** `libslirp-dev` (apt package providing the C library)

### API Design

#### Public Types

```rust
// libs/vnet/src/lib.rs

use std::net::Ipv4Addr;
use std::path::PathBuf;

/// Configuration for a vhost-user-net backend instance.
/// One instance per guest VM. See NFR-4 for the isolation model.
pub struct VnetConfig {
    /// Path to the vhost-user Unix socket.
    /// CH connects to this via: --net vhost_user=true,socket=<path>
    /// Must be unique per guest. Builder validates path is writable.
    pub socket_path: PathBuf,

    /// Guest-visible IP address. Assigned via DHCP.
    /// Default: 10.0.2.15
    pub guest_ipv4: Ipv4Addr,

    /// Host-side gateway IP inside libslirp's virtual network.
    /// Not visible on any host interface — exists only inside libslirp.
    /// Default: 10.0.2.2
    pub host_ipv4: Ipv4Addr,

    /// Subnet mask for the virtual network.
    /// Default: 255.255.255.0
    pub netmask: Ipv4Addr,

    /// DNS server IP presented to the guest via DHCP.
    /// Default: parsed from host /etc/resolv.conf when `.build()` is called (runtime).
    /// libslirp forwards guest DNS queries to this address on the host.
    pub dns_ipv4: Ipv4Addr,

    /// MAC address for the guest's virtio-net device.
    /// Default: 52:54:00:12:34:56 (QEMU convention).
    /// Safe to reuse across guests since each slirp instance is isolated.
    /// For multi-guest orchestrators that may bridge guests in the future,
    /// generate unique MACs (e.g. randomize the last 3 octets per guest).
    pub mac: [u8; 6],

    /// Optional host-to-guest TCP port forwards for standalone demos and
    /// debugging. These are not the primary composed-runtime ingress path.
    /// The long-term ingress design lives above `motlie-vnet` in the host
    /// runtime / REPL layer.
    ///
    /// Example: forward host 127.0.0.1:2222 → guest 10.0.2.15:22
    pub host_forwards: Vec<PortForward>,
}

/// A single host-to-guest TCP port forward rule.
/// libslirp binds `bind_addr:host_port` on the host and forwards
/// accepted connections to `guest_ipv4:guest_port` inside the
/// virtual network.
pub struct PortForward {
    /// Host-side bind address. Default: 127.0.0.1 (loopback only).
    pub bind_addr: Ipv4Addr,
    /// Host-side port to listen on.
    pub host_port: u16,
    /// Guest-side port to forward to.
    pub guest_port: u16,
}

/// Handle to a running vhost-user-net backend.
/// Returned by VnetBackend::start(). Drop to shut down.
pub struct VnetHandle {
    // join handle + shutdown signal
}

/// The backend. Constructed via builder, started with start().
pub struct VnetBackend {
    config: VnetConfig,
}
```

#### Builder Pattern

```rust
let config = VnetConfig::builder()
    .socket_path("/tmp/motlie-vnet-0.sock")  // required — no default
    .guest_ipv4([10, 0, 2, 15])              // optional, shown with default
    .dns_ipv4([8, 8, 8, 8])                  // override DNS
    .mac([0x52, 0x54, 0x00, 0x12, 0x34, 0x56])
    .host_forward_tcp(2222, 22)              // optional standalone debug forward
    .build()?;
// build() fails if:
//   - socket_path is empty
//   - socket_path parent directory doesn't exist or isn't writable
//   - guest_ipv4 and host_ipv4 are not in the same subnet
//   - host_forward host_port conflicts (same port forwarded twice)
```

#### Starting and Stopping

```rust
// Start the backend. Spawns a background thread.
// Removes stale socket file if it exists, binds new socket,
// and waits for CH to connect.
let handle: VnetHandle = VnetBackend::new(config).start()?;
// start() fails if:
//   - socket bind fails (address in use, permission denied)
//   - libslirp context creation fails

// The backend thread blocks in the vhost-user event loop until:
//   1. CH disconnects (guest shutdown / CH exit)
//   2. handle.shutdown() is called
//   3. handle is dropped

// Explicit shutdown (blocks until thread exits):
handle.shutdown()?;

// Or just drop — triggers shutdown, but does not block on thread exit.
// Prefer shutdown() when deterministic cleanup matters (e.g. ensuring
// the socket file is unlinked before starting a new backend).
drop(handle);
```

#### Multi-Guest Orchestrator Example

```rust
use motlie_vnet::{VnetBackend, VnetConfig};

fn start_networking(guest_id: usize) -> anyhow::Result<VnetHandle> {
    let config = VnetConfig::builder()
        .socket_path(format!("/tmp/motlie-vnet-{guest_id}.sock"))
        .guest_ipv4([10, 0, 2, 15])  // same IP is fine — each slirp is isolated
        .build()?;
    VnetBackend::new(config).start()
}

// In the orchestrator:
let mut handles = Vec::new();
for i in 0..num_guests {
    handles.push(start_networking(i)?);
}

// Each guest's CH is launched with:
//   --memory size=512M,shared=true
//   --net vhost_user=true,socket=/tmp/motlie-vnet-{i}.sock,num_queues=2

// Shutdown all:
for h in handles {
    h.shutdown()?;
}
```

#### Integration with repl_host

```rust
// In libs/vnet/examples/v1.2/repl_host.rs:
let _net_handle = {
    let socket = format!("/tmp/motlie-vnet-{}.sock", tag);
    let config = motlie_vnet::VnetConfig::builder()
        .socket_path(&socket)
        .build()?;
    let handle = motlie_vnet::VnetBackend::new(config).start()?;
    eprintln!("vhost-user-net: {socket}");
    eprintln!("  Launch CH with: --memory size=512M,shared=true");
    eprintln!("  --net vhost_user=true,socket={socket},num_queues=2");
    eprintln!("  Pair with existing TAP admin ingress in the short term,");
    eprintln!("  or with a future host-side russh proxy in the long term.");
    handle  // held until host process exits
};
```

#### Error Handling Contract

```rust
// All public methods return Result. No panics.
// Error types:
pub enum VnetError {
    /// Socket path validation failed (not writable, parent missing)
    SocketPath(String),
    /// Socket bind failed (EADDRINUSE, EACCES)
    SocketBind(std::io::Error),
    /// Best-effort stale socket cleanup failed before bind
    SocketCleanup(std::io::Error),
    /// libslirp context creation failed
    SlirpInit(String),
    /// vhost-user backend initialization failed
    BackendInit(String),
    /// Host DNS resolver discovery/parsing failed
    DnsResolver(String),
    /// Port forward bind failed (EADDRINUSE on host-side listener)
    PortForwardBind { host_port: u16, source: std::io::Error },
}

// Derive with thiserror — anyhow::Error conversion is automatic.
// impl std::error::Error for VnetError { ... }
```

### Threading Model

```
Thread 1: repl_host main (REPL input + vsock accept)
Thread 2: vsock connection handler (FsServer per connection)
Thread 3: vhost-user-net backend (per guest)
           ├── VhostUserDaemon::run() — epoll on:
           │     - vhost-user socket fd (CH protocol messages)
           │     - tx virtqueue kick eventfd (guest sent a frame)
           │     - libslirp timer fds (TCP retransmits, DHCP lease)
           │     - libslirp socket fds (host TCP/UDP connections)
           └── On each epoll iteration:
                 1. Process tx virtqueue → extract frames → slirp.input()
                 2. slirp.pollfds_poll() → processes host socket events
                 3. slirp callback send_packet() → queue rx frames
                 4. Drain rx queue → write to rx virtqueue → signal guest
```

The vhost-user-backend crate provides the epoll loop. We register libslirp's
fds alongside the virtqueue eventfds so both are driven by one `epoll_wait()`.

**Thread ownership:** libslirp is `!Send + !Sync` (C library, single-threaded).
The `Context` is created and used exclusively on the backend thread. It is
never shared or moved across threads.

### CH Integration

CH must be launched with shared memory for vhost-user to work:

```bash
cloud-hypervisor \
    --memory size=512M,shared=true \
    --net vhost_user=true,socket=/tmp/motlie-vnet-0.sock,num_queues=2,mac=12:34:56:78:90:ab \
    --kernel artifacts/Image \
    --disk "path=artifacts/rootfs.squashfs,readonly=on" \
           "path=artifacts/overlay.ext4" \
    --vsock "cid=3,socket=/tmp/motlie-vfs.vsock" \
    --serial tty --console off
```

**`shared=true` is required.** Without it, CH cannot share guest RAM with the
vhost-user backend via fd passing. CH validates this at startup and rejects
vhost-user devices without shared memory.

The backend socket must exist **before** CH starts. CH connects to it during
boot. If the socket doesn't exist, CH fails with a connection error.

**Startup order:**
1. Start repl_host (creates vsock socket + vhost-user-net socket)
2. Start CH (connects to both sockets)
3. Guest boots, gets DHCP IP, has internet

### Guest Experience

The guest sees a standard `virtio-net` device. In the current validated `v1.2`
image:

1. Kernel detects `virtio-net` PCI device (no driver changes needed)
2. The boot-time `motlie-vnet-egress` service brings up the egress NIC with the
   libslirp-compatible defaults used in `v1.2`
   - implemented as a oneshot systemd unit in the guest image
   - brings the egress NIC up, assigns `10.0.2.15/24`, adds default route via
     `10.0.2.2`, writes resolver `10.0.2.3`, and removes any competing default
     route from the TAP admin NIC
3. Guest gets: IP `10.0.2.15`, netmask `255.255.255.0`, gateway `10.0.2.2`,
   DNS `10.0.2.3`
4. Outbound TCP/UDP: `apt update`, `git clone`, and the agent CLI auth flows work
5. ICMP: `ping 8.8.8.8` works when `net.ipv4.ping_group_range` allows
   unprivileged ICMP sockets; otherwise ping silently fails while TCP/UDP
   continue to work

`systemd-networkd` is enabled in the image, but the current `v1.2`
implementation treats the boot-time `motlie-vnet-egress` service as the
authoritative source of egress configuration. A future pure-DHCP path remains
possible if that becomes the preferred steady-state design.

Inbound SSH is intentionally separated from outbound egress:

- **Short term migration:** preserve the existing TAP-based admin / SSH
  path (`ssh alice@192.168.249.2`) while `motlie-vnet` provides outbound
  internet on a separate guest path / NIC.
- **Long term target:** terminate client SSH in the host runtime
  (`russhd` / REPL layer) and proxy it into the guest over a
  vsock-backed PTY/control endpoint, leaving `motlie-vnet` responsible
  for outbound networking only and making guest `openssh-server`
  removable from the steady-state image later.

### Phased SSH Ingress Strategy

`motlie-vnet` owns outbound guest networking. SSH ingress is treated as a
separate system concern so the ingress path can evolve without forcing
egress, DHCP, and DNS to change at the same time:

- **Short term:** keep the current TAP-based admin / SSH path from the v1/v1.1
  lineage and move only outbound internet access to `motlie-vnet`.
- **Long term:** move SSH ingress into a host-side `russhd` / REPL-hosted
  proxy and keep `motlie-vnet` as the reusable outbound networking layer.

The long-term host-side proxy is preferred because it keeps the SSH control
plane with the orchestrator rather than with the guest image:

- Host-controlled authentication and authorization policy
- CA rotation or key distribution without rebuilding guest images
- Session multiplexing and richer REPL-driven workflows above raw TCP forwarding
- Smaller guest attack surface once guest `openssh-server` is no longer required
- A more opinionated, product-owned ingress stack instead of "whatever the
  guest image currently happens to run"

**What doesn't work:**
- `motlie-vnet` does not define the product ingress path by itself
- Guest-to-guest TCP/IP (each guest has its own isolated slirp network)
- Raw sockets / protocols other than TCP/UDP/ICMP
- IPv6 (libslirp supports it, but we disable it in v1 for simplicity)

### Zero-Host-Impact Model

With `vnet`, the **egress backend** requires no host networking configuration
changes. During the short-term migration, an existing TAP admin path may still
exist for SSH ingress; that exception lives above `motlie-vnet` and should be
treated as migration debt rather than part of the egress design.

| Concern | TAP (v1) | vhost-user + libslirp (vnet) |
|---------|----------|------------------------------|
| Kernel capabilities | `CAP_NET_ADMIN` on CH binary | None |
| Host interfaces | TAP device created per VM | No host interfaces created |
| Bridge / routing | Manual or scripted | Not needed |
| Network namespaces | Optional (passt requires them) | Not needed |
| iptables / nftables | NAT rules for internet | Not needed — libslirp uses user sockets |
| Host ports | Guest IP reachable directly | None required for egress; optional debug `hostfwd` or future host proxy may bind localhost listeners |
| Cleanup on crash | Stale TAP device + routes | Stale Unix socket (auto-cleaned by builder) |

All host-side egress I/O is via regular unprivileged sockets:
- Outbound: `connect()` / `sendto()` as regular user
- No `sudo`, no `setcap`, no `sysctl` changes (except default `ping_group_range`)

### Guest Image and Launcher Migration

The migration described below has now landed in the `v1.2` harness under
`libs/vnet/examples/v1.2/`. That implementation currently proves:

- split `--admin-net` / `--egress-net` launcher controls
- TAP admin ingress plus `motlie-vnet` vhost-user egress in one guest
- a reusable guest base image with the baked tooling needed for validation
  (`sudo`, `python3`, `npm`, `bubblewrap`, Codex CLI, Claude Code CLI)
- a dedicated `/agent-state` mount with home-path symlinks for Codex and Claude
  state during the lifetime of the running host REPL

The remaining work in this section is therefore about refining or generalizing
the current `v1.2` methodology, not about proving that the basic composed flow
exists.

The current v1 guest path (`libs/vfs/examples/v1`) uses one TAP-oriented
network design for both admin ingress and outbound internet. Migrating to
`motlie-vnet` requires splitting those concerns so ingress and egress can
evolve independently.

**Ingress / egress combinations:**

| Phase | Ingress path | Egress path | Guest NIC config |
|------|--------------|-------------|------------------|
| Legacy v1 | TAP (`eth0`) | TAP (`eth0`) | Static via kernel `ip=` |
| Short-term migration | TAP admin (`eth0`) | vhost-user / libslirp (`eth1`) | Admin NIC gets only the host-reachable management subnet; egress NIC owns the default route and DNS via DHCP |
| Long-term target | Host `russhd` proxy | vhost-user / libslirp | Egress NIC only; ingress no longer depends on guest-reachable TAP IP |

**Guest image changes (`build-guest.sh`):**

1. **Add `systemd-networkd` DHCP configuration for the egress NIC** —
   the launcher owns which guest interface is designated as egress and
   should stamp a stable MAC for that NIC. The guest image should match on
   that MAC rather than assuming `eth1`, because device enumeration order
   should not be the source of truth. In the short-term dual-NIC migration,
   the admin TAP path remains host-reachable only and must not own the
   default route; the vhost-user egress NIC owns the default route and DNS:
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

   The matching MAC is a launcher/runtime contract, not a hard-coded product
   value. The important constraint is "egress NIC is selected explicitly and
   owns the default route", not "it happens to be named `eth1`".

2. **Enable `systemd-networkd`** in the guest image:
   ```bash
   systemctl enable systemd-networkd
   ```

3. **Add validation packages** to the base image package list:
   ```
   curl, dnsutils   # needed for e2e network validation
   ```

4. **Keep static admin IP modular** — the kernel `ip=` parameter remains
   a launcher concern for the short-term TAP admin path and is never baked
   into the image.

**Launcher changes (`launch-ch.sh`):**

The launcher should stop treating ingress and egress as one knob. Instead it
should gain separate admin / egress controls:

```bash
# --admin-net=none --egress-net=none
#   no networking
#
# --admin-net=tap --egress-net=tap   (legacy all-in-one)
#   TAP handles both SSH ingress and outbound internet
#
# --admin-net=tap --egress-net=vhost-user   (short-term migration)
#   eth0: TAP admin NIC with static ip=
#   egress NIC (matched by launcher-assigned MAC) gets DHCP + default route
#
# long-term target:
#   no guest-reachable TAP ingress path required for SSH;
#   host russh proxy lives above the launcher/runtime
```

`--no-net` remains the backward-compatible "disable both admin and egress"
spelling during migration. The split flags make the new combinations explicit:
`--admin-net=none --egress-net=none` is the old `--no-net`, and callers can
incrementally move only the egress side to `vhost-user`.

**Modularity principle:** The filesystem stack (`motlie-vfs-guest`,
overlayfs, squashfs) is completely independent of networking. The same
base image works across the migration. The only differences are which
ingress / egress knobs the launcher uses and whether a vnet backend is running.

## Resolved Design Decisions

These were listed as open questions in the initial DESIGN. All are now resolved:

### 1. vhost-user Feature Negotiation

**Resolved:** The backend advertises a minimal feature set:

```rust
const BACKEND_FEATURES: u64 =
    1 << VIRTIO_NET_F_GUEST_CSUM    // guest handles checksum
  | 1 << VIRTIO_NET_F_CSUM          // backend handles checksum
  | 1 << VIRTIO_NET_F_MAC           // backend provides MAC
  | 1 << VIRTIO_F_VERSION_1         // virtio 1.0 (required by CH)
  | 1 << VIRTIO_F_RING_PACKED;      // packed virtqueue (optional — negotiated only if CH offers it;
                                     // the vhost-user-backend crate handles split vs packed transparently)
```

We do **not** advertise `VIRTIO_NET_F_MRG_RXBUF` (mergeable rx buffers) in v1
to keep the rx path simple — one frame per descriptor. This limits rx frame
size to the virtqueue buffer size (typically 1514 bytes for standard Ethernet).
If jumbo frames are needed later, `MRG_RXBUF` can be added.

CH negotiates features via the vhost-user protocol. The intersection of CH's
offered features and our advertised features becomes the active set.

### 2. Shared Memory

**Resolved:** CH requires `--memory size=NM,shared=true` for any vhost-user
device. Without `shared=true`, CH rejects the device at startup. This is
documented in CH's [memory docs](https://github.com/cloud-hypervisor/cloud-hypervisor/blob/main/docs/memory.md).

The vhost-user-backend crate handles memory mapping via `VhostUserMemoryRegion`
messages from CH. No hugepages required — regular `shared=true` memory works.

### 3. libslirp Thread Safety

**Resolved:** libslirp's C API is single-threaded. The `libslirp-rs` bindings
do **not** enforce `!Send` on the `Context` type, so we must enforce it ourselves:
the `Context` is created on the backend thread and never moved. The `VnetBackend`
struct is `Send` (can be moved to the thread), but the `Context` is created
inside `start()` after the thread is spawned.

## Components and Testing

| Component | Test approach |
|-----------|--------------|
| `VnetConfig` builder | Unit test: valid configs succeed, invalid fail with specific errors |
| `SlirpHandler` | Unit test: verify `send_packet()` queues frames, `clock_get_ns()` advances |
| `slirp.rs` event loop | Integration test: create context, feed DHCP discover frame, verify offer response |
| `backend.rs` tx path | Unit test: mock virtqueue descriptor with Ethernet frame, verify `slirp.input()` called |
| `backend.rs` rx path | Unit test: slirp produces frame, verify it appears in rx virtqueue |
| `VnetBackend::start/shutdown` | Integration test: start backend, verify socket exists, shutdown, verify socket removed |
| End-to-end (egress) | CH guest boots with vhost-user-net, `apt update` succeeds, DNS resolves |
| End-to-end (short-term composed) | VFS mounts + Internet egress via vnet + SSH ingress via TAP admin path all work in same guest |
| End-to-end (long-term composed) | VFS mounts + Internet egress via vnet + SSH ingress via host proxy all work in same guest |
