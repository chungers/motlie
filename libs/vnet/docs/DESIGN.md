# motlie-vnet: Embeddable User-Mode Networking for Virtual Machines

## Changelog

| Date | Who | Summary |
|------|-----|---------|
| 2026-03-31 | @claude | Complete API design: builder, start/shutdown, multi-guest, error types, repl_host integration. Resolve all open questions (features, shared memory, thread safety). Expand testing matrix |
| 2026-03-31 | @claude | Expand NFRs: instance model, isolation, socket parametrization, teardown, netns, performance on DGX Spark |
| 2026-03-31 | @claude | Initial DESIGN: problem statement, motivation, architecture, alternatives analysis |

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

## Non-Goals

- High-performance networking (DPDK-class throughput)
- Host-to-guest ingress connections (SSH into guest will continue to use vsock)
- IPv6 support in v1 (guest-initiated IPv4 egress only)
- Network policy enforcement (firewall rules, traffic shaping)
- Multi-VM networking (VM-to-VM communication)

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
    the default on modern Linux)
  - DNS: forwarded to host resolver via UDP socket

  If the orchestrator wants to isolate guests from each other at the
  network level (e.g. prevent guest A from connecting to guest B's
  host-side ports), it can run each backend in a separate network
  namespace. But this is not required and is outside the scope of
  this library.

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
├── src/
│   ├── lib.rs              # Public API: VnetBackend::new().serve(socket_path)
│   ├── backend.rs          # VhostUserBackend trait implementation
│   ├── slirp.rs            # libslirp wrapper: event loop, frame I/O
│   └── virtq.rs            # Virtqueue ↔ slirp frame bridging
├── docs/
│   └── DESIGN.md           # This document
└── Cargo.toml
```

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
    /// Default: parsed from host /etc/resolv.conf at build time.
    /// libslirp forwards guest DNS queries to this address on the host.
    pub dns_ipv4: Ipv4Addr,

    /// MAC address for the guest's virtio-net device.
    /// Default: 52:54:00:12:34:56 (QEMU convention)
    pub mac: [u8; 6],
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
    .build()?;
// build() fails if:
//   - socket_path is empty
//   - socket_path parent directory doesn't exist or isn't writable
//   - guest_ipv4 and host_ipv4 are not in the same subnet
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

// Or just drop — triggers shutdown, but does not wait:
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
// In examples/repl_host.rs:
#[cfg(feature = "net-backend")]
let _net_handle = {
    let socket = format!("/tmp/motlie-vnet-{}.sock", tag);
    let config = motlie_vnet::VnetConfig::builder()
        .socket_path(&socket)
        .build()?;
    let handle = motlie_vnet::VnetBackend::new(config).start()?;
    eprintln!("vhost-user-net: {socket}");
    eprintln!("  Launch CH with: --memory size=512M,shared=true");
    eprintln!("  --net vhost_user=true,socket={socket},num_queues=2");
    handle  // held until repl_host exits
};
```

#### Error Handling Contract

```rust
// All public methods return Result. No panics.
// Error types:
pub enum VnetError {
    /// Socket path validation failed (not writable, parent missing)
    InvalidConfig(String),
    /// Socket bind failed (EADDRINUSE, EACCES)
    SocketBind(std::io::Error),
    /// libslirp context creation failed
    SlirpInit(String),
    /// vhost-user daemon error (protocol negotiation, memory mapping)
    VhostUser(String),
    /// Backend thread panicked (should never happen — defensive)
    ThreadPanic,
}

impl From<VnetError> for anyhow::Error { ... }
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

The guest sees a standard `virtio-net` device. On boot:

1. Kernel detects `virtio-net` PCI device (no driver changes needed)
2. systemd-networkd or dhclient requests DHCP from libslirp (gateway 10.0.2.2)
3. Guest gets: IP `10.0.2.15`, netmask `255.255.255.0`, gateway `10.0.2.2`
4. DNS resolver set to host's nameserver (from `/etc/resolv.conf`)
5. Outbound TCP/UDP: `curl`, `apt update`, `git clone` work immediately
6. ICMP: `ping 8.8.8.8` works (libslirp translates to unprivileged ICMP socket)
7. No inbound connections from host — use vsock for SSH, not TCP

**What doesn't work:**
- Inbound connections from the host to the guest via TCP/IP (by design —
  use vsock for management access)
- Guest-to-guest TCP/IP (each guest has its own isolated slirp network)
- Raw sockets / protocols other than TCP/UDP/ICMP
- IPv6 (libslirp supports it, but we disable it in v1 for simplicity)

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
  | 1 << VIRTIO_F_RING_PACKED;      // packed virtqueue (if CH offers)
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
| End-to-end | CH guest boots with vhost-user-net, `curl http://example.com` succeeds, `apt update` works |
