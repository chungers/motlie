# motlie-vnet: Embeddable User-Mode Networking for Virtual Machines

## Changelog

| Date | Who | Summary |
|------|-----|---------|
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

```rust
use motlie_vnet::VnetBackend;

// In repl_host or VMM orchestrator:
let backend = VnetBackend::builder()
    .socket_path("/tmp/motlie-vhost-net.sock")
    .guest_ipv4("10.0.2.15")
    .host_ipv4("10.0.2.2")
    .build()?;

// Spawn on a background thread — blocks until CH disconnects
let handle = std::thread::spawn(move || backend.serve());

// Launch CH with:
//   --net vhost_user=true,socket=/tmp/motlie-vhost-net.sock,num_queues=2
```

### Threading Model

```
Thread 1: repl_host main (REPL input + vsock accept)
Thread 2: vsock connection handler (FsServer per connection)
Thread 3: vhost-user-net backend ← new
           ├── VhostUserDaemon event loop (virtqueue kicks)
           └── libslirp poll loop (socket I/O + timers)
```

The backend thread runs two interleaved loops:
- **Virtqueue events:** CH kicks a virtqueue → read tx frames → feed to slirp
- **Slirp events:** slirp socket becomes readable → slirp processes → produces
  rx frames → write to rx virtqueue → signal CH

Both loops are driven by `epoll` on a shared fd set.

### CH Integration

```bash
# launch-ch.sh updated:
cloud-hypervisor \
    --net vhost_user=true,socket=/tmp/motlie-vhost-net.sock,num_queues=2,mac=12:34:56:78:90:ab \
    ...
```

CH acts as the vhost-user **frontend** (it owns the virtqueues in shared
memory). Our backend is the **backend** (it processes frames). CH connects
to the Unix socket, negotiates features, and maps shared memory.

### Guest Experience

The guest sees a standard `virtio-net` device. On boot:

1. Kernel detects `virtio-net` PCI device
2. DHCP client requests IP from libslirp's built-in DHCP server
3. Guest gets: IP `10.0.2.15`, gateway `10.0.2.2`, DNS from host `/etc/resolv.conf`
4. Outbound TCP/UDP works immediately (e.g., `apt update`, `curl`, `git clone`)
5. ICMP ping works (libslirp translates to host ICMP sockets)

## Components and Testing

| Component | Test approach |
|-----------|--------------|
| `backend.rs` | Unit test: mock virtqueue, verify VhostUserBackend trait methods |
| `slirp.rs` | Integration test: feed raw Ethernet frames, verify socket creation |
| `virtq.rs` | Unit test: descriptor chain parsing, frame extraction |
| End-to-end | CH guest boots, `curl` reaches the internet |

## Open Questions

1. **vhost-user feature negotiation:** Which virtio-net features does CH
   require vs. which does libslirp support? Need to verify the feature
   intersection (e.g., VIRTIO_NET_F_MRG_RXBUF, VIRTIO_NET_F_CSUM).

2. **Shared memory:** vhost-user requires the frontend to share guest memory
   with the backend via fd passing. CH uses `--memory size=512M` without
   hugepages. Need to verify this works with the vhost-user-backend crate
   or if `shared=true` is required.

3. **libslirp thread safety:** libslirp is single-threaded. Need to confirm
   the Rust bindings enforce this or if we need to pin to one thread.
